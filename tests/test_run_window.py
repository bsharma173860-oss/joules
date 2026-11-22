"""joules run --from / --until: energy of the part of a run between two output lines, e.g. an agent benchmark after a
model server has loaded. Tested end to end on a stand-in laptop GPU that reports power only (no energy counter),
with a script shaped like a typical benchmark runner: a server started in the background and killed by a trap, a
model-loading phase, then the benchmark between two marker lines, each phase at its own power."""
import contextlib
import functools
import json
import shlex
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_read(fd):
    """Read what a pseudo-terminal has (b"" if nothing yet), or None once it is closed."""
    import select
    if not select.select([fd], [], [], 0.1)[0]:
        return b""
    try:
        return os.read(fd, 65536) or None
    except OSError:
        return None

# Power (mW) changes at least half a second before and after every marker line, so the window sees one constant
# power and its exact energy is that power times the window's length, whatever the sampling phase.
RUNNER = textwrap.dedent("""\
    #!/bin/bash
    P="$FAKE_NVML_POWER_FILE"
    ( while true; do echo "server: still serving"; sleep 0.3; done ) &   # the model server, holding our stdout
    SRV=$!
    trap 'kill $SRV 2>/dev/null' EXIT
    echo "starting server"
    echo 25000 > "$P"; sleep 1.5                                        # loading the model: 25 W
    echo 45000 > "$P"; sleep 0.6                                        # the benchmark: 45 W
    echo ">>> running trie (concurrency=4, duration=2)"
    echo "agent benchmark warming up" >&2
    sleep 2
    echo ">>> done. artifacts in /tmp/out"
    sleep 0.6; echo 6000 > "$P"; sleep 0.4                              # back to idle
    exit 3
    """)


@unittest.skipUnless(shutil.which("gcc") and shutil.which("bash") and os.name == "posix",
                     "needs gcc (to build the stand-in NVIDIA library) and bash")
class RunWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="joules-window-")
        cls.laptop = os.path.join(cls.tmp, "laptop", "libnvidia-ml.so.1")
        cls.server = os.path.join(cls.tmp, "server", "libnvidia-ml.so.1")
        for lib, src in ((cls.laptop, "fake_laptop_nvml.c"), (cls.server, "fake_nvml.c")):
            os.makedirs(os.path.dirname(lib))
            subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", lib, str(ROOT / "tests" / "fake_nvml" / src)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.d = tempfile.mkdtemp(dir=self.tmp)
        self.power = os.path.join(self.d, "power_mw")
        Path(self.power).write_text("6000")                              # idle: 6 W
        self.script = os.path.join(self.d, "run_trie.sh")
        Path(self.script).write_text(RUNNER)

    def joules(self, args, script=None, lib=None, extra_env=None, timeout=60):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1",
               "JOULES_TELEMETRY": "0", "JOULES_NVML_LIB": lib or self.laptop, "FAKE_NVML_POWER_FILE": self.power,
               "NO_COLOR": "1", **(extra_env or {})}
        env.pop("JOULES_SIMULATE", None)
        out = os.path.join(self.d, "r.json")
        cmd = [sys.executable, "-m", "joules", "run", "--no-cpu", "--out", out, *args, "--", "bash", script or self.script]
        t0 = time.monotonic()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        took = time.monotonic() - t0
        rep = json.loads(Path(out).read_text()) if os.path.exists(out) else None
        return p, rep, took

    def test_window_on_a_power_only_laptop_gpu(self):
        p, r, _ = self.joules(["--from", r">>> running trie", "--until", r">>> done\.", "--subtract-idle", "1"])
        self.assertEqual(p.returncode, 3, p.stderr)                      # the command's exit code, passed through
        self.assertEqual([x["method"] for x in r["devices"]], ["nvml power sampling"])
        self.assertIn("RTX 3050 Laptop", r["devices"][0]["name"])
        # the window is exactly the benchmark: 45 W for its length
        self.assertTrue(1.9 < r["window_s"] < 2.6, r["window_s"])
        self.assertAlmostEqual(r["window_energy_j"], 45.0 * r["window_s"], delta=1e-6 * r["window_energy_j"])
        self.assertAlmostEqual(r["devices"][0]["idle_w"], 6.0, places=6)
        self.assertAlmostEqual(r["window_above_idle_j"], (45.0 - 6.0) * r["window_s"], delta=1e-6 * r["window_energy_j"])
        # the whole run is still reported, model loading included
        self.assertGreater(r["totals"]["devices_j"], r["window_energy_j"] + 25.0 * 1.4)
        w = r["window"]
        self.assertTrue(w["found_from"] and w["found_until"])
        self.assertEqual(w["from_line"], ">>> running trie (concurrency=4, duration=2)")
        self.assertEqual(w["until_line"], ">>> done. artifacts in /tmp/out")
        # the command's output reached us unchanged, stdout and stderr, background server included
        lines = p.stdout.splitlines()
        for want in ("starting server", ">>> running trie (concurrency=4, duration=2)", ">>> done. artifacts in /tmp/out"):
            self.assertIn(want, lines)
        self.assertLess(lines.index("starting server"), lines.index(">>> done. artifacts in /tmp/out"))
        self.assertIn("server: still serving", lines)
        self.assertIn("agent benchmark warming up", p.stderr)
        self.assertIn("window", p.stderr)                                # the text report shows the window too

    def test_failed_power_reads_do_not_change_the_window(self):
        p, r, _ = self.joules(["--from", r">>> running trie", "--until", r">>> done\."], extra_env={"FAKE_NVML_FAIL_EVERY": "7"})
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertAlmostEqual(r["window_energy_j"], 45.0 * r["window_s"], delta=1e-6 * r["window_energy_j"])

    def test_output_is_live(self):
        """Each line reaches the terminal as it is printed, not when the command ends."""
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1",
               "JOULES_TELEMETRY": "0", "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power}
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "go", "--", "bash", "-c",
                              "echo first; sleep 3; echo go; sleep 0.5"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        t0 = time.monotonic()
        first = q.stdout.readline()
        seen = time.monotonic() - t0
        q.communicate(timeout=30)
        self.assertEqual(first.strip(), b"first")
        self.assertLess(seen, 2.5)                                       # before the command's 3 s sleep ended

    def test_from_never_appears(self):
        p, r, _ = self.joules(["--from", r"no such line", "--until", r">>> done\."])
        self.assertEqual(p.returncode, 3)
        self.assertIsNone(r["window_energy_j"])
        self.assertFalse(r["window"]["found_from"])
        self.assertTrue(any("never appeared" in n and "whole run" in n for n in r["notes"]), r["notes"])
        self.assertIn("never appeared", p.stderr)
        self.assertGreater(r["totals"]["devices_j"], 0)

    def test_until_missing_runs_to_the_end(self):
        p, r, _ = self.joules(["--from", r">>> running trie", "--until", r"no such line"])
        self.assertTrue(r["window"]["found_from"] and not r["window"]["found_until"])
        self.assertTrue(any("runs to the end" in n for n in r["notes"]), r["notes"])
        self.assertGreater(r["window_s"], 2.9)                           # benchmark + the tail after it

    def env(self, **extra):
        e = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1", "JOULES_TELEMETRY": "0",
             "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power, "NO_COLOR": "1", **extra}
        e.pop("JOULES_SIMULATE", None)
        if "PYTHONUNBUFFERED" not in extra:
            e.pop("PYTHONUNBUFFERED", None)   # (often set in CI): the tests see what joules sets
        return e

    def test_a_background_process_left_running_does_not_hang_joules(self):
        """joules itself finishes promptly; the leftover process keeps its output (here it holds it 30 s)."""
        Path(self.script).write_text("#!/bin/bash\n(sleep 30) &\necho '>>> running'\nsleep 0.5\necho '>>> done'\n")
        out = os.path.join(self.d, "r.json")
        with open(os.path.join(self.d, "stdout"), "wb") as so:
            t0 = time.monotonic()
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--out", out, "--from", ">>> running",
                                  "--until", ">>> done", "--", "bash", self.script], stdout=so, stderr=subprocess.DEVNULL, env=self.env())
            code = q.wait(timeout=25)
            took = time.monotonic() - t0
        r = json.loads(Path(out).read_text())
        self.assertEqual(code, 0)
        self.assertLess(took, 8)
        self.assertTrue(0.3 < r["window_s"] < 1.5, r["window_s"])
        self.assertLess(r["duration_s"], 2.0)                            # measuring stopped when the command exited

    def test_only_the_first_markers_count(self):
        """A repeated --from line does not restart the window, and a line matching both starts it but does not end it."""
        Path(self.script).write_text(textwrap.dedent("""\
            #!/bin/bash
            echo 45000 > "$FAKE_NVML_POWER_FILE"; sleep 0.5
            echo ">>> step: running and done"
            sleep 1
            echo ">>> step: running again"
            sleep 1
            echo ">>> step: done"
            sleep 0.5; echo ">>> step: done again"; sleep 0.3
            """))
        p, r, _ = self.joules(["--from", r"running", "--until", r"done"])
        self.assertEqual(r["window"]["from_line"], ">>> step: running and done")
        self.assertEqual(r["window"]["until_line"], ">>> step: done")
        self.assertTrue(1.9 < r["window_s"] < 2.6, r["window_s"])

    def test_markers_on_stderr_and_output_that_is_not_utf8(self):
        Path(self.script).write_text(textwrap.dedent("""\
            #!/bin/bash
            echo 45000 > "$FAKE_NVML_POWER_FILE"; sleep 0.5
            printf 'caf\\xe9 \\xff\\xfe raw bytes\\n'
            echo ">>> running" >&2
            sleep 1
            echo ">>> done" >&2
            sleep 0.3
            """))
        p = subprocess.run([sys.executable, "-m", "joules", "run", "--no-cpu", "--out", os.path.join(self.d, "r.json"),
                            "--from", ">>> running", "--until", ">>> done", "--", "bash", self.script], capture_output=True,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1",
                                "JOULES_TELEMETRY": "0", "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power},
                           timeout=60)
        r = json.loads(Path(self.d, "r.json").read_text())
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(b"caf\xe9 \xff\xfe raw bytes\n", p.stdout)         # passed on byte for byte
        self.assertTrue(r["window"]["found_until"])
        self.assertAlmostEqual(r["window_energy_j"], 45.0 * r["window_s"], delta=1e-6 * r["window_energy_j"])

    def test_ctrl_c_still_reports_and_passes_the_exit_code(self):
        import signal
        Path(self.script).write_text(textwrap.dedent("""\
            import signal, sys, time
            def stop(*_):
                print(">>> done", flush=True)
                sys.exit(7)
            signal.signal(signal.SIGINT, stop)
            open(sys.argv[1], "w").write("45000"); time.sleep(0.5)
            print(">>> running", flush=True)
            while True:
                time.sleep(0.1)
            """))
        out = os.path.join(self.d, "r.json")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1", "JOULES_TELEMETRY": "0",
               "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power}
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--out", out, "--from", ">>> running",
                              "--until", ">>> done", "--", sys.executable, self.script, self.power], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env, start_new_session=True)
        for ln in iter(q.stdout.readline, b""):                           # its handler is set and it is waiting
            if ln.strip() == b">>> running":
                break
        time.sleep(1.0)
        t_sig = time.monotonic()
        os.killpg(q.pid, signal.SIGINT)                                  # Ctrl-C in a terminal reaches the whole group
        so, se = q.communicate(timeout=30)
        took = time.monotonic() - t_sig
        r = json.loads(Path(out).read_text())
        self.assertEqual(q.returncode, 7, f"{took:.1f} s after SIGINT; stdout {so[-300:]!r}; stderr {se[-600:]!r}")
        self.assertEqual(r["exit_code"], 7)
        self.assertTrue(r["window"]["found_until"])
        self.assertAlmostEqual(r["window_energy_j"], 45.0 * r["window_s"], delta=1e-6 * r["window_energy_j"])

    # ---- found in review round 2
    def run_j(self, args, cmd, lib=None, timeout=60, **kw):
        out = os.path.join(self.d, "r.json")
        streams = {"stdout": kw.pop("stdout", subprocess.PIPE), "stderr": kw.pop("stderr", subprocess.PIPE)}
        p = subprocess.run([sys.executable, "-m", "joules", "run", "--out", out, *args, "--", *cmd], **streams,
                           env=self.env(JOULES_NVML_LIB=lib or self.laptop), timeout=timeout, **kw)
        return p, (json.loads(Path(out).read_text()) if os.path.exists(out) else None)

    def test_a_c_program_that_buffers_into_pipes_still_marks_on_time(self):
        """C stdio buffers a pipe until exit; on its own terminal the program flushes each line, so marks are on time."""
        src = os.path.join(self.d, "cbuf.c")
        Path(src).write_text('#include <stdio.h>\n#include <unistd.h>\nint main(void){puts("loading");sleep(1);'
                             'puts("start");sleep(2);puts("end");sleep(1);return 0;}\n')
        exe = os.path.join(self.d, "cbuf")
        subprocess.run(["gcc", "-O2", "-o", exe, src], check=True)
        shown, r = self.on_terminal(["--no-cpu", "--from", "^start", "--until", "^end"], [exe])
        self.assertTrue(1.8 < r["window_s"] < 2.4, r["window"])
        self.assertTrue(0.8 < r["window"]["starts_at_s"] < 1.5, r["window"])
        self.assertIn(b"loading\r\nstart\r\nend\r\n", shown)           # as on screen: the terminal adds \r once

    def on_terminal(self, args, cmd):
        """joules with its stdout and stderr on a terminal (a pseudo-terminal), as when run from a shell."""
        import pty
        out = os.path.join(self.d, "r.json")
        m, sl = pty.openpty()
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--out", out, *args, "--", *cmd],
                             stdout=sl, stderr=sl, env=self.env())
        os.close(sl)
        shown = b""
        while True:
            try:
                b = os.read(m, 65536)
            except OSError:
                break
            if not b:
                break
            shown += b
        os.close(m)
        q.wait(timeout=30)
        return shown, json.loads(Path(out).read_text())

    def test_the_command_sees_a_terminal_only_where_joules_has_one(self):
        cmd = ["bash", "-c", "[ -t 1 ] && echo out:TTY || echo out:NOTTY; [ -t 2 ] && echo err:TTY >&2 || echo err:NOTTY >&2; echo go"]
        p, r = self.run_j(["--no-cpu", "--from", "go"], cmd)
        self.assertIn(b"out:NOTTY", p.stdout)                             # joules' output is a pipe here
        self.assertIn(b"err:NOTTY", p.stderr)
        shown, r = self.on_terminal(["--no-cpu", "--from", "go"], cmd)
        self.assertIn(b"out:TTY", shown)
        self.assertIn(b"err:TTY", shown)
        self.assertTrue(r["window"]["found_from"])

    def test_anchored_markers_in_a_burst_of_lines(self):
        p, r = self.run_j(["--no-cpu", "--from", "^b$", "--until", "^c$"],
                          ["bash", "-c", "printf 'a\\nb\\r\\n'; sleep 0.5; printf 'x\\nc\\ny\\n'; sleep 0.2"])
        self.assertTrue(r["window"]["found_from"] and r["window"]["found_until"], r["notes"])
        self.assertTrue(0.3 < r["window_s"] < 1.0, r["window_s"])

    def test_stdout_and_stderr_keep_their_order_when_they_go_to_one_place(self):
        code = ("import sys\nfor i in range(200):\n    sys.stdout.write(f'o{i}\\n'); sys.stdout.flush()\n"
                "    sys.stderr.write(f'e{i}\\n'); sys.stderr.flush()\nprint('START'); sys.stdout.flush()\n"
                "sys.stderr.write('STOP\\n'); sys.stderr.flush()\nimport time; time.sleep(0.5)\n")
        for _ in range(3):
            p, r = self.run_j(["--no-cpu", "--from", "START", "--until", "STOP"], [sys.executable, "-c", code],
                              stderr=subprocess.STDOUT)
            lines = [x for x in p.stdout.decode().splitlines() if x[:1] in "oe" and x[1:].isdigit()]
            self.assertEqual(lines, [f"{c}{i}" for i in range(200) for c in "oe"])
            self.assertTrue(r["window"]["found_until"], r["notes"])         # STOP on stderr, right after START
            self.assertLess(r["window_s"], 0.3)

    def test_a_marker_printed_in_colour(self):
        p, r = self.run_j(["--no-cpu", "--from", ">>> running trie"],
                          ["bash", "-c", "printf '\\033[1;32m>>>\\033[0m running trie\\n'; sleep 0.3"])
        self.assertTrue(r["window"]["found_from"], r["notes"])
        self.assertEqual(r["window"]["from_line"], ">>> running trie")
        self.assertIn(b"\x1b[1;32m>>>\x1b[0m running trie", p.stdout)     # passed on with its colours

    def test_nothing_is_lost_when_a_leftover_process_is_handed_over(self):
        code = ("import os, sys, time\n"            # a writer that outlives the command, writing through the hand-over
                "if os.fork() == 0:\n"
                "    for i in range(1, 30001):\n        sys.stdout.write(f'L{i}\\n'); sys.stdout.flush(); time.sleep(0.0002)\n"
                "    os._exit(0)\n"
                "time.sleep(0.3)\n")
        outf = os.path.join(self.d, "stdout")
        with open(outf, "wb") as so:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "L100", "--until", "L200",
                                  "--", sys.executable, "-c", code], stdout=so, stderr=subprocess.DEVNULL, env=self.env())
            self.assertEqual(q.wait(timeout=30), 0)
        want = [f"L{i}" for i in range(1, 30001)]
        for _ in range(200):                                             # the copier may still be finishing
            got = [x for x in Path(outf).read_text().splitlines() if x.startswith("L")]
            if len(got) >= len(want):
                break
            time.sleep(0.1)
        self.assertEqual(got, want)

    def test_a_leftover_process_does_not_lengthen_the_measurement(self):
        p0, r0 = self.run_j(["--no-cpu"], ["bash", "-c", "(sleep 4) & sleep 1"], timeout=30)
        p1, r1 = self.run_j(["--no-cpu", "--from", "x"], ["bash", "-c", "(sleep 4) & sleep 1"], timeout=30)
        self.assertLess(r1["duration_s"], r0["duration_s"] + 0.3)
        self.assertLess(r1["totals"]["devices_j"], r0["totals"]["devices_j"] * 1.3)

    def test_a_server_left_running_keeps_its_output_after_joules_exits(self):
        log = os.path.join(self.d, "server.log")
        Path(self.script).write_text(textwrap.dedent("""\
            #!/bin/bash
            ( trap 'echo "server got SIGPIPE" >> "$LOG"; exit 1' PIPE
              for i in $(seq 15); do echo "server log $i" || { echo "server write failed" >> "$LOG"; exit 1; }; sleep 0.2; done
              echo "server finished normally" >> "$LOG" ) &
            echo start; sleep 0.5; echo end
            """))
        with open(os.path.join(self.d, "stdout"), "wb") as so:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "start", "--", "bash", self.script],
                                 stdout=so, stderr=subprocess.DEVNULL, env=self.env(LOG=log))
            self.assertEqual(q.wait(timeout=20), 0)
        for _ in range(100):
            if os.path.exists(log):
                break
            time.sleep(0.1)
        self.assertEqual(Path(log).read_text().strip(), "server finished normally")
        self.assertIn(b"server log 15", Path(self.d, "stdout").read_bytes())

    def test_a_closed_output_stops_the_command_as_without_a_window(self):
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from x -- yes | head -1", shell=True,
                           capture_output=True, env=self.env(), timeout=30)
        self.assertEqual(q.stdout, b"y\n")

    def test_the_cpu_time_estimate_is_not_put_in_a_window(self):
        p, r = self.run_j(["--from", "start", "--until", "end"], ["bash", "-c", "echo start; sleep 0.3; echo end"],
                          lib=os.path.join(self.d, "no-such-lib.so"))
        if not any(x["kind"] == "est" for x in r["devices"]):
            self.skipTest("this machine has real energy counters")
        self.assertTrue(any("CPU-time estimate" in n for n in r["notes"]), r["notes"])
        if all(x["kind"] == "est" for x in r["devices"]):
            self.assertIsNone(r["window_energy_j"])

    def test_a_marker_inside_a_very_long_line(self):
        code = 'import sys, time; sys.stdout.write("a" * 1500000 + "START" + "b" * 1500000 + "\\n"); sys.stdout.flush(); time.sleep(0.5)'
        p, r = self.run_j(["--no-cpu", "--from", "START"], [sys.executable, "-c", code])
        self.assertTrue(r["window"]["found_from"], r["notes"])
        self.assertEqual(len(p.stdout), 3000006)                         # passed on in full

    def test_chatty_output_is_not_slowed_down(self):
        cmd = ["bash", "-c", "yes | head -c 40000000"]
        t0 = time.monotonic(); self.run_j(["--no-cpu"], cmd); plain = time.monotonic() - t0
        t0 = time.monotonic(); p, r = self.run_j(["--no-cpu", "--from", "never"], cmd); win = time.monotonic() - t0
        self.assertEqual(len(p.stdout), 40000000)
        self.assertLess(win, plain + 3.0)

    def test_until_alone_and_an_empty_from(self):
        p, r = self.run_j(["--no-cpu", "--until", "no such line"], ["bash", "-c", "echo hi; sleep 0.3"])
        self.assertTrue(any("never appeared in the command's output" in n and "--from" not in n for n in r["notes"]), r["notes"])
        p, r = self.run_j(["--no-cpu", "--from", ""], ["bash", "-c", "sleep 0.3; echo first; sleep 0.5"])
        self.assertTrue(r["window"]["found_from"])                       # an empty pattern matches the first line
        self.assertTrue(0.2 < r["window"]["starts_at_s"] < 0.9, r["window"])

    # ---- found in review round 4
    def test_markers_on_streams_that_go_to_different_places(self):
        """print() to stdout and logging to stderr, each into its own file: both marks are found, every time."""
        code = ("import sys, time\ntime.sleep(0.3)\nsys.stdout.write('START\\n'); sys.stdout.flush()\n"
                "sys.stderr.write('END\\n'); sys.stderr.flush()\ntime.sleep(0.6)\n")
        out = os.path.join(self.d, "r.json")
        for _ in range(5):
            with open(os.path.join(self.d, "o"), "wb") as o, open(os.path.join(self.d, "e"), "wb") as e:
                subprocess.run([sys.executable, "-m", "joules", "run", "--no-cpu", "--out", out, "--from", "START",
                                "--until", "END", "--", sys.executable, "-c", code], stdout=o, stderr=e, env=self.env(), timeout=60)
            r = json.loads(Path(out).read_text())
            self.assertTrue(r["window"]["found_until"], r["notes"])
            self.assertLess(r["window_s"], 0.3)

    def test_charset_codes_anchors_and_a_closed_output(self):
        p, r = self.run_j(["--no-cpu", "--from", "^>>> running"],
                          ["bash", "-c", "printf '\\033(B\\033[m>>> running\\n'; sleep 0.3"])   # what tput sgr0 prints
        self.assertTrue(r["window"]["found_from"], r["notes"])
        p, r = self.run_j(["--no-cpu", "--from", r"\A>>> start"], ["bash", "-c", "printf 'one\\n>>> start\\nthree\\n'; sleep 0.3"])
        self.assertTrue(r["window"]["found_from"], r["notes"])
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from x -- bash -c 'echo hi; echo x; sleep 0.3' >&-",
                           shell=True, capture_output=True, env=self.env(), timeout=60)
        self.assertNotIn(b"Traceback", q.stderr)

    def test_until_is_never_the_same_long_line_as_from(self):
        code = ("import sys, time\nw = sys.stdout.buffer\n"                 # MARK just past where the long line is trimmed
                "w.write(b'a' * (33 * 65536 - 200) + b' MARK ' + b'b' * 300000 + b'\\n'); w.flush()\n"
                "time.sleep(1.0)\nw.write(b'MARK second\\n'); w.flush()\ntime.sleep(0.2)\n")
        p, r = self.run_j(["--no-cpu", "--from", "MARK", "--until", "MARK"], [sys.executable, "-c", code])
        self.assertEqual(r["window"]["until_line"], "MARK second")
        self.assertTrue(0.8 < r["window_s"] < 1.5, r["window_s"])

    def test_markers_read_only_at_exit_are_flagged(self):
        src = os.path.join(self.d, "late.c")
        Path(src).write_text('#include <stdio.h>\n#include <unistd.h>\nint main(void){puts(">>> running");sleep(1);'
                             'puts(">>> done");sleep(1);return 0;}\n')
        exe = os.path.join(self.d, "late")
        subprocess.run(["gcc", "-O2", "-o", exe, src], check=True)
        p, r = self.run_j(["--no-cpu", "--from", ">>> running", "--until", ">>> done"], [exe])   # into a pipe: held to exit
        self.assertTrue(any("only read when the command exited" in n for n in r["notes"]), r["notes"])
        p, r = self.run_j(["--no-cpu", "--from", ">>> running trie", "--until", r">>> done\."], ["bash", self.script])
        self.assertFalse(any("only read when" in n for n in r["notes"]), r["notes"])           # an on-time run: no note

    def test_a_resized_terminal_reaches_the_command(self):
        import fcntl
        import pty
        import signal as sig
        import struct
        import termios
        m, sl = pty.openpty()
        fcntl.ioctl(sl, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        code = ("import os, signal, time\n"   # (os.write: a resize may send two SIGWINCH at once, and print is not reentrant)
                "signal.signal(signal.SIGWINCH, lambda *a: os.write(1, b'size %d %d\\n' % tuple(os.get_terminal_size(1))))\n"
                "print('size', *os.get_terminal_size(1), flush=True)\ntime.sleep(2.5)\n")
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--until", "zzz", "--", sys.executable, "-c", code],
                             stdin=sl, stdout=sl, stderr=sl, env=self.env(), start_new_session=True)
        os.close(sl)
        shown, t0 = b"", time.monotonic()
        while time.monotonic() - t0 < 20:
            if b"size 100 30" in shown and not hasattr(self, "_resized"):
                time.sleep(0.3)
                fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 160, 0, 0))   # the user resizes the window
                os.killpg(q.pid, sig.SIGWINCH)
                self._resized = True
            r = select_read(m)
            if r is None:
                break
            shown += r
        os.close(m)
        q.wait(timeout=20)
        self.assertIn(b"size 160 50", shown)

    # ---- found in review round 5
    def test_a_slow_reader_on_joules_output_loses_nothing(self):
        """A pager, or tee to a slow disk: joules waits for its reader instead of dropping output at the end."""
        code = "import sys; print('go'); sys.stdout.write('x\\n' * 60_000)"   # fits in the pipes: the command exits first
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from go -- {shlex.quote(sys.executable)} -c "
                           f"{shlex.quote(code)} 2>/dev/null | (sleep 5; wc -c)", shell=True, capture_output=True,
                           env=self.env(), timeout=90)
        self.assertEqual(int(q.stdout.split()[0]), 3 + 120_000)

    def test_crlf_lines_lookaround_patterns_and_no_false_note(self):
        p, r = self.run_j(["--no-cpu", "--from", "start", "--until", "^$"],
                          ["bash", "-c", "printf 'start\\r\\n'; sleep 0.5; printf 'a\\r\\n\\r\\n'; sleep 0.2"])
        self.assertEqual(r["window"]["until_line"], "")                 # the real empty line, not a phantom one
        self.assertTrue(0.4 < r["window_s"] < 0.8, r["window_s"])
        p, r = self.run_j(["--no-cpu", "--from", "go", "--until", "(?s)done(?!.)"],
                          ["bash", "-c", "echo go; sleep 0.3; printf 'done\\nnext\\n'; sleep 0.2"])
        self.assertTrue(r["window"]["found_until"], r["notes"])
        p, r = self.run_j(["--no-cpu", "--from", "start"], ["bash", "-c", "sleep 1; echo start; sleep 0.05"])
        self.assertFalse(any("only read when" in n for n in r["notes"]), r["notes"])

    def _on_own_terminal(self, cmd, cols=50, rows=30, stdout_file=None, resize=None):
        """joules on a terminal of its own session (so a resize sends a real SIGWINCH); returns what it showed."""
        import fcntl
        import pty
        import struct
        import termios
        m, sl = pty.openpty()
        fcntl.ioctl(sl, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

        def own_session():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        out = open(stdout_file, "wb") if stdout_file else None
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", *cmd], stdin=sl, stdout=out or sl,
                             stderr=sl, env=self.env(), preexec_fn=own_session)
        os.close(sl)
        shown, t0, done = b"", time.monotonic(), False
        while time.monotonic() - t0 < 25:
            if resize and not done and time.monotonic() - t0 > 1.5:
                fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", resize[1], resize[0], 0, 0)); done = True
            b = select_read(m)
            if b is None:
                break
            shown += b
            if q.poll() is not None and not b:
                break
        os.close(m)
        q.wait(timeout=20)
        if out:
            out.close()
        return shown

    def test_stderr_gets_its_own_terminal_size_when_stdout_is_a_file(self):
        shown = self._on_own_terminal(["--from", "x", "--", sys.executable, "-c",
                                       "import os; print('x'); import sys; sys.stderr.write('%d %d\\n' % tuple(os.get_terminal_size(2)))"],
                                      stdout_file=os.path.join(self.d, "out"))
        self.assertIn(b"50 30", shown)

    def test_a_resize_reaches_the_command_s_children_too(self):
        code = ("import os, signal, time\ngot = []\n"
                "signal.signal(signal.SIGWINCH, lambda *a: got.append(os.get_terminal_size(1).columns))\n"
                "time.sleep(3)\nprint('seen', got, flush=True)\n")
        shown = self._on_own_terminal(["--until", "zzz", "--", "bash", "-c", f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}; true"],
                                      resize=(100, 40))
        seen = [ln for ln in shown.decode(errors="replace").splitlines() if ln.startswith("seen")]
        self.assertTrue(seen and "100" in seen[0], shown[-500:])

    # ---- found in review round 6
    def test_pythonunbuffered_only_where_output_is_a_pipe(self):
        code = "import os; print('go'); print('PU', os.environ.get('PYTHONUNBUFFERED'))"
        p, r = self.run_j(["--no-cpu", "--from", "go"], [sys.executable, "-c", code])
        self.assertIn(b"PU 1", p.stdout)                                  # into a pipe: set, so lines are not held
        shown = self._on_own_terminal(["--from", "go", "--", sys.executable, "-c", code])
        self.assertIn(b"PU None", shown)                                  # on a terminal: left alone

    def test_a_closed_stdout_stays_closed_for_the_command(self):
        code = "import os, sys\ntry:\n    os.write(1, b'go\\n'); sys.stderr.write('wrote\\n')\nexcept OSError as e:\n    sys.stderr.write('EBADF\\n' if e.errno == 9 else repr(e))\n"
        for extra in ([], ["--from", "go"]):
            q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu {' '.join(extra)} -- "
                               f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} >&-", shell=True, capture_output=True,
                               env=self.env(), timeout=60)
            self.assertIn(b"EBADF", q.stderr, extra)

    def test_the_windows_reader_closes_the_pipe_when_our_output_goes(self):
        """When our output goes away (a pager quits), the command gets a broken pipe instead of blocking for ever.
        Uses the streams _start sets up on Windows (pipes; Windows itself cannot be run here)."""
        import threading
        from unittest import mock
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Gone:
            buffer = None
            def write(self, *a): raise BrokenPipeError(32, "gone")
            def flush(self): pass

        class Win:
            first_out, paused = {}, False
            def pending(self): return False
        cmd = [sys.executable, "-c", "import sys\nwhile True: sys.stdout.write('y' * 4096)"]
        with open(os.devnull, "w") as quiet, mock.patch.object(cli.os, "name", "nt"), \
                mock.patch.object(sys, "stdout", quiet), mock.patch.object(sys, "stderr", quiet):
            q = cli._start(cmd, piped=True)
        fd, _, close = q.joules_streams[0]
        cli._pump(fd, Gone(), Win(), {"stop": threading.Event()}, close or None)
        try:
            q.wait(timeout=10)
        finally:
            if q.poll() is None:
                q.kill()
            for f in (q.stdout, q.stderr):
                if f:
                    with contextlib.suppress(OSError, ValueError):
                        f.close()
        self.assertIsNotNone(q.returncode)

    # ---- found in review round 7
    def test_a_slow_reader_does_not_move_the_window_or_the_report(self):
        """A pager or slow pipe on joules' output: marks keep the time each line was printed, no false notes, and
        the report comes after all of the command's output."""
        code = ("import sys, time\nfor i in range(1000): sys.stdout.write('y' * 99 + '\\n')\n"
                "print('>>> running', flush=True); time.sleep(2); print('>>> done', flush=True); time.sleep(0.5)\n")
        out = os.path.join(self.d, "r.json")
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from running --until done --out {out} -- "
                           f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} 2>&1 | (sleep 4; cat)", shell=True,
                           capture_output=True, env=self.env(), timeout=90)
        r = json.loads(Path(out).read_text())
        self.assertTrue(1.9 < r["window_s"] < 2.3, r["window_s"])
        self.assertEqual(r["notes"], [])
        text = q.stdout.decode()
        self.assertEqual(text.count("y" * 99), 1000)
        self.assertGreater(text.index("joules ·"), text.index(">>> done"))

    def test_held_output_after_an_earlier_log_line_is_flagged(self):
        src = os.path.join(self.d, "held.c")
        Path(src).write_text('#include <stdio.h>\n#include <unistd.h>\nint main(void){fprintf(stderr,"loading\\n");'
                             'sleep(1);puts(">>> running");sleep(1);puts(">>> done");sleep(1);return 0;}\n')
        exe = os.path.join(self.d, "held")
        subprocess.run(["gcc", "-O2", "-o", exe, src], check=True)
        p, r = self.run_j(["--no-cpu", "--from", ">>> running", "--until", ">>> done"], [exe], stderr=subprocess.STDOUT)
        self.assertTrue(any("under 10 ms" in n for n in r["notes"]), r["notes"])     # one shared stream
        err = open(os.path.join(self.d, "err"), "wb")
        p, r = self.run_j(["--no-cpu", "--from", ">>> running", "--until", ">>> done"], [exe], stderr=err)
        err.close()
        self.assertTrue(any("only read when" in n for n in r["notes"]), r["notes"])  # the log line was on stderr
        # (a run too short to matter gets no note: LateNote.test_a_very_short_run_gets_no_note; run end to end its
        # length depends on how fast the machine starts the command, so it is not checked here)

    def test_a_sigterm_to_joules_alone_leaves_the_command_running(self):
        """As without a window: joules ends, the command carries on and its output still arrives."""
        out = os.path.join(self.d, "out")
        with open(out, "wb") as f:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "tick", "--", "bash", "-c",
                                  "for i in 1 2 3 4 5 6; do echo tick $i; sleep 0.3; done"], stdout=f,
                                 stderr=subprocess.DEVNULL, env=self.env())
            time.sleep(0.8)
            q.terminate()
            self.assertEqual(q.wait(timeout=20), -15)
        deadline = time.monotonic() + 10
        while "tick 6" not in Path(out).read_text() and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertIn("tick 6", Path(out).read_text())

    def test_the_reader_handles_file_descriptors_past_1024(self):
        import resource
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard < 2100:
            self.skipTest("cannot open that many files here")
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2100), hard))
        self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))
        r, w = os.pipe()
        high = os.dup2(r, 2000)
        os.close(r)

        class Sink:
            buffer, got = None, []
            def write(self, t): self.got.append(t)
            def flush(self): pass

        class Win:
            first_out, behind, paused = {}, False, False
            def pending(self): return False
        os.write(w, b"hello\n")
        os.close(w)
        sink = Sink()
        outs = {high: cli._Out(sink)}
        state = {"stop": threading.Event(), "exited": threading.Event(), "caught_up": threading.Event()}
        cli._pump_all([(high, sink, functools.partial(os.close, high))], Win(), state, outs)
        outs[high].close()
        outs[high].thread.join(5)
        self.assertEqual("".join(sink.got), "hello\n")
        self.assertEqual(state[high], "done")

    def test_windows_children_get_unbuffered_python_and_consoles_get_the_bytes_as_written(self):
        """On Windows the command always gets pipes, so Python there must not hold its output; and a console gets
        the command's own bytes (its code page), not bytes reread as UTF-8. Windows itself cannot be run here."""
        from unittest import mock
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        code = "import os; print(os.environ.get('PYTHONUNBUFFERED'))"
        env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
        with open(os.devnull, "w") as quiet, mock.patch.object(cli.os, "name", "nt"), \
                mock.patch.object(cli, "_isatty", lambda f: True), mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(sys, "stdout", quiet), mock.patch.object(sys, "stderr", quiet):
            q = cli._start([sys.executable, "-c", code], piped=True)
        got = q.stdout.read()
        q.wait(timeout=20)
        for f in (q.stdout, q.stderr):
            if f:
                f.close()
        self.assertEqual(got.strip(), b"1")
        r, w = os.pipe()

        class Console:
            def fileno(self): return w
            def flush(self): pass
            buffer = None   # (a UTF-8 layer would be used if os.write were not)
        with mock.patch.object(cli.os, "name", "nt"), mock.patch.object(cli, "_isatty", lambda f: True):
            self.assertTrue(cli._write(Console(), b"caf\x82\n"))   # cp437 é
        os.close(w)
        self.assertEqual(os.read(r, 100), b"caf\x82\n")
        os.close(r)


    # ---- found in review round 8
    def test_a_sigterm_after_the_command_exits_loses_nothing(self):
        """SIGTERM while joules still passes on output for a slow reader, or waits on a server the command left: all
        the output still arrives, from a helper, as without a window."""
        out = os.path.join(self.d, "out")
        big = "import sys; sys.stdout.write('A\\n' + 'z' * (4 << 20))"
        slow = "import sys, time\nwhile True:\n    b = sys.stdin.buffer.read(65536)\n    if not b: break\n" \
               "    sys.stdout.buffer.write(b); time.sleep(0.1)\n"
        with open(out, "wb") as f:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "A", "--", sys.executable, "-c",
                                  big], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=self.env())
            rd = subprocess.Popen([sys.executable, "-c", slow], stdin=q.stdout, stdout=f)
            q.stdout.close()
            time.sleep(1.5)
            q.terminate()
            self.assertEqual(q.wait(timeout=20), -15)
            rd.wait(timeout=60)
        self.assertEqual(os.path.getsize(out), 2 + (4 << 20))
        srv = "(for i in $(seq 1 15); do echo srv $i; sleep 0.2; done; echo srv end) & echo A"
        with open(out, "wb") as f:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "A", "--", "bash", "-c", srv],
                                 stdout=f, stderr=subprocess.DEVNULL, env=self.env())
            time.sleep(1.2)
            q.terminate()
            self.assertEqual(q.wait(timeout=20), -15)
        deadline = time.monotonic() + 10
        while "srv end" not in Path(out).read_text() and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertIn("srv end", Path(out).read_text())

    def test_a_full_queue_at_exit_is_not_called_a_process_holding_the_output(self):
        """The command fills joules' queue for a slow reader and exits, starting nothing: no "still open" note."""
        code = "import os\nos.write(1, b'A\\n')\nfor i in range(256): os.write(1, b'x' * 65535 + b'\\n')\nos.write(1, b'y' * 65533 + b'\\n')\n"
        reader = ("import os, sys, time\ntime.sleep(4); n = 0\nwhile True:\n"
                  "    b = os.read(0, 1 << 20)\n    if not b: break\n    n += len(b)\nprint(n)\n")
        err = os.path.join(self.d, "err")
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from A -- {shlex.quote(sys.executable)} -c "
                           f"{shlex.quote(code)} 2>{err} | {shlex.quote(sys.executable)} -c {shlex.quote(reader)}", shell=True,
                           capture_output=True, env=self.env(), timeout=120)
        self.assertEqual(int(q.stdout.split()[0]), 257 * 65536)   # (joules' queue exactly full at the exit)
        self.assertNotIn("still open", Path(err).read_text())

    def test_an_empty_marker_line_is_named_as_such(self):
        p, r = self.run_j(["--no-cpu", "--from", "^$", "--until", "x"], ["sh", "-c", "echo; sleep 0.3; echo x"])
        self.assertIn("from: (an empty line)", p.stdout.decode() + p.stderr.decode())

    # ---- found in review round 9
    def test_a_sigterm_with_a_stuck_reader_ends_joules_and_loses_nothing(self):
        """A paused pager on joules' output: SIGTERM still ends joules at once, and once the reader reads again it
        gets every byte, once."""
        big = "import sys; sys.stdout.write(''.join('line %06d\\n' % i for i in range(30000)))"
        late = "import sys, time\ntime.sleep(4)\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n"
        out = os.path.join(self.d, "out")
        with open(out, "wb") as f:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "line", "--", sys.executable,
                                  "-c", big], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=self.env())
            rd = subprocess.Popen([sys.executable, "-c", late], stdin=q.stdout, stdout=f)
            q.stdout.close()
            time.sleep(1.5)
            t0 = time.monotonic()
            q.terminate()
            self.assertEqual(q.wait(timeout=20), -15)
            self.assertLess(time.monotonic() - t0, 2.5)
            rd.wait(timeout=30)
        self.assertEqual(Path(out).read_text(), "".join("line %06d\n" % i for i in range(30000)))

    def test_a_full_disk_does_not_kill_the_command(self):
        if not os.path.exists("/dev/full"):
            self.skipTest("no /dev/full here")
        with open("/dev/full", "wb") as full:
            p, r = self.run_j(["--no-cpu", "--from", "a"], ["bash", "-c", "echo a; sleep 0.3; echo b; echo 'still running' >&2"],
                              stdout=full)
        self.assertIn(b"still running", p.stderr)
        self.assertEqual(r["exit_code"], 0)

    def test_crlf_split_between_reads_is_one_line_end(self):
        p, r = self.run_j(["--no-cpu", "--from", "START", "--until", "^$"],
                          ["bash", "-c", "printf 'START\\r'; sleep 0.3; printf '\\nwork\\r\\n'; sleep 1; printf '\\r\\n'; sleep 0.2"])
        self.assertTrue(1.1 < r["window_s"] < 1.6, r["window_s"])

    def test_falling_more_than_16_mb_behind_is_noted(self):
        code = "import sys\nprint('START', flush=True)\nsys.stdout.write(('x' * 99 + '\\n') * 250000)\nprint('END')\n"
        err = os.path.join(self.d, "err")
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from ^START --until END -- "
                           f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} 2>{err} | (sleep 3; wc -c)", shell=True,
                           capture_output=True, env=self.env(), timeout=120)
        self.assertEqual(int(q.stdout.split()[0]), 6 + 25_000_000 + 4)
        self.assertIn("fell more than 16 MB behind", Path(err).read_text())

    # ---- found in review round 10
    def test_a_sigterm_with_a_stuck_reader_repeats_nothing_after_odd_sized_writes(self):
        """Odd-sized writes leave a pipe's pages part-filled: still nothing may be passed on twice."""
        code = ("import os, time\nos.write(1, b'a' * 2999 + b'\\n'); time.sleep(0.2)\nos.write(1, b'b' * 6999 + b'\\n')\n"
                "time.sleep(0.2)\nos.write(1, b''.join(b'line %07d\\n' % i for i in range(20000)))\n")
        late = "import sys, time\ntime.sleep(4)\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n"
        out = os.path.join(self.d, "out")
        with open(out, "wb") as f:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "line", "--", sys.executable,
                                  "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=self.env())
            rd = subprocess.Popen([sys.executable, "-c", late], stdin=q.stdout, stdout=f)
            q.stdout.close()
            time.sleep(1.5)
            q.terminate()
            self.assertEqual(q.wait(timeout=20), -15)
            rd.wait(timeout=30)
        want = b"a" * 2999 + b"\n" + b"b" * 6999 + b"\n" + b"".join(b"line %07d\n" % i for i in range(20000))
        self.assertEqual(Path(out).read_bytes(), want)

    def test_take_does_not_wait_out_its_timeout(self):
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        r, w = os.pipe()
        try:
            out = cli._Out(os.fdopen(w, "wb", closefd=False))
            out.put(b"x" * 300_000)   # more than the pipe holds: the writer gets stuck
            time.sleep(0.3)
            t0 = time.monotonic()
            _, rest = out.take()
            self.assertLess(time.monotonic() - t0, 0.4)
            got = b""
            while len(got) < 300_000 - len(rest):
                got += os.read(r, 65536)
            self.assertEqual(len(got) + len(rest), 300_000)   # nothing lost, nothing twice
        finally:
            os.close(r)
            os.close(w)

    def test_one_ctrl_c_after_the_exit_loses_no_output(self):
        """A pager that ignores Ctrl-C (like less) reads late; one Ctrl-C to joules must not drop what it holds."""
        code = "import sys\nfor i in range(30000): print('line %07d' % i + 'x' * 80)\n"
        late = ("import signal, sys, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(4)\n"
                "sys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        out, err = os.path.join(self.d, "out"), os.path.join(self.d, "err")
        with open(out, "wb") as f, open(err, "wb") as e:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "line", "--", sys.executable,
                                  "-c", code], stdout=subprocess.PIPE, stderr=e, env=self.env())
            rd = subprocess.Popen([sys.executable, "-c", late], stdin=q.stdout, stdout=f)
            q.stdout.close()
            time.sleep(2)
            q.send_signal(signal.SIGINT)
            q.wait(timeout=20)
            rd.wait(timeout=30)
        self.assertEqual(Path(out).read_text().count("\n"), 30000)
        self.assertIn("window", Path(err).read_text())

    def test_many_ctrl_cs_at_the_exit_still_give_the_report(self):
        code = ("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('A')\n"
                "[print('l' * 100) for i in range(2000)]\ntime.sleep(0.5)\n")
        for trial in range(6):
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "A", "--", sys.executable, "-c",
                                  code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env())
            time.sleep(0.4 + trial * 0.05)
            while q.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    q.send_signal(signal.SIGINT)
                time.sleep(0.003)
            o, e = q.communicate()
            self.assertEqual(o.count(b"l" * 100), 2000)
            self.assertIn(b"window", e)
            if b"Traceback" in e:   # only after the whole report (later steps are as in a plain run)
                self.assertLess(e.index(b"cost "), e.index(b"Traceback"), e[-800:])

    def test_an_ignored_sigterm_stays_ignored(self):
        q = subprocess.Popen(["bash", "-c", f"trap '' TERM; exec {shlex.quote(sys.executable)} -m joules run --no-cpu --from=A -- "
                              "sh -c 'echo A; sleep 2; echo B'"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env())
        time.sleep(1)
        q.terminate()
        o, e = q.communicate(timeout=20)
        self.assertEqual(q.returncode, 0)
        self.assertIn(b"B", o)
        self.assertIn(b"window", e)

    # ---- found in review round 11
    def _stopped_reader(self, cmd, sig_at, extra=()):
        """joules into a reader that is stopped (a pager waiting for a key); one Ctrl-C to joules at sig_at, then the
        reader goes on. Returns joules' exit code, what the reader got, and joules' stderr."""
        out, err = os.path.join(self.d, "out"), os.path.join(self.d, "err")
        with open(out, "wb") as f, open(err, "wb") as e:
            rd = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=f)
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "line", *extra, "--", *cmd],
                                 stdout=rd.stdin, stderr=e, env=self.env())
            rd.stdin.close()
            time.sleep(0.05)
            os.kill(rd.pid, signal.SIGSTOP)
            time.sleep(sig_at)
            q.send_signal(signal.SIGINT)
            try:
                rc = q.wait(timeout=10)
            finally:
                os.kill(rd.pid, signal.SIGCONT)
            rd.wait(timeout=30)
        return rc, out, Path(err).read_text()

    def test_one_ctrl_c_with_a_stopped_reader_and_a_server_left_running(self):
        """| less, with a server the command started: one Ctrl-C ends joules at once and nothing is lost."""
        gen = ("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time\\nfor i in range(5):\\n"
               "    time.sleep(0.5); print(\"bg\", i, flush=True)'])\n"
               "for i in range(4000): sys.stdout.write('line %06d %s\\n' % (i, 'x' * 40))\nsys.stdout.flush()\n")
        for at in (0.5, 2.8):
            rc, out, err = self._stopped_reader([sys.executable, "-c", gen], at)
            self.assertEqual(rc, 0)
            deadline = time.monotonic() + 10
            while "bg 4" not in Path(out).read_text() and time.monotonic() < deadline:
                time.sleep(0.2)
            text = Path(out).read_text()
            self.assertEqual(text.count("line "), 4000, at)
            self.assertEqual([ln for ln in text.splitlines() if ln.startswith("bg")], [f"bg {i}" for i in range(5)])

    def test_output_handed_over_on_ctrl_c_does_not_cut_up_the_report(self):
        gen = "import sys\nfor i in range(3000): sys.stdout.write('line %06d %s\\n' % (i, 'x' * 40))\n"
        out = os.path.join(self.d, "out")
        with open(out, "wb") as f:
            rd = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=f)
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "line", "--", sys.executable,
                                  "-c", gen], stdout=rd.stdin, stderr=rd.stdin, env=self.env())   # one shared stream
            rd.stdin.close()
            time.sleep(0.05)
            os.kill(rd.pid, signal.SIGSTOP)
            time.sleep(2)
            q.send_signal(signal.SIGINT)
            time.sleep(0.5)
            os.kill(rd.pid, signal.SIGCONT)   # (joules' own report goes into the same pipe)
            q.wait(timeout=10)
            rd.wait(timeout=30)
        text = Path(out).read_text()
        a = text.index("joules ·")
        b = text.index("\n", text.index("after joules finishes")) + 1
        report, rest = text[a:b], text[:a] + text[b:]
        self.assertEqual(rest, "".join("line %06d %s\n" % (i, "x" * 40) for i in range(3000)))   # once, whole, in order
        self.assertNotIn("\nline 0", report)   # the report is in one piece (it may start part-way through a line)

    # ---- found in review round 12
    def test_an_until_line_printed_earlier_on_the_other_stream_does_not_end_the_window(self):
        """>out 2>err: an --until line on stdout printed before the --from line on stderr, both read together."""
        out = os.path.join(self.d, "r.json")
        for _ in range(3):
            with open(os.path.join(self.d, "o"), "wb") as o, open(os.path.join(self.d, "e"), "wb") as e:
                subprocess.run([sys.executable, "-m", "joules", "run", "--no-cpu", "--out", out, "--from", ">>> running",
                                "--until", "done", "--", "sh", "-c",
                                "echo 'download done'; echo '>>> running bench' >&2; sleep 1; echo '>>> done'"],
                               stdout=o, stderr=e, env=self.env(), timeout=60)
            r = json.loads(Path(out).read_text())
            self.assertEqual(r["window"]["until_line"], ">>> done")
            self.assertTrue(0.9 < r["window_s"] < 1.4, r["window_s"])

    def test_a_pair_read_together_on_two_streams_says_the_order_is_unknown(self):
        """Fed straight to the window (no timing): --until on stderr read together with --from on stdout is kept aside,
        used only with no later --until, and then said to be of unknown order (and not called held back)."""
        import argparse
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            t = 1.0
            def snapshot(self): return {"t": self.t, "j": {"GPU 0": self.t * 10}}

        class M:
            name, kind, energy_j = "GPU 0", "gpu", 50.0
        clock = Clock()
        for later in (False, True):
            w = cli.Window(clock, 0.0, [M()], "START", "END")
            clock.t = 1.0
            w.feed(["START", "END"], [1, 2])
            self.assertIsNone(w.end)
            if later:
                clock.t = 2.0
                w.feed(["END again"], [1])
            w.freeze({"t": 5.0, "j": {"GPU 0": 50.0}})

            class S:
                t0, t1 = 0.0, 5.0
            out, notes = cli._window_report(argparse.Namespace(window_from="START", window_until="END"), w, S(), [M()], {})
            if later:
                self.assertEqual(out["window"]["until_line"], "END again")
                self.assertFalse([n for n in notes if "not known" in n])
            else:
                self.assertEqual(out["window"]["until_line"], "END")
                self.assertTrue([n for n in notes if "which came first is not known" in n], notes)
                self.assertFalse([n for n in notes if "10 ms" in n or "held" in n], notes)

    # ---- found in review round 13
    def test_anchored_markers_do_not_match_where_a_long_line_was_cut(self):
        code_tpl = ("import sys, time\nprint('A', flush=True)\nsys.stdout.write({line}); sys.stdout.flush()\n"
                    "time.sleep(0.7)\nprint('By', flush=True)\ntime.sleep(0.2)\n")
        for until, line in (("y$", "'x' + 'B' * 1000000 + 'y' * 1000000 + 'z\\n'"), ("^B", "'x' + 'B' * 3000000 + '\\n'")):
            p, r = self.run_j(["--no-cpu", "--from", "^A$", "--until", until],
                              [sys.executable, "-c", code_tpl.format(line=line)])
            self.assertEqual(r["window"]["until_line"], "By", until)

    def test_from_alone_read_at_the_exit_is_flagged(self):
        src = os.path.join(self.d, "late1.c")
        Path(src).write_text('#include <stdio.h>\n#include <unistd.h>\nint main(void){sleep(1);puts(">>> running");'
                             'sleep(1);return 0;}\n')
        exe = os.path.join(self.d, "late1")
        subprocess.run(["gcc", "-O2", "-o", exe, src], check=True)
        p, r = self.run_j(["--no-cpu", "--from", ">>> running"], [exe])   # into a pipe: held to exit
        self.assertTrue(any("only read when the command exited" in n for n in r["notes"]), r["notes"])

    def test_a_pause_with_no_marker_after_it_is_not_noted(self):
        code = "import sys\nprint('A', flush=True)\nsys.stdout.write(('x' * 99 + '\\n') * 250000)\n"
        err = os.path.join(self.d, "err")
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from A --until NEVER -- "
                           f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} 2>{err} | (sleep 3; wc -c)", shell=True,
                           capture_output=True, env=self.env(), timeout=120)
        self.assertEqual(int(q.stdout.split()[0]), 2 + 25_000_000)
        self.assertNotIn("16 MB", Path(err).read_text())

    def test_a_ctrl_c_while_a_server_holds_the_output_says_so(self):
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "A", "--", "sh", "-c",
                              "echo A; (sleep 3; echo bg) &"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env())
        time.sleep(1)
        q.send_signal(signal.SIGINT)
        o, e = q.communicate(timeout=20)
        self.assertIn(b"still open when it exited", e)
        self.assertIn(b"stopped waiting for the command's output", e)
        self.assertIn(b"bg", o)   # (the helper outlives joules; communicate waits for the pipe to close)

    def test_windows_sees_a_reader_gone_as_einval_and_notes_a_pause(self):
        """Windows: writing to a pipe whose reader left fails with EINVAL, not EPIPE; and its reader marks a pause
        for a slow reader. Windows itself cannot be run here."""
        import errno
        import threading
        from unittest import mock
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        out = cli._Out.__new__(cli._Out)
        out.cv, out.handed, out.off, out.kind = threading.Condition(), False, 0, "pipe"

        def einval(*a):
            raise OSError(errno.EINVAL, "invalid argument")
        with mock.patch.object(cli.os, "name", "nt"), mock.patch.object(cli.os, "write", einval), \
                mock.patch.object(cli._Out, "_piece", lambda self, fd: 4096):
            self.assertFalse(out._put(1, b"x" * 10))

        seen = []

        class Full:
            gone, size = False, 1 << 30
            def full(self):
                seen.append(win.paused)
                return True
            def put(self, c): pass

        class Win:
            first_out, paused = {}, False
            def pending(self): return True
        win, state = Win(), {"stop": threading.Event()}
        r, w = os.pipe()

        def stop():
            state["stop"].set()
            os.close(w)
        threading.Timer(0.3, stop).start()
        try:
            cli._pump(r, None, win, state, None, Full())
        finally:
            os.close(r)
        self.assertTrue(any(seen))      # paused while the stream was behind
        self.assertFalse(win.paused)    # and no longer once it ended

    # ---- found in review round 14
    def test_a_greedy_marker_early_in_a_long_line_is_found(self):
        code = ("import sys, time\nsys.stdout.write('Starting run id=42 ' + 'x' * 2000000); sys.stdout.flush()\n"
                "time.sleep(0.5)\nprint(flush=True)\ntime.sleep(0.5)\nprint('Finished', flush=True)\n")
        for pattern in ("Starting.*", "Starting run|WARN.*"):
            p, r = self.run_j(["--no-cpu", "--from", pattern, "--until", "Finished"], [sys.executable, "-c", code])
            self.assertTrue(r["window"]["found_from"], (pattern, r["notes"]))
            self.assertTrue(0.8 < r["window_s"] < 1.4, r["window_s"])

    def test_word_boundaries_and_lookbehinds_see_the_text_before_a_cut(self):
        """A long line written in 64 KB pieces, 'bar' ending one piece and 'foo' starting the next: no \\bfoo there."""
        code = ("import sys, time\nw = sys.stdout.buffer\nprint('A', flush=True); time.sleep(0.3)\n"
                "for i in range(40):\n    piece = bytearray(b'y' * 65536)\n    if i == 15: piece[-3:] = b'bar'\n"
                "    if i == 16: piece[:3] = b'foo'\n    w.write(piece); w.flush(); time.sleep(0.02)\n"
                "w.write(b'\\n'); w.flush(); time.sleep(0.5)\nprint('foo real', flush=True); time.sleep(0.3)\n")
        for until in (r"\bfoo", "(?<!bar)foo"):
            p, r = self.run_j(["--no-cpu", "--from", "^A$", "--until", until], [sys.executable, "-c", code])
            self.assertEqual(r["window"]["until_line"], "foo real", until)

    def test_a_burst_long_before_the_markers_is_not_noted(self):
        code = ("import sys, time\nsys.stdout.write(('x' * 99 + '\\n') * 250000); sys.stdout.flush()\n"
                "time.sleep(4)\nprint('A', flush=True); time.sleep(1); print('B', flush=True)\n")
        err = os.path.join(self.d, "err")
        q = subprocess.run(f"{shlex.quote(sys.executable)} -m joules run --no-cpu --from ^A$ --until ^B$ -- "
                           f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} 2>{err} | (sleep 1.5; wc -c)", shell=True,
                           capture_output=True, env=self.env(), timeout=120)
        self.assertEqual(int(q.stdout.split()[0]), 25_000_000 + 4)
        self.assertNotIn("16 MB", Path(err).read_text())

    # ---- found in review round 15
    def test_a_leading_dot_star_is_not_slow(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        for pattern, want in ((".*START", "START"), ("^.*START", "START"), (".*?START", "START"), (".*A|B", "A|B"),
                              (".*", ".*"), (".*+x", ".*+x"), ("(?i).*x", "(?i).*x"), ("START", "START"),
                              (".+START", ".START"), ("^.+.*START", ".START"), (".*.*START", "START"),
                              (".+?A|B", ".A|B"), ("^X", "^X"), (r"\.*X", r"\.*X"), (".+", ".+")):
            self.assertEqual(cli._searchable(pattern), want, pattern)
        code = ("import time\nfor i in range(10): print('a' * 30000, flush=True)\n"
                "print('x START', flush=True); time.sleep(1); print('END', flush=True)\n")
        for pattern in (".*START", ".+START", ".*.*START"):
            p, r = self.run_j(["--no-cpu", "--from", pattern, "--until", "END"], [sys.executable, "-c", code])
            self.assertLess(r["duration_s"], 2.0, pattern)
            self.assertTrue(0.9 < r["window_s"] < 1.3, (pattern, r["window_s"]))

    def test_slow_matching_is_said_instead_of_held_back_or_pause_notes(self):
        import argparse
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class M:
            name, kind, energy_j = "GPU 0", "gpu", 50.0
        w = cli.Window.__new__(cli.Window)
        w.lock, w.frozen, w.first_out, w.ttys, w.behind = threading.Lock(), {"t": 4.0, "j": {"GPU 0": 40}}, {1: 4.0}, {}, False
        w.begin = {"t": 4.0, "j": {"GPU 0": 40}, "line": "S", "late": True, "src": 1, "after_pause": True}
        w.end = {"t": 4.0, "j": {"GPU 0": 40}, "line": "E", "late": True, "src": 1, "after_pause": True}
        w.alt_end, w.match_s = None, 2.0

        class S:
            t0, t1 = 0.0, 4.0
        notes = cli._window_report(argparse.Namespace(window_from="S", window_until="E"), w, S(), [M()], {})[1]
        self.assertTrue([n for n in notes if "Matching --from/--until took 2.0 s" in n], notes)
        self.assertFalse([n for n in notes if "held its output" in n or "10 ms" in n or "16 MB" in n or "fell behind" in n])

    # ---- found in review round 16
    def test_a_full_queue_is_a_slow_reader_only_when_writes_wait(self):
        """A queue that fills because the command prints faster than joules writes (into a file, /dev/null) is not
        a slow reader; one whose writes wait on a pipe nobody reads is. Only the second may set the pause note."""
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        with open(os.path.join(self.d, "f"), "wb") as f:
            out = cli._Out(f)
            for _ in range(40):
                out.put(b"x" * 65536)
            out.wait_empty(10)
            self.assertFalse(out.slow_reader())
            out.close()
        with open(os.devnull, "wb") as f:
            o2 = cli._Out(f)
            o2._fd()
            self.assertEqual(o2.kind, "file")   # (/dev/null takes 64 KB at a time, like a file)
            o2.close()
        r, w = os.pipe()
        try:
            out = cli._Out(os.fdopen(w, "wb", closefd=False))
            out.put(b"x" * 300_000)   # more than the pipe holds: the write waits
            time.sleep(0.3)
            self.assertTrue(out.slow_reader())
            out.take()
        finally:
            os.close(r)
            os.close(w)

    def test_a_mark_set_while_the_command_exits_is_the_end_of_the_run(self):
        """The exit is recorded while a search is under way: the mark it then sets is late, at the exit."""
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class M:
            name, kind, energy_j = "GPU 0", "gpu", 50.0
        w = None

        class Clock:
            def snapshot(self):
                w.freeze({"t": 3.0, "j": {"GPU 0": 30.0}})   # the command exits during this read
                return {"t": 3.5, "j": {"GPU 0": 35.0}}
        w = cli.Window(Clock(), 0.0, [M()], "START", "END")
        w.feed(["START"], [1])
        self.assertEqual(w.begin["t"], 3.0)
        self.assertTrue(w.begin["late"])

    # ---- found in review round 17
    def test_much_output_with_a_plain_word_is_not_slow_matching(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "START", "END")
        block = ("x" * 99 + "\n") * 650   # 64 KB, as one read brings
        for _ in range(3000):              # ~200 MB of output before the markers
            w.feed([block], [1])
        self.assertLess(w.match_s, 0.5)
        w2 = cli.Window(Clock(), 0.0, [], "a*START", "END")
        w2.feed(["a" * 30000 + " START"], [1])   # slow in itself (a*START drops nothing here: it is not .*)
        self.assertGreater(w2.match_s, 0.0)

    def test_a_socket_output_hand_over_repeats_nothing(self):
        """joules' stdout is a socket (Node harnesses, the systemd journal) whose reader is slow: a Ctrl-C hand-over
        passes on exactly what the command wrote."""
        import socket
        a, b = socket.socketpair()
        a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        code = ("import sys; o = sys.stdout.buffer\nfor i in range(20000): o.write(b'%07d ' % i + b'z' * 92 + b'\\n')\n"
                "o.write(b'START\\nEND\\n'); o.flush()\n")
        want = b"".join(b"%07d " % i + b"z" * 92 + b"\n" for i in range(20000)) + b"START\nEND\n"
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "START", "--until", "END", "--",
                              sys.executable, "-c", code], stdout=a.fileno(), stderr=open(os.path.join(self.d, "err"), "wb"),
                             stdin=subprocess.DEVNULL, env=self.env())
        a.close()
        time.sleep(3)
        q.send_signal(signal.SIGINT)
        q.wait(timeout=20)
        got = b""
        b.settimeout(5)
        try:
            while len(got) < len(want) + 1:
                d = b.recv(65536)
                if not d:
                    break
                got += d
        except socket.timeout:
            pass
        b.close()
        self.assertEqual(got, want)

    def test_where_output_goes_decides_how_it_is_written(self):
        import socket
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        for path, kind in ((os.devnull, "file"), ("/dev/zero", "tty"), (os.path.join(self.d, "f"), "file")):
            if not os.path.exists(path) and path != os.path.join(self.d, "f"):
                continue
            with open(path, "wb") as f:
                o = cli._Out.__new__(cli._Out)
                o.dst = f
                o._fd()
                self.assertEqual(o.kind, kind, path)
        a, b = socket.socketpair()
        try:
            with os.fdopen(a.fileno(), "wb", closefd=False) as f:
                o = cli._Out.__new__(cli._Out)
                o.dst = f
                o._fd()
                self.assertEqual(o.kind, "socket")
                o.sock.close()
        finally:
            a.close()
            b.close()

    def test_windows_keeps_the_pause_while_another_stream_is_behind(self):
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Empty:
            gone, size = False, 0
            def full(self): return False
            def put(self, c): pass

        class Win:
            first_out = {}
            paused = True
            def pending(self): return True
        win = Win()
        win.paused_streams = {99}   # another stream is still behind
        r, w = os.pipe()
        os.close(w)   # (no output: the reader ends at once)
        try:
            cli._pump(r, None, win, {"stop": threading.Event()}, None, Empty())
        finally:
            os.close(r)
        self.assertTrue(win.paused)

    # ---- found in review round 18
    def test_a_slow_pattern_in_small_reads_is_counted(self):
        """A terminal gives joules about 4 KB at a time: a slow pattern must add up there too."""
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "START", r"\w*\dDONE")
        w.begin = {"t": 0.0, "j": {}, "line": "START"}
        for _ in range(30):
            w.feed(["a" * 5000 + " DONE"], [1])   # (it has the word, so the pattern runs, slowly, and fails)
        self.assertGreater(w.match_s, 0.2)

    def test_a_socket_output_on_a_high_fd_number(self):
        import resource
        import socket
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard < 2100:
            self.skipTest("cannot open that many files here")
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2100), hard))
        self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))
        a, b = socket.socketpair()
        high = os.dup2(a.fileno(), 2000)
        a.close()
        filler = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]   # (a parent that leaks fds: joules' own
        self.addCleanup(lambda: [os.close(x) for x in filler])            #  copy of the socket lands above 1024)
        try:
            with os.fdopen(high, "wb", closefd=False) as f:
                out = cli._Out(f)
                out.put(b"hello\n")
                out.close()
                out.thread.join(5)
                self.assertFalse(out.gone)
            b.settimeout(2)
            self.assertEqual(b.recv(100), b"hello\n")
        finally:
            os.close(high)
            b.close()

    def test_a_datagram_socket_output_keeps_its_messages(self):
        import socket
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        prog = "import os\nfor i in range(300): os.write(1, b'%05d ' % i + b'x' * 94 + b'\\n')\n"
        err = os.path.join(self.d, "err")
        with open(err, "wb") as e:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "A", "--", sys.executable,
                                  "-c", prog], stdout=a, stderr=e, env=self.env())
            a.close()
            got = []
            b.settimeout(3)
            try:
                while True:
                    got.append(b.recv(1 << 17))
            except socket.timeout:
                pass
            q.wait(timeout=20)
        b.close()
        self.assertEqual(len(got), 300)
        self.assertEqual(got[7], b"00007 " + b"x" * 94 + b"\n")
        self.assertIn("message socket", Path(err).read_text())

    def test_a_marker_from_a_process_left_running_is_said_to_come_after_the_exit(self):
        p, r = self.run_j(["--no-cpu", "--from", "A", "--until", "B"],
                          ["sh", "-c", "sleep 0.5; (sleep 0.3; echo A; sleep 0.5; echo B) & echo started"], timeout=60)
        self.assertTrue(any("printed after the command exited" in n for n in r["notes"]), r["notes"])
        self.assertFalse([n for n in r["notes"] if "held its output" in n])

    # ---- found in review round 19
    def test_a_slow_pattern_in_terminal_sized_reads_adds_up(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], r"(x+x+)+y", None)
        block = ("x" * 14 + " y\n") * 240   # ~4 KB, as a terminal gives; each line has the y, so the pattern runs
        for _ in range(20):
            w.feed([block], [1])
        self.assertGreater(w.match_s, 0.2)

    def test_held_output_on_a_terminal_is_not_called_a_process_left_running(self):
        """A pseudo-terminal reports only part of what waits in it: what the command wrote before exiting must not
        be taken for output of a process it left running."""
        code = ("import os, time\ntime.sleep(0.3)\n"
                "os.write(1, (b'a' * 1000 + b'\\n') * 300 + b'START\\nEND\\n')\n")
        shown = self._on_own_terminal(["--from", r"\w*START", "--until", r"\w*END", "--", sys.executable, "-c", code],
                                      cols=200)
        self.assertNotIn(b"printed after the command exited", shown)

    def test_a_marker_on_a_last_line_without_a_line_end_says_so(self):
        p, r = self.run_j(["--no-cpu", "--from", "START"], ["bash", "-c", 'sleep 0.3; printf "x\\nSTART"; sleep 0.5'])
        self.assertTrue(any("had no line end" in n for n in r["notes"]), r["notes"])

    # ---- found in review round 20
    def test_a_server_writing_to_a_terminal_does_not_keep_joules_waiting(self):
        import fcntl
        import pty
        import termios
        import threading
        m, sl = pty.openpty()

        def own_session():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        cmd = ("(for i in $(seq 1 300); do echo 'server: request served'; sleep 0.05; done) & "
               "echo START; sleep 1; echo bye")
        t0 = time.monotonic()
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "START", "--", "bash", "-c", cmd],
                             stdin=sl, stdout=sl, stderr=sl, env=self.env(), preexec_fn=own_session)
        os.close(sl)
        shown = []

        def drain():
            while True:
                b = select_read(m)
                if b is None:
                    return
                shown.append(b)
        t = threading.Thread(target=drain, daemon=True)
        t.start()
        try:
            q.wait(timeout=30)
        finally:
            if q.poll() is None:
                q.kill()
        took = time.monotonic() - t0
        self.assertLess(took, 12, took)
        time.sleep(0.5)
        os.close(m)
        self.assertIn(b"still open", b"".join(shown))

    def test_the_pre_search_is_off_where_it_could_miss_a_line(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        for pattern, text in ((r"START\s*+$", "START\n  x"), ("START[^#]*+$", "START foo\nbar#"),
                              ("(?-m:START$)", "START\nother"), (r"(?s)START(?>.*)\b", "START\n-")):
            w = cli.Window(Clock(), 0.0, [], pattern, None)
            self.assertTrue(w.feed([text], [1]), pattern)

    def test_a_coarse_cpu_clock_does_not_add_up_to_a_false_note(self):
        """Windows' thread CPU clock moves in 15.6 ms steps: a plain word must still not add up."""
        from unittest import mock
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        real = time.thread_time

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "NEVERSEEN", None)
        w.group = 10 * 0.015625   # (what _cpu_step measures there)
        block = ("2026-11-22 12:00:00 INFO step done loss=0.123 " + "x" * 50 + "\n") * 40   # ~4 KB
        with mock.patch.object(cli.time, "thread_time", lambda: (real() // 0.015625) * 0.015625):
            for _ in range(75000):   # ~300 MB
                w.feed([block], [1])
        self.assertLess(w.match_s, 0.5)

    # ---- found in review round 21
    def test_a_slow_search_does_not_hide_behind_cheap_ones_before_it(self):
        """Many cheap reads first (a long benchmark log): their unused budget must not cancel a slow search later."""
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "(a|aa)+c", None)
        for i in range(15000):
            w.feed(["tik %d c" % i], [1])       # (cheap: each has the c, and fails at once)
        w.feed(["a" * 28 + " c"], [1])   # ~0.1 s or more of backtracking
        self.assertGreater(w.match_s, 0.1)

    def test_a_marker_after_the_exit_on_a_terminal_kept_busy_by_a_process(self):
        """A process left running writes to the terminal every 50 ms and prints the --from line after the exit."""
        script = os.path.join(self.d, "orph.sh")
        Path(script).write_text("( trap '' HUP; i=0; while [ $i -lt 60 ]; do echo tick; sleep 0.05; i=$((i+1)); "
                                "[ $i = 20 ] && echo A; done ) &\necho x; sleep .5\n")
        shown = self._on_own_terminal(["--from", "^A", "--", "bash", script])
        self.assertIn(b"printed after the command exited", shown)
        self.assertNotIn(b"held its output", shown)

    # ---- found in review round 22
    def test_a_marker_after_the_exit_from_a_process_writing_big_lines_to_a_terminal(self):
        script = os.path.join(self.d, "orph2.sh")
        Path(script).write_text("( trap '' HUP; line=$(head -c 2000 /dev/zero | tr '\\0' y); i=0; "
                                "while [ $i -lt 100 ]; do echo \"$line\"; sleep 0.02; i=$((i+1)); "
                                "[ $i = 60 ] && echo START; done ) &\necho x; sleep .5\n")
        shown = self._on_own_terminal(["--from", "^START", "--", "bash", script], cols=200)
        self.assertIn(b"printed after the command exited", shown)
        self.assertNotIn(b"held its output", shown)

    def test_slow_line_by_line_matching_gets_advice_that_fits(self):
        import argparse
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}

        class S:
            t0, t1 = 0.0, 4.0
        for pattern, says in (("[0-9]+", "no plain text"), (r"\w*NEVER", "no leading repeat")):
            w = cli.Window(Clock(), 0.0, [], pattern, None)
            w.match_s = 2.0
            w.freeze({"t": 4.0, "j": {}})
            notes = cli._window_report(argparse.Namespace(window_from=pattern, window_until=None), w, S(), [], {})[1]
            self.assertTrue([n for n in notes if says in n], (pattern, notes))

    def _win_reader(self, before, at_exit, after, slow=0.0):
        """Windows' reader (_pump) on a pipe, PeekNamedPipe stood in for by FIONREAD: write `before`, let the reader
        start on it, write `at_exit`, record the exit (as cmd_run does), then write `after`. Returns the --from mark."""
        import threading
        from unittest import mock
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}

        class Out:
            gone, size = False, 0
            def full(self): return False
            def put(self, c): pass
        w = cli.Window(Clock(), 0.0, [], "START", None)
        if slow:
            real = w._feed
            w._feed = lambda *a, **k: (time.sleep(slow), real(*a, **k))[1]
        r, wr = os.pipe()
        state = {"stop": threading.Event(), "exited": threading.Event(), "fds": [r], "owed": {}, "blocked": {},
                 "reads": {}, "wlock": threading.Lock()}
        with mock.patch.object(cli, "_peek_pipe", cli._pending):
            t = threading.Thread(target=cli._pump, args=(r, None, w, state, None, Out()), daemon=True)
            t.start()
            if before:
                os.write(wr, before)
            time.sleep(0.1)
            if at_exit:
                os.write(wr, at_exit)
            w.freeze({"t": 1.0, "j": {}})
            state["exited"].set()
            with state["wlock"]:   # (as cmd_run does at the exit)
                state["reads_at_exit"] = dict(state["reads"])
                state["exit_t"] = time.monotonic()
            cli._win_caught_up(state, w)
            time.sleep(0.06)
            cli._win_caught_up(state, w)
            time.sleep(0.14)
            if after:
                os.write(wr, after)
            time.sleep(0.3 + 2 * slow)
            os.close(wr)
            t.join(5)
        os.close(r)
        return w.begin

    def test_windows_says_a_marker_came_after_the_exit(self):
        """Windows: what waits in the pipe at the exit is read first; a --from line after that came from a process
        left running. Simulated with a pipe here."""
        b = self._win_reader(b"held line\n", b"", b"START\n")
        self.assertTrue(b and b["after_exit"], b)

    def test_windows_does_not_take_a_line_written_before_the_exit_for_one_after_it(self):
        """The reader is still busy with a chunk read before the exit when the exit comes; START was written
        before the exit too."""
        b = self._win_reader(b"x" * 10000 + b"\n", b"START\n", b"", slow=0.3)
        self.assertTrue(b and not b["after_exit"], b)

    # ---- found in review round 23
    def test_slow_matching_advice_names_the_pattern_that_was_slow(self):
        import argparse
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}

        class S:
            t0, t1 = 0.0, 4.0
        w = cli.Window(Clock(), 0.0, [], "(?<!x)START", "(a+)+b")
        w.feed(["START"], [1])
        for _ in range(6):
            w.feed(["a" * 20], [1])
        w.freeze({"t": 4.0, "j": {}})
        notes = cli._window_report(argparse.Namespace(window_from="(?<!x)START", window_until="(a+)+b"), w, S(), [], {})[1]
        slow = [n for n in notes if "Matching" in n]
        if slow:   # (on a very fast machine it may stay under 1 s)
            self.assertIn("Make --until simpler", slow[0])
            self.assertNotIn("--from has a lookaround", slow[0])
        self.assertGreater(w.match_by.get("--until", 0), w.match_by.get("--from", 0))

    def test_the_quiet_check_works_on_high_fd_numbers(self):
        import resource
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard < 2100:
            self.skipTest("cannot open that many files here")
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2100), hard))
        self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))
        r, w = os.pipe()
        high = os.dup2(r, 2000)
        try:
            self.assertFalse(cli._readable(high, 0.005))   # nothing there
            os.write(w, b"x")
            self.assertTrue(cli._readable(high, 0.005))
        finally:
            for f in (r, w, high):
                os.close(f)

    # ---- found in review round 24
    def test_windows_a_line_written_at_the_exit_is_not_taken_for_one_after_it(self):
        """The reader's read returns the --from line written just before the exit, at the same moment as the exit
        is recorded: that read was not 'still waiting with nothing'."""
        for _ in range(5):
            b = self._win_reader(b"", b"START\n", b"")
            self.assertTrue(b and not b["after_exit"], b)

    def test_slow_matching_is_charged_to_the_pattern_that_was_slow_in_one_read(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "START", "(x+x+)+y")
        w.feed(["START\n" + "x" * 22 + "\nEND"], [1])   # the --from line and slow lines in one read
        self.assertGreater(w.match_by.get("--until", 0), 10 * w.match_by.get("--from", 0) + 0.01)

    def test_a_line_from_a_process_left_running_just_after_the_exit_on_a_terminal(self):
        shown = self._on_own_terminal(["--from", "^GO", "--", "bash", "-c", "(sleep 0.45; echo GO) & sleep 0.3; echo hi"])
        self.assertIn(b"printed after the command exited", shown)

    # ---- found in review round 25
    def test_windows_caught_up_waits_for_the_exit_to_be_recorded(self):
        import threading
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class W:
            caught_up = False
        state = {"exited": threading.Event(), "wlock": threading.Lock(), "fds": [5], "owed": {}, "blocked": {5: True},
                 "reads": {}}
        state["exited"].set()        # flagged, but the exit time and read counts not stored yet
        w = W()
        cli._win_caught_up(state, w)
        self.assertFalse(w.caught_up)
        with state["wlock"]:
            state["reads_at_exit"], state["exit_t"] = {}, time.monotonic() - 0.1
        cli._win_caught_up(state, w)
        self.assertTrue(w.caught_up)

    # ---- found in review round 26
    def test_a_marker_on_an_unfinished_line_still_counts_when_the_stream_is_handed_over(self):
        p, r = self.run_j(["--no-cpu", "--from", "START", "--until", "END"],
                          ["bash", "-c", "echo START; sleep 0.5; printf END; sleep 4 &"], timeout=60)
        self.assertTrue(r["window"]["found_until"], r["notes"])
        self.assertTrue(any("had no line end" in n for n in r["notes"]), r["notes"])
        p, r = self.run_j(["--no-cpu", "--from", "START"], ["bash", "-c", "sleep 0.3; printf START; sleep 4 &"], timeout=60)
        self.assertTrue(r["window"]["found_from"], r["notes"])
        self.assertFalse([n for n in r["notes"] if "printed after the command exited" in n], r["notes"])
        self.assertFalse([n for n in r["notes"] if "held its output" in n], r["notes"])   # (its own note says why)

    def test_each_search_keeps_its_own_cpu_times(self):
        """Windows feeds from two threads at once: each call's per-pattern times must be its own."""
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "START", None)
        seen = []
        real = w._feed

        def spy(*a, **k):
            seen.append(a[-1] if a and isinstance(a[-1], dict) else k.get("sec"))
            return real(*a, **k)
        w._feed = spy
        w.feed(["x"], [1])
        w.feed(["y"], [1])
        self.assertIsInstance(seen[0], dict)
        self.assertIsNot(seen[0], seen[1])
        self.assertFalse(hasattr(w, "_sec"))

    # ---- found in review round 27
    def test_a_hand_over_during_a_slow_search_loses_nothing(self):
        """Ctrl-C while joules is still busy with a slow pattern: the search stops between lines, and all the
        command's output (some still in the pipe) is passed on."""
        code = ("import sys, time\nsys.stdout.write(('a' * 24 + 'b\\n') * 250); sys.stdout.flush(); time.sleep(0.3)\n"
                "sys.stdout.write(''.join('TAIL %d\\n' % i for i in range(100))); sys.stdout.flush()\n")
        out = os.path.join(self.d, "out")
        with open(out, "wb") as f:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "(?=a)(a|aa)+$", "--",
                                  sys.executable, "-c", code], stdout=f, stderr=subprocess.DEVNULL, env=self.env(),
                                 preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
            time.sleep(1.5)
            q.send_signal(signal.SIGINT)
            q.wait(timeout=30)
        time.sleep(2)
        self.assertEqual(Path(out).read_text().count("TAIL "), 100)

    def test_a_search_cut_short_by_a_hand_over_says_so(self):
        """The --from line was read but not yet searched when Ctrl-C handed over: not "never appeared"."""
        code = "import sys\nprint('A')\nfor i in range(300): print('x' * 18)\nprint('B'); sys.stdout.flush()\n"
        err = os.path.join(self.d, "err")
        with open(err, "wb") as e:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "(?=B)B|(x+x+)+y", "--",
                                  sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=e, env=self.env(),
                                 preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
            time.sleep(1.5)
            q.send_signal(signal.SIGINT)
            q.wait(timeout=30)
        text = Path(err).read_text()
        self.assertNotIn("never appeared", text)
        self.assertIn("stopped looking", text)

    def test_the_quick_search_in_pieces_handles_empty_and_split_blocks(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "zz", None)
        self.assertFalse(w.feed([""], [1]))                       # (an empty block, no match: must not loop)
        self.assertTrue(cli.Window(Clock(), 0.0, [], "^$", None).feed([""], [1]))
        w = cli.Window(Clock(), 0.0, [], "needle", None)
        self.assertTrue(w.feed([("x" * 79 + "\n") * 3000 + "a needle\n"], [1]))   # found in a later piece

    def test_a_slow_pre_search_does_not_stretch_the_run(self):
        """One search over a whole read holds up every other thread: a slow one makes joules go line by line."""
        code = "import sys\nprint('A')\nfor i in range(300): print('x' * 18)\nprint('B')\n"
        p, r = self.run_j(["--no-cpu", "--from", "A", "--until", "(x+x+)+y|B"], [sys.executable, "-c", code], timeout=120)
        self.assertLess(r["duration_s"], 2.0, r["duration_s"])

    def test_colour_codes_are_not_counted_as_slow_matching(self):
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "NEVERSEEN", None)
        block = ("\x1b[m\x1b[1m.\x1b[0m" * 6 + "\n") * 2000   # ~64 KB of lines thick with escape codes
        for _ in range(600):                                     # ~40 MB
            w.feed([block], [1])
        self.assertLess(w.match_s, 0.5)

    # ---- found in review round 29
    def test_the_text_every_match_must_contain(self):
        """Read from the parsed pattern; the pattern itself then only runs on single lines that have it."""
        import re as re_
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli
        for pattern, want in (("START", (["START"], False)), (r"(\w+\s?)+X", (["X"], False)),
                              (r"^(\w+\s?)+done$", (["done"], False)), ("(?i)never", (["never"], True)),
                              ("a|bc", (["a", "bc"], False)), ("a|.*", None), ("(ab)?cd", (["cd"], False)),
                              ("(?i:x)y", (["y"], False)), (r">>> done\.", ([">>> done."], False)),
                              ("x{0}y", (["y"], False)), ("[0-9]+", None)):
            self.assertEqual(cli._must_contain(re_.compile(pattern)), want, pattern)

        class Clock:
            def snapshot(self): return {"t": 1.0, "j": {}}
        w = cli.Window(Clock(), 0.0, [], "needle", None)
        self.assertEqual(w._lines("aaa\na needle\nccc", w.start_re), [None, "a needle", None])
        self.assertIsNone(w._lines("aaa\nbbb", w.start_re))
        self.assertTrue(cli.Window(Clock(), 0.0, [], "(?i)NEVER", None).feed(["x\nsay never\ny"], [1]))

    def test_a_pattern_fast_line_by_line_is_fast_over_many_lines(self):
        """Patterns that can match a line end (\\s, [^...]) with a nested repeat are only ever run on one line."""
        p, r = self.run_j(["--no-cpu", "--from", "START", "--until", r"(\w+\s?)+X"],
                          ["bash", "-c", "echo START; for i in $(seq 20); do echo 'ab cd ef gh ij kl mn'; done; echo done"],
                          timeout=30)
        self.assertLess(r["duration_s"], 2.0)

    def test_a_hand_over_with_output_still_unread_says_joules_stopped_looking(self):
        """A slow reader on joules' output, the --until line printed but not yet read when Ctrl-C hands over."""
        code = ("import sys\nprint('START', flush=True)\nsys.stdout.write(('x' * 99 + '\\n') * 168400)\n"
                "print('END', flush=True)\n")
        err = os.path.join(self.d, "err")
        with open(err, "wb") as e:
            q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--from", "START", "--until", "END",
                                  "--", sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=e, env=self.env(),
                                 preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
            rd = subprocess.Popen(["sh", "-c", "sleep 6; wc -c"], stdin=q.stdout, stdout=subprocess.DEVNULL)
            q.stdout.close()
            time.sleep(3)
            q.send_signal(signal.SIGINT)
            q.wait(timeout=30)
            rd.wait(timeout=30)
        text = Path(err).read_text()
        self.assertNotIn("never appeared", text)

    def test_window_with_an_energy_counter(self):
        """The same window on a GPU with an energy counter (the stand-in H100 draws 650 W)."""
        p, r, _ = self.joules(["--gpus", "0", "--from", r">>> running trie", "--until", r">>> done\."], lib=self.server)
        self.assertEqual([x["method"] for x in r["devices"]], ["nvml counter"])
        self.assertAlmostEqual(r["window_energy_j"], 650.0 * r["window_s"], delta=0.01 * r["window_energy_j"])

    def test_a_bad_regex_is_refused_before_running(self):
        p, r, _ = self.joules(["--from", "([unclosed"])
        self.assertEqual(p.returncode, 2)
        self.assertIn("not a valid regular expression", p.stderr)
        self.assertIsNone(r)

    def test_doctor_names_power_readings_on_a_laptop_gpu(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1", "JOULES_TELEMETRY": "0",
               "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power, "NO_COLOR": "1"}
        p = subprocess.run([sys.executable, "-m", "joules", "doctor", "--offline"], capture_output=True, text=True, env=env, timeout=60)
        self.assertIn("power readings (no energy counter): sampled every 0.1 s", p.stdout + p.stderr)


class LateNote(unittest.TestCase):
    """The "markers only read at exit" note: given when output was held back, not for a last line read late."""

    def report(self, begin, end, exit_t, first_out, until="done", tty=False, which="only read when", behind=False):
        import argparse
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class M:
            name, kind, energy_j = "GPU 0", "gpu", 10.0
        w = cli.Window.__new__(cli.Window)
        import threading
        w.lock, w.frozen, w.first_out = threading.Lock(), {"t": exit_t, "j": {"GPU 0": 10.0}}, {1: first_out}
        if end is None:
            w.end = None
        w.ttys, w.behind = {1: tty}, behind
        w.begin = {"t": begin[0], "j": {"GPU 0": 1.0}, "line": "start", "late": begin[1], "src": 1}
        w.end = {"t": end[0], "j": {"GPU 0": 2.0}, "line": "done", "late": end[1]} if end else None

        class S:
            t0, t1 = 0.0, exit_t
        a = argparse.Namespace(window_from="start", window_until=until)
        return [n for n in cli._window_report(a, w, S(), [M()], {})[1] if which in n]

    def test_a_last_line_read_just_after_exit_gets_no_note(self):
        self.assertEqual(self.report(begin=(1.0, False), end=(1.51, True), exit_t=1.5, first_out=0.9), [])

    def test_output_held_until_exit_gets_the_note(self):
        self.assertTrue(self.report(begin=(4.0, False), end=(4.0, False), exit_t=4.0, first_out=4.0))
        self.assertTrue(self.report(begin=(4.01, True), end=(4.01, True), exit_t=4.0, first_out=4.01))

    def test_a_quick_pair_after_earlier_output_gets_a_softer_note(self):
        """Earlier output on the same stream (a log line on stderr with 2>&1) does not prove the markers were on time."""
        self.assertEqual(self.report(begin=(2.0, False), end=(2.0, False), exit_t=2.05, first_out=0.3), [])
        self.assertTrue(self.report(begin=(2.0, False), end=(2.0, False), exit_t=2.05, first_out=0.3, which="under 10 ms"))
        self.assertTrue(self.report(begin=(1.0, False), end=(1.001, False), exit_t=3.0, first_out=0.3, which="under 10 ms"))
        self.assertEqual(self.report(begin=(2.0, False), end=(2.0, False), exit_t=2.05, first_out=0.3, tty=True,
                                     which="under 10 ms"), [])   # on a terminal programs do not hold their output

    def test_falling_behind_after_early_markers_is_not_called_late(self):
        """Markers read at the start, then joules fell behind a slow reader: the markers were not late."""
        self.assertEqual(self.report(begin=(0.01, False), end=(0.011, False), exit_t=3.0, first_out=0.01, behind=True,
                                     which="fell behind"), [])
        self.assertTrue(self.report(begin=(0.01, False), end=(0.011, False), exit_t=3.0, first_out=0.01, behind=True,
                                    which="under 10 ms"))

    def test_from_alone_on_the_last_line_gets_no_note(self):
        """With --from alone the window runs to the exit, so a --from on the last line is a short window, not held output."""
        self.assertEqual(self.report(begin=(0.5, False), end=None, exit_t=0.501, first_out=0.0, until=None, which=""), [])

    def test_a_very_short_run_gets_no_note(self):
        self.assertEqual(self.report(begin=(0.05, True), end=(0.05, True), exit_t=0.05, first_out=0.05, which=""), [])

    def test_held_output_on_a_terminal_gets_no_terminal_advice(self):
        n = self.report(begin=(4.0, False), end=(4.0, False), exit_t=4.0, first_out=4.0, tty=True)
        self.assertTrue(n and "terminal" not in n[0], n)


if __name__ == "__main__":
    unittest.main()
