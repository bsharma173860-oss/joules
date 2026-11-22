"""joules run --from / --until: energy of the part of a run between two output lines, e.g. an agent benchmark after a
model server has loaded. Tested end to end on a stand-in laptop GPU that reports power only (no energy counter),
with a script shaped like a typical benchmark runner: a server started in the background and killed by a trap, a
model-loading phase, then the benchmark between two marker lines, each phase at its own power."""
import contextlib
import json
import shlex
import os
import shutil
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
        streams = {"stdout": subprocess.PIPE, "stderr": kw.pop("stderr", subprocess.PIPE)}
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
        code = ("import os, signal, time\nsignal.signal(signal.SIGWINCH, lambda *a: print('size', *os.get_terminal_size(1), flush=True))\n"
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
        os.environ.pop("PYTHONUNBUFFERED", None)
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
            first_out = None
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

    def report(self, begin, end, exit_t, first_out, until="done"):
        import argparse
        sys.path.insert(0, str(ROOT / "src"))
        from joules import cli

        class M:
            name, kind, energy_j = "GPU 0", "gpu", 10.0
        w = cli.Window.__new__(cli.Window)
        import threading
        w.lock, w.frozen, w.first_out = threading.Lock(), {"t": exit_t, "j": {"GPU 0": 10.0}}, first_out
        w.begin = {"t": begin[0], "j": {"GPU 0": 1.0}, "line": "start", "late": begin[1]}
        w.end = {"t": end[0], "j": {"GPU 0": 2.0}, "line": "done", "late": end[1]}

        class S:
            t0, t1 = 0.0, exit_t
        a = argparse.Namespace(window_from="start", window_until=until)
        return [n for n in cli._window_report(a, w, S(), [M()], {})[1] if "only read when" in n]

    def test_a_last_line_read_just_after_exit_gets_no_note(self):
        self.assertEqual(self.report(begin=(1.0, False), end=(1.51, True), exit_t=1.5, first_out=0.9), [])

    def test_output_held_until_exit_gets_the_note(self):
        self.assertTrue(self.report(begin=(4.0, False), end=(4.0, False), exit_t=4.0, first_out=4.0))
        self.assertTrue(self.report(begin=(4.01, True), end=(4.01, True), exit_t=4.0, first_out=4.01))

    def test_a_quick_pair_after_earlier_output_gets_no_note(self):
        self.assertEqual(self.report(begin=(2.0, False), end=(2.0, False), exit_t=2.05, first_out=0.3), [])


if __name__ == "__main__":
    unittest.main()
