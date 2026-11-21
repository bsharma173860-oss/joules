"""joules run --from / --until: energy of the part of a run between two output lines, e.g. an agent benchmark after a
model server has loaded. Tested end to end on a stand-in laptop GPU that reports power only (no energy counter),
with a script shaped like a typical benchmark runner: a server started in the background and killed by a trap, a
model-loading phase, then the benchmark between two marker lines, each phase at its own power."""
import json
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

    def test_a_background_process_left_running_does_not_hang_joules(self):
        Path(self.script).write_text("#!/bin/bash\n(sleep 30) &\necho '>>> running'\nsleep 0.5\necho '>>> done'\n")
        p, r, took = self.joules(["--from", r">>> running", "--until", r">>> done"], timeout=25)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertLess(took, 15)
        self.assertTrue(r["window_s"] > 0.3)

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
            #!/bin/bash
            trap 'echo ">>> done"; exit 7' INT
            echo 45000 > "$FAKE_NVML_POWER_FILE"; sleep 0.5
            echo ">>> running"
            sleep 20 & wait $!
            """))
        out = os.path.join(self.d, "r.json")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1", "JOULES_TELEMETRY": "0",
               "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power}
        q = subprocess.Popen([sys.executable, "-m", "joules", "run", "--no-cpu", "--out", out, "--from", ">>> running",
                              "--until", ">>> done", "--", "bash", self.script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, start_new_session=True)
        for _ in range(100):                                             # wait for the window to open
            time.sleep(0.1)
            if os.path.exists(self.power) and Path(self.power).read_text().strip() == "45000":
                break
        time.sleep(1.5)
        os.killpg(q.pid, signal.SIGINT)                                  # Ctrl-C in a terminal reaches the whole group
        q.communicate(timeout=30)
        r = json.loads(Path(out).read_text())
        self.assertEqual(q.returncode, 7)
        self.assertEqual(r["exit_code"], 7)
        self.assertTrue(r["window"]["found_until"])
        self.assertAlmostEqual(r["window_energy_j"], 45.0 * r["window_s"], delta=1e-6 * r["window_energy_j"])

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


if __name__ == "__main__":
    unittest.main()
