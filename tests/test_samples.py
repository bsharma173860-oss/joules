"""joules run --samples and joules window: each device's energy over time, and the energy between two times."""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from joules import samples  # noqa: E402
from joules.meters import PowerMeter  # noqa: E402

LAPTOP = "GPU 0 NVIDIA GeForce RTX 3050 Laptop GPU"


class Window(unittest.TestCase):
    rows = {"GPU 0": [(100.0, 0.0), (101.0, 50.0), (103.0, 250.0)]}   # 50 W for a second, then 100 W

    def test_edges_are_interpolated(self):
        r = samples.window(self.rows, 100.5, 102.0)
        self.assertAlmostEqual(r["devices"][0]["energy_j"], 125.0)   # 25 J + 100 J
        self.assertAlmostEqual(r["duration_s"], 1.5)
        r = samples.window(self.rows, 100.0, 103.0, {"GPU 0": 20.0})
        self.assertAlmostEqual(r["devices"][0]["energy_j"], 250.0)   # (the first and last samples exactly)
        self.assertAlmostEqual(r["devices"][0]["above_idle_j"], 190.0)
        self.assertAlmostEqual(r["totals"]["above_idle_j"], 190.0)

    def test_times_outside_the_samples_are_refused(self):
        for start, end in ((99.9, 101.0), (101.0, 103.1), (102.0, 101.0), (101.0, 101.0)):
            with self.assertRaises(ValueError):
                samples.window(self.rows, start, end)
        with self.assertRaises(ValueError):
            samples.window({}, 1.0, 2.0)

    def test_times(self):
        self.assertEqual(samples.parse_time("1795360794.25"), 1795360794.25)
        self.assertEqual(samples.parse_time("2026-11-22T14:03:07.250"), dt.datetime(2026, 11, 22, 14, 3, 7, 250000).timestamp())
        self.assertEqual(samples.parse_time("2026-11-22 14:03:07Z"), 1795356187.0)
        self.assertEqual(samples.parse_time("2026-11-22T14:03:07+01:00"), 1795356187.0 - 3600)
        with self.assertRaises(ValueError):
            samples.parse_time("yesterday")

    def test_reading_the_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, "s.csv")
        Path(p).write_text("unix_time,device,energy_j,power_w\n2,G,10,5\n1,G,0,5\n3,G,")   # (cut off mid-line)
        self.assertEqual(samples.read_samples(p), {"G": [(1.0, 0.0), (2.0, 10.0)]})
        Path(p).write_text("unix_time,device,energy_j,power_w\n1,G,x,5\n2,G,1,5\n")
        with self.assertRaises(ValueError):
            samples.read_samples(p)
        Path(p).write_text("time,joules\n1,2\n")
        with self.assertRaises(ValueError):
            samples.read_samples(p)


class Log(unittest.TestCase):
    def test_a_failed_reading_writes_no_row_and_the_next_spans_the_gap(self):
        class M(PowerMeter):
            kind, method, w = "gpu", "test", 40.0
            def read_w(self):
                if self.w is None:
                    raise OSError("no reading")
                return self.w
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p, notes, m = os.path.join(d, "s.csv"), [], M("GPU 0")
        log = samples.SampleLog(p, notes)
        m.start(10.0)
        log.write([m])
        m.poll(11.0)
        log.write([m])
        m.w = None
        with self.assertRaises(OSError):
            m.poll(12.0)
        log.write([m])   # (nothing new)
        m.w = 40.0
        m.poll(13.0)
        log.write([m])
        log.close()
        rows = samples.read_samples(p)["GPU 0"]
        self.assertEqual([round(t - log.wall0, 6) for t, _ in rows], [10.0, 11.0, 13.0])
        self.assertEqual([e for _, e in rows], [0.0, 40.0, 120.0])
        self.assertAlmostEqual(samples.window({"GPU 0": rows}, log.wall0 + 10.5, log.wall0 + 12.5)["totals"]["energy_j"], 80.0)
        self.assertEqual(notes, [])


@unittest.skipUnless(shutil.which("gcc") and os.name == "posix", "needs gcc (to build the stand-in NVIDIA library)")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.laptop, cls.server = os.path.join(cls.d, "laptop", "libnvidia-ml.so.1"), os.path.join(cls.d, "server", "libnvidia-ml.so.1")
        for lib, src in ((cls.laptop, "fake_laptop_nvml.c"), (cls.server, "fake_nvml.c")):
            os.makedirs(os.path.dirname(lib))
            subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", lib, str(ROOT / "tests" / "fake_nvml" / src)], check=True)
        cls.power = os.path.join(cls.d, "power")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def run_j(self, lib, script, *args, **env):
        Path(self.power).write_text("45000")
        out = os.path.join(self.d, "run.json")
        p = subprocess.run([sys.executable, "-m", "joules", "run", "--no-cpu", "--samples", os.path.join(self.d, "s.csv"),
                            "--out", out, *args, "--", "bash", "-c", script], capture_output=True, text=True, timeout=60,
                           cwd=self.d, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d,
                                            "JOULES_NO_PROMPT": "1", "JOULES_TELEMETRY": "0", "JOULES_NVML_LIB": lib,
                                            "FAKE_NVML_POWER_FILE": self.power, "NO_COLOR": "1", **env})
        self.assertEqual(p.returncode, 0, p.stderr)
        return p, json.loads(Path(out).read_text())

    def window(self, *args):
        p = subprocess.run([sys.executable, "-m", "joules", "window", os.path.join(self.d, "s.csv"), *args, "--json"],
                           capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def times(self):
        return [Path(self.d, f).read_text().strip() for f in ("t1", "t2")]

    STEP = "sleep 0.4; date +%s.%N > t1; sleep 1; date +%s.%N > t2; sleep 0.4"

    def test_power_only_gpu(self):
        p, run = self.run_j(self.laptop, self.STEP)
        t1, t2 = self.times()
        r = self.window("--start", t1, "--end", t2)
        self.assertAlmostEqual(r["totals"]["energy_j"], 45 * r["duration_s"], delta=0.01)   # 45 W, exactly
        rows = samples.read_samples(os.path.join(self.d, "s.csv"))[LAPTOP]
        self.assertAlmostEqual(rows[-1][1], run["devices"][0]["energy_j"], places=4)   # (the file ends with the run)
        self.assertAlmostEqual(rows[-1][0] - rows[0][0], run["duration_s"], delta=0.01)
        self.assertEqual(run["samples"], os.path.join(self.d, "s.csv"))

    def test_counter_gpu_and_a_power_only_one(self):
        p, run = self.run_j(self.server, self.STEP)
        t1, t2 = self.times()
        r = self.window("--start", t1, "--end", t2)
        by = {d["name"]: d["energy_j"] for d in r["devices"]}
        self.assertAlmostEqual(by["GPU 0 NVIDIA H100 80GB HBM3"], 650 * r["duration_s"], delta=0.65)   # counter, 1 mJ steps
        self.assertAlmostEqual(by["GPU 1 Tesla T4"], 300 * r["duration_s"], delta=0.01)

    def test_failed_reads_in_the_middle(self):
        p, run = self.run_j(self.laptop, self.STEP, FAKE_NVML_FAIL_EVERY="3")
        t1, t2 = self.times()
        r = self.window("--start", t1, "--end", t2)
        self.assertAlmostEqual(r["totals"]["energy_j"], 45 * r["duration_s"], delta=0.01)
        rows = samples.read_samples(os.path.join(self.d, "s.csv"))[LAPTOP]
        gaps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
        self.assertGreater(max(gaps), 0.15)   # (a failed read: no row, the next one spans the gap)

    def test_energy_above_idle(self):
        # idle 10 W while joules measures it, then the command raises the GPU to 45 W
        Path(self.power).write_text("10000")
        script = f"echo 45000 > {self.power}; " + self.STEP
        p = subprocess.run([sys.executable, "-m", "joules", "run", "--no-cpu", "--subtract-idle", "0.5", "--samples",
                            os.path.join(self.d, "s.csv"), "--out", os.path.join(self.d, "run.json"), "--", "bash", "-c",
                            script], capture_output=True, text=True, timeout=60, cwd=self.d,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1",
                                "JOULES_TELEMETRY": "0", "JOULES_NVML_LIB": self.laptop,
                                "FAKE_NVML_POWER_FILE": self.power, "NO_COLOR": "1"})
        self.assertEqual(p.returncode, 0, p.stderr)
        t1, t2 = self.times()
        r = self.window("--start", t1, "--end", t2, "--run", os.path.join(self.d, "run.json"))
        self.assertAlmostEqual(r["devices"][0]["idle_w"], 10.0)
        self.assertAlmostEqual(r["totals"]["above_idle_j"], 35 * r["duration_s"], delta=0.01)

    def test_clear_errors(self):
        self.run_j(self.laptop, "sleep 0.3")
        p = subprocess.run([sys.executable, "-m", "joules", "window", os.path.join(self.d, "s.csv"), "--start", "1",
                            "--end", "2"], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(p.returncode, 2)
        self.assertIn("is not inside that", p.stderr)
        p = subprocess.run([sys.executable, "-m", "joules", "run", "--samples", os.path.join(self.d, "no", "s.csv"), "--",
                            "true"], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": self.d, "JOULES_NO_PROMPT": "1",
                                "JOULES_TELEMETRY": "0", "JOULES_NVML_LIB": self.laptop, "FAKE_NVML_POWER_FILE": self.power})
        self.assertEqual(p.returncode, 2)
        self.assertIn("cannot write", p.stderr)

    def test_the_file_is_written_while_the_command_runs(self):
        script = "sleep 2.5; cp s.csv mid.csv; sleep 0.2"
        self.run_j(self.laptop, script)
        mid = Path(self.d, "mid.csv").read_text().splitlines()
        self.assertGreater(len(mid), 10)   # (flushed about every second: a crash keeps what came before)


if __name__ == "__main__":
    unittest.main()
