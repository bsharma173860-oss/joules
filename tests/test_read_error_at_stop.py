"""Sampler regression: a read error on the first or the last poll must not lose the whole run. The energy up
to the last good reading is kept and the result says what happened.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ["JOULES_NO_PROMPT"] = "1"

from joules import cli, meters  # noqa: E402
from joules.sampler import Sampler  # noqa: E402


class Flaky(meters.PowerMeter):
    """100 W; reads fail while self.broken is true."""
    kind, method = "gpu", "power sampling"
    def __init__(self, broken=False):
        super().__init__("GPU 0 test")
        self.broken = broken
    def read_w(self):
        if self.broken:
            raise OSError("device gone")
        return 100.0


class ReadErrorAtStop(unittest.TestCase):
    def test_failed_last_reading_keeps_energy_and_adds_a_note(self):
        m, notes = Flaky(), []
        with Sampler([m], 0.02, notes=notes) as s:
            time.sleep(0.3)
            m.broken = True
        self.assertAlmostEqual(m.energy_j, 100 * 0.3, delta=100 * 0.1)
        self.assertEqual(len(notes), 1)
        self.assertIn("last reading failed", notes[0])
        self.assertGreater(s.duration_s, 0)

    def test_failed_first_reading_is_retried(self):
        m, notes = Flaky(broken=True), []
        with Sampler([m], 0.02, notes=notes):
            time.sleep(0.1)
            m.broken = False
            time.sleep(0.3)
        self.assertAlmostEqual(m.energy_j, 100 * 0.3, delta=100 * 0.1)
        self.assertTrue(any("first reading failed" in n for n in notes))

    def test_never_readable_meter_counts_nothing(self):
        ok, bad, notes = Flaky(), Flaky(broken=True), []
        bad.name = "GPU 1 test"
        with Sampler([ok, bad], 0.02, notes=notes):
            time.sleep(0.2)
        self.assertEqual(bad.energy_j, 0.0)
        self.assertGreater(ok.energy_j, 10)
        self.assertTrue(any("GPU 1 test: no reading worked" in n for n in notes))

    def test_healthy_run_has_no_note(self):
        m, notes = Flaky(), []
        with Sampler([m], 0.02, notes=notes) as s:
            time.sleep(0.2)
        self.assertEqual(notes, [])
        self.assertAlmostEqual(m.energy_j, 100 * s.duration_s, places=6)


@unittest.skipIf(os.name == "nt", "amdgpu is read from Linux sysfs")
class RunWhenTheDeviceGoesAway(unittest.TestCase):
    def test_run_reports_what_was_measured(self):
        with tempfile.TemporaryDirectory() as t:
            hw = Path(t) / "class/drm/card0/device/hwmon/hwmon0"
            hw.mkdir(parents=True)
            (hw / "name").write_text("amdgpu")
            (hw / "power1_average").write_text("150000000")  # 150 W
            out_json = Path(t) / "r.json"
            err, out = io.StringIO(), io.StringIO()
            env = {"JOULES_SYSFS": t, "DO_NOT_TRACK": "1", "JOULES_NO_TRACK": "1"}
            with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                os.environ.pop("JOULES_SIMULATE", None)
                code = cli.main(["run", "--no-cpu", "--interval", "0.05", "--out", str(out_json), "--",
                                 sys.executable, "-c", f"import os, time; time.sleep(0.5); os.remove({str(hw / 'power1_average')!r})"])
            self.assertEqual(code, 0, err.getvalue())
            r = json.loads(out_json.read_text())
            self.assertAlmostEqual(r["devices"][0]["energy_j"], 75, delta=40)  # 150 W for ~0.5 s
            self.assertTrue(any("last reading failed" in n for n in r["notes"]))


if __name__ == "__main__":
    unittest.main()
