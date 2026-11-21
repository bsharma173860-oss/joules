"""NVML regression: one failed energy read (NVML returns an error, the binding returns None) must skip
that poll, not kill the sampler thread and with it every other meter.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import threading
import time
import unittest

from joules import meters
from joules.sampler import Sampler


class FlakyNvml:
    """300 W energy counter whose second read fails."""
    def __init__(self, clock):
        self.clock, self.reads = clock, 0
    def energy_mj(self, h):
        self.reads += 1
        return None if self.reads == 2 else int(1e9 + self.clock() * 300_000)


class Steady(meters.PowerMeter):
    kind, method = "cpu", "power sampling"
    def __init__(self):
        super().__init__("CPU")
        self.polls = 0
    def read_w(self):
        self.polls += 1
        return 50.0


class NvmlReadFailure(unittest.TestCase):
    def test_failed_read_raises_oserror(self):
        nv = FlakyNvml(lambda: 0.0)
        m = meters.NvmlCounterMeter(nv, 0, "GPU 0")
        m.start(0.0)
        with self.assertRaises(OSError):
            m.poll(1.0)

    def test_sampler_survives_a_failed_read(self):
        died = []
        old = threading.excepthook
        threading.excepthook = lambda a: died.append(a.exc_type)
        try:
            t0 = time.monotonic()
            gpu = meters.NvmlCounterMeter(FlakyNvml(lambda: time.monotonic() - t0), 0, "GPU 0")
            cpu = Steady()
            with Sampler([gpu, cpu], 0.02) as s:
                time.sleep(0.5)
        finally:
            threading.excepthook = old
        self.assertEqual(died, [])
        self.assertGreater(cpu.polls, 10)  # the other meter kept being polled
        self.assertAlmostEqual(gpu.energy_j, 300 * s.duration_s, delta=300 * 0.05)  # the skipped poll loses nothing from a counter
        self.assertAlmostEqual(cpu.energy_j, 50 * s.duration_s, delta=1)


if __name__ == "__main__":
    unittest.main()
