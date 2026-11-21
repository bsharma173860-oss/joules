"""Power-sampling regression: a failed power read (NVML fallback, ADLX) must be skipped, with the next good
reading bridging the gap, not integrated as 0 W.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import unittest

from joules import meters


class EveryOtherFails:
    """Steady 300 W; every second read fails (None), as NVML and ADLX report errors."""
    def __init__(self):
        self.n = 0
    def _read(self, value):
        self.n += 1
        return None if self.n % 2 == 0 else value
    def power_mw(self, h):
        return self._read(300_000)
    def power_w(self, gpu, board):
        return self._read(300.0)


def run_like_the_sampler(m, seconds=10.0, step=0.1):
    m.start(0.0)
    for i in range(1, int(round(seconds / step)) + 1):
        try:
            m.poll(i * step)
        except (OSError, ValueError):
            pass  # what Sampler._loop does with a read error
    return m.energy_j


class PowerReadFailure(unittest.TestCase):
    def test_nvml_power_fallback_skips_failed_reads(self):
        m = meters.NvmlPowerMeter(EveryOtherFails(), 1, "GPU 1 old card")
        self.assertAlmostEqual(run_like_the_sampler(m), 3000.0, places=6)  # 300 W for 10 s, not 1500
        self.assertEqual(m.peak_w, 300.0)

    def test_adlx_skips_failed_reads(self):
        m = meters.AdlxPowerMeter(EveryOtherFails(), None, "GPU 0 AMD Radeon", board=True)
        self.assertAlmostEqual(run_like_the_sampler(m), 3000.0, places=6)

    def test_failed_read_raises_oserror(self):
        nv = EveryOtherFails()
        nv.n = 1  # next read fails
        with self.assertRaises(OSError):
            meters.NvmlPowerMeter(nv, 1, "GPU 1").read_w()

    def test_healthy_reads_unchanged(self):
        class Ok:
            def power_mw(self, h): return 250_000
        self.assertAlmostEqual(run_like_the_sampler(meters.NvmlPowerMeter(Ok(), 0, "GPU 0")), 2500.0, places=6)


if __name__ == "__main__":
    unittest.main()
