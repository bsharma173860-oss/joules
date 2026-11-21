"""Counter meters regression: peak power must not alias with counters that refresh in steps. NVML's energy counter
moves about every 100 ms, so with 0.1 s polling one gap can hold two refreshes; a steady 300 W showed a ~600 W peak.
Energy is unchanged: only peak_w (in the JSON result) is taken over ~1 s windows.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import random
import unittest

from joules import meters


class SteppedCounter(meters.CounterMeter):
    """Steady 300 W, with the counter refreshed every 100 ms."""
    kind = "gpu"
    def __init__(self):
        super().__init__("GPU 0")
        self.now = 0.0
    def read_j(self):
        return int(self.now / 0.1 + 1e-9) * 0.1 * 300.0


def poll_with_jitter(m, n=600, seed=1):
    rng = random.Random(seed)
    m.start(0.0)
    t = 0.0
    for _ in range(n):
        t += 0.1 + rng.uniform(-0.02, 0.02)  # Event.wait(0.1) jitter
        m.now = t
        m.poll(t)
    m.stop(t)
    return t


class CounterPeak(unittest.TestCase):
    def test_stepped_counter_peak_is_near_the_real_power(self):
        m = SteppedCounter()
        t = poll_with_jitter(m)
        self.assertLess(m.peak_w, 340)   # was ~580 W
        self.assertGreaterEqual(m.peak_w, 299)
        self.assertAlmostEqual(m.energy_j, int(t / 0.1 + 1e-9) * 30.0, places=6)  # every joule the counter showed

    def test_a_real_burst_still_shows(self):
        class Burst(meters.CounterMeter):
            kind = "gpu"
            def __init__(self): super().__init__("GPU 0"); self.e = 0.0
            def read_j(self): return self.e
        m = Burst()
        m.start(0.0)
        t = 0.0
        for i in range(1, 51):  # 5 s at 0.1 s: 100 W, but 400 W from 2 s to 3.5 s
            t = i * 0.1
            m.e += (400 if 2.0 < t <= 3.5 else 100) * 0.1
            m.poll(t)
        m.stop(t)
        self.assertGreater(m.peak_w, 350)
        self.assertLessEqual(m.peak_w, 400 + 1e-6)

    def test_short_run_peak_is_its_average(self):
        m = SteppedCounter()
        m.start(0.0)
        m.now = 0.5
        m.stop(0.5)
        self.assertAlmostEqual(m.peak_w, 300.0)


if __name__ == "__main__":
    unittest.main()
