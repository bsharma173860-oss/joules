"""joules ci regression: one outlier run must not make a real regression read as run-to-run noise
(and turn off --budget), while genuine noise still reads as noise.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import unittest

from joules import ci


def runs(*energies):
    return [{"energy_j": e, "duration_s": 1} for e in energies]


class CiNoise(unittest.TestCase):
    def test_one_outlier_does_not_hide_a_regression(self):
        s = ci.summarize(runs(150, 151, 152, 149, 99), runs(100, 100, 101, 99, 100), budget=10)
        self.assertAlmostEqual(s["change_pct"], 50.0)
        self.assertFalse(s["within_noise"])
        self.assertTrue(s["over_budget"])
        md = ci.markdown(s, head_name="h", base_name="main", devices=["x"], estimated=False, command="c")
        self.assertIn("over the 10% budget", md)
        self.assertNotIn("within run-to-run noise", md)

    def test_outlier_in_the_base_does_not_hide_a_regression(self):
        s = ci.summarize(runs(150, 151, 152, 149, 150), runs(100, 100, 101, 99, 151), budget=10)
        self.assertFalse(s["within_noise"])
        self.assertTrue(s["over_budget"])

    def test_noise_still_reads_as_noise(self):
        s = ci.summarize(runs(103, 98, 105, 101, 99), runs(100, 104, 97, 102, 100), budget=1)
        self.assertTrue(s["within_noise"])
        self.assertFalse(s["over_budget"])
        md = ci.markdown(s, head_name="h", base_name="main", devices=["x"], estimated=False, command="c")
        self.assertIn("within run-to-run noise", md)


if __name__ == "__main__":
    unittest.main()
