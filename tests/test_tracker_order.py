"""proxy --track regression: requests are tracked in the order they finish. Receipt ids are given when a
request starts, so a long request that finishes after a later, shorter one was flushed must still be sent.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import unittest
from unittest import mock

from joules import meters, proxy


def receipt(i, j):
    return {"id": f"r-{i:06d}", "model": "m", "energy_j": j, "co2_g": 0.0, "cost": 0.0, "output_tokens": 10}


class TrackerOrder(unittest.TestCase):
    def setUp(self):
        self.p = proxy.ProxyServer(("127.0.0.1", 0), "http://127.0.0.1:9", meters.parse_simulation("gpu:Test GPU:100"), keep=3)
        self.sent = []
        patcher = mock.patch.object(proxy.sharing, "send_track", lambda r, k, u: self.sent.append(r))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.t = proxy.Tracker(self.p, "proj", "k", None, 300, "Test GPU", 1, "gpu")

    def tearDown(self):
        self.p._stop.set()
        self.p.server_close()

    def test_long_request_finishing_after_a_flush_is_tracked(self):
        self.p.record(receipt(2, 5.0))     # short request: started second, finished first
        self.t.flush()
        self.p.record(receipt(1, 5000.0))  # long request: started first, finishes after the flush
        self.t.flush()
        self.assertEqual(sum(r["energy_j"] for r in self.sent), 5005.0)
        self.assertEqual(self.t.flush(), [])
        self.assertEqual(len(self.sent), 2)  # nothing sent twice

    def test_more_than_kept_between_flushes_sends_what_is_kept(self):
        for i in range(1, 6):
            self.p.record(receipt(i, 1.0))
        self.t.flush()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["energy_j"], 3.0)  # only the newest 3 are still in memory


if __name__ == "__main__":
    unittest.main()
