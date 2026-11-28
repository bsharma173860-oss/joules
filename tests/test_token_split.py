"""joules proxy --split tokens: energy shared by each request's work (prompt tokens over prefill, output pieces as
they stream), with the equal split unchanged. Run: PYTHONPATH=src python3 -m unittest tests.test_token_split -v"""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from joules import meters, proxy


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Watts(meters.Meter):
    """A meter at constant power on the fake clock."""
    def __init__(self, w):
        super().__init__("GPU 0 test")
        self.kind, self.method, self.w = "gpu", "test", w

    def start(self, now):
        self.t = now

    def poll(self, now):
        self.energy_j += self.w * (now - self.t)
        self.t = now

    def stop(self, now):
        self.poll(now)


class LedgerSplit(unittest.TestCase):
    """The Ledger on a fake clock: exact numbers, no timing noise."""

    def setUp(self):
        self.clock = Clock()
        p = mock.patch("joules.proxy.time.monotonic", self.clock)
        p.start()
        self.addCleanup(p.stop)

    def ledger(self, split="tokens", w=1.0, watts=100.0, idle=None):
        return proxy.Ledger([Watts(watts)], idle_w=idle, split=split, prefill_weight=w)

    def tick(self, L, dt=0.1):
        self.clock.t += dt
        with L.lock:
            L._tick()

    def long_and_short(self, split):
        """A 150K-token prompt and a 20-token one, streamed, running together for 3 s at 100 W."""
        L = self.ledger(split)
        L.begin("long", 150_000, "test"); L.streaming("long", True)
        L.begin("short", 20, "test"); L.streaming("short", True)
        out = {"long": 0, "short": 0}
        for k in range(30):                       # 30 x 0.1 s
            self.tick(L)
            t = (k + 1) * 0.1
            if t > 0.05:                          # the short prompt's prefill ends at once
                out["short"] += 1; L.progress("short", out["short"])
            if t > 2.0:                           # the long one's prefill takes 2 s
                out["long"] += 1; L.progress("long", out["long"])
        a = L.end("long", output_tokens=out["long"], prompt_tokens=150_000)
        b = L.end("short", output_tokens=out["short"], prompt_tokens=20)
        return L, a, b

    def test_150k_prompt_and_20_token_prompt_get_very_different_shares(self):
        L, a, b = self.long_and_short("tokens")
        self.assertAlmostEqual(a["energy_j"] + b["energy_j"], 300.0, places=6)   # 100 W x 3 s, all of it
        # while the long prompt is in prefill (2 s) it does nearly all the work; after that both stream a token per
        # interval and share the last second equally
        self.assertTrue(35.0 < b["energy_j"] < 55.0, b["energy_j"])
        self.assertGreater(a["energy_j"], 5 * b["energy_j"])
        L, a, b = self.long_and_short("equal")
        self.assertAlmostEqual(a["energy_j"], b["energy_j"], places=6)           # the old behaviour: same bill

    def test_prefill_weight_moves_energy_between_prompt_and_output(self):
        shares = []
        for w in (0.001, 1.0):
            L = self.ledger(w=w)
            L.begin("reader", 1_000, "test"); L.streaming("reader", True)    # long prompt, 1 output token
            L.begin("writer", 10, "test"); L.streaming("writer", True)       # short prompt, many output tokens
            n = 0
            for k in range(20):
                self.tick(L)
                n += 1; L.progress("writer", n)
                if k == 9:
                    L.progress("reader", 1)
            r = L.end("reader", output_tokens=1, prompt_tokens=1_000)
            L.end("writer", output_tokens=n, prompt_tokens=10)
            shares.append(r["energy_j"])
        self.assertLess(shares[0], shares[1] / 3)  # a cheaper prompt token, a smaller bill for the reader

    def test_shares_always_add_up_to_the_measured_energy(self):
        import random
        rnd = random.Random(7)
        for split in ("equal", "tokens"):
            L = self.ledger(split, w=0.3, idle=40.0)
            got, live, out, nid = [], {}, {}, 0
            for step in range(400):
                if rnd.random() < 0.15 and len(live) < 6:
                    nid += 1; rid = f"r{nid}"
                    L.begin(rid, rnd.choice([None, 5, 500, 50_000]), "test")
                    live[rid] = rnd.random() < 0.7; out[rid] = 0
                    L.streaming(rid, live[rid])
                self.tick(L, dt=rnd.choice([0.05, 0.1, 0.3]))
                for rid, streamed in list(live.items()):
                    if streamed and rnd.random() < 0.5:
                        out[rid] += rnd.randint(1, 3); L.progress(rid, out[rid])
                    if rnd.random() < 0.05:
                        got.append(L.end(rid, output_tokens=out[rid] or rnd.randint(1, 50), prompt_tokens=rnd.randint(1, 9000)))
                        del live[rid]
            for rid in list(live):
                got.append(L.end(rid))
            self.assertEqual(L.pending, [])
            self.assertAlmostEqual(sum(r["energy_j"] for r in got), L.attributed_j, places=6, msg=split)
            idle_total = sum(r["above_idle_j"] for r in got)
            self.assertLessEqual(idle_total, L.attributed_j + 1e-9)
            self.assertTrue(all(r["energy_j"] >= 0 for r in got))

    def test_one_request_gets_all_of_it(self):
        for streamed in (True, False):
            L = self.ledger()
            L.begin("only", 1234, "test"); L.streaming("only", streamed)
            for k in range(10):
                self.tick(L)
                if streamed and k >= 3:
                    L.progress("only", k)
            r = L.end("only", output_tokens=7, prompt_tokens=1234)
            self.assertAlmostEqual(r["energy_j"], 100.0, places=6)  # 100 W x 1 s

    def test_non_streamed_work_is_spread_over_its_time(self):
        L = self.ledger(w=1.0)
        L.begin("a", 100, "test"); L.streaming("a", False)
        L.begin("b", 100, "test"); L.streaming("b", False)
        for _ in range(10):
            self.tick(L)
        self.assertEqual(len(L.pending), 10)   # nothing known until a response arrives
        a = L.end("a", output_tokens=900, prompt_tokens=100)   # 1000 units of work
        b = L.end("b", output_tokens=0, prompt_tokens=100)     # b's share was fixed when a ended (its estimate)
        self.assertAlmostEqual(a["energy_j"] + b["energy_j"], 100.0, places=6)
        self.assertGreater(a["energy_j"], 5 * b["energy_j"])

    def test_cached_prompt_tokens_do_not_count_as_prefill(self):
        res = {}
        for cached in (0, 9_990):
            L = self.ledger()
            L.begin("x", 10_000, "test"); L.streaming("x", False)
            L.begin("y", 10, "test"); L.streaming("y", False)
            for _ in range(10):
                self.tick(L)
            L.end("y", output_tokens=100, prompt_tokens=10)
            res[cached] = L.end("x", output_tokens=100, prompt_tokens=10_000, cached_tokens=cached)
        # x's share was fixed when y ended (on x's estimate), so compare the receipts' prompt basis instead
        self.assertEqual(res[9_990]["prompt"], 10)
        self.assertEqual(res[9_990]["prompt_source"], "usage minus cached")

    def test_old_intervals_are_shared_on_estimates(self):
        L = proxy.Ledger([Watts(100)], split="tokens", settle_after=1.0)
        L.begin("p", 500, "test"); L.streaming("p", True)   # stays in prefill
        for _ in range(30):
            self.tick(L)
        self.assertLess(len(L.pending), 13)                   # memory stays bounded
        r = L.end("p")
        self.assertAlmostEqual(r["energy_j"], 300.0, places=6)

    def test_bad_options(self):
        with self.assertRaises(ValueError):
            proxy.Ledger([Watts(1)], split="mystery")
        with self.assertRaises(ValueError):
            proxy.Ledger([Watts(1)], split="tokens", prefill_weight=-1)


# ---------------------------------------------------------------- through the proxy, with a stand-in engine
def engine(tokenize=True):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/tokenize":
                if not tokenize:
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                words = sum(len((m.get("content") or "").split()) for m in body.get("messages", []))
                return self._json({"count": words, "tokens": [], "max_model_len": 4096})
            words = sum(len((m.get("content") or "").split()) for m in body.get("messages", []))
            time.sleep(body.get("prefill_s", 0.05))
            n = body.get("n", 5)
            usage = {"prompt_tokens": words, "completion_tokens": n, "total_tokens": words + n}
            if not body.get("stream"):
                time.sleep(n * 0.01)
                return self._json({"choices": [{"message": {"content": "x " * n}}], "usage": usage})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                d = b"data: " + json.dumps(obj).encode() + b"\n\n" if obj != "[DONE]" else b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(d), d)); self.wfile.flush()
            for i in range(n):
                send({"choices": [{"delta": {"content": "t"}}]})
                time.sleep(0.01)
            if body.get("usage", True):
                send({"choices": [], "usage": usage})
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")

        def _json(self, obj):
            d = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)

    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


class ProxySplit(unittest.TestCase):
    def start(self, split="tokens", tokenize=True):
        self.up = engine(tokenize)
        self.p = proxy.ProxyServer(("127.0.0.1", 0), f"http://127.0.0.1:{self.up.server_port}",
                                   meters.parse_simulation("gpu:Test GPU:100"), interval=0.02, split=split, prefill_weight=0.5)
        threading.Thread(target=self.p.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.p.server_address[1]}"
        self.addCleanup(lambda: (self.p.shutdown(), self.p.server_close(), self.up.shutdown(), self.up.server_close()))

    def post(self, body):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            h, data = r.headers, r.read()
        with urllib.request.urlopen(self.url + h["X-Energy-Receipt"], timeout=10) as r:
            return json.loads(r.read())

    def test_streamed_and_not_streamed_receipts_state_the_split(self):
        self.start()
        msg = [{"role": "user", "content": "one two three four"}]
        for stream in (True, False):
            r = self.post({"model": "m", "messages": msg, "stream": stream, "n": 8})
            self.assertEqual(r["split"], "tokens")
            self.assertEqual(r["weights"], {"prefill_token": 0.5, "output_token": 1.0, "prefill_token_from": "given"})
            self.assertIn("in proportion to their work", r["basis"])
            self.assertEqual(r["work"]["prompt_tokens"], 4)
            self.assertEqual(r["work"]["prompt_tokens_from"], "usage")
            self.assertEqual(r["work"]["streamed"], stream)
            self.assertEqual(r["output_tokens"], 8)
            self.assertGreater(r["energy_j"], 0)

    def test_tokenize_gives_the_prompt_when_usage_does_not(self):
        self.start()
        r = self.post({"model": "m", "messages": [{"role": "user", "content": "a b c d e f g"}], "stream": True,
                       "usage": False, "prefill_s": 0.3})
        self.assertEqual((r["work"]["prompt_tokens"], r["work"]["prompt_tokens_from"]), (7, "tokenize"))

    def test_no_tokenize_endpoint_falls_back_to_the_estimate(self):
        self.start(tokenize=False)
        text = "x" * 400  # 400 characters, about 100 tokens
        r = self.post({"model": "m", "messages": [{"role": "user", "content": text}], "stream": True, "usage": False})
        self.assertEqual((r["work"]["prompt_tokens"], r["work"]["prompt_tokens_from"]), (100, "estimate"))
        self.assertFalse(self.p.tokenize_ok)        # asked once, then not again
        r = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(r["work"]["prompt_tokens_from"], "usage")

    def test_long_prompt_pays_more_than_short_one_through_the_proxy(self):
        self.start()
        out = {}
        long_msg = [{"role": "user", "content": "w " * 20_000}]
        short_msg = [{"role": "user", "content": "hi"}]
        ts = [threading.Thread(target=lambda: out.__setitem__("long", self.post({"model": "m", "messages": long_msg, "stream": True, "n": 5, "prefill_s": 0.6}))),
              threading.Thread(target=lambda: out.__setitem__("short", self.post({"model": "m", "messages": short_msg, "stream": True, "n": 5, "prefill_s": 0.6})))]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertGreater(out["long"]["energy_j"], 10 * out["short"]["energy_j"])

    def test_equal_split_keeps_its_receipt(self):
        self.start(split="equal")
        r = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r["basis"], "measured device energy, shared equally between requests running at the same time")
        self.assertEqual(r["split"], "equal")
        self.assertNotIn("weights", r)


if __name__ == "__main__":
    unittest.main()
