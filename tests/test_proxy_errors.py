"""Proxy regression: an upstream error body whose "message" is a string (vLLM, many gateways) must pass
through to the client and must not leave the request counted as running.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

os.environ["JOULES_NO_PROMPT"] = "1"

from joules import meters, proxy  # noqa: E402


def error_server():
    """model "missing" -> 404 with {"object": "error", "message": "..."}; anything else -> 200 after 0.3 s."""
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("model") == "missing":
                code, data = 404, json.dumps({"object": "error", "message": "The model `missing` does not exist.", "code": 404}).encode()
            else:
                time.sleep(0.3)
                code, data = 200, json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 10}}).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class ProxyErrorBodies(unittest.TestCase):
    def setUp(self):
        self.up = error_server()
        self.p = proxy.ProxyServer(("127.0.0.1", 0), f"http://127.0.0.1:{self.up.server_port}",
                                   meters.parse_simulation("gpu:Test GPU:100"), interval=0.02)
        threading.Thread(target=self.p.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.p.server_address[1]}"

    def tearDown(self):
        self.p.shutdown(); self.p.server_close()
        self.up.shutdown(); self.up.server_close()

    def post(self, model):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_string_message_error_passes_through_and_ends_the_request(self):
        code, h, data = self.post("missing")
        self.assertEqual(code, 404)  # not a dropped connection
        self.assertEqual(json.loads(data)["message"], "The model `missing` does not exist.")
        self.assertIn("X-Energy-Joules", h)
        self.assertEqual(list(self.p.ledger.active), [])
        code, h, data = self.post("ok")  # alone again, so it gets all of the 100 W while it runs
        self.assertEqual(code, 200)
        r = json.loads(urllib.request.urlopen(self.url + h["X-Energy-Receipt"], timeout=10).read())
        self.assertEqual(r["concurrency"], 1)
        self.assertAlmostEqual(r["energy_j"], 100 * r["duration_s"], delta=100 * 0.02 * 2)

    def test_unexpected_error_still_ends_the_request(self):
        with mock.patch.object(proxy.TokenScanner, "whole", side_effect=RuntimeError("boom")):
            try:
                self.post("ok")
            except Exception:
                pass  # the connection may drop; what matters is the ledger
        self.assertEqual(list(self.p.ledger.active), [])

    def test_scanner_ignores_non_dict_message_and_delta(self):
        s = proxy.TokenScanner()
        s.whole(b'{"object": "error", "message": "nope"}')
        s.feed(b'data: {"choices": [{"delta": "x"}, "y"]}\n')
        s.feed(b'data: {"message": {"content": "tok"}}\n')
        self.assertEqual(s.tokens(), (1, False))


if __name__ == "__main__":
    unittest.main()
