"""joules proxy: an energy receipt on every AI response.

Sits in front of any model server (Ollama, vLLM, llama.cpp, LM Studio, TGI, or anything that speaks
HTTP) and measures the hardware while each request runs. Every response carries its energy:

    X-Energy-Joules: 41.2
    X-Energy-Joules-Per-Token: 0.194
    X-Energy-CO2-Grams: 0.00423
    X-Energy-Receipt: /joules/receipts/r-000017

Streaming responses cannot carry headers written after they start, so they carry X-Energy-Receipt and
the full receipt is at that address once the stream ends.

How energy is split: the meters cover whole devices, so in every sampling interval the energy measured
is shared equally between the requests running at that moment. One request at a time gets all of it.
With --subtract-idle the share of energy above the idle machine is reported too.

Also serves /joules/receipts (recent receipts), /joules/metrics (Prometheus) and /joules/health.
"""
from __future__ import annotations

import http.client
import itertools
import json
import sys
import threading
import time
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler

from . import __version__
from . import share as sharing
from .agent import LocalHTTPServer
from .meters import Meter

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
EXPOSE = "X-Energy-Joules, X-Energy-Joules-Above-Idle, X-Energy-Joules-Per-Token, X-Energy-Output-Tokens, X-Energy-CO2-Grams, X-Energy-Receipt"


# ---------------------------------------------------------------- attribution
class Ledger:
    """Polls the meters and shares each interval's energy between the requests running in it."""

    def __init__(self, meters: list[Meter], idle_w: float | None = None):
        self.meters = meters
        self.idle_w = idle_w
        self.lock = threading.Lock()
        now = time.monotonic()
        for m in meters:
            m.start(now)
        self.last_total, self.last_t = 0.0, now
        self.active: dict[str, dict] = {}
        self.total_j = self.attributed_j = 0.0
        self.power_w = 0.0

    def _tick(self) -> None:  # call with self.lock held
        now = time.monotonic()
        for m in self.meters:
            try:
                m.poll(now)
            except (OSError, ValueError):
                pass
        total = sum(m.energy_j for m in self.meters)
        d_e, dt = max(0.0, total - self.last_total), now - self.last_t
        self.last_total, self.last_t = total, now
        self.total_j += d_e
        if dt > 0:
            self.power_w = d_e / dt
        if self.active:
            n = len(self.active)
            above = max(0.0, d_e - self.idle_w * dt) / n if self.idle_w is not None else 0.0
            for r in self.active.values():
                r["energy_j"] += d_e / n
                r["above_idle_j"] += above
            self.attributed_j += d_e

    def begin(self, rid: str) -> None:
        with self.lock:
            self._tick()  # energy so far belongs to the requests already running
            self.active[rid] = {"energy_j": 0.0, "above_idle_j": 0.0, "t0": time.monotonic(), "concurrency": 1}
            for r in self.active.values():
                r["concurrency"] = max(r["concurrency"], len(self.active))

    def end(self, rid: str) -> dict:
        with self.lock:
            self._tick()
            r = self.active.pop(rid)
        r["duration_s"] = time.monotonic() - r.pop("t0")
        return r

    def loop(self, stop: threading.Event, interval: float) -> None:
        while not stop.wait(interval):
            with self.lock:
                self._tick()

    def close(self) -> None:
        now = time.monotonic()
        for m in self.meters:
            try:
                m.stop(now)
            except (OSError, ValueError):
                pass


# ---------------------------------------------------------------- token counting
class TokenScanner:
    """Finds token counts in OpenAI-style (usage) and Ollama-style (eval_count) bodies, streamed or not."""

    def __init__(self):
        self.output = self.prompt = None
        self.chunks = 0  # streamed content pieces, used when the server sends no usage
        self._buf = b""

    def feed(self, data: bytes) -> None:
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        for line in lines:
            self._line(line)
        if len(self._buf) > 1 << 20:  # one huge line: not a stream we understand
            self._buf = b""

    def finish(self) -> None:
        if self._buf:
            self._line(self._buf)
            self._buf = b""

    def whole(self, body: bytes) -> None:
        try:
            self._obj(json.loads(body))
        except ValueError:
            pass

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line.startswith(b"{"):
            return
        try:
            self._obj(json.loads(line))
        except ValueError:
            pass

    def _obj(self, o) -> None:
        if not isinstance(o, dict):
            return
        u = o.get("usage")
        if isinstance(u, dict) and isinstance(u.get("completion_tokens", u.get("output_tokens")), int):
            self.output = u.get("completion_tokens", u.get("output_tokens"))
            p = u.get("prompt_tokens", u.get("input_tokens"))
            self.prompt = p if isinstance(p, int) else self.prompt
        if isinstance(o.get("eval_count"), int):  # Ollama's final object
            self.output = o["eval_count"]
            self.prompt = o.get("prompt_eval_count") if isinstance(o.get("prompt_eval_count"), int) else self.prompt
        for c in o.get("choices") or []:
            if isinstance(c, dict) and ((c.get("delta") or {}).get("content") or c.get("text")):
                self.chunks += 1
        if isinstance(o.get("response"), str) and o["response"] or isinstance((o.get("message") or {}).get("content"), str) and o["message"]["content"]:
            self.chunks += 1

    def tokens(self) -> tuple[int | None, bool]:
        """(output tokens, exact?)"""
        if self.output is not None:
            return self.output, True
        return (self.chunks, False) if self.chunks else (None, False)


# ---------------------------------------------------------------- server
class ProxyServer(LocalHTTPServer):
    def __init__(self, addr, upstream: str, meters: list[Meter], idle_w=None, price=0.10, g_per_kwh=370.0, pue=1.0,
                 interval=0.1, timeout=600.0, log=None, keep=1000):
        super().__init__(addr, Handler)
        self.upstream = urllib.parse.urlsplit(upstream.rstrip("/"))
        if self.upstream.scheme not in ("http", "https") or not self.upstream.hostname:
            raise ValueError("--upstream must be an http:// or https:// address")
        self.ledger = Ledger(meters, idle_w)
        self.price, self.g_per_kwh, self.pue, self.timeout = price, g_per_kwh, pue, timeout
        self.log = log
        self.receipts: deque[dict] = deque(maxlen=keep)
        self.by_id: dict[str, dict] = {}
        self.ids = itertools.count(1)
        self.stats = {"requests": 0, "output_tokens": 0, "energy_j": 0.0}
        self.stats_lock = threading.Lock()
        self.measured_by = ", ".join(sorted({m.method for m in meters})) or "nothing"
        self._stop = threading.Event()
        threading.Thread(target=self.ledger.loop, args=(self._stop, interval), daemon=True).start()

    def new_id(self) -> str:
        return f"r-{next(self.ids):06d}"

    def record(self, receipt: dict) -> None:
        with self.stats_lock:
            if len(self.receipts) == self.receipts.maxlen:
                self.by_id.pop(self.receipts[0]["id"], None)
            self.receipts.append(receipt)
            self.by_id[receipt["id"]] = receipt
            self.stats["requests"] += 1
            self.stats["energy_j"] += receipt["energy_j"]
            self.stats["output_tokens"] += receipt.get("output_tokens") or 0
        if self.log:
            self.log(receipt)

    def shutdown(self) -> None:
        self._stop.set()
        super().shutdown()
        self.ledger.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: ProxyServer

    def log_message(self, *a):  # one line per receipt instead, from ProxyServer.log
        pass

    def do_GET(self):
        self._handle()
    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_GET

    # ---- local endpoints
    def _json(self, code: int, obj) -> None:
        data = json.dumps(obj, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _local(self) -> None:
        s = self.server
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/joules/health":
            return self._json(200, {"ok": True, "joules": __version__, "measured_by": s.measured_by,
                                    "devices": [m.name for m in s.ledger.meters]})
        if url.path == "/joules/receipts":
            q = urllib.parse.parse_qs(url.query)
            try:
                n = max(1, min(1000, int(q.get("limit", ["50"])[0])))
            except ValueError:
                n = 50
            with s.stats_lock:
                items = list(s.receipts)[-n:][::-1]
            return self._json(200, {"receipts": items})
        if url.path.startswith("/joules/receipts/"):
            rid = url.path.rsplit("/", 1)[1]
            with s.stats_lock:
                r = s.by_id.get(rid)
            if r:
                return self._json(200, r)
            if rid in s.ledger.active:
                return self._json(202, {"id": rid, "status": "running"})
            return self._json(404, {"error": "no such receipt (receipts are kept in memory, newest 1000)"})
        if url.path == "/joules/metrics":
            data = metrics(s).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return self._json(404, {"error": "joules endpoints: /joules/health, /joules/receipts, /joules/receipts/<id>, /joules/metrics"})

    # ---- proxying
    def _handle(self) -> None:
        if self.path.startswith("/joules/"):
            return self._local()
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            return self._json(411, {"error": "joules proxy needs a Content-Length on request bodies"})
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        model = None
        try:
            j = json.loads(body) if body else None
            model = j.get("model") if isinstance(j, dict) and isinstance(j.get("model"), str) else None
        except ValueError:
            pass
        s, u = self.server, self.server.upstream
        conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(u.hostname, u.port, timeout=s.timeout)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Host"] = u.netloc
        if body:
            headers["Content-Length"] = str(len(body))
        rid = s.new_id()
        s.ledger.begin(rid)
        try:
            conn.request(self.command, (u.path or "") + self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            s.ledger.end(rid)
            conn.close()
            return self._json(502, {"error": f"joules proxy cannot reach {u.scheme}://{u.netloc} ({e})"})
        scan = TokenScanner()
        ctype = resp.getheader("Content-Type", "") or ""
        streaming = ("event-stream" in ctype or "ndjson" in ctype
                     or (resp.getheader("Content-Length") is None and resp.chunked and self.command != "HEAD"))
        passthrough = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP]
        try:
            if not streaming:
                data = resp.read()
                scan.whole(data)
                receipt = self._receipt(rid, model, resp.status, scan)
                self.send_response(resp.status, resp.reason)
                for k, v in passthrough:
                    self.send_header(k, v)
                for k, v in energy_headers(receipt):
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
                return
            self.send_response(resp.status, resp.reason)
            for k, v in passthrough:
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Energy-Receipt", f"/joules/receipts/{rid}")
            self.send_header("Access-Control-Expose-Headers", EXPOSE)
            self.end_headers()
            client_gone = False
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                scan.feed(chunk)
                if not client_gone:
                    try:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    except OSError:
                        client_gone = True  # keep reading so the receipt covers the whole generation
            scan.finish()
            self._receipt(rid, model, resp.status, scan)
            if not client_gone:
                self.wfile.write(b"0\r\n\r\n")
        except (OSError, http.client.HTTPException):
            if rid in s.ledger.active:
                self._receipt(rid, model, 502, scan)
            self.close_connection = True
        finally:
            conn.close()

    def _receipt(self, rid: str, model: str | None, status: int, scan: TokenScanner) -> dict:
        s = self.server
        r = s.ledger.end(rid)
        e = r["energy_j"] * s.pue
        kwh = e / 3.6e6
        out, exact = scan.tokens()
        receipt = {"id": rid, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": self.command,
                   "path": urllib.parse.urlsplit(self.path).path, "model": model, "status": status,
                   "duration_s": round(r["duration_s"], 4), "energy_j": e, "co2_g": kwh * s.g_per_kwh,
                   "cost": kwh * s.price, "concurrency": r["concurrency"], "measured_by": s.measured_by,
                   "basis": "measured device energy, shared equally between requests running at the same time"}
        if s.ledger.idle_w is not None:
            receipt["above_idle_j"] = r["above_idle_j"] * s.pue
        if out:
            receipt["output_tokens"] = out
            receipt["tokens_exact"] = exact
            receipt["j_per_token"] = receipt.get("above_idle_j", e) / out
        if scan.prompt is not None:
            receipt["prompt_tokens"] = scan.prompt
        s.record(receipt)
        return receipt


def energy_headers(r: dict) -> list[tuple[str, str]]:
    h = [("X-Energy-Joules", f"{r['energy_j']:.4g}"), ("X-Energy-CO2-Grams", f"{r['co2_g']:.3g}"),
         ("X-Energy-Receipt", f"/joules/receipts/{r['id']}"), ("Access-Control-Expose-Headers", EXPOSE)]
    if "above_idle_j" in r:
        h.append(("X-Energy-Joules-Above-Idle", f"{r['above_idle_j']:.4g}"))
    if "output_tokens" in r:
        h.append(("X-Energy-Output-Tokens", str(r["output_tokens"])))
        h.append(("X-Energy-Joules-Per-Token", f"{r['j_per_token']:.4g}"))
    return h


def _label(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def metrics(s: ProxyServer) -> str:
    led = s.ledger
    with led.lock:
        led._tick()
        devices = [(m.name, m.kind, m.method, m.energy_j) for m in led.meters]
        total, attributed, power, active = led.total_j, led.attributed_j, led.power_w, len(led.active)
    with s.stats_lock:
        st = dict(s.stats)
    out = ["# HELP joules_device_energy_joules_total Energy each device used since the proxy started.",
           "# TYPE joules_device_energy_joules_total counter"]
    for name, kind, method, e in devices:
        out.append(f'joules_device_energy_joules_total{{device="{_label(name)}",kind="{kind}",method="{_label(method)}"}} {e:.6g}')
    out += ["# HELP joules_power_watts Power of all measured devices over the last sampling interval.",
            "# TYPE joules_power_watts gauge", f"joules_power_watts {power:.6g}",
            "# HELP joules_energy_joules_total Energy of all measured devices since the proxy started.",
            "# TYPE joules_energy_joules_total counter", f"joules_energy_joules_total {total:.6g}",
            "# HELP joules_request_energy_joules_total Energy used while requests were running.",
            "# TYPE joules_request_energy_joules_total counter", f"joules_request_energy_joules_total {attributed:.6g}",
            "# HELP joules_requests_total Requests completed.", "# TYPE joules_requests_total counter",
            f"joules_requests_total {st['requests']}",
            "# HELP joules_output_tokens_total Output tokens counted in responses.", "# TYPE joules_output_tokens_total counter",
            f"joules_output_tokens_total {st['output_tokens']}",
            "# HELP joules_requests_in_flight Requests running now.", "# TYPE joules_requests_in_flight gauge",
            f"joules_requests_in_flight {active}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- tracking (one record per model per window)
class Tracker:
    def __init__(self, server: ProxyServer, project: str, key: str, api_url: str | None, every: float, device: str, count: int, sources: str):
        self.s, self.project, self.key, self.api_url, self.every = server, project, key, api_url, every
        self.device, self.count, self.sources = device[:120], count, sources
        self.since = 0
        self.t0 = time.time()

    def flush(self) -> list[str]:
        """Sends what finished since the last flush. Returns error messages, if any."""
        with self.s.stats_lock:
            new = [r for r in self.s.receipts if int(r["id"][2:]) > self.since]
        if not new:
            return []
        self.since = max(int(r["id"][2:]) for r in new)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.t0))
        window = max(time.time() - self.t0, 1e-3)
        self.t0 = time.time()
        errors = []
        groups: dict[str, list[dict]] = {}
        for r in new:
            groups.setdefault(r.get("model") or "requests", []).append(r)
        for model, rs in groups.items():
            e = sum(r["energy_j"] for r in rs)
            above = [r["above_idle_j"] for r in rs if "above_idle_j" in r]
            toks = sum(r.get("output_tokens") or 0 for r in rs)
            rec = {"project": self.project, "kind": "bench" if toks else "run",
                   "label": f"proxy · {model} · {len(rs)} requests"[:200], "started_at": started, "duration_s": window,
                   "energy_j": e, "co2_g": sum(r["co2_g"] for r in rs), "cost": sum(r["cost"] for r in rs),
                   "price_per_kwh": self.s.price, "g_per_kwh": self.s.g_per_kwh, "device": self.device,
                   "device_count": self.count, "idle_subtracted": bool(above), "simulated": False,
                   "joules_version": __version__, "sources": self.sources}
            if toks:
                rec["output_tokens"] = toks
                rec["j_per_token"] = (sum(above) if above else e) / toks
            try:
                sharing.send_track(rec, self.key, self.api_url)
            except RuntimeError as err:
                errors.append(str(err))
        return errors

    def loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.every):
            for e in self.flush():
                print(f"joules proxy: tracking failed: {e}", file=sys.stderr)
