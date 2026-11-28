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
is shared between the requests running at that moment: equally (--split equal, the default), or by each
request's work in the interval (--split tokens: output pieces streamed, plus prompt tokens x --prefill-weight
over its prefill). One request at a time gets all of it. With --subtract-idle the share of energy above the
idle machine is reported too.

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
EQUAL_BASIS = "measured device energy, shared equally between requests running at the same time"
SPLITS = ("equal", "tokens")
DEFAULT_PREFILL_WEIGHT = 1.0
SETTLE_AFTER_S = 30.0  # an interval waits at most this long for a request's prefill to end before it is shared on estimates


def tokens_basis(w: float) -> str:
    return ("measured device energy, shared between requests running at the same time in proportion to their work in "
            f"each sampling interval: output tokens streamed in it, plus prompt tokens x {w:g} spread over each request's "
            "prefill (from its start to its first output token). Non-streamed responses: the request's work spread "
            "evenly over its time")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class Ledger:
    """Polls the meters and shares each interval's energy between the requests running in it.

    split="equal": every request running in an interval gets the same share.
    split="tokens": shares follow each request's work in the interval. Decode work is the output pieces streamed in
    it; prefill work is the request's prompt tokens x prefill_weight, spread evenly from the request's start to its
    first output piece. A streamed request still in prefill has an unknown prefill length, so intervals it takes part
    in wait (at most SETTLE_AFTER_S, or until a request in them ends) and are then shared on what is known by then.
    Non-streamed requests show no progress until their response arrives: their work (prompt x weight + output) is
    spread evenly over their time. Each interval's energy is always shared out in full: nothing is lost or invented.
    """

    def __init__(self, meters: list[Meter], idle_w: float | None = None, split: str = "equal",
                 prefill_weight: float = DEFAULT_PREFILL_WEIGHT, settle_after: float = SETTLE_AFTER_S):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join(SPLITS)}")
        if prefill_weight < 0:
            raise ValueError("--prefill-weight must be 0 or more")
        self.meters = meters
        self.idle_w = idle_w
        self.split, self.prefill_weight, self.settle_after = split, prefill_weight, settle_after
        self.lock = threading.Lock()
        now = time.monotonic()
        for m in meters:
            m.start(now)
        self.last_total, self.last_t = 0.0, now
        self.active: dict[str, dict] = {}
        self.pending: list[dict] = []  # tokens split: intervals not shared out yet
        self.total_j = self.attributed_j = 0.0
        self.power_w = 0.0
        self.decode_rate = None  # output tokens per second of one request, from finished ones (for estimates)

    # ---- the meter clock
    def _tick(self) -> None:  # call with self.lock held
        now = time.monotonic()
        for m in self.meters:
            try:
                m.poll(now)
            except (OSError, ValueError):
                pass
        total = sum(m.energy_j for m in self.meters)
        d_e, dt = max(0.0, total - self.last_total), now - self.last_t
        t_a = self.last_t
        self.last_total, self.last_t = total, now
        self.total_j += d_e
        if dt > 0:
            self.power_w = d_e / dt
        if not self.active:
            return
        above = max(0.0, d_e - self.idle_w * dt) if self.idle_w is not None else 0.0
        self.attributed_j += d_e
        if self.split == "equal":
            n = len(self.active)
            for r in self.active.values():
                r["energy_j"] += d_e / n
                r["above_idle_j"] += above / n
            return
        dec = {}
        for rid, r in self.active.items():
            dec[rid] = r["out"] - r["out_tick"]
            r["out_tick"] = r["out"]
        if dt > 0 or d_e > 0:
            self.pending.append({"a": t_a, "b": now, "e": d_e, "above": above, "dec": dec})
        self._settle(now)

    # ---- tokens split
    def _resolved(self, r: dict, iv: dict) -> bool:
        if r.get("t_end") is not None:
            return True
        if r.get("streaming") is not True:
            return False  # not streamed, or response not started: its progress shows only at the end
        return r["t_first"] is not None or iv["b"] <= r["t0"]

    def _work(self, rid: str, r: dict, iv: dict, now: float) -> float:
        w = self.prefill_weight
        prompt = r["prompt"] or 0
        if r.get("streaming") is True:
            tf = r["t_first"] if r["t_first"] is not None else (r["t_end"] if r.get("t_end") is not None else now)
            pre = w * prompt * _overlap(iv["a"], iv["b"], r["t0"], tf) / max(tf - r["t0"], 1e-6)
            return iv["dec"].get(rid, 0) + pre
        t_end = r["t_end"] if r.get("t_end") is not None else now
        span = max(t_end - r["t0"], 1e-6)
        if r.get("t_end") is not None:
            total = w * prompt + (r["out"] or 0)
        else:  # still running, not streamed: its prompt, and output at the rate finished requests showed
            total = w * prompt + (self.decode_rate or 0.0) * span
        return total * _overlap(iv["a"], iv["b"], r["t0"], t_end) / span

    def _share(self, iv: dict, now: float) -> None:
        rs = {rid: self.active[rid] for rid in iv["dec"] if rid in self.active}
        if not rs:
            return
        work = {rid: self._work(rid, r, iv, now) for rid, r in rs.items()}
        total = sum(work.values())
        for rid, r in rs.items():
            f = work[rid] / total if total > 0 else 1.0 / len(rs)  # no visible work (e.g. GET /v1/models): equal
            r["energy_j"] += iv["e"] * f
            r["above_idle_j"] += iv["above"] * f

    def _settle(self, now: float, force: str | None = None) -> None:
        """Share out every pending interval whose requests' work is known; with force=rid, also every interval that
        request took part in (it is ending); intervals older than settle_after go on estimates."""
        keep = []
        for iv in self.pending:
            rs = [self.active[rid] for rid in iv["dec"] if rid in self.active]
            if (force in iv["dec"] or now - iv["b"] > self.settle_after
                    or all(self._resolved(r, iv) for r in rs)):
                self._share(iv, now)
            else:
                keep.append(iv)
        self.pending = keep

    # ---- requests
    def begin(self, rid: str, prompt_tokens: int | None = None, prompt_source: str | None = None) -> None:
        with self.lock:
            self._tick()  # energy so far belongs to the requests already running
            self.active[rid] = {"energy_j": 0.0, "above_idle_j": 0.0, "t0": time.monotonic(), "concurrency": 1,
                                "prompt": prompt_tokens, "prompt_source": prompt_source if prompt_tokens is not None else None,
                                "streaming": None, "out": 0, "out_tick": 0, "t_first": None, "t_end": None}
            for r in self.active.values():
                r["concurrency"] = max(r["concurrency"], len(self.active))

    def set_prompt(self, rid: str, tokens: int, source: str) -> None:
        with self.lock:
            r = self.active.get(rid)
            if r is not None:
                r["prompt"], r["prompt_source"] = tokens, source

    def streaming(self, rid: str, yes: bool) -> None:
        with self.lock:
            r = self.active.get(rid)
            if r is not None:
                r["streaming"] = yes

    def progress(self, rid: str, out_pieces: int) -> None:
        """Output pieces streamed so far (only grows)."""
        if self.split != "tokens":
            return
        with self.lock:
            r = self.active.get(rid)
            if r is None or out_pieces <= r["out"]:
                return
            if r["t_first"] is None:
                self._tick()  # energy up to now is still prefill
                r["t_first"] = time.monotonic()
            r["out"] = out_pieces

    def end(self, rid: str, output_tokens: int | None = None, prompt_tokens: int | None = None,
            cached_tokens: int | None = None) -> dict:
        with self.lock:
            self._tick()
            r = self.active[rid]
            now = time.monotonic()
            if self.split == "tokens":
                r["t_end"] = now
                if prompt_tokens is not None:  # the engine's own count beats an estimate
                    r["prompt"] = max(0, prompt_tokens - (cached_tokens or 0))
                    r["prompt_source"] = "usage" + (" minus cached" if cached_tokens else "")
                if output_tokens is not None and r.get("streaming") is not True:
                    r["out"] = output_tokens
                if r["t_first"] is not None and r["out"] and now > r["t_first"]:
                    rate = r["out"] / (now - r["t_first"])
                    self.decode_rate = rate if self.decode_rate is None else 0.8 * self.decode_rate + 0.2 * rate
                self._settle(now, force=rid)
            self.active.pop(rid)
        r["duration_s"] = time.monotonic() - r.pop("t0")
        return r

    def basis(self) -> str:
        return EQUAL_BASIS if self.split == "equal" else tokens_basis(self.prefill_weight)

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
        self.output = self.prompt = self.cached = None
        self.chunks = 0  # streamed content pieces, used when the server sends no usage
        self.pieces = 0  # streamed output pieces of any kind (content or reasoning): the decode progress
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
            d = u.get("prompt_tokens_details")
            c = d.get("cached_tokens") if isinstance(d, dict) else None
            self.cached = c if isinstance(c, int) else self.cached
        if isinstance(o.get("eval_count"), int):  # Ollama's final object
            self.output = o["eval_count"]
            self.prompt = o.get("prompt_eval_count") if isinstance(o.get("prompt_eval_count"), int) else self.prompt
        choices = o.get("choices")
        for c in choices if isinstance(choices, list) else []:
            delta = c.get("delta") if isinstance(c, dict) else None
            if isinstance(c, dict) and ((delta.get("content") if isinstance(delta, dict) else None) or c.get("text")):
                self.chunks += 1
                self.pieces += 1
            elif isinstance(delta, dict) and (delta.get("reasoning_content") or delta.get("reasoning")):
                self.pieces += 1
        msg = o.get("message")  # a chat message, or a plain string in error bodies (vLLM: {"object": "error", "message": "..."})
        if isinstance(o.get("response"), str) and o["response"] or isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"]:
            self.chunks += 1
            self.pieces += 1

    def tokens(self) -> tuple[int | None, bool]:
        """(output tokens, exact?)"""
        if self.output is not None:
            return self.output, True
        return (self.chunks, False) if self.chunks else (None, False)


def prompt_estimate(j: dict) -> int | None:
    """Prompt tokens guessed from a request body (about 4 characters a token), until the engine says."""
    chars, seen = 0, False
    for m in j.get("messages") if isinstance(j.get("messages"), list) else []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            chars += len(c); seen = True
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chars += len(part["text"]); seen = True
    p = j.get("prompt")
    if isinstance(p, str):
        chars += len(p); seen = True
    elif isinstance(p, list) and all(isinstance(x, int) for x in p):
        return len(p)  # already token ids
    elif isinstance(p, list):
        chars += sum(len(x) for x in p if isinstance(x, str)); seen = True
    return max(1, round(chars / 4)) if seen else None


# ---------------------------------------------------------------- server
class ProxyServer(LocalHTTPServer):
    def __init__(self, addr, upstream: str, meters: list[Meter], idle_w=None, price=0.10, g_per_kwh=370.0, pue=1.0,
                 interval=0.1, timeout=600.0, log=None, keep=1000, split="equal", prefill_weight=DEFAULT_PREFILL_WEIGHT,
                 prefill_weight_from="given"):
        super().__init__(addr, Handler)
        self.upstream = urllib.parse.urlsplit(upstream.rstrip("/"))
        if self.upstream.scheme not in ("http", "https") or not self.upstream.hostname:
            raise ValueError("--upstream must be an http:// or https:// address")
        self.ledger = Ledger(meters, idle_w, split=split, prefill_weight=prefill_weight)
        self.prefill_weight_from = prefill_weight_from
        self.tokenize_ok = True  # the upstream has a /tokenize endpoint, until it says otherwise
        self.price, self.g_per_kwh, self.pue, self.timeout = price, g_per_kwh, pue, timeout
        self.log = log
        self.receipts: deque[dict] = deque(maxlen=keep)
        self.by_id: dict[str, dict] = {}
        self.ids = itertools.count(1)
        self.stats = {"requests": 0, "output_tokens": 0, "energy_j": 0.0}
        self.recorded = 0  # receipts ever recorded, in the order requests finished (ids follow the order they started)
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
            self.recorded += 1
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
        model = j = None
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
        est = prompt_estimate(j) if s.ledger.split == "tokens" and isinstance(j, dict) else None
        s.ledger.begin(rid, est, "estimate")
        if est is not None and s.tokenize_ok and urllib.parse.urlsplit(self.path).path.endswith(("/chat/completions", "/completions")):
            threading.Thread(target=self._tokenize, args=(rid, j), daemon=True).start()
        try:
            self._forward(rid, model, body, headers, conn)
        finally:
            if rid in s.ledger.active:  # anything unexpected must not leave a phantom request taking a share of later energy
                s.ledger.end(rid)
                self.close_connection = True

    def _forward(self, rid: str, model: str | None, body: bytes, headers: dict, conn) -> None:
        s, u = self.server, self.server.upstream
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
        s.ledger.streaming(rid, streaming)
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
                s.ledger.progress(rid, scan.pieces)
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

    def _tokenize(self, rid: str, j: dict) -> None:
        """Ask the engine (vLLM and SGLang both serve POST /tokenize) for the exact prompt length; keep the estimate
        if it cannot say."""
        s, u = self.server, self.server.upstream
        body = {k: j[k] for k in ("model", "messages", "prompt", "tools") if k in j}
        conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(u.hostname, u.port, timeout=10)
        try:
            conn.request("POST", (u.path or "") + "/tokenize", body=json.dumps(body),
                         headers={"Content-Type": "application/json", "Host": u.netloc})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status == 404:
                s.tokenize_ok = False
                return
            n = json.loads(data).get("count") if resp.status == 200 else None
            if isinstance(n, list):
                n = sum(x for x in n if isinstance(x, int))
            if isinstance(n, int):
                s.ledger.set_prompt(rid, n, "tokenize")
        except (OSError, ValueError, http.client.HTTPException, AttributeError):
            pass
        finally:
            conn.close()

    def _receipt(self, rid: str, model: str | None, status: int, scan: TokenScanner) -> dict:
        s = self.server
        out, exact = scan.tokens()
        r = s.ledger.end(rid, output_tokens=out, prompt_tokens=scan.prompt, cached_tokens=scan.cached)
        e = r["energy_j"] * s.pue
        kwh = e / 3.6e6
        receipt = {"id": rid, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": self.command,
                   "path": urllib.parse.urlsplit(self.path).path, "model": model, "status": status,
                   "duration_s": round(r["duration_s"], 4), "energy_j": e, "co2_g": kwh * s.g_per_kwh,
                   "cost": kwh * s.price, "concurrency": r["concurrency"], "measured_by": s.measured_by,
                   "split": s.ledger.split, "basis": s.ledger.basis()}
        if s.ledger.split == "tokens":
            receipt["weights"] = {"prefill_token": s.ledger.prefill_weight, "output_token": 1.0,
                                  "prefill_token_from": s.prefill_weight_from}
            receipt["work"] = {"prompt_tokens": r["prompt"], "prompt_tokens_from": r["prompt_source"],
                               "output_pieces": r["out"], "streamed": bool(r["streaming"])}
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
        self.sent = 0  # how many receipts, counted in the order they finished, earlier flushes covered
        self.t0 = time.time()

    def flush(self) -> list[str]:
        """Sends what finished since the last flush. Returns error messages, if any."""
        with self.s.stats_lock:  # by finishing order, not id: a long request finishes after later, shorter ones
            n = min(self.s.recorded - self.sent, len(self.s.receipts))
            new = list(self.s.receipts)[len(self.s.receipts) - n:] if n > 0 else []
            self.sent = self.s.recorded
        if not new:
            return []
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
