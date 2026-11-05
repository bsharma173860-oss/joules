"""What an inference server on this machine is serving, for the Live page: tokens per second, requests running and
waiting, and the time to the first token, read from the server's own Prometheus counters (vLLM and SGLang).

joules agent reads http://127.0.0.1:8000/metrics (vLLM's default port) and http://127.0.0.1:30000/metrics (SGLang's)
unless --serve-metrics names another address or is "off". Rates are taken over the last 30 seconds, so one burst does
not swing the number. Only counts and times are read; never prompts, answers or model names.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request

AUTO = ("http://127.0.0.1:8000/metrics", "http://127.0.0.1:30000/metrics")
WINDOW_S = 30.0       # rates over the last 30 seconds
RETRY_S = 60.0        # with nothing found, look again a minute later
# engine: (output tokens, prompt tokens, requests running, requests waiting, time-to-first-token histogram)
ENGINES = {
    "vllm": ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
             "vllm:num_requests_waiting", "vllm:time_to_first_token_seconds"),
    "sglang": ("sglang:generation_tokens_total", "sglang:prompt_tokens_total", "sglang:num_running_reqs",
               "sglang:num_queue_reqs", "sglang:time_to_first_token_seconds"),
}
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+(\S+)")
# local addresses only: no proxy between the agent and a server on the same machine
_OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def parse(text: str) -> dict | None:
    """The engine's counters from a /metrics page, summed over every label set (one per model), or None when the page
    is not vLLM's or SGLang's."""
    sums: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if v == v and abs(v) != float("inf"):
            sums[m.group(1)] = sums.get(m.group(1), 0.0) + v
    for engine, (gen, prompt, running, waiting, ttft) in ENGINES.items():
        if gen in sums:
            return {"engine": engine, "gen": sums[gen], "prompt": sums.get(prompt, 0.0),
                    "running": sums.get(running), "waiting": sums.get(waiting),
                    "ttft_sum": sums.get(ttft + "_sum"), "ttft_count": sums.get(ttft + "_count")}
    return None


class ServeMetrics:
    """Reads an inference server's counters each heartbeat and gives the rates over the last WINDOW_S seconds."""

    def __init__(self, url: str = "auto", timeout: float = 0.5, fetch=None):
        self.urls = AUTO if url == "auto" else (url,)
        self.url: str | None = None if url == "auto" else url
        self.timeout = timeout
        self._fetch = fetch or self._get
        self._samples: list[tuple[float, dict]] = []
        self._next_look = 0.0

    def _get(self, url: str) -> str:
        with _OPEN(url, timeout=self.timeout) as r:
            return r.read(4 << 20).decode("utf-8", "replace")

    def _read(self, now: float) -> dict | None:
        if self.url is None:  # not found yet: look at the usual ports, at most once a minute
            if now < self._next_look:
                return None
            self._next_look = now + RETRY_S
            for u in self.urls:
                try:
                    p = parse(self._fetch(u))
                except (OSError, ValueError, urllib.error.URLError):
                    continue
                if p:
                    self.url = u
                    return p
            return None
        try:
            p = parse(self._fetch(self.url))
        except (OSError, ValueError, urllib.error.URLError):
            p = None
        if p is None and len(self.urls) > 1:  # the server went away: look for it again later
            self.url, self._samples, self._next_look = None, [], now + RETRY_S
        return p

    def read(self, now: float | None = None) -> dict | None:
        """{"engine", "tok_s", "prompt_tok_s", "running", "waiting", and "ttft_s" when there were first tokens}, or None
        until two readings at least 2 seconds apart exist."""
        now = time.monotonic() if now is None else now
        p = self._read(now)
        if p is None:
            return None
        if self._samples and (p["gen"] < self._samples[-1][1]["gen"] or p["engine"] != self._samples[-1][1]["engine"]):
            self._samples = []  # the server restarted: its counters began again
        self._samples.append((now, p))
        while len(self._samples) > 2 and now - self._samples[1][0] >= WINDOW_S:
            self._samples.pop(0)
        t0, a = self._samples[0]
        dt = now - t0
        if dt < 2:
            return None
        out = {"engine": p["engine"], "tok_s": round(max(0.0, p["gen"] - a["gen"]) / dt, 1),
               "prompt_tok_s": round(max(0.0, p["prompt"] - a["prompt"]) / dt, 1),
               "running": int(max(0, min(1e6, p["running"] or 0))), "waiting": int(max(0, min(1e6, p["waiting"] or 0)))}
        if None not in (p["ttft_sum"], p["ttft_count"], a["ttft_sum"], a["ttft_count"]):
            n = p["ttft_count"] - a["ttft_count"]
            if n > 0:
                out["ttft_s"] = round(min(3600.0, max(0.0, (p["ttft_sum"] - a["ttft_sum"]) / n)), 3)
        return out
