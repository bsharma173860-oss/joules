"""joules proxy --calibrate: measure --prefill-weight on your own machine.

The weight is the energy of one prompt token relative to one output token. It depends on the GPU, the model, the
engine and how many requests run at once, so it is measured, not assumed. Three phases of requests go straight to the
engine (not through a proxy), each measured with the same meters joules proxy uses, above the machine's idle power:

    C: short prompt, 1 output token        (the fixed cost of a request)
    A: long prompt, 1 output token         (A - C: the prompt tokens)
    B: short prompt, many output tokens    (B - C: the output tokens)

    energy per prompt token = (A - C) / (prompt tokens in A - prompt tokens in C)
    energy per output token = (B - C) / (output tokens in B - output tokens in C)
    --prefill-weight         = the first / the second

Prompts are random words, different every time, so no prefix cache can serve them. Output length is forced with
ignore_eos (vLLM and SGLang). Token counts are the engine's own (usage).
"""
from __future__ import annotations

import http.client
import json
import random
import statistics
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from .proxy import Ledger

WORDS = ("time year people way day man thing woman life child world school state family student group country problem "
         "hand part place case week company system program question work government number night point home water room "
         "mother area money story fact month lot right study book eye job word business issue side kind head house "
         "service friend father power hour game line end member law car city community name president team minute idea "
         "kid body information back parent face others level office door health person art war history party result").split()


def _post(upstream, path: str, body: dict, timeout: float) -> dict:
    u = upstream
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port, timeout=timeout)
    try:
        conn.request("POST", (u.path or "") + path, body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Host": u.netloc})
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path}: HTTP {r.status}: {data[:300]!r}")
        return json.loads(data)
    finally:
        conn.close()


def _prompt(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def calibrate(upstream: str, model: str, meters, interval: float = 0.1, long_tokens: int = 2048, short_tokens: int = 16,
              output_tokens: int = 256, requests: int = 8, concurrency: int = 1, rounds: int = 2, idle_s: float = 5.0,
              timeout: float = 600.0, log=print) -> dict:
    u = urllib.parse.urlsplit(upstream.rstrip("/"))
    rng = random.Random()
    L = Ledger(meters)
    stop = threading.Event()
    threading.Thread(target=L.loop, args=(stop, interval), daemon=True).start()
    try:
        log(f"idle: {idle_s:g} s with no requests")
        with L.lock:
            L._tick()
            e0, t0 = L.total_j, time.monotonic()
        time.sleep(idle_s)
        with L.lock:
            L._tick()
            idle_w = (L.total_j - e0) / (time.monotonic() - t0)
        log(f"  idle power {idle_w:.1f} W")

        def phase(name: str, prompt_words: int, max_tokens: int) -> dict:
            bodies = [{"model": model, "prompt": _prompt(rng, prompt_words), "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True} for _ in range(requests)]
            L.begin(name)
            t = time.monotonic()
            with ThreadPoolExecutor(concurrency) as ex:
                outs = list(ex.map(lambda b: _post(u, "/v1/completions", b, timeout), bodies))
            r = L.end(name)
            dur = time.monotonic() - t
            p = sum(o["usage"]["prompt_tokens"] for o in outs)
            o = sum(o["usage"]["completion_tokens"] for o in outs)
            above = max(0.0, r["energy_j"] - idle_w * dur)
            log(f"  {name}: {requests} requests, {p:,} prompt + {o:,} output tokens, {dur:.1f} s, {above:.1f} J above idle")
            return {"prompt_tokens": p, "output_tokens": o, "duration_s": dur, "energy_j": r["energy_j"], "above_idle_j": above}

        words_long = int(long_tokens * 0.75)  # about 0.75 words a token for these words; the engine's count is used
        words_short = max(1, int(short_tokens * 0.75))
        log("warm-up")
        phase("warm-up", words_short, 4)
        runs = []
        for k in range(rounds):
            log(f"round {k + 1} of {rounds}")
            C = phase("C short prompt, 1 output", words_short, 1)
            A = phase("A long prompt, 1 output", words_long, 1)
            B = phase(f"B short prompt, {output_tokens} output", words_short, output_tokens)
            key = "above_idle_j"
            noise = 0.01 * B["energy_j"]  # above-idle differences smaller than 1% of the total are rounding, not signal
            if B["above_idle_j"] - C["above_idle_j"] <= noise or A["above_idle_j"] - C["above_idle_j"] <= 0.01 * A["energy_j"]:
                key = "energy_j"  # nothing above idle: simulated or estimated meters, so this is a time ratio
            per_prompt = (A[key] - C[key]) / max(1, A["prompt_tokens"] - C["prompt_tokens"])
            per_output = (B[key] - C[key]) / max(1, B["output_tokens"] - C["output_tokens"])
            runs.append({"A": A, "B": B, "C": C, "j_per_prompt_token": per_prompt, "j_per_output_token": per_output,
                         "basis": "energy above idle" if key == "above_idle_j" else "total energy (none above idle measured)",
                         "prefill_weight": per_prompt / per_output if per_output > 0 and per_prompt > 0 else None})
            log(f"  {per_prompt:.4g} J per prompt token, {per_output:.4g} J per output token: weight "
                f"{runs[-1]['prefill_weight']:.3g}" if runs[-1]["prefill_weight"] is not None else "  no output energy measured")
    finally:
        stop.set()
        L.close()
    ws = [r["prefill_weight"] for r in runs if r["prefill_weight"] is not None and r["prefill_weight"] > 0]
    return {"upstream": upstream, "model": model, "concurrency": concurrency, "requests_per_phase": requests,
            "long_prompt_tokens": long_tokens, "output_tokens": output_tokens, "idle_w": idle_w,
            "devices": [{"name": m.name, "kind": m.kind, "method": m.method} for m in meters],
            "rounds": runs, "prefill_weight": statistics.median(ws) if ws else None,
            "basis": sorted({r["basis"] for r in runs}),
            "spread": (min(ws), max(ws)) if ws else None}
