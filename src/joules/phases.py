"""Phases: the energy of named stretches of work inside a longer measurement.

The meters keep running; a phase's energy is the change in each meter's running total between the mark that
starts it and the mark that ends it. Each mark reads every meter first, so a phase boundary is as sharp as the
hardware counter allows (RAPL updates about every millisecond, NVIDIA's energy counter every few to 100
milliseconds; a meter that only reports power is integrated between readings). Phases may overlap and nest.

Used by joules agent (POST /phase/start and /phase/end, GET /phases) and by the Python API (joules.Session).
"""
from __future__ import annotations

import itertools
import re
import statistics
import threading
import time

from .meters import Meter

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+#()=,-]{0,63}$")
MAX_OPEN = 1024      # phases started and not yet ended
MAX_NAMES = 256      # distinct phase names kept in the summary
KEEP = 10_000        # results kept per name for medians


class PhaseError(ValueError):
    """A bad phase request: unknown id, bad name, too many open phases."""


def _quantiles(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return xs[0], xs[0]
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], q[2]


def summarize_values(xs: list[float]) -> dict:
    """median, mean, min, max, the quartiles and the relative spread (interquartile range over the median)."""
    q1, q3 = _quantiles(xs)
    med = statistics.median(xs)
    return {"median": med, "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs), "q1": q1, "q3": q3,
            "spread_pct": (q3 - q1) / med * 100 if med else 0.0}


class PhaseClock:
    def __init__(self, meters: list[Meter], lock: threading.Lock | None = None):
        self.meters = meters
        self.lock = lock or threading.Lock()   # the same lock the meter-polling loop holds
        self._state = threading.Lock()         # this object's own bookkeeping
        self._open: dict[str, dict] = {}
        self._done: dict[str, list[dict]] = {}
        self._recent: list[dict] = []
        self._tot: dict[str, list] = {}         # name -> [count, joules, seconds] since the start (or a reset)
        self._ids = itertools.count(1)

    # ------------------------------------------------------------ marks
    def snapshot(self) -> dict:
        """Read every meter now and return each one's running total (and the GPUs' health totals)."""
        with self.lock:
            now = time.monotonic()
            for m in self.meters:
                try:
                    m.poll(now)
                except (OSError, ValueError):
                    pass
            return {"t": now, "wall": time.time(), "j": {m.name: m.energy_j for m in self.meters},
                    "h": {m.name: m.health_state() for m in self.meters if hasattr(m, "health_state")}}

    def begin(self, name: str) -> dict:
        name = (name or "").strip()
        if not NAME_RE.match(name):
            raise PhaseError("a phase name is 1 to 64 letters, digits, spaces or . _ : / + # ( ) = , - and starts with a letter or digit")
        with self._state:
            if len(self._open) >= MAX_OPEN:
                raise PhaseError(f"{MAX_OPEN} phases are already open; end some first")
            if name not in self._done and len(self._done) >= MAX_NAMES:
                raise PhaseError(f"at most {MAX_NAMES} different phase names; reset to start over")
            pid = f"p{next(self._ids)}"
        snap = self.snapshot()
        with self._state:
            self._open[pid] = {"id": pid, "name": name, "snap": snap}
            self._done.setdefault(name, [])
        return {"id": pid, "name": name, "started_at": snap["wall"]}

    def end(self, pid: str | None = None, name: str | None = None) -> dict:
        """End a phase by its id, or the most recently started open phase with this name."""
        snap = self.snapshot()
        with self._state:
            if pid:
                ph = self._open.pop(pid, None)
                if ph is None:
                    raise PhaseError(f"no open phase {pid}")
            elif name:
                ids = [k for k, v in self._open.items() if v["name"] == name]
                if not ids:
                    raise PhaseError(f"no open phase named {name}")
                ph = self._open.pop(ids[-1])
            else:
                raise PhaseError("give the phase's id or name")
        r = self._result(ph, snap)
        with self._state:
            lst = self._done.setdefault(r["name"], [])
            lst.append(r)
            del lst[:-KEEP]
            t = self._tot.setdefault(r["name"], [0, 0.0, 0.0])
            t[0] += 1; t[1] += r["energy_j"]; t[2] += r["duration_s"]
            self._recent.append(r)
            del self._recent[:-200]
        return r

    def _result(self, ph: dict, snap: dict) -> dict:
        a = ph["snap"]
        dur = max(0.0, snap["t"] - a["t"])
        by = {k: max(0.0, v - a["j"].get(k, v)) for k, v in snap["j"].items()}
        energy = sum(by.values())
        r = {"id": ph["id"], "name": ph["name"], "started_at": a["wall"], "ended_at": snap["wall"], "duration_s": dur,
             "energy_j": energy, "avg_w": energy / dur if dur > 0 else 0.0, "devices": by}
        gpu = {}
        for m in self.meters:
            before = a["h"].get(m.name)
            if before is not None and hasattr(m, "health_since"):
                h = m.health_since(before)
                if h:
                    gpu[m.name] = h
        if gpu:
            r["gpu"] = gpu
        return r

    # ------------------------------------------------------------ results
    def summary(self) -> dict:
        """Per phase name: how many ended, total energy and time, and the median, quartiles and spread of each."""
        with self._state:
            done = {k: list(v) for k, v in self._done.items()}
            tot = {k: tuple(v) for k, v in self._tot.items()}
            open_n = len(self._open)
        out = {}
        for name, rs in done.items():
            if not rs:
                continue
            e = [r["energy_j"] for r in rs]
            s = [r["duration_s"] for r in rs]
            n, tot_j, tot_s = tot[name]
            out[name] = {"count": n, "energy_j_total": tot_j, "duration_s_total": tot_s,
                         "avg_w": tot_j / tot_s if tot_s > 0 else 0.0,
                         "energy_j": summarize_values(e), "duration_s": summarize_values(s)}
        return {"open": open_n, "phases": out}

    def recent(self, n: int = 100) -> list[dict]:
        with self._state:
            return list(self._recent[-n:])

    def totals(self) -> dict[str, tuple[int, float, float]]:
        """name -> (count, joules, seconds), for Prometheus counters."""
        with self._state:
            return {k: tuple(v) for k, v in self._tot.items()}

    def reset(self) -> None:
        with self._state:
            self._open.clear(); self._done.clear(); self._recent.clear(); self._tot.clear()
