"""joules run --samples: each device's energy over time, written to a CSV while the command runs.
joules window: the energy between two moments, from that CSV (times taken afterwards, e.g. from a server's log)."""
from __future__ import annotations

import csv
import datetime as dt
import math
import re
import time

from .meters import Meter, PowerMeter

HEADER = ["unix_time", "device", "energy_j", "power_w"]


class SampleLog:
    """One row per device each time a reading of it worked: wall-clock time of that reading, energy since the start
    (the counter's, or power integrated over time) and power (the reading, or the average since the last row).
    Flushed about every second, so a crash keeps what came before."""

    def __init__(self, path: str, notes: list[str]):
        self.f = open(path, "w", newline="", encoding="utf-8")   # (a bad path fails before the command starts)
        self.csv = csv.writer(self.f)
        self.csv.writerow(HEADER)
        self.path, self.notes = path, notes
        self.last: dict[str, tuple[float, float]] = {}
        self.flushed = time.monotonic()

    def write(self, meters: list[Meter], unstarted=()) -> None:
        if self.f is None:
            return
        try:
            for m in meters:
                t = m._t_last
                if m.kind == "est" or m in unstarted or t is None:   # (CPU-time estimate: only known at the end)
                    continue
                prev = self.last.get(m.name)
                if prev and t <= prev[0]:   # no new reading (it failed): no row, the next one spans the gap
                    continue
                e = m.energy_j
                w = m._w if isinstance(m, PowerMeter) else (e - prev[1]) / (t - prev[0]) if prev else ""
                wall = time.time() - (time.monotonic() - t)   # (now, not once: the wall clock may be stepped)
                self.csv.writerow([f"{wall:.6f}", m.name, f"{e:.6f}", w if w == "" else f"{w:.3f}"])
                self.last[m.name] = (t, e)
            if time.monotonic() - self.flushed >= 1.0:
                self.f.flush()
                self.flushed = time.monotonic()
        except OSError as e:
            self._failed(e)

    def _failed(self, e: OSError) -> None:
        self.notes.append(f"Could not write the samples to {self.path} ({e.strerror or e}); they stop there.")
        self.close()

    def close(self) -> None:
        if self.f is not None:
            f, self.f = self.f, None
            try:
                f.close()
            except OSError as e:   # (the last flush: closed all the same)
                self.notes.append(f"Could not write the samples to {self.path} ({e.strerror or e}); the last second "
                                  "or so of them is missing.")


def parse_time(text: str) -> float:
    """Unix seconds, or an ISO date and time (no zone: local time)."""
    try:
        t = float(text)
    except ValueError:
        t = None
    if t is not None:
        if not math.isfinite(t) or t < 0:
            raise ValueError(f"{text!r} is not a time")
        if t > 1e11:
            raise ValueError(f"{text!r} looks like milliseconds or finer: give unix seconds (e.g. {t / 1000:.3f})")
        return t
    # (as Python 3.11 reads them, on 3.9 too: a comma before the fraction, any number of its digits, +HHMM)
    s = re.sub(r"[Zz]$", "+00:00", text.strip())
    s = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", s)
    s = re.sub(r"(:\d\d)[.,](\d+)", lambda m: m[1] + "." + (m[2] + "000000")[:6], s)
    try:
        return dt.datetime.fromisoformat(s).timestamp()   # (naive: local time)
    except (ValueError, OverflowError, OSError):
        raise ValueError(f"{text!r} is not a time: give unix seconds or an ISO date and time, "
                         "e.g. 2026-11-22T14:03:07.250 (local) or ...+00:00") from None


def read_samples(path: str) -> dict[str, list[list[tuple[float, float]]]]:
    """Each device's (time, energy) rows in the order read, in runs whose times go forward (a new run where the
    wall clock was set back during the run)."""
    out: dict[str, list[list[tuple[float, float]]]] = {}
    with open(path, newline="", encoding="utf-8") as f:   # (line by line: a day of readings is a big file)
        if next(csv.reader([f.readline()]), []) != HEADER:
            raise ValueError(f"{path} is not a joules samples file (its first line should be {','.join(HEADER)})")
        for n, line in enumerate(f, 2):
            if not line.endswith("\n"):   # (an unfinished last line: the run was cut off while writing it)
                break
            row = next(csv.reader([line]), [])
            try:
                if len(row) != len(HEADER):
                    raise ValueError
                t, e = float(row[0]), float(row[2])
            except ValueError:
                raise ValueError(f"{path} line {n} is not a sample: {line.strip()}") from None
            runs = out.setdefault(row[1], [[]])
            if runs[-1] and t < runs[-1][-1][0]:
                runs.append([])
            runs[-1].append((t, e))
    return out


def _at(rows: list[tuple[float, float]], t: float) -> float:
    """Energy at time t, interpolated between the readings either side."""
    lo, hi = 0, len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, e0), (t1, e1) = rows[lo], rows[hi]
    return e0 if t1 <= t0 else e0 + (e1 - e0) * (t - t0) / (t1 - t0)


def window(samples: dict[str, list[tuple[float, float]]], start: float, end: float, idle_w: dict | None = None) -> dict:
    if end <= start:
        raise ValueError("the end must be after the start")
    if not samples:
        raise ValueError("the samples file has no samples")
    devices = []
    for name, runs in samples.items():
        first, last = min(r[0][0] for r in runs), max(r[-1][0] for r in runs)
        if start < first or end > last:
            raise ValueError(f"{name} has samples from {_show(first)} to {_show(last)}; the window "
                             f"{_show(start)} to {_show(end)} is not inside that")
        (i, e0), (j, e1) = _find(runs, start, name), _find(runs, end, name)
        if j < i:
            raise ValueError(f"the wall clock was set back during the run, and {_show(end)} comes before "
                             f"{_show(start)} in {name}'s readings")
        row = {"name": name, "energy_j": e1 - e0}
        if idle_w and name in idle_w:
            row["idle_w"] = idle_w[name]
            row["above_idle_j"] = max(0.0, row["energy_j"] - idle_w[name] * (end - start))
        devices.append(row)
    total = {"energy_j": sum(d["energy_j"] for d in devices)}
    if devices and all("above_idle_j" in d for d in devices):
        total["above_idle_j"] = sum(d["above_idle_j"] for d in devices)
    return {"schema": "joules/1", "kind": "window", "start": start, "end": end, "duration_s": end - start,
            "devices": devices, "totals": total}


def _find(runs, t: float, name: str) -> tuple[int, float]:
    """Which run of readings has time t, and the energy then."""
    hits = [i for i, r in enumerate(runs) if r[0][0] <= t <= r[-1][0]]
    if len(hits) > 1:
        raise ValueError(f"the wall clock was set back during the run, so {_show(t)} happened twice in {name}'s "
                         "readings: pick a time outside the repeated stretch")
    if not hits:   # (in a gap where the clock jumped back: the readings either side are of different moments)
        raise ValueError(f"{_show(t)} falls where the wall clock was set back in {name}'s readings")
    return hits[0], _at(runs[hits[0]], t)


def _show(t: float) -> str:
    return f"{dt.datetime.fromtimestamp(t).isoformat(timespec='milliseconds')} ({t:.3f})"
