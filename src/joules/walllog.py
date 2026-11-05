"""A wall meter's time-stamped log, for `joules calibrate --wall-csv`.

A logging meter (a USB or serial logger, a smart plug polled once a second, a PDU outlet) writes one reading per line:
a time and the power in watts. joules integrates the readings over exactly the seconds it measured, so the wall energy
and the meters' energy cover the same time. Times can be Unix seconds (or milliseconds), or ISO 8601 / "YYYY-MM-DD
HH:MM:SS"; a time without a zone is the computer's local time. The columns are found by name (time, timestamp, date,
datetime, ts; watts, power, w, power_w, apower, "Power (W)") or, without a header, as the first two columns.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import statistics

TIME_NAMES = ("timestamp", "time", "datetime", "date", "ts", "unix", "epoch")
WATT_NAMES = ("watts", "power_w", "power", "w", "apower", "active_power", "power (w)", "watt")


def _time(v: str) -> float | None:
    v = v.strip()
    if not v:
        return None
    try:
        x = float(v)
        return x / 1000 if x > 1e11 else x  # milliseconds since 1970
    except ValueError:
        pass
    try:
        t = dt.datetime.fromisoformat(v.replace("Z", "+00:00").replace("/", "-"))
    except ValueError:
        return None
    return t.timestamp()  # a time without a zone is local time


def _watts(v: str) -> float | None:
    m = re.match(r"^\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", v or "")
    return float(m.group(1)) if m else None


def read(path: str) -> list[tuple[float, float]]:
    """(Unix time, watts) per reading, sorted by time. Raises ValueError with a reason when the file cannot be used."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")
    head = [c.strip().lower() for c in rows[0]]
    ti = next((i for i, c in enumerate(head) if c in TIME_NAMES), None)
    wi = next((i for i, c in enumerate(head) if c in WATT_NAMES or c.startswith("power")), None)
    if ti is not None and wi is not None:
        body = rows[1:]
    elif _time(rows[0][0]) is not None and len(rows[0]) >= 2 and _watts(rows[0][1]) is not None:
        ti, wi, body = 0, 1, rows
    else:
        raise ValueError(f"{path}: no time and watts columns found (name them time and watts, or put them first)")
    out = []
    for r in body:
        if len(r) <= max(ti, wi):
            continue
        t, w = _time(r[ti]), _watts(r[wi])
        if t is not None and w is not None and w >= 0:
            out.append((t, w))
    if len(out) < 2:
        raise ValueError(f"{path}: fewer than two readable readings")
    out.sort()
    return out


def integrate(readings: list[tuple[float, float]], t0: float, t1: float) -> dict:
    """The wall energy between t0 and t1 (Unix seconds), from readings joined by straight lines (trapezoids), and the
    share of that window the readings cover. A gap more than five times the usual spacing between readings counts as
    not covered."""
    if t1 <= t0:
        raise ValueError("the measured window is empty")
    steps = [b[0] - a[0] for a, b in zip(readings, readings[1:]) if b[0] > a[0]]
    usual = statistics.median(steps) if steps else 1.0
    joules = covered = 0.0
    for (ta, wa), (tb, wb) in zip(readings, readings[1:]):
        if tb <= ta or tb <= t0 or ta >= t1 or tb - ta > 5 * usual:
            continue
        lo, hi = max(ta, t0), min(tb, t1)
        w_at = lambda t: wa + (wb - wa) * (t - ta) / (tb - ta)
        joules += (w_at(lo) + w_at(hi)) / 2 * (hi - lo)
        covered += hi - lo
    n = sum(1 for t, _ in readings if t0 <= t <= t1)
    return {"energy_j": joules, "coverage": covered / (t1 - t0), "readings": n, "interval_s": usual}
