"""joules compare: the same work done two (or more) ways, measured fairly.

Runs each command several times, interleaved (A B A B, or A B B A to cancel a steady drift), after unmeasured
warm-up runs, optionally waiting for the GPUs to cool to the same temperature before each run and at one or
more fixed GPU power limits, optionally with the GPUs' clocks locked. Reports per command the median energy, time, average power and energy-delay
product with their spread, what the GPU was doing (clock, throttling, temperature), and for each command
against the baseline the change in each, with a Mann-Whitney p-value: the chance of a difference at least this
large if both were really the same. Warns when the GPU did not run under the same conditions for each command.
"""
from __future__ import annotations

import contextlib
import functools
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

from .meters import Discovery
from .phases import summarize_values
from .sampler import Sampler

# ---------------------------------------------------------------- statistics


@functools.lru_cache(maxsize=None)
def _u_count(n: int, m: int, u: int) -> int:
    """How many orderings of n x's and m y's (no ties) have exactly u pairs with x above y."""
    if u < 0 or u > n * m:
        return 0
    if n == 0 or m == 0:
        return 1 if u == 0 else 0
    return _u_count(n - 1, m, u - m) + _u_count(n, m - 1, u)


def mann_whitney(x: list[float], y: list[float]) -> float | None:
    """Two-sided p-value of the Mann-Whitney U test: exact when there are no ties and the samples are small,
    else the normal approximation with the tie correction. None with fewer than 2 values on a side."""
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return None
    u = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
    allv = list(x) + list(y)
    ties = len(set(allv)) < len(allv)
    if not ties and n + m <= 50:
        total = math.comb(n + m, n)
        k = int(round(u))
        lo = sum(_u_count(n, m, i) for i in range(0, k + 1)) / total
        hi = sum(_u_count(n, m, i) for i in range(k, n * m + 1)) / total
        return min(1.0, 2 * min(lo, hi))
    N = n + m
    counts: dict[float, int] = {}
    for v in allv:
        counts[v] = counts.get(v, 0) + 1
    t = sum(c ** 3 - c for c in counts.values())
    var = n * m / 12 * ((N + 1) - t / (N * (N - 1)))
    if var <= 0:
        return 1.0
    z = max(0.0, abs(u - n * m / 2) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(z / math.sqrt(2)))


# ---------------------------------------------------------------- running


def order(names: list[str], runs: int, how: str) -> list[list[int]]:
    """The command indices of each round. alternate: 0 1, 0 1, ...; abba: 0 1, 1 0, 0 1, ..."""
    out = []
    for r in range(runs):
        idx = list(range(len(names)))
        if how == "abba" and r % 2 == 1:
            idx.reverse()
        out.append(idx)
    return out


def split_command(s: str, windows: bool = os.name == "nt") -> list[str]:
    """A command line as arguments. On Windows a backslash is part of a path, not an escape, but quotes still
    group and are removed, so "C:\\Program Files\\py.exe" -c "print(1)" gives the path and the code unquoted."""
    if not windows:
        return shlex.split(s)
    lex = shlex.shlex(s, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    lex.escape = ""
    return list(lex)


def gpu_meters(d: Discovery) -> list:
    return [m for m in d.meters if m.kind == "gpu" and hasattr(m, "nvml")]


def wait_cool(gpus: list, target: float, timeout: float, log) -> float | None:
    """Wait until every NVIDIA GPU is at or below target °C. Returns the hottest temperature at the end."""
    deadline = time.monotonic() + timeout
    said = False
    while True:
        temps = [t for t in (g.temperature_now() for g in gpus) if t is not None]
        hot = max(temps) if temps else None
        if hot is None or hot <= target or time.monotonic() >= deadline:
            return hot
        if not said:
            log(f"waiting for the GPU to cool from {hot:.0f} °C to {target:g} °C…")
            said = True
        time.sleep(1.0)


def hottest(gpus: list) -> float | None:
    temps = [t for t in (g.temperature_now() for g in gpus) if t is not None]
    return max(temps) if temps else None


@contextlib.contextmanager
def power_limits(gpus: list):
    """set(pct) puts every GPU at pct% of its maximum power limit; the original limits come back at the end."""
    ranges, original = [], []
    for m in gpus:
        if not hasattr(m.nvml, "set_power_limit_mw"):
            raise RuntimeError(f"{m.name}: this driver cannot set a power limit")
        rng, cur = m.nvml.power_limit_range_mw(m.h), m.nvml.power_limit_mw(m.h)
        if not rng or not cur:
            raise RuntimeError(f"{m.name}: the driver does not report its power limits")
        ranges.append(rng); original.append(cur)

    def set_(pct: int) -> list[float]:
        watts = []
        for m, (lo, hi) in zip(gpus, ranges):
            mw = max(lo, min(hi, int(hi * pct / 100)))
            rc = m.nvml.set_power_limit_mw(m.h, mw)
            if rc != 0:
                why = "no permission: run joules compare with sudo" if rc == 4 else f"the driver refused (NVML error {rc})"
                raise RuntimeError(f"could not set the power limit on {m.name}: {why}")
            watts.append(mw / 1000)
        return watts
    try:
        yield set_
    finally:
        for m, mw in zip(gpus, original):
            try:
                m.nvml.set_power_limit_mw(m.h, mw)
            except Exception:
                pass


# ---------------------------------------------------------------- locked clocks (nvidia-smi -lgc)
SMI = "nvidia-smi"


class ClockLockError(RuntimeError):
    """Clocks could not be locked: nothing has run, and every GPU this got to is back to normal."""


def is_admin() -> bool:
    """Root on Linux and macOS, an administrator on Windows."""
    if os.name == "nt":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError):
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def gpu_index(m) -> int | None:
    """The nvidia-smi number of a GPU meter ("GPU 0 Tesla T4" -> 0)."""
    hit = re.match(r"GPU (\d+) ", m.name)
    return int(hit.group(1)) if hit else None


def smi(args: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        p = subprocess.run([SMI, *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "nvidia-smi was not found"
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"nvidia-smi did not run ({e})"
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()


def smi_reason(out: str) -> str:
    """nvidia-smi's refusal in a few words."""
    low = out.lower()
    if "nvidia-smi was not found" in low:
        return "nvidia-smi was not found"
    if "mig" in low:
        return "the GPU is in MIG mode, where clocks cannot be locked"
    if any(w in low for w in ("permission", "root", "not permitted", "administrator", "privilege")):
        return "insufficient permissions (needs root)"
    if "not supported" in low:
        return "this GPU or driver does not support locked clocks"
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    return first[:200] or "nvidia-smi refused without saying why"


def clock_info(idx: int) -> dict:
    """The card's maximum SM clock and its base clock (the default applications clock: what the card can hold
    under its power cap), in MHz, as nvidia-smi reports them; None for either it does not report."""
    rc, out = smi(["-i", str(idx), "--query-gpu=clocks.max.graphics,clocks.default_applications.graphics",
                   "--format=csv,noheader,nounits"])
    vals = [v.strip() for v in out.splitlines()[0].split(",")] if rc == 0 and out else []
    num = lambda v: int(float(v)) if re.fullmatch(r"\d+(\.\d+)?", v or "") else None
    return {"max": num(vals[0]) if vals else None, "base": num(vals[1]) if len(vals) > 1 else None}


@contextlib.contextmanager
def locked_clocks(gpus: list, spec, log):
    """Locks the SM clock of every GPU at spec MHz, or at its base clock with spec "base", before anything runs, and
    yields {meter name: MHz}. Every GPU it locked is put back (nvidia-smi -rgc) at the end, also after a failed run,
    an error or Ctrl-C. A GPU that cannot be locked raises ClockLockError before any command has run."""
    plan = []
    for m in gpus:
        idx = gpu_index(m)
        if idx is None:
            raise ClockLockError(f"cannot lock clocks on {m.name}: joules does not know its nvidia-smi number")
        info = clock_info(idx)
        mhz = info["base"] if spec == "base" else spec
        if not mhz:
            raise ClockLockError(f"cannot lock clocks on GPU {idx}: the driver does not report a base clock; give one in MHz, "
                                 "e.g. --lock-clocks 1200")
        if info["max"] and mhz > info["max"]:
            raise ClockLockError(f"cannot lock clocks on GPU {idx}: {mhz} MHz is above its maximum of {info['max']} MHz")
        plan.append((m, idx, mhz))
    locked: list[int] = []
    try:
        for m, idx, mhz in plan:
            locked.append(idx)  # before the call: Ctrl-C while nvidia-smi runs still gets this GPU reset
            rc, out = smi(["-i", str(idx), "-lgc", f"{mhz},{mhz}"])
            if rc != 0:
                locked.pop()
                why = smi_reason(out)
                if "permission" in why and is_admin():  # root already: sudo cannot help, the machine itself says no
                    raise ClockLockError(f"cannot lock clocks on GPU {idx}: the driver refused even as "
                                         f"{'administrator' if os.name == 'nt' else 'root'}, so this machine does not allow "
                                         "clock changes (common in containers and cloud notebooks such as Kaggle and Colab); "
                                         "run without --lock-clocks, or on a machine where you control the GPU")
                raise ClockLockError(f"cannot lock clocks on GPU {idx}: {why}; run without --lock-clocks"
                                     + (" or with sudo" if "permission" in why else ""))
            log(f"GPU {idx}: SM clock locked at {mhz} MHz")
        yield {m.name: mhz for m, _, mhz in plan}
    finally:
        for idx in locked:
            rc, out = smi(["-i", str(idx), "-rgc"])
            if rc == 0:
                log(f"GPU {idx}: clocks unlocked (nvidia-smi -rgc)")
            else:
                print(f"joules compare: could not unlock GPU {idx}'s clocks ({smi_reason(out)}). "
                      f"Run: sudo nvidia-smi -i {idx} -rgc", file=sys.stderr)


def lock_text(locks: dict | None) -> str:
    return " / ".join(f"{v} MHz" for v in sorted(set(locks.values()))) if locks else ""


def measure(d: Discovery, cmd, shell: bool, interval: float, quiet: bool) -> dict:
    out = subprocess.DEVNULL if quiet else None
    with Sampler(d.meters, interval) as s:
        p = subprocess.Popen(cmd, shell=shell, stdin=subprocess.DEVNULL, stdout=out)
        for m in d.meters:
            if hasattr(m, "attach"):
                m.attach(p)
        while True:
            try:
                p.wait()
                break
            except KeyboardInterrupt:
                p.terminate()
                raise
    devices = {m.name: m.energy_j for m in d.meters}
    gpu = {m.name: m.health() for m in d.meters if hasattr(m, "health") and m.health()}
    e = sum(devices.values())
    r = {"duration_s": s.duration_s, "energy_j": e, "avg_w": e / s.duration_s if s.duration_s > 0 else 0.0,
         "devices": devices, "exit_code": p.returncode}
    if gpu:
        r["gpu"] = gpu
    return r


# ---------------------------------------------------------------- summary


IDLE_DRIFT = 0.10  # idle that moved more than this share between the start and the end gets a note


def idle_above(runs: list[dict], start_w: float, end_w: float, t0: float, t1: float) -> list[str]:
    """Energy above idle for each run, with idle taken on the straight line from the idle measured before the first
    run (at t0) to the one measured after the last (at t1), at the middle of the run (its "_mid", a monotonic time).
    A desktop that settles over a long comparison would otherwise have the starting idle taken off every run.
    Returns a note when idle moved by more than IDLE_DRIFT."""
    for r in runs:
        mid = r.pop("_mid", t0)
        w = start_w + (end_w - start_w) * min(1.0, max(0.0, (mid - t0) / (t1 - t0))) if t1 > t0 else start_w
        r["idle_w"] = w
        r["above_idle_j"] = max(0.0, r["energy_j"] - w * r["duration_s"])
    if start_w > 0 and abs(end_w - start_w) / start_w > IDLE_DRIFT:
        return [f"Idle moved from {start_w:.1f} W before the runs to {end_w:.1f} W after them ({(end_w - start_w) / start_w * 100:+.0f}%). "
                "Energy above idle takes idle on a straight line between the two for each run; the total energy does not "
                "depend on idle. Close other programs and let the machine settle (or idle longer) for a steadier baseline."]
    return []


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(runs: list[dict], names: list[str], baseline: str, cool_to: float | None = None,
              locks: dict | None = None) -> list[dict]:
    """One block per power limit (one block, limit None, without --limits): per command its statistics, and
    against the baseline the changes and p-values. locks: {GPU meter name: MHz} when the clocks were locked."""
    blocks = []
    for lim in sorted({r.get("limit_pct") for r in runs}, key=lambda v: -(v or 1000)):
        rows = {}
        for n in names:
            rs = [r for r in runs if r["name"] == n and r.get("limit_pct") == lim]
            if not rs:
                continue
            e = [r["energy_j"] for r in rs]
            s = [r["duration_s"] for r in rs]
            edp = [r["energy_j"] * r["duration_s"] for r in rs]
            row = {"name": n, "runs": len(rs), "energy_j": summarize_values(e), "duration_s": summarize_values(s),
                   "avg_w": summarize_values([r["avg_w"] for r in rs]), "edp_js": summarize_values(edp),
                   "_e": e, "_s": s, "_edp": edp}
            if all("above_idle_j" in r for r in rs):
                row["above_idle_j"] = summarize_values([r["above_idle_j"] for r in rs])
                row["_ai"] = [r["above_idle_j"] for r in rs]
            g = [h for r in rs for h in (r.get("gpu") or {}).values()]
            if g:
                reasons: dict[str, float] = {}
                for h in g:
                    for k, v in (h.get("throttle_reasons") or {}).items():
                        reasons[k] = reasons.get(k, 0) + v
                row["gpu"] = {"clock_pct": _med(h.get("clock_pct") for h in g), "gpu_util_pct": _med(h.get("gpu_util_pct") for h in g),
                              "mem_util_pct": _med(h.get("mem_util_pct") for h in g), "throttled_pct": _med(h.get("throttled_pct") for h in g),
                              "temp_c_max": max((h["temp_c_max"] for h in g if h.get("temp_c_max") is not None), default=None),
                              "reason": max(reasons, key=reasons.get) if reasons else None,
                              "sm_clock_mhz": _med(h.get("sm_clock_mhz") for h in g)}
                if locks:  # the average SM clock as a share of the clock it was locked at
                    row["gpu"]["vs_lock_pct"] = _med(h["sm_clock_mhz"] / locks[gn] * 100 for r in rs for gn, h in (r.get("gpu") or {}).items()
                                                     if h.get("sm_clock_mhz") and locks.get(gn))
            starts = [r["temp_start_c"] for r in rs if r.get("temp_start_c") is not None]
            if starts:
                row["temp_start_c"] = {"min": min(starts), "max": max(starts), "median": _med(starts)}
            rows[n] = row
        base = rows.get(baseline)
        for n, row in rows.items():
            if base is None or n == baseline:
                continue
            ch = lambda a, b: (a - b) / b * 100 if b else None
            row["vs"] = {"baseline": baseline,
                         "energy_pct": ch(row["energy_j"]["median"], base["energy_j"]["median"]),
                         "time_pct": ch(row["duration_s"]["median"], base["duration_s"]["median"]),
                         "edp_pct": ch(row["edp_js"]["median"], base["edp_js"]["median"]),
                         "energy_p": mann_whitney(row["_e"], base["_e"]), "time_p": mann_whitney(row["_s"], base["_s"]),
                         "edp_p": mann_whitney(row["_edp"], base["_edp"]),
                         "power_pct": ch(row["avg_w"]["median"], base["avg_w"]["median"])}
            if "above_idle_j" in row and "above_idle_j" in base:
                row["vs"]["above_idle_pct"] = ch(row["above_idle_j"]["median"], base["above_idle_j"]["median"])
                row["vs"]["above_idle_p"] = mann_whitney(row["_ai"], base["_ai"])
        for row in rows.values():
            for k in ("_e", "_s", "_edp", "_ai"):
                row.pop(k, None)
        blocks.append({"limit_pct": lim, "rows": list(rows.values()), "notes": fairness_notes(list(rows.values()), cool_to, locks)})
    return blocks


def min_p(n: int, m: int) -> float:
    """The smallest two-sided p-value Mann-Whitney can give with n and m runs: when every run of one beats every run of the other."""
    return min(1.0, 2 / math.comb(n + m, n))


HELD_PCT = 97  # an average SM clock below this share of the locked clock: the card did not hold the lock


def fairness_notes(rows: list[dict], cool_to: float | None = None, locks: dict | None = None) -> list[str]:
    """Say so when the commands did not get the same GPU: different throttling, clocks or starting temperatures.
    With locked clocks, whether the GPU held the lock: a power or thermal cap is then the only thing besides the
    work that can make the commands differ. And when there are too few runs for any p-value to come out below 0.05."""
    notes = []
    ns = [r["runs"] for r in rows]
    if len(ns) >= 2 and min_p(min(ns), min(ns)) > 0.05:
        notes.append(f"With {min(ns)} runs each, p cannot go below {min_p(min(ns), min(ns)):.2f} even for a real difference; "
                     "use --runs 5 or more for a result that can reach p < 0.01.")
    g = [(r["name"], r["gpu"]) for r in rows if r.get("gpu")]
    thr = [(n, x["throttled_pct"]) for n, x in g if x.get("throttled_pct") is not None]
    if locks:
        held = [(n, x) for n, x in g if x.get("vs_lock_pct") is not None]
        below = [(n, x) for n, x in held if x["vs_lock_pct"] < HELD_PCT]
        if below:
            why = lambda x: (f", throttled {x['throttled_pct']:.0f}% of the time" if x.get("throttled_pct") else "") + (f": {x['reason']}" if x.get("reason") else "")
            notes.append(f"Clocks were locked at {lock_text(locks)}, but the GPU still ran below that under "
                         + " and ".join(f"{n} (average {x['sm_clock_mhz']:.0f} MHz{why(x)})" for n, x in below)
                         + ": its power or thermal cap set the speed there, and that, not only the work, can make the commands "
                         "differ. Lock lower (--lock-clocks base) to hold one clock for every command.")
        elif held and len(held) == len(g):
            notes.append(f"Clocks held at {lock_text(locks)} under every command (the GPU never dropped below the lock), so the "
                         "difference is the work, not boost or throttling.")
    elif len(thr) >= 2 and max(v for _, v in thr) - min(v for _, v in thr) > 20:
        hi = max(thr, key=lambda t: t[1]); lo = min(thr, key=lambda t: t[1])
        notes.append(f"The GPU was throttled {hi[1]:.0f}% of the time under {hi[0]} but {lo[1]:.0f}% under {lo[0]}: part of the "
                     "difference may be the card's power or thermal limit rather than the work. Compare again at a fixed "
                     "power limit (--limits) or with locked clocks (--lock-clocks base, needs root).")
    clk = [(n, x["clock_pct"]) for n, x in g if x.get("clock_pct") is not None]
    if not locks and len(clk) >= 2 and max(v for _, v in clk) - min(v for _, v in clk) > 10 and not notes:
        notes.append("The GPU's clock differed by more than 10 points of its maximum between the commands; lock the clocks "
                     "(--lock-clocks base, needs root) to compare at the same speed.")
    # Where each command's runs started. Alternating runs give every command the same spread, which only widens the
    # IQR; a command that started hotter as a rule is a bias. --cool-to waits for a hot card to cool, never warms a
    # cool one, so once it is set the advice is more warm-up rounds rather than --cool-to again.
    st = [(r["name"], r["temp_start_c"]) for r in rows if r.get("temp_start_c")]
    med = [(n, t["median"]) for n, t in st if t.get("median") is not None]
    if len(med) >= 2 and max(m for _, m in med) - min(m for _, m in med) > 10:
        hi = max(med, key=lambda t: t[1]); lo = min(med, key=lambda t: t[1])
        notes.append(f"{hi[0]}'s runs started at {hi[1]:.0f} °C (median) but {lo[0]}'s at {lo[1]:.0f} °C, so one side began on a "
                     "hotter card. " + ("--cool-to only waits for a card to cool, so add warm-up rounds (--warmup 3) to bring "
                     "the card to its working temperature first." if cool_to is not None else
                     f"--cool-to {lo[1]:.0f} starts every run from no hotter than {lo[1]:.0f} °C."))
    elif st and max(t["max"] for _, t in st) - min(t["min"] for _, t in st) > 10:
        lo, hi = min(t["min"] for _, t in st), max(t["max"] for _, t in st)
        notes.append(f"The runs started between {lo:.0f} and {hi:.0f} °C, alike for every command since they alternate, so this "
                     "widens the spread rather than favouring one side. " + ("--cool-to only waits for a card to cool; add "
                     "warm-up rounds (--warmup 3) to start them closer." if cool_to is not None else
                     f"--cool-to {lo + 10:.0f} starts every run from a closer temperature."))
    return notes


# ---------------------------------------------------------------- output


def _fmt_j(j: float) -> str:
    for unit, k in (("MJ", 1e6), ("kJ", 1e3)):
        if j >= k:
            return f"{j / k:.3g} {unit}"
    return f"{j:.3g} J"


def _fmt_s(s: float) -> str:
    return f"{s / 60:.3g} min" if s >= 600 else f"{s:.3g} s"


def _pct(x: float | None) -> str:
    return "" if x is None else f"{x:+.1f}%"


def _p(p: float | None) -> str:
    if p is None:
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def verdict_line(row: dict) -> str:
    v = row["vs"]
    line = (f"{row['name']} vs {v['baseline']}: energy {_pct(v['energy_pct'])} ({_p(v['energy_p'])}), "
            f"time {_pct(v['time_pct'])} ({_p(v['time_p'])}), energy × time {_pct(v['edp_pct'])} ({_p(v['edp_p'])})")
    if v.get("power_pct") is not None:
        line += f", average power {_pct(v['power_pct'])}"
    if v.get("above_idle_pct") is not None:
        line += f", above idle {_pct(v['above_idle_pct'])} ({_p(v['above_idle_p'])})"
    return line


def render(r: dict) -> str:
    head = (f"joules compare · {r['runs']} run{'s' if r['runs'] != 1 else ''} each, {r['order']}"
            + (f", {r['warmup']} warm-up" if r["warmup"] else "") + f" · {clocks_text(r)}{', '.join(r['devices'])}")
    lines = [head]
    for i, n in enumerate(r["names"]):
        lines.append(f"  {n}: {r['commands'][i]}")
    for b in r["blocks"]:
        if b["limit_pct"] is not None:
            lines.append(f"  ── power limit {b['limit_pct']}% ({', '.join(f'{w:.0f} W' for w in b['limit_w'])})")
        w = max([len(x["name"]) for x in b["rows"]] + [8])
        has_gpu = any(x.get("gpu") for x in b["rows"])
        mhz = has_gpu and bool(r.get("locked_mhz"))
        lines.append(f"  {'':<{w}}  {'energy':>9}  {'IQR':>6}  {'time':>8}  {'avg W':>7}  {'E×t':>10}"
                     + (f"  {'clock':>5}" + (f"  {'MHz':>5}" if mhz else "") + f"  {'thrott.':>7}  {'max °C':>6}" if has_gpu else ""))
        for x in b["rows"]:
            g = x.get("gpu") or {}
            clock = "" if g.get("clock_pct") is None else f"{g['clock_pct']:.0f}%"
            thr = "" if g.get("throttled_pct") is None else f"{g['throttled_pct']:.0f}%"
            temp = "" if g.get("temp_c_max") is None else f"{g['temp_c_max']:.0f}"
            sm = "" if g.get("sm_clock_mhz") is None else f"{g['sm_clock_mhz']:.0f}"
            lines.append(f"  {x['name']:<{w}}  {_fmt_j(x['energy_j']['median']):>9}  {x['energy_j']['spread_pct']:>5.1f}%  "
                         f"{_fmt_s(x['duration_s']['median']):>8}  {x['avg_w']['median']:>7.1f}  {_fmt_j(x['edp_js']['median']) + '·s':>10}"
                         + (f"  {clock:>5}" + (f"  {sm:>5}" if mhz else "") + f"  {thr:>7}  {temp:>6}" if has_gpu else ""))
        for x in b["rows"]:
            if "vs" in x:
                lines.append("  " + verdict_line(x))
        for n in b["notes"]:
            lines.append("  ! " + n)
    lines.append("  medians; IQR is the spread of the middle half of the runs as a share of the median; p is the chance of a "
                 "difference this large if the commands were really the same (Mann-Whitney)")
    for n in r["notes"]:
        lines.append("  ! " + n)
    return "\n".join(lines)


def markdown(r: dict) -> str:
    out = [f"### Energy comparison ({r['runs']} runs each, {r['order']}, joules {r['tool'].split()[-1]})", "",
           f"Devices: {', '.join(r['devices'])}", ""]
    if r.get("locked_mhz"):
        out += [f"GPU clocks: SM clock locked at {lock_text(r['locked_mhz'])} (nvidia-smi -lgc) for every run, warm-ups included.", ""]
    elif r.get("gpu_clocks") == "boost":
        out += ["GPU clocks: not locked (the card boosts and throttles as it would in a real job).", ""]
    for i, n in enumerate(r["names"]):
        out.append(f"- **{n}**: `{r['commands'][i]}`")
    out.append("")
    for b in r["blocks"]:
        if b["limit_pct"] is not None:
            out += [f"**Power limit {b['limit_pct']}%** ({', '.join(f'{w:.0f} W' for w in b['limit_w'])})", ""]
        has_gpu = any(x.get("gpu") for x in b["rows"])
        mhz = has_gpu and bool(r.get("locked_mhz"))
        out += ["| | energy (median) | IQR | time (median) | avg power | energy × time |"
                + (" GPU clock |" + (" SM clock |" if mhz else "") + " throttled | max temp |" if has_gpu else ""),
                "|---|---|---|---|---|---|" + ("---|" * (4 if mhz else 3) if has_gpu else "")]
        for x in b["rows"]:
            g = x.get("gpu") or {}
            f = lambda k, fmt: "" if g.get(k) is None else fmt.format(g[k])
            out.append(f"| {x['name']} | {_fmt_j(x['energy_j']['median'])} | {x['energy_j']['spread_pct']:.1f}% | "
                       f"{_fmt_s(x['duration_s']['median'])} | {x['avg_w']['median']:.1f} W | {_fmt_j(x['edp_js']['median'])}·s |"
                       + (f" {f('clock_pct', '{:.0f}%')} |" + (f" {f('sm_clock_mhz', '{:.0f} MHz')} |" if mhz else "")
                          + f" {f('throttled_pct', '{:.0f}%')} | {f('temp_c_max', '{:.0f} °C')} |" if has_gpu else ""))
        out.append("")
        for x in b["rows"]:
            if "vs" in x:
                out.append(f"- {verdict_line(x)}")
        for n in b["notes"]:
            out.append(f"- ⚠️ {n}")
        out.append("")
    out.append("Medians over interleaved runs. p: Mann-Whitney U, two-sided. Measured with [joules](https://wattops.dev/joules).")
    return "\n".join(out) + "\n"


CSV_COLS = ["name", "limit_pct", "round", "position", "started_at", "duration_s", "energy_j", "above_idle_j", "avg_w",
            "temp_start_c", "gpu_clock_pct", "gpu_util_pct", "gpu_mem_util_pct", "gpu_throttled_pct", "gpu_temp_c_max", "exit_code",
            "gpu_sm_clock_mhz", "locked_mhz"]


def csv_rows(r: dict) -> list[list]:
    rows = [CSV_COLS + [f"J {d}" for d in r["devices"]]]
    for x in r["results"]:
        g = list((x.get("gpu") or {}).values())
        gm = lambda k: _med(h.get(k) for h in g) if g else None
        rows.append([x["name"], x.get("limit_pct"), x["round"], x["position"], x["started_at"], x["duration_s"], x["energy_j"],
                     x.get("above_idle_j"), x["avg_w"], x.get("temp_start_c"), gm("clock_pct"), gm("gpu_util_pct"),
                     gm("mem_util_pct"), gm("throttled_pct"),
                     max((h["temp_c_max"] for h in g if h.get("temp_c_max") is not None), default=None), x["exit_code"],
                     gm("sm_clock_mhz"), lock_text(r.get("locked_mhz")).replace(" MHz", "") or None]
                    + [x["devices"].get(d) for d in r["devices"]])
    return rows


def clocks_text(r: dict) -> str:
    """For the header line: whether the GPU clocks were locked (and where), or left to boost."""
    if r.get("locked_mhz"):
        return f"SM clock locked at {lock_text(r['locked_mhz'])} · "
    return "clocks not locked · " if r.get("gpu_clocks") == "boost" else ""


def log(msg: str) -> None:
    print("\033[2m" + "joules compare: " + msg + "\033[0m" if sys.stderr.isatty() else "joules compare: " + msg, file=sys.stderr)
