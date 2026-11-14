"""Why a comparison came out the way it did.

`joules compare` says how much energy each command used and whether the difference is real. This module says why:
whether the saving came from finishing sooner or from drawing less power, how idle power moves the percentages, whether
the runs were fair (order, heat, throttling), whether an arm is too noisy to quote, and whether the work each command did
was the work you asked for (from an optional facts file you write, since only you know the workload).

It works on a saved `joules compare` report, or on a CSV of runs measured some other way (one row per run:
name, energy_j, duration_s, and optionally round, position, temp_start_c, idle_w, above_idle_j).
"""

from __future__ import annotations

import csv
import json
import math

from .compare import mann_whitney, min_p, summarize

UNSTABLE_RANGE = 0.30   # (max - min) / median above this: the arm is too noisy to quote
UNSTABLE_IQR = 20.0     # or an interquartile spread above this many % of the median
SIGNIFICANT_P = 0.05
IDLE_GAP_PTS = 5.0      # total and above-idle changes this far apart: explain the difference
MAX_SPLITS = 4          # arms explained one by one; the rest are listed
FACT_TOLERANCE = 0.10   # a requested and an actual workload number more than 10% apart is a mismatch


# ---------------------------------------------------------------- input

def load(path: str, baseline: str | None = None) -> dict:
    """A compare report (.json) or a CSV of runs, as a compare-shaped report."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("kind") != "compare":
            raise ValueError(f"{path} is not a joules compare report (save one with joules compare --out FILE.json)")
        if baseline:
            if baseline not in r["names"]:
                raise ValueError(f"--baseline must be one of {', '.join(r['names'])}")
            r = dict(r, baseline=baseline, blocks=summarize(r["results"], r["names"], baseline, r.get("cool_to_c"), r.get("locked_mhz")))
        return r
    return from_runs(read_csv(path), baseline)


def read_csv(path: str) -> list[dict]:
    num = ("energy_j", "duration_s", "temp_start_c", "idle_w", "above_idle_j")
    runs = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if not row.get("name") or not row.get("energy_j") or not row.get("duration_s"):
                raise ValueError(f"{path} line {i}: every row needs name, energy_j and duration_s")
            r = {"name": row["name"]}
            for k in num:
                if row.get(k):
                    r[k] = float(row[k])
            for k in ("round", "position"):
                if row.get(k):
                    r[k] = int(float(row[k]))
            if r["duration_s"] <= 0:
                raise ValueError(f"{path} line {i}: duration_s must be above 0")
            r["avg_w"] = r["energy_j"] / r["duration_s"]
            if "above_idle_j" not in r and "idle_w" in r:
                r["above_idle_j"] = max(0.0, r["energy_j"] - r["idle_w"] * r["duration_s"])
            runs.append(r)
    if not runs:
        raise ValueError(f"{path} has no runs")
    return runs


def from_runs(runs: list[dict], baseline: str | None = None) -> dict:
    names = list(dict.fromkeys(r["name"] for r in runs))
    if len(names) < 2:
        raise ValueError("explain needs runs of at least two commands to compare")
    baseline = baseline or names[0]
    if baseline not in names:
        raise ValueError(f"--baseline must be one of {', '.join(names)}")
    if not any("above_idle_j" in r for r in runs) or not all("above_idle_j" in r for r in runs):
        for r in runs:
            r.pop("above_idle_j", None)
    return {"kind": "compare", "names": names, "baseline": baseline, "order": None, "results": runs,
            "blocks": summarize(runs, names, baseline), "imported": True}


# ---------------------------------------------------------------- reasons

def _pct(x: float) -> str:
    return f"{x:+.0f}%" if abs(x) >= 1 else f"{x:+.1f}%"


def _less_more(x: float) -> str:
    return f"{abs(x):.0f}% {'less' if x < 0 else 'more'}"


def _fixed_order(runs: list[dict], names: list[str]) -> bool:
    """True when every command always ran in the same slot of its round (e.g. A B C, A B C, ...)."""
    pos = {}
    for r in runs:
        if r.get("position") is None:
            return False
        pos.setdefault(r["name"], set()).add(r["position"])
    rounds = {r.get("round") for r in runs}
    return len(rounds) > 1 and len(pos) == len(names) and all(len(v) == 1 for v in pos.values())


def _split(row: dict, base: dict) -> dict | None:
    """Energy = average power x time. How much of the change came from each."""
    t0, t1 = base["duration_s"]["median"], row["duration_s"]["median"]
    p0, p1 = base["avg_w"]["median"], row["avg_w"]["median"]
    if min(t0, t1, p0, p1) <= 0:
        return None
    tr, pr = t1 / t0, p1 / p0
    er = tr * pr
    if abs(math.log(er)) < 1e-9:
        return None
    share_t = math.log(tr) / math.log(er)
    return {"time_pct": (tr - 1) * 100, "power_pct": (pr - 1) * 100, "energy_pct": (er - 1) * 100, "time_share": share_t}


def _fact_mismatches(facts: dict, name: str) -> list[str]:
    arm = {**(facts.get("all") or {}), **((facts.get("arms") or {}).get(name) or {})}
    req, act = arm.get("requested") or {}, arm.get("actual") or {}
    out = []
    for k, want in req.items():
        if k not in act:
            continue
        got = act[k]
        try:
            want_f, got_f = float(want), float(got)
        except (TypeError, ValueError):
            if str(want) != str(got):
                out.append(f"{k}: asked for {want}, got {got}")
            continue
        if want_f == got_f:
            continue
        rel = abs(got_f - want_f) / abs(want_f) if want_f else math.inf
        if rel > FACT_TOLERANCE:
            out.append(f"{k}: asked for {want}, got {got}" + (" (none at all)" if got_f == 0 else f" ({rel * 100:.0f}% off)"))
    return out


def explain(report: dict, facts: dict | None = None) -> dict:
    """The reasons behind a comparison, most important first: problems to fix before quoting a number, then notes."""
    names, baseline, runs = report["names"], report["baseline"], report["results"]
    block = report["blocks"][0]
    rows = {r["name"]: r for r in block["rows"]}
    base = rows.get(baseline)
    reasons: list[dict] = []

    def add(severity: str, title: str, detail: str, **numbers):
        reasons.append({"severity": severity, "title": title, "detail": detail, "numbers": numbers})

    # 1. was the work the work you asked for?
    if facts:
        for n in names:
            bad = _fact_mismatches(facts, n)
            if bad:
                hint = ((facts.get("arms") or {}).get(n) or {}).get("hint") or (facts.get("all") or {}).get("hint")
                add("problem", f"{n}: the workload was not what you asked for",
                    "; ".join(bad) + ". The energy number measures the work that actually ran, so it does not answer the "
                    "question you asked." + (f" {hint}" if hint else ""), arm=n, mismatches=bad)

    # 2. arms too noisy to quote
    for n, row in rows.items():
        e = row["energy_j"]
        rng = (e["max"] - e["min"]) / e["median"] if e["median"] else 0.0
        if row["runs"] >= 2 and (rng > UNSTABLE_RANGE or e.get("spread_pct", 0) > UNSTABLE_IQR):
            add("problem", f"{n} is not stable yet",
                f"its runs ranged from {_fmt_j(e['min'])} to {_fmt_j(e['max'])} ({rng * 100:.0f}% of its median) over "
                f"{row['runs']} runs. Something changes between runs (warm-up, caching, clocks or the workload itself). "
                f"Find it, or run at least 5 more times, before quoting this arm.", arm=n, min_j=e["min"], max_j=e["max"], runs=row["runs"])

    # 3. fair conditions
    if _fixed_order(runs, names):
        add("problem", "the commands always ran in the same order",
            "every round ran them in one fixed order, so a later command always started on a warmer, busier GPU. "
            "Alternate the order (joules compare does this) and rest or cool between runs (--rest, --cool-to).")
    for note in block.get("notes") or []:
        if "p cannot go below" in note:  # said once below, with what to do
            continue
        add("problem", "the commands did not run under the same conditions", note)

    # 4. per arm against the baseline: is it real, where did it come from, what does idle do to it
    others = sorted((r for n, r in rows.items() if r.get("vs") and base is not None), key=lambda r: r["vs"]["energy_pct"])
    unclear = [r for r in others if r["vs"].get("energy_p") is None or r["vs"]["energy_p"] > SIGNIFICANT_P]
    if unclear:
        floor = max(min_p(r["runs"], base["runs"]) for r in unclear)
        if floor > SIGNIFICANT_P:
            n_runs = min(r["runs"] for r in unclear + [base])
            add("problem", "too few runs to call any difference real",
                f"with {n_runs} runs per command the smallest p-value possible is {floor:.2g}, so even a big gap could be luck. "
                f"Run 5 or more of each (5 against 5 can reach p = {min_p(5, 5):.2g}).", runs=n_runs, p_floor=floor)
        else:
            for r in unclear:
                p = r["vs"].get("energy_p")
                add("problem", f"{r['name']} vs {baseline}: the difference could be noise",
                    f"energy {_pct(r['vs']['energy_pct'])}, p = {p:.2g}. Run more times, or find what makes the runs vary."
                    if p is not None else f"energy {_pct(r['vs']['energy_pct'])}; too few runs for a p-value.", arm=r["name"], p=p)

    shown = [r for r in others if abs(r["vs"]["energy_pct"]) >= 5]
    for row in shown[:MAX_SPLITS]:
        s = _split(row, base)
        if not s:
            continue
        n = row["name"]
        if s["time_pct"] < 0 and s["power_pct"] <= 0:
            how = f"it finished sooner (time {_pct(s['time_pct'])}) and drew less power (average {_pct(s['power_pct'])})"
        elif s["time_pct"] < 0:
            how = f"it finished sooner (time {_pct(s['time_pct'])}) while drawing more power (average {_pct(s['power_pct'])})"
        elif s["power_pct"] < 0:
            how = f"it drew less power (average {_pct(s['power_pct'])}) but took longer (time {_pct(s['time_pct'])})"
        else:
            how = f"it took longer (time {_pct(s['time_pct'])}) and drew more power (average {_pct(s['power_pct'])})"
        share = s["time_share"]
        if not 0 <= share <= 1:
            main_txt = " Time and power pulled in opposite directions; the bigger one won."
        elif 0.4 <= share <= 0.6:
            main_txt = " Time and power each explain about half."
        else:
            main_txt = f" About {max(share, 1 - share) * 100:.0f}% of the change came from {'finishing time' if share > 0.5 else 'power draw'}."
        add("note", f"{n} used {_less_more(s['energy_pct'])} energy than {baseline}", f"{how}.{main_txt}", arm=n, **s)
    if len(shown) > MAX_SPLITS:
        rest = ", ".join(f"{r['name']} {_pct(r['vs']['energy_pct'])}" for r in shown[MAX_SPLITS:])
        add("note", "the other commands", f"{rest} (details with --json).")

    idle_gap = [r for r in others if "above_idle_pct" in r["vs"] and abs(r["vs"]["above_idle_pct"] - r["vs"]["energy_pct"]) > IDLE_GAP_PTS]
    if idle_gap:
        ex_ = idle_gap[0]["vs"]
        ib = sorted(r["idle_w"] for r in runs if r["name"] == baseline and r.get("idle_w"))
        idle_txt = f" Idle is about {ib[len(ib) // 2]:.0f} W here." if ib else ""
        add("note", "why the total and above-idle numbers differ",
            f"e.g. {idle_gap[0]['name']}: total {_pct(ex_['energy_pct'])}, above idle {_pct(ex_['above_idle_pct'])}. Both are right. "
            "Idle power is paid whatever runs, so it shrinks every change in the total. Quote above idle for what the code itself "
            "saves, and the total for what the machine's bill saves." + idle_txt)

    for row in others:
        g, gb = row.get("gpu") or {}, base.get("gpu") or {}
        if g.get("mem_util_pct") is not None and gb.get("mem_util_pct") is not None and gb["mem_util_pct"] - g["mem_util_pct"] >= 10:
            add("note", f"{row['name']} kept the GPU's memory less busy",
                f"memory busy {g['mem_util_pct']:.0f}% of the time vs {gb['mem_util_pct']:.0f}% for {baseline}. When the work is limited "
                "by memory traffic, moving less data is where the saving comes from, so expect it to shrink as the share of data "
                "that must move grows.", arm=row["name"])

    if report.get("imported"):
        add("note", "imported runs", "these runs were measured outside joules compare; joules can only check what the CSV says.")

    problems = [r for r in reasons if r["severity"] == "problem"]
    reasons = problems + [r for r in reasons if r["severity"] != "problem"]
    return {"baseline": baseline, "summary": _summary(rows, baseline, problems), "reasons": reasons}


def _summary(rows: dict, baseline: str, problems: list) -> str:
    others = [r for n, r in rows.items() if n != baseline and r.get("vs")]
    if not others:
        return "Nothing to compare."
    best = min(others, key=lambda r: r["vs"]["energy_pct"])
    vs = best["vs"]
    s = f"{best['name']} used {_less_more(vs['energy_pct'])} energy than {baseline}"
    if "above_idle_pct" in vs:
        s += f" ({_less_more(vs['above_idle_pct'])} above idle)"
    s += "."
    if problems:
        s += f" {len(problems)} thing{'s' if len(problems) > 1 else ''} to fix before quoting it, below."
    return s


def _fmt_j(j: float) -> str:
    return f"{j * 1000:.2f} mJ" if j < 1 else f"{j:.2f} J" if j < 1000 else f"{j / 1000:.2f} kJ"


def render(ex: dict) -> str:
    lines = ["", "Why", ex["summary"], ""]
    for i, r in enumerate(ex["reasons"], 1):
        mark = "!" if r["severity"] == "problem" else "-"
        lines.append(f"{i}. {mark} {r['title']}")
        lines.append(f"   {r['detail']}")
    return "\n".join(lines)
