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
            if not row.get("name") and row.get("arm"):
                row["name"] = row["arm"]
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
            extra = {k: v for k, v in row.items() if k not in (*num, "name", "arm", "round", "position") and v != ""}
            if extra:
                r["fields"] = extra  # the harness's own columns, kept as text for --checks to read
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
    for r in runs:
        if "avg_w" not in r:
            r["avg_w"] = r["energy_j"] / r["duration_s"] if r.get("duration_s") else 0.0
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
    """Energy = average power x time. How much of the change came from each. Power here is median energy over
    median time, so the split adds up to exactly the energy change quoted in the headline."""
    t0, t1 = base["duration_s"]["median"], row["duration_s"]["median"]
    e0, e1 = base["energy_j"]["median"], row["energy_j"]["median"]
    if min(t0, t1, e0, e1) <= 0:
        return None
    tr, er = t1 / t0, e1 / e0
    pr = er / tr
    if abs(math.log(er)) < 1e-9:
        return None
    share_t = math.log(tr) / math.log(er)
    p0, p1 = e0 / t0, e1 / t1   # the joules each part accounts for, split symmetrically (no leftover term)
    time_j, power_j = (t1 - t0) * (p0 + p1) / 2, (p1 - p0) * (t0 + t1) / 2
    return {"time_pct": (tr - 1) * 100, "power_pct": (pr - 1) * 100, "energy_pct": (er - 1) * 100, "time_share": share_t,
            "time_j": time_j, "power_j": power_j}


def _fact_mismatches(facts: dict, name: str) -> list[str]:
    arm = {**(facts.get("all") or {}), **((facts.get("arms") or {}).get(name) or {})}
    req, act = arm.get("requested") or {}, arm.get("actual") or {}
    out = []
    for k, want in req.items():
        if k not in act:
            out.append(f"{k}: asked for {want}, but the facts give no actual value")
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


def load_checks(paths: list[str] | None) -> list:
    """Your own checks: Python files that define check(report, facts) and return a list of
    {"severity": "problem" | "note", "title": ..., "detail": ...}. They run on your machine, like any script you run,
    so only load files you trust. This is how workload knowledge that joules can't know (your harness, your
    parameters) becomes part of the explanation, without putting it in joules."""
    import importlib.util
    out = []
    for path in paths or []:
        spec = importlib.util.spec_from_file_location(f"joules_checks_{len(out)}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"--checks {path}: not a Python file")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except OSError as e:
            raise ValueError(f"--checks {path}: {e}")
        if not callable(getattr(mod, "check", None)):
            raise ValueError(f"--checks {path} must define check(report, facts)")
        out.append((path, mod.check))
    return out


def explain(report: dict, facts: dict | None = None, checks: list | None = None) -> dict:
    """The reasons behind a comparison, most important first: problems to fix before quoting a number, then notes."""
    names, baseline, all_runs = report["names"], report["baseline"], report["results"]
    reasons: list[dict] = []
    if facts:
        unknown = sorted(set(facts.get("arms") or {}) - set(names))
        if unknown:
            raise ValueError(f"--facts names {', '.join(unknown)}, which {'is' if len(unknown) == 1 else 'are'} not in this comparison "
                             f"({', '.join(names)})")
    multi = len(report["blocks"]) > 1
    for block in report["blocks"]:
        prefix = f"at {block['limit_pct']}% power: " if multi and block.get("limit_pct") is not None else ""
        runs = [r for r in all_runs if r.get("limit_pct") == block.get("limit_pct")]
        _explain_block(block, runs, names, baseline, facts if block is report["blocks"][0] else None, prefix, reasons, report.get("order"))

    from_checks: list[dict] = []
    for path, fn in checks or []:
        try:
            got = fn(report, facts or {}) or []
        except Exception as e:  # a broken check must not hide the rest of the explanation
            got = [{"severity": "note", "title": f"check {path} failed", "detail": f"{type(e).__name__}: {e}"}]
        for r in got:
            sev = r.get("severity") if r.get("severity") in ("problem", "note") else "note"
            from_checks.append({"severity": sev, "title": str(r.get("title", "check")), "detail": str(r.get("detail", "")),
                                "numbers": r.get("numbers") or {}, "source": path})
    reasons[:0] = from_checks  # your own workload checks come first: a wrong workload makes every other number moot

    if report.get("imported"):
        reasons.append({"severity": "note", "title": "imported runs", "numbers": {},
                        "detail": "these runs were measured outside joules compare; joules can only check what the CSV says."})
        if any(r.get("position") is None for r in all_runs):
            reasons.append({"severity": "note", "title": "run order unknown", "numbers": {},
                            "detail": "the CSV has no round and position columns, so joules cannot check whether the order was fair."})
    if not any(r.get("gpu") for r in all_runs) and not report.get("imported"):
        reasons.append({"severity": "note", "title": "power only", "numbers": {},
                        "detail": "this meter reports power only, not the GPU's clocks, temperature or throttling, so heat and "
                                  "throttling could not be checked. Rest or cool between runs to be safe (--rest, --cool-to)."})

    problems = [r for r in reasons if r["severity"] == "problem"]
    reasons = problems + [r for r in reasons if r["severity"] != "problem"]
    first = report["blocks"][0]
    return {"baseline": baseline, "summary": _summary({r["name"]: r for r in first["rows"]}, baseline, problems), "reasons": reasons}


GOOD_NEWS = ("Clocks held at", "alike for every command")  # compare notes that report fairness, not a problem


def _explain_block(block: dict, runs: list[dict], names: list[str], baseline: str, facts: dict | None, prefix: str,
                   reasons: list[dict], report_order: str | None = None) -> None:
    rows = {r["name"]: r for r in block["rows"]}
    base = rows.get(baseline)

    def add(severity: str, title: str, detail: str, **numbers):
        reasons.append({"severity": severity, "title": prefix + title, "detail": detail, "numbers": numbers})

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
                f"{row['runs']} runs. Something changes between runs (warm-up, caching, clocks or the workload itself), so this "
                f"arm's number isn't settled yet; more runs, or a look at what varies, would settle it.", arm=n, min_j=e["min"], max_j=e["max"], runs=row["runs"])

    # 3. fair conditions
    if not report_order and len(names) >= 3 and _fixed_order(runs, names):  # joules compare's own orders are fair; 2 arms A B A B interleave
        add("problem", "the commands always ran in the same order",
            "every round ran them in one fixed order, so a later command always started on a warmer, busier GPU. "
            "Alternate the order (joules compare does this) and rest or cool between runs (--rest, --cool-to).")
    for note in block.get("notes") or []:
        if "p cannot go below" in note:  # said once below, with what to do
            continue
        if any(g in note for g in GOOD_NEWS):
            add("note", "conditions", note)
        else:
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
    rest_rows = [r for r in others if r not in shown[:MAX_SPLITS]]
    for row in shown[:MAX_SPLITS]:
        s = _split(row, base)
        if not s:
            continue
        n = row["name"]
        same = lambda v: abs(v) < 1
        if same(s["power_pct"]):
            how = f"it {'finished sooner' if s['time_pct'] < 0 else 'took longer'} (time {_pct(s['time_pct'])}) at about the same power"
        elif same(s["time_pct"]):
            how = f"it took about the same time and drew {'less' if s['power_pct'] < 0 else 'more'} power (average {_pct(s['power_pct'])})"
        elif s["time_pct"] < 0 and s["power_pct"] <= 0:
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
    if rest_rows:
        rest = ", ".join(f"{r['name']} {_pct(r['vs']['energy_pct'])}" for r in rest_rows)
        add("note", "the other commands", f"against {baseline}: {rest}.")

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
        s += f" {len(problems)} thing{'s' if len(problems) > 1 else ''} to check before quoting it, below."
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


# ---------------------------------------------------------------- fit: where does one way start to win?

def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float | None]:
    """Least squares y = a + b x. Returns a, b and R² (None when the line goes exactly through every point)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot) if ss_tot > 0 and len(set(xs)) > 2 else None


def load_points(files: list[str], x_name: str | None) -> list[dict]:
    """Runs with a workload value x: from VALUE=report.json (a whole compare at one workload) or CSVs with an x column.
    With no x_name, every run gets x = 0 (one workload), and a plain report.json is accepted."""
    runs = []
    for f in files:
        if x_name is None and "=" not in f:
            rep = load(f)
            runs += [{**r, "x": 0.0, "_baseline": rep.get("baseline")} for r in rep["results"]]
            continue
        if "=" in f and f.split("=", 1)[1].lower().endswith(".json"):
            v, path = f.split("=", 1)
            try:
                x = float(v)
            except ValueError:
                raise ValueError(f"{f}: write the workload value before '=', e.g. 25={path}")
            rep = load(path)
            for r in rep["results"]:
                runs.append({**r, "x": x, "_baseline": rep["baseline"]})
        elif f.lower().endswith(".csv"):
            with open(f, newline="", encoding="utf-8") as fh:
                head = next(csv.reader(fh), [])
            if x_name is not None and x_name not in [h.strip() for h in head]:
                raise ValueError(f"{f} has no {x_name} column")
            with open(f, newline="", encoding="utf-8") as fh:
                xs = [float(row[x_name]) if x_name else 0.0 for row in csv.DictReader(fh)]
            for r, x in zip(read_csv(f), xs):
                runs.append({**r, "x": x})
        else:
            raise ValueError(f"{f}: give VALUE=report.json (a joules compare report) or a .csv with a {x_name} column")
    if not runs:
        raise ValueError("no runs to fit")
    return runs


def fit(runs: list[dict], x_name: str, baseline: str | None = None, at: list[float] | None = None) -> dict:
    """Fit energy = fixed + per_unit * x for each command over every run, and find where each one starts to beat
    the baseline. Uses energy above idle as well when every run has it."""
    names = list(dict.fromkeys(r["name"] for r in runs))
    if len(names) < 2:
        raise ValueError("fit needs runs of at least two commands")
    baseline = baseline or next((r["_baseline"] for r in runs if r.get("_baseline")), None) or names[-1]
    if baseline not in names:
        raise ValueError(f"--baseline must be one of {', '.join(names)}")
    xs_all = sorted({r["x"] for r in runs})
    if len(xs_all) < 2:
        raise ValueError(f"fit needs runs at two or more values of {x_name} (got only {xs_all[0]:g})")
    bases = ["energy_j"] + (["above_idle_j"] if all("above_idle_j" in r for r in runs) else [])
    out = {"x": x_name, "baseline": baseline, "measured": xs_all, "fits": {}, "crossovers": {}, "at": [], "notes": []}
    for basis in bases:
        lines = {}
        for n in names:
            rs = [r for r in runs if r["name"] == n]
            xs = [r["x"] for r in rs]
            if len(set(xs)) < 2:
                raise ValueError(f"{n} was measured at only one value of {x_name}")
            a, b, r2 = _ols(xs, [r[basis] for r in rs])
            meds = {}
            for x in sorted(set(xs)):
                v = sorted(r[basis] for r in rs if r["x"] == x)
                meds[x] = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
            lines[n] = {"fixed_j": a, "per_unit_j": b, "r2": r2, "runs": len(rs), "medians": meds}
        out["fits"][basis] = lines
        cb = lines[baseline]
        out["crossovers"][basis] = {}
        for n, ln in lines.items():
            if n == baseline:
                continue
            db = cb["per_unit_j"] - ln["per_unit_j"]
            da = ln["fixed_j"] - cb["fixed_j"]
            if abs(db) < 1e-12:
                c = {"x": None, "why": "same cost per unit: one is always cheaper by its fixed cost"}
            else:
                xc = da / db
                wins_above = db > 0
                c = {"x": xc, "wins": "above" if wins_above else "below", "extra_fixed_j": da, "saved_per_unit_j": db,
                     "inside": xs_all[0] <= xc <= xs_all[-1]}
            out["crossovers"][basis][n] = c
    for x in (at or []):
        row = {"x": x, "measured": x in xs_all, "inside": xs_all[0] <= x <= xs_all[-1], "energy_j": {}}
        for n, ln in out["fits"]["energy_j"].items():
            row["energy_j"][n] = ln["fixed_j"] + ln["per_unit_j"] * x
        out["at"].append(row)
    if len(xs_all) == 2:
        out["notes"].append(f"Only two values of {x_name} were measured, so each line goes exactly through them and "
                            "there is no check that the cost really grows in a straight line. Measure a third value, "
                            "ideally near the break-even, before quoting it.")
    for basis, cs in out["crossovers"].items():
        for n, c in cs.items():
            if c.get("x") is not None and not c["inside"]:
                out["notes"].append(f"The break-even for {n} ({x_name} ≈ {c['x']:.3g}{' above idle' if basis != 'energy_j' else ''}) "
                                    f"is outside the measured range ({xs_all[0]:g}–{xs_all[-1]:g}): it is extrapolated. "
                                    f"Measure near it to confirm.")
    for basis, lines in out["fits"].items():
        for n, ln in lines.items():
            if ln["r2"] is not None and ln["r2"] < 0.9:
                out["notes"].append(f"{n}'s energy does not follow a straight line in {x_name} well (R² = {ln['r2']:.2f}); "
                                    "treat the break-even as rough.")
    return out


def _j(v: float) -> str:
    return f"{v:,.0f} J" if abs(v) >= 100 else f"{v:.3g} J"


def render_fit(f: dict) -> str:
    x = f["x"]
    meas = ", ".join(f"{v:g}" for v in f["measured"])
    L = ["", f"Energy model:  energy = fixed + per-unit × {x}      (fitted on every run; measured at {x} = {meas})", ""]
    for basis, lines in f["fits"].items():
        title = "Total GPU energy" if basis == "energy_j" else "Energy above idle"
        L.append(f"  {title}")
        w = max(len(n) for n in lines)
        L.append(f"    {'':<{w}}   {'fixed':>10}   {'per 1 ' + x:>16}   fit")
        for n, ln in lines.items():
            fit_txt = "exact (2 points)" if ln["r2"] is None else f"R² {ln['r2']:.3f}"
            L.append(f"    {n:<{w}}   {_j(ln['fixed_j']):>10}   {_j(ln['per_unit_j']):>16}   {fit_txt}")
        L.append("")
    b = f["baseline"]
    for basis, cs in f["crossovers"].items():
        lab = "" if basis == "energy_j" else " (above idle)"
        for n, c in cs.items():
            if c.get("x") is None:
                L.append(f"Break-even{lab}: none. {c['why']}.")
                continue
            side = "above" if c["wins"] == "above" else "below"
            L.append(f"Break-even{lab}: {n} beats {b} {side} {x} ≈ {c['x']:.3g}")
            if c["wins"] == "above":
                L.append(f"  why: {n}'s fitted line starts {_j(c['extra_fixed_j'])} higher but rises {_j(c['saved_per_unit_j'])} less "
                         f"per 1 {x}, so the two lines cross at {c['x']:.3g}.")
            else:
                L.append(f"  why: {n}'s fitted line starts {_j(-c['extra_fixed_j'])} lower but rises {_j(-c['saved_per_unit_j'])} more "
                         f"per 1 {x}, so the two lines cross at {c['x']:.3g}.")
        L.append("")
    if f["at"]:
        L.append(f"Predicted at {x} values you asked for (--at):")
        for row in f["at"]:
            e = row["energy_j"]
            base = e[b]
            tag = "measured" if row["measured"] else "inside the measured range" if row["inside"] else "outside the measured range: no data there"
            parts = ", ".join(f"{n} {_j(v)} ({(v - base) / base * 100:+.0f}%)" for n, v in e.items() if n != b)
            L.append(f"  {x} = {row['x']:g}:  {b} {_j(base)};  {parts}   [{tag}]")
        L.append("")
    for i, n in enumerate(f["notes"], 1):
        L.append(f"! {n}")
    return "\n".join(L).rstrip()


# ---------------------------------------------------------------- cost model export: let a runtime pick by energy

def export_model(f: dict, routers: tuple = ()) -> dict:
    """A small, stable file a runtime can read to pick the strategy with the least energy at a given workload:
    predicted energy = fixed_j + per_unit_j * x, valid inside measured range (extrapolate with care)."""
    basis = "above_idle_j" if "above_idle_j" in f["fits"] else "energy_j"
    lines = {n: ln for n, ln in f["fits"][basis].items() if n not in routers}
    return {"schema": "joules-costmodel/1", "x": f["x"], "basis": basis, "measured_range": [f["measured"][0], f["measured"][-1]],
            "strategies": {n: {"fixed_j": ln["fixed_j"], "per_unit_j": ln["per_unit_j"], "r2": ln["r2"], "runs": ln["runs"]}
                           for n, ln in lines.items()},
            "switch_points": switch_points(lines, f["measured"][0], f["measured"][-1])}


def predict(model: dict, x: float) -> dict:
    return {n: s["fixed_j"] + s["per_unit_j"] * x for n, s in model["strategies"].items()}


def choose(model: dict, x: float) -> str:
    """The strategy with the least predicted energy at workload x."""
    p = predict(model, x)
    return min(p, key=p.get)


def switch_points(lines: dict, lo: float, hi: float) -> list[dict]:
    """Where the cheapest strategy changes, scanning the lower envelope of the lines from lo to hi (and a little
    beyond, marked)."""
    cands = sorted({lo, hi} | {(b["fixed_j"] - a["fixed_j"]) / (a["per_unit_j"] - b["per_unit_j"])
                               for a in lines.values() for b in lines.values()
                               if abs(a["per_unit_j"] - b["per_unit_j"]) > 1e-12})
    span = hi - lo or 1.0
    cands = [c for c in cands if lo - span <= c <= hi + span]
    best = lambda x: min(lines, key=lambda n: lines[n]["fixed_j"] + lines[n]["per_unit_j"] * x)
    out, prev = [], None
    pts = sorted(set(cands))
    for i in range(len(pts) + 1):
        x = pts[0] - 1e-9 if i == 0 else (pts[i - 1] + pts[i]) / 2 if i < len(pts) else pts[-1] + 1e-9
        b = best(x)
        if prev is not None and b != prev:
            at = pts[i - 1]
            out.append({"at": at, "from": prev, "to": b, "inside": lo <= at <= hi})
        prev = b
    return out


# ---------------------------------------------------------------- router report card

def router_card(runs: list[dict], router: str, against: list[str] | None = None, mix: dict | None = None,
                basis: str = "energy_j") -> dict:
    """How close a runtime's own strategy picker (the router) comes to the best fixed strategy at each workload,
    and over a mix of workloads. Regret = router energy over the best fixed strategy's, at the same workload."""
    names = list(dict.fromkeys(r["name"] for r in runs))
    if router not in names:
        raise ValueError(f"--router must be one of {', '.join(names)}")
    against = against or [n for n in names if n != router]
    missing = [n for n in against if n not in names]
    if missing:
        raise ValueError(f"--against names {', '.join(missing)}, not in the runs ({', '.join(names)})")
    if basis == "above_idle_j" and not all("above_idle_j" in r for r in runs):
        basis = "energy_j"

    def med(vals):
        v = sorted(vals)
        return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2

    xs = sorted({r.get("x", 0.0) for r in runs if r["name"] == router})
    rows, notes = [], []
    for x in xs:
        at = lambda n: [r[basis] for r in runs if r["name"] == n and r.get("x", 0.0) == x]
        rv = at(router)
        fixed = {n: med(at(n)) for n in against if at(n)}
        if not fixed:
            continue
        best = min(fixed, key=fixed.get)
        r_med = med(rv)
        rng = (max(rv) - min(rv)) / r_med if r_med else 0.0
        rows.append({"x": x, "router_j": r_med, "best": best, "best_j": fixed[best], "regret_pct": (r_med / fixed[best] - 1) * 100,
                     "fixed": fixed, "router_range": rng, "runs": len(rv)})
        if rng > UNSTABLE_RANGE:
            notes.append(f"at {x:g} the router's runs ranged {rng * 100:.0f}% of their median ({_fmt_j(min(rv))} to {_fmt_j(max(rv))}): "
                         "its choice may differ from run to run, so its average isn't settled yet.")
    if not rows:
        raise ValueError("no workload where both the router and a fixed strategy were measured")
    w = mix or {row["x"]: 1.0 for row in rows}
    known = [row for row in rows if row["x"] in w]
    if not known:
        raise ValueError("--mix names no workload that was measured")
    tw = sum(w[row["x"]] for row in known)
    total = lambda key: sum(w[row["x"]] * key(row) for row in known) / tw
    perfect = total(lambda r: r["best_j"])
    rtot = total(lambda r: r["router_j"])
    always = {n: total(lambda r, n=n: r["fixed"].get(n, float("nan"))) for n in against
              if all(n in r["fixed"] for r in known)}
    return {"router": router, "against": against, "basis": basis, "rows": rows, "mix": {f"{k:g}": v for k, v in w.items()},
            "router_j": rtot, "perfect_j": perfect, "regret_pct": (rtot / perfect - 1) * 100,
            "always": {n: {"energy_j": v, "router_vs_pct": (rtot / v - 1) * 100} for n, v in always.items()}, "notes": notes}


def render_router(c: dict, x_name: str = "x") -> str:
    L = ["", f"Router report card: {c['router']}   (against {', '.join(c['against'])}; {'above idle' if c['basis'] != 'energy_j' else 'total energy'})", ""]
    L.append(f"  {x_name:>10}   {'router':>10}   {'best fixed strategy':<28}   regret")
    for r in c["rows"]:
        L.append(f"  {r['x']:>10g}   {_fmt_j(r['router_j']):>10}   {(r['best'] + ' ' + _fmt_j(r['best_j'])):<28}   {r['regret_pct']:+.1f}%")
    L.append("")
    reg = c["regret_pct"]
    verdict = ("matches the best choice" if abs(reg) < 0.5 else f"within {reg:.1f}% of perfect" if reg < 5 else
               f"{reg:.0f}% above perfect: it often picks the wrong strategy" if reg >= 5 else "")
    mix = ", ".join(f"{k}: {v:g}" for k, v in c["mix"].items())
    L.append(f"Over the mix ({x_name} {mix}): router {_fmt_j(c['router_j'])} vs perfect choice {_fmt_j(c['perfect_j'])} "
             f"→ {reg:+.1f}% ({verdict}).")
    for n, a in c["always"].items():
        L.append(f"  vs always {n}: {_fmt_j(a['energy_j'])} → the router uses {_less_more(a['router_vs_pct'])} energy")
    for n in c["notes"]:
        L.append(f"! {n}")
    return "\n".join(L).rstrip()
