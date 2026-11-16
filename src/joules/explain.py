"""Why a comparison came out the way it did.

`joules compare` says how much energy each command used and whether the difference is real. This module says why:
whether the saving came from finishing sooner or from drawing less power, how idle power moves the percentages, whether
the runs were fair (order, heat, throttling), whether an arm is too noisy to quote, and whether the work each command did
was the work you asked for (from an optional facts file you write, since only you know the workload).

It works on a saved `joules compare` report, or on a CSV of runs measured some other way (one row per run:
name, energy_j, duration_s, and optionally round, position, limit_pct, temp_start_c, idle_w, above_idle_j).
"""

from __future__ import annotations

import contextlib
import copy
import csv
import json
import math
import os
import sys

from .compare import mann_whitney, min_p, summarize

UNSTABLE_RANGE = 0.30   # (max - min) / median above this: the arm is too noisy to quote
UNSTABLE_IQR = 20.0     # or an interquartile spread above this many % of the median
SIGNIFICANT_P = 0.05
IDLE_GAP_PTS = 5.0      # total and above-idle changes this far apart: explain the difference
MAX_SPLITS = 4          # arms explained one by one; the rest are listed
FACT_TOLERANCE = 0.10   # a requested and an actual workload number more than 10% apart is a mismatch


# ---------------------------------------------------------------- input

def load(path: str, baseline: str | None = None) -> dict:
    """A compare report (.json, or JSON under any name: joules compare --out writes it wherever it is told) or a CSV
    of runs, as a compare-shaped report."""
    if path.lower().endswith(".json") or _looks_like_json(path):
        with open(path, "rb") as f:
            try:
                r = read_json(f.read())
            except RecursionError:  # nested thousands deep: not a report, and not worth a traceback
                raise ValueError(f"{path} is nested too deeply to be a joules compare report") from None
            except ValueError as e:  # bad JSON, or not UTF-8/16/32 text: name the file, as fit and router read several
                raise ValueError(f"{path}: not readable JSON ({e})") from None
        if not isinstance(r, dict) or r.get("kind") != "compare":
            raise ValueError(f"{path} is not a joules compare report (save one with joules compare --out FILE.json)")
        if not r.get("names") or not r.get("results") or not r.get("blocks") or not r.get("baseline"):
            raise ValueError(f"{path} has no runs to explain (it needs names, baseline, results and blocks)")
        for i, run in enumerate(r["results"], start=1):  # an edited report: a clean error, not a traceback or NaN in --json
            for k in ("energy_j", "duration_s", "above_idle_j"):
                v = run.get(k) if isinstance(run, dict) else None
                if (k in ("energy_j", "duration_s") or v is not None) and (
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
                    raise ValueError(f"{path}: run {i} needs {k} as a finite number, 0 or more (got {v!r})")
        if baseline:
            if baseline not in r["names"]:
                raise ValueError(f"--baseline must be one of {', '.join(r['names'])}")
            r = dict(r, baseline=baseline, blocks=summarize(r["results"], r["names"], baseline, r.get("cool_to_c"), r.get("locked_mhz")))
        return r
    return from_runs(read_csv(path), baseline)


def read_json(data: bytes):
    """JSON in UTF-8 (with or without a BOM), UTF-16 or UTF-32: Windows PowerShell 5.1's > writes UTF-16."""
    return json.loads(data)


def _looks_like_json(path: str) -> bool:
    """A file that starts as a JSON object ("{" and then a quoted key) is not a CSV, whatever its name (a CSV header
    can start with "{", but not with "{" and a quote)."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        text = head.decode(json.detect_encoding(head), "ignore").lstrip()
        return text.startswith("{") and text[1:].lstrip()[:1] in ('"', "}")
    except (OSError, ValueError, LookupError):  # missing, a folder, not text: the CSV reader says why
        return False


def read_csv(path: str) -> list[dict]:
    try:
        return _read_csv(path)
    except csv.Error as e:  # e.g. a field over the csv module's 128 KiB limit (e.g. a column holding a log)
        raise ValueError(f"{path}: not a readable CSV ({e})") from None
    except UnicodeDecodeError:  # e.g. Excel's plain "CSV" on Windows, saved in the system code page
        raise ValueError(f"{path}: not UTF-8 text; save it as UTF-8 (in Excel: CSV UTF-8)") from None


def _csv_encoding(path: str) -> str:
    """UTF-8 (with or without a BOM) or UTF-16/32: Windows PowerShell 5.1's > writes UTF-16."""
    with open(path, "rb") as f:
        return json.detect_encoding(f.read(4))


def _read_csv(path: str) -> list[dict]:
    num = ("energy_j", "duration_s", "temp_start_c", "idle_w", "above_idle_j")
    runs = []
    with open(path, newline="", encoding=_csv_encoding(path)) as f:
        reader = csv.DictReader(f)
        head = [h.strip() for h in reader.fieldnames or [] if h and h.strip()]
        dup = sorted({h for h in head if head.count(h) > 1})
        if dup:  # csv.DictReader keeps only the last of two same-named columns: never guess which one was meant
            raise ValueError(f"{path}: the column{'s' if len(dup) > 1 else ''} {', '.join(dup)} appear{'' if len(dup) > 1 else 's'} more than once")
        for row in reader:
            i = reader.line_num  # the file's own line: DictReader skips blank lines, so a count of rows would be off
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k and k.strip()}
            if not row.get("name") and row.get("arm"):
                row["name"] = row["arm"]
            if not row.get("name") or not row.get("energy_j") or not row.get("duration_s"):
                raise ValueError(f"{path} line {i}: every row needs name, energy_j and duration_s")
            r = {"name": row["name"]}
            try:
                for k in num:
                    if row.get(k):
                        r[k] = float(row[k])
                        if not math.isfinite(r[k]) or (k != "temp_start_c" and r[k] < 0):
                            raise ValueError(f"{k} must be a finite number, 0 or more (got {row[k]})")
                for k in ("round", "position", "limit_pct"):  # limit_pct: joules compare --csv writes it with --limits
                    if row.get(k):
                        v = float(row[k])
                        if not v.is_integer():
                            raise ValueError(f"{k} must be a whole number (got {row[k]})")
                        r[k] = int(v)
            except (ValueError, OverflowError) as e:
                raise ValueError(f"{path} line {i}: {e}") from None
            if r["duration_s"] <= 0:
                raise ValueError(f"{path} line {i}: duration_s must be above 0")
            extra = {k: v for k, v in row.items() if k not in (*num, "name", "arm", "round", "position", "limit_pct") and v != ""}
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
    try:
        blocks = summarize(runs, names, baseline, imported=True)  # no joules compare flags for runs it did not make
    except OverflowError:
        raise ValueError("the energy or time values are too large to add up (over 1e308)") from None
    return {"kind": "compare", "names": names, "baseline": baseline, "order": None, "results": runs,
            "blocks": blocks, "imported": True}


# ---------------------------------------------------------------- reasons

def _mag(x: float) -> str:
    """abs(x) as a percentage figure: whole above 1, one decimal below; a fall of 99.5% or more that is not all of
    it shows two decimals, cut rather than rounded, so 1 J against 2 kJ is never "100% less" (that is no energy at all)."""
    a = round(abs(x), 9)  # float noise: an exact doubling computes as 99.99999999999999
    if 99.5 <= a < 100:
        return f"{math.floor(a * 100) / 100:.2f}"
    return f"{a:.0f}" if a >= 1 else f"{a:.1f}"


def _pct(x: float) -> str:
    return f"{'-' if x < 0 else '+'}{_mag(x)}%"


def _less_more(x: float) -> str:
    return f"{_mag(x)}% {'less' if x < 0 else 'more'}"


def _used(x: float, baseline: str) -> str:
    """"used 12% less energy than old", as precise as the "the other commands" list; "about the same" below 0.05%."""
    return f"used about the same energy as {baseline}" if abs(x) < 0.05 else f"used {_less_more(x)} energy than {baseline}"


def _fixed_order(runs: list[dict], names: list[str]) -> bool:
    """True when every command always ran in the same slot of its round (e.g. A B C, A B C, ...)."""
    pos = {}
    for r in runs:
        if r.get("position") is None:
            return False
        pos.setdefault(r["name"], set()).add(r["position"])
    repeated = len(runs) > len({r["name"] for r in runs})  # some command ran more than once: there were rounds
    slots = {p for v in pos.values() for p in v}  # every run alone in slot 1 of its round: no command ran later than another
    return repeated and len(slots) > 1 and len(pos) == len(names) and all(len(v) == 1 for v in pos.values())


def _constant_predecessor(runs: list[dict], names: list[str]) -> dict:
    """{name: (predecessor, rounds it followed it, rounds it ran)} for commands that ran right after the same other
    command in every round where they did not run first (e.g. a rotation that shifts one slot per round): they
    started on a GPU just warmed by the same work."""
    by_round = {}
    for r in runs:
        if r.get("round") is None or r.get("position") is None:
            return {}
        by_round.setdefault(r["round"], []).append(r)
    if len(by_round) < 2:
        return {}
    before, ran = {}, {}
    for rs in by_round.values():
        rs = sorted(rs, key=lambda r: r["position"])
        for n in {r["name"] for r in rs}:
            ran[n] = ran.get(n, 0) + 1
        for prev, cur in zip(rs, rs[1:]):
            before.setdefault(cur["name"], []).append(prev["name"])
    # the same predecessor in every round where it had one (in a rotation it is first in at most one round)
    return {n: (p[0], len(p), ran[n]) for n, p in before.items() if len(p) >= max(2, len(by_round) - 1) and len(set(p)) == 1}


def _split(row: dict, base: dict) -> dict | None:
    """Energy = average power x time. How much of the change came from each. Power here is median energy over
    median time, so the split adds up to exactly the energy change quoted in the headline."""
    t0, t1 = base["duration_s"]["median"], row["duration_s"]["median"]
    e0, e1 = base["energy_j"]["median"], row["energy_j"]["median"]
    if min(t0, t1, e0, e1) <= 0:
        return None
    tr, er = t1 / t0, e1 / e0
    if not (0 < tr < math.inf and 0 < er < math.inf):  # a ratio beyond what a float holds: no split to give
        return None
    pr = er / tr
    if abs(math.log(er)) < 1e-9:
        return None
    share_t = math.log(tr) / math.log(er)
    p0, p1 = e0 / t0, e1 / t1   # the joules each part accounts for, split symmetrically (no leftover term)
    time_j, power_j = (t1 - t0) * (p0 + p1) / 2, (p1 - p0) * (t0 + t1) / 2
    # each change as joules compare and the summary work it out, (new - old) / old: (ratio - 1) x 100 can round the other
    # way at a .5 (120 J to 111 J in the same time: 7% less here, 8% less in the summary)
    time_pct, power_pct, energy_pct = (t1 - t0) / t0 * 100, (p1 - p0) / p0 * 100, (e1 - e0) / e0 * 100
    if not all(math.isfinite(v) for v in (pr, p0, p1, time_j, power_j, time_pct, power_pct, energy_pct)):  # beyond a float: no split, and no NaN in --json
        return None
    return {"time_pct": time_pct, "power_pct": power_pct, "energy_pct": energy_pct, "time_share": share_t,
            "time_j": time_j, "power_j": power_j}


def _fact_mismatches(facts: dict, name: str) -> list[str]:
    common, own = facts.get("all") or {}, (facts.get("arms") or {}).get(name) or {}
    req = {**(common.get("requested") or {}), **(own.get("requested") or {})}
    act = {**(common.get("actual") or {}), **(own.get("actual") or {})}
    out = []
    for k, want in req.items():
        if k not in act:
            out.append(f"{k}: asked for {want}, but the facts give no actual value")
            continue
        got = act[k]
        is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        try:  # numbers within FACT_TOLERANCE match; text (a version "12.1", a mode) must match exactly
            if not (is_num(want) or is_num(got)):
                raise ValueError
            want_f, got_f = float(want), float(got)
        except (TypeError, ValueError, OverflowError):
            # sort_keys: {"h": 1, "w": 2} and {"w": 2, "h": 1} are the same object, in whatever order a harness wrote it
            txt = lambda v: v if isinstance(v, str) else json.dumps(v, default=str, sort_keys=True)
            if not _same_fact(want, got):
                out.append(f"{k}: asked for {txt(want)}, got {txt(got)}")
            continue
        if want_f == got_f or (math.isnan(want_f) and math.isnan(got_f)):
            continue
        if not (math.isfinite(want_f) and math.isfinite(got_f)):
            out.append(f"{k}: asked for {want}, got {got}")
            continue
        rel = round(abs(got_f - want_f) / abs(want_f), 9) if want_f else math.inf  # float noise: 0.3 to 0.33 is 10%, not more
        if rel > FACT_TOLERANCE:
            out.append(f"{k}: asked for {want}, got {got}" + (" (none at all)" if got_f == 0 else
                                                              " (asked for none)" if want_f == 0 else f" ({rel * 100:.0f}% off)"))
    return out


def _same_fact(want, got) -> bool:
    """A nested value (a shape, a config object) matches as a top-level one does: numbers within FACT_TOLERANCE
    ([64, 64] and [64.0, 64.0] are the same shape), objects whatever their key order, anything else as text (true = "true")."""
    txt = lambda v: v if isinstance(v, str) else json.dumps(v, default=str, sort_keys=True)
    if isinstance(want, dict) and isinstance(got, dict):
        return want.keys() == got.keys() and all(_same_fact(want[k], got[k]) for k in want)
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(_same_fact(w, g) for w, g in zip(want, got))
    if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (want, got)):
        try:
            w, g = float(want), float(got)
        except (TypeError, ValueError, OverflowError):
            return txt(want) == txt(got)
        if w == g or (math.isnan(w) and math.isnan(g)):
            return True
        return math.isfinite(w) and math.isfinite(g) and w != 0 and round(abs(g - w) / abs(w), 9) <= FACT_TOLERANCE
    return txt(want) == txt(got)


def check_facts(facts: dict, names: list[str]) -> None:
    """Raise ValueError when the facts file names an unknown command or has the wrong shape."""
    arms, common = facts.get("arms") or {}, facts.get("all") or {}
    if not isinstance(arms, dict) or not isinstance(common, dict):
        raise ValueError('--facts: "arms" must map command names to objects, and "all" must be an object')
    unknown = sorted(set(arms) - set(names))
    if unknown:
        raise ValueError(f"--facts names {', '.join(unknown)}, which {'is' if len(unknown) == 1 else 'are'} not in this comparison "
                         f"({', '.join(names)})")
    for where, v in [("all", common)] + [(f"arms.{n}", a) for n, a in arms.items()]:
        if not isinstance(v, dict) or not all(isinstance(v.get(k) or {}, dict) for k in ("requested", "actual")):
            raise ValueError(f'--facts: {where} must be an object whose "requested" and "actual" are objects')


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
        sys.modules[spec.name] = mod  # what dataclasses, pickle and typing look a module's classes up in
        try:
            with contextlib.redirect_stdout(sys.stderr):  # a print at import must not corrupt --json on stdout either
                spec.loader.exec_module(mod)
        except (Exception, SystemExit) as e:  # missing file, syntax error, failed import, sys.exit(): a clean error
            sys.modules.pop(spec.name, None)
            raise ValueError(f"--checks {path}: {type(e).__name__}: {e}")
        if not callable(getattr(mod, "check", None)):
            raise ValueError(f"--checks {path} must define check(report, facts)")
        out.append((path, mod.check))
    return out


def explain(report: dict, facts: dict | None = None, checks: list | None = None) -> dict:
    """The reasons behind a comparison, most important first: problems to fix before quoting a number, then notes."""
    names, baseline, all_runs = report["names"], report["baseline"], report["results"]
    reasons: list[dict] = []
    if facts:
        check_facts(facts, names)
    multi = len(report["blocks"]) > 1
    if facts:  # the facts are about the workload, not one power limit: said once, without a power-limit prefix
        _explain_facts(facts, names, reasons)
    for block in report["blocks"]:
        prefix = f"at {block['limit_pct']}% power: " if multi and block.get("limit_pct") is not None else ""
        runs = [r for r in all_runs if r.get("limit_pct") == block.get("limit_pct")]
        _explain_block(block, runs, names, baseline, prefix, reasons, bool(report.get("imported")))

    from_checks: list[dict] = []
    for path, fn in checks or []:
        try:
            with contextlib.redirect_stdout(sys.stderr):  # a check's prints must not corrupt --json on stdout
                # on copies: a check that sorts or edits what it is given must not change the saved measurements
                got = list(fn(copy.deepcopy(report), copy.deepcopy(facts or {})) or [])
            if not all(isinstance(r, dict) for r in got):
                raise TypeError("check() must return a list of dicts")
            # read here, inside the try: a result whose numbers refer to themselves, or whose title cannot become text,
            # is a broken check too, not a traceback
            got = [{"severity": r.get("severity") if r.get("severity") in ("problem", "note") else "note",
                    "title": str(r.get("title", "check")), "detail": str(r.get("detail", "")),
                    "numbers": _json_safe(r.get("numbers") or {}), "source": path} for r in got]
        except (Exception, SystemExit) as e:  # a broken check (even one calling sys.exit) must not hide the rest
            got = [{"severity": "problem", "title": f"check {path} failed", "detail": f"{type(e).__name__}: {e}",
                    "numbers": {}, "source": path}]
        from_checks += got
    reasons[:0] = from_checks  # your own workload checks come first: a wrong workload makes every other number moot

    if report.get("imported"):
        reasons.append({"severity": "note", "title": "imported runs", "numbers": {},
                        "detail": "these runs come from a CSV rather than a joules compare report (joules compare --csv writes one too), "
                                  "so joules can only check what the CSV says."})
        if any(r.get("position") is None for r in all_runs):
            reasons.append({"severity": "note", "title": "run order unknown", "numbers": {},
                            "detail": "the CSV has no position for some or all runs, so joules cannot check whether the order was fair."})
    if not any(r.get("gpu") for r in all_runs) and not report.get("imported"):
        temps = any(r.get("temp_start_c") is not None for r in all_runs)
        reasons.append({"severity": "note", "title": "power only", "numbers": {},
                        "detail": ("the runs carry no GPU clock or throttling readings (only each run's starting temperature), so "
                                   "throttling could not be checked." if temps else
                                   "this meter reports power only, not the GPU's clocks, temperature or throttling, so heat and "
                                   "throttling could not be checked.")
                                  # --cool-to reads an NVIDIA GPU's temperature: with none read, it would not wait
                                  + (" Resting or cooling between runs (joules compare --rest, --cool-to) keeps heat out of it." if temps else
                                     " Resting between runs (joules compare --rest) keeps heat out of it; --cool-to needs an NVIDIA GPU's "
                                     "temperature, which these runs do not have.")})

    problems = [r for r in reasons if r["severity"] == "problem"]
    reasons = problems + [r for r in reasons if r["severity"] != "problem"]
    summary = "Nothing to compare."
    for block in report["blocks"]:  # the first power limit the baseline was measured at (it can be missing at some)
        lead = f"At {block['limit_pct']}% power, " if multi and block.get("limit_pct") is not None else ""
        summary = _summary({r["name"]: r for r in block["rows"]}, baseline, problems, lead)
        if summary != "Nothing to compare.":
            break
    return {"baseline": baseline, "summary": summary, "reasons": reasons}


def _json_safe(v):
    """A check's numbers as strict JSON: NaN and infinity become text, anything else unknown becomes its str()."""
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v if v is None or isinstance(v, (str, int, float, bool)) else str(v)


def _explain_facts(facts: dict, names: list[str], reasons: list[dict]) -> None:
    """Was the work the work you asked for?"""
    for n in names:
        bad = _fact_mismatches(facts, n)
        if bad:
            hint = ((facts.get("arms") or {}).get(n) or {}).get("hint") or (facts.get("all") or {}).get("hint")
            reasons.append({"severity": "problem", "title": f"{n}: the workload was not what you asked for",
                            "detail": "; ".join(bad) + ". The energy number measures the work that actually ran, so it does not "
                                      "answer the question you asked." + (f" {hint}" if hint else ""),
                            "numbers": {"arm": n, "mismatches": bad}})


GOOD_NEWS = ("Clocks held at", "alike for every command")  # compare notes that report fairness, not a problem


def _explain_block(block: dict, runs: list[dict], names: list[str], baseline: str, prefix: str, reasons: list[dict],
                   imported: bool = False) -> None:
    rows = {r["name"]: r for r in block["rows"]}
    base = rows.get(baseline)

    def add(severity: str, title: str, detail: str, **numbers):
        reasons.append({"severity": severity, "title": prefix + title, "detail": detail, "numbers": numbers})

    # 1. was the work the work you asked for? explain() says that once, for every power limit
    # 2. arms too noisy to quote
    for n, row in rows.items():
        e = row["energy_j"]
        rng = (e["max"] - e["min"]) / e["median"] if e["median"] else 0.0
        if row["runs"] >= 2 and (rng > UNSTABLE_RANGE or e.get("spread_pct", 0) > UNSTABLE_IQR):
            add("problem", f"{n} is not stable yet",
                f"its runs ranged from {_fmt_j(e['min'])} to {_fmt_j(e['max'])} ({rng * 100:.0f}% of its median) over "
                f"{row['runs']} runs. Something changes between runs (warm-up, caching, clocks or the workload itself), so this "
                f"arm's number isn't settled yet; more runs, or a look at what varies, would settle it.", arm=n, min_j=e["min"], max_j=e["max"], runs=row["runs"])

    # 3. fair conditions (checked on the positions the runs really had: joules compare's default order runs three or
    # more commands as A B C, A B C, ...; with two, A B A B interleaves)
    here = [n for n in names if n in rows]  # the commands measured at this power limit (a CSV can leave one out)
    fixed = len(here) >= 3 and _fixed_order(runs, here)
    pred = _constant_predecessor(runs, here) if len(here) >= 3 and not fixed else {}
    if baseline in pred:
        p, k, n_rounds = pred[baseline]
        every = k >= n_rounds
        add("note", f"the baseline {'always' if every else 'nearly always'} ran right after the same command",
            (f"in every round {baseline} followed {p}, so it always started" if every else
             f"in {k} of the {n_rounds} rounds it ran in, {baseline} followed {p} (it ran first in the other"
             f"{'' if n_rounds - k == 1 else 's'}), so it mostly started")
            + " on a GPU just warmed by that work. A fixed rotation is fairer than a fixed order, but this pairing is the "
              "same every time.", predecessor=p, rounds=k, of_rounds=n_rounds)
    if fixed:
        add("problem", "the commands always ran in the same order",
            "every round ran them in one fixed order, so a later command always started right after the same earlier work, on a "
            "GPU that work may have warmed. "
            + ("Varying the order between rounds (reversing it every other round, say) and resting or cooling between runs "
               "would take that out." if imported else
               "Varying the order between rounds (joules compare --order abba reverses it every other round) and resting or "
               "cooling between runs (--rest, --cool-to) would take that out."))
    for note in block.get("notes") or []:
        if "p cannot go below" in note or "there is no p-value" in note or "no p-value can be given" in note:  # said once below
            continue
        if fixed and "alike for every command" in note:  # the fixed order is reported above
            continue
        if any(g in note for g in GOOD_NEWS):
            add("note", "conditions", note)
        elif note.startswith("Clocks were locked at"):  # every command can fall below the lock alike: not a difference
            add("problem", "the GPU did not hold the locked clock", note)
        else:
            add("problem", "the commands did not run under the same conditions", note)

    # 4. per arm against the baseline: is it real, where did it come from, what does idle do to it
    others = sorted((r for n, r in rows.items() if r.get("vs") and r["vs"].get("energy_pct") is not None and base is not None),
                    key=lambda r: r["vs"]["energy_pct"])
    unclear = [r for r in others if r["vs"].get("energy_p") is None or r["vs"]["energy_p"] > SIGNIFICANT_P]
    few = [r for r in unclear if min_p(r["runs"], base["runs"]) > SIGNIFICANT_P]
    if few:
        floor = max(min_p(r["runs"], base["runs"]) for r in few)
        n_runs = min(r["runs"] for r in few + [base])
        who = "any difference" if len(few) == len(others) else ", ".join(r["name"] for r in few) + f" vs {baseline}"
        if n_runs < 2:  # mann_whitney gives no p-value at all with a single run on a side
            add("problem", f"too few runs to call {who} real",
                f"with only 1 run of a command there is no p-value at all, so no gap can be confirmed, however large. "
                f"With 5 runs of each, p can reach {min_p(5, 5):.2g}.", runs=n_runs, p_floor=None)
        else:
            add("problem", f"too few runs to call {who} real",
                f"with as few as {n_runs} runs the smallest p-value possible is {floor:.2g}, so the test can't confirm a gap at this run count, however large. "
                f"With 5 runs of each, p can reach {min_p(5, 5):.2g}.", runs=n_runs, p_floor=floor)
    for r in unclear:
        if r in few:
            continue
        p = r["vs"].get("energy_p")
        add("problem", f"{r['name']} vs {baseline}: the difference could be noise",
            f"energy {_pct(r['vs']['energy_pct'])}, p = {p:.2g}. More runs, or finding what makes the runs vary, would settle it."
            if p is not None else f"energy {_pct(r['vs']['energy_pct'])}; too few runs for a p-value.", arm=r["name"], p=p)

    shown = [r for r in others if abs(r["vs"]["energy_pct"]) >= 5 and _split(r, base)]  # no split (0 J): listed below
    rest_rows = [r for r in others if r not in shown[:MAX_SPLITS]]
    for row in shown[:MAX_SPLITS]:
        s = _split(row, base)
        n = row["name"]
        same = lambda v: abs(v) < 1
        if same(s["power_pct"]):
            how = f"it {'finished sooner' if s['time_pct'] < 0 else 'took longer'} (time {_pct(s['time_pct'])}) at about the same power"
        elif same(s["time_pct"]):
            how = f"it took about the same time and drew {'less' if s['power_pct'] < 0 else 'more'} power (average power {_pct(s['power_pct'])}, median energy over median time)"
        elif s["time_pct"] < 0 and s["power_pct"] <= 0:
            how = f"it finished sooner (time {_pct(s['time_pct'])}) and drew less power (average power {_pct(s['power_pct'])}, median energy over median time)"
        elif s["time_pct"] < 0:
            how = f"it finished sooner (time {_pct(s['time_pct'])}) while drawing more power (average power {_pct(s['power_pct'])}, median energy over median time)"
        elif s["power_pct"] < 0:
            how = f"it drew less power (average power {_pct(s['power_pct'])}, median energy over median time) but took longer (time {_pct(s['time_pct'])})"
        else:
            how = f"it took longer (time {_pct(s['time_pct'])}) and drew more power (average power {_pct(s['power_pct'])}, median energy over median time)"
        share = s["time_share"]
        if not 0 <= share <= 1 and not (same(s["power_pct"]) or same(s["time_pct"])):
            main_txt = " Time and power pulled in opposite directions; the bigger one won."
        elif 0.4 <= share <= 0.6:
            main_txt = " Time and power each explain about half."
        else:
            main_txt = f" About {min(1.0, max(share, 1 - share)) * 100:.0f}% of the change came from {'finishing time' if share > 0.5 else 'power draw'}."
        add("note", f"{n} used {_less_more(s['energy_pct'])} energy than {baseline}", f"{how}.{main_txt}", arm=n, **s)
    if rest_rows:
        rest = ", ".join(f"{r['name']} {_pct(r['vs']['energy_pct'])}" for r in rest_rows)
        add("note", "the other commands", f"against {baseline}: {rest}.")

    idle_gap = [r for r in others if r["vs"].get("above_idle_pct") is not None
                and abs(r["vs"]["above_idle_pct"] - r["vs"]["energy_pct"]) > IDLE_GAP_PTS]
    if idle_gap:
        ex_ = idle_gap[0]["vs"]
        ib = sorted(r["idle_w"] for r in runs if r["name"] == baseline and r.get("idle_w"))
        idle_txt = f" Idle is about {ib[len(ib) // 2]:.0f} W here." if ib else ""
        add("note", "why the total and above-idle numbers differ",
            f"e.g. {idle_gap[0]['name']}: total {_pct(ex_['energy_pct'])}, above idle {_pct(ex_['above_idle_pct'])}. Both are right. "
            + ("Idle power is paid whatever runs, so it shrinks every change in the total. "
               if abs(ex_["energy_pct"]) <= abs(ex_["above_idle_pct"]) and ex_["energy_pct"] * ex_["above_idle_pct"] >= 0 else
               "Idle power is paid for every second a command runs, so a change in time moves the total by idle power times "
               "that change, whatever the work itself drew. ")
            + "Above idle shows what the code itself saves; the total shows what the machine's bill saves." + idle_txt)

    for row in others:
        g, gb = row.get("gpu") or {}, base.get("gpu") or {}
        if (row["vs"]["energy_pct"] < 0 and g.get("mem_util_pct") is not None and gb.get("mem_util_pct") is not None
                and gb["mem_util_pct"] - g["mem_util_pct"] >= 10):  # "where the saving comes from" needs a saving
            add("note", f"{row['name']} kept the GPU's memory less busy",
                f"memory busy {g['mem_util_pct']:.0f}% of the time vs {gb['mem_util_pct']:.0f}% for {baseline}. When the work is limited "
                "by memory traffic, moving less data is where the saving comes from, so the saving would shrink as the share of data "
                "that must move grows.", arm=row["name"])


def _summary(rows: dict, baseline: str, problems: list, lead: str = "") -> str:
    others = [r for n, r in rows.items() if n != baseline and r.get("vs") and r["vs"].get("energy_pct") is not None]
    if not others:
        if baseline in rows and not rows[baseline]["energy_j"]["median"] and len(rows) > 1:
            return f"{lead}{baseline} used no energy (median 0 J), so no change against it can be given as a percentage."
        return "Nothing to compare."
    best = min(others, key=lambda r: r["vs"]["energy_pct"])
    vs = best["vs"]
    s = f"{lead}{best['name']} {_used(vs['energy_pct'], baseline)}"
    if vs.get("above_idle_pct") is not None:
        ai = vs["above_idle_pct"]
        s += f" ({'about the same' if abs(ai) < 0.05 else _less_more(ai)} above idle)"
    s += "."
    if problems:
        s += f" {len(problems)} problem{'s' if len(problems) > 1 else ''} listed below."  # not all need be about this arm
    return s


def _fmt_j(j: float) -> str:
    if 0 < j < 0.001:  # microjoules: "0.00 mJ" would hide a range that is really there
        return f"{j * 1e6:.2f} µJ"
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
    """Least squares y = a + b x. Returns a, b and R² (None with only two x values: the line just joins the mean at
    each, so R² would only measure run-to-run noise, not whether the cost grows in a straight line)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    if len(set(xs)) <= 2:
        return a, b, None
    tol = 1e-12 * max(abs(y) for y in ys)  # the same value at every x: its spread is only rounding in the mean
    return a, b, (1 - ss_res / ss_tot) if ss_tot > n * tol * tol else 1.0


def load_points(files: list[str], x_name: str | None) -> list[dict]:
    """Runs with a workload value x: from VALUE=report.json (a whole compare at one workload) or CSVs with an x column.
    With no x_name, every run gets x = 0 (one workload), and a plain report.json is accepted."""
    runs = []
    seen = [os.path.realpath(_value_path(f) if _valued(f) else f) for f in files]
    if len(set(seen)) < len(seen):  # ./a.csv and a.csv, or 5=r.json and 30=r.json, are the same runs too
        raise ValueError("the same input is given more than once: its runs would count twice")
    valued = [f for f in files if _valued(f)]
    if x_name is None and valued and len(valued) < len(files):  # the plain ones would be scored at a made-up workload 0
        plain = [f for f in files if f not in valued]
        raise ValueError(f"{', '.join(plain)} {'has' if len(plain) == 1 else 'have'} no workload value, unlike {valued[0]}, "
                         "and there is no --x column to read one from")
    for f in files:
        if x_name is None and not _valued(f) and _is_report(f):
            rep = load(f)
            runs += [{**r, "x": 0.0, "_baseline": rep.get("baseline")} for r in _report_runs(f, rep)]
            continue
        if x_name is None and not _valued(f):  # a CSV of runs, read as runs: one file may hold a single command's runs
            runs += [{**r, "x": 0.0} for r in read_csv(f)]
            continue
        if _valued(f):
            v, path = f.split("=", 1)[0], _value_path(f)
            try:
                x = float(v)
            except ValueError:
                raise ValueError(f"{f}: write the workload value before '=', e.g. 25={path}")
            if not math.isfinite(x):
                raise ValueError(f"{f}: the workload value must be a finite number")
            rep = load(path)
            for r in _report_runs(path, rep):
                runs.append({**r, "x": x, "_baseline": rep["baseline"]})
        elif x_name == "limit_pct" and _is_report(f):  # joules compare --limits: each run's limit is its workload
            rep = load(f)
            for r in _report_runs(f, rep):
                x = _limit(r.get("limit_pct", (r.get("fields") or {}).get("limit_pct")))
                if not isinstance(x, float) or not math.isfinite(x):
                    raise ValueError(f"{f}: a run has no power limit to read limit_pct from (got {r.get('limit_pct')!r})")
                runs.append({**r, "x": x, "_baseline": rep["baseline"]})
        elif f.lower().endswith(".csv") and not _is_report(f):
            rs = read_csv(f)  # first: a file that is not UTF-8 or not a readable CSV is named, as in joules explain
            with open(f, newline="", encoding=_csv_encoding(f)) as fh:
                head = next(csv.reader(fh), [])
            if x_name is not None and x_name not in [h.strip() for h in head]:
                raise ValueError(f"{f} has no {x_name} column")
            xs = []
            with open(f, newline="", encoding=_csv_encoding(f)) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    i = reader.line_num  # the file's own line: DictReader skips blank lines
                    row = {k.strip(): (v or "").strip() for k, v in row.items() if k}  # fields past the header come as a list under None
                    try:
                        xs.append(float(row[x_name]) if x_name else 0.0)
                        if not math.isfinite(xs[-1]):
                            raise ValueError
                    except ValueError:
                        raise ValueError(f"{f} line {i}: {x_name} must be a finite number (got {row[x_name]!r})")
            for r, x in zip(rs, xs):
                runs.append({**r, "x": x})
        else:
            raise ValueError(f"{f}: give VALUE=report.json (a joules compare report) or a .csv with a {x_name} column")
    if not runs:
        raise ValueError("no runs to fit")
    lim = lambda r: _limit(r.get("limit_pct", (r.get("fields") or {}).get("limit_pct")))
    lims = {lim(r) for r in runs}
    if len(lims) > 1 and not (x_name == "limit_pct" and all(lim(r) == r["x"] for r in runs)):  # the limit is the workload
        raise ValueError(f"the runs were measured at more than one GPU power limit "
                         f"({', '.join('not set' if v is None else f'{v:g}%' if isinstance(v, float) else str(v) for v in sorted(lims, key=lambda v: (v is not None, v if isinstance(v, float) else float('inf'), str(v))))}); "
                         "only one power limit can be read at a time")
    return runs


def _is_report(path: str) -> bool:
    """A joules compare report: .json, or JSON under another name, .csv included (joules compare --out writes it wherever
    it is told, and joules explain reads it under any name)."""
    return path.lower().endswith(".json") or _looks_like_json(path)


def _valued(f: str) -> bool:
    """VALUE=report: a compare report given with its workload value. A file that is there under the whole name is
    that file, not a value: runs/batch=8/report.json or batch=8.json."""
    return "=" in f and not os.path.isfile(f) and _is_report(_value_path(f))


def _value_path(f: str) -> str:
    """The report in VALUE=report, with a leading ~ expanded: no shell expands the ~ in 25=~/r.json."""
    return os.path.expanduser(f.split("=", 1)[1])


def _report_runs(path: str, rep: dict) -> list[dict]:
    """A report's runs, each with a name in text and its own columns as an object (an edited report: a clean error,
    not a traceback)."""
    for i, r in enumerate(rep["results"], start=1):
        if not isinstance(r.get("name"), str) or not r["name"].strip():
            raise ValueError(f"{path}: run {i} needs a name, as text (got {r.get('name')!r})")
        if not isinstance(r.get("fields") or {}, dict):
            raise ValueError(f"{path}: run {i} has fields that are not an object (got {r['fields']!r})")
    return [_measured(r) for r in rep["results"]]


def _measured(r: dict) -> dict:
    """A report's run without an above-idle energy it does not have (null in the report means not measured)."""
    return {k: v for k, v in r.items() if not (k == "above_idle_j" and v is None)}


def _limit(v):
    """A power limit as one comparable value: 100 from a report, "100" or "100.0" from a CSV are the same limit."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v).strip()


def _same_line_part(u: float, v: float, scale: float) -> bool:
    """Two fitted numbers equal for practical purposes: their difference is negligible against the energies involved
    (an absolute cutoff would call 1e-12 and 4e-13 J per FLOP the same, or 1e5 J per unit fitted in two orders different)."""
    return abs(u - v) <= 1e-9 * scale


def _scale(lines: list[dict], xmax: float) -> float:
    return max([abs(ln["fixed_j"]) + abs(ln["per_unit_j"]) * xmax for ln in lines] + [1e-300])


def _xv(v: float) -> str:
    """A workload value: 3 significant figures, but no exponent for everyday sizes (1,542 tokens, not 1.54e+03)."""
    return f"{v:,.0f}" if 999.5 <= abs(v) < 1e9 else f"{v:.3g}"  # 999.7 rounds to 1,000, not 1e+03


def fit(runs: list[dict], x_name: str, baseline: str | None = None, at: list[float] | None = None) -> dict:
    """Fit energy = fixed + per_unit * x for each command over every run, and find where each one starts to beat
    the baseline. Uses energy above idle as well when every run has it."""
    names = list(dict.fromkeys(r["name"] for r in runs))
    if len(names) < 2:
        raise ValueError("fit needs runs of at least two commands")
    baseline = baseline or next((r["_baseline"] for r in runs if r.get("_baseline") in names), None) or names[-1]
    if baseline not in names:
        raise ValueError(f"--baseline must be one of {', '.join(names)}")
    xs_all = sorted({r["x"] for r in runs})
    if len(xs_all) < 2:
        raise ValueError(f"fit needs runs at two or more values of {x_name} (got only {xs_all[0]:g})")
    bases = ["energy_j"] + (["above_idle_j"] if all("above_idle_j" in r for r in runs) else [])
    if any(not math.isfinite(v) for v in at or []):
        raise ValueError("--at takes finite numbers")
    rng = {n: (min(r["x"] for r in runs if r["name"] == n), max(r["x"] for r in runs if r["name"] == n)) for n in names}
    xmax = max(abs(xs_all[0]), abs(xs_all[-1]))
    out = {"x": x_name, "baseline": baseline, "measured": xs_all, "fits": {}, "crossovers": {}, "at": [], "notes": []}
    for basis in bases:
        lines = {}
        for n in names:
            rs = [r for r in runs if r["name"] == n]
            xs = [r["x"] for r in rs]
            if len(set(xs)) < 2:
                raise ValueError(f"{n} was measured at only one value of {x_name}")
            try:
                a, b, r2 = _ols(xs, [r[basis] for r in rs])
            except OverflowError:
                a = b = r2 = math.inf
            except ZeroDivisionError:  # the spread of x squared is below the smallest float (e.g. 1e-200 and 2e-200)
                raise ValueError(f"the values of {x_name} differ by too little to fit a line through them") from None
            if not all(math.isfinite(v) for v in (a, b, r2 if r2 is not None else 0.0)):
                raise ValueError(f"the energy or {x_name} values are too large to fit (over 1e308)")
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
            scale = _scale([cb, ln], xmax)
            if _same_line_part(db * xmax, 0.0, scale):
                c = {"x": None, "why": "same cost per unit and same fixed cost: the two lines are the same" if _same_line_part(da, 0.0, scale) else
                     f"same cost per unit: {n if da < 0 else baseline} is always cheaper, by {_j(abs(da))}"}
            else:
                xc = da / db
                wins_above = db > 0
                # the range where both were measured: a crossover past either one's last measurement is extrapolated
                lo, hi = max(rng[n][0], rng[baseline][0]), min(rng[n][1], rng[baseline][1])
                c = {"x": xc, "wins": "above" if wins_above else "below", "extra_fixed_j": da, "saved_per_unit_j": db,
                     "inside": lo <= xc <= hi, "range": [lo, hi]}
            out["crossovers"][basis][n] = c
    for x in (at or []):
        per = [sorted({r["x"] for r in runs if r["name"] == n}) for n in names]
        row = {"x": x, "measured": all(x in p for p in per), "inside": all(p[0] <= x <= p[-1] for p in per), "energy_j": {}}
        for n, ln in out["fits"]["energy_j"].items():
            row["energy_j"][n] = ln["fixed_j"] + ln["per_unit_j"] * x
        if not all(math.isfinite(v) for v in row["energy_j"].values()):
            raise ValueError(f"--at {x:g} is too far from the measured values of {x_name} to predict: the energy would be over 1e308")
        out["at"].append(row)
    two = [n for n in names if len({r["x"] for r in runs if r["name"] == n}) == 2]
    if two and len(xs_all) > 2:
        out["notes"].append(f"{', '.join(two)} {'was' if len(two) == 1 else 'were'} measured at only two values of {x_name}, so "
                            "the line goes through them with no check that the cost really grows in a straight line.")
    if len(xs_all) == 2:
        out["notes"].append(f"Only two values of {x_name} were measured, so each line just joins the two averages and "
                            "there is no check that the cost really grows in a straight line. A third value, ideally near the "
                            "break-even, would show whether the line holds.")
    for basis, cs in out["crossovers"].items():
        for n, c in cs.items():
            if c.get("x") is not None and not c["inside"]:
                lo, hi = c["range"]
                where = (f"outside the measured range ({lo:g}–{hi:g})" if lo <= hi else
                         f"where {n} and {baseline} were not both measured (they share no range of {x_name})")
                out["notes"].append(f"The break-even for {n}{' above idle' if basis != 'energy_j' else ''} ({x_name} ≈ {_xv(c['x'])}) "
                                    f"is {where}: it is extrapolated, and a measurement near it would confirm it.")
    for basis, lines in out["fits"].items():
        for n, ln in lines.items():
            off = max((abs(m - ln["fixed_j"] - ln["per_unit_j"] * v) / abs(m) for v, m in ln["medians"].items() if m), default=0.0)
            # run-to-run noise around a flat line is not a bend; a median far off the line is, whatever R² says
            if ln["r2"] is not None and off > 0.05 and (ln["r2"] < 0.9 or off > 0.2):
                out["notes"].append(f"{n}'s energy{' above idle' if basis != 'energy_j' else ''} does not follow a straight line in "
                                    f"{x_name} well (R² = {ln['r2']:.2f}; a measured median is {off * 100:.0f}% off the line), so "
                                    "the break-even is rough.")
    return out


def _j(v: float) -> str:
    return f"{v:,.0f} J" if abs(v) >= 100 else f"{v:.3g} J"


def render_fit(f: dict) -> str:
    x = f["x"]
    meas = ", ".join(f"{v:g}" for v in f["measured"])
    L = ["", f"Energy model:  energy = fixed + per-unit × {x}      (fitted on every run; measured at {x} = {meas})", ""]
    for basis, lines in f["fits"].items():
        title = "Energy" if basis == "energy_j" else "Energy above idle"
        L.append(f"  {title}")
        w = max(len(n) for n in lines)
        L.append(f"    {'':<{w}}   {'fixed':>10}   {'per 1 ' + x:>16}   fit")
        for n, ln in lines.items():
            fit_txt = "2 values: line not checked" if ln["r2"] is None else f"R² {ln['r2']:.3f}"
            L.append(f"    {n:<{w}}   {_j(ln['fixed_j']):>10}   {_j(ln['per_unit_j']):>16}   {fit_txt}")
        L.append("")
    b = f["baseline"]
    for basis, cs in f["crossovers"].items():
        lab = "" if basis == "energy_j" else " (above idle)"
        for n, c in cs.items():
            if c.get("x") is None:
                L.append(f"Break-even{lab}: none between {n} and {b}. {c['why'][0].upper()}{c['why'][1:]}.")
                continue
            lo, hi = c.get("range") or (f["measured"][0], f["measured"][-1])
            xc = _xv(c["x"])
            if not c["inside"] and lo > hi:
                L.append(f"Break-even{lab}: {n} and {b} were not measured over a common range of {x}; "
                         f"the lines cross at {x} ≈ {xc} (extrapolated)")
            elif not c["inside"]:
                always = (c["wins"] == "above") == (c["x"] < lo)
                L.append(f"Break-even{lab}: none in the measured range ({lo:g}–{hi:g}): {n} is "
                         f"{'cheaper' if always else 'dearer'} than {b} all through it; the lines cross at {x} ≈ {xc}")
            else:
                L.append(f"Break-even{lab}: {n} beats {b} {c['wins']} {x} ≈ {xc}")
            da, db = c["extra_fixed_j"], c["saved_per_unit_j"]
            pn, pb = f["fits"][basis][n]["per_unit_j"], f["fits"][basis][b]["per_unit_j"]
            if pn >= 0 and pb >= 0:
                slope = f"rises {_j(abs(db))} {'less' if db > 0 else 'more'} per 1 {x}"
            elif pn <= 0 and pb <= 0:  # both fall as the workload grows: "rises 3 J less" would describe a line that rises
                slope = f"falls {_j(abs(db))} {'more' if db > 0 else 'less'} per 1 {x}"
            else:
                slope = f"{'falls' if pn < 0 else 'rises'} {_j(abs(pn))} per 1 {x} where {b}'s {'falls' if pb < 0 else 'rises'} {_j(abs(pb))}"
            if _same_line_part(da, 0.0, _scale([f["fits"][basis][n], f["fits"][basis][b]], max(abs(f["measured"][0]), abs(f["measured"][-1])))):
                start = "starts at the same fixed cost and"
            else:
                start = f"starts {_j(abs(da))} {'higher' if da > 0 else 'lower'} {'but' if (da > 0) == (db > 0) else 'and'}"
            L.append(f"  why: {n}'s fitted line {start} {slope}, so the two lines cross at {xc}.")
        L.append("")
    if f["at"]:
        L.append(f"Predicted at {x} values you asked for (--at):")
        for row in f["at"]:
            e = row["energy_j"]
            base = e[b]
            tag = "measured" if row["measured"] else "inside the measured range" if row["inside"] else "outside the measured range: no data there"
            parts = ", ".join(f"{n} {_j(v)}" + (f" ({_pct((v - base) / base * 100)})" if base > 0 else "") for n, v in e.items() if n != b)
            L.append(f"  {x} = {row['x']:g}:  {b} {_j(base)};  {parts}   [{tag}]")
        L.append("")
    for i, n in enumerate(f["notes"], 1):
        L.append(f"! {n}")
    return "\n".join(L).rstrip()


# ---------------------------------------------------------------- cost model export: let a runtime pick by energy

def export_model(f: dict, routers: tuple = (), basis: str = "energy_j") -> dict:
    """A small, stable file a runtime can read to pick the strategy with the least energy at a given workload:
    predicted energy = fixed_j + per_unit_j * x, valid inside measured range (extrapolate with care)."""
    basis = basis if basis in f["fits"] else "energy_j"  # the same basis as the break-even headline and --at
    lines = {n: ln for n, ln in f["fits"][basis].items() if n not in routers}
    return {"schema": "joules-costmodel/1", "x": f["x"], "basis": basis, "measured_range": [f["measured"][0], f["measured"][-1]],
            "strategies": {n: {"fixed_j": ln["fixed_j"], "per_unit_j": ln["per_unit_j"], "r2": ln["r2"], "runs": ln["runs"],
                               **({"measured_range": [min(ln["medians"]), max(ln["medians"])]} if ln.get("medians") else {})}
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
    xmax = max(abs(lo), abs(hi))
    cands = sorted({lo, hi} | {(b["fixed_j"] - a["fixed_j"]) / (a["per_unit_j"] - b["per_unit_j"])
                               for a in lines.values() for b in lines.values()
                               if not _same_line_part(a["per_unit_j"] * xmax, b["per_unit_j"] * xmax, _scale([a, b], xmax))})
    width = hi - lo or 1.0
    eps = width * 1e-9
    cands = [c for c in cands if lo - width <= c <= hi + width]
    best = lambda x: min(lines, key=lambda n: lines[n]["fixed_j"] + lines[n]["per_unit_j"] * x)
    out, prev = [], None
    pts = sorted(set(cands))
    for i in range(len(pts) + 1):
        x = pts[0] - eps if i == 0 else (pts[i - 1] + pts[i]) / 2 if i < len(pts) else pts[-1] + eps
        b = best(x)
        if prev is not None and b != prev:
            at = pts[i - 1]
            # inside only where both strategies were measured (a fit carries each one's measured workloads)
            ins = lo <= at <= hi and all(min(lines[n]["medians"]) <= at <= max(lines[n]["medians"])
                                         for n in (prev, b) if lines[n].get("medians"))
            out.append({"at": at, "from": prev, "to": b, "inside": ins})
        prev = b
    return out


# ---------------------------------------------------------------- router report card

def router_card(runs: list[dict], router: str, against: list[str] | None = None, mix: dict | None = None,
                basis: str = "energy_j", valued: bool = True) -> dict:
    """How close a runtime's own strategy picker (the router) comes to the best fixed strategy at each workload,
    and over a mix of workloads. Regret = router energy over the best fixed strategy's, at the same workload.
    valued False: the runs carry no workload value (x made up as 0), so the notes name none."""
    names = list(dict.fromkeys(r["name"] for r in runs))
    if router not in names:
        raise ValueError(f"--router must be one of {', '.join(names)}")
    notes = []

    def contracts(n: str) -> set:  # every contract the arm's runs name (a blank field names none)
        return {c for r in runs if r["name"] == n for c in [str((r.get("fields") or {}).get("contract") or "").strip()] if c}

    def said(n: str) -> str:
        cs = sorted(contracts(n))
        return " / ".join(cs) if cs else "no contract given"

    for n in [router, *(n for n in (against if against is not None else names) if n in names)]:  # the arms being judged
        if len(contracts(n)) > 1:
            raise ValueError(f"{n}'s runs name more than one contract ({said(n)}): one arm should produce one kind of output")
    if against is not None:
        against = list(dict.fromkeys(n for n in against if n))
        if not against:
            raise ValueError("--against names no strategy")
        known = [n for n in [router, *against] if n in names and contracts(n)]
        if len({said(n) for n in known}) > 1:
            by_contract = {}
            for n in [router, *against]:
                if n in names:
                    by_contract.setdefault(said(n), []).append(n)
            groups = "; ".join(f"{c}: {', '.join(ns)}" for c, ns in by_contract.items())
            notes.append(f"the arms name different contracts ({groups}); they are compared here because --against names "
                         "them, which treats them as drop-in replacements for each other")
    else:
        against = [n for n in names if n != router]
        arms = [router, *against]
        if any(contracts(n) for n in arms) and len({said(n) for n in arms}) > 1:
            groups = ", ".join(f"{n} ({said(n)})" for n in against)
            raise ValueError(f"the arms do not all share one contract ({router}: {said(router)}; {groups}). "
                             "Pass --against with the strategies that are drop-in replacements for each other")
        notes.append(f"compared with every other arm ({', '.join(against)}); --against narrows this when some are not "
                     "interchangeable fixed strategies (another router, a different output contract)")
    if router in against:
        raise ValueError(f"--against includes the router {router} itself")
    missing = [n for n in against if n not in names]
    if missing:
        raise ValueError(f"--against names {', '.join(missing)}, not in the runs ({', '.join(names)})")
    if mix is not None:
        if any(not math.isfinite(v) or v < 0 for v in mix.values()):
            raise ValueError("--mix weights must be finite and not negative")
        if not any(v > 0 for v in mix.values()):
            raise ValueError("--mix needs at least one weight above 0")
    used = [r for r in runs if r["name"] == router or r["name"] in against]
    if basis == "above_idle_j" and not all("above_idle_j" in r for r in used):
        basis = "energy_j"
        notes.append("--above-idle asked for, but not every run of these arms has energy above idle: judged on total energy")

    def med(vals):
        v = sorted(vals)
        return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2

    xs = sorted({r.get("x", 0.0) for r in runs if r["name"] == router})
    where = (lambda x: f"at {_xs(x)} ") if valued else (lambda x: "")
    rows = []
    for x in xs:
        at = lambda n: [r[basis] for r in runs if r["name"] == n and r.get("x", 0.0) == x]
        rv = at(router)
        fixed = {n: med(at(n)) for n in against if at(n)}
        if not fixed:
            notes.append(f"{where(x)}the router was measured but no fixed strategy was: left out of the regret and the totals")
            continue
        absent = [n for n in against if n not in fixed]
        if absent:
            notes.append(f"{where(x)}{', '.join(absent)} {'was' if len(absent) == 1 else 'were'} not measured: "
                         "the best fixed strategy there is the best of the rest")
        best = min(fixed, key=fixed.get)
        if fixed[best] <= 0:
            raise ValueError(f"{where(x)}{best} used no energy ({'above idle' if basis != 'energy_j' else 'in total'}): regret can't be a percentage")
        r_med = med(rv)
        rng = (max(rv) - min(rv)) / r_med if r_med else 0.0
        rows.append({"x": x, "router_j": r_med, "best": best, "best_j": fixed[best], "regret_pct": (r_med / fixed[best] - 1) * 100,
                     "fixed": fixed, "router_range": rng, "runs": len(rv)})
        if rng > UNSTABLE_RANGE:
            notes.append(f"{where(x)}the router's runs ranged {rng * 100:.0f}% of their median ({_fmt_j(min(rv))} to {_fmt_j(max(rv))}): "
                         "its choice may differ from run to run, so its median isn't settled yet.")
    if not rows:
        raise ValueError("no workload where both the router and a fixed strategy were measured")
    if not all(math.isfinite(v) for row in rows for v in (row["router_j"], *row["fixed"].values())):
        raise ValueError("the energy values are too large to add up (over 1e308)")
    w = mix or {row["x"]: 1.0 for row in rows}
    known = [row for row in rows if w.get(row["x"], 0) > 0]
    if not known:
        raise ValueError("--mix gives weight to no workload where the router and a fixed strategy were both measured")
    unmatched = [k for k, v in w.items() if v > 0 and k not in {row["x"] for row in rows}]
    if unmatched:
        notes.append(f"--mix weights for {', '.join(_xs(k) for k in unmatched)} are left out: no scored measurement there")
    top = max(w[row["x"]] for row in known)  # weights relative to the largest: tiny ones must not round the totals to 0
    tw = sum(w[row["x"]] / top for row in known)
    total = lambda key: sum(w[row["x"]] / top * key(row) for row in known) / tw
    perfect = total(lambda r: r["best_j"])
    rtot = total(lambda r: r["router_j"])
    always = {n: total(lambda r, n=n: r["fixed"][n]) for n in against if all(n in r["fixed"] for r in known)}
    if not all(math.isfinite(v) for v in (perfect, rtot, *always.values())):
        raise ValueError("the energy values or --mix weights are too large to add up (over 1e308)")
    dropped = [n for n in against if n not in always]
    if dropped:
        notes.append(f"no 'always' total for {', '.join(dropped)}: not measured at every workload in the mix")
    return {"router": router, "against": against, "basis": basis, "rows": rows,
            "mix": {_xs(row["x"]): w[row["x"]] for row in known},
            "router_j": rtot, "perfect_j": perfect, "regret_pct": (rtot / perfect - 1) * 100,
            "always": {n: {"energy_j": v, "router_vs_pct": (rtot / v - 1) * 100} for n, v in always.items()}, "notes": notes}


def _xs(x: float) -> str:
    return f"{x:.15g}"


def render_router(c: dict, x_name: str | None = "x") -> str:
    """x_name None: the runs carry no workload value (one workload, x made up as 0), so none is printed."""
    one = x_name is None and len(c["rows"]) == 1
    L = ["", f"Router report card: {c['router']}   (against {', '.join(c['against'])}; {'above idle' if c['basis'] != 'energy_j' else 'total energy'})", ""]
    col = "" if one else f"{x_name or 'workload':>10}   "
    L.append(f"  {col}{'router':>10}   {'best fixed strategy':<28}   regret")
    for r in c["rows"]:
        col = "" if one else f"{_xs(r['x']):>10}   "
        L.append(f"  {col}{_fmt_j(r['router_j']):>10}   {(r['best'] + ' ' + _fmt_j(r['best_j'])):<28}   {r['regret_pct']:+.1f}%")
    L.append("")
    reg = c["regret_pct"]
    verdict = ("matches the best choice" if abs(reg) < 0.5 else
               f"{-reg:.1f}% below the best fixed choice: it beats every fixed strategy measured" if reg < 0 else
               f"within {reg:.1f}% of perfect" if reg < 5 else
               f"{reg:.0f}% above perfect: it picks the wrong strategy or its picking costs energy")
    mix = ", ".join(f"{k}: {v:g}" for k, v in c["mix"].items())
    where = "Over the runs" if one else f"Over the mix ({x_name or 'workload'} {mix})"
    L.append(f"{where}: router {_fmt_j(c['router_j'])} vs perfect choice {_fmt_j(c['perfect_j'])} "
             f"→ {reg:+.1f}% ({verdict}).")
    pct = lambda v: f"{abs(v):.1f}% {'less' if v < 0 else 'more'}" if abs(v) < 10 else _less_more(v)  # as precise as the regret
    for n, a in c["always"].items():
        L.append(f"  vs always {n}: {_fmt_j(a['energy_j'])} → the router uses "
                 + (f"{pct(a['router_vs_pct'])} energy" if abs(a['router_vs_pct']) >= 0.5 else "about the same energy"))
    for n in c["notes"]:
        L.append(f"! {n}")
    return "\n".join(L).rstrip()
