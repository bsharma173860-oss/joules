"""joules: like `time`, but for energy.

  joules run -- python train.py            measure any command (GPU, CPU, DRAM)
  joules -- python train.py                same thing, shorter
  joules bench ollama:llama3.1:8b          energy per token of a local model
  joules bench ollama:llama3.1:8b --share  ...and add it to the public leaderboard
  joules bench gpu --share                 the same one-minute test on any GPU, for the GPU leaderboard
  joules run --track myproject -- python train.py   ...and chart it on your Tracking page
  joules ci --budget 10 -- pytest -q       energy check for a pull request (vs the base branch)
  joules compare "./bench --new" "./bench --old"   the same work two ways: interleaved runs, medians, p-values
  joules proxy --upstream http://localhost:11434   energy receipt on every AI response
  joules agent --track cluster             energy per job on a GPU server or cluster node (Slurm, Kubernetes)
  joules connect                           put this machine on Live, Tracking and the GPU leaderboard, under one name
  joules devices                           what can be measured on this machine
  joules doctor                            check this machine and print a report you can share
  joules calibrate --wall-wh 12.5          compare the meters with a wall-plug meter over the same time
  joules tune ollama:llama3.1:8b           the GPU power limit that gives the cheapest token (sudo)
  joules login lgk_…                       save your API key (for --track and --share)
"""
from __future__ import annotations

import argparse
import codecs
import contextlib
import datetime as dt
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import threading
import time

from . import __version__
from .bench import DEFAULT_PROMPT, ensure_model, generate, model_params, parse_target
from .meters import Discovery, discover
from .model import (GPUS, GRIDS, calculator_link, calibrate, estimate_accelerator_j, gpu_from_name, gpu_verdict, landauer_floor_j,
                    power_sweep_pick, training_flops)
from .sampler import Sampler, idle_baseline
from . import share as sharing
from . import telemetry

# ---------------------------------------------------------------- formatting
_COLOR = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
GRID_HELP = ("grid for carbon intensity (default us): a country (us, eu, de, uk, in...), a Canadian province "
             "(bc, qc, on, ab...) or a cloud region's location (us-east4, europe-west3, asia-southeast1...)")

def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s
dim, bold, warn, good, accent = (lambda s: _c("2", s)), (lambda s: _c("1", s)), (lambda s: _c("33", s)), (lambda s: _c("32", s)), (lambda s: _c("36", s))


def fmt_j(j: float) -> str:
    for unit, k in (("GJ", 1e9), ("MJ", 1e6), ("kJ", 1e3)):
        if abs(j) >= k:
            return f"{j / k:.3g} {unit}"
    return f"{j:.3g} J"


def fmt_wh(j: float) -> str:
    wh = j / 3600
    for unit, k in (("GWh", 1e9), ("MWh", 1e6), ("kWh", 1e3), ("Wh", 1), ("mWh", 1e-3)):
        if abs(wh) >= k:
            return f"{wh / k:.3g} {unit}"
    return f"{wh * 1e6:.3g} µWh" if wh else "0 Wh"


def fmt_w(w: float) -> str:
    """'0.29', '1.7', '590': small machines draw well under a watt, so whole watts would read 0."""
    return f"{w:.2g}" if abs(w) < 10 else f"{w:.0f}"


def fmt_g(g: float) -> str:
    return "<0.001" if 0 < g < 0.001 else f"{g:.3g}"


def short_cmd(cmd: list[str], n: int = 60) -> str:
    s = " ".join(" ".join(cmd).split())  # inline scripts span lines; keep the header on one
    return s if len(s) <= n else s[:n - 1] + "…"


def fmt_s(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.2f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.2f} s"


def sci(x: float) -> str:
    return f"{x:.2e}".replace("e+0", "e").replace("e+", "e")


NO_KEY = ("joules: no API key yet. Create one at {site}/#/settings/keys and run the `joules login lgk_…` line it shows "
          "(or set $LANDAUER_API_KEY, or pass --key).")


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _auto_track(a) -> None:
    """After joules login, every run and bench on this computer goes to the Tracking page (project "default"),
    unless --no-track or JOULES_NO_TRACK. A key given by --key or $LANDAUER_API_KEY (scripts, CI) never does this:
    there, tracking stays an explicit --track."""
    if a.track or getattr(a, "no_track", False) or os.environ.get("JOULES_NO_TRACK") or os.environ.get("CI"):
        return
    if a.key or sharing.env_key() or not sharing.saved_key():
        return
    a.track, a.auto_track = "default", True


def _website_hint(a, cmd: list[str] | None = None) -> None:
    """Results stay on this machine unless asked; say once, in a terminal, how to see them on the website."""
    if a.track or getattr(a, "share", False) or getattr(a, "no_track", False) or not _interactive():
        return
    site = sharing.api_url(a.api_url)
    if not sharing.api_key(a.key):  # the usual case: this computer is not signed in yet
        print(dim(f"joules: this result stays on this computer. To see every run on your Tracking page at {site}, "
                  f"sign this computer in once:\n  joules login        (paste a key from Settings → API keys)"), file=sys.stderr)
        return
    if cmd is not None:
        text = (subprocess.list2cmdline if os.name == "nt" else shlex.join)(cmd)
        shown = text if len(text) <= 60 else "<your command>"
        print(dim(f"joules: this result stays on this computer. To keep it on your Tracking page at {site}, add --track:\n"
                  f"  joules --track my-project -- {shown}"), file=sys.stderr)
    else:
        print(dim(f"joules: this result stays on this computer. Add --share to put it on the public leaderboard "
                  f"({site}/#/board), or --track my-project to keep it on your Tracking page."), file=sys.stderr)


def ensure_key(a) -> str | None:
    """The API key for --track / --share: --key, $LANDAUER_API_KEY or the saved one. In a terminal with none yet,
    asks for it once (checked, then saved like `joules login`), so a first run does not end in a missing-key error."""
    key = sharing.api_key(getattr(a, "key", None))
    if key or not _interactive() or os.environ.get("JOULES_NO_PROMPT"):
        return key
    import getpass
    site = sharing.api_url(getattr(a, "api_url", None))
    print(f"joules: --track and --share need your free API key. Create one at {site}/#/settings/keys and paste it here.", file=sys.stderr)
    for _ in range(3):
        try:
            key = getpass.getpass("API key (lgk_…, or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None
        if not key:
            return None
        if not sharing.KEY_RE.match(key):
            print(warn("  that is not an API key: keys look like lgk_ followed by 40 letters and digits."), file=sys.stderr)
            continue
        try:
            sharing.check_key(key, getattr(a, "api_url", None))
        except RuntimeError as e:
            print(warn(f"  the key was not accepted: {e}"), file=sys.stderr)
            continue
        sharing.save_key(key)
        print(good("joules: signed in; the key is saved for next time."), file=sys.stderr)
        return key
    return None


def _not_found(cmd: list[str], windows: bool = os.name == "nt") -> int:
    if windows:
        hint = " On Windows Python is `python` or `py`." if cmd[0] in ("python3", "pip3") else ""
    else:
        hint = " On macOS and most Linux systems Python is `python3`." if cmd[0] in ("python", "pip") else ""
    print(warn(f"joules: command not found: {cmd[0]}.{hint}"), file=sys.stderr)
    return 127


def _start(cmd: list[str]):
    try:
        return subprocess.Popen(cmd)
    except FileNotFoundError:
        return None
    except PermissionError:
        print(warn(f"joules: {cmd[0]} is not executable."), file=sys.stderr)
        return False


# Plain stand-ins for the symbols joules prints, for a console or a pipe that cannot show them: never "?" boxes.
PLAIN = {"·": "-", "─": "-", "→": "->", "…": "...", "µ": "u", "×": "x", "₂": "2", "✓": "OK", "✗": "X", "⚡": "",
         "≈": "~", "−": "-", "–": "-", "—": "-", "²": "2", "°": " deg", "€": "EUR", "£": "GBP", "₹": "Rs", "≥": ">=", "≤": "<="}


def _plain(err):
    if isinstance(err, UnicodeEncodeError):
        return "".join(PLAIN.get(ch, "?") for ch in err.object[err.start:err.end]), err.end
    raise err


codecs.register_error("joules", _plain)


def _plain_streams() -> None:
    """A Windows pipe is read back with the console's code page, which is not the one Python writes in, so on
    Windows anything piped (| Out-String, > file) is plain ASCII, unless PYTHONIOENCODING says otherwise (UTF-8
    mode does not help: the reader still decodes with the console's code page). Everywhere else a symbol the
    terminal's encoding lacks gets its plain stand-in instead of crashing."""
    forced = os.environ.get("PYTHONIOENCODING")
    for stream in (sys.stdout, sys.stderr):
        try:
            if os.name == "nt" and not forced and not stream.isatty():
                stream.reconfigure(encoding="ascii", errors="joules")
            else:
                stream.reconfigure(errors="joules")
        except (AttributeError, ValueError, OSError):
            pass


# ---------------------------------------------------------------- shared options
def common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("measurement")
    g.add_argument("--interval", type=float, default=0.1, help="sampling interval in seconds (default 0.1)")
    g.add_argument("--gpus", help="only these GPU indices, e.g. 0,1 (default: CUDA_VISIBLE_DEVICES or all)")
    g.add_argument("--no-cpu", action="store_true", help="do not measure CPU and DRAM")
    g.add_argument("--subtract-idle", type=float, default=0, metavar="SECONDS",
                   help="first measure the idle machine for this long and report energy above idle too")
    c = p.add_argument_group("cost and carbon")
    c.add_argument("--price", type=float, default=0.10, help="electricity price per kWh (default 0.10)")
    c.add_argument("--grid", choices=sorted(GRIDS), default="us", metavar="NAME", help=GRID_HELP)
    c.add_argument("--ci", type=float, help="carbon intensity in g CO2/kWh (overrides --grid)")
    c.add_argument("--pue", type=float, default=1.0, help="facility overhead to apply (1.0 = device energy only)")
    o = p.add_argument_group("output")
    o.add_argument("--json", action="store_true", help="print the result as JSON")
    o.add_argument("--out", metavar="FILE", help="also write the JSON result to FILE")


def _gpu_list(s: str | None):
    return [int(x) for x in s.split(",")] if s else None


def _setup(a) -> tuple[Discovery, dict]:
    d = discover(_gpu_list(a.gpus), cpu=not a.no_cpu, estimate=getattr(a, "estimate", False))
    idle = {}
    if d.meters and a.subtract_idle > 0:
        print(dim(f"joules: measuring idle power for {a.subtract_idle:g} s…"), file=sys.stderr)
        idle = idle_baseline(d.meters, a.subtract_idle, a.interval)
    return d, idle


def _devices(d: Discovery, idle: dict, duration: float) -> list[dict]:
    out = []
    for m in d.meters:
        row = {"name": m.name, "kind": m.kind, "method": m.method, "energy_j": m.energy_j, "avg_w": m.avg_w, "peak_w": m.peak_w}
        if m.name in idle:
            row["idle_w"] = idle[m.name]
            row["above_idle_j"] = max(0.0, m.energy_j - idle[m.name] * duration)
        h = m.health() if hasattr(m, "health") else None
        if h:
            row["gpu"] = h
            v = gpu_verdict(h)
            if v:
                row["verdict"] = v
        out.append(row)
    return out


def health_line(h: dict, v: dict | None) -> str:
    """One line under a GPU: clocks, busy and memory percentages, temperature, power limit, and the verdict."""
    parts = []
    if "sm_clock_mhz" in h:
        parts.append(f"{h['sm_clock_mhz']:,.0f} MHz" + (f" ({h['clock_pct']:.0f}% of max)" if "clock_pct" in h else ""))
    if "gpu_util_pct" in h:
        parts.append(f"{h['gpu_util_pct']:.0f}% busy · memory {h['mem_util_pct']:.0f}%")
    if "temp_c_max" in h:
        parts.append(f"{h['temp_c_max']:.0f} °C max")
    if "power_limit_w" in h:
        parts.append(f"limit {h['power_limit_w']:.0f} W")
    if v and v["summary"]:
        parts.append(v["summary"])
    return " · ".join(parts)


def _totals(a, devices: list[dict], ops: float | None) -> dict:
    total = sum(x["energy_j"] for x in devices)
    ci = a.ci if a.ci is not None else GRIDS[a.grid]
    fac = total * a.pue
    kwh = fac / 3.6e6
    t = {"devices_j": total, "pue": a.pue, "facility_j": fac, "kwh": kwh,
         "cost": {"price_per_kwh": a.price, "amount": kwh * a.price},
         "co2": {"g_per_kwh": ci, "grams": kwh * ci}}
    if any("above_idle_j" in x for x in devices):
        t["above_idle_j"] = sum(x.get("above_idle_j", x["energy_j"]) for x in devices)
    if ops:
        floor = landauer_floor_j(getattr(a, "temp", 300.0))
        t["landauer"] = {"operations": ops, "joules_per_operation": fac / ops, "floor_j": floor, "gap": fac / ops / floor}
    return t


def _host() -> dict:
    return {"os": platform.system(), "machine": platform.machine(), "cpu": sharing.cpu_name(), "python": platform.python_version()}


def _emit(a, report: dict, text: str) -> None:
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(text, file=sys.stderr)


def _device_lines(devices: list[dict]) -> list[str]:
    w = max([len(x["name"]) for x in devices] + [10])
    lines = []
    for x in devices:
        extra = f"  {dim('above idle ' + fmt_j(x['above_idle_j']))}" if "above_idle_j" in x else ""
        meth = warn(x["method"]) if x["method"] == "SIMULATED" else dim(x["method"])
        lines.append(f"  {x['name']:<{w}}  {fmt_j(x['energy_j']):>9}  {fmt_w(x['avg_w']):>7} W avg  {meth}{extra}")
        if x.get("gpu"):
            lines.append(dim("    " + health_line(x["gpu"], x.get("verdict"))))
    if any(x["method"] == "powermetrics sampling" for x in devices):  # say what a Mac's figure covers
        lines.append(dim("  the chip only: the screen, SSD and charger are not included"))
    return lines


def _money(x: float) -> str:
    if x == 0:
        return "0"
    if x < 0.0001:
        return "<0.0001"
    return f"{x:.2g}" if x < 0.01 else f"{x:.3g}" if x < 1 else f"{x:,.2f}"


# ---------------------------------------------------------------- joules run
def cmd_run(a) -> int:
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        print("joules run: give a command, e.g. joules run -- python3 train.py", file=sys.stderr)
        return 2
    a.estimate = True  # with no energy counters, estimate from CPU time (and say so) rather than measure nothing
    _auto_track(a)
    if a.track:
        ensure_key(a)  # ask now, not after a long run
    d, idle = _setup(a)
    if not d.meters:
        for n in d.notes:
            print(warn("joules: " + n), file=sys.stderr)
        print(warn("joules: running the command without measuring."), file=sys.stderr)
        proc = _start(cmd)
        if not proc:
            return 126 if proc is False else _not_found(cmd)
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                continue
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        with Sampler(d.meters, a.interval) as s:
            proc = _start(cmd)
            if not proc:
                return 126 if proc is False else _not_found(cmd)
            for m in d.meters:
                if hasattr(m, "attach"):
                    m.attach(proc)
            while True:
                try:
                    code = proc.wait()
                    break
                except KeyboardInterrupt:
                    continue  # Ctrl-C reaches the child too; wait for it to finish, then report
    finally:
        d.close()
    devices = _devices(d, idle, s.duration_s)
    ops = a.flops or (training_flops(a.params, a.tokens) if a.params and a.tokens else None)
    report = {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "run", "command": cmd, "exit_code": code,
              "started_at": started, "duration_s": s.duration_s, "simulated": d.simulated, "host": _host(),
              "devices": devices, "totals": _totals(a, devices, ops), "notes": d.notes}

    gpu_rows = [m for m in d.meters if m.kind == "gpu"]
    kinds = {a.gpu} if a.gpu else {gpu_from_name(m.name) for m in gpu_rows}
    gid = next(iter(kinds)) if len(kinds) == 1 else None
    cal = None
    if ops and gpu_rows:
        if len(kinds) > 1:
            report["notes"].append("Different GPU models in one run: skipping estimate and calibration. Use --gpus to pick one model.")
        elif gid not in GPUS:
            report["notes"].append(f"Unknown GPU model ({gpu_rows[0].name}): pass --gpu to compare with the estimate.")
        else:
            cal = calibrate(ops, gid, len(gpu_rows), s.duration_s, sum(m.energy_j for m in gpu_rows))
            if cal["effective_mfu_pct"] > 100:
                report["notes"].append(
                    f"The workload you described ({ops:.2e} FLOP) is more than {len(gpu_rows)} × {GPUS[gid][0]} can compute in "
                    f"{fmt_s(s.duration_s)} (it would need {cal['effective_mfu_pct']:,.0f}% utilisation). Check --params/--tokens/--flops; "
                    "the Landauer gap above uses them too.")
                cal = None
    if cal:
        gpu_j = sum(m.energy_j for m in gpu_rows)
        est = estimate_accelerator_j(ops, gid, 40, 80)
        report["estimate"] = {"gpu": gid, "assumed_mfu_pct": 40, "assumed_draw_pct": 80, "accelerator_j": est,
                              "measured_accelerator_j": gpu_j, "difference_pct": (gpu_j - est) / est * 100}
        report["calibration"] = cal
        name, tdp, peak = GPUS[gid]
        state = {"mode": "training", "gpu": gid, "tdp": tdp, "peak": peak, "count": len(gpu_rows),
                 "mfu": round(cal["effective_mfu_pct"], 1), "draw": round(min(100, cal["avg_draw_pct"]), 1),
                 "price": a.price, "pue": max(1.0, a.pue), "ci": report["totals"]["co2"]["g_per_kwh"], "grid": a.grid if a.ci is None else "custom"}
        if a.params and a.tokens:
            state.update(params=a.params, tokens=a.tokens)
        report["calculator_link"] = calculator_link(state)
    _emit(a, report, render_run(report))
    if a.track:
        _track(a, report)  # a tracking problem is reported but never changes the command's exit code
    _website_hint(a, cmd)
    return code


def render_run(r: dict) -> str:
    t = r["totals"]
    head = f"{bold('joules')} · {short_cmd(r['command'])} · {fmt_s(r['duration_s'])} · exit {r['exit_code']}"
    lines = [head, *_device_lines(r["devices"]), dim("  " + "─" * 58)]
    lines.append(f"  {bold('total')}  {bold(fmt_j(t['devices_j']))} = {fmt_wh(t['devices_j'])}"
                 + (f"   {dim('above idle')} {fmt_j(t['above_idle_j'])}" if "above_idle_j" in t else ""))
    if t["pue"] != 1.0:
        lines.append(f"  facility (PUE {t['pue']:g})  {fmt_j(t['facility_j'])} = {fmt_wh(t['facility_j'])}")
    lines.append(f"  cost {_money(t['cost']['amount'])} at {t['cost']['price_per_kwh']:g}/kWh · CO₂ {fmt_g(t['co2']['grams'])} g at {t['co2']['g_per_kwh']:g} g/kWh")
    if "landauer" in t:
        L = t["landauer"]
        detail = "%s J per operation vs %s J floor" % (sci(L["joules_per_operation"]), sci(L["floor_j"]))
        lines.append(f"  landauer gap {accent(sci(L['gap']) + '×')}  {dim(detail)}")
    if "estimate" in r:
        e, c = r["estimate"], r["calibration"]
        sign = "+" if e["difference_pct"] >= 0 else ""
        lines.append(f"  estimate vs measured  {fmt_j(e['accelerator_j'])} estimated (40% MFU, 80% draw) → {fmt_j(e['measured_accelerator_j'])} measured ({sign}{e['difference_pct']:.0f}%)")
        mfu_s = "%.1f%% effective utilisation" % c["effective_mfu_pct"]
        draw_s = "%.0f%% of rated power" % c["avg_draw_pct"]
        lines.append(f"  your hardware ran at {good(mfu_s)} and {good(draw_s)}")
        lines.append(f"  open in the calculator with these numbers: {accent(r['calculator_link'])}")
    for n in r["notes"]:
        lines.append(warn("  ! " + n))
    return "\n".join(lines)


# ---------------------------------------------------------------- joules bench
def cmd_bench(a) -> int:
    if a.target == "gpu":
        return cmd_bench_gpu(a)
    try:
        target = parse_target(a.target, a.url)
    except ValueError as e:
        print(f"joules bench: {e}", file=sys.stderr)
        return 2
    _auto_track(a)
    if a.share or a.track:
        ensure_key(a)
    if a.share and a.subtract_idle <= 0:
        a.subtract_idle = 5  # the board compares energy above idle, measured the same way on every machine
    try:
        ensure_model(target, lambda m: print(dim("joules: " + m), file=sys.stderr))
        if not a.params:  # Ollama knows the model's size: the Landauer gap needs no --params
            a.params = model_params(target)
            if a.params:
                print(dim(f"joules: {target.model} has {a.params:.3g}B parameters (from Ollama); "
                          "the Landauer gap counts 2 operations per parameter per token"), file=sys.stderr)
        for _ in range(a.warmup):
            generate(target, a.prompt, min(a.max_tokens, 32))
    except RuntimeError as e:
        print(f"joules bench: {e}", file=sys.stderr)
        return 1
    d, idle = _setup(a)
    if not d.meters:
        for n in d.notes:
            print(warn("joules: " + n), file=sys.stderr)
        return 1
    out_tok = in_tok = 0
    gen_s = 0.0
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        with Sampler(d.meters, a.interval) as s:
            for _ in range(a.runs):
                g = generate(target, a.prompt, a.max_tokens)
                out_tok += g["output_tokens"]; in_tok += g["prompt_tokens"]; gen_s += g["gen_s"] or 0
    except RuntimeError as e:
        print(f"joules bench: {e}", file=sys.stderr)
        return 1
    finally:
        d.close()
    if out_tok == 0:
        print("joules bench: the server returned no output tokens.", file=sys.stderr)
        return 1
    devices = _devices(d, idle, s.duration_s)
    ops = 2 * a.params * 1e9 * (out_tok + in_tok) if a.params else None
    totals = _totals(a, devices, ops)
    if "above_idle_j" in totals and totals["above_idle_j"] <= 0.02 * totals["devices_j"]:
        # nothing measurable above idle: per-token energy would read 0. Use the total, and say why.
        totals.pop("above_idle_j")
        for x in devices:
            x.pop("above_idle_j", None)
        d.notes.append("The model added no measurable power above idle, so energy per token uses the total energy. "
                       "It may be running on hardware joules cannot see (joules devices lists what it measures).")
    basis = totals.get("above_idle_j", totals["devices_j"]) * a.pue
    j_tok = basis / out_tok
    kwh_m = j_tok * 1e6 / 3.6e6
    b = {"target": f"{target.kind}:{target.model}", "url": target.url, "runs": a.runs, "max_tokens": a.max_tokens,
         "output_tokens": out_tok, "prompt_tokens": in_tok, "tokens_per_s": out_tok / (gen_s or s.duration_s),
         "energy_basis": "above idle" if "above_idle_j" in totals else "total", "joules_per_output_token": j_tok,
         "kwh_per_million_output_tokens": kwh_m, "electricity_cost_per_million_output_tokens": kwh_m * a.price}
    if a.api_price:
        b["api_price_per_million_output_tokens"] = a.api_price
        b["api_vs_electricity_ratio"] = a.api_price / b["electricity_cost_per_million_output_tokens"] if b["electricity_cost_per_million_output_tokens"] else None
    report = {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "bench", "started_at": started,
              "duration_s": s.duration_s, "simulated": d.simulated, "host": _host(), "devices": devices,
              "totals": totals, "bench": b, "notes": d.notes}
    _emit(a, report, render_bench(report))
    code = _share(a, report) if a.share else 0
    if a.track:
        tracked = _track(a, report)
        if not getattr(a, "auto_track", False):  # an automatic upload never fails the benchmark
            code = tracked or code
    _website_hint(a)
    return code


def cmd_bench_gpu(a) -> int:
    from . import gpubench
    try:
        gpus = _gpu_list(a.gpus) or [0]
    except ValueError:
        print("joules bench gpu: --gpus takes one GPU number, e.g. --gpus 0", file=sys.stderr)
        return 2
    if len(gpus) > 1:
        print("joules bench gpu: the test runs on one GPU at a time; pick one with --gpus, e.g. --gpus 0", file=sys.stderr)
        return 2
    if a.track:
        print(dim("joules: the GPU test is not saved to Tracking; --share adds it to the GPU leaderboard."), file=sys.stderr)
    if a.share and not ensure_key(a):  # before a minute of testing, not after
        print(warn(NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
        return 2
    print(dim(f"joules: the GPU test takes about a minute: {gpubench.N}×{gpubench.N} matrix multiplications in fp16, then fp32"), file=sys.stderr)
    try:
        report = gpubench.run(gpus[0], interval=a.interval, log=lambda m: print(dim("joules: " + m), file=sys.stderr))
    except gpubench.NoTorch as e:
        py = gpubench.torch_python()
        if py and getattr(a, "argv", None):
            print(dim(f"joules: PyTorch is not in the Python joules runs in; running the test with {py}, which has it"), file=sys.stderr)
            return gpubench.run_with(py, a.argv)
        print(f"joules bench gpu: {e}", file=sys.stderr)
        return 1
    except gpubench.GpuBenchError as e:
        print(f"joules bench gpu: {e}", file=sys.stderr)
        return 1
    _emit(a, report, render_gpubench(report))
    if a.share:
        return _share_gpu(a, report)
    if _interactive():
        print(dim("joules: to put this GPU on the leaderboard: joules bench gpu --share"), file=sys.stderr)
    return 0


def render_gpubench(r: dict) -> str:
    n = r["n"]
    lines = [f"{bold('joules bench gpu')} · {r['device']} (GPU {r['gpu']}) · {n}×{n} matrix multiplications · {fmt_s(r['duration_s'])}"]
    for p, x in r["results"].items():
        eff = f"{x['gflop_per_j']:.0f} GFLOP/J"
        lines.append(f"  {p}  {x['tflops']:7.1f} TFLOP/s  {fmt_w(x['avg_w']):>5} W  {bold(eff)}")
        extra = []
        if "clock_pct" in x:
            extra.append(f"clock {x['clock_pct']:.0f}% of max")
        if "throttled_pct" in x:
            why = next(iter(x.get("throttle_reasons") or {}), None)
            extra.append(f"throttled {x['throttled_pct']:.0f}% of the time" + (f" ({why})" if why and x["throttled_pct"] >= 1 else ""))
        if "temp_c_start" in x and "temp_c_end" in x:
            extra.append(f"{x['temp_c_start']:.0f} → {x['temp_c_end']:.0f} °C")
        if extra:
            lines.append(dim("        " + " · ".join(extra)))
    meter = {"nvml counter": "the NVIDIA energy counter", "nvml power sampling": "NVIDIA power readings, sampled and added up",
             "amdgpu energy counter": "the AMD energy counter", "amdgpu power sampling": "AMD power readings, sampled and added up",
             "adlx power sampling": "AMD power readings from the Windows driver (ADLX), sampled and added up"}
    about = [f"energy from {meter.get(r['meter'], r['meter'])}"]
    if r.get("power_limit_w"):
        about.append(f"power limit {r['power_limit_w']:.0f} W")
    if r.get("torch"):
        about.append(f"PyTorch {r['torch']}" + (f" ({r['runtime']})" if r.get("runtime") else ""))
    lines.append(dim("  " + " · ".join(about)))
    for x in r["notes"]:
        lines.append(warn("  ! " + x))
    return "\n".join(lines)


def _share_gpu(a, report: dict) -> int:
    key = ensure_key(a)
    if not key:
        print(warn(NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
        return 2
    try:
        m = sharing.gpu_measurement(report)
        sharing.send_gpu(m, key, a.api_url)
    except (ValueError, RuntimeError) as e:
        print(warn(f"joules share: {e}"), file=sys.stderr)
        return 1
    best = max(report["results"].values(), key=lambda x: x["gflop_per_j"])
    print(good(f"joules: shared {m['device']} on the GPU leaderboard ({best['gflop_per_j']:.0f} GFLOP/J)."), file=sys.stderr)
    print(f"  leaderboard: {accent(sharing.gpu_board_link(m['device'], a.api_url))}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- joules share
def _share(a, report: dict) -> int:
    key = ensure_key(a)
    if not key:
        print(warn(NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
        return 2
    try:
        m = sharing.measurement(report)
        sharing.send(m, key, a.api_url)
    except (ValueError, RuntimeError) as e:
        print(warn(f"joules share: {e}"), file=sys.stderr)
        return 1
    print(good(f"joules: shared {m['model']} on {m['device_count']}× {m['device']} at {m['j_per_token']:.3g} J/token."), file=sys.stderr)
    print(f"  leaderboard: {accent(sharing.board_link(m['model'], a.api_url))}", file=sys.stderr)
    return 0


def _track(a, report: dict) -> int:
    key = ensure_key(a)
    if not key:
        print(warn(NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
        return 2
    try:
        r = sharing.track_record(report, a.track, a.label)
        reply = sharing.send_track(r, key, a.api_url) or {}
    except (ValueError, RuntimeError) as e:
        print(warn(f"joules track: {e}"), file=sys.stderr)
        return 1
    where = "your team's project" if reply.get("team") else "project"
    print(good(f"joules: tracked in {where} {r['project']}") + f"  {accent(sharing.track_link(r['project'], a.api_url))}", file=sys.stderr)
    for al in reply.get("alerts") or []:
        print(warn(f"joules: energy alert: this run used {al.get('change_pct', 0):.0f}% more than the recent median "
                   f"for {r['label']} (your alert is set at {al.get('threshold_pct', 0):g}%)."), file=sys.stderr)
    return 0


def cmd_share(a) -> int:
    try:
        with open(a.file, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError) as e:
        print(f"joules share: cannot read {a.file} ({e})", file=sys.stderr)
        return 2
    return _share(a, report)


def sharing_args(p: argparse.ArgumentParser, track: bool = True) -> None:
    g = p.add_argument_group("WattOps account")
    if track:
        g.add_argument("--track", nargs="?", const="default", metavar="PROJECT",
                       help="save the result to your private Tracking page under PROJECT (default: default). After "
                            "joules login this happens on its own for every run on this computer")
        g.add_argument("--no-track", action="store_true", help="keep this result on this computer only")
        g.add_argument("--label", help="name shown on the Tracking page (default: program and script name)")
    g.add_argument("--key", help=f"WattOps API key (default: ${sharing.KEY_ENV}, else the key saved by joules login)")
    g.add_argument("--api-url", help=f"API address (default: ${sharing.URL_ENV} or {sharing.SITE})")


def render_bench(r: dict) -> str:
    b, t = r["bench"], r["totals"]
    lines = [f"{bold('joules bench')} · {b['target']} · {b['runs']} runs × up to {b['max_tokens']} tokens · {fmt_s(r['duration_s'])}",
             *_device_lines(r["devices"]), dim("  " + "─" * 58),
             f"  output tokens  {b['output_tokens']:,}   ({b['prompt_tokens']:,} prompt)   {b['tokens_per_s']:.1f} tok/s",
             f"  energy         {fmt_j(t['devices_j'])}" + (f"  ({fmt_j(t['above_idle_j'])} above idle, used below)" if "above_idle_j" in t else ""),
             f"  per token      {bold('%.3g J' % b['joules_per_output_token'])}   ·   {b['kwh_per_million_output_tokens']:.3g} kWh per 1M tokens",
             f"  electricity    {bold(_money(b['electricity_cost_per_million_output_tokens']))} per 1M output tokens at {t['cost']['price_per_kwh']:g}/kWh"]
    if "api_price_per_million_output_tokens" in b and b.get("api_vs_electricity_ratio"):
        ratio = b["api_vs_electricity_ratio"]
        verdict = f"{ratio:.1f}× cheaper than the API" if ratio >= 1 else f"{1 / ratio:.1f}× more than the API"
        lines.append(f"  vs API         {_money(b['api_price_per_million_output_tokens'])} per 1M → electricity is {good(verdict) if ratio >= 1 else warn(verdict)} {dim('(hardware cost not included)')}")
    if "landauer" in t:
        lines.append(f"  landauer gap   {accent(sci(t['landauer']['gap']) + '×')}")
    for n in r["notes"]:
        lines.append(warn("  ! " + n))
    return "\n".join(lines)


# ---------------------------------------------------------------- joules ci
def cmd_ci(a) -> int:
    from . import ci
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        print("joules ci: give the command to measure, e.g. joules ci --budget 10 -- pytest -q", file=sys.stderr)
        return 2
    d = discover(_gpu_list(a.gpus), cpu=not a.no_cpu, estimate=True)
    for n in d.notes:
        print(warn("joules ci: " + n), file=sys.stderr)
    if not d.meters:
        return 1
    base_ref = a.base or ci.default_base()
    head_dir = os.getcwd()
    try:
        head_name = ci.git("rev-parse", "--short", "HEAD")
    except RuntimeError:
        head_name = "this checkout"
    try:
        with contextlib.ExitStack() as stack:
            base_dir = stack.enter_context(ci.worktree(base_ref)) if base_ref else None
            # a subdirectory of the repository (--workdir) is the same subdirectory in the base checkout
            # real paths on both sides: Windows may give this folder its short name (RUNNER~1) and git the long one
            rel = os.path.relpath(os.path.realpath(head_dir), os.path.realpath(ci.git("rev-parse", "--show-toplevel"))) if base_dir else "."
            base_cwd = os.path.normpath(os.path.join(base_dir, rel)) if base_dir else None
            ci.run_setup(a.setup, head_dir)
            if base_cwd:
                if not os.path.isdir(base_cwd):
                    raise RuntimeError(f"{rel} does not exist on {base_ref}; nothing to compare with")
                ci.run_setup(a.setup, base_cwd)
            head, base = [], []
            for i in range(a.warmup + a.runs):
                for side, cwd, out in (("this change", head_dir, head), ("base", base_cwd, base)):
                    if cwd is None:
                        continue
                    print(dim(f"joules ci: {side}, run {i + 1 - a.warmup if i >= a.warmup else 'warm-up'}…"), file=sys.stderr)
                    r = ci.measure(d, cmd, cwd, a.interval)
                    if r["exit_code"] != 0:
                        raise RuntimeError(f"the command failed on {side} (exit {r['exit_code']}): {short_cmd(cmd)}")
                    if i >= a.warmup:
                        out.append(r)
    except RuntimeError as e:
        print(warn(f"joules ci: {e}"), file=sys.stderr)
        return 1
    finally:
        d.close()
    summary = ci.summarize(head, base or None, a.budget)
    estimated = any(m.kind == "est" for m in d.meters)
    link = sharing.track_link(a.track, a.api_url) if a.track else None
    md = ci.markdown(summary, head_name=head_name, base_name=base_ref and base_ref.replace("origin/", ""),
                     devices=[m.name if m.kind == "est" else f"{m.name} ({m.method})" for m in d.meters], estimated=estimated,
                     command=short_cmd(cmd, 200), track_link=link)
    print(md)
    for path in filter(None, (a.markdown, os.environ.get("GITHUB_STEP_SUMMARY"))):
        with open(path, "a", encoding="utf-8") as f:  # the summary has "⚡"; Windows would default to cp1252
            f.write(md)
    ci.write_outputs(summary)
    if a.comment:
        print(dim("joules ci: " + ci.post_comment(md)), file=sys.stderr)
    if a.track:
        devices = [{"name": m.name, "kind": m.kind, "method": m.method, "energy_j": statistics.median(r["devices"][m.name] for r in head),
                    "avg_w": 0.0, "peak_w": 0.0} for m in d.meters]
        report = {"kind": "run", "command": cmd, "started_at": ci.now_iso(), "duration_s": summary["head_s"],
                  "simulated": d.simulated, "host": _host(), "devices": devices, "totals": _totals(a, devices, None), "exit_code": 0}
        a.label = a.label or f"ci {head_name} · {short_cmd(cmd, 120)}"
        _track(a, report)  # reported, never fails the check
    if summary.get("over_budget"):
        print(warn(f"joules ci: energy grew {summary['change_pct']:+.1f}%, over the {a.budget:g}% budget."), file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- joules compare
def cmd_compare(a) -> int:
    from . import compare as cmp
    cmds = [c for c in a.commands if c.strip()]
    if len(cmds) < 2:
        print('joules compare: give two or more commands in quotes, e.g. joules compare "./bench --runtime hst" "./bench --runtime baseline"', file=sys.stderr)
        return 2
    names = [n.strip() for n in a.names.split(",")] if a.names else None
    if names and (len(names) != len(cmds) or len(set(names)) != len(names) or not all(names)):
        print(f"joules compare: --names needs {len(cmds)} different names, one per command, e.g. --names hst,baseline", file=sys.stderr)
        return 2
    if not names:
        short = [short_cmd(cmp.split_command(c) if not a.shell else [c], 40) for c in cmds]
        names = short if len(set(short)) == len(short) else [chr(65 + i) for i in range(len(cmds))]
    baseline = a.baseline or names[0]
    if baseline not in names:
        print(f"joules compare: --baseline must be one of {', '.join(names)}", file=sys.stderr)
        return 2
    if a.runs < 1 or a.warmup < 0:
        print("joules compare: --runs is at least 1 and --warmup at least 0", file=sys.stderr)
        return 2
    try:
        limits = sorted({int(x) for x in a.limits.split(",") if x.strip()}, reverse=True) if a.limits else [None]
    except ValueError:
        limits = []
    if not limits or (limits[0] is not None and (limits[0] > 100 or limits[-1] < 10)):
        print("joules compare: --limits are percentages of the card's maximum power limit, from 10 to 100, e.g. 100,70", file=sys.stderr)
        return 2
    lock = None
    if a.lock_clocks is not None:
        v = a.lock_clocks.strip().lower()
        lock = "base" if v == "base" else int(v) if v.isdigit() and 100 <= int(v) <= 5000 else None
        if lock is None:
            print("joules compare: --lock-clocks takes an SM clock in MHz, e.g. 1590, or base (the card's base clock)", file=sys.stderr)
            return 2
    argvs = [c if a.shell else cmp.split_command(c) for c in cmds]
    if a.track:
        ensure_key(a)
    d, idle = _setup(a)
    if not d.meters:
        for n in d.notes:
            print(warn("joules: " + n), file=sys.stderr)
        d.close()
        return 1
    gpus = cmp.gpu_meters(d)
    notes = list(d.notes)
    if a.cool_to is not None and not gpus:
        notes.append("--cool-to needs an NVIDIA GPU read through the driver; runs started without waiting.")
    results, blocks_w, locks = [], {}, None
    total = len(limits) * (a.warmup + a.runs) * len(cmds)
    k = 0
    t_first, idle_end = time.monotonic(), {}
    try:
        with contextlib.ExitStack() as stack:
            if lock is not None:  # first, so the clocks are back to normal last, after the power limits
                if not gpus:
                    raise RuntimeError("--lock-clocks needs an NVIDIA GPU read through the driver (joules devices shows what is here)")
                locks = stack.enter_context(cmp.locked_clocks(gpus, lock, cmp.log))
            set_limit = stack.enter_context(cmp.power_limits(gpus)) if limits[0] is not None else None
            if limits[0] is not None and not gpus:
                raise RuntimeError("--limits needs an NVIDIA GPU read through the driver (joules devices shows what is here)")
            for lim in limits:
                if set_limit:
                    blocks_w[lim] = set_limit(lim)
                    cmp.log(f"power limit {lim}% ({', '.join(f'{w:.0f} W' for w in blocks_w[lim])})")
                for rnd, idx in enumerate(cmp.order(names, a.warmup, "alternate") + cmp.order(names, a.runs, a.order)):  # the measured rounds start the pattern
                    for pos, i in enumerate(idx):
                        k += 1
                        warm = rnd < a.warmup
                        if a.rest > 0 and k > 1:
                            time.sleep(a.rest)
                        t0 = cmp.wait_cool(gpus, a.cool_to, a.cool_timeout, cmp.log) if (a.cool_to is not None and gpus) else cmp.hottest(gpus)
                        cmp.log(f"{names[i]}, {'warm-up' if warm else f'run {rnd - a.warmup + 1} of {a.runs}'} ({k}/{total})")
                        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                        r = cmp.measure(d, argvs[i], a.shell, a.interval, a.quiet)
                        if r["exit_code"] != 0:
                            raise RuntimeError(f"{names[i]} failed (exit {r['exit_code']}): {cmds[i]}")
                        if warm:
                            continue
                        r.update(name=names[i], round=rnd - a.warmup + 1, position=pos + 1, started_at=started, limit_pct=lim)
                        if locks:
                            r["locked_mhz"] = dict(locks)
                        if t0 is not None:
                            r["temp_start_c"] = t0
                        if idle:
                            r["_mid"] = time.monotonic() - r["duration_s"] / 2
                        results.append(r)
            if idle and results:  # idle again at the end: a machine that settles (or wakes up) moves the baseline
                cmp.log(f"measuring idle power again for {a.subtract_idle:g} s…")
                idle_end = idle_baseline(d.meters, a.subtract_idle, a.interval)
                notes += cmp.idle_above(results, sum(idle.values()), sum(idle_end.values()), t_first, time.monotonic())
    except RuntimeError as e:
        print(warn(f"joules compare: {e}"), file=sys.stderr)
        d.close()
        return 1
    except KeyboardInterrupt:
        d.close()
        print(warn("joules compare: stopped."), file=sys.stderr)
        return 130
    blocks = cmp.summarize(results, names, baseline, a.cool_to, locks)
    for b in blocks:
        if b["limit_pct"] is not None:
            b["limit_w"] = blocks_w[b["limit_pct"]]
    devices = [m.name for m in d.meters]
    kinds = {m.name: (m.kind, m.method) for m in d.meters}
    d.close()
    report = {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "compare", "names": names, "commands": cmds,
              "baseline": baseline, "runs": a.runs, "warmup": a.warmup, "order": "A B B A" if a.order == "abba" else "alternating",
              "rest_s": a.rest, "cool_to_c": a.cool_to, "limits_pct": None if limits[0] is None else limits,
              "locked_mhz": locks, "gpu_clocks": "locked" if locks else "boost" if gpus else None,
              "idle_w": idle or None, "idle_w_end": idle_end or None, "simulated": d.simulated, "host": _host(), "devices": devices,
              "blocks": blocks, "results": results, "notes": notes}
    _emit(a, report, cmp.render(report))
    if a.markdown:
        with open(a.markdown, "a", encoding="utf-8") as f:
            f.write(cmp.markdown(report))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(cmp.csv_rows(report))
    if a.track:
        reports = []
        for r in results:  # each measured run goes to Tracking under its command's name, with what the GPU was doing
            devs = []
            for name, j in r["devices"].items():
                kind, method = kinds[name]
                row = {"name": name, "kind": kind, "method": method, "energy_j": j, "avg_w": j / r["duration_s"] if r["duration_s"] else 0.0, "peak_w": 0.0}
                if (r.get("gpu") or {}).get(name):
                    row["gpu"] = r["gpu"][name]
                    v = gpu_verdict(row["gpu"])
                    if v:
                        row["verdict"] = v
                devs.append(row)
            rep_ = {"kind": "run", "command": cmds[names.index(r["name"])].split()[:1] or ["compare"], "started_at": r["started_at"],
                    "duration_s": r["duration_s"], "simulated": d.simulated, "host": _host(), "devices": devs,
                    "totals": _totals(a, devs, None), "exit_code": 0}
            at = [f"{r['limit_pct']}% power"] if r.get("limit_pct") is not None else []
            if r.get("locked_mhz"):
                at.append(f"{cmp.lock_text(r['locked_mhz'])} locked")
            reports.append((rep_, r["name"] + (f" @ {', '.join(at)}" if at else "")))
        _track_many(a, reports)  # reported, never changes the exit code: the comparison itself succeeded
    return 0


def _track_many(a, reports: list) -> int:
    """Send many runs, waiting when the plan's per-minute limit says so (a Free account allows 10 a minute), so a
    long comparison never loses runs. Stops, and says how many went, at a monthly quota or any other error."""
    key = ensure_key(a)
    if not key:
        print(warn(NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
        return 2
    sent, waited = 0, False
    for rep_, label in reports:
        for attempt in range(10):
            try:
                sharing.send_track(sharing.track_record(rep_, a.track, label), key, a.api_url)
                sent += 1
                break
            except sharing.ServerError as e:
                if e.status == 429 and e.code == "rate_limited" and attempt < 9:
                    wait = min(65.0, max(1.0, e.retry_after or 10.0))
                    if not waited:
                        print(dim(f"joules: your plan takes a limited number of runs a minute; sending the rest as it allows (about {wait:.0f} s)…"), file=sys.stderr)
                        waited = True
                    time.sleep(wait)
                    continue
                print(warn(f"joules track: {sent} of {len(reports)} runs tracked, then: {e}"), file=sys.stderr)
                return 1
            except (ValueError, RuntimeError) as e:
                print(warn(f"joules track: {sent} of {len(reports)} runs tracked, then: {e}"), file=sys.stderr)
                return 1
    print(good(f"joules: tracked {sent} run{'s' if sent != 1 else ''} in project {a.track}") + f"  {accent(sharing.track_link(a.track, a.api_url))}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- joules proxy
def cmd_proxy(a) -> int:
    from . import proxy
    d, idle = _setup(a)
    if not d.meters:
        for n in d.notes:
            print(warn("joules proxy: " + n), file=sys.stderr)
        return 1
    ci_ = a.ci if a.ci is not None else GRIDS[a.grid]

    def log(r: dict) -> None:
        if a.quiet:
            return
        tok = f" · {r['output_tokens']} tokens · {r['j_per_token']:.3g} J/token" if "output_tokens" in r else ""
        print(f"{dim(r['at'][11:19])} {r['method']} {r['path']} {r['status']}{(' ' + r['model']) if r['model'] else ''}"
              f" · {fmt_s(r['duration_s'])} · {bold(fmt_j(r['energy_j']))}{tok}", file=sys.stderr)

    try:
        srv = proxy.ProxyServer((a.host, a.port), a.upstream, d.meters, idle_w=sum(idle.values()) if idle else None,
                                price=a.price, g_per_kwh=ci_, pue=a.pue, interval=a.interval, log=log)
    except (OSError, ValueError) as e:
        d.close()
        print(f"joules proxy: {e}", file=sys.stderr)
        return 2
    stop = threading.Event()
    tracker = None
    if a.track:
        key = ensure_key(a)
        if not key:
            print(warn("joules proxy: --track needs an API key. " + NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
            srv.server_close(); d.close()
            return 2
        gpus = [m for m in d.meters if m.kind == "gpu"]
        device = ", ".join(sorted({m.name for m in gpus})) if gpus else sharing.cpu_name()
        tracker = proxy.Tracker(srv, a.track, key, a.api_url, a.track_every, device, max(1, len(gpus)),
                                "+".join(sorted({m.kind for m in d.meters})))
        threading.Thread(target=tracker.loop, args=(stop,), daemon=True).start()
    here = f"http://{a.host}:{srv.server_address[1]}"
    print(f"{bold('joules proxy')} {here} → {a.upstream}", file=sys.stderr)
    print(f"  measuring: {', '.join(m.name for m in d.meters)}", file=sys.stderr)
    print(f"  point your client here: Ollama {accent(here)} · OpenAI-compatible base_url {accent(here + '/v1')}", file=sys.stderr)
    print(f"  receipts {here}/joules/receipts · Prometheus {here}/joules/metrics" + (f" · tracking to project {a.track}" if a.track else ""), file=sys.stderr)
    for n in d.notes:
        print(warn("  ! " + n), file=sys.stderr)
    try:
        srv.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if tracker:
            for e in tracker.flush():
                print(warn(f"joules proxy: tracking failed: {e}"), file=sys.stderr)
        srv.server_close()
        srv.ledger.close()
        d.close()
        st = srv.stats
        print(f"\njoules proxy: {st['requests']} requests · {fmt_j(st['energy_j'])} · {st['output_tokens']:,} output tokens", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- joules agent
def _keep_awake():
    """Stop this machine from sleeping while it is idle, for as long as this process runs. Returns the helper
    process (ended with the agent), or None with a warning when the system has no way to do it."""
    pid = str(os.getpid())
    if platform.system() == "Darwin":
        cmd = ["caffeinate", "-i", "-w", pid]
    else:
        cmd = ["systemd-inhibit", "--what=idle:sleep", "--who=joules", "--why=joules agent is measuring this machine",
               "--mode=block", "tail", "--pid", pid, "-f", "/dev/null"]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        print(warn(f"joules agent: --keep-awake needs {cmd[0]}, which is not available here; the machine may still sleep"), file=sys.stderr)
        return None


def cmd_agent(a) -> int:
    import queue
    import signal
    from . import agent as ag
    from .share import _device_name
    per_job = os.path.isdir(ag.proc_root())
    if not per_job and platform.system() != "Darwin":
        print("joules agent: needs Linux (/proc) or macOS. On other systems use joules run or joules proxy.", file=sys.stderr)
        return 2
    if a.serve_metrics not in ("auto", "off") and not a.serve_metrics.startswith(("http://", "https://")):
        print("joules agent: --serve-metrics is auto, off, or an http:// address such as http://127.0.0.1:8000/metrics", file=sys.stderr)
        return 2
    if a.live_every and a.live_every < 2:
        print("joules agent: --live-every is 0 (off) or at least 2 seconds", file=sys.stderr)
        return 2
    d = discover(_gpu_list(a.gpus), cpu=not a.no_cpu)
    for n in d.notes:
        print(warn("joules agent: " + n), file=sys.stderr)
    if not d.meters:
        return 1
    key = ensure_key(a) if a.track else sharing.api_key(a.key)
    if a.track:
        if not sharing.PROJECT_RE.match(a.track):
            print("joules agent: project names use letters, digits, dot, dash or underscore (1 to 60 characters)", file=sys.stderr)
            d.close(); return 2
        if not key:
            print(warn("joules agent: --track needs an API key. " + NO_KEY.format(site=sharing.api_url(a.api_url))), file=sys.stderr)
            d.close(); return 2
    g = a.ci if a.ci is not None else GRIDS[a.grid]
    names = {m.name: m for m in d.meters}
    sent: queue.Queue = queue.Queue()

    def sender() -> None:
        while True:
            r = sent.get()
            if r is None:
                return
            try:
                sharing.send_track(r, key, a.api_url)
            except (ValueError, RuntimeError, OSError) as e:
                print(warn(f"joules agent: tracking {r['label']} failed: {e}"), file=sys.stderr)

    def finished(j, row: dict) -> None:
        if row["duration_s"] < a.min_seconds or j.key.kind in ("system", "idle"):
            return
        if not a.quiet:
            print(f"{dim(time.strftime('%H:%M:%S'))} finished {j.key.kind} {bold(j.key.name)}" + (f" ({j.key.user})" if j.key.user else "")
                  + f" · {fmt_s(row['duration_s'])} · {bold(fmt_j(row['facility_j']))} · {_money(row['cost'])} · {fmt_g(row['co2_g'])} g CO2"
                  + (f" · {fmt_j(j.idle_j)} on idle GPUs" if j.idle_j > 0.05 * j.energy_j and j.idle_j > 0 else "")
                  + f" · {ATTRIBUTION_WORDS.get(row['attribution'], row['attribution'])}"
                  + (warn(f" · only {row['coverage'] * 100:.0f}% of its time sampled") if row["coverage"] < 0.95 else ""), file=sys.stderr)
        if a.track:
            gpus = [names[x] for x in row["gpus"] if x in names]
            if gpus:
                c = {}
                for m in gpus:
                    c[_device_name(m.name)] = c.get(_device_name(m.name), 0) + 1
                device = ", ".join(f"{n}× {k}" if n > 1 else k for k, n in c.items())
            else:
                device = sharing.cpu_name()
            r = ag.track_record(j, row, a.track, device, len(gpus), a.price, g)
            r["simulated"] = d.simulated
            sent.put(r)

    agent = ag.Agent(d.meters, interval=a.interval, idle_util=a.idle_util, finish_after=a.finish_after,
                     price=a.price, g_per_kwh=g, pue=a.pue, on_finish=finished, host=a.name)
    try:
        srv = ag.AgentServer((a.host, a.port), agent)
    except OSError as e:
        d.close()
        print(f"joules agent: cannot listen on {a.host}:{a.port} ({e})", file=sys.stderr)
        return 2
    def live() -> None:
        """A heartbeat every --live-every seconds for the Live view; stops for good if the plan has no Live. Over the
        plan's GPUs on Live (Free 1, a paid plan the GPUs bought), it says so once and tries again every minute, so it joins when a slot frees."""
        failing = False
        shown = False
        limited = False
        while not agent._stop.wait(60 if limited else a.live_every):
            try:
                r = sharing.send_live(agent.live_beat(), key, a.api_url)
            except sharing.ServerError as e:
                fields = ((e.body or {}).get("error") or {}).get("fields") if isinstance(e.body, dict) else None
                if e.status == 400 and agent.gpu_detail and isinstance(fields, dict) and any(k.startswith("gpu_list") for k in fields):
                    agent.gpu_detail = False  # a server from before per-GPU Live: send the machine's totals, as before
                    print(dim("joules agent: this WattOps server does not take per-GPU readings yet; sending each machine's totals"), file=sys.stderr)
                    continue
                if e.status == 400 and agent.serve is not None and isinstance(fields, dict) and any(k.startswith("serve") for k in fields):
                    agent.serve = None  # a server from before tokens on Live: send power only, as before
                    print(dim("joules agent: this WattOps server does not take tokens per second yet; sending power only"), file=sys.stderr)
                    continue
                if e.code == "gpu_limit":
                    if not limited:
                        print(warn(f"joules agent: not on Live: {e} Trying again every minute."), file=sys.stderr)
                    limited = True
                    continue
                if e.code in ("plan_required", "invalid_api_key", "missing_api_key"):
                    msg = (f"Live is not on your plan ({e}), so this node is not on the Live page. Finished jobs still go to Tracking."
                           if e.code == "plan_required" else f"Live stopped: {e}")
                    print(warn(f"joules agent: {msg}"), file=sys.stderr)
                    return
                if agent._stop.is_set():
                    return  # the agent stopped while this heartbeat was on its way
                if not failing:
                    print(warn(f"joules agent: Live heartbeat failed ({e}); retrying"), file=sys.stderr)
                failing = True
                continue
            except (RuntimeError, ValueError, OSError) as e:
                if agent._stop.is_set():
                    return  # the agent stopped while this heartbeat was on its way
                if not failing:
                    print(warn(f"joules agent: Live heartbeat failed ({e}); retrying"), file=sys.stderr)
                failing = True
                continue
            if failing or limited:
                print(dim("joules agent: Live heartbeats are getting through again"), file=sys.stderr)
            failing = limited = False
            if not shown and not r.get("skipped"):
                print(f"  live: {accent(sharing.live_link(a.api_url))}", file=sys.stderr)
                shown = True

    awake = _keep_awake() if a.keep_awake else None
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True).start()
    worker = threading.Thread(target=sender, daemon=True)
    worker.start()
    if a.track and a.live_every:
        if a.serve_metrics != "off":
            from .serving import ServeMetrics
            agent.serve = ServeMetrics(a.serve_metrics)
        threading.Thread(target=live, daemon=True).start()
    old = {sig: signal.signal(sig, lambda *_: agent.stop()) for sig in (signal.SIGTERM, signal.SIGINT)}
    here = f"http://{a.host}:{srv.server_address[1]}"
    print(f"{bold('joules agent')} on {agent.host} · {here}/metrics (Prometheus) · {here}/jobs", file=sys.stderr)
    print(f"  measuring: {', '.join(m.name for m in d.meters)}", file=sys.stderr)
    nv = any(getattr(m, "nvml", None) is not None for m in d.meters)
    if not per_job:
        print(dim("  this machine is measured as a whole (energy per job needs Linux)"), file=sys.stderr)
    else:
        print(dim("  a GPU shared by jobs is split by SM time (else by memory held); CPU and DRAM by CPU time. Each job says which." if nv else
                  "  CPU and DRAM energy is split by CPU time; a GPU goes to the one active job."), file=sys.stderr)
    if a.track:
        print(f"  finished jobs of {fmt_s(a.min_seconds)} or more go to project {a.track}: {accent(sharing.track_link(a.track, a.api_url))}", file=sys.stderr)
    if awake:
        print(dim("  keeping this machine awake while the agent runs (closing a laptop's lid still puts it to sleep)"), file=sys.stderr)
    elif a.track and a.live_every and platform.system() == "Darwin":
        print(dim("  a sleeping Mac sends nothing, so Live shows a gap while it sleeps; --keep-awake keeps it awake"), file=sys.stderr)
    try:
        agent.run(a.duration)
    finally:
        if awake:
            awake.terminate()
            try:
                awake.wait(timeout=5)
            except subprocess.TimeoutExpired:
                awake.kill()
        srv.shutdown(); srv.server_close()
        sent.put(None); worker.join(timeout=30)
        d.close()
        for sig, h in old.items():
            signal.signal(sig, h)
    return 0


# ---------------------------------------------------------------- joules connect
def cmd_connect(a) -> int:
    from . import connection as cn
    if a.stop:
        cn.stop()
        return 0
    if not (a.key or sharing.api_key()) and cn.in_shell_cell():
        print("joules connect: a notebook's ! cell cannot take typing, so it cannot ask for your key. Run it in a Python "
              "cell instead:\n  import joules; joules.connect()", file=sys.stderr)
        return 2
    try:
        cn.connect(name=a.name, key=a.key, project=a.project, board=not a.no_board, agent=not a.no_agent, gpu=a.gpu,
                   price=a.price, grid=a.grid, pue=a.pue, live_every=a.live_every, save=not a.no_save,
                   service=False if a.no_service else None, api_url=a.api_url)
    except cn.ConnectError as e:
        print(warn(f"joules connect: {e}"), file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"joules connect: {e}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------- joules login / logout
def cmd_login(a) -> int:
    key = (a.key_arg or "").strip()
    if not key:
        if not sys.stdin.isatty():
            print("joules login: give the key, e.g. joules login lgk_…", file=sys.stderr)
            return 2
        import getpass
        key = getpass.getpass("WattOps API key (from Settings → API keys): ").strip()
    if not sharing.KEY_RE.match(key):
        print(warn("joules login: that is not an API key. Keys look like lgk_ followed by 40 letters and digits; "
                   f"create one at {sharing.api_url(a.api_url)}/#/settings/keys."), file=sys.stderr)
        return 2
    try:
        sharing.check_key(key, a.api_url)
    except RuntimeError as e:
        print(warn(f"joules login: the key was not accepted: {e}"), file=sys.stderr)
        return 1
    path = sharing.save_key(key)
    print(good("joules: signed in.") + f" Key saved to {path} (only you can read it).", file=sys.stderr)
    print(f"  From now on every joules run and joules bench on this computer appears on your Tracking page:\n"
          f"  {accent(sharing.track_link('default', a.api_url))}  (add --no-track to keep one run on this computer)", file=sys.stderr)
    print(dim("  try: joules -- python3 -c \"sum(i*i for i in range(30_000_000))\""), file=sys.stderr)
    return 0


def cmd_logout(a) -> int:
    try:
        os.remove(sharing.key_file())
        print("joules: signed out; the saved key was removed from this computer.", file=sys.stderr)
    except FileNotFoundError:
        print("joules: no saved key on this computer.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- joules calibrate
def cmd_calibrate(a) -> int:
    """Read the meters for a fixed time (or while a command runs) and compare their total with a wall-plug meter's
    reading over the same time. The meters cover the devices (GPU board, CPU package, DRAM, the Apple SoC); the
    wall also sees the power supply's losses, fans, storage, the board and, on a laptop, the screen. The share
    the meters cover is what a calibration records: docs/CALIBRATION.md."""
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if sum(x is not None for x in (a.wall_wh, a.wall_w, a.wall_csv)) > 1:
        print(warn("joules calibrate: give one of --wall-wh, --wall-w or --wall-csv."), file=sys.stderr)
        return 2
    if a.report and not a.wall_csv:
        print(warn("joules calibrate: --report needs --wall-csv, the meter log to compare it with."), file=sys.stderr)
        return 2
    if a.report:
        try:
            with open(a.report, encoding="utf-8") as f:
                report = json.load(f)
            if report.get("kind") != "calibrate" or "ended_unix" not in report:
                raise ValueError("not a joules calibrate --out file from joules 0.8 or later")
        except (OSError, ValueError) as e:
            print(warn(f"joules calibrate: cannot use {a.report}: {e}"), file=sys.stderr)
            return 2
        report["notes"] = [n for n in report.get("notes", []) if not n.startswith("The meters report more")]
        err = _wall_from_csv(a.wall_csv, report)
        if err:
            print(warn(f"joules calibrate: {err}"), file=sys.stderr)
            return 2
        _emit(a, report, render_calibrate(report))
        return 0
    if (a.wall_wh is not None and a.wall_wh <= 0) or (a.wall_w is not None and a.wall_w <= 0):
        print(warn("joules calibrate: the wall reading must be above zero."), file=sys.stderr)
        return 2
    d, _ = _setup(a)
    if not d.meters:
        for n in d.notes:
            print(warn("joules: " + n), file=sys.stderr)
        print(warn("joules calibrate: nothing to measure on this machine, so there is nothing to compare."), file=sys.stderr)
        return 1
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    t_start = time.time()
    code = 0
    try:
        with Sampler(d.meters, a.interval) as s:
            if cmd:
                proc = _start(cmd)
                if not proc:
                    return 126 if proc is False else _not_found(cmd)
                for m in d.meters:
                    if hasattr(m, "attach"):
                        m.attach(proc)
                while True:
                    try:
                        code = proc.wait()
                        break
                    except KeyboardInterrupt:
                        continue
            else:
                print(dim(f"joules: reading the meters for {a.seconds:g} s. Note the wall meter now and when this ends."), file=sys.stderr)
                end = time.monotonic() + a.seconds
                while True:
                    left = end - time.monotonic()
                    if left <= 0:
                        break
                    try:
                        time.sleep(min(0.2, left))
                    except KeyboardInterrupt:
                        break
    finally:
        d.close()
    t_end = t_start + s.duration_s
    devices = _devices(d, {}, s.duration_s)
    metered = sum(x["energy_j"] for x in devices)
    report = {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "calibrate", "command": cmd, "exit_code": code,
              "started_at": started, "started_unix": round(t_start, 3), "ended_unix": round(t_end, 3),
              "duration_s": s.duration_s, "simulated": d.simulated, "host": _host(), "devices": devices,
              "metered": {"energy_j": metered, "avg_w": metered / s.duration_s if s.duration_s else 0.0}, "notes": list(d.notes)}
    if a.wall_csv:
        err = _wall_from_csv(a.wall_csv, report)
        if err:
            print(warn(f"joules calibrate: {err}. The meters' reading is below; compare later with --report and --wall-csv."), file=sys.stderr)
    else:
        wall_j = a.wall_wh * 3600 if a.wall_wh is not None else a.wall_w * s.duration_s if a.wall_w is not None else None
        if wall_j is not None:
            _wall(report, wall_j)
    _emit(a, report, render_calibrate(report))
    return code


def _wall(report: dict, wall_j: float, how: dict | None = None) -> None:
    """Add the wall reading to a calibrate report: its energy, and the share of it the meters cover."""
    metered = report["metered"]["energy_j"]
    d = (report["ended_unix"] - report["started_unix"]) if how and "ended_unix" in report else report["duration_s"]  # the log's own window
    report["wall"] = {"energy_j": wall_j, "avg_w": wall_j / d if d else 0.0, "metered_share_pct": metered / wall_j * 100 if wall_j else 0.0,
                      "unmetered_j": wall_j - metered, **(how or {})}
    if report.get("simulated"):
        report["notes"].append("Simulated meters: this comparison says nothing about real hardware.")
    if metered > wall_j * 1.02:
        report["notes"].append("The meters report more than the wall: the readings do not cover the same time, or the wall figure is wrong.")


def _wall_from_csv(path: str, report: dict) -> str | None:
    """Integrate a meter log over the report's own seconds and add it as the wall reading. Returns an error, or None."""
    from . import walllog
    try:
        got = walllog.integrate(walllog.read(path), report["started_unix"], report["ended_unix"])
    except (OSError, ValueError) as e:
        return str(e)
    if got["coverage"] < 0.95:
        return (f"the meter log covers only {got['coverage'] * 100:.0f}% of the {report['duration_s']:.0f} s measured "
                "(check the meter's clock against this computer's, and that it logged the whole time)")
    _wall(report, got["energy_j"], {"source": "log", "log_readings": got["readings"], "log_interval_s": round(got["interval_s"], 3),
                                    "log_coverage_pct": round(got["coverage"] * 100, 1)})
    return None


def render_calibrate(r: dict) -> str:
    m = r["metered"]
    head = f"{bold('joules calibrate')} · {fmt_s(r['duration_s'])}" + (f" · {short_cmd(r['command'])} · exit {r['exit_code']}" if r["command"] else "")
    lines = [head, *_device_lines(r["devices"]), dim("  " + "─" * 58),
             f"  {bold('metered')}  {bold(fmt_j(m['energy_j']))} = {fmt_wh(m['energy_j'])}   {fmt_w(m['avg_w'])} W avg"]
    if "wall" in r:
        w = r["wall"]
        lines.append(f"  wall     {fmt_j(w['energy_j'])} = {fmt_wh(w['energy_j'])}   {fmt_w(w['avg_w'])} W avg"
                     + (dim(f"   from {w['log_readings']} logged readings, one every {w['log_interval_s']:g} s") if w.get("source") == "log" else ""))
        lines.append(f"  the meters cover {accent('%.1f%%' % w['metered_share_pct'])} of the wall; "
                     f"{fmt_j(max(0.0, w['unmetered_j']))} went to what they cannot see")
        lines.append(dim("  not metered: the power supply's losses, fans, storage, the board; on a laptop the screen and charging"))
        lines.append(dim("  record this share with the machine, PSU and meter in docs/CALIBRATION.md; --pue sets the overhead to use"))
    else:
        lines.append(dim(f"  read a wall-plug meter over the same {fmt_s(r['duration_s'])} and rerun with --wall-wh <Wh> (or --wall-w <average W>),"))
        lines.append(dim("  or give a logging meter's readings: --wall-csv log.csv (now, or later with --report this.json)"))
    for n in r["notes"]:
        lines.append(warn("  ! " + n))
    return "\n".join(lines)


# ---------------------------------------------------------------- joules tune
def cmd_tune(a) -> int:
    """Benchmark a model at several GPU power limits and say which one gives the cheapest token without slowing
    it down by more than --max-slowdown. Needs NVIDIA GPUs and permission to set the limit (root). The original
    limit is put back whatever happens."""
    try:
        target = parse_target(a.target, a.url)
        limits = sorted({int(x) for x in a.limits.split(",") if x.strip()}, reverse=True)
    except ValueError as e:
        print(f"joules tune: {e}", file=sys.stderr)
        return 2
    if not limits or limits[0] > 100 or limits[-1] < 10:
        print("joules tune: --limits are percentages of the card's maximum power limit, from 10 to 100, e.g. 100,85,70,55", file=sys.stderr)
        return 2
    d, _ = _setup(a)
    gpus = [m for m in d.meters if m.kind == "gpu" and hasattr(m, "nvml") and hasattr(m.nvml, "set_power_limit_mw")]
    if not gpus:
        for n in d.notes:
            print(warn("joules: " + n), file=sys.stderr)
        print(warn("joules tune: needs an NVIDIA GPU measured through the driver (joules devices shows what is here)."), file=sys.stderr)
        d.close()
        return 1
    ranges, original = [], []
    for m in gpus:
        rng, cur = m.nvml.power_limit_range_mw(m.h), m.nvml.power_limit_mw(m.h)
        if not rng or not cur:
            print(warn(f"joules tune: {m.name}: the driver does not report its power limits."), file=sys.stderr)
            d.close()
            return 1
        ranges.append(rng); original.append(cur)
    try:
        ensure_model(target, lambda t: print(dim("joules: " + t), file=sys.stderr))
        for _ in range(a.warmup):
            generate(target, a.prompt, min(a.max_tokens, 32))
    except RuntimeError as e:
        print(f"joules tune: {e}", file=sys.stderr)
        d.close()
        return 1
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows, code = [], 0
    try:
        for pct in limits:
            watts = []
            for m, (lo, hi) in zip(gpus, ranges):
                mw = max(lo, min(hi, int(hi * pct / 100)))
                rc = m.nvml.set_power_limit_mw(m.h, mw)
                if rc != 0:
                    why = "no permission: run joules tune with sudo" if rc == 4 else f"the driver refused (NVML error {rc})"
                    print(warn(f"joules tune: could not set the power limit on {m.name}: {why}."), file=sys.stderr)
                    return 1
                watts.append(mw / 1000)
            print(dim(f"joules: power limit {pct}% ({', '.join(f'{w:.0f} W' for w in watts)}), {a.runs} run{'s' if a.runs != 1 else ''}…"), file=sys.stderr)
            out_tok, gen_s = 0, 0.0
            with Sampler(d.meters, a.interval) as s:
                for _ in range(a.runs):
                    g = generate(target, a.prompt, a.max_tokens)
                    out_tok += g["output_tokens"]; gen_s += g["gen_s"] or 0
            if out_tok == 0:
                print("joules tune: the server returned no output tokens.", file=sys.stderr)
                return 1
            gpu_j, total_j = sum(m.energy_j for m in gpus), sum(m.energy_j for m in d.meters)
            rows.append({"limit_pct": pct, "limit_w": watts, "output_tokens": out_tok, "duration_s": s.duration_s,
                         "tokens_per_s": out_tok / (gen_s or s.duration_s), "j_per_token": total_j / out_tok, "gpu_j_per_token": gpu_j / out_tok,
                         "gpu_avg_w": gpu_j / s.duration_s if s.duration_s else 0.0, "gpu": [m.health() for m in gpus]})
    except RuntimeError as e:
        print(f"joules tune: {e}", file=sys.stderr)
        code = 1
    finally:
        for m, mw in zip(gpus, original):  # the card is left as it was found
            try:
                m.nvml.set_power_limit_mw(m.h, mw)
            except Exception:
                pass
        d.close()
    if code or not rows:
        return code or 1
    report = {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "tune", "target": f"{target.kind}:{target.model}",
              "started_at": started, "simulated": d.simulated, "host": _host(), "gpus": [m.name for m in gpus],
              "original_limit_w": [mw / 1000 for mw in original], "max_slowdown_pct": a.max_slowdown, "rows": rows,
              "pick": power_sweep_pick(rows, a.max_slowdown), "notes": list(d.notes)}
    _emit(a, report, render_tune(report))
    return 0


def render_tune(r: dict) -> str:
    lines = [f"{bold('joules tune')} · {r['target']} · {', '.join(r['gpus'])}",
             f"  {'limit':>6}  {'W':>6}  {'tok/s':>7}  {'J/token':>8}  {'GPU W avg':>9}"]
    for x in r["rows"]:
        lines.append(f"  {x['limit_pct']:>5}%  {sum(x['limit_w']):>6.0f}  {x['tokens_per_s']:>7.1f}  {x['j_per_token']:>8.3g}  {x['gpu_avg_w']:>9.0f}")
    p = r["pick"]
    if p and p["limit_pct"] < p["base_limit_pct"]:
        lines.append(f"  {good('best')}: {p['limit_pct']}% of the limit uses {accent('%.0f%%' % p['energy_saved_pct'])} less energy per token "
                     f"and is {p['slowdown_pct']:.0f}% slower (within --max-slowdown {r['max_slowdown_pct']:g}%)")
        lines.append(dim(f"  set it with: sudo nvidia-smi -pl {sum(next(x['limit_w'] for x in r['rows'] if x['limit_pct'] == p['limit_pct'])) / max(1, len(r['gpus'])):.0f}"
                         f"   (per GPU; nvidia-smi -pl {r['original_limit_w'][0]:.0f} puts it back)"))
    elif p:
        lines.append("  the full limit is the best within --max-slowdown: lowering it costs more speed than it saves")
    lines.append(dim("  the limit was put back to what it was"))
    for n in r["notes"]:
        lines.append(warn("  ! " + n))
    return "\n".join(lines)


# ---------------------------------------------------------------- joules doctor
def cmd_doctor(a) -> int:
    from . import doctor
    r = doctor.run(_gpu_list(a.gpus), a.seconds, a.api_url, a.offline)
    data = r.as_dict()
    enc = (getattr(sys.stdout, "encoding", None) or "ascii")
    try:
        "✓✗·".encode(enc)
        ascii_only = False
    except (UnicodeEncodeError, LookupError):
        ascii_only = True
    text = doctor.render(r, color=sys.stdout.isatty() and not os.environ.get("NO_COLOR"), ascii_only=ascii_only)
    print(json.dumps(data, indent=2) if a.json else text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) if a.out.endswith(".json") else doctor.render(r, color=False, ascii_only=False) + "\n")
        print(dim(f"joules doctor: report saved to {a.out}"), file=sys.stderr)
    return 1 if r.counts()[doctor.FAIL] else 0


# ---------------------------------------------------------------- joules devices
def cmd_devices(a) -> int:
    d = discover(_gpu_list(a.gpus), cpu=not a.no_cpu)
    try:
        if a.json:
            print(json.dumps({"devices": [{"name": m.name, "kind": m.kind, "method": m.method} for m in d.meters], "notes": d.notes}, indent=2))
        else:
            print(bold("joules can measure:") if d.meters else bold("joules found nothing to measure."))
            for m in d.meters:
                print(f"  {m.name:<34} {dim(m.method)}")
            for n in d.notes:
                print(warn("  ! " + n))
    finally:
        d.close()
    return 0 if d.meters else 1


def cmd_telemetry(a) -> int:
    if a.action in ("on", "off"):
        telemetry.set_enabled(a.action == "on")
    for line in telemetry.status_lines():
        print(line)
    return 0


# How a finished job's energy was split, in the words the agent prints and the website shows.
ATTRIBUTION_WORDS = {"whole_gpu": "whole GPU", "whole_device": "whole device", "sm_time": "split by SM time",
                     "busy_time": "split by GPU busy time", "memory_share": "split by GPU memory held",
                     "cpu_time": "split by CPU time", "cpu_guess": "GPU given to the busiest job (no per-process data)"}


# ---------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="joules", description="Measure the energy, cost and carbon of any command or local model.",
                                epilog="Examples:\n  joules -- python train.py\n  joules run --params 8 --tokens 0.5 -- python finetune.py\n  joules bench ollama:llama3.1:8b --api-price 0.60\n  joules bench ollama:llama3.1:8b --share\n  joules bench gpu --share",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"joules {__version__}")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="measure a command", description="Measure the energy of a command. Its exit code is passed through.")
    common(r)
    w = r.add_argument_group("workload (optional: adds Landauer gap, estimate vs measured, calibration)")
    w.add_argument("--params", type=float, help="model parameters in billions")
    w.add_argument("--tokens", type=float, help="training tokens in billions")
    w.add_argument("--flops", type=float, help="total operations, if you know them")
    w.add_argument("--gpu", choices=sorted(GPUS), help="GPU model, if it is not detected")
    r.add_argument("--estimate", action="store_true", help=argparse.SUPPRESS)  # the default now: estimate when there are no counters
    sharing_args(r)
    r.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bench", help="energy per token of a local model server")
    b.add_argument("target", help="ollama:<model> or openai:<model>, or gpu: the same one-minute test on any GPU, for the GPU leaderboard")
    b.add_argument("--url", help="server URL (default localhost Ollama or http://localhost:8000/v1)")
    b.add_argument("--runs", type=int, default=5, help="measured requests (default 5)")
    b.add_argument("--warmup", type=int, default=1, help="unmeasured requests first, so model loading is not counted")
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--prompt", default=DEFAULT_PROMPT)
    b.add_argument("--params", type=float, help="active parameters in billions (adds the Landauer gap)")
    b.add_argument("--api-price", type=float, help="an API's price per 1M output tokens, to compare against")
    b.add_argument("--share", action="store_true", help="add the result to the public leaderboard (needs an API key; measures idle power first)")
    common(b)
    sharing_args(b)
    b.set_defaults(fn=cmd_bench)

    sh = sub.add_parser("share", help="add a saved bench result (--out FILE) to the leaderboard")
    sh.add_argument("file", help="JSON written by joules bench --out FILE")
    sharing_args(sh, track=False)
    sh.set_defaults(fn=cmd_share)

    c = sub.add_parser("ci", help="energy check for a pull request: this change vs the base branch",
                       description="Run a command on this checkout and on the base branch (a temporary git worktree), "
                                   "compare the median energy, write a Markdown summary and optionally comment on the pull request.")
    c.add_argument("--base", help="git ref to compare with (default: origin/$GITHUB_BASE_REF in GitHub Actions, else no comparison)")
    c.add_argument("--runs", type=int, default=3, help="measured runs per side (default 3)")
    c.add_argument("--warmup", type=int, default=0, help="unmeasured runs per side first")
    c.add_argument("--setup", help="shell command to prepare each checkout first, e.g. 'pip install -e .'")
    c.add_argument("--budget", type=float, help="fail when energy grows by more than this many percent")
    c.add_argument("--comment", action="store_true", help="post or update a comment on the pull request (needs GITHUB_TOKEN)")
    c.add_argument("--markdown", metavar="FILE", help="also append the summary to FILE")
    common(c)
    sharing_args(c)
    c.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    c.set_defaults(fn=cmd_ci)

    cp = sub.add_parser("compare", help="the same work done two or more ways: interleaved runs, medians, p-values",
                        description="Run each command several times, interleaved so drift and heat hit every command equally, after "
                                    "unmeasured warm-ups; optionally from the same GPU temperature (--cool-to) and at fixed GPU power "
                                    "limits (--limits, needs sudo). Reports the median energy, time, average power and energy x time of "
                                    "each, what the GPU was doing, the change against the baseline with a Mann-Whitney p-value, and "
                                    "a warning when the GPU did not run under the same conditions for each command.")
    cp.add_argument("commands", nargs="+", metavar="COMMAND", help='each command in quotes, e.g. "./bench --runtime hst"')
    cp.add_argument("--names", help="a short name per command, comma-separated (e.g. hst,baseline)")
    cp.add_argument("--baseline", metavar="NAME", help="the command the others are compared with (default: the first)")
    cp.add_argument("--runs", type=int, default=5, help="measured runs per command (default 5)")
    cp.add_argument("--warmup", type=int, default=1, help="unmeasured rounds first (default 1)")
    cp.add_argument("--order", choices=("alternate", "abba"), default="alternate",
                    help="alternate: A B A B; abba: A B B A, which also cancels a steady drift (default alternate)")
    cp.add_argument("--rest", type=float, default=0.0, metavar="SECONDS", help="pause between runs")
    cp.add_argument("--cool-to", type=float, metavar="CELSIUS", help="before each run, wait until every NVIDIA GPU is at or below this temperature")
    cp.add_argument("--cool-timeout", type=float, default=600.0, metavar="SECONDS", help="longest wait for --cool-to (default 600)")
    cp.add_argument("--limits", metavar="PCTS", help="repeat the comparison at these GPU power limits, %% of the maximum, e.g. 100,85,70,55 (needs sudo; put back afterwards)")
    cp.add_argument("--lock-clocks", metavar="MHZ", help="lock every measured GPU's SM clock at MHZ, or base (its base clock), before the first run, "
                                                         "warm-ups included (nvidia-smi -lgc; needs root; always unlocked afterwards with -rgc)")
    cp.add_argument("--shell", action="store_true", help="run each command through the shell (for pipes, && and variables)")
    cp.add_argument("--quiet", action="store_true", help="hide the commands' own output")
    cp.add_argument("--markdown", metavar="FILE", help="also append a Markdown table to FILE")
    cp.add_argument("--csv", metavar="FILE", help="also write one row per measured run to FILE")
    common(cp)
    sharing_args(cp)
    cp.set_defaults(fn=cmd_compare)

    px = sub.add_parser("proxy", help="energy receipt on every response of a model server",
                        description="Put this in front of Ollama, vLLM, llama.cpp, LM Studio or any HTTP model server. "
                                    "Every response gets X-Energy-* headers and a receipt at /joules/receipts/<id>.")
    px.add_argument("--upstream", default="http://localhost:11434", help="the model server (default Ollama, http://localhost:11434)")
    px.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1; 0.0.0.0 exposes it to your network)")
    px.add_argument("--port", type=int, default=8787, help="port to listen on (default 8787)")
    px.add_argument("--quiet", action="store_true", help="do not print a line per request")
    px.add_argument("--track-every", type=float, default=300, metavar="SECONDS", help="with --track, send a summary this often (default 300)")
    common(px)
    sharing_args(px)
    px.set_defaults(fn=cmd_proxy)

    ag = sub.add_parser("agent", help="energy per job on a shared GPU server or cluster node",
                        description="Measure the whole machine all the time and split its energy between the jobs on it "
                                    "(Slurm jobs, Kubernetes pods, containers or users' programs). Serves Prometheus metrics at "
                                    "/metrics and the jobs at /jobs; with --track, each finished job goes to your Tracking page, and the "
                                    "node's power and jobs show on the Live page every few seconds (Free shows 1 GPU on Live; more are paid per GPU).")
    ag.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1; 0.0.0.0 lets Prometheus on another machine scrape it)")
    ag.add_argument("--port", type=int, default=9877, help="port to listen on (default 9877)")
    ag.add_argument("--name", help="this machine's name in metrics (default: host name; in Kubernetes pass the node name)")
    ag.add_argument("--interval", type=float, default=1.0, help="seconds between readings (default 1)")
    ag.add_argument("--gpus", help="only these GPU indices, e.g. 0,1 (default all)")
    ag.add_argument("--no-cpu", action="store_true", help="do not measure CPU and DRAM")
    ag.add_argument("--idle-util", type=float, default=5.0, metavar="PERCENT",
                    help="a job's GPU energy counts as idle while the GPU is under this busy %% (default 5)")
    ag.add_argument("--finish-after", type=float, default=30.0, metavar="SECONDS",
                    help="a job is finished when none of its processes ran for this long (default 30)")
    ag.add_argument("--min-seconds", type=float, default=60.0, help="only report and track finished jobs at least this long (default 60)")
    ag.add_argument("--duration", type=float, metavar="SECONDS", help="stop after this long (default: run until stopped)")
    ag.add_argument("--quiet", action="store_true", help="do not print a line per finished job")
    ag.add_argument("--keep-awake", action="store_true",
                    help="keep this machine from sleeping while the agent runs, so Live has no gaps (macOS: caffeinate; Linux: systemd-inhibit)")
    c2 = ag.add_argument_group("cost and carbon")
    c2.add_argument("--price", type=float, default=0.10, help="electricity price per kWh (default 0.10)")
    c2.add_argument("--grid", choices=sorted(GRIDS), default="us", metavar="NAME", help=GRID_HELP)
    c2.add_argument("--ci", type=float, help="carbon intensity in g CO2/kWh (overrides --grid)")
    c2.add_argument("--pue", type=float, default=1.0, help="facility overhead for cost, carbon and tracked energy (default 1.0)")
    sharing_args(ag, track=False)
    ag.add_argument("--track", metavar="PROJECT", help="send each finished job to your Tracking page under PROJECT (a team key sends it to the team)")
    ag.add_argument("--live-every", type=float, default=5.0, metavar="SECONDS",
                    help="with --track, send this node's power and jobs to the Live page this often (default 5; 0 = off)")
    ag.add_argument("--serve-metrics", default="auto", metavar="URL",
                    help="an inference server's Prometheus page, for tokens per second and cost per token on Live "
                         "(default auto: vLLM on port 8000 or SGLang on 30000 on this machine; off = do not look)")
    ag.set_defaults(fn=cmd_agent)

    cn = sub.add_parser("connect", help="put this machine on WattOps: GPU leaderboard, Live and Tracking, under one name",
                        description="Asks for your API key (hidden), names this machine after where it runs and its GPUs "
                                    "(e.g. kaggle-2x-t4), runs the one-minute GPU test for the leaderboard, and starts joules agent "
                                    "in the background, so the machine's power shows on Live (1 GPU free) and its finished jobs on "
                                    "Tracking, in a project named after it. As root on a machine with systemd, the agent is installed "
                                    "as a service that starts with the machine. Run it again to change the name; --stop stops it. "
                                    "In a notebook, use a Python cell: import joules; joules.connect()")
    cn.add_argument("--name", help="this machine's name on Live and Tracking (default: where it runs and its GPUs, e.g. runpod-h100)")
    cn.add_argument("--project", help="the Tracking project for its jobs (default: its name)")
    cn.add_argument("--no-board", action="store_true", help="skip the one-minute GPU test for the leaderboard")
    cn.add_argument("--no-agent", action="store_true", help="only the GPU test, no agent (nothing on Live or Tracking)")
    cn.add_argument("--no-service", action="store_true", help="as root, run the agent in the background instead of installing the service")
    cn.add_argument("--no-save", action="store_true", help="do not save the key on this computer")
    cn.add_argument("--gpu", type=int, default=0, help="the GPU for the leaderboard test (default 0)")
    cn.add_argument("--live-every", type=float, default=5.0, metavar="SECONDS", help="seconds between Live readings (default 5; 0 = off)")
    cn.add_argument("--price", type=float, help="electricity price per kWh (default 0.10)")
    cn.add_argument("--grid", choices=sorted(GRIDS), metavar="NAME", help=GRID_HELP)
    cn.add_argument("--pue", type=float, help="facility overhead for cost, carbon and tracked energy (default 1.0)")
    cn.add_argument("--stop", action="store_true", help="stop the agent joules connect started (and, as root, disable the service)")
    cn.add_argument("--key", help=f"WattOps API key (default: ${sharing.KEY_ENV}, the saved key, else asked for without showing it)")
    cn.add_argument("--api-url", help=f"API address (default: ${sharing.URL_ENV} or {sharing.SITE})")
    cn.set_defaults(fn=cmd_connect)

    li = sub.add_parser("login", help="save your WattOps API key on this computer",
                        description="Checks the key with WattOps and saves it to ~/.config/landauer-gap/api-key "
                                    "(readable only by you), so --track and --share work in every terminal.")
    li.add_argument("key_arg", nargs="?", metavar="KEY", help="the key (lgk_…); asked for when left out")
    li.add_argument("--api-url", help=f"API address (default: ${sharing.URL_ENV} or {sharing.SITE})")
    li.set_defaults(fn=cmd_login)
    lo = sub.add_parser("logout", help="remove the saved API key from this computer")
    lo.set_defaults(fn=cmd_logout)

    dr = sub.add_parser("doctor", help="check this machine and print a report you can share",
                        description="Checks what joules can measure here (with a live reading from each meter), the GPU "
                                    "drivers, CPU energy counters, your API key, the connection to WattOps and Ollama. "
                                    "The report has no key, machine name, user name or home folder.")
    dr.add_argument("--gpus", help="only these GPU indices, e.g. 0,1")
    dr.add_argument("--seconds", type=float, default=1.0, help="how long to read each meter (default 1)")
    dr.add_argument("--offline", action="store_true", help="do not contact WattOps or Ollama")
    dr.add_argument("--json", action="store_true", help="print the report as JSON")
    dr.add_argument("--out", metavar="FILE", help="also save the report to FILE (.json for JSON)")
    dr.add_argument("--api-url", help=f"API address (default: ${sharing.URL_ENV} or {sharing.SITE})")
    dr.set_defaults(fn=cmd_doctor)

    ca = sub.add_parser("calibrate", help="compare the hardware meters with a wall-plug meter",
                        description="Read the meters for a fixed time, or while a command runs, and compare their total with a "
                                    "wall-plug meter's reading over the same time. Records what share of the machine's real draw the "
                                    "meters cover (docs/CALIBRATION.md).")
    ca.add_argument("--seconds", type=float, default=60.0, help="how long to read the meters when no command is given (default 60)")
    ca.add_argument("--wall-wh", type=float, metavar="WH", help="the wall meter's energy over the same time, in watt-hours")
    ca.add_argument("--wall-w", type=float, metavar="W", help="or the wall meter's average power over the same time, in watts")
    ca.add_argument("--wall-csv", metavar="FILE", help="or a logging meter's time-stamped readings (time, watts): joules integrates "
                                                       "them over exactly the seconds it measured")
    ca.add_argument("--report", metavar="JSON", help="with --wall-csv: compare an earlier calibrate --out JSON with a meter log "
                                                     "exported afterwards, instead of measuring again")
    common(ca)
    ca.add_argument("command", nargs=argparse.REMAINDER, help="optionally a command to run meanwhile, after --")
    ca.set_defaults(fn=cmd_calibrate)

    tu = sub.add_parser("tune", help="the GPU power limit that gives the cheapest token (needs sudo)",
                        description="Benchmark a local model at several GPU power limits and report which one uses the least energy "
                                    "per token without being more than --max-slowdown slower than the full limit. The limit is put back "
                                    "afterwards. Needs permission to set it: run with sudo.")
    tu.add_argument("target", help="ollama:<model> or openai:<model>")
    tu.add_argument("--url", help="server URL (default localhost Ollama or http://localhost:8000/v1)")
    tu.add_argument("--limits", default="100,85,70,55", help="power limits to try, as %% of the card's maximum (default 100,85,70,55)")
    tu.add_argument("--max-slowdown", type=float, default=10.0, metavar="PERCENT",
                    help="accept a lower limit only if it is at most this much slower than the full limit (default 10)")
    tu.add_argument("--runs", type=int, default=3, help="measured requests per limit (default 3)")
    tu.add_argument("--warmup", type=int, default=1, help="unmeasured requests first, so model loading is not counted")
    tu.add_argument("--max-tokens", type=int, default=256)
    tu.add_argument("--prompt", default=DEFAULT_PROMPT)
    common(tu)
    tu.set_defaults(fn=cmd_tune)

    te = sub.add_parser("telemetry", help="anonymous usage counts: on, off or status (off unless you said yes)",
                        description="Once a day, only if you said yes: the joules version, OS, GPU maker, how many commands ran "
                                    "and a random ID. Never commands, files, host names or results. Off in CI and with "
                                    "DO_NOT_TRACK=1 or JOULES_TELEMETRY=0.")
    te.add_argument("action", nargs="?", choices=("status", "on", "off"), default="status")
    te.set_defaults(fn=cmd_telemetry)

    dv = sub.add_parser("devices", help="list what can be measured here")
    dv.add_argument("--gpus"); dv.add_argument("--no-cpu", action="store_true"); dv.add_argument("--json", action="store_true")
    dv.set_defaults(fn=cmd_devices)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _plain_streams()
    if argv[:1] == ["--"] or (argv and argv[0].startswith("--") and argv[0] not in ("--help", "--version") and "--" in argv):
        argv = ["run", *argv]  # joules -- cmd, or joules --track p -- cmd
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help(sys.stderr)
        return 2
    if a.cmd in ("run", "ci", "calibrate") and a.command and a.command[0] != "--" and a.command[0].startswith("-"):
        p.error("put the command after --, e.g. joules run -- python train.py")
    a.argv = argv
    counted = a.cmd not in ("telemetry",)
    if counted:
        try:
            telemetry.record()
        except Exception:
            pass  # counting never gets in the way of a measurement
    rc = a.fn(a)
    if counted and rc == 0 and a.cmd not in ("agent", "proxy", "connect"):
        try:
            telemetry.maybe_ask()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
