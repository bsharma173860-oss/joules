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
import array
import codecs
import collections
import contextlib
import datetime as dt
import errno
import functools
import io
import json
import os
import platform
import re
import select
import selectors
import shlex
import shutil
import signal
import stat
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
from .phases import PhaseClock
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


def _term_size(dst) -> tuple[int, int]:
    """(columns, lines) of the terminal dst writes to (each stream's own: stdout may be a file while stderr is not)."""
    try:
        sz = os.get_terminal_size(dst.fileno())
        if sz.columns and sz.lines:
            return sz.columns, sz.lines
    except (OSError, ValueError, AttributeError):
        pass
    sz = shutil.get_terminal_size((80, 24))
    return sz.columns, sz.lines


def _pty_pair(dst=None) -> tuple[int, int]:
    """A pseudo-terminal for one of the command's streams. The command sees a terminal, so C programs flush every
    line (and keep colours and progress bars) as they would on screen; output processing is off, so the bytes reach
    joules unchanged (no \n -> \r\n)."""
    import pty
    import termios
    m, sl = pty.openpty()
    attrs = termios.tcgetattr(sl)
    attrs[1] &= ~termios.OPOST
    termios.tcsetattr(sl, termios.TCSANOW, attrs)
    try:
        import fcntl
        import struct
        cols, rows = _term_size(dst)
        fcntl.ioctl(sl, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except (OSError, ImportError):
        pass
    return m, sl


_SOCK_STREAM = 1


def _sock_type(fd: int):
    try:
        import socket
        s = socket.socket(fileno=os.dup(fd))
        try:
            return s.type
        finally:
            s.close()
    except (OSError, ValueError, ImportError):
        return None


def _message_socket(f) -> bool:
    """A datagram (or seqpacket) socket: its messages would not survive passing through joules, so the command gets
    it as it is (and joules does not read it)."""
    try:
        import socket
        fd = f.fileno()
        return os.name == "posix" and stat.S_ISSOCK(os.fstat(fd).st_mode) and _sock_type(fd) not in (None, socket.SOCK_STREAM)
    except (OSError, ValueError, AttributeError, ImportError):
        return False


def _same_place() -> bool:
    """Do our stdout and stderr go to the same terminal or file (as on screen, or with 2>&1)?"""
    try:
        a, b = os.fstat(sys.stdout.fileno()), os.fstat(sys.stderr.fileno())
        # Windows gives 0 for the identity of consoles, pipes and NUL: two of those cannot be told apart
        return os.path.samestat(a, b) and (os.name == "posix" or a.st_ino != 0)
    except (OSError, ValueError, AttributeError):
        return False


def _isatty(f) -> bool:
    try:
        return os.isatty(f.fileno())
    except (OSError, ValueError, AttributeError):
        return False


def _usable(f) -> bool:
    """Is this stream of ours open? (With `>&-` it is not: the command then inherits it closed, as without joules.)"""
    try:
        os.fstat(f.fileno())
        return True
    except (OSError, ValueError, AttributeError):
        return False


def _start(cmd: list[str], piped: bool = False):
    """piped (--from/--until): joules reads the command's output, passes it on and looks for the window's marks.
    Each stream is what it would be without joules: a terminal (a pseudo-terminal, POSIX) where ours is a terminal,
    a pipe where ours is a file or a pipe, ours as it is where ours is closed; and one shared stream where our
    stdout and stderr go to the same place, so their order is kept. Sets p.joules_streams = [(fd we read, where it
    goes, how to close it)]."""
    try:
        if piped:
            outs = [f for f in (sys.stdout, sys.stderr) if _usable(f) and not _message_socket(f)]
            untouched = [n for f, n in ((sys.stdout, "stdout"), (sys.stderr, "stderr")) if _usable(f) and f not in outs]
            same = len(outs) == 2 and _same_place()
            env = dict(os.environ)
            if os.name != "posix" or any(not _isatty(f) for f in outs):   # a pipe (always, on Windows): Python would hold its output
                env["PYTHONUNBUFFERED"] = "1"
            if os.name == "posix":
                made, ends = [], {}
                for f in outs:
                    if f is sys.stderr and same:
                        ends[f] = ends[sys.stdout]
                        continue
                    pair = _pty_pair(f) if _isatty(f) else os.pipe()
                    made.append(pair); ends[f] = pair
                try:
                    p = subprocess.Popen(cmd, stdout=ends[sys.stdout][1] if sys.stdout in ends else None,
                                         stderr=ends[sys.stderr][1] if sys.stderr in ends else None, env=env)
                except BaseException:
                    for r, _ in made:
                        os.close(r)
                    raise
                finally:
                    for _, w in made:
                        os.close(w)
                p.joules_streams = [(pair[0], f, functools.partial(os.close, pair[0]))
                                    for f, pair in ends.items() if not (f is sys.stderr and same)]
                p.joules_untouched = untouched
                return p
            out = subprocess.PIPE if sys.stdout in outs else None
            err = (subprocess.STDOUT if same else subprocess.PIPE) if sys.stderr in outs else None
            p = subprocess.Popen(cmd, stdout=out, stderr=err, bufsize=0, env=env)
            p.joules_streams = ([(p.stdout.fileno(), sys.stdout, p.stdout.close)] if p.stdout else []) + \
                               ([(p.stderr.fileno(), sys.stderr, p.stderr.close)] if p.stderr else [])
            return p
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
    g.add_argument("--interval", type=_interval_arg, default=0.1, help="sampling interval in seconds (default 0.1)")
    g.add_argument("--gpus", type=_gpus_arg, help="only these GPU indices, e.g. 0,1 (default: CUDA_VISIBLE_DEVICES or all)")
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


def _gpus_arg(s: str) -> str:
    """--gpus: comma-separated GPU indices, checked when the command line is read rather than after setup."""
    if not s.strip():
        return s  # empty: all GPUs, as before
    toks = [t.strip() for t in s.split(",")]
    if not all(t.isascii() and t.isdigit() for t in toks):
        raise argparse.ArgumentTypeError(f"expected GPU indices separated by commas, e.g. 0,1 (got {s!r})")
    return ",".join(toks)


def _regex_arg(s: str) -> str:
    try:
        re.compile(s)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"not a valid regular expression ({e}): {s!r}") from None
    return s


def _interval_arg(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of seconds (got {s!r})") from None
    if not v > 0 or v == float("inf"):
        raise argparse.ArgumentTypeError(f"must be more than 0 seconds (got {s})")
    return v


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
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(text, file=sys.stderr)
    if a.out:  # after printing, so a bad path never loses the result of a long run
        try:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except OSError as e:
            print(warn(f"joules: could not write {a.out} ({e.strerror or e}); the result is printed above."), file=sys.stderr)
            a.out_failed = True


def _out_problem(path: str) -> str | None:
    """Why --out cannot be written, found before a long run rather than after it."""
    folder = os.path.dirname(os.path.abspath(path))
    if os.path.isdir(path):
        return f"{path} is a folder; give a file name"
    if not os.path.isdir(folder):
        return f"the folder {folder} does not exist"
    return None


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
        with Sampler(d.meters, a.interval, notes=d.notes) as s:
            windowed = a.window_from is not None or a.window_until is not None
            # (from before the command starts: a signal at any moment after, before our reader exists, is held)
            term = _TermGuard() if windowed and os.name == "posix" else None
            if term:
                term.install()
            try:
                proc = _start(cmd, piped=windowed)
            except BaseException:
                if term:
                    term.restore()
                raise
            if not proc:
                if term:
                    term.restore()
                return 126 if proc is False else _not_found(cmd)
            for m in d.meters:
                if hasattr(m, "attach"):
                    m.attach(proc)
            if windowed:
                clock = PhaseClock(d.meters, s.lock)
                win = Window(clock, s.t0, d.meters, a.window_from, a.window_until)
                win.ttys = {fd: _isatty(dst) for fd, dst, _ in proc.joules_streams}
                win.untouched = getattr(proc, "joules_untouched", [])
                outs = {fd: _Out(dst) for fd, dst, _ in proc.joules_streams}   # written on by their own threads
                pump_state = {"stop": threading.Event(), "exited": threading.Event(), "caught_up": threading.Event(),
                              "fds": [fd for fd, _, _ in proc.joules_streams], "owed": {}, "blocked": {},
                              "reads": {}, "wlock": threading.Lock(), "wake": os.pipe() if os.name == "posix" else None}
                win.stop = pump_state["stop"]   # (a search stops between lines when joules hands the output over)
                if os.name == "posix":
                    pumps = [threading.Thread(target=_pump_all, args=(proc.joules_streams, win, pump_state, outs), daemon=True)]
                    old_winch = _follow_resizes(proc)
                else:
                    pumps = [threading.Thread(target=_pump, args=(fd, dst, win, pump_state, close, outs[fd]), daemon=True)
                             for fd, dst, close in proc.joules_streams]
                for t in pumps:
                    t.start()
            terminated = False
            while True:
                try:
                    with (term.raising() if term else contextlib.nullcontext()):
                        code = proc.wait()
                    break
                except KeyboardInterrupt:
                    continue  # Ctrl-C reaches the child too; wait for it to finish, then report
                except _Terminated:   # joules alone was told to stop: see _end_on_term
                    terminated = True
                    break
            if windowed and not terminated:
                while True:
                    try:
                        win.freeze(clock.snapshot())   # the run ends here; marks read from now on are its end
                        break
                    except KeyboardInterrupt:
                        continue
                if os.name != "posix":   # (recorded before the exit is flagged: readers look at it then)
                    with pump_state["wlock"]:
                        pump_state["reads_at_exit"] = dict(pump_state["reads"])
                        pump_state["exit_t"] = time.monotonic()
                pump_state["exited"].set()
                if pump_state.get("wake"):
                    with contextlib.suppress(OSError):
                        os.write(pump_state["wake"][1], b"x")
                if os.name != "posix":   # (POSIX: the reader notes it)
                    win.behind = any(o.full() for o in outs.values())
                    # (each reader looks at its own pipe, see _pump; a reader still in its read after 50 ms got nothing)
                    threading.Timer(0.06, _win_caught_up, (pump_state, win)).start()
    finally:
        d.close()
    if windowed:   # measuring stopped when the command exited, as without a window; now the rest of its output
        try:
            if terminated:
                raise _Terminated
            waited = _drain_after_exit(proc, pumps, pump_state, outs, d.notes, term)
            win.left_open = pump_state.get("left_open", False)
            if waited:
                for o in outs.values():
                    o.close()
                while any(o.thread.is_alive() for o in outs.values()):   # all output passed on before the report
                    try:
                        with (term.raising() if term else contextlib.nullcontext()):
                            for o in outs.values():
                                o.thread.join()
                    except KeyboardInterrupt:
                        if os.name == "posix":   # stop waiting for a slow reader (Windows: no helper, keep going)
                            waited = False
                            break
            if not waited:   # one Ctrl-C: hand everything over and report now
                held = _hand_over(proc, pumps, pump_state, outs)
                win.left_open = pump_state.get("left_open", False)
                if held:
                    d.notes.append("The command's output was still open when it exited (a process it started keeps "
                                   "it open).")
                d.notes.append("joules stopped waiting for the command's output (Ctrl-C); a small helper passes on "
                               "the rest of it after joules finishes.")
        except _Terminated:
            _end_on_term(proc, pumps, pump_state, outs, term.sig if term else None)
        if not any(t.is_alive() for t in pumps):
            for f in pump_state.get("wake") or ():
                with contextlib.suppress(OSError):
                    os.close(f)
        if term:
            term.restore(ctrl_c=False)
        if os.name == "posix":
            _stop_following(old_winch)
        if term and term.hit:   # a SIGTERM (SIGHUP) that came while nothing was left to hand over
            os.kill(os.getpid(), term.sig or signal.SIGTERM)
    devices = _devices(d, idle, s.duration_s)
    ops = a.flops or (training_flops(a.params, a.tokens) if a.params and a.tokens else None)
    report = {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "run", "command": cmd, "exit_code": code,
              "started_at": started, "duration_s": s.duration_s, "simulated": d.simulated, "host": _host(),
              "devices": devices, "totals": _totals(a, devices, ops), "notes": d.notes}
    if windowed:
        extra, wnotes = _window_report(a, win, s, d.meters, idle)
        report.update(extra)
        report["notes"] = [*wnotes, *report["notes"]]

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
    if windowed and term:
        term.restore()
    if a.track:
        _track(a, report)  # a tracking problem is reported but never changes the command's exit code
    _website_hint(a, cmd)
    return code


_FOLD = {0x131: "i", 0x17F: "s", 0x307: None}   # ı ſ (and İ's dot): what (?i) matches to i/s but lower() keeps


def _escape_start(b: bytes) -> bool:
    """Is b nothing, or only escape codes (the last perhaps unfinished)?"""
    rest = b[_ANSI_B.match(b).end():]
    return not rest or (rest[:1] == b"\x1b" and len(rest) < 64 and b"\n" not in rest and b"\r" not in rest)


def _word_runs(hay: str, words, runs) -> list:
    """The runs of whole lines of hay that have one of words, as (start, end) offsets in order. A word on many
    lines: one regex over the block (runs). A rarer one: from each place it is found (cost by occurrences)."""
    found = sum(hay.count(w) for w in words)
    if found * 4 > hay.count("\n") + 1:
        return [m.span() for m in runs.finditer(hay)]
    lines = {}
    for w in words:
        p = hay.find(w)
        while p >= 0:
            a = hay.rfind("\n", 0, p) + 1
            b = hay.find("\n", p)
            b = len(hay) if b < 0 else b + 1
            lines[a] = b
            p = hay.find(w, b)
    spans = []
    for a in sorted(lines):
        if spans and spans[-1][1] >= a:
            spans[-1][1] = max(spans[-1][1], lines[a])
        else:
            spans.append([a, lines[a]])
    return spans


def _runs_with(words):
    """A pattern for runs of whole lines that each have one of words (as plain text): one match per run, in C."""
    alt = "|".join(re.escape(w) for w in words)
    return re.compile(r"(?:^[^\n]*(?:%s)[^\n]*(?:\n|\Z))+" % alt, re.M)


def _fold(text: str) -> str:
    """Lower case as (?i) compares ASCII words: ı, ſ and İ count as i, s and i (only ever lets more lines through)."""
    return text.lower().translate(_FOLD)


def _line_bound(rx) -> bool:
    """Can no match of rx take in a line end, or look across one? Then searching many lines at once cannot run
    across lines (no blow-up between lines) and finds a line only where a line-by-line search would. Read from the
    parsed pattern; anything unsure says no."""
    try:
        try:
            from re import _parser as sp, _constants as sc
        except ImportError:
            import sre_parse as sp
            import sre_constants as sc
        tree = sp.parse(rx.pattern, rx.flags)
    except Exception:
        return False
    safe_cats = {getattr(sc, n) for n in ("CATEGORY_DIGIT", "CATEGORY_WORD", "CATEGORY_NOT_SPACE",
                                          "CATEGORY_NOT_LINEBREAK", "CATEGORY_UNI_DIGIT", "CATEGORY_UNI_WORD",
                                          "CATEGORY_UNI_NOT_SPACE", "CATEGORY_UNI_NOT_LINEBREAK", "CATEGORY_LOC_WORD")
                 if hasattr(sc, n)}
    reps = {sc.MAX_REPEAT, sc.MIN_REPEAT, getattr(sc, "POSSESSIVE_REPEAT", None)} - {None}
    atomic = getattr(sc, "ATOMIC_GROUP", None)
    nl = ord("\n")

    def ok(items, dotall) -> bool:
        for op, av in items:
            if op is sc.LITERAL:
                if av == nl:
                    return False
            elif op is sc.ANY:
                if dotall:
                    return False
            elif op is sc.IN:
                for iop, iav in av:
                    if iop is sc.NEGATE or (iop is sc.LITERAL and iav == nl) or \
                            (iop is sc.RANGE and iav[0] <= nl <= iav[1]) or (iop is sc.CATEGORY and iav not in safe_cats):
                        return False
                    if iop not in (sc.LITERAL, sc.RANGE, sc.CATEGORY):
                        return False
            elif op is sc.AT:
                if av in (sc.AT_BEGINNING_STRING, sc.AT_END_STRING):
                    return False
            elif op is sc.SUBPATTERN:
                add, dele = av[1] or 0, av[2] or 0
                if (add | dele) & (re.MULTILINE | re.DOTALL) or not ok(av[-1], dotall):
                    return False
            elif op in reps:
                if not ok(av[2], dotall):
                    return False
            elif op is sc.BRANCH:
                if not all(ok(b, dotall) for b in av[1]):
                    return False
            elif atomic is not None and op is atomic:
                if not ok(av, dotall):
                    return False
            elif op is sc.GROUPREF:
                pass   # (the group it repeats is checked where it is)
            elif op is sc.GROUPREF_EXISTS:
                if not ok(av[1], dotall) or (av[2] is not None and not ok(av[2], dotall)):
                    return False
            else:   # (lookarounds, and anything not known here)
                return False
        return True
    return ok(list(tree), bool(rx.flags & re.DOTALL))


def _must_contain(rx):
    """(words, fold): one of `words` is in every string rx matches (`fold`: compare in lower case), or None if no
    such plain text can be told. Read from the parsed pattern: literal runs that every match passes through."""
    try:
        try:
            from re import _parser as sp, _constants as sc   # (Python 3.11+)
        except ImportError:
            import sre_parse as sp
            import sre_constants as sc
        tree = sp.parse(rx.pattern, rx.flags)
    except Exception:   # (an unusual pattern: no shortcut, every line is searched)
        return None
    reps = {sc.MAX_REPEAT, sc.MIN_REPEAT, getattr(sc, "POSSESSIVE_REPEAT", None)} - {None}
    atomic = getattr(sc, "ATOMIC_GROUP", None)
    fold = bool(rx.flags & re.IGNORECASE)

    def best(items) -> str:   # the longest literal run every match of this sequence contains
        top, cur = "", ""
        for op, av in items:
            if op is sc.LITERAL:
                cur += chr(av)
                continue
            top, cur = max(top, cur, key=len), ""
            sub = ""
            if op is sc.SUBPATTERN:
                if av[1] or av[2]:   # (scoped flags: case may change inside; no shortcut there)
                    continue
                sub = best(av[-1])
            elif op in reps and av[0] >= 1:
                sub = best(av[2])
            elif atomic is not None and op is atomic:
                sub = best(av)
            top = max(top, sub, key=len)
        return max(top, cur, key=len)
    items = list(tree)
    if len(items) == 1 and items[0][0] is sc.BRANCH:   # a|b|c: one of the branches' words
        words = [best(b) for b in items[0][1][1]]
        if not all(words):
            return None
    else:
        w = best(items)
        if not w:
            return None
        words = [w]
    if fold:
        if not all(w.isascii() for w in words):   # (case folding beyond ASCII: no shortcut)
            return None
        words = [_fold(w) for w in words]
    if any("\n" in w for w in words):   # (cannot be on one line: let the search say so)
        return None
    return words, fold


def _cpu_step() -> float:
    """How finely this thread's CPU clock moves (Linux: well under a microsecond; Windows: 15.6 ms)."""
    t0 = time.thread_time()
    deadline = time.perf_counter() + 0.05
    while time.perf_counter() < deadline:
        t1 = time.thread_time()
        if t1 != t0:
            return t1 - t0
    return 0.016


_STEP = 0.0
_GROUP = None   # matching time is judged over groups of about this much CPU or budget (measured when first needed)


def _searchable(pattern: str) -> str:
    """The pattern to search with. Leading repeats of any character (.* .+ and their lazy forms, after an optional
    ^) change nothing in a search but make it slow on long lines (each start tried to the end): .* is dropped and
    each .+ becomes . (one character at least), e.g. ^.+.*START -> .START."""
    i, need = (1 if pattern.startswith("^") else 0), 0
    took = False
    while True:
        m = re.match(r"\.([*+])\??", pattern[i:])
        if not m:
            break
        need += m.group(1) == "+"
        i += m.end()
        took = True
    rest = pattern[i:]
    if not took or not rest or rest[0] in "*+?{":
        return pattern
    return "." * need + rest


_ANSI_B = re.compile(rb"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()*+].|\x1b[@-OQ-WYZ\\])*")
# (two-byte escapes: not ] P X ^ _, which start longer strings: an unfinished one is not taken for a whole one)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()*+].|\x1b[@-Z\\-_]")  # colours, moves, titles, charsets


class Window:
    """--from / --until: the stretch of a run between the first output line matching --from and the first later line
    matching --until (the whole run before or after it when one of them is not given). Each mark reads every meter
    at the moment the line arrives, so the window's energy works the same for energy counters and power sampling."""

    def __init__(self, clock: PhaseClock, start_t: float, meters, start_re: str | None, end_re: str | None):
        self.clock, self.meters = clock, meters
        self.start_re = re.compile(_searchable(start_re)) if start_re is not None else None
        self.end_re = re.compile(_searchable(end_re)) if end_re is not None else None
        self.match_s = 0.0    # time spent matching beyond a plain search (it holds up the rest of joules)
        self._grp = {}   # per pattern: [CPU, budget] of the group being added up
        self.match_by = {}    # the same, per pattern
        global _GROUP, _STEP
        if _GROUP is None:
            _STEP = _cpu_step()
            _GROUP = max(0.001, 10 * _STEP)
        self.group = _GROUP
        self.slow_piece = max(0.005, 2 * _STEP)   # a block piece slower than this: that pattern goes line by line
        self.lock = threading.Lock()
        self.begin = None if self.start_re else {"t": start_t, "j": {m.name: 0.0 for m in meters}, "line": None}
        self.end = None
        self.frozen = None   # set when the command has exited: a mark read after that is the end of the run
        self.first_out = {}   # when the command's first output was read, per stream
        self.ttys = {}        # which streams are terminals
        self.behind = False   # was output still waiting for a slow reader on our side when the command exited?
        self.paused = False   # did joules stop reading for a slow reader while a mark was still to come?
        self.alt_end = None   # an --until line that arrived with the --from line on another stream (order unknown)
        # the plain text every match must contain, looked for first with a plain (linear) substring search; the
        # pattern itself only ever runs on single lines that contain it (all lines, for a pattern without any)
        self.must = {rx: _must_contain(rx) for rx in (self.start_re, self.end_re) if rx}
        self.block = {rx: (re.compile(rx.pattern, rx.flags | re.M) if _line_bound(rx) else None)
                      for rx in (self.start_re, self.end_re) if rx}
        self.block_size = {}   # how much of a block is searched at once, per pattern (see _lines)
        self.runs = {rx: _runs_with(m[0]) for rx, m in self.must.items() if m}   # (lines that have the word)

    def _mark(self, text: str, src=None, eof: bool = False) -> dict:
        snap, late = self.frozen, True
        if snap is None:
            snap, late = self.clock.snapshot(), False
            exited = self.frozen
            if exited is not None and snap["t"] >= exited["t"]:   # the command exited meanwhile
                snap, late = exited, True
        return {"t": snap["t"], "j": snap["j"], "line": text.strip()[:200], "late": late, "src": src,
                "after_pause": self.paused, "after_exit": late and getattr(self, "caught_up", False) and not eof,
                "no_line_end": eof}

    @staticmethod
    def clean(text: str) -> str:
        if "\x1b" not in text and "\r" not in text:
            return text
        return ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")   # colour codes off; \r\n is one line end

    @classmethod
    def _cleaned(cls, text: str) -> str:
        out = cls.clean(text)
        if hasattr(text, "start"):   # (a cut line: where matches may start, already in cleaned terms)
            out = _Text(out)
            out.start = text.start
        return out

    def _stopping(self) -> bool:
        """Handing over (Ctrl-C, SIGTERM, a stream still open): stop searching between lines."""
        stop = getattr(self, "stop", None)
        return stop is not None and stop.is_set()

    def _hit(self, rx, line: str, partial: bool, pos: int = 0, keep_from=None) -> bool:
        """Does rx match this line, starting at pos (before it: only context, where the line was cut)? partial: the
        line is not finished yet, so a match that reaches the end of what has come so far (where $ would match, or a
        match could still grow) does not count yet, unless it starts before keep_from (that part is not kept)."""
        if not partial:
            return rx.search(line, pos) is not None
        return any(m.end() < len(line) or (keep_from is not None and m.start() < keep_from)
                   for m in rx.finditer(line, pos))

    def pending(self) -> bool:
        """Is a mark still to be found? Once not, output is passed on without looking at it."""
        return self.begin is None or (self.end is None and self.end_re is not None)

    def freeze(self, snap: dict) -> None:
        """The meters as the command exits (read while they are still running). Not under the lock: a search in
        progress must not delay it (a mark it then sets after the exit is the end of the run, see _mark)."""
        self.frozen = {"t": snap["t"], "j": dict(snap["j"])}

    def _lines(self, text: str, rx) -> list[str] | None:
        """The lines of a block (already cleaned, see _feed) to search with rx: None if no line can match; lines
        that cannot match are None in the list (keeping the numbering). First the text every match must contain
        (plain substring searches); the pattern is only ever tried on lines that have it (on every line, if it has
        none). For a pattern that cannot match a line end, the runs of such lines are searched at once (it cannot
        run across lines, so it costs no more than line by line, but in C), else line by line."""
        must = getattr(self, "must", {}).get(rx)
        spans = [(0, len(text))]
        if must:
            words, fold = must
            hay = _fold(text) if fold else text
            if not any(w in hay for w in words):
                return None
            # (where the lines with a word are: in the text itself, unless folding may have moved them)
            if not fold or text.isascii():
                spans = _word_runs(hay, words, getattr(self, "runs", {}).get(rx) or _runs_with(words))
            else:
                spans = None
        block = getattr(self, "block", {}).get(rx) if spans is not None and text else None
        done = 0   # (lines before this offset: searched with nothing, or without the word)
        if block is not None:
            for a, b in spans:   # in pieces of whole lines, small at first and doubling while fast (up to 64 KB):
                i = a            # a search holds up every other thread while it runs
                while i < b:
                    size = self.block_size.get(rx, 256)
                    j = text.find("\n", i + size, b)
                    j = b if j < 0 else j + 1
                    t0 = time.thread_time()
                    m = block.search(text, i, j)
                    if time.thread_time() - t0 > self.slow_piece:   # slow in itself: line by line from now on
                        self.block[rx] = None
                        done = m.start() if m is not None else i
                        break
                    self.block_size[rx] = min(65536, size * 2)
                    if m is not None:
                        lines = text.split("\n")
                        k = text.count("\n", 0, m.start())   # (lines before the first that can match)
                        lines[:k] = [None] * k
                        return lines
                    i = j
                else:
                    continue
                break
            else:
                return None   # (no line with the word can match)
        lines = text.split("\n")
        if must:
            lines = [ln if any(w in (_fold(ln) if fold else ln) for w in words) else None for ln in lines]
        k = text.count("\n", 0, done)
        lines[:k] = [None] * k
        return lines

    def feed(self, texts: list[str], srcs: list | None = None, partial: bool = False, keep_from=None,
             eof: bool = False) -> bool:
        sec = {}   # this call's own CPU time per pattern searched (a busy machine does not count; Windows feeds
        try:       # from two threads at once, so it is not kept on the window)
            return self._feed(texts, srcs, partial, keep_from, eof, sec)
        finally:
            # only what goes beyond a plain search of this much text counts (a plain word never adds up, however
            # much output; a slow pattern does, in big reads or small, e.g. 4 KB from a terminal)
            # Added up in small groups (closed once either reaches _GROUP): the CPU clock's steps (15.6 ms on Windows)
            # even out within a group, and unused budget is not carried over to hide a slow search later.
            n = sum(len(t) for t in texts)
            with self.lock:
                for key, cpu in sec.items():   # each pattern searched, with its own budget
                    g = self._grp.setdefault(key, [0.0, 0.0])
                    g[0] += cpu
                    g[1] += 0.0002 + n / 50e6
                    if g[0] >= self.group or g[1] >= self.group:
                        extra = max(0.0, g[0] - g[1])
                        self.match_s += extra
                        self.match_by[key] = self.match_by.get(key, 0.0) + extra
                        g[0] = g[1] = 0.0

    def _feed(self, texts: list[str], srcs: list | None = None, partial: bool = False, keep_from=None,
              eof: bool = False, sec: dict | None = None) -> bool:
        """Whole lines that arrived together, one block per stream. The --from line is looked for in all of them
        first, then the --until line after it on its own stream. Which came first of lines that arrived together
        on two streams is not known: an --until line on the other stream is kept aside (alt_end), used only if no
        later --until line appears. Returns True if a mark was set."""
        set_one = False
        srcs = srcs or [None] * len(texts)
        texts = [self._cleaned(t) for t in texts]   # once, outside the timed searches (it is not matching)
        with self.lock:
            where = None   # (block, line) of a --from line found now
            if self.begin is None:
                t0 = time.thread_time()
                for bi, text in enumerate(texts):
                    lines = self._lines(text, self.start_re)
                    for li, ln in enumerate(lines or ()):
                        if ln is None:
                            continue
                        if not eof and self._stopping():   # (the last unfinished line is still looked at)
                            self.cut_short = True
                            break
                        if self._hit(self.start_re, ln, partial, getattr(text, "start", 0) if li == 0 else 0, keep_from):
                            self.begin, where, set_one = self._mark(ln, srcs[bi], eof), (bi, li), True
                            break
                    if where:
                        break
                if sec is not None:
                    sec["--from"] = time.thread_time() - t0
                if self.begin is None:
                    return False
            if self.end is None and self.end_re is not None:
                t0 = time.thread_time()
                try:
                    self._search_end(texts, srcs, where, partial, keep_from, eof)
                finally:
                    if sec is not None:
                        sec["--until"] = time.thread_time() - t0
                set_one = set_one or self.end is not None
        return set_one

    def _search_end(self, texts, srcs, where, partial, keep_from, eof) -> None:
        """The --until line in what follows a --from line found now (on its stream), or in all of it."""
        for bi, text in enumerate(texts):
            lines = self._lines(text, self.end_re)
            if not lines:
                continue
            first = (where[1] + 1) if where and where[0] == bi else 0   # only lines after the --from line
            for li in range(first, len(lines)):
                ln = lines[li]
                if ln is None:
                    continue
                if not eof and self._stopping():
                    self.cut_short = True
                    return
                if self._hit(self.end_re, ln, partial, getattr(text, "start", 0) if li == 0 else 0, keep_from):
                    if where and where[0] != bi:   # arrived with the --from line, on another stream
                        if self.alt_end is None:
                            self.alt_end = self._mark(ln, srcs[bi], eof)
                        break
                    self.end = self._mark(ln, srcs[bi], eof)
                    break
            if self.end is not None:
                break


class _Text(str):
    """A block of lines whose first line was cut: matches in it start at .start (what is before is only context)."""
    start = 0


class _Lines:
    """The unfinished line of one stream, between reads. A line longer than 1 MB is searched (as unfinished) before
    only its end is kept, with a little of what came before as context (so \\b, lookbehinds and ^ see the real
    text, but no match starts there); once a mark is found in it, the rest of that line is skipped (the next mark
    must be on a later line)."""
    KEEP, CONTEXT = 65536, 256

    def __init__(self):
        self.buf, self.skip = b"", False
        self.cr = False   # the last line ended with \r at the very end of a read: a \n next belongs to it
        self.ctx = 0      # bytes at the start of buf that are only context (the line was cut after them)

    def _text(self, data: bytes, ctx: int) -> str:
        text = data.decode("utf-8", "replace")
        if not ctx:
            return text
        t = _Text(text)
        # (where the kept part starts, in cleaned terms; a cut inside an escape code or a character can shift it by a
        # few characters, within the long-line limits)
        t.start = len(Window.clean(data[:ctx].decode("utf-8", "replace")))
        return t

    def add(self, chunk: bytes, win: "Window", src=None) -> str | None:
        data, ctx = self.buf + chunk, self.ctx
        if self.cr:   # \r\n split between two reads is one line end (escape codes between them are dropped, as
            m = _ANSI_B.match(data)   # within one read)
            rest = data[m.end():]
            if rest[:1] == b"\n":
                data = rest[1:]
            elif _escape_start(rest):   # (only escape codes so far: the \n may still come)
                self.buf = data
                return None
        self.cr = False
        if self.skip:
            k = min((x for x in (data.find(b"\n"), data.find(b"\r")) if x >= 0), default=-1)
            if k < 0:
                self.buf, self.ctx = b"", 0
                return None
            if data[k:k + 1] == b"\r":   # (\r\n, also with escape codes between them, is one line end)
                e = _ANSI_B.match(data, k + 1).end()
                if data[e:e + 1] == b"\n":
                    k = e
            self.cr = data[k:k + 1] == b"\r" and _escape_start(data[k + 1:])
            data, self.skip, ctx = data[k + 1:], False, 0
        i = max(data.rfind(b"\n"), data.rfind(b"\r"))
        done = None
        if i >= 0:
            j = i
            if data[i:i + 1] == b"\n":   # \r\n is one line end, also with escape codes between them
                r = data.rfind(b"\r", 0, i)
                if r >= 0 and _ANSI_B.match(data, r + 1).end() == i:
                    j = r
            self.cr = data[i:i + 1] == b"\r" and _escape_start(data[i + 1:])   # (at the end, or only escapes after)
            done, data, ctx = self._text(data[:j], ctx), data[i + 1:], 0
        if len(data) > 1 << 20:
            if done is not None:
                win.feed([done], [src]); done = None
            text = self._text(data, ctx)
            keep_from = len(Window.clean(text)) - len(Window.clean(data[-self.KEEP:].decode("utf-8", "replace")))
            if win.feed([text], [src], partial=True, keep_from=keep_from):
                self.skip, data, ctx = True, b"", 0
            else:   # keep the end, after a little context
                data, ctx = data[-(self.KEEP + self.CONTEXT):], self.CONTEXT
        self.buf, self.ctx = data, ctx
        return done

    def end(self, win: "Window", src=None) -> None:
        # (a last line with no line end: it counts only now, when the output ends; escape codes alone are no line)
        if self.buf and not self.skip and win.pending() and _ANSI_B.match(self.buf).end() < len(self.buf):
            win.feed([self._text(self.buf, self.ctx)], [src], eof=True)
        self.buf, self.ctx = b"", 0


def _write(dst, chunk: bytes) -> bool:
    """Pass a chunk on unchanged. False if our own output is gone (closed, or a pipe whose reader left)."""
    try:
        if os.name == "nt" and _isatty(dst):   # a console: the bytes as the command wrote them (its code page), not UTF-8
            dst.flush()
            fd, view = dst.fileno(), memoryview(chunk)
            while view:
                view = view[os.write(fd, view):]
            return True
        out = getattr(dst, "buffer", None)
        if out is not None:
            out.write(chunk); out.flush()
        else:
            dst.write(chunk.decode("utf-8", "replace")); dst.flush()
        return True
    except (BrokenPipeError, ValueError, AttributeError):
        return False
    except OSError as e:   # a full disk and the like: this chunk is lost (as the command's own write would fail)
        return not (e.errno == errno.EBADF or (os.name == "nt" and e.errno == errno.EINVAL))


class _Out:
    """Where one of the command's streams goes, written by a thread of its own, so a slow reader on our side (a
    pager, a slow pipe, Ctrl-S) does not hold up reading the command's output: marks keep the time each line was
    printed. Up to LIMIT bytes wait here; past that, joules stops reading until there is room."""
    LIMIT = 16 << 20

    def __init__(self, dst):
        self.dst, self.q, self.size = dst, collections.deque(), 0
        self.gone = self.closed = False   # gone: our own output went away (e.g. `| head`)
        self.writing = self.handed = False   # handed: what is left goes to a helper (take)
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, chunk: bytes) -> None:
        with self.cv:
            if not self.gone:
                self.q.append(chunk)
                self.size += len(chunk)
                self.cv.notify_all()

    def full(self) -> bool:
        return self.size >= self.LIMIT

    def wait_empty(self, timeout=None) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.size == 0 or self.gone, timeout)

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def take(self, ending: bool = True) -> tuple[bytes, bytes]:
        """Stop writing and return (maybe, rest): what is still waiting, with the unwritten rest of the chunk under
        way. Its writer may be stuck (a paused pager), so joules hands that over rather than wait for it. A piece that
        may have to wait is at most PIPE_BUF bytes, which a pipe or terminal takes all at once or not at all (see
        _piece). If joules ends now (ending), a stuck piece dies with it and is in rest; otherwise it may still be
        written, so it is returned apart (maybe), and wrote_stuck() says at the end whether it was."""
        with self.cv:
            self.handed = True
            self.cv.notify_all()
            self.cv.wait_for(lambda: not self.writing, 0.1)   # (a write to a reader that reads ends at once)
            if self.gone:
                maybe, rest = b"", b""
            else:
                k = self.off + (self.inflight if self.writing and not ending else 0)
                maybe, rest = self.cur[self.off:k], self.cur[k:] + b"".join(self.q)
            self.taken_off = self.off
            self.q.clear()
            self.size, self.cur = 0, b""
            return maybe, rest

    def wrote_stuck(self) -> bool:
        with self.cv:
            return self.off > self.taken_off

    def _fd(self):
        try:
            fd = self.dst.fileno()
            self.dst.flush()   # (anything joules itself wrote first)
            if not isinstance(fd, int):
                return None
            mode = os.fstat(fd).st_mode
            st = os.fstat(fd)
            if stat.S_ISFIFO(mode):
                self.kind = "pipe"
            elif stat.S_ISSOCK(mode) and os.name == "posix" and _sock_type(fd) == _SOCK_STREAM:
                import socket
                self.kind, self.sock = "socket", socket.socket(fileno=os.dup(fd))
            elif os.isatty(fd) or (stat.S_ISCHR(mode) and st.st_rdev != os.stat(os.devnull).st_rdev):
                self.kind = "tty"    # a terminal, or a device that may stop part-way (written 4 KB at a time)
            else:
                self.kind = "file"   # (and /dev/null)
            return fd
        except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
            return None

    def _piece(self, fd) -> int:
        """How much to write at once: 64 KB into an empty pipe (Linux), else 4 KB (PIPE_BUF) where a write may wait
        (a pipe, a terminal: Ctrl-S), and 64 KB where it does not (a file). (A terminal write stopped by Ctrl-S part
        way is not atomic: there a hand-over (SIGTERM, Ctrl-C) may repeat up to 4 KB; sockets are sent exactly.)"""
        if self.kind == "pipe":
            try:
                import fcntl
                import termios
                n = array.array("i", [0])
                fcntl.ioctl(fd, termios.FIONREAD, n, True)
                # an empty pipe takes 64 KB at once; otherwise its room is in part-filled pages, so 4 KB (atomic)
                return 65536 if n[0] == 0 and fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ) >= 65536 else 4096
            except (ImportError, OSError, AttributeError, ValueError):
                return 4096
        return 4096 if self.kind == "tty" else 65536

    def _send(self, chunk: bytes) -> bool:
        """A socket: wait until it takes data, then send what it takes at once (never blocking part-way), under the
        lock, so how far it got is always exact (a hand-over then repeats nothing)."""
        import socket
        while self.off < len(chunk):
            self.write_t0 = self.write_t0 or time.monotonic()
            try:   # (poll: no limit on fd numbers)
                if self.poller is None:
                    self.poller = select.poll()
                    self.poller.register(self.sock, select.POLLOUT)
                ready = self.poller.poll(100)
            except (OSError, ValueError):
                return False
            with self.cv:
                if self.handed:
                    return True
                if not ready:
                    continue
                try:
                    n = self.sock.send(chunk[self.off:self.off + 65536], socket.MSG_DONTWAIT)
                except (BlockingIOError, InterruptedError):
                    continue
                except (BrokenPipeError, ConnectionResetError):
                    return False
                except OSError as e:
                    if e.errno == errno.EBADF:
                        return False
                    n = len(chunk) - self.off   # this chunk is lost, as the command's own would be
                self.off += n
            t1 = time.monotonic()
            if t1 - self.write_t0 > 0.05:
                self.last_block = t1
            self.write_t0 = None
        return True

    def _put(self, fd, chunk: bytes) -> bool:
        """Write one chunk in pieces, keeping how far it got. False if our own output is gone."""
        if self.kind == "socket":
            return self._send(chunk)
        while self.off < len(chunk):
            piece = chunk[self.off:self.off + self._piece(fd)]
            with self.cv:
                self.inflight = len(piece)
            self.write_t0 = time.monotonic()
            try:
                n = os.write(fd, piece)
            except BlockingIOError:   # (someone made our terminal non-blocking)
                time.sleep(0.01)
                continue
            except BrokenPipeError:
                return False
            except OSError as e:
                if e.errno == errno.EBADF or (os.name == "nt" and e.errno == errno.EINVAL):   # (Windows: reader gone)
                    return False
                n = len(chunk) - self.off   # a full disk and the like: this chunk is lost, as the command's own would be
            t1 = time.monotonic()
            if t1 - self.write_t0 > 0.05:   # what reads our output made us wait
                self.last_block = t1
            self.write_t0 = None
            with self.cv:
                self.off += n
                if self.handed:
                    return True
        return True

    cur, off, kind, inflight, taken_off, sock, poller = b"", 0, "file", 0, 0, None, None
    write_t0 = None   # when the write under way started
    last_block = 0.0  # when a write last had to wait for the reader

    def slow_reader(self) -> bool:
        """Is what reads our output holding us up (a write waiting now, or one did lately)? Else a full queue is
        only joules writing more slowly than a fast command prints, which does not move the marks."""
        t0, now = self.write_t0, time.monotonic()
        return (t0 is not None and now - t0 > 0.05) or now - self.last_block < 1.0

    def _run(self) -> None:
        fd = self._fd()
        while True:
            with self.cv:
                while not self.q and not self.closed and not self.handed:
                    self.cv.wait()
                if self.handed or not self.q:
                    return
                chunk = self.q.popleft()
                self.cur, self.off, self.writing = chunk, 0, True
            ok = self._put(fd, chunk) if fd is not None else _write(self.dst, chunk)
            with self.cv:
                self.writing = False
                self.cv.notify_all()
                if self.handed:
                    return
                self.cur, self.off = b"", 0
                self.size -= len(chunk)
                if not ok:
                    self.gone, self.size = True, 0
                    self.q.clear()
                self.cv.notify_all()


def _win_caught_up(state: dict, win) -> None:
    """Windows, after the exit: is all the command wrote before exiting read? Each stream is done, has read what
    waited in its pipe at its first look after the exit, or is still in a read begun before the exit (nothing came)."""
    exited, wl = state.get("exited"), state.get("wlock")
    if wl is None or not exited.is_set():
        return
    with wl:
        if "exit_t" not in state:   # (the exit is being recorded)
            return
        settled = time.monotonic() - state.get("exit_t", 0.0) >= 0.05
        at_exit, reads = state.get("reads_at_exit", {}), state.get("reads", {})
        ok = all(state.get(fd) == "done" or state["owed"].get(fd, 1) <= 0
                 or (fd not in state["owed"] and state["blocked"].get(fd) and settled
                     and reads.get(fd, 0) == at_exit.get(fd, 0))   # (still in the read it was in at the exit)
                 for fd in state["fds"])
    if ok:
        win.caught_up = True


def _peek_pipe(fd: int) -> int:
    """Windows: bytes waiting in a pipe (PeekNamedPipe), 0 if that cannot be told."""
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
        avail = wintypes.DWORD(0)
        ok = ctypes.windll.kernel32.PeekNamedPipe(wintypes.HANDLE(msvcrt.get_osfhandle(fd)), None, 0, None,
                                                  ctypes.byref(avail), None)
        return int(avail.value) if ok else 0
    except (ImportError, OSError, AttributeError, ValueError):
        return 0


def _readable(fd: int, timeout: float) -> bool:
    try:   # (poll: no limit on fd numbers)
        p = select.poll()
        p.register(fd, select.POLLIN)
        return bool(p.poll(int(timeout * 1000)))
    except (OSError, ValueError, AttributeError):
        return True


def _pending(fd: int) -> int:
    """Bytes waiting in a pipe or pseudo-terminal."""
    try:
        import array
        import fcntl
        import termios
        n = array.array("i", [0])
        fcntl.ioctl(fd, termios.FIONREAD, n, True)
        return n[0]
    except (ImportError, OSError, AttributeError, ValueError):
        return 0


def _pump_all(streams: list, win: "Window", state: dict, outs: dict) -> None:
    """POSIX: one loop over all the command's streams, so marks on stdout and stderr are seen in the order they come.
    Each chunk is looked at as it is read (marks in each finished line, ended by \n or \r, so progress bars count)
    and handed, unchanged, to its _Out. If our own output is closed (e.g. `| head`), that stream is closed too, so
    the command sees it closed, as it would without joules. Once the command has exited, sets state["caught_up"]
    when all it wrote before exiting has been read. Stops (leaving the streams open) when state["stop"] is set."""
    open_ = {fd: dst for fd, dst, _ in streams}
    closers = {fd: close for fd, _, close in streams}
    order = [fd for fd, _, _ in streams]
    lines = {fd: _Lines() for fd in order}
    sel, watched, owed = selectors.DefaultSelector(), set(), None   # (selectors: no limit on fd numbers)
    pty_wait = set()
    wake = state.get("wake")   # (written when the command exits)
    if wake:
        sel.register(wake[0], selectors.EVENT_READ)

    def finish(fd):
        lines[fd].end(win, fd)
        state[fd] = "done"
        del open_[fd]
        if fd in watched:
            sel.unregister(fd)
            watched.discard(fd)
        with contextlib.suppress(OSError):
            closers[fd]()
    try:
        while open_:
            if win.paused and all(o.size == 0 for o in outs.values()) and not any(_pending(fd) for fd in open_):
                win.paused = False   # caught up (nothing queued, nothing waiting in the pipes): later marks are on time
            if state["stop"].is_set():   # handing over to the copier: everything read so far is with its _Out
                if win.pending() and not getattr(win, "caught_up", False) and any(_pending(fd) > 0 for fd in open_):
                    win.cut_short = True   # (output still unread: a marker may be in it)
                for fd in open_:
                    lines[fd].end(win, fd)   # (an unfinished last line counts now, as at the end of the output)
                    state[fd] = "stopped"
                return
            for fd in list(open_):
                if outs[fd].gone:
                    finish(fd)
                    continue
                room = not outs[fd].full()
                if room and fd not in watched:
                    sel.register(fd, selectors.EVENT_READ)
                    watched.add(fd)
                elif not room and fd in watched:
                    sel.unregister(fd)
                    watched.discard(fd)
                if not room and win.pending() and outs[fd].slow_reader():
                    win.paused = True
            if not open_:
                break
            if owed is None and state["exited"].is_set():   # what the command wrote before it exited
                # (a pseudo-terminal reports only part of what waits in it: there, until a read finds nothing)
                # (a pseudo-terminal reports only part of what waits in it: after that part, until a read finds
                # nothing or a short one shows a process writing a little at a time, or 2 s on)
                owed = {fd: _pending(fd) for fd in open_}
                pty_wait = {fd for fd in open_ if getattr(win, "ttys", {}).get(fd)}
                pty_wait -= {f for f in pty_wait if owed[f] <= 0 and not _readable(f, 0)}   # empty now: read
                owed_t = time.monotonic()
                win.behind = any(outs[fd].full() for fd in open_)
            if owed is not None and all(owed.get(fd, 0) <= 0 and fd not in pty_wait for fd in open_):
                state["caught_up"].set()
                win.caught_up = True   # all the command wrote before exiting is read: later lines came after
            try:
                if watched or wake:
                    ready = {k.fd for k, _ in sel.select(0.1)}
                    if wake and wake[0] in ready:   # the command exited: look at once
                        with contextlib.suppress(OSError):
                            os.read(wake[0], 64)
                    if owed is not None and pty_wait:   # a watched pseudo-terminal with nothing to read, or 2 s on
                        late = time.monotonic() - owed_t > 2.0
                        pty_wait -= {f for f in pty_wait if owed.get(f, 0) <= 0 and f in watched and (late or f not in ready)}
                else:
                    ready = set()
                    time.sleep(0.05)
            except (OSError, ValueError):
                break
            got, srcs = [], []
            for fd in order:
                if fd not in ready or fd not in open_:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except OSError:   # EIO from a pseudo-terminal once every writer has gone: the end of the output
                    chunk = b""
                if not chunk:
                    finish(fd)
                    continue
                win.first_out.setdefault(fd, time.monotonic())
                if owed is not None:
                    owed[fd] = owed.get(fd, 0) - len(chunk)
                    if fd in pty_wait and owed[fd] <= 0 and (len(chunk) < 1024 or not _readable(fd, 0.005)):
                        pty_wait.discard(fd)   # a short read, or a pause, once the reported part is read: no held burst
                if win.pending():
                    text = lines[fd].add(chunk, win, fd)
                    if text is not None:
                        got.append(text)
                        srcs.append(fd)
                else:
                    lines[fd].buf = b""
                outs[fd].put(chunk)
            if got and win.pending():
                win.feed(got, srcs)
        for fd in list(open_):
            finish(fd)
    finally:
        sel.close()
        state["caught_up"].set()


def _pump(fd: int, dst, win: "Window", state: dict, close=None, out=None) -> None:
    """Windows (pipes cannot be polled there): one reader per stream, otherwise as _pump_all. The command's pipe is
    closed at the end, also when our own output went away, so the command sees a broken pipe instead of blocking."""
    out = out or _Out(dst)
    lines = _Lines()
    while not state["stop"].is_set():
        held = win.__dict__.setdefault("paused_streams", set())   # (Windows reads each stream apart)
        while out.full() and not out.gone and not state["stop"].is_set():
            if win.pending() and getattr(out, "slow_reader", lambda: True)():
                held.add(fd)
                win.paused = True
            time.sleep(0.05)
        if fd in held and out.size == 0:   # this stream caught up; paused while any other has not
            held.discard(fd)
            win.paused = bool(held)
        if out.gone:
            break
        exited, wl = state.get("exited"), state.get("wlock")
        if wl is not None:   # after the exit: what waits in our pipe now (we are the only one reading it) comes first
            with wl:
                if exited.is_set() and fd not in state["owed"]:
                    state["owed"][fd] = _peek_pipe(fd)
                state["blocked"][fd] = not exited.is_set()   # (a read begun before the exit)
            _win_caught_up(state, win)
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        finally:
            if wl is not None:
                with wl:
                    state["blocked"][fd] = False
                    state.setdefault("reads", {})[fd] = state.get("reads", {}).get(fd, 0) + 1
        if not chunk:
            break
        win.first_out.setdefault(fd, time.monotonic())
        if win.pending():
            text = lines.add(chunk, win, fd)
            if text is not None:
                win.feed([text], [fd])
        out.put(chunk)
        if wl is not None:
            with wl:
                if fd in state["owed"]:
                    state["owed"][fd] -= len(chunk)
    lines.end(win, fd)
    state[fd] = "done"
    if state.get("wlock") is not None:
        _win_caught_up(state, win)
    held = win.__dict__.get("paused_streams")
    if held is not None and fd in held:   # (a stream that ended is not behind)
        held.discard(fd)
        win.paused = bool(held)
    if close is not None:
        with contextlib.suppress(OSError, ValueError):
            close()


class _Terminated(Exception):
    pass


_ENDING = [s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s is not None]


class _TermGuard:
    """POSIX, until the command's output is all passed on: a SIGTERM to joules alone ends joules as it would without
    a window, but first hands the command's streams (and output not yet passed on) to a helper, so the command and
    anything it left running keep their output. The signal only interrupts joules where it waits (raising()); at
    any other moment it is noted and acted on at the next wait."""

    def __init__(self):
        self.hit = self.armed = False
        self.sig = None   # the signal that asked joules to end (SIGTERM, or SIGHUP: a closed terminal)
        self.old, self.old_int = {}, None

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        with contextlib.suppress(ValueError, OSError):
            for sig in _ENDING:
                if signal.getsignal(sig) is signal.SIG_DFL:   # (an ignored or handled one is left as it is)
                    self.old[sig] = signal.signal(sig, self._on)
            if signal.getsignal(signal.SIGINT) is signal.default_int_handler:   # Ctrl-C too only lands where we wait
                self.old_int = signal.signal(signal.SIGINT, self._on_int)

    def _on(self, sig, *_):
        self.hit = True
        self.sig = self.sig or sig
        if self.armed:
            self.armed = False
            raise _Terminated

    def _on_int(self, *_):
        if self.armed:
            self.armed = False
            raise KeyboardInterrupt

    @contextlib.contextmanager
    def raising(self):
        self.armed = True
        try:
            if self.hit:
                raise _Terminated
            yield
        finally:
            self.armed = False

    def restore(self, ctrl_c: bool = True) -> None:
        """Put the handlers back (Ctrl-C only once the report is out, if ctrl_c is False now)."""
        for sig, old in (*self.old.items(), (signal.SIGINT, self.old_int if ctrl_c else None)):
            if old is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, old)
        self.old = {}
        if ctrl_c:
            self.old_int = None


def _hung_up(fd: int) -> bool:
    """Has every writer of this pipe or pseudo-terminal gone (whatever is still unread in it)?"""
    try:
        p = select.poll()
        p.register(fd, select.POLLIN)
        return any(ev & select.POLLHUP for _, ev in p.poll(0))
    except (OSError, ValueError, AttributeError):
        return False


def _drain_after_exit(proc, pumps, state, outs, notes, term) -> bool:
    """After the command exits: read all it wrote before exiting, wait 2 s more for its streams to close (a slow
    reader on our side may make that take longer: a stream all of whose writers have gone is read to its end), then
    hand a stream still held open (a process the command left running) to a helper. False if a Ctrl-C asked joules
    to stop waiting (POSIX: the caller then hands everything over)."""
    guard = term.raising if term else contextlib.nullcontext

    def alive():
        return any(t.is_alive() for t in pumps)

    def wait_until(until):
        while alive() and time.monotonic() < until:
            with guard():
                for t in pumps:
                    t.join(max(0.0, until - time.monotonic()))
    try:
        if os.name == "posix":
            with guard():
                while not state["caught_up"].wait(0.2) and alive():
                    pass
        wait_until(time.monotonic() + 2.0)
        if os.name == "posix":
            while alive() and all(_hung_up(fd) for fd, _, _ in proc.joules_streams if state.get(fd) != "done"):
                wait_until(time.monotonic() + 0.5)
    except KeyboardInterrupt:
        if os.name == "posix":
            return False
    if not alive():
        return True
    if os.name != "posix":   # (no helper there) a slow reader still emptying our queue is not a process left running
        quiet = time.monotonic()
        while alive():
            if any(o.size for o in outs.values()):
                quiet = time.monotonic()
            elif time.monotonic() - quiet > 2.0:
                break
            try:
                for t in pumps:
                    t.join(0.2)
            except KeyboardInterrupt:
                continue   # (no helper on Windows to pass the rest on: keep going)
        if not alive():
            return True
        notes.append("The command's output was still open when it exited (a process it started keeps it open); "
                     "joules stopped reading it.")
        state["left_open"] = True
        return True
    _note_open(proc, state)   # (as joules stops reading: a writer may still print, and go, while a slow reader drains)
    state["stop"].set()
    while alive():   # our reader never waits on a writer, so it stops at once
        try:
            for t in pumps:
                t.join()
        except KeyboardInterrupt:
            continue
    left = [(fd, dst) for fd, dst, _ in proc.joules_streams if state.get(fd) == "stopped"]
    try:
        with guard():
            for fd, _ in left:
                outs[fd].wait_empty()   # never drop output, however slow the reader on our side
    except KeyboardInterrupt:
        return False
    held = [fd for fd, _ in left if not _hung_up(fd)]
    moved = bool(left) and all(_hand_off(fd, dst) for fd, dst in left)
    for fd, _ in left:
        state[fd] = "handed"
        with contextlib.suppress(OSError):
            os.close(fd)
    if held:
        notes.append("The command's output was still open when it exited (a process it started keeps it open); " + (
            "a small helper keeps passing it on after joules finishes." if moved else "joules stopped reading it."))
    return True


def _note_open(proc, state) -> bool:
    """Right as joules stops reading: is a stream still open (a process the command left running may print more)?
    Sets state["left_open"]."""
    left = any(state.get(fd) not in ("done", "handed") and not _hung_up(fd) for fd, _, _ in proc.joules_streams)
    state["left_open"] = state.get("left_open", False) or left
    return left


def _hand_over(proc, pumps, state, outs, ending: bool = False) -> bool:
    """POSIX: stop reading and give every stream, with what joules read but has not passed on yet, to helpers.
    ending: joules ends at once (SIGTERM). True if a stream was still held open (a process the command started)."""
    held = _note_open(proc, state)
    state["stop"].set()
    for t in pumps:   # our reader never waits on a writer, and a search stops between lines: it stops at once
        t.join()
    for fd, dst, _ in proc.joules_streams:
        if state.get(fd) == "handed":
            continue
        maybe, pre = outs[fd].take(ending)
        src = fd if state.get(fd) == "stopped" else None
        if src is not None or pre or maybe:
            _hand_off(src, dst, pre, maybe, outs[fd].wrote_stuck)
        if src is not None:
            held = held or not _hung_up(src)
            state[fd] = "handed"
            with contextlib.suppress(OSError):
                os.close(fd)
    return held


def _end_on_term(proc, pumps, state, outs, sig=None) -> None:
    """SIGTERM (or SIGHUP) to joules alone: hand everything to helpers and end as the signal would (no report)."""
    sig = sig or signal.SIGTERM
    for s in _ENDING:   # a second one ends joules at once
        if signal.getsignal(s) not in (signal.SIG_IGN, None):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(s, signal.SIG_DFL)
    _hand_over(proc, pumps, state, outs, ending=True)
    os.kill(os.getpid(), sig)
    time.sleep(1)
    sys.exit(128 + sig)


COPIER = ("import os, select, sys\n"
          "while True:\n"   # wait for joules to finish (it holds the other end of this pipe), so its report comes first
          "    try:\n"
          "        if not os.read(int(sys.argv[1]), 1):\n"
          "            break\n"
          "    except OSError:\n"
          "        break\n"
          "for fd in [int(x) for x in sys.argv[2:]] + [0]:\n"   # first what joules had not passed on yet, then the stream
          "    while True:\n"
          "        try:\n"
          "            b = os.read(fd, 65536)\n"
          "        except OSError:\n"
          "            break\n"
          "        if not b:\n"
          "            break\n"
          "        try:\n"
          "            while b:\n"
          "                try:\n"
          "                    b = b[os.write(1, b):]\n"
          "                except BlockingIOError:\n"   # (an output someone made non-blocking: wait, as joules does)
          "                    select.select([], [1], [])\n"
          "        except OSError:\n"
          "            sys.exit()\n")


def _follow_resizes(proc):
    """When the user resizes the terminal, give the command's pseudo-terminals the new size too."""
    ptys = [(fd, dst) for fd, dst, _ in proc.joules_streams if _isatty(dst)]
    if not ptys or not hasattr(signal, "SIGWINCH") or threading.current_thread() is not threading.main_thread():
        return None
    echo = {"n": 0}   # signals we sent ourselves (the command shares joules' process group)

    def resize(*_):
        if echo["n"] > 0:
            echo["n"] -= 1
            return
        try:
            import fcntl
            import struct
            import termios
            for fd, dst in ptys:
                cols, rows = _term_size(dst)
                with contextlib.suppress(OSError):
                    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except ImportError:
            return
        if proc.poll() is None:   # the command and its children may have read their size already: tell them again
            echo["n"] += 1
            with contextlib.suppress(OSError):
                os.killpg(os.getpgrp(), signal.SIGWINCH)
    try:
        return ("set", signal.signal(signal.SIGWINCH, resize))
    except (ValueError, OSError):
        return None


def _stop_following(old) -> None:
    if old and hasattr(signal, "SIGWINCH"):
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGWINCH, old[1] if old[1] is not None else signal.SIG_DFL)


_GATE = None
_KEEP = []


def _hand_off(fd, dst, pre: bytes = b"", maybe: bytes = b"", wrote=None) -> bool:
    """A process the command left running still writes here: pass its output on from a small helper that outlives
    joules, so the process is not cut off (killed by SIGPIPE) when joules exits. The helper starts once joules has
    exited, with what joules read but had not passed on: maybe (a piece a stuck write of ours may yet write: as
    joules exits, it is dropped if wrote() says it was written), then pre; then the stream (fd None: no stream)."""
    import atexit
    import tempfile
    files = []
    try:
        for data in (maybe, pre):
            if data:
                f = tempfile.TemporaryFile()
                f.write(data)
                f.flush()
                f.seek(0)
                files.append(f)
        global _GATE
        if _GATE is None:
            _GATE = os.pipe()   # joules keeps the write end until it exits
        extra = [f.fileno() for f in files]
        subprocess.Popen([sys.executable, "-c", COPIER, str(_GATE[0]), *map(str, extra)],
                         stdin=fd if fd is not None else subprocess.DEVNULL, stdout=dst.fileno(),
                         stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
                         pass_fds=[_GATE[0], *extra])
        if maybe:
            first = files[0]

            def settle():
                if wrote is not None and wrote():
                    with contextlib.suppress(OSError):
                        os.ftruncate(first.fileno(), 0)
            atexit.register(settle)
            _KEEP.append(first)   # open until joules exits
        return True
    except (OSError, ValueError, AttributeError):
        return False
    finally:
        for f in files[1 if maybe else 0:]:
            f.close()


def _window_report(a, w: Window, s: Sampler, meters, idle: dict) -> tuple[dict, list[str]]:
    notes = []
    with w.lock:
        begin, end = w.begin, w.end
        alt = getattr(w, "alt_end", None) if end is None else None
        found_end = end
    if getattr(w, "match_s", 0.0) >= 1.0:
        by = getattr(w, "match_by", None) or {}
        given = [k for k, rx in (("--from", getattr(w, "start_re", None)), ("--until", getattr(w, "end_re", None))) if rx]
        slow_ones = [k for k in given if by.get(k, 0.0) >= 0.25 * w.match_s] or given or ["--from/--until"]
        tips = []
        for k in slow_ones:
            rx = {"--from": getattr(w, "start_re", None), "--until": getattr(w, "end_re", None)}.get(k)
            if rx is not None and not getattr(w, "must", {}).get(rx):
                tips.append(f"{k} has no plain text that every match must contain, so joules tries it on every line: "
                            "put a word in it if you can.")
            else:
                tips.append(f"Make {k} simpler: no leading repeat like \\w* or [^ ]+, no nested repeats.")
        advice = " ".join(tips) + " (A plain word is fastest.)"
        notes.append(f"Matching --from/--until took {w.match_s:.1f} s of joules' time, which held up joules (passing "
                     "the output on, measuring, even Ctrl-C) and so the command once its pipe was full: the run (and "
                     "any window) may be off. " + advice)
    for which, mark in (("--from", begin), ("--until", end)):
        if mark and mark.get("no_line_end"):
            notes.append(f"The {which} line had no line end, so it counted only when joules finished reading the "
                         "command's output.")
    for name in getattr(w, "untouched", ()):
        notes.append(f"Your {name} is a message socket (datagram or seqpacket), so the command writes to it directly "
                     "(passing it through joules would merge its messages); markers on it are not seen.")
    if alt is not None:
        end = alt
        notes.append("The --until line arrived on the other stream together with the --from line, so which came "
                     "first is not known; with no later --until line, joules took it as the end.")
    info = {"from": a.window_from, "until": a.window_until,
            "found_from": begin is not None and a.window_from is not None, "found_until": end is not None}
    cut = getattr(w, "cut_short", False)
    # (a process the command left running kept its output open: joules read all written until it stopped reading)
    left_open = not cut and getattr(w, "left_open", False)
    open_why = ("was not found in what joules read (a process the command left running kept its output open, and "
                "joules stopped reading it after the command exited)")
    if begin is None and (cut or left_open):
        notes.append(f"--from {a.window_from!r} " + ("was not found before joules stopped looking (it handed the "
                     "output over before it had looked at all of it)" if cut else open_why) + ", so there is no "
                     "window: the figures above are the whole run.")
        return {"window": info, "window_s": None, "window_energy_j": None}, notes
    if begin is None:
        notes.append(f"--from {a.window_from!r} never appeared in the command's output, so there is no window: "
                     "the figures above are the whole run.")
        return {"window": info, "window_s": None, "window_energy_j": None}, notes
    end = end or {**(w.frozen or {"t": s.t1, "j": {m.name: m.energy_j for m in meters}}), "line": None}
    if end["t"] < begin["t"]:   # (cannot end before it starts)
        end = {**end, "t": begin["t"], "j": begin["j"]}
    if a.window_until is not None and info["found_until"] is False:
        after = "after the --from line" if a.window_from is not None else "in the command's output"
        notes.append(f"--until {a.window_until!r} " + (
            "was not found before joules stopped looking (it handed the output over before it had looked at all "
            "of it)" if cut else open_why if left_open else f"never appeared {after}") + ", so the window runs to "
            "the end of the command.")
    exit_t = (w.frozen or {}).get("t", s.t1)
    if (found_end or {}).get("after_exit") and not begin.get("after_exit") and exit_t - s.t0 >= 0.2:
        notes.append("The --until line was printed after the command exited (by a process it left running), and "
                     "measuring stops when the command exits, so the window ends at the exit and misses the rest.")
    # Held back: the --from line read only after the exit, or a window of no time at the exit (a late --until alone
    # is just the command's last line, read a moment after it exited). Not for a run too short for it to matter.
    src, firsts = begin.get("src"), getattr(w, "first_out", None) or {}
    first = firsts.get(src) if src in firsts else min(firsts.values(), default=None)
    nothing_before = first is None or first >= exit_t - 0.1   # on the --from line's stream
    tty = getattr(w, "ttys", {}).get(src, False)
    # (no benchmark is that short; not when the --until line is one whose order is unknown, said above)
    tiny = a.window_from is not None and info["found_until"] and alt is None and end["t"] - begin["t"] < 0.01
    # with --from alone the window ends at the exit: under 10 ms of it, with nothing before on that stream, is held
    tiny_from = a.window_from is not None and a.window_until is None and exit_t - begin["t"] < 0.01
    at_exit = (tiny or tiny_from) and begin["t"] >= exit_t - 0.1
    held = at_exit and nothing_before
    # did joules pause for a slow reader before a mark it then found?
    paused = bool(begin.get("after_pause") or (found_end or {}).get("after_pause"))
    slow = getattr(w, "match_s", 0.0) >= 1.0   # (said above: that, not held output, is why marks came late)
    if exit_t - s.t0 >= 0.2 and (begin.get("late") or tiny or held) and not slow and not begin.get("no_line_end"):
        if (getattr(w, "behind", False) or paused) and (begin.get("late") or at_exit):
            notes.append("The marker lines were only read when the command exited, so the window may be wrong: what "
                         "reads joules' output (a pager, a slow pipe) took it so slowly that joules fell behind.")
        elif begin.get("after_exit"):
            notes.append("The --from line was printed after the command exited (by a process it left running), and "
                         "measuring stops when the command exits, so the window is empty.")
        elif begin.get("late") or held:
            notes.append("The marker lines were only read when the command exited, so the window may be wrong: the "
                         "command probably held its output until it ended" + (
                             "." if tty else " (a C program writing into a file or pipe does). Run it with `stdbuf "
                             "-oL` (Linux), or let joules write to a terminal."))
        elif not tty:
            notes.append("The window is under 10 ms, so its marker lines probably arrived together: if the command "
                         "held its output (a C program writing into a file or pipe does), the window is wrong. Run it "
                         "with `stdbuf -oL` (Linux).")
    if paused and not slow and not any("fell behind" in n for n in notes):
        notes.append("What reads joules' output (a pager, a slow pipe) fell more than 16 MB behind before the window's "
                     "markers, so joules paused reading the command's output and the window may be off.")
    live = [m for m in meters if m.kind != "est"]   # the CPU-time estimate is only known once the command has ended
    if len(live) < len(meters):
        notes.append("The CPU-time estimate is only known for the whole run, so it is not in the window.")
    if not live:
        info["unavailable"] = "no meter here can be read during the run"
        return {"window": info, "window_s": None, "window_energy_j": None}, notes
    meters = live
    secs = max(0.0, end["t"] - begin["t"])
    per = {m.name: max(0.0, end["j"].get(m.name, 0.0) - begin["j"].get(m.name, 0.0)) for m in meters}
    info.update(from_line=begin.get("line"), until_line=end.get("line"),
                starts_at_s=round(begin["t"] - s.t0, 3), ends_at_s=round(end["t"] - s.t0, 3), devices_j=per)
    out = {"window": info, "window_s": secs, "window_energy_j": sum(per.values())}
    if idle:   # above idle, per device as for the whole run
        out["window_above_idle_j"] = sum(max(0.0, j - idle[n] * secs) if n in idle else j for n, j in per.items())
    return out, notes


def render_run(r: dict) -> str:
    t = r["totals"]
    head = f"{bold('joules')} · {short_cmd(r['command'])} · {fmt_s(r['duration_s'])} · exit {r['exit_code']}"
    lines = [head, *_device_lines(r["devices"]), dim("  " + "─" * 58)]
    lines.append(f"  {bold('total')}  {bold(fmt_j(t['devices_j']))} = {fmt_wh(t['devices_j'])}"
                 + (f"   {dim('above idle')} {fmt_j(t['above_idle_j'])}" if "above_idle_j" in t else ""))
    if r.get("window_energy_j") is not None:
        w = r["window"]
        span = f"{w['starts_at_s']:.1f}–{w['ends_at_s']:.1f} s"
        lines.append(f"  {bold('window')} {bold(fmt_j(r['window_energy_j']))} over {fmt_s(r['window_s'])} ({span} into the run)"
                     + (f"   {dim('above idle')} {fmt_j(r['window_above_idle_j'])}" if "window_above_idle_j" in r else ""))
        def label(found, line, none):
            return (line or "(an empty line)") if found else none
        lines.append(dim(f"    from: {label(w['found_from'], w['from_line'], 'the start')}")
                     + dim(f"   until: {label(w['found_until'], w['until_line'], 'the end')}"))
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
        with Sampler(d.meters, a.interval, notes=d.notes) as s:
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
                    try:
                        r = ci.measure(d, cmd, cwd, a.interval)
                    except FileNotFoundError:
                        return _not_found(cmd)
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
                        try:
                            r = cmp.measure(d, argvs[i], a.shell, a.interval, a.quiet)
                        except FileNotFoundError:
                            d.close()
                            return _not_found(argvs[i] if isinstance(argvs[i], list) else [argvs[i]])
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
    notes += [n for n in d.notes if n not in notes]  # read errors the sampler explained during the runs
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
        with Sampler(d.meters, a.interval, notes=d.notes) as s:
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
            with Sampler(d.meters, a.interval, notes=d.notes) as s:
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
    wg = r.add_argument_group("window (optional: count only part of the run, e.g. a benchmark after the model loads)")
    wg.add_argument("--from", dest="window_from", metavar="REGEX", type=_regex_arg,
                    help="start counting at the first output line matching REGEX (the output still reaches you unchanged)")
    wg.add_argument("--until", dest="window_until", metavar="REGEX", type=_regex_arg,
                    help="stop counting at the first later line matching REGEX (default: the end of the command)")
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
    ag.add_argument("--interval", type=_interval_arg, default=1.0, help="seconds between readings (default 1)")
    ag.add_argument("--gpus", type=_gpus_arg, help="only these GPU indices, e.g. 0,1 (default all)")
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
    dr.add_argument("--gpus", type=_gpus_arg, help="only these GPU indices, e.g. 0,1")
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
    dv.add_argument("--gpus", type=_gpus_arg); dv.add_argument("--no-cpu", action="store_true"); dv.add_argument("--json", action="store_true")
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
    if isinstance(getattr(a, "out", None), str) and a.out and (why := _out_problem(a.out)):
        print(f"joules: cannot write --out {a.out}: {why}.", file=sys.stderr)
        return 2
    counted = a.cmd not in ("telemetry",)
    if counted:
        try:
            telemetry.record()
        except Exception:
            pass  # counting never gets in the way of a measurement
    rc = a.fn(a)
    if rc == 0 and getattr(a, "out_failed", False):
        rc = 1  # the result was printed, but the file asked for is missing
    if counted and rc == 0 and a.cmd not in ("agent", "proxy", "connect"):
        try:
            telemetry.maybe_ask()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
