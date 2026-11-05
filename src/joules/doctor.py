"""joules doctor: checks this machine and prints a report anyone can share.

What joules can measure here and a live one-second reading from each meter, the GPU drivers, the CPU energy
counters, the Mac's power meters, the API key (whether one is saved and accepted, never the key), the connection
to WattOps and to Ollama. The report leaves out the machine's name, the user's name and home folder, and
any key, so it can be pasted into a chat or an issue as it is.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import __version__
from . import meters as M
from . import share as sharing

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
# a plausible one-second reading per meter kind, in watts
PLAUSIBLE = {"gpu": (1.0, 1500.0), "cpu": (0.05, 1000.0), "dram": (0.01, 300.0), "soc": (0.01, 200.0), "est": (0.0, 1e9)}


def _private(text: str) -> str:
    """The home folder and user name never appear in the report."""
    home = os.path.expanduser("~")
    if home and home not in ("/", "~"):
        text = text.replace(home, "~")
    for var in ("USER", "USERNAME", "LOGNAME"):
        u = os.environ.get(var)
        if u and len(u) > 2:
            text = text.replace(u, "<user>")
    return text


class Report:
    def __init__(self):
        self.sections: list[tuple[str, list[dict]]] = []

    def section(self, title: str) -> list[dict]:
        rows: list[dict] = []
        self.sections.append((title, rows))
        return rows

    def as_dict(self) -> dict:
        return {"joules_doctor": 1, "sections": [{"title": t, "checks": rows} for t, rows in self.sections]}

    def counts(self) -> dict:
        c = {OK: 0, WARN: 0, FAIL: 0, INFO: 0}
        for _, rows in self.sections:
            for r in rows:
                c[r["status"]] += 1
        return c


def add(rows: list[dict], status: str, what: str, detail: str = "") -> None:
    rows.append({"status": status, "check": what, "detail": _private(detail)})


def _get(url: str, timeout: float = 5, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": f"joules/{__version__}", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


# ---------------------------------------------------------------- the checks
def check_system(r: Report) -> None:
    s = r.section("This machine")
    add(s, INFO, "joules", __version__)
    add(s, INFO, "system", f"{platform.system()} {platform.release()} · {platform.machine()}")
    add(s, INFO, "CPU", sharing.cpu_name())
    v = sys.version_info
    add(s, OK if v >= (3, 9) else FAIL, "Python", f"{platform.python_version()} ({sys.executable})"
        + ("" if v >= (3, 9) else " · joules needs 3.9 or newer"))
    found = shutil.which("joules")
    add(s, OK if found else WARN, "joules command", found or "not on PATH: open a new terminal, or add ~/.local/bin to PATH")


def check_meters(r: Report, gpus: list[int] | None, seconds: float) -> list:
    s = r.section("What joules can measure")
    d = M.discover(gpus, cpu=True)
    meters = list(d.meters)
    try:
        if not meters:
            add(s, WARN, "energy meters", "none found, so joules estimates energy from CPU time and says so in every result")
        if meters:
            now = time.monotonic()
            for m in meters:
                m.start(now)
            end = now + seconds
            while time.monotonic() < end:
                time.sleep(0.1)
                t = time.monotonic()
                for m in meters:
                    m.poll(t)
            t = time.monotonic()
            for m in meters:
                m.stop(t)
            for m in meters:
                lo, hi = PLAUSIBLE.get(m.kind, (0.0, 1e9))
                w = m.avg_w
                if m.method == "SIMULATED":
                    st, note = WARN, f"{w:.3g} W, simulated (JOULES_SIMULATE is set), not a measurement"
                elif w <= 0:
                    st, note = FAIL, "reads 0 W: the meter is not counting"
                elif not lo <= w <= hi:
                    st, note = WARN, f"{w:.3g} W looks wrong for a {m.kind.upper()}"
                else:
                    st, note = OK, f"{w:.3g} W right now"
                add(s, st, m.name, f"{note} · {m.method}")
        for n in d.notes:
            if not n.startswith(("No energy counters found", "Simulated meters")):  # said above already
                add(s, WARN, "note", n)
    finally:
        d.close()
    return meters


def check_gpus(r: Report) -> None:
    s = r.section("GPU drivers")
    try:
        nv = M.NVML_FACTORY()
    except OSError as e:
        if "driver is installed" in str(e):
            add(s, WARN, "NVIDIA", str(e)[0].upper() + str(e)[1:])
        else:
            add(s, INFO, "NVIDIA", f"no driver library ({e})" if "not found" not in str(e) else "no NVIDIA driver on this machine")
        nv = None
    if nv is not None:
        try:
            n = nv.count()
            add(s, OK, "NVIDIA driver", f"{getattr(nv, 'driver_version', lambda: '')() or 'version unknown'} · {n} GPU{'s' if n != 1 else ''}")
            for i in range(n):
                h = nv.handle(i)
                energy, power = nv.energy_mj(h), nv.power_mw(h)
                how = "energy counter" if energy is not None else "power readings" if power is not None else "no energy or power"
                try:
                    procs = "process list available" if nv.processes(h) is not None else ""
                except (OSError, AttributeError):
                    procs = "no process list"
                extra = []
                for fn, fmt in (("clock_mhz", lambda v: f"{v} MHz"), ("temperature_c", lambda v: f"{v} °C"), ("power_limit_mw", lambda v: f"limit {v / 1000:.0f} W")):
                    try:
                        v = getattr(nv, fn)(h) if hasattr(nv, fn) else None
                    except (OSError, ValueError):
                        v = None
                    if v is not None:
                        extra.append(fmt(v))
                add(s, OK if energy is not None or power is not None else FAIL, f"GPU {i} {nv.name(h)}", " · ".join([how, procs, *extra]))
            if n:
                add(s, *clock_lock_status())
        finally:
            nv.close()
    amd = M._gpu_hwmons(("amdgpu",))
    intel = M._gpu_hwmons(("i915", "xe"))
    if amd:
        add(s, OK, "AMD GPUs (amdgpu)", f"{len(amd)} found")
    if intel:
        add(s, OK, "Intel GPUs (i915/xe)", f"{len(intel)} found")
    if platform.system() == "Windows":
        notes: list[str] = []
        meters, adlx = M.discover_adlx(notes)
        try:
            for m in meters:
                add(s, OK, m.name, f"ADLX · {m.reading}, sampled")
            if not meters:
                add(s, INFO, "AMD GPUs (ADLX)", notes[0] if notes else "none found (AMD graphics driver not installed)")
        finally:
            if adlx:
                adlx.close()
        add(s, INFO, "Intel GPUs", "not measured on Windows yet")


def check_cpu_counters(r: Report) -> None:
    sysname = platform.system()
    s = r.section("CPU energy counters")
    if sysname == "Linux":
        zones = sorted(glob.glob(os.path.join(M.sysfs(), "class/powercap/intel-rapl:*")))
        if not zones:
            add(s, WARN, "RAPL", "not present (normal in virtual machines and on ARM): CPU energy is estimated")
            return
        readable = 0
        for z in zones:
            try:
                with open(os.path.join(z, "energy_uj")) as f:
                    int(f.read())
                readable += 1
            except PermissionError:
                pass
            except (OSError, ValueError):
                pass
        if readable:
            add(s, OK, "RAPL", f"{readable} of {len(zones)} zones readable")
        else:
            add(s, WARN, "RAPL", "present but needs root: run with sudo, or: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj")
    elif sysname == "Darwin":
        root = hasattr(os, "geteuid") and os.geteuid() == 0
        if root:
            add(s, OK, "powermetrics", "running as root")
        elif shutil.which("sudo") and subprocess.call(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            add(s, OK, "powermetrics", "your password is cached; joules can read the power meters")
        else:
            add(s, INFO, "powermetrics", "joules asks for your password once when it measures (macOS allows only administrators)")
    elif sysname == "Windows":
        add(s, INFO, "CPU", "Windows does not let programs read CPU energy without a driver: CPU energy is estimated from CPU time")


def check_account(r: Report, api_url: str | None, offline: bool) -> None:
    s = r.section("WattOps")
    if sharing.bad_env_key():
        add(s, FAIL, "$" + sharing.KEY_ENV, sharing.NOT_A_KEY.format(env=sharing.KEY_ENV))
    key, env, saved = sharing.env_key() or sharing.saved_key(), sharing.env_key(), sharing.saved_key()
    where = "from $LANDAUER_API_KEY" if env else "saved by joules login" if saved else ""
    if not key:
        add(s, INFO, "API key", "none yet: --track and --share will ask for one (Settings → API keys)")
    if offline:
        add(s, INFO, "connection", "not checked (--offline)")
        return
    base = sharing.api_url(api_url)
    try:
        t0 = time.monotonic()
        code, _ = _get(base + "/api/health", 10)
        add(s, OK if code == 200 else WARN, "connection", f"{base} answered in {(time.monotonic() - t0) * 1000:.0f} ms")
    except (urllib.error.URLError, OSError, ValueError) as e:
        add(s, FAIL, "connection", f"cannot reach {base} ({getattr(e, 'reason', e)})")
        return
    if key:
        try:
            sharing.check_key(key, api_url)
            add(s, OK, "API key", f"accepted ({where})")
        except RuntimeError as e:
            add(s, FAIL, "API key", f"not accepted ({where}): {e}. Create a new one in Settings → API keys")


def check_ollama(r: Report, offline: bool) -> None:
    s = r.section("Local models")
    if offline:
        return
    try:
        _, body = _get("http://localhost:11434/api/tags", 3)
        models = json.loads(body).get("models") or []
        names = ", ".join(m.get("name", "?") for m in models[:6]) + (" …" if len(models) > 6 else "")
        add(s, OK, "Ollama", f"running · {len(models)} model{'s' if len(models) != 1 else ''}" + (f": {names}" if models else " (joules bench downloads one the first time)"))
    except (urllib.error.URLError, OSError, ValueError):
        from . import bench
        if bench._ollama_app() or shutil.which("ollama"):
            add(s, INFO, "Ollama", "installed, not running: joules bench starts it")
        else:
            add(s, INFO, "Ollama", "not installed (only needed for joules bench). " + bench.install_hint(then="").split(". ", 1)[1])


def _admin() -> bool:
    from .compare import is_admin
    return is_admin()


def clock_lock_status() -> tuple[str, str, str]:
    """Whether joules compare --lock-clocks can work here. Read-only: it asks nvidia-smi for GPU 0's clocks and
    checks for root (or sudo without a password); it never changes a clock."""
    from .compare import clock_info, smi
    what = "clock locking (--lock-clocks)"
    if not shutil.which("nvidia-smi"):
        return INFO, what, "nvidia-smi is not on PATH, so joules cannot lock clocks here"
    rc, _ = smi(["-i", "0", "--query-gpu=name", "--format=csv,noheader"], timeout=15)
    if rc != 0:
        return INFO, what, "nvidia-smi did not answer, so joules cannot lock clocks here"
    info = clock_info(0)
    clocks = ", ".join(x for x in (f"GPU 0 max {info['max']} MHz" if info["max"] else "",
                                   f"base {info['base']} MHz" if info["base"] else "") if x)
    clocks = f" ({clocks})" if clocks else ""
    if _admin():
        return OK, what, (f"permitted to try: running as {'administrator' if os.name == 'nt' else 'root'}{clocks}. The driver "
                          "still decides when you run it: containers and cloud notebooks (Kaggle, Colab) and MIG GPUs usually "
                          "refuse even root, and joules says so before any run")
    sudo = os.name != "nt" and shutil.which("sudo") and subprocess.call(["sudo", "-n", "true"], stdout=subprocess.DEVNULL,
                                                                       stderr=subprocess.DEVNULL) == 0
    if sudo:
        return OK, what, f"needs root: sudo works here without a password, so run sudo joules compare --lock-clocks …{clocks}"
    return INFO, what, (f"needs {'an administrator terminal' if os.name == 'nt' else 'root'}: run joules compare --lock-clocks "
                        f"{'as administrator' if os.name == 'nt' else 'with sudo'}{clocks}")


def run(gpus: list[int] | None = None, seconds: float = 1.0, api_url: str | None = None, offline: bool = False) -> Report:
    r = Report()
    check_system(r)
    check_meters(r, gpus, seconds)
    check_gpus(r)
    check_cpu_counters(r)
    check_account(r, api_url, offline)
    check_ollama(r, offline)
    return r


# ---------------------------------------------------------------- printing
def render(r: Report, color: bool, ascii_only: bool) -> str:
    marks = {OK: "OK  ", WARN: "!   ", FAIL: "FAIL", INFO: "-   "} if ascii_only else {OK: "✓", WARN: "!", FAIL: "✗", INFO: "·"}
    paint = {OK: "32", WARN: "33", FAIL: "31", INFO: "2"}
    c = (lambda code, t: f"\033[{code}m{t}\033[0m") if color else (lambda code, t: t)
    out = [f"joules doctor · joules {__version__}"]
    for title, rows in r.sections:
        if not rows:
            continue
        out.append("")
        out.append(title)
        w = max(len(x["check"]) for x in rows)
        for x in rows:
            out.append(f"  {c(paint[x['status']], marks[x['status']])} {x['check']:<{w}}  {x['detail']}")
    n = r.counts()
    out.append("")
    if n[FAIL]:
        out.append(c("31", f"{n[FAIL]} problem{'s' if n[FAIL] != 1 else ''} found") + (f", {n[WARN]} warning{'s' if n[WARN] != 1 else ''}" if n[WARN] else "") + ".")
    elif n[WARN]:
        out.append(c("33", f"Works, with {n[WARN]} warning{'s' if n[WARN] != 1 else ''}."))
    else:
        out.append(c("32", "Everything works."))
    out.append("Share this report as it is: it has no key, machine name, user name or home folder.")
    text = "\n".join(out)
    if ascii_only:  # a console that cannot show these gets plain equivalents, never "?" boxes
        for a, b in (("·", "-"), ("→", "->"), ("…", "..."), ("µ", "u"), ("×", "x"), ("─", "-")):
            text = text.replace(a, b)
    return text
