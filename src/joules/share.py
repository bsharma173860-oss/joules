"""Sends results to WattOps (needs a free API key): `--share` to the public leaderboard, `--track` to
your private Tracking page.

--share sends only what the leaderboard shows: the model, the hardware, energy per token, speed and
average power. --track sends the run's energy, duration, cost, CO2, hardware, Landauer gap and a short
label (program and script name only, or your --label). Never prompts, outputs, arguments, host names
or file contents.
"""
from __future__ import annotations

import http.client
import json
import os
import platform
import re
import sys
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from . import __version__
from .model import SITE

KEY_ENV, URL_ENV = "LANDAUER_API_KEY", "LANDAUER_API_URL"
MIN_TOKENS = 32
KEY_RE = re.compile(r"^lgk_[A-Za-z0-9]{40}$")


def _home() -> str:
    """The real user's home, also under sudo (so a key saved by `joules login` is found by `sudo joules`)."""
    u = os.environ.get("SUDO_USER")
    if u and hasattr(os, "geteuid") and os.geteuid() == 0:
        h = os.path.expanduser(f"~{u}")
        if not h.startswith("~"):
            return h
    return os.path.expanduser("~")


def key_file() -> str:
    """Always ~/.config/landauer-gap/api-key: the website's API examples read the key from there too."""
    return os.path.join(_home(), ".config", "landauer-gap", "api-key")


def saved_key() -> str | None:
    try:
        with open(key_file(), encoding="utf-8") as f:
            k = f.read().strip()
        return k if KEY_RE.match(k) else None
    except OSError:
        return None


def save_key(key: str) -> str:
    path = key_file()
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    os.chmod(path, 0o600)
    _give_back(path)
    return path


def _give_back(path: str) -> None:
    """Under sudo the key is saved in the real user's home: make it (and its folder) theirs, so joules without sudo reads it."""
    try:
        uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    except (KeyError, ValueError):
        return
    if not (hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("SUDO_USER")) or not path.startswith(_home() + os.sep):
        return
    for p in (os.path.dirname(path), path):
        try:
            os.chown(p, uid, gid)
        except OSError:
            pass


NOT_A_KEY = ("is set, but not to an API key (keys are lgk_ followed by 40 letters and digits; a copied "
             "placeholder like lgk_… is not one), so joules ignores it. Remove it: unset {env}, and delete its line "
             "from ~/.zshrc or ~/.bashrc")
_warned: list = []


def bad_env_key() -> bool:
    """$LANDAUER_API_KEY holds something that is not a key, e.g. the placeholder lgk_… pasted from an example."""
    v = (os.environ.get(KEY_ENV) or "").strip()
    return bool(v) and not KEY_RE.match(v)


def env_key() -> str | None:
    v = (os.environ.get(KEY_ENV) or "").strip()
    return v if KEY_RE.match(v) else None


def api_key(override: str | None = None) -> str | None:
    """--key, else $LANDAUER_API_KEY, else the key saved by `joules login`. A $LANDAUER_API_KEY that is not a key
    (the placeholder lgk_… from an example) is skipped, with one warning, so the saved key still works."""
    if override:
        return override.strip()
    if bad_env_key() and not _warned:
        _warned.append(1)
        print(f"joules: ${KEY_ENV} " + NOT_A_KEY.format(env=KEY_ENV) + ".", file=sys.stderr)
    return env_key() or saved_key()


def _require_key(key: str) -> None:
    """Never send what cannot be a key: a placeholder's … cannot even go in an HTTP header. (Anything else the
    server judges, with its own message.)"""
    if not key or not re.fullmatch(r"[\x21-\x7e]+", key):
        raise RuntimeError("that is not an API key: keys are lgk_ followed by 40 letters and digits. "
                           f"Create one in Settings → API keys at {SITE}")


def check_key(key: str, base: str | None = None, timeout: float = 20) -> None:
    """Raises RuntimeError with the server's reason when the key is not accepted."""
    _require_key(key)
    url = api_url(base) + "/api/v1/track?limit=1&days=1"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": f"joules/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode())["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = f"HTTP {e.code}"
        raise RuntimeError(msg) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url} ({e.reason})") from None
    except (OSError, http.client.HTTPException, ValueError) as e:  # timeouts, dropped connections, a non-JSON reply
        raise RuntimeError(f"no usable answer from {url} ({e or type(e).__name__}); try again in a minute") from None


def cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout
            if out.strip():
                return out.strip()
        if platform.system() == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return re.sub(r"\s+", " ", winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "CPU"


def _device_name(meter_name: str) -> str:
    # "GPU 0 NVIDIA H100 80GB HBM3" -> "NVIDIA H100 80GB HBM3"
    n = re.sub(r"^(GPU|CARD)\s*\d+\s+", "", meter_name.strip(), flags=re.I)
    return re.sub(r"\s+", " ", n)[:120]


def measurement(report: dict) -> dict:
    """The leaderboard entry for a bench report. Raises ValueError when it cannot be shared."""
    if report.get("kind") != "bench":
        raise ValueError("only `joules bench` results can be shared")
    if report.get("simulated"):
        raise ValueError("simulated results cannot be shared. Run the benchmark on real hardware.")
    if any(d.get("kind") == "est" for d in report.get("devices", [])):
        raise ValueError("estimated results cannot be shared; the leaderboard only takes measured energy.")
    b, t, devs = report["bench"], report["totals"], report["devices"]
    out = int(b["output_tokens"])
    if out < MIN_TOKENS:
        raise ValueError(f"only {out} output tokens were measured; the leaderboard needs at least {MIN_TOKENS}")
    gpus = [d for d in devs if d["kind"] == "gpu"]
    if gpus:
        names = Counter(_device_name(d["name"]) for d in gpus)
        if len(names) > 1:
            raise ValueError("this run used different GPU models. Pick one model with --gpus.")
        device, count = next(iter(names)), len(gpus)
    else:
        device, count = (report.get("host") or {}).get("cpu") or cpu_name(), 1
    if "above_idle_j" not in t:  # the board compares what the model added, the same way on every machine
        if any("idle_w" in d for d in devs):
            raise ValueError("the model added no measurable power above idle, so there is no energy per token to share. "
                             "It may be running on hardware joules cannot see (joules devices lists what it measures).")
        raise ValueError("the leaderboard compares energy above idle: run the benchmark again with --subtract-idle 5")
    kind, _, model = b["target"].partition(":")
    basis = t["above_idle_j"]
    return {
        "model": model[:120], "backend": kind, "device": device[:120], "device_count": count,
        "j_per_token": basis / out, "tokens_per_s": b.get("tokens_per_s") or None,
        "avg_w": sum(d["avg_w"] for d in devs) or None, "output_tokens": out,
        "idle_subtracted": "above_idle_j" in t, "joules_version": __version__,
        "sources": "+".join(sorted({d["kind"] for d in devs})),
    }


GPU_FIELDS = ("multiplications", "energy_j", "duration_s", "throttled_pct", "clock_pct", "temp_c_end")


def gpu_measurement(report: dict) -> dict:
    """The GPU board entry for a `joules bench gpu` report: per precision the work, energy and time as measured
    (the server works out TFLOP/s, watts and GFLOP per joule), and the card's throttling. Raises ValueError."""
    if report.get("kind") != "gpubench":
        raise ValueError("only `joules bench gpu` results go on the GPU board")
    if report.get("simulated"):
        raise ValueError("simulated results cannot be shared. Run the test on a real GPU.")
    results = {p: {k: r[k] for k in GPU_FIELDS if r.get(k) is not None} for p, r in (report.get("results") or {}).items()}
    if not results:
        raise ValueError("the report has no results")
    return {"test": report["test"], "device": report["device"][:120], "meter": "counter" if "counter" in report["meter"] else "power",
            "power_limit_w": report.get("power_limit_w"), "results": results, "joules_version": __version__}


def api_url(override: str | None = None) -> str:
    return (override or os.environ.get(URL_ENV) or SITE).rstrip("/")


def board_link(model: str, base: str | None = None) -> str:
    return f"{api_url(base)}/leaderboard?model={urllib.parse.quote(model)}"


def gpu_board_link(device: str, base: str | None = None) -> str:
    return f"{api_url(base)}/leaderboard?gpu={urllib.parse.quote(device)}"


def send(m: dict, key: str, base: str | None = None, timeout: float = 20) -> dict:
    return _post("/api/v1/measurements", m, key, base, timeout)


def send_gpu(m: dict, key: str, base: str | None = None, timeout: float = 20) -> dict:
    return _post("/api/v1/measurements?kind=gpu", m, key, base, timeout)


# Hosts an upload may be redirected to with the key still attached: the site itself, with or without www, and the
# address it had before wattops.dev. A redirect anywhere else is refused, so the key cannot be sent to a stranger.
SITE_HOSTS = {"wattops.dev", "www.wattops.dev", "landauer-gap.vercel.app"}


def _same_site(a: str, b: str) -> bool:
    ha, hb = [(urllib.parse.urlsplit(u).hostname or "").lower() for u in (a, b)]
    strip = lambda h: h[4:] if h.startswith("www.") else h
    return bool(ha and hb) and (strip(ha) == strip(hb) or (ha in SITE_HOSTS and hb in SITE_HOSTS))


def _post(path: str, m: dict, key: str, base: str | None, timeout: float) -> dict:
    _require_key(key)
    url = api_url(base) + path
    body = json.dumps(m).encode()
    for hop in range(3):  # a 307 or 308 (a domain moved) is followed with the same POST, to the same site only
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": f"joules/{__version__}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            to = e.headers.get("Location") if e.code in (307, 308) else None
            if to and hop < 2:
                to = urllib.parse.urljoin(url, to)
                if (to.startswith("https://") or to.split("://")[0] == url.split("://")[0]) and _same_site(url, to):  # never https -> http
                    url = to
                    continue
                raise RuntimeError(f"{url} redirects to {to}, which is not the WattOps site; set {URL_ENV} to the right address") from None
            raise ServerError(e) from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {url} ({e.reason})") from None
        except (OSError, http.client.HTTPException, ValueError) as e:  # timeouts, dropped connections, a non-JSON reply
            raise RuntimeError(f"no usable answer from {url} ({e or type(e).__name__}); try again in a minute") from None


class ServerError(RuntimeError):
    """The server answered with an error: str() is its message; status and code say which."""

    def __init__(self, e):
        try:
            self.body = json.loads(e.read().decode())
        except (ValueError, OSError):
            self.body = None
        self.status = e.code
        try:  # a 429 says how long to wait
            self.retry_after = float(e.headers.get("Retry-After"))
        except (TypeError, ValueError, AttributeError):
            self.retry_after = None
        err = self.body.get("error") if isinstance(self.body, dict) else None
        self.code = err.get("code") if isinstance(err, dict) else None
        super().__init__(_server_error(self))


def _server_error(e) -> str:
    """The server's message, with which fields it refused and why."""
    try:
        err = e.body["error"]
        msg, fields = err["message"], err.get("fields")
    except (KeyError, TypeError):
        return f"HTTP {e.status}"
    if isinstance(fields, dict) and fields:
        msg += " " + "; ".join(f"{k}: {v}" for k, v in fields.items())
    elif isinstance(fields, list) and fields:
        msg += " " + "; ".join(f"{f.get('field')}: {f.get('message')}" if isinstance(f, dict) else str(f) for f in fields)
    return msg


# ---------------------------------------------------------------- tracking (private history, the Tracking page)
PROJECT_RE = re.compile(r"^[A-Za-z0-9._-]{1,60}$")


def safe_label(command: list[str]) -> str:
    """'python train.py' from 'python train.py --lr 3e-4 --api-key=…': program and first script only, never flags or values.
    Some programs (Chromium, Electron apps, PostgreSQL, nginx) rewrite their process title, so /proc gives the whole
    command line as one string: then only what comes before the first flag, or the first word, is the program."""
    if command and any(c.isspace() for c in command[0]):
        head = command[0].strip()
        cut = re.search(r"\s-", head)
        head = head[:cut.start()] if cut else head
        if not head.startswith("/"):  # a path may contain spaces ("/Applications/Google Chrome.app/…"); a bare title not
            head = head.split()[0] if head.split() else ""
        command = [head]
    parts = [os.path.basename(command[0])[:120]] if command and command[0] else []
    for arg in command[1:]:
        if not arg.startswith("-"):
            if len(arg) <= 120 and not any(c.isspace() for c in arg):  # a script or module, never inline code
                parts.append(os.path.basename(arg))
            break
    return " ".join(parts)[:200] or "command"


def track_record(report: dict, project: str, label: str | None = None) -> dict:
    """The private Tracking-page record for a run or bench report. Raises ValueError on a bad project name."""
    if not PROJECT_RE.match(project):
        raise ValueError("project names use letters, digits, dot, dash or underscore (1 to 60 characters)")
    t, devs = report["totals"], report["devices"]
    gpus = [d for d in devs if d["kind"] == "gpu"]
    if gpus:
        names = Counter(_device_name(d["name"]) for d in gpus)
        device = ", ".join(f"{n}× {name}" if n > 1 else name for name, n in names.items())
    else:
        device = (report.get("host") or {}).get("cpu") or cpu_name()
    if report["kind"] == "bench":
        default_label = report["bench"]["target"]
    else:
        default_label = safe_label(report.get("command") or [])
    r = {
        "project": project, "kind": report["kind"], "label": (label or default_label)[:200],
        "started_at": report.get("started_at"), "duration_s": max(report["duration_s"], 1e-3),
        "energy_j": t["facility_j"], "co2_g": t["co2"]["grams"], "cost": t["cost"]["amount"],
        "price_per_kwh": t["cost"]["price_per_kwh"], "g_per_kwh": t["co2"]["g_per_kwh"],
        "device": device[:120], "device_count": max(1, len(gpus)), "idle_subtracted": "above_idle_j" in t,
        "simulated": bool(report.get("simulated")), "joules_version": __version__,
        "sources": "+".join(sorted({d["kind"] for d in devs})),
        # joules measured whole devices around this one command: everything on them in that time is in the number
        "attribution": "whole_gpu" if gpus else "whole_device",
    }
    if "landauer" in t:
        r["operations"] = t["landauer"]["operations"]
        r["gap"] = t["landauer"]["gap"]
    hs = [d["gpu"] for d in gpus if d.get("gpu")]
    if hs:
        from .model import gpu_verdict
        g: dict = {}
        for k in ("clock_pct", "gpu_util_pct", "mem_util_pct", "throttled_pct"):
            vals = [h[k] for h in hs if k in h]
            if vals:
                g[k] = round(sum(vals) / len(vals), 1)
        for k in ("temp_c_max", "power_limit_w"):
            vals = [h[k] for h in hs if k in h]
            if vals:
                g[k] = round(max(vals), 1)
        reasons: dict = {}
        for h in hs:
            for k, v in (h.get("throttle_reasons") or {}).items():
                reasons[k] = max(reasons.get(k, 0), v)
        if reasons:
            g["reason"] = max(reasons, key=reasons.get)[:20]
        v = gpu_verdict({**g, "throttle_reasons": reasons})
        if v and v["bound"]:
            g["bound"] = v["bound"]
        r["gpu"] = g
    if report.get("exit_code") is not None:
        r["exit_code"] = report["exit_code"]
    if report["kind"] == "bench":
        r["output_tokens"] = report["bench"]["output_tokens"]
        r["j_per_token"] = report["bench"]["joules_per_output_token"]
    return r


def track_link(project: str, base: str | None = None) -> str:
    return f"{api_url(base)}/#/track?project={urllib.parse.quote(project)}"


NEWER_FIELDS = ("attribution", "coverage")  # joules 0.8; a server from before them refuses them as unknown


def send_track(r: dict, key: str, base: str | None = None, timeout: float = 20) -> dict:
    try:
        return _post("/api/v1/track", r, key, base, timeout)
    except ServerError as e:
        fields = ((e.body or {}).get("error") or {}).get("fields") if isinstance(e.body, dict) else None
        if e.status == 400 and isinstance(fields, dict) and fields and set(fields) <= set(NEWER_FIELDS) and any(k in r for k in NEWER_FIELDS):
            return _post("/api/v1/track", {k: v for k, v in r.items() if k not in NEWER_FIELDS}, key, base, timeout)
        raise


def send_live(beat: dict, key: str, base: str | None = None, timeout: float = 10) -> dict:
    """One Live heartbeat from `joules agent` (Free shows 1 GPU on Live; more are paid per GPU). Not counted against the monthly API calls."""
    return _post("/api/v1/track?live=1", beat, key, base, timeout)


def live_link(base: str | None = None) -> str:
    return f"{api_url(base)}/#/live"
