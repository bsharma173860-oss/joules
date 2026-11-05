"""Anonymous usage counts, only for people who say yes.

joules asks once, the first time it finishes a command in a terminal. Nothing is sent until you answer yes. With
yes, once a day joules sends WattOps: a random ID made on this computer (not tied to you or your account), the
joules version, the OS, the GPU maker and how many joules commands ran since the last time. Never commands,
arguments, paths, file contents, host or user names, results, or your API key.

It never asks or sends in CI, when DO_NOT_TRACK=1 or JOULES_TELEMETRY=0 is set, never asks with JOULES_NO_PROMPT=1,
and never asks when the input or output is not a terminal (a pipe, a notebook or its ! cells, a service). Change your answer any time: `joules telemetry off` (or on, or status).
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import platform
import shutil
import sys
import threading
import urllib.request
import uuid

from . import __version__
from .share import _home, api_url

ENV_OFF = "JOULES_TELEMETRY"
CI_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "CIRCLECI", "JENKINS_URL", "TF_BUILD", "TEAMCITY_VERSION",
           "BITBUCKET_BUILD_NUMBER", "CODEBUILD_BUILD_ID", "DRONE", "TRAVIS")
FALSE = ("0", "false", "no", "off")
QUESTION = ("joules: Send anonymous usage counts to help improve joules? Once a day: the version, OS, GPU maker, "
            "how many commands ran and a random ID; never your commands, files or results. [y/N] ")


def state_file() -> str:
    return os.path.join(_home(), ".config", "landauer-gap", "telemetry.json")


def load() -> dict:
    try:
        with open(state_file(), encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def save(st: dict) -> None:
    path = state_file()
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except OSError:
        pass  # a read-only home: the question comes back next time, nothing is sent


def env_reason() -> str | None:
    """Why the environment turns counting off, or None."""
    if os.environ.get("DO_NOT_TRACK", "").strip().lower() not in ("", *FALSE):
        return "DO_NOT_TRACK is set"
    if os.environ.get(ENV_OFF, "").strip().lower() in FALSE:
        return f"{ENV_OFF} is off"
    ci = next((v for v in CI_VARS if os.environ.get(v)), None)
    return f"running in CI ({ci})" if ci else None


def enabled(st: dict | None = None) -> bool:
    st = load() if st is None else st
    return st.get("enabled") is True and env_reason() is None


def _interactive() -> bool:
    """A terminal someone can type into. A notebook's ! cell (Kaggle, Colab, Jupyter) runs the command in a
    pseudo-terminal, so isatty() says yes, but nothing typed in the notebook reaches it: a question there would hang
    the cell. So a ! cell is never interactive."""
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    from .connection import in_shell_cell
    return not in_shell_cell()


def gpu_maker() -> str:
    """A cheap guess without loading any driver library: nvidia, amd, intel, apple or none."""
    if sys.platform == "darwin":
        return "apple" if platform.machine() == "arm64" else "none"
    if os.path.exists("/proc/driver/nvidia") or shutil.which("nvidia-smi"):
        return "nvidia"
    vendors = set()
    for f in glob.glob("/sys/class/drm/card[0-9]*/device/vendor"):
        try:
            with open(f, encoding="ascii") as fh:
                vendors.add(fh.read().strip().lower())
        except OSError:
            pass
    return "amd" if "0x1002" in vendors else "intel" if "0x8086" in vendors else "none"


def os_name() -> str:
    return {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(sys.platform, "other")


def payload(st: dict, runs: int) -> dict:
    """Everything that is ever sent, nothing more."""
    return {"id": st["id"], "v": __version__, "os": os_name(), "gpu": gpu_maker(), "runs": max(0, min(int(runs), 100000))}


def _post(body: dict, timeout: float) -> None:
    req = urllib.request.Request(api_url() + "/api/health?ping=1", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": f"joules/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except Exception:
        pass  # offline, blocked or the site is down: counts are a nice-to-have, never an error


def record(today: str | None = None, sender=None) -> dict | None:
    """Count one command; once a day, send the count in the background. Returns what was sent, if anything."""
    st = load()
    if not enabled(st):
        return None
    st["runs"] = int(st.get("runs") or 0) + 1
    today = today or datetime.date.today().isoformat()
    body = None
    if st.get("last_sent") != today:
        body = payload(st, st["runs"])
        st["last_sent"], st["runs"] = today, 0
    save(st)
    if body:
        if sender:
            sender(body)
        else:
            threading.Thread(target=_post, args=(body, 3.0), daemon=True).start()
    return body


def set_enabled(on: bool) -> dict:
    st = load()
    st["enabled"] = bool(on)
    if on:
        st.setdefault("id", uuid.uuid4().hex)
        st.setdefault("runs", 0)
    save(st)
    return st


def maybe_ask(ask=input, sender=None) -> bool | None:
    """Ask once, in a terminal, outside CI. Returns the answer, or None when it did not ask."""
    st = load()
    if "enabled" in st or env_reason() or os.environ.get("JOULES_NO_PROMPT") or not _interactive():
        return None
    try:
        sys.stdout.flush()
        answer = ask(QUESTION).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    yes = answer in ("y", "yes")
    st = set_enabled(yes)
    if yes:
        body = payload(st, 1)
        st["last_sent"] = datetime.date.today().isoformat()
        save(st)
        (sender or (lambda b: _post(b, 2.0)))(body)
        print("joules: thank you. Turn this off any time with `joules telemetry off`.", file=sys.stderr)
    else:
        print("joules: nothing will be sent. Turn it on any time with `joules telemetry on`.", file=sys.stderr)
    return yes


def status_lines() -> list[str]:
    st = load()
    why = env_reason()
    if why:
        head = f"Usage counts: off ({why})."
    elif st.get("enabled") is True:
        head = "Usage counts: on. Once a day joules sends exactly this:"
    elif st.get("enabled") is False:
        head = "Usage counts: off (you said no). `joules telemetry on` turns them on."
    else:
        head = "Usage counts: not asked yet, so nothing is sent."
    out = [head]
    if st.get("enabled") is True and not why:
        out.append("  " + json.dumps(payload(st, st.get("runs") or 0)))
    out.append(f"Settings file: {state_file()}")
    return out
