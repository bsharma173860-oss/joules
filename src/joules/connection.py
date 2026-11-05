"""joules connect: one step that puts a machine on WattOps under one name: its GPU on the leaderboard, its power and
jobs on the Live page (1 GPU free, more paid per GPU), and its finished jobs on the Tracking page, in a project named after it.

  In a notebook (Kaggle, Colab, Jupyter), in a Python cell:   import joules; joules.connect()
  In a terminal:                                               joules connect

The key is asked for in a hidden box (or read from --key, $LANDAUER_API_KEY or the key saved by joules login), checked,
and saved like joules login does; it is never printed, and never put on a command line. The machine's name defaults to
where it runs and its GPUs (kaggle-2x-t4, runpod-h100, 8x-a100), never its host name.

The agent then runs in the background (as a systemd service when run as root on a machine that has systemd, so it
also starts with the machine). Running connect again replaces the agent it started before; joules connect --stop,
or .stop() on what joules.connect() returns, stops it."""
from __future__ import annotations

import collections
import contextlib
import getpass
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import __version__
from . import share as sharing

# Notebook and GPU clouds, by the environment variables they set in every session.
PROVIDERS = (
    ("kaggle", ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")),
    ("colab", ("COLAB_RELEASE_TAG", "COLAB_GPU")),
    ("runpod", ("RUNPOD_POD_ID",)),
    ("vast", ("VAST_CONTAINERLABEL",)),
    ("paperspace", ("PAPERSPACE_FQDN", "PAPERSPACE_CLUSTER_ID")),
    ("lightning", ("LIGHTNING_CLOUD_PROJECT_ID",)),
    ("modal", ("MODAL_TASK_ID",)),
)
AZURE_TAG = "7783-7084-3265-9085-8269-3286-77"  # the chassis asset tag of every Azure VM
NAME_MAX = 60  # a Tracking project name (Live takes 64)
ENV_FILE = "/etc/default/joules-agent"
UNIT_FILE = "/etc/systemd/system/joules-agent.service"
SERVICE = "joules-agent"
# The same unit as integrations/cluster-agent/systemd/joules-agent.service (a test keeps them equal), with the joules
# that runs connect in place of `joules` on the PATH.
UNIT = """# joules agent on a GPU server or Slurm compute node: energy per Slurm job, container and user.
#   sudo pipx install landauer-gap   (or: sudo pip install landauer-gap)
#   sudo cp joules-agent.service /etc/systemd/system/ && sudo cp joules-agent.env /etc/default/joules-agent
#   sudo systemctl daemon-reload && sudo systemctl enable --now joules-agent
#   curl -s localhost:9877/jobs
[Unit]
Description=joules agent: energy per job, measured from the hardware
Documentation=https://pypi.org/project/landauer-gap/
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=-/etc/default/joules-agent
# root: reads RAPL counters and every job's cgroup; it never writes anywhere but stdout
ExecStart=/bin/sh -c 'exec joules agent --host "${JOULES_HOST:-127.0.0.1}" --port "${JOULES_PORT:-9877}" --price "${PRICE_PER_KWH:-0.10}" --grid "${GRID:-us}" --pue "${PUE:-1.0}" $JOULES_ARGS'
Environment=PATH=/usr/local/bin:/usr/bin:/bin:/root/.local/bin
Restart=always
RestartSec=5
Nice=10
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true
NoNewPrivileges=true
MemoryMax=256M

[Install]
WantedBy=multi-user.target
"""


class ConnectError(RuntimeError):
    """connect could not finish, with the reason and what to do."""


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- the machine's name
def _dmi(f: str) -> str:
    from .meters import sysfs
    try:
        with open(os.path.join(sysfs(), "class", "dmi", "id", f), encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def provider() -> str | None:
    """Where this runs: a notebook or GPU cloud by its environment, else AWS, Google Cloud or Azure by the VM's
    firmware (world-readable, no root). None on a machine of your own."""
    for name, envs in PROVIDERS:
        if any(os.environ.get(e) for e in envs):
            return name
    vendor, product = _dmi("sys_vendor"), _dmi("product_name")
    if vendor == "Amazon EC2" or product.startswith("Amazon EC2"):
        return "aws"
    if vendor == "Google" or product == "Google Compute Engine":
        return "gcp"
    if vendor == "Microsoft Corporation" and _dmi("chassis_asset_tag") == AZURE_TAG:
        return "azure"
    return None


_VENDOR = {"nvidia", "amd", "ati", "intel", "tesla", "geforce", "quadro", "radeon", "instinct", "graphics", "corporation"}
_JUNK = re.compile(r"^(\d+gb|sxm\d*|pcie|hbm\d*e?|nvl|generation|gpu|\d+)$")
_TWO = {"rtx", "gtx", "rx", "arc", "titan"}
_SUFFIX = {"ti", "super", "xt", "xtx"}


def short_gpu(name: str) -> str:
    """'Tesla T4' -> t4, 'NVIDIA A100-SXM4-80GB' -> a100, 'NVIDIA GeForce RTX 4090' -> rtx-4090, 'AMD Instinct MI300X' -> mi300x."""
    words = [w for w in re.split(r"[\s_/()-]+", re.sub(r"\((r|tm|c)\)", " ", name.lower())) if w and w not in _VENDOR]
    if not words:
        return "gpu"
    out = [words[0]]
    rest = words[1:]
    if out[0] in _TWO and rest:
        out.append(rest.pop(0))
    elif rest and _JUNK.match(out[0]):  # "GPU 0 Tesla T4" was not stripped by the caller
        return short_gpu(" ".join(rest))
    if rest and rest[0] in _SUFFIX:
        out.append(rest[0])
    return clean_name("-".join(out)).lower() or "gpu"


def gpu_names() -> list[str]:
    """The model of each GPU joules can measure here ('Tesla T4'), from the same meters the agent uses."""
    from .gpubench import device_name
    from .meters import discover
    try:
        d = discover(None, cpu=False)
    except Exception:  # noqa: BLE001  (a broken driver only means no GPU part in the name)
        return []
    try:
        return [re.sub(r"^(GPU|CARD)\s+", "", device_name(m.name), flags=re.I) for m in d.meters if m.kind == "gpu"]
    finally:
        d.close()


def clean_name(s: str) -> str:
    """Letters, digits, dot, dash and underscore, as Live and Tracking take them."""
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", s or "")).strip("-.")[:NAME_MAX].rstrip("-.")


def default_name(where: str | None = None, gpus: list[str] | None = None) -> str:
    """kaggle-2x-t4, colab-t4, runpod-h100, 8x-a100, mac-m3-max: where it runs and what it has, never the host name."""
    where = provider() if where is None else where
    gpus = gpu_names() if gpus is None else gpus
    if gpus:
        model, _ = collections.Counter(short_gpu(g) for g in gpus).most_common(1)[0]
        what = f"{len(gpus)}x-{model}" if len(gpus) > 1 else model
    elif platform.system() == "Darwin":
        chip = sharing.cpu_name()
        what = "mac-" + short_gpu(re.sub(r"^Apple\s+", "", chip)) if chip.startswith("Apple") else "mac"
    else:
        what = "cpu"
    return clean_name("-".join(x for x in (where, what) if x)) or "machine"


# ---------------------------------------------------------------- the key
def get_key(key: str | None = None, base: str | None = None, ask=None) -> str:
    """The key from the argument, $LANDAUER_API_KEY or the saved one, else asked for in a hidden box (in a notebook
    a masked input field). Checked with WattOps. Raises ConnectError."""
    key = (key or "").strip() or sharing.api_key()
    site = sharing.api_url(base)
    if not key:
        if ask is None:
            if _unattended() or (not (sys.stdin and sys.stdin.isatty()) and not _in_notebook()):
                raise ConnectError(f"no API key. Create one at {site}/#/settings/keys, then run joules connect in a terminal "
                                   "(it asks for the key without showing it), or set $LANDAUER_API_KEY.")
            ask = getpass.getpass
        say(f"Your WattOps API key: create one at {site}/#/settings/keys (Create key), copy it, and paste it here.")
        for _ in range(3):
            try:
                key = (ask("API key (hidden as you type, Enter when done): ") or "").strip()
            except (EOFError, KeyboardInterrupt):
                raise ConnectError("no key given, so nothing was connected.") from None
            except NotImplementedError:   # IPython's StdinNotImplementedError: a notebook run with no input box
                raise ConnectError(f"this notebook run cannot ask for a key. Pass key=, or set $LANDAUER_API_KEY "
                                   f"(on Kaggle, from Add-ons → Secrets). Keys are at {site}/#/settings/keys.") from None
            if sharing.KEY_RE.match(key):
                break
            say("That is not an API key: keys are lgk_ followed by 40 letters and digits. Copy it again from Settings → API keys.")
        else:
            raise ConnectError("no valid key given, so nothing was connected.")
    if not sharing.KEY_RE.match(key):
        raise ConnectError("that is not an API key: keys are lgk_ followed by 40 letters and digits. "
                           f"Create one at {site}/#/settings/keys.")
    try:
        sharing.check_key(key, base)
    except RuntimeError as e:
        raise ConnectError(f"the key was not accepted: {e}") from None
    return key


def _unattended() -> bool:
    """A notebook run with no one at it: Kaggle's Save Version runs every cell as a batch job."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"


def _in_notebook() -> bool:
    """This is a Jupyter kernel (Kaggle, Colab, JupyterLab): input() and getpass() show a box in the notebook."""
    return "ipykernel" in sys.modules


def in_shell_cell() -> bool:
    """A notebook's ! cell: a command started by the kernel, which cannot be typed into."""
    return "ipykernel" not in sys.modules and bool(os.environ.get("JPY_PARENT_PID") or provider() in ("kaggle", "colab"))


@contextlib.contextmanager
def _key_env(key: str):
    """The key for code run meanwhile, in the environment (never on a command line, which other users can read)."""
    old = os.environ.get(sharing.KEY_ENV)
    os.environ[sharing.KEY_ENV] = key
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(sharing.KEY_ENV, None)
        else:
            os.environ[sharing.KEY_ENV] = old


# ---------------------------------------------------------------- the name already on Live?
def live_nodes(key: str, base: str | None = None, timeout: float = 15) -> dict[str, float] | None:
    """The machines sending to Live right now (name -> seconds since their last reading). None when that cannot be
    told (a plan or server without Live to read, or WattOps unreachable)."""
    req = urllib.request.Request(sharing.api_url(base) + "/api/v1/track?live=1&minutes=15",
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": f"joules/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode() or "{}")
    except (OSError, ValueError, urllib.error.URLError):
        return None
    out = {}
    for n in body.get("nodes") or []:
        if isinstance(n, dict) and isinstance(n.get("node"), str):
            out[n["node"]] = float(n.get("age_s") or 0)
    return out


def free_name(name: str, taken: dict[str, float]) -> str:
    if name not in taken:
        return name
    for i in range(2, 100):
        n = f"{name[:NAME_MAX - 1 - len(str(i))]}-{i}"
        if n not in taken:
            return n
    raise ConnectError(f"{name} and {name}-2 to -99 are all on Live already; pick another name.")


# ---------------------------------------------------------------- the agent started earlier
def state_file() -> str:
    return os.path.join(os.path.dirname(sharing.key_file()), "agent.json")


def log_file() -> str:
    return os.path.join(os.path.dirname(sharing.key_file()), "agent.log")


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _is_our_agent(pid) -> bool:
    """That pid is still a joules agent (not a process that got its number later)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    c = _cmdline(pid)
    return "joules" in c and " agent" in c


def running() -> dict | None:
    """The agent joules connect started in the background, if it still runs: {pid, name, project, log, started}."""
    try:
        with open(state_file(), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        return None
    return st if isinstance(st, dict) and _is_our_agent(st.get("pid")) else None


_children: dict = {}  # the agents this process started, so they are waited for once stopped


def _kill(pid: int, timeout: float = 10) -> None:
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    p = _children.pop(pid, None)
    if p is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(timeout)
    end = time.monotonic() + timeout
    while time.monotonic() < end and _is_our_agent(pid):
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, os.WNOHANG)
        time.sleep(0.1)
    if _is_our_agent(pid):
        with contextlib.suppress(OSError):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def stop_background() -> dict | None:
    """Stops the background agent; returns what it was, or None when none ran."""
    st = running()
    if st:
        _kill(st["pid"])
    with contextlib.suppress(OSError):
        os.remove(state_file())
    return st


def start_background(name: str, project: str, key: str, base: str | None = None, price: float | None = None,
                     grid: str | None = None, pue: float | None = None, live_every: float = 5.0) -> dict:
    """joules agent in its own process, which keeps running after this cell or command ends (also when a notebook's
    kernel restarts), with its output in agent.log. The key goes in its environment."""
    log = log_file()
    os.makedirs(os.path.dirname(log), mode=0o700, exist_ok=True)
    argv = [sys.executable, "-m", "joules", "agent", "--port", "0", "--name", name, "--track", project,
            "--live-every", f"{live_every:g}", "--price", f"{0.10 if price is None else price:g}", "--grid", grid or "us",
            "--pue", f"{1.0 if pue is None else pue:g}"]
    if base:
        argv += ["--api-url", base]
    env = dict(os.environ, **{sharing.KEY_ENV: key, "JOULES_NO_PROMPT": "1"})
    env.pop("JOULES_NO_REEXEC", None)
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        kw = {"start_new_session": True} if os.name == "posix" else {"creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)}
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env, close_fds=True, **kw)
    st = {"pid": p.pid, "name": name, "project": project, "log": log, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    fd = os.open(state_file(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(st, f)
    st["_proc"] = _children[p.pid] = p
    return st


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def live_status(read_log, alive, timeout: float) -> str:
    """What the agent's log says about Live: on, limit: why (over the plan's GPUs), plan: why, failed: why, dead, or waiting."""
    end = time.monotonic() + timeout
    while True:
        text = read_log()
        if "  live: " in text:
            return "on"
        m = re.search(r"not on Live: (.*?) Trying again every minute\.", text)
        if m:
            return "limit: " + m.group(1).strip()
        m = re.search(r"Live is not on your plan \((.*)\), so", text)
        if m or "Live is part of the Max plan" in text:  # the second from joules before Live was on every plan
            return "plan: " + (m.group(1) if m else "Live is part of the Max plan.")
        m = re.search(r"Live stopped: (.*)", text)
        if m:
            return "failed: " + m.group(1).strip()
        if not alive():
            return "dead"
        if time.monotonic() >= end:
            m = re.search(r"Live heartbeat failed \((.*)\); retrying", text)
            return "failed: " + m.group(1) if m else "waiting"
        time.sleep(0.25)


# ---------------------------------------------------------------- as a systemd service (root)
def can_install_service() -> bool:
    """Root, on a machine whose init is systemd (not a container or a notebook, where the agent runs in the background)."""
    from .compare import is_admin
    return os.name == "posix" and is_admin() and os.path.isdir("/run/systemd/system") and bool(shutil.which("systemctl"))


def unit_text() -> str:
    """The unit, starting this same joules: the one on the PATH the unit sets, else this Python's."""
    path = "/usr/local/bin:/usr/bin:/bin:/root/.local/bin"
    found = shutil.which("joules", path=path)
    me = shutil.which("joules")
    if found and me and os.path.realpath(found) == os.path.realpath(me):
        return UNIT
    exe = me if me and not re.search(r"[\s'\"\\$`]", me) else None
    if exe is None:
        if re.search(r"[\s'\"\\$`]", sys.executable):
            return UNIT
        exe = f"{sys.executable} -m joules"
    return UNIT.replace("exec joules agent", f"exec {exe} agent")


DEFAULTS = {"PRICE_PER_KWH": "0.10", "GRID": "us", "PUE": "1.0"}


def env_text(old: str, key: str, name: str, project: str, price: float | None = None, grid: str | None = None,
             pue: float | None = None) -> str:
    """/etc/default/joules-agent with the key, the name and the project set; every other line kept as it was. A price,
    grid or PUE already there stays unless one is given."""
    new = {"LANDAUER_API_KEY": key, "JOULES_ARGS": f"--track {project} --name {name}"}
    for k, v in (("PRICE_PER_KWH", price), ("GRID", grid), ("PUE", pue)):
        if v is not None:
            new[k] = f"{v:g}" if isinstance(v, float) else str(v)
    lines, done = [], set()
    for line in old.splitlines():
        k = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if k in new:
            if k == "JOULES_ARGS":  # keep any other agent options set there
                v = line.split("=", 1)[1].strip().strip('"')
                v = re.sub(r"--(track|name)(=|\s+)\S+", "", v).split()
                line = f"JOULES_ARGS={' '.join([*v, new[k]])}"
            else:
                line = f"{k}={new[k]}"
            done.add(k)
        lines.append(line)
    if not lines:
        lines.append("# written by joules connect; read by the joules-agent service. Readable by root only.")
    for k, v in {**DEFAULTS, **new}.items():
        if k not in done and (k in new or not re.search(rf"^\s*{k}=", old, re.M)):
            lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


def _write_private(path: str, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


def install_service(key: str, name: str, project: str, price: float | None = None, grid: str | None = None,
                    pue: float | None = None, run=subprocess.run) -> None:
    """Writes the settings (root only) and the unit, then (re)starts the service. Raises ConnectError."""
    _write_private(ENV_FILE, env_text(_read(ENV_FILE), key, name, project, price, grid, pue))
    unit = unit_text()
    if _read(UNIT_FILE) != unit:
        with open(UNIT_FILE, "w", encoding="utf-8") as f:
            f.write(unit)
    for args in (["daemon-reload"], ["enable", SERVICE], ["restart", SERVICE]):
        r = run(["systemctl", *args], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise ConnectError(f"systemctl {' '.join(args)} failed: {(r.stderr or r.stdout).strip()[:300]}")


def service_log(run=subprocess.run) -> str:
    try:
        return run(["journalctl", "-u", SERVICE, "-n", "80", "--no-pager", "-o", "cat"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def service_active(run=subprocess.run) -> bool:
    try:
        return run(["systemctl", "is-active", "--quiet", SERVICE], timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ---------------------------------------------------------------- the GPU board
def run_board(gpu: int = 0, base: str | None = None) -> int:
    """joules bench gpu --share, as on the leaderboard page; its own output says what it measured."""
    from . import cli
    argv = ["bench", "gpu", "--share", "--gpus", str(gpu)] + (["--api-url", base] if base else [])
    a = cli.build_parser().parse_args(argv)
    a.argv = argv
    return a.fn(a)


# ---------------------------------------------------------------- what joules.connect() returns
class Connection:
    """This machine on WattOps: .name, .project, .links; .running; .stop() stops its agent."""

    def __init__(self, name: str, project: str, mode: str | None, links: dict, pid: int | None = None, log: str | None = None,
                 live: str = "", board: str = ""):
        self.name, self.project, self.mode, self.links, self.pid, self.log = name, project, mode, links, pid, log
        self.live, self.board = live, board

    @property
    def running(self) -> bool:
        if self.mode == "service":
            return service_active()
        return self.mode == "background" and _is_our_agent(self.pid)

    def stop(self) -> None:
        stop()

    def log_tail(self, lines: int = 20) -> str:
        """The agent's last lines (finished jobs, Live problems)."""
        text = service_log() if self.mode == "service" else _read(self.log or "")
        return "\n".join(text.splitlines()[-lines:])

    def __repr__(self) -> str:
        state = ("running" if self.running else "stopped") if self.mode else "no agent"
        return f"<joules connection {self.name}: agent {state}, project {self.project}>"


def stop(log=say) -> bool:
    """Stops the agent joules connect started: the background one, and the service when run as root."""
    st = stop_background()
    found = bool(st)
    if st:
        log(f"Stopped the joules agent of {st.get('name')} (it ran since {st.get('started')}).")
    if can_install_service() and os.path.exists(UNIT_FILE):
        r = subprocess.run(["systemctl", "disable", "--now", SERVICE], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            found = True
            log(f"Stopped and disabled the {SERVICE} service. Its key stays in {ENV_FILE} (root only); delete that file to remove it.")
    if not found:
        log("No joules agent from joules connect is running here.")
    return found


def connect(name: str | None = None, key: str | None = None, project: str | None = None, board: bool = True, agent: bool = True,
            gpu: int = 0, price: float | None = None, grid: str | None = None, pue: float | None = None, live_every: float = 5.0,
            save: bool = True,
            service: bool | None = None, api_url: str | None = None, ask=None) -> Connection:
    """Puts this machine on WattOps under one name (see the module's docstring). Returns a Connection.

    name: what the machine is called on Live and Tracking (default: where it runs and its GPUs, e.g. kaggle-2x-t4).
    key: the API key (default: $LANDAUER_API_KEY or the saved key; else asked for, hidden). Never put it in a notebook:
         on Kaggle use Add-ons → Secrets, on Colab the key icon, then key=UserSecretsClient().get_secret(...).
    project: the Tracking project for this machine's jobs (default: its name).
    board: run the one-minute GPU test and put the result on the GPU leaderboard (needs PyTorch and a GPU).
    agent: start joules agent (Live and Tracking) in the background; service=True/False forces or avoids systemd.
    price, grid, pue: electricity price per kWh, grid for CO2, facility overhead (defaults 0.10, us, 1.0; as root, the
         ones already in /etc/default/joules-agent stay unless given)."""
    from .model import GRIDS
    if grid is not None and grid not in GRIDS:
        raise ValueError(f"grid: one of {', '.join(sorted(GRIDS))}")
    if live_every and live_every < 2:
        raise ValueError("live_every: 0 (off) or at least 2 seconds")
    base = api_url
    key = get_key(key, base, ask)
    if save:
        path = sharing.save_key(key)
        say(f"Key checked and saved to {path} (only you can read it).")
    else:
        say("Key checked.")

    gpus = gpu_names()
    explicit = bool(name)
    name = clean_name(name) if name else default_name(gpus=gpus)
    if not name:
        raise ConnectError("name: use letters, digits, dot, dash or underscore, e.g. kaggle-t4")
    previous = running()
    taken = live_nodes(key, base) or {}
    mine = previous.get("name") if previous else None
    if service is not False and can_install_service() and agent and os.path.exists(UNIT_FILE) and service_active():
        mine = mine or _service_name()
    taken.pop(mine or "", None)  # our own agent from before, about to be replaced
    if name in taken:
        new = free_name(name, taken)
        say(f"{name} is already on Live from another machine (last reading {taken[name]:.0f} s ago), so this one is {new}."
            f"{' If that was this machine before a restart, wait a minute and connect again.' if not explicit else ''}"
            f" To pick the name, pass name=… (joules connect --name …).")
        name = new
    project = project or name
    if not sharing.PROJECT_RE.match(project):
        raise ConnectError("project: letters, digits, dot, dash or underscore (1 to 60 characters)")
    say(f"This machine is {name}" + (f" ({len(gpus)}× {gpus[0]})" if len(gpus) > 1 and len(set(gpus)) == 1
                                     else f" ({', '.join(gpus)})" if gpus else " (no GPU found)") + ".")

    links = {"tracking": sharing.track_link(project, base), "live": sharing.live_link(base)}
    mode, st = None, None
    agent_ok = agent and (os.path.isdir("/proc") or platform.system() == "Darwin")
    if agent and not agent_ok:
        say("The agent (Live and Tracking) runs on Linux and macOS; on this system only the GPU test runs.")
    if agent_ok:
        use_service = can_install_service() if service is None else service
        if use_service and not can_install_service():
            raise ConnectError("service=True needs root on a machine whose init is systemd")
        if previous:
            stop_background()
            say(f"Replaced the agent started earlier for {previous.get('name')}.")
        if use_service:
            install_service(key, name, project, price, grid, pue)
            mode = "service"
            say(f"Agent: installed as the {SERVICE} service; it starts with the machine. Settings and key: {ENV_FILE} (root only).")
        else:
            st = start_background(name, project, key, base, price, grid, pue, live_every)
            mode = "background"
            say(f"Agent: running in the background (process {st['pid']}, log {st['log']}). It runs until this machine or "
                "notebook session stops; run connect again after a restart.")
    t_agent = time.monotonic()

    board_state = ""
    if board:
        if not gpus:
            board_state = "no GPU"
            say("GPU leaderboard: skipped, joules found no GPU here.")
        else:
            say(f"GPU leaderboard: running the one-minute test on GPU {gpu}…")
            with _key_env(key):
                rc = run_board(gpu, base)
            board_state = "shared" if rc == 0 else "failed"
            if rc == 0:
                links["leaderboard"] = sharing.gpu_board_link(gpus[min(gpu, len(gpus) - 1)], base)

    live = ""
    if mode and live_every:
        wait = max(0.0, live_every + 10 - (time.monotonic() - t_agent))
        if mode == "service":
            live = live_status(service_log, service_active, wait)
        else:
            p = st["_proc"]
            live = live_status(lambda: _read(st["log"]), lambda: p.poll() is None, wait)
            if live == "dead":
                tail = "\n".join(_read(st["log"]).strip().splitlines()[-5:])
                raise ConnectError(f"the agent stopped right after starting:\n{tail}")
    say("")
    say(f"{name} is connected:")
    if mode:
        say(f"  Tracking  {links['tracking']}")
        say(f"            every program that runs a minute or more here and then ends, in project {project}")
        if _in_notebook():
            say("            (code in this notebook's own Python is one job until the session ends: to track a cell, "
                f"wrap it in  with joules.measure(\"train\", track=\"{project}\"):)")
        live_line = {"on": "this machine's power and jobs every few seconds",
                     "waiting": "no reading has arrived yet; it should within a minute",
                     "": "off (live_every=0)"}.get(live) or (
            ("not on Live: " if live.startswith(("limit:", "plan:")) else "not working: ") + live.split(": ", 1)[-1])
        say(f"  Live      {links['live']}  {live_line}")
    if "leaderboard" in links:
        say(f"  GPU board {links['leaderboard']}")
    if mode == "background":
        say("  Stop it:  joules connect --stop" + ("   (or .stop() on what joules.connect() returned)" if _in_notebook() else ""))
    elif mode == "service":
        say(f"  Stop it:  sudo joules connect --stop   ·   its log: journalctl -u {SERVICE} -f")
    return Connection(name, project, mode, links, st["pid"] if st else None, st["log"] if st else None, live, board_state)


def _service_name() -> str | None:
    m = re.search(r"--name\s+(\S+)", _read(ENV_FILE))
    return m.group(1) if m else None
