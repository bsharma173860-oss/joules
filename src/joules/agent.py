"""joules agent: energy per job on a shared machine (a GPU server, a Slurm node, a Kubernetes node).

Runs next to the workloads, measures the whole machine every interval and splits each device's energy
between the jobs using it:

  GPUs (NVIDIA)  all of it to a job alone on the GPU; between several, by each process's SM time (the driver's
                 per-process utilisation samples, or accounting mode), else by the GPU memory each holds. A GPU
                 with no process is "idle" energy, and one held by a job but under --idle-util % busy counts as idle
  CPUs and DRAM  by the CPU time each process used in the interval
  AMD and Intel  between the processes by the GPU time each used (the kernel's DRM usage stats); on kernels
  GPUs           without them, to the job doing most of the CPU work, else "system"

A job is a Slurm job, a Kubernetes pod, a container, or else a user's program, found from /proc/<pid>/cgroup.
Serves Prometheus metrics at /metrics and the jobs at /jobs. A program can also mark phases of its own work
(POST /phase/start and /phase/end; GET /phases for the medians): each mark reads the meters at once, so a
phase's energy is the change in each device's total between its two marks (see phases.py). With --track, each finished job is sent to the
WattOps Tracking page (a team key sends it to the team), and a heartbeat every few
seconds (the node's power, idle GPUs and running jobs) feeds the Live page. On macOS there is no /proc, so
the whole machine is one "system" job. Reads names from a process's environment only
for SLURM_JOB_NAME, HOSTNAME (a pod's name) and POD_NAMESPACE; never command arguments or files.
"""
from __future__ import annotations

import json
import os
import re
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from . import share as sharing
from .meters import Meter
from .phases import PhaseClock, PhaseError

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def proc_root() -> str:
    return os.environ.get("JOULES_PROC", "/proc")


# ---------------------------------------------------------------- which job a process belongs to
_SLURM = re.compile(r"/job_(\d+)(?:/|$)")
_POD = re.compile(r"pod([0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{12})")
_DOCKER = re.compile(r"(?:docker[-/]|containerd[-/]|cri-containerd-|crio-)([0-9a-f]{12,64})")
_ENV_KEYS = (b"SLURM_JOB_NAME=", b"HOSTNAME=", b"POD_NAMESPACE=", b"SLURM_JOB_USER=")


@dataclass
class JobKey:
    kind: str       # slurm | k8s | container | process
    id: str
    name: str
    user: str
    namespace: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.id}"


def _read(path: str, mode: str = "r"):
    with open(path, mode) as f:
        return f.read()


def _user(uid: int) -> str:
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return str(uid)


def _env(pid: int) -> dict[str, str]:
    """Only the few names we label jobs with; nothing else from the environment is kept."""
    out: dict[str, str] = {}
    try:
        raw = _read(f"{proc_root()}/{pid}/environ", "rb")
    except OSError:
        return out
    for item in raw.split(b"\0"):
        for k in _ENV_KEYS:
            if item.startswith(k):
                out[k[:-1].decode()] = item[len(k):].decode(errors="replace")[:120]
    return out


def job_of(pid: int, uid: int, comm: str, cmdline: list[str]) -> JobKey:
    try:
        cg = _read(f"{proc_root()}/{pid}/cgroup")
    except OSError:
        cg = ""
    user = _user(uid)
    m = _SLURM.search(cg)
    if m:
        env = _env(pid)
        return JobKey("slurm", m.group(1), env.get("SLURM_JOB_NAME") or f"job {m.group(1)}", env.get("SLURM_JOB_USER") or user)
    m = _POD.search(cg)
    if m:
        uid_ = m.group(1).replace("_", "-")
        env = _env(pid)
        return JobKey("k8s", uid_, env.get("HOSTNAME") or f"pod {uid_[:8]}", user, env.get("POD_NAMESPACE", ""))
    m = _DOCKER.search(cg)
    if m:
        env = _env(pid)
        return JobKey("container", m.group(1)[:12], env.get("HOSTNAME") or f"container {m.group(1)[:12]}", user)
    label = sharing.safe_label(cmdline) if cmdline else comm
    return JobKey("process", f"{uid}:{label}", label, user)


@dataclass
class Proc:
    pid: int
    ticks: int          # user + system CPU time, in clock ticks
    job: JobKey


def scan_processes(cache: dict | None = None) -> dict[int, Proc]:
    """Every user-space process with its CPU time and job. cache keeps each process's job between scans (keyed by
    pid and start time, so a reused pid is looked up again) so cgroup and environment are read once per process."""
    out: dict[int, Proc] = {}
    seen = set()
    root = proc_root()
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            stat = _read(f"{root}/{pid}/stat")
            rest = stat[stat.rindex(")") + 2:].split()   # fields after "pid (comm)"
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            ppid, ticks, started = int(rest[1]), int(rest[11]) + int(rest[12]), rest[19]
            if ppid == 2 or pid == 2:
                continue  # kernel threads
            ck = (pid, started)
            if cache is not None and ck in cache:
                out[pid] = Proc(pid, ticks, cache[ck]); seen.add(ck)
                continue
            uid = 0
            for line in _read(f"{root}/{pid}/status").splitlines():
                if line.startswith("Uid:"):
                    uid = int(line.split()[1]); break
            cmd = [c for c in _read(f"{root}/{pid}/cmdline", "rb").decode(errors="replace").split("\0") if c]
        except (OSError, ValueError, IndexError):
            continue
        job = job_of(pid, uid, comm, cmd)
        out[pid] = Proc(pid, ticks, job)
        if cache is not None:
            cache[ck] = job; seen.add(ck)
    if cache is not None:
        for ck in [k for k in cache if k not in seen]:
            del cache[ck]
    return out


# ---------------------------------------------------------------- GPU time per process (AMD, Intel: DRM fdinfo)
_DRM_ENGINE = re.compile(r"^drm-(?:engine-[\w-]+:\s+(\d+)\s+ns|cycles-[\w-]+:\s+(\d+))\s*$", re.M)


class DrmUsage:
    """How long each process kept each GPU busy, from the kernel's DRM usage stats (/proc/<pid>/fdinfo, Linux 5.19+
    for amdgpu, 6.0+ for i915, xe). Engine time is in ns (amdgpu, i915) or cycles (xe); either way it is only compared
    between clients of the same card."""

    def __init__(self, rescan_s: float = 10.0):
        self.rescan_s = rescan_s
        self.fds: dict[int, tuple[float, list[str]]] = {}   # pid -> (when scanned, fd numbers open on /dev/dri)
        self.prev: dict[tuple, int] = {}                     # (pid, card, client id) -> busy so far

    def _drm_fds(self, pid: int, now: float) -> list[str]:
        hit = self.fds.get(pid)
        if hit and now - hit[0] < self.rescan_s:
            return hit[1]
        out = []
        base = f"{proc_root()}/{pid}/fd"
        try:
            for fd in os.listdir(base):
                try:
                    if os.readlink(f"{base}/{fd}").startswith("/dev/dri/"):
                        out.append(fd)
                except OSError:
                    continue
        except OSError:
            pass
        self.fds[pid] = (now, out)
        return out

    def step(self, pids, now: float) -> dict[str, dict[int, int]]:
        """card PCI address -> {pid: busy time since the last step}. Every process with the card open is listed."""
        out: dict[str, dict[int, int]] = {}
        cur: dict[tuple, int] = {}
        for pid in pids:
            for fd in self._drm_fds(pid, now):
                try:
                    info = _read(f"{proc_root()}/{pid}/fdinfo/{fd}")
                except OSError:
                    continue
                card = re.search(r"^drm-pdev:\s*(\S+)", info, re.M)
                client = re.search(r"^drm-client-id:\s*(\d+)", info, re.M)
                if not card:
                    continue
                k = (pid, card.group(1), client.group(1) if client else fd)
                if k in cur:
                    continue  # the same client through a duplicated fd
                cur[k] = sum(int(a or b) for a, b in _DRM_ENGINE.findall(info))
                d = max(0, cur[k] - self.prev.get(k, cur[k]))
                row = out.setdefault(k[1], {})
                row[pid] = row.get(pid, 0) + d
        self.prev = cur
        live = set(pids)
        self.fds = {p: v for p, v in self.fds.items() if p in live}
        return out


# ---------------------------------------------------------------- energy per job
@dataclass
class Job:
    key: JobKey
    first_seen: float
    last_seen: float
    energy_j: float = 0.0
    idle_j: float = 0.0          # GPU energy while this job held a GPU but used under --idle-util %
    cpu_s: float = 0.0
    devices: set = field(default_factory=set)
    gpu_devices: set = field(default_factory=set)
    kinds: set = field(default_factory=set)
    method_j: dict = field(default_factory=dict)   # how its energy was given to it: method -> joules (see METHODS)
    covered_s: float = 0.0                         # the part of its wall time the agent actually sampled

    def used(self, m: Meter) -> None:
        self.devices.add(m.name); self.kinds.add(m.kind)

    def give(self, m: Meter, joules: float, method: str) -> None:
        self.energy_j += joules
        self.method_j[method] = self.method_j.get(method, 0.0) + joules
        self.used(m)

    @property
    def attribution(self) -> str:
        """How the job's energy was split, by the method that gave it the most GPU energy (else its CPU split)."""
        gpu = {k: v for k, v in self.method_j.items() if k in GPU_METHODS}  # 0 J by SM time is still an SM-time split
        if gpu:
            return max(gpu, key=gpu.get)
        return "cpu_time"

    @property
    def coverage(self) -> float:
        """The share of the job's wall time that the agent sampled (1.0 when it saw every interval on time)."""
        wall = self.last_seen - self.first_seen
        return 1.0 if wall <= 0 else max(0.0, min(1.0, self.covered_s / wall))

    def as_dict(self, price: float, g_per_kwh: float, pue: float = 1.0) -> dict:
        """Times are Unix seconds. Cost and CO2 include the facility overhead (pue); energy_j is device energy."""
        kwh = self.energy_j * pue / 3.6e6
        return {"kind": self.key.kind, "id": self.key.id, "name": self.key.name, "user": self.key.user,
                "namespace": self.key.namespace or None, "started": self.first_seen, "last_seen": self.last_seen,
                "duration_s": round(self.last_seen - self.first_seen, 3), "energy_j": self.energy_j, "idle_gpu_j": self.idle_j,
                "cpu_s": round(self.cpu_s, 2), "facility_j": self.energy_j * pue, "cost": kwh * price, "co2_g": kwh * g_per_kwh,
                "devices": sorted(self.devices), "gpus": sorted(self.gpu_devices),
                "attribution": self.attribution, "attribution_j": {k: round(v, 3) for k, v in sorted(self.method_j.items())},
                "coverage": round(self.coverage, 4), "sampled_s": round(self.covered_s, 3)}


# How a shared device's energy is given to jobs, from exact to rough. Every finished job says which one it got.
METHODS = {
    "whole_gpu": "whole GPU: the job had the GPU to itself in that interval, so all of its energy is the job's",
    "sm_time": "SM time: a shared NVIDIA GPU split by each process's SM-busy time (NVML per-process utilisation or accounting)",
    "busy_time": "busy time: a shared AMD or Intel GPU split by each process's engine time (the kernel's DRM counters)",
    "memory_share": "memory share: a shared NVIDIA GPU split by the memory each process held (SM time not available)",
    "cpu_guess": "CPU guess: no per-process GPU data, so the GPU went to the job doing most of the CPU work",
    "cpu_time": "CPU time: CPU and DRAM split by each process's CPU time",
}
GPU_METHODS = ("whole_gpu", "sm_time", "busy_time", "memory_share", "cpu_guess")


SYSTEM = JobKey("system", "system", "system and other processes", "")
IDLE = JobKey("idle", "idle-gpu", "GPUs with no job", "")


class Accountant:
    """Splits each interval's device energy between jobs. Pure bookkeeping, so it can be tested without hardware."""

    def __init__(self, idle_util: float = 5.0, finish_after: float = 30.0, interval: float | None = None):
        self.idle_util, self.finish_after, self.interval = idle_util, finish_after, interval
        self.jobs: dict[str, Job] = {}
        self.finished: list[Job] = []
        self.ticks: dict[int, int] = {}
        self.device_j: dict[str, float] = {}
        self.device_w: dict[str, float] = {}
        self.gpu_util: dict[str, float] = {}
        self.gpu_health: dict[str, dict] = {}   # GPU -> the latest clocks, temperature, utilisation and throttle reading
        self.gpu_idle: dict[str, bool] = {}      # GPU -> nobody used it in the last interval
        self.job_w: dict[str, float] = {}        # job -> its power over the last interval
        self.job_idle_w: dict[str, float] = {}   # job -> the part of that on GPUs it held but barely used
        self.lock = threading.Lock()
        self.last_step: float | None = None
        self.own_pid = os.getpid()

    def _job(self, k: JobKey, now: float) -> Job:
        j = self.jobs.get(k.key)
        if j is None:  # it was already working during the interval that just ended
            j = self.jobs[k.key] = Job(k, self.last_step if self.last_step is not None else now, now)
        j.last_seen = now
        return j

    def step(self, now: float, dt: float, procs: dict[int, Proc], deltas: list[tuple[Meter, float]], gpu_info: dict,
             complete: bool = True) -> None:
        """deltas: (meter, joules in this interval). gpu_info: meter name -> {"procs": [(pid, GPU memory)], "util": % or None,
        "sm": {pid: SM-time weight} when the driver gives it} (NVIDIA) or {"busy": {pid: GPU time in this interval}} (AMD,
        Intel). complete: every meter was read this time; an interval with a missed reading does not count as sampled."""
        with self.lock:
            before = {k: (j.energy_j, j.idle_j) for k, j in self.jobs.items()}
            # CPU time per job in this interval
            d_ticks: dict[str, int] = {}
            keys: dict[str, JobKey] = {}
            total = 0
            for pid, p in procs.items():
                d = max(0, p.ticks - self.ticks.get(pid, p.ticks))
                if d:
                    d_ticks[p.job.key] = d_ticks.get(p.job.key, 0) + d
                    keys[p.job.key] = p.job
                    total += d
            self.ticks = {pid: p.ticks for pid, p in procs.items()}
            own = procs[self.own_pid].job.key if self.own_pid in procs else None
            busiest = max((k for k in d_ticks if k != own), key=lambda k: d_ticks[k], default=None)
            for k, t in d_ticks.items():
                self._job(keys[k], now).cpu_s += t / CLK_TCK

            for m, e in deltas:
                self.device_j[m.name] = self.device_j.get(m.name, 0.0) + e
                self.device_w[m.name] = e / dt if dt > 0 else 0.0
                info = gpu_info.get(m.name)
                if info is not None and info.get("health"):
                    self.gpu_health[m.name] = info["health"]
                if e <= 0:
                    continue
                if m.kind == "gpu" and info is not None and "busy" in info:
                    busy = {pid: t for pid, t in info["busy"].items() if pid in procs}
                    self.gpu_idle[m.name] = not any(busy.values())
                    if not busy:
                        j = self._job(IDLE, now); j.energy_j += e; j.idle_j += e; j.used(m)
                        continue
                    w: dict[str, float] = {}
                    for pid, t in busy.items():
                        keys[procs[pid].job.key] = procs[pid].job
                        w[procs[pid].job.key] = w.get(procs[pid].job.key, 0.0) + t
                    tot = sum(w.values())
                    method = "whole_gpu" if len(w) == 1 else "busy_time"
                    for k in w:  # the card was open but nobody used it: idle, shared equally
                        j = self._job(keys[k], now)
                        share = e * w[k] / tot if tot else e / len(w)
                        j.give(m, share, method)
                        if not tot:
                            j.idle_j += share
                        j.gpu_devices.add(m.name)
                elif m.kind == "gpu" and info is not None:
                    util = info.get("util")
                    if util is not None:
                        self.gpu_util[m.name] = util
                    holders = [(procs[pid].job, mem) for pid, mem in info.get("procs", []) if pid in procs]
                    self.gpu_idle[m.name] = not holders or (util is not None and util < self.idle_util)
                    if not holders:
                        j = self._job(IDLE, now); j.energy_j += e; j.idle_j += e; j.used(m)
                        continue
                    weights: dict[str, float] = {}
                    for jk, mem in holders:
                        keys[jk.key] = jk
                        weights[jk.key] = weights.get(jk.key, 0.0) + max(mem, 1)
                    idle = util is not None and util < self.idle_util
                    # one job on the card: all of it. Several: by SM time where the driver gives it, else by memory held.
                    sm: dict[str, float] = {}
                    for pid, v in (info.get("sm") or {}).items():
                        if pid in procs and v > 0:
                            keys[procs[pid].job.key] = procs[pid].job
                            sm[procs[pid].job.key] = sm.get(procs[pid].job.key, 0.0) + v
                    if len(weights) == 1:
                        split, method = weights, "whole_gpu"
                    elif sum(sm.values()) > 0:
                        split, method = sm, "sm_time"
                    else:
                        split, method = weights, "memory_share"
                    tot = sum(split.values())
                    for k in weights.keys() - split.keys():  # held the card but used no SM time: its share is 0 J, by SM time
                        j = self._job(keys[k], now); j.give(m, 0.0, method); j.gpu_devices.add(m.name)
                    for k, w in split.items():
                        j = self._job(keys[k], now)
                        j.give(m, e * w / tot, method)
                        if idle:
                            j.idle_j += e * w / tot
                        j.gpu_devices.add(m.name)
                elif m.kind == "gpu":
                    # no per-process GPU data (older kernels): the job doing most of the CPU work, if it does most of it
                    target = keys[busiest] if busiest and d_ticks[busiest] * 2 >= total else SYSTEM
                    j = self._job(target, now); j.give(m, e, "cpu_guess")
                    if target is not SYSTEM:
                        j.gpu_devices.add(m.name)
                else:
                    if not total:
                        j = self._job(SYSTEM, now); j.energy_j += e; j.used(m)
                        continue
                    for k, t in d_ticks.items():
                        self._job(keys[k], now).give(m, e * t / total, "cpu_time")

            if dt > 0:
                self.job_w = {k: (j.energy_j - before.get(k, (0.0, 0.0))[0]) / dt for k, j in self.jobs.items()}
                self.job_idle_w = {k: (j.idle_j - before.get(k, (0.0, 0.0))[1]) / dt for k, j in self.jobs.items()}
            # sample coverage: every job still running was sampled over this interval, if the interval came on time
            # (a stalled agent counts only up to two intervals) and every meter was read
            if dt > 0 and complete:
                got = min(dt, 2 * self.interval) if self.interval else dt
                for k in {p.job.key for p in procs.values()}:
                    if k in self.jobs:
                        self.jobs[k].covered_s += got
            self.last_step = now
            # jobs not seen for a while have finished
            for k in list(self.jobs):
                j = self.jobs[k]
                if j.key.kind not in ("system", "idle") and now - j.last_seen > self.finish_after:
                    self.finished.append(self.jobs.pop(k))

    def take_finished(self) -> list[Job]:
        with self.lock:
            out, self.finished = self.finished, []
            return out


# ---------------------------------------------------------------- Prometheus and JSON
def _l(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def metrics_text(acc: Accountant, price: float, g_per_kwh: float, host: str, pue: float = 1.0, phases: PhaseClock | None = None) -> str:
    with acc.lock:
        jobs = [j for j in acc.jobs.values()]
        dev = dict(acc.device_j); pw = dict(acc.device_w); util = dict(acc.gpu_util); health = dict(acc.gpu_health)
    h = f'host="{_l(host)}"'
    out = ["# HELP joules_agent_info The agent on this machine.", "# TYPE joules_agent_info gauge",
           f'joules_agent_info{{{h},version="{_l(__version__)}"}} 1',
           "# HELP joules_price_per_kwh Electricity price the agent uses (--price).", "# TYPE joules_price_per_kwh gauge",
           f"joules_price_per_kwh{{{h}}} {price:.6g}",
           "# HELP joules_grid_co2_grams_per_kwh Grid carbon intensity the agent uses.", "# TYPE joules_grid_co2_grams_per_kwh gauge",
           f"joules_grid_co2_grams_per_kwh{{{h}}} {g_per_kwh:.6g}",
           "# HELP joules_pue Facility overhead the agent applies to cost and CO2 (--pue).", "# TYPE joules_pue gauge",
           f"joules_pue{{{h}}} {pue:.6g}",
           "# HELP joules_device_energy_joules_total Energy each device used since the agent started.",
           "# TYPE joules_device_energy_joules_total counter"]
    out += [f'joules_device_energy_joules_total{{host="{_l(host)}",device="{_l(d)}"}} {e:.6g}' for d, e in dev.items()]
    out += ["# HELP joules_device_power_watts Power of each device over the last interval.", "# TYPE joules_device_power_watts gauge"]
    out += [f'joules_device_power_watts{{host="{_l(host)}",device="{_l(d)}"}} {w:.6g}' for d, w in pw.items()]
    out += ["# HELP joules_gpu_utilization_percent GPU busy percentage reported by the driver.", "# TYPE joules_gpu_utilization_percent gauge"]
    out += [f'joules_gpu_utilization_percent{{host="{_l(host)}",device="{_l(d)}"}} {u:.6g}' for d, u in util.items()]
    dl = lambda d: f'host="{_l(host)}",device="{_l(d)}"'
    gauges = [("joules_gpu_sm_clock_mhz", "The GPU's compute clock now.", "sm_clock_mhz"),
              ("joules_gpu_sm_clock_max_mhz", "The GPU's maximum compute clock.", "sm_clock_max_mhz"),
              ("joules_gpu_memory_clock_mhz", "The GPU's memory clock now.", "mem_clock_mhz"),
              ("joules_gpu_temperature_celsius", "GPU temperature.", "temp_c"),
              ("joules_gpu_memory_utilization_percent", "Share of time the GPU's memory controller was busy.", "mem_util_pct"),
              ("joules_gpu_power_limit_watts", "The power limit the GPU enforces.", "power_limit_w")]
    for name, help_, key in gauges:
        rows = [(d, h[key]) for d, h in health.items() if h.get(key) is not None]
        if rows:
            out += [f"# HELP {name} {help_}", f"# TYPE {name} gauge"] + [f"{name}{{{dl(d)}}} {v:.6g}" for d, v in rows]
    rows = [(d, h) for d, h in health.items() if h.get("throttled") is not None]
    if rows:
        out += ["# HELP joules_gpu_throttled 1 while the GPU runs below its maximum clock for a reason other than being idle (power cap, thermal, hardware slowdown).",
                "# TYPE joules_gpu_throttled gauge"]
        out += [f'joules_gpu_throttled{{{dl(d)},reason="{_l(h["throttle_reason"])}"}} {1 if h["throttled"] else 0}' for d, h in rows]
    out += ["# HELP joules_job_energy_joules_total Energy attributed to each running job.", "# TYPE joules_job_energy_joules_total counter"]
    lab = lambda j: f'host="{_l(host)}",kind="{_l(j.key.kind)}",job="{_l(j.key.id)}",name="{_l(j.key.name)}",user="{_l(j.key.user)}",namespace="{_l(j.key.namespace)}"'
    out += [f"joules_job_energy_joules_total{{{lab(j)}}} {j.energy_j:.6g}" for j in jobs]
    out += ["# HELP joules_job_idle_gpu_energy_joules_total GPU energy while the job held a GPU but barely used it (or no job held it).",
            "# TYPE joules_job_idle_gpu_energy_joules_total counter"]
    out += [f"joules_job_idle_gpu_energy_joules_total{{{lab(j)}}} {j.idle_j:.6g}" for j in jobs if j.idle_j]
    out += ["# HELP joules_job_cost Electricity cost attributed to each running job, at the agent's --price.", "# TYPE joules_job_cost gauge"]
    out += [f"joules_job_cost{{{lab(j)}}} {j.energy_j * pue / 3.6e6 * price:.6g}" for j in jobs]
    out += ["# HELP joules_job_co2_grams CO2 attributed to each running job, at the agent's grid intensity.", "# TYPE joules_job_co2_grams gauge"]
    out += [f"joules_job_co2_grams{{{lab(j)}}} {j.energy_j * pue / 3.6e6 * g_per_kwh:.6g}" for j in jobs]
    tot = phases.totals() if phases else {}
    if tot:
        pl = lambda n: f'host="{_l(host)}",phase="{_l(n)}"'
        out += ["# HELP joules_phase_energy_joules_total Energy of all ended phases with this name (all devices).",
                "# TYPE joules_phase_energy_joules_total counter"]
        out += [f"joules_phase_energy_joules_total{{{pl(n)}}} {v[1]:.6g}" for n, v in tot.items()]
        out += ["# HELP joules_phase_duration_seconds_total Time spent in ended phases with this name.",
                "# TYPE joules_phase_duration_seconds_total counter"]
        out += [f"joules_phase_duration_seconds_total{{{pl(n)}}} {v[2]:.6g}" for n, v in tot.items()]
        out += ["# HELP joules_phases_total Ended phases with this name.", "# TYPE joules_phases_total counter"]
        out += [f"joules_phases_total{{{pl(n)}}} {v[0]}" for n, v in tot.items()]
    return "\n".join(out) + "\n"


class LocalHTTPServer(ThreadingHTTPServer):
    """http.server without its reverse DNS lookup of the address it listens on (socket.getfqdn), which hangs for
    half a minute or more on a Mac or VM whose DNS is slow; nothing here uses the name it finds."""
    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]


class AgentServer(LocalHTTPServer):

    def __init__(self, addr, agent: "Agent"):
        super().__init__(addr, _Handler)
        self.agent = agent


class _Handler(BaseHTTPRequestHandler):
    server: AgentServer

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: str, ctype: str) -> None:
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        a = self.server.agent
        path = self.path.split("?")[0]
        if path == "/metrics":
            return self._send(200, metrics_text(a.acc, a.price, a.g_per_kwh, a.host, a.pue, a.phases), "text/plain; version=0.0.4")
        if path == "/jobs":
            with a.acc.lock:
                jobs = sorted((j.as_dict(a.price, a.g_per_kwh, a.pue) for j in a.acc.jobs.values()), key=lambda d: -d["energy_j"])
            return self._send(200, json.dumps({"host": a.host, "jobs": jobs, "finished": a.recent_finished[-50:]}, indent=1), "application/json")
        if path == "/health":
            return self._send(200, json.dumps({"ok": True, "joules": __version__, "devices": [m.name for m in a.meters]}), "application/json")
        if path == "/phases":
            return self._send(200, json.dumps({"host": a.host, **a.phases.summary(), "recent": a.phases.recent()}, indent=1), "application/json")
        if path == "/energy":
            snap = a.phases.snapshot()
            return self._send(200, json.dumps({"host": a.host, "at": snap["wall"], "monotonic_s": snap["t"], "devices_j": snap["j"],
                                               "total_j": sum(snap["j"].values())}), "application/json")
        return self._send(404, json.dumps({"error": "joules agent: GET /metrics, /jobs, /health, /energy, /phases; POST /phase/start, /phase/end, /phase/reset"}), "application/json")

    def do_POST(self):
        """Phase marks. Only from a program on this machine or your network, never from a web page: a request a
        browser sends carries an Origin header and is refused, and the body must be JSON (which a page cannot
        send to another site without asking first)."""
        a = self.server.agent
        path = self.path.split("?")[0]
        if path not in ("/phase/start", "/phase/end", "/phase/reset"):
            return self._send(404, json.dumps({"error": "POST /phase/start, /phase/end or /phase/reset"}), "application/json")
        if self.headers.get("Origin"):
            return self._send(403, json.dumps({"error": "phase marks are not accepted from web pages"}), "application/json")
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > 4096:
            return self._send(413, json.dumps({"error": "the body is at most 4 KB"}), "application/json")
        raw = self.rfile.read(n) if n else b""
        if raw and "json" not in (self.headers.get("Content-Type") or ""):
            return self._send(415, json.dumps({"error": "send the body as application/json"}), "application/json")
        try:
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._send(400, json.dumps({"error": "the body is a JSON object, e.g. {\"name\": \"update\"}"}), "application/json")
        try:
            if path == "/phase/start":
                out = a.phases.begin(str(body.get("name") or ""))
            elif path == "/phase/end":
                out = a.phases.end(pid=body.get("id") and str(body["id"]), name=body.get("name") and str(body["name"]))
            else:
                a.phases.reset()
                out = {"ok": True}
        except PhaseError as e:
            return self._send(400, json.dumps({"error": str(e)}), "application/json")
        return self._send(200, json.dumps(out), "application/json")


# ---------------------------------------------------------------- the loop
class Agent:
    def __init__(self, meters: list[Meter], interval: float = 1.0, idle_util: float = 5.0, finish_after: float = 30.0,
                 price: float = 0.10, g_per_kwh: float = 370.0, pue: float = 1.0, on_finish=None, host: str | None = None):
        import socket
        self.meters, self.interval = meters, interval
        self.acc = Accountant(idle_util, finish_after, interval)
        self.price, self.g_per_kwh, self.pue = price, g_per_kwh, pue
        self._jobs_cache: dict = {}
        self.drm = DrmUsage() if any(getattr(m, "pci", None) for m in meters) else None
        self.on_finish = on_finish
        self.host = host or socket.gethostname()
        self.recent_finished: list[dict] = []
        self._stop = threading.Event()
        self._last: dict[str, float] = {}
        self.gpu_detail = True         # each heartbeat carries a reading per GPU; off if the server does not take them
        self.serve = None              # an inference server's counters (serving.ServeMetrics): tokens per second on Live
        self.lock = threading.Lock()   # held while the meters are read, by the loop and by phase marks
        self.phases = PhaseClock(meters, self.lock)
        self._sm_since: dict[str, int] = {}             # GPU -> the newest per-process utilisation sample read
        self._acct_prev: dict[str, dict[int, float]] = {}  # GPU -> each process's SM milliseconds at the last reading

    def _gpu_info(self, procs: dict, now: float) -> dict:
        info = {}
        if self.drm is not None:
            by_card = self.drm.step(list(procs), now)
            for m in self.meters:
                if getattr(m, "pci", None):
                    info[m.name] = {"busy": by_card.get(m.pci, {})}
        for m in self.meters:
            nv, h = getattr(m, "nvml", None), getattr(m, "h", None)
            if m.kind == "gpu" and nv is not None and hasattr(nv, "processes"):
                try:
                    procs_on = nv.processes(h)
                    info[m.name] = {"procs": procs_on, "util": nv.utilization(h),
                                    "health": m.health_now() if hasattr(m, "health_now") else None}
                except (OSError, ValueError):
                    continue
                sm = self._sm_time(m, nv, h, [pid for pid, _ in procs_on])
                if sm is not None:
                    info[m.name]["sm"] = sm
        return info

    def _sm_time(self, m, nv, h, pids) -> dict | None:
        """Each process's share of SM time on this GPU since the last reading, as {pid: weight}, or None when the driver
        cannot say. First the driver's per-process utilisation samples; else accounting mode's per-process totals."""
        if hasattr(nv, "process_sm"):
            try:
                r = nv.process_sm(h, self._sm_since.get(m.name, 0))
            except (OSError, ValueError):
                r = None
            if r is not None:
                self._sm_since[m.name] = r[1]
                return r[0]
        if hasattr(nv, "accounting_sm_ms"):
            try:
                totals = nv.accounting_sm_ms(h, pids)
            except (OSError, ValueError):
                totals = None
            if totals is not None:
                prev = self._acct_prev.setdefault(m.name, {})
                out = {pid: max(0.0, ms - prev.get(pid, ms)) for pid, ms in totals.items()}
                self._acct_prev[m.name] = totals
                return out
        return None

    def tick(self, now: float, dt: float, wall: float | None = None) -> None:
        """now: monotonic seconds (for the meters); wall: Unix seconds (for the jobs' start and end)."""
        wall = time.time() if wall is None else wall
        deltas = []
        complete = True
        with self.lock:
            for m in self.meters:
                try:
                    m.poll(now)
                except (OSError, ValueError):
                    complete = False
                    continue
                prev = self._last.get(m.name, m.energy_j)
                deltas.append((m, max(0.0, m.energy_j - prev)))
                self._last[m.name] = m.energy_j
        procs = scan_processes(self._jobs_cache)
        self.acc.step(wall, dt, procs, deltas, self._gpu_info(procs, now), complete)
        for j in self.acc.take_finished():
            d = j.as_dict(self.price, self.g_per_kwh, self.pue)
            self.recent_finished.append(d)
            self.recent_finished = self.recent_finished[-200:]
            if self.on_finish and j.energy_j > 0:
                self.on_finish(j, d)

    def start(self, now: float, wall: float | None = None) -> None:
        with self.lock:
            for m in self.meters:
                m.start(now)
                self._last[m.name] = m.energy_j
        procs = scan_processes(self._jobs_cache)
        self._gpu_info(procs, now)  # first GPU-time reading
        self.acc.step(time.time() if wall is None else wall, 0, procs, [], {})  # first CPU-time reading

    def run(self, seconds: float | None = None) -> None:
        start = last = time.monotonic()
        self.start(start)
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            self.tick(now, now - last)
            last = now
            if seconds is not None and now - start >= seconds:
                break
        for m in self.meters:
            try:
                m.stop(time.monotonic())
            except (OSError, ValueError):
                pass

    def stop(self) -> None:
        self._stop.set()

    def live_beat(self, top: int = 50) -> dict:
        """What the Live view shows for this node right now: its power over the last interval (with the facility
        overhead), how much of it went to GPUs nobody used, and the jobs drawing power. Names and users only."""
        pue = self.pue
        with self.acc.lock:
            watts = sum(self.acc.device_w.values()) * pue
            idle_w = sum(self.acc.job_idle_w.values()) * pue
            gpus = [m.name for m in self.meters if m.kind == "gpu"]
            idle = sum(1 for g in gpus if self.acc.gpu_idle.get(g))
            jobs = []
            for k, j in self.acc.jobs.items():
                w = self.acc.job_w.get(k, 0.0)
                if j.key.kind == "idle" or w <= 0:
                    continue
                name = j.key.name + (f" ({j.key.namespace})" if j.key.namespace else "")
                jobs.append({"name": name[:120] or j.key.id[:120], "user": j.key.user[:60], "kind": j.key.kind,
                             "gpus": len(j.gpu_devices), "watts": round(w * pue, 2),
                             "idle_watts": round(min(w, max(0.0, self.acc.job_idle_w.get(k, 0.0))) * pue, 2)})
            health = [self.acc.gpu_health.get(g) for g in gpus]
            detail = self._gpu_list(gpus) if self.gpu_detail and gpus else None
        jobs.sort(key=lambda x: -x["watts"])
        beat = {"node": node_name(self.host), "watts": round(watts, 2), "idle_watts": round(min(watts, max(0.0, idle_w)), 2),
                "gpus": len(gpus), "idle_gpus": idle, "price_per_kwh": self.price if 0 <= self.price < 1e3 else None,
                "g_per_kwh": self.g_per_kwh if 0 <= self.g_per_kwh < 2000 else None,
                "jobs": jobs[:top], "version": __version__}
        health = [h for h in health if h]
        if health:  # the GPUs' health, from the latest reading of each card (NVIDIA only; an older driver gives none)
            throttled = [h for h in health if h.get("throttled")]
            beat["throttled_gpus"] = len(throttled)
            temps = [h["temp_c"] for h in health if h.get("temp_c") is not None]
            if temps:
                beat["temp_c_max"] = max(temps)
            reasons = [h["throttle_reason"] for h in throttled if h.get("throttle_reason")]
            if reasons:
                beat["throttle_reason"] = max(set(reasons), key=reasons.count)[:20]
        if detail:
            beat["gpu_list"] = detail
        if self.serve is not None:  # what a vLLM or SGLang server here is serving: tokens per second, requests
            s = self.serve.read()
            if s:
                beat["serve"] = s
        return beat

    def _gpu_list(self, gpus: list[str]) -> list[dict]:
        """One entry per GPU for the Live page's GPU grid: its number and model, its own power (device power, without
        --pue), how busy it and its memory controller are, the clock as a share of the maximum, temperature, power
        limit, whether it is throttled and why, whether it is idle, and the job using it most. Numbers and names only.
        Called with self.acc.lock held."""
        acc, out = self.acc, []
        for pos, g in enumerate(gpus[:64]):
            m = re.match(r"^GPU (\d+) (.+)$", g)
            idx, model = (int(m.group(1)), m.group(2)) if m else (pos, g)
            e: dict = {"i": min(idx, 4095), "name": model[:80], "watts": round(max(0.0, acc.device_w.get(g, 0.0)), 1)}
            h = acc.gpu_health.get(g) or {}
            util = h.get("gpu_util_pct", acc.gpu_util.get(g))
            if util is not None:
                e["util"] = round(min(100.0, max(0.0, float(util))), 1)
            if h.get("mem_util_pct") is not None:
                e["mem_util"] = round(min(100.0, max(0.0, float(h["mem_util_pct"]))), 1)
            if h.get("sm_clock_mhz") is not None and h.get("sm_clock_max_mhz"):
                e["clock_pct"] = round(min(200.0, h["sm_clock_mhz"] / h["sm_clock_max_mhz"] * 100), 1)
            if h.get("temp_c") is not None and -50 < h["temp_c"] < 200:
                e["temp_c"] = round(float(h["temp_c"]), 1)
            if h.get("power_limit_w"):
                e["power_limit_w"] = round(min(1e5 - 1, float(h["power_limit_w"])), 1)
            if h.get("throttled") is not None:
                e["throttled"] = bool(h["throttled"])
                if h["throttled"] and h.get("throttle_reason"):
                    e["reason"] = str(h["throttle_reason"])[:20]
            if g in acc.gpu_idle:
                e["idle"] = bool(acc.gpu_idle[g])
            users = [(acc.job_w.get(k, 0.0), j) for k, j in acc.jobs.items() if g in j.gpu_devices and j.key.kind != "idle"]
            if users:
                j = max(users, key=lambda x: x[0])[1]
                name = j.key.name + (f" ({j.key.namespace})" if j.key.namespace else "")
                e["job"] = (name or j.key.id)[:120]
            out.append(e)
        return out


def node_name(host: str) -> str:
    """The host name as the Live view accepts it: letters, digits, dot, dash or underscore, up to 64 characters."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", host or "").strip("-")[:64] or "node"


def track_record(j: Job, d: dict, project: str, device: str, device_count: int, price: float, g_per_kwh: float) -> dict:
    """The Tracking-page record for a finished job: what ran, where, and its energy. No arguments or paths."""
    import datetime as dt
    name = f"{j.key.kind} · {j.key.name}" + (f" · {j.key.namespace}" if j.key.namespace else "") + (f" · {j.key.user}" if j.key.user else "")
    return {"project": project, "kind": "run", "label": name[:200],
            "started_at": dt.datetime.fromtimestamp(j.first_seen, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration_s": max(d["duration_s"], 1e-3), "energy_j": d["facility_j"], "co2_g": d["co2_g"], "cost": d["cost"],
            "price_per_kwh": price, "g_per_kwh": g_per_kwh, "device": device[:120], "device_count": max(1, device_count),
            "idle_subtracted": False, "simulated": False, "joules_version": __version__,
            "sources": "+".join(sorted(j.kinds)) or "cpu",
            "attribution": j.attribution, "coverage": round(j.coverage, 4)}
