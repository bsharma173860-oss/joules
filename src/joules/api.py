"""joules as a library: measure phases of your own program, in process, with no agent and no network.

    import joules

    with joules.Session(gpus=[0]) as s:          # finds the meters once and keeps them running
        for batch in batches:
            with s.phase("incremental update") as p:
                runtime.update(batch)
            print(p.energy_j, p.duration_s, p.gpu)   # filled in when the block ends
        with s.phase("full recompute"):
            runtime.recompute()
        print(s.summary())                          # per name: count, median energy and time, spread
        s.save("phases.json")                       # every phase, for your own analysis

    with joules.measure("one solve") as p:          # one phase, when that is all you need
        solve()

Each mark reads every meter at once, so a phase's edges are as sharp as the counter allows (RAPL about 1 ms,
NVIDIA's energy counter a few to 100 ms). Phases may overlap and nest. GPU health (clocks, busy %, throttling)
is read twice a second, so a phase shorter than that has energy and time but no health. Nothing is sent
anywhere unless you ask: Session(track="my-project") saves the whole session to your WattOps account when it
ends (its energy, time, cost, CO2, GPU health and each phase's count, median and spread, never code or data), where
its PDF button on Tracking opens the report (Pro, Team and Fleet). It uses the key from `joules login` or $LANDAUER_API_KEY. On a machine with no energy
counters there is nothing to measure and Session raises NoMeters, unless estimate=True, which uses CPU time of
child processes (not useful in process)."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time

from . import __version__
from .meters import discover
from .phases import PhaseClock, PhaseError, summarize_values  # noqa: F401  (re-exported)
from .sampler import Sampler


class NoMeters(RuntimeError):
    """No energy counters on this machine (joules doctor says why)."""


class Phase:
    """One phase's result. The numbers are filled in when the phase ends."""

    def __init__(self, name: str):
        self.name = name
        self.result: dict | None = None

    @property
    def energy_j(self) -> float | None:
        return self.result and self.result["energy_j"]

    @property
    def duration_s(self) -> float | None:
        return self.result and self.result["duration_s"]

    @property
    def avg_w(self) -> float | None:
        return self.result and self.result["avg_w"]

    @property
    def devices(self) -> dict | None:
        """joules per device, e.g. {"GPU 0 NVIDIA GeForce RTX 3070 Ti": 81.2, "CPU package 0": 12.4}"""
        return self.result and self.result["devices"]

    @property
    def gpu(self) -> dict | None:
        """Per NVIDIA GPU over the phase: clock (and % of max), busy and memory-controller %, throttled % and why,
        temperature at start and end, power limit. None for a phase too short to have a health reading."""
        return self.result and self.result.get("gpu")

    def __repr__(self) -> str:
        if not self.result:
            return f"<joules phase {self.name!r} running>"
        return f"<joules phase {self.name!r} {self.energy_j:.4g} J in {self.duration_s:.4g} s>"


def _default_label() -> str:
    """What ran, for the Tracking page: the script's file name, or "notebook" inside Jupyter, Colab or Kaggle (whose
    kernels start as ipykernel_launcher.py or colab_kernel_launcher.py, never the notebook's own name)."""
    a0 = os.path.basename(sys.argv[0] or "") if sys.argv else ""
    if "ipykernel" in sys.modules or "kernel_launcher" in a0:
        return "notebook"
    if not a0 or a0 in ("-c", "python", "python3"):
        return "python"
    return f"python {a0}"


class Session:
    def __init__(self, gpus: list[int] | None = None, cpu: bool = True, interval: float = 0.1, estimate: bool = False,
                 track: str | None = None, name: str | None = None, price: float = 0.10, grid: str = "us",
                 ci: float | None = None, pue: float = 1.0, key: str | None = None, api_url: str | None = None):
        """track: a project name; the session is saved there when it ends (see the module's docstring).
        name: what to call it there (default: the script's name, or "notebook"). price per kWh, grid (or ci in
        g CO2/kWh) and pue give its cost and CO2, as in `joules run`."""
        from .model import GRIDS
        if track is not None:
            from .share import PROJECT_RE
            if not PROJECT_RE.match(track):
                raise ValueError("track: a project name of letters, digits, dot, dash or underscore (1 to 60 characters)")
        if ci is None and grid not in GRIDS:
            raise ValueError(f"grid: one of {', '.join(sorted(GRIDS))}, or give ci in g CO2/kWh")
        self.track, self.name, self.price, self.pue = track, name, price, pue
        self.g_per_kwh = ci if ci is not None else GRIDS[grid]
        self._key, self._api_url = key, api_url
        self.saved: dict | None = None  # the server's reply once the session is saved (id, page)
        self.discovery = discover(gpus, cpu=cpu, estimate=estimate)
        if not self.discovery.meters:
            self.discovery.close()
            raise NoMeters("; ".join(self.discovery.notes) or "no energy counters found")
        self.meters = self.discovery.meters
        self.notes = list(self.discovery.notes)
        self._sampler = Sampler(self.meters, interval)
        self._sampler.__enter__()
        self.clock = PhaseClock(self.meters, self._sampler.lock)
        self._start = self.clock.snapshot()  # the whole session is measured like one phase, outside the phase lists
        self._started_at = time.time()
        self._closed = False

    @property
    def devices(self) -> list[dict]:
        return [{"name": m.name, "kind": m.kind, "method": m.method} for m in self.meters]

    def begin(self, name: str) -> str:
        """Start a phase and return its id, for when a with-block does not fit (start and end in different places)."""
        return self.clock.begin(name)["id"]

    def end(self, phase_id: str | None = None, name: str | None = None) -> dict:
        return self.clock.end(pid=phase_id, name=name)

    @contextlib.contextmanager
    def phase(self, name: str):
        p = Phase(name)
        pid = self.clock.begin(name)["id"]
        try:
            yield p
        finally:
            p.result = self.clock.end(pid=pid)

    def snapshot(self) -> dict:
        """Every device's energy since the session started, read now."""
        s = self.clock.snapshot()
        return {"at": s["wall"], "devices_j": s["j"], "total_j": sum(s["j"].values())}

    def summary(self) -> dict:
        return self.clock.summary()["phases"]

    def results(self) -> list[dict]:
        with self.clock._state:
            return [r for rs in self.clock._done.values() for r in rs]

    def report(self) -> dict:
        return {"schema": "joules/1", "tool": f"joules {__version__}", "kind": "phases",
                "host": {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
                "devices": self.devices, "simulated": self.discovery.simulated, "notes": self.notes,
                "summary": self.summary(), "phases": sorted(self.results(), key=lambda r: r["started_at"])}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report(), f, indent=2)

    def whole(self) -> dict:
        """The session so far as one stretch: energy per device, time, average power and, on NVIDIA, GPU health."""
        return self.clock._result({"id": "session", "name": "session", "snap": self._start}, self.clock.snapshot())

    def record(self, project: str | None = None, label: str | None = None) -> dict:
        """What Session(track=…) sends: one Tracking record for the whole session, with each phase's summary."""
        import datetime as dt
        from .share import track_record, PROJECT_RE
        w = self.whole()
        kinds = {m.name: m.kind for m in self.meters}
        facility = w["energy_j"] * self.pue
        kwh = facility / 3.6e6
        devices = []
        for name, j in w["devices"].items():
            d = {"name": name, "kind": kinds.get(name, "cpu"), "energy_j": j}
            h = (w.get("gpu") or {}).get(name)
            if h:
                h = dict(h)
                temps = [t for t in (h.get("temp_c_start"), h.get("temp_c_end")) if t is not None]
                if temps:
                    h["temp_c_max"] = max(temps)
                d["gpu"] = h
            devices.append(d)
        report = {"kind": "run", "command": [], "duration_s": w["duration_s"], "simulated": self.discovery.simulated,
                  "started_at": dt.datetime.fromtimestamp(self._started_at, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                  "devices": devices,
                  "totals": {"facility_j": facility, "co2": {"grams": kwh * self.g_per_kwh, "g_per_kwh": self.g_per_kwh},
                             "cost": {"amount": kwh * self.price, "price_per_kwh": self.price}}}
        project = project or self.track or "default"
        if not PROJECT_RE.match(project):
            raise ValueError("project names use letters, digits, dot, dash or underscore (1 to 60 characters)")
        r = track_record(report, project, (label or self.name or _default_label())[:200])
        phases = []
        for n, p in sorted(self.summary().items(), key=lambda kv: -kv[1]["energy_j_total"])[:64]:
            phases.append({"name": n, "count": p["count"], "energy_j": p["energy_j_total"], "duration_s": p["duration_s_total"],
                           "median_j": p["energy_j"]["median"], "median_s": p["duration_s"]["median"], "avg_w": p["avg_w"],
                           "spread_pct": p["energy_j"]["spread_pct"]})
        if phases:
            r["phases"] = phases
        return r

    def upload(self, project: str | None = None, label: str | None = None) -> dict:
        """Save the session so far to your WattOps account now (Session(track=…) does this when it ends). Returns
        the server's reply: the run's id and its page."""
        from . import share
        key = share.api_key(self._key)
        if not key:
            raise RuntimeError(f"no API key: run `joules login` once, or set ${share.KEY_ENV} "
                               f"(create one at {share.api_url(self._api_url)}/#/settings/keys)")
        r = self.record(project, label)
        self.saved = share.send_track(r, key, self._api_url) or {}
        return self.saved

    def close(self) -> None:
        if not self._closed:
            if self.track and self.saved is None:  # saved once; a failed upload never breaks the program measured
                try:
                    reply = self.upload()
                    from . import share
                    print(f"joules: session saved to project {self.track}: {share.api_url(self._api_url)}/#/track"
                          f"?run={reply.get('id', '')}", file=sys.stderr)
                except (ValueError, RuntimeError, OSError) as e:
                    print(f"joules: the session was not saved to WattOps: {e}. Its phases are still here: s.save(\"phases.json\").",
                          file=sys.stderr)
            self._closed = True
            self._sampler.__exit__(None, None, None)
            self.discovery.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextlib.contextmanager
def measure(name: str = "measurement", **kw):
    """One phase in its own session: with joules.measure("solve") as p: ...; then p.energy_j."""
    with Session(**kw) as s, s.phase(name) as p:
        yield p
