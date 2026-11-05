"""Polls every meter on a background thread while a workload runs."""
from __future__ import annotations

import threading
import time

from .meters import Meter


class Sampler:
    def __init__(self, meters: list[Meter], interval: float = 0.1, lock: threading.Lock | None = None):
        self.meters = meters
        self.interval = interval
        self.lock = lock or threading.Lock()  # shared with a PhaseClock, which reads the meters between polls
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t0 = self.t1 = 0.0

    def __enter__(self) -> "Sampler":
        self.t0 = time.monotonic()
        for m in self.meters:
            m.start(self.t0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            with self.lock:
                now = time.monotonic()
                for m in self.meters:
                    try:
                        m.poll(now)
                    except (OSError, ValueError):
                        pass  # a transient read error loses one interval, not the run

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        with self.lock:
            self.t1 = time.monotonic()
            for m in self.meters:
                m.stop(self.t1)

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


def idle_baseline(meters: list[Meter], seconds: float, interval: float = 0.1) -> dict[str, float]:
    """Average watts per meter while nothing of ours runs."""
    if seconds <= 0:
        return {}
    with Sampler(meters, interval):
        time.sleep(seconds)
    return {m.name: m.avg_w for m in meters}
