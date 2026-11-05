"""What the Hugging Face and Lightning callbacks share: one joules.Session over the whole of training, a phase
per epoch, the numbers for the logs and the line printed at the end. Nothing in here may break training: every
failure becomes one warning and the callback stops measuring."""
from __future__ import annotations

import sys
import time


class EnergyTracker:
    def __init__(self, gpus=None, track=None, name=None, price=0.10, grid="us", ci=None, pue=1.0, log_prefix="energy/"):
        self.kw = dict(gpus=gpus, track=track, name=name, price=price, grid=grid, ci=ci, pue=pue)
        self.price, self.pue, self.prefix = price, pue, log_prefix
        self.session = None
        self.result: dict | None = None
        self.epochs = 0
        self._epoch = None   # the open epoch phase's id
        self._t0 = None
        self._warned = False
        self._finished = False

    def _warn(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"joules: {msg}", file=sys.stderr)

    @property
    def active(self) -> bool:
        return self.session is not None

    def start(self) -> None:
        if self.session is not None or self._finished:
            return
        try:
            from ..api import Session
            self.session = Session(**self.kw)
            self._t0 = time.monotonic()
        except Exception as e:  # no meters here, a bad argument, anything: training goes on unmeasured
            self.session = None
            self._warn(f"not measuring this training run ({str(e).rstrip('.')}). `joules doctor` says what this machine can measure.")

    def begin_epoch(self) -> None:
        if not self.session:
            return
        try:
            self.end_epoch()
            self._epoch = self.session.begin("epoch")
        except Exception as e:
            self._fail(e)

    def end_epoch(self) -> None:
        if not self.session or self._epoch is None:
            return
        try:
            self.session.end(self._epoch)
            self.epochs += 1
        except Exception as e:
            self._fail(e)
        finally:
            self._epoch = None

    def _fail(self, e: Exception) -> None:
        self._warn(f"stopped measuring ({e}); training goes on.")
        s, self.session = self.session, None
        try:
            s.close()
        except Exception:
            pass

    def _so_far(self) -> tuple[float, float]:
        j = self.session.snapshot()["total_j"]
        return j, max(time.monotonic() - self._t0, 1e-9)

    def metrics(self, steps: int) -> dict:
        """The numbers for the logs: energy so far (device energy, from the meters), kWh, average watts, J a step."""
        if not self.session:
            return {}
        try:
            j, secs = self._so_far()
        except Exception as e:
            self._fail(e)
            return {}
        p = self.prefix
        out = {f"{p}joules": j, f"{p}kwh": j / 3.6e6, f"{p}avg_watts": j / secs}
        if steps:
            out[f"{p}joules_per_step"] = j / steps
        return out

    def finish(self, steps: int) -> dict | None:
        """Close the last epoch, keep callback.result, save to WattOps when track= was given, close the session and
        print one line, the only one a run prints. Runs once; later calls return the same result."""
        if self._finished:
            return self.result
        self._finished = True
        if not self.session:
            return None
        s, self.session = self.session, None
        saved = None
        try:
            if self._epoch is not None:
                s.end(self._epoch)
                self.epochs += 1
                self._epoch = None
            j = s.snapshot()["total_j"]
            secs = max(time.monotonic() - self._t0, 1e-9)
            fac_kwh = j * self.pue / 3.6e6
            whole = s.whole()
            self.result = {"energy_j": j, "kwh": j / 3.6e6, "duration_s": secs, "avg_w": j / secs,
                           "j_per_step": j / steps if steps else None, "steps": steps, "epochs": self.epochs,
                           "cost": fac_kwh * self.price, "price_per_kwh": self.price, "co2_g": fac_kwh * s.g_per_kwh,
                           "pue": self.pue, "devices": s.devices, "gpu": whole.get("gpu") or {},
                           "simulated": s.discovery.simulated, "phases": s.summary(), "run_id": None}
            if s.track:  # saved here, not in close(), so its outcome joins the one line instead of printing its own
                try:
                    reply = s.upload()
                    self.result["run_id"] = (reply or {}).get("id")
                    saved = f"; saved to project {s.track}"
                except Exception as e:
                    s.saved = {}  # close() then does not try again, or print
                    saved = f"; not saved to WattOps ({e})"
            line = self._line(self.result) + (saved or "")
        except Exception as e:
            line = f"joules: could not read the final energy ({e})."
        try:
            s.close()
        except Exception:
            pass  # the meters are being released; nothing left that training needs
        print(line, file=sys.stderr)
        return self.result

    @staticmethod
    def _line(r: dict) -> str:
        try:
            from ..cli import fmt_g, fmt_j, fmt_s, fmt_w, fmt_wh
        except Exception:  # a plain fallback, should the CLI's helpers ever fail to import
            fmt_j = lambda x: f"{x:.3g} J"; fmt_wh = lambda x: f"{x / 3600:.3g} Wh"; fmt_s = lambda x: f"{x:.0f} s"
            fmt_w = lambda x: f"{x:.3g}"; fmt_g = lambda x: f"{x:.3g}"
        per = f", {fmt_j(r['j_per_step'])} per step" if r["j_per_step"] else ""
        sim = " (SIMULATED meters: not a measurement)" if r["simulated"] else ""
        return (f"joules: training used {fmt_j(r['energy_j'])} = {fmt_wh(r['energy_j'])} over {fmt_s(r['duration_s'])}, "
                f"{fmt_w(r['avg_w'])} W average{per}; cost {r['cost']:.3g} at {r['price_per_kwh']:g}/kWh, "
                f"CO₂ {fmt_g(r['co2_g'])} g{sim}")
