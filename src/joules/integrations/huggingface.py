"""Training energy in Hugging Face transformers' Trainer, from the hardware's own counters.

    from joules.integrations.huggingface import attach

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    energy = attach(trainer)          # measures from train start to train end
    trainer.train()
    print(energy.result)              # energy_j, kwh, avg_w, j_per_step, cost, co2_g, phases per epoch, ...

attach() puts the callback first, so the logs it adds (energy/joules, energy/kwh, energy/avg_watts,
energy/joules_per_step) reach W&B, TensorBoard and every other `report_to` logger. Passed as
Trainer(callbacks=[JoulesCallback()]) it measures the same, but the Trainer calls it after those loggers, so the
energy reaches only the printed line and .result, not their charts.

Only the local main process measures (state.is_local_process_zero), so each machine's GPUs are counted once. The
Trainer has no hook for a run stopped by an exception: the session then ends with the process. With nothing to
measure (`joules doctor` says why) the callback prints one line and training goes on unchanged."""
from __future__ import annotations

try:
    from transformers import TrainerCallback
except ImportError as e:  # pragma: no cover - exercised only without transformers
    raise ImportError("joules.integrations.huggingface needs transformers: pip install transformers") from e

from ._core import EnergyTracker


class JoulesCallback(TrainerCallback):
    def __init__(self, gpus=None, track=None, name=None, price=0.10, grid="us", ci=None, pue=1.0, log_prefix="energy/"):
        """gpus: GPU indices to measure (default: CUDA_VISIBLE_DEVICES or all). track: a WattOps project to save
        the run to when training ends (`joules login` once). name: what to call it there. price per kWh, grid (or
        ci in g CO2/kWh) and pue give its cost and CO2, as in `joules run`. log_prefix: the prefix of the log keys."""
        self._t = EnergyTracker(gpus, track, name, price, grid, ci, pue, log_prefix)

    @property
    def result(self) -> dict | None:
        return self._t.result

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(state, "is_local_process_zero", True):
            self._t.start()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t.begin_epoch()

    def on_epoch_end(self, args, state, control, **kwargs):
        self._t.end_epoch()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._t.active:
            logs.update(self._t.metrics(getattr(state, "global_step", 0)))

    def on_train_end(self, args, state, control, **kwargs):
        self._t.finish(getattr(state, "global_step", 0))


def attach(trainer, **kw) -> JoulesCallback:
    """Make a JoulesCallback and put it first in the trainer's callbacks, so its log keys reach every logger."""
    cb = JoulesCallback(**kw)
    handler = getattr(trainer, "callback_handler", None)
    if handler is not None and isinstance(getattr(handler, "callbacks", None), list):
        handler.callbacks.insert(0, cb)
    else:
        trainer.add_callback(cb)
    return cb
