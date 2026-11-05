"""Training energy in PyTorch Lightning, from the hardware's own counters.

    from joules.integrations.lightning import JoulesCallback

    energy = JoulesCallback()
    trainer = L.Trainer(max_epochs=3, callbacks=[energy])
    trainer.fit(model, data)
    print(energy.result)              # energy_j, kwh, avg_w, j_per_step, cost, co2_g, phases per epoch, ...

Every trainer.log_every_n_steps steps and at each epoch's end it logs energy/joules, energy/kwh,
energy/avg_watts and energy/joules_per_step through pl_module.log, so every logger (W&B, TensorBoard, CSV) shows
them next to the loss. Works with `lightning` and with the older `pytorch_lightning` package.

The local rank 0 process of each machine measures that machine's GPUs, so they are counted once; the logged values
come from global rank 0, so on several machines they are the first machine's, and each machine's callback has its
own .result. A run stopped by an exception still ends its measurement and prints its line. With nothing to measure
(`joules doctor` says why) the callback prints one line and training goes on unchanged."""
from __future__ import annotations

try:
    from lightning.pytorch import Callback
except ImportError:
    try:
        from pytorch_lightning import Callback
    except ImportError as e:  # pragma: no cover - exercised only without Lightning
        raise ImportError("joules.integrations.lightning needs Lightning: pip install lightning") from e

from ._core import EnergyTracker


class JoulesCallback(Callback):
    def __init__(self, gpus=None, track=None, name=None, price=0.10, grid="us", ci=None, pue=1.0, log_prefix="energy/"):
        """gpus: GPU indices to measure (default: CUDA_VISIBLE_DEVICES or all). track: a WattOps project to save
        the run to when training ends (`joules login` once). name: what to call it there. price per kWh, grid (or
        ci in g CO2/kWh) and pue give its cost and CO2, as in `joules run`. log_prefix: the prefix of the log keys."""
        super().__init__()
        self._t = EnergyTracker(gpus, track, name, price, grid, ci, pue, log_prefix)

    @property
    def result(self) -> dict | None:
        return self._t.result

    def _log(self, trainer, pl_module, on_step: bool) -> None:
        if not self._t.active or getattr(trainer, "global_rank", 0) != 0:
            return
        for k, v in self._t.metrics(getattr(trainer, "global_step", 0)).items():
            pl_module.log(k, float(v), on_step=on_step, on_epoch=not on_step, rank_zero_only=True)

    def on_train_start(self, trainer, pl_module):
        if getattr(trainer, "local_rank", 0) == 0:
            self._t.start()

    def on_train_epoch_start(self, trainer, pl_module):
        self._t.begin_epoch()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        every = getattr(trainer, "log_every_n_steps", 50) or 50
        if self._t.active and getattr(trainer, "global_step", 0) % every == 0:
            self._log(trainer, pl_module, on_step=True)

    def on_train_epoch_end(self, trainer, pl_module):
        self._t.end_epoch()
        self._log(trainer, pl_module, on_step=False)

    def on_train_end(self, trainer, pl_module):
        self._t.finish(getattr(trainer, "global_step", 0))

    def on_exception(self, trainer, pl_module, exception):
        self._t.finish(getattr(trainer, "global_step", 0))
