"""joules inside the training loops people already use. Each module needs its framework; joules itself does not.

    from joules.integrations.huggingface import JoulesCallback     # transformers.Trainer
    from joules.integrations.lightning import JoulesCallback       # lightning / pytorch_lightning

Both measure the GPU (and CPU and DRAM) energy of training from the hardware's own counters, add it to the logs
(energy/joules, energy/kwh, energy/avg_watts, energy/joules_per_step) so W&B, TensorBoard or any other logger
shows it next to the loss, print one line when training ends and keep the numbers in callback.result.
Only the local main process measures, so a machine's GPUs are counted once; if this machine has nothing to
measure, the callback says so once and training goes on unchanged."""
