"""The Hugging Face and Lightning callbacks, driven by hand against stand-in frameworks and simulated meters:
the log keys, the one line at the end, .result, rank handling, a machine with nothing to measure, and saving."""
import contextlib
import importlib
import io
import os
import sys
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SIM = {"JOULES_SIMULATE": "gpu:Test GPU:200"}


def fake_transformers():
    m = types.ModuleType("transformers")
    class TrainerCallback:  # the real one has every hook as a no-op
        pass
    m.TrainerCallback = TrainerCallback
    return m


def fake_lightning(old=False):
    class Callback:
        pass
    if old:
        pl = types.ModuleType("pytorch_lightning"); pl.Callback = Callback
        return {"pytorch_lightning": pl, "lightning": None, "lightning.pytorch": None}
    lt = types.ModuleType("lightning"); lp = types.ModuleType("lightning.pytorch")
    lp.Callback = Callback; lt.pytorch = lp
    return {"lightning": lt, "lightning.pytorch": lp}


def load(name, modules):
    """Import joules.integrations.<name> fresh, with these stand-ins in sys.modules."""
    for k in [k for k in sys.modules if k.startswith("joules.integrations.")]:
        del sys.modules[k]
    with mock.patch.dict(sys.modules, modules):
        return importlib.import_module(f"joules.integrations.{name}")


class State:
    def __init__(self, zero=True):
        self.global_step, self.is_local_process_zero = 0, zero


class HuggingFace(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, SIM)
        self.env.start()
        self.hf = load("huggingface", {"transformers": fake_transformers()})

    def tearDown(self):
        self.env.stop()

    def train(self, cb, state, epochs=2, steps_per_epoch=3):
        err = io.StringIO()
        seen = []
        with contextlib.redirect_stderr(err):
            cb.on_train_begin(None, state, None)
            for _ in range(epochs):
                cb.on_epoch_begin(None, state, None)
                for _ in range(steps_per_epoch):
                    time.sleep(0.03)
                    state.global_step += 1
                    logs = {"loss": 1.0}
                    cb.on_log(None, state, None, logs=logs)
                    seen.append(logs)
                cb.on_epoch_end(None, state, None)
            cb.on_train_end(None, state, None)
        return seen, err.getvalue()

    def test_logs_result_and_one_line(self):
        cb = self.hf.JoulesCallback()
        seen, err = self.train(cb, State())
        self.assertEqual(set(seen[0]) - {"loss"}, {"energy/joules", "energy/kwh", "energy/avg_watts", "energy/joules_per_step"})
        js = [l["energy/joules"] for l in seen]
        self.assertGreater(js[0], 0)
        self.assertTrue(all(b > a for a, b in zip(js, js[1:])), js)
        last = seen[-1]
        self.assertAlmostEqual(last["energy/joules_per_step"], last["energy/joules"] / 6)
        self.assertAlmostEqual(last["energy/kwh"], last["energy/joules"] / 3.6e6)
        self.assertGreater(last["energy/avg_watts"], 100)  # a 200 W simulated GPU
        r = cb.result
        self.assertEqual((r["steps"], r["epochs"]), (6, 2))
        self.assertGreaterEqual(r["energy_j"], last["energy/joules"])
        self.assertAlmostEqual(r["j_per_step"], r["energy_j"] / 6)
        self.assertEqual(r["phases"]["epoch"]["count"], 2)
        self.assertTrue(r["simulated"])
        self.assertIsNone(r["run_id"])
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn("joules: training used", err)
        self.assertIn("per step", err)
        self.assertIn("SIMULATED", err)

    def test_custom_prefix(self):
        cb = self.hf.JoulesCallback(log_prefix="hw/")
        seen, _ = self.train(cb, State(), epochs=1, steps_per_epoch=1)
        self.assertIn("hw/joules", seen[0])

    def test_other_processes_do_nothing(self):
        cb = self.hf.JoulesCallback()
        seen, err = self.train(cb, State(zero=False), epochs=1)
        self.assertEqual(seen[-1], {"loss": 1.0})
        self.assertIsNone(cb.result)
        self.assertEqual(err, "")

    def test_nothing_to_measure_is_one_line_and_no_error(self):
        from joules import api
        with mock.patch.object(api, "Session", side_effect=api.NoMeters("no energy counters found")):
            cb = self.hf.JoulesCallback()
            seen, err = self.train(cb, State())
        self.assertEqual(seen[-1], {"loss": 1.0})
        self.assertIsNone(cb.result)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn("not measuring this training run (no energy counters found)", err)

    def test_bad_argument_does_not_break_training(self):
        cb = self.hf.JoulesCallback(track="not a valid project name!")
        seen, err = self.train(cb, State(), epochs=1)
        self.assertIsNone(cb.result)
        self.assertIn("not measuring", err)

    def test_track_saves_once_in_the_same_line(self):
        from joules import share
        with mock.patch.object(share, "api_key", return_value="lgk_test"), \
             mock.patch.object(share, "send_track", return_value={"id": 42}) as send:
            cb = self.hf.JoulesCallback(track="fine-tune")
            _, err = self.train(cb, State(), epochs=1)
        self.assertEqual(send.call_count, 1)
        sent = send.call_args[0][0]
        self.assertEqual(sent["project"], "fine-tune")
        self.assertEqual(sent["phases"][0]["name"], "epoch")
        self.assertEqual(cb.result["run_id"], 42)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn("saved to project fine-tune", err)

    def test_failed_save_never_raises(self):
        from joules import share
        with mock.patch.object(share, "api_key", return_value="lgk_test"), \
             mock.patch.object(share, "send_track", side_effect=RuntimeError("cannot reach https://wattops.dev")) as send:
            cb = self.hf.JoulesCallback(track="fine-tune")
            _, err = self.train(cb, State(), epochs=1)
        self.assertEqual(send.call_count, 1)  # not tried again when the session closes
        self.assertIsNotNone(cb.result)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn("not saved to WattOps (cannot reach https://wattops.dev)", err)

    def test_attach_puts_it_first(self):
        trainer = types.SimpleNamespace(callback_handler=types.SimpleNamespace(callbacks=["flow", "wandb"]))
        cb = self.hf.attach(trainer, log_prefix="e/")
        self.assertIs(trainer.callback_handler.callbacks[0], cb)
        self.assertEqual(trainer.callback_handler.callbacks[1:], ["flow", "wandb"])

    def test_train_end_twice_is_harmless(self):
        cb = self.hf.JoulesCallback()
        _, err = self.train(cb, State(), epochs=1)
        with contextlib.redirect_stderr(io.StringIO()) as again:
            cb.on_train_end(None, State(), None)
        self.assertEqual(again.getvalue(), "")


class Trainer:
    def __init__(self, local_rank=0, global_rank=0, every=2):
        self.local_rank, self.global_rank, self.global_step, self.log_every_n_steps = local_rank, global_rank, 0, every


class Module:
    def __init__(self):
        self.logged = []
    def log(self, name, value, **kw):
        self.logged.append((name, value, kw))


class Lightning(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, SIM)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def fit(self, cb, trainer, module, epochs=2, batches=4, fail=False):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cb.on_train_start(trainer, module)
            for _ in range(epochs):
                cb.on_train_epoch_start(trainer, module)
                for i in range(batches):
                    time.sleep(0.02)
                    trainer.global_step += 1
                    cb.on_train_batch_end(trainer, module, None, None, i)
                if fail:
                    cb.on_exception(trainer, module, RuntimeError("CUDA out of memory"))
                    break
                cb.on_train_epoch_end(trainer, module)
            cb.on_train_end(trainer, module)
        return err.getvalue()

    def test_logs_through_the_module_and_result(self):
        lt = load("lightning", fake_lightning())
        cb, module = lt.JoulesCallback(), Module()
        err = self.fit(cb, Trainer(every=2), module)
        names = {n for n, _, _ in module.logged}
        self.assertEqual(names, {"energy/joules", "energy/kwh", "energy/avg_watts", "energy/joules_per_step"})
        self.assertTrue(all(kw["rank_zero_only"] for _, _, kw in module.logged))
        steps = [kw["on_step"] for n, _, kw in module.logged if n == "energy/joules"]
        self.assertEqual(steps.count(True), 4)   # every 2nd of 8 steps
        self.assertEqual(steps.count(False), 2)  # and at each epoch's end
        self.assertTrue(all(isinstance(v, float) for _, v, _ in module.logged))
        self.assertEqual((cb.result["steps"], cb.result["epochs"]), (8, 2))
        self.assertEqual(len(err.strip().splitlines()), 1, err)

    def test_older_pytorch_lightning_package(self):
        lt = load("lightning", fake_lightning(old=True))
        cb = lt.JoulesCallback()
        self.fit(cb, Trainer(), Module(), epochs=1)
        self.assertIsNotNone(cb.result)

    def test_local_rank_one_does_nothing_and_global_rank_one_does_not_log(self):
        lt = load("lightning", fake_lightning())
        cb, module = lt.JoulesCallback(), Module()
        self.assertEqual(self.fit(cb, Trainer(local_rank=1, global_rank=1), module), "")
        self.assertIsNone(cb.result)
        self.assertEqual(module.logged, [])
        cb, module = lt.JoulesCallback(), Module()  # the main process of a second machine: measures, does not log
        self.fit(cb, Trainer(local_rank=0, global_rank=8), module, epochs=1)
        self.assertIsNotNone(cb.result)
        self.assertEqual(module.logged, [])

    def test_exception_ends_the_measurement_once(self):
        lt = load("lightning", fake_lightning())
        cb = lt.JoulesCallback()
        err = self.fit(cb, Trainer(), Module(), fail=True)
        self.assertEqual(cb.result["epochs"], 1)
        self.assertEqual(len(err.strip().splitlines()), 1, err)


class Imports(unittest.TestCase):
    def test_joules_never_imports_the_frameworks(self):
        for k in [k for k in sys.modules if k.startswith("joules.integrations")]:
            del sys.modules[k]
        with mock.patch.dict(sys.modules, {"transformers": None, "lightning": None}):
            import joules.integrations  # noqa: F401  (the package itself needs neither)

    def test_missing_framework_says_what_to_install(self):
        with self.assertRaises(ImportError) as e:
            load("huggingface", {"transformers": None})
        self.assertIn("pip install transformers", str(e.exception))
        with self.assertRaises(ImportError) as e:
            load("lightning", {"lightning": None, "lightning.pytorch": None, "pytorch_lightning": None})
        self.assertIn("pip install lightning", str(e.exception))


if __name__ == "__main__":
    unittest.main()
