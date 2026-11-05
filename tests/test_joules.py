"""joules tests. No GPU needed: real sysfs layouts are recreated in a temp dir, NVML is faked
behind the same interface, and model servers are tiny local HTTP servers.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import base64
import contextlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from joules import cli, meters, model

ROOT = Path(__file__).resolve().parents[2]  # apps/landauer-gap
sys.path.insert(0, str(ROOT / "tests"))


def write(p: Path, text) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text))


def run_cli(argv, env=None):
    err, out = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=False), contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


SIM = {"JOULES_SIMULATE": "gpu:H100 SXM:600,cpu:package 0:100"}

# Never read or write the real ~/.config/landauer-gap of whoever runs the tests.
_CONFIG = tempfile.mkdtemp(prefix="joules-test-home-")
os.environ["HOME"] = _CONFIG
os.environ["USERPROFILE"] = _CONFIG  # Windows finds ~ through USERPROFILE, not HOME
os.environ["JOULES_NO_PROMPT"] = "1"
os.environ.pop("SUDO_USER", None)
os.environ.pop("LANDAUER_API_KEY", None)


@unittest.skipIf(os.name == "nt", "RAPL is Linux only (and its sysfs names contain ':', which Windows does not allow)")
class Rapl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        pc = self.root / "class/powercap"
        write(pc / "intel-rapl:0/name", "package-0"); write(pc / "intel-rapl:0/energy_uj", 1_000_000); write(pc / "intel-rapl:0/max_energy_range_uj", 10_000_000)
        write(pc / "intel-rapl:0:0/name", "core"); write(pc / "intel-rapl:0:0/energy_uj", 5)
        write(pc / "intel-rapl:0:1/name", "dram"); write(pc / "intel-rapl:0:1/energy_uj", 0)
        write(pc / "intel-rapl:1/name", "psys"); write(pc / "intel-rapl:1/energy_uj", 0)
        self.pc = pc

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovers_package_and_dram_only(self):
        with mock.patch.dict(os.environ, {"JOULES_SYSFS": str(self.root)}):
            found = meters.discover_rapl([])
        self.assertEqual([m.name for m in found], ["CPU package 0", "DRAM (package 0)"])
        self.assertEqual([m.kind for m in found], ["cpu", "dram"])

    def test_counter_handles_wraparound(self):
        with mock.patch.dict(os.environ, {"JOULES_SYSFS": str(self.root)}):
            m = meters.discover_rapl([])[0]
            m.start(0.0)
            write(self.pc / "intel-rapl:0/energy_uj", 9_000_000); m.poll(1.0)      # +8 J
            write(self.pc / "intel-rapl:0/energy_uj", 2_000_000); m.poll(2.0)      # wrapped: +3 J (10 - 9 + 2)
        self.assertAlmostEqual(m.energy_j, 11.0)
        self.assertAlmostEqual(m.peak_w, 8.0)
        self.assertAlmostEqual(m.avg_w, 5.5)


@unittest.skipIf(os.name == "nt", "amdgpu is read from Linux sysfs")
class AmdGpu(unittest.TestCase):
    def test_energy_counter_preferred_then_power(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            hw0 = root / "class/drm/card0/device/hwmon/hwmon2"
            write(hw0 / "name", "amdgpu"); write(hw0 / "energy1_input", 0)
            hw1 = root / "class/drm/card1/device/hwmon/hwmon3"
            write(hw1 / "name", "amdgpu"); write(hw1 / "power1_average", 300_000_000)
            with mock.patch.dict(os.environ, {"JOULES_SYSFS": str(root)}):
                found = meters.discover_amdgpu([])
                self.assertEqual([m.method for m in found], ["amdgpu energy counter", "amdgpu power sampling"])
                p = found[1]
                p.start(0.0); p.poll(2.0)
                self.assertAlmostEqual(p.energy_j, 600.0)

    def test_card_order_names_and_selection(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for card, hw in ((10, 5), (2, 1), (3, 2)):  # card10 sorts before card2 as text
                d = root / f"class/drm/card{card}/device"
                write(d / f"hwmon/hwmon{hw}/name", "amdgpu"); write(d / f"hwmon/hwmon{hw}/power1_average", 100_000_000)
            write(root / "class/drm/card2/device/product_name", "AMD Instinct MI300X")
            with mock.patch.dict(os.environ, {"JOULES_SYSFS": t}):
                os.environ.pop("HIP_VISIBLE_DEVICES", None); os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                self.assertEqual([m.name for m in meters.discover_amdgpu([])], ["GPU 0 AMD Instinct MI300X", "GPU 1 AMD GPU", "GPU 2 AMD GPU"])
                self.assertEqual(model.gpu_from_name("GPU 0 AMD Instinct MI300X"), "mi300x")
                self.assertEqual([m.name for m in meters.discover_amdgpu([], [2])], ["GPU 2 AMD GPU"])
                with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1"}):
                    self.assertEqual([m.name for m in meters.discover_amdgpu([])], ["GPU 1 AMD GPU"])


class AmdGpuNames(unittest.TestCase):
    """Cloud virtual functions (DigitalOcean's MI300X VF) have no product_name: name them by PCI IDs."""

    def test_pci_id_when_product_name_is_missing(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            d = root / "class/drm/card1/device"
            write(d / "hwmon/hwmon0/name", "amdgpu"); write(d / "hwmon/hwmon0/power1_input", 158_000_000)
            write(d / "vendor", "0x1002"); write(d / "device", "0x74b5")
            with mock.patch.dict(os.environ, {"JOULES_SYSFS": t}):
                os.environ.pop("HIP_VISIBLE_DEVICES", None); os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                m = meters.discover_amdgpu([])
            self.assertEqual([x.name for x in m], ["GPU 0 AMD Instinct MI300X VF"])
            self.assertEqual(model.gpu_from_name(m[0].name), "mi300x")

    def test_pci_ids_database_for_other_cards(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            d = root / "class/drm/card0/device"
            write(d / "hwmon/hwmon1/name", "amdgpu"); write(d / "hwmon/hwmon1/power1_average", 50_000_000)
            write(d / "vendor", "0x1002"); write(d / "device", "0x744c")
            ids = root / "pci.ids"
            ids.write_text("1002  Advanced Micro Devices, Inc. [AMD/ATI]\n\t744c  Navi 31 [Radeon RX 7900 XT/7900 XTX]\n"
                           "\t\t1002 0e3b  Radeon RX 7900 XTX\n1022  Advanced Micro Devices, Inc. [AMD]\n\t744c  Not this\n")
            with mock.patch.dict(os.environ, {"JOULES_SYSFS": t}), mock.patch.object(meters, "PCI_IDS", (str(ids),)):
                os.environ.pop("HIP_VISIBLE_DEVICES", None); os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                self.assertEqual([x.name for x in meters.discover_amdgpu([])], ["GPU 0 AMD Radeon RX 7900 XT/7900 XTX"])
            with mock.patch.dict(os.environ, {"JOULES_SYSFS": t}), mock.patch.object(meters, "PCI_IDS", ("/nonexistent",)):
                self.assertEqual([x.name for x in meters.discover_amdgpu([])], ["GPU 0 AMD GPU"])


@unittest.skipIf(os.name == "nt", "i915/xe are read from Linux sysfs")
class IntelGpu(unittest.TestCase):
    def test_arc_and_data_center_energy_counters(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write(root / "class/drm/card0/device/hwmon/hwmon3/name", "xe")
            write(root / "class/drm/card0/device/hwmon/hwmon3/energy1_input", 5_000_000)
            write(root / "class/drm/card1/device/hwmon/hwmon4/name", "i915")  # integrated: no counter
            with mock.patch.dict(os.environ, {"JOULES_SYSFS": t}):
                found = meters.discover_intelgpu([])
            self.assertEqual([(m.name, m.method, m.kind) for m in found], [("GPU 0 Intel GPU", "intel gpu energy counter", "gpu")])
            m = found[0]; m.start(0.0)
            write(root / "class/drm/card0/device/hwmon/hwmon3/energy1_input", 155_000_000); m.poll(1.0)
            self.assertAlmostEqual(m.energy_j, 150.0)


# Prints samples the way powermetrics does, without flushing, so they sit in a buffer when stdout is a
# pipe (what made Macs read almost 0 W) and arrive one by one on a terminal.
FAKE_POWERMETRICS = r'''
import os, sys, time
# Buffered like C stdio, whatever PYTHONUNBUFFERED says: by line on a terminal, in 64 KB blocks on a pipe.
out = os.fdopen(1, "w", buffering=1 if os.isatty(1) else 65536)
parts_only = sys.argv[1] == "parts"
if sys.argv[1] == "once":  # one sample, then quiet: the test feeds samples itself
    out.write("*** Sampled system activity (Tue Oct  6 10:00:00 2026 +0000) (200.00ms elapsed) ***\n\nCombined Power (CPU + GPU + ANE): 2500 mW\n")
    out.flush(); time.sleep(60); sys.exit()
last = time.monotonic() - 0.1
for i in range(400):
    now = time.monotonic()  # like powermetrics: the real length of each sample, which on a busy machine is over 100 ms
    out.write("*** Sampled system activity (Tue Oct  6 10:00:00 2026 +0000) (%.2fms elapsed) ***\n\n" % ((now - last) * 1000))
    last = now
    if sys.argv[1] == "intel":  # Intel Macs
        out.write("**** Processor usage ****\n\nIntel energy model derived package power (CPUs+GPU+DRAM): 7.50W\n")
        time.sleep(0.1); continue
    out.write("CPU Power: 2000 mW\nGPU Power: 400 mW\nANE Power: 100 mW\n")
    if not parts_only:
        out.write("Combined Power (CPU + GPU + ANE): 2500 mW\n")
    time.sleep(0.1)
'''


@unittest.skipIf(os.name != "posix", "pseudo-terminals are POSIX only")
class Powermetrics(unittest.TestCase):
    def measure(self, mode, seconds=1.5):
        from joules.sampler import Sampler
        with tempfile.TemporaryDirectory() as t:
            script = Path(t) / "fake_powermetrics.py"
            script.write_text(FAKE_POWERMETRICS)
            m = meters.PowermetricsMeter(cmd=[sys.executable, str(script), mode], intel=mode == "intel")
            try:
                with Sampler([m], 0.05) as s:
                    time.sleep(seconds)
                return m, s.duration_s
            finally:
                m.close()

    def test_combined_power_is_integrated_while_running(self):
        m, d = self.measure("combined")
        self.assertAlmostEqual(m.avg_w, 2.5, delta=0.5)  # within about one 0.1 s sample
        self.assertAlmostEqual(m.energy_j, 2.5 * d, delta=0.5)
        self.assertEqual(m.peak_w, 2.5)

    def test_only_the_part_of_a_sample_inside_the_run_counts(self):
        # Samples are averages over the 0.2 s that just ended: the first began before the run, the last ends
        # after it. Counting them whole added up to 0.4 s of power to every run.
        with tempfile.TemporaryDirectory() as t:
            script = Path(t) / "fake_powermetrics.py"
            script.write_text(FAKE_POWERMETRICS)
            m = meters.PowermetricsMeter(cmd=[sys.executable, str(script), "once"])
            try:
                m._window = 0.2
                m.start(10.0)
                for arrival in (10.1, 10.3, 10.5):  # windows 9.9-10.1, 10.1-10.3, 10.3-10.5
                    if arrival > 10.4:
                        m._t_stop = 10.4  # the run ended at 10.4
                    with mock.patch.object(meters.time, "monotonic", return_value=arrival):
                        m._add(2.5)
                self.assertAlmostEqual(m.energy_j, 2.5 * 0.4)  # 2.5 W for exactly the 0.4 s of the run
            finally:
                m.close()

    def test_per_part_lines_when_there_is_no_combined_line(self):
        m, d = self.measure("parts")
        self.assertAlmostEqual(m.avg_w, 2.5, delta=0.5)

    def test_each_sample_is_added_once_with_the_gpu_power_sampler(self):
        # macOS prints, per sample of `--samplers cpu_power,gpu_power`, the CPU/GPU/ANE lines, the Combined line,
        # then the GPU section with a second "GPU Power" line. Found by HorneSci: five samples became nine
        # additions, the extra GPU line added as a sample of its own at the start of the next one.
        sample = ("*** Sampled system activity (Fri Oct 23 10:00:00 2026 -0700) (200.11ms elapsed) ***\n",
                  "**** Processor usage ****\n", "CPU Power: 2000 mW\n", "GPU Power: 400 mW\n", "ANE Power: 100 mW\n",
                  "Combined Power (CPU + GPU + ANE): 2500 mW\n", "**** GPU usage ****\n",
                  "GPU HW active frequency: 1296 MHz\n", "GPU HW active residency:  40.12%\n", "GPU Power: 400 mW\n")
        with tempfile.TemporaryDirectory() as t:
            script = Path(t) / "fake_powermetrics.py"
            script.write_text(FAKE_POWERMETRICS)
            m = meters.PowermetricsMeter(cmd=[sys.executable, str(script), "once"])
            try:
                added = []
                with mock.patch.object(m, "_add", side_effect=added.append):
                    for _ in range(5):
                        for line in sample:
                            m._line(line)
                    m._line(sample[0])  # the sixth sample begins
                self.assertEqual(added, [2.5] * 5)
                added.clear()  # a Combined line printed twice in one sample still counts once
                with mock.patch.object(m, "_add", side_effect=added.append):
                    for line in (sample[0], "Combined Power (CPU + GPU + ANE): 1500 mW\n",
                                 "Combined Power (CPU + GPU + ANE): 1500 mW\n", sample[0]):
                        m._line(line)
                self.assertEqual(added, [1.5])
            finally:
                m.close()

    def test_intel_mac_package_power(self):
        m, _ = self.measure("intel")
        self.assertAlmostEqual(m.avg_w, 7.5, delta=1.5)
        self.assertEqual((m.kind, m.name), ("cpu", "Intel Mac CPU package (CPU + GPU + DRAM)"))

    def test_close_stops_powermetrics(self):
        m, _ = self.measure("combined", seconds=0.3)
        self.assertIsNotNone(m.proc.poll())


class FakeNVML:
    def __init__(self, devices):
        self.devices = devices  # list of dicts: name, energy_mj (or None), power_mw
    def count(self): return len(self.devices)
    def handle(self, i): return i
    def name(self, h): return self.devices[h]["name"]
    def energy_mj(self, h): return self.devices[h].get("energy_mj")
    def power_mw(self, h): return self.devices[h].get("power_mw")
    def uuid(self, h): return self.devices[h].get("uuid", "")
    def mig_parent_index(self, u): return next((i for i, d in enumerate(self.devices) if u in d.get("migs", ())), None)
    def close(self): pass


class Nvml(unittest.TestCase):
    def test_counter_power_fallback_and_visible_devices(self):
        devs = [{"name": "NVIDIA H100 80GB HBM3", "energy_mj": 1_000_000, "power_mw": 650_000},
                {"name": "Tesla T4", "energy_mj": None, "power_mw": 70_000},
                {"name": "NVIDIA A100-SXM4-40GB", "energy_mj": 5, "power_mw": 1}]
        with mock.patch.object(meters, "NVML_FACTORY", lambda: FakeNVML(devs)), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            found, _ = meters.discover_nvml([], None)
        self.assertEqual([m.method for m in found], ["nvml counter", "nvml power sampling"])
        self.assertEqual(found[0].name, "GPU 0 NVIDIA H100 80GB HBM3")
        m = found[0]; m.start(0.0)
        devs[0]["energy_mj"] += 650_000; m.poll(1.0)
        self.assertAlmostEqual(m.energy_j, 650.0)
        t4 = found[1]; t4.start(0.0); t4.poll(10.0)
        self.assertAlmostEqual(t4.energy_j, 700.0)
        self.assertEqual(model.gpu_from_name(found[0].name), "h100")
        self.assertEqual(model.gpu_from_name("NVIDIA A100-SXM4-40GB"), "a100")

    def test_uuids_mig_slices_and_ordering(self):
        devs = [{"name": "NVIDIA H100 80GB HBM3", "energy_mj": 1, "uuid": "GPU-aaaa1111-0000", "migs": ["MIG-1234"]},
                {"name": "NVIDIA L40S", "energy_mj": 1, "uuid": "GPU-bbbb2222-0000"},
                {"name": "NVIDIA L40S", "energy_mj": 1, "uuid": "GPU-cccc3333-0000"}]
        def names(env, gpus=None):
            notes = []
            with mock.patch.object(meters, "NVML_FACTORY", lambda: FakeNVML(devs)), mock.patch.dict(os.environ, env):
                for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
                    if k not in env: os.environ.pop(k, None)
                found, _ = meters.discover_nvml(notes, gpus)
            return [m.name for m in found], notes
        self.assertEqual(names({"CUDA_VISIBLE_DEVICES": "GPU-cccc3333-0000,GPU-bbbb"})[0], ["GPU 1 NVIDIA L40S", "GPU 2 NVIDIA L40S"])
        got, notes = names({"CUDA_VISIBLE_DEVICES": "MIG-1234"})
        self.assertEqual(got, ["GPU 0 NVIDIA H100 80GB HBM3"]); self.assertIn("MIG", notes[0])
        self.assertEqual(names({"CUDA_VISIBLE_DEVICES": "1,GPU-nope,2"})[0], ["GPU 1 NVIDIA L40S"])  # CUDA stops at a bad entry
        got, notes = names({"CUDA_VISIBLE_DEVICES": "0"})
        self.assertTrue(any("CUDA_DEVICE_ORDER=PCI_BUS_ID" in n for n in notes))
        self.assertEqual(names({"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[1], [])
        self.assertEqual(names({"CUDA_VISIBLE_DEVICES": "0"}, gpus=[1])[1], [])
        self.assertEqual(len(names({})[0]), 3)

    def test_no_driver_means_no_meters(self):
        def boom(): raise OSError("no lib")
        with mock.patch.object(meters, "NVML_FACTORY", boom):
            self.assertEqual(meters.discover_nvml([], None)[0], [])


class Format(unittest.TestCase):
    def test_small_machines_and_inline_scripts_read_well(self):
        self.assertEqual([cli.fmt_w(w) for w in (0.0, 0.2946, 1.72, 12.4, 590.2)], ["0", "0.29", "1.7", "12", "590"])
        self.assertEqual(cli._money(8.2e-08), "<0.0001")
        self.assertEqual([cli.fmt_wh(j) for j in (2.95, 28.7, 7200, 0)], ["819 µWh", "7.97 mWh", "2 Wh", "0 Wh"])
        self.assertEqual(cli._money(0.25), "0.25")
        self.assertEqual(cli.fmt_g(0.000303), "<0.001")
        self.assertEqual(cli.short_cmd(["python3", "-c", "x=0\nfor i in range(9): x+=i"]), "python3 -c x=0 for i in range(9): x+=i")
        self.assertTrue(cli.short_cmd(["python3", "-c", "y" * 100]).endswith("…"))


class Run(unittest.TestCase):
    def test_measures_and_passes_exit_code(self):
        code, out, err = run_cli(["run", "--json", "--", sys.executable, "-c", "import time,sys; time.sleep(0.6); sys.exit(3)"], SIM)
        self.assertEqual(code, 3)
        r = json.loads(out)
        self.assertEqual(r["exit_code"], 3)
        self.assertTrue(r["simulated"])
        d = r["duration_s"]
        self.assertGreater(d, 0.55)
        gpu = next(x for x in r["devices"] if x["kind"] == "gpu")
        self.assertAlmostEqual(gpu["energy_j"], 600 * d, delta=600 * 0.05)
        self.assertAlmostEqual(r["totals"]["devices_j"], 700 * d, delta=700 * 0.05)
        self.assertAlmostEqual(r["totals"]["kwh"], r["totals"]["devices_j"] / 3.6e6)
        self.assertAlmostEqual(r["totals"]["co2"]["grams"], r["totals"]["kwh"] * 370)

    def test_text_report_and_out_file(self):
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, "r.json")
            code, out, err = run_cli(["run", "--out", f, "--pue", "1.2", "--grid", "fr", "--", sys.executable, "-c", "pass"], SIM)
            self.assertEqual(code, 0)
            self.assertIn("SIMULATED", err)
            self.assertIn("facility (PUE 1.2)", err)
            with open(f) as fh:
                r = json.load(fh)
            self.assertEqual(r["totals"]["co2"]["g_per_kwh"], 55)
            self.assertAlmostEqual(r["totals"]["facility_j"], r["totals"]["devices_j"] * 1.2)

    def test_short_form_and_missing_command(self):
        code, _, _ = run_cli(["--", sys.executable, "-c", "pass"], SIM)
        self.assertEqual(code, 0)
        code, _, err = run_cli(["run", "--", "definitely-not-a-command-xyz"], SIM)
        self.assertEqual(code, 127)
        code, _, err = run_cli(["run"], SIM)
        self.assertEqual(code, 2)

    def test_estimate_calibration_and_calculator_link(self):
        code, out, _ = run_cli(["run", "--json", "--params", "0.001", "--tokens", "0.01", "--", sys.executable, "-c", "import time; time.sleep(0.5)"], SIM)
        r = json.loads(out)
        flops = 6 * 1e6 * 1e7
        self.assertAlmostEqual(r["totals"]["landauer"]["operations"], flops)
        e, c = r["estimate"], r["calibration"]
        self.assertAlmostEqual(e["accelerator_j"], 700 * 0.8 * flops / (989e12 * 0.4))
        self.assertAlmostEqual(c["effective_mfu_pct"], flops / (989e12 * r["duration_s"]) * 100, places=6)
        self.assertAlmostEqual(c["avg_draw_pct"], 600 / 700 * 100, delta=5)
        raw = r["calculator_link"].split("?s=")[1]
        state = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        self.assertEqual(state["gpu"], "h100"); self.assertEqual(state["params"], 0.001)
        self.assertAlmostEqual(state["draw"], round(c["avg_draw_pct"], 1))

    def test_subtract_idle(self):
        code, out, _ = run_cli(["run", "--json", "--subtract-idle", "0.3", "--", sys.executable, "-c", "import time; time.sleep(0.4)"], SIM)
        r = json.loads(out)
        # constant simulated power: everything is "idle", so energy above idle is ~0
        self.assertLess(r["totals"]["above_idle_j"], 0.05 * r["totals"]["devices_j"])
        self.assertAlmostEqual(r["devices"][0]["idle_w"], 600, delta=1)


class Sanity(unittest.TestCase):
    def test_impossible_workload_is_flagged_not_calibrated(self):
        code, out, _ = run_cli(["run", "--json", "--params", "100", "--tokens", "100", "--", sys.executable, "-c", "pass"], SIM)
        r = json.loads(out)
        self.assertNotIn("calibration", r)
        self.assertTrue(any("more than" in n and "utilisation" in n for n in r["notes"]), r["notes"])

    def test_mixed_gpu_models_skip_calibration(self):
        env = {"JOULES_SIMULATE": "gpu:H100 SXM:600,gpu:Tesla T4:70"}
        code, out, _ = run_cli(["run", "--json", "--params", "0.001", "--tokens", "0.001", "--", sys.executable, "-c", "pass"], env)
        r = json.loads(out)
        self.assertNotIn("calibration", r)
        self.assertTrue(any("Different GPU models" in n for n in r["notes"]))


@unittest.skipUnless(shutil.which("gcc") and os.name == "posix", "needs gcc to build the stand-in NVIDIA library")
class RealNvmlBinding(unittest.TestCase):
    """Builds a C stand-in for libnvidia-ml.so.1 and drives the real ctypes binding through it."""
    def test_ctypes_binding_against_compiled_library(self):
        src = Path(__file__).parent / "fake_nvml" / "fake_nvml.c"
        with tempfile.TemporaryDirectory() as t:
            lib = os.path.join(t, "libnvidia-ml.so.1")
            subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", lib, str(src)], check=True)
            code = ("import ctypes, json, sys, time; from joules import meters\n"
                    "meters.NVML_FACTORY = lambda: meters.NVML.__new__(meters.NVML)\n"
                    "n = meters.NVML.__new__(meters.NVML); n.lib = ctypes.CDLL(sys.argv[1]); n.lib.nvmlInit_v2()\n"
                    "meters.NVML_FACTORY = lambda: n\n"
                    "ms, _ = meters.discover_nvml([], None)\n"
                    "[m.start(time.monotonic()) for m in ms]; time.sleep(0.5); now = time.monotonic(); [m.stop(now) for m in ms]\n"
                    "h0 = n.handle(0); rc = n.set_power_limit_mw(h0, 490000); low = (n.power_mw(h0), n.clock_mhz(h0)); n.set_power_limit_mw(h0, 700000)\n"
                    "print(json.dumps([[m.name, m.method, m.avg_w] for m in ms] + [ms[0].health(), ms[1].health(), [rc, low, n.power_limit_range_mw(h0), n.memory_used_pct(h0)]]))")
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
            out = subprocess.run([sys.executable, "-c", code, lib], capture_output=True, text=True, env=env, check=True).stdout
        *rows, h0, h1, limits = json.loads(out)
        self.assertEqual([r[:2] for r in rows], [["GPU 0 NVIDIA H100 80GB HBM3", "nvml counter"], ["GPU 1 Tesla T4", "nvml power sampling"]])
        self.assertAlmostEqual(rows[0][2], 650, delta=15)
        self.assertAlmostEqual(rows[1][2], 300, delta=1)
        # health through the real ctypes binding: clocks, temperature, limit, throttle reasons, utilisation, memory
        self.assertEqual((h0["sm_clock_mhz"], h0["sm_clock_max_mhz"], h0["mem_clock_mhz"], h0["temp_c_max"], h0["power_limit_w"]), (1755, 1980, 2619, 68, 700))
        self.assertAlmostEqual(h0["clock_pct"], 1755 / 1980 * 100, places=3)
        self.assertEqual((h0["throttled_pct"], h0["throttle_reasons"], h0["gpu_util_pct"], h0["mem_util_pct"]), (100, {"power cap": 100}, 87, 40))
        self.assertAlmostEqual(h0["mem_used_pct_max"], 25.0)
        self.assertEqual((h1["throttled_pct"], h1["throttle_reasons"], h1["temp_c_max"]), (0, {}, 41))  # idle is not throttling
        self.assertEqual(limits, [0, [490000, 1400], [200000, 700000], 25.0])  # the limit took, and the card slowed down

class Model(unittest.TestCase):
    def test_estimate_matches_reference_model(self):
        try:
            import reference as R   # the website calculator's reference model, beside the package in the WattOps repository
        except ImportError:
            self.skipTest("the website's reference model is not in this checkout")
        s = {"mode": "training", "tdp": 700, "peak": 989, "params": 70, "tokens": 15000, "count": 2048, "mfu": 40,
             "draw": 80, "overhead": 1.3, "pue": 1.2, "price": 0.1, "ci": 370, "temp": 300, "bits": 1}
        ref = R.model(s)
        mine = model.estimate_accelerator_j(model.training_flops(70, 15000), "h100", 40, 80) / 3.6e6
        self.assertAlmostEqual(mine, ref["accel_kwh"], delta=ref["accel_kwh"] * 1e-12)
        self.assertAlmostEqual(model.landauer_floor_j(300), 2.870978885e-21, delta=1e-30)


class _Server:
    def __init__(self, handler):
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def handler(kind, tokens=200, delay=0.05, show=None):
    class H(BaseHTTPRequestHandler):
        calls = []
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            H.calls.append((self.path, body))
            if show is not None and self.path == "/api/show":
                data = json.dumps(show).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data); return
            time.sleep(delay)
            n = min(tokens, body.get("options", {}).get("num_predict", body.get("max_tokens", tokens)))
            if kind == "ollama":
                out = {"eval_count": n, "prompt_eval_count": 20, "eval_duration": int(delay * 1e9)}
            else:
                out = {"usage": {"completion_tokens": n, "prompt_tokens": 20}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
    return H


class Bench(unittest.TestCase):
    def test_ollama_energy_per_token_and_api_compare(self):
        H = handler("ollama"); srv = _Server(H)
        try:
            code, out, _ = run_cli(["bench", "ollama:llama3.1:8b", "--url", srv.url, "--runs", "3", "--max-tokens", "200",
                                    "--params", "8", "--api-price", "0.6", "--json"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 0)
        r = json.loads(out); b = r["bench"]
        self.assertEqual(b["output_tokens"], 600); self.assertEqual(b["prompt_tokens"], 60)
        self.assertEqual(H.calls[0][0], "/api/show")  # is the model there? (it is: no download)
        self.assertEqual([c[0] for c in H.calls[1:]], ["/api/generate"] * 4)  # 1 warm-up + 3 measured
        self.assertAlmostEqual(b["joules_per_output_token"], r["totals"]["devices_j"] / 600)
        self.assertAlmostEqual(b["kwh_per_million_output_tokens"], b["joules_per_output_token"] * 1e6 / 3.6e6)
        self.assertAlmostEqual(b["api_vs_electricity_ratio"], 0.6 / b["electricity_cost_per_million_output_tokens"])
        self.assertAlmostEqual(r["totals"]["landauer"]["operations"], 2 * 8e9 * 660)

    def test_ollama_model_size_gives_the_landauer_gap_without_params(self):
        H = handler("ollama", show={"model_info": {"general.parameter_count": 3212749888}, "details": {"parameter_size": "3.2B"}})
        srv = _Server(H)
        try:
            code, out, err = run_cli(["bench", "ollama:llama3.2:3b", "--url", srv.url, "--runs", "2", "--max-tokens", "100", "--json"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertIn("3.21B parameters (from Ollama)", err)
        self.assertAlmostEqual(r["totals"]["landauer"]["operations"], 2 * 3212749888 * (200 + 40))
        self.assertGreater(r["totals"]["landauer"]["gap"], 1)
        from joules import bench  # older Ollama: only the rounded size
        with mock.patch.object(bench, "_post", return_value={"details": {"parameter_size": "8.0B"}}):
            self.assertEqual(bench.model_params(bench.parse_target("ollama:x", None)), 8.0)
        with mock.patch.object(bench, "_post", return_value={}):
            self.assertIsNone(bench.model_params(bench.parse_target("ollama:x", None)))
        self.assertIsNone(bench.model_params(bench.parse_target("openai:x", "http://h/v1")))

    def test_openai_compatible_server(self):
        H = handler("openai", tokens=50); srv = _Server(H)
        try:
            code, out, err = run_cli(["bench", "openai:my-model", "--url", srv.url + "/v1", "--runs", "2", "--warmup", "0"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 0)
        self.assertEqual(H.calls[0][0], "/v1/chat/completions")
        self.assertIn("per token", err)
        self.assertIn("100", err)  # output tokens

    def test_clear_errors(self):
        code, _, err = run_cli(["bench", "llama3"], SIM)
        self.assertEqual(code, 2); self.assertIn("ollama:<model>", err)
        with mock.patch("joules.bench.start_ollama", return_value=False):
            code, _, err = run_cli(["bench", "ollama:x", "--url", "http://127.0.0.1:9"], SIM)
        self.assertEqual(code, 1); self.assertIn("Ollama is not installed", err)
        code, _, err = run_cli(["bench", "openai:x", "--url", "http://127.0.0.1:9/v1"], SIM)
        self.assertEqual(code, 1); self.assertIn("Is the model server running?", err)


class Devices(unittest.TestCase):
    def test_lists_meters_and_explains_when_none(self):
        code, out, _ = run_cli(["devices", "--json"], SIM)
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out)["devices"]), 2)
        with tempfile.TemporaryDirectory() as t, mock.patch.object(meters, "NVML_FACTORY", lambda: (_ for _ in ()).throw(OSError())):
            env = {"JOULES_SYSFS": t}
            with mock.patch.dict(os.environ, env), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("JOULES_SIMULATE", None)
                code, out, _ = run_cli(["devices"])
        self.assertEqual(code, 1)
        self.assertIn("No energy counters found", out)


def bench_report(**over):
    r = {"schema": "joules/1", "kind": "bench", "simulated": False, "duration_s": 10.0,
         "host": {"cpu": "AMD EPYC 9654"},
         "devices": [{"name": "GPU 0 NVIDIA H100 80GB HBM3", "kind": "gpu", "energy_j": 3000.0, "avg_w": 300.0},
                     {"name": "GPU 1 NVIDIA H100 80GB HBM3", "kind": "gpu", "energy_j": 3000.0, "avg_w": 300.0},
                     {"name": "CPU package 0", "kind": "cpu", "energy_j": 1000.0, "avg_w": 100.0}],
         "totals": {"devices_j": 7000.0, "above_idle_j": 7000.0},
         "bench": {"target": "ollama:llama3.1:8b", "output_tokens": 1000, "tokens_per_s": 100.0}}
    r.update(over)
    return r


def board_handler(status=201, reply=None):
    class H(BaseHTTPRequestHandler):
        calls = []
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            H.calls.append((self.path, self.headers["Authorization"], body))
            data = json.dumps(reply or {"id": 1}).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
    return H


class Share(unittest.TestCase):
    KEY = "lgk_" + "a" * 40

    def test_measurement_payload(self):
        from joules import share
        m = share.measurement(bench_report())
        self.assertEqual((m["model"], m["backend"], m["device"], m["device_count"]), ("llama3.1:8b", "ollama", "NVIDIA H100 80GB HBM3", 2))
        self.assertAlmostEqual(m["j_per_token"], 7.0)
        self.assertEqual((m["avg_w"], m["sources"], m["idle_subtracted"]), (700.0, "cpu+gpu", True))
        idle = bench_report(totals={"devices_j": 7000.0, "above_idle_j": 4000.0})
        self.assertAlmostEqual(share.measurement(idle)["j_per_token"], 4.0)
        cpu_only = bench_report(devices=[{"name": "CPU package 0", "kind": "cpu", "energy_j": 500.0, "avg_w": 50.0}], totals={"devices_j": 500.0, "above_idle_j": 400.0})
        self.assertEqual(share.measurement(cpu_only)["device"], "AMD EPYC 9654")

    def test_refuses_what_should_not_be_shared(self):
        from joules import share
        for bad, msg in [(bench_report(simulated=True), "simulated"), (bench_report(kind="run"), "only `joules bench`"),
                         (bench_report(bench={"target": "ollama:x", "output_tokens": 5}), "at least 32"),
                         (bench_report(devices=[{"name": "GPU 0 NVIDIA H100", "kind": "gpu", "energy_j": 1, "avg_w": 1},
                                                {"name": "GPU 1 NVIDIA A100", "kind": "gpu", "energy_j": 1, "avg_w": 1}]), "different GPU models"),
                         # the board compares energy above idle: none measured, or nothing measurable above it
                         (bench_report(totals={"devices_j": 7000.0}), "--subtract-idle 5"),
                         (bench_report(totals={"devices_j": 7000.0}, devices=[{"name": "CPU package 0", "kind": "cpu", "energy_j": 500.0,
                                                                                "avg_w": 50.0, "idle_w": 50.0}]), "no measurable power above idle")]:
            with self.assertRaisesRegex(ValueError, msg):
                share.measurement(bad)

    def test_share_file_posts_with_key_and_prints_link(self):
        H = board_handler(); srv = _Server(H)
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "r.json"; f.write_text(json.dumps(bench_report()))
            try:
                code, _, err = run_cli(["share", str(f), "--api-url", srv.url], {"LANDAUER_API_KEY": self.KEY})
            finally:
                srv.close()
        self.assertEqual(code, 0, err)
        path, auth, body = H.calls[0]
        self.assertEqual((path, auth), ("/api/v1/measurements", "Bearer " + self.KEY))
        self.assertEqual(body["model"], "llama3.1:8b")
        self.assertNotIn("host", body); self.assertNotIn("url", body)
        self.assertIn(srv.url + "/leaderboard?model=llama3.1%3A8b", err)

    def test_share_errors_are_explained(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANDAUER_API_KEY", None)
            with tempfile.TemporaryDirectory() as t:
                f = Path(t) / "r.json"; f.write_text(json.dumps(bench_report()))
                code, _, err = run_cli(["share", str(f)])
                self.assertEqual(code, 2); self.assertIn("LANDAUER_API_KEY", err)
                H = board_handler(401, {"error": {"code": "invalid_api_key", "message": "That API key is not valid or has been revoked."}})
                srv = _Server(H)
                try:
                    code, _, err = run_cli(["share", str(f), "--key", self.KEY, "--api-url", srv.url])
                finally:
                    srv.close()
                self.assertEqual(code, 1); self.assertIn("revoked", err)

    def test_bench_share_refuses_simulated_runs(self):
        H = handler("ollama"); srv = _Server(H)
        idle = []
        try:
            with mock.patch.object(cli, "idle_baseline", lambda m, secs, iv: idle.append(secs) or {}):
                code, _, err = run_cli(["bench", "ollama:llama3.1:8b", "--url", srv.url, "--runs", "1", "--share", "--key", self.KEY,
                                        "--api-url", "http://127.0.0.1:9"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 1)
        self.assertIn("simulated results cannot be shared", err)
        self.assertEqual(idle, [5])  # --share measures idle first, so the board compares energy above idle everywhere
        self.assertIn("measuring idle power for 5 s", err)


class Track(unittest.TestCase):
    KEY = "lgk_" + "b" * 40

    def test_label_never_carries_arguments(self):
        from joules import share
        self.assertEqual(share.safe_label(["python", "train.py", "--api-key=secret", "--lr", "3e-4"]), "python train.py")
        self.assertEqual(share.safe_label(["/usr/bin/python3", "-u", "/home/me/proj/train.py"]), "python3 train.py")
        self.assertEqual(share.safe_label(["python3", "-c", "import os; print(os.environ['TOKEN'])"]), "python3")
        self.assertEqual(share.safe_label(["torchrun", "--nproc_per_node=8", "-m", "my.module"]), "torchrun my.module")
        # a program that rewrote its process title: /proc gives the whole command line as one string
        self.assertEqual(share.safe_label(["headless_shell.real --type=renderer --headless=old --no-sandbox --enable-crash-reporter=secret"]), "headless_shell.real")
        self.assertEqual(share.safe_label(["/opt/chrome/chrome --type=gpu-process --token=abc"]), "chrome")
        self.assertEqual(share.safe_label(["postgres: ada mydb 10.0.0.5(5432) idle"]), "postgres:")
        self.assertEqual(share.safe_label(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --flag=x"]), "Google Chrome")
        self.assertEqual(share.safe_label([" "]), "command")

    def test_record_for_run_and_bench(self):
        from joules import share
        run = {"kind": "run", "command": ["python", "train.py", "--secret", "x"], "started_at": "2026-10-06T10:00:00+00:00",
               "duration_s": 100.0, "exit_code": 0, "simulated": False, "host": {"cpu": "AMD EPYC"},
               "devices": [{"name": "GPU 0 NVIDIA H100 80GB HBM3", "kind": "gpu", "energy_j": 5e4, "avg_w": 500},
                           {"name": "GPU 1 NVIDIA H100 80GB HBM3", "kind": "gpu", "energy_j": 5e4, "avg_w": 500}],
               "totals": {"devices_j": 1e5, "facility_j": 1.2e5, "pue": 1.2, "kwh": 1.2e5 / 3.6e6, "cost": {"price_per_kwh": 0.1, "amount": 0.0033},
                          "co2": {"g_per_kwh": 370, "grams": 12.3}, "landauer": {"operations": 1e17, "gap": 4.2e8}}}
        r = share.track_record(run, "llama-ft")
        self.assertEqual((r["project"], r["kind"], r["label"], r["device"], r["device_count"]), ("llama-ft", "run", "python train.py", "2× NVIDIA H100 80GB HBM3", 2))
        self.assertEqual((r["energy_j"], r["gap"], r["operations"], r["exit_code"]), (1.2e5, 4.2e8, 1e17, 0))
        self.assertNotIn("secret", json.dumps(r))
        self.assertEqual(share.track_record(run, "p", "nightly finetune")["label"], "nightly finetune")
        self.assertEqual(r["attribution"], "whole_gpu")  # one command, whole devices: says so
        b = bench_report(totals={"devices_j": 7000.0, "facility_j": 7000.0, "kwh": 7000 / 3.6e6, "cost": {"price_per_kwh": 0.1, "amount": 0.0002},
                                 "co2": {"g_per_kwh": 370, "grams": 0.7}}, bench={"target": "ollama:llama3.1:8b", "output_tokens": 1000,
                                 "tokens_per_s": 100.0, "joules_per_output_token": 7.0})
        r = share.track_record(b, "chatbot")
        self.assertEqual((r["kind"], r["label"], r["output_tokens"], r["j_per_token"]), ("bench", "ollama:llama3.1:8b", 1000, 7.0))
        with self.assertRaisesRegex(ValueError, "project names"):
            share.track_record(run, "bad name!")

    def test_run_track_posts_and_keeps_exit_code(self):
        H = board_handler(201, {"id": 7}); srv = _Server(H)
        try:
            code, _, err = run_cli(["--track", "nightly", "--api-url", srv.url, "--", sys.executable, "-c", "import sys; sys.exit(3)"],
                                   {**SIM, "LANDAUER_API_KEY": self.KEY})
        finally:
            srv.close()
        self.assertEqual(code, 3, err)
        path, auth, body = H.calls[0]
        self.assertEqual((path, auth), ("/api/v1/track", "Bearer " + self.KEY))
        self.assertEqual((body["project"], body["kind"], body["exit_code"], body["simulated"]), ("nightly", "run", 3, True))
        self.assertNotIn("sys.exit", body["label"])
        self.assertIn("#/track?project=nightly", err)

    def test_track_reports_team_and_energy_alerts(self):
        H = board_handler(201, {"id": 8, "team": True, "alerts": [{"project": "nightly", "change_pct": 41.6, "threshold_pct": 25}]}); srv = _Server(H)
        try:
            code, _, err = run_cli(["--track", "nightly", "--api-url", srv.url, "--", sys.executable, "-c", "pass"], {**SIM, "LANDAUER_API_KEY": self.KEY})
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        self.assertIn("tracked in your team's project nightly", err)
        self.assertIn("energy alert: this run used 42% more than the recent median", err)
        self.assertIn("set at 25%", err)

    def test_track_problems_never_change_the_exit_code(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANDAUER_API_KEY", None)
            code, _, err = run_cli(["run", "--track", "p", "--", sys.executable, "-c", "pass"], SIM)
            self.assertEqual(code, 0); self.assertIn("LANDAUER_API_KEY", err)
            code, _, err = run_cli(["run", "--track", "bad name", "--key", self.KEY, "--", sys.executable, "-c", "import sys; sys.exit(4)"], SIM)
            self.assertEqual(code, 4); self.assertIn("project names", err)

    def test_bench_track(self):
        H = handler("ollama"); srv = _Server(H)
        B = board_handler(201, {"id": 1}); board = _Server(B)
        try:
            code, _, err = run_cli(["bench", "ollama:llama3.1:8b", "--url", srv.url, "--runs", "1", "--track", "chat", "--key", self.KEY,
                                    "--api-url", board.url], SIM)
        finally:
            srv.close(); board.close()
        self.assertEqual(code, 0, err)
        self.assertEqual(B.calls[0][2]["kind"], "bench"); self.assertEqual(B.calls[0][2]["label"], "ollama:llama3.1:8b")



# ---------------------------------------------------------------- joules proxy
def model_server():
    """A threaded stand-in model server: OpenAI JSON, OpenAI SSE stream, Ollama NDJSON stream."""
    from http.server import ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(body.get("sleep", 0.3))
            if self.path == "/v1/chat/completions" and not body.get("stream"):
                data = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 30}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                return
            sse = self.path == "/v1/chat/completions"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if sse else "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            def chunk(b):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b)); self.wfile.flush()
            for i in range(5):
                if sse:
                    chunk(b'data: {"choices":[{"delta":{"content":"tok"}}]}\n\n')
                else:
                    chunk(json.dumps({"response": "tok", "done": False}).encode() + b"\n")
                time.sleep(0.05)
            if sse:
                if body.get("usage", True):
                    chunk(b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n')
                chunk(b"data: [DONE]\n\n")
            else:
                chunk(json.dumps({"done": True, "eval_count": 5, "prompt_eval_count": 3}).encode() + b"\n")
            self.wfile.write(b"0\r\n\r\n")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Proxy(unittest.TestCase):
    def setUp(self):
        from joules import proxy
        self.up = model_server()
        self.p = proxy.ProxyServer(("127.0.0.1", 0), f"http://127.0.0.1:{self.up.server_port}",
                                   meters.parse_simulation("gpu:Test GPU:100"), interval=0.02)
        threading.Thread(target=self.p.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.p.server_address[1]}"

    def tearDown(self):
        self.p.shutdown(); self.p.server_close()
        self.up.shutdown(); self.up.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.headers, r.read()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=10) as r:
            return r.read().decode()

    def test_receipt_headers_on_a_normal_response(self):
        h, data = self.post("/v1/chat/completions", {"model": "llama3", "messages": []})
        self.assertEqual(json.loads(data)["usage"]["completion_tokens"], 30)  # body untouched
        e = float(h["X-Energy-Joules"])
        self.assertEqual(h["X-Energy-Output-Tokens"], "30")
        self.assertAlmostEqual(float(h["X-Energy-Joules-Per-Token"]), e / 30, places=3)
        r = json.loads(self.get(h["X-Energy-Receipt"]))
        # 100 W for as long as the request took (the server waits 0.3 s; a slow machine adds to it), to within
        # two 0.02 s samples
        self.assertGreaterEqual(r["duration_s"], 0.3)
        self.assertAlmostEqual(e, 100 * r["duration_s"], delta=100 * 0.02 * 2)
        self.assertEqual((r["model"], r["status"], r["output_tokens"], r["prompt_tokens"], r["tokens_exact"]), ("llama3", 200, 30, 7, True))
        self.assertIn("X-Energy-Joules", h["Access-Control-Expose-Headers"])

    def test_streams_pass_through_and_get_a_receipt(self):
        h, data = self.post("/v1/chat/completions", {"model": "m", "stream": True, "sleep": 0.1})
        self.assertEqual(data.count(b'"content":"tok"'), 5)
        self.assertTrue(data.rstrip().endswith(b"data: [DONE]"))
        r = json.loads(self.get(h["X-Energy-Receipt"]))
        self.assertEqual((r["output_tokens"], r["tokens_exact"]), (5, True))
        self.assertGreater(r["energy_j"], 20)  # 0.1 s wait + 5 × 0.05 s of streaming at 100 W
        h, data = self.post("/api/generate", {"model": "llama3", "stream": True, "sleep": 0.05})  # Ollama NDJSON
        r = json.loads(self.get(h["X-Energy-Receipt"]))
        self.assertEqual((r["output_tokens"], r["prompt_tokens"]), (5, 3))
        h, _ = self.post("/v1/chat/completions", {"model": "m", "stream": True, "sleep": 0.01, "usage": False})
        r = json.loads(self.get(h["X-Energy-Receipt"]))
        self.assertEqual((r["output_tokens"], r["tokens_exact"]), (5, False))  # counted from the chunks

    def test_concurrent_requests_share_the_energy(self):
        out = []
        ts = [threading.Thread(target=lambda: out.append(self.post("/v1/chat/completions", {"model": "m", "sleep": 0.5})[0])) for _ in range(2)]
        t0 = time.monotonic()
        [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.monotonic() - t0
        es = [float(h["X-Energy-Joules"]) for h in out]
        ds = [json.loads(self.get(h["X-Energy-Receipt"]))["duration_s"] for h in out]
        slack = 100 * 0.02 * 2  # two 0.02 s samples
        # 100 W while either ran, counted once in total, not once per request (timings from the machine, which may be slow)
        self.assertGreaterEqual(sum(es), 100 * 0.5 - slack)
        self.assertLessEqual(sum(es), 100 * wall + slack)
        self.assertLess(sum(es), 100 * sum(ds) - 100 * 0.25)   # clearly not double-counted
        for e, d in zip(es, ds):
            self.assertGreaterEqual(e, 100 * 0.5 / 2 - slack)  # at least its half while both ran
            self.assertLessEqual(e, 100 * d + slack)            # at most the whole device for its own time
        self.assertEqual(json.loads(self.get("/joules/receipts"))["receipts"][0]["concurrency"], 2)

    def test_metrics_health_and_errors(self):
        self.post("/v1/chat/completions", {"model": "m", "sleep": 0.05})
        m = self.get("/joules/metrics")
        self.assertIn("joules_requests_total 1", m)
        self.assertIn("joules_output_tokens_total 30", m)
        self.assertIn('joules_device_energy_joules_total{device="GPU Test GPU",kind="gpu",method="SIMULATED"}', m)
        self.assertTrue(json.loads(self.get("/joules/health"))["ok"])
        with self.assertRaises(urllib.error.HTTPError) as c:
            self.get("/joules/receipts/r-999999")
        self.assertEqual(c.exception.code, 404)
        self.up.shutdown(); self.up.server_close()
        with self.assertRaises(urllib.error.HTTPError) as c:
            self.post("/v1/chat/completions", {"model": "m"})
        self.assertEqual(c.exception.code, 502)
        self.up = model_server()  # for tearDown

    def test_tracking_sends_one_summary_per_model(self):
        from joules import proxy
        self.post("/v1/chat/completions", {"model": "llama3", "sleep": 0.05})
        self.post("/v1/chat/completions", {"model": "llama3", "sleep": 0.05})
        sent = []
        with mock.patch.object(proxy.sharing, "send_track", lambda rec, key, url: sent.append(rec)):
            t = proxy.Tracker(self.p, "serving", "k", None, 300, "Test GPU", 1, "gpu")
            self.assertEqual(t.flush(), [])
            self.assertEqual(t.flush(), [])  # nothing new
        self.assertEqual(len(sent), 1)
        r = sent[0]
        self.assertEqual((r["project"], r["kind"], r["output_tokens"], r["label"]), ("serving", "bench", 60, "proxy · llama3 · 2 requests"))


# ---------------------------------------------------------------- joules ci
class Ci(unittest.TestCase):
    def repo(self, t):
        g = lambda *a: subprocess.run(["git", *a], cwd=t, check=True, capture_output=True)
        g("init", "-q", "-b", "main"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
        Path(t, "sub").mkdir()
        # Far apart, so a slow interpreter start on a busy runner (seen: Intel Macs) never makes the runs overlap
        Path(t, "sub/work.py").write_text("import time; time.sleep(0.05)\n")
        g("add", "."); g("commit", "-qm", "base")
        g("checkout", "-qb", "feature")
        Path(t, "sub/work.py").write_text("import time; time.sleep(1.5)\n")
        g("commit", "-qam", "slower")

    def test_compares_with_base_writes_summary_and_fails_over_budget(self):
        with tempfile.TemporaryDirectory() as t:
            self.repo(t)
            summ, outp = Path(t, "summary.md"), Path(t, "out.txt")
            env = {**SIM, "GITHUB_STEP_SUMMARY": str(summ), "GITHUB_OUTPUT": str(outp)}
            old = os.getcwd(); os.chdir(Path(t, "sub"))  # a subdirectory maps to the same one in the base
            try:
                code, out, err = run_cli(["ci", "--base", "main", "--runs", "2", "--budget", "10", "--", sys.executable, "work.py"], env)
            finally:
                os.chdir(old)
            self.assertEqual(code, 1, err)
            self.assertIn("over the 10% budget", out)
            self.assertIn(ci_marker(), summ.read_text(encoding="utf-8"))
            o = dict(l.split("=", 1) for l in outp.read_text(encoding="utf-8").split())
            self.assertGreater(float(o["change-pct"]), 100)
            self.assertEqual(o["over-budget"], "true")
            wt = subprocess.run(["git", "worktree", "list"], cwd=t, capture_output=True, text=True).stdout
            self.assertEqual(len(wt.strip().splitlines()), 1)  # temporary checkout removed

    @unittest.skipIf(os.name == "nt", "symlinks need extra rights on Windows; the Windows job covers the real case")
    def test_works_when_the_folder_has_two_names(self):
        # Windows can name a folder C:\Users\RUNNER~1\... while git says C:/Users/runneradmin/...; a symlink gives
        # the same mismatch here: the current folder through the link, git's top level through the real path.
        with tempfile.TemporaryDirectory() as t:
            real = Path(t, "real"); real.mkdir(); self.repo(real)
            link = Path(t, "link"); os.symlink(real, link)
            old = os.getcwd(); os.chdir(real / "sub")
            try:
                with mock.patch.object(cli.os, "getcwd", return_value=str(link / "sub")):
                    code, out, err = run_cli(["ci", "--base", "main", "--runs", "1", "--", sys.executable, "work.py"], SIM)
            finally:
                os.chdir(old)
        self.assertNotIn("does not exist on main", err)
        self.assertIn("work.py", out)

    def test_noise_is_not_a_regression_and_failures_are_reported(self):
        from joules import ci
        s = ci.summarize([{"energy_j": 10.5, "duration_s": 1}, {"energy_j": 9.0, "duration_s": 1}],
                         [{"energy_j": 9.5, "duration_s": 1}, {"energy_j": 10.0, "duration_s": 1}], budget=1)
        self.assertTrue(s["within_noise"]); self.assertFalse(s["over_budget"])
        with tempfile.TemporaryDirectory() as t:
            self.repo(t)
            old = os.getcwd(); os.chdir(t)
            try:
                code, _, err = run_cli(["ci", "--base", "main", "--runs", "1", "--", sys.executable, "-c", "raise SystemExit(3)"], SIM)
            finally:
                os.chdir(old)
            self.assertEqual(code, 1); self.assertIn("exit 3", err)

    def test_pull_request_comment_is_created_then_updated(self):
        from joules import ci
        class H(BaseHTTPRequestHandler):
            calls, comments = [], []
            def log_message(self, *a): pass
            def reply(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                H.calls.append(("GET", self.path, self.headers["Authorization"])); self.reply(H.comments)
            def body(self):
                return json.loads(self.rfile.read(int(self.headers["Content-Length"])))["body"]
            def do_POST(self):
                b = self.body(); H.calls.append(("POST", self.path)); H.comments.append({"id": 9, "body": b}); self.reply({})
            def do_PATCH(self):
                b = self.body(); H.calls.append(("PATCH", self.path)); H.comments[0]["body"] = b; self.reply({})
        s = _Server(H)
        with tempfile.TemporaryDirectory() as t:
            ev = Path(t, "event.json"); ev.write_text(json.dumps({"pull_request": {"number": 12}}))
            env = {"GITHUB_TOKEN": "tok", "GITHUB_REPOSITORY": "o/r", "GITHUB_API_URL": s.url, "GITHUB_EVENT_PATH": str(ev)}
            try:
                self.assertEqual(ci.post_comment(ci.MARKER + " one", env), "commented on #12")
                self.assertEqual(ci.post_comment(ci.MARKER + " two", env), "updated the energy comment on #12")
            finally:
                s.close()
            self.assertEqual([c[:2] for c in H.calls], [("GET", "/repos/o/r/issues/12/comments?per_page=100&page=1"), ("POST", "/repos/o/r/issues/12/comments"),
                                                        ("GET", "/repos/o/r/issues/12/comments?per_page=100&page=1"), ("PATCH", "/repos/o/r/issues/comments/9")])
            self.assertEqual(H.calls[0][2], "Bearer tok")
            self.assertTrue(H.comments[0]["body"].endswith("two"))
            self.assertIn("skipped", ci.post_comment("x", {}))


def ci_marker():
    from joules import ci
    return ci.MARKER


class Estimate(unittest.TestCase):
    def test_cpu_time_estimate_only_when_asked_and_never_shared(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(meters, "NVML_FACTORY", lambda: (_ for _ in ()).throw(OSError())), \
                mock.patch.object(meters.platform, "system", lambda: "Linux"):
            env = {"JOULES_SYSFS": t, "JOULES_CPU_W": "5"}
            with mock.patch.dict(os.environ, env):
                os.environ.pop("JOULES_SIMULATE", None)
                self.assertEqual(meters.discover().meters, [])
                code, out, err = run_cli(["run", "--estimate", "--json", "--", sys.executable, "-c", "x=0\nfor i in range(2_000_000): x+=i"])
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        dev = r["devices"][0]
        self.assertEqual((dev["kind"], dev["method"]), ("est", "ESTIMATE (CPU time)"))
        self.assertGreater(dev["energy_j"], 0.1)
        self.assertLess(dev["energy_j"], 5 * r["duration_s"] * (os.cpu_count() or 1) + 0.5)
        r["kind"] = "bench"; r["bench"] = {"target": "ollama:x", "output_tokens": 100}
        from joules import share
        with self.assertRaises(ValueError):
            share.measurement(r)


# ---------------------------------------------------------------- joules agent
from joules import agent as ag  # noqa: E402

POD = "1234abcd_5678_90ab_cdef_1234567890ab"
CTR = "ab" * 32


def fake_proc(root: Path, pid: int, comm: str, ticks: int, cgroup: str = "0::/user.slice", cmdline=(), env=(), ppid: int = 1, uid: int = 1000, start: int = 5000):
    d = root / str(pid)
    write(d / "stat", f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194304 10 0 0 0 {ticks - ticks // 4} {ticks // 4} 0 0 20 0 1 0 {start} 1000 100")
    write(d / "status", f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    write(d / "cgroup", cgroup + "\n")
    (d / "cmdline").write_bytes(b"\0".join(c.encode() for c in cmdline) + (b"\0" if cmdline else b""))
    (d / "environ").write_bytes(b"\0".join(e.encode() for e in env))


def cluster(root: Path, ticks=(100, 100, 100, 100)):
    fake_proc(root, 100, "python", ticks[0], "0::/system.slice/slurmstepd.scope/job_4711/step_0/user/task_0",
              ["python", "train.py"], ["SLURM_JOB_NAME=llama-ft", "SLURM_JOB_USER=ada", "AWS_SECRET_ACCESS_KEY=hunter2"])
    fake_proc(root, 200, "vllm (engine)", ticks[1], f"0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod{POD}.slice/cri-containerd-{CTR}.scope",
              ["python3", "-m", "vllm.entrypoints.api_server"], ["HOSTNAME=chatbot-7f9c", "POD_NAMESPACE=prod", "HF_TOKEN=hf_secret"])
    fake_proc(root, 300, "node", ticks[2], f"0::/system.slice/docker-{CTR}.scope", ["node", "server.js"], ["HOSTNAME=web1"])
    fake_proc(root, 400, "python", ticks[3], "0::/user.slice/user-1000.slice/session-3.scope", ["python", "eval.py", "--api-key=sk-secret"])
    fake_proc(root, 2, "kthreadd", 50, ppid=0, uid=0)
    fake_proc(root, 500, "kworker/0:1", 50, ppid=2, uid=0)
    write(root / "self" / "stat", "not a pid")


class FakeMeter(meters.Meter):
    def __init__(self, name, kind, watts, nvml=None, h=None):
        super().__init__(name)
        self.kind, self.w, self.method = kind, watts, "test"
        if nvml is not None:
            self.nvml, self.h = nvml, h
    def start(self, now):
        self.t0 = self.t = now
    def poll(self, now):
        self.energy_j += self.w * (now - self.t); self.t = now


class FakeNvmlProcs:
    def __init__(self, procs, util):
        self.p, self.u = procs, util
    def processes(self, h): return self.p.get(h, [])
    def utilization(self, h): return self.u.get(h)


@unittest.skipIf(os.name == "nt", "joules agent is Linux only")
class AgentJobs(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        cluster(self.root)
        self.env = mock.patch.dict(os.environ, {"JOULES_PROC": str(self.root)})
        self.env.start()
    def tearDown(self):
        self.env.stop(); self.t.cleanup()

    def test_finds_slurm_kubernetes_container_and_user_programs(self):
        procs = ag.scan_processes()
        self.assertEqual(sorted(procs), [100, 200, 300, 400])  # no kernel threads, no "self"
        jobs = {p: procs[p].job for p in procs}
        self.assertEqual((jobs[100].kind, jobs[100].id, jobs[100].name, jobs[100].user), ("slurm", "4711", "llama-ft", "ada"))
        self.assertEqual((jobs[200].kind, jobs[200].id, jobs[200].name, jobs[200].namespace), ("k8s", POD.replace("_", "-"), "chatbot-7f9c", "prod"))
        self.assertEqual((jobs[300].kind, jobs[300].id, jobs[300].name), ("container", CTR[:12], "web1"))
        self.assertEqual((jobs[400].kind, jobs[400].name), ("process", "python eval.py"))
        self.assertEqual(procs[200].ticks, 100)
        flat = json.dumps([vars(j) for j in jobs.values()])
        for secret in ("hunter2", "hf_secret", "sk-secret", "api-key"):
            self.assertNotIn(secret, flat)
        self.assertEqual(set(ag._env(100)), {"SLURM_JOB_NAME", "SLURM_JOB_USER"})

    def test_cache_reads_each_process_once_and_notices_reused_pids(self):
        cache = {}
        ag.scan_processes(cache)
        with mock.patch.object(ag, "job_of", side_effect=AssertionError("looked up again")):
            ag.scan_processes(cache)
        fake_proc(self.root, 400, "python", 10, cmdline=["python", "other.py"], start=9999)  # pid reused by a new program
        self.assertEqual(ag.scan_processes(cache)[400].job.name, "python other.py")
        self.assertEqual(len(cache), 4)

    def test_splits_cpu_by_time_and_gpu_by_memory(self):
        acc = ag.Accountant(idle_util=5, finish_after=30)
        cpu, g0, g1, g2 = (FakeMeter("CPU package 0", "cpu", 0), FakeMeter("GPU 0 H100", "gpu", 0),
                           FakeMeter("GPU 1 H100", "gpu", 0), FakeMeter("GPU 2 H100", "gpu", 0))
        acc.step(0, 0, ag.scan_processes(), [], {})
        cluster(self.root, ticks=(130, 110, 100, 100))  # slurm job 30 ticks, pod 10
        info = {g0.name: {"procs": [(100, 4 << 30), (200, 12 << 30)], "util": 87},
                g1.name: {"procs": [], "util": 0},
                g2.name: {"procs": [(200, 1 << 30)], "util": 2}}
        acc.step(1, 1, ag.scan_processes(), [(cpu, 100.0), (g0, 600.0), (g1, 300.0), (g2, 200.0)], info)
        j = {k: v for k, v in acc.jobs.items()}
        slurm, pod, idle = j["slurm:4711"], j["k8s:" + POD.replace("_", "-")], j["idle:idle-gpu"]
        self.assertAlmostEqual(slurm.energy_j, 75 + 150)
        self.assertAlmostEqual(pod.energy_j, 25 + 450 + 200)
        self.assertAlmostEqual(pod.idle_j, 200)          # GPU 2 was 2 % busy
        self.assertAlmostEqual(idle.energy_j, 300)        # nobody on GPU 1
        self.assertEqual(slurm.gpu_devices, {"GPU 0 H100"})
        self.assertEqual(slurm.kinds, {"cpu", "gpu"})
        self.assertAlmostEqual(slurm.cpu_s, 30 / ag.CLK_TCK)
        self.assertAlmostEqual(sum(x.energy_j for x in acc.jobs.values()), 1200)  # nothing lost
        self.assertNotIn("container:" + CTR[:12], j)      # used no CPU, holds no GPU
        acc.step(20, 19, ag.scan_processes(), [], {})
        self.assertEqual(acc.take_finished(), [])
        acc.step(40, 20, ag.scan_processes(), [], {})
        done = {x.key.key for x in acc.take_finished()}
        self.assertEqual(done, {"slurm:4711", "k8s:" + POD.replace("_", "-")})  # idle is never "finished"
        self.assertIn("idle:idle-gpu", acc.jobs)

    def test_gpu_without_per_process_data_goes_to_the_main_job(self):
        acc = ag.Accountant()
        amd = FakeMeter("GPU 0 MI300X", "gpu", 0)
        acc.step(0, 0, ag.scan_processes(), [], {})
        cluster(self.root, ticks=(150, 110, 100, 100))  # the Slurm job does most of the CPU work
        acc.step(1, 1, ag.scan_processes(), [(amd, 700.0)], {})
        self.assertAlmostEqual(acc.jobs["slurm:4711"].energy_j, 700)
        self.assertEqual(acc.jobs["slurm:4711"].gpu_devices, {"GPU 0 MI300X"})
        cluster(self.root, ticks=(160, 120, 110, 110))  # four equally busy jobs: cannot tell
        acc.step(2, 1, ag.scan_processes(), [(amd, 700.0)], {})
        self.assertAlmostEqual(acc.jobs["system:system"].energy_j, 700)

    def test_amd_and_intel_gpus_split_by_gpu_time(self):
        def card(pid, fd, pdev, client, gfx_ns, extra=""):
            d = self.root / str(pid)
            (d / "fd").mkdir(exist_ok=True); (d / "fdinfo").mkdir(exist_ok=True)
            link = d / "fd" / str(fd)
            if not link.is_symlink():
                os.symlink("/dev/dri/renderD128", link)
            write(d / "fdinfo" / str(fd), f"pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-pdev:\t{pdev}\n"
                                          f"drm-client-id:\t{client}\ndrm-engine-gfx:\t{gfx_ns} ns\ndrm-engine-compute:\t0 ns\n{extra}")
        (self.root / "100" / "fd").mkdir()
        os.symlink("/home/ada/data.bin", self.root / "100" / "fd" / "3")
        card(100, 7, "0000:03:00.0", 11, 1_000_000)
        card(100, 8, "0000:03:00.0", 11, 1_000_000)      # the same client through a dup'd fd: counted once
        card(200, 5, "0000:03:00.0", 12, 0)
        card(300, 5, "0000:83:00.0", 13, 0)              # holds GPU 1 open, never uses it
        mi = FakeMeter("GPU 0 MI300X", "gpu", 0); mi.pci = "0000:03:00.0"
        arc = FakeMeter("GPU 1 Arc", "gpu", 0); arc.pci = "0000:83:00.0"
        a = ag.Agent([mi, arc], interval=1)
        a.start(0, wall=0)
        card(100, 7, "0000:03:00.0", 11, 4_000_000)      # +3 ms
        card(100, 8, "0000:03:00.0", 11, 4_000_000)
        card(200, 5, "0000:03:00.0", 12, 1_000_000)      # +1 ms
        card(300, 5, "0000:83:00.0", 13, 0)
        mi.w, arc.w = 800, 50
        a.tick(1, 1, wall=1)
        jobs = a.acc.jobs
        self.assertAlmostEqual(jobs["slurm:4711"].energy_j, 600)
        self.assertAlmostEqual(jobs["k8s:" + POD.replace("_", "-")].energy_j, 200)
        self.assertAlmostEqual(jobs["container:" + CTR[:12]].idle_j, 50)
        self.assertEqual(jobs["slurm:4711"].gpu_devices, {"GPU 0 MI300X"})
        self.assertEqual(sorted(a.drm.fds[100][1]), ["7", "8"])  # not fd 3, an ordinary file

    def test_listening_never_waits_on_dns(self):
        # http.server looks up the listening address's name (socket.getfqdn), which hangs for half a minute on a
        # Mac whose DNS is slow (seen on GitHub's macOS runners); the agent and the proxy must start at once
        def hang(*a):
            raise AssertionError("getfqdn called")
        with mock.patch.object(socket, "getfqdn", hang):
            srv = ag.AgentServer(("127.0.0.1", 0), ag.Agent([FakeMeter("CPU package 0", "cpu", 100)], host="node-7"))
            self.assertEqual(srv.server_name, "127.0.0.1"); self.assertGreater(srv.server_port, 0)
            srv.server_close()
            from joules import proxy
            p = proxy.ProxyServer(("127.0.0.1", 0), "http://127.0.0.1:9", [FakeMeter("CPU package 0", "cpu", 100)])
            p.server_close()

    def test_metrics_jobs_and_health_over_http(self):
        nv = FakeNvmlProcs({0: [(100, 8 << 30)]}, {0: 91})
        ms = [FakeMeter('GPU 0 "H100"', "gpu", 700, nv, 0), FakeMeter("CPU package 0", "cpu", 100)]
        a = ag.Agent(ms, interval=0.05, price=0.2, g_per_kwh=400, pue=1.5, host="node-7")
        a.start(0, wall=1_759_999_990)
        cluster(self.root, ticks=(200, 100, 100, 100))
        a.tick(10, 10, wall=1_760_000_000)
        srv = ag.AgentServer(("127.0.0.1", 0), a)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            text = urllib.request.urlopen(base + "/metrics").read().decode()
            jobs = json.loads(urllib.request.urlopen(base + "/jobs").read())
            health = json.loads(urllib.request.urlopen(base + "/health").read())
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + "/nope")
            self.assertEqual(e.exception.code, 404)
        finally:
            srv.shutdown(); srv.server_close()
        self.assertIn('joules_device_energy_joules_total{host="node-7",device="GPU 0 \\"H100\\""} 7000', text)
        self.assertIn('joules_gpu_utilization_percent{host="node-7",device="GPU 0 \\"H100\\""} 91', text)
        self.assertRegex(text, r'joules_job_energy_joules_total\{host="node-7",kind="slurm",job="4711",name="llama-ft",user="ada",namespace=""\} 8000\n')
        self.assertIn("# TYPE joules_job_cost gauge", text)
        top = jobs["jobs"][0]
        self.assertEqual((top["kind"], top["name"], top["energy_j"], top["facility_j"]), ("slurm", "llama-ft", 8000, 12000))
        self.assertAlmostEqual(top["cost"], 12000 / 3.6e6 * 0.2)
        self.assertEqual(top["started"], 1_759_999_990)  # it was working in the first interval
        self.assertEqual(health["devices"], ['GPU 0 "H100"', "CPU package 0"])

    def test_live_beat_is_the_node_now_with_idle_gpus_and_jobs(self):
        cpu, g0, g1, g2 = (FakeMeter("CPU package 0", "cpu", 0), FakeMeter("GPU 0 H100", "gpu", 0),
                           FakeMeter("GPU 1 H100", "gpu", 0), FakeMeter("GPU 2 H100", "gpu", 0))
        a = ag.Agent([cpu, g0, g1, g2], price=0.12, g_per_kwh=370, pue=1.5, host="gpu node 7!")
        a.acc.step(0, 0, ag.scan_processes(), [], {})
        cluster(self.root, ticks=(130, 110, 100, 100))
        info = {g0.name: {"procs": [(100, 4 << 30), (200, 12 << 30)], "util": 87},
                g1.name: {"procs": [], "util": 0},
                g2.name: {"procs": [(200, 1 << 30)], "util": 2}}
        a.acc.step(1, 2, ag.scan_processes(), [(cpu, 200.0), (g0, 1200.0), (g1, 600.0), (g2, 400.0)], info)  # 2 s
        b = a.live_beat()
        self.assertEqual(set(b), {"node", "watts", "idle_watts", "gpus", "idle_gpus", "price_per_kwh", "g_per_kwh", "jobs", "version", "gpu_list"})
        # one reading per GPU: its own power (no --pue), how busy, idle or not, and the job using it most
        self.assertEqual(b["gpu_list"], [
            {"i": 0, "name": "H100", "watts": 600.0, "util": 87.0, "idle": False, "job": "chatbot-7f9c (prod)"},
            {"i": 1, "name": "H100", "watts": 300.0, "util": 0.0, "idle": True},
            {"i": 2, "name": "H100", "watts": 200.0, "util": 2.0, "idle": True, "job": "chatbot-7f9c (prod)"}])
        a.gpu_detail = False
        self.assertNotIn("gpu_list", a.live_beat())
        self.assertEqual(b["node"], "gpu-node-7")
        self.assertAlmostEqual(b["watts"], 1200 * 1.5)                    # 2400 J in 2 s, with the facility overhead
        self.assertEqual((b["gpus"], b["idle_gpus"]), (3, 2))              # GPU 1 has no job, GPU 2 is 2 % busy
        self.assertAlmostEqual(b["idle_watts"], (300 + 200) * 1.5)
        pod, slurm = b["jobs"]
        self.assertEqual((pod["name"], pod["kind"], pod["gpus"]), ("chatbot-7f9c (prod)", "k8s", 2))
        self.assertAlmostEqual(pod["watts"], (25 + 450 + 200) * 1.5)
        self.assertAlmostEqual(pod["idle_watts"], 200 * 1.5)
        self.assertEqual((slurm["name"], slurm["user"]), ("llama-ft", "ada"))
        self.assertAlmostEqual(slurm["watts"], (75 + 150) * 1.5)
        self.assertNotIn("idle", [j["kind"] for j in b["jobs"]])           # GPUs with no job are idle_watts, not a job
        self.assertNotIn("secret", json.dumps(b))
        self.assertEqual(ag.node_name("ip-10-0-1-7.ec2.internal"), "ip-10-0-1-7.ec2.internal")
        self.assertEqual(ag.node_name("§§"), "node")

    def test_track_record_has_no_command_arguments(self):
        acc = ag.Accountant()
        acc.step(1_760_000_000, 0, ag.scan_processes(), [], {})
        cluster(self.root, ticks=(100, 100, 100, 900))
        acc.step(1_760_000_100, 100, ag.scan_processes(), [(FakeMeter("CPU package 0", "cpu", 0), 3600.0)], {})
        j = acc.jobs["process:1000:python eval.py"]
        r = ag.track_record(j, j.as_dict(0.1, 370), "cluster", "AMD EPYC", 0, 0.1, 370)
        self.assertEqual(r["label"].split(" · ")[:2], ["process", "python eval.py"])
        self.assertNotIn("secret", json.dumps(r))
        self.assertEqual((r["sources"], r["device_count"], r["started_at"]), ("cpu", 1, "2025-10-09T08:53:20Z"))  # started in the interval it was first seen working
        from joules import share
        self.assertTrue(share.PROJECT_RE.match(r["project"]))

    def test_a_shared_gpu_is_split_by_sm_time_and_says_so(self):
        acc = ag.Accountant(interval=1)
        g = FakeMeter("GPU 0 H100", "gpu", 0)
        acc.step(0, 0, ag.scan_processes(), [], {})
        holders = [(100, 30 << 30), (200, 10 << 30)]  # the Slurm job holds 3x the memory...
        acc.step(1, 1, ag.scan_processes(), [(g, 800.0)], {g.name: {"procs": holders, "util": 90, "sm": {100: 60, 200: 180}}})
        slurm, pod = acc.jobs["slurm:4711"], acc.jobs["k8s:" + POD.replace("_", "-")]
        self.assertAlmostEqual(slurm.energy_j, 200)   # ...but the pod used 3x the SM time
        self.assertAlmostEqual(pod.energy_j, 600)
        self.assertEqual((slurm.attribution, pod.attribution), ("sm_time", "sm_time"))
        # a job that holds the card but whose SM time rounds to 0 gets 0 J, and still says it was split by SM time
        acc0 = ag.Accountant(interval=1)
        acc0.step(0, 0, ag.scan_processes(), [], {})
        acc0.step(1, 1, ag.scan_processes(), [(g, 800.0)], {g.name: {"procs": holders, "util": 90, "sm": {200: 95}}})
        idle_holder = acc0.jobs["slurm:4711"]
        self.assertEqual((idle_holder.energy_j, idle_holder.attribution), (0.0, "sm_time"))
        self.assertEqual(idle_holder.as_dict(0.1, 370)["gpus"], ["GPU 0 H100"])
        # without SM samples the same card falls back to memory share, and the job says that instead
        acc2 = ag.Accountant(interval=1)
        acc2.step(0, 0, ag.scan_processes(), [], {})
        acc2.step(1, 1, ag.scan_processes(), [(g, 800.0)], {g.name: {"procs": holders, "util": 90}})
        self.assertAlmostEqual(acc2.jobs["slurm:4711"].energy_j, 600)
        self.assertEqual(acc2.jobs["slurm:4711"].attribution, "memory_share")
        # one job alone on the card: all of it, exactly
        acc3 = ag.Accountant(interval=1)
        acc3.step(0, 0, ag.scan_processes(), [], {})
        acc3.step(1, 1, ag.scan_processes(), [(g, 800.0)], {g.name: {"procs": [(100, 1 << 30)], "util": 90, "sm": {100: 50}}})
        d = acc3.jobs["slurm:4711"].as_dict(0.1, 370)
        self.assertEqual((d["attribution"], d["attribution_j"]), ("whole_gpu", {"whole_gpu": 800.0}))

    def test_sample_coverage_counts_only_intervals_that_were_sampled(self):
        acc = ag.Accountant(interval=1)
        g = FakeMeter("GPU 0 H100", "gpu", 0)
        info = {g.name: {"procs": [(100, 1 << 30)], "util": 90}}
        acc.step(0, 0, ag.scan_processes(), [], {})
        for t in range(1, 11):
            acc.step(t, 1, ag.scan_processes(), [(g, 100.0)], info)
        self.assertEqual(acc.jobs["slurm:4711"].coverage, 1.0)
        acc.step(20, 10, ag.scan_processes(), [(g, 1000.0)], info)      # the agent stalled for 10 s: 2 s count
        acc.step(21, 1, ag.scan_processes(), [(g, 100.0)], info, complete=False)  # a meter was not read
        j = acc.jobs["slurm:4711"]
        self.assertAlmostEqual(j.coverage, 12 / 21)
        self.assertLess(j.as_dict(0.1, 370)["coverage"], 0.95)
        r = ag.track_record(j, j.as_dict(0.1, 370), "cluster", "H100", 1, 0.1, 370)
        self.assertEqual((r["attribution"], r["coverage"]), ("whole_gpu", round(12 / 21, 4)))
        self.assertAlmostEqual(j.energy_j, 2100)  # the energy itself is never lost, only the attribution is rougher

    def test_agent_reads_sm_time_from_the_driver_or_accounting(self):
        class Nv:
            def __init__(self, sm=None, acct=None):
                self.sm, self.acct, self.calls = sm, acct, []
            def processes(self, h): return [(100, 1 << 30), (200, 1 << 30)]
            def utilization(self, h): return 80
            def process_sm(self, h, since):
                self.calls.append(since)
                return None if self.sm is None else (self.sm, since + 1000)
            def accounting_sm_ms(self, h, pids):
                return None if self.acct is None else {p: self.acct.pop(0) for p in pids}
        g = FakeMeter("GPU 0 H100", "gpu", 0, nvml=Nv(sm={100: 50, 200: 150}), h=0)
        a = ag.Agent([g], interval=1)
        i1, i2 = a._gpu_info({}, 0), a._gpu_info({}, 1)
        self.assertEqual((i1[g.name]["sm"], g.nvml.calls), ({100: 50, 200: 150}, [0, 1000]))  # each read starts where the last ended
        acct = FakeMeter("GPU 1 H100", "gpu", 0, nvml=Nv(acct=[1000, 2000, 1500, 5000]), h=1)
        b = ag.Agent([acct], interval=1)
        b._gpu_info({}, 0)                                    # accounting totals so far: 1000 and 2000 ms
        self.assertEqual(b._gpu_info({}, 1)[acct.name]["sm"], {100: 500, 200: 3000})  # the change since then
        none = FakeMeter("GPU 2 T4", "gpu", 0, nvml=Nv(), h=2)
        self.assertNotIn("sm", ag.Agent([none], interval=1)._gpu_info({}, 0)[none.name])

    def test_nvml_process_utilization_is_read_safely(self):
        import ctypes

        class Lib:
            def __init__(self, samples, rc=0):
                self.samples, self.rc = samples, rc
            def nvmlDeviceGetProcessUtilization(self, h, buf, n, since):
                n = n._obj
                if self.rc:
                    return self.rc
                if buf is None:
                    n.value = len(self.samples)
                    return 7
                for i, (pid, ts, sm) in enumerate(self.samples[:n.value]):
                    buf[i].pid, buf[i].timeStamp, buf[i].smUtil = pid, ts, sm
                n.value = min(n.value, len(self.samples))
                return 0
        nv = meters.NVML.__new__(meters.NVML)
        nv.lib = Lib([(100, 10, 40), (100, 20, 60), (200, 20, 250), (300, 5, 90)])
        self.assertEqual(nv.process_sm(0, 5), ({100: 100, 200: 100}, 20))  # old samples skipped, SM % capped at 100
        nv.lib = Lib([], rc=6)
        self.assertEqual(nv.process_sm(0, 5), ({}, 5))                    # no new samples: nobody used it
        nv.lib = Lib([], rc=3)
        self.assertIsNone(nv.process_sm(0, 5))                           # not supported: fall back

    def test_old_servers_get_the_run_without_the_new_fields(self):
        from joules import share
        sent = []
        def fake_post(path, m, key, base, timeout):
            sent.append(dict(m))
            if "attribution" in m:
                class E: code = 400
                err = share.ServerError.__new__(share.ServerError)
                err.status, err.code = 400, "invalid_run"
                err.body = {"error": {"code": "invalid_run", "message": "bad", "fields": {"attribution": "Unknown field.", "coverage": "Unknown field."}}}
                raise err
            return {"id": 1}
        with mock.patch.object(share, "_post", fake_post):
            self.assertEqual(share.send_track({"label": "x", "attribution": "sm_time", "coverage": 0.99}, "k"), {"id": 1})
        self.assertEqual(sent[-1], {"label": "x"})


@unittest.skipIf(os.name == "nt", "joules agent is Linux only")
class AgentCli(unittest.TestCase):
    def test_tracks_a_finished_slurm_job(self):
        got = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                got.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                data = b'{"ok": true}'
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
        srv = _Server(H)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            cluster(root)

            def job():  # the Slurm job works for a moment, then ends
                time.sleep(0.6); cluster(root, ticks=(400, 100, 100, 100))
                time.sleep(0.6); shutil.rmtree(root / "100")
            threading.Thread(target=job, daemon=True).start()
            try:
                code, out, err = run_cli(["agent", "--port", "0", "--interval", "0.2", "--duration", "2.6", "--finish-after", "0.5",
                                          "--min-seconds", "0", "--track", "cluster", "--key", "lg_test", "--api-url", srv.url],
                                         {**SIM, "JOULES_PROC": t})
            finally:
                srv.close()
        self.assertEqual(code, 0, err)
        self.assertIn("finished slurm llama-ft (ada)", err)
        slurm = [b for p, k, b in got if b["label"].startswith("slurm")]
        self.assertEqual(len(slurm), 1, got)
        r = slurm[0]
        self.assertEqual((got[0][0], got[0][1]), ("/api/v1/track", "Bearer lg_test"))
        self.assertEqual((r["project"], r["label"], r["kind"], r["simulated"], r["device"], r["device_count"]),
                         ("cluster", "slurm · llama-ft · ada", "run", True, "GPU H100 SXM", 1))
        self.assertEqual(r["sources"], "cpu+gpu")
        self.assertGreater(r["energy_j"], 0)

    def test_a_server_that_does_not_take_per_gpu_readings_still_gets_heartbeats(self):
        got = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                got.append(body)
                if "gpu_list" in body:  # a WattOps server from before per-GPU Live
                    code, data = 400, b'{"error": {"code": "invalid_beat", "message": "Some fields are not valid.", "fields": {"gpu_list": "Unknown field."}}}'
                else:
                    code, data = 202, b'{"ok": true, "skipped": false, "page": "/#/live"}'
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
        srv = _Server(H)
        with tempfile.TemporaryDirectory() as t:
            cluster(Path(t), ticks=(400, 100, 100, 100))
            try:
                code, _, err = run_cli(["agent", "--port", "0", "--interval", "0.2", "--duration", "6.6", "--live-every", "2",
                                        "--track", "cluster", "--key", "lg_test", "--api-url", srv.url], {**SIM, "JOULES_PROC": t})
            finally:
                srv.close()
        self.assertEqual(code, 0, err)
        beats = [b for b in got if "node" in b]
        self.assertGreaterEqual(len(beats), 3, got)
        self.assertIn("gpu_list", beats[0]); self.assertEqual(beats[0]["gpu_list"][0]["name"], "GPU H100 SXM")
        self.assertTrue(all("gpu_list" not in b for b in beats[1:]), "after one refusal, the machine's totals only")
        self.assertIn("does not take per-GPU readings yet", err)
        self.assertNotIn("Live heartbeat failed", err)

    def test_live_heartbeats_go_to_the_live_page_and_stop_without_max(self):
        def serve(plan_ok):
            got = []

            class H(BaseHTTPRequestHandler):
                def log_message(self, *a): pass
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    got.append((self.path, body))
                    live = self.path.endswith("live=1")
                    code, data = ((202, b'{"ok": true, "skipped": false, "page": "/#/live"}') if plan_ok is True or not live else
                                  (403, b'{"error": {"code": "gpu_limit", "message": "Live on Free shows up to 2 GPUs at once; 2 GPUs are on Live already, so this machine (1 GPU) is not shown."}}')
                                  if plan_ok == "limit" else
                                  (403, b'{"error": {"code": "plan_required", "message": "Live is part of the Max plan."}}'))
                    self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); self.wfile.write(data)
            return _Server(H), got

        for plan_ok in (True, False, "limit"):
            srv, got = serve(plan_ok)
            with tempfile.TemporaryDirectory() as t:
                cluster(Path(t), ticks=(400, 100, 100, 100))
                try:
                    code, _, err = run_cli(["agent", "--port", "0", "--interval", "0.2", "--duration", "4.6", "--live-every", "2",
                                            "--track", "cluster", "--key", "lg_test", "--api-url", srv.url], {**SIM, "JOULES_PROC": t})
                finally:
                    srv.close()
            self.assertEqual(code, 0, err)
            beats = [b for p, b in got if p == "/api/v1/track?live=1"]
            if plan_ok == "limit":  # over the plan's GPUs on Live: said once, tried again a minute later
                self.assertEqual(len(beats), 1, got)
                self.assertEqual(err.count("not on Live: Live on Free shows up to 2 GPUs at once; 2 GPUs are on Live already, so this "
                                           "machine (1 GPU) is not shown. Trying again every minute."), 1, err)
                from joules import connection as cn
                self.assertEqual(cn.live_status(lambda: err, lambda: True, 0),
                                 "limit: Live on Free shows up to 2 GPUs at once; 2 GPUs are on Live already, so this machine (1 GPU) is not shown.")
            elif plan_ok:
                self.assertEqual(len(beats), 2, got)
                self.assertIn(f"live: {srv.url}/#/live", err)
                self.assertGreater(beats[-1]["watts"], 600)          # the simulated H100 and CPU, all the time
                self.assertEqual((beats[-1]["gpus"], beats[-1]["version"]), (1, __import__("joules").__version__))
            else:
                self.assertEqual(len(beats), 1, "no more heartbeats once the server says Live is not in the plan")
                self.assertIn("Live is part of the Max plan", err)
                self.assertNotIn("live: ", err)
        code, _, err = run_cli(["agent", "--live-every", "1", "--track", "cluster", "--key", "lg_test"], {**SIM, "JOULES_PROC": tempfile.mkdtemp()})
        self.assertEqual(code, 2); self.assertIn("--live-every", err)

    def test_keep_awake_holds_off_sleep_while_the_agent_runs(self):
        with tempfile.TemporaryDirectory() as t:
            bin_, log = Path(t) / "bin", Path(t) / "log"
            for name in ("caffeinate", "systemd-inhibit"):  # stand-ins that note how they were called, then wait to be ended
                write(bin_ / name, f'#!/bin/sh\necho "$$ {name} $*" > "{log}"\nexec sleep 30\n')
                (bin_ / name).chmod(0o755)
            env = {**SIM, "JOULES_PROC": tempfile.mkdtemp(), "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}"}
            code, _, err = run_cli(["agent", "--port", "0", "--interval", "0.2", "--duration", "1", "--keep-awake"], env)
            self.assertEqual(code, 0, err)
            self.assertIn("keeping this machine awake", err)
            pid, called = log.read_text().split(" ", 1)
            self.assertIn(f"-w {os.getpid()}" if sys.platform == "darwin" else "--what=idle:sleep", called)
            with self.assertRaises(ProcessLookupError):  # ended with the agent
                os.kill(int(pid), 0)
            code, _, err = run_cli(["agent", "--port", "0", "--interval", "0.2", "--duration", "0.5", "--keep-awake"],
                                   {**env, "PATH": str(Path(t) / "empty")})
            self.assertEqual(code, 0, err)
            self.assertIn("is not available here", err)

    def test_rejects_bad_project_and_missing_key(self):
        env = {**SIM, "JOULES_PROC": tempfile.mkdtemp()}  # any system: these checks come after the /proc check
        code, _, err = run_cli(["agent", "--track", "bad project!", "--key", "x"], env)
        self.assertEqual(code, 2); self.assertIn("project names", err)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANDAUER_API_KEY", None)
            code, _, err = run_cli(["agent", "--track", "cluster"], env)
        self.assertEqual(code, 2); self.assertIn("needs an API key", err)


@unittest.skipUnless(shutil.which("gcc") and os.name == "posix", "needs gcc to build the stand-in NVIDIA library")
class NvmlProcesses(unittest.TestCase):
    def test_ctypes_processes_and_utilization(self):
        src = Path(__file__).parent / "fake_nvml" / "fake_nvml.c"
        with tempfile.TemporaryDirectory() as t:
            lib = os.path.join(t, "libnvidia-ml.so.1")
            subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", lib, str(src)], check=True)
            code = ("import ctypes, json, sys; from joules import meters\n"
                    "n = meters.NVML.__new__(meters.NVML); n.lib = ctypes.CDLL(sys.argv[1]); n.lib.nvmlInit_v2()\n"
                    "h0, h1 = n.handle(0), n.handle(1)\n"
                    "print(json.dumps([n.utilization(h0), n.processes(h0), n.processes(h1)]))")  # the H100 (index 0) is busy
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
            out = subprocess.run([sys.executable, "-c", code, lib], capture_output=True, text=True, env=env, check=True).stdout
        util, p0, p1 = json.loads(out)
        self.assertEqual(util, 87)
        self.assertEqual(p0, [[4242, 4 << 30], [4343, 12 << 30]])
        self.assertEqual(p1, [])



# ---------------------------------------------------------------- first-run experience
class Login(unittest.TestCase):
    KEY = "lgk_" + "c" * 40

    def tearDown(self):
        from joules import share
        with contextlib.suppress(FileNotFoundError):
            os.remove(share.key_file())

    @staticmethod
    def server(ok=True):
        class H(BaseHTTPRequestHandler):
            calls = []
            def log_message(self, *a): pass
            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                H.calls.append(("GET", self.path, self.headers["Authorization"]))
                self.reply(200, {"runs": []}) if ok else self.reply(401, {"error": {"code": "invalid_api_key", "message": "That API key is not valid."}})
            def do_POST(self):
                H.calls.append(("POST", self.path, self.headers["Authorization"]))
                self.rfile.read(int(self.headers["Content-Length"]))
                self.reply(201, {"id": 1})
        return H

    def test_login_saves_the_key_for_every_later_command(self):
        from joules import share
        H = self.server(); srv = _Server(H)
        try:
            code, _, err = run_cli(["login", self.KEY, "--api-url", srv.url])
            self.assertEqual(code, 0, err)
            self.assertIn("signed in", err)
            path = share.key_file()
            self.assertTrue(path.startswith(_CONFIG))
            if os.name == "posix":  # Windows has no mode bits; a file in the user's own folder is private to them
                self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
            self.assertEqual(H.calls[0], ("GET", "/api/v1/track?limit=1&days=1", "Bearer " + self.KEY))
            # no --key and no environment variable: the saved key is used
            code, _, err = run_cli(["--track", "first-test", "--api-url", srv.url, "--", sys.executable, "-c", "pass"], SIM)
            self.assertEqual(code, 0, err)
            self.assertEqual(H.calls[1], ("POST", "/api/v1/track", "Bearer " + self.KEY))
            code, _, err = run_cli(["logout"])
            self.assertFalse(os.path.exists(path))
            self.assertIsNone(share.api_key())
        finally:
            srv.close()

    def test_after_login_every_run_goes_to_the_tracking_page(self):
        H = self.server(); srv = _Server(H)
        cmd = ["--api-url", srv.url, "--", sys.executable, "-c", "pass"]
        posts = lambda: sum(1 for c in H.calls if c[0] == "POST")
        try:
            with mock.patch.dict(os.environ):
                os.environ.pop("CI", None)  # GitHub Actions sets it; a person's computer does not
                code, _, err = run_cli(["login", self.KEY, "--api-url", srv.url])
                self.assertIn("appears on your Tracking page", err)
                code, _, err = run_cli(cmd, SIM)  # no --track
                self.assertEqual(code, 0, err)
                self.assertEqual(posts(), 1)
                self.assertIn("tracked in project default", err)
                code, _, err = run_cli(["--no-track", *cmd], SIM)
                self.assertEqual(posts(), 1)
                self.assertNotIn("tracked", err)
                code, _, err = run_cli(cmd, {**SIM, "JOULES_NO_TRACK": "1"})
                self.assertEqual(posts(), 1)
                # a key from the environment (scripts, CI) tracks only when asked
                code, _, err = run_cli(cmd, {**SIM, "LANDAUER_API_KEY": self.KEY})
                self.assertEqual(posts(), 1)
                code, _, err = run_cli(cmd, {**SIM, "CI": "true"})
                self.assertEqual(posts(), 1)
        finally:
            srv.close()

    def test_a_placeholder_in_the_environment_is_ignored_and_explained(self):
        from joules import doctor, share
        H = self.server(); srv = _Server(H)
        share._warned.clear()
        try:
            with mock.patch.dict(os.environ):
                os.environ.pop("CI", None)
                run_cli(["login", self.KEY, "--api-url", srv.url])
                # the line an example once showed, still in ~/.zshrc: the saved key must keep working
                code, _, err = run_cli(["--track", "x", "--api-url", srv.url, "--", sys.executable, "-c", "pass"],
                                       {**SIM, "LANDAUER_API_KEY": "lgk_…"})
                self.assertEqual(code, 0, err)
                self.assertIn("tracked in project x", err)
                self.assertEqual(H.calls[-1], ("POST", "/api/v1/track", "Bearer " + self.KEY))
                self.assertIn("not to an API key", err)
                self.assertNotIn("latin-1", err)
                with mock.patch.dict(os.environ, {"LANDAUER_API_KEY": "lgk_…"}):
                    r = doctor.run(api_url=srv.url, offline=False)
                rows = {x["check"]: x for _, rr in r.sections for x in rr}
                self.assertEqual(rows["$LANDAUER_API_KEY"]["status"], doctor.FAIL)
                self.assertIn("unset LANDAUER_API_KEY", rows["$LANDAUER_API_KEY"]["detail"])
                self.assertEqual(rows["API key"]["status"], doctor.OK)  # the saved one
            # a placeholder given with --key: a plain message, never a header encoding error
            with self.assertRaises(RuntimeError) as c:
                share.check_key("lgk_…", srv.url)
            self.assertIn("not an API key", str(c.exception))
        finally:
            srv.close()

    def test_placeholder_and_rejected_keys_are_not_saved(self):
        from joules import share
        code, _, err = run_cli(["login", "lgk_…"])
        self.assertEqual(code, 2); self.assertIn("not an API key", err)
        H = self.server(ok=False); srv = _Server(H)
        try:
            code, _, err = run_cli(["login", self.KEY, "--api-url", srv.url])
        finally:
            srv.close()
        self.assertEqual(code, 1); self.assertIn("That API key is not valid", err)
        self.assertFalse(os.path.exists(share.key_file()))

    def test_track_without_any_key_says_how_to_get_one(self):
        code, _, err = run_cli(["--track", "x", "--", sys.executable, "-c", "pass"], SIM)
        self.assertEqual(code, 0)  # the command still ran
        self.assertIn("joules login", err)
        self.assertIn("#/settings/keys", err)


class FirstRun(unittest.TestCase):
    def test_missing_command_is_a_message_not_a_traceback(self):
        for measured in (True, False):  # with meters, and with none (the command then runs unmeasured)
            with contextlib.nullcontext() if measured else mock.patch.object(cli, "discover", return_value=meters.Discovery([], ["none"])):
                code, _, err = run_cli(["--", "no-such-command-lg", "train.py"], SIM)
            self.assertEqual(code, 127)
            self.assertIn("command not found: no-such-command-lg", err)
            self.assertNotIn("Traceback", err)

    def test_says_how_to_see_results_on_the_website(self):
        with mock.patch.object(cli, "_interactive", return_value=True):
            code, _, err = run_cli(["--", sys.executable, "-c", "pass"], SIM)
        self.assertEqual(code, 0)
        self.assertIn("stays on this computer", err)
        self.assertIn("joules login", err)  # not signed in: that comes first
        with mock.patch.object(cli, "_interactive", return_value=True):
            code, _, err = run_cli(["--", sys.executable, "-c", "pass"], {**SIM, "LANDAUER_API_KEY": "lgk_" + "d" * 40})
        self.assertIn("joules --track my-project -- ", err)  # a key from the environment tracks only when asked
        code, _, err = run_cli(["--", sys.executable, "-c", "pass"], SIM)  # not in a terminal (scripts, CI): quiet
        self.assertNotIn("stays on this computer", err)

    def test_python_hint_fits_each_system(self):
        for cmd, windows, hint in ((["python"], False, "`python3`"), (["python3"], True, "`python` or `py`")):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cli._not_found(cmd, windows=windows), 127)
            self.assertIn(hint, err.getvalue())

    def test_ollama_not_installed_says_how_to_install_it(self):
        from joules import bench
        with mock.patch.object(bench.shutil, "which", return_value=None), mock.patch.object(bench, "_ollama_app", return_value=None):
            code, _, err = run_cli(["bench", "ollama:llama3.2:3b", "--url", "http://127.0.0.1:9"], SIM)
        self.assertEqual(code, 1)
        self.assertIn("Ollama is not installed", err)
        self.assertIn("ollama", err.split("install it")[-1])  # the command or page for this system
        for sysname, how in (("Darwin", "brew install ollama"), ("Windows", "winget install Ollama.Ollama"),
                             ("Linux", "ollama.com/install.sh")):
            with mock.patch.object(bench.platform, "system", return_value=sysname):
                self.assertIn(how, bench.install_hint())

    def test_ollama_on_another_machine_is_not_started_here(self):
        from joules import bench
        t = bench.parse_target("ollama:m", "http://gpu-box:11434")
        with mock.patch.object(bench, "_post", side_effect=RuntimeError("cannot reach http://gpu-box:11434/api/show")), \
                mock.patch.object(bench.subprocess, "Popen") as popen, self.assertRaises(RuntimeError) as c:
            bench.ensure_model(t)
        popen.assert_not_called()
        self.assertIn("Start it there", str(c.exception))

    @unittest.skipUnless(os.name == "posix", "the fake ollama command is a shell script")
    def test_installed_but_stopped_ollama_is_started(self):
        from joules import bench
        with tempfile.TemporaryDirectory() as t:
            pidfile = Path(t, "pid")
            server = Path(t, "fake_ollama_server.py")
            server.write_text(textwrap.dedent(f"""
                import json, os, sys, threading
                from http.server import BaseHTTPRequestHandler, HTTPServer
                open({str(pidfile)!r}, "w").write(str(os.getpid()))
                class H(BaseHTTPRequestHandler):
                    def log_message(self, *a): pass
                    def reply(self):
                        b = json.dumps({{"version": "0.0-test"}}).encode()
                        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                    do_GET = do_POST = reply
                host, port = os.environ["OLLAMA_HOST"].split(":")
                s = HTTPServer((host, int(port)), H)
                threading.Timer(20, s.shutdown).start()  # never outlives the test run
                s.serve_forever()
            """))
            fake = Path(t, "ollama")
            fake.write_text(f'#!/bin/sh\n[ "$1" = serve ] && exec {sys.executable} {server}\nexit 1\n')
            fake.chmod(0o755)
            sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
            target = bench.parse_target("ollama:m", f"http://127.0.0.1:{port}")
            said = []
            try:
                with mock.patch.dict(os.environ, {"PATH": t + os.pathsep + os.environ["PATH"]}), \
                        mock.patch.object(bench, "_ollama_app", return_value=None):
                    self.assertFalse(bench.ensure_model(target, said.append))  # started, and the model is there
            finally:
                for _ in range(50):
                    if pidfile.exists() and pidfile.read_text():
                        os.kill(int(pidfile.read_text()), signal.SIGTERM); break
                    time.sleep(0.1)
            self.assertIn("starting Ollama…", said)

    def test_ollama_downloads_a_missing_model_first(self):
        class H(BaseHTTPRequestHandler):
            calls, have = [], set()
            def log_message(self, *a): pass
            def send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                H.calls.append(self.path)
                if self.path == "/api/show":
                    return self.send(200, {"details": {}}) if body["model"] in H.have else self.send(404, {"error": f"model '{body['model']}' not found"})
                if self.path == "/api/pull":
                    if body["model"] == "nope:1b":
                        return self.send(200, b'{"status":"pulling manifest"}\n{"error":"pull model manifest: file does not exist"}\n', "application/x-ndjson")
                    H.have.add(body["model"])
                    lines = [{"status": "pulling manifest"}] + [{"status": "pulling 6a0746a1ec1a", "total": 2000, "completed": c} for c in (0, 500, 1000, 2000)] + [{"status": "success"}]
                    return self.send(200, "".join(json.dumps(x) + "\n" for x in lines).encode(), "application/x-ndjson")
                self.send(200, {"eval_count": 50, "prompt_eval_count": 10, "eval_duration": 50_000_000})
        srv = _Server(H)
        try:
            code, out, err = run_cli(["bench", "ollama:llama3.2:3b", "--url", srv.url, "--runs", "2", "--json"], SIM)
            self.assertEqual(code, 0, err)
            self.assertEqual(H.calls[:2], ["/api/show", "/api/pull"])
            self.assertIn("downloading llama3.2:3b", err)
            self.assertIn("100%", err)
            self.assertEqual(json.loads(out)["bench"]["output_tokens"], 100)
            code, _, err = run_cli(["bench", "ollama:nope:1b", "--url", srv.url], SIM)
            self.assertEqual(code, 1)
            self.assertIn("could not download nope:1b: pull model manifest: file does not exist", err)
            self.assertNotIn("Traceback", err)
        finally:
            srv.close()

    def test_no_counters_still_gives_a_result(self):
        with mock.patch.object(cli, "discover", side_effect=lambda gpus, cpu=True, estimate=False: meters.discover(gpus, cpu, estimate)), \
             mock.patch.object(meters, "discover_nvml", return_value=([], None)), mock.patch.object(meters, "discover_amdgpu", return_value=[]), \
             mock.patch.object(meters, "discover_intelgpu", return_value=[]), mock.patch.object(meters, "discover_rapl", return_value=[]), \
             mock.patch.object(meters.platform, "system", return_value="Linux"):
            code, _, err = run_cli(["--", sys.executable, "-c", "sum(i * i for i in range(2_000_000))"], {"JOULES_SIMULATE": ""})
        self.assertEqual(code, 0, err)
        self.assertIn("ESTIMATE (CPU time)", err)
        self.assertNotIn("without measuring", err)


class FirstKey(unittest.TestCase):
    KEY = "lgk_" + "d" * 40

    def tearDown(self):
        from joules import share
        with contextlib.suppress(FileNotFoundError):
            os.remove(share.key_file())

    def test_asks_for_the_key_once_in_a_terminal_then_saves_it(self):
        from joules import share
        H = Login.server(); srv = _Server(H)
        answers = iter(["lgk_…", self.KEY])
        try:
            with mock.patch.dict(os.environ, {"JOULES_NO_PROMPT": ""}), mock.patch.object(cli, "_interactive", return_value=True), \
                 mock.patch("getpass.getpass", side_effect=lambda p: next(answers)):
                code, _, err = run_cli(["--track", "first-test", "--api-url", srv.url, "--", sys.executable, "-c", "pass"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        self.assertIn("not an API key", err)        # the placeholder is refused, then the real key is taken
        self.assertIn("signed in", err)
        self.assertEqual(share.saved_key(), self.KEY)
        self.assertEqual([c[:2] for c in H.calls], [("GET", "/api/v1/track?limit=1&days=1"), ("POST", "/api/v1/track")])

    def test_enter_skips_and_the_command_still_runs(self):
        with mock.patch.dict(os.environ, {"JOULES_NO_PROMPT": ""}), mock.patch.object(cli, "_interactive", return_value=True), \
             mock.patch("getpass.getpass", return_value=""):
            code, _, err = run_cli(["--track", "x", "--", sys.executable, "-c", "import sys; sys.exit(4)"], SIM)
        self.assertEqual(code, 4)
        self.assertIn("joules login", err)


@unittest.skipIf(not shutil.which("sh") or os.name != "posix", "POSIX shell only")
@unittest.skipIf(__import__("importlib.util").util.find_spec("pip") is None, "needs pip to build the package")
class Installer(unittest.TestCase):
    """web/install.sh, the website's one-line install, in an empty home folder with no pipx."""
    def test_installs_and_runs_without_pip(self):
        script = ROOT / "web" / "install.sh"
        if not script.exists():
            self.skipTest("the website's installer is not in this checkout")
        with tempfile.TemporaryDirectory() as t:
            whl_dir = Path(t) / "dist"
            subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(whl_dir), str(Path(__file__).parents[1])],
                           check=True, capture_output=True)
            home = Path(t) / "home"; home.mkdir()
            env = {"HOME": str(home), "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SHELL": "/bin/zsh",
                   "JOULES_WHEEL": str(next(whl_dir.glob("*.whl")))}
            r = subprocess.run(["sh", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("is ready", r.stderr)
            self.assertIn(".zshrc", (home / ".zshrc").read_text(encoding="utf-8") and ".zshrc")  # PATH added for new terminals
            self.assertIn(".local/bin", (home / ".zshrc").read_text(encoding="utf-8"))
            joules = home / ".local" / "bin" / "joules"
            r = subprocess.run([str(joules), "--version"], env=env, capture_output=True, text=True)
            self.assertEqual(r.stdout.strip(), "joules " + __import__("joules").__version__)
            r = subprocess.run([str(home / ".local" / "bin" / "landauer-gap"), "--", sys.executable, "-c", "pass"],
                               env={**env, **SIM}, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            again = subprocess.run(["sh", str(script)], env=env, capture_output=True, text=True)  # updating is the same command
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual((home / ".zshrc").read_text(encoding="utf-8").count("joules (WattOps)"), 1)


@unittest.skipIf(os.name != "posix", "POSIX only")
class MacSudo(unittest.TestCase):
    def test_cached_password_is_used_without_asking(self):
        with mock.patch.object(meters.shutil, "which", return_value="/usr/bin/sudo"), \
             mock.patch.object(meters.subprocess, "call", return_value=0) as call:
            self.assertTrue(meters.mac_sudo([]))
        self.assertEqual(call.call_args_list[0].args[0], ["sudo", "-n", "true"])
        self.assertEqual(len(call.call_args_list), 1)

    def test_asks_once_in_a_terminal(self):
        calls = []
        def fake(cmd, **kw):
            calls.append(cmd); return 1 if cmd[:2] == ["sudo", "-n"] else 0
        with mock.patch.object(meters.shutil, "which", return_value="/usr/bin/sudo"), \
             mock.patch.object(meters.subprocess, "call", side_effect=fake), \
             mock.patch.object(meters.sys.stdin, "isatty", return_value=True), \
             mock.patch.object(meters.sys.stderr, "isatty", return_value=True):
            self.assertTrue(meters.mac_sudo([]))
        self.assertEqual(calls[1][:2], ["sudo", "-v"])

    def test_no_terminal_no_prompt(self):
        notes = []
        with mock.patch.object(meters.shutil, "which", return_value="/usr/bin/sudo"), \
             mock.patch.object(meters.subprocess, "call", return_value=1) as call, \
             mock.patch.object(meters.sys.stdin, "isatty", return_value=False):
            self.assertFalse(meters.mac_sudo(notes))
        self.assertEqual(len(call.call_args_list), 1)
        self.assertIn("sudo", notes[0])

    def test_powermetrics_runs_through_sudo_n(self):
        with tempfile.TemporaryDirectory() as t:
            script = Path(t) / "fake_powermetrics.py"
            script.write_text(FAKE_POWERMETRICS)
            sudo = Path(t) / "sudo"
            sudo.write_text(f"#!/bin/sh\n[ \"$1\" = -n ] || exit 9\nshift\nexec {sys.executable} {script} combined\n")
            sudo.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": f"{t}:{os.environ['PATH']}"}):
                m = meters.PowermetricsMeter(cmd=["powermetrics"], sudo=True)
            try:
                self.assertIsNone(m.proc.poll())
                self.assertGreater(m._seen, 0)
                self.assertEqual(m.proc.args[:3], ["sudo", "-n", "powermetrics"])
            finally:
                m.close()



class NetworkTrouble(unittest.TestCase):
    """A server that accepts the connection and never answers, or answers with something that is not JSON:
    a message, never a stack trace."""
    def silent_server(self):
        import socket
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(5)
        self.addCleanup(srv.close)
        return f"http://127.0.0.1:{srv.getsockname()[1]}"

    def test_timeouts_are_messages(self):
        from joules import bench, share
        url = self.silent_server()
        with self.assertRaisesRegex(RuntimeError, "did not answer"):
            bench._post(url + "/api/generate", {}, 0.3)
        with self.assertRaisesRegex(RuntimeError, "no usable answer"):
            share._post("/api/v1/track", {}, "k", url, 0.3)
        with self.assertRaisesRegex(RuntimeError, "no usable answer"):
            share.check_key("k", url, 0.3)

    def test_not_json_is_a_message(self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200); self.send_header("Content-Length", "5"); self.end_headers(); self.wfile.write(b"<html")
        srv = _Server(H)
        try:
            code, _, err = run_cli(["bench", "openai:x", "--url", srv.url + "/v1"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 1)
        self.assertIn("did not answer with JSON", err)
        self.assertNotIn("Traceback", err)

    def test_refused_fields_are_named(self):
        H = board_handler(400, {"error": {"code": "invalid_run", "message": "Some fields are not valid.", "fields": {"energy_j": "A positive number."}}})
        srv = _Server(H)
        try:
            code, _, err = run_cli(["--track", "p", "--key", "lgk_" + "e" * 40, "--api-url", srv.url, "--", sys.executable, "-c", "pass"], SIM)
        finally:
            srv.close()
        self.assertIn("Some fields are not valid. energy_j: A positive number.", err)

    def test_no_power_above_idle_uses_the_total(self):
        H = handler("ollama"); srv = _Server(H)
        try:  # simulated meters draw the same power idle and busy
            code, out, err = run_cli(["bench", "ollama:x", "--url", srv.url, "--runs", "2", "--subtract-idle", "0.3", "--json"], SIM)
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual(r["bench"]["energy_basis"], "total")
        self.assertGreater(r["bench"]["joules_per_output_token"], 0)
        self.assertTrue(any("no measurable power above idle" in n for n in r["notes"]))



class PlainOutput(unittest.TestCase):
    """A console or pipe that cannot show joules' symbols gets readable stand-ins, never "?" or a crash."""

    def test_symbols_become_plain_text(self):
        line = "joules · python3 -c pass · 0.03 s\n  ──────\n  total 0.09 J = 26 µWh · CO₂ <0.001 g ✓ → ✗ ⚡"
        out = line.encode("ascii", "joules").decode("ascii")
        self.assertEqual(out, "joules - python3 -c pass - 0.03 s\n  ------\n  total 0.09 J = 26 uWh - CO2 <0.001 g OK -> X ")
        self.assertEqual("日本".encode("ascii", "joules"), b"??")  # anything else is still printable

    def test_a_piped_stream_on_windows_is_ascii(self):
        import io
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        with mock.patch.object(cli.os, "name", "nt"), mock.patch.object(cli.sys, "stdout", stream), \
                mock.patch.dict(os.environ, {"PYTHONIOENCODING": ""}):
            cli._plain_streams()
            print("26 µWh · CO₂", file=cli.sys.stdout)
            stream.flush()
        self.assertEqual(raw.getvalue().replace(b"\r", b""), b"26 uWh - CO2\n")


class Doctor(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "fake RAPL files have ':' in their names")
    def test_report_with_gpus_cpu_counters_key_and_ollama(self):
        from joules import doctor, share
        key = "lgk_" + "f" * 40

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                body = {"/api/health": b'{"ok":true}', "/api/tags": b'{"models":[{"name":"llama3.2:3b"}]}'}.get(self.path.split("?")[0])
                if self.path.startswith("/api/v1/track"):
                    body = b'{"runs":[]}' if self.headers["Authorization"] == f"Bearer {key}" else None
                self.send_response(200 if body else 401); self.send_header("Content-Length", str(len(body or b"{}"))); self.end_headers()
                self.wfile.write(body or b"{}")
        srv = _Server(H)
        nv = FakeNVML([{"name": "NVIDIA H100 80GB HBM3", "energy_mj": 1_000_000, "power_mw": 650_000}])
        with tempfile.TemporaryDirectory() as t:
            zone = Path(t) / "class/powercap/intel-rapl:0"
            write(zone / "name", "package-0"); write(zone / "energy_uj", 5_000_000)
            env = {"JOULES_SIMULATE": "", "JOULES_SYSFS": t, "LANDAUER_API_KEY": key}
            with mock.patch.object(meters, "NVML_FACTORY", lambda: nv), mock.patch.object(meters.platform, "system", return_value="Linux"), \
                 mock.patch.object(doctor, "_get", side_effect=lambda url, timeout=5, headers=None: (200, H_get(srv.url, url))), \
                 mock.patch.dict(os.environ, env):
                r = doctor.run(seconds=0.2, api_url=srv.url)
        srv.close()
        rows = {x["check"]: x for _, rs in r.sections for x in rs}
        self.assertEqual(rows["NVIDIA driver"]["status"], "ok")
        self.assertIn("1 GPU", rows["NVIDIA driver"]["detail"])
        self.assertEqual(rows["GPU 0 NVIDIA H100 80GB HBM3"]["status"], "ok")
        self.assertIn("energy counter", rows["GPU 0 NVIDIA H100 80GB HBM3"]["detail"])
        self.assertEqual(rows["RAPL"]["status"], "ok")
        self.assertEqual(rows["API key"]["status"], "ok")
        self.assertIn("Ollama", rows)
        text = doctor.render(r, color=False, ascii_only=False)
        self.assertNotIn(key, text + json.dumps(r.as_dict()))

    def test_no_meters_bad_key_and_privacy(self):
        from joules import doctor
        home = os.path.expanduser("~")
        with mock.patch.object(meters, "discover", return_value=meters.Discovery([], ["No energy counters found. ..."])), \
             mock.patch.object(doctor.sys, "executable", os.path.join(home, ".local/share/landauer-gap/python3")):
            r = doctor.run(offline=True)
        text = doctor.render(r, color=False, ascii_only=True)
        self.assertNotIn(home, text)
        self.assertIn(os.path.join("~", ".local/share/landauer-gap/python3"), text)
        self.assertTrue(text.isascii(), [ch for ch in text if not ch.isascii()])
        self.assertIn("estimates energy from CPU time", text)
        self.assertEqual(text.count("No energy counters found"), 0)   # said once, in plain words
        self.assertNotIn("✓", text)                                    # plain characters for consoles without them

    def test_cli_exit_code_json_and_file(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "report.json")
            code, stdout, err = run_cli(["doctor", "--offline", "--seconds", "0.2", "--json", "--out", out], SIM)
            self.assertEqual(code, 0, err)
            data = json.loads(stdout)
            self.assertEqual(data["joules_doctor"], 1)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f), data)
            sim = [c for s in data["sections"] for c in s["checks"] if c["check"].startswith("GPU H100")]
            self.assertEqual(sim[0]["status"], "warn")                   # simulated is never a pass
            with mock.patch("joules.doctor.check_account", side_effect=lambda r, *a: doctor_fail(r)):
                code, _, _ = run_cli(["doctor", "--offline", "--seconds", "0.1"], SIM)
            self.assertEqual(code, 1)


def doctor_fail(r):
    from joules import doctor
    doctor.add(r.section("WattOps"), doctor.FAIL, "connection", "cannot reach")


def H_get(base, url):
    with urllib.request.urlopen(url.replace("http://localhost:11434", base), timeout=5) as r:
        return r.read()


if __name__ == "__main__":
    unittest.main()


class CalibrateTests(unittest.TestCase):
    def test_idle_window_against_wall_watts(self):
        code, out, err = run_cli(["calibrate", "--seconds", "0.4", "--wall-w", "1000", "--json"], SIM)
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual((r["kind"], r["command"], r["exit_code"]), ("calibrate", [], 0))
        self.assertAlmostEqual(r["metered"]["avg_w"], 700, delta=35)
        self.assertAlmostEqual(r["wall"]["metered_share_pct"], 70, delta=4)
        self.assertGreater(r["wall"]["unmetered_j"], 0)
        self.assertTrue(any("Simulated" in n for n in r["notes"]))

    def test_command_against_wall_wh_and_text(self):
        code, out, err = run_cli(["calibrate", "--wall-wh", "0.01", "--out", os.path.join(tempfile.gettempdir(), "cal.json"),
                                  "--", sys.executable, "-c", "import time; time.sleep(0.3)"], SIM)
        self.assertEqual(code, 0, err)
        self.assertIn("joules calibrate", err)
        self.assertIn("of the wall", err)
        with open(os.path.join(tempfile.gettempdir(), "cal.json")) as f:
            r = json.load(f)
        self.assertEqual(r["wall"]["energy_j"], 36.0)
        self.assertAlmostEqual(r["metered"]["energy_j"], 700 * r["duration_s"], delta=40)

    def test_without_a_wall_reading_it_says_what_to_do(self):
        code, out, err = run_cli(["calibrate", "--seconds", "0.2"], SIM)
        self.assertEqual(code, 0, err)
        self.assertIn("--wall-wh", err)
        self.assertNotIn("of the wall", err)

    def test_refuses_both_wall_options_and_bad_values(self):
        self.assertEqual(run_cli(["calibrate", "--seconds", "0.1", "--wall-wh", "1", "--wall-w", "1"], SIM)[0], 2)
        self.assertEqual(run_cli(["calibrate", "--seconds", "0.1", "--wall-w", "0"], SIM)[0], 2)

    def test_command_exit_code_passes_through(self):
        code, _, err = run_cli(["calibrate", "--", sys.executable, "-c", "import sys; sys.exit(3)"], SIM)
        self.assertEqual(code, 3, err)

    def test_a_wall_meter_log_is_integrated_over_exactly_the_measured_seconds(self):
        from joules import walllog
        d = Path(tempfile.mkdtemp())
        # a meter logging every second at 1000 W, 1 s before the window to 1 s after; then a 30 s gap
        write(d / "log.csv", "time,watts\n" + "".join(f"{t},{1000.0}\n" for t in range(99, 112)))
        got = walllog.integrate(walllog.read(str(d / "log.csv")), 100.0, 110.0)
        self.assertAlmostEqual(got["energy_j"], 10_000)
        self.assertAlmostEqual(got["coverage"], 1.0)
        # a ramp from 0 to 1000 W: trapezoids, clipped to the window
        write(d / "ramp.csv", "0,0\n10,1000\n")
        self.assertAlmostEqual(walllog.integrate(walllog.read(str(d / "ramp.csv")), 5, 10)["energy_j"], 3750)
        # other layouts meters write: ISO times with a zone, a "Power (W)" header with units, semicolons, milliseconds
        write(d / "iso.csv", "Date;Power (W)\n2026-10-20T10:00:00Z;200 W\n2026-10-20T10:00:10Z;200 W\n")
        import datetime as dt
        t0 = dt.datetime(2026, 10, 20, 10, tzinfo=dt.timezone.utc).timestamp()
        self.assertAlmostEqual(walllog.integrate(walllog.read(str(d / "iso.csv")), t0, t0 + 10)["energy_j"], 2000)
        write(d / "ms.csv", "ts,apower\n1700000000000,50\n1700000004000,50\n")
        self.assertAlmostEqual(walllog.integrate(walllog.read(str(d / "ms.csv")), 1.7e9, 1.7e9 + 4)["energy_j"], 200)
        # a long gap in the log is not covered, and is not made up
        write(d / "gap.csv", "time,watts\n0,100\n1,100\n2,100\n40,100\n41,100\n")
        self.assertAlmostEqual(walllog.integrate(walllog.read(str(d / "gap.csv")), 0, 41)["coverage"], 3 / 41)
        with self.assertRaisesRegex(ValueError, "no time and watts"):
            write(d / "bad.csv", "a,b\nx,y\n"); walllog.read(str(d / "bad.csv"))

    def test_calibrate_with_a_meter_log_now_or_later(self):
        d = Path(tempfile.mkdtemp())
        out = d / "cal.json"
        code, _, err = run_cli(["calibrate", "--seconds", "0.4", "--out", str(out)], SIM)
        self.assertEqual(code, 0, err)
        r = json.loads(out.read_text())
        self.assertLess(r["started_unix"], r["ended_unix"])
        # the meter's software exported the log afterwards: 1000 W, one reading every 0.1 s around the run
        t = r["started_unix"] - 1
        write(d / "log.csv", "time,watts\n" + "".join(f"{t + i / 10:.3f},1000\n" for i in range(int((r["ended_unix"] - t + 1) * 10) + 2)))
        code, out2, err = run_cli(["calibrate", "--report", str(out), "--wall-csv", str(d / "log.csv"), "--json"], SIM)
        self.assertEqual(code, 0, err)
        w = json.loads(out2)["wall"]
        self.assertAlmostEqual(w["avg_w"], 1000, delta=1)
        self.assertEqual(w["source"], "log")
        self.assertAlmostEqual(w["metered_share_pct"], 70, delta=5)  # simulated 700 W of meters against 1000 W at the wall
        # a log that does not cover the run is refused, not guessed
        write(d / "short.csv", f"time,watts\n{r['started_unix']},1000\n{r['started_unix'] + 0.05},1000\n")
        self.assertEqual(run_cli(["calibrate", "--report", str(out), "--wall-csv", str(d / "short.csv")], SIM)[0], 2)
        self.assertEqual(run_cli(["calibrate", "--report", str(out)], SIM)[0], 2)  # --report needs a log


class FakeNVMLHealth(FakeNVML):
    """FakeNVML plus the health calls. Lowering the power limit lowers the card's power and clock, as on a real card."""
    def __init__(self, devices):
        super().__init__(devices)
        self.limits = {i: d.get("limit_mw", 700_000) for i, d in enumerate(devices)}
        self.no_root, self.sets = False, []
    def power_mw(self, h): return min(self.devices[h]["power_mw"], self.limits[h])
    def energy_mj(self, h): return None  # power sampling, so a new limit shows in the energy at once
    def clock_mhz(self, h, which=1): return 2619 if which == 2 else int(1980 * min(1.0, self.limits[h] / 700_000))
    def max_clock_mhz(self, h, which=1): return 2619 if which == 2 else 1980
    def temperature_c(self, h): return self.devices[h].get("temp", 68)
    def power_limit_mw(self, h): return self.limits[h]
    def power_limit_range_mw(self, h): return (200_000, 700_000)
    def set_power_limit_mw(self, h, mw):
        if self.no_root: return 4
        self.sets.append((h, mw)); self.limits[h] = mw; return 0
    def throttle_reasons(self, h): return 0x4 if self.limits[h] < 700_000 or self.devices[h].get("capped") else 0x1
    def utilization_pair(self, h): return self.devices[h].get("util", (87, 40))
    def memory_used_pct(self, h): return 25.0


class GpuHealth(unittest.TestCase):
    H100 = {"name": "NVIDIA H100 80GB HBM3", "power_mw": 650_000, "capped": True, "util": (71, 62)}

    def test_meter_keeps_clocks_temperature_and_throttle_share(self):
        nv = FakeNVMLHealth([self.H100, {"name": "NVIDIA L40S", "power_mw": 300_000, "util": (3, 1), "temp": 35}])
        with mock.patch.object(meters, "NVML_FACTORY", lambda: nv), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            found, _ = meters.discover_nvml([], None)
        m, l40 = found
        m.start(0.0); l40.start(0.0)
        for t in (0.6, 1.2, 1.8):
            m.poll(t); l40.poll(t)
        m.stop(2.4); l40.stop(2.4)
        h = m.health()
        self.assertEqual((h["sm_clock_mhz"], h["sm_clock_max_mhz"], h["temp_c_max"], h["power_limit_w"]), (1980, 1980, 68, 700))
        self.assertEqual((h["gpu_util_pct"], h["mem_util_pct"], h["throttled_pct"], h["throttle_reasons"]), (71, 62, 100, {"power cap": 100}))
        v = model.gpu_verdict(h)
        self.assertEqual((v["bound"], v["throttled"], v["reason"]), ("memory", True, "power cap"))
        self.assertIn("memory-bound · throttled 100% of the time (power cap)", v["summary"])
        self.assertEqual(model.gpu_verdict(l40.health())["summary"], "mostly idle")
        now = m.health_now()
        self.assertEqual((now["sm_clock_mhz"], now["throttled"], now["throttle_reason"], now["gpu_util_pct"], now["power_limit_w"]), (1980, True, "power cap", 71, 700))
        self.assertIsNone(l40.health_now()["throttle_reason"] or None)

    def test_plain_fake_without_health_calls_still_measures(self):
        nv = FakeNVML([{"name": "NVIDIA H100 80GB HBM3", "energy_mj": 1_000_000, "power_mw": 650_000}])
        with mock.patch.object(meters, "NVML_FACTORY", lambda: nv), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            found, _ = meters.discover_nvml([], None)
        m = found[0]; m.start(0.0); nv.devices[0]["energy_mj"] += 650_000; m.poll(1.0); m.stop(1.0)
        self.assertAlmostEqual(m.energy_j, 650.0)
        self.assertIsNone(m.health()); self.assertIsNone(m.health_now())

    def test_verdict_rules(self):
        V = model.gpu_verdict
        self.assertIsNone(V(None)); self.assertIsNone(V({}))
        self.assertEqual(V({"gpu_util_pct": 95, "mem_util_pct": 30})["bound"], "compute")
        self.assertEqual(V({"gpu_util_pct": 60, "mem_util_pct": 55})["bound"], "memory")
        self.assertEqual(V({"gpu_util_pct": 5, "mem_util_pct": 1})["bound"], "idle")
        self.assertEqual(V({"gpu_util_pct": 40, "mem_util_pct": 20})["bound"], "mixed")
        self.assertEqual(V({"throttled_pct": 9})["throttled"], False)
        v = V({"gpu_util_pct": 95, "mem_util_pct": 30, "throttled_pct": 40, "throttle_reasons": {"thermal": 35, "power cap": 5}})
        self.assertEqual((v["throttled"], v["reason"], v["summary"]), (True, "thermal", "compute-bound · throttled 40% of the time (thermal)"))

    def test_power_sweep_pick(self):
        rows = [{"limit_pct": 100, "j_per_token": 3.0, "tokens_per_s": 100}, {"limit_pct": 85, "j_per_token": 2.4, "tokens_per_s": 97},
                {"limit_pct": 70, "j_per_token": 2.0, "tokens_per_s": 92}, {"limit_pct": 55, "j_per_token": 1.9, "tokens_per_s": 70}]
        p = model.power_sweep_pick(rows, 10)
        self.assertEqual((p["limit_pct"], p["base_limit_pct"], round(p["energy_saved_pct"]), round(p["slowdown_pct"])), (70, 100, 33, 8))
        self.assertEqual(model.power_sweep_pick(rows, 2)["limit_pct"], 100)
        self.assertEqual(model.power_sweep_pick(rows, 50)["limit_pct"], 55)
        self.assertIsNone(model.power_sweep_pick([], 10))

    def test_run_reports_health_and_the_tracked_record_carries_a_summary(self):
        from joules import share
        nv = FakeNVMLHealth([self.H100])
        with tempfile.TemporaryDirectory() as t, mock.patch.object(meters, "NVML_FACTORY", lambda: nv), \
                mock.patch.dict(os.environ, {"JOULES_SYSFS": t, "CUDA_VISIBLE_DEVICES": "0"}):
            os.environ.pop("JOULES_SIMULATE", None)
            code, out, err = run_cli(["run", "--no-cpu", "--json", "--", sys.executable, "-c", "import time; time.sleep(0.7)"])
        self.assertEqual(code, 0, err)
        r = json.loads(out); d = r["devices"][0]
        self.assertEqual(d["method"], "nvml power sampling")
        self.assertEqual((d["gpu"]["throttled_pct"], d["verdict"]["bound"], d["verdict"]["throttled"]), (100, "memory", True))
        text = cli.render_run(r)
        self.assertIn("1,980 MHz (100% of max) · 71% busy · memory 62% · 68 °C max · limit 700 W · memory-bound · throttled 100% of the time (power cap)", text)
        rec = share.track_record(r, "p")
        self.assertEqual(rec["gpu"], {"clock_pct": 100.0, "gpu_util_pct": 71.0, "mem_util_pct": 62.0, "throttled_pct": 100.0,
                                      "temp_c_max": 68, "power_limit_w": 700.0, "reason": "power cap", "bound": "memory"})

    def test_tune_sweeps_limits_restores_them_and_picks_the_cheapest_token(self):
        nv = FakeNVMLHealth([{"name": "NVIDIA H100 80GB HBM3", "power_mw": 650_000, "util": (95, 30)}])
        # each limit's window is two 0.2 s requests, so a few milliseconds of timing noise moves J/token by well under 2%
        H = handler("ollama", delay=0.2); srv = _Server(H)
        try:
            with tempfile.TemporaryDirectory() as t, mock.patch.object(meters, "NVML_FACTORY", lambda: nv), \
                    mock.patch.dict(os.environ, {"JOULES_SYSFS": t, "CUDA_VISIBLE_DEVICES": "0"}):
                os.environ.pop("JOULES_SIMULATE", None)
                code, out, err = run_cli(["tune", "ollama:llama3.1:8b", "--url", srv.url, "--limits", "100,70", "--runs", "2", "--warmup", "0", "--no-cpu", "--json"])
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual([x["limit_pct"] for x in r["rows"]], [100, 70])
        self.assertEqual([x["limit_w"] for x in r["rows"]], [[700.0], [490.0]])
        self.assertAlmostEqual(r["rows"][0]["gpu_avg_w"], 650, delta=5); self.assertAlmostEqual(r["rows"][1]["gpu_avg_w"], 490, delta=5)
        # the saving is what the rows measured; per second of work it is the power drop, whatever the machine's
        # timing noise adds to each window's length (a slow runner's first window can be a tenth longer)
        a, b = r["rows"]
        self.assertEqual(r["pick"]["limit_pct"], 70)
        self.assertAlmostEqual(r["pick"]["energy_saved_pct"], 100 * (1 - b["j_per_token"] / a["j_per_token"]), delta=1e-9)
        self.assertAlmostEqual(100 * (1 - (b["j_per_token"] / b["duration_s"]) / (a["j_per_token"] / a["duration_s"])), 100 * (1 - 490 / 650), delta=2.5)
        self.assertEqual(nv.sets, [(0, 700_000), (0, 490_000), (0, 700_000)])  # 100%, 70%, then the original limit back
        self.assertEqual(nv.limits[0], 700_000)
        self.assertEqual(r["rows"][1]["gpu"][0]["throttle_reasons"], {"power cap": 100})
        text = cli.render_tune(r)
        self.assertIn("best: 70% of the limit uses", text); self.assertIn("nvidia-smi -pl 490", text)

    def test_tune_without_permission_says_sudo_and_touches_nothing(self):
        nv = FakeNVMLHealth([{"name": "NVIDIA H100 80GB HBM3", "power_mw": 650_000}]); nv.no_root = True
        H = handler("ollama", delay=0.01); srv = _Server(H)
        try:
            with tempfile.TemporaryDirectory() as t, mock.patch.object(meters, "NVML_FACTORY", lambda: nv), \
                    mock.patch.dict(os.environ, {"JOULES_SYSFS": t, "CUDA_VISIBLE_DEVICES": "0"}):
                os.environ.pop("JOULES_SIMULATE", None)
                code, _, err = run_cli(["tune", "ollama:llama3.1:8b", "--url", srv.url, "--runs", "1", "--no-cpu"])
        finally:
            srv.close()
        self.assertEqual(code, 1); self.assertIn("sudo", err); self.assertEqual(nv.sets, [])
        self.assertEqual(run_cli(["tune", "ollama:x", "--limits", "150"], SIM)[0], 2)
        self.assertEqual(run_cli(["tune", "ollama:x", "--runs", "1"], SIM)[0], 1)  # simulated meters have no power limit


class AgentHealthMetrics(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "joules agent is Linux only")
    def test_metrics_carry_clocks_temperature_and_throttling(self):
        class HealthMeter(FakeMeter):
            def health_now(self):
                return {"sm_clock_mhz": 1400, "sm_clock_max_mhz": 1980, "mem_clock_mhz": 2619, "temp_c": 71, "gpu_util_pct": 91,
                        "mem_util_pct": 44, "throttled": True, "throttle_reason": "power cap", "power_limit_w": 490}
        t = tempfile.TemporaryDirectory(); root = Path(t.name); cluster(root)
        with mock.patch.dict(os.environ, {"JOULES_PROC": str(root)}):
            nv = FakeNvmlProcs({0: [(100, 8 << 30)]}, {0: 91})
            a = ag.Agent([HealthMeter("GPU 0 H100", "gpu", 700, nv, 0), FakeMeter("CPU package 0", "cpu", 100)], interval=0.05, price=0.2, g_per_kwh=400, host="n1")
            a.start(0, wall=1_759_999_990); a.tick(10, 10, wall=1_760_000_000)
            text = ag.metrics_text(a.acc, 0.2, 400, "n1")
        t.cleanup()
        for line in ('joules_gpu_sm_clock_mhz{host="n1",device="GPU 0 H100"} 1400', 'joules_gpu_sm_clock_max_mhz{host="n1",device="GPU 0 H100"} 1980',
                     'joules_gpu_temperature_celsius{host="n1",device="GPU 0 H100"} 71', 'joules_gpu_memory_utilization_percent{host="n1",device="GPU 0 H100"} 44',
                     'joules_gpu_power_limit_watts{host="n1",device="GPU 0 H100"} 490', 'joules_gpu_throttled{host="n1",device="GPU 0 H100",reason="power cap"} 1',
                     "# TYPE joules_gpu_throttled gauge"):
            self.assertIn(line, text)

    @unittest.skipIf(os.name == "nt", "joules agent is Linux only")
    def test_live_beat_carries_throttled_count_hottest_card_and_reason(self):
        class HealthMeter(FakeMeter):
            def __init__(self, *a, throttled=True, temp=71, **k):
                super().__init__(*a, **k); self._thr, self._temp = throttled, temp
            def health_now(self):
                return {"sm_clock_mhz": 1400, "sm_clock_max_mhz": 1980, "mem_clock_mhz": 2619, "temp_c": self._temp, "gpu_util_pct": 91,
                        "mem_util_pct": 44, "throttled": self._thr, "throttle_reason": "power cap" if self._thr else "", "power_limit_w": 490}
        t = tempfile.TemporaryDirectory(); root = Path(t.name); cluster(root)
        with mock.patch.dict(os.environ, {"JOULES_PROC": str(root)}):
            nv = FakeNvmlProcs({0: [(100, 8 << 30)], 1: []}, {0: 91, 1: 0})
            a = ag.Agent([HealthMeter("GPU 0 H100", "gpu", 700, nv, 0), HealthMeter("GPU 1 H100", "gpu", 80, nv, 1, throttled=False, temp=83),
                          FakeMeter("CPU package 0", "cpu", 100)], interval=0.05, price=0.2, g_per_kwh=400, host="n1")
            a.start(0, wall=1_759_999_990); a.tick(10, 10, wall=1_760_000_000)
            b = a.live_beat()
            plain = ag.Agent([FakeMeter("GPU 0 H100", "gpu", 700, nv, 0)], interval=0.05, price=0.2, g_per_kwh=400, host="n2")
            plain.start(0, wall=1_759_999_990); plain.tick(10, 10, wall=1_760_000_000)
            pb = plain.live_beat()
        t.cleanup()
        self.assertEqual((b["gpus"], b["throttled_gpus"], b["temp_c_max"], b["throttle_reason"]), (2, 1, 83, "power cap"))
        g0, g1 = b["gpu_list"]
        self.assertEqual((g0["i"], g0["clock_pct"], g0["temp_c"], g0["power_limit_w"], g0["mem_util"], g0["throttled"], g0["reason"]), (0, 70.7, 71.0, 490.0, 44.0, True, "power cap"))
        self.assertEqual((g1["i"], g1["throttled"], g1["temp_c"]), (1, False, 83.0)); self.assertNotIn("reason", g1)
        self.assertEqual(set(pb["gpu_list"][0]), {"i", "name", "watts", "util", "idle", "job"})  # an older driver: no clocks, temperature or throttling
        self.assertNotIn("throttled_gpus", pb); self.assertNotIn("temp_c_max", pb)  # an older driver, or no NVIDIA card: nothing to say


class UploadRedirects(unittest.TestCase):
    """A domain move answers 308: the upload is repeated to the new address with the same body, but only on the same site."""
    KEY = "lgk_" + "d" * 40

    @staticmethod
    def server(location_of):
        class H(BaseHTTPRequestHandler):
            calls = []
            def log_message(self, *a): pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                H.calls.append((self.path, self.headers["Authorization"], json.loads(body)))
                loc = location_of(self.path, self)
                if loc:
                    self.send_response(308); self.send_header("Location", loc); self.send_header("Content-Length", "0"); self.end_headers(); return
                data = b'{"id": 7}'
                self.send_response(201); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
        return H

    def test_same_site_redirect_is_followed_with_the_same_post(self):
        from joules import share
        H = self.server(lambda path, h: f"http://localhost:{h.server.server_port}/moved{path}" if not path.startswith("/moved") else None)
        srv = _Server(H)
        try:
            with mock.patch.object(share, "_same_site", lambda a, b: True):  # localhost stands in for the site; the rule itself is tested below
                r = share.send_track({"project": "p", "kind": "run"}, self.KEY, srv.url)
        finally:
            srv.close()
        self.assertEqual(r, {"id": 7})
        self.assertEqual([c[0] for c in H.calls], ["/api/v1/track", "/moved/api/v1/track"])
        self.assertEqual(H.calls[1][1], f"Bearer {self.KEY}"); self.assertEqual(H.calls[1][2], {"project": "p", "kind": "run"})

    def test_redirect_to_another_site_is_refused_and_the_key_stays_home(self):
        from joules import share
        H = self.server(lambda path, h: "https://evil.example/collect")
        srv = _Server(H)
        try:
            with self.assertRaises(RuntimeError) as e:
                share.send_track({"project": "p", "kind": "run"}, self.KEY, srv.url)
        finally:
            srv.close()
        self.assertIn("not the WattOps site", str(e.exception))
        self.assertEqual(len(H.calls), 1)

    def test_same_site_rule(self):
        from joules import share
        self.assertTrue(share._same_site("https://wattops.dev/api/v1/track", "https://www.wattops.dev/api/v1/track"))
        self.assertTrue(share._same_site("https://wattops.dev/x", "https://landauer-gap.vercel.app/x"))
        self.assertTrue(share._same_site("https://my-wattops.example.com/x", "https://www.my-wattops.example.com/x"))
        self.assertFalse(share._same_site("https://wattops.dev/x", "https://evil.example/x"))
        self.assertFalse(share._same_site("https://wattops.dev/x", "https://wattops.dev.evil.example/x"))


# ---------------------------------------------------------------- 0.7: phases, the library API, compare
class ManualCounter(meters.CounterMeter):
    """An energy counter the test sets by hand, so a phase's energy is known exactly."""
    method = "test counter"

    def __init__(self, name, kind="gpu"):
        super().__init__(name)
        self.kind, self.value = kind, 0.0
    def read_j(self): return self.value


class Phases(unittest.TestCase):
    def test_phase_energy_is_the_change_between_marks_and_phases_nest(self):
        from joules.phases import PhaseClock, PhaseError
        g, c = ManualCounter("GPU 0"), ManualCounter("CPU package 0", "cpu")
        for m in (g, c):
            m.start(time.monotonic())
        pc = PhaseClock([g, c])
        pc.begin("full recompute")
        g.value += 100; c.value += 10
        inner = pc.begin("incremental update")["id"]
        g.value += 30; c.value += 3
        r_in = pc.end(pid=inner)
        g.value += 5
        r_out = pc.end(name="full recompute")
        self.assertEqual((r_in["energy_j"], r_in["devices"]), (33, {"GPU 0": 30, "CPU package 0": 3}))
        self.assertEqual(r_out["energy_j"], 148)
        self.assertGreaterEqual(r_out["duration_s"], r_in["duration_s"])
        for x in (12, 10, 14, 11):
            p = pc.begin("incremental update")["id"]; g.value += x; pc.end(pid=p)
        s = pc.summary()
        self.assertEqual(s["open"], 0)
        u = s["phases"]["incremental update"]
        self.assertEqual((u["count"], u["energy_j_total"], u["energy_j"]["median"], u["energy_j"]["min"], u["energy_j"]["max"]), (5, 80, 12, 10, 33))
        self.assertEqual(pc.totals()["full recompute"][:2], (1, 148))
        for bad in ("", " ", "-x", "a" * 65, "a\nb", "<script>"):
            with self.assertRaises(PhaseError):
                pc.begin(bad)
        with self.assertRaises(PhaseError):
            pc.end(pid="p999")
        with self.assertRaises(PhaseError):
            pc.end(name="never started")
        with self.assertRaises(PhaseError):
            pc.end()
        pc.reset()
        self.assertEqual(pc.summary(), {"open": 0, "phases": {}}); self.assertEqual(pc.totals(), {})

    def test_too_many_open_phases_or_names_is_refused(self):
        from joules import phases
        pc = phases.PhaseClock([ManualCounter("GPU 0")])
        pc.meters[0].start(time.monotonic())
        with mock.patch.object(phases, "MAX_OPEN", 3), mock.patch.object(phases, "MAX_NAMES", 2):
            pc.begin("a"); pc.begin("a"); pc.begin("b")
            with self.assertRaisesRegex(phases.PhaseError, "already open"):
                pc.begin("a")
            pc.end(name="a")
            with self.assertRaisesRegex(phases.PhaseError, "different phase names"):
                pc.begin("c")
            pc.begin("b")  # a known name is fine

    def test_counter_keeps_energy_when_two_readers_poll_out_of_order(self):
        m = ManualCounter("GPU 0"); m.start(10.0)
        m.value = 50; m.poll(12.0)
        m.value = 80; m.poll(11.0)  # the agent's loop took its time before a phase mark, then polled
        self.assertEqual(m.energy_j, 80)
        self.assertEqual(m._t_last, 12.0)

    def test_gpu_health_over_a_phase_only(self):
        nv = FakeNVMLHealth([{"name": "NVIDIA GeForce RTX 3070 Ti", "power_mw": 250_000, "limit_mw": 290_000, "util": (97, 20), "temp": 61}])
        nv.limits[0] = 700_000  # full clock, not throttled
        with mock.patch.object(meters, "NVML_FACTORY", lambda: nv), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            m = meters.discover_nvml([], None)[0][0]
        m.start(0.0)
        for t in (0.6, 1.2):
            m.poll(t)
        before = m.health_state()
        nv.limits[0] = 350_000; nv.devices[0]["temp"] = 74  # now capped at half power: half clock, throttled
        for t in (1.8, 2.4, 3.0):
            m.poll(t)
        h = m.health_since(before)
        self.assertEqual((h["throttled_pct"], h["throttle_reasons"], h["clock_pct"]), (100, {"power cap": 100}, 50))
        self.assertEqual((h["temp_c_start"], h["temp_c_end"], h["gpu_util_pct"]), (61, 74, 97))
        self.assertLess(m.health()["throttled_pct"], 100)  # the whole run includes the unthrottled start
        self.assertIsNone(m.health_since(None))


@unittest.skipIf(os.name == "nt", "joules agent is Linux only")
class AgentPhases(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); root = Path(self.t.name); cluster(root)
        self.env = mock.patch.dict(os.environ, {"JOULES_PROC": str(root)}); self.env.start()
        self.g, self.c = ManualCounter("GPU 0 NVIDIA GeForce RTX 3070 Ti"), ManualCounter("CPU package 0", "cpu")
        self.a = ag.Agent([self.g, self.c], interval=0.05, host="lab-1")
        self.a.start(time.monotonic())
        self.srv = ag.AgentServer(("127.0.0.1", 0), self.a)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.env.stop(); self.t.cleanup()

    def post(self, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else json.dumps(body or {}).encode()
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(self.base + path, data=data, method="POST", headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path):
        return json.loads(urllib.request.urlopen(self.base + path).read())

    def test_start_end_summary_energy_and_metrics(self):
        code, s = self.post("/phase/start", {"name": "incremental update"})
        self.assertEqual((code, s["name"]), (200, "incremental update"))
        self.g.value += 42.5; self.c.value += 7.5
        self.a.tick(time.monotonic(), 0.05)  # the loop runs between the marks; the phase still gets every joule
        self.g.value += 10
        code, r = self.post("/phase/end", {"id": s["id"]})
        self.assertEqual(code, 200)
        self.assertEqual((r["energy_j"], r["devices"]), (60, {"GPU 0 NVIDIA GeForce RTX 3070 Ti": 52.5, "CPU package 0": 7.5}))
        _, s2 = self.post("/phase/start", {"name": "full recompute"}); self.g.value += 300
        code, r2 = self.post("/phase/end", {"name": "full recompute"})
        self.assertEqual((code, r2["energy_j"]), (200, 300))
        ph = self.get("/phases")
        self.assertEqual((ph["host"], ph["open"], sorted(ph["phases"])), ("lab-1", 0, ["full recompute", "incremental update"]))
        self.assertEqual(ph["phases"]["incremental update"]["energy_j"]["median"], 60)
        self.assertEqual([x["name"] for x in ph["recent"]], ["incremental update", "full recompute"])
        e = self.get("/energy")
        self.assertEqual((e["host"], e["total_j"]), ("lab-1", 360))
        text = urllib.request.urlopen(self.base + "/metrics").read().decode()
        self.assertIn('joules_phase_energy_joules_total{host="lab-1",phase="incremental update"} 60', text)
        self.assertIn('joules_phases_total{host="lab-1",phase="full recompute"} 1', text)
        self.assertIn("# TYPE joules_phase_duration_seconds_total counter", text)
        self.assertEqual(self.post("/phase/reset")[0], 200)
        self.assertEqual(self.get("/phases")["phases"], {})
        self.assertNotIn("joules_phase_energy", urllib.request.urlopen(self.base + "/metrics").read().decode())

    def test_refuses_web_pages_bad_bodies_and_unknown_phases(self):
        self.assertEqual(self.post("/phase/start", {"name": "x"}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.post("/phase/start", raw=b"name=x", headers={"Content-Type": "application/x-www-form-urlencoded"})[0], 415)
        self.assertEqual(self.post("/phase/start", raw=b"[1]")[0], 400)
        self.assertEqual(self.post("/phase/start", raw=b"{nope")[0], 400)
        self.assertEqual(self.post("/phase/start", raw=b"x" * 5000)[0], 413)
        code, r = self.post("/phase/start", {"name": "bad\nname"})
        self.assertEqual(code, 400); self.assertIn("phase name", r["error"])
        self.assertEqual(self.post("/phase/end", {"id": "p404"})[0], 400)
        self.assertEqual(self.post("/elsewhere")[0], 404)
        self.assertEqual(self.get("/phases")["open"], 0)


class LibraryApi(unittest.TestCase):
    def test_session_phases_measure_and_save(self):
        import joules
        with mock.patch.dict(os.environ, {"JOULES_SIMULATE": "gpu:RTX 3070 Ti:200,cpu:package 0:50"}):
            with joules.Session(interval=0.02) as s:
                self.assertEqual([d["name"] for d in s.devices], ["GPU RTX 3070 Ti", "CPU package 0"])
                with s.phase("full recompute") as full:
                    time.sleep(0.2)
                    with s.phase("incremental update") as inc:
                        time.sleep(0.05)
                self.assertIn("running", repr(joules.Phase("x")))
                pid = s.begin("manual"); time.sleep(0.05); r = s.end(pid)
                snap = s.snapshot()
                out = Path(_CONFIG) / "phases.json"; s.save(str(out))
            with joules.measure("one solve", interval=0.02) as one:
                time.sleep(0.1)
        self.assertAlmostEqual(full.energy_j, 250 * full.duration_s, delta=1e-6)  # constant power, exact at the marks
        self.assertAlmostEqual(full.avg_w, 250, delta=1e-6)
        self.assertLess(inc.energy_j, full.energy_j); self.assertGreater(inc.duration_s, 0.04)
        self.assertEqual(set(inc.devices), {"GPU RTX 3070 Ti", "CPU package 0"})
        self.assertIsNone(full.gpu)  # simulated meters have no health
        self.assertGreater(r["energy_j"], 0); self.assertGreater(snap["total_j"], full.energy_j)
        saved = json.loads(out.read_text())
        self.assertEqual((saved["kind"], saved["simulated"], len(saved["phases"])), ("phases", True, 3))
        self.assertEqual(set(saved["summary"]), {"full recompute", "incremental update", "manual"})
        self.assertAlmostEqual(one.energy_j, 250 * one.duration_s, delta=1e-6)
        self.assertIn("one solve", repr(one))

    def test_session_label_names_a_notebook(self):
        from joules import api
        for argv0, mods, want in [("/usr/local/lib/python3.11/site-packages/colab_kernel_launcher.py", {}, "notebook"),  # Kaggle, Colab
                                  ("/opt/conda/lib/python3.10/site-packages/ipykernel_launcher.py", {}, "notebook"),     # Jupyter
                                  ("train.py", {"ipykernel": object()}, "notebook"), ("/home/me/proj/train.py", {}, "python train.py"),
                                  ("-c", {}, "python"), ("", {}, "python")]:
            with mock.patch.object(sys, "argv", [argv0]), mock.patch.dict(sys.modules, mods):
                if not mods:
                    sys.modules.pop("ipykernel", None)
                self.assertEqual(api._default_label(), want, argv0)

    def test_session_saved_to_reports_with_its_phases(self):
        # Session(track=…) sends one Tracking record when it ends: the whole session's energy, cost and CO2, and each
        # phase's count, totals, median and spread. A server that is down or a missing key never breaks the program.
        import joules
        class H(BaseHTTPRequestHandler):
            calls = []
            def log_message(self, *a): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                H.calls.append((self.path, self.headers.get("Authorization"), body))
                data = json.dumps({"id": 77, "project": body["project"], "page": "/#/track"}).encode()
                self.send_response(201); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
        srv = _Server(H)
        key = "lgk_" + "s" * 40
        try:
            env = {"JOULES_SIMULATE": "gpu:RTX 3070 Ti:200,cpu:package 0:50", "LANDAUER_API_KEY": key, "LANDAUER_API_URL": srv.url}
            err = io.StringIO()
            with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(err):
                with joules.Session(interval=0.02, track="hst-study", name="incremental vs full", price=0.2, grid="eu") as s:
                    for _ in range(3):
                        with s.phase("incremental update"):
                            time.sleep(0.03)
                    with s.phase("full recompute"):
                        time.sleep(0.3)  # three times the incremental updates' 0.09 s, so the order holds on a slow machine
                    whole = s.whole()
                self.assertEqual(s.saved["id"], 77)
            path, auth, r = H.calls[0]
            self.assertEqual((path, auth), ("/api/v1/track", f"Bearer {key}"))
            self.assertEqual((r["project"], r["label"], r["kind"], r["simulated"], r["g_per_kwh"], r["price_per_kwh"]), ("hst-study", "incremental vs full", "run", True, 250, 0.2))
            self.assertGreaterEqual(r["energy_j"], whole["energy_j"]); self.assertAlmostEqual(r["co2_g"], r["energy_j"] / 3.6e6 * 250, delta=1e-9)
            names = [p["name"] for p in r["phases"]]
            self.assertEqual(names, ["full recompute", "incremental update"])  # the biggest energy first
            inc = r["phases"][1]
            self.assertEqual(inc["count"], 3); self.assertAlmostEqual(inc["avg_w"], 250, delta=1e-6)
            self.assertEqual(set(inc), {"name", "count", "energy_j", "duration_s", "median_j", "median_s", "avg_w", "spread_pct"})
            self.assertIn("session saved to project hst-study", err.getvalue()); self.assertIn("#/track?run=77", err.getvalue())
            self.assertEqual(len(H.calls), 1, "saved once")
            # no key, or the server down: the program carries on and is told why
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"JOULES_SIMULATE": "gpu:RTX 3070 Ti:200", "LANDAUER_API_URL": "http://127.0.0.1:9"}), \
                    mock.patch("joules.share.saved_key", return_value=None), \
                    contextlib.redirect_stderr(err):
                os.environ.pop("LANDAUER_API_KEY", None)
                with joules.Session(interval=0.02, track="p") as s2:
                    with s2.phase("x"):
                        time.sleep(0.02)
            self.assertIn("was not saved to WattOps: no API key", err.getvalue())
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"JOULES_SIMULATE": "gpu:RTX 3070 Ti:200", "LANDAUER_API_KEY": key, "LANDAUER_API_URL": "http://127.0.0.1:9"}), contextlib.redirect_stderr(err):
                with joules.Session(interval=0.02, track="p") as s3:
                    pass
            self.assertIn("was not saved to WattOps", err.getvalue()); self.assertIsNone(s3.saved)
            # without track= nothing is sent anywhere
            with mock.patch.dict(os.environ, env):
                with joules.Session(interval=0.02) as s4:
                    with s4.phase("y"):
                        time.sleep(0.02)
            self.assertEqual(len(H.calls), 1)
            with self.assertRaises(ValueError):
                joules.Session(track="../etc")
            with self.assertRaises(ValueError), mock.patch.dict(os.environ, env):
                joules.Session(track="p", grid="mars")
        finally:
            srv.close()

    def test_no_meters_raises(self):
        import joules
        from joules import api
        with mock.patch.object(api, "discover", lambda *a, **k: meters.Discovery([], ["No energy counters found."])):
            with self.assertRaisesRegex(joules.NoMeters, "No energy counters"):
                joules.Session()
        self.assertFalse(hasattr(joules, "nothing_here"))


class CompareStats(unittest.TestCase):
    def test_mann_whitney(self):
        from joules.compare import mann_whitney, min_p
        self.assertAlmostEqual(mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]), 2 / 252)
        self.assertAlmostEqual(mann_whitney([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]), 2 / 252)
        self.assertEqual(mann_whitney([1, 3, 5, 7], [2, 4, 6, 8]), mann_whitney([2, 4, 6, 8], [1, 3, 5, 7]))
        self.assertGreater(mann_whitney([1, 3, 5, 7], [2, 4, 6, 8]), 0.5)
        p = mann_whitney([1, 1, 2, 2, 3], [4, 4, 5, 5, 6])  # ties: the normal approximation
        self.assertTrue(0 < p < 0.05)
        self.assertEqual(mann_whitney([1, 1], [1, 1]), 1.0)
        self.assertIsNone(mann_whitney([1], [2, 3]))
        self.assertAlmostEqual(min_p(3, 3), 0.1); self.assertAlmostEqual(min_p(5, 5), 2 / 252)

    def test_order(self):
        from joules.compare import order
        self.assertEqual(order(["a", "b"], 3, "alternate"), [[0, 1], [0, 1], [0, 1]])
        self.assertEqual(order(["a", "b"], 4, "abba"), [[0, 1], [1, 0], [0, 1], [1, 0]])

    def test_fairness_notes(self):
        from joules.compare import summarize
        def run(name, e, thr, clock=97, t0=50):
            return {"name": name, "energy_j": e, "duration_s": e / 100, "avg_w": 100.0, "limit_pct": None, "temp_start_c": t0,
                    "devices": {"GPU 0": e}, "gpu": {"GPU 0": {"clock_pct": clock, "throttled_pct": thr, "temp_c_max": 70,
                                                                 "throttle_reasons": {"power cap": thr} if thr else {}}}}
        runs = [run("hst", 800 + i, 0) for i in range(5)] + [run("baseline", 1400 + i, 60, t0=65) for i in range(5)]
        b = summarize(runs, ["hst", "baseline"], "baseline")[0]
        hst = next(r for r in b["rows"] if r["name"] == "hst")
        self.assertAlmostEqual(hst["vs"]["energy_pct"], (802 - 1402) / 1402 * 100)
        self.assertAlmostEqual(hst["vs"]["energy_p"], 2 / 252)
        self.assertEqual(hst["gpu"]["throttled_pct"], 0)
        self.assertEqual(next(r for r in b["rows"] if r["name"] == "baseline")["gpu"]["reason"], "power cap")
        self.assertTrue(any("throttled 60% of the time under baseline but 0% under hst" in n for n in b["notes"]))
        self.assertTrue(any("--cool-to" in n for n in b["notes"]))
        self.assertTrue(any("baseline's runs started at 65 °C (median) but hst's at 50 °C" in n for n in b["notes"]))
        fair = summarize([run("a", 100 + i, 0) for i in range(3)] + [run("b", 200 + i, 0) for i in range(3)], ["a", "b"], "a")[0]
        self.assertEqual(len(fair["notes"]), 1); self.assertIn("--runs 5", fair["notes"][0])

    def test_start_temperatures_with_and_without_cool_to(self):
        # A Kaggle T4 with --cool-to 65: the first runs start on a cool card, the rest at the target, alternating, so
        # both commands share the spread. The note must not tell someone who used --cool-to to use --cool-to.
        from joules.compare import summarize
        starts = [48, 52, 58, 60, 63, 64, 64, 65, 65, 64]
        runs = [{"name": "fp16" if i % 2 == 0 else "fp32", "energy_j": 400.0 + i if i % 2 == 0 else 2200.0 + i, "duration_s": 8.0,
                 "avg_w": 55.0, "limit_pct": None, "temp_start_c": t, "devices": {"GPU 0": 400.0}} for i, t in enumerate(starts)]
        with_cool = summarize(runs, ["fp16", "fp32"], "fp32", cool_to=65)[0]["notes"]
        self.assertEqual(len(with_cool), 1, with_cool)
        self.assertIn("between 48 and 65 °C, alike for every command", with_cool[0])
        self.assertIn("--warmup 3", with_cool[0]); self.assertNotIn("--cool-to 5", with_cool[0])
        without = summarize(runs, ["fp16", "fp32"], "fp32")[0]["notes"]
        self.assertIn("--cool-to 58 starts every run from a closer temperature", without[0])
        # one side started hotter as a rule: said as a bias, with the fix that fits
        hot = [dict(r, temp_start_c=70 if r["name"] == "fp32" else 50) for r in runs]
        notes = summarize(hot, ["fp16", "fp32"], "fp32", cool_to=72)[0]["notes"]
        self.assertTrue(any("fp32's runs started at 70 °C (median) but fp16's at 50 °C" in n and "--warmup 3" in n for n in notes), notes)


class CompareIdle(unittest.TestCase):
    def test_idle_that_drifts_is_taken_on_a_line_between_the_two_measurements(self):
        # A desktop settling from 7.6 W to 5.9 W over the comparison: the starting idle taken off every run would
        # under-count the later runs' energy above idle.
        from joules.compare import idle_above
        runs = [{"energy_j": 100.0, "duration_s": 10.0, "_mid": t} for t in (0.0, 50.0, 100.0)]
        notes = idle_above(runs, 7.6, 5.9, 0.0, 100.0)
        self.assertEqual([round(r["idle_w"], 2) for r in runs], [7.6, 6.75, 5.9])
        self.assertAlmostEqual(runs[2]["above_idle_j"], 100 - 59)
        self.assertNotIn("_mid", runs[0])
        self.assertEqual(len(notes), 1); self.assertIn("from 7.6 W before the runs to 5.9 W after them (-22%)", notes[0])
        steady = [{"energy_j": 100.0, "duration_s": 10.0, "_mid": 5.0}]
        self.assertEqual(idle_above(steady, 8.0, 8.2, 0.0, 10.0), [])
        self.assertAlmostEqual(steady[0]["above_idle_j"], 100 - 81)

    def test_the_verdict_names_power_and_energy_above_idle(self):
        # HST on an M1 Max drew more power but finished sooner: the verdict has to show both, or a reader sees watts go up.
        from joules.compare import summarize, verdict_line
        def run(name, e, d, ai):
            return {"name": name, "energy_j": e, "duration_s": d, "avg_w": e / d, "above_idle_j": ai, "limit_pct": None, "devices": {}}
        runs = [run("hst", 143 + i, 11.9, 49 + i) for i in range(5)] + [run("baseline", 215 + i, 18.9, 70 + i) for i in range(5)]
        hst = next(r for r in summarize(runs, ["hst", "baseline"], "baseline")[0]["rows"] if r["name"] == "hst")
        self.assertGreater(hst["vs"]["power_pct"], 0); self.assertLess(hst["vs"]["above_idle_pct"], -25)
        line = verdict_line(hst)
        self.assertIn("average power +", line); self.assertIn("above idle -", line)


class CompareCli(unittest.TestCase):
    SLEEP = '"{py}" -c "import time; time.sleep({s})"'

    def cmd(self, s):
        return self.SLEEP.format(py=sys.executable, s=s)

    def test_interleaved_runs_medians_markdown_and_csv(self):
        d = Path(_CONFIG)
        md, csvf = d / "cmp.md", d / "cmp.csv"
        md.unlink(missing_ok=True)
        code, out, err = run_cli(["compare", "--names", "fast,slow", "--baseline", "slow", "--runs", "2", "--warmup", "1", "--order", "abba",
                                  "--quiet", "--no-cpu", "--json", "--no-track", "--markdown", str(md), "--csv", str(csvf),
                                  self.cmd(0.1), self.cmd(0.6)], {"JOULES_SIMULATE": "gpu:RTX 3070 Ti:200"})  # far apart: Python takes 0.2 s to start on Windows
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual((r["kind"], r["names"], r["baseline"], r["order"], len(r["results"])), ("compare", ["fast", "slow"], "slow", "A B B A", 4))
        self.assertEqual([x["name"] for x in r["results"]], ["fast", "slow", "slow", "fast"])  # warm-up not kept
        b = r["blocks"][0]
        fast = next(x for x in b["rows"] if x["name"] == "fast")
        self.assertLess(fast["vs"]["energy_pct"], -40); self.assertLess(fast["vs"]["time_pct"], -40)
        self.assertAlmostEqual(fast["avg_w"]["median"], 200, delta=1)
        self.assertIn("--runs 5", b["notes"][0])
        self.assertIn("fast vs slow: energy -", md.read_text(encoding="utf-8"))
        rows = csvf.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 5); self.assertTrue(rows[0].startswith("name,limit_pct,round,position"))
        self.assertIn("J GPU RTX 3070 Ti", rows[0])

    def test_commands_split_like_the_shell_of_each_system(self):
        from joules import compare as cmp
        win = cmp.split_command(r'"C:\Program Files\Python\python.exe" -c "import time; time.sleep(0.1)" C:\data\a#1.txt', windows=True)
        self.assertEqual(win, [r"C:\Program Files\Python\python.exe", "-c", "import time; time.sleep(0.1)", r"C:\data\a#1.txt"])
        self.assertEqual(cmp.split_command(r'python -c "print(1)" a\ b', windows=False), ["python", "-c", "print(1)", "a b"])

    def test_bad_arguments_and_a_failing_command(self):
        env = {"JOULES_SIMULATE": "gpu:X:100"}
        self.assertEqual(run_cli(["compare", self.cmd(0)], env)[0], 2)
        self.assertEqual(run_cli(["compare", "--names", "a", self.cmd(0), self.cmd(0)], env)[0], 2)
        self.assertEqual(run_cli(["compare", "--names", "a,a", self.cmd(0), self.cmd(0)], env)[0], 2)
        self.assertEqual(run_cli(["compare", "--baseline", "zzz", self.cmd(0), self.cmd(0)], env)[0], 2)
        self.assertEqual(run_cli(["compare", "--limits", "150", self.cmd(0), self.cmd(0)], env)[0], 2)
        self.assertEqual(run_cli(["compare", "--limits", "70", "--no-track", self.cmd(0), self.cmd(0)], env)[0], 1)  # no NVIDIA here
        code, _, err = run_cli(["compare", "--runs", "1", "--warmup", "0", "--no-track", self.cmd(0),
                                f'"{sys.executable}" -c "raise SystemExit(3)"'], env)
        self.assertEqual(code, 1); self.assertIn("failed (exit 3)", err)
        code, out, _ = run_cli(["compare", "--runs", "1", "--warmup", "0", "--no-track", "--json", self.cmd(0), self.cmd(0.01)], env)
        self.assertEqual(json.loads(out)["names"], ["A", "B"]) if code == 0 else self.fail("compare failed")

    def test_idle_is_measured_before_and_after_the_runs(self):
        code, out, err = run_cli(["compare", "--runs", "1", "--warmup", "0", "--no-track", "--no-cpu", "--quiet", "--json",
                                  "--subtract-idle", "0.3", self.cmd(0), self.cmd(0.01)], {"JOULES_SIMULATE": "gpu:X:100"})
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertTrue(r["idle_w"]); self.assertTrue(r["idle_w_end"])
        self.assertTrue(all("idle_w" in x and "above_idle_j" in x and "_mid" not in x for x in r["results"]))

    def test_power_limit_sweep_cool_down_and_tracking(self):
        nv = FakeNVMLHealth([{"name": "NVIDIA GeForce RTX 3070 Ti", "power_mw": 250_000, "util": (96, 30), "temp": 55}])
        H = board_handler(201, {"id": 1}); srv = _Server(H)
        try:
            with tempfile.TemporaryDirectory() as t, mock.patch.object(meters, "NVML_FACTORY", lambda: nv), \
                    mock.patch.dict(os.environ, {"JOULES_SYSFS": t, "CUDA_VISIBLE_DEVICES": "0"}):
                os.environ.pop("JOULES_SIMULATE", None)
                code, out, err = run_cli(["compare", "--names", "hst,baseline", "--runs", "1", "--warmup", "0", "--limits", "100,50",
                                          "--cool-to", "60", "--no-cpu", "--quiet", "--json", "--track", "hst-study", "--key", Track.KEY,
                                          "--api-url", srv.url, self.cmd(0.6), self.cmd(0.6)])
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual([b["limit_pct"] for b in r["blocks"]], [100, 50])
        self.assertEqual([b["limit_w"] for b in r["blocks"]], [[700.0], [350.0]])
        self.assertEqual(nv.sets, [(0, 700_000), (0, 350_000), (0, 700_000)])  # and the original limit back
        self.assertEqual(nv.limits[0], 700_000)
        capped = r["blocks"][1]["rows"][0]
        self.assertEqual((capped["gpu"]["throttled_pct"], capped["gpu"]["reason"]), (100, "power cap"))
        self.assertAlmostEqual(capped["avg_w"]["median"], 250, delta=5)
        self.assertEqual(r["results"][0]["temp_start_c"], 55)
        sent = [c[2] for c in H.calls]
        self.assertEqual([s["label"] for s in sent], ["hst @ 100% power", "baseline @ 100% power", "hst @ 50% power", "baseline @ 50% power"])
        self.assertEqual({s["project"] for s in sent}, {"hst-study"})
        self.assertEqual(sent[2]["gpu"]["reason"], "power cap")
        self.assertNotIn("time.sleep", json.dumps(sent))  # commands' arguments never leave the machine
        from joules import compare
        self.assertIn("── power limit 50% (350 W)", compare.render(r))

    def test_cool_to_waits_for_the_card(self):
        from joules import compare
        temps = iter([80, 72, 64, 58])
        class G:
            def temperature_now(self): return next(temps)
        said = []
        with mock.patch.object(compare.time, "sleep", lambda s: None):
            self.assertEqual(compare.wait_cool([G()], 60, 100, said.append), 58)
        self.assertEqual(said, ["waiting for the GPU to cool from 80 °C to 60 °C…"])


class CompareTrackPacing(unittest.TestCase):
    def test_waits_out_the_per_minute_limit_and_loses_no_run(self):
        class H(BaseHTTPRequestHandler):
            calls, refused = [], 0
            def log_message(self, *a): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if len(H.calls) == 2 and H.refused == 0:  # the third upload hits the limit once
                    H.refused += 1
                    data = json.dumps({"error": {"code": "rate_limited", "message": "Too many requests: 10 per minute on the free plan."}}).encode()
                    self.send_response(429); self.send_header("Retry-After", "7")
                else:
                    H.calls.append(body); data = b'{"id": 1}'; self.send_response(201)
                self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        srv = _Server(H); slept = []
        try:
            with mock.patch.object(cli.time, "sleep", slept.append):
                code, _, err = run_cli(["compare", "--names", "a,b", "--runs", "2", "--warmup", "0", "--quiet", "--no-cpu", "--track", "p",
                                        "--key", Track.KEY, "--api-url", srv.url, CompareCli().cmd(0), CompareCli().cmd(0.01)],
                                       {"JOULES_SIMULATE": "gpu:X:100"})
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        self.assertEqual(len(H.calls), 4); self.assertEqual(H.refused, 1)
        self.assertEqual(slept, [7.0])
        self.assertIn("tracked 4 runs in project p", err); self.assertIn("a limited number of runs a minute", err)

    def test_a_monthly_quota_stops_and_says_how_many_went(self):
        H = board_handler(429, {"error": {"code": "quota_exceeded", "message": "The free plan's 100 API calls this month are used."}}); srv = _Server(H)
        try:
            code, _, err = run_cli(["compare", "--runs", "1", "--warmup", "0", "--quiet", "--no-cpu", "--track", "p", "--key", Track.KEY,
                                    "--api-url", srv.url, CompareCli().cmd(0), CompareCli().cmd(0.01)], {"JOULES_SIMULATE": "gpu:X:100"})
        finally:
            srv.close()
        self.assertEqual(code, 0)  # the comparison worked; tracking is reported, not fatal
        self.assertIn("0 of 2 runs tracked", err); self.assertIn("100 API calls", err)
        self.assertEqual(len(H.calls), 1)


class NewGpus(unittest.TestCase):
    def test_names_pick_their_specs(self):
        self.assertEqual(model.gpu_from_name("GPU 0 NVIDIA GeForce RTX 3070 Ti"), "rtx3070ti")
        self.assertIsNone(model.gpu_from_name("GPU 0 NVIDIA GeForce RTX 3070"))  # not the Ti: a different card
        self.assertEqual(model.gpu_from_name("GPU 0 NVIDIA GeForce RTX 5090"), "rtx5090")
        self.assertEqual(model.gpu_from_name("GPU 1 NVIDIA RTX PRO 6000 Blackwell Workstation Edition"), "rtxpro6000")
        self.assertEqual(model.GPUS["rtx3070ti"][1:], (290, 43.5))


class NvmlLibrarySearch(unittest.TestCase):
    """The driver's library in a folder the loader does not search (Google Colab, WSL, containers) is still found."""
    @unittest.skipIf(os.name == "nt" or not shutil.which("gcc"), "needs gcc and a Linux loader")
    def test_found_in_a_driver_folder_the_loader_does_not_know(self):
        src = Path(__file__).parent / "fake_nvml" / "fake_nvml.c"
        with tempfile.TemporaryDirectory() as t:
            d = os.path.join(t, "lib64-nvidia"); os.makedirs(d)
            subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", os.path.join(d, "libnvidia-ml.so.1"), str(src)], check=True)
            env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
            env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
            code = ("import sys; from joules import meters\n"
                    "meters.NVML_DIRS = [sys.argv[1]]; meters._ldconfig_nvml = lambda: []\n"
                    "n = meters.NVML(); print(n.count(), n.name(n.handle(1)))")
            out = subprocess.run([sys.executable, "-c", code, d], capture_output=True, text=True, env=env)
            self.assertEqual(out.stdout.strip(), "2 Tesla T4", out.stderr)
            # JOULES_NVML_LIB names it outright, wherever it is
            code2 = "from joules import meters; meters.NVML_DIRS = []; meters._ldconfig_nvml = lambda: []; print(meters.NVML().count())"
            out = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True,
                                 env={**env, "JOULES_NVML_LIB": os.path.join(d, "libnvidia-ml.so.1")})
            self.assertEqual(out.stdout.strip(), "2", out.stderr)

    @unittest.skipIf(os.name == "nt", "Linux search order")
    def test_search_order_and_no_cuda_stubs(self):
        with tempfile.TemporaryDirectory() as t:
            Path(t, "libnvidia-ml.so.1").write_text("")
            with mock.patch.object(meters, "NVML_DIRS", [t, "/nope"]), \
                    mock.patch.object(meters, "_ldconfig_nvml", lambda: ["/usr/local/cuda/lib64/stubs/libnvidia-ml.so", "/opt/drv/libnvidia-ml.so.1"]), \
                    mock.patch.dict(os.environ, {"JOULES_NVML_LIB": "/my/libnvidia-ml.so.1"}):
                names = meters.nvml_library_names()
        self.assertEqual(names, ["/my/libnvidia-ml.so.1", "libnvidia-ml.so.1", "libnvidia-ml.so", os.path.join(t, "libnvidia-ml.so.1"), "/opt/drv/libnvidia-ml.so.1"])
        with tempfile.TemporaryDirectory() as t2:  # only the versioned file, no links
            Path(t2, "libnvidia-ml.so.550.54.15").write_text("")
            with mock.patch.object(meters, "NVML_DIRS", [t2]), mock.patch.object(meters, "_ldconfig_nvml", lambda: []):
                os.environ.pop("JOULES_NVML_LIB", None)
                self.assertEqual(meters.nvml_library_names()[-1], os.path.join(t2, "libnvidia-ml.so.550.54.15"))

    def test_a_driver_that_cannot_be_loaded_is_named_not_hidden(self):
        from joules import doctor
        def no_lib(name): raise OSError("cannot open")
        with mock.patch.object(meters.ctypes, "CDLL", no_lib), mock.patch.object(meters, "nvidia_driver_present", lambda: True), \
                mock.patch.object(meters, "_ldconfig_nvml", lambda: []):
            with self.assertRaisesRegex(OSError, r"driver is installed .*\(cannot open\).* JOULES_NVML_LIB"):
                meters.NVML()
            notes = []
            self.assertEqual(meters.discover_nvml(notes, None), ([], None))
            self.assertIn("JOULES_NVML_LIB", notes[0])
            r = doctor.Report(); doctor.check_gpus(r)
        text = doctor.render(r, color=False, ascii_only=False)
        self.assertIn("An NVIDIA driver is installed but its library", text)
        self.assertEqual(r.counts()[doctor.WARN], 1)
        with mock.patch.object(meters.ctypes, "CDLL", no_lib), mock.patch.object(meters, "nvidia_driver_present", lambda: False), \
                mock.patch.object(meters, "_ldconfig_nvml", lambda: []):
            with self.assertRaisesRegex(OSError, "was not found"):
                meters.NVML()


class FakeTorch:
    """Just enough of PyTorch for joules bench gpu: matmul sleeps, fp16 faster than fp32 (or far slower, on request)."""
    float16, float32 = "fp16", "fp32"
    __version__ = "2.5.1"

    def __init__(self, names=("Tesla T4",), free=15 << 30, fp16_s=0.004, fp32_s=0.012):
        torch = self
        self.names, self.free, self.speed, self.calls = list(names), free, {"fp16": fp16_s, "fp32": fp32_s}, []
        self.version = type("V", (), {"cuda": "12.4", "hip": None})()
        self.backends = type("B", (), {})(); self.backends.cuda = type("C", (), {})(); self.backends.cuda.matmul = type("M", (), {"allow_tf32": True})()

        class Cuda:
            def is_available(self): return bool(torch.names)
            def device_count(self): return len(torch.names)
            def get_device_name(self, i): return torch.names[i]
            def mem_get_info(self, i): return (torch.free, 16 << 30)
            def synchronize(self, dev=None): pass
            def empty_cache(self): pass
        self.cuda = Cuda()

    def device(self, s): return s
    def Generator(self, device=None):
        g = type("G", (), {})(); g.manual_seed = lambda seed: g; return g
    def randn(self, *shape, device=None, dtype=None, generator=None): return {"dtype": dtype, "tf32": self.backends.cuda.matmul.allow_tf32}
    def empty_like(self, a): return dict(a)
    def matmul(self, a, b, out=None):
        self.calls.append((a["dtype"], self.backends.cuda.matmul.allow_tf32)); time.sleep(self.speed[a["dtype"]])


class GpuBench(unittest.TestCase):
    T4 = {"name": "Tesla T4", "power_mw": 70_000, "limit_mw": 70_000, "capped": True, "util": (100, 60)}
    KEY = "lgk_" + "g" * 40

    def bench(self, nv_devices=None, torch=None, **kw):
        from joules import gpubench
        nv = FakeNVMLHealth(nv_devices or [dict(self.T4)])
        with mock.patch.object(meters, "NVML_FACTORY", lambda: nv), mock.patch.object(meters, "discover_rapl", lambda notes: []):
            os.environ.pop("JOULES_SIMULATE", None)
            return gpubench.run(0, torch=torch or FakeTorch(), n=64, seconds=0.7, warmup_s=0.05, interval=0.02, **kw)

    def test_runs_both_precisions_on_the_card_pytorch_uses(self):
        t = FakeTorch()
        r = self.bench(torch=t)
        self.assertEqual((r["kind"], r["test"], r["device"], r["meter"], r["simulated"]), ("gpubench", "matmul-8192-v1", "Tesla T4", "nvml power sampling", False))
        self.assertEqual(list(r["results"]), ["fp16", "fp32"])
        for p, x in r["results"].items():
            self.assertGreaterEqual(x["duration_s"], 0.7)
            self.assertEqual(x["flops"], 2 * 64 ** 3 * x["multiplications"])
            self.assertAlmostEqual(x["avg_w"], 70, delta=3)  # the meter integrates the card's 70 W
            self.assertAlmostEqual(x["gflop_per_j"], x["flops"] / x["energy_j"] / 1e9)
            self.assertAlmostEqual(x["tflops"], x["flops"] / x["duration_s"] / 1e12)
            self.assertEqual(x["throttled_pct"], 100)
        self.assertGreater(r["results"]["fp16"]["multiplications"], r["results"]["fp32"]["multiplications"])
        self.assertTrue(all(not tf32 for _, tf32 in t.calls), "fp32 runs without TF32")
        self.assertTrue(t.backends.cuda.matmul.allow_tf32, "and the setting is put back")
        self.assertEqual((r["power_limit_w"], r["runtime"], r["torch"]), (70.0, "CUDA 12.4", "2.5.1"))

    def test_refuses_what_it_cannot_measure_honestly(self):
        from joules import gpubench
        with self.assertRaisesRegex(gpubench.GpuBenchError, "there is no GPU 0"):
            self.bench(torch=FakeTorch(names=()))
        with self.assertRaisesRegex(gpubench.GpuBenchError, "GB free"):
            self.bench(torch=FakeTorch(free=1 << 20))
        with self.assertRaisesRegex(gpubench.GpuBenchError, "is a Tesla T4, but the meter joules found is NVIDIA L4"):
            self.bench([{"name": "NVIDIA L4", "power_mw": 72_000}])
        with self.assertRaisesRegex(gpubench.GpuBenchError, "only 3% busy"):  # the meter's card sat idle: wrong card
            self.bench([dict(self.T4, util=(3, 1))])

    def test_a_card_without_fast_fp16_is_measured_in_fp32_only(self):
        from joules import gpubench
        # fp32 takes 12 ms and fp16 0.5 s against a 0.2 s line, so a slow, busy machine still lands each on its side
        with mock.patch.object(gpubench, "TOO_SLOW_S", 0.2):
            r = self.bench(torch=FakeTorch(fp16_s=0.5))
        self.assertEqual(list(r["results"]), ["fp32"])
        self.assertTrue(any("fp16 is not measured" in n for n in r["notes"]))

    def test_cuda_visible_devices_picks_the_nvml_card(self):
        from joules import gpubench
        nv = FakeNVML([{"name": "Tesla T4"}, {"name": "Tesla T4"}, {"name": "Tesla T4"}])
        with mock.patch.object(meters, "NVML_FACTORY", lambda: nv):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}):
                self.assertEqual((gpubench.nvml_index(0), gpubench.nvml_index(1)), (2, 0))
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                self.assertEqual(gpubench.nvml_index(1), 1)
        self.assertTrue(gpubench.same_card("Tesla T4", "GPU 0 Tesla T4"))
        self.assertTrue(gpubench.same_card("NVIDIA GeForce RTX 4090", "GPU 1 NVIDIA GeForce RTX 4090"))
        self.assertFalse(gpubench.same_card("NVIDIA L4", "GPU 0 NVIDIA A100-SXM4-40GB"))

    def test_payload_and_cli_share(self):
        from joules import gpubench, share
        r = self.bench()
        m = share.gpu_measurement(r)
        self.assertEqual((m["test"], m["device"], m["meter"], m["power_limit_w"]), ("matmul-8192-v1", "Tesla T4", "power", 70.0))
        self.assertEqual(set(m["results"]["fp16"]), {"multiplications", "energy_j", "duration_s", "throttled_pct", "clock_pct", "temp_c_end"})
        self.assertNotIn("torch_name", m); self.assertNotIn("gpu", m)
        with self.assertRaisesRegex(ValueError, "simulated"):
            share.gpu_measurement(dict(r, simulated=True))
        with self.assertRaisesRegex(ValueError, "only `joules bench gpu`"):
            share.gpu_measurement(bench_report())
        # the command: runs the test, prints it, shares it, prints the board link
        H = board_handler(); srv = _Server(H)
        real = gpubench.run
        try:
            with mock.patch.object(gpubench, "run", lambda gpu, **kw: r):
                code, _, err = run_cli(["bench", "gpu", "--share", "--api-url", srv.url], {"LANDAUER_API_KEY": self.KEY})
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        self.assertIn("joules bench gpu · Tesla T4 (GPU 0)", err)
        self.assertRegex(err, r"fp16 +[\d.]+ TFLOP/s +\d+ W +\d+ GFLOP/J")
        self.assertIn("throttled 100% of the time (power cap)", err)
        path, auth, body = H.calls[0]
        self.assertEqual((path, auth, body["device"]), ("/api/v1/measurements?kind=gpu", "Bearer " + self.KEY, "Tesla T4"))
        self.assertIn(srv.url + "/leaderboard?gpu=Tesla%20T4", err)
        self.assertIs(gpubench.run, real)

    def test_cli_explains_problems(self):
        from joules import gpubench
        def boom(gpu, **kw): raise gpubench.GpuBenchError("PyTorch sees no GPU. On Kaggle, pick a GPU")
        with mock.patch.object(gpubench, "run", boom):
            code, _, err = run_cli(["bench", "gpu"])
        self.assertEqual(code, 1); self.assertIn("joules bench gpu: PyTorch sees no GPU", err)
        code, _, err = run_cli(["bench", "gpu", "--gpus", "0,1"])
        self.assertEqual(code, 2); self.assertIn("one GPU at a time", err)
        code, _, err = run_cli(["bench", "gpu", "--share"])  # no key: says so before a minute of testing
        self.assertEqual(code, 2); self.assertIn("no API key yet", err)
        with mock.patch.dict(sys.modules, {"torch": None}):  # PyTorch not installed
            with self.assertRaisesRegex(gpubench.GpuBenchError, "pytorch.org"):
                gpubench.load_torch()

    @unittest.skipIf(os.name == "nt", "a shell script stands in for python3")
    def test_finds_pytorch_in_another_python(self):
        # install.sh puts joules in its own environment; the user's PyTorch is in theirs. joules runs the test there.
        from joules import gpubench
        with tempfile.TemporaryDirectory() as t:
            bin_, site = Path(t, "bin"), Path(t, "site")
            write(site / "torch" / "__init__.py", "")
            write(bin_ / "python3", f'#!/bin/sh\nPYTHONPATH="{site}:$PYTHONPATH" exec "{sys.executable}" "$@"\n')
            os.chmod(bin_ / "python3", 0o755)
            with mock.patch.dict(os.environ, {"PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}"}):
                os.environ.pop("JOULES_NO_REEXEC", None); os.environ.pop("PYTHONPATH", None)
                py = gpubench.torch_python()
                self.assertEqual(py, str(bin_ / "python3"))
                out = subprocess.run([py, "-c", "import joules"], capture_output=True)  # not importable there by itself
                self.assertNotEqual(out.returncode, 0)
                with mock.patch.object(subprocess, "call", wraps=subprocess.call) as call:
                    self.assertEqual(gpubench.run_with(py, ["--version"]), 0)  # but run_with brings joules along
                self.assertEqual(call.call_args.kwargs["env"]["JOULES_NO_REEXEC"], "1")
                with mock.patch.dict(os.environ, {"JOULES_NO_REEXEC": "1"}):
                    self.assertIsNone(gpubench.torch_python())  # and the second Python does not go looking again

    def test_cli_hands_over_to_the_python_with_pytorch(self):
        from joules import gpubench
        def no_torch(gpu, **kw): raise gpubench.NoTorch("the GPU test runs on PyTorch, which is not installed here")
        ran = []
        with mock.patch.object(gpubench, "run", no_torch), mock.patch.object(gpubench, "torch_python", lambda: "/opt/conda/bin/python3"), \
                mock.patch.object(gpubench, "run_with", lambda py, argv: ran.append((py, argv)) or 0):
            code, _, err = run_cli(["bench", "gpu", "--gpus", "1"])
        self.assertEqual((code, ran), (0, [("/opt/conda/bin/python3", ["bench", "gpu", "--gpus", "1"])]))
        self.assertIn("running the test with /opt/conda/bin/python3", err)
        with mock.patch.object(gpubench, "run", no_torch), mock.patch.object(gpubench, "torch_python", lambda: None):
            code, _, err = run_cli(["bench", "gpu"])
        self.assertEqual(code, 1); self.assertIn("joules bench gpu: the GPU test runs on PyTorch, which is not installed here", err)


FAKE_SMI = r'''
import json, os, sys
a = sys.argv[1:]
with open(os.environ["FAKE_SMI_LOG"], "a") as f:
    f.write(json.dumps(a) + "\n")
mode = os.environ.get("FAKE_SMI_MODE", "ok")
idx = a[a.index("-i") + 1] if "-i" in a else "0"
if any(x.startswith("--query-gpu=clocks") for x in a):
    print(os.environ.get("FAKE_SMI_CLOCKS", "1980, 1395")); sys.exit(0)
if any(x.startswith("--query-gpu=") for x in a):
    print("Tesla T4"); sys.exit(0)
if "-lgc" in a:
    if mode == "denied":
        print("Setting locked GPU clocks requires root."); print("Insufficient Permissions"); sys.exit(4)
    if mode == "mig":
        print("Unable to set locked clocks: not supported while MIG mode is enabled"); sys.exit(3)
    if mode == "fail1" and idx == "1":
        print("Setting locked GPU clocks is not supported for GPU 00000000:00:05.0."); sys.exit(3)
    print(f'GPU clocks set to "(gpuClkMin {a[a.index("-lgc") + 1].split(",")[0]}, gpuClkMax ...)" for GPU 0000'); print("All done."); sys.exit(0)
if "-rgc" in a:
    print("All done."); sys.exit(0)
sys.exit(0)
'''


@unittest.skipIf(os.name == "nt", "a Python script stands in for nvidia-smi")
class LockClocks(unittest.TestCase):
    T4S = [{"name": "Tesla T4", "power_mw": 70_000, "limit_mw": 700_000, "util": (99, 50), "temp": 55},
           {"name": "Tesla T4", "power_mw": 70_000, "limit_mw": 700_000, "util": (99, 50), "temp": 55}]

    def setUp(self):
        from joules import compare
        self.tmp = tempfile.TemporaryDirectory()
        self.smi = Path(self.tmp.name) / "nvidia-smi"
        self.smi.write_text(f"#!{sys.executable}\n" + FAKE_SMI); self.smi.chmod(0o755)
        self.log = Path(self.tmp.name) / "calls.jsonl"
        self.env = {"FAKE_SMI_LOG": str(self.log), "JOULES_SYSFS": self.tmp.name}
        self.patches = [mock.patch.object(compare, "SMI", str(self.smi))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def changes(self):
        return [c for c in self.calls() if "-lgc" in c or "-rgc" in c]

    def compare(self, *extra, devices=None, env=None, cmds=None):
        nv = FakeNVMLHealth([dict(x) for x in (devices or self.T4S)])
        with mock.patch.object(meters, "NVML_FACTORY", lambda: nv), mock.patch.dict(os.environ, {**self.env, **(env or {})}):
            os.environ.pop("JOULES_SIMULATE", None); os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            res = run_cli(["compare", "--names", "fp16,fp32", "--runs", "1", "--warmup", "1", "--no-cpu", "--quiet", "--json", "--no-track",
                           *extra, *(cmds or [CompareCli().cmd(0.6), CompareCli().cmd(0.6)])])
        return res, nv

    def test_locks_every_measured_gpu_and_always_unlocks(self):
        (code, out, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0,1")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.changes(), [["-i", "0", "-lgc", "1590,1590"], ["-i", "1", "-lgc", "1590,1590"], ["-i", "0", "-rgc"], ["-i", "1", "-rgc"]])
        r = json.loads(out)
        self.assertEqual(r["locked_mhz"], {"GPU 0 Tesla T4": 1590, "GPU 1 Tesla T4": 1590})
        self.assertEqual(r["gpu_clocks"], "locked")
        self.assertTrue(all(x["locked_mhz"] == r["locked_mhz"] for x in r["results"]))
        self.assertIn("GPU 0: SM clock locked at 1590 MHz", err); self.assertIn("GPU 1: clocks unlocked (nvidia-smi -rgc)", err)
        # the fake card runs at 1980 MHz, above the lock: held
        self.assertTrue(any("Clocks held at 1590 MHz under every command" in n for n in r["blocks"][0]["notes"]), r["blocks"][0]["notes"])
        from joules import compare
        text = compare.render(r)
        self.assertIn("SM clock locked at 1590 MHz · GPU 0 Tesla T4", text); self.assertRegex(text, r"clock\s+MHz\s+thrott\.")
        md = compare.markdown(r)
        self.assertIn("SM clock locked at 1590 MHz (nvidia-smi -lgc) for every run, warm-ups included", md)
        self.assertIn("| GPU clock | SM clock | throttled | max temp |", md)
        rows = compare.csv_rows(r)
        self.assertEqual(rows[1][rows[0].index("locked_mhz")], "1590")

    def test_unlocked_runs_say_so(self):
        (code, out, err), _ = self.compare("--gpus", "0")
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual((r["locked_mhz"], r["gpu_clocks"]), (None, "boost"))
        self.assertEqual(self.changes(), [])  # nothing touched without the flag
        from joules import compare
        self.assertIn("clocks not locked · GPU 0 Tesla T4", compare.render(r))
        self.assertIn("GPU clocks: not locked (the card boosts and throttles", compare.markdown(r))
        rows = compare.csv_rows(r)
        self.assertIsNone(rows[1][rows[0].index("locked_mhz")])

    def test_base_clock(self):
        (code, out, err), _ = self.compare("--lock-clocks", "base", "--gpus", "0")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.changes()[0], ["-i", "0", "-lgc", "1395,1395"])  # the default applications clock nvidia-smi reports
        self.assertEqual(json.loads(out)["locked_mhz"], {"GPU 0 Tesla T4": 1395})
        (code, _, err), _ = self.compare("--lock-clocks", "base", "--gpus", "0", env={"FAKE_SMI_CLOCKS": "1980, [N/A]"})
        self.assertEqual(code, 1); self.assertIn("cannot lock clocks on GPU 0: the driver does not report a base clock", err)

    def test_refused_lock_stops_before_anything_runs(self):
        marker = Path(self.tmp.name) / "ran"
        touch = f'"{sys.executable}" -c "open(r\'{marker}\', \'a\').write(\'x\')"'
        from joules import compare
        # Kaggle: root inside a container, and the driver still says no. sudo cannot help, so joules does not suggest it.
        with mock.patch.object(compare, "is_admin", lambda: True):
            (code, _, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0", env={"FAKE_SMI_MODE": "denied"}, cmds=[touch, touch])
        self.assertEqual(code, 1)
        self.assertIn("cannot lock clocks on GPU 0: the driver refused even as root, so this machine does not allow clock changes", err)
        self.assertNotIn("sudo", err); self.assertFalse(marker.exists())
        self.log.unlink(missing_ok=True)
        for mode, msg in [("denied", "cannot lock clocks on GPU 0: insufficient permissions (needs root); run without --lock-clocks or with sudo"),
                          ("mig", "cannot lock clocks on GPU 0: the GPU is in MIG mode, where clocks cannot be locked; run without --lock-clocks")]:
            self.log.unlink(missing_ok=True)
            with mock.patch.object(compare, "is_admin", lambda: False):
                (code, out, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0", env={"FAKE_SMI_MODE": mode}, cmds=[touch, touch])
            self.assertEqual(code, 1); self.assertIn("joules compare: " + msg, err)
            self.assertFalse(marker.exists(), "no command ran")
            self.assertEqual(self.changes(), [["-i", "0", "-lgc", "1590,1590"]])  # nothing locked, so nothing to reset
        # the second GPU refuses: the first is unlocked again before joules stops
        self.log.unlink(missing_ok=True)
        (code, _, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0,1", env={"FAKE_SMI_MODE": "fail1"}, cmds=[touch, touch])
        self.assertEqual(code, 1); self.assertIn("cannot lock clocks on GPU 1: this GPU or driver does not support locked clocks", err)
        self.assertEqual(self.changes(), [["-i", "0", "-lgc", "1590,1590"], ["-i", "1", "-lgc", "1590,1590"], ["-i", "0", "-rgc"]])
        self.assertFalse(marker.exists())
        # above the card's maximum, and a bad value
        (code, _, err), _ = self.compare("--lock-clocks", "2500", "--gpus", "0")
        self.assertEqual(code, 1); self.assertIn("2500 MHz is above its maximum of 1980 MHz", err)
        self.assertEqual(run_cli(["compare", "--lock-clocks", "fast", "a", "b"])[0], 2)
        # no NVIDIA GPU read through the driver
        code, _, err = run_cli(["compare", "--lock-clocks", "1590", "--runs", "1", "--warmup", "0", "--no-track",
                                CompareCli().cmd(0), CompareCli().cmd(0)], {"JOULES_SIMULATE": "gpu:X:100"})
        self.assertEqual(code, 1); self.assertIn("--lock-clocks needs an NVIDIA GPU", err)

    def test_unlocked_after_a_failed_run_and_after_ctrl_c(self):
        (code, _, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0", cmds=[CompareCli().cmd(0), f'"{sys.executable}" -c "raise SystemExit(3)"'])
        self.assertEqual(code, 1); self.assertIn("failed (exit 3)", err)
        self.assertEqual(self.changes(), [["-i", "0", "-lgc", "1590,1590"], ["-i", "0", "-rgc"]])
        self.log.unlink()
        from joules import compare
        real = compare.measure
        def interrupted(*a, **k): raise KeyboardInterrupt
        with mock.patch.object(compare, "measure", interrupted):
            (code, _, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0")
        self.assertEqual(code, 130); self.assertIn("stopped", err)
        self.assertEqual(self.changes(), [["-i", "0", "-lgc", "1590,1590"], ["-i", "0", "-rgc"]])
        self.assertIs(compare.measure, real)

    def test_with_power_limits_and_cool_to(self):
        # clocks locked first, then the limits stepped; at 50% the card cannot hold 1590 MHz and the note says so
        (code, out, err), nv = self.compare("--lock-clocks", "1590", "--gpus", "0", "--limits", "100,50", "--cool-to", "60")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.changes(), [["-i", "0", "-lgc", "1590,1590"], ["-i", "0", "-rgc"]])
        self.assertEqual(nv.limits[0], 700_000)  # power limit back too
        r = json.loads(out)
        held, capped = r["blocks"]
        self.assertTrue(any("Clocks held at 1590 MHz" in n for n in held["notes"]), held["notes"])
        note = next(n for n in capped["notes"] if n.startswith("Clocks were locked at 1590 MHz"))
        self.assertIn("still ran below that under fp16 (average 990 MHz, throttled 100% of the time: power cap)", note)
        self.assertIn("--lock-clocks base", note)
        self.assertEqual(r["results"][0]["temp_start_c"], 55)
        self.assertIn("── power limit 50%", __import__("joules.compare", fromlist=["render"]).render(r))

    def test_tracked_runs_say_the_lock(self):
        H = board_handler(201, {"id": 1}); srv = _Server(H)
        try:
            (code, _, err), _ = self.compare("--lock-clocks", "1590", "--gpus", "0", "--limits", "100", "--track", "t4", "--key", Track.KEY,
                                             "--api-url", srv.url)
        finally:
            srv.close()
        self.assertEqual(code, 0, err)
        self.assertEqual([c[2]["label"] for c in H.calls], ["fp16 @ 100% power, 1590 MHz locked", "fp32 @ 100% power, 1590 MHz locked"])

    def test_notes_when_locked(self):
        from joules.compare import summarize
        lock = {"GPU 0": 1590}
        def run(name, e, sm, thr):
            return {"name": name, "energy_j": e, "duration_s": e / 100, "avg_w": 100.0, "limit_pct": None, "devices": {"GPU 0": e},
                    "gpu": {"GPU 0": {"clock_pct": sm / 19.8, "sm_clock_mhz": sm, "throttled_pct": thr, "temp_c_max": 70,
                                      "throttle_reasons": {"power cap": thr} if thr else {}}}}
        dropped = summarize([run("fp16", 300 + i, 1585, 0) for i in range(5)] + [run("fp32", 2000 + i, 1200, 92) for i in range(5)],
                            ["fp16", "fp32"], "fp32", locks=lock)[0]
        self.assertEqual([r["gpu"]["vs_lock_pct"] for r in dropped["rows"]], [1585 / 1590 * 100, 1200 / 1590 * 100])
        self.assertTrue(any("still ran below that under fp32 (average 1200 MHz, throttled 92% of the time: power cap)" in n for n in dropped["notes"]))
        self.assertFalse(any("throttled 92% of the time under fp32 but" in n for n in dropped["notes"]))  # the unlocked wording is replaced
        held = summarize([run("fp16", 300 + i, 1588, 0) for i in range(5)] + [run("fp32", 2000 + i, 1590, 0) for i in range(5)],
                         ["fp16", "fp32"], "fp32", locks=lock)[0]
        self.assertEqual(held["notes"], ["Clocks held at 1590 MHz under every command (the GPU never dropped below the lock), so the "
                                         "difference is the work, not boost or throttling."])
        unlocked = summarize([run("fp16", 300 + i, 1585, 0) for i in range(5)] + [run("fp32", 2000 + i, 1200, 92) for i in range(5)],
                             ["fp16", "fp32"], "fp32")[0]
        self.assertTrue(any("--lock-clocks base, needs root" in n for n in unlocked["notes"]))

    def test_doctor_says_whether_locking_is_permitted_and_changes_nothing(self):
        from joules import doctor
        with mock.patch.dict(os.environ, {**self.env, "PATH": f"{self.tmp.name}{os.pathsep}{os.environ['PATH']}"}):
            with mock.patch.object(doctor, "_admin", lambda: True):
                st, what, detail = doctor.clock_lock_status()
            self.assertEqual((st, what), (doctor.OK, "clock locking (--lock-clocks)"))
            self.assertIn("permitted to try: running as root (GPU 0 max 1980 MHz, base 1395 MHz)", detail)
            self.assertIn("Kaggle, Colab", detail)
            with mock.patch.object(doctor, "_admin", lambda: False), mock.patch.object(doctor.subprocess, "call", lambda *a, **k: 1):
                st, _, detail = doctor.clock_lock_status()
            self.assertEqual(st, doctor.INFO); self.assertIn("needs root: run joules compare --lock-clocks with sudo", detail)
        self.assertEqual(self.changes(), [])  # doctor only asked
        with mock.patch.dict(os.environ, {"PATH": self.tmp.name + "/none"}):
            self.assertIn("nvidia-smi is not on PATH", doctor.clock_lock_status()[2])


from joules import connection as cn  # noqa: E402
REAL_PROVIDER = cn.provider


@unittest.skipIf(os.name == "nt", "the agent needs Linux or macOS")
class Connect(unittest.TestCase):
    KEY = "lgk_" + "d" * 40

    def setUp(self):
        self.beats, self.tracked, self.live = [], [], []
        T = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                if self.headers.get("Authorization") != f"Bearer {T.KEY}":
                    return self.reply(401, {"error": {"code": "invalid_api_key", "message": "That API key is not valid."}})
                if "live=1" in self.path:
                    return self.reply(200, {"nodes": [{"node": n, "age_s": 3} for n in T.live]})
                self.reply(200, {"runs": []})
            def do_POST(self):
                b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if "live=1" in self.path:
                    T.beats.append(b)
                    if b["node"] not in T.live:
                        T.live.append(b["node"])
                    return self.reply(202, {"ok": True, "skipped": False})
                T.tracked.append(b)
                self.reply(201, {"ok": True})
        self.srv = _Server(H)
        self.out = io.StringIO()
        self.patches = [mock.patch.dict(os.environ, SIM), mock.patch.object(cn, "provider", lambda: "kaggle"),
                        mock.patch.object(cn, "can_install_service", lambda: False), contextlib.redirect_stdout(self.out)]
        for p in self.patches:
            p.__enter__()

    def tearDown(self):
        cn.stop_background()
        for p in reversed(self.patches):
            p.__exit__(None, None, None)
        self.srv.close()
        from joules import share
        for f in (share.key_file(), cn.log_file()):
            with contextlib.suppress(FileNotFoundError):
                os.remove(f)

    def connect(self, **kw):
        return cn.connect(api_url=self.srv.url, ask=lambda prompt: self.KEY, live_every=2, **{"board": False, **kw})

    def test_names_say_where_it_runs_and_what_it_has_never_the_host_name(self):
        for name, short in (("Tesla T4", "t4"), ("NVIDIA A100-SXM4-80GB", "a100"), ("NVIDIA GeForce RTX 4090", "rtx-4090"),
                            ("NVIDIA GeForce RTX 4070 Ti", "rtx-4070-ti"), ("AMD Instinct MI300X", "mi300x"), ("NVIDIA H100 80GB HBM3", "h100"),
                            ("Tesla P100-PCIE-16GB", "p100"), ("NVIDIA RTX A6000", "rtx-a6000"), ("AMD Radeon RX 7900 XTX", "rx-7900-xtx"),
                            ("Intel(R) Arc(TM) A770 Graphics", "arc-a770"), ("NVIDIA L4", "l4")):
            self.assertEqual(cn.short_gpu(name), short, name)
        self.assertEqual(cn.default_name("kaggle", ["Tesla T4", "Tesla T4"]), "kaggle-2x-t4")
        self.assertEqual(cn.default_name("colab", ["Tesla T4"]), "colab-t4")
        self.assertEqual(cn.default_name("", ["NVIDIA A100-SXM4-80GB"] * 8), "8x-a100")
        self.assertEqual(cn.clean_name("my lab / box #1"), "my-lab-box-1")
        self.assertNotIn(socket.gethostname(), cn.default_name("", []))

    def test_provider_from_the_environment_or_the_vm_firmware(self):
        with tempfile.TemporaryDirectory() as t:
            clear = {e: "" for _, envs in cn.PROVIDERS for e in envs}
            with mock.patch.dict(os.environ, {**clear, "JOULES_SYSFS": t}):
                self.assertIsNone(REAL_PROVIDER())
                with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "abc"}):
                    self.assertEqual(REAL_PROVIDER(), "runpod")
                with mock.patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
                    self.assertEqual(REAL_PROVIDER(), "kaggle")
                write(Path(t) / "class/dmi/id/sys_vendor", "Amazon EC2\n")
                self.assertEqual(REAL_PROVIDER(), "aws")
                write(Path(t) / "class/dmi/id/sys_vendor", "Microsoft Corporation\n")
                self.assertIsNone(REAL_PROVIDER())  # Hyper-V on a desk is not Azure
                write(Path(t) / "class/dmi/id/chassis_asset_tag", cn.AZURE_TAG + "\n")
                self.assertEqual(REAL_PROVIDER(), "azure")

    def test_connect_asks_for_the_key_starts_the_agent_and_reports_to_live_under_its_name(self):
        from joules import share
        c = self.connect()
        # connect waits live_every + 10 s for the first reading; a slow CI machine can take longer to start the agent,
        # and connect then says so ("waiting") rather than claiming it is on, so the reading may still be on its way
        self.assertEqual((c.name, c.project, c.mode), ("kaggle-h100", "kaggle-h100", "background"))
        self.assertIn(c.live, ("on", "waiting"))
        self.assertTrue(c.running)
        self.assertEqual(share.saved_key(), self.KEY)
        end = time.monotonic() + 30
        while not self.beats and time.monotonic() < end:
            time.sleep(0.25)
        self.assertTrue(self.beats and all(b["node"] == "kaggle-h100" for b in self.beats))
        self.assertNotIn(self.KEY, cn._cmdline(c.pid))  # the key is never on a command line
        self.assertIn(" agent ", cn._cmdline(c.pid))
        out = self.out.getvalue()
        self.assertNotIn(self.KEY, out)
        self.assertIn("kaggle-h100 is connected", out)
        self.assertIn(f"Tracking  {self.srv.url}/#/track?project=kaggle-h100", out)
        self.assertIn(f"Live      {self.srv.url}/#/live  " + ("this machine's power" if c.live == "on" else "no reading has arrived yet"), out)
        # again: the agent from before is replaced, and its own name on Live is not a clash
        first = c.pid
        c2 = self.connect()
        self.assertEqual(c2.name, "kaggle-h100")
        self.assertNotEqual(c2.pid, first)
        self.assertFalse(cn._is_our_agent(first))
        self.assertIn("Replaced the agent started earlier for kaggle-h100", self.out.getvalue())
        c2.stop()
        self.assertFalse(c2.running)
        self.assertIsNone(cn.running())

    def test_a_name_another_machine_has_on_live_gets_a_number(self):
        self.live[:] = ["kaggle-h100", "kaggle-h100-2"]
        c = self.connect(agent=False)
        self.assertEqual(c.name, "kaggle-h100-3")
        self.assertIn("kaggle-h100 is already on Live from another machine (last reading 3 s ago), so this one is kaggle-h100-3",
                      self.out.getvalue())
        self.assertIsNone(c.mode)

    def test_board_runs_the_gpu_test_and_links_its_row(self):
        calls = []
        with mock.patch.object(cn, "gpu_names", lambda: ["Tesla T4", "Tesla T4"]), \
                mock.patch.object(cn, "run_board", lambda gpu, base: calls.append((gpu, base, os.environ.get("LANDAUER_API_KEY"))) or 0):
            c = self.connect(agent=False, board=True)
        self.assertEqual(c.name, "kaggle-2x-t4")
        self.assertEqual(calls, [(0, self.srv.url, self.KEY)])  # the key reaches the test through the environment
        self.assertEqual(os.environ.get("LANDAUER_API_KEY"), None)
        self.assertEqual(c.links["leaderboard"], f"{self.srv.url}/leaderboard?gpu=Tesla%20T4")
        with mock.patch.object(cn, "gpu_names", lambda: []), mock.patch.object(cn.platform, "system", lambda: "Linux"):
            c = self.connect(agent=False, board=True)
        self.assertEqual((c.name, c.board), ("kaggle-cpu", "no GPU"))

    def test_wrong_keys_are_refused_and_nothing_starts(self):
        with self.assertRaises(cn.ConnectError) as e:
            cn.connect(api_url=self.srv.url, ask=lambda p: "lgk_…", board=False)
        self.assertIn("no valid key given", str(e.exception))
        with self.assertRaises(cn.ConnectError) as e:
            cn.connect(api_url=self.srv.url, ask=lambda p: "lgk_" + "x" * 40, board=False)
        self.assertIn("the key was not accepted: That API key is not valid.", str(e.exception))
        self.assertIsNone(cn.running())

    def test_a_notebook_run_with_no_input_box_is_refused_with_a_way_out(self):
        def no_box(prompt):  # what IPython raises when a run has no input box
            raise NotImplementedError("raw_input was called, but this frontend does not support input requests.")
        with self.assertRaises(cn.ConnectError) as e:
            cn.connect(api_url=self.srv.url, ask=no_box, board=False)
        self.assertIn("this notebook run cannot ask for a key. Pass key=", str(e.exception))
        with mock.patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}), self.assertRaises(cn.ConnectError) as e:
            cn.get_key(base=self.srv.url)  # Kaggle's Save Version: refused before any box is shown
        self.assertIn("no API key", str(e.exception))
        self.assertIsNone(cn.running())

    def test_a_notebook_shell_cell_without_a_key_points_to_a_python_cell(self):
        code, _, err = run_cli(["connect"], {"JPY_PARENT_PID": "1"})
        self.assertEqual(code, 2)
        self.assertIn("import joules; joules.connect()", err)

    def test_as_root_with_systemd_it_installs_the_service_and_keeps_your_settings(self):
        repo_unit = Path(__file__).resolve().parents[2] / "integrations/cluster-agent/systemd/joules-agent.service"
        if repo_unit.exists():
            self.assertEqual(cn.UNIT, repo_unit.read_text())
        got, want = cn.unit_text().splitlines(), cn.UNIT.splitlines()  # only which joules it starts may differ
        self.assertEqual([(g, w) for g, w in zip(got, want) if g != w and not (g.startswith("ExecStart=") and " agent " in g)], [])
        self.assertEqual(len(got), len(want))
        with tempfile.TemporaryDirectory() as t:
            env, unit = os.path.join(t, "joules-agent"), os.path.join(t, "joules-agent.service")
            write(Path(env), "# mine\nLANDAUER_API_KEY=lgk_old\nJOULES_ARGS=--track old --gpus 0,1\nPRICE_PER_KWH=0.31\nJOULES_HOST=0.0.0.0\n")
            ran = []
            ok = lambda args, **k: ran.append(args) or subprocess.CompletedProcess(args, 0, "", "")
            with mock.patch.object(cn, "ENV_FILE", env), mock.patch.object(cn, "UNIT_FILE", unit):
                cn.install_service(self.KEY, "lab-a100-1", "lab", run=ok)
            text = Path(env).read_text()
            self.assertEqual(text, f"# mine\nLANDAUER_API_KEY={self.KEY}\nJOULES_ARGS=--gpus 0,1 --track lab --name lab-a100-1\n"
                                   "PRICE_PER_KWH=0.31\nJOULES_HOST=0.0.0.0\nGRID=us\nPUE=1.0\n")
            self.assertEqual(os.stat(env).st_mode & 0o777, 0o600)
            self.assertEqual(Path(unit).read_text(), cn.unit_text())
            self.assertEqual(ran, [["systemctl", "daemon-reload"], ["systemctl", "enable", "joules-agent"], ["systemctl", "restart", "joules-agent"]])


class Telemetry(unittest.TestCase):
    """Usage counts: asked once in a terminal, nothing sent without a yes, never in CI or with DO_NOT_TRACK."""
    from joules import telemetry as t

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for v in (*self.t.CI_VARS, "DO_NOT_TRACK", "JOULES_TELEMETRY", "JOULES_NO_PROMPT"):
            os.environ.pop(v, None)
        with contextlib.suppress(OSError):
            os.remove(self.t.state_file())
        self.sent = []

    def tearDown(self):
        with contextlib.suppress(OSError):
            os.remove(self.t.state_file())
        self.env.stop()

    def ask(self, answer):
        with mock.patch.object(self.t, "_interactive", return_value=True), contextlib.redirect_stderr(io.StringIO()):
            return self.t.maybe_ask(ask=lambda q: answer, sender=self.sent.append)

    def test_nothing_is_sent_before_a_yes(self):
        self.assertIsNone(self.t.record(sender=self.sent.append))
        self.assertEqual(self.sent, [])

    def test_no_and_enter_are_no_and_it_asks_only_once(self):
        self.assertFalse(self.ask(""))
        self.assertIsNone(self.ask("y"))  # already answered
        self.assertIsNone(self.t.record(sender=self.sent.append))
        self.assertEqual(self.sent, [])

    def test_yes_sends_only_the_counts_once_a_day(self):
        self.assertTrue(self.ask("yes"))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(set(self.sent[0]), {"id", "v", "os", "gpu", "runs"})
        self.assertRegex(self.sent[0]["id"], r"^[0-9a-f]{32}$")
        self.assertIn(self.sent[0]["os"], ("linux", "darwin", "windows", "other"))
        self.assertIn(self.sent[0]["gpu"], ("nvidia", "amd", "intel", "apple", "none", "other"))
        today = self.t.load()["last_sent"]
        self.assertIsNone(self.t.record(today=today, sender=self.sent.append))  # counted, not sent again today
        self.assertIsNone(self.t.record(today=today, sender=self.sent.append))
        body = self.t.record(today="2999-01-01", sender=self.sent.append)
        self.assertEqual(body["runs"], 3)
        self.assertEqual(body["id"], self.sent[0]["id"])

    def test_never_asks_or_sends_in_ci_or_with_do_not_track(self):
        for var, val in (("CI", "true"), ("GITHUB_ACTIONS", "true"), ("DO_NOT_TRACK", "1"), ("JOULES_TELEMETRY", "0")):
            with self.subTest(var=var), mock.patch.dict(os.environ, {var: val}):
                with contextlib.suppress(OSError):
                    os.remove(self.t.state_file())
                self.assertIsNone(self.ask("y"))
                self.t.set_enabled(True)
                self.assertIsNone(self.t.record(today="2999-01-02", sender=self.sent.append))
        self.assertEqual(self.sent, [])

    def test_never_asks_without_a_terminal(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(self.t.maybe_ask(ask=lambda q: "y", sender=self.sent.append))
        self.assertNotIn("enabled", self.t.load())

    def test_never_asks_in_a_notebooks_shell_cell(self):
        # A ! cell runs in a pseudo-terminal (isatty() is true) that the notebook cannot type into: asking would hang it.
        tty = mock.MagicMock(); tty.isatty.return_value = True
        for var, val in (("KAGGLE_KERNEL_RUN_TYPE", "Interactive"), ("COLAB_RELEASE_TAG", "release-colab"), ("JPY_PARENT_PID", "4242")):
            with self.subTest(var=var), mock.patch.dict(os.environ, {var: val}), \
                    mock.patch("sys.stdin", tty), mock.patch("sys.stdout", tty), mock.patch("sys.stderr", tty):
                self.assertFalse(self.t._interactive())
                self.assertIsNone(self.t.maybe_ask(ask=lambda q: self.fail("asked in a ! cell"), sender=self.sent.append))
        with mock.patch("sys.stdin", tty), mock.patch("sys.stdout", tty), mock.patch("sys.stderr", tty):
            for var in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "COLAB_RELEASE_TAG", "COLAB_GPU", "JPY_PARENT_PID"):
                os.environ.pop(var, None)
            self.assertTrue(self.t._interactive())  # a real terminal still asks
        self.assertNotIn("enabled", self.t.load())

    def test_command_on_off_and_status(self):
        code, out, _ = run_cli(["telemetry"])
        self.assertEqual(code, 0)
        self.assertIn("not asked yet", out)
        code, out, _ = run_cli(["telemetry", "on"])
        self.assertIn("Usage counts: on", out)
        self.assertIn('"runs"', out)
        code, out, _ = run_cli(["telemetry", "off"])
        self.assertIn("you said no", out)
        self.assertFalse(self.t.enabled())

    def test_a_command_never_asks_in_tests_or_pipes(self):
        code, _, err = run_cli(["devices"], {"JOULES_SIMULATE": SIM["JOULES_SIMULATE"]})
        self.assertNotIn("usage counts", err.lower())
        self.assertNotIn("enabled", self.t.load())


from joules import serving  # noqa: E402

VLLM_PAGE = """# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{engine="0",model_name="m"}} {gen}
vllm:prompt_tokens_total{{engine="0",model_name="m"}} {prompt}
vllm:num_requests_running{{engine="0",model_name="m"}} {running}
vllm:num_requests_waiting{{engine="0",model_name="m"}} 3.0
vllm:time_to_first_token_seconds_bucket{{engine="0",le="0.5",model_name="m"}} 7.0
vllm:time_to_first_token_seconds_sum{{engine="0",model_name="m"}} {ttft_sum}
vllm:time_to_first_token_seconds_count{{engine="0",model_name="m"}} {ttft_n}
vllm:generation_tokens_created{{engine="0",model_name="m"}} 1.76e+09
"""


def vllm_page(gen, prompt=0, running=0, ttft_sum=0.0, ttft_n=0):
    return VLLM_PAGE.format(gen=gen, prompt=prompt, running=running, ttft_sum=ttft_sum, ttft_n=ttft_n)


class ServeMetricsTests(unittest.TestCase):
    def test_reads_vllm_and_sglang_counters_and_nothing_else(self):
        p = serving.parse(vllm_page(1000, 400, 12, 3.0, 6))
        self.assertEqual(p, {"engine": "vllm", "gen": 1000, "prompt": 400, "running": 12, "waiting": 3,
                             "ttft_sum": 3.0, "ttft_count": 6})
        # two models on one server: their counters add up
        two = vllm_page(1000) + 'vllm:generation_tokens_total{engine="1",model_name="n"} 500\n'
        self.assertEqual(serving.parse(two)["gen"], 1500)
        sg = "sglang:generation_tokens_total{model_name=\"q\"} 80\nsglang:num_running_reqs{model_name=\"q\"} 4\nsglang:num_queue_reqs 1\n"
        self.assertEqual(serving.parse(sg)["engine"], "sglang")
        self.assertEqual((serving.parse(sg)["running"], serving.parse(sg)["waiting"]), (4, 1))
        self.assertIsNone(serving.parse("process_cpu_seconds_total 12\n# just a Prometheus page\n"))
        self.assertIsNone(serving.parse("<html>not metrics</html>"))

    def test_tokens_per_second_over_the_last_30_seconds_and_first_token_time(self):
        pages = iter([vllm_page(0, 0, 8), vllm_page(5000, 2000, 8, 2.0, 4), vllm_page(12000, 4000, 16, 5.0, 10)])
        s = serving.ServeMetrics("http://127.0.0.1:1/metrics", fetch=lambda u: next(pages))
        self.assertIsNone(s.read(100.0))                         # one reading: no rate yet
        r = s.read(110.0)
        self.assertEqual(r, {"engine": "vllm", "tok_s": 500.0, "prompt_tok_s": 200.0, "running": 8, "waiting": 3, "ttft_s": 0.5})
        r = s.read(120.0)
        self.assertEqual((r["tok_s"], r["running"], r["ttft_s"]), (600.0, 16, 0.5))   # 12000 tokens in 20 s; 5 s over 10 first tokens

    def test_a_restarted_server_starts_the_rate_again(self):
        pages = iter([vllm_page(90000), vllm_page(100000), vllm_page(50), vllm_page(1050)])
        s = serving.ServeMetrics("http://127.0.0.1:1/metrics", fetch=lambda u: next(pages))
        s.read(0.0); self.assertEqual(s.read(10.0)["tok_s"], 1000.0)
        self.assertIsNone(s.read(20.0))                          # counters went down: a new server, no rate from the old one
        self.assertEqual(s.read(30.0)["tok_s"], 100.0)

    def test_auto_finds_the_server_and_looks_again_a_minute_later(self):
        calls, up = [], {"on": False}
        def fetch(u):
            calls.append(u)
            if not up["on"] or not u.endswith(":30000/metrics"):
                raise urllib.error.URLError("refused")
            return "sglang:generation_tokens_total 10\n"
        s = serving.ServeMetrics("auto", fetch=fetch)
        self.assertIsNone(s.read(0.0)); self.assertEqual(len(calls), 2)
        self.assertIsNone(s.read(5.0)); self.assertEqual(len(calls), 2)   # not again within the minute
        up["on"] = True
        s.read(61.0); self.assertEqual(s.url, "http://127.0.0.1:30000/metrics")
        up["on"] = False
        self.assertIsNone(s.read(66.0)); self.assertIsNone(s.url)          # gone: back to looking

    def test_live_beat_carries_what_the_server_serves(self):
        a = ag.Agent([FakeMeter("GPU 0 H200", "gpu", 0)], host="n1")
        a.acc.step(0, 0, {}, [], {})
        self.assertNotIn("serve", a.live_beat())
        pages = iter([vllm_page(0), vllm_page(64000, 0, 256)])
        a.serve = serving.ServeMetrics("http://127.0.0.1:1/metrics", fetch=lambda u: next(pages))
        a.serve.read(0.0)
        with mock.patch.object(serving.time, "monotonic", return_value=10.0):
            b = a.live_beat()
        self.assertEqual(b["serve"], {"engine": "vllm", "tok_s": 6400.0, "prompt_tok_s": 0.0, "running": 256, "waiting": 3})


@unittest.skipIf(os.name == "nt", "joules agent is Linux only")
class ServeMetricsCli(unittest.TestCase):
    def test_heartbeats_carry_tokens_and_an_older_server_gets_power_only(self):
        n = {"gen": 0}
        class V(BaseHTTPRequestHandler):  # a vLLM server making 1000 tokens every reading
            def log_message(self, *a): pass
            def do_GET(self):
                n["gen"] += 1000
                data = vllm_page(n["gen"], 0, 4).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        for old in (False, True):
            got = []
            class H(BaseHTTPRequestHandler):
                def log_message(self, *a): pass
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    got.append(body)
                    if old and "serve" in body:  # a WattOps server from before tokens on Live
                        code, data = 400, b'{"error": {"code": "invalid_beat", "message": "Some fields are not valid.", "fields": {"serve": "Unknown field."}}}'
                    else:
                        code, data = 202, b'{"ok": true, "skipped": false, "page": "/#/live"}'
                    self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); self.wfile.write(data)
            vs, srv = _Server(V), _Server(H)
            with tempfile.TemporaryDirectory() as t:
                cluster(Path(t), ticks=(400, 100, 100, 100))
                try:
                    code, _, err = run_cli(["agent", "--port", "0", "--interval", "0.2", "--duration", "8.6", "--live-every", "2",
                                            "--track", "cluster", "--key", "lg_test", "--api-url", srv.url,
                                            "--serve-metrics", vs.url + "/metrics"], {**SIM, "JOULES_PROC": t})
                finally:
                    srv.close(); vs.close()
            self.assertEqual(code, 0, err)
            beats = [b for b in got if "node" in b]
            served = [b for b in beats if "serve" in b]
            self.assertTrue(served, beats)
            self.assertEqual((served[0]["serve"]["engine"], served[0]["serve"]["running"]), ("vllm", 4))
            self.assertGreater(served[0]["serve"]["tok_s"], 0)
            if old:
                i = beats.index(served[0])
                self.assertEqual(len(served), 1, "after one refusal, power only")
                self.assertTrue(beats[i + 1:], beats)
                self.assertIn("does not take tokens per second yet", err)
            else:
                self.assertGreaterEqual(len(served), 2)
        code, _, err = run_cli(["agent", "--serve-metrics", "localhost:8000", "--track", "cluster", "--key", "lg_test"], {**SIM, "JOULES_PROC": tempfile.mkdtemp()})
        self.assertEqual(code, 2); self.assertIn("--serve-metrics", err)
