"""AMD APU regression: on a built-in AMD GPU amdgpu reports the whole chip's power, which RAPL's CPU package figure
also covers. Both are still measured (no number changes), but the result must say so, as the Windows (ADLX) path does.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from joules import meters


def write(p: Path, text) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text))


def fake_sysfs(root: Path, device_id: str, rapl=True) -> None:
    dev = root / "class/drm/card0/device"
    write(dev / "vendor", "0x1002"); write(dev / "device", device_id)
    write(dev / "hwmon/hwmon0/name", "amdgpu"); write(dev / "hwmon/hwmon0/power1_average", 25_000_000)  # 25 W
    if rapl:
        pc = root / "class/powercap"
        write(pc / "intel-rapl:0/name", "package-0"); write(pc / "intel-rapl:0/energy_uj", 1_000_000)
        write(pc / "intel-rapl:0/max_energy_range_uj", 10_000_000)


@unittest.skipIf(os.name == "nt", "amdgpu and RAPL are read from Linux sysfs (whose names contain ':')")
class ApuNote(unittest.TestCase):
    def discover(self, root, **kw):
        env = {"JOULES_SYSFS": str(root)}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(meters, "discover_nvml", return_value=([], None)), \
                mock.patch.object(meters, "discover_adlx", return_value=([], None)), \
                mock.patch.object(meters.platform, "system", return_value="Linux"):
            for k in ("JOULES_SIMULATE", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                os.environ.pop(k, None)
            return meters.discover(**kw)

    def test_apu_with_rapl_gets_a_note_and_keeps_both_meters(self):
        with tempfile.TemporaryDirectory() as t:
            fake_sysfs(Path(t), "0x15bf")  # Phoenix: Ryzen 7040 built-in Radeon 780M
            d = self.discover(Path(t))
            self.assertEqual([m.method for m in d.meters], ["amdgpu power sampling", "rapl counter"])
            self.assertTrue(any("built-in GPU" in n and "RAPL" in n for n in d.notes), d.notes)
            gpu = d.meters[0]
            gpu.start(0.0); gpu.poll(2.0)
            self.assertAlmostEqual(gpu.energy_j, 50.0)  # its number is untouched

    def test_no_note_for_a_discrete_card_or_without_rapl(self):
        with tempfile.TemporaryDirectory() as t:
            fake_sysfs(Path(t), "0x744c")  # Radeon RX 7900 XTX
            self.assertFalse(any("built-in GPU" in n for n in self.discover(Path(t)).notes))
        with tempfile.TemporaryDirectory() as t:
            fake_sysfs(Path(t), "0x15bf")
            self.assertFalse(any("built-in GPU" in n for n in self.discover(Path(t), cpu=False).notes))


if __name__ == "__main__":
    unittest.main()
