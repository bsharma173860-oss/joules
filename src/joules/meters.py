"""Energy meters for the hardware a training or inference job runs on.

Every meter reports joules used since start(). Two kinds:
  * counters  - the hardware keeps a running energy total (NVML on Volta and newer, Intel/AMD
                RAPL, amdgpu and Intel GPU energy1_input). Most accurate; we only handle wraparound.
  * sampled   - the hardware reports power (older NVIDIA, amdgpu power1_average, Mac powermetrics).
                We integrate it over time.

Meters measure whole devices, so other work on the same GPU or CPU is included. Use
--subtract-idle to remove the baseline the machine draws when your job is not running.
"""
from __future__ import annotations

import ctypes
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field


def sysfs() -> str:
    return os.environ.get("JOULES_SYSFS", "/sys")


class Meter:
    kind = "?"      # gpu | cpu | dram | soc
    method = "?"    # shown to the user, e.g. "nvml counter"

    def __init__(self, name: str):
        self.name = name
        self.energy_j = 0.0
        self.peak_w = 0.0
        self._t0 = None
        self._t_last = None

    def start(self, now: float) -> None:
        self.energy_j = 0.0
        self.peak_w = 0.0
        self._t0 = self._t_last = now

    def poll(self, now: float) -> None:
        raise NotImplementedError

    def stop(self, now: float) -> None:
        self.poll(now)

    def close(self) -> None:
        pass

    @property
    def duration_s(self) -> float:
        return (self._t_last or 0) - (self._t0 or 0)

    @property
    def avg_w(self) -> float:
        d = self.duration_s
        return self.energy_j / d if d > 0 else 0.0


class CounterMeter(Meter):
    """Hardware energy counter in joules; wrap is the counter's range (None if it never wraps)."""
    wrap_j: float | None = None

    def read_j(self) -> float:
        raise NotImplementedError

    def start(self, now: float) -> None:
        super().start(now)
        self._last = self.read_j()

    def poll(self, now: float) -> None:
        v = self.read_j()
        d = v - self._last
        if d < 0:
            d += self.wrap_j if self.wrap_j else 0.0
        dt = now - self._t_last
        if d >= 0:  # the counter's change is energy used whatever the clock says (two readers may poll out of order)
            if dt > 0:
                self.peak_w = max(self.peak_w, d / dt)
            self.energy_j += d
        self._last = v
        self._t_last = max(self._t_last, now)


class PowerMeter(Meter):
    """Instantaneous power in watts, integrated over time."""

    def read_w(self) -> float:
        raise NotImplementedError

    def start(self, now: float) -> None:
        super().start(now)
        self._w = self.read_w()
        self.peak_w = self._w

    def poll(self, now: float) -> None:
        w = self.read_w()
        dt = now - self._t_last
        if dt > 0:
            self.energy_j += (self._w + w) / 2 * dt
        self.peak_w = max(self.peak_w, w)
        self._w = w
        self._t_last = max(self._t_last, now)


# ---------------------------------------------------------------- Intel / AMD CPUs (RAPL)
class RaplMeter(CounterMeter):
    method = "rapl counter"

    def __init__(self, path: str, label: str, kind: str):
        super().__init__(label)
        self.kind = kind
        self.path = path
        try:
            self.wrap_j = int(_read(os.path.join(path, "max_energy_range_uj"))) / 1e6
        except (OSError, ValueError):
            self.wrap_j = None

    def read_j(self) -> float:
        return int(_read(os.path.join(self.path, "energy_uj"))) / 1e6


def discover_rapl(notes: list[str]) -> list[Meter]:
    meters: list[Meter] = []
    zones = sorted(glob.glob(os.path.join(sysfs(), "class/powercap/intel-rapl:*")))
    for z in zones:
        base = os.path.basename(z)
        try:
            name = _read(os.path.join(z, "name")).strip()
        except OSError:
            continue
        top = base.count(":") == 1
        if top and name.startswith("package"):
            label, kind = f"CPU {name.replace('package-', 'package ')}", "cpu"
        elif not top and name == "dram":
            label, kind = f"DRAM (package {base.split(':')[1]})", "dram"
        else:
            continue  # core/uncore are inside the package total; psys overlaps it
        try:
            int(_read(os.path.join(z, "energy_uj")))
        except PermissionError:
            notes.append(f"{label}: no permission to read {z}/energy_uj. Run with sudo, or: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj")
            continue
        except (OSError, ValueError):
            continue
        meters.append(RaplMeter(z, label, kind))
    return meters


# ---------------------------------------------------------------- AMD GPUs (amdgpu hwmon)
class HwmonEnergyMeter(CounterMeter):
    method = "amdgpu energy counter"
    kind = "gpu"

    def __init__(self, path: str, label: str):
        super().__init__(label)
        self.path = path

    def read_j(self) -> float:
        return int(_read(os.path.join(self.path, "energy1_input"))) / 1e6


class HwmonPowerMeter(PowerMeter):
    method = "amdgpu power sampling"
    kind = "gpu"

    def __init__(self, file: str, label: str):
        super().__init__(label)
        self.file = file

    def read_w(self) -> float:
        return int(_read(self.file)) / 1e6


def _gpu_hwmons(drivers: tuple[str, ...]) -> list[str]:
    """hwmon folders of the GPUs run by these drivers, in card order (card2 before card10)."""
    found = []
    for hw in glob.glob(os.path.join(sysfs(), "class/drm/card*/device/hwmon/hwmon*")):
        m = re.search(r"/card(\d+)/device/", hw)
        try:
            if m and _read(os.path.join(hw, "name")).strip() in drivers:
                found.append((int(m.group(1)), hw))
        except OSError:
            continue
    return [hw for _, hw in sorted(found)]


# AMD Instinct PCI device IDs (pci.ids, vendor 1002): virtual functions in clouds often have no product_name file.
_AMD_PCI_NAMES = {0x74a0: "AMD Instinct MI300A", 0x74a1: "AMD Instinct MI300X", 0x74a2: "AMD Instinct MI308X",
                  0x74a5: "AMD Instinct MI325X", 0x74a9: "AMD Instinct MI300X HF", 0x74b5: "AMD Instinct MI300X VF",
                  0x74b9: "AMD Instinct MI325X VF", 0x74bd: "AMD Instinct MI300X HF", 0x75a0: "AMD Instinct MI350X",
                  0x75a3: "AMD Instinct MI355X", 0x75a8: "AMD Instinct MI350P"}
PCI_IDS = ("/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids", "/usr/share/pci.ids")


def _pci_ids_name(vendor: int, device: int) -> str | None:
    """The device's name from the system's pci.ids database, e.g. "Instinct MI300X VF" from
    "Aqua Vanjaram [Instinct MI300X VF]"; None if there is no database or no entry."""
    for path in PCI_IDS:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                in_vendor = False
                for line in f:
                    if not line.startswith("\t"):
                        in_vendor = line[:4].lower() == f"{vendor:04x}"
                    elif in_vendor and not line.startswith("\t\t") and line[1:5].lower() == f"{device:04x}":
                        name = line[5:].strip()
                        m = re.search(r"\[([^\]]+)\]\s*$", name)
                        return m.group(1) if m else name
        except OSError:
            continue
    return None


def _product_name(hw: str, fallback: str) -> str:
    try:  # amdgpu has it for most Instinct and many Radeon cards, e.g. "AMD Instinct MI300X"
        name = _read(os.path.join(hw, "..", "..", "product_name")).strip()
    except OSError:
        name = ""
    if not name:  # no product_name (common for cloud virtual functions): name the card by its PCI IDs
        try:
            dev = os.path.join(hw, "..", "..")
            vendor, device = int(_read(os.path.join(dev, "vendor")), 16), int(_read(os.path.join(dev, "device")), 16)
        except (OSError, ValueError):
            vendor = device = None
        if vendor == 0x1002:
            name = _AMD_PCI_NAMES.get(device) or ""
            if not name:
                found = _pci_ids_name(vendor, device)
                name = f"AMD {found}" if found and not found.upper().startswith("AMD") else (found or "")
    return re.sub(r"\s+", " ", name)[:80] or fallback


def _pci_of(hw: str) -> str | None:
    """PCI address of the card a hwmon folder belongs to, e.g. 0000:03:00.0 (matches drm-pdev in /proc/<pid>/fdinfo)."""
    name = os.path.basename(os.path.realpath(os.path.join(hw, "device")))
    return name if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", name) else None


def _env_indices(*names: str) -> list[int] | None:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and re.fullmatch(r"\d+(,\d+)*", v):
            return [int(x) for x in v.split(",")]
    return None


def discover_amdgpu(notes: list[str], gpus: list[int] | None = None) -> list[Meter]:
    meters: list[Meter] = []
    wanted = gpus if gpus is not None else _env_indices("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    for i, hw in enumerate(_gpu_hwmons(("amdgpu",))):
        if wanted is not None and i not in wanted:
            continue
        label = f"GPU {i} {_product_name(hw, 'AMD GPU')}"
        if os.path.exists(os.path.join(hw, "energy1_input")):
            meters.append(HwmonEnergyMeter(hw, label))
            meters[-1].pci = _pci_of(hw)
            continue
        for f in ("power1_average", "power1_input"):
            p = os.path.join(hw, f)
            if os.path.exists(p):
                meters.append(HwmonPowerMeter(p, label))
                meters[-1].pci = _pci_of(hw)
                break
        else:
            notes.append(f"{label}: the amdgpu driver reports no energy or power here.")
    return meters


# ---------------------------------------------------------------- Intel GPUs (Arc, Flex, Max: i915 / xe hwmon)
class IntelGpuEnergyMeter(HwmonEnergyMeter):
    method = "intel gpu energy counter"


def discover_intelgpu(notes: list[str]) -> list[Meter]:
    meters: list[Meter] = []
    for i, hw in enumerate(_gpu_hwmons(("i915", "xe"))):  # only discrete cards have an energy counter
        label = f"GPU {i} {_product_name(hw, 'Intel GPU')}"
        f = os.path.join(hw, "energy1_input")
        if not os.path.exists(f):
            continue
        try:
            int(_read(f))
        except PermissionError:
            notes.append(f"{label}: no permission to read {f}. Run with sudo.")
            continue
        except (OSError, ValueError):
            continue
        meters.append(IntelGpuEnergyMeter(hw, label))
        meters[-1].pci = _pci_of(hw)
    return meters


# ---------------------------------------------------------------- AMD GPUs on Windows (ADLX via ctypes)
# amdadlx64.dll ships with every Radeon/Ryzen graphics driver. ADLX objects are C structs whose first field points
# to a table of functions (a "vtable"); the slot numbers below are the order of the C vtables in AMD's ADLX SDK
# headers (SDK/Include/ISystem.h and IPerformanceMonitoring.h, ADLX 2.0). On 64-bit Windows __stdcall and
# __cdecl are the same convention, so CFUNCTYPE fits both.
ADLX_DLL = "amdadlx64.dll"
ADLX_VERSION = (2 << 48) | (0 << 32) | (0 << 16) | 125   # ADLX_MAKE_FULL_VER(2, 0, 0, 125)
_ADLX_OK, _ADLX_ALREADY = 0, (1, 2)                       # ADLX_OK; ADLX_ALREADY_ENABLED, ADLX_ALREADY_INITIALIZED
# IADLXSystem (no reference counting)
_SYS_GET_GPUS, _SYS_GET_PERF = 1, 9
# every other interface starts with Acquire (0), Release (1), QueryInterface (2)
_RELEASE = 1
_LIST_SIZE, _LIST_AT_GPU = 3, 11                          # IADLXGPUList: Size, At_GPUList
_GPU_TYPE, _GPU_NAME = 5, 7                               # IADLXGPU: Type, Name
_PERF_CUR_GPU, _PERF_SUPPORTED_GPU = 18, 21               # IADLXPerformanceMonitoringServices
_SUP_GPU_POWER, _SUP_BOARD_POWER = 8, 9                   # IADLXGPUMetricsSupport: IsSupportedGPUPower, ...TotalBoardPower
_MET_GPU_POWER, _MET_BOARD_POWER = 9, 10                  # IADLXGPUMetrics: GPUPower, GPUTotalBoardPower
_ADLX_INTEGRATED = 1                                      # ADLX_GPU_TYPE: GPUTYPE_INTEGRATED


def _vcall(obj, slot: int, restype, *argtypes):
    """The function in slot `slot` of an ADLX object's vtable, ready to call with the object as first argument."""
    table = ctypes.cast(ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[slot])
    return lambda *a: fn(obj, *a)


def _adlx_ok(r: int) -> bool:
    return r == _ADLX_OK or r in _ADLX_ALREADY


def _load_adlx():
    """The ADLX library, or None. Tests replace this with a fake driver."""
    if platform.system() != "Windows" or ctypes.sizeof(ctypes.c_void_p) != 8:
        return None
    try:
        return ctypes.CDLL(ADLX_DLL)
    except OSError:
        return None


class Adlx:
    """One ADLX session: the system object, its performance-monitoring service and the AMD GPUs."""

    def __init__(self, lib):
        self.lib = lib
        self.sys = ctypes.c_void_p()
        init = lib.ADLXInitialize
        init.restype, init.argtypes = ctypes.c_int, [ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        r = init(ctypes.c_uint64(ADLX_VERSION), ctypes.byref(self.sys))
        if not _adlx_ok(r) or not self.sys.value:
            raise OSError(f"ADLXInitialize returned {r}")
        self.perf = ctypes.c_void_p()
        if not _adlx_ok(_vcall(self.sys, _SYS_GET_PERF, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(self.perf))):
            self.perf = None
        gl = ctypes.c_void_p()
        self.gpus: list[ctypes.c_void_p] = []
        if _adlx_ok(_vcall(self.sys, _SYS_GET_GPUS, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(gl))) and gl.value:
            n = _vcall(gl, _LIST_SIZE, ctypes.c_uint)()
            for i in range(n):
                g = ctypes.c_void_p()
                if _adlx_ok(_vcall(gl, _LIST_AT_GPU, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(i, ctypes.byref(g))) and g.value:
                    self.gpus.append(g)
            _vcall(gl, _RELEASE, ctypes.c_long)()

    def name(self, gpu) -> str:
        s = ctypes.c_char_p()
        ok = _adlx_ok(_vcall(gpu, _GPU_NAME, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))(ctypes.byref(s)))
        return s.value.decode(errors="replace").strip() if ok and s.value else "AMD GPU"

    def integrated(self, gpu) -> bool:
        t = ctypes.c_int()
        return _adlx_ok(_vcall(gpu, _GPU_TYPE, ctypes.c_int, ctypes.POINTER(ctypes.c_int))(ctypes.byref(t))) and t.value == _ADLX_INTEGRATED

    def supported(self, gpu) -> tuple[bool, bool]:
        """(total board power, GPU power) reported for this GPU?"""
        if not self.perf:
            return False, False
        sup = ctypes.c_void_p()
        if not _adlx_ok(_vcall(self.perf, _PERF_SUPPORTED_GPU, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(gpu, ctypes.byref(sup))) or not sup.value:
            return False, False
        out = []
        for slot in (_SUP_BOARD_POWER, _SUP_GPU_POWER):
            b = ctypes.c_uint8()
            out.append(_adlx_ok(_vcall(sup, slot, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8))(ctypes.byref(b))) and bool(b.value))
        _vcall(sup, _RELEASE, ctypes.c_long)()
        return out[0], out[1]

    def power_w(self, gpu, board: bool) -> float | None:
        """The GPU's power right now, in watts: total board power, or GPU power."""
        m = ctypes.c_void_p()
        if not _adlx_ok(_vcall(self.perf, _PERF_CUR_GPU, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(gpu, ctypes.byref(m))) or not m.value:
            return None
        v = ctypes.c_double()
        ok = _adlx_ok(_vcall(m, _MET_BOARD_POWER if board else _MET_GPU_POWER, ctypes.c_int, ctypes.POINTER(ctypes.c_double))(ctypes.byref(v)))
        _vcall(m, _RELEASE, ctypes.c_long)()
        return v.value if ok and v.value >= 0 else None

    def close(self) -> None:
        for g in self.gpus:
            _vcall(g, _RELEASE, ctypes.c_long)()
        self.gpus = []
        if self.perf:
            _vcall(self.perf, _RELEASE, ctypes.c_long)()
            self.perf = None
        if self.lib is not None:
            term = self.lib.ADLXTerminate
            term.restype, term.argtypes = ctypes.c_int, []
            term()
            self.lib = None


class AdlxPowerMeter(PowerMeter):
    kind = "gpu"
    method = "adlx power sampling"

    def __init__(self, adlx: Adlx, gpu, label: str, board: bool):
        super().__init__(label)
        self.adlx, self.gpu, self.board = adlx, gpu, board
        self.reading = "total board power" if board else "GPU power"

    def read_w(self) -> float:
        w = self.adlx.power_w(self.gpu, self.board)
        return w if w is not None else 0.0


def discover_adlx(notes: list[str], gpus: list[int] | None = None) -> tuple[list[Meter], Adlx | None]:
    lib = _load_adlx()
    if lib is None:
        return [], None
    try:
        adlx = Adlx(lib)
    except (OSError, AttributeError) as e:
        notes.append(f"AMD's ADLX library is installed but did not start ({e}); update the AMD graphics driver.")
        return [], None
    meters: list[Meter] = []
    wanted = gpus if gpus is not None else _env_indices("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")  # as on Linux
    for i, g in enumerate(adlx.gpus):
        if wanted is not None and i not in wanted:
            continue
        label = f"GPU {i} {adlx.name(g)}"
        board, gpu_power = adlx.supported(g)
        if not (board or gpu_power):
            notes.append(f"{label}: the AMD driver reports no power for this GPU.")
            continue
        m = AdlxPowerMeter(adlx, g, label, board)
        if not board and adlx.integrated(g):
            notes.append(f"{label}: built-in GPU. This is the GPU power AMD's driver reports; whether it also covers the "
                         "CPU cores depends on the chip and driver. A wall meter gives the whole machine.")
        meters.append(m)
    if not meters:
        adlx.close()
        return [], None
    return meters, adlx


# ---------------------------------------------------------------- NVIDIA GPUs (NVML via ctypes)
# Where Linux systems keep the NVIDIA driver's library when the loader is not told about it: Google Colab and
# Kaggle (/usr/lib64-nvidia), Debian and Ubuntu, RHEL and SUSE, NVIDIA's container toolkit and Kubernetes device
# plugin (/usr/local/nvidia), WSL 2 (/usr/lib/wsl/lib) and NixOS. A CUDA "stubs" folder is never used: its
# library has the right name but measures nothing.
NVML_DIRS = ["/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64", "/usr/lib",
             "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/wsl/lib", "/run/opengl-driver/lib"]


def _ldconfig_nvml() -> list[str]:
    """Full paths of libnvidia-ml the system's loader cache knows about (ldconfig -p), for odd layouts."""
    exe = shutil.which("ldconfig") or ("/sbin/ldconfig" if os.path.exists("/sbin/ldconfig") else None)
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "-p"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.split("=>")[-1].strip() for line in out.splitlines() if "libnvidia-ml.so" in line and "=>" in line]


def nvml_library_names() -> list[str]:
    """Where the NVIDIA driver puts NVML: libnvidia-ml on Linux; on Windows, System32 (drivers since 2020) or
    the older NVSMI folder. Python 3.8+ does not search PATH for DLLs, so Windows paths are given in full.
    JOULES_NVML_LIB names the library outright. On Linux the loader's own search comes first, then the folders
    drivers are known to use, then the loader cache, so a driver the loader was not told about is still found."""
    forced = os.environ.get("JOULES_NVML_LIB")
    if os.name != "nt":
        names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        names += [os.path.join(d, n) for d in NVML_DIRS for n in ("libnvidia-ml.so.1", "libnvidia-ml.so")
                  if os.path.exists(os.path.join(d, n))]
        # a driver folder may hold only the versioned file (libnvidia-ml.so.550.54.15) when its links are missing
        names += sorted(f for d in NVML_DIRS for f in glob.glob(os.path.join(glob.escape(d), "libnvidia-ml.so.*[0-9]")))
        names += [x for x in _ldconfig_nvml() if "stubs" not in x]
    else:
        win = os.environ.get("SystemRoot", r"C:\Windows")
        pf = os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles", r"C:\Program Files")
        names = ["nvml.dll", os.path.join(win, "System32", "nvml.dll"), os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvml.dll")]
    out = [forced] if forced else []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def nvidia_driver_present() -> bool:
    """An NVIDIA driver is installed, whether or not its library can be loaded: nvidia-smi is on PATH, or the
    kernel module reports itself."""
    return bool(shutil.which("nvidia-smi")) or os.path.exists("/proc/driver/nvidia/version")


class NVML:
    """Minimal NVML binding. Needs only the driver's libnvidia-ml, no Python packages."""

    def __init__(self):
        lib, why = None, ""
        for name in nvml_library_names():
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as e:
                # keep the most telling reason: a file that exists but will not load says more than "no such file"
                if not why or "No such file" in why or "cannot open shared object" in why:
                    why = str(e)
                continue
        if lib is None:
            if nvidia_driver_present():
                raise OSError("an NVIDIA driver is installed but its library (libnvidia-ml.so.1) could not be loaded"
                              + (f" ({why[:200]})" if why else "") + ": find it with `find / -name 'libnvidia-ml.so*' 2>/dev/null` "
                              "and set JOULES_NVML_LIB to that path")
            raise OSError("the NVIDIA driver library (NVML) was not found")
        self.lib = lib
        if lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")

    def count(self) -> int:
        n = ctypes.c_uint()
        self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        return n.value

    def handle(self, i: int):
        h = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h)) != 0:
            raise OSError(f"no handle for GPU {i}")
        return h

    def driver_version(self) -> str:
        buf = ctypes.create_string_buffer(80)
        try:
            return buf.value.decode() if self.lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(80)) == 0 else ""
        except AttributeError:
            return ""

    def name(self, h) -> str:
        buf = ctypes.create_string_buffer(96)
        self.lib.nvmlDeviceGetName(h, buf, ctypes.c_uint(96))
        return buf.value.decode(errors="replace")

    def energy_mj(self, h):
        v = ctypes.c_ulonglong()
        return v.value if self.lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(v)) == 0 else None

    def uuid(self, h) -> str:
        buf = ctypes.create_string_buffer(96)
        try:
            ok = self.lib.nvmlDeviceGetUUID(h, buf, ctypes.c_uint(96)) == 0
        except AttributeError:
            ok = False
        return buf.value.decode(errors="replace") if ok else ""

    def mig_parent_index(self, mig_uuid: str):
        """Index of the physical GPU a MIG slice lives on, or None."""
        mig, parent, idx = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint()
        try:
            if self.lib.nvmlDeviceGetHandleByUUID(mig_uuid.encode(), ctypes.byref(mig)) != 0:
                return None
            if self.lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig, ctypes.byref(parent)) != 0:
                return None
            return idx.value if self.lib.nvmlDeviceGetIndex(parent, ctypes.byref(idx)) == 0 else None
        except AttributeError:  # drivers older than MIG
            return None

    def utilization(self, h):
        """GPU busy percentage over the driver's last sample period, or None."""
        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        u = Util()
        try:
            return u.gpu if self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0 else None
        except AttributeError:
            return None

    def processes(self, h) -> list[tuple[int, int]]:
        """(pid, GPU memory in bytes) of the compute processes on a GPU. Empty when unknown."""
        class V3(ctypes.Structure):  # nvmlProcessInfo_t, CUDA 11.4+ drivers
            _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                        ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]
        class V1(ctypes.Structure):  # older drivers
            _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]
        for fn, st in (("nvmlDeviceGetComputeRunningProcesses_v3", V3), ("nvmlDeviceGetComputeRunningProcesses", V1)):
            f = getattr(self.lib, fn, None)
            if f is None:
                continue
            n = ctypes.c_uint(64)
            buf = (st * 64)()
            if f(h, ctypes.byref(n), buf) == 0:
                # NVML reports "not available" memory as all bits set; count such a process as 0 bytes.
                return [(buf[i].pid, 0 if buf[i].usedGpuMemory >= (1 << 63) else buf[i].usedGpuMemory) for i in range(min(n.value, 64))]
        return []

    def process_sm(self, h, since_us: int):
        """Each process's SM-busy samples since `since_us` (the driver's microsecond clock), as
        ({pid: sum of SM %}, newest sample time), or None when this GPU or driver cannot say (Maxwell and newer can)."""
        class Sample(ctypes.Structure):  # nvmlProcessUtilizationSample_t
            _fields_ = [("pid", ctypes.c_uint), ("timeStamp", ctypes.c_ulonglong), ("smUtil", ctypes.c_uint),
                        ("memUtil", ctypes.c_uint), ("encUtil", ctypes.c_uint), ("decUtil", ctypes.c_uint)]
        f = getattr(self.lib, "nvmlDeviceGetProcessUtilization", None)
        if f is None:
            return None
        n = ctypes.c_uint(0)
        rc = f(h, None, ctypes.byref(n), ctypes.c_ulonglong(since_us))
        if rc == 6:  # NVML_ERROR_NOT_FOUND: no new samples, so nobody used it
            return {}, since_us
        if rc not in (0, 7) or not n.value:  # 7: NVML_ERROR_INSUFFICIENT_SIZE, the expected answer to the size query
            return ({}, since_us) if rc == 0 else None
        n = ctypes.c_uint(n.value + 16)
        buf = (Sample * n.value)()
        rc = f(h, buf, ctypes.byref(n), ctypes.c_ulonglong(since_us))
        if rc == 6:
            return {}, since_us
        if rc != 0:
            return None
        out: dict[int, int] = {}
        newest = since_us
        for i in range(n.value):
            s = buf[i]
            if s.timeStamp <= since_us:
                continue
            out[s.pid] = out.get(s.pid, 0) + min(s.smUtil, 100)
            newest = max(newest, s.timeStamp)
        return out, newest

    def accounting_sm_ms(self, h, pids):
        """With accounting mode on (`nvidia-smi -am 1`): each process's SM-busy milliseconds so far, as {pid: ms}
        (its average SM % times its run time). None when accounting is off or unsupported."""
        class Stats(ctypes.Structure):  # nvmlAccountingStats_t
            _fields_ = [("gpuUtilization", ctypes.c_uint), ("memoryUtilization", ctypes.c_uint),
                        ("maxMemoryUsage", ctypes.c_ulonglong), ("time", ctypes.c_ulonglong),
                        ("startTime", ctypes.c_ulonglong), ("isRunning", ctypes.c_uint), ("reserved", ctypes.c_uint * 5)]
        mode_f, stats_f = getattr(self.lib, "nvmlDeviceGetAccountingMode", None), getattr(self.lib, "nvmlDeviceGetAccountingStats", None)
        if mode_f is None or stats_f is None:
            return None
        mode = ctypes.c_int(0)
        if mode_f(h, ctypes.byref(mode)) != 0 or mode.value != 1:
            return None
        out = {}
        for pid in pids:
            st = Stats()
            if stats_f(h, ctypes.c_uint(pid), ctypes.byref(st)) == 0:
                out[pid] = min(st.gpuUtilization, 100) / 100 * st.time
        return out

    def power_mw(self, h):
        v = ctypes.c_uint()
        return v.value if self.lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(v)) == 0 else None

    # ---- health: what the card is doing, not only what it draws. Every call answers None when the driver lacks it.
    def _uint(self, fn: str, h, *args):
        f = getattr(self.lib, fn, None)
        if f is None:
            return None
        v = ctypes.c_uint()
        try:
            return v.value if f(h, *args, ctypes.byref(v)) == 0 else None
        except (OSError, ValueError):
            return None

    def clock_mhz(self, h, which: int = 1):
        """The clock right now, in MHz: 1 = the SM (compute) clock, 2 = the memory clock."""
        return self._uint("nvmlDeviceGetClockInfo", h, ctypes.c_uint(which))

    def max_clock_mhz(self, h, which: int = 1):
        return self._uint("nvmlDeviceGetMaxClockInfo", h, ctypes.c_uint(which))

    def temperature_c(self, h):
        return self._uint("nvmlDeviceGetTemperature", h, ctypes.c_uint(0))  # NVML_TEMPERATURE_GPU

    def power_limit_mw(self, h):
        """The power limit the card enforces now: the software cap if one is set, else the management limit."""
        v = self._uint("nvmlDeviceGetEnforcedPowerLimit", h)
        return v if v is not None else self._uint("nvmlDeviceGetPowerManagementLimit", h)

    def power_limit_range_mw(self, h):
        f = getattr(self.lib, "nvmlDeviceGetPowerManagementLimitConstraints", None)
        if f is None:
            return None
        lo, hi = ctypes.c_uint(), ctypes.c_uint()
        try:
            return (lo.value, hi.value) if f(h, ctypes.byref(lo), ctypes.byref(hi)) == 0 else None
        except (OSError, ValueError):
            return None

    def set_power_limit_mw(self, h, mw: int) -> int:
        """NVML's result code: 0 done, 4 no permission (root is needed), 3 not supported."""
        f = getattr(self.lib, "nvmlDeviceSetPowerManagementLimit", None)
        if f is None:
            return 3
        try:
            return f(h, ctypes.c_uint(int(mw)))
        except (OSError, ValueError):
            return 3

    def throttle_reasons(self, h):
        """Why the clocks are below maximum right now, as NVML's bitmask (THROTTLE_REASONS names the bits), or None."""
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            f = getattr(self.lib, fn, None)
            if f is None:
                continue
            v = ctypes.c_ulonglong()
            try:
                if f(h, ctypes.byref(v)) == 0:
                    return v.value
            except (OSError, ValueError):
                pass
        return None

    def utilization_pair(self, h):
        """(GPU busy %, memory-controller busy %) over the driver's last sample period, or None."""
        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        u = Util()
        try:
            return (u.gpu, u.memory) if self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0 else None
        except AttributeError:
            return None

    def memory_used_pct(self, h):
        class Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
        f = getattr(self.lib, "nvmlDeviceGetMemoryInfo", None)
        if f is None:
            return None
        m = Mem()
        try:
            return m.used / m.total * 100 if f(h, ctypes.byref(m)) == 0 and m.total else None
        except (OSError, ValueError):
            return None

    def close(self) -> None:
        try:
            self.lib.nvmlShutdown()
        except Exception:
            pass


NVML_FACTORY = NVML  # tests replace this with a fake

# NVML's clocks-event (throttle) reason bits and what to call them. Idle (0x1) and the application or display clock
# settings (0x2, 0x100) are not throttling: the card is slow because nothing asked for more.
THROTTLE_REASONS = [(0x4, "power cap"), (0x20, "thermal"), (0x40, "thermal"), (0x8, "hw slowdown"), (0x80, "power brake"), (0x10, "sync boost")]
THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80 | 0x10


class GpuHealth:
    """Mixed into the NVML meters: about twice a second, also note the clocks, temperature, utilisation and throttle
    reasons, so a run can say not only how much energy the GPU used but whether it was slowed down and what it
    was waiting on. Everything here is optional: a driver (or a fake) without these calls gives None."""
    HEALTH_EVERY = 0.5

    def _health_start(self, now: float) -> None:
        self._hs = {"n": 0, "sm": 0.0, "mem_clock": 0.0, "util_n": 0, "util": 0.0, "mem_util": 0.0, "temp": None,
                    "r_n": 0, "throttled": 0, "reasons": {}, "mem_used": None, "last": None}
        self._h_t = now - self.HEALTH_EVERY
        nv, h = self.nvml, self.h
        self._h_max = nv.max_clock_mhz(h) if hasattr(nv, "max_clock_mhz") else None
        self._h_limit = nv.power_limit_mw(h) if hasattr(nv, "power_limit_mw") else None

    def _health_poll(self, now: float) -> None:
        if now - self._h_t < self.HEALTH_EVERY:
            return
        self._h_t = now
        nv, h, s = self.nvml, self.h, self._hs
        if not hasattr(nv, "clock_mhz"):
            return
        sm, mc, t, r = nv.clock_mhz(h, 1), nv.clock_mhz(h, 2), nv.temperature_c(h), nv.throttle_reasons(h)
        u, mu = nv.utilization_pair(h), nv.memory_used_pct(h)
        last = {"sm_clock_mhz": sm, "mem_clock_mhz": mc, "temp_c": t, "reasons": r, "gpu_util_pct": None, "mem_util_pct": None}
        if sm is not None:
            s["n"] += 1; s["sm"] += sm; s["mem_clock"] += mc or 0
        if t is not None:
            s["temp"] = t if s["temp"] is None else max(s["temp"], t)
        if r is not None:
            s["r_n"] += 1
            if r & THROTTLE_MASK:
                s["throttled"] += 1
                for bit, name in THROTTLE_REASONS:
                    if r & bit:
                        s["reasons"][name] = s["reasons"].get(name, 0) + 1
        if u is not None:
            s["util_n"] += 1; s["util"] += u[0]; s["mem_util"] += u[1]
            last["gpu_util_pct"], last["mem_util_pct"] = u[0], u[1]
        if mu is not None:
            s["mem_used"] = mu if s["mem_used"] is None else max(s["mem_used"], mu)
        s["last"] = last

    def health(self) -> dict | None:
        """The run's summary: average clocks (and as a share of the maximum), average busy and memory-controller
        percentages, the hottest temperature, the power limit, the share of samples the card was throttled and why,
        and the most memory in use. None when the driver gave none of it."""
        return self._summarize(getattr(self, "_hs", None))

    def health_state(self) -> dict | None:
        """A copy of the running totals, so a phase can later ask what happened since (health_since)."""
        s = getattr(self, "_hs", None)
        if not s:
            return None
        c = dict(s)
        c["reasons"] = dict(s["reasons"])
        return c

    def health_since(self, before: dict | None) -> dict | None:
        """What the card did between a health_state() and now: the same summary as health(), over that stretch only.
        Temperatures are the readings at its start and end (temp_c_start, temp_c_end) instead of a maximum."""
        s = getattr(self, "_hs", None)
        if not s or not before:
            return None
        d = {k: s[k] - before[k] for k in ("n", "sm", "mem_clock", "util_n", "util", "mem_util", "r_n", "throttled")}
        d["reasons"] = {k: v - before["reasons"].get(k, 0) for k, v in s["reasons"].items() if v - before["reasons"].get(k, 0) > 0}
        d["temp"] = d["mem_used"] = None
        out = self._summarize(d) or {}
        t0 = (before.get("last") or {}).get("temp_c")
        t1 = (s.get("last") or {}).get("temp_c")
        if t0 is not None:
            out["temp_c_start"] = t0
        if t1 is not None:
            out["temp_c_end"] = t1
        if self._h_limit:
            out["power_limit_w"] = self._h_limit / 1000
        return out or None

    def _summarize(self, s: dict | None) -> dict | None:
        if not s or (s["n"] == 0 and s["util_n"] == 0 and s["r_n"] == 0):
            return None
        out: dict = {}
        if s["n"]:
            out["sm_clock_mhz"] = s["sm"] / s["n"]
            out["mem_clock_mhz"] = s["mem_clock"] / s["n"]
            if self._h_max:
                out["sm_clock_max_mhz"] = self._h_max
                out["clock_pct"] = out["sm_clock_mhz"] / self._h_max * 100
        if s["util_n"]:
            out["gpu_util_pct"] = s["util"] / s["util_n"]
            out["mem_util_pct"] = s["mem_util"] / s["util_n"]
        if s["temp"] is not None:
            out["temp_c_max"] = s["temp"]
        if self._h_limit:
            out["power_limit_w"] = self._h_limit / 1000
        if s["r_n"]:
            out["throttled_pct"] = s["throttled"] / s["r_n"] * 100
            out["throttle_reasons"] = {k: v / s["r_n"] * 100 for k, v in sorted(s["reasons"].items(), key=lambda kv: -kv[1])}
        if s["mem_used"] is not None:
            out["mem_used_pct_max"] = s["mem_used"]
        return out

    def temperature_now(self) -> float | None:
        """A fresh temperature reading, for waiting until the card has cooled down between runs."""
        nv = self.nvml
        return nv.temperature_c(self.h) if hasattr(nv, "temperature_c") else None

    def health_now(self) -> dict | None:
        """The latest reading, for the agent's metrics: clocks, temperature, utilisation, whether throttled and why."""
        s = getattr(self, "_hs", None)
        if not s or not s["last"]:
            return None
        d = dict(s["last"])
        r = d.pop("reasons")
        d["throttled"] = None if r is None else bool(r & THROTTLE_MASK)
        d["throttle_reason"] = next((n for b, n in THROTTLE_REASONS if r and r & b), "")
        d["power_limit_w"] = self._h_limit / 1000 if self._h_limit else None
        d["sm_clock_max_mhz"] = self._h_max
        return d


class NvmlCounterMeter(GpuHealth, CounterMeter):
    method = "nvml counter"
    kind = "gpu"

    def __init__(self, nvml, h, label):
        super().__init__(label)
        self.nvml, self.h = nvml, h

    def read_j(self) -> float:
        return self.nvml.energy_mj(self.h) / 1000.0

    def start(self, now: float) -> None:
        super().start(now)
        self._health_start(now)

    def poll(self, now: float) -> None:
        super().poll(now)
        self._health_poll(now)


class NvmlPowerMeter(GpuHealth, PowerMeter):
    method = "nvml power sampling"
    kind = "gpu"

    def __init__(self, nvml, h, label):
        super().__init__(label)
        self.nvml, self.h = nvml, h

    def read_w(self) -> float:
        return (self.nvml.power_mw(self.h) or 0) / 1000.0

    def start(self, now: float) -> None:
        super().start(now)
        self._health_start(now)

    def poll(self, now: float) -> None:
        super().poll(now)
        self._health_poll(now)


def visible_gpus(explicit: list[int] | None, nvml=None, notes: list[str] | None = None) -> list[int] | None:
    """NVML indices of the GPUs this job can use: --gpus, else CUDA_VISIBLE_DEVICES (indices, GPU UUIDs
    as Kubernetes and Slurm set them, or MIG slices), else None for all."""
    if explicit is not None:
        return explicit
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env:
        return None
    notes = notes if notes is not None else []
    out: list[int] = []
    uuids = None
    for tok in (t.strip() for t in env.split(",")):
        if re.fullmatch(r"\d+", tok):
            out.append(int(tok))
        elif tok.startswith("GPU-") and nvml is not None:
            if uuids is None:
                uuids = [nvml.uuid(nvml.handle(i)) for i in range(nvml.count())]
            hit = [i for i, u in enumerate(uuids) if u and u.startswith(tok)]  # CUDA accepts a unique prefix
            if len(hit) != 1:
                break  # CUDA ignores this device and every one after it
            out.append(hit[0])
        elif tok.startswith("MIG-") and nvml is not None and hasattr(nvml, "mig_parent_index"):
            i = nvml.mig_parent_index(tok)
            if i is None:
                break
            if i not in out:
                out.append(i)
            msg = "MIG: NVIDIA reports power for the whole GPU, so other MIG slices on it are included."
            if msg not in notes:
                notes.append(msg)
        else:
            break
    return out


def discover_nvml(notes: list[str], gpus: list[int] | None) -> tuple[list[Meter], object | None]:
    try:
        nvml = NVML_FACTORY()
    except OSError as e:
        if "driver is installed" in str(e):
            notes.append(str(e)[0].upper() + str(e)[1:] + ".")
        return [], None
    meters: list[Meter] = []
    wanted = visible_gpus(gpus, nvml, notes)
    if (gpus is None and wanted is not None and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            and re.search(r"(^|,)\s*\d", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            and len({nvml.name(nvml.handle(i)) for i in range(nvml.count())}) > 1):
        notes.append("This machine has different GPU models. CUDA numbers them fastest first and nvidia-smi by "
                     "PCI bus, so CUDA_VISIBLE_DEVICES numbers may point at other cards. Set "
                     "CUDA_DEVICE_ORDER=PCI_BUS_ID, or pass --gpus with nvidia-smi numbers.")
    for i in range(nvml.count()):
        if wanted is not None and i not in wanted:
            continue
        h = nvml.handle(i)
        label = f"GPU {i} {nvml.name(h)}"
        if nvml.energy_mj(h) is not None:
            meters.append(NvmlCounterMeter(nvml, h, label))
        elif nvml.power_mw(h) is not None:
            meters.append(NvmlPowerMeter(nvml, h, label))
        else:
            notes.append(f"{label}: the driver reports neither energy nor power.")
    return meters, nvml


# ---------------------------------------------------------------- Apple Silicon (powermetrics)
_PM_ELAPSED = re.compile(r"\(([\d.]+)\s*ms elapsed\)")
_PM_COMBINED = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
_PM_PART = re.compile(r"^(CPU|GPU|ANE) Power:\s*([\d.]+)\s*mW")
_PM_INTEL = re.compile(r"package power[^:]*:\s*([\d.]+)\s*W", re.I)  # Intel Macs: "... package power (CPUs+GPU+DRAM): 5.47W"


def apple_silicon() -> bool:
    """True on M-series Macs, also when this Python is an Intel build running under Rosetta."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=5).stdout
        return out.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return platform.machine() == "arm64"


class PowermetricsMeter(Meter):
    """Apple Silicon. Each powermetrics sample is the average power over its own window, so the energy
    is the sum of power x window over the samples taken while the command ran."""
    method = "powermetrics sampling"
    kind = "soc"
    CMD = ("powermetrics", "--samplers", "cpu_power,gpu_power", "-i", "200")

    def __init__(self, cmd=None, intel: bool = False, sudo: bool = False):
        if intel:  # Intel Macs report one package figure (CPU cores, integrated GPU and DRAM)
            super().__init__("Intel Mac CPU package (CPU + GPU + DRAM)")
            self.kind = "cpu"
            cmd = cmd or ("powermetrics", "--samplers", "cpu_power", "-i", "200")
        else:
            super().__init__("Apple SoC (CPU + GPU + ANE)")
        cmd = tuple(cmd or self.CMD)
        if sudo:  # credentials were cached by `sudo -v`; -n never prompts inside the pseudo-terminal
            cmd = ("sudo", "-n") + cmd
        import pty  # POSIX only; this meter only exists on macOS
        self._cv = threading.Condition()
        self._on = False
        self._window = 0.2
        self._parts: dict[str, float] = {}
        self._combined = False  # this sample already had its Combined line
        self._seen = 0
        self._t_start, self._t_stop = 0.0, float("inf")
        # Writing to a pipe, powermetrics holds its output back in large blocks, so samples arrive late or
        # only when it exits (and the run looks almost free). A pseudo-terminal makes it print each sample
        # as soon as it is taken.
        master, slave = pty.openpty()
        try:
            self.proc = subprocess.Popen(list(cmd), stdin=subprocess.DEVNULL, stdout=slave,
                                         stderr=subprocess.DEVNULL, close_fds=True)
        finally:
            os.close(slave)
        self._out = os.fdopen(master, "r", errors="replace")
        threading.Thread(target=self._reader, daemon=True).start()
        with self._cv:
            self._cv.wait_for(lambda: self._seen > 0, timeout=3.0)

    def _add(self, watts: float) -> None:
        # A sample is the average over the window that just ended. Only the part of it inside the run counts:
        # the first sample began before the run and the last one ends after it (0.2 s each way, which on a
        # 2-second run is up to a fifth of its energy).
        t1 = time.monotonic()
        with self._cv:
            if self._on:
                inside = min(t1, self._t_stop) - max(t1 - self._window, self._t_start)
                if inside > 0:
                    self.energy_j += watts * inside
                    self.peak_w = max(self.peak_w, watts)
            self._seen += 1
            self._cv.notify_all()

    def _reader(self) -> None:
        try:
            for line in self._out:
                self._line(line)
        except OSError:
            pass  # the terminal reports EIO once powermetrics exits

    def _line(self, line: str) -> None:
        """One line of powermetrics output. Each sample is added exactly once: its Combined line when it has
        one, else the sum of its CPU, GPU and ANE lines. The gpu_power sampler prints a second "GPU Power"
        line after the Combined one; it is part of that same sample and must not be added again (it was, as
        a parts-only sample, until 0.8.0rc2: five samples became nine additions)."""
        m = _PM_ELAPSED.search(line)
        if m:  # a new sample starts; flush a previous one that had only per-part lines
            if self._parts and not self._combined:
                self._add(sum(self._parts.values()) / 1000.0)
            self._parts, self._combined = {}, False
            self._window = float(m.group(1)) / 1000.0
            return
        m = _PM_COMBINED.search(line)
        if m:
            if not self._combined:  # a second Combined line in one sample is not a second sample (Eric Spencer)
                self._add(float(m.group(1)) / 1000.0)
            self._parts, self._combined = {}, True
            return
        m = _PM_INTEL.search(line)
        if m:  # already in watts
            self._add(float(m.group(1)))
            return
        m = _PM_PART.search(line.strip())
        if m and not self._combined:
            self._parts[m.group(1)] = float(m.group(2))

    def start(self, now: float) -> None:
        with self._cv:
            super().start(now)
            self._t_start, self._t_stop = now, float("inf")
            self._on = True

    def poll(self, now: float) -> None:
        self._t_last = now

    def stop(self, now: float) -> None:
        with self._cv:  # take the sample that covers the last moments of the run
            self._t_stop = now
            target = self._seen + 1
            self._cv.wait_for(lambda: self._seen >= target, timeout=self._window + 0.5)
            self._on = False
        self._t_last = now

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        try:
            self._out.close()
        except OSError:
            pass


# ---------------------------------------------------------------- Estimate from CPU time (no counters, e.g. CI VMs)
class CpuTimeMeter(Meter):
    """Not a measurement: CPU seconds used by the command (and the processes it waited for) times an
    assumed power per busy core. Steady and comparable between runs, which is what a CI check needs;
    absolute numbers are only as good as the watts-per-core figure (JOULES_CPU_W, default 3)."""
    kind = "est"

    def __init__(self, watts_per_core: float | None = None):
        self.w = float(watts_per_core or os.environ.get("JOULES_CPU_W") or 3.0)
        super().__init__(f"CPU time × {self.w:g} W per core")
        self.method = "ESTIMATE (CPU time)"
        self.cpu_s = 0.0

    def attach(self, proc) -> None:
        """The process being measured. Windows has no children's CPU total, so its own counters are read."""
        self._proc = proc

    def _children_cpu(self) -> float:
        if os.name == "nt":
            return _windows_cpu_s(getattr(self, "_proc", None))
        import resource  # POSIX: every child the command waited for, at any depth
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime

    def start(self, now: float) -> None:
        super().start(now)
        self._c0 = 0.0 if os.name == "nt" else self._children_cpu()
        self.cpu_s = 0.0

    def poll(self, now: float) -> None:
        self._t_last = now  # children's CPU time is only known once they have been waited for

    def stop(self, now: float) -> None:
        self.cpu_s = max(0.0, self._children_cpu() - self._c0)
        self.energy_j = self.cpu_s * self.w
        self._t_last = now
        d = self.duration_s
        self.peak_w = self.energy_j / d if d > 0 else 0.0


def _windows_cpu_s(proc) -> float:
    """User + kernel CPU seconds of a finished process, from GetProcessTimes (its own threads; Windows keeps no
    total for the processes it started)."""
    if proc is None or getattr(proc, "_handle", None) is None:
        return 0.0
    from ctypes import wintypes
    ft = [wintypes.FILETIME() for _ in range(4)]  # creation, exit, kernel, user
    if not ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(proc._handle)), *[ctypes.byref(f) for f in ft]):
        return 0.0
    return sum(((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7 for f in ft[2:])


def cpu_estimate_available() -> bool:
    if os.name == "nt":
        return True
    try:
        import resource  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- Simulation (demos and tests)
class SimulatedMeter(PowerMeter):
    method = "SIMULATED"

    def __init__(self, kind: str, label: str, watts: float):
        super().__init__(label)
        self.kind = kind
        self.watts = watts

    def read_w(self) -> float:
        return self.watts


def parse_simulation(spec: str) -> list[Meter]:
    """JOULES_SIMULATE="gpu:H100 SXM:650,cpu:package 0:80" -> constant-power meters."""
    out: list[Meter] = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        kind, label, w = part.split(":")
        out.append(SimulatedMeter(kind, f"{kind.upper()} {label}", float(w)))
    return out


def mac_sudo(notes: list[str]) -> bool:
    """macOS only lets root read the power meters (powermetrics). Rather than running the whole command as root,
    ask for the password once (sudo caches it for a few minutes) and run only powermetrics with sudo."""
    if os.environ.get("JOULES_NO_SUDO") or not shutil.which("sudo"):
        notes.append("macOS: the power meters need administrator rights. Run joules with sudo.")
        return False
    if subprocess.call(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        return True  # already cached
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        notes.append("macOS: the power meters need administrator rights. Run joules with sudo, or in a terminal so it can ask.")
        return False
    sys.stderr.write("joules: macOS lets only administrators read the power meters; your command still runs as you.\n")
    sys.stderr.flush()
    if subprocess.call(["sudo", "-v", "-p", "Password for %u (to read the power meters): "]) == 0:
        return True
    notes.append("macOS: without your password joules cannot read the power meters.")
    return False


# ---------------------------------------------------------------- discovery
@dataclass
class Discovery:
    meters: list[Meter]
    notes: list[str] = field(default_factory=list)
    handles: list = field(default_factory=list)

    @property
    def simulated(self) -> bool:
        return any(isinstance(m, SimulatedMeter) for m in self.meters)

    def close(self) -> None:
        for m in self.meters:
            m.close()
        for h in self.handles:
            h.close()


def discover(gpus: list[int] | None = None, cpu: bool = True, estimate: bool = False) -> Discovery:
    notes: list[str] = []
    sim = os.environ.get("JOULES_SIMULATE")
    if sim:
        return Discovery(parse_simulation(sim), ["Simulated meters (JOULES_SIMULATE is set): numbers are not real measurements."])
    meters, handles = [], []
    nv, handle = discover_nvml(notes, gpus)
    meters += nv
    if handle:
        handles.append(handle)
    meters += discover_amdgpu(notes, gpus)
    amd_win, adlx = discover_adlx(notes, gpus)
    meters += amd_win
    if adlx:
        handles.append(adlx)
    meters += discover_intelgpu(notes)
    if cpu:
        meters += discover_rapl(notes)
        if platform.system() == "Darwin":
            root = hasattr(os, "geteuid") and os.geteuid() == 0
            sudo = not root and mac_sudo(notes)
            if root or sudo:
                try:
                    m = PowermetricsMeter(intel=not apple_silicon(), sudo=sudo)
                    if m.proc.poll() is not None and m.proc.returncode != 0:
                        m.close()
                        notes.append("powermetrics would not start (it needs an administrator account).")
                    elif m._seen == 0:  # started, but no reading within 3 s (seen on virtual Macs)
                        m.close()
                        notes.append("powermetrics gave no power readings on this Mac (common in virtual machines).")
                    else:
                        meters.append(m)
                except OSError:
                    notes.append("powermetrics could not be started.")
    if not meters and estimate and cpu_estimate_available():
        m = CpuTimeMeter()
        notes.append(f"No energy counters here (normal on cloud VMs and CI runners), so energy is ESTIMATED from CPU time at "
                     f"{m.w:g} W per busy core (set JOULES_CPU_W to change it). Good for comparing runs; for measured "
                     "numbers use a machine with counters.")
        return Discovery([m], notes, handles)
    if not meters:
        notes.append("No energy counters found. Supported: NVIDIA GPUs (driver installed), AMD GPUs (amdgpu on Linux, ADLX on Windows), Intel Arc/Data Center GPUs (i915/xe), Intel/AMD CPUs (RAPL on Linux), Macs (powermetrics, with your password).")
    return Discovery(meters, notes, handles)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()
