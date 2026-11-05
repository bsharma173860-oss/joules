"""joules bench gpu: one fixed test, the same on every card, so GPUs can be compared on the leaderboard.

Square matrix multiplications (8192 x 8192, random normal values) in fp16 and then fp32, each for 20 seconds after
a 5-second warm-up, on one GPU through PyTorch. Energy is the GPU's own meter over exactly the measured stretch: the
NVIDIA energy counter on Volta and newer, power sampled and integrated on older cards and on AMD. Each
multiplication is 2·n³ operations, so the results are TFLOP/s and GFLOP per joule. fp32 runs in full precision
(TF32 is switched off), so it means the same on every generation.

PyTorch is not a dependency of joules: the test uses the one already installed (Kaggle and Colab have it)."""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time

from . import meters
from .api import NoMeters, Session

TEST = "matmul-8192-v1"
N = 8192
SECONDS = 20.0
WARMUP_S = 5.0
PRECISIONS = ("fp16", "fp32")
TOO_SLOW_S = 4.0     # one multiplication slower than this: the card has no fast path for that precision
MIN_BUSY_PCT = 50.0  # the meter's card was mostly idle during the test: it is not the card that did the work


class GpuBenchError(RuntimeError):
    """The test could not run, with the reason and what to do."""


class NoTorch(GpuBenchError):
    """PyTorch is not installed in the Python that joules runs in."""


def load_torch():
    # CUDA numbers GPUs fastest first and NVML by PCI bus; in this process CUDA uses NVML's order, so GPU n is the
    # same card for both (CUDA reads this when it starts, so it has to be set before torch is imported).
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    try:
        import torch
    except ImportError:
        raise NoTorch("the GPU test runs on PyTorch, which is not installed here. Kaggle and Colab have it; "
                      "elsewhere install it from pytorch.org (the CUDA build for NVIDIA, ROCm for AMD).") from None
    if not torch.cuda.is_available():
        raise GpuBenchError("PyTorch sees no GPU. On Kaggle, pick a GPU under Settings → Accelerator; on Colab, Runtime → "
                            "Change runtime type. Elsewhere, check that the PyTorch build matches the driver (CUDA or ROCm).")
    return torch


def torch_python() -> str | None:
    """Another Python on this machine that has PyTorch: joules installed by install.sh lives in its own environment,
    and the user's PyTorch in theirs (a conda env, a venv, the system Python). None when there is none."""
    if os.environ.get("JOULES_NO_REEXEC"):
        return None
    here = os.path.abspath(sys.executable)
    for name in ("python3", "python"):
        py = shutil.which(name)
        if not py or os.path.abspath(py) == here:
            continue
        try:
            if subprocess.run([py, "-c", "import sys, torch; sys.exit(sys.version_info < (3, 9))"], capture_output=True, timeout=120).returncode == 0:
                return py
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def run_with(py: str, argv: list[str]) -> int:
    """joules bench gpu again, in that Python: joules has no dependencies, so its own package is put on the path."""
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=pkg + (os.pathsep + path if path else ""), JOULES_NO_REEXEC="1")
    return subprocess.call([py, "-m", "joules", *argv], env=env)


def nvml_index(i: int) -> int:
    """The NVML number of what PyTorch calls GPU i: CUDA_VISIBLE_DEVICES renumbers the cards a process sees."""
    try:
        nv = meters.NVML_FACTORY()
    except OSError:
        return i
    try:
        vis = meters.visible_gpus(None, nv, [])
        return vis[i] if vis is not None and i < len(vis) else i
    finally:
        nv.close()


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^(nvidia|amd)\s+", "", name.lower()))


def same_card(torch_name: str, meter_name: str) -> bool:
    a, b = _key(torch_name), _key(meter_name)
    return bool(a and b) and (a in b or b in a)


def device_name(meter_name: str) -> str:
    # "GPU 0 NVIDIA H100 80GB HBM3" -> "NVIDIA H100 80GB HBM3"
    return re.sub(r"\s+", " ", re.sub(r"^(GPU|CARD)\s*\d+\s+", "", meter_name.strip(), flags=re.I))[:120]


def _precision(torch, dev, prec: str, s: Session, meter, n: int, seconds: float, warmup_s: float, log) -> dict | None:
    dtype = torch.float16 if prec == "fp16" else torch.float32
    g = torch.Generator(device=dev).manual_seed(0)
    a = torch.randn(n, n, device=dev, dtype=dtype, generator=g)
    b = torch.randn(n, n, device=dev, dtype=dtype, generator=g)
    c = torch.empty_like(a)
    try:
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        torch.matmul(a, b, out=c)
        torch.cuda.synchronize(dev)
        first = time.perf_counter() - t0
        if first > TOO_SLOW_S:
            return None
        log(f"{prec}: warming up for {warmup_s:g} s…")
        k, t0 = 0, time.perf_counter()
        while k < 2 or time.perf_counter() - t0 < warmup_s:
            torch.matmul(a, b, out=c)
            torch.cuda.synchronize(dev)
            k += 1
        batch = max(1, int(0.25 / ((time.perf_counter() - t0) / k)))  # check the clock about four times a second
        log(f"{prec}: measuring for {seconds:g} s…")
        iters = 0
        with s.phase(prec) as p:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                for _ in range(batch):
                    torch.matmul(a, b, out=c)
                torch.cuda.synchronize(dev)
                iters += batch
        r = p.result
    finally:
        del a, b, c
        torch.cuda.empty_cache()
    e, d = r["devices"].get(meter.name, r["energy_j"]), r["duration_s"]
    if e <= 0 or d <= 0:
        raise GpuBenchError(f"the GPU's meter did not move during the {prec} test, so there is nothing to report.")
    flops = 2 * n ** 3 * iters
    h = (r.get("gpu") or {}).get(meter.name) or {}
    busy = h.get("gpu_util_pct")
    if busy is not None and busy < MIN_BUSY_PCT:
        raise GpuBenchError(f"the card joules measured ({meter.name}) was only {busy:.0f}% busy during the test, so another "
                            "card did the work. Set CUDA_VISIBLE_DEVICES to the one GPU to test.")
    out = {"multiplications": iters, "flops": flops, "duration_s": d, "energy_j": e, "tflops": flops / d / 1e12,
           "avg_w": e / d, "gflop_per_j": flops / e / 1e9}
    for k in ("clock_pct", "throttled_pct", "throttle_reasons", "temp_c_start", "temp_c_end", "gpu_util_pct", "power_limit_w"):
        if k in h:
            out[k] = h[k]
    return out


def run(gpu: int = 0, torch=None, n: int | None = None, seconds: float | None = None, warmup_s: float | None = None,
        interval: float = 0.1, log=lambda m: None) -> dict:
    """Runs the test on PyTorch's GPU `gpu` and returns the report (kind "gpubench"). Raises GpuBenchError.
    n, seconds and warmup_s are for tests; the board only takes the standard test."""
    n, seconds = n or N, seconds or SECONDS
    warmup_s = WARMUP_S if warmup_s is None else warmup_s
    torch = torch or load_torch()
    count = torch.cuda.device_count()
    if not 0 <= gpu < count:
        raise GpuBenchError(f"there is no GPU {gpu}: PyTorch sees {count} ({', '.join(str(i) for i in range(count))}).")
    dev = torch.device(f"cuda:{gpu}")
    tname = torch.cuda.get_device_name(gpu)
    free = torch.cuda.mem_get_info(gpu)[0]
    need = 3 * n * n * 4 + (256 << 20)
    if free < need:
        raise GpuBenchError(f"GPU {gpu} has {free / 2**30:.1f} GB free and the test needs about {need / 2**30:.1f} GB. "
                            "Stop what else runs on it (in a notebook, restart the session) and try again.")
    try:
        s = Session(gpus=[nvml_index(gpu)], cpu=False, interval=interval)
    except NoMeters as e:
        raise GpuBenchError(f"no energy meter for this GPU: {e}") from None
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    try:
        gm = [m for m in s.meters if m.kind == "gpu"]
        if len(gm) != 1:
            raise GpuBenchError(f"joules found {len(gm)} GPU meters for PyTorch's GPU {gpu}; it needs exactly one.")
        meter = gm[0]
        if meter.method.startswith("nvml") and not same_card(tname, meter.name):
            raise GpuBenchError(f"PyTorch's GPU {gpu} is a {tname}, but the meter joules found is {device_name(meter.name)}. "
                                "Set CUDA_VISIBLE_DEVICES to the one GPU to test.")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 means full fp32 on every card
        results, notes = {}, list(s.notes)
        try:
            for prec in PRECISIONS:
                r = _precision(torch, dev, prec, s, meter, n, seconds, warmup_s, log)
                if r is None:
                    notes.append(f"{prec} is not measured: one multiplication took more than {TOO_SLOW_S:g} s, so this card "
                                 f"has no fast {prec} path (common for fp16 on older consumer cards).")
                else:
                    results[prec] = r
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        simulated = s.discovery.simulated
        method = meter.method
    finally:
        s.close()
    if not results:
        raise GpuBenchError("neither precision could be measured on this card.")
    ver = getattr(torch, "version", None)
    runtime = (f"CUDA {ver.cuda}" if getattr(ver, "cuda", None) else f"ROCm {ver.hip}" if getattr(ver, "hip", None) else None)
    limit = next((r["power_limit_w"] for r in results.values() if r.get("power_limit_w")), None)
    return {"schema": "joules/1", "kind": "gpubench", "test": TEST, "n": n, "seconds": seconds, "started_at": started,
            "duration_s": time.perf_counter() - t0, "simulated": simulated, "gpu": gpu, "device": device_name(meter.name),
            "torch_name": tname, "meter": method, "power_limit_w": limit, "torch": getattr(torch, "__version__", None),
            "runtime": runtime, "results": results, "notes": notes}
