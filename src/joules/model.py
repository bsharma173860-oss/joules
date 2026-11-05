"""The WattOps estimate (same formulas and hardware table as the website and API), plus
calibration: what a measured run says the estimate's assumptions really were."""
from __future__ import annotations

import base64
import json
import math

K_B = 1.380649e-23
SITE = "https://wattops.dev"

GPUS = {
    "h100": ("NVIDIA H100 SXM", 700, 989), "h200": ("NVIDIA H200 SXM", 700, 989),
    "b200": ("NVIDIA B200", 1000, 2250), "a100": ("NVIDIA A100 SXM", 400, 312),
    "l40s": ("NVIDIA L40S", 350, 362), "mi300x": ("AMD Instinct MI300X", 750, 1307),
    "rtx4090": ("NVIDIA RTX 4090", 450, 165), "t4": ("NVIDIA Tesla T4", 70, 65),
    "rtx5090": ("NVIDIA RTX 5090", 575, 209), "rtxpro6000": ("NVIDIA RTX PRO 6000 Blackwell", 600, 504),
    "rtx3070ti": ("NVIDIA RTX 3070 Ti", 290, 43.5),
}
# Carbon intensity presets in g CO2e/kWh. Canadian provinces: ECCC consumption intensities (National Inventory
# Report) as used by the federal offset system; Ontario: the 2025 Toronto grid average from Google's data below.
# Cloud regions: each Google Cloud region's local grid average in 2025 (GoogleCloudPlatform/region-carbon-info,
# Apache-2.0); other clouds' regions in the same place draw from the same grid.
GRIDS = {
    "world": 480, "us": 370, "eu": 250, "ca": 120, "bc": 15, "qc": 2, "on": 74, "ab": 540, "sk": 730,
    "ns": 690, "fr": 55, "de": 380, "uk": 200, "no": 30, "in": 710, "cn": 560,
    "africa-south1": 641, "asia-east1": 428, "asia-east2": 489, "asia-northeast1": 449,
    "asia-northeast2": 299, "asia-northeast3": 359, "asia-south1": 598, "asia-south2": 473,
    "asia-southeast1": 366, "asia-southeast2": 588, "asia-southeast3": 365, "australia-southeast1": 468,
    "australia-southeast2": 416, "europe-central2": 499, "europe-north1": 30, "europe-north2": 19,
    "europe-southwest1": 96, "europe-west1": 126, "europe-west2": 109, "europe-west3": 276,
    "europe-west4": 194, "europe-west6": 21, "europe-west8": 202, "europe-west9": 16, "europe-west10": 276,
    "europe-west12": 202, "me-central1": 370, "me-central2": 381, "me-west1": 381,
    "northamerica-northeast1": 11, "northamerica-northeast2": 74, "northamerica-south1": 293,
    "southamerica-east1": 73, "southamerica-west1": 210, "us-central1": 432, "us-central2": 417,
    "us-east1": 579, "us-east2": 391, "us-east4": 343, "us-east5": 343, "us-east7": 340, "us-south1": 299,
    "us-west1": 98, "us-west2": 150, "us-west3": 568, "us-west4": 360,
}

# Fragments of NVML / sysfs device names -> table id, so a measured GPU picks its own specs.
_NAME_HINTS = [("h100", "h100"), ("h200", "h200"), ("b200", "b200"), ("a100", "a100"),
               ("l40s", "l40s"), ("mi300", "mi300x"), ("4090", "rtx4090"), ("tesla t4", "t4"), (" t4", "t4"),
               ("rtx pro 6000", "rtxpro6000"), ("5090", "rtx5090"), ("3070 ti", "rtx3070ti")]


def gpu_from_name(name: str) -> str | None:
    n = name.lower()
    for frag, gid in _NAME_HINTS:
        if frag in n:
            return gid
    return None


def landauer_floor_j(temp_k: float = 300.0, bits: float = 1.0) -> float:
    return K_B * temp_k * math.log(2) * bits


def training_flops(params_b: float, tokens_b: float) -> float:
    return 6 * params_b * 1e9 * tokens_b * 1e9


def estimate_accelerator_j(flops: float, gpu: str, mfu_pct: float, draw_pct: float) -> float:
    """Accelerator energy the website would predict for this much compute."""
    _, tdp, peak = GPUS[gpu]
    busy_s = flops / (peak * 1e12 * mfu_pct / 100)
    return tdp * (draw_pct / 100) * busy_s


def calibrate(flops: float, gpu: str, n_gpus: int, wall_s: float, gpu_j: float) -> dict:
    """From a measured run: effective utilisation and average power as a share of the rating."""
    _, tdp, peak = GPUS[gpu]
    mfu = flops / (peak * 1e12 * n_gpus * wall_s) * 100 if wall_s > 0 else 0.0
    draw = gpu_j / (tdp * n_gpus * wall_s) * 100 if wall_s > 0 else 0.0
    return {"gpu": gpu, "effective_mfu_pct": mfu, "avg_draw_pct": draw}


def calculator_link(state: dict) -> str:
    """Link that opens the website calculator with these inputs (same ?s= format as Share link)."""
    raw = base64.b64encode(json.dumps(state, separators=(",", ":")).encode()).decode()
    return f"{SITE}/?s=" + raw.replace("+", "-").replace("/", "_").rstrip("=")


def gpu_verdict(h: dict | None) -> dict | None:
    """What a run's GPU health numbers say, in a few words. The busy percentage is the share of time any kernel ran;
    NVML's memory percentage is the share of time the memory controller was busy, not bandwidth. Both are coarse,
    so the reading stays coarse: compute-bound, memory-bound, mostly idle or mixed, and whether the card was
    throttled (below its maximum clock for a reason other than having nothing to do) a meaningful share of the time."""
    if not h:
        return None
    util, mem = h.get("gpu_util_pct"), h.get("mem_util_pct")
    bound = None
    if util is not None:
        if util < 10:
            bound = "idle"
        elif mem is not None and mem >= 50 and mem >= 0.8 * util:
            bound = "memory"
        elif util >= 60:
            bound = "compute"
        else:
            bound = "mixed"
    thr = float(h.get("throttled_pct") or 0.0)
    reasons = h.get("throttle_reasons") or {}
    reason = max(reasons, key=reasons.get) if reasons else (h.get("reason") or "")
    throttled = thr >= 10
    words = {"memory": "memory-bound", "compute": "compute-bound", "idle": "mostly idle", "mixed": "mixed load"}
    parts = [words[bound]] if bound else []
    if throttled:
        parts.append(f"throttled {thr:.0f}% of the time" + (f" ({reason})" if reason else ""))
    return {"bound": bound, "throttled": throttled, "throttled_pct": thr, "reason": reason, "summary": " · ".join(parts)}


def power_sweep_pick(rows: list[dict], max_slowdown_pct: float = 10.0) -> dict | None:
    """Among power-limit settings (each with limit_pct, j_per_token, tokens_per_s), the one with the least energy per
    token that is at most max_slowdown_pct slower than the highest limit tried. The idea is from Zeus (You et al.,
    ML.ENERGY, 2023): a GPU's power rises faster than its speed as the clock climbs, so a lower cap often buys a
    large energy saving for a small delay."""
    rows = [r for r in rows if r.get("j_per_token") and r.get("tokens_per_s")]
    if not rows:
        return None
    base = max(rows, key=lambda r: r["limit_pct"])
    ok = [r for r in rows if r["tokens_per_s"] >= base["tokens_per_s"] * (1 - max_slowdown_pct / 100)]
    best = min(ok, key=lambda r: r["j_per_token"])
    return {"limit_pct": best["limit_pct"], "base_limit_pct": base["limit_pct"],
            "energy_saved_pct": (1 - best["j_per_token"] / base["j_per_token"]) * 100,
            "slowdown_pct": (1 - best["tokens_per_s"] / base["tokens_per_s"]) * 100}
