"""Preflight checks for Colab Pro+ before model loading.

The checks are conservative: they can choose a candidate profile from hardware
capacity, but real benchmark phases still require backend capability gates and
smoke tests before Core is allowed.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .utils import RESULTS_DIR, public_env_snapshot, utc_now_iso, write_json


def run_preflight(profile: str = "auto") -> dict[str, Any]:
    """Collect environment facts and choose a provisional action/profile."""

    gpu = nvidia_smi_summary()
    disk = shutil.disk_usage(Path.cwd())
    packages = package_versions()
    selected_profile, action, status_reasons = choose_profile(
        total_vram_gib=gpu.get("total_vram_gib"),
        free_disk_gib=disk.free / (1024**3),
        requested_profile=profile,
    )
    result = {
        "timestamp": utc_now_iso(),
        "requested_profile": profile,
        "selected_profile": selected_profile,
        "action": action,
        "status_reasons": status_reasons,
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "gpu": gpu,
            "disk_free_gib": round(disk.free / (1024**3), 2),
            "disk_total_gib": round(disk.total / (1024**3), 2),
        },
        "packages": packages,
        "secrets": public_env_snapshot(),
        "backend_capability_matrix": {
            "vllm": "not_checked_in_local_preflight",
            "llama_cpp": "not_checked_in_local_preflight",
            "mtp": "requires_colab_backend_gate",
        },
    }
    write_json(RESULTS_DIR / "preflight.json", result)
    write_environment_txt(RESULTS_DIR / "environment.txt", result)
    return result


def nvidia_smi_summary() -> dict[str, Any]:
    """Read basic GPU facts from `nvidia-smi` when available."""

    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False}

    line = proc.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return {"available": False, "raw": proc.stdout}
    total_mib = float(parts[1])
    free_mib = float(parts[2])
    return {
        "available": True,
        "name": parts[0],
        "total_vram_gib": round(total_mib / 1024, 2),
        "free_vram_gib": round(free_mib / 1024, 2),
        "compute_capability": parts[3],
    }


def package_versions() -> dict[str, Any]:
    """Collect optional package versions without importing heavy modules blindly."""

    versions: dict[str, Any] = {}
    for module_name in ("torch", "vllm"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    versions["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return versions


def choose_profile(
    total_vram_gib: float | None,
    free_disk_gib: float,
    requested_profile: str = "auto",
) -> tuple[str | None, str, list[str]]:
    """Choose the highest feasible profile from resource thresholds."""

    reasons: list[str] = []
    if total_vram_gib is None:
        return (None, "STOP", ["no_gpu_detected"])
    if free_disk_gib < 55:
        return (None, "STOP", ["free_disk_below_55_gib"])

    if requested_profile != "auto":
        return (requested_profile, "GO", ["profile_forced_by_user"])
    if total_vram_gib >= 48:
        return ("q6_core_native", "GO", ["q8_calibration_possible_after_core"])
    if total_vram_gib >= 40:
        return ("q6_core_native", "GO", reasons)
    if total_vram_gib >= 32:
        return ("q5_core_native", "REDUCE_SCOPE", ["q5_constrained_profile"])
    if total_vram_gib >= 24:
        return ("q4_pilot", "REDUCE_SCOPE", ["pilot_only"])
    return (None, "STOP", ["vram_below_24_gib"])


def write_environment_txt(path: Path, result: dict[str, Any]) -> None:
    """Write a human-readable environment summary next to JSON preflight."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"timestamp: {result['timestamp']}\n")
        fh.write(f"action: {result['action']}\n")
        fh.write(f"selected_profile: {result['selected_profile']}\n")
        fh.write(f"hardware: {result['hardware']}\n")
        fh.write(f"packages: {result['packages']}\n")
