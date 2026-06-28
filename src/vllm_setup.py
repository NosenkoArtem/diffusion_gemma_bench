"""vLLM setup gate for Colab.

The notebook performs the optional `pip install`; this phase records whether
the runtime is usable afterwards. Keeping installation and verification
separate makes failures easier to diagnose and report.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import Any

from .experiment_summary import write_experiment_summary
from .preflight import nvidia_smi_summary, package_versions
from .utils import RESULTS_DIR, git_revision, utc_now_iso, write_json


def run_vllm_setup(profile: str = "auto") -> dict[str, Any]:
    """Check vLLM installation status and persist a setup report."""

    gpu = nvidia_smi_summary()
    packages = package_versions() | {
        "transformers": module_version("transformers"),
        "huggingface_hub": module_version("huggingface_hub"),
        "requests": module_version("requests"),
    }
    import_status = import_module_status("vllm")
    reasons = blocking_reasons(gpu, packages, import_status)
    status = "VLLM_SETUP_PASSED" if not reasons else "VLLM_SETUP_NEEDS_SETUP"
    result = {
        "timestamp": utc_now_iso(),
        "phase": "vllm-setup",
        "profile": profile,
        "status": status,
        "reasons": reasons,
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": packages,
        "vllm_import": import_status,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": git_revision(),
        "next_step": next_step(status, reasons),
    }
    write_json(RESULTS_DIR / "vllm_setup.json", result)
    write_vllm_setup_summary(result)
    return result


def import_module_status(module_name: str) -> dict[str, Any]:
    """Import a module and return a JSON-safe status object."""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def module_version(module_name: str) -> str | None:
    """Return module version without failing the phase."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def blocking_reasons(gpu: dict[str, Any], packages: dict[str, Any], import_status: dict[str, Any]) -> list[str]:
    """Derive blockers for moving to model-gate after setup."""

    reasons: list[str] = []
    if not gpu.get("available"):
        reasons.append("no_gpu_detected")
    if packages.get("torch") is None:
        reasons.append("torch_not_importable")
    if not import_status.get("ok"):
        reasons.append("vllm_not_importable")
    if packages.get("huggingface_hub") is None:
        reasons.append("huggingface_hub_not_importable")
    return reasons


def write_vllm_setup_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Write the per-experiment Markdown/JSON summary for Experiment 4."""

    metrics = [
        {"metric": "setup_status", "value": result["status"], "status": result["status"], "note": ", ".join(result["reasons"])},
        {"metric": "gpu_available", "value": result["gpu"].get("available"), "status": "ok" if result["gpu"].get("available") else "blocked", "note": result["gpu"].get("name")},
        {"metric": "gpu_total_vram_gib", "value": result["gpu"].get("total_vram_gib"), "status": "info", "note": ""},
        {"metric": "torch_version", "value": result["packages"].get("torch"), "status": "ok" if result["packages"].get("torch") else "blocked", "note": ""},
        {"metric": "vllm_importable", "value": result["vllm_import"].get("ok"), "status": "ok" if result["vllm_import"].get("ok") else "blocked", "note": result["vllm_import"].get("error_type") or result["vllm_import"].get("version")},
        {"metric": "transformers_version", "value": result["packages"].get("transformers"), "status": "info", "note": ""},
        {"metric": "huggingface_hub_version", "value": result["packages"].get("huggingface_hub"), "status": "ok" if result["packages"].get("huggingface_hub") else "blocked", "note": ""},
    ]
    criteria = [
        {"criterion": "GPU is visible after installation", "passed": "no_gpu_detected" not in result["reasons"], "note": result["gpu"]},
        {"criterion": "torch is importable", "passed": "torch_not_importable" not in result["reasons"], "note": result["packages"].get("torch")},
        {"criterion": "vLLM is importable", "passed": "vllm_not_importable" not in result["reasons"], "note": result["vllm_import"]},
        {"criterion": "huggingface_hub is importable", "passed": "huggingface_hub_not_importable" not in result["reasons"], "note": result["packages"].get("huggingface_hub")},
    ]
    return write_experiment_summary(
        phase="vllm-setup",
        profile=result["profile"],
        title="Experiment 4: vLLM Backend Setup Gate",
        objective="Verify that the Colab runtime can import vLLM and keep GPU/CUDA/Hugging Face dependencies usable before model loading.",
        tasks=[
            "Optionally install vLLM from the notebook with an explicit flag.",
            "Capture Python, platform, GPU, and package versions after installation.",
            "Import vLLM and record any error type/message without hiding setup failures.",
            "Decide whether to rerun model-gate or repair the runtime first.",
        ],
        metrics=metrics,
        criteria=criteria,
        conclusion=result["next_step"],
        artifacts=["results/vllm_setup.json", "reports/experiment_summary.md", "reports/experiment_summary.json"],
    )


def next_step(status: str, reasons: list[str]) -> str:
    """Return the operator hint for the notebook."""

    if status == "VLLM_SETUP_PASSED":
        return "vLLM setup gate passed: rerun model-gate and inspect remaining repository/artifact blockers."
    if "vllm_not_importable" in reasons:
        return "vLLM is still not importable. Inspect pip output, restart the Colab runtime if packages changed, then rerun vllm-setup."
    if "no_gpu_detected" in reasons:
        return "Reconnect Colab with a GPU runtime before model-gate."
    return "Resolve listed setup blockers, then rerun vllm-setup."
