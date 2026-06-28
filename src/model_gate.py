"""Model access and backend feasibility gate.

This phase checks whether the current Colab runtime is ready for model smoke
tests without downloading 26B weights. It records model repository visibility,
expected artifact visibility for the selected profile, runtime resources, and
backend package readiness.
"""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path
from typing import Any

from .experiment_summary import write_experiment_summary
from .preflight import nvidia_smi_summary, package_versions
from .utils import PROJECT_ROOT, RESULTS_DIR, git_revision, load_yaml, project_path, utc_now_iso, write_json


MIN_DISK_FREE_GIB = 55
MIN_MODEL_SMOKE_VRAM_GIB = 24


def run_model_gate(profile: str = "auto") -> dict[str, Any]:
    """Run non-download model readiness checks and persist JSON/Markdown output."""

    models_config = load_yaml(project_path("configs", "models.yaml")).get("models", {})
    gpu = nvidia_smi_summary()
    disk = shutil.disk_usage(PROJECT_ROOT)
    disk_free_gib = round(disk.free / (1024**3), 2)
    packages = package_versions() | {
        "huggingface_hub": module_version("huggingface_hub"),
        "requests": module_version("requests"),
    }
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    models = [inspect_model(model_id, cfg, profile, hf_token_present) for model_id, cfg in models_config.items()]

    reasons = blocking_reasons(models=models, gpu=gpu, disk_free_gib=disk_free_gib, packages=packages, hf_token_present=hf_token_present)
    status = "MODEL_GATE_PASSED" if not reasons else "MODEL_GATE_NEEDS_SETUP"
    result = {
        "timestamp": utc_now_iso(),
        "phase": "model-gate",
        "profile": profile,
        "status": status,
        "reasons": reasons,
        "gpu": gpu,
        "disk": {"free_gib": disk_free_gib, "min_required_gib": MIN_DISK_FREE_GIB},
        "packages": packages,
        "hf_token": {"present": hf_token_present},
        "models": models,
        "git": git_revision(),
        "next_step": next_step(status, reasons),
    }
    write_json(RESULTS_DIR / "model_gate.json", result)
    write_model_gate_summary(result)
    return result


def inspect_model(model_id: str, cfg: dict[str, Any], profile: str, hf_token_present: bool) -> dict[str, Any]:
    """Inspect one configured model without downloading model artifacts."""

    repo_id = cfg.get("repo_id")
    expected_filename = expected_profile_filename(cfg, profile)
    record: dict[str, Any] = {
        "model_id": model_id,
        "role": cfg.get("role"),
        "repo_id": repo_id,
        "expected_filename": expected_filename,
        "repo_access_ok": None,
        "expected_file_visible": None,
        "visible_file_count": None,
        "error_type": None,
    }

    if not hf_token_present:
        record["error_type"] = "hf_token_missing"
        return record
    if module_version("huggingface_hub") is None:
        record["error_type"] = "huggingface_hub_not_importable"
        return record
    if not repo_id:
        record["error_type"] = "repo_id_missing"
        return record

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id)
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        record.update(
            {
                "repo_access_ok": True,
                "sha": getattr(info, "sha", None),
                "gated": getattr(info, "gated", None),
                "private": getattr(info, "private", None),
                "visible_file_count": len(files),
                "expected_file_visible": expected_filename in files if expected_filename else None,
                "candidate_files": select_candidate_files(files),
            }
        )
    except Exception as exc:
        record.update({"repo_access_ok": False, "expected_file_visible": False, "error_type": type(exc).__name__})

    assistant = cfg.get("assistant")
    if isinstance(assistant, dict) and assistant.get("repo_id"):
        record["assistant"] = inspect_assistant_repo(assistant["repo_id"], hf_token_present)
    return record


def inspect_assistant_repo(repo_id: str, hf_token_present: bool) -> dict[str, Any]:
    """Inspect the MTP assistant repository metadata without downloads."""

    if not hf_token_present:
        return {"repo_id": repo_id, "repo_access_ok": None, "error_type": "hf_token_missing"}
    if module_version("huggingface_hub") is None:
        return {"repo_id": repo_id, "repo_access_ok": None, "error_type": "huggingface_hub_not_importable"}
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id)
        return {"repo_id": repo_id, "repo_access_ok": True, "sha": getattr(info, "sha", None), "gated": getattr(info, "gated", None)}
    except Exception as exc:
        return {"repo_id": repo_id, "repo_access_ok": False, "error_type": type(exc).__name__}


def expected_profile_filename(cfg: dict[str, Any], profile: str) -> str | None:
    """Return the expected weight filename for the selected quant/profile."""

    filenames = cfg.get("filenames")
    if isinstance(filenames, dict):
        return filenames.get(profile)
    target_ref = cfg.get("target_model_ref")
    if target_ref:
        models = load_yaml(project_path("configs", "models.yaml")).get("models", {})
        target = models.get(target_ref, {})
        target_files = target.get("filenames", {})
        if isinstance(target_files, dict):
            return target_files.get(profile)
    return None


def select_candidate_files(files: list[str]) -> list[str]:
    """Keep the model metadata readable by returning only likely weight/config files."""

    suffixes = (".gguf", ".safetensors", "config.json", "tokenizer.json", "tokenizer.model")
    return [name for name in files if name.endswith(suffixes)][:20]


def blocking_reasons(
    *,
    models: list[dict[str, Any]],
    gpu: dict[str, Any],
    disk_free_gib: float,
    packages: dict[str, Any],
    hf_token_present: bool,
) -> list[str]:
    """Derive explicit setup blockers for the next model-smoke phase."""

    reasons: list[str] = []
    if not gpu.get("available"):
        reasons.append("no_gpu_detected")
    elif (gpu.get("total_vram_gib") or 0) < MIN_MODEL_SMOKE_VRAM_GIB:
        reasons.append("vram_below_model_smoke_floor")
    if disk_free_gib < MIN_DISK_FREE_GIB:
        reasons.append("free_disk_below_55_gib")
    if packages.get("vllm") is None:
        reasons.append("vllm_not_importable")
    if packages.get("huggingface_hub") is None:
        reasons.append("huggingface_hub_not_importable")
    if not hf_token_present:
        reasons.append("hf_token_missing")
    if any(model.get("repo_access_ok") is False for model in models):
        reasons.append("model_repo_access_failed")
    if any(model.get("expected_file_visible") is False for model in models if model.get("expected_filename")):
        reasons.append("expected_model_file_missing")
    if any(model.get("assistant", {}).get("repo_access_ok") is False for model in models):
        reasons.append("assistant_repo_access_failed")
    return reasons


def write_model_gate_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Write a per-experiment summary document for the model gate."""

    metrics = [
        {"metric": "gate_status", "value": result["status"], "status": result["status"], "note": ", ".join(result["reasons"])},
        {"metric": "gpu_available", "value": result["gpu"].get("available"), "status": "ok" if result["gpu"].get("available") else "blocked", "note": result["gpu"].get("name")},
        {"metric": "gpu_total_vram_gib", "value": result["gpu"].get("total_vram_gib"), "status": "info", "note": "model smoke floor is 24 GiB"},
        {"metric": "disk_free_gib", "value": result["disk"]["free_gib"], "status": "ok" if result["disk"]["free_gib"] >= MIN_DISK_FREE_GIB else "blocked", "note": "minimum 55 GiB"},
        {"metric": "vllm_importable", "value": result["packages"].get("vllm") is not None, "status": "ok" if result["packages"].get("vllm") else "blocked", "note": result["packages"].get("vllm")},
        {"metric": "hf_token_present", "value": result["hf_token"]["present"], "status": "ok" if result["hf_token"]["present"] else "blocked", "note": ""},
    ]
    for model in result["models"]:
        metrics.append(
            {
                "metric": f"{model['model_id']}_repo_access",
                "value": model.get("repo_access_ok"),
                "status": "ok" if model.get("repo_access_ok") else "blocked",
                "note": model.get("repo_id"),
            }
        )
        metrics.append(
            {
                "metric": f"{model['model_id']}_expected_file",
                "value": model.get("expected_file_visible"),
                "status": "ok" if model.get("expected_file_visible") else "unknown_or_blocked",
                "note": model.get("expected_filename"),
            }
        )

    criteria = [
        {"criterion": "GPU is visible and has at least 24 GiB VRAM", "passed": "no_gpu_detected" not in result["reasons"] and "vram_below_model_smoke_floor" not in result["reasons"], "note": result["gpu"]},
        {"criterion": "At least 55 GiB disk is free", "passed": "free_disk_below_55_gib" not in result["reasons"], "note": result["disk"]},
        {"criterion": "vLLM is importable", "passed": "vllm_not_importable" not in result["reasons"], "note": result["packages"].get("vllm")},
        {"criterion": "HF token and model repo metadata are accessible", "passed": "hf_token_missing" not in result["reasons"] and "model_repo_access_failed" not in result["reasons"], "note": ""},
        {"criterion": "Expected profile artifacts are visible", "passed": "expected_model_file_missing" not in result["reasons"], "note": ""},
    ]
    conclusion = result["next_step"]
    return write_experiment_summary(
        phase="model-gate",
        profile=result["profile"],
        title="Experiment 3: Model Access and Backend Feasibility Gate",
        objective="Check whether the current Colab Pro+ runtime can move from harness smoke tests to real model smoke tests without downloading 26B weights.",
        tasks=[
            "Capture runtime GPU, disk, package, and git metadata.",
            "Check Hugging Face token presence and model repository metadata access.",
            "Check expected profile artifact visibility for DG-Native, G26-AR, and G26-MTP.",
            "Check vLLM importability as the primary backend gate.",
            "Record blockers and next-step decision for the model-smoke phase.",
        ],
        metrics=metrics,
        criteria=criteria,
        conclusion=conclusion,
        artifacts=["results/model_gate.json", "reports/experiment_summary.md", "reports/experiment_summary.json"],
    )


def next_step(status: str, reasons: list[str]) -> str:
    """Return the operator decision for the next experiment."""

    if status == "MODEL_GATE_PASSED":
        return "Model gate passed: proceed to a minimal one-model load smoke before broader benchmarks."
    if "vllm_not_importable" in reasons:
        return "Install or repair vLLM in the Colab runtime, then rerun model-gate."
    if "model_repo_access_failed" in reasons or "expected_model_file_missing" in reasons:
        return "Fix model repository ids, gated access, or expected filenames before downloading weights."
    if "hf_token_missing" in reasons:
        return "Load HF_TOKEN/HUGGING_FACE_HUB_TOKEN into the Colab runtime and rerun model-gate."
    return "Resolve listed blockers and rerun model-gate before model smoke."


def module_version(module_name: str) -> str | None:
    """Return importable module version without failing the phase."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")
