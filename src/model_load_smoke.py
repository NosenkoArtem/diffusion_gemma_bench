"""Minimal real model-load smoke for confirmed GGUF artifacts.

Experiment 6 is the first phase that may download large model files and create
a real vLLM engine. It stays deliberately narrow: load one model first, record
resource metrics and errors, then decide whether to attempt the next model.
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from .experiment_summary import write_experiment_summary
from .model_gate import expected_profile_filename, module_version
from .utils import RESULTS_DIR, git_revision, load_yaml, project_path, utc_now_iso, write_json


DEFAULT_TARGETS = ("G26-AR",)
OPTIONAL_SECOND_TARGET = "DG-Native"


def run_model_load_smoke(
    profile: str = "q6_core_native",
    *,
    targets: tuple[str, ...] | list[str] = DEFAULT_TARGETS,
    download_enabled: bool = True,
    load_enabled: bool = True,
    max_model_len: int = 512,
    gpu_memory_utilization: float = 0.82,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Download confirmed artifacts and optionally instantiate a vLLM engine."""

    models_config = load_yaml(project_path("configs", "models.yaml")).get("models", {})
    target_ids = [target for target in targets if target]
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    packages = package_versions()
    model_results = [
        smoke_one_model(
            model_id,
            models_config.get(model_id, {}),
            profile=profile,
            hf_token_present=hf_token_present,
            download_enabled=download_enabled,
            load_enabled=load_enabled,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        for model_id in target_ids
    ]

    reasons = blocking_reasons(model_results, hf_token_present, packages, download_enabled, load_enabled)
    status = "MODEL_LOAD_SMOKE_PASSED" if not reasons else "MODEL_LOAD_SMOKE_NEEDS_REVIEW"
    result = {
        "timestamp": utc_now_iso(),
        "phase": "model-load-smoke",
        "profile": profile,
        "status": status,
        "reasons": reasons,
        "targets": target_ids,
        "settings": {
            "download_enabled": download_enabled,
            "load_enabled": load_enabled,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
        "hf_token": {"present": hf_token_present},
        "packages": packages,
        "hardware": hardware_snapshot(),
        "models": model_results,
        "git": git_revision(),
        "next_step": next_step(status, reasons),
    }
    write_json(results_dir / "model_load_smoke.json", result)
    write_model_load_smoke_summary(result, results_dir=results_dir, reports_dir=reports_dir)
    return result


def smoke_one_model(
    model_id: str,
    cfg: dict[str, Any],
    *,
    profile: str,
    hf_token_present: bool,
    download_enabled: bool,
    load_enabled: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    """Run the download/load smoke for one configured model."""

    repo_id = cfg.get("repo_id")
    filename = expected_profile_filename(cfg, profile)
    record: dict[str, Any] = {
        "model_id": model_id,
        "repo_id": repo_id,
        "filename": filename,
        "base_tokenizer": cfg.get("base_tokenizer"),
        "status": "PENDING",
        "started_at": utc_now_iso(),
        "download": {"enabled": download_enabled, "ok": None},
        "load": {"enabled": load_enabled, "ok": None},
        "memory_before": memory_snapshot(),
    }

    if not repo_id or not filename:
        record["status"] = "CONFIG_ERROR"
        record["error_type"] = "missing_repo_or_filename"
        return finalize_model_record(record)
    if not hf_token_present:
        record["status"] = "BLOCKED"
        record["error_type"] = "hf_token_missing"
        return finalize_model_record(record)
    if module_version("huggingface_hub") is None:
        record["status"] = "BLOCKED"
        record["error_type"] = "huggingface_hub_not_importable"
        return finalize_model_record(record)
    if load_enabled and module_version("vllm") is None:
        record["status"] = "BLOCKED"
        record["error_type"] = "vllm_not_importable"
        return finalize_model_record(record)

    local_path: Path | None = None
    if download_enabled:
        local_path = download_artifact(repo_id, filename, record)
        if not local_path:
            return finalize_model_record(record)
    else:
        record["download"].update({"ok": False, "skipped_reason": "download_disabled"})

    if load_enabled:
        if not local_path:
            record["load"].update({"ok": False, "skipped_reason": "no_local_artifact"})
            record["status"] = "LOAD_SKIPPED"
            return finalize_model_record(record)
        load_vllm_engine(local_path, cfg, max_model_len, gpu_memory_utilization, record)
    else:
        record["load"].update({"ok": False, "skipped_reason": "load_disabled"})
        record["status"] = "DOWNLOAD_PASSED" if record["download"].get("ok") else "DRY_RUN"

    return finalize_model_record(record)


def download_artifact(repo_id: str, filename: str, record: dict[str, Any]) -> Path | None:
    """Download one HF artifact into the normal Hugging Face cache."""

    started = time.perf_counter()
    try:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model"))
        record["download"].update(
            {
                "ok": True,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "local_path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "cache_hit_or_downloaded": True,
            }
        )
        return path
    except Exception as exc:
        record["status"] = "DOWNLOAD_FAILED"
        record["download"].update(
            {
                "ok": False,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": safe_error(exc),
            }
        )
        return None


def load_vllm_engine(
    local_path: Path,
    cfg: dict[str, Any],
    max_model_len: int,
    gpu_memory_utilization: float,
    record: dict[str, Any],
) -> None:
    """Instantiate vLLM with a local GGUF artifact and immediately release it."""

    started = time.perf_counter()
    try:
        from vllm import LLM

        kwargs = {
            "model": str(local_path),
            "tokenizer": cfg.get("base_tokenizer"),
            "load_format": "gguf",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        engine = LLM(**{key: value for key, value in kwargs.items() if value is not None})
        record["load"].update({"ok": True, "elapsed_s": round(time.perf_counter() - started, 3)})
        record["status"] = "LOAD_PASSED"
        del engine
        empty_cuda_cache()
    except Exception as exc:
        record["status"] = "LOAD_FAILED"
        record["load"].update(
            {
                "ok": False,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": safe_error(exc),
                "traceback_tail": traceback.format_exc(limit=6).splitlines()[-12:],
            }
        )
        empty_cuda_cache()


def finalize_model_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach final timestamps and memory metrics to a model smoke record."""

    record["finished_at"] = utc_now_iso()
    record["memory_after"] = memory_snapshot()
    return record


def blocking_reasons(
    models: list[dict[str, Any]],
    hf_token_present: bool,
    packages: dict[str, Any],
    download_enabled: bool,
    load_enabled: bool,
) -> list[str]:
    """Return blockers that should stop the next experiment."""

    reasons: list[str] = []
    if not hf_token_present:
        reasons.append("hf_token_missing")
    if packages.get("huggingface_hub") is None:
        reasons.append("huggingface_hub_not_importable")
    if load_enabled and packages.get("vllm") is None:
        reasons.append("vllm_not_importable")
    if not download_enabled:
        reasons.append("download_disabled")
    if not load_enabled:
        reasons.append("load_disabled")
    if any(model.get("status") == "DOWNLOAD_FAILED" for model in models):
        reasons.append("download_failed")
    if any(model.get("status") == "LOAD_FAILED" for model in models):
        reasons.append("load_failed")
    if any(model.get("status") in {"CONFIG_ERROR", "BLOCKED"} for model in models):
        reasons.append("model_blocked")
    if load_enabled and not any(model.get("status") == "LOAD_PASSED" for model in models):
        reasons.append("no_model_loaded")
    return unique(reasons)


def write_model_load_smoke_summary(
    result: dict[str, Any],
    *,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Write Markdown/JSON summary for Experiment 6."""

    metrics = [
        {"metric": "status", "value": result["status"], "status": result["status"], "note": ", ".join(result["reasons"])},
        {"metric": "download_enabled", "value": result["settings"]["download_enabled"], "status": "info", "note": ""},
        {"metric": "load_enabled", "value": result["settings"]["load_enabled"], "status": "info", "note": ""},
        {"metric": "gpu", "value": result["hardware"]["gpu"].get("name"), "status": "info", "note": result["hardware"]["gpu"]},
        {"metric": "disk_free_gib", "value": result["hardware"]["disk"].get("free_gib"), "status": "info", "note": ""},
    ]
    for model in result["models"]:
        metrics.extend(
            [
                {"metric": f"{model['model_id']}_status", "value": model["status"], "status": "ok" if model["status"] == "LOAD_PASSED" else "review", "note": ""},
                {"metric": f"{model['model_id']}_download_s", "value": model["download"].get("elapsed_s"), "status": "info", "note": model["download"].get("error_type") or ""},
                {"metric": f"{model['model_id']}_load_s", "value": model["load"].get("elapsed_s"), "status": "info", "note": model["load"].get("error_type") or ""},
                {"metric": f"{model['model_id']}_artifact_bytes", "value": model["download"].get("size_bytes"), "status": "info", "note": model.get("filename")},
            ]
        )
    criteria = [
        {"criterion": "HF token is loaded", "passed": "hf_token_missing" not in result["reasons"], "note": ""},
        {"criterion": "Target artifact downloads or is already cached", "passed": "download_failed" not in result["reasons"] and "download_disabled" not in result["reasons"], "note": ""},
        {"criterion": "At least one target model loads in vLLM", "passed": "no_model_loaded" not in result["reasons"] and "load_failed" not in result["reasons"] and "load_disabled" not in result["reasons"], "note": ""},
        {"criterion": "No model-load OOM or backend exception", "passed": "load_failed" not in result["reasons"], "note": ""},
    ]
    return write_experiment_summary(
        phase="model-load-smoke",
        profile=result["profile"],
        title="Experiment 6: Minimal Model Load Smoke",
        objective="Verify that the confirmed GGUF artifact can be downloaded/cached and loaded by vLLM before running generation benchmarks.",
        tasks=[
            "Resolve the configured repo id and GGUF filename for the selected target.",
            "Download the artifact through Hugging Face cache, or reuse it if already cached.",
            "Instantiate a minimal vLLM engine with conservative context length.",
            "Record load time, artifact size, hardware snapshot, and any failure traceback.",
        ],
        metrics=metrics,
        criteria=criteria,
        conclusion=result["next_step"],
        artifacts=["results/model_load_smoke.json", "reports/experiment_summary_model-load-smoke.md", "reports/experiment_summary_model-load-smoke.json"],
        results_dir=results_dir,
        reports_dir=reports_dir or project_path("reports"),
    )


def next_step(status: str, reasons: list[str]) -> str:
    """Return the next operator action after Experiment 6."""

    if status == "MODEL_LOAD_SMOKE_PASSED":
        return "G26-AR model-load smoke passed. Next: optionally run DG-Native load smoke, then minimal generation smoke."
    if "download_disabled" in reasons or "load_disabled" in reasons:
        return "This was a dry run. Enable download/load in Colab when ready to consume disk/VRAM."
    if "download_failed" in reasons:
        return "Inspect Hugging Face download error, token access, disk space, and cache path before retrying."
    if "load_failed" in reasons:
        return "Inspect vLLM load traceback and GPU memory; retry with lower max_model_len or lower gpu_memory_utilization."
    return "Resolve listed blockers, then rerun model-load-smoke."


def package_versions() -> dict[str, Any]:
    """Return dependency versions relevant to model loading."""

    return {
        "torch": module_version("torch"),
        "vllm": module_version("vllm"),
        "huggingface_hub": module_version("huggingface_hub"),
        "psutil": module_version("psutil"),
    }


def hardware_snapshot() -> dict[str, Any]:
    """Return lightweight hardware and disk information."""

    return {"gpu": gpu_snapshot(), "memory": memory_snapshot(), "disk": disk_snapshot()}


def gpu_snapshot() -> dict[str, Any]:
    """Return CUDA device info when torch is importable."""

    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "available": True,
            "index": index,
            "name": props.name,
            "total_memory_gib": round(props.total_memory / (1024**3), 2),
            "allocated_gib": round(torch.cuda.memory_allocated(index) / (1024**3), 3),
            "reserved_gib": round(torch.cuda.memory_reserved(index) / (1024**3), 3),
        }
    except Exception as exc:
        return {"available": None, "error_type": type(exc).__name__, "error": safe_error(exc)}


def memory_snapshot() -> dict[str, Any]:
    """Return process/system memory if psutil is available."""

    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "ram_total_gib": round(vm.total / (1024**3), 2),
            "ram_available_gib": round(vm.available / (1024**3), 2),
            "ram_percent": vm.percent,
        }
    except Exception:
        return {}


def disk_snapshot() -> dict[str, Any]:
    """Return disk capacity for the repository filesystem."""

    usage = shutil.disk_usage(project_path())
    return {
        "total_gib": round(usage.total / (1024**3), 2),
        "free_gib": round(usage.free / (1024**3), 2),
        "used_gib": round(usage.used / (1024**3), 2),
    }


def empty_cuda_cache() -> None:
    """Release cached CUDA memory after a load attempt when torch is present."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def unique(items: list[str] | tuple[str, ...]) -> list[str]:
    """Return unique strings preserving order."""

    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def safe_error(exc: Exception) -> str:
    """Return a short error message without secret material."""

    text = str(exc).replace("\n", " ")
    for marker in ("hf_", "Bearer "):
        if marker in text:
            text = text.split(marker)[0] + "[redacted]"
    return text[:500]
