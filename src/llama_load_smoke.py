"""llama.cpp compatibility smoke for the confirmed GGUF artifacts.

This phase is the corrected Experiment 6 path after vLLM proved unable to load
the Gemma 4 QAT GGUF metadata. It keeps the test deliberately small: resolve
the configured artifacts, optionally download them through the Hugging Face
cache, then run the same `llama-cli` command shape for Gemma 4 and
DiffusionGemma.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .experiment_summary import write_experiment_summary
from .model_gate import expected_profile_filename, module_version
from .model_load_smoke import disk_snapshot, gpu_snapshot, memory_snapshot, safe_error, unique
from .utils import RESULTS_DIR, git_revision, load_yaml, project_path, utc_now_iso, write_json


DEFAULT_TARGETS = ("G26-AR", "DG-Native")
DEFAULT_PROMPT = "Return exactly one short sentence about a benchmark smoke test."


def run_llama_load_smoke(
    profile: str = "q6_core_native",
    *,
    targets: tuple[str, ...] | list[str] = DEFAULT_TARGETS,
    download_enabled: bool = True,
    load_enabled: bool = True,
    llama_cli_path: str | None = None,
    max_context: int = 512,
    predict_tokens: int = 8,
    timeout_s: int = 300,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a minimal llama.cpp load/generation smoke for both target models."""

    models_config = load_yaml(project_path("configs", "models.yaml")).get("models", {})
    target_ids = [target for target in targets if target]
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    resolved_cli = resolve_llama_cli(llama_cli_path)
    settings = {
        "download_enabled": download_enabled,
        "load_enabled": load_enabled,
        "llama_cli_path": resolved_cli,
        "max_context": max_context,
        "predict_tokens": predict_tokens,
        "timeout_s": timeout_s,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "prompt": DEFAULT_PROMPT,
    }
    model_results = [
        smoke_one_model(
            model_id,
            models_config.get(model_id, {}),
            profile=profile,
            hf_token_present=hf_token_present,
            llama_cli_path=resolved_cli,
            settings=settings,
        )
        for model_id in target_ids
    ]

    reasons = blocking_reasons(model_results, hf_token_present, resolved_cli, download_enabled, load_enabled)
    status = "LLAMA_LOAD_SMOKE_PASSED" if not reasons else "LLAMA_LOAD_SMOKE_NEEDS_REVIEW"
    result = {
        "timestamp": utc_now_iso(),
        "phase": "llama-load-smoke",
        "profile": profile,
        "status": status,
        "reasons": reasons,
        "targets": target_ids,
        "settings": settings,
        "hf_token": {"present": hf_token_present},
        "packages": {"huggingface_hub": module_version("huggingface_hub")},
        "tools": {"llama_cli": {"path": resolved_cli, "present": bool(resolved_cli)}},
        "hardware": {"gpu": gpu_snapshot(), "memory": memory_snapshot(), "disk": disk_snapshot()},
        "models": model_results,
        "git": git_revision(),
        "next_step": next_step(status, reasons),
    }
    write_json(results_dir / "llama_load_smoke.json", result)
    write_llama_load_smoke_summary(result, results_dir=results_dir, reports_dir=reports_dir)
    return result


def smoke_one_model(
    model_id: str,
    cfg: dict[str, Any],
    *,
    profile: str,
    hf_token_present: bool,
    llama_cli_path: str | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Resolve, download, and optionally load one configured GGUF model."""

    repo_id = cfg.get("repo_id")
    filename = expected_profile_filename(cfg, profile)
    record: dict[str, Any] = {
        "model_id": model_id,
        "repo_id": repo_id,
        "filename": filename,
        "base_tokenizer": cfg.get("base_tokenizer"),
        "status": "PENDING",
        "started_at": utc_now_iso(),
        "download": {"enabled": settings["download_enabled"], "ok": None},
        "load": {"enabled": settings["load_enabled"], "ok": None},
        "memory_before": memory_snapshot(),
    }
    if not repo_id or not filename:
        record["status"] = "CONFIG_ERROR"
        record["error_type"] = "missing_repo_or_filename"
        return finalize(record)
    if not hf_token_present:
        record["status"] = "BLOCKED"
        record["error_type"] = "hf_token_missing"
        return finalize(record)

    local_path: Path | None = None
    if settings["download_enabled"]:
        local_path = download_artifact(repo_id, filename, record)
        if not local_path:
            return finalize(record)
    else:
        record["download"].update({"ok": False, "skipped_reason": "download_disabled"})

    if settings["load_enabled"]:
        if not llama_cli_path:
            record["status"] = "BLOCKED"
            record["error_type"] = "llama_cli_missing"
            record["load"].update({"ok": False, "skipped_reason": "llama_cli_missing"})
            return finalize(record)
        if not local_path:
            record["status"] = "LOAD_SKIPPED"
            record["load"].update({"ok": False, "skipped_reason": "no_local_artifact"})
            return finalize(record)
        run_llama_cli(llama_cli_path, local_path, settings, record)
    else:
        record["status"] = "DOWNLOAD_PASSED" if record["download"].get("ok") else "DRY_RUN"
        record["load"].update({"ok": False, "skipped_reason": "load_disabled"})

    return finalize(record)


def download_artifact(repo_id: str, filename: str, record: dict[str, Any]) -> Path | None:
    """Download or reuse one model file from the Hugging Face cache."""

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


def run_llama_cli(llama_cli_path: str, model_path: Path, settings: dict[str, Any], record: dict[str, Any]) -> None:
    """Run a tiny llama.cpp generation that forces the model to load."""

    command = [
        llama_cli_path,
        "-m",
        str(model_path),
        "-p",
        settings["prompt"],
        "-n",
        str(settings["predict_tokens"]),
        "-c",
        str(settings["max_context"]),
        "--temp",
        str(settings["temperature"]),
        "--top-p",
        str(settings["top_p"]),
        "--top-k",
        str(settings["top_k"]),
        "--no-display-prompt",
    ]
    started = time.perf_counter()
    record["load"]["command"] = redact_paths(command, model_path)
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=settings["timeout_s"],
        )
    except subprocess.TimeoutExpired as exc:
        record["status"] = "LOAD_FAILED"
        record["load"].update(
            {
                "ok": False,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error_type": "TimeoutExpired",
                "error_category": "llama_cli_timeout",
                "error": safe_error(exc),
            }
        )
        return
    except Exception as exc:
        record["status"] = "LOAD_FAILED"
        record["load"].update(
            {
                "ok": False,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error_category": "llama_cli_execution_error",
                "error": safe_error(exc),
            }
        )
        return

    ok = proc.returncode == 0
    record["status"] = "LOAD_PASSED" if ok else "LOAD_FAILED"
    record["load"].update(
        {
            "ok": ok,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "returncode": proc.returncode,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
            "error_category": None if ok else classify_llama_error(proc.stderr + "\n" + proc.stdout),
        }
    )


def blocking_reasons(
    models: list[dict[str, Any]],
    hf_token_present: bool,
    llama_cli_path: str | None,
    download_enabled: bool,
    load_enabled: bool,
) -> list[str]:
    """Return stable blocker codes for the combined 6b/6c gate."""

    reasons: list[str] = []
    if not hf_token_present:
        reasons.append("hf_token_missing")
    if load_enabled and not llama_cli_path:
        reasons.append("llama_cli_missing")
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
    if load_enabled and not all(model.get("status") == "LOAD_PASSED" for model in models):
        reasons.append("not_all_models_loaded")
    return unique(reasons)


def write_llama_load_smoke_summary(
    result: dict[str, Any],
    *,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the human-readable summary for the combined Experiment 6."""

    metrics = [
        {"metric": "status", "value": result["status"], "status": result["status"], "note": ", ".join(result["reasons"])},
        {"metric": "llama_cli_present", "value": result["tools"]["llama_cli"]["present"], "status": "ok" if result["tools"]["llama_cli"]["present"] else "blocked", "note": result["tools"]["llama_cli"]["path"]},
        {"metric": "download_enabled", "value": result["settings"]["download_enabled"], "status": "info", "note": ""},
        {"metric": "load_enabled", "value": result["settings"]["load_enabled"], "status": "info", "note": ""},
        {"metric": "gpu", "value": result["hardware"]["gpu"].get("name"), "status": "info", "note": result["hardware"]["gpu"]},
        {"metric": "disk_free_gib", "value": result["hardware"]["disk"].get("free_gib"), "status": "info", "note": ""},
    ]
    for model in result["models"]:
        metrics.extend(
            [
                {"metric": f"{model['model_id']}_status", "value": model["status"], "status": "ok" if model["status"] == "LOAD_PASSED" else "review", "note": ""},
                {"metric": f"{model['model_id']}_download_s", "value": model["download"].get("elapsed_s"), "status": "info", "note": model["download"].get("error") or ""},
                {"metric": f"{model['model_id']}_load_s", "value": model["load"].get("elapsed_s"), "status": "info", "note": model["load"].get("error_category") or model["load"].get("error") or ""},
                {"metric": f"{model['model_id']}_artifact_bytes", "value": model["download"].get("size_bytes"), "status": "info", "note": model.get("filename")},
            ]
        )
    criteria = [
        {"criterion": "HF token is loaded", "passed": "hf_token_missing" not in result["reasons"], "note": ""},
        {"criterion": "llama.cpp CLI is available", "passed": "llama_cli_missing" not in result["reasons"], "note": ""},
        {"criterion": "Both configured GGUF artifacts download or are cached", "passed": "download_failed" not in result["reasons"] and "download_disabled" not in result["reasons"], "note": ""},
        {"criterion": "Both target models load through the same llama.cpp command shape", "passed": result["status"] == "LLAMA_LOAD_SMOKE_PASSED", "note": ""},
    ]
    return write_experiment_summary(
        phase="llama-load-smoke",
        profile=result["profile"],
        title="Experiment 6: Combined llama.cpp Model Load Smoke",
        objective="Verify that Gemma 4 and DiffusionGemma GGUF artifacts can both be downloaded/cached and loaded through the same llama.cpp backend before generation benchmarks.",
        tasks=[
            "Resolve configured repo ids and GGUF filenames for G26-AR and DG-Native.",
            "Download or reuse both artifacts through the Hugging Face cache.",
            "Run one tiny llama.cpp generation per model with identical sampling and context settings.",
            "Record tool availability, return codes, load latency, artifact sizes, hardware snapshot, and failure tails.",
        ],
        metrics=metrics,
        criteria=criteria,
        conclusion=result["next_step"],
        artifacts=["results/llama_load_smoke.json", "reports/experiment_summary_llama-load-smoke.md", "reports/experiment_summary_llama-load-smoke.json"],
        results_dir=results_dir,
        reports_dir=reports_dir or project_path("reports"),
    )


def next_step(status: str, reasons: list[str]) -> str:
    """Return the next operator action after the combined Experiment 6."""

    if status == "LLAMA_LOAD_SMOKE_PASSED":
        return "Both GGUF artifacts load through llama.cpp. Next: run minimal OpenAI-compatible llama-server generation smoke."
    if "llama_cli_missing" in reasons:
        return "Install or build llama.cpp with CUDA in Colab, then rerun llama-load-smoke without changing model settings."
    if "download_failed" in reasons:
        return "Inspect Hugging Face token access, repo/file names, disk space, and cache path before retrying."
    if "load_failed" in reasons:
        return "Inspect llama.cpp stdout/stderr tails. Keep the same backend for both models before moving to benchmarks."
    if "download_disabled" in reasons or "load_disabled" in reasons:
        return "This was a dry run. Enable download/load in Colab when ready to consume disk, network, and VRAM."
    return "Resolve listed blockers, then rerun llama-load-smoke."


def resolve_llama_cli(explicit_path: str | None = None) -> str | None:
    """Find the llama.cpp CLI executable from an explicit path or PATH."""

    candidates = [explicit_path] if explicit_path else []
    candidates.extend(("llama-cli", "main"))
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def classify_llama_error(text: str) -> str:
    """Map common llama.cpp failures to stable report categories."""

    lowered = text.lower()
    if "out of memory" in lowered or "cuda error" in lowered:
        return "gpu_or_cuda_error"
    if "unknown model" in lowered or "unsupported" in lowered:
        return "unsupported_model_or_gguf"
    if "no such file" in lowered or "failed to open" in lowered:
        return "artifact_open_failed"
    return "llama_cli_nonzero_exit"


def tail(text: str, max_lines: int = 20) -> list[str]:
    """Return a compact non-empty tail for notebook/report diagnostics."""

    return [line[-500:] for line in text.splitlines() if line.strip()][-max_lines:]


def redact_paths(command: list[str], model_path: Path) -> list[str]:
    """Keep commands readable without storing long HF cache paths."""

    return ["<model.gguf>" if item == str(model_path) else item for item in command]


def finalize(record: dict[str, Any]) -> dict[str, Any]:
    """Attach final timestamps and memory metrics to a model record."""

    record["finished_at"] = utc_now_iso()
    record["memory_after"] = memory_snapshot()
    return record
