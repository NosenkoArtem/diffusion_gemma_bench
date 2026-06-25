"""Backend capability checks before real model smoke tests.

This phase is deliberately lighter than the model smoke test. It checks whether
the Colab runtime has the prerequisites for the next step: GPU, disk, Python
packages, Hugging Face token visibility, model repository access, and a free
localhost port for a vLLM OpenAI-compatible server.
"""

from __future__ import annotations

import importlib
import os
import socket
from typing import Any

from .preflight import nvidia_smi_summary, package_versions
from .utils import RESULTS_DIR, git_revision, utc_now_iso, write_json


MODEL_REPOS = (
    "unsloth/diffusiongemma-26B-A4B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "google/gemma-4-26B-A4B-it-assistant",
)


def run_backend_check(profile: str = "auto", host: str = "127.0.0.1", port: int = 8000) -> dict[str, Any]:
    """Run non-destructive backend readiness checks and persist JSON output."""

    gpu = nvidia_smi_summary()
    packages = package_versions()
    optional_packages = {
        "requests": module_version("requests"),
        "huggingface_hub": module_version("huggingface_hub"),
    }
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    hf_access = check_huggingface_access(MODEL_REPOS) if hf_token_present else {"checked": False, "reason": "hf_token_missing"}
    port_status = check_tcp_port_free(host, port)

    reasons: list[str] = []
    if not gpu.get("available"):
        reasons.append("no_gpu_detected")
    if packages.get("torch") is None:
        reasons.append("torch_not_importable")
    if packages.get("vllm") is None:
        reasons.append("vllm_not_importable")
    if optional_packages.get("huggingface_hub") is None:
        reasons.append("huggingface_hub_not_importable")
    if not hf_token_present:
        reasons.append("hf_token_missing")
    if not port_status["free"]:
        reasons.append("localhost_port_busy")
    if hf_access.get("checked") and not all(item["ok"] for item in hf_access["repositories"]):
        reasons.append("hf_model_access_failed")

    status = "BACKEND_CHECK_PASSED" if not reasons else "BACKEND_CHECK_NEEDS_SETUP"
    result = {
        "timestamp": utc_now_iso(),
        "profile": profile,
        "status": status,
        "reasons": reasons,
        "gpu": gpu,
        "packages": packages | optional_packages,
        "hf_token": {"present": hf_token_present},
        "hf_access": hf_access,
        "localhost": {"host": host, "port": port, "port_free": port_status["free"], "detail": port_status},
        "git": git_revision(),
        "next_step": next_step_from_reasons(reasons),
    }
    write_json(RESULTS_DIR / "backend_capability.json", result)
    return result


def module_version(module_name: str) -> str | None:
    """Return importable module version without failing the whole check."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def check_tcp_port_free(host: str, port: int) -> dict[str, Any]:
    """Return whether a local TCP port can be bound by a future server."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        return {"free": False, "error": str(exc)}
    finally:
        sock.close()
    return {"free": True}


def check_huggingface_access(repo_ids: tuple[str, ...]) -> dict[str, Any]:
    """Check repository metadata access through huggingface_hub without downloads."""

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {"checked": False, "reason": "huggingface_hub_not_importable", "error": str(exc)}

    api = HfApi()
    repositories = []
    for repo_id in repo_ids:
        try:
            info = api.model_info(repo_id)
            repositories.append(
                {
                    "repo_id": repo_id,
                    "ok": True,
                    "sha": getattr(info, "sha", None),
                    "private": getattr(info, "private", None),
                    "gated": getattr(info, "gated", None),
                }
            )
        except Exception as exc:
            repositories.append({"repo_id": repo_id, "ok": False, "error_type": type(exc).__name__})
    return {"checked": True, "repositories": repositories}


def next_step_from_reasons(reasons: list[str]) -> str:
    """Return a concise operator hint for the notebook and report."""

    if not reasons:
        return "Можно переходить к реальному vLLM capability gate и лёгкому model smoke."
    if "vllm_not_importable" in reasons:
        return "Установить/проверить vLLM в Colab runtime перед запуском модельного smoke."
    if "hf_token_missing" in reasons:
        return "Добавить HF_TOKEN/HUGGING_FACE_HUB_TOKEN в Colab secret или переменную окружения."
    if "hf_model_access_failed" in reasons:
        return "Проверить доступ Hugging Face к gated model repositories."
    if "localhost_port_busy" in reasons:
        return "Освободить порт 127.0.0.1:8000 или выбрать другой порт для vLLM."
    return "Исправить перечисленные причины и повторить backend-check."
