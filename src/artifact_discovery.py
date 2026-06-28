"""Hugging Face artifact discovery for configured benchmark models.

Experiment 5 resolves the remaining blocker after vLLM setup: configured model
repository ids or expected filenames may be stale. This phase searches Hugging
Face metadata and records candidate repos/files without downloading weights.
"""

from __future__ import annotations

import os
from typing import Any

from .experiment_summary import write_experiment_summary
from .model_gate import expected_profile_filename, module_version, select_candidate_files
from .utils import RESULTS_DIR, git_revision, load_yaml, project_path, utc_now_iso, write_json


SEARCH_LIMIT_PER_QUERY = 8
MAX_INSPECTED_REPOS_PER_MODEL = 12


DISCOVERY_QUERIES = {
    "DG-Native": ("diffusiongemma", "diffusiongemma 26B", "diffusion-gemma"),
    "G26-AR": ("gemma-4-26B-A4B", "gemma 4 26B A4B", "gemma-4-26b"),
    "G26-MTP": ("gemma-4-26B-A4B assistant", "gemma 4 26B A4B assistant", "gemma mtp assistant"),
}


def run_artifact_discovery(profile: str = "auto", *, enable_search: bool = True) -> dict[str, Any]:
    """Search HF metadata for repo/file candidates and persist artifacts."""

    models_config = load_yaml(project_path("configs", "models.yaml")).get("models", {})
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    hf_hub_available = module_version("huggingface_hub") is not None
    model_results = []
    for model_id, cfg in models_config.items():
        model_results.append(discover_model_artifacts(model_id, cfg, profile, hf_token_present, hf_hub_available, enable_search))

    reasons = blocking_reasons(model_results, hf_token_present, hf_hub_available, enable_search)
    status = "ARTIFACT_DISCOVERY_PASSED" if not reasons else "ARTIFACT_DISCOVERY_NEEDS_REVIEW"
    result = {
        "timestamp": utc_now_iso(),
        "phase": "artifact-discovery",
        "profile": profile,
        "status": status,
        "reasons": reasons,
        "hf_token": {"present": hf_token_present},
        "packages": {"huggingface_hub": module_version("huggingface_hub")},
        "models": model_results,
        "git": git_revision(),
        "next_step": next_step(status, reasons),
    }
    write_json(RESULTS_DIR / "artifact_discovery.json", result)
    write_artifact_discovery_summary(result)
    return result


def discover_model_artifacts(
    model_id: str,
    cfg: dict[str, Any],
    profile: str,
    hf_token_present: bool,
    hf_hub_available: bool,
    enable_search: bool,
) -> dict[str, Any]:
    """Discover candidate repositories/files for one configured model."""

    configured_repo_id = cfg.get("repo_id")
    expected_filename = expected_profile_filename(cfg, profile)
    queries = list(DISCOVERY_QUERIES.get(model_id, (model_id,)))
    if configured_repo_id:
        queries.insert(0, configured_repo_id)
    assistant = cfg.get("assistant") if isinstance(cfg.get("assistant"), dict) else None
    if assistant and assistant.get("repo_id"):
        queries.append(assistant["repo_id"])

    record: dict[str, Any] = {
        "model_id": model_id,
        "role": cfg.get("role"),
        "configured_repo_id": configured_repo_id,
        "expected_filename": expected_filename,
        "queries": unique(queries),
        "search_enabled": enable_search,
        "candidate_repos": [],
        "best_candidate": None,
        "error_type": None,
    }
    if not hf_token_present:
        record["error_type"] = "hf_token_missing"
        return record
    if not hf_hub_available:
        record["error_type"] = "huggingface_hub_not_importable"
        return record
    if not enable_search:
        record["error_type"] = "search_disabled"
        return record

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_ids = search_repo_ids(api, record["queries"])
        candidates = []
        for repo_id in repo_ids[:MAX_INSPECTED_REPOS_PER_MODEL]:
            candidates.append(inspect_repo_files(api, repo_id, expected_filename))
        record["candidate_repos"] = candidates
        record["best_candidate"] = choose_best_candidate(candidates)
    except Exception as exc:
        record["error_type"] = type(exc).__name__
    return record


def search_repo_ids(api: Any, queries: list[str]) -> list[str]:
    """Search HF model ids for multiple query strings."""

    repo_ids: list[str] = []
    for query in queries:
        try:
            models = api.list_models(search=query, limit=SEARCH_LIMIT_PER_QUERY)
            for model in models:
                repo_id = getattr(model, "modelId", None) or getattr(model, "id", None)
                if repo_id:
                    repo_ids.append(repo_id)
        except TypeError:
            models = api.list_models(filter=query, limit=SEARCH_LIMIT_PER_QUERY)
            for model in models:
                repo_id = getattr(model, "modelId", None) or getattr(model, "id", None)
                if repo_id:
                    repo_ids.append(repo_id)
    return unique(repo_ids)


def inspect_repo_files(api: Any, repo_id: str, expected_filename: str | None) -> dict[str, Any]:
    """Inspect one repository's visible files without downloads."""

    try:
        info = api.model_info(repo_id)
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        candidate_files = select_candidate_files(files)
        return {
            "repo_id": repo_id,
            "repo_access_ok": True,
            "sha": getattr(info, "sha", None),
            "gated": getattr(info, "gated", None),
            "private": getattr(info, "private", None),
            "visible_file_count": len(files),
            "expected_file_visible": expected_filename in files if expected_filename else None,
            "candidate_files": candidate_files,
            "weight_file_count": sum(1 for name in files if name.endswith((".gguf", ".safetensors", ".bin"))),
        }
    except Exception as exc:
        return {"repo_id": repo_id, "repo_access_ok": False, "error_type": type(exc).__name__}


def choose_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the most useful candidate for operator review."""

    accessible = [item for item in candidates if item.get("repo_access_ok")]
    if not accessible:
        return None
    exact = [item for item in accessible if item.get("expected_file_visible")]
    if exact:
        return exact[0]
    with_weights = [item for item in accessible if item.get("weight_file_count", 0) > 0]
    if with_weights:
        return sorted(with_weights, key=lambda item: item.get("weight_file_count", 0), reverse=True)[0]
    return accessible[0]


def blocking_reasons(models: list[dict[str, Any]], hf_token_present: bool, hf_hub_available: bool, enable_search: bool) -> list[str]:
    """Return review blockers from discovery output."""

    reasons: list[str] = []
    if not hf_token_present:
        reasons.append("hf_token_missing")
    if not hf_hub_available:
        reasons.append("huggingface_hub_not_importable")
    if not enable_search:
        reasons.append("search_disabled")
    if any(model.get("error_type") for model in models):
        reasons.append("model_search_failed")
    if any(not model.get("best_candidate") for model in models):
        reasons.append("candidate_repo_missing")
    if any(model.get("best_candidate") and not model["best_candidate"].get("expected_file_visible") for model in models):
        reasons.append("expected_filename_not_confirmed")
    return unique(reasons)


def write_artifact_discovery_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Write Markdown/JSON summary for Experiment 5."""

    metrics = [
        {"metric": "discovery_status", "value": result["status"], "status": result["status"], "note": ", ".join(result["reasons"])},
        {"metric": "hf_token_present", "value": result["hf_token"]["present"], "status": "ok" if result["hf_token"]["present"] else "blocked", "note": ""},
        {"metric": "huggingface_hub_version", "value": result["packages"].get("huggingface_hub"), "status": "ok" if result["packages"].get("huggingface_hub") else "blocked", "note": ""},
    ]
    for model in result["models"]:
        best = model.get("best_candidate") or {}
        metrics.append(
            {
                "metric": f"{model['model_id']}_best_repo",
                "value": best.get("repo_id"),
                "status": "ok" if best else "missing",
                "note": f"expected_file_visible={best.get('expected_file_visible')}",
            }
        )
        metrics.append(
            {
                "metric": f"{model['model_id']}_candidate_count",
                "value": len(model.get("candidate_repos", [])),
                "status": "info",
                "note": model.get("error_type") or "",
            }
        )
    criteria = [
        {"criterion": "HF token is loaded", "passed": "hf_token_missing" not in result["reasons"], "note": ""},
        {"criterion": "Hugging Face search can run", "passed": "model_search_failed" not in result["reasons"], "note": ""},
        {"criterion": "Each configured model has at least one accessible candidate", "passed": "candidate_repo_missing" not in result["reasons"], "note": ""},
        {"criterion": "Expected filenames are confirmed or require explicit config update", "passed": "expected_filename_not_confirmed" not in result["reasons"], "note": ""},
    ]
    return write_experiment_summary(
        phase="artifact-discovery",
        profile=result["profile"],
        title="Experiment 5: Model Artifact Discovery",
        objective="Find correct Hugging Face repositories and visible weight files for the configured models without downloading large artifacts.",
        tasks=[
            "Search Hugging Face metadata for DiffusionGemma/Gemma candidate repositories.",
            "Inspect visible files for candidate repos without downloading weights.",
            "Select best repo/file candidates for DG-Native, G26-AR, and G26-MTP.",
            "Produce a reviewable recommendation before editing model config.",
        ],
        metrics=metrics,
        criteria=criteria,
        conclusion=result["next_step"],
        artifacts=["results/artifact_discovery.json", "reports/experiment_summary_artifact-discovery.md", "reports/experiment_summary_artifact-discovery.json"],
    )


def next_step(status: str, reasons: list[str]) -> str:
    """Return operator hint for the next experiment."""

    if status == "ARTIFACT_DISCOVERY_PASSED":
        return "Artifact discovery passed: update configs/models.yaml with confirmed repo ids/files, rerun model-gate, then attempt minimal model-load smoke."
    if "candidate_repo_missing" in reasons:
        return "Review Hugging Face search queries and model naming; no accessible candidate repo was found for at least one model."
    if "expected_filename_not_confirmed" in reasons:
        return "Review candidate files and update expected filenames before model downloads."
    if "hf_token_missing" in reasons:
        return "Load HF_TOKEN/HUGGING_FACE_HUB_TOKEN and rerun artifact-discovery."
    return "Review discovery blockers, adjust model config/search queries, and rerun artifact-discovery."


def unique(items: list[str] | tuple[str, ...]) -> list[str]:
    """Return unique strings preserving order."""

    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
