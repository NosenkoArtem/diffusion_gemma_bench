"""Package benchmark outputs into versioned run directories.

The repository keeps code and result archives separate. A run directory contains
small, reviewable artifacts that can be committed to a `bench-results` branch,
while heavy raw logs, model weights, and caches stay outside GitHub.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
from pathlib import Path
from typing import Any

from .utils import PROJECT_ROOT, RESULTS_DIR, REPORTS_DIR, ensure_dir, git_revision, utc_now_iso, write_json


DENY_PATTERNS = (
    "*.gguf",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.log",
    "*server_stdout*",
    "*server_stderr*",
)

ALLOWED_ROOT_RESULTS = (
    "preflight.json",
    "environment.txt",
    "run_manifest.json",
    "model_artifacts.json",
    "summary_metrics.json",
    "smoke_status.json",
    "pilot_mtp_tuning.json",
    "pilot_resource_summary.json",
    "pilot_go_no_go.md",
    "pilot_estimated_budget.json",
    "bfcl_summary.json",
    "BLOCKED_HARDWARE.json",
    "BLOCKED_BACKEND.json",
)

ALLOWED_REPORTS = (
    "final_report.md",
    "final_report.html",
)

MAX_COMMIT_FILE_BYTES = 5 * 1024 * 1024
SECRET_RE = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")


def make_run_id(profile: str, phase: str, gpu_name: str | None = None, commit_sha: str | None = None) -> str:
    """Create a readable run id for result archive directories."""

    stamp = utc_now_iso().replace("-", "").replace(":", "").split("+")[0]
    safe_gpu = _safe_slug(gpu_name or "unknown_gpu")
    short_sha = (commit_sha or "nogit")[:8]
    return f"{stamp}_{profile}_{phase}_{safe_gpu}_{short_sha}"


def package_results(
    run_id: str,
    profile: str,
    phase: str,
    source_root: Path = PROJECT_ROOT,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Copy small benchmark artifacts into `results/runs/<run_id>`.

    The function returns a manifest with copied/skipped files. It never copies
    model weights, zip bundles, server logs, or files over the size limit.
    """

    run_dir = ensure_dir(results_dir / "runs" / run_id)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "profile": profile,
        "phase": phase,
        "packaged_at": utc_now_iso(),
        "git": git_revision(source_root),
        "copied_files": [],
        "skipped_files": [],
        "warnings": [],
    }

    for name in ALLOWED_ROOT_RESULTS:
        _copy_if_allowed(results_dir / name, run_dir / name, manifest)
    for name in ALLOWED_REPORTS:
        _copy_if_allowed(reports_dir / name, run_dir / name, manifest)

    write_json(run_dir / "result_manifest.json", manifest)
    return manifest


def validate_result_tree(run_dir: Path) -> dict[str, Any]:
    """Check a result directory before committing it to GitHub."""

    report: dict[str, Any] = {"run_dir": str(run_dir), "ok": True, "errors": [], "warnings": []}
    if not run_dir.exists():
        report["ok"] = False
        report["errors"].append("run_dir_missing")
        return report

    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir)
        reason = deny_reason(path)
        if reason:
            report["ok"] = False
            report["errors"].append(f"{rel}: {reason}")
            continue
        if path.stat().st_size > MAX_COMMIT_FILE_BYTES:
            report["ok"] = False
            report["errors"].append(f"{rel}: file_too_large")
            continue
        if _looks_text(path) and SECRET_RE.search(path.read_text(encoding="utf-8", errors="ignore")):
            report["ok"] = False
            report["errors"].append(f"{rel}: possible_secret")

    return report


def deny_reason(path: Path) -> str | None:
    """Return a reason if a file should never enter the GitHub result branch."""

    lower_name = path.name.lower()
    for pattern in DENY_PATTERNS:
        if fnmatch.fnmatch(lower_name, pattern.lower()):
            return "denied_pattern"
    return None


def _copy_if_allowed(src: Path, dst: Path, manifest: dict[str, Any]) -> None:
    if not src.exists():
        manifest["skipped_files"].append({"path": str(src), "reason": "missing"})
        return
    reason = deny_reason(src)
    if reason:
        manifest["skipped_files"].append({"path": str(src), "reason": reason})
        return
    if src.stat().st_size > MAX_COMMIT_FILE_BYTES:
        manifest["skipped_files"].append({"path": str(src), "reason": "file_too_large"})
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    manifest["copied_files"].append(str(dst))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:80] or "unknown"


def _looks_text(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".html", ".csv", ".yaml", ".yml"}
