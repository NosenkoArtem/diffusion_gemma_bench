"""Phase bootstrap helpers shared by CLI and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import RESULTS_DIR, ensure_dir, git_revision, new_run_id, utc_now_iso, write_json


def start_phase(phase: str, profile: str, run_id: str | None = None) -> dict[str, Any]:
    """Create a phase manifest and return it.

    The manifest is intentionally simple so notebook cells can inspect and pass
    it around without hidden global state.
    """

    manifest = {
        "run_id": run_id or new_run_id(phase),
        "phase": phase,
        "profile": profile,
        "started_at": utc_now_iso(),
        "git": git_revision(),
    }
    ensure_dir(RESULTS_DIR)
    write_json(RESULTS_DIR / "run_manifest.json", manifest)
    return manifest


def ensure_project_dirs(root: Path) -> None:
    """Create result/report directories used by all phases."""

    for name in ("results", "reports", "reports/figures"):
        ensure_dir(root / name)
