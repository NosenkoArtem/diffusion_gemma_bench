"""Small shared utilities used by CLI phases and notebook cells.

The harness intentionally keeps this module narrow: filesystem helpers,
JSON/JSONL persistence, timestamps, and configuration loading. Backend-specific
logic lives in adapter modules so that preflight and unit tests stay lightweight.
"""

from __future__ import annotations

import json
import os
import subprocess
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


SECRET_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY")


def utc_now_iso() -> str:
    """Return an ISO timestamp with timezone for machine-readable manifests."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str) -> str:
    """Create a short stable-looking run id for result filenames."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it for fluent call sites."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file.

    PyYAML is listed in `requirements.lock`; keeping YAML out of runtime imports
    until this function is called lets pure unit tests run in minimal notebooks.
    """

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load config files. Install requirements.lock first."
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at {path}")
    return loaded


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic UTF-8 JSON for manifests and summaries."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSONL record; callers should include `run_id` where relevant."""

    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, ignoring blank lines."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def public_env_snapshot() -> dict[str, Any]:
    """Return environment flags without leaking secret values."""

    return {key: {"present": bool(os.environ.get(key))} for key in SECRET_KEYS}


def git_revision(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return git revision metadata when the project is inside a git checkout."""

    def run_git(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run_git(["status", "--porcelain"])
    return {
        "is_git_checkout": commit is not None,
        "commit_sha": commit,
        "short_commit_sha": commit[:8] if commit else None,
        "branch": branch,
        "dirty": bool(dirty),
    }


def project_path(*parts: str) -> Path:
    """Build an absolute path under the repository root."""

    return PROJECT_ROOT.joinpath(*parts)
