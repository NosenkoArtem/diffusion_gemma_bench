"""Model artifact manifest helpers.

This module records what was requested and what was actually present locally.
Network downloads are intentionally opt-in from Colab code; local unit tests can
verify manifests without touching Hugging Face.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import utc_now_iso


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(model_id: str, repo_id: str, filename: str, local_path: Path | None = None) -> dict[str, Any]:
    """Build a model artifact record for `results/model_artifacts.json`."""

    record: dict[str, Any] = {
        "model_id": model_id,
        "repo_id": repo_id,
        "filename": filename,
        "timestamp": utc_now_iso(),
    }
    if local_path and local_path.exists():
        record.update(
            {
                "local_path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
            }
        )
    else:
        record["status"] = "not_downloaded"
    return record
