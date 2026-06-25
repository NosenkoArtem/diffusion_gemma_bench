"""Central experiment environment configuration helpers.

Secrets must come from environment variables or an untracked local env file.
This module never prints secret values; it returns only presence flags for
operator-facing summaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECRET_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")


@dataclass(frozen=True)
class ExperimentEnv:
    """Runtime variables shared by CLI, notebooks, and Colab scripts."""

    repo_url: str
    code_branch: str
    results_branch: str
    profile: str
    phase: str
    project_dir: str
    vllm_host: str
    vllm_port: int
    secret_presence: dict[str, bool]


DEFAULTS = {
    "REPO_URL": "https://github.com/NosenkoArtem/diffusion_gemma_bench.git",
    "CODE_BRANCH": "main",
    "RESULTS_BRANCH": "bench-results",
    "PROFILE": "q6_core_native",
    "PHASE": "backend-check",
    "PROJECT_DIR": "/content/diffusion_gemma_bench",
    "VLLM_HOST": "127.0.0.1",
    "VLLM_PORT": "8000",
}


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without external dependencies."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def apply_env_file(path: Path, override: bool = False) -> dict[str, str]:
    """Load env vars from a file into `os.environ`.

    Existing environment variables win by default, which is important for Colab
    secrets and CI environments.
    """

    loaded = load_env_file(path)
    for key, value in loaded.items():
        if value and (override or key not in os.environ):
            os.environ[key] = value
    return loaded


def get_experiment_env(env_file: Path | None = None) -> ExperimentEnv:
    """Return typed experiment variables with secret values redacted."""

    if env_file:
        apply_env_file(env_file)
    merged = {key: os.environ.get(key, default) for key, default in DEFAULTS.items()}
    return ExperimentEnv(
        repo_url=merged["REPO_URL"],
        code_branch=merged["CODE_BRANCH"],
        results_branch=merged["RESULTS_BRANCH"],
        profile=merged["PROFILE"],
        phase=merged["PHASE"],
        project_dir=merged["PROJECT_DIR"],
        vllm_host=merged["VLLM_HOST"],
        vllm_port=int(merged["VLLM_PORT"]),
        secret_presence={name: bool(os.environ.get(name)) for name in SECRET_NAMES},
    )


def sanitized_env_summary(config: ExperimentEnv) -> dict[str, Any]:
    """Return a notebook/report-safe view of experiment settings."""

    return {
        "repo_url": config.repo_url,
        "code_branch": config.code_branch,
        "results_branch": config.results_branch,
        "profile": config.profile,
        "phase": config.phase,
        "project_dir": config.project_dir,
        "vllm_host": config.vllm_host,
        "vllm_port": config.vllm_port,
        "secret_presence": config.secret_presence,
    }
