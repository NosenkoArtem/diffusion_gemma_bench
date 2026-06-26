"""Fail if source files contain token-looking secrets.

Run before committing notebook or docs changes. The patterns are intentionally
simple and conservative for Hugging Face and GitHub token formats.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "dist", "results"}
CHECK_SUFFIXES = {".py", ".md", ".ipynb", ".json", ".yaml", ".yml", ".txt", ".example"}
SECRET_PATTERNS = {
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "github_classic_token": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "github_fine_grained_token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in CHECK_SUFFIXES or path.name in {".env", ".gitignore"}:
            files.append(path)
    return files


def main() -> int:
    findings = []
    for path in iter_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append((path.relative_to(ROOT), name))
    if findings:
        print("Secret-like tokens found:")
        for rel, name in findings:
            print(f"- {rel}: {name}")
        return 1
    print("No secret-like tokens found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
