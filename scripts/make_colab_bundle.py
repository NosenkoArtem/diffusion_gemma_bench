"""Create a small source bundle for VS Code + Colab kernel smoke tests.

When a notebook is executed on a Colab kernel, the remote runtime may not see the
local VS Code workspace. This script packages the repository source files into a
zip that can be uploaded from the notebook with `google.colab.files.upload()`.

The bundle intentionally excludes generated caches, raw results, reports, and
large artifacts. It is for code smoke tests, not for preserving benchmark output.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "diffusion_gemma_bench_source.zip"

INCLUDE_DIRS = ("configs", "data", "notebooks", "scripts", "src", "tests")
INCLUDE_FILES = (".gitignore", "README.md", "requirements.lock", "run.py")
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def should_include(path: Path) -> bool:
    """Return True for repository source files that belong in the Colab bundle."""

    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return False
    return rel.parts[0] in INCLUDE_DIRS or str(rel) in INCLUDE_FILES


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and should_include(path):
                zf.write(path, path.relative_to(ROOT))
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
