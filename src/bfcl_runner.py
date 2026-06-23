"""BFCL-lite integration boundary.

BFCL-lite is optional and runs only after Core. The first scaffold records the
skip status rather than replacing the official evaluator with a custom scorer.
"""

from __future__ import annotations

from typing import Any


def run_bfcl_lite(*_: Any, **__: Any) -> dict[str, Any]:
    """Return a spec-compliant skip status until the official evaluator is wired."""

    return {"status": "SKIPPED_BFCL_INTEGRATION", "reason": "official evaluator not configured"}
