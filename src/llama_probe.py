"""llama.cpp diagnostic probe placeholders.

llama.cpp results are allowed only as technical probes unless the same stable
server runtime supports both models, streaming, repeated agent turns, and Gemma
MTP. This module keeps that path separate from the vLLM primary benchmark.
"""

from __future__ import annotations

from typing import Any


def probe_status() -> dict[str, Any]:
    """Return the current llama.cpp probe status."""

    return {"status": "ALTERNATIVE_LLAMA_CPP_PROBE", "implemented": False}
