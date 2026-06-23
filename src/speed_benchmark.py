"""Cold and warm speed benchmark helpers."""

from __future__ import annotations

import time
from typing import Any, Protocol


class StreamingAdapter(Protocol):
    def stream_complete(self, messages: list[dict[str, str]], generation_config: dict[str, Any]) -> dict[str, Any]:
        """Return text, TTFC, and E2E timing fields."""


def run_speed_prompt(
    adapter: StreamingAdapter,
    prompt: dict[str, Any],
    generation_config: dict[str, Any],
    run_context: dict[str, Any],
) -> dict[str, Any]:
    """Run one speed prompt and normalize the raw record shape."""

    start_timestamp = time.time()
    result = adapter.stream_complete(
        [{"role": "user", "content": prompt["prompt"]}],
        generation_config | {"max_tokens": prompt["max_output_tokens"]},
    )
    end_timestamp = time.time()
    visible_characters = len(result.get("text", ""))
    e2e_seconds = max(result.get("e2e_latency_ms", 0.0) / 1000, 1e-9)
    return {
        **run_context,
        "prompt_id": prompt["prompt_id"],
        "track": prompt.get("track", "cold"),
        "requested_max_output_tokens": prompt["max_output_tokens"],
        "visible_characters": visible_characters,
        "request_start_timestamp": start_timestamp,
        "request_end_timestamp": end_timestamp,
        "ttfc_ms": result.get("ttfc_ms"),
        "e2e_latency_ms": result.get("e2e_latency_ms"),
        "visible_characters_per_second": visible_characters / e2e_seconds,
        "error_type": result.get("error_type"),
    }
