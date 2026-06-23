"""Selection rules for Gemma 4 MTP depth.

The logic mirrors the technical spec: exclude unsafe candidates first, then
choose the lowest median E2E latency across S2-S4, breaking ties toward smaller
`num_speculative_tokens`.
"""

from __future__ import annotations

from typing import Any

from .metrics import median


def choose_mtp_depth(candidates: list[dict[str, Any]], ar_baseline: dict[str, float]) -> dict[str, Any]:
    """Return the selected MTP candidate and rejected candidates with reasons."""

    rejected = []
    viable = []
    for candidate in candidates:
        reasons = []
        if candidate.get("oom") or candidate.get("server_restart_count", 0) > 0:
            reasons.append("unstable_runtime")
        if candidate.get("schema_parsing_error"):
            reasons.append("schema_parsing_error")
        if candidate.get("valid_json_rate", 0.0) < ar_baseline.get("valid_json_rate", 0.0) - 0.03:
            reasons.append("valid_json_guard")
        if candidate.get("task_success_rate", 0.0) < ar_baseline.get("task_success_rate", 0.0) - 0.05:
            reasons.append("task_success_guard")

        if reasons:
            rejected.append({"num_speculative_tokens": candidate.get("num_speculative_tokens"), "reasons": reasons})
        else:
            viable.append(candidate)

    if not viable:
        return {"selected": None, "rejected": rejected, "status": "BLOCKED_MTP_BACKEND"}

    def score(candidate: dict[str, Any]) -> tuple[float, int]:
        latencies = [
            candidate.get("median_e2e_latency_ms", {}).get("S2"),
            candidate.get("median_e2e_latency_ms", {}).get("S3"),
            candidate.get("median_e2e_latency_ms", {}).get("S4"),
        ]
        return (median(latencies) or float("inf"), int(candidate["num_speculative_tokens"]))

    selected = sorted(viable, key=score)[0]
    return {"selected": selected, "rejected": rejected, "status": "OK"}
