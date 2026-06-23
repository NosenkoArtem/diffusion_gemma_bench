"""Metric helpers for speed, agent quality, MTP tuning, and reports."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def median(values: Iterable[float]) -> float | None:
    """Return median for non-empty numeric values, otherwise `None`."""

    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(clean) if clean else None


def rate(success_count: int, total_count: int) -> float:
    """Return a fraction in [0, 1], using 0 for empty denominators."""

    return success_count / total_count if total_count else 0.0


def wilson_interval(success_count: int, total_count: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score interval for a binomial rate.

    This is used for report CIs without pulling in scipy.
    """

    if total_count == 0:
        return (0.0, 0.0)
    p = success_count / total_count
    denom = 1 + z * z / total_count
    center = (p + z * z / (2 * total_count)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total_count)) / total_count) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def speed_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize speed records grouped by model, scenario, and track."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec.get("model_id", "unknown"), rec.get("prompt_id", "unknown"), rec.get("track", "cold"))
        groups.setdefault(key, []).append(rec)

    rows = []
    for (model_id, prompt_id, track), items in sorted(groups.items()):
        rows.append(
            {
                "model_id": model_id,
                "prompt_id": prompt_id,
                "track": track,
                "n": len(items),
                "median_ttfc_ms": median(x.get("ttfc_ms") for x in items),
                "median_e2e_latency_ms": median(x.get("e2e_latency_ms") for x in items),
                "median_visible_chars_per_second": median(
                    x.get("visible_characters_per_second") for x in items
                ),
                "median_peak_vram_mb": median(x.get("peak_vram_mb") for x in items),
            }
        )
    return {"groups": rows}


def agent_quality_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize MiniToolAgent raw records for one or more models."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec.get("model_id", "unknown"), []).append(rec)

    rows = []
    for model_id, items in sorted(groups.items()):
        total = len(items)
        successes = sum(1 for x in items if x.get("task_success"))
        valid_json = sum(1 for x in items if x.get("valid_json"))
        policy_violations = sum(1 for x in items if x.get("policy_violation"))
        lo, hi = wilson_interval(successes, total)
        rows.append(
            {
                "model_id": model_id,
                "n": total,
                "task_success_rate": rate(successes, total),
                "task_success_ci95_low": lo,
                "task_success_ci95_high": hi,
                "valid_json_rate": rate(valid_json, total),
                "policy_violation_rate": rate(policy_violations, total),
                "mean_agent_turns": median(x.get("turn_count") for x in items),
            }
        )
    return {"groups": rows}


def paired_success(records: list[dict[str, Any]], left_model: str, right_model: str) -> dict[str, int]:
    """Build paired task-success counts for two models on shared task ids."""

    by_task: dict[str, dict[str, bool]] = {}
    for rec in records:
        task_id = rec.get("task_id")
        model_id = rec.get("model_id")
        if not task_id or model_id not in {left_model, right_model}:
            continue
        by_task.setdefault(task_id, {})[model_id] = bool(rec.get("task_success"))

    out = {
        f"{left_model}_only_success": 0,
        f"{right_model}_only_success": 0,
        "both_success": 0,
        "both_failure": 0,
        "unpaired": 0,
    }
    for pair in by_task.values():
        if left_model not in pair or right_model not in pair:
            out["unpaired"] += 1
            continue
        left = pair[left_model]
        right = pair[right_model]
        if left and right:
            out["both_success"] += 1
        elif left:
            out[f"{left_model}_only_success"] += 1
        elif right:
            out[f"{right_model}_only_success"] += 1
        else:
            out["both_failure"] += 1
    return out
