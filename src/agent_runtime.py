"""MiniToolAgent execution loop.

Adapters only need to expose `complete(messages, generation_config)`. The loop
stores every raw model response and every tool output so failures can be audited
without retrying or repairing model output.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from .minitoolagent import SYSTEM_PROMPT, AgentProtocolError, call_tool, parse_response


class CompletionAdapter(Protocol):
    def complete(self, messages: list[dict[str, str]], generation_config: dict[str, Any]) -> dict[str, Any]:
        """Return a dict containing at least `text` from the model."""


def run_agent_task(
    adapter: CompletionAdapter,
    task: dict[str, Any],
    generation_config: dict[str, Any],
    max_turns: int = 6,
) -> dict[str, Any]:
    """Run one deterministic tool task and return a raw audit record."""

    started = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    trajectory: list[dict[str, Any]] = []
    valid_json = True
    policy_violation = False

    for turn_index in range(max_turns):
        response = adapter.complete(messages, generation_config)
        raw_text = response.get("text", "")
        step: dict[str, Any] = {"turn": turn_index + 1, "raw_model_text": raw_text}
        try:
            parsed = parse_response(raw_text)
            step["parsed"] = parsed
        except AgentProtocolError as exc:
            valid_json = False
            step["error_type"] = str(exc)
            trajectory.append(step)
            break

        if parsed["type"] == "final":
            step["final_answer"] = parsed["answer"]
            trajectory.append(step)
            break

        tool_result = call_tool(parsed["name"], parsed["arguments"])
        if tool_result.get("error", {}).get("code") == "policy_restricted":
            policy_violation = False
        step["tool_result"] = tool_result
        trajectory.append(step)
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({"role": "tool", "content": str(tool_result)})

    expected = task.get("expected", {})
    task_success = _evaluate_expected(trajectory, expected)
    ended = time.perf_counter()
    return {
        "task_id": task.get("task_id"),
        "language": task.get("language"),
        "category": task.get("category"),
        "valid_json": valid_json,
        "task_success": task_success,
        "policy_violation": policy_violation,
        "turn_count": len(trajectory),
        "tool_call_count": sum(1 for step in trajectory if "tool_result" in step),
        "task_e2e_latency_ms": (ended - started) * 1000,
        "trajectory": trajectory,
    }


def _evaluate_expected(trajectory: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    """Small deterministic scorer for local tasks.

    The scorer checks only declared expectations. This keeps task definitions
    transparent and avoids hidden LLM-as-a-judge behavior.
    """

    if not trajectory:
        return False
    if expected.get("final_contains"):
        final_texts = [step.get("final_answer", "") for step in trajectory]
        needles = expected["final_contains"]
        if isinstance(needles, str):
            needles = [needles]
        if not any(needle in text for needle in needles for text in final_texts):
            return False
    if expected.get("first_tool"):
        first_tool = next((step["parsed"]["name"] for step in trajectory if step.get("parsed", {}).get("type") == "tool_call"), None)
        if first_tool != expected["first_tool"]:
            return False
    if expected.get("first_arguments"):
        first_args = next(
            (step["parsed"]["arguments"] for step in trajectory if step.get("parsed", {}).get("type") == "tool_call"),
            None,
        )
        if first_args is None or not _contains_expected_args(first_args, expected["first_arguments"]):
            return False
    if expected.get("must_call"):
        called = {step["parsed"]["name"] for step in trajectory if step.get("parsed", {}).get("type") == "tool_call"}
        if not set(expected["must_call"]).issubset(called):
            return False
    if expected.get("tool_sequence"):
        called_in_order = [
            step["parsed"]["name"] for step in trajectory if step.get("parsed", {}).get("type") == "tool_call"
        ]
        if called_in_order[: len(expected["tool_sequence"])] != expected["tool_sequence"]:
            return False
    if expected.get("error_code"):
        error_codes = [
            step.get("tool_result", {}).get("error", {}).get("code")
            for step in trajectory
            if step.get("tool_result", {}).get("ok") is False
        ]
        if expected["error_code"] not in error_codes:
            return False
    return True


def _contains_expected_args(actual: dict[str, Any], expected_subset: dict[str, Any]) -> bool:
    """Return True when all expected argument keys and values are present."""

    for key, expected_value in expected_subset.items():
        if actual.get(key) != expected_value:
            return False
    return True
