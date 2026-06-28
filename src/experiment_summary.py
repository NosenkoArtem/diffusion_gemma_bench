"""Per-experiment summary documents.

Each Colab experiment should leave a small human-readable document next to raw
JSON artifacts. The summary is deliberately Markdown-first so it is easy to
read in Google Drive, GitHub, or a notebook output cell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import REPORTS_DIR, RESULTS_DIR, ensure_dir, utc_now_iso, write_json


def write_experiment_summary(
    *,
    phase: str,
    profile: str,
    title: str,
    objective: str,
    tasks: list[str],
    metrics: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
    conclusion: str,
    artifacts: list[str],
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Write `experiment_summary.json` and `experiment_summary.md`.

    The function keeps the schema small and stable: one objective, task list,
    metric table, criteria table, conclusion, and artifact references.
    """

    payload = {
        "timestamp": utc_now_iso(),
        "phase": phase,
        "profile": profile,
        "title": title,
        "objective": objective,
        "tasks": tasks,
        "metrics": metrics,
        "criteria": criteria,
        "conclusion": conclusion,
        "artifacts": artifacts,
    }
    ensure_dir(reports_dir)
    write_json(reports_dir / "experiment_summary.json", payload)
    (reports_dir / "experiment_summary.md").write_text(render_markdown(payload), encoding="utf-8")
    return payload


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown experiment document."""

    lines = [
        f"# {summary['title']}",
        "",
        f"- Phase: `{summary['phase']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Timestamp: `{summary['timestamp']}`",
        "",
        "## Objective",
        "",
        summary["objective"],
        "",
        "## Tasks",
        "",
    ]
    lines.extend(f"- {task}" for task in summary["tasks"])
    lines.extend(["", "## Metrics", ""])
    lines.extend(markdown_table(summary["metrics"], ("metric", "value", "status", "note")))
    lines.extend(["", "## Success Criteria", ""])
    lines.extend(markdown_table(summary["criteria"], ("criterion", "passed", "note")))
    lines.extend(["", "## Conclusion", "", summary["conclusion"], "", "## Artifacts", ""])
    lines.extend(f"- `{artifact}`" for artifact in summary["artifacts"])
    lines.append("")
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[str]:
    """Return a GitHub-flavored Markdown table for the selected columns."""

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [_cell(row.get(col, "")) for col in columns]
        body.append("| " + " | ".join(values) + " |")
    return [header, sep, *body]


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`"
    return str(value).replace("\n", " ").replace("|", "\\|")
