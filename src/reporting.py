"""Report generation from persisted benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import agent_quality_summary, speed_summary
from .utils import REPORTS_DIR, RESULTS_DIR, read_jsonl, write_json


REQUIRED_SECTIONS = [
    "Executive summary",
    "Hardware and runtime",
    "Result status",
    "Model artifacts and SHA256",
    "Backend/version matrix",
    "MTP configuration and tuning",
    "Cold speed benchmark",
    "Warm-agent benchmark",
    "MiniToolAgent",
    "G26-AR vs G26-MTP attribution",
    "Quantization sensitivity",
    "BFCL-lite",
    "Failure modes",
    "Limitations",
    "Final conclusion",
    "Reproduction commands",
]


def generate_report(results_dir: Path = RESULTS_DIR, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """Create Markdown/HTML report files from available raw results."""

    speed_records = read_jsonl(results_dir / "speed_raw.jsonl")
    agent_records = read_jsonl(results_dir / "minitoolagent_raw.jsonl")
    summary = {
        "speed": speed_summary(speed_records),
        "agent_quality": agent_quality_summary(agent_records),
        "sections": REQUIRED_SECTIONS,
    }
    write_json(results_dir / "summary_metrics.json", summary)
    markdown = render_markdown(summary)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    (reports_dir / "final_report.html").write_text(render_html(markdown), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a conservative report draft.

    Real conclusions are filled only when real raw measurements exist.
    """

    lines = ["# DiffusionGemma vs Gemma 4 Benchmark Report", ""]
    for section in REQUIRED_SECTIONS:
        lines.extend([f"## {section}", ""])
        if section == "Executive summary":
            lines.append("Pending real Colab benchmark results.")
        elif section == "Cold speed benchmark":
            lines.append(f"Speed groups available: {len(summary['speed']['groups'])}.")
        elif section == "MiniToolAgent":
            lines.append(f"Agent quality groups available: {len(summary['agent_quality']['groups'])}.")
        elif section == "Backend/version matrix":
            lines.append("See `results/backend_capability.json` and `results/backend_server_smoke.json` when available.")
        else:
            lines.append("Pending data from the corresponding phase.")
        lines.append("")
    return "\n".join(lines)


def render_html(markdown: str) -> str:
    """Render minimal HTML without extra dependencies."""

    body_lines = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{line[3:]}</h2>")
        elif line:
            body_lines.append(f"<p>{line}</p>")
    return "<!doctype html><html><body>\n" + "\n".join(body_lines) + "\n</body></html>\n"
