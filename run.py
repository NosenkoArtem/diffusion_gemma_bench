"""CLI entrypoint for the benchmark phases.

The same functions are safe to call from a notebook. Long-running phases are
gated until Colab preflight and backend smoke checks are implemented, which keeps
local development honest and prevents accidental fake benchmark results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.bootstrap import start_phase
from src.bfcl_runner import run_bfcl_lite
from src.preflight import run_preflight
from src.reporting import generate_report
from src.utils import RESULTS_DIR, append_jsonl, project_path, write_json


PHASES = {
    "preflight",
    "smoke",
    "pilot",
    "core",
    "quant-calibration",
    "bfcl-lite",
    "repeat-speed",
    "report",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffusionGemma/Gemma 4 benchmark harness")
    parser.add_argument("--profile", default="auto", help="Quantization/deployment profile from configs/profiles.yaml")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES), help="Benchmark phase to run")
    parser.add_argument("--run-id", default=None, help="Optional existing run id for resume workflows")
    parser.add_argument("--confirm-go", action="store_true", help="Required for phases that can consume long GPU time")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = start_phase(args.phase, args.profile, args.run_id)

    if args.phase == "preflight":
        result = run_preflight(args.profile)
        print_summary(result)
        return 0

    if args.phase == "report":
        summary = generate_report()
        print(f"Report written with {len(summary['sections'])} required sections.")
        return 0

    if args.phase in {"core", "quant-calibration", "bfcl-lite"} and not args.confirm_go:
        raise SystemExit(f"{args.phase} requires --confirm-go by design.")

    if args.phase == "bfcl-lite":
        result = run_bfcl_lite()
        write_json(RESULTS_DIR / "bfcl_summary.json", result)
        print(result["status"])
        return 0

    result = gated_phase_placeholder(args.phase, args.profile, manifest)
    print(result["status"])
    return 0


def gated_phase_placeholder(phase: str, profile: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Record that a model phase is intentionally waiting for Colab wiring."""

    # Keep placeholder phases runnable before Colab dependencies are installed.
    # Full YAML parsing happens in the real phase implementations.
    for rel_path in (
        ("configs", "profiles.yaml"),
        ("configs", "models.yaml"),
        ("configs", "generation.yaml"),
    ):
        path = project_path(*rel_path)
        if not path.exists():
            raise FileNotFoundError(path)

    status = {
        "run_id": manifest["run_id"],
        "phase": phase,
        "profile": profile,
        "status": "PENDING_COLAB_BACKEND_GATE",
        "reason": (
            "This scaffold does not fabricate model results. Run preflight in "
            "Colab, then implement/enable the vLLM capability gate before this phase."
        ),
    }
    write_json(RESULTS_DIR / f"{phase}_status.json", status)
    append_jsonl(RESULTS_DIR / "phase_events.jsonl", status)
    return status


def print_summary(result: dict[str, Any]) -> None:
    """Compact terminal/notebook summary that avoids secret values."""

    print(f"action: {result['action']}")
    print(f"selected_profile: {result['selected_profile']}")
    print(f"status_reasons: {result['status_reasons']}")
    print(f"gpu: {result['hardware']['gpu']}")
    print(f"disk_free_gib: {result['hardware']['disk_free_gib']}")


if __name__ == "__main__":
    sys.exit(main())
