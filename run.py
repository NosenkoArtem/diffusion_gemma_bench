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
from src.backend_check import run_backend_check
from src.backend_smoke import run_backend_smoke
from src.model_gate import run_model_gate
from src.preflight import run_preflight
from src.reporting import generate_report
from src.vllm_setup import run_vllm_setup
from src.utils import RESULTS_DIR, append_jsonl, project_path, write_json


PHASES = {
    "preflight",
    "backend-check",
    "backend-smoke",
    "model-gate",
    "vllm-setup",
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

    if args.phase == "backend-check":
        result = run_backend_check(args.profile)
        print_backend_check_summary(result)
        return 0

    if args.phase == "backend-smoke":
        result = run_backend_smoke(args.profile)
        print_backend_smoke_summary(result)
        return 0

    if args.phase == "model-gate":
        result = run_model_gate(args.profile)
        print_model_gate_summary(result)
        return 0

    if args.phase == "vllm-setup":
        result = run_vllm_setup(args.profile)
        print_vllm_setup_summary(result)
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


def print_backend_check_summary(result: dict[str, Any]) -> None:
    """Compact backend-check summary for notebook output."""

    print(f"status: {result['status']}")
    print(f"reasons: {result['reasons']}")
    print(f"gpu: {result['gpu']}")
    print(f"packages: {result['packages']}")
    print(f"hf_token_present: {result['hf_token']['present']}")
    print(f"localhost_port_free: {result['localhost']['port_free']}")
    print(f"next_step: {result['next_step']}")


def print_backend_smoke_summary(result: dict[str, Any]) -> None:
    """Compact backend-smoke summary for notebook output."""

    print(f"status: {result['status']}")
    print(f"server: {result.get('server_bound_host')}:{result.get('server_port')}")
    print(f"health_ok: {result.get('health_ok')}")
    print(f"non_streaming_ok: {result.get('non_streaming_ok')}")
    print(f"streaming_ok: {result.get('streaming_ok')}")
    print(f"strict_json_ok: {result.get('strict_json_ok')}")
    print(f"ttfc_ms: {result.get('ttfc_ms')}")
    print(f"e2e_latency_ms: {result.get('e2e_latency_ms')}")


def print_model_gate_summary(result: dict[str, Any]) -> None:
    """Compact model-gate summary for notebook output."""

    print(f"status: {result['status']}")
    print(f"reasons: {result['reasons']}")
    print(f"gpu: {result['gpu']}")
    print(f"disk: {result['disk']}")
    print(f"packages: {result['packages']}")
    print(f"hf_token_present: {result['hf_token']['present']}")
    for model in result["models"]:
        print(
            "model: "
            f"{model['model_id']} repo_access_ok={model.get('repo_access_ok')} "
            f"expected_file_visible={model.get('expected_file_visible')} "
            f"expected_filename={model.get('expected_filename')}"
        )
    print(f"next_step: {result['next_step']}")


def print_vllm_setup_summary(result: dict[str, Any]) -> None:
    """Compact vLLM setup summary for notebook output."""

    print(f"status: {result['status']}")
    print(f"reasons: {result['reasons']}")
    print(f"gpu: {result['gpu']}")
    print(f"packages: {result['packages']}")
    print(f"vllm_import: {result['vllm_import']}")
    print(f"next_step: {result['next_step']}")


if __name__ == "__main__":
    sys.exit(main())
