"""Package and optionally push benchmark results to a GitHub results branch.

Default mode is safe: package files, validate them, and print the next command.
Use `--commit` to create a commit and `--push` to push it. The script expects the
repository remote to be configured already, preferably without embedding tokens
in the remote URL.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.result_store import make_run_id, package_results, validate_result_tree  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package and push small benchmark result artifacts.")
    parser.add_argument("--profile", default="q6_core_native")
    parser.add_argument("--phase", default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--branch", default="bench-results")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--commit", action="store_true", help="Create a local git commit for the packaged run.")
    parser.add_argument("--push", action="store_true", help="Push the commit to the result branch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commit_sha = git(["rev-parse", "HEAD"], check=False)
    run_id = args.run_id or make_run_id(args.profile, args.phase, commit_sha=commit_sha)
    manifest = package_results(run_id=run_id, profile=args.profile, phase=args.phase)
    run_dir = Path(manifest["copied_files"][0]).parents[0] if manifest["copied_files"] else ROOT / "results" / "runs" / run_id
    validation = validate_result_tree(run_dir)
    print(f"run_dir: {run_dir}")
    print(f"copied_files: {len(manifest['copied_files'])}")
    print(f"validation_ok: {validation['ok']}")
    if validation["errors"]:
        print("validation_errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
        return 2

    if not args.commit and not args.push:
        print("Dry run complete. Re-run with --commit to create a Git commit.")
        return 0

    ensure_results_branch(args.branch)
    git(["add", "-f", str(run_dir.relative_to(ROOT))])
    git(["commit", "-m", f"Add benchmark results {run_id}"])

    if args.push:
        git(["push", args.remote, f"HEAD:{args.branch}"])
    return 0


def ensure_results_branch(branch: str) -> None:
    current = git(["rev-parse", "--abbrev-ref", "HEAD"])
    if current == branch:
        return
    existing = git(["branch", "--list", branch])
    if existing.strip():
        git(["switch", branch])
    else:
        git(["switch", "--orphan", branch])
        # Keep the orphan branch focused on result artifacts. Tracked source
        # files remain in the working tree but are not committed unless added.


def git(args: list[str], check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
