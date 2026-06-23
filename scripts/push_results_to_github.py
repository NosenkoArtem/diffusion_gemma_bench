"""Package and optionally push benchmark results to a GitHub results branch.

Default mode is safe: package files, validate them, and print the next command.
Use `--commit` to create a commit and `--push` to push it. The script expects the
repository remote to be configured already, preferably without embedding tokens
in the remote URL.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
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

    commit_results_in_temp_repo(
        run_dir=run_dir,
        branch=args.branch,
        remote=args.remote,
        run_id=run_id,
        push=args.push,
    )
    return 0


def commit_results_in_temp_repo(run_dir: Path, branch: str, remote: str, run_id: str, push: bool) -> None:
    """Commit result artifacts without switching the current working tree.

    Colab phases modify `results/*.json` in the code checkout. Switching that
    checkout to an orphan results branch would fail or risk confusing local
    state. A temporary checkout keeps the source tree untouched.
    """

    remote_url = git(["remote", "get-url", remote])
    user_name = git(["config", "user.name"], check=False) or "Colab Benchmark Bot"
    user_email = git(["config", "user.email"], check=False) or "colab-benchmark@example.invalid"
    rel_run_dir = run_dir.relative_to(ROOT)

    with tempfile.TemporaryDirectory(prefix="bench-results-") as tmp:
        temp_repo = Path(tmp) / "repo"
        temp_repo.mkdir(parents=True)
        git_in(temp_repo, ["init"])
        git_in(temp_repo, ["config", "user.name", user_name])
        git_in(temp_repo, ["config", "user.email", user_email])
        git_in(temp_repo, ["remote", "add", remote, remote_url])

        fetch = git_in(temp_repo, ["fetch", "--depth", "1", remote, branch], check=False)
        if fetch.returncode == 0:
            git_in(temp_repo, ["checkout", "-B", branch, "FETCH_HEAD"])
        else:
            git_in(temp_repo, ["checkout", "--orphan", branch])

        dest = temp_repo / rel_run_dir
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(run_dir, dest)

        git_in(temp_repo, ["add", "-f", str(rel_run_dir)])
        status = git_in(temp_repo, ["status", "--porcelain"]).stdout.strip()
        if not status:
            print("No result changes to commit.")
            return
        git_in(temp_repo, ["commit", "-m", f"Add benchmark results {run_id}"])
        if push:
            git_in(temp_repo, ["push", remote, f"HEAD:{branch}"])


def git(args: list[str], check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def git_in(cwd: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc


if __name__ == "__main__":
    raise SystemExit(main())
