"""Package and optionally push benchmark results to a GitHub results branch.

Default mode is safe: package files, validate them, and print the next command.
Use `--commit` to create a commit and `--push` to push it. The script expects the
repository remote to be configured already, preferably without embedding tokens
in the remote URL.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.result_store import make_run_id, package_results, validate_result_tree  # noqa: E402

PUSH_SCRIPT_VERSION = "2026-06-26-direct-url-push"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package and push small benchmark result artifacts.")
    parser.add_argument("--profile", default="q6_core_native")
    parser.add_argument("--phase", default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--branch", default="bench-results")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--commit", action="store_true", help="Create a local git commit for the packaged run.")
    parser.add_argument("--push", action="store_true", help="Push the commit to the result branch.")
    parser.add_argument(
        "--auth-check",
        action="store_true",
        help="Check GitHub push authentication and remote branch visibility without packaging results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.auth_check:
        check_push_auth(args.remote, args.branch)
        return 0
    if args.push:
        require_push_auth(args.remote)
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

    source_remote_url = git(["remote", "get-url", remote])
    remote_url = authenticated_remote_url(source_remote_url)
    user_name = git(["config", "user.name"], check=False) or "Colab Benchmark Bot"
    user_email = git(["config", "user.email"], check=False) or "colab-benchmark@example.invalid"
    rel_run_dir = run_dir.relative_to(ROOT)

    with tempfile.TemporaryDirectory(prefix="bench-results-") as tmp:
        temp_repo = Path(tmp) / "repo"
        temp_repo.mkdir(parents=True)
        git_in(temp_repo, ["init"])
        git_in(temp_repo, ["config", "user.name", user_name])
        git_in(temp_repo, ["config", "user.email", user_email])

        print(f"push_script_version: {PUSH_SCRIPT_VERSION}")
        print(f"push_remote_type: {remote_type(source_remote_url)}")
        branch_exists = remote_branch_exists(temp_repo, remote_url, branch)
        if branch_exists:
            git_in(temp_repo, ["fetch", "--depth", "1", remote_url, branch])
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
            git_in(temp_repo, ["push", remote_url, f"HEAD:{branch}"])


def git(args: list[str], check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(redact_secret(proc.stderr.strip() or proc.stdout.strip()))
    return proc.stdout.strip()


def require_push_auth(remote: str) -> None:
    """Fail early when a GitHub HTTPS push has no token available."""

    remote_url = git(["remote", "get-url", remote])
    if needs_github_token(remote_url) and not os.environ.get("GITHUB_TOKEN"):
        raise RuntimeError(
            "GITHUB_TOKEN is required for pushing results to GitHub from Colab. "
            "Load it into os.environ from /content/experiment.env before running this script."
        )


def check_push_auth(remote: str, branch: str) -> None:
    """Print a safe GitHub auth diagnostic without exposing token values."""

    remote_url = git(["remote", "get-url", remote])
    require_push_auth(remote)
    probe_url = authenticated_remote_url(remote_url)
    proc = subprocess.run(["git", "ls-remote", "--heads", probe_url, branch], cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(redact_secret(proc.stderr.strip() or proc.stdout.strip()))

    print(f"remote: {remote}")
    print(f"push_script_version: {PUSH_SCRIPT_VERSION}")
    print(f"remote_type: {remote_type(remote_url)}")
    print(f"github_token_present: {bool(os.environ.get('GITHUB_TOKEN'))}")
    print(f"branch: {branch}")
    print(f"branch_exists: {bool(proc.stdout.strip())}")
    print("auth_ok: True")


def needs_github_token(remote_url: str) -> bool:
    """Return True for GitHub HTTPS remotes that need token injection in Colab."""

    parsed = urlsplit(remote_url)
    return parsed.scheme == "https" and parsed.hostname == "github.com"


def remote_type(remote_url: str) -> str:
    """Classify the remote URL for safe diagnostics."""

    if needs_github_token(remote_url):
        return "github_https"
    if remote_url.startswith("git@github.com:"):
        return "github_ssh"
    return "other"


def authenticated_remote_url(remote_url: str) -> str:
    """Return a GitHub HTTPS URL authenticated with `GITHUB_TOKEN`.

    Colab clones may leave `origin` as either `https://github.com/...` or as a
    URL that already contains incomplete credentials, for example
    `https://TOKEN@github.com/...`. Git push treats the latter as username-only
    auth and then prompts for a password, which fails in notebook runtimes.
    Normalizing the netloc here keeps token handling in one temporary remote.
    """

    token = os.environ.get("GITHUB_TOKEN")
    if not token or not needs_github_token(remote_url):
        return remote_url
    parsed = urlsplit(remote_url)
    return urlunsplit((parsed.scheme, f"x-access-token:{token}@github.com", parsed.path, parsed.query, parsed.fragment))


def git_in(cwd: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(redact_secret(proc.stderr.strip() or proc.stdout.strip()))
    return proc


def remote_branch_exists(cwd: Path, remote_url: str, branch: str) -> bool:
    """Return True if the results branch exists; raise on auth/network errors."""

    proc = git_in(cwd, ["ls-remote", "--heads", remote_url, branch], check=False)
    if proc.returncode != 0:
        raise RuntimeError(redact_secret(proc.stderr.strip() or proc.stdout.strip()))
    return bool(proc.stdout.strip())


def redact_secret(text: str) -> str:
    """Remove token-like substrings from git error text."""

    token = os.environ.get("GITHUB_TOKEN")
    if token:
        text = text.replace(token, "***")
    text = re.sub(r"https://[^/@\s]+@github\.com", "https://***@github.com", text)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
