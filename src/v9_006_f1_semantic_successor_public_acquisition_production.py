"""Stage-2B standard-library HTTP and production output boundary."""
from __future__ import annotations

from http.client import HTTPException, IncompleteRead, RemoteDisconnected
from hashlib import sha256
import argparse
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src import v9_006_f1_semantic_successor_public_acquisition as acquisition
from src import v9_006_f1_semantic_successor_public_acquisition_runtime as runtime
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL, validate_jpx_url

AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
DESIGN_BLOB = "ea612a777dd2915121f1747cdd3a14ff7f668efb"
DESIGN_PATH = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_DESIGN.md"
SAFE_RESULT = "safe-result.json"
IMPLEMENTATION_FAILURE_MARKER = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_IMPLEMENTATION_FAILURE"


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTransport:
    def __init__(self, opener=None):
        self.opener = opener or build_opener(NoRedirectHandler())

    def fetch(self, url: str, _attempt: int) -> acquisition.FetchOutcome:
        validate_jpx_url(url)
        request = Request(url, method="GET")
        try:
            response = self.opener.open(request)
        except HTTPError as error:
            resolved = error.geturl() or url
            return acquisition.FetchOutcome(error.code, None, False, resolved)
        except (ssl.SSLError, HTTPException, URLError, socket.timeout, TimeoutError, ConnectionError, RemoteDisconnected):
            return acquisition.FetchOutcome(None, None, False, url)
        status = response.getcode()
        resolved = response.geturl() or url
        if status != 200:
            response.close()
            return acquisition.FetchOutcome(status, None, False, resolved)
        try:
            payload = response.read()
        except (ssl.SSLError, HTTPException, IncompleteRead, URLError, socket.timeout, TimeoutError, ConnectionError, RemoteDisconnected):
            return acquisition.FetchOutcome(200, None, False, resolved)
        finally:
            response.close()
        return acquisition.FetchOutcome(200, payload, True, resolved)


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(("git", *arguments), cwd=repo_root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def check_bindings(implementation_git_sha: str, repo_root: Path) -> None:
    if not acquisition._hex(implementation_git_sha, 40):
        raise ValueError("implementation binding")
    if _git_output(repo_root, "branch", "--show-current") != AUTHORITATIVE_BRANCH:
        raise ValueError("branch binding")
    if _git_output(repo_root, "rev-parse", "HEAD") != implementation_git_sha:
        raise ValueError("HEAD binding")
    if _git_output(repo_root, "status", "--porcelain"):
        raise ValueError("worktree binding")
    if _git_output(repo_root, "rev-parse", f"HEAD:{DESIGN_PATH}") != DESIGN_BLOB:
        raise ValueError("design binding")


def _publish_safe_result(state_root: Path, value: dict[str, object]) -> str | None:
    acquisition.validate_safe_acquisition_result(value)
    canonical = acquisition.canonical_json(value)
    path = state_root / SAFE_RESULT
    if not runtime._write_exclusive(path, canonical.encode("utf-8")):
        return None
    try:
        reread = path.read_bytes()
        if reread != canonical.encode("utf-8"):
            return None
        parsed = json.loads(reread.decode("utf-8"))
        acquisition.validate_safe_acquisition_result(parsed)
        if acquisition.canonical_json(parsed) != canonical:
            return None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    return canonical


def run_production(implementation_git_sha: str, state_root: Path, *, repo_root: Path | None = None, root_fetch: Callable | None = None, terminal_fetch: Callable | None = None, locator_runner=None, binding_check: Callable[[str, Path], None] = check_bindings) -> tuple[dict[str, object] | None, str | None]:
    repository = repo_root or Path(__file__).resolve().parents[1]
    binding_check(implementation_git_sha, repository)
    transport = HttpTransport()
    root_fetch = root_fetch or transport.fetch
    terminal_fetch = terminal_fetch or transport.fetch
    result = runtime.run_durable_acquisition(implementation_git_sha, Path(state_root), LISTED_ISSUES_PAGE_URL, root_fetch, terminal_fetch, locator_runner=locator_runner or acquisition.locator.run_fresh_root_locator)
    if (result["result"], result["failure_stage"]) == ("GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT"):
        return result, acquisition.canonical_json(result)
    canonical = _publish_safe_result(Path(state_root), result)
    if canonical is None:
        return None, None
    return result, canonical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--implementation-git-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, canonical = run_production(args.implementation_git_sha, Path(args.state_root))
        if result is None or canonical is None:
            raise RuntimeError("safe result publication")
        sys.stdout.write(canonical + "\n")
        return 0 if result["result"] == "SUCCESS" else 2
    except Exception:
        sys.stdout.write("")
        sys.stderr.write(IMPLEMENTATION_FAILURE_MARKER + "\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
