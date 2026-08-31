"""Read-only production seam for the F1 terminal structure diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from src import v9_006_f1_semantic_successor_public_acquisition as acquisition
from src import v9_006_f1_semantic_successor_public_acquisition_runtime as runtime
from src import v9_006_f1_terminal_month_structure_evidence as diagnostic

AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
DESIGN_PATH = "V9_006_F1_TERMINAL_MONTH_STRUCTURE_EVIDENCE_DESIGN.md"
DESIGN_BLOB = "6112b92f39f34c594d36a28d72072dcb255b9eee"
IMPLEMENTATION_FAILURE_MARKER = "V9_006_F1_TERMINAL_MONTH_STRUCTURE_EVIDENCE_IMPLEMENTATION_FAILURE"


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(("git", *arguments), cwd=repo_root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def check_bindings(implementation_git_sha: str, repo_root: Path, *, git_output: Callable[..., str] = _git_output) -> None:
    if not diagnostic._hex(implementation_git_sha, 40):
        raise ValueError("implementation binding")
    if git_output(repo_root, "branch", "--show-current") != AUTHORITATIVE_BRANCH:
        raise ValueError("branch binding")
    if git_output(repo_root, "rev-parse", "HEAD") != implementation_git_sha:
        raise ValueError("HEAD binding")
    if git_output(repo_root, "status", "--porcelain"):
        raise ValueError("worktree binding")
    if git_output(repo_root, "rev-parse", f"HEAD:{DESIGN_PATH}") != DESIGN_BLOB:
        raise ValueError("design binding")
    git_output(repo_root, "fetch", "--no-tags", "origin", AUTHORITATIVE_BRANCH)
    if git_output(repo_root, "rev-parse", f"origin/{AUTHORITATIVE_BRANCH}") != implementation_git_sha:
        raise ValueError("remote binding")


def derive_state_root(repo_root: Path) -> Path:
    return repo_root.parent / diagnostic.STATE_ROOT_BASENAME


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("safe result")
    acquisition.validate_safe_acquisition_result(value)
    return value


def _acquisition_binding_ok(value: dict[str, Any]) -> bool:
    return (
        value.get("result") == "SUCCESS" and value.get("failure_stage") == "NONE" and
        value.get("design_git_sha") == diagnostic.ACQUISITION_DESIGN_GIT_SHA and
        value.get("implementation_git_sha") == diagnostic.ACQUISITION_IMPLEMENTATION_GIT_SHA and
        value.get("terminal_payload_sha256") == diagnostic.TERMINAL_PAYLOAD_SHA256 and
        value.get("terminal_byte_length") == diagnostic.TERMINAL_BYTE_LENGTH and
        value.get("raw_lock_set_sha256") == diagnostic.RAW_LOCK_SET_SHA256 and
        value.get("raw_lock_count") == 2 and value.get("safe_provenance_verified") is True
    )


def run_from_state(implementation_git_sha: str, repo_root: Path, *, binding_check: Callable[[str, Path], None] = check_bindings, profiler: Callable[[bytes], dict[str, Any]] | None = None) -> dict[str, Any]:
    binding_check(implementation_git_sha, repo_root)
    state_root = derive_state_root(repo_root)
    try:
        acquisition_result = _read_json(state_root / "safe-result.json")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "PRE_READ_BINDING")
    if not _acquisition_binding_ok(acquisition_result):
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "PRE_READ_BINDING")
    state = runtime.DurableState(state_root)
    try:
        if set(item.name for item in (state_root / "raw").iterdir()) != {acquisition.ROOT_PERIOD, acquisition.TERMINAL_PERIOD}:
            return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
    except OSError:
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
    root = state._read(acquisition.ROOT_PERIOD)
    terminal = state._read(acquisition.TERMINAL_PERIOD)
    if root is None or terminal is None:
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
    if root.lock.payload_sha256 != acquisition_result.get("discovery_root_payload_sha256") or root.lock.byte_length != acquisition_result.get("discovery_root_byte_length"):
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
    if terminal.lock.payload_sha256 != acquisition_result.get("terminal_payload_sha256") or terminal.lock.byte_length != acquisition_result.get("terminal_byte_length"):
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
    if acquisition.raw_lock_set_sha256(root.lock, terminal.lock) != diagnostic.RAW_LOCK_SET_SHA256 or terminal.lock.payload_sha256 != diagnostic.TERMINAL_PAYLOAD_SHA256 or terminal.lock.byte_length != diagnostic.TERMINAL_BYTE_LENGTH:
        return diagnostic._base(implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
    if profiler is None:
        return diagnostic.run_terminal_structure_diagnostic(terminal.payload, implementation_git_sha)
    return diagnostic.run_terminal_structure_diagnostic(terminal.payload, implementation_git_sha, profiler=profiler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diagnostic-implementation-git-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        repo_root = Path(__file__).resolve().parents[1]
        result = run_from_state(args.diagnostic_implementation_git_sha, repo_root)
        diagnostic.validate_safe_result(result)
        print(diagnostic.canonical_json(result))
        return 0 if result["diagnostic_result"] == "EVIDENCE_CAPTURED" else 2
    except Exception:
        return_code = 3
        import sys
        sys.stdout.write("")
        sys.stderr.write(IMPLEMENTATION_FAILURE_MARKER + "\n")
        return return_code


if __name__ == "__main__":
    raise SystemExit(main())
