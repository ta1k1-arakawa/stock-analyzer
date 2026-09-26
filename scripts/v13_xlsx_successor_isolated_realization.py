"""One-shot offline realization of the frozen XLSX candidate. Default is rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from scripts import check_current_protected_environment as current
from scripts import v13_xlsx_phase_b_candidate_evidence as evidence
from scripts import v13_xlsx_phase_b_resolution as phase_b
from scripts import v13_xlsx_successor_freeze as freeze
from scripts.v10c_t0_ml_environment_contract import verify_wheelhouse

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "docs/v13/V13_XLSX_SUCCESSOR_ISOLATED_REALIZATION_AUTHORIZATION.json"
AUTH_BLOB = "a966fa47871285453dfdd9ed7696c1b03bea8bc4"
BRANCH = "v13-conditional-cross-sectional-short-horizon"
PREDECESSOR = "511d19e057f7859955671113794191c7a270c7cd"
SHA = re.compile(r"[0-9a-f]{40}\Z")


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def reviewed_bytes(path: Path) -> bytes:
    rel = path.relative_to(ROOT).as_posix()
    raw = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
                         capture_output=True, check=True).stdout
    worktree_blob = subprocess.run(["git", "-C", str(ROOT), "hash-object", "--path", rel,
                                    "--", str(path)], capture_output=True,
                                   check=True, text=True).stdout.strip()
    require(worktree_blob == blob(raw), "WORKTREE_BLOB_MISMATCH")
    return raw


def unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def authorization() -> dict:
    raw = reviewed_bytes(AUTH)
    require(blob(raw) == AUTH_BLOB, "AUTHORIZATION_BLOB_MISMATCH")
    value = json.loads(raw, object_pairs_hook=unique)
    # Exact schema and bindings are fixed by the reviewed repository artifact.
    expected = {
        "schema_version": "V13_XLSX_SUCCESSOR_ISOLATED_REALIZATION_AUTHORIZATION_V1",
        "status": "AUTHORIZED", "authoritative_branch": BRANCH, "github_issue": 105,
        "predecessor_issue": 104, "predecessor_gpt_result": "PASS",
        "predecessor_gpt_reviewed_sha": PREDECESSOR,
        "predecessor_gpt_review_comment_id": 5847530715,
        "human_approval_comment_id": 5847574276,
        "human_approval_text": "V13 XLSX successor の isolated candidate environment realization を承認します．既存の固定済み29-wheelのみを使用し，package-index再アクセス，active .venv-real-execution の変更，successor promotion は承認しません．",
        "candidate_lock": {"path": freeze.LOCK.name, "git_blob_sha1": "dfe440dc6398dc3b1a2dd31c81047e4209e61536", "sha256": freeze.LOCK_SHA256, "package_count": 29},
        "freeze_record": {"path": freeze.RECORD.relative_to(ROOT).as_posix(), "git_blob_sha1": "ed302b9530290411ac1b7ee2453748fe7f7a1d30", "source_candidate_sha256": freeze.SOURCE_CANDIDATE_SHA256, "source_safe_evidence_sha256": freeze.SOURCE_EVIDENCE_SHA256, "wheel_count": 29},
        "current_authority_lock": {"path": current.CURRENT_AUTHORITY_LOCK_PATH, "git_blob_sha1": current.CURRENT_AUTHORITY_LOCK_BLOB_SHA1, "sha256": current.CURRENT_AUTHORITY_LOCK_SHA256, "package_count": 27},
        "source_phase_b_authorization_consumed": True,
        "source_phase_b_authorization_reusable": False,
        "one_shot": True, "consumed": False, "reusable_after_durable_boundary": False,
        "existing_fixed_29_wheels_only": True,
        "package_index_resolution_allowed": False, "network_download_allowed": False,
        "active_venv_real_execution_mutation_allowed": False,
        "current_authority_change_allowed": False, "successor_promotion_allowed": False,
        "isolated_candidate_realization_allowed": True,
        "isolated_candidate_readiness_allowed": True,
        "jpx_yahoo_requests_allowed": False, "private_t1_reads_allowed": False,
        "selected_500_allowed": False,
        "model_fit_backtest_aq_paper_live_trading_allowed": False,
    }
    require(value == expected and set(value) == set(expected), "AUTHORIZATION_INVALID")
    for key, path in (("candidate_lock", freeze.LOCK), ("freeze_record", freeze.RECORD),
                      ("current_authority_lock", current.CURRENT_LOCK_PATH)):
        require(blob(reviewed_bytes(path)) == value[key]["git_blob_sha1"], "BOUND_BLOB_MISMATCH")
    require(hashlib.sha256(reviewed_bytes(freeze.LOCK)).hexdigest() == freeze.LOCK_SHA256,
            "CANDIDATE_LOCK_MISMATCH")
    require(hashlib.sha256(reviewed_bytes(current.CURRENT_LOCK_PATH)).hexdigest() == current.CURRENT_AUTHORITY_LOCK_SHA256,
            "CURRENT_LOCK_MISMATCH")
    freeze.validate_freeze()
    return value


def reviewed_pass(head: str) -> bool:
    result = subprocess.run(["gh", "api", "repos/ta1k1-arakawa/stock-analyzer/issues/105/comments",
                             "--paginate"], capture_output=True, check=True)
    comments = json.loads(result.stdout)
    return any(c.get("user", {}).get("login") == "ta1k1-arakawa" and
               {"MODEL=GPT-5.6_SOL", "MODE=GPT_EXACT_SHA_INDEPENDENT_REVIEW",
                "GITHUB_ISSUE=105", f"REVIEWED_SHA={head}", "RESULT=PASS"}
               <= set(c.get("body", "").splitlines()) for c in comments)


def preflight(head: str) -> dict:
    require(sys.platform == "win32", "DIRECT_WINDOWS_REQUIRED")
    require(bool(SHA.fullmatch(head)), "EXECUTION_HEAD_INVALID")
    require(git("remote", "get-url", "origin").rstrip("/") in {
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "https://github.com/ta1k1-arakawa/stock-analyzer",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
    }, "REPOSITORY_MISMATCH")
    require(git("branch", "--show-current") == BRANCH, "BRANCH_MISMATCH")
    require(git("rev-parse", "HEAD") == head, "HEAD_MISMATCH")
    require(git("status", "--porcelain", "--untracked-files=all") == "", "DIRTY_WORKTREE")
    require(git("ls-remote", "origin", f"refs/heads/{BRANCH}") ==
            f"{head}\trefs/heads/{BRANCH}", "REMOTE_HEAD_MISMATCH")
    require(git("merge-base", PREDECESSOR, head) == PREDECESSOR,
            "PREDECESSOR_NOT_ANCESTOR")
    require(reviewed_pass(head), "GPT_EXACT_SHA_PASS_MISSING")
    auth = authorization()
    for item in (AUTH, freeze.LOCK, freeze.RECORD, current.CURRENT_LOCK_PATH):
        rel = item.relative_to(ROOT).as_posix()
        require(blob(reviewed_bytes(item)) == blob(subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{head}:{rel}"],
            capture_output=True, check=True).stdout), "WORKTREE_BLOB_MISMATCH")
    require(current.run_current_readiness()["CURRENT_ENVIRONMENT_READY"],
            "CURRENT_AUTHORITY_NOT_READY")
    return auth


def canonical_candidate_root() -> Path:
    base = phase_b._local_state_base()
    return base / "stock-analyzer" / "protected-execution" / "v13-xlsx-isolated-candidate" / AUTH_BLOB


def safe_existing(path: Path, directory: bool) -> None:
    info = os.lstat(path)
    require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
            "REPARSE_PATH_BLOCKED")
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
            "DURABLE_STATE_TYPE_INVALID")


def precheck_paths(candidate_root: Path, wheelhouse: Path) -> None:
    require(candidate_root.is_absolute() and wheelhouse.is_absolute(), "PATH_INVALID")
    require(not os.path.lexists(candidate_root), "ONE_SHOT_ALREADY_STARTED")
    for target in (candidate_root, wheelhouse):
        require(".." not in target.parts and "." not in target.parts, "PATH_INVALID")
        node = target.parent if target == candidate_root else target
        while True:
            if os.path.lexists(node):
                safe_existing(node, True)
            if node == node.parent:
                break
            node = node.parent
    real_root = Path(os.path.realpath(candidate_root))
    repo = Path(os.path.realpath(ROOT))
    active = Path(os.path.realpath(current.CANONICAL_VENV_DIR))
    require(not real_root.is_relative_to(repo) and not repo.is_relative_to(real_root) and
            not real_root.is_relative_to(active) and not active.is_relative_to(real_root),
            "GOVERNED_PATH_OVERLAP")


def verified_wheels() -> tuple[Path, tuple[dict, ...]]:
    phase_root = evidence._safe_root()
    # This rechecks the consumed attempt, candidate, receipt and all wheel metadata.
    _, source_hash = evidence.export_evidence()
    require(source_hash == freeze.SOURCE_EVIDENCE_SHA256, "SOURCE_EVIDENCE_MISMATCH")
    wheelhouse = phase_root / "wheelhouse"
    record = json.loads(freeze.RECORD.read_bytes(), object_pairs_hook=unique)
    wheels = verify_wheelhouse(wheelhouse, record["resolved_wheels"])
    require(len(wheels) == 29, "WHEEL_COUNT_MISMATCH")
    return wheelhouse, wheels


def validate_installed(records: list[dict], expected: dict[str, str]) -> None:
    require(type(records) is list and len(records) == 29, "INSTALLED_COUNT_MISMATCH")
    actual = {}
    for item in records:
        require(type(item) is dict and set(item) == {"name", "version"} and
                type(item["name"]) is str and type(item["version"]) is str,
                "INSTALLED_METADATA_INVALID")
        name = current._normalize_package_name(item["name"])
        require(name not in actual, "INSTALLED_DUPLICATE")
        actual[name] = item["version"]
    require(actual == expected and actual.get("openpyxl") == "3.1.5" and
            actual.get("et-xmlfile") == "2.0.0", "INSTALLED_SET_MISMATCH")


CHILD = r'''
import importlib.metadata as md
import json, os, platform, sys, sysconfig
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from scripts.v13_xlsx_readiness import probe_production_xlsx_route
records = [{"name": d.metadata["Name"], "version": d.version} for d in md.distributions()]
ready = probe_production_xlsx_route()
print(json.dumps({"records": records, "ready": ready, "executable": sys.executable,
                  "isolated": sys.flags.isolated == 1, "no_user_site": sys.flags.no_user_site == 1,
                  "python_version": platform.python_version(), "platform": sysconfig.get_platform(),
                  "pythonpath_absent": "PYTHONPATH" not in os.environ}, sort_keys=True))
'''


def child_environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PIP_") and
           k.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE",
                             "PYTHONSTARTUP", "VIRTUAL_ENV"}}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def inspect_candidate(interpreter: Path, expected: dict[str, str]) -> None:
    result = subprocess.run([str(interpreter), "-I", "-B", "-c", CHILD, str(ROOT)],
                            capture_output=True, text=True, env=child_environment(), check=False)
    require(result.returncode == 0 and not result.stderr.strip(), "ISOLATED_OBSERVER_FAILED")
    value = json.loads(result.stdout, object_pairs_hook=unique)
    require(set(value) == {"records", "ready", "executable", "isolated", "no_user_site",
                           "python_version", "platform", "pythonpath_absent"}, "OBSERVER_SCHEMA_INVALID")
    require(value["executable"].casefold() == str(interpreter).casefold() and
            value["isolated"] is True and value["no_user_site"] is True and
            value["pythonpath_absent"] is True and value["python_version"] == "3.12.10" and
            value["platform"] == "win-amd64", "OBSERVER_IDENTITY_INVALID")
    validate_installed(value["records"], expected)
    require(value["ready"] is True, "PRODUCTION_XLSX_PROBE_FAILED")


def realize(head: str) -> dict:
    auth = preflight(head)
    candidate_root = canonical_candidate_root()
    wheelhouse, wheels = verified_wheels()
    precheck_paths(candidate_root, wheelhouse)
    expected = dict((w["name"], w["version"]) for w in wheels)
    runtime = Path(sys._base_executable).resolve(strict=True)
    require(runtime.is_file() and runtime != current.CANONICAL_INTERPRETER.resolve() and
            sys.version_info[:3] == (3, 12, 10), "CANONICAL_RUNTIME_INVALID")
    # All checks above are pre-gate. Exclusive mkdir is the one-shot boundary.
    phase_b._prepare_operation_parent(candidate_root)
    precheck_paths(candidate_root, wheelhouse)
    candidate_root.mkdir()
    phase_b._write_json(candidate_root / "attempt.json", {
        "schema_version": "V13_XLSX_ISOLATED_REALIZATION_ATTEMPT_V1",
        "status": "CONSUMED_PENDING", "execution_head": head,
        "authorization_blob": AUTH_BLOB, "one_shot": True, "reusable": False})
    candidate = candidate_root / "candidate-venv"
    subprocess.run([str(runtime), "-I", "-m", "venv", "--without-pip", str(candidate)],
                   capture_output=True, env=child_environment(), check=True)
    interpreter = candidate / "Scripts" / "python.exe"
    require(interpreter.is_file(), "CANDIDATE_INTERPRETER_MISSING")
    verify_wheelhouse(wheelhouse, wheels)
    # pip's --python supports an empty venv; its only install inputs are the 29 frozen wheels.
    argv = [str(current.CANONICAL_INTERPRETER), "-I", "-m", "pip", "--python", str(candidate),
            "install", "--no-index", "--no-deps",
            "--no-cache-dir", "--disable-pip-version-check", "--no-input", "--no-build-isolation",
            *[str(wheelhouse / w["filename"]) for w in wheels]]
    subprocess.run(argv, capture_output=True, env=child_environment(), check=True)
    inspect_candidate(interpreter, expected)
    require(current.run_current_readiness()["CURRENT_ENVIRONMENT_READY"],
            "CURRENT_AUTHORITY_POSTCHECK_FAILED")
    receipt = {"schema_version": "V13_XLSX_ISOLATED_REALIZATION_RECEIPT_V1",
               "status": "PASS_CONSUMED", "execution_head": head,
               "authorization_blob": AUTH_BLOB,
               "candidate_lock_sha256": freeze.LOCK_SHA256,
               "wheel_manifest_sha256": freeze.MANIFEST_SHA256,
               "source_safe_evidence_sha256": freeze.SOURCE_EVIDENCE_SHA256,
               "package_count": 29, "wheel_count": 29, "installed_package_count": 29,
               "production_xlsx_readiness": "PASS", "network_requests": 0,
               "wheel_downloads": 0, "active_environment_mutated": False,
               "successor_promoted": False, "authorization_reusable": False}
    phase_b._write_json(candidate_root / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-head")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"rehearsal": True, "network_requests": 0, "wheel_downloads": 0,
                          "package_installations": 0, "environment_mutations": 0,
                          "durable_candidate_creation": 0, "private_reads": 0,
                          "jpx_yahoo_requests": 0}, sort_keys=True))
        return 0
    stage = "PRE_GATE_FAILURE"
    try:
        require(args.execution_head is not None, "EXPLICIT_EXECUTION_HEAD_REQUIRED")
        # The boundary status is determined by the durable root, including crashes before marker publication.
        result = realize(args.execution_head)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        try:
            if sys.platform == "win32" and os.path.lexists(canonical_candidate_root()):
                stage = "CONSUMED_REALIZATION_FAILURE"
        except Exception:
            pass
        reason = str(error) if type(error) is ValueError else "REALIZATION_FAILED"
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", reason):
            reason = "REALIZATION_FAILED"
        print(json.dumps({"status": "FAIL", "failure_class": stage, "reason": reason,
                          "authorization_reusable": stage == "PRE_GATE_FAILURE"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
