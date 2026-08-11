"""V5-B calibration input preflight: the R1 input-provenance / byte-binding
layer only (``V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md`` §3.2,
§13 R1). See ``V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_SPEC.md`` for the full
contract.

This module does **not** perform R2 canonical OHLCV parsing (§3.3) and does
**not** execute calibration. It never JSON-parses a raw Yahoo payload body,
never inspects OHLCV values, and never calls ``parse_ticker_observations``
or ``run_data_quality_calibration``. Its only questions are: does the
single fixed cache root exist; does its manifest match the pinned hash and
strictly validate; and do the manifest-designated 300 payload files exist,
at their declared byte length, at their declared SHA-256 -- nothing else.

There is exactly **one** filesystem-capable entry point in this module,
full stop -- not merely in its ``__all__``-advertised public surface, but
among every callable this module defines at module level, exported or
not: ``run_production_v5b_calibration_input_preflight``. It:

1. validates the exact human-gate confirmation token FIRST -- before any
   other check, before any filesystem access;
2. only then verifies that the caller-supplied ``implementation_git_commit``
   equals this repository's actual Git HEAD, and that the on-disk bytes of
   every file in ``_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS`` (which
   includes the reused calibration-core dependency,
   ``src/v8b_data_quality_calibration.py``, since this module imports and
   executes ``validate_v5b_manifest_provenance()`` from it and reads its
   fixed expected-provenance constants from it) exactly match what is
   committed at that HEAD (so dirty local edits cannot execute while
   claiming a reviewed, committed HEAD);
3. only then reads the single fixed ``V5B_CACHE_ROOT``.

There is no parameter, anywhere in this module or in the CLI that wraps
it, that accepts an alternate cache path, manifest path, input directory,
or dataset. The byte-binding logic that actually walks ``V5B_CACHE_ROOT``
is defined as a **local closure nested inside**
``run_production_v5b_calibration_input_preflight`` itself -- it is not a
module-level name at all, has no independent existence outside a call to
the gated entry point, and cannot be imported, monkeypatched onto, or
invoked separately from it. The only module-level helpers besides the
gated entry point are the Git/repository-verification functions
(``_resolve_actual_git_head``, ``_read_committed_bytes``,
``_verify_implementation_matches_repository_head``); none of them reads
the V5-B cache -- they only compare working-tree bytes of the fixed,
named implementation files above against what Git has committed, which is
a repository-provenance check, not V5-B cache access.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import v8b_data_quality_calibration as _v8b_calibration_module
except ImportError:  # pragma: no cover - import-style fallback, mirrors sibling modules
    import v8b_data_quality_calibration as _v8b_calibration_module

# Bound directly: stable across the whole test suite, never monkeypatched.
EXPECTED_V5B_TICKER_COUNT = _v8b_calibration_module.EXPECTED_V5B_TICKER_COUNT
STUDY = _v8b_calibration_module.STUDY
V8BCalibrationBlocked = _v8b_calibration_module.V8BCalibrationBlocked
canonical_json_bytes = _v8b_calibration_module.canonical_json_bytes
sha256_hex = _v8b_calibration_module.sha256_hex
validate_v5b_manifest_provenance = _v8b_calibration_module.validate_v5b_manifest_provenance

# NOTE: EXPECTED_V5B_MANIFEST_SHA256 and EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256
# are deliberately NOT copied into module-level names here. They are
# content-derived pins that validate_v5b_manifest_provenance() reads
# dynamically from _v8b_calibration_module's own namespace at call time, and
# tests monkeypatch that same namespace attribute (see
# tests/test_v8b_data_quality_calibration.py's existing pattern) to exercise
# this preflight against synthetic fixtures. A plain `from ... import NAME`
# would have captured a separate, unpatchable copy at import time and made
# this module's own reported "expected_*" fields silently diverge from what
# validate_v5b_manifest_provenance() actually checked.

# ---------------------------------------------------------------------------
# Fixed production input (§1). Tests monkeypatch V5B_CACHE_ROOT itself to a
# temporary synthetic fixture; this module never derives a cache root from
# any CLI option, environment variable, or caller-supplied argument.
# ---------------------------------------------------------------------------

FIXED_V5B_CACHE_ROOT_WINDOWS_PATH = r"C:\taiki\hobbies\v5-b-evaluation-cache-retry1"
V5B_CACHE_ROOT: Path = Path(FIXED_V5B_CACHE_ROOT_WINDOWS_PATH)

_MANIFEST_FILENAME = "cache_manifest.json"

# ---------------------------------------------------------------------------
# Repository / Git-HEAD binding (finding 2). Tests monkeypatch _REPO_ROOT to
# a temporary synthetic Git repository, exactly as V5B_CACHE_ROOT is
# monkeypatched, rather than exposing either as a public parameter.
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Every file whose on-disk bytes must exactly match what is committed at
# the verified actual Git HEAD before real V5-B cache access is permitted.
# src/v8b_data_quality_calibration.py is included because this module
# imports and executes validate_v5b_manifest_provenance() from it and
# reads its fixed EXPECTED_V5B_MANIFEST_SHA256 / EXPECTED_V5B_PAYLOAD_
# HASH_LIST_SHA256 / EXPECTED_V5B_TICKER_COUNT constants from it -- a dirty
# copy of that file could silently change what "PASS" means.
_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS: tuple[str, ...] = (
    "src/v8b_v5b_calibration_input_preflight.py",
    "scripts/preflight_v8b_v5b_calibration_input.py",
    "V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_SPEC.md",
    "src/v8b_data_quality_calibration.py",
)

_GIT_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Human gate (§2). Matching this token is necessary but not sufficient: its
# mere presence in this source file does not authorize real execution. This
# implementation task does not invoke run_production_v5b_calibration_input_
# preflight() against the real fixed cache root; only tests exercise it,
# and only with V5B_CACHE_ROOT and _REPO_ROOT monkeypatched to synthetic
# fixtures.
# ---------------------------------------------------------------------------

PREFLIGHT_GATE_CONFIRMATION = "V5B_CALIBRATION_INPUT_PREFLIGHT_GATE"

# Single outward blocker reason (§5), matching the exact string already
# recognized as an R1 blocker by src/v8b_data_quality_calibration.py's
# _RUN_INVALID_REASON_FLAGS.
PREFLIGHT_BLOCKER = "V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED"

PREFLIGHT_ROLE = "R1_V5B_CALIBRATION_INPUT_PREFLIGHT"
PREFLIGHT_RESULT_SCHEMA_VERSION = "V5B_CALIBRATION_INPUT_PREFLIGHT_RESULT_V1"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "study",
        "role",
        "status",
        "detail_reason",
        "implementation_git_commit",
        "expected_manifest_sha256",
        "observed_manifest_sha256",
        "expected_payload_hash_list_sha256",
        "observed_payload_hash_list_sha256",
        "expected_payload_count",
        "checked_payload_count",
        "byte_count_mismatch_count",
        "sha256_mismatch_count",
        "missing_or_unreadable_count",
        "run_started_utc",
        "run_completed_utc",
        "artifact_self_hash",
    }
)

_PREFLIGHT_DETAIL_RE = re.compile(
    r"^(?:"
    r"PREFLIGHT_GATE_CONFIRMATION_REQUIRED|IMPLEMENTATION_COMMIT_INVALID|"
    r"GIT_HEAD_UNRESOLVABLE|GIT_REPOSITORY_IDENTITY_MISMATCH|"
    r"IMPLEMENTATION_COMMIT_HEAD_MISMATCH|IMPLEMENTATION_FILE_UNVERIFIABLE|"
    r"IMPLEMENTATION_FILE_DIRTY|CACHE_ROOT_INACCESSIBLE|CACHE_ROOT_NOT_A_DIRECTORY|"
    r"CACHE_ROOT_REPARSE_POINT|MANIFEST_UNREADABLE|MANIFEST_NOT_REGULAR|MANIFEST_REPARSE_POINT|"
    r"MANIFEST_PATH_ESCAPE_DETECTED|DESIGNATED_PAYLOAD_COUNT_MISMATCH|"
    r"PAYLOAD_PATH_RESOLUTION_FAILED|PAYLOAD_PATH_ESCAPE_DETECTED|"
    r"PAYLOAD_REPARSE_POINT|PAYLOAD_NOT_REGULAR|PAYLOAD_READ_FAILED|"
    r"PAYLOAD_BINDING_FAILED)"
    r"$"
)

_MANIFEST_PROVENANCE_INVALID_PREFIX = "MANIFEST_PROVENANCE_INVALID:"


class V5BCalibrationInputPreflightBlocked(RuntimeError):
    """Fail-closed error for every preflight blocking condition.

    ``reason`` is always the single generic ``PREFLIGHT_BLOCKER`` constant
    (§5's exact required blocker token), so downstream calibration
    run-validity classification (R1) has exactly one outward reason.
    ``detail`` carries a safe, structural, non-identity-revealing sub-code
    (never a ticker or path) for diagnostics and tests. ``result``, when
    present, is the same safe aggregate-only dict a caller would have
    received on PASS (§6), with ``status="BLOCK"``.
    """

    def __init__(self, detail: str, result: Mapping[str, Any] | None = None) -> None:
        self.reason = PREFLIGHT_BLOCKER
        self.detail = detail
        self.result = dict(result) if result is not None else None
        super().__init__(detail)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finalize(fields: dict[str, Any]) -> dict[str, Any]:
    result = dict(fields)
    result["artifact_self_hash"] = sha256_hex(canonical_json_bytes(fields))
    return result


def _canonical_block_result(
    detail: str,
    *,
    implementation_git_commit: str | None = None,
    observed_manifest_sha256: str | None = None,
    observed_payload_hash_list_sha256: str | None = None,
    checked_payload_count: int = 0,
    byte_count_mismatch_count: int = 0,
    sha256_mismatch_count: int = 0,
    missing_or_unreadable_count: int = 0,
) -> dict[str, Any]:
    started = _utc_now_iso()
    fields = {
        "schema_version": PREFLIGHT_RESULT_SCHEMA_VERSION,
        "study": STUDY,
        "role": PREFLIGHT_ROLE,
        "status": "BLOCK",
        "detail_reason": detail,
        "implementation_git_commit": implementation_git_commit,
        "expected_manifest_sha256": _v8b_calibration_module.EXPECTED_V5B_MANIFEST_SHA256,
        "observed_manifest_sha256": observed_manifest_sha256,
        "expected_payload_hash_list_sha256": _v8b_calibration_module.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
        "observed_payload_hash_list_sha256": observed_payload_hash_list_sha256,
        "expected_payload_count": EXPECTED_V5B_TICKER_COUNT,
        "checked_payload_count": checked_payload_count,
        "byte_count_mismatch_count": byte_count_mismatch_count,
        "sha256_mismatch_count": sha256_mismatch_count,
        "missing_or_unreadable_count": missing_or_unreadable_count,
        "run_started_utc": started,
        "run_completed_utc": started,
    }
    return _finalize(fields)


def _verify_artifact_self_hash(result: Mapping[str, Any]) -> None:
    supplied = result.get("artifact_self_hash")
    if not isinstance(supplied, str) or _SHA256_RE.fullmatch(supplied) is None:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_SELF_HASH_INVALID")
    fields = dict(result)
    del fields["artifact_self_hash"]
    if sha256_hex(canonical_json_bytes(fields)) != supplied:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_SELF_HASH_MISMATCH")


def _is_exact_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return True


def validate_preflight_result_semantics(
    result: Mapping[str, Any],
    *,
    expected_implementation_git_commit: str,
) -> None:
    """Accept only a canonical, self-hashed, semantically valid artifact
    that is bound to a caller-trusted implementation commit.

    ``expected_implementation_git_commit`` is a required keyword-only
    argument with no default: the persisted artifact's own
    ``implementation_git_commit`` field must never be its own authority for
    which implementation commit was reviewed. An external, independently
    trusted commit must always be supplied, and whenever the artifact
    records a commit at all (any state after Git verification has run, and
    always for a PASS), it must equal that trusted value exactly. Legitimate
    early BLOCK states raised before Git verification ever ran (e.g. a
    rejected confirmation token, or a malformed caller-supplied commit) may
    still record ``implementation_git_commit=None`` -- there was nothing to
    bind yet -- but the trusted argument itself is still mandatory to
    supply, and must itself be well-formed.

    This is intentionally an independent acceptance API. It never invokes
    the production preflight constructor and does not perform filesystem or
    network I/O.
    """

    if (
        not isinstance(expected_implementation_git_commit, str)
        or _COMMIT_RE.fullmatch(expected_implementation_git_commit) is None
    ):
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_EXPECTED_COMMIT_INVALID")

    if not isinstance(result, Mapping) or set(result) != _ARTIFACT_KEYS:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_SCHEMA_INVALID")
    if result["schema_version"] != PREFLIGHT_RESULT_SCHEMA_VERSION:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_SCHEMA_INVALID")
    if result["study"] != STUDY or result["role"] != PREFLIGHT_ROLE:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_SCHEMA_INVALID")
    if result["status"] not in {"PASS", "BLOCK"}:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATUS_INVALID")
    detail = result["detail_reason"]
    if result["status"] == "PASS":
        if detail is not None:
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")
    elif isinstance(detail, str) and detail.startswith(_MANIFEST_PROVENANCE_INVALID_PREFIX):
        inner_reason = detail[len(_MANIFEST_PROVENANCE_INVALID_PREFIX) :]
        if inner_reason not in _v8b_calibration_module._RECOGNIZED_MANIFEST_BLOCKER_REASONS:
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_DETAIL_INVALID")
    elif not isinstance(detail, str) or _PREFLIGHT_DETAIL_RE.fullmatch(detail) is None:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_DETAIL_INVALID")

    commit = result["implementation_git_commit"]
    if commit is not None and (not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None):
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_COMMIT_INVALID")
    if commit is not None and commit != expected_implementation_git_commit:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_COMMIT_MISMATCH")

    expected_manifest = _v8b_calibration_module.EXPECTED_V5B_MANIFEST_SHA256
    expected_payload_list = _v8b_calibration_module.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256
    if result["expected_manifest_sha256"] != expected_manifest:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_PROVENANCE_INVALID")
    if result["expected_payload_hash_list_sha256"] != expected_payload_list:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_PROVENANCE_INVALID")
    for key in ("expected_manifest_sha256", "expected_payload_hash_list_sha256"):
        if not isinstance(result[key], str) or _SHA256_RE.fullmatch(result[key]) is None:
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_PROVENANCE_INVALID")
    for key in ("observed_manifest_sha256", "observed_payload_hash_list_sha256"):
        value = result[key]
        if value is not None and (not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_PROVENANCE_INVALID")
    if (
        result["observed_manifest_sha256"] is not None
        and result["observed_manifest_sha256"] != expected_manifest
        and not (isinstance(detail, str) and detail.startswith("MANIFEST_PROVENANCE_INVALID:"))
    ):
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_PROVENANCE_INVALID")
    if (
        result["observed_payload_hash_list_sha256"] is not None
        and result["observed_payload_hash_list_sha256"] != expected_payload_list
    ):
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_PROVENANCE_INVALID")
    for key in (
        "expected_payload_count",
        "checked_payload_count",
        "byte_count_mismatch_count",
        "sha256_mismatch_count",
        "missing_or_unreadable_count",
    ):
        if not _is_exact_nonnegative_int(result[key]):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_COUNT_INVALID")
    if result["expected_payload_count"] != EXPECTED_V5B_TICKER_COUNT:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_COUNT_INVALID")
    if result["checked_payload_count"] > result["expected_payload_count"]:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_COUNT_INVALID")
    if not _is_valid_utc_timestamp(result["run_started_utc"]):
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_TIMESTAMP_INVALID")
    if not _is_valid_utc_timestamp(result["run_completed_utc"]):
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_TIMESTAMP_INVALID")
    if result["run_completed_utc"] < result["run_started_utc"]:
        raise V5BCalibrationInputPreflightBlocked("ARTIFACT_TIMESTAMP_INVALID")

    pre_manifest_details = {
        "PREFLIGHT_GATE_CONFIRMATION_REQUIRED",
        "IMPLEMENTATION_COMMIT_INVALID",
        "GIT_HEAD_UNRESOLVABLE",
        "GIT_REPOSITORY_IDENTITY_MISMATCH",
        "IMPLEMENTATION_COMMIT_HEAD_MISMATCH",
        "IMPLEMENTATION_FILE_UNVERIFIABLE",
        "IMPLEMENTATION_FILE_DIRTY",
        "CACHE_ROOT_INACCESSIBLE",
        "CACHE_ROOT_NOT_A_DIRECTORY",
        "CACHE_ROOT_REPARSE_POINT",
        "MANIFEST_UNREADABLE",
        "MANIFEST_NOT_REGULAR",
        "MANIFEST_REPARSE_POINT",
        "MANIFEST_PATH_ESCAPE_DETECTED",
    }
    if detail in pre_manifest_details:
        if any(
            result[key] is not None
            for key in ("observed_manifest_sha256", "observed_payload_hash_list_sha256")
        ) or any(
            result[key] != 0
            for key in (
                "checked_payload_count",
                "byte_count_mismatch_count",
                "sha256_mismatch_count",
                "missing_or_unreadable_count",
            )
        ):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")
    elif isinstance(detail, str) and detail.startswith("MANIFEST_PROVENANCE_INVALID:"):
        if result["observed_manifest_sha256"] is None or result["observed_payload_hash_list_sha256"] is not None:
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")
        if any(
            result[key] != 0
            for key in (
                "checked_payload_count",
                "byte_count_mismatch_count",
                "sha256_mismatch_count",
                "missing_or_unreadable_count",
            )
        ):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")
    elif detail in {
        "DESIGNATED_PAYLOAD_COUNT_MISMATCH",
        "PAYLOAD_PATH_RESOLUTION_FAILED",
        "PAYLOAD_PATH_ESCAPE_DETECTED",
        "PAYLOAD_REPARSE_POINT",
        "PAYLOAD_NOT_REGULAR",
        "PAYLOAD_READ_FAILED",
        "PAYLOAD_BINDING_FAILED",
    }:
        if (
            result["observed_manifest_sha256"] != expected_manifest
            or result["observed_payload_hash_list_sha256"] != expected_payload_list
        ):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")

    _verify_artifact_self_hash(result)

    if result["status"] == "PASS":
        if (
            commit is None
            or commit != expected_implementation_git_commit
            or result["observed_manifest_sha256"] != expected_manifest
            or result["observed_payload_hash_list_sha256"] != expected_payload_list
            or result["checked_payload_count"] != EXPECTED_V5B_TICKER_COUNT
            or result["byte_count_mismatch_count"] != 0
            or result["sha256_mismatch_count"] != 0
            or result["missing_or_unreadable_count"] != 0
        ):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")
    else:
        if (
            result["observed_manifest_sha256"] == expected_manifest
            and result["observed_payload_hash_list_sha256"] == expected_payload_list
            and result["checked_payload_count"] == EXPECTED_V5B_TICKER_COUNT
            and result["byte_count_mismatch_count"] == 0
            and result["sha256_mismatch_count"] == 0
            and result["missing_or_unreadable_count"] == 0
        ):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")
        if detail == "PAYLOAD_BINDING_FAILED" and (
            result["checked_payload_count"] + result["missing_or_unreadable_count"]
            != EXPECTED_V5B_TICKER_COUNT
        ):
            raise V5BCalibrationInputPreflightBlocked("ARTIFACT_STATE_INVALID")


# ---------------------------------------------------------------------------
# Repository / Git-HEAD binding (finding 2)
# ---------------------------------------------------------------------------


def _sanitized_git_environment() -> dict[str, str]:
    """Return an environment that cannot redirect Git repository routing."""

    return {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}


def _resolve_actual_git_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_sanitized_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    head = completed.stdout.strip()
    if not _COMMIT_RE.match(head):
        return None
    return head


def _resolve_git_top_level(repo_root: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_sanitized_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        return Path(value).resolve(strict=True)
    except OSError:
        return None


def _read_committed_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_sanitized_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _verify_implementation_matches_repository_head(
    *,
    repo_root: Path,
    implementation_git_commit: str,
    relevant_relative_paths: Sequence[str] = _RELEVANT_IMPLEMENTATION_RELATIVE_PATHS,
) -> str:
    """Require ``implementation_git_commit`` to equal this repository's
    actual Git HEAD, and every relevant implementation file's on-disk bytes
    to exactly match what is committed at that HEAD. Returns the verified
    HEAD on success; raises ``V5BCalibrationInputPreflightBlocked`` (never
    returning a partial/ambiguous result) on any failure, including
    detached/unresolvable/malformed Git state.
    """

    actual_head = _resolve_actual_git_head(repo_root)
    if actual_head is None:
        raise V5BCalibrationInputPreflightBlocked(
            "GIT_HEAD_UNRESOLVABLE", result=_canonical_block_result("GIT_HEAD_UNRESOLVABLE")
        )
    try:
        expected_root = repo_root.resolve(strict=True)
    except OSError:
        raise V5BCalibrationInputPreflightBlocked(
            "GIT_REPOSITORY_IDENTITY_MISMATCH",
            result=_canonical_block_result("GIT_REPOSITORY_IDENTITY_MISMATCH"),
        )
    actual_root = _resolve_git_top_level(repo_root)
    if actual_root is None or actual_root != expected_root:
        raise V5BCalibrationInputPreflightBlocked(
            "GIT_REPOSITORY_IDENTITY_MISMATCH",
            result=_canonical_block_result("GIT_REPOSITORY_IDENTITY_MISMATCH"),
        )
    if implementation_git_commit != actual_head:
        raise V5BCalibrationInputPreflightBlocked(
            "IMPLEMENTATION_COMMIT_HEAD_MISMATCH",
            result=_canonical_block_result("IMPLEMENTATION_COMMIT_HEAD_MISMATCH"),
        )
    for relative_path in relevant_relative_paths:
        committed_bytes = _read_committed_bytes(repo_root, actual_head, relative_path)
        if committed_bytes is None:
            raise V5BCalibrationInputPreflightBlocked(
                "IMPLEMENTATION_FILE_UNVERIFIABLE",
                result=_canonical_block_result("IMPLEMENTATION_FILE_UNVERIFIABLE"),
            )
        try:
            working_tree_bytes = (repo_root / relative_path).read_bytes()
        except OSError:
            raise V5BCalibrationInputPreflightBlocked(
                "IMPLEMENTATION_FILE_UNVERIFIABLE",
                result=_canonical_block_result("IMPLEMENTATION_FILE_UNVERIFIABLE"),
            )
        if working_tree_bytes != committed_bytes:
            raise V5BCalibrationInputPreflightBlocked(
                "IMPLEMENTATION_FILE_DIRTY", result=_canonical_block_result("IMPLEMENTATION_FILE_DIRTY")
            )
    return actual_head


def run_production_v5b_calibration_input_preflight(
    *,
    confirmation: str,
    implementation_git_commit: str,
) -> dict[str, Any]:
    """The single filesystem-capable entry point in this module (§2), full
    stop -- not merely the single *exported* one. The byte-binding logic
    that walks ``V5B_CACHE_ROOT`` is defined as a local closure nested
    inside this function's own body (see ``_walk_cache_root`` below): it
    has no module-level name, cannot be imported, monkeypatched onto, or
    invoked independently of a call to this function.

    Order is fixed and security-relevant: (1) exact confirmation token, (2)
    ``implementation_git_commit`` equals this repository's actual Git HEAD
    and every relevant implementation file -- including the reused
    calibration-core dependency, ``src/v8b_data_quality_calibration.py``
    -- is byte-identical to what is committed there, (3) only then read the
    single fixed ``V5B_CACHE_ROOT``. There is no parameter to override any
    of these. This implementation task does not invoke this function
    against the real cache; it exists so a future, separately authorized
    task can call it with the genuine human-supplied confirmation token
    against a clean, committed HEAD.
    """

    if confirmation != PREFLIGHT_GATE_CONFIRMATION:
        raise V5BCalibrationInputPreflightBlocked(
            "PREFLIGHT_GATE_CONFIRMATION_REQUIRED",
            result=_canonical_block_result("PREFLIGHT_GATE_CONFIRMATION_REQUIRED"),
        )

    if not isinstance(implementation_git_commit, str) or not _COMMIT_RE.match(implementation_git_commit):
        raise V5BCalibrationInputPreflightBlocked(
            "IMPLEMENTATION_COMMIT_INVALID", result=_canonical_block_result("IMPLEMENTATION_COMMIT_INVALID")
        )

    _verify_implementation_matches_repository_head(
        repo_root=_REPO_ROOT, implementation_git_commit=implementation_git_commit
    )

    # ------------------------------------------------------------------
    # Nested closure: the only code in this module that ever reads the
    # V5-B cache. It exists only for the duration of this call; it is not
    # a module attribute, so `preflight.<anything>` can never reach it,
    # and it cannot be exercised without first passing both gates above.
    # ------------------------------------------------------------------

    def _walk_cache_root(cache_root: Path, run_started_utc: str | None = None) -> dict[str, Any]:
        """Core preflight logic (§3, §5). The only I/O this closure
        performs is reading files under ``cache_root``: a stat/existence
        check on the root, a read of its ``cache_manifest.json``, and --
        for exactly the 300 manifest-designated payloads, never any other
        file -- an existence check, a path-containment check, a raw byte
        read, and a SHA-256/byte-count comparison against the validated
        manifest's own declared values.

        Never JSON-parses a payload body, never inspects OHLCV, never
        touches anything outside ``cache_root``. Raises
        ``V5BCalibrationInputPreflightBlocked`` on any failure (§5);
        returns a safe aggregate-only dict (§6) only on a full PASS.
        """

        run_started = run_started_utc or _utc_now_iso()
        fields: dict[str, Any] = {
            "schema_version": PREFLIGHT_RESULT_SCHEMA_VERSION,
            "study": STUDY,
            "role": PREFLIGHT_ROLE,
            "status": "BLOCK",
            "detail_reason": None,
            "implementation_git_commit": implementation_git_commit,
            "expected_manifest_sha256": _v8b_calibration_module.EXPECTED_V5B_MANIFEST_SHA256,
            "observed_manifest_sha256": None,
            "expected_payload_hash_list_sha256": _v8b_calibration_module.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
            "observed_payload_hash_list_sha256": None,
            "expected_payload_count": EXPECTED_V5B_TICKER_COUNT,
            "checked_payload_count": 0,
            "byte_count_mismatch_count": 0,
            "sha256_mismatch_count": 0,
            "missing_or_unreadable_count": 0,
            "run_started_utc": run_started,
            "run_completed_utc": None,
        }

        def block(detail: str) -> V5BCalibrationInputPreflightBlocked:
            fields["status"] = "BLOCK"
            fields["detail_reason"] = detail
            fields["run_completed_utc"] = _utc_now_iso()
            result = _finalize(fields)
            validate_preflight_result_semantics(
                result, expected_implementation_git_commit=implementation_git_commit
            )
            return V5BCalibrationInputPreflightBlocked(detail, result=result)

        def is_reparse_point(path: Path) -> bool:
            try:
                stat_result = path.lstat()
            except OSError:
                return False
            if path.is_symlink():
                return True
            return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400)

        def verify_root(path: Path) -> Path:
            # Check every existing component without following it. On
            # Windows this rejects junction/reparse redirection in parents;
            # on POSIX it rejects symlink parents as the portable equivalent.
            try:
                absolute = path.absolute()
                for component in (absolute, *absolute.parents):
                    if is_reparse_point(component):
                        raise block("CACHE_ROOT_REPARSE_POINT")
                resolved = absolute.resolve(strict=True)
                if os.name == "nt" and normalized_path(resolved) != normalized_path(absolute):
                    raise block("CACHE_ROOT_REPARSE_POINT")
            except V5BCalibrationInputPreflightBlocked:
                raise
            except OSError:
                raise block("CACHE_ROOT_INACCESSIBLE")
            try:
                if not resolved.is_dir() or is_reparse_point(resolved):
                    raise block("CACHE_ROOT_NOT_A_DIRECTORY" if not resolved.is_dir() else "CACHE_ROOT_REPARSE_POINT")
            except OSError:
                raise block("CACHE_ROOT_INACCESSIBLE")
            return resolved

        def normalized_path(value: str | Path) -> str:
            text = os.fspath(value)
            if os.name == "nt":
                if text.startswith("\\\\?\\UNC\\"):
                    text = "\\\\" + text[8:]
                elif text.startswith("\\\\?\\"):
                    text = text[4:]
            return os.path.normcase(os.path.normpath(text))

        def is_within(path: str | Path, root_path: str | Path) -> bool:
            candidate = normalized_path(path)
            root_name = normalized_path(root_path)
            try:
                return os.path.commonpath((candidate, root_name)) == root_name
            except ValueError:
                return False

        def reject_reparse_components(path: Path, root_path: Path) -> None:
            current = path
            while True:
                if is_reparse_point(current):
                    raise block("PAYLOAD_REPARSE_POINT")
                if current == root_path or current.parent == current:
                    break
                current = current.parent

        def read_verified_file(path: Path, root_path: Path, *, kind: str = "PAYLOAD") -> bytes | None:
            """Read/hash one already-designated file through one checked handle."""

            def detail(suffix: str) -> str:
                if kind == "MANIFEST":
                    return {
                        "REPARSE_POINT": "MANIFEST_REPARSE_POINT",
                        "NOT_REGULAR": "MANIFEST_NOT_REGULAR",
                        "PATH_ESCAPE_DETECTED": "MANIFEST_PATH_ESCAPE_DETECTED",
                        "READ_FAILED": "MANIFEST_UNREADABLE",
                    }[suffix]
                return "PAYLOAD_" + suffix

            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                import msvcrt

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                create_file = kernel32.CreateFileW
                create_file.argtypes = [
                    wintypes.LPCWSTR,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.HANDLE,
                ]
                create_file.restype = wintypes.HANDLE
                close_handle = kernel32.CloseHandle
                close_handle.argtypes = [wintypes.HANDLE]
                close_handle.restype = wintypes.BOOL
                get_attrs = kernel32.GetFileInformationByHandle
                get_attrs.restype = wintypes.BOOL
                get_final = kernel32.GetFinalPathNameByHandleW
                get_final.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
                get_final.restype = wintypes.DWORD
                get_file_type = kernel32.GetFileType
                get_file_type.argtypes = [wintypes.HANDLE]
                get_file_type.restype = wintypes.DWORD
                invalid_handle = wintypes.HANDLE(-1).value
                handle = create_file(
                    str(path),
                    0x80000000,  # GENERIC_READ
                    0x00000007,  # share read/write/delete
                    None,
                    3,  # OPEN_EXISTING
                    0x00200000 | 0x02000000,  # OPEN_REPARSE_POINT | BACKUP_SEMANTICS
                    None,
                )
                if handle == invalid_handle:
                    error = ctypes.get_last_error()
                    if error in (2, 3, 5, 32):
                        return None
                    raise block(detail("READ_FAILED"))
                try:
                    class _ByHandleFileInformation(ctypes.Structure):
                        _fields_ = [
                            ("file_attributes", wintypes.DWORD),
                            ("creation_time_low", wintypes.DWORD),
                            ("creation_time_high", wintypes.DWORD),
                            ("last_access_low", wintypes.DWORD),
                            ("last_access_high", wintypes.DWORD),
                            ("last_write_low", wintypes.DWORD),
                            ("last_write_high", wintypes.DWORD),
                            ("volume_serial", wintypes.DWORD),
                            ("file_size_high", wintypes.DWORD),
                            ("file_size_low", wintypes.DWORD),
                            ("number_of_links", wintypes.DWORD),
                            ("file_index_high", wintypes.DWORD),
                            ("file_index_low", wintypes.DWORD),
                        ]

                    info = _ByHandleFileInformation()
                    if not get_attrs(handle, ctypes.byref(info)):
                        raise block(detail("READ_FAILED"))
                    if get_file_type(handle) != 1:  # FILE_TYPE_DISK
                        raise block(detail("NOT_REGULAR"))
                    if info.file_attributes & 0x400:
                        raise block(detail("REPARSE_POINT"))
                    if info.file_attributes & 0x10:
                        raise block(detail("NOT_REGULAR"))
                    buffer = ctypes.create_unicode_buffer(32768)
                    length = get_final(handle, buffer, len(buffer), 0)
                    if length == 0 or not is_within(buffer.value, root_path):
                        raise block(detail("PATH_ESCAPE_DETECTED"))
                    fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
                    handle = invalid_handle
                    with os.fdopen(fd, "rb", closefd=True) as stream:
                        return stream.read()
                finally:
                    if handle != invalid_handle:
                        close_handle(handle)

            # POSIX: walk directories without following symlinks, open the
            # final file with O_NOFOLLOW, then validate and read that fd.
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            if not nofollow:
                raise block(detail("READ_FAILED"))
            relative = os.path.relpath(path, root_path)
            if relative == os.curdir or relative.startswith(".."):
                raise block(detail("PATH_ESCAPE_DETECTED"))
            directory_fd: int | None = None
            file_fd = -1
            try:
                directory_fd = os.open(root_path, flags | os.O_DIRECTORY | nofollow)
                parts = Path(relative).parts
                for part in parts[:-1]:
                    next_fd = os.open(part, flags | os.O_DIRECTORY | nofollow, dir_fd=directory_fd)
                    os.close(directory_fd)
                    directory_fd = next_fd
                file_fd = os.open(parts[-1], flags | nofollow, dir_fd=directory_fd)
                stat_result = os.fstat(file_fd)
                if not stat_result:
                    raise block(detail("READ_FAILED"))
                if not stat.S_ISREG(stat_result.st_mode):
                    raise block(detail("NOT_REGULAR"))
                proc_path = f"/proc/self/fd/{file_fd}"
                if os.path.exists(proc_path) and not is_within(os.path.realpath(proc_path), root_path):
                    raise block(detail("PATH_ESCAPE_DETECTED"))
                with os.fdopen(file_fd, "rb", closefd=True) as stream:
                    file_fd = -1
                    return stream.read()
            except FileNotFoundError:
                return None
            except V5BCalibrationInputPreflightBlocked:
                raise
            except OSError:
                return None
            finally:
                if file_fd >= 0:
                    try:
                        os.close(file_fd)
                    except OSError:
                        pass
                if directory_fd is not None:
                    try:
                        os.close(directory_fd)
                    except OSError:
                        pass

        root = Path(cache_root)
        root_resolved = verify_root(root)

        manifest_path = root_resolved / _MANIFEST_FILENAME
        if is_reparse_point(manifest_path):
            raise block("MANIFEST_REPARSE_POINT")
        try:
            manifest_bytes = read_verified_file(manifest_path, root_resolved, kind="MANIFEST")
        except OSError:
            raise block("MANIFEST_UNREADABLE")
        if manifest_bytes is None:
            raise block("MANIFEST_UNREADABLE")

        fields["observed_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()

        try:
            validated_manifest = validate_v5b_manifest_provenance(manifest_bytes)
        except V8BCalibrationBlocked as error:
            raise block("MANIFEST_PROVENANCE_INVALID:" + error.reason) from error

        fields["observed_payload_hash_list_sha256"] = validated_manifest["payload_hash_list_sha256"]

        payloads = validated_manifest["payloads"]
        if not isinstance(payloads, list) or len(payloads) != EXPECTED_V5B_TICKER_COUNT:
            raise block("DESIGNATED_PAYLOAD_COUNT_MISMATCH")

        checked_count = 0
        missing_count = 0
        byte_mismatch_count = 0
        sha_mismatch_count = 0

        for record in payloads:
            relative_path = record["relative_path"]
            candidate_path = root_resolved / relative_path
            reject_reparse_components(candidate_path, root_resolved)
            try:
                resolved_candidate = candidate_path.resolve(strict=False)
            except OSError:
                raise block("PAYLOAD_PATH_RESOLUTION_FAILED")
            if resolved_candidate != root_resolved and root_resolved not in resolved_candidate.parents:
                raise block("PAYLOAD_PATH_ESCAPE_DETECTED")
            try:
                payload_bytes = read_verified_file(candidate_path, root_resolved)
            except OSError:
                missing_count += 1
                continue
            if payload_bytes is None:
                missing_count += 1
                continue

            checked_count += 1
            if len(payload_bytes) != record["byte_count"]:
                byte_mismatch_count += 1
            if hashlib.sha256(payload_bytes).hexdigest() != record["sha256"]:
                sha_mismatch_count += 1

        fields["checked_payload_count"] = checked_count
        fields["byte_count_mismatch_count"] = byte_mismatch_count
        fields["sha256_mismatch_count"] = sha_mismatch_count
        fields["missing_or_unreadable_count"] = missing_count
        fields["run_completed_utc"] = _utc_now_iso()

        if (
            missing_count != 0
            or byte_mismatch_count != 0
            or sha_mismatch_count != 0
            or checked_count != EXPECTED_V5B_TICKER_COUNT
        ):
            fields["status"] = "BLOCK"
            fields["detail_reason"] = "PAYLOAD_BINDING_FAILED"
            raise V5BCalibrationInputPreflightBlocked("PAYLOAD_BINDING_FAILED", result=_finalize(fields))

        fields["status"] = "PASS"
        fields["detail_reason"] = None
        result = _finalize(fields)
        validate_preflight_result_semantics(
            result, expected_implementation_git_commit=implementation_git_commit
        )
        return result

    return _walk_cache_root(V5B_CACHE_ROOT)


# ---------------------------------------------------------------------------
# §3 (LOW finding): meaningful, repository-only static check. Zero V5-B
# cache access, zero network access -- reads only this module's own source
# and introspects its own public API surface.
# ---------------------------------------------------------------------------


def run_static_check() -> None:
    """Repository-only verification. Raises ``V5BCalibrationInputPreflight
    Blocked`` on any drift; returns ``None`` (passes) otherwise. Never
    touches the V5-B cache and never makes a network call.
    """

    if FIXED_V5B_CACHE_ROOT_WINDOWS_PATH != r"C:\taiki\hobbies\v5-b-evaluation-cache-retry1":
        raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_CACHE_ROOT_DRIFT")

    if PREFLIGHT_GATE_CONFIRMATION != "V5B_CALIBRATION_INPUT_PREFLIGHT_GATE":
        raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_GATE_TOKEN_DRIFT")

    production_params = set(inspect.signature(run_production_v5b_calibration_input_preflight).parameters)
    if production_params != {"confirmation", "implementation_git_commit"}:
        raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_PRODUCTION_API_SURFACE_DRIFT")

    # Scan the ENTIRE module callable surface -- every module-level name,
    # exported or not, not merely __all__ -- for a second filesystem-
    # capable bypass. The only callable defined in this module that may
    # accept a cache_root/path/manifest_path/input_dir/dataset parameter
    # is none at all: the actual cache-walking logic is a closure nested
    # inside run_production_v5b_calibration_input_preflight and therefore
    # never appears at module level in the first place.
    module_globals = globals()
    banned_names = {
        "run_v5b_calibration_input_preflight",
        "_run_v5b_calibration_input_preflight_against_root",
        "_walk_cache_root",
    }
    if banned_names & set(module_globals) or banned_names & set(__all__):
        raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED")

    forbidden_param_names = {"cache_root", "path", "manifest_path", "input_dir", "dataset"}
    for name, candidate in list(module_globals.items()):
        if name.startswith("__") or not callable(candidate) or inspect.isclass(candidate):
            continue
        if getattr(candidate, "__module__", None) != __name__:
            # Defined elsewhere (e.g. reused from the calibration core) --
            # not this module's own API surface.
            continue
        try:
            params = set(inspect.signature(candidate).parameters)
        except (TypeError, ValueError):
            continue
        if params & forbidden_param_names:
            raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED")

    if EXPECTED_V5B_TICKER_COUNT != 300:
        raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_PAYLOAD_COUNT_DRIFT")

    if validate_v5b_manifest_provenance is not _v8b_calibration_module.validate_v5b_manifest_provenance:
        raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_MANIFEST_VALIDATOR_DRIFT")

    # Scan only the functional/security-relevant code above this function's
    # own definition -- NOT this function's own body, which necessarily
    # names these same tokens as literal strings in order to check for
    # them, and would otherwise always self-match.
    source = Path(__file__).read_text(encoding="utf-8")
    functional_source = source[: source.index("\ndef run_static_check")]
    forbidden_source_tokens = [
        "parse_ticker_observations(",
        "run_data_quality_calibration(",
        "_row_invalid_reason(",
        "select_synthetic_bases(",
        "compute_global_envelope(",
        "select_policy(",
        "apply_corruption(",
        "urllib",
        "requests",
        "yfinance",
        "query1.finance.yahoo.com",
        "http://",
        "https://",
    ]
    for token in forbidden_source_tokens:
        if token in functional_source:
            raise V5BCalibrationInputPreflightBlocked("STATIC_CHECK_FORBIDDEN_SOURCE_TOKEN")


__all__ = [
    "FIXED_V5B_CACHE_ROOT_WINDOWS_PATH",
    "PREFLIGHT_BLOCKER",
    "PREFLIGHT_GATE_CONFIRMATION",
    "PREFLIGHT_RESULT_SCHEMA_VERSION",
    "PREFLIGHT_ROLE",
    "V5BCalibrationInputPreflightBlocked",
    "V5B_CACHE_ROOT",
    "validate_preflight_result_semantics",
    "run_production_v5b_calibration_input_preflight",
    "run_static_check",
]
