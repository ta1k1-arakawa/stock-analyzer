"""V8B production calibration execution adapter.

This module wires the single fixed, real V5-B cache root to the existing,
frozen, pure ``run_data_quality_calibration()`` (``src/v8b_data_quality_
calibration.py``). It performs no methodology: no grid, window, parser,
synthetic-design, defensibility, selection, or result-semantics decision is
made here. Its entire job is to securely acquire exactly the manifest and
300 designated payload byte sequences from the real cache and hand them,
unmodified, to that already-reviewed pure function.

There is exactly **one** filesystem-capable entry point in this module,
full stop -- not merely in its ``__all__``-advertised public surface, but
among every callable this module defines at module level, exported or not:
``run_production_v8b_data_quality_calibration``. It:

1. validates the exact human-gate confirmation token FIRST -- before any
   other check, before any filesystem access;
2. validates the caller-supplied ``implementation_git_commit`` and
   ``calibration_attempt_id`` formats;
3. verifies that ``implementation_git_commit`` equals this repository's
   actual Git HEAD, and that the on-disk bytes of every file in
   ``_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS`` -- this module's own three
   files, the reused calibration core, and the reused preflight module
   (whose Git-verification helper this module imports) -- exactly match
   what is committed at that HEAD, reusing the already-reviewed
   ``_verify_implementation_matches_repository_head()`` from
   ``src/v8b_v5b_calibration_input_preflight.py`` for the actual Git
   subprocess work (sanitized ``GIT_*`` environment, repository-identity
   binding, dirty-file rejection) rather than reimplementing it;
4. only then reads the single fixed ``V5B_CACHE_ROOT``, using the same
   handle-bound, reparse/TOCTOU-safe read pattern as the reviewed
   preflight (duplicated here, not imported, because that logic is
   deliberately a private closure with no module-level name in the
   preflight module -- there is no way to import it, by design);
5. independently re-derives and re-verifies manifest provenance and every
   payload's byte-count/SHA-256 against the frozen expected pins -- this
   execution never treats a prior preflight PASS artifact as permission to
   skip its own from-scratch verification;
6. only once all 300 designated payloads are verified in memory does it
   call the existing, frozen ``run_data_quality_calibration()`` with
   exactly those same in-memory bytes -- never a second, separate read.

The byte-acquisition logic that actually walks ``V5B_CACHE_ROOT`` is
defined as a local closure nested inside
``run_production_v8b_data_quality_calibration`` itself -- it is not a
module-level name at all, has no independent existence outside a call to
the gated entry point, and cannot be imported, monkeypatched onto, or
invoked separately from it.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from . import v8b_data_quality_calibration as _v8b_calibration_module
    from . import v8b_v5b_calibration_input_preflight as _v8b_preflight_module
except ImportError:  # pragma: no cover - import-style fallback, mirrors sibling modules
    import v8b_data_quality_calibration as _v8b_calibration_module
    import v8b_v5b_calibration_input_preflight as _v8b_preflight_module

# Bound directly: stable across the whole test suite, never monkeypatched.
EXPECTED_V5B_TICKER_COUNT = _v8b_calibration_module.EXPECTED_V5B_TICKER_COUNT
STUDY = _v8b_calibration_module.STUDY
V8BCalibrationBlocked = _v8b_calibration_module.V8BCalibrationBlocked
InMemoryPayload = _v8b_calibration_module.InMemoryPayload
canonical_json_bytes = _v8b_calibration_module.canonical_json_bytes
sha256_hex = _v8b_calibration_module.sha256_hex
validate_v5b_manifest_provenance = _v8b_calibration_module.validate_v5b_manifest_provenance
run_data_quality_calibration = _v8b_calibration_module.run_data_quality_calibration

# NOTE: EXPECTED_V5B_MANIFEST_SHA256 / EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256
# are deliberately NOT copied into module-level names here -- see the
# identical note in v8b_v5b_calibration_input_preflight.py. They are read
# dynamically from _v8b_calibration_module's own namespace at call time so
# tests can monkeypatch that single, shared attribute.

# ---------------------------------------------------------------------------
# Fixed production input. Tests monkeypatch V5B_CACHE_ROOT itself to a
# temporary synthetic fixture; this module never derives a cache root from
# any CLI option, environment variable, or caller-supplied argument. Must be
# identical to the reviewed preflight's fixed root.
# ---------------------------------------------------------------------------

FIXED_V5B_CACHE_ROOT_WINDOWS_PATH = r"C:\taiki\hobbies\v5-b-evaluation-cache-retry1"
V5B_CACHE_ROOT: Path = Path(FIXED_V5B_CACHE_ROOT_WINDOWS_PATH)

_MANIFEST_FILENAME = "cache_manifest.json"

# ---------------------------------------------------------------------------
# Repository / Git-HEAD binding. Tests monkeypatch _REPO_ROOT to a temporary
# synthetic Git repository, exactly as V5B_CACHE_ROOT is monkeypatched,
# rather than exposing either as a public parameter.
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Every file whose on-disk bytes must exactly match what is committed at the
# verified actual Git HEAD before real V5-B cache access is permitted.
# src/v8b_data_quality_calibration.py is bound because this module imports
# and executes run_data_quality_calibration() / validate_v5b_manifest_
# provenance() from it and reads its fixed EXPECTED_V5B_* constants from it.
# src/v8b_v5b_calibration_input_preflight.py is bound because this module
# reuses its already-reviewed _verify_implementation_matches_repository_
# head() for the actual Git subprocess/sanitization work below, rather than
# reimplementing it -- a dirty copy of either file could silently change
# what "verified" means.
_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS: tuple[str, ...] = (
    "src/v8b_data_quality_calibration_execution.py",
    "scripts/run_v8b_data_quality_calibration.py",
    "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_SPEC.md",
    "src/v8b_data_quality_calibration.py",
    "src/v8b_v5b_calibration_input_preflight.py",
)

# ---------------------------------------------------------------------------
# Human gate. Matching this token is necessary but not sufficient: its mere
# presence in this source file does not authorize real execution. This
# implementation task does not invoke run_production_v8b_data_quality_
# calibration() against the real fixed cache root; only tests exercise it,
# and only with V5B_CACHE_ROOT and _REPO_ROOT monkeypatched to synthetic
# fixtures.
# ---------------------------------------------------------------------------

EXECUTION_GATE_CONFIRMATION = "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_GATE"

# Single outward blocker reason for every gate-level (pre-calibration)
# failure -- distinct from the calibration-result artifact's own
# CALIBRATION_RESULT_STATE_INVALID / run_invalid_reason_or_null vocabulary,
# which src/v8b_data_quality_calibration.py already owns.
EXECUTION_BLOCKER = "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_BLOCKED"

EXECUTION_STATUS_ROLE = "EXECUTION_GATE"
EXECUTION_STATUS_SCHEMA_VERSION = "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_STATUS_V1"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EXECUTION_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "study",
        "role",
        "status",
        "detail_reason",
        "implementation_git_commit",
        "calibration_attempt_id",
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

_EXECUTION_DETAIL_RE = re.compile(
    r"^(?:"
    r"EXECUTION_GATE_CONFIRMATION_REQUIRED|IMPLEMENTATION_COMMIT_INVALID|"
    r"CALIBRATION_ATTEMPT_ID_INVALID|"
    r"GIT_HEAD_UNRESOLVABLE|GIT_REPOSITORY_IDENTITY_MISMATCH|"
    r"IMPLEMENTATION_COMMIT_HEAD_MISMATCH|IMPLEMENTATION_FILE_UNVERIFIABLE|"
    r"IMPLEMENTATION_FILE_DIRTY|CACHE_ROOT_INACCESSIBLE|CACHE_ROOT_NOT_A_DIRECTORY|"
    r"CACHE_ROOT_REPARSE_POINT|MANIFEST_UNREADABLE|MANIFEST_NOT_REGULAR|MANIFEST_REPARSE_POINT|"
    r"MANIFEST_PATH_ESCAPE_DETECTED|DESIGNATED_PAYLOAD_COUNT_MISMATCH|"
    r"PAYLOAD_PATH_RESOLUTION_FAILED|PAYLOAD_PATH_ESCAPE_DETECTED|"
    r"PAYLOAD_REPARSE_POINT|PAYLOAD_NOT_REGULAR|PAYLOAD_READ_FAILED|"
    r"PAYLOAD_BINDING_FAILED|CALIBRATION_CORE_BLOCKED:[A-Z0-9_]+)"
    r"$"
)

_MANIFEST_PROVENANCE_INVALID_PREFIX = "MANIFEST_PROVENANCE_INVALID:"
_CALIBRATION_CORE_BLOCKED_PREFIX = "CALIBRATION_CORE_BLOCKED:"


class V8BCalibrationExecutionBlocked(RuntimeError):
    """Fail-closed error for every gate-level (pre-calibration) blocking
    condition in this module.

    ``reason`` is always the single generic ``EXECUTION_BLOCKER`` constant,
    so downstream classification has exactly one outward reason for "this
    execution attempt did not run the frozen calibration at all". ``detail``
    carries a safe, structural, non-identity-revealing sub-code (never a
    ticker or path) for diagnostics and tests. ``result``, when present, is
    a safe aggregate-only "execution status" dict (never the calibration
    result artifact schema), with ``status="BLOCKED"``.
    """

    def __init__(self, detail: str, result: Mapping[str, Any] | None = None) -> None:
        self.reason = EXECUTION_BLOCKER
        self.detail = detail
        self.result = dict(result) if result is not None else None
        super().__init__(detail)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finalize_status(fields: dict[str, Any]) -> dict[str, Any]:
    result = dict(fields)
    result["artifact_self_hash"] = sha256_hex(canonical_json_bytes(fields))
    return result


def _canonical_execution_status(
    detail: str,
    *,
    implementation_git_commit: str | None = None,
    calibration_attempt_id: str | None = None,
    observed_manifest_sha256: str | None = None,
    observed_payload_hash_list_sha256: str | None = None,
    checked_payload_count: int = 0,
    byte_count_mismatch_count: int = 0,
    sha256_mismatch_count: int = 0,
    missing_or_unreadable_count: int = 0,
) -> dict[str, Any]:
    started = _utc_now_iso()
    fields = {
        "schema_version": EXECUTION_STATUS_SCHEMA_VERSION,
        "study": STUDY,
        "role": EXECUTION_STATUS_ROLE,
        "status": "BLOCKED",
        "detail_reason": detail,
        "implementation_git_commit": implementation_git_commit,
        "calibration_attempt_id": calibration_attempt_id,
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
    return _finalize_status(fields)


def _verify_execution_status_self_hash(result: Mapping[str, Any]) -> None:
    supplied = result.get("artifact_self_hash")
    if not isinstance(supplied, str) or _SHA256_RE.fullmatch(supplied) is None:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_SELF_HASH_INVALID")
    fields = dict(result)
    del fields["artifact_self_hash"]
    if sha256_hex(canonical_json_bytes(fields)) != supplied:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_SELF_HASH_MISMATCH")


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


def _is_valid_calibration_attempt_id_format(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 128
        and not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)
    )


def validate_execution_status_semantics(
    result: Mapping[str, Any],
    *,
    expected_implementation_git_commit: str,
    expected_calibration_attempt_id: str,
) -> None:
    """Accept only a canonical, self-hashed, semantically valid gate-level
    execution status, bound to a caller-trusted implementation commit and
    calibration attempt id.

    This validates ONLY the "did not reach the frozen calibration at all,
    or reached it but got no canonical RESULT artifact back" status shape
    produced by this module (``_EXECUTION_STATUS_KEYS``). It does not, and
    must not, replace ``src/v8b_data_quality_calibration.py``::
    ``validate_result_artifact_semantics`` -- the acceptance API for the
    calibration RESULT artifact this module returns unmodified whenever
    ``run_data_quality_calibration()`` actually returns one. A normal,
    canonical INVALID calibration RESULT artifact (the core's own internal
    "this run is invalid" determination) is never converted into an
    execution status by this module; it is passed through unmodified as
    RESULT_V1, and this function must never be used to validate it.

    ``expected_implementation_git_commit`` and
    ``expected_calibration_attempt_id`` are both required with no default,
    for exactly the reason ``expected_implementation_git_commit`` is
    required on the reviewed preflight's own
    ``validate_preflight_result_semantics``: a persisted status's own
    ``implementation_git_commit`` / ``calibration_attempt_id`` fields must
    never be their own authority for which commit/attempt was reviewed.
    """

    if (
        not isinstance(expected_implementation_git_commit, str)
        or _COMMIT_RE.fullmatch(expected_implementation_git_commit) is None
    ):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_EXPECTED_COMMIT_INVALID")

    if not _is_valid_calibration_attempt_id_format(expected_calibration_attempt_id):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_EXPECTED_ATTEMPT_ID_INVALID")

    if not isinstance(result, Mapping) or set(result) != _EXECUTION_STATUS_KEYS:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_SCHEMA_INVALID")
    if result["schema_version"] != EXECUTION_STATUS_SCHEMA_VERSION:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_SCHEMA_INVALID")
    if result["study"] != STUDY or result["role"] != EXECUTION_STATUS_ROLE:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_SCHEMA_INVALID")
    if result["status"] != "BLOCKED":
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATUS_INVALID")
    detail = result["detail_reason"]
    if (
        not isinstance(detail, str)
        or (
            _EXECUTION_DETAIL_RE.fullmatch(detail) is None
            and not (
                isinstance(detail, str)
                and detail.startswith(_MANIFEST_PROVENANCE_INVALID_PREFIX)
                and detail[len(_MANIFEST_PROVENANCE_INVALID_PREFIX) :]
                in _v8b_calibration_module._RECOGNIZED_MANIFEST_BLOCKER_REASONS
            )
        )
    ):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_DETAIL_INVALID")

    commit = result["implementation_git_commit"]
    if commit is not None and (not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_COMMIT_INVALID")
    if commit is not None and commit != expected_implementation_git_commit:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_COMMIT_MISMATCH")

    attempt_id = result["calibration_attempt_id"]
    if attempt_id is not None and not _is_valid_calibration_attempt_id_format(attempt_id):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_ATTEMPT_ID_INVALID")
    if attempt_id is not None and attempt_id != expected_calibration_attempt_id:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_ATTEMPT_ID_MISMATCH")

    expected_manifest = _v8b_calibration_module.EXPECTED_V5B_MANIFEST_SHA256
    expected_payload_list = _v8b_calibration_module.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256
    if result["expected_manifest_sha256"] != expected_manifest:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_PROVENANCE_INVALID")
    if result["expected_payload_hash_list_sha256"] != expected_payload_list:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_PROVENANCE_INVALID")
    for key in ("expected_manifest_sha256", "expected_payload_hash_list_sha256"):
        if not isinstance(result[key], str) or _SHA256_RE.fullmatch(result[key]) is None:
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_PROVENANCE_INVALID")
    for key in ("observed_manifest_sha256", "observed_payload_hash_list_sha256"):
        value = result[key]
        if value is not None and (not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None):
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_PROVENANCE_INVALID")

    for key in (
        "expected_payload_count",
        "checked_payload_count",
        "byte_count_mismatch_count",
        "sha256_mismatch_count",
        "missing_or_unreadable_count",
    ):
        if not _is_exact_nonnegative_int(result[key]):
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_COUNT_INVALID")
    if result["expected_payload_count"] != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_COUNT_INVALID")
    if result["checked_payload_count"] > result["expected_payload_count"]:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_COUNT_INVALID")

    if not _is_valid_utc_timestamp(result["run_started_utc"]):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_TIMESTAMP_INVALID")
    if not _is_valid_utc_timestamp(result["run_completed_utc"]):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_TIMESTAMP_INVALID")
    if result["run_completed_utc"] < result["run_started_utc"]:
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_TIMESTAMP_INVALID")

    pre_manifest_details = {
        "EXECUTION_GATE_CONFIRMATION_REQUIRED",
        "IMPLEMENTATION_COMMIT_INVALID",
        "CALIBRATION_ATTEMPT_ID_INVALID",
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
            result[key] is not None for key in ("observed_manifest_sha256", "observed_payload_hash_list_sha256")
        ) or any(
            result[key] != 0
            for key in (
                "checked_payload_count",
                "byte_count_mismatch_count",
                "sha256_mismatch_count",
                "missing_or_unreadable_count",
            )
        ):
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")
    elif detail.startswith(_MANIFEST_PROVENANCE_INVALID_PREFIX):
        if result["observed_manifest_sha256"] is None or result["observed_payload_hash_list_sha256"] is not None:
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")
        if any(
            result[key] != 0
            for key in (
                "checked_payload_count",
                "byte_count_mismatch_count",
                "sha256_mismatch_count",
                "missing_or_unreadable_count",
            )
        ):
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")
    payload_stage_details = {
        "DESIGNATED_PAYLOAD_COUNT_MISMATCH",
        "PAYLOAD_PATH_RESOLUTION_FAILED",
        "PAYLOAD_PATH_ESCAPE_DETECTED",
        "PAYLOAD_REPARSE_POINT",
        "PAYLOAD_NOT_REGULAR",
        "PAYLOAD_READ_FAILED",
        "PAYLOAD_BINDING_FAILED",
    }
    is_calibration_core_blocked = detail.startswith(_CALIBRATION_CORE_BLOCKED_PREFIX)
    fully_clean_bind = (
        result["observed_manifest_sha256"] == expected_manifest
        and result["observed_payload_hash_list_sha256"] == expected_payload_list
        and result["checked_payload_count"] == EXPECTED_V5B_TICKER_COUNT
        and result["byte_count_mismatch_count"] == 0
        and result["sha256_mismatch_count"] == 0
        and result["missing_or_unreadable_count"] == 0
    )

    if detail in payload_stage_details:
        if (
            result["observed_manifest_sha256"] != expected_manifest
            or result["observed_payload_hash_list_sha256"] != expected_payload_list
        ):
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")
        if fully_clean_bind:
            # A payload-stage failure detail (e.g. "binding failed") can
            # never legitimately co-occur with a fully clean 300/300 bind
            # -- that combination is self-contradictory.
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")
    elif is_calibration_core_blocked:
        # CALIBRATION_CORE_BLOCKED:* records an unexpected exception from
        # the frozen core AFTER it was genuinely invoked -- which, by this
        # adapter's own design, only ever happens once the adapter's own
        # from-scratch manifest/payload byte-binding has fully and cleanly
        # succeeded. This is therefore the one detail category that MUST
        # show a fully clean 300/300 bind; anything less is inconsistent
        # with how this detail can ever legitimately be produced.
        if not fully_clean_bind:
            raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")

    _verify_execution_status_self_hash(result)

    if fully_clean_bind and not is_calibration_core_blocked:
        # Outside CALIBRATION_CORE_BLOCKED:*, a gate-level status may never
        # claim a fully clean 300/300 bind -- every other detail category
        # is, by construction, raised strictly before such a state could
        # exist. A full clean bind under any other detail would mean
        # calibration WAS invoked and returned a canonical RESULT artifact
        # (the other schema), never this one.
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")
    if detail == "PAYLOAD_BINDING_FAILED" and (
        result["checked_payload_count"] + result["missing_or_unreadable_count"] != EXPECTED_V5B_TICKER_COUNT
    ):
        raise V8BCalibrationExecutionBlocked("EXECUTION_STATUS_STATE_INVALID")


def run_production_v8b_data_quality_calibration(
    *,
    confirmation: str,
    implementation_git_commit: str,
    calibration_attempt_id: str,
) -> dict[str, Any]:
    """The single filesystem-capable entry point in this module, full stop
    -- not merely the single *exported* one.

    Order is fixed and security-relevant: (1) exact confirmation token, (2)
    caller-supplied ``implementation_git_commit`` / ``calibration_attempt_id``
    format validation, (3) ``implementation_git_commit`` equals this
    repository's actual Git HEAD and every relevant implementation file
    (this module's own three files, the reused calibration core, and the
    reused preflight module) is byte-identical to what is committed there,
    (4) only then read the single fixed ``V5B_CACHE_ROOT`` and, once its
    manifest and all 300 designated payloads are independently re-verified
    from scratch, invoke the existing frozen ``run_data_quality_
    calibration()`` with exactly those in-memory bytes. There is no
    parameter to override any of these. This implementation task does not
    invoke this function against the real cache; it exists so a future,
    separately authorized task can call it with the genuine human-supplied
    confirmation token against a clean, committed HEAD.

    Raises ``V8BCalibrationExecutionBlocked`` for every gate-level failure
    that occurs before the frozen calibration is ever invoked. Once
    invoked, ``run_data_quality_calibration()`` never raises -- it always
    returns the canonical calibration result artifact (whether the run it
    describes is valid or itself reports an invalid run) -- and this
    function returns that artifact completely unmodified.
    """

    if confirmation != EXECUTION_GATE_CONFIRMATION:
        raise V8BCalibrationExecutionBlocked(
            "EXECUTION_GATE_CONFIRMATION_REQUIRED",
            result=_canonical_execution_status("EXECUTION_GATE_CONFIRMATION_REQUIRED"),
        )

    if not isinstance(implementation_git_commit, str) or not _COMMIT_RE.match(implementation_git_commit):
        raise V8BCalibrationExecutionBlocked(
            "IMPLEMENTATION_COMMIT_INVALID",
            result=_canonical_execution_status("IMPLEMENTATION_COMMIT_INVALID"),
        )

    if not _is_valid_calibration_attempt_id_format(calibration_attempt_id):
        raise V8BCalibrationExecutionBlocked(
            "CALIBRATION_ATTEMPT_ID_INVALID",
            result=_canonical_execution_status(
                "CALIBRATION_ATTEMPT_ID_INVALID", implementation_git_commit=implementation_git_commit
            ),
        )

    try:
        _v8b_preflight_module._verify_implementation_matches_repository_head(
            repo_root=_REPO_ROOT,
            implementation_git_commit=implementation_git_commit,
            relevant_relative_paths=_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS,
        )
    except _v8b_preflight_module.V5BCalibrationInputPreflightBlocked as error:
        raise V8BCalibrationExecutionBlocked(
            error.detail,
            result=_canonical_execution_status(
                error.detail,
                implementation_git_commit=implementation_git_commit,
                calibration_attempt_id=calibration_attempt_id,
            ),
        ) from error

    # ------------------------------------------------------------------
    # Nested closure: the only code in this module that ever reads the
    # V5-B cache. It exists only for the duration of this call; it is not
    # a module attribute, so `execution.<anything>` can never reach it,
    # and it cannot be exercised without first passing every gate above.
    # ------------------------------------------------------------------

    def _walk_and_execute(cache_root: Path) -> dict[str, Any]:
        run_started = _utc_now_iso()

        def block(detail: str, **counts: Any) -> V8BCalibrationExecutionBlocked:
            return V8BCalibrationExecutionBlocked(
                detail,
                result=_canonical_execution_status(
                    detail,
                    implementation_git_commit=implementation_git_commit,
                    calibration_attempt_id=calibration_attempt_id,
                    **counts,
                ),
            )

        def is_reparse_point(path: Path) -> bool:
            try:
                stat_result = path.lstat()
            except OSError:
                return False
            if path.is_symlink():
                return True
            return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400)

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

        def verify_root(path: Path) -> Path:
            try:
                absolute = path.absolute()
                for component in (absolute, *absolute.parents):
                    if is_reparse_point(component):
                        raise block("CACHE_ROOT_REPARSE_POINT")
                resolved = absolute.resolve(strict=True)
                if os.name == "nt" and normalized_path(resolved) != normalized_path(absolute):
                    raise block("CACHE_ROOT_REPARSE_POINT")
            except V8BCalibrationExecutionBlocked:
                raise
            except OSError:
                raise block("CACHE_ROOT_INACCESSIBLE")
            try:
                if not resolved.is_dir() or is_reparse_point(resolved):
                    raise block("CACHE_ROOT_NOT_A_DIRECTORY" if not resolved.is_dir() else "CACHE_ROOT_REPARSE_POINT")
            except OSError:
                raise block("CACHE_ROOT_INACCESSIBLE")
            return resolved

        def reject_reparse_components(path: Path, root_path: Path) -> None:
            current = path
            while True:
                if is_reparse_point(current):
                    raise block("PAYLOAD_REPARSE_POINT")
                if current == root_path or current.parent == current:
                    break
                current = current.parent

        def read_verified_file(path: Path, root_path: Path, *, kind: str = "PAYLOAD") -> bytes | None:
            """Read one already-designated file through one checked handle.
            The exact same bytes returned here are the bytes used for
            SHA-256/byte-count checking AND the bytes later handed to
            run_data_quality_calibration() -- there is no second read."""

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
            except V8BCalibrationExecutionBlocked:
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

        observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

        try:
            validated_manifest = validate_v5b_manifest_provenance(manifest_bytes)
        except V8BCalibrationBlocked as error:
            raise block(
                "MANIFEST_PROVENANCE_INVALID:" + error.reason,
                observed_manifest_sha256=observed_manifest_sha256,
            ) from error

        observed_payload_hash_list_sha256 = validated_manifest["payload_hash_list_sha256"]

        payloads = validated_manifest["payloads"]
        if not isinstance(payloads, list) or len(payloads) != EXPECTED_V5B_TICKER_COUNT:
            raise block(
                "DESIGNATED_PAYLOAD_COUNT_MISMATCH",
                observed_manifest_sha256=observed_manifest_sha256,
                observed_payload_hash_list_sha256=observed_payload_hash_list_sha256,
            )

        checked_count = 0
        missing_count = 0
        byte_mismatch_count = 0
        sha_mismatch_count = 0
        ticker_payloads: dict[str, InMemoryPayload] = {}

        for record in payloads:
            relative_path = record["relative_path"]
            candidate_path = root_resolved / relative_path
            reject_reparse_components(candidate_path, root_resolved)
            try:
                resolved_candidate = candidate_path.resolve(strict=False)
            except OSError:
                raise block(
                    "PAYLOAD_PATH_RESOLUTION_FAILED",
                    observed_manifest_sha256=observed_manifest_sha256,
                    observed_payload_hash_list_sha256=observed_payload_hash_list_sha256,
                )
            if resolved_candidate != root_resolved and root_resolved not in resolved_candidate.parents:
                raise block(
                    "PAYLOAD_PATH_ESCAPE_DETECTED",
                    observed_manifest_sha256=observed_manifest_sha256,
                    observed_payload_hash_list_sha256=observed_payload_hash_list_sha256,
                )
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

            # The exact bytes just verified above are stored, unmodified,
            # for the sole possible later use: handing them to
            # run_data_quality_calibration(). No second read ever occurs.
            ticker_payloads[str(record["ticker"])] = InMemoryPayload(
                relative_path=relative_path, payload_bytes=payload_bytes
            )

        if (
            missing_count != 0
            or byte_mismatch_count != 0
            or sha_mismatch_count != 0
            or checked_count != EXPECTED_V5B_TICKER_COUNT
        ):
            raise block(
                "PAYLOAD_BINDING_FAILED",
                observed_manifest_sha256=observed_manifest_sha256,
                observed_payload_hash_list_sha256=observed_payload_hash_list_sha256,
                checked_payload_count=checked_count,
                byte_count_mismatch_count=byte_mismatch_count,
                sha256_mismatch_count=sha_mismatch_count,
                missing_or_unreadable_count=missing_count,
            )

        # ------------------------------------------------------------
        # Every one of the 300 designated payloads, plus the manifest,
        # independently re-verified from scratch against the frozen
        # pins -- never trusting any prior preflight result. Only now
        # does the existing, frozen, pure calibration function run.
        # ------------------------------------------------------------
        try:
            return run_data_quality_calibration(
                repository_root=_REPO_ROOT,
                manifest_bytes=manifest_bytes,
                ticker_payloads=ticker_payloads,
                implementation_git_commit=implementation_git_commit,
                calibration_attempt_id=calibration_attempt_id,
                run_started_utc=run_started,
            )
        except V8BCalibrationBlocked as error:
            raise block(
                _CALIBRATION_CORE_BLOCKED_PREFIX + error.reason,
                observed_manifest_sha256=observed_manifest_sha256,
                observed_payload_hash_list_sha256=observed_payload_hash_list_sha256,
                checked_payload_count=checked_count,
                missing_or_unreadable_count=missing_count,
            ) from error

    return _walk_and_execute(V5B_CACHE_ROOT)


# ---------------------------------------------------------------------------
# Static-check mode: repository-only. Zero V5-B cache access, zero network
# access -- reads only this module's own source and introspects its own
# callable surface.
# ---------------------------------------------------------------------------


def run_static_check() -> None:
    """Repository-only verification. Raises ``V8BCalibrationExecutionBlocked``
    on any drift; returns ``None`` (passes) otherwise. Never touches the
    V5-B cache and never makes a network call.
    """

    if FIXED_V5B_CACHE_ROOT_WINDOWS_PATH != r"C:\taiki\hobbies\v5-b-evaluation-cache-retry1":
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_CACHE_ROOT_DRIFT")

    if EXECUTION_GATE_CONFIRMATION != "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_GATE":
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_GATE_TOKEN_DRIFT")

    production_params = set(inspect.signature(run_production_v8b_data_quality_calibration).parameters)
    if production_params != {"confirmation", "implementation_git_commit", "calibration_attempt_id"}:
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_PRODUCTION_API_SURFACE_DRIFT")

    # Scan the ENTIRE module callable surface -- every module-level name,
    # exported or not, not merely __all__ -- for a filesystem-capable
    # bypass. The actual cache-walking logic is a closure nested inside
    # run_production_v8b_data_quality_calibration and therefore never
    # appears at module level in the first place.
    module_globals = globals()
    banned_names = {"_walk_and_execute", "run_v8b_data_quality_calibration"}
    if banned_names & set(module_globals) or banned_names & set(__all__):
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED")

    forbidden_param_names = {"cache_root", "path", "manifest_path", "input_dir", "dataset"}
    for name, candidate in list(module_globals.items()):
        if name.startswith("__") or not callable(candidate) or inspect.isclass(candidate):
            continue
        if getattr(candidate, "__module__", None) != __name__:
            # Defined elsewhere (e.g. reused from the calibration core or
            # the preflight module) -- not this module's own API surface.
            continue
        try:
            params = set(inspect.signature(candidate).parameters)
        except (TypeError, ValueError):
            continue
        if params & forbidden_param_names:
            raise V8BCalibrationExecutionBlocked("STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED")

    if EXPECTED_V5B_TICKER_COUNT != 300:
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_PAYLOAD_COUNT_DRIFT")

    if validate_v5b_manifest_provenance is not _v8b_calibration_module.validate_v5b_manifest_provenance:
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_MANIFEST_VALIDATOR_DRIFT")

    if run_data_quality_calibration is not _v8b_calibration_module.run_data_quality_calibration:
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_CALIBRATION_ENTRY_POINT_DRIFT")

    git_verifier = _v8b_preflight_module._verify_implementation_matches_repository_head
    git_verifier_params = set(inspect.signature(git_verifier).parameters)
    if not {"repo_root", "implementation_git_commit", "relevant_relative_paths"} <= git_verifier_params:
        raise V8BCalibrationExecutionBlocked("STATIC_CHECK_GIT_VERIFIER_DRIFT")

    # Scan only the functional/security-relevant code above this function's
    # own definition -- NOT this function's own body, which necessarily
    # names these same tokens as literal strings in order to check for
    # them, and would otherwise always self-match.
    source = Path(__file__).read_text(encoding="utf-8")
    functional_source = source[: source.index("\ndef run_static_check")]
    forbidden_source_tokens = [
        "parse_ticker_observations(",
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
            raise V8BCalibrationExecutionBlocked("STATIC_CHECK_FORBIDDEN_SOURCE_TOKEN")


__all__ = [
    "EXECUTION_BLOCKER",
    "EXECUTION_GATE_CONFIRMATION",
    "EXECUTION_STATUS_ROLE",
    "EXECUTION_STATUS_SCHEMA_VERSION",
    "FIXED_V5B_CACHE_ROOT_WINDOWS_PATH",
    "V5B_CACHE_ROOT",
    "V8BCalibrationExecutionBlocked",
    "run_production_v8b_data_quality_calibration",
    "run_static_check",
    "validate_execution_status_semantics",
]
