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
import re
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


def _minimal_block_result(detail: str) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_RESULT_SCHEMA_VERSION,
        "study": STUDY,
        "role": PREFLIGHT_ROLE,
        "status": "BLOCK",
        "detail_reason": detail,
    }


# ---------------------------------------------------------------------------
# Repository / Git-HEAD binding (finding 2)
# ---------------------------------------------------------------------------


def _resolve_actual_git_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    head = completed.stdout.strip()
    if not _COMMIT_RE.match(head):
        return None
    return head


def _read_committed_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
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
            "GIT_HEAD_UNRESOLVABLE", result=_minimal_block_result("GIT_HEAD_UNRESOLVABLE")
        )
    if implementation_git_commit != actual_head:
        raise V5BCalibrationInputPreflightBlocked(
            "IMPLEMENTATION_COMMIT_HEAD_MISMATCH",
            result=_minimal_block_result("IMPLEMENTATION_COMMIT_HEAD_MISMATCH"),
        )
    for relative_path in relevant_relative_paths:
        committed_bytes = _read_committed_bytes(repo_root, actual_head, relative_path)
        if committed_bytes is None:
            raise V5BCalibrationInputPreflightBlocked(
                "IMPLEMENTATION_FILE_UNVERIFIABLE",
                result=_minimal_block_result("IMPLEMENTATION_FILE_UNVERIFIABLE"),
            )
        try:
            working_tree_bytes = (repo_root / relative_path).read_bytes()
        except OSError:
            raise V5BCalibrationInputPreflightBlocked(
                "IMPLEMENTATION_FILE_UNVERIFIABLE",
                result=_minimal_block_result("IMPLEMENTATION_FILE_UNVERIFIABLE"),
            )
        if working_tree_bytes != committed_bytes:
            raise V5BCalibrationInputPreflightBlocked(
                "IMPLEMENTATION_FILE_DIRTY", result=_minimal_block_result("IMPLEMENTATION_FILE_DIRTY")
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
            result=_minimal_block_result("PREFLIGHT_GATE_CONFIRMATION_REQUIRED"),
        )

    if not isinstance(implementation_git_commit, str) or not _COMMIT_RE.match(implementation_git_commit):
        raise V5BCalibrationInputPreflightBlocked(
            "IMPLEMENTATION_COMMIT_INVALID", result=_minimal_block_result("IMPLEMENTATION_COMMIT_INVALID")
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
            return V5BCalibrationInputPreflightBlocked(detail, result=_finalize(fields))

        root = Path(cache_root)
        try:
            root_resolved = root.resolve(strict=True)
        except OSError:
            raise block("CACHE_ROOT_INACCESSIBLE")
        if not root_resolved.is_dir():
            raise block("CACHE_ROOT_NOT_A_DIRECTORY")

        manifest_path = root_resolved / _MANIFEST_FILENAME
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError:
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
            try:
                resolved_candidate = candidate_path.resolve(strict=False)
            except OSError:
                raise block("PAYLOAD_PATH_RESOLUTION_FAILED")
            if resolved_candidate != root_resolved and root_resolved not in resolved_candidate.parents:
                raise block("PAYLOAD_PATH_ESCAPE_DETECTED")

            if not resolved_candidate.is_file():
                missing_count += 1
                continue
            try:
                payload_bytes = resolved_candidate.read_bytes()
            except OSError:
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
        return _finalize(fields)

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
    "run_production_v5b_calibration_input_preflight",
    "run_static_check",
]
