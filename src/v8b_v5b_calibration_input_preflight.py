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

The production entry point (``run_production_v5b_calibration_input_
preflight``) may read only the single fixed cache root below. There is no
parameter, in this module or in the CLI that wraps it, that accepts an
alternate cache path, manifest path, input directory, or dataset. Real
execution additionally requires an exact human-gate confirmation token;
presence of that token in this source file does not itself authorize
execution -- see the module docstring note on ``PREFLIGHT_GATE_
CONFIRMATION`` below.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
# Human gate (§2). Matching this token is necessary but not sufficient: its
# mere presence in this source file does not authorize real execution. This
# implementation task does not invoke run_production_v5b_calibration_input_
# preflight() against the real fixed cache root; only tests exercise it,
# and only with V5B_CACHE_ROOT monkeypatched to a synthetic fixture.
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


def run_v5b_calibration_input_preflight(
    *,
    cache_root: Path | str,
    implementation_git_commit: str,
    run_started_utc: str | None = None,
) -> dict[str, Any]:
    """Core preflight logic (§3, §5). The only I/O this function performs is
    reading files under ``cache_root``: a stat/existence check on the root,
    a read of its ``cache_manifest.json``, and -- for exactly the 300
    manifest-designated payloads, never any other file -- an existence
    check, a path-containment check, a raw byte read, and a SHA-256/byte-
    count comparison against the validated manifest's own declared values.

    Never JSON-parses a payload body, never inspects OHLCV, never touches
    anything outside ``cache_root``. Raises ``V5BCalibrationInputPreflight
    Blocked`` on any failure (§5); returns a safe aggregate-only dict (§6)
    only on a full PASS.
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

    if not isinstance(implementation_git_commit, str) or not _COMMIT_RE.match(implementation_git_commit):
        raise block("IMPLEMENTATION_COMMIT_INVALID")

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


def run_production_v5b_calibration_input_preflight(
    *,
    confirmation: str,
    implementation_git_commit: str,
) -> dict[str, Any]:
    """Human-gated production entry point (§2). Accesses only the single
    fixed ``V5B_CACHE_ROOT`` -- there is no parameter to override it. This
    implementation task does not invoke this function against the real
    cache; it exists so a future, separately authorized task can call it
    with the genuine human-supplied confirmation token.
    """

    if confirmation != PREFLIGHT_GATE_CONFIRMATION:
        raise V5BCalibrationInputPreflightBlocked(
            "PREFLIGHT_GATE_CONFIRMATION_REQUIRED",
            result={
                "schema_version": PREFLIGHT_RESULT_SCHEMA_VERSION,
                "study": STUDY,
                "role": PREFLIGHT_ROLE,
                "status": "BLOCK",
                "detail_reason": "PREFLIGHT_GATE_CONFIRMATION_REQUIRED",
            },
        )
    return run_v5b_calibration_input_preflight(
        cache_root=V5B_CACHE_ROOT,
        implementation_git_commit=implementation_git_commit,
    )


__all__ = [
    "FIXED_V5B_CACHE_ROOT_WINDOWS_PATH",
    "PREFLIGHT_BLOCKER",
    "PREFLIGHT_GATE_CONFIRMATION",
    "PREFLIGHT_RESULT_SCHEMA_VERSION",
    "PREFLIGHT_ROLE",
    "V5BCalibrationInputPreflightBlocked",
    "V5B_CACHE_ROOT",
    "run_production_v5b_calibration_input_preflight",
    "run_v5b_calibration_input_preflight",
]
