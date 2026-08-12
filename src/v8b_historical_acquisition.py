"""V8B raw-only historical OHLCV acquisition for `T1B` (new) and `T2` (reused).

`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §7.6, §7.7, §10, §11, §12.1,
§12.3, §12.4, §12.6. This is the `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_
IMPLEMENTATION` production module (§12's gate sequence). It does **not**
perform, and this repository has not authorized, any real Yahoo or JPX
request, any real `T1B` allocation, or any real private V8/V8B partition
access -- every test exercising this module is fake/synthetic-only
(dependency-injected opener, clock, Git resolver, classifier-blob resolver,
and ZoneInfo loader). Importing this module performs no I/O.

This module never imports, reads, or modifies `src/v7_yahoo_collector.py`
or `src/v8_historical_acquisition.py`. It reuses `src.v7_yahoo_collector`
read-only for the already-accepted, generic, single-ticker Yahoo Chart
transport and canonical parser (§13: `SAFE_TO_REUSE`), and reuses
`src.v8_partition` read-only for the original V8 partition-manifest reader
and its provenance/output-path-safety primitives. It reimplements (does not
import) the private V8 trusted-partition-anchor read/validate logic that
`src/v8_historical_acquisition.py` already established, so that this
security-sensitive V8B production boundary is self-contained and
independently auditable without depending on another module's non-exported
internals, while remaining bit-for-bit the same validation semantics
(`OPTION_2`'s requirement that `T2` re-verify through the *original
immutable V8 authority chain*, §11.3.E).

Two, and only two, logical blocks may be acquired here:

- `T1B` -- new `V8B_HISTORICAL_RESEARCH` validation block, bound to the new
  successor allocation-authority chain (§11.3.A-D: a verified private
  allocation artifact, `src/v8b_allocation.py`, pinned by a trust-pin object
  validated by `src/v8b_trust_pin.py`).
- `T2` -- reused `V8_HISTORICAL_RESEARCH` sealed holdout, bound to the
  original, immutable `V8_TRUSTED_PARTITION.json` authority plus the
  explicit `OPTION_2` bridge (`V8B_T2_AUTHORITY_BRIDGE.json`, §11.3.E).

`T0`, old `T1`, `T3`, and `T_spare` (whole) remain unconditionally
prohibited from acquisition through this module.

Every raw acquisition is gated by `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_
ROW_QUALITY_GATE` (§7.6): `invalid_returned_row_count * 252 <=
total_returned_row_count` (exact integer comparison, never floating-point),
`max_consecutive_invalid_returned_rows = 1`, checked over the full `P_hist`
series and independently over each of the eight frozen production test
years 2018-2025 (not the calibration-evidence 2019-2025 span). Before any
Yahoo request, production must also verify (§7.7, §12.1.I/J) that
`Asia/Tokyo` `ZoneInfo` data is available and that the exact pinned
`src/v7_yahoo_collector.py` Git blob is unchanged; this module's public
production entrypoint runs the full pre-network ordering required by this
implementation's own review checklist (repo/provenance, freeze approval,
reviewed-implementation binding, classifier blob, ZoneInfo, authority
chain, block count/hash, output/staging safety) strictly before the first
Yahoo request.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.v7_yahoo_collector import (
    FRAME_FIELDS,
    HOST,
    V7YahooCollectorBlocked,
    canonical_ticker,
    fetch_chart_once,
)

from src.v8_partition import (
    BLOCK_SIZE as V8_BLOCK_SIZE,
    DESIGN_COMMIT as V8_DESIGN_COMMIT,
    SCHEMA_VERSION as V8_PARTITION_MANIFEST_SCHEMA_VERSION,
    STUDY_NAME as V8_STUDY_NAME,
    V8PartitionBlocked,
    read_partition_manifest,
    require_absolute_output_path_outside_repository,
    resolve_verified_production_git_commit,
    ticker_list_sha256,
)

from src.v8b_allocation import (
    V8BAllocationBlocked,
    verify_allocation_artifact_self_hash,
)
from src.v8b_trust_pin import (
    V8BTrustPinBlocked,
    validate_trust_pin,
)

SCHEMA_VERSION = "V8B_HISTORICAL_ACQUISITION_V1"
STUDY_NAME = "V8B_HISTORICAL_RESEARCH"
MODE = "V8B_RAW_HISTORICAL_ACQUISITION"

# The exact 40-hex commit this implementation phase is bound to
# (V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md's frozen design, per
# V8B_DESIGN_FREEZE_APPROVAL.json). This is a distinct commit from V8's own
# frozen V8_DESIGN_COMMIT above -- the two studies have separate identities.
V8B_FROZEN_DESIGN_COMMIT = "eedf198b93185b963b825170ed0be97e93f923b7"

DATA_SOURCE = "Yahoo Chart"
DATA_SOURCE_HOST = HOST
DATA_SOURCE_SCHEMA = "Yahoo Chart v8/finance/chart interval=1d events=div,splits includeAdjustedClose=true"

REQUEST_START = "2016-04-01"
REQUEST_END_EXCLUSIVE = "2026-01-01"

MIN_REQUEST_INTERVAL_SECONDS = 2.0
RETRY_COUNT = 0

# POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE (§7.6). The fraction
# is stored/compared as an exact integer numerator/denominator -- never a
# float -- per "floating_point_threshold_decision=PROHIBITED".
MALFORMED_OHLCV_POLICY_NAME = "POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE"
MALFORMED_OHLCV_INVALID_FRACTION_NUMERATOR = 1
MALFORMED_OHLCV_INVALID_FRACTION_DENOMINATOR = 252
MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS = 1
MALFORMED_OHLCV_FULL_P_HIST_CHECK_REQUIRED = True
MALFORMED_OHLCV_TEST_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
MALFORMED_OHLCV_EXPECTED_CALENDAR_MISSING_DATES_TREATED_AS_MALFORMED = False
MALFORMED_OHLCV_THRESHOLD_EXCEEDANCE_ACTION = "BLOCK_WHOLE_ACQUISITION"

MALFORMED_OHLCV_POLICY_METADATA_FIELDS = (
    "policy_name",
    "invalid_fraction_numerator",
    "invalid_fraction_denominator",
    "max_consecutive_invalid_returned_rows",
    "full_p_hist_check_required",
    "test_years",
    "expected_calendar_missing_dates_treated_as_malformed",
    "threshold_exceedance_action",
)

# §7.6: production must bind to exactly this pinned classifier/parser blob.
CANONICAL_PARSER_CLASSIFIER_FILE = "src/v7_yahoo_collector.py"
CANONICAL_PARSER_CLASSIFIER_GIT_COMMIT = "28e281c3ee30d6b4c2f981c5da3ddc983c09724d"
CANONICAL_PARSER_CLASSIFIER_BLOB_SHA = "76b57b077f3214e666ff9dc06d9c224afc16df9f"

ALLOWED_ACQUISITION_BLOCKS = ("T1B", "T2")
PROHIBITED_ACQUISITION_BLOCKS = ("T0", "T1", "T3", "T_spare")

BLOCK_ROLE = {"T1B": "VALIDATION", "T2": "SEALED_HOLDOUT"}
BLOCK_STATUS = {"T1B": "RAW_ACQUIRED_NOT_OPENED", "T2": "RAW_ACQUIRED_SEALED"}
BLOCK_SEALED = {"T1B": False, "T2": True}
BLOCK_AUTHORITY_CHAIN = {
    "T1B": "V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
    "T2": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE",
}

ACQUISITIONS_DIRNAME = "v8b_acquisitions"
RAW_DIRNAME = "raw"
MANIFEST_FILENAME = "acquisition_manifest.json"
SEALED_FILENAME = "SEALED.json"

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TRUSTED_PARTITION_ANCHOR_GIT_PATH = "V8_TRUSTED_PARTITION.json"
TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION = "V8_TRUSTED_PARTITION_V1"
TRUSTED_PARTITION_ANCHOR_FIELDS = (
    "schema_version",
    "study_name",
    "design_commit",
    "authorization_status",
    "authorized_partition_manifest_sha256",
    "authorized_partition_implementation_git_commit",
    "authorization_note",
)

T2_AUTHORITY_BRIDGE_GIT_PATH = "V8B_T2_AUTHORITY_BRIDGE.json"
T2_AUTHORITY_BRIDGE_SCHEMA_VERSION = "V8B_T2_AUTHORITY_BRIDGE_V1"
T2_AUTHORITY_BRIDGE_FIELDS = (
    "schema_version",
    "study",
    "role",
    "source_authority",
    "v8_trust_anchor_git_path",
    "v8_trust_anchor_git_identity",
    "authorized_parent_v8_partition_manifest_sha256",
    "expected_t2_ticker_list_sha256",
    "t2_acquired_before_authorized_acquisition",
    "t2_research_open_count_before_official_opening",
    "v8b_frozen_design_commit",
    "t2_membership_reassignment",
    "v8_trusted_partition_json_mutated_or_repinned",
    "option",
    "human_gate",
    "authorization_note",
)

DESIGN_FREEZE_APPROVAL_GIT_PATH = "V8B_DESIGN_FREEZE_APPROVAL.json"

IMPLEMENTATION_REVIEW_GIT_PATH = "V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json"
IMPLEMENTATION_REVIEW_SCHEMA_VERSION = "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_V1"
IMPLEMENTATION_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "reviewed_implementation_git_commit",
    "review_result",
    "approval_status",
)

ACQUISITION_MANIFEST_FIELDS = (
    "schema_version",
    "study_name",
    "v8b_frozen_design_commit",
    "implementation_git_commit",
    "reviewed_production_implementation_commit",
    "block",
    "role",
    "status",
    "sealed",
    "research_access_authorized",
    "authority_chain",
    "authority_binding",
    "data_source",
    "data_source_host",
    "data_source_schema",
    "canonical_parser_classifier_blob_sha",
    "request_start",
    "request_end_exclusive",
    "ticker_count",
    "ticker_list_sha256",
    "request_count",
    "retry_count",
    "http_429_count",
    "success_transport_count",
    "valid_price_row_count",
    "invalid_price_row_count",
    "invalid_reason_counts",
    "malformed_ohlcv_policy",
    "split_event_count",
    "payload_manifest",
    "payload_manifest_sha256",
    "canonical_price_rows_sha256",
    "canonical_split_events_sha256",
    "acquisition_started_utc",
    "acquisition_completed_utc",
    "validation_access_count",
    "feature_computation_count",
    "outcome_access_count",
    "sealed_holdout_access_count",
)

PAYLOAD_RECORD_FIELDS = (
    "ticker",
    "payload_sha256",
    "byte_count",
    "canonical_price_rows_sha256",
    "canonical_split_events_sha256",
    "valid_price_row_count",
    "invalid_price_row_count",
    "split_event_count",
)

ACQUISITION_MANIFEST_ZERO_ACCESS_COUNTER_FIELDS = (
    "validation_access_count",
    "feature_computation_count",
    "outcome_access_count",
    "sealed_holdout_access_count",
)


class V8BHistoricalAcquisitionBlocked(RuntimeError):
    """Fail-closed V8B historical acquisition transport, schema, or seal error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Canonical hashing helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V8BHistoricalAcquisitionBlocked("NONFINITE_VALUE") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _ticker_list_sha(tickers: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8"))


def _require_git_commit(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8BHistoricalAcquisitionBlocked(reason)
    return value


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8BHistoricalAcquisitionBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8BHistoricalAcquisitionBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8BHistoricalAcquisitionBlocked(invalid_reason)
    return parsed


# ---------------------------------------------------------------------------
# Malicious-environment isolation for every git invocation in this module.
#
# ``git -C <root> ...`` does NOT override GIT_DIR/GIT_WORK_TREE/GIT_INDEX_
# FILE/GIT_OBJECT_DIRECTORY/GIT_ALTERNATE_OBJECT_DIRECTORIES/GIT_COMMON_DIR
# if they are present in the process environment -- an attacker who
# controls the environment (not the repository) could otherwise redirect
# every "verified Git object" read in this module (design-freeze approval,
# implementation review, classifier blob, trusted-partition anchor, T2
# bridge) to a repository of their own choosing, defeating every
# provenance check below in one shot. Every git subprocess this module
# spawns therefore runs with these variables explicitly stripped.
# ---------------------------------------------------------------------------

_ISOLATED_GIT_ENV_BLOCKLIST = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
)


def _isolated_git_subprocess_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _ISOLATED_GIT_ENV_BLOCKLIST}


class _isolated_ambient_git_environment:
    """Temporarily strip the malicious-redirection GIT_* variables from
    ``os.environ`` for the duration of a call into code this module does not
    own (`src.v8_partition.resolve_verified_production_git_commit`, which
    spawns its own ``git`` subprocess without accepting an ``env=``
    override). Restores the prior values on exit regardless of outcome."""

    def __enter__(self) -> None:
        self._saved = {key: os.environ.pop(key, None) for key in _ISOLATED_GIT_ENV_BLOCKLIST}
        return None

    def __exit__(self, *exc_info: Any) -> None:
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Step 1: repo/provenance
# ---------------------------------------------------------------------------


def _resolve_verified_canonical_production_git_commit() -> str:
    try:
        with _isolated_ambient_git_environment():
            return resolve_verified_production_git_commit(CANONICAL_REPOSITORY_ROOT)
    except V8PartitionBlocked as error:
        raise V8BHistoricalAcquisitionBlocked(error.reason) from error


# ---------------------------------------------------------------------------
# Step 2: freeze approval (V8B_DESIGN_FREEZE_APPROVAL.json)
# ---------------------------------------------------------------------------


def _git_show_bytes(commit: str, path: str, *, read_failed_reason: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(CANONICAL_REPOSITORY_ROOT), "show", commit + ":" + path],
            capture_output=True,
            check=False,
            timeout=10,
            env=_isolated_git_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8BHistoricalAcquisitionBlocked(read_failed_reason) from error
    if result.returncode != 0:
        raise V8BHistoricalAcquisitionBlocked(read_failed_reason)
    return result.stdout


def _read_design_freeze_approval_from_verified_head(verified_head: str) -> dict[str, Any]:
    commit = _require_git_commit(verified_head, "DESIGN_FREEZE_APPROVAL_HEAD_INVALID")
    raw = _git_show_bytes(
        commit, DESIGN_FREEZE_APPROVAL_GIT_PATH, read_failed_reason="V8B_DESIGN_FREEZE_APPROVAL_READ_FAILED"
    )
    approval = _strict_json_object(
        raw,
        invalid_reason="V8B_DESIGN_FREEZE_APPROVAL_INVALID_JSON",
        duplicate_reason="V8B_DESIGN_FREEZE_APPROVAL_DUPLICATE_KEY",
    )
    if approval.get("frozen_design_git_commit") != V8B_FROZEN_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("V8B_DESIGN_FREEZE_APPROVAL_COMMIT_MISMATCH")
    if approval.get("approval_status") != "APPROVED":
        raise V8BHistoricalAcquisitionBlocked("V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")
    if approval.get("final_independent_review_result") != "PASS":
        raise V8BHistoricalAcquisitionBlocked("V8B_DESIGN_FREEZE_FINAL_REVIEW_NOT_PASS")
    if approval.get("preservation_recheck_result") != "PASS":
        raise V8BHistoricalAcquisitionBlocked("V8B_DESIGN_FREEZE_PRESERVATION_RECHECK_NOT_PASS")
    return approval


# ---------------------------------------------------------------------------
# Step 3: reviewed implementation binding (V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json)
# ---------------------------------------------------------------------------


def _read_production_implementation_review_from_verified_head(
    verified_head: str, implementation_git_commit: str
) -> dict[str, Any]:
    """Require a PASSing `INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` record.

    This gate's artifact does not exist in this repository yet -- this
    implementation phase (§12.1) precedes that separate, later review gate
    (§12.3) in the frozen sequence. In real production this therefore
    always BLOCKs today, which is the correct fail-closed behavior; tests
    inject a fake reader to exercise the ordering and downstream checks.
    """
    commit = _require_git_commit(verified_head, "IMPLEMENTATION_REVIEW_HEAD_INVALID")
    raw = _git_show_bytes(
        commit, IMPLEMENTATION_REVIEW_GIT_PATH, read_failed_reason="V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"
    )
    review = _strict_json_object(
        raw,
        invalid_reason="V8B_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON",
        duplicate_reason="V8B_PRODUCTION_IMPLEMENTATION_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(IMPLEMENTATION_REVIEW_FIELDS):
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != IMPLEMENTATION_REVIEW_SCHEMA_VERSION:
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_STUDY_MISMATCH")
    if review["reviewed_implementation_git_commit"] != implementation_git_commit:
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_COMMIT_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED")
    return review


# ---------------------------------------------------------------------------
# Step 4: classifier blob binding (§7.6)
# ---------------------------------------------------------------------------


def _resolve_classifier_blob_sha_from_verified_head(verified_head: str) -> str:
    commit = _require_git_commit(verified_head, "CLASSIFIER_BLOB_HEAD_INVALID")
    try:
        result = subprocess.run(
            ["git", "-C", str(CANONICAL_REPOSITORY_ROOT), "rev-parse", commit + ":" + CANONICAL_PARSER_CLASSIFIER_FILE],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            env=_isolated_git_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8BHistoricalAcquisitionBlocked("V8B_CLASSIFIER_BLOB_RESOLUTION_FAILED") from error
    if result.returncode != 0:
        raise V8BHistoricalAcquisitionBlocked("V8B_CLASSIFIER_BLOB_RESOLUTION_FAILED")
    blob_sha = result.stdout.strip()
    if len(blob_sha) != 40 or any(char not in "0123456789abcdef" for char in blob_sha):
        raise V8BHistoricalAcquisitionBlocked("V8B_CLASSIFIER_BLOB_RESOLUTION_FAILED")
    return blob_sha


def verify_classifier_blob(classifier_blob_sha: str) -> None:
    if classifier_blob_sha != CANONICAL_PARSER_CLASSIFIER_BLOB_SHA:
        raise V8BHistoricalAcquisitionBlocked("V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH")


# ---------------------------------------------------------------------------
# Step 5: Asia/Tokyo ZoneInfo pre-network check (§7.7)
# ---------------------------------------------------------------------------


def _default_zoneinfo_loader() -> Any:
    return ZoneInfo("Asia/Tokyo")


def verify_asia_tokyo_zoneinfo_available(zoneinfo_loader: Callable[[], Any] = _default_zoneinfo_loader) -> None:
    try:
        zoneinfo_loader()
    except Exception as error:  # noqa: BLE001 - any failure here must fail closed before network
        raise V8BHistoricalAcquisitionBlocked("V8B_ASIA_TOKYO_ZONEINFO_UNAVAILABLE") from error


# ---------------------------------------------------------------------------
# Step 6a: T1B successor allocation-authority chain (§11.3.A-D)
# ---------------------------------------------------------------------------


def _validated_t1b_binding(
    allocation_artifact: Mapping[str, Any],
    trust_pin: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    try:
        pin = validate_trust_pin(trust_pin)
    except V8BTrustPinBlocked as error:
        raise V8BHistoricalAcquisitionBlocked("V8B_TRUST_PIN_INVALID:" + error.reason) from error
    if pin["authorization_status"] != "AUTHORIZED":
        raise V8BHistoricalAcquisitionBlocked("V8B_TRUST_PIN_NOT_AUTHORIZED")
    try:
        artifact = verify_allocation_artifact_self_hash(allocation_artifact)
    except V8BAllocationBlocked as error:
        raise V8BHistoricalAcquisitionBlocked("V8B_ALLOCATION_ARTIFACT_INVALID:" + error.reason) from error

    if artifact["artifact_self_hash"] != pin["authorized_allocation_artifact_self_hash"]:
        raise V8BHistoricalAcquisitionBlocked("V8B_TRUST_PIN_ALLOCATION_ARTIFACT_MISMATCH")
    if artifact["v8b_frozen_design_commit"] != V8B_FROZEN_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("V8B_DESIGN_COMMIT_MISMATCH:T1B_ARTIFACT")
    if pin["v8b_frozen_design_commit"] != V8B_FROZEN_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("V8B_DESIGN_COMMIT_MISMATCH:T1B_PIN")

    tickers = tuple(artifact["t1b_tickers"])
    if V8_BLOCK_SIZE != 300 or len(tickers) != 300:
        raise V8BHistoricalAcquisitionBlocked("V8B_T1B_TICKER_COUNT_INVALID")
    computed = ticker_list_sha256(tickers)
    if computed != artifact["t1b_ticker_list_sha256"]:
        raise V8BHistoricalAcquisitionBlocked("V8B_T1B_TICKER_LIST_SHA_MISMATCH:ARTIFACT")
    if computed != pin["t1b_ticker_list_sha256"]:
        raise V8BHistoricalAcquisitionBlocked("V8B_T1B_TICKER_LIST_SHA_MISMATCH:TRUST_PIN")

    binding = {
        "authorized_allocation_artifact_self_hash": artifact["artifact_self_hash"],
        "parent_v8_partition_manifest_sha256": artifact["parent_v8_partition_manifest_sha256"],
        "parent_v8_partition_implementation_commit": artifact["parent_v8_partition_implementation_commit"],
        "trust_pin_human_gate": pin["human_gate"],
    }
    return tickers, binding


# ---------------------------------------------------------------------------
# Step 6b: T2 original immutable V8 authority + OPTION_2 bridge (§11.3.E)
# ---------------------------------------------------------------------------


def _read_trusted_partition_anchor_bytes(raw: bytes) -> dict[str, Any]:
    """Mirrors `src/v8_historical_acquisition.py`'s equivalent, unmodified,
    validation semantics -- reimplemented locally (not imported) so this
    module has no non-exported dependency on that file (module docstring)."""
    anchor = _strict_json_object(
        raw,
        invalid_reason="TRUSTED_PARTITION_ANCHOR_INVALID_JSON",
        duplicate_reason="TRUSTED_PARTITION_ANCHOR_DUPLICATE_KEY",
    )
    if not isinstance(anchor, Mapping) or set(anchor) != set(TRUSTED_PARTITION_ANCHOR_FIELDS):
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_ANCHOR_SCHEMA_INVALID")
    if anchor["schema_version"] != TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION:
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION_MISMATCH")
    if anchor["study_name"] != V8_STUDY_NAME:
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_ANCHOR_STUDY_NAME_MISMATCH")
    if anchor["design_commit"] != V8_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_ANCHOR_DESIGN_COMMIT_MISMATCH")
    status = anchor["authorization_status"]
    if status not in ("NOT_AUTHORIZED", "AUTHORIZED"):
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_AUTHORIZATION_STATUS_INVALID")
    manifest_sha = anchor["authorized_partition_manifest_sha256"]
    implementation_git_commit = anchor["authorized_partition_implementation_git_commit"]
    if status == "NOT_AUTHORIZED":
        if manifest_sha is not None or implementation_git_commit is not None:
            raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_UNAUTHORIZED_FIELDS_INVALID")
    else:
        if (
            not isinstance(manifest_sha, str)
            or len(manifest_sha) != 64
            or any(char not in "0123456789abcdef" for char in manifest_sha)
        ):
            raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_MANIFEST_SHA_INVALID")
        _require_git_commit(implementation_git_commit, "IMPLEMENTATION_GIT_COMMIT_INVALID")
    if not isinstance(anchor["authorization_note"], str):
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_AUTHORIZATION_NOTE_INVALID")
    return dict(anchor)


def _read_trusted_partition_anchor_from_verified_head(verified_head: str) -> dict[str, Any]:
    commit = _require_git_commit(verified_head, "IMPLEMENTATION_GIT_COMMIT_INVALID")
    raw = _git_show_bytes(
        commit, TRUSTED_PARTITION_ANCHOR_GIT_PATH, read_failed_reason="TRUSTED_PARTITION_ANCHOR_GIT_READ_FAILED"
    )
    return _read_trusted_partition_anchor_bytes(raw)


def _read_t2_authority_bridge_bytes(raw: bytes) -> dict[str, Any]:
    bridge = _strict_json_object(
        raw,
        invalid_reason="V8B_T2_AUTHORITY_BRIDGE_INVALID_JSON",
        duplicate_reason="V8B_T2_AUTHORITY_BRIDGE_DUPLICATE_KEY",
    )
    if not isinstance(bridge, Mapping) or set(bridge) != set(T2_AUTHORITY_BRIDGE_FIELDS):
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_SCHEMA_INVALID")
    if bridge["schema_version"] != T2_AUTHORITY_BRIDGE_SCHEMA_VERSION:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_SCHEMA_VERSION_MISMATCH")
    if bridge["study"] != STUDY_NAME:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_STUDY_MISMATCH")
    if bridge["role"] != "SEALED_HOLDOUT":
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_ROLE_MISMATCH")
    if bridge["source_authority"] != "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY":
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_SOURCE_AUTHORITY_MISMATCH")
    if bridge["option"] != "OPTION_2":
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_OPTION_MISMATCH")
    if bridge["v8b_frozen_design_commit"] != V8B_FROZEN_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_DESIGN_COMMIT_MISMATCH")
    if bridge["t2_membership_reassignment"] != "PROHIBITED":
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_MEMBERSHIP_REASSIGNMENT_INVALID")
    if bridge["t2_acquired_before_authorized_acquisition"] is not False:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_ACQUIRED_BEFORE_INVALID")
    if bridge["t2_research_open_count_before_official_opening"] != 0:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_OPEN_COUNT_INVALID")
    if bridge["v8_trusted_partition_json_mutated_or_repinned"] is not False:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_ANCHOR_MUTATION_INVALID")
    return dict(bridge)


def _read_t2_authority_bridge_from_verified_head(verified_head: str) -> dict[str, Any]:
    commit = _require_git_commit(verified_head, "IMPLEMENTATION_GIT_COMMIT_INVALID")
    raw = _git_show_bytes(
        commit, T2_AUTHORITY_BRIDGE_GIT_PATH, read_failed_reason="V8B_T2_AUTHORITY_BRIDGE_GIT_READ_FAILED"
    )
    return _read_t2_authority_bridge_bytes(raw)


def _validated_t2_binding(
    partition_manifest_path: str | os.PathLike[str],
    anchor: Mapping[str, Any],
    bridge: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")
    try:
        partition_manifest = read_partition_manifest(partition_manifest_path)
    except V8PartitionBlocked as error:
        raise V8BHistoricalAcquisitionBlocked(error.reason) from error

    partition_manifest_sha256 = partition_manifest["manifest_sha256"]
    if (
        not isinstance(partition_manifest_sha256, str)
        or len(partition_manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in partition_manifest_sha256)
    ):
        raise V8BHistoricalAcquisitionBlocked("PARTITION_MANIFEST_SHA_INVALID")
    if partition_manifest_sha256 != anchor["authorized_partition_manifest_sha256"]:
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    partition_implementation_git_commit = partition_manifest["partition_implementation_git_commit"]
    if (
        _require_git_commit(partition_implementation_git_commit, "IMPLEMENTATION_GIT_COMMIT_INVALID")
        != anchor["authorized_partition_implementation_git_commit"]
    ):
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")

    if partition_manifest["schema_version"] != V8_PARTITION_MANIFEST_SCHEMA_VERSION:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_MANIFEST_SCHEMA_VERSION_MISMATCH")
    if partition_manifest["study_name"] != V8_STUDY_NAME:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_MANIFEST_STUDY_NAME_MISMATCH")
    if partition_manifest["design_commit"] != V8_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_MANIFEST_DESIGN_COMMIT_MISMATCH")
    if partition_manifest["source_reproduction_status"] != "PASS":
        raise V8BHistoricalAcquisitionBlocked("PARTITION_MANIFEST_SOURCE_REPRODUCTION_NOT_PASS")
    if partition_manifest["source_host"] != "www.jpx.co.jp":
        raise V8BHistoricalAcquisitionBlocked("PARTITION_MANIFEST_SOURCE_HOST_MISMATCH")

    assignments = partition_manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or "T2" not in assignments:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING:T2")
    assignment = assignments["T2"]
    if not isinstance(assignment, list):
        raise V8BHistoricalAcquisitionBlocked("PARTITION_BLOCK_ASSIGNMENT_INVALID:T2")
    tickers = tuple(assignment)
    if V8_BLOCK_SIZE != 300 or len(tickers) != 300:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_TICKER_COUNT_INVALID:T2")
    if ticker_list_sha256(tickers) != partition_manifest["t2_ticker_list_sha256"]:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_TICKER_LIST_SHA_MISMATCH:T2")

    # OPTION_2 bridge cross-checks (§11.3.E): the bridge must point at
    # exactly this verified parent manifest and this verified T2 ticker set.
    if bridge["authorized_parent_v8_partition_manifest_sha256"] != partition_manifest_sha256:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH")
    if bridge["expected_t2_ticker_list_sha256"] != ticker_list_sha256(tickers):
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_TICKER_LIST_SHA_MISMATCH")

    binding = {
        "v8_partition_manifest_sha256": partition_manifest_sha256,
        "v8_partition_implementation_commit": partition_implementation_git_commit,
        "v8_trust_anchor_git_identity": bridge["v8_trust_anchor_git_identity"],
        "option_2_bridge_human_gate": bridge["human_gate"],
    }
    return tickers, binding


# ---------------------------------------------------------------------------
# Step 8: output/staging safety helpers
# ---------------------------------------------------------------------------


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# Transport (duplicated, generic, exact-origin Yahoo opener -- see
# src/v7_yahoo_collector.py::HOST; not a modification of that file)
# ---------------------------------------------------------------------------


def _require_exact_origin(value: object, *, hostname: str, invalid_reason: str) -> str:
    if not isinstance(value, str):
        raise V8BHistoricalAcquisitionBlocked(invalid_reason)
    try:
        parsed = urllib.parse.urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise V8BHistoricalAcquisitionBlocked(invalid_reason) from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise V8BHistoricalAcquisitionBlocked(invalid_reason)
    return value


def _require_trusted_yahoo_url(value: object) -> str:
    return _require_exact_origin(value, hostname=HOST, invalid_reason="V8B_YAHOO_SOURCE_ORIGIN_INVALID")


class _TrustedYahooRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _require_trusted_yahoo_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_trusted_yahoo_opener(request_obj: Any) -> Any:
    _require_trusted_yahoo_url(getattr(request_obj, "full_url", None))
    opener = urllib.request.build_opener(_TrustedYahooRedirectHandler())
    return opener.open(request_obj)


def _require_trusted_yahoo_response_url(response: Any) -> None:
    response_url = getattr(response, "url", None)
    if response_url is None:
        geturl = getattr(response, "geturl", None)
        response_url = geturl() if callable(geturl) else None
    _require_trusted_yahoo_url(response_url)


class _RecordingResponse:
    def __init__(self, response: Any, capture: bytearray) -> None:
        self._response = response
        self._capture = capture

    @property
    def status(self) -> Any:
        return getattr(self._response, "status", None)

    @property
    def url(self) -> Any:
        return getattr(self._response, "url", None)

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        value = self._response.read(*args, **kwargs)
        if isinstance(value, bytes):
            self._capture.extend(value)
        return value

    def close(self) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def _classify_error(error: BaseException) -> tuple[str, bool]:
    code = getattr(error, "code", None)
    if code == 429:
        return "HTTP_STATUS_429", True
    reason = getattr(error, "reason", str(error))
    text = str(reason)
    return text, text == "HTTP_STATUS_429"


def _wait_for_next_request_start(
    index: int,
    previous_start: float | None,
    monotonic_clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    if index == 0 or previous_start is None:
        return monotonic_clock()
    elapsed = monotonic_clock() - previous_start
    remaining = MIN_REQUEST_INTERVAL_SECONDS - elapsed
    if remaining > 0:
        sleep_fn(remaining)
    return monotonic_clock()


# ---------------------------------------------------------------------------
# Malformed-OHLCV quality gate (§7.6 -- F1_C1 thresholds)
# ---------------------------------------------------------------------------


def _malformed_ohlcv_policy_metadata() -> dict[str, Any]:
    return {
        "policy_name": MALFORMED_OHLCV_POLICY_NAME,
        "invalid_fraction_numerator": MALFORMED_OHLCV_INVALID_FRACTION_NUMERATOR,
        "invalid_fraction_denominator": MALFORMED_OHLCV_INVALID_FRACTION_DENOMINATOR,
        "max_consecutive_invalid_returned_rows": MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS,
        "full_p_hist_check_required": MALFORMED_OHLCV_FULL_P_HIST_CHECK_REQUIRED,
        "test_years": list(MALFORMED_OHLCV_TEST_YEARS),
        "expected_calendar_missing_dates_treated_as_malformed": (
            MALFORMED_OHLCV_EXPECTED_CALENDAR_MISSING_DATES_TREATED_AS_MALFORMED
        ),
        "threshold_exceedance_action": MALFORMED_OHLCV_THRESHOLD_EXCEEDANCE_ACTION,
    }


def _require_valid_malformed_ohlcv_policy_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(MALFORMED_OHLCV_POLICY_METADATA_FIELDS):
        raise V8BHistoricalAcquisitionBlocked("MALFORMED_OHLCV_POLICY_METADATA_SCHEMA_INVALID")
    if dict(value) != _malformed_ohlcv_policy_metadata():
        raise V8BHistoricalAcquisitionBlocked("MALFORMED_OHLCV_POLICY_METADATA_MISMATCH")
    return dict(value)


def _malformed_ohlcv_returned_observations(
    valid_rows: Sequence[Mapping[str, Any]],
    invalid_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, bool]]:
    observations = [(str(row["trading_date"]), True) for row in valid_rows]
    observations.extend((str(row["trading_date"]), False) for row in invalid_rows)
    observations.sort(key=lambda item: item[0])
    return observations


def _malformed_ohlcv_check_window(
    observations: Sequence[tuple[str, bool]],
    *,
    allow_empty: bool,
    fraction_reason: str,
    consecutive_reason: str,
) -> None:
    total = len(observations)
    if total == 0:
        if allow_empty:
            return
        raise V8BHistoricalAcquisitionBlocked("MALFORMED_OHLCV_QUALITY_GATE:EMPTY_SERIES")
    invalid_count = sum(1 for _, is_valid in observations if not is_valid)
    # F1_C1 exact integer comparison: invalid_count/total <= 1/252 is
    # authoritative as invalid_count * 252 <= total -- never floating-point.
    if invalid_count * MALFORMED_OHLCV_INVALID_FRACTION_DENOMINATOR > total * MALFORMED_OHLCV_INVALID_FRACTION_NUMERATOR:
        raise V8BHistoricalAcquisitionBlocked(fraction_reason)
    run = 0
    for _, is_valid in observations:
        if is_valid:
            run = 0
        else:
            run += 1
            if run > MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS:
                raise V8BHistoricalAcquisitionBlocked(consecutive_reason)


def _require_malformed_ohlcv_quality_gate(
    valid_rows: Sequence[Mapping[str, Any]],
    invalid_rows: Sequence[Mapping[str, Any]],
) -> None:
    observations = _malformed_ohlcv_returned_observations(valid_rows, invalid_rows)
    _malformed_ohlcv_check_window(
        observations,
        allow_empty=False,
        fraction_reason="MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED",
        consecutive_reason="MALFORMED_OHLCV_QUALITY_GATE:CONSECUTIVE_EXCEEDED",
    )
    for year in MALFORMED_OHLCV_TEST_YEARS:
        prefix = str(year) + "-"
        year_observations = [item for item in observations if item[0].startswith(prefix)]
        _malformed_ohlcv_check_window(
            year_observations,
            allow_empty=True,
            fraction_reason="MALFORMED_OHLCV_QUALITY_GATE:TEST_YEAR_FRACTION_EXCEEDED",
            consecutive_reason="MALFORMED_OHLCV_QUALITY_GATE:TEST_YEAR_CONSECUTIVE_EXCEEDED",
        )


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [{field: row[field] for field in FRAME_FIELDS} for row in rows],
        key=lambda row: (str(row["ticker"]), str(row["trading_date"])),
    )


def _parse_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise V8BHistoricalAcquisitionBlocked("INVALID_DATE:" + field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V8BHistoricalAcquisitionBlocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise V8BHistoricalAcquisitionBlocked("INVALID_DATE:" + field)
    return parsed


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8BHistoricalAcquisitionBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Public production boundary
# ---------------------------------------------------------------------------


def acquire_v8b_historical_block_bundle(
    *,
    output_root: str | os.PathLike[str],
    block: str,
    partition_manifest_path: str | os.PathLike[str] | None = None,
    t1b_allocation_artifact_path: str | os.PathLike[str] | None = None,
    t1b_trust_pin_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Acquire one verified `T1B`/`T2` block in production.

    ``block == "T2"`` requires ``partition_manifest_path`` (the real,
    private, already-built V8 partition manifest). ``block == "T1B"``
    requires ``t1b_allocation_artifact_path`` (the private §11.3.B
    allocation artifact) and ``t1b_trust_pin_path`` (the future public
    §11.3.C trust pin -- read here from a caller-supplied path rather than
    a verified Git object, since `V8B_TRUSTED_ALLOCATION.json` does not yet
    exist in this repository; once `CREATE_V8B_TRUSTED_ALLOCATION_PIN`
    actually happens, production wiring must switch this one input to a
    verified-Git-HEAD read exactly like the T2 anchor/bridge above -- a
    small future wiring change, not a schema or validation-semantics
    change).
    """
    return _acquire_production_v8b_historical_block_bundle_with_dependencies(
        output_root=output_root,
        block=block,
        partition_manifest_path=partition_manifest_path,
        t1b_allocation_artifact_path=t1b_allocation_artifact_path,
        t1b_trust_pin_path=t1b_trust_pin_path,
        git_commit_resolver=_resolve_verified_canonical_production_git_commit,
        design_freeze_approval_reader=_read_design_freeze_approval_from_verified_head,
        implementation_review_reader=_read_production_implementation_review_from_verified_head,
        classifier_blob_resolver=_resolve_classifier_blob_sha_from_verified_head,
        zoneinfo_loader=_default_zoneinfo_loader,
        git_anchor_reader=_read_trusted_partition_anchor_from_verified_head,
        git_bridge_reader=_read_t2_authority_bridge_from_verified_head,
        opener=_default_trusted_yahoo_opener,
        clock=lambda: datetime.now(timezone.utc),
        monotonic_clock=time.monotonic,
        sleep_fn=time.sleep,
    )


def _acquire_production_v8b_historical_block_bundle_with_dependencies(
    *,
    output_root: str | os.PathLike[str],
    block: str,
    partition_manifest_path: str | os.PathLike[str] | None,
    t1b_allocation_artifact_path: str | os.PathLike[str] | None,
    t1b_trust_pin_path: str | os.PathLike[str] | None,
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    implementation_review_reader: Callable[[str, str], Mapping[str, Any]],
    classifier_blob_resolver: Callable[[str], str],
    zoneinfo_loader: Callable[[], Any],
    git_anchor_reader: Callable[[str], Mapping[str, Any]],
    git_bridge_reader: Callable[[str], Mapping[str, Any]],
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Private fake-only seam for exercising the full production call ordering.

    Runs, strictly in order and strictly before any Yahoo request or the
    per-ticker acquisition loop: (1) repo/provenance, (2) freeze approval,
    (3) reviewed implementation binding, (4) classifier blob, (5) ZoneInfo,
    (6) authority chain (T1B or T2), (7) block count/hash (folded into (6)'s
    binding checks), (8) output/staging safety.
    """
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8BHistoricalAcquisitionBlocked("V8B_BLOCK_ACQUISITION_PROHIBITED:" + str(block))

    # (1) repo/provenance
    implementation_git_commit = _require_git_commit(git_commit_resolver(), "IMPLEMENTATION_GIT_COMMIT_INVALID")

    # (2) freeze approval
    design_freeze_approval_reader(implementation_git_commit)

    # (3) reviewed implementation binding
    implementation_review_reader(implementation_git_commit, implementation_git_commit)

    # (4) classifier blob
    classifier_blob_sha = classifier_blob_resolver(implementation_git_commit)
    verify_classifier_blob(classifier_blob_sha)

    # (5) Asia/Tokyo ZoneInfo
    verify_asia_tokyo_zoneinfo_available(zoneinfo_loader)

    # (6)+(7) authority chain, block count/hash
    if block == "T1B":
        if t1b_allocation_artifact_path is None or t1b_trust_pin_path is None:
            raise V8BHistoricalAcquisitionBlocked("V8B_T1B_INPUTS_MISSING")
        allocation_artifact = _strict_json_object(
            Path(t1b_allocation_artifact_path).read_bytes(),
            invalid_reason="V8B_ALLOCATION_ARTIFACT_INVALID_JSON",
            duplicate_reason="V8B_ALLOCATION_ARTIFACT_DUPLICATE_KEY",
        )
        trust_pin = _strict_json_object(
            Path(t1b_trust_pin_path).read_bytes(),
            invalid_reason="V8B_TRUST_PIN_INVALID_JSON",
            duplicate_reason="V8B_TRUST_PIN_DUPLICATE_KEY",
        )
        tickers, authority_binding = _validated_t1b_binding(allocation_artifact, trust_pin)
    else:
        if partition_manifest_path is None:
            raise V8BHistoricalAcquisitionBlocked("V8B_T2_INPUTS_MISSING")
        anchor = git_anchor_reader(implementation_git_commit)
        bridge = git_bridge_reader(implementation_git_commit)
        tickers, authority_binding = _validated_t2_binding(partition_manifest_path, anchor, bridge)

    return _acquire_v8b_block_bundle_with_validated_inputs(
        output_root=output_root,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        block=block,
        tickers=tickers,
        authority_binding=authority_binding,
        implementation_git_commit=implementation_git_commit,
        classifier_blob_sha=classifier_blob_sha,
        opener=opener,
        clock=clock,
        monotonic_clock=monotonic_clock,
        sleep_fn=sleep_fn,
        request_start=REQUEST_START,
        request_end_exclusive=REQUEST_END_EXCLUSIVE,
    )


def _acquire_v8b_block_bundle_with_validated_inputs(
    *,
    output_root: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    block: str,
    tickers: Sequence[str],
    authority_binding: Mapping[str, Any],
    implementation_git_commit: str,
    classifier_blob_sha: str,
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_start: str = REQUEST_START,
    request_end_exclusive: str = REQUEST_END_EXCLUSIVE,
) -> dict[str, Any]:
    """(8) output/staging safety, then the per-ticker Yahoo acquisition loop.

    Every row is fetched via the already-accepted
    ``v7_yahoo_collector.fetch_chart_once`` transport (sequential, one HTTP
    request per ticker); this function adds no additional network path.
    Exceeding ``POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE`` for
    any ticker BLOCKs the whole acquisition -- the entire block is
    atomically discarded, never published partially, and no row is ever
    filled, interpolated, imputed, or repaired.
    """
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8BHistoricalAcquisitionBlocked("V8B_BLOCK_ACQUISITION_PROHIBITED:" + str(block))

    start = _parse_date(request_start, "request_start")
    end = _parse_date(request_end_exclusive, "request_end_exclusive")
    if not start < end:
        raise V8BHistoricalAcquisitionBlocked("REQUEST_DATE_BOUNDS_INVALID")

    tickers_list = list(tickers)
    if not tickers_list or len(set(tickers_list)) != len(tickers_list):
        raise V8BHistoricalAcquisitionBlocked("V8B_TICKER_LIST_INVALID")
    for ticker in tickers_list:
        try:
            if canonical_ticker(ticker) != ticker:
                raise V8BHistoricalAcquisitionBlocked("V8B_TICKER_NOT_CANONICAL")
        except V7YahooCollectorBlocked as error:
            raise V8BHistoricalAcquisitionBlocked("V8B_TICKER_NOT_CANONICAL") from error

    try:
        output_path = require_absolute_output_path_outside_repository(output_root, repository_root)
    except V8PartitionBlocked as error:
        raise V8BHistoricalAcquisitionBlocked(error.reason) from error

    acquisitions_root = output_path / ACQUISITIONS_DIRNAME
    final_dir = acquisitions_root / block
    if final_dir.exists():
        raise V8BHistoricalAcquisitionBlocked("V8B_ACQUISITION_ALREADY_EXISTS:" + block)
    acquisitions_root.mkdir(parents=True, exist_ok=True)
    if any(entry.name.startswith(block + ".staging-") for entry in acquisitions_root.iterdir()):
        raise V8BHistoricalAcquisitionBlocked("V8B_PARTIAL_ACQUISITION_COMMIT:" + block)

    started_dt = _utc_timestamp(clock(), "acquisition_started_utc")

    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f"{block}.staging-", dir=str(acquisitions_root)))
        (staging / RAW_DIRNAME).mkdir()

        payload_manifest: list[dict[str, Any]] = []
        all_price_rows: list[dict[str, Any]] = []
        all_split_rows: list[dict[str, Any]] = []
        invalid_reason_counts: Counter[str] = Counter()
        request_count = 0
        http_429_count = 0
        success_transport_count = 0
        previous_start: float | None = None

        for index, ticker in enumerate(tickers_list):
            previous_start = _wait_for_next_request_start(index, previous_start, monotonic_clock, sleep_fn)
            capture = bytearray()

            def recording_opener(request_obj: Any, *, _capture: bytearray = capture) -> Any:
                _require_trusted_yahoo_url(getattr(request_obj, "full_url", None))
                response = opener(request_obj)
                try:
                    _require_trusted_yahoo_response_url(response)
                except BaseException:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                    raise
                return _RecordingResponse(response, _capture)

            request_count += 1
            try:
                parsed = fetch_chart_once(
                    ticker, request_start, request_end_exclusive, opener=recording_opener
                )
            except V7YahooCollectorBlocked as error:
                reason = error.reason
                if reason == "HTTP_STATUS_429":
                    http_429_count += 1
                raise V8BHistoricalAcquisitionBlocked("TICKER_FETCH_BLOCKED:" + reason) from error
            except BaseException as error:
                reason, is_429 = _classify_error(error)
                if is_429:
                    http_429_count += 1
                raise V8BHistoricalAcquisitionBlocked("TICKER_FETCH_BLOCKED:" + reason) from error

            payload_bytes = bytes(capture)
            if sha256_bytes(payload_bytes) != parsed.get("payload_sha256"):
                raise V8BHistoricalAcquisitionBlocked("RAW_PAYLOAD_SHA_MISMATCH")
            if len(payload_bytes) != parsed.get("byte_count"):
                raise V8BHistoricalAcquisitionBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH")

            invalid_rows = parsed["invalid_price_rows"]
            valid_rows_raw = parsed["valid_price_rows"]

            _require_malformed_ohlcv_quality_gate(valid_rows_raw, invalid_rows)

            if invalid_rows:
                for row in invalid_rows:
                    invalid_reason_counts[str(row["reason"])] += 1

            _write_bytes(staging / RAW_DIRNAME / (ticker + ".json"), payload_bytes)

            valid_rows = [dict(row) for row in valid_rows_raw]
            split_rows = [dict(row) for row in parsed["canonical_split_events"]]
            all_price_rows.extend(valid_rows)
            all_split_rows.extend(split_rows)

            payload_manifest.append({
                "ticker": ticker,
                "payload_sha256": parsed["payload_sha256"],
                "byte_count": parsed["byte_count"],
                "canonical_price_rows_sha256": parsed["canonical_price_rows_sha256"],
                "canonical_split_events_sha256": parsed["canonical_split_events_sha256"],
                "valid_price_row_count": len(valid_rows),
                "invalid_price_row_count": len(invalid_rows),
                "split_event_count": len(split_rows),
            })
            success_transport_count += 1

        keyed_rows: set[tuple[str, str]] = set()
        for row in all_price_rows:
            key = (str(row["ticker"]), str(row["trading_date"]))
            if key in keyed_rows:
                raise V8BHistoricalAcquisitionBlocked("DUPLICATE_TICKER_DATE")
            keyed_rows.add(key)
        canonical_rows = _canonical_rows(all_price_rows)
        canonical_splits = sorted(all_split_rows, key=lambda row: (row["effective_date"], row["ticker"]))

        completed_dt = _utc_timestamp(clock(), "acquisition_completed_utc")
        if completed_dt < started_dt:
            raise V8BHistoricalAcquisitionBlocked("ACQUISITION_CLOCK_NONMONOTONIC")

        payload_manifest_bytes = canonical_json_bytes(payload_manifest)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "v8b_frozen_design_commit": V8B_FROZEN_DESIGN_COMMIT,
            "implementation_git_commit": implementation_git_commit,
            "reviewed_production_implementation_commit": implementation_git_commit,
            "block": block,
            "role": BLOCK_ROLE[block],
            "status": BLOCK_STATUS[block],
            "sealed": BLOCK_SEALED[block],
            "research_access_authorized": False,
            "authority_chain": BLOCK_AUTHORITY_CHAIN[block],
            "authority_binding": dict(authority_binding),
            "data_source": DATA_SOURCE,
            "data_source_host": DATA_SOURCE_HOST,
            "data_source_schema": DATA_SOURCE_SCHEMA,
            "canonical_parser_classifier_blob_sha": classifier_blob_sha,
            "request_start": request_start,
            "request_end_exclusive": request_end_exclusive,
            "ticker_count": len(tickers_list),
            "ticker_list_sha256": _ticker_list_sha(tickers_list),
            "request_count": request_count,
            "retry_count": RETRY_COUNT,
            "http_429_count": http_429_count,
            "success_transport_count": success_transport_count,
            "valid_price_row_count": len(canonical_rows),
            "invalid_price_row_count": sum(entry["invalid_price_row_count"] for entry in payload_manifest),
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "malformed_ohlcv_policy": _malformed_ohlcv_policy_metadata(),
            "split_event_count": len(canonical_splits),
            "payload_manifest": payload_manifest,
            "payload_manifest_sha256": sha256_bytes(payload_manifest_bytes),
            "canonical_price_rows_sha256": canonical_sha256(canonical_rows),
            "canonical_split_events_sha256": canonical_sha256(canonical_splits),
            "acquisition_started_utc": _timestamp_text(started_dt),
            "acquisition_completed_utc": _timestamp_text(completed_dt),
            "validation_access_count": 0,
            "feature_computation_count": 0,
            "outcome_access_count": 0,
            "sealed_holdout_access_count": 0,
        }
        if set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
            raise V8BHistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")

        _write_bytes(staging / MANIFEST_FILENAME, canonical_json_bytes(manifest))
        if block == "T2":
            sealed_record = {
                "sealed": True,
                "research_access_authorized": False,
                "note": (
                    "Procedural seal, not cryptographic. Opening this block for "
                    "feature generation, candidate generation, validation, "
                    "backtest, or profit evaluation requires the FROZEN_FINAL_"
                    "CANDIDATE gate and §10's still-unresolved security "
                    "requirements; the official V8B access-guard API in this "
                    "module BLOCKs every such call while sealed=true."
                ),
            }
            _write_bytes(staging / SEALED_FILENAME, canonical_json_bytes(sealed_record))

        os.replace(str(staging), str(final_dir))
        staging = None
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def read_acquisition_manifest(output_root: str | os.PathLike[str], block: str) -> dict[str, Any]:
    """Read-only load of a previously published block manifest.

    Fails closed on duplicate JSON keys and re-validates every immutable
    acquisition-time invariant against this module's own constants.
    """
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8BHistoricalAcquisitionBlocked("V8B_BLOCK_ACQUISITION_PROHIBITED:" + str(block))
    manifest_path = Path(output_root) / ACQUISITIONS_DIRNAME / block / MANIFEST_FILENAME
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_READ_FAILED") from error
    manifest = _strict_json_object(
        raw,
        invalid_reason="ACQUISITION_MANIFEST_INVALID_JSON",
        duplicate_reason="ACQUISITION_MANIFEST_DUPLICATE_KEY",
    )
    if set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
        raise V8BHistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")
    _require_valid_malformed_ohlcv_policy_metadata(manifest["malformed_ohlcv_policy"])
    block_value = manifest.get("block")
    if block_value != block:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_BLOCK_MISMATCH")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_SCHEMA_VERSION_MISMATCH")
    if manifest.get("study_name") != STUDY_NAME:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_STUDY_NAME_MISMATCH")
    if manifest.get("v8b_frozen_design_commit") != V8B_FROZEN_DESIGN_COMMIT:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_DESIGN_COMMIT_MISMATCH")
    if manifest.get("role") != BLOCK_ROLE[block]:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_ROLE_MISMATCH")
    if manifest.get("status") != BLOCK_STATUS[block]:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_STATUS_MISMATCH")
    if manifest.get("sealed") is not BLOCK_SEALED[block]:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_SEALED_MISMATCH")
    if manifest.get("research_access_authorized") is not False:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_RESEARCH_ACCESS_INVARIANT_VIOLATED")
    for field in ACQUISITION_MANIFEST_ZERO_ACCESS_COUNTER_FIELDS:
        value = manifest.get(field)
        if type(value) is not int or value != 0:
            raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_ACCESS_COUNTER_INVARIANT_VIOLATED")
    if manifest.get("retry_count") != RETRY_COUNT:
        raise V8BHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_RETRY_COUNT_MISMATCH")
    return dict(manifest)


# ---------------------------------------------------------------------------
# T2 sealed-holdout access guard (procedural, not cryptographic)
# ---------------------------------------------------------------------------


class V8BSealedHoldoutBlocked(RuntimeError):
    """Raised by the official V8B access-guard API when a caller attempts to
    open a sealed block for research use.

    Procedural seal only, mirroring `src/v8_historical_acquisition.py`'s
    `V8SealedHoldoutBlocked` precedent: it refuses to proceed when the
    manifest it is handed says ``sealed=true``; it does not physically lock
    the underlying raw files.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_research_access_authorized(acquisition_manifest: Mapping[str, Any], operation: str) -> None:
    if not isinstance(acquisition_manifest, Mapping) or "sealed" not in acquisition_manifest:
        raise V8BSealedHoldoutBlocked("ACQUISITION_MANIFEST_INVALID:" + operation)
    if acquisition_manifest["sealed"] is True:
        raise V8BSealedHoldoutBlocked("SEALED_HOLDOUT_ACCESS_DENIED:" + operation)
    if acquisition_manifest.get("research_access_authorized") is not True:
        raise V8BSealedHoldoutBlocked("RESEARCH_ACCESS_NOT_AUTHORIZED:" + operation)


def open_for_feature_generation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "feature_generation")


def open_for_candidate_generation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "candidate_generation")


def open_for_validation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "validation")


def open_for_backtest(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "backtest")


def open_for_profit_evaluation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "profit_evaluation")


__all__ = [
    "ACQUISITIONS_DIRNAME",
    "ACQUISITION_MANIFEST_FIELDS",
    "ALLOWED_ACQUISITION_BLOCKS",
    "BLOCK_AUTHORITY_CHAIN",
    "BLOCK_ROLE",
    "BLOCK_SEALED",
    "BLOCK_STATUS",
    "CANONICAL_PARSER_CLASSIFIER_BLOB_SHA",
    "CANONICAL_PARSER_CLASSIFIER_FILE",
    "CANONICAL_PARSER_CLASSIFIER_GIT_COMMIT",
    "DATA_SOURCE",
    "DATA_SOURCE_HOST",
    "DATA_SOURCE_SCHEMA",
    "MALFORMED_OHLCV_EXPECTED_CALENDAR_MISSING_DATES_TREATED_AS_MALFORMED",
    "MALFORMED_OHLCV_FULL_P_HIST_CHECK_REQUIRED",
    "MALFORMED_OHLCV_INVALID_FRACTION_DENOMINATOR",
    "MALFORMED_OHLCV_INVALID_FRACTION_NUMERATOR",
    "MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS",
    "MALFORMED_OHLCV_POLICY_METADATA_FIELDS",
    "MALFORMED_OHLCV_POLICY_NAME",
    "MALFORMED_OHLCV_TEST_YEARS",
    "MALFORMED_OHLCV_THRESHOLD_EXCEEDANCE_ACTION",
    "MANIFEST_FILENAME",
    "MIN_REQUEST_INTERVAL_SECONDS",
    "MODE",
    "PAYLOAD_RECORD_FIELDS",
    "PROHIBITED_ACQUISITION_BLOCKS",
    "RAW_DIRNAME",
    "REQUEST_END_EXCLUSIVE",
    "REQUEST_START",
    "RETRY_COUNT",
    "SCHEMA_VERSION",
    "SEALED_FILENAME",
    "STUDY_NAME",
    "T2_AUTHORITY_BRIDGE_FIELDS",
    "T2_AUTHORITY_BRIDGE_SCHEMA_VERSION",
    "TRUSTED_PARTITION_ANCHOR_FIELDS",
    "TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION",
    "V8BHistoricalAcquisitionBlocked",
    "V8BSealedHoldoutBlocked",
    "V8B_FROZEN_DESIGN_COMMIT",
    "acquire_v8b_historical_block_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "open_for_backtest",
    "open_for_candidate_generation",
    "open_for_feature_generation",
    "open_for_profit_evaluation",
    "open_for_validation",
    "read_acquisition_manifest",
    "sha256_bytes",
    "verify_asia_tokyo_zoneinfo_available",
    "verify_classifier_blob",
]
