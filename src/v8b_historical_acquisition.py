"""V8B raw-only historical OHLCV acquisition for `T1B` (new) and `T2` (reused).

`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §7.6, §7.7, §10, §11, §12.1,
§12.3, §12.4, §12.6. This is the `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_
IMPLEMENTATION` production module (§12's gate sequence), remediated against
`INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW`'s first-round findings.
It does **not** perform, and this repository has not authorized, any real
Yahoo or JPX request, any real `T1B` allocation, or any real private
V8/V8B partition access -- every test exercising this module is
fake/synthetic-only. Importing this module performs no I/O.

This module never imports, reads, or modifies `src/v7_yahoo_collector.py`
or `src/v8_historical_acquisition.py`. It reuses `src.v7_yahoo_collector`
read-only for the already-accepted, generic, single-ticker Yahoo Chart
transport and canonical parser (§13: `SAFE_TO_REUSE`). All Git-provenance
resolution and exact-blob authority verification goes through
`src.v8b_git_provenance` / `src.v8b_production_provenance` -- **not**
`src.v8_partition.resolve_verified_production_git_commit()`, which is
hardcoded to V8's own production branch and cannot serve as V8B's
provenance root (first-round finding HIGH-1). Only
`src.v8_partition.require_absolute_output_path_outside_repository` and
`ticker_list_sha256` (generic, non-trust-bearing utilities) are still
reused from that module.

Two, and only two, logical blocks may be acquired here:

- `T1B` -- new `V8B_HISTORICAL_RESEARCH` validation block, bound to the new
  successor allocation-authority chain: a verified private allocation
  artifact (`src/v8b_allocation.py`) pinned by `V8B_TRUSTED_ALLOCATION.json`,
  read here from a **verified Git object**, never a caller-supplied path
  (first-round finding HIGH-3). That pin file does not exist in this
  repository yet, so `T1B` acquisition fails closed today by construction.
- `T2` -- reused `V8_HISTORICAL_RESEARCH` sealed holdout, bound to the
  original, immutable `V8_TRUSTED_PARTITION.json` authority (verified
  against its exact frozen Git blob, not merely trusted at face value) plus
  the explicit `OPTION_2` bridge (`V8B_T2_AUTHORITY_BRIDGE.json`, §11.3.E).

`T0`, old `T1`, `T3`, and `T_spare` (whole) remain unconditionally
prohibited from acquisition through this module.

Every raw acquisition is gated by `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_
ROW_QUALITY_GATE` (§7.6): `invalid_returned_row_count * 252 <=
total_returned_row_count` (exact integer comparison, never floating-point),
`max_consecutive_invalid_returned_rows = 1`, checked over the full `P_hist`
series and independently over each of the eight frozen production test
years 2018-2025. Before any Yahoo request, production verifies (in order)
repo/provenance, design-freeze approval (exact blob), reviewed-
implementation binding (exact per-file blob equality against the reviewed
commit, §12.3), the classifier blob (§7.6), `Asia/Tokyo` `ZoneInfo`
availability (§7.7), the block's authority chain, and output/staging
safety -- strictly in that order, strictly before the first Yahoo request.
Every lower-layer transport/parser failure is mapped through a finite
whitelist before becoming part of any public reason string; no ticker,
date, URL, private path, or raw exception text is ever exposed (first-round
finding HIGH-6). This module does not export any research-opening
(`open_for_*`) API -- that remains explicitly out of scope until §10's
security requirements are implemented and reviewed (first-round finding
MEDIUM-3).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
    STUDY_NAME as V8_STUDY_NAME,
    V8PartitionBlocked,
    read_partition_manifest,
    require_absolute_output_path_outside_repository,
    ticker_list_sha256,
)

from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_git_blob,
    resolve_verified_v8b_production_git_commit,
)
from src.v8b_production_provenance import (
    EXPECTED_T2_TICKER_COUNT,
    EXPECTED_T2_TICKER_LIST_SHA256,
    EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
    STUDY_NAME as PROVENANCE_STUDY_NAME,
    V8BProductionProvenanceBlocked,
    V8_DESIGN_COMMIT,
    read_and_verify_design_freeze_approval,
    read_and_verify_t2_authority_bridge,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8b_allocation import V8BAllocationBlocked, verify_allocation_artifact_self_hash
from src.v8b_trust_pin import V8BTrustPinBlocked, validate_trust_pin
from src.v8b_t2_reuse_recheck import V8BT2PreservationRecheckBlocked, resolve_and_recheck_t2_reuse_conditions

SCHEMA_VERSION = "V8B_HISTORICAL_ACQUISITION_V1"
STUDY_NAME = PROVENANCE_STUDY_NAME
MODE = "V8B_RAW_HISTORICAL_ACQUISITION"

V8B_FROZEN_DESIGN_COMMIT = EXPECTED_V8B_FROZEN_DESIGN_COMMIT

DATA_SOURCE = "Yahoo Chart"
DATA_SOURCE_HOST = HOST
DATA_SOURCE_SCHEMA = "Yahoo Chart v8/finance/chart interval=1d events=div,splits includeAdjustedClose=true"

REQUEST_START = "2016-04-01"
REQUEST_END_EXCLUSIVE = "2026-01-01"

MIN_REQUEST_INTERVAL_SECONDS = 2.0
RETRY_COUNT = 0

# POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE (§7.6). Stored and
# compared as an exact integer numerator/denominator -- never a float.
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

# The future public §11.3.C trust pin. Does not exist in this repository
# yet (CREATE_V8B_TRUSTED_ALLOCATION_PIN is a separate, later, human-gated
# action) -- production therefore fails closed reading it today, exactly
# like V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json.
T1B_TRUST_PIN_GIT_PATH = "V8B_TRUSTED_ALLOCATION.json"

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
    """Fail-closed V8B historical acquisition transport, schema, or seal error.

    ``authorization_consumed`` is ``False`` for every pre-network failure
    (confirmation, provenance, freeze, review, classifier, ZoneInfo,
    authority chain, output/staging safety) and ``True`` for any failure
    at or after the first Yahoo request begins -- a safe boolean, never a
    ticker or path (round-2 finding HIGH-1).
    """

    def __init__(self, reason: str, *, authorization_consumed: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.authorization_consumed = authorization_consumed


# Frozen production confirmation literals (§12's T1B_RAW_ACQUISITION_HUMAN_
# GATE / T2_RAW_ACQUISITION_HUMAN_GATE). Upstream-approved as mechanical
# confirmation syntax only -- these literals do NOT themselves constitute
# real human authorization; they exist so an operator cannot invoke the
# wrong block's acquisition by accident, mirroring this repository's
# existing `--confirmation V8_PRODUCTION_ACQUIRE_T1` convention.
T1B_ACQUISITION_CONFIRMATION = "V8B_PRODUCTION_ACQUIRE_T1B"
T2_ACQUISITION_CONFIRMATION = "V8B_PRODUCTION_ACQUIRE_T2"
ACQUISITION_CONFIRMATION_BY_BLOCK = {
    "T1B": T1B_ACQUISITION_CONFIRMATION,
    "T2": T2_ACQUISITION_CONFIRMATION,
}


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8BHistoricalAcquisitionBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BHistoricalAcquisitionBlocked(missing_reason)
    if isinstance(reason, str):
        return V8BHistoricalAcquisitionBlocked(reason)
    return V8BHistoricalAcquisitionBlocked("PROVENANCE_CHECK_FAILED")


# ---------------------------------------------------------------------------
# Canonical hashing / JSON helpers
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
# Step 4: classifier blob binding (§7.6)
# ---------------------------------------------------------------------------


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
# Step 6a: T1B successor allocation-authority chain, Git-bound trust pin
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


def read_t1b_trust_pin_from_verified_head(
    repository_root, verified_head: str, git_object_reader: Callable[[str, str, str], bytes] = read_git_object_bytes
) -> dict[str, Any]:
    """Read the future public §11.3.C trust pin from a **verified Git
    object** -- never a caller-supplied path (HIGH-3 remediation). The real
    ``V8B_TRUSTED_ALLOCATION.json`` does not exist in this repository yet,
    so this fails closed with ``V8B_TRUSTED_ALLOCATION_MISSING`` today.

    Public (not module-private) so other production boundaries -- e.g. the
    §12.6 acquisition-artifact verifier's production resolver -- read this
    trust pin through the exact same Git-bound logic, rather than each
    re-implementing its own copy (round-3 finding HIGH-4).
    """
    raw = git_object_reader(repository_root, verified_head, T1B_TRUST_PIN_GIT_PATH)
    return _strict_json_object(
        raw,
        invalid_reason="V8B_TRUSTED_ALLOCATION_INVALID_JSON",
        duplicate_reason="V8B_TRUSTED_ALLOCATION_DUPLICATE_KEY",
    )


# ---------------------------------------------------------------------------
# Step 6b: T2 original immutable V8 authority + OPTION_2 bridge (§11.3.E)
# ---------------------------------------------------------------------------


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
    if partition_implementation_git_commit != anchor["authorized_partition_implementation_git_commit"]:
        raise V8BHistoricalAcquisitionBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")

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

    # Pinned to the exact frozen literal values (HIGH-4) -- never merely
    # "internally consistent with whatever this manifest/anchor says".
    if len(tickers) != EXPECTED_T2_TICKER_COUNT:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_TICKER_COUNT_INVALID:T2")
    computed_hash = ticker_list_sha256(tickers)
    if computed_hash != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_TICKER_LIST_SHA_MISMATCH:T2")
    if computed_hash != partition_manifest["t2_ticker_list_sha256"]:
        raise V8BHistoricalAcquisitionBlocked("PARTITION_TICKER_LIST_SHA_MISMATCH:T2")

    # OPTION_2 bridge cross-checks (§11.3.E): the bridge must point at
    # exactly this verified parent manifest.
    if bridge["authorized_parent_v8_partition_manifest_sha256"] != partition_manifest_sha256:
        raise V8BHistoricalAcquisitionBlocked("V8B_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH")

    binding = {
        "v8_partition_manifest_sha256": partition_manifest_sha256,
        "v8_partition_implementation_commit": partition_implementation_git_commit,
        "v8_trust_anchor_git_identity": bridge["v8_trust_anchor_git_identity"],
        "option_2_bridge_human_gate": bridge["human_gate"],
    }
    return tickers, binding


# ---------------------------------------------------------------------------
# Transport (generic, exact-origin Yahoo opener -- see
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


# ---------------------------------------------------------------------------
# HIGH-6: finite, privacy-safe transport-error classification.
#
# Never touches str(error), error.args, error.reason on an unrecognised
# exception, a URL, a ticker, a date, or a private path. Every branch
# below returns one of a small, fixed set of generic reason codes.
# ---------------------------------------------------------------------------

# The exact, fixed vocabulary `src/v7_yahoo_collector.py::V7YahooCollectorBlocked`
# is known (by direct reading of that file, read-only, at review time) to
# ever raise. None of these literal/prefixed forms can embed a ticker,
# trading date, URL, or private path -- every ":"-suffixed form is bound to
# a fixed, short, alphanumeric/underscore field or section name, verified
# below rather than assumed.
_SAFE_V7_COLLECTOR_REASON_LITERALS = frozenset({
    "EMPTY_TICKER",
    "INVALID_REQUEST_DATE_ORDER",
    "RESPONSE_HOST_MISMATCH",
    "RESPONSE_BYTES_INVALID",
    "PAYLOAD_JSON_INVALID",
    "PAYLOAD_ROOT_INVALID",
    "CHART_ERROR",
    "CHART_RESULT_INVALID",
    "METADATA_MISSING",
    "SYMBOL_MISMATCH",
    "TIMESTAMP_MISSING",
    "TIMESTAMP_INVALID",
    "INDICATORS_MISSING",
    "OUT_OF_REQUEST_WINDOW",
    "DUPLICATE_TRADING_DATE",
    "SPLIT_RATIO_INVALID",
    "SPLIT_OUT_OF_REQUEST_WINDOW",
    "DUPLICATE_SPLIT_EVENT",
    "SPLIT_NUMERATOR_DENOMINATOR_MISSING",
    "SPLIT_NUMERATOR_DENOMINATOR_INVALID",
    "SPLIT_RATIO_MISMATCH",
    "EVENTS_INVALID",
    "SPLITS_INVALID",
    "SPLIT_EVENT_INVALID",
    "ARRAY_LENGTH_MISMATCH",
    "INDICATOR_SECTION_INVALID",
})
_SAFE_V7_COLLECTOR_REASON_PREFIXES = ("INVALID_DATE:", "INDICATOR_SECTION_INVALID:", "ARRAY_LENGTH_MISMATCH:", "HTTP_STATUS_")
_MAX_SAFE_REASON_SUFFIX_LENGTH = 40


def _safe_transport_reason(raw_reason: object) -> str:
    """Map a `V7YahooCollectorBlocked.reason` to a known-safe public reason.

    Anything not exactly matching the known-safe fixed vocabulary --
    including any reason value carrying an unexpected/oversized suffix --
    is replaced with a generic fallback rather than ever being forwarded
    as-is.
    """
    if not isinstance(raw_reason, str):
        return "UNCLASSIFIED_PARSER_ERROR"
    if raw_reason in _SAFE_V7_COLLECTOR_REASON_LITERALS:
        return raw_reason
    for prefix in _SAFE_V7_COLLECTOR_REASON_PREFIXES:
        if raw_reason.startswith(prefix):
            suffix = raw_reason[len(prefix):]
            if suffix and len(suffix) <= _MAX_SAFE_REASON_SUFFIX_LENGTH and all(
                char.isalnum() or char == "_" for char in suffix
            ):
                return raw_reason
            return "UNCLASSIFIED_PARSER_ERROR"
    return "UNCLASSIFIED_PARSER_ERROR"


def _classify_transport_exception(error: BaseException) -> tuple[str, bool]:
    """Classify any non-`V7YahooCollectorBlocked` transport exception.

    Never reads ``str(error)``, ``error.args``, or ``error.reason`` --
    only the exception's numeric HTTP ``code`` (if present) and its Python
    type. Anything unrecognised maps to the fixed generic fallback.
    """
    code = getattr(error, "code", None)
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599:
        if code == 429:
            return "HTTP_STATUS_429", True
        return f"HTTP_STATUS_{code}", False
    if isinstance(error, TimeoutError):
        return "TRANSPORT_TIMEOUT", False
    if isinstance(error, ConnectionError):
        return "TRANSPORT_CONNECTION_ERROR", False
    if isinstance(error, OSError):
        return "TRANSPORT_OS_ERROR", False
    return "UNCLASSIFIED_TRANSPORT_ERROR", False


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


def _write_bytes(path: Path, value: bytes, *, reason: str = "STAGING_WRITE_FAILED") -> None:
    """Write ``value`` to ``path``, fsync'd.

    Round-3 repeat finding HIGH-3: every filesystem exception here is
    mapped to a fixed, generic reason -- never ``str(error)``, ``.args``,
    or the path itself, any of which could otherwise carry a private
    staging path or a ticker-derived filename into a public reason.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise V8BHistoricalAcquisitionBlocked(reason) from error


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
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str] | None = None,
    t1b_allocation_artifact_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Acquire one verified `T1B`/`T2` block in production.

    ``confirmation`` must exactly equal the block-specific frozen literal
    (``T1B_ACQUISITION_CONFIRMATION`` / ``T2_ACQUISITION_CONFIRMATION``) --
    this is mechanical confirmation syntax only, not real human
    authorization (round-2 finding HIGH-1). ``block == "T2"`` requires
    ``partition_manifest_path`` (the real, private, already-built V8
    partition manifest). ``block == "T1B"`` requires
    ``t1b_allocation_artifact_path`` (the private §11.3.B allocation
    artifact -- private data, so a caller-supplied path remains
    appropriate) -- there is deliberately no ``t1b_trust_pin_path``
    parameter: the public §11.3.C trust pin is read from a verified Git
    object, never a caller-suppliable path (HIGH-3 remediation).
    """
    return _acquire_production_v8b_historical_block_bundle_with_dependencies(
        output_root=output_root,
        block=block,
        confirmation=confirmation,
        partition_manifest_path=partition_manifest_path,
        t1b_allocation_artifact_path=t1b_allocation_artifact_path,
        git_commit_resolver=lambda: resolve_verified_v8b_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        classifier_blob_resolver=lambda head: resolve_git_blob(
            CANONICAL_REPOSITORY_ROOT, head, CANONICAL_PARSER_CLASSIFIER_FILE
        ),
        zoneinfo_loader=_default_zoneinfo_loader,
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        bridge_reader=lambda head: read_and_verify_t2_authority_bridge(CANONICAL_REPOSITORY_ROOT, head),
        t2_reuse_recheck_resolver=resolve_and_recheck_t2_reuse_conditions,
        t1b_trust_pin_reader=lambda head: read_t1b_trust_pin_from_verified_head(
            CANONICAL_REPOSITORY_ROOT, head, read_git_object_bytes
        ),
        opener=_default_trusted_yahoo_opener,
        clock=lambda: datetime.now(timezone.utc),
        monotonic_clock=time.monotonic,
        sleep_fn=time.sleep,
    )


def _acquire_production_v8b_historical_block_bundle_with_dependencies(
    *,
    output_root: str | os.PathLike[str],
    block: str,
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str] | None,
    t1b_allocation_artifact_path: str | os.PathLike[str] | None,
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    frozen_design_object_verifier: Callable[[], None],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    classifier_blob_resolver: Callable[[str], str],
    zoneinfo_loader: Callable[[], Any],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    bridge_reader: Callable[[str], Mapping[str, Any]],
    t2_reuse_recheck_resolver: Callable[[str], Mapping[str, Any]],
    t1b_trust_pin_reader: Callable[[str], Mapping[str, Any]],
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Private fake-only seam for exercising the full production call ordering.

    Runs, strictly in order and strictly before any Yahoo request or the
    per-ticker acquisition loop: (0) confirmation token, (1) repo/
    provenance, (2) frozen design object + freeze approval (exact blob),
    (3) reviewed-implementation binding (exact per-file blob equality),
    (4) classifier blob, (5) ZoneInfo, (6) authority chain (T1B or T2;
    T2 additionally requires fresh POST_FREEZE reuse-conditions evidence),
    (7) block count/hash (folded into (6)'s binding checks), (8)
    output/staging safety. ``authorization_consumed`` on the resulting
    manifest -- and on any raised ``V8BHistoricalAcquisitionBlocked`` --
    is ``False`` for every one of these steps and only becomes ``True``
    once the per-ticker loop attempts its first Yahoo request.
    """
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8BHistoricalAcquisitionBlocked("V8B_BLOCK_ACQUISITION_PROHIBITED:" + str(block))

    # (0) explicit, exact, block-specific acquisition-gate confirmation
    # token -- a T1B token can never authorize T2 acquisition or vice versa.
    if confirmation != ACQUISITION_CONFIRMATION_BY_BLOCK[block]:
        raise V8BHistoricalAcquisitionBlocked("V8B_ACQUISITION_CONFIRMATION_INVALID")

    # (1) repo/provenance -- V8B's own branch, never V8's.
    try:
        verified_head = git_commit_resolver()
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error) from error

    # (2) frozen design object + freeze approval (exact blob + exact fields)
    try:
        frozen_design_object_verifier()
        design_freeze_approval_reader(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    # (3) reviewed implementation binding (exact per-file blob equality)
    try:
        review_binding = reviewed_implementation_binder(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_implementation_git_commit = review_binding["reviewed_implementation_git_commit"]

    # (4) classifier blob
    try:
        classifier_blob_sha = classifier_blob_resolver(verified_head)
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error) from error
    verify_classifier_blob(classifier_blob_sha)

    # (5) Asia/Tokyo ZoneInfo
    verify_asia_tokyo_zoneinfo_available(zoneinfo_loader)

    # (6)+(7) authority chain, block count/hash
    if block == "T1B":
        if t1b_allocation_artifact_path is None:
            raise V8BHistoricalAcquisitionBlocked("V8B_T1B_INPUTS_MISSING")
        try:
            allocation_artifact_raw = Path(t1b_allocation_artifact_path).read_bytes()
        except OSError as error:
            raise V8BHistoricalAcquisitionBlocked("V8B_ALLOCATION_ARTIFACT_READ_FAILED") from error
        allocation_artifact = _strict_json_object(
            allocation_artifact_raw,
            invalid_reason="V8B_ALLOCATION_ARTIFACT_INVALID_JSON",
            duplicate_reason="V8B_ALLOCATION_ARTIFACT_DUPLICATE_KEY",
        )
        try:
            trust_pin = t1b_trust_pin_reader(verified_head)
        except V8BGitProvenanceBlocked as error:
            raise _wrap(error, "V8B_TRUSTED_ALLOCATION_MISSING") from error
        tickers, authority_binding = _validated_t1b_binding(allocation_artifact, trust_pin)
    else:
        if partition_manifest_path is None:
            raise V8BHistoricalAcquisitionBlocked("V8B_T2_INPUTS_MISSING")
        try:
            anchor = anchor_reader(verified_head)
            bridge = bridge_reader(verified_head)
        except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
            raise _wrap(error) from error
        # READ_ONLY_T2_REUSE_CONDITIONS_RECHECK (§12.4): fresh POST_FREEZE
        # evidence, never the §12.2 pre-freeze document (round-2 finding
        # HIGH-2). Fails closed today -- the real artifact does not exist.
        try:
            t2_reuse_recheck_resolver(verified_head)
        except (V8BT2PreservationRecheckBlocked, V8BGitProvenanceBlocked) as error:
            raise _wrap(error) from error
        tickers, authority_binding = _validated_t2_binding(partition_manifest_path, anchor, bridge)

    return _acquire_v8b_block_bundle_with_validated_inputs(
        output_root=output_root,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        block=block,
        tickers=tickers,
        authority_binding=authority_binding,
        implementation_git_commit=reviewed_implementation_git_commit,
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
    try:
        acquisitions_root.mkdir(parents=True, exist_ok=True)
        has_partial_staging = any(
            entry.name.startswith(block + ".staging-") for entry in acquisitions_root.iterdir()
        )
    except OSError as error:
        raise V8BHistoricalAcquisitionBlocked("OUTPUT_DIRECTORY_UNAVAILABLE") from error
    if has_partial_staging:
        raise V8BHistoricalAcquisitionBlocked("V8B_PARTIAL_ACQUISITION_COMMIT:" + block)

    started_dt = _utc_timestamp(clock(), "acquisition_started_utc")

    # One-shot human-gate consumption (round-2 finding HIGH-1; exact
    # transition point corrected by round-3 repeat finding MEDIUM-1): the
    # acquisition-gate confirmation is consumed exactly when the first
    # trusted Yahoo opener invocation is actually about to occur --
    # strictly *after* local pacing/request-preparation (monotonic clock
    # wait, URL-origin validation) has already succeeded, and strictly
    # *before* that opener call is made, regardless of the call's outcome.
    # If pacing or request preparation fails first, the opener is never
    # invoked (opener calls = 0) and this stays False. Never reset once
    # True, for this or any later ticker. Everything above this point
    # (steps 0-8) is pre-network and always leaves this False.
    consumed = False

    staging: Path | None = None
    try:
        try:
            staging = Path(tempfile.mkdtemp(prefix=f"{block}.staging-", dir=str(acquisitions_root)))
            (staging / RAW_DIRNAME).mkdir()
        except OSError as error:
            raise V8BHistoricalAcquisitionBlocked("STAGING_DIRECTORY_CREATE_FAILED") from error

        payload_manifest: list[dict[str, Any]] = []
        all_price_rows: list[dict[str, Any]] = []
        all_split_rows: list[dict[str, Any]] = []
        invalid_reason_counts: Counter[str] = Counter()
        request_count = 0
        http_429_count = 0
        success_transport_count = 0
        previous_start: float | None = None

        for index, ticker in enumerate(tickers_list):
            # MEDIUM-1: pacing/request-preparation happens strictly before
            # ``consumed`` can ever become True (see ``recording_opener``
            # below) -- a failure here leaves it at whatever it already
            # was (False, for a first ticker), never flips it, and the
            # underlying opener is never reached (opener calls stay 0 for
            # this ticker).
            try:
                previous_start = _wait_for_next_request_start(index, previous_start, monotonic_clock, sleep_fn)
            except V8BHistoricalAcquisitionBlocked:
                raise
            except BaseException as error:
                raise V8BHistoricalAcquisitionBlocked("REQUEST_PACING_FAILED") from error
            capture = bytearray()

            def recording_opener(request_obj: Any, *, _capture: bytearray = capture) -> Any:
                nonlocal consumed
                _require_trusted_yahoo_url(getattr(request_obj, "full_url", None))
                # MEDIUM-1: consumption happens exactly here -- immediately
                # before the real, underlying opener (the actual network
                # boundary) is invoked -- never earlier. A failure in the
                # URL-origin check above is local request preparation, not
                # network I/O, so it must never flip this to True.
                consumed = True
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
                safe_reason = _safe_transport_reason(error.reason)
                if error.reason == "HTTP_STATUS_429":
                    http_429_count += 1
                raise V8BHistoricalAcquisitionBlocked("TICKER_FETCH_BLOCKED:" + safe_reason) from error
            except BaseException as error:
                reason, is_429 = _classify_transport_exception(error)
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

            _write_bytes(staging / RAW_DIRNAME / (ticker + ".json"), payload_bytes, reason="RAW_PAYLOAD_WRITE_FAILED")

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

        _write_bytes(staging / MANIFEST_FILENAME, canonical_json_bytes(manifest), reason="MANIFEST_WRITE_FAILED")
        if block == "T2":
            sealed_record = {
                "sealed": True,
                "research_access_authorized": False,
                "note": (
                    "Procedural seal, not cryptographic. Opening this block for "
                    "feature generation, candidate generation, validation, "
                    "backtest, or profit evaluation requires the FROZEN_FINAL_"
                    "CANDIDATE gate and §10's still-unresolved security "
                    "requirements; no research-opening API exists in this "
                    "module yet (§10 remains a later, separately reviewed gate)."
                ),
            }
            _write_bytes(staging / SEALED_FILENAME, canonical_json_bytes(sealed_record), reason="SEALED_WRITE_FAILED")

        try:
            os.replace(str(staging), str(final_dir))
        except OSError as error:
            raise V8BHistoricalAcquisitionBlocked("ATOMIC_PUBLISH_FAILED") from error
        staging = None
        return manifest
    except V8BHistoricalAcquisitionBlocked as error:
        error.authorization_consumed = consumed
        raise
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


__all__ = [
    "ACQUISITIONS_DIRNAME",
    "ACQUISITION_CONFIRMATION_BY_BLOCK",
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
    "T1B_ACQUISITION_CONFIRMATION",
    "T1B_TRUST_PIN_GIT_PATH",
    "T2_ACQUISITION_CONFIRMATION",
    "V8BHistoricalAcquisitionBlocked",
    "V8B_FROZEN_DESIGN_COMMIT",
    "acquire_v8b_historical_block_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "read_acquisition_manifest",
    "read_t1b_trust_pin_from_verified_head",
    "sha256_bytes",
    "verify_asia_tokyo_zoneinfo_available",
    "verify_classifier_blob",
]
