"""V8C raw-only historical OHLCV acquisition for `T1C` (new) and `T2` (reused).

`V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §1, §4, §4.1, §4.2, §5, §10, §12.
This is the `V8C_TRANSPORT_AND_ACQUISITION_IMPLEMENTATION` production
module. It does **not** perform, and this repository has not authorized,
any real Yahoo request, any real `T1C` allocation, or any real private
V8/V8C partition access -- every test exercising this module is
fake/synthetic-only. Importing this module performs no I/O.

This module never imports, reads, or modifies `src/v7_yahoo_collector.py`,
`src/v8b_historical_acquisition.py`, or `src/v8_historical_acquisition.py`.
It reuses `src.v7_yahoo_collector` read-only for the already-accepted,
generic, single-ticker Yahoo Chart transport and canonical parser.

Unlike V8B (which used zero transport retries), every real Yahoo request
made by this module -- for both `T1C` and `T2`, and for both readiness and
raw acquisition -- goes through `src.v8c_transport.attempt_with_frozen_
retry`, the shared frozen §4/§4.3 retry/classification layer:
``maximum_attempts_per_ticker=3``, ``maximum_retries=2``,
``backoff_seconds=[5, 30]``, ``jitter=false``.

Two, and only two, logical blocks may be acquired here:

- `T1C` -- bound to the V8C-specific successor allocation-authority chain:
  a verified private allocation artifact (`src/v8c_t1c_allocation.py`)
  pinned by `V8C_TRUSTED_ALLOCATION.json`, read here from a **verified Git
  object**, never a caller-supplied path.
- `T2` -- bound to the original, immutable `V8_TRUSTED_PARTITION.json`
  authority plus the V8C-specific `V8C_T2_AUTHORITY_BRIDGE.json` (§7.2) and
  the live `READ_ONLY_T2_PRESERVATION_RECHECK` (§7.1 ``recheck_2``).

`T0`, old `T1`, V8B's `T1B`, `T3`, and `T_spare` (whole) remain
unconditionally prohibited from acquisition through this module.

Every raw acquisition is gated by `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_
ROW_QUALITY_GATE` (§1.1, carried forward unchanged from V8B):
``invalid_returned_row_count * 252 <= total_returned_row_count`` (exact
integer comparison), ``max_consecutive_invalid_returned_rows = 1``, checked
over the full `P_hist` series and independently over each of the eight
frozen test years 2018-2025.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v7_yahoo_collector import FRAME_FIELDS, HOST, V7YahooCollectorBlocked, canonical_ticker, fetch_chart_once
from src.v8_partition import V8PartitionBlocked, read_partition_manifest, require_absolute_output_path_outside_repository
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_git_blob,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T1C_RAW_ACQUISITION,
    GATE_T2_RAW_ACQUISITION,
    V8CHumanGateConsumptionBlocked,
    consume_gate_once,
    require_gate_not_yet_consumed,
)
from src.v8c_production_provenance import (
    CANONICAL_PARSER_CLASSIFIER_FILE,
    EXPECTED_T2_TICKER_COUNT,
    EXPECTED_T2_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    STUDY_NAME as PROVENANCE_STUDY_NAME,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_trust_pin_independent_review,
    read_and_verify_v8_trusted_partition_anchor,
    verify_classifier_blob,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_t1c_allocation import V8CAllocationBlocked, verify_allocation_artifact_self_hash
from src.v8c_t2_bridge import (
    V8CT2BridgeBlocked,
    read_and_verify_v8c_t2_authority_bridge,
    read_and_verify_v8c_t2_authority_bridge_independent_review,
)
from src.v8c_t2_preservation_recheck import V8CT2PreservationRecheckBlocked, resolve_and_recheck_t2_preservation
from src.v8c_trust_pin import V8CTrustPinBlocked, validate_trust_pin
from src.v8c_transport import (
    BACKOFF_SECONDS,
    JITTER,
    MAXIMUM_ATTEMPTS_PER_TICKER,
    MAXIMUM_RETRIES,
    V8CTransportNamedFailure,
    attempt_with_frozen_retry,
)
from src.v8c_stage_state import V8CStageEvidenceBlocked, read_valid_readiness_pass

SCHEMA_VERSION = "V8C_HISTORICAL_ACQUISITION_V1"
STUDY_NAME = PROVENANCE_STUDY_NAME
MODE = "V8C_RAW_HISTORICAL_ACQUISITION"

V8C_FROZEN_DESIGN_COMMIT = EXPECTED_V8C_FROZEN_DESIGN_COMMIT
V8_DESIGN_COMMIT = "c414d3191cba356734d7ed08bdf1abc7d51fc384"

DATA_SOURCE = "Yahoo Chart"
DATA_SOURCE_HOST = HOST
DATA_SOURCE_SCHEMA = "Yahoo Chart v8/finance/chart interval=1d events=div,splits includeAdjustedClose=true"

REQUEST_START = "2016-04-01"
REQUEST_END_EXCLUSIVE = "2026-01-01"

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

ALLOWED_ACQUISITION_BLOCKS = ("T1C", "T2")
PROHIBITED_ACQUISITION_BLOCKS = ("T0", "T1", "T1B", "T3", "T_spare")

BLOCK_ROLE = {"T1C": "VALIDATION", "T2": "SEALED_HOLDOUT"}
BLOCK_STATUS = {"T1C": "RAW_ACQUIRED_NOT_OPENED", "T2": "RAW_ACQUIRED_SEALED"}
BLOCK_SEALED = {"T1C": False, "T2": True}
BLOCK_AUTHORITY_CHAIN = {
    "T1C": "V8C_SUCCESSOR_ALLOCATION_AUTHORITY",
    "T2": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_V8C_BRIDGE",
}

ACQUISITIONS_DIRNAME = "v8c_acquisitions"
RAW_DIRNAME = "raw"
MANIFEST_FILENAME = "acquisition_manifest.json"
SEALED_FILENAME = "SEALED.json"

T1C_TRUST_PIN_GIT_PATH = "V8C_TRUSTED_ALLOCATION.json"

ACQUISITION_MANIFEST_FIELDS = (
    "schema_version",
    "study_name",
    "v8c_frozen_design_commit",
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
    "max_attempts_per_ticker",
    "max_retries",
    "backoff_seconds",
    "jitter",
    "total_retry_count",
    "total_request_attempts",
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
    "retry_audit_all_intermediate_failures_retryable",
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
    "attempts",
    "retry_count",
    "transport_audit",
)

ACQUISITION_MANIFEST_ZERO_ACCESS_COUNTER_FIELDS = (
    "validation_access_count",
    "feature_computation_count",
    "outcome_access_count",
    "sealed_holdout_access_count",
)

ACQUISITION_GATE_BY_BLOCK = {"T1C": GATE_T1C_RAW_ACQUISITION, "T2": GATE_T2_RAW_ACQUISITION}
T1C_ACQUISITION_CONFIRMATION = "V8C_PRODUCTION_ACQUIRE_T1C"
T2_ACQUISITION_CONFIRMATION = "V8C_PRODUCTION_ACQUIRE_T2"
ACQUISITION_CONFIRMATION_BY_BLOCK = {"T1C": T1C_ACQUISITION_CONFIRMATION, "T2": T2_ACQUISITION_CONFIRMATION}


class V8CHistoricalAcquisitionBlocked(RuntimeError):
    """Fail-closed V8C historical acquisition transport, schema, or seal error.

    ``authorization_consumed`` is ``False`` for every pre-network failure
    and ``True`` for any failure at or after the first real Yahoo request
    begins -- a safe boolean, never a ticker or path.
    """

    def __init__(self, reason: str, *, authorization_consumed: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.authorization_consumed = authorization_consumed


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CHistoricalAcquisitionBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CHistoricalAcquisitionBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CHistoricalAcquisitionBlocked(reason)
    return V8CHistoricalAcquisitionBlocked("PROVENANCE_CHECK_FAILED")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V8CHistoricalAcquisitionBlocked("NONFINITE_VALUE") from error


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
                raise V8CHistoricalAcquisitionBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CHistoricalAcquisitionBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8CHistoricalAcquisitionBlocked(invalid_reason)
    return parsed


# ---------------------------------------------------------------------------
# Transport (exact-origin trusted Yahoo opener)
# ---------------------------------------------------------------------------


def _require_trusted_yahoo_url(value: object) -> str:
    if not isinstance(value, str):
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
    try:
        parsed = urllib.parse.urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
    return value


class _TrustedYahooRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        try:
            _require_trusted_yahoo_url(newurl)
        except V8CTransportNamedFailure as error:
            raise error
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_trusted_yahoo_opener(request_obj: Any) -> Any:
    _require_trusted_yahoo_url(getattr(request_obj, "full_url", None))
    opener = urllib.request.build_opener(_TrustedYahooRedirectHandler())
    return opener.open(request_obj)


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


_SAFE_V7_COLLECTOR_REASON_LITERALS = frozenset({
    "EMPTY_TICKER", "INVALID_REQUEST_DATE_ORDER", "RESPONSE_HOST_MISMATCH", "RESPONSE_BYTES_INVALID",
    "PAYLOAD_JSON_INVALID", "PAYLOAD_ROOT_INVALID", "CHART_ERROR", "CHART_RESULT_INVALID", "METADATA_MISSING",
    "SYMBOL_MISMATCH", "TIMESTAMP_MISSING", "TIMESTAMP_INVALID", "INDICATORS_MISSING", "OUT_OF_REQUEST_WINDOW",
    "DUPLICATE_TRADING_DATE", "SPLIT_RATIO_INVALID", "SPLIT_OUT_OF_REQUEST_WINDOW", "DUPLICATE_SPLIT_EVENT",
    "SPLIT_NUMERATOR_DENOMINATOR_MISSING", "SPLIT_NUMERATOR_DENOMINATOR_INVALID", "SPLIT_RATIO_MISMATCH",
    "EVENTS_INVALID", "SPLITS_INVALID", "SPLIT_EVENT_INVALID", "ARRAY_LENGTH_MISMATCH", "INDICATOR_SECTION_INVALID",
})


def _classify_v7_collector_error(reason: object) -> str:
    """Map `V7YahooCollectorBlocked.reason` to one of the frozen §4.3 named
    nonretryable conditions for the transport wrapper. Never a substring/
    message heuristic -- only exact literal membership."""
    if reason == "SYMBOL_MISMATCH":
        return "SYMBOL_MISMATCH"
    if reason == "RESPONSE_HOST_MISMATCH":
        return "RESPONSE_HOST_MISMATCH"
    return "PARSER_SCHEMA_FAILURE"


def _fetch_one_ticker_with_retry_and_gate(
    ticker: str,
    request_start: str,
    request_end_exclusive: str,
    opener: Callable[[Any], Any],
    sleep_fn: Callable[[float], None],
    *,
    gate_consumer: Callable[[], None],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Identical to ``_fetch_one_ticker_with_retry`` except the very first
    real opener invocation across the *entire* acquisition run durably
    consumes the raw-acquisition gate immediately before it occurs --
    never before local pacing/request-construction, and never again for
    any later ticker or retry."""
    capture_holder: list[bytearray] = []

    def attempt() -> dict[str, Any]:
        capture = bytearray()
        capture_holder.append(capture)

        def recording_opener(request_obj: Any) -> Any:
            _require_trusted_yahoo_url(getattr(request_obj, "full_url", None))
            gate_consumer()
            response = opener(request_obj)
            try:
                _require_trusted_yahoo_url(getattr(response, "url", None))
            except V8CTransportNamedFailure:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
                raise
            return _RecordingResponse(response, capture)

        try:
            return fetch_chart_once(ticker, request_start, request_end_exclusive, opener=recording_opener)
        except V7YahooCollectorBlocked as error:
            raise V8CTransportNamedFailure(_classify_v7_collector_error(error.reason)) from error

    fingerprint_material = {
        "logical_request_identity": ticker,
        "request_start": request_start,
        "request_end_exclusive": request_end_exclusive,
        "provider": DATA_SOURCE,
        "host": HOST,
        "request_parameters": {"interval": "1d", "events": "div,splits", "includeAdjustedClose": True},
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_material)).hexdigest()
    result, audit = attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn, request_fingerprint=fingerprint)
    raw_bytes = bytes(capture_holder[-1])
    return result, raw_bytes, audit


# ---------------------------------------------------------------------------
# Malformed-OHLCV quality gate (§1.1, carried forward from V8B unchanged)
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
        raise V8CHistoricalAcquisitionBlocked("MALFORMED_OHLCV_POLICY_METADATA_SCHEMA_INVALID")
    if dict(value) != _malformed_ohlcv_policy_metadata():
        raise V8CHistoricalAcquisitionBlocked("MALFORMED_OHLCV_POLICY_METADATA_MISMATCH")
    return dict(value)


def _malformed_ohlcv_returned_observations(
    valid_rows: Sequence[Mapping[str, Any]], invalid_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[str, bool]]:
    observations = [(str(row["trading_date"]), True) for row in valid_rows]
    observations.extend((str(row["trading_date"]), False) for row in invalid_rows)
    observations.sort(key=lambda item: item[0])
    return observations


def _malformed_ohlcv_check_window(
    observations: Sequence[tuple[str, bool]], *, allow_empty: bool, fraction_reason: str, consecutive_reason: str
) -> None:
    total = len(observations)
    if total == 0:
        if allow_empty:
            return
        raise V8CHistoricalAcquisitionBlocked("MALFORMED_OHLCV_QUALITY_GATE:EMPTY_SERIES")
    invalid_count = sum(1 for _, is_valid in observations if not is_valid)
    if invalid_count * MALFORMED_OHLCV_INVALID_FRACTION_DENOMINATOR > total * MALFORMED_OHLCV_INVALID_FRACTION_NUMERATOR:
        raise V8CHistoricalAcquisitionBlocked(fraction_reason)
    run = 0
    for _, is_valid in observations:
        if is_valid:
            run = 0
        else:
            run += 1
            if run > MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS:
                raise V8CHistoricalAcquisitionBlocked(consecutive_reason)


def _require_malformed_ohlcv_quality_gate(
    valid_rows: Sequence[Mapping[str, Any]], invalid_rows: Sequence[Mapping[str, Any]]
) -> None:
    observations = _malformed_ohlcv_returned_observations(valid_rows, invalid_rows)
    _malformed_ohlcv_check_window(
        observations, allow_empty=False,
        fraction_reason="MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED",
        consecutive_reason="MALFORMED_OHLCV_QUALITY_GATE:CONSECUTIVE_EXCEEDED",
    )
    for year in MALFORMED_OHLCV_TEST_YEARS:
        prefix = str(year) + "-"
        year_observations = [item for item in observations if item[0].startswith(prefix)]
        _malformed_ohlcv_check_window(
            year_observations, allow_empty=True,
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
        raise V8CHistoricalAcquisitionBlocked("INVALID_DATE:" + field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V8CHistoricalAcquisitionBlocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise V8CHistoricalAcquisitionBlocked("INVALID_DATE:" + field)
    return parsed


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8CHistoricalAcquisitionBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Authority-chain binding (§6, §9, §7.2)
# ---------------------------------------------------------------------------


def read_t1c_trust_pin_from_verified_head(
    repository_root, verified_head: str, git_object_reader: Callable[[str, str, str], bytes] = read_git_object_bytes
) -> dict[str, Any]:
    raw = git_object_reader(repository_root, verified_head, T1C_TRUST_PIN_GIT_PATH)
    return _strict_json_object(
        raw, invalid_reason="V8C_TRUSTED_ALLOCATION_INVALID_JSON", duplicate_reason="V8C_TRUSTED_ALLOCATION_DUPLICATE_KEY"
    )


def _validated_t1c_binding(
    allocation_artifact: Mapping[str, Any], trust_pin: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Any]]:
    try:
        pin = validate_trust_pin(trust_pin)
    except V8CTrustPinBlocked as error:
        raise V8CHistoricalAcquisitionBlocked("V8C_TRUST_PIN_INVALID:" + error.reason) from error
    if pin["authorization_status"] != "AUTHORIZED":
        raise V8CHistoricalAcquisitionBlocked("V8C_TRUST_PIN_NOT_AUTHORIZED")
    try:
        artifact = verify_allocation_artifact_self_hash(allocation_artifact)
    except V8CAllocationBlocked as error:
        raise V8CHistoricalAcquisitionBlocked("V8C_ALLOCATION_ARTIFACT_INVALID:" + error.reason) from error

    if artifact["artifact_self_hash"] != pin["authorized_allocation_artifact_self_hash"]:
        raise V8CHistoricalAcquisitionBlocked("V8C_TRUST_PIN_ALLOCATION_ARTIFACT_MISMATCH")
    if artifact["v8c_frozen_design_commit"] != V8C_FROZEN_DESIGN_COMMIT:
        raise V8CHistoricalAcquisitionBlocked("V8C_DESIGN_COMMIT_MISMATCH:T1C_ARTIFACT")
    if pin["v8c_frozen_design_commit"] != V8C_FROZEN_DESIGN_COMMIT:
        raise V8CHistoricalAcquisitionBlocked("V8C_DESIGN_COMMIT_MISMATCH:T1C_PIN")

    tickers = tuple(artifact["t1c_tickers"])
    if len(tickers) != 300:
        raise V8CHistoricalAcquisitionBlocked("V8C_T1C_TICKER_COUNT_INVALID")
    computed = _ticker_list_sha(tickers)
    if computed != artifact["t1c_ticker_list_sha256"]:
        raise V8CHistoricalAcquisitionBlocked("V8C_T1C_TICKER_LIST_SHA_MISMATCH:ARTIFACT")
    if computed != pin["t1c_ticker_list_sha256"]:
        raise V8CHistoricalAcquisitionBlocked("V8C_T1C_TICKER_LIST_SHA_MISMATCH:TRUST_PIN")

    binding = {
        "authorized_allocation_artifact_self_hash": artifact["artifact_self_hash"],
        "parent_v8_partition_manifest_sha256": artifact["parent_v8_partition_manifest_sha256"],
        "parent_v8_partition_implementation_commit": artifact["parent_v8_partition_implementation_commit"],
        "trust_pin_human_gate": pin["human_gate"],
    }
    return tickers, binding


def _validated_t2_binding(
    partition_manifest_path: str | os.PathLike[str], anchor: Mapping[str, Any], bridge: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8CHistoricalAcquisitionBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")
    try:
        partition_manifest = read_partition_manifest(partition_manifest_path)
    except V8PartitionBlocked as error:
        raise V8CHistoricalAcquisitionBlocked(error.reason) from error

    partition_manifest_sha256 = partition_manifest["manifest_sha256"]
    if partition_manifest_sha256 != anchor["authorized_partition_manifest_sha256"]:
        raise V8CHistoricalAcquisitionBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    partition_implementation_git_commit = partition_manifest["partition_implementation_git_commit"]
    if partition_implementation_git_commit != anchor["authorized_partition_implementation_git_commit"]:
        raise V8CHistoricalAcquisitionBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    if partition_manifest["study_name"] != "V8_HISTORICAL_RESEARCH":
        raise V8CHistoricalAcquisitionBlocked("PARTITION_MANIFEST_STUDY_NAME_MISMATCH")
    if partition_manifest["design_commit"] != V8_DESIGN_COMMIT:
        raise V8CHistoricalAcquisitionBlocked("PARTITION_MANIFEST_DESIGN_COMMIT_MISMATCH")

    assignments = partition_manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or "T2" not in assignments:
        raise V8CHistoricalAcquisitionBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING:T2")
    tickers = tuple(assignments["T2"])
    if len(tickers) != EXPECTED_T2_TICKER_COUNT:
        raise V8CHistoricalAcquisitionBlocked("PARTITION_TICKER_COUNT_INVALID:T2")
    computed_hash = _ticker_list_sha(tickers)
    if computed_hash != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8CHistoricalAcquisitionBlocked("PARTITION_TICKER_LIST_SHA_MISMATCH:T2")
    if computed_hash != partition_manifest["t2_ticker_list_sha256"]:
        raise V8CHistoricalAcquisitionBlocked("PARTITION_TICKER_LIST_SHA_MISMATCH:T2")

    if bridge["authorized_parent_v8_partition_manifest_sha256"] != partition_manifest_sha256:
        raise V8CHistoricalAcquisitionBlocked("V8C_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH")

    binding = {
        "v8_partition_manifest_sha256": partition_manifest_sha256,
        "v8_partition_implementation_commit": partition_implementation_git_commit,
        "v8_trust_anchor_git_identity": bridge["v8_trust_anchor_git_identity"],
        "v8c_t2_authority_bridge_human_gate": bridge["exact_human_bridge_authorization_identity"],
    }
    return tickers, binding


# ---------------------------------------------------------------------------
# Public production boundary
# ---------------------------------------------------------------------------


def acquire_v8c_historical_block_bundle(
    *,
    output_root: str | os.PathLike[str],
    block: str,
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str] | None = None,
    t1c_allocation_artifact_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Acquire one verified `T1C`/`T2` block in production. **Not
    executed** by this implementation phase."""
    manifest = _acquire_production_v8c_historical_block_bundle_with_dependencies(
        output_root=output_root,
        block=block,
        confirmation=confirmation,
        partition_manifest_path=partition_manifest_path,
        t1c_allocation_artifact_path=t1c_allocation_artifact_path,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(CANONICAL_REPOSITORY_ROOT, head),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, head),
        classifier_blob_resolver=lambda head: resolve_git_blob(CANONICAL_REPOSITORY_ROOT, head, CANONICAL_PARSER_CLASSIFIER_FILE),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        bridge_reader=lambda head: read_and_verify_v8c_t2_authority_bridge(CANONICAL_REPOSITORY_ROOT, head),
        bridge_review_reader=lambda head, blob: read_and_verify_v8c_t2_authority_bridge_independent_review(
            CANONICAL_REPOSITORY_ROOT, head, expected_bridge_git_blob_sha=blob
        ),
        t2_preservation_recheck_resolver=resolve_and_recheck_t2_preservation,
        t1c_trust_pin_reader=lambda head: read_t1c_trust_pin_from_verified_head(CANONICAL_REPOSITORY_ROOT, head, read_git_object_bytes),
        trust_pin_review_reader=lambda head, artifact_hash, human_gate: read_and_verify_trust_pin_independent_review(
            CANONICAL_REPOSITORY_ROOT, head, expected_allocation_artifact_self_hash=artifact_hash, expected_trust_pin_human_gate=human_gate
        ),
        readiness_pass_reader=lambda stage, implementation_commit, classifier_sha, authority: read_valid_readiness_pass(
            CANONICAL_CONSUMPTION_STATE_ROOT,
            stage=stage,
            frozen_design_commit=V8C_FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=implementation_commit,
            classifier_blob_sha=classifier_sha,
            authority_prerequisites=authority,
        ),
        opener=_default_trusted_yahoo_opener,
        clock=lambda: datetime.now(timezone.utc),
        sleep_fn=time.sleep,
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )
    return public_acquisition_summary(manifest)


def _acquire_production_v8c_historical_block_bundle_with_dependencies(
    *,
    output_root: str | os.PathLike[str],
    block: str,
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str] | None,
    t1c_allocation_artifact_path: str | os.PathLike[str] | None,
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    frozen_design_object_verifier: Callable[[], None],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    classifier_blob_resolver: Callable[[str], str],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    bridge_reader: Callable[[str], Mapping[str, Any]],
    bridge_review_reader: Callable[[str, str], Mapping[str, Any]],
    t2_preservation_recheck_resolver: Callable[[], Mapping[str, Any]],
    t1c_trust_pin_reader: Callable[[str], Mapping[str, Any]],
    trust_pin_review_reader: Callable[[str, str, str], Mapping[str, Any]],
    readiness_pass_reader: Callable[[str, str, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    consumption_state_root: str | os.PathLike[str],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8CHistoricalAcquisitionBlocked("V8C_BLOCK_ACQUISITION_PROHIBITED")

    if confirmation != ACQUISITION_CONFIRMATION_BY_BLOCK[block]:
        raise V8CHistoricalAcquisitionBlocked("V8C_ACQUISITION_CONFIRMATION_INVALID")

    try:
        require_gate_not_yet_consumed(consumption_state_root, ACQUISITION_GATE_BY_BLOCK[block], V8C_FROZEN_DESIGN_COMMIT)
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CHistoricalAcquisitionBlocked(error.reason) from error

    try:
        verified_head = git_commit_resolver()
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        frozen_design_object_verifier()
        design_freeze_approval_reader(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        review_binding = reviewed_implementation_binder(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_implementation_git_commit = review_binding["reviewed_implementation_git_commit"]

    try:
        classifier_blob_sha = classifier_blob_resolver(verified_head)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error
    try:
        verify_classifier_blob(classifier_blob_sha)
    except V8CProductionProvenanceBlocked as error:
        raise _wrap(error) from error

    if block == "T1C":
        if t1c_allocation_artifact_path is None:
            raise V8CHistoricalAcquisitionBlocked("V8C_T1C_INPUTS_MISSING")
        try:
            allocation_artifact_raw = Path(t1c_allocation_artifact_path).read_bytes()
        except OSError as error:
            raise V8CHistoricalAcquisitionBlocked("V8C_ALLOCATION_ARTIFACT_READ_FAILED") from error
        allocation_artifact = _strict_json_object(
            allocation_artifact_raw,
            invalid_reason="V8C_ALLOCATION_ARTIFACT_INVALID_JSON",
            duplicate_reason="V8C_ALLOCATION_ARTIFACT_DUPLICATE_KEY",
        )
        try:
            trust_pin = t1c_trust_pin_reader(verified_head)
        except V8CGitProvenanceBlocked as error:
            raise _wrap(error, "V8C_TRUSTED_ALLOCATION_MISSING") from error
        tickers, authority_binding = _validated_t1c_binding(allocation_artifact, trust_pin)
        if trust_pin.get("v8c_reviewed_production_implementation_commit") != reviewed_implementation_git_commit:
            raise V8CHistoricalAcquisitionBlocked("V8C_TRUST_PIN_PRODUCTION_IMPLEMENTATION_MISMATCH")
        try:
            trust_pin_review_reader(
                verified_head, authority_binding["authorized_allocation_artifact_self_hash"], authority_binding["trust_pin_human_gate"]
            )
        except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
            raise _wrap(error, "V8C_TRUST_PIN_INDEPENDENT_REVIEW_MISSING") from error
        if readiness_pass_reader is not None:
            try:
                readiness_pass_reader("T1C", reviewed_implementation_git_commit, classifier_blob_sha, authority_binding)
            except Exception as error:  # noqa: BLE001 - stale/missing readiness blocks before raw gate
                raise V8CHistoricalAcquisitionBlocked(getattr(error, "reason", "V8C_READINESS_PASS_REQUIRED")) from error
    else:
        if partition_manifest_path is None:
            raise V8CHistoricalAcquisitionBlocked("V8C_T2_INPUTS_MISSING")
        try:
            anchor = anchor_reader(verified_head)
            bridge = bridge_reader(verified_head)
        except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked, V8CT2BridgeBlocked) as error:
            raise _wrap(error) from error
        try:
            bridge_blob = resolve_git_blob(CANONICAL_REPOSITORY_ROOT, verified_head, "V8C_T2_AUTHORITY_BRIDGE.json")
            bridge_review_reader(verified_head, bridge_blob)
        except (V8CT2BridgeBlocked, V8CGitProvenanceBlocked) as error:
            raise _wrap(error, "V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_MISSING") from error
        try:
            t2_preservation_recheck_resolver()
        except (V8CT2PreservationRecheckBlocked, V8CGitProvenanceBlocked) as error:
            raise _wrap(error) from error
        tickers, authority_binding = _validated_t2_binding(partition_manifest_path, anchor, bridge)
        if readiness_pass_reader is not None:
            try:
                readiness_pass_reader("T2", reviewed_implementation_git_commit, classifier_blob_sha, authority_binding)
            except Exception as error:  # noqa: BLE001
                raise V8CHistoricalAcquisitionBlocked(getattr(error, "reason", "V8C_READINESS_PASS_REQUIRED")) from error

    return _acquire_v8c_block_bundle_with_validated_inputs(
        output_root=output_root,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        block=block,
        tickers=tickers,
        authority_binding=authority_binding,
        implementation_git_commit=reviewed_implementation_git_commit,
        consumption_gate=ACQUISITION_GATE_BY_BLOCK[block],
        consumption_state_root=consumption_state_root,
        classifier_blob_sha=classifier_blob_sha,
        opener=opener,
        clock=clock,
        sleep_fn=sleep_fn,
        request_start=REQUEST_START,
        request_end_exclusive=REQUEST_END_EXCLUSIVE,
    )


def _acquire_v8c_block_bundle_with_validated_inputs(
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
    consumption_gate: str,
    consumption_state_root: str | os.PathLike[str],
    sleep_fn: Callable[[float], None] = time.sleep,
    request_start: str = REQUEST_START,
    request_end_exclusive: str = REQUEST_END_EXCLUSIVE,
) -> dict[str, Any]:
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8CHistoricalAcquisitionBlocked("V8C_BLOCK_ACQUISITION_PROHIBITED")

    start = _parse_date(request_start, "request_start")
    end = _parse_date(request_end_exclusive, "request_end_exclusive")
    if not start < end:
        raise V8CHistoricalAcquisitionBlocked("REQUEST_DATE_BOUNDS_INVALID")

    tickers_list = list(tickers)
    if not tickers_list or len(set(tickers_list)) != len(tickers_list):
        raise V8CHistoricalAcquisitionBlocked("V8C_TICKER_LIST_INVALID")
    for ticker in tickers_list:
        try:
            if canonical_ticker(ticker) != ticker:
                raise V8CHistoricalAcquisitionBlocked("V8C_TICKER_NOT_CANONICAL")
        except V7YahooCollectorBlocked as error:
            raise V8CHistoricalAcquisitionBlocked("V8C_TICKER_NOT_CANONICAL") from error

    try:
        output_path = require_absolute_output_path_outside_repository(output_root, repository_root)
    except V8PartitionBlocked as error:
        raise V8CHistoricalAcquisitionBlocked(error.reason) from error

    acquisitions_root = output_path / ACQUISITIONS_DIRNAME
    final_dir = acquisitions_root / block
    if final_dir.exists():
        raise V8CHistoricalAcquisitionBlocked("V8C_ACQUISITION_ALREADY_EXISTS:" + block)
    try:
        acquisitions_root.mkdir(parents=True, exist_ok=True)
        has_partial_staging = any(entry.name.startswith(block + ".staging-") for entry in acquisitions_root.iterdir())
    except OSError as error:
        raise V8CHistoricalAcquisitionBlocked("OUTPUT_DIRECTORY_UNAVAILABLE") from error
    if has_partial_staging:
        raise V8CHistoricalAcquisitionBlocked("V8C_PARTIAL_ACQUISITION_COMMIT:" + block)

    started_dt = _utc_timestamp(clock(), "acquisition_started_utc")

    consumed = False

    def gate_consumer() -> None:
        nonlocal consumed
        if not consumed:
            try:
                consume_gate_once(consumption_state_root, consumption_gate, V8C_FROZEN_DESIGN_COMMIT, clock=clock)
            except V8CHumanGateConsumptionBlocked as error:
                raise V8CHistoricalAcquisitionBlocked(error.reason) from error
            consumed = True

    staging: Path | None = None
    try:
        try:
            staging = Path(tempfile.mkdtemp(prefix=f"{block}.staging-", dir=str(acquisitions_root)))
            (staging / RAW_DIRNAME).mkdir()
        except OSError as error:
            raise V8CHistoricalAcquisitionBlocked("STAGING_DIRECTORY_CREATE_FAILED") from error

        payload_manifest: list[dict[str, Any]] = []
        all_price_rows: list[dict[str, Any]] = []
        all_split_rows: list[dict[str, Any]] = []
        invalid_reason_counts: Counter[str] = Counter()
        request_count = 0
        http_429_count = 0
        success_transport_count = 0
        total_retry_count = 0
        all_intermediate_retryable = True

        for ticker in tickers_list:
            request_count += 1
            try:
                parsed, payload_bytes, audit = _fetch_one_ticker_with_retry_and_gate(
                    ticker, request_start, request_end_exclusive, opener, sleep_fn, gate_consumer=gate_consumer
                )
            except V7YahooCollectorBlocked as error:  # pragma: no cover - classified before reaching here
                raise V8CHistoricalAcquisitionBlocked("TICKER_FETCH_BLOCKED:UNCLASSIFIED_PARSER_ERROR") from error
            except BaseException as error:
                audit = getattr(error, "transport_audit", None)
                if isinstance(audit, list) and audit:
                    for entry in audit:
                        if entry.get("classification") == "HTTP_429":
                            http_429_count += 1
                    terminal = audit[-1]
                    raise V8CHistoricalAcquisitionBlocked("TICKER_FETCH_BLOCKED:" + str(terminal.get("classification"))) from error
                raise V8CHistoricalAcquisitionBlocked("TICKER_FETCH_BLOCKED:UNKNOWN_FAIL_CLOSED_NONRETRYABLE") from error

            attempts_used = audit["attempts"]
            retry_used = audit["retry_count"]
            total_retry_count += retry_used
            for entry in audit["history"][:-1]:
                if entry.get("classification") == "HTTP_429":
                    http_429_count += 1
                if entry.get("retryable") is not True:
                    all_intermediate_retryable = False

            if sha256_bytes(payload_bytes) != parsed.get("payload_sha256"):
                raise V8CHistoricalAcquisitionBlocked("RAW_PAYLOAD_SHA_MISMATCH")
            if len(payload_bytes) != parsed.get("byte_count"):
                raise V8CHistoricalAcquisitionBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH")

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
                "attempts": attempts_used,
                "retry_count": retry_used,
                "transport_audit": audit["history"],
            })
            success_transport_count += 1

        keyed_rows: set[tuple[str, str]] = set()
        for row in all_price_rows:
            key = (str(row["ticker"]), str(row["trading_date"]))
            if key in keyed_rows:
                raise V8CHistoricalAcquisitionBlocked("DUPLICATE_TICKER_DATE")
            keyed_rows.add(key)
        canonical_rows = _canonical_rows(all_price_rows)
        canonical_splits = sorted(all_split_rows, key=lambda row: (row["effective_date"], row["ticker"]))

        completed_dt = _utc_timestamp(clock(), "acquisition_completed_utc")
        if completed_dt < started_dt:
            raise V8CHistoricalAcquisitionBlocked("ACQUISITION_CLOCK_NONMONOTONIC")

        payload_manifest_bytes = canonical_json_bytes(payload_manifest)
        total_request_attempts = len(tickers_list) + total_retry_count
        if total_retry_count < 0 or total_retry_count > len(tickers_list) * MAXIMUM_RETRIES:
            raise V8CHistoricalAcquisitionBlocked("RETRY_AUDIT_TOTAL_RETRY_COUNT_INVALID")

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "v8c_frozen_design_commit": V8C_FROZEN_DESIGN_COMMIT,
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
            "max_attempts_per_ticker": MAXIMUM_ATTEMPTS_PER_TICKER,
            "max_retries": MAXIMUM_RETRIES,
            "backoff_seconds": list(BACKOFF_SECONDS),
            "jitter": JITTER,
            "total_retry_count": total_retry_count,
            "total_request_attempts": total_request_attempts,
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
            "retry_audit_all_intermediate_failures_retryable": all_intermediate_retryable,
            "acquisition_started_utc": _timestamp_text(started_dt),
            "acquisition_completed_utc": _timestamp_text(completed_dt),
            "validation_access_count": 0,
            "feature_computation_count": 0,
            "outcome_access_count": 0,
            "sealed_holdout_access_count": 0,
        }
        if set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
            raise V8CHistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")

        _write_bytes(staging / MANIFEST_FILENAME, canonical_json_bytes(manifest), reason="MANIFEST_WRITE_FAILED")
        if block == "T2":
            sealed_record = {
                "sealed": True,
                "research_access_authorized": False,
                "note": (
                    "Procedural seal, not cryptographic. Opening this block requires "
                    "the FROZEN_FINAL_CANDIDATE-equivalent gate and §10's security "
                    "requirements; no research-opening API exists in this module."
                ),
            }
            _write_bytes(staging / SEALED_FILENAME, canonical_json_bytes(sealed_record), reason="SEALED_WRITE_FAILED")

        try:
            os.replace(str(staging), str(final_dir))
        except OSError as error:
            raise V8CHistoricalAcquisitionBlocked("ATOMIC_PUBLISH_FAILED") from error
        staging = None
        return manifest
    except V8CHistoricalAcquisitionBlocked as error:
        error.authorization_consumed = consumed
        raise
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _write_bytes(path: Path, value: bytes, *, reason: str = "STAGING_WRITE_FAILED") -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise V8CHistoricalAcquisitionBlocked(reason) from error


def read_acquisition_manifest(output_root: str | os.PathLike[str], block: str) -> dict[str, Any]:
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8CHistoricalAcquisitionBlocked("V8C_BLOCK_ACQUISITION_PROHIBITED")
    manifest_path = Path(output_root) / ACQUISITIONS_DIRNAME / block / MANIFEST_FILENAME
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_READ_FAILED") from error
    manifest = _strict_json_object(raw, invalid_reason="ACQUISITION_MANIFEST_INVALID_JSON", duplicate_reason="ACQUISITION_MANIFEST_DUPLICATE_KEY")
    if set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
        raise V8CHistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")
    _require_valid_malformed_ohlcv_policy_metadata(manifest["malformed_ohlcv_policy"])
    if manifest.get("block") != block:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_BLOCK_MISMATCH")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_SCHEMA_VERSION_MISMATCH")
    if manifest.get("study_name") != STUDY_NAME:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_STUDY_NAME_MISMATCH")
    if manifest.get("v8c_frozen_design_commit") != V8C_FROZEN_DESIGN_COMMIT:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_DESIGN_COMMIT_MISMATCH")
    if manifest.get("role") != BLOCK_ROLE[block]:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_ROLE_MISMATCH")
    if manifest.get("status") != BLOCK_STATUS[block]:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_STATUS_MISMATCH")
    if manifest.get("sealed") is not BLOCK_SEALED[block]:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_SEALED_MISMATCH")
    if manifest.get("research_access_authorized") is not False:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_RESEARCH_ACCESS_INVARIANT_VIOLATED")
    for field in ACQUISITION_MANIFEST_ZERO_ACCESS_COUNTER_FIELDS:
        value = manifest.get(field)
        if type(value) is not int or value != 0:
            raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_ACCESS_COUNTER_INVARIANT_VIOLATED")
    if manifest.get("max_attempts_per_ticker") != MAXIMUM_ATTEMPTS_PER_TICKER or manifest.get("max_retries") != MAXIMUM_RETRIES:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_RETRY_POLICY_MISMATCH")
    if manifest.get("backoff_seconds") != list(BACKOFF_SECONDS) or manifest.get("jitter") is not JITTER:
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_RETRY_POLICY_MISMATCH")
    if manifest.get("total_request_attempts") != manifest.get("ticker_count", 0) + manifest.get("total_retry_count", -1):
        raise V8CHistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_RETRY_AUDIT_INVALID")
    return dict(manifest)


SEALED_RECORD_FIELDS = ("sealed", "research_access_authorized", "note")


def read_sealed_record(output_root: str | os.PathLike[str], block: str) -> dict[str, Any]:
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8CHistoricalAcquisitionBlocked("V8C_BLOCK_ACQUISITION_PROHIBITED")
    sealed_path = Path(output_root) / ACQUISITIONS_DIRNAME / block / SEALED_FILENAME
    try:
        raw = sealed_path.read_bytes()
    except OSError as error:
        raise V8CHistoricalAcquisitionBlocked("SEALED_RECORD_READ_FAILED") from error
    record = _strict_json_object(raw, invalid_reason="SEALED_RECORD_INVALID_JSON", duplicate_reason="SEALED_RECORD_DUPLICATE_KEY")
    if set(record) != set(SEALED_RECORD_FIELDS):
        raise V8CHistoricalAcquisitionBlocked("SEALED_RECORD_SCHEMA_INVALID")
    if record["sealed"] is not True:
        raise V8CHistoricalAcquisitionBlocked("SEALED_RECORD_SEALED_INVARIANT_VIOLATED")
    if record["research_access_authorized"] is not False:
        raise V8CHistoricalAcquisitionBlocked("SEALED_RECORD_RESEARCH_ACCESS_INVARIANT_VIOLATED")
    if not isinstance(record["note"], str) or not record["note"]:
        raise V8CHistoricalAcquisitionBlocked("SEALED_RECORD_NOTE_INVALID")
    return dict(record)


PUBLIC_ACQUISITION_SUMMARY_FIELDS = tuple(field for field in ACQUISITION_MANIFEST_FIELDS if field != "payload_manifest")


def public_acquisition_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
        raise V8CHistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")
    return {key: value for key, value in manifest.items() if key != "payload_manifest"}


__all__ = [
    "ACQUISITIONS_DIRNAME",
    "ACQUISITION_CONFIRMATION_BY_BLOCK",
    "ACQUISITION_GATE_BY_BLOCK",
    "ACQUISITION_MANIFEST_FIELDS",
    "ALLOWED_ACQUISITION_BLOCKS",
    "BLOCK_AUTHORITY_CHAIN",
    "BLOCK_ROLE",
    "BLOCK_SEALED",
    "BLOCK_STATUS",
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
    "MODE",
    "PAYLOAD_RECORD_FIELDS",
    "PROHIBITED_ACQUISITION_BLOCKS",
    "PUBLIC_ACQUISITION_SUMMARY_FIELDS",
    "RAW_DIRNAME",
    "REQUEST_END_EXCLUSIVE",
    "REQUEST_START",
    "SCHEMA_VERSION",
    "SEALED_FILENAME",
    "SEALED_RECORD_FIELDS",
    "STUDY_NAME",
    "T1C_ACQUISITION_CONFIRMATION",
    "T1C_TRUST_PIN_GIT_PATH",
    "T2_ACQUISITION_CONFIRMATION",
    "V8CHistoricalAcquisitionBlocked",
    "V8C_FROZEN_DESIGN_COMMIT",
    "acquire_v8c_historical_block_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "public_acquisition_summary",
    "read_acquisition_manifest",
    "read_sealed_record",
    "read_t1c_trust_pin_from_verified_head",
    "sha256_bytes",
    "verify_classifier_blob",
]
