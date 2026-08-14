"""`READ_ONLY_T1C_ACQUISITION_ARTIFACT_VERIFICATION` / `_T2_` (§10, §10.1).

Data-integrity checks only, over an already-published `T1C`/`T2`
acquisition bundle produced by `src/v8c_historical_acquisition.py`. This
module never computes features, strategy results, profit, trades, or any
other research outcome. Any mismatch is `BLOCK`, with no research opening.
Performs no network access.

Two, and only two, ways to call this module, mirroring
`src.v8b_acquisition_artifact_verification`'s proven pattern:

- ``_verify_acquisition_artifact`` -- a **private/pure integrity checker**,
  fake/synthetic tests only.
- ``resolve_and_verify_acquisition_artifact`` -- the sole **public
  production resolver**, deriving every ``expected_*`` value from
  **verified Git objects**, never a caller-supplied trust root.

In addition to every §10 data-integrity requirement, this module
independently re-verifies the frozen §10.1 retry-audit invariants:
every logical member's ``attempts`` is in ``[1, 3]``, the manifest's own
``total_retry_count`` is in ``[0, 600]`` and exactly equals the sum of
every payload record's ``retry_count``, ``total_request_attempts`` equals
``300 + total_retry_count``, and the manifest's own frozen retry-policy
fields (``max_attempts_per_ticker``, ``max_retries``, ``backoff_seconds``,
``jitter``) exactly match this module's frozen constants.
"""

from __future__ import annotations

import errno
import hashlib
import os
import socket
import stat
from pathlib import Path
from typing import Any

from src.v7_yahoo_collector import V7YahooCollectorBlocked, canonical_ticker
from src.v8_partition import ticker_list_sha256 as v8_ticker_list_sha256
from src.v8c_git_provenance import CANONICAL_REPOSITORY_ROOT, V8CGitProvenanceBlocked, resolve_verified_v8c_production_git_commit
from src.v8c_historical_acquisition import (
    ACQUISITIONS_DIRNAME,
    BLOCK_ROLE,
    BLOCK_SEALED,
    BLOCK_STATUS,
    DATA_SOURCE,
    DATA_SOURCE_HOST,
    DATA_SOURCE_SCHEMA,
    MANIFEST_FILENAME,
    PAYLOAD_RECORD_FIELDS,
    RAW_DIRNAME,
    SEALED_FILENAME,
    V8CHistoricalAcquisitionBlocked,
    canonical_json_bytes,
    read_acquisition_manifest,
    read_sealed_record,
    read_t1c_trust_pin_from_verified_head,
    sha256_bytes,
)
from src.v8c_production_provenance import (
    CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
    EXPECTED_T2_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_trust_pin_independent_review,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_t2_bridge import V8CT2BridgeBlocked, read_and_verify_v8c_t2_authority_bridge
from src.v8c_trust_pin import V8CTrustPinBlocked, validate_trust_pin
from src.v8c_transport import BACKOFF_SECONDS, JITTER, MAXIMUM_ATTEMPTS_PER_TICKER, MAXIMUM_RETRIES, RETRYABLE_CLASSES, NONRETRYABLE_CLASSES

EXPECTED_AUTHORITY_CHAIN_BY_BLOCK = {
    "T1C": "V8C_SUCCESSOR_ALLOCATION_AUTHORITY",
    "T2": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_V8C_BRIDGE",
}

_EXPECTED_MALFORMED_OHLCV_POLICY_METADATA = {
    "policy_name": "POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE",
    "invalid_fraction_numerator": 1,
    "invalid_fraction_denominator": 252,
    "max_consecutive_invalid_returned_rows": 1,
    "full_p_hist_check_required": True,
    "test_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "expected_calendar_missing_dates_treated_as_malformed": False,
    "threshold_exceedance_action": "BLOCK_WHOLE_ACQUISITION",
}

_EXPECTED_AUTHORITY_BINDING_FIELDS = {
    "T1C": frozenset({
        "authorized_allocation_artifact_self_hash",
        "parent_v8_partition_manifest_sha256",
        "parent_v8_partition_implementation_commit",
        "trust_pin_human_gate",
    }),
    "T2": frozenset({
        "v8_partition_manifest_sha256",
        "v8_partition_implementation_commit",
        "v8_trust_anchor_git_identity",
        "v8c_t2_authority_bridge_human_gate",
    }),
}

_ZERO_ACCESS_COUNTER_FIELDS = (
    "validation_access_count",
    "feature_computation_count",
    "outcome_access_count",
    "sealed_holdout_access_count",
)


class V8CAcquisitionArtifactVerificationBlocked(RuntimeError):
    """Fail-closed §10 raw-acquisition-artifact integrity check error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _verify_member_transport_audit(ticker: str, audit: Any) -> tuple[int, str]:
    if not isinstance(audit, list) or not (1 <= len(audit) <= MAXIMUM_ATTEMPTS_PER_TICKER):
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_HISTORY_INVALID")
    expected_fingerprint = hashlib.sha256(canonical_json_bytes({
        "logical_request_identity": ticker,
        "request_start": "2016-04-01",
        "request_end_exclusive": "2026-01-01",
        "provider": DATA_SOURCE,
        "host": DATA_SOURCE_HOST,
        "request_parameters": {"interval": "1d", "events": "div,splits", "includeAdjustedClose": True},
    })).hexdigest()
    fingerprints = []
    for index, entry in enumerate(audit, start=1):
        if not isinstance(entry, dict) or set(entry) != {"attempt", "classification", "retryable", "classification_metadata", "request_fingerprint"}:
            raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_ENTRY_SCHEMA_INVALID")
        if entry["attempt"] != index or entry["request_fingerprint"] != expected_fingerprint:
            raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_REQUEST_FINGERPRINT_MISMATCH")
        fingerprints.append(entry["request_fingerprint"])
        classification = entry["classification"]
        metadata = entry["classification_metadata"]
        if not isinstance(metadata, dict) or "exception_type" not in metadata:
            raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_METADATA_INVALID")
        if index < len(audit):
            if classification not in RETRYABLE_CLASSES or entry["retryable"] is not True:
                raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_NONRETRYABLE_INTERMEDIATE_FAILURE")
            if metadata.get("exception_type") is None:
                raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_INTERMEDIATE_EXCEPTION_MISSING")
            if "http_code" in metadata:
                # Exact HTTPError concrete metadata -- never accept a valid
                # numeric code paired with a forged concrete exception type.
                if set(metadata) != {"exception_type", "http_code"}:
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_METADATA_INVALID")
                if metadata["exception_type"] != "HTTPError":
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH")
                code = metadata.get("http_code")
                derived = f"HTTP_{code}" if isinstance(code, int) and not isinstance(code, bool) else None
            elif "named_condition" in metadata:
                # Exact V8CTransportNamedFailure representation -- never
                # accept a named condition paired with a forged concrete
                # exception type, and never accept the outer classification
                # and the named condition disagreeing.
                if set(metadata) != {"exception_type", "named_condition"}:
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_METADATA_INVALID")
                if metadata["exception_type"] != "V8CTransportNamedFailure":
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH")
                derived = metadata.get("named_condition")
                if derived != classification:
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH")
            else:
                if set(metadata) != {"exception_type", "reason_type", "errno", "classification"}:
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_METADATA_INVALID")
                # The outer per-attempt classification and this metadata's
                # own embedded classification must agree -- a forged inner
                # field that disagrees with the outer classification (while
                # exception_type/reason_type/errno still derive the outer
                # value) must not silently pass.
                if metadata["classification"] != classification:
                    raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH")
                exception_type = metadata["exception_type"]
                reason_type = metadata["reason_type"]
                error_number = metadata["errno"]
                derived = None
                if classification == "NETWORK_TIMEOUT" and (
                    exception_type in {"TimeoutError", "socket.timeout"}
                    or reason_type in {"TimeoutError", "socket.timeout"}
                ):
                    derived = "NETWORK_TIMEOUT"
                elif classification == "CONNECTION_RESET" and (
                    # Mirrors ``src.v8c_transport._connection_reset_errno``
                    # exactly: an outer ``ConnectionResetError`` (regardless
                    # of errno), or a concrete ``OSError`` whose errno is
                    # exactly ECONNRESET -- never an errno match alone
                    # against an arbitrary/forged concrete type name.
                    exception_type == "ConnectionResetError"
                    or reason_type == "ConnectionResetError"
                    or (exception_type == "OSError" and error_number == errno.ECONNRESET)
                    or (reason_type == "OSError" and error_number == errno.ECONNRESET)
                ):
                    derived = "CONNECTION_RESET"
                elif classification == "TEMPORARY_DNS_FAILURE" and (
                    reason_type == "gaierror" and error_number == socket.EAI_AGAIN
                ):
                    derived = "TEMPORARY_DNS_FAILURE"
            if derived != classification:
                raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH")
        else:
            if classification != "SUCCESS" or entry["retryable"] is not None or metadata != {"exception_type": None}:
                raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_TERMINAL_SUCCESS_INVALID")
    if len(set(fingerprints)) != 1:
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_REQUEST_FINGERPRINT_DRIFT")
    return len(audit) - 1, expected_fingerprint


def _verify_acquisition_artifact(
    output_root,
    block: str,
    *,
    expected_v8c_frozen_design_commit: str,
    expected_reviewed_production_implementation_commit: str,
    expected_authority_chain: str,
    expected_ticker_list_sha256: str,
    expected_authority_binding: dict[str, Any],
) -> dict[str, Any]:
    try:
        manifest = read_acquisition_manifest(output_root, block)
    except V8CHistoricalAcquisitionBlocked as error:
        raise V8CAcquisitionArtifactVerificationBlocked("ACQUISITION_MANIFEST_INVALID:" + error.reason) from error

    if block not in _EXPECTED_AUTHORITY_BINDING_FIELDS:
        raise V8CAcquisitionArtifactVerificationBlocked("BLOCK_INVALID")

    if manifest["v8c_frozen_design_commit"] != expected_v8c_frozen_design_commit:
        raise V8CAcquisitionArtifactVerificationBlocked("FROZEN_DESIGN_COMMIT_MISMATCH")
    if manifest["implementation_git_commit"] != expected_reviewed_production_implementation_commit:
        raise V8CAcquisitionArtifactVerificationBlocked("IMPLEMENTATION_COMMIT_MISMATCH")
    if manifest["reviewed_production_implementation_commit"] != expected_reviewed_production_implementation_commit:
        raise V8CAcquisitionArtifactVerificationBlocked("REVIEWED_IMPLEMENTATION_COMMIT_MISMATCH")
    if manifest["authority_chain"] != expected_authority_chain:
        raise V8CAcquisitionArtifactVerificationBlocked("AUTHORITY_CHAIN_MISMATCH")

    if not isinstance(manifest.get("ticker_list_sha256"), str) or manifest["ticker_list_sha256"] != expected_ticker_list_sha256:
        raise V8CAcquisitionArtifactVerificationBlocked("TICKER_LIST_SHA_MISMATCH")

    authority_binding = manifest["authority_binding"]
    if not isinstance(authority_binding, dict) or set(authority_binding) != _EXPECTED_AUTHORITY_BINDING_FIELDS[block]:
        raise V8CAcquisitionArtifactVerificationBlocked("AUTHORITY_BINDING_SCHEMA_INVALID")
    if not isinstance(expected_authority_binding, dict) or set(expected_authority_binding) != _EXPECTED_AUTHORITY_BINDING_FIELDS[block]:
        raise V8CAcquisitionArtifactVerificationBlocked("EXPECTED_AUTHORITY_BINDING_SCHEMA_INVALID")
    if authority_binding != expected_authority_binding:
        raise V8CAcquisitionArtifactVerificationBlocked("AUTHORITY_BINDING_VALUE_MISMATCH")

    if manifest["role"] != BLOCK_ROLE[block]:
        raise V8CAcquisitionArtifactVerificationBlocked("ROLE_MISMATCH")
    if manifest["status"] != BLOCK_STATUS[block]:
        raise V8CAcquisitionArtifactVerificationBlocked("STATUS_MISMATCH")
    if manifest["sealed"] is not BLOCK_SEALED[block]:
        raise V8CAcquisitionArtifactVerificationBlocked("SEALED_MISMATCH")
    if manifest["research_access_authorized"] is not False:
        raise V8CAcquisitionArtifactVerificationBlocked("RESEARCH_ACCESS_INVARIANT_VIOLATED")
    for field in _ZERO_ACCESS_COUNTER_FIELDS:
        if type(manifest[field]) is not int or manifest[field] != 0:
            raise V8CAcquisitionArtifactVerificationBlocked("ACCESS_COUNTER_INVARIANT_VIOLATED")

    if manifest["data_source"] != DATA_SOURCE:
        raise V8CAcquisitionArtifactVerificationBlocked("DATA_SOURCE_MISMATCH")
    if manifest["data_source_host"] != DATA_SOURCE_HOST:
        raise V8CAcquisitionArtifactVerificationBlocked("DATA_SOURCE_HOST_INVALID")
    if manifest["data_source_schema"] != DATA_SOURCE_SCHEMA:
        raise V8CAcquisitionArtifactVerificationBlocked("DATA_SOURCE_SCHEMA_MISMATCH")

    if manifest["canonical_parser_classifier_blob_sha"] != CANONICAL_PARSER_CLASSIFIER_BLOB_SHA:
        raise V8CAcquisitionArtifactVerificationBlocked("CLASSIFIER_BLOB_MISMATCH")

    if manifest["malformed_ohlcv_policy"] != _EXPECTED_MALFORMED_OHLCV_POLICY_METADATA:
        raise V8CAcquisitionArtifactVerificationBlocked("MALFORMED_OHLCV_POLICY_METADATA_MISMATCH")

    if manifest["ticker_count"] != 300:
        raise V8CAcquisitionArtifactVerificationBlocked("TICKER_COUNT_INVALID")
    if manifest["request_start"] != "2016-04-01" or manifest["request_end_exclusive"] != "2026-01-01":
        raise V8CAcquisitionArtifactVerificationBlocked("REQUEST_WINDOW_INVALID")
    if manifest["request_count"] != 300 or manifest["success_transport_count"] != 300:
        raise V8CAcquisitionArtifactVerificationBlocked("REQUEST_COUNT_INVALID")

    # §10.1 frozen retry-policy binding.
    if manifest["max_attempts_per_ticker"] != MAXIMUM_ATTEMPTS_PER_TICKER or manifest["max_retries"] != MAXIMUM_RETRIES:
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_POLICY_MISMATCH")
    if manifest["backoff_seconds"] != list(BACKOFF_SECONDS) or manifest["jitter"] is not JITTER:
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_POLICY_MISMATCH")
    total_retry_count = manifest["total_retry_count"]
    if type(total_retry_count) is not int or not (0 <= total_retry_count <= 300 * MAXIMUM_RETRIES):
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_TOTAL_RETRY_COUNT_INVALID")
    if manifest["total_request_attempts"] != 300 + total_retry_count:
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_TOTAL_REQUEST_ATTEMPTS_INVALID")

    payload_manifest = manifest["payload_manifest"]
    if not isinstance(payload_manifest, list) or len(payload_manifest) != 300:
        raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_RECORD_COUNT_INVALID")

    payload_tickers: list[str] = []
    retry_count_sum = 0
    derived_all_intermediate_retryable = True
    for entry in payload_manifest:
        if not isinstance(entry, dict) or set(entry) != set(PAYLOAD_RECORD_FIELDS):
            raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_RECORD_SCHEMA_INVALID")
        ticker = entry["ticker"]
        if not isinstance(ticker, str):
            raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_TICKER_INVALID")
        try:
            canonical = canonical_ticker(ticker)
        except V7YahooCollectorBlocked as error:
            raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_TICKER_NOT_CANONICAL") from error
        if canonical != ticker:
            raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_TICKER_NOT_CANONICAL")
        payload_tickers.append(ticker)

        attempts = entry["attempts"]
        retry_count = entry["retry_count"]
        if type(attempts) is not int or not (1 <= attempts <= MAXIMUM_ATTEMPTS_PER_TICKER):
            raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_ATTEMPTS_INVALID")
        if type(retry_count) is not int or retry_count != attempts - 1:
            raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_RETRY_COUNT_INVALID")
        derived_retry_count, _ = _verify_member_transport_audit(ticker, entry["transport_audit"])
        if derived_retry_count != retry_count or attempts != len(entry["transport_audit"]):
            raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_DERIVED_COUNT_MISMATCH")
        retry_count_sum += retry_count
        derived_all_intermediate_retryable = derived_all_intermediate_retryable and all(
            item["retryable"] is True for item in entry["transport_audit"][:-1]
        )

    if retry_count_sum != total_retry_count:
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_TOTAL_RETRY_COUNT_MISMATCH")
    if manifest["retry_audit_all_intermediate_failures_retryable"] is not derived_all_intermediate_retryable:
        raise V8CAcquisitionArtifactVerificationBlocked("RETRY_AUDIT_SUMMARY_MISMATCH")

    if len(payload_tickers) != 300 or len(set(payload_tickers)) != 300:
        raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_DUPLICATE_TICKER")

    if sha256_bytes(canonical_json_bytes(payload_manifest)) != manifest["payload_manifest_sha256"]:
        raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_SHA_MISMATCH")

    if v8_ticker_list_sha256(payload_tickers) != manifest["ticker_list_sha256"]:
        raise V8CAcquisitionArtifactVerificationBlocked("PAYLOAD_TICKER_MEMBERSHIP_HASH_MISMATCH")

    raw_dir = Path(output_root) / ACQUISITIONS_DIRNAME / block / RAW_DIRNAME
    try:
        with os.scandir(raw_dir) as scan:
            raw_entries = list(scan)
    except OSError as error:
        raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_DIRECTORY_UNREADABLE") from error

    expected_files = {entry["ticker"] + ".json" for entry in payload_manifest}
    actual_entries = {entry.name for entry in raw_entries}
    for entry in raw_entries:
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY")
        except OSError as error:
            raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY") from error
    if expected_files - actual_entries:
        raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING")
    if actual_entries - expected_files:
        raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_UNEXPECTED_EXTRA")

    for entry in payload_manifest:
        path = raw_dir / (entry["ticker"] + ".json")
        try:
            mode = os.lstat(path).st_mode
            if not stat.S_ISREG(mode):
                raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY")
                with os.fdopen(descriptor, "rb") as stream:
                    descriptor = -1
                    raw = stream.read()
            finally:
                if descriptor != -1:
                    os.close(descriptor)
        except OSError as error:
            raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING") from error
        if len(raw) != entry["byte_count"]:
            raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH")
        if hashlib.sha256(raw).hexdigest() != entry["payload_sha256"]:
            raise V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_SHA256_MISMATCH")

    block_dir = Path(output_root) / ACQUISITIONS_DIRNAME / block
    try:
        top_level_entries = {entry.name for entry in block_dir.iterdir()}
    except OSError as error:
        raise V8CAcquisitionArtifactVerificationBlocked("BLOCK_BUNDLE_DIRECTORY_UNREADABLE") from error

    if block == "T1C" and SEALED_FILENAME in top_level_entries:
        raise V8CAcquisitionArtifactVerificationBlocked("T1C_BUNDLE_MUST_NOT_CONTAIN_SEALED_RECORD")

    expected_top_level = {MANIFEST_FILENAME, RAW_DIRNAME} | ({SEALED_FILENAME} if block == "T2" else set())
    if top_level_entries != expected_top_level:
        raise V8CAcquisitionArtifactVerificationBlocked("BLOCK_BUNDLE_TOP_LEVEL_ENTRIES_INVALID")

    if block == "T2":
        try:
            read_sealed_record(output_root, block)
        except V8CHistoricalAcquisitionBlocked as error:
            raise V8CAcquisitionArtifactVerificationBlocked("SEALED_RECORD_INVALID:" + error.reason) from error

    return {
        "result": "PASS",
        "block": block,
        "role": manifest["role"],
        "ticker_count": manifest["ticker_count"],
        "payload_manifest_record_count": len(payload_manifest),
        "payload_manifest_sha256": manifest["payload_manifest_sha256"],
        "canonical_price_rows_sha256": manifest["canonical_price_rows_sha256"],
        "sealed": manifest["sealed"],
        "research_access_authorized": manifest["research_access_authorized"],
        "sealed_holdout_access_count": manifest["sealed_holdout_access_count"],
        "validation_access_count": manifest["validation_access_count"],
        "total_retry_count": manifest["total_retry_count"],
        "total_request_attempts": manifest["total_request_attempts"],
    }


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CAcquisitionArtifactVerificationBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CAcquisitionArtifactVerificationBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CAcquisitionArtifactVerificationBlocked(reason)
    return V8CAcquisitionArtifactVerificationBlocked("PROVENANCE_CHECK_FAILED")


def _resolve_and_verify_acquisition_artifact_with_repository_root(output_root, block: str, *, repository_root) -> dict[str, Any]:
    if block not in EXPECTED_AUTHORITY_CHAIN_BY_BLOCK:
        raise V8CAcquisitionArtifactVerificationBlocked("BLOCK_INVALID")
    root = repository_root

    try:
        verified_head = resolve_verified_v8c_production_git_commit(root)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        verify_frozen_design_object(root)
        read_and_verify_design_freeze_approval(root, verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        review_binding = verify_reviewed_implementation_binding(root, verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    if block == "T2":
        try:
            anchor = read_and_verify_v8_trusted_partition_anchor(root, verified_head)
            bridge = read_and_verify_v8c_t2_authority_bridge(root, verified_head)
        except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked, V8CT2BridgeBlocked) as error:
            raise _wrap(error) from error
        expected_ticker_list_sha256 = EXPECTED_T2_TICKER_LIST_SHA256
        expected_authority_binding = {
            "v8_partition_manifest_sha256": anchor["authorized_partition_manifest_sha256"],
            "v8_partition_implementation_commit": anchor["authorized_partition_implementation_git_commit"],
            "v8_trust_anchor_git_identity": bridge["v8_trust_anchor_git_identity"],
            "v8c_t2_authority_bridge_human_gate": bridge["exact_human_bridge_authorization_identity"],
        }
    else:
        try:
            raw_pin = read_t1c_trust_pin_from_verified_head(root, verified_head)
        except V8CGitProvenanceBlocked as error:
            raise _wrap(error, "V8C_TRUSTED_ALLOCATION_MISSING") from error
        try:
            pin = validate_trust_pin(raw_pin)
        except V8CTrustPinBlocked as error:
            raise V8CAcquisitionArtifactVerificationBlocked("V8C_TRUST_PIN_INVALID:" + error.reason) from error
        if pin["authorization_status"] != "AUTHORIZED":
            raise V8CAcquisitionArtifactVerificationBlocked("V8C_TRUST_PIN_NOT_AUTHORIZED")
        try:
            read_and_verify_trust_pin_independent_review(
                root, verified_head,
                expected_allocation_artifact_self_hash=pin["authorized_allocation_artifact_self_hash"],
                expected_trust_pin_human_gate=pin["human_gate"],
            )
        except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
            raise _wrap(error, "V8C_TRUST_PIN_INDEPENDENT_REVIEW_MISSING") from error
        expected_ticker_list_sha256 = pin["t1c_ticker_list_sha256"]
        expected_authority_binding = {
            "authorized_allocation_artifact_self_hash": pin["authorized_allocation_artifact_self_hash"],
            "parent_v8_partition_manifest_sha256": pin["parent_v8_partition_manifest_sha256"],
            "parent_v8_partition_implementation_commit": pin["parent_v8_partition_implementation_commit"],
            "trust_pin_human_gate": pin["human_gate"],
        }

    return _verify_acquisition_artifact(
        output_root,
        block,
        expected_v8c_frozen_design_commit=EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        expected_reviewed_production_implementation_commit=reviewed_commit,
        expected_authority_chain=EXPECTED_AUTHORITY_CHAIN_BY_BLOCK[block],
        expected_ticker_list_sha256=expected_ticker_list_sha256,
        expected_authority_binding=expected_authority_binding,
    )


def resolve_and_verify_acquisition_artifact(output_root, block: str) -> dict[str, Any]:
    """The sole public production §10 boundary. Always resolves trust from
    ``CANONICAL_REPOSITORY_ROOT``."""
    return _resolve_and_verify_acquisition_artifact_with_repository_root(output_root, block, repository_root=CANONICAL_REPOSITORY_ROOT)


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "EXPECTED_AUTHORITY_CHAIN_BY_BLOCK",
    "V8CAcquisitionArtifactVerificationBlocked",
    "resolve_and_verify_acquisition_artifact",
]
