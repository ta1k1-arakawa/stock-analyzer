"""`READ_ONLY_T1B_ACQUISITION_ARTIFACT_VERIFICATION` / `_T2_` (§12.6).

Data-integrity checks only, over an already-published `T1B`/`T2`
acquisition bundle produced by `src/v8b_historical_acquisition.py`. This
module never computes features, strategy results, profit, trades, or any
other research outcome, and never parses raw payload bytes into OHLCV --
it reads them solely to recompute `byte_count`/SHA-256 and compare against
the acquisition manifest's own `payload_manifest`. Any mismatch is
`BLOCK`, with no research opening. Performs no network access.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.v8b_historical_acquisition import (
    ACQUISITIONS_DIRNAME,
    BLOCK_ROLE,
    BLOCK_SEALED,
    BLOCK_STATUS,
    CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
    DATA_SOURCE,
    DATA_SOURCE_HOST,
    DATA_SOURCE_SCHEMA,
    RAW_DIRNAME,
    RETRY_COUNT,
    V8BHistoricalAcquisitionBlocked,
    canonical_json_bytes,
    read_acquisition_manifest,
    sha256_bytes,
)

# The exact §7.6 F1_C1 policy metadata every honest manifest must carry.
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

# The exact authority_binding key set expected per block (§12.6:
# "authority_binding semantics appropriate to T1B/T2").
_EXPECTED_AUTHORITY_BINDING_FIELDS = {
    "T1B": frozenset({
        "authorized_allocation_artifact_self_hash",
        "parent_v8_partition_manifest_sha256",
        "parent_v8_partition_implementation_commit",
        "trust_pin_human_gate",
    }),
    "T2": frozenset({
        "v8_partition_manifest_sha256",
        "v8_partition_implementation_commit",
        "v8_trust_anchor_git_identity",
        "option_2_bridge_human_gate",
    }),
}

_ZERO_ACCESS_COUNTER_FIELDS = (
    "validation_access_count",
    "feature_computation_count",
    "outcome_access_count",
    "sealed_holdout_access_count",
)


class V8BAcquisitionArtifactVerificationBlocked(RuntimeError):
    """Fail-closed §12.6 raw-acquisition-artifact integrity check error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def verify_acquisition_artifact(
    output_root,
    block: str,
    *,
    expected_v8b_frozen_design_commit: str,
    expected_reviewed_production_implementation_commit: str,
    expected_authority_chain: str,
) -> dict[str, Any]:
    """Verify §12.6's full checklist for one published `T1B`/`T2` bundle.

    Returns a safe aggregate PASS result (counts/hashes/status only, never
    a ticker or raw OHLCV value) or raises
    ``V8BAcquisitionArtifactVerificationBlocked`` on the first failing
    check.
    """
    try:
        manifest = read_acquisition_manifest(output_root, block)
    except V8BHistoricalAcquisitionBlocked as error:
        raise V8BAcquisitionArtifactVerificationBlocked(
            "ACQUISITION_MANIFEST_INVALID:" + error.reason
        ) from error

    if block not in _EXPECTED_AUTHORITY_BINDING_FIELDS:
        raise V8BAcquisitionArtifactVerificationBlocked("BLOCK_INVALID")

    if manifest["v8b_frozen_design_commit"] != expected_v8b_frozen_design_commit:
        raise V8BAcquisitionArtifactVerificationBlocked("FROZEN_DESIGN_COMMIT_MISMATCH")
    if manifest["implementation_git_commit"] != expected_reviewed_production_implementation_commit:
        raise V8BAcquisitionArtifactVerificationBlocked("IMPLEMENTATION_COMMIT_MISMATCH")
    if manifest["reviewed_production_implementation_commit"] != expected_reviewed_production_implementation_commit:
        raise V8BAcquisitionArtifactVerificationBlocked("REVIEWED_IMPLEMENTATION_COMMIT_MISMATCH")
    if manifest["authority_chain"] != expected_authority_chain:
        raise V8BAcquisitionArtifactVerificationBlocked("AUTHORITY_CHAIN_MISMATCH")

    # authority_binding semantics appropriate to T1B/T2 (§12.6).
    authority_binding = manifest["authority_binding"]
    if not isinstance(authority_binding, dict) or set(authority_binding) != _EXPECTED_AUTHORITY_BINDING_FIELDS[block]:
        raise V8BAcquisitionArtifactVerificationBlocked("AUTHORITY_BINDING_SCHEMA_INVALID")

    # Exact block role/status/sealed (defense-in-depth: read_acquisition_manifest
    # already re-validates these against this module's own constants).
    if manifest["role"] != BLOCK_ROLE[block]:
        raise V8BAcquisitionArtifactVerificationBlocked("ROLE_MISMATCH")
    if manifest["status"] != BLOCK_STATUS[block]:
        raise V8BAcquisitionArtifactVerificationBlocked("STATUS_MISMATCH")
    if manifest["sealed"] is not BLOCK_SEALED[block]:
        raise V8BAcquisitionArtifactVerificationBlocked("SEALED_MISMATCH")
    if manifest["research_access_authorized"] is not False:
        raise V8BAcquisitionArtifactVerificationBlocked("RESEARCH_ACCESS_INVARIANT_VIOLATED")
    for field in _ZERO_ACCESS_COUNTER_FIELDS:
        if type(manifest[field]) is not int or manifest[field] != 0:
            raise V8BAcquisitionArtifactVerificationBlocked("ACCESS_COUNTER_INVARIANT_VIOLATED")

    # Exact data-source binding.
    if manifest["data_source"] != DATA_SOURCE:
        raise V8BAcquisitionArtifactVerificationBlocked("DATA_SOURCE_MISMATCH")
    if manifest["data_source_host"] != DATA_SOURCE_HOST:
        raise V8BAcquisitionArtifactVerificationBlocked("DATA_SOURCE_HOST_INVALID")
    if manifest["data_source_schema"] != DATA_SOURCE_SCHEMA:
        raise V8BAcquisitionArtifactVerificationBlocked("DATA_SOURCE_SCHEMA_MISMATCH")

    # Exact classifier blob binding (§7.6).
    if manifest["canonical_parser_classifier_blob_sha"] != CANONICAL_PARSER_CLASSIFIER_BLOB_SHA:
        raise V8BAcquisitionArtifactVerificationBlocked("CLASSIFIER_BLOB_MISMATCH")

    # Exact F1_C1 policy metadata (defense-in-depth alongside read_acquisition_manifest).
    if manifest["malformed_ohlcv_policy"] != _EXPECTED_MALFORMED_OHLCV_POLICY_METADATA:
        raise V8BAcquisitionArtifactVerificationBlocked("MALFORMED_OHLCV_POLICY_METADATA_MISMATCH")

    if manifest["ticker_count"] != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("TICKER_COUNT_INVALID")
    if manifest["request_start"] != "2016-04-01" or manifest["request_end_exclusive"] != "2026-01-01":
        raise V8BAcquisitionArtifactVerificationBlocked("REQUEST_WINDOW_INVALID")
    if manifest["retry_count"] != RETRY_COUNT:
        raise V8BAcquisitionArtifactVerificationBlocked("RETRY_COUNT_INVALID")
    if manifest["request_count"] != 300 or manifest["success_transport_count"] != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("REQUEST_COUNT_INVALID")

    payload_manifest = manifest["payload_manifest"]
    if not isinstance(payload_manifest, list) or len(payload_manifest) != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_RECORD_COUNT_INVALID")

    # Recompute payload_manifest_sha256 from the actual payload_manifest
    # list -- never merely trust the manifest's own stated hash field.
    if sha256_bytes(canonical_json_bytes(payload_manifest)) != manifest["payload_manifest_sha256"]:
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_SHA_MISMATCH")

    raw_dir = Path(output_root) / ACQUISITIONS_DIRNAME / block / RAW_DIRNAME
    try:
        actual_files = {entry.name for entry in raw_dir.iterdir() if entry.is_file()}
    except OSError as error:
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_DIRECTORY_UNREADABLE") from error

    expected_files = {entry["ticker"] + ".json" for entry in payload_manifest}
    if len(expected_files) != len(payload_manifest):
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_DUPLICATE_TICKER")
    if expected_files - actual_files:
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING")
    if actual_files - expected_files:
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_UNEXPECTED_EXTRA")

    for entry in payload_manifest:
        path = raw_dir / (entry["ticker"] + ".json")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING") from error
        if len(raw) != entry["byte_count"]:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH")
        if hashlib.sha256(raw).hexdigest() != entry["payload_sha256"]:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_SHA256_MISMATCH")

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
    }


__all__ = [
    "V8BAcquisitionArtifactVerificationBlocked",
    "verify_acquisition_artifact",
]
