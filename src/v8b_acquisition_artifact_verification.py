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
    RAW_DIRNAME,
    RETRY_COUNT,
    V8BHistoricalAcquisitionBlocked,
    read_acquisition_manifest,
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

    if manifest["v8b_frozen_design_commit"] != expected_v8b_frozen_design_commit:
        raise V8BAcquisitionArtifactVerificationBlocked("FROZEN_DESIGN_COMMIT_MISMATCH")
    if manifest["reviewed_production_implementation_commit"] != expected_reviewed_production_implementation_commit:
        raise V8BAcquisitionArtifactVerificationBlocked("REVIEWED_IMPLEMENTATION_COMMIT_MISMATCH")
    if manifest["authority_chain"] != expected_authority_chain:
        raise V8BAcquisitionArtifactVerificationBlocked("AUTHORITY_CHAIN_MISMATCH")
    if manifest["ticker_count"] != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("TICKER_COUNT_INVALID")
    if manifest["request_start"] != "2016-04-01" or manifest["request_end_exclusive"] != "2026-01-01":
        raise V8BAcquisitionArtifactVerificationBlocked("REQUEST_WINDOW_INVALID")
    if manifest["data_source_host"] != "query1.finance.yahoo.com":
        raise V8BAcquisitionArtifactVerificationBlocked("DATA_SOURCE_HOST_INVALID")
    if manifest["retry_count"] != RETRY_COUNT:
        raise V8BAcquisitionArtifactVerificationBlocked("RETRY_COUNT_INVALID")
    if manifest["request_count"] != 300 or manifest["success_transport_count"] != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("REQUEST_COUNT_INVALID")

    payload_manifest = manifest["payload_manifest"]
    if not isinstance(payload_manifest, list) or len(payload_manifest) != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_RECORD_COUNT_INVALID")

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
