"""`READ_ONLY_T1B_ACQUISITION_ARTIFACT_VERIFICATION` / `_T2_` (§12.6).

Data-integrity checks only, over an already-published `T1B`/`T2`
acquisition bundle produced by `src/v8b_historical_acquisition.py`. This
module never computes features, strategy results, profit, trades, or any
other research outcome, and never parses raw payload bytes into OHLCV --
it reads them solely to recompute `byte_count`/SHA-256 and compare against
the acquisition manifest's own `payload_manifest`. Any mismatch is
`BLOCK`, with no research opening. Performs no network access.

Two, and only two, ways to call this module (round-3 finding HIGH-4,
corrected further in round 3's repeat review):

- `_verify_acquisition_artifact` -- a **private/pure integrity checker**
  (round-3 repeat finding MEDIUM-2: not part of the production public
  surface -- fake/synthetic tests import and call it directly as an
  internal helper). It compares the published manifest's own fields
  against caller-supplied ``expected_*`` values and performs no Git access
  of its own. It exists so fake/synthetic tests can exercise every
  mismatch branch directly; it is not, by itself, a safe *production*
  trust root, because nothing stops a caller from fabricating favorable
  ``expected_*`` values.
- `resolve_and_verify_acquisition_artifact` -- the sole **public
  production resolver**. It derives every ``expected_*`` value -- the
  reviewed implementation commit, the block's exact expected
  ``ticker_list_sha256``, and the *exact values* (not merely the key set)
  of `authority_binding` -- from **verified Git objects** read from the
  one fixed, non-overridable production repository root (round-3 repeat
  finding HIGH-1: no ``repository_root`` parameter exists on this public
  function; a private DI-testable variant carries that parameter for
  fake/synthetic tests only). It never accepts a caller-supplied expected
  hash or authority string as the trust root.

Round-3 repeat finding HIGH-2: block membership is bound to the *concrete*
payload -- every `payload_manifest` record's schema and ticker canonical
form are validated, exactly 300 unique tickers are required, and
`ticker_list_sha256` is **recomputed** from those 300 concrete ticker
values (preserving `payload_manifest`'s own order) rather than merely
compared against the manifest's self-reported field. A forged bundle whose
manifest carries the correct trusted hash while its `payload_manifest`/raw
files actually name a different 300-ticker set BLOCKs.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from src.v7_yahoo_collector import V7YahooCollectorBlocked, canonical_ticker
from src.v8_partition import ticker_list_sha256 as v8_ticker_list_sha256
from src.v8b_git_provenance import V8BGitProvenanceBlocked, resolve_verified_v8b_production_git_commit
from src.v8b_historical_acquisition import (
    ACQUISITIONS_DIRNAME,
    BLOCK_ROLE,
    BLOCK_SEALED,
    BLOCK_STATUS,
    CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
    DATA_SOURCE,
    DATA_SOURCE_HOST,
    DATA_SOURCE_SCHEMA,
    MANIFEST_FILENAME,
    PAYLOAD_RECORD_FIELDS,
    RAW_DIRNAME,
    RETRY_COUNT,
    SEALED_FILENAME,
    V8BHistoricalAcquisitionBlocked,
    canonical_json_bytes,
    read_acquisition_manifest,
    read_sealed_record,
    read_t1b_trust_pin_from_verified_head,
    sha256_bytes,
)
from src.v8b_production_provenance import (
    EXPECTED_T2_TICKER_LIST_SHA256,
    EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
    V8BProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_t2_authority_bridge,
    read_and_verify_trust_pin_independent_review,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8b_trust_pin import V8BTrustPinBlocked, validate_trust_pin

# The one fixed, non-overridable production repository root -- round-3
# repeat finding HIGH-1: no public function in this module accepts a
# caller-supplied repository_root as its trust root.
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_AUTHORITY_CHAIN_BY_BLOCK = {
    "T1B": "V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
    "T2": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE",
}

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


def _verify_acquisition_artifact(
    output_root,
    block: str,
    *,
    expected_v8b_frozen_design_commit: str,
    expected_reviewed_production_implementation_commit: str,
    expected_authority_chain: str,
    expected_ticker_list_sha256: str,
    expected_authority_binding: dict[str, Any],
) -> dict[str, Any]:
    """Private/pure integrity checker -- fake/synthetic tests only, not a
    production trust root (see module docstring). Verify §12.6's full
    checklist for one published `T1B`/`T2` bundle against caller-supplied
    ``expected_*`` values.

    ``expected_ticker_list_sha256`` and ``expected_authority_binding``
    prove **block membership authority**, not merely internal
    self-consistency (round-3 finding HIGH-4): the manifest's own
    ``ticker_list_sha256`` must equal the exact expected hash, and
    ``authority_binding`` must equal ``expected_authority_binding`` value
    for value, not merely share its key set.

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

    # HIGH-4: block membership authority -- the manifest's own ticker-list
    # hash must equal the exact expected value, never merely be internally
    # present/well-formed.
    if not isinstance(manifest.get("ticker_list_sha256"), str) or manifest["ticker_list_sha256"] != expected_ticker_list_sha256:
        raise V8BAcquisitionArtifactVerificationBlocked("TICKER_LIST_SHA_MISMATCH")

    # authority_binding semantics appropriate to T1B/T2 (§12.6): exact key
    # set AND exact values, never merely "same field names present"
    # (round-3 finding HIGH-4).
    authority_binding = manifest["authority_binding"]
    if not isinstance(authority_binding, dict) or set(authority_binding) != _EXPECTED_AUTHORITY_BINDING_FIELDS[block]:
        raise V8BAcquisitionArtifactVerificationBlocked("AUTHORITY_BINDING_SCHEMA_INVALID")
    if not isinstance(expected_authority_binding, dict) or set(expected_authority_binding) != _EXPECTED_AUTHORITY_BINDING_FIELDS[block]:
        raise V8BAcquisitionArtifactVerificationBlocked("EXPECTED_AUTHORITY_BINDING_SCHEMA_INVALID")
    if authority_binding != expected_authority_binding:
        raise V8BAcquisitionArtifactVerificationBlocked("AUTHORITY_BINDING_VALUE_MISMATCH")

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

    # HIGH-2 (round-3 repeat finding): every payload_manifest record must
    # have exactly the expected schema and a canonical ticker identifier --
    # a forged record with extra/missing fields, or a non-canonical ticker
    # spelling, must BLOCK before its ticker is ever trusted as part of the
    # membership set. Order is preserved exactly as payload_manifest states
    # it (never re-sorted) -- ticker_list_sha256 is order-sensitive.
    payload_tickers: list[str] = []
    for entry in payload_manifest:
        if not isinstance(entry, dict) or set(entry) != set(PAYLOAD_RECORD_FIELDS):
            raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_RECORD_SCHEMA_INVALID")
        ticker = entry["ticker"]
        if not isinstance(ticker, str):
            raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_TICKER_INVALID")
        try:
            canonical = canonical_ticker(ticker)
        except V7YahooCollectorBlocked as error:
            raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_TICKER_NOT_CANONICAL") from error
        if canonical != ticker:
            raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_TICKER_NOT_CANONICAL")
        payload_tickers.append(ticker)

    if len(payload_tickers) != 300 or len(set(payload_tickers)) != 300:
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_DUPLICATE_TICKER")

    # Recompute payload_manifest_sha256 from the actual payload_manifest
    # list -- never merely trust the manifest's own stated hash field.
    if sha256_bytes(canonical_json_bytes(payload_manifest)) != manifest["payload_manifest_sha256"]:
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_MANIFEST_SHA_MISMATCH")

    # HIGH-2: block membership authority is bound to the CONCRETE payload,
    # not merely the manifest's own claimed ticker_list_sha256 field --
    # recompute the membership hash from the actual 300 payload_manifest
    # ticker values using the exact production ticker-list hashing rule
    # (`src.v8_partition.ticker_list_sha256`, the same rule the manifest's
    # own field was originally computed with) and require it to equal the
    # manifest's claimed hash, which is itself already pinned above to the
    # Git-derived expected hash. A forged bundle whose manifest carries the
    # correct trusted ticker_list_sha256 while its payload_manifest/raw
    # files actually name a different 300-ticker set BLOCKs here.
    if v8_ticker_list_sha256(payload_tickers) != manifest["ticker_list_sha256"]:
        raise V8BAcquisitionArtifactVerificationBlocked("PAYLOAD_TICKER_MEMBERSHIP_HASH_MISMATCH")

    raw_dir = Path(output_root) / ACQUISITIONS_DIRNAME / block / RAW_DIRNAME
    try:
        with os.scandir(raw_dir) as scan:
            raw_entries = list(scan)
    except OSError as error:
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_DIRECTORY_UNREADABLE") from error

    expected_files = {entry["ticker"] + ".json" for entry in payload_manifest}
    actual_entries = {entry.name for entry in raw_entries}
    for entry in raw_entries:
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY")
        except OSError as error:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY") from error
    if len(raw_entries) != 300 or actual_entries != expected_files:
        if expected_files - actual_entries:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING")
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_UNEXPECTED_EXTRA")

    if expected_files - actual_entries:
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING")
    if actual_entries - expected_files:
        raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_UNEXPECTED_EXTRA")

    for entry in payload_manifest:
        path = raw_dir / (entry["ticker"] + ".json")
        try:
            mode = os.lstat(path).st_mode
            if not stat.S_ISREG(mode):
                raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_NON_REGULAR_ENTRY")
                with os.fdopen(descriptor, "rb") as stream:
                    descriptor = -1
                    raw = stream.read()
            finally:
                if descriptor != -1:
                    os.close(descriptor)
        except OSError as error:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_MISSING") from error
        if len(raw) != entry["byte_count"]:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH")
        if hashlib.sha256(raw).hexdigest() != entry["payload_sha256"]:
            raise V8BAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_SHA256_MISMATCH")

    # Repeat-round finding MEDIUM-3: independently verify the actual
    # on-disk bundle shape and, for T2, the actual SEALED.json state --
    # not merely the acquisition manifest's own self-reported
    # sealed/research_access_authorized fields (which a forged manifest
    # could claim honestly while the real SEALED.json is missing,
    # modified, or absent entirely). Exactly the expected top-level
    # entries are required -- no unexpected extra files/directories --
    # and T1B must never carry the T2-only SEALED.json contract.
    block_dir = Path(output_root) / ACQUISITIONS_DIRNAME / block
    try:
        top_level_entries = {entry.name for entry in block_dir.iterdir()}
    except OSError as error:
        raise V8BAcquisitionArtifactVerificationBlocked("BLOCK_BUNDLE_DIRECTORY_UNREADABLE") from error

    if block == "T1B" and SEALED_FILENAME in top_level_entries:
        raise V8BAcquisitionArtifactVerificationBlocked("T1B_BUNDLE_MUST_NOT_CONTAIN_SEALED_RECORD")

    expected_top_level = {MANIFEST_FILENAME, RAW_DIRNAME} | ({SEALED_FILENAME} if block == "T2" else set())
    if top_level_entries != expected_top_level:
        raise V8BAcquisitionArtifactVerificationBlocked("BLOCK_BUNDLE_TOP_LEVEL_ENTRIES_INVALID")

    if block == "T2":
        try:
            read_sealed_record(output_root, block)
        except V8BHistoricalAcquisitionBlocked as error:
            raise V8BAcquisitionArtifactVerificationBlocked("SEALED_RECORD_INVALID:" + error.reason) from error

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


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8BAcquisitionArtifactVerificationBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BAcquisitionArtifactVerificationBlocked(missing_reason)
    if isinstance(reason, str):
        return V8BAcquisitionArtifactVerificationBlocked(reason)
    return V8BAcquisitionArtifactVerificationBlocked("PROVENANCE_CHECK_FAILED")


def _resolve_and_verify_acquisition_artifact_with_repository_root(
    output_root,
    block: str,
    *,
    repository_root,
) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only, not
    a production API (round-3 repeat finding HIGH-1). ``repository_root``
    is caller-injectable here so fake tests can exercise this ordering
    against a bogus/synthetic repository; the public
    ``resolve_and_verify_acquisition_artifact`` below is the only
    production entrypoint, and it always passes
    ``CANONICAL_REPOSITORY_ROOT`` -- never a caller-suppliable value.

    Derives every trust value -- the reviewed implementation commit, the
    block's exact expected ``ticker_list_sha256``, and the exact
    ``authority_binding`` values (`T2`: the immutable V8 anchor + `OPTION_2`
    bridge; `T1B`: the Git-sourced `V8B_TRUSTED_ALLOCATION.json` trust pin)
    -- from **verified Git objects**, never from a caller-supplied expected
    hash or authority string. Delegates the actual integrity checklist to
    ``_verify_acquisition_artifact``, the same pure checker fake/synthetic
    tests use directly with their own synthetic ``expected_*`` values.
    Performs no network access and parses no OHLCV.
    """
    if block not in EXPECTED_AUTHORITY_CHAIN_BY_BLOCK:
        raise V8BAcquisitionArtifactVerificationBlocked("BLOCK_INVALID")
    root = repository_root

    try:
        verified_head = resolve_verified_v8b_production_git_commit(root)
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        verify_frozen_design_object(root)
        read_and_verify_design_freeze_approval(root, verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        review_binding = verify_reviewed_implementation_binding(root, verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    if block == "T2":
        try:
            anchor = read_and_verify_v8_trusted_partition_anchor(root, verified_head)
            bridge = read_and_verify_t2_authority_bridge(root, verified_head)
        except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
            raise _wrap(error) from error
        expected_ticker_list_sha256 = EXPECTED_T2_TICKER_LIST_SHA256
        expected_authority_binding = {
            "v8_partition_manifest_sha256": anchor["authorized_partition_manifest_sha256"],
            "v8_partition_implementation_commit": anchor["authorized_partition_implementation_git_commit"],
            "v8_trust_anchor_git_identity": bridge["v8_trust_anchor_git_identity"],
            "option_2_bridge_human_gate": bridge["human_gate"],
        }
    else:
        try:
            raw_pin = read_t1b_trust_pin_from_verified_head(root, verified_head)
        except V8BGitProvenanceBlocked as error:
            raise _wrap(error, "V8B_TRUSTED_ALLOCATION_MISSING") from error
        try:
            pin = validate_trust_pin(raw_pin)
        except V8BTrustPinBlocked as error:
            raise V8BAcquisitionArtifactVerificationBlocked("V8B_TRUST_PIN_INVALID:" + error.reason) from error
        if pin["authorization_status"] != "AUTHORIZED":
            raise V8BAcquisitionArtifactVerificationBlocked("V8B_TRUST_PIN_NOT_AUTHORIZED")
        # HIGH-1 (repeat round): READ_ONLY_T1B_ACQUISITION_ARTIFACT_
        # VERIFICATION must re-establish the *complete* T1B authority
        # chain, including INDEPENDENT_TRUST_PIN_REVIEW -- not merely the
        # trust pin's own authorization_status. Fails closed today (the
        # real review artifact does not exist).
        try:
            read_and_verify_trust_pin_independent_review(
                root,
                verified_head,
                expected_allocation_artifact_self_hash=pin["authorized_allocation_artifact_self_hash"],
                expected_trust_pin_human_gate=pin["human_gate"],
            )
        except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
            raise _wrap(error, "V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING") from error
        expected_ticker_list_sha256 = pin["t1b_ticker_list_sha256"]
        expected_authority_binding = {
            "authorized_allocation_artifact_self_hash": pin["authorized_allocation_artifact_self_hash"],
            "parent_v8_partition_manifest_sha256": pin["parent_v8_partition_manifest_sha256"],
            "parent_v8_partition_implementation_commit": pin["parent_v8_partition_implementation_commit"],
            "trust_pin_human_gate": pin["human_gate"],
        }

    return _verify_acquisition_artifact(
        output_root,
        block,
        expected_v8b_frozen_design_commit=EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
        expected_reviewed_production_implementation_commit=reviewed_commit,
        expected_authority_chain=EXPECTED_AUTHORITY_CHAIN_BY_BLOCK[block],
        expected_ticker_list_sha256=expected_ticker_list_sha256,
        expected_authority_binding=expected_authority_binding,
    )


def resolve_and_verify_acquisition_artifact(output_root, block: str) -> dict[str, Any]:
    """The sole public production §12.6 boundary (round-3 repeat finding
    HIGH-1). Always resolves trust from ``CANONICAL_REPOSITORY_ROOT`` --
    this signature deliberately exposes no ``repository_root`` (or any
    other trust-root) override. See module docstring for the full
    ordering.
    """
    return _resolve_and_verify_acquisition_artifact_with_repository_root(
        output_root, block, repository_root=CANONICAL_REPOSITORY_ROOT
    )


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "EXPECTED_AUTHORITY_CHAIN_BY_BLOCK",
    "V8BAcquisitionArtifactVerificationBlocked",
    "resolve_and_verify_acquisition_artifact",
]
