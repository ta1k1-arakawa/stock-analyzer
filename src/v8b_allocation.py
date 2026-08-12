"""V8B `T1B` validation-block allocation (successor authority chain, part B).

`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §4 and §11.3.B. This module
implements the private `T1B` allocation artifact: a deterministic,
zero-discretion slice of the trusted parent V8 `T_spare` set --

    T1B = parent_T_spare[0:300]
    remaining_T_spare = parent_T_spare[300:]

-- and nothing else. It never fetches, opens, or reasons about JPX/Yahoo
data, never touches `V8_TRUSTED_PARTITION.json` or the real V8 partition
manifest directly (those remain the caller's responsibility to read and
verify, exactly as `V8_HISTORICAL_RESEARCH_DESIGN.md` and this draft's §11.1
require -- this module receives the already-verified parent `T_spare`
ticker sequence as a plain argument, so its own logic is data-source
agnostic and fully exercisable with synthetic fixtures), and performs no
network I/O of any kind. Importing this module performs no I/O.

The artifact this module builds is the **private** layer of §11.3: it may
contain the exact `T1B` and remaining-`T_spare` ticker identities, and must
therefore never be printed, logged, committed to this repository, or
otherwise disclosed in full. `public_allocation_summary()` is the only
function in this module that is safe to log or persist publicly -- it
strips every ticker-identity field and keeps only hashes, counts, and
provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

STUDY_NAME = "V8B_HISTORICAL_RESEARCH"
PARENT_STUDY_NAME = "V8_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8B_T1B_ALLOCATION_V1"
ARTIFACT_ROLE = "VALIDATION_BLOCK_ALLOCATION"
LOGICAL_BLOCK = "T1B"

# §4: frozen, zero-implementation-time-discretion slice boundary.
T1B_OFFSET_WITHIN_PARENT_T_SPARE = 0
T1B_SLICE_START_INCLUSIVE = 0
T1B_SLICE_END_EXCLUSIVE = 300
T1B_TICKER_COUNT = 300

# §4's frozen `t1b_selection_rule_text`, byte-for-byte. A future independent
# verifier (§11.4) compares this exact string against the artifact's
# `selection_rule_canonical_text_or_hash`.
SELECTION_RULE_ID = "DETERMINISTIC_PREDECLARED_ZERO_DISCRETION"
SELECTION_RULE_TEXT = (
    "T1B = parent_T_spare[0:300]; remaining_T_spare = parent_T_spare[300:], "
    "where parent_T_spare is the canonical ordered T_spare sequence already "
    "contained in / derivable from the trusted parent V8 partition manifest "
    "under V8_HISTORICAL_RESEARCH_DESIGN.md §5.1's frozen deterministic "
    "ordering (sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending)"
)

ALLOCATION_ARTIFACT_FIELDS = (
    "schema_version",
    "study_name",
    "artifact_role",
    "logical_block",
    "parent_study",
    "parent_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "parent_t_spare_ticker_count",
    "parent_t_spare_ticker_list_sha256",
    "selection_rule_id",
    "selection_rule_canonical_text_or_hash",
    "t1b_offset_within_parent_t_spare",
    "t1b_slice_start_inclusive",
    "t1b_slice_end_exclusive",
    "t1b_ticker_count",
    "t1b_tickers",
    "t1b_ticker_list_sha256",
    "remaining_t_spare_ticker_count",
    "remaining_t_spare_tickers",
    "remaining_t_spare_ticker_list_sha256",
    "v8b_frozen_design_commit",
    "v8b_allocation_implementation_commit",
    "created_at_utc",
    "artifact_self_hash",
)

# The fields of ALLOCATION_ARTIFACT_FIELDS that carry ticker identities and
# must never appear in a publicly-logged/committed summary.
_PRIVATE_TICKER_IDENTITY_FIELDS = ("t1b_tickers", "remaining_t_spare_tickers")


class V8BAllocationBlocked(RuntimeError):
    """Fail-closed T1B allocation construction/read error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V8BAllocationBlocked("NONFINITE_VALUE") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _ticker_list_sha(tickers: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8"))


def ticker_list_sha256(tickers: Sequence[str]) -> str:
    """Public wrapper: the single authoritative T_spare/T1B ticker-list hash."""
    return _ticker_list_sha(tickers)


def _require_git_commit(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8BAllocationBlocked(reason)
    return value


def _require_sha256_hex(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8BAllocationBlocked(reason)
    return value


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8BAllocationBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_t1b_allocation_artifact(
    parent_t_spare_tickers: Sequence[str],
    *,
    parent_v8_partition_manifest_sha256: str,
    parent_v8_partition_implementation_commit: str,
    parent_t_spare_ticker_list_sha256: str,
    v8b_frozen_design_commit: str,
    v8b_allocation_implementation_commit: str,
    clock: Any,
) -> dict[str, Any]:
    """Build (never writes) the private §11.3.B `T1B` allocation artifact.

    ``parent_t_spare_tickers`` must already be the verified, canonically
    ordered `T_spare` sequence from the trusted parent V8 partition manifest
    -- this function performs zero source-of-truth resolution itself, only
    the frozen §4 zero-offset slice and artifact construction. Every
    duplicate/size/hash precondition is re-verified here rather than trusted
    from the caller, so a defective caller cannot silently produce a
    non-compliant artifact.
    """
    tickers = list(parent_t_spare_tickers)
    if len(set(tickers)) != len(tickers):
        raise V8BAllocationBlocked("PARENT_T_SPARE_DUPLICATE_TICKER")
    if len(tickers) < T1B_SLICE_END_EXCLUSIVE:
        raise V8BAllocationBlocked("PARENT_T_SPARE_INSUFFICIENT_SIZE")
    computed_parent_hash = _ticker_list_sha(tickers)
    if computed_parent_hash != parent_t_spare_ticker_list_sha256:
        raise V8BAllocationBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")

    parent_manifest_sha = _require_sha256_hex(
        parent_v8_partition_manifest_sha256, "PARENT_V8_PARTITION_MANIFEST_SHA_INVALID"
    )
    parent_impl_commit = _require_git_commit(
        parent_v8_partition_implementation_commit, "PARENT_V8_PARTITION_IMPLEMENTATION_COMMIT_INVALID"
    )
    design_commit = _require_git_commit(v8b_frozen_design_commit, "V8B_FROZEN_DESIGN_COMMIT_INVALID")
    allocation_commit = _require_git_commit(
        v8b_allocation_implementation_commit, "V8B_ALLOCATION_IMPLEMENTATION_COMMIT_INVALID"
    )

    t1b_tickers = tickers[T1B_SLICE_START_INCLUSIVE:T1B_SLICE_END_EXCLUSIVE]
    remaining_tickers = tickers[T1B_SLICE_END_EXCLUSIVE:]
    if len(t1b_tickers) != T1B_TICKER_COUNT:
        raise V8BAllocationBlocked("T1B_SLICE_SIZE_INVALID")
    if len(t1b_tickers) + len(remaining_tickers) != len(tickers):
        raise V8BAllocationBlocked("T1B_SLICE_ACCOUNTING_INVALID")
    if set(t1b_tickers) & set(remaining_tickers):
        raise V8BAllocationBlocked("T1B_REMAINING_OVERLAP")

    created = _utc_timestamp(clock() if callable(clock) else clock, "created_at_utc")

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "artifact_role": ARTIFACT_ROLE,
        "logical_block": LOGICAL_BLOCK,
        "parent_study": PARENT_STUDY_NAME,
        "parent_v8_partition_manifest_sha256": parent_manifest_sha,
        "parent_v8_partition_implementation_commit": parent_impl_commit,
        "parent_t_spare_ticker_count": len(tickers),
        "parent_t_spare_ticker_list_sha256": computed_parent_hash,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_canonical_text_or_hash": SELECTION_RULE_TEXT,
        "t1b_offset_within_parent_t_spare": T1B_OFFSET_WITHIN_PARENT_T_SPARE,
        "t1b_slice_start_inclusive": T1B_SLICE_START_INCLUSIVE,
        "t1b_slice_end_exclusive": T1B_SLICE_END_EXCLUSIVE,
        "t1b_ticker_count": len(t1b_tickers),
        "t1b_tickers": list(t1b_tickers),
        "t1b_ticker_list_sha256": _ticker_list_sha(t1b_tickers),
        "remaining_t_spare_ticker_count": len(remaining_tickers),
        "remaining_t_spare_tickers": list(remaining_tickers),
        "remaining_t_spare_ticker_list_sha256": _ticker_list_sha(remaining_tickers),
        "v8b_frozen_design_commit": design_commit,
        "v8b_allocation_implementation_commit": allocation_commit,
        "created_at_utc": _timestamp_text(created),
    }
    self_hash = canonical_sha256(artifact)
    artifact["artifact_self_hash"] = self_hash
    if set(artifact) != set(ALLOCATION_ARTIFACT_FIELDS):
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_SCHEMA_INVALID")
    return artifact


def public_allocation_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Strip every ticker-identity field; safe to log, print, or persist.

    Never call ``str()``/``repr()``/logging on the raw artifact itself --
    only on this summary's return value.
    """
    if not isinstance(artifact, Mapping) or set(artifact) != set(ALLOCATION_ARTIFACT_FIELDS):
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_SCHEMA_INVALID")
    return {
        key: value
        for key, value in artifact.items()
        if key not in _PRIVATE_TICKER_IDENTITY_FIELDS
    }


def verify_allocation_artifact_self_hash(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Re-verify ``artifact_self_hash`` against every other field's bytes."""
    if not isinstance(artifact, Mapping) or set(artifact) != set(ALLOCATION_ARTIFACT_FIELDS):
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_SCHEMA_INVALID")
    stated = artifact["artifact_self_hash"]
    recomputed = canonical_sha256({key: value for key, value in artifact.items() if key != "artifact_self_hash"})
    if stated != recomputed:
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH")
    t1b_tickers = artifact["t1b_tickers"]
    remaining_tickers = artifact["remaining_t_spare_tickers"]
    if not isinstance(t1b_tickers, list) or not isinstance(remaining_tickers, list):
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_TICKER_FIELDS_INVALID")
    if len(t1b_tickers) != artifact["t1b_ticker_count"] or artifact["t1b_ticker_count"] != T1B_TICKER_COUNT:
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_T1B_COUNT_MISMATCH")
    if len(remaining_tickers) != artifact["remaining_t_spare_ticker_count"]:
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_REMAINING_COUNT_MISMATCH")
    if _ticker_list_sha(t1b_tickers) != artifact["t1b_ticker_list_sha256"]:
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_T1B_HASH_MISMATCH")
    if _ticker_list_sha(remaining_tickers) != artifact["remaining_t_spare_ticker_list_sha256"]:
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_REMAINING_HASH_MISMATCH")
    if set(t1b_tickers) & set(remaining_tickers):
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_T1B_REMAINING_OVERLAP")
    return dict(artifact)


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_INVALID_JSON") from error
    if not isinstance(parsed, dict):
        raise V8BAllocationBlocked("ALLOCATION_ARTIFACT_INVALID_JSON")
    return parsed


def read_t1b_allocation_artifact_bytes(raw: bytes) -> dict[str, Any]:
    """Duplicate-key-safe parse plus full self-hash/schema re-verification.

    This is the sole read path a future verifier/production acquisition
    caller should use for a persisted private allocation artifact -- it
    never trusts JSON structure alone.
    """
    parsed = _strict_json_object(raw)
    return verify_allocation_artifact_self_hash(parsed)


__all__ = [
    "ALLOCATION_ARTIFACT_FIELDS",
    "ARTIFACT_ROLE",
    "LOGICAL_BLOCK",
    "PARENT_STUDY_NAME",
    "SCHEMA_VERSION",
    "SELECTION_RULE_ID",
    "SELECTION_RULE_TEXT",
    "STUDY_NAME",
    "T1B_OFFSET_WITHIN_PARENT_T_SPARE",
    "T1B_SLICE_END_EXCLUSIVE",
    "T1B_SLICE_START_INCLUSIVE",
    "T1B_TICKER_COUNT",
    "V8BAllocationBlocked",
    "build_t1b_allocation_artifact",
    "canonical_json_bytes",
    "canonical_sha256",
    "public_allocation_summary",
    "read_t1b_allocation_artifact_bytes",
    "sha256_bytes",
    "ticker_list_sha256",
    "verify_allocation_artifact_self_hash",
]
