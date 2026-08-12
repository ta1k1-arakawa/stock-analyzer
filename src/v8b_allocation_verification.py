"""`READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` -- §11.4 invariants.

Independently verifies every invariant `V8B_HISTORICAL_RESEARCH_DESIGN_
DRAFT.md` §11.4 requires against a *concrete* `T1B` allocation artifact
(`src/v8b_allocation.py`'s output) -- not merely against the implementation
code that produced it. Any single invariant failing is `BLOCK`: no trust
pin may be created and no acquisition may proceed (§11.4, §12's
`READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` gate).

This module performs no I/O and no network access. It never returns ticker
identities -- ``verify_t1b_allocation_artifact`` returns a safe aggregate
public result (hashes/counts/status only) on PASS, and raises
``V8BAllocationVerificationBlocked`` (carrying only a reason code, never a
ticker) on any failure.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.v8b_allocation import (
    ALLOCATION_ARTIFACT_FIELDS,
    SELECTION_RULE_TEXT,
    T1B_TICKER_COUNT,
    V8BAllocationBlocked,
    ticker_list_sha256,
    verify_allocation_artifact_self_hash,
)


class V8BAllocationVerificationBlocked(RuntimeError):
    """Fail-closed §11.4 invariant verification error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def verify_t1b_allocation_artifact(
    artifact: Mapping[str, Any],
    *,
    parent_t_spare_tickers: Sequence[str],
    t0_tickers: Sequence[str],
    old_t1_tickers: Sequence[str],
    t2_tickers: Sequence[str],
    t3_tickers: Sequence[str],
    expected_parent_t_spare_ticker_list_sha256: str,
    expected_v8b_frozen_design_commit: str,
) -> dict[str, Any]:
    """Verify every §11.4 invariant; return a safe public PASS summary.

    ``parent_t_spare_tickers``/``t0_tickers``/``old_t1_tickers``/
    ``t2_tickers``/``t3_tickers`` are supplied by the caller from whatever
    already-verified, trusted source it holds them (in production: the
    real, private trusted V8 partition manifest; in tests: synthetic
    fixtures). This function performs no partition/trust-anchor resolution
    of its own -- it only checks the invariants over what it is given.
    """
    try:
        verified = verify_allocation_artifact_self_hash(artifact)
    except V8BAllocationBlocked as error:
        raise V8BAllocationVerificationBlocked("ARTIFACT_SELF_HASH_INVALID:" + error.reason) from error

    parent = list(parent_t_spare_tickers)
    if len(set(parent)) != len(parent):
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_DUPLICATE_TICKER")

    t1b = verified["t1b_tickers"]
    remaining = verified["remaining_t_spare_tickers"]

    # T1B = original_parent_T_spare[0:300]; remaining_T_spare = original_parent_T_spare[300:]
    if t1b != parent[:300]:
        raise V8BAllocationVerificationBlocked("T1B_NOT_EXACT_ZERO_OFFSET_SLICE")
    if remaining != parent[300:]:
        raise V8BAllocationVerificationBlocked("REMAINING_T_SPARE_NOT_EXACT_TAIL_SLICE")

    # len(T1B) = 300
    if len(t1b) != T1B_TICKER_COUNT:
        raise V8BAllocationVerificationBlocked("T1B_SIZE_INVALID")

    # len(T1B) + len(remaining_T_spare) = len(original_parent_T_spare)
    if len(t1b) + len(remaining) != len(parent):
        raise V8BAllocationVerificationBlocked("T1B_REMAINING_ACCOUNTING_INVALID")

    t1b_set = set(t1b)
    remaining_set = set(remaining)
    parent_set = set(parent)

    # T1B ∩ remaining_T_spare = ∅
    if t1b_set & remaining_set:
        raise V8BAllocationVerificationBlocked("T1B_REMAINING_NOT_DISJOINT")

    # T1B ∪ remaining_T_spare = original_parent_T_spare
    if (t1b_set | remaining_set) != parent_set:
        raise V8BAllocationVerificationBlocked("T1B_REMAINING_UNION_MISMATCH")

    # T1B disjoint from T0 / old T1 / T2 / T3
    if t1b_set & set(t0_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_T0")
    if t1b_set & set(old_t1_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_OLD_T1")
    if t1b_set & set(t2_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_T2")
    if t1b_set & set(t3_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_T3")

    # parent_t_spare_ticker_list_sha256 matches the original trusted V8 partition manifest
    computed_parent_hash = ticker_list_sha256(parent)
    if computed_parent_hash != expected_parent_t_spare_ticker_list_sha256:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_HASH_MISMATCH_TRUSTED_ANCHOR")
    if verified["parent_t_spare_ticker_list_sha256"] != computed_parent_hash:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_HASH_MISMATCH_ARTIFACT")

    # t1b_ticker_list_sha256 / remaining_t_spare_ticker_list_sha256 match the artifact
    if ticker_list_sha256(t1b) != verified["t1b_ticker_list_sha256"]:
        raise V8BAllocationVerificationBlocked("T1B_TICKER_LIST_SHA_MISMATCH")
    if ticker_list_sha256(remaining) != verified["remaining_t_spare_ticker_list_sha256"]:
        raise V8BAllocationVerificationBlocked("REMAINING_T_SPARE_TICKER_LIST_SHA_MISMATCH")

    # selection_rule_canonical_text_or_hash exactly matches the frozen V8B design (§4)
    if verified["selection_rule_canonical_text_or_hash"] != SELECTION_RULE_TEXT:
        raise V8BAllocationVerificationBlocked("SELECTION_RULE_TEXT_MISMATCH")

    # v8b_frozen_design_commit matches the authorized/frozen V8B design
    if verified["v8b_frozen_design_commit"] != expected_v8b_frozen_design_commit:
        raise V8BAllocationVerificationBlocked("V8B_FROZEN_DESIGN_COMMIT_MISMATCH")

    if set(verified) != set(ALLOCATION_ARTIFACT_FIELDS):
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SCHEMA_INVALID")

    return {
        "result": "PASS",
        "logical_block": verified["logical_block"],
        "study_name": verified["study_name"],
        "parent_t_spare_ticker_count": verified["parent_t_spare_ticker_count"],
        "parent_t_spare_ticker_list_sha256": verified["parent_t_spare_ticker_list_sha256"],
        "t1b_ticker_count": verified["t1b_ticker_count"],
        "t1b_ticker_list_sha256": verified["t1b_ticker_list_sha256"],
        "remaining_t_spare_ticker_count": verified["remaining_t_spare_ticker_count"],
        "remaining_t_spare_ticker_list_sha256": verified["remaining_t_spare_ticker_list_sha256"],
        "artifact_self_hash": verified["artifact_self_hash"],
        "v8b_frozen_design_commit": verified["v8b_frozen_design_commit"],
        "v8b_allocation_implementation_commit": verified["v8b_allocation_implementation_commit"],
        "parent_v8_partition_manifest_sha256": verified["parent_v8_partition_manifest_sha256"],
        "parent_v8_partition_implementation_commit": verified["parent_v8_partition_implementation_commit"],
        "no_membership_choice_based_on_ohlcv_or_data_quality_outcomes": True,
    }


__all__ = [
    "V8BAllocationVerificationBlocked",
    "verify_t1b_allocation_artifact",
]
