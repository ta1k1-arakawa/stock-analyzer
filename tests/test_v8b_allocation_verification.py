from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import v8b_allocation as allocation
from src import v8b_allocation_verification as verification

SYNTHETIC_COMMIT_A = "a" * 40
SYNTHETIC_DESIGN_COMMIT = "d" * 40
SYNTHETIC_ALLOC_COMMIT = "e" * 40


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def build_artifact(parent, *, design_commit=SYNTHETIC_DESIGN_COMMIT):
    return allocation.build_t1b_allocation_artifact(
        parent_t_spare_tickers=parent,
        parent_v8_partition_manifest_sha256="0" * 64,
        parent_v8_partition_implementation_commit=SYNTHETIC_COMMIT_A,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8b_frozen_design_commit=design_commit,
        v8b_allocation_implementation_commit=SYNTHETIC_ALLOC_COMMIT,
        clock=clock_stub,
    )


def verify_kwargs(parent, artifact, **overrides):
    kwargs = dict(
        artifact=artifact,
        parent_t_spare_tickers=parent,
        t0_tickers=_tickers("T0", 300),
        old_t1_tickers=_tickers("OLDT1", 300),
        t2_tickers=_tickers("T2", 300),
        t3_tickers=_tickers("T3", 300),
        expected_parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        expected_v8b_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
    )
    kwargs.update(overrides)
    return kwargs


def test_pass_returns_safe_aggregate_summary_only():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    result = verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, artifact))
    assert result["result"] == "PASS"
    assert result["t1b_ticker_count"] == 300
    assert "t1b_tickers" not in result
    assert "remaining_t_spare_tickers" not in result
    assert result["no_membership_choice_based_on_ohlcv_or_data_quality_outcomes"] is True


def test_disjoint_from_t0():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    overlapping_t0 = list(parent[:300])
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, artifact, t0_tickers=overlapping_t0))
    assert excinfo.value.reason == "T1B_NOT_DISJOINT_FROM_T0"


def test_disjoint_from_old_t1():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    overlapping = list(parent[:1])
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, artifact, old_t1_tickers=overlapping))
    assert excinfo.value.reason == "T1B_NOT_DISJOINT_FROM_OLD_T1"


def test_disjoint_from_t2():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    overlapping = list(parent[299:300])
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, artifact, t2_tickers=overlapping))
    assert excinfo.value.reason == "T1B_NOT_DISJOINT_FROM_T2"


def test_disjoint_from_t3():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    overlapping = list(parent[:1])
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, artifact, t3_tickers=overlapping))
    assert excinfo.value.reason == "T1B_NOT_DISJOINT_FROM_T3"


def test_parent_hash_mismatch_against_trusted_anchor_blocks():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(
            **verify_kwargs(parent, artifact, expected_parent_t_spare_ticker_list_sha256="f" * 64)
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_HASH_MISMATCH_TRUSTED_ANCHOR"


def test_design_commit_mismatch_blocks():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(
            **verify_kwargs(parent, artifact, expected_v8b_frozen_design_commit="9" * 40)
        )
    assert excinfo.value.reason == "V8B_FROZEN_DESIGN_COMMIT_MISMATCH"


def test_selection_rule_text_tamper_blocks():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    tampered = dict(artifact)
    tampered["selection_rule_canonical_text_or_hash"] = "some other rule"
    tampered["artifact_self_hash"] = allocation.canonical_sha256(
        {k: v for k, v in tampered.items() if k != "artifact_self_hash"}
    )
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, tampered))
    assert excinfo.value.reason == "SELECTION_RULE_TEXT_MISMATCH"


def test_non_zero_offset_slice_blocks():
    """A tampered artifact whose T1B is a different, still internally
    consistent (disjoint, union-complete) 300-slice -- e.g. offset 300
    instead of offset 0 -- must still BLOCK at the §11.4 zero-offset
    invariant, even though it passes the lower-level self-hash/overlap
    checks on its own internal fields."""
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    tampered = dict(artifact)
    shifted_t1b = list(parent[300:600])
    shifted_remaining = list(parent[0:300]) + list(parent[600:])
    tampered["t1b_tickers"] = shifted_t1b
    tampered["t1b_ticker_list_sha256"] = allocation.ticker_list_sha256(shifted_t1b)
    tampered["remaining_t_spare_tickers"] = shifted_remaining
    tampered["remaining_t_spare_ticker_list_sha256"] = allocation.ticker_list_sha256(shifted_remaining)
    tampered["artifact_self_hash"] = allocation.canonical_sha256(
        {k: v for k, v in tampered.items() if k != "artifact_self_hash"}
    )
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, tampered))
    assert excinfo.value.reason == "T1B_NOT_EXACT_ZERO_OFFSET_SLICE"


def test_corrupted_self_hash_blocks_before_any_other_check():
    parent = _tickers("P", 1904)
    artifact = build_artifact(parent)
    tampered = dict(artifact)
    tampered["artifact_self_hash"] = "0" * 64
    with pytest.raises(verification.V8BAllocationVerificationBlocked) as excinfo:
        verification.verify_t1b_allocation_artifact(**verify_kwargs(parent, tampered))
    assert excinfo.value.reason.startswith("ARTIFACT_SELF_HASH_INVALID:")
