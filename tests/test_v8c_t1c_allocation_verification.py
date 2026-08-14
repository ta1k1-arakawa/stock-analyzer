from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import v8c_t1c_allocation as allocation
from src import v8c_t1c_allocation_verification as verification

SYNTHETIC_DESIGN_COMMIT = "a" * 40
SYNTHETIC_IMPL_COMMIT = "b" * 40
SYNTHETIC_MANIFEST_SHA = "c" * 64


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def _parent(count: int = 1904) -> list[str]:
    return _tickers("SPARE", count)


def _artifact(parent):
    return allocation.build_t1c_allocation_artifact(
        parent,
        parent_v8_partition_manifest_sha256=SYNTHETIC_MANIFEST_SHA,
        parent_v8_partition_implementation_commit=SYNTHETIC_IMPL_COMMIT,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8c_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
        v8c_allocation_implementation_commit=SYNTHETIC_IMPL_COMMIT,
        clock=clock_stub,
    )


def _verify(artifact, parent, **overrides):
    kwargs = dict(
        parent_t_spare_tickers=parent,
        t0_tickers=_tickers("T0", 300),
        old_t1_tickers=_tickers("T1", 300),
        t1b_tickers=_tickers("T1B", 300),
        t2_tickers=_tickers("T2", 300),
        t3_tickers=_tickers("T3", 300),
        expected_parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        expected_v8c_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
    )
    kwargs.update(overrides)
    return verification._verify_t1c_allocation_artifact(artifact, **kwargs)


def test_valid_artifact_passes():
    parent = _parent()
    artifact = _artifact(parent)
    result = _verify(artifact, parent)
    assert result["result"] == "PASS"
    assert result["t1c_ticker_count"] == 300


def test_disjoint_from_t1b_required():
    parent = _parent()
    artifact = _artifact(parent)
    # Make T1B overlap with T1C's actual slice.
    overlapping_t1b = list(parent[300:600])
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, t1b_tickers=overlapping_t1b)
    assert excinfo.value.reason == "T1C_NOT_DISJOINT_FROM_T1B"


def test_disjoint_from_t0_required():
    parent = _parent()
    artifact = _artifact(parent)
    overlapping_t0 = list(parent[300:600])
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, t0_tickers=overlapping_t0)
    assert excinfo.value.reason == "T1C_NOT_DISJOINT_FROM_T0"


def test_disjoint_from_t2_required():
    parent = _parent()
    artifact = _artifact(parent)
    overlapping_t2 = list(parent[300:600])
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, t2_tickers=overlapping_t2)
    assert excinfo.value.reason == "T1C_NOT_DISJOINT_FROM_T2"


def test_disjoint_from_t3_required():
    parent = _parent()
    artifact = _artifact(parent)
    overlapping_t3 = list(parent[300:600])
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, t3_tickers=overlapping_t3)
    assert excinfo.value.reason == "T1C_NOT_DISJOINT_FROM_T3"


def test_disjoint_from_old_t1_required():
    parent = _parent()
    artifact = _artifact(parent)
    overlapping_old_t1 = list(parent[300:600])
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, old_t1_tickers=overlapping_old_t1)
    assert excinfo.value.reason == "T1C_NOT_DISJOINT_FROM_OLD_T1"


def test_wrong_slice_blocked():
    parent = _parent()
    artifact = _artifact(parent)
    tampered = dict(artifact)
    # Recompute a self-consistent-but-wrong slice (shifted by one), keeping
    # t1c/remaining disjoint and their union equal to the parent so the
    # self-hash/consistency check passes and only the exact-slice check
    # inside the verification module catches the shift.
    wrong_t1c = parent[301:601]
    wrong_remaining = parent[:301] + parent[601:]
    tampered["t1c_tickers"] = wrong_t1c
    tampered["t1c_ticker_list_sha256"] = allocation.ticker_list_sha256(wrong_t1c)
    tampered["remaining_t_spare_tickers"] = wrong_remaining
    tampered["remaining_t_spare_ticker_list_sha256"] = allocation.ticker_list_sha256(wrong_remaining)
    tampered["artifact_self_hash"] = allocation.canonical_sha256(
        {k: v for k, v in tampered.items() if k != "artifact_self_hash"}
    )
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(tampered, parent)
    assert excinfo.value.reason == "T1C_NOT_EXACT_300_600_SLICE"


def test_design_commit_mismatch_blocked():
    parent = _parent()
    artifact = _artifact(parent)
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, expected_v8c_frozen_design_commit="f" * 40)
    assert excinfo.value.reason == "V8C_FROZEN_DESIGN_COMMIT_MISMATCH"


def test_parent_hash_mismatch_against_trusted_anchor_blocked():
    parent = _parent()
    artifact = _artifact(parent)
    with pytest.raises(verification.V8CAllocationVerificationBlocked) as excinfo:
        _verify(artifact, parent, expected_parent_t_spare_ticker_list_sha256="0" * 64)
    assert excinfo.value.reason == "PARENT_T_SPARE_HASH_MISMATCH_TRUSTED_ANCHOR"
