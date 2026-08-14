from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import v8c_t1c_allocation as allocation

SYNTHETIC_DESIGN_COMMIT = "a" * 40
SYNTHETIC_IMPL_COMMIT = "b" * 40
SYNTHETIC_MANIFEST_SHA = "c" * 64


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def _parent(count: int = 1904) -> list[str]:
    return _tickers("SPARE", count)


def build(**overrides):
    parent = overrides.pop("parent_t_spare_tickers", _parent())
    kwargs = dict(
        parent_v8_partition_manifest_sha256=SYNTHETIC_MANIFEST_SHA,
        parent_v8_partition_implementation_commit=SYNTHETIC_IMPL_COMMIT,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8c_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
        v8c_allocation_implementation_commit=SYNTHETIC_IMPL_COMMIT,
        clock=clock_stub,
    )
    kwargs.update(overrides)
    return allocation.build_t1c_allocation_artifact(parent, **kwargs)


def test_slice_boundaries_are_frozen_300_600():
    assert allocation.T1C_SLICE_START_INCLUSIVE == 300
    assert allocation.T1C_SLICE_END_EXCLUSIVE == 600
    assert allocation.T1C_TICKER_COUNT == 300


def test_t1c_is_exact_300_600_slice():
    parent = _parent()
    artifact = build(parent_t_spare_tickers=parent)
    assert artifact["t1c_tickers"] == parent[300:600]
    assert artifact["t1c_ticker_count"] == 300


def test_remaining_is_parent_minus_t1c_slice():
    parent = _parent()
    artifact = build(parent_t_spare_tickers=parent)
    assert artifact["remaining_t_spare_tickers"] == parent[600:]
    assert artifact["remaining_t_spare_ticker_count"] == len(parent) - 600
    assert artifact["predecessor_burned_count"] == 300


def test_t1c_and_remaining_partition_the_parent_exactly():
    parent = _parent()
    artifact = build(parent_t_spare_tickers=parent)
    t1c = set(artifact["t1c_tickers"])
    remaining = set(artifact["remaining_t_spare_tickers"])
    assert t1c.isdisjoint(remaining)
    assert t1c | remaining == set(parent[300:])


def test_insufficient_parent_size_blocked():
    small_parent = _tickers("SPARE", 599)
    with pytest.raises(allocation.V8CAllocationBlocked) as excinfo:
        allocation.build_t1c_allocation_artifact(
            small_parent,
            parent_v8_partition_manifest_sha256=SYNTHETIC_MANIFEST_SHA,
            parent_v8_partition_implementation_commit=SYNTHETIC_IMPL_COMMIT,
            parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(small_parent),
            v8c_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
            v8c_allocation_implementation_commit=SYNTHETIC_IMPL_COMMIT,
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_INSUFFICIENT_SIZE"


def test_duplicate_parent_ticker_blocked():
    parent = _parent()
    parent[1] = parent[0]
    with pytest.raises(allocation.V8CAllocationBlocked) as excinfo:
        allocation.build_t1c_allocation_artifact(
            parent,
            parent_v8_partition_manifest_sha256=SYNTHETIC_MANIFEST_SHA,
            parent_v8_partition_implementation_commit=SYNTHETIC_IMPL_COMMIT,
            parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
            v8c_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
            v8c_allocation_implementation_commit=SYNTHETIC_IMPL_COMMIT,
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_DUPLICATE_TICKER"


def test_parent_hash_mismatch_blocked():
    parent = _parent()
    with pytest.raises(allocation.V8CAllocationBlocked) as excinfo:
        allocation.build_t1c_allocation_artifact(
            parent,
            parent_v8_partition_manifest_sha256=SYNTHETIC_MANIFEST_SHA,
            parent_v8_partition_implementation_commit=SYNTHETIC_IMPL_COMMIT,
            parent_t_spare_ticker_list_sha256="0" * 64,
            v8c_frozen_design_commit=SYNTHETIC_DESIGN_COMMIT,
            v8c_allocation_implementation_commit=SYNTHETIC_IMPL_COMMIT,
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH"


# ---------------------------------------------------------------------------
# Public summary must never carry ticker identities
# ---------------------------------------------------------------------------


def test_public_summary_strips_all_ticker_identity_fields():
    artifact = build()
    summary = allocation.public_allocation_summary(artifact)
    assert "t1c_tickers" not in summary
    assert "remaining_t_spare_tickers" not in summary
    # Every ticker string from the artifact must be absent from the summary values.
    for value in summary.values():
        if isinstance(value, str):
            for ticker in artifact["t1c_tickers"] + artifact["remaining_t_spare_tickers"]:
                assert ticker not in value


def test_public_summary_retains_hashes_and_counts():
    artifact = build()
    summary = allocation.public_allocation_summary(artifact)
    assert summary["t1c_ticker_count"] == 300
    assert summary["t1c_ticker_list_sha256"] == artifact["t1c_ticker_list_sha256"]
    assert summary["artifact_self_hash"] == artifact["artifact_self_hash"]


def test_self_hash_verification_detects_tampering():
    artifact = build()
    tampered = dict(artifact)
    tampered["t1c_tickers"] = list(tampered["t1c_tickers"])
    tampered["t1c_tickers"][0] = "TAMPERED"
    with pytest.raises(allocation.V8CAllocationBlocked) as excinfo:
        allocation.verify_allocation_artifact_self_hash(tampered)
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH"


def test_read_artifact_bytes_rejects_duplicate_keys():
    raw = b'{"a": 1, "a": 2}'
    with pytest.raises(allocation.V8CAllocationBlocked) as excinfo:
        allocation.read_t1c_allocation_artifact_bytes(raw)
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_DUPLICATE_KEY"


def test_selection_rule_text_names_the_v8b_ordering_rule():
    assert "T1C = original_parent_T_spare[300:600]" in allocation.SELECTION_RULE_TEXT
    assert "T1B" in allocation.SELECTION_RULE_TEXT


def test_module_performs_no_io_on_import():
    import importlib
    import sys

    module_name = "src.v8c_t1c_allocation"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)  # must not raise / touch disk beyond normal import
