from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import v8b_allocation as allocation

SYNTHETIC_COMMIT_A = "a" * 40
SYNTHETIC_COMMIT_B = "b" * 40
SYNTHETIC_COMMIT_C = "c" * 40


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def build_kwargs(parent_tickers, **overrides):
    kwargs = dict(
        parent_t_spare_tickers=parent_tickers,
        parent_v8_partition_manifest_sha256="0" * 64,
        parent_v8_partition_implementation_commit=SYNTHETIC_COMMIT_A,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent_tickers),
        v8b_frozen_design_commit=SYNTHETIC_COMMIT_B,
        v8b_allocation_implementation_commit=SYNTHETIC_COMMIT_C,
        clock=clock_stub,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# T1B first-300/remainder deterministic slice
# ---------------------------------------------------------------------------


def test_t1b_is_exact_first_300_and_remainder_is_exact_tail():
    parent = _tickers("T", 1904)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    assert artifact["t1b_tickers"] == parent[:300]
    assert artifact["remaining_t_spare_tickers"] == parent[300:]
    assert artifact["t1b_ticker_count"] == 300
    assert artifact["remaining_t_spare_ticker_count"] == 1904 - 300
    assert artifact["t1b_offset_within_parent_t_spare"] == 0
    assert artifact["t1b_slice_start_inclusive"] == 0
    assert artifact["t1b_slice_end_exclusive"] == 300


def test_allocation_accounting_invariant_holds_for_minimal_pool():
    parent = _tickers("M", 300)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    assert artifact["t1b_ticker_count"] + artifact["remaining_t_spare_ticker_count"] == len(parent)
    assert artifact["remaining_t_spare_tickers"] == []


def test_insufficient_parent_pool_blocks():
    parent = _tickers("S", 299)
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    assert excinfo.value.reason == "PARENT_T_SPARE_INSUFFICIENT_SIZE"


def test_duplicate_parent_ticker_blocks():
    parent = _tickers("D", 300) + ["D0000"]
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    assert excinfo.value.reason == "PARENT_T_SPARE_DUPLICATE_TICKER"


def test_parent_hash_mismatch_blocks():
    parent = _tickers("H", 400)
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.build_t1b_allocation_artifact(
            **build_kwargs(parent, parent_t_spare_ticker_list_sha256="f" * 64)
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("parent_v8_partition_manifest_sha256", "not-a-hash", "PARENT_V8_PARTITION_MANIFEST_SHA_INVALID"),
        ("parent_v8_partition_implementation_commit", "short", "PARENT_V8_PARTITION_IMPLEMENTATION_COMMIT_INVALID"),
        ("v8b_frozen_design_commit", "short", "V8B_FROZEN_DESIGN_COMMIT_INVALID"),
        ("v8b_allocation_implementation_commit", "short", "V8B_ALLOCATION_IMPLEMENTATION_COMMIT_INVALID"),
    ],
)
def test_invalid_provenance_fields_block(field, value, reason):
    parent = _tickers("P", 400)
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.build_t1b_allocation_artifact(**build_kwargs(parent, **{field: value}))
    assert excinfo.value.reason == reason


# ---------------------------------------------------------------------------
# Artifact self-hash / duplicate-key / forgery rejection
# ---------------------------------------------------------------------------


def test_artifact_self_hash_validates_on_honest_artifact():
    parent = _tickers("V", 500)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    verified = allocation.verify_allocation_artifact_self_hash(artifact)
    assert verified["artifact_self_hash"] == artifact["artifact_self_hash"]


def test_tampered_field_fails_self_hash_verification():
    parent = _tickers("T", 500)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    tampered = dict(artifact)
    tampered["t1b_tickers"] = list(tampered["t1b_tickers"])
    tampered["t1b_tickers"][0] = "FORGED"
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.verify_allocation_artifact_self_hash(tampered)
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH"


def test_forged_hash_without_matching_content_fails():
    parent = _tickers("F", 500)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    forged = dict(artifact)
    forged["artifact_self_hash"] = "0" * 64
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.verify_allocation_artifact_self_hash(forged)
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH"


def test_read_bytes_rejects_duplicate_top_level_key():
    raw = b'{"schema_version": "a", "schema_version": "b"}'
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.read_t1b_allocation_artifact_bytes(raw)
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_DUPLICATE_KEY"


def test_read_bytes_round_trips_honest_artifact():
    parent = _tickers("R", 500)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    raw = allocation.canonical_json_bytes(artifact)
    reloaded = allocation.read_t1b_allocation_artifact_bytes(raw)
    assert reloaded == artifact


def test_missing_schema_field_blocks():
    parent = _tickers("X", 400)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    incomplete = dict(artifact)
    del incomplete["created_at_utc"]
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.verify_allocation_artifact_self_hash(incomplete)
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# No public identity leakage
# ---------------------------------------------------------------------------


def test_public_summary_never_contains_ticker_fields():
    parent = _tickers("N", 500)
    artifact = allocation.build_t1b_allocation_artifact(**build_kwargs(parent))
    summary = allocation.public_allocation_summary(artifact)
    assert "t1b_tickers" not in summary
    assert "remaining_t_spare_tickers" not in summary
    for value in summary.values():
        if isinstance(value, list):
            assert value == []
    assert summary["t1b_ticker_count"] == 300
    assert summary["t1b_ticker_list_sha256"] == artifact["t1b_ticker_list_sha256"]


def test_public_summary_rejects_malformed_artifact():
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.public_allocation_summary({"not": "an artifact"})
    assert excinfo.value.reason == "ALLOCATION_ARTIFACT_SCHEMA_INVALID"


def test_selection_rule_text_is_frozen_and_matches_design_draft_section_4():
    assert allocation.SELECTION_RULE_TEXT.startswith("T1B = parent_T_spare[0:300]")
    assert "remaining_T_spare = parent_T_spare[300:]" in allocation.SELECTION_RULE_TEXT
    assert "§5.1" in allocation.SELECTION_RULE_TEXT


def test_clock_accepts_plain_datetime_value_not_only_callable():
    parent = _tickers("C", 400)
    artifact = allocation.build_t1b_allocation_artifact(
        **build_kwargs(parent, clock=datetime(2026, 1, 1, tzinfo=timezone.utc))
    )
    assert artifact["created_at_utc"] == "2026-01-01T00:00:00Z"


def test_naive_clock_blocks():
    parent = _tickers("Z", 400)
    with pytest.raises(allocation.V8BAllocationBlocked) as excinfo:
        allocation.build_t1b_allocation_artifact(**build_kwargs(parent, clock=lambda: datetime(2026, 1, 1)))
    assert excinfo.value.reason == "UTC_TIMESTAMP_INVALID:created_at_utc"
