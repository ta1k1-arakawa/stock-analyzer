from __future__ import annotations

import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8_partition as partition
from src import v8c_human_gate_consumption as gate_consumption
from src import v8c_t1c_allocator as allocator

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real network call executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def write_partition_manifest(path: Path, *, t_spare: list[str]) -> dict:
    blocks = {
        "T0": _tickers("T0BLK", 300), "T1": _tickers("T1BLK", 300),
        "T2": _tickers("T2BLK", 300), "T3": _tickers("T3BLK", 300),
        "T_spare": list(t_spare),
    }
    manifest = {
        "schema_version": partition.SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "source_snapshot_semantics": partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": "6" * 40,
        "created_utc": "2026-08-09T00:00:00Z",
        "source_url": "https://www.jpx.co.jp/synthetic/data_j.xls",
        "source_host": "www.jpx.co.jp",
        "source_acquisition_utc": "2026-08-09T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 0,
        "v4_source_raw_sha256_reference": "1" * 64,
        "v4_raw_sha_equality_required": partition.V4_RAW_SHA_EQUALITY_REQUIRED,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[k] for k in blocks), [])),
        "selection_rule": "synthetic fixture selection rule",
        "deterministic_ordering_rule": partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": partition.ticker_list_sha256(blocks["T_spare"]),
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": partition.ticker_list_sha256([]),
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "block_assignments": blocks,
        "p_hist_start": partition.P_HIST_START,
        "p_hist_end": partition.P_HIST_END,
        "t1_role": partition.T1_ROLE,
        "t2_role": partition.T2_ROLE,
        "t3_role": partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    manifest["manifest_sha256"] = partition.canonical_sha256(manifest)
    assert set(manifest) == set(partition.MANIFEST_FIELDS)
    path.write_bytes(partition.canonical_json_bytes(manifest))
    return manifest


def _valid_anchor_for(manifest: dict) -> dict:
    return {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": manifest["manifest_sha256"],
        "authorized_partition_implementation_git_commit": manifest["partition_implementation_git_commit"],
    }


def run(**overrides):
    overrides.setdefault("consumption_state_root", Path(tempfile.gettempdir()) / ("v8c_gate_state-" + uuid.uuid4().hex))
    return allocator._allocate_t1c_production_with_dependencies(**overrides)


def _base_deps(**overrides):
    deps = dict(
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        clock=clock_stub,
    )
    deps.update(overrides)
    return deps


def test_confirmation_token_is_frozen_literal():
    assert allocator.ALLOCATION_CONFIRMATION == "V8C_PRODUCTION_ALLOCATE_T1C"


def test_wrong_confirmation_blocks_before_any_dependency_call(tmp_path):
    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(allocator.V8CT1CAllocatorBlocked) as excinfo:
        run(
            confirmation="wrong",
            partition_manifest_path=tmp_path / "missing.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
            anchor_reader=forbidden, clock=clock_stub,
        )
    assert excinfo.value.reason == "V8C_ALLOCATION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


def test_gate_already_consumed_blocks_before_provenance(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(state_root, gate_consumption.GATE_ALLOCATE_T1C, allocator.EXPECTED_V8C_FROZEN_DESIGN_COMMIT, clock=clock_stub)

    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called after gate already consumed")

    with pytest.raises(allocator.V8CT1CAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "unread.json",
            output_path=tmp_path / "out.json",
            consumption_state_root=state_root,
            git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
            anchor_reader=forbidden, clock=clock_stub,
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1C
    assert not (tmp_path / "unread.json").exists()


def test_anchor_not_authorized_blocks_before_private_manifest_read(tmp_path):
    with pytest.raises(allocator.V8CT1CAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "unread.json",
            output_path=tmp_path / "out.json",
            **_base_deps(anchor_reader=lambda head: {"authorization_status": "NOT_AUTHORIZED"}),
        )
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"
    assert excinfo.value.authorization_consumed is False
    assert not (tmp_path / "unread.json").exists()


def test_gate_durably_consumed_exactly_once_across_two_calls(tmp_path):
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("REAL", allocator.EXPECTED_PARENT_T_SPARE_TICKER_COUNT)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    state_root = tmp_path / "state"

    import src.v8c_t1c_allocator as allocator_module
    orig_count = allocator_module.EXPECTED_PARENT_T_SPARE_TICKER_COUNT
    orig_hash = allocator_module.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256
    allocator_module.EXPECTED_PARENT_T_SPARE_TICKER_COUNT = len(t_spare)
    allocator_module.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 = partition.ticker_list_sha256(t_spare)
    try:
        summary = run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=tmp_path / "out.json",
            consumption_state_root=state_root,
            **_base_deps(anchor_reader=lambda head: _valid_anchor_for(manifest)),
        )
        assert summary["t1c_ticker_count"] == 300
        assert "t1c_tickers" not in summary
        assert summary["t1c_ticker_list_sha256"] == partition.ticker_list_sha256(t_spare[300:600])

        # A second call under the same design commit must BLOCK before any
        # provenance/private-access step -- the gate stays consumed.
        def forbidden(*_a, **_kw):
            raise AssertionError("must not be called: gate already consumed")

        with pytest.raises(allocator.V8CT1CAllocatorBlocked) as excinfo:
            run(
                confirmation=allocator.ALLOCATION_CONFIRMATION,
                partition_manifest_path=manifest_path,
                output_path=tmp_path / "out2.json",
                consumption_state_root=state_root,
                git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
                frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
                anchor_reader=forbidden, clock=clock_stub,
            )
        assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1C
    finally:
        allocator_module.EXPECTED_PARENT_T_SPARE_TICKER_COUNT = orig_count
        allocator_module.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 = orig_hash


def test_parent_t_spare_count_mismatch_blocks_with_authorization_consumed_true(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path, t_spare=_tickers("TS", 1903))
    with pytest.raises(allocator.V8CT1CAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=tmp_path / "out.json",
            **_base_deps(anchor_reader=lambda head: _valid_anchor_for(manifest)),
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_TICKER_COUNT_INVALID"
    assert excinfo.value.authorization_consumed is True


def test_public_entrypoint_rejects_wrong_confirmation():
    with pytest.raises(allocator.V8CT1CAllocatorBlocked) as excinfo:
        allocator.allocate_t1c_production(confirmation="wrong token", partition_manifest_path="/tmp/x", output_path="/tmp/y")
    assert excinfo.value.reason == "V8C_ALLOCATION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False
