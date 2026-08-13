from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8b_human_gate_consumption as gate_consumption

SYNTHETIC_DESIGN_COMMIT = "a" * 40


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def test_known_gates_are_the_exact_three_named_in_the_finding():
    assert set(gate_consumption.KNOWN_GATES) == {
        "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B",
        "T1B_RAW_ACQUISITION_HUMAN_GATE",
        "T2_RAW_ACQUISITION_HUMAN_GATE",
    }
    assert gate_consumption.GATE_ALLOCATE_T1B == "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B"
    assert gate_consumption.GATE_T1B_RAW_ACQUISITION == "T1B_RAW_ACQUISITION_HUMAN_GATE"
    assert gate_consumption.GATE_T2_RAW_ACQUISITION == "T2_RAW_ACQUISITION_HUMAN_GATE"


def test_unknown_gate_rejected():
    with pytest.raises(gate_consumption.V8BHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            "/tmp/whatever", "SOME_OTHER_GATE", SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_UNKNOWN"


def test_malformed_design_commit_rejected():
    with pytest.raises(gate_consumption.V8BHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            "/tmp/whatever", gate_consumption.GATE_ALLOCATE_T1B, "not-a-commit", clock=clock_stub
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_DESIGN_COMMIT_INVALID"


def test_not_yet_consumed_never_raises(tmp_path):
    gate_consumption.require_gate_not_yet_consumed(
        tmp_path / "state", gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT
    )
    assert gate_consumption.has_gate_been_consumed(
        tmp_path / "state", gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT
    ) is False


def test_consume_once_then_require_not_yet_consumed_blocks(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT
    ) is True
    with pytest.raises(gate_consumption.V8BHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.require_gate_not_yet_consumed(
            state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1B


def test_consume_twice_raises_never_silently_succeeds(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    with pytest.raises(gate_consumption.V8BHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1B


def test_different_gates_under_the_same_state_root_are_independent(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1B_RAW_ACQUISITION, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T2_RAW_ACQUISITION, SYNTHETIC_DESIGN_COMMIT
    ) is False
    # T2's gate remains consumable even though T1B's was just consumed.
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T2_RAW_ACQUISITION, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )


def test_different_design_commits_under_the_same_gate_are_independent(tmp_path):
    state_root = tmp_path / "state"
    other_commit = "b" * 40
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, other_commit
    ) is False


def test_receipt_is_durable_bytes_on_disk_with_no_ticker_or_path_content(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    receipts = list(Path(state_root).glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_bytes())
    assert set(receipt) == {"schema_version", "study_name", "gate", "v8b_frozen_design_commit", "consumed_at_utc"}
    assert receipt["schema_version"] == gate_consumption.SCHEMA_VERSION
    assert receipt["study_name"] == gate_consumption.STUDY_NAME
    assert receipt["gate"] == gate_consumption.GATE_ALLOCATE_T1B
    assert receipt["v8b_frozen_design_commit"] == SYNTHETIC_DESIGN_COMMIT
    assert receipt["consumed_at_utc"] == "2026-08-12T00:00:00Z"


def test_no_staging_files_left_behind_after_consumption(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    entries = list(Path(state_root).iterdir())
    assert len(entries) == 1
    assert ".staging-" not in entries[0].name


def test_write_failure_never_leaks_private_state_root_path(tmp_path, monkeypatch):
    secret = "/very/secret/private/state/root"

    def poisoned_fsync(fd):
        raise OSError(f"disk full at {secret}")

    monkeypatch.setattr(gate_consumption.os, "fsync", poisoned_fsync)
    with pytest.raises(gate_consumption.V8BHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            tmp_path / "state", gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_STATE_WRITE_FAILED"
    assert secret not in excinfo.value.reason


def test_canonical_state_root_is_outside_the_repository():
    assert gate_consumption.CANONICAL_REPOSITORY_ROOT not in gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.parents
    assert gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT != gate_consumption.CANONICAL_REPOSITORY_ROOT


def test_module_performs_no_io_on_import():
    import importlib
    import sys

    module_name = "src.v8b_human_gate_consumption"
    sys.modules.pop(module_name, None)
    before = set(Path(gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT).parent.glob("*")) if gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.parent.exists() else set()
    importlib.import_module(module_name)
    after = set(Path(gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT).parent.glob("*")) if gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.parent.exists() else set()
    assert before == after
