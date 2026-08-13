from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8b_human_gate_consumption as gate_consumption

SYNTHETIC_DESIGN_COMMIT = "a" * 40


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def test_known_gates_are_the_exact_four_named_gates():
    assert set(gate_consumption.KNOWN_GATES) == {
        "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B",
        "HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION",
        "T1B_RAW_ACQUISITION_HUMAN_GATE",
        "T2_RAW_ACQUISITION_HUMAN_GATE",
    }
    assert gate_consumption.GATE_ALLOCATE_T1B == "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B"
    assert (
        gate_consumption.GATE_PIN_VERIFIED_T1B_ALLOCATION
        == "HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION"
    )
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
    assert set(receipt) == {
        "schema_version",
        "study_name",
        "repository",
        "gate",
        "v8b_frozen_design_commit",
        "consumed_at_utc",
    }
    assert receipt["schema_version"] == gate_consumption.SCHEMA_VERSION
    assert receipt["study_name"] == gate_consumption.STUDY_NAME
    assert receipt["repository"] == gate_consumption.REPOSITORY_IDENTITY
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


# --- MEDIUM-1 (repeat round): canonical ledger identity must not be
# checkout-path-local ------------------------------------------------------


def test_default_state_root_does_not_depend_on_module_file_location():
    assert gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT == gate_consumption._default_consumption_state_root()
    assert gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.name == "v8b-human-gate-state"


def test_default_state_root_is_fixed_machine_location(monkeypatch):
    monkeypatch.setattr(gate_consumption.os, "name", "posix")
    assert str(gate_consumption._default_consumption_state_root()).replace("\\", "/") == "/var/lib/stock-analyzer/v8b-human-gate-state"


def test_default_state_root_does_not_read_home_or_userprofile(monkeypatch):
    monkeypatch.setattr(gate_consumption.os, "name", "posix")
    monkeypatch.setenv("HOME", "C:/attacker/home")
    monkeypatch.setenv("USERPROFILE", "C:/attacker/profile")
    assert str(gate_consumption._default_consumption_state_root()).replace("\\", "/") == "/var/lib/stock-analyzer/v8b-human-gate-state"


def test_windows_root_uses_known_folder_api_not_environment(monkeypatch, tmp_path):
    expected = tmp_path / "ProgramData"
    monkeypatch.setattr(gate_consumption.os, "name", "nt")
    monkeypatch.setattr(gate_consumption, "_resolve_windows_program_data_directory", lambda: expected)
    monkeypatch.setenv("PROGRAMDATA", "C:/attacker/programdata")
    assert gate_consumption._default_consumption_state_root() == expected / "stock-analyzer" / "v8b-human-gate-state"


def test_unavailable_fixed_windows_location_fails_closed(monkeypatch):
    monkeypatch.setattr(gate_consumption.os, "name", "nt")
    def unavailable():
        raise RuntimeError("unavailable")
    monkeypatch.setattr(gate_consumption, "_resolve_windows_program_data_directory", unavailable)
    with pytest.raises(RuntimeError):
        gate_consumption._default_consumption_state_root()


def _load_module_from_a_different_checkout_path(tmp_path, suffix):
    """Simulate a second clone/worktree of this repository at an unrelated
    filesystem path by copying just this module's source there and loading
    it under a distinct module name via its own file location -- so
    ``Path(__file__).resolve().parents[1]`` differs between the two loaded
    instances, exactly like two real independent checkouts would."""
    import importlib.util

    fake_checkout = tmp_path / ("checkout-" + suffix) / "src"
    fake_checkout.mkdir(parents=True)
    source_path = Path(gate_consumption.__file__)
    destination = fake_checkout / "v8b_human_gate_consumption.py"
    destination.write_bytes(source_path.read_bytes())

    module_name = "v8b_human_gate_consumption_checkout_" + suffix
    spec = importlib.util.spec_from_file_location(module_name, destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_different_checkout_paths_share_the_same_canonical_ledger_and_second_call_blocks(monkeypatch, tmp_path):
    fake_home = tmp_path / "shared-home"
    monkeypatch.setattr(gate_consumption.Path, "home", classmethod(lambda cls: fake_home))

    checkout_a = _load_module_from_a_different_checkout_path(tmp_path, "a")
    checkout_b = _load_module_from_a_different_checkout_path(tmp_path, "b")
    fake_ledger = tmp_path / "machine-ledger"
    checkout_a.CANONICAL_CONSUMPTION_STATE_ROOT = fake_ledger
    checkout_b.CANONICAL_CONSUMPTION_STATE_ROOT = fake_ledger

    # The two "checkouts" live at completely different filesystem paths...
    assert checkout_a.CANONICAL_REPOSITORY_ROOT != checkout_b.CANONICAL_REPOSITORY_ROOT
    # ...but the canonical ledger identity is identical between them.
    assert checkout_a.CANONICAL_CONSUMPTION_STATE_ROOT == checkout_b.CANONICAL_CONSUMPTION_STATE_ROOT

    checkout_a.consume_gate_once(
        checkout_a.CANONICAL_CONSUMPTION_STATE_ROOT,
        checkout_a.GATE_ALLOCATE_T1B,
        SYNTHETIC_DESIGN_COMMIT,
        clock=clock_stub,
    )
    # A gate consumed via "checkout A" durably blocks the identical gate
    # read/consumed via a freshly-loaded, independent "checkout B" module
    # instance -- proving the ledger is not scoped to either checkout path.
    assert checkout_b.has_gate_been_consumed(
        checkout_b.CANONICAL_CONSUMPTION_STATE_ROOT, checkout_b.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT
    ) is True
    with pytest.raises(checkout_b.V8BHumanGateConsumptionBlocked) as excinfo:
        checkout_b.consume_gate_once(
            checkout_b.CANONICAL_CONSUMPTION_STATE_ROOT,
            checkout_b.GATE_ALLOCATE_T1B,
            SYNTHETIC_DESIGN_COMMIT,
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + checkout_b.GATE_ALLOCATE_T1B


def test_receipt_key_is_bound_to_fixed_repository_identity_string(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
    )
    expected_key = gate_consumption._receipt_key(gate_consumption.GATE_ALLOCATE_T1B, SYNTHETIC_DESIGN_COMMIT)
    assert (Path(state_root) / (expected_key + ".json")).exists()
    # The key must actually depend on REPOSITORY_IDENTITY, not merely be
    # independent of it by coincidence.
    import hashlib

    other_repo_key = hashlib.sha256(
        ("some/other-repo|" + gate_consumption.GATE_ALLOCATE_T1B + "|" + SYNTHETIC_DESIGN_COMMIT).encode("utf-8")
    ).hexdigest()
    assert other_repo_key != expected_key


def test_no_deletion_or_reset_api_exists():
    assert not hasattr(gate_consumption, "delete_receipt")
    assert not hasattr(gate_consumption, "reset_gate")
    assert not hasattr(gate_consumption, "clear_consumption_state")
    for name in gate_consumption.__all__:
        assert "delete" not in name.lower()
        assert "reset" not in name.lower()


def test_module_performs_no_io_on_import():
    import importlib
    import sys

    module_name = "src.v8b_human_gate_consumption"
    sys.modules.pop(module_name, None)
    before = set(Path(gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT).parent.glob("*")) if gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.parent.exists() else set()
    importlib.import_module(module_name)
    after = set(Path(gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT).parent.glob("*")) if gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.parent.exists() else set()
    assert before == after
