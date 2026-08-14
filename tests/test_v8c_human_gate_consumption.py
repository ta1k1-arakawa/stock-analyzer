from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8c_human_gate_consumption as gate_consumption

SYNTHETIC_DESIGN_COMMIT = "a" * 40


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def test_known_gates_are_the_exact_nine_named_gates():
    assert set(gate_consumption.KNOWN_GATES) == {
        "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1C",
        "HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION",
        "T1C_TRANSPORT_READINESS_HUMAN_GATE",
        "T1C_RAW_ACQUISITION_HUMAN_GATE",
        "SEPARATE_T1C_RESEARCH_OPENING_GATE",
        "HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE",
        "T2_TRANSPORT_READINESS_HUMAN_GATE",
        "T2_RAW_ACQUISITION_HUMAN_GATE",
        "SEPARATE_T2_RESEARCH_OPENING_GATE",
    }


def test_readiness_gates_are_the_exact_per_authorization_set():
    assert gate_consumption.PER_AUTHORIZATION_GATES == {
        gate_consumption.GATE_T1C_TRANSPORT_READINESS,
        gate_consumption.GATE_T2_TRANSPORT_READINESS,
    }


def test_unknown_gate_rejected():
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once("/tmp/whatever", "SOME_OTHER_GATE", SYNTHETIC_DESIGN_COMMIT, clock=clock_stub)
    assert excinfo.value.reason == "V8C_HUMAN_GATE_UNKNOWN"


# ---------------------------------------------------------------------------
# Durable one-shot forever gates (non-readiness)
# ---------------------------------------------------------------------------


def test_consume_once_then_second_call_blocks(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(state_root, gate_consumption.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub)
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(state_root, gate_consumption.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub)
    assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1C


def test_consumption_durable_across_new_module_load_simulating_restart(tmp_path):
    """A 'process restart' is simulated by reloading the module fresh and
    re-pointing its consumption root at the same on-disk ledger; the second
    logical process must still see the gate as consumed."""
    import importlib
    import sys

    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(state_root, gate_consumption.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub)

    sys.modules.pop("src.v8c_human_gate_consumption", None)
    reloaded = importlib.import_module("src.v8c_human_gate_consumption")
    assert reloaded.has_gate_been_consumed(state_root, reloaded.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT) is True
    with pytest.raises(reloaded.V8CHumanGateConsumptionBlocked):
        reloaded.consume_gate_once(state_root, reloaded.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub)
    importlib.import_module("src.v8c_human_gate_consumption")


def test_non_readiness_gate_rejects_authorization_identity(tmp_path):
    state_root = tmp_path / "state"
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            state_root, gate_consumption.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
            authorization_identity="should-not-be-allowed",
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_AUTHORIZATION_IDENTITY_NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Per-authorization readiness gates: same authorization blocks a replay,
# but a fresh authorization is a fresh, consumable key.
# ---------------------------------------------------------------------------


def test_readiness_gate_requires_authorization_identity(tmp_path):
    state_root = tmp_path / "state"
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_AUTHORIZATION_IDENTITY_REQUIRED"


def test_readiness_gate_same_authorization_replay_blocks(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-TOKEN-1",
    )
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.consume_gate_once(
            state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
            authorization_identity="AUTH-TOKEN-1",
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_T1C_TRANSPORT_READINESS


def test_readiness_gate_fresh_authorization_permits_a_new_probe(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-TOKEN-1",
    )
    # A distinct authorization identity is a fresh key -- must NOT block.
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-TOKEN-2",
    )
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT,
        authorization_identity="AUTH-TOKEN-1",
    ) is True
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT,
        authorization_identity="AUTH-TOKEN-3",
    ) is False


def test_t1c_and_t2_readiness_gates_are_independent(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-TOKEN-1",
    )
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T2_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT,
        authorization_identity="AUTH-TOKEN-1",
    ) is False


def test_readiness_does_not_consume_raw_acquisition_gate(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-TOKEN-1",
    )
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, SYNTHETIC_DESIGN_COMMIT
    ) is False


def test_no_deletion_or_reset_api_exists():
    assert not hasattr(gate_consumption, "delete_receipt")
    assert not hasattr(gate_consumption, "reset_gate")
    for name in gate_consumption.__all__:
        assert "delete" not in name.lower()
        assert "reset" not in name.lower()


def test_canonical_state_root_is_outside_the_repository():
    assert gate_consumption.CANONICAL_REPOSITORY_ROOT not in gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.parents
    assert gate_consumption.CANONICAL_CONSUMPTION_STATE_ROOT.name == "v8c-human-gate-state"


# ---------------------------------------------------------------------------
# HIGH-1: receipts carry a privacy-safe authorization-identity hash, and a
# public reader can mechanically re-validate a receipt located by key --
# never by trusting another artifact's claim about it.
# ---------------------------------------------------------------------------


def test_per_authorization_receipt_carries_identity_hash_not_raw_identity(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="RAW-HUMAN-TOKEN",
    )
    key = gate_consumption.compute_receipt_key(
        gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, "RAW-HUMAN-TOKEN"
    )
    receipt = gate_consumption.read_gate_consumption_receipt(state_root, key)
    assert receipt["gate"] == gate_consumption.GATE_T1C_TRANSPORT_READINESS
    assert receipt["v8c_frozen_design_commit"] == SYNTHETIC_DESIGN_COMMIT
    assert receipt["per_authorization_gate"] is True
    import hashlib

    assert receipt["authorization_identity_sha256"] == hashlib.sha256(b"RAW-HUMAN-TOKEN").hexdigest()
    raw_text = (state_root / (key + ".json")).read_text()
    assert "RAW-HUMAN-TOKEN" not in raw_text


def test_non_per_authorization_receipt_has_null_identity_hash(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(state_root, gate_consumption.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub)
    key = gate_consumption.compute_receipt_key(gate_consumption.GATE_ALLOCATE_T1C, SYNTHETIC_DESIGN_COMMIT)
    receipt = gate_consumption.read_gate_consumption_receipt(state_root, key)
    assert receipt["authorization_identity_sha256"] is None
    assert receipt["per_authorization_gate"] is False


def test_read_gate_consumption_receipt_missing_blocks(tmp_path):
    state_root = tmp_path / "state"
    key = gate_consumption.compute_receipt_key(
        gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, "NEVER-CONSUMED"
    )
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(state_root, key)
    assert excinfo.value.reason == "V8C_HUMAN_GATE_RECEIPT_MISSING"


def test_read_gate_consumption_receipt_rejects_malformed_key():
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt("/tmp/whatever", "not-a-valid-hex-key")
    assert excinfo.value.reason == "V8C_HUMAN_GATE_RECEIPT_KEY_INVALID"


def test_read_gate_consumption_receipt_rejects_tampered_schema(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-1",
    )
    key = gate_consumption.compute_receipt_key(
        gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, "AUTH-1"
    )
    path = state_root / (key + ".json")
    path.write_text(path.read_text().replace('"per_authorization_gate":true', '"per_authorization_gate":false'))
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(state_root, key)
    assert excinfo.value.reason == "V8C_HUMAN_GATE_RECEIPT_PER_AUTHORIZATION_FLAG_MISMATCH"


# ---------------------------------------------------------------------------
# HIGH-1 (round 2): the receipt key must be mechanically recomputable from
# the receipt's own safe content -- never merely a syntactically valid
# 64-hex filename with self-consistent-looking field values.
# ---------------------------------------------------------------------------


def test_receipt_key_recomputes_from_content_after_normal_consumption(tmp_path):
    """Test A."""
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-1",
    )
    key = gate_consumption.compute_receipt_key(
        gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, "AUTH-1"
    )
    receipt = gate_consumption.read_gate_consumption_receipt(state_root, key)
    recomputed = gate_consumption._receipt_key_from_identity_hash(
        receipt["gate"], receipt["v8c_frozen_design_commit"], receipt["authorization_identity_sha256"]
    )
    assert recomputed == key


def test_receipt_at_arbitrary_wrong_filename_blocks(tmp_path):
    """Test B: a syntactically valid receipt (correct schema, correct
    fields) written at an ARBITRARY 64-hex filename that does not equal
    the canonical key its own content derives."""
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": gate_consumption.SCHEMA_VERSION,
        "study_name": gate_consumption.STUDY_NAME,
        "repository": gate_consumption.REPOSITORY_IDENTITY,
        "gate": gate_consumption.GATE_T1C_TRANSPORT_READINESS,
        "v8c_frozen_design_commit": SYNTHETIC_DESIGN_COMMIT,
        "per_authorization_gate": True,
        "authorization_identity_sha256": "e" * 64,
        "consumed_at_utc": "2026-08-14T00:00:00Z",
    }
    import json as _json

    arbitrary_key = "0" * 64
    (state_root / (arbitrary_key + ".json")).write_text(
        _json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(state_root, arbitrary_key)
    assert excinfo.value.reason == "V8C_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH"


def test_valid_receipt_copied_to_another_filename_blocks(tmp_path):
    """Test D: copy a genuinely-consumed receipt's exact bytes to a
    DIFFERENT 64-hex filename -- the copy's own content still recomputes
    to the ORIGINAL key, not the new filename it was copied to."""
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, clock=clock_stub,
        authorization_identity="AUTH-1",
    )
    original_key = gate_consumption.compute_receipt_key(
        gate_consumption.GATE_T1C_TRANSPORT_READINESS, SYNTHETIC_DESIGN_COMMIT, "AUTH-1"
    )
    original_bytes = (state_root / (original_key + ".json")).read_bytes()
    copy_key = "1" * 64
    assert copy_key != original_key
    (state_root / (copy_key + ".json")).write_bytes(original_bytes)
    with pytest.raises(gate_consumption.V8CHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(state_root, copy_key)
    assert excinfo.value.reason == "V8C_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH"
    # The original, correctly-keyed receipt still validates.
    assert gate_consumption.read_gate_consumption_receipt(state_root, original_key)["gate"] == gate_consumption.GATE_T1C_TRANSPORT_READINESS
