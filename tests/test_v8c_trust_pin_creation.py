from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8c_human_gate_consumption as gate_consumption
from src import v8c_trust_pin as trust_pin
from src import v8c_trust_pin_creation as creation

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40
SYNTHETIC_HASH = "c" * 64


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _summary(**overrides):
    summary = {
        "result": "PASS",
        "parent_v8_partition_manifest_sha256": "d" * 64,
        "parent_v8_partition_implementation_commit": SYNTHETIC_COMMIT,
        "parent_t_spare_ticker_count": 1904,
        "parent_t_spare_ticker_list_sha256": "e" * 64,
        "t1c_ticker_count": 300,
        "t1c_ticker_list_sha256": "f" * 64,
        "remaining_t_spare_ticker_count": 1604,
        "remaining_t_spare_ticker_list_sha256": "0" * 64,
        "v8c_frozen_design_commit": SYNTHETIC_COMMIT,
        "v8c_allocation_implementation_commit": SYNTHETIC_REVIEWED_COMMIT,
        "artifact_self_hash": SYNTHETIC_HASH,
    }
    summary.update(overrides)
    return summary


def run(**overrides):
    overrides.setdefault("consumption_state_root", Path(tempfile.gettempdir()) / ("v8c_gate_state-" + uuid.uuid4().hex))
    return creation._create_v8c_trusted_allocation_pin_production_with_dependencies(**overrides)


def _base_deps(**overrides):
    deps = dict(
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        allocation_verification_resolver=lambda: _summary(),
        clock=clock_stub,
    )
    deps.update(overrides)
    return deps


def test_confirmation_token_is_frozen_literal():
    assert creation.PIN_CREATION_CONFIRMATION == "V8C_PRODUCTION_CREATE_TRUSTED_ALLOCATION_PIN"


def test_wrong_confirmation_blocks(tmp_path):
    with pytest.raises(creation.V8CTrustPinCreationBlocked) as excinfo:
        run(
            confirmation="wrong",
            human_pin_authorization="whatever",
            allocation_artifact_path=tmp_path / "x.json",
            partition_manifest_path=tmp_path / "y.json",
            t1b_allocation_artifact_path=tmp_path / "t1b.json",
            output_path=tmp_path / "out.json",
            authorization_note="n",
            **_base_deps(),
        )
    assert excinfo.value.reason == "V8C_PIN_CREATION_CONFIRMATION_INVALID"


def test_wrong_human_authorization_blocks_before_gate_consumption(tmp_path):
    state_root = tmp_path / "state"
    with pytest.raises(creation.V8CTrustPinCreationBlocked) as excinfo:
        run(
            confirmation=creation.PIN_CREATION_CONFIRMATION,
            human_pin_authorization="not the right gate string",
            allocation_artifact_path=tmp_path / "x.json",
            partition_manifest_path=tmp_path / "y.json",
            t1b_allocation_artifact_path=tmp_path / "t1b.json",
            output_path=tmp_path / "out.json",
            authorization_note="n",
            consumption_state_root=state_root,
            **_base_deps(),
        )
    assert excinfo.value.reason == "V8C_HUMAN_PIN_AUTHORIZATION_INVALID"
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_PIN_T1C_ALLOCATION, creation.EXPECTED_V8C_FROZEN_DESIGN_COMMIT
    ) is False


def test_valid_flow_writes_pin_and_consumes_gate_exactly_once(tmp_path):
    state_root = tmp_path / "state"
    output_path = tmp_path / "V8C_TRUSTED_ALLOCATION.json"
    human_auth = trust_pin.expected_human_gate(SYNTHETIC_HASH)

    pin = run(
        confirmation=creation.PIN_CREATION_CONFIRMATION,
        human_pin_authorization=human_auth,
        allocation_artifact_path=tmp_path / "x.json",
        partition_manifest_path=tmp_path / "y.json",
        t1b_allocation_artifact_path=tmp_path / "t1b.json",
        output_path=output_path,
        authorization_note="note",
        consumption_state_root=state_root,
        **_base_deps(),
    )
    assert pin["authorization_status"] == "AUTHORIZED"
    assert output_path.exists()
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_PIN_T1C_ALLOCATION, creation.EXPECTED_V8C_FROZEN_DESIGN_COMMIT
    ) is True

    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called: gate already consumed")

    with pytest.raises(creation.V8CTrustPinCreationBlocked) as excinfo:
        run(
            confirmation=creation.PIN_CREATION_CONFIRMATION,
            human_pin_authorization=human_auth,
            allocation_artifact_path=tmp_path / "x.json",
            partition_manifest_path=tmp_path / "y.json",
            t1b_allocation_artifact_path=tmp_path / "t1b.json",
            output_path=tmp_path / "out2.json",
            authorization_note="note",
            consumption_state_root=state_root,
            git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
            allocation_verification_resolver=forbidden, clock=clock_stub,
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_PIN_T1C_ALLOCATION


def test_writing_twice_to_same_path_blocks(tmp_path):
    output_path = tmp_path / "V8C_TRUSTED_ALLOCATION.json"
    human_auth = trust_pin.expected_human_gate(SYNTHETIC_HASH)
    run(
        confirmation=creation.PIN_CREATION_CONFIRMATION, human_pin_authorization=human_auth,
        allocation_artifact_path=tmp_path / "x.json", partition_manifest_path=tmp_path / "y.json",
        t1b_allocation_artifact_path=tmp_path / "t1b.json",
        output_path=output_path, authorization_note="note",
        consumption_state_root=tmp_path / "state1", **_base_deps(),
    )
    with pytest.raises(creation.V8CTrustPinCreationBlocked) as excinfo:
        run(
            confirmation=creation.PIN_CREATION_CONFIRMATION, human_pin_authorization=human_auth,
            allocation_artifact_path=tmp_path / "x.json", partition_manifest_path=tmp_path / "y.json",
            t1b_allocation_artifact_path=tmp_path / "t1b.json",
            output_path=output_path, authorization_note="note",
            consumption_state_root=tmp_path / "state2", **_base_deps(),
        )
    assert excinfo.value.reason == "V8C_TRUSTED_ALLOCATION_PIN_ALREADY_EXISTS"
