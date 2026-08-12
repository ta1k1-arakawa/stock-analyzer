from __future__ import annotations

import pytest

from src import v8b_trust_pin as trust_pin

SYNTHETIC_COMMIT_A = "a" * 40
SYNTHETIC_COMMIT_B = "b" * 40


def _pass_summary(**overrides):
    summary = {
        "result": "PASS",
        "parent_v8_partition_manifest_sha256": "0" * 64,
        "parent_v8_partition_implementation_commit": SYNTHETIC_COMMIT_A,
        "parent_t_spare_ticker_count": 1904,
        "parent_t_spare_ticker_list_sha256": "1" * 64,
        "t1b_ticker_count": 300,
        "t1b_ticker_list_sha256": "2" * 64,
        "remaining_t_spare_ticker_count": 1604,
        "remaining_t_spare_ticker_list_sha256": "3" * 64,
        "v8b_frozen_design_commit": SYNTHETIC_COMMIT_B,
        "v8b_allocation_implementation_commit": SYNTHETIC_COMMIT_A,
        "artifact_self_hash": "4" * 64,
    }
    summary.update(overrides)
    return summary


def not_authorized_pin() -> dict:
    return {field: None for field in trust_pin.TRUST_PIN_FIELDS} | {
        "schema_version": trust_pin.SCHEMA_VERSION,
        "study_name": trust_pin.STUDY_NAME,
        "artifact_role": trust_pin.ARTIFACT_ROLE,
        "logical_block": trust_pin.LOGICAL_BLOCK,
        "authorization_status": "NOT_AUTHORIZED",
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_trust_pin_from_pass_summary():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(),
        human_gate="V8B_HUMAN_AUTHORIZE_T1B_ALLOCATION_PIN",
        authorization_note="test-only pin",
    )
    assert pin["authorization_status"] == "AUTHORIZED"
    assert pin["t1b_ticker_count"] == 300
    assert set(pin) == set(trust_pin.TRUST_PIN_FIELDS)


def test_build_trust_pin_rejects_non_pass_result():
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.build_trust_pin(
            verification_result_summary=_pass_summary(result="BLOCK"),
            human_gate="gate",
            authorization_note="note",
        )
    assert excinfo.value.reason == "VERIFICATION_RESULT_NOT_PASS"


@pytest.mark.parametrize("forbidden", ("t1b_tickers", "remaining_t_spare_tickers", "parent_t_spare_tickers"))
def test_build_trust_pin_rejects_summary_with_ticker_identities(forbidden):
    summary = _pass_summary()
    summary[forbidden] = ["0001", "0002"]
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.build_trust_pin(
            verification_result_summary=summary, human_gate="gate", authorization_note="note"
        )
    assert excinfo.value.reason == "VERIFICATION_SUMMARY_CONTAINS_TICKER_IDENTITIES"


def test_build_trust_pin_rejects_incomplete_summary():
    summary = _pass_summary()
    del summary["artifact_self_hash"]
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.build_trust_pin(
            verification_result_summary=summary, human_gate="gate", authorization_note="note"
        )
    assert excinfo.value.reason == "VERIFICATION_SUMMARY_SCHEMA_INVALID"


def test_build_trust_pin_rejects_empty_human_gate():
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.build_trust_pin(
            verification_result_summary=_pass_summary(), human_gate="   ", authorization_note="note"
        )
    assert excinfo.value.reason == "HUMAN_GATE_INVALID"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validate_authorized_pin_round_trips():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    validated = trust_pin.validate_trust_pin(pin)
    assert validated == pin


def test_validate_not_authorized_placeholder():
    validated = trust_pin.validate_trust_pin(not_authorized_pin())
    assert validated["authorization_status"] == "NOT_AUTHORIZED"


def test_not_authorized_pin_with_populated_field_blocks():
    pin = not_authorized_pin()
    pin["t1b_ticker_count"] = 300
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(pin)
    assert excinfo.value.reason == "TRUST_PIN_UNAUTHORIZED_FIELDS_INVALID"


def test_authorized_pin_with_wrong_t1b_count_blocks():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    tampered = dict(pin)
    tampered["t1b_ticker_count"] = 301
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(tampered)
    assert excinfo.value.reason == "T1B_TICKER_COUNT_INVALID"


def test_authorized_pin_with_inconsistent_remaining_count_blocks():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    tampered = dict(pin)
    tampered["remaining_t_spare_ticker_count"] = 999999
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(tampered)
    assert excinfo.value.reason == "REMAINING_T_SPARE_TICKER_COUNT_INVALID"


def test_validate_rejects_unknown_authorization_status():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    tampered = dict(pin)
    tampered["authorization_status"] = "MAYBE"
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(tampered)
    assert excinfo.value.reason == "TRUST_PIN_AUTHORIZATION_STATUS_INVALID"


def test_validate_rejects_wrong_schema_version():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    tampered = dict(pin)
    tampered["schema_version"] = "V0"
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(tampered)
    assert excinfo.value.reason == "TRUST_PIN_SCHEMA_VERSION_MISMATCH"


def test_validate_rejects_missing_field():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    incomplete = dict(pin)
    del incomplete["authorization_note"]
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(incomplete)
    assert excinfo.value.reason == "TRUST_PIN_SCHEMA_INVALID"


@pytest.mark.parametrize("forbidden", ("t1b_tickers", "remaining_t_spare_tickers", "parent_t_spare_tickers"))
def test_validate_rejects_pin_with_ticker_identity_field(forbidden):
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    tampered = dict(pin)
    tampered[forbidden] = ["0001"]
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(tampered)
    assert excinfo.value.reason == "TRUST_PIN_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# read_trust_pin_bytes: duplicate-key rejection, no writer exists
# ---------------------------------------------------------------------------


def test_read_trust_pin_bytes_rejects_duplicate_key():
    raw = b'{"schema_version": "a", "schema_version": "b"}'
    with pytest.raises(trust_pin.V8BTrustPinBlocked) as excinfo:
        trust_pin.read_trust_pin_bytes(raw)
    assert excinfo.value.reason == "TRUST_PIN_DUPLICATE_KEY"


def test_read_trust_pin_bytes_round_trips():
    pin = trust_pin.build_trust_pin(
        verification_result_summary=_pass_summary(), human_gate="gate", authorization_note="note"
    )
    import json

    raw = json.dumps(pin).encode("utf-8")
    reloaded = trust_pin.read_trust_pin_bytes(raw)
    assert reloaded == pin


def test_module_defines_no_write_function():
    """This implementation phase does not authorize creating the real
    V8B_TRUSTED_ALLOCATION.json pin artifact -- there must be no writer."""
    assert not any(name.startswith("write_") for name in trust_pin.__all__)
    assert not hasattr(trust_pin, "write_trust_pin_once")
