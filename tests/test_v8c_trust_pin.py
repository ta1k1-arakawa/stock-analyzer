from __future__ import annotations

import pytest

from src import v8c_trust_pin as trust_pin

SYNTHETIC_HASH = "a" * 64
SYNTHETIC_COMMIT = "b" * 40


def _summary(**overrides):
    summary = {
        "result": "PASS",
        "parent_v8_partition_manifest_sha256": "c" * 64,
        "parent_v8_partition_implementation_commit": SYNTHETIC_COMMIT,
        "parent_t_spare_ticker_count": 1904,
        "parent_t_spare_ticker_list_sha256": "d" * 64,
        "t1c_ticker_count": 300,
        "t1c_ticker_list_sha256": SYNTHETIC_HASH,
        "remaining_t_spare_ticker_count": 1604,
        "remaining_t_spare_ticker_list_sha256": "e" * 64,
        "v8c_frozen_design_commit": SYNTHETIC_COMMIT,
        "v8c_allocation_implementation_commit": SYNTHETIC_COMMIT,
        "artifact_self_hash": SYNTHETIC_HASH,
    }
    summary.update(overrides)
    return summary


def test_build_trust_pin_requires_pass_result():
    with pytest.raises(trust_pin.V8CTrustPinBlocked) as excinfo:
        trust_pin.build_trust_pin(verification_result_summary=_summary(result="BLOCK"), authorization_note="n")
    assert excinfo.value.reason == "VERIFICATION_RESULT_NOT_PASS"


def test_build_trust_pin_rejects_summary_with_ticker_identities():
    summary = _summary()
    summary["t1c_tickers"] = ["FAKE0001"]
    with pytest.raises(trust_pin.V8CTrustPinBlocked) as excinfo:
        trust_pin.build_trust_pin(verification_result_summary=summary, authorization_note="n")
    assert excinfo.value.reason == "VERIFICATION_SUMMARY_CONTAINS_TICKER_IDENTITIES"


def test_build_trust_pin_produces_frozen_human_gate_grammar():
    pin = trust_pin.build_trust_pin(verification_result_summary=_summary(), authorization_note="note")
    assert pin["human_gate"] == trust_pin.HUMAN_GATE_PREFIX + SYNTHETIC_HASH
    assert pin["authorization_status"] == "AUTHORIZED"
    assert set(pin) == set(trust_pin.TRUST_PIN_FIELDS)


def test_validate_trust_pin_accepts_well_formed_authorized_pin():
    pin = trust_pin.build_trust_pin(verification_result_summary=_summary(), authorization_note="note")
    validated = trust_pin.validate_trust_pin(pin)
    assert validated == pin


def test_validate_trust_pin_rejects_fabricated_human_gate():
    pin = trust_pin.build_trust_pin(verification_result_summary=_summary(), authorization_note="note")
    pin["human_gate"] = "V8C_HUMAN_AUTHORIZE_T1C_ALLOCATION_PIN_AT_" + ("0" * 64)
    with pytest.raises(trust_pin.V8CTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(pin)
    assert excinfo.value.reason == "TRUST_PIN_HUMAN_GATE_INVALID"


def test_validate_trust_pin_rejects_wrong_ticker_count():
    pin = trust_pin.build_trust_pin(verification_result_summary=_summary(), authorization_note="note")
    pin["t1c_ticker_count"] = 299
    with pytest.raises(trust_pin.V8CTrustPinBlocked) as excinfo:
        trust_pin.validate_trust_pin(pin)
    assert excinfo.value.reason == "T1C_TICKER_COUNT_INVALID"


def test_validate_trust_pin_not_authorized_requires_all_nullable_fields_none():
    pin = {field: None for field in trust_pin.TRUST_PIN_FIELDS}
    pin["schema_version"] = trust_pin.SCHEMA_VERSION
    pin["study_name"] = trust_pin.STUDY_NAME
    pin["artifact_role"] = trust_pin.ARTIFACT_ROLE
    pin["logical_block"] = trust_pin.LOGICAL_BLOCK
    pin["authorization_status"] = "NOT_AUTHORIZED"
    pin["authorization_note"] = "n"
    validated = trust_pin.validate_trust_pin(pin)
    assert validated["authorization_status"] == "NOT_AUTHORIZED"


def test_read_trust_pin_bytes_rejects_duplicate_keys():
    raw = b'{"a": 1, "a": 2}'
    with pytest.raises(trust_pin.V8CTrustPinBlocked) as excinfo:
        trust_pin.read_trust_pin_bytes(raw)
    assert excinfo.value.reason == "TRUST_PIN_DUPLICATE_KEY"


def test_no_write_function_exists():
    assert not hasattr(trust_pin, "write_trust_pin")
    for name in trust_pin.__all__:
        assert "write" not in name.lower()
