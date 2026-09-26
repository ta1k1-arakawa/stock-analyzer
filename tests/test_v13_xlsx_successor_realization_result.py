"""Repository-only checks for the consumed isolated XLSX result checkpoint."""

import json

import pytest

from scripts import v13_xlsx_successor_realization_result as result


def test_safe_result_and_frozen_locks_pass():
    assert result.validate_result() == {
        "status": "PASS", "candidate_package_count": 29,
        "current_authority_package_count": 27, "successor_promoted": False,
    }


@pytest.mark.parametrize("key,value", [
    ("execution_head", "0" * 40),
    ("authorization_blob", "0" * 40),
    ("candidate_lock_sha256", "0" * 64),
    ("source_safe_evidence_sha256", "0" * 64),
    ("wheel_manifest_sha256", "0" * 64),
    ("status", "PASS"),
    ("authorization_consumed", False),
    ("authorization_reusable", True),
    ("installed_package_count", 28),
    ("production_xlsx_readiness", "NOT_YET_PROVEN"),
    ("current_authority_preserved", False),
    ("successor_promoted", True),
    ("successor_promotion_authorized", True),
    ("network_requests", 1),
    ("source_issue", True),  # bool must not pass as integer 105
])
def test_changed_binding_fails(tmp_path, key, value):
    value_map = json.loads(result.RESULT.read_bytes())
    value_map[key] = value
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value_map), encoding="utf-8")
    with pytest.raises(ValueError, match="SAFE_RESULT_SCHEMA_OR_BINDING_MISMATCH"):
        result.validate_result(result_path=path)


@pytest.mark.parametrize("change", ["extra", "missing", "duplicate"])
def test_schema_is_closed(tmp_path, change):
    value_map = json.loads(result.RESULT.read_bytes())
    if change == "extra":
        value_map["candidate_path"] = "forbidden"
    elif change == "missing":
        del value_map["future_profitability_established"]
    raw = json.dumps(value_map)
    if change == "duplicate":
        raw = raw.replace('"status": "PASS_CONSUMED"',
                          '"status": "PASS_CONSUMED", "status": "PASS_CONSUMED"')
    path = tmp_path / "result.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        result.validate_result(result_path=path)
