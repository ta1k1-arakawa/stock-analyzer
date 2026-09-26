"""Repository-only fail-closed checks for the resolved XLSX candidate freeze."""

import json

import pytest

from scripts import v13_xlsx_successor_freeze as freeze


def test_frozen_candidate_passes():
    assert freeze.validate_freeze() == {
        "status": "PASS", "candidate_package_count": 29, "artifact_count": 29,
        "current_authority_package_count": 27,
        "production_xlsx_readiness": "NOT_YET_PROVEN",
    }


@pytest.mark.parametrize("old,new", [
    (b"openpyxl==3.1.5", b"openpyxl==3.1.4"),
    (b"et-xmlfile==2.0.0\n", b""),
    (b"xlrd==2.0.2\n", b"xlrd==2.0.2\nother==1.0\n"),
    (b"\n", b"\r\n"),
])
def test_changed_candidate_lock_fails(tmp_path, old, new):
    path = tmp_path / "candidate.txt"
    path.write_bytes(freeze.LOCK.read_bytes().replace(old, new, 1))
    with pytest.raises(ValueError):
        freeze.validate_freeze(lock_path=path)


@pytest.mark.parametrize("field,value", [
    ("source_gpt_reviewed_sha", "0" * 40),
    ("source_candidate_sha256", "0" * 64),
    ("source_safe_evidence_sha256", "0" * 64),
    ("authorization_blob", "0" * 40),
    ("historical_phase_b_wheel_download_count", 28),
    ("repository_task_network_requests", 1),
    ("candidate_installed", True),
    ("production_xlsx_readiness", "PASS"),
])
def test_changed_binding_fails(tmp_path, field, value):
    record = json.loads(freeze.RECORD.read_bytes())
    record[field] = value
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.validate_freeze(record_path=path)


@pytest.mark.parametrize("field,value", [
    ("name", "other"), ("version", "0"),
    ("filename", "openpyxl-3.1.4-py2.py3-none-any.whl"),
    ("sha256", "0" * 64),
])
def test_changed_wheel_fails(tmp_path, field, value):
    record = json.loads(freeze.RECORD.read_bytes())
    record["resolved_wheels"][11][field] = value
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.validate_freeze(record_path=path)


def test_missing_and_extra_wheel_fail(tmp_path):
    record = json.loads(freeze.RECORD.read_bytes())
    for wheels in (record["resolved_wheels"][:-1],
                   record["resolved_wheels"] + [record["resolved_wheels"][-1]]):
        record["resolved_wheels"] = wheels
        path = tmp_path / "record.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError):
            freeze.validate_freeze(record_path=path)
