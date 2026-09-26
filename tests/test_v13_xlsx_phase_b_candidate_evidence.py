"""Synthetic-only tests for the consumed candidate evidence export."""

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import v13_xlsx_phase_b_candidate_evidence as evidence
from scripts import v13_xlsx_phase_b_resolution as phase_b


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(phase_b, "_local_state_base", lambda: tmp_path)
    base = {f"pkg{i:02}": "1.0" for i in range(27)}
    monkeypatch.setattr(evidence.successor, "validate_direct_spec", lambda: base)
    root = phase_b._canonical_root(phase_b.AUTH_BLOB)
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    for name in list(base) + ["openpyxl", "et-xmlfile"]:
        version = "3.1.5" if name == "openpyxl" else "1.0"
        filename = f"{name}-{version}-py3-none-any.whl"
        dist = f"{name}-{version}.dist-info"
        metadata = f"Name: {name}\nVersion: {version}\n"
        if name == "openpyxl":
            metadata += "Requires-Dist: et-xmlfile\n"
        with zipfile.ZipFile(wheelhouse / filename, "w") as archive:
            archive.writestr(f"{dist}/METADATA", metadata)
            archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\n")
    inspected = phase_b.inspect_wheels(wheelhouse, base)
    candidate = dict(inspected, execution_head=evidence.EXECUTION_HEAD,
                     approved_phase_a_sha=phase_b.PHASE_A,
                     authorization_blob=phase_b.AUTH_BLOB, official_index_url=phase_b.INDEX,
                     resolver="pip download --only-binary=:all:", package_index_request_count=29,
                     wheel_download_count=29, package_installations=0,
                     active_environment_mutated=False, successor_promoted=False,
                     jpx_yahoo_requests=0, private_reads=0, selected_500_constructed=False,
                     model_fits=0, backtests=0, aq_executions=0, trades=0)
    _write(root / "candidate.json", candidate)
    digest = hashlib.sha256((root / "candidate.json").read_bytes()).hexdigest()
    monkeypatch.setattr(evidence, "CANDIDATE_SHA256", digest)
    _write(root / "attempt.json", {
        "schema_version": "V13_XLSX_PHASE_B_ATTEMPT_V1", "status": "CONSUMED_PENDING",
        "execution_head": evidence.EXECUTION_HEAD, "authorization_blob": phase_b.AUTH_BLOB,
        "one_shot": True, "reusable": False, "package_installations": 0})
    _write(root / "receipt.json", {
        "schema_version": "V13_XLSX_PHASE_B_RECEIPT_V1", "status": "PASS_CONSUMED",
        "execution_head": evidence.EXECUTION_HEAD, "authorization_blob": phase_b.AUTH_BLOB,
        "candidate_sha256": digest, "package_count": 29, "artifact_count": 29,
        "package_index_request_count": 29, "wheel_download_count": 29,
        "package_installations": 0, "active_environment_mutated": False,
        "successor_promoted": False, "reusable": False})
    return root, candidate


def test_rehearsal_and_checkpoint(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evidence"])
    monkeypatch.setattr(evidence, "_safe_root", lambda: pytest.fail("durable state read"))
    assert evidence.main() == 0
    assert json.loads(capsys.readouterr().out)["durable_state_reads"] == 0
    state = (phase_b.ROOT / "PROJECT_STATE.md").read_text(encoding="utf-8")
    log = (phase_b.ROOT / "PROJECT_DECISION_LOG.md").read_text(encoding="utf-8")
    for value in (evidence.EXECUTION_HEAD, phase_b.AUTH_BLOB, evidence.CANDIDATE_SHA256,
                  "PASS_CONSUMED", "29", "27"):
        assert value in state and value in log


def test_synthetic_export_is_stable_and_read_only(tmp_path: Path, monkeypatch) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    first, sha = evidence.export_evidence()
    second, sha2 = evidence.export_evidence()
    assert (first, sha) == (second, sha2)
    assert first["openpyxl_version"] == "3.1.5"
    assert {p["name"] for p in first["successor_closure"]} == {"openpyxl", "et-xmlfile"}
    assert len(first["resolved_wheels"]) == 29
    assert str(tmp_path) not in json.dumps(first)
    raw = (json.dumps(first, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    assert sha == hashlib.sha256(raw).hexdigest()
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("file,key,value", [
    ("attempt.json", "execution_head", "wrong"),
    ("attempt.json", "authorization_blob", "wrong"),
    ("receipt.json", "status", "FAIL"),
    ("receipt.json", "reusable", True),
    ("receipt.json", "package_count", 28),
    ("receipt.json", "wheel_download_count", 28),
    ("candidate.json", "execution_head", "wrong"),
])
def test_mismatched_state_fails(tmp_path: Path, monkeypatch, file: str, key: str, value) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    path = root / file
    record = json.loads(path.read_text())
    record[key] = value
    _write(path, record)
    with pytest.raises(ValueError):
        evidence.export_evidence()


def test_missing_malformed_and_wheel_drift_fail(tmp_path: Path, monkeypatch) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    wheel = next((root / "wheelhouse").iterdir())
    wheel.write_bytes(wheel.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="CANDIDATE_CONTENT_MISMATCH"):
        evidence.export_evidence()
    (root / "receipt.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        evidence.export_evidence()
    (root / "attempt.json").unlink()
    with pytest.raises(ValueError, match="DURABLE_STATE_MISSING"):
        evidence.export_evidence()


def test_missing_root_and_unsafe_wheel_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(phase_b, "_local_state_base", lambda: tmp_path)
    with pytest.raises(ValueError, match="DURABLE_STATE_MISSING"):
        evidence.export_evidence()
    root, _ = _fixture(tmp_path, monkeypatch)
    wheelhouse = root / "wheelhouse"
    wheel = next(wheelhouse.iterdir())
    original_lstat = os.lstat

    def fake_lstat(path):
        result = original_lstat(path)
        if Path(path) == wheel:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(evidence.os, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="REPARSE_PATH_BLOCKED"):
        evidence.export_evidence()


def test_no_caller_selected_root_or_network_calls(tmp_path: Path, monkeypatch) -> None:
    _fixture(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        evidence.export_evidence(tmp_path / "alternate")
    monkeypatch.setattr(sys, "argv", ["evidence", "--export", "--operation-root", str(tmp_path)])
    assert evidence.main() == 1
    source = (phase_b.ROOT / "scripts/v13_xlsx_phase_b_candidate_evidence.py").read_text()
    assert "import subprocess" not in source and "import requests" not in source
    assert ".mkdir(" not in source and ".write_" not in source and ".unlink(" not in source
