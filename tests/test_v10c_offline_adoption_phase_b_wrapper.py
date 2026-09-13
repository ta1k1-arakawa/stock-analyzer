from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_v10c_offline_adoption_phase_b.ps1"


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_uses_module_execution_and_not_direct_script() -> None:
    source = _source()
    assert "-m scripts.run_v10c_locked_training_cache_provenance_adoption" in source
    assert "scripts\\run_v10c_locked_training_cache_provenance_adoption.py" not in source
    assert "Start-Process" not in source


def test_wrapper_captures_last_exit_code_immediately() -> None:
    lines = [line.strip() for line in _source().splitlines() if line.strip()]
    invoke_index = next(i for i, line in enumerate(lines) if line.startswith("& $canonicalPython -m scripts.run_v10c_locked_training_cache_provenance_adoption"))
    assert lines[invoke_index + 1] == "$processExitCode = $LASTEXITCODE"


def test_wrapper_has_no_network_or_marker_mutation_commands() -> None:
    source = _source().lower()
    for prohibited in ("git fetch", "invoke-webrequest", "curl", "wget", "set-content", "out-file", "new-item", "remove-item"):
        assert prohibited not in source
    assert "authorizationmarker" in source
    assert "V10C_PHASE_B_WRAPPER_FAILURE" in _source()


def test_wrapper_evidence_contract_and_capture_preconditions_are_fixed() -> None:
    source = _source()
    assert "V10C_PHASE_B_WRAPPER_EVIDENCE_V2" in source
    assert "Assert-NewFile" in source
    assert "Assert-NoReparseAncestor $capture" in source
    assert "Assert-External $capture" in source
    assert 'Join-Path $capture "stdout.txt"' in source
    assert 'Join-Path $capture "stderr.txt"' in source
    assert 'Join-Path $capture "wrapper_evidence.json"' in source
    assert "CreateNew" in source
    assert "Flush($true)" in source
    assert '"automatic_retry_performed":false' in source


def test_wrapper_accepts_only_operational_inputs() -> None:
    source = _source()
    for forbidden in ("Provider", "TickerList", "Threshold", "RetrySettings", "Coverage", "QuerySpecification"):
        assert forbidden.lower() not in source.lower()
    for required in ("CandidateRoot", "ImplementationSha", "AuthorizationMarker", "ReceiptPath", "CaptureRoot"):
        assert required in source


def test_wrapper_synthetic_exit_capture_when_environment_is_ready(tmp_path: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    canonical_python = ROOT / ".venv-real-execution" / "Scripts" / "python.exe"
    if shell is None or not canonical_python.is_file():
        pytest.skip("Windows PowerShell and reviewed canonical Python are required")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        pytest.skip("integration probe requires a clean committed checkout")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    candidate = tmp_path / "candidate"
    capture = tmp_path / "capture"
    candidate.mkdir()
    capture.mkdir()
    marker = tmp_path / "invalid-marker.json"
    marker.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WRAPPER),
         "-CandidateRoot", str(candidate), "-ImplementationSha", head,
         "-AuthorizationMarker", str(marker), "-ReceiptPath", str(receipt),
         "-CaptureRoot", str(capture)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 4
    stdout_path = capture / "stdout.txt"
    stderr_path = capture / "stderr.txt"
    evidence_path = capture / "wrapper_evidence.json"
    assert stdout_path.is_file()
    assert stderr_path.is_file()
    assert evidence_path.is_file()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "V10C_PHASE_B_WRAPPER_EVIDENCE_V2"
    assert evidence["process_exit_code"] == 4
    assert evidence["automatic_retry_performed"] is False
    assert evidence["invocation_mode"] == "PYTHON_MODULE"
    assert not (candidate / "locked_raw").exists()
    assert not receipt.exists()
    assert not Path(str(receipt) + ".gate.json").exists()
