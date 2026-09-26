"""Offline tests only; no real JPX provider import or calendar construction."""
from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import v13_public_data_lock as lock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "v13_master_calendar_generate.py"
WRAPPER = ROOT / "scripts" / "run_v13_master_calendar_generation_direct_windows.ps1"
AUTHORIZATION = ROOT / "docs" / "v13" / "V13_MASTER_CALENDAR_POINT_OF_USE_AUTHORIZATION.json"
AUTHORIZATION_BLOB = "2ccb3283fbc212d8f5d942237da79924e7a2ccf5"


def load_runner():
    spec = importlib.util.spec_from_file_location("v13_calendar_generation_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule():
    return pd.DataFrame(
        {"market_close": [pd.Timestamp("2015-01-05T15:00:00+09:00"),
                          pd.Timestamp("2020-10-02T15:00:00+09:00")]},
        index=pd.DatetimeIndex(["2015-01-05", "2020-10-02"]),
    )


def test_import_is_provider_safe_and_ordered_source_gate():
    before = set(sys.modules)
    load_runner()
    assert "pandas_market_calendars" not in set(sys.modules) - before
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    preflight = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_generation_preflight")
    generation = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    assert "validate_calendar_release_artifact" in ast.unparse(preflight)
    statements = generation.body
    validation = next(i for i, node in enumerate(statements) if "validate_generation_preflight" in ast.unparse(node))
    provider = next(i for i, node in enumerate(statements) if "get_calendar" in ast.unparse(node))
    assert validation < provider
    assert ast.unparse(statements[provider]).count('get_calendar(\'JPX\')') == 1
    assert "fallback" not in RUNNER.read_text(encoding="utf-8").lower()


def test_synthetic_exclusive_publication(tmp_path):
    runner = load_runner()
    output = tmp_path / "new-calendar"
    receipt = runner.publish_calendar(schedule(), output, "a" * 40)
    payload = (output / runner.CALENDAR_NAME).read_bytes()
    assert payload == b"2015-01-05\n2020-10-02\n"
    assert receipt["calendar_sha256"] == lock.digest(payload)
    assert receipt["session_count"] == 2
    safe = json.loads((output / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert safe == receipt
    assert "2015-01-05" not in json.dumps(safe)
    assert "2020-10-02" not in json.dumps(safe)
    with pytest.raises(FileExistsError):
        runner.publish_calendar(schedule(), output, "a" * 40)
    assert (output / runner.CALENDAR_NAME).read_bytes() == payload

    failed = tmp_path / "failed-calendar"
    runner._publish_failure(failed, "a" * 40)
    failure = json.loads((failed / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert failure["status"] == "FAIL"
    assert failure["calendar_sha256"] is None
    assert failure["session_count"] is None


def test_wrapper_static_gate_and_protected_bindings():
    script = WRAPPER.read_text(encoding="utf-8")
    assert '$ErrorActionPreference = "Stop"' in script
    assert ".venv-real-execution\\Scripts\\python.exe" in script
    assert "scripts/check_current_protected_environment.py" in script
    assert "scripts/v13_master_calendar_generate.py" in script
    assert "POINT_OF_USE_GATE_NOT_DEFINED" not in script
    assert script.index("--preflight-only") < script.index("Write-NewDurableJson $gatePath")
    assert script.index("Assert-DirectoryWritable $gateDir") < script.index("Write-NewDurableJson $gatePath")
    assert script.index("Write-NewDurableJson $gatePath") < script.index('"scripts/v13_master_calendar_generate.py" --official-wheel $wheelFile.FullName --output-root $outputFull --implementation-sha $ExpectedReviewedHead 2>&1')
    assert "[System.IO.FileMode]::CreateNew" in script
    assert "GENERATION_GATE_ALREADY_CONSUMED" in script
    assert "GENERATION_RESULT_ALREADY_EXISTS" in script
    assert "POST_GATE_FAILURE_NO_RETRY" in script
    blob_check = 'Require-GitValue @("rev-parse", "HEAD:docs/v13/V13_MASTER_CALENDAR_POINT_OF_USE_AUTHORIZATION.json") "2ccb3283fbc212d8f5d942237da79924e7a2ccf5"'
    assert blob_check in script
    assert script.index(blob_check) < script.index("Write-NewDurableJson $gatePath")
    assert "human_approval_evidence =" not in script
    approval = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert approval["reviewed_implementation_sha"] == "983c3c6da51865a46a8bb9361bb13604a5b0c112"
    assert approval["authorization_scope"] == "MASTER_CALENDAR_GENERATION_ONLY"
    assert all(value is False for key, value in approval.items() if key.endswith("_authorized"))
    for forbidden in ("Install-Module", "pip install", "Invoke-WebRequest", "Invoke-RestMethod", "git pull ", "git merge ", "git rebase ", "git cherry-pick ", "--force"):
        assert forbidden.lower() not in script.lower()


def test_wrapper_ascii_and_authorization_artifact_unchanged():
    source = WRAPPER.read_bytes()
    assert all(byte < 128 for byte in source)
    assert "human_approval_evidence" not in source.decode("ascii").split("$required = @{", 1)[1].split("}", 1)[0]
    for args in (
        ["rev-parse", "HEAD:docs/v13/V13_MASTER_CALENDAR_POINT_OF_USE_AUTHORIZATION.json"],
        ["hash-object", "--", str(AUTHORIZATION)],
    ):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == AUTHORIZATION_BLOB
    assert "Issue #90 の V13 master-calendar 1回生成を承認します" in AUTHORIZATION.read_text(encoding="utf-8")


def test_windows_powershell_51_parse_only():
    executable = shutil.which("powershell.exe")
    if executable is None:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    path = str(WRAPPER).replace("'", "''")
    command = (
        "if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1) { exit 2 }; "
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | Out-String | Write-Error; exit 1 }"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
