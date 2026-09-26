"""Offline tests only; no real JPX provider import or calendar construction."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import v13_public_data_lock as lock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "v13_master_calendar_generate.py"
WRAPPER = ROOT / "scripts" / "run_v13_master_calendar_generation_direct_windows.ps1"


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
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    statements = function.body
    validation = next(i for i, node in enumerate(statements) if "validate_calendar_release_artifact" in ast.unparse(node))
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
    assert 'throw "POINT_OF_USE_GATE_NOT_DEFINED"' in script
    assert script.index('throw "POINT_OF_USE_GATE_NOT_DEFINED"') < script.index('"scripts/v13_master_calendar_generate.py"')
    for forbidden in ("Install-Module", "pip install", "Invoke-WebRequest", "Invoke-RestMethod", "git pull", "git merge", "git rebase", "git cherry-pick", "--force"):
        assert forbidden.lower() not in script.lower()
