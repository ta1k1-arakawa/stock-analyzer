from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from scripts import run_v7_forward_operations as cli
from src import v7_forward_operations as operations

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v7_forward_operations.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


@pytest.fixture(scope="module")
def synthetic_result():
    """Exercises the REAL candidate generator end to end (no monkeypatch)."""
    return cli.run_synthetic_operations_test()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_has_exactly_one_authorized_option():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }
    assert options == {"--synthetic-operations-test"}


def test_cli_has_no_real_path_network_or_activation_option():
    text = SCRIPT.read_text(encoding="utf-8")
    for flag in (
        "--study-root", "--durable-root", "--activate", "--network",
        "--real", "--output-root", "--activation-manifest",
    ):
        assert flag not in text


def test_cli_performs_no_urlopen():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert calls == []


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run([PYTHON, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--real-study-root"], cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    assert result.returncode != 0


@pytest.mark.slow
def test_cli_subprocess_exit_zero_and_reports_result():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-operations-test"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["network_requests"] == 0
    assert payload["actual_activation_created"] is False


def test_cli_leaves_no_manifest_in_repository():
    assert not (ROOT / "activation_manifest.json").exists()
    assert not list((ROOT / "data").glob("activation_manifest*.json"))
    assert not list(ROOT.glob("durable-study-root*"))


# ---------------------------------------------------------------------------
# Synthetic result shape (real candidate generator, exercised in-process)
# ---------------------------------------------------------------------------


REQUIRED_RESULT_FIELDS = {
    "status", "mode", "activation_manifest_verified", "engine_day",
    "acquisition_verified", "processing_verified", "persistence_verified",
    "already_committed", "network_requests", "actual_activation_created",
    "real_forward_processing", "profit_metrics_exposed",
}


def test_synthetic_result_has_required_fields(synthetic_result):
    assert REQUIRED_RESULT_FIELDS <= set(synthetic_result)


def test_synthetic_result_values(synthetic_result):
    assert synthetic_result["status"] == "PASS"
    assert synthetic_result["mode"] == "STATIC_SYNTHETIC_ONLY"
    assert synthetic_result["activation_manifest_verified"] is True
    assert synthetic_result["acquisition_verified"] is True
    assert synthetic_result["processing_verified"] is True
    assert synthetic_result["persistence_verified"] is True
    assert synthetic_result["already_committed"] is False
    assert synthetic_result["network_requests"] == 0
    assert synthetic_result["actual_activation_created"] is False
    assert synthetic_result["real_forward_processing"] == 0
    assert synthetic_result["profit_metrics_exposed"] is False
    assert synthetic_result["restart_equivalence"] is True


def test_synthetic_result_engine_day_is_second_day(synthetic_result):
    from src.v7_jpx_calendar import load_calendar_snapshot, next_jpx_trading_day
    from src.v7_activation_manifest import expected_activation_boundary

    snapshot = load_calendar_snapshot(cli.CALENDAR_PATH)
    boundary = expected_activation_boundary(snapshot, cli.AUTHORIZATION_UTC)
    assert synthetic_result["engine_day"] == next_jpx_trading_day(snapshot, boundary)


def test_synthetic_result_exposes_no_profit_tokens(synthetic_result):
    keys = {key.lower() for key in synthetic_result}
    for token in ("realized", "drawdown", "profit_factor", "win_rate", "pnl", "equity_value"):
        offending = [key for key in keys if token in key]
        assert offending == [], token
    for token in ("profit",):
        offending = [key for key in keys if token in key and key != "profit_metrics_exposed"]
        assert offending == [], token


# ---------------------------------------------------------------------------
# Static safety: src/v7_forward_operations.py
# ---------------------------------------------------------------------------


def test_module_source_has_no_network_imports():
    text = Path(operations.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines), token


def test_module_source_has_no_real_order_or_activation_creation_tokens():
    text = Path(operations.__file__).read_text(encoding="utf-8")
    for token in ("place_order", "submit_order", "real_order", "build_activation_manifest_candidate("):
        assert token not in text, token


def _executable_identifiers(path: Path) -> set[str]:
    """Names, attributes and literal strings, ignoring comments and docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings:
            identifiers.add(node.value)
    return {value.lower() for value in identifiers}


def test_module_source_has_no_profit_or_evaluation_tokens():
    identifiers = _executable_identifiers(Path(operations.__file__))
    for token in ("realized_net_profit", "profit_factor", "win_rate", "drawdown", "formal_evaluation"):
        offending = [value for value in identifiers if token in value]
        assert offending == [], token


def test_module_never_calls_write_activation_manifest_once():
    tree = ast.parse(Path(operations.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "write_activation_manifest_once" not in called


def test_module_reuses_accepted_primitives_without_reimplementation():
    text = Path(operations.__file__).read_text(encoding="utf-8")
    for primitive in (
        "validate_activation_manifest_candidate", "read_activation_manifest",
        "validate_output_root", "validate_acquisition_window",
        "acquire_daily_bundle", "verify_daily_acquisition_bundle",
        "process_forward_day", "verify_processed_forward_day",
        "is_jpx_trading_day", "load_calendar_snapshot",
    ):
        assert primitive in text, primitive


def test_module_only_write_function_is_none() -> None:
    """This module must never itself write study artifacts to disk; all
    durable writes happen inside the already-accepted lower-layer modules
    it orchestrates (acquire_daily_bundle / ForwardStudyStore.write_day)."""
    tree = ast.parse(Path(operations.__file__).read_text(encoding="utf-8"))
    writers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {
            "write_bytes", "write_text", "mkdir", "unlink", "fsync", "replace",
        }:
            writers.add(func.attr)
        elif isinstance(func, ast.Name) and func.id == "open":
            writers.add("open")
    assert writers == set()


def test_frozen_commit_constants_match_lower_layers():
    from src.v7_daily_acquisition import CALENDAR_COMMIT, COLLECTOR_COMMIT

    assert operations.EXPECTED_CALENDAR_COMMIT == CALENDAR_COMMIT
    assert operations.EXPECTED_COLLECTOR_COMMIT == COLLECTOR_COMMIT
