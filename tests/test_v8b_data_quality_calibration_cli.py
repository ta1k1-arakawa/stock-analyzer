from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_v8b_data_quality_calibration as cli

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_v8b_data_quality_calibration.py"
PYTHON = sys.executable


def _cli_options() -> set[str]:
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
    return options


def test_cli_has_exactly_one_authorized_option():
    assert _cli_options() == {"--static-check"}


def test_cli_has_no_real_data_or_network_option():
    forbidden = {"--cache-path", "--input-dir", "--v5b-cache", "--execute-real"}
    assert _cli_options().isdisjoint(forbidden)


def test_cli_script_source_has_no_network_or_real_cache_strings():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "urllib",
        "requests",
        "yfinance",
        "query1.finance.yahoo.com",
        "v5-b-evaluation-cache-retry1",
        "--cache",
        "--input-dir",
        "--execute-real",
    ]
    for token in forbidden:
        assert token not in source, f"forbidden token found: {token}"


def test_cli_main_returns_zero_and_prints_exact_success_message(capsys):
    exit_code = cli.main(["--static-check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "V8B_CALIBRATION_IMPLEMENTATION_STATIC_PASS"


def test_cli_subprocess_static_check_passes():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--static-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "V8B_CALIBRATION_IMPLEMENTATION_STATIC_PASS"


def test_cli_missing_flag_fails_with_exit_code_2():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2


def test_cli_rejects_unknown_real_data_flag():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--static-check", "--cache-path", "C:\\taiki\\hobbies\\v5-b-evaluation-cache-retry1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2


def test_cli_blocked_output_contains_only_generic_reason_code(tmp_path, monkeypatch):
    fake_repo = tmp_path
    (fake_repo / "src").mkdir(parents=True)
    (fake_repo / "scripts").mkdir(parents=True)
    (fake_repo / "V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md").write_text("not the real plan", encoding="utf-8")
    (fake_repo / "V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json").write_text("{}", encoding="utf-8")
    (fake_repo / "src" / "v7_yahoo_collector.py").write_text("", encoding="utf-8")
    fake_script = fake_repo / "scripts" / "check_v8b_data_quality_calibration.py"
    fake_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    (fake_repo / "src" / "v8b_data_quality_calibration.py").write_bytes(
        (ROOT / "src" / "v8b_data_quality_calibration.py").read_bytes()
    )
    completed = subprocess.run(
        [PYTHON, str(fake_script), "--static-check"],
        cwd=fake_repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    combined = (completed.stdout + completed.stderr).strip()
    assert combined == "CALIBRATION_PLAN_BLOB_MISMATCH"
