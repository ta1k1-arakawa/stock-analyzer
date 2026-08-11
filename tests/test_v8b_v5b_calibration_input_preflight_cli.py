from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import preflight_v8b_v5b_calibration_input as cli

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_v8b_v5b_calibration_input.py"
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


def test_cli_has_exactly_the_authorized_options():
    assert _cli_options() == {"--static-check", "--confirm", "--implementation-git-commit"}


def test_cli_exposes_no_arbitrary_cache_root_or_dataset_argument():
    forbidden = {
        "--cache-path",
        "--cache-root",
        "--input-dir",
        "--manifest-path",
        "--v5b-cache",
        "--dataset",
        "--execute-real",
    }
    assert _cli_options().isdisjoint(forbidden)


def test_cli_script_source_has_no_network_or_arbitrary_path_strings():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "urllib",
        "requests",
        "yfinance",
        "query1.finance.yahoo.com",
        "--cache",
        "--input-dir",
        "--manifest-path",
        "--execute-real",
        "spec_from_file_location",
        "exec_module",
    ]
    for token in forbidden:
        assert token not in source, f"forbidden token found: {token}"


def test_cli_static_check_prints_exact_success_message(capsys):
    exit_code = cli.main(["--static-check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_STATIC_PASS"


def test_cli_static_check_subprocess_touches_no_real_cache_and_exits_zero():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--static-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_STATIC_PASS"
    assert completed.stderr == ""


def test_cli_static_check_rejects_extra_arguments():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--static-check", "--implementation-git-commit", "a" * 40],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2


def test_cli_requires_a_mode():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2


def test_cli_static_and_confirm_are_mutually_exclusive():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--static-check", "--confirm", "X"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2


def test_cli_confirm_without_commit_is_rejected(capsys):
    exit_code = cli.main(["--confirm", "WRONG_TOKEN"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "IMPLEMENTATION_COMMIT_REQUIRED" in captured.err


def test_cli_wrong_confirmation_blocks_without_touching_real_cache(capsys, monkeypatch):
    # Wrong confirmation must be rejected before any cache access, so this
    # is safe to run for real (subprocess-free) even though the real fixed
    # cache root does not exist in this environment.
    exit_code = cli.main(["--confirm", "NOT_THE_TOKEN", "--implementation-git-commit", "a" * 40])
    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "BLOCK"
    assert payload["detail_reason"] == "PREFLIGHT_GATE_CONFIRMATION_REQUIRED"


def test_cli_wrong_confirmation_subprocess_blocks():
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--confirm", "NOT_THE_TOKEN", "--implementation-git-commit", "a" * 40],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BLOCK"
    assert payload["detail_reason"] == "PREFLIGHT_GATE_CONFIRMATION_REQUIRED"


def test_cli_correct_confirmation_against_synthetic_fixture_via_monkeypatched_root(monkeypatch, tmp_path, capsys):
    import hashlib

    from src import v8b_data_quality_calibration as calib
    from src import v8b_v5b_calibration_input_preflight as preflight

    root = tmp_path / "synthetic_cache"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    payload_records = []
    for index in range(300):
        ticker = f"T{index:04d}"
        content = json.dumps({"marker": index}).encode("utf-8")
        (raw_dir / f"{ticker}.json").write_bytes(content)
        payload_records.append(
            {
                "ticker": ticker,
                "relative_path": f"raw/{ticker}.json",
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    manifest = {
        "schema_version": 2,
        "complete": True,
        "usable_for_evaluation": True,
        "attempted_ticker_count": 300,
        "success_count": 300,
        "failed_count": 0,
        "ticker_count": 300,
        "failed_tickers": [],
        "circuit_breaker_triggered": False,
        "request_start": "2019-01-01",
        "request_end": "2026-01-31",
        "payloads": payload_records,
    }
    manifest["payload_hash_list_sha256"] = calib._recompute_payload_hash_list_sha256(payload_records)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)

    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", manifest["payload_hash_list_sha256"])
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    monkeypatch.setattr(cli, "run_production_v5b_calibration_input_preflight", preflight.run_production_v5b_calibration_input_preflight)

    exit_code = cli.main(
        ["--confirm", preflight.PREFLIGHT_GATE_CONFIRMATION, "--implementation-git-commit", "a" * 40]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "PASS"


def test_cli_blocked_output_never_leaks_ticker_or_path(capsys):
    cli.main(["--confirm", "WRONG", "--implementation-git-commit", "a" * 40])
    captured = capsys.readouterr()
    assert "raw/" not in captured.out
    for index in range(300):
        assert f"T{index:04d}" not in captured.out
