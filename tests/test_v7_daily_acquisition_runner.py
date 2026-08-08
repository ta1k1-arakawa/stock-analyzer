from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from scripts import check_v7_daily_acquisition as cli
from src import v7_daily_acquisition as daily

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_v7_daily_acquisition.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def test_cli_has_exactly_one_authorized_option():
    parser = cli.main.__globals__["argparse"].ArgumentParser()
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
    assert options == {"--synthetic-acquisition-test"}


def test_cli_module_has_no_network_option():
    text = SCRIPT.read_text(encoding="utf-8")
    for forbidden_flag in ("--real", "--network", "--activation", "--output-root", "--seed"):
        assert forbidden_flag not in text


def test_cli_module_has_no_real_yahoo_or_jpx_urlopen_at_import():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    urlopen_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert urlopen_calls == []


def test_run_synthetic_acquisition_test_in_process():
    result = cli.run_synthetic_acquisition_test()
    assert result["status"] == "PASS"
    assert result["mode"] == "STATIC_SYNTHETIC_ONLY"
    assert result["ticker_count"] == 300
    assert result["request_count"] == 300
    assert result["retry_count"] == 0
    assert result["bundle_verification"] == "PASS"
    assert result["atomic_publish"] is True
    assert result["network_requests"] == 0
    assert result["candidate_generation"] == 0
    assert result["portfolio_processing"] == 0
    assert result["activation_created"] is False


def test_cli_subprocess_synthetic_restart_test_exit_zero():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-acquisition-test"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["ticker_count"] == 300
    assert payload["request_count"] == 300


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run(
        [PYTHON, str(SCRIPT)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--real-network"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0


def test_synthetic_fixture_uses_real_universe_file():
    tickers = cli._universe_tickers()
    assert len(tickers) == 300


def test_synthetic_calendar_snapshot_marks_engine_day_as_trading_day():
    from src.v7_jpx_calendar import is_jpx_trading_day, load_calendar_snapshot

    snapshot = load_calendar_snapshot(cli._synthetic_calendar_snapshot())
    assert is_jpx_trading_day(snapshot, cli.ENGINE_DAY) is True


def test_module_constants_bind_fixed_lineage():
    assert daily.CALENDAR_COMMIT == "03ce048b0eedca632f79ad925a627cb9e967d78d"
    assert daily.CALENDAR_DEFINITION_VERSION == "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1"
    assert daily.COLLECTOR_COMMIT == "4ca41c53895e75910ae65809fea6018868929afa"
    assert daily.DATA_SOURCE == "Yahoo Chart"
    assert daily.DATA_SOURCE_HOST == "query1.finance.yahoo.com"
    assert daily.SCHEMA_VERSION == "V7_DAILY_ACQUISITION_V1"
    assert daily.MODE == "FORWARD_DAILY_ACQUISITION"


def test_module_has_no_network_import_tokens():
    text = Path(daily.__file__).read_text(encoding="utf-8")
    lowered = text.lower()
    import_lines = [line.strip() for line in lowered.splitlines() if line.strip().startswith(("import ", "from "))]
    network_tokens = ("re" + "quests", "http" + "x", "aio" + "http", "so" + "cket")
    assert not any(any(token in line for token in network_tokens) for line in import_lines)


def test_module_source_has_no_activation_or_real_order_tokens():
    text = Path(daily.__file__).read_text(encoding="utf-8")
    for token in ("activation_authorization", "activation_boundary_first", "place_order", "real_order"):
        assert token not in text


def test_min_request_interval_is_two_seconds():
    assert daily.MIN_REQUEST_INTERVAL_SECONDS == 2.0


def test_expected_ticker_count_is_300():
    assert daily.EXPECTED_TICKER_COUNT == 300
