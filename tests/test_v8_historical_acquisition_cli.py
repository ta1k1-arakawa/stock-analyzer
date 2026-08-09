from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from scripts import acquire_v8_historical as cli
from src import v8_historical_acquisition as acquisition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_v8_historical.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


@pytest.fixture(scope="module")
def synthetic_result():
    return cli.run_synthetic_acquisition_test()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


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


def test_cli_has_exactly_authorized_options():
    assert _cli_options() == {
        "--synthetic-test", "--production-acquire", "--block",
        "--partition-manifest", "--output-root", "--confirmation",
    }


def test_cli_has_no_ticker_or_bypass_override():
    """Checks actual add_argument() calls via AST, not raw source text --
    this file's own docstring names the forbidden flags as examples of what
    must NOT exist, so a substring search over the whole source would false-
    positive on its own documentation."""
    options = _cli_options()
    for flag in (
        "--tickers", "--ticker-file", "--ticker-list", "--partition-manifest-sha",
        "--implementation-git-commit", "--yahoo-host", "--request-start", "--request-end",
        "--retry-count", "--force", "--override", "--allow-t3", "--open-t2", "--unseal",
        "--authorize-research-access", "--skip-source-hash", "--ignore-parity", "--network", "--all",
        "--trusted-manifest-sha", "--trusted-registry", "--trust-anchor", "--authorization-file", "--expected-sha",
    ):
        assert flag not in options


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run([PYTHON, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--partition", "T3"], cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0


def test_cli_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.main(["--synthetic-test", "--production-acquire"])


@pytest.mark.parametrize("args", (
    ["--production-acquire"],
    ["--production-acquire", "--block", "T1"],
    ["--production-acquire", "--block", "T1", "--partition-manifest", "C:/missing.json"],
    ["--production-acquire", "--block", "T1", "--partition-manifest", "C:/missing.json", "--output-root", "C:/private"],
))
def test_production_cli_requires_all_inputs_before_runner(monkeypatch, args):
    monkeypatch.setattr(cli, "run_production_acquisition", lambda **_: pytest.fail("runner reached"))
    with pytest.raises(SystemExit):
        cli.main(args)


def test_production_cli_wrong_confirmation_blocks_before_runner(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_production_acquisition", lambda **_: pytest.fail("runner reached"))
    assert cli.main(["--production-acquire", "--block", "T1", "--partition-manifest", "C:/x", "--output-root", "C:/y", "--confirmation", "WRONG"]) == 2
    assert json.loads(capsys.readouterr().out) == {"reason": "CONFIRMATION_MISMATCH", "status": "BLOCKED"}


@pytest.mark.parametrize("block", ("T3", "UNKNOWN"))
def test_production_cli_invalid_block_blocks_before_runner(monkeypatch, capsys, block):
    monkeypatch.setattr(cli, "run_production_acquisition", lambda **_: pytest.fail("runner reached"))
    assert cli.main(["--production-acquire", "--block", block, "--partition-manifest", "C:/x", "--output-root", "C:/y", "--confirmation", "anything"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_production_runner_signature_has_only_required_inputs():
    assert tuple(inspect.signature(cli.run_production_acquisition).parameters) == (
        "block", "partition_manifest_path", "output_root"
    )


def test_production_runner_delegates_only_to_hardened_public_api(monkeypatch, tmp_path):
    observed: dict[str, object] = {}

    def fake_public_api(**kwargs):
        observed.update(kwargs)
        return {
            "block": "T2", "role": "SEALED_HOLDOUT", "sealed": True,
            "partition_manifest_sha256": "a" * 64, "implementation_git_commit": "b" * 40,
        }

    monkeypatch.setattr(cli, "acquire_historical_block_bundle", fake_public_api)
    result = cli.run_production_acquisition(
        block="T2", partition_manifest_path=tmp_path / "partition.json", output_root=tmp_path / "private"
    )
    assert observed == {
        "output_root": tmp_path / "private", "block": "T2", "partition_manifest_path": tmp_path / "partition.json"
    }
    assert result["role"] == "SEALED_HOLDOUT" and result["sealed"] is True


def test_cli_valid_production_invocation_passes_only_required_runner_inputs(monkeypatch, capsys):
    observed: dict[str, object] = {}

    def fake_runner(**kwargs):
        observed.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(cli, "run_production_acquisition", fake_runner)
    assert cli.main([
        "--production-acquire", "--block", "T1", "--partition-manifest", "C:/partition.json",
        "--output-root", "C:/private", "--confirmation", "V8_PRODUCTION_ACQUIRE_T1",
    ]) == 0
    assert observed == {
        "block": "T1", "partition_manifest_path": Path("C:/partition.json"), "output_root": Path("C:/private")
    }
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


@pytest.mark.slow
def test_cli_subprocess_exit_zero_and_reports_result():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-test"], cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["network_requests"] == 0
    assert payload["t3_acquisition_blocked"] is True
    assert payload["t2_opened"] is False


def test_cli_leaves_no_acquisition_bundle_in_repository():
    assert not list(ROOT.glob("**/acquisition_manifest.json"))
    assert not (ROOT / "acquisitions").exists()


# ---------------------------------------------------------------------------
# Synthetic result shape (in-process)
# ---------------------------------------------------------------------------


REQUIRED_RESULT_FIELDS = {
    "status", "mode", "request_start", "request_end_exclusive",
    "t1_status", "t1_ticker_count", "t1_validation_access_count",
    "t2_status", "t2_sealed", "t2_research_access_authorized", "t2_opened",
    "t3_acquisition_blocked", "guard_blocks_all_research_operations",
    "retry_count", "http_429_count", "network_requests", "data_acquired",
    "real_acquisition_created", "backtests", "profit_calculated", "models_fitted",
}


def test_synthetic_result_has_required_fields(synthetic_result):
    assert REQUIRED_RESULT_FIELDS <= set(synthetic_result)


def test_synthetic_result_values(synthetic_result):
    assert synthetic_result["status"] == "PASS"
    assert synthetic_result["mode"] == "STATIC_SYNTHETIC_ONLY"
    assert synthetic_result["request_start"] == "2016-04-01"
    assert synthetic_result["request_end_exclusive"] == "2026-01-01"
    assert synthetic_result["t1_status"] == "RAW_ACQUIRED_NOT_OPENED"
    assert synthetic_result["t1_validation_access_count"] == 0
    assert synthetic_result["t2_status"] == "RAW_ACQUIRED_SEALED"
    assert synthetic_result["t2_sealed"] is True
    assert synthetic_result["t2_research_access_authorized"] is False
    assert synthetic_result["t2_opened"] is False
    assert synthetic_result["t3_acquisition_blocked"] is True
    assert synthetic_result["guard_blocks_all_research_operations"] is True
    assert synthetic_result["retry_count"] == 0
    assert synthetic_result["http_429_count"] == 0
    assert synthetic_result["network_requests"] == 0
    assert synthetic_result["data_acquired"] == 0
    assert synthetic_result["real_acquisition_created"] is False
    assert synthetic_result["backtests"] == 0
    assert synthetic_result["profit_calculated"] == 0
    assert synthetic_result["models_fitted"] == 0


def test_synthetic_result_exposes_no_profit_tokens(synthetic_result):
    keys = {key.lower() for key in synthetic_result}
    for token in ("realized", "drawdown", "profit_factor", "win_rate", "pnl", "equity_value"):
        offending = [key for key in keys if token in key]
        assert offending == [], token


# ---------------------------------------------------------------------------
# Static safety: src/v8_historical_acquisition.py and this CLI
# ---------------------------------------------------------------------------


def test_module_source_has_no_network_imports_beyond_v7_collector():
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "aio" + "http", "http" + "x", "y" + "finance"):
        assert not any(token in line for line in import_lines), token


def test_cli_source_has_no_urlopen_call():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert calls == []


def test_module_never_touches_v7_activation_or_study_root():
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    for token in ("v7_forward", "v7_activation_manifest", "v7_daily_acquisition", "v7_seed_acquisition", "durable_study_root"):
        assert token not in text


def test_t3_prohibited_constant_matches_design():
    assert "T3" not in acquisition.ALLOWED_ACQUISITION_BLOCKS
    assert acquisition.ALLOWED_ACQUISITION_BLOCKS == ("T1", "T2")


def test_frozen_request_window_matches_p_hist():
    assert acquisition.REQUEST_START == "2016-04-01"
    assert acquisition.REQUEST_END_EXCLUSIVE == "2026-01-01"


def test_min_request_interval_matches_v7_precedent():
    assert acquisition.MIN_REQUEST_INTERVAL_SECONDS == 2.0
    assert acquisition.RETRY_COUNT == 0
