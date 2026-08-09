from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import acquire_v8_historical as cli
from src import v8_historical_acquisition as acquisition
from src import v8_partition as partition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_v8_historical.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def write_trusted_partition_anchor(path: Path, *, manifest: dict | None = None) -> None:
    authorized = manifest is not None
    path.write_bytes(partition.canonical_json_bytes({
        "schema_version": acquisition.TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "authorization_status": "AUTHORIZED" if authorized else "NOT_AUTHORIZED",
        "authorized_partition_manifest_sha256": manifest["manifest_sha256"] if manifest else None,
        "authorized_partition_implementation_git_commit": (
            manifest["partition_implementation_git_commit"] if manifest else None
        ),
        "authorization_note": "test-only anchor",
    }))


@pytest.fixture(autouse=True)
def test_trusted_partition_anchor(monkeypatch, tmp_path):
    anchor_path = tmp_path / "V8_TRUSTED_PARTITION.json"
    monkeypatch.setattr(acquisition, "TRUSTED_PARTITION_ANCHOR_PATH", anchor_path)
    write_trusted_partition_anchor(anchor_path)
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


def _tickers(start: int) -> list[str]:
    return [f"{code:04d}" for code in range(start, start + 300)]


def write_partition_manifest(path: Path, *, mutation=None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = {"T0": _tickers(4000), "T1": _tickers(1000), "T2": _tickers(2000),
              "T3": _tickers(3000), "T_spare": _tickers(5000)}
    manifest = {
        "schema_version": partition.SCHEMA_VERSION, "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT, "created_utc": "2026-08-09T00:00:00Z",
        "partition_implementation_git_commit": "a" * 40,
        "source_url": "https://example.invalid/jpx", "source_host": "example.invalid",
        "source_acquisition_utc": "2026-08-09T00:00:00Z", "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 0, "expected_v4_source_raw_sha256": "0" * 64,
        "source_reproduction_status": "SYNTHETIC", "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[key] for key in blocks), [])),
        "deterministic_ordering_rule": partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": partition.ticker_list_sha256(blocks["T_spare"]),
        "legacy_exclude_list": [], "legacy_exclude_list_sha256": partition.ticker_list_sha256([]),
        "block_sizes": {key: len(value) for key, value in blocks.items()}, "block_assignments": blocks,
        "p_hist_start": partition.P_HIST_START, "p_hist_end": partition.P_HIST_END,
        "t1_role": partition.T1_ROLE, "t2_role": partition.T2_ROLE, "t3_role": partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    if mutation:
        mutation(manifest)
    manifest["manifest_sha256"] = partition.canonical_sha256(manifest)
    path.write_bytes(partition.canonical_json_bytes(manifest))
    write_trusted_partition_anchor(acquisition.TRUSTED_PARTITION_ANCHOR_PATH, manifest=manifest)
    return manifest


def production_kwargs(tmp_path, block="T1", *, mutation=None):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path, mutation=mutation)
    opener = cli.FakeYahooOpener(base_price=1000.0)
    return manifest, opener, {
        "block": block, "partition_manifest_path": partition_path, "output_root": tmp_path / "private",
        "opener": opener, "clock": lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
        "implementation_git_commit_resolver": lambda _: "a" * 40,
        "monotonic_clock": lambda: 0.0, "sleep_fn": lambda _: None,
    }


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


@pytest.mark.parametrize("block, role, sealed", (("T1", "VALIDATION", False), ("T2", "SEALED_HOLDOUT", True)))
def test_production_runner_reaches_only_fake_transport(tmp_path, block, role, sealed):
    manifest, opener, kwargs = production_kwargs(tmp_path, block)
    result = cli.run_production_acquisition(**kwargs)
    assert opener.calls == manifest["block_assignments"][block]
    assert result["role"] == role and result["sealed"] is sealed
    assert result["partition_manifest_sha256"] == manifest["manifest_sha256"]


def test_production_runner_unauthorized_trust_anchor_blocks_before_network(tmp_path):
    _, opener, kwargs = production_kwargs(tmp_path)
    write_trusted_partition_anchor(acquisition.TRUSTED_PARTITION_ANCHOR_PATH)
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        cli.run_production_acquisition(**kwargs)
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"
    assert opener.calls == []


@pytest.mark.parametrize("mutation", (
    lambda value: value.__setitem__("study_name", "WRONG"),
    lambda value: value.__setitem__("t1_ticker_list_sha256", "0" * 64),
))
def test_production_runner_binding_failure_blocks_before_network(tmp_path, mutation):
    _, opener, kwargs = production_kwargs(tmp_path, mutation=mutation)
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        cli.run_production_acquisition(**kwargs)
    assert opener.calls == []


def test_production_runner_storage_and_provenance_fail_closed(tmp_path):
    _, opener, kwargs = production_kwargs(tmp_path)
    kwargs["output_root"] = Path("relative-private")
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        cli.run_production_acquisition(**kwargs)
    assert opener.calls == []
    _, opener, kwargs = production_kwargs(tmp_path / "git-dirty")

    def dirty_git(_):
        raise acquisition.V8HistoricalAcquisitionBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    kwargs["implementation_git_commit_resolver"] = dirty_git
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        cli.run_production_acquisition(**kwargs)
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"
    assert opener.calls == []
    _, opener, kwargs = production_kwargs(tmp_path / "inside-repository")
    kwargs["output_root"] = ROOT / "would-be-private-v8-storage"
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        cli.run_production_acquisition(**kwargs)
    assert opener.calls == []
    _, opener, kwargs = production_kwargs(tmp_path / "provenance")
    kwargs["implementation_git_commit_resolver"] = lambda _: "invalid"
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        cli.run_production_acquisition(**kwargs)
    assert opener.calls == []


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
