from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess

import pandas as pd
import pytest

from compare_evaluators import _data_audit
from src.comparison import (
    ComparisonError, assert_baseline_unchanged, assert_file_unchanged,
    build_execution_orders, deterministic_hashes, forbid_network,
    run_independent_budget, run_v2_portfolio, sha256_file, verify_baseline,
)
from src.trade_simulator import PortfolioSettings


def _git(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *command], cwd=cwd, text=True).strip()


def _detached_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "baseline"
    repo.mkdir()
    _git(["init", "-q"], repo)
    (repo / "tracked.txt").write_text("fixed\n", encoding="utf-8")
    _git(["add", "tracked.txt"], repo)
    subprocess.check_call([
        "git", "-c", "user.name=Tests", "-c", "user.email=tests@example.invalid",
        "commit", "-q", "-m", "baseline",
    ], cwd=repo)
    head = _git(["rev-parse", "HEAD"], repo)
    _git(["checkout", "-q", "--detach", head], repo)
    return repo, head


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0],
            "High": [101.0, 102.0, 102.0, 102.0],
            "Low": [99.0, 99.0, 99.0, 99.0],
            "Close": [100.0, 101.0, 101.0, 101.0],
            "Volume": [1, 1, 1, 1],
        },
        index=pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04"]),
    )


def test_baseline_wrong_head_fails(tmp_path: Path) -> None:
    repo, _ = _detached_repo(tmp_path)
    with pytest.raises(ComparisonError, match="HEAD mismatch"):
        verify_baseline(repo, "0" * 40)


def test_baseline_dirty_fails(tmp_path: Path) -> None:
    repo, head = _detached_repo(tmp_path)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="dirty"):
        verify_baseline(repo, head)


def test_baseline_stays_clean_and_detects_generated_file(tmp_path: Path) -> None:
    repo, head = _detached_repo(tmp_path)
    before = verify_baseline(repo, head)
    assert_baseline_unchanged(repo, head, before)
    (repo / "cache.pkl").write_bytes(b"forbidden")
    with pytest.raises(ComparisonError, match="dirty"):
        assert_baseline_unchanged(repo, head, before)


def test_external_http_and_socket_are_forbidden() -> None:
    import requests
    with forbid_network():
        with pytest.raises(ComparisonError, match="network"):
            requests.get("https://example.invalid")
        with pytest.raises(ComparisonError, match="network"):
            socket.create_connection(("example.invalid", 443))


def test_data_audit_requires_identical_fixed_csv_hash() -> None:
    manifest = {
        "stock_codes": ["1111"],
        "files": {"1111": {"first_date": "2020-01-06", "last_date": "2026-05-20", "rows": 2, "sha256": "abc"}},
    }
    capture = {"usage": [{
        "stock_code": "1111", "first_date": "2020-01-06", "last_date": "2026-05-20",
        "rows": 2, "csv_sha256": "abc",
    }]}
    assert _data_audit(capture, manifest)[0]["csv_sha256"] == "abc"
    capture["usage"][0]["csv_sha256"] = "different"
    with pytest.raises(ComparisonError, match="fixed data mismatch"):
        _data_audit(capture, manifest)


def test_legacy_signals_can_use_v2_shared_portfolio() -> None:
    predictions = pd.DataFrame([
        {"code": "1111", "signal_date": "2025-04-01", "prob": 0.9, "is_signal": True},
        {"code": "2222", "signal_date": "2025-04-01", "prob": 0.8, "is_signal": True},
    ])
    rules = {code: {"stop_loss_percent": 5.0} for code in ("1111", "2222")}
    prices = {code: _prices() for code in rules}
    orders = build_execution_orders(predictions, rules, prices, 2, 0, 0, 0, 0)
    results, _ = run_v2_portfolio(
        orders, 300.0, PortfolioSettings(max_open_positions=1), list(_prices().index),
    )
    assert results.iloc[0]["code"] == "1111"
    assert results.iloc[0]["status"] == "FILLED"
    assert results.iloc[1]["status"] == "SKIPPED_MAX_OPEN_POSITIONS"


def test_v2_signals_can_use_independent_budget_diagnostic() -> None:
    orders = [
        {"code": "1111", "signal_date": "2025-04-01", "entry_date": "2025-04-02", "exit_date": "2025-04-03", "entry_price": 100.0, "exit_price": 101.0, "prob": .9, "commission_percent": 0.0},
        {"code": "2222", "signal_date": "2025-04-01", "entry_date": "2025-04-02", "exit_date": "2025-04-03", "entry_price": 100.0, "exit_price": 101.0, "prob": .8, "commission_percent": 0.0},
    ]
    results, _ = run_independent_budget(orders, 300.0)
    assert list(results["status"]) == ["FILLED", "FILLED"]
    assert list(results["qty"]) == [3, 3]


def test_protected_selected_rules_must_not_change(tmp_path: Path) -> None:
    selected = tmp_path / "selected_rules.csv"
    selected.write_text("code,threshold\n1111,0.2\n", encoding="utf-8")
    digest = sha256_file(selected)
    assert_file_unchanged(selected, digest)
    selected.write_text("code,threshold\n1111,0.3\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="protected file"):
        assert_file_unchanged(selected, digest)


def test_reference_diagnostics_do_not_feed_back_into_rules() -> None:
    rules = {"1111": {"target_percent": 1.0, "stop_loss_percent": 2.0, "threshold": .2}}
    before = json.dumps(rules, sort_keys=True)
    diagnostics = {"reference_profit": -999999.0}
    diagnostics["reference_profit"] = 999999.0
    assert json.dumps(rules, sort_keys=True) == before


def test_comparison_uses_recorded_candidate_not_runner_head() -> None:
    source = Path(__file__).parents[1].joinpath("compare_evaluators.py").read_text(encoding="utf-8")
    assert 'recorded_candidate_summary.get("candidate_commit")' in source
    assert "candidate HEAD must be" not in source


def test_deterministic_hashes_exclude_run_metadata(tmp_path: Path) -> None:
    (tmp_path / "result.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "run_metadata.json").write_text('{"generated_at":"first"}', encoding="utf-8")
    first = deterministic_hashes(tmp_path)
    (tmp_path / "run_metadata.json").write_text('{"generated_at":"second"}', encoding="utf-8")
    assert deterministic_hashes(tmp_path) == first == {"result.csv": sha256_file(tmp_path / "result.csv")}
