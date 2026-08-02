from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest
from backtest import (
    BlindValidationViolation, LoopValidationPriceSource, Rule,
    loop_validation_summary,
)
from src.benchmark import sha256_file, snapshot_hash


def _price_frame(last: str = "2025-03-31") -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-06", last])
    return pd.DataFrame(
        {"Open": [100.0, 100.0], "High": [101.0, 101.0],
         "Low": [99.0, 99.0], "Close": [100.0, 100.0], "Volume": [1, 1]},
        index=index,
    )


class _RecordingLoader:
    manifest = {"stock_codes": ["1111"]}

    def __init__(self, returned_last: str = "2025-03-31") -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.returned_last = returned_last

    def get_daily_stock_prices(self, code: str, start: str, end: str) -> pd.DataFrame:
        self.calls.append((code, start, end))
        return _price_frame(self.returned_last)


def _validation_metrics() -> dict:
    return {
        "total_profit": 30.0,
        "worst_max_drawdown_percent": 4.0,
        "month_win_rate": 75.0,
        "max_stock_profit_share": 60.0,
        "trade_count": 9,
        "skip_counts": {"SKIPPED_MAX_OPEN_POSITIONS": 2},
        "by_fold": [
            {"fold": 3, "profit": 5.0},
            {"fold": 1, "profit": 10.0},
            {"fold": 2, "profit": 15.0},
        ],
        "selection_status": "PASS",
        "research_test_profit": -999999.0,
        "reference_profit": 999999.0,
        "trades": [{"reference_only": True}],
        "skipped": [],
    }


def _install_fast_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    loader = _RecordingLoader()
    stock = SimpleNamespace(stock_code="1111")
    config = SimpleNamespace(
        raw={
            "backtest_settings": {
                "research_from": backtest.RESEARCH_FROM,
                "research_to": backtest.RESEARCH_TO,
                "final_from": backtest.REFERENCE_FROM,
                "final_to": backtest.REFERENCE_TO,
                "budget": 100,
                "min_research_trades": 1,
                "max_drawdown_percent": 100.0,
                "min_month_win_rate": 0.0,
                "max_stock_profit_share": 100.0,
            },
            "portfolio_settings": {
                "lot_size": 1, "max_position_percent": 100.0,
                "max_open_positions": 1,
            },
        },
        stocks=[stock],
        ai_params=SimpleNamespace(budget=100),
        for_stock=lambda _stock: SimpleNamespace(),
    )
    rule = Rule("1111", 1.0, 2.0, 0.2, 30.0)
    source_class = backtest.LoopValidationPriceSource
    monkeypatch.setattr(backtest, "load_app", lambda **_kwargs: (config, None))
    monkeypatch.setattr(
        backtest, "LoopValidationPriceSource", lambda _path: source_class(loader)
    )
    monkeypatch.setattr(
        backtest, "research_candidate_cache",
        lambda *_args: (pd.DataFrame([{"candidate": 1}]), {}),
    )
    monkeypatch.setattr(backtest, "select_rule_from_diagnostics", lambda *_args: rule)
    monkeypatch.setattr(backtest, "candidate_rules_from_diagnostics", lambda *_args: [rule])
    monkeypatch.setattr(backtest, "assert_validation_folds_non_overlapping", lambda *_args: None)
    monkeypatch.setattr(
        backtest, "coordinate_select_rules",
        lambda *_args, **_kwargs: (
            {"1111": rule}, deepcopy(_validation_metrics()), [], 1, 1,
        ),
    )
    monkeypatch.setattr(
        backtest, "research_test_diagnostics_after_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("research-test diagnostics must not run")
        ),
    )
    monkeypatch.setattr(
        backtest, "_reference_predictions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reference evaluation must not run")
        ),
    )
    monkeypatch.setattr(backtest, "LOOP_RESULT_DIR", tmp_path / "loop-results")
    return loader


def test_loop_price_source_rejects_rows_after_research_period() -> None:
    loader = _RecordingLoader()
    guarded = LoopValidationPriceSource(loader)
    with pytest.raises(BlindValidationViolation, match="limited"):
        guarded.get_daily_stock_prices("1111", "2020-01-01", "2025-04-01")
    assert loader.calls == []

    bad_loader = _RecordingLoader("2025-04-01")
    with pytest.raises(BlindValidationViolation, match="returned a row"):
        LoopValidationPriceSource(bad_loader).get_daily_stock_prices(
            "1111", "2020-01-01", "2025-03-31"
        )


def test_loop_snapshot_reader_does_not_parse_reference_ohlcv(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    ohlcv = root / "ohlcv"
    ohlcv.mkdir(parents=True)
    csv_path = ohlcv / "1111.csv"
    csv_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2025-03-31,100,101,99,100,1\n"
        "2025-04-01,FORBIDDEN,FORBIDDEN,FORBIDDEN,FORBIDDEN,FORBIDDEN\n",
        encoding="utf-8",
    )
    files = {
        "1111": {
            "first_date": "2025-03-31", "last_date": "2025-04-01",
            "rows": 2, "sha256": sha256_file(csv_path),
        }
    }
    manifest = {
        "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
        "files": files, "snapshot_hash": snapshot_hash(files),
        "stock_codes": ["1111"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    frame = LoopValidationPriceSource(root).get_daily_stock_prices(
        "1111", "2020-01-01", "2025-03-31"
    )
    assert list(frame.index) == [pd.Timestamp("2025-03-31")]


def test_loop_mode_never_calls_research_test_or_reference(monkeypatch, tmp_path) -> None:
    loader = _install_fast_validation(monkeypatch, tmp_path)
    result = backtest.run_backtest("loop-validation")
    assert result["profit"] == 30.0
    assert loader.calls == [("1111", "2020-01-01", "2025-03-31")]


def test_reference_values_cannot_change_loop_output() -> None:
    first = _validation_metrics()
    second = deepcopy(first)
    second["research_test_profit"] = 10**12
    second["reference_profit"] = -(10**12)
    second["trades"] = [{"changed_reference_trade": True}]
    assert loop_validation_summary(first) == loop_validation_summary(second)


def test_loop_output_contains_validation_metrics_only(monkeypatch, tmp_path) -> None:
    _install_fast_validation(monkeypatch, tmp_path)
    result = backtest.run_backtest("loop-validation")
    assert set(result) == {
        "profit", "max_drawdown_percent", "fold_profits", "monthly_win_rate",
        "max_stock_profit_share", "trade_count", "skip_counts",
        "config_hash", "config_hash_method",
    }
    output_dir = tmp_path / "loop-results"
    assert [path.name for path in output_dir.iterdir()] == ["summary.json"]
    text = (output_dir / "summary.json").read_text(encoding="utf-8").lower()
    assert all(word not in text for word in ("reference", "test", "baseline"))


def test_two_loop_runs_are_byte_identical(monkeypatch, tmp_path) -> None:
    _install_fast_validation(monkeypatch, tmp_path)
    backtest.run_backtest("loop-validation")
    first = (tmp_path / "loop-results" / "summary.json").read_bytes()
    backtest.run_backtest("loop-validation")
    second = (tmp_path / "loop-results" / "summary.json").read_bytes()
    assert first == second


def test_full_mode_remains_default_and_diagnostic_outputs_exist() -> None:
    assert inspect.signature(backtest.run_backtest).parameters["mode"].default == "full"
    existing = {path.name for path in Path("data/backtest_results").iterdir()}
    assert {"research_diagnostics.csv", "reference_predictions.csv", "reference_trades.csv"} <= existing
