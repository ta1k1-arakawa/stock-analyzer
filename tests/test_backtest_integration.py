from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from backtest import eligible_training_rows, select_rule_from_diagnostics
from src.trade_simulator import PortfolioSettings, simulate_portfolio


def test_backtest_has_no_yahoo_fetcher_fallback():
    source = Path("backtest.py").read_text(encoding="utf-8")
    assert "YFinanceFetcher" not in source
    assert "FixedOHLCVLoader" in source


def test_rule_selection_uses_validation_only():
    rows = pd.DataFrame(
        [
            {"TargetPercent": 1.0, "StopLossPercent": 2.0, "Threshold": 0.2,
             "ValidationProfit": 100.0, "ValidationTrades": 10,
             "TestProfitDiagnosticOnly": -10_000.0, "ReferenceProfit": -99_999.0},
            {"TargetPercent": 2.0, "StopLossPercent": 5.0, "Threshold": 0.5,
             "ValidationProfit": 50.0, "ValidationTrades": 10,
             "TestProfitDiagnosticOnly": 10_000.0, "ReferenceProfit": 99_999.0},
        ]
    )
    selected = select_rule_from_diagnostics("1234", rows, min_trades=1)
    assert selected.target_percent == 1.0
    assert selected.validation_score == 100.0


def test_training_cutoff_excludes_unconfirmed_and_same_day_labels():
    frame = pd.DataFrame(
        {
            "Feature": [1.0, 2.0, 3.0],
            "Target": [1, 0, 1],
            "LabelConfirmedDate": pd.to_datetime(["2025-01-02", "2025-01-05", "2025-01-06"]),
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    )
    eligible = eligible_training_rows(frame, "2025-01-05", ["Feature"])
    assert list(eligible.index) == [pd.Timestamp("2025-01-01")]
    assert (eligible["LabelConfirmedDate"] < pd.Timestamp("2025-01-05")).all()


def _orders():
    return [
        {"code": "LOW", "signal_date": "2025-01-01", "order_date": "2025-01-02",
         "entry_date": "2025-01-02", "exit_date": "2025-01-03", "prob": 0.7,
         "entry_price": 60.0, "exit_price": 60.0},
        {"code": "HIGH", "signal_date": "2025-01-01", "order_date": "2025-01-02",
         "entry_date": "2025-01-02", "exit_date": "2025-01-03", "prob": 0.9,
         "entry_price": 60.0, "exit_price": 60.0},
        {"code": "NEXT", "signal_date": "2025-01-02", "order_date": "2025-01-03",
         "entry_date": "2025-01-03", "exit_date": "2025-01-06", "prob": 0.8,
         "entry_price": 60.0, "exit_price": 60.0},
    ]


def test_shared_cash_probability_order_and_no_same_day_reuse():
    results, ledger = simulate_portfolio(
        _orders(), 100.0, PortfolioSettings(max_open_positions=1),
        pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    assert [(row["code"], row["status"]) for row in results] == [
        ("HIGH", "FILLED"),
        ("LOW", "SKIPPED_MAX_OPEN_POSITIONS"),
        ("NEXT", "SKIPPED_MAX_OPEN_POSITIONS"),
    ]
    assert min(row["available_cash"] for row in ledger) >= 0
    assert max(row["open_positions"] for row in ledger) <= 1


def test_lot_and_position_limit_skip_reason():
    order = [{"code": "A", "signal_date": "2025-01-01", "order_date": "2025-01-02",
              "entry_date": "2025-01-02", "exit_date": "2025-01-03", "prob": 1.0,
              "entry_price": 60.0, "exit_price": 60.0}]
    results, _ = simulate_portfolio(
        order, 100.0,
        PortfolioSettings(lot_size=2, max_position_percent=50, max_open_positions=1),
    )
    assert results[0]["status"] == "SKIPPED_POSITION_LIMIT"


def test_portfolio_output_is_deterministic():
    settings = PortfolioSettings(max_open_positions=1)
    calendar = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    first = simulate_portfolio(_orders(), 100.0, settings, calendar)
    second = simulate_portfolio(_orders(), 100.0, settings, calendar)
    assert first == second


def test_generated_backtest_outputs_and_cutoffs_are_valid():
    result_dir = Path("data/backtest_results")
    required = {
        "summary.json",
        "selected_rules.csv",
        "research_diagnostics.csv",
        "reference_predictions.csv",
        "reference_trades.csv",
        "reference_skipped_orders.csv",
        "daily_ledger.csv",
        "training_cutoffs.csv",
    }
    assert required <= {path.name for path in result_dir.iterdir()}
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["periods"] == {
        "reference_from": "2025-04-01",
        "reference_to": "2026-05-20",
        "research_from": "2020-01-01",
        "research_to": "2025-03-31",
    }
    cutoffs = pd.read_csv(result_dir / "training_cutoffs.csv")
    assert (
        pd.to_datetime(cutoffs["training_data_last_label_confirmed_date"])
        < pd.to_datetime(cutoffs["signal_date"])
    ).all()
    ledger = pd.read_csv(result_dir / "daily_ledger.csv")
    assert ledger["available_cash"].min() >= 0
    assert ledger["open_positions"].max() <= summary["portfolio_settings"]["max_open_positions"]
