from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from backtest import (
    CachedValidationPrediction,
    PredictionCacheKey,
    Rule,
    SelectionConstraints,
    assert_validation_folds_non_overlapping,
    classify_validation_metrics,
    coordinate_select_rules,
    eligible_training_rows,
    evaluate_validation_portfolio,
    select_rule_from_diagnostics,
)
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


def _coordinate_fixture():
    active_a = Rule("A", 1.0, 5.0, 0.5, 30.0)
    inactive_a = Rule("A", 1.0, 5.0, 0.95, 0.0)
    active_b = Rule("B", 1.0, 5.0, 0.5, 300.0)
    cache = {}
    fold_dates = {
        1: ("2024-01-02", "2024-01-03", "2024-01-04"),
        2: ("2024-02-01", "2024-02-02", "2024-02-05"),
        3: ("2024-03-01", "2024-03-04", "2024-03-05"),
    }
    for fold, (signal_date, entry_date, exit_date) in fold_dates.items():
        for code, probability, exit_price in (("A", 0.9, 110.0), ("B", 0.8, 200.0)):
            key = PredictionCacheKey(
                code, fold, 1.0, 5.0, ("Feature",), 42, signal_date
            )
            cache[key] = CachedValidationPrediction(
                key=key,
                validation_from=signal_date,
                validation_to=exit_date,
                validation_dates=[signal_date, entry_date, exit_date],
                orders=[
                    {
                        "code": code,
                        "signal_date": signal_date,
                        "planned_entry_date": entry_date,
                        "order_date": entry_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "prob": probability,
                        "entry_price": 100.0,
                        "exit_price": exit_price,
                        "exit_reason": "TIME",
                        "return_percent": exit_price - 100.0,
                        "commission_percent": 0.0,
                    }
                ],
                training_row_count=100,
                training_last_feature_date="2023-12-28",
                training_last_label_confirmed_date="2023-12-29",
            )
    constraints = SelectionConstraints(1, 100.0, 0.0, 100.0)
    return (
        {"A": active_a, "B": active_b},
        {"A": [active_a, inactive_a], "B": [active_b]},
        cache,
        constraints,
    )


def test_portfolio_coordinate_choice_differs_from_independent_best():
    initial, candidates, cache, constraints = _coordinate_fixture()
    selected, metrics, _, _, _ = coordinate_select_rules(
        initial, candidates, cache, 100.0,
        PortfolioSettings(max_open_positions=1), constraints,
    )
    assert initial["A"].threshold == 0.5
    assert selected["A"].threshold == 0.95
    assert metrics["total_profit"] == 300.0
    assert all(row["profit"] == 100.0 for row in metrics["by_fold"])


def test_validation_fold_cash_resets_and_all_stocks_share_limits():
    initial, _, cache, constraints = _coordinate_fixture()
    metrics = evaluate_validation_portfolio(
        initial, cache, 100.0, PortfolioSettings(max_open_positions=1), constraints
    )
    assert [row["profit"] for row in metrics["by_fold"]] == [10.0, 10.0, 10.0]
    assert metrics["skip_counts"] == {"SKIPPED_MAX_OPEN_POSITIONS": 3}
    assert {row["code"] for row in metrics["trades"]} == {"A"}


def test_validation_fold_overlap_is_rejected():
    _, _, cache, _ = _coordinate_fixture()
    entry = next(value for value in cache.values() if value.key.fold == 2)
    entry.validation_dates.append("2024-01-03")
    with pytest.raises(ValueError, match="overlap"):
        assert_validation_folds_non_overlapping(cache)


def test_test_and_reference_values_cannot_change_independent_initial_rule():
    base = pd.DataFrame(
        [
            {"TargetPercent": 1.0, "StopLossPercent": 2.0, "Threshold": 0.2,
             "ValidationProfit": 100.0, "ValidationTrades": 10,
             "TestProfitDiagnosticOnly": 1.0, "ReferenceProfit": 1.0},
            {"TargetPercent": 2.0, "StopLossPercent": 5.0, "Threshold": 0.5,
             "ValidationProfit": 50.0, "ValidationTrades": 10,
             "TestProfitDiagnosticOnly": 2.0, "ReferenceProfit": 2.0},
        ]
    )
    changed = base.copy()
    changed["TestProfitDiagnosticOnly"] = [999999.0, -999999.0]
    changed["ReferenceProfit"] = [-999999.0, 999999.0]
    assert select_rule_from_diagnostics("X", base, 1) == select_rule_from_diagnostics("X", changed, 1)


def _passing_metrics():
    return {
        "total_profit": 100.0,
        "positive_folds": 3,
        "min_fold_profit": 10.0,
        "worst_max_drawdown_percent": 5.0,
        "month_win_rate": 75.0,
        "max_stock_profit_share": 60.0,
        "trade_count": 20,
        "skip_counts": {},
    }


@pytest.mark.parametrize(
    ("field", "value", "failed"),
    [
        ("worst_max_drawdown_percent", 16.0, "drawdown_ok"),
        ("month_win_rate", 49.0, "monthly_stability_ok"),
        ("max_stock_profit_share", 71.0, "stock_dependency_ok"),
    ],
)
def test_portfolio_selection_constraints(field, value, failed):
    metrics = _passing_metrics()
    metrics[field] = value
    result = classify_validation_metrics(
        metrics, SelectionConstraints(10, 15.0, 50.0, 70.0)
    )
    assert result["selection_status"] == "REVIEW"
    assert failed in result["failed_checks"]


def test_coordinate_search_is_deterministic():
    initial, candidates, cache, constraints = _coordinate_fixture()
    args = (initial, candidates, cache, 100.0, PortfolioSettings(max_open_positions=1), constraints)
    first = coordinate_select_rules(*args)
    second = coordinate_select_rules(*args)
    assert first == second
