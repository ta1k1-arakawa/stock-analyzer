from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.v4_meta_label_mvp import (
    PORTFOLIO_QUANTITY, aggregate_portfolio_metrics, baseline_filled_classification_metrics,
    evaluate_acceptance_conditions, evaluate_blocked_conditions, make_synthetic_phase2b_oof,
    portfolio_metrics, run_baseline_portfolio, run_v4_portfolio,
)


def setup_portfolios():
    oof, universe = make_synthetic_phase2b_oof()
    baseline_orders, baseline_ledger = run_baseline_portfolio(oof, universe)
    v4_orders, v4_ledger = run_v4_portfolio(baseline_orders)
    return oof, baseline_orders, baseline_ledger, v4_orders, v4_ledger


def test_quantity_is_always_100():
    _, orders, _, v4, _ = setup_portfolios()
    assert orders.loc[orders.portfolio_status == "FILLED", "quantity"].eq(PORTFOLIO_QUANTITY).all()
    assert v4.loc[v4.portfolio_status == "FILLED", "quantity"].eq(100).all()


def test_cheap_order_never_buys_more_than_one_lot():
    _, orders, _, _, _ = setup_portfolios()
    assert orders.quantity.max() == 100


def test_each_fold_resets_to_300000():
    _, _, ledger, _, _ = setup_portfolios()
    assert ledger.groupby("fold").first().available_cash.eq(300_000).all()


def test_same_day_exit_proceeds_are_unavailable_and_next_day_available():
    _, orders, _, _, _ = setup_portfolios()
    assert orders.skip_reason.eq("SAME_DAY_PROCEEDS_UNAVAILABLE").any()
    assert ((orders.EntryDate > orders.ExitDate.min()) & orders.portfolio_status.eq("FILLED")).any()


def test_max_one_open_position_and_no_negative_cash():
    _, _, ledger, _, v4_ledger = setup_portfolios()
    assert ledger.open_positions.max() <= 1 and v4_ledger.open_positions.max() <= 1
    assert ledger.available_cash.min() >= 0 and v4_ledger.available_cash.min() >= 0


def test_baseline_ignores_probability_and_v4_uses_only_baseline_fills():
    _, baseline, _, v4, _ = setup_portfolios()
    assert (baseline.loc[baseline.model_decision == "ABSTAIN", "portfolio_status"] == "FILLED").any()
    baseline_keys = set(map(tuple, baseline.loc[baseline.portfolio_status == "FILLED", ["fold", "signal_date", "ticker"]].to_numpy()))
    v4_keys = set(map(tuple, v4.loc[v4.portfolio_status == "FILLED", ["fold", "signal_date", "ticker"]].to_numpy()))
    assert v4_keys <= baseline_keys


def test_v4_abstain_does_not_create_substitute_trade():
    _, baseline, _, v4, _ = setup_portfolios()
    assert v4.portfolio_status.eq("ABSTAIN").sum() == baseline.loc[baseline.portfolio_status == "FILLED", "model_decision"].eq("ABSTAIN").sum()


def test_profit_and_zero_commission_are_preserved():
    _, baseline, _, _, _ = setup_portfolios()
    filled = baseline.loc[baseline.portfolio_status == "FILLED"].iloc[0]
    assert filled.realized_net_profit_yen == pytest.approx((filled.ExitPrice - filled.EntryPrice) * 100)
    assert filled.commission_cost == 0


def test_ledger_equity_and_drawdown_formula():
    _, orders, ledger, _, _ = setup_portfolios()
    assert (ledger.equity == ledger.available_cash + ledger.pending_cash + ledger.locked_entry_capital).all()
    metrics = portfolio_metrics(orders, ledger)
    assert metrics["max_drawdown_percent"] >= 0


def test_win_monthly_yearly_and_concentration_metrics():
    _, orders, ledger, _, _ = setup_portfolios()
    metrics = portfolio_metrics(orders, ledger)
    assert 0 <= metrics["win_rate"] <= 1 and 0 <= metrics["monthly_win_rate"] <= 1
    assert metrics["yearly_net_profit"] and 0 <= metrics["max_stock_positive_profit_share"] <= 1 and 0 <= metrics["max_industry_positive_profit_share"] <= 1


def test_zero_positive_profit_has_zero_shares():
    _, orders, ledger, _, _ = setup_portfolios()
    orders.loc[orders.portfolio_status == "FILLED", "realized_net_profit_yen"] = -1
    metrics = portfolio_metrics(orders, ledger)
    assert metrics["max_stock_positive_profit_share"] == metrics["top5_stock_positive_profit_share"] == metrics["max_industry_positive_profit_share"] == 0


def test_classification_uses_baseline_filled_subset_and_single_class_fold_is_none():
    _, orders, _, _, _ = setup_portfolios()
    metrics = baseline_filled_classification_metrics(orders)
    assert metrics["overall"]["sample_count"] == int(orders.portfolio_status.eq("FILLED").sum())
    orders.loc[orders.fold == 1, "label"] = 1
    assert baseline_filled_classification_metrics(orders)["folds"]["1"]["status"] == "BLOCKED"


def test_baseline_under_40_and_multiple_reasons_block():
    evidence = {"price_success_tickers": 149, "fold_sufficiency": {"1": {"reasons": ["TRAIN_CANDIDATES_LT_100"]}}, "baseline_closed_trades": {"1": 39}, "hashes_fixed": False, "post_2020_rows": 1, "network_hosts_allowed": False, "deterministic": False}
    result = evaluate_blocked_conditions(evidence)
    assert result["status"] == "FREE_META_LABEL_PROTOTYPE_BLOCKED" and len(result["reasons"]) >= 6


def acceptance_fixture():
    baseline_fold = {str(i): {"net_profit": 0, "max_drawdown_percent": 2} for i in (1,2,3)}
    v4_fold = {str(i): {"net_profit": 1, "max_drawdown_percent": 1} for i in (1,2,3)}
    baseline = {"aggregate_net_profit": 0, "max_drawdown_percent": 2, "win_rate": .4, "folds": baseline_fold, "negative_cash_count": 0, "capital_reuse_count": 0, "duplicate_order_count": 0}
    v4 = {"aggregate_net_profit": 1, "max_drawdown_percent": 1, "win_rate": .5, "closed_trades": 100, "folds": v4_fold, "negative_cash_count": 0, "capital_reuse_count": 0, "duplicate_order_count": 0, "max_stock_positive_profit_share": .3, "top5_stock_positive_profit_share": .5, "max_industry_positive_profit_share": .4}
    classification = {"overall": {"roc_auc": .6}, "folds": {str(i): {"roc_auc": .6} for i in (1,2,3)}}
    audit = {"price_success_tickers": 300, "fold_sufficiency": {}, "baseline_closed_trades": {str(i): 40 for i in (1,2,3)}, "hashes_fixed": True, "post_2020_rows": 0, "network_hosts_allowed": True, "deterministic": True, "byte_identical": True, "model_acceptance_rate": .5}
    return baseline, v4, classification, audit


def test_blocked_takes_priority_over_acceptance():
    base, v4, cls, audit = acceptance_fixture(); audit["post_2020_rows"] = 1
    assert evaluate_acceptance_conditions(base, v4, cls, audit)["status"] == "FREE_META_LABEL_PROTOTYPE_BLOCKED"


def test_all_17_conditions_pass_is_promising_and_one_failure_is_not_promising():
    base, v4, cls, audit = acceptance_fixture()
    result = evaluate_acceptance_conditions(base, v4, cls, audit)
    assert result["status"] == "FREE_META_LABEL_PROTOTYPE_PROMISING" and len(result["conditions"]) == 17
    v4["closed_trades"] = 99
    assert evaluate_acceptance_conditions(base, v4, cls, audit)["status"] == "FREE_META_LABEL_PROTOTYPE_NOT_PROMISING"


def test_safety_counters_and_aggregate_metrics_are_zero_or_deterministic():
    _, orders, ledger, _, _ = setup_portfolios()
    aggregate = aggregate_portfolio_metrics(orders, ledger)
    assert aggregate["negative_cash_count"] == aggregate["capital_reuse_count"] == aggregate["duplicate_order_count"] == 0
    assert aggregate["aggregate_ending_equity_equivalent"] == 300_000 + aggregate["aggregate_net_profit"]


def test_same_input_is_deterministic():
    _, first, first_ledger, _, _ = setup_portfolios()
    _, second, second_ledger, _, _ = setup_portfolios()
    assert first.equals(second) and first_ledger.equals(second_ledger)


def test_synthetic_phase2b_smoke_test_completes():
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, "scripts/run_v4_meta_label_mvp.py", "--synthetic-phase2b-smoke-test"], cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Phase 2B synthetic smoke test passed" in result.stdout
