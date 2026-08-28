from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.v4_meta_label_mvp import (
    FEATURE_COLUMNS, baseline_filled_acceptance_evidence, cash_safety_audit,
    evaluate_acceptance_conditions, evaluate_blocked_conditions, make_synthetic_phase2b_oof,
    run_baseline_portfolio, run_v4_portfolio,
)
from tests.test_v4_meta_label_mvp_phase2b import acceptance_fixture


def setup_events():
    oof, universe = make_synthetic_phase2b_oof()
    baseline, ledger, events = run_baseline_portfolio(oof, universe, return_events=True)
    v4, v4_ledger, v4_events = run_v4_portfolio(baseline, return_events=True)
    return baseline, ledger, events, v4, v4_ledger, v4_events


def test_normal_events_have_zero_capital_reuse_and_not_a_constant():
    _, _, events, _, _, _ = setup_events()
    assert cash_safety_audit(events)["capital_reuse_count"] == 0
    broken = events.copy(); index = broken.index[broken.event_type.eq("ENTRY_FILLED")][0]; broken.loc[index, "available_cash_before"] = 0
    assert cash_safety_audit(broken)["capital_reuse_count"] >= 1


def test_same_day_exit_after_entry_and_exit_cash_mutation_are_detected():
    _, _, events, _, _, _ = setup_events()
    broken = events.copy(); exit_index = broken.index[broken.event_type.eq("EXIT_TO_PENDING")][0]; entry_index = broken.index[broken.event_type.eq("ENTRY_FILLED")][0]
    broken.loc[exit_index, "date"] = broken.loc[entry_index, "date"]
    broken.loc[exit_index, "sequence"] = broken.loc[entry_index, "sequence"] - 1
    broken.loc[exit_index, "available_cash_after"] = broken.loc[exit_index, "available_cash_before"] + 1
    assert cash_safety_audit(broken)["capital_reuse_count"] >= 2


def test_all_portfolio_records_keep_feature_schema():
    baseline, _, _, v4, _, _ = setup_events()
    assert set(FEATURE_COLUMNS).issubset(baseline.columns) and set(FEATURE_COLUMNS).issubset(v4.columns)
    assert set(FEATURE_COLUMNS).issubset(baseline.loc[baseline.portfolio_status == "SKIPPED"].columns)
    assert set(FEATURE_COLUMNS).issubset(v4.loc[v4.portfolio_status == "ABSTAIN"].columns)


def test_acceptance_denominator_is_baseline_filled_only_and_empty_is_blocked():
    baseline, _, _, _, _, _ = setup_events()
    evidence = baseline_filled_acceptance_evidence(baseline)
    assert evidence["baseline_filled_opportunity_count"] == int(baseline.portfolio_status.eq("FILLED").sum())
    assert evidence["baseline_filled_opportunity_count"] < len(baseline)
    assert baseline_filled_acceptance_evidence(baseline.loc[baseline.portfolio_status != "FILLED"])["status"] == "BLOCKED"


def test_missing_evidence_fails_closed_for_all_folds_and_post_2020():
    reasons = evaluate_blocked_conditions({})["reasons"]
    assert "POST_2020_ROWS_EVIDENCE_MISSING_OR_INVALID" in reasons
    assert all(f"FOLD_{fold}_SUFFICIENCY_EVIDENCE_MISSING" in reasons for fold in (1,2,3))
    assert all(f"FOLD_{fold}_BASELINE_CLOSED_TRADES_EVIDENCE_MISSING" in reasons for fold in (1,2,3))


def test_one_missing_fold_evidence_blocks():
    evidence = {"price_success_tickers": 300, "fold_sufficiency": {"1": {"reasons": []}, "2": {"reasons": []}}, "baseline_closed_trades": {"1": 40, "2": 40, "3": 40}, "hashes_fixed": True, "post_2020_rows": 0, "network_hosts_allowed": True, "deterministic": True}
    assert "FOLD_3_SUFFICIENCY_EVIDENCE_MISSING" in evaluate_blocked_conditions(evidence)["reasons"]


def test_actual_values_for_conditions_4_5_10_14_are_concrete():
    baseline, v4, classification, audit = acceptance_fixture()
    conditions = evaluate_acceptance_conditions(baseline, v4, classification, audit)["conditions"]
    actual = {item["condition_number"]: item["actual_value"] for item in conditions}
    assert actual[4]["winning_fold_count"] == 3
    assert "folds" in actual[5] and "fold_roc_auc" in actual[10]
    assert len(actual[14]) == 6


def test_phase2c_smoke_test_completes_and_events_are_deterministic():
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, "scripts/run_v4_meta_label_mvp.py", "--synthetic-phase2c-smoke-test"], cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Phase 2C synthetic smoke test passed" in result.stdout
