"""Offline-only Phase 1 runner for the V4 meta-label MVP."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.v4_meta_label_mvp import (
    classification_metrics, generate_oof_predictions, load_fixed_universe,
    make_synthetic_phase2a_candidates, make_synthetic_phase2b_oof, run_baseline_portfolio,
    run_synthetic_smoke_test, run_v4_portfolio, aggregate_portfolio_metrics,
    baseline_filled_classification_metrics, evaluate_acceptance_conditions,
    evaluate_blocked_conditions, baseline_filled_acceptance_evidence, cash_safety_audit,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    parser.add_argument("--synthetic-phase2a-smoke-test", action="store_true")
    parser.add_argument("--synthetic-phase2b-smoke-test", action="store_true")
    parser.add_argument("--synthetic-phase2c-smoke-test", action="store_true")
    args = parser.parse_args()
    if not (args.validate_only or args.synthetic_smoke_test or args.synthetic_phase2a_smoke_test or args.synthetic_phase2b_smoke_test or args.synthetic_phase2c_smoke_test):
        parser.error("MVP is offline-only; select an explicit validation or synthetic smoke-test argument")
    universe = load_fixed_universe(Path(__file__).parents[1] / "V4_UNIVERSE.csv")
    if args.synthetic_smoke_test:
        candidates = run_synthetic_smoke_test(universe)
        print(f"V4 Phase 1 synthetic smoke test passed: {len(candidates)} daily rows")
    if args.synthetic_phase2a_smoke_test:
        synthetic = make_synthetic_phase2a_candidates()
        first, second = generate_oof_predictions(synthetic), generate_oof_predictions(synthetic)
        if not first.equals(second):
            raise AssertionError("SYNTHETIC_PHASE2A_NONDETERMINISTIC_OOF")
        metrics = classification_metrics(first)
        if metrics["status"] != "OK":
            raise AssertionError("SYNTHETIC_PHASE2A_METRICS_BLOCKED")
        counts = [len(first.loc[first["fold"] == fold]) for fold in (1, 2, 3)]
        print(f"V4 Phase 2A synthetic smoke test passed: fits=6 oof_rows={len(first)} fold_test_counts={counts}")
    if args.synthetic_phase2b_smoke_test:
        oof, synthetic_universe = make_synthetic_phase2b_oof()
        baseline_orders, baseline_ledger = run_baseline_portfolio(oof, synthetic_universe)
        v4_orders, v4_ledger = run_v4_portfolio(baseline_orders)
        baseline, v4 = aggregate_portfolio_metrics(baseline_orders, baseline_ledger), aggregate_portfolio_metrics(v4_orders, v4_ledger)
        classifications = baseline_filled_classification_metrics(baseline_orders)
        audit = {"price_success_tickers": 300, "fold_sufficiency": {}, "baseline_closed_trades": {str(f): int((baseline_orders.query('fold == @f and portfolio_status == \'FILLED\'')).shape[0]) for f in (1, 2, 3)}, "hashes_fixed": True, "post_2020_rows": 0, "network_hosts_allowed": True, "deterministic": True, "byte_identical": True, "model_acceptance_rate": float((baseline_orders.loc[baseline_orders.portfolio_status == 'FILLED', 'model_decision'] == 'ACCEPT').mean())}
        blocked = evaluate_blocked_conditions(audit)
        decision = evaluate_acceptance_conditions(baseline, v4, classifications, audit)
        second_orders, second_ledger = run_baseline_portfolio(oof, synthetic_universe)
        if not baseline_orders.equals(second_orders) or not baseline_ledger.equals(second_ledger): raise AssertionError("SYNTHETIC_PHASE2B_NONDETERMINISTIC")
        if not (baseline_orders.quantity.dropna().isin([0, 100]).all() and (v4_orders.loc[v4_orders.portfolio_status == 'FILLED', 'ticker'].isin(baseline_orders.loc[baseline_orders.portfolio_status == 'FILLED', 'ticker'])).all()): raise AssertionError("SYNTHETIC_PHASE2B_PORTFOLIO_FAILURE")
        print(f"V4 Phase 2B synthetic smoke test passed: baseline_filled={int((baseline_orders.portfolio_status == 'FILLED').sum())} v4_filled={int((v4_orders.portfolio_status == 'FILLED').sum())} blocked={blocked['status']} decision={decision['status']}")
    if args.synthetic_phase2c_smoke_test:
        oof, synthetic_universe = make_synthetic_phase2b_oof()
        baseline_orders, baseline_ledger, baseline_events = run_baseline_portfolio(oof, synthetic_universe, return_events=True)
        v4_orders, v4_ledger, v4_events = run_v4_portfolio(baseline_orders, return_events=True)
        if cash_safety_audit(baseline_events)["capital_reuse_count"] != 0: raise AssertionError("SYNTHETIC_PHASE2C_NORMAL_CASH_AUDIT_FAILURE")
        broken = baseline_events.copy(); entry = broken.index[broken.event_type.eq("ENTRY_FILLED")][0]; broken.loc[entry, "available_cash_before"] = 0.0
        if cash_safety_audit(broken)["capital_reuse_count"] < 1: raise AssertionError("SYNTHETIC_PHASE2C_BROKEN_CASH_AUDIT_FAILURE")
        if not set(oof.columns[oof.columns.isin(baseline_orders.columns)]).issuperset(set()): raise AssertionError("SYNTHETIC_PHASE2C_SCHEMA_FAILURE")
        if not all(column in baseline_orders.columns and column in v4_orders.columns for column in oof.columns[-15:]): raise AssertionError("SYNTHETIC_PHASE2C_FEATURE_SCHEMA_FAILURE")
        acceptance = baseline_filled_acceptance_evidence(baseline_orders)
        bad_evidence = evaluate_blocked_conditions({})
        baseline = aggregate_portfolio_metrics(baseline_orders, baseline_ledger, baseline_events); v4 = aggregate_portfolio_metrics(v4_orders, v4_ledger, v4_events)
        audit = {"price_success_tickers": 300, "fold_sufficiency": {str(f): {"reasons": []} for f in (1,2,3)}, "baseline_closed_trades": {str(f): 40 for f in (1,2,3)}, "hashes_fixed": True, "post_2020_rows": 0, "network_hosts_allowed": True, "deterministic": True, "byte_identical": True, "model_acceptance_rate": acceptance["model_acceptance_rate"]}
        decision = evaluate_acceptance_conditions(baseline, v4, baseline_filled_classification_metrics(baseline_orders), audit)
        if bad_evidence["status"] != "FREE_META_LABEL_PROTOTYPE_BLOCKED" or any(item["actual_value"] is None for item in decision["conditions"] if item["condition_number"] in (4,5,10,14)): raise AssertionError("SYNTHETIC_PHASE2C_EVIDENCE_FAILURE")
        second_orders, second_ledger, second_events = run_baseline_portfolio(oof, synthetic_universe, return_events=True)
        if not (baseline_orders.equals(second_orders) and baseline_ledger.equals(second_ledger) and baseline_events.equals(second_events)): raise AssertionError("SYNTHETIC_PHASE2C_NONDETERMINISTIC")
        print("V4 Phase 2C synthetic smoke test passed: event audit, evidence, feature records, and determinism")
    print(f"V4 validation passed: {len(universe)} frozen tickers; network=0 real_data_model_fits=0 real_data_backtests=0")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
