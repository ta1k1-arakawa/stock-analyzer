"""Offline-only Phase 1 runner for the V4 meta-label MVP."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.v4_meta_label_mvp import (
    classification_metrics, generate_oof_predictions, load_fixed_universe,
    make_synthetic_phase2a_candidates, run_synthetic_smoke_test,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    parser.add_argument("--synthetic-phase2a-smoke-test", action="store_true")
    args = parser.parse_args()
    if not (args.validate_only or args.synthetic_smoke_test or args.synthetic_phase2a_smoke_test):
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
    print(f"V4 validation passed: {len(universe)} frozen tickers; network=0 real_data_model_fits=0 real_data_backtests=0")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
