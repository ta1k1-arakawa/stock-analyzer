"""Offline-only Phase 1 runner for the V4 meta-label MVP."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.v4_meta_label_mvp import load_fixed_universe

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    args = parser.parse_args()
    if not (args.validate_only or args.synthetic_smoke_test):
        parser.error("Phase 1 is offline-only; use --validate-only or --synthetic-smoke-test")
    universe = load_fixed_universe(Path(__file__).parents[1] / "V4_UNIVERSE.csv")
    print(f"V4 Phase 1 validation passed: {len(universe)} frozen tickers; network=0 model_fits=0 backtests=0")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
