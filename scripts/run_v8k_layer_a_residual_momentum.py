"""Explicit offline runner for the V8K Layer A residual-momentum measurement."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v8k_layer_a_residual_momentum import run_cache_measurement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-cache", action="store_true")
    parser.add_argument("--evaluation-cache")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if not args.measure_cache:
        raise SystemExit("MEASURE_CACHE_FLAG_REQUIRED")
    if not args.evaluation_cache or not args.output_dir:
        raise SystemExit("EVALUATION_CACHE_AND_OUTPUT_DIR_REQUIRED")
    repository_root = Path(__file__).resolve().parents[1]
    run_cache_measurement(Path(args.evaluation_cache), Path(args.output_dir), repository_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
