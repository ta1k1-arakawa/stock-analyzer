"""Synthetic-only Gate 2 runner for V6-A-R2."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v6_a_r2_causal_breakout import run_synthetic_golden, write_synthetic_artifacts


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--preflight-formal-path" in raw or "--evaluate-cache" in raw:
        print("GATE_2_REAL_CACHE_PREFLIGHT_NOT_AUTHORIZED" if "--preflight-formal-path" in raw
              else "GATE_3_FORMAL_EVALUATION_NOT_AUTHORIZED")
        return 2
    parser = argparse.ArgumentParser(prog="run_v6_a_r2_causal_breakout")
    parser.add_argument("--synthetic-golden-test", action="store_true")
    args = parser.parse_args(raw)
    if not args.synthetic_golden_test:
        parser.error("only --synthetic-golden-test is authorized")
    result = run_synthetic_golden()
    temp = Path(__file__).resolve().parents[1] / ".v6_a_r2_synthetic_tmp"
    if temp.exists():
        shutil.rmtree(temp)
    try:
        first = temp / "pass1"
        second = temp / "pass2"
        write_synthetic_artifacts(first, result)
        second_result = run_synthetic_golden()
        write_synthetic_artifacts(second, second_result)
        for name in result.artifacts:
            if (first / name).read_bytes() != (second / name).read_bytes():
                raise RuntimeError("SYNTHETIC_TWO_PASS_BYTE_MISMATCH")
        print("SYNTHETIC_GOLDEN_PASS")
        print(f"future_read_error={result.future_read_error}")
        print("artifacts=summary.json,trades.csv,candidates.csv,daily_equity.csv")
        print("two_pass_byte_identical=true")
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
