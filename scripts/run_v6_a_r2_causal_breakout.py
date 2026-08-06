"""Synthetic-only Gate 2 runner for V6-A-R2."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v6_a_r2_causal_breakout import run_synthetic_golden, write_synthetic_artifacts
from v6_a_r2_preflight import PreflightBlocked, blocked_json_payload, run_read_only_preflight


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--preflight-formal-path" in raw or "--evaluate-cache" in raw:
        if "--evaluate-cache" in raw:
            print("GATE_3_FORMAL_EVALUATION_NOT_AUTHORIZED")
            return 2
    parser = argparse.ArgumentParser(prog="run_v6_a_r2_causal_breakout")
    parser.add_argument("--synthetic-golden-test", action="store_true")
    parser.add_argument("--preflight-formal-path", action="store_true")
    parser.add_argument("--training-cache")
    parser.add_argument("--evaluation-cache")
    args = parser.parse_args(raw)
    if args.synthetic_golden_test and args.preflight_formal_path:
        parser.error("choose one authorized mode")
    if args.preflight_formal_path:
        if not args.training_cache or not args.evaluation_cache:
            parser.error("preflight requires --training-cache and --evaluation-cache")
        if args.training_cache is None or args.evaluation_cache is None:
            parser.error("preflight requires cache paths")
        repo = Path(__file__).resolve().parents[1]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        clean = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo,
                               check=True, capture_output=True, text=True).stdout == ""
        try:
            result = run_read_only_preflight(args.training_cache, args.evaluation_cache,
                                             commit, branch, clean)
        except PreflightBlocked as error:
            print(json.dumps(blocked_json_payload(error), ensure_ascii=False, sort_keys=True))
            return 1
        except Exception as error:
            print(json.dumps({"verdict": "V6_A_R2_REAL_CACHE_PREFLIGHT_BLOCKED",
                              "blocked_stage": "CACHE_VALIDATION",
                              "error": type(error).__name__}, sort_keys=True))
            return 1
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.synthetic_golden_test:
        parser.error("only --synthetic-golden-test or --preflight-formal-path is authorized")
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
