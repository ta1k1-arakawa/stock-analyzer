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
from v6_a_r2_preflight import PreflightBlocked, blocked_json_payload, load_read_only_formal_inputs, run_read_only_preflight
from v6_a_r2_formal import CONFIRMATION, atomic_write_formal_artifacts, build_formal_bundle, run_formal_two_pass


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="run_v6_a_r2_causal_breakout")
    parser.add_argument("--synthetic-golden-test", action="store_true")
    parser.add_argument("--preflight-formal-path", action="store_true")
    parser.add_argument("--training-cache")
    parser.add_argument("--evaluation-cache")
    parser.add_argument("--evaluate-cache", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--confirmation")
    args = parser.parse_args(raw)
    if args.evaluate_cache:
        # This dispatch intentionally performs all guards before cache or engine access.
        if args.confirmation != CONFIRMATION:
            print("GATE_4_FORMAL_EVALUATION_CONFIRMATION_REQUIRED")
            return 2
        repo = Path(__file__).resolve().parents[1]
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()
        origin = subprocess.run(["git", "rev-parse", "origin/v6-a-r2-causal-breakout-baseline"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, check=True,
                               capture_output=True, text=True).stdout != ""
        if branch != "v6-a-r2-causal-breakout-baseline" or head != origin or dirty:
            print("GATE_4_REPOSITORY_GUARD_FAILED")
            return 2
        if not args.training_cache or not args.evaluation_cache or not args.output_dir:
            parser.error("formal evaluation requires cache paths and --output-dir")
        if repo == Path(args.output_dir).resolve() or repo in Path(args.output_dir).resolve().parents:
            print("GATE_4_OUTPUT_DIRECTORY_INSIDE_REPOSITORY")
            return 2
        try:
            preflight = run_read_only_preflight(args.training_cache, args.evaluation_cache, head, branch, True)
            inputs = load_read_only_formal_inputs(args.training_cache, args.evaluation_cache)
            bundle = build_formal_bundle(preflight, inputs["raw_price_frames"], inputs["common_calendar"], inputs["accepted_candidates"], inputs["full_candidate_audit"], inputs["market_gate_audit"])
            metadata = {"repository_commit": head, "branch": branch, "training_manifest_sha": preflight["training_manifest_sha"], "evaluation_manifest_sha": preflight["evaluation_manifest_sha"], "universe_csv_sha": preflight["universe_csv_sha"], "ticker_list_sha": preflight["ticker_list_sha"], "candidate_rules": "frozen_v6_a", "ranking_rules": "frozen_v6_a", "portfolio_rules": "causal_d0_d1_d10", "event_phase_order": ["phase1_release_proceeds", "phase2_attempt_entries", "phase3_execute_exits", "phase4_record_equity", "phase5_queue_signals"]}
            result = run_formal_two_pass(bundle, metadata)
            atomic_write_formal_artifacts(args.output_dir, result["artifacts"], repo)
        except Exception as error:
            print(f"V6_A_BREAKOUT_BASELINE_EXPLORATORY_BLOCKED:{type(error).__name__}")
            return 1
        print(result["summary"]["verdict"])
        return 0
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
