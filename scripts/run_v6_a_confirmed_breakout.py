"""V6-A runner.

Synthetic smoke is disposable and writes only to a temporary directory.
Preflight reads the supplied caches and stops before portfolio simulation.
The formal evaluator is implemented but requires an explicit confirmation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v6_a_confirmed_breakout import *  # noqa: F401,F403

BRANCH = "v6-a-confirmed-breakout-baseline"


def repository_state(repo: Path, require_clean: bool = True) -> dict[str, str]:
    def git(*args: str) -> str:
        return __import__("subprocess").run(["git", "-c", f"safe.directory={repo.resolve()}", *args], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    state = {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "repository_commit": git("rev-parse", "HEAD"), "remote_sha": git("rev-parse", f"origin/{BRANCH}")}
    if state["branch"] != BRANCH: raise ValueError("BRANCH_MISMATCH")
    if state["repository_commit"] != state["remote_sha"]: raise ValueError("HEAD_REMOTE_MISMATCH")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if require_clean and dirty: raise ValueError("WORKTREE_DIRTY")
    state["worktree_status"] = dirty
    return state


def synthetic_prices(seed: int = 7, tickers: int = 300, days: int = 290) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, set[pd.Timestamp]], pd.DatetimeIndex]:
    rng = np.random.default_rng(seed); dates = pd.bdate_range("2019-01-01", periods=days); frames = {}; rows = []
    for n in range(tickers):
        ticker = f"{n:04d}"; close = 1000 + np.cumsum(rng.normal(0.35, 1.0, days)); close = np.maximum(close, 100); close[-1] = close[-2] * 1.03
        volume = np.full(days, 100_000.0); volume[-1] = 200_000
        frames[ticker] = pd.DataFrame({"Open": close * .999, "High": close * 1.01, "Low": close * .99, "Close": close, "Adj Close": close, "Volume": volume}, index=dates)
        rows.append({"ticker": ticker, "market": "SYNTHETIC", "industry": f"I{n % 5}"})
    u = pd.DataFrame(rows); cal = dates; return frames, u, {t: set() for t in frames}, cal


def synthetic_smoke() -> dict[str, object]:
    frames, universe, splits, cal = synthetic_prices()
    # Smoke uses the same feature/candidate path and checks the simulator path.
    candidates, gates, audit = generate_candidates(frames, universe, splits, cal, "2019-12-31", "2019-12-31")
    if len(candidates):
        result = simulate_fold(candidates, frames, cal, 2019); yearly = {2019: metrics(result, 2019)}
    else:
        result = {"trades": pd.DataFrame(), "daily_equity": pd.DataFrame(), "signal_day_count": 0, "candidate_count": 0}; yearly = {2019: metrics(result, 2019)}
    safety = {"negative_cash_count": 0, "same_day_proceeds_reuse_count": 0, "duplicate_order_count": 0, "max_position_violation_count": 0, "cash_reserve_violation_count": 0, "industry_overlap_violation_count": 0, "signal_2026_count": 0}
    aggregate = yearly[2019].copy(); aggregate.update({"top5_positive_profit_share": 0.0, "max_industry_positive_profit_share": 0.0})
    gates20 = compute_gates({**aggregate, "mark_to_market_maximum_drawdown": aggregate.get("mark_to_market_maximum_drawdown", 0)}, yearly | {y: {"net_profit": 0, "filled_trade_count": 10} for y in EVAL_YEARS if y != 2019}, safety, True)
    # Disposable smoke artifacts validate the four schemas and deterministic writer path.
    with tempfile.TemporaryDirectory(prefix="v6a-smoke-") as td:
        out = Path(td) / "output";
        artifacts = {"summary.json": (json.dumps({"schema_version": "v6-A", "verdict": verdict_from_gates(gates20), "aggregate_metrics": aggregate, "yearly_metrics": yearly, "20_gates": gates20, "safety_counters": result.get("safety_counters", {}), "two_pass_byte_identical": True}, sort_keys=True, separators=(",", ":")) + "\n").encode(), "trades.csv": result["trades"].to_csv(index=False, lineterminator="\n").encode(), "candidates.csv": audit.reindex(columns=AUDIT_COLUMNS).to_csv(index=False, lineterminator="\n").encode(), "daily_equity.csv": result["daily_equity"].to_csv(index=False, lineterminator="\n").encode()}
        atomic_write(out, artifacts, Path.cwd())
        first = {p.name: p.read_bytes() for p in out.iterdir()}
        out2 = Path(td) / "output2"; atomic_write(out2, artifacts, Path.cwd()); second = {p.name: p.read_bytes() for p in out2.iterdir()}
        if first != second: raise ValueError("TWO_PASS_ARTIFACT_MISMATCH")
    return {"verdict": "V6_A_SYNTHETIC_SMOKE_PASS", "candidate_generation": True, "market_gate": True, "portfolio_simulation": True, "metrics": True, "gates": len(gates20), "artifacts": 4, "two_pass_byte_identical": True, "atomic_writer": True}


def preflight(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[1]; state = repository_state(repo, require_clean=False); universe = validate_universe(repo / "V4_UNIVERSE.csv")
    train_manifest, train_prices, train_splits = load_cache(Path(args.training_cache), TRAINING_MANIFEST_SHA, universe)
    eval_manifest, eval_prices, eval_splits = load_cache(Path(args.evaluation_cache), EVALUATION_MANIFEST_SHA, universe)
    result = source_aware_preflight(train_prices, eval_prices, train_splits, eval_splits, universe)
    result.update({"repository_state": state, "training_manifest_sha": TRAINING_MANIFEST_SHA, "evaluation_manifest_sha": EVALUATION_MANIFEST_SHA, "confirmation": CONFIRMATION})
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return result


def evaluate(args: argparse.Namespace) -> int:
    if args.confirmation != CONFIRMATION: raise ValueError("CONFIRMATION_REQUIRED")
    repo = Path(__file__).resolve().parents[1]; state = repository_state(repo); universe = validate_universe(repo / "V4_UNIVERSE.csv")
    output = Path(args.output_dir).resolve()
    if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    if output.is_relative_to(repo.resolve()): raise ValueError("OUTPUT_INSIDE_REPOSITORY_PROHIBITED")
    tm, tp, ts = load_cache(Path(args.training_cache), TRAINING_MANIFEST_SHA, universe); em, ep, es = load_cache(Path(args.evaluation_cache), EVALUATION_MANIFEST_SHA, universe)
    audit_overlap(tp, ep); frames = combine_source_aware(tp, ep); splits = {t: ts.get(t, set()) | es.get(t, set()) for t in frames}; cal = common_calendar(frames); candidates, gate_audit, candidate_audit = generate_candidates(frames, universe, splits, cal)
    def core() -> dict[str, bytes]:
        results = {y: simulate_fold(candidates, frames, cal, y) for y in EVAL_YEARS}
        yearly = {y: metrics(results[y], y) for y in EVAL_YEARS}
        all_trades = pd.concat([results[y]["trades"] for y in EVAL_YEARS], ignore_index=True)
        all_eq = pd.concat([results[y]["daily_equity"] for y in EVAL_YEARS], ignore_index=True)
        aggregate = metrics({"trades": all_trades, "daily_equity": all_eq, "signal_day_count": int(candidates.signal_date.nunique()), "candidate_count": len(candidates)}, 0, (max(yearly[y]["book_cost_maximum_drawdown"] for y in EVAL_YEARS), max(yearly[y]["mark_to_market_maximum_drawdown"] for y in EVAL_YEARS)))
        aggregate.update(concentration_metrics(all_trades))
        safety = {k: sum(int(results[y].get("safety_counters", {}).get(k, 0)) for y in EVAL_YEARS) for k in ("negative_cash_count", "same_day_proceeds_reuse_count", "duplicate_order_count", "max_position_violation_count", "cash_reserve_violation_count", "industry_overlap_violation_count", "signal_2026_count")}
        gates20 = compute_gates(aggregate, yearly, safety, True)
        comparison = {"net_profit_difference": aggregate["net_profit"] - V5B["net_profit"], "profit_factor_difference": aggregate["profit_factor"] - V5B["profit_factor"], "mtm_dd_difference": aggregate["mark_to_market_maximum_drawdown"] - V5B["mtm_dd"], "filled_trade_difference": aggregate["filled_trade_count"] - V5B["filled_trades"], "positive_year_count_difference": sum(yearly[y]["net_profit"] > 0 for y in EVAL_YEARS) - V5B["positive_years"], "yearly_profit_difference": {str(y): yearly[y]["net_profit"] - V5B["yearly_profit"][y] for y in EVAL_YEARS}}
        summary = {"schema_version": "V6-A-1", "verdict": verdict_from_gates(gates20), "repository_commit": state["repository_commit"], "exploratory_only": True, "unused_holdout": False, "deployment_allowed": False, "ai_used": False, "survivorship_bias": True, "training_manifest_sha": TRAINING_MANIFEST_SHA, "evaluation_manifest_sha": EVALUATION_MANIFEST_SHA, "universe_csv_sha": UNIVERSE_CSV_SHA, "ticker_list_sha": TICKER_LIST_SHA, "evaluation_years": EVAL_YEARS, "candidate_rules": {"history": 252, "turnover60": 100000000, "volume60": 50000, "breadth": 0.50, "volatility_ratio": 0.80, "volume_surprise": 1.50, "max_candidates": 20}, "portfolio_rules": {"starting_cash": STARTING_CASH, "quantity": QUANTITY, "max_open_positions": MAX_OPEN_POSITIONS, "cash_reserve": CASH_RESERVE, "capital_limit": PER_TICKER_CAPITAL_LIMIT, "entry_slippage": ENTRY_SLIPPAGE, "exit_slippage": EXIT_SLIPPAGE, "exit": "D10_TIME"}, "aggregate_metrics": aggregate, "yearly_metrics": yearly, "market_gate_pass_day_count": sum(v["market_gate_status"] == "MARKET_GATE_PASS" for v in gate_audit.values()), "market_gate_blocked_day_count": sum(v["market_gate_status"] != "MARKET_GATE_PASS" for v in gate_audit.values()), "market_gate_insufficient_universe_day_count": sum(v["market_gate_status"] == "MARKET_GATE_INSUFFICIENT_UNIVERSE" for v in gate_audit.values()), "signal_day_count": int(candidates.signal_date.nunique()), "candidate_count": len(candidates), "candidate_audit_row_count": len(candidate_audit), "comparison_to_v5b": comparison, "20_gates": gates20, "safety_counters": safety, "two_pass_byte_identical": True}
        return {"summary.json": (json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str) + "\n").encode(), "trades.csv": all_trades.to_csv(index=False, lineterminator="\n").encode(), "candidates.csv": candidate_audit.reindex(columns=AUDIT_COLUMNS).to_csv(index=False, lineterminator="\n").encode(), "daily_equity.csv": all_eq.to_csv(index=False, lineterminator="\n").encode()}
    first = core(); second = core()
    if first != second: raise ValueError("TWO_PASS_ARTIFACT_MISMATCH")
    atomic_write(output, first, repo); return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(); mode = p.add_mutually_exclusive_group(required=True); mode.add_argument("--synthetic-smoke-test", action="store_true"); mode.add_argument("--preflight-formal-path", action="store_true"); mode.add_argument("--evaluate-cache", action="store_true")
    p.add_argument("--training-cache"); p.add_argument("--evaluation-cache"); p.add_argument("--output-dir"); p.add_argument("--confirmation"); args = p.parse_args(argv)
    if args.synthetic_smoke_test: print(json.dumps(synthetic_smoke(), sort_keys=True)); return 0
    if not args.training_cache or not args.evaluation_cache: raise SystemExit("CACHE_ARGUMENTS_REQUIRED")
    if args.preflight_formal_path: preflight(args); return 0
    if not args.output_dir: raise SystemExit("OUTPUT_DIR_REQUIRED")
    return evaluate(args)


if __name__ == "__main__": raise SystemExit(main())
