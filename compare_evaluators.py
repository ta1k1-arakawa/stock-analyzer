"""Compare immutable legacy baseline with evaluator-v2 on one fixed snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.benchmark import FixedOHLCVLoader, sha256_file
from src.comparison import (
    ComparisonError, assert_baseline_unchanged, build_execution_orders,
    capital_overlap, deterministic_hashes, run_independent_budget,
    run_v2_portfolio, scenario_metrics, verify_baseline,
)
from src.trade_simulator import PortfolioSettings


BASELINE_COMMIT = "2975e3375c615052bd3a1ab2e5a24e723e94c46b"
CANDIDATE_COMMIT = "94b016575d49436bd4017a21a1de252ab0d95834"
OUTPUT_NAMES = [
    "comparison_manifest.json", "legacy_summary.json", "candidate_summary.json",
    "scenario_comparison.csv", "comparison_by_stock.csv", "trade_alignment.csv",
    "capital_overlap.csv", "difference_report.md", "legacy_predictions.csv",
    "legacy_trades.csv", "run_metadata.json",
]


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_value(row) for row in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else round(float(value), 8)
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(value, 8)
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    output = frame.copy()
    if columns is not None:
        output = output.reindex(columns=columns)
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.8f")


def _legacy_child(
    repo: Path, baseline: Path, benchmark: Path, before: dict[str, str]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="baseline-comparison-") as temp_name:
        temp = Path(temp_name)
        shutil.copy2(baseline / "config.yaml", temp / "config.yaml")
        selection = baseline / "data" / "backtest_selection.yaml"
        if selection.is_file():
            (temp / "data").mkdir()
            shutil.copy2(selection, temp / "data" / selection.name)
        capture_path = temp / "capture.pkl"
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(repo / "scripts" / "run_legacy_baseline.py"),
             "--baseline", str(baseline), "--benchmark", str(benchmark),
             "--workspace", str(temp), "--capture", str(capture_path)],
            cwd=temp, env=env, text=True, capture_output=True, check=False,
        )
        if completed.returncode:
            raise ComparisonError(
                "immutable baseline execution failed\n"
                + completed.stdout[-4000:] + "\n" + completed.stderr[-4000:]
            )
        capture = pickle.loads(capture_path.read_bytes())
    assert_baseline_unchanged(baseline, BASELINE_COMMIT, before)
    return capture


def _normalise_legacy(capture: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    predictions: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    rules: dict[str, dict[str, float]] = {}
    for code in sorted(capture["final"]):
        item = capture["final"][code]
        rule = item["rule"]
        rules[code] = {
            "target_percent": float(rule["target_percent"]),
            "stop_loss_percent": float(rule["stop_loss_percent"]),
            "threshold": float(rule["threshold"]),
        }
        prediction = item["predictions"].copy()
        prediction["code"] = code
        prediction = prediction.rename(columns={"date": "signal_date", "signal": "is_signal"})
        predictions.append(prediction[["code", "signal_date", "prob", "is_signal"]])
        trade = item["trades"].copy()
        if not trade.empty:
            trade["code"] = code
            trade["status"] = "FILLED"
            trade["stop_loss_percent"] = rules[code]["stop_loss_percent"]
            trades.append(trade)
    pred = pd.concat(predictions, ignore_index=True)
    pred["signal_date"] = pd.to_datetime(pred["signal_date"]).dt.strftime("%Y-%m-%d")
    pred = pred.sort_values(["signal_date", "code"], kind="mergesort").reset_index(drop=True)
    trade_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    for column in ["signal_date", "entry_date", "exit_date"]:
        if column in trade_frame:
            trade_frame[column] = pd.to_datetime(trade_frame[column]).dt.strftime("%Y-%m-%d")
    if not trade_frame.empty:
        trade_frame = trade_frame.sort_values(["signal_date", "code"], kind="mergesort").reset_index(drop=True)
    return pred, trade_frame, rules


def _candidate_rules(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path, dtype={"code": str})
    return {
        str(row.code): {
            "target_percent": float(row.target_percent),
            "stop_loss_percent": float(row.stop_loss_percent),
            "threshold": float(row.threshold),
        }
        for row in frame.itertuples()
    }


def _candidate_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(results_dir / "reference_trades.csv", dtype={"code": str})
    skipped = pd.read_csv(results_dir / "reference_skipped_orders.csv", dtype={"code": str})
    results = pd.concat([trades, skipped], ignore_index=True, sort=False)
    ledger = pd.read_csv(results_dir / "daily_ledger.csv")
    return results, ledger


def _data_audit(capture: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in capture["usage"]:
        latest[str(row["stock_code"])] = row
    expected_codes = sorted(str(code) for code in manifest["stock_codes"])
    if sorted(latest) != expected_codes:
        raise ComparisonError("baseline and candidate stock sets differ")
    audit = []
    for code in expected_codes:
        row = latest[code]
        metadata = manifest["files"][code]
        checks = {
            "first_date": metadata["first_date"], "last_date": metadata["last_date"],
            "rows": int(metadata["rows"]), "csv_sha256": metadata["sha256"],
        }
        if any(row[key] != value for key, value in checks.items()):
            raise ComparisonError(f"fixed data mismatch for {code}: {row} != {checks}")
        audit.append({"stock_code": code, **checks})
    return audit


def _scenario_rows(scenarios: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat = []
    by_stock = []
    for name, scenario in scenarios.items():
        metrics = scenario["metrics"]
        flat.append({
            "scenario": name,
            **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
            "skip_counts": json.dumps(metrics["skip_counts"], sort_keys=True, separators=(",", ":")),
        })
        results = scenario["results"]
        filled = results[results.get("status", pd.Series(dtype=str)) == "FILLED"]
        for code in sorted(set(metrics["profit_by_stock"]) | set(filled.get("code", []))):
            group = filled[filled["code"].astype(str) == str(code)]
            profits = pd.to_numeric(group.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0)
            by_stock.append({
                "scenario": name, "stock_code": code,
                "profit": float(profits.sum()), "trades": len(group),
                "win_rate": float((profits > 0).mean() * 100) if len(profits) else 0.0,
            })
    return pd.DataFrame(flat), pd.DataFrame(by_stock)


def _trade_alignment(
    legacy_predictions: pd.DataFrame, candidate_predictions: pd.DataFrame,
    legacy_trades: pd.DataFrame, candidate_results: pd.DataFrame,
    legacy_rules: dict[str, dict[str, float]], candidate_rules: dict[str, dict[str, float]],
) -> pd.DataFrame:
    lp = legacy_predictions.rename(columns={"prob": "legacy_probability", "is_signal": "legacy_signal"})
    vp = candidate_predictions.rename(columns={"prob": "v2_probability", "is_signal": "v2_signal"})
    base = lp.merge(vp[["code", "signal_date", "v2_probability", "v2_signal"]], on=["code", "signal_date"], how="outer")
    lt_columns = ["code", "signal_date", "entry_date", "exit_date", "exit_reason", "exit_price", "profit"]
    lt = legacy_trades.reindex(columns=lt_columns).rename(columns={
        "entry_date": "legacy_entry_date", "exit_date": "legacy_exit_date",
        "exit_reason": "legacy_exit_reason", "exit_price": "legacy_exit_price", "profit": "legacy_profit",
    })
    ct_columns = ["code", "signal_date", "entry_date", "exit_date", "exit_reason", "exit_price", "profit", "status"]
    ct = candidate_results.reindex(columns=ct_columns).rename(columns={
        "entry_date": "v2_entry_date", "exit_date": "v2_exit_date",
        "exit_reason": "v2_exit_reason", "exit_price": "v2_exit_price", "profit": "v2_profit", "status": "v2_order_status",
    })
    aligned = base.merge(lt, on=["code", "signal_date"], how="left").merge(ct, on=["code", "signal_date"], how="left")

    def category(row: pd.Series) -> str:
        ls, vs = bool(row.get("legacy_signal", False)), bool(row.get("v2_signal", False))
        if pd.isna(row.get("legacy_probability")) or pd.isna(row.get("v2_probability")):
            return "NO_MATCHING_SIGNAL"
        if ls != vs:
            lr, vr = legacy_rules[str(row.code)], candidate_rules[str(row.code)]
            if lr != vr:
                return "RULE_DIFFERENCE"
            return "LABEL_OR_MODEL_DIFFERENCE"
        status = str(row.get("v2_order_status", ""))
        if status == "SKIPPED_INSUFFICIENT_CASH":
            return "INSUFFICIENT_CASH"
        if status.startswith("SKIPPED_"):
            return "PORTFOLIO_CONFLICT"
        if not ls:
            return "SAME"
        legacy_reason, v2_reason = row.get("legacy_exit_reason"), row.get("v2_exit_reason")
        if legacy_reason == "STOP" and v2_reason == "STOP" and row.get("legacy_exit_price") != row.get("v2_exit_price"):
            return "GAP_STOP_DIFFERENCE"
        if legacy_reason != v2_reason or row.get("legacy_exit_date") != row.get("v2_exit_date"):
            return "STOP_EXECUTION_DIFFERENCE"
        if abs(float(row.get("legacy_profit", 0) or 0) - float(row.get("v2_profit", 0) or 0)) > 0.01:
            return "SLIPPAGE_OR_COMMISSION_DIFFERENCE"
        return "SAME"

    aligned["difference_category"] = aligned.apply(category, axis=1)
    columns = [
        "code", "signal_date", "legacy_probability", "v2_probability", "legacy_signal", "v2_signal",
        "legacy_entry_date", "v2_entry_date", "legacy_exit_date", "v2_exit_date",
        "legacy_exit_reason", "v2_exit_reason", "legacy_profit", "v2_profit",
        "v2_order_status", "difference_category",
    ]
    return aligned.reindex(columns=columns).sort_values(["signal_date", "code"], kind="mergesort")


def _report(scenarios: dict[str, dict[str, Any]], alignment: pd.DataFrame) -> str:
    a = scenarios["legacy_as_is"]["metrics"]
    d = scenarios["v2_full"]["metrics"]
    counts = alignment["difference_category"].value_counts().sort_index().to_dict()
    return f"""# Fixed baseline vs evaluator-v2 diagnostic report

## Scope and fairness

Both evaluators used the same immutable adjusted-OHLCV snapshot. The baseline code ran as-is at detached commit `{BASELINE_COMMIT}` with only its data fetch method temporarily replaced in-process. Network access was forbidden. evaluator-v2 outputs were read without changing selected rules.

The legacy research score uses validation profit *and research-internal test profit*, and its saved adoption result also inspects the reference period. Therefore `legacy_selection_uses_non_validation_data` is true. Its reference result is not a fair estimate of unknown-data performance. The v2 reference interval is also previously observed and is diagnostic only.

## Method differences

- A (`legacy_as_is`) preserves legacy labels, models, rules, stop behavior, and independent per-signal budgets.
- B (`legacy_signals_v2_portfolio`) holds legacy signals/rules fixed and applies v2 execution plus one shared portfolio.
- C (`v2_signals_independent_budget`) holds v2 signals/rules fixed but gives every signal an independent full budget; it is diagnostic only.
- D (`v2_full`) is the recorded v2 shared-portfolio result.

A-to-D differences contain interactions and are **not** claimed to be a complete additive attribution.

## Main observations

- Legacy as-is profit: {a['profit']:.2f}; trades: {a['trades']}; max drawdown: {a['max_drawdown']:.2f}.
- v2 full profit: {d['profit']:.2f}; trades: {d['trades']}; max drawdown: {d['max_drawdown']:.2f}.
- Legacy maximum simultaneously committed notional: {a['max_simultaneous_locked_capital']:.2f}.
- Legacy maximum capital overlap above one budget: {a['max_capital_overlap']:.2f}.
- Legacy maximum simultaneous position equivalents: {a['max_simultaneous_positions']}.
- Legacy trades participating in duplicated-capital intervals: {a['duplicate_capital_trade_count']}.
- Alignment categories: `{json.dumps(counts, sort_keys=True)}`.

Legacy can overstate attainable profit when concurrent trades each reuse the full budget, when competing same-day signals ignore rank and position limits, or when same-day exit proceeds are effectively reusable too early. It can also understate a trade when its normal-stop-price fill is worse than another convention in a non-gap case; gap-down handling can overstate fills because legacy fills at the stop rather than the lower opening-price basis. Commission and slippage rounding can move either direction.

Label/model differences are visible through changed probabilities and signals; stop-execution differences through stop categories and aligned exits; rule-selection changes through differing fixed rule tuples. Portfolio conflicts and insufficient cash explain signals that cannot become v2 trades.

The current v2 result establishes a feasible shared-cash execution path with deterministic ordering, but it does not establish superior predictive performance. Remaining uncertainty includes prior observation of the reference interval, model estimation error, market-impact realism beyond configured costs, and interactions among changed labels, rules, execution, and portfolio constraints.
"""


def run_comparison(repo: Path, baseline: Path, output: Path) -> dict[str, str]:
    before = verify_baseline(baseline, BASELINE_COMMIT)
    results_dir = repo / "data" / "backtest_results"
    benchmark_dir = repo / "data" / "benchmark"
    selected_path = results_dir / "selected_rules.csv"
    selected_hash = sha256_file(selected_path)
    recorded_candidate_summary = json.loads(
        (results_dir / "summary.json").read_text(encoding="utf-8")
    )
    if recorded_candidate_summary.get("candidate_commit") != CANDIDATE_COMMIT:
        raise ComparisonError(
            "candidate backtest output commit mismatch: "
            f"{recorded_candidate_summary.get('candidate_commit')} != {CANDIDATE_COMMIT}"
        )
    loader = FixedOHLCVLoader(benchmark_dir)
    manifest = loader.manifest
    config_bytes = (repo / "config.yaml").read_bytes()
    config = yaml.safe_load(config_bytes)
    stocks = sorted(str(item["code"]) for item in config["stocks"])
    prices = {code: loader.get_daily_stock_prices(code) for code in stocks}
    capture = _legacy_child(repo, baseline, benchmark_dir, before)
    audit = _data_audit(capture, manifest)
    if capture.get("network_attempts") != 0:
        raise ComparisonError("legacy reported a network attempt")

    legacy_predictions, legacy_trades, legacy_rules = _normalise_legacy(capture)
    candidate_predictions = pd.read_csv(results_dir / "reference_predictions.csv", dtype={"code": str})
    candidate_rules = _candidate_rules(selected_path)
    candidate_results, candidate_ledger = _candidate_results(results_dir)
    ai = config["ai_params"]
    portfolio_raw = config["portfolio_settings"]
    portfolio = PortfolioSettings(**portfolio_raw)
    budget = float(config["backtest_settings"]["budget"])
    calendar = sorted({date for frame in prices.values() for date in frame.index})

    legacy_orders = build_execution_orders(
        legacy_predictions, legacy_rules, prices, ai["future_days"], ai["commission_percent"],
        ai["entry_slippage_percent"], ai["exit_slippage_percent"], ai["stop_slippage_percent"],
    )
    b_results, b_ledger = run_v2_portfolio(legacy_orders, budget, portfolio, calendar)
    v2_orders = build_execution_orders(
        candidate_predictions, candidate_rules, prices, ai["future_days"], ai["commission_percent"],
        ai["entry_slippage_percent"], ai["exit_slippage_percent"], ai["stop_slippage_percent"],
    )
    c_results, c_ledger = run_independent_budget(v2_orders, budget, portfolio.lot_size)
    a_overlap = capital_overlap(legacy_trades, budget)
    a_ledger = pd.DataFrame({
        "date": a_overlap.get("date", pd.Series(dtype=str)),
        "equity": budget + a_overlap.get("cumulative_realized_profit", pd.Series(dtype=float)),
    })
    scenarios = {
        "legacy_as_is": {"results": legacy_trades, "ledger": a_ledger},
        "legacy_signals_v2_portfolio": {"results": b_results, "ledger": b_ledger},
        "v2_signals_independent_budget": {"results": c_results, "ledger": c_ledger},
        "v2_full": {"results": candidate_results, "ledger": candidate_ledger},
    }
    for scenario in scenarios.values():
        scenario["metrics"] = scenario_metrics(scenario["results"], scenario["ledger"], budget, prices)
    scenario_frame, stock_frame = _scenario_rows(scenarios)
    alignment = _trade_alignment(
        legacy_predictions, candidate_predictions, legacy_trades, candidate_results,
        legacy_rules, candidate_rules,
    )

    settings = config["backtest_settings"]
    comparison_manifest = {
        "baseline_commit": BASELINE_COMMIT, "candidate_commit": CANDIDATE_COMMIT,
        "snapshot_id": manifest["snapshot_id"], "snapshot_hash": manifest["snapshot_hash"],
        "config_hash": hashlib.sha256(config_bytes).hexdigest(), "seed": 42,
        "stock_codes": stocks,
        "research_period": {"from": settings["research_from"], "to": settings["research_to"]},
        "reference_period": {"from": settings["final_from"], "to": settings["final_to"]},
        "budget": budget, "future_days": ai["future_days"],
        "commission_percent": ai["commission_percent"],
        "entry_slippage_percent": ai["entry_slippage_percent"],
        "exit_slippage_percent": ai["exit_slippage_percent"],
        "stop_slippage_percent": ai["stop_slippage_percent"],
        "feature_columns": config["feature_columns"],
        "candidate_grid": {
            "target_percent": [1.0, 1.5, 2.0, 2.5, 3.0],
            "stop_loss_percent": [2.0, 3.0, 5.0],
            "threshold": [0.15, 0.20, 0.30, 0.40, 0.50],
        },
        "baseline_unavailable_v2_settings": {
            "lot_size": portfolio.lot_size,
            "max_position_percent": portfolio.max_position_percent,
            "max_open_positions": portfolio.max_open_positions,
            "shared_cash": True, "next_day_exit_proceeds": True,
            "gap_stop_uses_open": True, "label_includes_stop_and_commission": True,
        },
        "data_verification": audit,
    }
    research_selected = {
        code: capture["research"][code]["selected"] for code in sorted(capture["research"])
    }
    legacy_summary = {
        "legacy_selection_uses_non_validation_data": True,
        "selection_explanation": "ResearchScore includes validation and research-test profit; saved adoption status uses reference-period metrics.",
        "selected_rules": legacy_rules, "research_selection": research_selected,
        "saved_selection": capture["final_selection"],
        "reference_metrics": scenarios["legacy_as_is"]["metrics"],
        "network_attempts": capture["network_attempts"],
        "baseline_clean_before_and_after": True,
    }
    candidate_summary = recorded_candidate_summary
    candidate_summary["comparison_reference_metrics"] = scenarios["v2_full"]["metrics"]
    candidate_summary["selected_rules_sha256"] = selected_hash

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "comparison_manifest.json", comparison_manifest)
    _write_json(output / "legacy_summary.json", legacy_summary)
    _write_json(output / "candidate_summary.json", candidate_summary)
    _write_csv(output / "scenario_comparison.csv", scenario_frame)
    _write_csv(output / "comparison_by_stock.csv", stock_frame)
    _write_csv(output / "trade_alignment.csv", alignment)
    _write_csv(output / "capital_overlap.csv", a_overlap)
    _write_csv(output / "legacy_predictions.csv", legacy_predictions)
    _write_csv(output / "legacy_trades.csv", legacy_trades)
    (output / "difference_report.md").write_text(_report(scenarios, alignment), encoding="utf-8", newline="\n")
    if sha256_file(selected_path) != selected_hash:
        raise ComparisonError("selected_rules.csv changed during comparison")
    assert_baseline_unchanged(baseline, BASELINE_COMMIT, before)
    _write_json(output / "run_metadata.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_files_unchanged": True, "external_network_attempts": 0,
        "selected_rules_before_sha256": selected_hash,
        "selected_rules_after_sha256": sha256_file(selected_path),
        "deterministic_output_hashes": deterministic_hashes(output),
    })
    return deterministic_hashes(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-worktree", default=r"C:\taiki\hobbies\stock-analyzer-baseline")
    parser.add_argument("--output", default="data/backtest_comparison")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    hashes = run_comparison(repo, Path(args.baseline_worktree), repo / args.output)
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
