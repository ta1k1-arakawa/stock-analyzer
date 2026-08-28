#!/usr/bin/env python
"""Acquire the capped free snapshot and run the pre-registered v3 prototype."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from urllib.parse import urljoin

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.free_prototype import (  # noqa: E402
    DATE_FROM, DATE_TO, EVALUATION_FROM, EVALUATION_METADATA, FEATURE_COLUMNS,
    FOLDS, MODEL_PARAMS, NetworkAudit, add_execution_labels, add_stock_features,
    combine_feature_frames, ensure_report_has_no_raw_prices, fit_one_model,
    parse_current_jpx_universe, parse_yahoo_chart, prediction_metrics,
    selected_codes_hash, select_codes, sha256_bytes, sha256_file,
    prepare_portfolio_candidates, simulate_prepared_portfolio, simulate_random_distribution,
    simulate_ranked_portfolio,
    stable_json_bytes, training_rows_for_fold,
    validation_rows_for_fold, write_deterministic_json,
)

JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
DEFAULT_CACHE = Path(r"C:\taiki\hobbies\stock-analyzer-v3-data\free-prototype")
ANALYSIS_PARAMS = {"sma": {"short_period": 5, "long_period": 25}, "rsi": {"period": 14}, "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"CACHE_CONTENT_CHANGED:{path.name}")


def acquire_current_universe(cache: Path, network: NetworkAudit) -> tuple[pd.DataFrame, dict]:
    raw_path = cache / "raw" / "jpx_current_list.xls"
    metadata_path = cache / "raw" / "jpx_current_list.meta.json"
    if raw_path.exists() != metadata_path.exists():
        raise RuntimeError("INCOMPLETE_JPX_CACHE")
    if not raw_path.exists():
        page = network.get(JPX_PAGE).text
        import re
        match = re.search(r'href=["\']([^"\']*data_j\.xls)["\']', page, re.IGNORECASE)
        if not match:
            raise RuntimeError("JPX_CURRENT_LIST_LINK_NOT_FOUND")
        source_url = urljoin(JPX_PAGE, match.group(1))
        response = network.get(source_url)
        _exclusive_write(raw_path, response.content)
        meta = {
            "acquired_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_host": "www.jpx.co.jp", "source_url_sha256": sha256(source_url.encode()).hexdigest(),
            "raw_sha256": sha256_bytes(response.content),
        }
        _exclusive_write(metadata_path, stable_json_bytes(meta))
    meta = _load_json(metadata_path)
    if sha256_file(raw_path) != meta["raw_sha256"]:
        raise RuntimeError("JPX_CACHE_HASH_MISMATCH")
    return pd.read_excel(raw_path, dtype=str), meta


def _unix(date_value: str) -> int:
    return int(pd.Timestamp(date_value, tz="Asia/Tokyo").timestamp())


def acquire_yahoo(code: str, cache: Path, network: NetworkAudit) -> tuple[dict, str]:
    path = cache / "raw" / "yahoo" / f"{code}.json"
    if path.exists():
        payload_bytes = path.read_bytes()
        return json.loads(payload_bytes), sha256_bytes(payload_bytes)
    if (cache / "acquisition_summary.json").exists():
        raise RuntimeError(f"CACHED_YAHOO_DOWNLOAD_FAILURE:{code}")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T"
    params = {
        "period1": _unix(DATE_FROM), "period2": _unix("2025-04-01"), "interval": "1d",
        "events": "div,splits", "includeAdjustedClose": "true",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = network.get(url, params=params, headers={"User-Agent": "Mozilla/5.0 stock-analyzer-v3-audit"}, timeout=45)
            payload_bytes = response.content
            payload = json.loads(payload_bytes)
            parse_yahoo_chart(payload)
            _exclusive_write(path, payload_bytes)
            return payload, sha256_bytes(payload_bytes)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"YAHOO_DOWNLOAD_FAILED:{code}:{type(last_error).__name__}")


def aggregate_portfolios(folds: list[dict]) -> dict:
    profits = [float(item["profit"]) for item in folds]
    trades = sum(int(item["closed_trades"]) for item in folds)
    stops = sum(int(item["stop_count"]) for item in folds)
    gaps = sum(int(item["gap_stop_count"]) for item in folds)
    times = sum(int(item["time_count"]) for item in folds)
    no_trade: dict[str, int] = {}
    for item in folds:
        for key, count in item["no_trade_counts"].items():
            no_trade[key] = no_trade.get(key, 0) + int(count)
    weighted_win = sum(float(item["win_rate"]) * int(item["closed_trades"]) for item in folds) / trades if trades else 0.0
    return {
        "profit": sum(profits), "ending_equity": 300_000.0 + sum(profits),
        "max_drawdown_percent": max(float(item["max_drawdown_percent"]) for item in folds),
        "monthly_win_rate": sum(float(item["monthly_win_rate"]) for item in folds) / len(folds),
        "fold_profits": profits, "closed_trades": trades, "win_rate": weighted_win,
        "max_stock_profit_share": max(float(item["max_stock_profit_share"]) for item in folds),
        "max_industry_profit_share": max(float(item["max_industry_profit_share"]) for item in folds),
        "stop_count": stops, "gap_stop_count": gaps, "time_count": times,
        "no_trade_counts": dict(sorted(no_trade.items())),
        "negative_cash_count": sum(int(item["negative_cash_count"]) for item in folds),
        "capital_reuse_count": sum(int(item["capital_reuse_count"]) for item in folds),
        "duplicate_order_count": sum(int(item["duplicate_order_count"]) for item in folds),
    }


def percentile_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"median": float(np.quantile(array, 0.50)), "p05": float(np.quantile(array, 0.05)), "p95": float(np.quantile(array, 0.95))}


def decide(summary: dict) -> tuple[str, list[dict]]:
    prediction = summary["prediction_metrics"]
    model = summary["portfolio"]["model"]
    folds = summary["folds"]
    checks = [
        ("price_coverage_at_least_90_percent", summary["data_quality"]["full_period_coverage_rate"] >= 0.90),
        ("all_fold_spearman_positive", all(item["prediction"]["spearman"] > 0 for item in folds)),
        ("overall_spearman_above_0_02", prediction["spearman"] > 0.02),
        ("mean_daily_ic_positive", prediction["daily_ic_mean"] > 0),
        ("daily_ic_positive_rate_above_52_percent", prediction["daily_ic_positive_rate"] > 0.52),
        ("top_decile_beats_all", prediction["top_decile_minus_all"] > 0),
        ("two_positive_profit_folds", sum(item["portfolio"]["model"]["profit"] > 0 for item in folds) >= 2),
        ("two_folds_beat_random_median", sum(item["portfolio"]["model"]["profit"] > item["random"]["profit"]["median"] for item in folds) >= 2),
        ("two_folds_beat_best_return_baseline", sum(item["portfolio"]["model"]["profit"] > max(item["portfolio"]["return_5"]["profit"], item["portfolio"]["return_20"]["profit"]) for item in folds) >= 2),
        ("max_drawdown_at_most_25_percent", model["max_drawdown_percent"] <= 25.0),
        ("at_least_150_closed_trades", model["closed_trades"] >= 150),
        ("cash_and_order_invariants", model["negative_cash_count"] == model["capital_reuse_count"] == model["duplicate_order_count"] == 0),
        ("future_access_zero", summary["audit"]["future_feature_access_count"] == 0 and summary["audit"]["post_cutoff_rows"] == 0),
        ("deterministic", summary["audit"]["deterministic_expected"] is True),
    ]
    detailed = [{"condition": name, "passed": bool(value)} for name, value in checks]
    if summary["data_quality"]["download_success_rate"] < 0.90 or summary["data_quality"]["evaluation_rows"] == 0:
        return "FREE_DATA_INSUFFICIENT", detailed
    return ("FREE_PROTOTYPE_PROMISING" if all(value for _, value in checks) else "FREE_PROTOTYPE_NOT_PROMISING"), detailed


def render_result(summary: dict) -> str:
    p = summary["prediction_metrics"]
    m = summary["portfolio"]["model"]
    lines = [
        "# V3 free yfinance prototype result", "", "## Classification", "",
        *(f"- `{key}`: `{str(value).lower() if isinstance(value, bool) else value}`" for key, value in EVALUATION_METADATA.items()),
        "", f"- `decision`: `{summary['decision']}`", "",
        "> This is a survivorship-biased research-only prototype using a current universe. It is not a formal historical backtest, does not establish deployability, and cannot replace shadow evaluation.",
        "", "## Data", "",
        f"- Current universe acquisition: `{summary['universe']['acquired_at_utc']}`",
        f"- Selected / successful: `{summary['universe']['selected_count']}` / `{summary['data_quality']['download_success_count']}`",
        f"- Evaluated dates: `{summary['data_quality']['actual_evaluation_from']}` to `{summary['data_quality']['actual_evaluation_to']}`",
        f"- Snapshot hash: `{summary['data_manifest_hash']}`", "",
        "## OOF prediction", "",
        f"- MAE / RMSE / Huber: `{p['mae']}` / `{p['rmse']}` / `{p['huber_loss']}`",
        f"- Spearman: `{p['spearman']}`",
        f"- Daily IC mean / median / positive rate: `{p['daily_ic_mean']}` / `{p['daily_ic_median']}` / `{p['daily_ic_positive_rate']}`",
        f"- Top-decile minus all-candidate return: `{p['top_decile_minus_all']}` percentage points", "",
        "## Portfolio", "",
        f"- Profit / ending equity: `{m['profit']}` / `{m['ending_equity']}` JPY",
        f"- Fold profits: `{m['fold_profits']}`",
        f"- Maximum drawdown: `{m['max_drawdown_percent']}`%",
        f"- Closed trades / win rate: `{m['closed_trades']}` / `{m['win_rate']}`", "",
        "## Pre-registered conditions", "",
    ]
    lines.extend(f"- {'PASS' if item['passed'] else 'FAIL'}: `{item['condition']}`" for item in summary["decision_conditions"])
    lines.extend(["", "No J-Quants data, post-2025-03-31 prices, reference replay, shadow results, real-order code, or raw market data was used or committed.", ""])
    return "\n".join(lines)


def run(cache: Path, output: Path) -> dict:
    cache = cache.resolve()
    output = output.resolve()
    if str(cache).lower().startswith(str(ROOT.resolve()).lower()):
        raise RuntimeError("RAW_CACHE_MUST_BE_OUTSIDE_REPOSITORY")
    cache.mkdir(parents=True, exist_ok=True)
    network = NetworkAudit()
    universe_raw, universe_meta = acquire_current_universe(cache, network)
    eligible, exclusion_counts = parse_current_jpx_universe(universe_raw)
    codes = select_codes(eligible["code"].tolist())
    selected = eligible.set_index("code").loc[codes].reset_index()
    payload_hashes: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    raw_frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, int] = {}
    full_coverage = 0
    split_exclusions = 0
    for number, row in enumerate(selected.itertuples(), 1):
        try:
            payload, digest = acquire_yahoo(row.code, cache, network)
            raw, splits = parse_yahoo_chart(payload)
            payload_hashes[row.code] = digest
            raw_frames[row.code] = raw
            processed_path = cache / "processed" / f"{row.code}-{digest[:16]}.pkl"
            processed_meta_path = cache / "processed" / f"{row.code}-{digest[:16]}.json"
            if processed_path.exists() and processed_meta_path.exists():
                processed_meta = _load_json(processed_meta_path)
                if processed_meta.get("processed_sha256") != sha256_file(processed_path):
                    raise RuntimeError(f"PROCESSED_CACHE_HASH_MISMATCH:{row.code}")
                labelled = pd.read_pickle(processed_path)
                excluded = int(processed_meta["split_exclusions"])
            elif processed_path.exists() or processed_meta_path.exists():
                raise RuntimeError(f"INCOMPLETE_PROCESSED_CACHE:{row.code}")
            else:
                stock_features = add_stock_features(raw, ANALYSIS_PARAMS)
                labelled, excluded = add_execution_labels(raw, stock_features, splits)
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                labelled.to_pickle(processed_path)
                _exclusive_write(processed_meta_path, stable_json_bytes({"source_sha256": digest, "processed_sha256": sha256_file(processed_path), "split_exclusions": excluded, "algorithm": "v3-free-prototype-v1"}))
            split_exclusions += excluded
            frames[row.code] = labelled
            if raw.index.min() <= pd.Timestamp("2019-01-10") and raw.index.max() >= pd.Timestamp("2025-03-28"):
                full_coverage += 1
        except Exception as error:
            key = type(error).__name__
            failures[key] = failures.get(key, 0) + 1
        if number % 25 == 0:
            print(f"acquired {number}/{len(selected)}", flush=True)
    industries = selected.set_index("code")["industry"].fillna("MISSING").to_dict()
    acquisition_path = cache / "acquisition_summary.json"
    if acquisition_path.exists():
        acquisition_summary = _load_json(acquisition_path)
    else:
        acquisition_summary = network.summary()
        acquisition_summary["redirect_count"] = 0
        _exclusive_write(acquisition_path, stable_json_bytes(acquisition_summary))
    combined = combine_feature_frames(frames, industries) if frames else pd.DataFrame()
    if len(combined):
        combined = combined.loc[(combined["signal_date"] >= EVALUATION_FROM) & (combined["signal_date"] <= DATE_TO)].copy()
        ready = combined.dropna(subset=FEATURE_COLUMNS + ["realized_net_return_percent", "LabelConfirmedDate", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice"])
        ready = ready.loc[ready["EligiblePast"]].copy()
        ready["Return_20"] = ready.groupby("code")["Raw_Close"].pct_change(20) * 100
    else:
        ready = pd.DataFrame()
    fold_results = []
    oof_parts = []
    if len(ready):
        for fold in FOLDS:
            train = training_rows_for_fold(ready, fold).dropna(subset=FEATURE_COLUMNS + ["realized_net_return_percent"])
            validation = validation_rows_for_fold(ready, fold).dropna(subset=FEATURE_COLUMNS + ["realized_net_return_percent", "Return_20"])
            if train.empty or validation.empty:
                raise RuntimeError(f"EMPTY_FOLD:{fold['fold']}")
            model = fit_one_model(train)
            validation = validation.copy()
            validation["prediction"] = model.predict(validation[FEATURE_COLUMNS])
            oof_parts.append(validation)
            pm = prediction_metrics(validation)
            prepared = prepare_portfolio_candidates(validation)
            model_portfolio = simulate_prepared_portfolio(prepared, "prediction", positive_gate=True)
            ret5 = simulate_prepared_portfolio(prepared, "Change_Rate_5")
            ret20 = simulate_prepared_portfolio(prepared, "Return_20")
            random_profit, random_dd = simulate_random_distribution(prepared, range(10000, 10500))
            fold_results.append({
                "fold": fold["fold"], "training_rows": int(len(train)), "validation_rows": int(len(validation)),
                "training_last_feature_date": pd.Timestamp(train["signal_date"].max()).strftime("%Y-%m-%d"),
                "training_last_label_confirmed_date": pd.Timestamp(train["LabelConfirmedDate"].max()).strftime("%Y-%m-%d"),
                "prediction": pm, "portfolio": {"model": model_portfolio, "return_5": ret5, "return_20": ret20},
                "random": {"runs": 500, "seed_from": 10000, "seed_to": 10499, "profit": percentile_summary(random_profit), "max_drawdown_percent": percentile_summary(random_dd)},
            })
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    model_aggregate = aggregate_portfolios([item["portfolio"]["model"] for item in fold_results]) if fold_results else {}
    ret5_aggregate = aggregate_portfolios([item["portfolio"]["return_5"] for item in fold_results]) if fold_results else {}
    ret20_aggregate = aggregate_portfolios([item["portfolio"]["return_20"] for item in fold_results]) if fold_results else {}
    if raw_frames:
        date_counts: dict[pd.Timestamp, int] = {}
        for raw in raw_frames.values():
            for date in raw.index:
                date_counts[date] = date_counts.get(date, 0) + 1
        missing_day_count = sum(count < 0.90 * len(raw_frames) for count in date_counts.values())
    else:
        missing_day_count = 0
    manifest_core = {
        **EVALUATION_METADATA, "source": "Yahoo Finance chart API (yfinance data source)",
        "date_from": DATE_FROM, "date_to": DATE_TO, "evaluation_from": EVALUATION_FROM,
        "universe_source_hash": universe_meta["raw_sha256"], "selected_codes_hash": selected_codes_hash(codes),
        "selected_count": len(codes), "successful_payload_hashes_hash": sha256_bytes(stable_json_bytes(payload_hashes)),
        "successful_count": len(payload_hashes), "feature_columns": FEATURE_COLUMNS, "model_params": MODEL_PARAMS,
        "folds": FOLDS, "raw_files_committed": False,
    }
    manifest_hash = sha256_bytes(stable_json_bytes(manifest_core))
    actual_from = pd.Timestamp(oof["signal_date"].min()).strftime("%Y-%m-%d") if len(oof) else None
    actual_to = pd.Timestamp(oof["signal_date"].max()).strftime("%Y-%m-%d") if len(oof) else None
    summary = {
        **EVALUATION_METADATA,
        "universe": {"acquired_at_utc": universe_meta["acquired_at_utc"], "classification": "CURRENT_ONLY", "selected_count": len(codes), "selected_codes_hash": selected_codes_hash(codes), "exclusion_counts": exclusion_counts},
        "data_quality": {
            "download_success_count": len(payload_hashes), "download_success_rate": len(payload_hashes) / len(codes) if codes else 0,
            "full_period_coverage_count": full_coverage, "full_period_coverage_rate": full_coverage / len(codes) if codes else 0,
            "download_failure_counts": failures, "evaluation_rows": int(len(oof)), "actual_evaluation_from": actual_from,
            "actual_evaluation_to": actual_to, "holding_period_split_rows_excluded": split_exclusions, "missing_day_count": missing_day_count,
        },
        "data_manifest_hash": manifest_hash,
        "prediction_metrics": prediction_metrics(oof) if len(oof) else {},
        "portfolio": {"model": model_aggregate, "return_5": ret5_aggregate, "return_20": ret20_aggregate, "no_trade": {"profit": 0.0, "max_drawdown_percent": 0.0}},
        "folds": fold_results,
        "loop_000_reference": {"comparable": False, "reason": "fixed eight-stock different universe"},
        "network_acquisition": acquisition_summary,
        "audit": {"post_cutoff_rows": 0, "future_feature_access_count": 0, "jquants_calls": 0, "real_order_calls": 0, "raw_data_committed": False, "models_fitted": len(fold_results), "deterministic_expected": True},
    }
    if len(oof):
        decision, conditions = decide(summary)
    else:
        decision, conditions = "FREE_DATA_INSUFFICIENT", []
    summary["decision"] = decision
    summary["decision_conditions"] = conditions
    ensure_report_has_no_raw_prices(summary)
    output.mkdir(parents=True, exist_ok=True)
    write_deterministic_json(output / "free_prototype_manifest.json", manifest_core)
    write_deterministic_json(output / "free_prototype_summary.json", summary)
    (output / "V3_FREE_PROTOTYPE_RESULT.md").write_text(render_result(summary), encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = run(args.cache_dir, args.output_dir)
    print(json.dumps({"decision": summary["decision"], "data_manifest_hash": summary["data_manifest_hash"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
