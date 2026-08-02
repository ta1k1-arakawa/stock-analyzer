"""Deterministic research selection and shared-cash reference backtest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import lightgbm as lgb
import pandas as pd

from src.analysis import calculate_indicators, create_target_variable, sanitize_ohlcv
from src.benchmark import FixedOHLCVLoader, sha256_file
from src.config import AIParams, AppConfig, load_app
from src.trade_simulator import PortfolioSettings, simulate_execution, simulate_portfolio


TARGET_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
STOP_GRID = [2.0, 3.0, 5.0]
THRESHOLD_GRID = [0.15, 0.20, 0.30, 0.40, 0.50]
FOLDS = [(0.60, 0.70, 0.80), (0.70, 0.80, 0.90), (0.80, 0.90, 1.00)]
MODEL_SEED = 42
BASELINE_COMMIT = "2975e3375c615052bd3a1ab2e5a24e723e94c46b"
RESEARCH_FROM = "2020-01-01"
RESEARCH_TO = "2025-03-31"
REFERENCE_FROM = "2025-04-01"
REFERENCE_TO = "2026-05-20"
RESULT_DIR = Path("data/backtest_results")


@dataclass(frozen=True)
class Rule:
    code: str
    target_percent: float
    stop_loss_percent: float
    threshold: float
    validation_score: float


def _build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        random_state=MODEL_SEED,
        verbose=-1,
        force_col_wise=True,
        deterministic=True,
        n_jobs=1,
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    result = sanitize_ohlcv(df.copy())
    index = pd.to_datetime(result.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    result.index = index
    return result.sort_index()


def _labelled(
    df_ta: pd.DataFrame, ai: AIParams, target: float, stop: float,
) -> pd.DataFrame:
    result = create_target_variable(
        df_ta,
        ai.future_days,
        target,
        ai.entry_slippage_percent,
        ai.exit_slippage_percent,
        stop,
        ai.stop_slippage_percent,
        ai.commission_percent,
    )
    result["LabelConfirmedDate"] = pd.NaT
    column = result.columns.get_loc("LabelConfirmedDate")
    for position in range(len(result)):
        execution = simulate_execution(
            result,
            position,
            ai.future_days,
            stop,
            ai.entry_slippage_percent,
            ai.exit_slippage_percent,
            ai.stop_slippage_percent,
            ai.commission_percent,
        )
        if execution is not None:
            result.iloc[position, column] = result.index[execution.exit_index]
    return result


def eligible_training_rows(
    labelled: pd.DataFrame, prediction_date: str | pd.Timestamp, feature_columns: list[str],
) -> pd.DataFrame:
    """Return rows whose features and label outcome are known before prediction."""
    cutoff = pd.Timestamp(prediction_date)
    ready = labelled.dropna(subset=feature_columns + ["Target", "LabelConfirmedDate"])
    return ready[(ready.index < cutoff) & (ready["LabelConfirmedDate"] < cutoff)]


def _trade_metrics(
    df_ta: pd.DataFrame,
    dates: pd.Index,
    probabilities: Any,
    threshold: float,
    ai: AIParams,
    stop: float,
    budget: int,
) -> dict[str, float | int]:
    profits: list[float] = []
    for date, probability in zip(dates, probabilities):
        if float(probability) < threshold:
            continue
        position = int(df_ta.index.get_loc(date))
        execution = simulate_execution(
            df_ta, position, ai.future_days, stop,
            ai.entry_slippage_percent, ai.exit_slippage_percent,
            ai.stop_slippage_percent, ai.commission_percent,
        )
        if execution is None or execution.entry_price <= 0:
            continue
        quantity = int(budget / execution.entry_price)
        if quantity <= 0:
            continue
        profits.append(execution.return_percent / 100 * execution.entry_price * quantity)
    return {
        "profit": round(sum(profits), 8),
        "trades": len(profits),
        "wins": sum(value > 0 for value in profits),
    }


def research_candidate_rows(
    code: str,
    prices: pd.DataFrame,
    config: AppConfig,
    budget: int,
) -> pd.DataFrame:
    """Evaluate candidates; test metrics are diagnostic and never selection inputs."""
    ai = config.ai_params
    features = config.feature_columns
    research = prices[(prices.index >= RESEARCH_FROM) & (prices.index <= RESEARCH_TO)]
    df_ta = calculate_indicators(research, config.tech_params)
    rows: list[dict[str, Any]] = []
    for target in TARGET_GRID:
        for stop in STOP_GRID:
            labelled = _labelled(df_ta, ai, target, stop)
            usable = labelled.dropna(subset=features + ["Target", "LabelConfirmedDate"])
            for fold, (train_ratio, validation_ratio, test_ratio) in enumerate(FOLDS, 1):
                count = len(usable)
                train_end = int(count * train_ratio)
                validation_end = int(count * validation_ratio)
                test_end = int(count * test_ratio)
                validation = usable.iloc[train_end:validation_end]
                test = usable.iloc[validation_end:test_end]
                if len(validation) < 10 or len(test) < 10:
                    continue
                train = eligible_training_rows(usable.iloc[:train_end], validation.index[0], features)
                if len(train) < 100 or train["Target"].nunique() < 2:
                    continue
                model = _build_model()
                model.fit(train[features], train["Target"].astype(int))
                validation_probabilities = model.predict_proba(validation[features])[:, 1]
                test_probabilities = model.predict_proba(test[features])[:, 1]
                for threshold in THRESHOLD_GRID:
                    validation_metrics = _trade_metrics(
                        df_ta, validation.index, validation_probabilities,
                        threshold, ai, stop, budget,
                    )
                    test_metrics = _trade_metrics(
                        df_ta, test.index, test_probabilities,
                        threshold, ai, stop, budget,
                    )
                    rows.append(
                        {
                            "Code": code,
                            "Fold": fold,
                            "TargetPercent": target,
                            "StopLossPercent": stop,
                            "Threshold": threshold,
                            "ValidationProfit": validation_metrics["profit"],
                            "ValidationTrades": validation_metrics["trades"],
                            "ValidationWins": validation_metrics["wins"],
                            "TestProfitDiagnosticOnly": test_metrics["profit"],
                            "TestTradesDiagnosticOnly": test_metrics["trades"],
                            "TrainLastFeatureDate": train.index.max().strftime("%Y-%m-%d"),
                            "TrainLastLabelConfirmedDate": pd.Timestamp(train["LabelConfirmedDate"].max()).strftime("%Y-%m-%d"),
                            "ValidationFrom": validation.index.min().strftime("%Y-%m-%d"),
                            "ValidationTo": validation.index.max().strftime("%Y-%m-%d"),
                            "TestFromDiagnosticOnly": test.index.min().strftime("%Y-%m-%d"),
                            "TestToDiagnosticOnly": test.index.max().strftime("%Y-%m-%d"),
                        }
                    )
    return pd.DataFrame(rows)


def select_rule_from_diagnostics(code: str, diagnostics: pd.DataFrame, min_trades: int) -> Rule:
    """Select solely from validation columns; test/reference columns are ignored."""
    grouped = diagnostics.groupby(
        ["TargetPercent", "StopLossPercent", "Threshold"], as_index=False
    ).agg(
        ValidationProfit=("ValidationProfit", "sum"),
        ValidationTrades=("ValidationTrades", "sum"),
    )
    grouped["ValidationScore"] = (
        grouped["ValidationProfit"]
        - (grouped["ValidationTrades"] < min_trades).astype(int) * 50_000
    )
    selected = grouped.sort_values(
        ["ValidationScore", "TargetPercent", "StopLossPercent", "Threshold"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    return Rule(
        code=code,
        target_percent=float(selected["TargetPercent"]),
        stop_loss_percent=float(selected["StopLossPercent"]),
        threshold=float(selected["Threshold"]),
        validation_score=float(selected["ValidationScore"]),
    )


def _reference_predictions(
    code: str, prices: pd.DataFrame, config: AppConfig, rule: Rule,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ai = config.ai_params
    features = config.feature_columns
    df_ta = calculate_indicators(prices, config.tech_params)
    labelled = _labelled(df_ta, ai, rule.target_percent, rule.stop_loss_percent)
    signal_frame = df_ta.dropna(subset=features)
    predictions: list[dict[str, Any]] = []
    cutoffs: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    for signal_date, signal_row in signal_frame.iterrows():
        if signal_date < pd.Timestamp(REFERENCE_FROM) or signal_date > pd.Timestamp(REFERENCE_TO):
            continue
        train = eligible_training_rows(labelled, signal_date, features)
        if len(train) < 100 or train["Target"].nunique() < 2:
            continue
        last_feature = train.index.max()
        last_confirmed = pd.Timestamp(train["LabelConfirmedDate"].max())
        if last_confirmed >= signal_date:
            raise AssertionError("future label entered training data")
        model = _build_model()
        model.fit(train[features], train["Target"].astype(int))
        probability = float(model.predict_proba(signal_row[features].to_frame().T)[:, 1][0])
        common = {
            "code": code,
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "prob": round(probability, 8),
            "training_data_last_feature_date": last_feature.strftime("%Y-%m-%d"),
            "training_data_last_label_confirmed_date": last_confirmed.strftime("%Y-%m-%d"),
            "training_row_count": len(train),
            "model_seed": MODEL_SEED,
            "target_percent": rule.target_percent,
            "stop_loss_percent": rule.stop_loss_percent,
            "threshold": rule.threshold,
        }
        predictions.append({**common, "is_signal": probability >= rule.threshold})
        cutoffs.append(common)
        if probability < rule.threshold:
            continue
        signal_position = int(df_ta.index.get_loc(signal_date))
        execution = simulate_execution(
            df_ta, signal_position, ai.future_days, rule.stop_loss_percent,
            ai.entry_slippage_percent, ai.exit_slippage_percent,
            ai.stop_slippage_percent, ai.commission_percent,
        )
        entry_position = signal_position + 1
        planned_entry = df_ta.index[entry_position] if entry_position < len(df_ta) else signal_date
        order = {
            "code": code,
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "planned_entry_date": planned_entry.strftime("%Y-%m-%d"),
            "order_date": planned_entry.strftime("%Y-%m-%d"),
            "prob": round(probability, 8),
            "commission_percent": ai.commission_percent,
        }
        if execution is None or df_ta.index[execution.exit_index] > pd.Timestamp(REFERENCE_TO):
            orders.append({**order, "skip_reason": "SKIPPED_NO_FUTURE_DATA"})
            continue
        orders.append(
            {
                **order,
                "entry_date": df_ta.index[execution.entry_index].strftime("%Y-%m-%d"),
                "exit_date": df_ta.index[execution.exit_index].strftime("%Y-%m-%d"),
                "entry_price": round(execution.entry_price, 8),
                "exit_price": round(execution.exit_price, 8),
                "exit_reason": execution.exit_reason,
            }
        )
    return predictions, cutoffs, orders


def _git_state() -> tuple[str, bool, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    ).splitlines()
    relevant = [line for line in status_lines if "data/backtest_results/" not in line.replace("\\", "/")]
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)data/backtest_results"],
    )
    digest = hashlib.sha256(diff + "\n".join(relevant).encode("utf-8")).hexdigest()
    return commit, bool(relevant), digest


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    output = frame.reindex(columns=columns)
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.8f")


def _summary(
    results: pd.DataFrame,
    ledger: pd.DataFrame,
    manifest: dict[str, Any],
    config_hash: str,
    settings: PortfolioSettings,
) -> dict[str, Any]:
    trades = results[results["status"] == "FILLED"].copy() if not results.empty else results
    profits = pd.to_numeric(trades.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    equity = pd.to_numeric(ledger["equity"], errors="coerce")
    peak = equity.cummax()
    drawdown = equity - peak
    exit_dates = pd.to_datetime(trades.get("exit_date", pd.Series(dtype=str)))
    monthly = profits.groupby(exit_dates.dt.to_period("M")).sum() if len(profits) else pd.Series(dtype=float)
    commit, dirty, diff_hash = _git_state()
    skipped = results[results["status"] != "FILLED"] if not results.empty else results
    return {
        "baseline_commit": BASELINE_COMMIT,
        "candidate_commit": commit,
        "working_tree_dirty": dirty,
        "working_tree_diff_hash": diff_hash,
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_hash": manifest["snapshot_hash"],
        "config_hash": config_hash,
        "seed": MODEL_SEED,
        "periods": {
            "research_from": RESEARCH_FROM,
            "research_to": RESEARCH_TO,
            "reference_from": REFERENCE_FROM,
            "reference_to": REFERENCE_TO,
        },
        "portfolio_settings": asdict(settings),
        "profit": round(float(profits.sum()), 8),
        "trades": int(len(trades)),
        "win_rate": round(float((profits > 0).mean() * 100), 8) if len(profits) else 0.0,
        "max_drawdown": round(float(drawdown.min()), 8) if len(drawdown) else 0.0,
        "max_drawdown_percent": round(float((equity / peak - 1).min() * 100), 8) if len(equity) else 0.0,
        "monthly_win_rate": round(float((monthly > 0).mean() * 100), 8) if len(monthly) else 0.0,
        "profit_by_stock": {
            str(code): round(float(group["profit"].sum()), 8)
            for code, group in trades.groupby("code", sort=True)
        } if len(trades) else {},
        "skipped_by_reason": {
            str(reason): int(count)
            for reason, count in skipped["status"].value_counts().sort_index().items()
        } if len(skipped) else {},
    }


def run_backtest() -> dict[str, Any]:
    config, _ = load_app(log_file="backtest.log")
    raw = config.raw.get("backtest_settings", {})
    if any(
        str(raw.get(key)) != expected
        for key, expected in {
            "research_from": RESEARCH_FROM,
            "research_to": RESEARCH_TO,
            "final_from": REFERENCE_FROM,
            "final_to": REFERENCE_TO,
        }.items()
    ):
        raise ValueError("config period boundaries must match the frozen evaluator periods")
    portfolio_raw = config.raw.get("portfolio_settings", {})
    portfolio_settings = PortfolioSettings(
        lot_size=int(portfolio_raw.get("lot_size", 1)),
        max_position_percent=float(portfolio_raw.get("max_position_percent", 100.0)),
        max_open_positions=int(portfolio_raw.get("max_open_positions", 1)),
    )
    loader = FixedOHLCVLoader("data/benchmark")
    manifest = loader.manifest
    configured_codes = [stock.stock_code for stock in config.stocks]
    if configured_codes != manifest.get("stock_codes"):
        raise ValueError("configured stocks differ from benchmark manifest")
    prices = {
        code: _normalize(loader.get_daily_stock_prices(code, RESEARCH_FROM, REFERENCE_TO))
        for code in configured_codes
    }
    budget = int(raw.get("budget", config.ai_params.budget))
    min_trades = int(raw.get("min_research_trades", 10))
    all_diagnostics: list[pd.DataFrame] = []
    selected_rules: list[Rule] = []
    for stock in config.stocks:
        stock_config = config.for_stock(stock)
        diagnostics = research_candidate_rows(stock.stock_code, prices[stock.stock_code], stock_config, budget)
        if diagnostics.empty:
            raise RuntimeError(f"no research diagnostics for {stock.stock_code}")
        rule = select_rule_from_diagnostics(stock.stock_code, diagnostics, min_trades)
        selected_rules.append(rule)
        selected_only = diagnostics[
            (diagnostics["TargetPercent"] == rule.target_percent)
            & (diagnostics["StopLossPercent"] == rule.stop_loss_percent)
            & (diagnostics["Threshold"] == rule.threshold)
        ].copy()
        selected_only["SelectionBasis"] = "VALIDATION_ONLY"
        all_diagnostics.append(selected_only)

    predictions: list[dict[str, Any]] = []
    cutoffs: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    stock_map = {stock.stock_code: stock for stock in config.stocks}
    for rule in selected_rules:
        stock_config = config.for_stock(stock_map[rule.code])
        stock_predictions, stock_cutoffs, stock_orders = _reference_predictions(
            rule.code, prices[rule.code], stock_config, rule
        )
        predictions.extend(stock_predictions)
        cutoffs.extend(stock_cutoffs)
        orders.extend(stock_orders)
    predictions.sort(key=lambda row: (row["signal_date"], row["code"]))
    cutoffs.sort(key=lambda row: (row["signal_date"], row["code"]))
    orders.sort(key=lambda row: (row["order_date"], -row["prob"], row["code"]))
    calendar = sorted(
        set().union(*[
            set(frame[(frame.index >= REFERENCE_FROM) & (frame.index <= REFERENCE_TO)].index)
            for frame in prices.values()
        ])
    )
    result_rows, ledger_rows = simulate_portfolio(orders, budget, portfolio_settings, calendar)
    result_rows.sort(key=lambda row: (row.get("order_date", ""), -float(row["prob"]), row["code"]))
    trades = pd.DataFrame([row for row in result_rows if row["status"] == "FILLED"])
    skipped = pd.DataFrame([row for row in result_rows if row["status"] != "FILLED"])
    ledger = pd.DataFrame(ledger_rows)
    rules_frame = pd.DataFrame([asdict(rule) for rule in selected_rules]).sort_values("code")
    diagnostics_frame = pd.concat(all_diagnostics, ignore_index=True).sort_values(["Code", "Fold"])
    predictions_frame = pd.DataFrame(predictions)
    cutoffs_frame = pd.DataFrame(cutoffs)
    results_frame = pd.DataFrame(result_rows)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rules_frame, RESULT_DIR / "selected_rules.csv", ["code", "target_percent", "stop_loss_percent", "threshold", "validation_score"])
    _write_csv(diagnostics_frame, RESULT_DIR / "research_diagnostics.csv", list(diagnostics_frame.columns))
    prediction_columns = ["code", "signal_date", "prob", "is_signal", "training_data_last_feature_date", "training_data_last_label_confirmed_date", "training_row_count", "model_seed", "target_percent", "stop_loss_percent", "threshold"]
    _write_csv(predictions_frame, RESULT_DIR / "reference_predictions.csv", prediction_columns)
    trade_columns = ["code", "signal_date", "planned_entry_date", "entry_date", "exit_date", "prob", "entry_price", "exit_price", "exit_reason", "available_cash", "qty", "entry_commission", "exit_commission", "profit", "status"]
    _write_csv(trades, RESULT_DIR / "reference_trades.csv", trade_columns)
    skipped_columns = ["code", "signal_date", "planned_entry_date", "prob", "status", "available_cash"]
    _write_csv(skipped, RESULT_DIR / "reference_skipped_orders.csv", skipped_columns)
    _write_csv(ledger, RESULT_DIR / "daily_ledger.csv", ["date", "available_cash", "pending_cash", "locked_capital", "open_positions", "equity"])
    cutoff_columns = ["code", "signal_date", "training_data_last_feature_date", "training_data_last_label_confirmed_date", "training_row_count", "model_seed", "target_percent", "stop_loss_percent", "threshold"]
    _write_csv(cutoffs_frame, RESULT_DIR / "training_cutoffs.csv", cutoff_columns)
    config_hash = sha256_file(Path("config.yaml"))
    summary = _summary(results_frame, ledger, manifest, config_hash, portfolio_settings)
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (RESULT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run_backtest()
