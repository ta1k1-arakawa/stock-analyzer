from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
import yaml

from src.analysis import calculate_indicators, create_target_variable, sanitize_ohlcv
from src.config import AppConfig, load_app
from src.fetchers.yfinance import YFinanceFetcher


TARGET_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
STOP_GRID = [2.0, 3.0, 5.0]
THRESHOLD_GRID = [0.15, 0.20, 0.30, 0.40, 0.50]

# Fold ratios are applied only inside the research period.
FOLDS = [
    (0.60, 0.70, 0.80),
    (0.70, 0.80, 0.90),
    (0.80, 0.90, 1.00),
]


@dataclass(frozen=True)
class BacktestSettings:
    research_from: str
    research_to: str
    final_from: str
    final_to: str
    budget: int
    min_research_trades: int
    min_final_trades: int
    max_drawdown_percent: float
    min_month_win_rate: float
    max_single_month_profit_share: float
    result_path: Path


@dataclass(frozen=True)
class FixedRule:
    code: str
    target_percent: float
    stop_loss_percent: float
    threshold: float


def _date_after(date_str: str, days: int = 1) -> str:
    return (pd.Timestamp(date_str) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _resolve_auto_date(value: str | None) -> str | None:
    if value == "auto":
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    return value


def _load_backtest_settings(config: AppConfig) -> BacktestSettings:
    raw = config.raw.get("backtest_settings", {})
    training = config.training_settings

    research_from = raw.get("research_from") or training.get("data_from") or "2020-01-01"
    research_to = raw.get("research_to") or "2025-03-31"
    final_from = raw.get("final_from") or _date_after(research_to)
    final_to = raw.get("final_to") or _resolve_auto_date(training.get("data_to")) or "2026-05-20"

    return BacktestSettings(
        research_from=str(research_from),
        research_to=str(research_to),
        final_from=str(final_from),
        final_to=str(final_to),
        budget=int(raw.get("budget", config.ai_params.budget)),
        min_research_trades=int(raw.get("min_research_trades", 10)),
        min_final_trades=int(raw.get("min_final_trades", 10)),
        max_drawdown_percent=float(raw.get("max_drawdown_percent", 15.0)),
        min_month_win_rate=float(raw.get("min_month_win_rate", 50.0)),
        max_single_month_profit_share=float(raw.get("max_single_month_profit_share", 70.0)),
        result_path=Path(raw.get("result_path", "data/backtest_selection.yaml")),
    )


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    return df.sort_index()


def _slice_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    df = _normalize_index(df)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df[(df.index >= start_ts) & (df.index <= end_ts)]


def _build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        deterministic=True,
        n_jobs=1,
    )


def _simulate_one_signal(
    df: pd.DataFrame,
    signal_pos: int,
    prob: float,
    budget: float,
    future_days: int,
    stop_loss_pct: float,
    entry_slippage_pct: float,
    exit_slippage_pct: float,
    stop_slippage_pct: float,
    commission_pct: float,
) -> tuple[dict[str, Any] | None, float]:
    entry_idx = signal_pos + 1
    if entry_idx >= len(df):
        return None, budget

    exit_idx = entry_idx + future_days - 1
    if exit_idx >= len(df):
        return None, budget

    entry_price = float(df.iloc[entry_idx]["Open"]) * (1 + entry_slippage_pct / 100)
    if entry_price <= 0:
        return None, budget

    stop_price = entry_price * (1 - stop_loss_pct / 100)
    exit_price = None
    exit_reason = "TIME"
    actual_exit_idx = exit_idx

    for j in range(entry_idx, exit_idx + 1):
        if float(df.iloc[j]["Low"]) <= stop_price:
            exit_price = stop_price * (1 - stop_slippage_pct / 100)
            exit_reason = "STOP"
            actual_exit_idx = j
            break

    if exit_price is None:
        exit_price = float(df.iloc[exit_idx]["Close"]) * (1 - exit_slippage_pct / 100)

    qty = int(budget / entry_price)
    if qty <= 0:
        return None, budget

    gross_profit = (exit_price - entry_price) * qty
    commission = (entry_price + exit_price) * qty * commission_pct / 100
    profit = gross_profit - commission
    next_budget = budget + profit

    return (
        {
            "signal_date": df.index[signal_pos],
            "entry_date": df.index[entry_idx],
            "exit_date": df.index[actual_exit_idx],
            "prob": float(prob),
            "entry_price": round(entry_price, 4),
            "exit_price": round(float(exit_price), 4),
            "qty": qty,
            "profit": int(profit),
            "equity": int(next_budget),
            "win": profit > 0,
            "exit_reason": exit_reason,
        },
        next_budget,
    )


def _simulate_trades(
    df: pd.DataFrame,
    probs,
    threshold: float,
    future_days: int,
    stop_loss_pct: float,
    entry_slippage_pct: float,
    exit_slippage_pct: float,
    stop_slippage_pct: float,
    commission_pct: float,
    starting_budget: int,
) -> pd.DataFrame:
    budget = float(starting_budget)
    trades: list[dict[str, Any]] = []

    for i in range(len(df) - 1):
        prob = float(probs[i])
        if prob < threshold:
            continue

        trade, budget = _simulate_one_signal(
            df,
            i,
            prob,
            budget,
            future_days,
            stop_loss_pct,
            entry_slippage_pct,
            exit_slippage_pct,
            stop_slippage_pct,
            commission_pct,
        )
        if trade is not None:
            trades.append(trade)

    return pd.DataFrame(trades)


def _max_losing_streak(profits: pd.Series) -> int:
    worst = 0
    current = 0
    for profit in profits:
        if profit <= 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def _summarize_trades(trades: pd.DataFrame, starting_budget: int) -> dict[str, Any]:
    if trades.empty:
        return {
            "Profit": 0,
            "Trades": 0,
            "Wins": 0,
            "WinRate": 0.0,
            "MaxDD": 0,
            "MaxDDPct": 0.0,
            "WorstMonth": 0,
            "BestMonth": 0,
            "MonthWinRate": 0.0,
            "Months": 0,
            "LosingStreak": 0,
            "SingleMonthShare": 0.0,
        }

    profits = trades["profit"].astype(float)
    equity = starting_budget + profits.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    drawdown_pct = (equity / peak - 1) * 100

    exit_dates = pd.to_datetime(trades["exit_date"])
    monthly = profits.groupby(exit_dates.dt.to_period("M")).sum()
    best_month = int(monthly.max()) if not monthly.empty else 0
    worst_month = int(monthly.min()) if not monthly.empty else 0
    positive_months = int((monthly > 0).sum()) if not monthly.empty else 0
    month_win_rate = round(positive_months / len(monthly) * 100, 1) if len(monthly) else 0.0
    total_profit = int(profits.sum())
    single_month_share = round(best_month / total_profit * 100, 1) if total_profit > 0 and best_month > 0 else 0.0

    return {
        "Profit": total_profit,
        "Trades": int(len(trades)),
        "Wins": int((profits > 0).sum()),
        "WinRate": float(round((profits > 0).mean() * 100, 1)),
        "MaxDD": int(drawdown.min()),
        "MaxDDPct": round(float(drawdown_pct.min()), 1),
        "WorstMonth": worst_month,
        "BestMonth": best_month,
        "MonthWinRate": month_win_rate,
        "Months": int(len(monthly)),
        "LosingStreak": _max_losing_streak(profits),
        "SingleMonthShare": single_month_share,
    }


def _monthly_profit_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["Month", "Profit", "Trades", "Wins", "WinRate"])

    df = trades.copy()
    df["Month"] = pd.to_datetime(df["exit_date"]).dt.to_period("M").astype(str)
    monthly = (
        df.groupby("Month")
        .agg(
            Profit=("profit", "sum"),
            Trades=("profit", "count"),
            Wins=("win", "sum"),
        )
        .reset_index()
    )
    monthly["WinRate"] = (monthly["Wins"] / monthly["Trades"] * 100).round(1)
    return monthly


def _research_combo_rows(
    labeled_by_tp: dict[float, pd.DataFrame],
    feature_cols: list[str],
    config: AppConfig,
    settings: BacktestSettings,
) -> pd.DataFrame:
    ai = config.ai_params
    rows: list[dict[str, Any]] = []

    for tp, df_labeled in labeled_by_tp.items():
        for fold_no, ratios in enumerate(FOLDS, 1):
            tr_ratio, val_ratio, te_ratio = ratios
            m = len(df_labeled)
            tr_end = int(m * tr_ratio)
            val_end = int(m * val_ratio)
            te_end = int(m * te_ratio)

            train_label_end = max(tr_end - ai.future_days, 0)
            train_df = df_labeled.iloc[:train_label_end]
            val_df = df_labeled.iloc[tr_end:val_end]
            test_df = df_labeled.iloc[val_end:te_end]

            if len(train_df["Target"].unique()) < 2 or len(val_df) < 10 or len(test_df) < 10:
                continue

            model = _build_model()
            model.fit(train_df[feature_cols], train_df["Target"])
            val_probs = model.predict_proba(val_df[feature_cols])[:, 1]
            test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

            for stop in STOP_GRID:
                for threshold in THRESHOLD_GRID:
                    val_trades = _simulate_trades(
                        val_df,
                        val_probs,
                        threshold,
                        ai.future_days,
                        stop,
                        ai.entry_slippage_percent,
                        ai.exit_slippage_percent,
                        ai.stop_slippage_percent,
                        ai.commission_percent,
                        settings.budget,
                    )
                    test_trades = _simulate_trades(
                        test_df,
                        test_probs,
                        threshold,
                        ai.future_days,
                        stop,
                        ai.entry_slippage_percent,
                        ai.exit_slippage_percent,
                        ai.stop_slippage_percent,
                        ai.commission_percent,
                        settings.budget,
                    )
                    val = _summarize_trades(val_trades, settings.budget)
                    test = _summarize_trades(test_trades, settings.budget)
                    rows.append(
                        {
                            "fold": fold_no,
                            "tp": tp,
                            "stop": stop,
                            "thr": threshold,
                            "val_profit": val["Profit"],
                            "val_trades": val["Trades"],
                            "test_profit": test["Profit"],
                            "test_trades": test["Trades"],
                            "test_wins": test["Wins"],
                            "test_win_rate": test["WinRate"],
                            "test_max_dd": test["MaxDD"],
                            "test_max_dd_pct": test["MaxDDPct"],
                        }
                    )

    return pd.DataFrame(rows)


def _select_research_rule(code: str, combo_rows: pd.DataFrame, settings: BacktestSettings) -> dict[str, Any] | None:
    if combo_rows.empty:
        return None

    agg = (
        combo_rows.groupby(["tp", "stop", "thr"])
        .agg(
            ValProfit=("val_profit", "sum"),
            ValTrades=("val_trades", "sum"),
            TestProfit=("test_profit", "sum"),
            MinFold=("test_profit", "min"),
            Folds=("fold", "nunique"),
            Trades=("test_trades", "sum"),
            Wins=("test_wins", "sum"),
        )
        .reset_index()
    )
    folds_pos = (
        combo_rows.groupby(["tp", "stop", "thr"])["test_profit"]
        .apply(lambda s: int((s > 0).sum()))
        .rename("FoldsPosCount")
        .reset_index()
    )
    agg = agg.merge(folds_pos, on=["tp", "stop", "thr"], how="left")
    agg["WinRate"] = (agg["Wins"] / agg["Trades"] * 100).fillna(0).round(1)
    agg["ResearchScore"] = (
        agg["ValProfit"]
        + agg["TestProfit"]
        + agg["MinFold"].clip(upper=0) * 2
        - (agg["Trades"] < settings.min_research_trades).astype(int) * 50_000
    )

    selected = agg.sort_values(
        ["ResearchScore", "FoldsPosCount", "MinFold", "ValProfit", "Trades"],
        ascending=[False, False, False, False, False],
    ).iloc[0]

    return {
        "Code": code,
        "TargetPercent": float(selected["tp"]),
        "StopLossPercent": float(selected["stop"]),
        "Threshold": float(selected["thr"]),
        "ResearchScore": int(selected["ResearchScore"]),
        "ValProfit": int(selected["ValProfit"]),
        "TestProfit": int(selected["TestProfit"]),
        "MinFold": int(selected["MinFold"]),
        "FoldsPosCount": int(selected["FoldsPosCount"]),
        "FoldsPos": f"{int(selected['FoldsPosCount'])}/{int(selected['Folds'])}",
        "Trades": int(selected["Trades"]),
        "WinRate": float(selected["WinRate"]),
    }


def _evaluate_research_stock(
    code: str,
    df_prices: pd.DataFrame,
    config: AppConfig,
    settings: BacktestSettings,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    ai = config.ai_params
    feature_cols = config.feature_columns
    df_research = _slice_dates(df_prices, settings.research_from, settings.research_to)

    if df_research.empty:
        return None, pd.DataFrame()

    df_ta = calculate_indicators(df_research, config.tech_params)
    labeled_by_tp: dict[float, pd.DataFrame] = {}

    for tp in TARGET_GRID:
        df_labeled = create_target_variable(
            df_ta,
            ai.future_days,
            tp,
            ai.entry_slippage_percent,
            ai.exit_slippage_percent,
        )
        df_labeled = df_labeled.dropna(subset=feature_cols + ["Target"])
        if len(df_labeled) >= 100:
            labeled_by_tp[tp] = df_labeled

    if not labeled_by_tp:
        return None, pd.DataFrame()

    combo_rows = _research_combo_rows(labeled_by_tp, feature_cols, config, settings)
    selected = _select_research_rule(code, combo_rows, settings)
    return selected, combo_rows


def _final_evaluation_rolling(
    df_prices: pd.DataFrame,
    rule: FixedRule,
    config: AppConfig,
    settings: BacktestSettings,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ai = config.ai_params
    feature_cols = config.feature_columns

    df_full = _slice_dates(df_prices, settings.research_from, settings.final_to)
    df_ta = calculate_indicators(df_full, config.tech_params)
    df_labeled = create_target_variable(
        df_ta,
        ai.future_days,
        rule.target_percent,
        ai.entry_slippage_percent,
        ai.exit_slippage_percent,
    )
    df_labeled = df_labeled.dropna(subset=feature_cols + ["Target"])

    final_start = pd.Timestamp(settings.final_from)
    final_end = pd.Timestamp(settings.final_to)
    budget = float(settings.budget)
    trades: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    df_signal = df_ta.dropna(subset=feature_cols)
    for signal_date, signal_row in df_signal.iterrows():
        if signal_date < final_start or signal_date > final_end:
            continue

        signal_pos = df_ta.index.get_loc(signal_date)
        exit_pos = signal_pos + ai.future_days
        if exit_pos >= len(df_ta) or df_ta.index[exit_pos] > final_end:
            continue

        label_cutoff_pos = signal_pos - ai.future_days
        if label_cutoff_pos < 0:
            continue
        label_cutoff_date = df_ta.index[label_cutoff_pos]
        train_df = df_labeled[df_labeled.index <= label_cutoff_date]

        if len(train_df) < 100 or len(train_df["Target"].unique()) < 2:
            continue

        model = _build_model()
        model.fit(train_df[feature_cols], train_df["Target"])
        prob = float(model.predict_proba(signal_row[feature_cols].to_frame().T)[:, 1][0])
        is_signal = prob >= rule.threshold
        predictions.append({"date": signal_date, "prob": prob, "signal": is_signal})

        if not is_signal:
            continue

        trade, budget = _simulate_one_signal(
            df_ta,
            signal_pos,
            prob,
            budget,
            ai.future_days,
            rule.stop_loss_percent,
            ai.entry_slippage_percent,
            ai.exit_slippage_percent,
            ai.stop_slippage_percent,
            ai.commission_percent,
        )
        if trade is not None:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)
    predictions_df = pd.DataFrame(predictions)
    summary = _summarize_trades(trades_df, settings.budget)
    return summary, trades_df, predictions_df


def _adoption_checks(summary: dict[str, Any], settings: BacktestSettings) -> tuple[str, list[str]]:
    checks = [
        ("profit_positive", summary["Profit"] > 0),
        ("enough_trades", summary["Trades"] >= settings.min_final_trades),
        ("drawdown_ok", abs(summary["MaxDDPct"]) <= settings.max_drawdown_percent),
        ("monthly_stability_ok", summary["MonthWinRate"] >= settings.min_month_win_rate),
        (
            "not_one_month_dependent",
            summary["SingleMonthShare"] <= settings.max_single_month_profit_share or summary["Profit"] <= 0,
        ),
    ]
    failed = [name for name, ok in checks if not ok]
    if not failed:
        return "PASS", []
    if "profit_positive" in failed or "enough_trades" in failed:
        return "REJECT", failed
    return "REVIEW", failed


def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _fixed_rule_from_row(row: pd.Series) -> FixedRule:
    return FixedRule(
        code=str(row["Code"]),
        target_percent=float(row["TargetPercent"]),
        stop_loss_percent=float(row["StopLossPercent"]),
        threshold=float(row["Threshold"]),
    )


def _print_final_result(
    rule: FixedRule,
    summary: dict[str, Any],
    trades: pd.DataFrame,
    predictions: pd.DataFrame,
    status: str,
    failed_checks: list[str],
) -> None:
    _print_header(f"Final evaluation: {rule.code}")
    print(f"Locked rule           : target={rule.target_percent}, stop={rule.stop_loss_percent}, threshold={rule.threshold}")
    print(f"Adoption status       : {status}")
    if failed_checks:
        print(f"Failed checks         : {', '.join(failed_checks)}")
    print(f"Predicted days        : {len(predictions)}")
    print(f"Signals / trades      : {summary['Trades']}")
    print(f"Profit                : {summary['Profit']:+,d}")
    print(f"Max drawdown          : {summary['MaxDD']:+,d} ({summary['MaxDDPct']}%)")
    print(f"Win rate              : {summary['WinRate']}%")
    print(f"Monthly win rate      : {summary['MonthWinRate']}% ({summary['Months']} months)")
    print(f"Worst month           : {summary['WorstMonth']:+,d}")
    print(f"Best month            : {summary['BestMonth']:+,d}")
    print(f"Max losing streak     : {summary['LosingStreak']}")
    print(f"Best-month dependency : {summary['SingleMonthShare']}% of total profit")
    print("\nMonthly result")
    print(_monthly_profit_table(trades).to_string(index=False))


def _save_selection_result(
    final_results: pd.DataFrame,
    settings: BacktestSettings,
) -> str | None:
    passed = final_results[final_results["Status"] == "PASS"].copy()
    passed = passed.sort_values("ResearchScore", ascending=False)
    recommended_code = str(passed.iloc[0]["Code"]) if not passed.empty else None

    paper_test_rules = final_results[
        final_results["Status"].isin(["PASS", "REVIEW"])
    ].copy()
    rules = {
        str(row["Code"]): {
            "target_percent": float(row["TargetPercent"]),
            "stop_loss_percent": float(row["StopLossPercent"]),
            "threshold": float(row["Threshold"]),
        }
        for _, row in paper_test_rules.iterrows()
    }
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recommended_stock_code": recommended_code,
        "selection_method": "highest_research_score_among_final_pass",
        "rules": rules,
    }

    settings.result_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.result_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    return recommended_code


def run_backtest() -> None:
    config, _ = load_app(log_file="backtest.log")
    ai = config.ai_params
    settings = _load_backtest_settings(config)
    stock_configs = {stock.stock_code: config.for_stock(stock) for stock in config.stocks}
    candidates = list(stock_configs)

    _print_header("Backtest: research selection + locked final evaluation")
    print(f"Research period : {settings.research_from} to {settings.research_to}")
    print(f"Final period    : {settings.final_from} to {settings.final_to}")
    print(f"Candidates      : {', '.join(candidates)}")
    print(f"Future days     : {ai.future_days}")
    print(f"Budget          : {settings.budget:,}")
    print(f"Feature columns : {', '.join(config.feature_columns)}")
    print(
        "Cost model      : "
        f"commission={ai.commission_percent:.3f}%, "
        f"entry_slippage={ai.entry_slippage_percent:.3f}%, "
        f"exit_slippage={ai.exit_slippage_percent:.3f}%, "
        f"stop_slippage={ai.stop_slippage_percent:.3f}%"
    )
    print(
        "Grid            : "
        f"target={TARGET_GRID}, stop={STOP_GRID}, threshold={THRESHOLD_GRID}"
    )
    print(
        "Adoption checks : "
        f"profit>0, trades>={settings.min_final_trades}, "
        f"max_dd<={settings.max_drawdown_percent:.1f}%, "
        f"month_win_rate>={settings.min_month_win_rate:.1f}%, "
        f"best_month_share<={settings.max_single_month_profit_share:.1f}%"
    )

    fetcher = YFinanceFetcher()
    fetch_from = settings.research_from
    fetch_to = _date_after(settings.final_to)

    research_rows: list[dict[str, Any]] = []
    price_cache: dict[str, pd.DataFrame] = {}

    _print_header("1) Research selection")
    for code in candidates:
        print(f"-- {code} --")
        try:
            df_prices = fetcher.get_daily_stock_prices(code, fetch_from, fetch_to)
        except Exception as exc:
            print(f"  error: {exc}")
            continue

        if df_prices is None or df_prices.empty:
            print("  skipped: no data")
            continue

        df_prices = sanitize_ohlcv(_normalize_index(df_prices))
        price_cache[code] = df_prices

        selected, _ = _evaluate_research_stock(
            code,
            df_prices,
            stock_configs[code],
            settings,
        )
        if selected is None:
            print("  skipped: not enough valid research data")
            continue

        research_rows.append(selected)
        print(
            "  selected in research: "
            f"tp={selected['TargetPercent']} "
            f"stop={selected['StopLossPercent']} "
            f"thr={selected['Threshold']} "
            f"score={selected['ResearchScore']:+,d} "
            f"research_test={selected['TestProfit']:+,d} "
            f"trades={selected['Trades']} "
            f"folds_pos={selected['FoldsPos']}"
        )

    if not research_rows:
        print("No research result.")
        return

    df_research = pd.DataFrame(research_rows).sort_values(
        ["ResearchScore", "FoldsPosCount", "MinFold", "ValProfit", "Trades"],
        ascending=[False, False, False, False, False],
    )

    _print_header("Research summary")
    print(df_research.to_string(index=False))

    df_final_candidates = df_research.copy()

    _print_header("2) Locked rules for per-stock final evaluation")
    print(
        df_final_candidates[
            ["Code", "TargetPercent", "StopLossPercent", "Threshold"]
        ].to_string(index=False)
    )
    print("Final evaluation will not search or adjust these values.")

    final_rows: list[dict[str, Any]] = []
    for _, selected in df_final_candidates.iterrows():
        rule = _fixed_rule_from_row(selected)
        if rule.code not in price_cache:
            df_prices = fetcher.get_daily_stock_prices(rule.code, fetch_from, fetch_to)
            if df_prices is None or df_prices.empty:
                final_rows.append(
                    {
                        "Code": rule.code,
                        "TargetPercent": rule.target_percent,
                        "StopLossPercent": rule.stop_loss_percent,
                        "Threshold": rule.threshold,
                        "ResearchScore": int(selected["ResearchScore"]),
                        "Status": "ERROR",
                        "FailedChecks": "no_final_data",
                    }
                )
                continue
            price_cache[rule.code] = sanitize_ohlcv(_normalize_index(df_prices))

        try:
            final_summary, final_trades, final_predictions = _final_evaluation_rolling(
                price_cache[rule.code],
                rule,
                stock_configs[rule.code],
                settings,
            )
        except Exception as exc:
            print(f"  final evaluation error for {rule.code}: {exc}")
            final_rows.append(
                {
                    "Code": rule.code,
                    "TargetPercent": rule.target_percent,
                    "StopLossPercent": rule.stop_loss_percent,
                    "Threshold": rule.threshold,
                    "ResearchScore": int(selected["ResearchScore"]),
                    "Status": "ERROR",
                    "FailedChecks": "final_evaluation_error",
                }
            )
            continue
        status, failed_checks = _adoption_checks(final_summary, settings)
        _print_final_result(
            rule,
            final_summary,
            final_trades,
            final_predictions,
            status,
            failed_checks,
        )
        final_rows.append(
            {
                "Code": rule.code,
                "TargetPercent": rule.target_percent,
                "StopLossPercent": rule.stop_loss_percent,
                "Threshold": rule.threshold,
                "ResearchScore": int(selected["ResearchScore"]),
                "Status": status,
                "Profit": final_summary["Profit"],
                "Trades": final_summary["Trades"],
                "WinRate": final_summary["WinRate"],
                "MaxDDPct": final_summary["MaxDDPct"],
                "MonthWinRate": final_summary["MonthWinRate"],
                "SingleMonthShare": final_summary["SingleMonthShare"],
                "FailedChecks": ",".join(failed_checks),
            }
        )

    df_final = pd.DataFrame(final_rows)
    _print_header("3) Per-stock final evaluation summary")
    print(df_final.to_string(index=False))

    paper_test_rules = df_final[df_final["Status"].isin(["PASS", "REVIEW"])]
    _print_header("Paper-test rules (PASS and REVIEW)")
    if paper_test_rules.empty:
        print("No stock qualified for a paper-test rule.")
    else:
        for _, row in paper_test_rules.iterrows():
            print(f"- code: \"{row['Code']}\"  # {row['Status']}")
            print("  ai_params:")
            print(f"    target_percent: {row['TargetPercent']}")
            print(f"    threshold: {row['Threshold']}")
            print(f"    stop_loss_percent: {row['StopLossPercent']}")

    recommended_code = _save_selection_result(df_final, settings)
    _print_header("Recommended stock")
    if recommended_code:
        print(f"Recommended: {recommended_code}")
        print("Method  : highest research score among stocks that passed final checks")
    else:
        print("No stock passed all final checks.")
    print("Slack notification targets are controlled by notify_slack in config.yaml.")
    print(f"Saved   : {settings.result_path}")


if __name__ == "__main__":
    run_backtest()
