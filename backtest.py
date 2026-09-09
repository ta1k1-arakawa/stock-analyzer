"""Deterministic research selection and shared-cash reference backtest."""

from __future__ import annotations

import argparse
import csv
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
from src.benchmark import (
    REQUIRED_COLUMNS, FixedOHLCVLoader, sha256_file, snapshot_hash,
)
from src.config import AIParams, AppConfig, load_app
from src.reproducibility import CONFIG_HASH_METHOD, config_hash
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
LOOP_RESULT_DIR = Path("data/loop_validation_results")


class BlindValidationViolation(RuntimeError):
    """Raised when blind validation attempts to cross an allowed-data boundary."""


class LoopValidationPriceSource:
    """Validate the snapshot but parse OHLCV rows only through research end."""

    def __init__(self, source: str | Path | Any) -> None:
        self._delegate = source if hasattr(source, "get_daily_stock_prices") else None
        if self._delegate is not None:
            self.root = None
            self.manifest = source.manifest
            return
        self.root = Path(source)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = self.manifest.get("files", {})
        if self.manifest.get("columns") != REQUIRED_COLUMNS or not files:
            raise BlindValidationViolation("invalid fixed benchmark manifest")
        for code, metadata in files.items():
            path = self.root / "ohlcv" / f"{code}.csv"
            if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
                raise BlindValidationViolation(f"fixed benchmark mismatch for {code}")
        if snapshot_hash(files) != self.manifest.get("snapshot_hash"):
            raise BlindValidationViolation("fixed benchmark snapshot hash mismatch")

    def get_daily_stock_prices(
        self, stock_code: str, date_from_str: str, date_to_str: str,
    ) -> pd.DataFrame:
        requested_from = pd.Timestamp(date_from_str)
        requested_to = pd.Timestamp(date_to_str)
        if requested_from < pd.Timestamp(RESEARCH_FROM) or requested_to > pd.Timestamp(RESEARCH_TO):
            raise BlindValidationViolation(
                "loop-validation price access is limited to "
                f"{RESEARCH_FROM}..{RESEARCH_TO}: requested "
                f"{requested_from.date()}..{requested_to.date()}"
            )
        if self._delegate is not None:
            frame = self._delegate.get_daily_stock_prices(
                stock_code, date_from_str, date_to_str
            )
        else:
            code = str(stock_code).removesuffix(".T")
            if code not in self.manifest["files"]:
                raise BlindValidationViolation(f"stock is not in fixed benchmark: {code}")
            rows: list[dict[str, Any]] = []
            previous_date: pd.Timestamp | None = None
            with (self.root / "ohlcv" / f"{code}.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != REQUIRED_COLUMNS:
                    raise BlindValidationViolation(f"invalid fixed CSV columns for {code}")
                for raw_row in reader:
                    row_date = pd.Timestamp(raw_row["Date"])
                    if row_date > pd.Timestamp(RESEARCH_TO):
                        break
                    if previous_date is not None and row_date <= previous_date:
                        raise BlindValidationViolation(f"invalid date ordering for {code}")
                    previous_date = row_date
                    if row_date < requested_from or row_date > requested_to:
                        continue
                    rows.append({
                        "Date": row_date,
                        **{
                            column: pd.to_numeric(raw_row[column], errors="raise")
                            for column in REQUIRED_COLUMNS[1:]
                        },
                    })
            frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS).set_index("Date")
        if not frame.empty and pd.Timestamp(frame.index.max()) > pd.Timestamp(RESEARCH_TO):
            raise BlindValidationViolation("loader returned a row on or after 2025-04-01")
        return frame


@dataclass(frozen=True)
class Rule:
    code: str
    target_percent: float
    stop_loss_percent: float
    threshold: float
    validation_score: float


@dataclass(frozen=True)
class PredictionCacheKey:
    stock_code: str
    fold: int
    target_percent: float
    stop_loss_percent: float
    feature_columns: tuple[str, ...]
    seed: int
    training_cutoff: str


@dataclass
class CachedValidationPrediction:
    key: PredictionCacheKey
    validation_from: str
    validation_to: str
    validation_dates: list[str]
    orders: list[dict[str, Any]]
    training_row_count: int
    training_last_feature_date: str
    training_last_label_confirmed_date: str


@dataclass(frozen=True)
class SelectionConstraints:
    min_trades: int
    max_drawdown_percent: float
    min_month_win_rate: float
    max_stock_profit_share: float


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


def _independent_metrics(
    orders: list[dict[str, Any]], threshold: float, budget: int,
) -> dict[str, float | int]:
    profits: list[float] = []
    for order in orders:
        if float(order["prob"]) < threshold or order.get("skip_reason"):
            continue
        entry_price = float(order["entry_price"])
        if entry_price <= 0:
            continue
        quantity = int(budget / entry_price)
        if quantity <= 0:
            continue
        profits.append(float(order["return_percent"]) / 100 * entry_price * quantity)
    return {
        "profit": round(sum(profits), 8),
        "trades": len(profits),
        "wins": sum(value > 0 for value in profits),
    }


def _validation_order(
    code: str,
    df_until_validation_end: pd.DataFrame,
    signal_date: pd.Timestamp,
    probability: float,
    ai: AIParams,
    stop: float,
) -> dict[str, Any]:
    signal_position = int(df_until_validation_end.index.get_loc(signal_date))
    entry_position = signal_position + 1
    planned_entry = (
        df_until_validation_end.index[entry_position]
        if entry_position < len(df_until_validation_end) else signal_date
    )
    base = {
        "code": code,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "planned_entry_date": planned_entry.strftime("%Y-%m-%d"),
        "order_date": planned_entry.strftime("%Y-%m-%d"),
        "prob": round(float(probability), 8),
        "commission_percent": ai.commission_percent,
    }
    execution = simulate_execution(
        df_until_validation_end, signal_position, ai.future_days, stop,
        ai.entry_slippage_percent, ai.exit_slippage_percent,
        ai.stop_slippage_percent, ai.commission_percent,
    )
    if execution is None:
        return {**base, "skip_reason": "SKIPPED_NO_FUTURE_DATA"}
    return {
        **base,
        "entry_date": df_until_validation_end.index[execution.entry_index].strftime("%Y-%m-%d"),
        "exit_date": df_until_validation_end.index[execution.exit_index].strftime("%Y-%m-%d"),
        "entry_price": round(execution.entry_price, 8),
        "exit_price": round(execution.exit_price, 8),
        "exit_reason": execution.exit_reason,
        "return_percent": round(execution.return_percent, 8),
    }


def research_candidate_cache(
    code: str,
    prices: pd.DataFrame,
    config: AppConfig,
    budget: int,
) -> tuple[pd.DataFrame, dict[PredictionCacheKey, CachedValidationPrediction]]:
    """Train once per target/stop/fold and cache validation probabilities."""
    ai = config.ai_params
    features = config.feature_columns
    research = prices[(prices.index >= RESEARCH_FROM) & (prices.index <= RESEARCH_TO)]
    df_ta = calculate_indicators(research, config.tech_params)
    rows: list[dict[str, Any]] = []
    cache: dict[PredictionCacheKey, CachedValidationPrediction] = {}
    for target in TARGET_GRID:
        for stop in STOP_GRID:
            labelled = _labelled(df_ta, ai, target, stop)
            usable = labelled.dropna(subset=features + ["Target", "LabelConfirmedDate"])
            for fold, (train_ratio, validation_ratio, _test_ratio) in enumerate(FOLDS, 1):
                count = len(usable)
                train_end = int(count * train_ratio)
                validation_end = int(count * validation_ratio)
                validation = usable.iloc[train_end:validation_end]
                if len(validation) < 10:
                    continue
                train = eligible_training_rows(usable.iloc[:train_end], validation.index[0], features)
                if len(train) < 100 or train["Target"].nunique() < 2:
                    continue
                model = _build_model()
                model.fit(train[features], train["Target"].astype(int))
                validation_probabilities = model.predict_proba(validation[features])[:, 1]
                training_cutoff = validation.index[0].strftime("%Y-%m-%d")
                key = PredictionCacheKey(
                    stock_code=code,
                    fold=fold,
                    target_percent=target,
                    stop_loss_percent=stop,
                    feature_columns=tuple(features),
                    seed=MODEL_SEED,
                    training_cutoff=training_cutoff,
                )
                validation_only_prices = df_ta[df_ta.index <= validation.index.max()]
                orders = [
                    _validation_order(
                        code, validation_only_prices, signal_date, probability, ai, stop
                    )
                    for signal_date, probability in zip(validation.index, validation_probabilities)
                ]
                cache[key] = CachedValidationPrediction(
                    key=key,
                    validation_from=validation.index.min().strftime("%Y-%m-%d"),
                    validation_to=validation.index.max().strftime("%Y-%m-%d"),
                    validation_dates=[date.strftime("%Y-%m-%d") for date in validation.index],
                    orders=orders,
                    training_row_count=len(train),
                    training_last_feature_date=train.index.max().strftime("%Y-%m-%d"),
                    training_last_label_confirmed_date=pd.Timestamp(
                        train["LabelConfirmedDate"].max()
                    ).strftime("%Y-%m-%d"),
                )
                for threshold in THRESHOLD_GRID:
                    validation_metrics = _independent_metrics(orders, threshold, budget)
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
                            "TrainLastFeatureDate": train.index.max().strftime("%Y-%m-%d"),
                            "TrainLastLabelConfirmedDate": pd.Timestamp(train["LabelConfirmedDate"].max()).strftime("%Y-%m-%d"),
                            "ValidationFrom": validation.index.min().strftime("%Y-%m-%d"),
                            "ValidationTo": validation.index.max().strftime("%Y-%m-%d"),
                        }
                    )
    return pd.DataFrame(rows), cache


def research_candidate_rows(
    code: str, prices: pd.DataFrame, config: AppConfig, budget: int,
) -> pd.DataFrame:
    """Compatibility wrapper returning independent validation diagnostics."""
    rows, _ = research_candidate_cache(code, prices, config, budget)
    return rows


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


def _rule_signature(rules: dict[str, Rule]) -> tuple[tuple[str, float, float, float], ...]:
    return tuple(
        (code, rules[code].target_percent, rules[code].stop_loss_percent, rules[code].threshold)
        for code in sorted(rules)
    )


def _cache_entry(
    cache: dict[PredictionCacheKey, CachedValidationPrediction], code: str,
    fold: int, rule: Rule,
) -> CachedValidationPrediction:
    matches = [
        value for key, value in cache.items()
        if key.stock_code == code and key.fold == fold
        and key.target_percent == rule.target_percent
        and key.stop_loss_percent == rule.stop_loss_percent
    ]
    if len(matches) != 1:
        raise RuntimeError(f"validation cache lookup failed: {code} fold={fold} rule={rule}")
    return matches[0]


def assert_validation_folds_non_overlapping(
    cache: dict[PredictionCacheKey, CachedValidationPrediction],
) -> None:
    fold_dates: dict[int, set[pd.Timestamp]] = {}
    for entry in cache.values():
        fold_dates.setdefault(entry.key.fold, set()).update(
            pd.Timestamp(value) for value in entry.validation_dates
        )
    folds = sorted(fold_dates)
    for index, left in enumerate(folds):
        for right in folds[index + 1:]:
            overlap = fold_dates[left] & fold_dates[right]
            if overlap:
                raise ValueError(
                    f"validation folds overlap: {left} and {right} at {min(overlap).date()}"
                )


def _portfolio_fold_metrics(
    fold: int,
    results: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
    trades = [row for row in results if row["status"] == "FILLED"]
    skipped = [row for row in results if row["status"] != "FILLED"]
    profits = pd.Series([float(row.get("profit", 0.0)) for row in trades], dtype=float)
    equity = pd.Series(
        [float(budget)] + [float(row["equity"]) for row in ledger], dtype=float
    )
    drawdown_percent = (equity / equity.cummax() - 1) * 100
    return {
        "fold": fold,
        "profit": round(float(profits.sum()), 8),
        "trades": len(trades),
        "wins": int((profits > 0).sum()),
        "max_drawdown_percent": round(abs(float(drawdown_percent.min())), 8),
        "skip_counts": {
            str(reason): int(count)
            for reason, count in pd.Series(
                [row["status"] for row in skipped], dtype="object"
            ).value_counts().sort_index().items()
        },
    }


def classify_validation_metrics(
    metrics: dict[str, Any], constraints: SelectionConstraints,
) -> dict[str, Any]:
    checks = {
        "total_profit_positive": metrics["total_profit"] > 0,
        "positive_folds": metrics["positive_folds"] >= 2,
        "enough_trades": metrics["trade_count"] >= constraints.min_trades,
        "drawdown_ok": metrics["worst_max_drawdown_percent"] <= constraints.max_drawdown_percent,
        "monthly_stability_ok": metrics["month_win_rate"] >= constraints.min_month_win_rate,
        "stock_dependency_ok": metrics["max_stock_profit_share"] <= constraints.max_stock_profit_share,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**metrics, "selection_status": "PASS" if not failed else "REVIEW", "failed_checks": failed, "failed_check_count": len(failed)}


def evaluate_validation_portfolio(
    rules: dict[str, Rule],
    cache: dict[PredictionCacheKey, CachedValidationPrediction],
    budget: int,
    portfolio_settings: PortfolioSettings,
    constraints: SelectionConstraints,
) -> dict[str, Any]:
    """Evaluate only cached validation predictions, resetting cash each fold."""
    assert_validation_folds_non_overlapping(cache)
    by_fold: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []
    for fold in sorted({key.fold for key in cache}):
        orders: list[dict[str, Any]] = []
        calendar: set[pd.Timestamp] = set()
        for code in sorted(rules):
            rule = rules[code]
            entry = _cache_entry(cache, code, fold, rule)
            calendar.update(pd.Timestamp(value) for value in entry.validation_dates)
            orders.extend(
                {**order, "validation_fold": fold}
                for order in entry.orders if float(order["prob"]) >= rule.threshold
            )
        orders.sort(key=lambda row: (row["order_date"], -float(row["prob"]), row["code"]))
        results, ledger = simulate_portfolio(
            orders, budget, portfolio_settings, sorted(calendar)
        )
        fold_metrics = _portfolio_fold_metrics(fold, results, ledger, budget)
        by_fold.append(fold_metrics)
        all_trades.extend({**row, "validation_fold": fold} for row in results if row["status"] == "FILLED")
        all_skipped.extend({**row, "validation_fold": fold} for row in results if row["status"] != "FILLED")

    trade_frame = pd.DataFrame(all_trades)
    profits = pd.to_numeric(trade_frame.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    exit_dates = pd.to_datetime(trade_frame.get("exit_date", pd.Series(dtype=str)))
    monthly = profits.groupby(exit_dates.dt.to_period("M")).sum() if len(profits) else pd.Series(dtype=float)
    stock_profit = (
        trade_frame.assign(profit=profits).groupby("code", sort=True)["profit"].sum()
        if len(trade_frame) else pd.Series(dtype=float)
    )
    positive_stock_profit = stock_profit[stock_profit > 0]
    positive_total = float(positive_stock_profit.sum())
    dependency = (
        float(positive_stock_profit.max()) / positive_total * 100
        if positive_total > 0 else 0.0
    )
    skip_counts = {
        str(reason): int(count)
        for reason, count in pd.Series(
            [row["status"] for row in all_skipped], dtype="object"
        ).value_counts().sort_index().items()
    }
    metrics = {
        "total_profit": round(sum(row["profit"] for row in by_fold), 8),
        "positive_folds": sum(row["profit"] > 0 for row in by_fold),
        "min_fold_profit": round(min(row["profit"] for row in by_fold), 8),
        "worst_max_drawdown_percent": round(max(row["max_drawdown_percent"] for row in by_fold), 8),
        "month_win_rate": round(float((monthly > 0).mean() * 100), 8) if len(monthly) else 0.0,
        "max_stock_profit_share": round(dependency, 8),
        "trade_count": len(all_trades),
        "skip_counts": skip_counts,
        "profit_by_stock": {str(code): round(float(value), 8) for code, value in stock_profit.items()},
        "by_fold": by_fold,
        "trades": all_trades,
        "skipped": all_skipped,
    }
    return classify_validation_metrics(metrics, constraints)


def _selection_key(
    metrics: dict[str, Any], rules: dict[str, Rule],
) -> tuple[Any, ...]:
    return (
        0 if metrics["selection_status"] == "PASS" else 1,
        0 if metrics["selection_status"] == "PASS" else metrics["failed_check_count"],
        -metrics["total_profit"],
        -metrics["min_fold_profit"],
        metrics["worst_max_drawdown_percent"],
        -metrics["month_win_rate"],
        metrics["max_stock_profit_share"],
        -metrics["trade_count"],
        _rule_signature(rules),
    )


def candidate_rules_from_diagnostics(
    code: str, diagnostics: pd.DataFrame,
) -> list[Rule]:
    grouped = diagnostics.groupby(
        ["TargetPercent", "StopLossPercent", "Threshold"], as_index=False
    )["ValidationProfit"].sum()
    return [
        Rule(code, float(row.TargetPercent), float(row.StopLossPercent),
             float(row.Threshold), float(row.ValidationProfit))
        for row in grouped.sort_values(
            ["TargetPercent", "StopLossPercent", "Threshold"], kind="mergesort"
        ).itertuples(index=False)
    ]


def coordinate_select_rules(
    initial_rules: dict[str, Rule],
    candidates: dict[str, list[Rule]],
    cache: dict[PredictionCacheKey, CachedValidationPrediction],
    budget: int,
    portfolio_settings: PortfolioSettings,
    constraints: SelectionConstraints,
    max_passes: int = 3,
) -> tuple[dict[str, Rule], dict[str, Any], list[dict[str, Any]], int, int]:
    rules = dict(initial_rules)
    trace: list[dict[str, Any]] = []
    evaluations = 0
    completed_passes = 0
    for pass_number in range(1, max_passes + 1):
        changed = False
        completed_passes = pass_number
        for code in sorted(rules):
            previous = rules[code]
            evaluated: list[tuple[tuple[Any, ...], Rule, dict[str, Any], dict[str, Rule]]] = []
            for candidate in candidates[code]:
                candidate_set = dict(rules)
                candidate_set[code] = candidate
                metrics = evaluate_validation_portfolio(
                    candidate_set, cache, budget, portfolio_settings, constraints
                )
                evaluations += 1
                evaluated.append((_selection_key(metrics, candidate_set), candidate, metrics, candidate_set))
            evaluated.sort(key=lambda item: item[0])
            _, best_rule, best_metrics, _ = evaluated[0]
            accepted_change = (
                best_rule.target_percent,
                best_rule.stop_loss_percent,
                best_rule.threshold,
            ) != (
                previous.target_percent,
                previous.stop_loss_percent,
                previous.threshold,
            )
            if accepted_change:
                rules[code] = best_rule
                changed = True
            for _, candidate, metrics, _ in evaluated:
                trace.append(
                    {
                        "pass_number": pass_number,
                        "stock_code": code,
                        "previous_rule": json.dumps(asdict(previous), sort_keys=True),
                        "candidate_rule": json.dumps(asdict(candidate), sort_keys=True),
                        "candidate_metrics": json.dumps(
                            {key: value for key, value in metrics.items() if key not in {"trades", "skipped"}},
                            sort_keys=True,
                        ),
                        "accepted": bool(accepted_change and candidate == best_rule),
                        "reason": (
                            "ACCEPTED_BEST_PORTFOLIO"
                            if accepted_change and candidate == best_rule
                            else "RETAINED_CURRENT_RULE"
                            if not accepted_change and candidate == best_rule
                            else "NOT_BEST_PORTFOLIO"
                        ),
                    }
                )
        if not changed:
            break
    final_metrics = evaluate_validation_portfolio(
        rules, cache, budget, portfolio_settings, constraints
    )
    return rules, final_metrics, trace, completed_passes, evaluations


def research_test_diagnostics_after_selection(
    code: str,
    prices: pd.DataFrame,
    config: AppConfig,
    rule: Rule,
    budget: int,
) -> pd.DataFrame:
    """Read the research test slices only after portfolio rule selection ends."""
    ai = config.ai_params
    features = config.feature_columns
    research = prices[(prices.index >= RESEARCH_FROM) & (prices.index <= RESEARCH_TO)]
    df_ta = calculate_indicators(research, config.tech_params)
    labelled = _labelled(df_ta, ai, rule.target_percent, rule.stop_loss_percent)
    usable = labelled.dropna(subset=features + ["Target", "LabelConfirmedDate"])
    rows: list[dict[str, Any]] = []
    for fold, (train_ratio, validation_ratio, test_ratio) in enumerate(FOLDS, 1):
        count = len(usable)
        train_end = int(count * train_ratio)
        validation_end = int(count * validation_ratio)
        test_end = int(count * test_ratio)
        validation = usable.iloc[train_end:validation_end]
        test = usable.iloc[validation_end:test_end]
        train = eligible_training_rows(usable.iloc[:train_end], validation.index[0], features)
        if len(train) < 100 or len(test) < 10 or train["Target"].nunique() < 2:
            continue
        model = _build_model()
        model.fit(train[features], train["Target"].astype(int))
        probabilities = model.predict_proba(test[features])[:, 1]
        test_only_prices = df_ta[df_ta.index <= test.index.max()]
        orders = [
            _validation_order(code, test_only_prices, date, probability, ai, rule.stop_loss_percent)
            for date, probability in zip(test.index, probabilities)
        ]
        metrics = _independent_metrics(orders, rule.threshold, budget)
        rows.append(
            {
                "Code": code,
                "Fold": fold,
                "TestProfitDiagnosticOnly": metrics["profit"],
                "TestTradesDiagnosticOnly": metrics["trades"],
                "TestFromDiagnosticOnly": test.index.min().strftime("%Y-%m-%d"),
                "TestToDiagnosticOnly": test.index.max().strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


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


def sort_skipped_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared deterministic ordering for skipped-order artifacts."""
    if frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(
        ["signal_date", "planned_entry_date", "prob", "code", "status"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def loop_validation_summary(
    validation: dict[str, Any], normalized_config_hash: str | None = None,
) -> dict[str, Any]:
    """Expose validation-portfolio metrics only; ignore every diagnostic field."""
    summary = {
        "fold_profits": [
            {"fold": int(row["fold"]), "profit": round(float(row["profit"]), 8)}
            for row in sorted(validation["by_fold"], key=lambda row: int(row["fold"]))
        ],
        "max_drawdown_percent": round(float(validation["worst_max_drawdown_percent"]), 8),
        "max_stock_profit_share": round(float(validation["max_stock_profit_share"]), 8),
        "monthly_win_rate": round(float(validation["month_win_rate"]), 8),
        "profit": round(float(validation["total_profit"]), 8),
        "skip_counts": {
            str(reason): int(count)
            for reason, count in sorted(validation["skip_counts"].items())
        },
        "trade_count": int(validation["trade_count"]),
    }
    if normalized_config_hash is not None:
        summary.update(
            config_hash=normalized_config_hash,
            config_hash_method=CONFIG_HASH_METHOD,
        )
    return summary


def _write_loop_validation(summary: dict[str, Any]) -> None:
    LOOP_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in LOOP_RESULT_DIR.iterdir()
        if path.name != "summary.json"
    )
    if unexpected:
        raise BlindValidationViolation(
            f"unexpected files in blind output directory: {unexpected}"
        )
    (LOOP_RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary(
    results: pd.DataFrame,
    ledger: pd.DataFrame,
    manifest: dict[str, Any],
    config_hash: str,
    settings: PortfolioSettings,
    validation: dict[str, Any],
    coordinate_passes: int,
    coordinate_evaluations: int,
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
        "config_hash_method": CONFIG_HASH_METHOD,
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
        "selection_status": validation["selection_status"],
        "validation_total_profit": validation["total_profit"],
        "validation_positive_folds": validation["positive_folds"],
        "validation_min_fold_profit": validation["min_fold_profit"],
        "validation_worst_max_drawdown_percent": validation["worst_max_drawdown_percent"],
        "validation_month_win_rate": validation["month_win_rate"],
        "validation_max_stock_profit_share": validation["max_stock_profit_share"],
        "validation_trade_count": validation["trade_count"],
        "validation_skip_counts": validation["skip_counts"],
        "validation_failed_check_count": validation["failed_check_count"],
        "validation_failed_checks": validation["failed_checks"],
        "coordinate_passes": coordinate_passes,
        "coordinate_evaluations": coordinate_evaluations,
    }


def run_backtest(mode: str = "full") -> dict[str, Any]:
    if mode not in {"full", "loop-validation"}:
        raise ValueError(f"unsupported backtest mode: {mode}")
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
    loader = (
        LoopValidationPriceSource("data/benchmark")
        if mode == "loop-validation" else FixedOHLCVLoader("data/benchmark")
    )
    manifest = loader.manifest
    configured_codes = [stock.stock_code for stock in config.stocks]
    if configured_codes != manifest.get("stock_codes"):
        raise ValueError("configured stocks differ from benchmark manifest")
    research_prices = {
        code: _normalize(loader.get_daily_stock_prices(code, RESEARCH_FROM, RESEARCH_TO))
        for code in configured_codes
    }
    budget = int(raw.get("budget", config.ai_params.budget))
    constraints = SelectionConstraints(
        min_trades=int(raw.get("min_research_trades", 10)),
        max_drawdown_percent=float(raw.get("max_drawdown_percent", 15.0)),
        min_month_win_rate=float(raw.get("min_month_win_rate", 50.0)),
        max_stock_profit_share=float(raw.get("max_stock_profit_share", 70.0)),
    )
    diagnostics_by_code: dict[str, pd.DataFrame] = {}
    prediction_cache: dict[PredictionCacheKey, CachedValidationPrediction] = {}
    initial_rules: dict[str, Rule] = {}
    candidates_by_code: dict[str, list[Rule]] = {}
    for stock in config.stocks:
        stock_config = config.for_stock(stock)
        diagnostics, stock_cache = research_candidate_cache(
            stock.stock_code, research_prices[stock.stock_code], stock_config, budget
        )
        if diagnostics.empty:
            raise RuntimeError(f"no research diagnostics for {stock.stock_code}")
        diagnostics_by_code[stock.stock_code] = diagnostics
        prediction_cache.update(stock_cache)
        initial_rules[stock.stock_code] = select_rule_from_diagnostics(
            stock.stock_code, diagnostics, constraints.min_trades
        )
        candidates_by_code[stock.stock_code] = candidate_rules_from_diagnostics(
            stock.stock_code, diagnostics
        )

    assert_validation_folds_non_overlapping(prediction_cache)
    selected_rule_map, validation_metrics, selection_trace, coordinate_passes, coordinate_evaluations = coordinate_select_rules(
        initial_rules,
        candidates_by_code,
        prediction_cache,
        budget,
        portfolio_settings,
        constraints,
        max_passes=3,
    )
    selected_rules = [selected_rule_map[code] for code in sorted(selected_rule_map)]
    if mode == "loop-validation":
        summary = loop_validation_summary(validation_metrics, config_hash(Path("config.yaml")))
        _write_loop_validation(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return summary

    all_diagnostics: list[pd.DataFrame] = []
    stock_map = {stock.stock_code: stock for stock in config.stocks}
    for rule in selected_rules:
        diagnostics = diagnostics_by_code[rule.code]
        selected_only = diagnostics[
            (diagnostics["TargetPercent"] == rule.target_percent)
            & (diagnostics["StopLossPercent"] == rule.stop_loss_percent)
            & (diagnostics["Threshold"] == rule.threshold)
        ].copy()
        selected_only["SelectionBasis"] = "VALIDATION_PORTFOLIO_COORDINATE_SEARCH"
        test_diagnostics = research_test_diagnostics_after_selection(
            rule.code,
            research_prices[rule.code],
            config.for_stock(stock_map[rule.code]),
            rule,
            budget,
        )
        selected_only = selected_only.merge(
            test_diagnostics, on=["Code", "Fold"], how="left", validate="one_to_one"
        )
        all_diagnostics.append(selected_only)

    # Reference rows are exposed only after validation selection and research-test
    # diagnostics have fully completed.
    prices = {
        code: _normalize(loader.get_daily_stock_prices(code, RESEARCH_FROM, REFERENCE_TO))
        for code in configured_codes
    }

    predictions: list[dict[str, Any]] = []
    cutoffs: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
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
    skipped = sort_skipped_orders(
        pd.DataFrame([row for row in result_rows if row["status"] != "FILLED"])
    )
    ledger = pd.DataFrame(ledger_rows)
    rules_frame = pd.DataFrame([asdict(rule) for rule in selected_rules]).sort_values("code")
    diagnostics_frame = pd.concat(all_diagnostics, ignore_index=True).sort_values(["Code", "Fold"])
    predictions_frame = pd.DataFrame(predictions)
    cutoffs_frame = pd.DataFrame(cutoffs)
    results_frame = pd.DataFrame(result_rows)
    validation_by_fold = pd.DataFrame(validation_metrics["by_fold"]).sort_values("fold")
    validation_trades = pd.DataFrame(validation_metrics["trades"])
    validation_skipped = sort_skipped_orders(pd.DataFrame(validation_metrics["skipped"]))
    trace_frame = pd.DataFrame(selection_trace)

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
    _write_csv(
        trace_frame,
        RESULT_DIR / "portfolio_selection_trace.csv",
        ["pass_number", "stock_code", "previous_rule", "candidate_rule", "candidate_metrics", "accepted", "reason"],
    )
    _write_csv(
        validation_by_fold,
        RESULT_DIR / "validation_portfolio_by_fold.csv",
        ["fold", "profit", "trades", "wins", "max_drawdown_percent", "skip_counts"],
    )
    validation_trade_columns = ["validation_fold", "code", "signal_date", "planned_entry_date", "entry_date", "exit_date", "prob", "entry_price", "exit_price", "exit_reason", "available_cash", "qty", "entry_commission", "exit_commission", "profit", "status"]
    _write_csv(validation_trades, RESULT_DIR / "validation_portfolio_trades.csv", validation_trade_columns)
    validation_skip_columns = ["validation_fold", "code", "signal_date", "planned_entry_date", "prob", "status", "available_cash"]
    _write_csv(validation_skipped, RESULT_DIR / "validation_portfolio_skipped_orders.csv", validation_skip_columns)
    selection_summary = {
        key: value for key, value in validation_metrics.items()
        if key not in {"trades", "skipped"}
    }
    selection_summary.update(
        {
            "coordinate_passes": coordinate_passes,
            "coordinate_evaluations": coordinate_evaluations,
            "initial_rules": {code: asdict(initial_rules[code]) for code in sorted(initial_rules)},
            "selected_rules": {code: asdict(selected_rule_map[code]) for code in sorted(selected_rule_map)},
        }
    )
    (RESULT_DIR / "portfolio_selection_summary.json").write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    normalized_config_hash = config_hash(Path("config.yaml"))
    summary = _summary(
        results_frame, ledger, manifest, normalized_config_hash, portfolio_settings,
        validation_metrics, coordinate_passes, coordinate_evaluations,
    )
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (RESULT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    global RESULT_DIR, LOOP_RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["full", "loop-validation"], default="full",
        help="full diagnostics or blind research-validation-only evaluation",
    )
    parser.add_argument(
        "--output-dir",
        help="write artifacts outside tracked result directories (recommended for verification)",
    )
    args = parser.parse_args()
    if args.output_dir:
        if args.mode == "loop-validation":
            LOOP_RESULT_DIR = Path(args.output_dir)
        else:
            RESULT_DIR = Path(args.output_dir)
    run_backtest(args.mode)


if __name__ == "__main__":
    main()
