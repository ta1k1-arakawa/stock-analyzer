"""Survivorship-biased yfinance prototype for stock-analyzer v3.

Raw market data is kept outside the repository.  This module contains the
deterministic transformations and evaluation primitives used by the audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.analysis import calculate_indicators
from src.trade_simulator import simulate_execution


EVALUATION_METADATA = {
    "evaluation_type": "SURVIVORSHIP_BIASED_RESEARCH_ONLY",
    "formal_backtest": False,
    "point_in_time_universe": False,
    "deployment_decision_allowed": False,
    "shadow_replacement_allowed": False,
    "reference_period_used": False,
}
DATE_FROM = "2019-01-01"
DATE_TO = "2025-03-31"
YAHOO_PERIOD2 = "2025-04-01"
EVALUATION_FROM = "2020-01-01"
MAX_CODES = 300
ALLOWED_HOSTS = {
    "www.jpx.co.jp",
    "jpx.co.jp",
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
}
FEATURE_COLUMNS = [
    "SMA_5_Rate", "SMA_25_Rate", "RSI_14", "MACD_Rate", "BB_Position",
    "ATR_Rate", "ADX_14", "Change_Rate_1", "Change_Rate_3",
    "Change_Rate_5", "Volume_Change_1", "Realized_Volatility_20",
    "Realized_Volatility_60", "Log_Turnover_20", "Median_Turnover_60",
    "Cross_Sectional_Return_Rank", "Cross_Sectional_Volatility_Rank",
]
MODEL_PARAMS = {
    "objective": "huber", "alpha": 0.90, "n_estimators": 300,
    "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1,
    "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
    "reg_alpha": 0.0, "reg_lambda": 0.0, "random_state": 42,
    "n_jobs": 1, "deterministic": True, "force_col_wise": True,
    "verbosity": -1,
}
FOLDS = [
    {"fold": 1, "train_from": "2020-01-01", "train_to": "2020-12-31", "validation_from": "2021-01-01", "validation_to": "2022-03-31"},
    {"fold": 2, "train_from": "2020-01-01", "train_to": "2022-03-31", "validation_from": "2022-04-01", "validation_to": "2023-09-30"},
    {"fold": 3, "train_from": "2020-01-01", "train_to": "2023-09-30", "validation_from": "2023-10-01", "validation_to": "2025-03-31"},
]


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_allowed_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_HOSTS:
        raise RuntimeError(f"PROHIBITED_NETWORK_DESTINATION:{host or '<missing>'}")
    return host


class NetworkAudit:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        host = assert_allowed_url(url)
        response = requests.get(url, allow_redirects=False, timeout=kwargs.pop("timeout", 45), **kwargs)
        self.calls.append({"method": "GET", "host": host})
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location", "")
            redirect_host = urlparse(urljoin(url, location)).hostname or ""
            raise RuntimeError(f"REDIRECT_BLOCKED:{redirect_host.lower()}")
        response.raise_for_status()
        return response

    def summary(self) -> dict[str, Any]:
        hosts: dict[str, int] = {}
        for call in self.calls:
            hosts[call["host"]] = hosts.get(call["host"], 0) + 1
        return {"total_calls": len(self.calls), "hosts": dict(sorted(hosts.items()))}


def select_codes(codes: list[str], limit: int = MAX_CODES) -> list[str]:
    normalized = sorted({str(code).strip().upper() for code in codes if str(code).strip()})
    return sorted(normalized, key=lambda code: (sha256(code.encode("utf-8")).hexdigest(), code))[:limit]


def selected_codes_hash(codes: list[str]) -> str:
    return sha256_bytes(("\n".join(codes) + "\n").encode("utf-8"))


def _find_column(columns: list[str], needles: tuple[str, ...]) -> str:
    for column in columns:
        compact = re.sub(r"\s+", "", str(column))
        if all(needle in compact for needle in needles):
            return column
    raise ValueError(f"required JPX column missing: {needles}")


def parse_current_jpx_universe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [str(c) for c in frame.columns]
    code_col = _find_column(columns, ("コード",))
    name_col = _find_column(columns, ("銘柄名",))
    market_col = _find_column(columns, ("市場", "区分"))
    sector_col = next((c for c in columns if "33業種区分" in re.sub(r"\s+", "", c)), None)
    work = frame.rename(columns={code_col: "code", name_col: "name", market_col: "market"}).copy()
    if sector_col:
        work = work.rename(columns={sector_col: "industry"})
    else:
        work["industry"] = "MISSING"
    work["code"] = work["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()
    work["market"] = work["market"].astype(str).str.strip()
    prime_standard = work["market"].str.contains("プライム|Prime|スタンダード|Standard", case=False, regex=True)
    domestic = work["market"].str.contains("内国株式|Domestic Stocks", case=False, regex=True)
    ordinary_code = work["code"].str.fullmatch(r"[0-9A-Z]{4}")
    reasons = {
        "input_rows": int(len(work)),
        "excluded_non_prime_standard": int((~prime_standard).sum()),
        "excluded_non_domestic_stock": int((prime_standard & ~domestic).sum()),
        "excluded_non_four_character_code": int((prime_standard & domestic & ~ordinary_code).sum()),
    }
    eligible = work.loc[prime_standard & domestic & ordinary_code, ["code", "name", "market", "industry"]].drop_duplicates("code")
    reasons["eligible_current_only"] = int(len(eligible))
    return eligible.sort_values("code", kind="mergesort").reset_index(drop=True), reasons


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    result = frame.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("duplicate or unordered OHLCV dates")
    if len(result) and result.index.max() > pd.Timestamp(DATE_TO):
        raise ValueError("PROHIBITED_POST_CUTOFF_DATA")
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    finite = result[["Open", "High", "Low", "Close", "Volume"]].replace([np.inf, -np.inf], np.nan)
    if finite[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("non-finite OHLC price")
    invalid = (result["Low"] > result[["Open", "Close", "High"]].min(axis=1)) | (result["High"] < result[["Open", "Close", "Low"]].max(axis=1))
    if invalid.any() or (result[["Open", "High", "Low", "Close"]] <= 0).any().any() or (result["Volume"] < 0).any():
        raise ValueError("invalid OHLCV relationship")
    return result


def parse_yahoo_chart(payload: dict[str, Any]) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart error: {chart.get('error')}")
    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps or adj is None:
        raise ValueError("empty Yahoo chart response")
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    frame = pd.DataFrame({
        "Open": quote.get("open"), "High": quote.get("high"), "Low": quote.get("low"),
        "Close": quote.get("close"), "Adj Close": adj, "Volume": quote.get("volume"),
    }, index=index)
    frame = frame.loc[(frame.index >= DATE_FROM) & (frame.index <= DATE_TO)]
    splits = {
        pd.to_datetime(int(item["date"]), unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
        for item in (result.get("events", {}).get("splits", {}) or {}).values()
        if item.get("date") is not None
    }
    return validate_ohlcv(frame), splits


def add_stock_features(raw: pd.DataFrame, analysis_params: dict[str, Any]) -> pd.DataFrame:
    raw = validate_ohlcv(raw)
    factor = raw["Adj Close"] / raw["Close"]
    adjusted = pd.DataFrame(index=raw.index)
    for column in ("Open", "High", "Low", "Close"):
        adjusted[column] = raw[column] * factor
    adjusted["Volume"] = raw["Volume"]
    features = calculate_indicators(adjusted, analysis_params)
    features["Realized_Volatility_20"] = adjusted["Close"].pct_change().rolling(20).std(ddof=0) * math.sqrt(252) * 100
    features["Realized_Volatility_60"] = adjusted["Close"].pct_change().rolling(60).std(ddof=0) * math.sqrt(252) * 100
    turnover = raw["Close"] * raw["Volume"]
    features["Log_Turnover_20"] = np.log(turnover.rolling(20).mean().where(lambda value: value > 0))
    features["Median_Turnover_60"] = turnover.rolling(60).median()
    features["Median_Volume_60"] = raw["Volume"].rolling(60).median()
    features["Raw_Close"] = raw["Close"]
    features["History_Count"] = np.arange(1, len(features) + 1)
    return features


def add_execution_labels(raw: pd.DataFrame, features: pd.DataFrame, split_dates: set[pd.Timestamp]) -> tuple[pd.DataFrame, int]:
    result = features.copy()
    for column in ("realized_net_return_percent", "LabelConfirmedDate", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "ExitReason"):
        result[column] = np.nan if column not in {"LabelConfirmedDate", "EntryDate", "ExitDate", "ExitReason"} else None
    excluded_splits = 0
    for pos in range(len(raw)):
        execution = simulate_execution(raw[["Open", "High", "Low", "Close", "Volume"]], pos, 2, 5.0, 0.03, 0.03, 0.10, 0.0)
        if execution is None:
            continue
        entry_date = raw.index[execution.entry_index]
        exit_date = raw.index[execution.exit_index]
        if any(entry_date <= split_date <= exit_date for split_date in split_dates):
            excluded_splits += 1
            continue
        entry_open = float(raw.iloc[execution.entry_index]["Open"])
        stop_price = execution.entry_price * 0.95
        exit_row = raw.iloc[execution.exit_index]
        if execution.exit_reason == "TIME":
            reason = "TIME"
        elif float(exit_row["Open"]) <= stop_price:
            reason = "GAP_STOP"
        else:
            reason = "STOP"
        idx = raw.index[pos]
        result.at[idx, "realized_net_return_percent"] = execution.return_percent
        result.at[idx, "LabelConfirmedDate"] = exit_date
        result.at[idx, "EntryDate"] = entry_date
        result.at[idx, "ExitDate"] = exit_date
        result.at[idx, "EntryPrice"] = execution.entry_price
        result.at[idx, "ExitPrice"] = execution.exit_price
        result.at[idx, "ExitReason"] = reason
    return result, excluded_splits


def combine_feature_frames(frames: dict[str, pd.DataFrame], industries: dict[str, str]) -> pd.DataFrame:
    parts = []
    for code in sorted(frames):
        part = frames[code].copy()
        part["code"] = code
        part["industry"] = industries.get(code, "MISSING")
        part["signal_date"] = part.index
        parts.append(part.reset_index(drop=True))
    combined = pd.concat(parts, ignore_index=True)
    combined["Cross_Sectional_Return_Rank"] = combined.groupby("signal_date")["Change_Rate_1"].rank(pct=True, method="average")
    combined["Cross_Sectional_Volatility_Rank"] = combined.groupby("signal_date")["Realized_Volatility_20"].rank(pct=True, method="average")
    combined["EligiblePast"] = (
        (combined["History_Count"] >= 252)
        & (combined["Median_Turnover_60"] >= 100_000_000)
        & (combined["Median_Volume_60"] >= 50_000)
        & (combined["Raw_Close"] * 100 <= 300_000)
    )
    return combined.sort_values(["signal_date", "code"], kind="mergesort").reset_index(drop=True)


def training_rows_for_fold(data: pd.DataFrame, fold: dict[str, Any]) -> pd.DataFrame:
    validation_start = pd.Timestamp(fold["validation_from"])
    dates = sorted(pd.to_datetime(data.loc[data["signal_date"] < validation_start, "signal_date"].unique()))
    embargo_cutoff = dates[-2] if len(dates) >= 2 else pd.Timestamp.min
    mask = (
        (data["signal_date"] >= pd.Timestamp(fold["train_from"]))
        & (data["signal_date"] <= pd.Timestamp(fold["train_to"]))
        & (data["signal_date"] < embargo_cutoff)
        & (pd.to_datetime(data["LabelConfirmedDate"]) < validation_start)
    )
    return data.loc[mask].copy()


def validation_rows_for_fold(data: pd.DataFrame, fold: dict[str, Any]) -> pd.DataFrame:
    mask = (data["signal_date"] >= pd.Timestamp(fold["validation_from"])) & (data["signal_date"] <= pd.Timestamp(fold["validation_to"]))
    return data.loc[mask].copy()


def fit_one_model(train: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train["realized_net_return_percent"])
    return model


def huber_metric(y_true: np.ndarray, y_pred: np.ndarray, delta_quantile: float = 0.90) -> float:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    delta = float(np.quantile(np.abs(residual), delta_quantile))
    if delta <= 0:
        return 0.0
    absolute = np.abs(residual)
    loss = np.where(absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta))
    return float(np.mean(loss))


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["realized_net_return_percent"].to_numpy(float)
    predicted = frame["prediction"].to_numpy(float)
    rho = float(spearmanr(predicted, actual).statistic) if len(np.unique(predicted)) > 1 else 0.0
    daily = frame.groupby("signal_date", sort=True).apply(
        lambda group: spearmanr(group["prediction"], group["realized_net_return_percent"]).statistic if len(group) >= 2 and group["prediction"].nunique() > 1 else np.nan,
        include_groups=False,
    ).dropna()
    cutoff = frame["prediction"].quantile(0.90)
    top = frame.loc[frame["prediction"] >= cutoff, "realized_net_return_percent"]
    positive = frame.loc[frame["prediction"] > 0, "realized_net_return_percent"]
    return {
        "sample_count": int(len(frame)), "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5), "huber_loss": huber_metric(actual, predicted),
        "spearman": rho, "daily_ic_count": int(len(daily)), "daily_ic_mean": float(daily.mean()),
        "daily_ic_median": float(daily.median()), "daily_ic_std": float(daily.std(ddof=0)),
        "daily_ic_positive_rate": float((daily > 0).mean()), "top_decile_mean_return": float(top.mean()),
        "all_candidate_mean_return": float(frame["realized_net_return_percent"].mean()),
        "top_decile_minus_all": float(top.mean() - frame["realized_net_return_percent"].mean()),
        "positive_prediction_mean_return": float(positive.mean()) if len(positive) else 0.0,
    }


def deterministic_random_score(seed: int, signal_date: Any, code: str) -> float:
    raw = f"{seed}|{pd.Timestamp(signal_date).strftime('%Y-%m-%d')}|{code}".encode()
    return int.from_bytes(sha256(raw).digest()[:8], "big") / 2**64


def prepare_portfolio_candidates(candidates: pd.DataFrame) -> dict[str, Any]:
    candidates = candidates.sort_values(["EntryDate", "code"], kind="mergesort").copy()
    calendar_values = list(set(candidates["signal_date"]) | set(candidates["EntryDate"]) | set(candidates["ExitDate"]))
    calendar = sorted(pd.to_datetime(calendar_values))
    by_entry = {date: group for date, group in candidates.groupby(pd.to_datetime(candidates["EntryDate"]), sort=True)}
    raw_close = {(pd.Timestamp(row.signal_date), row.code): float(row.Raw_Close) for row in candidates.itertuples()}
    return {"calendar": calendar, "by_entry": by_entry, "raw_close": raw_close}


def simulate_prepared_portfolio(prepared: dict[str, Any], score_column: str, positive_gate: bool = False, random_seed: int | None = None) -> dict[str, Any]:
    calendar = prepared["calendar"]
    by_entry = prepared["by_entry"]
    raw_close = prepared["raw_close"]
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    cash = 300_000.0
    pending = 0.0
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    no_trade = {"NO_TRADE_NON_POSITIVE_PREDICTION": 0, "NO_TRADE_INSUFFICIENT_CASH": 0, "NO_TRADE_POSITION_OPEN": 0}
    for date in calendar:
        cash += pending
        pending = 0.0
        group = by_entry.get(date)
        if group is not None:
            if position is not None:
                no_trade["NO_TRADE_POSITION_OPEN"] += 1
            else:
                if rng is None:
                    ranked = group.sort_values([score_column, "code"], ascending=[False, True], kind="mergesort")
                else:
                    ranked = group.assign(_score=rng.random(len(group))).sort_values(["_score", "code"], ascending=[False, True], kind="mergesort")
                if positive_gate and float(ranked.iloc[0][score_column]) <= 0:
                    no_trade["NO_TRADE_NON_POSITIVE_PREDICTION"] += 1
                else:
                    affordable = ranked.loc[ranked["EntryPrice"] * 100 <= cash + 1e-8]
                    if affordable.empty:
                        no_trade["NO_TRADE_INSUFFICIENT_CASH"] += 1
                    else:
                        chosen = affordable.iloc[0]
                        cost = float(chosen["EntryPrice"]) * 100
                        cash -= cost
                        if cash < -1e-8:
                            raise AssertionError("negative cash")
                        position = chosen.to_dict()
                        position["entry_value"] = cost
        if position is not None and pd.Timestamp(position["ExitDate"]) == date:
            proceeds = float(position["ExitPrice"]) * 100
            profit = proceeds - float(position["entry_value"])
            pending += proceeds
            trades.append({"code": position["code"], "industry": position["industry"], "signal_date": pd.Timestamp(position["signal_date"]), "exit_date": date, "profit": profit, "exit_reason": position["ExitReason"]})
            position = None
        marked = 0.0
        if position is not None:
            marked = raw_close.get((date, position["code"]), float(position["entry_value"]) / 100) * 100
        ledger.append({"date": date, "cash": cash, "pending": pending, "marked": marked, "equity": cash + pending + marked, "open_positions": int(position is not None)})
    trade_frame = pd.DataFrame(trades)
    ledger_frame = pd.DataFrame(ledger)
    equity = ledger_frame["equity"] if len(ledger_frame) else pd.Series([300_000.0])
    drawdown = (equity / equity.cummax() - 1) * 100
    profit = float(trade_frame["profit"].sum()) if len(trade_frame) else 0.0
    positive_by_stock = trade_frame.groupby("code")["profit"].sum().clip(lower=0) if len(trade_frame) else pd.Series(dtype=float)
    positive_by_industry = trade_frame.groupby("industry")["profit"].sum().clip(lower=0) if len(trade_frame) else pd.Series(dtype=float)
    monthly = trade_frame.assign(month=lambda x: x["exit_date"].dt.to_period("M")).groupby("month")["profit"].sum() if len(trade_frame) else pd.Series(dtype=float)
    yearly = trade_frame.assign(year=lambda x: x["exit_date"].dt.year).groupby("year")["profit"].sum() if len(trade_frame) else pd.Series(dtype=float)
    reasons = trade_frame["exit_reason"].value_counts().to_dict() if len(trade_frame) else {}
    return {
        "profit": profit, "ending_equity": 300_000.0 + profit, "max_drawdown_percent": float(abs(drawdown.min())),
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "yearly_profit": {str(k): float(v) for k, v in yearly.items()}, "closed_trades": int(len(trade_frame)),
        "win_rate": float((trade_frame["profit"] > 0).mean()) if len(trade_frame) else 0.0,
        "max_stock_profit_share": float(positive_by_stock.max() / positive_by_stock.sum()) if positive_by_stock.sum() > 0 else 0.0,
        "max_industry_profit_share": float(positive_by_industry.max() / positive_by_industry.sum()) if positive_by_industry.sum() > 0 else 0.0,
        "stop_count": int(reasons.get("STOP", 0)), "gap_stop_count": int(reasons.get("GAP_STOP", 0)),
        "time_count": int(reasons.get("TIME", 0)), "no_trade_counts": no_trade,
        "negative_cash_count": int((ledger_frame["cash"] < -1e-8).sum()) if len(ledger_frame) else 0,
        "capital_reuse_count": 0, "duplicate_order_count": 0,
    }


def simulate_ranked_portfolio(candidates: pd.DataFrame, score_column: str, positive_gate: bool = False, random_seed: int | None = None) -> dict[str, Any]:
    return simulate_prepared_portfolio(prepare_portfolio_candidates(candidates), score_column, positive_gate, random_seed)


def simulate_random_distribution(prepared: dict[str, Any], seeds: range) -> tuple[list[float], list[float]]:
    """Evaluate random rankings without rebuilding pandas objects per seed."""
    calendar = prepared["calendar"]
    raw_close = prepared["raw_close"]
    compact_groups: dict[pd.Timestamp, list[tuple[str, float, float, pd.Timestamp]]] = {}
    for date, group in prepared["by_entry"].items():
        compact_groups[date] = [
            (str(row.code), float(row.EntryPrice), float(row.ExitPrice), pd.Timestamp(row.ExitDate))
            for row in group.sort_values("code", kind="mergesort").itertuples()
        ]
    profits: list[float] = []
    drawdowns: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cash = 300_000.0
        pending = 0.0
        position: tuple[str, float, float, pd.Timestamp] | None = None
        peak = 300_000.0
        worst = 0.0
        realized = 0.0
        for date in calendar:
            cash += pending
            pending = 0.0
            group = compact_groups.get(date)
            if group and position is None:
                scores = rng.random(len(group))
                order = np.argsort(-scores, kind="stable")
                for offset in order:
                    candidate = group[int(offset)]
                    cost = candidate[1] * 100
                    if cost <= cash + 1e-8:
                        cash -= cost
                        position = candidate
                        break
            if position is not None and position[3] == date:
                proceeds = position[2] * 100
                realized += proceeds - position[1] * 100
                pending += proceeds
                position = None
            marked = raw_close.get((date, position[0]), position[1]) * 100 if position is not None else 0.0
            equity = cash + pending + marked
            peak = max(peak, equity)
            worst = min(worst, equity / peak - 1)
        profits.append(realized)
        drawdowns.append(abs(worst * 100))
    return profits, drawdowns


def round_floats(value: Any, digits: int = 10) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(k): round_floats(v, digits) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [round_floats(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_deterministic_json(path: Path, value: Any) -> None:
    path.write_bytes(stable_json_bytes(round_floats(value)))


def ensure_report_has_no_raw_prices(value: Any) -> None:
    forbidden = {"open", "high", "low", "close", "adj close", "volume", "entryprice", "exitprice"}
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if str(key).lower().replace("_", "") in {x.replace(" ", "") for x in forbidden}:
                    raise AssertionError(f"raw price field in report: {key}")
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(value)
