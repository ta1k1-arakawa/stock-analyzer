"""Deterministic, offline V4 meta-label prototype primitives.

This module deliberately has no downloader and does not run an evaluation at
import time.  It is the implementation of the pre-registered V4 design; a
separate, explicitly authorised execution step must supply verified data.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import csv
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

UNIVERSE_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
DESIGN_SHA256 = "07039948aa7a1180d506b3089a0bd5612dda24559968c510e0cb92935b48055a"
FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "volatility_20d",
    "volume_ratio_5d_20d", "close_to_ma20", "close_to_ma60",
    "high_low_range_20d", "required_cash_ratio", "momentum_20d_percentile_rank",
    "relative_momentum_20d", "cross_section_median_return_20d",
    "cross_section_breadth_above_ma20", "cross_section_median_volatility_20d",
    "cross_section_eligible_count",
]
MODEL_PARAMS = {
    "objective": "binary", "n_estimators": 300, "learning_rate": 0.03,
    "num_leaves": 15, "max_depth": -1, "min_child_samples": 40,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "random_state": 20260803,
    "n_jobs": 1, "deterministic": True, "force_col_wise": True,
    "verbosity": -1, "class_weight": None,
}
FOLDS = (
    {"fold": 1, "train_from": "2016-04-01", "train_to": "2016-12-31", "test_from": "2017-01-01", "test_to": "2017-12-31"},
    {"fold": 2, "train_from": "2016-04-01", "train_to": "2017-12-31", "test_from": "2018-01-01", "test_to": "2018-12-31"},
    {"fold": 3, "train_from": "2016-04-01", "train_to": "2018-12-31", "test_from": "2019-01-01", "test_to": "2019-12-31"},
)
THRESHOLD = 0.55
BLOCKED_CONDITIONS = 10
ACCEPTANCE_CONDITIONS = 17


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def model_parameter_hash() -> str:
    return sha256_bytes(stable_json_bytes(MODEL_PARAMS))


def _canonical_csv_bytes(raw: bytes) -> bytes:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UNIVERSE_BOM_FORBIDDEN")
    if raw.replace(b"\r\n", b"").count(b"\r"):
        raise ValueError("UNIVERSE_STANDALONE_CR_FORBIDDEN")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("UNIVERSE_UTF8_INVALID") from exc
    return raw.replace(b"\r\n", b"\n")


def validate_fixed_universe(csv_path: Path, manifest_path: Path) -> list[dict[str, str]]:
    """Validate bytes *and* CSV content, accepting only LF/CRLF checkout variance."""
    raw = csv_path.read_bytes()
    canonical = _canonical_csv_bytes(raw)
    if sha256_bytes(canonical) != UNIVERSE_SHA256:
        raise ValueError("UNIVERSE_CANONICAL_SHA256_MISMATCH")
    if not canonical.endswith(b"\n") or canonical.endswith(b"\n\n"):
        raise ValueError("UNIVERSE_FINAL_NEWLINE_INVALID")
    rows = list(csv.reader(io.StringIO(canonical.decode("utf-8"), newline="")))
    if not rows or rows[0] != ["ticker", "market", "industry"]:
        raise ValueError("UNIVERSE_COLUMNS_MISMATCH")
    data = rows[1:]
    if len(data) != 300 or any(len(row) != 3 for row in data):
        raise ValueError("UNIVERSE_ROW_COUNT_OR_SHAPE_MISMATCH")
    tickers = [row[0] for row in data]
    if any(not ticker for ticker in tickers) or len(set(tickers)) != 300:
        raise ValueError("UNIVERSE_TICKERS_EMPTY_OR_DUPLICATE")
    if not all(re.fullmatch(r"[0-9A-Z]{4}", ticker) for ticker in tickers):
        raise ValueError("UNIVERSE_TICKER_FORMAT_MISMATCH")
    ticker_hash = sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8"))
    if ticker_hash != TICKER_LIST_SHA256:
        raise ValueError("UNIVERSE_TICKER_LIST_SHA256_MISMATCH")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("UNIVERSE_MANIFEST_INVALID") from exc
    required = {
        "selected_count": 300, "universe_csv_sha256": UNIVERSE_SHA256,
        "ticker_list_sha256": TICKER_LIST_SHA256, "v3_ticker_list_sha256": TICKER_LIST_SHA256,
        "matches_v3_ticker_list": True,
    }
    if any(manifest.get(key) != expected for key, expected in required.items()):
        raise ValueError("UNIVERSE_MANIFEST_MISMATCH")
    return [{"ticker": row[0], "market": row[1], "industry": row[2]} for row in data]


def validate_design(path: Path) -> None:
    if sha256_bytes(path.read_bytes()) != DESIGN_SHA256:
        raise ValueError("DESIGN_SHA256_MISMATCH")


def validate_fixed_inputs(root: Path) -> dict[str, Any]:
    validate_design(root / "V4_META_LABEL_DESIGN.md")
    universe = validate_fixed_universe(root / "V4_UNIVERSE.csv", root / "V4_UNIVERSE_MANIFEST.json")
    raw = (root / "V4_UNIVERSE.csv").read_bytes()
    canonical = _canonical_csv_bytes(raw)
    return {"universe_count": len(universe), "raw_csv_sha256": sha256_bytes(raw),
            "canonical_csv_sha256": sha256_bytes(canonical), "ticker_list_sha256": TICKER_LIST_SHA256,
            "design_sha256": DESIGN_SHA256, "crlf_count": raw.count(b"\r\n"), "network_calls": 0,
            "model_fits": 0, "real_data_backtests": 0}


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if any(col not in frame.columns for col in required):
        raise ValueError("OHLCV_COLUMNS_MISSING")
    result = frame[required].copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("OHLCV_DATE_ORDER_INVALID")
    if len(result) and result.index.max() >= pd.Timestamp("2020-01-01"):
        raise ValueError("POST_CUTOFF_PRICE_ROW")
    for col in required:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if not np.isfinite(result.to_numpy(float)).all() or (result[["Open", "High", "Low", "Close"]] <= 0).any().any() or (result["Volume"] < 0).any():
        raise ValueError("OHLCV_VALUES_INVALID")
    if ((result["Low"] > result[["Open", "High", "Close"]].min(axis=1)) | (result["High"] < result[["Open", "Low", "Close"]].max(axis=1))).any():
        raise ValueError("OHLCV_RANGE_INVALID")
    return result


def stock_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = validate_ohlcv(raw)
    adjusted_factor = raw["Adj Close"] / raw["Close"]
    adjusted = pd.DataFrame({name: raw[name] * adjusted_factor for name in ("Open", "High", "Low", "Close")}, index=raw.index)
    ret = adjusted["Close"].pct_change()
    out = pd.DataFrame(index=raw.index)
    out["return_5d"] = adjusted["Close"].pct_change(5)
    out["return_20d"] = adjusted["Close"].pct_change(20)
    out["return_60d"] = adjusted["Close"].pct_change(60)
    out["volatility_20d"] = ret.rolling(20).std(ddof=0) * math.sqrt(252)
    out["volume_ratio_5d_20d"] = raw["Volume"].rolling(5).mean() / raw["Volume"].rolling(20).mean()
    out["close_to_ma20"] = adjusted["Close"] / adjusted["Close"].rolling(20).mean() - 1
    out["close_to_ma60"] = adjusted["Close"] / adjusted["Close"].rolling(60).mean() - 1
    out["high_low_range_20d"] = adjusted["High"].rolling(20).max() / adjusted["Low"].rolling(20).min() - 1
    out["required_cash_ratio"] = raw["Close"] * 100 / 300_000
    out["raw_close"] = raw["Close"]; out["raw_volume"] = raw["Volume"]
    out["history_count"] = np.arange(1, len(raw) + 1)
    out["adjusted_close"] = adjusted["Close"]
    return out


def eligible_rows(ticker: str, features: pd.DataFrame) -> pd.DataFrame:
    eligible = features.loc[(features["history_count"] >= 252) &
                            ((features["raw_close"] * features["raw_volume"]).rolling(60).median() >= 100_000_000) &
                            (features["raw_volume"].rolling(60).median() >= 50_000) &
                            (features["raw_close"] * 100 <= 300_000) &
                            np.isfinite(features[FEATURE_COLUMNS[:9]]).all(axis=1)].copy()
    eligible["ticker"] = ticker
    return eligible


def baseline_candidates(per_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One deterministic candidate per day, with same-day cross-sectional features."""
    rows = pd.concat([eligible_rows(ticker, frame) for ticker, frame in sorted(per_ticker.items())]) if per_ticker else pd.DataFrame()
    if rows.empty: return rows
    rows["signal_date"] = rows.index
    grouped = rows.groupby("signal_date", sort=True)
    rows["momentum_20d_percentile_rank"] = grouped["return_20d"].rank(pct=True, method="average")
    rows["relative_momentum_20d"] = rows["return_20d"] - grouped["return_20d"].transform("median")
    rows["cross_section_median_return_20d"] = grouped["return_20d"].transform("median")
    rows["cross_section_breadth_above_ma20"] = grouped["close_to_ma20"].transform(lambda s: (s > 0).mean())
    rows["cross_section_median_volatility_20d"] = grouped["volatility_20d"].transform("median")
    rows["cross_section_eligible_count"] = grouped["ticker"].transform("size")
    rows = rows.sort_values(["signal_date", "return_20d", "ticker"], ascending=[True, False, True], kind="mergesort")
    return rows.groupby("signal_date", sort=False).head(1).reset_index(drop=True)


@dataclass(frozen=True)
class Execution:
    entry_date: pd.Timestamp; exit_date: pd.Timestamp; entry_price: float; exit_price: float; exit_reason: str; realized_net_return_percent: float


def execute_candidate(raw: pd.DataFrame, signal_date: Any, split_dates: Iterable[Any] = ()) -> Execution | None:
    raw = validate_ohlcv(raw); signal = pd.Timestamp(signal_date)
    if signal not in raw.index: raise ValueError("SIGNAL_DATE_MISSING")
    pos = raw.index.get_loc(signal); entry_pos, final_pos = pos + 1, pos + 2
    if final_pos >= len(raw): return None
    entry_date, final_date = raw.index[entry_pos], raw.index[final_pos]
    if any(entry_date <= pd.Timestamp(day) <= final_date for day in split_dates): return None
    entry = float(raw.iloc[entry_pos]["Open"]) * 1.0003; stop = entry * .95
    reason, exit_pos, base = "TIME", final_pos, float(raw.iloc[final_pos]["Close"])
    for i in range(entry_pos, final_pos + 1):
        row = raw.iloc[i]
        if float(row["Low"]) <= stop:
            reason, exit_pos = ("GAP_STOP" if float(row["Open"]) <= stop else "STOP"), i
            base = float(row["Open"]) if reason == "GAP_STOP" else stop
            break
    exit_price = base * (0.999 if reason != "TIME" else 0.9997)
    return Execution(entry_date, raw.index[exit_pos], entry, exit_price, reason, (exit_price - entry) / entry * 100)


def fold_training_rows(candidates: pd.DataFrame, fold: dict[str, Any]) -> pd.DataFrame:
    start = pd.Timestamp(fold["train_from"]); end = pd.Timestamp(fold["train_to"]); test_start = pd.Timestamp(fold["test_from"])
    return candidates.loc[(pd.to_datetime(candidates["signal_date"]) >= start) & (pd.to_datetime(candidates["signal_date"]) <= end) & (pd.to_datetime(candidates["ExitDate"]) < test_start)].copy()


def data_sufficiency_blockers(folds: Iterable[dict[str, Any]], successful_tickers: int, hashes_fixed: bool, post_cutoff_rows: int, prohibited_network: bool, deterministic: bool) -> list[str]:
    blocked: list[str] = []
    if successful_tickers < 150: blocked.append("successful_tickers_under_150")
    for fold in folds:
        train, test, baseline = fold["train"], fold["test"], fold["baseline"]
        if len(train) < 100: blocked.append("train_candidates_under_100")
        labels = train["label"]
        if labels.nunique() != 2: blocked.append("train_one_class")
        if min(int((labels == 1).sum()), int((labels == 0).sum())) < 20: blocked.append("train_class_under_20")
        if len(baseline) < 40: blocked.append("baseline_closed_trades_under_40")
        if test["label"].nunique() != 2: blocked.append("test_one_class")
    if not hashes_fixed: blocked.append("hashes_not_fixed")
    if post_cutoff_rows: blocked.append("post_cutoff_rows")
    if prohibited_network: blocked.append("prohibited_network")
    if not deterministic: blocked.append("determinism_not_confirmed")
    return blocked


def acceptance_checks(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    f = metrics["folds"]; b = metrics["baseline"]; v = metrics["v4"]
    values = [v["profit"] > b["profit"], v["profit"] > 0, v["max_drawdown"] < b["max_drawdown"], sum(x["v4_profit"] > x["baseline_profit"] for x in f) >= 2, all(x["v4_drawdown"] <= x["baseline_drawdown"] for x in f), v["win_rate"] > b["win_rate"], v["closed_trades"] >= 100, .2 <= metrics["acceptance_rate"] <= .8, metrics["roc_auc"] > .52, sum(x["roc_auc"] > .5 for x in f) >= 2, v["max_ticker_share"] <= .35, v["top5_ticker_share"] <= .60, v["max_industry_share"] <= .50, all(v[k] == b[k] == 0 for k in ("negative_cash", "capital_reuse", "duplicate_order")), metrics["byte_identical_runs"], metrics["post_cutoff_count"] == 0, metrics["prohibited_network_count"] == 0]
    return [{"condition": f"acceptance_{i + 1:02d}", "passed": bool(value)} for i, value in enumerate(values)]


def preflight(root: Path, cache_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    for path in (cache_dir, output_dir):
        if path is not None and (path.resolve() == root or root in path.resolve().parents):
            raise ValueError("CACHE_OR_OUTPUT_MUST_BE_OUTSIDE_REPOSITORY")
    return validate_fixed_inputs(root)
