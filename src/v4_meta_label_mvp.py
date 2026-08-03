"""Phase 1 deterministic transformations for the V4 meta-label MVP.

This module deliberately has no HTTP client, model fitting, or portfolio
backtest entry point.  Market data is supplied as Yahoo chart payloads or
OHLCV DataFrames so the pre-registered data boundary can be tested offline.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.free_prototype import validate_ohlcv
from src.trade_simulator import simulate_execution

PRICE_FROM = pd.Timestamp("2015-01-01")
PRICE_TO = pd.Timestamp("2019-12-31")
SIGNAL_FROM = pd.Timestamp("2016-04-01")
SIGNAL_TO = pd.Timestamp("2019-12-31")
UNIVERSE_COUNT = 300
UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
FEATURE_COLUMNS = (
    "return_5d", "return_20d", "return_60d", "volatility_20d",
    "volume_ratio_5d_20d", "close_to_ma20", "close_to_ma60",
    "high_low_range_20d", "required_cash_ratio",
    "momentum_20d_percentile_rank", "relative_momentum_20d",
    "cross_section_median_return_20d", "cross_section_breadth_above_ma20",
    "cross_section_median_volatility_20d", "cross_section_eligible_count",
)
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
PRELIMINARY_STOCK_FEATURE_COLUMNS = (
    "return_5d", "return_20d", "return_60d", "volatility_20d",
    "volume_ratio_5d_20d", "close_to_ma20", "close_to_ma60",
    "high_low_range_20d", "required_cash_ratio",
)


def load_fixed_universe(path: str | Path = "V4_UNIVERSE.csv") -> pd.DataFrame:
    """Read the frozen universe without sorting or otherwise changing its order."""
    path = Path(path)
    # The registered digest uses UTF-8/LF canonical bytes, so it is stable
    # across a Windows checkout with autocrlf enabled.
    canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    if sha256(canonical).hexdigest() != UNIVERSE_CSV_SHA256:
        raise ValueError("V4_UNIVERSE_CSV_HASH_MISMATCH")
    universe = pd.read_csv(path, dtype={"ticker": str})
    required = ["ticker", "market", "industry"]
    if list(universe.columns) != required or len(universe) != UNIVERSE_COUNT:
        raise ValueError("V4_UNIVERSE_SCHEMA_OR_COUNT_MISMATCH")
    universe["ticker"] = universe["ticker"].str.strip().str.upper()
    if universe["ticker"].duplicated().any() or (universe["ticker"].str.fullmatch(r"[0-9A-Z]{4}") == False).any():
        raise ValueError("V4_UNIVERSE_TICKER_INVALID")
    ticker_hash = sha256(("\n".join(universe["ticker"]) + "\n").encode("utf-8")).hexdigest()
    if ticker_hash != TICKER_LIST_SHA256:
        raise ValueError("V4_UNIVERSE_TICKER_HASH_MISMATCH")
    return universe


def validate_v4_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse V3 OHLCV validation and enforce V4's earlier, closed boundary."""
    result = validate_ohlcv(frame)
    if len(result) and (result.index.min() < PRICE_FROM or result.index.max() > PRICE_TO):
        raise ValueError("PROHIBITED_V4_PRICE_DATE")
    return result


def parse_v4_yahoo_chart(payload: Mapping[str, Any]) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """Yahoo-chart adapter equivalent to V3 parsing, parameterized for V4 dates."""
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart error: {chart.get('error')}")
    result = chart["result"][0]
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    timestamps = result.get("timestamp") or []
    if not timestamps or adjusted is None:
        raise ValueError("empty Yahoo chart response")
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    raw = pd.DataFrame({"Open": quote.get("open"), "High": quote.get("high"), "Low": quote.get("low"),
                        "Close": quote.get("close"), "Adj Close": adjusted, "Volume": quote.get("volume")}, index=index)
    splits = {pd.to_datetime(int(item["date"]), unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
              for item in (result.get("events", {}).get("splits", {}) or {}).values() if item.get("date") is not None}
    return validate_v4_ohlcv(raw), splits


def prepare_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Expose raw and adjusted OHLC, preserving raw Volume as in the V3 code."""
    raw = validate_v4_ohlcv(raw)
    factor = raw["Adj Close"] / raw["Close"]
    out = raw.copy()
    out["adjustment_factor"] = factor
    for column in ("Open", "High", "Low", "Close"):
        out[f"adjusted_{column.lower()}"] = raw[column] * factor
    return out


def _stock_features(price: pd.DataFrame) -> pd.DataFrame:
    adjusted_close = price["adjusted_close"]
    adjusted_high = price["adjusted_high"]
    adjusted_low = price["adjusted_low"]
    result = pd.DataFrame(index=price.index)
    result["return_5d"] = adjusted_close / adjusted_close.shift(5) - 1
    result["return_20d"] = adjusted_close / adjusted_close.shift(20) - 1
    result["return_60d"] = adjusted_close / adjusted_close.shift(60) - 1
    result["volatility_20d"] = adjusted_close.pct_change().rolling(20).std(ddof=0) * np.sqrt(252)
    result["volume_ratio_5d_20d"] = price["Volume"].rolling(5).mean() / price["Volume"].rolling(20).mean()
    ma20, ma60 = adjusted_close.rolling(20).mean(), adjusted_close.rolling(60).mean()
    result["close_to_ma20"] = adjusted_close / ma20 - 1
    result["close_to_ma60"] = adjusted_close / ma60 - 1
    result["high_low_range_20d"] = adjusted_high.rolling(20).max() / adjusted_low.rolling(20).min() - 1
    result["required_cash_ratio"] = price["Close"] * 100 / 300_000
    result["History_Count"] = np.arange(1, len(result) + 1)
    result["median_turnover_60d"] = (price["Close"] * price["Volume"]).rolling(60).median()
    result["median_volume_60d"] = price["Volume"].rolling(60).median()
    result["above_ma20"] = adjusted_close > ma20
    return result


def build_feature_frame(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """Calculate all 15 causal features for supplied frozen-universe prices."""
    allowed = set(universe["ticker"])
    if set(prices) - allowed:
        raise ValueError("PRICE_TICKER_NOT_IN_UNIVERSE")
    parts = []
    industry = universe.set_index("ticker")["industry"].to_dict()
    market = universe.set_index("ticker")["market"].to_dict()
    for ticker in universe["ticker"]:
        if ticker not in prices:
            continue
        stock = _stock_features(prepare_price_frame(prices[ticker]))
        stock["ticker"], stock["industry"], stock["market"] = ticker, industry[ticker], market[ticker]
        stock["signal_date"] = stock.index
        parts.append(stock.reset_index(drop=True))
    if not parts:
        return pd.DataFrame(columns=["signal_date", "ticker", "industry", "market", *FEATURE_COLUMNS])
    all_rows = pd.concat(parts, ignore_index=True)
    # This is the complete causal population for every cross-sectional feature.
    # In particular, incomplete-history or illiquid rows never influence a rank
    # or aggregate observed by a preliminary-eligible ticker.
    stock_values = all_rows.loc[:, PRELIMINARY_STOCK_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    preliminary_eligible = (
        (all_rows["History_Count"] >= 252)
        & (all_rows["median_turnover_60d"] >= 100_000_000)
        & (all_rows["median_volume_60d"] >= 50_000)
        & (all_rows["required_cash_ratio"] <= 1)
        & np.isfinite(stock_values.to_numpy(dtype=float)).all(axis=1)
    )
    eligible_returns = all_rows["return_20d"].where(preliminary_eligible)
    all_rows["momentum_20d_percentile_rank"] = eligible_returns.groupby(all_rows["signal_date"]).rank(pct=True, method="average")
    all_rows["cross_section_median_return_20d"] = eligible_returns.groupby(all_rows["signal_date"]).transform("median")
    all_rows["relative_momentum_20d"] = (all_rows["return_20d"] - all_rows["cross_section_median_return_20d"]).where(preliminary_eligible)
    all_rows["cross_section_breadth_above_ma20"] = all_rows["above_ma20"].where(preliminary_eligible).groupby(all_rows["signal_date"]).transform("mean")
    all_rows["cross_section_median_volatility_20d"] = all_rows["volatility_20d"].where(preliminary_eligible).groupby(all_rows["signal_date"]).transform("median")
    all_rows["cross_section_eligible_count"] = preliminary_eligible.astype(int).groupby(all_rows["signal_date"]).transform("sum")
    all_rows.loc[~preliminary_eligible, [column for column in FEATURE_COLUMNS if column not in PRELIMINARY_STOCK_FEATURE_COLUMNS]] = np.nan
    feature_values = all_rows.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    all_rows.loc[:, FEATURE_COLUMNS] = feature_values
    all_rows["eligible"] = preliminary_eligible & np.isfinite(feature_values.to_numpy(dtype=float)).all(axis=1)
    return all_rows.loc[(all_rows["signal_date"] >= SIGNAL_FROM) & (all_rows["signal_date"] <= SIGNAL_TO)].reset_index(drop=True)


def add_execution_labels(features: pd.DataFrame, prices: Mapping[str, pd.DataFrame], splits: Mapping[str, set[pd.Timestamp]]) -> pd.DataFrame:
    """Attach fixed two-day realized labels.  Split-spanning samples fail closed."""
    result = features.copy()
    for column in ("EntryDate", "ExitDate", "LabelConfirmedDate", "ExitReason"):
        result[column] = pd.NaT if column != "ExitReason" else None
    for column in ("EntryPrice", "ExitPrice", "realized_net_return_percent", "realized_net_profit_yen", "label"):
        result[column] = np.nan
    for ticker, index in result.groupby("ticker", sort=False).groups.items():
        raw = validate_v4_ohlcv(prices[ticker])
        positions = {date: pos for pos, date in enumerate(raw.index)}
        for row_index in index:
            signal_date = result.at[row_index, "signal_date"]
            pos = positions.get(signal_date)
            if pos is None:
                continue
            execution = simulate_execution(raw[["Open", "High", "Low", "Close", "Volume"]], pos, 2, 5.0, 0.03, 0.03, 0.10, 0.0)
            if execution is None:
                result.at[row_index, "eligible"] = False
                continue
            entry_date, exit_date = raw.index[execution.entry_index], raw.index[execution.exit_index]
            if any(entry_date <= split <= exit_date for split in splits.get(ticker, set())):
                result.at[row_index, "eligible"] = False
                continue
            stop = execution.entry_price * .95
            exit_row = raw.iloc[execution.exit_index]
            reason = "TIME" if execution.exit_reason == "TIME" else ("GAP_STOP" if float(exit_row["Open"]) <= stop else "STOP")
            result.loc[row_index, ["EntryDate", "ExitDate", "LabelConfirmedDate", "EntryPrice", "ExitPrice", "realized_net_return_percent"]] = [entry_date, exit_date, exit_date, execution.entry_price, execution.exit_price, execution.return_percent]
            result.at[row_index, "ExitReason"] = reason
            result.at[row_index, "realized_net_profit_yen"] = (execution.exit_price - execution.entry_price) * 100
            result.at[row_index, "label"] = int(execution.return_percent > 0)
    return result


def select_daily_candidates(labelled: pd.DataFrame) -> pd.DataFrame:
    """One deterministic baseline candidate per date; affordability never reranks it."""
    rows = []
    for date, group in labelled.groupby("signal_date", sort=True):
        eligible = group.loc[group["eligible"]].sort_values(["return_20d", "ticker"], ascending=[False, True], kind="mergesort")
        if eligible.empty:
            rows.append({"signal_date": date, "candidate_status": "NO_CANDIDATE"})
        else:
            chosen = eligible.iloc[0].to_dict()
            chosen["candidate_status"] = "CANDIDATE"
            chosen["purchase_status"] = "INSUFFICIENT_CASH" if float(chosen["EntryPrice"]) * 100 > 300_000 else "AFFORDABLE"
            rows.append(chosen)
    return pd.DataFrame(rows)


def run_synthetic_smoke_test(universe: pd.DataFrame) -> pd.DataFrame:
    """Execute the entire Phase 1 transformation path with no network access."""
    tickers = universe["ticker"].head(3).tolist()
    dates = pd.date_range("2015-01-01", periods=400, freq="B")
    prices: dict[str, pd.DataFrame] = {}
    for offset, ticker in enumerate(tickers):
        close = 1_000.0 + offset * 20 + np.arange(len(dates), dtype=float) * (1 + offset * .1)
        prices[ticker] = pd.DataFrame({
            "Open": close, "High": close + 3, "Low": close - 3,
            "Close": close, "Adj Close": close, "Volume": 200_000.0,
        }, index=dates)
    features = build_feature_frame(prices, universe.loc[universe["ticker"].isin(tickers)].copy())
    labelled = add_execution_labels(features, prices, {ticker: set() for ticker in tickers})
    candidates = select_daily_candidates(labelled)
    selected = candidates.loc[candidates["candidate_status"] == "CANDIDATE"]
    if selected.empty or not set(FEATURE_COLUMNS).issubset(features.columns):
        raise AssertionError("SYNTHETIC_SMOKE_FEATURE_OR_CANDIDATE_FAILURE")
    if selected[["EntryDate", "ExitDate", "label"]].isna().any().any():
        raise AssertionError("SYNTHETIC_SMOKE_LABEL_FAILURE")
    return candidates


_CANDIDATE_REQUIRED_COLUMNS = (
    "candidate_status", "eligible", "signal_date", "ticker", "label",
    "LabelConfirmedDate", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice",
    "ExitReason", "realized_net_return_percent", *FEATURE_COLUMNS,
)


def validate_candidate_samples(candidates: pd.DataFrame) -> pd.DataFrame:
    """Fail closed before a model sees deterministic daily candidate samples."""
    missing = [column for column in _CANDIDATE_REQUIRED_COLUMNS if column not in candidates.columns]
    if missing:
        raise ValueError(f"CANDIDATE_REQUIRED_COLUMNS_MISSING:{missing}")
    result = candidates.copy()
    candidate_rows = result["candidate_status"] == "CANDIDATE"
    if (~result["candidate_status"].isin(["CANDIDATE", "NO_CANDIDATE"])).any():
        raise ValueError("INVALID_CANDIDATE_STATUS")
    for column in ("signal_date", "LabelConfirmedDate", "EntryDate", "ExitDate"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
        if result.loc[candidate_rows, column].isna().any():
            raise ValueError(f"INVALID_CANDIDATE_DATE:{column}")
    if (result.loc[candidate_rows, ["signal_date", "LabelConfirmedDate", "EntryDate", "ExitDate"]] >= pd.Timestamp("2020-01-01")).any().any():
        raise ValueError("PROHIBITED_POST_2019_CANDIDATE")
    if result.loc[candidate_rows, "ticker"].isna().any() or (result.loc[candidate_rows, "ticker"].astype(str).str.strip() == "").any():
        raise ValueError("INVALID_CANDIDATE_TICKER")
    if result.loc[candidate_rows, "signal_date"].duplicated().any():
        raise ValueError("DUPLICATE_DAILY_CANDIDATE")
    labels = result.loc[candidate_rows, "label"]
    if (~labels.isin([0, 1])).any():
        raise ValueError("INVALID_CANDIDATE_LABEL")
    values = result.loc[candidate_rows, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("NONFINITE_CANDIDATE_FEATURE")
    if not result.loc[candidate_rows, "eligible"].eq(True).all():
        raise ValueError("INELIGIBLE_CANDIDATE")
    return result.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True)


def make_walk_forward_fold(candidates: pd.DataFrame, fold_spec: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract sorted train/test samples, applying the pre-test label embargo."""
    valid = validate_candidate_samples(candidates)
    train_from, train_to = pd.Timestamp(fold_spec["train_from"]), pd.Timestamp(fold_spec["train_to"])
    test_from, test_to = pd.Timestamp(fold_spec["test_from"]), pd.Timestamp(fold_spec["test_to"])
    candidate = valid["candidate_status"].eq("CANDIDATE")
    train = valid.loc[candidate & valid["signal_date"].between(train_from, train_to) & (valid["LabelConfirmedDate"] < test_from)]
    test = valid.loc[candidate & valid["signal_date"].between(test_from, test_to)]
    return (
        train.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True),
        test.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True),
    )


def build_meta_label_model() -> LGBMClassifier:
    """Construct exactly the preregistered pooled binary classifier."""
    return LGBMClassifier(**MODEL_PARAMS)


def check_fold_data_sufficiency(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Report Phase 2A availability blockers without inventing substitute data."""
    reasons: list[str] = []
    train_labels, test_labels = train["label"], test["label"]
    positive, negative = int((train_labels == 1).sum()), int((train_labels == 0).sum())
    if len(train) < 100: reasons.append("TRAIN_CANDIDATES_LT_100")
    if train_labels.nunique() != 2: reasons.append("TRAIN_LABEL_NOT_TWO_CLASSES")
    if positive < 20: reasons.append("TRAIN_POSITIVE_LT_20")
    if negative < 20: reasons.append("TRAIN_NEGATIVE_LT_20")
    if test_labels.nunique() != 2: reasons.append("TEST_LABEL_NOT_TWO_CLASSES")
    return {"blocked": bool(reasons), "reasons": reasons, "train_count": int(len(train)), "test_count": int(len(test)), "train_positive": positive, "train_negative": negative}


def generate_oof_predictions(candidates: pd.DataFrame, model_factory: Any = None) -> pd.DataFrame:
    """Fit exactly one fixed classifier per walk-forward fold and return OOF rows."""
    factory = build_meta_label_model if model_factory is None else model_factory
    outputs = []
    for fold in FOLDS:
        train, test = make_walk_forward_fold(candidates, fold)
        sufficiency = check_fold_data_sufficiency(train, test)
        if sufficiency["blocked"]:
            raise ValueError(f"FOLD_{fold['fold']}_BLOCKED:{','.join(sufficiency['reasons'])}")
        model = factory()
        model.fit(train.loc[:, FEATURE_COLUMNS], train["label"])
        probability = np.asarray(model.predict_proba(test.loc[:, FEATURE_COLUMNS]))[:, 1]
        part = test.loc[:, ["signal_date", "ticker", "label", "realized_net_return_percent", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "ExitReason", *FEATURE_COLUMNS]].copy()
        part.insert(0, "fold", fold["fold"])
        part["probability"] = probability
        part["decision"] = np.where(part["probability"] >= .55, "ACCEPT", "ABSTAIN")
        outputs.append(part)
    columns = ["fold", "signal_date", "ticker", "label", "probability", "decision", "realized_net_return_percent", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "ExitReason", *FEATURE_COLUMNS]
    return pd.concat(outputs, ignore_index=True).loc[:, columns].sort_values(["fold", "signal_date", "ticker"], kind="mergesort").reset_index(drop=True)


def classification_metrics(oof: pd.DataFrame) -> dict[str, Any]:
    """Classification-only metrics; single-class inputs explicitly remain blocked."""
    labels = oof["label"]
    if len(oof) == 0 or labels.nunique() != 2:
        return {"status": "BLOCKED", "reason": "LABEL_NOT_TWO_CLASSES"}
    probabilities = oof["probability"].astype(float)
    accepted = oof.loc[oof["decision"] == "ACCEPT", "realized_net_return_percent"]
    abstained = oof.loc[oof["decision"] == "ABSTAIN", "realized_net_return_percent"]
    accept_mean = float(accepted.mean()) if len(accepted) else None
    abstain_mean = float(abstained.mean()) if len(abstained) else None
    return {
        "status": "OK", "sample_count": int(len(oof)), "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)), "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "probability_minimum": float(probabilities.min()), "probability_maximum": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()), "probability_median": float(probabilities.median()),
        "accept_rate_at_055": float((probabilities >= .55).mean()),
        "accept_mean_realized_net_return_percent": accept_mean,
        "abstain_mean_realized_net_return_percent": abstain_mean,
        "accept_minus_abstain_mean_realized_net_return_percent": None if accept_mean is None or abstain_mean is None else accept_mean - abstain_mean,
    }


def make_synthetic_phase2a_candidates() -> pd.DataFrame:
    """Deterministic offline candidates satisfying every Phase 2A fold minimum."""
    dates = pd.date_range("2016-04-01", "2019-12-27", freq="B")
    rows = []
    for position, date in enumerate(dates):
        label = position % 2
        row = {"candidate_status": "CANDIDATE", "eligible": True, "signal_date": date, "ticker": "3633",
               "label": label, "LabelConfirmedDate": date + pd.offsets.BDay(2), "EntryDate": date + pd.offsets.BDay(1),
               "ExitDate": date + pd.offsets.BDay(2), "EntryPrice": 1_000.0, "ExitPrice": 1_000.0 + (10 if label else -10),
               "ExitReason": "TIME", "realized_net_return_percent": 1.0 if label else -1.0}
        row.update({column: ((position + index * 3) % 17) / 17 for index, column in enumerate(FEATURE_COLUMNS)})
        rows.append(row)
    return pd.DataFrame(rows)


PORTFOLIO_REQUIRED_COLUMNS = (
    "fold", "signal_date", "ticker", "probability", "decision", "EntryDate", "ExitDate",
    "EntryPrice", "ExitPrice", "ExitReason", "realized_net_return_percent", *FEATURE_COLUMNS,
)
PORTFOLIO_STARTING_CASH = 300_000.0
PORTFOLIO_QUANTITY = 100


def validate_portfolio_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Validate the immutable Phase 1/2A opportunity fields before execution."""
    missing = [column for column in PORTFOLIO_REQUIRED_COLUMNS if column not in oof.columns]
    if missing:
        raise ValueError(f"PORTFOLIO_REQUIRED_COLUMNS_MISSING:{missing}")
    result = oof.copy()
    if (~result["fold"].isin([1, 2, 3])).any(): raise ValueError("INVALID_PORTFOLIO_FOLD")
    for column in ("signal_date", "EntryDate", "ExitDate"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
        if result[column].isna().any(): raise ValueError(f"INVALID_PORTFOLIO_DATE:{column}")
    if (result[["signal_date", "EntryDate", "ExitDate"]] >= pd.Timestamp("2020-01-01")).any().any(): raise ValueError("PROHIBITED_POST_2019_PORTFOLIO_ROW")
    if not (result["EntryDate"] > result["signal_date"]).all() or not (result["ExitDate"] >= result["EntryDate"]).all(): raise ValueError("INVALID_PORTFOLIO_DATE_ORDER")
    if result["ticker"].isna().any() or (result["ticker"].astype(str).str.strip() == "").any(): raise ValueError("INVALID_PORTFOLIO_TICKER")
    if result.duplicated(["fold", "signal_date", "ticker"]).any(): raise ValueError("DUPLICATE_PORTFOLIO_OPPORTUNITY")
    for column in ("EntryPrice", "ExitPrice", "probability"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[["EntryPrice", "ExitPrice", "probability"]].to_numpy(dtype=float)).all() or (result[["EntryPrice", "ExitPrice"]] <= 0).any().any(): raise ValueError("INVALID_PORTFOLIO_PRICE_OR_PROBABILITY")
    if not result["probability"].between(0, 1).all(): raise ValueError("INVALID_PORTFOLIO_PROBABILITY")
    expected = np.where(result["probability"] >= .55, "ACCEPT", "ABSTAIN")
    if not (result["decision"].to_numpy() == expected).all(): raise ValueError("INVALID_PORTFOLIO_DECISION")
    feature_values = result.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(feature_values.to_numpy(dtype=float)).all(): raise ValueError("NONFINITE_PORTFOLIO_FEATURE")
    return result.sort_values(["fold", "EntryDate", "signal_date", "ticker"], kind="mergesort").reset_index(drop=True)


def _portfolio_record(row: Mapping[str, Any], strategy: str, industry: str, status: str, skip: str | None,
                      cash_before: float, cash_after_entry: float, cash_after_exit: float) -> dict[str, Any]:
    entry, exit_ = float(row["EntryPrice"]), float(row["ExitPrice"])
    quantity = PORTFOLIO_QUANTITY
    filled = status == "FILLED"
    entry_cost, proceeds = (entry * quantity, exit_ * quantity) if filled else (0.0, 0.0)
    commission = 0.0
    return {"strategy": strategy, "fold": int(row["fold"]), "signal_date": row["signal_date"], "ticker": str(row["ticker"]), "industry": industry,
            "EntryDate": row["EntryDate"], "ExitDate": row["ExitDate"], "EntryPrice": entry, "ExitPrice": exit_, "ExitReason": row["ExitReason"],
            "quantity": quantity if filled else 0, "entry_cost": entry_cost, "exit_proceeds": proceeds, "commission_cost": commission,
            "realized_net_profit_yen": (proceeds - entry_cost - commission) if filled else 0.0,
            "label": int(row["label"]) if "label" in row else None,
            "realized_net_return_percent": float(row["realized_net_return_percent"]), "probability": float(row["probability"]),
            "model_decision": row["decision"], "portfolio_status": status, "skip_reason": skip,
            "cash_before": cash_before, "cash_after_entry": cash_after_entry, "cash_after_exit": cash_after_exit,
            **{column: float(row[column]) for column in FEATURE_COLUMNS}}


def _execute_fixed_lot(opportunities: pd.DataFrame, strategy: str, industry_map: Mapping[str, str], accept_only: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders, ledgers, events = [], [], []
    for fold, group in opportunities.groupby("fold", sort=True):
        cash, pending, position, sequence = PORTFOLIO_STARTING_CASH, 0.0, None, 0
        initial = {"fold": int(fold), "strategy": strategy, "date": pd.Timestamp(group["EntryDate"].min()) - pd.Timedelta(days=1), "available_cash": cash, "pending_cash": 0.0, "locked_entry_capital": 0.0, "open_positions": 0, "equity": cash}
        ledgers.append(initial)
        entries = {date: frame.sort_values(["signal_date", "ticker"], kind="mergesort") for date, frame in group.groupby("EntryDate", sort=True)}
        dates = sorted(set(group["EntryDate"]) | set(group["ExitDate"]))
        for date in dates:
            before_cash, before_pending = cash, pending
            cash += pending; pending = 0.0
            events.append({"fold": int(fold), "strategy": strategy, "date": date, "sequence": sequence, "event_type": "PRIOR_PENDING_RELEASE", "ticker": None, "signal_date": pd.NaT, "available_cash_before": before_cash, "available_cash_after": cash, "pending_cash_before": before_pending, "pending_cash_after": pending, "amount": before_pending}); sequence += 1
            for _, row in entries.get(date, pd.DataFrame()).iterrows():
                industry = industry_map.get(str(row["ticker"]), "MISSING")
                before = cash
                if accept_only and row["decision"] == "ABSTAIN":
                    orders.append(_portfolio_record(row, strategy, industry, "ABSTAIN", "MODEL_ABSTAIN", before, cash, cash)); continue
                if position is not None:
                    reason = "SAME_DAY_PROCEEDS_UNAVAILABLE" if pd.Timestamp(position["ExitDate"]) == date else "MAX_OPEN_POSITION"
                    orders.append(_portfolio_record(row, strategy, industry, "SKIPPED", reason, before, cash, cash)); continue
                cost = float(row["EntryPrice"]) * PORTFOLIO_QUANTITY
                if not np.isfinite(cost) or cost <= 0:
                    orders.append(_portfolio_record(row, strategy, industry, "SKIPPED", "INVALID_ORDER", before, cash, cash)); continue
                if cash + 1e-8 < cost:
                    orders.append(_portfolio_record(row, strategy, industry, "SKIPPED", "INSUFFICIENT_CASH", before, cash, cash)); continue
                cash -= cost
                position = row.to_dict()
                events.append({"fold": int(fold), "strategy": strategy, "date": date, "sequence": sequence, "event_type": "ENTRY_FILLED", "ticker": str(row["ticker"]), "signal_date": row["signal_date"], "available_cash_before": before, "available_cash_after": cash, "pending_cash_before": pending, "pending_cash_after": pending, "amount": cost}); sequence += 1
                record = _portfolio_record(row, strategy, industry, "FILLED", None, before, cash, cash)
                orders.append(record)
            if position is not None and pd.Timestamp(position["ExitDate"]) == date:
                proceeds = float(position["ExitPrice"]) * PORTFOLIO_QUANTITY
                before_pending = pending
                pending += proceeds
                events.append({"fold": int(fold), "strategy": strategy, "date": date, "sequence": sequence, "event_type": "EXIT_TO_PENDING", "ticker": str(position["ticker"]), "signal_date": position["signal_date"], "available_cash_before": cash, "available_cash_after": cash, "pending_cash_before": before_pending, "pending_cash_after": pending, "amount": proceeds}); sequence += 1
                for record in reversed(orders):
                    if record["portfolio_status"] == "FILLED" and record["fold"] == fold and record["signal_date"] == position["signal_date"] and record["ticker"] == position["ticker"]:
                        record["cash_after_exit"] = cash
                        break
                position = None
            locked = float(position["EntryPrice"]) * PORTFOLIO_QUANTITY if position is not None else 0.0
            ledgers.append({"fold": int(fold), "strategy": strategy, "date": date, "available_cash": cash, "pending_cash": pending, "locked_entry_capital": locked, "open_positions": int(position is not None), "equity": cash + pending + locked})
            if cash < -1e-8: raise AssertionError("NEGATIVE_CASH")
    return pd.DataFrame(orders), pd.DataFrame(ledgers), pd.DataFrame(events)


def run_baseline_portfolio(oof: pd.DataFrame, universe: pd.DataFrame | None = None, return_events: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = validate_portfolio_oof(oof)
    industries = {} if universe is None else universe.set_index("ticker")["industry"].astype(str).to_dict()
    orders, ledger, events = _execute_fixed_lot(valid, "BASELINE", industries, False)
    orders.attrs["event_ledger"] = events
    return (orders, ledger, events) if return_events else (orders, ledger)


def run_v4_portfolio(baseline_orders: pd.DataFrame, return_events: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opportunities = baseline_orders.loc[baseline_orders["portfolio_status"] == "FILLED"].copy()
    if opportunities.empty:
        empty_orders, empty_ledger, empty_events = pd.DataFrame(columns=baseline_orders.columns), pd.DataFrame(), pd.DataFrame()
        return (empty_orders, empty_ledger, empty_events) if return_events else (empty_orders, empty_ledger)
    opportunities["decision"] = opportunities["model_decision"]
    industries = opportunities.set_index("ticker")["industry"].astype(str).to_dict()
    orders, ledger, events = _execute_fixed_lot(opportunities, "V4", industries, True)
    orders.attrs["event_ledger"] = events
    return (orders, ledger, events) if return_events else (orders, ledger)


def cash_safety_audit(events: pd.DataFrame) -> dict[str, int]:
    """Mechanically derive cash-safety counters from the ordered cash event ledger."""
    required = {"fold", "date", "sequence", "event_type", "available_cash_before", "available_cash_after", "pending_cash_before", "pending_cash_after", "amount", "ticker", "signal_date"}
    if not required.issubset(events.columns):
        raise ValueError("CASH_EVENT_SCHEMA_MISSING")
    work = events.sort_values(["fold", "date", "sequence"], kind="mergesort")
    negative = int(((work["available_cash_before"] < -1e-8) | (work["available_cash_after"] < -1e-8)).sum())
    violations = 0
    for (_, _), day in work.groupby(["fold", "date"], sort=False):
        exits_seen = False
        for event in day.itertuples():
            if event.event_type == "EXIT_TO_PENDING":
                exits_seen = True
                if abs(float(event.available_cash_after) - float(event.available_cash_before)) > 1e-8: violations += 1
            elif event.event_type == "ENTRY_FILLED":
                if float(event.available_cash_before) + 1e-8 < float(event.amount): violations += 1
                if exits_seen: violations += 1
                if abs(float(event.pending_cash_after) - float(event.pending_cash_before)) > 1e-8: violations += 1
    entries = work.loc[work["event_type"] == "ENTRY_FILLED"]
    duplicate = int(entries.duplicated(["fold", "ticker", "signal_date"]).sum())
    return {"negative_cash_count": negative, "capital_reuse_count": violations, "duplicate_order_count": duplicate}


def baseline_filled_acceptance_evidence(baseline_orders: pd.DataFrame) -> dict[str, Any]:
    """Acceptance denominator is exactly the fixed Baseline FILLED opportunity set."""
    filled = baseline_orders.loc[baseline_orders["portfolio_status"] == "FILLED"]
    if filled.empty:
        return {"status": "BLOCKED", "reason": "BASELINE_FILLED_OPPORTUNITIES_EMPTY", "baseline_filled_opportunity_count": 0, "model_acceptance_count": 0, "model_abstain_count": 0, "model_acceptance_rate": None}
    accepted = int(filled["model_decision"].eq("ACCEPT").sum())
    count = int(len(filled))
    return {"status": "OK", "baseline_filled_opportunity_count": count, "model_acceptance_count": accepted, "model_abstain_count": count - accepted, "model_acceptance_rate": accepted / count}


def portfolio_metrics(orders: pd.DataFrame, ledger: pd.DataFrame, events: pd.DataFrame | None = None) -> dict[str, Any]:
    filled = orders.loc[orders["portfolio_status"] == "FILLED"].copy()
    profit = float(filled["realized_net_profit_yen"].sum()) if len(filled) else 0.0
    equity = ledger["equity"] if len(ledger) else pd.Series([PORTFOLIO_STARTING_CASH])
    drawdown = (equity.cummax() - equity) / equity.cummax() * 100
    monthly = filled.assign(month=filled["ExitDate"].dt.to_period("M")).groupby("month")["realized_net_profit_yen"].sum() if len(filled) else pd.Series(dtype=float)
    yearly = filled.assign(year=filled["ExitDate"].dt.year).groupby("year")["realized_net_profit_yen"].sum() if len(filled) else pd.Series(dtype=float)
    positives = filled.loc[filled["realized_net_profit_yen"] > 0]
    denominator = float(positives["realized_net_profit_yen"].sum())
    by_ticker = positives.groupby("ticker")["realized_net_profit_yen"].sum() if len(positives) else pd.Series(dtype=float)
    by_industry = positives.groupby("industry")["realized_net_profit_yen"].sum() if len(positives) else pd.Series(dtype=float)
    event_frame = events if events is not None else orders.attrs.get("event_ledger", pd.DataFrame())
    safety = cash_safety_audit(event_frame) if len(event_frame) else {"negative_cash_count": 0, "capital_reuse_count": 0, "duplicate_order_count": 0}
    return {"net_profit": profit, "ending_equity": PORTFOLIO_STARTING_CASH + profit, "max_drawdown_percent": float(drawdown.max()), "closed_trades": int(len(filled)),
            "win_rate": float((filled["realized_net_profit_yen"] > 0).mean()) if len(filled) else 0.0, "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
            "yearly_net_profit": {str(key): float(value) for key, value in yearly.items()}, "insufficient_cash_count": int(orders["skip_reason"].eq("INSUFFICIENT_CASH").sum()),
            "stop_count": int((filled["ExitReason"] == "STOP").sum()), "gap_stop_count": int((filled["ExitReason"] == "GAP_STOP").sum()), "time_count": int((filled["ExitReason"] == "TIME").sum()),
            **safety,
            "model_acceptance_count": int(orders["model_decision"].eq("ACCEPT").sum()), "model_abstain_count": int(orders["model_decision"].eq("ABSTAIN").sum()),
            "model_acceptance_rate": float(orders["model_decision"].eq("ACCEPT").mean()) if len(orders) else 0.0,
            "accept_insufficient_cash_count": int(((orders["model_decision"] == "ACCEPT") & (orders["skip_reason"] == "INSUFFICIENT_CASH")).sum()),
            "max_stock_positive_profit_share": float(by_ticker.max() / denominator) if denominator else 0.0, "top5_stock_positive_profit_share": float(by_ticker.nlargest(5).sum() / denominator) if denominator else 0.0,
            "max_industry_positive_profit_share": float(by_industry.max() / denominator) if denominator else 0.0}


def aggregate_portfolio_metrics(orders: pd.DataFrame, ledger: pd.DataFrame, events: pd.DataFrame | None = None) -> dict[str, Any]:
    event_frame = events if events is not None else orders.attrs.get("event_ledger", pd.DataFrame())
    folds = {str(fold): portfolio_metrics(orders.loc[orders["fold"] == fold], ledger.loc[ledger["fold"] == fold], event_frame.loc[event_frame["fold"] == fold] if len(event_frame) else None) for fold in (1, 2, 3)}
    aggregate_profit = sum(item["net_profit"] for item in folds.values())
    overall = portfolio_metrics(orders, ledger, event_frame if len(event_frame) else None)
    overall.update({"aggregate_net_profit": aggregate_profit, "aggregate_ending_equity_equivalent": PORTFOLIO_STARTING_CASH + aggregate_profit,
                    "max_drawdown_percent": max((item["max_drawdown_percent"] for item in folds.values()), default=0.0), "folds": folds})
    return overall


def baseline_filled_classification_metrics(baseline_orders: pd.DataFrame) -> dict[str, Any]:
    """Use only fixed Baseline fills, never all OOF candidates, for Phase 2B metrics."""
    filled = baseline_orders.loc[baseline_orders["portfolio_status"] == "FILLED"].copy()
    filled["decision"] = filled["model_decision"]
    overall = classification_metrics(filled)
    return {"overall": overall, "folds": {str(fold): classification_metrics(filled.loc[filled["fold"] == fold]) for fold in (1, 2, 3)}}


def evaluate_blocked_conditions(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate every preregistered blocker; callers inject audit evidence in formal runs."""
    reasons: list[str] = []
    price = evidence.get("price_success_tickers")
    if not isinstance(price, int): reasons.append("PRICE_SUCCESS_TICKERS_EVIDENCE_MISSING_OR_INVALID")
    elif price < 150: reasons.append("PRICE_SUCCESS_TICKERS_LT_150")
    sufficiency, closed = evidence.get("fold_sufficiency"), evidence.get("baseline_closed_trades")
    if not isinstance(sufficiency, Mapping): reasons.append("FOLD_SUFFICIENCY_EVIDENCE_MISSING_OR_INVALID"); sufficiency = {}
    if not isinstance(closed, Mapping): reasons.append("BASELINE_CLOSED_TRADES_EVIDENCE_MISSING_OR_INVALID"); closed = {}
    for fold in (1, 2, 3):
        key = str(fold)
        status = sufficiency.get(key, sufficiency.get(fold))
        if not isinstance(status, Mapping) or not isinstance(status.get("reasons"), list): reasons.append(f"FOLD_{fold}_SUFFICIENCY_EVIDENCE_MISSING")
        else:
            for reason in status["reasons"]: reasons.append(f"FOLD_{fold}_{reason}")
        count = closed.get(key, closed.get(fold))
        if not isinstance(count, int): reasons.append(f"FOLD_{fold}_BASELINE_CLOSED_TRADES_EVIDENCE_MISSING")
        elif count < 40: reasons.append(f"FOLD_{fold}_BASELINE_CLOSED_TRADES_LT_40")
    for key, reason in (("hashes_fixed", "REQUIRED_HASH_EVIDENCE_MISSING_OR_INVALID"), ("network_hosts_allowed", "NETWORK_HOST_EVIDENCE_MISSING_OR_INVALID"), ("deterministic", "DETERMINISM_EVIDENCE_MISSING_OR_INVALID")):
        if not isinstance(evidence.get(key), bool): reasons.append(reason)
    if evidence.get("hashes_fixed") is False: reasons.append("REQUIRED_HASH_NOT_FIXED")
    post = evidence.get("post_2020_rows")
    if not isinstance(post, int): reasons.append("POST_2020_ROWS_EVIDENCE_MISSING_OR_INVALID")
    elif post > 0: reasons.append("POST_2020_ROWS_DETECTED")
    if evidence.get("network_hosts_allowed") is False: reasons.append("PROHIBITED_NETWORK_HOST_DETECTED")
    if evidence.get("deterministic") is False: reasons.append("DETERMINISM_NOT_CONFIRMED")
    return {"status": "FREE_META_LABEL_PROTOTYPE_BLOCKED" if reasons else "CLEAR", "reasons": reasons}


def evaluate_acceptance_conditions(baseline: Mapping[str, Any], v4: Mapping[str, Any], classification: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Return the 17 pre-registered conditions with BLOCKED taking strict priority."""
    blocked = evaluate_blocked_conditions(audit)
    if blocked["reasons"]:
        return {"status": "FREE_META_LABEL_PROTOTYPE_BLOCKED", "blocked_reasons": blocked["reasons"], "conditions": []}
    base_folds, v4_folds = baseline["folds"], v4["folds"]
    overall, fold_class = classification["overall"], classification["folds"]
    acceptance_rate = float(audit["model_acceptance_rate"])
    zero_safety = all(value == 0 for metrics in (baseline, v4) for value in (metrics.get("negative_cash_count", 0), metrics.get("capital_reuse_count", 0), metrics.get("duplicate_order_count", 0)))
    specs = [
        ("aggregate_net_profit_beats_baseline", v4["aggregate_net_profit"] > baseline["aggregate_net_profit"], v4["aggregate_net_profit"], f"> {baseline['aggregate_net_profit']}"),
        ("aggregate_net_profit_positive", v4["aggregate_net_profit"] > 0, v4["aggregate_net_profit"], "> 0"),
        ("max_drawdown_below_baseline", v4["max_drawdown_percent"] < baseline["max_drawdown_percent"], v4["max_drawdown_percent"], f"< {baseline['max_drawdown_percent']}"),
        ("two_folds_profit_beat_baseline", sum(v4_folds[str(f)]["net_profit"] > base_folds[str(f)]["net_profit"] for f in (1,2,3)) >= 2, {"winning_fold_count": sum(v4_folds[str(f)]["net_profit"] > base_folds[str(f)]["net_profit"] for f in (1,2,3)), "folds": {str(f): {"baseline_net_profit": base_folds[str(f)]["net_profit"], "v4_net_profit": v4_folds[str(f)]["net_profit"], "v4_beats_baseline": v4_folds[str(f)]["net_profit"] > base_folds[str(f)]["net_profit"]} for f in (1,2,3)}}, ">= 2 folds"),
        ("all_folds_drawdown_not_above_baseline", all(v4_folds[str(f)]["max_drawdown_percent"] <= base_folds[str(f)]["max_drawdown_percent"] for f in (1,2,3)), {"passing_fold_count": sum(v4_folds[str(f)]["max_drawdown_percent"] <= base_folds[str(f)]["max_drawdown_percent"] for f in (1,2,3)), "folds": {str(f): {"baseline_max_drawdown_percent": base_folds[str(f)]["max_drawdown_percent"], "v4_max_drawdown_percent": v4_folds[str(f)]["max_drawdown_percent"]} for f in (1,2,3)}}, "all folds <= baseline"),
        ("win_rate_beats_baseline", v4["win_rate"] > baseline["win_rate"], v4["win_rate"], f"> {baseline['win_rate']}"),
        ("closed_trades_at_least_100", v4["closed_trades"] >= 100, v4["closed_trades"], ">= 100"),
        ("acceptance_rate_20_to_80_percent", .2 <= acceptance_rate <= .8, acceptance_rate, "0.20 <= rate <= 0.80"),
        ("overall_roc_auc_above_052", overall.get("roc_auc") is not None and overall.get("roc_auc") > .52, overall.get("roc_auc"), "> 0.52"),
        ("two_folds_roc_auc_above_050", sum((fold_class[str(f)].get("roc_auc") or -np.inf) > .50 for f in (1,2,3)) >= 2, {"passing_fold_count": sum((fold_class[str(f)].get("roc_auc") or -np.inf) > .50 for f in (1,2,3)), "fold_roc_auc": {str(f): fold_class[str(f)].get("roc_auc") for f in (1,2,3)}}, ">= 2 folds"),
        ("max_stock_positive_profit_share_at_most_35_percent", v4["max_stock_positive_profit_share"] <= .35, v4["max_stock_positive_profit_share"], "<= 0.35"),
        ("top5_stock_positive_profit_share_at_most_60_percent", v4["top5_stock_positive_profit_share"] <= .60, v4["top5_stock_positive_profit_share"], "<= 0.60"),
        ("max_industry_positive_profit_share_at_most_50_percent", v4["max_industry_positive_profit_share"] <= .50, v4["max_industry_positive_profit_share"], "<= 0.50"),
        ("no_cash_reuse_or_duplicate_orders", zero_safety, {"baseline_negative_cash_count": baseline.get("negative_cash_count"), "baseline_capital_reuse_count": baseline.get("capital_reuse_count"), "baseline_duplicate_order_count": baseline.get("duplicate_order_count"), "v4_negative_cash_count": v4.get("negative_cash_count"), "v4_capital_reuse_count": v4.get("capital_reuse_count"), "v4_duplicate_order_count": v4.get("duplicate_order_count")}, "all counters = 0"),
        ("two_full_runs_byte_identical", isinstance(audit.get("byte_identical"), bool) and audit.get("byte_identical") is True, audit.get("byte_identical"), "True"),
        ("no_post_2020_rows", int(audit.get("post_2020_rows", 0)) == 0, audit.get("post_2020_rows", 0), "0"),
        ("no_prohibited_network_calls", isinstance(audit.get("network_hosts_allowed"), bool) and audit.get("network_hosts_allowed") is True, audit.get("network_hosts_allowed"), "True"),
    ]
    conditions = [{"condition_number": index + 1, "name": name, "passed": bool(passed), "actual_value": actual, "required_value": required} for index, (name, passed, actual, required) in enumerate(specs)]
    return {"status": "FREE_META_LABEL_PROTOTYPE_PROMISING" if all(item["passed"] for item in conditions) else "FREE_META_LABEL_PROTOTYPE_NOT_PROMISING", "blocked_reasons": [], "conditions": conditions}


def make_synthetic_phase2b_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hand-made offline OOF rows covering fixed-lot execution edge cases."""
    rows = []
    for fold, year in ((1, 2017), (2, 2018), (3, 2019)):
        dates = pd.date_range(f"{year}-01-01", periods=7, freq="B")
        specs = [
            ("3633", dates[0], dates[1], dates[2], 1000., 1100., .60, "TIME"),
            ("2984", dates[1], dates[2], dates[3], 1000., 900., .40, "STOP"),
            ("6150", dates[2], dates[3], dates[4], 4000., 4100., .60, "GAP_STOP"),
            ("7203", dates[2], dates[3], dates[5], 1000., 900., .40, "TIME"),
        ]
        for index, (ticker, signal, entry, exit_, entry_price, exit_price, probability, reason) in enumerate(specs):
            label = int(exit_price > entry_price)
            row = {"fold": fold, "signal_date": signal, "ticker": ticker, "label": label, "probability": probability,
                   "decision": "ACCEPT" if probability >= .55 else "ABSTAIN", "realized_net_return_percent": (exit_price / entry_price - 1) * 100,
                   "EntryDate": entry, "ExitDate": exit_, "EntryPrice": entry_price, "ExitPrice": exit_price, "ExitReason": reason}
            row.update({column: (index + feature_index + fold) / 100 for feature_index, column in enumerate(FEATURE_COLUMNS)})
            rows.append(row)
    universe = pd.DataFrame({"ticker": ["3633", "2984", "6150", "7203"], "industry": ["A", "B", "C", "A"], "market": ["M"] * 4})
    return pd.DataFrame(rows), universe
