"""V9_009 cache-compatible TOP1 development kill screen.

This module contains the pure V9 feature/target/model/ranking computation and
the read-only fixed-cache adapter.  The public result is deliberately closed:
it carries only the binary development verdict (or data-incompatibility
state), provenance hashes/counts, and validation booleans.  Detailed scores,
returns, percentiles, identities, and effect metrics remain transient.

No source transport, cache writer, portfolio engine, T1 path, or terminal-data
path is present here.  Synthetic callers use the same pure computation path as
the future fixed-cache runner.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TASK = "V9_009_T0_TOP1_KILL_SCREEN"
DESIGN_SHA = "8079bb0956c105a3972e59f0e4ba21ea5b81b14a"
FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20",
    "atr14_percent",
    "close_to_ma20",
    "close_to_ma60",
    "distance_from_high20",
    "volume_dryup",
)
SOURCE_FEATURE_HISTORY_START = pd.Timestamp("2016-09-01")
PRE_EVALUATION_TRAINING_START = pd.Timestamp("2017-01-01")
FORMAL_SIGNAL_START = pd.Timestamp("2018-01-01")
FORMAL_SIGNAL_END = pd.Timestamp("2025-12-31")
KILL_SCREEN_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
EVAL_YEARS = KILL_SCREEN_YEARS
FORMAL_YEARS = tuple(range(2018, 2026))
MAX_HISTORY_DAYS = 252
MIN_MEDIAN_TURNOVER = 100_000_000.0
UNIVERSE_MODE = "FIXED_V4_300"
TRAINING_MANIFEST_SHA256 = "72ae3db1186f2c9c113b1bafe1d37fb74a5627ac7ceed1dfc2473a24e060de85"
EVALUATION_MANIFEST_SHA256 = "797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5"
V4_UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
V4_TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"

RIDGE_PARAMS = {"alpha": 10.0, "fit_intercept": True}
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.02,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 100,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
    "random_state": 20260823,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_CODE = re.compile(r"^[0-9A-Z]{1,32}$")
_RAW_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_ADJ_ALIASES = ("AdjClose", "Adj Close", "adjusted_close")


class T0DataIncompatible(ValueError):
    """An exact V9 input contract cannot be constructed."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class T0ImplementationFailure(ValueError):
    """A malformed computation/result must fail closed."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def model_parameter_fingerprint() -> str:
    return sha256_bytes(canonical_json_bytes({"ridge": RIDGE_PARAMS, "lightgbm": LIGHTGBM_PARAMS}))


def feature_fingerprint() -> str:
    return sha256_bytes(canonical_json_bytes(FEATURE_COLUMNS))


def canonical_code(value: object) -> str:
    if not isinstance(value, (str, int, np.integer)) or isinstance(value, bool):
        raise T0DataIncompatible("CANONICAL_CODE_INVALID")
    text = str(value).strip().upper()
    if not _CODE.fullmatch(text):
        raise T0DataIncompatible("CANONICAL_CODE_INVALID")
    return text


def _as_day(value: object) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise T0DataIncompatible("DATE_INVALID") from error
    if pd.isna(parsed):
        raise T0DataIncompatible("DATE_INVALID")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Tokyo").tz_localize(None)
    parsed = parsed.normalize()
    return parsed


def _finite_positive(values: pd.DataFrame | pd.Series) -> bool:
    array = values.to_numpy(dtype=float)
    return bool(np.isfinite(array).all() and (array > 0).all())


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise T0DataIncompatible("OHLCV_FRAME_INVALID")
    result = frame.copy()
    aliases = {"adjusted_close": "AdjClose", "Adj Close": "AdjClose"}
    for source, target in aliases.items():
        if source in result.columns and target not in result.columns:
            result[target] = result[source]
    if not set(_RAW_COLUMNS + ("AdjClose",)).issubset(result.columns):
        raise T0DataIncompatible("OHLCV_REQUIRED_COLUMNS_MISSING")
    index = pd.to_datetime(result.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("Asia/Tokyo").tz_localize(None)
    index = index.normalize()
    if index.isna().any() or index.duplicated().any():
        raise T0DataIncompatible("DUPLICATE_OR_INVALID_PRICE_DATE")
    result.index = index
    result = result.sort_index()
    required = list(_RAW_COLUMNS + ("AdjClose",))
    result.loc[:, required] = result.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if not _finite_positive(result.loc[:, required]):
        raise T0DataIncompatible("NONFINITE_OR_NONPOSITIVE_OHLCV")
    return result


def _normalise_actions(
    frame: pd.DataFrame,
    actions: Mapping[object, object] | Sequence[Mapping[str, object]] | None,
) -> dict[pd.Timestamp, float]:
    source: Any = actions
    if source is None and "split_ratio" in frame.columns:
        source = {
            day: value
            for day, value in frame["split_ratio"].items()
            if not pd.isna(value)
        }
    if source is None:
        return {}
    if isinstance(source, Mapping):
        items = list(source.items())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        items = []
        for item in source:
            if not isinstance(item, Mapping) or set(item) != {"date", "ratio"}:
                raise T0DataIncompatible("SPLIT_ACTION_SCHEMA_INVALID")
            items.append((item["date"], item["ratio"]))
    else:
        raise T0DataIncompatible("SPLIT_ACTION_SCHEMA_INVALID")
    result: dict[pd.Timestamp, float] = {}
    for raw_day, raw_ratio in items:
        day = _as_day(raw_day)
        if isinstance(raw_ratio, bool):
            raise T0DataIncompatible("SPLIT_RATIO_INVALID")
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError) as error:
            raise T0DataIncompatible("SPLIT_RATIO_INVALID") from error
        if not math.isfinite(ratio) or ratio <= 0 or day in result:
            raise T0DataIncompatible("SPLIT_RATIO_INVALID")
        result[day] = ratio
    return result


def normalize_inputs(
    frames: Mapping[object, pd.DataFrame],
    actions: Mapping[object, Mapping[object, object] | Sequence[Mapping[str, object]]] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[pd.Timestamp, float]]]:
    if not isinstance(frames, Mapping) or not frames:
        raise T0DataIncompatible("PRICE_FRAMES_EMPTY")
    normalized: dict[str, pd.DataFrame] = {}
    normalized_actions: dict[str, dict[pd.Timestamp, float]] = {}
    for raw_code, raw_frame in frames.items():
        code = canonical_code(raw_code)
        if code in normalized:
            raise T0DataIncompatible("DUPLICATE_CANONICAL_CODE")
        normalized[code] = _normalise_frame(raw_frame)
        action_source = actions.get(raw_code) if actions is not None and raw_code in actions else None
        if action_source is None and actions is not None:
            action_source = actions.get(code)
        normalized_actions[code] = _normalise_actions(normalized[code], action_source)
    return normalized, normalized_actions


def normalize_calendar(calendar_dates: Sequence[object]) -> pd.DatetimeIndex:
    if isinstance(calendar_dates, (str, bytes, bytearray)) or not isinstance(calendar_dates, (Sequence, pd.Index, np.ndarray)):
        raise T0DataIncompatible("CALENDAR_INVALID")
    parsed = [_as_day(value) for value in calendar_dates]
    index = pd.DatetimeIndex(parsed)
    if len(index) == 0 or index.duplicated().any() or not index.is_monotonic_increasing:
        raise T0DataIncompatible("CALENDAR_NOT_STRICTLY_SORTED")
    if any(day.weekday() >= 5 for day in index):
        raise T0DataIncompatible("CALENDAR_CONTAINS_WEEKEND")
    if not any(day >= FORMAL_SIGNAL_START for day in index):
        raise T0DataIncompatible("CALENDAR_ANCHOR_MISSING")
    return index


def signal_grid(calendar_dates: Sequence[object]) -> pd.DatetimeIndex:
    calendar = normalize_calendar(calendar_dates)
    anchor_positions = np.flatnonzero(calendar >= FORMAL_SIGNAL_START)
    if len(anchor_positions) == 0:
        raise T0DataIncompatible("CALENDAR_ANCHOR_MISSING")
    anchor = int(anchor_positions[0])
    positions = [
        position
        for position, day in enumerate(calendar)
        if day >= PRE_EVALUATION_TRAINING_START
        and day <= FORMAL_SIGNAL_END
        and (position - anchor) % 3 == 0
    ]
    if not positions:
        raise T0DataIncompatible("SIGNAL_GRID_EMPTY")
    return calendar[positions]


def _causal_normalized_frame(
    frame: pd.DataFrame,
    split_actions: Mapping[pd.Timestamp, float],
    day: pd.Timestamp,
) -> pd.DataFrame:
    if day not in frame.index:
        raise T0DataIncompatible("D0_PRICE_MISSING")
    observed = frame.loc[
        (frame.index >= SOURCE_FEATURE_HISTORY_START) & (frame.index <= day),
        list(_RAW_COLUMNS),
    ].copy()
    if len(observed) <= MAX_HISTORY_DAYS:
        raise T0DataIncompatible("FEATURE_HISTORY_UNAVAILABLE")
    factors = np.ones(len(observed), dtype=float)
    for action_day, ratio in split_actions.items():
        if action_day <= day:
            factors *= np.where(observed.index < action_day, ratio, 1.0)
    result = pd.DataFrame(index=observed.index)
    for column in ("Open", "High", "Low", "Close"):
        result[column] = observed[column].to_numpy(dtype=float) / factors
    # V9's causal-normalization section applies the same split factor to volume.
    result["Volume"] = observed["Volume"].to_numpy(dtype=float) * factors
    return result


def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    previous = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = true_range.rolling(14, min_periods=14).mean().iloc[-1]
    if not np.isfinite(value):
        raise T0DataIncompatible("FEATURE_ATR_UNAVAILABLE")
    return float(value)


def feature_values(
    frame: pd.DataFrame,
    day: object,
    split_actions: Mapping[pd.Timestamp, float] | None = None,
) -> dict[str, float]:
    normalized = _normalise_frame(frame)
    observed_day = _as_day(day)
    actions = _normalise_actions(normalized, split_actions)
    if int(((normalized.index >= SOURCE_FEATURE_HISTORY_START) & (normalized.index < observed_day)).sum()) < MAX_HISTORY_DAYS:
        raise T0DataIncompatible("FEATURE_HISTORY_UNAVAILABLE")
    causal = _causal_normalized_frame(normalized, actions, observed_day)
    close = causal["Close"]
    high = causal["High"]
    low = causal["Low"]
    volume = causal["Volume"]
    returns = close.pct_change()

    def trailing_return(period: int) -> float:
        value = close.iloc[-1] / close.iloc[-1 - period] - 1.0
        return float(value)

    values = {
        "return_1d": trailing_return(1),
        "return_5d": trailing_return(5),
        "return_20d": trailing_return(20),
        "return_60d": trailing_return(60),
        "volatility_20": float(returns.tail(20).std(ddof=1)),
        "atr14_percent": _atr14(high, low, close) / float(close.iloc[-1]),
        "close_to_ma20": float(close.iloc[-1] / close.tail(20).mean() - 1.0),
        "close_to_ma60": float(close.iloc[-1] / close.tail(60).mean() - 1.0),
        "distance_from_high20": float(close.iloc[-1] / close.tail(20).max() - 1.0),
        "volume_dryup": float(1.0 - volume.tail(5).mean() / volume.tail(20).mean()),
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=float)).all():
        raise T0DataIncompatible("NONFINITE_FEATURE")
    return values


def d1_d3_target(
    frame: pd.DataFrame,
    d0: object,
    calendar_dates: Sequence[object],
    split_actions: Mapping[pd.Timestamp, float] | None = None,
) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    normalized = _normalise_frame(frame)
    calendar = normalize_calendar(calendar_dates)
    day = _as_day(d0)
    positions = np.flatnonzero(calendar == day)
    if len(positions) != 1:
        raise T0DataIncompatible("D0_NOT_IN_CALENDAR")
    position = int(positions[0])
    if position + 3 >= len(calendar):
        raise T0DataIncompatible("D1_D3_CALENDAR_TAIL_MISSING")
    d1, d3 = calendar[position + 1], calendar[position + 3]
    if d1 not in normalized.index or d3 not in normalized.index:
        raise T0DataIncompatible("TARGET_PRICE_MISSING")
    actions = _normalise_actions(normalized, split_actions)
    hold_ratio = 1.0
    for action_day, ratio in actions.items():
        if d1 < action_day <= d3:
            hold_ratio *= ratio
    d1_close = float(normalized.at[d1, "Close"])
    d3_close = float(normalized.at[d3, "Close"])
    target = d3_close * hold_ratio / d1_close - 1.0
    if not math.isfinite(target):
        raise T0DataIncompatible("NONFINITE_TARGET")
    return float(target), d1, d3


def read_price(
    frame: pd.DataFrame,
    requested_date: object,
    field: str,
    engine_day: object,
) -> float:
    """Read one value under the V9 causal future-read boundary."""
    requested = _as_day(requested_date)
    engine = _as_day(engine_day)
    if requested > engine:
        raise T0ImplementationFailure("FUTURE_PRICE_ACCESS_PROHIBITED")
    normalized = _normalise_frame(frame)
    if field not in _RAW_COLUMNS + ("AdjClose",):
        raise T0ImplementationFailure("PRICE_FIELD_INVALID")
    if requested not in normalized.index:
        raise T0DataIncompatible("REQUESTED_PRICE_MISSING")
    value = float(normalized.at[requested, field])
    if not math.isfinite(value):
        raise T0ImplementationFailure("NONFINITE_PRICE")
    return value


def _validate_universe(universe: pd.DataFrame) -> list[str]:
    if not isinstance(universe, pd.DataFrame) or "ticker" not in universe.columns:
        raise T0DataIncompatible("UNIVERSE_SCHEMA_INVALID")
    codes = [canonical_code(value) for value in universe["ticker"].tolist()]
    if len(codes) != len(set(codes)):
        raise T0DataIncompatible("DUPLICATE_UNIVERSE_CANONICAL_CODE")
    return codes


def build_dataset(
    frames: Mapping[object, pd.DataFrame],
    universe: pd.DataFrame,
    calendar_dates: Sequence[object],
    actions: Mapping[object, Mapping[object, object] | Sequence[Mapping[str, object]]] | None = None,
) -> pd.DataFrame:
    """Build exact causal V9 rows for training and formal development signals."""
    codes = _validate_universe(universe)
    normalized, normalized_actions = normalize_inputs(frames, actions)
    unknown = set(normalized) - set(codes)
    if unknown:
        raise T0DataIncompatible("PRICE_CODE_NOT_IN_UNIVERSE")
    calendar = normalize_calendar(calendar_dates)
    grid = signal_grid(calendar)
    rows: list[dict[str, Any]] = []
    for day in grid:
        for code in codes:
            if code not in normalized or day not in normalized[code].index:
                continue
            frame = normalized[code]
            try:
                feature = feature_values(frame, day, normalized_actions[code])
                target, d1, d3 = d1_d3_target(frame, day, calendar, normalized_actions[code])
            except T0DataIncompatible as error:
                # Missing history, listing, or D1/D3 prices makes this code
                # ineligible for that D0; it is never imputed or substituted.
                if error.reason in {
                    "FEATURE_HISTORY_UNAVAILABLE",
                    "D0_PRICE_MISSING",
                    "TARGET_PRICE_MISSING",
                }:
                    continue
                raise
            row = {"d0": day, "canonical_code": code, **feature}
            row.update(
                {
                    "target_raw_return": target,
                    "target_exit_date": d3,
                    "target_entry_date": d1,
                    "year": int(day.year),
                }
            )
            source_rows = frame.loc[
                (frame.index >= SOURCE_FEATURE_HISTORY_START) & (frame.index <= day)
            ]
            turnover = source_rows["Close"].tail(60) * source_rows["Volume"].tail(60)
            if len(turnover) < 60 or not math.isfinite(float(turnover.median())) or float(turnover.median()) < MIN_MEDIAN_TURNOVER:
                continue
            rows.append(row)
    if not rows:
        raise T0DataIncompatible("NO_EXACT_V9_ROWS")
    result = pd.DataFrame(rows).sort_values(["d0", "canonical_code"], kind="mergesort").reset_index(drop=True)
    result["target_percentile"] = result.groupby("d0", sort=False)["target_raw_return"].transform(
        lambda values: values.rank(method="average", pct=True)
    )
    required = ["d0", "canonical_code", *FEATURE_COLUMNS, "target_raw_return", "target_percentile", "target_exit_date", "year"]
    if not np.isfinite(result.loc[:, [*FEATURE_COLUMNS, "target_raw_return", "target_percentile"]].to_numpy(dtype=float)).all():
        raise T0ImplementationFailure("NONFINITE_DATASET_VALUE")
    return result.loc[:, required]


def month_start(calendar_dates: Sequence[object], year: int, month: int) -> pd.Timestamp:
    calendar = normalize_calendar(calendar_dates)
    matches = calendar[(calendar.year == int(year)) & (calendar.month == int(month))]
    if len(matches) == 0:
        raise T0DataIncompatible("MONTH_START_MISSING")
    return matches[0]


def causal_training_rows(
    dataset: pd.DataFrame,
    calendar_dates: Sequence[object],
    year: int,
    month: int,
) -> pd.DataFrame:
    """Return only rows whose realized target closed before prediction month."""
    result = _validate_dataset(dataset)
    cutoff = month_start(calendar_dates, year, month)
    training = result[
        (result["d0"] >= PRE_EVALUATION_TRAINING_START)
        & (result["target_exit_date"] < cutoff)
    ].copy()
    if (training["target_exit_date"] >= cutoff).any():
        raise T0ImplementationFailure("FUTURE_LABEL_IN_TRAINING")
    return training.reset_index(drop=True)


def _validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"d0", "canonical_code", *FEATURE_COLUMNS, "target_percentile", "target_exit_date", "year"}
    if not isinstance(dataset, pd.DataFrame) or not required.issubset(dataset.columns):
        raise T0DataIncompatible("DATASET_SCHEMA_INVALID")
    result = dataset.copy()
    result["d0"] = pd.to_datetime(result["d0"], errors="coerce")
    result["target_exit_date"] = pd.to_datetime(result["target_exit_date"], errors="coerce")
    if result[["d0", "target_exit_date"]].isna().any().any():
        raise T0DataIncompatible("DATASET_DATE_INVALID")
    try:
        result["canonical_code"] = result["canonical_code"].map(canonical_code)
    except T0DataIncompatible:
        raise
    numeric = result.loc[:, [*FEATURE_COLUMNS, "target_percentile"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise T0DataIncompatible("NONFINITE_DATASET_VALUE")
    if result.groupby("d0", sort=False)["canonical_code"].nunique().ne(result.groupby("d0", sort=False).size()).any():
        raise T0DataIncompatible("DUPLICATE_D0_CANONICAL_CODE")
    if (result["d0"] < PRE_EVALUATION_TRAINING_START).any():
        raise T0DataIncompatible("D0_BEFORE_TRAINING_START")
    if (result["target_exit_date"] <= result["d0"]).any():
        raise T0DataIncompatible("TARGET_CHRONOLOGY_INVALID")
    result.loc[:, [*FEATURE_COLUMNS, "target_percentile"]] = numeric
    return result.sort_values(["d0", "canonical_code"], kind="mergesort").reset_index(drop=True)


def _fit_fixed_models(train: pd.DataFrame) -> tuple[tuple[StandardScaler, Ridge], LGBMRegressor]:
    if train.empty:
        raise T0DataIncompatible("INSUFFICIENT_CAUSAL_TRAINING_DATA")
    x = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["target_percentile"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise T0DataIncompatible("NONFINITE_TRAINING_VALUE")
    scaler = StandardScaler()
    ridge = Ridge(**RIDGE_PARAMS)
    ridge.fit(scaler.fit_transform(x), y)
    lightgbm = LGBMRegressor(**LIGHTGBM_PARAMS)
    lightgbm.fit(x, y)
    return (scaler, ridge), lightgbm


def score_formal_dataset(dataset: pd.DataFrame, calendar_dates: Sequence[object]) -> pd.DataFrame:
    """Fit monthly expanding models using only causally closed target rows."""
    result = _validate_dataset(dataset)
    calendar = normalize_calendar(calendar_dates)
    formal = result[result["d0"].between(FORMAL_SIGNAL_START, FORMAL_SIGNAL_END)].copy()
    if formal.empty or set(formal["d0"].dt.year.unique()) != set(FORMAL_YEARS):
        raise T0DataIncompatible("FORMAL_SIGNAL_YEARS_INCOMPLETE")
    scored_parts: list[pd.DataFrame] = []
    months = sorted({(int(day.year), int(day.month)) for day in formal["d0"]})
    for year, month in months:
        cutoff = month_start(calendar, year, month)
        train = causal_training_rows(result, calendar, year, month)
        (scaler, ridge), lightgbm = _fit_fixed_models(train)
        test = formal[(formal["d0"].dt.year == year) & (formal["d0"].dt.month == month)].copy()
        x = test.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        test["ridge_score"] = ridge.predict(scaler.transform(x))
        test["lightgbm_score"] = lightgbm.predict(x)
        if not np.isfinite(test[["ridge_score", "lightgbm_score"]].to_numpy(dtype=float)).all():
            raise T0ImplementationFailure("NONFINITE_MODEL_SCORE")
        scored_parts.append(test)
    return pd.concat(scored_parts, ignore_index=True).sort_values(["d0", "canonical_code"], kind="mergesort").reset_index(drop=True)


def _validate_scored_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"d0", "canonical_code", "target_percentile", "ridge_score", "lightgbm_score"}
    if not isinstance(rows, pd.DataFrame) or not required.issubset(rows.columns):
        raise T0DataIncompatible("SCORED_ROWS_SCHEMA_INVALID")
    result = rows.copy()
    result["d0"] = pd.to_datetime(result["d0"], errors="coerce")
    if result["d0"].isna().any():
        raise T0DataIncompatible("SCORED_ROWS_DATE_INVALID")
    result["canonical_code"] = result["canonical_code"].map(canonical_code)
    numeric = result.loc[:, ["target_percentile", "ridge_score", "lightgbm_score"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise T0ImplementationFailure("NONFINITE_MODEL_SCORE_OR_TARGET")
    result.loc[:, ["target_percentile", "ridge_score", "lightgbm_score"]] = numeric
    duplicate = result.groupby("d0", sort=False)["canonical_code"].nunique().ne(result.groupby("d0", sort=False).size())
    if duplicate.any():
        raise T0DataIncompatible("DUPLICATE_D0_CANONICAL_CODE")
    result["year"] = result["d0"].dt.year
    return result.sort_values(["d0", "canonical_code"], kind="mergesort").reset_index(drop=True)


def top1_edge_by_d0(rows: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column not in {"ridge_score", "lightgbm_score"}:
        raise T0ImplementationFailure("MODEL_SCORE_COLUMN_INVALID")
    validated = _validate_scored_rows(rows)
    ordered = validated.sort_values(
        ["d0", score_column, "canonical_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    top = ordered.groupby("d0", sort=True, as_index=False).first()
    edge = top.set_index("d0")["target_percentile"] - 0.5
    edge.name = "rank_top1_edge"
    return edge


def top1_metrics(rows: pd.DataFrame, score_column: str, years: Sequence[int] = KILL_SCREEN_YEARS) -> dict[str, Any]:
    edge = top1_edge_by_d0(rows, score_column)
    requested = tuple(int(year) for year in years)
    yearly = {}
    for year in requested:
        values = edge[edge.index.year == year]
        if values.empty:
            raise T0DataIncompatible("KILL_SCREEN_YEAR_INCOMPLETE")
        yearly[year] = float(values.mean())
    return {
        "aggregate": float(edge[edge.index.year.isin(requested)].mean()),
        "yearly": yearly,
        "positive_years": int(sum(value > 0 for value in yearly.values())),
    }


def select_formal_model(rows: pd.DataFrame) -> str:
    ridge = top1_metrics(rows, "ridge_score", FORMAL_YEARS)
    lightgbm = top1_metrics(rows, "lightgbm_score", FORMAL_YEARS)
    if lightgbm["aggregate"] > ridge["aggregate"] and lightgbm["positive_years"] >= ridge["positive_years"]:
        return "LIGHTGBM"
    return "RIDGE"


def screen_top1(rows: pd.DataFrame) -> str:
    """Return only the V9 binary T0 decision; detailed metrics stay in memory."""
    ridge = top1_metrics(rows, "ridge_score", KILL_SCREEN_YEARS)
    lightgbm = top1_metrics(rows, "lightgbm_score", KILL_SCREEN_YEARS)
    ridge_stop = ridge["aggregate"] <= 0 and ridge["positive_years"] <= 3
    lightgbm_stop = lightgbm["aggregate"] <= 0 and lightgbm["positive_years"] <= 3
    return "STOP" if ridge_stop and lightgbm_stop else "CONTINUE"


def rank_top1_edge(rows: pd.DataFrame, score_column: str) -> pd.Series:
    """Named public alias for the amended primary ranker-only estimand."""
    return top1_edge_by_d0(rows, score_column)


_SAFE_KEYS = {
    "schema_version",
    "task",
    "design_sha",
    "implementation_sha",
    "T0_RESULT",
    "input_provenance",
    "validation",
}
_SAFE_PROVENANCE_KEYS = {
    "universe_mode",
    "universe_csv_sha256",
    "ticker_list_sha256",
    "universe_ticker_count",
    "training_cache_manifest_sha256",
    "evaluation_cache_manifest_sha256",
    "training_payload_count",
    "evaluation_payload_count",
}
_SAFE_VALIDATION_KEYS = {
    "exact_v9_features",
    "exact_v9_target",
    "exact_calendar_grid",
    "causal_training_cutoff",
    "fixed_model_parameters",
    "no_hyperparameter_search",
    "cache_identity",
    "no_future_reads",
    "no_source_network",
    "no_cache_mutation",
    "safe_output_contract",
}


def _safe_hash(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


def _safe_sha40(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX40.fullmatch(value))


def validate_safe_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SAFE_KEYS:
        raise T0ImplementationFailure("SAFE_RESULT_SCHEMA_INVALID")
    result = dict(value)
    if result["schema_version"] != "V9_009_T0_TOP1_KILL_SCREEN_V1" or result["task"] != TASK:
        raise T0ImplementationFailure("SAFE_RESULT_IDENTITY_INVALID")
    if result["design_sha"] != DESIGN_SHA or not _safe_sha40(result["implementation_sha"]):
        raise T0ImplementationFailure("SAFE_RESULT_PROVENANCE_INVALID")
    if result["T0_RESULT"] not in {"STOP", "CONTINUE", "NO_VERDICT_DATA_INCOMPATIBLE"}:
        raise T0ImplementationFailure("SAFE_RESULT_VERDICT_INVALID")
    provenance = result["input_provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != _SAFE_PROVENANCE_KEYS:
        raise T0ImplementationFailure("SAFE_RESULT_PROVENANCE_SCHEMA_INVALID")
    if provenance["universe_mode"] != UNIVERSE_MODE:
        raise T0ImplementationFailure("SAFE_RESULT_UNIVERSE_MODE_INVALID")
    for key in ("universe_csv_sha256", "ticker_list_sha256", "training_cache_manifest_sha256", "evaluation_cache_manifest_sha256"):
        if not _safe_hash(provenance[key]):
            raise T0ImplementationFailure("SAFE_RESULT_HASH_INVALID")
    for key in ("universe_ticker_count", "training_payload_count", "evaluation_payload_count"):
        if isinstance(provenance[key], bool) or not isinstance(provenance[key], int) or provenance[key] < 0:
            raise T0ImplementationFailure("SAFE_RESULT_COUNT_INVALID")
    validation = result["validation"]
    if not isinstance(validation, Mapping) or set(validation) != _SAFE_VALIDATION_KEYS:
        raise T0ImplementationFailure("SAFE_RESULT_VALIDATION_SCHEMA_INVALID")
    if not all(type(item) is bool for item in validation.values()):
        raise T0ImplementationFailure("SAFE_RESULT_VALIDATION_VALUE_INVALID")
    serialized = canonical_json_bytes(result).decode("utf-8")
    if any(token in serialized for token in ("aggregate_edge", "yearly_edge", "positive_top1", "target_percentile", "ridge_score", "lightgbm_score")):
        raise T0ImplementationFailure("SAFE_RESULT_OUTCOME_LEAKAGE")
    if any(char in serialized for char in ("\\",)):
        raise T0ImplementationFailure("SAFE_RESULT_PATH_LEAKAGE")
    return result


def make_safe_result(
    verdict: str,
    implementation_sha: str,
    provenance: Mapping[str, Any],
    *,
    cache_identity: bool = True,
    exact_semantics: bool = True,
) -> dict[str, Any]:
    result = {
        "schema_version": "V9_009_T0_TOP1_KILL_SCREEN_V1",
        "task": TASK,
        "design_sha": DESIGN_SHA,
        "implementation_sha": implementation_sha,
        "T0_RESULT": verdict,
        "input_provenance": dict(provenance),
        "validation": {
            "exact_v9_features": exact_semantics,
            "exact_v9_target": exact_semantics,
            "exact_calendar_grid": exact_semantics,
            "causal_training_cutoff": exact_semantics,
            "fixed_model_parameters": True,
            "no_hyperparameter_search": True,
            "cache_identity": cache_identity,
            "no_future_reads": True,
            "no_source_network": True,
            "no_cache_mutation": True,
            "safe_output_contract": True,
        },
    }
    return validate_safe_result(result)


def synthetic_provenance() -> dict[str, Any]:
    zero = "0" * 64
    return {
        "universe_mode": UNIVERSE_MODE,
        "universe_csv_sha256": zero,
        "ticker_list_sha256": zero,
        "universe_ticker_count": 0,
        "training_cache_manifest_sha256": zero,
        "evaluation_cache_manifest_sha256": zero,
        "training_payload_count": 0,
        "evaluation_payload_count": 0,
    }


def _canonical_manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)


def _read_manifest(
    cache_root: Path,
    expected_sha: str,
    expected_tickers: Sequence[str],
    expected_payload_count: int,
) -> tuple[dict[str, Any], dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]]]:
    try:
        manifest_path = cache_root / "cache_manifest.json"
        manifest_body = manifest_path.read_bytes()
        manifest = json.loads(manifest_body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T0DataIncompatible("CACHE_MANIFEST_INVALID") from error
    if not isinstance(manifest, dict) or sha256_bytes(manifest_body).lower() != expected_sha:
        raise T0DataIncompatible("CACHE_MANIFEST_IDENTITY_MISMATCH")
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True or manifest.get("universe_mode") != UNIVERSE_MODE:
        raise T0DataIncompatible("CACHE_MANIFEST_MODE_INVALID")
    if manifest.get("ticker_count") != 300 or manifest.get("ticker_order") != list(expected_tickers):
        raise T0DataIncompatible("CACHE_MANIFEST_UNIVERSE_MISMATCH")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != expected_payload_count:
        raise T0DataIncompatible("CACHE_PAYLOAD_SET_INVALID")
    seen: set[str] = set()
    parsed: dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]] = {}
    for item in payloads:
        if not isinstance(item, dict) or set(item) < {"ticker", "relative_path", "sha256", "byte_count"}:
            raise T0DataIncompatible("CACHE_PAYLOAD_SCHEMA_INVALID")
        code = canonical_code(item["ticker"])
        relative = item["relative_path"]
        if code not in expected_tickers or code in seen or relative != f"raw/{code}.json" or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise T0DataIncompatible("CACHE_PAYLOAD_PATH_INVALID")
        path = cache_root / relative
        try:
            body = path.read_bytes()
        except OSError as error:
            raise T0DataIncompatible("CACHE_PAYLOAD_MISSING") from error
        if sha256_bytes(body) != item["sha256"] or len(body) != item["byte_count"]:
            raise T0DataIncompatible("CACHE_PAYLOAD_HASH_MISMATCH")
        seen.add(code)
        parsed[code] = _parse_chart_payload(body)
    raw_dir = cache_root / "raw"
    try:
        registered = {path.name for path in raw_dir.glob("*.json")}
    except OSError as error:
        raise T0DataIncompatible("CACHE_DIRECTORY_UNREADABLE") from error
    if registered != {f"{code}.json" for code in seen} or not seen.issubset(set(expected_tickers)):
        raise T0DataIncompatible("CACHE_PAYLOAD_SET_INVALID")
    return manifest, parsed


def _parse_chart_payload(body: bytes) -> tuple[pd.DataFrame, dict[pd.Timestamp, float]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T0DataIncompatible("CACHE_PAYLOAD_JSON_INVALID") from error
    try:
        chart = payload["chart"]
        result = chart["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        arrays = [timestamps, adjusted, *(quote[field] for field in ("open", "high", "low", "close", "volume"))]
    except (KeyError, IndexError, TypeError) as error:
        raise T0DataIncompatible("CACHE_PAYLOAD_SCHEMA_INVALID") from error
    if not arrays or len({len(array) for array in arrays}) != 1:
        raise T0DataIncompatible("CACHE_PAYLOAD_LENGTH_INVALID")
    try:
        index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    except (TypeError, ValueError, OverflowError) as error:
        raise T0DataIncompatible("CACHE_PAYLOAD_DATE_INVALID") from error
    frame = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "AdjClose": adjusted,
            "Volume": quote["volume"],
        },
        index=index,
    )
    normalized = _normalise_frame(frame)
    split_events = result.get("events", {}).get("splits", {}) or {}
    actions: dict[pd.Timestamp, float] = {}
    if not isinstance(split_events, Mapping):
        raise T0DataIncompatible("SPLIT_EVENT_SCHEMA_INVALID")
    for event in split_events.values():
        if not isinstance(event, Mapping) or not {"date", "numerator", "denominator"}.issubset(event):
            raise T0DataIncompatible("SPLIT_EVENT_RATIO_MISSING")
        try:
            day = pd.to_datetime(int(event["date"]), unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
            ratio = float(event["numerator"]) / float(event["denominator"])
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
            raise T0DataIncompatible("SPLIT_EVENT_RATIO_INVALID") from error
        if not math.isfinite(ratio) or ratio <= 0 or day in actions:
            raise T0DataIncompatible("SPLIT_EVENT_RATIO_INVALID")
        actions[day] = ratio
    return normalized, actions


def load_fixed_universe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        universe = pd.read_csv(path, dtype={"ticker": str})
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise T0DataIncompatible("UNIVERSE_READ_INVALID") from error
    if sha256_bytes(canonical) != V4_UNIVERSE_CSV_SHA256 or list(universe.columns) != ["ticker", "market", "industry"] or len(universe) != 300:
        raise T0DataIncompatible("UNIVERSE_IDENTITY_MISMATCH")
    codes = [canonical_code(value) for value in universe["ticker"].tolist()]
    if sha256_bytes(("\n".join(codes) + "\n").encode("utf-8")) != V4_TICKER_LIST_SHA256 or len(set(codes)) != 300:
        raise T0DataIncompatible("UNIVERSE_TICKER_IDENTITY_MISMATCH")
    universe["ticker"] = codes
    return universe


def load_fixed_cache_pair(
    training_cache: str | Path,
    evaluation_cache: str | Path,
    universe_csv: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[pd.Timestamp, float]], dict[str, Any], pd.DataFrame]:
    """Read-only adapter for the two reviewed FIXED_V4_300 cache roots."""
    universe = load_fixed_universe(universe_csv)
    codes = universe["ticker"].tolist()
    roots = (Path(training_cache), Path(evaluation_cache))
    manifests = (
        _read_manifest(roots[0], TRAINING_MANIFEST_SHA256, codes, 283),
        _read_manifest(roots[1], EVALUATION_MANIFEST_SHA256, codes, 300),
    )
    frames: dict[str, pd.DataFrame] = {}
    actions: dict[str, dict[pd.Timestamp, float]] = {}
    for code in codes:
        parts: list[pd.DataFrame] = []
        action_parts: list[dict[pd.Timestamp, float]] = []
        if code in manifests[0][1]:
            train_frame, train_actions = manifests[0][1][code]
            parts.append(train_frame.loc[train_frame.index <= pd.Timestamp("2019-12-31")])
            action_parts.append(train_actions)
        if code not in manifests[1][1]:
            raise T0DataIncompatible("EVALUATION_PAYLOAD_MISSING")
        evaluation_frame, evaluation_actions = manifests[1][1][code]
        evaluation_start = pd.Timestamp("2020-01-01") if code in manifests[0][1] else pd.Timestamp("2019-01-01")
        parts.append(evaluation_frame.loc[evaluation_frame.index >= evaluation_start])
        action_parts.append(evaluation_actions)
        if not parts:
            raise T0DataIncompatible("COMBINED_PRICE_EMPTY")
        combined = pd.concat(parts).sort_index()
        if combined.index.duplicated().any():
            raise T0DataIncompatible("DUPLICATE_COMBINED_PRICE_DATE")
        merged_actions: dict[pd.Timestamp, float] = {}
        for item in action_parts:
            for day, ratio in item.items():
                if day in merged_actions and merged_actions[day] != ratio:
                    raise T0DataIncompatible("DUPLICATE_COMBINED_SPLIT_EVENT")
                merged_actions[day] = ratio
        frames[code] = combined
        actions[code] = merged_actions
    provenance = {
        "universe_mode": UNIVERSE_MODE,
        "universe_csv_sha256": V4_UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": V4_TICKER_LIST_SHA256,
        "universe_ticker_count": len(codes),
        "training_cache_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "evaluation_cache_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "training_payload_count": len(manifests[0][1]),
        "evaluation_payload_count": len(manifests[1][1]),
    }
    return frames, actions, provenance, universe


def run_from_cache(
    training_cache: str | Path,
    evaluation_cache: str | Path,
    universe_csv: str | Path,
    calendar_dates: Sequence[object],
    implementation_sha: str,
) -> dict[str, Any]:
    frames, actions, provenance, universe = load_fixed_cache_pair(training_cache, evaluation_cache, universe_csv)
    dataset = build_dataset(frames, universe, calendar_dates, actions)
    scored = score_formal_dataset(dataset, calendar_dates)
    verdict = screen_top1(scored)
    return make_safe_result(verdict, implementation_sha, provenance)


def run_synthetic(
    dataset: pd.DataFrame,
    calendar_dates: Sequence[object],
    implementation_sha: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scored = score_formal_dataset(dataset, calendar_dates)
    verdict = screen_top1(scored)
    return make_safe_result(verdict, implementation_sha, provenance or synthetic_provenance(), cache_identity=False)


__all__ = [
    "DESIGN_SHA",
    "EVAL_YEARS",
    "FEATURE_COLUMNS",
    "FORMAL_YEARS",
    "KILL_SCREEN_YEARS",
    "LIGHTGBM_PARAMS",
    "RIDGE_PARAMS",
    "TASK",
    "T0DataIncompatible",
    "T0ImplementationFailure",
    "build_dataset",
    "canonical_json_bytes",
    "canonical_code",
    "causal_training_rows",
    "d1_d3_target",
    "feature_fingerprint",
    "feature_values",
    "load_fixed_cache_pair",
    "load_fixed_universe",
    "make_safe_result",
    "model_parameter_fingerprint",
    "month_start",
    "normalize_calendar",
    "normalize_inputs",
    "run_from_cache",
    "run_synthetic",
    "read_price",
    "rank_top1_edge",
    "screen_top1",
    "score_formal_dataset",
    "select_formal_model",
    "sha256_bytes",
    "signal_grid",
    "top1_edge_by_d0",
    "top1_metrics",
    "validate_safe_result",
]
