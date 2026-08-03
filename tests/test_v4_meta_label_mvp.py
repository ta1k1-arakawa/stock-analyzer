from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v4_meta_label_mvp import (
    FEATURE_COLUMNS, add_execution_labels, build_feature_frame, prepare_price_frame,
    select_daily_candidates, validate_v4_ohlcv,
)


def raw_frame(n: int = 270, close: float = 1000.0) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    values = close + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": values, "High": values + 2, "Low": values - 2,
                         "Close": values, "Adj Close": values, "Volume": 100_000.0}, index=dates)


def universe(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "industry": ["x"] * len(tickers), "market": ["m"] * len(tickers)})


def labelled_row(raw: pd.DataFrame, ticker: str = "1111", signal_pos: int = 0) -> pd.DataFrame:
    return pd.DataFrame({"ticker": [ticker], "signal_date": [raw.index[signal_pos]], "eligible": [True],
                         "return_20d": [.1], **{name: [0.0] for name in FEATURE_COLUMNS if name != "return_20d"}})


def test_no_future_information_is_used():
    raw = raw_frame(400)
    before = build_feature_frame({"1111": raw}, universe("1111"))
    changed = raw.copy(); changed.loc[changed.index[-1], ["Open", "High", "Low", "Close", "Adj Close"]] = 999_999
    after = build_feature_frame({"1111": changed}, universe("1111"))
    date = before.iloc[0].signal_date
    assert before.loc[before.signal_date == date, list(FEATURE_COLUMNS)].equals(after.loc[after.signal_date == date, list(FEATURE_COLUMNS)])


def test_fifteen_feature_names_and_order():
    assert FEATURE_COLUMNS == ("return_5d", "return_20d", "return_60d", "volatility_20d", "volume_ratio_5d_20d", "close_to_ma20", "close_to_ma60", "high_low_range_20d", "required_cash_ratio", "momentum_20d_percentile_rank", "relative_momentum_20d", "cross_section_median_return_20d", "cross_section_breadth_above_ma20", "cross_section_median_volatility_20d", "cross_section_eligible_count")


def test_return_and_volatility_formulas():
    raw = raw_frame(400); raw["Adj Close"] = 100 * (1.01 ** np.arange(len(raw)))
    raw[["Open", "High", "Low", "Close"]] = np.repeat(raw["Adj Close"].to_numpy()[:, None], 4, axis=1)
    row = build_feature_frame({"1111": raw}, universe("1111")).iloc[-1]
    assert row.return_20d == pytest.approx(1.01 ** 20 - 1)
    assert row.volatility_20d == pytest.approx(0.0, abs=1e-12)


def test_252_day_history_requirement():
    raw = raw_frame(260); raw.index = pd.date_range("2016-01-01", periods=260, freq="B")
    features = build_feature_frame({"1111": raw}, universe("1111"))
    assert features.loc[features.History_Count < 252, "eligible"].eq(False).all()
    assert features.loc[features.History_Count >= 252, "eligible"].any()


def test_20_day_momentum_selects_first_ranked():
    date = pd.Timestamp("2016-04-01")
    data = pd.DataFrame({"signal_date": [date, date], "ticker": ["1111", "2222"], "eligible": [True, True], "return_20d": [.2, .1], "EntryPrice": [100, 100]})
    assert select_daily_candidates(data).iloc[0].ticker == "1111"


def test_momentum_tie_uses_ticker_ascending():
    date = pd.Timestamp("2016-04-01")
    data = pd.DataFrame({"signal_date": [date, date], "ticker": ["2222", "1111"], "eligible": [True, True], "return_20d": [.2, .2], "EntryPrice": [100, 100]})
    assert select_daily_candidates(data).iloc[0].ticker == "1111"


def test_unaffordable_winner_is_not_replaced_by_second_rank():
    date = pd.Timestamp("2016-04-01")
    data = pd.DataFrame({"signal_date": [date, date], "ticker": ["1111", "2222"], "eligible": [True, True], "return_20d": [.2, .1], "EntryPrice": [4000, 100]})
    result = select_daily_candidates(data).iloc[0]
    assert result.ticker == "1111" and result.purchase_status == "INSUFFICIENT_CASH"


def test_stop_label():
    raw = raw_frame(3, 100); raw.iloc[1] = [100, 101, 94, 96, 96, 100_000]
    result = add_execution_labels(labelled_row(raw), {"1111": raw}, {"1111": set()})
    assert result.iloc[0].ExitReason == "STOP"


def test_gap_stop_label():
    raw = raw_frame(3, 100); raw.iloc[1] = [100, 101, 99, 100, 100, 100_000]; raw.iloc[2] = [90, 91, 89, 90, 90, 100_000]
    result = add_execution_labels(labelled_row(raw), {"1111": raw}, {"1111": set()})
    assert result.iloc[0].ExitReason == "GAP_STOP"


def test_time_label():
    raw = raw_frame(3, 100); raw.iloc[1] = [100, 101, 99, 100, 100, 100_000]; raw.iloc[2] = [101, 103, 100, 102, 102, 100_000]
    result = add_execution_labels(labelled_row(raw), {"1111": raw}, {"1111": set()})
    assert result.iloc[0].ExitReason == "TIME"


def test_split_spanning_candidate_is_excluded():
    raw = raw_frame(3, 100)
    result = add_execution_labels(labelled_row(raw), {"1111": raw}, {"1111": {raw.index[1]}})
    assert not result.iloc[0].eligible and pd.isna(result.iloc[0].label)


def test_exit_date_confirms_label():
    raw = raw_frame(3, 100); raw.iloc[2] = [101, 103, 100, 102, 102, 100_000]
    result = add_execution_labels(labelled_row(raw), {"1111": raw}, {"1111": set()}).iloc[0]
    assert result.LabelConfirmedDate == result.ExitDate == raw.index[2] and result.label == 1


def test_price_adapter_exposes_adjusted_ohlc_and_rejects_2020():
    price = raw_frame(2)
    price["Adj Close"] = price["Close"] * .5
    result = prepare_price_frame(price)
    assert result.adjustment_factor.iloc[0] == .5 and result.adjusted_open.iloc[0] == 500
    price.index = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    with pytest.raises(ValueError, match="PROHIBITED_V4"):
        validate_v4_ohlcv(price)
