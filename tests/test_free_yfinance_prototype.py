from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.free_prototype import (
    DATE_TO, EVALUATION_METADATA, FEATURE_COLUMNS, NetworkAudit,
    add_execution_labels, assert_allowed_url, deterministic_random_score,
    ensure_report_has_no_raw_prices, parse_current_jpx_universe,
    select_codes, selected_codes_hash, simulate_ranked_portfolio,
    training_rows_for_fold, validate_ohlcv, write_deterministic_json,
)


def ohlcv(rows: list[tuple]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])


def test_metadata_declares_current_only_research():
    assert EVALUATION_METADATA == {
        "evaluation_type": "SURVIVORSHIP_BIASED_RESEARCH_ONLY",
        "formal_backtest": False,
        "point_in_time_universe": False,
        "deployment_decision_allowed": False,
        "shadow_replacement_allowed": False,
        "reference_period_used": False,
    }


def test_code_selection_is_deterministic_and_capped():
    codes = [f"{i:04d}" for i in range(500, 0, -1)]
    first = select_codes(codes)
    second = select_codes(list(reversed(codes)))
    assert first == second
    assert len(first) == 300
    assert selected_codes_hash(first) == selected_codes_hash(second)


def test_current_universe_filter_is_mechanical():
    frame = pd.DataFrame({
        "コード": ["1111", "2222", "3333", "44444", "5555"],
        "銘柄名": ["a", "b", "c", "d", "e"],
        "市場・商品区分": ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）", "プライム（内国株式）", "プライム（外国株式）"],
        "33業種区分": ["A"] * 5,
    })
    eligible, counts = parse_current_jpx_universe(frame)
    assert eligible["code"].tolist() == ["1111", "2222"]
    assert counts["eligible_current_only"] == 2


def test_only_allowlisted_https_hosts_are_allowed():
    assert assert_allowed_url("https://query1.finance.yahoo.com/v8/finance/chart/1.T") == "query1.finance.yahoo.com"
    with pytest.raises(RuntimeError, match="PROHIBITED_NETWORK_DESTINATION"):
        assert_allowed_url("https://example.com/data")
    with pytest.raises(RuntimeError, match="PROHIBITED_NETWORK_DESTINATION"):
        assert_allowed_url("http://www.jpx.co.jp/data.xls")


def test_network_audit_blocks_redirect(monkeypatch):
    class Response:
        status_code = 302
        headers = {"Location": "https://example.com/file"}
        def raise_for_status(self): pass
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="REDIRECT_BLOCKED"):
        NetworkAudit().get("https://www.jpx.co.jp/file")


def test_post_cutoff_ohlcv_is_rejected():
    frame = ohlcv([(100, 101, 99, 100, 100, 1000)])
    frame.index = pd.DatetimeIndex([pd.Timestamp(DATE_TO) + pd.Timedelta(days=1)])
    with pytest.raises(ValueError, match="PROHIBITED_POST_CUTOFF_DATA"):
        validate_ohlcv(frame)


def test_duplicate_and_invalid_ohlcv_are_rejected():
    frame = ohlcv([(100, 101, 99, 100, 100, 1000), (100, 101, 99, 100, 100, 1000)])
    frame.index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(frame)


def test_realized_return_label_uses_shared_execution():
    raw = ohlcv([
        (100, 101, 99, 100, 100, 1000),
        (100, 102, 99, 101, 101, 1000),
        (101, 103, 100, 102, 102, 1000),
    ])
    features = pd.DataFrame(index=raw.index)
    labelled, excluded = add_execution_labels(raw, features, set())
    assert excluded == 0
    expected_entry = 100 * 1.0003
    expected_exit = 102 * 0.9997
    expected = (expected_exit - expected_entry) / expected_entry * 100
    assert labelled.iloc[0]["realized_net_return_percent"] == pytest.approx(expected)
    assert labelled.iloc[0]["ExitReason"] == "TIME"


def test_split_during_holding_is_fail_closed():
    raw = ohlcv([(100, 101, 99, 100, 100, 1000)] * 3)
    labelled, excluded = add_execution_labels(raw, pd.DataFrame(index=raw.index), {raw.index[1]})
    assert excluded >= 1
    assert pd.isna(labelled.iloc[0]["realized_net_return_percent"])


def test_training_rows_require_confirmed_labels_and_embargo():
    dates = pd.date_range("2020-12-20", periods=15, freq="D")
    frame = pd.DataFrame({"signal_date": dates, "LabelConfirmedDate": dates + pd.Timedelta(days=2)})
    fold = {"train_from": "2020-01-01", "train_to": "2020-12-31", "validation_from": "2021-01-01"}
    selected = training_rows_for_fold(frame, fold)
    assert (pd.to_datetime(selected["LabelConfirmedDate"]) < pd.Timestamp("2021-01-01")).all()
    assert selected["signal_date"].max() < pd.Timestamp("2020-12-30")


def candidates(scores=(1.0, 0.5), entry_date="2024-01-02", exit_date="2024-01-03") -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "EntryDate": pd.to_datetime([entry_date, entry_date]), "ExitDate": pd.to_datetime([exit_date, exit_date]),
        "code": ["1111", "2222"], "industry": ["A", "B"], "prediction": list(scores),
        "Change_Rate_5": [1.0, 2.0], "EntryPrice": [1000.0, 1100.0], "ExitPrice": [1010.0, 1110.0],
        "ExitReason": ["TIME", "TIME"], "Raw_Close": [1000.0, 1100.0],
    })


def test_non_positive_prediction_means_no_trade():
    result = simulate_ranked_portfolio(candidates(scores=(-0.1, -0.2)), "prediction", positive_gate=True)
    assert result["closed_trades"] == 0
    assert result["no_trade_counts"]["NO_TRADE_NON_POSITIVE_PREDICTION"] == 1


def test_portfolio_uses_exactly_one_lot_and_never_negative_cash():
    result = simulate_ranked_portfolio(candidates(), "prediction", positive_gate=True)
    assert result["closed_trades"] == 1
    assert result["profit"] == pytest.approx(1000.0)
    assert result["negative_cash_count"] == 0
    assert result["capital_reuse_count"] == 0


def test_sale_proceeds_are_not_reused_for_same_day_open():
    first = candidates().iloc[[0]].copy()
    second = candidates(entry_date="2024-01-03", exit_date="2024-01-04").iloc[[1]].copy()
    second["signal_date"] = pd.Timestamp("2024-01-02")
    result = simulate_ranked_portfolio(pd.concat([first, second], ignore_index=True), "prediction", positive_gate=True)
    assert result["closed_trades"] == 1
    assert result["no_trade_counts"]["NO_TRADE_POSITION_OPEN"] == 1


def test_random_baseline_is_fixed_seed_deterministic():
    first = deterministic_random_score(10000, "2024-01-01", "1111")
    second = deterministic_random_score(10000, "2024-01-01", "1111")
    assert first == second
    assert first != deterministic_random_score(10001, "2024-01-01", "1111")


def test_security_code_is_not_a_model_feature():
    assert "code" not in {column.lower() for column in FEATURE_COLUMNS}
    assert "name" not in {column.lower() for column in FEATURE_COLUMNS}
    assert len(FEATURE_COLUMNS) == 17


def test_report_rejects_raw_price_values(tmp_path):
    ensure_report_has_no_raw_prices({"metrics": {"profit": 1.0}})
    with pytest.raises(AssertionError, match="raw price"):
        ensure_report_has_no_raw_prices({"Open": 100.0})
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_deterministic_json(a, {"z": 1.234567890123, "a": True})
    write_deterministic_json(b, {"a": True, "z": 1.234567890123})
    assert a.read_bytes() == b.read_bytes()


def test_no_real_order_or_jquants_logic_exists():
    source = (Path(__file__).parents[1] / "scripts" / "run_free_yfinance_prototype.py").read_text(encoding="utf-8")
    assert "JQUANTS_API_KEY" not in source
    assert "submit_order" not in source
    assert "broker" not in source.lower()
