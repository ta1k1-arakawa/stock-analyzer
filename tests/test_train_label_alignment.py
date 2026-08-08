from __future__ import annotations

from pathlib import Path

import pandas as pd

import train
from src.config import AIParams, AppConfig


def test_train_passes_all_execution_parameters_to_target_builder(
    monkeypatch, tmp_path: Path,
) -> None:
    """Verify label arguments without fetching data, fitting, or saving a model."""
    ai = AIParams(
        future_days=7,
        target_percent=4.25,
        entry_slippage_percent=0.17,
        exit_slippage_percent=0.23,
        stop_loss_percent=6.75,
        stop_slippage_percent=0.41,
        commission_percent=0.37,
    )
    config = AppConfig(
        stock_code="1570",
        feature_columns=["feature"],
        ai_params=ai,
        tech_params={},
        training_settings={"data_from": "2020-01-01", "data_to": "2025-03-31"},
        model_path=tmp_path / "must-not-be-created.pkl",
    )
    prices = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0],
         "Close": [100.0], "Volume": [1]},
        index=pd.to_datetime(["2025-03-31"]),
    )

    class StubFetcher:
        def get_daily_stock_prices(self, code: str, date_from: str, date_to: str):
            assert (code, date_from, date_to) == (
                "1570", "2020-01-01", "2025-03-31"
            )
            return prices

    captured: dict[str, object] = {}

    def stub_target_builder(frame: pd.DataFrame, **kwargs):
        captured["frame"] = frame
        captured["kwargs"] = kwargs
        # Stop before the LightGBM construction path without raising.
        return pd.DataFrame(columns=["feature", "Target"])

    monkeypatch.setattr(train, "YFinanceFetcher", StubFetcher)
    monkeypatch.setattr(train, "sanitize_ohlcv", lambda frame: frame)
    monkeypatch.setattr(train, "calculate_indicators", lambda frame, _params: frame)
    monkeypatch.setattr(train, "create_target_variable", stub_target_builder)
    monkeypatch.setattr(
        train.lgb, "LGBMClassifier",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("model training must not be reached")
        ),
    )
    monkeypatch.setattr(
        train.joblib, "dump",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model saving must not be reached")
        ),
    )

    assert train.train_ai_model(config) is False
    assert captured["frame"] is prices
    assert captured["kwargs"] == {
        "future_days": 7,
        "target_percent": 4.25,
        "entry_slippage_percent": 0.17,
        "exit_slippage_percent": 0.23,
        "stop_loss_percent": 6.75,
        "stop_slippage_percent": 0.41,
        "commission_percent": 0.37,
    }
    assert not config.model_path.exists()
