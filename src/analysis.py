"""
テクニカル指標の計算および正解ラベル作成モジュール。

technical_analyzer.py と prepare_target.py を統合。
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401 — DataFrame.ta accessor を有効化

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 列を数値型に変換する（共通の前処理）。"""
    df = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# テクニカル指標計算
# ---------------------------------------------------------------------------

def calculate_indicators(df_ohlcv: pd.DataFrame, analysis_params: dict) -> pd.DataFrame:
    """
    テクニカル指標を一括計算し、AI学習用に正規化（率への変換）を行う集約関数。

    backtest, train, main すべてでこの関数を使用することで整合性を保つ。
    """
    if df_ohlcv is None or df_ohlcv.empty:
        logger.warning("入力データが空です。")
        return pd.DataFrame()

    df = df_ohlcv.copy()

    # テクニカル指標の計算
    try:
        # SMA (単純移動平均)
        sma_s = analysis_params.get("sma", {}).get("short_period", 5)
        sma_l = analysis_params.get("sma", {}).get("long_period", 25)

        df.ta.sma(length=sma_s, close=df["Close"], append=True, col_names=(f"SMA_{sma_s}",))
        df.ta.sma(length=sma_l, close=df["Close"], append=True, col_names=(f"SMA_{sma_l}",))

        # RSI (相対力指数)
        rsi_p = analysis_params.get("rsi", {}).get("period", 14)
        df.ta.rsi(length=rsi_p, close=df["Close"], append=True, col_names=("RSI_14",))

        # MACD (移動平均収束拡散)
        macd_fast = analysis_params.get("macd", {}).get("fast_period", 12)
        macd_slow = analysis_params.get("macd", {}).get("slow_period", 26)
        macd_signal = analysis_params.get("macd", {}).get("signal_period", 9)

        df.ta.macd(close=df["Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)

        # ボリンジャーバンド (2σ)
        df.ta.bbands(close=df["Close"], length=20, std=2, append=True)

        # ATR (Average True Range: 真の値幅平均)
        df.ta.atr(length=14, append=True, col_names=("ATR_14",))

        # ADX (Average Directional Index)
        df.ta.adx(length=14, append=True)

    except Exception as e:
        logger.error("テクニカル指標の計算中にエラーが発生しました: %s", e)
        return df

    # --- 特徴量の正規化 ---

    # SMA 乖離率 (%)
    if f"SMA_{sma_s}" in df.columns:
        df[f"SMA_{sma_s}_Rate"] = (df["Close"] - df[f"SMA_{sma_s}"]) / df[f"SMA_{sma_s}"] * 100

    if f"SMA_{sma_l}" in df.columns:
        df[f"SMA_{sma_l}_Rate"] = (df["Close"] - df[f"SMA_{sma_l}"]) / df[f"SMA_{sma_l}"] * 100

    # MACD の正規化 (株価に対する比率に変換)
    macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    if macd_col in df.columns:
        df["MACD_Rate"] = df[macd_col] / df["Close"] * 100
    elif "MACD_12_26_9" in df.columns:
        df["MACD_Rate"] = df["MACD_12_26_9"] / df["Close"] * 100

    # ボリンジャーバンドの位置
    bbp_cols = [c for c in df.columns if c.startswith("BBP_20")]
    if bbp_cols:
        df["BB_Position"] = df[bbp_cols[0]]
    else:
        # フェイルセーフ
        try:
            temp_bb = df.ta.bbands(close=df["Close"], length=20, std=2, append=False)
            target_col = [c for c in temp_bb.columns if c.startswith("BBP")][0]
            df["BB_Position"] = temp_bb[target_col]
        except Exception:
            pass  # ここでもダメなら欠損とする

    # ATR 比率 (%)
    if "ATR_14" in df.columns:
        df["ATR_Rate"] = df["ATR_14"] / df["Close"] * 100

    # 価格変化率 (Return)
    df["Change_Rate_1"] = df["Close"].pct_change(1) * 100
    df["Change_Rate_3"] = df["Close"].pct_change(3) * 100
    df["Change_Rate_5"] = df["Close"].pct_change(5) * 100

    # 出来高変化率
    df["Volume_Change_1"] = df["Volume"].pct_change(1) * 100

    return df


# ---------------------------------------------------------------------------
# 正解ラベル作成
# ---------------------------------------------------------------------------

def create_target_variable(
    df: pd.DataFrame,
    future_days: int,
    target_percent: float,
) -> pd.DataFrame:
    """
    AIの正解データを作成する。

    戦略: 「翌日の始値(Open)」で買い、「N日後の終値(Close)」で売る。
    ``future_days`` 日後の収益率が ``target_percent`` 以上なら ``Target=1``（買い）。
    """
    if df.empty:
        return df

    df = df.copy()

    # エントリー価格 = 翌日の始値 (Shift -1)
    buy_price = df["Open"].shift(-1)

    # エグジット価格 = N日後の終値 (Shift -N)
    sell_price = df["Close"].shift(-future_days)

    # 収益率 (%)
    return_percent = (sell_price - buy_price) / buy_price * 100

    # 正解ラベル付け
    df["Target"] = 0
    valid_idx = return_percent.notna()
    df.loc[valid_idx & (return_percent >= target_percent), "Target"] = 1

    return df
