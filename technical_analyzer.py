import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger('stock_analyzer_app')

def calculate_indicators(df_ohlcv: pd.DataFrame, analysis_params: dict) -> pd.DataFrame:
    """
    テクニカル指標を計算し、AI学習用に正規化（率への変換）を行います。
    """
    if df_ohlcv is None or df_ohlcv.empty:
        logger.warning("入力データが空です。")
        return pd.DataFrame()

    df = df_ohlcv.copy()
    
    # --- 1. テクニカル指標の計算 (pandas_ta利用) ---
    try:
        # SMA (単純移動平均)
        if 'sma' in analysis_params:
            sma_s = analysis_params['sma']['short_period']
            sma_l = analysis_params['sma']['long_period']
            df.ta.sma(length=sma_s, close=df['Close'], append=True, col_names=(f'SMA_{sma_s}',))
            df.ta.sma(length=sma_l, close=df['Close'], append=True, col_names=(f'SMA_{sma_l}',))

        # RSI
        if 'rsi' in analysis_params:
            rsi_p = analysis_params['rsi']['period']
            df.ta.rsi(length=rsi_p, close=df['Close'], append=True, col_names=('RSI_14',))

        # MACD
        if 'macd' in analysis_params:
            # pandas_taのデフォルト列名は MACD_12_26_9, MACDh_..., MACDs_... となる
            df.ta.macd(close=df['Close'], append=True)

    except Exception as e:
        logger.error(f"指標計算エラー: {e}")
        return df

    # --- 2. 特徴量の正規化 (価格依存を排除し、率に変換) ---
    # これにより、価格帯が違う銘柄でも同じモデルが使えるようになります。

    # SMA乖離率 (%) : (終値 - SMA) / SMA * 100
    if f'SMA_{sma_s}' in df.columns:
        df[f'SMA_{sma_s}_Rate'] = (df['Close'] - df[f'SMA_{sma_s}']) / df[f'SMA_{sma_s}'] * 100
        
    if f'SMA_{sma_l}' in df.columns:
        df[f'SMA_{sma_l}_Rate'] = (df['Close'] - df[f'SMA_{sma_l}']) / df[f'SMA_{sma_l}'] * 100

    # 価格変化率 (Return)
    df['Change_Rate_1'] = df['Close'].pct_change(1) * 100 # 前日比
    df['Change_Rate_3'] = df['Close'].pct_change(3) * 100 # 3日前比
    
    # 出来高変化率
    # 0除算を防ぐため replace で 0 を NaN か 1 にする処理を入れるのが安全ですが簡易的に
    df['Volume_Change_1'] = df['Volume'].pct_change(1) * 100

    # MACDの正規化 (簡易的に株価に対する比率とする)
    if 'MACD_12_26_9' in df.columns:
         df['MACD_Rate'] = df['MACD_12_26_9'] / df['Close'] * 100

    return df