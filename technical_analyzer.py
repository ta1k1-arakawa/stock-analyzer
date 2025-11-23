import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger('stock_analyzer_app')

def calculate_indicators(df_ohlcv: pd.DataFrame, analysis_params: dict) -> pd.DataFrame:
    """
    テクニカル指標を一括計算し、AI学習用に正規化（率への変換）を行う集約モジュール。
    backtest, train, main すべてでこの関数を使用することで整合性を保ちます。
    """
    if df_ohlcv is None or df_ohlcv.empty:
        logger.warning("入力データが空です。")
        return pd.DataFrame()

    df = df_ohlcv.copy()
    
    # テクニカル指標の計算 
    try:
        # SMA (単純移動平均)
        sma_s = analysis_params.get('sma', {}).get('short_period', 5)
        sma_l = analysis_params.get('sma', {}).get('long_period', 25)
        
        df.ta.sma(length=sma_s, close=df['Close'], append=True, col_names=(f'SMA_{sma_s}',))
        df.ta.sma(length=sma_l, close=df['Close'], append=True, col_names=(f'SMA_{sma_l}',))

        # RSI (相対力指数)
        rsi_p = analysis_params.get('rsi', {}).get('period', 14)
        df.ta.rsi(length=rsi_p, close=df['Close'], append=True, col_names=('RSI_14',))

        # MACD (移動平均収束拡散)
        macd_fast = analysis_params.get('macd', {}).get('fast_period', 12)
        macd_slow = analysis_params.get('macd', {}).get('slow_period', 26)
        macd_signal = analysis_params.get('macd', {}).get('signal_period', 9)
        
        df.ta.macd(close=df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        # pandas_taの生成列名: MACD_12_26_9, MACDh_..., MACDs_... などを想定
        
        # ボリンジャーバンド (2σ)
        df.ta.bbands(close=df['Close'], length=20, std=2, append=True)

        # ATR (Average True Range: 真の値幅平均)
        df.ta.atr(length=14, append=True, col_names=('ATR_14',))

        # ADX (Average Directional Index)
        df.ta.adx(length=14, append=True)
        # 生成される列: ADX_14, DMP_14, DMN_14

    except Exception as e:
        logger.error(f"テクニカル指標の計算中にエラーが発生しました: {e}")
        return df

    # 特徴量の正規化 
    # SMA乖離率 (%)
    if f'SMA_{sma_s}' in df.columns:
        df[f'SMA_{sma_s}_Rate'] = (df['Close'] - df[f'SMA_{sma_s}']) / df[f'SMA_{sma_s}'] * 100
        
    if f'SMA_{sma_l}' in df.columns:
        df[f'SMA_{sma_l}_Rate'] = (df['Close'] - df[f'SMA_{sma_l}']) / df[f'SMA_{sma_l}'] * 100

    # MACDの正規化 (株価に対する比率に変換)
    macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
    if macd_col in df.columns:
         df['MACD_Rate'] = df[macd_col] / df['Close'] * 100
    elif 'MACD_12_26_9' in df.columns:
         df['MACD_Rate'] = df['MACD_12_26_9'] / df['Close'] * 100

    # ボリンジャーバンドの位置 
    bbp_cols = [c for c in df.columns if c.startswith('BBP_20')]
    if bbp_cols:
        # 見つかった最初のBBPカラムを使用
        df['BB_Position'] = df[bbp_cols[0]]
    else:
        # 万が一見つからない場合は計算して補完 (フェイルセーフ)
        try:
            temp_bb = df.ta.bbands(close=df['Close'], length=20, std=2, append=False)
            target_col = [c for c in temp_bb.columns if c.startswith('BBP')][0]
            df['BB_Position'] = temp_bb[target_col]
        except:
            pass # ここでもダメなら欠損とする

    # ATR比率 (%)
    if 'ATR_14' in df.columns:
        df['ATR_Rate'] = df['ATR_14'] / df['Close'] * 100

    # 価格変化率 (Return)
    df['Change_Rate_1'] = df['Close'].pct_change(1) * 100
    df['Change_Rate_3'] = df['Close'].pct_change(3) * 100
    df['Change_Rate_5'] = df['Close'].pct_change(5) * 100
    
    # 出来高変化率
    df['Volume_Change_1'] = df['Volume'].pct_change(1) * 100

    return df