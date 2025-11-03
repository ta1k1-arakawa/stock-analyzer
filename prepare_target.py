import pandas as pd
import logging

# 既存のロガー名を指定
logger = logging.getLogger('stock_analyzer_app')

def create_target_variable(df: pd.DataFrame, future_days: int, target_percent: float) -> pd.DataFrame:
    """
    AIに予測させる「答え（目的変数）」を作成する。

    Args:
        df (pd.DataFrame): テクニカル指標が計算済みのDataFrame
        future_days (int): 予測する未来の日数 (例: 5)
        target_percent (float): 目標とする上昇率 (例: 2.0)

    Returns:
        pd.DataFrame: 'Target' 列 (1=上昇, 0=それ以外) が追加されたDataFrame
    """
    if df.empty:
        logger.warning("Target作成: 入力DataFrameが空です。")
        return df

    logger.info(f"AIの「答え」を作成します (基準: {future_days}営業日後に{target_percent}%以上上昇)")

    # N日後の終値
    future_price = df['Close'].shift(-future_days)
    
    # 目標とする価格 (例: 2%UPなら 1.02倍)
    target_price = df['Close'] * (1 + target_percent / 100.0)
    
    # AIの答え (1 = 目標達成, 0 = 目標未達)
    df['Target'] = (future_price > target_price).astype(int)
    
    # 'Target' が計算できない未来の行 (NaN) を削除
    df_cleaned = df.dropna(subset=['Target'])
    
    logger.info(f"Target作成完了。計算不能な最新{future_days}日分を除外しました。")
    
    return df_cleaned