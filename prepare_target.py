import pandas as pd
import logging

logger = logging.getLogger('stock_analyzer_app')

def create_target_variable(df: pd.DataFrame, future_days: int, target_percent: float) -> pd.DataFrame:
    """
    AIに予測させる「答え（目的変数）」を作成する。
    3クラス分類: 買い(1) / 売り(-1) / 何もしない(0)

    Args:
        df (pd.DataFrame): テクニカル指標が計算済みのDataFrame
        future_days (int): 予測する未来の日数 (例: 5)
        target_percent (float): 目標とする変動率（取引コスト考慮後の最小利益スレッショルド、例: 2.0）

    Returns:
        pd.DataFrame: 'Target' 列 (1=買い, -1=売り, 0=何もしない) が追加されたDataFrame
    """
    if df.empty:
        logger.warning("Target作成: 入力DataFrameが空です。")
        return df

    logger.info(f"AIの「答え」を3クラス分類で作成します (基準: {future_days}営業日後に±{target_percent}%)")

    # N日後の終値
    future_price = df['Close'].shift(-future_days)
    
    # 上昇目標価格 (例: 2%UP)
    buy_target_price = df['Close'] * (1 + target_percent / 100.0)
    
    # 下落目標価格 (例: 2%DOWN)
    sell_target_price = df['Close'] * (1 - target_percent / 100.0)
    
    # 3クラスの判定
    # 買い (1): N日後に目標上昇率を超えた
    # 売り (-1): N日後に目標下落率を下回った
    # 何もしない (0): その他
    df['Target'] = 0  # デフォルトは「何もしない」
    df.loc[future_price > buy_target_price, 'Target'] = 1   # 買い
    df.loc[future_price < sell_target_price, 'Target'] = -1  # 売り
    
    # Target列をint型に明示的に変換
    df['Target'] = df['Target'].astype(int)
    
    # 'Target' が計算できない未来の行 (NaN) を削除
    df_cleaned = df.dropna(subset=['Target'])
    
    logger.info(f"Target作成完了。計算不能な最新{future_days}日分を除外しました。")
    logger.info(f"クラス分布 - 買い(1): {(df_cleaned['Target']==1).sum()}, "
                f"売り(-1): {(df_cleaned['Target']==-1).sum()}, "
                f"何もしない(0): {(df_cleaned['Target']==0).sum()}")
    
    return df_cleaned