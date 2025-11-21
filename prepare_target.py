import pandas as pd
import logging

logger = logging.getLogger('stock_analyzer_app')

def create_target_variable(df: pd.DataFrame, future_days: int, target_percent: float) -> pd.DataFrame:
    """
    AIの正解データを作成する。
    戦略: 「翌日の始値(Open)」で買い、「N日後の終値(Close)」で売る。
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. エントリー価格 = 翌日の始値 (Shift -1)
    # 今日のデータを見て「明日買おう」と判断するため
    buy_price = df['Open'].shift(-1)
    
    # 2. エグジット価格 = N日後の終値 (Shift -N)
    # future_days=1なら「翌日の終値」＝デイトレード的な動き
    # future_days=3なら「3日後の終値」
    sell_price = df['Close'].shift(-future_days)
    
    # 3. 収益率 (%)
    return_percent = (sell_price - buy_price) / buy_price * 100
    
    # 4. 正解ラベル付け (Target)
    # 目標%を超えたら「買い(1)」、それ以外は「0」
    df['Target'] = 0
    
    # 未来のデータが存在し、かつ利益目標を超えている場所を 1 にする
    valid_idx = return_percent.notna()
    df.loc[valid_idx & (return_percent >= target_percent), 'Target'] = 1
    
    # 学習に使えない行（未来のデータがない行など）は、
    # 呼び出し元の train.py や backtest.py で dropna して削除します。
    # ここでは計算結果をそのまま返します。
    
    return df