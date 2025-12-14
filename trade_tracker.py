import os
import csv
import pandas as pd
from datetime import datetime

class TradeTracker:
    # ★修正1: 初期化時に budget を受け取って保存するように変更
    def __init__(self, budget, filename='trade_log.csv'):
        self.filename = filename
        self.budget = budget  # ここで予算を記憶
        self.columns = [
            'signal_date', 'stock_code', 'stock_name', 'prob', 'threshold', 
            'future_days', 'status', 'buy_price', 'sell_price', 'profit', 'profit_rate'
        ]
        self._init_csv()

    def _init_csv(self):
        """CSVファイルがなければ作る"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    # ★修正2: 引数から budget を削除（initで持っているので不要）
    def log_signal(self, date_str, code, name, prob, threshold, future_days):
        """買いシグナルが出た日に記録する"""
        if os.path.exists(self.filename):
            df = pd.read_csv(self.filename, encoding='utf-8')
            if not df.empty:
                # すでに同じ日・同じ銘柄があれば重複して記録しない
                exists = df[(df['signal_date'] == date_str) & (df['stock_code'] == str(code))]
                if not exists.empty:
                    return 
        
        # 新規記録 (結果はまだ分からないのでPENDINGとして保存)
        new_row = [date_str, code, name, prob, threshold, future_days, 'PENDING', 0, 0, 0, 0]
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

    def get_daily_report(self, stock_code, df_daily):
        """
        過去のトレードの答え合わせを行い、LINE通知用の文章を作って返す
        """
        # 1. 過去データの答え合わせ実行
        self._evaluate_past_trades(str(stock_code), df_daily)
        
        # 2. メッセージ作成
        msg_parts = []
        
        # 直近の結果があれば追加
        last_result = self._get_latest_result_msg()
        if last_result:
            msg_parts.append("📝 【直近の答え合わせ】")
            msg_parts.append(last_result)
            msg_parts.append("-" * 15)

        # 通算成績があれば追加
        summary = self._get_summary_msg()
        if summary:
            msg_parts.append(summary)
            msg_parts.append("-" * 15)
            
        return "\n".join(msg_parts) if msg_parts else ""

    def _evaluate_past_trades(self, stock_code, df_daily):
        """(内部処理) 過去のPENDINGデータを計算して更新"""
        if not os.path.exists(self.filename): return

        df_log = pd.read_csv(self.filename, encoding='utf-8')
        if df_log.empty: return
        
        # まだ結果が出ていない行を抽出
        targets = df_log[(df_log['stock_code'] == stock_code) & (df_log['status'] == 'PENDING')]
        updated = False
        
        # 日付型変換
        df_daily.index = pd.to_datetime(df_daily.index)

        for i, row in targets.iterrows():
            try:
                signal_date = pd.to_datetime(row['signal_date'])
                future_days = int(row['future_days'])
                
                # データ内にシグナル日があるか確認
                if signal_date not in df_daily.index: continue
                
                # 位置を取得
                sig_loc = df_daily.index.get_loc(signal_date)
                
                # 売り日（future_days後）のデータが存在するか確認
                if sig_loc + future_days < len(df_daily):
                    # 買い: 翌日(sig_loc+1)の始値
                    buy_price = df_daily.iloc[sig_loc + 1]['Open']
                    # 売り: 期限日(sig_loc+future_days)の終値
                    sell_price = df_daily.iloc[sig_loc + future_days]['Close']
                    
                    # ★修正3: self.budget を使用して購入株数を計算
                    lots = int(self.budget / buy_price)
                    if lots < 1: lots = 1

                    profit = (sell_price - buy_price) * lots
                    profit_rate = (profit / (buy_price * lots)) * 100
                    
                    # CSVを更新
                    df_log.at[i, 'buy_price'] = int(buy_price)
                    df_log.at[i, 'sell_price'] = int(sell_price)
                    df_log.at[i, 'profit'] = int(profit)
                    df_log.at[i, 'profit_rate'] = round(profit_rate, 2)
                    df_log.at[i, 'status'] = 'DONE'
                    updated = True
            except:
                continue

        if updated:
            df_log.to_csv(self.filename, index=False, encoding='utf-8')

    def _get_latest_result_msg(self):
        """(内部処理) 最近の結果メッセージ"""
        if not os.path.exists(self.filename): return None
        df = pd.read_csv(self.filename, encoding='utf-8')
        done = df[df['status'] == 'DONE']
        if done.empty: return None
        
        last = done.iloc[-1]
        icon = "🏆 勝ち" if last['profit'] > 0 else "💀 負け"
        return f"{last['signal_date']}シグナル → {icon}\n損益: {last['profit']:+.0f}円 ({last['profit_rate']:+.1f}%)"

    def _get_summary_msg(self):
        """(内部処理) 通算成績メッセージ"""
        if not os.path.exists(self.filename): return None
        df = pd.read_csv(self.filename, encoding='utf-8')
        done = df[df['status'] == 'DONE']
        if done.empty: return None

        total = len(done)
        wins = len(done[done['profit'] > 0])
        win_rate = (wins / total) * 100
        total_profit = done['profit'].sum()
        
        return (
            f"📊 通算成績 (フォワードテスト)\n"
            f"戦績: {total}戦 {wins}勝 {total-wins}敗\n"
            f"勝率: {win_rate:.1f}%\n"
            f"損益: {total_profit:+.0f}円"
        )