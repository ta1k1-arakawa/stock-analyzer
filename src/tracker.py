"""トレード記録および成績レポートモジュール。"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import pandas as pd

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TradeTracker:
    """買いシグナルのログ記録と成績追跡を行う。"""

    COLUMNS = [
        "signal_date",
        "stock_code",
        "stock_name",
        "prob",
        "threshold",
        "future_days",
        "status",
        "buy_price",
        "sell_price",
        "profit",
        "profit_rate",
    ]

    def __init__(self, budget: int, filepath: str | Path = "data/trade_log.csv") -> None:
        self.filepath = Path(filepath)
        self.budget = budget
        self._init_csv()

    def _init_csv(self) -> None:
        """CSV ファイルが無ければヘッダ付きで作成する。"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.COLUMNS)

    # ------------------------------------------------------------------
    # シグナル記録
    # ------------------------------------------------------------------

    def log_signal(
        self,
        date_str: str,
        code: str,
        name: str,
        prob: float,
        threshold: float,
        future_days: int,
    ) -> None:
        """買いシグナルが出た日に記録する。同日・同銘柄の重複は無視。"""
        if self.filepath.exists():
            df = pd.read_csv(self.filepath, encoding="utf-8")
            if not df.empty:
                exists = df[(df["signal_date"] == date_str) & (df["stock_code"] == str(code))]
                if not exists.empty:
                    return

        new_row = [date_str, code, name, prob, threshold, future_days, "PENDING", 0, 0, 0, 0]
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(new_row)

    # ------------------------------------------------------------------
    # デイリーレポート
    # ------------------------------------------------------------------

    def get_daily_report(self, stock_code: str, df_daily: pd.DataFrame) -> str:
        """過去トレードの答え合わせを行い、LINE 通知用メッセージを返す。"""
        self._evaluate_past_trades(str(stock_code), df_daily)

        parts: list[str] = []

        last_result = self._get_latest_result_msg()
        if last_result:
            parts.append("📝 【直近の答え合わせ】")
            parts.append(last_result)
            parts.append("-" * 15)

        summary = self._get_summary_msg()
        if summary:
            parts.append(summary)
            parts.append("-" * 15)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------

    def _evaluate_past_trades(self, stock_code: str, df_daily: pd.DataFrame) -> None:
        """PENDING 状態のトレードを実価格で評価して CSV を更新する。"""
        if not self.filepath.exists():
            return

        df_log = pd.read_csv(self.filepath, encoding="utf-8")
        if df_log.empty:
            return

        targets = df_log[(df_log["stock_code"] == stock_code) & (df_log["status"] == "PENDING")]
        updated = False

        df_daily.index = pd.to_datetime(df_daily.index)

        for i, row in targets.iterrows():
            try:
                signal_date = pd.to_datetime(row["signal_date"])
                future_days = int(row["future_days"])

                if signal_date not in df_daily.index:
                    continue

                sig_loc = df_daily.index.get_loc(signal_date)

                if sig_loc + future_days < len(df_daily):
                    buy_price = df_daily.iloc[sig_loc + 1]["Open"]
                    sell_price = df_daily.iloc[sig_loc + future_days]["Close"]

                    lots = max(int(self.budget / buy_price), 1)
                    profit = (sell_price - buy_price) * lots
                    profit_rate = (profit / (buy_price * lots)) * 100

                    df_log.at[i, "buy_price"] = int(buy_price)
                    df_log.at[i, "sell_price"] = int(sell_price)
                    df_log.at[i, "profit"] = int(profit)
                    df_log.at[i, "profit_rate"] = round(profit_rate, 2)
                    df_log.at[i, "status"] = "DONE"
                    updated = True
            except Exception:
                continue

        if updated:
            df_log.to_csv(self.filepath, index=False, encoding="utf-8")

    def _get_latest_result_msg(self) -> str | None:
        if not self.filepath.exists():
            return None
        df = pd.read_csv(self.filepath, encoding="utf-8")
        done = df[df["status"] == "DONE"]
        if done.empty:
            return None

        last = done.iloc[-1]
        icon = "🏆 勝ち" if last["profit"] > 0 else "💀 負け"
        return f"{last['signal_date']}シグナル → {icon}\n損益: {last['profit']:+.0f}円 ({last['profit_rate']:+.1f}%)"

    def _get_summary_msg(self) -> str | None:
        if not self.filepath.exists():
            return None
        df = pd.read_csv(self.filepath, encoding="utf-8")
        done = df[df["status"] == "DONE"]
        if done.empty:
            return None

        total = len(done)
        wins = len(done[done["profit"] > 0])
        win_rate = (wins / total) * 100
        total_profit = done["profit"].sum()

        return (
            f"📊 通算成績 (フォワードテスト)\n"
            f"戦績: {total}戦 {wins}勝 {total - wins}敗\n"
            f"勝率: {win_rate:.1f}%\n"
            f"損益: {total_profit:+.0f}円"
        )
