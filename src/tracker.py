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
        "stop_loss_percent",
        "entry_slippage_percent",
        "exit_slippage_percent",
        "stop_slippage_percent",
        "commission_percent",
        "status",
        "planned_buy_price",
        "actual_buy_price",
        "planned_sell_price",
        "actual_sell_price",
        "buy_price",
        "sell_price",
        "profit",
        "profit_rate",
        "commission",
        "exit_reason",
    ]

    def __init__(
        self,
        budget: int,
        filepath: str | Path = "data/trade_log_8306.csv",
        stop_loss_percent: float = 0.0,
        entry_slippage_percent: float = 0.0,
        exit_slippage_percent: float = 0.0,
        stop_slippage_percent: float = 0.0,
        commission_percent: float = 0.0,
    ) -> None:
        self.filepath = Path(filepath)
        self.budget = budget
        self.stop_loss_percent = stop_loss_percent
        self.entry_slippage_percent = entry_slippage_percent
        self.exit_slippage_percent = exit_slippage_percent
        self.stop_slippage_percent = stop_slippage_percent
        self.commission_percent = commission_percent
        self._init_csv()

    def _init_csv(self) -> None:
        """CSV ファイルが無ければ作成し、古い列構成なら補完する。"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.COLUMNS)
            return

        existing_columns = list(pd.read_csv(self.filepath, encoding="utf-8", nrows=0).columns)
        df = self._read_log()
        if existing_columns != self.COLUMNS:
            df.to_csv(self.filepath, index=False, encoding="utf-8")

    def _read_log(self) -> pd.DataFrame:
        """CSV を読み込み、銘柄コードの型と列順を正規化する。"""
        df = pd.read_csv(self.filepath, encoding="utf-8", dtype={"stock_code": str})
        for col in self.COLUMNS:
            if col not in df.columns:
                if col == "stop_loss_percent":
                    df[col] = self.stop_loss_percent
                elif col == "entry_slippage_percent":
                    df[col] = self.entry_slippage_percent
                elif col == "exit_slippage_percent":
                    df[col] = self.exit_slippage_percent
                elif col == "stop_slippage_percent":
                    df[col] = self.stop_slippage_percent
                elif col == "commission_percent":
                    df[col] = self.commission_percent
                elif col in {
                    "planned_buy_price",
                    "actual_buy_price",
                    "planned_sell_price",
                    "actual_sell_price",
                    "buy_price",
                    "sell_price",
                    "profit",
                    "profit_rate",
                    "commission",
                }:
                    df[col] = 0
                else:
                    df[col] = ""
        df["stock_code"] = df["stock_code"].astype(str)
        df["status"] = df["status"].fillna("").astype(str)
        df["exit_reason"] = df["exit_reason"].fillna("").astype(str)
        for col in (
            "prob",
            "threshold",
            "future_days",
            "stop_loss_percent",
            "entry_slippage_percent",
            "exit_slippage_percent",
            "stop_slippage_percent",
            "commission_percent",
            "planned_buy_price",
            "actual_buy_price",
            "planned_sell_price",
            "actual_sell_price",
            "buy_price",
            "sell_price",
            "profit",
            "profit_rate",
            "commission",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["future_days"] = df["future_days"].astype(int)
        df["profit"] = df["profit"].astype(int)
        for col in (
            "prob",
            "threshold",
            "stop_loss_percent",
            "entry_slippage_percent",
            "exit_slippage_percent",
            "stop_slippage_percent",
            "commission_percent",
            "planned_buy_price",
            "actual_buy_price",
            "planned_sell_price",
            "actual_sell_price",
            "buy_price",
            "sell_price",
            "profit_rate",
            "commission",
        ):
            df[col] = df[col].astype(float)
        return df[self.COLUMNS]

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
        stop_loss_percent: float | None = None,
        entry_slippage_percent: float | None = None,
        exit_slippage_percent: float | None = None,
        stop_slippage_percent: float | None = None,
        commission_percent: float | None = None,
    ) -> None:
        """買いシグナルが出た日に記録する。同日・同銘柄の重複は無視。"""
        if self.filepath.exists():
            df = self._read_log()
            if not df.empty:
                exists = df[(df["signal_date"] == date_str) & (df["stock_code"] == str(code))]
                if not exists.empty:
                    return

        stop_loss = self.stop_loss_percent if stop_loss_percent is None else stop_loss_percent
        entry_slippage = self.entry_slippage_percent if entry_slippage_percent is None else entry_slippage_percent
        exit_slippage = self.exit_slippage_percent if exit_slippage_percent is None else exit_slippage_percent
        stop_slippage = self.stop_slippage_percent if stop_slippage_percent is None else stop_slippage_percent
        commission_pct = self.commission_percent if commission_percent is None else commission_percent
        new_row = [
            date_str,
            code,
            name,
            prob,
            threshold,
            future_days,
            stop_loss,
            entry_slippage,
            exit_slippage,
            stop_slippage,
            commission_pct,
            "PENDING",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "",
        ]
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

        df_log = self._read_log()
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
                    planned_buy_price = df_daily.iloc[sig_loc + 1]["Open"]
                    entry_slippage_percent = float(row["entry_slippage_percent"])
                    exit_slippage_percent = float(row["exit_slippage_percent"])
                    stop_slippage_percent = float(row["stop_slippage_percent"])
                    commission_percent = float(row["commission_percent"])

                    actual_buy_price = planned_buy_price * (1 + entry_slippage_percent / 100)
                    exit_idx = sig_loc + future_days
                    planned_sell_price = df_daily.iloc[exit_idx]["Close"]
                    actual_sell_price = planned_sell_price * (1 - exit_slippage_percent / 100)
                    exit_reason = "TIME"
                    stop_loss_percent = float(row["stop_loss_percent"])

                    if stop_loss_percent > 0:
                        stop_price = actual_buy_price * (1 - stop_loss_percent / 100)
                        for j in range(sig_loc + 1, exit_idx + 1):
                            if df_daily.iloc[j]["Low"] <= stop_price:
                                planned_sell_price = stop_price
                                actual_sell_price = stop_price * (1 - stop_slippage_percent / 100)
                                exit_reason = "STOP"
                                break

                    lots = max(int(self.budget / actual_buy_price), 1)
                    gross_profit = (actual_sell_price - actual_buy_price) * lots
                    commission = (actual_buy_price + actual_sell_price) * lots * commission_percent / 100
                    profit = gross_profit - commission
                    profit_rate = (profit / (actual_buy_price * lots)) * 100

                    df_log.at[i, "planned_buy_price"] = round(float(planned_buy_price), 2)
                    df_log.at[i, "actual_buy_price"] = round(float(actual_buy_price), 2)
                    df_log.at[i, "planned_sell_price"] = round(float(planned_sell_price), 2)
                    df_log.at[i, "actual_sell_price"] = round(float(actual_sell_price), 2)
                    df_log.at[i, "buy_price"] = round(float(actual_buy_price), 2)
                    df_log.at[i, "sell_price"] = round(float(actual_sell_price), 2)
                    df_log.at[i, "profit"] = int(profit)
                    df_log.at[i, "profit_rate"] = round(profit_rate, 2)
                    df_log.at[i, "commission"] = round(float(commission), 2)
                    df_log.at[i, "exit_reason"] = exit_reason
                    df_log.at[i, "status"] = "DONE"
                    updated = True
            except Exception:
                continue

        if updated:
            df_log.to_csv(self.filepath, index=False, encoding="utf-8")

    def _get_latest_result_msg(self) -> str | None:
        if not self.filepath.exists():
            return None
        df = self._read_log()
        done = df[df["status"] == "DONE"]
        if done.empty:
            return None

        last = done.iloc[-1]
        icon = "🏆 勝ち" if last["profit"] > 0 else "💀 負け"
        exit_note = "（損切り）" if last["exit_reason"] == "STOP" else ""
        return f"{last['signal_date']}シグナル → {icon}{exit_note}\n損益: {last['profit']:+.0f}円 ({last['profit_rate']:+.1f}%)"

    def _get_summary_msg(self) -> str | None:
        if not self.filepath.exists():
            return None
        df = self._read_log()
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
