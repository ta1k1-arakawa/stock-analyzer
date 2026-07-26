"""トレード記録および成績レポートモジュール。"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from numbers import Integral
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
        "exit_date",
    ]

    def __init__(
        self,
        budget: int,
        filepath: str | Path,
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
        df["exit_date"] = df["exit_date"].fillna("").astype(str)
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
            "",
        ]
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(new_row)

    # ------------------------------------------------------------------
    # デイリーレポート
    # ------------------------------------------------------------------

    def get_daily_report(
        self,
        stock_code: str,
        df_daily: pd.DataFrame,
        report_date: str | pd.Timestamp | None = None,
    ) -> str:
        """過去トレードを評価し、当日売却分と通算成績を返す。"""
        self._evaluate_past_trades(str(stock_code), df_daily)

        parts: list[str] = []

        if report_date is None:
            report_date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            report_date_str = pd.to_datetime(report_date).strftime("%Y-%m-%d")

        todays_results = self._get_results_msg_for_date(report_date_str)
        parts.append("📝 【本日の答え合わせ】")
        parts.append(todays_results or "本日売却した取引はありません。")
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

        df_daily = df_daily.copy()
        df_daily.index = pd.to_datetime(df_daily.index)

        for i, row in targets.iterrows():
            try:
                result = self._calculate_trade_result(row, df_daily)
                if result is None:
                    continue

                for column, value in result.items():
                    df_log.at[i, column] = value
                df_log.at[i, "status"] = "DONE"
                updated = True
            except Exception:
                continue

        # exit_date 導入前の完了済みログも、価格データから売却日を復元する。
        missing_exit_date = df_log[
            (df_log["stock_code"] == stock_code)
            & (df_log["status"] == "DONE")
            & (df_log["exit_date"] == "")
        ]
        for i, row in missing_exit_date.iterrows():
            try:
                result = self._calculate_trade_result(row, df_daily)
                if result is not None:
                    df_log.at[i, "exit_date"] = result["exit_date"]
                    updated = True
            except Exception:
                continue

        if updated:
            df_log.to_csv(self.filepath, index=False, encoding="utf-8")

    def _calculate_trade_result(
        self,
        row: pd.Series,
        df_daily: pd.DataFrame,
    ) -> dict[str, float | int | str] | None:
        """利用可能な価格データ内で売却済みなら、取引結果を返す。"""
        signal_date = pd.to_datetime(row["signal_date"])
        future_days = int(row["future_days"])

        if signal_date not in df_daily.index:
            return None

        sig_loc = df_daily.index.get_loc(signal_date)
        if not isinstance(sig_loc, Integral):
            return None
        sig_loc = int(sig_loc)

        buy_idx = sig_loc + 1
        planned_exit_idx = sig_loc + future_days
        if buy_idx >= len(df_daily) or planned_exit_idx < buy_idx:
            return None

        planned_buy_price = float(df_daily.iloc[buy_idx]["Open"])
        entry_slippage_percent = float(row["entry_slippage_percent"])
        exit_slippage_percent = float(row["exit_slippage_percent"])
        stop_slippage_percent = float(row["stop_slippage_percent"])
        commission_percent = float(row["commission_percent"])
        stop_loss_percent = float(row["stop_loss_percent"])
        actual_buy_price = planned_buy_price * (1 + entry_slippage_percent / 100)

        exit_idx: int | None = None
        exit_reason = ""
        planned_sell_price = 0.0
        actual_sell_price = 0.0

        if stop_loss_percent > 0:
            stop_price = actual_buy_price * (1 - stop_loss_percent / 100)
            last_check_idx = min(planned_exit_idx, len(df_daily) - 1)
            for j in range(buy_idx, last_check_idx + 1):
                if float(df_daily.iloc[j]["Low"]) <= stop_price:
                    exit_idx = j
                    planned_sell_price = stop_price
                    actual_sell_price = stop_price * (1 - stop_slippage_percent / 100)
                    exit_reason = "STOP"
                    break

        if exit_idx is None:
            if planned_exit_idx >= len(df_daily):
                return None
            exit_idx = planned_exit_idx
            planned_sell_price = float(df_daily.iloc[exit_idx]["Close"])
            actual_sell_price = planned_sell_price * (1 - exit_slippage_percent / 100)
            exit_reason = "TIME"

        lots = max(int(self.budget / actual_buy_price), 1)
        gross_profit = (actual_sell_price - actual_buy_price) * lots
        commission = (actual_buy_price + actual_sell_price) * lots * commission_percent / 100
        profit = gross_profit - commission
        profit_rate = (profit / (actual_buy_price * lots)) * 100

        return {
            "planned_buy_price": round(planned_buy_price, 2),
            "actual_buy_price": round(actual_buy_price, 2),
            "planned_sell_price": round(planned_sell_price, 2),
            "actual_sell_price": round(actual_sell_price, 2),
            "buy_price": round(actual_buy_price, 2),
            "sell_price": round(actual_sell_price, 2),
            "profit": int(profit),
            "profit_rate": round(profit_rate, 2),
            "commission": round(commission, 2),
            "exit_reason": exit_reason,
            "exit_date": pd.Timestamp(df_daily.index[exit_idx]).strftime("%Y-%m-%d"),
        }

    def _get_results_msg_for_date(self, exit_date: str) -> str | None:
        if not self.filepath.exists():
            return None
        df = self._read_log()
        sold = df[(df["status"] == "DONE") & (df["exit_date"] == exit_date)]
        if sold.empty:
            return None

        results: list[str] = []
        for _, trade in sold.iterrows():
            icon = "🏆 勝ち" if trade["profit"] > 0 else "💀 負け"
            exit_note = "（損切り）" if trade["exit_reason"] == "STOP" else ""
            results.append(
                f"シグナル日: {trade['signal_date']} → {icon}{exit_note}\n"
                f"売却日: {trade['exit_date']}\n"
                f"損益: {trade['profit']:+.0f}円 ({trade['profit_rate']:+.1f}%)"
            )
        return "\n\n".join(results)

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
