from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.tracker import TradeTracker


class TradeTrackerDailyReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "trade_log.csv"
        self.tracker = TradeTracker(budget=100_000, filepath=self.log_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _prices() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [100.0, 100.0, 105.0, 110.0, 112.0],
                "Low": [99.0, 99.0, 104.0, 109.0, 111.0],
                "Close": [100.0, 101.0, 110.0, 111.0, 120.0],
            },
            index=pd.to_datetime(
                [
                    "2026-07-20",
                    "2026-07-21",
                    "2026-07-22",
                    "2026-07-23",
                    "2026-07-24",
                ]
            ),
        )

    def _log_signal(self, signal_date: str, future_days: int = 2) -> None:
        self.tracker.log_signal(
            date_str=signal_date,
            code="1234",
            name="テスト株",
            prob=0.8,
            threshold=0.5,
            future_days=future_days,
        )

    def test_report_shows_only_trades_sold_on_report_date(self) -> None:
        self._log_signal("2026-07-20")
        self._log_signal("2026-07-22")

        report = self.tracker.get_daily_report(
            "1234",
            self._prices(),
            report_date="2026-07-24",
        )

        self.assertIn("【本日の答え合わせ】", report)
        self.assertIn("シグナル日: 2026-07-22", report)
        self.assertIn("売却日: 2026-07-24", report)
        self.assertNotIn("シグナル日: 2026-07-20", report)
        self.assertIn("📊 通算成績 (フォワードテスト)", report)
        self.assertIn("戦績: 2戦", report)

    def test_report_does_not_repeat_an_old_result_when_nothing_was_sold_today(self) -> None:
        self._log_signal("2026-07-20")

        report = self.tracker.get_daily_report(
            "1234",
            self._prices(),
            report_date="2026-07-24",
        )

        self.assertIn("【本日の答え合わせ】", report)
        self.assertIn("本日売却した取引はありません。", report)
        self.assertNotIn("シグナル日: 2026-07-20", report)
        self.assertIn("📊 通算成績 (フォワードテスト)", report)
        self.assertIn("戦績: 1戦", report)

    def test_stop_loss_is_reported_on_the_day_it_triggers(self) -> None:
        tracker = TradeTracker(
            budget=100_000,
            filepath=self.log_path,
            stop_loss_percent=3.0,
        )
        tracker.log_signal(
            date_str="2026-07-20",
            code="1234",
            name="テスト株",
            prob=0.8,
            threshold=0.5,
            future_days=5,
            stop_loss_percent=3.0,
        )
        prices = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "Low": [99.0, 96.0],
                "Close": [100.0, 97.0],
            },
            index=pd.to_datetime(["2026-07-20", "2026-07-21"]),
        )

        report = tracker.get_daily_report(
            "1234",
            prices,
            report_date="2026-07-21",
        )

        self.assertIn("【本日の答え合わせ】", report)
        self.assertIn("売却日: 2026-07-21", report)
        self.assertIn("（損切り）", report)

        saved = pd.read_csv(self.log_path, dtype={"stock_code": str})
        self.assertEqual(saved.loc[0, "status"], "DONE")
        self.assertEqual(saved.loc[0, "exit_date"], "2026-07-21")

    def test_existing_done_log_gets_its_exit_date_backfilled(self) -> None:
        old_columns = TradeTracker.COLUMNS[:-1]
        old_row = [
            "2026-07-20",
            "1234",
            "テスト株",
            0.8,
            0.5,
            2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "DONE",
            100.0,
            100.0,
            110.0,
            110.0,
            100.0,
            110.0,
            10_000,
            10.0,
            0.0,
            "TIME",
        ]
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(old_columns)
            writer.writerow(old_row)

        tracker = TradeTracker(budget=100_000, filepath=self.log_path)
        report = tracker.get_daily_report(
            "1234",
            self._prices(),
            report_date="2026-07-22",
        )

        self.assertIn("【本日の答え合わせ】", report)
        self.assertIn("売却日: 2026-07-22", report)
        saved = pd.read_csv(self.log_path, dtype={"stock_code": str})
        self.assertIn("exit_date", saved.columns)
        self.assertEqual(saved.loc[0, "exit_date"], "2026-07-22")


if __name__ == "__main__":
    unittest.main()
