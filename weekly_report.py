# weekly_report.py — trade_log.csv の週次サマリーレポート

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import load_app



def _fmt(value: float, suffix: str = "") -> str:
    return f"{value:+,.0f}{suffix}" if isinstance(value, (int, float)) else str(value)


def _stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "profit": 0, "avg_profit": 0}
    wins = int((df["profit"] > 0).sum())
    profit = int(df["profit"].sum())
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n * 100,
        "profit": profit,
        "avg_profit": profit / n,
    }


def _print_block(title: str, s: dict) -> None:
    print(f"【{title}】")
    if s["n"] == 0:
        print("  シグナルなし")
        return
    print(f"  シグナル数: {s['n']}件 ({s['wins']}勝 / {s['n'] - s['wins']}敗)")
    print(f"  勝率: {s['win_rate']:.1f}%")
    print(f"  損益: {_fmt(s['profit'], '円')} (平均 {_fmt(s['avg_profit'], '円/件')})")


def _report_stock(
    code: str,
    name: str,
    log_path: Path,
    today: date,
) -> dict | None:
    print("=" * 60)
    print(f"{name} ({code})")
    print(f"ログ: {log_path}")
    print("=" * 60)

    if not log_path.exists():
        print("ログはまだ作成されていません。\n")
        return None

    df = pd.read_csv(log_path, encoding="utf-8", dtype={"stock_code": str})
    if df.empty:
        print("ログが空です。\n")
        return {
            "code": code,
            "name": name,
            "n": 0,
            "wins": 0,
            "win_rate": 0.0,
            "profit": 0,
            "avg_profit": 0,
            "pending": 0,
        }

    done = df[df["status"] == "DONE"].copy()
    pending = df[df["status"] == "PENDING"].copy()

    if done.empty:
        print("答え合わせ済みのトレードはまだありません。")
        if not pending.empty:
            print(f"PENDING: {len(pending)}件")
        print()
        return {
            "code": code,
            "name": name,
            "n": 0,
            "wins": 0,
            "win_rate": 0.0,
            "profit": 0,
            "avg_profit": 0,
            "pending": len(pending),
        }

    done["signal_date"] = pd.to_datetime(done["signal_date"])
    done["profit"] = pd.to_numeric(done["profit"], errors="coerce").fillna(0)
    one_week_ago = today - timedelta(days=7)

    week = done[done["signal_date"].dt.date >= one_week_ago]

    print(f"週次レポート ({today})")
    _print_block("直近7日", _stats(week))
    print()
    _print_block("通算", _stats(done))

    if len(pending) > 0:
        print(f"\n  (PENDING: {len(pending)}件 答え合わせ待ち)")

    result = _stats(done)
    result.update(
        {
            "code": code,
            "name": name,
            "pending": len(pending),
        }
    )
    return result


def run() -> None:
    config, _ = load_app(log_file="weekly_report.log", console=False)
    targets = [(config.stock_code, config.stock_name, config.trade_log_path)] + [
        (stock.stock_code, stock.stock_name, stock.trade_log_path)
        for stock in config.log_only_stocks
    ]
    today = datetime.now().date()
    summaries = [
        summary
        for code, name, path in targets
        if (summary := _report_stock(code, name, path, today)) is not None
    ]

    if not summaries:
        print("集計できるログがまだありません。")
        return

    print("=" * 60)
    print("銘柄別 通算成績（損益順）")
    print("=" * 60)
    for summary in sorted(summaries, key=lambda item: item["profit"], reverse=True):
        print(
            f"{summary['code']} {summary['name']}: "
            f"{summary['n']}戦 / 勝率 {summary['win_rate']:.1f}% / "
            f"損益 {summary['profit']:+,.0f}円 / PENDING {summary['pending']}件"
        )


if __name__ == "__main__":
    run()
