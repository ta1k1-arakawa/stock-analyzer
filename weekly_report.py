# weekly_report.py — trade_log.csv の週次サマリーレポート

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

LOG_PATH = Path("data/trade_log.csv")

# 月末判定の継続基準（事前合意）
WIN_RATE_MIN = 45.0
PROFIT_MIN = -5000
MIN_SIGNALS = 15


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


def run() -> None:
    if not LOG_PATH.exists():
        print(f"ログが存在しません: {LOG_PATH}")
        return

    df = pd.read_csv(LOG_PATH, encoding="utf-8")
    if df.empty:
        print("ログが空です。")
        return

    done = df[df["status"] == "DONE"].copy()
    pending = df[df["status"] == "PENDING"].copy()

    if done.empty:
        print("答え合わせ済みのトレードはまだありません。")
        if not pending.empty:
            print(f"PENDING: {len(pending)}件")
        return

    done["signal_date"] = pd.to_datetime(done["signal_date"])
    today = datetime.now().date()
    one_week_ago = today - timedelta(days=7)

    week = done[done["signal_date"].dt.date >= one_week_ago]

    print("=" * 50)
    print(f"週次レポート ({today})")
    print("=" * 50)

    _print_block("直近7日", _stats(week))
    print()
    _print_block("通算", _stats(done))

    if len(pending) > 0:
        print(f"\n  (PENDING: {len(pending)}件 答え合わせ待ち)")

    # 月末判定基準との照合
    total_stats = _stats(done)
    print("\n" + "-" * 50)
    print("【月末判定基準との照合 (通算ベース)】")
    checks = [
        ("勝率 ≥ 45%", total_stats["win_rate"] >= WIN_RATE_MIN,
         f"{total_stats['win_rate']:.1f}%"),
        ("利益 ≥ -5,000円", total_stats["profit"] >= PROFIT_MIN,
         _fmt(total_stats["profit"], "円")),
        ("シグナル数 ≥ 15件", total_stats["n"] >= MIN_SIGNALS,
         f"{total_stats['n']}件"),
    ]
    for name, ok, value in checks:
        mark = "OK " if ok else "NG "
        print(f"  [{mark}] {name:<20} (現在: {value})")

    all_ok = all(ok for _, ok, _ in checks)
    print("-" * 50)
    print(f"判定: {'継続条件を満たしています' if all_ok else '要注意 (基準未達)'}")


if __name__ == "__main__":
    run()
