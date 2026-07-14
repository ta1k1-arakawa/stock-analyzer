# weekly_report.py — 銘柄別 trade_log_<code>.csv の週次サマリーレポート

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import load_app



def _empty_stats() -> dict:
    return {"n": 0, "wins": 0, "win_rate": 0.0, "profit": 0, "avg_profit": 0}


def _fmt(value: float, suffix: str = "") -> str:
    return f"{value:+,.0f}{suffix}" if isinstance(value, (int, float)) else str(value)


def _stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return _empty_stats()
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


def _count_pending(
    pending: pd.DataFrame,
    one_week_ago: date,
    one_month_ago: date,
) -> dict:
    if pending.empty:
        return {"week": 0, "month": 0, "all": 0}

    signal_dates = pd.to_datetime(pending["signal_date"], errors="coerce").dt.date
    return {
        "week": int((signal_dates >= one_week_ago).sum()),
        "month": int((signal_dates >= one_month_ago).sum()),
        "all": len(pending),
    }


def _empty_stock_summary(code: str, name: str, pending: dict | None = None) -> dict:
    return {
        "code": code,
        "name": name,
        "pending": pending or {"week": 0, "month": 0, "all": 0},
        "week": _empty_stats(),
        "month": _empty_stats(),
        "all": _empty_stats(),
    }


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
        return _empty_stock_summary(code, name)

    done = df[df["status"] == "DONE"].copy()
    pending = df[df["status"] == "PENDING"].copy()
    one_week_ago = today - timedelta(days=7)
    one_month_ago = today - timedelta(days=30)
    pending_counts = _count_pending(pending, one_week_ago, one_month_ago)

    if done.empty:
        print("答え合わせ済みのトレードはまだありません。")
        if not pending.empty:
            print(f"PENDING: {len(pending)}件")
        print()
        return _empty_stock_summary(code, name, pending=pending_counts)

    done["signal_date"] = pd.to_datetime(done["signal_date"])
    done["profit"] = pd.to_numeric(done["profit"], errors="coerce").fillna(0)

    week = done[done["signal_date"].dt.date >= one_week_ago]
    month = done[done["signal_date"].dt.date >= one_month_ago]

    print(f"週次レポート ({today})")
    _print_block("直近7日", _stats(week))
    print()
    _print_block("直近1か月（30日）", _stats(month))
    print()
    _print_block("通算", _stats(done))

    if len(pending) > 0:
        print(f"\n  (PENDING: {len(pending)}件 答え合わせ待ち)")

    return {
        "code": code,
        "name": name,
        "pending": pending_counts,
        "week": _stats(week),
        "month": _stats(month),
        "all": _stats(done),
    }


def _print_ranking(title: str, summaries: list[dict], stats_key: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    for summary in sorted(
        summaries,
        key=lambda item: item[stats_key]["profit"],
        reverse=True,
    ):
        stats = summary[stats_key]
        pending = summary["pending"][stats_key]
        print(
            f"{summary['code']} {summary['name']}: "
            f"{stats['n']}戦 / 勝率 {stats['win_rate']:.1f}% / "
            f"損益 {stats['profit']:+,.0f}円 / PENDING {pending}件"
        )


def run() -> None:
    config, _ = load_app(log_file="weekly_report.log", console=False)
    targets = [
        (stock.stock_code, stock.stock_name, stock.trade_log_path)
        for stock in config.stocks
        if stock.paper_trade
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

    _print_ranking("銘柄別 通算成績（損益順）", summaries, "all")
    _print_ranking("銘柄別 直近7日成績（損益順）", summaries, "week")
    _print_ranking("銘柄別 直近1か月成績（30日・損益順）", summaries, "month")


if __name__ == "__main__":
    run()
