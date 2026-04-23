# backtest.py — 銘柄選定 & パラメータ探索バックテスト
# target_percent × stop_loss × threshold を train/val/test の walk-forward で評価し、
# val で選定した params を test で評価する（不偏推定）。損切りは日次 Low で再現。

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import lightgbm as lgb
import pandas as pd

from src.analysis import calculate_indicators, create_target_variable, sanitize_ohlcv
from src.config import load_app
from src.fetchers.yfinance import YFinanceFetcher


TARGET_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
STOP_GRID = [2.0, 3.0, 5.0]
THRESHOLD_GRID = [0.15, 0.20, 0.30, 0.40, 0.50]

# (train_end, val_end, test_end) の比率
FOLDS = [
    (0.60, 0.70, 0.80),
    (0.70, 0.80, 0.90),
    (0.80, 0.90, 1.00),
]
BUDGET = 200_000


def _simulate(
    df: pd.DataFrame,
    probs,
    threshold: float,
    future_days: int,
    stop_loss_pct: float,
) -> tuple[int, int, int]:
    """threshold 超の日にエントリ、future_days 以内に stop に触れたら損切り。"""
    budget = BUDGET
    trades = 0
    wins = 0

    for i in range(len(df) - 1):
        if probs[i] < threshold:
            continue
        entry_idx = i + 1
        if entry_idx >= len(df):
            break

        entry_price = df.iloc[entry_idx]["Open"]
        if entry_price <= 0:
            continue

        exit_idx = min(entry_idx + future_days - 1, len(df) - 1)
        stop_price = entry_price * (1 - stop_loss_pct / 100)

        exit_price = None
        for j in range(entry_idx, exit_idx + 1):
            if df.iloc[j]["Low"] <= stop_price:
                exit_price = stop_price
                break
        if exit_price is None:
            exit_price = df.iloc[exit_idx]["Close"]

        qty = int(budget / entry_price)
        if qty <= 0:
            continue

        profit = (exit_price - entry_price) * qty
        budget += profit
        trades += 1
        if profit > 0:
            wins += 1

    return int(budget - BUDGET), trades, wins


def _build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        deterministic=True,
        n_jobs=1,
    )


def _run_fold(
    labeled_by_tp: dict[float, pd.DataFrame],
    feature_cols: list[str],
    future_days: int,
    ratios: tuple[float, float, float],
) -> dict | None:
    """1 fold: tp ごとに学習し、val で (tp, stop, thr) 最良を選び test で評価。"""
    tr_ratio, val_ratio, te_ratio = ratios

    best_val_profit = float("-inf")
    best_test_result: dict | None = None

    for tp, df_labeled in labeled_by_tp.items():
        m = len(df_labeled)
        tr_end = int(m * tr_ratio)
        val_end = int(m * val_ratio)
        te_end = int(m * te_ratio)

        train_df = df_labeled.iloc[:tr_end]
        val_df = df_labeled.iloc[tr_end:val_end]
        test_df = df_labeled.iloc[val_end:te_end]

        if len(train_df["Target"].unique()) < 2 or len(val_df) < 10 or len(test_df) < 10:
            continue

        model = _build_model()
        model.fit(train_df[feature_cols], train_df["Target"])
        val_probs = model.predict_proba(val_df[feature_cols])[:, 1]
        test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

        for sl in STOP_GRID:
            for th in THRESHOLD_GRID:
                val_profit, _, _ = _simulate(val_df, val_probs, th, future_days, sl)
                if val_profit > best_val_profit:
                    best_val_profit = val_profit
                    t_profit, t_trades, t_wins = _simulate(test_df, test_probs, th, future_days, sl)
                    best_test_result = {
                        "tp": tp,
                        "stop": sl,
                        "thr": th,
                        "val_profit": val_profit,
                        "test_profit": t_profit,
                        "test_trades": t_trades,
                        "test_wins": t_wins,
                        "test_n": len(test_df),
                    }

    return best_test_result


def run_backtest() -> None:
    config, _ = load_app(log_file="backtest.log")
    ai = config.ai_params
    feature_cols = config.feature_columns
    candidates = config.backtest_candidates or [str(config.stock_code)]

    data_from = config.training_settings.get("data_from")
    data_to = config.training_settings.get("data_to")
    if data_to == "auto":
        data_to = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    total_combos = len(TARGET_GRID) * len(STOP_GRID) * len(THRESHOLD_GRID)
    print(f"\n=== Walk-forward グリッドサーチ ({len(candidates)}銘柄 × {len(FOLDS)} folds) ===")
    print(f"期間: {data_from} 〜 {data_to}")
    print(f"future_days: {ai.future_days}日 / budget: {BUDGET:,}円")
    print(f"グリッド: tp×{len(TARGET_GRID)} × stop×{len(STOP_GRID)} × thr×{len(THRESHOLD_GRID)} = {total_combos} 組合せ")
    print("各 fold: train で学習 → val で (tp, stop, thr) 選定 → test で評価\n")

    fetcher = YFinanceFetcher()
    per_stock_summary: list[dict] = []
    all_fold_rows: list[dict] = []

    for code in candidates:
        code = str(code)
        print(f"-- {code} --")
        try:
            df = fetcher.get_daily_stock_prices(code, data_from, data_to)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        if df is None or df.empty:
            print("  Skip (No Data)")
            continue

        df = sanitize_ohlcv(df)
        df_ta = calculate_indicators(df, config.tech_params)

        # tp ごとにラベル作成を一度だけ行う
        labeled_by_tp: dict[float, pd.DataFrame] = {}
        for tp in TARGET_GRID:
            df_labeled = create_target_variable(df_ta, ai.future_days, tp)
            df_labeled = df_labeled.dropna(subset=feature_cols + ["Target"])
            if len(df_labeled) >= 100:
                labeled_by_tp[tp] = df_labeled

        if not labeled_by_tp:
            print("  Skip (データ不足)")
            continue

        fold_results = []
        for fi, ratios in enumerate(FOLDS, 1):
            result = _run_fold(labeled_by_tp, feature_cols, ai.future_days, ratios)
            if result is None:
                print(f"  Fold{fi}: skip (分割不足)")
                continue
            row = {"code": code, "fold": fi, **result}
            fold_results.append(row)
            all_fold_rows.append(row)
            print(
                f"  Fold{fi}: val選定 tp={result['tp']} stop={result['stop']} thr={result['thr']} "
                f"→ test {result['test_profit']:+,d}円 "
                f"(取引{result['test_trades']}, 勝ち{result['test_wins']})"
            )

        if not fold_results:
            continue

        total_profit = sum(r["test_profit"] for r in fold_results)
        avg_profit = int(total_profit / len(fold_results))
        min_profit = min(r["test_profit"] for r in fold_results)
        folds_pos = sum(1 for r in fold_results if r["test_profit"] > 0)
        total_trades = sum(r["test_trades"] for r in fold_results)
        total_wins = sum(r["test_wins"] for r in fold_results)
        win_rate = round(total_wins / total_trades * 100, 1) if total_trades else 0.0

        combos = [(r["tp"], r["stop"], r["thr"]) for r in fold_results]
        combo_mode, combo_agree = Counter(combos).most_common(1)[0]
        tp_mode, stop_mode, thr_mode = combo_mode

        per_stock_summary.append({
            "Code": code,
            "TotalTest": total_profit,
            "AvgPerFold": avg_profit,
            "MinFold": min_profit,
            "FoldsPos": f"{folds_pos}/{len(fold_results)}",
            "Trades": total_trades,
            "WinRate": win_rate,
            "ModeTP": tp_mode,
            "ModeStop": stop_mode,
            "ModeThr": thr_mode,
            "ComboAgree": f"{combo_agree}/{len(fold_results)}",
        })

    # レポート
    print("\n" + "=" * 96)
    print("【銘柄別サマリー (walk-forward test 合計利益で降順)】")
    print("=" * 96)

    if not per_stock_summary:
        print("結果なし")
        return

    df_sum = pd.DataFrame(per_stock_summary).sort_values("TotalTest", ascending=False)
    print(df_sum.to_string(index=False))

    print("\n" + "=" * 96)
    print("【全 fold 詳細】")
    print("=" * 96)
    df_fold = pd.DataFrame(all_fold_rows)[
        ["code", "fold", "tp", "stop", "thr", "val_profit", "test_profit", "test_trades", "test_wins"]
    ].sort_values(["code", "fold"])
    print(df_fold.to_string(index=False))

    best = df_sum.iloc[0]
    print("\n" + "=" * 96)
    print(">>> 推奨設定")
    print("=" * 96)
    print(f"  銘柄              : {best['Code']}")
    print(f"  target_percent    : {best['ModeTP']}")
    print(f"  stop_loss_percent : {best['ModeStop']}")
    print(f"  threshold         : {best['ModeThr']}")
    print(f"  walk-forward test : 合計 {int(best['TotalTest']):+,d}円 / 平均 {int(best['AvgPerFold']):+,d}円/fold")
    print(f"  最悪 fold         : {int(best['MinFold']):+,d}円")
    print(f"  勝ち fold         : {best['FoldsPos']}")
    print(f"  取引数 / 勝率     : {best['Trades']} 回 / {best['WinRate']}%")
    print(f"  組合せ一致度      : {best['ComboAgree']}  (fold 間で同じ組合せが選ばれた数)")
    print("\n※ val で選定した params を test で評価した不偏推定です。")
    print("  ComboAgree が低い銘柄はパラメータが時期依存なので、運用時の分散も検討してください。")


if __name__ == "__main__":
    run_backtest()
