# backtest.py — 銘柄選定バックテスト エントリポイント
# Walk-forward + train/val/test 分割で threshold 選定バイアスを排除する。

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import lightgbm as lgb
import pandas as pd

from src.analysis import calculate_indicators, create_target_variable, sanitize_ohlcv
from src.config import load_app
from src.fetchers.yfinance import YFinanceFetcher


THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Walk-forward の各 fold: (train終端, val終端, test終端) を全体に対する割合で定義
FOLDS = [
    (0.60, 0.70, 0.80),
    (0.70, 0.80, 0.90),
    (0.80, 0.90, 1.00),
]


def _simulate(test_df: pd.DataFrame, probs, threshold: float, future_days: int) -> tuple[int, int, int]:
    """threshold 条件で期間をシミュレートし、(利益, 取引数, 勝数) を返す。"""
    initial_budget = 100000
    budget = initial_budget
    trade_count = 0
    wins = 0

    for i in range(len(test_df) - 1):
        if probs[i] < threshold:
            continue
        entry_idx = i + 1
        if entry_idx >= len(test_df):
            break

        entry_price = test_df.iloc[entry_idx]["Open"]
        exit_idx = entry_idx + (future_days - 1)
        if exit_idx >= len(test_df):
            continue

        exit_price = test_df.iloc[exit_idx]["Close"]
        if entry_price <= 0:
            continue

        qty = int(budget / entry_price)
        if qty <= 0:
            continue

        profit = (exit_price - entry_price) * qty
        budget += profit
        trade_count += 1
        if profit > 0:
            wins += 1

    return int(budget - initial_budget), trade_count, wins


def _build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        deterministic=True,
        n_jobs=1,
    )


def _run_fold(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    ai,
    ratios: tuple[float, float, float],
) -> dict | None:
    """1 fold を実行: train で学習 → val で閾値選定 → test で評価。"""
    n = len(df_model)
    tr_end = int(n * ratios[0])
    val_end = int(n * ratios[1])
    test_end = int(n * ratios[2])

    train_df = df_model.iloc[:tr_end]
    val_df = df_model.iloc[tr_end:val_end]
    test_df = df_model.iloc[val_end:test_end]

    if len(train_df["Target"].unique()) < 2 or len(val_df) < 10 or len(test_df) < 10:
        return None

    model = _build_model()
    model.fit(train_df[feature_cols], train_df["Target"])

    # val で threshold 選定
    val_probs = model.predict_proba(val_df[feature_cols])[:, 1]
    best_th = THRESHOLD_GRID[0]
    best_val_profit = float("-inf")
    for th in THRESHOLD_GRID:
        profit, _, _ = _simulate(val_df, val_probs, th, ai.future_days)
        if profit > best_val_profit:
            best_val_profit = profit
            best_th = th

    # test で選定した threshold を評価（未知データ）
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]
    test_profit, test_trades, test_wins = _simulate(test_df, test_probs, best_th, ai.future_days)
    test_winrate = (test_wins / test_trades * 100) if test_trades > 0 else 0.0

    return {
        "TrainN": len(train_df),
        "ValN": len(val_df),
        "TestN": len(test_df),
        "BestTh": best_th,
        "ValProfit": int(best_val_profit),
        "TestProfit": test_profit,
        "TestWinRate": round(test_winrate, 1),
        "TestTrades": test_trades,
    }


def run_backtest() -> None:
    """全候補銘柄 × Walk-forward folds でバックテストを実行する。"""
    config, _ = load_app(log_file="backtest.log")

    candidates = config.backtest_candidates
    ai = config.ai_params
    feature_cols = config.feature_columns

    # 期間
    data_from = config.training_settings.get("data_from")
    data_to = config.training_settings.get("data_to")
    if data_to == "auto":
        data_to = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"終了日を自動設定: {data_to}")

    fetcher = YFinanceFetcher()

    print(f"\n=== Walk-forward バックテスト ({len(candidates)}銘柄 × {len(FOLDS)} folds) ===")
    print(f"期間: {data_from} 〜 {data_to}")
    print(f"ラベル条件: {ai.future_days}日後に +{ai.target_percent}% 以上")
    print("各 fold: train で学習 → val で閾値選定 → test で未知データ評価\n")

    all_fold_rows: list[dict] = []
    per_stock_summary: list[dict] = []

    for code in candidates:
        print(f"Testing: {code}")

        try:
            df = fetcher.get_daily_stock_prices(str(code), data_from, data_to)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if df is None or df.empty:
            print("  Skip (No Data)")
            continue

        df = sanitize_ohlcv(df)
        df_ta = calculate_indicators(df, config.tech_params)
        df_model = create_target_variable(df_ta, ai.future_days, ai.target_percent)

        missing = [c for c in feature_cols if c not in df_model.columns]
        if missing:
            print(f"  Skip (特徴量不足: {missing})")
            continue

        df_model = df_model.dropna(subset=feature_cols + ["Target"])

        if len(df_model) < 100:
            print(f"  Skip (データ不足: {len(df_model)}件)")
            continue

        fold_results: list[dict] = []
        for fi, ratios in enumerate(FOLDS, 1):
            result = _run_fold(df_model, feature_cols, ai, ratios)
            if result is None:
                print(f"  Fold{fi}: skip (分割不足 or 学習データ単一)")
                continue
            result = {"Code": code, "Fold": fi, **result}
            fold_results.append(result)
            all_fold_rows.append(result)
            print(
                f"  Fold{fi} [tr={result['TrainN']} val={result['ValN']} test={result['TestN']}] "
                f"val選定th={result['BestTh']} valProfit={result['ValProfit']:+d} "
                f"→ testProfit={result['TestProfit']:+d} "
                f"(勝率{result['TestWinRate']}% N={result['TestTrades']})"
            )

        if not fold_results:
            continue

        test_profits = [r["TestProfit"] for r in fold_results]
        ths = [r["BestTh"] for r in fold_results]
        th_counter = Counter(ths)
        mode_th, mode_count = th_counter.most_common(1)[0]
        folds_positive = sum(1 for p in test_profits if p > 0)
        avg_profit = int(sum(test_profits) / len(test_profits))
        min_profit = min(test_profits)
        total_profit = sum(test_profits)

        per_stock_summary.append({
            "Code": code,
            "TotalTestProfit": total_profit,
            "AvgPerFold": avg_profit,
            "MinFoldProfit": min_profit,
            "FoldsPositive": f"{folds_positive}/{len(test_profits)}",
            "ModeTh": mode_th,
            "ThAgreement": f"{mode_count}/{len(ths)}",
        })

    # --- レポート出力 ---
    print("\n" + "=" * 72)
    print("【AI設定 (config.yaml)】")
    print(f"  -予測日数 (future_days)  : {ai.future_days}日後")
    print(f"  -目標利益 (target_percent): {ai.target_percent}%")
    print(f"  -現在の閾値 (threshold)  : {ai.threshold}")
    print("=" * 72)

    if per_stock_summary:
        print("\n-- 銘柄別サマリー (Walk-forward のテスト期間合計) --")
        df_sum = pd.DataFrame(per_stock_summary).sort_values(by="TotalTestProfit", ascending=False)
        print(df_sum.to_string(index=False))

        best = df_sum.iloc[0]
        print(
            f"\n>>> 推奨: 銘柄 {best['Code']} / threshold {best['ModeTh']} "
            f"→ テスト合計利益 {best['TotalTestProfit']:+d}円 "
            f"(勝ち fold {best['FoldsPositive']}, 閾値一致 {best['ThAgreement']})"
        )
        print(
            "   ※ val で閾値を選び test で評価したため、"
            "以前の単一分割バックテストより控えめな値が「本当の期待値」に近い。"
        )

    if all_fold_rows:
        print("\n-- 全 fold 詳細 --")
        cols = ["Code", "Fold", "BestTh", "ValProfit", "TestProfit", "TestWinRate", "TestTrades"]
        df_all = pd.DataFrame(all_fold_rows).sort_values(by=["Code", "Fold"])[cols]
        print(df_all.to_string(index=False))
    else:
        print("テスト結果がありませんでした。候補銘柄やデータ取得期間を確認してください。")


if __name__ == "__main__":
    run_backtest()
