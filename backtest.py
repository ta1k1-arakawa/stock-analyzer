# backtest.py — 銘柄選定バックテスト エントリポイント

from __future__ import annotations

import logging
from datetime import datetime

import lightgbm as lgb
import pandas as pd

from src import LOGGER_NAME
from src.analysis import calculate_indicators, create_target_variable, sanitize_ohlcv
from src.config import load_app
from src.fetchers.yfinance import YFinanceFetcher


def run_backtest() -> None:
    """全候補銘柄に対してバックテストを実行し、ランキングを表示する。"""
    config, _ = load_app(log_file="backtest.log")
    logger = logging.getLogger(LOGGER_NAME)

    candidates = config.backtest_candidates
    ai = config.ai_params
    feature_cols = config.feature_columns

    # 期間
    data_from = config.training_settings.get("data_from")
    data_to = config.training_settings.get("data_to")
    if data_to == "auto":
        data_to = datetime.now().strftime("%Y-%m-%d")
        print(f"終了日を自動設定: {data_to}")

    fetcher = YFinanceFetcher()

    print(f"\n=== 銘柄選定バックテスト開始 ({len(candidates)}銘柄) ===")
    print(f"期間: {data_from} 〜 {data_to}\n")

    results: list[dict] = []

    for code in candidates:
        print(f"Testing: {code}...", end=" ", flush=True)

        try:
            df = fetcher.get_daily_stock_prices(str(code), data_from, data_to)
        except Exception as e:
            print(f"Error: {e}")
            continue

        if df is None or df.empty:
            print("Skip (No Data)")
            continue

        df = sanitize_ohlcv(df)

        # 特徴量計算 → ターゲット作成
        df_ta = calculate_indicators(df, config.tech_params)
        df_model = create_target_variable(df_ta, ai.future_days, ai.target_percent)

        # 必要な列チェック
        missing = [c for c in feature_cols if c not in df_model.columns]
        if missing:
            print(f"Skip (特徴量不足: {missing})")
            continue

        df_model = df_model.dropna(subset=feature_cols + ["Target"])

        if len(df_model) < 50:
            print(f"Skip (データ不足: {len(df_model)}件)")
            continue

        # 時系列分割 (後半20%をテスト)
        split_idx = int(len(df_model) * 0.8)
        train_df = df_model.iloc[:split_idx]
        test_df = df_model.iloc[split_idx:]

        if len(train_df["Target"].unique()) < 2:
            print("Skip (学習データのラベルが単一)")
            continue

        # 学習
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
        model.fit(train_df[feature_cols], train_df["Target"])

        # 予測 & シミュレーション
        probs = (
            model.predict_proba(test_df[feature_cols])[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(test_df[feature_cols])
        )

        initial_budget = 300_000
        budget = initial_budget
        trade_count = 0
        wins = 0

        for i in range(len(test_df) - 1):
            if probs[i] >= ai.threshold:
                entry_idx = i + 1
                if entry_idx >= len(test_df):
                    break

                entry_price = test_df.iloc[entry_idx]["Open"]
                exit_idx = entry_idx + (ai.future_days - 1)

                if exit_idx < len(test_df):
                    exit_price = test_df.iloc[exit_idx]["Close"]
                else:
                    continue

                if entry_price > 0:
                    qty = int(budget / entry_price)
                    if qty > 0:
                        profit = (exit_price - entry_price) * qty
                        budget += profit
                        trade_count += 1
                        if profit > 0:
                            wins += 1

        # 評価指標
        total_profit = budget - initial_budget
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        weeks = len(test_df) / 5.0
        trades_per_week = trade_count / weeks if weeks > 0 else 0

        print(f"完了 -> 利益: {total_profit:+.0f}円, 週取引: {trades_per_week:.1f}回, 勝率: {win_rate:.1f}%")

        results.append({
            "Code": code,
            "Profit": int(total_profit),
            "Trades/Week": round(trades_per_week, 1),
            "WinRate": round(win_rate, 1),
            "TotalTrades": trade_count,
        })

    # ランキング表示
    print("\n" + "-" * 60)
    print("   選定結果ランキング (利益順)")
    print("-" * 60)
    print("【AI設定 (config.yaml)】")
    print(f"  • 予測日数 (future_days) : {ai.future_days}日後")
    print(f"  • 目標利益 (target_percent): {ai.target_percent}%")
    print(f"  • 買い閾値 (threshold)     : {ai.threshold}")
    print("-" * 60)

    if results:
        df_res = pd.DataFrame(results).sort_values(by="Profit", ascending=False)
        print(df_res.to_string(index=False))
    else:
        print("テスト結果がありませんでした。候補銘柄やデータ取得期間を確認してください。")


if __name__ == "__main__":
    run_backtest()