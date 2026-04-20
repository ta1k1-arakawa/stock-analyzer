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


THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _simulate(test_df: pd.DataFrame, probs, threshold: float, future_days: int) -> tuple[int, int, int]:
    """threshold 条件でテスト期間をシミュレートし、(利益, 取引数, 勝数) を返す。"""
    initial_budget = 300_000
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


def run_backtest() -> None:
    """全候補銘柄×閾値グリッドでバックテストを実行し、最良組み合わせを表示する。"""
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

    print(f"\n=== 銘柄選定バックテスト開始 ({len(candidates)}銘柄 × {len(THRESHOLD_GRID)}閾値) ===")
    print(f"期間: {data_from} 〜 {data_to}")
    print(f"ラベル条件: {ai.future_days}日後に +{ai.target_percent}% 以上\n")

    all_results: list[dict] = []
    best_per_stock: list[dict] = []
    label_stats: list[dict] = []

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

        # ラベル分布（陽性率）
        pos_rate_train = train_df["Target"].mean() * 100
        pos_rate_test = test_df["Target"].mean() * 100
        label_stats.append({
            "Code": code,
            "Train+%": round(pos_rate_train, 1),
            "Test+%": round(pos_rate_test, 1),
            "TrainN": len(train_df),
            "TestN": len(test_df),
        })

        # 予測
        probs = (
            model.predict_proba(test_df[feature_cols])[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(test_df[feature_cols])
        )
        prob_max = float(probs.max()) if len(probs) else 0.0
        prob_mean = float(probs.mean()) if len(probs) else 0.0

        # 閾値スイープ
        weeks = len(test_df) / 5.0
        stock_rows: list[dict] = []
        for th in THRESHOLD_GRID:
            profit, trades, wins = _simulate(test_df, probs, th, ai.future_days)
            win_rate = (wins / trades * 100) if trades > 0 else 0
            trades_per_week = trades / weeks if weeks > 0 else 0
            row = {
                "Code": code,
                "Threshold": th,
                "Profit": profit,
                "Trades/Week": round(trades_per_week, 1),
                "WinRate": round(win_rate, 1),
                "TotalTrades": trades,
            }
            stock_rows.append(row)
            all_results.append(row)

        # この銘柄のベスト（取引0は除外）
        traded = [r for r in stock_rows if r["TotalTrades"] > 0]
        if traded:
            best = max(traded, key=lambda r: r["Profit"])
            print(
                f"完了 -> 確率 max={prob_max:.2f}/mean={prob_mean:.2f} | "
                f"best th={best['Threshold']}: 利益{best['Profit']:+d}円, "
                f"勝率{best['WinRate']}%, 週{best['Trades/Week']}回"
            )
            best_per_stock.append(best)
        else:
            print(f"完了 -> 確率 max={prob_max:.2f}/mean={prob_mean:.2f} | 全閾値で取引ゼロ")

    # --- レポート出力 ---
    print("\n" + "=" * 72)
    print("【AI設定 (config.yaml)】")
    print(f"  -予測日数 (future_days)  : {ai.future_days}日後")
    print(f"  -目標利益 (target_percent): {ai.target_percent}%")
    print(f"  -現在の閾値 (threshold)  : {ai.threshold}")
    print("=" * 72)

    if label_stats:
        print("\n-- ラベル分布 (Target=1 の比率) --")
        print(pd.DataFrame(label_stats).to_string(index=False))

    if best_per_stock:
        print("\n-- 銘柄別ベスト閾値 (利益最大) --")
        df_best = pd.DataFrame(best_per_stock).sort_values(by="Profit", ascending=False)
        print(df_best.to_string(index=False))

        top = df_best.iloc[0]
        print(
            f"\n>>> 推奨: 銘柄 {top['Code']} / threshold {top['Threshold']} "
            f"→ 利益 {top['Profit']:+d}円, 勝率 {top['WinRate']}%"
        )

    if all_results:
        print("\n-- 全 (銘柄×閾値) 結果 --")
        df_all = pd.DataFrame(all_results).sort_values(
            by=["Code", "Threshold"]
        )
        print(df_all.to_string(index=False))
    else:
        print("テスト結果がありませんでした。候補銘柄やデータ取得期間を確認してください。")


if __name__ == "__main__":
    run_backtest()