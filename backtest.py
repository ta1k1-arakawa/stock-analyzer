# backtest.py
import pandas as pd
import lightgbm as lgb
import logging
import sys
import os
import yaml
import numpy as np
from dotenv import load_dotenv 

# 自作モジュール
from jquants_fetcher import JQuantsFetcher
from technical_analyzer import calculate_indicators
from prepare_target import create_target_variable

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtest')

def load_config():
    if not os.path.exists('config.yaml'):
        logger.error("config.yaml が見つかりません。")
        sys.exit(1)
        
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_backtest():
    load_dotenv()

    config = load_config()
    
    # 候補銘柄
    candidates = config.get('backtest_candidates')
    
    # 共通設定
    # configの構造が変わっても動くように安全に取得
    target_stock_conf = config.get('target_stock')
    ai_params = target_stock_conf.get('ai_params')
    # 必須パラメータがない場合のデフォルト値
    future_days = ai_params.get('future_days')
    target_percent = ai_params.get('target_percent')
    threshold = ai_params.get('threshold')
    
    tech_params = config.get('technical_analysis_params')
    feature_cols = target_stock_conf.get('feature_columns')
    
    # 期間
    train_settings = config.get('training_settings')
    DATA_FROM = train_settings.get('data_from')
    DATA_TO = train_settings.get('data_to')
    
    # J-Quants認証
    j_mail = os.getenv("JQUANTS_MAIL")
    j_pass = os.getenv("JQUANTS_PASSWORD")
    
    # 認証情報のデバッグ表示（パスワードは隠す）
    if not j_mail:
        logger.error("メールアドレスが設定されていません。config.yaml または .env を確認してください。")
        return
    else:
        print(f"J-Quants User: {j_mail}")

    fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)
    
    # IDトークン取得エラー時の処理
    if not fetcher.get_id_token():
        logger.error("J-Quants認証に失敗しました。メールアドレスとパスワードを確認してください。")
        return

    print(f"\n=== 銘柄選定バックテスト開始 ({len(candidates)}銘柄) ===")
    print(f"期間: {DATA_FROM} 〜 {DATA_TO}\n")

    results = []

    for code in candidates:
        print(f"Testing: {code}...", end=" ", flush=True)
        
        # データ取得
        try:
            df = fetcher.get_daily_stock_prices(str(code), DATA_FROM, DATA_TO)
        except Exception as e:
            print(f"Error: {e}")
            continue

        if df is None or df.empty:
            print("Skip (No Data)")
            continue
            
        # 数値変換
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 特徴量計算
        df_ta = calculate_indicators(df, tech_params)
        
        # ターゲット作成
        df_model = create_target_variable(df_ta, future_days, target_percent)
        
        # 必要な列が存在するかチェック
        missing_cols = [c for c in feature_cols if c not in df_model.columns]
        if missing_cols:
            print(f"Skip (特徴量不足: {missing_cols})")
            continue

        # 特徴量の欠損削除
        df_model = df_model.dropna(subset=feature_cols + ['Target'])
        
        if len(df_model) < 50: # データが少なすぎる場合はスキップ
            print(f"Skip (データ不足: {len(df_model)}件)")
            continue

        # 学習/テスト分割 (時系列で分割: 後半20%をテスト)
        split_idx = int(len(df_model) * 0.8)
        train_df = df_model.iloc[:split_idx]
        test_df = df_model.iloc[split_idx:]
        
        # 学習データに正解ラベルが1種類しかない場合は学習不可
        if len(train_df['Target'].unique()) < 2:
             print("Skip (学習データのラベルが単一)")
             continue

        # 学習
        # LightGBMの出力を抑制
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
        
        # 特徴量データのみ抽出
        X_train = train_df[feature_cols]
        y_train = train_df['Target']
        
        model.fit(X_train, y_train)
        
        # 予測 & シミュレーション
        X_test = test_df[feature_cols]
        # 確率を取得 (クラス1 = 買い の確率)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test) # predictで確率が返る場合

        # シミュレーション
        initial_budget = 300000 
        budget = initial_budget
        trade_count = 0
        wins = 0
        
        # テスト期間の日数（営業日）
        test_days_count = len(test_df)
        
        # シミュレーションループ
        for i in range(len(test_df) - 1):
            # AIの確信度が閾値を超えたら「翌日Open」で買う
            if probs[i] >= threshold:
                # 翌日のデータ参照
                next_day = test_df.iloc[i+1]
                entry_price = next_day['Open'] # 翌日始値
                exit_price = next_day['Close'] # 翌日終値 (デイトレ)
                
                # 予算内で買えるだけ買う（単利運用）
                if entry_price > 0:
                    qty = int(budget / entry_price)
                    if qty > 0:
                        profit = (exit_price - entry_price) * qty
                        budget += profit
                        trade_count += 1
                        if profit > 0: wins += 1
        
        # 評価指標の計算
        total_profit = budget - initial_budget
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        # 週平均取引回数 (1週間=5営業日換算)
        weeks = test_days_count / 5.0
        trades_per_week = trade_count / weeks if weeks > 0 else 0
        
        print(f"完了 -> 利益: {total_profit:+.0f}円, 週取引: {trades_per_week:.1f}回, 勝率: {win_rate:.1f}%")
        
        results.append({
            "Code": code,
            "Profit": int(total_profit),
            "Trades/Week": round(trades_per_week, 1),
            "WinRate": round(win_rate, 1),
            "TotalTrades": trade_count
        })

    # 結果発表
    print("\n" + "="*60)
    print("   選定結果ランキング (利益順)")
    print("="*60)
    
    if results:
        df_res = pd.DataFrame(results)
        # 利益順に並べ替え
        df_res = df_res.sort_values(by="Profit", ascending=False)
        print(df_res.to_string(index=False))
        
        # 条件に近いものを推薦 (週1.5回以上取引 & 利益プラス)
        good_candidates = df_res[ (df_res['Trades/Week'] >= 1.5) & (df_res['Profit'] > 0) ]
        
        print("\n★ おすすめ銘柄 (週1.5回以上取引 & 利益プラス):")
        if not good_candidates.empty:
            best = good_candidates.iloc[0]
            print(f"銘柄コード: {best['Code']}")
            print(f"利益: {best['Profit']}円, 勝率: {best['WinRate']}%, 週平均取引: {best['Trades/Week']}回")
            print(f"-> config.yaml の 'target_stock' > 'code' を '{best['Code']}' に変更して train.py を実行してください。")
        else:
            print("条件(週1.5回以上かつプラス)に完全一致する銘柄はありませんでした。")
            print("利益最優先ならリスト最上位の銘柄を選んでください。")
    else:
        print("テスト結果がありませんでした。候補銘柄やデータ取得期間を確認してください。")

if __name__ == "__main__":
    run_backtest()