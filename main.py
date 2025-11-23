# main.py (AI予測・本番運用版)

import sys
import os
import yaml
import logging
import pandas as pd
import time
import joblib 
from datetime import datetime, timedelta

# 自作モジュールのインポート
from config_loader import load_config_and_logger 
# from jquants_fetcher import JQuantsFetcher # JQuantsを使う場合はこちら
from yfinance_fetcher import YFinanceFetcher 
from technical_analyzer import calculate_indicators
from line_notifier import LineNotifier

if __name__ == "__main__":
    logger, config = load_config_and_logger(log_file_name='app_main.log')

    if not logger or not config:
        print("設定読み込み失敗。終了します。")
        exit()
        
    logger.info("=== 株価分析・通知システム (AI本番運用版) 起動 ===")

    # 設定の読み込み 
    target_stock_conf = config.get('target_stock', {})
    tech_params = config.get('technical_analysis_params', {})
    training_settings = config.get('training_settings', {})
    
    stock_code = target_stock_conf.get('code')
    stock_name = target_stock_conf.get('name', stock_code)
    feature_columns = target_stock_conf.get('feature_columns', [])
    ai_params = target_stock_conf.get('ai_params', {})
    
    # AI判断基準
    ai_threshold = ai_params.get('threshold', 0.55)
    stop_loss_percent = ai_params.get('stop_loss_percent', 5.0) 

    model_path = training_settings.get('model_save_path', 'stock_ai_model.pkl')

    if not stock_code:
        logger.critical("config.yaml に 監視対象銘柄(target_stock.code) が設定されていません。")
        exit()

    # API初期化 (YFinanceFetcher)
    # 認証不要で初期化
    fetcher = YFinanceFetcher()
    
    # LINE通知の初期化
    line_creds = config.get('api_credentials', {}).get('line', {})
    line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    line_user_id = os.getenv("LINE_USER_ID")
    
    line_notifier = None
    if line_token and line_user_id:
        line_notifier = LineNotifier(channel_access_token=line_token)
    else:
        logger.warning("LINE設定が見つからないため、通知は行われません。")
    
    # AIモデルの読み込み
    ai_model = None
    try:
        if os.path.exists(model_path):
            ai_model = joblib.load(model_path)
            logger.info(f"AIモデル {model_path} を読み込みました。")
        else:
            logger.critical(f"モデルファイル {model_path} が見つかりません。先に train.py を実行してください。")
            exit()
    except Exception as e:
        logger.critical(f"AIモデル読み込みエラー: {e}")
        exit()

    # 本処理開始
    logger.info(f"--- 処理開始: {stock_name} ({stock_code}) ---")

    # データ取得 
    today = datetime.now()
    start_date = today - timedelta(days=150) 
    data_from_str = start_date.strftime('%Y-%m-%d')
    data_to_str = today.strftime('%Y-%m-%d')
    
    logger.info(f"データ取得期間: {data_from_str} 〜 {data_to_str}")
    
    # YFinanceFetcherで取得
    df_prices = fetcher.get_daily_stock_prices(stock_code, data_from_str, data_to_str)
    
    if df_prices is not None and not df_prices.empty:
        # 数値型変換 
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df_prices.columns:
                df_prices[c] = pd.to_numeric(df_prices[c], errors='coerce')

        # 特徴量計算
        df_with_indicators = calculate_indicators(df_prices, tech_params)
        
        # 最新行の取得
        latest_data = df_with_indicators.iloc[-1:]
        
        # 欠損チェック
        if latest_data[feature_columns].isnull().any().any():
            logger.warning("最新データに必要な特徴量が計算できていません（データ不足の可能性）。")
        else:
            # AI予測
            try:
                # 買いの確率を取得
                prediction_proba = ai_model.predict_proba(latest_data[feature_columns])
                buy_prob = prediction_proba[0][1]
                
                logger.info(f"AI予測結果: 買い確率 {buy_prob:.2%} (閾値: {ai_threshold:.0%})")

                # 最新データの情報
                latest_close = latest_data['Close'].iloc[0]
                latest_date_str = latest_data.index[-1].strftime('%Y-%m-%d')
                
                # 判定と通知
                msg = ""
                
                # 閾値を超えたら買いシグナル
                if buy_prob >= ai_threshold:
                    signal = "BUY"
                    stop_loss_price = latest_close * (1 - stop_loss_percent / 100)
                    
                    msg = (
                        f"【AI買いシグナル】\n"
                        f"銘柄: {stock_name} ({stock_code})\n"
                        f"確信度: {buy_prob:.1%}\n"
                        f"基準: {ai_threshold:.0%}\n"
                        f"現在値: {latest_close:.0f}円\n"
                        f"日時: {latest_date_str}\n"
                        f"⚠️ 損切り目安: {stop_loss_price:.0f}円 (-{stop_loss_percent}%)"
                    )
                    logger.info("判定: BUY (通知送信)")

                else:
                    # テストプレイ用
                    signal = "HOLD"
                    msg = (
                        f"【AI予測 (様子見)】\n"
                        f"銘柄: {stock_name} ({stock_code})\n"
                        f"確信度: {buy_prob:.1%} (基準未満)\n"
                        f"基準: {ai_threshold:.0%}\n"
                        f"現在値: {latest_close:.0f}円\n"
                        f"日時: {latest_date_str}\n"
                        f"判定: 今回は見送ります。"
                    )
                    logger.info("判定: HOLD (テスト通知送信)")

                # LINE送信 (BUYでもHOLDでも送る)
                if line_notifier:
                    line_notifier.send_push_message(line_user_id, msg)
                    logger.info("LINE通知を送信しました。")
                else:
                    logger.info("LINE設定がないため通知はスキップしました。")

            except Exception as e:
                logger.error(f"予測プロセス中にエラーが発生: {e}")
    else:
        logger.warning("株価データが取得できませんでした。")

    logger.info("=== 全処理完了 ===")