# main.py (AI予測専用・自動日付計算版)

import sys
import os
import yaml
import logging
import pandas as pd
import time
import joblib 
from datetime import datetime, timedelta # ★日付計算用に必須

# 自作モジュールのインポート
from config_loader import load_config_and_logger 
from jquants_fetcher import JQuantsFetcher
from technical_analyzer import calculate_indicators
from line_notifier import LineNotifier

if __name__ == "__main__":
    logger, config = load_config_and_logger(log_file_name='app_main.log')

    if not logger or not config:
        print("設定ファイルの読み込みに失敗したため、main.py を終了します。")
        exit()
        
    logger.info("=== 株価分析・通知システム (AI版) 起動 ===")

    # 共通設定の読み込み 
    common_config = config.get('common_feature_settings', {})
    feature_columns = common_config.get('feature_columns')
    technical_params = common_config.get('technical_analysis_params')

    if not feature_columns or not technical_params:
        logger.critical("config.yaml に common_feature_settings が設定されていません。")
        exit()

    # JQuantsFetcher初期化
    jquants_creds = config.get('api_credentials', {}).get('jquants', {})
    j_mail = jquants_creds.get('mail_address')
    j_pass = jquants_creds.get('password')
    
    # 環境変数等から取得するロジックを含む JQuantsFetcher の利用
    fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)

    if not fetcher.get_id_token():
        logger.critical("J-Quants認証失敗。処理を終了します。")
        exit()
    logger.info(f"JQuants認証成功。")
    
    # LineNotifier初期化
    line_creds = config.get('api_credentials', {}).get('line', {})
    line_token = line_creds.get('channel_access_token')
    line_user_id = line_creds.get('user_id')
    
    line_notifier = None
    if line_token and line_user_id:
        line_notifier = LineNotifier(channel_access_token=line_token)
    
    # AIモデルの読み込み
    ai_settings = config.get('ai_prediction_settings', {})
    ai_model = None
    
    if ai_settings.get('enabled', False):
        model_path = ai_settings.get('model_load_path')
        try:
            ai_model = joblib.load(model_path)
            logger.info(f"AIモデル {model_path} を読み込みました。")
        except Exception as e:
            logger.critical(f"AIモデルの読み込みに失敗: {e}")
            exit()
    
    if ai_model is None:
        logger.critical("AIモデルが無効か読み込めませんでした。終了します。")
        exit()
            
    # 銘柄ごとの処理ループ
    stocks_to_monitor = config.get('stocks_to_monitor', [])

    for stock_info in stocks_to_monitor:
        if not stock_info.get('enabled', False):
            continue

        stock_code = stock_info.get('code')
        stock_name = stock_info.get('name', stock_code)
        logger.info(f"--- 処理開始: {stock_name} ({stock_code}) ---")

        # テクニカル指標計算のために、今日から過去120日分遡って取得する
        today = datetime.now()
        past_days_needed = 120 
        start_date = today - timedelta(days=past_days_needed)
        
        data_from_str = start_date.strftime('%Y-%m-%d')
        data_to_str = today.strftime('%Y-%m-%d')
        
        logger.info(f"データ取得期間(自動計算): {data_from_str} 〜 {data_to_str}")
        
        # データ取得
        df_prices = fetcher.get_daily_stock_prices(stock_code, data_from_str, data_to_str)
        
        # テクニカル指標の計算
        df_with_indicators = None
        if df_prices is not None and not df_prices.empty:
            df_with_indicators = calculate_indicators(df_prices, technical_params)
        else:
            logger.warning(f"銘柄 {stock_name}: データ取得失敗。")
            continue

        # AI判定
        signal = "HOLD" 
        signal_details = {"message": "判定スキップ"}
        
        if df_with_indicators is not None and not df_with_indicators.empty:
            # 最新行を取得
            latest_data = df_with_indicators.iloc[-1:] 
            
            # 必要な特徴量が全て計算できているかチェック (NaNがないか)
            if latest_data[feature_columns].isnull().any().any():
                logger.warning("最新データに必要な特徴量が不足(NaN)しています。期間不足の可能性があります。")
                signal_details = {"message": "データ不足(NaN)"}
            else:
                try:
                    prediction_proba = ai_model.predict_proba(latest_data[feature_columns])
                    buy_prob = prediction_proba[0][1] 
                    
                    ai_threshold = stock_info.get('ai_threshold', 0.75)
                    
                    if buy_prob >= ai_threshold:
                        signal = "BUY"
                        signal_details = {
                            "rule_applied": f"AI予測 (確率 {buy_prob:.1%})",
                            "message": f"AI確信度: {buy_prob:.1%} (閾値 {ai_threshold:.0%})",
                            "Close": f"{latest_data['Close'].iloc[0]:.0f}"
                        }
                    else:
                        signal = "HOLD"
                        signal_details = {
                            "rule_applied": "AI予測 (HOLD)",
                            "message": f"AI確信度: {buy_prob:.1%} (閾値未満)"
                        }
                except Exception as e:
                    logger.error(f"AI予測エラー: {e}")
                    signal_details = {"message": "予測エラー"}

        # LINE通知
        notification_settings = stock_info.get('notification_settings', {})
        should_notify = False
        
        if signal == "BUY" and notification_settings.get('on_buy_signal', False):
            should_notify = True
        elif signal == "HOLD" and notification_settings.get('on_hold_signal', False):
            should_notify = True
        
        if should_notify and line_notifier:
            msg = f"【{signal} (AI)】{stock_name}\n"
            if 'message' in signal_details: msg += f"{signal_details['message']}\n"
            
            # 参考情報
            if df_with_indicators is not None:
                last_row = df_with_indicators.iloc[-1]
                date_str = last_row.name.strftime('%m/%d')
                msg += f"日付: {date_str}\n終値: {last_row['Close']:.0f}円"

            line_notifier.send_push_message(line_user_id, msg)
            logger.info(f"LINE通知送信: {signal}")
        else:
            logger.info(f"通知対象外または設定オフ: {signal}")

        time.sleep(1) 

    logger.info("=== 全処理完了 ===")