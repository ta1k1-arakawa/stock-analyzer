# main.py (AI予測専用・config_loader使用)

import sys
import os
import yaml
import logging
import pandas as pd
import time
import joblib 

#　自作モジュールのインポート
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

    #　JQuantsFetcher初期化
    jquants_creds = config.get('api_credentials', {}).get('jquants', {})
    j_mail = jquants_creds.get('mail_address')
    j_pass = jquants_creds.get('password')
    
    fetcher = None
    if j_mail and j_pass:
        fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)
    else:
        logger.critical("J-Quants APIの認証情報が設定されていません。処理を終了します。")
        exit()

    if not fetcher.get_id_token():
        logger.critical("J-Quants認証失敗。処理を終了します。")
        exit()
    logger.info(f"JQuantsFetcher経由でIDトークンを正常に取得/確認しました。")
    
    #　LineNotifier初期化
    line_creds = config.get('api_credentials', {}).get('line', {})
    line_channel_access_token = line_creds.get('channel_access_token')
    line_user_id_to_notify = line_creds.get('user_id')
    
    line_notifier_instance = None
    if line_channel_access_token and line_user_id_to_notify:
        try:
            line_notifier_instance = LineNotifier(channel_access_token=line_channel_access_token)
            logger.info("LineNotifierのインスタンス化に成功しました。")
        except Exception as e:
            logger.error(f"LineNotifierの初期化中にエラー: {e}")
    else:
        logger.warning("LINEの認証情報が不足しています。")


    #　AIモデルの読み込み
    ai_settings = config.get('ai_prediction_settings', {})
    ai_model = None
    feature_columns = [] 
    
    if ai_settings.get('enabled', False):
        model_path = ai_settings.get('model_load_path')
        try:
            ai_model = joblib.load(model_path)
            feature_columns = ai_settings.get('feature_columns', [])
            if not feature_columns:
                 logger.error("configに 'feature_columns' が未設定です。AI予測ができません。")
                 ai_model = None 
            else:
                 logger.info(f"AIモデル {model_path} と特徴量リスト({len(feature_columns)}個)を読み込みました。")
        except FileNotFoundError:
            logger.error(f"AIモデルファイル {model_path} が見つかりません。")
        except Exception as e:
            logger.error(f"AIモデルの読み込みに失敗: {e}")
            
    if ai_model is None:
        logger.critical("AIモデルが利用できません。処理を終了します。")
        exit()
            
    stocks_to_monitor = config.get('stocks_to_monitor', [])

    for stock_info in stocks_to_monitor:
        if not stock_info.get('enabled', False):
            logger.info(f"銘柄 {stock_info.get('code', 'N/A')} は無効のためスキップします。")
            continue

        stock_code = stock_info.get('code')
        stock_name = stock_info.get('name', stock_code)
        logger.info(f"--- 処理開始: {stock_name} ({stock_code}) ---")

        # データ取得
        jquants_settings = config.get('jquants_api_settings', {})
        data_from_str = "2024-01-01" 
        data_to_str = "2024-12-31"   
        logger.info(f"データ取得期間: {data_from_str} から {data_to_str}")
        
        df_prices = fetcher.get_daily_stock_prices(stock_code, data_from_str, data_to_str)
        
        # テクニカル指標の計算
        df_with_indicators = None
        if df_prices is not None and not df_prices.empty:
            logger.info(f"銘柄 {stock_name}: {len(df_prices)} 件の株価データを取得。")
            technical_params = ai_settings.get('technical_analysis_params_for_prediction')
            if not technical_params:
                logger.error(f"AI予測用の 'technical_analysis_params_for_prediction' がconfig未設定。")
                continue
            
            logger.info(f"銘柄 {stock_name}: AI予測用のテクニカル指標計算開始。")
            df_with_indicators = calculate_indicators(df_prices, technical_params)
            logger.info(f"銘柄 {stock_name}: テクニカル指標計算完了。")
        else:
            logger.warning(f"銘柄 {stock_name}: 株価データ取得失敗または空。分析スキップ。")
            continue

        #　AIによるシグナル判定
        signal = "HOLD" 
        signal_details = {"message": "判定スキップ"}
        
        if df_with_indicators is None or df_with_indicators.empty:
            logger.warning(f"銘柄 {stock_name}: 分析データがないためAI判定をスキップ。")
            continue

        logger.info(f"銘柄 {stock_name}: AI予測モードで判定します。")
        latest_data = df_with_indicators.iloc[-1:] 
        
        if latest_data.empty or latest_data[feature_columns].isnull().any().any():
            logger.warning(f"最新データに特徴量がないかNaNが含まれるため、AI予測をスキップします。")
            signal_details = {"message": "最新データ不備(NaN等)"}
        else:
            try:
                prediction_proba = ai_model.predict_proba(latest_data[feature_columns])
                buy_probability = prediction_proba[0][1] 
                ai_threshold = stock_info.get('ai_threshold', 0.75)
                
                if buy_probability >= ai_threshold:
                    signal = "BUY"
                    signal_details = {
                        "rule_applied": f"AI予測 (確率 {buy_probability:.2%})",
                        "message": f"AIが {buy_probability:.2%} の確率で「買い」と予測 (閾値: {ai_threshold:.0%})",
                        "RSI_14": round(latest_data['RSI_14'].iloc[0], 2), # 例
                    }
                else:
                    signal = "HOLD"
                    signal_details = {
                        "rule_applied": "AI予測 (HOLD)",
                        "message": f"AIの買い確率は {buy_probability:.2%} で、閾値未満です。"
                    }
            except KeyError as e:
                logger.error(f"AI予測エラー: モデルが学習した特徴量 {e} がデータにありません。")
                signal_details = {"message": "AI予測エラー(特徴量不一致)"}
            except Exception as e:
                logger.error(f"AI予測中に予期せぬエラーが発生: {e}")
                signal_details = {"message": "AI予測エラー"}

        # LINE通知
        notification_settings = stock_info.get('notification_settings', {})
        should_notify = False
        
        if signal == "BUY" and notification_settings.get('on_buy_signal', False):
            should_notify = True
        elif signal == "SELL" and notification_settings.get('on_sell_signal', False):
            should_notify = True
        elif signal == "HOLD" and notification_settings.get('on_hold_signal', False):
            should_notify = True
        
        if should_notify:
            logger.info(f"銘柄 {stock_name}: シグナル '{signal}' のため、通知処理に進みます。")
            message_to_send = f"【{signal}シグナル検出(AI)】\n"
            message_to_send += f"銘柄: {stock_name} ({stock_code})\n"
            if 'rule_applied' in signal_details: message_to_send += f"判定: {signal_details['rule_applied']}\n"
            if 'message' in signal_details: message_to_send += f"詳細: {signal_details['message']}\n"

            details_for_message = {k: v for k, v in signal_details.items() if k not in ['rule_applied', 'message']}
            if details_for_message:
                message_to_send += "--- 参考指標 ---\n"
                for key, value in details_for_message.items():
                    message_to_send += f"{key}: {value}\n"
            
            if isinstance(df_with_indicators.index, pd.DatetimeIndex) and not df_with_indicators.empty:
                 latest_date_str = df_with_indicators.index[-1].strftime('%Y-%m-%d')
                 latest_close = df_with_indicators.iloc[-1]['Close'] 
                 message_to_send += f"最終データ: {latest_date_str} 終値: {latest_close:.2f}"
            
            if line_notifier_instance and line_user_id_to_notify:
                send_success = line_notifier_instance.send_push_message(line_user_id_to_notify, message_to_send.strip())
                if send_success: logger.info(f"銘柄 {stock_name}: LINE通知を送信しました。")
                else: logger.error(f"銘柄 {stock_name}: LINE通知の送信に失敗しました。")
            else:
                logger.warning(f"銘柄 {stock_name}: LineNotifier未初期化またはユーザーID未設定のため通知スキップ。")
        else:
             logger.info(f"銘柄 {stock_name}: シグナル '{signal}' は通知対象外(HOLD含む)です。")

        logger.info(f"--- 処理終了: {stock_name} ({stock_code}) ---")
        time.sleep(1) 

    logger.info("=== 全ての銘柄の処理が完了しました ===")