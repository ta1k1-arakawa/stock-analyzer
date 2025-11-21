# main.py (AI予測専用バージョン)

import sys
import os
import yaml
from dotenv import load_dotenv
import logging
import pandas as pd
import time
import joblib # ★AIモデル読み込みのため必須

# --- 自作モジュールのインポート ---
from logger_setup import setup_logger, APP_LOGGER_NAME
from jquants_fetcher import JQuantsFetcher
from technical_analyzer import calculate_indicators
# from signal_evaluator import evaluate_signals # ← ルールベースなので不要
from line_notifier import LineNotifier

# --- グローバルな準備 (既存のコード) ---
load_dotenv_success = load_dotenv()
CONFIG_FILE_PATH = 'config.yaml'
# ... (ロガー設定のコードはそのまま) ...
logger = setup_logger(log_level_str='INFO') 
# ... (config読み込みのコードはそのまま) ...
# ... (環境変数の解決コードはそのまま) ...
logger.info("設定ファイルの読み込みと環境変数の解決が完了しました。")


if __name__ == "__main__":
    logger.info("=== 株価分析・通知システム (AI版) 起動 ===")

    # --- JQuantsFetcher初期化 (既存のコード) ---
    jquants_creds = config.get('api_credentials', {}).get('jquants', {})
    # ... (fetcher の初期化ロジックはそのまま) ...
    fetcher = JQuantsFetcher(mail_address=jquants_creds.get('mail_address'), 
                             password=jquants_creds.get('password'))
    if not fetcher.get_id_token():
        logger.critical("J-Quants認証失敗。処理を終了します。")
        exit()
    logger.info(f"JQuantsFetcher経由でIDトークンを正常に取得/確認しました。")
    
    # --- LineNotifier初期化 (既存のコード) ---
    line_creds = config.get('api_credentials', {}).get('line', {})
    line_channel_access_token = line_creds.get('channel_access_token')
    line_user_id_to_notify = line_creds.get('user_id')
    # ... (line_notifier_instance の初期化ロジックはそのまま) ...
    line_notifier_instance = LineNotifier(channel_access_token=line_channel_access_token)
    logger.info("LineNotifierのインスタンス化に成功しました。")

    # --- ★AIモデルの読み込み (必須) ---
    ai_settings = config.get('ai_prediction_settings', {})
    ai_model = None
    feature_columns = [] # AIが学習した特徴量リスト
    
    if ai_settings.get('enabled', False):
        model_path = ai_settings.get('model_load_path', 'stock_ai_model.pkl')
        try:
            ai_model = joblib.load(model_path)
            feature_columns = ai_settings.get('feature_columns', [])
            if not feature_columns:
                 logger.error("configに 'feature_columns' が未設定です。AI予測ができません。")
                 ai_model = None # 特徴量が不明ならモデルを使えない
            else:
                 logger.info(f"AIモデル {model_path} と特徴量リスト({len(feature_columns)}個)を読み込みました。")
        except FileNotFoundError:
            logger.error(f"AIモデルファイル {model_path} が見つかりません。")
        except Exception as e:
            logger.error(f"AIモデルの読み込みに失敗: {e}")
            
    if ai_model is None:
        logger.critical("AIモデルが利用できません。処理を終了します。")
        exit()
            
    # --- メインループ (AI予測専用) ---
    stocks_to_monitor = config.get('stocks_to_monitor', [])
    if not stocks_to_monitor:
        logger.warning("監視対象の銘柄がconfig.yamlに設定されていません。")

    for stock_info in stocks_to_monitor:
        if not stock_info.get('enabled', False):
            logger.info(f"銘柄 {stock_info.get('code', 'N/A')} は無効のためスキップします。")
            continue

        stock_code = stock_info.get('code')
        stock_name = stock_info.get('name', stock_code)
        logger.info(f"--- 処理開始: {stock_name} ({stock_code}) ---")

        # --- 1. データ取得 (既存のコード) ---
        jquants_settings = config.get('jquants_api_settings', {})
        data_range_days = jquants_settings.get('data_range_days', 90) # AI予測に必要な最低限の日数を取得
        
        # J-Quants APIのデータカバレッジに合わせた日付を指定 (ここは既存のロジックを使用)
        data_from_str = "2025-02-01" # J-Quantsのデータカバレッジ内の過去日付
        data_to_str = "2025-02-16"   # J-Quantsのデータカバレッジ内の過去日付
        logger.info(f"データ取得期間: {data_from_str} から {data_to_str}")
        
        df_prices = fetcher.get_daily_stock_prices(stock_code, data_from_str, data_to_str)
        
        # --- 2. テクニカル指標の計算 (既存のコード) ---
        df_with_indicators = None
        if df_prices is not None and not df_prices.empty:
            logger.info(f"銘柄 {stock_name}: {len(df_prices)} 件の株価データを取得。")
            
            # AIが学習した特徴量を計算するため、train.pyと「同じ」設定を使う
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

        # --- 3. AIによるシグナル判定 ---
        signal = "HOLD" 
        signal_details = {"message": "判定スキップ"}
        
        if df_with_indicators is None or df_with_indicators.empty:
            logger.warning(f"銘柄 {stock_name}: 分析データがないためAI判定をスキップ。")
            continue

        logger.info(f"銘柄 {stock_name}: AI予測モードで判定します。")
        
        # 予測に使う最新のデータ（一番最後の行）
        latest_data = df_with_indicators.iloc[-1:] 
        
        if latest_data.empty or latest_data[feature_columns].isnull().any().any():
            logger.warning(f"最新データに特徴量がないかNaNが含まれるため、AI予測をスキップします。")
            signal_details = {"message": "最新データ不備(NaN等)"}
        else:
            try:
                # AIに予測させる (確率を予測)
                # [0] = クラス0 (上がらない) の確率, [1] = クラス1 (上がる) の確率
                prediction_proba = ai_model.predict_proba(latest_data[feature_columns])
                buy_probability = prediction_proba[0][1] # 「上がる」確率
                
                # configで「買い」と判断する確率の閾値(しきいち)を指定
                ai_threshold = stock_info.get('ai_threshold', 0.75) # 75%以上ならBUYなど
                
                if buy_probability >= ai_threshold:
                    signal = "BUY"
                    signal_details = {
                        "rule_applied": f"AI予測 (確率 {buy_probability:.2%})",
                        "message": f"AIが {buy_probability:.2%} の確率で「買い」と予測 (閾値: {ai_threshold:.0%})",
                        # (通知用に参考指標を追加)
                        "RSI_14": round(latest_data['RSI_14'].iloc[0], 2),
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
                logger.exception("スタックトレース:")
                signal_details = {"message": "AI予測エラー"}

        # --- 4. LINE通知 (既存のコードをほぼ流用) ---
        notification_settings = stock_info.get('notification_settings', {})
        should_notify = False
        
        if signal == "BUY" and notification_settings.get('on_buy_signal', False):
            should_notify = True
        elif signal == "SELL" and notification_settings.get('on_sell_signal', False):
            should_notify = True
        elif signal == "HOLD" and notification_settings.get('on_hold_signal', False):
            logger.info(f"銘柄 {stock_name}: HOLDシグナルですが、テストのため通知対象とします (on_hold_signal: true)。")
            should_notify = True
        
        if should_notify:
            logger.info(f"銘柄 {stock_name}: シグナル '{signal}' のため、通知処理に進みます。")
            
            message_to_send = f"【{signal}シグナル検出(AI)】\n" # AI版と明記
            message_to_send += f"銘柄: {stock_name} ({stock_code})\n"
            if 'rule_applied' in signal_details:
                message_to_send += f"判定: {signal_details['rule_applied']}\n"
            if 'message' in signal_details:
                message_to_send += f"詳細: {signal_details['message']}\n"

            details_for_message = {}
            for key, value in signal_details.items():
               if key not in ['rule_applied', 'message']:
                   details_for_message[key] = value
            
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
                if send_success:
                    logger.info(f"銘柄 {stock_name}: LINE通知を送信しました。")
                else:
                    logger.error(f"銘柄 {stock_name}: LINE通知の送信に失敗しました。")
            else:
                logger.warning(f"銘柄 {stock_name}: LineNotifier未初期化またはユーザーID未設定のため通知スキップ。")
        else:
             logger.info(f"銘柄 {stock_name}: シグナル '{signal}' は通知対象外(HOLD含む)です。")

        logger.info(f"--- 処理終了: {stock_name} ({stock_code}) ---")
        time.sleep(1) # API負荷軽減

    logger.info("=== 全ての銘柄の処理が完了しました ===")