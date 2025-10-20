# main.py

import sys
import os
import yaml
from dotenv import load_dotenv
import logging
import pandas as pd
from datetime import date, timedelta
import time

# --- 自作モジュールのインポート ---
from logger_setup import setup_logger, APP_LOGGER_NAME
from jquants_fetcher import JQuantsFetcher
from technical_analyzer import calculate_indicators
from signal_evaluator import evaluate_signals
from line_notifier import LineNotifier

# --- グローバルな準備 ---
load_dotenv_success = load_dotenv()
CONFIG_FILE_PATH = 'config.yaml'

temp_log_settings = {'log_file': 'app_main.log', 'log_level': 'INFO'}
try:
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f_cfg_for_log:
            _config_for_log = yaml.safe_load(f_cfg_for_log)
            if _config_for_log and 'logging_settings' in _config_for_log:
                temp_log_settings = _config_for_log['logging_settings']
    else:
        print(f"警告: 設定ファイル {CONFIG_FILE_PATH} が見つかりません。デフォルトのログ設定を使用します。", file=sys.stderr)
except Exception as e:
    print(f"警告: 設定ファイルからログ設定の読み込み中にエラー: {e}。デフォルトのログ設定を使用します。", file=sys.stderr)

logger = setup_logger(
    log_file_path=temp_log_settings.get('log_file', 'app_main.log'),
    log_level_str=temp_log_settings.get('log_level', 'INFO'),
    logger_name=APP_LOGGER_NAME
)

if load_dotenv_success:
    logger.info(".env ファイルが正常に読み込まれました。")
else:
    logger.warning(".env ファイルが見つからないか、読み込めませんでした。環境変数が直接設定されていることを期待します。")

config = {}
try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f_cfg:
        config_from_yaml = yaml.safe_load(f_cfg)
        if config_from_yaml is None: config_from_yaml = {}
    logger.info(f"設定ファイル {CONFIG_FILE_PATH} を読み込みました。")

    resolved_api_creds = {}
    if 'api_credentials' in config_from_yaml and isinstance(config_from_yaml['api_credentials'], dict):
        for api_name, creds_config in config_from_yaml['api_credentials'].items():
            resolved_api_creds[api_name] = {}
            if isinstance(creds_config, dict):
                for key_type, env_var_name in creds_config.items():
                    if key_type.endswith("_env_var"):
                        actual_key_name = key_type[:-len("_env_var")]
                        env_value = os.getenv(str(env_var_name))
                        if env_value is not None:
                            resolved_api_creds[api_name][actual_key_name] = env_value
                            logger.debug(f"解決: api_credentials.{api_name}.{actual_key_name} <-- 環:{env_var_name}")
                        else:
                            resolved_api_creds[api_name][actual_key_name] = None
                            logger.warning(f"環境変数 '{env_var_name}' が見つかりません ({api_name}.{actual_key_name} 用)。")
                    else:
                        resolved_api_creds[api_name][key_type] = env_var_name
    
    config = config_from_yaml
    if resolved_api_creds:
        if 'api_credentials' not in config: config['api_credentials'] = {}
        config['api_credentials'].update(resolved_api_creds)

except FileNotFoundError:
    logger.critical(f"設定ファイル {CONFIG_FILE_PATH} が見つかりません。処理を終了します。")
    exit()
except yaml.YAMLError as e:
    logger.critical(f"設定ファイル {CONFIG_FILE_PATH} の解析に失敗しました: {e}。処理を終了します。")
    exit()
except Exception as e:
    logger.critical(f"設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
    logger.exception("スタックトレース:")
    exit()
logger.info("設定ファイルの読み込みと環境変数の解決が完了しました。")

if __name__ == "__main__":
    logger.info("=== 株価分析・通知システム起動 ===")

    jquants_creds = config.get('api_credentials', {}).get('jquants', {})
    j_mail = jquants_creds.get('mail_address')
    j_pass = jquants_creds.get('password')
    j_refresh_token = jquants_creds.get('refresh_token')

    fetcher = None
    if j_refresh_token:
        logger.info("JQuantsFetcherをリフレッシュトークンで初期化します...")
        fetcher = JQuantsFetcher(refresh_token=j_refresh_token)
    elif j_mail and j_pass:
        logger.info("JQuantsFetcherをメールアドレスとパスワードで初期化します...")
        fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)
    else:
        logger.critical("J-Quants APIの認証情報が設定されていません。処理を終了します。")
        exit()

    id_token = fetcher.get_id_token()
    if not id_token:
        logger.critical("JQuantsFetcher経由でのIDトークン取得に失敗。処理を終了します。")
        exit()
    logger.info(f"JQuantsFetcher経由でIDトークンを正常に取得/確認しました。")
    
    line_creds = config.get('api_credentials', {}).get('line', {})
    line_channel_access_token = line_creds.get('channel_access_token')
    line_user_id_to_notify = line_creds.get('user_id')
    
    line_notifier_instance = None
    if line_channel_access_token and line_user_id_to_notify:
        try:
            line_notifier_instance = LineNotifier(channel_access_token=line_channel_access_token)
            logger.info("LineNotifierのインスタンス化に成功しました。")
        except Exception as e:
            logger.error(f"LineNotifierの初期化中にエラー: {e}。通知機能は利用できません。")
            logger.exception("スタックトレース:")
    else:
        logger.warning("LINEの認証情報が不足しているため、LineNotifierを初期化できません。")

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

        jquants_settings = config.get('jquants_api_settings', {})
        data_range_days = jquants_settings.get('data_range_days', 200)
        
        # 運用に合わせて日付範囲の決定ロジックを調整してください
        data_from_str = "2025-02-01" # J-Quantsのデータカバレッジ内の過去日付
        data_to_str = "2025-02-16"   # J-Quantsのデータカバレッジ内の過去日付
        logger.info(f"データ取得期間: {data_from_str} から {data_to_str}")

        df_prices = fetcher.get_daily_stock_prices(
            stock_code=stock_code,
            date_from_str=data_from_str,
            date_to_str=data_to_str
        )
        
        df_with_indicators = None
        if df_prices is not None and not df_prices.empty:
            logger.info(f"銘柄 {stock_name}: {len(df_prices)} 件の株価データを取得。")
            
            technical_params = stock_info.get('technical_analysis_params', {})
            if not technical_params:
                logger.warning(f"銘柄 {stock_name}: technical_analysis_params未設定。分析スキップ。")
                df_with_indicators = df_prices.copy() 
            else:
                logger.info(f"銘柄 {stock_name}: テクニカル指標計算開始。パラメータ: {technical_params}")
                df_with_indicators = calculate_indicators(df_prices, technical_params)

                if df_with_indicators is not None and len(df_with_indicators.columns) > len(df_prices.columns):
                    logger.info(f"銘柄 {stock_name}: テクニカル指標計算完了。")
                elif df_with_indicators is not None and df_with_indicators.equals(df_prices):
                     logger.warning(f"銘柄 {stock_name}: テクニカル指標計算後、データに変化なし。")
                else: 
                    logger.error(f"銘柄 {stock_name}: テクニカル指標計算失敗または結果不正。")
                    df_with_indicators = df_prices.copy() 
        else:
            logger.warning(f"銘柄 {stock_name}: 株価データ取得失敗または空。分析・判定スキップ。")

        signal = "HOLD" 
        signal_details = {"message": "分析データなし"}

        if df_with_indicators is not None and not df_with_indicators.empty:
            logger.info(f"銘柄 {stock_name}: シグナル判定処理に進みます。データ件数: {len(df_with_indicators)}")
            
            signal_rules = stock_info.get('signal_rules', []) # ここが正しく読み込めるか
            if not signal_rules: # signal_rules が None または空リストの場合
                logger.warning(f"銘柄 {stock_name}: config.yamlにsignal_rules未設定。判定スキップ。")
                signal_details = {"message": "シグナルルール未設定"}
            else:
                logger.info(f"銘柄 {stock_name}: シグナル判定開始。ルール数: {len(signal_rules)}")
                signal, signal_details = evaluate_signals(df_with_indicators, signal_rules)
                logger.info(f"銘柄 {stock_name}: シグナル判定結果: {signal}, 詳細: {signal_details}")
            
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
                
                message_to_send = f"【{signal}シグナル検出のお知らせ】\n"
                message_to_send += f"銘柄: {stock_name} ({stock_code})\n"
                if 'rule_applied' in signal_details:
                    message_to_send += f"検知ルール: {signal_details['rule_applied']}\n"
                elif 'message' in signal_details:
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
                     latest_close = df_with_indicators.iloc[-1]['Close'] # 'Close'列が存在する前提
                     message_to_send += f"最終データ: {latest_date_str} 終値: {latest_close:.2f}"
                
                logger.debug(f"送信予定メッセージ:\n{message_to_send.strip()}")
                
                if line_notifier_instance and line_user_id_to_notify:
                    send_success = line_notifier_instance.send_push_message(line_user_id_to_notify, message_to_send.strip())
                    if send_success:
                        logger.info(f"銘柄 {stock_name}: LINE通知を送信しました。")
                    else:
                        logger.error(f"銘柄 {stock_name}: LINE通知の送信に失敗しました。")
                else:
                    logger.warning(f"銘柄 {stock_name}: LineNotifier未初期化またはユーザーID未設定のため通知スキップ。")
            elif signal != "HOLD":
                logger.info(f"銘柄 {stock_name}: シグナル '{signal}' は通知対象外です。")
            else:
                logger.info(f"銘柄 {stock_name}: シグナルは 'HOLD' でした。通知はありません。")
        else:
            logger.warning(f"銘柄 {stock_name}: 分析データがないためシグナル判定・通知をスキップ。")
        
        logger.info(f"--- 処理終了: {stock_name} ({stock_code}) ---")
        logger.debug("次の銘柄処理まで1秒待機します...")
        time.sleep(1) 

    logger.info("=== 全ての銘柄の処理が完了しました ===")