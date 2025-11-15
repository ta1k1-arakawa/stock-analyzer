import pandas as pd
import logging
from datetime import datetime, timedelta
import sys

# 既存の自作モジュールをインポート
try:
    from config_loader import load_config_and_logger
    from jquants_fetcher import JQuantsFetcher
    from prepare_target import create_target_variable
except ImportError as e:
    print(f"エラー: 必要なモジュールが見つかりません: {e}")
    print("このスクリプトは、config_loader.py, jquants_fetcher.py, prepare_target.py と同じ階層に置いて実行してください。")
    sys.exit(1)

def analyze_target_frequency():
    """
    config.yamlの設定に基づき、AIの「答え」の頻度を分析する。
    """
    # config と logger を読み込む
    logger, config = load_config_and_logger(log_file_name='check_frequency.log')
    if not logger or not config:
        print("設定ファイルの読み込みに失敗しました。")
        return

    # --- JQuants API 認証 ---
    jquants_creds = config.get('api_credentials', {}).get('jquants', {})
    j_mail = jquants_creds.get('mail_address')
    j_pass = jquants_creds.get('password')
    
    fetcher = None
    if j_mail and j_pass:
        fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)
    else:
        logger.critical("J-Quants APIの認証情報が設定されていません。")
        return

    if not fetcher.get_id_token():
        logger.critical("J-Quants認証失敗。")
        return
        
    logger.info("J-Quants認証成功。")

    # --- config.yaml から設定値を取得 ---
    try:
        STOCK_CODE = config['ai_training_settings']['stock_code']
        FUTURE_DAYS = config['ai_training_settings']['future_days']
        TARGET_PERCENT = config['ai_training_settings']['target_percent']
        date_from = config['ai_training_settings']['data_from']
        date_to = config['ai_training_settings']['data_to']
    except KeyError as e:
        logger.error(f"config.yaml に {e} が設定されていません。")
        return

    logger.info(f"銘柄 {STOCK_CODE} の株価データを取得します...")
    logger.info(f"期間: {date_from} から {date_to}")

    # データ取得（ページネーション対応済みの get_daily_stock_prices を使用）
    df_prices = fetcher.get_daily_stock_prices(STOCK_CODE, date_from, date_to)
    
    if df_prices is None or df_prices.empty:
        logger.error(f"銘柄 {STOCK_CODE} のデータ取得に失敗したか、空でした。")
        return

    logger.info(f"データ取得成功。{len(df_prices)} 件のデータを分析します。")

    # 'Close'列を数値型に変換
    df_prices['Close'] = pd.to_numeric(df_prices['Close'], errors='coerce')
    df_prices.dropna(subset=['Close'], inplace=True)

    # ターゲット変数を作成 (prepare_target.py を使用)
    df_with_target = create_target_variable(df_prices, FUTURE_DAYS, TARGET_PERCENT)
    
    # --- 結果の集計 ---
    print("\n" + "="*40)
    print(f"--- 分析結果 ({FUTURE_DAYS}営業日後に±{TARGET_PERCENT}%変動する頻度) ---")
    
    # .dropna() は計算不能な最新N日分を除外
    target_counts = df_with_target['Target'].dropna().value_counts()
    target_percentage = df_with_target['Target'].dropna().value_counts(normalize=True) * 100
    
    print("\n[発生回数]")
    print(target_counts)
    
    print("\n[発生頻度 (%)]")
    print(target_percentage.round(2))
    
    print("\n" + "="*40)
    
    buy_freq = target_percentage.get(1, 0)
    sell_freq = target_percentage.get(-1, 0)
    hold_freq = target_percentage.get(0, 0)

    print(f"\n・「買い(1)」(5日後に+{TARGET_PERCENT}%以上 上昇): {buy_freq:.2f} %")
    print(f"・「売り(-1)」(5日後に-{TARGET_PERCENT}%以上 下落): {sell_freq:.2f} %")
    print(f"・「何もしない(0)」(変動が±{TARGET_PERCENT}%未満): {hold_freq:.2f} %")
    print("\n" + "="*40)

    # --- 結論 ---
    if hold_freq > 70:
        print("結論: 「何もしない(0)」が70%を超えています。")
        print("あなたの懸念通り、変動が小さく「まれ」と言えるかもしれません。")
        print("config.yaml の target_percent (現在 2.0%) を小さくする（例: 1.5%）か、")
        print("future_days (現在 5日) を長くする（例: 7日）ことを検討してください。")
    elif hold_freq < 50:
        print("結論: 「何もしない(0)」が50%未満です。")
        print("「買い(1)」または「売り(-1)」が頻繁に発生しており、AIが学習するのに十分な変動があります。")
        print("現在の手法で問題ない可能性が高いです。")
    else:
        print("結論: バランスの取れたデータです。")
        print("このまま学習を進めて問題ないでしょう。")

if __name__ == "__main__":
    analyze_target_frequency()