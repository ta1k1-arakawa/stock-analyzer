import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from datetime import datetime
import joblib
import logging
import os

# --- 自作モジュールのインポート ---
from config_loader import load_config_and_logger 
from jquants_fetcher import JQuantsFetcher
from yfinance_fetcher import YFinanceFetcher
from technical_analyzer import calculate_indicators
from prepare_target import create_target_variable

def train_ai_model(logger, config):
    """ AIモデルの学習を実行し、ファイルに保存する """
    logger.info("=== AIモデル学習プロセス開始 ===")

    # --- 設定の読み込み (新config構造に対応) ---
    target_stock_conf = config.get('target_stock', {})
    training_settings = config.get('training_settings', {})
    tech_params = config.get('technical_analysis_params', {})
    
    # 必要な設定値の取得
    stock_code = target_stock_conf.get('code')
    feature_columns = target_stock_conf.get('feature_columns', [])
    ai_params = target_stock_conf.get('ai_params', {})
    
    data_from = training_settings.get('data_from')
    data_to = training_settings.get('data_to')

    if data_to == "auto":
        data_to = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"終了日(data_to)を自動で今日の日付({data_to})に設定しました。")

    model_save_path = training_settings.get('model_save_path', 'stock_ai_model.pkl')

    if not stock_code or not feature_columns:
        logger.critical("config.yaml に target_stock (code, feature_columns) が正しく設定されていません。")
        return

    fetcher = YFinanceFetcher()
    # J-Quants認証
    # j_mail = os.getenv("JQUANTS_MAIL")
    # j_pass = os.getenv("JQUANTS_PASSWORD")
    # fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)

    # データ取得
    logger.info(f"銘柄 {stock_code} の {data_from} から {data_to} までのデータを取得します...")
    df_prices = fetcher.get_daily_stock_prices(
        stock_code=stock_code,
        date_from_str=data_from,
        date_to_str=data_to
    )
    
    if df_prices is None or df_prices.empty:
        logger.error("データ取得失敗またはデータが空です。学習を中止します。")
        return
    
    # 数値型変換
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df_prices.columns:
            df_prices[c] = pd.to_numeric(df_prices[c], errors='coerce')

    # 特徴量計算
    logger.info("テクニカル指標(特徴量)を計算します...")
    df_features = calculate_indicators(df_prices, tech_params)

    # 目的変数(Target)の作成
    future_days = ai_params.get('future_days', 1)
    target_percent = ai_params.get('target_percent', 1.0)
    
    df_ready = create_target_variable(df_features, future_days, target_percent)

    # 学習データのクリーニング 
    target_column = 'Target'
    # 使用する特徴量カラムが存在するかチェック
    missing_cols = [c for c in feature_columns if c not in df_ready.columns]
    if missing_cols:
        logger.error(f"計算できていない特徴量があります: {missing_cols}")
        return

    # 特徴量とターゲットにあるNaNを削除
    df_final_data = df_ready.dropna(subset=feature_columns + [target_column])

    if df_final_data.empty:
        logger.error("有効な学習データが0件になりました。データ期間を広げるか条件を見直してください。")
        return

    logger.info(f"学習データ件数: {len(df_final_data)} 件")
    
    # データを分割
    x = df_final_data[feature_columns]
    y = df_final_data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    # モデル学習 (LightGBM)
    logger.info("LightGBMモデルの学習を開始します...")
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(x_train, y_train)
    logger.info("学習完了。")

    # 評価
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision_buy = precision_score(y_test, y_pred, labels=[1], average='binary', zero_division=0)
    
    logger.info(f"--- モデル評価 (テストデータ) ---")
    logger.info(f"正解率 (Accuracy): {accuracy * 100:.2f} %")
    logger.info(f"適合率 (Precision - 買い): {precision_buy * 100:.2f} %")
    
    # モデル保存
    try:
        joblib.dump(model, model_save_path)
        logger.info(f"学習済みモデルを保存しました: {model_save_path}")
    except Exception as e:
        logger.error(f"モデルの保存に失敗しました: {e}")

if __name__ == "__main__":
    logger, config = load_config_and_logger(log_file_name='train_ai.log')
    if logger and config:
        train_ai_model(logger, config)
    else:
        print("設定ファイルの読み込みに失敗しました。")