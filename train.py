import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
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
    """ AIモデルの学習を実行し、ファイルに保存する (フル学習モード) """
    logger.info("=== AIモデル学習プロセス開始 (フル学習モード) ===")

    # 設定の読み込み
    target_stock_conf = config.get('target_stock', {})
    training_settings = config.get('training_settings', {})
    tech_params = config.get('technical_analysis_params', {})
    
    # 必要な設定値の取得
    stock_code = target_stock_conf.get('code')
    feature_columns = target_stock_conf.get('feature_columns', [])
    ai_params = target_stock_conf.get('ai_params', {})
    
    data_from = training_settings.get('data_from')
    data_to = training_settings.get('data_to')

    # 日付の自動設定 (今日を含めるため「明日」を設定)
    if data_to == "auto":
        data_to = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        logger.info(f"終了日(data_to)を自動で {data_to} (明日) に設定しました。これで今日のデータが含まれます。")

    model_save_path = training_settings.get('model_save_path', 'stock_ai_model.pkl')
    
    # データ取得
    logger.info(f"銘柄 {stock_code} の {data_from} から {data_to} までのデータを取得します...")
    
    fetcher = YFinanceFetcher()
    df = fetcher.get_daily_stock_prices(str(stock_code), data_from, data_to)
    
    if df is None or df.empty:
        logger.error("データ取得に失敗したため、学習を中止します。")
        return

    # 数値型への変換
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 特徴量生成
    logger.info("テクニカル指標を計算中...")
    df_ta = calculate_indicators(df, tech_params)

    # 正解ラベルの作成
    future_days = ai_params.get('future_days', 1)
    target_percent = ai_params.get('target_percent', 1.0)
    
    logger.info(f"正解ラベル作成: {future_days}日後に {target_percent}% 上昇なら勝ち(1)")
    df_ready = create_target_variable(df_ta, future_days, target_percent)
    
    # 必要なカラム確認
    missing_cols = [c for c in feature_columns if c not in df_ready.columns]
    if missing_cols:
        logger.error(f"必要な特徴量が不足しています: {missing_cols}")
        return

    # 特徴量とターゲットにあるNaNを削除
    # Target列も含めてdropnaすることで、未来のデータがない最新の日付（学習に使えない行）も自動的に除外される
    target_column = 'Target'
    df_final_data = df_ready.dropna(subset=feature_columns + [target_column])

    if df_final_data.empty:
        logger.error("有効な学習データが0件になりました。データ期間を広げるか条件を見直してください。")
        return

    logger.info(f"学習データ件数: {len(df_final_data)} 件 (全データを学習に使用)")
    
    # 学習
    x = df_final_data[feature_columns]
    y = df_final_data[target_column]

    logger.info("LightGBMモデルの学習を開始します...")
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(x, y)
    logger.info("学習完了。")


    # モデルの保存
    try:
        joblib.dump(model, model_save_path)
        logger.info(f"モデルを保存しました: {model_save_path}")
    except Exception as e:
        logger.error(f"モデル保存エラー: {e}")

if __name__ == "__main__":
    logger, config = load_config_and_logger(log_file_name='app_train.log')
    if logger and config:
        train_ai_model(logger, config)