import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import joblib
import logging
import matplotlib.pyplot as plt

# --- 自作モジュールのインポート ---
from config_loader import load_config_and_logger 
from jquants_fetcher import JQuantsFetcher
from technical_analyzer import calculate_indicators
from prepare_target import create_target_variable

def train_ai_model(logger, config):
    """ AIモデルの学習を実行し、ファイルに保存する """
    logger.info("=== AIモデル学習プロセス開始 ===")

    # 設定の読み込み
    train_config = config.get('ai_training_settings')
    common_config = config.get('common_feature_settings') 

    if not train_config or not common_config:
        logger.critical("config.yaml に必要な設定(ai_training_settings / common_feature_settings)が見つかりません。")
        return

    # 共通設定から特徴量リストとパラメータを取得
    feature_columns = common_config.get('feature_columns')
    technical_params = common_config.get('technical_analysis_params')

    stock_code = train_config['stock_code']
    data_from = train_config['data_from']
    data_to = train_config['data_to']
    
    # 認証とデータ取得
    jquants_creds = config.get('api_credentials').get('jquants')
    j_mail = jquants_creds.get('mail_address') or jquants_creds.get('mail_address_env_var') # 環境変数のキー名対応等はload_config側依存だが念のため
    j_pass = jquants_creds.get('password') or jquants_creds.get('password_env_var')
    
    fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)
    
    if not fetcher.get_id_token():
        logger.critical("J-Quants認証失敗。学習を中止します。")
        return

    logger.info(f"銘柄 {stock_code} の {data_from} から {data_to} までのデータを取得します...")
    df_prices = fetcher.get_daily_stock_prices(
        stock_code=stock_code,
        date_from_str=data_from,
        date_to_str=data_to
    )
    
    if df_prices is None or df_prices.empty:
        logger.error("データ取得失敗またはデータが空です。学習を中止します。")
        return

    # 特徴量の計算 (共通設定を使用)
    logger.info("特徴量を計算します...")
    df_features = calculate_indicators(df_prices, technical_params)

    # 目的変数の作成 
    future_days = train_config.get('future_days')
    target_percent = train_config.get('target_percent')
    
    df_ready = create_target_variable(df_features, future_days, target_percent)

    # 学習データセットの最終準備 
    target_column = 'Target'
    # 特徴量とターゲットにあるNaNを削除
    df_final_data = df_ready.dropna(subset=feature_columns + [target_column])

    if df_final_data.empty:
        logger.error("NaNを削除した結果、学習データが0件になりました。期間を広げるか条件を見直してください。")
        return

    logger.info(f"最終的な学習データ件数: {len(df_final_data)}")
    x = df_final_data[feature_columns]
    y = df_final_data[target_column]

    # データを「学習用」と「テスト用」に分割
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    # AI (LightGBM) の学習
    logger.info("LightGBMモデルの学習を開始します...")
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(x_train, y_train)
    logger.info("学習完了。")

    # 特徴量重要度の表示
    try:
        feature_names = x_train.columns
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        logger.info("--- 特徴量の重要度 (Top 10) ---")
        logger.info(f"\n{importance_df.head(10).to_string()}")

    except Exception as e:
        logger.warning(f"特徴量重要度の表示中にエラー: {e}")

    # モデルの評価
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision_buy = precision_score(y_test, y_pred, labels=[1], average='micro', zero_division=0)
    
    logger.info(f"--- モデル評価 (テストデータ) ---")
    logger.info(f"正解率 (Accuracy): {accuracy * 100:.2f} %")
    logger.info(f"適合率 (買いシグナル): {precision_buy * 100:.2f} %")
    
    # モデル保存
    model_filename = train_config.get('model_save_path')
    try:
        joblib.dump(model, model_filename)
        logger.info(f"学習済みモデルを {model_filename} として保存しました。")
    except Exception as e:
        logger.error(f"モデルの保存に失敗しました: {e}")

if __name__ == "__main__":
    logger, config = load_config_and_logger(log_file_name='train_ai.log')
    if logger and config:
        train_ai_model(logger, config)
    else:
        print("設定ファイルの読み込みに失敗しました。")