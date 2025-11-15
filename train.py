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

    # 学習用データの取得
    train_config = config.get('ai_training_settings')
    if not train_config:
        logger.critical("config.yaml に 'ai_training_settings' が見つかりません。")
        return

    stock_code = train_config['stock_code']
    data_from = train_config['data_from']
    data_to = train_config['data_to']
    
    # 認証情報を config から取得 
    jquants_creds = config.get('api_credentials').get('jquants')
    j_mail = jquants_creds.get('mail_address')
    j_pass = jquants_creds.get('password')
    
    fetcher = None
    if j_mail and j_pass:
        fetcher = JQuantsFetcher(mail_address=j_mail, password=j_pass)
    else:
        logger.critical("J-Quants APIの認証情報が設定されていません。処理を終了します。")
        return
    
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

    # 特徴量の計算 
    technical_params = train_config.get('technical_analysis_params')
    if not technical_params:
        logger.error("configに 'technical_analysis_params' がありません。")
        return
        
    logger.info("特徴量を計算します...")
    df_features = calculate_indicators(df_prices, technical_params)

    # 目的変数の作成 
    future_days = train_config.get('future_days')
    target_percent = train_config.get('target_percent')
    
    df_ready = create_target_variable(df_features, future_days, target_percent)

    # 学習データセットの最終準備 
    feature_columns = train_config.get('feature_columns')
    if not feature_columns:
        logger.error("configに 'feature_columns'がありません。")
        return
        
    # 'Target' 列を含めてNaNを削除
    target_column = 'Target'
    df_final_data = df_ready.dropna(subset=feature_columns + [target_column])

    if df_final_data.empty:
        logger.error("NaNを削除した結果、学習データが0件になりました。")
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

    try:
        logger.info("特徴量の重要度を計算・表示します...")
        # 特徴量の重要度を表示
        feature_names = x_train.columns
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        logger.info("--- 特徴量の重要度 (Top 10) ---")
        logger.info(f"\n{importance_df.head(10).to_string()}")
        logger.info("----------------------------------")

    except ImportError:
        logger.warning("matplotlib がインストールされていないため、特徴量の重要度グラフを保存できません。")
    except Exception as e:
        logger.error(f"特徴量の重要度グラフの保存中にエラーが発生しました: {e}")

    # モデルの評価（3クラス分類対応）
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 3クラス分類用の評価指標（weighted average）
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # クラスごとの適合率（買い=1のみ）
    precision_buy = precision_score(y_test, y_pred, labels=[1], average='micro', zero_division=0)
    
    logger.info(f"--- モデル評価 (テストデータ) ---")
    logger.info(f"正解率 (Accuracy): {accuracy * 100:.2f} %")
    logger.info(f"適合率 (Weighted): {precision_weighted * 100:.2f} % (←全クラス加重平均)")
    logger.info(f"適合率 (買いシグナル): {precision_buy * 100:.2f} % (←「買い(1)」予測の精度)")
    
    # クラスごとのサンプル数を表示
    unique, counts = pd.Series(y_test).value_counts().sort_index().index, pd.Series(y_test).value_counts().sort_index().values
    logger.info(f"テストデータのクラス分布: 売り(-1)={counts[0] if -1 in unique else 0}, 何もしない(0)={counts[1] if 0 in unique else 0}, 買い(1)={counts[2] if 1 in unique else 0}")
    logger.info("----------------------------------")

    # 学習済みモデルの保存
    model_filename = train_config.get('model_save_path')
    try:
        joblib.dump(model, model_filename)
        logger.info(f"学習済みモデルを {model_filename} として保存しました。")
    except Exception as e:
        logger.error(f"モデルの保存に失敗しました: {e}")


if __name__ == "__main__":
    # 共通モジュールを呼び出して logger と config を取得
    logger, config = load_config_and_logger(log_file_name='train_ai.log')
    
    if logger and config:
        train_ai_model(logger, config)
    else:
        print("設定ファイルの読み込みに失敗したため、train.py を終了します。")