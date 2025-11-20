# backtest.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging
import sys
import yaml
import os

# --- 本番用モジュールのインポート (ここが重要) ---
# これにより、本番と全く同じ計算ロジックが保証されます
try:
    from jquants_fetcher import JQuantsFetcher
    from technical_analyzer import calculate_indicators
    from prepare_target import create_target_variable
except ImportError as e:
    print(f"【重要エラー】必要なファイルが見つかりません: {e}")
    print("backtest.py は、technical_analyzer.py, prepare_target.py, jquants_fetcher.py と同じ場所に置いてください。")
    sys.exit(1)

# 簡易ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backtest')

def load_config(config_path='config.yaml'):
    """config.yamlを読み込む"""
    if not os.path.exists(config_path):
        logger.error(f"{config_path} が見つかりません。")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_backtest():
    print("\n" + "="*60)
    print("   AI自動売買シミュレーション (本番環境完全連動版)")
    print("="*60)

    # 1. 設定読み込み
    config = load_config()
    ai_settings = config.get('ai_training_settings', {})
    
    STOCK_CODE = ai_settings.get('stock_code')
    # シミュレーション用に期間を少し長めに書き換えてもOKですが、一旦config通りにします
    DATA_FROM = ai_settings.get('data_from', '2024-01-01') 
    DATA_TO = ai_settings.get('data_to', '2025-12-31')
    
    TARGET_PERCENT = ai_settings.get('target_percent', 2.0)
    FUTURE_DAYS = ai_settings.get('future_days', 3)
    
    # 本番で使う特徴量のリスト（これ以外は学習に使わない）
    FEATURE_COLUMNS = ai_settings.get('feature_columns', [])
    
    # 本番で使うテクニカル指標の計算パラメータ
    TECH_PARAMS = ai_settings.get('technical_analysis_params', {})

    # シミュレーション設定
    INITIAL_BUDGET = 300_000
    AI_THRESHOLD = 0.70  # 買い判定の閾値（configのmonitor設定に合わせるのが理想）

    print(f" 対象銘柄: {STOCK_CODE}")
    print(f" 期間    : {DATA_FROM} 〜 {DATA_TO}")
    print(f" 特徴量数: {len(FEATURE_COLUMNS)}個 (config.yaml定義準拠)")

    # 2. データ取得 (JQuants)
    jquants_creds = config.get('api_credentials', {}).get('jquants', {})
    fetcher = JQuantsFetcher(mail_address=jquants_creds.get('mail_address_env_var') or os.getenv("JQUANTS_MAIL"), 
                             password=jquants_creds.get('password_env_var') or os.getenv("JQUANTS_PASSWORD"))
    
    # 環境変数から読み込めない場合のフォールバック（直書き用、非推奨）
    if not fetcher.mail_address: 
         fetcher.mail_address = jquants_creds.get('mail_address') # configに直接書いている場合
         fetcher.password = jquants_creds.get('password')

    if not fetcher.get_id_token():
        print("エラー: J-Quants認証に失敗しました。")
        return

    logger.info(f"株価データ取得開始: {STOCK_CODE}")
    df_prices = fetcher.get_daily_stock_prices(STOCK_CODE, DATA_FROM, DATA_TO)
    if df_prices is None or df_prices.empty:
        print("エラー: データが取得できませんでした。")
        return

    # 数値型への変換 (念のため)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_prices[col] = pd.to_numeric(df_prices[col], errors='coerce')

    # 3. 特徴量計算 (★本番モジュールを使用)
    logger.info("テクニカル指標を計算中 (technical_analyzer.py使用)...")
    df_with_indicators = calculate_indicators(df_prices, TECH_PARAMS)

    # 4. 正解データ作成 (★本番モジュールを使用)
    logger.info("学習用データを生成中 (prepare_target.py使用)...")
    df_dataset = create_target_variable(df_with_indicators, FUTURE_DAYS, TARGET_PERCENT)

    # 5. 特徴量の選定 (config.yamlにある列だけを使う)
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_dataset.columns]
    if missing_cols:
        logger.error(f"Configで指定された以下の特徴量が計算されていません: {missing_cols}")
        logger.error("technical_analyzer.py または config.yaml を確認してください。")
        return

    # 6. データ分割 (時系列分割: 直近60日をテストデータとする)
    TEST_DAYS = 60
    if len(df_dataset) < TEST_DAYS * 2:
        logger.warning("データが少なすぎます。テスト期間を短縮します。")
        TEST_DAYS = int(len(df_dataset) * 0.2)

    train_df = df_dataset.iloc[:-TEST_DAYS]
    test_df = df_dataset.iloc[-TEST_DAYS:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['Target']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['Target'] # 評価用

    print(f"学習データ: {len(train_df)}件, テストデータ: {len(test_df)}件")

    # 7. モデル学習
    logger.info("LightGBMモデルを学習中...")
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    # 8. シミュレーション実行 (厳密な先読みなしロジック)
    logger.info("収益シミュレーションを実行中...")

    # クラス分布の確認
    if 1 not in model.classes_:
        print("【警告】学習データに「買い(1)」が存在しないため、一度も買い注文が出ません。")
        return
    
    # 「買い(1)」クラスのインデックスを取得
    buy_class_index = list(model.classes_).index(1)
    
    # 確率予測
    probs = model.predict_proba(X_test)
    
    current_budget = INITIAL_BUDGET
    holdings = 0
    trade_log = []
    asset_history = [] # 日々の資産推移

    # ループ範囲: テストデータの「最後から2番目」まで
    # (理由: 今日の予測で明日のOpenで買うには、明日のデータが必要なため)
    for i in range(len(test_df) - 1):
        # 本日のデータ観測時点
        today_idx = test_df.index[i]
        
        # 1. AI予測 (今日の終値確定後に行う想定)
        # 今日のデータを使って予測する
        buy_prob = probs[i][buy_class_index]

        # 2. 売買判定
        # 翌日のデータ (実際に売買する価格)
        next_day_row = test_df.iloc[i + 1]
        next_day_open = next_day_row['Open']
        next_day_close = next_day_row['Close'] # デイトレならここで売る、スイングなら保持ロジックへ

        action = "STAY"
        
        # 買い条件: AI確率が高い かつ 予算がある
        if buy_prob >= AI_THRESHOLD and current_budget >= next_day_open:
            # 翌日始値で買えるだけ買う
            qty = int(current_budget // next_day_open)
            if qty > 0:
                buy_cost = qty * next_day_open
                commission = 0 # 必要なら buy_cost * 0.001 など
                
                current_budget -= (buy_cost + commission)
                
                # ここでは簡易的に「翌日の引け(Close)で売却」するデイトレ/短期売買とする
                # (FUTURE_DAYS保有などのロジックにする場合はここを調整)
                sell_revenue = qty * next_day_close
                profit = sell_revenue - buy_cost
                
                current_budget += sell_revenue
                
                action = "TRADE"
                trade_log.append({
                    'SignalDate': today_idx.strftime('%Y-%m-%d'),
                    'TradeDate': next_day_row.name.strftime('%Y-%m-%d'),
                    'Type': 'BUY->SELL',
                    'Price_In': next_day_open,
                    'Price_Out': next_day_close,
                    'Qty': qty,
                    'Profit': int(profit),
                    'AI_Prob': f"{buy_prob:.1%}"
                })

        # 資産記録 (その日の終わりの現金残高)
        asset_history.append(current_budget)

    # 9. 結果表示
    final_profit = current_budget - INITIAL_BUDGET
    roi = (final_profit / INITIAL_BUDGET) * 100

    print("\n" + "="*60)
    print(f"【シミュレーション結果】 {STOCK_CODE}")
    print(f" 初期予算 : {INITIAL_BUDGET:,} 円")
    print(f" 最終残高 : {int(current_budget):,} 円")
    print(f" 純利益   : {int(final_profit):,} 円 ({roi:+.2f}%)")
    print(f" 取引回数 : {len(trade_log)} 回")
    print("="*60)

    if trade_log:
        print("--- 取引履歴 (直近5件) ---")
        df_log = pd.DataFrame(trade_log)
        print(df_log.tail(5).to_string(index=False))
    else:
        print("取引は発生しませんでした。(閾値を下げるか、期間を変更してください)")

    # 10. グラフ描画
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    
    # 株価 (テスト期間)
    # asset_historyは len(test_df)-1 の長さなので、それに合わせて描画
    plot_dates = test_df.index[:-1]
    plot_close = test_df['Close'].iloc[:-1]
    
    ax1.plot(plot_dates, plot_close, color='gray', alpha=0.4, label='Stock Price')
    ax1.set_ylabel('Price', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    # 資産推移
    ax2 = ax1.twinx()
    ax2.plot(plot_dates, asset_history, color='blue', linewidth=2, label='Assets')
    ax2.set_ylabel('Assets (JPY)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title(f"Backtest: {STOCK_CODE} | Profit: {int(final_profit):,} JPY")
    plt.grid(True, alpha=0.3)
    
    save_path = f'backtest_{STOCK_CODE}.png'
    plt.savefig(save_path)
    print(f"\nグラフを保存しました: {save_path}")

if __name__ == "__main__":
    run_backtest()