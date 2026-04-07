"""AI 予測・LINE 通知のメインロジック。"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from src import LOGGER_NAME
from src.analysis import calculate_indicators, sanitize_ohlcv
from src.config import AppConfig
from src.fetchers.yfinance import YFinanceFetcher
from src.notifier import LineNotifier
from src.tracker import TradeTracker

logger = logging.getLogger(LOGGER_NAME)


def run_prediction(config: AppConfig) -> None:
    """AI モデルによる株価予測とLINE通知を実行する。"""

    logger.info("=== 株価分析・通知システム (AI本番運用版) 起動 ===")

    if not config.stock_code:
        logger.critical("config.yaml に監視対象銘柄 (target_stock.code) が設定されていません。")
        return

    # --- データ取得 ---
    fetcher = YFinanceFetcher()

    today = datetime.now()
    start_date = today - timedelta(days=365)
    data_from_str = start_date.strftime("%Y-%m-%d")
    data_to_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("--- 処理開始: %s (%s) ---", config.stock_name, config.stock_code)
    logger.info("データ取得期間: %s 〜 %s", data_from_str, data_to_str)

    df_prices = fetcher.get_daily_stock_prices(config.stock_code, data_from_str, data_to_str)

    if df_prices is None or df_prices.empty:
        logger.warning("株価データが取得できませんでした。")
        return

    df_prices = sanitize_ohlcv(df_prices)

    # --- LINE 通知の初期化 ---
    line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    line_user_id = os.getenv("LINE_USER_ID")

    line_notifier: LineNotifier | None = None
    if line_token and line_user_id:
        line_notifier = LineNotifier(channel_access_token=line_token)
    else:
        logger.warning("LINE 設定が見つからないため、通知は行われません。")

    # --- AI モデルの読み込み ---
    model_path = config.model_path
    if not model_path.exists():
        logger.critical("モデルファイル %s が見つかりません。先に train.py を実行してください。", model_path)
        return

    try:
        ai_model = joblib.load(model_path)
        logger.info("AI モデル %s を読み込みました。", model_path)
    except Exception as e:
        logger.critical("AI モデル読み込みエラー: %s", e)
        return

    # --- 特徴量計算 ---
    df_with_indicators = calculate_indicators(df_prices, config.tech_params)
    latest_data = df_with_indicators.iloc[-1:]

    if latest_data[config.feature_columns].isnull().any().any():
        logger.warning("最新データに必要な特徴量が計算できていません（データ不足の可能性）。")
        return

    # --- AI 予測 ---
    try:
        prediction_proba = ai_model.predict_proba(latest_data[config.feature_columns])
        buy_prob: float = prediction_proba[0][1]
        logger.info("AI 予測結果: 買い確率 %.2f%% (閾値: %.0f%%)", buy_prob * 100, config.ai_params.threshold * 100)

        latest_close = latest_data["Close"].iloc[0]
        latest_date_str = latest_data.index[-1].strftime("%Y-%m-%d")

        # データ日付チェック
        today_str = datetime.now().strftime("%Y-%m-%d")
        data_warning = ""
        if latest_date_str != today_str:
            logger.warning("注意: 最新データの日付(%s)が今日(%s)ではありません。", latest_date_str, today_str)
            data_warning = (
                f"⚠️【データ未更新の可能性】\n"
                f"データ日付が {latest_date_str} です。\n"
                f"まだ本日のデータが反映されていない可能性があります。\n\n"
            )

        # トレードトラッカー
        tracker = TradeTracker(budget=config.ai_params.budget, filepath=config.trade_log_path)
        report_msg = tracker.get_daily_report(config.stock_code, df_prices)

        # 判定と通知メッセージ作成
        ai = config.ai_params
        if buy_prob >= ai.threshold:
            stop_loss_price = latest_close * (1 - ai.stop_loss_percent / 100)
            tracker.log_signal(latest_date_str, config.stock_code, config.stock_name, buy_prob, ai.threshold, ai.future_days)
            msg = (
                f"【AI買いシグナル】\n"
                f"銘柄: {config.stock_name} ({config.stock_code})\n"
                f"確信度: {buy_prob:.1%}\n"
                f"基準: {ai.threshold:.0%}\n"
                f"現在値: {latest_close:.0f}円\n"
                f"日時: {latest_date_str}\n"
                f"⚠️ 損切り目安: {stop_loss_price:.0f}円 (-{ai.stop_loss_percent}%)"
            )
            logger.info("判定: BUY (通知送信)")
        else:
            msg = (
                f"【AI予測 (様子見)】\n"
                f"銘柄: {config.stock_name} ({config.stock_code})\n"
                f"確信度: {buy_prob:.1%} (基準未満)\n"
                f"基準: {ai.threshold:.0%}\n"
                f"現在値: {latest_close:.0f}円\n"
                f"日時: {latest_date_str}\n"
                f"判定: 今回は見送ります。"
            )
            logger.info("判定: HOLD (テスト通知送信)")

        # メッセージ組み立て
        full_msg = data_warning
        if report_msg:
            full_msg += report_msg + "\n"
        full_msg += msg

        # LINE 送信
        if line_notifier and line_user_id:
            line_notifier.send_push_message(line_user_id, full_msg)
            logger.info("LINE 通知を送信しました。")
        else:
            logger.info("LINE 設定がないため通知はスキップしました。")

    except Exception as e:
        logger.error("予測プロセス中にエラーが発生: %s", e)

    logger.info("=== 全処理完了 ===")
