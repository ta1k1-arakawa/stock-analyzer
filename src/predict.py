"""AI 予測・LINE 通知のメインロジック。"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from src import LOGGER_NAME
from src.analysis import calculate_indicators, sanitize_ohlcv
from src.config import AppConfig
from src.fetchers.yfinance import YFinanceFetcher
from src.notifier import SlackNotifier
from src.tracker import TradeTracker

logger = logging.getLogger(LOGGER_NAME)

# 日付判定はすべて日本時間 (JST) 基準で行う。
# GitHub Actions のランナーは UTC で動くため、タイムゾーンを指定しない
# datetime.now() を使うと日本の暦日とずれてしまう。
JST = ZoneInfo("Asia/Tokyo")

# 最新データの鮮度として許容する経過日数 (暦日)。
# 週末や祝日を挟んでも「直近営業日の終値」を有効として扱うための上限で、
# 金曜引け → 月曜夕方で3日、祝日を挟む連休でも概ねこの範囲に収まる。
# 一方、データ取得が数日止まっている場合はこれを超えて見送りになる。
MAX_DATA_AGE_DAYS = 4


def is_fresh_data(latest_date: date, today_jst: date) -> bool:
    """最新データの日付が、JST の今日から見て鮮度範囲内かを返す。

    旧判定は「最新データの日付 == 今日」の完全一致だったため、GitHub Actions の
    起動が遅れて日付をまたいだ実行は、直近営業日の終値が取れていても
    すべて「データ未更新」として見送られていた。
    """
    age_days = (today_jst - latest_date).days
    return 0 <= age_days <= MAX_DATA_AGE_DAYS


def run_prediction(config: AppConfig, notification_enabled: bool = False) -> None:
    """1銘柄を予測し、必要なら通知しつつ売買ログを更新する。"""

    logger.info("=== 株価分析・通知システム (AI本番運用版) 起動 ===")

    if not config.stock_code:
        logger.critical("実行対象の銘柄コードが設定されていません。")
        return

    # --- データ取得 ---
    fetcher = YFinanceFetcher()

    today = datetime.now(JST)
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

    # --- 通知チャネルの初期化 ---
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL") if notification_enabled else None
    slack_notifier: SlackNotifier | None = None
    if slack_webhook:
        slack_notifier = SlackNotifier(webhook_url=slack_webhook)
    elif notification_enabled:
        logger.warning("SLACK_WEBHOOK_URL が設定されていないため、Slack 通知は行われません。")
    else:
        logger.info("config.yaml で通知対象外のため、Slack通知は行いません。")

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

        # データ日付チェック (JST基準・直近営業日の終値なら有効)
        today_str = today.strftime("%Y-%m-%d")
        latest_date = latest_data.index[-1].date()
        data_age_days = (today.date() - latest_date).days
        is_latest_data = is_fresh_data(latest_date, today.date())
        data_warning = ""
        if not is_latest_data:
            logger.warning("注意: 最新データの日付(%s)がJSTの今日(%s)から%d日離れています。", latest_date_str, today_str, data_age_days)
            data_warning = (
                f"⚠️【データ未更新の可能性】\n"
                f"データ日付が {latest_date_str} です。\n"
                f"まだ本日のデータが反映されていない可能性があるため、売買判定はスキップします。\n\n"
            )
        elif data_age_days > 0:
            logger.info(
                "最新データは %s (JSTの今日 %s の%d日前) です。直近営業日の終値として判定します。",
                latest_date_str,
                today_str,
                data_age_days,
            )

        # 二重実行ガード: 同じデータ日付で処理済みなら、通知・記録を繰り返さない。
        # 外部スケジューラからの起動と GitHub の schedule (遅延・保険) が重なっても安全にするため。
        # 手動実行で再通知したい場合は環境変数 FORCE_PREDICTION=true (workflow_dispatch の force 入力) を使う。
        force_prediction = os.getenv("FORCE_PREDICTION", "").strip().lower() in ("1", "true", "yes")
        marker_path = config.trade_log_path.parent / f"prediction_done_{config.stock_code}.txt"
        if is_latest_data and not force_prediction and marker_path.exists():
            try:
                if marker_path.read_text(encoding="utf-8").strip() == latest_date_str:
                    logger.info("データ日付 %s は処理済みのため、この実行では通知・記録を行いません。", latest_date_str)
                    return
            except OSError:
                logger.warning("処理済みマーカーを読み込めませんでした: %s", marker_path)

        # トレードトラッカー
        tracker = TradeTracker(
            budget=config.ai_params.budget,
            filepath=config.trade_log_path,
            stop_loss_percent=config.ai_params.stop_loss_percent,
            entry_slippage_percent=config.ai_params.entry_slippage_percent,
            exit_slippage_percent=config.ai_params.exit_slippage_percent,
            stop_slippage_percent=config.ai_params.stop_slippage_percent,
            commission_percent=config.ai_params.commission_percent,
        )
        # 答え合わせの対象日は「直近営業日 (最新データの日付)」にする。
        # 実行日の暦日を使うと、起動が遅れて日付をまたいだ日に前営業日の
        # 売却結果がレポートから落ちてしまうため。
        report_msg = tracker.get_daily_report(config.stock_code, df_prices, report_date=latest_date_str)

        # 判定と通知メッセージ作成
        ai = config.ai_params
        if not is_latest_data:
            msg = (
                f"【AI予測 (判定スキップ)】\n"
                f"銘柄: {config.stock_name} ({config.stock_code})\n"
                f"確信度: {buy_prob:.1%}\n"
                f"基準: {ai.threshold:.0%}\n"
                f"現在値: {latest_close:.0f}円\n"
                f"データ日付: {latest_date_str}\n"
                f"実行日: {today_str}\n"
                f"判定: データ未更新のため見送ります。"
            )
            logger.info("判定: SKIP (データ未更新のためBUY通知・ログ記録なし)")
        elif buy_prob >= ai.threshold:
            stop_loss_price = latest_close * (1 - ai.stop_loss_percent / 100)
            tracker.log_signal(
                latest_date_str,
                config.stock_code,
                config.stock_name,
                buy_prob,
                ai.threshold,
                ai.future_days,
                ai.stop_loss_percent,
                ai.entry_slippage_percent,
                ai.exit_slippage_percent,
                ai.stop_slippage_percent,
                ai.commission_percent,
            )
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

        # Slack 送信
        if slack_notifier:
            slack_notifier.send_message(full_msg)
            logger.info("Slack 通知を送信しました。")
        elif notification_enabled:
            logger.info("Slack 設定がないため通知はスキップしました。")
        else:
            logger.info("通知対象外銘柄の予測・ログ更新が完了しました。")

        # 判定 (BUY / HOLD) まで完了したら、そのデータ日付を処理済みとして記録する。
        # データ未更新で見送った場合は記録しない (後の再実行で判定できるようにするため)。
        if is_latest_data:
            try:
                marker_path.write_text(latest_date_str + "\n", encoding="utf-8")
            except OSError:
                logger.warning("処理済みマーカーを書き込めませんでした: %s", marker_path)

    except Exception as e:
        logger.error("予測プロセス中にエラーが発生: %s", e)

    logger.info("=== 全処理完了 ===")


def run_all_predictions(config: AppConfig) -> None:
    """全銘柄のログを更新し、設定で選んだ銘柄だけ通知する。"""

    active_stocks = [stock for stock in config.stocks if stock.paper_trade]
    notification_codes = [stock.stock_code for stock in active_stocks if stock.notify_slack]
    if notification_codes:
        logger.info("Slack通知対象: %s", ", ".join(notification_codes))
    else:
        logger.warning("config.yaml で notify_slack: true の銘柄がありません。")

    for stock in active_stocks:
        stock_config = config.for_stock(stock)
        notification_enabled = stock.notify_slack
        try:
            run_prediction(stock_config, notification_enabled=notification_enabled)
        except Exception:
            logger.exception(
                "銘柄 %s (%s) の予測処理に失敗しました。次の銘柄へ進みます。",
                stock_config.stock_name,
                stock_config.stock_code,
            )
