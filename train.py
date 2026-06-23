# train.py — AIモデル学習 エントリポイント

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import joblib
import lightgbm as lgb
import pandas as pd

from src import LOGGER_NAME
from src.analysis import calculate_indicators, create_target_variable, sanitize_ohlcv
from src.config import AppConfig, load_app
from src.fetchers.yfinance import YFinanceFetcher


def train_ai_model(config: AppConfig) -> bool:
    """AIモデルを学習・保存し、成功した場合だけ ``True`` を返す。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("=== AIモデル学習プロセス開始 (フル学習モード) ===")

    # 前回のモデルを誤って再利用しないよう、学習前に必ず削除する。
    try:
        config.model_path.unlink(missing_ok=True)
    except OSError:
        logger.exception("古いモデルを削除できませんでした: %s", config.model_path)
        return False

    data_from = config.training_settings.get("data_from")
    data_to = config.training_settings.get("data_to")

    if data_to == "auto":
        data_to = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("終了日(data_to)を自動で %s (明日) に設定しました。", data_to)

    # データ取得
    logger.info("銘柄 %s の %s から %s までのデータを取得します...", config.stock_code, data_from, data_to)
    fetcher = YFinanceFetcher()
    df = fetcher.get_daily_stock_prices(str(config.stock_code), data_from, data_to)

    if df is None or df.empty:
        logger.error("データ取得に失敗したため、学習を中止します。")
        return False

    df = sanitize_ohlcv(df)

    # 特徴量生成
    logger.info("テクニカル指標を計算中...")
    df_ta = calculate_indicators(df, config.tech_params)

    # 正解ラベルの作成
    ai = config.ai_params
    logger.info("正解ラベル作成: %d日後に %.1f%% 上昇なら勝ち(1)", ai.future_days, ai.target_percent)
    df_ready = create_target_variable(
        df_ta,
        ai.future_days,
        ai.target_percent,
        ai.entry_slippage_percent,
        ai.exit_slippage_percent,
    )

    # 必要なカラム確認
    missing_cols = [c for c in config.feature_columns if c not in df_ready.columns]
    if missing_cols:
        logger.error("必要な特徴量が不足しています: %s", missing_cols)
        return False

    # NaN 削除
    target_column = "Target"
    df_final = df_ready.dropna(subset=config.feature_columns + [target_column])

    if df_final.empty:
        logger.error("有効な学習データが0件になりました。データ期間を広げるか条件を見直してください。")
        return False

    logger.info("学習データ件数: %d 件 (全データを学習に使用)", len(df_final))

    # 学習
    x = df_final[config.feature_columns]
    y = df_final[target_column]

    logger.info("LightGBM モデルの学習を開始します...")
    model = lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        deterministic=True,
        n_jobs=1,
    )
    model.fit(x, y)
    logger.info("学習完了。")

    # モデルの保存
    try:
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.model_path)
        logger.info("モデルを保存しました: %s", config.model_path)
        return True
    except Exception as e:
        logger.error("モデル保存エラー: %s", e)
        return False


def train_all_models(config: AppConfig) -> bool:
    """全銘柄を学習し、すべて成功した場合だけ ``True`` を返す。"""

    targets = [config] + [
        config.for_log_only_stock(stock) for stock in config.log_only_stocks
    ]
    logger = logging.getLogger(LOGGER_NAME)
    failed_stocks: list[str] = []
    for stock_config in targets:
        try:
            if not train_ai_model(stock_config):
                failed_stocks.append(stock_config.stock_code)
        except Exception:
            logger.exception(
                "銘柄 %s (%s) の学習中に予期しないエラーが発生しました。",
                stock_config.stock_name,
                stock_config.stock_code,
            )
            failed_stocks.append(stock_config.stock_code)

    if failed_stocks:
        logger.critical(
            "学習に失敗した銘柄があります。予測処理を停止します: %s",
            ", ".join(failed_stocks),
        )
        return False

    return True


if __name__ == "__main__":
    config, logger = load_app(log_file="app_train.log")
    if not train_all_models(config):
        raise SystemExit(1)
