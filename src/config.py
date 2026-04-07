"""
設定管理およびロガー初期化モジュール。

config_loader.py と logger_setup.py を統合。
config.yaml / .env の読み込みと、dataclass によるアクセスを提供する。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src import LOGGER_NAME

# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class AIParams:
    budget: int = 100_000
    future_days: int = 2
    target_percent: float = 0.7
    threshold: float = 0.5
    stop_loss_percent: float = 3.0


@dataclass
class AppConfig:
    """アプリケーション全体の設定を保持する。"""

    stock_code: str = ""
    stock_name: str = ""
    feature_columns: list[str] = field(default_factory=list)
    ai_params: AIParams = field(default_factory=AIParams)
    tech_params: dict[str, Any] = field(default_factory=dict)
    training_settings: dict[str, Any] = field(default_factory=dict)
    backtest_candidates: list[str] = field(default_factory=list)
    model_path: Path = Path("models/stock_ai_model.pkl")
    trade_log_path: Path = Path("data/trade_log.csv")

    # 元の生 YAML 辞書（後方互換用）
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# ロガー設定
# ---------------------------------------------------------------------------

def setup_logger(
    log_file: str = "app.log",
    log_level_str: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """パッケージ共通のルートロガーを構成して返す。"""

    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    full_log_path = log_dir / log_file

    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"警告: 無効なログレベル '{log_level_str}'. INFO を使用します。", file=sys.stderr)
        numeric_level = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(numeric_level)

    # 既存ハンドラをクリア（多重登録防止）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ファイルハンドラ
    try:
        fh = logging.handlers.RotatingFileHandler(
            full_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"エラー: ログファイルハンドラの設定に失敗しました: {e}", file=sys.stderr)

    # コンソールハンドラ
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------

def _resolve_env_credentials(raw_creds: dict[str, Any]) -> dict[str, Any]:
    """api_credentials セクションの ``*_env_var`` キーを環境変数値に解決する。"""
    logger = logging.getLogger(LOGGER_NAME)
    resolved: dict[str, Any] = {}

    for api_name, creds_config in raw_creds.items():
        resolved[api_name] = {}
        if not isinstance(creds_config, dict):
            continue
        for key_type, env_var_name in creds_config.items():
            if key_type.endswith("_env_var"):
                actual_key = key_type.removesuffix("_env_var")
                env_value = os.getenv(str(env_var_name))
                if env_value is not None:
                    resolved[api_name][actual_key] = env_value
                    logger.debug("解決: api_credentials.%s.%s <-- 環:%s", api_name, actual_key, env_var_name)
                else:
                    resolved[api_name][actual_key] = None
                    logger.warning("環境変数 '%s' が見つかりません (%s.%s 用)。", env_var_name, api_name, actual_key)
            else:
                resolved[api_name][key_type] = env_var_name

    return resolved


def load_app(
    log_file: str = "app.log",
    config_path: str = "config.yaml",
) -> tuple[AppConfig, logging.Logger]:
    """
    .env と config.yaml を読み込み、ロガーを初期化して ``(AppConfig, Logger)`` を返す。

    失敗時は ``sys.exit(1)`` する。
    """

    # .env
    dotenv_ok = load_dotenv()

    # まずロガーを仮設定
    logger = setup_logger(log_file=log_file)

    if dotenv_ok:
        logger.info(".env ファイルが正常に読み込まれました。")
    else:
        logger.warning(".env ファイルが見つからないか、読み込めませんでした。")

    # config.yaml
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        logger.info("設定ファイル %s を読み込みました。", config_path)
    except FileNotFoundError:
        logger.critical("設定ファイル %s が見つかりません。", config_path)
        sys.exit(1)
    except Exception as e:
        logger.critical("設定読み込み中に予期せぬエラー: %s", e)
        sys.exit(1)

    # 環境変数の解決
    if "api_credentials" in raw and isinstance(raw["api_credentials"], dict):
        resolved = _resolve_env_credentials(raw["api_credentials"])
        raw.setdefault("api_credentials", {}).update(resolved)

    logger.info("設定ファイルの読み込みと環境変数の解決が完了しました。")

    # ログレベルの再設定（YAML に logging_settings があれば反映）
    log_settings = raw.get("logging_settings", {})
    if log_settings.get("log_level"):
        logger = setup_logger(
            log_file=log_file,
            log_level_str=log_settings["log_level"],
        )

    # AppConfig 構築
    ts = raw.get("target_stock", {})
    ai_raw = ts.get("ai_params", {})
    training = raw.get("training_settings", {})

    config = AppConfig(
        stock_code=ts.get("code", ""),
        stock_name=ts.get("name", ts.get("code", "")),
        feature_columns=ts.get("feature_columns", []),
        ai_params=AIParams(
            budget=ai_raw.get("budget", 100_000),
            future_days=ai_raw.get("future_days", 2),
            target_percent=ai_raw.get("target_percent", 0.7),
            threshold=ai_raw.get("threshold", 0.5),
            stop_loss_percent=ai_raw.get("stop_loss_percent", 3.0),
        ),
        tech_params=raw.get("technical_analysis_params", {}),
        training_settings=training,
        backtest_candidates=raw.get("backtest_candidates", []),
        model_path=Path(training.get("model_save_path", "models/stock_ai_model.pkl")),
        trade_log_path=Path("data/trade_log.csv"),
        raw=raw,
    )

    return config, logger
