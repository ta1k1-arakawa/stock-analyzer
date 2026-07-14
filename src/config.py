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
from dataclasses import dataclass, field, replace
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
    commission_percent: float = 0.0
    entry_slippage_percent: float = 0.03
    exit_slippage_percent: float = 0.03
    stop_slippage_percent: float = 0.10


@dataclass
class StockConfig:
    """学習・バックテスト・予測で共通して扱う銘柄設定。"""

    stock_code: str
    stock_name: str
    ai_params: AIParams
    model_path: Path
    trade_log_path: Path
    notify_slack: bool = False


@dataclass
class AppConfig:
    """アプリケーション全体の設定を保持する。"""

    stock_code: str = ""
    stock_name: str = ""
    feature_columns: list[str] = field(default_factory=list)
    ai_params: AIParams = field(default_factory=AIParams)
    tech_params: dict[str, Any] = field(default_factory=dict)
    training_settings: dict[str, Any] = field(default_factory=dict)
    stocks: list[StockConfig] = field(default_factory=list)
    model_path: Path = Path()
    trade_log_path: Path = Path()
    selection_path: Path = Path("data/backtest_selection.yaml")

    # 元の生 YAML 辞書（後方互換用）
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def for_stock(self, stock: StockConfig) -> "AppConfig":
        """共通設定へ1銘柄の実行時設定を重ねる。"""

        return replace(
            self,
            stock_code=stock.stock_code,
            stock_name=stock.stock_name,
            ai_params=stock.ai_params,
            model_path=stock.model_path,
            trade_log_path=stock.trade_log_path,
            stocks=[],
        )


# ---------------------------------------------------------------------------
# ロガー設定
# ---------------------------------------------------------------------------

def setup_logger(
    log_file: str = "app.log",
    log_level_str: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
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
    if console:
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


def _build_ai_params(raw: dict[str, Any], base: AIParams | None = None) -> AIParams:
    """省略値を ``base`` から引き継いで AI パラメータを構築する。"""

    defaults = base or AIParams()
    return AIParams(
        budget=raw.get("budget", defaults.budget),
        future_days=raw.get("future_days", defaults.future_days),
        target_percent=raw.get("target_percent", defaults.target_percent),
        threshold=raw.get("threshold", defaults.threshold),
        stop_loss_percent=raw.get("stop_loss_percent", defaults.stop_loss_percent),
        commission_percent=raw.get("commission_percent", defaults.commission_percent),
        entry_slippage_percent=raw.get("entry_slippage_percent", defaults.entry_slippage_percent),
        exit_slippage_percent=raw.get("exit_slippage_percent", defaults.exit_slippage_percent),
        stop_slippage_percent=raw.get("stop_slippage_percent", defaults.stop_slippage_percent),
    )


def _stock_model_path(code: str) -> Path:
    return Path(f"models/stock_ai_model_{code}.pkl")


def _stock_trade_log_path(code: str) -> Path:
    return Path(f"data/trade_log_{code}.csv")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _load_selection_result(path: Path, logger: logging.Logger) -> dict[str, Any]:
    if not path.exists():
        logger.warning(
            "バックテスト結果 %s がありません。銘柄別ルールには共通設定を使います。",
            path,
        )
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            result = yaml.safe_load(f) or {}
    except Exception:
        logger.exception("バックテスト結果 %s を読み込めませんでした。", path)
        return {}

    if not isinstance(result, dict):
        logger.warning("バックテスト結果 %s の形式が不正です。", path)
        return {}
    return result


def load_app(
    log_file: str = "app.log",
    config_path: str = "config.yaml",
    console: bool = True,
) -> tuple[AppConfig, logging.Logger]:
    """
    .env と config.yaml を読み込み、ロガーを初期化して ``(AppConfig, Logger)`` を返す。

    失敗時は ``sys.exit(1)`` する。
    """

    # .env
    dotenv_ok = load_dotenv()

    # まずロガーを仮設定
    logger = setup_logger(log_file=log_file, console=console)

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
            console=console,
        )

    # AppConfig 構築: 銘柄は stocks の1か所だけを情報源にする。
    common_ai_params = _build_ai_params(raw.get("ai_params", {}))
    training = raw.get("training_settings", {})
    backtest_raw = raw.get("backtest_settings", {})
    selection_path = Path(backtest_raw.get("result_path", "data/backtest_selection.yaml"))
    selection = _load_selection_result(selection_path, logger)
    accepted_rules = selection.get("rules", {})
    if not isinstance(accepted_rules, dict):
        accepted_rules = {}

    stocks: list[StockConfig] = []
    seen_codes: set[str] = set()
    for item in raw.get("stocks", []):
        if not isinstance(item, dict):
            logger.warning("stocks の不正な設定をスキップします: %r", item)
            continue

        code = str(item.get("code", "")).strip()
        if not code or code in seen_codes:
            logger.warning("stocks の空または重複した銘柄をスキップします: %r", code)
            continue

        item_ai_raw = item.get("ai_params", {})
        if not isinstance(item_ai_raw, dict):
            logger.warning("%s の ai_params が不正なため共通設定を使用します。", code)
            item_ai_raw = {}
        stock_ai_params = _build_ai_params(item_ai_raw, common_ai_params)

        rule = accepted_rules.get(code, {})
        if isinstance(rule, dict):
            stock_ai_params = _build_ai_params(rule, stock_ai_params)

        stocks.append(
            StockConfig(
                stock_code=code,
                stock_name=str(item.get("name", code)),
                ai_params=stock_ai_params,
                model_path=_stock_model_path(code),
                trade_log_path=_stock_trade_log_path(code),
                notify_slack=_as_bool(item.get("notify_slack"), False),
            )
        )
        seen_codes.add(code)

    if not stocks:
        logger.critical("config.yaml の stocks に有効な銘柄がありません。")

    config = AppConfig(
        feature_columns=raw.get("feature_columns", []),
        ai_params=common_ai_params,
        tech_params=raw.get("technical_analysis_params", {}),
        training_settings=training,
        stocks=stocks,
        selection_path=selection_path,
        raw=raw,
    )

    return config, logger
