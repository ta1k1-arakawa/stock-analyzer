"""ルールベースのシグナル評価モジュール。

現在は AI 予測（LightGBM）に移行済みだが、
将来的にルールベースとのハイブリッド判定に再利用する可能性がある。
"""

from __future__ import annotations

import logging

import pandas as pd

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def evaluate_signals(
    df_with_indicators: pd.DataFrame,
    signal_rules: list[dict],
) -> tuple[str, dict]:
    """
    テクニカル指標が追加された DataFrame とルール定義に基づき売買シグナルを判定する。

    Args:
        df_with_indicators: テクニカル指標列を含む株価 DataFrame。
        signal_rules: config.yaml から渡されるシグナルルールのリスト。

    Returns:
        (シグナル文字列, 判定詳細辞書) のタプル。
        シグナルは ``"BUY"`` / ``"SELL"`` / ``"HOLD"`` のいずれか。
    """
    if df_with_indicators is None or df_with_indicators.empty or len(df_with_indicators) < 2:
        logger.warning("シグナル判定に必要なデータが不足しているため、HOLD とします。")
        return "HOLD", {"message": "データ不足のため判定不可"}

    latest = df_with_indicators.iloc[-1]
    previous = df_with_indicators.iloc[-2]

    final_signal = "HOLD"
    evaluation_details: dict = {"message": "合致するシグナルルールなし"}

    for rule in signal_rules:
        rule_name = rule.get("rule_name", "無名ルール")
        action = rule.get("action")
        conditions = rule.get("conditions", [])

        if not action or not conditions:
            logger.warning("ルール '%s' に action または conditions が定義されていません。スキップします。", rule_name)
            continue

        logger.debug("ルール '%s' (アクション: %s) の評価を開始します。", rule_name, action)
        all_conditions_met = True
        condition_str_current = ""

        try:
            for condition_str in conditions:
                condition_str_current = condition_str
                local_vars = {"latest": latest, "previous": previous, "pd": pd}
                evaluated_value = eval(condition_str, {"__builtins__": {}}, local_vars)
                condition_is_true = bool(evaluated_value)
                logger.debug("  条件 '%s' → %s", condition_str, condition_is_true)

                if not condition_is_true:
                    all_conditions_met = False
                    break

            if all_conditions_met:
                final_signal = action
                evaluation_details = {
                    "rule_applied": rule_name,
                    "message": f"ルール '{rule_name}' に合致しました。",
                }
                for indicator in ("RSI_14", "SMA_5", "SMA_25"):
                    if indicator in latest:
                        evaluation_details[indicator] = round(latest[indicator], 2)

                logger.info("ルール '%s' に合致。シグナル: %s", rule_name, final_signal)
                break

        except KeyError as e:
            logger.error(
                "ルール '%s' の条件式評価中にキーエラー: %s。条件式: '%s'", rule_name, e, condition_str_current
            )
        except Exception as e:
            logger.error("ルール '%s' の条件式評価中に予期せぬエラー: %s。条件式: '%s'", rule_name, e, condition_str_current)
            logger.exception("スタックトレース:")

    if final_signal == "HOLD":
        if not signal_rules:
            logger.info("評価するシグナルルールが設定されていませんでした。")
            evaluation_details = {"message": "シグナルルール未設定"}
        elif evaluation_details.get("message") == "合致するシグナルルールなし":
            logger.info("いずれのシグナルルールにも合致しませんでした。")

    return final_signal, evaluation_details
