"""通知モジュール (Slack)。"""

from __future__ import annotations

import json
import logging

import requests

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class SlackNotifier:
    """Slack Incoming Webhook を使った通知クラス。"""

    def __init__(self, webhook_url: str) -> None:
        if not webhook_url:
            raise ValueError("Slack webhook URL cannot be empty.")
        self.webhook_url = webhook_url
        logger.info("SlackNotifier が初期化されました。")

    def send_message(self, message_text: str) -> bool:
        if not message_text:
            logger.warning("送信するメッセージ内容が空です。送信をスキップします。")
            return False

        payload = {"text": message_text}
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if response.status_code == 200:
                logger.info("Slack への通知に成功しました。")
                return True
            logger.error("Slack 通知失敗: %s - %s", response.status_code, response.text[:300])
            return False
        except Exception as e:
            logger.error("Slack 送信エラー: %s", e)
            return False
