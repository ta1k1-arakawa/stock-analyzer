"""LINE Messaging API によるプッシュ通知モジュール。"""

from __future__ import annotations

import logging

import requests
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
)
from linebot.v3.messaging.models import TextMessage

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# LINE メッセージテキストの最大長
_MAX_MESSAGE_LENGTH = 4800


class LineNotifier:
    """LINE Messaging API を使ったプッシュ通知クラス。"""

    def __init__(self, channel_access_token: str) -> None:
        if not channel_access_token:
            raise ValueError("Channel access token cannot be empty.")
        self.configuration = Configuration(access_token=channel_access_token)
        logger.info("LineNotifier が初期化されました。")

    def send_push_message(self, user_id: str, message_text: str) -> bool:
        """ユーザーにテキストメッセージをプッシュ送信する。"""
        if not user_id:
            logger.error("送信先のユーザーIDが指定されていません。")
            return False
        if not message_text:
            logger.warning("送信するメッセージ内容が空です。送信をスキップします。")
            return False

        text = message_text
        if len(text) > _MAX_MESSAGE_LENGTH:
            logger.warning("メッセージが長すぎます (%d 文字)。先頭 %d 文字に短縮します。", len(text), _MAX_MESSAGE_LENGTH)
            text = text[:_MAX_MESSAGE_LENGTH] + "..."

        try:
            with ApiClient(self.configuration) as api_client:
                api = MessagingApi(api_client)
                preview = text[:50].replace("\n", " ")
                logger.info("ユーザーID '%s' にメッセージを送信します: 「%s...」", user_id, preview)

                api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=text)],
                    )
                )
                logger.info("ユーザーID '%s' へのメッセージ送信に成功しました。", user_id)
                return True

        except requests.exceptions.HTTPError as e:
            logger.error("LINE Messaging API への HTTP エラー。")
            if e.response is not None:
                logger.error("ステータスコード: %s", e.response.status_code)
                try:
                    logger.error("エラー詳細 (JSON): %s", e.response.json())
                except ValueError:
                    logger.error("エラー詳細 (Text): %s", e.response.text[:500])
            logger.exception("スタックトレース:")
            return False
        except Exception as e:
            logger.error("LINE メッセージ送信中に予期せぬエラー: %s", e)
            logger.exception("スタックトレース:")
            return False
