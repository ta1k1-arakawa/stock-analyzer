# line_notifier.py (例外処理部分の修正案)

import logging
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest
)
from linebot.v3.messaging.models import (
    TextMessage
)
# from linebot.v3.exceptions import ApiException # ← これが見つからないのでコメントアウトまたは削除

# requests ライブラリの例外を直接捕捉することを検討
import requests # requestsをインポート

logger = logging.getLogger('stock_analyzer_app')

class LineNotifier:
    def __init__(self, channel_access_token: str):
        if not channel_access_token:
            logger.error("チャネルアクセストークンが提供されていません。LineNotifierを初期化できません。")
            raise ValueError("Channel access token cannot be empty.")
        self.configuration = Configuration(access_token=channel_access_token)
        logger.info("LineNotifierが初期化されました。")

    def send_push_message(self, user_id: str, message_text: str) -> bool:
        if not user_id:
            logger.error("送信先のユーザーIDが指定されていません。")
            return False
        if not message_text:
            logger.warning("送信するメッセージ内容が空です。送信をスキップします。")
            return False
        
        message_text_to_send = message_text
        if len(message_text) > 4800:
             logger.warning(f"メッセージが長すぎます ({len(message_text)}文字)。先頭4800文字に短縮します。")
             message_text_to_send = message_text[:4800] + "..."

        try:
            with ApiClient(self.configuration) as api_client:
                messaging_api_instance = MessagingApi(api_client)
                
                log_message_preview = message_text_to_send[:50].replace('\n', ' ')
                logger.info(f"ユーザーID '{user_id}' にメッセージを送信します: 「{log_message_preview}...」")
                
                push_message_request_object = PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=message_text_to_send)]
                )
                
                messaging_api_instance.push_message(push_message_request_object)
                
                logger.info(f"ユーザーID '{user_id}' へのメッセージ送信に成功しました。")
                return True
        # ★★★ 例外処理の修正箇所 ★★★
        except requests.exceptions.HTTPError as e: # requestsライブラリのHTTPErrorを捕捉
            logger.error(f"LINE Messaging APIへのリクエストでHTTPエラーが発生しました。")
            if e.response is not None:
                logger.error(f"ステータスコード: {e.response.status_code}")
                # e.response.text や e.response.json() で詳細を取得できることがある
                try:
                    error_details = e.response.json() # JSON形式でエラー詳細が返ってくる場合
                    logger.error(f"エラー詳細 (JSON): {error_details}")
                except ValueError: # JSONでなければテキストで
                    logger.error(f"エラー詳細 (Text): {e.response.text[:500]}...") # 長すぎる可能性があるので一部表示
            else:
                logger.error(f"エラーレスポンスがありません。詳細: {e}")
            logger.exception("スタックトレース (HTTPError):")
            return False
        except Exception as e: # その他の予期せぬエラーをキャッチ
            logger.error(f"LINEメッセージ送信中に予期せぬエラーが発生しました: {e}")
            logger.exception("スタックトレース (予期せぬエラー):")
            return False

# --- 単体テストコード (変更なし) ---
if __name__ == '__main__':
    # ... (前回と同じテストコード) ...
    import sys
    if not logging.getLogger('stock_analyzer_app').hasHandlers():
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    logger.info("--- line_notifier.py 単体テスト開始 ---")

    from dotenv import load_dotenv
    import os
    import pandas as pd 
    
    if load_dotenv():
        logger.info(".envファイルから環境変数をロードしました。")
    else:
        logger.warning(".envファイルが見つかりませんでした。")

    TEST_LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    TEST_LINE_USER_ID = os.getenv("LINE_USER_ID")

    if not TEST_LINE_CHANNEL_ACCESS_TOKEN or not TEST_LINE_USER_ID:
        logger.critical(".envファイルに LINE_CHANNEL_ACCESS_TOKEN と LINE_USER_ID を設定してください。")
    else:
        try:
            notifier = LineNotifier(channel_access_token=TEST_LINE_CHANNEL_ACCESS_TOKEN)
            now_jst_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S %Z')
            test_message_body = f"これは line_notifier.py からのテストメッセージです。(例外修正版)\n現在時刻: {now_jst_str}"
            logger.info(f"テストメッセージをユーザーID '{TEST_LINE_USER_ID}' に送信します。")
            success = notifier.send_push_message(user_id=TEST_LINE_USER_ID, message_text=test_message_body)
            if success:
                logger.info("テストメッセージの送信に成功しました！LINEアプリで確認してください。")
            else:
                logger.error("テストメッセージの送信に失敗しました。ログを確認してください。")
        except ValueError as ve:
            logger.error(f"LineNotifierの初期化に失敗: {ve}")
        except Exception as e:
            logger.error(f"単体テスト中に予期せぬエラー: {e}")
            logger.exception("スタックトレース:")
    logger.info("--- line_notifier.py 単体テスト終了 ---")