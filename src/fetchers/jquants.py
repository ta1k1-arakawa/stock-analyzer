"""J-Quants API を使った株価データ取得モジュール。"""

from __future__ import annotations

import json
import logging
import os
import time

import pandas as pd
import requests

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class JQuantsFetcher:
    """
    J-Quants API からデータを取得するためのクラス。

    認証（IDトークン取得）と日次株価データ取得メソッドを提供する。
    """

    TOKEN_AUTH_USER_URL = "https://api.jquants.com/v1/token/auth_user"
    TOKEN_AUTH_REFRESH_URL_BASE = "https://api.jquants.com/v1/token/auth_refresh"
    DAILY_QUOTES_URL = "https://api.jquants.com/v1/prices/daily_quotes"

    def __init__(
        self,
        mail_address: str | None = None,
        password: str | None = None,
        retry_count: int = 3,
        retry_delay: int = 5,
    ) -> None:
        self.mail_address = mail_address
        self.password = password
        self.id_token: str | None = None
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        if self.mail_address and self.password:
            logger.info("メールアドレスとパスワードを使用してIDトークンを取得します。")
            self._authenticate_and_get_id_token()
        else:
            logger.warning("認証情報が提供されていません。IDトークンは未取得です。")

    # ------------------------------------------------------------------
    # HTTP リクエスト共通（リトライ付き）
    # ------------------------------------------------------------------

    def _make_request(
        self,
        method: str,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        data: str | None = None,
        timeout: int = 15,
    ) -> requests.Response | None:
        """リトライ付き HTTP リクエスト。"""
        response: requests.Response | None = None

        for attempt in range(self.retry_count + 1):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, headers=headers, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, headers=headers, params=params, data=data, timeout=timeout)
                else:
                    logger.error("未対応のHTTPメソッドです: %s", method)
                    return None

                response.raise_for_status()
                return response

            except requests.exceptions.Timeout:
                logger.warning("リクエストがタイムアウトしました (%s)。試行 %d/%d", url, attempt + 1, self.retry_count + 1)
            except requests.exceptions.HTTPError as e:
                logger.warning("HTTPエラー (%s): %s。試行 %d/%d", url, e, attempt + 1, self.retry_count + 1)
                if e.response is not None and 400 <= e.response.status_code < 500:
                    logger.error("クライアントエラーのためリトライしません。")
                    return e.response
            except requests.exceptions.RequestException as e:
                logger.warning("リクエストエラー (%s): %s。試行 %d/%d", url, e, attempt + 1, self.retry_count + 1)

            if attempt < self.retry_count:
                logger.info("%d 秒後にリトライします...", self.retry_delay)
                time.sleep(self.retry_delay)
            else:
                logger.error("最大リトライ回数 (%d 回) に達しました。", self.retry_count + 1)

        return response

    # ------------------------------------------------------------------
    # 認証
    # ------------------------------------------------------------------

    def _authenticate_and_get_id_token(self) -> bool:
        if not self.mail_address or not self.password:
            logger.error("メールアドレスまたはパスワードが設定されていません。")
            return False

        logger.info("認証ユーザーAPI へリクエスト送信中 (ユーザー: %s)...", self.mail_address)
        auth_data = json.dumps({"mailaddress": self.mail_address, "password": self.password})
        resp = self._make_request("POST", self.TOKEN_AUTH_USER_URL, data=auth_data)

        if resp and resp.status_code == 200:
            try:
                refresh_token = resp.json().get("refreshToken")
                if not refresh_token:
                    logger.error("リフレッシュトークンの取得に失敗しました。レスポンス: %s", resp.text)
                    return False
                logger.info("リフレッシュトークン取得成功。")
                return self._refresh_id_token(refresh_token)
            except json.JSONDecodeError:
                logger.error("auth_user レスポンスの JSON デコードに失敗: %s", resp.text)
                return False

        status = resp.status_code if resp else "N/A"
        logger.error("auth_user API による認証に失敗しました。ステータス: %s", status)
        return False

    def _refresh_id_token(self, refresh_token: str) -> bool:
        if not refresh_token:
            logger.error("IDトークン更新のためのリフレッシュトークンがありません。")
            return False

        refresh_url = f"{self.TOKEN_AUTH_REFRESH_URL_BASE}?refreshtoken={refresh_token}"
        resp = self._make_request("POST", refresh_url)

        if resp and resp.status_code == 200:
            try:
                new_id_token = resp.json().get("idToken")
                if new_id_token:
                    self.id_token = new_id_token
                    logger.info("新しいIDトークンの取得・更新に成功しました。")
                    return True
                logger.error("レスポンスからIDトークンが見つかりませんでした。")
                return False
            except json.JSONDecodeError:
                logger.error("IDトークンAPIレスポンスのJSONデコードに失敗: %s", resp.text)
                return False

        status = resp.status_code if resp else "N/A"
        logger.error("IDトークンの更新に失敗しました。ステータス: %s", status)
        return False

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def get_daily_stock_prices(
        self,
        stock_code: str,
        date_from_str: str | None = None,
        date_to_str: str | None = None,
    ) -> pd.DataFrame | None:
        """指定銘柄・期間の日次株価データを DataFrame で返す。"""
        if not self.id_token:
            logger.error("IDトークンが利用できません。株価データを取得できません。")
            return None

        logger.info("日次株価情報取得開始 (銘柄: %s, 期間: %s〜%s)", stock_code, date_from_str, date_to_str)
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params: dict[str, str] = {"code": stock_code}
        if date_from_str:
            params["from"] = date_from_str
        if date_to_str:
            params["to"] = date_to_str

        if not date_from_str or not date_to_str:
            logger.warning("日付範囲(from, to)が指定されていません。")

        resp = self._make_request("GET", self.DAILY_QUOTES_URL, headers=headers, params=params)

        if resp and resp.status_code == 200:
            try:
                data = resp.json()
                if "daily_quotes" in data and data["daily_quotes"]:
                    df = pd.DataFrame(data["daily_quotes"])
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.set_index("Date")
                    logger.info("銘柄 %s: %d 件の株価データを取得しました。", stock_code, len(df))
                    return df
                if "daily_quotes" in data:
                    logger.info("銘柄 %s: 指定された期間の株価データは空でした。", stock_code)
                    return pd.DataFrame()
                logger.warning("銘柄 %s: レスポンスに 'daily_quotes' がないか、形式が不正です。", stock_code)
                return None
            except json.JSONDecodeError:
                logger.error("日次株価レスポンスの JSON デコードに失敗: %s", resp.text)
                return None

        status = resp.status_code if resp else "N/A"
        logger.error("銘柄 %s: 日次株価データの取得に失敗しました。ステータス: %s", stock_code, status)
        return None
