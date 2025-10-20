import requests
import json
import pandas as pd
import logging
import os
import time # リトライ時の待機のため
import logging
from logger_setup import APP_LOGGER_NAME # logger_setup.py からインポート
from datetime import datetime, timedelta # IDトークンの有効期限管理に使える (今回は簡易版)

# このモジュール用のロガーを取得 (main.py等で設定された共通ロガーを使う想定)
# from logger_setup import APP_LOGGER_NAME # APP_LOGGER_NAMEを共通定義からインポートする場合
# logger = logging.getLogger(APP_LOGGER_NAME)
# もし上記が別ファイルの場合、以下のように直接指定も可 (ただし、main で設定したロガーと同一名に)
logger = logging.getLogger('stock_analyzer_app')


class JQuantsFetcher:
    """
    J-Quants APIからデータを取得するためのクラス。
    認証（IDトークン取得）と各種データ取得メソッドを提供します。
    """
    TOKEN_AUTH_USER_URL = "https://api.jquants.com/v1/token/auth_user"
    TOKEN_AUTH_REFRESH_URL_BASE = "https://api.jquants.com/v1/token/auth_refresh"
    DAILY_QUOTES_URL = "https://api.jquants.com/v1/prices/daily_quotes"

    def __init__(self, mail_address=None, password=None, refresh_token=None, retry_count=3, retry_delay=5):
        """
        JQuantsFetcherを初期化します。

        Args:
            mail_address (str, optional): J-Quants APIのメールアドレス。
            password (str, optional): J-Quants APIのパスワード。
            refresh_token (str, optional): 事前に取得したリフレッシュトークン。推奨。
            retry_count (int): APIリクエスト失敗時のリトライ回数。
            retry_delay (int): リトライ時の待機時間（秒）。
        """
        self.mail_address = mail_address
        self.password = password
        self.provided_refresh_token = refresh_token # ユーザーが指定したリフレッシュトークン
        
        self.id_token = None
        self.id_token_expires_at = None # IDトークンの有効期限 (今回は簡易的に未実装)
        
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # 初期化時にIDトークンを取得試行
        # 優先順位: 1. 提供されたリフレッシュトークン, 2. メール/パスワード
        if self.provided_refresh_token:
            logger.info("提供されたリフレッシュトークンを使用してIDトークンを取得します。")
            self._refresh_id_token(self.provided_refresh_token)
        elif self.mail_address and self.password:
            logger.info("メールアドレスとパスワードを使用してIDトークンを取得します。")
            self._authenticate_and_get_id_token()
        else:
            logger.warning("認証情報（リフレッシュトークンまたはメール/パスワード）が提供されていません。IDトークンは未取得です。")

    def _make_request(self, method, url, headers=None, params=None, data=None, timeout=15):
        """
        指定されたHTTPメソッドでリクエストを送信し、レスポンスを返します。
        リトライ処理を含みます。
        """
        for attempt in range(self.retry_count + 1):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, headers=headers, params=params, timeout=timeout)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=headers, params=params, data=data, timeout=timeout)
                else:
                    logger.error(f"未対応のHTTPメソッドです: {method}")
                    return None
                
                response.raise_for_status() # HTTPエラーがあれば例外を発生させる
                return response
            
            except requests.exceptions.Timeout:
                logger.warning(f"リクエストがタイムアウトしました ({url})。試行 {attempt + 1}/{self.retry_count + 1}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTPエラーが発生しました ({url}): {e.response.status_code} - {e.response.reason}。試行 {attempt + 1}/{self.retry_count + 1}")
                try:
                    logger.debug(f"エラーレスポンス: {e.response.json()}")
                except json.JSONDecodeError:
                    logger.debug(f"エラーレスポンス (非JSON): {e.response.text}")
                # 4xx系のエラー(認証エラーなど)の場合はリトライしないことが多い
                if 400 <= e.response.status_code < 500:
                    logger.error("クライアントエラーのためリトライしません。")
                    return e.response # エラーレスポンスを返す
            except requests.exceptions.RequestException as e:
                logger.warning(f"リクエストエラーが発生しました ({url}): {e}。試行 {attempt + 1}/{self.retry_count + 1}")
            
            if attempt < self.retry_count:
                logger.info(f"{self.retry_delay}秒後にリトライします...")
                time.sleep(self.retry_delay)
            else:
                logger.error(f"最大リトライ回数 ({self.retry_count + 1}回) に達しました。")
                # 最後の試行でresponseが設定されていればそれを返す
                return response if 'response' in locals() and response is not None else None
        return None


    def _authenticate_and_get_id_token(self):
        if not self.mail_address or not self.password:
            logger.error("メールアドレスまたはパスワードが設定されていません。")
            return False

        logger.info(f"認証ユーザーAPI ({self.TOKEN_AUTH_USER_URL}) へリクエスト送信中 (ユーザー: {self.mail_address})...")
        auth_data = {"mailaddress": self.mail_address, "password": self.password}
        
        response_auth = self._make_request("POST", self.TOKEN_AUTH_USER_URL, data=json.dumps(auth_data))

        if response_auth and response_auth.status_code == 200:
            try:
                refresh_token_from_auth = response_auth.json().get("refreshToken")
                if not refresh_token_from_auth:
                    logger.error(f"リフレッシュトークンの取得に失敗しました（auth_userレスポンス）。レスポンス: {response_auth.text}")
                    return False
                logger.info("リフレッシュトークン（auth_user経由）の取得成功。")
                return self._refresh_id_token(refresh_token_from_auth)
            except json.JSONDecodeError:
                logger.error(f"auth_userレスポンスのJSONデコードに失敗: {response_auth.text}")
                return False
        else:
            status = response_auth.status_code if response_auth else "N/A"
            logger.error(f"auth_user APIによる認証に失敗しました。ステータス: {status}")
            return False

    def _refresh_id_token(self, refresh_token_to_use):
        """
        指定されたリフレッシュトークンを使用して新しいIDトークンを取得し、インスタンス変数に格納します。
        """
        if not refresh_token_to_use:
            logger.error("IDトークン更新のためのリフレッシュトークンがありません。")
            return False

        logger.info(f"リフレッシュトークンAPI ({self.TOKEN_AUTH_REFRESH_URL_BASE}) へリクエスト送信中...")
        refresh_url = f"{self.TOKEN_AUTH_REFRESH_URL_BASE}?refreshtoken={refresh_token_to_use}"
        
        response_id = self._make_request("POST", refresh_url)

        if response_id and response_id.status_code == 200:
            try:
                new_id_token = response_id.json().get("idToken")
                if new_id_token:
                    self.id_token = new_id_token
                    # self.id_token_expires_at = datetime.now() + timedelta(hours=1) # 仮: 1時間有効と想定
                    logger.info("新しいIDトークンの取得・更新に成功しました。")
                    return True
                else:
                    logger.error(f"レスポンスからIDトークンが見つかりませんでした。レスポンス: {response_id.text}")
                    return False
            except json.JSONDecodeError:
                logger.error(f"IDトークンAPIレスポンスのJSONデコードに失敗: {response_id.text}")
                return False
        else:
            status = response_id.status_code if response_id else "N/A"
            logger.error(f"IDトークンの更新に失敗しました。ステータス: {status}")
            return False

    def get_id_token(self):
        """
        現在のIDトークンを返します。
        将来的にはここで有効期限チェックと自動更新を行うことができます。
        """
        # if self.id_token and self.id_token_expires_at and datetime.now() >= self.id_token_expires_at:
        #     logger.info("IDトークンが期限切れです。更新を試みます...")
        #     if self.provided_refresh_token: # ユーザー指定のリフレッシュトークンがあればそれを使う
        #          self._refresh_id_token(self.provided_refresh_token)
        #     elif self.mail_address and self.password: # なければメール/パスワード認証から再試行 (非推奨)
        #          self._authenticate_and_get_id_token()
        #     else:
        #          logger.warning("IDトークン更新のための認証情報がありません。")

        return self.id_token

    def get_daily_stock_prices(self, stock_code, date_from_str=None, date_to_str=None):
        """
        指定された銘柄コードと期間の日次株価情報を取得し、Pandas DataFrameで返します。
        """
        current_id_token = self.get_id_token() # 有効期限チェックと更新を考慮する場合
        if not current_id_token:
            logger.error("IDトークンが利用できません。株価データを取得できません。")
            return None

        logger.info(f"日次株価情報取得開始 (銘柄: {stock_code}, 期間: {date_from_str}～{date_to_str})")
        headers = {'Authorization': f'Bearer {current_id_token}'}
        params = {"code": stock_code}
        if date_from_str:
            params["from"] = date_from_str
        if date_to_str:
            params["to"] = date_to_str
        
        if not date_from_str or not date_to_str: # J-Quantsでは日付指定がほぼ必須
            logger.warning("日付範囲(from, to)が指定されていません。API仕様を確認してください。")
            # デフォルトの日付範囲を設定するか、エラーとするかなどを検討
            # 例: 今日から30日前までなど (ただしAPIのデータカバレッジによる)
            # end_date = datetime.today()
            # start_date = end_date - timedelta(days=30)
            # params["from"] = start_date.strftime("%Y-%m-%d")
            # params["to"] = end_date.strftime("%Y-%m-%d")
            # logger.info(f"日付未指定のため、仮の日付範囲 {params['from']}～{params['to']} を使用します。")


        response = self._make_request("GET", self.DAILY_QUOTES_URL, headers=headers, params=params)

        if response and response.status_code == 200:
            try:
                data = response.json()
                if "daily_quotes" in data and data["daily_quotes"]:
                    df = pd.DataFrame(data["daily_quotes"])
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date')
                    logger.info(f"銘柄 {stock_code}: {len(df)} 件の株価データをDataFrameとして取得しました。")
                    return df
                elif "daily_quotes" in data and not data["daily_quotes"]: # データが空の場合
                    logger.info(f"銘柄 {stock_code}: 指定された期間の株価データは空でした。")
                    return pd.DataFrame() # 空のDataFrameを返す
                else:
                    logger.warning(f"銘柄 {stock_code}: レスポンスに 'daily_quotes' がないか、形式が不正です。レスポンス: {data}")
                    return None
            except json.JSONDecodeError:
                logger.error(f"日次株価レスポンスのJSONデコードに失敗: {response.text}")
                return None
        else:
            status = response.status_code if response else "N/A"
            logger.error(f"銘柄 {stock_code}: 日次株価データの取得に失敗しました。ステータス: {status}")
            return None


# --- このモジュールを直接実行した際のテストコード ---
if __name__ == '__main__':
    # 簡易的なロガー設定 (main.pyで設定されたものを使うのが本番)
    if not logging.getLogger(APP_LOGGER_NAME).hasHandlers(): # main.pyで未設定の場合のみ
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.StreamHandler()]) # コンソール出力のみ
    logger.info("--- JQuantsFetcherの単体テスト開始 ---")

    # .envファイルから認証情報を読み込む (このテストコード用に dotenv をインポート)
    from dotenv import load_dotenv
    load_dotenv() # プロジェクトルートの .env を読み込む

    TEST_MAIL_ADDRESS = os.getenv("JQUANTS_MAIL")
    TEST_PASSWORD = os.getenv("JQUANTS_PASSWORD")
    # TEST_REFRESH_TOKEN = os.getenv("JQUANTS_REFRESH_TOKEN") # リフレッシュトークンを使う場合

    if not TEST_MAIL_ADDRESS or not TEST_PASSWORD:
        logger.critical(".envファイルに JQUANTS_MAIL と JQUANTS_PASSWORD を設定してください。テストを中止します。")
    else:
        # メール/パスワードで初期化
        fetcher = JQuantsFetcher(mail_address=TEST_MAIL_ADDRESS, password=TEST_PASSWORD)
        
        # リフレッシュトークンで初期化する場合の例 (上記とどちらかを選択)
        # if TEST_REFRESH_TOKEN:
        #     fetcher = JQuantsFetcher(refresh_token=TEST_REFRESH_TOKEN)
        # else:
        #    logger.critical(".envファイルに JQUANTS_REFRESH_TOKEN を設定してください。")
        #    fetcher = None


        if fetcher and fetcher.get_id_token():
            logger.info(f"テスト用のIDトークン取得成功: {fetcher.get_id_token()[:30]}...") # 長いので一部表示

            # 株価取得テスト
            target_code = "7203" # トヨタ
            # J-Quants APIのデータカバレッジに合わせた日付を指定してください
            # 例: 2025年5月14日現在であれば、2025年2月16日以前のデータしか取得できない場合が多いです。
            # ご自身のプランで取得可能な日付を指定してください。
            date_from = "2025-02-01"
            date_to = "2025-02-10" 
            
            logger.info(f"銘柄 {target_code} の株価を期間 {date_from} ～ {date_to} で取得します...")
            df_prices = fetcher.get_daily_stock_prices(target_code, date_from, date_to)

            if df_prices is not None:
                if not df_prices.empty:
                    print("\n--- 取得データ (先頭5行) ---")
                    print(df_prices.head())
                    print(f"\n取得件数: {len(df_prices)}")
                else:
                    print(f"銘柄 {target_code}: データは取得できましたが、空でした。期間や銘柄コードを確認してください。")
            else:
                print(f"銘柄 {target_code}: 株価データの取得に失敗しました。")
        else:
            logger.error("テスト用のIDトークン取得に失敗しました。認証情報やAPIの状態を確認してください。")

    logger.info("--- JQuantsFetcherの単体テスト終了 ---")