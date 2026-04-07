"""Yahoo Finance を使った株価データ取得モジュール。"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class YFinanceFetcher:
    """yfinance ライブラリ経由で株価データを取得する。認証不要。"""

    def get_daily_stock_prices(
        self,
        stock_code: str,
        date_from_str: str | None = None,
        date_to_str: str | None = None,
    ) -> pd.DataFrame | None:
        """日次株価データを取得して DataFrame で返す。"""
        # 日本株の場合、証券コードの末尾に ".T" をつける
        ticker_symbol = f"{stock_code}.T" if not stock_code.endswith(".T") else stock_code

        logger.info("yfinance でデータ取得開始: %s (%s 〜 %s)", ticker_symbol, date_from_str, date_to_str)

        try:
            df = yf.download(ticker_symbol, start=date_from_str, end=date_to_str, progress=False)

            if df is None or df.empty:
                logger.warning("データが見つかりませんでした: %s", ticker_symbol)
                return None

            # マルチインデックスの場合の対応
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index)

            logger.info("取得成功: %d 件", len(df))
            return df

        except Exception as e:
            logger.error("yfinance 取得エラー: %s", e)
            return None
