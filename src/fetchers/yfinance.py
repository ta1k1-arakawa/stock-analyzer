"""Yahoo Finance chart API based stock price fetcher."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class YFinanceFetcher:
    """Fetch daily OHLCV data from Yahoo Finance without the yfinance package."""

    _CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    _EXCHANGE_TZ = ZoneInfo("Asia/Tokyo")
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }

    def get_daily_stock_prices(
        self,
        stock_code: str,
        date_from_str: str | None = None,
        date_to_str: str | None = None,
    ) -> pd.DataFrame | None:
        """Return daily adjusted OHLCV data as a DataFrame."""
        ticker_symbol = f"{stock_code}.T" if not stock_code.endswith(".T") else stock_code

        logger.info(
            "Yahoo Finance chart API でデータ取得開始: %s (%s 〜 %s)",
            ticker_symbol,
            date_from_str,
            date_to_str,
        )

        try:
            params = {
                "period1": self._date_to_unix(date_from_str, datetime(1970, 1, 1)),
                "period2": self._date_to_unix(
                    date_to_str,
                    datetime.now(timezone.utc) + timedelta(days=1),
                ),
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
            response = requests.get(
                self._CHART_URL.format(ticker=ticker_symbol),
                params=params,
                headers=self._HEADERS,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            df = self._parse_chart_response(data, ticker_symbol)
            if df is None or df.empty:
                logger.warning("データが見つかりませんでした: %s", ticker_symbol)
                return None

            logger.info("取得成功: %d 件", len(df))
            return df

        except Exception as e:
            logger.error("Yahoo Finance chart API 取得エラー: %s", e)
            return None

    @staticmethod
    def _date_to_unix(date_str: str | None, default: datetime) -> int:
        if date_str:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=YFinanceFetcher._EXCHANGE_TZ)
        else:
            dt = default
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    def _parse_chart_response(data: dict[str, Any], ticker_symbol: str) -> pd.DataFrame | None:
        chart = data.get("chart", {})
        if chart.get("error"):
            logger.error("Yahoo Finance chart API エラー: %s", chart["error"])
            return None

        results = chart.get("result") or []
        if not results:
            return None

        result = results[0]
        timestamps = result.get("timestamp") or []
        quotes = result.get("indicators", {}).get("quote") or []
        if not timestamps or not quotes:
            return None

        quote = quotes[0]
        df = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert("Asia/Tokyo")
            .tz_localize(None)
            .normalize(),
        )

        adjclose_blocks = result.get("indicators", {}).get("adjclose") or []
        if adjclose_blocks and adjclose_blocks[0].get("adjclose") is not None:
            adjusted_close = pd.Series(adjclose_blocks[0]["adjclose"], index=df.index)
            factor = adjusted_close / df["Close"]
            for col in ("Open", "High", "Low"):
                df[col] = df[col] * factor
            df["Close"] = adjusted_close

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
        df.index.name = None

        logger.debug("Yahoo Finance chart API parsed %s rows for %s", len(df), ticker_symbol)
        return df
