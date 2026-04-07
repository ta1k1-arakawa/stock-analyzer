"""データ取得の共通インターフェース定義。"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class StockDataFetcher(Protocol):
    """株価データ取得クラスが実装すべきインターフェース。"""

    def get_daily_stock_prices(
        self,
        stock_code: str,
        date_from_str: str | None = None,
        date_to_str: str | None = None,
    ) -> pd.DataFrame | None:
        """日次株価データを DataFrame で返す。取得失敗時は None。"""
        ...
