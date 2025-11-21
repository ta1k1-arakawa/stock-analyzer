# yfinance_fetcher.py
import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger('stock_analyzer_app')

class YFinanceFetcher:
    def __init__(self):
        # yfinanceは認証不要なので初期化は空でOK
        pass

    def get_id_token(self):
        # 互換性のためのダミーメソッド
        return True

    def get_daily_stock_prices(self, stock_code, date_from_str=None, date_to_str=None):
        """
        Yahoo Financeから株価データを取得する
        """
        # 日本株の場合、証券コードの末尾に ".T" をつける必要がある
        ticker_symbol = f"{stock_code}.T" if not stock_code.endswith(".T") else stock_code
        
        logger.info(f"yfinanceでデータ取得開始: {ticker_symbol} ({date_from_str} 〜 {date_to_str})")
        
        try:
            # yfinanceでデータ取得
            df = yf.download(ticker_symbol, start=date_from_str, end=date_to_str, progress=False)
            
            if df is None or df.empty:
                logger.warning(f"データが見つかりませんでした: {ticker_symbol}")
                return None

            # カラム名の調整 (yfinanceは 'Adj Close' などがあるが、今回は 'Close' を使う)
            # マルチインデックスの場合の対応（バージョンによって挙動が違うため念のため）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # J-Quantsに合わせて必要な列だけ残す（Volumeなど）
            # yfinanceは: Open, High, Low, Close, Adj Close, Volume
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 日付インデックスの型を確認
            df.index = pd.to_datetime(df.index)

            logger.info(f"取得成功: {len(df)}件")
            return df

        except Exception as e:
            logger.error(f"yfinance取得エラー: {e}")
            return None

# テスト用
if __name__ == "__main__":
    fetcher = YFinanceFetcher()
    df = fetcher.get_daily_stock_prices("7203", "2023-01-01", "2023-12-31")
    print(df.head())