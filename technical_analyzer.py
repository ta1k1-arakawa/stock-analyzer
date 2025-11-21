import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger('stock_analyzer_app')

def calculate_indicators(df_ohlcv: pd.DataFrame, analysis_params: dict) -> pd.DataFrame:
    """
    テクニカル指標を計算してDataFrameに追加します。
    
    注意: config.yamlのtechnical_analysis_paramsに以下のパラメータを設定してください:
    - sma: { short_period: 5, long_period: 25 }
    - rsi: { period: 14 }
    - macd: { fast_period: 12, slow_period: 26, signal_period: 9 }
    - atr: { period: 14 }
    - bbands: { length: 20, std: 2 }
    
    Args:
        df_ohlcv (pd.DataFrame): OHLCV形式の株価データ
        analysis_params (dict): テクニカル指標の計算パラメータ
        
    Returns:
        pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    if df_ohlcv is None or df_ohlcv.empty:
        logger.warning("入力されたDataFrameがNoneまたは空のため、テクニカル指標は計算できません。")
        return df_ohlcv.copy() if df_ohlcv is not None else pd.DataFrame() 

    df_to_analyze = df_ohlcv.copy() 

    logger.info(f"テクニカル指標の計算を開始します。対象期間: {df_to_analyze.index.min()} から {df_to_analyze.index.max() if not df_to_analyze.empty else 'N/A'}。")
    logger.debug(f"使用する分析パラメータ: {analysis_params}")

    try:
        # SMA (単純移動平均) の計算
        if 'sma' in analysis_params and isinstance(analysis_params['sma'], dict):
            sma_p = analysis_params['sma']
            if 'short_period' in sma_p and isinstance(sma_p['short_period'], int):
                df_to_analyze.ta.sma(length=sma_p['short_period'], close=df_to_analyze['Close'], append=True, col_names=(f"SMA_{sma_p['short_period']}",))
            if 'long_period' in sma_p and isinstance(sma_p['long_period'], int):
                df_to_analyze.ta.sma(length=sma_p['long_period'], close=df_to_analyze['Close'], append=True, col_names=(f"SMA_{sma_p['long_period']}",))
            logger.debug("SMAを計算・追加しました。")

        # RSI (相対力指数) の計算
        if 'rsi' in analysis_params and isinstance(analysis_params['rsi'], dict):
            rsi_p = analysis_params['rsi']
            if 'period' in rsi_p and isinstance(rsi_p['period'], int):
                df_to_analyze.ta.rsi(length=rsi_p['period'], close=df_to_analyze['Close'], append=True, col_names=(f"RSI_{rsi_p['period']}",))
            logger.debug("RSIを計算・追加しました。")
            
        # MACD の計算
        if 'macd' in analysis_params and isinstance(analysis_params['macd'], dict):
            macd_p = analysis_params['macd']
            if all(k in macd_p and isinstance(macd_p[k], int) for k in ['fast_period', 'slow_period', 'signal_period']):
                df_to_analyze.ta.macd(
                    fast=macd_p['fast_period'],
                    slow=macd_p['slow_period'],
                    signal=macd_p['signal_period'],
                    close=df_to_analyze['Close'],
                    append=True
                )
            logger.debug("MACDを計算・追加しました。")

        # ATR (Average True Range) の計算
        if 'atr' in analysis_params and isinstance(analysis_params['atr'], dict):
            atr_p = analysis_params['atr']
            if 'period' in atr_p and isinstance(atr_p['period'], int):
                df_to_analyze.ta.atr(length=atr_p['period'], append=True, col_names=(f"ATR_{atr_p['period']}",))
            logger.debug("ATRを計算・追加しました。")

        # ボリンジャーバンド の計算
        if 'bbands' in analysis_params and isinstance(analysis_params['bbands'], dict):
            bb_p = analysis_params['bbands']
            if 'length' in bb_p and isinstance(bb_p['length'], int) and 'std' in bb_p:
                df_to_analyze.ta.bbands(
                    length=bb_p['length'],
                    std=bb_p['std'],
                    close=df_to_analyze['Close'],
                    append=True
                )
            logger.debug("ボリンジャーバンドを計算・追加しました。")

        # ラグ特徴量の追加
        df_to_analyze['Lag_Close_1'] = df_to_analyze['Close'].shift(1)
        df_to_analyze['Lag_Close_3'] = df_to_analyze['Close'].shift(3)
        
        # RSI_14が計算済みの場合のみラグRSIを追加
        if 'RSI_14' in df_to_analyze.columns:
            df_to_analyze['Lag_RSI_1'] = df_to_analyze['RSI_14'].shift(1)
            logger.debug("ラグRSI特徴量を追加しました。")
        
        logger.debug("ラグ特徴量（Lag_Close_1, Lag_Close_3）を追加しました。")

        logger.info("テクニカル指標の計算が完了しました。")

    except AttributeError as e:
        logger.error(f"テクニカル指標の計算中に属性エラーが発生しました: {e}。pandas-taが正しくインストールされているか、列名が正しいか確認してください。")
        logger.exception("スタックトレース:")
        return df_ohlcv.copy()
    except Exception as e:
        logger.error(f"テクニカル指標の計算中に予期せぬエラーが発生しました: {e}")
        logger.exception("スタックトレース:")
        return df_ohlcv.copy()

    return df_to_analyze

# --- このモジュールを単体で実行した際のテストコード ---
if __name__ == '__main__':
    if not logging.getLogger('stock_analyzer_app').hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    logger.info("--- technical_analyzer.py 単体テスト開始 ---")

    # テスト用のダミー株価データ (Pandas DataFrame) を作成
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                 '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                 '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                                 '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
                                 '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25',
                                 '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-29', '2023-01-30']),
        'Open':  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 102, 103, 105, 107, 108, 106, 105, 107, 109],
        'High':  [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 104, 105, 107, 109, 110, 108, 107, 109, 111],
        'Low':   [99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99,  98,  100, 101, 103, 105, 106, 104, 103, 105, 107],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 101, 103, 104, 106, 108, 107, 105, 106, 108, 110],
        'Volume':[1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,1950,1850,1750,1650,1550,1450,1350,1250,1150,1050,1200,1350,1550,1750,1800,1600,1500,1700,1900]
    }
    test_df = pd.DataFrame(data)
    test_df = test_df.set_index('Date')

    logger.info("テスト用DataFrame作成完了:")
    print(test_df.head())

    # テスト用の分析パラメータ
    test_analysis_params = {
        'sma': {'short_period': 5, 'long_period': 10},
        'rsi': {'period': 7},
        'macd': {'fast_period': 6, 'slow_period': 13, 'signal_period': 4},
        'atr': {'period': 14},
        'bbands': {'length': 20, 'std': 2},
    }

    # 関数呼び出し
    df_with_ta = calculate_indicators(test_df, test_analysis_params)

    logger.info("テクニカル指標計算後のDataFrame:")
    if df_with_ta is not None:
        print(df_with_ta.tail())
        print("\n追加された列名:")
        print(df_with_ta.columns)
    else:
        print("テクニカル指標の計算に失敗しました。")

    logger.info("--- technical_analyzer.py 単体テスト終了 ---")