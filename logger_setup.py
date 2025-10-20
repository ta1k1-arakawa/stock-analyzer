# logger_setup.py

import logging
import logging.handlers
import sys

APP_LOGGER_NAME = 'stock_analyzer_app' # アプリケーション共通のロガー名

def setup_logger(
    log_file_path='stock_analyzer.log',
    log_level_str='INFO',
    logger_name=APP_LOGGER_NAME,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
):
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        # デフォルトレベルを設定するか、エラーを発生させる
        print(f"警告: 無効なログレベル '{log_level_str}'. INFOレベルを使用します。", file=sys.stderr)
        numeric_level = logging.INFO

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"エラー: ログファイルハンドラの設定に失敗しました: {e}", file=sys.stderr)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level) # コンソールも同じレベルにするか、DEBUGなど別のレベルにも設定可能
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':
    # このモジュール単体でテストする場合
    # config.yaml から読み込む logging_settings を想定
    test_log_settings = {
        'log_file': 'test_logger.log',
        'log_level': 'DEBUG'
    }

    logger = setup_logger(
        log_file_path=test_log_settings['log_file'],
        log_level_str=test_log_settings['log_level'],
        logger_name="TestLoggerFromSetup"
    )
    logger.debug("これはテスト用のデバッグメッセージです。")
    logger.info("これはテスト用の情報メッセージです。")
    print(f"テストログは '{test_log_settings['log_file']}' とコンソールに出力されました。")