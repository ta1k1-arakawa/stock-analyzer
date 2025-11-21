# config_loader.py

import sys
import os
import yaml
from dotenv import load_dotenv
import logging

from logger_setup import setup_logger, APP_LOGGER_NAME

def load_config_and_logger(log_file_name='app.log'):
    """
    config.yaml と .env を読み込み、ロガーを設定する。
    main.py と train.py から呼び出される共通関数。
    """
    
    # .env の読み込み
    load_dotenv_success = load_dotenv()
    
    # ロガーの設定 
    CONFIG_FILE_PATH = 'config.yaml'
    temp_log_settings = {'log_file': log_file_name, 'log_level': 'INFO'}
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f_cfg_for_log:
                _config_for_log = yaml.safe_load(f_cfg_for_log)
                if _config_for_log and 'logging_settings' in _config_for_log:
                    # YAMLの設定を使いつつ、ログファイル名だけ上書きする
                    temp_log_settings = _config_for_log['logging_settings']
                    temp_log_settings['log_file'] = log_file_name 
    except Exception as e:
        print(f"警告: ログ設定の読み込み中にエラー: {e}。デフォルト設定を使います。", file=sys.stderr)

    logger = setup_logger(
        log_file_path=temp_log_settings.get('log_file', log_file_name),
        log_level_str=temp_log_settings.get('log_level', 'INFO'),
        logger_name=APP_LOGGER_NAME
    )

    if load_dotenv_success:
        logger.info(".env ファイルが正常に読み込まれました。")
    else:
        logger.warning(".env ファイルが見つからないか、読み込めませんでした。")

    #　config.yaml と 環境変数の解決
    config = {}
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f_cfg:
            config_from_yaml = yaml.safe_load(f_cfg)
            if config_from_yaml is None: config_from_yaml = {}
        logger.info(f"設定ファイル {CONFIG_FILE_PATH} を読み込みました。")

        resolved_api_creds = {}
        if 'api_credentials' in config_from_yaml and isinstance(config_from_yaml['api_credentials'], dict):
            for api_name, creds_config in config_from_yaml['api_credentials'].items():
                resolved_api_creds[api_name] = {}
                if isinstance(creds_config, dict):
                    for key_type, env_var_name in creds_config.items():
                        if key_type.endswith("_env_var"):
                            actual_key_name = key_type[:-len("_env_var")]
                            env_value = os.getenv(str(env_var_name))
                            if env_value is not None:
                                resolved_api_creds[api_name][actual_key_name] = env_value
                                logger.debug(f"解決: api_credentials.{api_name}.{actual_key_name} <-- 環:{env_var_name}")
                            else:
                                resolved_api_creds[api_name][actual_key_name] = None
                                logger.warning(f"環境変数 '{env_var_name}' が見つかりません ({api_name}.{actual_key_name} 用)。")
                        else:
                            resolved_api_creds[api_name][key_type] = env_var_name
        
        config = config_from_yaml
        if resolved_api_creds:
            if 'api_credentials' not in config: config['api_credentials'] = {}
            config['api_credentials'].update(resolved_api_creds)
        
        logger.info("設定ファイルの読み込みと環境変数の解決が完了しました。")

    except FileNotFoundError:
        logger.critical(f"設定ファイル {CONFIG_FILE_PATH} が見つかりません。")
        return None, None
    except Exception as e:
        logger.critical(f"設定読み込み中に予期せぬエラー: {e}")
        return None, None
    
    return logger, config