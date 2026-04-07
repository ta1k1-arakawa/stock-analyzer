# main.py — AI予測・本番運用 エントリポイント

from src.config import load_app
from src.predict import run_prediction

if __name__ == "__main__":
    config, logger = load_app(log_file="app_main.log")
    run_prediction(config)