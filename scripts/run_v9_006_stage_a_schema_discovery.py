"""Future schema-discovery gate; intentionally performs no acquisition."""
from __future__ import annotations
import argparse
from src.v9_006_stage_a_schema_discovery import prepare_future_acquisition

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--confirmation", required=True)
    prepare_future_acquisition(parser.parse_args().confirmation)
