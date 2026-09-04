"""Canonical Python entry point for the later one-shot Phase-B runner."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v9_014_jpx_monthly_auction_activity_source_b_calibration_acquisition import main


if __name__ == "__main__":
    raise SystemExit(main())
