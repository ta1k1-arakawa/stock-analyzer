"""Thin CLI for one human-authorized JPX calendar source acquisition."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_jpx_calendar import (  # noqa: E402
    CONFIRMATION,
    V7JpxCalendarBlocked,
    acquire_jpx_calendar,
    canonical_json_bytes,
)


def _production_opener(request_obj, timeout=30):
    return urllib.request.urlopen(request_obj, timeout=timeout)


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire the official JPX market-holiday snapshot")
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--calendar-output", required=True)
    parser.add_argument("--confirmation", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = acquire_jpx_calendar(
            raw_output=args.raw_output,
            calendar_output=args.calendar_output,
            confirmation=args.confirmation,
            opener=_production_opener,
            clock=_utc_clock,
        )
    except V7JpxCalendarBlocked as error:
        sys.stdout.buffer.write(canonical_json_bytes({"status": "BLOCKED", "reason": error.reason}))
        return 2
    summary = {
        "status": "PASS",
        "calendar_source": snapshot["calendar_source"],
        "calendar_source_host": snapshot["calendar_source_host"],
        "covered_years": snapshot["covered_years"],
        "market_holiday_count": len(snapshot["market_holidays"]),
        "source_payload_sha256": snapshot["source_payload_sha256"],
        "study_calendar_generated": snapshot["study_calendar_generated"],
        "activation_boundary_status": snapshot["activation_boundary_status"],
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
