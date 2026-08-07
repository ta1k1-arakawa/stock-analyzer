"""Thin CLI for the pre-activation V7 seed acquisition boundary."""

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

from src.v7_seed_acquisition import (  # noqa: E402
    acquire_seed_bundle,
    canonical_json_bytes,
    V7SeedAcquisitionBlocked,
)


def _production_opener(request_obj):
    return urllib.request.urlopen(request_obj, timeout=30)


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire the V7 pre-activation feature seed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--request-start", required=True)
    parser.add_argument("--request-end-exclusive", required=True)
    parser.add_argument("--seed-cutoff", required=True)
    parser.add_argument("--confirmation", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = acquire_seed_bundle(
            output_dir=args.output_dir,
            universe_csv=args.universe_csv,
            request_start=args.request_start,
            request_end_exclusive=args.request_end_exclusive,
            seed_cutoff=args.seed_cutoff,
            confirmation=args.confirmation,
            opener=_production_opener,
            clock=_utc_clock,
        )
    except V7SeedAcquisitionBlocked as error:
        print(json.dumps({"status": "BLOCKED", "reason": error.reason}, sort_keys=True))
        return 2
    summary = {
        "status": "PASS",
        "mode": manifest["mode"],
        "ticker_count": manifest["ticker_count"],
        "request_count": manifest["request_count"],
        "success_count": manifest["success_count"],
        "failed_count": manifest["failed_count"],
        "eligible_seed_ticker_count": manifest["eligible_seed_ticker_count"],
        "ineligible_seed_ticker_count": manifest["ineligible_seed_ticker_count"],
        "seed_row_count": manifest["seed_row_count"],
        "seed_payload_manifest_sha256": manifest["seed_payload_manifest_sha256"],
        "seed_canonical_csv_sha256": manifest["seed_canonical_csv_sha256"],
        "activation_status": manifest["activation_status"],
        "study_calendar_generated": manifest["study_calendar_generated"],
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
