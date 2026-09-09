"""Thin, read-only CLI for the V7 Gate 4 provenance preflight."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_gate4_preflight import (  # noqa: E402
    V7Gate4PreflightBlocked,
    canonical_json_bytes,
    run_gate4_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only V7 Gate 4 provenance preflight")
    parser.add_argument("--seed-bundle", required=True)
    parser.add_argument("--calendar-json", required=True)
    parser.add_argument("--calendar-raw", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--prospective-boundary", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_gate4_preflight(
            seed_bundle=args.seed_bundle,
            calendar_json=args.calendar_json,
            calendar_raw=args.calendar_raw,
            universe_csv=args.universe_csv,
            prospective_boundary=args.prospective_boundary,
        )
    except V7Gate4PreflightBlocked as error:
        sys.stdout.buffer.write(canonical_json_bytes({"status": "BLOCKED", "reason": error.reason}))
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
