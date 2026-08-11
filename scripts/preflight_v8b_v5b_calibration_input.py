#!/usr/bin/env python3
"""Gated, fixed-root-only CLI for the V5-B calibration input (R1) preflight.

Exactly two mutually exclusive modes:

``--static-check``
    Repository-only verification. Performs no filesystem access to any
    V5-B cache and no network call.

``--confirm TOKEN --implementation-git-commit COMMIT``
    Human-gated production execution against the single fixed cache root
    (``src/v8b_v5b_calibration_input_preflight.py``'s ``V5B_CACHE_ROOT``).
    ``TOKEN`` must exactly equal the required confirmation constant.

There is no option, in either mode, that accepts an alternate cache path,
manifest path, input directory, or dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v8b_v5b_calibration_input_preflight import (  # noqa: E402
    V5BCalibrationInputPreflightBlocked,
    canonical_json_bytes,
    run_production_v5b_calibration_input_preflight,
    run_static_check,
)

STATIC_SUCCESS_MESSAGE = "V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_STATIC_PASS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gated, fixed-root-only CLI for the V5-B calibration input (R1) preflight.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--static-check",
        action="store_true",
        help="Repository-only verification. No V5-B cache access, no network.",
    )
    mode.add_argument(
        "--confirm",
        help="Exact human-gate confirmation token for real, fixed-cache-root execution.",
    )
    parser.add_argument(
        "--implementation-git-commit",
        help="40-hex Git commit of this implementation. Required with --confirm.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.static_check:
        if args.confirm is not None or args.implementation_git_commit is not None:
            print("STATIC_CHECK_TAKES_NO_OTHER_ARGUMENTS", file=sys.stderr)
            return 2
        try:
            run_static_check()
        except V5BCalibrationInputPreflightBlocked as error:
            print(error.detail, file=sys.stderr)
            return 2
        print(STATIC_SUCCESS_MESSAGE)
        return 0

    if not args.implementation_git_commit:
        print("IMPLEMENTATION_COMMIT_REQUIRED", file=sys.stderr)
        return 2

    try:
        result = run_production_v5b_calibration_input_preflight(
            confirmation=args.confirm,
            implementation_git_commit=args.implementation_git_commit,
        )
    except V5BCalibrationInputPreflightBlocked as error:
        payload = error.result if error.result is not None else {"status": "BLOCK", "reason": error.reason, "detail_reason": error.detail}
        sys.stdout.buffer.write(canonical_json_bytes(payload))
        return 2

    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
