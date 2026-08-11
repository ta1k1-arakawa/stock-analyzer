#!/usr/bin/env python3
"""Static-only check for the V8B data-quality calibration implementation.

Supports exactly ``--static-check``. Reads repository files only; there is
no option that accepts a V5-B cache path, and no network-capable or
real-data execution path exists in this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v8b_data_quality_calibration import V8BCalibrationBlocked, run_static_check

SUCCESS_MESSAGE = "V8B_CALIBRATION_IMPLEMENTATION_STATIC_PASS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static-only check for the V8B data-quality calibration implementation.",
    )
    parser.add_argument(
        "--static-check",
        action="store_true",
        required=True,
        help="Run the repository-only static verification (no real data, no network).",
    )
    args = parser.parse_args(argv)
    del args

    repository_root = Path(__file__).resolve().parents[1]
    try:
        run_static_check(repository_root)
    except V8BCalibrationBlocked as error:
        print(error.reason, file=sys.stderr)
        return 2

    print(SUCCESS_MESSAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
