"""Production entrypoint for the fixed V10B public acquisition."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running a script by path places ``scripts`` (not the repository root) on
# sys.path. Bind imports to this checked-out source tree without exposing any
# caller-controlled implementation module path.
REPO_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_SOURCE_ROOT))

from src.v10b_training_cache_reacquisition import (
    GovernanceFailure,
    V10BError,
    run_production,
)


PREFLIGHT_FAILURE_TOKEN = "V10B_ACQUISITION_PREFLIGHT_FAILURE"
IMPLEMENTATION_FAILURE_TOKEN = "V10B_ACQUISITION_IMPLEMENTATION_FAILURE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded V10B fixed-universe Yahoo acquisition."
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--implementation-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_production(args.repo_root, args.attempt_root, args.implementation_sha)
    except GovernanceFailure:
        print(PREFLIGHT_FAILURE_TOKEN, file=sys.stderr)
        return 4
    except V10BError:
        print(IMPLEMENTATION_FAILURE_TOKEN, file=sys.stderr)
        return 3
    except Exception:
        print(IMPLEMENTATION_FAILURE_TOKEN, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
