"""Read-only V9_009 fixed-cache runner.

This entrypoint has no transport fallback and never writes cache or result
artifacts.  It emits exactly one validated safe JSON line for a closed result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v9_009_t0_top1_kill_screen import (  # noqa: E402
    T0DataIncompatible,
    make_safe_result,
    run_from_cache,
    synthetic_provenance,
    validate_safe_result,
)


IMPLEMENTATION_FAILURE = "V9_009_T0_TOP1_KILL_SCREEN_IMPLEMENTATION_FAILURE"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only V9 TOP1 kill screen")
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--evaluation-cache", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--calendar-file", required=True)
    parser.add_argument("--implementation-sha", required=True)
    return parser.parse_args(argv)


def _calendar_values(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("trading_dates")
    if not isinstance(value, list):
        raise T0DataIncompatible("CALENDAR_FILE_SCHEMA_INVALID")
    return value


def main(argv: list[str] | None = None) -> int:
    try:
        args = _arguments(argv)
        calendar = _calendar_values(Path(args.calendar_file))
        result = run_from_cache(
            args.training_cache,
            args.evaluation_cache,
            args.universe_csv,
            calendar,
            args.implementation_sha,
        )
        sys.stdout.write(json.dumps(validate_safe_result(result), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except T0DataIncompatible:
        try:
            result = make_safe_result(
                "NO_VERDICT_DATA_INCOMPATIBLE",
                args.implementation_sha,  # type: ignore[name-defined]
                synthetic_provenance(),
                cache_identity=False,
                exact_semantics=False,
            )
            sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            return 0
        except Exception:
            pass
    except Exception:
        pass
    sys.stderr.write(IMPLEMENTATION_FAILURE + "\n")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
