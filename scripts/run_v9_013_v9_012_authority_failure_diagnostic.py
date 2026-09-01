"""Safe runner for the explicit, read-only V9_013 diagnostic operation."""

from __future__ import annotations

import argparse
import sys

from src.v9_013_v9_012_authority_failure_diagnostic import (
    DiagnosticError,
    diagnose_preserved_state,
    safe_error_bytes,
    serialize_public_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9_013 authority-failure diagnostic")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--state-root", required=True)
    diagnose.add_argument("--diagnostic-design-git-sha", required=True)
    diagnose.add_argument("--diagnostic-implementation-git-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation != "diagnose":
        return 3
    try:
        result = diagnose_preserved_state(
            args.state_root,
            diagnostic_design_git_sha=args.diagnostic_design_git_sha,
            diagnostic_implementation_git_sha=args.diagnostic_implementation_git_sha,
        )
        sys.stdout.buffer.write(serialize_public_result(result))
        return 0
    except DiagnosticError as exc:
        sys.stderr.buffer.write(safe_error_bytes(exc.reason))
        return 2
    except Exception:
        sys.stderr.buffer.write(safe_error_bytes("IMPLEMENTATION_FAILURE"))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
