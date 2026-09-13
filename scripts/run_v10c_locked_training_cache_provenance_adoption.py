"""Production CLI for the V10C offline provenance-adoption validator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.v10c_locked_training_cache_provenance_adoption import (
    GovernanceProvenanceFailure,
    ImplementationFailure,
    LockedArtifactIntegrityFailure,
    phase_a_preflight,
    phase_b_offline_adoption,
)


RUNNER_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one locked V10C candidate offline.")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--implementation-sha", required=True)
    parser.add_argument("--authorization-marker", required=True)
    parser.add_argument("--receipt-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        phase_a = phase_a_preflight(
            RUNNER_REPO_ROOT,
            Path(args.candidate_root),
            args.implementation_sha,
            marker_path=Path(args.authorization_marker),
            receipt_path=Path(args.receipt_path),
        )
        receipt = phase_b_offline_adoption(
            phase_a["candidate_root"],
            Path(args.authorization_marker),
            args.implementation_sha,
            phase_a["ticker_order"],
            repo_root=RUNNER_REPO_ROOT,
            receipt_path=Path(args.receipt_path),
        )
        sys.stdout.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except GovernanceProvenanceFailure:
        sys.stderr.write("V10C_OFFLINE_ADOPTION_GOVERNANCE_FAILURE\n")
        return 4
    except LockedArtifactIntegrityFailure:
        sys.stderr.write("V10C_OFFLINE_ADOPTION_LOCKED_ARTIFACT_INTEGRITY_FAILURE\n")
        return 5
    except ImplementationFailure:
        sys.stderr.write("V10C_OFFLINE_ADOPTION_IMPLEMENTATION_FAILURE\n")
        return 3
    except Exception:
        sys.stderr.write("V10C_OFFLINE_ADOPTION_IMPLEMENTATION_FAILURE\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
