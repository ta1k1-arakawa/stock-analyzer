"""V9_006 F6 root-structure diagnostic production boundary; the
confirmation token is never accepted on argv -- only via the
V9_006_F6_ROOT_STRUCTURE_PROBE_CONFIRMATION environment variable, set at
point of use immediately before a fresh, separate, explicit one-shot human
authorization. This diagnostic confirmation is dedicated and distinct from
production Stage-A's V9_005_STAGE_A_CONFIRMATION -- the production token
does NOT satisfy this gate (see run_f6_root_structure_probe_network's own
confirmation check). Running this script directly against a real network is
NOT authorized by this implementation task; it exists so a later,
separately authorized run can invoke it. It never calls production Stage
A's orchestration entrypoint, and its fetcher/retry policy can only ever
request exactly TOPIX_ROOT_URL -- no discovery of any other URL, no href
following, no child fetch, no F5/other source.

The real HTTP fetcher and clock below are not duplicated: they import the
exact same, already-reviewed production-safe fetch/redirect primitive and
UTC clock from run_v9_005_stage_a_jpx_probe.py (same directory), unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_v9_005_stage_a_jpx_probe import _production_fetcher, _utc_clock  # noqa: E402

from src.v9_005_stage_a_jpx_probe import (  # noqa: E402
    F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME,
    V9005StageABlocked,
    run_f6_root_structure_probe_network,
)

CONFIRMATION_ENV = "V9_006_F6_ROOT_STRUCTURE_PROBE_CONFIRMATION"
FAILURE_SCHEMA_VERSION = "V9_006_F6_ROOT_STRUCTURE_PROBE_FAILURE_V1"

# Only these fields -- and the derived artifact_path -- ever reach stdout.
# Occurrences, anchors, raw href values, raw payload bytes, and page/index
# text stay only in the durable diagnostic artifact for later GPT review.
_SAFE_RESULT_FIELDS = (
    "status", "label_occurrence_count", "requested_url", "resolved_url",
    "http_status", "byte_length", "sha256", "retrieval_timestamp_utc",
    "network_request_count",
)


def _print(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _safe_failure(failure_class: str, network_request_count: int) -> dict[str, object]:
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "execution_result": "BLOCKED",
        "failure_class": failure_class,
        "network_request_count": network_request_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the V9_006 F6 root-structure diagnostic probe (TOPIX_ROOT_URL only)",
    )
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    confirmation = os.environ.get(CONFIRMATION_ENV)
    if not confirmation:
        _print(_safe_failure("GOVERNANCE_FAILURE", 0))
        return 2
    try:
        artifact = run_f6_root_structure_probe_network(
            output_root=args.output_root,
            confirmation=confirmation,
            fetcher=_production_fetcher,
            sleep=time.sleep,
            clock=_utc_clock,
        )
    except V9005StageABlocked as exc:
        _print(_safe_failure(exc.failure_class, exc.network_request_count))
        return 2
    except Exception:
        # Defense-in-depth: an ordinary (non-V9005StageABlocked) exception
        # must never reach the user as a raw traceback/message.
        _print(_safe_failure("IMPLEMENTATION_FAILURE", 0))
        return 2
    finally:
        confirmation = ""
    safe_result = {field: artifact.get(field) for field in _SAFE_RESULT_FIELDS}
    safe_result["artifact_path"] = str(Path(args.output_root) / F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME)
    _print(safe_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
