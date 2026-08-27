"""V9_006 F6 production ROOT/GLOBAL raw acquisition production boundary;
the confirmation token is never accepted on argv -- only via the
V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION environment
variable, set at point of use immediately before a fresh, separate, explicit
one-shot human authorization. This dedicated confirmation is distinct from
both production Stage-A's V9_005_STAGE_A_CONFIRMATION and the F6
root-structure diagnostic's V9_006_F6_ROOT_STRUCTURE_PROBE_CONFIRMATION --
neither of those satisfies this gate, and this one satisfies neither of
those. Running this script directly against a real network is NOT
authorized by this implementation task; it exists so a later, separately
authorized run can invoke it. It never calls production Stage A's
orchestration entrypoint, and its fetcher/retry policy can only ever
request exactly TOPIX_ROOT_URL and then, only after locator success, the
one URL mechanically resolved from the newly locked production ROOT -- no
discovery of any other URL, no F1-F5/F7 request, and no use of the F6
root-structure diagnostic lock or its previously observed child href/URL.

This script never performs environment/dependency bootstrap itself; per the
reviewed design, a future Windows PowerShell entrypoint separately enforces
canonical `.venv-real-execution` readiness BEFORE supplying the confirmation
environment variable at all.

The real HTTP fetcher and clock below are not duplicated: they import the
exact same, already-reviewed production-safe fetch/redirect primitive and
UTC clock from run_v9_005_stage_a_jpx_probe.py (same directory), unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
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
    GOVERNANCE_FAILURE,
    V9005StageABlocked,
    read_f6_production_acquisition_gate_consumed_state,
    run_f6_production_root_global_raw_acquisition_network,
)

CONFIRMATION_ENV = "V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION"
FAILURE_SCHEMA_VERSION = "V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_FAILURE_V1"


def _url_sha256(url: object) -> str | None:
    if not isinstance(url, str):
        return None
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _print(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _safe_failure(
    failure_class: str, network_request_count: object, *, gate_consumed: object,
) -> dict[str, object]:
    # network_request_count and gate_consumed are tri-state: a bool/int when
    # mechanically known, otherwise the literal string "unknown" -- never a
    # fabricated 0/false when the true value cannot be proven.
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "execution_result": "BLOCKED",
        "failure_class": failure_class,
        "network_request_count": (
            network_request_count
            if isinstance(network_request_count, int) and not isinstance(network_request_count, bool)
            else "unknown"
        ),
        "gate_consumed": gate_consumed if isinstance(gate_consumed, bool) else "unknown",
        "authorization_reusable": False,
        "second_execution_allowed": False,
    }


def _safe_gate_consumed_state(output_root: object) -> bool | str:
    """Return only proven durable gate state for reporting.

    The reader is itself fail-closed, but safe failure reporting must remain
    JSON-only even if an unexpected filesystem or reader failure escapes it.
    No exception detail or local path may reach stdout.
    """
    try:
        state = read_f6_production_acquisition_gate_consumed_state(output_root)
    except Exception:
        return "unknown"
    return state if isinstance(state, bool) else "unknown"


def _safe_success(artifact: dict[str, object]) -> dict[str, object]:
    # Only hashes, counts, statuses, and equality booleans ever reach
    # stdout -- never a raw requested/resolved URL, payload bytes, the
    # receipt path, or the output_root path.
    root = artifact["root"]
    child = artifact["child"]
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "execution_result": "COMPLETE",
        "status": artifact["status"],
        "gate_consumed": artifact["gate_consumed"],
        "authorization_reusable": False,
        "second_execution_allowed": False,
        "root_http_status": root["http_status"],
        "root_byte_length": root["byte_length"],
        "root_sha256": root["sha256"],
        "root_retrieval_timestamp_utc": root["retrieval_timestamp_utc"],
        "root_requested_url_sha256": _url_sha256(root["requested_url"]),
        "root_resolved_url_sha256": _url_sha256(root["resolved_url"]),
        "root_requested_resolved_url_equal": root["requested_url"] == root["resolved_url"],
        "locator_status": artifact["locator_status"],
        "candidate_anchor_count": artifact["candidate_anchor_count"],
        "child_http_status": child["http_status"],
        "child_byte_length": child["byte_length"],
        "child_sha256": child["sha256"],
        "child_retrieval_timestamp_utc": child["retrieval_timestamp_utc"],
        "child_requested_url_sha256": _url_sha256(child["requested_url"]),
        "child_resolved_url_sha256": _url_sha256(child["resolved_url"]),
        "child_requested_resolved_url_equal": child["requested_url"] == child["resolved_url"],
        "root_network_request_count": artifact["root_network_request_count"],
        "child_network_request_count": artifact["child_network_request_count"],
        "network_request_count": artifact["network_request_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V9_006 F6 production ROOT/GLOBAL raw acquisition "
            "(TOPIX_DISCOVERY_ROOT then, only after locator success, "
            "TOPIX_GLOBAL_2017_2025)"
        ),
    )
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    confirmation = os.environ.get(CONFIRMATION_ENV)
    if not confirmation:
        # Zero filesystem access: never stat the output_root/receipt here --
        # this gate was never even evaluated, so it is deterministically
        # unconsumed regardless of unrelated prior state at that path.
        _print(_safe_failure("GOVERNANCE_FAILURE", 0, gate_consumed=False))
        return 2
    try:
        artifact = run_f6_production_root_global_raw_acquisition_network(
            output_root=args.output_root,
            confirmation=confirmation,
            fetcher=_production_fetcher,
            sleep=time.sleep,
            clock=_utc_clock,
        )
    except V9005StageABlocked as exc:
        if exc.failure_class == GOVERNANCE_FAILURE:
            # Only ever raised here for a wrong confirmation, always before
            # any filesystem access -- same deterministic case as above.
            gate_consumed: object = False
        else:
            # Durable receipt state is the ground truth for gate_consumed,
            # never whether this Python call happened to return cleanly.
            gate_consumed = _safe_gate_consumed_state(args.output_root)
        _print(_safe_failure(exc.failure_class, exc.network_request_count, gate_consumed=gate_consumed))
        return 2
    except Exception:
        # Defense-in-depth: an ordinary (non-V9005StageABlocked) exception
        # must never reach the user as a raw traceback/message. The core
        # function itself converts every reachable unexpected failure into
        # a V9005StageABlocked with an accurate cumulative count; this
        # branch is only for something that somehow still escaped that, so
        # the request count genuinely cannot be proven here.
        gate_consumed = _safe_gate_consumed_state(args.output_root)
        _print(_safe_failure("IMPLEMENTATION_FAILURE", "unknown", gate_consumed=gate_consumed))
        return 2
    finally:
        confirmation = ""
    _print(_safe_success(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
