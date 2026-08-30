"""Phase-1 Python CLI wiring; real execution still requires a future PowerShell gate.

The dedicated confirmation is never accepted on argv. A future reviewed
Windows PowerShell entrypoint may provide it through the environment only
after its own pre-gate readiness checks and fresh human authorization.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_v9_005_stage_a_jpx_probe import _production_fetcher, _utc_clock  # noqa: E402
from src.v9_005_stage_a_jpx_probe import V9005StageABlocked  # noqa: E402
from src.v9_006_stage_a_schema_discovery import (  # noqa: E402
    SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION,
    Phase1SchemaDiscoveryResult,
    read_phase1_schema_discovery_gate_consumed_state,
    run_phase1_schema_discovery_one_shot,
)

CONFIRMATION_ENV = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_CONFIRMATION"
FAILURE_SCHEMA_VERSION = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_CLI_FAILURE_V1"
SUCCESS_SCHEMA_VERSION = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_CLI_SUCCESS_V1"


def _print(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _safe_gate_consumed_state() -> bool | str:
    try:
        state = read_phase1_schema_discovery_gate_consumed_state()
    except Exception:
        return "unknown"
    return state if isinstance(state, bool) else "unknown"


def _safe_failure(failure_class: str, network_attempt_count: object, gate_consumed: object) -> dict[str, object]:
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "execution_result": "BLOCKED",
        "failure_class": failure_class,
        "network_attempt_count": network_attempt_count if isinstance(network_attempt_count, int) and not isinstance(network_attempt_count, bool) else "unknown",
        "gate_consumed": gate_consumed if isinstance(gate_consumed, bool) else "unknown",
    }


def _safe_success(result: Phase1SchemaDiscoveryResult) -> dict[str, object]:
    slot_digest = sha256("\n".join(result.evidence_slot_ids).encode("ascii")).hexdigest()
    return {
        "schema_version": SUCCESS_SCHEMA_VERSION,
        "execution_result": "COMPLETE",
        "gate_consumed": True,
        "evidence_count": len(result.evidence_slot_ids),
        "support_raw_lock_count": 12,
        "total_raw_lock_pair_count": 353,
        "network_attempt_count": result.network_attempt_count,
        "evidence_slot_ids_sha256": slot_digest,
        "safe_profile_count": len(result.safe_profiles),
        "representative_safe_profile_count": len(result.representative_safe_profiles),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reviewed V9_006 Phase-1 schema-discovery boundary")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execution-sha", required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    fetcher: Callable[[str], Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], Any] | None = None,
    confirmation: object = None,
) -> int:
    args = build_parser().parse_args(argv)
    supplied_confirmation = os.environ.get(CONFIRMATION_ENV) if confirmation is None else confirmation
    if supplied_confirmation != SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION:
        _print(_safe_failure("GOVERNANCE_FAILURE", 0, False))
        return 2
    bound_fetcher = _production_fetcher if fetcher is None else fetcher
    bound_sleep = time.sleep if sleep is None else sleep
    bound_clock = _utc_clock if clock is None else clock
    try:
        result = run_phase1_schema_discovery_one_shot(
            args.output_root,
            confirmation=supplied_confirmation,
            execution_sha=args.execution_sha,
            fetcher=bound_fetcher,
            sleep=bound_sleep,
            clock=bound_clock,
        )
    except V9005StageABlocked as exc:
        gate_consumed = _safe_gate_consumed_state()
        # After receipt publication, an exception-local count cannot establish
        # the cumulative Phase-1 request total.  Preserve exact pre-gate zero
        # reporting, but fail safely when the canonical gate is consumed.
        network_attempt_count: object = "unknown" if gate_consumed is True else exc.network_request_count
        _print(_safe_failure(exc.failure_class, network_attempt_count, gate_consumed))
        return 2
    except Exception:
        _print(_safe_failure("IMPLEMENTATION_FAILURE", "unknown", _safe_gate_consumed_state()))
        return 2
    finally:
        supplied_confirmation = ""
    _print(_safe_success(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
