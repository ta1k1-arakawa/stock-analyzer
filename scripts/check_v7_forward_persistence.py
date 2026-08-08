"""Synthetic-only V7 forward persistence and restart verification CLI.

This CLI intentionally has no real study-root, network, collector, or
activation option. It only exercises the append-only persistence and
engine-runtime restart contract against a local synthetic fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_capacity_engine import CausalEventEngine, V7EngineParameters
from src.v7_forward_persistence import (
    ForwardStudyStore,
    canonical_json_bytes,
    export_engine_runtime,
    restore_engine_runtime,
    verify_forward_store,
)

ACTIVATION_MANIFEST_SHA256 = "a" * 64
COLLECTOR_COMMIT = "b" * 40
SPLIT_INDEX = 6


def _calendar(count: int = 16) -> list[str]:
    start = date(2020, 1, 2)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _fixture() -> tuple[list[str], dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    calendar = _calendar()
    tickers = ("AAA", "BBB", "CCC")
    industries = {"AAA": "TECH", "BBB": "FINANCE", "CCC": "ENERGY"}
    frames: dict[str, dict[str, dict[str, float]]] = {
        ticker: {
            day: {"Open": 100.0 + index, "Close": 100.0 + index}
            for index, day in enumerate(calendar)
        }
        for ticker in tickers
    }
    candidates = []
    for signal_index, ticker in enumerate(tickers):
        signal_day = calendar[signal_index]
        candidates.append({
            "signal_year": 2020,
            "signal_date": signal_day,
            "ticker": ticker,
            "industry": industries[ticker],
            "rank": 1,
            "signal_raw_close": frames[ticker][signal_day]["Close"],
            "entry_attempt_date": calendar[signal_index + 1],
            "planned_exit_date": calendar[signal_index + 10],
            "candidate_status": "ACCEPTED_TOP20",
        })
    return calendar, frames, candidates


def _run_continuous(parameters: V7EngineParameters) -> dict[str, Any]:
    calendar, frames, candidates = _fixture()
    engine = CausalEventEngine(frames, calendar, candidates, parameters)
    for day in calendar:
        engine.process_day(day)
    return export_engine_runtime(engine)


def _run_with_restart(parameters: V7EngineParameters) -> dict[str, Any]:
    calendar, frames, candidates = _fixture()
    engine = CausalEventEngine(frames, calendar, candidates, parameters)
    for day in calendar[:SPLIT_INDEX]:
        engine.process_day(day)
    exported = export_engine_runtime(engine)
    resumed = CausalEventEngine(frames, calendar, candidates, parameters)
    restore_engine_runtime(resumed, exported)
    for day in calendar[SPLIT_INDEX:]:
        resumed.process_day(day)
    return export_engine_runtime(resumed)


def _byte_identical(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


def _checkpoint_chain_pass() -> bool:
    calendar, frames, candidates = _fixture()
    control = CausalEventEngine(frames, calendar, candidates, V7EngineParameters.control())
    variant = CausalEventEngine(frames, calendar, candidates, V7EngineParameters.capacity_3())
    with tempfile.TemporaryDirectory(prefix="v7-forward-persistence-chain-") as temporary:
        store = ForwardStudyStore(temporary)
        for day in calendar:
            control.process_day(day)
            variant.process_day(day)
            store.write_day(
                day,
                price_snapshot={"date": day},
                candidate_snapshot={
                    "date": day,
                    "count": sum(1 for row in candidates if row["signal_date"] == day),
                },
                market_gate_snapshot={"date": day},
                arm_a_runtime=export_engine_runtime(control),
                arm_b_runtime=export_engine_runtime(variant),
                activation_manifest_sha256=ACTIVATION_MANIFEST_SHA256,
                collector_commit=COLLECTOR_COMMIT,
            )
        result = verify_forward_store(temporary, ACTIVATION_MANIFEST_SHA256, COLLECTOR_COMMIT)
        return result["status"] == "PASS" and result["day_count"] == len(calendar)


def _atomic_day_commit_pass() -> bool:
    calendar, frames, candidates = _fixture()
    engine = CausalEventEngine(frames, calendar, candidates, V7EngineParameters.control())
    day = calendar[0]
    engine.process_day(day)
    runtime = export_engine_runtime(engine)

    class _InjectedFailure(RuntimeError):
        pass

    def _fail(_stage: str) -> None:
        raise _InjectedFailure(_stage)

    with tempfile.TemporaryDirectory(prefix="v7-forward-persistence-atomic-") as temporary:
        store = ForwardStudyStore(temporary)
        try:
            store.write_day(
                day,
                price_snapshot={"date": day},
                candidate_snapshot={"date": day},
                market_gate_snapshot={"date": day},
                arm_a_runtime=runtime,
                arm_b_runtime=runtime,
                activation_manifest_sha256=ACTIVATION_MANIFEST_SHA256,
                collector_commit=COLLECTOR_COMMIT,
                fault_injector=_fail,
            )
        except _InjectedFailure:
            pass
        else:
            return False
        return not (Path(temporary) / "days" / day).exists()


def run_synthetic_restart_test() -> dict[str, Any]:
    control_byte_identical = _byte_identical(
        _run_continuous(V7EngineParameters.control()),
        _run_with_restart(V7EngineParameters.control()),
    )
    variant_byte_identical = _byte_identical(
        _run_continuous(V7EngineParameters.capacity_3()),
        _run_with_restart(V7EngineParameters.capacity_3()),
    )
    checkpoint_chain_pass = _checkpoint_chain_pass()
    atomic_day_commit_pass = _atomic_day_commit_pass()

    if not control_byte_identical:
        raise AssertionError("CONTROL_RESTART_NOT_BYTE_IDENTICAL")
    if not variant_byte_identical:
        raise AssertionError("VARIANT_RESTART_NOT_BYTE_IDENTICAL")
    if not checkpoint_chain_pass:
        raise AssertionError("CHECKPOINT_CHAIN_NOT_PASS")
    if not atomic_day_commit_pass:
        raise AssertionError("ATOMIC_DAY_COMMIT_NOT_PASS")

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "restart_equivalence": True,
        "control_byte_identical": control_byte_identical,
        "variant_byte_identical": variant_byte_identical,
        "checkpoint_chain_pass": checkpoint_chain_pass,
        "atomic_day_commit_pass": atomic_day_commit_pass,
        "network_requests": 0,
        "activation_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 forward persistence synthetic-only checks")
    parser.add_argument("--synthetic-restart-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_restart_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
