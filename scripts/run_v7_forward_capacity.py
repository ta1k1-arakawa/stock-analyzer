"""Restricted Gate 2 synthetic/static runner.

This CLI intentionally has no activation, collector, network, seed-acquisition,
evaluation, replay, or real-order option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_capacity_engine import V7EngineParameters, canonical_sha256
from src.v7_forward_protocol import (
    ArmInputHashes,
    create_dual_arm_study,
    sha256_bytes,
    validate_seed_rows,
)


def _calendar(count: int = 12) -> list[str]:
    start = date(2020, 1, 2)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def synthetic_forward_fixture() -> tuple[
    list[str], dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]
]:
    calendar = _calendar()
    tickers = ("AAA", "BBB", "CCC")
    frames: dict[str, dict[str, dict[str, float]]] = {}
    for ticker in tickers:
        frames[ticker] = {}
        for day in calendar:
            frames[ticker][day] = {"Open": 100.0, "Close": 100.0}
    candidates = [
        {
            "signal_year": 2020,
            "signal_date": calendar[0],
            "ticker": ticker,
            "industry": industry,
            "rank": rank,
            "signal_raw_close": 100.0,
            "entry_attempt_date": calendar[1],
            "planned_exit_date": calendar[10],
            "candidate_status": "ACCEPTED_TOP20",
        }
        for rank, (ticker, industry) in enumerate(
            zip(tickers, ("TECH", "FINANCE", "ENERGY")), start=1
        )
    ]
    return calendar, frames, candidates


def _synthetic_seed_rows(count: int = 252) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = date(2018, 1, 2)
    for ticker in ("AAA", "BBB"):
        for index in range(count):
            day = start + timedelta(days=index)
            rows.append({
                "ticker": ticker,
                "trading_date": day.isoformat(),
                "raw_open": 100.0,
                "raw_high": 101.0,
                "raw_low": 99.0,
                "raw_close": 100.0,
                "raw_volume": 1000,
            })
    return rows


def _input_hashes() -> ArmInputHashes:
    value = "a" * 64
    return ArmInputHashes(value, value, value, value)


def _state_hash(engine: Any) -> str:
    return canonical_sha256(engine.state_snapshot())


def run_golden_pass() -> dict[str, Any]:
    calendar, frames, candidates = synthetic_forward_fixture()
    hashes = _input_hashes()
    study = create_dual_arm_study(
        frames,
        calendar,
        candidates,
        hashes,
        hashes,
        V7EngineParameters.control(),
        V7EngineParameters.capacity_3(),
    ).run()
    control_skips = study.control.skip_reason_counts()
    variant_skips = study.variant.skip_reason_counts()
    if control_skips.get("MAX_OPEN_POSITIONS", 0) != 1:
        raise AssertionError("CONTROL_CAPACITY_SKIP_MISSING")
    if len([row for row in study.control.state.completed_trades if row["status"] in {"FILLED", "CLOSED"}]) != 2:
        raise AssertionError("CONTROL_FILL_COUNT_MISMATCH")
    if len([row for row in study.variant.state.completed_trades if row["status"] in {"FILLED", "CLOSED"}]) != 3:
        raise AssertionError("VARIANT_FILL_COUNT_MISMATCH")
    seed_result = validate_seed_rows(
        _synthetic_seed_rows(),
        ("AAA", "BBB"),
        "2019-01-01",
    )
    if seed_result["row_count"] != 504 or seed_result["eligible_ticker_count"] != 2:
        raise AssertionError("SEED_GOLDEN_MISMATCH")
    summary = {
        "mode": "DRY_RUN_ONLY",
        "synthetic_only": True,
        "seed_study_event_count": 0,
        "control_parameters_sha256": study.control.parameters.sha256(),
        "variant_parameters_sha256": study.variant.parameters.sha256(),
        "input_hashes": hashes.as_dict(),
        "control_state_sha256": _state_hash(study.control),
        "variant_state_sha256": _state_hash(study.variant),
        "control_filled": 2,
        "variant_filled": 3,
        "control_max_position_skips": control_skips["MAX_OPEN_POSITIONS"],
        "safety_control": study.control.safety_counters(),
        "safety_variant": study.variant.safety_counters(),
        "seed_canonical_sha256": seed_result["seed_canonical_sha256"],
        "seed_payload_manifest_sha256": seed_result["seed_payload_manifest_sha256"],
        "two_pass_byte_identical": True,
    }
    return summary


def run_synthetic_golden() -> dict[str, Any]:
    first = run_golden_pass()
    second = run_golden_pass()
    first_bytes = (json.dumps(first, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    second_bytes = (json.dumps(second, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if first_bytes != second_bytes:
        raise AssertionError("SYNTHETIC_SUMMARY_BYTE_MISMATCH")
    if first["control_state_sha256"] != second["control_state_sha256"] or first["variant_state_sha256"] != second["variant_state_sha256"]:
        raise AssertionError("SYNTHETIC_STATE_HASH_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="v7-gate2-") as temporary:
        root = Path(temporary)
        output = root / "golden"
        output.mkdir()
        (output / "summary.json").write_bytes(first_bytes)
        digest = sha256_bytes((output / "summary.json").read_bytes())
        if digest != hashlib.sha256(first_bytes).hexdigest():
            raise AssertionError("SYNTHETIC_OUTPUT_HASH_MISMATCH")
        shutil.rmtree(output)
    return first


def run_static_check() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    prohibited = ("requests", "urllib", "http://", "https://")
    inspected = []
    for name in ("v7_capacity_engine.py", "v7_forward_protocol.py"):
        text = (source_root / name).read_text(encoding="utf-8").lower()
        inspected.append(name)
        if any(token in text for token in prohibited):
            raise AssertionError("STATIC_PROHIBITED_OPERATION")
    return {
        "mode": "DRY_RUN_ONLY",
        "static_check": "PASS",
        "inspected": inspected,
        "activation_created": False,
        "network": False,
        "seed_acquisition": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 Gate 2 offline-only checks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-golden-test", action="store_true")
    group.add_argument("--gate2-static-check", action="store_true")
    args = parser.parse_args(argv)
    result = run_synthetic_golden() if args.synthetic_golden_test else run_static_check()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
