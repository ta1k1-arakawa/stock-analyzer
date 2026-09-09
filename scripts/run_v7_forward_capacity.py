"""Restricted Gate 2 synthetic/static runner.

This CLI intentionally has no activation, collector, network, seed-acquisition,
evaluation, replay, or real-order option.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_capacity_engine import V7EngineParameters, canonical_sha256
from src.v7_gate3_dry_run import canonical_json_bytes, run_gate3_dry_run
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
                "adj_close": 100.0,
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
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    inspected_paths = [
        source_root / "v7_capacity_engine.py",
        source_root / "v7_forward_candidate.py",
        source_root / "v7_forward_protocol.py",
        source_root / "v7_gate3_dry_run.py",
        Path(__file__).resolve(),
    ]
    texts = {path.name: path.read_text(encoding="utf-8") for path in inspected_paths}
    if "DESIGN_COMMIT = \"e3e1367efd913b601a70328a815d88c20af6d147\"" not in texts["v7_gate3_dry_run.py"]:
        raise AssertionError("STATIC_DESIGN_COMMIT_MISMATCH")
    if "LATEST_PREREGISTRATION_UTC = \"2026-08-07T02:48:27Z\"" not in texts["v7_gate3_dry_run.py"]:
        raise AssertionError("STATIC_PREREGISTRATION_MISMATCH")
    gate3_text = texts["v7_gate3_dry_run.py"]
    for constant, expected in (
        ("MODE", "DRY_RUN_ONLY"),
        ("ACTIVATION_STATUS", "NOT_ACTIVATED"),
        ("ACTIVATION_BOUNDARY", "NOT_SET"),
    ):
        if f'{constant} = "{expected}"' not in gate3_text:
            raise AssertionError("STATIC_GATE3_BOUNDARY_MISMATCH:" + constant)

    script_tree = ast.parse(texts[Path(__file__).name])
    cli_options = {
        node.args[0].value
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }
    expected_options = {
        "--synthetic-golden-test",
        "--gate2-static-check",
        "--gate3-dry-run",
        "--gate3-static-check",
    }
    if cli_options != expected_options:
        raise AssertionError("STATIC_CLI_OPTIONS_MISMATCH")

    network_tokens = ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance")
    network_urls = ("h" + "ttp://", "h" + "ttps://")
    for name, text in texts.items():
        lowered = text.lower()
        import_lines = [line.strip().lower() for line in lowered.splitlines() if line.strip().startswith(("import ", "from "))]
        if any(any(token in line for token in network_tokens) for line in import_lines):
            raise AssertionError("STATIC_PROHIBITED_OPERATION:" + name)
        if any(token in lowered for token in network_urls):
            raise AssertionError("STATIC_PROHIBITED_OPERATION:" + name)
    inspected = [path.relative_to(repository_root).as_posix() for path in inspected_paths]
    return {
        "mode": "DRY_RUN_ONLY",
        "static_check": "PASS",
        "gate3_static_check": "PASS",
        "design_commit": "e3e1367efd913b601a70328a815d88c20af6d147",
        "latest_preregistration_utc": "2026-08-07T02:48:27Z",
        "inspected": inspected,
        "activation_created": False,
        "persistent_study_root_created": False,
        "real_order_path": False,
        "network": False,
        "seed_acquisition": False,
    }


def gate3_synthetic_fixture() -> dict[str, Any]:
    """Build the deterministic 300-ticker C1-equivalent local fixture."""
    calendar = pd.bdate_range("2019-01-02", periods=264)
    engine_day = calendar[252]
    tickers = [f"T{index:03d}" for index in range(300)]
    universe = pd.DataFrame({
        "ticker": tickers,
        "market": ["JP"] * len(tickers),
        "industry": [f"IND{index:03d}" for index in range(len(tickers))],
    })
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        close = 1000.0 + np.arange(len(calendar), dtype=float)
        volume = np.full(len(calendar), 100000.0)
        volume[252] = 200000.0
        frames[ticker] = pd.DataFrame({
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }, index=calendar)
    seed_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for index, day in enumerate(calendar[:252]):
            price = 1000.0 + index
            seed_rows.append({
                "ticker": ticker,
                "trading_date": day.strftime("%Y-%m-%d"),
                "raw_open": price,
                "raw_high": price + 2.0,
                "raw_low": price - 2.0,
                "raw_close": price,
                "adj_close": price,
                "raw_volume": 100000.0,
            })
    return {
        "frames": frames,
        "universe": universe,
        "split_history": {ticker: set() for ticker in tickers},
        "study_calendar": calendar,
        "engine_day": engine_day,
        "seed_rows": seed_rows,
        "collector_commit": "b" * 40,
    }


def _gate3_summary(result: dict[str, Any]) -> dict[str, Any]:
    cases = result["case_results"]
    case_by_id = {case["case_id"]: case for case in cases}
    if len(cases) != 12 or result["case_pass_count"] != 12 or result["case_fail_count"] != 0:
        raise AssertionError("GATE3_CASES_NOT_ALL_PASS")
    if case_by_id[2]["status"] != "PASS":
        raise AssertionError("GATE3_CANDIDATE_PARITY_NOT_PASS")
    if result["control_input_hashes"] != result["variant_input_hashes"]:
        raise AssertionError("GATE3_ARM_INPUT_HASH_MISMATCH")
    return {
        "verdict": "V7_FORWARD_CAPACITY_GATE3_DRY_RUN_PASS",
        "mode": result["mode"],
        "activation_status": result["activation_status"],
        "activation_boundary": result["activation_boundary"],
        "design_commit": result["design_commit"],
        "latest_preregistration_utc": result["latest_preregistration_utc"],
        "case_pass_count": result["case_pass_count"],
        "case_fail_count": result["case_fail_count"],
        "candidate_generation_count": result["candidate_generation_count"],
        "candidate_parity": "PASS",
        "candidate_future_reads": result["case_results"][0]["details"]["future_candidate_data_access"],
        "future_split_reads": result["case_results"][0]["details"]["future_split_access"],
        "seed_canonical_sha256": result["seed_canonical_sha256"],
        "price_snapshot_sha256": result["price_snapshot_sha256"],
        "candidate_snapshot_sha256": result["candidate_snapshot_sha256"],
        "market_gate_snapshot_sha256": result["market_gate_snapshot_sha256"],
        "control_input_hashes": result["control_input_hashes"],
        "variant_input_hashes": result["variant_input_hashes"],
        "arm_input_hash_equal": result["arm_input_hash_equal"],
        "control_state_sha256": result["control_state_sha256"],
        "variant_state_sha256": result["variant_state_sha256"],
        "control_max_open_positions": result["control_max_open_positions"],
        "variant_max_open_positions": result["variant_max_open_positions"],
        "single_changed_parameter": result["single_changed_parameter"],
        "state_objects_independent": result["state_objects_independent"],
        "rank21_promotion": case_by_id[9]["details"]["rank21_promoted"],
        "case6_block_reason": case_by_id[6]["details"]["observed_block_reason"],
        "case7_block_reason": case_by_id[7]["details"]["observed_block_reason"],
        "case8_block_reason": case_by_id[8]["details"]["observed_block_reason"],
        "pre_activation_persisted_study_events": result["pre_activation_persisted_study_events"],
        "network_requests": result["network_requests"],
        "seed_acquisition": result["seed_acquisition"],
        "real_data_read": result["real_data_read"],
        "historical_replay": result["historical_replay"],
        "real_portfolio_simulation": result["real_portfolio_simulation"],
        "two_pass_byte_identical": True,
        "temporary_output_removed": True,
    }


def run_gate3_dry_run_cli() -> dict[str, Any]:
    fixture = gate3_synthetic_fixture()
    arguments = (
        fixture["frames"], fixture["universe"], fixture["split_history"],
        fixture["study_calendar"], fixture["engine_day"], fixture["seed_rows"],
        fixture["collector_commit"],
    )
    first = run_gate3_dry_run(*arguments)
    second = run_gate3_dry_run(*arguments)
    first_bytes = canonical_json_bytes(first)
    second_bytes = canonical_json_bytes(second)
    if first_bytes != second_bytes:
        raise AssertionError("GATE3_INTERNAL_TWO_PASS_MISMATCH")
    for field in (
        "case_results", "candidate_snapshot_sha256", "price_snapshot_sha256",
        "market_gate_snapshot_sha256", "seed_canonical_sha256",
        "control_input_hashes", "variant_input_hashes", "control_state_sha256",
        "variant_state_sha256", "enriched_event_audit",
    ):
        if first[field] != second[field]:
            raise AssertionError("GATE3_TWO_PASS_FIELD_MISMATCH:" + field)
    summary = _gate3_summary(first)
    expected_bytes = canonical_json_bytes(summary)
    with tempfile.TemporaryDirectory(prefix="v7-gate3-") as temporary:
        output = Path(temporary) / "summary.json"
        output.write_bytes(expected_bytes)
        read_back = output.read_bytes()
        if read_back != expected_bytes:
            raise AssertionError("GATE3_SUMMARY_READBACK_MISMATCH")
        if hashlib.sha256(read_back).hexdigest() != hashlib.sha256(expected_bytes).hexdigest():
            raise AssertionError("GATE3_SUMMARY_HASH_MISMATCH")
        output.unlink()
        if output.exists():
            raise AssertionError("GATE3_TEMPORARY_OUTPUT_REMAINS")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 offline-only checks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-golden-test", action="store_true")
    group.add_argument("--gate2-static-check", action="store_true")
    group.add_argument("--gate3-dry-run", action="store_true")
    group.add_argument("--gate3-static-check", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic_golden_test:
        result = run_synthetic_golden()
    elif args.gate2_static_check:
        result = run_static_check()
    elif args.gate3_dry_run:
        result = run_gate3_dry_run_cli()
    else:
        result = run_static_check()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
