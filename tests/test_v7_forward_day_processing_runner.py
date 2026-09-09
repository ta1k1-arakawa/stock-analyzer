from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from scripts import check_v7_forward_day_processing as cli
from src import v7_forward_day_processing as processing
from src.v7_capacity_engine import CausalEventEngine, V7EngineParameters
from src.v7_forward_persistence import ForwardStudyStore, canonical_json_bytes, export_engine_runtime
from src.v7_jpx_calendar import load_calendar_snapshot

from tests.test_v7_forward_day_processing import (
    ENGINE_DAYS,
    SNAPSHOT,
    TICKERS,
    activation_context,
    day_price,
    fake_candidate_result,
    make_acquisition,
    patch_candidates,
    run_day,
    seed_rows_for,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_v7_forward_day_processing.py"
PYTHON = sys.executable
MULTI_DAY_COUNT = 12


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_has_exactly_one_authorized_option():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }
    assert options == {"--synthetic-processing-test"}


def test_cli_has_no_real_study_root_network_or_activation_option():
    text = SCRIPT.read_text(encoding="utf-8")
    for forbidden_flag in ("--study-root", "--network", "--activate", "--activation", "--real", "--evaluate", "--order"):
        assert forbidden_flag not in text


def test_cli_module_performs_no_urlopen():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert calls == []


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run([PYTHON, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--real-study-root"], cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0


def test_module_source_has_no_network_imports():
    text = Path(processing.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines)


def test_module_source_has_no_activation_creation_tokens():
    text = Path(processing.__file__).read_text(encoding="utf-8")
    for token in ("activation_authorization", "build_activation_manifest", "place_order", "real_order"):
        assert token not in text


def _executable_identifiers(path: Path) -> set[str]:
    """Names, attributes and literal dict keys, ignoring comments and docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    identifiers: set[str] = set()
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings:
            identifiers.add(node.value)
    return {value.lower() for value in identifiers}


def test_module_exposes_no_profit_aggregation_identifiers():
    identifiers = _executable_identifiers(Path(processing.__file__))
    for token in ("realized_net_profit", "profit_factor", "win_rate", "drawdown", "max_dd", "net_profit"):
        offending = [value for value in identifiers if token in value]
        assert offending == [], token


def test_cli_exposes_no_profit_aggregation_identifiers():
    identifiers = _executable_identifiers(SCRIPT)
    for token in ("realized_net_profit", "profit_factor", "win_rate", "drawdown", "max_dd", "net_profit"):
        offending = [
            value for value in identifiers
            if token in value and value != "profit_metrics_exposed"
        ]
        assert offending == [], token


def test_module_constants_bind_fixed_lineage():
    assert processing.EXPECTED_CALENDAR_COMMIT == "03ce048b0eedca632f79ad925a627cb9e967d78d"
    assert processing.EXPECTED_COLLECTOR_COMMIT == "4ca41c53895e75910ae65809fea6018868929afa"
    assert processing.CONTROL_PARAMETERS_SHA256 == V7EngineParameters.control().sha256()
    assert processing.CAPACITY_3_PARAMETERS_SHA256 == V7EngineParameters.capacity_3().sha256()
    assert processing.PRICE_SCHEMA_VERSION == "V7_FORWARD_PROCESSING_PRICE_V1"
    assert processing.CANDIDATE_SCHEMA_VERSION == "V7_FORWARD_PROCESSING_CANDIDATE_V1"
    assert processing.MARKET_GATE_SCHEMA_VERSION == "V7_FORWARD_PROCESSING_MARKET_GATE_V1"


def test_module_reuses_accepted_primitives_without_reimplementation():
    text = Path(processing.__file__).read_text(encoding="utf-8")
    for primitive in (
        "validate_universe_file", "validate_seed_rows", "load_calendar_snapshot",
        "next_jpx_trading_day", "generate_engine_days", "is_jpx_trading_day",
        "verify_daily_acquisition_bundle", "build_forward_frames_from_seed_and_d0",
        "generate_forward_candidates_for_day", "validate_single_parameter_difference",
        "CausalEventEngine", "export_engine_runtime", "restore_engine_runtime",
        "ForwardStudyStore", "verify_forward_store",
    ):
        assert primitive in text


# ---------------------------------------------------------------------------
# Real candidate-generator integration (no monkeypatched candidate fixture)
# ---------------------------------------------------------------------------



def test_real_candidate_generator_integration_end_to_end():
    """Drive the actual seed bridge + candidate generator over 300 tickers."""
    result = cli.run_synthetic_processing_test()
    assert result["status"] == "PASS"
    assert result["candidate_generation_pass"] is True
    assert result["future_candidate_data_access_count"] == 0
    assert result["future_split_access_count"] == 0
    assert result["processing_verification"] == "PASS"
    assert result["network_requests"] == 0
    assert result["activation_created"] is False



def test_real_candidate_generator_uses_only_d1_and_d10_calendar_days(tmp_path):
    snapshot = cli.synthetic_calendar_snapshot()
    calendar = load_calendar_snapshot(snapshot)
    tickers = cli.universe_tickers()
    seed_rows = cli.synthetic_seed_rows(tickers, cli.seed_trading_days(snapshot))
    context = cli.build_activation_context(seed_rows, tickers)
    engine_day = cli.ACTIVATION_BOUNDARY
    cli.build_acquisition_bundle(tmp_path, snapshot, engine_day, cli._price_for(cli.SEED_OBSERVATION_COUNT))
    processing.process_forward_day(
        study_root=tmp_path,
        engine_day=engine_day,
        universe_csv=cli.UNIVERSE_CSV,
        calendar_snapshot=snapshot,
        seed_rows=seed_rows,
        activation_context=context,
    )
    payload = json.loads(
        (ForwardStudyStore(tmp_path).days_root / engine_day / "candidate_snapshot.json").read_text(encoding="utf-8")
    )
    engine_days = cli.engine_days_from_boundary(snapshot, 12)
    assert payload["entry_attempt_date"] == engine_days[1]
    assert payload["planned_exit_date"] == engine_days[10]
    assert payload["accepted_top20"]
    for row in payload["accepted_top20"]:
        assert row["signal_date"] == engine_day
        assert row["candidate_status"] == "ACCEPTED_TOP20"



def test_real_candidate_generator_price_frames_never_exceed_engine_day(tmp_path):
    snapshot = cli.synthetic_calendar_snapshot()
    tickers = cli.universe_tickers()
    seed_rows = cli.synthetic_seed_rows(tickers, cli.seed_trading_days(snapshot))
    context = cli.build_activation_context(seed_rows, tickers)
    engine_day = cli.ACTIVATION_BOUNDARY
    cli.build_acquisition_bundle(tmp_path, snapshot, engine_day, cli._price_for(cli.SEED_OBSERVATION_COUNT))
    processing.process_forward_day(
        study_root=tmp_path,
        engine_day=engine_day,
        universe_csv=cli.UNIVERSE_CSV,
        calendar_snapshot=snapshot,
        seed_rows=seed_rows,
        activation_context=context,
    )
    price_payload = json.loads(
        (ForwardStudyStore(tmp_path).days_root / engine_day / "price_snapshot.json").read_text(encoding="utf-8")
    )
    assert {row["trading_date"] for row in price_payload["d0_price_rows"]} == {engine_day}


# ---------------------------------------------------------------------------
# Multi-day orchestration equivalence
# ---------------------------------------------------------------------------


def _multi_day_days() -> list[str]:
    return ENGINE_DAYS[:MULTI_DAY_COUNT]


def _all_candidates(days) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in days:
        for row in fake_candidate_result(day, candidate_count=_candidate_count_for(day))["accepted_top20"]:
            rows.append({
                "signal_year": int(row["signal_year"]),
                "signal_date": row["signal_date"],
                "ticker": row["ticker"],
                "industry": row["industry"],
                "rank": int(row["rank"]),
                "signal_raw_close": float(row["raw_close"]),
                "entry_attempt_date": row["entry_date"],
                "planned_exit_date": row["exit_date"],
                "candidate_status": "ACCEPTED_TOP20",
            })
    return rows


def _candidate_count_for(day: str) -> int:
    return 3 if day == ENGINE_DAYS[0] else 0


def _reference_frames(days) -> dict[str, dict[str, dict[str, float]]]:
    """Full-history nested frames for the uninterrupted reference engines."""
    frames: dict[str, dict[str, dict[str, float]]] = {}
    seed_days = sorted({row["trading_date"] for row in seed_rows_for()})
    for ticker in TICKERS:
        rows: dict[str, dict[str, float]] = {}
        for index, seed_day in enumerate(seed_days):
            price = 1000.0 + float(index)
            rows[seed_day] = {
                "Open": price, "High": price + 2.0, "Low": price - 2.0,
                "Close": price, "Adj Close": price, "Volume": 100000.0,
            }
        for day in days:
            price = day_price(day)
            rows[day] = {
                "Open": price, "High": price + 2.0, "Low": price - 2.0,
                "Close": price, "Adj Close": price, "Volume": 200000.0,
            }
        frames[ticker] = rows
    return frames


def _reference_runtimes(days) -> tuple[dict[str, Any], dict[str, Any]]:
    calendar = ENGINE_DAYS
    frames = _reference_frames(days)
    candidates = _all_candidates(days)
    control = CausalEventEngine(frames, tuple(calendar), candidates, V7EngineParameters.control())
    variant = CausalEventEngine(frames, tuple(calendar), candidates, V7EngineParameters.capacity_3())
    for day in days:
        control.process_day(day)
        variant.process_day(day)
    return export_engine_runtime(control), export_engine_runtime(variant)


@pytest.fixture(scope="module")
def multi_day_run(tmp_path_factory):
    """Drive B: restore-per-day orchestration over 12 consecutive JPX engine days."""
    days = _multi_day_days()
    study_root = tmp_path_factory.mktemp("v7-multi-day")

    def per_day(engine_day: str) -> dict[str, Any]:
        return fake_candidate_result(engine_day, candidate_count=_candidate_count_for(engine_day))

    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_candidates(monkeypatch, override=per_day)
        for day in days:
            make_acquisition(study_root, day)
            run_day(study_root, day)

    latest = ForwardStudyStore(study_root).load_latest_runtime()
    assert latest["day"] == days[-1]
    reference_control, reference_variant = _reference_runtimes(days)
    return {
        "days": days,
        "study_root": study_root,
        "orchestrated_control": latest["arm_a_runtime"],
        "orchestrated_variant": latest["arm_b_runtime"],
        "reference_control": reference_control,
        "reference_variant": reference_variant,
    }


def test_twelve_day_control_runtime_byte_identical(multi_day_run):
    assert canonical_json_bytes(multi_day_run["orchestrated_control"]) == canonical_json_bytes(
        multi_day_run["reference_control"]
    )


def test_twelve_day_capacity3_runtime_byte_identical(multi_day_run):
    assert canonical_json_bytes(multi_day_run["orchestrated_variant"]) == canonical_json_bytes(
        multi_day_run["reference_variant"]
    )


@pytest.mark.parametrize("field", [
    "pending_orders_by_entry_date", "open_positions", "pending_proceeds_by_available_date",
    "completed_trades", "daily_equity", "event_audit", "safety_counters", "skip_reason_counts",
])
def test_twelve_day_state_field_equivalence(multi_day_run, field):
    assert multi_day_run["orchestrated_control"][field] == multi_day_run["reference_control"][field]
    assert multi_day_run["orchestrated_variant"][field] == multi_day_run["reference_variant"][field]


def test_twelve_day_arms_are_not_identical(multi_day_run):
    assert multi_day_run["orchestrated_control"]["completed_trades"] != (
        multi_day_run["orchestrated_variant"]["completed_trades"]
    )


def test_twelve_day_orchestration_exercises_full_trade_lifecycle(multi_day_run):
    control = multi_day_run["orchestrated_control"]
    variant = multi_day_run["orchestrated_variant"]
    assert any(row["status"] == "CLOSED" for row in control["completed_trades"])
    assert len(control["daily_equity"]) == MULTI_DAY_COUNT
    assert len(variant["daily_equity"]) == MULTI_DAY_COUNT


def test_twelve_day_store_has_exactly_twelve_complete_days(multi_day_run):
    store = ForwardStudyStore(multi_day_run["study_root"])
    assert store._final_days() == multi_day_run["days"]
    result = processing.verify_processed_forward_day(
        study_root=multi_day_run["study_root"],
        engine_day=multi_day_run["days"][-1],
        universe_csv=cli.UNIVERSE_CSV,
        activation_context=activation_context(),
    )
    assert result["status"] == "PASS"
    assert result["verified_day_count"] == MULTI_DAY_COUNT


def test_twelve_day_checkpoint_chain_is_continuous(multi_day_run):
    store = ForwardStudyStore(multi_day_run["study_root"])
    previous = None
    for day in multi_day_run["days"]:
        checkpoint = json.loads((store.days_root / day / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["previous_checkpoint_sha256"] == previous
        assert checkpoint["last_completed_engine_day"] == day
        previous = checkpoint["current_checkpoint_sha256"]


def test_twelve_day_each_persisted_day_binds_its_predecessor(multi_day_run):
    store = ForwardStudyStore(multi_day_run["study_root"])
    days = multi_day_run["days"]
    for index, day in enumerate(days):
        payload = json.loads((store.days_root / day / "price_snapshot.json").read_text(encoding="utf-8"))
        expected = None if index == 0 else days[index - 1]
        assert payload["previous_complete_engine_day"] == expected
