from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from src import v7_forward_day_processing as processing
from src import v7_forward_operations as operations
from src.v7_activation_manifest import (
    HUMAN_ACTIVATION_CONFIRMATION,
    SeedProvenanceExpectation,
    build_activation_manifest_candidate,
    canonical_json_bytes as manifest_canonical_json_bytes,
    compute_manifest_sha256,
    expected_activation_boundary,
    read_activation_manifest,
    read_seed_csv_rows,
    validate_seed_provenance,
    write_activation_manifest_once,
)
from src.v7_daily_acquisition import acquire_daily_bundle
from src.v7_forward_persistence import ForwardStudyStore, canonical_json_bytes
from src.v7_jpx_calendar import generate_engine_days, load_calendar_snapshot, next_jpx_trading_day
from src.v7_seed_acquisition import validate_universe_file

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "V4_UNIVERSE.csv"
CALENDAR_PATH = ROOT / "data" / "v7_jpx_calendar_2026_2027.json"

AUTHORIZATION_UTC = "2026-08-07T09:00:00Z"
SEED_ACQUISITION_UTC = "2026-08-07T03:10:00Z"
ACQUISITION_WINDOW_JST = "17:00-18:00 Asia/Tokyo"
SEED_OBSERVATION_COUNT = 252
SEED_CSV_COLUMNS = (
    "ticker", "trading_date", "raw_open", "raw_high", "raw_low",
    "raw_close", "adj_close", "raw_volume",
)
CANDIDATE_INDUSTRIES = ("SYN_TECH", "SYN_FINANCE", "SYN_ENERGY", "SYN_RETAIL")

TICKERS = validate_universe_file(UNIVERSE_CSV)["tickers"]
SNAPSHOT = load_calendar_snapshot(CALENDAR_PATH)
BOUNDARY = expected_activation_boundary(SNAPSHOT, AUTHORIZATION_UTC)
ENGINE_DAYS = generate_engine_days(SNAPSHOT, BOUNDARY, "2027-06-30")[:20]


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


# ---------------------------------------------------------------------------
# Deterministic fake candidate generation (real generator covered elsewhere)
# ---------------------------------------------------------------------------


def day_price(day: str) -> float:
    return 1000.0 + float(SEED_OBSERVATION_COUNT + ENGINE_DAYS.index(day))


def fake_candidate_result(engine_day: str, *, candidate_count: int = 3) -> dict[str, Any]:
    index = ENGINE_DAYS.index(engine_day)
    entry_date = ENGINE_DAYS[index + 1]
    exit_date = ENGINE_DAYS[index + 10]
    price = day_price(engine_day)
    accepted = [
        {
            "candidate_status": "ACCEPTED_TOP20",
            "engine_day": engine_day,
            "signal_date": engine_day,
            "signal_year": int(engine_day[:4]),
            "ticker": TICKERS[position],
            "industry": CANDIDATE_INDUSTRIES[position],
            "rank": position + 1,
            "raw_close": price,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "collector_commit": operations.EXPECTED_COLLECTOR_COMMIT,
        }
        for position in range(candidate_count)
    ]
    market_gate = {
        "engine_day": engine_day,
        "market_gate_status": "MARKET_GATE_PASS",
        "market_denominator_count": len(TICKERS),
        "breadth_above_ma60": 1.0,
        "cross_sectional_median_return20": 0.01,
    }
    return {
        "engine_day": engine_day,
        "market_gate": market_gate,
        "accepted_top20": accepted,
        "full_candidate_audit": list(accepted),
        "candidate_snapshot_sha256": hashlib.sha256(canonical_json_bytes(accepted)).hexdigest(),
        "market_gate_snapshot_sha256": hashlib.sha256(canonical_json_bytes(market_gate)).hexdigest(),
        "price_snapshot_sha256": hashlib.sha256(canonical_json_bytes({"engine_day": engine_day})).hexdigest(),
        "future_candidate_data_access_count": 0,
        "future_split_access_count": 0,
        "entry_attempt_date": entry_date,
        "planned_exit_date": exit_date,
    }


def patch_candidates(monkeypatch, *, candidate_count: int = 3) -> None:
    def fake(frames, universe, split_history, study_calendar, engine_day, collector_commit):
        return fake_candidate_result(engine_day, candidate_count=candidate_count)

    monkeypatch.setattr(processing, "generate_forward_candidates_for_day", fake)


# ---------------------------------------------------------------------------
# Synthetic seed / acquisition / activation-manifest fixture helpers
# ---------------------------------------------------------------------------


def seed_observation_days(boundary: str = BOUNDARY, count: int = SEED_OBSERVATION_COUNT) -> list[str]:
    from datetime import date, timedelta

    days: list[str] = []
    current = date.fromisoformat(boundary) - timedelta(days=1)
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current -= timedelta(days=1)
    return sorted(days)


def synthetic_seed_rows(tickers: Sequence[str], days: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for index, day in enumerate(days):
            price = 1000.0 + float(index)
            rows.append({
                "ticker": ticker, "trading_date": day,
                "raw_open": price, "raw_high": price + 2.0, "raw_low": price - 2.0,
                "raw_close": price, "adj_close": price, "raw_volume": 100000.0,
            })
    return rows


def write_seed_csv(path: Path, rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SEED_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (str(item["ticker"]), str(item["trading_date"]))):
        writer.writerow({column: row[column] for column in SEED_CSV_COLUMNS})
    payload = stream.getvalue().encode("utf-8")
    path.write_bytes(payload)
    return payload


def synthetic_seed_acquisition_manifest(tickers: Sequence[str]) -> dict[str, Any]:
    import hashlib

    return {
        "mode": "PRE_ACTIVATION_SEED_ACQUISITION",
        "payload_manifest": [
            {"ticker": t, "payload_sha256": hashlib.sha256(("x:" + t).encode()).hexdigest(), "byte_count": 1000 + i}
            for i, t in enumerate(tickers)
        ],
    }


@pytest.fixture(scope="module")
def seed_fixture(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("v7-ops-seed")
    days = seed_observation_days()
    rows = synthetic_seed_rows(TICKERS, days)
    seed_csv = workspace / "seed.csv"
    write_seed_csv(seed_csv, rows)
    seed_acquisition_manifest = synthetic_seed_acquisition_manifest(TICKERS)
    seed_provenance = validate_seed_provenance(
        universe_csv=UNIVERSE_CSV, seed_csv=seed_csv,
        seed_acquisition_manifest=seed_acquisition_manifest,
        activation_boundary_first_jpx_trading_date=BOUNDARY, expected=None,
    )
    expectation = SeedProvenanceExpectation(**{
        field: seed_provenance[field]
        for field in (
            "seed_source_payload_manifest_sha256", "seed_ticker_manifest_sha256",
            "seed_canonical_csv_sha256", "seed_ticker_count", "seed_row_count",
            "seed_cutoff_trading_date",
        )
    })
    return {
        "seed_csv": seed_csv,
        "seed_acquisition_manifest": seed_acquisition_manifest,
        "seed_provenance": seed_provenance,
        "expectation": expectation,
    }


def build_manifest(seed_fixture: dict[str, Any], durable_root: Path, **overrides) -> tuple[Path, dict[str, Any]]:
    """Build + write a fresh activation manifest whose output_root == durable_root."""
    kwargs = dict(
        activation_authorization_utc=AUTHORIZATION_UTC,
        activation_boundary_first_jpx_trading_date=BOUNDARY,
        acquisition_window_jst=ACQUISITION_WINDOW_JST,
        output_root=str(durable_root.resolve()),
        seed_acquisition_utc=SEED_ACQUISITION_UTC,
        seed_provenance=seed_fixture["seed_provenance"],
    )
    kwargs.update(overrides)
    manifest = build_activation_manifest_candidate(**kwargs)
    manifest_path = durable_root.parent / (durable_root.name + "-manifest.json")
    write_activation_manifest_once(
        output_path=manifest_path,
        manifest=manifest,
        repository_root=ROOT,
        confirmation=HUMAN_ACTIVATION_CONFIRMATION,
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=seed_fixture["seed_csv"],
        seed_acquisition_manifest=seed_fixture["seed_acquisition_manifest"],
        expected_seed_provenance=seed_fixture["expectation"],
    )
    return manifest_path, manifest


@pytest.fixture(scope="module")
def empty_env(tmp_path_factory, seed_fixture):
    """A pristine (never processes any engine day) manifest + durable root,
    shared by every test that must BLOCK before touching acquisition/processing."""
    durable_root = tmp_path_factory.mktemp("v7-ops-empty-durable")
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    return {"durable_root": durable_root, "manifest_path": manifest_path, "manifest": manifest}


def within_window_clock(day: str = BOUNDARY, hour: int = 8, minute: int = 30) -> datetime:
    return datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=timezone.utc)


def fake_opener_for(day: str, index: int):
    from scripts.run_v7_forward_operations import FakeAcquisitionOpener, _price_for

    return FakeAcquisitionOpener(day, _price_for(SEED_OBSERVATION_COUNT + index))


def run_day(env: dict[str, Any], seed_fixture: dict[str, Any], engine_day: str, index: int, **overrides) -> dict[str, Any]:
    kwargs = dict(
        activation_manifest_path=env["manifest_path"],
        durable_output_root=env["durable_root"],
        universe_csv=UNIVERSE_CSV,
        calendar_path=CALENDAR_PATH,
        seed_csv=seed_fixture["seed_csv"],
        seed_acquisition_manifest=seed_fixture["seed_acquisition_manifest"],
        engine_day=engine_day,
        repository_root=ROOT,
        opener=fake_opener_for(engine_day, index),
        clock=lambda: within_window_clock(engine_day),
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda seconds: None,
        expected_seed_provenance=seed_fixture["expectation"],
    )
    kwargs.update(overrides)
    return operations.run_forward_operations_day(**kwargs)


# ---------------------------------------------------------------------------
# Activation manifest gate
# ---------------------------------------------------------------------------


def test_valid_activated_manifest_and_valid_day_pass(tmp_path, seed_fixture, monkeypatch):
    patch_candidates(monkeypatch)
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    env = {"durable_root": durable_root, "manifest_path": manifest_path, "manifest": manifest}
    result = run_day(env, seed_fixture, BOUNDARY, 0)
    assert result["status"] == "PASS"
    assert result["already_committed"] is False
    assert result["activation_manifest_verified"] is True
    assert result["acquisition_verified"] is True
    assert result["processing_verified"] is True
    assert result["persistence_verified"] is True


def test_activation_status_not_activated_blocked(empty_env, seed_fixture, monkeypatch):
    patch_candidates(monkeypatch)
    tampered = {**empty_env["manifest"], "activation_status": "NOT_ACTIVATED"}
    tampered["manifest_sha256"] = compute_manifest_sha256(tampered)
    manifest_path = empty_env["durable_root"].parent / "tampered-status-manifest.json"
    manifest_path.write_bytes(manifest_canonical_json_bytes(tampered))
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({**empty_env, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason.startswith("ACTIVATION_MANIFEST_VALIDATION_FAILED:")


def test_tampered_activation_manifest_hash_blocked(empty_env, seed_fixture, monkeypatch):
    patch_candidates(monkeypatch)
    tampered = {**empty_env["manifest"], "manifest_sha256": "9" * 64}
    manifest_path = empty_env["durable_root"].parent / "hash-tampered-manifest.json"
    manifest_path.write_bytes(manifest_canonical_json_bytes(tampered))
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({**empty_env, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_VALIDATION_FAILED:MANIFEST_SHA_MISMATCH"


def test_tampered_activation_manifest_ticker_count_blocked(empty_env, seed_fixture, monkeypatch):
    patch_candidates(monkeypatch)
    tampered = {**empty_env["manifest"], "ticker_count": 299}
    tampered["manifest_sha256"] = compute_manifest_sha256(tampered)
    manifest_path = empty_env["durable_root"].parent / "ticker-count-tampered-manifest.json"
    manifest_path.write_bytes(manifest_canonical_json_bytes(tampered))
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({**empty_env, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason.startswith("ACTIVATION_MANIFEST_VALIDATION_FAILED:")


def test_missing_activation_manifest_file_blocked(empty_env, seed_fixture):
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(
            {**empty_env, "manifest_path": empty_env["durable_root"].parent / "absent.json"},
            seed_fixture, BOUNDARY, 0,
        )
    assert excinfo.value.reason.startswith("ACTIVATION_MANIFEST_READ_FAILED:")


def test_load_and_verify_activation_manifest_returns_manifest(empty_env, seed_fixture):
    manifest = operations.load_and_verify_activation_manifest(
        activation_manifest_path=empty_env["manifest_path"],
        repository_root=ROOT,
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=seed_fixture["seed_csv"],
        seed_acquisition_manifest=seed_fixture["seed_acquisition_manifest"],
        expected_seed_provenance=seed_fixture["expectation"],
    )
    assert manifest["activation_status"] == "ACTIVATED"
    assert manifest["manifest_sha256"] == empty_env["manifest"]["manifest_sha256"]


def test_build_processing_activation_context_field_mapping(empty_env):
    manifest = empty_env["manifest"]
    context = operations.build_processing_activation_context(manifest)
    assert set(context) == set(processing.ACTIVATION_CONTEXT_FIELDS)
    assert context["activation_manifest_sha256"] == manifest["manifest_sha256"]
    assert context["expected_seed_canonical_sha256"] == manifest["seed_canonical_csv_sha256"]
    assert context["expected_seed_ticker_manifest_sha256"] == manifest["seed_ticker_manifest_sha256"]
    assert context["collector_commit"] == manifest["collector_commit"]


# ---------------------------------------------------------------------------
# Durable output root
# ---------------------------------------------------------------------------


def test_durable_output_root_mismatch_blocked(empty_env, seed_fixture):
    other_root = empty_env["durable_root"].parent / "other-root"
    other_root.mkdir()
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, BOUNDARY, 0, durable_output_root=other_root)
    assert excinfo.value.reason == "DURABLE_OUTPUT_ROOT_MISMATCH"


def test_durable_output_root_not_found_blocked(seed_fixture, tmp_path):
    durable_root = tmp_path / "not-yet-created"
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    assert not durable_root.exists()
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({"durable_root": durable_root, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason == "DURABLE_OUTPUT_ROOT_NOT_FOUND"


def test_durable_output_root_inside_repository_blocked(seed_fixture, tmp_path):
    inside_repo = ROOT / "tmp-v7-ops-test-root"
    try:
        manifest_path, manifest = build_manifest(seed_fixture, inside_repo, output_root=str(inside_repo))
        assert False, "manifest build should have blocked"
    except Exception as error:
        assert "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY" in str(error)
    finally:
        if inside_repo.exists():
            inside_repo.rmdir()


def test_require_durable_output_root_helper_matches_and_writable(empty_env):
    result = operations.require_durable_output_root(empty_env["durable_root"], empty_env["manifest"], ROOT)
    assert result == Path(empty_env["durable_root"])


def test_require_durable_output_root_helper_mismatch(empty_env, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        operations.require_durable_output_root(other, empty_env["manifest"], ROOT)
    assert excinfo.value.reason == "DURABLE_OUTPUT_ROOT_MISMATCH"


# ---------------------------------------------------------------------------
# Engine day / calendar gate
# ---------------------------------------------------------------------------


def test_before_activation_boundary_blocked(empty_env, seed_fixture):
    before = ENGINE_DAYS[0]
    # any real trading day strictly earlier than BOUNDARY
    earlier_day = "2026-08-07"
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, earlier_day, 0)
    assert excinfo.value.reason == "ENGINE_DAY_BEFORE_ACTIVATION_BOUNDARY"


def test_non_trading_day_blocked(empty_env, seed_fixture):
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, "2026-08-15", 0)  # Saturday
    assert excinfo.value.reason == "ENGINE_DAY_NOT_JPX_TRADING_DAY"


def test_market_holiday_blocked(empty_env, seed_fixture):
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, "2026-08-11", 0)  # Mountain Day
    assert excinfo.value.reason == "ENGINE_DAY_NOT_JPX_TRADING_DAY"


def test_engine_day_outside_calendar_coverage_blocked(empty_env, seed_fixture):
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, "2028-01-04", 0)
    assert excinfo.value.reason == "ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE"


def test_engine_day_exactly_boundary_ready(empty_env):
    operations.require_engine_day_ready(BOUNDARY, empty_env["manifest"], CALENDAR_PATH)


# ---------------------------------------------------------------------------
# Acquisition window gate
# ---------------------------------------------------------------------------


def test_outside_acquisition_window_before_open_blocked(empty_env, seed_fixture):
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, BOUNDARY, 0, clock=lambda: within_window_clock(BOUNDARY, hour=1, minute=0))
    assert excinfo.value.reason == "ACQUISITION_WINDOW_NOT_OPEN"


def test_outside_acquisition_window_after_close_blocked(empty_env, seed_fixture):
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(empty_env, seed_fixture, BOUNDARY, 0, clock=lambda: within_window_clock(BOUNDARY, hour=10, minute=0))
    assert excinfo.value.reason == "ACQUISITION_WINDOW_NOT_OPEN"


def test_exactly_at_window_start_passes():
    operations.require_within_acquisition_window(within_window_clock(BOUNDARY, 8, 0), ACQUISITION_WINDOW_JST)


def test_exactly_at_window_end_blocked():
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        operations.require_within_acquisition_window(within_window_clock(BOUNDARY, 9, 0), ACQUISITION_WINDOW_JST)
    assert excinfo.value.reason == "ACQUISITION_WINDOW_NOT_OPEN"


def test_naive_clock_blocked():
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        operations.require_within_acquisition_window(datetime(2026, 8, 10, 8, 30), ACQUISITION_WINDOW_JST)
    assert excinfo.value.reason == "OPERATIONS_CLOCK_INVALID"


def test_malformed_window_string_blocked():
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        operations.require_within_acquisition_window(within_window_clock(BOUNDARY), "not-a-window")
    assert excinfo.value.reason.startswith("ACQUISITION_WINDOW_INVALID:")


# ---------------------------------------------------------------------------
# Idempotence / atomicity / state probe
# ---------------------------------------------------------------------------


def test_probe_engine_day_state_none(empty_env):
    assert operations.probe_engine_day_state(empty_env["durable_root"], BOUNDARY) == "NONE"


def test_probe_engine_day_state_partial_acquisition_only(tmp_path):
    root = tmp_path / "root"
    (root / "acquisitions" / BOUNDARY).mkdir(parents=True)
    assert operations.probe_engine_day_state(root, BOUNDARY) == "PARTIAL"


def test_probe_engine_day_state_partial_forward_only(tmp_path):
    root = tmp_path / "root"
    (root / "days" / BOUNDARY).mkdir(parents=True)
    assert operations.probe_engine_day_state(root, BOUNDARY) == "PARTIAL"


def test_probe_engine_day_state_complete(tmp_path):
    root = tmp_path / "root"
    (root / "acquisitions" / BOUNDARY).mkdir(parents=True)
    (root / "days" / BOUNDARY).mkdir(parents=True)
    assert operations.probe_engine_day_state(root, BOUNDARY) == "COMPLETE"


@pytest.fixture(scope="module")
def committed_env(tmp_path_factory, seed_fixture, request):
    """One fully committed engine day (BOUNDARY), shared read-only across
    idempotence/tamper/verification tests -- monkeypatch here is scoped for
    the module-fixture's own setup and does not leak into other tests."""
    from _pytest.monkeypatch import MonkeyPatch

    durable_root = tmp_path_factory.mktemp("v7-ops-committed-durable")
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    env = {"durable_root": durable_root, "manifest_path": manifest_path, "manifest": manifest}
    mp = MonkeyPatch()
    patch_candidates(mp)
    try:
        result = run_day(env, seed_fixture, BOUNDARY, 0)
    finally:
        mp.undo()
    assert result["status"] == "PASS"
    return env


def test_duplicate_completed_day_already_committed(committed_env, seed_fixture):
    result = run_day(committed_env, seed_fixture, BOUNDARY, 0)
    assert result["status"] == "ALREADY_COMMITTED"
    assert result["already_committed"] is True
    assert result["acquisition_verified"] is True
    assert result["processing_verified"] is True
    assert result["persistence_verified"] is True


def test_already_committed_does_not_call_opener(committed_env, seed_fixture):
    opener = fake_opener_for(BOUNDARY, 0)
    run_day(committed_env, seed_fixture, BOUNDARY, 0, opener=opener)
    assert opener.calls == []


def test_already_committed_does_not_mutate_persisted_bytes(committed_env, seed_fixture):
    store = ForwardStudyStore(committed_env["durable_root"])
    checkpoint_path = store.days_root / BOUNDARY / "checkpoint.json"
    before = checkpoint_path.read_bytes()
    run_day(committed_env, seed_fixture, BOUNDARY, 0)
    assert checkpoint_path.read_bytes() == before


def test_partial_day_forward_missing_blocked(tmp_path, seed_fixture, committed_env):
    """Simulate a crash between acquisition commit and forward-day commit."""
    import shutil

    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    env = {"durable_root": durable_root, "manifest_path": manifest_path}
    shutil.copytree(
        committed_env["durable_root"] / "acquisitions" / BOUNDARY,
        durable_root / "acquisitions" / BOUNDARY,
    )
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(env, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason == "PARTIAL_ENGINE_DAY_STATE"


def test_partial_day_acquisition_missing_blocked(tmp_path, seed_fixture, committed_env):
    import shutil

    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    env = {"durable_root": durable_root, "manifest_path": manifest_path}
    shutil.copytree(
        committed_env["durable_root"] / "days" / BOUNDARY,
        durable_root / "days" / BOUNDARY,
    )
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day(env, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason == "PARTIAL_ENGINE_DAY_STATE"


def test_acquisition_bundle_already_partially_exists_staging_remnant_blocked(tmp_path, seed_fixture):
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    (durable_root / "acquisitions").mkdir(parents=True)
    (durable_root / "acquisitions" / f"{BOUNDARY}.staging-remnant").mkdir()
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({"durable_root": durable_root, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert "PARTIAL_ACQUISITION_COMMIT" in excinfo.value.reason


# ---------------------------------------------------------------------------
# Tamper detection on an already-committed day
# ---------------------------------------------------------------------------


def test_tampered_acquisition_bundle_blocked(tmp_path, seed_fixture, committed_env):
    import shutil

    durable_root = tmp_path / "durable"
    shutil.copytree(committed_env["durable_root"], durable_root)
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    price_path = durable_root / "acquisitions" / BOUNDARY / "price_snapshot.json"
    price_path.write_text("[]", encoding="utf-8")
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({"durable_root": durable_root, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason.startswith("ACQUISITION_VERIFICATION_FAILED:")


def test_checkpoint_hash_chain_mismatch_blocked(tmp_path, seed_fixture, committed_env):
    import shutil

    durable_root = tmp_path / "durable"
    shutil.copytree(committed_env["durable_root"], durable_root)
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    checkpoint_path = durable_root / "days" / BOUNDARY / "checkpoint.json"
    record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record["current_checkpoint_sha256"] = "f" * 64
    checkpoint_path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(operations.V7ForwardOperationsBlocked) as excinfo:
        run_day({"durable_root": durable_root, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert excinfo.value.reason.startswith("PROCESSING_VERIFICATION_FAILED:")


# ---------------------------------------------------------------------------
# Restart equivalence (operations layer)
# ---------------------------------------------------------------------------


def test_restart_equivalence_two_day_sequence_passes(tmp_path, seed_fixture, monkeypatch):
    from src.v7_forward_persistence import export_engine_runtime, restore_engine_runtime
    from src.v7_capacity_engine import CausalEventEngine, V7EngineParameters

    patch_candidates(monkeypatch)
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    env = {"durable_root": durable_root, "manifest_path": manifest_path}

    day1 = BOUNDARY
    day2 = next_jpx_trading_day(SNAPSHOT, day1)
    result_1 = run_day(env, seed_fixture, day1, 0)
    assert result_1["status"] == "PASS"
    result_1_again = run_day(env, seed_fixture, day1, 0)
    assert result_1_again["status"] == "ALREADY_COMMITTED"
    result_2 = run_day(env, seed_fixture, day2, 1)
    assert result_2["status"] == "PASS"

    store = ForwardStudyStore(durable_root)
    latest = store.load_latest_runtime()
    assert latest["day"] == day2
    assert latest["arm_a_runtime"]["engine_day"] == day2
    assert latest["arm_b_runtime"]["engine_day"] == day2


def test_restart_equivalence_matches_direct_processing_layer(tmp_path, seed_fixture, monkeypatch):
    """The operations layer must produce byte-identical persisted state to
    calling v7_forward_day_processing directly for the same two days."""
    patch_candidates(monkeypatch)
    durable_root = tmp_path / "durable-ops"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    env = {"durable_root": durable_root, "manifest_path": manifest_path}

    day1 = BOUNDARY
    day2 = next_jpx_trading_day(SNAPSHOT, day1)
    run_day(env, seed_fixture, day1, 0)
    run_day(env, seed_fixture, day2, 1)
    ops_latest = ForwardStudyStore(durable_root).load_latest_runtime()

    reference_root = tmp_path / "durable-reference"
    reference_root.mkdir()
    context = operations.build_processing_activation_context(manifest)
    rows, _ = read_seed_csv_rows(seed_fixture["seed_csv"])
    for index, day in enumerate((day1, day2)):
        acquire_daily_bundle(
            output_root=reference_root, universe_csv=UNIVERSE_CSV, calendar_snapshot=CALENDAR_PATH,
            engine_day=day, opener=fake_opener_for(day, index),
            clock=lambda d=day: within_window_clock(d), monotonic_clock=lambda: 0.0, sleep_fn=lambda s: None,
        )
        processing.process_forward_day(
            study_root=reference_root, engine_day=day, universe_csv=UNIVERSE_CSV,
            calendar_snapshot=CALENDAR_PATH, seed_rows=rows, activation_context=context,
        )
    reference_latest = ForwardStudyStore(reference_root).load_latest_runtime()

    assert canonical_json_bytes(ops_latest["arm_a_runtime"]) == canonical_json_bytes(reference_latest["arm_a_runtime"])
    assert canonical_json_bytes(ops_latest["arm_b_runtime"]) == canonical_json_bytes(reference_latest["arm_b_runtime"])


# ---------------------------------------------------------------------------
# Safety invariants
# ---------------------------------------------------------------------------


def test_no_profit_metrics_in_result(committed_env, seed_fixture):
    result = run_day(committed_env, seed_fixture, BOUNDARY, 0)
    text = json.dumps(result, default=str).lower()
    for token in ("profit", "realized", "drawdown", "profit_factor", "win_rate", "pnl"):
        assert token not in text


def test_fresh_result_exposes_only_structural_counters(tmp_path, seed_fixture, monkeypatch):
    patch_candidates(monkeypatch)
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    manifest_path, manifest = build_manifest(seed_fixture, durable_root)
    result = run_day({"durable_root": durable_root, "manifest_path": manifest_path}, seed_fixture, BOUNDARY, 0)
    assert set(result["control"]) == {
        "open_position_count", "pending_order_count", "pending_proceeds_count",
        "closed_trade_count", "ledger_row_count", "daily_equity_row_count",
        "event_audit_row_count", "safety_counters", "skip_reason_counts", "parameters_sha256",
    }


def test_module_has_no_direct_urlopen_call():
    text = Path(operations.__file__).read_text(encoding="utf-8")
    assert "urlopen(" not in text


def test_module_has_no_network_imports():
    text = Path(operations.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines)


def test_module_never_writes_activation_manifest():
    text = Path(operations.__file__).read_text(encoding="utf-8")
    assert "write_activation_manifest_once" not in text
