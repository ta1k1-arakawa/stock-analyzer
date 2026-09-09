"""One-engine-day V7 forward processing orchestration.

This module is a thin, fail-closed orchestration layer that binds already
accepted V7 primitives together for exactly one JPX engine day:

    verified daily acquisition bundle
    + immutable validated seed
    + past COMPLETE forward D0 rows
        -> causal D0 frames
        -> candidate generation (once, shared by both arms)
        -> CONTROL / CAPACITY_3 execution
        -> append-only forward persistence

It owns no network, collector, activation, seed-acquisition, or formal
evaluation path, and it never computes or exposes profit, drawdown, profit
factor, win rate, or any arm performance comparison.  It does not create an
activation manifest; the activation context is supplied read-only by the
caller.  All existing primitives are reused, never reimplemented.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.v7_capacity_engine import (
    CausalEventEngine,
    V7EngineParameters,
    V7StudyBlocked,
    validate_single_parameter_difference,
)
from src.v7_daily_acquisition import (
    ACQUISITIONS_DIRNAME,
    CALENDAR_COMMIT,
    COLLECTOR_COMMIT,
    MANIFEST_FILENAME,
    MISSING_SNAPSHOT_FILENAME,
    PRICE_SNAPSHOT_FILENAME,
    SPLIT_SNAPSHOT_FILENAME,
    V7DailyAcquisitionBlocked,
    verify_daily_acquisition_bundle,
)
from src.v7_forward_candidate import V7CandidateBlocked, generate_forward_candidates_for_day
from src.v7_forward_persistence import (
    ForwardStudyStore,
    V7ForwardPersistenceBlocked,
    canonical_json_bytes,
    canonical_sha256,
    export_engine_runtime,
    restore_engine_runtime,
    sha256_bytes,
    verify_forward_store,
)
from src.v7_forward_protocol import ProtocolBlocked, validate_seed_rows
from src.v7_jpx_calendar import (
    V7JpxCalendarBlocked,
    generate_engine_days,
    is_jpx_trading_day,
    load_calendar_snapshot,
    next_jpx_trading_day,
)
from src.v7_seed_acquisition import V7SeedAcquisitionBlocked, validate_universe_file
from src.v7_seed_bridge import V7SeedBridgeBlocked, build_forward_frames_from_seed_and_d0


PRICE_SCHEMA_VERSION = "V7_FORWARD_PROCESSING_PRICE_V1"
CANDIDATE_SCHEMA_VERSION = "V7_FORWARD_PROCESSING_CANDIDATE_V1"
MARKET_GATE_SCHEMA_VERSION = "V7_FORWARD_PROCESSING_MARKET_GATE_V1"

EXPECTED_CALENDAR_COMMIT = CALENDAR_COMMIT
EXPECTED_COLLECTOR_COMMIT = COLLECTOR_COMMIT
CONTROL_PARAMETERS_SHA256 = "0ace638e6c40a222cd5b4ca107ddf6012c1f4e40e45dd45d49f37a1673b71b41"
CAPACITY_3_PARAMETERS_SHA256 = "d505d325d1c573595b9af26e141564f69a6ac8efdb8e6388d7eb61d50440a779"

FRAME_FIELD_BY_COLUMN = (
    ("Open", "Open"),
    ("High", "High"),
    ("Low", "Low"),
    ("Close", "Close"),
    ("Adj Close", "Adj Close"),
    ("Volume", "Volume"),
)

ACTIVATION_CONTEXT_FIELDS = (
    "activation_manifest_sha256",
    "activation_boundary_first_jpx_trading_date",
    "implementation_commit",
    "collector_commit",
    "expected_seed_canonical_sha256",
    "expected_seed_ticker_manifest_sha256",
)

PRICE_SNAPSHOT_FIELDS = (
    "schema_version",
    "engine_day",
    "implementation_commit",
    "acquisition_manifest_sha256",
    "acquisition_price_snapshot_sha256",
    "acquisition_missing_snapshot_sha256",
    "acquisition_split_snapshot_sha256",
    "previous_complete_engine_day",
    "d0_price_rows",
    "d0_missing_rows",
    "d0_split_events",
)

CANDIDATE_SNAPSHOT_FIELDS = (
    "schema_version",
    "engine_day",
    "implementation_commit",
    "source_processing_price_snapshot_sha256",
    "candidate_input_frame_sha256",
    "candidate_snapshot_sha256",
    "accepted_top20",
    "full_candidate_audit",
    "future_candidate_data_access_count",
    "future_split_access_count",
    "entry_attempt_date",
    "planned_exit_date",
)

MARKET_GATE_SNAPSHOT_FIELDS = (
    "schema_version",
    "engine_day",
    "implementation_commit",
    "market_gate",
    "market_gate_snapshot_sha256",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class V7ForwardDayProcessingBlocked(RuntimeError):
    """Fail-closed forward day orchestration boundary violation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _blocked(reason: str) -> V7ForwardDayProcessingBlocked:
    return V7ForwardDayProcessingBlocked(reason)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(COMMIT_RE.fullmatch(value))


def _parse_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise _blocked("INVALID_DATE:" + field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise _blocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise _blocked("INVALID_DATE:" + field)
    return parsed


def _json_safe(value: Any) -> Any:
    """Idempotent JSON normalization mirroring the accepted candidate-audit encoding.

    ``generate_forward_candidates_for_day`` hashes its own output through that
    encoding but returns the market-gate audit row with raw pandas timestamps.
    Rows that are already normalized pass through unchanged, so applying this
    before persistence never alters an accepted canonical hash.
    """
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item") and hasattr(value, "dtype"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _blocked("BUNDLE_FILE_READ_FAILED:" + label) from error


def _file_sha256(path: Path, label: str) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise _blocked("BUNDLE_FILE_READ_FAILED:" + label) from error


# ---------------------------------------------------------------------------
# Activation context (read-only; never created here)
# ---------------------------------------------------------------------------


def validate_activation_context(
    activation_context: Mapping[str, Any], calendar_snapshot_obj: Any
) -> dict[str, Any]:
    if not isinstance(activation_context, Mapping):
        raise _blocked("ACTIVATION_CONTEXT_INVALID")
    if set(activation_context) != set(ACTIVATION_CONTEXT_FIELDS):
        raise _blocked("ACTIVATION_CONTEXT_SCHEMA_INVALID")
    if not _valid_sha256(activation_context["activation_manifest_sha256"]):
        raise _blocked("ACTIVATION_MANIFEST_SHA_INVALID")
    if not _valid_commit(activation_context["implementation_commit"]):
        raise _blocked("IMPLEMENTATION_COMMIT_INVALID")
    if not _valid_commit(activation_context["collector_commit"]):
        raise _blocked("COLLECTOR_COMMIT_INVALID")
    if activation_context["collector_commit"] != EXPECTED_COLLECTOR_COMMIT:
        raise _blocked("COLLECTOR_COMMIT_MISMATCH")
    for field in ("expected_seed_canonical_sha256", "expected_seed_ticker_manifest_sha256"):
        if not _valid_sha256(activation_context[field]):
            raise _blocked("SEED_EXPECTED_SHA_INVALID:" + field)
    boundary = activation_context["activation_boundary_first_jpx_trading_date"]
    _parse_iso_date(boundary, "activation_boundary_first_jpx_trading_date")
    try:
        trading = is_jpx_trading_day(calendar_snapshot_obj, boundary)
    except V7JpxCalendarBlocked as error:
        raise _blocked("ACTIVATION_BOUNDARY_OUTSIDE_CALENDAR_COVERAGE") from error
    if not trading:
        raise _blocked("ACTIVATION_BOUNDARY_NOT_JPX_TRADING_DAY")
    return dict(activation_context)


# ---------------------------------------------------------------------------
# Engine-day sequencing (strict consecutive JPX trading days)
# ---------------------------------------------------------------------------


def resolve_engine_day_sequence(
    store: ForwardStudyStore,
    calendar_snapshot_obj: Any,
    engine_day: str,
    activation_boundary: str,
) -> str | None:
    persisted_days = store._final_days()
    if any(day >= engine_day for day in persisted_days):
        raise _blocked("ENGINE_DAY_NOT_FORWARD_OF_PERSISTED_STORE")
    checkpoint = store.load_latest_checkpoint()
    if checkpoint is None:
        if engine_day != activation_boundary:
            raise _blocked("FIRST_ENGINE_DAY_NOT_ACTIVATION_BOUNDARY")
        return None
    previous_day = checkpoint["last_completed_engine_day"]
    try:
        expected = next_jpx_trading_day(calendar_snapshot_obj, previous_day)
    except V7JpxCalendarBlocked as error:
        raise _blocked("PREVIOUS_ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE") from error
    if engine_day != expected:
        raise _blocked("ENGINE_DAY_NOT_NEXT_JPX_TRADING_DAY")
    return previous_day


# ---------------------------------------------------------------------------
# Acquisition bundle reads (only after verification)
# ---------------------------------------------------------------------------


def _acquisition_day_dir(study_root: Path, engine_day: str) -> Path:
    return study_root / ACQUISITIONS_DIRNAME / engine_day


def read_verified_acquisition(study_root: Path, engine_day: str, universe_csv: Any) -> dict[str, Any]:
    try:
        verify_daily_acquisition_bundle(
            study_root,
            engine_day,
            EXPECTED_CALENDAR_COMMIT,
            EXPECTED_COLLECTOR_COMMIT,
            universe_csv,
        )
    except V7DailyAcquisitionBlocked as error:
        raise _blocked("ACQUISITION_VERIFICATION_FAILED:" + error.reason) from error
    day_dir = _acquisition_day_dir(study_root, engine_day)
    return {
        "acquisition_manifest_sha256": _file_sha256(day_dir / MANIFEST_FILENAME, MANIFEST_FILENAME),
        "acquisition_price_snapshot_sha256": _file_sha256(day_dir / PRICE_SNAPSHOT_FILENAME, PRICE_SNAPSHOT_FILENAME),
        "acquisition_missing_snapshot_sha256": _file_sha256(day_dir / MISSING_SNAPSHOT_FILENAME, MISSING_SNAPSHOT_FILENAME),
        "acquisition_split_snapshot_sha256": _file_sha256(day_dir / SPLIT_SNAPSHOT_FILENAME, SPLIT_SNAPSHOT_FILENAME),
        "price_rows": _read_json(day_dir / PRICE_SNAPSHOT_FILENAME, PRICE_SNAPSHOT_FILENAME),
        "missing_rows": _read_json(day_dir / MISSING_SNAPSHOT_FILENAME, MISSING_SNAPSHOT_FILENAME),
        "split_events": _read_json(day_dir / SPLIT_SNAPSHOT_FILENAME, SPLIT_SNAPSHOT_FILENAME),
    }


# ---------------------------------------------------------------------------
# Past COMPLETE forward-day history (never reads current or future days)
# ---------------------------------------------------------------------------


def _validate_processing_price_snapshot(payload: Any, day: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != set(PRICE_SNAPSHOT_FIELDS):
        raise _blocked("PROCESSING_PRICE_SNAPSHOT_SCHEMA_INVALID:" + day)
    if payload["schema_version"] != PRICE_SCHEMA_VERSION:
        raise _blocked("PROCESSING_PRICE_SCHEMA_VERSION_MISMATCH:" + day)
    if payload["engine_day"] != day:
        raise _blocked("PROCESSING_PRICE_ENGINE_DAY_MISMATCH:" + day)
    return dict(payload)


def read_past_forward_history(store: ForwardStudyStore, engine_day: str) -> dict[str, Any]:
    """Read D0 price rows and split events from strictly earlier COMPLETE days only."""
    history_rows: list[dict[str, Any]] = []
    split_events: list[dict[str, Any]] = []
    for day in store._final_days():
        if day >= engine_day:
            raise _blocked("ENGINE_DAY_NOT_FORWARD_OF_PERSISTED_STORE")
        payload = _validate_processing_price_snapshot(
            _read_json(store.days_root / day / "price_snapshot.json", "price_snapshot.json"), day
        )
        for row in payload["d0_price_rows"]:
            if row["trading_date"] != day:
                raise _blocked("PAST_D0_ROW_DATE_MISMATCH:" + day)
            history_rows.append({key: row[key] for key in row if key != "payload_sha256"})
        for event in payload["d0_split_events"]:
            if event["effective_date"] != day:
                raise _blocked("PAST_SPLIT_EVENT_DATE_MISMATCH:" + day)
            split_events.append(dict(event))
    return {"history_rows": history_rows, "split_events": split_events}


# ---------------------------------------------------------------------------
# Causal split provenance
# ---------------------------------------------------------------------------


def build_split_history(
    past_split_events: Sequence[Mapping[str, Any]],
    current_split_events: Sequence[Mapping[str, Any]],
    engine_day: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (split_history_by_ticker, split_events_by_day) bound to <= engine_day."""
    by_ticker: dict[str, set[str]] = {}
    by_day: dict[str, set[str]] = {}
    for event in list(past_split_events) + list(current_split_events):
        ticker = str(event["ticker"]).strip().upper()
        effective_date = str(event["effective_date"])
        _parse_iso_date(effective_date, "effective_date")
        if effective_date > engine_day:
            raise _blocked("FUTURE_SPLIT_ACCESS")
        by_ticker.setdefault(ticker, set()).add(effective_date)
        by_day.setdefault(effective_date, set()).add(ticker)
    return (
        {ticker: sorted(dates) for ticker, dates in sorted(by_ticker.items())},
        {day: sorted(tickers) for day, tickers in sorted(by_day.items())},
    )


# ---------------------------------------------------------------------------
# Engine frames
# ---------------------------------------------------------------------------


def build_engine_frames(
    frames: Mapping[str, pd.DataFrame], engine_day: str
) -> dict[str, dict[str, dict[str, float]]]:
    boundary = pd.Timestamp(engine_day)
    engine_frames: dict[str, dict[str, dict[str, float]]] = {}
    for ticker in sorted(frames):
        frame = frames[ticker]
        rows: dict[str, dict[str, float]] = {}
        for timestamp, row in frame.iterrows():
            if timestamp > boundary:
                raise _blocked("ENGINE_FRAME_FUTURE_DATE:" + str(ticker))
            rows[pd.Timestamp(timestamp).strftime("%Y-%m-%d")] = {
                field: float(row[column]) for column, field in FRAME_FIELD_BY_COLUMN
            }
        engine_frames[str(ticker)] = rows
    return engine_frames


def build_engine_candidates(
    accepted_top20: Sequence[Mapping[str, Any]], engine_day: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in accepted_top20:
        if str(row["candidate_status"]) != "ACCEPTED_TOP20":
            raise _blocked("CANDIDATE_STATUS_NOT_ACCEPTED_TOP20")
        if str(row["signal_date"]) != engine_day:
            raise _blocked("CANDIDATE_SIGNAL_DATE_MISMATCH")
        candidates.append({
            "signal_year": int(row["signal_year"]),
            "signal_date": str(row["signal_date"]),
            "ticker": str(row["ticker"]),
            "industry": str(row["industry"]),
            "rank": int(row["rank"]),
            "signal_raw_close": float(row["raw_close"]),
            "entry_attempt_date": str(row["entry_date"]),
            "planned_exit_date": str(row["exit_date"]),
            "candidate_status": "ACCEPTED_TOP20",
        })
    return candidates


# ---------------------------------------------------------------------------
# Dual-arm construction and restart
# ---------------------------------------------------------------------------


def _build_arm(
    engine_frames: Mapping[str, Any],
    study_calendar: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    parameters: V7EngineParameters,
    split_events_by_day: Mapping[str, Sequence[str]],
) -> CausalEventEngine:
    return CausalEventEngine(
        engine_frames,
        tuple(study_calendar),
        [dict(row) for row in candidates],
        parameters,
        split_events_by_day={day: list(items) for day, items in split_events_by_day.items()},
    )


def build_dual_arms(
    engine_frames: Mapping[str, Any],
    study_calendar: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    split_events_by_day: Mapping[str, Sequence[str]],
) -> tuple[CausalEventEngine, CausalEventEngine]:
    control_parameters = V7EngineParameters.control()
    variant_parameters = V7EngineParameters.capacity_3()
    validate_single_parameter_difference(control_parameters, variant_parameters)
    if control_parameters.sha256() != CONTROL_PARAMETERS_SHA256:
        raise _blocked("CONTROL_PARAMETERS_SHA_MISMATCH")
    if variant_parameters.sha256() != CAPACITY_3_PARAMETERS_SHA256:
        raise _blocked("CAPACITY_3_PARAMETERS_SHA_MISMATCH")
    control = _build_arm(engine_frames, study_calendar, candidates, control_parameters, split_events_by_day)
    variant = _build_arm(engine_frames, study_calendar, candidates, variant_parameters, split_events_by_day)
    if control.state is variant.state:
        raise _blocked("ARM_STATE_NOT_INDEPENDENT")
    return control, variant


def restore_previous_runtimes(
    store: ForwardStudyStore,
    control: CausalEventEngine,
    variant: CausalEventEngine,
    previous_day: str,
) -> None:
    latest = store.load_latest_runtime()
    if latest is None or latest["day"] != previous_day:
        raise _blocked("PREVIOUS_RUNTIME_NOT_AVAILABLE")
    try:
        restore_engine_runtime(control, latest["arm_a_runtime"])
        restore_engine_runtime(variant, latest["arm_b_runtime"])
    except V7ForwardPersistenceBlocked as error:
        raise _blocked("RUNTIME_RESTORE_FAILED:" + error.reason) from error
    for engine, label in ((control, "arm_a"), (variant, "arm_b")):
        if engine.state.engine_day != previous_day:
            raise _blocked("RESTORED_RUNTIME_ENGINE_DAY_MISMATCH:" + label)


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def build_processing_price_snapshot(
    *,
    engine_day: str,
    implementation_commit: str,
    acquisition: Mapping[str, Any],
    previous_complete_engine_day: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": PRICE_SCHEMA_VERSION,
        "engine_day": engine_day,
        "implementation_commit": implementation_commit,
        "acquisition_manifest_sha256": acquisition["acquisition_manifest_sha256"],
        "acquisition_price_snapshot_sha256": acquisition["acquisition_price_snapshot_sha256"],
        "acquisition_missing_snapshot_sha256": acquisition["acquisition_missing_snapshot_sha256"],
        "acquisition_split_snapshot_sha256": acquisition["acquisition_split_snapshot_sha256"],
        "previous_complete_engine_day": previous_complete_engine_day,
        "d0_price_rows": [dict(row) for row in acquisition["price_rows"]],
        "d0_missing_rows": [dict(row) for row in acquisition["missing_rows"]],
        "d0_split_events": [dict(event) for event in acquisition["split_events"]],
    }


def build_candidate_snapshot(
    *,
    engine_day: str,
    implementation_commit: str,
    processing_price_snapshot_sha256: str,
    candidate_result: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = _json_safe(list(candidate_result["accepted_top20"]))
    if canonical_sha256(accepted) != candidate_result["candidate_snapshot_sha256"]:
        raise _blocked("CANDIDATE_SNAPSHOT_HASH_MISMATCH")
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "engine_day": engine_day,
        "implementation_commit": implementation_commit,
        "source_processing_price_snapshot_sha256": processing_price_snapshot_sha256,
        "candidate_input_frame_sha256": candidate_result["price_snapshot_sha256"],
        "candidate_snapshot_sha256": candidate_result["candidate_snapshot_sha256"],
        "accepted_top20": accepted,
        "full_candidate_audit": _json_safe(list(candidate_result["full_candidate_audit"])),
        "future_candidate_data_access_count": int(candidate_result["future_candidate_data_access_count"]),
        "future_split_access_count": int(candidate_result["future_split_access_count"]),
        "entry_attempt_date": candidate_result["entry_attempt_date"],
        "planned_exit_date": candidate_result["planned_exit_date"],
    }


def build_market_gate_snapshot(
    *, engine_day: str, implementation_commit: str, candidate_result: Mapping[str, Any]
) -> dict[str, Any]:
    market_gate = _json_safe(dict(candidate_result["market_gate"]))
    if canonical_sha256(market_gate) != candidate_result["market_gate_snapshot_sha256"]:
        raise _blocked("MARKET_GATE_SNAPSHOT_HASH_MISMATCH")
    return {
        "schema_version": MARKET_GATE_SCHEMA_VERSION,
        "engine_day": engine_day,
        "implementation_commit": implementation_commit,
        "market_gate": market_gate,
        "market_gate_snapshot_sha256": candidate_result["market_gate_snapshot_sha256"],
    }


# ---------------------------------------------------------------------------
# Public one-day orchestration
# ---------------------------------------------------------------------------


def process_forward_day(
    *,
    study_root: str | Path,
    engine_day: str,
    universe_csv: str | Path,
    calendar_snapshot: Mapping[str, Any] | str | Path,
    seed_rows: Sequence[Mapping[str, Any]],
    activation_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Process exactly one JPX engine day and append-only persist both arms."""
    _parse_iso_date(engine_day, "engine_day")
    root = Path(study_root)

    try:
        calendar = load_calendar_snapshot(calendar_snapshot)
    except V7JpxCalendarBlocked as error:
        raise _blocked("CALENDAR_SNAPSHOT_INVALID") from error
    context = validate_activation_context(activation_context, calendar)
    implementation_commit = context["implementation_commit"]
    activation_boundary = context["activation_boundary_first_jpx_trading_date"]

    try:
        if not is_jpx_trading_day(calendar, engine_day):
            raise _blocked("ENGINE_DAY_NOT_JPX_TRADING_DAY")
    except V7JpxCalendarBlocked as error:
        raise _blocked("ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE") from error
    if engine_day < activation_boundary:
        raise _blocked("ENGINE_DAY_BEFORE_ACTIVATION_BOUNDARY")

    try:
        universe = validate_universe_file(universe_csv)
    except V7SeedAcquisitionBlocked as error:
        raise _blocked("UNIVERSE_VALIDATION_FAILED:" + error.reason) from error

    try:
        seed_validation = validate_seed_rows(
            seed_rows,
            universe["tickers"],
            activation_boundary,
            expected_seed_canonical_sha256=context["expected_seed_canonical_sha256"],
        )
    except ProtocolBlocked as error:
        raise _blocked("SEED_VALIDATION_FAILED:" + str(error)) from error
    if seed_validation["seed_payload_manifest_sha256"] != context["expected_seed_ticker_manifest_sha256"]:
        raise _blocked("SEED_TICKER_MANIFEST_HASH_MISMATCH")

    store = ForwardStudyStore(root)
    previous_complete_engine_day = resolve_engine_day_sequence(
        store, calendar, engine_day, activation_boundary
    )

    acquisition = read_verified_acquisition(root, engine_day, universe_csv)
    history = read_past_forward_history(store, engine_day)

    split_history, split_events_by_day = build_split_history(
        history["split_events"], acquisition["split_events"], engine_day
    )

    history_rows = list(seed_validation["canonical_rows"]) + history["history_rows"]
    current_d0_rows = [
        {key: row[key] for key in row if key != "payload_sha256"}
        for row in acquisition["price_rows"]
    ]
    try:
        frames = build_forward_frames_from_seed_and_d0(history_rows, current_d0_rows, engine_day)
    except V7SeedBridgeBlocked as error:
        raise _blocked("FORWARD_FRAME_BUILD_FAILED:" + str(error)) from error

    study_calendar = generate_engine_days(
        calendar, activation_boundary, calendar.coverage_end.isoformat()
    )
    if engine_day not in study_calendar:
        raise _blocked("ENGINE_DAY_NOT_IN_STUDY_CALENDAR")

    try:
        candidate_result = generate_forward_candidates_for_day(
            frames,
            pd.DataFrame({
                "ticker": universe["tickers"],
                "market": ["JP"] * len(universe["tickers"]),
                "industry": [universe["industries"][ticker] for ticker in universe["tickers"]],
            }),
            split_history,
            study_calendar,
            engine_day,
            context["collector_commit"],
        )
    except V7CandidateBlocked as error:
        raise _blocked("CANDIDATE_GENERATION_FAILED:" + error.reason) from error
    if int(candidate_result["future_candidate_data_access_count"]) != 0:
        raise _blocked("FUTURE_CANDIDATE_DATA_ACCESS")
    if int(candidate_result["future_split_access_count"]) != 0:
        raise _blocked("FUTURE_SPLIT_ACCESS")

    engine_frames = build_engine_frames(frames, engine_day)
    engine_candidates = build_engine_candidates(candidate_result["accepted_top20"], engine_day)

    try:
        control, variant = build_dual_arms(
            engine_frames, study_calendar, engine_candidates, split_events_by_day
        )
    except ValueError as error:
        raise _blocked("ARM_CONSTRUCTION_FAILED:" + str(error)) from error

    if previous_complete_engine_day is not None:
        restore_previous_runtimes(store, control, variant, previous_complete_engine_day)

    for engine, label in ((control, "CONTROL"), (variant, "CAPACITY_3")):
        try:
            engine.process_day(engine_day)
        except (V7StudyBlocked, ValueError, AssertionError) as error:
            reason = getattr(error, "reason", None) or str(error)
            raise _blocked("ARM_PROCESSING_FAILED:" + label + ":" + reason) from error

    price_snapshot = build_processing_price_snapshot(
        engine_day=engine_day,
        implementation_commit=implementation_commit,
        acquisition=acquisition,
        previous_complete_engine_day=previous_complete_engine_day,
    )
    candidate_snapshot = build_candidate_snapshot(
        engine_day=engine_day,
        implementation_commit=implementation_commit,
        processing_price_snapshot_sha256=canonical_sha256(price_snapshot),
        candidate_result=candidate_result,
    )
    market_gate_snapshot = build_market_gate_snapshot(
        engine_day=engine_day,
        implementation_commit=implementation_commit,
        candidate_result=candidate_result,
    )

    try:
        checkpoint = store.write_day(
            engine_day,
            price_snapshot=price_snapshot,
            candidate_snapshot=candidate_snapshot,
            market_gate_snapshot=market_gate_snapshot,
            arm_a_runtime=export_engine_runtime(control),
            arm_b_runtime=export_engine_runtime(variant),
            activation_manifest_sha256=context["activation_manifest_sha256"],
            collector_commit=context["collector_commit"],
        )
    except V7ForwardPersistenceBlocked as error:
        raise _blocked("FORWARD_DAY_PERSISTENCE_FAILED:" + error.reason) from error

    return _processing_summary(
        engine_day=engine_day,
        previous_complete_engine_day=previous_complete_engine_day,
        checkpoint=checkpoint,
        acquisition=acquisition,
        candidate_result=candidate_result,
        control=control,
        variant=variant,
    )


def _arm_counts(engine: CausalEventEngine) -> dict[str, Any]:
    """Structural counters only.  No profit, equity value, or performance metric."""
    return {
        "open_position_count": len(engine.state.open_positions),
        "pending_order_count": sum(
            len(items) for items in engine.state.pending_orders_by_entry_date.values()
        ),
        "pending_proceeds_count": sum(
            len(items) for items in engine.state.pending_proceeds_by_available_date.values()
        ),
        "closed_trade_count": sum(
            1 for row in engine.state.completed_trades if row["status"] == "CLOSED"
        ),
        "ledger_row_count": len(engine.state.completed_trades),
        "daily_equity_row_count": len(engine.state.daily_equity),
        "event_audit_row_count": len(engine.state.event_audit),
        "safety_counters": engine.safety_counters(),
        "skip_reason_counts": engine.skip_reason_counts(),
        "parameters_sha256": engine.parameters.sha256(),
    }


def _processing_summary(
    *,
    engine_day: str,
    previous_complete_engine_day: str | None,
    checkpoint: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    control: CausalEventEngine,
    variant: CausalEventEngine,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "engine_day": engine_day,
        "previous_complete_engine_day": previous_complete_engine_day,
        "acquisition_verified": True,
        "seed_verified": True,
        "valid_d0_count": len(acquisition["price_rows"]),
        "missing_d0_count": len(acquisition["missing_rows"]),
        "split_event_count": len(acquisition["split_events"]),
        "accepted_candidate_count": len(candidate_result["accepted_top20"]),
        "market_gate_status": candidate_result["market_gate"]["market_gate_status"],
        "future_candidate_data_access_count": int(candidate_result["future_candidate_data_access_count"]),
        "future_split_access_count": int(candidate_result["future_split_access_count"]),
        "entry_attempt_date": candidate_result["entry_attempt_date"],
        "planned_exit_date": candidate_result["planned_exit_date"],
        "control": _arm_counts(control),
        "variant": _arm_counts(variant),
        "checkpoint_sha256": checkpoint["current_checkpoint_sha256"],
        "forward_day_persisted": True,
        "network_requests": 0,
        "activation_created": False,
        "profit_metrics_exposed": False,
    }


# ---------------------------------------------------------------------------
# Read-only processed-day verifier
# ---------------------------------------------------------------------------


def verify_processed_forward_day(
    *,
    study_root: str | Path,
    engine_day: str,
    universe_csv: str | Path,
    activation_context: Mapping[str, Any],
) -> dict[str, Any]:
    _parse_iso_date(engine_day, "engine_day")
    root = Path(study_root)
    if not isinstance(activation_context, Mapping) or set(activation_context) != set(ACTIVATION_CONTEXT_FIELDS):
        raise _blocked("ACTIVATION_CONTEXT_SCHEMA_INVALID")
    implementation_commit = activation_context["implementation_commit"]
    activation_manifest_sha256 = activation_context["activation_manifest_sha256"]
    collector_commit = activation_context["collector_commit"]

    try:
        verify_daily_acquisition_bundle(
            root, engine_day, EXPECTED_CALENDAR_COMMIT, EXPECTED_COLLECTOR_COMMIT, universe_csv
        )
    except V7DailyAcquisitionBlocked as error:
        raise _blocked("ACQUISITION_VERIFICATION_FAILED:" + error.reason) from error

    try:
        store_result = verify_forward_store(root, activation_manifest_sha256, collector_commit)
    except V7ForwardPersistenceBlocked as error:
        raise _blocked("FORWARD_STORE_VERIFICATION_FAILED:" + error.reason) from error
    if not store_result["verified_days"] or store_result["verified_days"][-1] != engine_day:
        raise _blocked("LATEST_COMPLETE_DAY_MISMATCH")

    store = ForwardStudyStore(root)
    day_dir = store.days_root / engine_day
    price_snapshot = _validate_processing_price_snapshot(
        _read_json(day_dir / "price_snapshot.json", "price_snapshot.json"), engine_day
    )
    candidate_snapshot = _read_json(day_dir / "candidate_snapshot.json", "candidate_snapshot.json")
    market_gate_snapshot = _read_json(day_dir / "market_gate_snapshot.json", "market_gate_snapshot.json")

    if set(candidate_snapshot) != set(CANDIDATE_SNAPSHOT_FIELDS):
        raise _blocked("PROCESSING_CANDIDATE_SNAPSHOT_SCHEMA_INVALID")
    if candidate_snapshot["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise _blocked("PROCESSING_CANDIDATE_SCHEMA_VERSION_MISMATCH")
    if set(market_gate_snapshot) != set(MARKET_GATE_SNAPSHOT_FIELDS):
        raise _blocked("PROCESSING_MARKET_GATE_SNAPSHOT_SCHEMA_INVALID")
    if market_gate_snapshot["schema_version"] != MARKET_GATE_SCHEMA_VERSION:
        raise _blocked("PROCESSING_MARKET_GATE_SCHEMA_VERSION_MISMATCH")

    for snapshot, label in (
        (price_snapshot, "price"),
        (candidate_snapshot, "candidate"),
        (market_gate_snapshot, "market_gate"),
    ):
        if snapshot["engine_day"] != engine_day:
            raise _blocked("PROCESSING_SNAPSHOT_ENGINE_DAY_MISMATCH:" + label)
        if snapshot["implementation_commit"] != implementation_commit:
            raise _blocked("IMPLEMENTATION_COMMIT_MISMATCH:" + label)

    acquisition_day_dir = _acquisition_day_dir(root, engine_day)
    for filename, field in (
        (MANIFEST_FILENAME, "acquisition_manifest_sha256"),
        (PRICE_SNAPSHOT_FILENAME, "acquisition_price_snapshot_sha256"),
        (MISSING_SNAPSHOT_FILENAME, "acquisition_missing_snapshot_sha256"),
        (SPLIT_SNAPSHOT_FILENAME, "acquisition_split_snapshot_sha256"),
    ):
        if _file_sha256(acquisition_day_dir / filename, filename) != price_snapshot[field]:
            raise _blocked("ACQUISITION_SNAPSHOT_HASH_MISMATCH:" + field)

    for filename, key, label in (
        (PRICE_SNAPSHOT_FILENAME, "d0_price_rows", "price"),
        (MISSING_SNAPSHOT_FILENAME, "d0_missing_rows", "missing"),
        (SPLIT_SNAPSHOT_FILENAME, "d0_split_events", "split"),
    ):
        actual = _read_json(acquisition_day_dir / filename, filename)
        if canonical_json_bytes(actual) != canonical_json_bytes(price_snapshot[key]):
            raise _blocked("ACQUISITION_D0_CONTENT_PARITY_MISMATCH:" + label)

    if canonical_sha256(candidate_snapshot["accepted_top20"]) != candidate_snapshot["candidate_snapshot_sha256"]:
        raise _blocked("CANDIDATE_SNAPSHOT_HASH_MISMATCH")
    if canonical_sha256(market_gate_snapshot["market_gate"]) != market_gate_snapshot["market_gate_snapshot_sha256"]:
        raise _blocked("MARKET_GATE_SNAPSHOT_HASH_MISMATCH")
    if canonical_sha256(price_snapshot) != candidate_snapshot["source_processing_price_snapshot_sha256"]:
        raise _blocked("PROCESSING_PRICE_PROVENANCE_HASH_MISMATCH")
    if candidate_snapshot["future_candidate_data_access_count"] != 0:
        raise _blocked("FUTURE_CANDIDATE_DATA_ACCESS")
    if candidate_snapshot["future_split_access_count"] != 0:
        raise _blocked("FUTURE_SPLIT_ACCESS")

    latest = store.load_latest_runtime()
    if latest is None or latest["day"] != engine_day:
        raise _blocked("LATEST_RUNTIME_DAY_MISMATCH")
    if latest["arm_a_runtime"]["parameters_sha256"] != CONTROL_PARAMETERS_SHA256:
        raise _blocked("ARM_A_PARAMETERS_SHA_MISMATCH")
    if latest["arm_b_runtime"]["parameters_sha256"] != CAPACITY_3_PARAMETERS_SHA256:
        raise _blocked("ARM_B_PARAMETERS_SHA_MISMATCH")
    for label, runtime in (("arm_a", latest["arm_a_runtime"]), ("arm_b", latest["arm_b_runtime"])):
        if runtime["engine_day"] != engine_day:
            raise _blocked("RUNTIME_ENGINE_DAY_MISMATCH:" + label)

    return {
        "status": "PASS",
        "engine_day": engine_day,
        "verified_day_count": store_result["day_count"],
        "accepted_candidate_count": len(candidate_snapshot["accepted_top20"]),
        "valid_d0_count": len(price_snapshot["d0_price_rows"]),
        "missing_d0_count": len(price_snapshot["d0_missing_rows"]),
        "split_event_count": len(price_snapshot["d0_split_events"]),
    }


__all__ = [
    "ACTIVATION_CONTEXT_FIELDS",
    "CANDIDATE_SCHEMA_VERSION",
    "CANDIDATE_SNAPSHOT_FIELDS",
    "CAPACITY_3_PARAMETERS_SHA256",
    "CONTROL_PARAMETERS_SHA256",
    "EXPECTED_CALENDAR_COMMIT",
    "EXPECTED_COLLECTOR_COMMIT",
    "MARKET_GATE_SCHEMA_VERSION",
    "MARKET_GATE_SNAPSHOT_FIELDS",
    "PRICE_SCHEMA_VERSION",
    "PRICE_SNAPSHOT_FIELDS",
    "V7ForwardDayProcessingBlocked",
    "build_candidate_snapshot",
    "build_dual_arms",
    "build_engine_candidates",
    "build_engine_frames",
    "build_market_gate_snapshot",
    "build_processing_price_snapshot",
    "build_split_history",
    "process_forward_day",
    "read_past_forward_history",
    "read_verified_acquisition",
    "resolve_engine_day_sequence",
    "restore_previous_runtimes",
    "validate_activation_context",
    "verify_processed_forward_day",
]
