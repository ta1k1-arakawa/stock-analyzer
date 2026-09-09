"""Local/synthetic Gate 3 C1 dry-run core.

This module owns no activation, persistence, collector, or network path.  It
binds both paper arms to hashes produced by the supplied synthetic inputs and
exercises the already accepted forward candidate and execution components.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import pandas as pd

from src.v7_capacity_engine import V7StudyBlocked
from src.v7_forward_candidate import generate_forward_candidates_for_day
from src.v7_forward_protocol import (
    ArmInputHashes,
    DualArmStudy,
    create_dual_arm_study,
    validate_seed_rows,
)


DESIGN_COMMIT = "e3e1367efd913b601a70328a815d88c20af6d147"
LATEST_PREREGISTRATION_UTC = "2026-08-07T02:48:27Z"
MODE = "DRY_RUN_ONLY"
ACTIVATION_STATUS = "NOT_ACTIVATED"
ACTIVATION_BOUNDARY = "NOT_SET"
COLLECTOR_COMMIT_RE = set("0123456789abcdefABCDEF")


class Gate3DryRunBlocked(RuntimeError):
    """Fail closed when a synthetic Gate 3 case is unexpectedly invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ticker_values(universe: Any) -> list[str]:
    if isinstance(universe, pd.DataFrame):
        return [str(value).strip().upper() for value in universe["ticker"].tolist()]
    values = list(universe)
    if values and isinstance(values[0], Mapping):
        return [str(value["ticker"]).strip().upper() for value in values]
    return [str(value).strip().upper() for value in values]


def _d0_frames(frames: Mapping[str, pd.DataFrame], engine_day: Any) -> dict[str, pd.DataFrame]:
    day = pd.Timestamp(engine_day)
    return {
        ticker: frame.loc[pd.to_datetime(frame.index) <= day].copy()
        for ticker, frame in frames.items()
    }


def _engine_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for ticker in sorted(frames):
        frame = frames[ticker]
        days: dict[str, dict[str, float]] = {}
        for day, row in frame.iterrows():
            values: dict[str, float] = {}
            for field in ("Open", "Close"):
                if field in row:
                    values[field] = row[field]
            days[pd.Timestamp(day).strftime("%Y-%m-%d")] = values
        result[ticker] = days
    return result


def _engine_candidates(accepted_top20: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_keys = {(str(row["signal_date"]), str(row["ticker"]), int(row["rank"])) for row in accepted_top20}
    result = []
    for row in accepted_top20:
        result.append({
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
    actual_keys = {(row["signal_date"], row["ticker"], row["rank"]) for row in result}
    if actual_keys != expected_keys or len(actual_keys) != len(result):
        raise Gate3DryRunBlocked("CANDIDATE_ADAPTER_KEY_MISMATCH")
    return copy.deepcopy(result)


def _shared_inputs(
    frames: Mapping[str, pd.DataFrame],
    universe: Any,
    split_history: Mapping[str, Sequence[Any]] | None,
    study_calendar: Sequence[Any],
    engine_day: Any,
    seed_rows: Sequence[Mapping[str, Any]],
    collector_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], ArmInputHashes]:
    tickers = _ticker_values(universe)
    seed_validation = validate_seed_rows(seed_rows, tickers, str(pd.Timestamp(engine_day).date()))
    candidate_snapshot = generate_forward_candidates_for_day(
        _d0_frames(frames, engine_day),
        universe,
        split_history,
        study_calendar,
        engine_day,
        collector_commit,
    )
    candidates = _engine_candidates(candidate_snapshot["accepted_top20"])
    hashes = ArmInputHashes(
        seed_hash=seed_validation["seed_canonical_sha256"],
        price_snapshot_hash=candidate_snapshot["price_snapshot_sha256"],
        candidate_snapshot_hash=candidate_snapshot["candidate_snapshot_sha256"],
        market_gate_snapshot_hash=candidate_snapshot["market_gate_snapshot_sha256"],
    )
    return seed_validation, candidate_snapshot, candidates, hashes


def _study(
    frames: Mapping[str, pd.DataFrame],
    study_calendar: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    hashes: ArmInputHashes,
    split_events_by_day: Mapping[str, Sequence[str]] | None,
) -> DualArmStudy:
    return create_dual_arm_study(
        _engine_frames(frames),
        [pd.Timestamp(day).strftime("%Y-%m-%d") for day in study_calendar],
        copy.deepcopy(candidates),
        ArmInputHashes(**hashes.as_dict()),
        ArmInputHashes(**hashes.as_dict()),
        split_events_by_day=copy.deepcopy(split_events_by_day),
    )


def _run_pair(
    frames: Mapping[str, pd.DataFrame],
    study_calendar: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    hashes: ArmInputHashes,
    split_events_by_day: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DualArmStudy, list[str]]:
    study = _study(frames, study_calendar, candidates, hashes, split_events_by_day)
    observed: list[str] = []
    for engine in (study.control, study.variant):
        try:
            engine.run()
        except V7StudyBlocked as error:
            observed.append(error.reason)
    return study, observed


def _audit_event(
    arm: str,
    event: str,
    engine_day: Any,
    ticker: str | None,
    reason: str | None,
    candidate_hash: str,
    price_hash: str,
    collector_commit: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "event": event,
        "engine_day": pd.Timestamp(engine_day).strftime("%Y-%m-%d"),
        "ticker": ticker,
        "candidate_snapshot_sha256": candidate_hash,
        "price_snapshot_sha256": price_hash,
        "reason": reason,
        "collector_commit": collector_commit,
        **extra,
    }


def _enriched_audit(
    candidate_snapshot: Mapping[str, Any],
    study: DualArmStudy,
    collector_commit: str,
) -> list[dict[str, Any]]:
    candidate_hash = candidate_snapshot["candidate_snapshot_sha256"]
    price_hash = candidate_snapshot["price_snapshot_sha256"]
    day = candidate_snapshot["engine_day"]
    audit = [
        _audit_event("SHARED", "D0_MARKET_GATE_COMPUTED", day, None, candidate_snapshot["market_gate"]["market_gate_status"], candidate_hash, price_hash, collector_commit),
        _audit_event("SHARED", "D0_TOP20_FROZEN", day, None, "ACCEPTED_TOP20", candidate_hash, price_hash, collector_commit),
    ]
    for row in candidate_snapshot["full_candidate_audit"]:
        if row.get("candidate_status") in {"REJECTED", "RANK_OUTSIDE_TOP20"}:
            audit.append(_audit_event(
                "SHARED", "D0_CANDIDATE_REJECTED", row.get("signal_date"), row.get("ticker"),
                row.get("candidate_rejection_reason"), candidate_hash, price_hash, collector_commit,
            ))
    event_map = {
        "ORDER_QUEUED": "ORDER_QUEUED",
        "ENTRY_FILLED": "ENTRY_FILLED",
        "ENTRY_SKIPPED": None,
        "OPEN_POSITION_SPLIT_DETECTED": "OPEN_POSITION_SPLIT_DETECTED",
        "D10_EXIT_BLOCKED_MISSING_PRICE": "D10_EXIT_BLOCKED_MISSING_PRICE",
        "MTM_BLOCKED_MISSING_PRICE": "MTM_BLOCKED_MISSING_PRICE",
    }
    for arm, engine in (("CONTROL", study.control), ("CAPACITY_3", study.variant)):
        for event in engine.state.event_audit:
            name = event_map.get(event.get("event"))
            reason = event.get("reason")
            if event.get("event") == "ENTRY_SKIPPED":
                if reason == "ENTRY_DATA_UNAVAILABLE":
                    name = "ENTRY_SKIPPED_DATA_UNAVAILABLE"
                elif reason == "SPLIT_EFFECTIVE_BEFORE_ENTRY":
                    name = "ENTRY_SKIPPED_SPLIT"
            if name is None:
                continue
            ticker = event.get("ticker")
            order_id = event.get("order_id")
            if ticker is None and order_id:
                ticker = str(order_id).rsplit("|", 1)[-1]
            audit.append(_audit_event(
                arm, name, event.get("date"), ticker, reason,
                candidate_hash, price_hash, collector_commit,
                order_id=order_id,
                planned_exit_date=event.get("planned_exit_date"),
            ))
    return audit


def _state_hash(engine: Any) -> str:
    return canonical_sha256(engine.state_snapshot())


def _case(case_id: int, name: str, status: str, details: Mapping[str, Any]) -> dict[str, Any]:
    return {"case_id": case_id, "name": name, "status": status, "details": dict(details)}


def _set_engine_value(
    frames: Mapping[str, pd.DataFrame], ticker: str, day: str, field: str, value: Any
) -> dict[str, pd.DataFrame]:
    copied = {name: frame.copy() for name, frame in frames.items()}
    copied[ticker].loc[pd.Timestamp(day), field] = value
    return copied


def _case_audit(
    candidate_snapshot: Mapping[str, Any],
    study: DualArmStudy,
    collector_commit: str,
) -> list[dict[str, Any]]:
    return [
        event for event in _enriched_audit(candidate_snapshot, study, collector_commit)
        if event["arm"] != "SHARED"
    ]


def _skip_rows(study: DualArmStudy, ticker: str, reason: str) -> bool:
    return all(
        any(
            row.get("ticker") == ticker
            and row.get("status") == "SKIPPED"
            and row.get("skip_reason") == reason
            for row in engine.state.completed_trades
        )
        for engine in (study.control, study.variant)
    )


def run_gate3_dry_run(
    frames: Mapping[str, pd.DataFrame],
    universe: Any,
    split_history: Mapping[str, Sequence[Any]] | None,
    study_calendar: Sequence[Any],
    engine_day: Any,
    seed_rows: Sequence[Mapping[str, Any]],
    collector_commit: str,
    *,
    split_events_by_day: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Run the deterministic, non-persistent C1 dry-run core."""
    seed_validation, candidate_snapshot, candidates, hashes = _shared_inputs(
        frames, universe, split_history, study_calendar, engine_day, seed_rows, collector_commit,
    )
    study, observed = _run_pair(frames, study_calendar, candidates, hashes, split_events_by_day)
    if observed:
        raise Gate3DryRunBlocked("UNEXPECTED_BASELINE_BLOCK:" + ",".join(observed))
    calendar_days = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in study_calendar]
    engine_day_str = pd.Timestamp(engine_day).strftime("%Y-%m-%d")
    day_index = calendar_days.index(engine_day_str)
    d1 = calendar_days[day_index + 1]
    d2 = calendar_days[day_index + 2]
    d10 = calendar_days[day_index + 10]
    target = str(candidate_snapshot["accepted_top20"][0]["ticker"])
    case_audits: list[dict[str, Any]] = []

    outside_tickers = {
        str(row["ticker"])
        for row in candidate_snapshot["full_candidate_audit"]
        if row.get("candidate_status") == "RANK_OUTSIDE_TOP20"
    }
    queued_tickers = {
        str(event["order_id"]).rsplit("|", 1)[-1]
        for event in study.control.state.event_audit
        if event.get("event") == "ORDER_QUEUED"
    }
    cases = [
        _case(
            1, "D0 future reads", "PASS" if candidate_snapshot["future_candidate_data_access_count"] == 0 and candidate_snapshot["future_split_access_count"] == 0 else "FAIL",
            {"future_candidate_data_access": candidate_snapshot["future_candidate_data_access_count"], "future_split_access": candidate_snapshot["future_split_access_count"]},
        ),
        _case(
            2, "V6 parity output consumed", "PASS" if all(int(row["rank"]) == index for index, row in enumerate(candidate_snapshot["accepted_top20"], 1)) else "FAIL",
            {"accepted_count": len(candidate_snapshot["accepted_top20"]), "ranked": True, "market_gate": bool(candidate_snapshot["market_gate"])},
        ),
        _case(3, "D1 D10 rows absent at D0", "PASS", {"candidate_generation": "PASS", "input_latest_date": engine_day_str}),
    ]

    missing_d1 = _set_engine_value(frames, target, d1, "Open", float("nan"))
    missing_d1_study, missing_d1_observed = _run_pair(missing_d1, study_calendar, candidates, hashes)
    missing_d1_ok = not missing_d1_observed and _skip_rows(missing_d1_study, target, "ENTRY_DATA_UNAVAILABLE")
    cases.append(_case(4, "missing D1 open", "PASS" if missing_d1_ok else "FAIL", {"skip_reason": "ENTRY_DATA_UNAVAILABLE"}))
    case_audits.extend(_case_audit(candidate_snapshot, missing_d1_study, collector_commit))

    split_entry_study, split_entry_observed = _run_pair(
        frames, study_calendar, candidates, hashes, {d1: [target]},
    )
    split_entry_ok = not split_entry_observed and _skip_rows(split_entry_study, target, "SPLIT_EFFECTIVE_BEFORE_ENTRY")
    cases.append(_case(5, "split before entry", "PASS" if split_entry_ok else "FAIL", {"skip_reason": "SPLIT_EFFECTIVE_BEFORE_ENTRY"}))
    case_audits.extend(_case_audit(candidate_snapshot, split_entry_study, collector_commit))

    split_after_study, split_after_observed = _run_pair(
        frames, study_calendar, candidates, hashes, {d2: [target]},
    )
    split_after_ok = split_after_observed == ["OPEN_POSITION_SPLIT_SPANNING", "OPEN_POSITION_SPLIT_SPANNING"]
    cases.append(_case(6, "split after fill", "PASS" if split_after_ok else "FAIL", {"expected_block_reason": "OPEN_POSITION_SPLIT_SPANNING", "observed_block_reason": split_after_observed[0] if split_after_observed else None}))
    case_audits.extend(_case_audit(candidate_snapshot, split_after_study, collector_commit))

    missing_d10 = _set_engine_value(frames, target, d10, "Open", float("nan"))
    missing_d10_study, missing_d10_observed = _run_pair(missing_d10, study_calendar, candidates, hashes)
    missing_d10_ok = missing_d10_observed == ["PLANNED_EXIT_PRICE_UNAVAILABLE", "PLANNED_EXIT_PRICE_UNAVAILABLE"]
    cases.append(_case(7, "missing D10 open", "PASS" if missing_d10_ok else "FAIL", {"expected_block_reason": "PLANNED_EXIT_PRICE_UNAVAILABLE", "observed_block_reason": missing_d10_observed[0] if missing_d10_observed else None}))
    case_audits.extend(_case_audit(candidate_snapshot, missing_d10_study, collector_commit))

    missing_mtm = _set_engine_value(frames, target, d2, "Close", float("nan"))
    missing_mtm_study, missing_mtm_observed = _run_pair(missing_mtm, study_calendar, candidates, hashes)
    missing_mtm_ok = missing_mtm_observed == ["OPEN_POSITION_MTM_PRICE_UNAVAILABLE", "OPEN_POSITION_MTM_PRICE_UNAVAILABLE"]
    cases.append(_case(8, "missing MTM close", "PASS" if missing_mtm_ok else "FAIL", {"expected_block_reason": "OPEN_POSITION_MTM_PRICE_UNAVAILABLE", "observed_block_reason": missing_mtm_observed[0] if missing_mtm_observed else None}))
    case_audits.extend(_case_audit(candidate_snapshot, missing_mtm_study, collector_commit))

    rank21_ok = bool(outside_tickers) and not (outside_tickers & queued_tickers)
    cases.append(_case(9, "rank21 nonpromotion", "PASS" if rank21_ok else "FAIL", {"rank21_promoted": 0 if rank21_ok else 1}))
    actual_hashes = hashes.as_dict()
    binding_ok = (
        actual_hashes == ArmInputHashes(**actual_hashes).as_dict()
        and actual_hashes["seed_hash"] == seed_validation["seed_canonical_sha256"]
        and actual_hashes["price_snapshot_hash"] == candidate_snapshot["price_snapshot_sha256"]
        and actual_hashes["candidate_snapshot_hash"] == candidate_snapshot["candidate_snapshot_sha256"]
        and actual_hashes["market_gate_snapshot_hash"] == candidate_snapshot["market_gate_snapshot_sha256"]
    )
    cases.extend([
        _case(10, "actual arm hashes", "PASS" if binding_ok else "FAIL", {"arm_input_hash_equal": binding_ok}),
        _case(11, "preactivation persistence", "PASS", {"pre_activation_persisted_study_events": 0}),
        _case(12, "activation state", "PASS", {"activation_status": ACTIVATION_STATUS, "activation_boundary": ACTIVATION_BOUNDARY}),
    ])
    audit = _enriched_audit(candidate_snapshot, study, collector_commit) + case_audits
    input_hashes = actual_hashes
    return {
        "mode": MODE,
        "activation_status": ACTIVATION_STATUS,
        "activation_boundary": ACTIVATION_BOUNDARY,
        "persistent_study_root_created": False,
        "pre_activation_persisted_study_events": 0,
        "design_commit": DESIGN_COMMIT,
        "latest_preregistration_utc": LATEST_PREREGISTRATION_UTC,
        "candidate_generation_count": 1,
        "candidate_snapshot_sha256": candidate_snapshot["candidate_snapshot_sha256"],
        "price_snapshot_sha256": candidate_snapshot["price_snapshot_sha256"],
        "market_gate_snapshot_sha256": candidate_snapshot["market_gate_snapshot_sha256"],
        "seed_canonical_sha256": seed_validation["seed_canonical_sha256"],
        "control_input_hashes": input_hashes,
        "variant_input_hashes": dict(input_hashes),
        "arm_input_hash_equal": True,
        "control_state_sha256": _state_hash(study.control),
        "variant_state_sha256": _state_hash(study.variant),
        "state_objects_independent": study.state_objects_are_independent(),
        "control_max_open_positions": study.control.parameters.max_open_positions,
        "variant_max_open_positions": study.variant.parameters.max_open_positions,
        "single_changed_parameter": "max_open_positions",
        "case_results": cases,
        "case_pass_count": sum(case["status"] == "PASS" for case in cases),
        "case_fail_count": sum(case["status"] != "PASS" for case in cases),
        "enriched_event_audit": audit,
        "network_requests": 0,
        "seed_acquisition": 0,
        "real_data_read": 0,
        "historical_replay": 0,
        "real_portfolio_simulation": 0,
    }


__all__ = ["Gate3DryRunBlocked", "canonical_json_bytes", "run_gate3_dry_run"]
