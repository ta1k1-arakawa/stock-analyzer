"""Read-only V6-A-R2 candidate preflight.

This module loads and validates the frozen caches, reuses only the frozen
V6-A data/candidate functions, adapts accepted top-20 rows, and performs
candidate parity.  It never constructs or runs the R2 portfolio engine and
does not create output files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from v6_a_confirmed_breakout import (
    EVALUATION_MANIFEST_SHA,
    TICKER_LIST_SHA,
    TRAINING_MANIFEST_SHA,
    UNIVERSE_CSV_SHA,
    audit_overlap,
    combine_source_aware,
    common_calendar,
    generate_candidates,
    load_cache,
    validate_universe,
)
from v6_a_r2_causal_breakout import validate_candidate_schema


EXPECTED_PREFLIGHT = {
    "training_tickers": 283,
    "evaluation_tickers": 300,
    "overlap_tickers": 283,
    "overlap_rows": 67843,
    "raw_ohlcv_mismatches": 0,
    "adj_close_mismatches": 482,
    "affected_revised_tickers": ["4768", "7609"],
    "market_gate_pass_days": 691,
    "market_gate_blocked_days": 774,
    "accepted_top20_candidates": 608,
    "signal_days": 346,
    "yearly_candidate_counts": {"2020": 109, "2021": 107, "2022": 63,
                                 "2023": 118, "2024": 87, "2025": 124},
    "D1_missing": 0,
    "D10_missing": 0,
    "split_violations": 0,
    "nonfinite_accepted": 0,
    "duplicate_accepted_key": 0,
    "2026_signals": 0,
}

R2_CANDIDATE_COLUMNS = (
    "signal_year", "signal_date", "ticker", "industry", "rank",
    "signal_raw_close", "entry_attempt_date", "planned_exit_date",
    "candidate_status",
)


class PreflightBlocked(RuntimeError):
    def __init__(self, stage: str, error_code: str, diagnostics: Mapping[str, Any] | None = None):
        self.stage = stage
        self.error_code = error_code
        self.diagnostics = dict(diagnostics or {})
        super().__init__(error_code)


@dataclass(frozen=True)
class ParityResult:
    missing_in_r2: int
    extra_in_r2: int
    duplicate_keys: int
    accepted_candidate_key_sha256: str

@dataclass(frozen=True)
class ReadOnlyPreparation:
    preflight_result: dict[str, Any]
    raw_price_frames: Mapping[str, pd.DataFrame]
    common_calendar: pd.DatetimeIndex
    accepted_candidates: list[dict[str, Any]]
    full_candidate_audit: pd.DataFrame
    market_gate_audit: Mapping[Any, Mapping[str, Any]]
    combined_splits: Mapping[str, set[pd.Timestamp]]


def _iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("INVALID_CANDIDATE_DATE")
    return timestamp.strftime("%Y-%m-%d")


def _key(row: Mapping[str, Any]) -> str:
    return f"{row['signal_date']}|{str(row['ticker'])}|{int(row['rank'])}"


def adapt_accepted_candidates(accepted: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert frozen accepted rows to exactly the nine R2 input columns."""
    required = {"signal_date", "ticker", "industry", "rank", "raw_close",
                "entry_date", "exit_date", "candidate_status"}
    missing = sorted(required.difference(accepted.columns))
    if missing:
        raise ValueError(f"ACCEPTED_CANDIDATE_SCHEMA_MISSING:{','.join(missing)}")
    rows: list[dict[str, Any]] = []
    for record in accepted.to_dict(orient="records"):
        signal_date = _iso(record["signal_date"])
        entry_date = _iso(record["entry_date"])
        exit_date = _iso(record["exit_date"])
        rank = int(record["rank"])
        if record["candidate_status"] != "ACCEPTED_TOP20":
            raise ValueError("ADAPTER_INPUT_NOT_ACCEPTED_TOP20")
        if rank > 20:
            raise ValueError("OUTSIDE_TOP20_CANDIDATE_PROHIBITED")
        rows.append({
            "signal_year": int(signal_date[:4]),
            "signal_date": signal_date,
            "ticker": str(record["ticker"]),
            "industry": str(record["industry"]),
            "rank": rank,
            "signal_raw_close": float(record["raw_close"]),
            "entry_attempt_date": entry_date,
            "planned_exit_date": exit_date,
            "candidate_status": "ACCEPTED_TOP20",
        })
    return rows


def canonical_candidate_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(_key(row) for row in rows)


def candidate_key_sha256(keys: Sequence[str]) -> str:
    canonical = "".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compare_candidate_parity(accepted: pd.DataFrame, r2_rows: Sequence[Mapping[str, Any]]) -> ParityResult:
    old_rows = [{"signal_date": _iso(row["signal_date"]), "ticker": str(row["ticker"]),
                 "rank": int(row["rank"])} for row in accepted.to_dict(orient="records")]
    old_keys = [_key(row) for row in old_rows]
    r2_keys = [_key(row) for row in r2_rows]
    old_set, r2_set = set(old_keys), set(r2_keys)
    duplicate_keys = (len(old_keys) - len(old_set)) + (len(r2_keys) - len(r2_set))
    return ParityResult(
        missing_in_r2=len(old_set - r2_set),
        extra_in_r2=len(r2_set - old_set),
        duplicate_keys=duplicate_keys,
        accepted_candidate_key_sha256=candidate_key_sha256(r2_keys),
    )


def _source_overlap(overlap: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overlap_tickers": int(overlap["overlap_tickers"]),
        "overlap_rows": int(overlap["overlap_rows"]),
        "raw_ohlcv_mismatches": int(overlap["raw_ohlcv_mismatch"]),
        "adj_close_mismatches": int(overlap["adjclose_mismatch"]),
        "affected_revised_tickers": list(overlap["adjclose_mismatch_tickers"]),
        "overlap_min": pd.Timestamp(overlap["overlap_min"]).strftime("%Y-%m-%d"),
        "overlap_max": pd.Timestamp(overlap["overlap_max"]).strftime("%Y-%m-%d"),
    }


def _candidate_counts(accepted: pd.DataFrame, audit: pd.DataFrame, gates: Mapping[Any, Mapping[str, Any]],
                      combined_splits: Mapping[str, set[pd.Timestamp]],
                      calendar: pd.DatetimeIndex) -> dict[str, Any]:
    yearly = {str(year): int((pd.to_datetime(accepted["signal_date"]).dt.year == year).sum())
              for year in range(2020, 2026)}
    blocked = sum(value["market_gate_status"] in {"MARKET_GATE_BLOCKED", "MARKET_GATE_INSUFFICIENT_UNIVERSE"}
                  for value in gates.values())
    passed = sum(value["market_gate_status"] == "MARKET_GATE_PASS" for value in gates.values())
    calendar_set = {pd.Timestamp(day).normalize() for day in calendar}
    invalid_rows = 0
    d1_missing = d10_missing = 0
    split_violations = 0
    for record in accepted.to_dict(orient="records"):
        raw_close = pd.to_numeric(pd.Series([record.get("raw_close")]), errors="coerce").iloc[0]
        rank = pd.to_numeric(pd.Series([record.get("rank")]), errors="coerce").iloc[0]
        entry = pd.to_datetime(record.get("entry_date"), errors="coerce")
        exit_ = pd.to_datetime(record.get("exit_date"), errors="coerce")
        entry_ok = pd.notna(entry) and pd.Timestamp(entry).normalize() in calendar_set
        exit_ok = pd.notna(exit_) and pd.Timestamp(exit_).normalize() in calendar_set
        if not entry_ok:
            d1_missing += 1
        if not exit_ok:
            d10_missing += 1
        row_invalid = not (pd.notna(raw_close) and pd.notna(rank) and entry_ok and exit_ok)
        invalid_rows += int(row_invalid)
        if entry_ok and exit_ok:
            ticker_splits = {pd.Timestamp(day).normalize() for day in combined_splits.get(str(record["ticker"]), set())}
            split_violations += int(any(pd.Timestamp(entry).normalize() <= day <= pd.Timestamp(exit_).normalize()
                                        for day in ticker_splits))
    rejected_split_spanning = int((audit.get("candidate_rejection_reason", pd.Series(dtype=str)) == "SPLIT_SPANNING").sum())
    return {
        "market_gate_counts": {"pass_days": passed, "blocked_days": blocked},
        "candidate_counts": {"accepted_top20": int(len(accepted)), "signal_days": int(accepted["signal_date"].nunique()) if len(accepted) else 0},
        "yearly_candidate_counts": yearly,
        "D1_missing": d1_missing,
        "D10_missing": d10_missing,
        "split_violations": split_violations,
        "rejected_split_spanning_count": rejected_split_spanning,
        "nonfinite_accepted": invalid_rows,
        "duplicate_accepted_key": int(accepted.duplicated(["signal_date", "ticker", "rank"]).sum()) if len(accepted) else 0,
        "2026_signals": int(pd.to_datetime(accepted["signal_date"]).dt.year.eq(2026).sum()) if len(accepted) else 0,
    }


def _generate_candidates_read_only(frames: Mapping[str, pd.DataFrame], universe: pd.DataFrame,
                                   splits: Mapping[str, set[pd.Timestamp]], calendar: pd.DatetimeIndex):
    """Call the frozen generator with an in-memory normalization cache only.

    The old generator's feature and ranking logic is unchanged.  Its global
    normalizer is restored in ``finally``; no repository or cache state is
    touched.
    """
    original_normalizer = generate_candidates.__globals__["adjusted_columns"]
    normalized: dict[int, pd.DataFrame] = {}

    def cached_normalizer(frame: pd.DataFrame) -> pd.DataFrame:
        key = id(frame)
        if key not in normalized:
            normalized[key] = original_normalizer(frame)
        return normalized[key]

    generate_candidates.__globals__["adjusted_columns"] = cached_normalizer
    try:
        return generate_candidates(frames, universe, splits, calendar)
    finally:
        generate_candidates.__globals__["adjusted_columns"] = original_normalizer


def validate_preflight_expectations(result: Mapping[str, Any], expected: Mapping[str, Any] = EXPECTED_PREFLIGHT) -> None:
    overlap = result["source_overlap_audit"]
    checks = {
        "training_tickers": result["training_tickers"],
        "evaluation_tickers": result["evaluation_tickers"],
        "overlap_tickers": overlap["overlap_tickers"],
        "overlap_rows": overlap["overlap_rows"],
        "raw_ohlcv_mismatches": overlap["raw_ohlcv_mismatches"],
        "adj_close_mismatches": overlap["adj_close_mismatches"],
        "affected_revised_tickers": overlap["affected_revised_tickers"],
        "market_gate_pass_days": result["market_gate_counts"]["pass_days"],
        "market_gate_blocked_days": result["market_gate_counts"]["blocked_days"],
        "accepted_top20_candidates": result["candidate_counts"]["accepted_top20"],
        "signal_days": result["candidate_counts"]["signal_days"],
        "yearly_candidate_counts": result["yearly_candidate_counts"],
        "D1_missing": result["D1_missing"], "D10_missing": result["D10_missing"],
        "split_violations": result["split_violations"], "nonfinite_accepted": result["nonfinite_accepted"],
        "duplicate_accepted_key": result["duplicate_accepted_key"], "2026_signals": result["2026_signals"],
    }
    mismatches = {key: {"actual": checks[key], "expected": expected[key]}
                  for key in expected if checks[key] != expected[key]}
    if mismatches:
        raise PreflightBlocked(
            "FIXED_EXPECTATION_VALIDATION", "FIXED_EXPECTATION_MISMATCH",
            {"actual_preflight_values": checks,
             "expected_preflight_values": dict(expected),
             "expectation_mismatches": mismatches,
             **_parity_diagnostics(result)},
        )


def _parity_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return only bounded, safe diagnostic values suitable for stdout."""
    keys = ("market_gate_counts", "candidate_counts", "yearly_candidate_counts",
            "missing_in_r2", "extra_in_r2", "duplicate_keys",
            "accepted_candidate_key_sha256", "split_violations",
            "rejected_split_spanning_count")
    return {key: result[key] for key in keys if key in result}


def blocked_json_payload(error: PreflightBlocked) -> dict[str, Any]:
    return {"verdict": "V6_A_R2_REAL_CACHE_PREFLIGHT_BLOCKED",
            "blocked_stage": error.stage, "error": error.error_code,
            **error.diagnostics}


def _load_read_only_formal_inputs(training_cache: str | Path, evaluation_cache: str | Path) -> dict[str, Any]:
    """Return validated in-memory inputs for the later formal evaluator."""
    repo = Path(__file__).resolve().parents[1]
    try:
        universe = validate_universe(repo / "V4_UNIVERSE.csv")
        _, training_prices, training_splits = load_cache(Path(training_cache), TRAINING_MANIFEST_SHA, universe)
        _, evaluation_prices, evaluation_splits = load_cache(Path(evaluation_cache), EVALUATION_MANIFEST_SHA, universe)
        frames = combine_source_aware(training_prices, evaluation_prices)
        calendar = common_calendar(frames)
        splits = {ticker: training_splits.get(ticker, set()) | evaluation_splits.get(ticker, set()) for ticker in frames}
        accepted_all, gates, audit = _generate_candidates_read_only(frames, universe, splits, calendar)
        accepted = accepted_all[accepted_all["candidate_status"] == "ACCEPTED_TOP20"].copy()
        rows = adapt_accepted_candidates(accepted)
    except Exception as error:
        raise PreflightBlocked("FORMAL_INPUT_BUNDLE", "FORMAL_INPUT_BUNDLE_FAILED") from error
    return {"raw_price_frames": frames, "common_calendar": calendar, "accepted_candidates": rows,
            "full_candidate_audit": audit, "market_gate_audit": gates}


def prepare_read_only_formal_bundle(training_cache: str | Path, evaluation_cache: str | Path,
                                    repository_commit: str, branch: str, worktree_clean: bool) -> ReadOnlyPreparation:
    """Compatibility wrapper for the canonical preparation path."""
    return prepare_read_only_context(training_cache, evaluation_cache, repository_commit, branch, worktree_clean)


def prepare_read_only_context(training_cache: str | Path, evaluation_cache: str | Path,
                            repository_commit: str, branch: str, worktree_clean: bool) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    try:
        universe = validate_universe(repo / "V4_UNIVERSE.csv")
        _, training_prices, training_splits = load_cache(Path(training_cache), TRAINING_MANIFEST_SHA, universe)
        _, evaluation_prices, evaluation_splits = load_cache(Path(evaluation_cache), EVALUATION_MANIFEST_SHA, universe)
    except Exception as error:
        raise PreflightBlocked("CACHE_VALIDATION", "CACHE_VALIDATION_FAILED") from error
    try:
        overlap = audit_overlap(training_prices, evaluation_prices)
        frames = combine_source_aware(training_prices, evaluation_prices)
        calendar = common_calendar(frames)
    except Exception as error:
        raise PreflightBlocked("SOURCE_OVERLAP", "SOURCE_OVERLAP_FAILED") from error
    try:
        combined_splits = {ticker: training_splits.get(ticker, set()) | evaluation_splits.get(ticker, set())
                           for ticker in frames}
        accepted_all, gates, audit = _generate_candidates_read_only(frames, universe, combined_splits, calendar)
        if accepted_all.empty:
            raise ValueError("NO_CANDIDATES")
    except PreflightBlocked:
        raise
    except Exception as error:
        raise PreflightBlocked("CANDIDATE_GENERATION", "CANDIDATE_GENERATION_FAILED") from error
    accepted = accepted_all[accepted_all["candidate_status"] == "ACCEPTED_TOP20"].copy()
    try:
        rows = adapt_accepted_candidates(accepted)
        iso_calendar = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar]
        validate_candidate_schema(iso_calendar, rows)
        counts = _candidate_counts(accepted, audit, gates, combined_splits, calendar)
    except Exception as error:
        raise PreflightBlocked("CANDIDATE_ADAPTER", "CANDIDATE_ADAPTER_FAILED") from error
    try:
        parity = compare_candidate_parity(accepted, rows)
    except Exception as error:
        raise PreflightBlocked("CANDIDATE_PARITY", "CANDIDATE_PARITY_FAILED") from error
    result: dict[str, Any] = {
        "verdict": "V6_A_R2_REAL_CACHE_PREFLIGHT_PASS",
        "repository_commit": repository_commit, "branch": branch, "worktree_clean": bool(worktree_clean),
        "training_manifest_sha": TRAINING_MANIFEST_SHA, "evaluation_manifest_sha": EVALUATION_MANIFEST_SHA,
        "universe_csv_sha": UNIVERSE_CSV_SHA, "ticker_list_sha": TICKER_LIST_SHA,
        "training_tickers": len(training_prices), "evaluation_tickers": len(evaluation_prices),
        "source_overlap_audit": _source_overlap(overlap),
        "market_gate_counts": counts["market_gate_counts"],
        "candidate_counts": counts["candidate_counts"],
        "yearly_candidate_counts": counts["yearly_candidate_counts"],
        "D1_missing": counts["D1_missing"], "D10_missing": counts["D10_missing"],
        "split_violations": counts["split_violations"],
        "rejected_split_spanning_count": counts["rejected_split_spanning_count"],
        "nonfinite_accepted": counts["nonfinite_accepted"],
        "duplicate_accepted_key": counts["duplicate_accepted_key"], "2026_signals": counts["2026_signals"],
        "missing_in_r2": parity.missing_in_r2, "extra_in_r2": parity.extra_in_r2,
        "duplicate_keys": parity.duplicate_keys,
        "accepted_candidate_key_sha256": parity.accepted_candidate_key_sha256,
        "future_price_values_in_engine_candidates": 0,
        "portfolio_engine_instantiated": False, "portfolio_engine_run_calls": 0,
        "portfolio_simulation": 0, "profit_calculation": 0, "formal_evaluation": 0,
        "formal_artifacts": 0, "network": 0, "cache_modification": 0,
    }
    if not worktree_clean or parity.missing_in_r2 or parity.extra_in_r2 or parity.duplicate_keys:
        raise PreflightBlocked("CANDIDATE_PARITY", "CANDIDATE_PARITY_MISMATCH", _parity_diagnostics(result))
    try:
        validate_preflight_expectations(result)
    except PreflightBlocked:
        raise
    return ReadOnlyPreparation(result, frames, calendar, rows, audit, gates, combined_splits)


def run_read_only_preflight(training_cache: str | Path, evaluation_cache: str | Path, repository_commit: str, branch: str, worktree_clean: bool) -> dict[str, Any]:
    return prepare_read_only_context(training_cache, evaluation_cache, repository_commit, branch, worktree_clean).preflight_result
