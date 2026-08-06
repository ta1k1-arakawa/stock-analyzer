from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v6_a_r2_preflight import (  # noqa: E402
    EXPECTED_PREFLIGHT,
    R2_CANDIDATE_COLUMNS,
    PreflightBlocked,
    adapt_accepted_candidates,
    candidate_key_sha256,
    canonical_candidate_keys,
    compare_candidate_parity,
    blocked_json_payload,
    _candidate_counts,
    ReadOnlyPreparation,
    prepare_read_only_context,
    prepare_read_only_formal_bundle,
    run_read_only_preflight,
    validate_preflight_expectations,
)
from v6_a_r2_causal_breakout import validate_candidate_schema  # noqa: E402


def _accepted_rows():
    return pd.DataFrame([
        {"signal_year": 2020, "signal_date": pd.Timestamp("2020-01-02"), "ticker": "1111", "industry": "A",
         "rank": 2, "raw_close": 100.0, "entry_date": pd.Timestamp("2020-01-03"),
         "exit_date": pd.Timestamp("2020-01-12"), "candidate_status": "ACCEPTED_TOP20"},
        {"signal_year": 2020, "signal_date": pd.Timestamp("2020-01-02"), "ticker": "2222", "industry": "B",
         "rank": 1, "raw_close": 200.0, "entry_date": pd.Timestamp("2020-01-03"),
         "exit_date": pd.Timestamp("2020-01-12"), "candidate_status": "ACCEPTED_TOP20"},
    ])


def _calendar():
    return [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20)]


def test_adapter_has_exactly_nine_columns():
    rows = adapt_accepted_candidates(_accepted_rows())
    assert set(rows[0]) == set(R2_CANDIDATE_COLUMNS)
    assert tuple(rows[0]) == R2_CANDIDATE_COLUMNS


def test_adapter_contains_no_future_price_values():
    rows = adapt_accepted_candidates(_accepted_rows())
    forbidden = {"d1_open", "d10_open", "entry_price", "exit_price", "future_return", "future_profit"}
    assert forbidden.isdisjoint(rows[0])
    assert rows[0]["signal_raw_close"] == 100.0


def test_old_and_r2_keys_are_exactly_equal():
    accepted = _accepted_rows()
    rows = adapt_accepted_candidates(accepted)
    parity = compare_candidate_parity(accepted, rows)
    assert parity.missing_in_r2 == 0
    assert parity.extra_in_r2 == 0
    assert parity.duplicate_keys == 0


def test_canonical_hash_is_order_independent_and_newline_delimited():
    rows = adapt_accepted_candidates(_accepted_rows())
    keys = canonical_candidate_keys(rows)
    expected = hashlib.sha256("".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")).hexdigest()
    assert candidate_key_sha256(list(reversed(keys))) == expected
    assert keys == sorted(keys)


def test_duplicate_key_fails_closed_in_engine_schema_validation():
    rows = adapt_accepted_candidates(_accepted_rows())
    with pytest.raises(ValueError, match="DUPLICATE_CANDIDATE_KEY"):
        validate_candidate_schema(_calendar(), rows + [copy.deepcopy(rows[0])])


def test_rank21_fails_closed():
    accepted = _accepted_rows()
    accepted.loc[0, "rank"] = 21
    with pytest.raises(ValueError, match="OUTSIDE_TOP20_CANDIDATE_PROHIBITED"):
        adapt_accepted_candidates(accepted)


def test_2026_signal_fails_closed():
    rows = adapt_accepted_candidates(_accepted_rows())
    rows[0]["signal_year"] = 2026
    rows[0]["signal_date"] = "2026-01-02"
    rows[0]["entry_attempt_date"] = "2026-01-03"
    rows[0]["planned_exit_date"] = "2026-01-12"
    calendar = [(pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20)]
    with pytest.raises(ValueError, match="SIGNAL_2026_PROHIBITED"):
        validate_candidate_schema(calendar, rows)


def test_invalid_date_fails_closed():
    rows = adapt_accepted_candidates(_accepted_rows())
    rows[0]["signal_date"] = "2020-1-2"
    with pytest.raises(ValueError, match="INVALID_DATE_FORMAT"):
        validate_candidate_schema(_calendar(), rows)


def test_preflight_source_never_calls_engine_or_phases():
    source = (Path(__file__).resolve().parents[1] / "src" / "v6_a_r2_preflight.py").read_text(encoding="utf-8")
    assert "CausalEventEngine" not in source
    assert ".run(" not in source
    assert "process_day" not in source
    assert not any(f"phase{i}_" in source for i in range(1, 6))


def test_forbidden_old_simulator_and_formal_helpers_are_not_imported():
    source = (Path(__file__).resolve().parents[1] / "src" / "v6_a_r2_preflight.py").read_text(encoding="utf-8")
    for forbidden in ("simulate_fold", "metrics", "compute_gates", "verdict_from_gates", "atomic_write"):
        assert forbidden not in source


def test_preflight_has_no_profit_or_artifact_calls():
    source = (Path(__file__).resolve().parents[1] / "src" / "v6_a_r2_preflight.py").read_text(encoding="utf-8")
    assert "profit_calculation" in source  # output declaration only
    assert ".write(" not in source
    assert "write_bytes" not in source
    assert "metrics(" not in source
    assert "compute_gates(" not in source


def test_preflight_has_no_network_import_or_function():
    source = (Path(__file__).resolve().parents[1] / "src" / "v6_a_r2_preflight.py").read_text(encoding="utf-8").lower()
    assert all(token not in source for token in ("requests", "urllib", "httpx", "yfinance", "socket"))


def test_evaluate_cache_mode_remains_unauthorized(capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_v6_a_r2_causal_breakout as runner
    assert runner.main(["--evaluate-cache"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "GATE_4_FORMAL_EVALUATION_CONFIRMATION_REQUIRED"
    assert payload["portfolio_simulation_started"] == 0


def test_preflight_json_schema_keys():
    required = {"verdict", "repository_commit", "branch", "worktree_clean",
                "training_manifest_sha", "evaluation_manifest_sha", "universe_csv_sha", "ticker_list_sha",
                "source_overlap_audit", "market_gate_counts", "candidate_counts", "yearly_candidate_counts",
                "missing_in_r2", "extra_in_r2", "duplicate_keys", "accepted_candidate_key_sha256",
                "future_price_values_in_engine_candidates", "portfolio_engine_instantiated",
                "portfolio_engine_run_calls", "portfolio_simulation", "profit_calculation", "formal_evaluation",
                "formal_artifacts", "network", "cache_modification"}
    source = (Path(__file__).resolve().parents[1] / "src" / "v6_a2_preflight.py")
    assert not source.exists()
    module_source = (Path(__file__).resolve().parents[1] / "src" / "v6_a2_preflight.py")
    assert not module_source.exists()
    preflight_source = (Path(__file__).resolve().parents[1] / "src" / "v6_a2_preflight.py")
    assert not preflight_source.exists()
    # The executable result schema is represented by the result construction keys.
    text = (Path(__file__).resolve().parents[1] / "src" / "v6_a_r2_preflight.py").read_text(encoding="utf-8")
    assert required.issubset({key for key in required if f'"{key}"' in text})


def test_cache_hash_mismatch_fails_closed():
    result = {"training_tickers": EXPECTED_PREFLIGHT["training_tickers"],
              "evaluation_tickers": EXPECTED_PREFLIGHT["evaluation_tickers"],
              "source_overlap_audit": {"overlap_tickers": 283, "overlap_rows": 67843,
                                        "raw_ohlcv_mismatches": 0, "adj_close_mismatches": 482,
                                        "affected_revised_tickers": ["4768", "7609"]},
              "market_gate_counts": {"pass_days": 691, "blocked_days": 774},
              "candidate_counts": {"accepted_top20": 608, "signal_days": 346},
              "yearly_candidate_counts": EXPECTED_PREFLIGHT["yearly_candidate_counts"],
              "D1_missing": 0, "D10_missing": 0, "split_violations": 0,
              "nonfinite_accepted": 0, "duplicate_accepted_key": 0, "2026_signals": 0}
    expected = dict(EXPECTED_PREFLIGHT)
    expected["training_tickers"] = 999
    with pytest.raises(PreflightBlocked, match="FIXED_EXPECTATION_MISMATCH") as caught:
        validate_preflight_expectations(result, expected)
    assert caught.value.stage == "FIXED_EXPECTATION_VALIDATION"
    assert caught.value.diagnostics["expectation_mismatches"]["training_tickers"] == {
        "actual": 283, "expected": 999}


def test_candidate_expectation_mismatch_fails_closed():
    result = {"training_tickers": 283, "evaluation_tickers": 300,
              "source_overlap_audit": {"overlap_tickers": 283, "overlap_rows": 67843,
                                        "raw_ohlcv_mismatches": 0, "adj_close_mismatches": 482,
                                        "affected_revised_tickers": ["4768", "7609"]},
              "market_gate_counts": {"pass_days": 691, "blocked_days": 774},
              "candidate_counts": {"accepted_top20": 607, "signal_days": 346},
              "yearly_candidate_counts": EXPECTED_PREFLIGHT["yearly_candidate_counts"],
              "D1_missing": 0, "D10_missing": 0, "split_violations": 0,
              "nonfinite_accepted": 0, "duplicate_accepted_key": 0, "2026_signals": 0}
    with pytest.raises(PreflightBlocked) as caught:
        validate_preflight_expectations(result)
    payload = blocked_json_payload(caught.value)
    assert payload["blocked_stage"] == "FIXED_EXPECTATION_VALIDATION"
    assert payload["actual_preflight_values"]["accepted_top20_candidates"] == 607
    assert payload["expected_preflight_values"]["accepted_top20_candidates"] == 608
    assert "accepted_top20_candidates" in payload["expectation_mismatches"]


def _audit_and_gates():
    return pd.DataFrame([{"candidate_rejection_reason": "SPLIT_SPANNING"}]), {
        "2020-01-01": {"market_gate_status": "MARKET_GATE_PASS"}}


def test_rejected_split_spanning_is_not_accepted_split_violation():
    accepted = _accepted_rows().iloc[:1].copy()
    audit, gates = _audit_and_gates()
    counts = _candidate_counts(accepted, audit, gates,
                               {"1111": {pd.Timestamp("2020-01-20")} },
                               pd.date_range("2020-01-01", periods=20))
    assert counts["split_violations"] == 0
    assert counts["rejected_split_spanning_count"] == 1


def test_accepted_split_violation_includes_entry_and_exit_boundaries():
    accepted = _accepted_rows().iloc[:1].copy()
    counts = _candidate_counts(accepted, pd.DataFrame(),
                               {"2020-01-01": {"market_gate_status": "MARKET_GATE_PASS"}},
                               {"1111": {pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-12")}},
                               pd.date_range("2020-01-01", periods=20))
    assert counts["split_violations"] == 1


def test_accepted_only_dates_and_nonfinite_are_measured():
    accepted = pd.concat([_accepted_rows().iloc[:1], _accepted_rows().iloc[:1]], ignore_index=True)
    accepted.loc[1, "raw_close"] = float("nan")
    accepted.loc[1, "rank"] = float("nan")
    accepted.loc[1, "entry_date"] = pd.NaT
    accepted.loc[1, "exit_date"] = pd.NaT
    counts = _candidate_counts(accepted,
                               pd.DataFrame([{"candidate_rejection_reason": "D1_MISSING"}]),
                               {"2020-01-01": {"market_gate_status": "MARKET_GATE_PASS"}},
                               {}, pd.date_range("2020-01-01", periods=20))
    assert counts["D1_missing"] == 1
    assert counts["D10_missing"] == 1
    assert counts["nonfinite_accepted"] == 1


def test_expected_preflight_constants_are_not_mutated_by_validation():
    before = copy.deepcopy(EXPECTED_PREFLIGHT)
    result = {"training_tickers": 0, "evaluation_tickers": 0,
              "source_overlap_audit": {"overlap_tickers": 0, "overlap_rows": 0,
                                        "raw_ohlcv_mismatches": 0, "adj_close_mismatches": 0,
                                        "affected_revised_tickers": []},
              "market_gate_counts": {"pass_days": 0, "blocked_days": 0},
              "candidate_counts": {"accepted_top20": 0, "signal_days": 0},
              "yearly_candidate_counts": {}, "D1_missing": 0, "D10_missing": 0,
              "split_violations": 0, "nonfinite_accepted": 0,
              "duplicate_accepted_key": 0, "2026_signals": 0}
    with pytest.raises(PreflightBlocked):
        validate_preflight_expectations(result)
    assert EXPECTED_PREFLIGHT == before


def test_single_preparation_path_call_counts(monkeypatch):
    import v6_a_r2_preflight as module

    calls = {"load_cache": [], "audit_overlap": 0, "combine": 0, "generate": 0,
             "adapt": 0, "parity": 0, "expectations": 0}
    accepted = _accepted_rows()
    prices = {"1111": pd.DataFrame({"Open": [100.0], "Close": [100.0]},
                                    index=[pd.Timestamp("2020-01-01")])}
    overlap = {"overlap_tickers": 283, "overlap_rows": 67843, "raw_ohlcv_mismatch": 0,
               "adjclose_mismatch": 482, "adjclose_mismatch_tickers": ["4768", "7609"],
               "overlap_min": pd.Timestamp("2019-01-04"), "overlap_max": pd.Timestamp("2019-12-30")}
    gates = {"2020-01-01": {"market_gate_status": "MARKET_GATE_PASS"}}
    audit = pd.DataFrame()

    monkeypatch.setattr(module, "validate_universe", lambda path: pd.DataFrame())
    def fake_load(path, manifest, universe):
        calls["load_cache"].append((str(path), manifest))
        return ({}, prices, {"1111": set()})
    monkeypatch.setattr(module, "load_cache", fake_load)
    monkeypatch.setattr(module, "audit_overlap", lambda a, b: overlap)
    monkeypatch.setattr(module, "combine_source_aware", lambda a, b: calls.__setitem__("combine", calls["combine"] + 1) or prices)
    monkeypatch.setattr(module, "common_calendar", lambda frames: pd.date_range("2020-01-01", periods=20))
    monkeypatch.setattr(module, "_generate_candidates_read_only", lambda *args: calls.__setitem__("generate", calls["generate"] + 1) or (accepted, gates, audit))
    original_adapt = module.adapt_accepted_candidates
    monkeypatch.setattr(module, "adapt_accepted_candidates", lambda value: calls.__setitem__("adapt", calls["adapt"] + 1) or original_adapt(value))
    monkeypatch.setattr(module, "validate_candidate_schema", lambda *args: None)
    original_parity = module.compare_candidate_parity
    monkeypatch.setattr(module, "compare_candidate_parity", lambda *args: calls.__setitem__("parity", calls["parity"] + 1) or original_parity(*args))
    monkeypatch.setattr(module, "validate_preflight_expectations", lambda result: calls.__setitem__("expectations", calls["expectations"] + 1))

    preparation = prepare_read_only_context("training", "evaluation", "sha", "branch", True)
    assert isinstance(preparation, ReadOnlyPreparation)
    assert len(calls["load_cache"]) == 2
    assert calls["load_cache"][0][0] == "training" and calls["load_cache"][1][0] == "evaluation"
    assert calls["combine"] == calls["generate"] == calls["adapt"] == calls["parity"] == calls["expectations"] == 1


def test_run_preflight_uses_context_once_and_returns_preflight_result(monkeypatch):
    import v6_a_r2_preflight as module
    result = {"verdict": "PASS"}
    preparation = ReadOnlyPreparation(result, {}, pd.DatetimeIndex([]), [], pd.DataFrame(), {}, {})
    calls = []
    monkeypatch.setattr(module, "prepare_read_only_context", lambda *args: calls.append(args) or preparation)
    assert run_read_only_preflight("training", "evaluation", "sha", "branch", True) is result
    assert len(calls) == 1


def test_formal_bundle_is_thin_single_context_wrapper(monkeypatch):
    import v6_a_r2_preflight as module
    preparation = ReadOnlyPreparation({"verdict": "PASS"}, {}, pd.DatetimeIndex([]), [], pd.DataFrame(), {}, {})
    calls = []
    monkeypatch.setattr(module, "prepare_read_only_context", lambda *args: calls.append(args) or preparation)
    assert prepare_read_only_formal_bundle("training", "evaluation", "sha", "branch", True) is preparation
    assert len(calls) == 1


def test_duplicate_formal_input_helper_removed():
    source = (Path(__file__).resolve().parents[1] / "src" / "v6_a_r2_preflight.py").read_text(encoding="utf-8")
    assert "_load_read_only_formal_inputs" not in source
    assert "-> ReadOnlyPreparation" in source
