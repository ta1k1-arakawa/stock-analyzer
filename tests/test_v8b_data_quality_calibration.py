from __future__ import annotations

import hashlib
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from src import v8b_data_quality_calibration as calib

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Local fake Yahoo chart payload builder (not imported from test_v7_yahoo_collector.py).
# ---------------------------------------------------------------------------


def _epoch(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp())


def _consecutive_days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=index) for index in range(count)]


def _payload_bytes(
    symbol: str,
    days: list[date],
    *,
    quote_overrides: dict[str, list] | None = None,
    include_adjclose: bool = True,
) -> bytes:
    n = len(days)
    quote = {
        "open": [100.0 + index for index in range(n)],
        "high": [101.0 + index for index in range(n)],
        "low": [99.0 + index for index in range(n)],
        "close": [100.5 + index for index in range(n)],
        "volume": [1000.0 + index for index in range(n)],
    }
    if quote_overrides:
        for field, values in quote_overrides.items():
            quote[field] = values
    indicators = {"quote": [quote]}
    if include_adjclose:
        indicators["adjclose"] = [{"adjclose": [100.25 + index for index in range(n)]}]
    body = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": symbol},
                    "timestamp": [_epoch(day) for day in days],
                    "indicators": indicators,
                    "events": {},
                }
            ],
        }
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.fixture(scope="module")
def pinned_module():
    return calib._load_pinned_collector(REPO_ROOT)


def _fabricated_clean_base(base_index: int) -> calib.CleanBase:
    row = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "adjclose": 100.25, "volume": 1000.0}
    return calib.CleanBase(
        base_index=base_index,
        ticker_sha256=hashlib.sha256(f"FAKE{base_index}\n".encode()).hexdigest(),
        window_start="2019-01-01",
        window_end="2019-09-09",
        rows=tuple(dict(row) for _ in range(calib.SYNTHETIC_SEQUENCE_LENGTH)),
    )


FABRICATED_BASES = tuple(_fabricated_clean_base(index) for index in range(calib.SYNTHETIC_BASE_COUNT))


def _repo_copy(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / calib.PREREGISTRATION_PATH).write_bytes((REPO_ROOT / calib.PREREGISTRATION_PATH).read_bytes())
    (tmp_path / calib.APPROVAL_ARTIFACT_PATH).write_bytes((REPO_ROOT / calib.APPROVAL_ARTIFACT_PATH).read_bytes())
    (tmp_path / calib.PINNED_COLLECTOR_PATH).write_bytes((REPO_ROOT / calib.PINNED_COLLECTOR_PATH).read_bytes())
    return tmp_path


VALID_TIMESTAMPS = dict(run_started_utc="2026-01-01T00:00:00Z", run_completed_or_blocked_utc="2026-01-01T00:00:01Z")
VALID_COMMIT = "0" * 40


# ---------------------------------------------------------------------------
# A-D. Repository contract verification
# ---------------------------------------------------------------------------


def test_repository_contract_passes_at_approved_state():
    result = calib.verify_repository_contract(REPO_ROOT)
    assert result == {
        "plan_blob_sha": calib.APPROVED_PLAN_BLOB_SHA,
        "approval_blob_sha": calib.APPROVAL_ARTIFACT_BLOB_SHA,
        "classifier_blob_sha": calib.PINNED_COLLECTOR_BLOB_SHA,
    }


def test_plan_one_byte_mutation_blocks(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.PREREGISTRATION_PATH
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 0xFF
    path.write_bytes(bytes(mutated))
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.verify_repository_contract(root)
    assert excinfo.value.reason == "CALIBRATION_PLAN_BLOB_MISMATCH"


def test_approval_one_byte_mutation_blocks(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.APPROVAL_ARTIFACT_PATH
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 0xFF
    path.write_bytes(bytes(mutated))
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.verify_repository_contract(root)
    assert excinfo.value.reason == "CALIBRATION_APPROVAL_BLOB_MISMATCH"


def test_classifier_one_byte_mutation_blocks(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.PINNED_COLLECTOR_PATH
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 0xFF
    path.write_bytes(bytes(mutated))
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.verify_repository_contract(root)
    assert excinfo.value.reason == "CALIBRATION_CLASSIFIER_VERSION_MISMATCH"


# ---------------------------------------------------------------------------
# Finding 1: pinned parser content binding (TOCTOU fix)
# ---------------------------------------------------------------------------


def test_pinned_loader_reads_the_file_exactly_once_never_reopens(tmp_path):
    root = _repo_copy(tmp_path)
    target = root / calib.PINNED_COLLECTOR_PATH
    call_count = {"n": 0}
    original_read_bytes = Path.read_bytes

    def counting_read_bytes(self):
        if self == target:
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise AssertionError("pinned collector file reopened after verification")
        return original_read_bytes(self)

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(Path, "read_bytes", counting_read_bytes)
        module = calib._load_pinned_collector(root)
        assert module.canonical_ticker("7203.T") == "7203"
    finally:
        mp.undo()
    assert call_count["n"] == 1


def test_pinned_loader_executes_exactly_the_verified_bytes_not_a_later_swap(tmp_path):
    root = _repo_copy(tmp_path)
    target = root / calib.PINNED_COLLECTOR_PATH
    real_bytes = target.read_bytes()
    malicious_bytes = real_bytes + b"\nBACKDOOR = True\n"
    reads = {"n": 0}
    original_read_bytes = Path.read_bytes

    def swap_after_first_read(self):
        if self == target:
            reads["n"] += 1
            if reads["n"] == 1:
                return real_bytes  # this is what gets verified
            return malicious_bytes  # a reopen would see this instead
        return original_read_bytes(self)

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(Path, "read_bytes", swap_after_first_read)
        module = calib._load_pinned_collector(root)
    finally:
        mp.undo()
    assert not hasattr(module, "BACKDOOR")
    assert reads["n"] == 1


def test_pinned_loader_uses_no_spec_from_file_location_or_exec_module():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration.py").read_text(encoding="utf-8")
    assert "spec_from_file_location" not in source
    assert "exec_module" not in source


def test_pinned_loader_rejects_mutated_bytes(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.PINNED_COLLECTOR_PATH
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 0xFF
    path.write_bytes(bytes(mutated))
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib._load_pinned_collector(root)
    assert excinfo.value.reason == "CALIBRATION_CLASSIFIER_VERSION_MISMATCH"


# ---------------------------------------------------------------------------
# E. Strict JSON duplicate-key rejection / git blob / canonical json
# ---------------------------------------------------------------------------


def test_duplicate_key_json_rejected():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.parse_strict_json('{"a":1,"a":2}')
    assert excinfo.value.reason == "STRICT_JSON_DUPLICATE_KEY"


def test_malformed_json_rejected():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.parse_strict_json("{not json")
    assert excinfo.value.reason == "STRICT_JSON_MALFORMED"


def test_git_blob_sha1_matches_known_approved_files():
    plan_bytes = (REPO_ROOT / calib.PREREGISTRATION_PATH).read_bytes()
    assert calib.git_blob_sha1(plan_bytes) == calib.APPROVED_PLAN_BLOB_SHA
    approval_bytes = (REPO_ROOT / calib.APPROVAL_ARTIFACT_PATH).read_bytes()
    assert calib.git_blob_sha1(approval_bytes) == calib.APPROVAL_ARTIFACT_BLOB_SHA
    classifier_bytes = (REPO_ROOT / calib.PINNED_COLLECTOR_PATH).read_bytes()
    assert calib.git_blob_sha1(classifier_bytes) == calib.PINNED_COLLECTOR_BLOB_SHA


def test_canonical_json_bytes_is_sorted_compact_and_newline_terminated():
    raw = calib.canonical_json_bytes({"b": 1, "a": 2})
    assert raw == b'{"a":2,"b":1}\n'


# ---------------------------------------------------------------------------
# F-G. Candidate grid
# ---------------------------------------------------------------------------


def test_candidate_grid_has_exactly_30_in_strictness_order():
    assert len(calib.CANDIDATES) == 30
    expected_ids = [
        f"{fraction_id}_C{c}"
        for fraction_id in ("F1", "F2", "FQ1", "F3", "F4", "F5")
        for c in range(1, 6)
    ]
    assert [candidate.id for candidate in calib.CANDIDATES] == expected_ids


def test_f2_declared_representation_stays_2_over_252():
    f2_candidates = [c for c in calib.CANDIDATES if c.fraction_id == "F2"]
    assert len(f2_candidates) == 5
    for candidate in f2_candidates:
        assert candidate.declared_numerator == 2
        assert candidate.declared_denominator == 252
    assert calib.candidate_fraction_value(f2_candidates[0]) == Fraction(1, 126)


# ---------------------------------------------------------------------------
# H. Exact fraction boundary arithmetic (no floats)
# ---------------------------------------------------------------------------


def test_fraction_guard_uses_exact_integer_arithmetic():
    flags_at_boundary = [True] + [False] * 251
    assert calib.quality_policy_pass(flags_at_boundary, 1, 252, 1) is True
    flags_over_boundary = [True, True] + [False] * 250
    assert calib.quality_policy_pass(flags_over_boundary, 1, 252, 2) is False


def test_quality_policy_pass_zero_observations_fails_closed():
    assert calib.quality_policy_pass([], 5, 252, 5) is False


# ---------------------------------------------------------------------------
# I. Calendar missing dates do not create observations
# ---------------------------------------------------------------------------


def test_missing_calendar_dates_are_simply_absent(pinned_module):
    days = [date(2019, 1, 2), date(2019, 1, 3), date(2019, 1, 7)]  # weekend gap, no fill
    payload = _payload_bytes("TICKA", days)
    canonical, observations = calib.parse_ticker_observations("TICKA", payload, pinned_module)
    assert canonical == "TICKA"
    assert len(observations) == 3
    assert [o.trading_date for o in observations] == ["2019-01-02", "2019-01-03", "2019-01-07"]


def test_calibration_window_restriction_excludes_january_2026(pinned_module):
    days = [date(2025, 12, 30), date(2026, 1, 15)]
    payload = _payload_bytes("TICKB", days)
    canonical, observations = calib.parse_ticker_observations("TICKB", payload, pinned_module)
    assert canonical == "TICKB"
    assert len(observations) == 1
    assert observations[0].trading_date == "2025-12-30"


# ---------------------------------------------------------------------------
# J-K. Year / full-span zero-observation classification
# ---------------------------------------------------------------------------


def test_year_with_zero_observations_is_not_applicable(pinned_module):
    days = _consecutive_days(date(2020, 1, 2), 5)
    payload = _payload_bytes("TICKC", days)
    _, observations = calib.parse_ticker_observations("TICKC", payload, pinned_module)
    yearly = calib.compute_yearly_window_stats(observations)
    assert yearly[2019] is None
    assert yearly[2020] is not None
    assert yearly[2020].total_returned == 5


def test_full_span_zero_observations_blocks():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.compute_full_span_stats(())
    assert excinfo.value.reason == "CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED"


# ---------------------------------------------------------------------------
# L. Cross-year consecutive run visible in full-span statistic
# ---------------------------------------------------------------------------


def test_cross_year_consecutive_run_visible_in_full_span_only(pinned_module):
    days = _consecutive_days(date(2019, 12, 29), 6)  # 12/29,30,31,1/1,1/2,1/3
    quote_overrides = {"close": [100.0, None, None, None, 100.0, 100.0]}
    payload = _payload_bytes("TICKD", days, quote_overrides=quote_overrides)
    _, observations = calib.parse_ticker_observations("TICKD", payload, pinned_module)
    full_span = calib.compute_full_span_stats(observations)
    assert full_span.max_consecutive_invalid_returned_rows == 3

    yearly = calib.compute_yearly_window_stats(observations)
    assert yearly[2019].max_consecutive_invalid_returned_rows == 2
    assert yearly[2020].max_consecutive_invalid_returned_rows == 1


# ---------------------------------------------------------------------------
# M-P. Global envelope and candidate defensibility
# ---------------------------------------------------------------------------


def _stats(total, invalid, max_run):
    return calib.WindowStats(
        total_returned=total,
        valid_returned=total - invalid,
        invalid_returned=invalid,
        invalid_fraction=Fraction(invalid, total),
        max_consecutive_invalid_returned_rows=max_run,
    )


def test_global_envelope_is_exact_fraction_maximum():
    windows = [_stats(252, 1, 1), _stats(100, 3, 2), _stats(50, 1, 1)]
    envelope = calib.compute_global_envelope(windows)
    assert envelope.m_fraction == Fraction(3, 100)
    assert envelope.m_fraction_source_window_count == 1
    assert envelope.m_consecutive == 2
    assert envelope.m_consecutive_source_window_count == 1


def test_candidate_equal_to_m_fraction_is_not_defensible():
    candidate = next(c for c in calib.CANDIDATES if c.fraction_id == "F1" and c.max_consecutive == 5)
    m_fraction = calib.candidate_fraction_value(candidate)
    assert calib.is_candidate_defensible(candidate, m_fraction, 0) is False


def test_candidate_equal_to_m_consecutive_is_not_defensible():
    candidate = next(c for c in calib.CANDIDATES if c.fraction_id == "F5" and c.max_consecutive == 3)
    assert calib.is_candidate_defensible(candidate, Fraction(0, 1), 3) is False


def test_candidate_strictly_greater_on_both_axes_is_defensible():
    candidate = next(c for c in calib.CANDIDATES if c.fraction_id == "F5" and c.max_consecutive == 5)
    assert calib.is_candidate_defensible(candidate, Fraction(0, 1), 0) is True


# ---------------------------------------------------------------------------
# Q-S. Selection determinism and run-validity separation (Finding 5)
# ---------------------------------------------------------------------------


def test_selection_picks_strictest_defensible_candidate_deterministically():
    selected, defensible = calib.select_policy(calib.VALID_RUN, Fraction(0, 1), 0)
    assert selected == "F1_C1"
    assert defensible[0].id == "F1_C1"
    assert len(defensible) == 30  # every candidate strictly clears a zero envelope


def test_valid_run_with_no_defensible_candidate_reports_no_defensible_policy():
    huge = Fraction(1, 1)
    selected, defensible = calib.select_policy(calib.VALID_RUN, huge, 999)
    assert selected == calib.CALIBRATION_NO_DEFENSIBLE_POLICY
    assert defensible == ()


def test_select_policy_requires_valid_run():
    invalid = calib.run_validity_for_reason("SYNTHETIC_BASE_SELECTION_BLOCKED")
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.select_policy(invalid, Fraction(0, 1), 0)
    assert excinfo.value.reason == "CALIBRATION_SELECTION_REQUIRES_VALID_RUN"


def test_select_policy_requires_non_null_envelope_even_if_run_object_valid():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.select_policy(calib.VALID_RUN, None, None)
    assert excinfo.value.reason == "CALIBRATION_SELECTION_REQUIRES_VALID_RUN"


def test_run_validity_state_invariant_rejects_contradictions():
    with pytest.raises(calib.V8BCalibrationBlocked):
        calib.RunValidity(r0_classifier_pinned=False, failure_reason=None)
    with pytest.raises(calib.V8BCalibrationBlocked):
        calib.RunValidity(failure_reason="SOMETHING")  # all flags true (valid) but has a reason


def test_invalid_run_reasons_all_produce_invalid_run_validity():
    for reason in calib._RUN_INVALID_REASON_FLAGS:
        rv = calib.run_validity_for_reason(reason)
        assert rv.valid is False
        assert rv.failure_reason == reason


def test_manifest_prefixed_reason_falls_back_to_r1():
    rv = calib.run_validity_for_reason("MANIFEST_SOME_NEW_REASON")
    assert rv.valid is False
    assert rv.r1_v5b_preflight is False


def test_invalid_run_artifact_never_reports_no_defensible_policy():
    artifact = calib.build_result_artifact(
        run_validity=calib.run_validity_for_reason("SYNTHETIC_BASE_SELECTION_BLOCKED"),
        selected_policy=calib.NOT_EVALUATED,
        candidate_selection_executed=False,
        candidate_results=[],
        m_fraction=None,
        m_fraction_window_count=0,
        m_consecutive=None,
        m_consecutive_window_count=0,
        synthetic_base_count=0,
        synthetic_scenario_count=0,
        synthetic_candidate_comparison_count=0,
        synthetic_truth_table_mismatch_count=0,
        synthetic_base_metadata=[],
        input_provenance_hashes={"invalid_reason_count": 1},
        error_counts={"invalid_reason_count": 1},
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test",
        **VALID_TIMESTAMPS,
    )
    assert artifact["calibration_run_valid"] is False
    assert artifact["selected_policy"] == calib.NOT_EVALUATED
    assert artifact["candidate_selection_executed"] is False
    assert artifact["selected_policy"] != calib.CALIBRATION_NO_DEFENSIBLE_POLICY


# ---------------------------------------------------------------------------
# T-W. Synthetic base selection (Finding 3: canonicalize + reject collisions)
# ---------------------------------------------------------------------------


def _valid_observations(count: int, start: date = date(2019, 1, 1)) -> tuple[calib.Observation, ...]:
    days = _consecutive_days(start, count)
    return tuple(
        calib.Observation(
            trading_date=day.isoformat(),
            valid=True,
            invalid_reason=None,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            adjclose=100.25,
            volume=1000.0,
        )
        for day in days
    )


def _invalid_observation(day: date) -> calib.Observation:
    return calib.Observation(
        trading_date=day.isoformat(),
        valid=False,
        invalid_reason="NONFINITE_CLOSE",
        open=None,
        high=None,
        low=None,
        close=None,
        adjclose=None,
        volume=None,
    )


def test_earliest_clean_slice_skips_a_leading_invalid_run():
    obs = (_invalid_observation(date(2019, 1, 1)),) + _valid_observations(252, date(2019, 1, 2))
    start = calib.find_earliest_clean_slice(obs, 252)
    assert start == 1


def test_earliest_clean_slice_returns_none_when_too_short():
    obs = _valid_observations(10)
    assert calib.find_earliest_clean_slice(obs, 252) is None


def test_synthetic_base_selection_contributes_at_most_one_slice_per_ticker(pinned_module):
    observations_by_ticker = {f"T{index:02d}": _valid_observations(600) for index in range(20)}
    bases = calib.select_synthetic_bases(observations_by_ticker, pinned_module)
    assert len(bases) == 20
    assert len({b.ticker_sha256 for b in bases}) == 20


def test_more_than_20_qualifying_tickers_takes_first_20_only(pinned_module):
    observations_by_ticker = {f"T{index:02d}": _valid_observations(252) for index in range(25)}
    bases = calib.select_synthetic_bases(observations_by_ticker, pinned_module)
    assert len(bases) == 20
    expected_first_20 = sorted(observations_by_ticker)[:20]
    expected_hashes = {calib.ticker_sha256(t) for t in expected_first_20}
    assert {b.ticker_sha256 for b in bases} == expected_hashes


def test_fewer_than_20_qualifying_tickers_blocks(pinned_module):
    observations_by_ticker = {f"T{index:02d}": _valid_observations(252) for index in range(19)}
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.select_synthetic_bases(observations_by_ticker, pinned_module)
    assert excinfo.value.reason == "SYNTHETIC_BASE_SELECTION_BLOCKED"


def test_select_synthetic_bases_rejects_canonical_alias_collision(pinned_module):
    observations_by_ticker = {
        "7203": _valid_observations(300),
        "7203.T": _valid_observations(300),
    }
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.select_synthetic_bases(observations_by_ticker, pinned_module)
    assert excinfo.value.reason == "CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION"


def test_select_synthetic_bases_canonicalizes_before_sorting(pinned_module):
    # "7203.T" and "7203" would be a collision; use distinct real tickers with
    # a mix of raw forms to prove canonicalization runs before sort/selection.
    observations_by_ticker = {f"{index}.T": _valid_observations(252) for index in range(1000, 1020)}
    bases = calib.select_synthetic_bases(observations_by_ticker, pinned_module)
    assert len(bases) == 20
    expected_hashes = {calib.ticker_sha256(str(t)) for t in range(1000, 1020)}
    assert {b.ticker_sha256 for b in bases} == expected_hashes


# ---------------------------------------------------------------------------
# X-Z, AA-AD. Synthetic corruption / placement / truth table
# ---------------------------------------------------------------------------


def test_all_12_corruption_classes_produce_their_intended_pinned_reason(pinned_module):
    clean_row = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "adjclose": 100.25, "volume": 1000.0}
    assert len(calib.CORRUPTIONS) == 12
    for corruption in calib.CORRUPTIONS:
        row = dict(clean_row)
        row[corruption.field] = corruption.value
        assert pinned_module._row_invalid_reason(row) == corruption.name


def test_placement_formulas_exact_for_k_equals_3():
    n = calib.SYNTHETIC_SEQUENCE_LENGTH
    assert calib.corrupted_indices(3, "ISOLATED_EVENLY_SPACED") == tuple((j + 1) * n // 4 for j in range(3))
    start = (n - 3) // 2
    assert calib.corrupted_indices(3, "CONSECUTIVE_RUN") == (start, start + 1, start + 2)
    assert calib.corrupted_indices(3, "START_RUN") == (0, 1, 2)
    assert calib.corrupted_indices(3, "END_RUN") == (n - 3, n - 2, n - 1)


def test_k_zero_has_no_indices():
    assert calib.corrupted_indices(0, "NONE") == ()


@pytest.mark.parametrize("k", range(1, 7))
def test_isolated_evenly_spaced_max_run_is_always_1(k):
    n = calib.SYNTHETIC_SEQUENCE_LENGTH
    indices = calib.corrupted_indices(k, "ISOLATED_EVENLY_SPACED")
    flags = [False] * n
    for index in indices:
        flags[index] = True
    assert calib.longest_true_run(flags) == 1


def test_synthetic_scenario_count_is_exactly_6000():
    assert sum(1 for _ in calib.iter_synthetic_scenarios()) == 6000
    assert calib.SYNTHETIC_SCENARIO_COUNT == 6000


def test_candidate_scenario_comparison_count_is_exactly_180000():
    assert calib.SYNTHETIC_SCENARIO_COUNT * len(calib.CANDIDATES) == 180000
    assert calib.SYNTHETIC_CANDIDATE_COMPARISON_COUNT == 180000


@pytest.mark.slow
def test_exhaustive_synthetic_truth_table_matches_generic_policy(pinned_module):
    result = calib.run_synthetic_semantics_verification(FABRICATED_BASES, pinned_module)
    assert result.classifier_mismatch is False
    assert result.scenario_count == 6000
    assert result.comparison_count == 180000
    assert result.truth_table_mismatch_count == 0


def test_selection_signature_never_takes_synthetic_inputs():
    params = set(inspect.signature(calib.select_policy).parameters)
    assert params == {"run_validity", "m_fraction", "m_consecutive"}


# ---------------------------------------------------------------------------
# AH. Manifest structural validation + Finding 4 hardening
# ---------------------------------------------------------------------------


def _synthetic_manifest_payloads(count: int) -> list[dict]:
    return [
        {
            "ticker": f"T{index:04d}",
            "relative_path": f"raw/T{index:04d}.json",
            "sha256": hashlib.sha256(str(index).encode("utf-8")).hexdigest(),
            "byte_count": 100 + index,
        }
        for index in range(count)
    ]


def _synthetic_manifest(payloads=None, **overrides) -> dict:
    payloads = payloads if payloads is not None else _synthetic_manifest_payloads(300)
    manifest = {
        "schema_version": 2,
        "complete": True,
        "usable_for_evaluation": True,
        "attempted_ticker_count": 300,
        "success_count": 300,
        "failed_count": 0,
        "ticker_count": 300,
        "failed_tickers": [],
        "circuit_breaker_triggered": False,
        "request_start": "2019-01-01",
        "request_end": "2026-01-31",
        "payloads": payloads,
        "payload_hash_list_sha256": calib._recompute_payload_hash_list_sha256(payloads),
    }
    manifest.update(overrides)
    return manifest


def test_manifest_wrong_attempted_ticker_count_rejected():
    manifest = _synthetic_manifest(attempted_ticker_count=299)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_ATTEMPTED_TICKER_COUNT_MISMATCH"


def test_manifest_nonzero_failed_count_rejected():
    manifest = _synthetic_manifest(failed_count=1)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_FAILED_COUNT_MISMATCH"


def test_manifest_duplicate_ticker_rejected():
    payloads = _synthetic_manifest_payloads(300)
    payloads[1] = dict(payloads[1])
    payloads[1]["ticker"] = payloads[0]["ticker"]
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_DUPLICATE_TICKER"


def test_manifest_duplicate_relative_path_rejected():
    payloads = _synthetic_manifest_payloads(300)
    payloads[1] = dict(payloads[1])
    payloads[1]["relative_path"] = payloads[0]["relative_path"]
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_DUPLICATE_RELATIVE_PATH"


@pytest.mark.parametrize("bad_path", ["raw/../secret.json", "../raw/x.json", "other/x.json", "raw/./x.json"])
def test_manifest_path_traversal_rejected(bad_path):
    payloads = _synthetic_manifest_payloads(300)
    payloads[0] = dict(payloads[0])
    payloads[0]["relative_path"] = bad_path
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_RELATIVE_PATH_INVALID"


def test_manifest_bad_sha256_rejected():
    payloads = _synthetic_manifest_payloads(300)
    payloads[0] = dict(payloads[0])
    payloads[0]["sha256"] = "not-hex-data"
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_SHA256_INVALID"


def test_manifest_non_int_byte_count_rejected():
    payloads = _synthetic_manifest_payloads(300)
    payloads[0] = dict(payloads[0])
    payloads[0]["byte_count"] = 12.5
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_BYTE_COUNT_INVALID"


def test_manifest_negative_byte_count_rejected():
    payloads = _synthetic_manifest_payloads(300)
    payloads[0] = dict(payloads[0])
    payloads[0]["byte_count"] = -1
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_BYTE_COUNT_INVALID"


def test_manifest_wrong_payload_hash_list_rejected():
    manifest = _synthetic_manifest()
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_HASH_LIST_MISMATCH"


def test_manifest_payload_count_mismatch_rejected():
    manifest = _synthetic_manifest(payloads=_synthetic_manifest_payloads(299))
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_COUNT_MISMATCH"


def test_manifest_recompute_rule_matches_original_v5b_formula():
    payloads = _synthetic_manifest_payloads(5)
    expected = hashlib.sha256(
        json.dumps([p["sha256"] for p in payloads], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert calib._recompute_payload_hash_list_sha256(payloads) == expected


# --- Finding 4: exact integer types (reject bool/float equivalents) -------


@pytest.mark.parametrize("bad_value", [2.0, True, False])
def test_manifest_schema_version_rejects_non_exact_int(bad_value):
    manifest = _synthetic_manifest(schema_version=bad_value)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_SCHEMA_VERSION_MISMATCH"


@pytest.mark.parametrize("bad_value", [300.0, True, False])
def test_manifest_attempted_ticker_count_rejects_non_exact_int(bad_value):
    manifest = _synthetic_manifest(attempted_ticker_count=bad_value)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_ATTEMPTED_TICKER_COUNT_MISMATCH"


@pytest.mark.parametrize("bad_value", [100.0, True, False])
def test_manifest_byte_count_rejects_non_exact_int(bad_value):
    payloads = _synthetic_manifest_payloads(300)
    payloads[0] = dict(payloads[0])
    payloads[0]["byte_count"] = bad_value
    manifest = _synthetic_manifest(payloads=payloads)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_BYTE_COUNT_INVALID"


def test_manifest_uppercase_sha_normalized_to_lowercase(monkeypatch):
    payloads = _synthetic_manifest_payloads(300)
    payloads[0] = dict(payloads[0])
    payloads[0]["sha256"] = payloads[0]["sha256"].upper()
    normalized_shas = [p["sha256"].lower() for p in payloads]
    expected_hash_list = hashlib.sha256(
        json.dumps(normalized_shas, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = _synthetic_manifest(payloads=payloads, payload_hash_list_sha256=expected_hash_list.upper())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", expected_hash_list)
    result = calib.validate_v5b_manifest_structure(manifest)
    assert result["payloads"][0]["sha256"] == payloads[0]["sha256"].lower()
    assert result["payload_hash_list_sha256"] == expected_hash_list


# ---------------------------------------------------------------------------
# AI. Public manifest validator has no override
# ---------------------------------------------------------------------------


def test_public_manifest_validator_exposes_no_expected_hash_override():
    params = set(inspect.signature(calib.validate_v5b_manifest_provenance).parameters)
    assert params == {"manifest_bytes"}


def test_public_manifest_validator_rejects_wrong_whole_file_hash():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_provenance(b"{}")
    assert excinfo.value.reason == "MANIFEST_SHA256_MISMATCH"


def test_public_manifest_validator_accepts_monkeypatched_synthetic_manifest(monkeypatch):
    payloads = _synthetic_manifest_payloads(300)
    manifest = _synthetic_manifest(payloads=payloads)
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", manifest["payload_hash_list_sha256"])
    result = calib.validate_v5b_manifest_provenance(manifest_bytes)
    assert len(result["payloads"]) == 300


# ---------------------------------------------------------------------------
# Finding 2: payload binding to the R1-validated manifest
# ---------------------------------------------------------------------------


def _manifest_and_payloads(count: int, *, corrupt=None):
    """Build a small monkeypatch-friendly manifest + matching InMemoryPayloads.

    ``corrupt`` is an optional callable(payloads, manifest_records) that may
    mutate either list in place to construct a specific mismatch scenario.
    """

    tickers = [f"TICK{index:02d}" for index in range(count)]
    days = _consecutive_days(date(2019, 1, 2), 5)
    payload_bytes_by_ticker = {t: _payload_bytes(f"{t}.T", days) for t in tickers}
    manifest_records = [
        {
            "ticker": t,
            "relative_path": f"raw/{t}.json",
            "sha256": hashlib.sha256(payload_bytes_by_ticker[t]).hexdigest(),
            "byte_count": len(payload_bytes_by_ticker[t]),
        }
        for t in tickers
    ]
    supplied = {
        t: calib.InMemoryPayload(relative_path=f"raw/{t}.json", payload_bytes=payload_bytes_by_ticker[t])
        for t in tickers
    }
    if corrupt is not None:
        corrupt(supplied, manifest_records)
    manifest = {"payloads": manifest_records}
    return manifest, supplied


@pytest.fixture
def pinned(pinned_module):
    return pinned_module


def test_bind_payloads_happy_path(pinned):
    manifest, supplied = _manifest_and_payloads(5)
    bound = calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert set(bound) == {f"TICK{index:02d}" for index in range(5)}


def test_bind_payloads_missing_ticker_rejected(pinned):
    manifest, supplied = _manifest_and_payloads(5)
    del supplied["TICK04"]
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH"


def test_bind_payloads_extra_ticker_rejected(pinned):
    manifest, supplied = _manifest_and_payloads(5)
    supplied["EXTRA99"] = calib.InMemoryPayload(relative_path="raw/EXTRA99.json", payload_bytes=b"{}")
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH"


def test_bind_payloads_wrong_relative_path_rejected(pinned):
    manifest, supplied = _manifest_and_payloads(5)
    original = supplied["TICK00"]
    supplied["TICK00"] = calib.InMemoryPayload(relative_path="raw/WRONG.json", payload_bytes=original.payload_bytes)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_PAYLOAD_PATH_MISMATCH"


def test_bind_payloads_wrong_byte_count_rejected(pinned):
    manifest, supplied = _manifest_and_payloads(5)
    manifest["payloads"][0] = dict(manifest["payloads"][0])
    manifest["payloads"][0]["byte_count"] += 1
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_PAYLOAD_BYTE_COUNT_MISMATCH"


def test_bind_payloads_wrong_sha_rejected(pinned):
    manifest, supplied = _manifest_and_payloads(5)
    manifest["payloads"][0] = dict(manifest["payloads"][0])
    manifest["payloads"][0]["sha256"] = "0" * 64
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_PAYLOAD_SHA256_MISMATCH"


def test_bind_payloads_rejects_manifest_side_canonical_alias_collision(pinned):
    manifest, supplied = _manifest_and_payloads(2)
    manifest["payloads"][1] = dict(manifest["payloads"][1])
    manifest["payloads"][1]["ticker"] = manifest["payloads"][0]["ticker"] + ".T"  # collides after canonicalization
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION"


def test_bind_payloads_rejects_supplied_side_canonical_alias_collision(pinned):
    manifest, supplied = _manifest_and_payloads(2)
    extra_bytes = list(supplied.values())[0].payload_bytes
    supplied["TICK00.T"] = calib.InMemoryPayload(relative_path="raw/whatever.json", payload_bytes=extra_bytes)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.bind_payloads_to_manifest(manifest, supplied, pinned)
    assert excinfo.value.reason == "CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION"


def test_empty_json_manifest_cannot_produce_a_valid_result():
    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=b"{}",
        ticker_payloads={},
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-empty-manifest",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "MANIFEST_SHA256_MISMATCH"


def test_wrong_manifest_hash_produces_invalid_run():
    manifest_bytes = json.dumps({"schema_version": 2}, sort_keys=True, separators=(",", ":")).encode()
    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads={},
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-wrong-hash",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "MANIFEST_SHA256_MISMATCH"


# ---------------------------------------------------------------------------
# Finding 8: provenance field validation
# ---------------------------------------------------------------------------


def _blocked_artifact_kwargs(**overrides):
    base = dict(
        run_validity=calib.run_validity_for_reason("SYNTHETIC_BASE_SELECTION_BLOCKED"),
        selected_policy=calib.NOT_EVALUATED,
        candidate_selection_executed=False,
        candidate_results=[],
        m_fraction=None,
        m_fraction_window_count=0,
        m_consecutive=None,
        m_consecutive_window_count=0,
        synthetic_base_count=0,
        synthetic_scenario_count=0,
        synthetic_candidate_comparison_count=0,
        synthetic_truth_table_mismatch_count=0,
        synthetic_base_metadata=[],
        input_provenance_hashes={"invalid_reason_count": 1},
        error_counts={"invalid_reason_count": 1},
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test",
        run_started_utc="2026-01-01T00:00:00Z",
        run_completed_or_blocked_utc="2026-01-01T00:00:01Z",
    )
    base.update(overrides)
    return base


def test_provenance_rejects_non_hex_commit():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(implementation_git_commit="not-a-commit"))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_COMMIT_INVALID"


def test_provenance_rejects_uppercase_commit():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(implementation_git_commit="A" * 40))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_COMMIT_INVALID"


def test_provenance_rejects_wrong_length_commit():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(implementation_git_commit="0" * 39))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_COMMIT_INVALID"


def test_provenance_rejects_empty_attempt_id():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(calibration_attempt_id=""))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_ATTEMPT_ID_INVALID"


def test_provenance_rejects_too_long_attempt_id():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(calibration_attempt_id="x" * 129))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_ATTEMPT_ID_INVALID"


def test_provenance_rejects_control_char_attempt_id():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(calibration_attempt_id="bad\nid"))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_ATTEMPT_ID_INVALID"


def test_provenance_rejects_malformed_timestamp():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(run_started_utc="2026-01-01 00:00:00"))
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_TIMESTAMP_INVALID"


def test_provenance_rejects_completed_before_started():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(
            **_blocked_artifact_kwargs(
                run_started_utc="2026-01-01T00:00:05Z",
                run_completed_or_blocked_utc="2026-01-01T00:00:00Z",
            )
        )
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_TIMESTAMP_INVALID"


def test_provenance_accepts_equal_started_and_completed():
    artifact = calib.build_result_artifact(
        **_blocked_artifact_kwargs(
            run_started_utc="2026-01-01T00:00:00Z",
            run_completed_or_blocked_utc="2026-01-01T00:00:00Z",
        )
    )
    assert artifact["run_started_utc"] == artifact["run_completed_or_blocked_utc"]


# ---------------------------------------------------------------------------
# Finding 5: contradictory result-state rejection
# ---------------------------------------------------------------------------


def test_build_result_artifact_rejects_invalid_state_with_nonempty_candidates():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(candidate_results=[{"candidate_id": "F1_C1"}]))
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_build_result_artifact_rejects_invalid_state_that_executed_selection():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**_blocked_artifact_kwargs(candidate_selection_executed=True))
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# Deep semantic validation of VALID artifacts (independent recomputation).
# ---------------------------------------------------------------------------


def _make_windows(count, *, invalid=0, max_run=0, total=252):
    return [
        calib.WindowStats(
            total_returned=total,
            valid_returned=total - invalid,
            invalid_returned=invalid,
            invalid_fraction=Fraction(invalid, total),
            max_consecutive_invalid_returned_rows=max_run,
        )
        for _ in range(count)
    ]


def _make_synthetic_bases(count=20, seed="BASE"):
    row = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "adjclose": 100.25, "volume": 1000.0}
    return tuple(
        calib.CleanBase(
            base_index=index,
            ticker_sha256=hashlib.sha256(f"{seed}{index}\n".encode("utf-8")).hexdigest(),
            window_start="2019-01-01",
            window_end="2019-09-09",
            rows=tuple(dict(row) for _ in range(calib.SYNTHETIC_SEQUENCE_LENGTH)),
        )
        for index in range(count)
    )


def _default_synthetic_base_metadata(bases):
    return calib._reference_synthetic_base_metadata(bases)


def _build_synthetic_manifest_bytes(monkeypatch, **overrides):
    payloads = _synthetic_manifest_payloads(300)
    manifest = _synthetic_manifest(payloads=payloads, **overrides)
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", manifest["payload_hash_list_sha256"])
    return manifest_bytes


def _valid_artifact_kwargs(full_span_windows, yearly_windows, *, synthetic_bases, manifest_bytes, **overrides):
    """Build a genuinely self-consistent VALID kwargs set by calling the
    real recompute functions, not by hand-typing matching fake numbers."""

    envelope = calib.compute_global_envelope(list(yearly_windows) + list(full_span_windows))
    candidate_results = calib._compute_candidate_results(
        yearly_windows, full_span_windows, envelope.m_fraction, envelope.m_consecutive
    )
    selected_policy, _ = calib.select_policy(calib.VALID_RUN, envelope.m_fraction, envelope.m_consecutive)
    fraction_headroom, consecutive_headroom = calib._compute_selected_headrooms(
        selected_policy, envelope.m_fraction, envelope.m_consecutive
    )
    validated_manifest = calib.validate_v5b_manifest_provenance(manifest_bytes)
    error_counts = calib._expected_error_counts_from_manifest(validated_manifest)
    kwargs = dict(
        run_validity=calib.VALID_RUN,
        selected_policy=selected_policy,
        candidate_selection_executed=True,
        candidate_results=candidate_results,
        yearly_windows=yearly_windows,
        full_span_windows=full_span_windows,
        synthetic_bases=synthetic_bases,
        manifest_bytes=manifest_bytes,
        m_fraction=envelope.m_fraction,
        m_fraction_window_count=envelope.m_fraction_source_window_count,
        m_consecutive=envelope.m_consecutive,
        m_consecutive_window_count=envelope.m_consecutive_source_window_count,
        synthetic_base_count=calib.SYNTHETIC_BASE_COUNT,
        synthetic_scenario_count=calib.SYNTHETIC_SCENARIO_COUNT,
        synthetic_candidate_comparison_count=calib.SYNTHETIC_CANDIDATE_COMPARISON_COUNT,
        synthetic_truth_table_mismatch_count=0,
        synthetic_base_metadata=_default_synthetic_base_metadata(synthetic_bases),
        input_provenance_hashes={
            "manifest_sha256": calib.EXPECTED_V5B_MANIFEST_SHA256,
            "payload_hash_list_sha256": calib.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
            "manifest_payload_count": 300,
            "bound_payload_count": 300,
        },
        error_counts=error_counts,
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test",
        run_started_utc="2026-01-01T00:00:00Z",
        run_completed_or_blocked_utc="2026-01-01T00:00:01Z",
        selected_candidate_fraction_headroom_exact_or_null=fraction_headroom,
        selected_candidate_consecutive_headroom_or_null=consecutive_headroom,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def clean_valid_kwargs(monkeypatch):
    full_span = _make_windows(300, invalid=0, max_run=0)
    yearly = _make_windows(300, invalid=0, max_run=0)
    bases = _make_synthetic_bases()
    manifest_bytes = _build_synthetic_manifest_bytes(monkeypatch)
    return _valid_artifact_kwargs(full_span, yearly, synthetic_bases=bases, manifest_bytes=manifest_bytes)


def test_build_result_artifact_accepts_genuinely_consistent_valid_state(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    assert artifact["calibration_run_valid"] is True
    assert artifact["selected_policy"] == "F1_C1"
    assert len(artifact["candidate_results"]) == 30
    assert calib.verify_artifact_self_hash(artifact) is True


def test_build_result_artifact_accepts_no_defensible_policy_state(monkeypatch):
    full_span = _make_windows(300, invalid=252, total=252, max_run=252)  # worst possible envelope
    yearly = _make_windows(300, invalid=252, total=252, max_run=252)
    bases = _make_synthetic_bases()
    manifest_bytes = _build_synthetic_manifest_bytes(monkeypatch)
    kwargs = _valid_artifact_kwargs(full_span, yearly, synthetic_bases=bases, manifest_bytes=manifest_bytes)
    artifact = calib.build_result_artifact(**kwargs)
    assert artifact["selected_policy"] == calib.CALIBRATION_NO_DEFENSIBLE_POLICY
    assert artifact["calibration_run_valid"] is True
    assert all(row["DEFENSIBLE"] is False for row in artifact["candidate_results"])


def test_build_result_artifact_rejects_wrong_full_span_window_count(clean_valid_kwargs):
    clean_valid_kwargs["full_span_windows"] = _make_windows(299)  # must be exactly 300
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_build_result_artifact_rejects_empty_yearly_windows(clean_valid_kwargs):
    clean_valid_kwargs["yearly_windows"] = []
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- Point 10: adversarial candidate-result mutations -----------------


def _mutate_row(kwargs, index, **row_overrides):
    rows = [dict(row) for row in kwargs["candidate_results"]]
    rows[index] = {**rows[index], **row_overrides}
    kwargs["candidate_results"] = rows
    return kwargs


def test_rejects_duplicate_candidate_id(clean_valid_kwargs):
    rows = [dict(row) for row in clean_valid_kwargs["candidate_results"]]
    rows[1] = dict(rows[0])  # duplicate row 0's full content into slot 1
    clean_valid_kwargs["candidate_results"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_missing_candidate_row(clean_valid_kwargs):
    rows = list(clean_valid_kwargs["candidate_results"])[:-1]
    clean_valid_kwargs["candidate_results"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_unknown_candidate_id(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, candidate_id="BOGUS_CANDIDATE")
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_declared_fraction(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, declared_fraction={"declared_numerator": 99, "declared_denominator": 252})
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_max_consecutive(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, max_consecutive=999)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_ticker_year_pass_count(clean_valid_kwargs):
    _mutate_row(
        clean_valid_kwargs, 0, observed_ticker_year_pass_count_over_denominator={"pass_count": 0, "denominator": 300}
    )
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_full_ticker_pass_denominator(clean_valid_kwargs):
    _mutate_row(
        clean_valid_kwargs, 0, observed_full_ticker_pass_count_over_denominator={"pass_count": 300, "denominator": 1}
    )
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_defensible_flag(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, DEFENSIBLE=not clean_valid_kwargs["candidate_results"][0]["DEFENSIBLE"])
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_failed_criterion_ids(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, failed_criterion_ids=["D1", "D2"])
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_fraction_headroom(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, fraction_headroom_exact={"numerator": 999, "denominator": 1})
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_consecutive_headroom(clean_valid_kwargs):
    _mutate_row(clean_valid_kwargs, 0, consecutive_headroom=999)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- Point 10: adversarial M/envelope mutations ------------------------


def test_rejects_wrong_m_fraction(clean_valid_kwargs):
    clean_valid_kwargs["m_fraction"] = Fraction(1, 2)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_m_fraction_source_window_count(clean_valid_kwargs):
    clean_valid_kwargs["m_fraction_window_count"] += 1
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_m_consecutive(clean_valid_kwargs):
    clean_valid_kwargs["m_consecutive"] = 5
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_m_consecutive_source_window_count(clean_valid_kwargs):
    clean_valid_kwargs["m_consecutive_window_count"] += 1
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_non_strictest_selected_candidate(clean_valid_kwargs):
    # Every candidate is defensible for a zero envelope, so F1_C1 is the
    # unique correct (strictest) answer; anything else must be rejected.
    clean_valid_kwargs["selected_policy"] = "F2_C1"
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_headroom_present_for_no_defensible_policy(monkeypatch):
    full_span = _make_windows(300, invalid=252, total=252, max_run=252)
    yearly = _make_windows(300, invalid=252, total=252, max_run=252)
    bases = _make_synthetic_bases()
    manifest_bytes = _build_synthetic_manifest_bytes(monkeypatch)
    kwargs = _valid_artifact_kwargs(full_span, yearly, synthetic_bases=bases, manifest_bytes=manifest_bytes)
    assert kwargs["selected_policy"] == calib.CALIBRATION_NO_DEFENSIBLE_POLICY
    kwargs["selected_candidate_fraction_headroom_exact_or_null"] = {"numerator": 1, "denominator": 1}
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_selected_headroom(clean_valid_kwargs):
    clean_valid_kwargs["selected_candidate_consecutive_headroom_or_null"] = 999
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_float_equivalent_consecutive_headroom_at_construction(clean_valid_kwargs):
    # True numeric equality (1 == 1.0) must not be enough at construction
    # time either -- the headroom comparison must be exact-type-aware.
    clean_valid_kwargs["selected_candidate_consecutive_headroom_or_null"] = float(
        clean_valid_kwargs["selected_candidate_consecutive_headroom_or_null"]
    )
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_float_numerator_in_fraction_headroom_at_construction(clean_valid_kwargs):
    headroom = dict(clean_valid_kwargs["selected_candidate_fraction_headroom_exact_or_null"])
    headroom["numerator"] = float(headroom["numerator"])
    clean_valid_kwargs["selected_candidate_fraction_headroom_exact_or_null"] = headroom
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- Point 10: adversarial synthetic-count mutations --------------------


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("synthetic_base_count", 0),
        ("synthetic_base_count", 19),
        ("synthetic_scenario_count", 0),
        ("synthetic_scenario_count", 5999),
        ("synthetic_candidate_comparison_count", 0),
        ("synthetic_candidate_comparison_count", 179999),
        ("synthetic_truth_table_mismatch_count", 1),
    ],
)
def test_rejects_wrong_synthetic_counts(clean_valid_kwargs, field, bad_value):
    clean_valid_kwargs[field] = bad_value
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- Point 10: malformed/duplicate synthetic base metadata --------------


def test_rejects_synthetic_metadata_wrong_row_count(clean_valid_kwargs):
    clean_valid_kwargs["synthetic_base_metadata"] = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])[:19]
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_duplicate_base_index(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[1] = dict(rows[1], base_index=0)  # duplicate of row 0's index; 19 unique instead of 20
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_base_index_out_of_range(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[0] = dict(rows[0], base_index=20)
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_uppercase_ticker_hash(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[0] = dict(rows[0], ticker_sha256="0" * 63 + "A")  # uppercase hex char must be rejected
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_malformed_date(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[0] = dict(rows[0], window_start="not-a-date")
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_start_after_end(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[0] = dict(rows[0], window_start="2019-12-01", window_end="2019-01-01")
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_date_outside_calibration_span(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[0] = dict(rows[0], window_start="2018-12-31", window_end="2019-01-05")
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_synthetic_metadata_extra_field(clean_valid_kwargs):
    rows = _default_synthetic_base_metadata(clean_valid_kwargs["synthetic_bases"])
    rows[0] = dict(rows[0], extra_field="unexpected")
    clean_valid_kwargs["synthetic_base_metadata"] = rows
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- Point 10: empty/wrong provenance ------------------------------------


def test_rejects_empty_provenance_on_valid_artifact(clean_valid_kwargs):
    clean_valid_kwargs["input_provenance_hashes"] = {"invalid_reason_count": 1}
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_manifest_sha_in_provenance(clean_valid_kwargs):
    provenance = dict(clean_valid_kwargs["input_provenance_hashes"])
    provenance["manifest_sha256"] = "0" * 64
    clean_valid_kwargs["input_provenance_hashes"] = provenance
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_bound_payload_count_in_provenance(clean_valid_kwargs):
    provenance = dict(clean_valid_kwargs["input_provenance_hashes"])
    provenance["bound_payload_count"] = 299
    clean_valid_kwargs["input_provenance_hashes"] = provenance
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_float_payload_count_in_provenance(clean_valid_kwargs):
    provenance = dict(clean_valid_kwargs["input_provenance_hashes"])
    provenance["bound_payload_count"] = 300.0
    clean_valid_kwargs["input_provenance_hashes"] = provenance
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- Point 10: wrong/error-count types/values ----------------------------


def test_rejects_nonzero_failed_count(clean_valid_kwargs):
    errors = dict(clean_valid_kwargs["error_counts"])
    errors["failed_count"] = 1
    clean_valid_kwargs["error_counts"] = errors
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_negative_error_count(clean_valid_kwargs):
    errors = dict(clean_valid_kwargs["error_counts"])
    errors["retry_count"] = -1
    clean_valid_kwargs["error_counts"] = errors
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_float_error_count(clean_valid_kwargs):
    errors = dict(clean_valid_kwargs["error_counts"])
    errors["http_429_count"] = 1.0
    clean_valid_kwargs["error_counts"] = errors
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_missing_error_count_key(clean_valid_kwargs):
    errors = dict(clean_valid_kwargs["error_counts"])
    del errors["http_5xx_count"]
    clean_valid_kwargs["error_counts"] = errors
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_extra_error_count_key(clean_valid_kwargs):
    errors = dict(clean_valid_kwargs["error_counts"])
    errors["unexpected"] = 0
    clean_valid_kwargs["error_counts"] = errors
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# Point 1: circular candidate validation removed.
#
# _reference_candidate_row / _verify_candidate_rows_independently must never
# call _compute_candidate_results, _window_passes_candidate,
# is_candidate_defensible, or _failed_criterion_ids. The oracle test below
# is hand-computed, not derived from _compute_candidate_results, so it is a
# genuine independent check rather than the implementation agreeing with
# itself.
# ---------------------------------------------------------------------------


def test_reference_functions_never_call_forbidden_helpers():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration.py").read_text(encoding="utf-8")
    start = source.index("def _reference_window_pass(")
    end = source.index("def _validate_synthetic_base_metadata(")
    reference_block = source[start:end]
    forbidden = ["_compute_candidate_results(", "_window_passes_candidate(", "is_candidate_defensible(", "_failed_criterion_ids("]
    for token in forbidden:
        assert token not in reference_block, f"forbidden call found in reference block: {token}"


def test_reference_candidate_row_matches_hand_computed_oracle():
    # Hand-computed oracle for F1_C1 (1/252, max_consecutive=1) against a
    # fully worked-out scenario. Computed by hand -- not via
    # _compute_candidate_results -- to avoid testing the implementation
    # against itself.
    candidate = next(c for c in calib.CANDIDATES if c.id == "F1_C1")
    yearly = [
        calib.WindowStats(
            total_returned=252, valid_returned=251, invalid_returned=1,
            invalid_fraction=Fraction(1, 252), max_consecutive_invalid_returned_rows=1,
        )
    ]  # exactly at F1's boundary on both axes -> passes
    full_span = [
        calib.WindowStats(
            total_returned=252, valid_returned=250, invalid_returned=2,
            invalid_fraction=Fraction(2, 252), max_consecutive_invalid_returned_rows=2,
        )
    ]  # exceeds F1 on both axes -> fails
    m_fraction = Fraction(2, 252)
    m_consecutive = 2

    row = calib._reference_candidate_row(candidate, yearly, full_span, m_fraction, m_consecutive)

    assert row == {
        "candidate_id": "F1_C1",
        "exact_fraction_rational": {"numerator": 1, "denominator": 252},
        "declared_fraction": {"declared_numerator": 1, "declared_denominator": 252},
        "max_consecutive": 1,
        "observed_ticker_year_pass_count_over_denominator": {"pass_count": 1, "denominator": 1},
        "observed_full_ticker_pass_count_over_denominator": {"pass_count": 0, "denominator": 1},
        "DEFENSIBLE": False,
        "failed_criterion_ids": ["D1", "D2"],
        "fraction_headroom_exact": {"numerator": -1, "denominator": 252},
        "consecutive_headroom": -1,
    }


def test_semantic_validation_rejects_scientifically_wrong_row_from_compromised_constructor(monkeypatch, clean_valid_kwargs):
    """Even if _compute_candidate_results() itself were buggy/compromised
    and produced an internally well-formed but scientifically wrong row, a
    caller who (mistakenly) used it to build candidate_results must still
    be rejected by build_result_artifact()'s independent semantic check."""

    original = calib._compute_candidate_results

    def compromised(yearly_windows, full_span_windows, m_fraction, m_consecutive):
        rows = [dict(row) for row in original(yearly_windows, full_span_windows, m_fraction, m_consecutive)]
        rows[0] = dict(rows[0], observed_ticker_year_pass_count_over_denominator={"pass_count": 999, "denominator": 300})
        return rows

    monkeypatch.setattr(calib, "_compute_candidate_results", compromised)

    envelope = calib.compute_global_envelope(
        list(clean_valid_kwargs["yearly_windows"]) + list(clean_valid_kwargs["full_span_windows"])
    )
    clean_valid_kwargs["candidate_results"] = calib._compute_candidate_results(  # picks up the monkeypatch
        clean_valid_kwargs["yearly_windows"], clean_valid_kwargs["full_span_windows"], envelope.m_fraction, envelope.m_consecutive
    )
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# Point 2: synthetic metadata bound to actual bases (structural validity
# alone is insufficient -- it must match the actual CleanBase objects used).
# ---------------------------------------------------------------------------


def test_rejects_synthetic_metadata_structurally_valid_but_not_matching_actual_bases(clean_valid_kwargs):
    other_bases = _make_synthetic_bases(seed="ROGUE")
    clean_valid_kwargs["synthetic_base_metadata"] = calib._reference_synthetic_base_metadata(other_bases)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_wrong_synthetic_bases_count(clean_valid_kwargs):
    clean_valid_kwargs["synthetic_bases"] = _make_synthetic_bases(count=19)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**clean_valid_kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# Point 3: error_counts bound to the exact validated manifest.
# ---------------------------------------------------------------------------


def test_rejects_error_counts_not_matching_validated_manifest(monkeypatch):
    full_span = _make_windows(300, invalid=0, max_run=0)
    yearly = _make_windows(300, invalid=0, max_run=0)
    bases = _make_synthetic_bases()
    manifest_bytes = _build_synthetic_manifest_bytes(monkeypatch, retry_count=5)
    kwargs = _valid_artifact_kwargs(full_span, yearly, synthetic_bases=bases, manifest_bytes=manifest_bytes)
    assert kwargs["error_counts"]["retry_count"] == 5  # sanity: derived correctly from the real manifest
    wrong_errors = dict(kwargs["error_counts"])
    wrong_errors["retry_count"] = 0  # caller claims something other than the manifest says
    kwargs["error_counts"] = wrong_errors
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.build_result_artifact(**kwargs)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_valid_artifact_with_wrong_manifest_bytes(clean_valid_kwargs):
    clean_valid_kwargs["manifest_bytes"] = b"{}"
    with pytest.raises(calib.V8BCalibrationBlocked):
        calib.build_result_artifact(**clean_valid_kwargs)


@pytest.mark.parametrize("field", ["retry_count", "http_429_count", "http_5xx_count"])
def test_manifest_structural_validation_rejects_negative_optional_counter(field):
    manifest = _synthetic_manifest(**{field: -1})
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == f"MANIFEST_{field.upper()}_INVALID"


@pytest.mark.parametrize("field", ["retry_count", "http_429_count", "http_5xx_count"])
def test_manifest_structural_validation_rejects_float_optional_counter(field):
    manifest = _synthetic_manifest(**{field: 1.0})
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == f"MANIFEST_{field.upper()}_INVALID"


def test_manifest_structural_validation_accepts_valid_optional_counters(monkeypatch):
    payloads = _synthetic_manifest_payloads(300)
    manifest = _synthetic_manifest(payloads=payloads, retry_count=2, http_429_count=0, http_5xx_count=1)
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", manifest["payload_hash_list_sha256"])
    result = calib.validate_v5b_manifest_structure(manifest)
    assert result["retry_count"] == 2
    assert result["http_429_count"] == 0
    assert result["http_5xx_count"] == 1


# ---------------------------------------------------------------------------
# Point 4: public full persisted-artifact semantic verifier.
# ---------------------------------------------------------------------------


def test_validate_result_artifact_semantics_accepts_genuinely_valid_artifact(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    calib.validate_result_artifact_semantics(
        artifact,
        yearly_windows=clean_valid_kwargs["yearly_windows"],
        full_span_windows=clean_valid_kwargs["full_span_windows"],
        synthetic_bases=clean_valid_kwargs["synthetic_bases"],
        manifest_bytes=clean_valid_kwargs["manifest_bytes"],
    )  # must not raise


def test_validate_result_artifact_semantics_rejects_genuinely_invalid_artifact_without_trusted_reason():
    # A genuinely self-consistent INVALID artifact must still be rejected
    # when the caller supplies no trusted expected_invalid_reason: the
    # persisted artifact is never its own authority for which blocker fired.
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(artifact)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_validate_result_artifact_semantics_accepts_genuinely_invalid_artifact_with_trusted_reason():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    calib.validate_result_artifact_semantics(
        artifact, expected_invalid_reason="SYNTHETIC_BASE_SELECTION_BLOCKED"
    )  # must not raise


def test_validate_result_artifact_semantics_rejects_bad_self_hash(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    tampered = dict(artifact)
    tampered["artifact_self_hash"] = "0" * 64
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            tampered,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_ARTIFACT_SELF_HASH_MISMATCH"


def test_validate_result_artifact_semantics_is_not_implemented_via_build_result_artifact():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration.py").read_text(encoding="utf-8")
    start = source.index("def validate_result_artifact_semantics(")
    end = source.index("\n# ---", start)
    body = source[start:end]
    assert "build_result_artifact(" not in body


# ---------------------------------------------------------------------------
# Invalid-reason binding: run_invalid_reason_or_null must be a recognized
# calibration blocker, and an optional trusted expected_invalid_reason must
# match exactly (and itself be recognized). An untrusted artifact field
# alone never dictates a retry/governance meaning.
# ---------------------------------------------------------------------------


def test_is_recognized_invalid_reason_rejects_bool_int_float_and_unknown_strings():
    assert calib._is_recognized_invalid_reason(True) is False
    assert calib._is_recognized_invalid_reason(1) is False
    assert calib._is_recognized_invalid_reason(1.0) is False
    assert calib._is_recognized_invalid_reason("") is False
    assert calib._is_recognized_invalid_reason("SOMETHING_MADE_UP") is False
    assert calib._is_recognized_invalid_reason(None) is False


def test_is_recognized_invalid_reason_accepts_known_reasons_and_exact_manifest_blockers():
    for known_reason in calib._RUN_INVALID_REASON_FLAGS:
        assert calib._is_recognized_invalid_reason(known_reason) is True
    for manifest_reason in calib._RECOGNIZED_MANIFEST_BLOCKER_REASONS:
        assert calib._is_recognized_invalid_reason(manifest_reason) is True


def test_is_recognized_invalid_reason_rejects_arbitrary_manifest_prefixed_strings():
    # The old wildcard rule (reason.startswith("MANIFEST_")) is gone: only
    # the exact, finite set of manifest blocker strings current production
    # validation code can actually raise is recognized.
    assert calib._is_recognized_invalid_reason("MANIFEST_ANYTHING_STRUCTURAL") is False
    assert calib._is_recognized_invalid_reason("MANIFEST_FAKE_REASON") is False
    assert calib._is_recognized_invalid_reason("MANIFEST_") is False


@pytest.mark.parametrize("manifest_reason", sorted(calib._RECOGNIZED_MANIFEST_BLOCKER_REASONS))
def test_validate_result_artifact_semantics_accepts_every_real_manifest_blocker_with_trusted_reason(manifest_reason):
    # Every real manifest blocker reason emitted by current production
    # validation must be recognized and, when supplied as a trusted
    # expected_invalid_reason matching the persisted reason, accepted.
    artifact = calib.build_result_artifact(
        **_blocked_artifact_kwargs(run_validity=calib.run_validity_for_reason(manifest_reason))
    )
    calib.validate_result_artifact_semantics(
        artifact, expected_invalid_reason=manifest_reason
    )  # must not raise


def test_validate_result_artifact_semantics_rejects_recognized_invalid_reason_without_trusted_reason():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(artifact)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_validate_result_artifact_semantics_rejects_arbitrary_manifest_fake_reason_as_expected():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(artifact, expected_invalid_reason="MANIFEST_FAKE_REASON")
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_validate_result_artifact_semantics_rejects_rehashed_arbitrary_manifest_fake_persisted_reason():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["run_invalid_reason_or_null"] = "MANIFEST_FAKE_REASON"
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated, expected_invalid_reason="MANIFEST_FAKE_REASON")
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_validate_result_artifact_semantics_accepts_matching_expected_invalid_reason():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    calib.validate_result_artifact_semantics(
        artifact, expected_invalid_reason="SYNTHETIC_BASE_SELECTION_BLOCKED"
    )  # must not raise


def test_validate_result_artifact_semantics_rejects_mismatched_expected_invalid_reason():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            artifact, expected_invalid_reason="CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED"
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_validate_result_artifact_semantics_rejects_unrecognized_expected_invalid_reason():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(artifact, expected_invalid_reason="TOTALLY_MADE_UP_REASON")
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_validate_result_artifact_semantics_rejects_non_null_expected_reason_on_valid_artifact(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            artifact,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
            expected_invalid_reason="SYNTHETIC_BASE_SELECTION_BLOCKED",
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


@pytest.mark.parametrize("bad_reason", [1, True, 1.0])
def test_rejects_rehashed_invalid_reason_as_non_str(bad_reason):
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["run_invalid_reason_or_null"] = bad_reason
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_rehashed_invalid_reason_arbitrary_unknown_string():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["run_invalid_reason_or_null"] = "NOT_A_REAL_BLOCKER_CODE"
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_rehashed_recognized_to_recognized_invalid_reason_without_trusted_reason():
    # Attack: persisted reason swapped from one recognized blocker to
    # another recognized blocker, self-hash recomputed over the mutation,
    # and no trusted expected_invalid_reason supplied at all.
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["run_invalid_reason_or_null"] = "CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED"  # recognized, but not what happened
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_rehashed_recognized_but_wrong_invalid_reason_when_expected_supplied():
    # Same attack, but this time the caller does supply a trusted reason --
    # the original, correct one. It must still be rejected because it no
    # longer matches the (attacker-mutated) persisted reason.
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["run_invalid_reason_or_null"] = "CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED"  # recognized, but not what happened
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated, expected_invalid_reason="SYNTHETIC_BASE_SELECTION_BLOCKED")
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# Exact typing for selected headroom in persisted semantic validation
# (float numerator/denominator, bool-as-int consecutive headroom).
# ---------------------------------------------------------------------------


def test_rejects_selected_fraction_headroom_float_numerator_persisted(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    headroom = dict(mutated["selected_candidate_fraction_headroom_exact_or_null"])
    headroom["numerator"] = float(headroom["numerator"])
    mutated["selected_candidate_fraction_headroom_exact_or_null"] = headroom
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_selected_fraction_headroom_float_denominator_persisted(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    headroom = dict(mutated["selected_candidate_fraction_headroom_exact_or_null"])
    headroom["denominator"] = float(headroom["denominator"])
    mutated["selected_candidate_fraction_headroom_exact_or_null"] = headroom
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_selected_consecutive_headroom_bool_instead_of_int_persisted(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    assert mutated["selected_candidate_consecutive_headroom_or_null"] == 1  # F1_C1 vs a zero envelope
    mutated["selected_candidate_consecutive_headroom_or_null"] = True  # True == 1 under plain ==
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_selected_consecutive_headroom_float_instead_of_int_persisted(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["selected_candidate_consecutive_headroom_or_null"] = float(
        mutated["selected_candidate_consecutive_headroom_or_null"]
    )
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# Complete persisted-artifact metadata validation: a single common validator
# runs BEFORE valid/invalid branching, identically for both artifact kinds,
# rejecting missing/unknown keys, mutated fixed identifiers, mutated
# calibration window/years, wrong candidate_count, mutated synthetic
# rule/formula version strings, malformed self-hash format, and malformed
# provenance -- all via a rehashed artifact where integrity alone passes.
# ---------------------------------------------------------------------------


_COMMON_METADATA_MUTATIONS = [
    ("schema_version", "NOT_" + calib.RESULT_SCHEMA_VERSION),
    ("study", "NOT_" + calib.STUDY),
    ("calibration_plan_version", "NOT_" + calib.PLAN_VERSION),
    ("calibration_plan_commit_or_hash", "0" * 40),
    ("approved_plan_commit", "0" * 40),
    ("approved_plan_blob_sha", "0" * 40),
    ("approval_artifact_blob_sha", "0" * 40),
    ("calibration_start", "2019-01-02"),
    ("calibration_end_exclusive", "2026-01-02"),
    ("candidate_count", 29),
    ("candidate_count", 30.0),
    ("synthetic_base_selection_rule", "NOT_" + calib.SYNTHETIC_BASE_SELECTION_RULE_VERSION),
    ("exact_synthetic_placement_formulas_version", "NOT_" + calib.SYNTHETIC_PLACEMENT_FORMULAS_VERSION),
    ("implementation_git_commit", "not-a-commit"),
    ("calibration_attempt_id", ""),
    ("run_started_utc", "not-a-timestamp"),
]

_ACCEPTABLE_COMMON_METADATA_REASONS = (
    "CALIBRATION_RESULT_STATE_INVALID",
    "CALIBRATION_PROVENANCE_COMMIT_INVALID",
    "CALIBRATION_PROVENANCE_ATTEMPT_ID_INVALID",
    "CALIBRATION_PROVENANCE_TIMESTAMP_INVALID",
)


@pytest.mark.parametrize("field,bad_value", _COMMON_METADATA_MUTATIONS)
def test_rehashed_valid_artifact_with_mutated_common_metadata_is_rejected(clean_valid_kwargs, field, bad_value):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated[field] = bad_value
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True  # integrity alone would pass
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason in _ACCEPTABLE_COMMON_METADATA_REASONS


@pytest.mark.parametrize("field,bad_value", _COMMON_METADATA_MUTATIONS)
def test_rehashed_invalid_artifact_with_mutated_common_metadata_is_rejected(field, bad_value):
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated[field] = bad_value
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated)
    assert excinfo.value.reason in _ACCEPTABLE_COMMON_METADATA_REASONS


def test_rejects_mutated_calibration_years_wrong_list(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["calibration_years"] = [2019, 2020, 2021, 2022, 2023, 2024]  # missing 2025
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_mutated_calibration_years_float_element(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    years = list(mutated["calibration_years"])
    years[0] = float(years[0])
    mutated["calibration_years"] = years
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_missing_top_level_key(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = {key: value for key, value in artifact.items() if key != "study"}
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_extra_top_level_key(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["unexpected_extra_field"] = "surprise"
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_missing_top_level_key_on_invalid_artifact():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = {key: value for key, value in artifact.items() if key != "study"}
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_common_metadata_validator_rejects_uppercase_self_hash_format(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["artifact_self_hash"] = mutated["artifact_self_hash"].upper()
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib._verify_common_persisted_artifact_metadata(mutated)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_completed_before_started_on_persisted_artifact(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["run_started_utc"] = "2026-01-01T00:00:05Z"
    mutated["run_completed_or_blocked_utc"] = "2026-01-01T00:00:00Z"
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_PROVENANCE_TIMESTAMP_INVALID"


def test_rejects_calibration_run_valid_as_int_not_bool():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["calibration_run_valid"] = 0  # int, not bool
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated)
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_candidate_selection_executed_as_int_not_bool():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["candidate_selection_executed"] = 0
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated, expected_invalid_reason="SYNTHETIC_BASE_SELECTION_BLOCKED")
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- exact JSON numeric typing on VALID artifacts (float/bool substitutes) -


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("M_fraction_source_window_count", 300.0),
        ("M_consecutive_source_window_count", True),
        ("synthetic_base_count", 20.0),
        ("synthetic_base_ticker_count", 20.0),
        ("synthetic_scenario_count", 6000.0),
        ("synthetic_candidate_comparison_count", 180000.0),
        ("full_expected_vs_observed_synthetic_truth_table_mismatch_count", False),
    ],
)
def test_rejects_valid_artifact_exact_int_field_as_float_or_bool(clean_valid_kwargs, field, bad_value):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated[field] = bad_value
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("M_fraction_source_window_count", 0.0),
        ("M_consecutive_source_window_count", False),
        ("synthetic_base_count", 0.0),
        ("synthetic_base_ticker_count", False),
        ("synthetic_scenario_count", 0.0),
        ("synthetic_candidate_comparison_count", False),
        ("full_expected_vs_observed_synthetic_truth_table_mismatch_count", 0.0),
    ],
)
def test_rejects_invalid_artifact_exact_int_field_as_float_or_bool(field, bad_value):
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated[field] = bad_value
    mutated = _rehash(mutated)
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(mutated, expected_invalid_reason="SYNTHETIC_BASE_SELECTION_BLOCKED")
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# --- candidate-row numeric fields: float/bool substitutes must not pass ---
# naive dict equality (1.0 == 1, True == 1 in Python).


def test_rejects_candidate_row_float_equivalent_max_consecutive(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["candidate_results"] = [dict(row) for row in mutated["candidate_results"]]
    mutated["candidate_results"][0] = dict(mutated["candidate_results"][0], max_consecutive=1.0)
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_rejects_candidate_row_defensible_as_int_not_bool(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["candidate_results"] = [dict(row) for row in mutated["candidate_results"]]
    original_flag = mutated["candidate_results"][0]["DEFENSIBLE"]
    mutated["candidate_results"][0] = dict(mutated["candidate_results"][0], DEFENSIBLE=int(original_flag))
    mutated = _rehash(mutated)
    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_strict_equal_rejects_float_int_and_bool_int_substitution():
    assert calib._strict_equal(1, 1.0) is False
    assert calib._strict_equal(True, 1) is False
    assert calib._strict_equal({"x": 1}, {"x": 1.0}) is False
    assert calib._strict_equal({"x": True}, {"x": 1}) is False
    assert calib._strict_equal([1, 2], [1.0, 2]) is False
    assert calib._strict_equal({"x": 1}, {"x": 1}) is True
    assert calib._strict_equal([1, "a", True], [1, "a", True]) is True


def test_canonical_artifact_keys_matches_actual_build_result_artifact_output(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    assert set(artifact.keys()) == calib._CANONICAL_ARTIFACT_KEYS


# ---------------------------------------------------------------------------
# Point 5: self-hash is integrity-only, never acceptance.
# ---------------------------------------------------------------------------


def test_verify_artifact_self_hash_not_in_public_api():
    assert "verify_artifact_self_hash" not in calib.__all__


def test_validate_result_artifact_semantics_in_public_api():
    assert "validate_result_artifact_semantics" in calib.__all__


def _rehash(artifact):
    without_hash = {key: value for key, value in artifact.items() if key != "artifact_self_hash"}
    digest = calib.sha256_hex(calib.canonical_json_bytes(without_hash))
    return {**without_hash, "artifact_self_hash": digest}


def test_self_hash_integrity_insufficient_for_candidate_pass_count_mutation(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["candidate_results"] = [dict(row) for row in mutated["candidate_results"]]
    mutated["candidate_results"][0] = dict(
        mutated["candidate_results"][0],
        observed_ticker_year_pass_count_over_denominator={"pass_count": 999, "denominator": 300},
    )
    mutated = _rehash(mutated)

    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_self_hash_integrity_insufficient_for_synthetic_metadata_mutation(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    other_bases = _make_synthetic_bases(seed="ROGUE")
    mutated["synthetic_base_window_start_and_end_metadata"] = calib._reference_synthetic_base_metadata(other_bases)
    mutated = _rehash(mutated)

    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


def test_self_hash_integrity_insufficient_for_error_counts_mutation(clean_valid_kwargs):
    artifact = calib.build_result_artifact(**clean_valid_kwargs)
    mutated = dict(artifact)
    mutated["error_counts"] = {"failed_count": 0, "retry_count": 999, "http_429_count": 0, "http_5xx_count": 0}
    mutated = _rehash(mutated)

    assert calib.verify_artifact_self_hash(mutated) is True
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_result_artifact_semantics(
            mutated,
            yearly_windows=clean_valid_kwargs["yearly_windows"],
            full_span_windows=clean_valid_kwargs["full_span_windows"],
            synthetic_bases=clean_valid_kwargs["synthetic_bases"],
            manifest_bytes=clean_valid_kwargs["manifest_bytes"],
        )
    assert excinfo.value.reason == "CALIBRATION_RESULT_STATE_INVALID"


# ---------------------------------------------------------------------------
# AE-AG. Result artifact / self-hash
# ---------------------------------------------------------------------------


def test_artifact_self_hash_round_trip():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    assert calib.verify_artifact_self_hash(artifact) is True


def test_artifact_self_hash_detects_mutation():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    mutated = dict(artifact)
    mutated["calibration_attempt_id"] = "mutated"
    assert calib.verify_artifact_self_hash(mutated) is False


def test_all_required_artifact_metadata_present_on_blocked_artifact():
    artifact = calib.build_result_artifact(**_blocked_artifact_kwargs())
    required_keys = {
        "schema_version",
        "calibration_plan_commit_or_hash",
        "input_provenance_hashes",
        "synthetic_base_ticker_count",
        "synthetic_base_selection_rule",
        "synthetic_base_window_start_and_end_metadata",
        "exact_synthetic_placement_formulas_version",
        "full_expected_vs_observed_synthetic_truth_table_mismatch_count",
        "error_counts",
        "mechanically_selected_candidate_or_NO_DEFENSIBLE_POLICY_or_NOT_EVALUATED",
    }
    assert required_keys.issubset(artifact.keys())
    assert artifact["synthetic_base_selection_rule"] == calib.SYNTHETIC_BASE_SELECTION_RULE_VERSION
    assert artifact["exact_synthetic_placement_formulas_version"] == calib.SYNTHETIC_PLACEMENT_FORMULAS_VERSION


# ---------------------------------------------------------------------------
# Full-pipeline integration (happy path + invalid-run paths), no real data.
# Tests may monkeypatch the fixed expected manifest hashes to use synthetic
# manifests; the production validate_v5b_manifest_provenance() itself keeps
# no such override (see test_public_manifest_validator_exposes_no_expected_hash_override).
# ---------------------------------------------------------------------------


def _build_full_manifest_and_payloads(*, empty_ticker_index: int | None = None, short_data_only: bool = False):
    """300 canonical tickers TICK000..TICK299. First 20 (sorted) get a full
    clean 252-day run in 2019 so they qualify as synthetic bases; the rest
    get a short (5-day) valid window, unless short_data_only is set."""

    tickers = [f"TICK{index:03d}" for index in range(300)]
    payload_bytes_by_ticker: dict[str, bytes] = {}
    for index, ticker in enumerate(tickers):
        if empty_ticker_index is not None and index == empty_ticker_index:
            days = [date(2026, 1, 15)]  # inside V5B request window, outside calibration window -> R3
        elif not short_data_only and index < 20:
            days = _consecutive_days(date(2019, 1, 2), calib.SYNTHETIC_SEQUENCE_LENGTH)
        else:
            days = _consecutive_days(date(2019, 1, 2), 5)
        payload_bytes_by_ticker[ticker] = _payload_bytes(f"{ticker}.T", days)

    manifest_records = [
        {
            "ticker": ticker,
            "relative_path": f"raw/{ticker}.json",
            "sha256": hashlib.sha256(payload_bytes_by_ticker[ticker]).hexdigest(),
            "byte_count": len(payload_bytes_by_ticker[ticker]),
        }
        for ticker in tickers
    ]
    hash_list = calib._recompute_payload_hash_list_sha256(manifest_records)
    manifest = {
        "schema_version": 2,
        "complete": True,
        "usable_for_evaluation": True,
        "attempted_ticker_count": 300,
        "success_count": 300,
        "failed_count": 0,
        "ticker_count": 300,
        "failed_tickers": [],
        "circuit_breaker_triggered": False,
        "request_start": "2019-01-01",
        "request_end": "2026-01-31",
        "payloads": manifest_records,
        "payload_hash_list_sha256": hash_list,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    supplied = {
        ticker: calib.InMemoryPayload(relative_path=f"raw/{ticker}.json", payload_bytes=payload_bytes_by_ticker[ticker])
        for ticker in tickers
    }
    return manifest_bytes, hash_list, supplied


@pytest.mark.slow
def test_full_run_happy_path_selects_strictest_candidate_and_is_valid(monkeypatch):
    manifest_bytes, hash_list, supplied = _build_full_manifest_and_payloads()
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", hash_list)

    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=supplied,
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-happy-path",
    )
    assert result["calibration_run_valid"] is True
    assert result["run_invalid_reason_or_null"] is None
    assert result["candidate_selection_executed"] is True
    assert result["selected_policy"] == "F1_C1"
    assert len(result["candidate_results"]) == 30
    for row in result["candidate_results"]:
        for key in (
            "candidate_id",
            "exact_fraction_rational",
            "declared_fraction",
            "max_consecutive",
            "observed_ticker_year_pass_count_over_denominator",
            "observed_full_ticker_pass_count_over_denominator",
            "DEFENSIBLE",
            "failed_criterion_ids",
        ):
            assert key in row
    assert result["synthetic_base_count"] == 20
    assert result["synthetic_scenario_count"] == 6000
    assert result["synthetic_candidate_comparison_count"] == 180000
    assert result["full_expected_vs_observed_synthetic_truth_table_mismatch_count"] == 0
    assert result["input_provenance_hashes"]["bound_payload_count"] == 300
    assert result["input_provenance_hashes"]["manifest_payload_count"] == 300
    assert calib.verify_artifact_self_hash(result) is True
    serialized = json.dumps(result)
    assert "TICK000" not in serialized  # no raw ticker identity leaks into the artifact


@pytest.mark.slow
def test_full_run_blocks_on_empty_full_span(monkeypatch):
    manifest_bytes, hash_list, supplied = _build_full_manifest_and_payloads(empty_ticker_index=250)
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", hash_list)

    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=supplied,
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-empty-span",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED"
    assert result["selected_policy"] == calib.NOT_EVALUATED
    assert result["candidate_selection_executed"] is False


@pytest.mark.slow
def test_full_run_blocks_when_fewer_than_20_qualifying_bases(monkeypatch):
    manifest_bytes, hash_list, supplied = _build_full_manifest_and_payloads(short_data_only=True)
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", hash_list)

    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=supplied,
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-few-bases",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "SYNTHETIC_BASE_SELECTION_BLOCKED"
    assert result["selected_policy"] == calib.NOT_EVALUATED


@pytest.mark.slow
def test_full_run_missing_payload_blocks_at_binding(monkeypatch):
    manifest_bytes, hash_list, supplied = _build_full_manifest_and_payloads()
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", hash_list)
    del supplied["TICK299"]

    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=supplied,
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-missing-payload",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH"


def test_full_run_blocks_on_repository_contract_mismatch(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.PREREGISTRATION_PATH
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 0xFF
    path.write_bytes(bytes(mutated))
    result = calib.run_data_quality_calibration(
        repository_root=root,
        manifest_bytes=b"{}",
        ticker_payloads={},
        implementation_git_commit=VALID_COMMIT,
        calibration_attempt_id="test-contract-mismatch",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "CALIBRATION_PLAN_BLOB_MISMATCH"


# ---------------------------------------------------------------------------
# No network / no real-data-path smoke checks
# ---------------------------------------------------------------------------


def test_module_performs_no_io_on_import():
    import importlib
    import src.v8b_data_quality_calibration as reimported

    importlib.reload(reimported)  # must not raise or touch the filesystem beyond bytecode


def test_module_source_has_no_network_or_real_cache_strings():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration.py").read_text(encoding="utf-8")
    forbidden = [
        "urllib",
        "requests",
        "yfinance",
        "query1.finance.yahoo.com",
        "v5-b-evaluation-cache-retry1",
        "--cache",
        "--input-dir",
        "--execute-real",
        "spec_from_file_location",
        "exec_module",
    ]
    for token in forbidden:
        assert token not in source, f"forbidden token found: {token}"
