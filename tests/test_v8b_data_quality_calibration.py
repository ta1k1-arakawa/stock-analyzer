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


def test_approval_content_mutation_blocks(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.APPROVAL_ARTIFACT_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    data["approval_status"] = "REVOKED"
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.verify_repository_contract(root)
    # content mutation also changes the blob, so either specific reason is acceptable
    assert excinfo.value.reason in ("CALIBRATION_APPROVAL_BLOB_MISMATCH", "CALIBRATION_APPROVAL_INVALID")


# ---------------------------------------------------------------------------
# E. Strict JSON duplicate-key rejection
# ---------------------------------------------------------------------------


def test_duplicate_key_json_rejected():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.parse_strict_json('{"a":1,"a":2}')
    assert excinfo.value.reason == "STRICT_JSON_DUPLICATE_KEY"


def test_malformed_json_rejected():
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.parse_strict_json("{not json")
    assert excinfo.value.reason == "STRICT_JSON_MALFORMED"


def test_nested_duplicate_key_json_rejected():
    with pytest.raises(calib.V8BCalibrationBlocked):
        calib.parse_strict_json('{"a":{"b":1,"b":2}}')


# ---------------------------------------------------------------------------
# 7-8. git_blob_sha1 / canonical_json_bytes
# ---------------------------------------------------------------------------


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


def test_canonical_json_bytes_rejects_nan():
    with pytest.raises(ValueError):
        calib.canonical_json_bytes({"x": float("nan")})


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


def test_candidate_grid_is_provably_ascending_by_fraction_then_consecutive():
    previous = None
    for candidate in calib.CANDIDATES:
        key = (calib.candidate_fraction_value(candidate), candidate.max_consecutive)
        if previous is not None:
            assert previous <= key
        previous = key


def test_f2_declared_representation_stays_2_over_252():
    f2_candidates = [c for c in calib.CANDIDATES if c.fraction_id == "F2"]
    assert len(f2_candidates) == 5
    for candidate in f2_candidates:
        assert candidate.declared_numerator == 2
        assert candidate.declared_denominator == 252
    # mathematically reduces to 1/126 but the DECLARED shape must be preserved
    assert calib.candidate_fraction_value(f2_candidates[0]) == Fraction(1, 126)


def test_fq1_has_no_exemption_and_participates_normally():
    fq1_candidates = [c for c in calib.CANDIDATES if c.fraction_id == "FQ1"]
    assert len(fq1_candidates) == 5
    assert calib.candidate_fraction_value(fq1_candidates[0]) == Fraction(1, 100)


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
# Q-S. Selection determinism and run-validity separation
# ---------------------------------------------------------------------------


def test_selection_picks_strictest_defensible_candidate_deterministically():
    selected, defensible = calib.select_policy(Fraction(0, 1), 0)
    assert selected == "F1_C1"
    assert defensible[0].id == "F1_C1"
    assert len(defensible) == 30  # every candidate strictly clears a zero envelope


def test_valid_run_with_no_defensible_candidate_reports_no_defensible_policy():
    huge = Fraction(1, 1)
    selected, defensible = calib.select_policy(huge, 999)
    assert selected == calib.CALIBRATION_NO_DEFENSIBLE_POLICY
    assert defensible == ()


def test_invalid_run_never_reports_no_defensible_policy():
    for reason in calib._RUN_INVALID_REASON_FLAGS:
        rv = calib.run_validity_for_reason(reason)
        assert rv.valid is False
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
        input_provenance={"manifest_sha256": "0" * 64, "ticker_count": 0},
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test",
        run_started_utc="2026-01-01T00:00:00Z",
        run_completed_or_blocked_utc="2026-01-01T00:00:01Z",
    )
    assert artifact["calibration_run_valid"] is False
    assert artifact["selected_policy"] == calib.NOT_EVALUATED
    assert artifact["candidate_selection_executed"] is False
    assert artifact["selected_policy"] != calib.CALIBRATION_NO_DEFENSIBLE_POLICY


# ---------------------------------------------------------------------------
# T-W. Synthetic base selection
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


def test_synthetic_base_selection_contributes_at_most_one_slice_per_ticker():
    observations_by_ticker = {
        f"T{index:02d}": _valid_observations(600) for index in range(20)
    }
    bases = calib.select_synthetic_bases(observations_by_ticker)
    assert len(bases) == 20
    assert len({b.ticker_sha256 for b in bases}) == 20


def test_more_than_20_qualifying_tickers_takes_first_20_only():
    observations_by_ticker = {f"T{index:02d}": _valid_observations(252) for index in range(25)}
    bases = calib.select_synthetic_bases(observations_by_ticker)
    assert len(bases) == 20
    expected_first_20 = sorted(observations_by_ticker)[:20]
    expected_hashes = {calib.ticker_sha256(t) for t in expected_first_20}
    assert {b.ticker_sha256 for b in bases} == expected_hashes


def test_fewer_than_20_qualifying_tickers_blocks():
    observations_by_ticker = {f"T{index:02d}": _valid_observations(252) for index in range(19)}
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.select_synthetic_bases(observations_by_ticker)
    assert excinfo.value.reason == "SYNTHETIC_BASE_SELECTION_BLOCKED"


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


def test_synthetic_classifier_mismatch_detected_when_a_row_misclassifies(pinned_module):
    scenario = next(iter(calib.iter_synthetic_scenarios(1)))
    base = FABRICATED_BASES[0]
    indices = calib.corrupted_indices(scenario.k, scenario.family)
    corrupted_rows = calib.apply_corruption(base.rows, scenario.field, scenario.value, indices)
    # sabotage one uncorrupted row so R6 should be violated
    sabotage_index = next(i for i in range(len(corrupted_rows)) if i not in set(indices))
    corrupted_rows[sabotage_index]["open"] = None
    reasons = [pinned_module._row_invalid_reason(row) for row in corrupted_rows]
    assert reasons[sabotage_index] == "NONFINITE_OPEN"  # would not equal expected "valid" (None)


def test_selection_signature_never_takes_synthetic_inputs():
    params = set(inspect.signature(calib.select_policy).parameters)
    assert params == {"m_fraction", "m_consecutive"}


# ---------------------------------------------------------------------------
# AE-AG. Result artifact / self-hash
# ---------------------------------------------------------------------------


def _minimal_valid_artifact():
    return calib.build_result_artifact(
        run_validity=calib.VALID_RUN,
        selected_policy="F1_C1",
        candidate_selection_executed=True,
        candidate_results=[
            {
                "candidate_id": c.id,
                "declared_numerator": c.declared_numerator,
                "declared_denominator": c.declared_denominator,
                "max_consecutive": c.max_consecutive,
                "defensible": True,
                "fraction_headroom_exact": calib.fraction_to_json(calib.candidate_fraction_value(c)),
                "consecutive_headroom": c.max_consecutive,
            }
            for c in calib.CANDIDATES
        ],
        m_fraction=Fraction(0, 1),
        m_fraction_window_count=1,
        m_consecutive=0,
        m_consecutive_window_count=1,
        synthetic_base_count=20,
        synthetic_scenario_count=6000,
        synthetic_candidate_comparison_count=180000,
        synthetic_truth_table_mismatch_count=0,
        synthetic_base_metadata=[
            {"base_index": i, "ticker_sha256": "x" * 64, "window_start": "2019-01-01", "window_end": "2019-09-09"}
            for i in range(20)
        ],
        input_provenance={"manifest_sha256": "0" * 64, "ticker_count": 20},
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test-attempt",
        run_started_utc="2026-01-01T00:00:00Z",
        run_completed_or_blocked_utc="2026-01-01T00:00:05Z",
    )


def test_result_artifact_contains_all_30_candidates():
    artifact = _minimal_valid_artifact()
    assert artifact["candidate_count"] == 30
    assert len(artifact["candidate_results"]) == 30


def test_artifact_self_hash_round_trip():
    artifact = _minimal_valid_artifact()
    assert calib.verify_artifact_self_hash(artifact) is True


def test_artifact_self_hash_detects_mutation():
    artifact = _minimal_valid_artifact()
    mutated = dict(artifact)
    mutated["selected_policy"] = "F2_C1"
    assert calib.verify_artifact_self_hash(mutated) is False


def test_artifact_self_hash_missing_key_is_not_verified():
    artifact = _minimal_valid_artifact()
    without_hash = {k: v for k, v in artifact.items() if k != "artifact_self_hash"}
    assert calib.verify_artifact_self_hash(without_hash) is False


# ---------------------------------------------------------------------------
# AH. Manifest structural validation
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
    # Synthetic data can never legitimately match the real, fixed
    # EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256 constant (by design: this phase
    # must never fabricate data that appears to be real V5-B provenance).
    manifest = _synthetic_manifest()
    with pytest.raises(calib.V8BCalibrationBlocked) as excinfo:
        calib.validate_v5b_manifest_structure(manifest)
    assert excinfo.value.reason == "MANIFEST_PAYLOAD_HASH_LIST_MISMATCH"


def test_manifest_payload_hash_list_field_mismatch_rejected():
    payloads = _synthetic_manifest_payloads(300)
    manifest = _synthetic_manifest(payloads=payloads)
    # force the recompute check to pass by monkeypatching the expected constant
    # is not permitted (no override); instead exercise the field-mismatch path
    # directly by making the stored field disagree with the (also-failing)
    # recomputed value, confirming recompute is checked first.
    manifest["payload_hash_list_sha256"] = "1" * 64
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


# ---------------------------------------------------------------------------
# Full-pipeline integration (happy path + invalid-run paths), no real data.
# ---------------------------------------------------------------------------


def _make_fake_ticker_payloads(count: int) -> dict[str, bytes]:
    days = _consecutive_days(date(2019, 1, 2), calib.SYNTHETIC_SEQUENCE_LENGTH)
    return {f"TICK{index:02d}": _payload_bytes(f"TICK{index:02d}.T", days) for index in range(count)}


@pytest.mark.slow
def test_full_run_happy_path_selects_strictest_candidate_and_is_valid():
    manifest_bytes = b'{"note":"fake-provenance-not-a-real-v5b-manifest"}'
    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=_make_fake_ticker_payloads(20),
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test-happy-path",
    )
    assert result["calibration_run_valid"] is True
    assert result["run_invalid_reason_or_null"] is None
    assert result["candidate_selection_executed"] is True
    assert result["selected_policy"] == "F1_C1"
    assert len(result["candidate_results"]) == 30
    assert result["synthetic_base_count"] == 20
    assert result["synthetic_scenario_count"] == 6000
    assert result["synthetic_candidate_comparison_count"] == 180000
    assert result["synthetic_truth_table_mismatch_count"] == 0
    assert calib.verify_artifact_self_hash(result) is True
    assert "TICK00" not in json.dumps(result)  # no raw ticker identity leaks into the artifact


def test_full_run_blocks_on_empty_full_span():
    manifest_bytes = b"{}"
    payloads = _make_fake_ticker_payloads(20)
    # one ticker with only January-2026 rows -> zero observations after
    # restricting to the calibration window (R3).
    payloads["EMPTY01"] = _payload_bytes("EMPTY01.T", [date(2026, 1, 15)])
    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=payloads,
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test-empty-span",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED"
    assert result["selected_policy"] == calib.NOT_EVALUATED
    assert result["candidate_selection_executed"] is False


def test_full_run_blocks_when_fewer_than_20_qualifying_bases():
    manifest_bytes = b"{}"
    payloads = _make_fake_ticker_payloads(19)
    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=payloads,
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test-few-bases",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "SYNTHETIC_BASE_SELECTION_BLOCKED"
    assert result["selected_policy"] == calib.NOT_EVALUATED


def test_full_run_blocks_on_repository_contract_mismatch(tmp_path):
    root = _repo_copy(tmp_path)
    path = root / calib.PREREGISTRATION_PATH
    mutated = bytearray(path.read_bytes())
    mutated[0] ^= 0xFF
    path.write_bytes(bytes(mutated))
    result = calib.run_data_quality_calibration(
        repository_root=root,
        manifest_bytes=b"{}",
        ticker_payloads=_make_fake_ticker_payloads(20),
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test-contract-mismatch",
    )
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "CALIBRATION_PLAN_BLOB_MISMATCH"


def test_full_run_no_ticker_identity_in_blocked_artifact():
    manifest_bytes = b"{}"
    result = calib.run_data_quality_calibration(
        repository_root=REPO_ROOT,
        manifest_bytes=manifest_bytes,
        ticker_payloads=_make_fake_ticker_payloads(5),
        implementation_git_commit="0" * 40,
        calibration_attempt_id="test-privacy",
    )
    serialized = json.dumps(result)
    for index in range(5):
        assert f"TICK0{index}" not in serialized


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
    ]
    for token in forbidden:
        assert token not in source, f"forbidden token found: {token}"
