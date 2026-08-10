"""Fake-only threshold-arithmetic tests for the malformed-OHLCV quality gate.

POLICY_G_PRIME_V1_UNIFORM_RETURNED_ROW_QUALITY_GATE
(V8_HISTORICAL_RESEARCH_DESIGN.md §17). These tests exercise
``src.v8_historical_acquisition._require_malformed_ohlcv_quality_gate`` and
its helpers directly, against synthetic row dicts -- no network, no fake
Yahoo transport, no partition/manifest plumbing. Integration-level behaviour
(manifest shape, staging cleanup, ticker/date leakage, T1/T2 uniformity)
lives in test_v8_historical_acquisition.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src import v7_yahoo_collector as v7
from src import v8_historical_acquisition as acquisition


def _dates(start: tuple[int, int, int], count: int) -> list[str]:
    start_date = date(*start)
    return [(start_date + timedelta(days=i)).isoformat() for i in range(count)]


def valid_row(trading_date: str) -> dict:
    return {"trading_date": trading_date}


def invalid_row(trading_date: str, reason: str = "NONPOSITIVE_CLOSE") -> dict:
    return {"trading_date": trading_date, "reason": reason}


# ---------------------------------------------------------------------------
# A-C: fraction threshold
# ---------------------------------------------------------------------------


def test_a_zero_invalid_rows_passes():
    dates = _dates((2016, 4, 1), 10)
    acquisition._require_malformed_ohlcv_quality_gate([valid_row(d) for d in dates], [])


def test_b_exactly_one_percent_invalid_passes():
    dates = _dates((2016, 4, 1), 100)
    valid = [valid_row(d) for d in dates[1:]]
    invalid = [invalid_row(dates[0])]
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)


def test_c_two_percent_invalid_blocks():
    dates = _dates((2016, 4, 1), 100)
    valid = [valid_row(d) for d in dates[2:]]
    invalid = [invalid_row(dates[0]), invalid_row(dates[1])]
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


# ---------------------------------------------------------------------------
# D-E: consecutive threshold
# ---------------------------------------------------------------------------


def test_d_exactly_five_consecutive_invalid_passes_with_fraction_headroom():
    dates = _dates((2016, 4, 1), 500)
    invalid_dates = dates[200:205]
    valid_dates = dates[:200] + dates[205:]
    valid = [valid_row(d) for d in valid_dates]
    invalid = [invalid_row(d) for d in invalid_dates]
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)


def test_e_six_consecutive_invalid_blocks():
    dates = _dates((2016, 4, 1), 600)
    invalid_dates = dates[200:206]
    valid_dates = dates[:200] + dates[206:]
    valid = [valid_row(d) for d in valid_dates]
    invalid = [invalid_row(d) for d in invalid_dates]
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:CONSECUTIVE_EXCEEDED"


# ---------------------------------------------------------------------------
# F-H: per-test-year semantics
# ---------------------------------------------------------------------------


def test_f_full_series_passes_but_one_test_year_fraction_exceeded_blocks():
    padding = _dates((2016, 4, 1), 300)  # pre-2018, not a test year
    year_dates = _dates((2020, 1, 1), 50)
    invalid_year_dates = year_dates[:2]
    valid_year_dates = year_dates[2:]
    valid = [valid_row(d) for d in padding] + [valid_row(d) for d in valid_year_dates]
    invalid = [invalid_row(d) for d in invalid_year_dates]
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:TEST_YEAR_FRACTION_EXCEEDED"


def test_g_full_series_and_every_applicable_year_pass():
    dates_2018 = _dates((2018, 1, 1), 30)
    dates_2020 = _dates((2020, 6, 1), 40)
    dates_2024 = _dates((2024, 3, 1), 20)
    all_dates = dates_2018 + dates_2020 + dates_2024
    acquisition._require_malformed_ohlcv_quality_gate([valid_row(d) for d in all_dates], [])


def test_h_late_ipo_years_with_no_observations_are_not_applicable():
    # Ticker only listed from 2022 onward: 2018-2021 have zero returned
    # observations and must not themselves BLOCK the acquisition.
    dates = _dates((2022, 1, 1), 100)
    acquisition._require_malformed_ohlcv_quality_gate([valid_row(d) for d in dates], [])


# ---------------------------------------------------------------------------
# I: absent calendar dates are not observations
# ---------------------------------------------------------------------------


def test_i_sparse_calendar_gaps_are_not_treated_as_invalid():
    start = date(2016, 4, 1)
    dates = [(start + timedelta(days=30 * i)).isoformat() for i in range(10)]
    acquisition._require_malformed_ohlcv_quality_gate([valid_row(d) for d in dates], [])


def test_i_observations_sequence_contains_only_returned_rows():
    dates = _dates((2016, 4, 1), 5)
    valid = [valid_row(d) for d in dates[:3]]
    invalid = [invalid_row(d) for d in dates[3:]]
    observations = acquisition._malformed_ohlcv_returned_observations(valid, invalid)
    assert len(observations) == 5
    assert sorted(d for d, _ in observations) == dates


# ---------------------------------------------------------------------------
# J: zero observations
# ---------------------------------------------------------------------------


def test_j_zero_observations_full_series_blocks():
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate([], [])
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:EMPTY_SERIES"


# ---------------------------------------------------------------------------
# Cross-year consecutive run: full-series gate must see it as continuous
# ---------------------------------------------------------------------------


def test_cross_year_consecutive_run_caught_by_full_series_gate():
    padding = _dates((2016, 4, 1), 600)
    dec_dates = _dates((2019, 12, 26), 3)
    jan_dates = _dates((2020, 1, 1), 3)
    valid = [valid_row(d) for d in padding]
    invalid = [invalid_row(d) for d in dec_dates + jan_dates]
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:CONSECUTIVE_EXCEEDED"


# ---------------------------------------------------------------------------
# M: uniform severity across every reason label the canonical parser can
# actually produce -- discovered dynamically, never assumed to be a fixed
# count.
# ---------------------------------------------------------------------------


def _discover_invalid_reasons() -> set[str]:
    base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adjclose": 100.0, "volume": 10.0}
    reasons: set[str] = set()
    for field in ("open", "high", "low", "close", "adjclose"):
        broken_nonfinite = dict(base)
        broken_nonfinite[field] = None
        reasons.add(v7._row_invalid_reason(broken_nonfinite))
        broken_nonpositive = dict(base)
        broken_nonpositive[field] = 0.0
        reasons.add(v7._row_invalid_reason(broken_nonpositive))
    broken_volume_nonfinite = dict(base)
    broken_volume_nonfinite["volume"] = None
    reasons.add(v7._row_invalid_reason(broken_volume_nonfinite))
    broken_volume_negative = dict(base)
    broken_volume_negative["volume"] = -1.0
    reasons.add(v7._row_invalid_reason(broken_volume_negative))
    reasons.discard(None)
    return reasons


def test_m_all_discovered_invalid_reasons_receive_uniform_policy_treatment():
    reasons = _discover_invalid_reasons()
    # Sanity: the canonical parser really does produce more than one label;
    # the exact count is never assumed or hard-coded by this test.
    assert len(reasons) >= 2

    dates = _dates((2016, 4, 1), 200)
    for reason in reasons:
        # Low fraction (1/200): must PASS regardless of which reason label.
        valid = [valid_row(d) for d in dates[1:]]
        invalid = [invalid_row(dates[0], reason=reason)]
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)

        # High fraction (3/200 > 1%): must BLOCK identically regardless of label.
        valid2 = [valid_row(d) for d in dates[3:]]
        invalid2 = [invalid_row(dates[i], reason=reason) for i in range(3)]
        with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
            acquisition._require_malformed_ohlcv_quality_gate(valid2, invalid2)
        assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


def test_m_no_reason_specific_exception_no_tolerance_for_volume_or_adjclose():
    """Guards against a regression that special-cases any one field (e.g.
    'volume invalid is tolerated more')."""
    dates = _dates((2016, 4, 1), 100)
    for reason in ("NEGATIVE_VOLUME", "NONFINITE_VOLUME", "NONFINITE_ADJCLOSE", "NONPOSITIVE_ADJCLOSE"):
        valid = [valid_row(d) for d in dates[2:]]
        invalid = [invalid_row(dates[0], reason=reason), invalid_row(dates[1], reason=reason)]
        with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
            acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
        assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


# ---------------------------------------------------------------------------
# T: exact 1% boundary, deterministic integer arithmetic (no float rounding)
# ---------------------------------------------------------------------------


def test_t_boundary_is_exact_integer_arithmetic_not_float_rounding():
    total = 700
    dates = _dates((2016, 4, 1), total)
    invalid_positions = set(range(0, total, 100))  # 7 isolated positions
    assert len(invalid_positions) == 7
    valid = [valid_row(d) for i, d in enumerate(dates) if i not in invalid_positions]
    invalid = [invalid_row(d) for i, d in enumerate(dates) if i in invalid_positions]
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)  # exactly 1% -> PASS

    invalid_positions_over = invalid_positions | {50}  # an 8th isolated invalid row
    valid_over = [valid_row(d) for i, d in enumerate(dates) if i not in invalid_positions_over]
    invalid_over = [invalid_row(d) for i, d in enumerate(dates) if i in invalid_positions_over]
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid_over, invalid_over)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


# ---------------------------------------------------------------------------
# Policy constants match the frozen §17 clarification exactly
# ---------------------------------------------------------------------------


def test_policy_constants_match_frozen_design_clarification():
    assert acquisition.MALFORMED_OHLCV_POLICY_NAME == "POLICY_G_PRIME_V1_UNIFORM_RETURNED_ROW_QUALITY_GATE"
    assert acquisition.MALFORMED_OHLCV_INVALID_FRACTION_THRESHOLD == 0.01
    assert acquisition.MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS == 5
    assert acquisition.MALFORMED_OHLCV_FULL_P_HIST_CHECK_REQUIRED is True
    assert acquisition.MALFORMED_OHLCV_TEST_YEARS == (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
    assert acquisition.MALFORMED_OHLCV_EXPECTED_CALENDAR_MISSING_DATES_TREATED_AS_MALFORMED is False
    assert acquisition.MALFORMED_OHLCV_THRESHOLD_EXCEEDANCE_ACTION == "BLOCK_WHOLE_ACQUISITION"


def test_q_prohibited_blocks_still_prohibited():
    assert acquisition.PROHIBITED_ACQUISITION_BLOCKS == ("T0", "T3", "T_spare")
    assert acquisition.ALLOWED_ACQUISITION_BLOCKS == ("T1", "T2")
