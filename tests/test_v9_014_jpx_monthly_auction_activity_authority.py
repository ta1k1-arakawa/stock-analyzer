"""Synthetic tests for the V9_014 deterministic OFFLINE core.

All fixtures here are synthetic. No real JPX/J-Quants/Yahoo data, no
protected V9_012 bytes, and no network access are used anywhere in this
file.
"""

from src import v9_014_jpx_monthly_auction_activity_authority as v9014


# ---------------------------------------------------------------------------
# Frozen constants / provenance
# ---------------------------------------------------------------------------

def test_frozen_provenance_constants():
    assert v9014.FROZEN_DESIGN_GIT_SHA == "efee3d0efca368645c00aeed63cb8e0637cd3672"
    assert v9014.FROZEN_DESIGN_BLOB_SHA == "2bbacbf37ab961d1cbf416b7fd476db18778c5b7"
    assert v9014.COVERAGE_START == "2017-01-01"
    assert v9014.COVERAGE_END == "2026-01-31"
    assert v9014.LOGICAL_COVERAGE_MONTH_COUNT == 109
    assert v9014.REQUIRED_PHYSICAL_SOURCE_B_OBJECT_COUNT == 110
    assert v9014.SOURCE_A_CHAIN_SHA256 == (
        "aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45"
    )
    assert v9014.SOURCE_A_PAGE_COUNT == 1
    assert v9014.SOURCE_A_FRESH_ACQUISITION_AUTHORIZED is False
    assert v9014.SOURCE_B_OBJECT_FORMAT == "PDF"
    assert v9014.SOURCE_B_CAN_PROVE_INACTIVITY is False
    assert v9014.SOURCE_C_DOCUMENT_DATE == "2020-10-01"
    assert v9014.EXPECTED_UNPROVEN_SET == frozenset({"2020-10-01"})
    assert v9014.SENTINEL_PROVEN_ACTIVE_DATES == ("2020-09-30", "2020-10-02")


def test_required_segments_frozen_per_era():
    assert v9014.REQUIRED_SEGMENTS_PRE == (
        "1st Section", "2nd Section", "Mothers", "JASDAQ Standard", "JASDAQ Growth",
    )
    assert v9014.REQUIRED_SEGMENTS_POST == ("Prime", "Standard", "Growth")
    assert "TOKYO PRO Market" not in v9014.REQUIRED_SEGMENTS_PRE
    assert "TOKYO PRO Market" not in v9014.REQUIRED_SEGMENTS_POST
    assert "TOKYO PRO Market" in v9014.NOT_REQUIRED_SEGMENTS


def test_logical_month_and_object_bundle_count_invariant():
    assert len(v9014.REQUIRED_LOGICAL_MONTHS) == 109
    assert v9014.REQUIRED_LOGICAL_MONTHS[0] == "2017-01"
    assert v9014.REQUIRED_LOGICAL_MONTHS[-1] == "2026-01"
    total_parts = sum(
        len(v9014.required_source_b_object_parts(m)) for m in v9014.REQUIRED_LOGICAL_MONTHS
    )
    assert total_parts == 110


# ---------------------------------------------------------------------------
# Declared share-unit token mapping (design Section 5.2)
# ---------------------------------------------------------------------------

def test_unit_english_tokens_alone_valid():
    assert v9014.resolve_declared_unit(("shs.",)).unit == v9014.SHARES
    assert v9014.resolve_declared_unit(("thous.shs.",)).unit == v9014.THOUSAND_SHARES


def test_unit_bilingual_agreement_valid():
    r1 = v9014.resolve_declared_unit(("株", "shs."))
    assert r1.unit == v9014.SHARES
    r2 = v9014.resolve_declared_unit(("千株", "thous.shs."))
    assert r2.unit == v9014.THOUSAND_SHARES


def test_unit_bilingual_contradiction_fails_closed():
    r = v9014.resolve_declared_unit(("株", "thous.shs."))
    assert not r.ok
    assert r.failure_reason == v9014.UNIT_CONTRADICTORY_BILINGUAL_FAILURE
    r2 = v9014.resolve_declared_unit(("千株", "shs."))
    assert not r2.ok
    assert r2.failure_reason == v9014.UNIT_CONTRADICTORY_BILINGUAL_FAILURE


def test_unit_japanese_token_alone_not_recognized():
    r = v9014.resolve_declared_unit(("株",))
    assert not r.ok
    assert r.failure_reason == v9014.UNIT_UNSUPPORTED_TOKEN_FAILURE
    r2 = v9014.resolve_declared_unit(("千株",))
    assert not r2.ok


def test_unit_unknown_case_changed_and_invented_aliases_rejected():
    for tokens in (("SHS.",), ("Shs.",), ("Thous.Shs.",), ("thous.SHS.",), ("kabu",),
                   ("1000shares",), ("shares",), ("shs",), ("thous.shs",), ("thousand shares",)):
        r = v9014.resolve_declared_unit(tokens)
        assert not r.ok, tokens
        assert r.failure_reason == v9014.UNIT_UNSUPPORTED_TOKEN_FAILURE, tokens


def test_unit_ambiguous_multiple_english_tokens_fails_closed():
    r = v9014.resolve_declared_unit(("shs.", "thous.shs."))
    assert not r.ok
    assert r.failure_reason == v9014.UNIT_AMBIGUOUS_MULTIPLE_TOKENS_FAILURE


def test_unit_absent_tokens_fails_closed():
    r = v9014.resolve_declared_unit(())
    assert not r.ok
    assert r.failure_reason == v9014.UNIT_ABSENT_FAILURE


# ---------------------------------------------------------------------------
# Reported-value interval semantics (design Section 5.1 / 5.5)
# ---------------------------------------------------------------------------

def test_interval_shares_multiplier_one():
    assert v9014.share_interval(0, v9014.SHARES) == (0, 0)
    assert v9014.share_interval(5, v9014.SHARES) == (5, 5)


def test_interval_thousand_shares_multiplier_1000():
    assert v9014.share_interval(1, v9014.THOUSAND_SHARES) == (1000, 1999)
    assert v9014.share_interval(2, v9014.THOUSAND_SHARES) == (2000, 2999)


def test_interval_numeric_zero_with_multiplier_gt_1_is_not_exact_zero():
    lower, upper = v9014.share_interval(0, v9014.THOUSAND_SHARES)
    assert (lower, upper) == (0, 999)
    assert upper != 0  # not pinned to an exact zero


# ---------------------------------------------------------------------------
# Per-segment adjudication (design Section 5.3 / 5.5)
# ---------------------------------------------------------------------------

def _numeric(quantity, unit_token):
    return v9014.NumericCell(quantity=quantity, unit_tokens=(unit_token,))


def test_segment_definitely_active():
    total = _numeric(2, "thous.shs.")
    tostnet = _numeric(1, "thous.shs.")
    result = v9014.classify_segment(total, tostnet, v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES)
    assert result.status == v9014.DEFINITELY_AUCTION_ACTIVE


def test_segment_not_proven_when_intervals_overlap():
    total = _numeric(1, "thous.shs.")
    tostnet = _numeric(1, "thous.shs.")
    result = v9014.classify_segment(total, tostnet, v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES)
    assert result.status == v9014.NOT_PROVEN


def test_segment_dq_when_tostnet_structurally_exceeds_total():
    total = _numeric(1, "thous.shs.")     # [1000, 1999]
    tostnet = _numeric(5, "thous.shs.")   # [5000, 5999]
    result = v9014.classify_segment(total, tostnet, v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES)
    assert result.status == v9014.DQ
    assert result.reason == v9014.TOSTNET_EXCEEDS_TOTAL_FAILURE


def test_segment_dq_dash_total_with_positive_tostnet():
    total = v9014.DashCell()
    tostnet = _numeric(1, "shs.")
    result = v9014.classify_segment(total, tostnet, v9014.THOUSAND_SHARES, v9014.SHARES)
    assert result.status == v9014.DQ
    assert result.reason == v9014.DASH_TOTAL_WITH_POSITIVE_TOSTNET_FAILURE


def test_segment_not_proven_dash_total_with_zero_tostnet():
    total = v9014.DashCell()
    tostnet = _numeric(0, "thous.shs.")
    result = v9014.classify_segment(total, tostnet, v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES)
    assert result.status == v9014.NOT_PROVEN


def test_segment_not_proven_numeric_total_with_dash_tostnet_never_active():
    total = _numeric(5, "thous.shs.")
    tostnet = v9014.DashCell()
    result = v9014.classify_segment(total, tostnet, v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES)
    assert result.status == v9014.NOT_PROVEN


def test_segment_not_proven_dash_dash():
    result = v9014.classify_segment(
        v9014.DashCell(), v9014.DashCell(), v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES
    )
    assert result.status == v9014.NOT_PROVEN


def test_segment_dq_blank_required_cell():
    result_total_blank = v9014.classify_segment(
        v9014.BlankCell(), _numeric(1, "shs."), v9014.THOUSAND_SHARES, v9014.SHARES
    )
    assert result_total_blank.status == v9014.DQ
    assert result_total_blank.reason == v9014.BLANK_REQUIRED_CELL_FAILURE

    result_tostnet_blank = v9014.classify_segment(
        _numeric(1, "thous.shs."), v9014.BlankCell(), v9014.THOUSAND_SHARES, v9014.THOUSAND_SHARES
    )
    assert result_tostnet_blank.status == v9014.DQ
    assert result_tostnet_blank.reason == v9014.BLANK_REQUIRED_CELL_FAILURE


def test_segment_dq_malformed_cell():
    result = v9014.classify_segment(
        v9014.MalformedCell(), _numeric(0, "shs."), v9014.THOUSAND_SHARES, v9014.SHARES
    )
    assert result.status == v9014.DQ
    assert result.reason == v9014.MALFORMED_VALUE_FAILURE


def test_segment_dq_unsupported_unit_token():
    result = v9014.classify_segment(
        _numeric(1, "kabu"), _numeric(0, "shs."), v9014.THOUSAND_SHARES, v9014.SHARES
    )
    assert result.status == v9014.DQ
    assert result.reason == v9014.UNIT_UNSUPPORTED_TOKEN_FAILURE


def test_segment_dq_unexpected_unit_mismatch_against_expectation():
    # Declared unit resolves fine (THOUSAND_SHARES) but does not match the
    # frozen expected unit (SHARES) for this column.
    result = v9014.classify_segment(
        _numeric(1, "thous.shs."), _numeric(0, "shs."), v9014.SHARES, v9014.SHARES
    )
    assert result.status == v9014.DQ
    assert result.reason == v9014.UNEXPECTED_UNIT_OR_LAYOUT_CHANGE_FAILURE


def test_segment_dq_negative_and_boolean_quantity_malformed():
    result_neg = v9014.classify_segment(
        _numeric(-1, "shs."), _numeric(0, "shs."), v9014.SHARES, v9014.SHARES
    )
    assert result_neg.status == v9014.DQ
    assert result_neg.reason == v9014.MALFORMED_VALUE_FAILURE

    result_bool = v9014.classify_segment(
        _numeric(True, "shs."), _numeric(0, "shs."), v9014.SHARES, v9014.SHARES
    )
    assert result_bool.status == v9014.DQ
    assert result_bool.reason == v9014.MALFORMED_VALUE_FAILURE


def test_no_status_ever_represents_proven_inactivity():
    all_status_values = {v9014.DQ, v9014.DEFINITELY_AUCTION_ACTIVE, v9014.NOT_PROVEN}
    for status in all_status_values:
        assert "INACTIVE" not in status


# ---------------------------------------------------------------------------
# Era binding, Mothers ToSTNeT unit split, and required-unit expectations
# ---------------------------------------------------------------------------

def test_era_for_date_boundaries():
    assert v9014.era_for_date("2017-01-01") == v9014.ERA_PRE
    assert v9014.era_for_date("2022-04-01") == v9014.ERA_PRE
    assert v9014.era_for_date("2022-04-02") is None
    assert v9014.era_for_date("2022-04-03") is None
    assert v9014.era_for_date("2022-04-04") == v9014.ERA_POST
    assert v9014.era_for_date("2026-01-31") == v9014.ERA_POST
    assert v9014.era_for_date("2026-02-01") is None


def test_mothers_tostnet_unit_split_boundaries():
    assert v9014.expected_unit(v9014.ERA_PRE, "Mothers", v9014.COLUMN_TOSTNET, "2017-01-01") == v9014.SHARES
    assert v9014.expected_unit(v9014.ERA_PRE, "Mothers", v9014.COLUMN_TOSTNET, "2019-12-31") == v9014.SHARES
    assert (
        v9014.expected_unit(v9014.ERA_PRE, "Mothers", v9014.COLUMN_TOSTNET, "2020-01-01")
        == v9014.THOUSAND_SHARES
    )
    assert (
        v9014.expected_unit(v9014.ERA_PRE, "Mothers", v9014.COLUMN_TOSTNET, "2022-04-01")
        == v9014.THOUSAND_SHARES
    )


def test_expected_units_pre_and_post_static_segments():
    assert v9014.expected_unit(v9014.ERA_PRE, "1st Section", v9014.COLUMN_TOTAL, "2018-05-01") == v9014.THOUSAND_SHARES
    assert v9014.expected_unit(v9014.ERA_PRE, "Mothers", v9014.COLUMN_TOTAL, "2018-05-01") == v9014.THOUSAND_SHARES
    for segment in v9014.REQUIRED_SEGMENTS_POST:
        assert v9014.expected_unit(v9014.ERA_POST, segment, v9014.COLUMN_TOTAL, "2023-06-01") == v9014.THOUSAND_SHARES
        assert v9014.expected_unit(v9014.ERA_POST, segment, v9014.COLUMN_TOSTNET, "2023-06-01") == v9014.THOUSAND_SHARES


# ---------------------------------------------------------------------------
# Date-level classification (design Section 5.4) -- happy paths both eras
# ---------------------------------------------------------------------------

def _all_not_proven_cells_for_era(era: str, date: str):
    cells = {}
    for segment in v9014.required_segments_for_era(era):
        cells[segment] = {
            v9014.COLUMN_TOTAL: _numeric(1, "thous.shs."),
            v9014.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
        }
    return cells


def test_classify_date_happy_path_pre_era_proven_active():
    date = "2019-05-15"
    cells = _all_not_proven_cells_for_era(v9014.ERA_PRE, date)
    # Mothers ToSTNeT unit for 2019 is SHARES, not THOUSAND_SHARES.
    cells["Mothers"] = {
        v9014.COLUMN_TOTAL: _numeric(1, "thous.shs."),
        v9014.COLUMN_TOSTNET: _numeric(1, "shs."),
    }
    # Make "1st Section" DEFINITELY_AUCTION_ACTIVE.
    cells["1st Section"] = {
        v9014.COLUMN_TOTAL: _numeric(5, "thous.shs."),
        v9014.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
    }
    result = v9014.classify_date(date, cells)
    assert result.status == v9014.PROVEN_AUCTION_ACTIVE


def test_classify_date_happy_path_post_era_proven_active():
    date = "2023-06-01"
    cells = _all_not_proven_cells_for_era(v9014.ERA_POST, date)
    cells["Prime"] = {
        v9014.COLUMN_TOTAL: _numeric(5, "thous.shs."),
        v9014.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
    }
    result = v9014.classify_date(date, cells)
    assert result.status == v9014.PROVEN_AUCTION_ACTIVE


def test_classify_date_not_proven_when_no_segment_definitely_active():
    date = "2023-06-01"
    cells = _all_not_proven_cells_for_era(v9014.ERA_POST, date)
    result = v9014.classify_date(date, cells)
    assert result.status == v9014.NOT_PROVEN


def test_classify_date_tokyo_pro_market_ignored_and_optional():
    date = "2023-06-01"
    cells = _all_not_proven_cells_for_era(v9014.ERA_POST, date)
    cells["Prime"] = {
        v9014.COLUMN_TOTAL: _numeric(5, "thous.shs."),
        v9014.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
    }
    # TOKYO PRO Market absent entirely -- must not fail the date.
    result_without = v9014.classify_date(date, cells)
    assert result_without.status == v9014.PROVEN_AUCTION_ACTIVE

    # TOKYO PRO Market present with only a dash -- must not fail or affect
    # the date, and its content is never read.
    cells_with = dict(cells)
    cells_with["TOKYO PRO Market"] = {
        v9014.COLUMN_TOTAL: v9014.DashCell(),
        v9014.COLUMN_TOSTNET: v9014.DashCell(),
    }
    result_with = v9014.classify_date(date, cells_with)
    assert result_with.status == v9014.PROVEN_AUCTION_ACTIVE


def test_classify_date_dq_missing_required_segment():
    date = "2023-06-01"
    cells = _all_not_proven_cells_for_era(v9014.ERA_POST, date)
    del cells["Standard"]
    result = v9014.classify_date(date, cells)
    assert result.status == v9014.DQ
    assert result.reason == v9014.MISSING_REQUIRED_SEGMENT_FAILURE
    assert result.segment == "Standard"


def test_classify_date_dq_missing_entire_date_table():
    # An empty mapping represents a date entirely missing from required
    # in-era coverage.
    result = v9014.classify_date("2023-06-01", {})
    assert result.status == v9014.DQ
    assert result.reason == v9014.MISSING_REQUIRED_SEGMENT_FAILURE


def test_classify_date_dq_unexpected_unit_layout_change():
    date = "2018-01-01"
    cells = _all_not_proven_cells_for_era(v9014.ERA_PRE, date)
    cells["Mothers"] = {
        v9014.COLUMN_TOTAL: _numeric(1, "thous.shs."),
        # Expected SHARES for a 2018 Mothers ToSTNeT column; declaring
        # THOUSAND_SHARES here is an unexpected unit/layout change.
        v9014.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
    }
    result = v9014.classify_date(date, cells)
    assert result.status == v9014.DQ
    assert result.reason == v9014.UNEXPECTED_UNIT_OR_LAYOUT_CHANGE_FAILURE
    assert result.segment == "Mothers"


def test_classify_date_dq_date_outside_known_era():
    for date in ("2016-12-31", "2022-04-02", "2022-04-03", "2026-02-01"):
        result = v9014.classify_date(date, {})
        assert result.status == v9014.DQ
        assert result.reason == v9014.DATE_OUTSIDE_KNOWN_ERA_FAILURE


# ---------------------------------------------------------------------------
# SOURCE_B physical object-bundle validation, incl. April-2022 two-part rule
# ---------------------------------------------------------------------------

def test_object_bundle_normal_month_ok():
    result = v9014.validate_source_b_object_bundle(
        "2017-01", (v9014.NORMAL_MONTHLY_REPORT2_OBJECT,)
    )
    assert result.status == v9014.OBJECT_BUNDLE_OK


def test_object_bundle_april_2022_two_part_ok():
    result = v9014.validate_source_b_object_bundle(
        "2022-04", (v9014.PRE_APRIL_1_REFERENCE_OBJECT, v9014.NORMAL_MONTHLY_REPORT2_OBJECT)
    )
    assert result.status == v9014.OBJECT_BUNDLE_OK


def test_object_bundle_april_2022_single_object_claim_rejected():
    # This is exactly the prohibited claim: one physical object covering
    # the whole of April 1 plus the rest of the month.
    result = v9014.validate_source_b_object_bundle(
        "2022-04", (v9014.NORMAL_MONTHLY_REPORT2_OBJECT,)
    )
    assert result.status == v9014.OBJECT_BUNDLE_MISSING_OR_UNEXPECTED_PART_FAILURE


def test_object_bundle_duplicate_part_rejected():
    result = v9014.validate_source_b_object_bundle(
        "2017-01", (v9014.NORMAL_MONTHLY_REPORT2_OBJECT, v9014.NORMAL_MONTHLY_REPORT2_OBJECT)
    )
    assert result.status == v9014.OBJECT_BUNDLE_DUPLICATE_PART_FAILURE


def test_object_bundle_wrong_parts_for_normal_month_rejected():
    result = v9014.validate_source_b_object_bundle(
        "2017-01", (v9014.PRE_APRIL_1_REFERENCE_OBJECT, v9014.NORMAL_MONTHLY_REPORT2_OBJECT)
    )
    assert result.status == v9014.OBJECT_BUNDLE_MISSING_OR_UNEXPECTED_PART_FAILURE


def test_object_bundle_unknown_month_rejected():
    result = v9014.validate_source_b_object_bundle(
        "2016-12", (v9014.NORMAL_MONTHLY_REPORT2_OBJECT,)
    )
    assert result.status == v9014.OBJECT_BUNDLE_UNKNOWN_MONTH_FAILURE


# ---------------------------------------------------------------------------
# SOURCE_C confirmation and frozen cross-source exact-set/sentinel relation
# ---------------------------------------------------------------------------

def test_source_c_confirmed_only_when_both_assertions_true():
    assert v9014.source_c_confirmed_exception_set(True, True) == frozenset({"2020-10-01"})
    assert v9014.source_c_confirmed_exception_set(True, False) == frozenset()
    assert v9014.source_c_confirmed_exception_set(False, True) == frozenset()
    assert v9014.source_c_confirmed_exception_set(False, False) == frozenset()


def _scheduled_and_proven_window():
    scheduled = [
        "2020-09-28", "2020-09-29", "2020-09-30",
        "2020-10-01", "2020-10-02", "2020-10-05",
    ]
    proven_active = [d for d in scheduled if d != "2020-10-01"]
    return scheduled, proven_active


def test_relation_passes_with_confirmed_source_c_and_matching_evidence():
    scheduled, proven_active = _scheduled_and_proven_window()
    source_c = v9014.source_c_confirmed_exception_set(True, True)
    result = v9014.evaluate_cross_source_relation(scheduled, proven_active, source_c)
    assert result.status == v9014.RELATION_PASS
    assert result.left_diff == frozenset({"2020-10-01"})
    assert result.right_diff == frozenset()
    assert result.left_exact_expected is True
    assert result.right_empty is True
    assert result.cross_source_consistent is True
    assert result.sentinel_2020_09_30_proven_active is True
    assert result.sentinel_2020_10_02_proven_active is True


def test_relation_fails_when_source_c_not_confirmed():
    scheduled, proven_active = _scheduled_and_proven_window()
    unconfirmed_source_c = v9014.source_c_confirmed_exception_set(True, False)
    result = v9014.evaluate_cross_source_relation(scheduled, proven_active, unconfirmed_source_c)
    assert result.status == v9014.RELATION_FAILURE
    assert result.left_exact_expected is True
    assert result.cross_source_consistent is False


def test_relation_fails_on_extra_unproven_date_beyond_2020_10_01():
    scheduled, proven_active = _scheduled_and_proven_window()
    proven_active_with_extra_gap = [d for d in proven_active if d != "2020-09-28"]
    source_c = v9014.source_c_confirmed_exception_set(True, True)
    result = v9014.evaluate_cross_source_relation(scheduled, proven_active_with_extra_gap, source_c)
    assert result.status == v9014.RELATION_FAILURE
    assert result.left_exact_expected is False
    assert result.left_diff == frozenset({"2020-10-01", "2020-09-28"})


def test_relation_fails_on_right_diff_extra_proven_date_not_scheduled():
    scheduled, proven_active = _scheduled_and_proven_window()
    proven_active_with_bogus = list(proven_active) + ["2020-10-99"]
    source_c = v9014.source_c_confirmed_exception_set(True, True)
    result = v9014.evaluate_cross_source_relation(scheduled, proven_active_with_bogus, source_c)
    assert result.status == v9014.RELATION_FAILURE
    assert result.right_empty is False


def test_relation_fails_when_sentinel_missing_independent_of_diffs():
    # Deliberately drop 2020-09-30 from BOTH scheduled and proven-active so
    # it does not appear in either diff, isolating the sentinel check.
    scheduled = ["2020-09-28", "2020-09-29", "2020-10-01", "2020-10-02", "2020-10-05"]
    proven_active = ["2020-09-28", "2020-09-29", "2020-10-02", "2020-10-05"]
    source_c = v9014.source_c_confirmed_exception_set(True, True)
    result = v9014.evaluate_cross_source_relation(scheduled, proven_active, source_c)
    assert result.left_diff == frozenset({"2020-10-01"})
    assert result.right_diff == frozenset()
    assert result.sentinel_2020_09_30_proven_active is False
    assert result.status == v9014.RELATION_FAILURE


def test_relation_never_derives_trading_dates_attribute():
    # The relation result carries no materialized trading-dates output.
    scheduled, proven_active = _scheduled_and_proven_window()
    source_c = v9014.source_c_confirmed_exception_set(True, True)
    result = v9014.evaluate_cross_source_relation(scheduled, proven_active, source_c)
    assert not hasattr(result, "trading_dates")
    assert not hasattr(v9014, "materialize_trading_dates")
