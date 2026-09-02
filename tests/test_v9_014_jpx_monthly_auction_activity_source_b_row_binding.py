"""Synthetic tests for the V9_014 deterministic OFFLINE SOURCE_B extracted
row / physical-object binding.

All fixtures here are synthetic already-extracted cells reusing the
existing core Cell types. No real JPX/PDF data, no network access, and no
filesystem access are used anywhere in this file.
"""

from src import v9_014_jpx_monthly_auction_activity_authority as core
from src import v9_014_jpx_monthly_auction_activity_source_b_row_binding as rb


def _numeric(quantity, unit_token):
    return core.NumericCell(quantity=quantity, unit_tokens=(unit_token,))


def _pre_era_segment_cells(date, active):
    total_quantity = 5 if active else 1
    cells = {}
    for segment in core.REQUIRED_SEGMENTS_PRE:
        if segment == "Mothers":
            tostnet_token = (
                "shs." if date < core.MOTHERS_TOSTNET_UNIT_SPLIT_DATE else "thous.shs."
            )
        else:
            tostnet_token = "thous.shs."
        cells[segment] = {
            core.COLUMN_TOTAL: _numeric(total_quantity, "thous.shs."),
            core.COLUMN_TOSTNET: _numeric(1, tostnet_token),
        }
    return cells


def _post_era_segment_cells(date, active):
    total_quantity = 5 if active else 1
    cells = {}
    for segment in core.REQUIRED_SEGMENTS_POST:
        cells[segment] = {
            core.COLUMN_TOTAL: _numeric(total_quantity, "thous.shs."),
            core.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
        }
    return cells


def _row(date, cells):
    return rb.SourceBDailyRow(date=date, segment_cells=cells)


def _bundle(logical_month, object_part, rows):
    return rb.SourceBObjectRowBundle(
        logical_month=logical_month, object_part=object_part, rows=rows
    )


# ---------------------------------------------------------------------------
# Happy paths -- classifications come straight from core classify_date
# ---------------------------------------------------------------------------

def test_normal_pre_month_happy_path_uses_core_classifications():
    # Dates chosen in the post-2020-01-01 portion of ERA_PRE so Mothers
    # ToSTNeT uses THOUSAND_SHARES like every other PRE segment, avoiding
    # the SHARES-vs-THOUSAND_SHARES asymmetry exercised deliberately by
    # test_mothers_tostnet_pre_2020_shares_semantics_remains_core_controlled.
    rows = [
        _row("2020-06-03", _pre_era_segment_cells("2020-06-03", active=True)),
        _row("2020-06-04", _pre_era_segment_cells("2020-06-04", active=False)),
    ]
    result = rb.bind_source_b_object_rows(
        _bundle("2020-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, rows)
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    assert result.date_classifications["2020-06-03"] == core.classify_date(
        "2020-06-03", _pre_era_segment_cells("2020-06-03", active=True)
    )
    assert result.date_classifications["2020-06-04"] == core.classify_date(
        "2020-06-04", _pre_era_segment_cells("2020-06-04", active=False)
    )
    assert result.date_classifications["2020-06-03"].status == core.PROVEN_AUCTION_ACTIVE
    assert result.date_classifications["2020-06-04"].status == core.NOT_PROVEN
    # Insertion order matches supplied row order.
    assert list(result.date_classifications.keys()) == ["2020-06-03", "2020-06-04"]


def test_normal_post_month_happy_path_uses_core_classifications():
    rows = [
        _row("2023-06-01", _post_era_segment_cells("2023-06-01", active=True)),
        _row("2023-06-02", _post_era_segment_cells("2023-06-02", active=False)),
    ]
    result = rb.bind_source_b_object_rows(
        _bundle("2023-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, rows)
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    assert result.date_classifications["2023-06-01"].status == core.PROVEN_AUCTION_ACTIVE
    assert result.date_classifications["2023-06-02"].status == core.NOT_PROVEN


def test_mothers_tostnet_pre_2020_shares_semantics_remains_core_controlled():
    date = "2019-12-15"
    cells = _pre_era_segment_cells(date, active=False)
    # Declaring THOUSAND_SHARES for Mothers ToSTNeT before 2020 is an
    # unexpected unit/layout change -- this module never overrides that;
    # it is exactly core's own expected_unit rule.
    wrong_unit_cells = dict(cells)
    wrong_unit_cells["Mothers"] = {
        core.COLUMN_TOTAL: _numeric(1, "thous.shs."),
        core.COLUMN_TOSTNET: _numeric(1, "thous.shs."),  # wrong: expected SHARES
    }
    result = rb.bind_source_b_object_rows(
        _bundle("2019-12", core.NORMAL_MONTHLY_REPORT2_OBJECT, [_row(date, wrong_unit_cells)])
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    classification = result.date_classifications[date]
    assert classification.status == core.DQ
    assert classification.reason == core.UNEXPECTED_UNIT_OR_LAYOUT_CHANGE_FAILURE
    assert classification.segment == "Mothers"
    # Confirm this is identical to calling core.classify_date directly.
    assert classification == core.classify_date(date, wrong_unit_cells)


def test_mothers_tostnet_2020_onward_thousand_shares_remains_core_controlled():
    date = "2020-03-10"
    cells = _pre_era_segment_cells(date, active=True)
    result = rb.bind_source_b_object_rows(
        _bundle("2020-03", core.NORMAL_MONTHLY_REPORT2_OBJECT, [_row(date, cells)])
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    assert result.date_classifications[date] == core.classify_date(date, cells)
    assert result.date_classifications[date].status == core.PROVEN_AUCTION_ACTIVE


def test_missing_required_segment_yields_existing_core_dq_unchanged():
    date = "2019-06-03"
    cells = _pre_era_segment_cells(date, active=True)
    del cells["Mothers"]
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, [_row(date, cells)])
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    classification = result.date_classifications[date]
    assert classification.status == core.DQ
    assert classification.reason == core.MISSING_REQUIRED_SEGMENT_FAILURE
    assert classification.segment == "Mothers"
    assert classification == core.classify_date(date, cells)


def test_malformed_blank_unit_mismatch_yields_existing_core_dq_unchanged():
    date = "2019-06-03"
    cells = _pre_era_segment_cells(date, active=True)
    cells["1st Section"] = {
        core.COLUMN_TOTAL: core.BlankCell(),
        core.COLUMN_TOSTNET: _numeric(1, "thous.shs."),
    }
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, [_row(date, cells)])
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    classification = result.date_classifications[date]
    assert classification.status == core.DQ
    assert classification.reason == core.BLANK_REQUIRED_CELL_FAILURE
    assert classification == core.classify_date(date, cells)


# ---------------------------------------------------------------------------
# Object-binding failures
# ---------------------------------------------------------------------------

def test_duplicate_date_fails_object_binding():
    rows = [
        _row("2019-06-03", _pre_era_segment_cells("2019-06-03", active=True)),
        _row("2019-06-03", _pre_era_segment_cells("2019-06-03", active=True)),
    ]
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, rows)
    )
    assert result.status == rb.DUPLICATE_ROW_DATE_FAILURE
    assert result.date == "2019-06-03"
    assert result.date_classifications is None


def test_date_outside_logical_month_fails():
    rows = [_row("2019-07-01", _pre_era_segment_cells("2019-07-01", active=True))]
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, rows)
    )
    assert result.status == rb.ROW_DATE_OUTSIDE_LOGICAL_MONTH_FAILURE
    assert result.date == "2019-07-01"
    assert result.date_classifications is None


def test_malformed_non_canonical_date_fails():
    for bad_date in ("2019/06/03", "19-06-03", "2019-13-01", "2019-06-31", "not-a-date", ""):
        rows = [_row(bad_date, {})]
        result = rb.bind_source_b_object_rows(
            _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, rows)
        )
        assert result.status == rb.MALFORMED_ROW_DATE_FAILURE, bad_date
        assert result.date_classifications is None


def test_bool_date_fails_closed():
    rows = [rb.SourceBDailyRow(date=True, segment_cells={})]
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, rows)
    )
    assert result.status == rb.MALFORMED_ROW_DATE_FAILURE
    assert result.date_classifications is None


def test_empty_normal_object_fails():
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, [])
    )
    assert result.status == rb.EMPTY_ROW_COLLECTION_FAILURE
    assert result.date_classifications is None


def test_invalid_object_part_month_combination_fails():
    # PRE_APRIL_1_REFERENCE_OBJECT is only permitted for logical_month
    # "2022-04"; requesting it for a normal month must fail before any
    # row is even considered.
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.PRE_APRIL_1_REFERENCE_OBJECT, [])
    )
    assert result.status == rb.INVALID_OBJECT_PART_FAILURE
    assert result.date_classifications is None


def test_invalid_logical_month_fails():
    result = rb.bind_source_b_object_rows(
        _bundle("2016-12", core.NORMAL_MONTHLY_REPORT2_OBJECT, [])
    )
    assert result.status == rb.INVALID_LOGICAL_MONTH_FAILURE
    assert result.date_classifications is None


def test_invalid_object_part_string_fails():
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", "SOME_OTHER_OBJECT", [])
    )
    assert result.status == rb.INVALID_OBJECT_PART_FAILURE
    assert result.date_classifications is None


# ---------------------------------------------------------------------------
# April 2022 special-case binding
# ---------------------------------------------------------------------------

def test_april_pre_exact_2022_04_01_passes():
    row = _row("2022-04-01", _pre_era_segment_cells("2022-04-01", active=True))
    result = rb.bind_source_b_object_rows(
        _bundle("2022-04", core.PRE_APRIL_1_REFERENCE_OBJECT, [row])
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    assert list(result.date_classifications.keys()) == ["2022-04-01"]
    assert result.date_classifications["2022-04-01"] == core.classify_date(
        "2022-04-01", _pre_era_segment_cells("2022-04-01", active=True)
    )


def test_april_pre_missing_2022_04_01_fails():
    result = rb.bind_source_b_object_rows(
        _bundle("2022-04", core.PRE_APRIL_1_REFERENCE_OBJECT, [])
    )
    assert result.status == rb.APRIL_PRE_REQUIRED_DATE_MISSING_FAILURE
    assert result.date_classifications is None


def test_april_pre_any_other_date_fails():
    for other_date in ("2022-03-31", "2022-04-04", "2021-04-01"):
        row = _row(other_date, {})
        result = rb.bind_source_b_object_rows(
            _bundle("2022-04", core.PRE_APRIL_1_REFERENCE_OBJECT, [row])
        )
        assert result.status == rb.APRIL_PRE_WRONG_DATE_FAILURE, other_date
        assert result.date_classifications is None


def test_april_normal_rejects_2022_04_01():
    row = _row("2022-04-01", _post_era_segment_cells("2022-04-01", active=True))
    result = rb.bind_source_b_object_rows(
        _bundle("2022-04", core.NORMAL_MONTHLY_REPORT2_OBJECT, [row])
    )
    assert result.status == rb.ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_FAILURE
    assert result.date == "2022-04-01"
    assert result.date_classifications is None


def test_april_normal_rejects_non_business_gap_days():
    for gap_date in ("2022-04-02", "2022-04-03"):
        row = _row(gap_date, {})
        result = rb.bind_source_b_object_rows(
            _bundle("2022-04", core.NORMAL_MONTHLY_REPORT2_OBJECT, [row])
        )
        assert result.status == rb.ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_FAILURE, gap_date


def test_april_normal_accepts_2022_04_04_post_row():
    row = _row("2022-04-04", _post_era_segment_cells("2022-04-04", active=True))
    result = rb.bind_source_b_object_rows(
        _bundle("2022-04", core.NORMAL_MONTHLY_REPORT2_OBJECT, [row])
    )
    assert result.status == rb.OBJECT_ROW_BINDING_OK
    assert result.date_classifications["2022-04-04"].status == core.PROVEN_AUCTION_ACTIVE


# ---------------------------------------------------------------------------
# No float, no trading_dates/relation/profitability output
# ---------------------------------------------------------------------------

def test_no_float_conversion_or_arithmetic_added():
    import inspect

    source = inspect.getsource(rb)
    assert "float(" not in source
    assert ": float" not in source


def test_no_trading_dates_relation_or_profitability_output():
    assert not hasattr(rb, "trading_dates")
    assert not hasattr(rb, "materialize_trading_dates")
    assert not hasattr(rb, "evaluate_cross_source_relation")
    assert not hasattr(rb, "RelationEvaluation")
    row = _row("2019-06-03", _pre_era_segment_cells("2019-06-03", active=True))
    result = rb.bind_source_b_object_rows(
        _bundle("2019-06", core.NORMAL_MONTHLY_REPORT2_OBJECT, [row])
    )
    assert not hasattr(result, "trading_dates")
    assert not hasattr(result, "profitability")
