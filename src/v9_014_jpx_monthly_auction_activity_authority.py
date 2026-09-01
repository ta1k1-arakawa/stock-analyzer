"""Deterministic OFFLINE core for the V9_014 JPX monthly auction-activity
authority study.

This module implements ONLY the pure, deterministic classification and
validation logic frozen by the V9_014 design (see
``V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR_DESIGN_DRAFT.md``,
frozen at design git SHA ``efee3d0efca368645c00aeed63cb8e0637cd3672``, design
blob SHA ``2bbacbf37ab961d1cbf416b7fd476db18778c5b7``). It operates only on
caller-supplied, already-extracted synthetic cells and dates.

It performs NO PDF extraction, NO raw unit-cell text normalization, NO URL
discovery/guessing, NO network acquisition, and touches NO durable
production/protected state; it is not a real runner. It never materializes a
final ``trading_dates`` sequence -- per the frozen design (Section 6.1 and
Section 8), that subtraction is authority-driven and occurs only in a later
reviewed implementation/execution stage after every frozen validation PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union


STUDY_ID = "V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR"

FROZEN_DESIGN_GIT_SHA = "efee3d0efca368645c00aeed63cb8e0637cd3672"
FROZEN_DESIGN_BLOB_SHA = "2bbacbf37ab961d1cbf416b7fd476db18778c5b7"

COVERAGE_START = "2017-01-01"
COVERAGE_END = "2026-01-31"
LOGICAL_COVERAGE_MONTH_COUNT = 109
REQUIRED_PHYSICAL_SOURCE_B_OBJECT_COUNT = 110

# --- SOURCE_A: scheduled TSE business-day superset (reused, not acquired) --
SOURCE_A_ROLE = "SCHEDULED_TSE_BUSINESS_DAY_SUPERSET"
SOURCE_A_PROVIDER = "OFFICIAL_JQUANTS_MARKETS_CALENDAR"
SOURCE_A_ENDPOINT = "https://api.jquants.com/v2/markets/calendar"
SOURCE_A_BASE_QUERY = {"from": COVERAGE_START, "to": COVERAGE_END}
SOURCE_A_CHAIN_SHA256 = (
    "aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45"
)
SOURCE_A_PAGE_COUNT = 1
SOURCE_A_FRESH_ACQUISITION_AUTHORIZED = False

# --- SOURCE_B: regular-auction activity proof evidence ---------------------
SOURCE_B_ROLE = "ACTUAL_TSE_REGULAR_AUCTION_ACTIVITY_DATE_EVIDENCE"
SOURCE_B_PROVIDER = "OFFICIAL_JPX_TSE_MONTHLY_STATISTICS_REPORT_ARCHIVE"
SOURCE_B_ARCHIVE_ROOT = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html"
)
SOURCE_B_REPORT = 'Report 2 "Stock Trading Volume & Value"'
SOURCE_B_TABLE = "Trading Volume & Value (Daily)"
SOURCE_B_OBJECT_FORMAT = "PDF"
SOURCE_B_CAN_PROVE_INACTIVITY = False

# --- SOURCE_C: exceptional full-day auction-closure authority --------------
SOURCE_C_ROLE = "EXCEPTIONAL_FULL_DAY_AUCTION_CLOSURE_AUTHORITY"
SOURCE_C_PROVIDER = "OFFICIAL_JPX_TSE_MARKET_NEWS"
SOURCE_C_DOCUMENT_DATE = "2020-10-01"
SOURCE_C_DOCUMENT_TITLE = "Treatment of Trades for Today at arrowhead"
SOURCE_C_DOCUMENT_LANGUAGE = "ENGLISH"
SOURCE_C_FIXED_DATE_SET = frozenset({SOURCE_C_DOCUMENT_DATE})

EXPECTED_UNPROVEN_SET = frozenset({"2020-10-01"})
SENTINEL_PROVEN_ACTIVE_DATES = ("2020-09-30", "2020-10-02")

# --- Era boundaries and required segments -----------------------------------
ERA_PRE = "ERA_PRE"
ERA_POST = "ERA_POST"
ERA_PRE_START = "2017-01-01"
ERA_PRE_END = "2022-04-01"
ERA_POST_START = "2022-04-04"
ERA_POST_END = "2026-01-31"

REQUIRED_SEGMENTS_PRE = (
    "1st Section",
    "2nd Section",
    "Mothers",
    "JASDAQ Standard",
    "JASDAQ Growth",
)
REQUIRED_SEGMENTS_POST = ("Prime", "Standard", "Growth")
REQUIRED_SEGMENTS_BY_ERA = {ERA_PRE: REQUIRED_SEGMENTS_PRE, ERA_POST: REQUIRED_SEGMENTS_POST}
NOT_REQUIRED_SEGMENTS = frozenset({"TOKYO PRO Market"})

# --- Canonical share-unit semantics (design Section 5.2) -------------------
SHARES = "SHARES"
THOUSAND_SHARES = "THOUSAND_SHARES"
CANONICAL_UNITS = (SHARES, THOUSAND_SHARES)
UNIT_MULTIPLIER = {SHARES: 1, THOUSAND_SHARES: 1000}

EN_UNIT_TOKENS = {"shs.": SHARES, "thous.shs.": THOUSAND_SHARES}
JA_UNIT_TOKENS = {"株": SHARES, "千株": THOUSAND_SHARES}

MOTHERS_TOSTNET_UNIT_SPLIT_DATE = "2020-01-01"

COLUMN_TOTAL = "TOTAL_TRADING_VOLUME"
COLUMN_TOSTNET = "TOSTNET_TRADING_VOLUME"

_STATIC_REQUIRED_UNIT_PRE = {
    ("1st Section", COLUMN_TOTAL): THOUSAND_SHARES,
    ("1st Section", COLUMN_TOSTNET): THOUSAND_SHARES,
    ("2nd Section", COLUMN_TOTAL): THOUSAND_SHARES,
    ("2nd Section", COLUMN_TOSTNET): THOUSAND_SHARES,
    ("Mothers", COLUMN_TOTAL): THOUSAND_SHARES,
    # ("Mothers", COLUMN_TOSTNET) is date-dependent; see expected_unit().
    ("JASDAQ Standard", COLUMN_TOTAL): THOUSAND_SHARES,
    ("JASDAQ Standard", COLUMN_TOSTNET): THOUSAND_SHARES,
    ("JASDAQ Growth", COLUMN_TOTAL): THOUSAND_SHARES,
    ("JASDAQ Growth", COLUMN_TOSTNET): THOUSAND_SHARES,
}
_STATIC_REQUIRED_UNIT_POST = {
    (segment, column): THOUSAND_SHARES
    for segment in REQUIRED_SEGMENTS_POST
    for column in (COLUMN_TOTAL, COLUMN_TOSTNET)
}

# --- Classification status codes --------------------------------------------
DQ = "DQ"
DEFINITELY_AUCTION_ACTIVE = "DEFINITELY_AUCTION_ACTIVE"
NOT_PROVEN = "NOT_PROVEN"
PROVEN_AUCTION_ACTIVE = "PROVEN_AUCTION_ACTIVE"

# --- Frozen data-quality failure reason codes -------------------------------
UNIT_ABSENT_FAILURE = "UNIT_ABSENT_DQ_FAILURE"
UNIT_UNSUPPORTED_TOKEN_FAILURE = "UNSUPPORTED_SHARE_UNIT_DQ_FAILURE"
UNIT_AMBIGUOUS_MULTIPLE_TOKENS_FAILURE = "UNIT_AMBIGUOUS_MULTIPLE_TOKENS_DQ_FAILURE"
UNIT_CONTRADICTORY_BILINGUAL_FAILURE = "CONTRADICTORY_BILINGUAL_UNIT_DQ_FAILURE"
BLANK_REQUIRED_CELL_FAILURE = "BLANK_REQUIRED_IN_ERA_CELL_DQ_FAILURE"
MALFORMED_VALUE_FAILURE = "MALFORMED_VALUE_DQ_FAILURE"
UNEXPECTED_UNIT_OR_LAYOUT_CHANGE_FAILURE = "UNEXPECTED_UNIT_OR_LAYOUT_CHANGE_DQ_FAILURE"
DASH_TOTAL_WITH_POSITIVE_TOSTNET_FAILURE = "DASH_TOTAL_WITH_POSITIVE_TOSTNET_DQ_FAILURE"
TOSTNET_EXCEEDS_TOTAL_FAILURE = "TOSTNET_EXCEEDS_TOTAL_STRUCTURALLY_IMPOSSIBLE_DQ_FAILURE"
MISSING_REQUIRED_SEGMENT_FAILURE = "MISSING_REQUIRED_SEGMENT_DQ_FAILURE"
DATE_OUTSIDE_KNOWN_ERA_FAILURE = "DATE_OUTSIDE_KNOWN_ERA_DQ_FAILURE"
OBJECT_BUNDLE_OK = "OBJECT_BUNDLE_OK"
OBJECT_BUNDLE_MISSING_OR_UNEXPECTED_PART_FAILURE = (
    "OBJECT_BUNDLE_MISSING_OR_UNEXPECTED_PART_DQ_FAILURE"
)
OBJECT_BUNDLE_DUPLICATE_PART_FAILURE = "OBJECT_BUNDLE_DUPLICATE_PART_DQ_FAILURE"
OBJECT_BUNDLE_UNKNOWN_MONTH_FAILURE = "OBJECT_BUNDLE_UNKNOWN_MONTH_DQ_FAILURE"

RELATION_PASS = "RELATION_PASS"
RELATION_FAILURE = "ACTUAL_TRADING_DAY_AUTHORITY_FAILURE"


# --- Cell input types (already-extracted synthetic cells) ------------------
@dataclass(frozen=True)
class BlankCell:
    """A required in-era cell with no content at all."""


@dataclass(frozen=True)
class DashCell:
    """An explicit dash / "Nil or no value" token. Not malformed."""


@dataclass(frozen=True)
class MalformedCell:
    """A cell whose content is present but not a valid numeric observation."""


@dataclass(frozen=True)
class NumericCell:
    """A numeric quantity with its exact declared-unit tokens.

    ``unit_tokens`` are the exact normalized semantic tokens already present
    in the declared unit cell (see design Section 5.2); this module performs
    no raw-text normalization of them.
    """

    quantity: int
    unit_tokens: Sequence[str]


Cell = Union[BlankCell, DashCell, MalformedCell, NumericCell]


@dataclass(frozen=True)
class UnitResolution:
    unit: Optional[str]
    failure_reason: Optional[str]

    @property
    def ok(self) -> bool:
        return self.unit is not None and self.failure_reason is None


def resolve_declared_unit(tokens: Sequence[str]) -> UnitResolution:
    """Resolve a declared-unit cell's exact tokens to a canonical unit.

    Recognition uses exact string equality against the frozen token set only
    -- no fuzzy matching, case-folding, or invented aliases. A Japanese token
    is recognized only when paired with its matching English token in the
    same cell; a Japanese token without an accompanying English token is not
    recognized. Any unknown token, any ambiguous (multiply-declared) token
    set, or any semantically contradictory bilingual pair fails closed.
    """

    if not tokens:
        return UnitResolution(None, UNIT_ABSENT_FAILURE)

    for token in tokens:
        if token not in EN_UNIT_TOKENS and token not in JA_UNIT_TOKENS:
            return UnitResolution(None, UNIT_UNSUPPORTED_TOKEN_FAILURE)

    en_matches = [EN_UNIT_TOKENS[t] for t in tokens if t in EN_UNIT_TOKENS]
    ja_matches = [JA_UNIT_TOKENS[t] for t in tokens if t in JA_UNIT_TOKENS]

    if len(en_matches) > 1 or len(ja_matches) > 1:
        return UnitResolution(None, UNIT_AMBIGUOUS_MULTIPLE_TOKENS_FAILURE)

    if ja_matches and not en_matches:
        # Japanese tokens are recognized only within a bilingual cell.
        return UnitResolution(None, UNIT_UNSUPPORTED_TOKEN_FAILURE)

    if en_matches and ja_matches:
        if en_matches[0] != ja_matches[0]:
            return UnitResolution(None, UNIT_CONTRADICTORY_BILINGUAL_FAILURE)
        return UnitResolution(en_matches[0], None)

    return UnitResolution(en_matches[0], None)


def share_interval(quantity: int, unit: str) -> tuple:
    """Exact-integer reported-value interval for ``quantity`` under ``unit``.

    ``m == 1`` yields ``[q, q]``; ``m > 1`` yields ``[q*m, q*m + (m-1)]``.
    Integer arithmetic only.
    """

    multiplier = UNIT_MULTIPLIER[unit]
    lower = quantity * multiplier
    upper = lower if multiplier == 1 else lower + (multiplier - 1)
    return lower, upper


@dataclass(frozen=True)
class ResolvedNumeric:
    lower: int
    upper: int


@dataclass(frozen=True)
class ResolvedDash:
    pass


@dataclass(frozen=True)
class ResolvedFailure:
    reason: str


ResolvedCell = Union[ResolvedNumeric, ResolvedDash, ResolvedFailure]


def resolve_cell(cell: Cell, expected_unit_value: str) -> ResolvedCell:
    """Resolve one already-extracted cell against its preregistered unit."""

    if isinstance(cell, BlankCell):
        return ResolvedFailure(BLANK_REQUIRED_CELL_FAILURE)
    if isinstance(cell, MalformedCell):
        return ResolvedFailure(MALFORMED_VALUE_FAILURE)
    if isinstance(cell, DashCell):
        return ResolvedDash()
    if isinstance(cell, NumericCell):
        quantity = cell.quantity
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
            return ResolvedFailure(MALFORMED_VALUE_FAILURE)
        unit_resolution = resolve_declared_unit(cell.unit_tokens)
        if not unit_resolution.ok:
            return ResolvedFailure(unit_resolution.failure_reason)
        if unit_resolution.unit != expected_unit_value:
            return ResolvedFailure(UNEXPECTED_UNIT_OR_LAYOUT_CHANGE_FAILURE)
        lower, upper = share_interval(quantity, unit_resolution.unit)
        return ResolvedNumeric(lower, upper)
    raise TypeError(f"unsupported cell type: {type(cell)!r}")


@dataclass(frozen=True)
class SegmentClassification:
    status: str
    reason: Optional[str] = None


def classify_segment(
    total_cell: Cell,
    tostnet_cell: Cell,
    total_expected_unit: str,
    tostnet_expected_unit: str,
) -> SegmentClassification:
    """Frozen per-segment adjudication (design Section 5.3/5.5).

    Returns exactly one of ``DQ``, ``DEFINITELY_AUCTION_ACTIVE``, or
    ``NOT_PROVEN``. There is no status representing proven inactivity: a
    dash can never, by itself, prove positive activity or exact zero.
    """

    total_state = resolve_cell(total_cell, total_expected_unit)
    tostnet_state = resolve_cell(tostnet_cell, tostnet_expected_unit)

    if isinstance(total_state, ResolvedFailure):
        return SegmentClassification(DQ, total_state.reason)
    if isinstance(tostnet_state, ResolvedFailure):
        return SegmentClassification(DQ, tostnet_state.reason)

    if isinstance(total_state, ResolvedDash) and isinstance(tostnet_state, ResolvedNumeric):
        if tostnet_state.lower > 0:
            return SegmentClassification(DQ, DASH_TOTAL_WITH_POSITIVE_TOSTNET_FAILURE)
        return SegmentClassification(NOT_PROVEN, None)

    if isinstance(total_state, ResolvedNumeric) and isinstance(tostnet_state, ResolvedNumeric):
        if total_state.upper < tostnet_state.lower:
            return SegmentClassification(DQ, TOSTNET_EXCEEDS_TOTAL_FAILURE)
        if total_state.lower > tostnet_state.upper:
            return SegmentClassification(DEFINITELY_AUCTION_ACTIVE, None)
        return SegmentClassification(NOT_PROVEN, None)

    # Every remaining combination involves a dash that does not trigger the
    # structural-impossibility check above (dash/dash, or a numeric total
    # with a dash ToSTNeT cell). A dash never, by itself, proves activity.
    return SegmentClassification(NOT_PROVEN, None)


def era_for_date(date: str) -> Optional[str]:
    if ERA_PRE_START <= date <= ERA_PRE_END:
        return ERA_PRE
    if ERA_POST_START <= date <= ERA_POST_END:
        return ERA_POST
    return None


def required_segments_for_era(era: str) -> tuple:
    try:
        return REQUIRED_SEGMENTS_BY_ERA[era]
    except KeyError as exc:
        raise ValueError(f"unknown era: {era!r}") from exc


def expected_unit(era: str, segment: str, column: str, date: str) -> str:
    """Exact preregistered expected unit for one segment/column/date.

    There is no first-observed-unit learning: this table is the sole
    authority, never inferred from observed data or prior objects.
    """

    if era == ERA_PRE and segment == "Mothers" and column == COLUMN_TOSTNET:
        return SHARES if date < MOTHERS_TOSTNET_UNIT_SPLIT_DATE else THOUSAND_SHARES
    table = _STATIC_REQUIRED_UNIT_PRE if era == ERA_PRE else _STATIC_REQUIRED_UNIT_POST
    try:
        return table[(segment, column)]
    except KeyError as exc:
        raise ValueError(
            f"no frozen unit expectation for era={era!r} segment={segment!r} column={column!r}"
        ) from exc


@dataclass(frozen=True)
class DateClassification:
    status: str
    reason: Optional[str] = None
    segment: Optional[str] = None


def classify_date(
    date: str,
    segment_cells: Mapping[str, Mapping[str, Cell]],
) -> DateClassification:
    """Frozen date-level adjudication (design Section 5.4).

    ``segment_cells`` maps required segment name to a mapping with keys
    ``COLUMN_TOTAL`` and ``COLUMN_TOSTNET``. Segments outside the frozen
    required set for the date's era (for example ``"TOKYO PRO Market"``)
    are never read and never fail the date, per design Section 7.1.

    Required segments are evaluated in frozen order and the first
    encountered data-quality failure is reported (deterministic
    first-failure precedence). A date is ``PROVEN_AUCTION_ACTIVE`` iff at
    least one required segment is ``DEFINITELY_AUCTION_ACTIVE``; otherwise
    it is ``NOT_PROVEN``. There is no status representing proven
    inactivity.
    """

    era = era_for_date(date)
    if era is None:
        return DateClassification(DQ, DATE_OUTSIDE_KNOWN_ERA_FAILURE, None)

    any_active = False
    for segment in required_segments_for_era(era):
        cells = segment_cells.get(segment)
        if cells is None or COLUMN_TOTAL not in cells or COLUMN_TOSTNET not in cells:
            return DateClassification(DQ, MISSING_REQUIRED_SEGMENT_FAILURE, segment)
        total_unit = expected_unit(era, segment, COLUMN_TOTAL, date)
        tostnet_unit = expected_unit(era, segment, COLUMN_TOSTNET, date)
        result = classify_segment(cells[COLUMN_TOTAL], cells[COLUMN_TOSTNET], total_unit, tostnet_unit)
        if result.status == DQ:
            return DateClassification(DQ, result.reason, segment)
        if result.status == DEFINITELY_AUCTION_ACTIVE:
            any_active = True

    return DateClassification(PROVEN_AUCTION_ACTIVE if any_active else NOT_PROVEN)


# --- SOURCE_B physical object-bundle validation (2022-04 two-part rule) ----
def _month_sequence(start_year: int, start_month: int, end_year: int, end_month: int) -> tuple:
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return tuple(months)


REQUIRED_LOGICAL_MONTHS = _month_sequence(2017, 1, 2026, 1)
assert len(REQUIRED_LOGICAL_MONTHS) == LOGICAL_COVERAGE_MONTH_COUNT

APRIL_2022_LOGICAL_MONTH = "2022-04"
PRE_APRIL_1_REFERENCE_OBJECT = "PRE_APRIL_1_REFERENCE_OBJECT"
NORMAL_MONTHLY_REPORT2_OBJECT = "NORMAL_MONTHLY_REPORT2_OBJECT"


def required_source_b_object_parts(logical_month: str) -> tuple:
    if logical_month == APRIL_2022_LOGICAL_MONTH:
        return (PRE_APRIL_1_REFERENCE_OBJECT, NORMAL_MONTHLY_REPORT2_OBJECT)
    return (NORMAL_MONTHLY_REPORT2_OBJECT,)


assert (
    sum(len(required_source_b_object_parts(m)) for m in REQUIRED_LOGICAL_MONTHS)
    == REQUIRED_PHYSICAL_SOURCE_B_OBJECT_COUNT
)


@dataclass(frozen=True)
class ObjectBundleValidation:
    status: str


def validate_source_b_object_bundle(
    logical_month: str, present_parts: Sequence[str]
) -> ObjectBundleValidation:
    """Validate a logical month's SOURCE_B object bundle.

    2022-04 is a two-part bundle: the PRE reference object for
    ``2022-04-01`` and the POST normal Report 2 object for ``2022-04-04``
    onward. One physical object never covers April 1 for that month.
    """

    if logical_month not in REQUIRED_LOGICAL_MONTHS:
        return ObjectBundleValidation(OBJECT_BUNDLE_UNKNOWN_MONTH_FAILURE)
    if len(present_parts) != len(set(present_parts)):
        return ObjectBundleValidation(OBJECT_BUNDLE_DUPLICATE_PART_FAILURE)
    if set(present_parts) != set(required_source_b_object_parts(logical_month)):
        return ObjectBundleValidation(OBJECT_BUNDLE_MISSING_OR_UNEXPECTED_PART_FAILURE)
    return ObjectBundleValidation(OBJECT_BUNDLE_OK)


# --- SOURCE_C and frozen cross-source relation/sentinel validation ---------
def source_c_confirmed_exception_set(
    auction_market_had_no_execution: bool,
    tostnet_orders_received_by_0856_had_executions: bool,
) -> frozenset:
    """Frozen SOURCE_C confirmation (design Section 4.3).

    Both required assertions must be present for the fixed, preregistered
    document date to be confirmed. The date itself is never a caller
    parameter and is never injected; it is the fixed module constant
    ``SOURCE_C_FIXED_DATE_SET``.
    """

    if auction_market_had_no_execution and tostnet_orders_received_by_0856_had_executions:
        return SOURCE_C_FIXED_DATE_SET
    return frozenset()


@dataclass(frozen=True)
class RelationEvaluation:
    status: str
    left_diff: frozenset
    right_diff: frozenset
    left_exact_expected: bool
    right_empty: bool
    cross_source_consistent: bool
    sentinel_2020_09_30_proven_active: bool
    sentinel_2020_10_02_proven_active: bool


def evaluate_cross_source_relation(
    scheduled_open_dates: Sequence[str],
    proven_auction_active_dates: Sequence[str],
    source_c_exception_set: Sequence[str],
) -> RelationEvaluation:
    """Frozen cross-source exact-set/sentinel validation (design Section 6).

    Uses only caller-supplied synthetic scheduled-open evidence, SOURCE_B
    proven-active evidence, and a SOURCE_C confirmed exception set. This
    function never materializes a final ``trading_dates`` sequence.
    """

    scheduled = frozenset(scheduled_open_dates)
    proven_active = frozenset(proven_auction_active_dates)
    source_c = frozenset(source_c_exception_set)

    left_diff = scheduled - proven_active
    right_diff = proven_active - scheduled
    left_exact_expected = left_diff == EXPECTED_UNPROVEN_SET
    right_empty = len(right_diff) == 0
    cross_source_consistent = left_diff == source_c and left_diff == EXPECTED_UNPROVEN_SET

    sentinel_1 = SENTINEL_PROVEN_ACTIVE_DATES[0] in proven_active
    sentinel_2 = SENTINEL_PROVEN_ACTIVE_DATES[1] in proven_active

    overall_pass = (
        left_exact_expected
        and right_empty
        and cross_source_consistent
        and sentinel_1
        and sentinel_2
    )
    status = RELATION_PASS if overall_pass else RELATION_FAILURE

    return RelationEvaluation(
        status=status,
        left_diff=left_diff,
        right_diff=right_diff,
        left_exact_expected=left_exact_expected,
        right_empty=right_empty,
        cross_source_consistent=cross_source_consistent,
        sentinel_2020_09_30_proven_active=sentinel_1,
        sentinel_2020_10_02_proven_active=sentinel_2,
    )
