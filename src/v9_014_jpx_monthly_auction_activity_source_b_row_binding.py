"""Deterministic OFFLINE SOURCE_B extracted-row/physical-object binding for
V9_014.

This module binds caller-supplied, already-extracted SYNTHETIC SOURCE_B
daily rows to their frozen physical-object identity (a logical month plus
one of its frozen required object parts), then -- only after that binding
succeeds -- classifies every bound row through the existing, unchanged core
``classify_date``. It performs NO PDF byte handling, NO PDF library or OCR
use, NO raw unit-cell text normalization, NO network request, and NO
filesystem/raw-lock production-state read; it is not a real runner.

Cell values reuse the existing core ``Cell`` types (``BlankCell``,
``DashCell``, ``MalformedCell``, ``NumericCell``) unchanged; a
``NumericCell``'s ``unit_tokens`` are already-normalized exact semantic
tokens supplied by the caller, never normalized or tokenized here. Unit
resolution, interval arithmetic, required segments, era rules, dash
semantics, and date-level classification are never reimplemented -- this
module calls the existing frozen ``classify_date`` verbatim for every bound
row and preserves its result exactly, including ``DQ``; a ``DQ``
classification is never repaired, dropped, or redrawn.

Object-binding failure makes a successful, classification-bearing result
unreachable: :class:`ObjectRowBindingResult.date_classifications` is
``None`` on every failure status and populated only when
``status == OBJECT_ROW_BINDING_OK``. The closed object-binding status
vocabulary below introduces no new scientific classification -- there is no
``PROVEN_INACTIVE`` or equivalent state anywhere in this module.

This module does not validate real PDF table layout or schema (that remains
a later, separate checkpoint), does not resolve V9_014 design LOW_1's
deferred raw unit-cell text normalization, and never materializes
``trading_dates``, a relation result, a T0/backtest/model outcome, or any
profitability claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date
from typing import Mapping, Optional, Sequence

from src.v9_014_jpx_monthly_auction_activity_authority import (
    APRIL_2022_LOGICAL_MONTH,
    ERA_POST,
    ERA_PRE_END,
    NORMAL_MONTHLY_REPORT2_OBJECT,
    PRE_APRIL_1_REFERENCE_OBJECT,
    REQUIRED_LOGICAL_MONTHS,
    Cell,
    DateClassification,
    classify_date,
    era_for_date,
    required_source_b_object_parts,
)

__all__ = [
    "OBJECT_ROW_BINDING_OK",
    "INVALID_INPUT_FAILURE",
    "INVALID_LOGICAL_MONTH_FAILURE",
    "INVALID_OBJECT_PART_FAILURE",
    "MALFORMED_ROW_DATE_FAILURE",
    "DUPLICATE_ROW_DATE_FAILURE",
    "ROW_DATE_OUTSIDE_LOGICAL_MONTH_FAILURE",
    "ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_FAILURE",
    "EMPTY_ROW_COLLECTION_FAILURE",
    "APRIL_PRE_REQUIRED_DATE_MISSING_FAILURE",
    "APRIL_PRE_WRONG_DATE_FAILURE",
    "APRIL_1_2022_REQUIRED_DATE",
    "SourceBDailyRow",
    "SourceBObjectRowBundle",
    "ObjectRowBindingResult",
    "bind_source_b_object_rows",
]

# --- Closed object-binding status vocabulary --------------------------------
OBJECT_ROW_BINDING_OK = "OBJECT_ROW_BINDING_OK"
INVALID_INPUT_FAILURE = "INVALID_INPUT_DQ_FAILURE"
INVALID_LOGICAL_MONTH_FAILURE = "INVALID_LOGICAL_MONTH_DQ_FAILURE"
INVALID_OBJECT_PART_FAILURE = "INVALID_OBJECT_PART_FOR_LOGICAL_MONTH_DQ_FAILURE"
MALFORMED_ROW_DATE_FAILURE = "MALFORMED_ROW_DATE_DQ_FAILURE"
DUPLICATE_ROW_DATE_FAILURE = "DUPLICATE_ROW_DATE_DQ_FAILURE"
ROW_DATE_OUTSIDE_LOGICAL_MONTH_FAILURE = "ROW_DATE_OUTSIDE_LOGICAL_MONTH_DQ_FAILURE"
ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_FAILURE = "ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_DQ_FAILURE"
EMPTY_ROW_COLLECTION_FAILURE = "EMPTY_ROW_COLLECTION_DQ_FAILURE"
APRIL_PRE_REQUIRED_DATE_MISSING_FAILURE = "APRIL_PRE_REQUIRED_DATE_MISSING_DQ_FAILURE"
APRIL_PRE_WRONG_DATE_FAILURE = "APRIL_PRE_WRONG_DATE_DQ_FAILURE"

# The PRE_APRIL_1_REFERENCE_OBJECT's sole permitted date, reused from the
# core module's own frozen ERA_PRE_END rather than a duplicated literal.
APRIL_1_2022_REQUIRED_DATE = ERA_PRE_END

_STRICT_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_strict_iso_date(value: object) -> bool:
    """Strict YYYY-MM-DD: parseable and round-trip isoformat() == value."""

    if not isinstance(value, str) or not _STRICT_ISO_DATE_RE.fullmatch(value):
        return False
    try:
        parsed = _date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


# --- Closed typed synthetic inputs ------------------------------------------
@dataclass(frozen=True)
class SourceBDailyRow:
    """One already-extracted synthetic SOURCE_B daily row.

    ``segment_cells`` is exactly the ``Mapping[str, Mapping[str, Cell]]``
    shape already consumed by the core ``classify_date`` -- reused
    unchanged, never redefined here.
    """

    date: str
    segment_cells: Mapping[str, Mapping[str, Cell]]


@dataclass(frozen=True)
class SourceBObjectRowBundle:
    """The complete row collection claimed to belong to one frozen SOURCE_B
    physical-object identity (a logical month plus one required object
    part)."""

    logical_month: str
    object_part: str
    rows: Sequence[SourceBDailyRow]


@dataclass(frozen=True)
class ObjectRowBindingResult:
    status: str
    logical_month: Optional[str] = None
    object_part: Optional[str] = None
    date: Optional[str] = None
    date_classifications: Optional[Mapping[str, DateClassification]] = None


def _failure(
    status: str,
    *,
    logical_month: Optional[str] = None,
    object_part: Optional[str] = None,
    date: Optional[str] = None,
) -> ObjectRowBindingResult:
    return ObjectRowBindingResult(
        status=status, logical_month=logical_month, object_part=object_part, date=date
    )


def bind_source_b_object_rows(bundle: SourceBObjectRowBundle) -> ObjectRowBindingResult:
    """Bind a claimed SOURCE_B physical-object row collection to its frozen
    identity, then classify every bound row via the existing core
    ``classify_date``.

    Binding order (deterministic, first-failure precedence):

    1. ``logical_month`` must be one of the frozen ``REQUIRED_LOGICAL_MONTHS``;
       ``object_part`` must be exactly one of
       ``required_source_b_object_parts(logical_month)``.
    2. Rows are processed in order; for each row its date must be a strict
       ``YYYY-MM-DD`` string (parseable and round-tripping through
       ``date.fromisoformat().isoformat()``), and must not duplicate an
       earlier row's date in this same bundle.
    3. For ``NORMAL_MONTHLY_REPORT2_OBJECT``: the collection must be
       non-empty; every row's calendar month must equal ``logical_month``
       exactly; for ``logical_month == "2022-04"`` every row must fall in
       ``ERA_POST`` (2022-04-04 onward -- 2022-04-01 is forbidden in the
       normal object); for every other logical month every row must fall in
       a known frozen era.
    4. For ``PRE_APRIL_1_REFERENCE_OBJECT`` (permitted only when
       ``logical_month == "2022-04"``, already enforced by step 1): every
       row's date must equal exactly ``APRIL_1_2022_REQUIRED_DATE``
       (2022-04-01), and that date must actually be present in the
       collection.

    Only after every check above passes does this function call
    ``classify_date`` once per bound row, in row order, and return their
    exact results unchanged under ``date_classifications`` -- a mapping
    keyed by date, insertion-ordered to match the supplied row order. Any
    binding failure returns ``date_classifications=None``.
    """

    if not isinstance(bundle, SourceBObjectRowBundle):
        return _failure(INVALID_INPUT_FAILURE)

    logical_month = bundle.logical_month
    object_part = bundle.object_part
    rows = bundle.rows

    if not isinstance(logical_month, str) or logical_month not in REQUIRED_LOGICAL_MONTHS:
        return _failure(INVALID_LOGICAL_MONTH_FAILURE, logical_month=logical_month)

    required_parts = required_source_b_object_parts(logical_month)
    if not isinstance(object_part, str) or object_part not in required_parts:
        return _failure(
            INVALID_OBJECT_PART_FAILURE, logical_month=logical_month, object_part=object_part
        )

    if not isinstance(rows, (list, tuple)):
        return _failure(
            INVALID_INPUT_FAILURE, logical_month=logical_month, object_part=object_part
        )
    for row in rows:
        if not isinstance(row, SourceBDailyRow):
            return _failure(
                INVALID_INPUT_FAILURE, logical_month=logical_month, object_part=object_part
            )

    seen_dates = set()
    for row in rows:
        if not _is_strict_iso_date(row.date):
            return _failure(
                MALFORMED_ROW_DATE_FAILURE,
                logical_month=logical_month,
                object_part=object_part,
                date=row.date if isinstance(row.date, str) else None,
            )
        if row.date in seen_dates:
            return _failure(
                DUPLICATE_ROW_DATE_FAILURE,
                logical_month=logical_month,
                object_part=object_part,
                date=row.date,
            )
        seen_dates.add(row.date)

    if object_part == NORMAL_MONTHLY_REPORT2_OBJECT:
        if not rows:
            return _failure(
                EMPTY_ROW_COLLECTION_FAILURE, logical_month=logical_month, object_part=object_part
            )
        for row in rows:
            if row.date[:7] != logical_month:
                return _failure(
                    ROW_DATE_OUTSIDE_LOGICAL_MONTH_FAILURE,
                    logical_month=logical_month,
                    object_part=object_part,
                    date=row.date,
                )
            row_era = era_for_date(row.date)
            if logical_month == APRIL_2022_LOGICAL_MONTH:
                if row_era != ERA_POST:
                    return _failure(
                        ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_FAILURE,
                        logical_month=logical_month,
                        object_part=object_part,
                        date=row.date,
                    )
            elif row_era is None:
                return _failure(
                    ROW_DATE_WRONG_ERA_FOR_OBJECT_PART_FAILURE,
                    logical_month=logical_month,
                    object_part=object_part,
                    date=row.date,
                )
    elif object_part == PRE_APRIL_1_REFERENCE_OBJECT:
        for row in rows:
            if row.date != APRIL_1_2022_REQUIRED_DATE:
                return _failure(
                    APRIL_PRE_WRONG_DATE_FAILURE,
                    logical_month=logical_month,
                    object_part=object_part,
                    date=row.date,
                )
        if APRIL_1_2022_REQUIRED_DATE not in seen_dates:
            return _failure(
                APRIL_PRE_REQUIRED_DATE_MISSING_FAILURE,
                logical_month=logical_month,
                object_part=object_part,
            )
    else:  # pragma: no cover - unreachable given the step-1 membership check
        return _failure(
            INVALID_OBJECT_PART_FAILURE, logical_month=logical_month, object_part=object_part
        )

    date_classifications = {row.date: classify_date(row.date, row.segment_cells) for row in rows}

    return ObjectRowBindingResult(
        status=OBJECT_ROW_BINDING_OK,
        logical_month=logical_month,
        object_part=object_part,
        date_classifications=date_classifications,
    )
