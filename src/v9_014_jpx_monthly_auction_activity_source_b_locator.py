"""Deterministic OFFLINE SOURCE_B archive-locator contract for V9_014.

This module implements ONLY the pure, deterministic locator selection logic
frozen by the V9_014 design (see
``V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR_DESIGN_DRAFT.md``
Section 7.4, frozen at design git SHA
``efee3d0efca368645c00aeed63cb8e0637cd3672``, design blob SHA
``2bbacbf37ab961d1cbf416b7fd476db18778c5b7``), following the already-frozen
official Monthly Statistics archive traversal discipline in
``V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md`` (root -> required year
via the official archive selector -> exact semantic report label -> required
month -> unique same-domain object).

Frozen traversal:

    root=https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html
      -> required year (candidate links from the official-root snapshot only)
      -> exact Report 2 "Stock Trading Volume & Value"
      -> required month
      -> unique same-domain PDF object

2022-04 requires TWO physical objects: the special official
``(Reference) Status on April 1, 2022`` branch for the PRE part, and the
normal Report 2 monthly object (resolved through the same normal-month path
as every other month) for the POST part. Archive numbering is never
hardcoded; no child URL is ever guessed, pattern-derived, or reconstructed
from memory. Zero or multiple matching candidates for a required slot fail
closed.

This module operates only on caller-supplied, already-extracted SYNTHETIC
link/semantic records -- each candidate carries an exact href and exact
normalized traversal-semantic label(s), already fully resolved to an
absolute URL. It performs NO raw HTML normalization or parsing, NO page or
file read, NO PDF inspection or table extraction, and NO PDF unit-cell
normalization (design LOW_1 explicitly defers raw unit-cell normalization to
a later stage; this module does not invent it). Label/report/month matching
uses exact string equality only -- no fuzzy matching, case-folding,
substring matching, or invented aliases.

It performs NO network request, NO filesystem access, NO subprocess, NO
environment/credential read, and NO durable-state access; it is not a real
runner. It reuses the existing reviewed ``validate_jpx_url`` domain/scheme
validator from ``src.v9_005_stage_a_jpx_probe`` verbatim (HTTPS, exactly
``jpx.co.jp`` or a subdomain, no credentials/nonstandard port/fragment) and
does not weaken it. It produces no ``DateClassification``, no relation
result, no ``trading_dates``, and no profitability output.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Optional, Sequence

from src.v9_005_stage_a_jpx_probe import V9005StageABlocked, validate_jpx_url
from src.v9_014_jpx_monthly_auction_activity_authority import (
    APRIL_2022_LOGICAL_MONTH,
    LOGICAL_COVERAGE_MONTH_COUNT,
    NORMAL_MONTHLY_REPORT2_OBJECT,
    PRE_APRIL_1_REFERENCE_OBJECT,
    REQUIRED_LOGICAL_MONTHS,
    REQUIRED_PHYSICAL_SOURCE_B_OBJECT_COUNT,
    SOURCE_B_ARCHIVE_ROOT,
    SOURCE_B_OBJECT_FORMAT,
    SOURCE_B_REPORT,
    required_source_b_object_parts,
)

__all__ = [
    "SOURCE_B_ARCHIVE_ROOT",
    "SOURCE_B_REPORT",
    "SOURCE_B_OBJECT_FORMAT",
    "APRIL_1_2022_REFERENCE_LABEL",
    "LOCATOR_OK",
    "LOCATOR_ZERO_CANDIDATES_FAILURE",
    "LOCATOR_MULTIPLE_CANDIDATES_FAILURE",
    "LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE",
    "LOCATOR_NON_PDF_FAILURE",
    "LOCATOR_INVALID_INPUT_FAILURE",
    "RootYearCandidate",
    "MonthlyReportCandidate",
    "AprilPreReferenceCandidate",
    "LocatorResult",
    "SourceBRequestPlanEntry",
    "resolve_source_b_year_page",
    "resolve_source_b_normal_month_object",
    "resolve_source_b_april_pre_object",
    "plan_source_b_requests",
]

# The special official branch used only for the 2022-04 PRE part (design
# Section 7.3/7.4). Exact literal label; no alternate wording is accepted.
APRIL_1_2022_REFERENCE_LABEL = "(Reference) Status on April 1, 2022"

# --- Closed locator result status codes ------------------------------------
LOCATOR_OK = "LOCATOR_OK"
LOCATOR_ZERO_CANDIDATES_FAILURE = "LOCATOR_ZERO_CANDIDATES_DQ_FAILURE"
LOCATOR_MULTIPLE_CANDIDATES_FAILURE = "LOCATOR_MULTIPLE_CANDIDATES_DQ_FAILURE"
LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE = "LOCATOR_OFF_DOMAIN_OR_INVALID_URL_DQ_FAILURE"
LOCATOR_NON_PDF_FAILURE = "LOCATOR_NON_PDF_DQ_FAILURE"
LOCATOR_INVALID_INPUT_FAILURE = "LOCATOR_INVALID_INPUT_DQ_FAILURE"


# --- Closed typed synthetic inputs ------------------------------------------
@dataclass(frozen=True)
class RootYearCandidate:
    """One candidate link from the official-root archive-year snapshot.

    ``label`` is the candidate's exact normalized selector text (for example
    the literal year label ``"2019"``); ``href`` is its exact, already fully
    resolved absolute URL.
    """

    label: str
    href: str


@dataclass(frozen=True)
class MonthlyReportCandidate:
    """One candidate object from a selected year's archive snapshot.

    ``report_label`` and ``month`` are the candidate's exact normalized
    traversal-semantic labels (never fuzzy-matched); ``href`` is its exact,
    already fully resolved absolute URL.
    """

    report_label: str
    month: str
    href: str


@dataclass(frozen=True)
class AprilPreReferenceCandidate:
    """One candidate object for the special 2022-04 PRE reference branch."""

    label: str
    href: str


@dataclass(frozen=True)
class LocatorResult:
    status: str
    url: Optional[str] = None


def _finalize_selected_url(href: object, *, require_pdf: bool) -> LocatorResult:
    if not isinstance(href, str) or not href:
        return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)
    try:
        validated_url = validate_jpx_url(href)
    except V9005StageABlocked:
        return LocatorResult(LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE)
    if require_pdf and not urllib.parse.urlparse(validated_url).path.endswith(".pdf"):
        return LocatorResult(LOCATOR_NON_PDF_FAILURE)
    return LocatorResult(LOCATOR_OK, url=validated_url)


def _select_unique(candidates: Sequence, predicate) -> LocatorResult | list:
    """Return the single matching candidate, or a closed-failure result.

    Zero or multiple candidate links for a required slot fail closed; this
    function never chooses among multiple matches after the fact.
    """

    matched = [c for c in candidates if predicate(c)]
    if len(matched) == 0:
        return LocatorResult(LOCATOR_ZERO_CANDIDATES_FAILURE)
    if len(matched) > 1:
        return LocatorResult(LOCATOR_MULTIPLE_CANDIDATES_FAILURE)
    return matched


def resolve_source_b_year_page(
    root_candidates: Sequence[RootYearCandidate], requested_year: int
) -> LocatorResult:
    """Resolve exactly one archive-year page URL from official-root candidates.

    The required year comes only from candidate links supplied by the
    official-root snapshot; no year or archive URL is ever constructed.
    Year-page candidates match by exact label equality to the literal year
    string only -- no fuzzy or partial match.
    """

    if isinstance(requested_year, bool) or not isinstance(requested_year, int):
        return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)
    if not isinstance(root_candidates, (list, tuple)):
        return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)
    for candidate in root_candidates:
        if not isinstance(candidate, RootYearCandidate):
            return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)

    selected = _select_unique(root_candidates, lambda c: c.label == str(requested_year))
    if isinstance(selected, LocatorResult):
        return selected
    return _finalize_selected_url(selected[0].href, require_pdf=False)


def resolve_source_b_normal_month_object(
    year_page_candidates: Sequence[MonthlyReportCandidate], requested_month: str
) -> LocatorResult:
    """Resolve exactly one normal Report-2 monthly PDF object.

    The object comes only from the selected year's snapshot, matched by the
    exact Report 2 "Stock Trading Volume & Value" semantic label and the
    exact requested month -- both by exact string equality only. This is the
    same normal-month path used for every required logical month, including
    the POST part of 2022-04; the April PRE special branch is resolved
    separately by :func:`resolve_source_b_april_pre_object` and never
    substituted here.
    """

    if not isinstance(requested_month, str) or requested_month not in REQUIRED_LOGICAL_MONTHS:
        return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)
    if not isinstance(year_page_candidates, (list, tuple)):
        return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)
    for candidate in year_page_candidates:
        if not isinstance(candidate, MonthlyReportCandidate):
            return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)

    selected = _select_unique(
        year_page_candidates,
        lambda c: c.report_label == SOURCE_B_REPORT and c.month == requested_month,
    )
    if isinstance(selected, LocatorResult):
        return selected
    return _finalize_selected_url(selected[0].href, require_pdf=True)


def resolve_source_b_april_pre_object(
    candidates: Sequence[AprilPreReferenceCandidate],
) -> LocatorResult:
    """Resolve exactly one 2022-04 PRE special-reference-branch PDF object.

    The PRE object comes only from the supplied exact special-reference
    branch candidates, matched by exact label equality to
    :data:`APRIL_1_2022_REFERENCE_LABEL` only. This is entirely separate
    from the normal April Report-2 object resolved by
    :func:`resolve_source_b_normal_month_object`.
    """

    if not isinstance(candidates, (list, tuple)):
        return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)
    for candidate in candidates:
        if not isinstance(candidate, AprilPreReferenceCandidate):
            return LocatorResult(LOCATOR_INVALID_INPUT_FAILURE)

    selected = _select_unique(candidates, lambda c: c.label == APRIL_1_2022_REFERENCE_LABEL)
    if isinstance(selected, LocatorResult):
        return selected
    return _finalize_selected_url(selected[0].href, require_pdf=True)


# --- Deterministic request planner (design Sections 3 and 7.3) -------------
@dataclass(frozen=True)
class SourceBRequestPlanEntry:
    logical_month: str
    kind: str


def plan_source_b_requests() -> tuple:
    """Deterministic plan of every required SOURCE_B physical resolution
    request across the frozen coverage: exactly 109 logical months yielding
    exactly 110 physical requests, with the 2022-04 two-part split. This
    reuses the existing frozen ``required_source_b_object_parts`` from the
    V9_014 core module unchanged; no new coverage or split methodology is
    introduced here.
    """

    entries = []
    for month in REQUIRED_LOGICAL_MONTHS:
        for kind in required_source_b_object_parts(month):
            entries.append(SourceBRequestPlanEntry(logical_month=month, kind=kind))
    return tuple(entries)


assert len(REQUIRED_LOGICAL_MONTHS) == LOGICAL_COVERAGE_MONTH_COUNT
assert len(plan_source_b_requests()) == REQUIRED_PHYSICAL_SOURCE_B_OBJECT_COUNT
