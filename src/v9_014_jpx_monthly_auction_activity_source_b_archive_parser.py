"""Deterministic OFFLINE SOURCE_B archive HTML candidate extraction for V9_014.

This module parses caller-supplied, already-locked/synthetic JPX archive HTML
bytes into the already-reviewed typed locator candidates defined in
``src.v9_014_jpx_monthly_auction_activity_source_b_locator``:
``RootYearCandidate``, ``MonthlyReportCandidate``, and
``AprilPreReferenceCandidate``. It performs NO network request and NO PDF
parsing; it is the offline bridge between locked archive-page HTML bytes and
the existing, unchanged locator resolvers.

For archive-locator HTML, this module reuses -- verbatim, by direct import,
never reimplemented -- the exact already-reviewed V9_006 parsing mechanics
in ``src.v9_005_stage_a_jpx_probe``: the private ``_MonthlyStatisticsHtmlParser``
/ ``_parse_monthly_statistics_html`` (bytes-only input; strict UTF-8 decode,
fail closed on decode failure; ``html.parser.HTMLParser`` with
``convert_charrefs=True``; malformed or nested relevant table/tr/td/th/a
structure fails closed) and ``_resolve_locked_page_link`` (relative href
resolution only via ``urllib.parse.urljoin(exact_locked_page_url, raw_href)``,
followed by the existing reviewed ``validate_jpx_url`` on every resolved
URL). Locator semantic text normalization is exactly
``" ".join(raw_text.split())`` (already applied by the reused parser's
``.text`` property and anchor-text accumulation) -- no lowercasing,
case-folding, fuzzy matching, substring matching, or invented alias is ever
applied. This module's parsing/normalization authority is scoped only to
archive-locator HTML; it does not touch, resolve, or invent design LOW_1's
deferred PDF unit-cell text normalization.

Multiplicity is deliberately preserved, never manually disambiguated here:
every anchor or table-cell href whose normalized label(s) exactly match is
emitted as its own separate candidate object. Zero, one, or many resulting
candidates are all returned as-is; the already-reviewed locator resolvers
(``resolve_source_b_year_page``, ``resolve_source_b_normal_month_object``,
``resolve_source_b_april_pre_object``, unchanged by this module) remain the
sole place that fails closed on zero or multiple matches. A genuinely
malformed input (non-bytes, decode failure, invalid table/anchor structure,
a matching candidate with a missing/empty href, an out-of-bounds table
column, or a resolved URL that fails the reviewed domain/scheme policy)
raises ``V9005StageABlocked`` -- the exact same reviewed exception class and
``IMPLEMENTATION_FAILURE`` reason already used by the mechanics this module
reuses -- rather than silently dropping the offending candidate.

It performs NO network request, NO filesystem access, NO subprocess, NO
environment/credential read, and NO durable-state access; it is not a real
runner. It produces no ``DateClassification``, no relation result, no
``trading_dates``, and no profitability output, and it never downloads,
reads, or parses a PDF.

``extract_root_year_candidates`` requires ``root_url`` to equal the frozen
``SOURCE_B_ARCHIVE_ROOT`` exactly -- not merely pass the same-JPX-domain
policy, and never normalized, redirect-equated, path/query-rewritten, or
case-folded into an equivalent root; a same-domain but otherwise unrelated
JPX page can never produce a ``RootYearCandidate``.
``extract_april_pre_candidates`` requires an explicit, caller-supplied
``selected_year`` keyword argument equal exactly to ``2022`` (derived from
the single frozen ``APRIL_2022_LOGICAL_MONTH`` source of truth, never a
duplicated literal); the year is never inferred from
``selected_year_page_url``'s path or an anchor's text, and any other year
context -- 2019, 2020, 2023, or a non-``int``/``bool`` value -- fails closed
before any HTML parsing occurs, even given an otherwise perfect exact
reference anchor and a valid JPX parent URL.
"""

from __future__ import annotations

from typing import Tuple

from src.v9_005_stage_a_jpx_probe import (
    IMPLEMENTATION_FAILURE,
    V9005StageABlocked,
    _parse_monthly_statistics_html,
    _parse_year_month,
    _resolve_locked_page_link,
)
from src.v9_014_jpx_monthly_auction_activity_authority import (
    APRIL_2022_LOGICAL_MONTH,
    REQUIRED_LOGICAL_MONTHS,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    APRIL_1_2022_REFERENCE_LABEL,
    SOURCE_B_ARCHIVE_ROOT,
    SOURCE_B_REPORT,
    AprilPreReferenceCandidate,
    MonthlyReportCandidate,
    RootYearCandidate,
)

__all__ = [
    "extract_root_year_candidates",
    "extract_normal_month_candidates",
    "extract_april_pre_candidates",
]

# The exact required year context for the April-2022 PRE special-reference
# branch, derived from the single frozen source of truth
# (APRIL_2022_LOGICAL_MONTH) rather than a duplicated literal.
REQUIRED_APRIL_PRE_YEAR, _REQUIRED_APRIL_PRE_MONTH_NUMBER = _parse_year_month(
    APRIL_2022_LOGICAL_MONTH
)


def extract_root_year_candidates(
    root_bytes: bytes, root_url: str, requested_year: int
) -> Tuple[RootYearCandidate, ...]:
    """Extract every official-root candidate whose exact normalized visible
    anchor text equals ``str(requested_year)``.

    ``root_url`` must equal the frozen :data:`SOURCE_B_ARCHIVE_ROOT` exactly
    -- not merely pass the same-JPX-domain policy, and never normalized,
    redirect-equated, query/path-rewritten, or case-folded into an
    equivalent root. A same-domain but otherwise unrelated JPX page can
    never produce a candidate. Every relative href is resolved against this
    exact bound root; no archive-year URL is ever constructed. Multiplicity
    is preserved: zero, one, or many resulting candidates are all returned,
    deferring the uniqueness decision to :func:`resolve_source_b_year_page`.
    """

    if isinstance(requested_year, bool) or not isinstance(requested_year, int):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if root_url != SOURCE_B_ARCHIVE_ROOT:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    parser = _parse_monthly_statistics_html(root_bytes)
    matches = [(href, text) for href, text in parser.anchors if text == str(requested_year)]

    candidates = []
    for href, _text in matches:
        resolved_url = _resolve_locked_page_link(root_url, href)
        candidates.append(RootYearCandidate(label=str(requested_year), href=resolved_url))
    return tuple(candidates)


def extract_normal_month_candidates(
    year_page_bytes: bytes,
    selected_year_page_url: str,
    requested_month: str,
    *,
    selected_year: int,
) -> Tuple[MonthlyReportCandidate, ...]:
    """Extract every normal Report-2 monthly candidate from a selected
    archive-year page's locked HTML bytes.

    ``requested_month`` must be one of the frozen ``REQUIRED_LOGICAL_MONTHS``,
    and its year component must equal the caller-supplied ``selected_year``
    context -- never inferred from ``selected_year_page_url``'s path. For
    each table independently, every row containing a cell whose exact
    normalized text equals :data:`SOURCE_B_REPORT` is matched against every
    ``th`` column in that SAME table whose exact normalized text equals
    ``requested_month``; a semantic match in one table can never bind to a
    column header in a different table. Every href present at each such
    row/column intersection becomes its own separate candidate -- duplicate
    rows, duplicate columns, and duplicate hrefs are all preserved rather
    than silently disambiguated, deferring uniqueness to
    :func:`resolve_source_b_normal_month_object`. A structurally
    out-of-bounds intersection (a row shorter than the matched column index)
    fails closed rather than being silently skipped.
    """

    if not isinstance(requested_month, str) or requested_month not in REQUIRED_LOGICAL_MONTHS:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    year, _month = _parse_year_month(requested_month)
    if isinstance(selected_year, bool) or not isinstance(selected_year, int) or year != selected_year:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    parser = _parse_monthly_statistics_html(year_page_bytes)

    matching_rows = []
    for table in parser.tables:
        for row in table.rows:
            if any(cell.text == SOURCE_B_REPORT for cell in row):
                matching_rows.append((table, row))

    candidates = []
    for table, semantic_row in matching_rows:
        month_columns = [
            index
            for row in table.rows
            for index, cell in enumerate(row)
            if cell.tag == "th" and cell.text == requested_month
        ]
        for column in month_columns:
            if column >= len(semantic_row):
                raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
            for href in semantic_row[column].hrefs:
                resolved_url = _resolve_locked_page_link(selected_year_page_url, href)
                candidates.append(
                    MonthlyReportCandidate(
                        parent_year_page_url=selected_year_page_url,
                        report_label=SOURCE_B_REPORT,
                        month=requested_month,
                        href=resolved_url,
                    )
                )
    return tuple(candidates)


def extract_april_pre_candidates(
    year_page_bytes: bytes,
    selected_year_page_url: str,
    *,
    selected_year: int,
) -> Tuple[AprilPreReferenceCandidate, ...]:
    """Extract every 2022-04 PRE special-reference-branch candidate from a
    selected archive-year page's locked HTML bytes.

    ``selected_year`` is the caller-supplied semantic year context from the
    already-selected traversal (never inferred from ``selected_year_page_url``'s
    path, an anchor's text, or any other pattern) and must equal exactly
    :data:`REQUIRED_APRIL_PRE_YEAR` (``2022``), as a genuine ``int`` and not
    a ``bool``; any other value fails closed before any HTML parsing occurs.
    A caller-declared 2019/2020/2023/etc. context can never yield a
    candidate, even given an otherwise perfect exact reference anchor and a
    valid JPX parent URL. Candidates are matched only by exact normalized
    anchor-text equality to :data:`APRIL_1_2022_REFERENCE_LABEL` -- no
    case-changed, substring, or fuzzy variant is ever accepted.
    ``selected_year_page_url`` is the exact locked page URL every relative
    href is resolved against, and it is preserved exactly as each
    candidate's ``parent_year_page_url``; no URL is ever inferred from
    anchor text or path. Multiplicity is preserved, deferring the
    uniqueness decision to :func:`resolve_source_b_april_pre_object`.
    """

    if (
        isinstance(selected_year, bool)
        or not isinstance(selected_year, int)
        or selected_year != REQUIRED_APRIL_PRE_YEAR
    ):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    parser = _parse_monthly_statistics_html(year_page_bytes)
    matches = [
        (href, text) for href, text in parser.anchors if text == APRIL_1_2022_REFERENCE_LABEL
    ]

    candidates = []
    for href, _text in matches:
        resolved_url = _resolve_locked_page_link(selected_year_page_url, href)
        candidates.append(
            AprilPreReferenceCandidate(
                parent_year_page_url=selected_year_page_url,
                label=APRIL_1_2022_REFERENCE_LABEL,
                href=resolved_url,
            )
        )
    return tuple(candidates)
