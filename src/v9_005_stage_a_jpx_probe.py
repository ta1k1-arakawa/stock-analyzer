"""V9_005 Stage-A free official JPX metadata/calendar probe implementation.

Implements, without executing any real network request, the exact Stage-A
contract frozen in
`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` and the exact
reviewed locator/inventory contract in
`V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md` and
`V9_006_F1_TERMINAL_SEED_PREFREEZE_AMENDMENT.md`: the seven source
families' heterogeneous `SOURCE_OBJECT_INVENTORY` slot kinds (`MONTHLY`,
`YEAR`, `TERMINAL`, `GLOBAL`), the deterministic 648-record
`MONTHLY_COVERAGE_MATRIX` (F2-F7 x 2017-01..2025-12; F1 has zero monthly
cells and is `TERMINAL_SEED` only), first-complete-payload raw locking with
full provenance, the reconstruction/validation evidence items, and the
exact `FREE_JPX_METADATA_PROBE_PASS` conjunction. Importing or
unit-testing this module performs no network I/O; production execution is
gated behind `scripts/run_v9_005_stage_a_jpx_probe.py` and the single
atomic `scripts/run_v9_005_stage_a_jpx_probe.ps1` entrypoint, neither of
which this implementation task authorizes to run against a real network.

Source-locator discipline (V9_006 contract item 4): this module never
guesses a JPX endpoint URL. Every source family's locator strategy in
`LOCATOR_STRATEGIES` is reused verbatim from the exact roots/semantic
traversal rules bound in
`V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md`: F1's terminal
listed-issues page + `data_j.xls` extraction; F2/F4's shared Monthly
Statistics root with semantic-row traversal; F3's delisted-company
archive-year root; F5's listing/co root; F6's TOPIX root; and F7's exact
GPT-bound `{YYYY}{MM:02d}.html` per-month calendar template. No archive
number, mirror, alternate provider, or off-domain locator is ever guessed.

V9_006_HIGH_1 / HIGH_1A / F1_TERMINAL_SEED / STAGE_A_LOCATOR_CONTRACT:
`verify_locator_contract_complete()` is a pre-network methodology-
completeness check, not a claim that any concrete per-month URL is
already known. It verifies that every required slot -- the base
648-record matrix, F1's mandatory `TERMINAL` slot, F2's post-2025 bridge
slots (mechanically derived from the terminal snapshot month `T`), and
F7's envelope slots outside 2017-2025 -- has a *reviewed deterministic
locator strategy* bound in `LOCATOR_STRATEGIES`. It does not, and must
not, require a child URL that can only be discovered by traversing a
locked official JPX root response at real execution time -- that
traversal is real Stage-A network work, gated behind a fresh, separate,
explicit human authorization this task does not create. Before touching
the filesystem, git, or the network at all, `run_stage_a` calls this
check, which raises `V9005StageABlocked(STAGE_A_SOURCE_LOCATOR_CONTRACT_
INCOMPLETE)` -- reported as `failure_class=CHATGPT_DECISION_REQUIRED`,
never `SOURCE_OR_DATA_FEASIBILITY_FAILURE` -- only if some required slot
still has no reviewed strategy at all. `SOURCE_OR_DATA_FEASIBILITY_
FAILURE` remains reserved for a genuine result produced only after real
Stage-A execution actually attempts (and fails) the reviewed traversal.

V9_006_LOCATOR_IMPL_HIGH_1: a complete locator-*strategy* registry is not
the same thing as a complete acquisition *implementation*. No code in this
module yet actually walks a locked official F2-F7 root response to find
each required child object for every base/bridge/envelope slot -- that
traversal-fetch pipeline is separate, future, authorized work. Running
Stage A today would otherwise fetch only the two objects that do have an
implemented fetch path (F1's terminal snapshot, the calendar page) and then
report the remaining 648 slots `MISSING` -- a knowingly incomplete
acquisition run, no better than the knowingly doomed run V9_006_HIGH_1
already forbade. `run_stage_a` therefore also calls
`verify_acquisition_implementation_ready()`, immediately after the locator-
contract check and still before touching the filesystem, git, or the
network, which raises `V9005StageABlocked(STAGE_A_ACQUISITION_
IMPLEMENTATION_INCOMPLETE)` (`failure_class=CHATGPT_DECISION_REQUIRED`)
unconditionally until `ACQUISITION_IMPLEMENTATION_COMPLETE` is flipped to
`True` by that future task.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from src.v7_jpx_calendar import V7JpxCalendarBlocked, parse_jpx_holiday_html
from src.v8c_transport import classify_transport_exception
from src.v9_005_stage_a_semantics import (
    SemanticEvent,
    TerminalIdentityState,
    compute_semantic_validation_result,
)

STUDY = "V9_CROSS_SECTIONAL_CLOSE_AUCTION"
STAGE = "V9_005_STAGE_A_JPX_METADATA_PROBE"
CONFIRMATION = "V9_005_STAGE_A_HUMAN_AUTHORIZE_JPX_METADATA_PROBE"

# --- Signal-grid prefreeze binding (V9_005_HIGH_2B) ------------------------
BOUND_SIGNAL_GRID_PATH = "V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md"
BOUND_SIGNAL_GRID_BLOB_SHA = "9135183b7fc5097602fa40fcda8f1b0448220244"

# --- Allowed network domain (contract item 1) -------------------------------
ALLOWED_HOST_SUFFIX = "jpx.co.jp"

# --- Seven Stage-A source families ------------------------------------------
SOURCE_FAMILY_LISTED_ISSUES_MONTH_END = "LISTED_ISSUES_MONTH_END"
SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT = "MONTHLY_STATISTICS_CHANGES_REPORT"
SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE = "DELISTED_COMPANY_ARCHIVE"
SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE = "EX_RIGHTS_SPLIT_RATIO_ARCHIVE"
SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS = "MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS"
SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE = "TOPIX_HISTORICAL_INDEX_VALUE"
SOURCE_FAMILY_JPX_CALENDAR = "JPX_CALENDAR_MARKET_BUSINESS_DAY"

SOURCE_FAMILIES: tuple[str, ...] = (
    SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE,
    SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS,
    SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
    SOURCE_FAMILY_JPX_CALENDAR,
)

# Per V9_006_F1_TERMINAL_SEED_PREFREEZE_AMENDMENT: F1 is TERMINAL_SEED only
# and has zero base MONTHLY_COVERAGE_MATRIX cells. The monthly grid covers
# F2-F7 only.
MONTHLY_COVERAGE_FAMILIES: tuple[str, ...] = (
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE,
    SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS,
    SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
    SOURCE_FAMILY_JPX_CALENDAR,
)

INVENTORY_FIRST_YEAR_MONTH = (2017, 1)
INVENTORY_LAST_YEAR_MONTH = (2025, 12)

INVENTORY_AVAILABLE = "AVAILABLE"
INVENTORY_NOT_APPLICABLE = "NOT_APPLICABLE_BY_SOURCE_CONTRACT"
INVENTORY_MISSING = "MISSING"
_VALID_INVENTORY_STATUSES = frozenset({INVENTORY_AVAILABLE, INVENTORY_NOT_APPLICABLE, INVENTORY_MISSING})

# --- Known reviewed JPX endpoints (never guessed) ---------------------------
# V9_006_LOCATOR_IMPL_HIGH_2: F1's authoritative root is exactly the English
# listed-issues page bound in V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md
# (root=https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html).
# No alias, fallback root, redirect-based substitution, non-English
# alternative, or guessed historical root is permitted.
LISTED_ISSUES_PAGE_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"
LISTED_ISSUES_PAGE_HOST = "www.jpx.co.jp"
_DATA_LINK_RE = re.compile(r"href=[\"']([^\"']*data_j\.xls)[\"']", re.I)

CALENDAR_PAGE_URL = "https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html"
CALENDAR_PAGE_HOST = "www.jpx.co.jp"
CALENDAR_PAGE_COVERED_YEARS = (2026, 2027)
CALENDAR_PAGE_COVERAGE_START = "2026-01-01"
CALENDAR_PAGE_COVERAGE_END = "2027-12-31"

# F2/F4 share one root + semantic-row traversal (V9_006_STAGE_A_SOURCE_SLOT_
# LOCATOR_METHODOLOGY.md F2/F4).
MONTHLY_STATISTICS_ROOT_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html"
F2_SEMANTIC_ROW_LABEL = "Changes in Listed Companies and Issues, Etc."
F4_SEMANTIC_ROW_LABEL = "Ex-New, Ex-Rights, Etc."
MONTHLY_STATISTICS_DISCOVERY_ROOT = "MONTHLY_STATISTICS_DISCOVERY_ROOT"

# F3: delisted-company archive (YEAR objects).
DELISTED_COMPANY_ROOT_URL = "https://www.jpx.co.jp/english/listing/stocks/delisted/index.html"
DELISTED_COMPANY_DISCOVERY_ROOT = "DELISTED_COMPANY_DISCOVERY_ROOT"

# F5: monthly aggregate listed-issue counts (auxiliary=true).
LISTING_CO_ROOT_URL = "https://www.jpx.co.jp/english/listing/co/index.html"

# F6: TOPIX Historical Index Value (one GLOBAL object).
TOPIX_ROOT_URL = "https://www.jpx.co.jp/english/markets/indices/topix/"
F6_SEMANTIC_SECTION_LABEL = "Historical Index Value"

# F7: exact GPT-bound per-month calendar locator template and acquisition
# envelope (V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md F7).
CALENDAR_MONTHLY_LOCATOR_TEMPLATE = "https://www.jpx.co.jp/calendar/{year:04d}{month:02d}.html"
CALENDAR_ENVELOPE_FIRST_YEAR_MONTH = (2016, 9)
CALENDAR_ENVELOPE_LAST_YEAR_MONTH = (2026, 3)

TERMINAL_PERIOD = "TERMINAL"
TERMINAL_DISCOVERY_ROOT = "TERMINAL_DISCOVERY_ROOT"
CALENDAR_PERIOD = "CURRENT"

# --- Source-slot kinds (V9_006_SOURCE_SLOT_LOCATOR_HIGH_1) ------------------
SLOT_KIND_MONTHLY = "MONTHLY"
SLOT_KIND_YEAR = "YEAR"
SLOT_KIND_TERMINAL = "TERMINAL"
SLOT_KIND_GLOBAL = "GLOBAL"
VALID_SLOT_KINDS = frozenset({SLOT_KIND_MONTHLY, SLOT_KIND_YEAR, SLOT_KIND_TERMINAL, SLOT_KIND_GLOBAL})

# --- Failure classes ---------------------------------------------------------
PLUMBING_FAILURE_RETRIABLE = "PLUMBING_FAILURE_RETRIABLE"
SOURCE_OR_DATA_FEASIBILITY_FAILURE = "SOURCE_OR_DATA_FEASIBILITY_FAILURE"
GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
PROBE_SIGNAL_GRID_CONTRACT_MISMATCH = "PROBE_SIGNAL_GRID_CONTRACT_MISMATCH"

# --- V9_006_HIGH_1: pre-network locator-readiness stop ----------------------
# This is a governance/methodology stop, not a source/data feasibility
# result: it means the deterministic locator contract for the seven Stage-A
# source families and every required monthly slot is not yet mechanically
# complete from already-reviewed repository evidence, so real execution
# must not cross the JPX network boundary at all. It must never be reported
# as SOURCE_OR_DATA_FEASIBILITY_FAILURE, which is reserved for a genuine
# probe result after a complete locator contract was actually executed.
CHATGPT_DECISION_REQUIRED = "CHATGPT_DECISION_REQUIRED"
STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE = "STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE"

# --- V9_006_LOCATOR_IMPL_HIGH_1: pre-network acquisition-readiness stop ----
# A distinct, separate gate from the locator-*methodology* completeness
# check above. The reviewed LOCATOR_STRATEGIES registry can be fully
# complete (every family has a reviewed root/traversal or template) while
# the actual acquisition *implementation* -- the real code that walks a
# locked official root response to find each required child object for
# F2-F7, including every mandatory base/bridge/envelope/source-object slot
# -- does not yet exist. A knowingly incomplete acquisition pipeline must
# never be allowed to cross the network boundary and produce a guaranteed
# "fetch a couple of objects, then report the rest MISSING" result; that is
# not materially different from the doomed-run problem V9_006_HIGH_1
# already forbade. This also must never be reported as SOURCE_OR_DATA_
# FEASIBILITY_FAILURE, which remains reserved for a genuine result after a
# complete acquisition pipeline actually ran.
STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE = "STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE"

PUBLIC_FAILURE_CLASSES = frozenset({
    PLUMBING_FAILURE_RETRIABLE,
    SOURCE_OR_DATA_FEASIBILITY_FAILURE,
    GOVERNANCE_FAILURE,
    IMPLEMENTATION_FAILURE,
    PROBE_SIGNAL_GRID_CONTRACT_MISMATCH,
    CHATGPT_DECISION_REQUIRED,
})

_INTERNAL_REASON_TO_PUBLIC_FAILURE_CLASS: dict[str, str] = {
    PLUMBING_FAILURE_RETRIABLE: PLUMBING_FAILURE_RETRIABLE,
    SOURCE_OR_DATA_FEASIBILITY_FAILURE: SOURCE_OR_DATA_FEASIBILITY_FAILURE,
    GOVERNANCE_FAILURE: GOVERNANCE_FAILURE,
    IMPLEMENTATION_FAILURE: IMPLEMENTATION_FAILURE,
    PROBE_SIGNAL_GRID_CONTRACT_MISMATCH: PROBE_SIGNAL_GRID_CONTRACT_MISMATCH,
    "OFF_DOMAIN_REQUEST_REJECTED": SOURCE_OR_DATA_FEASIBILITY_FAILURE,
    "OFF_DOMAIN_REDIRECT_REJECTED": SOURCE_OR_DATA_FEASIBILITY_FAILURE,
    STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE: CHATGPT_DECISION_REQUIRED,
    STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE: CHATGPT_DECISION_REQUIRED,
}

# Flips to True only when a future, separately reviewed task implements the
# complete acquisition pipeline for every F1-F7 source-object slot
# (base/bridge/envelope), not merely the locator-strategy registry.
ACQUISITION_IMPLEMENTATION_COMPLETE = False

MAX_ATTEMPTS = 3
MAX_RETRIES = 2
BACKOFF_SECONDS: tuple[int, ...] = (5, 30)


def _public_failure_class(reason: str) -> str:
    return _INTERNAL_REASON_TO_PUBLIC_FAILURE_CLASS.get(reason, IMPLEMENTATION_FAILURE)


class V9005StageABlocked(RuntimeError):
    """Internal reason stays in .reason/str(exc); only .failure_class is public-safe."""

    def __init__(self, reason: str, *, network_request_count: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.failure_class = _public_failure_class(reason)
        self.network_request_count = (
            network_request_count if isinstance(network_request_count, int) and network_request_count >= 0 else 0
        )


# --- Canonical bytes / hashing ----------------------------------------------

def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Off-domain rejection (contract item 1) ---------------------------------

def _is_allowed_jpx_host(hostname: object) -> bool:
    if not isinstance(hostname, str) or not hostname:
        return False
    lowered = hostname.lower()
    return lowered == ALLOWED_HOST_SUFFIX or lowered.endswith("." + ALLOWED_HOST_SUFFIX)


def validate_jpx_url(url: object, *, reason: str = "OFF_DOMAIN_REQUEST_REJECTED") -> str:
    """Fail closed unless url is https, on jpx.co.jp or a subdomain, with no
    credentials, nonstandard port, or fragment."""
    if not isinstance(url, str):
        raise V9005StageABlocked(reason)
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not _is_allowed_jpx_host(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise V9005StageABlocked(reason)
    return url


# --- Terminal-snapshot locator (reused verbatim from reviewed code) --------

def extract_data_j_xls_url(page_bytes: bytes) -> str:
    try:
        text = page_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE) from exc
    match = _DATA_LINK_RE.search(text)
    if not match:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    resolved = urllib.parse.urljoin(LISTED_ISSUES_PAGE_URL, match.group(1))
    return validate_jpx_url(resolved, reason="OFF_DOMAIN_REDIRECT_REJECTED")


@dataclass
class _MonthlyStatisticsCell:
    tag: str
    text_parts: list[str]
    hrefs: list[str]

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


@dataclass
class _MonthlyStatisticsTable:
    rows: list[list[_MonthlyStatisticsCell]]


class _MonthlyStatisticsHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str]] = []
        self.tables: list[_MonthlyStatisticsTable] = []
        self._table_stack: list[_MonthlyStatisticsTable] = []
        self._current_row: list[_MonthlyStatisticsCell] | None = None
        self._current_cell: _MonthlyStatisticsCell | None = None
        self._anchor_stack: list[list[str]] = []
        self._relevant_tag_stack: list[str] = []
        self._invalid_structure = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            if self._relevant_tag_stack:
                self._invalid_structure = True
                return
            table = _MonthlyStatisticsTable(rows=[])
            self.tables.append(table)
            self._table_stack.append(table)
            self._relevant_tag_stack.append(tag)
        elif tag == "tr":
            if not self._relevant_tag_stack or self._relevant_tag_stack[-1] != "table":
                self._invalid_structure = True
                return
            self._current_row = []
            self._table_stack[-1].rows.append(self._current_row)
            self._relevant_tag_stack.append(tag)
        elif tag in {"th", "td"}:
            if not self._relevant_tag_stack or self._relevant_tag_stack[-1] != "tr" or self._current_row is None:
                self._invalid_structure = True
                return
            self._current_cell = _MonthlyStatisticsCell(tag=tag, text_parts=[], hrefs=[])
            self._current_row.append(self._current_cell)
            self._relevant_tag_stack.append(tag)
        elif tag == "a":
            if "a" in self._relevant_tag_stack:
                self._invalid_structure = True
                return
            href = attributes.get("href")
            self._anchor_stack.append([href or "", ""])
            self._relevant_tag_stack.append(tag)
            if self._current_cell is not None and href is not None:
                self._current_cell.hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag not in {"table", "tr", "th", "td", "a"}:
            return
        if not self._relevant_tag_stack or self._relevant_tag_stack[-1] != tag:
            self._invalid_structure = True
            return
        self._relevant_tag_stack.pop()
        if tag == "a":
            href, text = self._anchor_stack.pop()
            self.anchors.append((href, " ".join(text.split())))
        elif tag in {"th", "td"}:
            self._current_cell = None
        elif tag == "tr":
            self._current_row = None
        elif tag == "table" and self._table_stack:
            self._table_stack.pop()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"table", "tr", "th", "td", "a"}:
            self._invalid_structure = True

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.text_parts.append(data)
        for anchor in self._anchor_stack:
            anchor[1] += data


def _parse_monthly_statistics_html(page_bytes: bytes) -> _MonthlyStatisticsHtmlParser:
    if not isinstance(page_bytes, bytes):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    try:
        parser = _MonthlyStatisticsHtmlParser()
        parser.feed(page_bytes.decode("utf-8"))
        parser.close()
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
    if (
        parser._invalid_structure
        or parser._relevant_tag_stack
        or parser._anchor_stack
        or parser._table_stack
        or parser._current_row is not None
        or parser._current_cell is not None
    ):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return parser


def _resolve_locked_page_link(page_url: str, href: object) -> str:
    try:
        validate_jpx_url(page_url)
    except V9005StageABlocked as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
    if not isinstance(href, str) or not href:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    try:
        return validate_jpx_url(
            urllib.parse.urljoin(page_url, href), reason="OFF_DOMAIN_REDIRECT_REJECTED",
        )
    except V9005StageABlocked as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc


def resolve_monthly_statistics_year_page_url(root_bytes: bytes, root_url: str, requested_year: int) -> str:
    """Resolve exactly one official archive-year page from locked root bytes."""
    if isinstance(requested_year, bool) or not isinstance(requested_year, int) or not 1000 <= requested_year <= 9999:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    parser = _parse_monthly_statistics_html(root_bytes)
    candidates = [
        _resolve_locked_page_link(root_url, href)
        for href, text in parser.anchors
        if text == str(requested_year)
    ]
    if len(candidates) != 1:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return candidates[0]


def resolve_delisted_company_year_url(root_bytes: bytes, root_page_url: str, requested_year: int) -> str:
    """Resolve one F3 archive YEAR object from locked root HTML only."""
    if isinstance(requested_year, bool) or not isinstance(requested_year, int) or not 1000 <= requested_year <= 9999:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    parser = _parse_monthly_statistics_html(root_bytes)
    candidates = [
        _resolve_locked_page_link(root_page_url, href)
        for href, text in parser.anchors
        if text == str(requested_year)
    ]
    if len(candidates) != 1:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return candidates[0]


def resolve_monthly_statistics_evidence_url(
    year_page_bytes: bytes,
    year_page_url: str,
    source_family: str,
    requested_month: str,
    *,
    selected_year: int,
) -> str:
    """Resolve one F2/F4 monthly object from a locked selected-year page."""
    labels = {
        SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT: F2_SEMANTIC_ROW_LABEL,
        SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE: F4_SEMANTIC_ROW_LABEL,
    }
    label = labels.get(source_family)
    if label is None:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    year, _month = _parse_year_month(requested_month)
    if isinstance(selected_year, bool) or not isinstance(selected_year, int) or year != selected_year:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    parser = _parse_monthly_statistics_html(year_page_bytes)
    matching_rows: list[tuple[_MonthlyStatisticsTable, list[_MonthlyStatisticsCell]]] = []
    for table in parser.tables:
        for row in table.rows:
            if any(cell.text == label for cell in row):
                matching_rows.append((table, row))
    if len(matching_rows) != 1:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    table, semantic_row = matching_rows[0]
    month_columns = [
        index
        for row in table.rows
        for index, cell in enumerate(row)
        if cell.tag == "th" and cell.text == requested_month
    ]
    if len(month_columns) != 1:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    column = month_columns[0]
    if column >= len(semantic_row) or len(semantic_row[column].hrefs) != 1:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return _resolve_locked_page_link(year_page_url, semantic_row[column].hrefs[0])


def resolve_f7_calendar_url(year: int, month: int) -> str:
    """Exact GPT-bound per-month F7 locator: no traversal, no discovery --
    the URL is computed directly from the template."""
    if isinstance(year, bool) or isinstance(month, bool) or not isinstance(year, int) or not isinstance(month, int):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if not 1 <= month <= 12:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    url = CALENDAR_MONTHLY_LOCATOR_TEMPLATE.format(year=year, month=month)
    return validate_jpx_url(url)


# --- Reviewed deterministic locator strategy registry (V9_006_SOURCE_SLOT_
# LOCATOR_HIGH_1 / F1_TERMINAL_SEED_PREFREEZE_AMENDMENT) --------------------
#
# Each entry records the exact reviewed root/semantic-traversal rule (or,
# for F7, the exact reviewed per-month template) for one of the seven
# source families -- never a resolved child URL. A family's presence here
# means its locator *strategy* is reviewed and deterministic; the concrete
# child object for a given month/year is only discoverable by actually
# traversing the locked official root response at real Stage-A execution
# time, which this registry never requires in advance.

@dataclass(frozen=True)
class LocatorStrategy:
    source_family: str
    slot_kind: str
    root_url: str | None
    traversal: str
    auxiliary: bool = False
    locator_template: str | None = None

    def __post_init__(self) -> None:
        if self.slot_kind not in VALID_SLOT_KINDS:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        if self.root_url is not None:
            validate_jpx_url(self.root_url)
        if self.root_url is None and self.locator_template is None:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)


LOCATOR_STRATEGIES: dict[str, LocatorStrategy] = {
    SOURCE_FAMILY_LISTED_ISSUES_MONTH_END: LocatorStrategy(
        source_family=SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
        slot_kind=SLOT_KIND_TERMINAL,
        root_url=LISTED_ISSUES_PAGE_URL,
        traversal="unique same-domain data_j.xls link from the official listed-issues page",
    ),
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT: LocatorStrategy(
        source_family=SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        slot_kind=SLOT_KIND_MONTHLY,
        root_url=MONTHLY_STATISTICS_ROOT_URL,
        traversal=(
            "official archive-year selector -> semantic row "
            f"'{F2_SEMANTIC_ROW_LABEL}' -> requested month column -> unique same-domain object"
        ),
    ),
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE: LocatorStrategy(
        source_family=SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE,
        slot_kind=SLOT_KIND_YEAR,
        root_url=DELISTED_COMPANY_ROOT_URL,
        traversal="official archive-year selector -> one YEAR object may cover its 12 months",
    ),
    SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE: LocatorStrategy(
        source_family=SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
        slot_kind=SLOT_KIND_MONTHLY,
        root_url=MONTHLY_STATISTICS_ROOT_URL,
        traversal=(
            "official archive-year selector -> semantic row "
            f"'{F4_SEMANTIC_ROW_LABEL}' -> requested month column -> unique same-domain object"
        ),
    ),
    SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS: LocatorStrategy(
        source_family=SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS,
        slot_kind=SLOT_KIND_MONTHLY,
        root_url=LISTING_CO_ROOT_URL,
        traversal="official archive selector -> requested month -> unique same-domain object",
        auxiliary=True,
    ),
    SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE: LocatorStrategy(
        source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
        slot_kind=SLOT_KIND_GLOBAL,
        root_url=TOPIX_ROOT_URL,
        traversal=f"unique same-domain object under semantic section '{F6_SEMANTIC_SECTION_LABEL}'",
    ),
    SOURCE_FAMILY_JPX_CALENDAR: LocatorStrategy(
        source_family=SOURCE_FAMILY_JPX_CALENDAR,
        slot_kind=SLOT_KIND_MONTHLY,
        root_url=None,
        traversal="exact bound per-month template, no discovery",
        locator_template=CALENDAR_MONTHLY_LOCATOR_TEMPLATE,
    ),
}


# --- Transport with retry (item 3: classify only per AI_REAL_EXECUTION_RUNBOOK.md) --

@dataclass(frozen=True)
class FetchResult:
    """One actually observed transport response. Its status, final URL,
    and payload stay coupled so raw-lock provenance cannot be fabricated
    separately from the bytes that were consumed."""

    payload: bytes
    resolved_url: str
    http_status: int


@dataclass(frozen=True)
class F2F4RequiredSlotAcquisition:
    """Separate F2/F4 base and F2-only bridge reference domains."""

    base_coverage_references: Mapping[tuple[str, str], tuple[str, ...]]
    f2_bridge_references: Mapping[str, tuple[str, ...]]
    network_attempt_count: int


@dataclass(frozen=True)
class F3RequiredSlotAcquisition:
    """F3's complete-year fan-out references and aggregate attempts."""

    base_coverage_references: Mapping[tuple[str, str], tuple[str, ...]]
    network_attempt_count: int

def fetch_once_with_retry(
    url: str,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
) -> tuple[FetchResult, int]:
    """Fetch url, rejecting off-domain requests/redirects before content is
    consumed. Retries only classified retryable transport failures, up to
    the frozen attempt/backoff policy, per AI_REAL_EXECUTION_RUNBOOK.md."""
    validate_jpx_url(url, reason="OFF_DOMAIN_REQUEST_REJECTED")
    requests_used = 0
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            requests_used += 1
            result = fetcher(url)
            if (
                not isinstance(result, FetchResult)
                or not isinstance(result.payload, bytes)
                or not isinstance(result.resolved_url, str)
                or isinstance(result.http_status, bool)
                or not isinstance(result.http_status, int)
            ):
                raise V9005StageABlocked(IMPLEMENTATION_FAILURE, network_request_count=requests_used)
            validate_jpx_url(result.resolved_url, reason="OFF_DOMAIN_REDIRECT_REJECTED")
            if not result.payload:
                raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE, network_request_count=requests_used)
            return result, requests_used
        except V9005StageABlocked as exc:
            exc.network_request_count = requests_used
            raise
        except Exception as exc:
            try:
                _label, retryable = classify_transport_exception(exc)
            except Exception as classifier_exc:
                raise V9005StageABlocked(IMPLEMENTATION_FAILURE, network_request_count=requests_used) from classifier_exc
            if not retryable:
                raise V9005StageABlocked(IMPLEMENTATION_FAILURE, network_request_count=requests_used) from exc
            last = exc
            if attempt < MAX_RETRIES:
                try:
                    sleep(BACKOFF_SECONDS[attempt])
                except Exception as sleep_exc:
                    raise V9005StageABlocked(IMPLEMENTATION_FAILURE, network_request_count=requests_used) from sleep_exc
    raise V9005StageABlocked(PLUMBING_FAILURE_RETRIABLE, network_request_count=requests_used) from last


# --- Raw first-complete-payload locking (contract item 3) ------------------

def _record_key(source_family: str, applicable_period: str, requested_url: str) -> str:
    material = (
        "V9_005_STAGE_A_RAW_LOCK_KEY_V1\0" + source_family + "\0" + applicable_period + "\0" + requested_url
    ).encode("utf-8")
    return sha256_bytes(material)


def source_object_slot_id(source_family: str, applicable_period: str, requested_url: str) -> str:
    """Return the existing raw-lock key for a coverage-evidence object."""
    return _record_key(source_family, applicable_period, requested_url)


def _raw_paths(output_root: Path, key: str) -> tuple[Path, Path]:
    raw_dir = Path(output_root) / "raw"
    return raw_dir / (key + ".bin"), raw_dir / (key + ".json")


_REQUIRED_LOCK_META_FIELDS = frozenset({
    "schema_version", "source_family", "applicable_period", "requested_url",
    "resolved_url", "http_status", "retrieval_timestamp_utc", "byte_length", "sha256",
})
_RAW_LOCK_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _is_canonical_raw_lock_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.strptime(value, _RAW_LOCK_TIMESTAMP_FORMAT)
    except ValueError:
        return False
    return parsed.strftime(_RAW_LOCK_TIMESTAMP_FORMAT) == value


def _is_valid_fetch_result(value: object) -> bool:
    return (
        isinstance(value, FetchResult)
        and isinstance(value.payload, bytes)
        and bool(value.payload)
        and isinstance(value.resolved_url, str)
        and bool(value.resolved_url)
        and not isinstance(value.http_status, bool)
        and isinstance(value.http_status, int)
        and 100 <= value.http_status <= 599
    )


def _lock_meta_matches_raw(meta: object, raw: bytes, *, expected_key: str) -> bool:
    if not isinstance(meta, dict) or set(meta) != _REQUIRED_LOCK_META_FIELDS:
        return False
    if (
        meta["schema_version"] != "V9_005_STAGE_A_RAW_LOCK_V1"
        or meta["source_family"] not in SOURCE_FAMILIES
        or not isinstance(meta["applicable_period"], str) or not meta["applicable_period"]
        or not isinstance(meta["requested_url"], str) or not meta["requested_url"]
        or not isinstance(meta["resolved_url"], str) or not meta["resolved_url"]
        or isinstance(meta["http_status"], bool) or not isinstance(meta["http_status"], int)
        or not 100 <= meta["http_status"] <= 599
        or not _is_canonical_raw_lock_timestamp(meta["retrieval_timestamp_utc"])
        or isinstance(meta["byte_length"], bool) or not isinstance(meta["byte_length"], int)
    ):
        return False
    try:
        validate_jpx_url(meta["requested_url"])
        validate_jpx_url(meta["resolved_url"])
    except V9005StageABlocked:
        return False
    return (
        _record_key(meta["source_family"], meta["applicable_period"], meta["requested_url"]) == expected_key
        and meta["sha256"] == sha256_bytes(raw)
        and meta["byte_length"] == len(raw)
    )


def read_locked_payload(
    output_root: str | os.PathLike[str],
    source_family: str,
    applicable_period: str,
    requested_url: str,
) -> dict[str, Any] | None:
    key = _record_key(source_family, applicable_period, requested_url)
    raw_path, meta_path = _raw_paths(Path(output_root), key)
    if not raw_path.exists() and not meta_path.exists():
        return None
    if not raw_path.exists() or not meta_path.exists():
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    try:
        raw = raw_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
    if not _lock_meta_matches_raw(meta, raw, expected_key=key):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return {"raw": raw, **meta}


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.parent / (path.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(stage, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, path)
    except FileExistsError as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
    finally:
        if stage.exists():
            stage.unlink(missing_ok=True)


def lock_first_complete_payload(
    output_root: str | os.PathLike[str],
    *,
    source_family: str,
    applicable_period: str,
    requested_url: str,
    fetch_result: FetchResult,
    retrieval_timestamp_utc: str,
) -> dict[str, Any]:
    if source_family not in SOURCE_FAMILIES:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if (
        not isinstance(requested_url, str) or not requested_url
        or not _is_valid_fetch_result(fetch_result)
        or not _is_canonical_raw_lock_timestamp(retrieval_timestamp_utc)
    ):
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    try:
        validate_jpx_url(requested_url)
        validate_jpx_url(fetch_result.resolved_url)
    except V9005StageABlocked as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
    key = _record_key(source_family, applicable_period, requested_url)
    raw_path, meta_path = _raw_paths(Path(output_root), key)
    meta = {
        "schema_version": "V9_005_STAGE_A_RAW_LOCK_V1",
        "source_family": source_family,
        "applicable_period": applicable_period,
        "requested_url": requested_url,
        "resolved_url": fetch_result.resolved_url,
        "http_status": fetch_result.http_status,
        "retrieval_timestamp_utc": retrieval_timestamp_utc,
        "byte_length": len(fetch_result.payload),
        "sha256": sha256_bytes(fetch_result.payload),
    }
    _atomic_create(raw_path, fetch_result.payload)
    _atomic_create(meta_path, canonical_bytes(meta))
    return {"raw": fetch_result.payload, **meta}


def ensure_locked_payload(
    output_root: str | os.PathLike[str],
    *,
    source_family: str,
    applicable_period: str,
    requested_url: str,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> tuple[dict[str, Any], int]:
    """Never fetch twice for the same key: reprocess already-locked bytes."""
    existing = read_locked_payload(output_root, source_family, applicable_period, requested_url)
    if existing is not None:
        return existing, 0
    result, requests_used = fetch_once_with_retry(requested_url, fetcher, sleep)
    now = clock()
    locked = lock_first_complete_payload(
        output_root,
        source_family=source_family,
        applicable_period=applicable_period,
        requested_url=requested_url,
        fetch_result=result,
        retrieval_timestamp_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return locked, requests_used


def verify_raw_provenance(output_root: str | os.PathLike[str]) -> bool:
    """Independently re-verify every locked raw/meta pair on disk has the
    complete required provenance field set and a matching hash."""
    raw_dir = Path(output_root) / "raw"
    if not raw_dir.exists():
        return True
    raw_paths = {path.with_suffix("") for path in raw_dir.glob("*.bin")}
    meta_paths = {path.with_suffix("") for path in raw_dir.glob("*.json")}
    if raw_paths != meta_paths:
        return False
    for stem in sorted(raw_paths):
        meta_path = stem.with_suffix(".json")
        raw_path = meta_path.with_suffix(".bin")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = raw_path.read_bytes()
        except Exception:
            return False
        if not _lock_meta_matches_raw(meta, raw, expected_key=stem.name):
            return False
    return True


# --- Durable output root (contract item 6 / AI_REAL_EXECUTION_RUNBOOK.md SS8) --

def initialize_output_root(output_root: str | os.PathLike[str]) -> Path:
    root = Path(output_root)
    if root.exists():
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    root.mkdir(parents=True)
    (root / "raw").mkdir()
    return root


# --- Signal-grid prefreeze binding point-of-use check (contract item 5) ----

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def verify_signal_grid_binding(
    repo_root: str | os.PathLike[str],
    *,
    git: Callable[[list[str]], str] | None = None,
) -> str:
    if git is None:
        def git(args: list[str]) -> str:
            try:
                return subprocess.run(
                    ["git", *args], cwd=str(repo_root), check=True, text=True, capture_output=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise V9005StageABlocked(PROBE_SIGNAL_GRID_CONTRACT_MISMATCH) from exc
    try:
        head = git(["rev-parse", "HEAD"])
        blob = git(["rev-parse", f"HEAD:{BOUND_SIGNAL_GRID_PATH}"])
    except V9005StageABlocked:
        raise
    except Exception as exc:
        raise V9005StageABlocked(PROBE_SIGNAL_GRID_CONTRACT_MISMATCH) from exc
    if not _HEX40.fullmatch(head or "") or blob != BOUND_SIGNAL_GRID_BLOB_SHA:
        raise V9005StageABlocked(PROBE_SIGNAL_GRID_CONTRACT_MISMATCH)
    return head


# --- Mechanical signal-grid endpoint derivation (contract item 5) ----------

def _parse_iso_date(value: object) -> date:
    if not isinstance(value, str):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
    if parsed.isoformat() != value:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return parsed


def build_trading_day_set(
    market_holiday_dates: Sequence[str],
    coverage_start: str,
    coverage_end: str,
) -> tuple[str, ...]:
    """Deterministic sorted JPX trading days in [coverage_start, coverage_end]
    given a locked official set of non-trading (holiday) dates. Saturdays and
    Sundays are always non-trading."""
    holidays = {_parse_iso_date(value) for value in market_holiday_dates}
    start = _parse_iso_date(coverage_start)
    end = _parse_iso_date(coverage_end)
    if not start <= end:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    days: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(days)


def derive_final_signal_d0(trading_days: Sequence[str], *, coverage_start: str) -> str:
    """j0 = index (within the complete JPX trading-day sequence) of the
    first JPX trading day >= 2018-01-01; D0 at index j iff (j-j0) mod 3 == 0;
    FINAL_SIGNAL_D0 = last such D0 <= 2025-12-31. Requires the supplied
    trading_days to already start on/before 2018-01-01 so index positions
    are the true global calendar indices, not an offset window."""
    if list(trading_days) != sorted(trading_days):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if _parse_iso_date(coverage_start) > _parse_iso_date("2018-01-01"):
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    lower = _parse_iso_date("2018-01-01")
    upper = _parse_iso_date("2025-12-31")
    j0 = None
    for index, value in enumerate(trading_days):
        if _parse_iso_date(value) >= lower:
            j0 = index
            break
    if j0 is None:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    final_d0 = None
    for index in range(j0, len(trading_days)):
        value = trading_days[index]
        if _parse_iso_date(value) > upper:
            break
        if (index - j0) % 3 == 0:
            final_d0 = value
    if final_d0 is None:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    return final_d0


def nth_trading_day_after(trading_days: Sequence[str], value: str, n: int) -> str:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    try:
        start_index = list(trading_days).index(value)
    except ValueError as exc:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE) from exc
    target_index = start_index + n
    if target_index >= len(trading_days):
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    return trading_days[target_index]


def derive_stage_b_global_end_exclusive(
    trading_days: Sequence[str],
    *,
    coverage_start: str,
) -> dict[str, str]:
    """FINAL_SIGNAL_D0 -> FINAL_PLANNED_D3 (3rd trading day after) ->
    FINAL_POSSIBLE_EXIT_DAY (20th exit-attempt date, D3=attempt day 1) ->
    STAGE_B_GLOBAL_END_EXCLUSIVE (calendar date immediately after that)."""
    final_signal_d0 = derive_final_signal_d0(trading_days, coverage_start=coverage_start)
    final_planned_d3 = nth_trading_day_after(trading_days, final_signal_d0, 3)
    final_possible_exit_day = nth_trading_day_after(trading_days, final_planned_d3, 19)
    stage_b_global_end_exclusive = (_parse_iso_date(final_possible_exit_day) + timedelta(days=1)).isoformat()
    return {
        "final_signal_d0": final_signal_d0,
        "final_planned_d3": final_planned_d3,
        "final_possible_exit_day": final_possible_exit_day,
        "stage_b_global_end_exclusive": stage_b_global_end_exclusive,
    }


# --- Deterministic monthly SOURCE_INVENTORY ---------------------------------

def _year_month_range(first: tuple[int, int], last: tuple[int, int]) -> tuple[str, ...]:
    months: list[str] = []
    year, month = first
    while (year, month) <= last:
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return tuple(months)


def inventory_months() -> tuple[str, ...]:
    """The base MONTHLY_COVERAGE_MATRIX months: 2017-01 through 2025-12
    (108 months). F1 has zero cells over this range (TERMINAL_SEED only);
    the matrix covers MONTHLY_COVERAGE_FAMILIES only (F2-F7)."""
    return _year_month_range(INVENTORY_FIRST_YEAR_MONTH, INVENTORY_LAST_YEAR_MONTH)


def _parse_year_month(value: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if not match:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return year, month


def monthly_statistics_discovery_year_period(year: int) -> str:
    if isinstance(year, bool) or not isinstance(year, int) or not 1000 <= year <= 9999:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return f"MONTHLY_STATISTICS_DISCOVERY_YEAR_{year:04d}"


def acquire_f2_f4_monthly_evidence(
    output_root: str | os.PathLike[str],
    *,
    source_family: str,
    requested_month: str,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> tuple[str, int]:
    """Lock one F2/F4 child using the shared F2-owned support objects only."""
    if source_family not in {
        SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    }:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    year, _month = _parse_year_month(requested_month)
    support_owner = SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT
    root, root_attempts = ensure_locked_payload(
        output_root,
        source_family=support_owner,
        applicable_period=MONTHLY_STATISTICS_DISCOVERY_ROOT,
        requested_url=MONTHLY_STATISTICS_ROOT_URL,
        fetcher=fetcher,
        sleep=sleep,
        clock=clock,
    )
    year_page_url = resolve_monthly_statistics_year_page_url(
        root["raw"], root["resolved_url"], year,
    )
    year_page, year_attempts = ensure_locked_payload(
        output_root,
        source_family=support_owner,
        applicable_period=monthly_statistics_discovery_year_period(year),
        requested_url=year_page_url,
        fetcher=fetcher,
        sleep=sleep,
        clock=clock,
    )
    child_url = resolve_monthly_statistics_evidence_url(
        year_page["raw"], year_page["resolved_url"], source_family, requested_month, selected_year=year,
    )
    _child, child_attempts = ensure_locked_payload(
        output_root,
        source_family=source_family,
        applicable_period=requested_month,
        requested_url=child_url,
        fetcher=fetcher,
        sleep=sleep,
        clock=clock,
    )
    return source_object_slot_id(source_family, requested_month, child_url), root_attempts + year_attempts + child_attempts


def f2_bridge_months(terminal_month: str) -> tuple[str, ...]:
    """F2's mandatory post-2025 bridge slots, mechanically derived from the
    terminal snapshot month T: every month from 2026-01 through T
    inclusive, needed to reverse-reconstruct from T back through 2025-12.
    These are additional mandatory SOURCE_OBJECT_INVENTORY slots outside
    the 648-record base matrix, using the exact same reviewed F2 strategy.
    Returns an empty tuple if T is on/before 2025-12 (no bridge needed)."""
    terminal_year_month = _parse_year_month(terminal_month)
    if terminal_year_month <= INVENTORY_LAST_YEAR_MONTH:
        return ()
    return _year_month_range((2026, 1), terminal_year_month)


def _validate_f2_f4_required_slot_references(
    output_root: str | os.PathLike[str],
    base_references: Mapping[tuple[str, str], tuple[str, ...]],
    bridge_references: Mapping[str, tuple[str, ...]],
    bridge_months: Sequence[str],
) -> None:
    families = (
        SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    )
    base_months = inventory_months()
    expected_base_keys = {(family, month) for month in base_months for family in families}
    if set(base_references) != expected_base_keys:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if set(bridge_references) != set(bridge_months) or set(bridge_references) & set(base_months):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    verified_locks = _verified_raw_lock_index(output_root)

    def validate_one(slot_ids: tuple[str, ...], family: str, month: str) -> None:
        if not isinstance(slot_ids, tuple) or len(slot_ids) != 1:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        slot_id = slot_ids[0]
        if not isinstance(slot_id, str) or re.fullmatch(r"[0-9a-f]{64}", slot_id) is None:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        metadata = verified_locks.get(slot_id)
        if (
            metadata is None
            or metadata.get("source_family") != family
            or metadata.get("applicable_period") != month
        ):
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    for (family, month), slot_ids in base_references.items():
        validate_one(slot_ids, family, month)
    for month, slot_ids in bridge_references.items():
        validate_one(slot_ids, SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, month)


def acquire_f2_f4_required_slots(
    output_root: str | os.PathLike[str],
    *,
    terminal_month: str,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> F2F4RequiredSlotAcquisition:
    """Acquire exactly the required F2/F4 base slots and F2 bridge slots."""
    bridge_months = f2_bridge_months(terminal_month)
    base_references: dict[tuple[str, str], tuple[str, ...]] = {}
    bridge_references: dict[str, tuple[str, ...]] = {}
    attempts = 0
    for month in inventory_months():
        for family in (
            SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
            SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
        ):
            slot_id, slot_attempts = acquire_f2_f4_monthly_evidence(
                output_root,
                source_family=family,
                requested_month=month,
                fetcher=fetcher,
                sleep=sleep,
                clock=clock,
            )
            base_references[(family, month)] = (slot_id,)
            attempts += slot_attempts
    for month in bridge_months:
        slot_id, slot_attempts = acquire_f2_f4_monthly_evidence(
            output_root,
            source_family=SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
            requested_month=month,
            fetcher=fetcher,
            sleep=sleep,
            clock=clock,
        )
        bridge_references[month] = (slot_id,)
        attempts += slot_attempts
    _validate_f2_f4_required_slot_references(
        output_root, base_references, bridge_references, bridge_months,
    )
    return F2F4RequiredSlotAcquisition(base_references, bridge_references, attempts)


def _validate_f3_required_slot_references(
    output_root: str | os.PathLike[str],
    base_references: Mapping[tuple[str, str], tuple[str, ...]],
) -> None:
    family = SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE
    expected_keys = {(family, month) for month in inventory_months()}
    if set(base_references) != expected_keys:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    verified_locks = _verified_raw_lock_index(output_root)
    year_slot_ids: set[str] = set()
    for year in range(2017, 2026):
        months = tuple(f"{year}-{month:02d}" for month in range(1, 13))
        slot_ids = [base_references[(family, month)] for month in months]
        if any(not isinstance(ids, tuple) or len(ids) != 1 for ids in slot_ids):
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        slot_id = slot_ids[0][0]
        if any(ids[0] != slot_id for ids in slot_ids):
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        if not isinstance(slot_id, str) or re.fullmatch(r"[0-9a-f]{64}", slot_id) is None:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        metadata = verified_locks.get(slot_id)
        if (
            metadata is None
            or metadata.get("source_family") != family
            or metadata.get("applicable_period") != str(year)
            or source_object_slot_id(family, str(year), metadata.get("requested_url")) != slot_id
        ):
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        year_slot_ids.add(slot_id)
    if len(year_slot_ids) != 9:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)


def acquire_f3_required_slots(
    output_root: str | os.PathLike[str],
    *,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> F3RequiredSlotAcquisition:
    """Acquire nine F3 YEAR objects and fan each to its twelve base months."""
    family = SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE
    root, attempts = ensure_locked_payload(
        output_root,
        source_family=family,
        applicable_period=DELISTED_COMPANY_DISCOVERY_ROOT,
        requested_url=DELISTED_COMPANY_ROOT_URL,
        fetcher=fetcher,
        sleep=sleep,
        clock=clock,
    )
    base_references: dict[tuple[str, str], tuple[str, ...]] = {}
    for year in range(2017, 2026):
        year_url = resolve_delisted_company_year_url(root["raw"], root["resolved_url"], year)
        _year_lock, year_attempts = ensure_locked_payload(
            output_root,
            source_family=family,
            applicable_period=str(year),
            requested_url=year_url,
            fetcher=fetcher,
            sleep=sleep,
            clock=clock,
        )
        attempts += year_attempts
        slot_id = source_object_slot_id(family, str(year), year_url)
        for month in range(1, 13):
            base_references[(family, f"{year}-{month:02d}")] = (slot_id,)
    _validate_f3_required_slot_references(output_root, base_references)
    return F3RequiredSlotAcquisition(base_references, attempts)


def calendar_envelope_months() -> tuple[str, ...]:
    """All required F7 calendar months: 2016-09 through 2026-03 inclusive."""
    return _year_month_range(CALENDAR_ENVELOPE_FIRST_YEAR_MONTH, CALENDAR_ENVELOPE_LAST_YEAR_MONTH)


def calendar_envelope_extra_months() -> tuple[str, ...]:
    """F7 envelope months outside the base 2017-01..2025-12 matrix: these
    are additional mandatory slots outside the 648-record base matrix,
    using the exact same reviewed F7 per-month template."""
    base = set(inventory_months())
    return tuple(month for month in calendar_envelope_months() if month not in base)


def resolve_month_locator(source_family: str, month: str) -> LocatorStrategy:
    """Return the reviewed deterministic locator strategy bound for this
    monthly-coverage source family (F2-F7). This never requires a concrete
    child URL that can only be discovered by traversing a locked official
    JPX root response at real execution time -- it only requires that a
    reviewed strategy (root + semantic traversal, or F7's exact per-month
    template) already exists in `LOCATOR_STRATEGIES`. F1 is excluded: it
    is TERMINAL_SEED only and has zero monthly cells (V9_006_F1_TERMINAL_
    SEED_PREFREEZE_AMENDMENT)."""
    if source_family not in MONTHLY_COVERAGE_FAMILIES:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if month not in inventory_months():
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    strategy = LOCATOR_STRATEGIES.get(source_family)
    if strategy is None:
        raise V9005StageABlocked(STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE)
    return strategy


def verify_locator_contract_complete() -> None:
    """Pre-network locator-*methodology*-readiness check. Real Stage-A
    execution must not cross the JPX network boundary while any required
    slot -- F1's mandatory TERMINAL slot, the 648-record base
    MONTHLY_COVERAGE_MATRIX (F2-F7 x 108 months), or F7's envelope slots
    outside 2017-2025 -- has no reviewed deterministic locator strategy
    bound in `LOCATOR_STRATEGIES`. This performs no I/O and invents no
    URL, cadence, N/A rule, archive period, retry rule, or source
    substitution. It does NOT require that a concrete per-month/per-year
    child URL is already known -- discovering that child URL by
    traversing a locked official JPX root response is real Stage-A
    network work, gated behind a fresh, separate, explicit human
    authorization this check does not create. F2's post-2025 bridge slots
    are not enumerated here because they depend on the terminal snapshot
    month T, which is only known after a real F1 fetch; they reuse the
    exact same reviewed F2 strategy verified below, so their completeness
    already follows from F2's binding (see `f2_bridge_months`)."""
    missing_family_strategies = [family for family in SOURCE_FAMILIES if family not in LOCATOR_STRATEGIES]
    if missing_family_strategies:
        raise V9005StageABlocked(STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE)
    for month in inventory_months():
        for family in MONTHLY_COVERAGE_FAMILIES:
            resolve_month_locator(family, month)
    # F7 envelope slots (2016-09..2016-12, 2026-01..2026-03) reuse the
    # exact same F2/F7 family-level strategy already verified above; no
    # separate per-slot lookup is needed for locator-methodology
    # completeness.


def verify_acquisition_implementation_ready() -> None:
    """V9_006_LOCATOR_IMPL_HIGH_1 pre-network acquisition-*implementation*-
    readiness gate. Distinct from, and in addition to,
    `verify_locator_contract_complete()`: the locator-strategy registry may
    be fully reviewed and complete while the actual code that acquires
    every required F1-F7 slot (base 648-record matrix, F1's mandatory
    TERMINAL object, F2's post-2025 bridge slots, and F7's envelope slots)
    does not yet exist. Real Stage-A execution must not cross the JPX
    network boundary while that acquisition pipeline is incomplete, even
    though the locator contract itself is fully bound -- a knowingly
    incomplete acquisition run is not an acceptable substitute for
    stopping, for exactly the same reason a knowingly incomplete locator
    contract was not. This raises unconditionally until
    `ACQUISITION_IMPLEMENTATION_COMPLETE` is flipped to `True` by a future,
    separately reviewed task that actually implements the complete
    acquisition pipeline for every source-object slot."""
    if not ACQUISITION_IMPLEMENTATION_COMPLETE:
        raise V9005StageABlocked(STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE)


def _normalized_coverage_references(
    coverage_references: Mapping[tuple[str, str], Sequence[str]] | None,
) -> dict[tuple[str, str], list[str]]:
    if coverage_references is None:
        return {}
    if not isinstance(coverage_references, Mapping):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    normalized: dict[tuple[str, str], list[str]] = {}
    valid_months = frozenset(inventory_months())
    for key, slot_ids in coverage_references.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        source_family, month = key
        if source_family not in MONTHLY_COVERAGE_FAMILIES or month not in valid_months:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        if isinstance(slot_ids, (str, bytes)) or not isinstance(slot_ids, Sequence):
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        for slot_id in slot_ids:
            if not isinstance(slot_id, str) or re.fullmatch(r"[0-9a-f]{64}", slot_id) is None:
                raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        normalized[(source_family, month)] = sorted(set(slot_ids))
    return normalized


def _verified_raw_lock_index(output_root: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Index only complete raw-lock pairs that pass the existing validator."""
    raw_dir = Path(output_root) / "raw"
    if not raw_dir.is_dir():
        return {}
    raw_paths = {path.with_suffix(""): path for path in raw_dir.glob("*.bin")}
    meta_paths = {path.with_suffix(""): path for path in raw_dir.glob("*.json")}
    verified: dict[str, dict[str, Any]] = {}
    for stem in sorted(set(raw_paths) & set(meta_paths)):
        try:
            raw = raw_paths[stem].read_bytes()
            meta = json.loads(meta_paths[stem].read_text(encoding="utf-8"))
        except Exception:
            continue
        if _lock_meta_matches_raw(meta, raw, expected_key=stem.name):
            verified[stem.name] = meta
    return verified


def build_source_inventory(
    coverage_references: Mapping[tuple[str, str], Sequence[str]] | None = None,
    *,
    output_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """The base MONTHLY_COVERAGE_MATRIX: exactly `MONTHLY_COVERAGE_
    FAMILIES` (F2-F7) x `inventory_months()` (108 months) = 648 records.
    F1 has no record here at all -- not AVAILABLE, not NOT_APPLICABLE_
    BY_SOURCE_CONTRACT, not MISSING -- per V9_006_F1_TERMINAL_SEED_
    PREFREEZE_AMENDMENT."""
    normalized_references = _normalized_coverage_references(coverage_references)
    referenced_slot_ids = {
        slot_id for slot_ids in normalized_references.values() for slot_id in slot_ids
    }
    if referenced_slot_ids and output_root is None:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    verified_locks = _verified_raw_lock_index(output_root) if referenced_slot_ids else {}
    for (source_family, _month), slot_ids in normalized_references.items():
        for slot_id in slot_ids:
            metadata = verified_locks.get(slot_id)
            if metadata is None or metadata["source_family"] != source_family:
                raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    records: list[dict[str, Any]] = []
    for month in inventory_months():
        for family in MONTHLY_COVERAGE_FAMILIES:
            # This foundation accepts only validated raw-lock slot-ID
            # references. Family-specific sufficiency (F3 YEAR fan-out, F6
            # GLOBAL fan-out, F5 comparability) remains future work.
            resolve_month_locator(family, month)
            source_object_slot_ids = normalized_references.get((family, month), [])
            status = INVENTORY_AVAILABLE if source_object_slot_ids else INVENTORY_MISSING
            records.append({
                "source_family": family,
                "month": month,
                "status": status,
                "source_object_slot_ids": source_object_slot_ids,
            })
    return records


def _family_fully_covered(inventory: Sequence[Mapping[str, Any]], family: str) -> bool:
    covered = False
    for record in inventory:
        if record["source_family"] != family:
            continue
        if record["status"] not in _VALID_INVENTORY_STATUSES:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        covered = True
        if record["status"] == INVENTORY_MISSING:
            return False
    return covered


# --- Semantic reconstruction (evidence items 2-8; V9_006_HIGH_2_SEMANTIC_
# VALIDATION_IMPLEMENTATION) ---------------------------------------------

def reconstruct_security_state(
    *,
    terminal_identities: Mapping[str, TerminalIdentityState] = MappingProxyType({}),
    events: Sequence[SemanticEvent] = (),
) -> dict[str, Any]:
    """Deterministic semantic reconstruction, delegating to
    `src.v9_005_stage_a_semantics.compute_semantic_validation_result` --
    replacing the prior placeholder that always reported
    `reconstructed_identity_count=0` regardless of input. With no acquired
    terminal identities (the current production state, since F2-F7
    acquisition/parser integration is separate, future, authorized work)
    this fails closed via the semantics engine's own empty-input default;
    it never fabricates a nonzero reconstruction."""
    result = compute_semantic_validation_result(terminal_identities=terminal_identities, events=events)
    return {
        "schema_version": "V9_005_STAGE_A_RECONSTRUCTION_V2",
        "reconstructed_identity_count": result["reconstructed_identity_count"],
        "canonical_state": result["canonical_state"],
    }


def reconstruction_is_deterministic(
    *,
    terminal_identities: Mapping[str, TerminalIdentityState] = MappingProxyType({}),
    events: Sequence[SemanticEvent] = (),
) -> bool:
    """Evidence item 7's first requirement: two independent deterministic
    reconstructions from identical input must produce byte-identical
    canonical state output. The Stage-A evidence gate combines this result
    with the separate reverse/forward consistency check in
    `semantic_result["deterministic_reconstruction_pass"]`."""
    first = canonical_bytes(reconstruct_security_state(terminal_identities=terminal_identities, events=events))
    second = canonical_bytes(reconstruct_security_state(terminal_identities=terminal_identities, events=events))
    return first == second


# --- Month-end crosscheck (evidence item 9) ---------------------------------

def compute_month_end_mismatch_count(
    official_aggregate_counts: Mapping[str, int],
    reconstructed_counts: Mapping[str, int],
) -> int:
    mismatches = 0
    for month, official_count in official_aggregate_counts.items():
        if month in reconstructed_counts and reconstructed_counts[month] != official_count:
            mismatches += 1
    return mismatches


# --- Evidence conjunction and FREE_JPX_METADATA_PROBE_PASS -----------------

def compute_stage_a_evidence(
    *,
    inventory: Sequence[Mapping[str, Any]],
    terminal_snapshot_locked: bool,
    trading_calendar_derived: bool,
    semantic_result: Mapping[str, Any],
    terminal_identities: Mapping[str, TerminalIdentityState],
    events: Sequence[SemanticEvent],
    comparable_month_end_mismatch_count: int,
    raw_provenance_pass: bool,
) -> dict[str, Any]:
    """V9_006_HIGH_2_SEMANTIC_VALIDATION_IMPLEMENTATION: `listing_transition_
    pass`, `delisting_transition_pass`, `market_transition_pass`,
    `security_type_pass`, `canonical_identity_pass`, `effective_date_pass`,
    are fed directly from
    `semantic_result` (produced by
    `src.v9_005_stage_a_semantics.compute_semantic_validation_result`),
    never derived from monthly `SOURCE_INVENTORY` family coverage and never
    a caller-supplied arbitrary boolean. Deterministic reconstruction is
    the conjunction of semantic_result's reverse/forward check and a fresh
    actual two-run reconstruction over these structured inputs. `terminal_snapshot_pass` remains
    an independent gate based solely on terminal-snapshot locking, and
    `trading_calendar_pass` remains based on F7 calendar-family coverage
    plus successful trading-calendar derivation -- neither of those two is
    a semantic-evidence item and neither is changed by this task."""
    required_inventory_missing_count = sum(1 for record in inventory if record["status"] == INVENTORY_MISSING)
    calendar_family_covered = _family_fully_covered(inventory, SOURCE_FAMILY_JPX_CALENDAR)
    trading_calendar_pass = bool(calendar_family_covered and trading_calendar_derived)
    two_run_determinism_pass = reconstruction_is_deterministic(
        terminal_identities=terminal_identities, events=events,
    )

    evidence: dict[str, Any] = {
        "required_inventory_missing_count": required_inventory_missing_count,
        "terminal_snapshot_pass": bool(terminal_snapshot_locked),
        "listing_transition_pass": bool(semantic_result["listing_transition_pass"]),
        "delisting_transition_pass": bool(semantic_result["delisting_transition_pass"]),
        "market_transition_pass": bool(semantic_result["market_transition_pass"]),
        "security_type_pass": bool(semantic_result["security_type_pass"]),
        "canonical_identity_pass": bool(semantic_result["canonical_identity_pass"]),
        "effective_date_pass": bool(semantic_result["effective_date_pass"]),
        "trading_calendar_pass": trading_calendar_pass,
        "deterministic_reconstruction_pass": bool(
            semantic_result["deterministic_reconstruction_pass"] and two_run_determinism_pass
        ),
        "comparable_month_end_mismatch_count": int(comparable_month_end_mismatch_count),
        "raw_provenance_pass": bool(raw_provenance_pass),
    }
    evidence["FREE_JPX_METADATA_PROBE_PASS"] = (
        evidence["required_inventory_missing_count"] == 0
        and evidence["terminal_snapshot_pass"]
        and evidence["listing_transition_pass"]
        and evidence["delisting_transition_pass"]
        and evidence["market_transition_pass"]
        and evidence["security_type_pass"]
        and evidence["canonical_identity_pass"]
        and evidence["effective_date_pass"]
        and evidence["trading_calendar_pass"]
        and evidence["deterministic_reconstruction_pass"]
        and evidence["comparable_month_end_mismatch_count"] == 0
        and evidence["raw_provenance_pass"]
    )
    evidence["failure_class"] = None if evidence["FREE_JPX_METADATA_PROBE_PASS"] else SOURCE_OR_DATA_FEASIBILITY_FAILURE
    return evidence


# --- Safe aggregate summary (contract item 6: stdout must be safe-only) ----

def build_safe_summary(
    evidence: Mapping[str, Any],
    *,
    network_request_count: int,
    signal_grid_binding_head: str,
    endpoint: Mapping[str, str] | None,
    endpoint_derivation_failure_reason: str | None,
) -> dict[str, Any]:
    endpoint = endpoint or {}
    return {
        "schema_version": "V9_005_STAGE_A_RESULT_V1",
        "study": STUDY,
        "stage": STAGE,
        "status": "PASS" if evidence["FREE_JPX_METADATA_PROBE_PASS"] else "FAIL",
        "failure_class": evidence["failure_class"],
        "required_inventory_missing_count": evidence["required_inventory_missing_count"],
        "terminal_snapshot_pass": evidence["terminal_snapshot_pass"],
        "listing_transition_pass": evidence["listing_transition_pass"],
        "delisting_transition_pass": evidence["delisting_transition_pass"],
        "market_transition_pass": evidence["market_transition_pass"],
        "security_type_pass": evidence["security_type_pass"],
        "canonical_identity_pass": evidence["canonical_identity_pass"],
        "effective_date_pass": evidence["effective_date_pass"],
        "trading_calendar_pass": evidence["trading_calendar_pass"],
        "deterministic_reconstruction_pass": evidence["deterministic_reconstruction_pass"],
        "comparable_month_end_mismatch_count": evidence["comparable_month_end_mismatch_count"],
        "raw_provenance_pass": evidence["raw_provenance_pass"],
        "network_request_count": int(network_request_count),
        "signal_grid_binding_verified_head": signal_grid_binding_head,
        "signal_grid_binding_path": BOUND_SIGNAL_GRID_PATH,
        "signal_grid_binding_blob_sha256": BOUND_SIGNAL_GRID_BLOB_SHA,
        "final_signal_d0": endpoint.get("final_signal_d0"),
        "final_planned_d3": endpoint.get("final_planned_d3"),
        "final_possible_exit_day": endpoint.get("final_possible_exit_day"),
        "stage_b_global_end_exclusive": endpoint.get("stage_b_global_end_exclusive"),
        "endpoint_derivation_failure_reason": endpoint_derivation_failure_reason,
    }


# --- Orchestration -----------------------------------------------------------

def run_stage_a(
    *,
    output_root: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    confirmation: str,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
    git: Callable[[list[str]], str] | None = None,
) -> dict[str, Any]:
    """Real-execution orchestration entrypoint. NEVER invoked with a real
    fetcher by this implementation task -- production real execution is
    gated behind the .ps1 atomic entrypoint and a fresh, separate, explicit
    human network authorization not created by this task.

    Per V9_006_HIGH_1, this stops before touching the filesystem, git, or
    the network at all if the deterministic Stage-A locator contract is not
    yet mechanically complete (see `verify_locator_contract_complete`).
    Per V9_006_LOCATOR_IMPL_HIGH_1, it then also stops -- still before any
    filesystem, git, or network access -- if the actual acquisition
    implementation for every required F1-F7 slot is not yet complete (see
    `verify_acquisition_implementation_ready`), even once the locator
    contract itself is fully bound."""
    if confirmation != CONFIRMATION:
        raise V9005StageABlocked(GOVERNANCE_FAILURE)
    verify_locator_contract_complete()
    verify_acquisition_implementation_ready()
    root = initialize_output_root(output_root)
    signal_grid_head = verify_signal_grid_binding(repo_root, git=git)

    requests_used = 0
    locked_discovery = read_locked_payload(
        root, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, TERMINAL_DISCOVERY_ROOT, LISTED_ISSUES_PAGE_URL,
    )
    if locked_discovery is None:
        discovery_result, used = fetch_once_with_retry(LISTED_ISSUES_PAGE_URL, fetcher, sleep)
        requests_used += used
        now = clock()
        locked_discovery = lock_first_complete_payload(
            root,
            source_family=SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
            applicable_period=TERMINAL_DISCOVERY_ROOT,
            requested_url=LISTED_ISSUES_PAGE_URL,
            fetch_result=discovery_result,
            retrieval_timestamp_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    derived_xls_url = extract_data_j_xls_url(locked_discovery["raw"])
    locked_terminal = read_locked_payload(
        root, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, TERMINAL_PERIOD, derived_xls_url,
    )
    if locked_terminal is None:
        xls_result, used = fetch_once_with_retry(derived_xls_url, fetcher, sleep)
        requests_used += used
        now = clock()
        locked_terminal = lock_first_complete_payload(
            root,
            source_family=SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
            applicable_period=TERMINAL_PERIOD,
            requested_url=derived_xls_url,
            fetch_result=xls_result,
            retrieval_timestamp_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    locked_calendar, used2 = ensure_locked_payload(
        root,
        source_family=SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period=CALENDAR_PERIOD,
        requested_url=CALENDAR_PAGE_URL,
        fetcher=fetcher,
        sleep=sleep,
        clock=clock,
    )
    requests_used += used2

    endpoint: dict[str, str] | None = None
    endpoint_failure_reason: str | None = None
    trading_calendar_derived = False
    try:
        holidays = parse_jpx_holiday_html(
            locked_calendar["raw"], covered_years=CALENDAR_PAGE_COVERED_YEARS, expected_row_counts=None,
        )
        trading_days = build_trading_day_set(
            [row["date"] for row in holidays], CALENDAR_PAGE_COVERAGE_START, CALENDAR_PAGE_COVERAGE_END,
        )
        endpoint = derive_stage_b_global_end_exclusive(trading_days, coverage_start=CALENDAR_PAGE_COVERAGE_START)
        trading_calendar_derived = True
    except (V7JpxCalendarBlocked, V9005StageABlocked) as exc:
        endpoint_failure_reason = getattr(exc, "reason", str(exc))

    inventory = build_source_inventory()

    # V9_006_HIGH_2_SEMANTIC_VALIDATION_IMPLEMENTATION: production has no
    # acquired F2-F7 structured semantic evidence yet -- the F2-F7
    # acquisition/parser-integration task that turns locked raw JPX bytes
    # into SemanticEvent/TerminalIdentityState input is separate, future,
    # authorized work (and this code path is unreachable in production
    # today, since `verify_acquisition_implementation_ready()` above
    # already stops every real run first). Pass empty terminal_identities/
    # events explicitly so `compute_semantic_validation_result`'s own
    # fail-closed empty-input default governs -- this must never be
    # replaced by a caller-supplied arbitrary PASS boolean.
    terminal_identities: Mapping[str, TerminalIdentityState] = {}
    events: Sequence[SemanticEvent] = ()
    semantic_result = compute_semantic_validation_result(terminal_identities=terminal_identities, events=events)
    reconstruction = reconstruct_security_state(terminal_identities=terminal_identities, events=events)
    raw_provenance_pass = verify_raw_provenance(root)

    evidence = compute_stage_a_evidence(
        inventory=inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=trading_calendar_derived,
        semantic_result=semantic_result,
        terminal_identities=terminal_identities,
        events=events,
        comparable_month_end_mismatch_count=0,
        raw_provenance_pass=raw_provenance_pass,
    )
    summary = build_safe_summary(
        evidence,
        network_request_count=requests_used,
        signal_grid_binding_head=signal_grid_head,
        endpoint=endpoint,
        endpoint_derivation_failure_reason=endpoint_failure_reason,
    )

    _atomic_create(root / "inventory.json", canonical_bytes(inventory))
    _atomic_create(root / "reconstruction.json", canonical_bytes({
        "reconstruction": reconstruction,
        "deterministic_reconstruction_pass": semantic_result["deterministic_reconstruction_pass"],
        "endpoint": endpoint,
        "endpoint_derivation_failure_reason": endpoint_failure_reason,
    }))
    _atomic_create(root / "result.json", canonical_bytes(summary))
    _atomic_create(root / "receipt.json", canonical_bytes({
        "schema_version": "V9_005_STAGE_A_RECEIPT_V1",
        "study": STUDY,
        "stage": STAGE,
        "signal_grid_binding_verified_head": signal_grid_head,
        "signal_grid_binding_path": BOUND_SIGNAL_GRID_PATH,
        "signal_grid_binding_blob_sha256": BOUND_SIGNAL_GRID_BLOB_SHA,
        "network_request_count": requests_used,
        "result_status": summary["status"],
    }))
    return summary


__all__ = [
    "ACQUISITION_IMPLEMENTATION_COMPLETE",
    "ALLOWED_HOST_SUFFIX", "BOUND_SIGNAL_GRID_BLOB_SHA", "BOUND_SIGNAL_GRID_PATH",
    "CALENDAR_ENVELOPE_FIRST_YEAR_MONTH", "CALENDAR_ENVELOPE_LAST_YEAR_MONTH",
    "CALENDAR_MONTHLY_LOCATOR_TEMPLATE", "CALENDAR_PAGE_URL", "CHATGPT_DECISION_REQUIRED", "CONFIRMATION",
    "DELISTED_COMPANY_DISCOVERY_ROOT", "DELISTED_COMPANY_ROOT_URL", "F2_SEMANTIC_ROW_LABEL", "F4_SEMANTIC_ROW_LABEL", "F6_SEMANTIC_SECTION_LABEL",
    "GOVERNANCE_FAILURE", "IMPLEMENTATION_FAILURE", "INVENTORY_AVAILABLE", "INVENTORY_MISSING",
    "INVENTORY_NOT_APPLICABLE", "LISTED_ISSUES_PAGE_URL", "LISTING_CO_ROOT_URL", "LOCATOR_STRATEGIES",
    "F2F4RequiredSlotAcquisition", "F3RequiredSlotAcquisition", "FetchResult", "LocatorStrategy", "MONTHLY_COVERAGE_FAMILIES", "MONTHLY_STATISTICS_DISCOVERY_ROOT",
    "MONTHLY_STATISTICS_ROOT_URL", "PLUMBING_FAILURE_RETRIABLE",
    "PROBE_SIGNAL_GRID_CONTRACT_MISMATCH", "SLOT_KIND_GLOBAL", "SLOT_KIND_MONTHLY", "SLOT_KIND_TERMINAL",
    "SLOT_KIND_YEAR", "SOURCE_FAMILIES", "SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE",
    "SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE", "SOURCE_FAMILY_JPX_CALENDAR",
    "SOURCE_FAMILY_LISTED_ISSUES_MONTH_END", "SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS",
    "SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT", "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE",
    "SOURCE_OR_DATA_FEASIBILITY_FAILURE", "STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE",
    "STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE", "STAGE", "STUDY",
    "TOPIX_ROOT_URL", "VALID_SLOT_KINDS",
    "V9005StageABlocked", "acquire_f2_f4_monthly_evidence", "acquire_f2_f4_required_slots", "acquire_f3_required_slots", "build_safe_summary", "build_source_inventory", "build_trading_day_set",
    "calendar_envelope_extra_months", "calendar_envelope_months", "canonical_bytes",
    "compute_month_end_mismatch_count", "compute_stage_a_evidence",
    "derive_final_signal_d0", "derive_stage_b_global_end_exclusive", "ensure_locked_payload",
    "extract_data_j_xls_url", "f2_bridge_months", "fetch_once_with_retry",
    "initialize_output_root", "inventory_months", "lock_first_complete_payload", "monthly_statistics_discovery_year_period", "nth_trading_day_after",
    "read_locked_payload", "reconstruct_security_state", "reconstruction_is_deterministic",
    "resolve_delisted_company_year_url", "resolve_f7_calendar_url", "resolve_month_locator", "resolve_monthly_statistics_evidence_url",
    "resolve_monthly_statistics_year_page_url", "run_stage_a",
    "sha256_bytes", "source_object_slot_id", "validate_jpx_url", "verify_acquisition_implementation_ready",
    "verify_locator_contract_complete", "verify_raw_provenance",
    "verify_signal_grid_binding",
]
