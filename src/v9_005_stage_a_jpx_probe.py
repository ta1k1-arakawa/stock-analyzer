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

# F6 root-structure diagnostic: a dedicated one-shot confirmation, distinct
# from production Stage-A's CONFIRMATION above. The production token must
# never satisfy this diagnostic gate -- see run_f6_root_structure_probe_
# network's confirmation check.
F6_ROOT_STRUCTURE_PROBE_CONFIRMATION = "V9_006_F6_ROOT_STRUCTURE_PROBE_ONE_SHOT"

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

# F6 root-structure diagnostic (V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_
# IMPLEMENTATION_CONTRACT): a dedicated raw-lock applicable_period, distinct
# from TOPIX_DISCOVERY_ROOT/TOPIX_GLOBAL_2017_2025, so its raw lock and
# derived artifact never alias or get mistaken for production F6
# support/evidence identity. See the offline-only helpers near the bottom
# of this module.
F6_ROOT_STRUCTURE_DIAGNOSTIC = "F6_ROOT_STRUCTURE_DIAGNOSTIC"

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

def extract_data_j_xls_url(page_bytes: bytes, page_url: str) -> str:
    try:
        text = page_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE) from exc
    match = _DATA_LINK_RE.search(text)
    if not match:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    validate_jpx_url(page_url, reason="OFF_DOMAIN_REDIRECT_REJECTED")
    resolved = urllib.parse.urljoin(page_url, match.group(1))
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


@dataclass(frozen=True)
class F7RequiredSlotAcquisition:
    """F7 base and envelope-extra calendar coverage references."""

    base_coverage_references: Mapping[tuple[str, str], tuple[str, ...]]
    envelope_extra_references: Mapping[str, tuple[str, ...]]
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


def _validate_f7_required_slot_references(
    output_root: str | os.PathLike[str],
    base_references: Mapping[tuple[str, str], tuple[str, ...]],
    extra_references: Mapping[str, tuple[str, ...]],
) -> None:
    family = SOURCE_FAMILY_JPX_CALENDAR
    base_months = inventory_months()
    extras = calendar_envelope_extra_months()
    if set(base_references) != {(family, month) for month in base_months}:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if set(extra_references) != set(extras):
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    verified_locks = _verified_raw_lock_index(output_root)

    def validate_one(slot_ids: tuple[str, ...], month: str) -> None:
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
            or source_object_slot_id(family, month, metadata.get("requested_url")) != slot_id
        ):
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

    for (_family, month), slot_ids in base_references.items():
        validate_one(slot_ids, month)
    for month, slot_ids in extra_references.items():
        validate_one(slot_ids, month)


def acquire_f7_required_slots(
    output_root: str | os.PathLike[str],
    *,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> F7RequiredSlotAcquisition:
    """Acquire every exact-template F7 envelope object in ascending order."""
    family = SOURCE_FAMILY_JPX_CALENDAR
    base_months = frozenset(inventory_months())
    base_references: dict[tuple[str, str], tuple[str, ...]] = {}
    extra_references: dict[str, tuple[str, ...]] = {}
    attempts = 0
    for month in calendar_envelope_months():
        year, month_number = _parse_year_month(month)
        requested_url = resolve_f7_calendar_url(year, month_number)
        _locked, slot_attempts = ensure_locked_payload(
            output_root,
            source_family=family,
            applicable_period=month,
            requested_url=requested_url,
            fetcher=fetcher,
            sleep=sleep,
            clock=clock,
        )
        attempts += slot_attempts
        slot_ids = (source_object_slot_id(family, month, requested_url),)
        if month in base_months:
            base_references[(family, month)] = slot_ids
        else:
            extra_references[month] = slot_ids
    _validate_f7_required_slot_references(output_root, base_references, extra_references)
    return F7RequiredSlotAcquisition(base_references, extra_references, attempts)


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


# --- F6 root-structure diagnostic (offline-only) ----------------------------
# V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_OFFLINE_IMPLEMENTATION: this section
# parses an ALREADY-LOCKED F6_ROOT_STRUCTURE_DIAGNOSTIC raw payload into the
# deterministic `V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT.json` artifact
# bound by `V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md`.
# It performs zero network I/O: every entry point here reads only an
# already-durable raw lock via the existing `read_locked_payload`, never
# accepts a fetcher/sleep/clock, and never calls `fetch_once_with_retry`,
# `ensure_locked_payload`, or `lock_first_complete_payload`. Its structural
# outcomes (`STRUCTURE_CAPTURED`/`STRUCTURE_AMBIGUOUS`/`STRUCTURE_EXTRACTION_
# FAILED`) are diagnostic-only and are never mapped to, or usable as,
# `INVENTORY_AVAILABLE`/`INVENTORY_MISSING` -- nothing here calls or feeds
# `build_source_inventory`.

STRUCTURE_CAPTURED = "STRUCTURE_CAPTURED"
STRUCTURE_AMBIGUOUS = "STRUCTURE_AMBIGUOUS"
STRUCTURE_EXTRACTION_FAILED = "STRUCTURE_EXTRACTION_FAILED"

F6_ROOT_STRUCTURE_PROBE_RESULT_SCHEMA_VERSION = "V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT_V1"
F6_ROOT_STRUCTURE_PROBE_DIAGNOSTIC_NAME = "V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE"
F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME = "V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT.json"

_F6_PAYLOAD_DECODE_FAILED = "PAYLOAD_DECODE_FAILED"
_F6_MALFORMED_DOM_STRUCTURE = "MALFORMED_DOM_STRUCTURE"
_F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE = "AMBIGUOUS_RAW_HREF_ATTRIBUTE"

_F6_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


class _F6RootStructureExtractionFailed(Exception):
    """Internal, deterministic parse-failure signal carrying a stable,
    non-secret reason token for the STRUCTURE_EXTRACTION_FAILED artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _F6DomElement:
    __slots__ = ("tag", "attrs", "children", "parent", "raw_starttag_text")

    def __init__(self, tag: str, attrs: dict[str, str | None], parent: "_F6DomElement | None") -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list["_F6DomElement | str"] = []
        self.parent = parent
        self.raw_starttag_text: str | None = None


class _F6RootStructureHtmlParser(HTMLParser):
    """Builds a full generic DOM tree with strict stack-based start/end tag
    validation (every tag, not only a fixed relevant subset), since the DOM
    path and leaf-most label rule require deterministic structure for the
    whole document, not just table markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _F6DomElement(tag="#document", attrs={}, parent=None)
        self._stack: list[_F6DomElement] = [self.root]
        self.invalid_structure = False
        self._open_anchor_count = 0

    def _open(self, tag: str, attrs: list[tuple[str, str | None]]) -> _F6DomElement:
        tag_lower = tag.lower()
        # HTML forbids nested <a>; reject it deterministically rather than
        # build an ambiguous anchor tree (mirrors the existing nested-anchor
        # rejection already reviewed in _MonthlyStatisticsHtmlParser).
        if tag_lower == "a" and self._open_anchor_count > 0:
            self.invalid_structure = True
        attributes: dict[str, str | None] = {}
        for name, value in attrs:
            key = name.lower()
            if key not in attributes:
                attributes[key] = value
        parent = self._stack[-1]
        element = _F6DomElement(tag_lower, attributes, parent)
        parent.children.append(element)
        element.raw_starttag_text = self.get_starttag_text()
        self._stack.append(element)
        if tag_lower == "a":
            self._open_anchor_count += 1
        return element

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = self._open(tag, attrs)
        if element.tag in _F6_VOID_ELEMENTS:
            self._stack.pop()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = self._open(tag, attrs)
        if element.tag == "a":
            self._open_anchor_count -= 1
        self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in _F6_VOID_ELEMENTS:
            return
        if len(self._stack) <= 1 or self._stack[-1].tag != tag_lower:
            self.invalid_structure = True
            return
        if tag_lower == "a":
            self._open_anchor_count -= 1
        self._stack.pop()

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def _f6_normalize_text(value: str) -> str:
    """Unicode whitespace runs collapsed to one ASCII space,
    leading/trailing whitespace stripped. Comparison of the result stays
    case-sensitive (no casefold applied).

    HTML character-reference resolution happens exactly once, upstream in
    `_F6RootStructureHtmlParser` (`HTMLParser(convert_charrefs=True)`),
    which already decodes references before this function ever sees the
    text. This function must NOT call `html.unescape` again: value here
    (DOM label text or anchor visible text) is already-parsed text, not
    raw source bytes -- a second pass would recursively decode a literal
    `&amp;nbsp;`/`&amp;#32;` in the original source into a real space,
    silently matching text the source never actually rendered as the
    target label (V9_006_F6_ROOT_OFFLINE_MEDIUM_1)."""
    return " ".join(value.split())


def _f6_element_siblings(parent: _F6DomElement) -> list[_F6DomElement]:
    return [child for child in parent.children if isinstance(child, _F6DomElement)]


def _f6_normalized_classes(value: str | None) -> list[str]:
    if not value:
        return []
    tokens = [token for token in re.split(r"\s+", value.strip()) if token]
    return sorted(set(tokens))


def _f6_dom_path(element: _F6DomElement) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    node: _F6DomElement | None = element
    while node is not None and node.parent is not None:
        siblings = _f6_element_siblings(node.parent)
        sibling_index = next(index for index, sibling in enumerate(siblings) if sibling is node)
        chain.append({
            "tag": node.tag,
            "sibling_index": sibling_index,
            "id": node.attrs.get("id"),
            "classes": _f6_normalized_classes(node.attrs.get("class")),
        })
        node = node.parent
    chain.reverse()
    return chain


_F6_TAG_NAME_RE = re.compile(r"^<[^\s/>]+")
_F6_ATTR_TOKEN_RE = re.compile(r'([^\s"\'>/=]+)(?:\s*=\s*("([^"]*)"|\'([^\']*)\'|([^\s"\'=<>`]+)))?')


def _f6_raw_attribute_value(starttag_text: str | None, attr_name: str) -> tuple[bool, str | None]:
    """Return (unambiguous, raw_value_or_None) reading the exact raw
    start-tag source text captured by the parser -- never HTML-entity
    decoded -- so a recorded href keeps its exact source spelling
    (including entity spelling) and is never resolved/reconstructed."""
    if not isinstance(starttag_text, str):
        return False, None
    tag_match = _F6_TAG_NAME_RE.match(starttag_text)
    if not tag_match:
        return False, None
    rest = starttag_text[tag_match.end():]
    if rest.endswith("/>"):
        rest = rest[:-2]
    elif rest.endswith(">"):
        rest = rest[:-1]
    found: list[str | None] = []
    pos = 0
    length = len(rest)
    while pos < length:
        while pos < length and rest[pos].isspace():
            pos += 1
        if pos >= length:
            break
        match = _F6_ATTR_TOKEN_RE.match(rest, pos)
        if not match or match.end() == pos:
            return False, None
        name = match.group(1)
        if match.group(2) is not None:
            value = match.group(3)
            if value is None:
                value = match.group(4)
            if value is None:
                value = match.group(5)
        else:
            value = None
        if name.lower() == attr_name:
            found.append(value)
        pos = match.end()
    if len(found) > 1:
        return False, None
    return True, (found[0] if found else None)


def _f6_anchor_of(node: _F6DomElement, raw_text: dict[int, str]) -> dict[str, Any]:
    unambiguous, raw_href = _f6_raw_attribute_value(node.raw_starttag_text, "href")
    if not unambiguous:
        raise _F6RootStructureExtractionFailed(_F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE)
    return {
        "text": _f6_normalize_text(raw_text.get(id(node), "")),
        "href": raw_href,
        "dom_path": _f6_dom_path(node),
    }


def _f6_following_element_sibling(element: _F6DomElement) -> _F6DomElement | None:
    if element.parent is None:
        return None
    siblings = _f6_element_siblings(element.parent)
    index = next(i for i, sibling in enumerate(siblings) if sibling is element)
    if index + 1 < len(siblings):
        return siblings[index + 1]
    return None


def _f6_build_anchors(element: _F6DomElement, raw_text: dict[int, str]) -> dict[str, Any]:
    self_anchor = _f6_anchor_of(element, raw_text) if element.tag == "a" else None
    children = [_f6_anchor_of(c, raw_text) for c in _f6_element_siblings(element) if c.tag == "a"]
    parent_children = (
        [_f6_anchor_of(c, raw_text) for c in _f6_element_siblings(element.parent) if c.tag == "a"]
        if element.parent is not None else []
    )
    following_sibling = _f6_following_element_sibling(element)
    following_sibling_children = (
        [_f6_anchor_of(c, raw_text) for c in _f6_element_siblings(following_sibling) if c.tag == "a"]
        if following_sibling is not None else []
    )
    return {
        "self": self_anchor,
        "children": children,
        "parent_children": parent_children,
        "following_sibling_children": following_sibling_children,
    }


def _f6_analyze_dom(
    root: _F6DomElement, target_normalized: str,
) -> tuple[list[_F6DomElement], dict[int, str], set[int]]:
    """Single traversal: document order, per-element raw descendant text,
    and the leaf-most exact-normalized-label occurrence set (an element
    matches only if none of its descendant elements also match)."""
    doc_order: list[_F6DomElement] = []
    raw_text: dict[int, str] = {}
    matches: dict[int, bool] = {}
    has_matching_descendant: dict[int, bool] = {}

    def visit(node: _F6DomElement) -> None:
        if node is not root:
            doc_order.append(node)
        text_parts: list[str] = []
        child_has_match = False
        for child in node.children:
            if isinstance(child, str):
                text_parts.append(child)
            else:
                visit(child)
                text_parts.append(raw_text[id(child)])
                if matches.get(id(child), False) or has_matching_descendant.get(id(child), False):
                    child_has_match = True
        text = "".join(text_parts)
        raw_text[id(node)] = text
        if node is not root:
            matches[id(node)] = _f6_normalize_text(text) == target_normalized
        has_matching_descendant[id(node)] = child_has_match

    visit(root)
    leaf_most_ids = {
        id(node) for node in doc_order
        if matches.get(id(node), False) and not has_matching_descendant.get(id(node), False)
    }
    return doc_order, raw_text, leaf_most_ids


def _f6_parse_full_dom(text: str) -> _F6DomElement:
    """Parse already-decoded document text into a validated full DOM tree,
    or raise `_F6RootStructureExtractionFailed(_F6_MALFORMED_DOM_STRUCTURE)`.
    Shared by the root-structure label-occurrence extractor and the section
    neighborhood probe so both use exactly one HTML normalization/parsing
    methodology -- no second, inconsistent DOM-building path is created."""
    parser = _F6RootStructureHtmlParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise _F6RootStructureExtractionFailed(_F6_MALFORMED_DOM_STRUCTURE) from exc
    if parser.invalid_structure or len(parser._stack) != 1:
        raise _F6RootStructureExtractionFailed(_F6_MALFORMED_DOM_STRUCTURE)
    return parser.root


def _f6_extract_label_occurrences(text: str) -> list[dict[str, Any]]:
    root = _f6_parse_full_dom(text)
    target_normalized = _f6_normalize_text(F6_SEMANTIC_SECTION_LABEL)
    doc_order, raw_text, leaf_most_ids = _f6_analyze_dom(root, target_normalized)
    occurrence_elements = [node for node in doc_order if id(node) in leaf_most_ids]
    return [
        {"dom_path": _f6_dom_path(node), "anchors": _f6_build_anchors(node, raw_text)}
        for node in occurrence_elements
    ]


def _f6_decode_strict_utf8(raw: bytes) -> str:
    payload = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    return payload.decode("utf-8")


def _f6_root_structure_base_fields(locked: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": F6_ROOT_STRUCTURE_PROBE_RESULT_SCHEMA_VERSION,
        "diagnostic": F6_ROOT_STRUCTURE_PROBE_DIAGNOSTIC_NAME,
        "requested_url": locked["requested_url"],
        "resolved_url": locked["resolved_url"],
        "http_status": locked["http_status"],
        "byte_length": locked["byte_length"],
        "sha256": locked["sha256"],
        "retrieval_timestamp_utc": locked["retrieval_timestamp_utc"],
        "target_label": F6_SEMANTIC_SECTION_LABEL,
    }


def read_f6_root_structure_diagnostic_lock(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Read ONLY the already-existing F6_ROOT_STRUCTURE_DIAGNOSTIC raw lock
    for the bound TOPIX_ROOT_URL. Fails closed (raises) if it is absent,
    corrupt, or does not match the expected identity. Never fetches,
    retries, sleeps, or accepts a clock -- this is a pure filesystem read."""
    locked = read_locked_payload(
        output_root,
        SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
        F6_ROOT_STRUCTURE_DIAGNOSTIC,
        TOPIX_ROOT_URL,
    )
    if locked is None:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    return locked


def parse_f6_root_structure_probe(locked: Mapping[str, Any]) -> dict[str, Any]:
    """Pure and deterministic: turn an already-locked diagnostic raw
    payload into the V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT_V1
    artifact dict. Touches no filesystem, network, or clock."""
    base = _f6_root_structure_base_fields(locked)
    try:
        text = _f6_decode_strict_utf8(locked["raw"])
    except UnicodeDecodeError:
        return {
            **base, "status": STRUCTURE_EXTRACTION_FAILED,
            "label_occurrence_count": None, "occurrences": [],
            "failure_reason": _F6_PAYLOAD_DECODE_FAILED,
        }
    try:
        occurrences = _f6_extract_label_occurrences(text)
    except _F6RootStructureExtractionFailed as exc:
        return {
            **base, "status": STRUCTURE_EXTRACTION_FAILED,
            "label_occurrence_count": None, "occurrences": [],
            "failure_reason": exc.reason,
        }
    count = len(occurrences)
    status = STRUCTURE_CAPTURED if count == 1 else STRUCTURE_AMBIGUOUS
    return {
        **base, "status": status,
        "label_occurrence_count": count, "occurrences": occurrences,
        "failure_reason": None,
    }


def write_f6_root_structure_probe_artifact(
    output_root: str | os.PathLike[str], artifact: Mapping[str, Any],
) -> Path:
    """Write the diagnostic artifact under the same dedicated diagnostic
    output_root (never production Stage-A output). First write is an
    atomic create; if the artifact already exists, reuse it only when the
    recomputed canonical bytes are byte-identical, otherwise fail closed.
    Never overwrites."""
    path = Path(output_root) / F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME
    payload = canonical_bytes(artifact)
    if path.exists():
        try:
            existing = path.read_bytes()
        except Exception as exc:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
        if existing != payload:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        return path
    _atomic_create(path, payload)
    return path


def run_f6_root_structure_probe_offline(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Fully offline seam: reads only the already-locked
    F6_ROOT_STRUCTURE_DIAGNOSTIC raw payload, parses it deterministically,
    and writes/reuses the result artifact. Never accepts a fetcher, sleep,
    or clock, and never calls a network/fetch/retry/ensure_locked_payload
    function -- SOURCE_DATA_NETWORK_REQUESTS is always 0 for this path."""
    locked = read_f6_root_structure_diagnostic_lock(output_root)
    artifact = parse_f6_root_structure_probe(locked)
    write_f6_root_structure_probe_artifact(output_root, artifact)
    return artifact


# --- F6 section neighborhood diagnostic (fully offline) ----------------------
# V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION:
# implements exactly section 4 (and the section-2.2 semantic-heading rule
# it depends on) of
# V9_006_STAGE_A_F6_ROOT_STRUCTURE_ADJUDICATION_AND_NEIGHBORHOOD_PROBE_
# DESIGN.md. This seam reads ONLY the already-existing
# F6_ROOT_STRUCTURE_DIAGNOSTIC raw lock (via the existing
# read_f6_root_structure_diagnostic_lock -- no fetcher/sleep/clock, no
# network/fetch/retry/ensure_locked_payload/lock_first_complete_payload
# call, no new or modified raw lock) and reuses the existing reviewed F6
# DOM/parser/raw-href utilities verbatim. It never resolves or follows any
# href, never selects/ranks/binds a GLOBAL child, and never maps its
# diagnostic-only outcomes to F6 AVAILABLE/MISSING.

F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION = "V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_V1"
F6_SECTION_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME = "V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE"
F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME = "V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT.json"

NEIGHBORHOOD_CAPTURED = "NEIGHBORHOOD_CAPTURED"
SEMANTIC_HEADING_AMBIGUOUS = "SEMANTIC_HEADING_AMBIGUOUS"
# STRUCTURE_EXTRACTION_FAILED (already defined above for the root-structure
# probe) is reused verbatim as this diagnostic's third outcome.

NEIGHBORHOOD_RELATION_BEFORE_HEADING = "BEFORE_HEADING"
NEIGHBORHOOD_RELATION_HEADING = "HEADING"
NEIGHBORHOOD_RELATION_AFTER_HEADING = "AFTER_HEADING"
NEIGHBORHOOD_RELATION_INSIDE_HEADING = "INSIDE_HEADING"

_F6_REQUIRED_SEMANTIC_HEADING_TAG = "h2"
_F6_REQUIRED_SEMANTIC_HEADING_CLASS = "heading-title"
_F6_FRAGMENT_HREF_RE = re.compile(r"^#([^#]+)$")
_F6_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def _f6_is_self_or_descendant(node: _F6DomElement, ancestor: _F6DomElement) -> bool:
    current: _F6DomElement | None = node
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False


def _f6_is_proper_descendant(node: _F6DomElement, ancestor: _F6DomElement) -> bool:
    return node is not ancestor and _f6_is_self_or_descendant(node, ancestor)


def _f6_element_identity(node: _F6DomElement) -> dict[str, Any]:
    return {
        "dom_path": _f6_dom_path(node),
        "tag": node.tag,
        "id": node.attrs.get("id"),
        "classes": _f6_normalized_classes(node.attrs.get("class")),
    }


def _f6_identify_semantic_heading(
    doc_order: Sequence[_F6DomElement], occurrence_elements: Sequence[_F6DomElement],
) -> _F6DomElement | None:
    """Deterministic semantic-heading identity rule (design section 2.2).
    Returns None on any zero/multiple/wrong-tag/wrong-class/inconsistent
    outcome -- the caller reports SEMANTIC_HEADING_AMBIGUOUS, never a
    fallback, ranked, or guessed candidate. No literal fragment or id value
    (including `heading_14`) is ever hardcoded; every candidate is derived
    only from the locked payload's own leaf-most label occurrences and DOM
    `id` attributes each time this runs."""
    candidates: list[tuple[_F6DomElement, str]] = []
    for node in occurrence_elements:
        if node.tag != "a":
            continue
        unambiguous, raw_href = _f6_raw_attribute_value(node.raw_starttag_text, "href")
        if not unambiguous:
            raise _F6RootStructureExtractionFailed(_F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE)
        if raw_href is None:
            continue
        match = _F6_FRAGMENT_HREF_RE.match(raw_href)
        if match:
            candidates.append((node, match.group(1)))
    if len(candidates) != 1:
        return None
    _, fragment_id = candidates[0]

    id_matches = [node for node in doc_order if node.attrs.get("id") == fragment_id]
    if len(id_matches) != 1:
        return None
    target = id_matches[0]

    if target.tag != _F6_REQUIRED_SEMANTIC_HEADING_TAG:
        return None
    if _F6_REQUIRED_SEMANTIC_HEADING_CLASS not in _f6_normalized_classes(target.attrs.get("class")):
        return None

    # Design section 2.2 step 6 requires the qualifying leaf-most exact-label
    # occurrence to be found AMONG THE TARGET H2'S DESCENDANTS -- the target
    # itself must never satisfy this step (V9_006_F6_NEIGHBORHOOD_MEDIUM_1).
    # Using self-or-descendant here would let an h2 whose own entire text is
    # the label count as its own qualifying occurrence, silently broadening
    # the frozen methodology beyond what section 2.2 actually specifies.
    contained = [node for node in occurrence_elements if _f6_is_proper_descendant(node, target)]
    if len(contained) != 1:
        return None
    return target


def _f6_neighborhood_children(parent: _F6DomElement, heading: _F6DomElement) -> list[dict[str, Any]]:
    siblings = _f6_element_siblings(parent)
    heading_index = next(index for index, sibling in enumerate(siblings) if sibling is heading)
    children: list[dict[str, Any]] = []
    for index, node in enumerate(siblings):
        if index < heading_index:
            relation = NEIGHBORHOOD_RELATION_BEFORE_HEADING
        elif index == heading_index:
            relation = NEIGHBORHOOD_RELATION_HEADING
        else:
            relation = NEIGHBORHOOD_RELATION_AFTER_HEADING
        children.append({**_f6_element_identity(node), "relation": relation})
    return children


def _f6_neighborhood_anchors_and_headings(
    parent: _F6DomElement, heading: _F6DomElement,
    doc_order: Sequence[_F6DomElement], raw_text: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_of = {id(node): index for index, node in enumerate(doc_order)}
    heading_index = index_of[id(heading)]
    anchors: list[dict[str, Any]] = []
    headings: list[dict[str, Any]] = []
    for node in doc_order:
        if not _f6_is_proper_descendant(node, parent):
            continue
        if _f6_is_self_or_descendant(node, heading):
            relation = NEIGHBORHOOD_RELATION_INSIDE_HEADING
        elif index_of[id(node)] < heading_index:
            relation = NEIGHBORHOOD_RELATION_BEFORE_HEADING
        else:
            relation = NEIGHBORHOOD_RELATION_AFTER_HEADING
        if node.tag == "a":
            anchor = _f6_anchor_of(node, raw_text)
            anchor["relation"] = relation
            anchors.append(anchor)
        elif node.tag in _F6_HEADING_TAGS:
            headings.append({
                "dom_path": _f6_dom_path(node),
                "tag": node.tag,
                "text": _f6_normalize_text(raw_text.get(id(node), "")),
            })
    return anchors, headings


def _f6_neighborhood_base_fields(locked: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION,
        "diagnostic": F6_SECTION_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME,
        "requested_url": locked["requested_url"],
        "resolved_url": locked["resolved_url"],
        "byte_length": locked["byte_length"],
        "sha256": locked["sha256"],
        "retrieval_timestamp_utc": locked["retrieval_timestamp_utc"],
    }


_F6_NEIGHBORHOOD_EMPTY_RESULT_FIELDS: dict[str, Any] = {
    "semantic_heading": None, "parent_container": None,
    "children": [], "anchors": [], "headings": [],
}


def parse_f6_section_neighborhood_probe(locked: Mapping[str, Any]) -> dict[str, Any]:
    """Pure and deterministic: derive the
    V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_V1 artifact from an
    already-locked F6_ROOT_STRUCTURE_DIAGNOSTIC raw payload. Touches no
    filesystem, network, or clock. Never records arbitrary page text,
    numerical TOPIX/index observations, raw payload bytes, a resolved href,
    or a chosen/ranked child URL."""
    base = _f6_neighborhood_base_fields(locked)
    try:
        text = _f6_decode_strict_utf8(locked["raw"])
    except UnicodeDecodeError:
        return {
            **base, **_F6_NEIGHBORHOOD_EMPTY_RESULT_FIELDS,
            "status": STRUCTURE_EXTRACTION_FAILED, "failure_reason": _F6_PAYLOAD_DECODE_FAILED,
        }
    try:
        root = _f6_parse_full_dom(text)
        target_normalized = _f6_normalize_text(F6_SEMANTIC_SECTION_LABEL)
        doc_order, raw_text, leaf_most_ids = _f6_analyze_dom(root, target_normalized)
        occurrence_elements = [node for node in doc_order if id(node) in leaf_most_ids]
        heading = _f6_identify_semantic_heading(doc_order, occurrence_elements)
        if heading is None:
            return {
                **base, **_F6_NEIGHBORHOOD_EMPTY_RESULT_FIELDS,
                "status": SEMANTIC_HEADING_AMBIGUOUS, "failure_reason": None,
            }
        parent = heading.parent
        if parent is None:
            # The semantic heading is the document's own top-level node --
            # there is no immediate parent element to scope a container
            # around. Deterministic, but not a captured neighborhood.
            return {
                **base, **_F6_NEIGHBORHOOD_EMPTY_RESULT_FIELDS,
                "status": STRUCTURE_EXTRACTION_FAILED, "failure_reason": _F6_MALFORMED_DOM_STRUCTURE,
            }
        children = _f6_neighborhood_children(parent, heading)
        anchors, headings = _f6_neighborhood_anchors_and_headings(parent, heading, doc_order, raw_text)
    except _F6RootStructureExtractionFailed as exc:
        return {
            **base, **_F6_NEIGHBORHOOD_EMPTY_RESULT_FIELDS,
            "status": STRUCTURE_EXTRACTION_FAILED, "failure_reason": exc.reason,
        }
    return {
        **base,
        "status": NEIGHBORHOOD_CAPTURED,
        "failure_reason": None,
        "semantic_heading": _f6_element_identity(heading),
        "parent_container": _f6_element_identity(parent),
        "children": children,
        "anchors": anchors,
        "headings": headings,
    }


def write_f6_section_neighborhood_probe_artifact(
    output_root: str | os.PathLike[str], artifact: Mapping[str, Any],
) -> Path:
    """Write the section-neighborhood diagnostic artifact under the same
    dedicated diagnostic output_root (never production Stage-A output).
    First write is an atomic create; if the artifact already exists, reuse
    it only when the recomputed canonical bytes are byte-identical,
    otherwise fail closed. Never overwrites."""
    path = Path(output_root) / F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME
    payload = canonical_bytes(artifact)
    if path.exists():
        try:
            existing = path.read_bytes()
        except Exception as exc:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
        if existing != payload:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        return path
    _atomic_create(path, payload)
    return path


def run_f6_section_neighborhood_probe_offline(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Fully offline seam: reads only the already-locked
    F6_ROOT_STRUCTURE_DIAGNOSTIC raw payload (the same diagnostic raw lock
    the root-structure probe reuses -- no new raw lock is created or
    modified), parses it deterministically, and writes/reuses the
    section-neighborhood result artifact. Never accepts a fetcher, sleep,
    or clock, and never calls a network/fetch/retry/ensure_locked_payload/
    lock_first_complete_payload function -- SOURCE_DATA_NETWORK_REQUESTS is
    always 0 for this path. Never selects, ranks, or binds a GLOBAL child."""
    locked = read_f6_root_structure_diagnostic_lock(output_root)
    artifact = parse_f6_section_neighborhood_probe(locked)
    write_f6_section_neighborhood_probe_artifact(output_root, artifact)
    return artifact


# --- F6 one-level expanded neighborhood diagnostic (fully offline) ----------
# V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION:
# implements the exact one-level-expanded scope G from the reviewed design.
# It reuses the shared F6 DOM, text-normalization, raw-href, identity, lock
# reader, canonical serialization, and atomic-create utilities above. It reads
# only the existing F6_ROOT_STRUCTURE_DIAGNOSTIC raw lock, never accepts a
# fetcher/sleep/clock, performs no network operation, and never selects or
# binds a GLOBAL child.

F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION = (
    "V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_V1"
)
F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME = (
    "V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE"
)
F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME = (
    "V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT.json"
)

EXPANDED_NEIGHBORHOOD_CAPTURED = "EXPANDED_NEIGHBORHOOD_CAPTURED"
ONE_LEVEL_RELATION_BEFORE_P = "BEFORE_P"
ONE_LEVEL_RELATION_P = "P"
ONE_LEVEL_RELATION_AFTER_P = "AFTER_P"

_F6_ONE_LEVEL_PARENT_MISSING = "ONE_LEVEL_PARENT_MISSING"
_F6_ONE_LEVEL_EXPANDED_PARENT_MISSING = "ONE_LEVEL_EXPANDED_PARENT_MISSING"
_F6_ONE_LEVEL_PARENT_NOT_DIRECT_CHILD = "ONE_LEVEL_PARENT_NOT_DIRECT_CHILD"
_F6_ONE_LEVEL_OWNER_NOT_DIRECT_CHILD = "ONE_LEVEL_OWNER_NOT_DIRECT_CHILD"


def _f6_one_level_empty_result_fields() -> dict[str, Any]:
    return {
        "semantic_heading": None,
        "parent_container": None,
        "expanded_container": None,
        "children": [],
        "anchors": [],
        "headings": [],
    }


def _f6_one_level_children(
    expanded_parent: _F6DomElement, parent: _F6DomElement,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    direct_children = _f6_element_siblings(expanded_parent)
    parent_matches = [child for child in direct_children if child is parent]
    if len(parent_matches) != 1:
        raise _F6RootStructureExtractionFailed(_F6_ONE_LEVEL_PARENT_NOT_DIRECT_CHILD)
    parent_index = next(index for index, child in enumerate(direct_children) if child is parent)

    children: list[dict[str, Any]] = []
    relation_by_child_id: dict[int, str] = {}
    for index, child in enumerate(direct_children):
        if index < parent_index:
            relation = ONE_LEVEL_RELATION_BEFORE_P
        elif index == parent_index:
            relation = ONE_LEVEL_RELATION_P
        else:
            relation = ONE_LEVEL_RELATION_AFTER_P
        relation_by_child_id[id(child)] = relation
        children.append({**_f6_element_identity(child), "relation_to_P": relation})
    return children, relation_by_child_id


def _f6_one_level_owner(
    node: _F6DomElement, expanded_parent: _F6DomElement,
) -> _F6DomElement:
    current = node
    while current.parent is not expanded_parent:
        if current.parent is None:
            raise _F6RootStructureExtractionFailed(_F6_ONE_LEVEL_OWNER_NOT_DIRECT_CHILD)
        current = current.parent
    return current


def _f6_one_level_descendant_records(
    expanded_parent: _F6DomElement,
    doc_order: Sequence[_F6DomElement],
    raw_text: Mapping[int, str],
    relation_by_child_id: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors: list[dict[str, Any]] = []
    headings: list[dict[str, Any]] = []
    for node in doc_order:
        if not _f6_is_proper_descendant(node, expanded_parent):
            continue
        if node.tag != "a" and node.tag not in _F6_HEADING_TAGS:
            continue

        owner = _f6_one_level_owner(node, expanded_parent)
        relation = relation_by_child_id.get(id(owner))
        if relation is None:
            raise _F6RootStructureExtractionFailed(_F6_ONE_LEVEL_OWNER_NOT_DIRECT_CHILD)
        owner_identity = _f6_element_identity(owner)

        if node.tag == "a":
            anchor = _f6_anchor_of(node, raw_text)
            anchors.append({
                "dom_path": anchor["dom_path"],
                "normalized_visible_text": anchor["text"],
                "raw_href": anchor["href"],
                "owning_immediate_element_child_of_G": owner_identity,
                "owning_child_relation_to_P": relation,
            })
        else:
            headings.append({
                "dom_path": _f6_dom_path(node),
                "tag": node.tag,
                "normalized_heading_text": _f6_normalize_text(raw_text.get(id(node), "")),
                "owning_immediate_element_child_of_G": owner_identity,
                "owning_child_relation_to_P": relation,
            })
    return anchors, headings


def _f6_one_level_base_fields(locked: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION,
        "diagnostic": F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME,
        "requested_url": locked["requested_url"],
        "resolved_url": locked["resolved_url"],
        "byte_length": locked["byte_length"],
        "sha256": locked["sha256"],
        "retrieval_timestamp_utc": locked["retrieval_timestamp_utc"],
    }


def parse_f6_one_level_expanded_neighborhood_probe(
    locked: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact G-scope artifact from one existing F6 raw lock.

    The parser and DOM analysis are the already-reviewed F6 utilities. This
    function touches no filesystem, network, or clock, and records no page
    text, numerical observation, resolved href, or selected child URL.
    """
    base = _f6_one_level_base_fields(locked)
    try:
        text = _f6_decode_strict_utf8(locked["raw"])
    except UnicodeDecodeError:
        return {
            **base,
            **_f6_one_level_empty_result_fields(),
            "status": STRUCTURE_EXTRACTION_FAILED,
            "failure_reason": _F6_PAYLOAD_DECODE_FAILED,
        }

    try:
        root = _f6_parse_full_dom(text)
        target_normalized = _f6_normalize_text(F6_SEMANTIC_SECTION_LABEL)
        doc_order, raw_text, leaf_most_ids = _f6_analyze_dom(root, target_normalized)
        occurrence_elements = [node for node in doc_order if id(node) in leaf_most_ids]
        heading = _f6_identify_semantic_heading(doc_order, occurrence_elements)
        if heading is None:
            return {
                **base,
                **_f6_one_level_empty_result_fields(),
                "status": SEMANTIC_HEADING_AMBIGUOUS,
                "failure_reason": None,
            }

        parent = heading.parent
        if parent is None or parent is root:
            raise _F6RootStructureExtractionFailed(_F6_ONE_LEVEL_PARENT_MISSING)
        expanded_parent = parent.parent
        if expanded_parent is None or expanded_parent is root:
            raise _F6RootStructureExtractionFailed(_F6_ONE_LEVEL_EXPANDED_PARENT_MISSING)

        children, relation_by_child_id = _f6_one_level_children(expanded_parent, parent)
        anchors, headings = _f6_one_level_descendant_records(
            expanded_parent, doc_order, raw_text, relation_by_child_id,
        )
    except _F6RootStructureExtractionFailed as exc:
        return {
            **base,
            **_f6_one_level_empty_result_fields(),
            "status": STRUCTURE_EXTRACTION_FAILED,
            "failure_reason": exc.reason,
        }

    return {
        **base,
        "status": EXPANDED_NEIGHBORHOOD_CAPTURED,
        "failure_reason": None,
        "semantic_heading": _f6_element_identity(heading),
        "parent_container": _f6_element_identity(parent),
        "expanded_container": _f6_element_identity(expanded_parent),
        "children": children,
        "anchors": anchors,
        "headings": headings,
    }


def write_f6_one_level_expanded_neighborhood_probe_artifact(
    output_root: str | os.PathLike[str], artifact: Mapping[str, Any],
) -> Path:
    """Atomically create or byte-identically reuse the G-scope artifact.

    A divergent existing artifact fails closed and is never overwritten.
    """
    path = Path(output_root) / F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME
    payload = canonical_bytes(artifact)
    if path.exists():
        try:
            existing = path.read_bytes()
        except Exception as exc:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc
        if existing != payload:
            raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
        return path
    _atomic_create(path, payload)
    return path


def run_f6_one_level_expanded_neighborhood_probe_offline(
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read only the existing F6 diagnostic raw lock and produce the G artifact.

    This seam accepts no fetcher, sleep, or clock, performs no network
    operation, and never selects, ranks, or binds an F6 GLOBAL child.
    """
    locked = read_f6_root_structure_diagnostic_lock(output_root)
    artifact = parse_f6_one_level_expanded_neighborhood_probe(locked)
    write_f6_one_level_expanded_neighborhood_probe_artifact(output_root, artifact)
    return artifact



# --- F6 root-structure diagnostic (network executor) -------------------------
# V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_NETWORK_EXECUTOR: executable
# plumbing only. NEVER invoked with a real fetcher by this implementation
# task. Sets no network authorization flag, consumes no production human
# authorization, and does not authorize production Stage A, a GLOBAL-child
# fetch, or F5/any other source. Future real execution still requires its
# own fresh, explicit, one-shot human authorization obtained AFTER GPT
# exact-SHA review of this implementation, per
# V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md.

def run_f6_root_structure_probe_network(
    *,
    output_root: str | os.PathLike[str],
    confirmation: str,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    """Real-execution diagnostic entrypoint for exactly TOPIX_ROOT_URL --
    one logical source object, no discovery of any other URL, no href
    following, no child fetch. Requires its own dedicated one-shot
    F6_ROOT_STRUCTURE_PROBE_CONFIRMATION; production Stage-A's CONFIRMATION
    does NOT satisfy this gate. The confirmation check happens before any
    filesystem initialization or fetcher call. output_root must be brand
    new -- initialize_output_root is called exactly once and itself fails
    closed if output_root already exists, so rerunning against an
    already-used output_root fails closed rather than acquiring/refetching.
    Acquisition uses only the existing reviewed fetch_once_with_retry
    retry/backoff/redirect policy verbatim -- no alternate URL, provider,
    manual retry, or fallback. The first complete payload is raw-locked via
    lock_first_complete_payload BEFORE any parsing. Only then is the
    already-reviewed offline seam (run_f6_root_structure_probe_offline)
    invoked to parse the just-locked bytes -- this never duplicates parser
    logic, and a parser/extraction failure after the raw lock exists never
    triggers a refetch or a child request; the raw lock and whatever
    deterministic artifact the offline seam produces are both preserved."""
    if confirmation != F6_ROOT_STRUCTURE_PROBE_CONFIRMATION:
        raise V9005StageABlocked(GOVERNANCE_FAILURE)
    root = initialize_output_root(output_root)
    result, requests_used = fetch_once_with_retry(TOPIX_ROOT_URL, fetcher, sleep)
    now = clock()
    lock_first_complete_payload(
        root,
        source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
        applicable_period=F6_ROOT_STRUCTURE_DIAGNOSTIC,
        requested_url=TOPIX_ROOT_URL,
        fetch_result=result,
        retrieval_timestamp_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    artifact = run_f6_root_structure_probe_offline(root)
    return {**artifact, "network_request_count": requests_used}


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
    derived_xls_url = extract_data_j_xls_url(locked_discovery["raw"], locked_discovery["resolved_url"])
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
    "F6_ROOT_STRUCTURE_DIAGNOSTIC", "F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME",
    "F6_ROOT_STRUCTURE_PROBE_RESULT_SCHEMA_VERSION", "F6_ROOT_STRUCTURE_PROBE_DIAGNOSTIC_NAME",
    "F6_ROOT_STRUCTURE_PROBE_CONFIRMATION",
    "F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION", "F6_SECTION_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME",
    "F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME",
    "NEIGHBORHOOD_CAPTURED", "SEMANTIC_HEADING_AMBIGUOUS",
    "NEIGHBORHOOD_RELATION_BEFORE_HEADING", "NEIGHBORHOOD_RELATION_HEADING",
    "NEIGHBORHOOD_RELATION_AFTER_HEADING", "NEIGHBORHOOD_RELATION_INSIDE_HEADING",
    "F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION",
    "F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME",
    "F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME",
    "EXPANDED_NEIGHBORHOOD_CAPTURED", "ONE_LEVEL_RELATION_BEFORE_P",
    "ONE_LEVEL_RELATION_P", "ONE_LEVEL_RELATION_AFTER_P",
    "GOVERNANCE_FAILURE", "IMPLEMENTATION_FAILURE", "INVENTORY_AVAILABLE", "INVENTORY_MISSING",
    "INVENTORY_NOT_APPLICABLE", "LISTED_ISSUES_PAGE_URL", "LISTING_CO_ROOT_URL", "LOCATOR_STRATEGIES",
    "F2F4RequiredSlotAcquisition", "F3RequiredSlotAcquisition", "F7RequiredSlotAcquisition", "FetchResult", "LocatorStrategy", "MONTHLY_COVERAGE_FAMILIES", "MONTHLY_STATISTICS_DISCOVERY_ROOT",
    "MONTHLY_STATISTICS_ROOT_URL", "PLUMBING_FAILURE_RETRIABLE",
    "PROBE_SIGNAL_GRID_CONTRACT_MISMATCH", "SLOT_KIND_GLOBAL", "SLOT_KIND_MONTHLY", "SLOT_KIND_TERMINAL",
    "SLOT_KIND_YEAR", "SOURCE_FAMILIES", "SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE",
    "SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE", "SOURCE_FAMILY_JPX_CALENDAR",
    "SOURCE_FAMILY_LISTED_ISSUES_MONTH_END", "SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS",
    "SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT", "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE",
    "SOURCE_OR_DATA_FEASIBILITY_FAILURE", "STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE",
    "STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE", "STAGE", "STUDY",
    "STRUCTURE_CAPTURED", "STRUCTURE_AMBIGUOUS", "STRUCTURE_EXTRACTION_FAILED",
    "TOPIX_ROOT_URL", "VALID_SLOT_KINDS",
    "V9005StageABlocked", "acquire_f2_f4_monthly_evidence", "acquire_f2_f4_required_slots", "acquire_f3_required_slots", "acquire_f7_required_slots", "build_safe_summary", "build_source_inventory", "build_trading_day_set",
    "calendar_envelope_extra_months", "calendar_envelope_months", "canonical_bytes",
    "compute_month_end_mismatch_count", "compute_stage_a_evidence",
    "derive_final_signal_d0", "derive_stage_b_global_end_exclusive", "ensure_locked_payload",
    "extract_data_j_xls_url", "f2_bridge_months", "fetch_once_with_retry",
    "initialize_output_root", "inventory_months", "lock_first_complete_payload", "monthly_statistics_discovery_year_period", "nth_trading_day_after",
    "parse_f6_root_structure_probe", "parse_f6_section_neighborhood_probe", "read_f6_root_structure_diagnostic_lock",
    "parse_f6_one_level_expanded_neighborhood_probe",
    "read_locked_payload", "reconstruct_security_state", "reconstruction_is_deterministic",
    "resolve_delisted_company_year_url", "resolve_f7_calendar_url", "resolve_month_locator", "resolve_monthly_statistics_evidence_url",
    "resolve_monthly_statistics_year_page_url", "run_f6_root_structure_probe_offline",
    "run_f6_root_structure_probe_network", "run_f6_section_neighborhood_probe_offline", "run_stage_a",
    "run_f6_one_level_expanded_neighborhood_probe_offline",
    "sha256_bytes", "source_object_slot_id", "validate_jpx_url", "verify_acquisition_implementation_ready",
    "verify_locator_contract_complete", "verify_raw_provenance",
    "verify_signal_grid_binding", "write_f6_root_structure_probe_artifact",
    "write_f6_section_neighborhood_probe_artifact",
    "write_f6_one_level_expanded_neighborhood_probe_artifact",
]
