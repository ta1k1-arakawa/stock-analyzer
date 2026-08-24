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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v7_jpx_calendar import V7JpxCalendarBlocked, parse_jpx_holiday_html
from src.v8c_transport import classify_transport_exception

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

# F3: delisted-company archive (YEAR objects).
DELISTED_COMPANY_ROOT_URL = "https://www.jpx.co.jp/english/listing/stocks/delisted/index.html"

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

def fetch_once_with_retry(
    url: str,
    fetcher: Callable[[str], tuple[bytes, str]],
    sleep: Callable[[int], None],
) -> tuple[bytes, str, int]:
    """Fetch url, rejecting off-domain requests/redirects before content is
    consumed. Retries only classified retryable transport failures, up to
    the frozen attempt/backoff policy, per AI_REAL_EXECUTION_RUNBOOK.md."""
    validate_jpx_url(url, reason="OFF_DOMAIN_REQUEST_REJECTED")
    requests_used = 0
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            requests_used += 1
            payload, final_url = fetcher(url)
            validate_jpx_url(final_url, reason="OFF_DOMAIN_REDIRECT_REJECTED")
            if not payload:
                raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE, network_request_count=requests_used)
            return payload, final_url, requests_used
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


def _raw_paths(output_root: Path, key: str) -> tuple[Path, Path]:
    raw_dir = Path(output_root) / "raw"
    return raw_dir / (key + ".bin"), raw_dir / (key + ".json")


_REQUIRED_LOCK_META_FIELDS = frozenset({
    "schema_version", "source_family", "applicable_period", "requested_url",
    "resolved_url", "http_status", "retrieval_timestamp_utc", "byte_length", "sha256",
})


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
    if (
        not isinstance(meta, dict)
        or set(meta) != _REQUIRED_LOCK_META_FIELDS
        or meta["sha256"] != sha256_bytes(raw)
        or meta["byte_length"] != len(raw)
    ):
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
    resolved_url: str,
    http_status: int,
    payload: bytes,
    retrieval_timestamp_utc: str,
) -> dict[str, Any]:
    if source_family not in SOURCE_FAMILIES:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE)
    if not payload:
        raise V9005StageABlocked(SOURCE_OR_DATA_FEASIBILITY_FAILURE)
    key = _record_key(source_family, applicable_period, requested_url)
    raw_path, meta_path = _raw_paths(Path(output_root), key)
    meta = {
        "schema_version": "V9_005_STAGE_A_RAW_LOCK_V1",
        "source_family": source_family,
        "applicable_period": applicable_period,
        "requested_url": requested_url,
        "resolved_url": resolved_url,
        "http_status": int(http_status),
        "retrieval_timestamp_utc": retrieval_timestamp_utc,
        "byte_length": len(payload),
        "sha256": sha256_bytes(payload),
    }
    _atomic_create(raw_path, payload)
    _atomic_create(meta_path, canonical_bytes(meta))
    return {"raw": payload, **meta}


def ensure_locked_payload(
    output_root: str | os.PathLike[str],
    *,
    source_family: str,
    applicable_period: str,
    requested_url: str,
    fetcher: Callable[[str], tuple[bytes, str]],
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> tuple[dict[str, Any], int]:
    """Never fetch twice for the same key: reprocess already-locked bytes."""
    existing = read_locked_payload(output_root, source_family, applicable_period, requested_url)
    if existing is not None:
        return existing, 0
    payload, final_url, requests_used = fetch_once_with_retry(requested_url, fetcher, sleep)
    now = clock()
    locked = lock_first_complete_payload(
        output_root,
        source_family=source_family,
        applicable_period=applicable_period,
        requested_url=requested_url,
        resolved_url=final_url,
        http_status=200,
        payload=payload,
        retrieval_timestamp_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return locked, requests_used


def verify_raw_provenance(output_root: str | os.PathLike[str]) -> bool:
    """Independently re-verify every locked raw/meta pair on disk has the
    complete required provenance field set and a matching hash."""
    raw_dir = Path(output_root) / "raw"
    if not raw_dir.exists():
        return True
    for meta_path in sorted(raw_dir.glob("*.json")):
        raw_path = meta_path.with_suffix(".bin")
        if not raw_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = raw_path.read_bytes()
        except Exception:
            return False
        if (
            not isinstance(meta, dict)
            or set(meta) != _REQUIRED_LOCK_META_FIELDS
            or meta["sha256"] != sha256_bytes(raw)
            or meta["byte_length"] != len(raw)
        ):
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


def build_source_inventory(
    locked_index: Mapping[tuple[str, str], Any] | None = None,
) -> list[dict[str, Any]]:
    """The base MONTHLY_COVERAGE_MATRIX: exactly `MONTHLY_COVERAGE_
    FAMILIES` (F2-F7) x `inventory_months()` (108 months) = 648 records.
    F1 has no record here at all -- not AVAILABLE, not NOT_APPLICABLE_
    BY_SOURCE_CONTRACT, not MISSING -- per V9_006_F1_TERMINAL_SEED_
    PREFREEZE_AMENDMENT."""
    locked_index = locked_index or {}
    records: list[dict[str, Any]] = []
    for month in inventory_months():
        for family in MONTHLY_COVERAGE_FAMILIES:
            # resolve_month_locator validates the reviewed strategy exists;
            # a cell is AVAILABLE only once actually present in
            # locked_index (i.e. really fetched and locked this run),
            # otherwise MISSING -- never a guessed AVAILABLE/NOT_APPLICABLE
            # status.
            resolve_month_locator(family, month)
            if (family, month) in locked_index:
                status = INVENTORY_AVAILABLE
            else:
                status = INVENTORY_MISSING
            records.append({"source_family": family, "month": month, "status": status})
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


# --- Reconstruction determinism (evidence item 8) ---------------------------

def reconstruct_security_state(locked_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically fold whatever locked terminal-snapshot bytes exist
    into a canonical reconstructed-state summary. With no locked evidence
    this deterministically returns an empty reconstruction; the property
    verified by `reconstruction_is_deterministic` is that identical input
    always yields byte-identical output, independent of evidence volume."""
    terminal = locked_evidence.get("terminal_snapshot")
    terminal_sha256 = terminal["sha256"] if isinstance(terminal, Mapping) else None
    return {
        "schema_version": "V9_005_STAGE_A_RECONSTRUCTION_V1",
        "terminal_snapshot_sha256": terminal_sha256,
        "reconstructed_identity_count": 0,
    }


def reconstruction_is_deterministic(locked_evidence: Mapping[str, Any]) -> bool:
    first = canonical_bytes(reconstruct_security_state(locked_evidence))
    second = canonical_bytes(reconstruct_security_state(locked_evidence))
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
    reconstruction_deterministic: bool,
    comparable_month_end_mismatch_count: int,
    raw_provenance_pass: bool,
) -> dict[str, Any]:
    required_inventory_missing_count = sum(1 for record in inventory if record["status"] == INVENTORY_MISSING)
    listing_transition_pass = _family_fully_covered(inventory, SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT)
    delisting_transition_pass = _family_fully_covered(inventory, SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE)
    market_transition_pass = listing_transition_pass
    # Per V9_006_F1_TERMINAL_SEED_PREFREEZE_AMENDMENT, F1 (source of
    # security-type classification) has zero MONTHLY_COVERAGE_MATRIX cells
    # -- it is gated exclusively by the mandatory TERMINAL object, not
    # monthly-family coverage.
    security_type_pass = bool(terminal_snapshot_locked)
    effective_date_pass = listing_transition_pass and delisting_transition_pass and market_transition_pass
    canonical_identity_pass = bool(terminal_snapshot_locked) and security_type_pass
    calendar_family_covered = _family_fully_covered(inventory, SOURCE_FAMILY_JPX_CALENDAR)
    trading_calendar_pass = bool(calendar_family_covered and trading_calendar_derived)

    evidence: dict[str, Any] = {
        "required_inventory_missing_count": required_inventory_missing_count,
        "terminal_snapshot_pass": bool(terminal_snapshot_locked),
        "listing_transition_pass": listing_transition_pass,
        "delisting_transition_pass": delisting_transition_pass,
        "market_transition_pass": market_transition_pass,
        "security_type_pass": security_type_pass,
        "canonical_identity_pass": canonical_identity_pass,
        "effective_date_pass": effective_date_pass,
        "trading_calendar_pass": trading_calendar_pass,
        "deterministic_reconstruction_pass": bool(reconstruction_deterministic),
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
    fetcher: Callable[[str], tuple[bytes, str]],
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
    locked_terminal = read_locked_payload(
        root, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, TERMINAL_PERIOD, LISTED_ISSUES_PAGE_URL,
    )
    if locked_terminal is None:
        xls_bytes, xls_final_url, used = fetch_terminal_snapshot(fetcher, sleep)
        requests_used += used
        now = clock()
        locked_terminal = lock_first_complete_payload(
            root,
            source_family=SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
            applicable_period=TERMINAL_PERIOD,
            requested_url=LISTED_ISSUES_PAGE_URL,
            resolved_url=xls_final_url,
            http_status=200,
            payload=xls_bytes,
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

    inventory = build_source_inventory(locked_index={})
    locked_evidence = {"terminal_snapshot": locked_terminal}
    reconstruction = reconstruct_security_state(locked_evidence)
    reconstruction_deterministic = reconstruction_is_deterministic(locked_evidence)
    raw_provenance_pass = verify_raw_provenance(root)

    evidence = compute_stage_a_evidence(
        inventory=inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=trading_calendar_derived,
        reconstruction_deterministic=reconstruction_deterministic,
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
        "deterministic_reconstruction_pass": reconstruction_deterministic,
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


def fetch_terminal_snapshot(
    fetcher: Callable[[str], tuple[bytes, str]],
    sleep: Callable[[int], None],
) -> tuple[bytes, str, int]:
    """Two independently retried requests: the listing page, then its
    extracted (and off-domain-validated) `data_j.xls` link."""
    page_bytes, _page_final_url, used_page = fetch_once_with_retry(LISTED_ISSUES_PAGE_URL, fetcher, sleep)
    xls_url = extract_data_j_xls_url(page_bytes)
    xls_bytes, xls_final_url, used_xls = fetch_once_with_retry(xls_url, fetcher, sleep)
    return xls_bytes, xls_final_url, used_page + used_xls


__all__ = [
    "ACQUISITION_IMPLEMENTATION_COMPLETE",
    "ALLOWED_HOST_SUFFIX", "BOUND_SIGNAL_GRID_BLOB_SHA", "BOUND_SIGNAL_GRID_PATH",
    "CALENDAR_ENVELOPE_FIRST_YEAR_MONTH", "CALENDAR_ENVELOPE_LAST_YEAR_MONTH",
    "CALENDAR_MONTHLY_LOCATOR_TEMPLATE", "CALENDAR_PAGE_URL", "CHATGPT_DECISION_REQUIRED", "CONFIRMATION",
    "DELISTED_COMPANY_ROOT_URL", "F2_SEMANTIC_ROW_LABEL", "F4_SEMANTIC_ROW_LABEL", "F6_SEMANTIC_SECTION_LABEL",
    "GOVERNANCE_FAILURE", "IMPLEMENTATION_FAILURE", "INVENTORY_AVAILABLE", "INVENTORY_MISSING",
    "INVENTORY_NOT_APPLICABLE", "LISTED_ISSUES_PAGE_URL", "LISTING_CO_ROOT_URL", "LOCATOR_STRATEGIES",
    "LocatorStrategy", "MONTHLY_COVERAGE_FAMILIES", "MONTHLY_STATISTICS_ROOT_URL", "PLUMBING_FAILURE_RETRIABLE",
    "PROBE_SIGNAL_GRID_CONTRACT_MISMATCH", "SLOT_KIND_GLOBAL", "SLOT_KIND_MONTHLY", "SLOT_KIND_TERMINAL",
    "SLOT_KIND_YEAR", "SOURCE_FAMILIES", "SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE",
    "SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE", "SOURCE_FAMILY_JPX_CALENDAR",
    "SOURCE_FAMILY_LISTED_ISSUES_MONTH_END", "SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS",
    "SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT", "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE",
    "SOURCE_OR_DATA_FEASIBILITY_FAILURE", "STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE",
    "STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE", "STAGE", "STUDY",
    "TOPIX_ROOT_URL", "VALID_SLOT_KINDS",
    "V9005StageABlocked", "build_safe_summary", "build_source_inventory", "build_trading_day_set",
    "calendar_envelope_extra_months", "calendar_envelope_months", "canonical_bytes",
    "compute_month_end_mismatch_count", "compute_stage_a_evidence",
    "derive_final_signal_d0", "derive_stage_b_global_end_exclusive", "ensure_locked_payload",
    "extract_data_j_xls_url", "f2_bridge_months", "fetch_once_with_retry", "fetch_terminal_snapshot",
    "initialize_output_root", "inventory_months", "lock_first_complete_payload", "nth_trading_day_after",
    "read_locked_payload", "reconstruct_security_state", "reconstruction_is_deterministic",
    "resolve_f7_calendar_url", "resolve_month_locator", "run_stage_a",
    "sha256_bytes", "validate_jpx_url", "verify_acquisition_implementation_ready",
    "verify_locator_contract_complete", "verify_raw_provenance",
    "verify_signal_grid_binding",
]
