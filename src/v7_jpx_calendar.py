"""JPX official market-holiday snapshot and deterministic calendar helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SOURCE_URL = "https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html"
SOURCE_HOST = "www.jpx.co.jp"
CALENDAR_SOURCE = "JPX_OFFICIAL_MARKET_HOLIDAYS"
CALENDAR_TIMEZONE = "Asia/Tokyo"
CALENDAR_DEFINITION_VERSION = "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1"
FUTURE_EXTENSION_POLICY = "APPEND_ONLY_SAME_OFFICIAL_JPX_SOURCE_SAME_PARSER_HUMAN_GATE"
SUPPORTED_YEARS = (2026, 2027)
EXPECTED_HOLIDAY_ROW_COUNTS = {2026: 21, 2027: 20}
CONFIRMATION = "V7_FIX_JPX_CALENDAR_SOURCE"
USER_AGENT = "V7-JPX-Calendar-Fixation/1.0"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
    "Connection": "close",
}
DATE_RE = re.compile(
    r"(?P<year>20\d{2})\s*[-/]\s*(?P<month>\d{1,2})\s*[-/]\s*(?P<day>\d{1,2})"
    r"|(?P<month2>\d{1,2})\s*[-/]\s*(?P<day2>\d{1,2})\s*[-/]\s*(?P<year2>20\d{2})"
    r"|(?P<year3>20\d{2})年\s*(?P<month3>\d{1,2})月\s*(?P<day3>\d{1,2})日?"
    r"|(?P<month4>\d{1,2})月\s*(?P<day4>\d{1,2})日"
    r"|(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)\s*[,\-]?\s*(?P<day_name>\d{1,2})(?:\s*[,\-]?\s*(?P<year_name>20\d{2}))?"
    r"|(?P<day_first>\d{1,2})\s+(?P<month_first>January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)\s*(?P<year_first>20\d{2})?",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(2026|2027)\b")
WEEKDAY_RE = re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day)?\b|[月火水木金土日](?:曜日)?", re.I)
DERIVATIVE_RE = re.compile(r"derivative|futures?|options?|commodity", re.I)
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


class V7JpxCalendarBlocked(ValueError):
    """Fail closed for source, parser, coverage, or calendar integrity errors."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_date(value: object) -> date:
    if not isinstance(value, str):
        raise V7JpxCalendarBlocked("INVALID_DATE")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V7JpxCalendarBlocked("INVALID_DATE") from error
    if parsed.isoformat() != value:
        raise V7JpxCalendarBlocked("INVALID_DATE")
    return parsed


def _date_from_match(match: re.Match[str], year_hint: int | None) -> tuple[date, str] | None:
    groups = match.groupdict()
    if groups["year"]:
        year, month, day = int(groups["year"]), int(groups["month"]), int(groups["day"])
    elif groups["year2"]:
        year, month, day = int(groups["year2"]), int(groups["month2"]), int(groups["day2"])
    elif groups["year3"]:
        year, month, day = int(groups["year3"]), int(groups["month3"]), int(groups["day3"])
    elif groups["month4"]:
        if year_hint is None:
            return None
        year, month, day = year_hint, int(groups["month4"]), int(groups["day4"])
    elif groups["month_name"]:
        year = int(groups["year_name"]) if groups["year_name"] else year_hint
        if year is None:
            return None
        month = MONTHS[groups["month_name"].rstrip(".").lower()]
        day = int(groups["day_name"])
    else:
        year = int(groups["year_first"]) if groups["year_first"] else year_hint
        if year is None:
            return None
        month = MONTHS[groups["month_first"].rstrip(".").lower()]
        day = int(groups["day_first"])
    try:
        return date(year, month, day), match.group(0)
    except ValueError as error:
        raise V7JpxCalendarBlocked("INVALID_DATE") from error


class _HolidayHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, Any]] = []
        self._table_depth = 0
        self._table_meta = ""
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._table_text: list[str] = []
        self._heading: list[str] | None = None
        self._heading_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"h1", "h2", "h3", "h4", "h5", "h6", "caption"}:
            self._heading = []
            self._heading_tag = lowered
        elif lowered == "table":
            self._table_depth += 1
            self._table_text = []
            self._table_meta = " ".join(value or "" for key, value in attrs if key.lower() in {"id", "class", "aria-label", "summary"})
            self.events.append(("table", self._table_meta))
        elif lowered == "tr" and self._table_depth:
            self._row = []
        elif lowered in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"h1", "h2", "h3", "h4", "h5", "h6", "caption"} and self._heading is not None:
            self.events.append(("heading", " ".join("".join(self._heading).split())))
            self._heading = None
            self._heading_tag = None
            return
        if lowered in {"td", "th"} and self._row is not None and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            self._row.append(text)
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            self.events.append(("row", (self._row, self._table_meta, " ".join(self._table_text))))
            self._row = None
        elif lowered == "table" and self._table_depth:
            self._table_depth -= 1
            self.events.append(("table_end", self._table_meta))
            self._table_meta = ""

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._heading is not None:
            self._heading.append(text)
        if self._table_depth:
            self._table_text.append(text)
        if self._cell is not None:
            self._cell.append(text)


def parse_jpx_holiday_html(
    payload: bytes,
    covered_years: Sequence[int] = SUPPORTED_YEARS,
    *,
    expected_row_counts: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, bytes):
        raise V7JpxCalendarBlocked("SOURCE_PAYLOAD_NOT_BYTES")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V7JpxCalendarBlocked("SOURCE_ENCODING_INVALID") from error
    parser = _HolidayHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as error:
        raise V7JpxCalendarBlocked("SOURCE_HTML_INVALID") from error
    supported = set(int(year) for year in covered_years)
    current_year: int | None = None
    market_section_active = False
    awaiting_market_table = False
    market_table_active = False
    holidays: list[dict[str, Any]] = []
    for kind, value in parser.events:
        if kind == "heading":
            heading = str(value)
            lower = heading.lower()
            if "market holiday" in lower:
                market_section_active = True
                awaiting_market_table = True
                years = [int(year) for year in YEAR_RE.findall(heading)]
                if years:
                    current_year = years[0]
            elif market_section_active:
                years = [int(year) for year in YEAR_RE.findall(heading)]
                if years and not re.sub(r"\b20\d{2}\b", "", heading).strip():
                    current_year = years[0]
                    awaiting_market_table = True
                elif heading.strip():
                    market_section_active = False
                    awaiting_market_table = False
            continue
        if kind == "table":
            table_meta = str(value)
            market_table_active = awaiting_market_table and DERIVATIVE_RE.search(table_meta) is None
            continue
        if kind == "table_end":
            market_table_active = False
            awaiting_market_table = False
            continue
        if kind != "row":
            continue
        cells, table_meta, table_context = value
        joined = " ".join(cells)
        context = (table_meta + " " + table_context + " " + joined).strip()
        derivative_context = DERIVATIVE_RE.search(context) is not None
        explicit_years = [int(year) for year in YEAR_RE.findall(joined)]
        year_header = re.fullmatch(r"\s*(2026|2027)(?:\s+market\s+holidays?)?\s*", joined, re.I)
        if year_header:
            market_table_active = True
            market_section_active = True
            awaiting_market_table = False
            current_year = int(year_header.group(1))
            continue
        if explicit_years and not DATE_RE.search(joined):
            if "market holiday" in joined.lower() or market_table_active:
                market_table_active = True
                market_section_active = True
                current_year = explicit_years[0]
            continue
        if derivative_context:
            continue
        match = DATE_RE.search(joined)
        if match is None:
            continue
        row_is_market = market_table_active or "market holiday" in context.lower()
        parsed = _date_from_match(match, current_year if row_is_market else None)
        if parsed is None:
            if row_is_market:
                raise V7JpxCalendarBlocked("MISSING_YEAR_CONTEXT")
            continue
        observed, date_text = parsed
        if observed.year not in supported:
            continue
        label_parts = [cell for cell in cells if date_text not in cell]
        label = " ".join(label_parts).strip()
        if not label:
            label = joined.replace(date_text, " ")
        label = WEEKDAY_RE.sub(" ", label)
        label = re.sub(r"\b20\d{2}\b", " ", label)
        label = re.sub(r"[()（）,:;|]+", " ", label)
        label = " ".join(label.split()).strip(" -")
        if not label:
            label = "JPX Market Holiday"
        holidays.append({"date": observed.isoformat(), "label": label, "year": observed.year})
    if not holidays:
        raise V7JpxCalendarBlocked("MARKET_HOLIDAYS_NOT_FOUND")
    canonical = _canonical_holidays(holidays, supported)
    if expected_row_counts is not None:
        actual = {year: sum(1 for row in canonical if row["year"] == year) for year in supported}
        for year, expected in expected_row_counts.items():
            if actual.get(int(year), 0) != int(expected):
                raise V7JpxCalendarBlocked("HOLIDAY_ROW_COUNT_MISMATCH")
    return canonical


def _canonical_holidays(holidays: Sequence[Mapping[str, Any]], covered_years: set[int]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    canonical: list[dict[str, Any]] = []
    for source in holidays:
        if set(source) != {"date", "label", "year"}:
            raise V7JpxCalendarBlocked("HOLIDAY_SCHEMA_INVALID")
        observed = _parse_date(source["date"])
        if observed.year not in covered_years or int(source["year"]) != observed.year:
            raise V7JpxCalendarBlocked("HOLIDAY_YEAR_INVALID")
        if observed.isoformat() in seen:
            raise V7JpxCalendarBlocked("DUPLICATE_HOLIDAY")
        label = str(source["label"]).strip()
        if not label:
            raise V7JpxCalendarBlocked("HOLIDAY_LABEL_INVALID")
        seen.add(observed.isoformat())
        canonical.append({"date": observed.isoformat(), "label": label, "year": observed.year})
    return sorted(canonical, key=lambda row: (row["date"], row["label"], row["year"]))


def build_calendar_snapshot(
    payload: bytes,
    holidays: Sequence[Mapping[str, Any]],
    source_acquisition_utc: str,
    *,
    source_url: str = SOURCE_URL,
    source_host: str = SOURCE_HOST,
) -> dict[str, Any]:
    if source_host != SOURCE_HOST or urllib.parse.urlparse(source_url).scheme != "https":
        raise V7JpxCalendarBlocked("SOURCE_IDENTITY_INVALID")
    canonical_holidays = _canonical_holidays(holidays, set(SUPPORTED_YEARS))
    years_present = {row["year"] for row in canonical_holidays}
    if years_present != set(SUPPORTED_YEARS):
        raise V7JpxCalendarBlocked("MISSING_COVERED_YEAR")
    return {
        "schema_version": "V7_JPX_CALENDAR_SNAPSHOT_V1",
        "calendar_source": CALENDAR_SOURCE,
        "calendar_source_url": source_url,
        "calendar_source_host": SOURCE_HOST,
        "calendar_timezone": CALENDAR_TIMEZONE,
        "calendar_definition_version": CALENDAR_DEFINITION_VERSION,
        "source_acquisition_utc": source_acquisition_utc,
        "source_payload_sha256": sha256_bytes(payload),
        "source_byte_count": len(payload),
        "covered_years": list(SUPPORTED_YEARS),
        "market_holidays": canonical_holidays,
        "coverage_start": "2026-01-01",
        "coverage_end": "2027-12-31",
        "calendar_generation_method": "WEEKDAYS_MINUS_CANONICAL_JPX_MARKET_HOLIDAYS",
        "study_calendar_generated": False,
        "future_extension_policy": FUTURE_EXTENSION_POLICY,
        "activation_boundary_status": "NOT_SET",
    }


@dataclass(frozen=True)
class CalendarSnapshot:
    holidays: frozenset[date]
    coverage_start: date
    coverage_end: date


def load_calendar_snapshot(snapshot: Mapping[str, Any] | str | os.PathLike[str]) -> CalendarSnapshot:
    if isinstance(snapshot, (str, os.PathLike)):
        try:
            value = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V7JpxCalendarBlocked("CALENDAR_SNAPSHOT_INVALID") from error
    else:
        value = dict(snapshot)
    if value.get("calendar_source") != CALENDAR_SOURCE or value.get("calendar_timezone") != CALENDAR_TIMEZONE:
        raise V7JpxCalendarBlocked("CALENDAR_SNAPSHOT_IDENTITY_INVALID")
    if tuple(value.get("covered_years", ())) != SUPPORTED_YEARS:
        raise V7JpxCalendarBlocked("CALENDAR_COVERAGE_INVALID")
    if value.get("study_calendar_generated") is not False:
        raise V7JpxCalendarBlocked("SOURCE_STUDY_CALENDAR_FLAG_INVALID")
    holidays = _canonical_holidays(value.get("market_holidays", []), set(SUPPORTED_YEARS))
    if {row["year"] for row in holidays} != set(SUPPORTED_YEARS):
        raise V7JpxCalendarBlocked("MISSING_COVERED_YEAR")
    coverage_start = _parse_date(value.get("coverage_start"))
    coverage_end = _parse_date(value.get("coverage_end"))
    if coverage_start != date(2026, 1, 1) or coverage_end != date(2027, 12, 31):
        raise V7JpxCalendarBlocked("CALENDAR_COVERAGE_INVALID")
    return CalendarSnapshot(frozenset(_parse_date(row["date"]) for row in holidays), coverage_start, coverage_end)


def _ensure_covered(snapshot: CalendarSnapshot, value: date) -> None:
    if value < snapshot.coverage_start or value > snapshot.coverage_end:
        raise V7JpxCalendarBlocked("CALENDAR_DATE_OUTSIDE_COVERAGE")


def is_jpx_trading_day(snapshot: CalendarSnapshot, value: str | date) -> bool:
    observed = _parse_date(value) if isinstance(value, str) else value
    _ensure_covered(snapshot, observed)
    return observed.weekday() < 5 and observed not in snapshot.holidays


def next_jpx_trading_day(snapshot: CalendarSnapshot, value: str | date) -> str:
    observed = _parse_date(value) if isinstance(value, str) else value
    _ensure_covered(snapshot, observed)
    candidate = observed + timedelta(days=1)
    while candidate <= snapshot.coverage_end:
        if is_jpx_trading_day(snapshot, candidate):
            return candidate.isoformat()
        candidate += timedelta(days=1)
    raise V7JpxCalendarBlocked("CALENDAR_DATE_OUTSIDE_COVERAGE")


def nth_jpx_trading_day_after(snapshot: CalendarSnapshot, value: str | date, n: int) -> str:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise V7JpxCalendarBlocked("INVALID_TRADING_DAY_OFFSET")
    current = _parse_date(value) if isinstance(value, str) else value
    _ensure_covered(snapshot, current)
    for _ in range(n):
        current = _parse_date(next_jpx_trading_day(snapshot, current))
    return current.isoformat()


def generate_engine_days(snapshot: CalendarSnapshot, start: str, end_exclusive: str) -> list[str]:
    first = _parse_date(start)
    end = _parse_date(end_exclusive)
    if not first < end:
        raise V7JpxCalendarBlocked("INVALID_CALENDAR_RANGE")
    _ensure_covered(snapshot, first)
    _ensure_covered(snapshot, end - timedelta(days=1))
    days: list[str] = []
    current = first
    while current < end:
        if is_jpx_trading_day(snapshot, current):
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _validate_response_host(url: object) -> None:
    if not isinstance(url, str):
        raise V7JpxCalendarBlocked("RESPONSE_HOST_MISMATCH")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != SOURCE_HOST:
        raise V7JpxCalendarBlocked("RESPONSE_HOST_MISMATCH")


def fetch_jpx_source_once(opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    request = urllib.request.Request(SOURCE_URL, headers=dict(HEADERS), method="GET")
    response = opener(request, timeout=30)
    try:
        if getattr(response, "status", None) != 200:
            raise V7JpxCalendarBlocked("HTTP_STATUS_" + str(getattr(response, "status", None)))
        _validate_response_host(getattr(response, "url", None))
        payload = response.read()
        if not isinstance(payload, bytes):
            raise V7JpxCalendarBlocked("SOURCE_PAYLOAD_NOT_BYTES")
        return {
            "payload": payload,
            "source_payload_sha256": sha256_bytes(payload),
            "source_byte_count": len(payload),
            "response_host": SOURCE_HOST,
        }
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def acquire_jpx_calendar(
    *,
    raw_output: str | os.PathLike[str],
    calendar_output: str | os.PathLike[str],
    confirmation: str,
    opener: Callable[..., Any],
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise V7JpxCalendarBlocked("CONFIRMATION_MISMATCH")
    raw_path, calendar_path = Path(raw_output), Path(calendar_output)
    if raw_path.exists() or calendar_path.exists():
        raise V7JpxCalendarBlocked("OUTPUT_EXISTS")
    if raw_path.parent != calendar_path.parent or not raw_path.parent.exists():
        raise V7JpxCalendarBlocked("OUTPUT_PARENT_INVALID")
    acquired = clock()
    if not isinstance(acquired, datetime) or acquired.tzinfo is None or acquired.utcoffset() != timedelta(0):
        raise V7JpxCalendarBlocked("ACQUISITION_UTC_INVALID")
    fetched = fetch_jpx_source_once(opener)
    raw_path.write_bytes(fetched["payload"])
    holidays = parse_jpx_holiday_html(fetched["payload"], expected_row_counts=EXPECTED_HOLIDAY_ROW_COUNTS)
    snapshot = build_calendar_snapshot(fetched["payload"], holidays, acquired.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v7-jpx-calendar-", dir=str(raw_path.parent)))
    try:
        (staging / raw_path.name).write_bytes(fetched["payload"])
        (staging / calendar_path.name).write_bytes(canonical_json_bytes(snapshot))
        os.replace(staging / raw_path.name, raw_path)
        os.replace(staging / calendar_path.name, calendar_path)
        return snapshot
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass


__all__ = [
    "CALENDAR_DEFINITION_VERSION", "CALENDAR_SOURCE", "CALENDAR_TIMEZONE", "CONFIRMATION",
    "EXPECTED_HOLIDAY_ROW_COUNTS", "FUTURE_EXTENSION_POLICY", "HEADERS", "SOURCE_HOST", "SOURCE_URL", "SUPPORTED_YEARS",
    "CalendarSnapshot", "V7JpxCalendarBlocked", "acquire_jpx_calendar", "build_calendar_snapshot",
    "canonical_json_bytes", "fetch_jpx_source_once", "generate_engine_days", "is_jpx_trading_day",
    "load_calendar_snapshot", "next_jpx_trading_day", "nth_jpx_trading_day_after",
    "parse_jpx_holiday_html", "sha256_bytes",
]
