from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.v7_jpx_calendar import (
    CALENDAR_DEFINITION_VERSION,
    CALENDAR_SOURCE,
    CALENDAR_TIMEZONE,
    CONFIRMATION,
    FUTURE_EXTENSION_POLICY,
    SOURCE_HOST,
    SOURCE_URL,
    V7JpxCalendarBlocked,
    acquire_jpx_calendar,
    build_calendar_snapshot,
    canonical_json_bytes,
    fetch_jpx_source_once,
    generate_engine_days,
    is_jpx_trading_day,
    load_calendar_snapshot,
    next_jpx_trading_day,
    nth_jpx_trading_day_after,
    parse_jpx_holiday_html,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 7, 3, 0, tzinfo=timezone.utc)


def _fixture() -> bytes:
    holidays_2026 = [
        ("Jan. 1", "New Year's Day"), ("Jan. 2", "Market Holiday"), ("Jan. 3", "Market Holiday"),
        ("Jan. 12", "Coming of Age Day"), ("Feb. 11", "National Foundation Day"), ("Feb. 23", "Emperor's Birthday"),
        ("Mar. 20", "Vernal Equinox Day"), ("Apr. 29", "Showa Day"), ("May 3", "Constitution Memorial Day"),
        ("May 4", "Greenery Day"), ("May 5", "Children's Day"), ("May 6", "Market Holiday"),
        ("Jul. 20", "Marine Day"), ("Aug. 11", "Mountain Day"), ("Sep. 21", "Respect for the Aged Day"),
        ("Sep. 22", "Market Holiday"), ("Sep. 23", "Autumnal Equinox Day"), ("Oct. 12", "Sports Day"),
        ("Nov. 3", "Culture Day"), ("Nov. 23", "Labor Thanksgiving Day"), ("Dec. 31", "Market Holiday"),
    ]
    holidays_2027 = [
        ("Jan. 1", "New Year's Day"), ("Jan. 2", "Market Holiday"), ("Jan. 3", "Market Holiday"),
        ("Jan. 11", "Coming of Age Day"), ("Feb. 11", "National Foundation Day"), ("Feb. 23", "Emperor's Birthday"),
        ("Mar. 21", "Vernal Equinox Day"), ("Mar. 22", "Market Holiday"), ("Apr. 29", "Showa Day"),
        ("May 3", "Constitution Memorial Day"), ("May 4", "Greenery Day"), ("May 5", "Children's Day"),
        ("Jul. 19", "Marine Day"), ("Aug. 11", "Mountain Day"), ("Sep. 20", "Respect for the Aged Day"),
        ("Sep. 23", "Autumnal Equinox Day"), ("Oct. 11", "Sports Day"), ("Nov. 3", "Culture Day"),
        ("Nov. 23", "Labor Thanksgiving Day"), ("Dec. 31", "Market Holiday"),
    ]
    rows = []
    for year, holidays in ((2026, holidays_2026), (2027, holidays_2027)):
        rows.append(f"<tr><th>{year}</th></tr>")
        for index, (day, label) in enumerate(holidays):
            date_cell = f"<span>{day}</span><sup>*</sup> (Tue.)" if index == 13 else f" {day} (Thu.) "
            rows.append(f"<tr>\n<td> {date_cell} </td>\n<td> {label} </td>\n</tr>")
    html = """
    <html><body>
    <nav><a href='/calendar/2026'>2026</a><a href='/calendar/2027'>2027</a></nav>
    <h2>Market Holidays</h2>
    <table class='calendar-table'>
    """ + "\n".join(rows) + """
    </table>
    <table id='derivatives-calendar'><tr><th>Derivatives holiday trading</th></tr><tr><td>Aug. 11</td><td>Night Session</td></tr></table>
    <table><tr><td>2026 News Calendar</td></tr><tr><td>Jan. 1</td><td>News</td></tr></table>
    </body></html>
    """
    return html.encode("utf-8")


FIXTURE = _fixture()


def snapshot(payload=FIXTURE):
    holidays = parse_jpx_holiday_html(payload)
    return build_calendar_snapshot(payload, holidays, "2026-08-07T03:00:00Z")


class Response:
    def __init__(self, payload=FIXTURE, status=200, url=SOURCE_URL):
        self.payload, self.status, self.url = payload, status, url
        self.closed = False
    def read(self):
        return self.payload
    def close(self):
        self.closed = True


class Opener:
    def __init__(self, response=None):
        self.response = response or Response()
        self.calls = []
    def __call__(self, request, timeout=30):
        self.calls.append((request, timeout))
        return self.response


def test_official_source_identity_is_fixed():
    assert SOURCE_HOST == "www.jpx.co.jp"
    assert SOURCE_URL.startswith("https://www.jpx.co.jp/")
    assert CALENDAR_SOURCE == "JPX_OFFICIAL_MARKET_HOLIDAYS"


def test_confirmation_mismatch_makes_zero_requests(tmp_path):
    opener = Opener()
    with pytest.raises(V7JpxCalendarBlocked, match="CONFIRMATION_MISMATCH"):
        acquire_jpx_calendar(raw_output=tmp_path / "raw.html", calendar_output=tmp_path / "calendar.json", confirmation="WRONG", opener=opener, clock=lambda: NOW)
    assert opener.calls == []


def test_fetch_uses_https_fixed_host_headers_and_timeout():
    opener = Opener()
    result = fetch_jpx_source_once(opener)
    request, timeout = opener.calls[0]
    assert timeout == 30
    assert request.full_url == SOURCE_URL
    assert request.get_header("User-agent") == "V7-JPX-Calendar-Fixation/1.0"
    assert result["response_host"] == SOURCE_HOST


def test_non_200_blocks_without_retry():
    opener = Opener(Response(status=503))
    with pytest.raises(V7JpxCalendarBlocked, match="HTTP_STATUS_503"):
        fetch_jpx_source_once(opener)
    assert len(opener.calls) == 1


def test_redirect_other_host_blocks():
    opener = Opener(Response(url="https://evil.example/index.html"))
    with pytest.raises(V7JpxCalendarBlocked, match="RESPONSE_HOST_MISMATCH"):
        fetch_jpx_source_once(opener)


def test_2026_and_2027_are_parsed():
    holidays = parse_jpx_holiday_html(FIXTURE)
    assert {row["year"] for row in holidays} == {2026, 2027}
    assert {row["date"] for row in holidays} >= {"2026-08-11", "2027-08-11"}


def test_actual_style_counts_are_fixed():
    holidays = parse_jpx_holiday_html(FIXTURE, expected_row_counts={2026: 21, 2027: 20})
    assert sum(row["year"] == 2026 for row in holidays) == 21
    assert sum(row["year"] == 2027 for row in holidays) == 20


def test_nested_span_sup_footnote_and_whitespace_are_normalized():
    holidays = parse_jpx_holiday_html(FIXTURE)
    mountain = next(row for row in holidays if row["date"] == "2026-08-11")
    assert mountain["label"] == "Mountain Day"


def test_navigation_and_unrelated_tables_are_ignored():
    holidays = parse_jpx_holiday_html(FIXTURE)
    assert all(row["label"] != "News" for row in holidays)
    assert len(holidays) == 41


def test_separate_year_tables_bind_dates_to_their_year_heading():
    payload = b"""
    <h2>Market Holidays</h2>
    <h3>2026</h3><table><tr><td>Jan. 1 (Thu.)</td><td>New Year's Day</td></tr></table>
    <h3>2027</h3><table><tr><td>Jan. 1 (Fri.)</td><td>New Year's Day</td></tr></table>
    """
    holidays = parse_jpx_holiday_html(payload)
    assert holidays == [
        {"date": "2026-01-01", "label": "New Year's Day", "year": 2026},
        {"date": "2027-01-01", "label": "New Year's Day", "year": 2027},
    ]


def test_date_row_without_market_year_context_blocks():
    bad = b"<h2>Market Holidays</h2><table><tr><td>Jan. 1 (Thu.)</td><td>Holiday</td></tr></table>"
    with pytest.raises(V7JpxCalendarBlocked, match="MISSING_YEAR_CONTEXT"):
        parse_jpx_holiday_html(bad)


def test_known_holiday_guards_are_present_and_august_10_is_not_holiday():
    dates = {row["date"] for row in parse_jpx_holiday_html(FIXTURE)}
    for value in ("2026-01-01", "2026-01-02", "2026-08-11", "2026-12-31", "2027-01-01", "2027-08-11", "2027-12-31"):
        assert value in dates
    assert "2026-08-10" not in dates and "2026-08-12" not in dates


def test_duplicate_holiday_blocks():
    holidays = parse_jpx_holiday_html(FIXTURE)
    with pytest.raises(V7JpxCalendarBlocked, match="DUPLICATE_HOLIDAY"):
        build_calendar_snapshot(FIXTURE, holidays + [dict(holidays[0])], "2026-08-07T03:00:00Z")


def test_invalid_date_blocks():
    bad = b"<table><tr><th>2026 Market Holidays</th></tr><tr><td>2026-02-30</td><td>Bad</td></tr><tr><td>2027-01-01</td><td>Good</td></tr></table>"
    with pytest.raises(V7JpxCalendarBlocked, match="INVALID_DATE"):
        parse_jpx_holiday_html(bad)


def test_canonical_holiday_sort_and_json_bytes():
    value = snapshot()
    dates = [row["date"] for row in value["market_holidays"]]
    assert dates == sorted(dates)
    assert canonical_json_bytes(value).endswith(b"\n")
    assert json.loads(canonical_json_bytes(value)) == value


def test_source_sha_and_byte_count_are_bound():
    value = snapshot()
    assert value["source_payload_sha256"] == hashlib.sha256(FIXTURE).hexdigest()
    assert value["source_byte_count"] == len(FIXTURE)


def test_source_dict_and_holiday_input_order_are_deterministic():
    holidays = parse_jpx_holiday_html(FIXTURE)
    first = build_calendar_snapshot(FIXTURE, holidays, "2026-08-07T03:00:00Z")
    second = build_calendar_snapshot(FIXTURE, list(reversed(holidays)), "2026-08-07T03:00:00Z")
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_timezone_and_definition_metadata():
    value = snapshot()
    assert value["calendar_timezone"] == "Asia/Tokyo"
    assert value["calendar_definition_version"] == CALENDAR_DEFINITION_VERSION
    assert value["study_calendar_generated"] is False


def test_weekday_ordinary_day_is_trading_day():
    cal = load_calendar_snapshot(snapshot())
    assert is_jpx_trading_day(cal, "2026-08-10") is True
    assert is_jpx_trading_day(cal, "2026-08-08") is False
    assert is_jpx_trading_day(cal, "2026-08-09") is False
    assert is_jpx_trading_day(cal, "2026-08-11") is False


def test_next_jpx_days_are_fixed():
    cal = load_calendar_snapshot(snapshot())
    assert next_jpx_trading_day(cal, "2026-08-07") == "2026-08-10"
    assert next_jpx_trading_day(cal, "2026-08-10") == "2026-08-12"


def test_d1_and_d10_are_deterministic():
    cal = load_calendar_snapshot(snapshot())
    d1 = next_jpx_trading_day(cal, "2026-08-10")
    d10 = nth_jpx_trading_day_after(cal, "2026-08-10", 10)
    assert d1 == "2026-08-12"
    assert d10 == nth_jpx_trading_day_after(cal, "2026-08-10", 10)
    assert d1 != d10


def test_known_d10_value_after_2026_08_10():
    cal = load_calendar_snapshot(snapshot())
    assert nth_jpx_trading_day_after(cal, "2026-08-10", 10) == "2026-08-25"


def test_generate_engine_days_is_distinct_from_source_study_flag():
    cal = load_calendar_snapshot(snapshot())
    days = generate_engine_days(cal, "2026-08-10", "2026-08-14")
    assert days == ["2026-08-10", "2026-08-12", "2026-08-13"]
    assert snapshot()["study_calendar_generated"] is False


def test_date_outside_coverage_blocks_without_prediction():
    cal = load_calendar_snapshot(snapshot())
    with pytest.raises(V7JpxCalendarBlocked, match="CALENDAR_DATE_OUTSIDE_COVERAGE"):
        is_jpx_trading_day(cal, "2028-01-04")


def test_future_extension_policy_is_append_only_human_gated():
    assert snapshot()["future_extension_policy"] == FUTURE_EXTENSION_POLICY


def test_missing_ticker_data_is_irrelevant_to_calendar():
    cal = load_calendar_snapshot(snapshot())
    assert is_jpx_trading_day(cal, "2026-08-10") is True


def test_derivatives_holiday_event_is_ignored():
    dates = [row["date"] for row in parse_jpx_holiday_html(FIXTURE)]
    assert dates.count("2026-08-11") == 1


def test_activation_status_remains_not_set():
    value = snapshot()
    assert value["activation_boundary_status"] == "NOT_SET"


def test_calendar_output_is_readable_from_json(tmp_path):
    path = tmp_path / "calendar.json"
    path.write_bytes(canonical_json_bytes(snapshot()))
    cal = load_calendar_snapshot(path)
    assert next_jpx_trading_day(cal, "2026-08-07") == "2026-08-10"


def test_real_global_urlopen_is_not_used_by_parser(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen used")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert parse_jpx_holiday_html(FIXTURE)


def test_one_fake_acquisition_publishes_two_canonical_files(tmp_path):
    opener = Opener()
    result = acquire_jpx_calendar(raw_output=tmp_path / "source.html", calendar_output=tmp_path / "calendar.json", confirmation=CONFIRMATION, opener=opener, clock=lambda: NOW)
    assert len(opener.calls) == 1
    assert (tmp_path / "source.html").read_bytes() == FIXTURE
    assert json.loads((tmp_path / "calendar.json").read_text(encoding="utf-8")) == result


def test_raw_is_persisted_before_parser_failure_and_can_be_reparsed_offline(tmp_path):
    bad_payload = b"<html><body><h2>Market Holidays</h2><table><tr><td>bad</td></tr></table></body></html>"
    opener = Opener(Response(payload=bad_payload))
    raw = tmp_path / "source.html"
    calendar = tmp_path / "calendar.json"
    with pytest.raises(V7JpxCalendarBlocked, match="MARKET_HOLIDAYS_NOT_FOUND"):
        acquire_jpx_calendar(raw_output=raw, calendar_output=calendar, confirmation=CONFIRMATION, opener=opener, clock=lambda: NOW)
    assert raw.read_bytes() == bad_payload
    assert not calendar.exists()
    assert len(opener.calls) == 1
    assert parse_jpx_holiday_html(FIXTURE)


def test_no_yahoo_derived_calendar_source():
    value = snapshot()
    assert "yahoo" not in value["calendar_source"].lower()
    assert value["calendar_source_host"] == SOURCE_HOST


def test_source_payload_order_does_not_change_holiday_hash():
    holidays = parse_jpx_holiday_html(FIXTURE)
    a = build_calendar_snapshot(FIXTURE, holidays, "2026-08-07T03:00:00Z")
    b = build_calendar_snapshot(FIXTURE, list(reversed(holidays)), "2026-08-07T03:00:00Z")
    assert hashlib.sha256(canonical_json_bytes(a)).hexdigest() == hashlib.sha256(canonical_json_bytes(b)).hexdigest()


def test_calendar_source_snapshot_does_not_mean_activation():
    value = snapshot()
    assert value["study_calendar_generated"] is False
    assert value["activation_boundary_status"] == "NOT_SET"
