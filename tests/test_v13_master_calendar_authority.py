"""Synthetic checks for the frozen V13 calendar authority; no provider import."""

from __future__ import annotations

import ast
import hashlib
import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import pytest
import pandas as pd

from src import v13_public_data_lock as calendar


def test_exact_frozen_provenance_and_window():
    expected = {
        "source_identity": "PANDAS_MARKET_CALENDARS_JPX_5_4_0_RELEASE_ARTIFACT",
        "calendar_name": "JPX",
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "4.13.2",
        "release_tag": "v5.4.0",
        "release_tag_commit": "275890784073a3a3a347e4f05f4dc986456e6a75",
        "jpx_source_file": "pandas_market_calendars/calendars/jpx.py",
        "jpx_source_blob": "a7a59b6cf910e325c85fc042459ff57ca8f70613",
        "holiday_source_file": "pandas_market_calendars/holidays/jp.py",
        "holiday_source_blob": "4c34214d06862e02ac22e946757463f748074fde",
        "official_pypi_wheel": "pandas_market_calendars-5.4.0-py3-none-any.whl",
        "official_pypi_wheel_sha256": "bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3",
        "coverage_start": "2015-01-01",
        "coverage_end": "2025-12-31",
    }
    assert dict(calendar.CALENDAR_PROVENANCE) == expected
    assert (calendar.START, calendar.END) == (date(2015, 1, 1), date(2025, 12, 31))
    calendar.validate_calendar_provenance(expected)
    for key in expected:
        bad = dict(expected)
        bad[key] = "substitute"
        with pytest.raises(ValueError, match="CALENDAR_PROVENANCE_MISMATCH"):
            calendar.validate_calendar_provenance(bad)
    with pytest.raises(ValueError, match="CALENDAR_PROVENANCE_MISMATCH"):
        calendar.validate_calendar_provenance({**expected, "fallback_provider": "other"})


def test_canonical_round_trip_and_anchor_states():
    days = (date(2015, 1, 1), date(2020, 10, 2), date(2025, 12, 31))
    raw = calendar.serialize_calendar(days)
    assert raw == b"2015-01-01\n2020-10-02\n2025-12-31\n"
    lock = calendar.RawLock.from_bytes(raw)
    parsed, manifest = calendar.parse_canonical_calendar(lock)
    assert parsed == days
    assert calendar.serialize_calendar(parsed) == raw
    assert manifest["session_sha256"] == manifest["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    calendar.validate_calendar_anchors(parsed)
    with pytest.raises(ValueError, match="CALENDAR_CANONICAL_BYTES_MISMATCH"):
        calendar.parse_canonical_calendar(calendar.RawLock(raw, "0" * 64, len(raw)))
    with pytest.raises(ValueError, match="CALENDAR_ANCHOR_MISMATCH"):
        calendar.validate_calendar_anchors((date(2020, 10, 1), date(2020, 10, 2)))
    with pytest.raises(ValueError, match="CALENDAR_ANCHOR_MISMATCH"):
        calendar.validate_calendar_anchors((date(2020, 9, 30),))


def test_synthetic_schedule_to_existing_parser():
    schedule = pd.DataFrame(
        {"market_close": [pd.Timestamp("2015-01-05T15:00:00+09:00"),
                          pd.Timestamp("2020-10-02T15:00:00+09:00")]},
        index=pd.DatetimeIndex(["2015-01-05", "2020-10-02"]),
    )
    raw = calendar.serialize_calendar_schedule(schedule)
    assert raw == b"2015-01-05\n2020-10-02\n"
    sessions, _ = calendar.parse_canonical_calendar(calendar.RawLock.from_bytes(raw))
    assert sessions == (date(2015, 1, 5), date(2020, 10, 2))

    for bad in (
        schedule.drop(columns="market_close"),
        schedule.iloc[::-1],
        schedule.set_axis(pd.DatetimeIndex(["2020-10-02", "2020-10-02"])),
        schedule.set_axis(pd.DatetimeIndex(["2015-01-05", "2020-10-02"], tz="Asia/Tokyo")),
        schedule.set_axis(pd.DatetimeIndex(["2020-10-02 01:00", "2015-01-05"])),
        schedule.assign(market_close=[pd.Timestamp("2020-10-02 15:00"),
                                      pd.Timestamp("2015-01-05T15:00:00+09:00")]),
    ):
        with pytest.raises(ValueError, match="CALENDAR_SCHEDULE_MISMATCH"):
            calendar.serialize_calendar_schedule(bad)
    with pytest.raises(ValueError, match="CALENDAR_ANCHOR_MISMATCH"):
        calendar.serialize_calendar_schedule(schedule.set_axis(pd.DatetimeIndex(["2015-01-05", "2020-10-01"])))


@pytest.mark.parametrize("days", [
    (), (date(2020, 10, 2), date(2020, 10, 2)),
    (date(2020, 10, 2), date(2020, 9, 30)),
    (date(2014, 12, 31),), (date(2026, 1, 1),),
    (datetime(2020, 10, 2),),
])
def test_serializer_rejects_invalid_sessions(days):
    with pytest.raises(ValueError, match="CALENDAR_ORDER_OR_RANGE_MISMATCH"):
        calendar.serialize_calendar(days)


@pytest.mark.parametrize("raw", [
    b"2020-10-02", b"\xef\xbb\xbf2020-10-02\n", b"2020-10-02\r\n",
    b"2020-10-02\n\n", b"2020-10-02\n2020-10-02\n",
    b"2020-10-02\n2020-09-30\n", b"2014-12-31\n", b"2026-01-01\n",
    b"2020-2-2\n", b"2020-13-02\n", b"2020-10-02\x00\n",
])
def test_bridge_rejects_noncanonical_or_invalid_bytes(raw):
    with pytest.raises((ValueError, UnicodeError)):
        calendar.parse_canonical_calendar(calendar.RawLock.from_bytes(raw))


def test_synthetic_release_byte_binding_rejects_substitution(monkeypatch):
    paths = ["pandas_market_calendars/calendars/jpx.py", "pandas_market_calendars/holidays/jp.py"]
    sources = {paths[0]: b"synthetic jpx", paths[1]: b"synthetic holidays"}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, raw in sources.items():
            archive.writestr(path, raw)
    wheel = output.getvalue()
    blobs = {path: calendar._git_blob_sha1(raw) for path, raw in sources.items()}
    wheel_sha = hashlib.sha256(wheel).hexdigest()
    calendar._validate_wheel_source_bytes(wheel, sources, wheel_sha, blobs)
    synthetic_provenance = dict(calendar.CALENDAR_PROVENANCE)
    synthetic_provenance.update(official_pypi_wheel_sha256=wheel_sha,
                                jpx_source_blob=blobs[paths[0]], holiday_source_blob=blobs[paths[1]])
    monkeypatch.setattr(calendar, "CALENDAR_PROVENANCE", tuple(synthetic_provenance.items()))
    calendar.validate_calendar_release_artifact(wheel, sources, synthetic_provenance)
    with pytest.raises(ValueError, match="CALENDAR_SOURCE_MISMATCH"):
        calendar._validate_wheel_source_bytes(wheel, sources, "0" * 64, blobs)
    with pytest.raises(ValueError, match="CALENDAR_SOURCE_MISMATCH"):
        calendar._validate_wheel_source_bytes(wheel, {**sources, paths[0]: b"changed"}, wheel_sha, blobs)
    with pytest.raises(ValueError, match="CALENDAR_SOURCE_MISMATCH"):
        calendar._validate_wheel_source_bytes(wheel, sources, wheel_sha, {**blobs, paths[1]: "0" * 40})
    with pytest.raises(ValueError, match="CALENDAR_SOURCE_MISMATCH"):
        calendar.validate_calendar_release_artifact(wheel, {**sources, paths[0]: b"changed"}, synthetic_provenance)

    missing = io.BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr(paths[0], sources[paths[0]])
    missing_raw = missing.getvalue()
    with pytest.raises(ValueError, match="CALENDAR_SOURCE_MISMATCH"):
        calendar._validate_wheel_source_bytes(missing_raw, sources, hashlib.sha256(missing_raw).hexdigest(), blobs)

    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        for path, raw in sources.items():
            archive.writestr(path, raw)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(paths[0], sources[paths[0]])
    raw = duplicate.getvalue()
    with pytest.raises(ValueError, match="CALENDAR_SOURCE_MISMATCH"):
        calendar._validate_wheel_source_bytes(raw, sources, hashlib.sha256(raw).hexdigest(), blobs)


def test_no_provider_construction_or_fallback_in_calendar_path():
    module_path = Path(calendar.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("pandas_market_calendars" not in ast.unparse(node) for node in imports)
    for name in ("serialize_calendar", "serialize_calendar_schedule", "parse_canonical_calendar",
                 "validate_calendar_release_artifact"):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert "get_calendar" not in ast.unparse(function)
