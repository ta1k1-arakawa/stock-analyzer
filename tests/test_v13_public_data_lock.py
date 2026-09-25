from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src import v13_public_data_lock as p
from src.v13_feasibility import select_universe as frozen_select


def _t1():
    return [f"{n:04d}" for n in range(1000, 1300)]


def _state(codes):
    return {"schema": "V13_V8_T1_IDENTITY_STATE_V1",
            "source_partition_manifest_stated_sha256": p.SOURCE_SHA256,
            "t1_ticker_list_sha256": p.code_hash(codes), "t1_count": 300,
            "known_definitely_acquired_prefix_count": 297, "t1_membership": codes}


def test_t1_contract_and_exclusions():
    codes = _t1()
    state = _state(codes)
    assert p.validate_t1_state(state, expected_hash=p.code_hash(codes)) == tuple(codes)
    for change in ({"t1_count": True}, {"known_definitely_acquired_prefix_count": 296},
                   {"source_partition_manifest_stated_sha256": "wrong"},
                   {"t1_membership": codes[:-1] + [codes[0]]}):
        with pytest.raises(ValueError):
            p.validate_t1_state({**state, **change}, expected_hash=p.code_hash(codes))
    v4 = [f"{n:04d}" for n in range(1200, 1500)]
    excluded, manifest = p.build_exclusions(v4, codes)
    assert len(excluded) == 507  # 100 T1/V4 overlap plus seven legacy codes
    assert manifest["exclusion_sha256"] == p.code_hash(sorted(excluded))
    assert set(manifest) == {"exclusion_count", "exclusion_sha256", "v4_count",
                             "legacy_outside_v4_count", "t1_count", "t1_sha256"}


def test_v4_csv_and_universe_selector(tmp_path: Path):
    v4 = b"ticker,market,industry\n" + b"".join(f"{n:04d},Prime,Sector\n".encode() for n in range(1000, 1300))
    assert len(p.read_v4_codes(v4)) == 300
    eligible = {f"{n:04d}": "Sector" for n in range(1000, 1900)}
    lock = p.lock_payload(b"synthetic jpx", tmp_path / "jpx.raw")
    chosen, manifest = p.select_universe(eligible, set(_t1()), lock, "a" * 40, "2026-09-25T00:00:00Z")
    assert chosen == frozen_select(eligible, set(_t1()), p.SEED)
    assert len(chosen) == 500 and manifest["selected_sha256"] == p.digest("|".join(chosen).encode())
    assert "selected" not in manifest and "eligible" not in manifest


def test_jpx_filter_sector_and_lock(tmp_path: Path):
    raw = ("コード,市場・区分,33業種区分\n"
           "1000,プライム（内国株式）,機械\n"
           "1001,Standard Domestic Stocks,Services\n"
           "1002,グロース（内国株式）,機械\n"
           "1003,Prime Foreign Stocks,機械\n"
           "100A,Prime Domestic Stocks,機械\n").encode()
    lock = p.lock_payload(raw, tmp_path / "jpx.raw")
    assert (tmp_path / "jpx.raw").read_bytes() == raw
    assert lock.sha256 == hashlib.sha256(raw).hexdigest()
    assert p.parse_jpx(lock) == {"1000": "機械", "1001": "Services"}
    with pytest.raises(FileExistsError):
        p.lock_payload(raw, tmp_path / "jpx.raw")


def test_existing_offline_jpx_workbook_fixture():
    source = Path(__file__).parent / "fixtures" / "synthetic_jpx_source_snapshot.xls"
    # This older fixture intentionally uses alphanumeric identities.
    with pytest.raises(ValueError, match="JPX_NO_ELIGIBLE_CODES"):
        p.parse_jpx(p.RawLock.from_bytes(source.read_bytes()))


def _timestamp(day):
    return int(datetime(day.year, day.month, day.day, 15, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp())


def _yahoo(days):
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": "1000.T"}, "timestamp": [_timestamp(d) for d in days],
        "indicators": {"quote": [{"open": [100.0, 51.0], "high": [110.0, 55.0],
                                  "low": [95.0, 49.0], "close": [100.0, 50.0],
                                  "volume": [1000, 2000]}],
                       "adjclose": [{"adjclose": [40.0, 49.0]}]},
        "events": {"splits": {"one": {"date": _timestamp(days[1]), "numerator": 2,
                                      "denominator": 1}},
                   "dividends": {"one": {"date": _timestamp(days[1]), "amount": 10}}}
    }]}}


def test_yahoo_split_only_raw_execution_and_2026_rejection(tmp_path: Path):
    days = [date(2025, 12, 29), date(2025, 12, 30)]
    raw = json.dumps(_yahoo(days)).encode()
    lock = p.lock_payload(raw, tmp_path / "yahoo.raw")
    rows, manifest = p.parse_yahoo(lock, "1000")
    assert rows[days[0]]["open"] == 100.0
    assert rows[days[0]]["adj_open"] == 50.0
    assert rows[days[1]]["adj_close"] == 50.0
    assert rows[days[0]]["traded_value"] == 100000.0
    assert manifest["split_event_count"] == 1 and manifest["raw_sha256"] == p.digest(raw)
    assert "1000" not in json.dumps(manifest) and "dividends" not in json.dumps(manifest)
    bad = _yahoo([date(2025, 12, 31), date(2026, 1, 2)])
    with pytest.raises(ValueError, match="OUT_OF_WINDOW"):
        p.parse_yahoo(p.RawLock.from_bytes(json.dumps(bad).encode()), "1000")


def test_calendar_exact_offsets_and_retry(tmp_path: Path):
    raw = b"2025-12-26\n2025-12-29\n2025-12-30\n2025-12-31\n"
    sessions, manifest = p.parse_calendar(p.lock_payload(raw, tmp_path / "calendar.raw"))
    assert p.session_offset(sessions, date(2025, 12, 26), 1) == date(2025, 12, 29)
    assert p.session_offset(sessions, date(2025, 12, 26), 3) == date(2025, 12, 31)
    assert p.session_offset(sessions, date(2025, 12, 29), 3) is None
    assert manifest["real_acquisition_scope_extension_required"] is True
    assert p.retry_class("PUBLIC_TRANSPORT", complete_content_locked=False) == "PLUMBING_FAILURE_RETRIABLE"
    for failure, locked in (("PUBLIC_TRANSPORT", True), ("SEMANTIC", False), ("MODEL", False)):
        assert p.retry_class(failure, complete_content_locked=locked) == "NO_RETRY_AUTHORITY"
