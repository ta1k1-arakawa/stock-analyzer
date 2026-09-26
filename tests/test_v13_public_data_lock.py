from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src import v13_public_data_lock as p
from src.v13_feasibility import select_universe as frozen_select
from scripts import v13_resolve_jquants_t1_exclusion_state as producer
from scripts import v13_public_data_lock_execute as runner


def _t1():
    return [f"{n:04d}" for n in range(1000, 1300)]


def _state(codes):
    return {"schema": "V13_JQUANTS_T1_EXCLUSION_STATE_V1",
            "source_schema": p.SOURCE_SCHEMA, "source_contract": p.SOURCE_CONTRACT,
            "source_commit": p.SOURCE_COMMIT, "source_blob": p.SOURCE_BLOB,
            "eligible_count": 3110, "eligible_sha256": p.ELIGIBLE_SHA256,
            "t1_ticker_list_sha256": p.code_hash(codes), "t1_count": 300,
            "known_definitely_acquired_prefix_count": 297,
            "exclusion_disposition": "EXCLUDE_FULL_RECOVERED_T1_BLOCK", "t1_membership": codes}


def test_t1_contract_and_exclusions():
    assert (p.SOURCE_SCHEMA, p.SOURCE_CONTRACT, p.SOURCE_COMMIT, p.SOURCE_BLOB,
            p.ELIGIBLE_SHA256, p.T1_SHA256) == (
                producer.SCHEMA, producer.CONTRACT, producer.SOURCE_COMMIT,
                producer.SOURCE_BLOB, producer.ELIGIBLE_SHA, producer.T1_SHA)
    codes = _t1()
    state = _state(codes)
    assert p.validate_t1_state(state, expected_hash=p.code_hash(codes)) == tuple(codes)
    for change in ({"schema": "V13_V8_T1_IDENTITY_STATE_V1"},
                   {"source_schema": "wrong"}, {"source_contract": "wrong"},
                   {"source_commit": "wrong"}, {"source_blob": "wrong"},
                   {"eligible_count": 3109}, {"eligible_sha256": "wrong"},
                   {"t1_count": True}, {"t1_count": 299},
                   {"t1_ticker_list_sha256": "wrong"},
                   {"known_definitely_acquired_prefix_count": 296},
                   {"exclusion_disposition": "wrong"},
                   {"t1_membership": codes[:-1] + [codes[0]]},
                   {"t1_membership": codes[:-1] + ["@BAD"]},
                   {"t1_membership": codes[::-1]},
                   {"extra": "wrong"}):
        with pytest.raises(ValueError):
            p.validate_t1_state({**state, **change}, expected_hash=p.code_hash(codes))
    with pytest.raises(ValueError):
        p.validate_t1_state({k: v for k, v in state.items() if k != "source_blob"}, expected_hash=p.code_hash(codes))
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


def test_selector_mixed_codes_deterministic_and_numeric_compatible(monkeypatch):
    numeric = [f"{n:04d}" for n in range(1000, 1600)]
    mixed = numeric + ["130A", "130B", "1A30"]
    seed = p.SEED
    expected_numeric = sorted(numeric, key=lambda c: (hashlib.sha256((seed + "|" + c).encode()).hexdigest(), int(c)))[:500]
    assert frozen_select(numeric, [], seed) == expected_numeric
    expected_mixed = sorted(mixed, key=lambda c: (hashlib.sha256((seed + "|" + c).encode()).hexdigest(), c))[:500]
    assert frozen_select(reversed(mixed), [], seed) == frozen_select(mixed, [], seed) == expected_mixed
    assert any(code in expected_mixed for code in ("130A", "130B", "1A30"))
    assert frozen_select([c.lower() for c in mixed], [], seed) == expected_mixed
    monkeypatch.setattr("src.v13_feasibility.sha256_text", lambda value: "same")
    assert frozen_select(mixed, [], seed) == sorted(mixed)[:500]


def test_jpx_filter_sector_and_lock(tmp_path: Path):
    raw = ("コード,市場・区分,33業種区分\n"
           "1000,プライム（内国株式）,機械\n"
           "1001,Standard Domestic Stocks,Services\n"
           "1002,グロース（内国株式）,機械\n"
           "1003,Prime Foreign Stocks,機械\n"
           "100a,Prime Domestic Stocks,機械\n"
           "1005,Prime Domestic Stocks ETF,機械\n"
           "1004,Prime ETF,機械\n").encode()
    lock = p.lock_payload(raw, tmp_path / "jpx.raw")
    assert (tmp_path / "jpx.raw").read_bytes() == raw
    assert lock.sha256 == hashlib.sha256(raw).hexdigest()
    assert p.parse_jpx(lock) == {"1000": "機械", "1001": "Services", "100A": "機械"}
    with pytest.raises(FileExistsError):
        p.lock_payload(raw, tmp_path / "jpx.raw")


def test_existing_offline_jpx_workbook_fixture():
    source = Path(__file__).parent / "fixtures" / "synthetic_jpx_source_snapshot.xls"
    eligible = p.parse_jpx(p.RawLock.from_bytes(source.read_bytes()))
    assert eligible == {"ZZA1": "SYNTHETIC_SECTOR_A", "ZZB2": "SYNTHETIC_SECTOR_A",
                        "ZZC3": "SYNTHETIC_SECTOR_B", "ZZD4": "SYNTHETIC_SECTOR_B",
                        "ZZE5": "SYNTHETIC_SECTOR_C"}


def _timestamp(day):
    return int(datetime(day.year, day.month, day.day, 15, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp())


def _yahoo(days, code="1000"):
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": code + ".T"}, "timestamp": [_timestamp(d) for d in days],
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
    alpha = p.RawLock.from_bytes(json.dumps(_yahoo(days, "130A")).encode())
    alpha_rows, _ = p.parse_yahoo(alpha, "130a")
    assert len(alpha_rows) == 2


def test_calendar_exact_offsets_and_retry(tmp_path: Path):
    raw = b"2025-12-26\n2025-12-29\n2025-12-30\n2025-12-31\n"
    sessions, manifest = p.parse_calendar(p.lock_payload(raw, tmp_path / "calendar.raw"))
    assert p.session_offset(sessions, date(2025, 12, 26), 1) == date(2025, 12, 29)
    assert p.session_offset(sessions, date(2025, 12, 26), 3) == date(2025, 12, 31)
    assert p.session_offset(sessions, date(2025, 12, 29), 3) is None
    assert manifest["real_acquisition_scope_extension_required"] is False
    assert p.retry_class("PUBLIC_TRANSPORT", complete_content_locked=False) == "PLUMBING_FAILURE_RETRIABLE"
    for failure, locked in (("PUBLIC_TRANSPORT", True), ("SEMANTIC", False), ("MODEL", False)):
        assert p.retry_class(failure, complete_content_locked=locked) == "NO_RETRY_AUTHORITY"


def test_frozen_calendar_result_binding(monkeypatch):
    result_path = Path(__file__).resolve().parents[1] / "docs/v13/V13_MASTER_CALENDAR_REAL_GENERATION_SAFE_RESULT.json"
    raw = result_path.read_bytes()
    assert p._git_blob_sha1(raw) == p.MASTER_CALENDAR_SAFE_RESULT_BLOB
    p.validate_generated_calendar_result(raw)
    result = json.loads(raw)
    for key, value in (("schema", "bad"), ("status", "FAIL"), ("gate_consumed", False),
                       ("source_identity", "OTHER"), ("session_count", 2686),
                       ("anchor_2020_10_01", "ELIGIBLE"),
                       ("calendar_sha256", "0" * 64)):
        altered = {**result, key: value}
        with pytest.raises(ValueError):
            p.validate_generated_calendar_result(json.dumps(altered).encode())
    with pytest.raises(ValueError, match="CALENDAR_LOCK_MISMATCH"):
        p.validate_generated_calendar(b"synthetic", "0" * 64)
    monkeypatch.setattr(p, "digest", lambda raw: p.MASTER_CALENDAR_SHA256)
    monkeypatch.setattr(p, "parse_canonical_calendar",
                        lambda lock: ((date(2020, 10, 2),), {}))
    with pytest.raises(ValueError, match="COUNT_OR_SPAN"):
        p.validate_generated_calendar(b"synthetic", p.MASTER_CALENDAR_SHA256)
    monkeypatch.setattr(p, "parse_canonical_calendar",
                        lambda lock: ((date(2020, 10, 1), date(2020, 10, 2)), {}))
    with pytest.raises(ValueError, match="ANCHOR"):
        p.validate_generated_calendar(b"synthetic", p.MASTER_CALENDAR_SHA256)


def _synthetic_run(tmp_path, monkeypatch):
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("synthetic")
    t1 = tmp_path / "t1.json"
    t1.write_text("{}")
    v4 = tmp_path / "v4.csv"
    v4.write_text("synthetic")
    monkeypatch.setattr(p, "validate_generated_calendar",
                        lambda raw, sha: ((date(2025, 12, 30),),
                                          {"session_count": 2687, "session_sha256": p.MASTER_CALENDAR_SHA256,
                                           "real_acquisition_scope_extension_required": False}))
    monkeypatch.setattr(p, "validate_t1_state", lambda state: tuple(_t1()))
    monkeypatch.setattr(p, "read_v4_codes", lambda raw: tuple(f"{n:04d}" for n in range(1200, 1500)))
    output = tmp_path / "output"
    args = {"t1_state": t1, "v4_csv": v4, "calendar_lock": calendar,
            "calendar_sha256": p.MASTER_CALENDAR_SHA256, "output": output,
            "implementation_sha": "a" * 40}
    counts = {}
    fail_code = [None]
    semantic_code = [None]
    def fetch(url):
        counts[url] = counts.get(url, 0) + 1
        if "jpx.co.jp" in url:
            if url == p.JPX_PAGE:
                return b'<a href="/data_j.xls">listed</a>'
            return ("コード,市場・区分,33業種区分\n" +
                    "".join(f"{n:04d},Prime Domestic Stocks,Sector\n" for n in range(2000, 2600))).encode()
        code = url.split("/chart/")[1].split(".T")[0]
        if code == fail_code[0]:
            raise OSError("synthetic transport")
        if code == semantic_code[0]:
            return b'{"chart":{"error":"synthetic","result":[]}}'
        timestamp = _timestamp(date(2025, 12, 30))
        return json.dumps({"chart": {"error": None, "result": [{
            "meta": {"symbol": code + ".T"}, "timestamp": [timestamp],
            "indicators": {"quote": [{"open": [10], "high": [11],
                                      "low": [9], "close": [10], "volume": [100]}]}
        }]}}).encode()
    return args, output, counts, fail_code, semantic_code, fetch


def test_resume_only_fetches_missing_payloads(tmp_path, monkeypatch, capsys):
    args, output, counts, fail_code, _, fetch = _synthetic_run(tmp_path, monkeypatch)
    selected = frozen_select({f"{n:04d}": "Sector" for n in range(2000, 2600)},
                             set(_t1()) | set(f"{n:04d}" for n in range(1200, 1500))
                             | p.LEGACY_OUTSIDE_V4, p.SEED)
    fail_code[0] = selected[4]
    with pytest.raises(OSError):
        runner.execute(**args, fetch=fetch)
    before = dict(counts)
    assert len(before) == 7  # two JPX and five Yahoo attempts
    fail_code[0] = None
    safe = runner.execute(**args, fetch=fetch)
    assert safe["yahoo_payload_count"] == 500
    for url, count in before.items():
        assert counts[url] == (count + 1 if selected[4] in url else count)
    assert counts[p.JPX_PAGE] == 1
    assert sum(counts.values()) == 502 + 1
    assert "selected" not in safe and "t1_membership" not in json.dumps(safe)
    assert selected[0] not in json.dumps(safe)
    assert capsys.readouterr().out == ""
    previous = dict(counts)
    runner.execute(**args, fetch=fetch)
    assert counts == previous
    missing = output / f"price-{selected[0]}.private.json"
    missing.unlink()
    runner.execute(**args, fetch=fetch)
    assert missing.exists() and counts == previous
    selected_state = output / "selected.private.json"
    selected_state.write_text(json.dumps({"selected": selected[::-1]}))
    with pytest.raises(ValueError, match="DURABLE_STATE_MISMATCH"):
        runner.execute(**args, fetch=fetch)
    assert counts == previous


def test_semantic_failure_and_pending_fail_closed(tmp_path, monkeypatch):
    args, output, counts, _, semantic_code, fetch = _synthetic_run(tmp_path, monkeypatch)
    selected = frozen_select({f"{n:04d}": "Sector" for n in range(2000, 2600)},
                             set(_t1()) | set(f"{n:04d}" for n in range(1200, 1500))
                             | p.LEGACY_OUTSIDE_V4, p.SEED)
    semantic_code[0] = selected[0]
    with pytest.raises(ValueError, match="YAHOO_CHART_ERROR"):
        runner.execute(**args, fetch=fetch)
    semantic_code[0] = None
    previous = dict(counts)
    with pytest.raises(ValueError, match="YAHOO_CHART_ERROR"):
        runner.execute(**args, fetch=fetch)
    assert counts == previous
    raw = output / f"yahoo-{selected[0]}.raw"
    pending = raw.with_name(raw.name + ".pending")
    pending.write_bytes(b"ambiguous")
    with pytest.raises(ValueError, match="AMBIGUOUS_PENDING_RAW_LOCK"):
        runner.execute(**args, fetch=fetch)
    assert pending.exists() and counts == previous


def test_runner_rejects_calendar_drift_before_network(tmp_path, monkeypatch):
    args, output, counts, _, _, fetch = _synthetic_run(tmp_path, monkeypatch)
    def validate(raw, sha):
        if sha != p.MASTER_CALENDAR_SHA256:
            raise ValueError("CALENDAR_LOCK_MISMATCH")
        return ((), {"session_count": 2687})
    monkeypatch.setattr(p, "validate_generated_calendar", validate)
    result = tmp_path / "wrong-result.json"
    original = Path(__file__).resolve().parents[1] / "docs/v13/V13_MASTER_CALENDAR_REAL_GENERATION_SAFE_RESULT.json"
    changed = json.loads(original.read_bytes())
    changed["source_identity"] = "OTHER"
    result.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="CALENDAR_RESULT_BLOB_MISMATCH"):
        runner.execute(**args, fetch=fetch, safe_result_path=result)
    with pytest.raises(ValueError, match="CALENDAR_LOCK_MISMATCH"):
        runner.execute(**{**args, "calendar_sha256": "0" * 64}, fetch=fetch)
    assert not output.exists() and counts == {}


def test_empty_complete_response_is_locked(tmp_path: Path):
    destination = tmp_path / "empty.raw"
    lock = p.lock_payload(b"", destination)
    assert lock.byte_count == 0 and destination.exists()
    assert p.existing_raw_lock(destination).raw == b""
    with pytest.raises(FileExistsError):
        p.lock_payload(b"later", destination)
