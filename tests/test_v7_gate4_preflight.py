from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.v7_gate4_preflight import (
    CALENDAR_DEFINITION_VERSION,
    CALENDAR_SOURCE,
    CALENDAR_TIMEZONE,
    COLLECTOR_COMMIT,
    DESIGN_COMMIT,
    PREREGISTRATION_UTC,
    V7Gate4PreflightBlocked,
    canonical_json_bytes,
    file_sha256,
    hash_payload_manifest,
    hash_ticker_manifest,
    validate_arm_provenance,
    validate_artifact_hashes,
    validate_calendar_provenance,
    validate_payload_manifest_records,
    validate_seed_manifest_identity,
    validate_seed_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
CALENDAR_JSON = ROOT / "data" / "v7_jpx_calendar_2026_2027.json"
CALENDAR_RAW = ROOT / "data" / "v7_jpx_calendar_source_2026_2027.html"


def _seed_rows(counts=(252, 252), tickers=("A", "B")):
    rows = []
    start = date(2025, 1, 1)
    for ticker, count in zip(tickers, counts):
        for index in range(count):
            day = (start + timedelta(days=index)).isoformat()
            rows.append({
                "ticker": ticker, "trading_date": day,
                "raw_open": 100.0, "raw_high": 101.0, "raw_low": 99.0,
                "raw_close": 100.5, "adj_close": 100.5, "raw_volume": 100000.0,
            })
    return rows


def _raw_bundle(tmp_path, tickers=("A", "B")):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for ticker in tickers:
        body = ("{\"ticker\":\"" + ticker + "\"}\n").encode()
        (raw / f"{ticker}.json").write_bytes(body)
        records.append({"ticker": ticker, "payload_sha256": hashlib.sha256(body).hexdigest(), "byte_count": len(body)})
    return raw, records


def _manifest():
    return {
        "mode": "PRE_ACTIVATION_SEED_ACQUISITION", "design_commit": DESIGN_COMMIT,
        "collector_commit": COLLECTOR_COMMIT, "ticker_count": 300,
        "request_count": 300, "success_count": 300, "failed_count": 0,
        "retry_count": 0, "http_429_count": 0, "eligible_seed_ticker_count": 300,
        "ineligible_seed_ticker_count": 0, "seed_row_count": 75600,
        "activation_boundary_status": "NOT_SET", "activation_status": "NOT_ACTIVATED",
        "study_calendar_generated": False,
        "seed_payload_manifest_sha256": "0" * 64,
        "seed_canonical_csv_sha256": "1" * 64,
        "canonical_price_rows_csv_sha256": "2" * 64,
        "canonical_split_events_sha256": "3" * 64,
        "acquisition_started_utc": "2026-08-07T10:11:28.113966Z",
        "acquisition_completed_utc": "2026-08-07T10:22:32.000446Z",
    }


def _calendar():
    return json.loads(CALENDAR_JSON.read_text(encoding="utf-8"))


def test_payload_manifest_hash_is_canonical_and_deterministic():
    records = [{"ticker": "B", "byte_count": 2}, {"ticker": "A", "byte_count": 1}]
    assert hash_payload_manifest(records) == hash_payload_manifest(copy.deepcopy(records))
    assert len(hash_payload_manifest(records)) == 64


def test_source_payload_mutation_changes_only_source_hash():
    source = [{"ticker": "A", "payload_sha256": "a" * 64, "byte_count": 1}]
    ticker = [{"ticker": "A", "ticker_payload_sha256": "b" * 64}]
    changed = [{**source[0], "byte_count": 2}]
    assert hash_payload_manifest(source) != hash_payload_manifest(changed)
    assert hash_ticker_manifest(ticker) == hash_ticker_manifest(copy.deepcopy(ticker))


def test_selected_seed_mutation_changes_ticker_hash_not_source_hash():
    source = [{"ticker": "A", "payload_sha256": "a" * 64, "byte_count": 1}]
    first = [{"ticker": "A", "ticker_payload_sha256": "b" * 64}]
    second = [{"ticker": "A", "ticker_payload_sha256": "c" * 64}]
    assert hash_payload_manifest(source) == hash_payload_manifest(source)
    assert hash_ticker_manifest(first) != hash_ticker_manifest(second)


def test_payload_manifest_count_mismatch_blocks(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    with pytest.raises(V7Gate4PreflightBlocked, match="COUNT_MISMATCH"):
        validate_payload_manifest_records(records[:1], raw, ["A", "B"])


def test_payload_manifest_duplicate_ticker_blocks(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    duplicate = [records[0], {**records[0]}]
    with pytest.raises(V7Gate4PreflightBlocked, match="DUPLICATE_TICKER"):
        validate_payload_manifest_records(duplicate, raw, ["A", "B"])


def test_payload_manifest_order_mismatch_blocks(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    with pytest.raises(V7Gate4PreflightBlocked, match="ORDER_MISMATCH"):
        validate_payload_manifest_records(records, raw, ["B", "A"])


def test_raw_file_count_mismatch_blocks(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    (raw / "extra.json").write_bytes(b"{}")
    with pytest.raises(V7Gate4PreflightBlocked, match="RAW_FILE_COUNT_MISMATCH"):
        validate_payload_manifest_records(records, raw, ["A", "B"])


def test_raw_payload_sha_mismatch_blocks(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    records[0]["payload_sha256"] = "f" * 64
    with pytest.raises(V7Gate4PreflightBlocked, match="RAW_PAYLOAD_SHA_MISMATCH"):
        validate_payload_manifest_records(records, raw, ["A", "B"])


def test_raw_payload_byte_count_mismatch_blocks(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    records[0]["byte_count"] += 1
    with pytest.raises(V7Gate4PreflightBlocked, match="BYTE_COUNT_MISMATCH"):
        validate_payload_manifest_records(records, raw, ["A", "B"])


def test_payload_manifest_success_checks_all_raw_files(tmp_path):
    raw, records = _raw_bundle(tmp_path)
    result = validate_payload_manifest_records(records, raw, ["A", "B"])
    assert result["raw_file_count"] == result["payload_manifest_record_count"] == 2
    assert result["payload_ticker_order_parity"] is True


def test_seed_semantics_accepts_252_rows_per_ticker():
    result = validate_seed_semantics(_seed_rows(), ["A", "B"], "2026-08-10")
    assert (result["row_count"], result["eligible_ticker_count"], result["ineligible_ticker_count"]) == (504, 2, 0)
    assert len(result["seed_canonical_sha256"]) == 64


def test_seed_semantics_marks_251_rows_ineligible():
    result = validate_seed_semantics(_seed_rows((251, 252)), ["A", "B"], "2026-08-10")
    assert result["ineligible_ticker_count"] == 1


def test_seed_semantics_boundary_row_blocks():
    rows = _seed_rows(); rows.append({**rows[0], "trading_date": "2026-08-10"})
    with pytest.raises(V7Gate4PreflightBlocked, match="SEED_ROW_ON_OR_AFTER"):
        validate_seed_semantics(rows, ["A", "B"], "2026-08-10")


def test_seed_semantics_duplicate_blocks():
    rows = _seed_rows(); rows.append(dict(rows[0]))
    with pytest.raises(V7Gate4PreflightBlocked, match="DUPLICATE_TICKER_DATE"):
        validate_seed_semantics(rows, ["A", "B"], "2026-08-10")


def test_seed_semantics_invalid_adj_close_blocks():
    rows = _seed_rows(); rows[0]["adj_close"] = 0.0
    with pytest.raises(V7Gate4PreflightBlocked, match="NONPOSITIVE_ADJ_CLOSE"):
        validate_seed_semantics(rows, ["A", "B"], "2026-08-10")


def test_seed_semantics_nonfinite_adj_close_blocks():
    rows = _seed_rows(); rows[0]["adj_close"] = float("nan")
    with pytest.raises(V7Gate4PreflightBlocked, match="NONFINITE_ADJ_CLOSE"):
        validate_seed_semantics(rows, ["A", "B"], "2026-08-10")


def test_seed_hash_is_stable_under_input_order():
    first = validate_seed_semantics(_seed_rows(), ["A", "B"], "2026-08-10")
    second = validate_seed_semantics(list(reversed(_seed_rows())), ["A", "B"], "2026-08-10")
    assert first["seed_canonical_sha256"] == second["seed_canonical_sha256"]
    assert first["seed_payload_manifest_sha256"] == second["seed_payload_manifest_sha256"]


def test_artifact_hashes_pass(tmp_path):
    files = {"canonical_price_rows.csv": b"price\n", "canonical_split_events.json": b"[]\n", "seed.csv": b"seed\n"}
    for name, body in files.items(): (tmp_path / name).write_bytes(body)
    manifest = {key: file_sha256(tmp_path / name)[0] for key, name in {
        "canonical_price_rows_csv_sha256": "canonical_price_rows.csv",
        "canonical_split_events_sha256": "canonical_split_events.json",
        "seed_canonical_csv_sha256": "seed.csv",
    }.items()}
    assert validate_artifact_hashes(tmp_path, manifest) == manifest


def test_seed_artifact_hash_mismatch_blocks(tmp_path):
    (tmp_path / "canonical_price_rows.csv").write_bytes(b"p")
    (tmp_path / "canonical_split_events.json").write_bytes(b"s")
    (tmp_path / "seed.csv").write_bytes(b"seed")
    manifest = {"canonical_price_rows_csv_sha256": "0" * 64, "canonical_split_events_sha256": file_sha256(tmp_path / "canonical_split_events.json")[0], "seed_canonical_csv_sha256": file_sha256(tmp_path / "seed.csv")[0]}
    with pytest.raises(V7Gate4PreflightBlocked, match="ARTIFACT_HASH_MISMATCH"):
        validate_artifact_hashes(tmp_path, manifest)


def test_calendar_raw_sha_and_byte_count_pass():
    value = _calendar()
    result = validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())
    assert result["calendar_source_payload_sha256"] == value["source_payload_sha256"]
    assert result["calendar_source_byte_count"] == value["source_byte_count"]


def test_calendar_raw_sha_mismatch_blocks():
    value = _calendar()
    with pytest.raises(V7Gate4PreflightBlocked, match="CALENDAR_RAW_SHA_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes() + b"x")


def test_calendar_raw_byte_count_mismatch_blocks():
    value = _calendar(); value["source_byte_count"] += 1
    with pytest.raises(V7Gate4PreflightBlocked, match="BYTE_COUNT_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())


def test_calendar_source_identity_is_fixed():
    value = _calendar(); value["calendar_source"] = "YAHOO_OBSERVED_DATES"
    with pytest.raises(V7Gate4PreflightBlocked, match="SOURCE_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())


def test_calendar_timezone_and_definition_are_fixed():
    value = _calendar(); value["calendar_timezone"] = "UTC"
    with pytest.raises(V7Gate4PreflightBlocked, match="TIMEZONE_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())
    value = _calendar(); value["calendar_definition_version"] = "OTHER"
    with pytest.raises(V7Gate4PreflightBlocked, match="VERSION_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())


def test_calendar_holiday_counts_and_coverage_are_fixed():
    value = _calendar(); value["market_holidays"] = value["market_holidays"][:-1]
    with pytest.raises(V7Gate4PreflightBlocked, match="HOLIDAY_COUNT_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())
    value = _calendar(); value["coverage_end"] = "2028-12-31"
    with pytest.raises(V7Gate4PreflightBlocked, match="COVERAGE_END_MISMATCH"):
        validate_calendar_provenance(value, CALENDAR_RAW.read_bytes())


def test_calendar_snapshot_hash_is_deterministic():
    value = _calendar(); first = canonical_json_bytes(value)
    second = canonical_json_bytes(json.loads(first.decode()))
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_calendar_metadata_constants_match_source():
    value = _calendar()
    assert (value["calendar_source"], value["calendar_timezone"], value["calendar_definition_version"]) == (CALENDAR_SOURCE, CALENDAR_TIMEZONE, CALENDAR_DEFINITION_VERSION)


def test_arm_hashes_are_actual_and_single_parameter():
    result = validate_arm_provenance()
    assert len(result["arm_a_parameters_sha256"]) == len(result["arm_b_parameters_sha256"]) == 64
    assert result["single_parameter_difference"] == {"max_open_positions": [2, 3]}


def test_arm_shared_rules_hash_is_stable():
    first = validate_arm_provenance(); second = validate_arm_provenance()
    assert first["shared_rules_sha256"] == second["shared_rules_sha256"]


def test_preregistration_constant_is_utc_and_fixed():
    assert PREREGISTRATION_UTC == "2026-08-07T02:48:27Z"
    assert DESIGN_COMMIT == "e3e1367efd913b601a70328a815d88c20af6d147"


def test_manifest_identity_passes_fixed_acquisition_metadata():
    validate_seed_manifest_identity(_manifest())


def test_manifest_identity_wrong_mode_blocks():
    value = _manifest(); value["mode"] = "ACTIVE"
    with pytest.raises(V7Gate4PreflightBlocked, match="SEED_MODE_MISMATCH"):
        validate_seed_manifest_identity(value)


def test_manifest_identity_wrong_lineage_blocks():
    value = _manifest(); value["collector_commit"] = "f" * 40
    with pytest.raises(V7Gate4PreflightBlocked, match="COLLECTOR_COMMIT_MISMATCH"):
        validate_seed_manifest_identity(value)


def test_manifest_identity_bad_acquisition_order_blocks():
    value = _manifest(); value["acquisition_started_utc"] = "2026-08-07T02:00:00Z"
    with pytest.raises(V7Gate4PreflightBlocked, match="ACQUISITION_PREREGISTRATION_ORDER_INVALID"):
        validate_seed_manifest_identity(value)


def test_manifest_identity_requires_utc_aware_timestamp():
    value = _manifest(); value["acquisition_started_utc"] = "2026-08-07T10:11:28"
    with pytest.raises(V7Gate4PreflightBlocked, match="UTC_TIMESTAMP"):
        validate_seed_manifest_identity(value)


def test_no_activation_fields_are_created_by_helpers():
    result = validate_arm_provenance()
    assert "activation_authorization_utc" not in result
    assert "activation_manifest_created" not in result


def test_network_and_study_side_effects_are_not_in_preflight_module():
    source = (ROOT / "src" / "v7_gate4_preflight.py").read_text(encoding="utf-8")
    assert "urlopen" not in source and "urllib" not in source and "requests" not in source
    assert "run_gate4_preflight" in source


def test_actual_calendar_known_boundary_metadata():
    value = _calendar()
    assert value["activation_boundary_status"] == "NOT_SET"
    assert value["study_calendar_generated"] is False
