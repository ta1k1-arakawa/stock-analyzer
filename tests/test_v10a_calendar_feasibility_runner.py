import hashlib
import inspect
import json
import sys
from copy import deepcopy

import pandas as pd
import pytest

from scripts import v10a_calendar_feasibility_runner as runner


IMPLEMENTATION_SHA = "1" * 40


def schedule(labels, closes=None):
    if closes is None:
        closes = [pd.Timestamp("2020-10-02 15:00", tz="Asia/Tokyo")] * len(labels)
    return pd.DataFrame({"market_close": closes}, index=labels)


def valid_schedule():
    return schedule(["2020-10-02", "2019-12-30"], [
        pd.Timestamp("2020-10-02 12:30", tz="Asia/Tokyo"),
        pd.Timestamp("2019-12-30 15:00", tz="Asia/Tokyo"),
    ])


def assert_failure(frame, code):
    with pytest.raises(runner.CalendarFeasibilityError, match=f"^{code}$"):
        runner.validate_schedule(frame)


def test_valid_synthetic_schedule_passes_and_anchors_are_fixed():
    result = runner.validate_schedule(valid_schedule())
    assert result.trading_dates == ("2019-12-30", "2020-10-02")
    assert result.anchor_2020_10_01 == "INELIGIBLE"
    assert result.anchor_2020_10_02 == "ELIGIBLE"


def test_shortened_or_early_emitted_session_is_still_eligible():
    frame = schedule(["2020-10-02"], [pd.Timestamp("2020-10-02 10:00", tz="Asia/Tokyo")])
    assert runner.validate_schedule(frame).anchor_2020_10_02 == "ELIGIBLE"


def test_duplicate_canonical_label_precedes_other_schedule_failures():
    assert_failure(schedule(["2020-10-02", pd.Timestamp("2020-10-02")], [pd.NaT, pd.NaT]),
                   "DUPLICATE_SESSION_LABEL")


@pytest.mark.parametrize("label", ["not-a-date", pd.Timestamp("2020-10-02", tz="UTC"), "2020-10-02 00:01"])
def test_malformed_session_labels_fail(label):
    assert_failure(schedule([label]), "MALFORMED_SESSION_LABEL")


def test_out_of_coverage_label_fails_after_valid_canonicalization():
    assert_failure(schedule(["2026-02-01"]), "OUT_OF_COVERAGE_SESSION_LABEL")


@pytest.mark.parametrize("frame", [
    pd.DataFrame(index=["2020-10-02"]),
    schedule(["2020-10-02"], ["not-a-timestamp"]),
    schedule(["2020-10-02"], [pd.NaT]),
    schedule(["2020-10-02"], [pd.Timestamp("2020-10-02 15:00")]),
])
def test_missing_or_invalid_market_close_fails(frame):
    assert_failure(frame, "INVALID_MARKET_CLOSE")


@pytest.mark.parametrize("frame, code", [
    (pd.DataFrame(index=["2020-10-02", "2020-10-02"]), "DUPLICATE_SESSION_LABEL"),
    (pd.DataFrame(index=["not-a-date"]), "MALFORMED_SESSION_LABEL"),
    (pd.DataFrame(index=["2026-02-01"]), "OUT_OF_COVERAGE_SESSION_LABEL"),
])
def test_missing_market_close_does_not_bypass_higher_precedence_label_failures(frame, code):
    assert_failure(frame, code)


def test_anchor_failures_are_ordered_after_session_checks():
    assert_failure(schedule(["2020-10-01"]), "ANCHOR_2020_10_01_FAILURE")
    assert_failure(schedule(["2020-10-03"]), "ANCHOR_2020_10_02_FAILURE")


def test_anchor_failures_retain_only_observed_anchor_classifications():
    with pytest.raises(runner.CalendarFeasibilityError) as first:
        runner.validate_schedule(schedule(["2020-10-01"]))
    assert (first.value.anchor_2020_10_01, first.value.anchor_2020_10_02) == ("ELIGIBLE", "NOT_CHECKED")
    with pytest.raises(runner.CalendarFeasibilityError) as second:
        runner.validate_schedule(schedule(["2020-10-03"]))
    assert (second.value.anchor_2020_10_01, second.value.anchor_2020_10_02) == ("INELIGIBLE", "INELIGIBLE")


def test_runtime_lock_provenance_mismatch_and_legacy_blob_fail_closed():
    with pytest.raises(runner.CalendarFeasibilityError, match="RUNTIME_CALENDAR_PROVENANCE_MISMATCH"):
        runner.verify_runtime_lock_bytes(b"{}")
    assert runner.CALENDAR_SOURCE_BLOB != runner.LEGACY_V10_CALENDAR_SOURCE_BLOB


def test_canonical_artifact_exact_keys_and_self_excluding_digest():
    artifact = runner.build_canonical_artifact(("2019-12-30", "2020-10-02"), IMPLEMENTATION_SHA)
    assert set(artifact) == set(runner.ARTIFACT_KEYS)
    assert artifact["trading_date_count"] == len(artifact["trading_dates"])
    assert artifact["trading_dates"] == sorted(set(artifact["trading_dates"]))
    without_digest = dict(artifact)
    digest = without_digest.pop("canonical_calendar_sha256")
    assert digest == hashlib.sha256(runner.canonical_json_bytes(without_digest)).hexdigest()
    runner.validate_canonical_artifact(artifact)


def test_canonical_json_is_compact_lf_terminated_and_has_no_cr():
    encoded = runner.canonical_json_bytes({"z": 1, "a": "x"})
    assert encoded == b'{"a":"x","z":1}\n'
    assert b"\r" not in encoded


def test_canonicalization_failure_for_unsorted_or_duplicate_dates():
    with pytest.raises(runner.CalendarFeasibilityError, match="CANONICALIZATION_FAILURE"):
        runner.build_canonical_artifact(("2020-10-02", "2019-12-30"), IMPLEMENTATION_SHA)


def reseal_artifact(artifact):
    without_digest = dict(artifact)
    without_digest.pop("canonical_calendar_sha256")
    artifact["canonical_calendar_sha256"] = hashlib.sha256(runner.canonical_json_bytes(without_digest)).hexdigest()


@pytest.mark.parametrize("field, value", [
    ("generator_implementation_git_sha", "A" * 40),
    ("trading_dates", ["2020-1-02"]),
    ("trading_date_count", True),
    ("trading_date_count", 0),
])
def test_canonical_artifact_validator_rejects_frozen_contract_violations(field, value):
    artifact = deepcopy(runner.build_canonical_artifact(("2019-12-30", "2020-10-02"), IMPLEMENTATION_SHA))
    artifact[field] = value
    reseal_artifact(artifact)
    with pytest.raises(runner.CalendarFeasibilityError, match="CANONICALIZATION_FAILURE"):
        runner.validate_canonical_artifact(artifact)


def test_safe_receipt_exact_keys_and_fixed_non_authority_counters():
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="FAIL", failure_code="INVALID_MARKET_CLOSE",
                                        calendar_artifact_created=False, canonical_calendar_sha256=None,
                                        trading_date_count=None, anchor_2020_10_01="NOT_CHECKED",
                                        anchor_2020_10_02="NOT_CHECKED")
    assert set(receipt) == set(runner.RECEIPT_KEYS)
    assert receipt["research_data_network_requests"] == 0
    assert receipt["historical_calendar_data_acquisition"] == 0
    assert receipt["private_or_sealed_reads"] == 0
    assert receipt["human_gate_consumed"] == 0
    assert receipt["t0_run"] == "NOT_RUN"


def test_receipt_validator_rejects_extra_field_and_invalid_pass_semantics():
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="FAIL", failure_code="INVALID_MARKET_CLOSE",
                                        calendar_artifact_created=False, canonical_calendar_sha256=None,
                                        trading_date_count=None, anchor_2020_10_01="NOT_CHECKED",
                                        anchor_2020_10_02="NOT_CHECKED")
    receipt["extra"] = True
    with pytest.raises(ValueError, match="field set"):
        runner.validate_safe_receipt(receipt)


@pytest.mark.parametrize("mutator", [
    lambda receipt: receipt.__setitem__("canonical_calendar_sha256", "not-a-hash"),
    lambda receipt: receipt.__setitem__("trading_date_count", 0),
    lambda receipt: receipt.__setitem__("trading_date_count", True),
])
def test_pass_receipt_validator_rejects_malformed_hash_and_invalid_counts(mutator):
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="PASS", failure_code="NONE",
                                        calendar_artifact_created=True, canonical_calendar_sha256="a" * 64,
                                        trading_date_count=2, anchor_2020_10_01="INELIGIBLE",
                                        anchor_2020_10_02="ELIGIBLE")
    mutator(receipt)
    with pytest.raises(ValueError):
        runner.validate_safe_receipt(receipt)


@pytest.mark.parametrize("field, value", [
    ("canonical_calendar_sha256", "a" * 64),
    ("trading_date_count", 2),
])
def test_pre_artifact_failure_rejects_non_null_prepared_values(field, value):
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="FAIL", failure_code="INVALID_MARKET_CLOSE",
                                        calendar_artifact_created=False, canonical_calendar_sha256=None,
                                        trading_date_count=None, anchor_2020_10_01="NOT_CHECKED",
                                        anchor_2020_10_02="NOT_CHECKED")
    receipt[field] = value
    with pytest.raises(ValueError, match="pre-artifact"):
        runner.validate_safe_receipt(receipt)


@pytest.mark.parametrize("field", ["canonical_calendar_sha256", "trading_date_count"])
def test_durable_write_failure_requires_prepared_hash_and_count(field):
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="FAIL",
                                        failure_code="DURABLE_ARTIFACT_WRITE_FAILURE",
                                        calendar_artifact_created=False, canonical_calendar_sha256="a" * 64,
                                        trading_date_count=2, anchor_2020_10_01="INELIGIBLE",
                                        anchor_2020_10_02="ELIGIBLE")
    receipt[field] = None
    with pytest.raises(ValueError, match="durable-write"):
        runner.validate_safe_receipt(receipt)


def valid_artifact_and_receipt():
    artifact = runner.build_canonical_artifact(("2019-12-30", "2020-10-02"), IMPLEMENTATION_SHA)
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="PASS", failure_code="NONE",
                                        calendar_artifact_created=True,
                                        canonical_calendar_sha256=artifact["canonical_calendar_sha256"],
                                        trading_date_count=artifact["trading_date_count"],
                                        anchor_2020_10_01="INELIGIBLE", anchor_2020_10_02="ELIGIBLE")
    return artifact, receipt


def test_cross_artifact_pass_validator_accepts_exact_synthetic_pair():
    artifact, receipt = valid_artifact_and_receipt()
    runner.validate_persisted_pass_artifacts(artifact, receipt, IMPLEMENTATION_SHA)


@pytest.mark.parametrize("target, field, value", [
    ("receipt", "canonical_calendar_sha256", "b" * 64),
    ("receipt", "trading_date_count", 3),
    ("receipt", "generator_implementation_git_sha", "2" * 40),
])
def test_cross_artifact_pass_validator_rejects_independent_mismatches(target, field, value):
    artifact, receipt = valid_artifact_and_receipt()
    selected = artifact if target == "artifact" else receipt
    selected[field] = value
    with pytest.raises((ValueError, runner.CalendarFeasibilityError)):
        runner.validate_persisted_pass_artifacts(artifact, receipt, IMPLEMENTATION_SHA)


def test_failure_precedence_runtime_then_duplicate_then_malformed():
    with pytest.raises(runner.CalendarFeasibilityError, match="RUNTIME_CALENDAR_PROVENANCE_MISMATCH"):
        runner.verify_runtime_lock_bytes(b"not-json")
    assert_failure(schedule(["2020-10-02", "2020-10-02", "bad"], [pd.NaT] * 3),
                   "DUPLICATE_SESSION_LABEL")


def test_exclusive_durable_write_never_overwrites(tmp_path):
    path = tmp_path / runner.CANONICAL_ARTIFACT_NAME
    runner._write_new(path, b"first")
    with pytest.raises(FileExistsError):
        runner._write_new(path, b"second")
    assert path.read_bytes() == b"first"


def test_cli_has_only_operational_inputs_and_rejects_methodology_overrides(capsys):
    parser = runner.build_parser()
    help_text = parser.format_help()
    for forbidden in ("--calendar-name", "--provider", "--coverage", "--anchor", "--fallback", "--retry", "--repair", "--force", "--synthetic"):
        assert forbidden not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["--calendar-name", "JPX"])


def test_module_and_synthetic_tests_do_not_import_or_create_real_calendar():
    source = inspect.getsource(runner)
    assert "import pandas_market_calendars as mcal" in source
    assert "get_calendar(CALENDAR_NAME)" in source
    assert "pandas_market_calendars" not in sys.modules


def test_receipt_canonical_json_round_trips_without_extra_fields():
    receipt = runner.build_safe_receipt(IMPLEMENTATION_SHA, status="FAIL", failure_code="ANCHOR_2020_10_02_FAILURE",
                                        calendar_artifact_created=False, canonical_calendar_sha256=None,
                                        trading_date_count=None, anchor_2020_10_01="INELIGIBLE",
                                        anchor_2020_10_02="INELIGIBLE")
    assert set(json.loads(runner.canonical_json_bytes(receipt))) == set(runner.RECEIPT_KEYS)
