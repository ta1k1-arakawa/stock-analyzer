import importlib
import inspect
import json

import pytest

import src.v9_013_v9_012_authority_failure_diagnostic as diag


SMALL_DATES = ["2020-09-30", "2020-10-01", "2020-10-02"]
IMPLEMENTATION_SHA = "1" * 40


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=True).encode("utf-8")


def page(data):
    return encoded({"data": data})


def use_small_coverage(monkeypatch):
    monkeypatch.setattr(diag, "_coverage_dates", lambda: list(SMALL_DATES))


def valid_a_rows():
    return [{"Date": date, "HolDiv": "1"} for date in SMALL_DATES]


def valid_b_rows(*, extra_active=False):
    rows = [
        {"Date": "2020-09-30", "O": 1.0, "H": 2.0, "L": 0.5, "C": 1.5},
        {"Date": "2020-10-01", "O": None, "H": None, "L": None, "C": None},
        {"Date": "2020-10-02", "O": 2.0, "H": 3.0, "L": 1.5, "C": 2.5},
    ]
    if extra_active:
        rows[1] = {"Date": "2020-10-01", "O": 2.0, "H": 3.0, "L": 1.5, "C": 2.5}
    return rows


def valid_result(monkeypatch, *, extra_active=False):
    use_small_coverage(monkeypatch)
    return diag.diagnose_payloads(
        [page(valid_a_rows())],
        [page(valid_b_rows(extra_active=extra_active))],
        diagnostic_implementation_git_sha=IMPLEMENTATION_SHA,
    )


def assert_a_failure(result, category, *, row_index=None, field=None, observed=None):
    assert result["diagnostic_class"] == "SOURCE_A_SEMANTIC_FAILURE"
    assert result["source_a_category"] == category
    location = result["source_a_failure_location"]
    assert location["source_role"] == diag.SOURCE_A_ROLE
    assert location["page_index"] == (1 if category != "A_COVERAGE_DATE_SET_MISMATCH" else None)
    assert location["row_index"] == row_index
    assert location["field_name"] == field
    assert location["observed_json_type"] == observed
    assert result["source_b_category"] is None
    assert result["source_b_row_count"] is None
    assert result["source_b_failure_location"] is None
    assert result["relation_evaluated"] is False
    assert all(result[key] is None for key in diag.RELATION_KEYS)


def assert_b_failure(result, category, *, row_index=1, field=None, observed=None):
    assert result["diagnostic_class"] == "SOURCE_B_SEMANTIC_FAILURE"
    assert result["source_a_category"] == "A_VALID"
    assert result["source_b_category"] == category
    location = result["source_b_failure_location"]
    assert location["source_role"] == diag.SOURCE_B_ROLE
    assert location["page_index"] == 1
    assert location["row_index"] == row_index
    assert location["field_name"] == field
    assert location["observed_json_type"] == observed
    assert result["relation_evaluated"] is False
    assert all(result[key] is None for key in diag.RELATION_KEYS)


@pytest.mark.parametrize(
    ("payload", "category", "field", "observed"),
    [
        (b"not-json", "A_PAYLOAD_JSON_DECODE_FAILURE", None, None),
        (encoded([]), "A_PAYLOAD_ROOT_SCHEMA_FAILURE", "root", "list"),
        (encoded({"data": None}), "A_DATA_FIELD_SCHEMA_FAILURE", "data", "null"),
        (encoded({}), "A_DATA_FIELD_SCHEMA_FAILURE", "data", "null"),
        (page([1]), "A_ROW_SCHEMA_FAILURE", None, "int"),
        (page([{"HolDiv": "1"}]), "A_REQUIRED_FIELD_MISSING", "Date", None),
        (page([{"Date": 1, "HolDiv": "1"}]), "A_DATE_TYPE_OR_FORMAT_INVALID", "Date", "int"),
        (page([{"Date": "2020/01/01", "HolDiv": "1"}]), "A_DATE_TYPE_OR_FORMAT_INVALID", "Date", "string"),
        (page([{"Date": "2020-02-30", "HolDiv": "1"}]), "A_DATE_VALUE_INVALID", "Date", "string"),
        (page([{"Date": "2016-12-31", "HolDiv": "1"}]), "A_DATE_OUT_OF_COVERAGE", "Date", "string"),
        (page([{"Date": "2020-09-30", "HolDiv": 1}]), "A_HOLDIV_TYPE_OR_DOMAIN_INVALID", "HolDiv", "int"),
        (page([{"Date": "2020-09-30", "HolDiv": "9"}]), "A_HOLDIV_TYPE_OR_DOMAIN_INVALID", "HolDiv", "string"),
        (page([{"Date": "2020-09-30", "HolDiv": "1"}, {"Date": "2020-09-30", "HolDiv": "1"}]), "A_DUPLICATE_DATE", "Date", "string"),
    ],
)
def test_every_source_a_category(payload, category, field, observed):
    result = diag.diagnose_payloads([payload], [b"not-json"], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA)
    row_index = None if category in {"A_PAYLOAD_JSON_DECODE_FAILURE", "A_PAYLOAD_ROOT_SCHEMA_FAILURE", "A_DATA_FIELD_SCHEMA_FAILURE"} else (2 if category == "A_DUPLICATE_DATE" else 1)
    assert_a_failure(result, category, row_index=row_index, field=field, observed=observed)


def test_source_a_coverage_mismatch():
    result = diag.diagnose_payloads(
        [page([{"Date": "2020-09-30", "HolDiv": "1"}])],
        [b"not-json"],
        diagnostic_implementation_git_sha=IMPLEMENTATION_SHA,
    )
    assert_a_failure(result, "A_COVERAGE_DATE_SET_MISMATCH", field="Date", observed=None)


def test_source_a_valid_requires_exact_coverage(monkeypatch):
    result = valid_result(monkeypatch)
    assert result["source_a_category"] == "A_VALID"
    assert result["source_a_failure_location"] is None
    assert result["source_a_row_count"] == 3
    assert result["scheduled_open_count"] == 3


@pytest.mark.parametrize(
    ("payload", "category", "field", "observed"),
    [
        (b"not-json", "B_PAYLOAD_JSON_DECODE_FAILURE", None, None),
        (encoded([]), "B_PAYLOAD_ROOT_SCHEMA_FAILURE", "root", "list"),
        (encoded({"data": None}), "B_DATA_FIELD_SCHEMA_FAILURE", "data", "null"),
        (encoded({}), "B_DATA_FIELD_SCHEMA_FAILURE", "data", "null"),
        (page([1]), "B_ROW_SCHEMA_FAILURE", None, "int"),
        (page([{"O": 1, "H": 2, "L": 0, "C": 1}]), "B_REQUIRED_FIELD_MISSING", "Date", None),
        (page([{"Date": "2020-09-30", "H": 2, "L": 0, "C": 1}]), "B_REQUIRED_FIELD_MISSING", "O", None),
        (page([{"Date": 1, "O": 1, "H": 2, "L": 0, "C": 1}]), "B_DATE_TYPE_OR_FORMAT_INVALID", "Date", "int"),
        (page([{"Date": "2020/01/01", "O": 1, "H": 2, "L": 0, "C": 1}]), "B_DATE_TYPE_OR_FORMAT_INVALID", "Date", "string"),
        (page([{"Date": "2020-02-30", "O": 1, "H": 2, "L": 0, "C": 1}]), "B_DATE_VALUE_INVALID", "Date", "string"),
        (page([{"Date": "2016-12-31", "O": 1, "H": 2, "L": 0, "C": 1}]), "B_DATE_OUT_OF_COVERAGE", "Date", "string"),
        (page([{"Date": "2020-09-30", "O": 1, "H": 2, "L": 0, "C": 1}, {"Date": "2020-09-30", "O": None, "H": None, "L": None, "C": None}]), "B_DUPLICATE_DATE", "Date", "string"),
        (page([{"Date": "2020-09-30", "O": None, "H": 2, "L": None, "C": None}]), "B_OHLC_MIXED_NULL_FAILURE", None, None),
        (encoded({"data": [{"Date": "2020-09-30", "O": True, "H": 2, "L": 0, "C": 1}]}), "B_OHLC_NONFINITE_OR_TYPE_FAILURE", "O", "bool"),
        (encoded({"data": [{"Date": "2020-09-30", "O": None, "H": None, "L": None, "C": None}]}), "B_VALID", None, None),
    ],
)
def test_every_source_b_category(monkeypatch, payload, category, field, observed):
    use_small_coverage(monkeypatch)
    result = diag.diagnose_payloads(
        [page(valid_a_rows())], [payload], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA,
    )
    if category == "B_VALID":
        assert result["source_b_category"] == "B_VALID"
        assert result["topix_active_count"] == 0
    else:
        row_index = None if category in {
            "B_PAYLOAD_JSON_DECODE_FAILURE", "B_PAYLOAD_ROOT_SCHEMA_FAILURE",
            "B_DATA_FIELD_SCHEMA_FAILURE",
        } else (2 if category == "B_DUPLICATE_DATE" else 1)
        assert_b_failure(result, category, row_index=row_index, field=field, observed=observed)


def test_source_b_full_finite_failure_and_bool_rejection(monkeypatch):
    use_small_coverage(monkeypatch)
    rows = [{"Date": "2020-09-30", "O": 1, "H": float("nan"), "L": 0, "C": 1}]
    result = diag.diagnose_payloads([page(valid_a_rows())], [encoded({"data": rows})], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA)
    assert_b_failure(result, "B_OHLC_NONFINITE_OR_TYPE_FAILURE", field="H", observed="float")


def test_source_a_global_row_scan_precedes_earlier_row_field_error():
    result = diag.diagnose_payloads(
        [page([{}, 1])], [b"not-json"], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA,
    )
    assert_a_failure(result, "A_ROW_SCHEMA_FAILURE", row_index=2, observed="int")


def test_source_b_global_row_scan_precedes_earlier_row_field_error(monkeypatch):
    use_small_coverage(monkeypatch)
    result = diag.diagnose_payloads(
        [page(valid_a_rows())], [page([{"Date": "2020-09-30"}, 1])], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA,
    )
    assert_b_failure(result, "B_ROW_SCHEMA_FAILURE", row_index=2, observed="int")


def test_required_field_precedence_duplicate_and_ohlc_order(monkeypatch):
    use_small_coverage(monkeypatch)
    a = [page(valid_a_rows())]
    result = diag.diagnose_payloads(a, [page([{}])], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA)
    assert_b_failure(result, "B_REQUIRED_FIELD_MISSING", field="Date")
    result = diag.diagnose_payloads(a, [page([{"Date": "2020-09-30", "O": 1, "H": 2, "L": 0, "C": 1}, {"Date": "2020-09-30", "O": "bad", "H": 2, "L": 0, "C": 1}])], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA)
    assert_b_failure(result, "B_DUPLICATE_DATE", row_index=2, field="Date", observed="string")
    result = diag.diagnose_payloads(a, [encoded({"data": [{"Date": "2020-09-30", "O": "bad", "H": True, "L": float("inf"), "C": 1}]})], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA)
    assert_b_failure(result, "B_OHLC_NONFINITE_OR_TYPE_FAILURE", field="O", observed="string")


def test_source_a_failure_suppresses_source_b_and_relation(monkeypatch):
    use_small_coverage(monkeypatch)
    original = diag._diagnose_source_b
    monkeypatch.setattr(diag, "_diagnose_source_b", lambda _pages: pytest.fail("SOURCE_B semantic path was reached"))
    result = diag.diagnose_payloads([page([{}])], [page(valid_b_rows())], diagnostic_implementation_git_sha=IMPLEMENTATION_SHA)
    assert result["diagnostic_class"] == "SOURCE_A_SEMANTIC_FAILURE"
    monkeypatch.setattr(diag, "_diagnose_source_b", original)


def test_relation_success_class_is_diagnostic_inconsistency(monkeypatch):
    result = valid_result(monkeypatch)
    assert result["diagnostic_class"] == "NO_V9_012_FAILURE_REPRODUCED"
    assert result["relation_evaluated"] is True
    assert result["left_diff_count"] == 1
    assert result["right_diff_count"] == 0
    assert result["unexpected_left_diff_count"] == 0
    assert result["missing_expected_exception_count"] == 0
    assert result["left_exact_expected"] is True
    assert result["right_empty"] is True
    assert result["neighbor_2020_09_30_active"] is True
    assert result["sentinel_2020_10_01_inactive"] is True
    assert result["neighbor_2020_10_02_active"] is True
    assert all(value is not None for key, value in result.items() if key in diag.RELATION_KEYS)


def test_relation_failure_has_all_nonnull_relation_diagnostics(monkeypatch):
    result = valid_result(monkeypatch, extra_active=True)
    assert result["diagnostic_class"] == "RELATION_OR_SENTINEL_FAILURE"
    assert result["relation_evaluated"] is True
    assert result["sentinel_2020_10_01_inactive"] is False
    assert all(result[key] is not None for key in diag.RELATION_KEYS)


def test_public_schema_validator_rejects_extra_keys_nullability_and_raw_date(monkeypatch):
    result = valid_result(monkeypatch)
    extra = dict(result)
    extra["unexpected"] = True
    with pytest.raises(diag.DiagnosticError, match="DIAGNOSTIC_RESULT_VALIDATION_FAILURE"):
        diag.validate_public_result(extra)
    broken = dict(result)
    broken["relation_evaluated"] = False
    with pytest.raises(diag.DiagnosticError, match="DIAGNOSTIC_RESULT_VALIDATION_FAILURE"):
        diag.validate_public_result(broken)
    broken = dict(result)
    broken["source_a_row_count"] = "3"
    with pytest.raises(diag.DiagnosticError, match="DIAGNOSTIC_RESULT_VALIDATION_FAILURE"):
        diag.validate_public_result(broken)
    broken = dict(result)
    broken["diagnostic_implementation_git_sha"] = "2020-01-01"
    with pytest.raises(diag.DiagnosticError, match="DIAGNOSTIC_RESULT_VALIDATION_FAILURE"):
        diag.validate_public_result(broken)


def test_public_serializer_is_canonical_no_final_lf_and_has_no_outcome_or_path_values(monkeypatch):
    result = valid_result(monkeypatch)
    output = diag.serialize_public_result(result)
    assert not output.endswith(b"\n")
    assert b"raw_pages" not in output
    assert b"state-root" not in output
    assert b"2020-09-30" not in output
    assert b"2020-10-01" not in output
    assert b"2020-10-02" not in output
    decoded = json.loads(output.decode("utf-8"))
    assert set(decoded) == diag.PUBLIC_KEYS


def test_unknown_runtime_type_fails_closed():
    with pytest.raises(diag.DiagnosticError, match="DIAGNOSTIC_RESULT_VALIDATION_FAILURE"):
        diag.observed_json_type(object())


def test_frozen_chain_mismatch_stops_before_semantic_diagnosis(tmp_path, monkeypatch):
    (tmp_path / "source_a" / "raw_pages").mkdir(parents=True)
    (tmp_path / "source_a" / "page_locks").mkdir(parents=True)
    (tmp_path / "source_b" / "raw_pages").mkdir(parents=True)
    (tmp_path / "source_b" / "page_locks").mkdir(parents=True)
    monkeypatch.setattr(diag, "_diagnose_source_a", lambda _pages: pytest.fail("semantic read occurred"))
    with pytest.raises(diag.DiagnosticError, match="PRESERVED_V9_012_INPUT_BINDING_FAILURE"):
        diag.diagnose_preserved_state(
            tmp_path,
            diagnostic_design_git_sha=diag.FROZEN_DESIGN_GIT_SHA,
            diagnostic_implementation_git_sha=IMPLEMENTATION_SHA,
        )


def test_no_acquisition_or_network_path_is_imported_or_reachable():
    source = inspect.getsource(diag)
    assert "urllib" not in source
    assert "requests" not in source
    assert "socket" not in source
    assert "API_KEY" not in source
    assert "import src.v9_012" not in source


def test_runner_help_does_not_read_state_root(capsys):
    runner = importlib.import_module("scripts.run_v9_013_v9_012_authority_failure_diagnostic")
    with pytest.raises(SystemExit) as exc:
        runner.main(["--help"])
    assert exc.value.code == 0
    assert "diagnose" in capsys.readouterr().out


def test_runner_known_error_is_safe_and_does_not_echo_state_root(tmp_path, capsys):
    runner = importlib.import_module("scripts.run_v9_013_v9_012_authority_failure_diagnostic")
    secret_path = str(tmp_path / "private-state-root")
    exit_code = runner.main([
        "diagnose", "--state-root", secret_path,
        "--diagnostic-design-git-sha", diag.FROZEN_DESIGN_GIT_SHA,
        "--diagnostic-implementation-git-sha", IMPLEMENTATION_SHA,
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert secret_path not in captured.err
    assert json.loads(captured.err) == {
        "reason": "PRESERVED_V9_012_INPUT_BINDING_FAILURE",
        "status": "BLOCKED",
    }


def test_safe_error_unknown_reason_collapses_to_implementation_failure():
    assert json.loads(diag.safe_error_bytes("secret details").decode("utf-8")) == {
        "reason": "IMPLEMENTATION_FAILURE", "status": "BLOCKED",
    }
