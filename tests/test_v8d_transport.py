from __future__ import annotations

import errno
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from src import v8d_audit, v8d_historical_acquisition as acquisition, v8d_readiness as readiness
from src.v7_yahoo_collector import V7YahooCollectorBlocked
from src.v8d_transport import (
    BACKOFF_SECONDS,
    CANONICAL_PARSER_CLASSIFIER_BLOB,
    CANONICAL_PARSER_CLASSIFIER_COMMIT,
    FROZEN_DESIGN_COMMIT,
    DurableV8DAuditStore,
    V8DAuditPersistenceBlocked,
    V8DNamedFailure,
    V8DRequestContext,
    V8DRequestPlan,
    V8DTransportBlocked,
    attempt_with_frozen_retry,
    build_yahoo_request_plan,
    canonical_json_bytes,
    canonical_sha256,
    classify_transport_exception,
    execute_v8d_stage,
    make_request_fingerprint,
    origin_guard_evidence,
    sha256_url,
)


IMPLEMENTATION_SHA = "a" * 40
SAFE_URL = "https://query1.finance.yahoo.com/synthetic"


def _raise(error: BaseException):
    def call():
        raise error
    return call


def _plan(stage: str, coordinate: int, request_fn, *, start="2025-12-01", end="2025-12-08") -> V8DRequestPlan:
    block = "T1C" if stage.startswith("T1C") else "T2"
    return V8DRequestPlan(
        request_fn=request_fn,
        request_fingerprint=make_request_fingerprint(
            logical_stage=stage,
            logical_block=block,
            logical_coordinate=coordinate,
            window_start=start,
            window_end_exclusive=end,
            request_parameters={"interval": "1d", "events": "div,splits"},
        ),
        request_url_sha256=sha256_url(SAFE_URL),
    )


def _context(stage="T1C_RAW_ACQUISITION", coordinate=0):
    block = "T1C" if stage.startswith("T1C") else "T2"
    readiness = "READINESS" in stage
    start, end = ("2025-12-01", "2025-12-08") if readiness else ("2020-01-01", "2020-01-08")
    return V8DRequestContext(
        logical_stage=stage,
        logical_block=block,
        logical_coordinate=coordinate,
        window_start=start,
        window_end_exclusive=end,
        request_fingerprint=make_request_fingerprint(
            logical_stage=stage, logical_block=block, logical_coordinate=coordinate,
            window_start=start, window_end_exclusive=end,
        ),
        request_url_sha256=sha256_url(SAFE_URL),
        sentinel_indices=(0, 149, 299) if readiness else None,
    )


def _single_attempt(tmp_path, fn, *, stage="T1C_RAW_ACQUISITION"):
    store = DurableV8DAuditStore(tmp_path)
    dossier_id = store.new_id()
    return attempt_with_frozen_retry(
        fn,
        store=store,
        dossier_id=dossier_id,
        context=_context(stage),
        reviewed_implementation_commit=IMPLEMENTATION_SHA,
        sleep_fn=lambda _seconds: None,
    ), store, dossier_id


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_every_frozen_retryable_http_status_is_retried_and_audited(tmp_path, status):
    calls = []
    error = urllib.error.HTTPError(SAFE_URL, status, "synthetic message", {}, None)

    def request():
        calls.append(1)
        raise error

    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, request)
    assert len(calls) == 3
    dossier = next(tmp_path.glob("dossier-*.json"))
    checked = v8d_audit.verify_dossier(dossier, expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)
    assert [record["classification"] for record in checked["attempts"]] == [f"HTTP_{status}"] * 3
    assert [record["retryable"] for record in checked["attempts"]] == [True, True, True]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_every_frozen_nonretryable_http_status_stops_after_one_attempt(tmp_path, status):
    calls = []
    error = urllib.error.HTTPError(SAFE_URL, status, "synthetic message", {}, None)

    def request():
        calls.append(1)
        raise error

    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, request)
    assert len(calls) == 1
    checked = v8d_audit.verify_dossier(next(tmp_path.glob("dossier-*.json")), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)
    assert checked["attempts"][0]["classification"] == f"HTTP_{status}"
    assert checked["attempts"][0]["retryable"] is False


def test_success_without_retry_persists_before_return_and_verifies(tmp_path):
    result, store, dossier_id = _single_attempt(tmp_path, lambda: {"synthetic": "success"})
    value, audit = result
    assert value == {"synthetic": "success"}
    assert audit["attempts"] == 1
    checked = v8d_audit.verify_dossier(store._path(dossier_id), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)
    assert checked["attempts"][0]["terminal_state"] == "SUCCESS"


def test_429_then_success_sleeps_only_after_durable_failure(tmp_path):
    calls = []
    sleeps = []

    def request():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(SAFE_URL, 429, "synthetic", {}, None)
        return "ok"

    store = DurableV8DAuditStore(tmp_path)
    dossier_id = store.new_id()
    value, audit = attempt_with_frozen_retry(
        request, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=sleeps.append,
    )
    assert value == "ok" and audit["attempts"] == 2
    assert sleeps == [float(BACKOFF_SECONDS[0])]
    checked = v8d_audit.verify_dossier(store._path(dossier_id), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)
    assert [item["classification"] for item in checked["attempts"]] == ["HTTP_429", "SUCCESS"]


@pytest.mark.parametrize(
    "error,expected",
    [
        (TimeoutError("timeout text"), "NETWORK_TIMEOUT"),
        (socket.timeout("socket text"), "NETWORK_TIMEOUT"),
        (urllib.error.URLError(TimeoutError("timeout text")), "NETWORK_TIMEOUT"),
        (ConnectionResetError(errno.ECONNRESET, "reset text"), "CONNECTION_RESET"),
        (urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary dns text")), "TEMPORARY_DNS_FAILURE"),
        (socket.gaierror(getattr(socket, "EAI_NONAME", -2), "permanent dns text"), "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"),
        (ValueError("unknown private message"), "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"),
    ],
)
def test_frozen_runtime_exception_classification_uses_no_message_heuristics(error, expected):
    assert classify_transport_exception(error) == (expected, expected in {
        "NETWORK_TIMEOUT", "CONNECTION_RESET", "TEMPORARY_DNS_FAILURE",
    })


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("PAYLOAD_JSON_INVALID", "PARSER_SCHEMA_FAILURE"),
        ("TIMESTAMP_MISSING", "PARSER_SCHEMA_FAILURE"),
        ("CHART_RESULT_INVALID", "PARSER_SCHEMA_FAILURE"),
        ("SYMBOL_MISMATCH", "SYMBOL_MISMATCH"),
        ("RESPONSE_HOST_MISMATCH", "RESPONSE_HOST_MISMATCH"),
        ("HTTP_STATUS_429", "HTTP_429"),
        ("HTTP_STATUS_400", "HTTP_400"),
    ],
)
def test_canonical_v7_reason_matrix_is_exact(reason, expected):
    error = V7YahooCollectorBlocked(reason)
    assert classify_transport_exception(error)[0] == expected


def test_v7_request_path_uses_the_durable_v8d_wrapper_with_a_fake_opener(tmp_path):
    opener_calls = []

    def fake_opener(_request):
        opener_calls.append(1)
        raise urllib.error.HTTPError(SAFE_URL, 400, "synthetic", {}, None)

    def factory(coordinate):
        return build_yahoo_request_plan(
            logical_stage="T1C_RAW_ACQUISITION", logical_block="T1C", logical_coordinate=coordinate,
            ticker="SYNTHETIC_ONLY", request_start="2020-01-01", request_end_exclusive="2020-01-08",
            opener=fake_opener,
        )

    result = acquisition.execute_raw_acquisition_transport(
        stage="T1C_RAW_ACQUISITION", request_factory=factory, audit_root=tmp_path,
        reviewed_implementation_commit=IMPLEMENTATION_SHA, request_start="2020-01-01",
        request_end_exclusive="2020-01-08", request_count=1, sleep_fn=lambda _seconds: None,
    )
    assert opener_calls == [1]
    dossier = v8d_audit.verify_dossier(result["dossier_paths"][0], expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)
    assert dossier["attempts"][0]["classification"] == "HTTP_400"


def test_named_condition_evidence_covers_quality_redirect_and_parser_failure():
    redirect = V8DNamedFailure("UNTRUSTED_REDIRECT", evidence=origin_guard_evidence("https://evil.invalid", context="REDIRECT_TARGET"))
    host = V8DNamedFailure("RESPONSE_HOST_MISMATCH", evidence=origin_guard_evidence(123, context="INITIAL_OR_FINAL_RESPONSE"))
    quality = V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
        "nonempty_timestamp": True, "valid_price_row_count": 0, "trading_date_fields_valid": False,
    })
    assert classify_transport_exception(redirect) == ("UNTRUSTED_REDIRECT", False)
    assert classify_transport_exception(host) == ("RESPONSE_HOST_MISMATCH", False)
    assert classify_transport_exception(quality) == ("DATA_QUALITY_GATE_FAILURE", False)


@pytest.mark.parametrize(
    "value,context,expected",
    [
        ("http://query1.finance.yahoo.com/x", "INITIAL_OR_FINAL_RESPONSE", "RESPONSE_HOST_MISMATCH"),
        ("https://evil.invalid/x", "INITIAL_OR_FINAL_RESPONSE", "RESPONSE_HOST_MISMATCH"),
        ("https://user:pass@query1.finance.yahoo.com/x", "INITIAL_OR_FINAL_RESPONSE", "RESPONSE_HOST_MISMATCH"),
        ("https://query1.finance.yahoo.com:8443/x", "INITIAL_OR_FINAL_RESPONSE", "RESPONSE_HOST_MISMATCH"),
        (123, "INITIAL_OR_FINAL_RESPONSE", "RESPONSE_HOST_MISMATCH"),
        ("https://[invalid", "INITIAL_OR_FINAL_RESPONSE", "RESPONSE_HOST_MISMATCH"),
        ("http://evil.invalid/x", "REDIRECT_TARGET", "UNTRUSTED_REDIRECT"),
    ],
)
def test_origin_guard_frozen_failure_labels(value, context, expected):
    evidence = origin_guard_evidence(value, context=context)
    with pytest.raises(V8DNamedFailure) as excinfo:
        raise V8DNamedFailure(expected, evidence=evidence)
    assert excinfo.value.condition == expected
    if not evidence["input_is_string"] or not evidence["origin_parse_success"]:
        assert evidence["scheme_https"] is False
        assert evidence["hostname_matches_expected"] is False
        assert evidence["credentials_absent"] is False
        assert evidence["port_allowed"] is False


def test_readiness_end_to_end_success_and_independent_aggregate_verification(tmp_path):
    calls = []

    def factory(coordinate):
        def request():
            calls.append(coordinate)
            return {"valid_price_rows": 1}
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request)

    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS", request_factory=factory, audit_root=tmp_path,
        reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=lambda _seconds: None,
    )
    assert result["aggregate"]["result"] == "PASS"
    assert calls == [0, 149, 299]
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"],
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
        expected_stage="T1C_TRANSPORT_READINESS",
    )
    assert checked["sentinel_count"] == 3 and checked["sentinel_pass_count"] == 3


def test_acquisition_end_to_end_block_retains_all_terminal_evidence(tmp_path):
    reasons = ["EMPTY_TICKER", "PAYLOAD_JSON_INVALID", "SYMBOL_MISMATCH"]

    def factory(coordinate):
        return _plan("T2_RAW_ACQUISITION", coordinate, _raise(V7YahooCollectorBlocked(reasons[coordinate])))

    result = acquisition.execute_raw_acquisition_transport(
        stage="T2_RAW_ACQUISITION", request_factory=factory, audit_root=tmp_path,
        reviewed_implementation_commit=IMPLEMENTATION_SHA, request_start="2020-01-01",
        request_end_exclusive="2020-01-08", request_count=3, sleep_fn=lambda _seconds: None,
    )
    assert result["aggregate"]["result"] == "BLOCK"
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"],
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
        expected_stage="T2_RAW_ACQUISITION",
    )
    assert checked["terminal_classification_histogram"] == {
        "PARSER_SCHEMA_FAILURE": 2, "SYMBOL_MISMATCH": 1,
    }


def test_retry_exhaustion_has_three_attempts_and_two_backoffs(tmp_path):
    calls, sleeps = [], []

    def request():
        calls.append(1)
        raise TimeoutError("not persisted")

    with pytest.raises(TimeoutError):
        store = DurableV8DAuditStore(tmp_path)
        attempt_with_frozen_retry(
            request, store=store, dossier_id=store.new_id(), context=_context(),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=sleeps.append,
        )
    assert len(calls) == 3 and sleeps == [5.0, 30.0]


@pytest.mark.parametrize("return_success", [False, True])
def test_audit_write_failure_prevents_retry_or_success_return(tmp_path, monkeypatch, return_success):
    calls, sleeps = [], []
    store = DurableV8DAuditStore(tmp_path)

    def request():
        calls.append(1)
        if return_success:
            return "must not escape"
        raise TimeoutError("must be audited first")

    def fail_write(*_args, **_kwargs):
        raise V8DAuditPersistenceBlocked()

    monkeypatch.setattr(store, "write_attempt", fail_write)
    with pytest.raises(V8DAuditPersistenceBlocked):
        attempt_with_frozen_retry(
            request, store=store, dossier_id=store.new_id(), context=_context(),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=sleeps.append,
        )
    assert calls == [1]
    assert sleeps == []
    assert not list(tmp_path.glob("aggregate-*.json"))


def test_failed_attempt_audit_is_durable_before_next_request(tmp_path):
    order = []

    def request():
        order.append("request")
        if order.count("request") == 1:
            raise urllib.error.HTTPError(SAFE_URL, 429, "hidden", {}, None)
        order.append("success")
        return "ok"

    class OrderedStore(DurableV8DAuditStore):
        def write_attempt(self, *args, **kwargs):
            order.append("persist")
            return super().write_attempt(*args, **kwargs)

    store = OrderedStore(tmp_path)
    attempt_with_frozen_retry(
        request, store=store, dossier_id=store.new_id(), context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=lambda _seconds: order.append("sleep"),
    )
    assert order == ["request", "persist", "sleep", "request", "success", "persist"]


def test_audit_persistence_failure_stops_readiness_before_next_opener_and_publishing(tmp_path, monkeypatch):
    calls = []
    original = DurableV8DAuditStore.write_attempt

    def fail_on_second_coordinate(self, dossier_id, context, reviewed_commit, record):
        if context.logical_coordinate == 149:
            raise V8DAuditPersistenceBlocked()
        return original(self, dossier_id, context, reviewed_commit, record)

    monkeypatch.setattr(DurableV8DAuditStore, "write_attempt", fail_on_second_coordinate)

    def factory(coordinate):
        def request():
            calls.append(coordinate)
            return "ok"
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request)

    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness.execute_transport_readiness_probe(
            stage="T1C_TRANSPORT_READINESS", request_factory=factory, audit_root=tmp_path,
            reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=lambda _seconds: None,
        )
    assert excinfo.value.reason == "V8D_AUDIT_PERSISTENCE_FAILED"
    assert calls == [0, 149]
    assert not list(tmp_path.glob("aggregate-*.json"))


def _tamper_dossier(path: Path, mutate):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(canonical_json_bytes(value))


def _rehashed_dossier(path: Path, mutate):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value["audit_artifact_self_hash"] = canonical_sha256({key: item for key, item in value.items() if key != "audit_artifact_self_hash"})
    path.write_bytes(canonical_json_bytes(value))


def _rehashed_aggregate(result):
    aggregate_path = Path(result["aggregate_path"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    dossier_hashes = [json.loads(Path(path).read_text(encoding="utf-8"))["audit_artifact_self_hash"] for path in result["dossier_paths"]]
    aggregate["audit_artifact_self_hash"] = canonical_sha256(sorted(dossier_hashes))
    aggregate["aggregate_self_hash"] = canonical_sha256({key: item for key, item in aggregate.items() if key != "aggregate_self_hash"})
    aggregate_path.write_bytes(canonical_json_bytes(aggregate))


def _success_artifacts(tmp_path):
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path, reviewed_implementation_commit=IMPLEMENTATION_SHA,
        sleep_fn=lambda _seconds: None,
    )
    return result


def test_malformed_or_tampered_audit_and_missing_attempt_are_rejected(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _tamper_dossier(dossier, lambda value: value["attempts"][0].update({"classification": "HTTP_400"}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)

    retry_root = tmp_path / "retry"
    retry_root.mkdir()
    calls = []

    def retry_then_success():
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(SAFE_URL, 429, "x", {}, None)
        return "ok"

    store = DurableV8DAuditStore(retry_root)
    dossier_id = store.new_id()
    attempt_with_frozen_retry(
        retry_then_success, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=lambda _seconds: None,
    )
    path = store._path(dossier_id)
    _rehashed_dossier(path, lambda value: value["attempts"].pop(0))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(path, expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


@pytest.mark.parametrize("field,value", [
    ("frozen_design_commit", "b" * 40),
    ("reviewed_production_implementation_commit", "c" * 40),
    ("canonical_parser_classifier_commit", "d" * 40),
])
def test_provenance_binding_tampering_is_rejected(tmp_path, field, value):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda record: record.update({field: value}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


@pytest.mark.parametrize("field", ["request_fingerprint", "request_url_sha256"])
def test_request_binding_tampering_across_retries_is_rejected(tmp_path, field):
    calls = []

    def request():
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(SAFE_URL, 429, "x", {}, None)
        return "ok"

    store = DurableV8DAuditStore(tmp_path)
    dossier_id = store.new_id()
    attempt_with_frozen_retry(
        request, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, sleep_fn=lambda _seconds: None,
    )
    path = store._path(dossier_id)
    _rehashed_dossier(path, lambda value: value["attempts"][1].update({field: "e" * 64}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(path, expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


def test_aggregate_tampering_is_rejected_even_when_dossiers_are_intact(tmp_path):
    result = _success_artifacts(tmp_path)
    aggregate_path = Path(result["aggregate_path"])
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    value["result"] = "BLOCK"
    aggregate_path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate(aggregate_path, result["dossier_paths"], expected_stage="T1C_TRANSPORT_READINESS")


def test_wrong_readiness_sentinel_window_and_logical_stage_bindings_are_rejected(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"sentinel_indices": [0, 1, 2]}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, expected_stage="T1C_TRANSPORT_READINESS")


def test_readiness_dossier_coordinate_outside_exact_sentinel_set_is_rejected(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"logical_coordinate": 1}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, expected_stage="T1C_TRANSPORT_READINESS")


def test_readiness_aggregate_rederives_exact_coordinate_set_from_dossiers(tmp_path):
    result = _success_artifacts(tmp_path)
    _rehashed_dossier(result["dossier_paths"][1], lambda value: value.update({"logical_coordinate": 0}))
    _rehashed_aggregate(result)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate(
            result["aggregate_path"], result["dossier_paths"], expected_stage="T1C_TRANSPORT_READINESS"
        )


def test_readiness_aggregate_accepts_exact_coordinate_set_in_any_dossier_order(tmp_path):
    result = _success_artifacts(tmp_path)
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], list(reversed(result["dossier_paths"])),
        expected_stage="T1C_TRANSPORT_READINESS",
    )
    assert checked["result"] == "PASS"


def test_public_aggregate_and_private_dossier_never_store_raw_url_or_exception_message(tmp_path):
    secret = "PRIVATE_EXCEPTION_MESSAGE_NOT_ALLOWED"

    def request():
        raise ValueError(secret)

    result = acquisition.execute_raw_acquisition_transport(
        stage="T1C_RAW_ACQUISITION",
        request_factory=lambda coordinate: _plan("T1C_RAW_ACQUISITION", coordinate, request, start="2020-01-01", end="2020-01-08"),
        audit_root=tmp_path, reviewed_implementation_commit=IMPLEMENTATION_SHA,
        request_start="2020-01-01", request_end_exclusive="2020-01-08", request_count=1,
        sleep_fn=lambda _seconds: None,
    )
    aggregate_raw = Path(result["aggregate_path"]).read_bytes()
    dossier_raw = Path(result["dossier_paths"][0]).read_bytes()
    assert secret.encode() not in aggregate_raw and secret.encode() not in dossier_raw
    assert SAFE_URL.encode() not in aggregate_raw and SAFE_URL.encode() not in dossier_raw
    assert b"request_fingerprint" not in aggregate_raw


def test_forged_concrete_metadata_is_rejected(tmp_path):
    store = DurableV8DAuditStore(tmp_path)
    dossier_id = store.new_id()
    with pytest.raises(TimeoutError):
        attempt_with_frozen_retry(
            lambda: (_ for _ in ()).throw(TimeoutError("x")), store=store, dossier_id=dossier_id,
            context=_context(), reviewed_implementation_commit=IMPLEMENTATION_SHA,
            sleep_fn=lambda _seconds: None,
        )
    path = store._path(dossier_id)
    _rehashed_dossier(path, lambda value: value["attempts"][0].update({"concrete_exception_type": "HTTPError"}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(path, expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)
