from __future__ import annotations

import errno
import json
import socket
import subprocess
import urllib.error
from pathlib import Path

import pytest

from src import v8d_audit, v8d_git_provenance, v8d_historical_acquisition as acquisition
from src import v8d_production_provenance, v8d_readiness as readiness
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


# ===========================================================================
# V8D_PROD_HIGH_1A_REVIEWED_IMPLEMENTATION_BINDING
#
# src.v8d_git_provenance / src.v8d_production_provenance: fail-closed V8D
# Git provenance and future independent-production-review binding. A
# caller-supplied arbitrary 40-hex SHA must never, by itself, be sufficient
# evidence that a V8D production implementation was independently
# reviewed -- the real V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json does not
# exist in this repository yet, so every positive test below uses a
# synthetic temporary Git repository. Human-gate receipt binding is
# explicitly out of scope for this subtask.
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def _git_config_commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", message],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def _repo_with_raw_file(tmp_path: Path, name: str, relative_path: str, raw_bytes: bytes) -> tuple[Path, str]:
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    head = _git_config_commit(repo, "synthetic")
    return repo, head


def _valid_review_json(reviewed_commit: str) -> bytes:
    return json.dumps(
        {
            "schema_version": v8d_production_provenance.IMPLEMENTATION_REVIEW_SCHEMA_VERSION,
            "study": v8d_production_provenance.STUDY_NAME,
            "artifact_role": v8d_production_provenance.IMPLEMENTATION_REVIEW_ARTIFACT_ROLE,
            "reviewed_implementation_git_commit": reviewed_commit,
            "review_result": "PASS",
            "approval_status": "APPROVED",
        }
    ).encode()


def _build_bound_file_repo(tmp_path: Path, *, mutate_file: str | None, reviewed_commit_override: str | None = None):
    """A synthetic two-commit repository: commit 1 is the "reviewed"
    implementation state (every BOUND_PRODUCTION_FILES path present);
    commit 2 is the current "verified HEAD" state, carrying the review
    artifact plus either an identical or a mutated bound file."""
    repo = tmp_path / "bound_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    base_content = {
        path: ("# " + path + " v1\n").encode() for path in v8d_production_provenance.BOUND_PRODUCTION_FILES
    }
    for relative_path, content in base_content.items():
        file_path = repo / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
    reviewed_commit = _git_config_commit(repo, "reviewed implementation")

    if mutate_file is not None:
        (repo / mutate_file).write_bytes(base_content[mutate_file] + b"# mutated\n")
    else:
        (repo / "UNRELATED_DOC.md").write_text("docs-only change, no bound blob drift")

    review_path = repo / v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH
    review_path.write_bytes(_valid_review_json(reviewed_commit_override or reviewed_commit))
    head_commit = _git_config_commit(repo, "current verified HEAD state")
    return repo, reviewed_commit, head_commit


# --- Frozen object binding --------------------------------------------------


def test_v8d_frozen_object_constants_match_task():
    assert v8d_production_provenance.EXPECTED_V8D_FROZEN_DESIGN_COMMIT == "eda657cde2383718d986c4c4bfaae794784fe04d"
    assert (
        v8d_production_provenance.EXPECTED_V8D_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT
        == "9577a88c7bf46483b941aec3301c6064d9734c1f"
    )
    assert v8d_production_provenance.EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB == "67e3e1ab1e252b5c8f7583eb0605ec0333e487f6"


def test_verify_frozen_design_object_passes_against_real_repository():
    v8d_production_provenance.verify_frozen_design_object(REPO_ROOT)


def test_verify_frozen_design_object_blocks_on_blob_mismatch(monkeypatch):
    # Required test 9: frozen design blob mismatch -> BLOCK.
    monkeypatch.setattr(v8d_production_provenance, "EXPECTED_V8D_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT", "0" * 40)
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_frozen_design_object(REPO_ROOT)
    assert excinfo.value.reason == "V8D_FROZEN_DESIGN_OBJECT_MUTATED"


def test_verify_design_freeze_approval_blob_passes_against_real_head():
    blob = v8d_production_provenance.verify_design_freeze_approval_blob(REPO_ROOT, _real_head())
    assert blob == v8d_production_provenance.EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB


def test_verify_design_freeze_approval_blob_blocks_on_mismatch(monkeypatch):
    # Required test 10: freeze approval blob mismatch -> BLOCK.
    monkeypatch.setattr(v8d_production_provenance, "EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB", "0" * 40)
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_design_freeze_approval_blob(REPO_ROOT, _real_head())
    assert excinfo.value.reason == "V8D_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED"


def test_verify_design_freeze_approval_blob_missing_file_fails_closed(tmp_path):
    repo = tmp_path / "no_approval"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "x.txt").write_text("x")
    head = _git_config_commit(repo, "init")
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_design_freeze_approval_blob(repo, head)
    assert excinfo.value.reason == "V8D_DESIGN_FREEZE_APPROVAL_MISSING"


def test_bound_production_files_all_exist_at_head():
    """Every file this module binds review to must actually exist -- a
    typo'd path would silently make ``verify_reviewed_implementation_
    binding`` vacuously trivial for that file."""
    for path in v8d_production_provenance.BOUND_PRODUCTION_FILES:
        assert (REPO_ROOT / path).is_file(), path


# --- Required test 1: missing review artifact -> BLOCK ----------------------


def test_verify_reviewed_implementation_binding_missing_fails_closed():
    """The real V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json does not exist in
    this repository -- this must fail closed today."""
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(REPO_ROOT, _real_head())
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


# --- Required tests 2-8: malformed/duplicate/extra/missing/wrong-value ------


def test_verify_reviewed_implementation_binding_malformed_json_blocks(tmp_path):
    repo, head = _repo_with_raw_file(
        tmp_path, "malformed", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, b"{not valid json"
    )
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON"


def test_verify_reviewed_implementation_binding_duplicate_key_blocks(tmp_path):
    raw = (
        b'{"schema_version": "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_V1", '
        b'"schema_version": "DUPLICATE", '
        b'"study": "V8D_HISTORICAL_RESEARCH", '
        b'"artifact_role": "PRODUCTION_IMPLEMENTATION_REVIEW", '
        b'"reviewed_implementation_git_commit": "' + b"a" * 40 + b'", '
        b'"review_result": "PASS", "approval_status": "APPROVED"}'
    )
    repo, head = _repo_with_raw_file(tmp_path, "dup_key", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, raw)
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_DUPLICATE_KEY"


def test_verify_reviewed_implementation_binding_extra_field_blocks(tmp_path):
    payload = json.loads(_valid_review_json("a" * 40))
    payload["unexpected_extra_field"] = "x"
    repo, head = _repo_with_raw_file(
        tmp_path, "extra_field", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode()
    )
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID"


@pytest.mark.parametrize("missing_field", v8d_production_provenance.IMPLEMENTATION_REVIEW_FIELDS)
def test_verify_reviewed_implementation_binding_missing_field_blocks(tmp_path, missing_field):
    payload = json.loads(_valid_review_json("a" * 40))
    del payload[missing_field]
    repo, head = _repo_with_raw_file(
        tmp_path, "missing_field_" + missing_field, v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH,
        json.dumps(payload).encode(),
    )
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID"


def test_verify_reviewed_implementation_binding_review_result_not_pass_blocks(tmp_path):
    payload = json.loads(_valid_review_json("a" * 40))
    payload["review_result"] = "FAIL"
    repo, head = _repo_with_raw_file(
        tmp_path, "not_pass", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode()
    )
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS"


def test_verify_reviewed_implementation_binding_approval_status_not_approved_blocks(tmp_path):
    payload = json.loads(_valid_review_json("a" * 40))
    payload["approval_status"] = "PENDING"
    repo, head = _repo_with_raw_file(
        tmp_path, "not_approved", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode()
    )
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED"


def test_verify_reviewed_implementation_binding_invalid_commit_blocks(tmp_path):
    payload = json.loads(_valid_review_json("a" * 40))
    payload["reviewed_implementation_git_commit"] = "not-a-valid-sha"
    repo, head = _repo_with_raw_file(
        tmp_path, "invalid_commit", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode()
    )
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_COMMIT_INVALID"


# --- Required test 11: bound file blob drift -> BLOCK ------------------------


def test_verify_reviewed_implementation_binding_bound_file_drift_blocks(tmp_path):
    mutated_path = v8d_production_provenance.BOUND_PRODUCTION_FILES[0]
    repo, reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=mutated_path)
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head_commit)
    assert excinfo.value.reason == "V8D_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:" + mutated_path


# --- Required test 12: all bound blobs identical + valid artifact -> PASS ----


def test_verify_reviewed_implementation_binding_passes_when_blobs_identical(tmp_path):
    repo, reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=None)
    result = v8d_production_provenance.verify_reviewed_implementation_binding(repo, head_commit)
    assert result["reviewed_implementation_git_commit"] == reviewed_commit
    assert result["verified_head"] == head_commit
    assert result["bound_files_verified"] == len(v8d_production_provenance.BOUND_PRODUCTION_FILES)


# --- Required test 13: an arbitrary caller-provided SHA is not authority ----


def test_arbitrary_sha_without_committed_review_artifact_cannot_substitute(tmp_path):
    """The API accepts no caller-supplied "reviewed implementation commit"
    parameter at all: only `repository_root` and `verified_head`. Naming an
    arbitrary, syntactically valid 40-hex SHA inside a forged review
    artifact -- one that was never actually committed as that reviewed
    state -- still BLOCKs, because the bound files can't be resolved at a
    commit that does not exist in this repository's history."""
    import inspect

    signature = inspect.signature(v8d_production_provenance.verify_reviewed_implementation_binding)
    assert list(signature.parameters) == ["repository_root", "verified_head"]

    arbitrary_unrelated_sha = "b" * 40
    repo, reviewed_commit, head_commit = _build_bound_file_repo(
        tmp_path, mutate_file=None, reviewed_commit_override=arbitrary_unrelated_sha
    )
    assert arbitrary_unrelated_sha != reviewed_commit
    with pytest.raises(v8d_production_provenance.V8DProductionProvenanceBlocked) as excinfo:
        v8d_production_provenance.verify_reviewed_implementation_binding(repo, head_commit)
    assert excinfo.value.reason.startswith("V8D_BOUND_FILE_MISSING_AT_REVIEWED_COMMIT:")


# --- Required tests 14-16: v8d_git_provenance branch/origin/HEAD binding ----


def _init_bogus_git_repo(root: Path, *, files: dict[str, bytes], origin_url: str | None) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    if origin_url is not None:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin_url], check=True)
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return _git_config_commit(root, "bogus")


CANONICAL_ORIGIN_URL = "https://github.com/ta1k1-arakawa/stock-analyzer.git"


def test_v8d_production_branch_is_its_own_branch_not_v8c_or_v8b():
    assert v8d_git_provenance.PRODUCTION_BRANCH == "v8d-transport-audit-design"
    assert v8d_git_provenance.PRODUCTION_BRANCH not in (
        "v8c-transport-resilience-implementation",
        "v8b-allocation-authority-acquisition-implementation",
    )


def test_wrong_repository_origin_blocks(tmp_path):
    # Required test 14: wrong repository/origin identity -> BLOCK.
    bogus = tmp_path / "wrong_repo"
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus, files={"README.md": b"hello"}, origin_url="https://github.com/someone-else/unrelated-repo.git"
    )
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + v8d_git_provenance.PRODUCTION_BRANCH, commit],
        check=True,
    )
    with pytest.raises(v8d_git_provenance.V8DGitProvenanceBlocked) as excinfo:
        v8d_git_provenance.resolve_verified_v8d_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH"


def test_dirty_worktree_blocks(tmp_path):
    # Required test 15: dirty production worktree -> BLOCK.
    bogus = tmp_path / "dirty"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url=CANONICAL_ORIGIN_URL)
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + v8d_git_provenance.PRODUCTION_BRANCH, commit],
        check=True,
    )
    (bogus / "dirty.txt").write_text("uncommitted")
    with pytest.raises(v8d_git_provenance.V8DGitProvenanceBlocked) as excinfo:
        v8d_git_provenance.resolve_verified_v8d_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"


def test_head_not_equal_to_origin_ref_blocks(tmp_path):
    # Required test 16: HEAD != origin/v8d-transport-audit-design -> BLOCK.
    bogus = tmp_path / "diverged"
    bogus.mkdir()
    commit1 = _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url=CANONICAL_ORIGIN_URL)
    (bogus / "second.txt").write_text("second commit")
    second_commit = _git_config_commit(bogus, "second")
    assert second_commit != commit1
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + v8d_git_provenance.PRODUCTION_BRANCH, commit1],
        check=True,
    )
    with pytest.raises(v8d_git_provenance.V8DGitProvenanceBlocked) as excinfo:
        v8d_git_provenance.resolve_verified_v8d_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_HEAD_NOT_ORIGIN"


def test_canonical_intended_origin_reaches_and_passes_full_resolution(tmp_path):
    bogus = tmp_path / "canonical_pass"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url=CANONICAL_ORIGIN_URL)
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + v8d_git_provenance.PRODUCTION_BRANCH, commit],
        check=True,
    )
    assert v8d_git_provenance.resolve_verified_v8d_production_git_commit(bogus) == commit


def test_v8d_git_provenance_module_never_invokes_git_fetch(monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy_run(args, **kwargs):
        calls.append(list(args))
        return real_run(args, **kwargs)

    monkeypatch.setattr(v8d_git_provenance.subprocess, "run", spy_run)
    try:
        v8d_git_provenance.resolve_verified_v8d_production_git_commit(REPO_ROOT)
    except v8d_git_provenance.V8DGitProvenanceBlocked:
        pass
    assert not any("fetch" in call for call in calls)
