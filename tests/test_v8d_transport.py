from __future__ import annotations

import errno
import hashlib
import inspect
import json
import socket
import subprocess
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8d_audit, v8d_authority_bridge, v8d_git_provenance, v8d_historical_acquisition as acquisition
from src import v8d_human_gate_consumption as gate_consumption
from src import (
    v8d_production_provenance,
    v8d_readiness as readiness,
    v8d_readiness_audit_verification as readiness_audit_verification,
)
from src import v8_partition
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
    require_nonempty_quality,
    sha256_url,
)


IMPLEMENTATION_SHA = "a" * 40
SAFE_URL = "https://query1.finance.yahoo.com/synthetic"
AUTH_IDENTITY = "synthetic-authorization-identity"


def _raise(error: BaseException):
    def call():
        raise error
    return call


def _fixed_clock():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _gate_root(tmp_path: Path) -> Path:
    return tmp_path / "gate-state"


def _consume_gate(root: Path, *, stage: str, reviewed_commit: str = IMPLEMENTATION_SHA,
                  auth_identity: str = AUTH_IDENTITY,
                  frozen_design_commit: str = FROZEN_DESIGN_COMMIT) -> dict[str, str]:
    """Durably consume the exact gate for ``stage`` under ``root`` (a
    per-test synthetic gate-receipt state root -- never production state)
    and return the plain 4-field mapping shape `execute_v8d_stage`/
    `attempt_with_frozen_retry` require."""
    binding = gate_consumption.consume_gate_and_bind(
        root, logical_stage=stage, v8d_frozen_design_commit=frozen_design_commit,
        reviewed_production_implementation_commit=reviewed_commit,
        raw_authorization_identity=auth_identity, clock=_fixed_clock,
    )
    return {
        "human_gate": binding.human_gate,
        "gate_receipt_key_sha256": binding.gate_receipt_key_sha256,
        "gate_receipt_bytes_sha256": binding.gate_receipt_bytes_sha256,
        "authorization_identity_sha256": binding.authorization_identity_sha256,
    }


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
    gate_binding = _consume_gate(_gate_root(tmp_path), stage=stage)
    result = attempt_with_frozen_retry(
        fn,
        store=store,
        dossier_id=dossier_id,
        context=_context(stage),
        reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )
    return result, store, dossier_id


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
    checked = v8d_audit.verify_dossier(
        dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA
    )
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
    checked = v8d_audit.verify_dossier(
        next(tmp_path.glob("dossier-*.json")), gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
    assert checked["attempts"][0]["classification"] == f"HTTP_{status}"
    assert checked["attempts"][0]["retryable"] is False


def test_success_without_retry_persists_before_return_and_verifies(tmp_path):
    result, store, dossier_id = _single_attempt(tmp_path, lambda: {"synthetic": "success"})
    value, audit = result
    assert value == {"synthetic": "success"}
    assert audit["attempts"] == 1
    checked = v8d_audit.verify_dossier(
        store._path(dossier_id), gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
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
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    value, audit = attempt_with_frozen_retry(
        request, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=sleeps.append,
    )
    assert value == "ok" and audit["attempts"] == 2
    assert sleeps == [float(BACKOFF_SECONDS[0])]
    checked = v8d_audit.verify_dossier(
        store._path(dossier_id), gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
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

    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    result = acquisition.execute_raw_acquisition_transport(
        stage="T1C_RAW_ACQUISITION", request_factory=factory, audit_root=tmp_path,
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, request_start="2020-01-01",
        request_end_exclusive="2020-01-08", request_count=1, sleep_fn=lambda _seconds: None,
    )
    assert opener_calls == [1]
    dossier = v8d_audit.verify_dossier(
        result["dossier_paths"][0], gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
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

    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_TRANSPORT_READINESS")
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS", request_factory=factory, audit_root=tmp_path,
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    assert result["aggregate"]["result"] == "PASS"
    assert calls == [0, 149, 299]
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
        expected_stage="T1C_TRANSPORT_READINESS",
    )
    assert checked["sentinel_count"] == 3 and checked["sentinel_pass_count"] == 3


def test_acquisition_end_to_end_block_retains_all_terminal_evidence(tmp_path):
    reasons = ["EMPTY_TICKER", "PAYLOAD_JSON_INVALID", "SYMBOL_MISMATCH"]

    def factory(coordinate):
        return _plan("T2_RAW_ACQUISITION", coordinate, _raise(V7YahooCollectorBlocked(reasons[coordinate])))

    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T2_RAW_ACQUISITION")
    result = acquisition.execute_raw_acquisition_transport(
        stage="T2_RAW_ACQUISITION", request_factory=factory, audit_root=tmp_path,
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, request_start="2020-01-01",
        request_end_exclusive="2020-01-08", request_count=3, sleep_fn=lambda _seconds: None,
    )
    assert result["aggregate"]["result"] == "BLOCK"
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
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
        gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
        attempt_with_frozen_retry(
            request, store=store, dossier_id=store.new_id(), context=_context(),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=sleeps.append,
        )
    assert len(calls) == 3 and sleeps == [5.0, 30.0]


@pytest.mark.parametrize("return_success", [False, True])
def test_audit_write_failure_prevents_retry_or_success_return(tmp_path, monkeypatch, return_success):
    calls, sleeps = [], []
    store = DurableV8DAuditStore(tmp_path)
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")

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
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=sleeps.append,
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
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    attempt_with_frozen_retry(
        request, store=store, dossier_id=store.new_id(), context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: order.append("sleep"),
    )
    assert order == ["request", "persist", "sleep", "request", "success", "persist"]


def test_audit_persistence_failure_stops_readiness_before_next_opener_and_publishing(tmp_path, monkeypatch):
    calls = []
    original = DurableV8DAuditStore.write_attempt

    def fail_on_second_coordinate(self, dossier_id, context, reviewed_commit, gate_binding, record):
        if context.logical_coordinate == 149:
            raise V8DAuditPersistenceBlocked()
        return original(self, dossier_id, context, reviewed_commit, gate_binding, record)

    monkeypatch.setattr(DurableV8DAuditStore, "write_attempt", fail_on_second_coordinate)

    def factory(coordinate):
        def request():
            calls.append(coordinate)
            return "ok"
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request)

    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_TRANSPORT_READINESS")
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness.execute_transport_readiness_probe(
            stage="T1C_TRANSPORT_READINESS", request_factory=factory, audit_root=tmp_path,
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
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


def _success_artifacts(tmp_path, *, stage="T1C_TRANSPORT_READINESS"):
    gate_binding = _consume_gate(_gate_root(tmp_path), stage=stage)
    result = readiness.execute_transport_readiness_probe(
        stage=stage,
        request_factory=lambda coordinate: _plan(stage, coordinate, lambda: "ok"),
        audit_root=tmp_path, reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )
    return result


def _success_readiness_receipt(tmp_path, *, logical_stage="T1C_TRANSPORT_READINESS"):
    gate_root = _gate_root(tmp_path)
    gate_binding = _consume_gate(gate_root, stage=logical_stage)
    audit_root = tmp_path / "readiness-audit"
    result = readiness.execute_transport_readiness_probe(
        stage=logical_stage,
        request_factory=lambda coordinate: _plan(logical_stage, coordinate, lambda: "ok"),
        audit_root=audit_root, reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    receipt_root = tmp_path / "receipt-state"
    receipt = readiness_audit_verification._write_synthetic_receipt_for_tests(
        logical_stage=logical_stage, aggregate_path=result["aggregate_path"],
        dossier_paths=result["dossier_paths"], audit_root=audit_root, receipt_root=receipt_root,
        aggregate_verifier=v8d_audit.verify_aggregate, dossier_verifier=v8d_audit.verify_dossier,
        gate_root=gate_root,
    )
    return {
        "result": result, "receipt": receipt, "gate_root": gate_root,
        "audit_root": audit_root, "receipt_root": receipt_root,
    }


def _patch_synthetic_production_reader(monkeypatch, artifacts, *, logical_stage):
    if logical_stage == "T1C_TRANSPORT_READINESS":
        receipt_path_name = "t1c-readiness-audit-verification.json"
        reader = readiness_audit_verification.require_t1c_readiness_audit_verification_pass
    else:
        receipt_path_name = "t2-readiness-audit-verification.json"
        reader = readiness_audit_verification.require_t2_readiness_audit_verification_pass
    monkeypatch.setattr(readiness_audit_verification, "CANONICAL_PRODUCTION_AUDIT_ROOT", artifacts["audit_root"])
    monkeypatch.setattr(readiness_audit_verification, "CANONICAL_CONSUMPTION_STATE_ROOT", artifacts["gate_root"])
    monkeypatch.setattr(
        readiness_audit_verification, "CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT", artifacts["receipt_root"]
    )
    monkeypatch.setattr(
        readiness_audit_verification,
        "T1C_RECEIPT_PATH",
        artifacts["receipt_root"] / "t1c-readiness-audit-verification.json",
    )
    monkeypatch.setattr(
        readiness_audit_verification,
        "T2_RECEIPT_PATH",
        artifacts["receipt_root"] / "t2-readiness-audit-verification.json",
    )
    monkeypatch.setattr(readiness_audit_verification, "verify_aggregate_production", v8d_audit.verify_aggregate)
    monkeypatch.setattr(readiness_audit_verification, "verify_dossier_production", v8d_audit.verify_dossier)
    return reader, artifacts["receipt_root"] / receipt_path_name


def _matching_production_execution_binding_fixture(*, binding_result="PASS", aggregate_result="PASS"):
    gate = {
        "gate_receipt_key_sha256": "1" * 64,
        "gate_receipt_bytes_sha256": "2" * 64,
        "authorization_identity_sha256": "3" * 64,
    }
    dossier_hashes = {0: "4" * 64, 149: "5" * 64, 299: "6" * 64}
    dossier_bindings = [
        {
            "filename": f"dossier-{coordinate}.json",
            "audit_artifact_self_hash": dossier_hashes[coordinate],
            "logical_coordinate": coordinate,
        }
        for coordinate in (0, 149, 299)
    ]
    aggregate = {
        "frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "reviewed_production_implementation_commit": IMPLEMENTATION_SHA,
        "sentinel_indices": [0, 149, 299],
        "window_start": "2025-12-01",
        "window_end_exclusive": "2025-12-08",
        "result": aggregate_result,
        "aggregate_self_hash": "a" * 64,
    }
    dossiers = [
        {
            "frozen_design_commit": FROZEN_DESIGN_COMMIT,
            "reviewed_production_implementation_commit": IMPLEMENTATION_SHA,
            "logical_stage": "T1C_TRANSPORT_READINESS",
            "window_start": "2025-12-01",
            "window_end_exclusive": "2025-12-08",
            "sentinel_indices": [0, 149, 299],
            "logical_coordinate": coordinate,
            "audit_artifact_self_hash": dossier_hashes[coordinate],
        }
        for coordinate in (0, 149, 299)
    ]
    binding = {
        "logical_stage": "T1C_TRANSPORT_READINESS",
        "aggregate_filename": "aggregate-test.json",
        "frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "reviewed_production_implementation_commit": IMPLEMENTATION_SHA,
        "sentinel_indices": [0, 149, 299],
        "window_start": "2025-12-01",
        "window_end_exclusive": "2025-12-08",
        "execution_result": binding_result,
        "aggregate_artifact_self_hash": "a" * 64,
        "dossier_bindings": dossier_bindings,
        **gate,
    }
    return binding, aggregate, dossiers, [item["filename"] for item in dossier_bindings], gate


def test_production_execution_binding_compares_execution_result_to_transport_result():
    binding, aggregate, dossiers, filenames, gate = _matching_production_execution_binding_fixture()
    readiness_audit_verification._require_matching_production_execution_binding(
        binding=binding, logical_stage="T1C_TRANSPORT_READINESS", aggregate=aggregate,
        aggregate_filename="aggregate-test.json", dossiers=dossiers,
        dossier_filenames=filenames, gate_binding=gate,
    )


@pytest.mark.parametrize("binding_result,aggregate_result", [("PASS", "BLOCK"), ("BLOCK", "PASS")])
def test_production_execution_binding_result_mismatch_blocks(binding_result, aggregate_result):
    binding, aggregate, dossiers, filenames, gate = _matching_production_execution_binding_fixture(
        binding_result=binding_result, aggregate_result=aggregate_result,
    )
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._require_matching_production_execution_binding(
            binding=binding, logical_stage="T1C_TRANSPORT_READINESS", aggregate=aggregate,
            aggregate_filename="aggregate-test.json", dossiers=dossiers,
            dossier_filenames=filenames, gate_binding=gate,
        )


def _publication_receipt_fixture():
    return readiness_audit_verification._build_receipt(
        verification_stage=readiness_audit_verification.T1C_VERIFICATION_STAGE,
        logical_stage="T1C_TRANSPORT_READINESS",
        aggregate_filename="aggregate-test.json",
        aggregate={
            "reviewed_production_implementation_commit": IMPLEMENTATION_SHA,
            "aggregate_self_hash": "a" * 64,
        },
        dossier_bindings=[
            {"filename": f"dossier-{coordinate}.json", "audit_artifact_self_hash": str(index) * 64}
            for index, coordinate in enumerate((0, 149, 299), start=1)
        ],
        gate_binding={
            "gate_receipt_key_sha256": "1" * 64,
            "gate_receipt_bytes_sha256": "2" * 64,
            "authorization_identity_sha256": "3" * 64,
        },
    )


def test_readiness_receipt_publication_absent_and_identical_existing_are_idempotent(tmp_path):
    receipt = _publication_receipt_fixture()
    destination = tmp_path / "t1c-readiness-audit-verification.json"
    assert readiness_audit_verification._persist_receipt(receipt, destination) == receipt
    assert readiness_audit_verification._persist_receipt(receipt, destination) == receipt
    assert readiness_audit_verification._validate_receipt(
        json.loads(destination.read_text(encoding="utf-8")),
        expected_verification_stage=readiness_audit_verification.T1C_VERIFICATION_STAGE,
    ) == receipt


def test_readiness_receipt_publication_conflicting_or_malformed_existing_blocks(tmp_path):
    receipt = _publication_receipt_fixture()
    destination = tmp_path / "t1c-readiness-audit-verification.json"
    conflicting = dict(receipt)
    conflicting["aggregate_filename"] = "other-aggregate.json"
    conflicting["receipt_self_hash"] = canonical_sha256(
        {key: value for key, value in conflicting.items() if key != "receipt_self_hash"}
    )
    destination.write_bytes(canonical_json_bytes(conflicting))
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._persist_receipt(receipt, destination)

    destination.write_bytes(b"not-json")
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._persist_receipt(receipt, destination)


def test_readiness_receipt_publication_symlink_blocks(tmp_path):
    receipt = _publication_receipt_fixture()
    destination = tmp_path / "t1c-readiness-audit-verification.json"
    target = tmp_path / "target.json"
    target.write_bytes(canonical_json_bytes(receipt))
    try:
        destination.symlink_to(target)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._persist_receipt(receipt, destination)


@pytest.mark.parametrize("winner_conflicts", [False, True])
def test_readiness_receipt_publication_race_rechecks_exclusive_winner(tmp_path, monkeypatch, winner_conflicts):
    receipt = _publication_receipt_fixture()
    destination = tmp_path / "t1c-readiness-audit-verification.json"
    winner = dict(receipt)
    if winner_conflicts:
        winner["aggregate_filename"] = "race-winner.json"
        winner["receipt_self_hash"] = canonical_sha256(
            {key: value for key, value in winner.items() if key != "receipt_self_hash"}
        )

    def race_link(_staging, actual_destination):
        Path(actual_destination).write_bytes(canonical_json_bytes(winner))
        raise FileExistsError

    monkeypatch.setattr(readiness_audit_verification.os, "link", race_link)
    if winner_conflicts:
        with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
            readiness_audit_verification._persist_receipt(receipt, destination)
    else:
        assert readiness_audit_verification._persist_receipt(receipt, destination) == receipt
    assert destination.read_bytes() == canonical_json_bytes(winner)
    assert not list(tmp_path.glob("*.staging-*"))


def test_readiness_receipt_publication_never_uses_replace_and_cleans_failed_stage(tmp_path, monkeypatch):
    receipt = _publication_receipt_fixture()
    destination = tmp_path / "t1c-readiness-audit-verification.json"

    def forbidden_replace(*_args, **_kwargs):
        raise AssertionError("receipt publication must not use os.replace")

    monkeypatch.setattr(readiness_audit_verification.os, "replace", forbidden_replace)
    assert readiness_audit_verification._persist_receipt(receipt, destination) == receipt

    destination.unlink()

    def failed_link(*_args, **_kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(readiness_audit_verification.os, "link", failed_link)
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._persist_receipt(receipt, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.staging-*"))


def test_canonical_readiness_receipts_are_stage_specific_and_privacy_safe(tmp_path):
    for logical_stage in ("T1C_TRANSPORT_READINESS", "T2_TRANSPORT_READINESS"):
        artifacts = _success_readiness_receipt(tmp_path / logical_stage, logical_stage=logical_stage)
        receipt = artifacts["receipt"]
        assert set(receipt) == set(readiness_audit_verification.RECEIPT_FIELDS)
        assert receipt["logical_stage"] == logical_stage
        receipt_bytes = (artifacts["receipt_root"] / f"{'t1c' if logical_stage.startswith('T1C') else 't2'}-readiness-audit-verification.json").read_bytes()
        assert SAFE_URL.encode() not in receipt_bytes
        assert str(tmp_path).encode() not in receipt_bytes
        assert b"request_fingerprint" not in receipt_bytes


def test_canonical_readiness_production_api_has_only_locators_or_no_arguments():
    for name in (
        "record_t1c_readiness_audit_verification",
        "record_t2_readiness_audit_verification",
    ):
        parameters = inspect.signature(getattr(readiness_audit_verification, name)).parameters
        assert set(parameters) == {"aggregate_path", "dossier_paths"}
        assert "request_factory" not in parameters
        assert "audit_root" not in parameters
    for name in (
        "require_t1c_readiness_audit_verification_pass",
        "require_t2_readiness_audit_verification_pass",
    ):
        assert inspect.signature(getattr(readiness_audit_verification, name)).parameters == {}


def test_synthetic_receipt_helper_cannot_consume_gate_and_reader_reverifies_audit(tmp_path, monkeypatch):
    artifacts = _success_readiness_receipt(tmp_path)
    reader, receipt_path = _patch_synthetic_production_reader(
        monkeypatch, artifacts, logical_stage="T1C_TRANSPORT_READINESS"
    )
    assert reader()["verification_result"] == "PASS"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["aggregate_artifact_self_hash"] = "f" * 64
    receipt["receipt_self_hash"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_self_hash"}
    )
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        reader()


def test_receipt_reader_rejects_self_dossier_gate_and_implementation_binding_tamper(tmp_path, monkeypatch):
    artifacts = _success_readiness_receipt(tmp_path)
    reader, receipt_path = _patch_synthetic_production_reader(
        monkeypatch, artifacts, logical_stage="T1C_TRANSPORT_READINESS"
    )
    original = receipt_path.read_bytes()
    for field, value in (
        ("reviewed_production_implementation_commit", "b" * 40),
        ("gate_receipt_key_sha256", "c" * 64),
    ):
        receipt = json.loads(original.decode("utf-8"))
        receipt[field] = value
        receipt["receipt_self_hash"] = canonical_sha256(
            {key: item for key, item in receipt.items() if key != "receipt_self_hash"}
        )
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
            reader()
    receipt_path.write_bytes(original)
    dossier = Path(artifacts["result"]["dossier_paths"][0])
    _rehashed_dossier(dossier, lambda value: value.update({"logical_stage": "T2_TRANSPORT_READINESS"}))
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        reader()


def test_receipt_validation_rejects_duplicate_json_and_sentinel_binding(tmp_path):
    artifacts = _success_readiness_receipt(tmp_path)
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._strict_json_object(
            b'{"schema_version":1,"schema_version":2}',
            invalid_reason="invalid", duplicate_reason="duplicate",
        )
    aggregate = json.loads(Path(artifacts["result"]["aggregate_path"]).read_text(encoding="utf-8"))
    aggregate["sentinel_pass_count"] = 2
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._require_readiness_pass(aggregate, "T1C_TRANSPORT_READINESS")


def test_t1c_receipt_never_satisfies_t2_reader(tmp_path, monkeypatch):
    artifacts = _success_readiness_receipt(tmp_path)
    _, t1_receipt_path = _patch_synthetic_production_reader(
        monkeypatch, artifacts, logical_stage="T1C_TRANSPORT_READINESS"
    )
    t2_receipt_path = artifacts["receipt_root"] / "t2-readiness-audit-verification.json"
    t2_receipt_path.write_bytes(t1_receipt_path.read_bytes())
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification.require_t2_readiness_audit_verification_pass()


def test_receipt_rejects_stage_mismatch_and_missing_or_extra_fields(tmp_path):
    artifacts = _success_readiness_receipt(tmp_path)
    receipt = dict(artifacts["receipt"])
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._validate_receipt(
            receipt, expected_verification_stage=readiness_audit_verification.T2_VERIFICATION_STAGE
        )
    for mutation in (
        lambda value: value.update({"extra": True}),
        lambda value: value.pop("study"),
    ):
        malformed = dict(receipt)
        mutation(malformed)
        with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
            readiness_audit_verification._validate_receipt(
                malformed, expected_verification_stage=readiness_audit_verification.T1C_VERIFICATION_STAGE
            )


def test_receipt_writer_blocks_aggregate_failure_sentinel_mismatch_and_wrong_dossiers(tmp_path):
    result = _success_artifacts(tmp_path)
    artifacts = {
        "result": result,
        "audit_root": tmp_path,
        "gate_root": _gate_root(tmp_path),
    }
    aggregate_path = Path(result["aggregate_path"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["result"] = "BLOCK"
    aggregate["aggregate_self_hash"] = canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "aggregate_self_hash"}
    )
    aggregate_path.write_bytes(canonical_json_bytes(aggregate))
    receipt_path = tmp_path / "receipts" / "t1c-readiness-audit-verification.json"
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._write_synthetic_receipt_for_tests(
            logical_stage="T1C_TRANSPORT_READINESS", aggregate_path=aggregate_path,
            dossier_paths=result["dossier_paths"], audit_root=artifacts["audit_root"],
            receipt_root=receipt_path.parent, aggregate_verifier=v8d_audit.verify_aggregate,
            dossier_verifier=v8d_audit.verify_dossier, gate_root=artifacts["gate_root"],
        )
    assert not receipt_path.exists()
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._write_synthetic_receipt_for_tests(
            logical_stage="T1C_TRANSPORT_READINESS", aggregate_path=aggregate_path,
            dossier_paths=result["dossier_paths"][:2], audit_root=artifacts["audit_root"],
            receipt_root=receipt_path.parent, aggregate_verifier=v8d_audit.verify_aggregate,
            dossier_verifier=v8d_audit.verify_dossier, gate_root=artifacts["gate_root"],
        )


def test_receipt_writer_rejects_outside_root_and_symlink_substitution(tmp_path):
    artifacts = _success_readiness_receipt(tmp_path)
    aggregate = Path(artifacts["result"]["aggregate_path"])
    outside = tmp_path / "outside.json"
    outside.write_bytes(aggregate.read_bytes())
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._write_synthetic_receipt_for_tests(
            logical_stage="T1C_TRANSPORT_READINESS", aggregate_path=outside,
            dossier_paths=artifacts["result"]["dossier_paths"], audit_root=artifacts["audit_root"],
            receipt_root=tmp_path / "outside-receipts", aggregate_verifier=v8d_audit.verify_aggregate,
            dossier_verifier=v8d_audit.verify_dossier, gate_root=artifacts["gate_root"],
        )
    link = artifacts["audit_root"] / "aggregate-link.json"
    try:
        link.symlink_to(aggregate)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(readiness_audit_verification.V8DReadinessAuditVerificationBlocked):
        readiness_audit_verification._write_synthetic_receipt_for_tests(
            logical_stage="T1C_TRANSPORT_READINESS", aggregate_path=link,
            dossier_paths=artifacts["result"]["dossier_paths"], audit_root=artifacts["audit_root"],
            receipt_root=tmp_path / "link-receipts", aggregate_verifier=v8d_audit.verify_aggregate,
            dossier_verifier=v8d_audit.verify_dossier, gate_root=artifacts["gate_root"],
        )


def test_malformed_or_tampered_audit_and_missing_attempt_are_rejected(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _tamper_dossier(dossier, lambda value: value["attempts"][0].update({"classification": "HTTP_400"}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)

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
    gate_binding = _consume_gate(_gate_root(retry_root), stage="T1C_RAW_ACQUISITION")
    attempt_with_frozen_retry(
        retry_then_success, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    path = store._path(dossier_id)
    _rehashed_dossier(path, lambda value: value["attempts"].pop(0))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(path, gate_receipt_state_root=_gate_root(retry_root), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


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
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


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
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    attempt_with_frozen_retry(
        request, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    path = store._path(dossier_id)
    _rehashed_dossier(path, lambda value: value["attempts"][1].update({field: "e" * 64}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(path, gate_receipt_state_root=_gate_root(tmp_path), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


def test_aggregate_tampering_is_rejected_even_when_dossiers_are_intact(tmp_path):
    result = _success_artifacts(tmp_path)
    aggregate_path = Path(result["aggregate_path"])
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    value["result"] = "BLOCK"
    aggregate_path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate(
            aggregate_path, result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
            expected_stage="T1C_TRANSPORT_READINESS",
        )


def test_wrong_readiness_sentinel_window_and_logical_stage_bindings_are_rejected(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"sentinel_indices": [0, 1, 2]}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_stage="T1C_TRANSPORT_READINESS")


def test_readiness_dossier_coordinate_outside_exact_sentinel_set_is_rejected(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"logical_coordinate": 1}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_stage="T1C_TRANSPORT_READINESS")


def test_readiness_aggregate_rederives_exact_coordinate_set_from_dossiers(tmp_path):
    result = _success_artifacts(tmp_path)
    _rehashed_dossier(result["dossier_paths"][1], lambda value: value.update({"logical_coordinate": 0}))
    _rehashed_aggregate(result)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate(
            result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
            expected_stage="T1C_TRANSPORT_READINESS",
        )


def test_readiness_aggregate_accepts_exact_coordinate_set_in_any_dossier_order(tmp_path):
    result = _success_artifacts(tmp_path)
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], list(reversed(result["dossier_paths"])), gate_receipt_state_root=_gate_root(tmp_path),
        expected_stage="T1C_TRANSPORT_READINESS",
    )
    assert checked["result"] == "PASS"


def test_public_aggregate_and_private_dossier_never_store_raw_url_or_exception_message(tmp_path):
    secret = "PRIVATE_EXCEPTION_MESSAGE_NOT_ALLOWED"

    def request():
        raise ValueError(secret)

    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    result = acquisition.execute_raw_acquisition_transport(
        stage="T1C_RAW_ACQUISITION",
        request_factory=lambda coordinate: _plan("T1C_RAW_ACQUISITION", coordinate, request, start="2020-01-01", end="2020-01-08"),
        audit_root=tmp_path, reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding,
        request_start="2020-01-01", request_end_exclusive="2020-01-08", request_count=1,
        sleep_fn=lambda _seconds: None,
    )
    aggregate_raw = Path(result["aggregate_path"]).read_bytes()
    dossier_raw = Path(result["dossier_paths"][0]).read_bytes()
    assert secret.encode() not in aggregate_raw and secret.encode() not in dossier_raw
    assert SAFE_URL.encode() not in aggregate_raw and SAFE_URL.encode() not in dossier_raw
    assert b"request_fingerprint" not in aggregate_raw
    assert AUTH_IDENTITY.encode() not in aggregate_raw and AUTH_IDENTITY.encode() not in dossier_raw


def test_forged_concrete_metadata_is_rejected(tmp_path):
    store = DurableV8DAuditStore(tmp_path)
    dossier_id = store.new_id()
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    with pytest.raises(TimeoutError):
        attempt_with_frozen_retry(
            lambda: (_ for _ in ()).throw(TimeoutError("x")), store=store, dossier_id=dossier_id,
            context=_context(), reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding,
            sleep_fn=lambda _seconds: None,
        )
    path = store._path(dossier_id)
    _rehashed_dossier(path, lambda value: value["attempts"][0].update({"concrete_exception_type": "HTTPError"}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier(path, gate_receipt_state_root=_gate_root(tmp_path), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA)


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


# ===========================================================================
# V8D_PROD_HIGH_1B_GATE_CONSUMPTION_RECEIPT_BINDING
#
# src.v8d_human_gate_consumption: durable, fail-closed, one-shot receipts
# for the four Yahoo-request-bearing V8D human gates. src.v8d_transport /
# src.v8d_audit: every private dossier now carries an exact safe gate
# binding, and the independent verifier never trusts a dossier's own
# claims -- it independently locates and re-validates the real durable
# receipt. src.v8d_readiness / src.v8d_historical_acquisition: the new
# production entrypoints derive authority only from HIGH-1A provenance
# plus real one-shot gate consumption -- never from a caller-supplied
# reviewed_implementation_commit. No real human gate is consumed anywhere
# in this test module; every gate-state root is a per-test tmp_path.
# ===========================================================================


def _valid_receipt_payload(*, stage="T1C_TRANSPORT_READINESS", reviewed_commit=IMPLEMENTATION_SHA,
                           frozen_design_commit=FROZEN_DESIGN_COMMIT, auth_hash="0" * 64):
    return {
        "schema_version": gate_consumption.SCHEMA_VERSION,
        "study": gate_consumption.STUDY_NAME,
        "repository": gate_consumption.REPOSITORY_IDENTITY,
        "gate": gate_consumption.STAGE_GATE[stage],
        "logical_stage": stage,
        "v8d_frozen_design_commit": frozen_design_commit,
        "reviewed_production_implementation_commit": reviewed_commit,
        "authorization_identity_sha256": auth_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": gate_consumption.CONSUMPTION_BOUNDARY,
        "consumed_at_utc": "2026-01-01T00:00:00Z",
    }


def _write_receipt(root: Path, *, stage="T1C_TRANSPORT_READINESS", payload=None) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    key = gate_consumption.compute_receipt_key(gate_consumption.STAGE_GATE[stage], FROZEN_DESIGN_COMMIT)
    body = payload if payload is not None else _valid_receipt_payload(stage=stage)
    path = root / (key + ".json")
    path.write_bytes(json.dumps(body).encode())
    return path, key


@pytest.mark.parametrize("timestamp", [
    "2026-01-01T00:00:00Z",
    "2026-01-01T00:00:00.123456Z",
])
def test_gate_receipt_accepts_canonical_utc_timestamps(tmp_path, timestamp):
    payload = _valid_receipt_payload()
    payload["consumed_at_utc"] = timestamp
    root = tmp_path / "gate-state"
    _path, key = _write_receipt(root, payload=payload)
    assert gate_consumption.read_gate_consumption_receipt(root, key)["consumed_at_utc"] == timestamp


def test_consume_gate_writer_timestamp_remains_canonical_and_readable(tmp_path):
    root = _gate_root(tmp_path)
    binding = _consume_gate(root, stage="T1C_TRANSPORT_READINESS")
    receipt = gate_consumption.read_gate_consumption_receipt(
        root, binding["gate_receipt_key_sha256"],
        expected_gate=gate_consumption.GATE_T1C_TRANSPORT_READINESS,
        expected_v8d_frozen_design_commit=FROZEN_DESIGN_COMMIT,
    )
    assert receipt["consumed_at_utc"] == "2026-01-01T00:00:00Z"


@pytest.mark.parametrize("timestamp", [
    "",
    "not-a-timestamp",
    "2026-01-01T00:00:00+00:00",
    "2026-01-01T00:00:00+01:00",
    "2026-01-01T00:00:00",
    "2026-01-01T00:00:00z",
    "2026-01-01 00:00:00Z",
    "2026-01-01T00:00:00.1Z",
    "2026-01-01T00:00:00.12345Z",
    "2026-01-01T00:00:00.1234567Z",
    "2026-02-29T00:00:00Z",
    " 2026-01-01T00:00:00Z",
    "2026-01-01T00:00:00Z ",
    "2026-01-01T00:00:00Ztrailing",
])
def test_gate_receipt_rejects_noncanonical_or_invalid_utc_timestamps(tmp_path, timestamp):
    payload = _valid_receipt_payload()
    payload["consumed_at_utc"] = timestamp
    root = tmp_path / "gate-state"
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID"


def test_tampered_gate_receipt_timestamp_blocks_through_normal_read_path(tmp_path):
    root = _gate_root(tmp_path)
    binding = _consume_gate(root, stage="T1C_TRANSPORT_READINESS")
    path = root / (binding["gate_receipt_key_sha256"] + ".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["consumed_at_utc"] = "2026-01-01T00:00:00+00:00"
    path.write_bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, binding["gate_receipt_key_sha256"])
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID"


# --- Required test 1: all four exact stage->gate mappings -------------------


def test_all_four_stage_gate_mappings_are_exact():
    assert gate_consumption.STAGE_GATE == {
        "T1C_TRANSPORT_READINESS": "T1C_TRANSPORT_READINESS_HUMAN_GATE",
        "T1C_RAW_ACQUISITION": "T1C_RAW_ACQUISITION_HUMAN_GATE",
        "T2_TRANSPORT_READINESS": "T2_TRANSPORT_READINESS_HUMAN_GATE",
        "T2_RAW_ACQUISITION": "T2_RAW_ACQUISITION_HUMAN_GATE",
    }
    assert len(gate_consumption.KNOWN_GATES) == 4 and len(set(gate_consumption.KNOWN_GATES)) == 4


# --- Required test 2: wrong stage/gate mapping -> BLOCK ----------------------


def test_wrong_stage_gate_mapping_blocks_at_transport_layer(tmp_path):
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_TRANSPORT_READINESS")
    store = DurableV8DAuditStore(tmp_path)
    with pytest.raises(V8DTransportBlocked) as excinfo:
        attempt_with_frozen_retry(
            lambda: "ok", store=store, dossier_id=store.new_id(), context=_context("T1C_RAW_ACQUISITION"),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _s: None,
        )
    assert excinfo.value.reason == "V8D_GATE_BINDING_STAGE_MISMATCH"


def test_wrong_stage_gate_mapping_in_receipt_blocks_independent_read(tmp_path):
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload(stage="T1C_TRANSPORT_READINESS")
    payload["gate"] = gate_consumption.GATE_T2_RAW_ACQUISITION
    path, key = _write_receipt(root, stage="T1C_TRANSPORT_READINESS", payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_STAGE_GATE_MISMATCH"


# --- Required test 3: empty/missing raw authorization identity -> BLOCK -----


def test_empty_or_missing_authorization_identity_blocks_before_request_readiness(tmp_path):
    for bad_identity in ("", None):
        with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
            readiness.execute_t1c_transport_readiness_production(
                human_authorization_identity=bad_identity,
                partition_manifest_path=tmp_path / "partition.json",
            )
        assert excinfo.value.reason == "V8D_READINESS_HUMAN_AUTHORIZATION_IDENTITY_REQUIRED"


def test_empty_or_missing_authorization_identity_blocks_before_request_acquisition(tmp_path):
    def request_factory(_coordinate):
        raise AssertionError("request_factory must not be invoked")

    for bad_identity in ("", None):
        with pytest.raises(acquisition.V8DAcquisitionBlocked) as excinfo:
            acquisition._execute_production_raw_acquisition(
                stage="T1C_RAW_ACQUISITION", human_authorization_identity=bad_identity,
                request_factory=request_factory, audit_root=tmp_path / "audit",
                request_start="2020-01-01", request_end_exclusive="2020-01-08", request_count=1,
                consumption_state_root=tmp_path / "gate-state",
            )
        assert excinfo.value.reason == "V8D_ACQUISITION_HUMAN_AUTHORIZATION_IDENTITY_REQUIRED"


# --- Required tests 4-5: raw identity never persisted; hash is correct ------


def test_raw_authorization_identity_never_appears_in_durable_artifacts(tmp_path):
    secret_identity = "SUPER-SECRET-RAW-AUTH-IDENTITY-NOT-A-HASH"
    gate_root = _gate_root(tmp_path)
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", auth_identity=secret_identity)
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path, reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )
    receipt_path = gate_root / (gate_binding["gate_receipt_key_sha256"] + ".json")
    receipt_raw = receipt_path.read_bytes()
    aggregate_raw = Path(result["aggregate_path"]).read_bytes()
    dossier_raw_all = b"".join(Path(p).read_bytes() for p in result["dossier_paths"])
    assert secret_identity.encode() not in receipt_raw
    assert secret_identity.encode() not in aggregate_raw
    assert secret_identity.encode() not in dossier_raw_all


def test_authorization_identity_sha256_is_correct(tmp_path):
    identity = "check-this-exact-identity"
    binding = gate_consumption.consume_gate_and_bind(
        _gate_root(tmp_path), logical_stage="T1C_TRANSPORT_READINESS", v8d_frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_production_implementation_commit=IMPLEMENTATION_SHA, raw_authorization_identity=identity,
        clock=_fixed_clock,
    )
    assert binding.authorization_identity_sha256 == hashlib.sha256(identity.encode("utf-8")).hexdigest()


# --- Required test 6: receipt exact schema ----------------------------------


def test_receipt_exact_schema_enforced():
    assert gate_consumption.RECEIPT_FIELDS == (
        "schema_version", "study", "repository", "gate", "logical_stage",
        "v8d_frozen_design_commit", "reviewed_production_implementation_commit",
        "authorization_identity_sha256", "consumed", "consumption_count",
        "consumption_boundary", "consumed_at_utc",
    )


# --- Required test 7: duplicate receipt JSON key -----------------------------


def test_duplicate_receipt_json_key_blocks(tmp_path):
    root = tmp_path / "gate-state"
    root.mkdir()
    key = gate_consumption.compute_receipt_key(gate_consumption.GATE_T1C_TRANSPORT_READINESS, FROZEN_DESIGN_COMMIT)
    raw = (
        b'{"schema_version": "' + gate_consumption.SCHEMA_VERSION.encode() + b'", '
        b'"schema_version": "DUPLICATE", "study": "x", "repository": "y"}'
    )
    (root / (key + ".json")).write_bytes(raw)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_DUPLICATE_KEY"


# --- Required test 8: missing receipt field ----------------------------------


@pytest.mark.parametrize("missing_field", gate_consumption.RECEIPT_FIELDS)
def test_missing_receipt_field_blocks(tmp_path, missing_field):
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    del payload[missing_field]
    _, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_SCHEMA_INVALID"


# --- Required test 9: extra receipt field ------------------------------------


def test_extra_receipt_field_blocks(tmp_path):
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["unexpected_extra_field"] = "x"
    _, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_SCHEMA_INVALID"


# --- Required test 10: consumed != true --------------------------------------


def test_consumed_flag_not_true_blocks(tmp_path):
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["consumed"] = False
    _, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_CONSUMED_FLAG_INVALID"


# --- Required test 11: consumption_count != 1 --------------------------------


def test_consumption_count_not_one_blocks(tmp_path):
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["consumption_count"] = 2
    _, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_CONSUMPTION_COUNT_INVALID"


# --- Required test 12: wrong consumption_boundary ----------------------------


def test_wrong_consumption_boundary_blocks(tmp_path):
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["consumption_boundary"] = "SOMETIME_LATER"
    _, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_CONSUMPTION_BOUNDARY_INVALID"


# --- Required test 13: wrong frozen design commit ----------------------------


def test_wrong_frozen_design_commit_in_receipt_blocks(tmp_path):
    root = tmp_path / "gate-state"
    binding = gate_consumption.consume_gate_and_bind(
        root, logical_stage="T1C_TRANSPORT_READINESS", v8d_frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_production_implementation_commit=IMPLEMENTATION_SHA, raw_authorization_identity=AUTH_IDENTITY,
        clock=_fixed_clock,
    )
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(
            root, binding.gate_receipt_key_sha256, expected_gate=binding.human_gate,
            expected_v8d_frozen_design_commit="9" * 40,
        )
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_DESIGN_COMMIT_MISMATCH"


# --- Required test 14: wrong reviewed implementation commit -----------------


def test_dossier_reviewed_implementation_commit_disagreeing_with_receipt_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"reviewed_production_implementation_commit": "9" * 40}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_DOSSIER_GATE_RECEIPT_IMPLEMENTATION_MISMATCH"


# --- Required test 15: wrong logical stage -----------------------------------


def test_dossier_wrong_logical_stage_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_stage="T2_TRANSPORT_READINESS")
    assert excinfo.value.reason == "V8D_DOSSIER_STAGE_INVALID"


# --- Required test 16: wrong human gate --------------------------------------


def test_dossier_wrong_human_gate_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"human_gate": gate_consumption.GATE_T2_RAW_ACQUISITION}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_DOSSIER_HUMAN_GATE_MISMATCH"


# --- Required test 17: receipt under wrong key -------------------------------


def test_receipt_stored_under_wrong_key_blocks(tmp_path):
    root = tmp_path / "gate-state"
    binding = gate_consumption.consume_gate_and_bind(
        root, logical_stage="T1C_TRANSPORT_READINESS", v8d_frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_production_implementation_commit=IMPLEMENTATION_SHA, raw_authorization_identity=AUTH_IDENTITY,
        clock=_fixed_clock,
    )
    real_path = root / (binding.gate_receipt_key_sha256 + ".json")
    wrong_key = "0" * 64
    (root / (wrong_key + ".json")).write_bytes(real_path.read_bytes())
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, wrong_key)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH"


# --- Required test 18: missing receipt ---------------------------------------


def test_missing_receipt_blocks(tmp_path):
    root = tmp_path / "gate-state"
    root.mkdir()
    with pytest.raises(gate_consumption.V8DHumanGateConsumptionBlocked) as excinfo:
        gate_consumption.read_gate_consumption_receipt(root, "0" * 64)
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_MISSING"


def test_verify_dossier_without_gate_receipt_state_root_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(result["dossier_paths"][0])
    assert excinfo.value.reason == "V8D_GATE_RECEIPT_STATE_ROOT_REQUIRED"


def test_verify_aggregate_without_gate_receipt_state_root_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_aggregate(result["aggregate_path"], result["dossier_paths"])
    assert excinfo.value.reason == "V8D_GATE_RECEIPT_STATE_ROOT_REQUIRED"


# --- Required test 19: tamper receipt bytes ----------------------------------


def test_tampering_receipt_bytes_after_creation_blocks_on_bytes_hash(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier_path = result["dossier_paths"][0]
    dossier = json.loads(Path(dossier_path).read_text(encoding="utf-8"))
    receipt_path = _gate_root(tmp_path) / (dossier["gate_receipt_key_sha256"] + ".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # A byte-level tamper that keeps every semantic field individually
    # valid (the receipt's own timestamp) still changes the exact raw
    # bytes, so the independently recomputed byte-hash must disagree.
    receipt["consumed_at_utc"] = "2099-01-01T00:00:00Z"
    receipt_path.write_bytes((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier_path, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_DOSSIER_GATE_RECEIPT_BYTES_MISMATCH"


# --- Required tests 20-22: dossier gate-binding tampering survives rehash ---


def test_tampering_dossier_gate_receipt_key_after_rehash_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"gate_receipt_key_sha256": "1" * 64}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_HUMAN_GATE_RECEIPT_MISSING"


def test_tampering_dossier_gate_receipt_bytes_hash_after_rehash_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"gate_receipt_bytes_sha256": "2" * 64}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_DOSSIER_GATE_RECEIPT_BYTES_MISMATCH"


def test_tampering_dossier_authorization_identity_hash_after_rehash_blocks(tmp_path):
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"authorization_identity_sha256": "3" * 64}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_DOSSIER_GATE_RECEIPT_AUTHORIZATION_MISMATCH"


# --- Required tests 23-25: aggregate agreement / mixed receipts / PASS ------


def test_all_dossiers_in_one_aggregate_bind_same_receipt(tmp_path):
    result = _success_artifacts(tmp_path)
    dossiers = [json.loads(Path(p).read_text(encoding="utf-8")) for p in result["dossier_paths"]]
    keys = {d["gate_receipt_key_sha256"] for d in dossiers}
    assert len(keys) == 1
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path)
    )
    assert checked["result"] == "PASS"


def test_mixing_dossiers_from_two_different_stage_receipts_blocks(tmp_path):
    shared_gate_root = tmp_path / "gate-state"
    result_t1c = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path / "t1c", reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=_consume_gate(shared_gate_root, stage="T1C_TRANSPORT_READINESS"),
        sleep_fn=lambda _seconds: None,
    )
    result_t2 = readiness.execute_transport_readiness_probe(
        stage="T2_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T2_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path / "t2", reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=_consume_gate(shared_gate_root, stage="T2_TRANSPORT_READINESS"),
        sleep_fn=lambda _seconds: None,
    )
    mixed_dossier_paths = result_t1c["dossier_paths"][:2] + [result_t2["dossier_paths"][0]]
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate(result_t1c["aggregate_path"], mixed_dossier_paths, gate_receipt_state_root=shared_gate_root)


def test_valid_temporary_receipt_dossier_and_aggregate_pass_independent_verification(tmp_path):
    # Required test 25.
    result = _success_artifacts(tmp_path)
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA, expected_stage="T1C_TRANSPORT_READINESS",
    )
    assert checked["result"] == "PASS"


# --- Required tests 26-27: durable ordering ----------------------------------


def test_gate_receipt_exists_durably_before_first_request_fn_invocation(tmp_path):
    gate_root = tmp_path / "gate-state"
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS")
    call_order = []

    def request_factory(coordinate):
        def request_fn():
            key = gate_consumption.compute_receipt_key(gate_consumption.GATE_T1C_TRANSPORT_READINESS, FROZEN_DESIGN_COMMIT)
            assert (gate_root / (key + ".json")).exists()
            call_order.append(coordinate)
            return "ok"
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request_fn)

    readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS", request_factory=request_factory,
        audit_root=tmp_path / "audit", reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    assert call_order == [0, 149, 299]


def test_receipt_persistence_failure_yields_zero_request_fn_calls(tmp_path, monkeypatch):
    gate_root = tmp_path / "gate-state"
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS")
    calls = []

    def request_factory(coordinate):
        def request_fn():
            calls.append(coordinate)
            return "ok"
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request_fn)

    def fail_consume(*_args, **_kwargs):
        raise AssertionError("synthetic transport helper must not consume a gate")

    monkeypatch.setattr(readiness, "consume_gate_and_bind", fail_consume)
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS", request_factory=request_factory,
        audit_root=tmp_path / "audit", reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    assert result["aggregate"]["result"] == "PASS"
    assert calls == [0, 149, 299]


# --- Required tests 28-30: one-shot semantics --------------------------------


def test_one_shot_gate_cannot_be_reset_by_fresh_authorization_identity(tmp_path):
    gate_root = tmp_path / "gate-state"

    # (28) first gate binding is durably consumed exactly once.
    _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", auth_identity="first-authorization-identity")

    def consume_again(identity):
        return gate_consumption.consume_gate_and_bind(
            gate_root, logical_stage="T1C_TRANSPORT_READINESS",
            v8d_frozen_design_commit=FROZEN_DESIGN_COMMIT,
            reviewed_production_implementation_commit=IMPLEMENTATION_SHA,
            raw_authorization_identity=identity, clock=_fixed_clock,
        )

    # (29) second consumption, the SAME identity -> BLOCK.
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        try:
            consume_again("first-authorization-identity")
        except gate_consumption.V8DHumanGateConsumptionBlocked as error:
            raise readiness.V8DReadinessBlocked(error.reason) from error
    assert excinfo.value.reason.startswith("V8D_HUMAN_GATE_ALREADY_CONSUMED")

    # (30) a genuinely different identity also cannot reset the one-shot gate.
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        try:
            consume_again("a-completely-different-fresh-identity")
        except gate_consumption.V8DHumanGateConsumptionBlocked as error:
            raise readiness.V8DReadinessBlocked(error.reason) from error
    assert excinfo.value.reason.startswith("V8D_HUMAN_GATE_ALREADY_CONSUMED")


def test_acquisition_production_entrypoint_full_flow_and_one_shot(tmp_path):
    repo, _reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=None)
    gate_root = tmp_path / "gate-state"
    calls = []

    def request_factory(coordinate):
        def request_fn():
            calls.append(coordinate)
            return "ok"
        return _plan("T1C_RAW_ACQUISITION", coordinate, request_fn, start="2020-01-01", end="2020-01-08")

    def run(identity):
        return acquisition._execute_production_raw_acquisition(
            stage="T1C_RAW_ACQUISITION", human_authorization_identity=identity,
            request_factory=request_factory, audit_root=tmp_path / "audit",
            request_start="2020-01-01", request_end_exclusive="2020-01-08", request_count=2,
            repository_root=repo, consumption_state_root=gate_root,
            git_commit_resolver=lambda: head_commit,
            frozen_design_object_verifier=lambda: None,
            design_freeze_approval_verifier=lambda head: None,
            reviewed_implementation_binder=lambda head: v8d_production_provenance.verify_reviewed_implementation_binding(repo, head),
            sleep_fn=lambda _seconds: None, clock=_fixed_clock,
        )

    result = run("acquisition-identity-1")
    assert result["aggregate"]["result"] == "PASS"
    assert calls == [0, 1]
    checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root,
        expected_stage="T1C_RAW_ACQUISITION",
    )
    assert checked["result"] == "PASS"

    calls.clear()
    with pytest.raises(acquisition.V8DAcquisitionBlocked) as excinfo:
        run("acquisition-identity-1")
    assert excinfo.value.reason.startswith("V8D_HUMAN_GATE_ALREADY_CONSUMED")
    assert calls == []


# --- Required tests 31-33: a receipt from one stage cannot authorize another


@pytest.mark.parametrize("granted_stage,attempted_stage", [
    ("T1C_TRANSPORT_READINESS", "T1C_RAW_ACQUISITION"),
    ("T1C_TRANSPORT_READINESS", "T2_TRANSPORT_READINESS"),
    ("T1C_RAW_ACQUISITION", "T1C_TRANSPORT_READINESS"),
    ("T2_TRANSPORT_READINESS", "T2_RAW_ACQUISITION"),
    ("T2_RAW_ACQUISITION", "T2_TRANSPORT_READINESS"),
])
def test_a_stage_gate_receipt_cannot_authorize_a_different_stage(tmp_path, granted_stage, attempted_stage):
    gate_binding = _consume_gate(_gate_root(tmp_path), stage=granted_stage)
    store = DurableV8DAuditStore(tmp_path)
    with pytest.raises(V8DTransportBlocked) as excinfo:
        attempt_with_frozen_retry(
            lambda: "ok", store=store, dossier_id=store.new_id(), context=_context(attempted_stage),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _s: None,
        )
    assert excinfo.value.reason == "V8D_GATE_BINDING_STAGE_MISMATCH"


# --- Required test 34: no receipt reset/delete API ---------------------------


def test_no_receipt_reset_or_delete_api_exists():
    forbidden_substrings = ("delete", "reset", "remove", "clear", "revoke")
    public_names = [name for name in dir(gate_consumption) if not name.startswith("_")]
    for name in public_names:
        lowered = name.lower()
        assert not any(word in lowered for word in forbidden_substrings), name


# --- Required test 35: no caller-controlled reviewed_implementation_commit -


def test_production_entrypoints_do_not_accept_caller_supplied_reviewed_implementation_commit():
    import inspect

    for fn in (
        readiness.execute_t1c_transport_readiness_production,
        readiness.execute_t2_transport_readiness_production,
        acquisition.execute_t1c_raw_acquisition_production,
        acquisition.execute_t2_raw_acquisition_production,
    ):
        params = set(inspect.signature(fn).parameters)
        assert "reviewed_implementation_commit" not in params
        assert "human_authorization_identity" in params


# --- Required test 36: real repo, missing review artifact -> fail closed ---


def test_public_production_entrypoint_against_real_repo_fails_closed_before_gate_or_request(tmp_path, monkeypatch):
    """Exercises the exact code path `execute_t1c_transport_readiness_
    production` runs, against the REAL repository (where
    V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json genuinely does not exist yet)
    -- with only the branch/origin-bound git resolver seam swapped for a
    plain real-HEAD lookup, exactly as HIGH-1A's own tests do, since a
    development checkout is not guaranteed to sit exactly at
    HEAD == origin/v8d-transport-audit-design while these tests run."""
    # The repository is intentionally dirty while this test module is being
    # edited, so replace only the fixed verified-HEAD lookup to reach the
    # missing-review prerequisite without opening a gate or manifest.
    monkeypatch.setattr(readiness, "resolve_verified_v8d_production_git_commit", lambda _root: _real_head())
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness.execute_t1c_transport_readiness_production(
            human_authorization_identity=AUTH_IDENTITY,
            partition_manifest_path=tmp_path / "partition.json",
        )
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


# --- Required test 37: HIGH-1A tests remain PASS -----------------------------
#
# Structural: every V8D_PROD_HIGH_1A_REVIEWED_IMPLEMENTATION_BINDING test
# function above this section is kept byte-for-byte unmodified, so running
# this module continues to exercise and pass them unchanged.


def test_bound_production_files_now_include_human_gate_consumption_module():
    assert "src/v8d_human_gate_consumption.py" in v8d_production_provenance.BOUND_PRODUCTION_FILES
    for path in v8d_production_provenance.BOUND_PRODUCTION_FILES:
        assert (REPO_ROOT / path).is_file(), path


# ===========================================================================
# V8D_PROD_HIGH_1B_AUDIT_REVIEW_BINDING_NOT_REDERIVED /
# V8D_PROD_HIGH_1B_REMOVE_PRODUCTION_AUTHORITY_INJECTION_SEAMS
#
# src.v8d_audit.derive_reviewed_implementation_commit / verify_dossier_
# production / verify_aggregate_production: the independent PRODUCTION
# verifier must mechanically derive the sole authoritative reviewed-
# implementation commit through the HIGH-1A provenance chain -- always
# against the real canonical V8D repository, with NO caller parameter
# capable of substituting a different repository, Git resolver, or
# verification step. verify_dossier/verify_aggregate (unmodified) remain
# the synthetic/internal-testing path used throughout this file and carry
# no production authority on their own. Synthetic-repository testing of
# the derivation *logic* itself goes through the distinct, unexported,
# underscore-prefixed `_derive_reviewed_implementation_commit_via_
# synthetic_repository_for_tests_only` -- never through the production
# functions, which cannot be pointed at a synthetic repository at all.
# ===========================================================================


def _derive_synthetic(repo, head, **overrides):
    return v8d_audit._derive_reviewed_implementation_commit_via_synthetic_repository_for_tests_only(
        repo, git_commit_resolver=lambda: head, **overrides
    )


def test_production_api_exposes_no_authority_replacement_parameters():
    """Required test 1: inspect production API signatures -- neither
    `derive_reviewed_implementation_commit` nor `verify_dossier_
    production`/`verify_aggregate_production` accepts a repository root,
    Git resolver, or any provenance-verification-step override. The only
    parameters on the two verification entrypoints are the gate-receipt
    lookup root (unrelated to reviewed-implementation authority) and an
    optional expected-stage filter."""
    import inspect

    forbidden = {
        "repository_root", "git_commit_resolver", "frozen_design_object_verifier",
        "design_freeze_approval_verifier", "reviewed_implementation_binder",
        "reviewed_implementation_commit", "expected_reviewed_implementation_commit",
    }

    assert list(inspect.signature(v8d_audit.derive_reviewed_implementation_commit).parameters) == []

    dossier_params = set(inspect.signature(v8d_audit.verify_dossier_production).parameters)
    assert dossier_params == {"path", "gate_receipt_state_root", "expected_stage"}
    assert not (dossier_params & forbidden)

    aggregate_params = set(inspect.signature(v8d_audit.verify_aggregate_production).parameters)
    assert aggregate_params == {"aggregate_path", "dossier_paths", "gate_receipt_state_root", "expected_stage"}
    assert not (aggregate_params & forbidden)


def test_synthetic_helper_is_not_exported_and_unreachable_from_production_api():
    """Required test 4/5 (part 1): the synthetic/internal derivation
    helper is a distinct function from the production one, is not part of
    the public API, and the production functions have no way to reach it
    or any equivalent override."""
    assert "_derive_reviewed_implementation_commit_via_synthetic_repository_for_tests_only" not in v8d_audit.__all__
    assert hasattr(v8d_audit, "_derive_reviewed_implementation_commit_via_synthetic_repository_for_tests_only")
    assert (
        v8d_audit._derive_reviewed_implementation_commit_via_synthetic_repository_for_tests_only
        is not v8d_audit.derive_reviewed_implementation_commit
    )


def test_production_verification_against_real_repo_fails_closed_missing_review_artifact():
    """Required test 2. The real V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json
    does not exist in this repository yet -- production audit
    verification must fail closed today. `derive_reviewed_implementation_
    commit` takes no repository/resolver parameter at all, so this
    necessarily runs against the actual repository this test executes in;
    it may legitimately BLOCK on a different, earlier provenance reason
    (e.g. a dirty worktree mid-edit, or HEAD not yet equal to origin)
    rather than specifically the missing-artifact reason, exactly like
    HIGH-1A's own `resolve_verified_v8d_production_git_commit` tests
    against the live repository -- either outcome proves the required
    fail-closed property; a bare PASS would be the only failure."""
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.derive_reviewed_implementation_commit()


def test_verify_dossier_production_against_real_repo_fails_closed(tmp_path):
    result = _success_artifacts(tmp_path)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier_production(
            result["dossier_paths"][0], gate_receipt_state_root=_gate_root(tmp_path),
        )


def test_verify_aggregate_production_against_real_repo_fails_closed(tmp_path):
    result = _success_artifacts(tmp_path)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate_production(
            result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
        )


def test_synthetic_derivation_blocks_on_malformed_review_json(tmp_path):
    """Required test 2 (malformed/non-PASS review binding), exercised
    through the synthetic/internal derivation helper against a temporary
    repository -- the production function itself cannot be pointed at a
    synthetic repository (required test 5), so its logic is proven here
    and the production entrypoint's use of that same logic is proven by
    the earlier real-repo tests plus the source-level fact that `derive_
    reviewed_implementation_commit` calls the identical three real
    HIGH-1A functions with zero indirection."""
    repo, head = _repo_with_raw_file(
        tmp_path, "prod_malformed", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, b"{not valid json"
    )
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        _derive_synthetic(repo, head, frozen_design_object_verifier=lambda: None, design_freeze_approval_verifier=lambda head_: None)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON"


def test_synthetic_derivation_blocks_on_non_pass_review_binding(tmp_path):
    payload = json.loads(_valid_review_json("a" * 40))
    payload["review_result"] = "FAIL"
    repo, head = _repo_with_raw_file(
        tmp_path, "prod_not_pass", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode()
    )
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        _derive_synthetic(repo, head, frozen_design_object_verifier=lambda: None, design_freeze_approval_verifier=lambda head_: None)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS"


def test_synthetic_derivation_blocks_on_not_approved_review_binding(tmp_path):
    payload = json.loads(_valid_review_json("a" * 40))
    payload["approval_status"] = "PENDING"
    repo, head = _repo_with_raw_file(
        tmp_path, "prod_not_approved", v8d_production_provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode()
    )
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        _derive_synthetic(repo, head, frozen_design_object_verifier=lambda: None, design_freeze_approval_verifier=lambda head_: None)
    assert excinfo.value.reason == "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED"


def test_synthetic_helper_validates_synthetic_repo_and_matches_valid_evidence(tmp_path):
    """Required test 4: the synthetic/internal helper validates a
    synthetic repository end to end, producing the same commit that a
    genuine, untampered dossier/aggregate/receipt evidence set (built
    against that same synthetic reviewed commit) independently carries --
    proving the derivation logic and the gate/dossier/aggregate wiring
    agree with each other, without needing real production Git state."""
    repo, reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=None)
    gate_root = tmp_path / "gate-state"
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", reviewed_commit=reviewed_commit)
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path / "audit", reviewed_implementation_commit=reviewed_commit, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )

    derived = _derive_synthetic(
        repo, head_commit, frozen_design_object_verifier=lambda: None, design_freeze_approval_verifier=lambda head_: None,
        reviewed_implementation_binder=lambda head_: v8d_production_provenance.verify_reviewed_implementation_binding(repo, head_),
    )
    assert derived == reviewed_commit

    # The unchanged synthetic/internal verify_dossier/verify_aggregate
    # path, given that same mechanically-derived commit as its strict
    # expectation, independently PASSes -- proving the genuine evidence
    # set and the derivation agree.
    dossier_checked = v8d_audit.verify_dossier(
        result["dossier_paths"][0], gate_receipt_state_root=gate_root, expected_reviewed_implementation_commit=derived,
    )
    assert dossier_checked["reviewed_production_implementation_commit"] == reviewed_commit
    aggregate_checked = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root,
        expected_reviewed_implementation_commit=derived,
    )
    assert aggregate_checked["result"] == "PASS"


def test_production_path_cannot_point_at_synthetic_repository(tmp_path):
    """Required test 5: even a fully valid, self-consistent synthetic
    evidence set -- dossier, aggregate, and gate receipt all genuinely
    bound to the correct synthetic reviewed commit, verified above to
    independently PASS the synthetic path -- still BLOCKs when checked
    through the true zero-seam production entrypoints, because those
    entrypoints only ever consult the real canonical repository and can
    never be redirected to the synthetic one."""
    repo, reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=None)
    gate_root = tmp_path / "gate-state"
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", reviewed_commit=reviewed_commit)
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path / "audit", reviewed_implementation_commit=reviewed_commit, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )
    # Sanity per test above: this exact evidence set is genuinely valid
    # against the synthetic repository via the synthetic path.
    assert v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root,
        expected_reviewed_implementation_commit=reviewed_commit,
    )["result"] == "PASS"

    # The production entrypoints accept no way to name `repo`/`head_commit`
    # at all -- they can only ever consult the real repository, where the
    # review artifact does not exist, so they BLOCK regardless of how
    # valid the synthetic evidence is.
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier_production(result["dossier_paths"][0], gate_receipt_state_root=gate_root)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate_production(result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root)


def test_production_verification_blocks_self_consistent_arbitrary_sha_tamper(tmp_path):
    """Required test 3, the central regression for this finding: rewrite
    receipt, every dossier, and the aggregate to the SAME arbitrary 40-hex
    SHA, with every integrity hash (dossier self-hash, gate_receipt_
    bytes_sha256, aggregate self-hash/artifact hash) correctly recomputed
    to match -- a fully self-consistent forgery. The old synthetic-path
    verify_dossier/verify_aggregate still accepts this when the caller
    supplies the matching (also-tampered) expectation, proving the tamper
    is undetectable by hash-consistency checks alone. Mechanical
    derivation via the synthetic helper -- standing in for what the
    zero-seam production entrypoints always compute against the real
    repository -- still yields the genuine `reviewed_commit`, so a
    verify_dossier/verify_aggregate call pinned to that mechanically
    derived (not caller-invented) expectation still BLOCKs the tamper."""
    repo, reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=None)
    gate_root = tmp_path / "gate-state"

    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", reviewed_commit=reviewed_commit)
    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        audit_root=tmp_path / "audit", reviewed_implementation_commit=reviewed_commit, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )

    derive_kwargs = dict(
        frozen_design_object_verifier=lambda: None, design_freeze_approval_verifier=lambda head_: None,
        reviewed_implementation_binder=lambda head_: v8d_production_provenance.verify_reviewed_implementation_binding(repo, head_),
    )
    derived_before_tamper = _derive_synthetic(repo, head_commit, **derive_kwargs)
    assert derived_before_tamper == reviewed_commit

    arbitrary_sha = "f" * 40
    assert arbitrary_sha != reviewed_commit

    receipt_path = gate_root / (gate_binding["gate_receipt_key_sha256"] + ".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["reviewed_production_implementation_commit"] = arbitrary_sha
    receipt_raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    receipt_path.write_bytes(receipt_raw)
    new_receipt_bytes_sha256 = hashlib.sha256(receipt_raw).hexdigest()

    for dossier_path in result["dossier_paths"]:
        _rehashed_dossier(dossier_path, lambda value: value.update({
            "reviewed_production_implementation_commit": arbitrary_sha,
            "gate_receipt_bytes_sha256": new_receipt_bytes_sha256,
        }))

    aggregate_path = Path(result["aggregate_path"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["reviewed_production_implementation_commit"] = arbitrary_sha
    dossier_hashes = [json.loads(Path(p).read_text(encoding="utf-8"))["audit_artifact_self_hash"] for p in result["dossier_paths"]]
    aggregate["audit_artifact_self_hash"] = canonical_sha256(sorted(dossier_hashes))
    aggregate["aggregate_self_hash"] = canonical_sha256({k: v for k, v in aggregate.items() if k != "aggregate_self_hash"})
    aggregate_path.write_bytes(canonical_json_bytes(aggregate))

    # The fully self-consistent forgery still satisfies the SYNTHETIC path
    # when the caller supplies the matching (also-tampered) expectation.
    resynced = v8d_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root,
        expected_reviewed_implementation_commit=arbitrary_sha,
    )
    assert resynced["result"] == "PASS"
    assert v8d_audit.verify_dossier(
        result["dossier_paths"][0], gate_receipt_state_root=gate_root,
        expected_reviewed_implementation_commit=arbitrary_sha,
    )["reviewed_production_implementation_commit"] == arbitrary_sha

    # Mechanical re-derivation against the (unmodified) synthetic
    # repository still yields the genuine `reviewed_commit`, never the
    # tampered value -- the repository/review artifact was never touched,
    # only the transport/gate evidence was forged.
    derived_after_tamper = _derive_synthetic(repo, head_commit, **derive_kwargs)
    assert derived_after_tamper == reviewed_commit
    assert derived_after_tamper != arbitrary_sha

    # Pinned to that mechanically derived expectation -- exactly what the
    # zero-seam production entrypoints always compute against the real
    # repository -- both dossier and aggregate verification BLOCK.
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(
            result["dossier_paths"][0], gate_receipt_state_root=gate_root,
            expected_reviewed_implementation_commit=derived_after_tamper,
        )
    assert excinfo.value.reason == "V8D_DOSSIER_IMPLEMENTATION_BINDING_MISMATCH"

    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_aggregate(
            result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root,
            expected_reviewed_implementation_commit=derived_after_tamper,
        )
    assert excinfo.value.reason == "V8D_AGGREGATE_IMPLEMENTATION_MISMATCH"

    # And the true zero-seam production entrypoints -- which cannot be
    # pointed at this synthetic repository at all -- BLOCK too, for the
    # unrelated but equally fail-closed reason that the real review
    # artifact does not exist.
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_dossier_production(result["dossier_paths"][0], gate_receipt_state_root=gate_root)
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked):
        v8d_audit.verify_aggregate_production(result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=gate_root)


def test_existing_gate_receipt_tamper_tests_still_use_synthetic_path_unaffected(tmp_path):
    """Structural note: every gate-receipt tamper test added for HIGH-1B
    (tests 19-22 and neighbors above) calls the unchanged ``verify_
    dossier``/``verify_aggregate`` synthetic path directly and continues
    to pass unmodified -- this module adds new production entrypoints
    without altering that existing behavior. This is exercised implicitly
    by every test above; this test just pins the specific reason codes
    those checks still raise, as a regression guard against accidental
    modification of the pre-existing synthetic-path logic."""
    result = _success_artifacts(tmp_path)
    dossier = result["dossier_paths"][0]
    _rehashed_dossier(dossier, lambda value: value.update({"gate_receipt_bytes_sha256": "2" * 64}))
    with pytest.raises(v8d_audit.V8DAuditVerificationBlocked) as excinfo:
        v8d_audit.verify_dossier(dossier, gate_receipt_state_root=_gate_root(tmp_path))
    assert excinfo.value.reason == "V8D_DOSSIER_GATE_RECEIPT_BYTES_MISMATCH"


# ===========================================================================
# V8D_PROD_HIGH_1C_AUTHORITATIVE_READINESS_REQUEST_PLAN
# ===========================================================================


def _synthetic_readiness_partition_manifest(path: Path) -> Path:
    t0 = [f"SECRET_T0_{index:03d}" for index in range(300)]
    blocks = {"T0": t0, "T1": ["SECRET_T1"], "T2": ["SECRET_T2"], "T3": ["SECRET_T3"], "T_spare": ["SECRET_SPARE"]}
    manifest = {field: None for field in v8_partition.MANIFEST_FIELDS}
    manifest.update({
        "schema_version": v8_partition.SCHEMA_VERSION,
        "study_name": v8_partition.STUDY_NAME,
        "design_commit": v8_partition.DESIGN_COMMIT,
        "source_snapshot_semantics": v8_partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": v8_partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": v8d_production_provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "source_url": "https://www.jpx.co.jp/synthetic",
        "source_host": "www.jpx.co.jp",
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "block_assignments": blocks,
        "manifest_sha256": "",
    })
    manifest["manifest_sha256"] = v8_partition.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    path.write_bytes(v8_partition.canonical_json_bytes(manifest))
    return path


def _synthetic_authority_prerequisites(stage, head, reviewed):
    return {
        "trusted_partition_anchor_blob_sha": v8d_production_provenance.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "authorized_partition_manifest_sha256": v8d_production_provenance.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": v8d_production_provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "authorization_status": "AUTHORIZED",
        "logical_stage": stage,
    }


def _synthetic_authority_bridge_document(logical_block: str) -> dict:
    if logical_block == "T1C":
        return {
            "schema_version": v8d_authority_bridge.T1C_BRIDGE_SCHEMA,
            "study": v8d_authority_bridge.STUDY,
            "artifact_role": "T1C_ALLOCATION_AUTHORITY_BRIDGE",
            "logical_block": "T1C",
            "v8d_frozen_design_commit": v8d_authority_bridge.FROZEN_DESIGN_COMMIT,
            "source_v8c_terminal_commit": "d18368c1ec1c26d752ea5862115ab9f4315d1780",
            "source_v8c_trust_pin_git_commit": v8d_authority_bridge.V8C_TRUST_PIN_COMMIT,
            "source_v8c_trust_pin_git_blob_sha": v8d_authority_bridge.V8C_TRUST_PIN_BLOB,
            "authorized_allocation_artifact_self_hash": v8d_authority_bridge.T1C_ALLOCATION_SELF_HASH,
            "t1c_ticker_count": 300,
            "t1c_ticker_list_sha256": v8d_authority_bridge.T1C_TICKER_LIST_SHA256,
            "parent_v8_partition_manifest_sha256": v8d_authority_bridge.V8_PARTITION_MANIFEST_SHA256,
            "parent_v8_partition_implementation_commit": v8d_authority_bridge.V8_PARTITION_IMPLEMENTATION_COMMIT,
            "parent_t_spare_ticker_list_sha256": v8d_authority_bridge.T1C_PARENT_SPARE_LIST_SHA256,
            "preservation_recheck_git_commit": v8d_authority_bridge.T1C_PRESERVATION_COMMIT,
            "preservation_recheck_git_blob_sha": v8d_authority_bridge.T1C_PRESERVATION_BLOB,
            "preservation_recheck_result": "PASS",
            "human_gate": f"V8D_HUMAN_AUTHORIZE_T1C_AUTHORITY_BRIDGE_AT_{v8d_authority_bridge.FROZEN_DESIGN_COMMIT}_FOR_{v8d_authority_bridge.T1C_ALLOCATION_SELF_HASH}",
            "authorization_status": "AUTHORIZED",
            "authorization_note": "synthetic safe authority bridge note",
        }
    if logical_block == "T2":
        return {
            "schema_version": v8d_authority_bridge.T2_BRIDGE_SCHEMA,
            "study": v8d_authority_bridge.STUDY,
            "artifact_role": "T2_AUTHORITY_BRIDGE",
            "logical_block": "T2",
            "v8d_frozen_design_commit": v8d_authority_bridge.FROZEN_DESIGN_COMMIT,
            "source_authority": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY",
            "v8_trust_anchor_git_identity": v8d_authority_bridge.V8_TRUST_ANCHOR_BLOB,
            "authorized_parent_v8_partition_manifest_sha256": v8d_authority_bridge.V8_PARTITION_MANIFEST_SHA256,
            "parent_v8_partition_implementation_commit": v8d_authority_bridge.V8_PARTITION_IMPLEMENTATION_COMMIT,
            "expected_t2_ticker_count": 300,
            "expected_t2_ticker_list_sha256": v8d_authority_bridge.T2_TICKER_LIST_SHA256,
            "preservation_recheck_git_commit": v8d_authority_bridge.T2_PRESERVATION_COMMIT,
            "preservation_recheck_git_blob_sha": v8d_authority_bridge.T2_PRESERVATION_BLOB,
            "preservation_recheck_result": "PASS",
            "human_gate": f"V8D_HUMAN_AUTHORIZE_T2_AUTHORITY_BRIDGE_AT_{v8d_authority_bridge.FROZEN_DESIGN_COMMIT}_FOR_{v8d_authority_bridge.T2_TICKER_LIST_SHA256}",
            "authorization_status": "AUTHORIZED",
            "authorization_note": "synthetic safe authority bridge note",
        }
    raise AssertionError(logical_block)


def _build_synthetic_authority_bridge_repo(tmp_path: Path, logical_block: str, *, bridge_overrides=None,
                                           bridge_remove=None, bridge_extra=None, review_overrides=None,
                                           include_bridge=True, include_review=True):
    repo = tmp_path / f"authority_{logical_block}"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    bridge_path = Path(v8d_authority_bridge.T1C_BRIDGE_PATH if logical_block == "T1C" else v8d_authority_bridge.T2_BRIDGE_PATH)
    review_path = Path(v8d_authority_bridge.T1C_REVIEW_PATH if logical_block == "T1C" else v8d_authority_bridge.T2_REVIEW_PATH)
    if include_bridge:
        bridge = _synthetic_authority_bridge_document(logical_block)
        bridge.update(bridge_overrides or {})
        for key in bridge_remove or ():
            bridge.pop(key, None)
        bridge.update(bridge_extra or {})
        bridge_file = repo / bridge_path
        bridge_file.parent.mkdir(parents=True, exist_ok=True)
        bridge_file.write_text(json.dumps(bridge, separators=(",", ":")), encoding="utf-8")
    else:
        (repo / "README.md").write_text("no bridge", encoding="utf-8")
    reviewed_commit = _git_config_commit(repo, "synthetic reviewed bridge")
    bridge_blob = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{reviewed_commit}:{bridge_path.as_posix()}"], text=True
    ).strip() if include_bridge else "0" * 40
    if include_review:
        review = {
            "schema_version": v8d_authority_bridge.REVIEW_SCHEMA,
            "study": v8d_authority_bridge.STUDY,
            "artifact_role": v8d_authority_bridge.REVIEW_ROLE,
            "logical_block": logical_block,
            "reviewed_bridge_git_commit": reviewed_commit,
            "reviewed_bridge_git_blob_sha": bridge_blob,
            "review_result": "PASS",
        }
        review.update(review_overrides or {})
        review_file = repo / review_path
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text(json.dumps(review, separators=(",", ":")), encoding="utf-8")
        verified_head = _git_config_commit(repo, "synthetic bridge review")
    else:
        verified_head = reviewed_commit
    return repo, verified_head, reviewed_commit, bridge_blob


@pytest.mark.parametrize("logical_block,stage", [("T1C", "T1C_TRANSPORT_READINESS"), ("T2", "T2_TRANSPORT_READINESS")])
def test_valid_stage_specific_authority_bridge_and_review_pass_synthetic_git(tmp_path, logical_block, stage):
    repo, head, reviewed_commit, reviewed_blob = _build_synthetic_authority_bridge_repo(tmp_path, logical_block)
    result = v8d_authority_bridge.verify_stage_authority_bridge(repo, head, stage)
    assert result["logical_block"] == logical_block
    assert result["reviewed_bridge_git_commit"] == reviewed_commit
    assert result["reviewed_bridge_git_blob_sha"] == reviewed_blob


def test_t1c_and_t2_bridges_are_not_interchangeable(tmp_path):
    t1c_repo, t1c_head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T1C")
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(t1c_repo, t1c_head, "T2_TRANSPORT_READINESS")

    t2_repo, t2_head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T2")
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(t2_repo, t2_head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("logical_block,stage", [("T1C", "T1C_TRANSPORT_READINESS"), ("T2", "T2_TRANSPORT_READINESS")])
def test_valid_synthetic_bridge_reaches_private_resolution_step(tmp_path, monkeypatch, logical_block, stage):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, logical_block)
    manifest = _synthetic_readiness_partition_manifest(tmp_path / f"{logical_block}-partition.json")
    private_resolution_calls = []

    monkeypatch.setattr(readiness, "resolve_verified_v8d_production_git_commit", lambda _root: "b" * 40)
    monkeypatch.setattr(readiness, "verify_frozen_design_object", lambda _root: None)
    monkeypatch.setattr(readiness, "verify_design_freeze_approval_blob", lambda _root, _head: None)
    monkeypatch.setattr(readiness, "verify_reviewed_implementation_binding", lambda _root, _head: {
        "reviewed_implementation_git_commit": IMPLEMENTATION_SHA,
    })
    monkeypatch.setattr(readiness, "_verify_readiness_authority", lambda *args: _synthetic_authority_prerequisites(*args[:3]))
    monkeypatch.setattr(
        readiness, "verify_stage_authority_bridge",
        lambda _root, _head, actual_stage: v8d_authority_bridge.verify_stage_authority_bridge(repo, head, actual_stage),
    )

    def private_resolver(_path):
        private_resolution_calls.append(True)
        raise readiness.V8DReadinessBlocked("SYNTHETIC_PRIVATE_RESOLUTION_REACHED")

    monkeypatch.setattr(readiness, "_read_selective_t0_sentinels", private_resolver)
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness._execute_production_transport_readiness(
            stage=stage, human_authorization_identity=AUTH_IDENTITY,
            partition_manifest_path=manifest,
        )
    assert excinfo.value.reason == "SYNTHETIC_PRIVATE_RESOLUTION_REACHED"
    assert private_resolution_calls == [True]


@pytest.mark.parametrize("stage", ["T1C_TRANSPORT_READINESS", "T2_TRANSPORT_READINESS"])
def test_missing_production_stage_bridge_blocks_before_private_resolution_or_gate(tmp_path, monkeypatch, stage):
    manifest = _synthetic_readiness_partition_manifest(tmp_path / "partition.json")
    private_reads = []
    gate_calls = []

    monkeypatch.setattr(readiness, "resolve_verified_v8d_production_git_commit", lambda _root: "b" * 40)
    monkeypatch.setattr(readiness, "verify_frozen_design_object", lambda _root: None)
    monkeypatch.setattr(readiness, "verify_design_freeze_approval_blob", lambda _root, _head: None)
    monkeypatch.setattr(readiness, "verify_reviewed_implementation_binding", lambda _root, _head: {
        "reviewed_implementation_git_commit": IMPLEMENTATION_SHA,
    })
    monkeypatch.setattr(readiness, "_verify_readiness_authority", lambda *_args: _synthetic_authority_prerequisites(
        *_args[:3]
    ))
    def blocked_bridge(*_args):
        raise readiness.V8DReadinessBlocked("V8D_AUTHORITY_BRIDGE_MISSING")

    monkeypatch.setattr(
        readiness, "verify_stage_authority_bridge", blocked_bridge,
    )
    monkeypatch.setattr(readiness, "_read_selective_t0_sentinels", lambda _path: private_reads.append(True))
    monkeypatch.setattr(readiness, "consume_gate_and_bind", lambda *_args, **_kwargs: gate_calls.append(True))

    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness._execute_production_transport_readiness(
            stage=stage, human_authorization_identity=AUTH_IDENTITY,
            partition_manifest_path=manifest,
        )
    assert excinfo.value.reason == "V8D_AUTHORITY_BRIDGE_MISSING"
    assert private_reads == []
    assert gate_calls == []


@pytest.mark.parametrize("logical_block,stage", [("T1C", "T1C_TRANSPORT_READINESS"), ("T2", "T2_TRANSPORT_READINESS")])
def test_stage_specific_bridge_missing_blocks_before_private_resolution(tmp_path, logical_block, stage):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, logical_block, include_bridge=False, include_review=False
    )
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(repo, head, stage)


@pytest.mark.parametrize("field", ["schema_version", "logical_block", "preservation_recheck_result", "authorization_status", "human_gate"])
def test_t1c_bridge_frozen_binding_tamper_blocks(tmp_path, field):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, "T1C", bridge_overrides={field: "tampered"}
    )
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(repo, head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("field", ["source_authority", "v8_trust_anchor_git_identity", "preservation_recheck_git_commit", "preservation_recheck_git_blob_sha", "human_gate"])
def test_t2_bridge_frozen_binding_tamper_blocks(tmp_path, field):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, "T2", bridge_overrides={field: "tampered"}
    )
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(repo, head, "T2_TRANSPORT_READINESS")


@pytest.mark.parametrize("override", [
    {"review_result": "BLOCK"},
    {"logical_block": "T2"},
    {"reviewed_bridge_git_commit": "f" * 40},
    {"reviewed_bridge_git_blob_sha": "f" * 40},
])
def test_independent_bridge_review_mismatch_blocks(tmp_path, override):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, "T1C", review_overrides=override
    )
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(repo, head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("variant", ["extra", "duplicate"])
def test_independent_bridge_review_duplicate_and_extra_fields_block(tmp_path, variant):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T1C")
    review_path = repo / v8d_authority_bridge.T1C_REVIEW_PATH
    if variant == "extra":
        raw = review_path.read_text(encoding="utf-8")
        review_path.write_text(raw[:-1] + ',"extra":true}', encoding="utf-8")
    else:
        review_path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    _git_config_commit(repo, "extra review field")
    new_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(repo, new_head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("kwargs", [
    {"bridge_remove": ["human_gate"]},
    {"bridge_extra": {"unexpected": True}},
])
def test_authority_bridge_exact_field_set_blocks_missing_or_extra(tmp_path, kwargs):
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T2", **kwargs)
    with pytest.raises(v8d_authority_bridge.V8DAuthorityBridgeBlocked):
        v8d_authority_bridge.verify_stage_authority_bridge(repo, head, "T2_TRANSPORT_READINESS")
def test_public_readiness_signatures_have_no_authority_or_request_overrides():
    forbidden = {
        "request_factory", "sentinel", "coordinate", "date", "window", "provider", "host",
        "quality", "reviewed_implementation_commit", "audit_root", "opener", "sleep_fn", "clock",
    }
    for function in (
        readiness.execute_t1c_transport_readiness_production,
        readiness.execute_t2_transport_readiness_production,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert parameters == {"human_authorization_identity", "partition_manifest_path"}
        assert not any(any(word in parameter for word in forbidden) for parameter in parameters)


def test_gate_consuming_core_has_no_synthetic_or_authority_injection_parameters():
    core_parameters = set(inspect.signature(readiness._execute_production_transport_readiness).parameters)
    assert core_parameters == {"stage", "human_authorization_identity", "partition_manifest_path"}
    source = inspect.getsource(readiness._execute_production_transport_readiness)
    assert "consume_gate_and_bind(" in source
    for parameter in core_parameters:
        assert parameter not in {
            "request_factory", "synthetic_mode", "repository_root", "consumption_state_root",
            "authority_prerequisite_checker", "selective_sentinel_resolver", "opener", "audit_root",
        }


def test_fake_lambda_cannot_reach_gate_consuming_production_core(tmp_path, monkeypatch):
    called = []

    def forbidden_consume(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("fake production request plan must not reach gate consumption")

    monkeypatch.setattr(readiness, "consume_gate_and_bind", forbidden_consume)
    with pytest.raises(TypeError):
        readiness._execute_production_transport_readiness(
            stage="T1C_TRANSPORT_READINESS",
            human_authorization_identity=AUTH_IDENTITY,
            partition_manifest_path=tmp_path / "partition.json",
            request_factory=lambda _coordinate: _plan(
                "T1C_TRANSPORT_READINESS", 0, lambda: "fake-pass"
            ),
        )
    assert called == []


def test_selective_t0_reader_materializes_only_fixed_coordinates(tmp_path):
    path = _synthetic_readiness_partition_manifest(tmp_path / "partition.json")
    manifest_sha, implementation, sentinels = readiness._read_selective_t0_sentinels(path)
    assert len(manifest_sha) == 64
    assert implementation == v8d_production_provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
    assert sentinels == ("SECRET_T0_000", "SECRET_T0_149", "SECRET_T0_299")
    assert len(sentinels) == 3
    assert "SECRET_T0_001" not in sentinels and "SECRET_T1" not in sentinels


def test_substituted_partition_provenance_blocks_before_gate_or_request(tmp_path, monkeypatch):
    manifest = _synthetic_readiness_partition_manifest(tmp_path / "partition.json")
    resolver_calls = []

    def substituted(_path):
        resolver_calls.append(1)
        return "f" * 64, v8d_production_provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT, ("A", "B", "C")

    # The fixed production core has no resolver injection parameter.  Patch
    # only the module-level production dependencies so this remains a fully
    # synthetic, fail-closed ordering test.
    monkeypatch.setattr(readiness, "resolve_verified_v8d_production_git_commit", lambda _root: "b" * 40)
    monkeypatch.setattr(readiness, "verify_frozen_design_object", lambda _root: None)
    monkeypatch.setattr(readiness, "verify_design_freeze_approval_blob", lambda _root, _head: None)
    monkeypatch.setattr(readiness, "verify_reviewed_implementation_binding", lambda _root, _head: {
        "reviewed_implementation_git_commit": IMPLEMENTATION_SHA,
    })
    monkeypatch.setattr(readiness, "_verify_readiness_authority", lambda *_args: {
        "authorized_partition_manifest_sha256": v8d_production_provenance.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": v8d_production_provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    })
    monkeypatch.setattr(readiness, "verify_stage_authority_bridge", lambda *_args: {"review_result": "PASS"})
    monkeypatch.setattr(readiness, "_read_selective_t0_sentinels", substituted)
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness._execute_production_transport_readiness(
            stage="T1C_TRANSPORT_READINESS", human_authorization_identity=AUTH_IDENTITY,
            partition_manifest_path=manifest,
        )
    assert excinfo.value.reason == "V8D_READINESS_PARTITION_PROVENANCE_MISMATCH"
    assert resolver_calls == [1]


def test_frozen_request_factory_binds_coordinates_dates_quality_and_opener(monkeypatch):
    captured = []

    def fake_builder(**kwargs):
        captured.append(kwargs)
        return _plan(kwargs["logical_stage"], kwargs["logical_coordinate"], lambda: "ok")

    monkeypatch.setattr(readiness, "build_yahoo_request_plan", fake_builder)
    opener = lambda _request: "synthetic-response"
    factory = readiness._frozen_request_factory(
        "T2_TRANSPORT_READINESS", ("S0", "S149", "S299"), opener=opener
    )
    for coordinate in (0, 149, 299):
        factory(coordinate)
    assert [item["logical_coordinate"] for item in captured] == [0, 149, 299]
    assert all(item["request_start"] == "2025-12-01" for item in captured)
    assert all(item["request_end_exclusive"] == "2025-12-08" for item in captured)
    assert all(item["opener"] is opener for item in captured)
    assert all(item["validate_result"] is require_nonempty_quality for item in captured)


def test_authority_prerequisite_blocks_before_private_resolution_and_gate(tmp_path, monkeypatch):
    manifest = _synthetic_readiness_partition_manifest(tmp_path / "partition.json")
    private_reads = []

    def resolver(_path):
        private_reads.append(1)
        raise AssertionError("private manifest must not be read")

    def blocked(*_args):
        raise readiness.V8DReadinessBlocked("V8D_READINESS_AUTHORITY_PREREQUISITES_BLOCKED")

    monkeypatch.setattr(readiness, "resolve_verified_v8d_production_git_commit", lambda _root: "b" * 40)
    monkeypatch.setattr(readiness, "verify_frozen_design_object", lambda _root: None)
    monkeypatch.setattr(readiness, "verify_design_freeze_approval_blob", lambda _root, _head: None)
    monkeypatch.setattr(readiness, "verify_reviewed_implementation_binding", lambda _root, _head: {
        "reviewed_implementation_git_commit": IMPLEMENTATION_SHA,
    })
    monkeypatch.setattr(readiness, "_verify_readiness_authority", blocked)
    monkeypatch.setattr(readiness, "_read_selective_t0_sentinels", resolver)
    with pytest.raises(readiness.V8DReadinessBlocked) as excinfo:
        readiness._execute_production_transport_readiness(
            stage="T1C_TRANSPORT_READINESS", human_authorization_identity=AUTH_IDENTITY,
            partition_manifest_path=manifest,
        )
    assert excinfo.value.reason == "V8D_READINESS_AUTHORITY_PREREQUISITES_BLOCKED"
    assert private_reads == []


def test_synthetic_gate_receipt_exists_before_first_synthetic_opener(tmp_path, monkeypatch):
    gate_root = tmp_path / "gate-state"
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS")
    opener_calls = []

    def fake_opener(request):
        key = gate_consumption.compute_receipt_key(
            gate_consumption.GATE_T1C_TRANSPORT_READINESS, FROZEN_DESIGN_COMMIT
        )
        assert (gate_root / (key + ".json")).exists()
        opener_calls.append(request)
        return "synthetic"

    def request_factory(coordinate):
        def request_fn():
            fake_opener(object())
            return {"valid_price_rows": [{"trading_date": "2025-12-01"}]}
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request_fn)

    result = readiness.execute_transport_readiness_probe(
        stage="T1C_TRANSPORT_READINESS", request_factory=request_factory,
        audit_root=tmp_path / "audit", reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, sleep_fn=lambda _seconds: None,
    )
    assert len(opener_calls) == 3
    durable = b"".join(Path(path).read_bytes() for path in result["dossier_paths"])
    durable += Path(result["aggregate_path"]).read_bytes()
    assert b"SECRET_T0_" not in durable
    assert b"query1.finance.yahoo.com" not in durable
