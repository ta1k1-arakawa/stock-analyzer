"""Independent read-only verification for V8D transport artifacts."""

from __future__ import annotations

import errno
import json
import re
import socket
import urllib.error
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.v8d_transport import (
    ACQUISITION_STAGES,
    ALL_STAGES,
    BACKOFF_SECONDS,
    CANONICAL_PARSER_CLASSIFIER_BLOB,
    CANONICAL_PARSER_CLASSIFIER_COMMIT,
    FROZEN_DESIGN_COMMIT,
    MAXIMUM_ATTEMPTS,
    MAXIMUM_RETRIES,
    NONRETRYABLE_CLASSES,
    READINESS_STAGES,
    RETRYABLE_CLASSES,
    RETRYABLE_HTTP_CODES,
    SENTINEL_END_EXCLUSIVE,
    SENTINEL_INDICES,
    SENTINEL_START,
    STUDY,
    canonical_json_bytes,
    canonical_sha256,
    classify_collector_reason,
    origin_guard_evidence,
)


class V8DAuditVerificationBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_FIELDS = frozenset({
    "attempt", "classification", "retryable", "concrete_exception_type", "http_code", "reason_type",
    "errno", "named_condition", "detector_evidence", "request_fingerprint", "request_url_sha256",
    "terminal_state",
})
_DOSSIER_FIELDS = frozenset({
    "schema_version", "study", "frozen_design_commit", "reviewed_production_implementation_commit",
    "canonical_parser_classifier_commit", "canonical_parser_classifier_blob", "logical_stage", "logical_block",
    "logical_coordinate", "window_start", "window_end_exclusive", "sentinel_indices", "attempts",
    "audit_artifact_self_hash",
})
_AGGREGATE_FIELDS = frozenset({
    "schema_version", "study", "frozen_design_commit", "reviewed_production_implementation_commit",
    "canonical_parser_classifier_commit", "canonical_parser_classifier_blob", "logical_stage", "logical_block",
    "result", "sentinel_indices", "window_start", "window_end_exclusive", "sentinel_count",
    "sentinel_pass_count", "request_count", "total_request_attempts", "retry_count", "retryable_attempt_count",
    "nonretryable_attempt_count", "terminal_classification_histogram", "attempt_classification_histogram",
    "attempt_count_histogram", "http_status_histogram", "audit_artifact_count", "audit_artifact_self_hash",
    "audit_evidence_complete", "no_missing_terminal_failure_evidence", "aggregate_self_hash",
})


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V8DAuditVerificationBlocked("V8D_AUDIT_JSON_INVALID") from error
    if not isinstance(value, dict):
        raise V8DAuditVerificationBlocked("V8D_AUDIT_OBJECT_INVALID")
    return value


def _require_hex(value: object, length: int, reason: str) -> None:
    pattern = _HEX40 if length == 40 else _HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise V8DAuditVerificationBlocked(reason)


def _derive_origin_condition(evidence: object) -> str | None:
    if not isinstance(evidence, Mapping):
        raise V8DAuditVerificationBlocked("V8D_ORIGIN_EVIDENCE_INVALID")
    required = {"detector_source", "input_is_string", "origin_parse_success", "scheme_https",
                "hostname_matches_expected", "credentials_absent", "port_allowed", "context"}
    if set(evidence) != required or evidence.get("detector_source") != "ORIGIN_GUARD":
        raise V8DAuditVerificationBlocked("V8D_ORIGIN_EVIDENCE_SCHEMA_INVALID")
    booleans = [evidence[key] for key in (
        "input_is_string", "origin_parse_success", "scheme_https", "hostname_matches_expected",
        "credentials_absent", "port_allowed",
    )]
    if any(type(value) is not bool for value in booleans):
        raise V8DAuditVerificationBlocked("V8D_ORIGIN_EVIDENCE_BOOLEAN_INVALID")
    if evidence["context"] not in {"REDIRECT_TARGET", "INITIAL_OR_FINAL_RESPONSE"}:
        raise V8DAuditVerificationBlocked("V8D_ORIGIN_CONTEXT_INVALID")
    if (not evidence["input_is_string"] or not evidence["origin_parse_success"]) and any(booleans[2:]):
        raise V8DAuditVerificationBlocked("V8D_ORIGIN_DOWNSTREAM_FILL_INVALID")
    invalid = not all(booleans)
    if not invalid:
        raise V8DAuditVerificationBlocked("V8D_ORIGIN_FAILURE_WITH_VALID_ORIGIN")
    return "UNTRUSTED_REDIRECT" if evidence["context"] == "REDIRECT_TARGET" else "RESPONSE_HOST_MISMATCH"


def _derive_named_condition(named: object, evidence: object) -> tuple[str, bool]:
    if not isinstance(named, str):
        raise V8DAuditVerificationBlocked("V8D_NAMED_CONDITION_MISSING")
    if isinstance(evidence, Mapping) and evidence.get("detector_source") == "ORIGIN_GUARD":
        derived = _derive_origin_condition(evidence)
        if named != derived:
            raise V8DAuditVerificationBlocked("V8D_NAMED_CONDITION_MISMATCH")
        return derived, False
    if named == "SYMBOL_MISMATCH":
        if evidence != {"expected_symbol_binding": False}:
            raise V8DAuditVerificationBlocked("V8D_SYMBOL_EVIDENCE_INVALID")
        return named, False
    if named == "DATA_QUALITY_GATE_FAILURE":
        if not isinstance(evidence, Mapping) or set(evidence) != {"nonempty_timestamp", "valid_price_row_count", "trading_date_fields_valid"}:
            raise V8DAuditVerificationBlocked("V8D_QUALITY_EVIDENCE_SCHEMA_INVALID")
        if type(evidence["nonempty_timestamp"]) is not bool or type(evidence["trading_date_fields_valid"]) is not bool:
            raise V8DAuditVerificationBlocked("V8D_QUALITY_EVIDENCE_BOOLEAN_INVALID")
        if type(evidence["valid_price_row_count"]) is not int or evidence["valid_price_row_count"] < 0:
            raise V8DAuditVerificationBlocked("V8D_QUALITY_EVIDENCE_COUNT_INVALID")
        if evidence["nonempty_timestamp"] and evidence["valid_price_row_count"] > 0 and evidence["trading_date_fields_valid"]:
            raise V8DAuditVerificationBlocked("V8D_QUALITY_FAILURE_WITH_VALID_EVIDENCE")
        return named, False
    if named == "PARSER_SCHEMA_FAILURE":
        if isinstance(evidence, Mapping) and set(evidence) == {"parser_schema_valid", "canonical_collector_reason_code_or_family"}:
            if evidence["parser_schema_valid"] is not False:
                raise V8DAuditVerificationBlocked("V8D_PARSER_EVIDENCE_SCHEMA_INVALID")
            reason = evidence["canonical_collector_reason_code_or_family"]
            classification, retryable = classify_collector_reason(reason)
            if classification != named or retryable:
                raise V8DAuditVerificationBlocked("V8D_PARSER_REASON_CLASSIFICATION_MISMATCH")
            return named, False
        if not isinstance(evidence, Mapping) or evidence.get("detector_source") != "CANONICAL_COLLECTOR_REASON":
            raise V8DAuditVerificationBlocked("V8D_PARSER_EVIDENCE_SCHEMA_INVALID")
        reason = evidence.get("canonical_collector_reason_code_or_family")
        classification, retryable = classify_collector_reason(reason)
        if classification not in {"PARSER_SCHEMA_FAILURE", "RESPONSE_HOST_MISMATCH", "SYMBOL_MISMATCH"} or retryable:
            raise V8DAuditVerificationBlocked("V8D_PARSER_REASON_INVALID")
        if classification != named:
            raise V8DAuditVerificationBlocked("V8D_PARSER_REASON_CLASSIFICATION_MISMATCH")
        return named, False
    raise V8DAuditVerificationBlocked("V8D_NAMED_EVIDENCE_INVALID")


def _derive_transport_classification(record: Mapping[str, Any]) -> tuple[str, bool | None]:
    classification = record.get("classification")
    if classification == "SUCCESS":
        if record.get("retryable") is not None or record.get("concrete_exception_type") is not None:
            raise V8DAuditVerificationBlocked("V8D_SUCCESS_METADATA_INVALID")
        if any(record.get(key) is not None for key in ("http_code", "reason_type", "errno", "named_condition", "detector_evidence")):
            raise V8DAuditVerificationBlocked("V8D_SUCCESS_METADATA_INVALID")
        return "SUCCESS", None
    named = record.get("named_condition")
    evidence = record.get("detector_evidence")
    if named is not None:
        if record.get("concrete_exception_type") != "V8DNamedFailure":
            raise V8DAuditVerificationBlocked("V8D_NAMED_EXCEPTION_TYPE_INVALID")
        return _derive_named_condition(named, evidence)
    evidence_source = evidence.get("detector_source") if isinstance(evidence, Mapping) else None
    if evidence_source == "CANONICAL_COLLECTOR_REASON":
        if record.get("concrete_exception_type") != "V7YahooCollectorBlocked":
            raise V8DAuditVerificationBlocked("V8D_COLLECTOR_EXCEPTION_TYPE_INVALID")
        reason = evidence.get("canonical_collector_reason_code_or_family")
        derived, retryable = classify_collector_reason(reason)
        return derived, retryable
    concrete = record.get("concrete_exception_type")
    code = record.get("http_code")
    if code is not None:
        if type(code) is not int:
            raise V8DAuditVerificationBlocked("V8D_HTTP_CODE_INVALID")
        if concrete != "HTTPError":
            raise V8DAuditVerificationBlocked("V8D_HTTP_EXCEPTION_TYPE_INVALID")
        if record.get("reason_type") is not None or record.get("errno") is not None:
            raise V8DAuditVerificationBlocked("V8D_HTTP_METADATA_INVALID")
        return f"HTTP_{code}", code in RETRYABLE_HTTP_CODES
    reason_type = record.get("reason_type")
    observed_errno = record.get("errno")
    if observed_errno is not None and type(observed_errno) is not int:
        raise V8DAuditVerificationBlocked("V8D_ERRNO_INVALID")
    if concrete in {"TimeoutError", "socket.timeout"}:
        if reason_type != concrete or observed_errno is not None:
            raise V8DAuditVerificationBlocked("V8D_TIMEOUT_METADATA_INVALID")
        return "NETWORK_TIMEOUT", True
    if concrete == "ConnectionResetError" or observed_errno == errno.ECONNRESET:
        if reason_type != concrete:
            raise V8DAuditVerificationBlocked("V8D_RESET_METADATA_INVALID")
        return "CONNECTION_RESET", True
    if concrete == "URLError":
        if reason_type in {"TimeoutError", "socket.timeout"}:
            return "NETWORK_TIMEOUT", True
        if reason_type == "gaierror":
            if observed_errno == socket.EAI_AGAIN:
                return "TEMPORARY_DNS_FAILURE", True
            return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
        if reason_type in {"ConnectionResetError", "OSError"} and observed_errno == errno.ECONNRESET:
            return "CONNECTION_RESET", True
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
    if concrete == "gaierror":
        if reason_type != concrete:
            raise V8DAuditVerificationBlocked("V8D_DNS_METADATA_INVALID")
        if observed_errno == socket.EAI_AGAIN:
            return "TEMPORARY_DNS_FAILURE", True
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
    if not isinstance(concrete, str) or reason_type != concrete:
        raise V8DAuditVerificationBlocked("V8D_UNKNOWN_METADATA_INVALID")
    return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False


def verify_dossier(path: str | Path, *, expected_reviewed_implementation_commit: str | None = None,
                   expected_stage: str | None = None) -> dict[str, Any]:
    dossier = _read_json(path)
    if set(dossier) != _DOSSIER_FIELDS:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_SCHEMA_INVALID")
    if dossier["schema_version"] != "V8D_PRIVATE_TRANSPORT_AUDIT_V1" or dossier["study"] != STUDY:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_STUDY_SCHEMA_INVALID")
    if dossier["frozen_design_commit"] != FROZEN_DESIGN_COMMIT:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_DESIGN_BINDING_MISMATCH")
    _require_hex(dossier["reviewed_production_implementation_commit"], 40, "V8D_DOSSIER_IMPLEMENTATION_SHA_INVALID")
    if expected_reviewed_implementation_commit is not None and dossier["reviewed_production_implementation_commit"] != expected_reviewed_implementation_commit:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_IMPLEMENTATION_BINDING_MISMATCH")
    if dossier["canonical_parser_classifier_commit"] != CANONICAL_PARSER_CLASSIFIER_COMMIT or dossier["canonical_parser_classifier_blob"] != CANONICAL_PARSER_CLASSIFIER_BLOB:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_CLASSIFIER_BINDING_MISMATCH")
    stage = dossier["logical_stage"]
    if stage not in ALL_STAGES or (expected_stage is not None and stage != expected_stage):
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_STAGE_INVALID")
    expected_block = "T1C" if stage.startswith("T1C") else "T2"
    if dossier["logical_block"] != expected_block or type(dossier["logical_coordinate"]) is not int or dossier["logical_coordinate"] < 0:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_LOGICAL_BINDING_INVALID")
    if stage in READINESS_STAGES:
        if dossier["sentinel_indices"] != list(SENTINEL_INDICES) or dossier["window_start"] != SENTINEL_START or dossier["window_end_exclusive"] != SENTINEL_END_EXCLUSIVE:
            raise V8DAuditVerificationBlocked("V8D_DOSSIER_SENTINEL_BINDING_INVALID")
        if dossier["logical_coordinate"] not in SENTINEL_INDICES:
            raise V8DAuditVerificationBlocked("V8D_DOSSIER_SENTINEL_COORDINATE_INVALID")
    elif dossier["sentinel_indices"] != []:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_ACQUISITION_SENTINEL_INVALID")
    attempts = dossier["attempts"]
    if not isinstance(attempts, list) or not attempts or len(attempts) > MAXIMUM_ATTEMPTS:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_ATTEMPTS_INVALID")
    if dossier["audit_artifact_self_hash"] != canonical_sha256({key: value for key, value in dossier.items() if key != "audit_artifact_self_hash"}):
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_SELF_HASH_MISMATCH")
    fingerprints: set[str] = set()
    urls: set[str] = set()
    derived: list[tuple[str, bool | None]] = []
    for expected_attempt, record in enumerate(attempts, start=1):
        if not isinstance(record, Mapping) or set(record) != _ATTEMPT_FIELDS or record["attempt"] != expected_attempt:
            raise V8DAuditVerificationBlocked("V8D_DOSSIER_ATTEMPT_ORDER_INVALID")
        _require_hex(record["request_fingerprint"], 64, "V8D_DOSSIER_FINGERPRINT_INVALID")
        _require_hex(record["request_url_sha256"], 64, "V8D_DOSSIER_URL_HASH_INVALID")
        fingerprints.add(record["request_fingerprint"])
        urls.add(record["request_url_sha256"])
        if record["terminal_state"] not in {"SUCCESS", "RETRYABLE_FAILURE", "TERMINAL_FAILURE"}:
            raise V8DAuditVerificationBlocked("V8D_DOSSIER_TERMINAL_STATE_INVALID")
        actual = _derive_transport_classification(record)
        if actual[0] != record["classification"] or actual[1] != record["retryable"]:
            raise V8DAuditVerificationBlocked("V8D_DOSSIER_CLASSIFICATION_MISMATCH")
        derived.append(actual)
    if len(fingerprints) != 1 or len(urls) != 1:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_REQUEST_BINDING_MISMATCH")
    for index, (_, retryable) in enumerate(derived[:-1]):
        if retryable is not True or attempts[index]["terminal_state"] != "RETRYABLE_FAILURE":
            raise V8DAuditVerificationBlocked("V8D_DOSSIER_RETRY_POLICY_INVALID")
    final_record = attempts[-1]
    if final_record["terminal_state"] not in {"SUCCESS", "TERMINAL_FAILURE"}:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_MISSING_TERMINAL_EVIDENCE")
    if final_record["terminal_state"] == "SUCCESS" and final_record["classification"] != "SUCCESS":
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_SUCCESS_TERMINAL_INVALID")
    if final_record["terminal_state"] == "TERMINAL_FAILURE" and final_record["classification"] == "SUCCESS":
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_FAILURE_TERMINAL_INVALID")
    if len(attempts) - 1 > MAXIMUM_RETRIES:
        raise V8DAuditVerificationBlocked("V8D_DOSSIER_RETRY_COUNT_INVALID")
    return dossier


def _histogram(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def verify_aggregate(aggregate_path: str | Path, dossier_paths: Sequence[str | Path], *,
                     expected_reviewed_implementation_commit: str | None = None,
                     expected_stage: str | None = None) -> dict[str, Any]:
    aggregate = _read_json(aggregate_path)
    if set(aggregate) != _AGGREGATE_FIELDS:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_SCHEMA_INVALID")
    if aggregate["schema_version"] != "V8D_PUBLIC_TRANSPORT_AGGREGATE_V1" or aggregate["study"] != STUDY:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_STUDY_SCHEMA_INVALID")
    if aggregate["aggregate_self_hash"] != canonical_sha256({key: value for key, value in aggregate.items() if key != "aggregate_self_hash"}):
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_SELF_HASH_MISMATCH")
    if aggregate["frozen_design_commit"] != FROZEN_DESIGN_COMMIT or aggregate["canonical_parser_classifier_commit"] != CANONICAL_PARSER_CLASSIFIER_COMMIT or aggregate["canonical_parser_classifier_blob"] != CANONICAL_PARSER_CLASSIFIER_BLOB:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_PROVENANCE_MISMATCH")
    if expected_stage is not None and aggregate["logical_stage"] != expected_stage:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_STAGE_MISMATCH")
    if aggregate["logical_stage"] not in ALL_STAGES:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_STAGE_INVALID")
    if aggregate["logical_block"] != ("T1C" if aggregate["logical_stage"].startswith("T1C") else "T2"):
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_BLOCK_INVALID")
    if aggregate["logical_stage"] in READINESS_STAGES:
        if aggregate["sentinel_indices"] != list(SENTINEL_INDICES) or aggregate["window_start"] != SENTINEL_START or aggregate["window_end_exclusive"] != SENTINEL_END_EXCLUSIVE:
            raise V8DAuditVerificationBlocked("V8D_AGGREGATE_SENTINEL_BINDING_INVALID")
    elif aggregate["sentinel_indices"] != []:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_ACQUISITION_SENTINEL_INVALID")
    _require_hex(aggregate["reviewed_production_implementation_commit"], 40, "V8D_AGGREGATE_IMPLEMENTATION_SHA_INVALID")
    if expected_reviewed_implementation_commit is not None and aggregate["reviewed_production_implementation_commit"] != expected_reviewed_implementation_commit:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_IMPLEMENTATION_MISMATCH")
    dossiers = [verify_dossier(path, expected_reviewed_implementation_commit=aggregate["reviewed_production_implementation_commit"], expected_stage=aggregate["logical_stage"]) for path in dossier_paths]
    if len(dossiers) != aggregate["audit_artifact_count"] or not dossiers:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_DOSSIER_COUNT_MISMATCH")
    if any(dossier["logical_block"] != aggregate["logical_block"] for dossier in dossiers):
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_BLOCK_MISMATCH")
    if any(dossier["window_start"] != aggregate["window_start"] or dossier["window_end_exclusive"] != aggregate["window_end_exclusive"] or dossier["sentinel_indices"] != aggregate["sentinel_indices"] for dossier in dossiers):
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_WINDOW_BINDING_MISMATCH")
    if aggregate["logical_stage"] in READINESS_STAGES:
        coordinates = [dossier["logical_coordinate"] for dossier in dossiers]
        if len(dossiers) != 3 or len(set(coordinates)) != 3 or sorted(coordinates) != list(SENTINEL_INDICES):
            raise V8DAuditVerificationBlocked("V8D_AGGREGATE_SENTINEL_COORDINATE_SET_INVALID")
    artifact_hash = canonical_sha256(sorted(dossier["audit_artifact_self_hash"] for dossier in dossiers))
    if aggregate["audit_artifact_self_hash"] != artifact_hash:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_ARTIFACT_HASH_MISMATCH")
    attempts = [record for dossier in dossiers for record in dossier["attempts"]]
    terminal = [dossier["attempts"][-1] for dossier in dossiers]
    derived_result = "PASS" if all(record["classification"] == "SUCCESS" for record in terminal) else "BLOCK"
    if aggregate["result"] != derived_result:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_RESULT_MISMATCH")
    expected = {
        "request_count": len(dossiers),
        "total_request_attempts": len(attempts),
        "retry_count": len(attempts) - len(dossiers),
        "retryable_attempt_count": sum(record["retryable"] is True for record in attempts),
        "nonretryable_attempt_count": sum(record["retryable"] is False for record in attempts),
        "terminal_classification_histogram": _histogram(record["classification"] for record in terminal),
        "attempt_classification_histogram": _histogram(record["classification"] for record in attempts),
        "attempt_count_histogram": _histogram(str(len(dossier["attempts"])) for dossier in dossiers),
        "http_status_histogram": _histogram(str(record["http_code"]) for record in attempts if record["http_code"] is not None),
        "sentinel_count": 3 if aggregate["logical_stage"] in READINESS_STAGES else 0,
        "sentinel_pass_count": sum(record["classification"] == "SUCCESS" for record in terminal) if aggregate["logical_stage"] in READINESS_STAGES else 0,
    }
    for key, value in expected.items():
        if aggregate[key] != value:
            raise V8DAuditVerificationBlocked("V8D_AGGREGATE_DERIVED_VALUE_MISMATCH:" + key)
    if aggregate["result"] == "PASS" and aggregate["sentinel_pass_count"] != aggregate["sentinel_count"]:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_PASS_SENTINEL_MISMATCH")
    if aggregate["audit_evidence_complete"] is not True or aggregate["no_missing_terminal_failure_evidence"] is not True:
        raise V8DAuditVerificationBlocked("V8D_AGGREGATE_EVIDENCE_FLAGS_INVALID")
    return aggregate


__all__ = ["V8DAuditVerificationBlocked", "verify_aggregate", "verify_dossier"]
