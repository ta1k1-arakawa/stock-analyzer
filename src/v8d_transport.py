"""V8D durable transport classification, retry, and audit persistence.

This module is deliberately transport-only.  It does not consume a human
gate, read a partition, or start a production execution.  Callers provide a
single-request callable; the wrapper makes the frozen retry policy and the
per-attempt durability boundary impossible to bypass.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from src.v7_yahoo_collector import V7YahooCollectorBlocked, build_chart_request, fetch_chart_once
from src.v8d_human_gate_consumption import STAGE_GATE as _GATE_STAGE_MAP


STUDY = "V8D_HISTORICAL_RESEARCH"
FROZEN_DESIGN_COMMIT = "eda657cde2383718d986c4c4bfaae794784fe04d"
CANONICAL_PARSER_CLASSIFIER_COMMIT = "28e281c3ee30d6b4c2f981c5da3ddc983c09724d"
CANONICAL_PARSER_CLASSIFIER_BLOB = "76b57b077f3214e666ff9dc06d9c224afc16df9f"
YAHOO_HOST = "query1.finance.yahoo.com"

MAXIMUM_ATTEMPTS = 3
MAXIMUM_RETRIES = 2
BACKOFF_SECONDS: tuple[int, ...] = (5, 30)
JITTER = False
RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
NONRETRYABLE_HTTP_CODES = frozenset({400, 401, 403, 404, 410, 422})

RETRYABLE_CLASSES = frozenset({
    "NETWORK_TIMEOUT",
    "CONNECTION_RESET",
    "TEMPORARY_DNS_FAILURE",
    "HTTP_408",
    "HTTP_425",
    "HTTP_429",
    "HTTP_500",
    "HTTP_502",
    "HTTP_503",
    "HTTP_504",
})
NONRETRYABLE_CLASSES = frozenset({
    "HTTP_400",
    "HTTP_401",
    "HTTP_403",
    "HTTP_404",
    "HTTP_410",
    "HTTP_422",
    "UNTRUSTED_REDIRECT",
    "RESPONSE_HOST_MISMATCH",
    "PARSER_SCHEMA_FAILURE",
    "SYMBOL_MISMATCH",
    "DATA_QUALITY_GATE_FAILURE",
    "UNKNOWN_FAIL_CLOSED_NONRETRYABLE",
})
ALL_STAGES = frozenset({
    "T1C_TRANSPORT_READINESS",
    "T1C_RAW_ACQUISITION",
    "T2_TRANSPORT_READINESS",
    "T2_RAW_ACQUISITION",
})
READINESS_STAGES = frozenset({"T1C_TRANSPORT_READINESS", "T2_TRANSPORT_READINESS"})
ACQUISITION_STAGES = frozenset({"T1C_RAW_ACQUISITION", "T2_RAW_ACQUISITION"})
SENTINEL_INDICES = (0, 149, 299)
SENTINEL_START = "2025-12-01"
SENTINEL_END_EXCLUSIVE = "2025-12-08"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HTTP_REASON = re.compile(r"^HTTP_STATUS_(None|-?(?:0|[1-9][0-9]*))$")
_INVALID_REASON_LITERAL = frozenset({
    "EMPTY_TICKER", "INVALID_REQUEST_DATE_ORDER", "RESPONSE_HOST_MISMATCH",
    "TIMESTAMP_INVALID", "PAYLOAD_JSON_INVALID", "PAYLOAD_ROOT_INVALID",
    "CHART_ERROR", "CHART_RESULT_INVALID", "INDICATORS_MISSING",
    "SPLIT_RATIO_INVALID", "EVENTS_INVALID", "SPLITS_INVALID",
    "SPLIT_EVENT_INVALID", "SPLIT_OUT_OF_REQUEST_WINDOW", "DUPLICATE_SPLIT_EVENT",
    "SPLIT_NUMERATOR_DENOMINATOR_MISSING", "SPLIT_NUMERATOR_DENOMINATOR_INVALID",
    "SPLIT_RATIO_MISMATCH", "METADATA_MISSING", "SYMBOL_MISMATCH",
    "TIMESTAMP_MISSING", "OUT_OF_REQUEST_WINDOW", "DUPLICATE_TRADING_DATE",
    "RESPONSE_BYTES_INVALID",
})
_INVALID_REASON_FAMILIES = (
    re.compile(r"^INVALID_DATE:(?:start|end_exclusive)$"),
    re.compile(r"^INDICATOR_SECTION_INVALID:(?:quote|adjclose)$"),
    re.compile(r"^ARRAY_LENGTH_MISMATCH:(?:open|high|low|close|volume|adjclose)$"),
)

T = TypeVar("T")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_hex(value: object, length: int, reason: str) -> str:
    pattern = _HEX40 if length == 40 else _HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise V8DTransportBlocked(reason)
    return value


_GATE_BINDING_FIELDS = frozenset({
    "human_gate", "gate_receipt_key_sha256", "gate_receipt_bytes_sha256", "authorization_identity_sha256",
})


def _require_gate_binding(gate_binding: Mapping[str, Any], *, stage: str) -> dict[str, str]:
    """Structural validation only: this transport layer deliberately does
    not consume a human gate and does not know how a binding was produced
    -- it only requires the exact safe shape `src.v8d_human_gate_
    consumption.V8DGateReceiptBinding` publishes, and that the bound
    ``human_gate`` is the frozen gate for ``stage`` (never any other
    stage/gate combination). The actual gate-consumption/receipt evidence
    is independently re-verified later by `src.v8d_audit`, which never
    trusts these dossier fields on their own."""
    if not isinstance(gate_binding, Mapping) or set(gate_binding) != _GATE_BINDING_FIELDS:
        raise V8DTransportBlocked("V8D_GATE_BINDING_SCHEMA_INVALID")
    human_gate = gate_binding["human_gate"]
    if not isinstance(human_gate, str) or human_gate not in _GATE_STAGE_MAP.values():
        raise V8DTransportBlocked("V8D_GATE_BINDING_HUMAN_GATE_INVALID")
    if _GATE_STAGE_MAP.get(stage) != human_gate:
        raise V8DTransportBlocked("V8D_GATE_BINDING_STAGE_MISMATCH")
    return {
        "human_gate": human_gate,
        "gate_receipt_key_sha256": _require_hex(gate_binding["gate_receipt_key_sha256"], 64, "V8D_GATE_BINDING_HASH_INVALID"),
        "gate_receipt_bytes_sha256": _require_hex(gate_binding["gate_receipt_bytes_sha256"], 64, "V8D_GATE_BINDING_HASH_INVALID"),
        "authorization_identity_sha256": _require_hex(gate_binding["authorization_identity_sha256"], 64, "V8D_GATE_BINDING_HASH_INVALID"),
    }


class V8DTransportBlocked(RuntimeError):
    """Fail-closed V8D transport or audit error."""

    def __init__(self, reason: str, *, no_next_request: bool = False, no_success_return: bool = False,
                 no_aggregate_pass: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.no_next_request = no_next_request
        self.no_success_return = no_success_return
        self.no_aggregate_pass = no_aggregate_pass


class V8DAuditPersistenceBlocked(V8DTransportBlocked):
    """Raised when an attempt cannot be durably persisted."""

    def __init__(self, reason: str = "V8D_AUDIT_PERSISTENCE_FAILED") -> None:
        super().__init__(reason, no_next_request=True, no_success_return=True, no_aggregate_pass=True)


class V8DNamedFailure(RuntimeError):
    """A named nonretryable condition with only privacy-safe detector evidence."""

    def __init__(self, condition: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        if condition not in NONRETRYABLE_CLASSES:
            raise V8DTransportBlocked("V8D_UNKNOWN_NAMED_CONDITION")
        super().__init__(condition)
        self.condition = condition
        self.evidence = dict(evidence) if evidence is not None else None


def _is_timeout(value: object) -> bool:
    return isinstance(value, (TimeoutError, socket.timeout))


def _connection_reset(value: object) -> bool:
    return isinstance(value, ConnectionResetError) or (
        isinstance(value, OSError) and getattr(value, "errno", None) == errno.ECONNRESET
    )


def _temporary_dns(value: object) -> bool:
    return isinstance(value, socket.gaierror) and getattr(value, "errno", None) == socket.EAI_AGAIN


def _permanent_dns(value: object) -> bool:
    return isinstance(value, socket.gaierror) and not _temporary_dns(value)


def classify_named_condition(name: str) -> tuple[str, bool]:
    if name not in NONRETRYABLE_CLASSES:
        raise V8DTransportBlocked("V8D_UNKNOWN_NAMED_CONDITION")
    return name, False


def classify_transport_exception(error: BaseException) -> tuple[str, bool]:
    """Classify using only concrete types and numeric attributes."""
    if isinstance(error, V8DNamedFailure):
        return classify_named_condition(error.condition)
    if isinstance(error, V7YahooCollectorBlocked):
        return classify_collector_reason(error.reason)
    if isinstance(error, urllib.error.HTTPError):
        code = error.code
        if isinstance(code, int) and not isinstance(code, bool):
            return f"HTTP_{code}", code in RETRYABLE_HTTP_CODES
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
    if _is_timeout(error):
        return "NETWORK_TIMEOUT", True
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        if _is_timeout(reason):
            return "NETWORK_TIMEOUT", True
        if _temporary_dns(reason):
            return "TEMPORARY_DNS_FAILURE", True
        if _connection_reset(reason):
            return "CONNECTION_RESET", True
        if _permanent_dns(reason):
            return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
    if _temporary_dns(error):
        return "TEMPORARY_DNS_FAILURE", True
    if _permanent_dns(error):
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
    if _connection_reset(error):
        return "CONNECTION_RESET", True
    return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False


def _valid_canonical_reason(reason: object) -> bool:
    if not isinstance(reason, str):
        return False
    if reason in _INVALID_REASON_LITERAL:
        return True
    return any(pattern.fullmatch(reason) for pattern in _INVALID_REASON_FAMILIES) or bool(_HTTP_REASON.fullmatch(reason))


def classify_collector_reason(reason: object) -> tuple[str, bool]:
    """Map the exact frozen V7 reason grammar; never use message heuristics."""
    if not _valid_canonical_reason(reason):
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
    if reason == "SYMBOL_MISMATCH":
        return "SYMBOL_MISMATCH", False
    if reason == "RESPONSE_HOST_MISMATCH":
        return "RESPONSE_HOST_MISMATCH", False
    status_match = _HTTP_REASON.fullmatch(reason)
    if status_match:
        status_text = status_match.group(1)
        if status_text == "None":
            return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
        status = int(status_text)
        return f"HTTP_{status}", status in RETRYABLE_HTTP_CODES
    return "PARSER_SCHEMA_FAILURE", False


def _safe_errno(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _classification_metadata(error: BaseException, classification: str) -> dict[str, Any]:
    if isinstance(error, urllib.error.HTTPError):
        return {"concrete_exception_type": type(error).__name__, "http_code": error.code,
                "reason_type": None, "errno": None, "named_condition": None,
                "detector_evidence": None}
    if isinstance(error, V8DNamedFailure):
        return {"concrete_exception_type": type(error).__name__, "http_code": None,
                "reason_type": None, "errno": None, "named_condition": error.condition,
                "detector_evidence": error.evidence}
    if isinstance(error, V7YahooCollectorBlocked):
        return {"concrete_exception_type": type(error).__name__, "http_code": None,
                "reason_type": None, "errno": None, "named_condition": None,
                "detector_evidence": {"detector_source": "CANONICAL_COLLECTOR_REASON",
                                       "canonical_collector_reason_code_or_family": error.reason}}
    reason = getattr(error, "reason", None) if isinstance(error, urllib.error.URLError) else error
    return {"concrete_exception_type": type(error).__name__, "http_code": None,
            "reason_type": type(reason).__name__, "errno": _safe_errno(getattr(reason, "errno", None)),
            "named_condition": None, "detector_evidence": None}


def origin_guard_evidence(value: object, *, context: str) -> dict[str, Any]:
    if context not in {"REDIRECT_TARGET", "INITIAL_OR_FINAL_RESPONSE"}:
        raise V8DTransportBlocked("V8D_ORIGIN_CONTEXT_INVALID")
    input_is_string = isinstance(value, str)
    parse_success = False
    scheme_https = False
    hostname_matches = False
    credentials_absent = False
    port_allowed = False
    if input_is_string:
        try:
            parsed = urllib.parse.urlparse(value)
            port = parsed.port
        except ValueError:
            pass
        else:
            parse_success = True
            scheme_https = parsed.scheme == "https"
            hostname_matches = parsed.hostname == YAHOO_HOST
            credentials_absent = parsed.username is None and parsed.password is None
            port_allowed = port in (None, 443)
    if not input_is_string or not parse_success:
        scheme_https = hostname_matches = credentials_absent = port_allowed = False
    return {
        "detector_source": "ORIGIN_GUARD",
        "input_is_string": input_is_string,
        "origin_parse_success": parse_success,
        "scheme_https": scheme_https,
        "hostname_matches_expected": hostname_matches,
        "credentials_absent": credentials_absent,
        "port_allowed": port_allowed,
        "context": context,
    }


def require_trusted_origin(value: object, *, context: str) -> dict[str, Any]:
    evidence = origin_guard_evidence(value, context=context)
    valid = all(evidence[key] is True for key in (
        "input_is_string", "origin_parse_success", "scheme_https",
        "hostname_matches_expected", "credentials_absent", "port_allowed",
    ))
    if not valid:
        condition = "UNTRUSTED_REDIRECT" if context == "REDIRECT_TARGET" else "RESPONSE_HOST_MISMATCH"
        raise V8DNamedFailure(condition, evidence=evidence)
    return evidence


class V8DTrustedYahooRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        require_trusted_origin(newurl, context="REDIRECT_TARGET")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def default_trusted_yahoo_opener(request_obj: Any) -> Any:
    require_trusted_origin(getattr(request_obj, "full_url", None), context="INITIAL_OR_FINAL_RESPONSE")
    return urllib.request.build_opener(V8DTrustedYahooRedirectHandler()).open(request_obj)


@dataclass(frozen=True)
class V8DRequestContext:
    logical_stage: str
    logical_block: str
    logical_coordinate: int
    window_start: str
    window_end_exclusive: str
    request_fingerprint: str
    request_url_sha256: str
    sentinel_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.logical_stage not in ALL_STAGES:
            raise V8DTransportBlocked("V8D_LOGICAL_STAGE_INVALID")
        expected_block = "T1C" if self.logical_stage.startswith("T1C") else "T2"
        if self.logical_block != expected_block:
            raise V8DTransportBlocked("V8D_LOGICAL_BLOCK_INVALID")
        if type(self.logical_coordinate) is not int or self.logical_coordinate < 0:
            raise V8DTransportBlocked("V8D_LOGICAL_COORDINATE_INVALID")
        if not isinstance(self.window_start, str) or not isinstance(self.window_end_exclusive, str):
            raise V8DTransportBlocked("V8D_REQUEST_WINDOW_INVALID")
        _require_hex(self.request_fingerprint, 64, "V8D_REQUEST_FINGERPRINT_INVALID")
        _require_hex(self.request_url_sha256, 64, "V8D_REQUEST_URL_HASH_INVALID")
        if self.logical_stage in READINESS_STAGES:
            if self.sentinel_indices != SENTINEL_INDICES or self.window_start != SENTINEL_START or self.window_end_exclusive != SENTINEL_END_EXCLUSIVE:
                raise V8DTransportBlocked("V8D_SENTINEL_WINDOW_BINDING_INVALID")
        elif self.sentinel_indices is not None:
            raise V8DTransportBlocked("V8D_ACQUISITION_SENTINEL_BINDING_INVALID")


@dataclass(frozen=True)
class V8DRequestPlan:
    request_fn: Callable[[], Any]
    request_fingerprint: str
    request_url_sha256: str


def make_request_fingerprint(*, logical_stage: str, logical_block: str, logical_coordinate: int,
                             window_start: str, window_end_exclusive: str,
                             request_parameters: Mapping[str, Any] | None = None) -> str:
    material = {
        "logical_stage": logical_stage,
        "logical_block": logical_block,
        "logical_coordinate": logical_coordinate,
        "window_start": window_start,
        "window_end_exclusive": window_end_exclusive,
        "provider": "Yahoo",
        "host": YAHOO_HOST,
        "request_parameters": dict(request_parameters or {}),
    }
    return canonical_sha256(material)


def sha256_url(value: object) -> str:
    if not isinstance(value, str):
        raise V8DTransportBlocked("V8D_REQUEST_URL_INVALID")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_yahoo_request_plan(*, logical_stage: str, logical_block: str, logical_coordinate: int,
                             ticker: object, request_start: str, request_end_exclusive: str,
                             opener: Callable[[Any], Any] = default_trusted_yahoo_opener,
                             request_parameters: Mapping[str, Any] | None = None,
                             validate_result: Callable[[Mapping[str, Any]], Any] | None = None) -> V8DRequestPlan:
    """Create a V7-compatible Yahoo request callable without persisting its URL.

    The ticker remains only in the caller's closure and in the transient V7
    parser call.  The durable context contains a coordinate and URL digest.
    """
    request_object = build_chart_request(ticker, request_start, request_end_exclusive)
    request_url = getattr(request_object, "full_url", None)
    request_url_hash = sha256_url(request_url)
    fingerprint = make_request_fingerprint(
        logical_stage=logical_stage,
        logical_block=logical_block,
        logical_coordinate=logical_coordinate,
        window_start=request_start,
        window_end_exclusive=request_end_exclusive,
        request_parameters=request_parameters,
    )

    def guarded_opener(request_obj: Any) -> Any:
        require_trusted_origin(getattr(request_obj, "full_url", None), context="INITIAL_OR_FINAL_RESPONSE")
        response = opener(request_obj)
        try:
            require_trusted_origin(getattr(response, "url", None), context="INITIAL_OR_FINAL_RESPONSE")
        except V8DNamedFailure:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise
        return response

    def request_fn() -> Any:
        parsed = fetch_chart_once(
            ticker, request_start, request_end_exclusive, opener=guarded_opener
        )
        if validate_result is not None:
            validate_result(parsed)
        return parsed

    return V8DRequestPlan(request_fn=request_fn, request_fingerprint=fingerprint, request_url_sha256=request_url_hash)


def require_nonempty_quality(parsed: Mapping[str, Any]) -> None:
    rows = parsed.get("valid_price_rows")
    timestamps_nonempty = bool(rows)
    row_count = len(rows) if isinstance(rows, list) else 0
    fields_valid = isinstance(rows, list) and all(
        isinstance(row, Mapping) and isinstance(row.get("trading_date"), str) and bool(row["trading_date"])
        for row in rows
    )
    if not timestamps_nonempty or row_count == 0 or not fields_valid:
        raise V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
            "nonempty_timestamp": timestamps_nonempty,
            "valid_price_row_count": row_count,
            "trading_date_fields_valid": fields_valid,
        })


def _attempt_record(context: V8DRequestContext, attempt: int, classification: str,
                   retryable: bool | None, metadata: Mapping[str, Any], terminal_state: str) -> dict[str, Any]:
    evidence = metadata["detector_evidence"]
    if evidence is not None:
        try:
            _validate_detector_evidence(evidence)
        except V8DTransportBlocked as error:
            raise V8DAuditPersistenceBlocked("V8D_ATTEMPT_EVIDENCE_INVALID") from error
    return {
        "attempt": attempt,
        "classification": classification,
        "retryable": retryable,
        "concrete_exception_type": metadata["concrete_exception_type"],
        "http_code": metadata["http_code"],
        "reason_type": metadata["reason_type"],
        "errno": metadata["errno"],
        "named_condition": metadata["named_condition"],
        "detector_evidence": metadata["detector_evidence"],
        "request_fingerprint": context.request_fingerprint,
        "request_url_sha256": context.request_url_sha256,
        "terminal_state": terminal_state,
}


_ATTEMPT_FIELDS = frozenset({
    "attempt", "classification", "retryable", "concrete_exception_type", "http_code", "reason_type",
    "errno", "named_condition", "detector_evidence", "request_fingerprint", "request_url_sha256",
    "terminal_state",
})


def _validate_detector_evidence(value: object) -> None:
    if not isinstance(value, Mapping):
        raise V8DTransportBlocked("V8D_DETECTOR_EVIDENCE_SCHEMA_INVALID")
    source = value.get("detector_source")
    if source == "ORIGIN_GUARD":
        required = {"detector_source", "input_is_string", "origin_parse_success", "scheme_https",
                    "hostname_matches_expected", "credentials_absent", "port_allowed", "context"}
        if set(value) != required or value["context"] not in {"REDIRECT_TARGET", "INITIAL_OR_FINAL_RESPONSE"}:
            raise V8DTransportBlocked("V8D_ORIGIN_EVIDENCE_SCHEMA_INVALID")
        if any(type(value[key]) is not bool for key in required - {"detector_source", "context"}):
            raise V8DTransportBlocked("V8D_ORIGIN_EVIDENCE_BOOLEAN_INVALID")
        if (not value["input_is_string"] or not value["origin_parse_success"]) and any(value[key] for key in (
            "scheme_https", "hostname_matches_expected", "credentials_absent", "port_allowed")):
            raise V8DTransportBlocked("V8D_ORIGIN_DOWNSTREAM_FILL_INVALID")
        return
    if source == "CANONICAL_COLLECTOR_REASON":
        if set(value) != {"detector_source", "canonical_collector_reason_code_or_family"} or not _valid_canonical_reason(value["canonical_collector_reason_code_or_family"]):
            raise V8DTransportBlocked("V8D_PARSER_EVIDENCE_SCHEMA_INVALID")
        return
    if set(value) == {"expected_symbol_binding"} and value["expected_symbol_binding"] is False:
        return
    if set(value) == {"parser_schema_valid", "canonical_collector_reason_code_or_family"} and value["parser_schema_valid"] is False and _valid_canonical_reason(value["canonical_collector_reason_code_or_family"]):
        return
    if set(value) == {"nonempty_timestamp", "valid_price_row_count", "trading_date_fields_valid"}:
        if type(value["nonempty_timestamp"]) is bool and type(value["trading_date_fields_valid"]) is bool and type(value["valid_price_row_count"]) is int and value["valid_price_row_count"] >= 0:
            return
    raise V8DTransportBlocked("V8D_DETECTOR_EVIDENCE_SCHEMA_INVALID")


class DurableV8DAuditStore:
    """Atomic JSON persistence for private attempt dossiers and aggregates."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    @staticmethod
    def new_id() -> str:
        return os.urandom(16).hex()

    def _path(self, dossier_id: str) -> Path:
        if not isinstance(dossier_id, str) or not re.fullmatch(r"[0-9a-f]{32}", dossier_id):
            raise V8DTransportBlocked("V8D_DOSSIER_ID_INVALID")
        return self.root / ("dossier-" + dossier_id + ".json")

    def aggregate_path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise V8DTransportBlocked("V8D_RUN_ID_INVALID")
        return self.root / ("aggregate-" + run_id + ".json")

    def _atomic_write(self, destination: Path, value: Mapping[str, Any]) -> None:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".staging-", dir=str(destination.parent))
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(canonical_json_bytes(value))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(str(temporary), str(destination))
            finally:
                if temporary.exists():
                    temporary.unlink()
        except (OSError, TypeError, ValueError) as error:
            raise V8DAuditPersistenceBlocked() from error

    def _read_object(self, path: Path) -> dict[str, Any]:
        try:
            def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("duplicate key")
                    result[key] = value
                return result
            value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise V8DAuditPersistenceBlocked("V8D_AUDIT_EXISTING_RECORD_INVALID") from error
        if not isinstance(value, dict):
            raise V8DAuditPersistenceBlocked("V8D_AUDIT_EXISTING_RECORD_INVALID")
        return value

    def write_attempt(self, dossier_id: str, context: V8DRequestContext, reviewed_commit: str,
                      gate_binding: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
        reviewed_commit = _require_hex(reviewed_commit, 40, "V8D_REVIEWED_IMPLEMENTATION_COMMIT_INVALID")
        gate_fields = _require_gate_binding(gate_binding, stage=context.logical_stage)
        if set(record) != _ATTEMPT_FIELDS:
            raise V8DAuditPersistenceBlocked("V8D_ATTEMPT_SCHEMA_INVALID")
        if record["request_fingerprint"] != context.request_fingerprint or record["request_url_sha256"] != context.request_url_sha256:
            raise V8DAuditPersistenceBlocked("V8D_ATTEMPT_REQUEST_BINDING_INVALID")
        path = self._path(dossier_id)
        if path.exists():
            dossier = self._read_object(path)
            dossier_hash = dossier.get("audit_artifact_self_hash")
            body = {key: value for key, value in dossier.items() if key != "audit_artifact_self_hash"}
            if dossier_hash != canonical_sha256(body):
                raise V8DAuditPersistenceBlocked("V8D_AUDIT_SELF_HASH_INVALID")
            if dossier.get("reviewed_production_implementation_commit") != reviewed_commit:
                raise V8DAuditPersistenceBlocked("V8D_AUDIT_IMPLEMENTATION_BINDING_MISMATCH")
            if any(dossier.get(key) != value for key, value in gate_fields.items()):
                raise V8DAuditPersistenceBlocked("V8D_AUDIT_GATE_BINDING_MISMATCH")
        else:
            dossier = {
                "schema_version": "V8D_PRIVATE_TRANSPORT_AUDIT_V1",
                "study": STUDY,
                "frozen_design_commit": FROZEN_DESIGN_COMMIT,
                "reviewed_production_implementation_commit": reviewed_commit,
                "human_gate": gate_fields["human_gate"],
                "gate_receipt_key_sha256": gate_fields["gate_receipt_key_sha256"],
                "gate_receipt_bytes_sha256": gate_fields["gate_receipt_bytes_sha256"],
                "authorization_identity_sha256": gate_fields["authorization_identity_sha256"],
                "canonical_parser_classifier_commit": CANONICAL_PARSER_CLASSIFIER_COMMIT,
                "canonical_parser_classifier_blob": CANONICAL_PARSER_CLASSIFIER_BLOB,
                "logical_stage": context.logical_stage,
                "logical_block": context.logical_block,
                "logical_coordinate": context.logical_coordinate,
                "window_start": context.window_start,
                "window_end_exclusive": context.window_end_exclusive,
                "sentinel_indices": list(context.sentinel_indices) if context.sentinel_indices is not None else [],
                "attempts": [],
            }
        if dossier.get("logical_stage") != context.logical_stage or dossier.get("logical_block") != context.logical_block or dossier.get("logical_coordinate") != context.logical_coordinate:
            raise V8DAuditPersistenceBlocked("V8D_AUDIT_CONTEXT_MISMATCH")
        attempts = dossier.get("attempts")
        if not isinstance(attempts, list):
            raise V8DAuditPersistenceBlocked("V8D_AUDIT_ATTEMPTS_INVALID")
        attempts = list(attempts)
        attempts.append(dict(record))
        body = {key: value for key, value in dossier.items() if key != "audit_artifact_self_hash"}
        body["attempts"] = attempts
        body["audit_artifact_self_hash"] = canonical_sha256(body)
        self._atomic_write(path, body)
        return body

    def read_dossier(self, dossier_id: str) -> dict[str, Any]:
        return self._read_object(self._path(dossier_id))

    def write_aggregate(self, run_id: str, aggregate: Mapping[str, Any]) -> Path:
        body = dict(aggregate)
        body.pop("aggregate_self_hash", None)
        body["aggregate_self_hash"] = canonical_sha256(body)
        path = self.aggregate_path(run_id)
        self._atomic_write(path, body)
        return path

    def read_aggregate(self, run_id: str) -> dict[str, Any]:
        return self._read_object(self.aggregate_path(run_id))


def _metadata_for_success() -> dict[str, Any]:
    return {"concrete_exception_type": None, "http_code": None, "reason_type": None,
            "errno": None, "named_condition": None, "detector_evidence": None}


def attempt_with_frozen_retry(attempt_fn: Callable[[], T], *, store: DurableV8DAuditStore,
                              dossier_id: str, context: V8DRequestContext,
                              reviewed_implementation_commit: str,
                              gate_binding: Mapping[str, Any],
                              sleep_fn: Callable[[float], None] = time.sleep) -> tuple[T, dict[str, Any]]:
    """Run one logical request; persist each attempt before any transition.

    ``gate_binding`` must be the exact safe binding produced by a real,
    already-successful `src.v8d_human_gate_consumption.consume_gate_and_
    bind` call for this stage -- never a plain caller-constructed mapping
    invented to look right. This function only checks its shape; the
    actual gate-consumption/receipt evidence is independently re-verified
    later by `src.v8d_audit`.
    """
    reviewed_implementation_commit = _require_hex(reviewed_implementation_commit, 40, "V8D_REVIEWED_IMPLEMENTATION_COMMIT_INVALID")
    _require_gate_binding(gate_binding, stage=context.logical_stage)
    for attempt_number in range(1, MAXIMUM_ATTEMPTS + 1):
        try:
            result = attempt_fn()
        except BaseException as error:  # noqa: BLE001 - every concrete exception is audited
            classification, retryable = classify_transport_exception(error)
            terminal = retryable is False or attempt_number == MAXIMUM_ATTEMPTS
            record = _attempt_record(
                context, attempt_number, classification, retryable,
                _classification_metadata(error, classification),
                "TERMINAL_FAILURE" if terminal else "RETRYABLE_FAILURE",
            )
            stored = store.write_attempt(dossier_id, context, reviewed_implementation_commit, gate_binding, record)
            try:
                setattr(error, "v8d_dossier_id", dossier_id)
                setattr(error, "v8d_audit_artifact_self_hash", stored["audit_artifact_self_hash"])
            except Exception:
                pass
            if terminal:
                raise
            sleep_fn(float(BACKOFF_SECONDS[attempt_number - 1]))
            continue
        else:
            record = _attempt_record(context, attempt_number, "SUCCESS", None, _metadata_for_success(), "SUCCESS")
            stored = store.write_attempt(dossier_id, context, reviewed_implementation_commit, gate_binding, record)
            return result, {
                "dossier_id": dossier_id,
                "attempts": attempt_number,
                "retry_count": attempt_number - 1,
                "audit_artifact_self_hash": stored["audit_artifact_self_hash"],
            }
    raise V8DTransportBlocked("V8D_RETRY_LOOP_INVARIANT_VIOLATED")


def execute_v8d_stage(*, stage: str, request_factory: Callable[[int], V8DRequestPlan],
                      store: DurableV8DAuditStore, reviewed_implementation_commit: str,
                      gate_binding: Mapping[str, Any],
                      window_start: str, window_end_exclusive: str, request_count: int,
                      sleep_fn: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Execute a synthetic-injectable readiness/acquisition catch path.

    The request factory may close over private production state, but only the
    integer logical coordinate and hashes enter the durable/public artifacts.

    ``gate_binding`` is applied identically to every dossier produced by
    this one stage execution (every logical coordinate shares the exact
    same already-consumed gate receipt) -- see `attempt_with_frozen_retry`.
    """
    if stage not in ALL_STAGES:
        raise V8DTransportBlocked("V8D_LOGICAL_STAGE_INVALID")
    _require_hex(reviewed_implementation_commit, 40, "V8D_REVIEWED_IMPLEMENTATION_COMMIT_INVALID")
    _require_gate_binding(gate_binding, stage=stage)
    if type(request_count) is not int or request_count <= 0:
        raise V8DTransportBlocked("V8D_REQUEST_COUNT_INVALID")
    if stage in READINESS_STAGES and request_count != len(SENTINEL_INDICES):
        raise V8DTransportBlocked("V8D_READINESS_REQUEST_COUNT_INVALID")
    if stage in READINESS_STAGES:
        coordinates = list(SENTINEL_INDICES)
        sentinel_indices: tuple[int, ...] | None = SENTINEL_INDICES
    else:
        coordinates = list(range(request_count))
        sentinel_indices = None
    block = "T1C" if stage.startswith("T1C") else "T2"
    run_id = store.new_id()
    dossier_ids: list[str] = []
    for coordinate in coordinates:
        plan = request_factory(coordinate)
        if not isinstance(plan, V8DRequestPlan):
            raise V8DTransportBlocked("V8D_REQUEST_PLAN_INVALID")
        context = V8DRequestContext(
            logical_stage=stage,
            logical_block=block,
            logical_coordinate=coordinate,
            window_start=window_start,
            window_end_exclusive=window_end_exclusive,
            request_fingerprint=plan.request_fingerprint,
            request_url_sha256=plan.request_url_sha256,
            sentinel_indices=sentinel_indices,
        )
        dossier_id = store.new_id()
        dossier_ids.append(dossier_id)
        try:
            attempt_with_frozen_retry(
                plan.request_fn,
                store=store,
                dossier_id=dossier_id,
                context=context,
                reviewed_implementation_commit=reviewed_implementation_commit,
                gate_binding=gate_binding,
                sleep_fn=sleep_fn,
            )
        except V8DAuditPersistenceBlocked:
            # Crucially, do not catch this as an ordinary stage BLOCK: no
            # following coordinate may be requested and no aggregate may be
            # published when an attempt audit is missing.
            raise
        except BaseException:
            # A concrete terminal failure is now durably represented.  The
            # stage may continue to its next logical request and ultimately
            # publish a BLOCK aggregate containing all evidence.
            continue

    dossiers = [store.read_dossier(dossier_id) for dossier_id in dossier_ids]
    terminal = [dossier["attempts"][-1] for dossier in dossiers]
    attempts = [record for dossier in dossiers for record in dossier["attempts"]]

    def histogram(values: Sequence[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))

    aggregate: dict[str, Any] = {
        "schema_version": "V8D_PUBLIC_TRANSPORT_AGGREGATE_V1",
        "study": STUDY,
        "frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "reviewed_production_implementation_commit": reviewed_implementation_commit,
        "canonical_parser_classifier_commit": CANONICAL_PARSER_CLASSIFIER_COMMIT,
        "canonical_parser_classifier_blob": CANONICAL_PARSER_CLASSIFIER_BLOB,
        "logical_stage": stage,
        "logical_block": block,
        "result": "PASS" if all(record["classification"] == "SUCCESS" for record in terminal) else "BLOCK",
        "sentinel_indices": list(sentinel_indices or []),
        "window_start": window_start,
        "window_end_exclusive": window_end_exclusive,
        "sentinel_count": len(sentinel_indices or []),
        "sentinel_pass_count": sum(record["classification"] == "SUCCESS" for record in terminal) if sentinel_indices is not None else 0,
        "request_count": len(dossiers),
        "total_request_attempts": len(attempts),
        "retry_count": len(attempts) - len(dossiers),
        "retryable_attempt_count": sum(record["retryable"] is True for record in attempts),
        "nonretryable_attempt_count": sum(record["retryable"] is False for record in attempts),
        "terminal_classification_histogram": histogram([record["classification"] for record in terminal]),
        "attempt_classification_histogram": histogram([record["classification"] for record in attempts]),
        "attempt_count_histogram": histogram([str(len(dossier["attempts"])) for dossier in dossiers]),
        "http_status_histogram": histogram([str(record["http_code"]) for record in attempts if record["http_code"] is not None]),
        "audit_artifact_count": len(dossiers),
        "audit_artifact_self_hash": canonical_sha256(sorted(dossier["audit_artifact_self_hash"] for dossier in dossiers)),
        "audit_evidence_complete": True,
        "no_missing_terminal_failure_evidence": True,
    }
    aggregate_path = store.write_aggregate(run_id, aggregate)
    durable_aggregate = store.read_aggregate(run_id)
    return {
        "aggregate": durable_aggregate,
        "aggregate_path": aggregate_path,
        "dossier_paths": [store._path(dossier_id) for dossier_id in dossier_ids],  # private test/verification handle only
    }


__all__ = [
    "ACQUISITION_STAGES", "ALL_STAGES", "BACKOFF_SECONDS", "CANONICAL_PARSER_CLASSIFIER_BLOB",
    "CANONICAL_PARSER_CLASSIFIER_COMMIT", "DurableV8DAuditStore", "FROZEN_DESIGN_COMMIT",
    "JITTER", "MAXIMUM_ATTEMPTS", "MAXIMUM_RETRIES", "NONRETRYABLE_CLASSES",
    "NONRETRYABLE_HTTP_CODES", "READINESS_STAGES", "RETRYABLE_CLASSES", "RETRYABLE_HTTP_CODES",
    "SENTINEL_END_EXCLUSIVE", "SENTINEL_INDICES", "SENTINEL_START", "STUDY", "V8DAuditPersistenceBlocked",
    "V8DNamedFailure", "V8DRequestContext", "V8DRequestPlan", "V8DTransportBlocked",
    "V8DTrustedYahooRedirectHandler", "attempt_with_frozen_retry", "canonical_json_bytes",
    "canonical_sha256", "classify_collector_reason", "classify_named_condition", "classify_transport_exception",
    "default_trusted_yahoo_opener", "make_request_fingerprint", "origin_guard_evidence",
    "build_yahoo_request_plan", "execute_v8d_stage", "require_nonempty_quality",
    "require_trusted_origin", "sha256_url",
]
