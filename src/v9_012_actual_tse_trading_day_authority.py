"""V9_012 actual TSE cash-equity trading-day authority.

Acquisition and offline materialization are deliberately separate. Importing
this module performs no network I/O, credential lookup, or durable-state read.
The acquisition core accepts an injected fetcher for synthetic tests; only
the future production seam binds it to J-Quants after fail-closed checks.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v8c_transport import (
    BACKOFF_SECONDS,
    MAXIMUM_ATTEMPTS_PER_TICKER,
    NONRETRYABLE_HTTP_CODES,
    RETRYABLE_HTTP_CODES,
    V8CTransportBlocked,
    V8CTransportNamedFailure,
    attempt_with_frozen_retry,
)


STUDY = "V9_012_ACTUAL_TSE_TRADING_DAY_AUTHORITY_SUCCESSOR"
SOURCE_A = "SOURCE_A"
SOURCE_B = "SOURCE_B"
SOURCE_A_ENDPOINT = "https://api.jquants.com/v2/markets/calendar"
SOURCE_B_ENDPOINT = "https://api.jquants.com/v2/indices/bars/daily/topix"
SOURCE_A_ROLE = "SCHEDULED_TSE_BUSINESS_DAY_SUPERSET"
SOURCE_B_ROLE = "ACTUAL_TSE_MARKET_ACTIVITY_DATE_EVIDENCE"
SOURCE_A_BASE_QUERY_OBJECT = {"from": "2017-01-01", "to": "2026-01-31"}
SOURCE_B_BASE_QUERY_OBJECT = {"from": "2017-01-01", "to": "2026-01-31"}
SOURCE_A_BASE_QUERY = SOURCE_A_BASE_QUERY_OBJECT
SOURCE_B_BASE_QUERY = SOURCE_B_BASE_QUERY_OBJECT
COVERED_START = "2017-01-01"
COVERED_END = "2026-01-31"
EXPECTED_EXCEPTION_SET = frozenset({"2020-10-01"})
HUMAN_CONFIRMATION = "CONFIRM_V9_012_ACTUAL_TSE_TRADING_DAY_AUTHORITY_SUCCESSOR"
API_KEY_ENVIRONMENT_VARIABLE = "JQUANTS_API_KEY"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
REVIEWED_DESIGN_GIT_SHA = "04d63d17795b582f597fc8097491ed8662f2392c"
REVIEWED_DESIGN_BLOB_SHA = "ff6c092cf3ecebb4bab7fc8871e8b70bcf6b3224"
DESIGN_PATH = "V9_012_ACTUAL_TSE_TRADING_DAY_AUTHORITY_SUCCESSOR_DESIGN.md"
MAX_PRE_COMPLETE_ATTEMPTS = 3
FROZEN_BACKOFF_SECONDS = (5, 30)
FROZEN_RETRYABLE_CLASSES = frozenset({
    "NETWORK_TIMEOUT", "CONNECTION_RESET", "TEMPORARY_DNS_FAILURE",
    "HTTP_408", "HTTP_425", "HTTP_429", "HTTP_500", "HTTP_502",
    "HTTP_503", "HTTP_504",
})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^20[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$")

assert MAXIMUM_ATTEMPTS_PER_TICKER == MAX_PRE_COMPLETE_ATTEMPTS
assert tuple(BACKOFF_SECONDS) == FROZEN_BACKOFF_SECONDS


@dataclass(frozen=True)
class SourceSpec:
    key: str
    endpoint: str
    role: str
    base_query: Mapping[str, str]
    state_name: str


SOURCE_SPECS = {
    SOURCE_A: SourceSpec(
        SOURCE_A, SOURCE_A_ENDPOINT, SOURCE_A_ROLE,
        SOURCE_A_BASE_QUERY_OBJECT, "source_a",
    ),
    SOURCE_B: SourceSpec(
        SOURCE_B, SOURCE_B_ENDPOINT, SOURCE_B_ROLE,
        SOURCE_B_BASE_QUERY_OBJECT, "source_b",
    ),
}

CANONICAL_CONTENT_KEYS = frozenset({
    "schema_version", "covered_start", "covered_end", "trading_dates",
    "scheduled_calendar_source_chain_sha256", "topix_source_chain_sha256",
    "scheduled_open_count", "actual_trading_date_count",
    "expected_exception_dates", "observed_exception_dates",
    "scheduled_calendar_source_api_identity", "topix_source_api_identity",
    "scheduled_calendar_base_query_sha256", "topix_base_query_sha256",
    "acquisition_design_git_sha", "acquisition_implementation_git_sha",
})
CANONICAL_RECEIPT_KEYS = frozenset({
    "schema_version", "status", "canonical_artifact_sha256",
})
SOURCE_CHAIN_KEYS = frozenset({
    "base_query_sha256", "page_count", "pages", "source_api_identity",
    "source_role", "terminal_page_index",
})
SOURCE_CHAIN_PAGE_KEYS = frozenset({
    "byte_count", "continuation_issued", "continuation_key_sha256",
    "page_index", "page_request_identity_sha256", "payload_sha256",
})
LOCK_KEYS = frozenset({
    "byte_count", "http_status", "page_index", "page_request_identity_sha256",
    "payload_sha256", "source_api_identity", "source_role",
})


def canonical_json_no_lf(value: object) -> bytes:
    """Return the frozen CANONICAL_JSON_NO_LF byte sequence."""
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return rendered.encode("utf-8")


CANONICAL_JSON_NO_LF = canonical_json_no_lf


def canonical_json_bytes(value: object) -> bytes:
    """Return the frozen public JSON sequence with exactly one final LF."""
    return canonical_json_no_lf(value) + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_utf8(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _check_sha(value: object, pattern: re.Pattern[str], reason: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise V9012Error(reason)
    return value


def _strict_int(value: object, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def _source(source_key: str) -> SourceSpec:
    try:
        return SOURCE_SPECS[source_key]
    except KeyError as exc:
        raise V9012Error("SOURCE_ID_INVALID") from exc


def source_api_identity(source_key: str) -> str:
    return _source(source_key).endpoint


def source_role(source_key: str) -> str:
    return _source(source_key).role


def base_query_object(source_key: str) -> dict[str, str]:
    return dict(_source(source_key).base_query)


def base_query_sha256(source_key: str) -> str:
    return sha256_bytes(canonical_json_no_lf(base_query_object(source_key)))


def pagination_key_sha256(key: str) -> str:
    if type(key) is not str or key == "":
        raise V9012Error("PAGINATION_KEY_INVALID")
    return sha256_utf8(key)


@dataclass(frozen=True)
class PageRequest:
    source_key: str
    page_index: int
    continuation_key: str | None = None

    def __post_init__(self) -> None:
        _source(self.source_key)
        if type(self.page_index) is not int or self.page_index < 1:
            raise V9012Error("PAGE_REQUEST_INDEX_INVALID")
        if self.continuation_key is not None and (
            type(self.continuation_key) is not str or self.continuation_key == ""
        ):
            raise V9012Error("PAGINATION_KEY_INVALID")

    @property
    def params(self) -> dict[str, str]:
        result = base_query_object(self.source_key)
        if self.continuation_key is not None:
            result["pagination_key"] = self.continuation_key
        return result


def expected_request_url(request: PageRequest) -> str:
    spec = _source(request.source_key)
    query = [("from", COVERED_START), ("to", COVERED_END)]
    if request.continuation_key is not None:
        query.append(("pagination_key", request.continuation_key))
    return spec.endpoint + "?" + urllib.parse.urlencode(query)


def page_request_identity(request: PageRequest) -> dict[str, object]:
    return {
        "base_query_sha256": base_query_sha256(request.source_key),
        "continuation_key_sha256": (
            None if request.continuation_key is None
            else pagination_key_sha256(request.continuation_key)
        ),
        "page_index": request.page_index,
        "source_api_identity": source_api_identity(request.source_key),
        "source_role": source_role(request.source_key),
    }


def page_request_identity_sha256(request: PageRequest) -> str:
    return sha256_bytes(canonical_json_no_lf(page_request_identity(request)))


@dataclass(frozen=True)
class PageFetchResult:
    payload: bytes
    http_status: int
    resolved_url: str


@dataclass(frozen=True)
class LockedPage:
    record: dict[str, object]
    payload: bytes
    continuation_issued: bool
    continuation_key: str | None


@dataclass(frozen=True)
class MaterializedResult:
    scheduled_source_chain: dict[str, object]
    topix_source_chain: dict[str, object]
    canonical_content: dict[str, object]
    canonical_bytes: bytes
    canonical_artifact_sha256: str
    receipt: dict[str, object]


class V9012Error(RuntimeError):
    """URL-, credential-, payload-, and path-free fail-closed error."""

    def __init__(
        self,
        reason: str,
        *,
        attempts: int = 0,
        requests: int = 0,
        source_key: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts
        self.network_request_count = requests
        self.source_key = source_key


class LockConflictError(V9012Error):
    pass


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except Exception as exc:
            raise LockConflictError("DURABLE_STATE_READ_FAILURE") from exc
        if existing != content:
            raise LockConflictError("DURABLE_STATE_CONFLICT")


def _state_source_root(state_root: str | Path, source_key: str) -> Path:
    spec = _source(source_key)
    return Path(state_root).resolve() / spec.state_name


def _validate_state_root_container(state_root: str | Path) -> None:
    root = Path(state_root).resolve()
    if not root.exists():
        return
    if not root.is_dir():
        raise V9012Error("DURABLE_STATE_CONTAINER_INVALID")
    try:
        children = list(root.iterdir())
    except Exception as exc:
        raise V9012Error("DURABLE_STATE_ENUMERATION_FAILURE") from exc
    if any(child.name not in {"source_a", "source_b"} for child in children):
        raise V9012Error("DURABLE_STATE_UNEXPECTED_FILE")


def _source_directory_present(state_root: str | Path, source_key: str) -> bool:
    """Return whether any durable directory/state exists for a source."""
    return _state_source_root(state_root, source_key).exists()


def _source_terminal_proven(state_root: str | Path, source_key: str) -> bool:
    """Prove terminal completion from transport/provenance state only.

    This deliberately reads no Date, HolDiv, or OHLC values.  The pagination
    envelope is inspected only because it is the durable evidence needed to
    prove the locked chain's terminal page.
    """
    source_root = _state_source_root(state_root, source_key)
    if not source_root.exists() or not source_root.is_dir():
        return False
    try:
        children = {child.name for child in source_root.iterdir()}
    except Exception:
        return False
    if children != {"raw_pages", "page_locks"}:
        return False
    try:
        pages = PageLockStore(state_root, source_key).read_locked_chain(require_terminal=True)
    except V9012Error:
        return False
    return bool(pages) and not pages[-1].continuation_issued


def validate_durable_source_order(state_root: str | Path) -> dict[str, bool]:
    """Validate SOURCE_A-before-SOURCE_B durable state without semantics."""
    _validate_state_root_container(state_root)
    source_a_present = _source_directory_present(state_root, SOURCE_A)
    source_b_present = _source_directory_present(state_root, SOURCE_B)
    source_a_terminal = (
        _source_terminal_proven(state_root, SOURCE_A) if source_a_present else False
    )
    if source_b_present and not source_a_terminal:
        raise V9012Error("DURABLE_SOURCE_ORDER_VIOLATION")
    return {
        "source_a_present": source_a_present,
        "source_a_terminal": source_a_terminal,
        "source_b_present": source_b_present,
    }


validate_source_order_state = validate_durable_source_order


def _valid_page_name(path: Path, suffix: str) -> bool:
    return (
        path.is_file() and path.suffix == suffix and path.stem.isdigit()
        and int(path.stem) >= 1 and path.name == f"{int(path.stem):06d}{suffix}"
    )


def _inspect_pagination_envelope(payload: bytes) -> tuple[bool, str | None]:
    """Inspect only transport pagination metadata, never source semantics."""
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise V9012Error("PAGINATION_ENVELOPE_MALFORMED") from exc
    if type(value) is not dict:
        raise V9012Error("PAGINATION_ENVELOPE_MALFORMED")
    if "pagination_key" not in value:
        return False, None
    key = value["pagination_key"]
    if type(key) is not str or key == "":
        raise V9012Error("PAGINATION_KEY_INVALID")
    return True, key


class PageLockStore:
    """Source-identity-bound raw payload and immutable lock storage."""

    def __init__(self, state_root: str | Path, source_key: str) -> None:
        self.source_key = source_key
        self.spec = _source(source_key)
        self.root = _state_source_root(state_root, source_key)
        self.payload_dir = self.root / "raw_pages"
        self.lock_dir = self.root / "page_locks"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.payload_dir.mkdir(exist_ok=True)
            self.lock_dir.mkdir(exist_ok=True)
        except Exception as exc:
            raise V9012Error("DURABLE_STATE_INITIALIZATION_FAILURE", source_key=source_key) from exc
        try:
            if any(child.name not in {"raw_pages", "page_locks"} for child in self.root.iterdir()):
                raise V9012Error("DURABLE_STATE_UNEXPECTED_FILE", source_key=source_key)
        except V9012Error:
            raise
        except Exception as exc:
            raise V9012Error("DURABLE_STATE_ENUMERATION_FAILURE", source_key=source_key) from exc

    def _payload_path(self, index: int) -> Path:
        return self.payload_dir / f"{index:06d}.bin"

    def _lock_path(self, index: int) -> Path:
        return self.lock_dir / f"{index:06d}.json"

    def lock_page(self, request: PageRequest, result: PageFetchResult) -> dict[str, object]:
        if request.source_key != self.source_key:
            raise V9012Error("SOURCE_ID_BINDING_MISMATCH", source_key=self.source_key)
        if (
            type(result.payload) is not bytes
            or type(result.http_status) is not int
            or type(result.resolved_url) is not str
        ):
            raise V9012Error("TRANSPORT_RESPONSE_TYPE_INVALID", source_key=self.source_key)
        if result.http_status != 200 or result.resolved_url != expected_request_url(request):
            raise V9012Error("TRANSPORT_RESPONSE_NOT_LOCKABLE", source_key=self.source_key)
        record: dict[str, object] = {
            "byte_count": len(result.payload),
            "http_status": result.http_status,
            "page_index": request.page_index,
            "page_request_identity_sha256": page_request_identity_sha256(request),
            "payload_sha256": sha256_bytes(result.payload),
            "source_api_identity": source_api_identity(self.source_key),
            "source_role": source_role(self.source_key),
        }
        _exclusive_write(self._payload_path(request.page_index), result.payload)
        _exclusive_write(self._lock_path(request.page_index), canonical_json_bytes(record))
        return record

    def read_locked_chain(self, *, require_terminal: bool = False) -> list[LockedPage]:
        try:
            lock_paths = list(self.lock_dir.iterdir())
            payload_paths = list(self.payload_dir.iterdir())
        except Exception as exc:
            raise V9012Error("DURABLE_STATE_ENUMERATION_FAILURE", source_key=self.source_key) from exc
        for path in lock_paths:
            if not _valid_page_name(path, ".json"):
                raise V9012Error("DURABLE_STATE_EXTRA_LOCK", source_key=self.source_key)
        for path in payload_paths:
            if not _valid_page_name(path, ".bin"):
                raise V9012Error("DURABLE_STATE_EXTRA_PAYLOAD", source_key=self.source_key)
        indices = sorted(int(path.stem) for path in lock_paths)
        if not indices:
            if payload_paths:
                raise V9012Error("DURABLE_STATE_INCOMPLETE_PAIR", source_key=self.source_key)
            if require_terminal:
                raise V9012Error("PAGE_CHAIN_EMPTY", source_key=self.source_key)
            return []
        if indices != list(range(1, max(indices) + 1)):
            raise V9012Error("PAGE_CHAIN_ORDER_INVALID", source_key=self.source_key)
        if {int(path.stem) for path in payload_paths} != set(indices):
            raise V9012Error("DURABLE_STATE_INCOMPLETE_PAIR", source_key=self.source_key)

        result: list[LockedPage] = []
        previous_key: str | None = None
        seen_keys: set[str] = set()
        for index in indices:
            try:
                raw_record = self._lock_path(index).read_bytes()
                record = json.loads(raw_record.decode("utf-8"))
                payload = self._payload_path(index).read_bytes()
            except Exception as exc:
                raise V9012Error("DURABLE_STATE_LOCK_READ_FAILURE", source_key=self.source_key) from exc
            if type(record) is not dict or set(record) != LOCK_KEYS:
                raise V9012Error("PAGE_LOCK_SCHEMA_INVALID", source_key=self.source_key)
            if type(record["http_status"]) is not int or record["http_status"] != 200:
                raise V9012Error("PAGE_LOCK_HTTP_STATUS_INVALID", source_key=self.source_key)
            request = PageRequest(self.source_key, index, previous_key)
            if (
                record["page_index"] != index
                or record["source_api_identity"] != source_api_identity(self.source_key)
                or record["source_role"] != source_role(self.source_key)
                or record["page_request_identity_sha256"] != page_request_identity_sha256(request)
            ):
                raise V9012Error("PAGE_LOCK_REQUEST_BINDING_MISMATCH", source_key=self.source_key)
            _check_sha(record["payload_sha256"], HEX64, "PAGE_LOCK_DIGEST_INVALID")
            if (
                not _strict_int(record["byte_count"], minimum=0)
                or record["byte_count"] != len(payload)
                or record["payload_sha256"] != sha256_bytes(payload)
                or raw_record != canonical_json_bytes(record)
            ):
                raise V9012Error("PAGE_LOCK_PAYLOAD_MISMATCH", source_key=self.source_key)
            continued, key = _inspect_pagination_envelope(payload)
            if continued:
                if key in seen_keys:
                    raise V9012Error("PAGINATION_KEY_REPEATED", source_key=self.source_key)
                seen_keys.add(key)  # type: ignore[arg-type]
                previous_key = key
            else:
                previous_key = None
            result.append(LockedPage(dict(record), payload, continued, key))
            if not continued and index != indices[-1]:
                raise V9012Error("PAGE_CHAIN_AFTER_TERMINAL", source_key=self.source_key)
        if require_terminal and result[-1].continuation_issued:
            raise V9012Error("PAGE_CHAIN_TERMINAL_MISSING", source_key=self.source_key)
        return result


def validate_source_chain_manifest(value: object, source_key: str) -> dict[str, object]:
    spec = _source(source_key)
    if type(value) is not dict or set(value) != SOURCE_CHAIN_KEYS:
        raise V9012Error("SOURCE_CHAIN_MANIFEST_SCHEMA_INVALID", source_key=source_key)
    if (
        value["source_api_identity"] != spec.endpoint
        or value["source_role"] != spec.role
        or value["base_query_sha256"] != base_query_sha256(source_key)
    ):
        raise V9012Error("SOURCE_CHAIN_MANIFEST_BINDING_INVALID", source_key=source_key)
    page_count = value["page_count"]
    if not _strict_int(page_count, minimum=1) or value["terminal_page_index"] != page_count:
        raise V9012Error("SOURCE_CHAIN_MANIFEST_COUNT_INVALID", source_key=source_key)
    pages = value["pages"]
    if type(pages) is not list or len(pages) != page_count:
        raise V9012Error("SOURCE_CHAIN_MANIFEST_COUNT_INVALID", source_key=source_key)
    previous_key_sha: str | None = None
    for expected_index, page in enumerate(pages, 1):
        if type(page) is not dict or set(page) != SOURCE_CHAIN_PAGE_KEYS:
            raise V9012Error("SOURCE_CHAIN_MANIFEST_PAGE_INVALID", source_key=source_key)
        if page["page_index"] != expected_index:
            raise V9012Error("SOURCE_CHAIN_MANIFEST_ORDER_INVALID", source_key=source_key)
        _check_sha(page["page_request_identity_sha256"], HEX64, "PAGE_REQUEST_IDENTITY_DIGEST_INVALID")
        expected_request_identity = {
            "base_query_sha256": base_query_sha256(source_key),
            "continuation_key_sha256": previous_key_sha,
            "page_index": expected_index,
            "source_api_identity": spec.endpoint,
            "source_role": spec.role,
        }
        if page["page_request_identity_sha256"] != sha256_bytes(canonical_json_no_lf(expected_request_identity)):
            raise V9012Error("SOURCE_CHAIN_MANIFEST_REQUEST_BINDING_INVALID", source_key=source_key)
        _check_sha(page["payload_sha256"], HEX64, "PAGE_PAYLOAD_DIGEST_INVALID")
        if not _strict_int(page["byte_count"], minimum=0) or type(page["continuation_issued"]) is not bool:
            raise V9012Error("SOURCE_CHAIN_MANIFEST_PAGE_INVALID", source_key=source_key)
        issued_sha = page["continuation_key_sha256"]
        if page["continuation_issued"]:
            _check_sha(issued_sha, HEX64, "PAGINATION_KEY_DIGEST_INVALID")
            previous_key_sha = issued_sha
        elif issued_sha is not None:
            raise V9012Error("SOURCE_CHAIN_MANIFEST_TERMINAL_INVALID", source_key=source_key)
        else:
            previous_key_sha = None
        if expected_index == page_count and page["continuation_issued"]:
            raise V9012Error("SOURCE_CHAIN_MANIFEST_TERMINAL_INVALID", source_key=source_key)
        if expected_index < page_count and not page["continuation_issued"]:
            raise V9012Error("SOURCE_CHAIN_MANIFEST_TERMINAL_INVALID", source_key=source_key)
    return dict(value)


def build_source_chain_manifest(source_key: str, pages: Sequence[LockedPage]) -> dict[str, object]:
    if not pages:
        raise V9012Error("PAGE_CHAIN_EMPTY", source_key=source_key)
    entries: list[dict[str, object]] = []
    for page in pages:
        entries.append({
            "byte_count": page.record["byte_count"],
            "continuation_issued": page.continuation_issued,
            "continuation_key_sha256": (
                None if page.continuation_key is None
                else pagination_key_sha256(page.continuation_key)
            ),
            "page_index": page.record["page_index"],
            "page_request_identity_sha256": page.record["page_request_identity_sha256"],
            "payload_sha256": page.record["payload_sha256"],
        })
    return validate_source_chain_manifest({
        "base_query_sha256": base_query_sha256(source_key),
        "page_count": len(entries),
        "pages": entries,
        "source_api_identity": source_api_identity(source_key),
        "source_role": source_role(source_key),
        "terminal_page_index": len(entries),
    }, source_key)


def source_chain_sha256(manifest: object, source_key: str) -> str:
    checked = validate_source_chain_manifest(manifest, source_key)
    return sha256_bytes(canonical_json_no_lf(checked))


def _transport_failure(exc: BaseException, requests: int, source_key: str) -> V9012Error:
    audit = getattr(exc, "transport_audit", ())
    history = list(audit) if isinstance(audit, list) else []
    label = history[-1].get("classification") if history and isinstance(history[-1], dict) else None
    if label in FROZEN_RETRYABLE_CLASSES and requests == MAX_PRE_COMPLETE_ATTEMPTS:
        reason = "PLUMBING_FAILURE_RETRIABLE"
    elif label in {"HTTP_401", "HTTP_403"}:
        reason = "AUTH_OR_PLAN_FAILURE"
    elif isinstance(label, str) and label.startswith("HTTP_"):
        reason = label
    elif label in {"UNTRUSTED_REDIRECT", "RESPONSE_HOST_MISMATCH"}:
        reason = label
    else:
        reason = "IMPLEMENTATION_FAILURE"
    return V9012Error(reason, attempts=requests, requests=requests, source_key=source_key)


def _validate_transport_result(request: PageRequest, result: object) -> PageFetchResult:
    if type(result) is not PageFetchResult:
        raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
    if (
        type(result.payload) is not bytes
        or type(result.http_status) is not int
        or type(result.resolved_url) is not str
    ):
        raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
    if result.http_status != 200:
        if result.http_status in RETRYABLE_HTTP_CODES or result.http_status in NONRETRYABLE_HTTP_CODES:
            raise urllib.error.HTTPError(
                _source(request.source_key).endpoint, result.http_status, "", {}, None
            )
        if 300 <= result.http_status < 400:
            raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")
        raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
    if result.resolved_url != expected_request_url(request):
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
    return result


def acquire_source(
    state_root: str | Path,
    source_key: str,
    *,
    fetcher: Callable[[PageRequest], PageFetchResult],
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> tuple[dict[str, object], int]:
    """Acquire one source chain without inspecting Date/HolDiv/OHLC semantics."""
    order_state = validate_durable_source_order(state_root)
    if source_key == SOURCE_B and not order_state["source_a_terminal"]:
        raise V9012Error("DURABLE_SOURCE_ORDER_VIOLATION")
    store = PageLockStore(state_root, source_key)
    locked = store.read_locked_chain()
    requests = 0
    seen_keys = {page.continuation_key for page in locked if page.continuation_issued}
    if locked and not locked[-1].continuation_issued:
        manifest = build_source_chain_manifest(source_key, locked)
        return manifest, 0
    if locked:
        last = locked[-1]
        next_request = PageRequest(source_key, last.record["page_index"] + 1, last.continuation_key)
    else:
        next_request = PageRequest(source_key, 1, None)
    while True:
        current = next_request
        try:
            def attempt() -> PageFetchResult:
                nonlocal requests
                requests += 1
                return _validate_transport_result(current, fetcher(current))

            result, _audit = attempt_with_frozen_retry(
                attempt,
                sleep_fn=sleep,
                request_fingerprint=page_request_identity_sha256(current),
            )
        except V9012Error as exc:
            exc.network_request_count = requests
            raise
        except (V8CTransportNamedFailure, V8CTransportBlocked) as exc:
            raise _transport_failure(exc, requests, source_key) from exc
        except BaseException as exc:  # noqa: BLE001 - frozen transport boundary
            raise _transport_failure(exc, requests, source_key) from exc

        try:
            store.lock_page(current, result)
            continued, key = _inspect_pagination_envelope(result.payload)
            if continued and key in seen_keys:
                raise V9012Error("PAGINATION_KEY_REPEATED", attempts=1, requests=requests, source_key=source_key)
            if not continued:
                complete = store.read_locked_chain(require_terminal=True)
                return build_source_chain_manifest(source_key, complete), requests
            seen_keys.add(key)  # type: ignore[arg-type]
            next_request = PageRequest(source_key, current.page_index + 1, key)
        except V9012Error as exc:
            exc.network_request_count = requests
            raise


def acquire_sources(
    state_root: str | Path,
    *,
    fetcher: Callable[[PageRequest], PageFetchResult],
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> tuple[dict[str, dict[str, object]], int]:
    """Acquire SOURCE_A completely before beginning SOURCE_B."""
    scheduled, requests_a = acquire_source(
        state_root, SOURCE_A, fetcher=fetcher, sleep=sleep
    )
    topix, requests_b = acquire_source(
        state_root, SOURCE_B, fetcher=fetcher, sleep=sleep
    )
    return {SOURCE_A: scheduled, SOURCE_B: topix}, requests_a + requests_b


def _coverage_dates() -> list[str]:
    start = _dt.date.fromisoformat(COVERED_START)
    end = _dt.date.fromisoformat(COVERED_END)
    return [
        (start + _dt.timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def _payload_data(payload: bytes) -> list[dict[str, object]]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise V9012Error("PARSER_SCHEMA_FAILURE") from exc
    if type(value) is not dict or type(value.get("data")) is not list:
        raise V9012Error("PARSER_SCHEMA_FAILURE")
    rows = value["data"]
    if any(type(row) is not dict for row in rows):
        raise V9012Error("PARSER_SCHEMA_FAILURE")
    return rows


def _strict_date(value: object) -> str:
    if type(value) is not str or DATE_RE.fullmatch(value) is None:
        raise V9012Error("DATA_QUALITY_FAILURE")
    try:
        _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise V9012Error("DATA_QUALITY_FAILURE") from exc
    if value < COVERED_START or value > COVERED_END:
        raise V9012Error("DATA_QUALITY_FAILURE")
    return value


def validate_source_a_rows(pages: Sequence[LockedPage]) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in pages:
        for row in _payload_data(page.payload):
            if "Date" not in row or "HolDiv" not in row:
                raise V9012Error("PARSER_SCHEMA_FAILURE")
            date_value = _strict_date(row["Date"])
            hol_div = row["HolDiv"]
            if type(hol_div) is not str or hol_div not in {"0", "1", "2", "3"}:
                raise V9012Error("DATA_QUALITY_FAILURE")
            if date_value in seen:
                raise V9012Error("DATA_QUALITY_FAILURE")
            seen.add(date_value)
            rows.append({"Date": date_value, "HolDiv": hol_div})
    expected = _coverage_dates()
    if sorted(seen) != expected:
        raise V9012Error("DATA_QUALITY_FAILURE")
    rows.sort(key=lambda row: row["Date"])
    return rows, {row["Date"] for row in rows if row["HolDiv"] in {"1", "2"}}


def _finite_real(value: object) -> bool:
    if type(value) not in {int, float} or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def validate_source_b_rows(pages: Sequence[LockedPage]) -> set[str]:
    active: set[str] = set()
    seen: set[str] = set()
    for page in pages:
        for row in _payload_data(page.payload):
            for field in ("Date", "O", "H", "L", "C"):
                if field not in row:
                    raise V9012Error("DATA_QUALITY_FAILURE")
            date_value = _strict_date(row["Date"])
            if date_value in seen:
                raise V9012Error("DATA_QUALITY_FAILURE")
            seen.add(date_value)
            values = [row[field] for field in ("O", "H", "L", "C")]
            null_count = sum(value is None for value in values)
            if null_count == 4:
                continue
            if null_count != 0:
                raise V9012Error("DATA_QUALITY_FAILURE")
            if not all(_finite_real(value) for value in values):
                raise V9012Error("DATA_QUALITY_FAILURE")
            active.add(date_value)
    return active


def _get_complete_pages(state_root: str | Path, source_key: str) -> list[LockedPage]:
    return PageLockStore(state_root, source_key).read_locked_chain(require_terminal=True)


def _adjudicate_dates(scheduled_open_dates: set[str], topix_active_dates: set[str]) -> tuple[list[str], list[str]]:
    observed = sorted(scheduled_open_dates - topix_active_dates)
    if (
        scheduled_open_dates - topix_active_dates != EXPECTED_EXCEPTION_SET
        or topix_active_dates - scheduled_open_dates != set()
        or "2020-09-30" not in topix_active_dates
        or "2020-10-01" in topix_active_dates
        or "2020-10-02" not in topix_active_dates
    ):
        raise V9012Error("ACTUAL_TRADING_DAY_AUTHORITY_FAILURE")
    return sorted(EXPECTED_EXCEPTION_SET), observed


def validate_canonical_content(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != CANONICAL_CONTENT_KEYS:
        raise V9012Error("CANONICAL_CONTENT_SCHEMA_INVALID")
    if value["schema_version"] != "V9_012_CANONICAL_ACTUAL_TSE_TRADING_DAYS_V1":
        raise V9012Error("CANONICAL_CONTENT_BINDING_INVALID")
    if value["covered_start"] != COVERED_START or value["covered_end"] != COVERED_END:
        raise V9012Error("CANONICAL_CONTENT_BINDING_INVALID")
    if value["scheduled_calendar_source_api_identity"] != SOURCE_A_ENDPOINT:
        raise V9012Error("CANONICAL_CONTENT_BINDING_INVALID")
    if value["topix_source_api_identity"] != SOURCE_B_ENDPOINT:
        raise V9012Error("CANONICAL_CONTENT_BINDING_INVALID")
    if value["scheduled_calendar_base_query_sha256"] != base_query_sha256(SOURCE_A):
        raise V9012Error("CANONICAL_CONTENT_BINDING_INVALID")
    if value["topix_base_query_sha256"] != base_query_sha256(SOURCE_B):
        raise V9012Error("CANONICAL_CONTENT_BINDING_INVALID")
    for field in (
        "scheduled_calendar_source_chain_sha256", "topix_source_chain_sha256",
    ):
        _check_sha(value[field], HEX64, "CANONICAL_CONTENT_DIGEST_INVALID")
    for field in ("acquisition_design_git_sha", "acquisition_implementation_git_sha"):
        _check_sha(value[field], HEX40, "CANONICAL_CONTENT_GIT_SHA_INVALID")
    if value["expected_exception_dates"] != ["2020-10-01"] or value["observed_exception_dates"] != ["2020-10-01"]:
        raise V9012Error("CANONICAL_CONTENT_EXCEPTION_INVALID")
    if type(value["trading_dates"]) is not list or any(
        type(date_value) is not str or DATE_RE.fullmatch(date_value) is None
        for date_value in value["trading_dates"]
    ):
        raise V9012Error("CANONICAL_CONTENT_TRADING_DATES_INVALID")
    if value["trading_dates"] != sorted(value["trading_dates"]):
        raise V9012Error("CANONICAL_CONTENT_TRADING_DATES_INVALID")
    if len(set(value["trading_dates"])) != len(value["trading_dates"]):
        raise V9012Error("CANONICAL_CONTENT_TRADING_DATES_INVALID")
    if not _strict_int(value["scheduled_open_count"], minimum=0) or not _strict_int(value["actual_trading_date_count"], minimum=0):
        raise V9012Error("CANONICAL_CONTENT_COUNT_INVALID")
    if value["actual_trading_date_count"] != len(value["trading_dates"]):
        raise V9012Error("CANONICAL_CONTENT_COUNT_INVALID")
    return dict(value)


def validate_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != CANONICAL_RECEIPT_KEYS:
        raise V9012Error("CANONICAL_RECEIPT_SCHEMA_INVALID")
    if value["schema_version"] != "V9_012_CANONICAL_HASH_RECEIPT_V1" or value["status"] != "COMPLETE":
        raise V9012Error("CANONICAL_RECEIPT_BINDING_INVALID")
    _check_sha(value["canonical_artifact_sha256"], HEX64, "CANONICAL_RECEIPT_DIGEST_INVALID")
    return dict(value)


def materialize_sources(
    state_root: str | Path,
    *,
    acquisition_design_git_sha: str,
    acquisition_implementation_git_sha: str,
) -> MaterializedResult:
    """Materialize only complete locked source chains; no network path exists."""
    design_sha = _check_sha(acquisition_design_git_sha, HEX40, "ACQUISITION_DESIGN_GIT_SHA_INVALID")
    implementation_sha = _check_sha(acquisition_implementation_git_sha, HEX40, "ACQUISITION_IMPLEMENTATION_GIT_SHA_INVALID")
    validate_durable_source_order(state_root)
    try:
        source_a_pages = _get_complete_pages(state_root, SOURCE_A)
        source_b_pages = _get_complete_pages(state_root, SOURCE_B)
        source_a_manifest = build_source_chain_manifest(SOURCE_A, source_a_pages)
        source_b_manifest = build_source_chain_manifest(SOURCE_B, source_b_pages)
        source_a_sha = source_chain_sha256(source_a_manifest, SOURCE_A)
        source_b_sha = source_chain_sha256(source_b_manifest, SOURCE_B)
        _source_a_rows, scheduled_open_dates = validate_source_a_rows(source_a_pages)
        topix_active_dates = validate_source_b_rows(source_b_pages)
        expected_exceptions, observed_exceptions = _adjudicate_dates(scheduled_open_dates, topix_active_dates)
    except V9012Error as exc:
        if exc.reason not in {"ACTUAL_TRADING_DAY_AUTHORITY_FAILURE", "DURABLE_STATE_ENUMERATION_FAILURE", "DURABLE_STATE_INITIALIZATION_FAILURE"}:
            raise V9012Error("ACTUAL_TRADING_DAY_AUTHORITY_FAILURE") from exc
        raise
    trading_dates = sorted(topix_active_dates)
    content = validate_canonical_content({
        "schema_version": "V9_012_CANONICAL_ACTUAL_TSE_TRADING_DAYS_V1",
        "covered_start": COVERED_START,
        "covered_end": COVERED_END,
        "trading_dates": trading_dates,
        "scheduled_calendar_source_chain_sha256": source_a_sha,
        "topix_source_chain_sha256": source_b_sha,
        "scheduled_open_count": len(scheduled_open_dates),
        "actual_trading_date_count": len(trading_dates),
        "expected_exception_dates": expected_exceptions,
        "observed_exception_dates": observed_exceptions,
        "scheduled_calendar_source_api_identity": SOURCE_A_ENDPOINT,
        "topix_source_api_identity": SOURCE_B_ENDPOINT,
        "scheduled_calendar_base_query_sha256": base_query_sha256(SOURCE_A),
        "topix_base_query_sha256": base_query_sha256(SOURCE_B),
        "acquisition_design_git_sha": design_sha,
        "acquisition_implementation_git_sha": implementation_sha,
    })
    canonical_bytes = canonical_json_bytes(content)
    if b"canonical_artifact_sha256" in canonical_bytes:
        raise V9012Error("CANONICAL_CONTENT_SELF_REFERENCE")
    artifact_sha = sha256_bytes(canonical_bytes)
    receipt = validate_receipt({
        "schema_version": "V9_012_CANONICAL_HASH_RECEIPT_V1",
        "status": "COMPLETE",
        "canonical_artifact_sha256": artifact_sha,
    })
    return MaterializedResult(
        source_a_manifest, source_b_manifest, content, canonical_bytes, artifact_sha, receipt
    )


def write_materialized_artifacts(result: MaterializedResult, output_root: str | Path) -> None:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _exclusive_write(
        root / "V9_012_CANONICAL_ACTUAL_TSE_TRADING_DAYS_V1.json",
        result.canonical_bytes,
    )
    _exclusive_write(
        root / "V9_012_CANONICAL_HASH_RECEIPT_V1.json",
        canonical_json_bytes(result.receipt),
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def _reject(self, *_args: object) -> None:
        raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


def _build_request(request: PageRequest, api_key: str) -> urllib.request.Request:
    if type(api_key) is not str or api_key == "":
        raise V9012Error("API_KEY_MISSING", source_key=request.source_key)
    return urllib.request.Request(
        expected_request_url(request), headers={"x-api-key": api_key}, method="GET"
    )


def fetch_http_page(request: PageRequest) -> PageFetchResult:
    """The only network function; credential lookup occurs at request time."""
    api_key = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
    if type(api_key) is not str or api_key == "":
        raise V9012Error("API_KEY_MISSING", source_key=request.source_key)
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(_build_request(request, api_key), timeout=30.0) as response:
        return PageFetchResult(response.read(), response.getcode(), response.geturl())


def _default_git_runner(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_protected_environment(repo_root: str | Path) -> dict[str, object]:
    from src.v9_010_jpx_calendar_stage_a_acquisition import verify_protected_environment as verify
    return verify(repo_root)


def _restartable_root_shape(output: Path) -> bool:
    if not output.exists() or not output.is_dir():
        return False
    try:
        children = {child.name for child in output.iterdir()}
    except Exception as exc:
        raise V9012Error("DURABLE_STATE_ENUMERATION_FAILURE") from exc
    if not children or not children.issubset({"source_a", "source_b"}):
        return False
    for name in children:
        child = output / name
        if not child.is_dir():
            return False
        nested = {item.name for item in child.iterdir()}
        if not nested.issubset({"raw_pages", "page_locks"}):
            return False
    return True


def verify_production_preflight(
    repo_root: str | Path,
    output_root: str | Path,
    *,
    expected_implementation_sha: str,
    confirmation: str,
    git_runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, object]:
    implementation_sha = _check_sha(expected_implementation_sha, HEX40, "IMPLEMENTATION_SHA_INVALID")
    if confirmation != HUMAN_CONFIRMATION:
        raise V9012Error("FRESH_HUMAN_CONFIRMATION_INVALID")
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    if output == root or output in root.parents or root in output.parents:
        raise V9012Error("DURABLE_STATE_MUST_BE_EXTERNAL")
    if output.exists() and not _restartable_root_shape(output):
        raise V9012Error("DURABLE_EXECUTION_ROOT_COLLISION")
    if output.exists():
        validate_durable_source_order(output)
    runner = git_runner or (lambda args: _default_git_runner(root, args))
    try:
        branch = runner(["branch", "--show-current"])
        local = runner(["rev-parse", "HEAD"])
        remote = runner(["rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"])
        status = runner(["status", "--porcelain=v1", "--untracked-files=all"])
        design_commit = runner(["rev-parse", f"{REVIEWED_DESIGN_GIT_SHA}^{{commit}}"])
        design_blob = runner(["rev-parse", f"{REVIEWED_DESIGN_GIT_SHA}:{DESIGN_PATH}"])
    except Exception as exc:
        raise V9012Error("GIT_PROVENANCE_UNAVAILABLE") from exc
    if branch != AUTHORITATIVE_BRANCH or local != implementation_sha or remote != implementation_sha:
        raise V9012Error("GIT_PROVENANCE_MISMATCH")
    if status != "":
        raise V9012Error("GIT_WORKTREE_DIRTY")
    if design_commit != REVIEWED_DESIGN_GIT_SHA or design_blob != REVIEWED_DESIGN_BLOB_SHA:
        raise V9012Error("DESIGN_BINDING_MISMATCH")
    try:
        environment = verify_protected_environment(root)
    except V9012Error:
        raise
    except Exception as exc:
        raise V9012Error("PROTECTED_ENVIRONMENT_CHECK_FAILURE") from exc
    return {
        "authoritative_branch": branch,
        "implementation_git_sha": implementation_sha,
        "local_head": local,
        "remote_tracking_head": remote,
        "reviewed_design_git_sha": REVIEWED_DESIGN_GIT_SHA,
        "design_blob_sha256": REVIEWED_DESIGN_BLOB_SHA,
        "source_order": [SOURCE_A, SOURCE_B],
        "network_authorized": True,
        "resume_state": output.exists(),
        **environment,
    }


def run_production_acquisition(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    expected_implementation_sha: str,
    confirmation: str,
    sleep: Callable[[float], None] = lambda seconds: __import__("time").sleep(seconds),
) -> tuple[dict[str, dict[str, object]], int]:
    verify_production_preflight(
        repo_root,
        output_root,
        expected_implementation_sha=expected_implementation_sha,
        confirmation=confirmation,
    )
    return acquire_sources(output_root, fetcher=fetch_http_page, sleep=sleep)


def _safe_source_result(state_root: str | Path, source_key: str) -> dict[str, object]:
    try:
        pages = PageLockStore(state_root, source_key).read_locked_chain()
    except V9012Error:
        return {
            "source_role": source_role(source_key),
            "page_count": 0,
            "terminal": False,
            "first_missing_page": 1,
        }
    if not pages:
        return {
            "source_role": source_role(source_key),
            "page_count": 0,
            "terminal": False,
            "first_missing_page": 1,
        }
    result: dict[str, object] = {
        "source_role": source_role(source_key),
        "page_count": len(pages),
        "terminal": not pages[-1].continuation_issued,
    }
    if pages[-1].continuation_issued:
        result["first_missing_page"] = pages[-1].record["page_index"] + 1
    else:
        manifest = build_source_chain_manifest(source_key, pages)
        result["source_chain_sha256"] = source_chain_sha256(manifest, source_key)
    return result


def safe_acquisition_result(state_root: str | Path, requests: int) -> dict[str, object]:
    return {
        "schema_version": "V9_012_ACQUISITION_RESULT_V1",
        "status": "COMPLETE",
        "source_a": _safe_source_result(state_root, SOURCE_A),
        "source_b": _safe_source_result(state_root, SOURCE_B),
        "network_request_count": requests,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V9_012 actual TSE trading-day authority operations")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output-root", required=True)
    acquire.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    acquire.add_argument("--expected-implementation-sha", required=True)
    acquire.add_argument("--confirmation", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--state-root", required=True)
    materialize.add_argument("--output-root", required=True)
    materialize.add_argument("--acquisition-design-git-sha", required=True)
    materialize.add_argument("--acquisition-implementation-git-sha", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "acquire":
            _provenance, requests = run_production_acquisition(
                args.output_root,
                repo_root=args.repo_root,
                expected_implementation_sha=args.expected_implementation_sha,
                confirmation=args.confirmation,
            )
            print(canonical_json_bytes(safe_acquisition_result(args.output_root, requests)).decode("utf-8"), end="")
        else:
            result = materialize_sources(
                args.state_root,
                acquisition_design_git_sha=args.acquisition_design_git_sha,
                acquisition_implementation_git_sha=args.acquisition_implementation_git_sha,
            )
            write_materialized_artifacts(result, args.output_root)
            print(canonical_json_bytes(result.receipt).decode("utf-8"), end="")
        return 0
    except V9012Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.reason}, separators=(",", ":")), file=sys.stderr)
        return 2
    except Exception:
        print("V9_012_IMPLEMENTATION_FAILURE", file=sys.stderr)
        return 3


__all__ = [
    "AUTHORITATIVE_BRANCH", "CANONICAL_CONTENT_KEYS", "CANONICAL_JSON_NO_LF", "COVERED_END", "COVERED_START",
    "EXPECTED_EXCEPTION_SET", "FROZEN_BACKOFF_SECONDS", "FROZEN_RETRYABLE_CLASSES",
    "HUMAN_CONFIRMATION", "LockConflictError", "MAX_PRE_COMPLETE_ATTEMPTS",
    "MaterializedResult", "PageFetchResult", "PageLockStore", "PageRequest",
    "REVIEWED_DESIGN_BLOB_SHA", "REVIEWED_DESIGN_GIT_SHA", "SOURCE_A", "SOURCE_A_ENDPOINT",
    "SOURCE_A_BASE_QUERY", "SOURCE_A_BASE_QUERY_OBJECT", "SOURCE_A_ROLE", "SOURCE_B", "SOURCE_B_BASE_QUERY", "SOURCE_B_BASE_QUERY_OBJECT", "SOURCE_B_ENDPOINT", "SOURCE_B_ROLE", "STUDY",
    "V9012Error", "acquire_source", "acquire_sources", "base_query_object", "base_query_sha256",
    "build_source_chain_manifest", "canonical_json_bytes", "canonical_json_no_lf",
    "expected_request_url", "fetch_http_page", "main", "materialize_sources",
    "page_request_identity", "page_request_identity_sha256", "pagination_key_sha256",
    "safe_acquisition_result", "sha256_bytes", "sha256_utf8", "source_api_identity",
    "source_chain_sha256", "source_role", "validate_canonical_content", "validate_receipt",
    "validate_durable_source_order", "validate_source_a_rows", "validate_source_b_rows",
    "validate_source_chain_manifest", "validate_source_order_state",
    "verify_production_preflight", "write_materialized_artifacts",
]
