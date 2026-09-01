"""V9_011 J-Quants Trading Calendar acquisition and offline materialization.

The acquisition and materialization operations are deliberately separate.
Importing this module performs no network I/O, reads no credential, and does
not inspect any locked calendar payload.  The acquisition core accepts an
injected fetcher for synthetic tests; only the production seam binds that
core to the J-Quants endpoint after fail-closed provenance and environment
checks.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
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
    RETRYABLE_HTTP_CODES,
    V8CTransportBlocked,
    V8CTransportNamedFailure,
    attempt_with_frozen_retry,
)


STUDY = "V9_011_JQUANTS_TRADING_CALENDAR_SUCCESSOR"
ENDPOINT = "https://api.jquants.com/v2/markets/calendar"
API_CONTRACT_VERSION = "V2"
BASE_QUERY = {"from": "2017-01-01", "hol_div": None, "to": "2026-01-31"}
COVERED_START = "2017-01-01"
COVERED_END = "2026-01-31"
MINIMUM_EXPECTED_PLAN = "STANDARD"
API_KEY_ENVIRONMENT_VARIABLE = "JQUANTS_API_KEY"
HUMAN_CONFIRMATION = "CONFIRM_V9_011_JQUANTS_TRADING_CALENDAR_SUCCESSOR"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
REVIEWED_DESIGN_GIT_SHA = "c9ce98c720150a75afdfc9bba7ef0f9655bc942e"
REVIEWED_DESIGN_BLOB_SHA = "87b6c08423c7466f83da65229918952c3d579a68"
DESIGN_PATH = "V9_011_JQUANTS_TRADING_CALENDAR_SUCCESSOR_DESIGN.md"
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


PAGE_CHAIN_PROVENANCE_KEYS = frozenset({
    "schema_version", "base_query_identity_sha256", "endpoint_identity_sha256",
    "page_count", "pages", "terminal_page_index", "terminal_page_reached",
    "chain_lock_status", "semantic_processing_precondition",
})
PAGE_PROVENANCE_ENTRY_KEYS = frozenset({
    "page_index", "page_request_identity_sha256", "byte_count",
    "payload_sha256", "continuation_issued", "continuation_key_sha256",
})
SOURCE_CHAIN_MANIFEST_KEYS = frozenset({
    "base_query_identity_sha256", "endpoint_identity_sha256", "page_count",
    "pages", "terminal_page_index",
})
SOURCE_CHAIN_PAGE_KEYS = frozenset({
    "byte_count", "continuation_issued", "continuation_key_sha256",
    "page_index", "page_request_identity_sha256", "payload_sha256",
})
PROJECTED_CALENDAR_KEYS = frozenset({"covered_end", "covered_start", "rows"})
PROJECTED_ROW_KEYS = frozenset({"Date", "HolDiv"})
CANONICAL_CONTENT_KEYS = frozenset({
    "schema_version", "calendar_source_family", "covered_start", "covered_end",
    "trading_dates", "source_chain_sha256", "source_page_chain_provenance_sha256",
    "projected_calendar_sha256", "source_row_count", "trading_date_count",
    "acquisition_design_git_sha", "acquisition_implementation_git_sha",
    "api_contract_version", "endpoint_identity_sha256", "base_query_identity_sha256",
})
CANONICAL_HASH_RECEIPT_KEYS = frozenset({
    "schema_version", "status", "canonical_calendar_sha256",
})


def _canonical_json(value: object, *, final_lf: bool) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (rendered + ("\n" if final_lf else "")).encode("utf-8")


def identity_json_bytes(value: object) -> bytes:
    """Canonical identity JSON: UTF-8, sorted, and no final LF."""
    return _canonical_json(value, final_lf=False)


def canonical_json_bytes(value: object) -> bytes:
    """Canonical artifact JSON: UTF-8, sorted, and exactly one final LF."""
    return _canonical_json(value, final_lf=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_utf8(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


ENDPOINT_IDENTITY_SHA256 = sha256_utf8(ENDPOINT)
BASE_QUERY_IDENTITY_SHA256 = sha256_bytes(identity_json_bytes(BASE_QUERY))


class V9011Error(RuntimeError):
    """URL-, credential-, and payload-free fail-closed error."""

    def __init__(self, reason: str, *, attempts: int = 0, requests: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts
        self.network_request_count = requests


class LockConflictError(V9011Error):
    pass


@dataclass(frozen=True)
class PageRequest:
    page_index: int
    continuation_key: str | None = None

    def __post_init__(self) -> None:
        if type(self.page_index) is not int or self.page_index < 1:
            raise V9011Error("PAGE_REQUEST_INDEX_INVALID")
        if self.continuation_key is not None and (
            type(self.continuation_key) is not str or self.continuation_key == ""
        ):
            raise V9011Error("PAGINATION_KEY_INVALID")

    @property
    def params(self) -> dict[str, str]:
        result = {"from": COVERED_START, "to": COVERED_END}
        if self.continuation_key is not None:
            result["pagination_key"] = self.continuation_key
        return result


def expected_request_url(request: PageRequest) -> str:
    """Return the exact secret-free URL bound to this frozen page request."""
    query = [("from", COVERED_START), ("to", COVERED_END)]
    if request.continuation_key is not None:
        query.append(("pagination_key", request.continuation_key))
    return ENDPOINT + "?" + urllib.parse.urlencode(query)


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
class MaterializedCalendar:
    page_chain_provenance: dict[str, object]
    page_chain_provenance_bytes: bytes
    source_chain_manifest: dict[str, object]
    source_chain_sha256: str
    projected_calendar: dict[str, object]
    projected_calendar_sha256: str
    canonical_content: dict[str, object]
    canonical_bytes: bytes
    canonical_calendar_sha256: str
    canonical_hash_receipt: dict[str, object]


def _check_sha(value: object, pattern: re.Pattern[str], reason: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise V9011Error(reason)
    return value


def _strict_int(value: object, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def continuation_key_sha256(key: str) -> str:
    if type(key) is not str or key == "":
        raise V9011Error("PAGINATION_KEY_INVALID")
    return sha256_utf8(key)


def page_request_identity(request: PageRequest) -> dict[str, object]:
    key_digest = None if request.continuation_key is None else continuation_key_sha256(request.continuation_key)
    return {
        "base_query_identity_sha256": BASE_QUERY_IDENTITY_SHA256,
        "continuation_key_sha256": key_digest,
        "endpoint_identity_sha256": ENDPOINT_IDENTITY_SHA256,
        "page_index": request.page_index,
    }


def page_request_identity_sha256(request: PageRequest) -> str:
    return sha256_bytes(identity_json_bytes(page_request_identity(request)))


def validate_page_chain_provenance(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != PAGE_CHAIN_PROVENANCE_KEYS:
        raise V9011Error("PAGE_CHAIN_PROVENANCE_SCHEMA_INVALID")
    if value["schema_version"] != "V9_011_PAGE_CHAIN_PROVENANCE_V1":
        raise V9011Error("PAGE_CHAIN_PROVENANCE_BINDING_INVALID")
    if value["base_query_identity_sha256"] != BASE_QUERY_IDENTITY_SHA256 or value["endpoint_identity_sha256"] != ENDPOINT_IDENTITY_SHA256:
        raise V9011Error("PAGE_CHAIN_PROVENANCE_BINDING_INVALID")
    if not _strict_int(value["page_count"], minimum=1) or not _strict_int(value["terminal_page_index"], minimum=1):
        raise V9011Error("PAGE_CHAIN_PROVENANCE_COUNT_INVALID")
    if value["terminal_page_index"] != value["page_count"] or type(value["terminal_page_reached"]) is not bool or value["terminal_page_reached"] is not True:
        raise V9011Error("PAGE_CHAIN_PROVENANCE_TERMINAL_INVALID")
    if value["chain_lock_status"] != "COMPLETE" or value["semantic_processing_precondition"] != "ALL_PAGES_LOCKED_BEFORE_DATE_HOLDIV_INSPECTION":
        raise V9011Error("PAGE_CHAIN_PROVENANCE_ORDER_INVALID")
    pages = value["pages"]
    if type(pages) is not list or len(pages) != value["page_count"]:
        raise V9011Error("PAGE_CHAIN_PROVENANCE_COUNT_INVALID")
    for expected_index, page in enumerate(pages, 1):
        if type(page) is not dict or set(page) != PAGE_PROVENANCE_ENTRY_KEYS:
            raise V9011Error("PAGE_CHAIN_PROVENANCE_PAGE_SCHEMA_INVALID")
        if page["page_index"] != expected_index:
            raise V9011Error("PAGE_CHAIN_PROVENANCE_ORDER_INVALID")
        _check_sha(page["page_request_identity_sha256"], HEX64, "PAGE_REQUEST_IDENTITY_DIGEST_INVALID")
        _check_sha(page["payload_sha256"], HEX64, "PAGE_PAYLOAD_DIGEST_INVALID")
        if not _strict_int(page["byte_count"], minimum=0) or type(page["continuation_issued"]) is not bool:
            raise V9011Error("PAGE_CHAIN_PROVENANCE_PAGE_VALUE_INVALID")
        key_digest = page["continuation_key_sha256"]
        if page["continuation_issued"]:
            _check_sha(key_digest, HEX64, "PAGINATION_KEY_DIGEST_INVALID")
        elif key_digest is not None:
            raise V9011Error("PAGE_CHAIN_PROVENANCE_TERMINAL_INVALID")
        if expected_index == value["page_count"] and page["continuation_issued"]:
            raise V9011Error("PAGE_CHAIN_PROVENANCE_TERMINAL_INVALID")
        if expected_index < value["page_count"] and not page["continuation_issued"]:
            raise V9011Error("PAGE_CHAIN_PROVENANCE_TERMINAL_INVALID")
    return dict(value)


def validate_source_chain_manifest(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != SOURCE_CHAIN_MANIFEST_KEYS:
        raise V9011Error("SOURCE_CHAIN_MANIFEST_SCHEMA_INVALID")
    if value["base_query_identity_sha256"] != BASE_QUERY_IDENTITY_SHA256 or value["endpoint_identity_sha256"] != ENDPOINT_IDENTITY_SHA256:
        raise V9011Error("SOURCE_CHAIN_MANIFEST_BINDING_INVALID")
    if not _strict_int(value["page_count"], minimum=1) or value["terminal_page_index"] != value["page_count"]:
        raise V9011Error("SOURCE_CHAIN_MANIFEST_COUNT_INVALID")
    pages = value["pages"]
    if type(pages) is not list or len(pages) != value["page_count"]:
        raise V9011Error("SOURCE_CHAIN_MANIFEST_COUNT_INVALID")
    for expected_index, page in enumerate(pages, 1):
        if type(page) is not dict or set(page) != SOURCE_CHAIN_PAGE_KEYS or page["page_index"] != expected_index:
            raise V9011Error("SOURCE_CHAIN_MANIFEST_PAGE_INVALID")
        _check_sha(page["page_request_identity_sha256"], HEX64, "PAGE_REQUEST_IDENTITY_DIGEST_INVALID")
        _check_sha(page["payload_sha256"], HEX64, "PAGE_PAYLOAD_DIGEST_INVALID")
        if not _strict_int(page["byte_count"], minimum=0) or type(page["continuation_issued"]) is not bool:
            raise V9011Error("SOURCE_CHAIN_MANIFEST_PAGE_INVALID")
        if page["continuation_issued"]:
            _check_sha(page["continuation_key_sha256"], HEX64, "PAGINATION_KEY_DIGEST_INVALID")
        elif page["continuation_key_sha256"] is not None:
            raise V9011Error("SOURCE_CHAIN_MANIFEST_PAGE_INVALID")
        if expected_index == value["page_count"] and page["continuation_issued"]:
            raise V9011Error("SOURCE_CHAIN_MANIFEST_TERMINAL_INVALID")
        if expected_index < value["page_count"] and not page["continuation_issued"]:
            raise V9011Error("SOURCE_CHAIN_MANIFEST_TERMINAL_INVALID")
    return dict(value)


def source_chain_sha256(source_chain_manifest: object) -> str:
    return sha256_bytes(identity_json_bytes(validate_source_chain_manifest(source_chain_manifest)))


def validate_projected_calendar(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != PROJECTED_CALENDAR_KEYS:
        raise V9011Error("PROJECTED_CALENDAR_SCHEMA_INVALID")
    if value["covered_start"] != COVERED_START or value["covered_end"] != COVERED_END or type(value["rows"]) is not list:
        raise V9011Error("PROJECTED_CALENDAR_BINDING_INVALID")
    expected = _coverage_dates()
    seen: list[str] = []
    for row in value["rows"]:
        if type(row) is not dict or set(row) != PROJECTED_ROW_KEYS:
            raise V9011Error("PROJECTED_CALENDAR_ROW_SCHEMA_INVALID")
        date_value = row["Date"]
        hol_div = row["HolDiv"]
        if type(date_value) is not str or DATE_RE.fullmatch(date_value) is None or type(hol_div) is not str or hol_div not in {"0", "1", "2", "3"}:
            raise V9011Error("PROJECTED_CALENDAR_ROW_VALUE_INVALID")
        try:
            _dt.date.fromisoformat(date_value)
        except ValueError as exc:
            raise V9011Error("PROJECTED_CALENDAR_ROW_VALUE_INVALID") from exc
        seen.append(date_value)
    if seen != expected or len(set(seen)) != len(seen):
        raise V9011Error("PROJECTED_CALENDAR_COVERAGE_INVALID")
    if any(row["Date"] == "2020-10-01" and row["HolDiv"] in {"1", "2"} for row in value["rows"]):
        raise V9011Error("CALENDAR_SEMANTIC_SENTINEL_FAILURE")
    return dict(value)


def projected_calendar_sha256(projected_calendar: object) -> str:
    return sha256_bytes(identity_json_bytes(validate_projected_calendar(projected_calendar)))


def validate_canonical_content(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != CANONICAL_CONTENT_KEYS:
        raise V9011Error("CANONICAL_CONTENT_SCHEMA_INVALID")
    if value["schema_version"] != "V9_011_CANONICAL_TSE_TRADING_CALENDAR_V1" or value["calendar_source_family"] != "JPX_JQUANTS_API_V2_TRADING_CALENDAR":
        raise V9011Error("CANONICAL_CONTENT_BINDING_INVALID")
    if value["covered_start"] != COVERED_START or value["covered_end"] != COVERED_END or value["api_contract_version"] != API_CONTRACT_VERSION:
        raise V9011Error("CANONICAL_CONTENT_BINDING_INVALID")
    _check_sha(value["source_chain_sha256"], HEX64, "CANONICAL_CONTENT_DIGEST_INVALID")
    _check_sha(value["source_page_chain_provenance_sha256"], HEX64, "CANONICAL_CONTENT_DIGEST_INVALID")
    _check_sha(value["projected_calendar_sha256"], HEX64, "CANONICAL_CONTENT_DIGEST_INVALID")
    if value["endpoint_identity_sha256"] != ENDPOINT_IDENTITY_SHA256 or value["base_query_identity_sha256"] != BASE_QUERY_IDENTITY_SHA256:
        raise V9011Error("CANONICAL_CONTENT_BINDING_INVALID")
    _check_sha(value["acquisition_design_git_sha"], HEX40, "CANONICAL_CONTENT_GIT_SHA_INVALID")
    _check_sha(value["acquisition_implementation_git_sha"], HEX40, "CANONICAL_CONTENT_GIT_SHA_INVALID")
    if not _strict_int(value["source_row_count"], minimum=0) or not _strict_int(value["trading_date_count"], minimum=0) or type(value["trading_dates"]) is not list:
        raise V9011Error("CANONICAL_CONTENT_COUNT_INVALID")
    trading_dates = value["trading_dates"]
    if value["trading_date_count"] != len(trading_dates):
        raise V9011Error("CANONICAL_CONTENT_COUNT_INVALID")
    if any(type(date_value) is not str or DATE_RE.fullmatch(date_value) is None for date_value in trading_dates):
        raise V9011Error("CANONICAL_CONTENT_TRADING_DATES_INVALID")
    if trading_dates != sorted(trading_dates) or len(set(trading_dates)) != len(trading_dates):
        raise V9011Error("CANONICAL_CONTENT_TRADING_DATES_INVALID")
    coverage = set(_coverage_dates())
    if any(date_value not in coverage for date_value in trading_dates):
        raise V9011Error("CANONICAL_CONTENT_TRADING_DATES_INVALID")
    return dict(value)


def validate_canonical_hash_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != CANONICAL_HASH_RECEIPT_KEYS:
        raise V9011Error("CANONICAL_HASH_RECEIPT_SCHEMA_INVALID")
    if value["schema_version"] != "V9_011_CANONICAL_HASH_RECEIPT_V1" or value["status"] != "COMPLETE":
        raise V9011Error("CANONICAL_HASH_RECEIPT_BINDING_INVALID")
    _check_sha(value["canonical_calendar_sha256"], HEX64, "CANONICAL_HASH_RECEIPT_DIGEST_INVALID")
    return dict(value)


def _coverage_dates() -> list[str]:
    start = _dt.date.fromisoformat(COVERED_START)
    end = _dt.date.fromisoformat(COVERED_END)
    return [(start + _dt.timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]


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


class PageLockStore:
    """External durable raw pages, lock records, and private envelopes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.payload_dir = self.root / "raw_pages"
        self.lock_dir = self.root / "page_locks"
        self.envelope_dir = self.root / "page_envelopes"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.payload_dir.mkdir(exist_ok=True)
            self.lock_dir.mkdir(exist_ok=True)
            self.envelope_dir.mkdir(exist_ok=True)
        except Exception as exc:
            raise V9011Error("DURABLE_STATE_INITIALIZATION_FAILURE") from exc
        allowed = {"raw_pages", "page_locks", "page_envelopes"}
        try:
            if any(child.name not in allowed for child in self.root.iterdir()):
                raise V9011Error("DURABLE_STATE_UNEXPECTED_FILE")
        except V9011Error:
            raise
        except Exception as exc:
            raise V9011Error("DURABLE_STATE_ENUMERATION_FAILURE") from exc

    def _payload_path(self, index: int) -> Path:
        return self.payload_dir / f"{index:06d}.bin"

    def _lock_path(self, index: int) -> Path:
        return self.lock_dir / f"{index:06d}.json"

    def _envelope_path(self, index: int) -> Path:
        return self.envelope_dir / f"{index:06d}.json"

    def lock_page(self, request: PageRequest, result: PageFetchResult) -> dict[str, object]:
        if type(result.payload) is not bytes or type(result.http_status) is not int or type(result.resolved_url) is not str:
            raise V9011Error("TRANSPORT_RESPONSE_TYPE_INVALID")
        if result.http_status != 200 or result.resolved_url != expected_request_url(request):
            raise V9011Error("TRANSPORT_RESPONSE_NOT_LOCKABLE")
        record: dict[str, object] = {
            "page_index": request.page_index,
            "page_request_identity_sha256": page_request_identity_sha256(request),
            "byte_count": len(result.payload),
            "payload_sha256": sha256_bytes(result.payload),
        }
        _exclusive_write(self._payload_path(request.page_index), result.payload)
        _exclusive_write(self._lock_path(request.page_index), canonical_json_bytes(record))
        return record

    def persist_envelope(self, index: int, continuation_key: str | None) -> None:
        if continuation_key is not None:
            continuation_key_sha256(continuation_key)
            value: dict[str, object] = {"pagination_key": continuation_key}
        else:
            value = {}
        _exclusive_write(self._envelope_path(index), canonical_json_bytes(value))

    def _read_envelope(self, index: int) -> tuple[bool, str | None]:
        path = self._envelope_path(index)
        if not path.exists():
            raise V9011Error("PAGINATION_ENVELOPE_MISSING")
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise V9011Error("PAGINATION_ENVELOPE_MALFORMED") from exc
        if type(value) is not dict or set(value) not in (set(), {"pagination_key"}) or raw != canonical_json_bytes(value):
            raise V9011Error("PAGINATION_ENVELOPE_MALFORMED")
        if "pagination_key" not in value:
            return False, None
        key = value["pagination_key"]
        if type(key) is not str or key == "":
            raise V9011Error("PAGINATION_KEY_INVALID")
        return True, key

    def read_locked_chain(self, *, require_terminal: bool = False) -> list[LockedPage]:
        try:
            lock_paths = list(self.lock_dir.iterdir())
            payload_paths = list(self.payload_dir.iterdir())
            envelope_paths = list(self.envelope_dir.iterdir())
        except Exception as exc:
            raise V9011Error("DURABLE_STATE_ENUMERATION_FAILURE") from exc
        def valid_page_name(path: Path, suffix: str) -> bool:
            if not path.is_file() or path.suffix != suffix:
                return False
            if not path.stem.isdigit() or int(path.stem) < 1:
                return False
            return path.name == f"{int(path.stem):06d}{suffix}"

        for path in lock_paths:
            if not valid_page_name(path, ".json"):
                raise V9011Error("DURABLE_STATE_EXTRA_LOCK")
        for path in payload_paths:
            if not path.is_file() or path.suffix != ".bin" or not path.stem.isdigit() or int(path.stem) < 1:
                raise V9011Error("DURABLE_STATE_EXTRA_PAYLOAD")
        for path in envelope_paths:
            if not valid_page_name(path, ".json"):
                raise V9011Error("DURABLE_STATE_EXTRA_ENVELOPE")
        indices = sorted(int(path.stem) for path in lock_paths)
        if not indices:
            if payload_paths or envelope_paths:
                raise V9011Error("DURABLE_STATE_INCOMPLETE_PAIR")
            if require_terminal:
                raise V9011Error("PAGE_CHAIN_EMPTY")
            return []
        if indices != list(range(1, max(indices) + 1)):
            raise V9011Error("PAGE_CHAIN_ORDER_INVALID")
        if {int(path.stem) for path in payload_paths} != set(indices) or {int(path.stem) for path in envelope_paths} != set(indices):
            raise V9011Error("DURABLE_STATE_INCOMPLETE_PAIR")
        result: list[LockedPage] = []
        previous_key: str | None = None
        seen_keys: set[str] = set()
        for index in indices:
            try:
                raw_record = self._lock_path(index).read_bytes()
                record = json.loads(raw_record.decode("utf-8"))
                payload = self._payload_path(index).read_bytes()
            except Exception as exc:
                raise V9011Error("DURABLE_STATE_LOCK_READ_FAILURE") from exc
            if type(record) is not dict or set(record) != {"page_index", "page_request_identity_sha256", "byte_count", "payload_sha256"}:
                raise V9011Error("PAGE_LOCK_SCHEMA_INVALID")
            request = PageRequest(index, previous_key)
            if record["page_index"] != index or record["page_request_identity_sha256"] != page_request_identity_sha256(request):
                raise V9011Error("PAGE_LOCK_REQUEST_BINDING_MISMATCH")
            _check_sha(record["payload_sha256"], HEX64, "PAGE_LOCK_DIGEST_INVALID")
            if not _strict_int(record["byte_count"], minimum=0) or record["byte_count"] != len(payload) or record["payload_sha256"] != sha256_bytes(payload) or raw_record != canonical_json_bytes(record):
                raise V9011Error("PAGE_LOCK_PAYLOAD_MISMATCH")
            continuation, key = self._read_envelope(index)
            if continuation:
                if key in seen_keys:
                    raise V9011Error("PAGINATION_KEY_REPEATED")
                seen_keys.add(key)  # type: ignore[arg-type]
                previous_key = key
            else:
                previous_key = None
            result.append(LockedPage(dict(record), payload, continuation, key))
            if not continuation and index != indices[-1]:
                raise V9011Error("PAGE_CHAIN_AFTER_TERMINAL")
        if require_terminal and not result[-1].continuation_issued:
            return result
        if require_terminal:
            raise V9011Error("PAGE_CHAIN_TERMINAL_MISSING")
        return result

    def read_page_payload(self, index: int) -> bytes:
        chain = self.read_locked_chain()
        for page in chain:
            if page.record["page_index"] == index:
                return page.payload
        raise V9011Error("PAGE_LOCK_MISSING")


def _transport_failure(exc: BaseException, requests: int) -> V9011Error:
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
    return V9011Error(reason, attempts=requests, requests=requests)


def _validate_transport_result(request: PageRequest, result: object) -> PageFetchResult:
    if type(result) is not PageFetchResult:
        raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
    if type(result.payload) is not bytes or type(result.http_status) is not int or type(result.resolved_url) is not str:
        raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
    if result.http_status != 200:
        if result.http_status in RETRYABLE_HTTP_CODES or result.http_status in {400, 401, 403, 404, 410, 422}:
            raise urllib.error.HTTPError(ENDPOINT, result.http_status, "", {}, None)
        if 300 <= result.http_status < 400:
            raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")
        raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
    if result.resolved_url != expected_request_url(request):
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
    return result


def _inspect_pagination_envelope(payload: bytes) -> tuple[bool, str | None]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise V9011Error("PAGINATION_ENVELOPE_MALFORMED") from exc
    if type(value) is not dict:
        raise V9011Error("PAGINATION_ENVELOPE_MALFORMED")
    if "pagination_key" not in value:
        return False, None
    key = value["pagination_key"]
    if type(key) is not str or key == "":
        raise V9011Error("PAGINATION_KEY_INVALID")
    return True, key


def acquire_page_chain(
    state_root: str | Path,
    *,
    fetcher: Callable[[PageRequest], PageFetchResult],
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> tuple[dict[str, object], int]:
    """Acquire the complete page chain; no Date/HolDiv inspection occurs."""
    store = PageLockStore(state_root)
    locked = store.read_locked_chain()
    requests = 0
    seen_keys: set[str] = set()
    next_request = PageRequest(1, None)
    if locked:
        for page in locked:
            if page.continuation_issued:
                if page.continuation_key in seen_keys:
                    raise V9011Error("PAGINATION_KEY_REPEATED")
                seen_keys.add(page.continuation_key)  # type: ignore[arg-type]
                next_request = PageRequest(page.record["page_index"] + 1, page.continuation_key)
            else:
                next_request = PageRequest(page.record["page_index"] + 1, None)
        if not locked[-1].continuation_issued:
            provenance = build_page_chain_provenance(locked)
            return provenance, 0
    while True:
        current_request = next_request
        try:
            def attempt() -> PageFetchResult:
                nonlocal requests
                requests += 1
                return _validate_transport_result(current_request, fetcher(current_request))

            result, _audit = attempt_with_frozen_retry(
                attempt,
                sleep_fn=sleep,
                request_fingerprint=page_request_identity_sha256(current_request),
            )
        except (V8CTransportNamedFailure, V8CTransportBlocked) as exc:
            raise _transport_failure(exc, requests) from exc
        except BaseException as exc:
            raise _transport_failure(exc, requests) from exc
        try:
            store.lock_page(current_request, result)
            continued, key = _inspect_pagination_envelope(result.payload)
            store.persist_envelope(current_request.page_index, key if continued else None)
            if continued and key in seen_keys:
                raise V9011Error("PAGINATION_KEY_REPEATED", attempts=1, requests=requests)
            if not continued:
                pages = store.read_locked_chain(require_terminal=True)
                return build_page_chain_provenance(pages), requests
            seen_keys.add(key)  # type: ignore[arg-type]
            next_request = PageRequest(current_request.page_index + 1, key)
        except V9011Error as exc:
            exc.network_request_count = requests
            raise


def build_page_chain_provenance(pages: Sequence[LockedPage]) -> dict[str, object]:
    if not pages:
        raise V9011Error("PAGE_CHAIN_EMPTY")
    entries: list[dict[str, object]] = []
    for page in pages:
        entries.append({
            "page_index": page.record["page_index"],
            "page_request_identity_sha256": page.record["page_request_identity_sha256"],
            "byte_count": page.record["byte_count"],
            "payload_sha256": page.record["payload_sha256"],
            "continuation_issued": page.continuation_issued,
            "continuation_key_sha256": None if page.continuation_key is None else continuation_key_sha256(page.continuation_key),
        })
    value = {
        "schema_version": "V9_011_PAGE_CHAIN_PROVENANCE_V1",
        "base_query_identity_sha256": BASE_QUERY_IDENTITY_SHA256,
        "endpoint_identity_sha256": ENDPOINT_IDENTITY_SHA256,
        "page_count": len(entries),
        "pages": entries,
        "terminal_page_index": len(entries),
        "terminal_page_reached": True,
        "chain_lock_status": "COMPLETE",
        "semantic_processing_precondition": "ALL_PAGES_LOCKED_BEFORE_DATE_HOLDIV_INSPECTION",
    }
    return validate_page_chain_provenance(value)


def page_chain_provenance_bytes(provenance: object) -> bytes:
    return canonical_json_bytes(validate_page_chain_provenance(provenance))


def build_source_chain_manifest(provenance: object) -> dict[str, object]:
    checked = validate_page_chain_provenance(provenance)
    pages = []
    for page in checked["pages"]:
        pages.append({
            "byte_count": page["byte_count"],
            "continuation_issued": page["continuation_issued"],
            "continuation_key_sha256": page["continuation_key_sha256"],
            "page_index": page["page_index"],
            "page_request_identity_sha256": page["page_request_identity_sha256"],
            "payload_sha256": page["payload_sha256"],
        })
    return validate_source_chain_manifest({
        "base_query_identity_sha256": BASE_QUERY_IDENTITY_SHA256,
        "endpoint_identity_sha256": ENDPOINT_IDENTITY_SHA256,
        "page_count": checked["page_count"],
        "pages": pages,
        "terminal_page_index": checked["terminal_page_index"],
    })


def _project_locked_pages(pages: Sequence[LockedPage]) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for page in pages:
        try:
            value = json.loads(page.payload.decode("utf-8"))
        except Exception as exc:
            raise V9011Error("PARSER_SCHEMA_FAILURE") from exc
        if type(value) is not dict or type(value.get("data")) is not list:
            raise V9011Error("PARSER_SCHEMA_FAILURE")
        for row in value["data"]:
            if type(row) is not dict or "Date" not in row or "HolDiv" not in row:
                raise V9011Error("PARSER_SCHEMA_FAILURE")
            date_value = row["Date"]
            hol_div = row["HolDiv"]
            if type(date_value) is not str or DATE_RE.fullmatch(date_value) is None or type(hol_div) is not str or hol_div not in {"0", "1", "2", "3"}:
                raise V9011Error("DATA_QUALITY_FAILURE")
            try:
                _dt.date.fromisoformat(date_value)
            except ValueError as exc:
                raise V9011Error("DATA_QUALITY_FAILURE") from exc
            rows.append({"Date": date_value, "HolDiv": hol_div})
    rows.sort(key=lambda row: row["Date"])
    projected = {"covered_end": COVERED_END, "covered_start": COVERED_START, "rows": rows}
    checked = validate_projected_calendar(projected)
    checked["rows"] = rows
    checked["trading_dates"] = [row["Date"] for row in rows if row["HolDiv"] in {"1", "2"}]
    return checked


def build_canonical_hash_receipt(canonical_calendar_sha: str) -> dict[str, object]:
    _check_sha(canonical_calendar_sha, HEX64, "CANONICAL_HASH_RECEIPT_DIGEST_INVALID")
    return validate_canonical_hash_receipt({
        "schema_version": "V9_011_CANONICAL_HASH_RECEIPT_V1",
        "status": "COMPLETE",
        "canonical_calendar_sha256": canonical_calendar_sha,
    })


def materialize_calendar(
    state_root: str | Path,
    *,
    acquisition_design_git_sha: str,
    acquisition_implementation_git_sha: str,
) -> MaterializedCalendar:
    """Read only a complete locked chain; this function has no fetcher/network path."""
    design_sha = _check_sha(acquisition_design_git_sha, HEX40, "ACQUISITION_DESIGN_GIT_SHA_INVALID")
    implementation_sha = _check_sha(acquisition_implementation_git_sha, HEX40, "ACQUISITION_IMPLEMENTATION_GIT_SHA_INVALID")
    store = PageLockStore(state_root)
    pages = store.read_locked_chain(require_terminal=True)
    provenance = build_page_chain_provenance(pages)
    provenance_bytes = page_chain_provenance_bytes(provenance)
    provenance_sha = sha256_bytes(provenance_bytes)
    source_manifest = build_source_chain_manifest(provenance)
    source_sha = source_chain_sha256(source_manifest)
    projected_with_dates = _project_locked_pages(pages)
    rows = projected_with_dates["rows"]
    projected = {
        "covered_end": COVERED_END,
        "covered_start": COVERED_START,
        "rows": rows,
    }
    projected = validate_projected_calendar(projected)
    projected_sha = projected_calendar_sha256(projected)
    trading_dates = [row["Date"] for row in rows if row["HolDiv"] in {"1", "2"}]
    content: dict[str, object] = {
        "schema_version": "V9_011_CANONICAL_TSE_TRADING_CALENDAR_V1",
        "calendar_source_family": "JPX_JQUANTS_API_V2_TRADING_CALENDAR",
        "covered_start": COVERED_START,
        "covered_end": COVERED_END,
        "trading_dates": trading_dates,
        "source_chain_sha256": source_sha,
        "source_page_chain_provenance_sha256": provenance_sha,
        "projected_calendar_sha256": projected_sha,
        "source_row_count": len(rows),
        "trading_date_count": len(trading_dates),
        "acquisition_design_git_sha": design_sha,
        "acquisition_implementation_git_sha": implementation_sha,
        "api_contract_version": API_CONTRACT_VERSION,
        "endpoint_identity_sha256": ENDPOINT_IDENTITY_SHA256,
        "base_query_identity_sha256": BASE_QUERY_IDENTITY_SHA256,
    }
    content = validate_canonical_content(content)
    canonical_bytes = canonical_json_bytes(content)
    if b"canonical_calendar_sha256" in canonical_bytes:
        raise V9011Error("CANONICAL_CONTENT_SELF_REFERENCE")
    canonical_sha = sha256_bytes(canonical_bytes)
    receipt = build_canonical_hash_receipt(canonical_sha)
    return MaterializedCalendar(
        provenance, provenance_bytes, source_manifest, source_sha,
        projected, projected_sha, content, canonical_bytes, canonical_sha, receipt,
    )


def write_materialized_artifacts(result: MaterializedCalendar, output_root: str | Path) -> None:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _exclusive_write(root / "V9_011_PAGE_CHAIN_PROVENANCE_V1.json", result.page_chain_provenance_bytes)
    _exclusive_write(root / "V9_011_CANONICAL_TSE_TRADING_CALENDAR_V1.json", result.canonical_bytes)
    _exclusive_write(root / "V9_011_CANONICAL_HASH_RECEIPT_V1.json", canonical_json_bytes(result.canonical_hash_receipt))


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
        raise V9011Error("API_KEY_MISSING")
    return urllib.request.Request(
        expected_request_url(request), headers={"x-api-key": api_key}, method="GET"
    )


def fetch_http_page(request: PageRequest) -> PageFetchResult:
    """The only production network function; redirects are disabled."""
    verify_protected_environment(Path(__file__).resolve().parents[1])
    api_key = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
    if type(api_key) is not str or api_key == "":
        raise V9011Error("API_KEY_MISSING")
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(_build_request(request, api_key), timeout=30.0) as response:
        return PageFetchResult(response.read(), response.getcode(), response.geturl())


def _default_git_runner(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def verify_protected_environment(repo_root: str | Path) -> dict[str, object]:
    from src.v9_010_jpx_calendar_stage_a_acquisition import verify_protected_environment as verify

    return verify(repo_root)


def _credential_exists() -> bool:
    value = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
    return type(value) is str and value != ""


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
        raise V9011Error("FRESH_HUMAN_CONFIRMATION_INVALID")
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    if output == root or root in output.parents:
        raise V9011Error("DURABLE_STATE_MUST_BE_EXTERNAL")
    if output.exists():
        raise V9011Error("DURABLE_EXECUTION_ROOT_COLLISION")
    runner = git_runner or (lambda args: _default_git_runner(root, args))
    try:
        branch = runner(["branch", "--show-current"])
        local = runner(["rev-parse", "HEAD"])
        remote = runner(["rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"])
        status = runner(["status", "--porcelain=v1", "--untracked-files=all"])
        design_commit = runner(["rev-parse", f"{REVIEWED_DESIGN_GIT_SHA}^{{commit}}"])
        design_blob = runner(["rev-parse", f"{REVIEWED_DESIGN_GIT_SHA}:{DESIGN_PATH}"])
    except Exception as exc:
        raise V9011Error("GIT_PROVENANCE_UNAVAILABLE") from exc
    if branch != AUTHORITATIVE_BRANCH or local != implementation_sha or remote != implementation_sha:
        raise V9011Error("GIT_PROVENANCE_MISMATCH")
    if status != "":
        raise V9011Error("GIT_WORKTREE_DIRTY")
    if design_commit != REVIEWED_DESIGN_GIT_SHA or design_blob != REVIEWED_DESIGN_BLOB_SHA:
        raise V9011Error("DESIGN_BINDING_MISMATCH")
    if not _credential_exists():
        raise V9011Error("API_KEY_MISSING")
    try:
        environment = verify_protected_environment(root)
    except V9011Error:
        raise
    except Exception as exc:
        raise V9011Error("PROTECTED_ENVIRONMENT_CHECK_FAILURE") from exc
    return {
        "authoritative_branch": branch,
        "implementation_git_sha": implementation_sha,
        "local_head": local,
        "remote_tracking_head": remote,
        "reviewed_design_git_sha": REVIEWED_DESIGN_GIT_SHA,
        "design_blob_sha256": REVIEWED_DESIGN_BLOB_SHA,
        "api_contract_version": API_CONTRACT_VERSION,
        "base_query_identity_sha256": BASE_QUERY_IDENTITY_SHA256,
        "endpoint_identity_sha256": ENDPOINT_IDENTITY_SHA256,
        "purchase_authorized": False,
        "network_authorized": True,
        "credential_present": True,
        **environment,
    }


def run_production_acquisition(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    expected_implementation_sha: str,
    confirmation: str,
    sleep: Callable[[float], None] = lambda seconds: __import__("time").sleep(seconds),
) -> tuple[dict[str, object], int]:
    """Future-only production seam; no environment bypass is exposed."""
    verify_production_preflight(
        repo_root,
        output_root,
        expected_implementation_sha=expected_implementation_sha,
        confirmation=confirmation,
    )
    return acquire_page_chain(output_root, fetcher=fetch_http_page, sleep=sleep)


def _safe_acquisition_result(provenance: Mapping[str, object], requests: int) -> dict[str, object]:
    return {
        "schema_version": "V9_011_ACQUISITION_RESULT_V1",
        "status": "COMPLETE",
        "page_count": provenance["page_count"],
        "source_page_chain_provenance_sha256": sha256_bytes(page_chain_provenance_bytes(provenance)),
        "network_request_count": requests,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V9_011 J-Quants calendar operations")
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
            provenance, requests = run_production_acquisition(
                args.output_root,
                repo_root=args.repo_root,
                expected_implementation_sha=args.expected_implementation_sha,
                confirmation=args.confirmation,
            )
            print(canonical_json_bytes(_safe_acquisition_result(provenance, requests)).decode("utf-8"), end="")
        else:
            result = materialize_calendar(
                args.state_root,
                acquisition_design_git_sha=args.acquisition_design_git_sha,
                acquisition_implementation_git_sha=args.acquisition_implementation_git_sha,
            )
            write_materialized_artifacts(result, args.output_root)
            print(canonical_json_bytes(result.canonical_hash_receipt).decode("utf-8"), end="")
        return 0
    except V9011Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.reason}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except Exception:
        print("V9_011_IMPLEMENTATION_FAILURE", file=sys.stderr)
        return 3


__all__ = [
    "API_CONTRACT_VERSION", "AUTHORITATIVE_BRANCH", "BASE_QUERY",
    "BASE_QUERY_IDENTITY_SHA256", "CANONICAL_CONTENT_KEYS", "COVERED_END",
    "COVERED_START", "ENDPOINT", "ENDPOINT_IDENTITY_SHA256", "FROZEN_BACKOFF_SECONDS",
    "FROZEN_RETRYABLE_CLASSES", "HUMAN_CONFIRMATION", "LockConflictError",
    "MaterializedCalendar", "MAX_PRE_COMPLETE_ATTEMPTS", "MINIMUM_EXPECTED_PLAN",
    "PageFetchResult", "PageLockStore", "PageRequest", "PROJECTED_CALENDAR_KEYS",
    "REVIEWED_DESIGN_BLOB_SHA", "REVIEWED_DESIGN_GIT_SHA", "SOURCE_CHAIN_MANIFEST_KEYS",
    "V9011Error", "acquire_page_chain", "build_canonical_hash_receipt",
    "build_page_chain_provenance", "build_source_chain_manifest", "canonical_json_bytes",
    "continuation_key_sha256", "expected_request_url", "fetch_http_page", "identity_json_bytes", "main",
    "materialize_calendar", "page_chain_provenance_bytes", "page_request_identity",
    "page_request_identity_sha256", "projected_calendar_sha256", "run_production_acquisition",
    "sha256_bytes", "sha256_utf8", "source_chain_sha256", "validate_canonical_content",
    "validate_canonical_hash_receipt", "validate_page_chain_provenance", "validate_projected_calendar",
    "validate_source_chain_manifest", "verify_protected_environment", "verify_production_preflight",
    "write_materialized_artifacts",
]
