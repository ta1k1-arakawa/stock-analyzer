"""V9_010 Stage-A JPX calendar manifest and raw-payload acquisition.

Stage A deliberately has no calendar parser.  It binds one exact URL per
month, retries only the frozen V9_006 transport classes before the first
complete payload, and durably locks the first complete HTTP-200 bytes before
any content inspection.  The production network seam is
``run_production_acquisition``; importing this module never performs I/O to a
network and the offline core accepts only an injected fetcher.
"""

from __future__ import annotations

import argparse
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


TASK = "V9_010_STAGE_A_JPX_CALENDAR_RAW_ACQUISITION_IMPLEMENTATION"
MANIFEST_FILENAME = "V9_010_STAGE_A_JPX_CALENDAR_SOURCE_MANIFEST.json"
DESIGN_PATH = "V9_010_T0_HISTORICAL_JPX_CALENDAR_BINDING_DESIGN.md"
REVIEWED_DESIGN_GIT_SHA = "79d97256621edf9689705d17ac264c4f8d4ecc37"
REVIEWED_DESIGN_BLOB_GIT_SHA = "095ea988f8559c9e04fdfcf8953e7ca8f83e6a4d"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
MANIFEST_SHA256 = "f4ff70170cff064da3ac6fb5f6086339b8c232708b5640ca0f5375976e8ed5a4"
SOURCE_HOST = "www.jpx.co.jp"
SOURCE_SCHEME = "https"
SOURCE_URL_TEMPLATE = "https://www.jpx.co.jp/calendar/{year:04d}{month:02d}.html"
FIRST_SOURCE_YEAR_MONTH = (2017, 1)
LAST_SOURCE_YEAR_MONTH = (2026, 1)
SOURCE_SLOT_COUNT = 109
FALLBACK_SOURCE_OBJECTS = 0

MAX_PRE_COMPLETE_ATTEMPTS = 3
MAX_PRE_COMPLETE_RETRIES = 2
FROZEN_BACKOFF_SECONDS = (5, 30)
FROZEN_RETRYABLE_CLASSES = frozenset({
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
HUMAN_CONFIRMATION = "CONFIRM_V9_010_STAGE_A_JPX_CALENDAR_RAW_ACQUISITION"
LOCK_SCHEMA_KEYS = frozenset({
    "source_slot",
    "source_url_sha256",
    "http_status",
    "byte_count",
    "payload_sha256",
})
SAFE_RECEIPT_KEYS = frozenset({
    "schema_version",
    "task",
    "status",
    "source_manifest_sha256",
    "raw_lock_set_sha256",
    "raw_lock_count",
    "network_request_count",
})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SLOT_RE = re.compile(r"^(20[0-9]{2})-(0[1-9]|1[0-2])$")

assert MAXIMUM_ATTEMPTS_PER_TICKER == MAX_PRE_COMPLETE_ATTEMPTS
assert len(BACKOFF_SECONDS) == MAX_PRE_COMPLETE_RETRIES
assert tuple(BACKOFF_SECONDS) == FROZEN_BACKOFF_SECONDS


class StageAError(RuntimeError):
    """Safe, URL-free failure from the Stage-A implementation."""

    def __init__(self, reason: str, *, source_slot: str | None = None, attempts: int = 0, requests: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.source_slot = source_slot
        self.attempts = attempts
        self.network_request_count = requests


class LockConflictError(StageAError):
    pass


@dataclass(frozen=True)
class FetchResult:
    """One transport-complete response returned by the injected seam."""

    payload: bytes
    http_status: int
    resolved_url: str


def canonical_json_bytes(value: object) -> bytes:
    """The V9_010 canonical JSON byte procedure, including one final LF."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _is_sha(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _month_slots() -> tuple[str, ...]:
    slots: list[str] = []
    year, month = FIRST_SOURCE_YEAR_MONTH
    while (year, month) <= LAST_SOURCE_YEAR_MONTH:
        slots.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return tuple(slots)


def expected_manifest() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for slot in _month_slots():
        year, month = (int(part) for part in slot.split("-"))
        url = SOURCE_URL_TEMPLATE.format(year=year, month=month)
        result.append({
            "source_slot": slot,
            "source_url": url,
            "source_url_sha256": sha256_bytes(url.encode("utf-8")),
        })
    return result


def validate_manifest_url(url: object) -> None:
    if type(url) is not str:
        raise StageAError("MANIFEST_URL_TYPE_INVALID")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != SOURCE_SCHEME or parsed.hostname != SOURCE_HOST or parsed.netloc != SOURCE_HOST or parsed.port is not None:
        raise StageAError("MANIFEST_URL_IDENTITY_INVALID")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise StageAError("MANIFEST_URL_COMPONENT_INVALID")
    if not re.fullmatch(r"/calendar/20[0-9]{4}\.html", parsed.path):
        raise StageAError("MANIFEST_URL_PATH_INVALID")


def validate_manifest(value: object) -> list[dict[str, str]]:
    if type(value) is not list:
        raise StageAError("MANIFEST_SCHEMA_INVALID")
    expected = expected_manifest()
    if len(value) != SOURCE_SLOT_COUNT or len(expected) != SOURCE_SLOT_COUNT:
        raise StageAError("MANIFEST_COUNT_INVALID")
    normalized: list[dict[str, str]] = []
    for item, expected_item in zip(value, expected):
        if type(item) is not dict or set(item) != {"source_slot", "source_url", "source_url_sha256"}:
            raise StageAError("MANIFEST_SCHEMA_INVALID")
        if any(type(item[key]) is not str for key in item):
            raise StageAError("MANIFEST_FIELD_TYPE_INVALID")
        slot = item["source_slot"]
        if SLOT_RE.fullmatch(slot) is None:
            raise StageAError("MANIFEST_SLOT_INVALID")
        validate_manifest_url(item["source_url"])
        if item["source_url_sha256"] != sha256_bytes(item["source_url"].encode("utf-8")):
            raise StageAError("MANIFEST_URL_DIGEST_INVALID")
        if item != expected_item:
            raise StageAError("MANIFEST_FROZEN_BINDING_MISMATCH")
        normalized.append(dict(item))
    if normalized != expected or len({item["source_slot"] for item in normalized}) != SOURCE_SLOT_COUNT:
        raise StageAError("MANIFEST_ORDER_OR_UNIQUENESS_INVALID")
    return normalized


def load_manifest(path: str | Path) -> tuple[list[dict[str, str]], str]:
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise StageAError("MANIFEST_READ_OR_JSON_INVALID") from exc
    manifest = validate_manifest(value)
    canonical = canonical_json_bytes(manifest)
    if raw != canonical or sha256_bytes(canonical) != MANIFEST_SHA256:
        raise StageAError("MANIFEST_CANONICAL_DIGEST_MISMATCH")
    return manifest, MANIFEST_SHA256


def validate_raw_lock_record(record: object, manifest_item: Mapping[str, str], payload: bytes | None = None) -> dict[str, object]:
    if type(record) is not dict or set(record) != LOCK_SCHEMA_KEYS:
        raise StageAError("RAW_LOCK_SCHEMA_INVALID", source_slot=manifest_item["source_slot"])
    if type(record["source_slot"]) is not str or type(record["source_url_sha256"]) is not str:
        raise StageAError("RAW_LOCK_FIELD_TYPE_INVALID", source_slot=manifest_item["source_slot"])
    if not _is_exact_int(record["http_status"]) or not _is_exact_int(record["byte_count"]):
        raise StageAError("RAW_LOCK_FIELD_TYPE_INVALID", source_slot=manifest_item["source_slot"])
    if not _is_sha(record["payload_sha256"], HEX64):
        raise StageAError("RAW_LOCK_DIGEST_INVALID", source_slot=manifest_item["source_slot"])
    if record["source_slot"] != manifest_item["source_slot"] or record["source_url_sha256"] != manifest_item["source_url_sha256"]:
        raise StageAError("RAW_LOCK_ENDPOINT_BINDING_MISMATCH", source_slot=manifest_item["source_slot"])
    if record["http_status"] != 200 or record["byte_count"] <= 0:
        raise StageAError("RAW_LOCK_COMPLETENESS_INVALID", source_slot=manifest_item["source_slot"])
    if payload is not None and (record["byte_count"] != len(payload) or record["payload_sha256"] != sha256_bytes(payload)):
        raise StageAError("RAW_LOCK_PAYLOAD_DIGEST_MISMATCH", source_slot=manifest_item["source_slot"])
    return dict(record)


def validate_raw_lock_set(records: Sequence[object], manifest: Sequence[Mapping[str, str]] | None = None) -> list[dict[str, object]]:
    bound_manifest = list(manifest) if manifest is not None else expected_manifest()
    if len(bound_manifest) != SOURCE_SLOT_COUNT or len(records) != SOURCE_SLOT_COUNT:
        raise StageAError("RAW_LOCK_SET_COUNT_INVALID")
    validated: list[dict[str, object]] = []
    for record, manifest_item in zip(records, bound_manifest):
        validated.append(validate_raw_lock_record(record, manifest_item))
    if [record["source_slot"] for record in validated] != [item["source_slot"] for item in bound_manifest]:
        raise StageAError("RAW_LOCK_SET_ORDER_INVALID")
    if len({record["source_slot"] for record in validated}) != SOURCE_SLOT_COUNT:
        raise StageAError("RAW_LOCK_SET_DUPLICATE_INVALID")
    return validated


def raw_lock_set_sha256(records: Sequence[object], manifest: Sequence[Mapping[str, str]] | None = None) -> str:
    validated = validate_raw_lock_set(records, manifest)
    return sha256_bytes(canonical_json_bytes(validated))


def _exclusive_write(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            import os
            os.fsync(handle.fileno())
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except Exception as exc:
            raise LockConflictError("DURABLE_STATE_READ_FAILURE") from exc
        if existing != content:
            raise LockConflictError("DURABLE_STATE_CONFLICT")


class RawLockStore:
    """External machine-local raw bytes and exact lock records.

    The store never overwrites an existing payload or lock.  An exact
    existing pair is reusable, which permits a crash-restart to skip already
    locked slots; every conflicting or incomplete pair fails closed.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.payload_dir = self.root / "raw_payloads"
        self.lock_dir = self.root / "raw_locks"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.payload_dir.mkdir(exist_ok=True)
            self.lock_dir.mkdir(exist_ok=True)
        except Exception as exc:
            raise StageAError("DURABLE_STATE_INITIALIZATION_FAILURE") from exc
        allowed = {"raw_payloads", "raw_locks", "safe_receipt.json"}
        try:
            extras = {child.name for child in self.root.iterdir() if child.name not in allowed}
        except Exception as exc:
            raise StageAError("DURABLE_STATE_ENUMERATION_FAILURE") from exc
        if extras:
            raise StageAError("DURABLE_STATE_UNEXPECTED_FILE")

    def _payload_path(self, slot: str) -> Path:
        return self.payload_dir / f"{slot}.bin"

    def _lock_path(self, slot: str) -> Path:
        return self.lock_dir / f"{slot}.json"

    def read_existing(self, manifest: Sequence[Mapping[str, str]]) -> dict[str, dict[str, object]]:
        expected_slots = {item["source_slot"] for item in manifest}
        try:
            lock_files = list(self.lock_dir.iterdir())
            payload_files = list(self.payload_dir.iterdir())
        except Exception as exc:
            raise StageAError("DURABLE_STATE_ENUMERATION_FAILURE") from exc
        for path in lock_files:
            if not path.is_file() or path.suffix != ".json" or path.stem not in expected_slots:
                raise StageAError("DURABLE_STATE_EXTRA_LOCK")
        for path in payload_files:
            if not path.is_file() or path.suffix != ".bin" or path.stem not in expected_slots:
                raise StageAError("DURABLE_STATE_EXTRA_PAYLOAD")
        by_slot: dict[str, dict[str, object]] = {}
        by_manifest = {item["source_slot"]: item for item in manifest}
        for slot, item in by_manifest.items():
            lock_path = self._lock_path(slot)
            payload_path = self._payload_path(slot)
            if lock_path.exists() != payload_path.exists():
                raise LockConflictError("DURABLE_STATE_INCOMPLETE_PAIR", source_slot=slot)
            if not lock_path.exists():
                continue
            try:
                record = json.loads(lock_path.read_text(encoding="utf-8"))
                payload = payload_path.read_bytes()
            except Exception as exc:
                raise LockConflictError("DURABLE_STATE_LOCK_READ_FAILURE", source_slot=slot) from exc
            by_slot[slot] = validate_raw_lock_record(record, item, payload)
        return by_slot

    def read_one(self, manifest_item: Mapping[str, str]) -> dict[str, object] | None:
        """Read one exact durable pair without requesting any other slot."""
        slot = manifest_item["source_slot"]
        lock_path = self._lock_path(slot)
        payload_path = self._payload_path(slot)
        if lock_path.exists() != payload_path.exists():
            raise LockConflictError("DURABLE_STATE_INCOMPLETE_PAIR", source_slot=slot)
        if not lock_path.exists():
            return None
        try:
            record = json.loads(lock_path.read_text(encoding="utf-8"))
            payload = payload_path.read_bytes()
        except Exception as exc:
            raise LockConflictError("DURABLE_STATE_LOCK_READ_FAILURE", source_slot=slot) from exc
        return validate_raw_lock_record(record, manifest_item, payload)

    def lock(self, manifest_item: Mapping[str, str], response: FetchResult) -> dict[str, object]:
        slot = manifest_item["source_slot"]
        if type(response.payload) is not bytes or type(response.http_status) is not int or type(response.resolved_url) is not str:
            raise StageAError("TRANSPORT_RESPONSE_TYPE_INVALID", source_slot=slot)
        if response.http_status != 200 or response.resolved_url != manifest_item["source_url"] or not response.payload:
            raise StageAError("TRANSPORT_RESPONSE_NOT_LOCKABLE", source_slot=slot)
        record: dict[str, object] = {
            "source_slot": slot,
            "source_url_sha256": manifest_item["source_url_sha256"],
            "http_status": 200,
            "byte_count": len(response.payload),
            "payload_sha256": sha256_bytes(response.payload),
        }
        validate_raw_lock_record(record, manifest_item, response.payload)
        _exclusive_write(self._payload_path(slot), response.payload)
        _exclusive_write(self._lock_path(slot), canonical_json_bytes(record))
        return record

    def write_safe_receipt(self, receipt: Mapping[str, object]) -> None:
        if set(receipt) != SAFE_RECEIPT_KEYS:
            raise StageAError("SAFE_RECEIPT_SCHEMA_INVALID")
        _exclusive_write(self.root / "safe_receipt.json", canonical_json_bytes(dict(receipt)))

    def read_safe_receipt(self) -> dict[str, object] | None:
        path = self.root / "safe_receipt.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise StageAError("SAFE_RECEIPT_READ_FAILURE") from exc
        if type(value) is not dict or set(value) != SAFE_RECEIPT_KEYS:
            raise StageAError("SAFE_RECEIPT_SCHEMA_INVALID")
        if value["schema_version"] != "V9_010_STAGE_A_SAFE_RECEIPT_V1" or value["task"] != TASK or value["status"] != "COMPLETE":
            raise StageAError("SAFE_RECEIPT_BINDING_INVALID")
        if value["source_manifest_sha256"] != MANIFEST_SHA256 or not _is_sha(value["raw_lock_set_sha256"], HEX64):
            raise StageAError("SAFE_RECEIPT_DIGEST_INVALID")
        if type(value["raw_lock_count"]) is not int or value["raw_lock_count"] != SOURCE_SLOT_COUNT or type(value["network_request_count"]) is not int or value["network_request_count"] < 0:
            raise StageAError("SAFE_RECEIPT_COUNT_INVALID")
        try:
            if path.read_bytes() != canonical_json_bytes(value):
                raise StageAError("SAFE_RECEIPT_CANONICAL_INVALID")
        except StageAError:
            raise
        except Exception as exc:
            raise StageAError("SAFE_RECEIPT_READ_FAILURE") from exc
        return dict(value)


def _transport_failure_from_exception(exc: BaseException, slot: str, attempts: int) -> StageAError:
    audit = getattr(exc, "transport_audit", ())
    history = list(audit) if isinstance(audit, list) else []
    label = history[-1].get("classification") if history and isinstance(history[-1], dict) else None
    if label in FROZEN_RETRYABLE_CLASSES and attempts == MAX_PRE_COMPLETE_ATTEMPTS:
        reason = "PLUMBING_FAILURE_RETRIABLE"
    elif isinstance(label, str) and label.startswith("HTTP_"):
        reason = label
    elif label in {"UNTRUSTED_REDIRECT", "RESPONSE_HOST_MISMATCH"}:
        reason = label
    elif label == "DATA_QUALITY_GATE_FAILURE":
        reason = "DATA_QUALITY_FAILURE"
    else:
        reason = "IMPLEMENTATION_FAILURE"
    return StageAError(reason, source_slot=slot, attempts=attempts, requests=attempts)


def acquire_one(
    manifest_item: Mapping[str, str],
    *,
    store: RawLockStore,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> tuple[dict[str, object], int]:
    """Acquire exactly one not-yet-locked manifest object."""
    slot = manifest_item["source_slot"]
    existing = store.read_one(manifest_item)
    if existing is not None:
        return existing, 0
    requests = 0

    def attempt() -> FetchResult:
        nonlocal requests
        requests += 1
        result = fetcher(manifest_item["source_url"])
        if type(result) is not FetchResult:
            raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
        if type(result.http_status) is not int or type(result.resolved_url) is not str or type(result.payload) is not bytes:
            raise V8CTransportNamedFailure("UNKNOWN_FAIL_CLOSED_NONRETRYABLE")
        if result.http_status != 200:
            if result.http_status in RETRYABLE_HTTP_CODES or result.http_status in {400, 401, 403, 404, 410, 422}:
                raise urllib.error.HTTPError(
                    manifest_item["source_url"], result.http_status, "", {}, None,
                )
            if 300 <= result.http_status < 400:
                raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")
            raise V8CTransportNamedFailure("DATA_QUALITY_GATE_FAILURE")
        if result.resolved_url != manifest_item["source_url"]:
            raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
        if not result.payload:
            raise V8CTransportNamedFailure("DATA_QUALITY_GATE_FAILURE")
        return result

    try:
        result, audit = attempt_with_frozen_retry(attempt, sleep_fn=sleep, request_fingerprint=manifest_item["source_url_sha256"])
    except (V8CTransportNamedFailure, V8CTransportBlocked) as exc:
        raise _transport_failure_from_exception(exc, slot, requests) from exc
    except BaseException as exc:  # the shared classifier intentionally handles concrete transport BaseExceptions
        raise _transport_failure_from_exception(exc, slot, requests) from exc
    try:
        return store.lock(manifest_item, result), requests
    except StageAError as exc:
        exc.network_request_count = requests
        raise


def build_safe_receipt(raw_lock_records: Sequence[object], network_request_count: int) -> dict[str, object]:
    validated = validate_raw_lock_set(raw_lock_records)
    if type(network_request_count) is not int or network_request_count < 0:
        raise StageAError("NETWORK_REQUEST_COUNT_INVALID")
    return {
        "schema_version": "V9_010_STAGE_A_SAFE_RECEIPT_V1",
        "task": TASK,
        "status": "COMPLETE",
        "source_manifest_sha256": MANIFEST_SHA256,
        "raw_lock_set_sha256": sha256_bytes(canonical_json_bytes(validated)),
        "raw_lock_count": len(validated),
        "network_request_count": network_request_count,
    }


def acquire_stage_a(
    output_root: str | Path,
    *,
    fetcher: Callable[[str], FetchResult],
    sleep: Callable[[float], None] = lambda _seconds: None,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Offline-testable Stage-A orchestration; never chooses a source."""
    path = Path(manifest_path) if manifest_path is not None else Path(__file__).resolve().parents[1] / MANIFEST_FILENAME
    manifest, _ = load_manifest(path)
    store = RawLockStore(output_root)
    existing = store.read_existing(manifest)
    receipt = store.read_safe_receipt()
    if receipt is not None:
        records = [existing[item["source_slot"]] for item in manifest]
        if len(existing) != SOURCE_SLOT_COUNT or raw_lock_set_sha256(records, manifest) != receipt["raw_lock_set_sha256"]:
            raise StageAError("SAFE_RECEIPT_LOCK_SET_MISMATCH")
        return receipt
    total_requests = 0
    records_by_slot = dict(existing)
    for item in manifest:
        slot = item["source_slot"]
        if slot in records_by_slot:
            continue
        try:
            record, requests = acquire_one(item, store=store, fetcher=fetcher, sleep=sleep)
        except StageAError as exc:
            exc.network_request_count += total_requests
            raise
        total_requests += requests
        records_by_slot[slot] = record
    records = [records_by_slot[item["source_slot"]] for item in manifest]
    validated = validate_raw_lock_set(records, manifest)
    if len(validated) != SOURCE_SLOT_COUNT:
        raise StageAError("RAW_LOCK_SET_COUNT_INVALID", requests=total_requests)
    safe = build_safe_receipt(validated, total_requests)
    store.write_safe_receipt(safe)
    return safe


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def _reject(self, *_args: object) -> None:
        raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


def fetch_http_once(url: str) -> FetchResult:
    """The only production HTTP request function; redirects are disabled."""
    verify_protected_environment(Path(__file__).resolve().parents[1])
    validate_manifest_url(url)
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=30.0) as response:
        status = response.getcode()
        resolved = response.geturl()
        payload = response.read()
    return FetchResult(payload=payload, http_status=status, resolved_url=resolved)


def _default_git_runner(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _check_sha(value: object, reason: str) -> str:
    if not _is_sha(value, HEX40):
        raise StageAError(reason)
    return value


def _load_real_execution_checker() -> Any:
    """Load the repository's existing no-network readiness checker."""
    from scripts import check_real_execution_env

    return check_real_execution_env


def _run_real_readiness_checks() -> dict[str, Any]:
    """Run the actual checker in the current interpreter."""
    return _load_real_execution_checker().run_readiness_checks()


def verify_protected_environment(repo_root: str | Path) -> dict[str, object]:
    """Require the reviewed canonical protected environment before JPX I/O.

    The checker result is accepted only when its exact public readiness
    contract passes.  No caller-supplied checker, result, interpreter, or
    bypass flag is exposed by the production API; tests replace the private
    no-network runner only to exercise failure/validity branches offline.
    """
    root = Path(repo_root).resolve()
    expected_executable = (root / ".venv-real-execution" / "Scripts" / "python.exe").resolve()
    try:
        actual_executable = Path(sys.executable).resolve()
    except Exception as exc:
        raise StageAError("PROTECTED_ENVIRONMENT_INTERPRETER_UNRESOLVABLE") from exc
    if os.name != "nt" or sys.platform != "win32":
        raise StageAError("PROTECTED_ENVIRONMENT_WRONG_PLATFORM")
    if actual_executable != expected_executable:
        raise StageAError("PROTECTED_ENVIRONMENT_WRONG_INTERPRETER")
    try:
        checker = _load_real_execution_checker()
        if checker.REPO_ROOT.resolve() != root or checker.CANONICAL_WINDOWS_INTERPRETER.resolve() != expected_executable:
            raise StageAError("PROTECTED_ENVIRONMENT_CHECKER_BINDING_MISMATCH")
        result = _run_real_readiness_checks()
    except StageAError:
        raise
    except Exception as exc:
        raise StageAError("PROTECTED_ENVIRONMENT_CHECKER_FAILURE") from exc
    if type(result) is not dict:
        raise StageAError("PROTECTED_ENVIRONMENT_CHECKER_OUTPUT_MALFORMED")
    true_fields = (
        "REAL_EXECUTION_ENVIRONMENT_READY",
        "REAL_EXECUTION_ENVIRONMENT_FROZEN",
        "INTERPRETER_MATCH",
        "PYTHON_PATCH_MATCH",
    )
    pass_fields = (
        "DEPENDENCY_READINESS",
        "JPX_XLS_PARSER_SYNTHETIC_PROBE",
        "TLS_STDLIB_PROBE",
        "TRUSTED_HOST_REQUEST_CONSTRUCTION_PROBE",
        "FILESYSTEM_PROBE",
        "ENVIRONMENT_LOCK_CHECK",
        "ENVIRONMENT_FREEZE_CHECK",
    )
    try:
        if any(result[field] is not True for field in true_fields):
            raise StageAError("PROTECTED_ENVIRONMENT_READINESS_NOT_PASS")
        if any(type(result[field]) is not str or result[field] != "PASS" for field in pass_fields):
            raise StageAError("PROTECTED_ENVIRONMENT_READINESS_NOT_PASS")
        if any(type(result[field]) is not int or result[field] != 0 for field in ("REAL_NETWORK_REQUESTS", "PRIVATE_READS", "GATES_CONSUMED")):
            raise StageAError("PROTECTED_ENVIRONMENT_SAFETY_COUNTER_NONZERO")
    except KeyError as exc:
        raise StageAError("PROTECTED_ENVIRONMENT_CHECKER_OUTPUT_MALFORMED") from exc
    return {
        "protected_environment_verified": True,
        "canonical_interpreter": str(expected_executable),
        "readiness_contract": "PASS",
    }


def verify_production_preflight(
    repo_root: str | Path,
    *,
    expected_implementation_sha: str,
    confirmation: str,
    git_runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, object]:
    """Fail-closed checks required immediately before any future request."""
    implementation_sha = _check_sha(expected_implementation_sha, "IMPLEMENTATION_SHA_INVALID")
    if confirmation != HUMAN_CONFIRMATION:
        raise StageAError("FRESH_HUMAN_CONFIRMATION_INVALID")
    root = Path(repo_root)
    runner = git_runner or (lambda args: _default_git_runner(root, args))
    try:
        branch = runner(["branch", "--show-current"])
        local = runner(["rev-parse", "HEAD"])
        remote = runner(["rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"])
        status = runner(["status", "--porcelain=v1", "--untracked-files=all"])
        design_commit = runner(["rev-parse", f"{REVIEWED_DESIGN_GIT_SHA}^{{commit}}"])
        reviewed_design_blob = runner(["rev-parse", f"{REVIEWED_DESIGN_GIT_SHA}:{DESIGN_PATH}"])
        current_design_blob = runner(["rev-parse", f"HEAD:{DESIGN_PATH}"])
    except Exception as exc:
        raise StageAError("GIT_PROVENANCE_UNAVAILABLE") from exc
    if branch != AUTHORITATIVE_BRANCH or local != implementation_sha or remote != implementation_sha:
        raise StageAError("GIT_PROVENANCE_MISMATCH")
    if status != "":
        raise StageAError("GIT_WORKTREE_DIRTY")
    if design_commit != REVIEWED_DESIGN_GIT_SHA or reviewed_design_blob != REVIEWED_DESIGN_BLOB_GIT_SHA or current_design_blob != REVIEWED_DESIGN_BLOB_GIT_SHA:
        raise StageAError("DESIGN_BINDING_MISMATCH")
    manifest, digest = load_manifest(root / MANIFEST_FILENAME)
    if digest != MANIFEST_SHA256 or len(manifest) != SOURCE_SLOT_COUNT:
        raise StageAError("MANIFEST_BINDING_MISMATCH")
    environment = verify_protected_environment(root)
    return {
        "authoritative_branch": branch,
        "implementation_git_sha": implementation_sha,
        "local_head": local,
        "remote_tracking_head": remote,
        "reviewed_design_git_sha": REVIEWED_DESIGN_GIT_SHA,
        "source_manifest_sha256": digest,
        "source_slot_count": len(manifest),
        "human_confirmation_verified": True,
        **environment,
    }


def run_production_acquisition(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    expected_implementation_sha: str,
    confirmation: str,
    sleep: Callable[[float], None] = lambda seconds: __import__("time").sleep(seconds),
) -> dict[str, object]:
    """Future-only production seam.  This task never calls this function."""
    repo_path = Path(repo_root).resolve()
    output_path = Path(output_root).resolve()
    if output_path == repo_path or repo_path in output_path.parents:
        raise StageAError("RAW_STATE_MUST_BE_EXTERNAL")
    verify_production_preflight(
        repo_path,
        expected_implementation_sha=expected_implementation_sha,
        confirmation=confirmation,
    )
    return acquire_stage_a(output_path, fetcher=fetch_http_once, sleep=sleep, manifest_path=repo_path / MANIFEST_FILENAME)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V9_010 Stage-A JPX raw acquisition seam")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--expected-implementation-sha", required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_production_acquisition(
            args.output_root,
            repo_root=args.repo_root,
            expected_implementation_sha=args.expected_implementation_sha,
            confirmation=args.confirmation,
        )
    except StageAError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.reason}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except Exception:
        print("V9_010_STAGE_A_IMPLEMENTATION_FAILURE", file=sys.stderr)
        return 3
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


__all__ = [
    "AUTHORITATIVE_BRANCH",
    "BACKOFF_SECONDS",
    "DESIGN_PATH",
    "FALLBACK_SOURCE_OBJECTS",
    "FetchResult",
    "FROZEN_BACKOFF_SECONDS",
    "FROZEN_RETRYABLE_CLASSES",
    "HUMAN_CONFIRMATION",
    "LOCK_SCHEMA_KEYS",
    "MANIFEST_FILENAME",
    "MANIFEST_SHA256",
    "MAX_PRE_COMPLETE_ATTEMPTS",
    "MAX_PRE_COMPLETE_RETRIES",
    "RawLockStore",
    "REVIEWED_DESIGN_BLOB_GIT_SHA",
    "REVIEWED_DESIGN_GIT_SHA",
    "SOURCE_HOST",
    "SOURCE_SCHEME",
    "SOURCE_SLOT_COUNT",
    "SOURCE_URL_TEMPLATE",
    "StageAError",
    "acquire_one",
    "acquire_stage_a",
    "build_safe_receipt",
    "canonical_json_bytes",
    "expected_manifest",
    "fetch_http_once",
    "load_manifest",
    "main",
    "raw_lock_set_sha256",
    "run_production_acquisition",
    "sha256_bytes",
    "validate_manifest",
    "validate_manifest_url",
    "validate_raw_lock_record",
    "validate_raw_lock_set",
    "verify_protected_environment",
    "verify_production_preflight",
]
