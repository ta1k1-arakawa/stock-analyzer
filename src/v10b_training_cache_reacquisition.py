"""V10B fixed-source training-cache acquisition and manifest validation.

The production entrypoint binds every source and retry choice to constants in
this module.  The narrow transport and semantic-validator arguments on the
internal acquisition function exist only so synthetic tests can exercise the
same manifest closure without making HTTP requests.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


STUDY_IDENTITY = "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR"
MANIFEST_SCHEMA = "V10B_TRAINING_CACHE_MANIFEST_V1"
ATTEMPT_RECEIPT_SCHEMA = "V10B_ACQUISITION_ATTEMPT_RECEIPT_V1"
EXPECTED_REPOSITORY_URL = "https://github.com/ta1k1-arakawa/stock-analyzer.git"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
FROZEN_DESIGN_COMMIT = "1260538c5bef899478806f74ae32d9e4be7b023b"
FROZEN_DESIGN_BLOB = "bc57720b6e73cc8c4cf793258a98f78a635af783"
FREEZE_APPROVAL_BLOB = "dde0418589088932559b289b3ed26a40e29b62b6"
FREEZE_APPROVAL_FILE = "V10B_DESIGN_FREEZE_APPROVAL.json"
DESIGN_FILE = "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR_DESIGN_DRAFT.md"
UNIVERSE_FILE = "V4_UNIVERSE.csv"
UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
TICKER_COUNT = 300
PRICE_FROM = "2015-01-01"
PRICE_TO = "2019-12-31"
YAHOO_SCHEME = "https"
YAHOO_HOST = "query1.finance.yahoo.com"
YAHOO_PATH_PREFIX = "/v8/finance/chart/"
QUERY_SPECIFICATION = (
    ("period1", "1420070400"),
    ("period2", "1577836800"),
    ("interval", "1d"),
    ("events", "div,splits"),
    ("includeAdjustedClose", "true"),
)
LOCKED_RAW_DIRECTORY = "locked_raw"
ATTEMPT_RECEIPT_FILE = "V10B_ACQUISITION_ATTEMPT_RECEIPT.json"
MANIFEST_FILE = "cache_manifest.json"
MAX_TRANSPORT_ATTEMPTS = 3
ALLOWED_ERROR_TYPES = frozenset(
    {None, "TRANSPORT_EXCEPTION", "HTTP_ERROR", "REDIRECT", "EMPTY_BODY", "PARSER_FAILURE"}
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "complete",
        "universe_mode",
        "universe_csv_sha256",
        "ticker_list_sha256",
        "ticker_count",
        "ticker_order",
        "price_from",
        "price_to",
        "query_specification",
        "payloads",
        "network_audit",
        "successful_ticker_count",
        "failed_tickers",
        "payload_hash_list_sha256",
    }
)
PAYLOAD_FIELDS = frozenset({"ticker", "relative_path", "sha256", "byte_count"})
AUDIT_FIELDS = frozenset(
    {
        "ticker",
        "attempt",
        "scheme",
        "host",
        "path",
        "query_specification",
        "status",
        "error_type",
        "redirect_detected",
        "body_byte_count",
        "payload_sha256",
        "retry",
        "final",
        "success",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class V10BError(RuntimeError):
    """Base class for safe, bounded V10B errors."""


class GovernanceFailure(V10BError):
    pass


class PostBoundaryFailure(V10BError):
    """Operational failure after at least one transport invocation."""


class ManifestValidationError(V10BError):
    pass


class PayloadSemanticFailure(V10BError):
    pass


# These are the only parser-side exceptions that represent malformed payload
# content for the reviewed V4 Yahoo-chart parser.  In particular, generic
# RuntimeError/AssertionError/NameError exceptions remain implementation
# failures and must not become failed tickers.
PAYLOAD_CONTENT_EXCEPTIONS = (
    UnicodeDecodeError,
    json.JSONDecodeError,
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    AttributeError,
    OverflowError,
)


@dataclass(frozen=True)
class AttemptBinding:
    frozen_design_commit: str
    frozen_design_blob: str
    freeze_approval_blob: str
    implementation_sha: str


@dataclass
class _TransportBoundary:
    crossed: bool = False

    def invoke(
        self,
        transport: Callable[[str, int], tuple[Any, bytes, bool]],
        url: str,
        attempt: int,
    ) -> tuple[Any, bytes, bool]:
        # The boundary is crossed before calling transport so an exception
        # raised by that invocation is still post-boundary for classification.
        self.crossed = True
        return transport(url, attempt)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_sha1(value: Any) -> bool:
    return isinstance(value, str) and SHA1_RE.fullmatch(value) is not None


def _canonical_text_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _query_as_lists() -> list[list[str]]:
    return [[key, value] for key, value in QUERY_SPECIFICATION]


def _retry_allowed(status: Any, redirect_detected: bool, attempt: int) -> bool:
    """Apply the one frozen retry predicate used by acquisition and audit."""
    if redirect_detected or attempt >= MAX_TRANSPORT_ATTEMPTS:
        return False
    return status == "TRANSPORT_EXCEPTION" or (
        type(status) is int and (status == 429 or 500 <= status <= 599)
    )


def _require_fixed_ticker_order(ticker_order: Sequence[str]) -> list[str]:
    order = list(ticker_order)
    if len(order) != TICKER_COUNT or len(set(order)) != TICKER_COUNT:
        raise ManifestValidationError("FIXED_UNIVERSE_CARDINALITY_MISMATCH")
    if any(not isinstance(ticker, str) or not ticker.isalnum() or len(ticker) != 4 for ticker in order):
        raise ManifestValidationError("FIXED_TICKER_FORMAT_MISMATCH")
    return order


def yahoo_url(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.isalnum() or len(ticker) != 4:
        raise ManifestValidationError("FIXED_TICKER_FORMAT_MISMATCH")
    query = "&".join(f"{key}={value}" for key, value in QUERY_SPECIFICATION)
    return f"{YAHOO_SCHEME}://{YAHOO_HOST}{YAHOO_PATH_PREFIX}{ticker}.T?{query}"


def _safe_regular_file(path: Path) -> None:
    try:
        stat = path.lstat()
    except FileNotFoundError as exc:
        raise GovernanceFailure("REQUIRED_FILE_MISSING") from exc
    if not path.is_file() or path.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & 0x400):
        raise GovernanceFailure("REQUIRED_FILE_UNSAFE")


def _assert_external_path(path: Path, repo_root: Path) -> None:
    if not path.is_absolute():
        raise GovernanceFailure("ATTEMPT_ROOT_NOT_ABSOLUTE")
    repo = repo_root.resolve()
    candidate = path.resolve(strict=False)
    try:
        candidate.relative_to(repo)
    except ValueError:
        pass
    else:
        raise GovernanceFailure("ATTEMPT_ROOT_INSIDE_REPOSITORY")
    for ancestor in (path.parent, *path.parent.parents):
        if not ancestor.exists():
            continue
        try:
            stat = ancestor.lstat()
        except OSError as exc:
            raise GovernanceFailure("ATTEMPT_ROOT_ANCESTOR_UNAVAILABLE") from exc
        if not ancestor.is_dir() or ancestor.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & 0x400):
            raise GovernanceFailure("ATTEMPT_ROOT_ANCESTOR_UNSAFE")


def create_exclusive_attempt_root(attempt_root: Path, repo_root: Path) -> Path:
    _assert_external_path(attempt_root, repo_root)
    if attempt_root.exists() or attempt_root.is_symlink():
        raise GovernanceFailure("ATTEMPT_ROOT_ALREADY_EXISTS")
    try:
        attempt_root.mkdir()
    except FileExistsError as exc:
        raise GovernanceFailure("ATTEMPT_ROOT_ALREADY_EXISTS") from exc
    except OSError as exc:
        raise GovernanceFailure("ATTEMPT_ROOT_CREATE_FAILED") from exc
    return attempt_root


def _assert_child_path(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestValidationError("PAYLOAD_PATH_ESCAPE") from exc
    if candidate.is_symlink():
        raise ManifestValidationError("PAYLOAD_PATH_SYMLINK")
    return candidate


def _write_exclusive(
    path: Path,
    body: bytes,
    *,
    failure_cls: type[V10BError] = GovernanceFailure,
) -> None:
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise failure_cls("DURABLE_OVERWRITE_PROHIBITED") from exc
    except OSError as exc:
        raise failure_cls("DURABLE_WRITE_FAILED") from exc


def _write_post_boundary(path: Path, body: bytes) -> None:
    try:
        _write_exclusive(path, body, failure_cls=PostBoundaryFailure)
    except PostBoundaryFailure:
        raise
    except GovernanceFailure as exc:
        # A lower-level helper must not be able to turn a post-transport
        # durable failure back into a preflight classification.
        raise PostBoundaryFailure("POST_BOUNDARY_DURABLE_WRITE_FAILURE") from exc


def _write_attempt_receipt(attempt_root: Path, binding: AttemptBinding) -> None:
    if not all(
        (
            _is_sha1(binding.frozen_design_commit),
            _is_sha1(binding.frozen_design_blob),
            _is_sha1(binding.freeze_approval_blob),
            _is_sha1(binding.implementation_sha),
        )
    ):
        raise GovernanceFailure("ATTEMPT_BINDING_INVALID")
    receipt = {
        "schema_version": ATTEMPT_RECEIPT_SCHEMA,
        "study_identity": STUDY_IDENTITY,
        "frozen_design_git_commit": binding.frozen_design_commit,
        "frozen_design_git_blob_sha": binding.frozen_design_blob,
        "freeze_approval_git_blob_sha": binding.freeze_approval_blob,
        "implementation_sha": binding.implementation_sha,
        "attempt_started": True,
        "network_acquisition_scope": "V10B_PUBLIC_YAHOO_ONLY",
    }
    _write_exclusive(attempt_root / ATTEMPT_RECEIPT_FILE, canonical_json_bytes(receipt))


def _payload_hash_list_sha256(payloads: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(payloads)))


def _resolve_inherited_parser() -> Callable[[Mapping[str, Any]], Any]:
    """Resolve the reviewed parser before production transport begins."""
    try:
        from src.v4_meta_label_formal import parse_v4_yahoo_chart
    except Exception as exc:
        raise GovernanceFailure("INHERITED_PARSER_UNAVAILABLE") from exc
    if not callable(parse_v4_yahoo_chart):
        raise GovernanceFailure("INHERITED_PARSER_UNCALLABLE")
    return parse_v4_yahoo_chart


def _semantic_validator_for_parser(
    parser: Callable[[Mapping[str, Any]], Any],
) -> Callable[[bytes], None]:
    """Build a validator with a closed semantic/data-error boundary."""
    if not callable(parser):
        raise GovernanceFailure("INHERITED_PARSER_UNCALLABLE")

    def validate(body: bytes) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PayloadSemanticFailure("PAYLOAD_SEMANTIC_FAILURE") from exc
        try:
            parser(payload)
        except PAYLOAD_CONTENT_EXCEPTIONS as exc:
            raise PayloadSemanticFailure("PAYLOAD_SEMANTIC_FAILURE") from exc

    return validate


def _default_semantic_validator(body: bytes) -> None:
    """Compatibility helper; production resolves the parser before transport."""
    _semantic_validator_for_parser(_resolve_inherited_parser())(body)


def _production_yahoo_transport(url: str, attempt: int) -> tuple[Any, bytes, bool]:
    import requests

    response = requests.get(
        url,
        timeout=45,
        allow_redirects=False,
        headers={"User-Agent": "stock-analyzer-v10b-training-cache/1.0"},
    )
    body = response.content
    redirect = bool(300 <= response.status_code < 400 or response.headers.get("Location"))
    return int(response.status_code), body, redirect


def _audit_entry(
    ticker: str,
    attempt: int,
    status: Any,
    body: bytes,
    redirect: bool,
    error_type: str | None,
    retry: bool,
    final: bool,
    success: bool,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "attempt": attempt,
        "scheme": YAHOO_SCHEME,
        "host": YAHOO_HOST,
        "path": f"{YAHOO_PATH_PREFIX}{ticker}.T",
        "query_specification": _query_as_lists(),
        "status": status,
        "error_type": error_type,
        "redirect_detected": bool(redirect),
        "body_byte_count": len(body) if isinstance(body, bytes) else 0,
        "payload_sha256": sha256_bytes(body) if isinstance(body, bytes) and body else None,
        "retry": retry,
        "final": final,
        "success": success,
    }


def _validate_audit_records(audit: Any, ticker_order: Sequence[str]) -> None:
    if not isinstance(audit, list) or not audit:
        raise ManifestValidationError("NETWORK_AUDIT_MISSING")
    by_ticker: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in ticker_order}
    order: list[tuple[int, int]] = []
    for item in audit:
        if not isinstance(item, dict) or set(item) != AUDIT_FIELDS:
            raise ManifestValidationError("NETWORK_AUDIT_FIELDSET_INVALID")
        ticker = item["ticker"]
        attempt = item["attempt"]
        if ticker not in by_ticker or type(attempt) is not int or not 1 <= attempt <= MAX_TRANSPORT_ATTEMPTS:
            raise ManifestValidationError("NETWORK_AUDIT_BINDING_INVALID")
        if (
            item["scheme"] != YAHOO_SCHEME
            or item["host"] != YAHOO_HOST
            or item["path"] != f"{YAHOO_PATH_PREFIX}{ticker}.T"
            or item["query_specification"] != _query_as_lists()
        ):
            raise ManifestValidationError("NETWORK_AUDIT_SOURCE_INVALID")
        if item["error_type"] not in ALLOWED_ERROR_TYPES:
            raise ManifestValidationError("NETWORK_AUDIT_ERROR_TYPE_INVALID")
        if any(type(item[key]) is not bool for key in ("retry", "final", "success", "redirect_detected")):
            raise ManifestValidationError("NETWORK_AUDIT_BOOLEAN_INVALID")
        status = item["status"]
        retryable = _retry_allowed(status, item["redirect_detected"], attempt)
        if item["retry"] != retryable:
            raise ManifestValidationError("NETWORK_AUDIT_RETRY_INVALID")
        if item["success"]:
            if (
                status != 200
                or item["redirect_detected"]
                or item["error_type"] is not None
                or type(item["body_byte_count"]) is not int
                or item["body_byte_count"] <= 0
                or not _is_sha256(item["payload_sha256"])
                or item["retry"]
                or not item["final"]
            ):
                raise ManifestValidationError("NETWORK_AUDIT_SUCCESS_INVALID")
        else:
            if item["redirect_detected"]:
                if (
                    item["retry"]
                    or not item["final"]
                    or item["success"]
                    or item["error_type"] != "REDIRECT"
                ):
                    raise ManifestValidationError("NETWORK_AUDIT_REDIRECT_INVALID")
            elif item["error_type"] == "REDIRECT":
                raise ManifestValidationError("NETWORK_AUDIT_REDIRECT_INVALID")
            if item["final"] is False and not item["retry"]:
                raise ManifestValidationError("NETWORK_AUDIT_NONFINAL_INVALID")
            if item["final"] and item["retry"]:
                raise ManifestValidationError("NETWORK_AUDIT_FINAL_RETRY_INVALID")
            if item["error_type"] is None:
                raise ManifestValidationError("NETWORK_AUDIT_FAILURE_TYPE_MISSING")
            if item["payload_sha256"] is not None and not _is_sha256(item["payload_sha256"]):
                raise ManifestValidationError("NETWORK_AUDIT_PAYLOAD_HASH_INVALID")
        by_ticker[ticker].append(item)
        order.append((ticker_order.index(ticker), attempt))
    if order != sorted(order):
        raise ManifestValidationError("NETWORK_AUDIT_ORDER_INVALID")
    for ticker, records in by_ticker.items():
        if not records:
            raise ManifestValidationError("NETWORK_AUDIT_TICKER_MISSING")
        attempts = [record["attempt"] for record in records]
        if attempts != list(range(1, len(records) + 1)):
            raise ManifestValidationError("NETWORK_AUDIT_ATTEMPT_SEQUENCE_INVALID")
        if sum(record["final"] for record in records) != 1 or records[-1]["final"] is not True:
            raise ManifestValidationError("NETWORK_AUDIT_TERMINAL_STATE_INVALID")
        if any(record["final"] for record in records[:-1]):
            raise ManifestValidationError("NETWORK_AUDIT_EARLY_FINAL_INVALID")
        successes = [record for record in records if record["success"]]
        if len(successes) > 1 or (successes and successes[0] is not records[-1]):
            raise ManifestValidationError("NETWORK_AUDIT_SUCCESS_SEQUENCE_INVALID")


def validate_manifest(
    manifest: Mapping[str, Any],
    attempt_root: Path,
    ticker_order: Sequence[str],
) -> dict[str, Any]:
    order = _require_fixed_ticker_order(ticker_order)
    if not isinstance(manifest, Mapping) or set(manifest) != MANIFEST_FIELDS:
        raise ManifestValidationError("MANIFEST_FIELDSET_INVALID")
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["complete"] is not True:
        raise ManifestValidationError("MANIFEST_SCHEMA_OR_COMPLETION_INVALID")
    if (
        manifest["universe_mode"] != "FIXED_V4_300"
        or manifest["universe_csv_sha256"] != UNIVERSE_CSV_SHA256
        or manifest["ticker_list_sha256"] != TICKER_LIST_SHA256
        or type(manifest["ticker_count"]) is not int
        or manifest["ticker_count"] != TICKER_COUNT
        or manifest["ticker_order"] != order
        or manifest["price_from"] != PRICE_FROM
        or manifest["price_to"] != PRICE_TO
        or manifest["query_specification"] != _query_as_lists()
    ):
        raise ManifestValidationError("MANIFEST_FIXED_BINDING_INVALID")
    payloads = manifest["payloads"]
    if not isinstance(payloads, list):
        raise ManifestValidationError("PAYLOAD_LIST_INVALID")
    accepted: list[str] = []
    for item in payloads:
        if not isinstance(item, dict) or set(item) != PAYLOAD_FIELDS:
            raise ManifestValidationError("PAYLOAD_FIELDSET_INVALID")
        ticker = item["ticker"]
        if ticker not in order or ticker in accepted or item["relative_path"] != f"{LOCKED_RAW_DIRECTORY}/{ticker}.json":
            raise ManifestValidationError("PAYLOAD_BINDING_INVALID")
        if not _is_sha256(item["sha256"]) or type(item["byte_count"]) is not int or item["byte_count"] <= 0:
            raise ManifestValidationError("PAYLOAD_METADATA_INVALID")
        path = _assert_child_path(attempt_root, item["relative_path"])
        _safe_regular_file(path)
        body = path.read_bytes()
        if len(body) != item["byte_count"] or sha256_bytes(body) != item["sha256"]:
            raise ManifestValidationError("PAYLOAD_HASH_MISMATCH")
        accepted.append(ticker)
    if accepted != [ticker for ticker in order if ticker in set(accepted)]:
        raise ManifestValidationError("PAYLOAD_ORDER_INVALID")
    if type(manifest["successful_ticker_count"]) is not int or manifest["successful_ticker_count"] != len(payloads):
        raise ManifestValidationError("SUCCESS_COUNT_INVALID")
    if manifest["payload_hash_list_sha256"] != _payload_hash_list_sha256(payloads):
        raise ManifestValidationError("PAYLOAD_HASH_LIST_MISMATCH")
    failed = manifest["failed_tickers"]
    if not isinstance(failed, list) or failed != [ticker for ticker in order if ticker not in set(accepted)]:
        raise ManifestValidationError("FAILED_TICKER_COMPLEMENT_INVALID")
    if set(accepted) & set(failed) or set(accepted) | set(failed) != set(order):
        raise ManifestValidationError("SUCCESS_FAILURE_PARTITION_INVALID")
    if manifest["successful_ticker_count"] + len(failed) != TICKER_COUNT:
        raise ManifestValidationError("SUCCESS_FAILURE_CARDINALITY_INVALID")
    audit = manifest["network_audit"]
    _validate_audit_records(audit, order)
    for ticker in order:
        records = [record for record in audit if record["ticker"] == ticker]
        success_records = [record for record in records if record["success"]]
        if ticker in accepted:
            if len(success_records) != 1:
                raise ManifestValidationError("PAYLOAD_AUDIT_SUCCESS_MISSING")
            payload = next(item for item in payloads if item["ticker"] == ticker)
            if success_records[0]["payload_sha256"] != payload["sha256"] or success_records[0]["body_byte_count"] != payload["byte_count"]:
                raise ManifestValidationError("PAYLOAD_AUDIT_HASH_MISMATCH")
        elif success_records:
            raise ManifestValidationError("FAILED_TICKER_AUDIT_SUCCESS_INVALID")
    return dict(manifest)


def _run_acquisition_loop(
    attempt_root: Path,
    ticker_order: Sequence[str],
    transport: Callable[[str, int], tuple[Any, bytes, bool]],
    semantic_validator: Callable[[bytes], Any],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    order = _require_fixed_ticker_order(ticker_order)
    locked_raw = attempt_root / LOCKED_RAW_DIRECTORY
    try:
        locked_raw.mkdir()
    except OSError as exc:
        raise GovernanceFailure("LOCKED_RAW_DIRECTORY_CREATE_FAILED") from exc
    boundary = _TransportBoundary()
    payloads: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    failed: list[str] = []
    for ticker in order:
        for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
            try:
                status, body, redirect = boundary.invoke(transport, yahoo_url(ticker), attempt)
            except Exception:
                status, body, redirect = "TRANSPORT_EXCEPTION", b"", False
            if status == 200 and isinstance(body, bytes) and body and not redirect:
                digest = sha256_bytes(body)
                target = locked_raw / f"{ticker}.json"
                _write_post_boundary(target, body)
                try:
                    accepted = semantic_validator(body)
                    if accepted is False:
                        raise PayloadSemanticFailure("PAYLOAD_SEMANTIC_FAILURE")
                except PayloadSemanticFailure:
                    audit.append(_audit_entry(ticker, attempt, status, body, redirect, "PARSER_FAILURE", False, True, False))
                    failed.append(ticker)
                    break
                audit.append(_audit_entry(ticker, attempt, status, body, redirect, None, False, True, True))
                payloads.append(
                    {
                        "ticker": ticker,
                        "relative_path": f"{LOCKED_RAW_DIRECTORY}/{ticker}.json",
                        "sha256": digest,
                        "byte_count": len(body),
                    }
                )
                break
            retry = _retry_allowed(status, redirect, attempt)
            if redirect:
                error_type = "REDIRECT"
            elif status == 200:
                error_type = "EMPTY_BODY"
            elif status == "TRANSPORT_EXCEPTION":
                error_type = "TRANSPORT_EXCEPTION"
            else:
                error_type = "HTTP_ERROR"
            audit.append(_audit_entry(ticker, attempt, status, body if isinstance(body, bytes) else b"", redirect, error_type, retry, not retry, False))
            if retry:
                sleep(0)
                continue
            failed.append(ticker)
            break
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "complete": True,
        "universe_mode": "FIXED_V4_300",
        "universe_csv_sha256": UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": TICKER_LIST_SHA256,
        "ticker_count": TICKER_COUNT,
        "ticker_order": order,
        "price_from": PRICE_FROM,
        "price_to": PRICE_TO,
        "query_specification": _query_as_lists(),
        "payloads": payloads,
        "network_audit": audit,
        "successful_ticker_count": len(payloads),
        "failed_tickers": failed,
        "payload_hash_list_sha256": _payload_hash_list_sha256(payloads),
    }
    try:
        validate_manifest(manifest, attempt_root, order)
    except GovernanceFailure as exc:
        raise PostBoundaryFailure("POST_BOUNDARY_CLOSURE_FAILURE") from exc
    _write_post_boundary(
        attempt_root / MANIFEST_FILE,
        canonical_json_bytes(manifest),
    )
    return manifest


def acquire_cache(
    repo_root: Path,
    attempt_root: Path,
    ticker_order: Sequence[str],
    *,
    transport: Callable[[str, int], tuple[Any, bytes, bool]],
    semantic_validator: Callable[[bytes], Any] = _default_semantic_validator,
    sleep: Callable[[float], None] = time.sleep,
    binding: AttemptBinding,
) -> dict[str, Any]:
    """Create one exclusive synthetic/production attempt and acquire once.

    The fixed production runner supplies the fixed universe, transport, and
    parser. Synthetic tests may supply only a deterministic transport and
    semantic predicate; the production CLI exposes none of those overrides.
    """
    root = create_exclusive_attempt_root(attempt_root, repo_root)
    _write_attempt_receipt(root, binding)
    return _run_acquisition_loop(root, ticker_order, transport, semantic_validator, sleep)


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        raise GovernanceFailure("GIT_PROVENANCE_UNAVAILABLE") from exc
    return result.stdout.strip()


def _normalize_repository_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or parsed.netloc != "github.com" or parsed.path != "/ta1k1-arakawa/stock-analyzer.git":
        raise GovernanceFailure("REPOSITORY_IDENTITY_MISMATCH")
    return EXPECTED_REPOSITORY_URL


def _validate_freeze_approval(repo_root: Path) -> None:
    path = repo_root / FREEZE_APPROVAL_FILE
    _safe_regular_file(path)
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GovernanceFailure("FREEZE_APPROVAL_INVALID") from exc
    required = {
        "schema_version": "V10B_DESIGN_FREEZE_APPROVAL_V1",
        "study": STUDY_IDENTITY,
        "artifact_role": "DESIGN_FREEZE_APPROVAL",
        "frozen_design_git_commit": FROZEN_DESIGN_COMMIT,
        "frozen_design_git_blob_sha": FROZEN_DESIGN_BLOB,
        "design_document": DESIGN_FILE,
        "final_independent_review_result": "PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0",
        "final_independent_review_design_commit": FROZEN_DESIGN_COMMIT,
        "approval_status": "APPROVED",
        "human_design_freeze_complete": True,
        "approval_scope": "DESIGN_FREEZE_ONLY",
        "approval_artifact_authorizes_implementation_phase_only": True,
        "network_acquisition_authorized": False,
        "t0_authorized": False,
        "historical_evaluation_authorized": False,
        "private_sealed_access_authorized": False,
        "implementation_performed_by_this_artifact": False,
        "methodology_change_after_freeze_requires": "NEW_STUDY_REQUIRED",
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise GovernanceFailure("FREEZE_APPROVAL_BINDING_MISMATCH")


def load_fixed_universe(repo_root: Path) -> list[str]:
    path = repo_root / UNIVERSE_FILE
    _safe_regular_file(path)
    raw = path.read_bytes()
    if sha256_bytes(_canonical_text_bytes(raw)) != UNIVERSE_CSV_SHA256:
        raise GovernanceFailure("UNIVERSE_CSV_HASH_MISMATCH")
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        if not rows or set(rows[0]) != {"ticker", "market", "industry"}:
            raise ValueError
        tickers = [row["ticker"] for row in rows]
    except Exception as exc:
        raise GovernanceFailure("UNIVERSE_CSV_INVALID") from exc
    if len(tickers) != TICKER_COUNT or sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8")) != TICKER_LIST_SHA256:
        raise GovernanceFailure("TICKER_LIST_HASH_MISMATCH")
    return _require_fixed_ticker_order(tickers)


def validate_repository_preflight(repo_root: Path, implementation_sha: str) -> list[str]:
    if not _is_sha1(implementation_sha):
        raise GovernanceFailure("IMPLEMENTATION_SHA_INVALID")
    if _normalize_repository_url(_git(repo_root, "config", "--get", "remote.origin.url")) != EXPECTED_REPOSITORY_URL:
        raise GovernanceFailure("REPOSITORY_IDENTITY_MISMATCH")
    if _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != AUTHORITATIVE_BRANCH:
        raise GovernanceFailure("BRANCH_MISMATCH")
    if _git(repo_root, "rev-parse", "HEAD") != implementation_sha:
        raise GovernanceFailure("IMPLEMENTATION_HEAD_MISMATCH")
    if _git(repo_root, "rev-parse", f"origin/{AUTHORITATIVE_BRANCH}") != implementation_sha:
        raise GovernanceFailure("REMOTE_HEAD_MISMATCH")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise GovernanceFailure("WORKTREE_DIRTY")
    if _git(repo_root, "rev-parse", f"HEAD:{DESIGN_FILE}") != FROZEN_DESIGN_BLOB:
        raise GovernanceFailure("DESIGN_BLOB_MISMATCH")
    if _git(repo_root, "rev-parse", f"HEAD:{FREEZE_APPROVAL_FILE}") != FREEZE_APPROVAL_BLOB:
        raise GovernanceFailure("FREEZE_APPROVAL_BLOB_MISMATCH")
    _validate_freeze_approval(repo_root)
    return load_fixed_universe(repo_root)


def run_production(repo_root: Path, attempt_root: Path, implementation_sha: str) -> dict[str, Any]:
    ticker_order = validate_repository_preflight(repo_root, implementation_sha)
    parser = _resolve_inherited_parser()
    binding = AttemptBinding(FROZEN_DESIGN_COMMIT, FROZEN_DESIGN_BLOB, FREEZE_APPROVAL_BLOB, implementation_sha)
    return acquire_cache(
        repo_root,
        attempt_root,
        ticker_order,
        transport=_production_yahoo_transport,
        semantic_validator=_semantic_validator_for_parser(parser),
        binding=binding,
    )
