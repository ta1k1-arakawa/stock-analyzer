"""Offline-only V10C adoption validation for the locked V10B candidate.

The module deliberately has no network transport and never parses a locked
payload body.  Phase A validates repository and candidate metadata.  Only the
explicit Phase-B function, after a matching authorization marker, reads the
locked files for byte-count and SHA-256 closure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlparse


STUDY_IDENTITY = "V10C_LOCKED_TRAINING_CACHE_PROVENANCE_ADOPTION_SUCCESSOR"
FROZEN_V10C_DESIGN_COMMIT = "f88a748f6f5e3fa39bfe57f1720316415b961c0b"
FROZEN_V10C_DESIGN_BLOB = "d4e6e9b15dfee423aabc0a4052e68e4970739319"
DESIGN_FILE = "V10C_LOCKED_TRAINING_CACHE_PROVENANCE_ADOPTION_SUCCESSOR_DESIGN_DRAFT.md"
DESIGN_FREEZE_APPROVAL_FILE = "V10C_DESIGN_FREEZE_APPROVAL.json"
DESIGN_FREEZE_APPROVAL_COMMIT = "14df47f1e5ea0448042a0b929150bb34f0702fd8"
DESIGN_FREEZE_APPROVAL_BLOB = "4676eb87c10dfc47fc19be387bfcb2ca17ebc59d"
V10B_SOURCE_FILE = "src/v10b_training_cache_reacquisition.py"
V10B_SOURCE_BLOB = "abb17129870241f18eba2f31a49c5606475a1a0d"
V10B_ACQUISITION_IMPLEMENTATION_SHA = "b2172723df28b3edfce386c77ee79ce38a716925"
V10B_TERMINAL_ADJUDICATION_FILE = "V10B_PHASE_B_PHASE_C_TERMINAL_ADJUDICATION.json"
V10B_TERMINAL_ADJUDICATION_COMMIT = "77ac5f907fbde1dae6d58b7f8e7fde3ae2e8db16"
V10B_TERMINAL_ADJUDICATION_BLOB = "5d1d83fb6bdf1fbe0630e9158c9d039b2e486796"
CANDIDATE_MANIFEST_SHA256 = "887c031a004f91a080fa53ab511711fff92c92527cb119878ab2c295ee13cd44"
CANDIDATE_ATTEMPT_RECEIPT_SHA256 = "44387e68bfc5de8bf54de8ca9694ba9f12bad2be779224946888eaea36d96eb7"
CANDIDATE_SUCCESS_COUNT = 283
CANDIDATE_FAILED_COUNT = 17
CANDIDATE_TICKER_COUNT = 300
MANIFEST_FILE = "cache_manifest.json"
ATTEMPT_RECEIPT_FILE = "V10B_ACQUISITION_ATTEMPT_RECEIPT.json"
LOCKED_RAW_DIRECTORY = "locked_raw"
MANIFEST_SCHEMA = "V10B_TRAINING_CACHE_MANIFEST_V1"
ATTEMPT_RECEIPT_SCHEMA = "V10B_ACQUISITION_ATTEMPT_RECEIPT_V1"
AUTHORIZATION_MARKER_SCHEMA = "V10C_OFFLINE_ADOPTION_AUTHORIZATION_MARKER_V1"
AUTHORIZATION_SCOPE = "ONE_V10C_OFFLINE_PROVENANCE_ADOPTION_EXECUTION_ONLY"
RECEIPT_SCHEMA = "V10C_OFFLINE_ADOPTION_EXECUTION_RECEIPT_V1"
UNIVERSE_FILE = "V4_UNIVERSE.csv"
UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
EXPECTED_REPOSITORY_URL = "https://github.com/ta1k1-arakawa/stock-analyzer.git"
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
PRICE_FROM = "2015-01-01"
PRICE_TO = "2019-12-31"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "complete", "universe_mode", "universe_csv_sha256",
        "ticker_list_sha256", "ticker_count", "ticker_order", "price_from",
        "price_to", "query_specification", "payloads", "network_audit",
        "successful_ticker_count", "failed_tickers", "payload_hash_list_sha256",
    }
)
PAYLOAD_FIELDS = frozenset({"ticker", "relative_path", "sha256", "byte_count"})
AUDIT_FIELDS = frozenset(
    {
        "ticker", "attempt", "scheme", "host", "path", "query_specification",
        "status", "error_type", "redirect_detected", "body_byte_count",
        "payload_sha256", "retry", "final", "success",
    }
)
ALLOWED_ERROR_TYPES = frozenset(
    {None, "TRANSPORT_EXCEPTION", "HTTP_ERROR", "REDIRECT", "EMPTY_BODY", "PARSER_FAILURE"}
)


class V10CError(RuntimeError):
    """Base for bounded V10C validator failures."""


class GovernanceProvenanceFailure(V10CError):
    pass


class LockedArtifactIntegrityFailure(V10CError):
    pass


class ImplementationFailure(V10CError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ImplementationFailure("SERIALIZATION_FAILURE") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode("ascii") + value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_sha1(value: Any) -> bool:
    return isinstance(value, str) and SHA1_RE.fullmatch(value) is not None


def _query_as_lists() -> list[list[str]]:
    return [[key, value] for key, value in QUERY_SPECIFICATION]


def _is_reparse(stat_result: Any) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400) or bool(
        getattr(stat_result, "st_reparse_tag", 0)
    )


def _safe_regular_file(path: Path, failure_cls: type[V10CError]) -> None:
    try:
        stat_result = path.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise failure_cls("REQUIRED_FILE_UNAVAILABLE") from exc
    if path.is_symlink() or _is_reparse(stat_result) or not path.is_file():
        raise failure_cls("REQUIRED_FILE_UNSAFE")


def _assert_existing_ancestor_chain(path: Path) -> None:
    current = path
    while True:
        try:
            stat_result = current.lstat()
        except (FileNotFoundError, OSError) as exc:
            raise GovernanceProvenanceFailure("CANDIDATE_ANCESTOR_UNAVAILABLE") from exc
        if current.is_symlink() or _is_reparse(stat_result):
            raise GovernanceProvenanceFailure("CANDIDATE_ANCESTOR_UNSAFE")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _assert_external_existing_file(path: Path, repo_root: Path, candidate_root: Path) -> Path:
    """Validate an existing operational file without erasing reparse ancestry."""
    if not path.is_absolute():
        raise GovernanceProvenanceFailure("OPERATIONAL_PATH_NOT_ABSOLUTE")
    _assert_existing_ancestor_chain(path)
    _safe_regular_file(path, GovernanceProvenanceFailure)
    resolved = path.resolve(strict=True)
    for protected_root in (repo_root.resolve(strict=True), candidate_root.resolve(strict=True)):
        if resolved == protected_root or resolved.is_relative_to(protected_root):
            raise GovernanceProvenanceFailure("OPERATIONAL_PATH_NOT_EXTERNAL")
    return path


def validate_receipt_output_path(receipt_path: Path, repo_root: Path, candidate_root: Path) -> Path:
    """Validate a new external receipt destination without filesystem mutation."""
    if not receipt_path.is_absolute():
        raise GovernanceProvenanceFailure("RECEIPT_PATH_NOT_ABSOLUTE")
    parent = receipt_path.parent
    _assert_existing_ancestor_chain(parent)
    try:
        parent_stat = parent.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise GovernanceProvenanceFailure("RECEIPT_PARENT_UNAVAILABLE") from exc
    if parent.is_symlink() or _is_reparse(parent_stat) or not parent.is_dir():
        raise GovernanceProvenanceFailure("RECEIPT_PARENT_UNSAFE")
    try:
        receipt_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise GovernanceProvenanceFailure("RECEIPT_PATH_UNAVAILABLE") from exc
    else:
        raise GovernanceProvenanceFailure("RECEIPT_PATH_EXISTS")
    resolved = receipt_path.resolve(strict=False)
    for protected_root in (repo_root.resolve(strict=True), candidate_root.resolve(strict=True)):
        if resolved == protected_root or resolved.is_relative_to(protected_root):
            raise GovernanceProvenanceFailure("RECEIPT_PATH_NOT_EXTERNAL")
    return receipt_path


def assert_candidate_root_safe(candidate_root: Path, repo_root: Path) -> Path:
    if not candidate_root.is_absolute():
        raise GovernanceProvenanceFailure("CANDIDATE_ROOT_NOT_ABSOLUTE")
    _assert_existing_ancestor_chain(candidate_root)
    try:
        root_stat = candidate_root.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise GovernanceProvenanceFailure("CANDIDATE_ROOT_UNAVAILABLE") from exc
    if candidate_root.is_symlink() or _is_reparse(root_stat) or not candidate_root.is_dir():
        raise GovernanceProvenanceFailure("CANDIDATE_ROOT_UNSAFE")
    resolved = candidate_root.resolve(strict=True)
    repo_resolved = repo_root.resolve(strict=True)
    try:
        resolved.relative_to(repo_resolved)
    except ValueError:
        return resolved
    raise GovernanceProvenanceFailure("CANDIDATE_ROOT_INSIDE_REPOSITORY")


def _assert_child_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise LockedArtifactIntegrityFailure("PAYLOAD_PATH_INVALID")
    candidate = root / relative_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise LockedArtifactIntegrityFailure("PAYLOAD_PATH_ESCAPE") from exc
    if candidate.is_symlink():
        raise LockedArtifactIntegrityFailure("PAYLOAD_PATH_SYMLINK")
    return candidate


def _read_json_file(path: Path, failure_cls: type[V10CError]) -> tuple[bytes, Any]:
    _safe_regular_file(path, failure_cls)
    try:
        raw = path.read_bytes()
        return raw, json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise failure_cls("JSON_FILE_INVALID") from exc


def _retry_allowed(status: Any, redirect_detected: bool, attempt: int) -> bool:
    if redirect_detected or attempt >= 3:
        return False
    return status == 429 or (isinstance(status, int) and not isinstance(status, bool) and 500 <= status <= 599)


def _validate_audit_records(audit: Any, ticker_order: Sequence[str]) -> None:
    if not isinstance(audit, list) or not audit:
        raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_MISSING")
    by_ticker: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in ticker_order}
    order: list[tuple[int, int]] = []
    for item in audit:
        if not isinstance(item, dict) or set(item) != AUDIT_FIELDS:
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_FIELDSET_INVALID")
        ticker = item["ticker"]
        attempt = item["attempt"]
        if ticker not in by_ticker or type(attempt) is not int or not 1 <= attempt <= 3:
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_BINDING_INVALID")
        if (
            item["scheme"] != YAHOO_SCHEME
            or item["host"] != YAHOO_HOST
            or item["path"] != f"{YAHOO_PATH_PREFIX}{ticker}.T"
            or item["query_specification"] != _query_as_lists()
        ):
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_SOURCE_INVALID")
        if item["error_type"] not in ALLOWED_ERROR_TYPES:
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_ERROR_TYPE_INVALID")
        if any(type(item[key]) is not bool for key in ("retry", "final", "success", "redirect_detected")):
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_BOOLEAN_INVALID")
        status = item["status"]
        if status is not None and (type(status) is not int or not 100 <= status <= 999):
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_STATUS_INVALID")
        if item["retry"] != _retry_allowed(status, item["redirect_detected"], attempt):
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_RETRY_INVALID")
        if item["success"]:
            if (
                status != 200 or item["redirect_detected"] or item["error_type"] is not None
                or type(item["body_byte_count"]) is not int or item["body_byte_count"] <= 0
                or not _is_sha256(item["payload_sha256"]) or item["retry"] or not item["final"]
            ):
                raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_SUCCESS_INVALID")
        else:
            if item["redirect_detected"]:
                if item["retry"] or not item["final"] or item["error_type"] != "REDIRECT":
                    raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_REDIRECT_INVALID")
            elif item["error_type"] == "REDIRECT":
                raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_REDIRECT_INVALID")
            if item["final"] is False and not item["retry"]:
                raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_NONFINAL_INVALID")
            if item["final"] and item["retry"]:
                raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_FINAL_RETRY_INVALID")
            if item["error_type"] is None:
                raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_FAILURE_TYPE_MISSING")
            if item["payload_sha256"] is not None and not _is_sha256(item["payload_sha256"]):
                raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_PAYLOAD_HASH_INVALID")
        by_ticker[ticker].append(item)
        order.append((ticker_order.index(ticker), attempt))
    if order != sorted(order):
        raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_ORDER_INVALID")
    for ticker, records in by_ticker.items():
        if not records:
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_TICKER_MISSING")
        attempts = [record["attempt"] for record in records]
        if attempts != list(range(1, len(records) + 1)):
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_ATTEMPT_SEQUENCE_INVALID")
        if sum(record["final"] for record in records) != 1 or records[-1]["final"] is not True:
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_TERMINAL_STATE_INVALID")
        successes = [record for record in records if record["success"]]
        if len(successes) > 1 or (successes and successes[0] is not records[-1]):
            raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_SUCCESS_SEQUENCE_INVALID")


def validate_manifest_structure(manifest: Mapping[str, Any], ticker_order: Sequence[str]) -> dict[str, Any]:
    order = list(ticker_order)
    if len(order) != CANDIDATE_TICKER_COUNT or len(set(order)) != CANDIDATE_TICKER_COUNT:
        raise ImplementationFailure("SYNTHETIC_OR_FIXED_TICKER_ORDER_INVALID")
    if not isinstance(manifest, Mapping) or set(manifest) != MANIFEST_FIELDS:
        raise LockedArtifactIntegrityFailure("MANIFEST_FIELDSET_INVALID")
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["complete"] is not True:
        raise LockedArtifactIntegrityFailure("MANIFEST_SCHEMA_OR_COMPLETION_INVALID")
    if (
        manifest["universe_mode"] != "FIXED_V4_300"
        or manifest["universe_csv_sha256"] != UNIVERSE_CSV_SHA256
        or manifest["ticker_list_sha256"] != TICKER_LIST_SHA256
        or type(manifest["ticker_count"]) is not int
        or manifest["ticker_count"] != CANDIDATE_TICKER_COUNT
        or manifest["ticker_order"] != order
        or manifest["price_from"] != PRICE_FROM
        or manifest["price_to"] != PRICE_TO
        or manifest["query_specification"] != _query_as_lists()
    ):
        raise LockedArtifactIntegrityFailure("MANIFEST_FIXED_BINDING_INVALID")
    payloads = manifest["payloads"]
    if not isinstance(payloads, list):
        raise LockedArtifactIntegrityFailure("PAYLOAD_LIST_INVALID")
    accepted: list[str] = []
    for item in payloads:
        if not isinstance(item, dict) or set(item) != PAYLOAD_FIELDS:
            raise LockedArtifactIntegrityFailure("PAYLOAD_FIELDSET_INVALID")
        ticker = item["ticker"]
        if ticker not in order or ticker in accepted or item["relative_path"] != f"{LOCKED_RAW_DIRECTORY}/{ticker}.json":
            raise LockedArtifactIntegrityFailure("PAYLOAD_BINDING_INVALID")
        if not _is_sha256(item["sha256"]) or type(item["byte_count"]) is not int or item["byte_count"] <= 0:
            raise LockedArtifactIntegrityFailure("PAYLOAD_METADATA_INVALID")
        accepted.append(ticker)
    if accepted != [ticker for ticker in order if ticker in set(accepted)]:
        raise LockedArtifactIntegrityFailure("PAYLOAD_ORDER_INVALID")
    if type(manifest["successful_ticker_count"]) is not int or manifest["successful_ticker_count"] != len(payloads):
        raise LockedArtifactIntegrityFailure("SUCCESS_COUNT_INVALID")
    if manifest["successful_ticker_count"] != CANDIDATE_SUCCESS_COUNT:
        raise LockedArtifactIntegrityFailure("CANDIDATE_SUCCESS_COUNT_INVALID")
    if manifest["payload_hash_list_sha256"] != sha256_bytes(canonical_json_bytes(payloads)):
        raise LockedArtifactIntegrityFailure("PAYLOAD_HASH_LIST_MISMATCH")
    failed = manifest["failed_tickers"]
    expected_failed = [ticker for ticker in order if ticker not in set(accepted)]
    if not isinstance(failed, list) or failed != expected_failed:
        raise LockedArtifactIntegrityFailure("FAILED_TICKER_COMPLEMENT_INVALID")
    if len(failed) != CANDIDATE_FAILED_COUNT or set(accepted) & set(failed) or set(accepted) | set(failed) != set(order):
        raise LockedArtifactIntegrityFailure("SUCCESS_FAILURE_PARTITION_INVALID")
    if manifest["successful_ticker_count"] + len(failed) != CANDIDATE_TICKER_COUNT:
        raise LockedArtifactIntegrityFailure("SUCCESS_FAILURE_CARDINALITY_INVALID")
    audit = manifest["network_audit"]
    if not isinstance(audit, list) or len(audit) != CANDIDATE_TICKER_COUNT:
        raise LockedArtifactIntegrityFailure("NETWORK_AUDIT_COUNT_INVALID")
    _validate_audit_records(audit, order)
    for ticker in order:
        records = [record for record in audit if record["ticker"] == ticker]
        success_records = [record for record in records if record["success"]]
        if ticker in accepted:
            if len(success_records) != 1:
                raise LockedArtifactIntegrityFailure("PAYLOAD_AUDIT_SUCCESS_MISSING")
            payload = next(item for item in payloads if item["ticker"] == ticker)
            if success_records[0]["payload_sha256"] != payload["sha256"] or success_records[0]["body_byte_count"] != payload["byte_count"]:
                raise LockedArtifactIntegrityFailure("PAYLOAD_AUDIT_HASH_MISMATCH")
        elif success_records:
            raise LockedArtifactIntegrityFailure("FAILED_TICKER_AUDIT_SUCCESS_INVALID")
    return dict(manifest)


def _validate_attempt_receipt(receipt: Any) -> None:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA:
        raise LockedArtifactIntegrityFailure("ATTEMPT_RECEIPT_SCHEMA_INVALID")
    expected = {
        "study_identity": "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR",
        "frozen_design_git_commit": "1260538c5bef899478806f74ae32d9e4be7b023b",
        "frozen_design_git_blob_sha": "bc57720b6e73cc8c4cf793258a98f78a635af783",
        "freeze_approval_git_blob_sha": "dde0418589088932559b289b3ed26a40e29b62b6",
        "implementation_sha": V10B_ACQUISITION_IMPLEMENTATION_SHA,
        "attempt_started": True,
        "network_acquisition_scope": "V10B_PUBLIC_YAHOO_ONLY",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise LockedArtifactIntegrityFailure("ATTEMPT_RECEIPT_BINDING_INVALID")


def _validate_terminal_adjudication(adjudication: Any) -> None:
    if not isinstance(adjudication, dict):
        raise GovernanceProvenanceFailure("V10B_TERMINAL_ADJUDICATION_INVALID")
    expected = {
        "study": "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR",
        "adjudication_result": "BLOCK",
        "failure_class": "IMPLEMENTATION_FAILURE",
        "failure_subclass": "WRAPPER_EVIDENCE_PROVENANCE_FAILURE",
        "manifest_promoted": False,
        "second_acquisition_allowed": False,
        "t0_authorized": False,
        "historical_evaluation_authorized": False,
        "private_sealed_access_authorized": False,
        "future_profitability_established": False,
    }
    if any(adjudication.get(key) != value for key, value in expected.items()):
        raise GovernanceProvenanceFailure("V10B_TERMINAL_ADJUDICATION_BINDING_INVALID")


def _normalize_repository_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.port not in (None, 443)
        or parsed.username is not None or parsed.password is not None
        or parsed.path != "/ta1k1-arakawa/stock-analyzer.git" or parsed.query or parsed.fragment
    ):
        raise GovernanceProvenanceFailure("REPOSITORY_IDENTITY_MISMATCH")
    return EXPECTED_REPOSITORY_URL


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_root, text=True, capture_output=True, check=True)
    except Exception as exc:
        raise GovernanceProvenanceFailure("GIT_PROVENANCE_UNAVAILABLE") from exc
    return result.stdout.strip()


def _validate_freeze_approval(repo_root: Path) -> None:
    raw, approval = _read_json_file(repo_root / DESIGN_FREEZE_APPROVAL_FILE, GovernanceProvenanceFailure)
    del raw
    expected = {
        "schema_version": "V10C_DESIGN_FREEZE_APPROVAL_V1",
        "study": STUDY_IDENTITY,
        "artifact_role": "DESIGN_FREEZE_APPROVAL",
        "frozen_design_git_commit": FROZEN_V10C_DESIGN_COMMIT,
        "frozen_design_git_blob_sha": FROZEN_V10C_DESIGN_BLOB,
        "design_document": DESIGN_FILE,
        "final_independent_review_result": "PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0",
        "final_independent_review_design_commit": FROZEN_V10C_DESIGN_COMMIT,
        "approval_status": "APPROVED",
        "human_design_freeze_complete": True,
        "approval_scope": "DESIGN_FREEZE_ONLY",
        "approval_artifact_authorizes_implementation_phase_only": True,
        "offline_adoption_execution_authorized": False,
        "raw_locked_payload_read_authorized": False,
        "network_access_authorized": False,
        "yahoo_refetch_authorized": False,
        "t0_authorized": False,
        "historical_evaluation_authorized": False,
        "private_sealed_access_authorized": False,
        "model_fit_authorized": False,
        "backtest_authorized": False,
        "training_input_provenance_adopted": False,
        "manifest_promoted": False,
        "future_profitability_established": False,
        "predecessor_study": "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR",
        "predecessor_status": "TERMINAL_BLOCK_NONREUSABLE",
        "v10b_second_acquisition_allowed": False,
        "fresh_offline_adoption_authorization_required": True,
        "methodology_change_after_freeze_requires": "NEW_STUDY_REQUIRED",
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise GovernanceProvenanceFailure("DESIGN_FREEZE_APPROVAL_BINDING_INVALID")


def load_fixed_universe(repo_root: Path) -> list[str]:
    path = repo_root / UNIVERSE_FILE
    _safe_regular_file(path, GovernanceProvenanceFailure)
    try:
        raw = path.read_bytes()
        normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if sha256_bytes(normalized) != UNIVERSE_CSV_SHA256:
            raise GovernanceProvenanceFailure("UNIVERSE_CSV_HASH_MISMATCH")
        rows = list(csv.DictReader(normalized.decode("utf-8").splitlines()))
        if not rows or set(rows[0]) != {"ticker", "market", "industry"}:
            raise GovernanceProvenanceFailure("UNIVERSE_CSV_INVALID")
        tickers = [row["ticker"] for row in rows]
    except GovernanceProvenanceFailure:
        raise
    except Exception as exc:
        raise GovernanceProvenanceFailure("UNIVERSE_CSV_INVALID") from exc
    if len(tickers) != CANDIDATE_TICKER_COUNT or sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8")) != TICKER_LIST_SHA256:
        raise GovernanceProvenanceFailure("TICKER_LIST_HASH_MISMATCH")
    return tickers


def validate_repository_preflight(repo_root: Path, implementation_sha: str) -> list[str]:
    if not _is_sha1(implementation_sha):
        raise GovernanceProvenanceFailure("IMPLEMENTATION_SHA_INVALID")
    if _normalize_repository_url(_git(repo_root, "config", "--get", "remote.origin.url")) != EXPECTED_REPOSITORY_URL:
        raise GovernanceProvenanceFailure("REPOSITORY_IDENTITY_MISMATCH")
    if _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != AUTHORITATIVE_BRANCH:
        raise GovernanceProvenanceFailure("BRANCH_MISMATCH")
    if _git(repo_root, "rev-parse", "HEAD") != implementation_sha:
        raise GovernanceProvenanceFailure("IMPLEMENTATION_HEAD_MISMATCH")
    if _git(repo_root, "rev-parse", f"origin/{AUTHORITATIVE_BRANCH}") != implementation_sha:
        raise GovernanceProvenanceFailure("REMOTE_HEAD_MISMATCH")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise GovernanceProvenanceFailure("WORKTREE_DIRTY")
    if _git(repo_root, "rev-parse", f"HEAD:{DESIGN_FILE}") != FROZEN_V10C_DESIGN_BLOB:
        raise GovernanceProvenanceFailure("V10C_DESIGN_BLOB_MISMATCH")
    _git(repo_root, "merge-base", "--is-ancestor", FROZEN_V10C_DESIGN_COMMIT, "HEAD")
    if _git(repo_root, "rev-parse", f"HEAD:{DESIGN_FREEZE_APPROVAL_FILE}") != DESIGN_FREEZE_APPROVAL_BLOB:
        raise GovernanceProvenanceFailure("V10C_APPROVAL_BLOB_MISMATCH")
    _git(repo_root, "merge-base", "--is-ancestor", DESIGN_FREEZE_APPROVAL_COMMIT, "HEAD")
    if _git(repo_root, "rev-parse", f"HEAD:{V10B_SOURCE_FILE}") != V10B_SOURCE_BLOB:
        raise GovernanceProvenanceFailure("V10B_SOURCE_BLOB_MISMATCH")
    if _git(repo_root, "rev-parse", f"HEAD:{V10B_TERMINAL_ADJUDICATION_FILE}") != V10B_TERMINAL_ADJUDICATION_BLOB:
        raise GovernanceProvenanceFailure("V10B_TERMINAL_BLOB_MISMATCH")
    _git(repo_root, "merge-base", "--is-ancestor", V10B_TERMINAL_ADJUDICATION_COMMIT, "HEAD")
    _validate_freeze_approval(repo_root)
    _, terminal = _read_json_file(repo_root / V10B_TERMINAL_ADJUDICATION_FILE, GovernanceProvenanceFailure)
    _validate_terminal_adjudication(terminal)
    return load_fixed_universe(repo_root)


def _validate_candidate_metadata(candidate_root: Path, ticker_order: Sequence[str]) -> dict[str, Any]:
    manifest_path = candidate_root / MANIFEST_FILE
    receipt_path = candidate_root / ATTEMPT_RECEIPT_FILE
    receipt_raw, receipt = _read_json_file(receipt_path, LockedArtifactIntegrityFailure)
    if sha256_bytes(receipt_raw) != CANDIDATE_ATTEMPT_RECEIPT_SHA256:
        raise LockedArtifactIntegrityFailure("ATTEMPT_RECEIPT_SHA256_MISMATCH")
    _validate_attempt_receipt(receipt)
    manifest_raw, manifest = _read_json_file(manifest_path, LockedArtifactIntegrityFailure)
    if sha256_bytes(manifest_raw) != CANDIDATE_MANIFEST_SHA256:
        raise LockedArtifactIntegrityFailure("MANIFEST_SHA256_MISMATCH")
    return validate_manifest_structure(manifest, ticker_order)


def phase_a_preflight(
    repo_root: Path,
    candidate_root: Path,
    implementation_sha: str,
    *,
    marker_path: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    ticker_order = validate_repository_preflight(repo_root, implementation_sha)
    safe_root = assert_candidate_root_safe(candidate_root, repo_root)
    if marker_path is not None:
        _assert_external_existing_file(marker_path, repo_root, safe_root)
        validate_authorization_marker(marker_path, implementation_sha)
    if receipt_path is not None:
        validate_receipt_output_path(receipt_path, repo_root, safe_root)
    manifest = _validate_candidate_metadata(safe_root, ticker_order)
    return {"candidate_root": safe_root, "ticker_order": ticker_order, "manifest": manifest, "locked_raw_bytes_read": 0}


def validate_candidate_metadata(candidate_root: Path, ticker_order: Sequence[str]) -> dict[str, Any]:
    """Synthetic-testable Phase-A metadata validation; never reads locked_raw."""
    safe_root = candidate_root
    if not safe_root.is_absolute():
        raise GovernanceProvenanceFailure("CANDIDATE_ROOT_NOT_ABSOLUTE")
    _assert_existing_ancestor_chain(safe_root)
    root_stat = safe_root.lstat()
    if safe_root.is_symlink() or _is_reparse(root_stat) or not safe_root.is_dir():
        raise GovernanceProvenanceFailure("CANDIDATE_ROOT_UNSAFE")
    return _validate_candidate_metadata(safe_root, ticker_order)


def validate_authorization_marker(
    marker_path: Path,
    implementation_sha: str,
    *,
    repo_root: Path | None = None,
    candidate_root: Path | None = None,
) -> dict[str, Any]:
    if repo_root is not None and candidate_root is not None:
        _assert_external_existing_file(marker_path, repo_root, candidate_root)
    raw, marker = _read_json_file(marker_path, GovernanceProvenanceFailure)
    del raw
    expected_fields = {
        "schema_version", "study", "authorization_scope",
        "reviewed_v10c_implementation_sha", "candidate_manifest_sha256",
        "human_authorization_confirmed",
    }
    if not isinstance(marker, dict) or set(marker) != expected_fields:
        raise GovernanceProvenanceFailure("AUTHORIZATION_MARKER_INVALID")
    if (
        marker["schema_version"] != AUTHORIZATION_MARKER_SCHEMA
        or marker["study"] != STUDY_IDENTITY
        or marker["authorization_scope"] != AUTHORIZATION_SCOPE
        or marker["reviewed_v10c_implementation_sha"] != implementation_sha
        or marker["candidate_manifest_sha256"] != CANDIDATE_MANIFEST_SHA256
        or marker["human_authorization_confirmed"] is not True
    ):
        raise GovernanceProvenanceFailure("AUTHORIZATION_MARKER_BINDING_INVALID")
    return marker


def validate_locked_payload_closure(candidate_root: Path, manifest: Mapping[str, Any]) -> int:
    locked_raw = candidate_root / LOCKED_RAW_DIRECTORY
    try:
        stat_result = locked_raw.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise LockedArtifactIntegrityFailure("LOCKED_RAW_DIRECTORY_UNAVAILABLE") from exc
    if locked_raw.is_symlink() or _is_reparse(stat_result) or not locked_raw.is_dir():
        raise LockedArtifactIntegrityFailure("LOCKED_RAW_DIRECTORY_UNSAFE")
    payloads = manifest["payloads"]
    expected_names = {f"{item['ticker']}.json" for item in payloads}
    try:
        entries = list(locked_raw.iterdir())
    except OSError as exc:
        raise LockedArtifactIntegrityFailure("LOCKED_RAW_DIRECTORY_READ_FAILURE") from exc
    if {entry.name for entry in entries} != expected_names or any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise LockedArtifactIntegrityFailure("LOCKED_PAYLOAD_CLOSURE_INVALID")
    total_bytes = 0
    for item in payloads:
        path = _assert_child_path(candidate_root, item["relative_path"])
        _safe_regular_file(path, LockedArtifactIntegrityFailure)
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise LockedArtifactIntegrityFailure("LOCKED_PAYLOAD_READ_FAILURE") from exc
        if len(body) != item["byte_count"] or sha256_bytes(body) != item["sha256"]:
            raise LockedArtifactIntegrityFailure("LOCKED_PAYLOAD_HASH_MISMATCH")
        total_bytes += len(body)
    return total_bytes


def build_safe_receipt(implementation_sha: str, manifest: Mapping[str, Any], *, authorization_consumed: bool) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "study": STUDY_IDENTITY,
        "reviewed_v10c_implementation_sha": implementation_sha,
        "frozen_v10c_design_commit": FROZEN_V10C_DESIGN_COMMIT,
        "frozen_v10c_design_blob_sha": FROZEN_V10C_DESIGN_BLOB,
        "design_freeze_approval_blob_sha": DESIGN_FREEZE_APPROVAL_BLOB,
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "candidate_attempt_receipt_sha256": CANDIDATE_ATTEMPT_RECEIPT_SHA256,
        "successful_ticker_count": manifest["successful_ticker_count"],
        "failed_ticker_count": len(manifest["failed_tickers"]),
        "ticker_count": manifest["ticker_count"],
        "manifest_validation": "PASS",
        "locked_payload_hash_closure": "PASS",
        "network_requests": 0,
        "semantic_payload_parsing": False,
        "t0_authorized": False,
        "historical_evaluation_authorized": False,
        "private_sealed_access_authorized": False,
        "authorization_consumed": authorization_consumed,
        "execution_result": "PASS",
        "failure_class": "NONE",
    }


def _write_exclusive(path: Path, body: bytes) -> None:
    try:
        if not path.parent.is_dir():
            raise OSError("receipt parent is not an existing directory")
        with path.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except (FileExistsError, OSError) as exc:
        raise ImplementationFailure("SAFE_RECEIPT_WRITE_FAILURE") from exc


def phase_b_offline_adoption(
    candidate_root: Path,
    marker_path: Path,
    implementation_sha: str,
    ticker_order: Sequence[str],
    *,
    repo_root: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Read locked payloads only after marker validation, for hash closure."""
    validate_authorization_marker(
        marker_path,
        implementation_sha,
        repo_root=repo_root,
        candidate_root=candidate_root if repo_root is not None else None,
    )
    if receipt_path is not None:
        if repo_root is None:
            raise GovernanceProvenanceFailure("RECEIPT_REPO_BINDING_REQUIRED")
        validate_receipt_output_path(receipt_path, repo_root, candidate_root)
    manifest = _validate_candidate_metadata(candidate_root, ticker_order)
    validate_locked_payload_closure(candidate_root, manifest)
    receipt = build_safe_receipt(implementation_sha, manifest, authorization_consumed=True)
    if receipt_path is not None:
        _write_exclusive(receipt_path, canonical_json_bytes(receipt))
    return receipt
