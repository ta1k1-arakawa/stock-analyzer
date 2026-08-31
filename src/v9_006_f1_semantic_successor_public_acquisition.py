"""Pure, dependency-injected Stage-1 successor-public-acquisition core."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Callable

from src import v9_006_f1_semantic_successor_locator as locator
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL, validate_jpx_url

SCHEMA_VERSION = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_V1"
TASK = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION"
DESIGN_GIT_SHA = "0ee4b338110c626fb92267343586fa6936699805"
OPERATION_CLASS = "RETRIABLE_PUBLIC_PLUMBING"
ROOT_FAMILY, ROOT_PERIOD, TERMINAL_PERIOD = "LISTED_ISSUES_MONTH_END", "TERMINAL_DISCOVERY_ROOT", "TERMINAL"
RESULTS = frozenset({"SUCCESS", "PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "DATA_QUALITY_FAILURE", "INPUT_BINDING_FAILURE", "GOVERNANCE_FAILURE", "IMPLEMENTATION_FAILURE"})
STAGES = frozenset({"NONE", "PRE_NETWORK_INPUT_BINDING", "EXECUTION_BINDING_CONFLICT", "ROOT_TRANSPORT", "TERMINAL_TRANSPORT", "ROOT_LOCATOR", "ROOT_LOCATOR_INPUT_BINDING", "ROOT_PERSISTENCE_EXHAUSTED", "TERMINAL_PERSISTENCE_EXHAUSTED", "IMPLEMENTATION_PRE_ROOT", "IMPLEMENTATION_ROOT_TRANSPORT", "IMPLEMENTATION_POST_ROOT_PRE_LOCATOR", "IMPLEMENTATION_ROOT_LOCATOR", "IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL", "IMPLEMENTATION_TERMINAL_TRANSPORT", "IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE"})
_HEX = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_ROWS = {
    ("SUCCESS", "NONE"): (True, "SUCCESSOR_LOCATOR_MATCHED", True),
    ("PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "ROOT_TRANSPORT"): (False, None, False),
    ("PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "TERMINAL_TRANSPORT"): (True, "SUCCESSOR_LOCATOR_MATCHED", False),
    ("DATA_QUALITY_FAILURE", "ROOT_LOCATOR"): (True, {"SOURCE_OR_DATA_FEASIBILITY_FAILURE", "HTML_STRUCTURE_UNSUPPORTED"}, False),
    ("INPUT_BINDING_FAILURE", "ROOT_LOCATOR_INPUT_BINDING"): (True, "INPUT_BINDING_FAILURE", False),
    ("INPUT_BINDING_FAILURE", "PRE_NETWORK_INPUT_BINDING"): (False, None, False),
    ("GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT"): (False, None, False),
    ("GOVERNANCE_FAILURE", "ROOT_PERSISTENCE_EXHAUSTED"): (False, None, False),
    ("GOVERNANCE_FAILURE", "TERMINAL_PERSISTENCE_EXHAUSTED"): (True, "SUCCESSOR_LOCATOR_MATCHED", False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_PRE_ROOT"): (False, None, False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_TRANSPORT"): (False, None, False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_ROOT_PRE_LOCATOR"): (True, None, False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_LOCATOR"): (True, "SAFE_OUTPUT_VALIDATION_FAILURE", False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL"): (True, "SUCCESSOR_LOCATOR_MATCHED", False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_TERMINAL_TRANSPORT"): (True, "SUCCESSOR_LOCATOR_MATCHED", False),
    ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE"): (True, "SUCCESSOR_LOCATOR_MATCHED", True),
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FetchOutcome:
    http_status: int | None
    payload: bytes | None
    complete: bool
    resolved_url: str


@dataclass(frozen=True)
class VerifiedLock:
    source_family: str
    applicable_period: str
    http_status: int
    payload_sha256: str
    byte_length: int
    resolved_url: str


def _is_int(value: object) -> bool: return type(value) is int
def _hex(value: object, length: int = 64) -> bool: return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def raw_lock_set_sha256(root: VerifiedLock | None, terminal: VerifiedLock | None) -> str | None:
    locks = [item for item in (root, terminal) if item is not None]
    if not locks: return None
    records = [{"source_family": item.source_family, "applicable_period": item.applicable_period, "http_status": item.http_status, "byte_length": item.byte_length, "payload_sha256": item.payload_sha256} for item in locks]
    return sha256(canonical_json(records).encode("utf-8")).hexdigest()


def finalize_safe_result(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["structural_evidence_sha256"] = sha256(canonical_json(result).encode("utf-8")).hexdigest()
    validate_safe_acquisition_result(result)
    return result


def validate_safe_acquisition_result(value: object) -> None:
    keys = {"schema_version", "task", "design_git_sha", "implementation_git_sha", "operation_class", "result", "failure_stage", "discovery_root_http_status", "terminal_http_status", "discovery_root_payload_sha256", "terminal_payload_sha256", "discovery_root_byte_length", "terminal_byte_length", "discovery_root_attempt_count", "terminal_attempt_count", "network_request_count", "discovery_root_locked", "terminal_locked", "semantic_locator_succeeded", "semantic_locator_result", "safe_provenance_verified", "semantic_locator_structural_evidence_sha256", "raw_lock_count", "raw_lock_set_sha256", "structural_evidence_sha256"}
    if type(value) is not dict or set(value) != keys: raise ValueError("keys")
    if value["schema_version"] != SCHEMA_VERSION or value["task"] != TASK or value["design_git_sha"] != DESIGN_GIT_SHA or value["operation_class"] != OPERATION_CLASS or not _hex(value["implementation_git_sha"], 40): raise ValueError("fixed")
    pair = (value["result"], value["failure_stage"])
    if pair not in _ROWS: raise ValueError("row")
    for name in ("discovery_root_attempt_count", "terminal_attempt_count", "network_request_count", "raw_lock_count"):
        if not _is_int(value[name]) or value[name] < 0: raise ValueError("count")
    if value["discovery_root_attempt_count"] > 3 or value["terminal_attempt_count"] > 3 or value["network_request_count"] != value["discovery_root_attempt_count"] + value["terminal_attempt_count"]: raise ValueError("attempts")
    for name in ("discovery_root_locked", "terminal_locked", "semantic_locator_succeeded", "safe_provenance_verified"):
        if type(value[name]) is not bool: raise ValueError("bool")
    for status in (value["discovery_root_http_status"], value["terminal_http_status"]):
        if status is not None and (not _is_int(status) or not 0 <= status <= 599): raise ValueError("status")
    for digest in (value["discovery_root_payload_sha256"], value["terminal_payload_sha256"], value["raw_lock_set_sha256"], value["semantic_locator_structural_evidence_sha256"]):
        if digest is not None and not _hex(digest): raise ValueError("digest")
    for length in (value["discovery_root_byte_length"], value["terminal_byte_length"]):
        if length is not None and (not _is_int(length) or length < 0): raise ValueError("length")
    root_locked, locator_rule, terminal_locked = _ROWS[pair]
    if value["discovery_root_locked"] is not root_locked or value["terminal_locked"] is not terminal_locked: raise ValueError("locks")
    for locked, status, digest, length in ((root_locked, value["discovery_root_http_status"], value["discovery_root_payload_sha256"], value["discovery_root_byte_length"]), (terminal_locked, value["terminal_http_status"], value["terminal_payload_sha256"], value["terminal_byte_length"])):
        if locked != (status == 200 and digest is not None and length is not None): raise ValueError("lock fields")
    if pair[1] in {"ROOT_PERSISTENCE_EXHAUSTED", "TERMINAL_PERSISTENCE_EXHAUSTED"}:
        status = value["discovery_root_http_status"] if pair[1] == "ROOT_PERSISTENCE_EXHAUSTED" else value["terminal_http_status"]
        if status != 200: raise ValueError("persistence status")
    locator = value["semantic_locator_result"]
    if locator_rule is None:
        if locator is not None or value["semantic_locator_structural_evidence_sha256"] is not None or value["semantic_locator_succeeded"]: raise ValueError("locator absent")
    else:
        allowed = locator_rule if type(locator_rule) is set else {locator_rule}
        if locator not in allowed or value["semantic_locator_structural_evidence_sha256"] is None or value["semantic_locator_succeeded"] != (locator == "SUCCESSOR_LOCATOR_MATCHED"): raise ValueError("locator")
    if value["terminal_attempt_count"] and not value["semantic_locator_succeeded"]: raise ValueError("terminal before locator")
    expected_locks = int(root_locked) + int(terminal_locked)
    if value["raw_lock_count"] != expected_locks: raise ValueError("lock count")
    if expected_locks == 0:
        if value["raw_lock_set_sha256"] is not None: raise ValueError("set")
    else:
        records = []
        if root_locked: records.append({"source_family": ROOT_FAMILY, "applicable_period": ROOT_PERIOD, "http_status": 200, "byte_length": value["discovery_root_byte_length"], "payload_sha256": value["discovery_root_payload_sha256"]})
        if terminal_locked: records.append({"source_family": ROOT_FAMILY, "applicable_period": TERMINAL_PERIOD, "http_status": 200, "byte_length": value["terminal_byte_length"], "payload_sha256": value["terminal_payload_sha256"]})
        if value["raw_lock_set_sha256"] != sha256(canonical_json(records).encode("utf-8")).hexdigest(): raise ValueError("set")
    provenance = pair not in {("PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "ROOT_TRANSPORT"), ("INPUT_BINDING_FAILURE", "PRE_NETWORK_INPUT_BINDING"), ("GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT"), ("GOVERNANCE_FAILURE", "ROOT_PERSISTENCE_EXHAUSTED"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_PRE_ROOT"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_TRANSPORT"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE")}
    if value["safe_provenance_verified"] is not provenance: raise ValueError("provenance")
    structural = dict(value); digest = structural.pop("structural_evidence_sha256")
    if not _hex(digest) or sha256(canonical_json(structural).encode("utf-8")).hexdigest() != digest: raise ValueError("structural")


def _lock_ok(lock: object, payload: bytes, family: str, period: str, resolved_url: str) -> bool:
    try:
        if type(resolved_url) is not str: return False
        validate_jpx_url(resolved_url)
        if type(lock) is not VerifiedLock or type(lock.resolved_url) is not str: return False
        validate_jpx_url(lock.resolved_url)
    except Exception: return False
    return lock.source_family == family and lock.applicable_period == period and lock.http_status == 200 and lock.payload_sha256 == sha256(payload).hexdigest() and lock.byte_length == len(payload) and lock.resolved_url == resolved_url


def _transport(fetch: Callable[[str, int], FetchOutcome], url: str, delay: Callable[[int], None]) -> tuple[FetchOutcome | None, int, int | None, bool]:
    latest = None
    for attempt, seconds in enumerate((0, 2, 5), 1):
        delay(seconds)
        try: outcome = fetch(url, attempt)
        except Exception: return None, attempt, latest, True
        if type(outcome) is not FetchOutcome: return None, attempt, latest, True
        try:
            if type(outcome.resolved_url) is not str: raise ValueError("resolved_url")
            validate_jpx_url(outcome.resolved_url)
            if outcome.resolved_url != url: raise ValueError("endpoint")
        except Exception: return None, attempt, latest, True
        if outcome.http_status is not None and _is_int(outcome.http_status) and 0 <= outcome.http_status <= 599: latest = outcome.http_status
        if outcome.http_status == 200 and outcome.complete is True and type(outcome.payload) is bytes:
            return outcome, attempt, latest, False
    return None, 3, latest, False


def _persist(callback: Callable[[str, str, bytes, str, int], VerifiedLock | None], payload: bytes, resolved_url: str, period: str, delay: Callable[[int], None]) -> VerifiedLock | None:
    for attempt, seconds in enumerate((0, 1, 2), 1):
        delay(seconds)
        try: lock = callback(ROOT_FAMILY, period, payload, resolved_url, attempt)
        except Exception: lock = None
        if _lock_ok(lock, payload, ROOT_FAMILY, period, resolved_url): return lock
    return None


def _base(implementation_git_sha: str, result: str, stage: str, root: VerifiedLock | None = None, terminal: VerifiedLock | None = None, *, root_status: int | None = None, terminal_status: int | None = None, root_attempts: int = 0, terminal_attempts: int = 0, locator_result: str | None = None, locator_hash: str | None = None) -> dict[str, object]:
    root_status = root.http_status if root else root_status; terminal_status = terminal.http_status if terminal else terminal_status
    provenance = (result, stage) not in {("PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "ROOT_TRANSPORT"), ("INPUT_BINDING_FAILURE", "PRE_NETWORK_INPUT_BINDING"), ("GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT"), ("GOVERNANCE_FAILURE", "ROOT_PERSISTENCE_EXHAUSTED"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_PRE_ROOT"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_TRANSPORT"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE")}
    return {"schema_version": SCHEMA_VERSION, "task": TASK, "design_git_sha": DESIGN_GIT_SHA, "implementation_git_sha": implementation_git_sha, "operation_class": OPERATION_CLASS, "result": result, "failure_stage": stage, "discovery_root_http_status": root_status, "terminal_http_status": terminal_status, "discovery_root_payload_sha256": root.payload_sha256 if root else None, "terminal_payload_sha256": terminal.payload_sha256 if terminal else None, "discovery_root_byte_length": root.byte_length if root else None, "terminal_byte_length": terminal.byte_length if terminal else None, "discovery_root_attempt_count": root_attempts, "terminal_attempt_count": terminal_attempts, "network_request_count": root_attempts + terminal_attempts, "discovery_root_locked": root is not None, "terminal_locked": terminal is not None, "semantic_locator_succeeded": locator_result == "SUCCESSOR_LOCATOR_MATCHED", "semantic_locator_result": locator_result, "safe_provenance_verified": provenance, "semantic_locator_structural_evidence_sha256": locator_hash, "raw_lock_count": int(root is not None) + int(terminal is not None), "raw_lock_set_sha256": raw_lock_set_sha256(root, terminal)}


def run_pure_acquisition(implementation_git_sha: str, root_url: str, root_fetch: Callable[[str, int], FetchOutcome], terminal_fetch: Callable[[str, int], FetchOutcome], persist: Callable[[str, str, bytes, str, int], VerifiedLock | None], delay: Callable[[int], None] = lambda _seconds: None, locator_runner: Callable[[bytes, str, str, int], tuple[dict[str, object], str | None]] = locator.run_fresh_root_locator) -> dict[str, object]:
    if not _hex(implementation_git_sha, 40): raise ValueError("implementation_git_sha")
    try:
        if type(root_url) is not str or root_url != LISTED_ISSUES_PAGE_URL: raise ValueError("root endpoint")
        validate_jpx_url(root_url)
    except Exception: return finalize_safe_result(_base(implementation_git_sha, "INPUT_BINDING_FAILURE", "PRE_NETWORK_INPUT_BINDING"))
    outcome, root_attempts, root_status, impl = _transport(root_fetch, root_url, delay)
    if outcome is None:
        return finalize_safe_result(_base(implementation_git_sha, "IMPLEMENTATION_FAILURE" if impl else "PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "IMPLEMENTATION_ROOT_TRANSPORT" if impl else "ROOT_TRANSPORT", root_status=root_status, root_attempts=root_attempts))
    root = _persist(persist, outcome.payload, outcome.resolved_url, ROOT_PERIOD, delay)
    if root is None: return finalize_safe_result(_base(implementation_git_sha, "GOVERNANCE_FAILURE", "ROOT_PERSISTENCE_EXHAUSTED", root_status=200, root_attempts=root_attempts))
    try:
        safe_locator, private_url = locator_runner(outcome.payload, root.resolved_url, root.payload_sha256, root.byte_length); locator.validate_fresh_safe_result(safe_locator)
        if safe_locator["input_payload_sha256"] != root.payload_sha256 or safe_locator["input_payload_byte_length"] != root.byte_length: raise ValueError("binding")
    except Exception:
        return finalize_safe_result(_base(implementation_git_sha, "IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_LOCATOR", root, root_attempts=root_attempts, locator_result="SAFE_OUTPUT_VALIDATION_FAILURE", locator_hash="0" * 64))
    locator_result, locator_hash = safe_locator["result"], safe_locator["structural_evidence_sha256"]
    if locator_result in {"SOURCE_OR_DATA_FEASIBILITY_FAILURE", "HTML_STRUCTURE_UNSUPPORTED"}: return finalize_safe_result(_base(implementation_git_sha, "DATA_QUALITY_FAILURE", "ROOT_LOCATOR", root, root_attempts=root_attempts, locator_result=locator_result, locator_hash=locator_hash))
    if locator_result == "INPUT_BINDING_FAILURE": return finalize_safe_result(_base(implementation_git_sha, "INPUT_BINDING_FAILURE", "ROOT_LOCATOR_INPUT_BINDING", root, root_attempts=root_attempts, locator_result=locator_result, locator_hash=locator_hash))
    if locator_result != "SUCCESSOR_LOCATOR_MATCHED": return finalize_safe_result(_base(implementation_git_sha, "IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_LOCATOR", root, root_attempts=root_attempts, locator_result="SAFE_OUTPUT_VALIDATION_FAILURE", locator_hash=locator_hash))
    try:
        if type(private_url) is not str: raise ValueError("private_url")
        validate_jpx_url(private_url)
        selected_url_sha256 = safe_locator["selected_resolved_url_sha256"]
        if not _hex(selected_url_sha256) or sha256(private_url.encode("utf-8")).hexdigest() != selected_url_sha256: raise ValueError("selected_url_binding")
    except Exception: return finalize_safe_result(_base(implementation_git_sha, "IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL", root, root_attempts=root_attempts, locator_result=locator_result, locator_hash=locator_hash))
    outcome, terminal_attempts, terminal_status, impl = _transport(terminal_fetch, private_url, delay)
    if outcome is None: return finalize_safe_result(_base(implementation_git_sha, "IMPLEMENTATION_FAILURE" if impl else "PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "IMPLEMENTATION_TERMINAL_TRANSPORT" if impl else "TERMINAL_TRANSPORT", root, root_status=root.http_status, terminal_status=terminal_status, root_attempts=root_attempts, terminal_attempts=terminal_attempts, locator_result=locator_result, locator_hash=locator_hash))
    terminal = _persist(persist, outcome.payload, outcome.resolved_url, TERMINAL_PERIOD, delay)
    if terminal is None: return finalize_safe_result(_base(implementation_git_sha, "GOVERNANCE_FAILURE", "TERMINAL_PERSISTENCE_EXHAUSTED", root, terminal_status=200, root_attempts=root_attempts, terminal_attempts=terminal_attempts, locator_result=locator_result, locator_hash=locator_hash))
    return finalize_safe_result(_base(implementation_git_sha, "SUCCESS", "NONE", root, terminal, root_attempts=root_attempts, terminal_attempts=terminal_attempts, locator_result=locator_result, locator_hash=locator_hash))
