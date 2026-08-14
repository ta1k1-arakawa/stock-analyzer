"""Durable, privacy-safe V8C stage evidence.

Only aggregate provenance is persisted here.  No ticker, payload, path, or
private membership is ever accepted in a readiness/recheck receipt.  The
receipt is deliberately self-describing and is revalidated at point of use;
an absent, malformed, stale, or mismatched receipt is a BLOCK.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.v8c_human_gate_consumption import (
    GATE_T1C_TRANSPORT_READINESS,
    GATE_T2_TRANSPORT_READINESS,
    V8CHumanGateConsumptionBlocked,
    has_gate_been_consumed,
)

SCHEMA_VERSION = "V8C_DURABLE_STAGE_EVIDENCE_V1"
READINESS_PASS_PREFIX = "v8c_readiness_pass_"
T2_RECHECK_PASS_FILENAME = "v8c_t2_preservation_recheck_pass.json"

# The exact durable readiness human-gate for each stage. A readiness result
# (PASS or BLOCK) may only be recorded here if this exact gate has actually
# been durably, per-authorization consumed -- never merely because the
# caller supplied otherwise-correct/public-looking values. This is what
# stops a caller from manufacturing production-valid PASS evidence purely
# from known public constants (frozen design commit, reviewed
# implementation commit, classifier blob sha, authority prerequisites are
# all independently derivable from the repository) without ever actually
# running a real probe and consuming the one-shot per-authorization gate.
STAGE_READINESS_GATE = {"T1C": GATE_T1C_TRANSPORT_READINESS, "T2": GATE_T2_TRANSPORT_READINESS}


class V8CStageEvidenceBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _canonical(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_NONFINITE_OR_UNSERIALIZABLE") from error


def _strict_object(raw: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_INVALID_JSON") from error
    if not isinstance(parsed, dict):
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_INVALID_JSON")
    return parsed


def _safe_prerequisites(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_AUTHORITY_PREREQUISITES_INVALID")
    if any(not isinstance(key, str) or not isinstance(item, (str, bool, int)) for key, item in value.items()):
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_AUTHORITY_PREREQUISITES_INVALID")
    return dict(value)


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise V8CStageEvidenceBlocked(reason)
    return value


def write_readiness_pass(state_root, *, stage: str, result: str, frozen_design_commit: str,
                         reviewed_implementation_commit: str, sentinel_indices: list[int],
                         probe_start: str, probe_end_exclusive: str,
                         classifier_blob_sha: str, authority_prerequisites: Mapping[str, Any],
                         sentinel_count: int, sentinel_pass_count: int,
                         human_authorization_identity: str,
                         consumption_state_root: Any,
                         clock_text: str) -> dict[str, Any]:
    """Durably record the authoritative result -- PASS or BLOCK -- of one
    real readiness execution. This is deliberately not a generic writer
    that accepts an arbitrary caller-supplied mapping of otherwise-public
    values: it independently re-verifies, against the real durable
    per-authorization human-gate consumption ledger
    (``src.v8c_human_gate_consumption``), that ``human_authorization_identity``
    has actually, durably consumed the exact readiness gate for this stage/
    design-commit -- a real one-shot action a caller cannot fabricate merely
    by knowing public constants. Called after *every* completed readiness
    execution, regardless of outcome, so a later authorized BLOCK durably
    overwrites (and thereby invalidates) an earlier PASS at the same
    destination.
    """
    if stage not in {"T1C", "T2"}:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_STAGE_INVALID")
    if result not in {"PASS", "BLOCK"}:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_RESULT_INVALID")
    if sentinel_indices != [0, 149, 299] or probe_start != "2025-12-01" or probe_end_exclusive != "2025-12-08":
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_SENTINEL_BINDING_INVALID")
    _require_hex(frozen_design_commit, 40, "STAGE_EVIDENCE_DESIGN_COMMIT_INVALID")
    _require_hex(reviewed_implementation_commit, 40, "STAGE_EVIDENCE_IMPLEMENTATION_COMMIT_INVALID")
    _require_hex(classifier_blob_sha, 40, "STAGE_EVIDENCE_CLASSIFIER_SHA_INVALID")
    prerequisites = _safe_prerequisites(authority_prerequisites)
    if type(sentinel_count) is not int or sentinel_count != 3:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_SENTINEL_COUNT_INVALID")
    if type(sentinel_pass_count) is not int or not (0 <= sentinel_pass_count <= 3):
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_SENTINEL_PASS_COUNT_INVALID")
    if result == "PASS" and sentinel_pass_count != 3:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_PASS_REQUIRES_ALL_SENTINELS")
    if not isinstance(human_authorization_identity, str) or not human_authorization_identity:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_AUTHORIZATION_IDENTITY_INVALID")

    gate = STAGE_READINESS_GATE[stage]
    try:
        consumed = has_gate_been_consumed(
            consumption_state_root, gate, frozen_design_commit,
            authorization_identity=human_authorization_identity,
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_GATE_STATE_UNAVAILABLE") from error
    if not consumed:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_NO_MATCHING_CONSUMED_GATE_RECEIPT")

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result": result,
        "stage": stage,
        "frozen_design_commit": frozen_design_commit,
        "reviewed_implementation_commit": reviewed_implementation_commit,
        "sentinel_indices": sentinel_indices,
        "probe_start": probe_start,
        "probe_end_exclusive": probe_end_exclusive,
        "classifier_blob_sha": classifier_blob_sha,
        "authority_prerequisites": prerequisites,
        "sentinel_count": sentinel_count,
        "sentinel_pass_count": sentinel_pass_count,
        "authorization_identity_sha256": hashlib.sha256(human_authorization_identity.encode("utf-8")).hexdigest(),
        "recorded_at_utc": clock_text,
    }
    evidence["evidence_self_hash"] = hashlib.sha256(_canonical(evidence)).hexdigest()
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (READINESS_PASS_PREFIX + stage + ".json")
    staging = root / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "wb") as stream:
            stream.write(_canonical(evidence))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, destination)
    except OSError as error:
        raise V8CStageEvidenceBlocked("STAGE_EVIDENCE_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
    return dict(evidence)


def read_valid_readiness_pass(state_root, *, stage: str, frozen_design_commit: str,
                              reviewed_implementation_commit: str, classifier_blob_sha: str,
                              authority_prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(state_root) / (READINESS_PASS_PREFIX + stage + ".json")
    try:
        evidence = _strict_object(path.read_bytes())
    except OSError as error:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_MISSING") from error
    required = {"schema_version", "result", "stage", "frozen_design_commit", "reviewed_implementation_commit",
                "sentinel_indices", "probe_start", "probe_end_exclusive", "classifier_blob_sha",
                "authority_prerequisites", "sentinel_count", "sentinel_pass_count",
                "authorization_identity_sha256", "recorded_at_utc", "evidence_self_hash"}
    if set(evidence) != required:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_SCHEMA_INVALID")
    stated = evidence["evidence_self_hash"]
    recomputed = hashlib.sha256(_canonical({key: value for key, value in evidence.items() if key != "evidence_self_hash"})).hexdigest()
    if stated != recomputed:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_SELF_HASH_MISMATCH")
    # A later authorized BLOCK durably overwrites (via os.replace) any
    # earlier PASS recorded at this exact destination -- so an older PASS
    # can never remain usable once a later real execution has BLOCKed.
    if evidence["result"] != "PASS":
        raise V8CStageEvidenceBlocked("V8C_READINESS_LATEST_RESULT_NOT_PASS")
    if evidence["stage"] != stage or evidence["frozen_design_commit"] != frozen_design_commit:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_BINDING_MISMATCH")
    if evidence["reviewed_implementation_commit"] != reviewed_implementation_commit:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_IMPLEMENTATION_MISMATCH")
    if evidence["sentinel_indices"] != [0, 149, 299] or evidence["probe_start"] != "2025-12-01" or evidence["probe_end_exclusive"] != "2025-12-08":
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_SENTINEL_MISMATCH")
    if evidence["classifier_blob_sha"] != classifier_blob_sha:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_CLASSIFIER_MISMATCH")
    if evidence["authority_prerequisites"] != dict(authority_prerequisites):
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_AUTHORITY_MISMATCH")
    if evidence["sentinel_count"] != 3 or evidence["sentinel_pass_count"] != 3:
        raise V8CStageEvidenceBlocked("V8C_READINESS_PASS_NOT_ALL_SENTINELS")
    return evidence


_REQUIRED_T2_RECHECK_CONDITIONS = (
    ("t2_real_data_acquired", False), ("t2_opened", False),
    ("t2_research_access_count", 0), ("t2_features_observed", False),
    ("t2_outcomes_observed", False), ("t2_membership_reassigned", False),
    ("universe_definition_compatible", True),
    ("partition_algorithm_compatible", True),
    ("data_quality_policy_unchanged", True),
)


def write_t2_recheck_pass(state_root, evidence: Mapping[str, Any]) -> dict[str, Any]:
    if evidence.get("result") != "PASS" or evidence.get("recheck_point") != "recheck_2":
        raise V8CStageEvidenceBlocked("V8C_T2_RECHECK_PASS_REQUIRED")
    # Not a generic writer accepting an arbitrary caller-supplied mapping:
    # every one of the nine frozen §7.1 conditions must already hold its
    # exact required value before a durable PASS record can be written.
    for field, expected_value in _REQUIRED_T2_RECHECK_CONDITIONS:
        if evidence.get(field) != expected_value:
            raise V8CStageEvidenceBlocked("V8C_T2_RECHECK_PASS_CONDITION_INVALID:" + field)
    safe = dict(evidence)
    safe.pop("evidence_self_hash", None)
    safe["schema_version"] = SCHEMA_VERSION
    safe["result"] = "PASS"
    safe["recheck_point"] = "recheck_2"
    safe["evidence_self_hash"] = hashlib.sha256(_canonical(safe)).hexdigest()
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / T2_RECHECK_PASS_FILENAME
    staging = root / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "wb") as stream:
            stream.write(_canonical(safe)); stream.flush(); os.fsync(stream.fileno())
        os.replace(staging, destination)
    except OSError as error:
        raise V8CStageEvidenceBlocked("V8C_T2_RECHECK_PASS_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists(): staging.unlink()
        except OSError: pass
    return safe


def read_valid_t2_recheck_pass(state_root, *, frozen_design_commit: str,
                               reviewed_implementation_commit: str) -> dict[str, Any]:
    path = Path(state_root) / T2_RECHECK_PASS_FILENAME
    try:
        evidence = _strict_object(path.read_bytes())
    except OSError as error:
        raise V8CStageEvidenceBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_MISSING") from error
    if evidence.get("schema_version") != SCHEMA_VERSION or evidence.get("result") != "PASS" or evidence.get("recheck_point") != "recheck_2":
        raise V8CStageEvidenceBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_INVALID")
    expected = hashlib.sha256(_canonical({key: value for key, value in evidence.items() if key != "evidence_self_hash"})).hexdigest()
    if evidence.get("evidence_self_hash") != expected:
        raise V8CStageEvidenceBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_SELF_HASH_MISMATCH")
    if evidence.get("frozen_design_commit") != frozen_design_commit or evidence.get("reviewed_implementation_commit") != reviewed_implementation_commit:
        raise V8CStageEvidenceBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_BINDING_MISMATCH")
    for field, expected_value in (("t2_real_data_acquired", False), ("t2_opened", False),
                                  ("t2_research_access_count", 0), ("t2_features_observed", False),
                                  ("t2_outcomes_observed", False), ("t2_membership_reassigned", False),
                                  ("universe_definition_compatible", True),
                                  ("partition_algorithm_compatible", True),
                                  ("data_quality_policy_unchanged", True)):
        if evidence.get(field) != expected_value:
            raise V8CStageEvidenceBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_CONDITION_MISMATCH")
    return evidence


__all__ = ["SCHEMA_VERSION", "V8CStageEvidenceBlocked", "read_valid_readiness_pass",
           "read_valid_t2_recheck_pass", "write_readiness_pass", "write_t2_recheck_pass"]
