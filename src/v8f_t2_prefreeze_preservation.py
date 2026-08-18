"""Safe V8F T2 pre-freeze preservation audit/review support.

This is the minimum support authorized by V8F design §8/§10 and inherited
V8E §9.A item 3 ("safe V8F T2 prefreeze preservation audit/review support").
Only caller-supplied safe evidence mappings enter this module.  There is no
private-manifest path, ticker-identity reader, gate consumer, artifact
writer, acquisition authority, or research-opening path.  The exact V8F T2
record is produced in memory for synthetic verification only; the committed
checkpoint is a later separately authorized stage.

Scope note: unlike the V8E precedent this module mechanically rebinds, it
does not attempt to independently re-derive the nine conditions from a chain
of historical V8/V8B/V8C/V8D/V8E committed Git objects, because those exact
blob bindings were not supplied to this task.  Safe evidence for the nine
conditions must be supplied by the caller (dependency injection); this
module's responsibility is exact-schema validation and fail-closed
derivation from that supplied evidence, exactly as the V8E precedent does in
its own pure ``_validate_safe_evidence``/``verify_t2_prefreeze_record`` path.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


V8F_STUDY_NAME = "V8F_HISTORICAL_RESEARCH"
V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT = "cd67a9d61172da74364504cd4f93caec521a2bfc"
V8F_DESIGN_CANDIDATE_BLOB_SHA = "b7eec2b84217ad53d2e2f7dfe917396f13e51428"
V8F_T2_PREFREEZE_CHECKPOINT = "V8F_T2_PREFREEZE_PRESERVATION_RECHECK"
V8F_T2_PREFREEZE_DOCUMENT_TYPE = "T2_PREFREEZE_PRESERVATION_RECHECK_AUDIT_RECORD"

V8F_T2_PREFREEZE_RECORD_FIELDS = (
    "study",
    "document_type",
    "reviewed_v8f_design_candidate_commit",
    "checkpoint",
    "recheck_1",
    "T2_real_data_acquired",
    "T2_opened",
    "T2_research_access_count",
    "T2_features_observed",
    "T2_outcomes_observed",
    "T2_membership_reassigned",
    "universe_definition_compatible",
    "partition_algorithm_compatible",
    "data_quality_policy_unchanged",
    "v8_trusted_partition_git_blob",
    "original_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "t2_count",
    "t2_ticker_list_sha256",
    "T2_PREFREEZE_PRESERVATION_RECHECK",
    "OVERALL_RESULT",
)

T2_SAFE_CONDITION_FIELDS = (
    "T2_real_data_acquired",
    "T2_opened",
    "T2_research_access_count",
    "T2_features_observed",
    "T2_outcomes_observed",
    "T2_membership_reassigned",
    "universe_definition_compatible",
    "partition_algorithm_compatible",
    "data_quality_policy_unchanged",
)

V8F_T2_SAFE_EVIDENCE_FIELDS = T2_SAFE_CONDITION_FIELDS + (
    "v8_trusted_partition_git_blob",
    "original_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "t2_count",
    "t2_ticker_list_sha256",
)

EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA = "61faade0625139cec3fb61216ab2f97f572a7028"
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
EXPECTED_T2_COUNT = 300
EXPECTED_T2_TICKER_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"

_HEX = set("0123456789abcdef")


class V8FT2PrefreezePreservationBlocked(RuntimeError):
    """Fail-closed V8F T2 safe-evidence error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in _HEX for char in value):
        raise V8FT2PrefreezePreservationBlocked(reason)
    return value


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8FT2PrefreezePreservationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8FT2PrefreezePreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8FT2PrefreezePreservationBlocked(invalid_reason)
    return parsed


def _validate_nine_conditions(safe_conditions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(safe_conditions, Mapping) or set(safe_conditions) != set(T2_SAFE_CONDITION_FIELDS):
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_SAFE_CONDITIONS_SCHEMA_INVALID")
    for field in (
        "T2_real_data_acquired",
        "T2_opened",
        "T2_features_observed",
        "T2_outcomes_observed",
        "T2_membership_reassigned",
    ):
        if type(safe_conditions[field]) is not bool:
            raise V8FT2PrefreezePreservationBlocked("V8F_T2_CONDITION_TYPE_INVALID:" + field)
        if safe_conditions[field] is not False:
            raise V8FT2PrefreezePreservationBlocked("V8F_T2_CONDITION_BLOCKED:" + field)
    if type(safe_conditions["T2_research_access_count"]) is not int:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_CONDITION_TYPE_INVALID:T2_research_access_count")
    if safe_conditions["T2_research_access_count"] != 0:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_CONDITION_BLOCKED:T2_research_access_count")
    for field in (
        "universe_definition_compatible",
        "partition_algorithm_compatible",
        "data_quality_policy_unchanged",
    ):
        if type(safe_conditions[field]) is not bool:
            raise V8FT2PrefreezePreservationBlocked("V8F_T2_CONDITION_TYPE_INVALID:" + field)
        if safe_conditions[field] is not True:
            raise V8FT2PrefreezePreservationBlocked("V8F_T2_CONDITION_BLOCKED:" + field)
    return {key: safe_conditions[key] for key in T2_SAFE_CONDITION_FIELDS}


def _validate_safe_evidence(safe_evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(safe_evidence, Mapping) or set(safe_evidence) != set(V8F_T2_SAFE_EVIDENCE_FIELDS):
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_SAFE_EVIDENCE_SCHEMA_INVALID")
    conditions = _validate_nine_conditions(
        {key: safe_evidence[key] for key in T2_SAFE_CONDITION_FIELDS}
    )
    _require_hex(safe_evidence["v8_trusted_partition_git_blob"], 40, "V8F_T2_TRUSTED_BLOB_INVALID")
    _require_hex(safe_evidence["original_v8_partition_manifest_sha256"], 64, "V8F_T2_MANIFEST_SHA_INVALID")
    _require_hex(safe_evidence["parent_v8_partition_implementation_commit"], 40, "V8F_T2_IMPLEMENTATION_COMMIT_INVALID")
    if safe_evidence["v8_trusted_partition_git_blob"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_TRUSTED_BLOB_MISMATCH")
    if safe_evidence["original_v8_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_MANIFEST_SHA_MISMATCH")
    if safe_evidence["parent_v8_partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_IMPLEMENTATION_COMMIT_MISMATCH")
    if type(safe_evidence["t2_count"]) is not int or safe_evidence["t2_count"] != EXPECTED_T2_COUNT:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_COUNT_MISMATCH")
    _require_hex(safe_evidence["t2_ticker_list_sha256"], 64, "V8F_T2_TICKER_LIST_SHA_INVALID")
    if safe_evidence["t2_ticker_list_sha256"] != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_TICKER_LIST_SHA_MISMATCH")
    return {
        **conditions,
        "v8_trusted_partition_git_blob": safe_evidence["v8_trusted_partition_git_blob"],
        "original_v8_partition_manifest_sha256": safe_evidence["original_v8_partition_manifest_sha256"],
        "parent_v8_partition_implementation_commit": safe_evidence["parent_v8_partition_implementation_commit"],
        "t2_count": safe_evidence["t2_count"],
        "t2_ticker_list_sha256": safe_evidence["t2_ticker_list_sha256"],
    }


def verify_t2_prefreeze_record(
    record: Mapping[str, Any], *, safe_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently validate the exact record against safe evidence."""
    if not isinstance(record, Mapping) or set(record) != set(V8F_T2_PREFREEZE_RECORD_FIELDS):
        raise V8FT2PrefreezePreservationBlocked("V8F_T2_RECORD_SCHEMA_INVALID")
    safe = _validate_safe_evidence(safe_evidence)
    exact = {
        "study": V8F_STUDY_NAME,
        "document_type": V8F_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "checkpoint": V8F_T2_PREFREEZE_CHECKPOINT,
        "recheck_1": "before_V8F_design_freeze",
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }
    for key, expected in exact.items():
        if record[key] != expected:
            raise V8FT2PrefreezePreservationBlocked("V8F_T2_RECORD_VALUE_MISMATCH:" + key)
    for key in V8F_T2_SAFE_EVIDENCE_FIELDS:
        if record[key] != safe[key]:
            raise V8FT2PrefreezePreservationBlocked("V8F_T2_RECORD_EVIDENCE_MISMATCH:" + key)
    # Re-run the nine conditions on the record itself; a declared PASS is not
    # evidence without this independent derivation and comparison.
    _validate_nine_conditions({key: record[key] for key in T2_SAFE_CONDITION_FIELDS})
    return {
        "result": "PASS",
        "checkpoint": V8F_T2_PREFREEZE_CHECKPOINT,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "nine_conditions_independently_verified": True,
        "provenance_independently_verified": True,
    }


def build_t2_prefreeze_record(safe_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build an in-memory synthetic record only; never writes the checkpoint."""
    safe = _validate_safe_evidence(safe_evidence)
    record = {
        "study": V8F_STUDY_NAME,
        "document_type": V8F_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "checkpoint": V8F_T2_PREFREEZE_CHECKPOINT,
        "recheck_1": "before_V8F_design_freeze",
        **safe,
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }
    verify_t2_prefreeze_record(record, safe_evidence=safe)
    return record


def verify_t2_prefreeze_record_bytes(raw: bytes, *, safe_evidence: Mapping[str, Any]) -> dict[str, Any]:
    record = _strict_json_object(raw, "V8F_T2_RECORD_INVALID_JSON", "V8F_T2_RECORD_DUPLICATE_KEY")
    return verify_t2_prefreeze_record(record, safe_evidence=safe_evidence)


__all__ = [
    "EXPECTED_T2_COUNT",
    "EXPECTED_T2_TICKER_LIST_SHA256",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA",
    "T2_SAFE_CONDITION_FIELDS",
    "V8F_DESIGN_CANDIDATE_BLOB_SHA",
    "V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT",
    "V8F_STUDY_NAME",
    "V8F_T2_PREFREEZE_RECORD_FIELDS",
    "V8F_T2_SAFE_EVIDENCE_FIELDS",
    "V8FT2PrefreezePreservationBlocked",
    "build_t2_prefreeze_record",
    "verify_t2_prefreeze_record",
    "verify_t2_prefreeze_record_bytes",
]
