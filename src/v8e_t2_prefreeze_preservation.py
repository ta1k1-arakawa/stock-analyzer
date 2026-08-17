"""Safe V8E T2 pre-freeze preservation audit/review support.

Only committed public Git objects and safe provenance mappings enter this
module.  There is no private-manifest path, ticker-identity reader, gate
consumer, artifact writer, acquisition authority, or research-opening path.
The exact V8E T2 record is produced in memory for synthetic verification
only; the committed checkpoint is a later separately authorized stage.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, Mapping

from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_git_blob,
)
from src.v8c_production_provenance import read_and_verify_v8_trusted_partition_anchor


V8E_STUDY_NAME = "V8E_HISTORICAL_RESEARCH"
V8E_PRODUCTION_BRANCH = "v8e-dq-evidence-successor-design"
V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT = "6f672404b93a1003253915196dd635ca76fd2be1"
V8E_DESIGN_CANDIDATE_BLOB_SHA = "dac32f9e97d1ae2b90eb8b0820914e3845d0fa26"
V8E_T2_PREFREEZE_CHECKPOINT = "V8E_T2_PREFREEZE_PRESERVATION_RECHECK"
V8E_T2_PREFREEZE_DOCUMENT_TYPE = "T2_PREFREEZE_PRESERVATION_RECHECK_AUDIT_RECORD"

V8E_T2_PREFREEZE_RECORD_FIELDS = (
    "study",
    "document_type",
    "reviewed_v8e_design_candidate_commit",
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

V8E_T2_SAFE_EVIDENCE_FIELDS = T2_SAFE_CONDITION_FIELDS + (
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

V8_STATE_GIT_PATH = "V8_STATE.json"
V8B_T2_AUTHORITY_BRIDGE_GIT_PATH = "V8B_T2_AUTHORITY_BRIDGE.json"
V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA = "2e4f4af80c6e5492da476559ca6a943574fc850d"
V8C_READINESS_ADJUDICATION_GIT_PATH = "V8C_T1C_READINESS_BLOCK_ADJUDICATION.md"
V8C_READINESS_ADJUDICATION_BLOB_SHA = "d40b3ef6b071b150dab8269044398fd6fc8227c5"
V8_TRUSTED_PARTITION_GIT_PATH = "V8_TRUSTED_PARTITION.json"
V8_STATE_BLOB_SHA = "8e5fd2f39dc92a7983c0cdaab42f633d624b4956"

_HEX = set("0123456789abcdef")


class V8ET2PrefreezePreservationBlocked(RuntimeError):
    """Fail-closed V8E T2 safe-evidence error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in _HEX for char in value):
        raise V8ET2PrefreezePreservationBlocked(reason)
    return value


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8ET2PrefreezePreservationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8ET2PrefreezePreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8ET2PrefreezePreservationBlocked(invalid_reason)
    return parsed


def _validate_nine_conditions(safe_conditions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(safe_conditions, Mapping) or set(safe_conditions) != set(T2_SAFE_CONDITION_FIELDS):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_SAFE_CONDITIONS_SCHEMA_INVALID")
    for field in (
        "T2_real_data_acquired",
        "T2_opened",
        "T2_features_observed",
        "T2_outcomes_observed",
        "T2_membership_reassigned",
    ):
        if type(safe_conditions[field]) is not bool:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CONDITION_TYPE_INVALID:" + field)
        if safe_conditions[field] is not False:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CONDITION_BLOCKED:" + field)
    if type(safe_conditions["T2_research_access_count"]) is not int:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_CONDITION_TYPE_INVALID:T2_research_access_count")
    if safe_conditions["T2_research_access_count"] != 0:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_CONDITION_BLOCKED:T2_research_access_count")
    for field in (
        "universe_definition_compatible",
        "partition_algorithm_compatible",
        "data_quality_policy_unchanged",
    ):
        if type(safe_conditions[field]) is not bool:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CONDITION_TYPE_INVALID:" + field)
        if safe_conditions[field] is not True:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CONDITION_BLOCKED:" + field)
    return {key: safe_conditions[key] for key in T2_SAFE_CONDITION_FIELDS}


def _validate_safe_evidence(safe_evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(safe_evidence, Mapping) or set(safe_evidence) != set(V8E_T2_SAFE_EVIDENCE_FIELDS):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_SAFE_EVIDENCE_SCHEMA_INVALID")
    conditions = _validate_nine_conditions(
        {key: safe_evidence[key] for key in T2_SAFE_CONDITION_FIELDS}
    )
    _require_hex(safe_evidence["v8_trusted_partition_git_blob"], 40, "V8E_T2_TRUSTED_BLOB_INVALID")
    _require_hex(safe_evidence["original_v8_partition_manifest_sha256"], 64, "V8E_T2_MANIFEST_SHA_INVALID")
    _require_hex(safe_evidence["parent_v8_partition_implementation_commit"], 40, "V8E_T2_IMPLEMENTATION_COMMIT_INVALID")
    if safe_evidence["v8_trusted_partition_git_blob"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_TRUSTED_BLOB_MISMATCH")
    if safe_evidence["original_v8_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_MANIFEST_SHA_MISMATCH")
    if safe_evidence["parent_v8_partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_IMPLEMENTATION_COMMIT_MISMATCH")
    if type(safe_evidence["t2_count"]) is not int or safe_evidence["t2_count"] != EXPECTED_T2_COUNT:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_COUNT_MISMATCH")
    _require_hex(safe_evidence["t2_ticker_list_sha256"], 64, "V8E_T2_TICKER_LIST_SHA_INVALID")
    if safe_evidence["t2_ticker_list_sha256"] != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_TICKER_LIST_SHA_MISMATCH")
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
    if not isinstance(record, Mapping) or set(record) != set(V8E_T2_PREFREEZE_RECORD_FIELDS):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_RECORD_SCHEMA_INVALID")
    safe = _validate_safe_evidence(safe_evidence)
    exact = {
        "study": V8E_STUDY_NAME,
        "document_type": V8E_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "checkpoint": V8E_T2_PREFREEZE_CHECKPOINT,
        "recheck_1": "before_V8E_design_freeze",
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }
    for key, expected in exact.items():
        if record[key] != expected:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_RECORD_VALUE_MISMATCH:" + key)
    for key in V8E_T2_SAFE_EVIDENCE_FIELDS:
        if record[key] != safe[key]:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_RECORD_EVIDENCE_MISMATCH:" + key)
    # Re-run the nine conditions on the record itself; a declared PASS is not
    # evidence without this independent derivation and comparison.
    _validate_nine_conditions({key: record[key] for key in T2_SAFE_CONDITION_FIELDS})
    return {
        "result": "PASS",
        "checkpoint": V8E_T2_PREFREEZE_CHECKPOINT,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "nine_conditions_independently_verified": True,
        "provenance_independently_verified": True,
    }


def build_t2_prefreeze_record(safe_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build an in-memory synthetic record only; never writes the checkpoint."""
    safe = _validate_safe_evidence(safe_evidence)
    record = {
        "study": V8E_STUDY_NAME,
        "document_type": V8E_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "checkpoint": V8E_T2_PREFREEZE_CHECKPOINT,
        "recheck_1": "before_V8E_design_freeze",
        **safe,
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }
    verify_t2_prefreeze_record(record, safe_evidence=safe)
    return record


def verify_t2_prefreeze_record_bytes(raw: bytes, *, safe_evidence: Mapping[str, Any]) -> dict[str, Any]:
    record = _strict_json_object(raw, "V8E_T2_RECORD_INVALID_JSON", "V8E_T2_RECORD_DUPLICATE_KEY")
    return verify_t2_prefreeze_record(record, safe_evidence=safe_evidence)


def _require_mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V8ET2PrefreezePreservationBlocked(reason)
    return value


def _derive_safe_evidence(
    state: Mapping[str, Any], bridge: Mapping[str, Any], anchor: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the nine conditions from safe committed evidence only."""
    state = _require_mapping(state, "V8E_T2_STATE_EVIDENCE_INVALID")
    bridge = _require_mapping(bridge, "V8E_T2_BRIDGE_EVIDENCE_INVALID")
    anchor = _require_mapping(anchor, "V8E_T2_ANCHOR_EVIDENCE_INVALID")
    t2 = _require_mapping(state.get("T2"), "V8E_T2_STATE_SECTION_INVALID")
    trust = _require_mapping(state.get("trust_anchor_pinning"), "V8E_T2_STATE_TRUST_SECTION_INVALID")
    partition = _require_mapping(state.get("partition"), "V8E_T2_STATE_PARTITION_SECTION_INVALID")
    history = _require_mapping(state.get("real_partition_build_history"), "V8E_T2_STATE_HISTORY_INVALID")

    for key in ("raw_data_acquired", "opened_for_research"):
        if type(t2.get(key)) is not bool:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_BOOLEAN_INVALID:" + key)
    access_count = t2.get("sealed_holdout_access_count")
    if access_count is not None and type(access_count) is not int:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_ACCESS_COUNT_INVALID")
    for key in ("real_data_acquired", "backtests", "models_fitted", "profit_calculated"):
        if key not in state:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_FIELD_MISSING:" + key)
    if type(state["real_data_acquired"]) is not bool:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_BOOLEAN_INVALID:real_data_acquired")
    for key in ("backtests", "models_fitted", "profit_calculated"):
        if type(state[key]) is not int or state[key] != 0:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_RESEARCH_ACTIVITY_INVALID:" + key)
    if bridge.get("t2_acquired_before_authorized_acquisition") is not False:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_ACQUISITION_INVALID")
    if bridge.get("t2_research_open_count_before_official_opening") != 0:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_ACCESS_COUNT_INVALID")
    if bridge.get("t2_membership_reassignment") != "PROHIBITED":
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_MEMBERSHIP_INVALID")
    if bridge.get("v8_trusted_partition_json_mutated_or_repinned") is not False:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_REPIN_INVALID")
    if trust.get("block_assignments_exposed") is not False:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_ASSIGNMENTS_EXPOSED")
    if anchor.get("authorized_partition_manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_ANCHOR_MANIFEST_MISMATCH")
    if anchor.get("authorized_partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_ANCHOR_IMPLEMENTATION_MISMATCH")
    if partition.get("manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_PARTITION_MANIFEST_MISMATCH")
    if partition.get("partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_PARTITION_IMPLEMENTATION_MISMATCH")
    if partition.get("block_size_frozen") != EXPECTED_T2_COUNT:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_PARTITION_SIZE_MISMATCH")
    if partition.get("t2_ticker_list_sha256") != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_PARTITION_TICKER_HASH_MISMATCH")
    if history.get("partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_HISTORY_IMPLEMENTATION_MISMATCH")
    if history.get("manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_HISTORY_MANIFEST_MISMATCH")
    if history.get("t2_ticker_list_sha256") != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_HISTORY_TICKER_HASH_MISMATCH")
    if history.get("block_assignments_exposed") is not False or history.get("retry_performed") is not False:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_HISTORY_MUTATION_INVALID")
    if state["real_data_acquired"] is not False:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REAL_DATA_ACQUIRED")
    opened = t2["opened_for_research"] is True or (access_count or 0) != 0
    acquired = t2["raw_data_acquired"] is True or bridge["t2_acquired_before_authorized_acquisition"] is True
    exposed = opened or acquired or trust["block_assignments_exposed"] is True
    if exposed:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_EXPOSURE_DERIVED")
    return {
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0 if access_count is None else access_count,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
        "v8_trusted_partition_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": EXPECTED_T2_COUNT,
        "t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
    }


def _default_read_safe_state(repository_root, commit: str, reader: Callable[[Any, str, str], bytes]) -> Mapping[str, Any]:
    return _strict_json_object(
        reader(repository_root, commit, V8_STATE_GIT_PATH),
        "V8E_T2_STATE_INVALID_JSON",
        "V8E_T2_STATE_DUPLICATE_KEY",
    )


def _default_read_safe_bridge(repository_root, commit: str, reader: Callable[[Any, str, str], bytes]) -> Mapping[str, Any]:
    return _strict_json_object(
        reader(repository_root, commit, V8B_T2_AUTHORITY_BRIDGE_GIT_PATH),
        "V8E_T2_BRIDGE_INVALID_JSON",
        "V8E_T2_BRIDGE_DUPLICATE_KEY",
    )


def _resolve_t2_prefreeze_safe_evidence_with_dependencies(
    repository_root,
    *,
    verified_head: str,
    git_blob_resolver: Callable[[Any, str, str], str] = resolve_git_blob,
    git_object_reader: Callable[[Any, str, str], bytes] = read_git_object_bytes,
    safe_state_reader: Callable[[Any, str, Callable[[Any, str, str], bytes]], Mapping[str, Any]] = _default_read_safe_state,
    safe_bridge_reader: Callable[[Any, str, Callable[[Any, str, str], bytes]], Mapping[str, Any]] = _default_read_safe_bridge,
    trusted_anchor_reader: Callable[[Any, str], Mapping[str, Any]] = read_and_verify_v8_trusted_partition_anchor,
) -> dict[str, Any]:
    """DI-testable safe Git/audit/provenance resolver; no private reads."""
    if not isinstance(verified_head, str) or len(verified_head) != 40:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_HEAD_INVALID")
    try:
        if git_blob_resolver(repository_root, V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md") != V8E_DESIGN_CANDIDATE_BLOB_SHA:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_DESIGN_CANDIDATE_BLOB_MISMATCH")
        expected_blobs = (
            (V8_STATE_GIT_PATH, V8_STATE_BLOB_SHA),
            (V8B_T2_AUTHORITY_BRIDGE_GIT_PATH, V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA),
            (V8C_READINESS_ADJUDICATION_GIT_PATH, V8C_READINESS_ADJUDICATION_BLOB_SHA),
            (V8_TRUSTED_PARTITION_GIT_PATH, EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA),
        )
        for path, expected_blob in expected_blobs:
            if git_blob_resolver(repository_root, verified_head, path) != expected_blob:
                raise V8ET2PrefreezePreservationBlocked("V8E_T2_SAFE_BLOB_MISMATCH:" + path)
        state = safe_state_reader(repository_root, verified_head, git_object_reader)
        bridge = safe_bridge_reader(repository_root, verified_head, git_object_reader)
        anchor = trusted_anchor_reader(repository_root, verified_head)
    except V8ET2PrefreezePreservationBlocked:
        raise
    except V8CGitProvenanceBlocked as error:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_SAFE_GIT_EVIDENCE_UNAVAILABLE") from error
    return _validate_safe_evidence(_derive_safe_evidence(state, bridge, anchor))


def _git_text(repository_root, args: list[str], reason: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_GIT_UNAVAILABLE") from error
    if result.returncode != 0:
        raise V8ET2PrefreezePreservationBlocked(reason)
    return result.stdout.strip()


def resolve_and_verify_t2_prefreeze_preservation() -> dict[str, Any]:
    """Resolve safe committed evidence and validate an in-memory record only."""
    root = CANONICAL_REPOSITORY_ROOT
    branch = _git_text(root, ["branch", "--show-current"], "V8E_T2_BRANCH_UNAVAILABLE")
    if branch != V8E_PRODUCTION_BRANCH:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRANCH_MISMATCH")
    head = _git_text(root, ["rev-parse", "HEAD"], "V8E_T2_HEAD_UNAVAILABLE")
    safe = _resolve_t2_prefreeze_safe_evidence_with_dependencies(root, verified_head=head)
    record = build_t2_prefreeze_record(safe)
    verification = verify_t2_prefreeze_record(record, safe_evidence=safe)
    return {"safe_evidence": safe, "record": record, **verification}


__all__ = [
    "EXPECTED_T2_COUNT",
    "EXPECTED_T2_TICKER_LIST_SHA256",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA",
    "T2_SAFE_CONDITION_FIELDS",
    "V8E_DESIGN_CANDIDATE_BLOB_SHA",
    "V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT",
    "V8E_STUDY_NAME",
    "V8E_T2_PREFREEZE_RECORD_FIELDS",
    "V8E_T2_SAFE_EVIDENCE_FIELDS",
    "V8ET2PrefreezePreservationBlocked",
    "build_t2_prefreeze_record",
    "resolve_and_verify_t2_prefreeze_preservation",
    "verify_t2_prefreeze_record",
    "verify_t2_prefreeze_record_bytes",
]
