"""Safe V8E T2 pre-freeze preservation audit/review support.

Only committed public Git objects and safe provenance mappings enter this
module.  There is no private-manifest path, ticker-identity reader, gate
consumer, artifact writer, acquisition authority, or research-opening path.
The exact V8E T2 record is produced in memory for synthetic verification
only; the committed checkpoint is a later separately authorized stage.
"""

from __future__ import annotations

import json
import re
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
V8D_TERMINAL_COMMIT = "b8f8d0d500d349ccaa5d3e49294f351dc53ea7e8"
V8D_TERMINAL_GIT_PATH = "V8D_DQ_EVIDENCE_CONTRACT_BLOCK_ADJUDICATION.md"
V8D_TERMINAL_BLOB_SHA = "f81106f529c339e6762e60d3075e03e790335610"
V8D_T2_PREFREEZE_GIT_PATH = "V8D_T2_PREFREEZE_PRESERVATION_RECHECK.md"
V8D_T2_PREFREEZE_BLOB_SHA = "d023913b435ffd18eadef1e213c7ea43a49db331"

V8E_EXPECTED_PREFREEZE_CHRONOLOGY_PATHS = frozenset(
    {
        "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md",
        "src/v8e_t1c_preservation.py",
        "tests/test_v8e_t1c_preservation.py",
        "src/v8e_t2_prefreeze_preservation.py",
        "tests/test_v8e_t2_prefreeze_preservation.py",
    }
)
V8E_ORIGIN_URLS = frozenset(
    {
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git",
    }
)

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


def _parse_first_text_block(raw: bytes, reason: str) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8ET2PrefreezePreservationBlocked(reason) from error
    blocks = re.findall(r"```text\r?\n(.*?)\r?\n```", text, flags=re.DOTALL)
    if not blocks:
        raise V8ET2PrefreezePreservationBlocked(reason)
    result: dict[str, str] = {}
    for line in blocks[0].splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if not key or key in result:
            raise V8ET2PrefreezePreservationBlocked(reason)
        result[key] = value
    return result


_V8D_T2_HISTORICAL_FIELDS = frozenset(
    field if field != "reviewed_v8e_design_candidate_commit" else "reviewed_design_candidate_commit"
    for field in V8E_T2_PREFREEZE_RECORD_FIELDS
)


def _validate_v8d_terminal(raw: bytes) -> dict[str, Any]:
    values = _parse_first_text_block(raw, "V8E_T2_V8D_TERMINAL_INVALID")
    expected = {
        "study": "V8D_HISTORICAL_RESEARCH",
        "terminal_status": "BLOCK_CLOSED",
        "failure_class": "DESIGN_AUDITABILITY_FAILURE",
        "terminal_implementation_head": "a862efec34dcf4a89005c88b55b35c39be12b7bc",
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_V8D_TERMINAL_MISMATCH")
    if "No T1C/T2 outcomes or features were observed." not in raw.decode("utf-8"):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_V8D_TERMINAL_ABSENCE_INVALID")
    return {**expected, "t2_features_observed": False, "t2_outcomes_observed": False}


def _validate_v8d_t2_historical_record(raw: bytes) -> dict[str, Any]:
    values = _parse_first_text_block(raw, "V8E_T2_V8D_HISTORICAL_RECORD_INVALID")
    if set(values) != _V8D_T2_HISTORICAL_FIELDS:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_V8D_HISTORICAL_SCHEMA_INVALID")
    expected = {
        "study": "V8D_HISTORICAL_RESEARCH",
        "document_type": V8E_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_design_candidate_commit": "eda657cde2383718d986c4c4bfaae794784fe04d",
        "checkpoint": "V8D_T2_PREFREEZE_PRESERVATION_RECHECK",
        "recheck_1": "before_V8D_design_freeze",
        "T2_real_data_acquired": "false",
        "T2_opened": "false",
        "T2_research_access_count": "0",
        "T2_features_observed": "false",
        "T2_outcomes_observed": "false",
        "T2_membership_reassigned": "false",
        "universe_definition_compatible": "true",
        "partition_algorithm_compatible": "true",
        "data_quality_policy_unchanged": "true",
        "v8_trusted_partition_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": str(EXPECTED_T2_COUNT),
        "t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }
    if values != expected:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_V8D_HISTORICAL_VALUE_MISMATCH")
    return {
        **values,
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
        "t2_count": EXPECTED_T2_COUNT,
    }


_V8B_T2_BRIDGE_FIELDS = frozenset(
    {
        "schema_version", "study", "role", "source_authority", "v8_trust_anchor_git_path",
        "v8_trust_anchor_git_identity", "authorized_parent_v8_partition_manifest_sha256",
        "expected_t2_ticker_list_sha256", "t2_acquired_before_authorized_acquisition",
        "t2_research_open_count_before_official_opening", "v8b_frozen_design_commit",
        "t2_membership_reassignment", "v8_trusted_partition_json_mutated_or_repinned", "option",
        "human_gate", "authorization_note",
    }
)


def _validate_v8b_bridge(bridge: Mapping[str, Any]) -> dict[str, Any]:
    if set(bridge) != _V8B_T2_BRIDGE_FIELDS:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_SCHEMA_INVALID")
    expected = {
        "schema_version": "V8B_T2_AUTHORITY_BRIDGE_V1",
        "study": "V8B_HISTORICAL_RESEARCH",
        "role": "SEALED_HOLDOUT",
        "source_authority": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY",
        "v8_trust_anchor_git_path": V8_TRUSTED_PARTITION_GIT_PATH,
        "v8_trust_anchor_git_identity": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "authorized_parent_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "expected_t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
        "t2_acquired_before_authorized_acquisition": False,
        "t2_research_open_count_before_official_opening": 0,
        "v8b_frozen_design_commit": "eedf198b93185b963b825170ed0be97e93f923b7",
        "t2_membership_reassignment": "PROHIBITED",
        "v8_trusted_partition_json_mutated_or_repinned": False,
        "option": "OPTION_2",
        "human_gate": "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED",
    }
    for key, value in expected.items():
        if type(bridge[key]) is not type(value) or bridge[key] != value:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_VALUE_MISMATCH:" + key)
    if not isinstance(bridge["authorization_note"], str) or "never mutates" not in bridge["authorization_note"]:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRIDGE_NOTE_INVALID")
    return dict(bridge)


def _validate_v8c_readiness(raw: bytes) -> dict[str, bool]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_READINESS_INVALID") from error
    required = (
        "T2_ACCESS=PROHIBITED",
        "T1C_RAW_ACQUISITION=PROHIBITED",
        "T1C_RESEARCH_OPENING=PROHIBITED",
        "successor_study_required=true",
        "readiness_result=BLOCK",
    )
    if any(value not in text for value in required):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_READINESS_EVIDENCE_INVALID")
    return {"t2_access_prohibited": True}


def _validate_anchor(anchor: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "study_name", "design_commit", "authorization_status",
        "authorized_partition_manifest_sha256", "authorized_partition_implementation_git_commit",
        "authorization_note",
    }
    if set(anchor) != required:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_ANCHOR_SCHEMA_INVALID")
    exact = {
        "schema_version": "V8_TRUSTED_PARTITION_V1",
        "study_name": "V8_HISTORICAL_RESEARCH",
        "design_commit": "c414d3191cba356734d7ed08bdf1abc7d51fc384",
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    }
    for key, value in exact.items():
        if anchor[key] != value:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_ANCHOR_VALUE_MISMATCH:" + key)
    if not isinstance(anchor["authorization_note"], str) or "does NOT authorize T2 acquisition" not in anchor["authorization_note"]:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_ANCHOR_NOTE_INVALID")
    return dict(anchor)


def _validate_v8e_design(raw: bytes) -> dict[str, bool]:
    required = (
        "policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE",
        "invalid_fraction_threshold=1/252",
        "max_consecutive_invalid_returned_rows=1",
        "full_P_hist_check=true",
        "threshold_failure_action=BLOCK_WHOLE_ACQUISITION",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_DESIGN_INVALID") from error
    if any(value not in text for value in required):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_DESIGN_POLICY_INVALID")
    return {"policy_unchanged": True}


def _derive_safe_evidence(
    state: Mapping[str, Any],
    bridge: Mapping[str, Any],
    anchor: Mapping[str, Any],
    *,
    historical_t2: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
    readiness: Mapping[str, Any] | None = None,
    design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive all nine conditions from independently validated safe facts."""
    state = _require_mapping(state, "V8E_T2_STATE_EVIDENCE_INVALID")
    bridge = _validate_v8b_bridge(_require_mapping(bridge, "V8E_T2_BRIDGE_EVIDENCE_INVALID"))
    anchor = _validate_anchor(_require_mapping(anchor, "V8E_T2_ANCHOR_EVIDENCE_INVALID"))
    if not all(isinstance(value, Mapping) for value in (historical_t2, terminal, readiness, design)):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REQUIRED_SAFE_EVIDENCE_MISSING")
    historical_t2 = dict(historical_t2)
    terminal = dict(terminal)
    if readiness.get("t2_access_prohibited") is not True or design.get("policy_unchanged") is not True:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REQUIRED_SAFE_EVIDENCE_INVALID")
    t2 = _require_mapping(state.get("T2"), "V8E_T2_STATE_SECTION_INVALID")
    trust = _require_mapping(state.get("trust_anchor_pinning"), "V8E_T2_STATE_TRUST_SECTION_INVALID")
    partition = _require_mapping(state.get("partition"), "V8E_T2_STATE_PARTITION_SECTION_INVALID")
    history_value = state.get("real_partition_build_history")
    if not isinstance(history_value, list):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_HISTORY_NOT_LIST")
    if len(history_value) != 1:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_HISTORY_ENTRY_COUNT_INVALID")
    history = _require_mapping(history_value[0], "V8E_T2_STATE_HISTORY_ENTRY_INVALID")
    history_expected = {
        "authorized_implementation_head": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "mode": "PRODUCTION",
        "process_result": "PASS",
        "exit_code": 0,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "partition_manifest_written": True,
        "real_block_assignments_created": True,
        "manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_git_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "manifest_schema_version": "V8_PARTITION_MANIFEST_V3",
        "t1_ticker_list_sha256": "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d",
        "t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
        "one_time_authorization_consumed": True,
        "retry_performed": False,
        "raw_jpx_bytes_persisted": False,
        "block_assignments_exposed": False,
    }
    for key, value in history_expected.items():
        if type(history.get(key)) is not type(value) or history.get(key) != value:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_HISTORY_VALUE_MISMATCH:" + key)
    block_sizes = history.get("block_sizes")
    if not isinstance(block_sizes, Mapping) or dict(block_sizes) != {"T0": 300, "T1": 300, "T2": 300, "T3": 300, "T_spare": 1904}:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_HISTORY_BLOCK_SIZES_INVALID")
    state_exact = {
        "schema_version": "V8_STATE_SNAPSHOT_V1",
        "study": "V8_HISTORICAL_RESEARCH",
        "real_data_acquired": False,
        "backtests": 0,
        "models_fitted": 0,
        "profit_calculated": 0,
    }
    for key, value in state_exact.items():
        if type(state.get(key)) is not type(value) or state.get(key) != value:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_VALUE_MISMATCH:" + key)
    t2_expected = {
        "raw_data_acquired": False,
        "opened_for_research": False,
        "real_acquisition_authorized": False,
        "sealed_holdout_access_count": None,
        "research_access_authorized": None,
        "ticker_count_frozen": EXPECTED_T2_COUNT,
    }
    for key, value in t2_expected.items():
        if type(t2.get(key)) is not type(value) or t2.get(key) != value:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_STATE_T2_VALUE_MISMATCH:" + key)
    if trust.get("block_assignments_exposed") is not False:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_ASSIGNMENTS_EXPOSED")
    partition_expected = {
        "manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_git_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "manifest_schema_version": "V8_PARTITION_MANIFEST_V3",
        "block_size_frozen": EXPECTED_T2_COUNT,
        "t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
        "block_assignments_recorded": False,
    }
    for key, value in partition_expected.items():
        if type(partition.get(key)) is not type(value) or partition.get(key) != value:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_PARTITION_VALUE_MISMATCH:" + key)
    acquired = (
        t2["raw_data_acquired"] is False
        and state["real_data_acquired"] is False
        and bridge["t2_acquired_before_authorized_acquisition"] is False
        and historical_t2["T2_real_data_acquired"] is False
    )
    opened = (
        t2["opened_for_research"] is False
        and bridge["t2_research_open_count_before_official_opening"] == 0
        and historical_t2["T2_opened"] is False
        and historical_t2["T2_research_access_count"] == 0
    )
    access_zero = opened and t2["sealed_holdout_access_count"] is None and historical_t2["T2_research_access_count"] == 0
    membership_unchanged = (
        historical_t2["T2_membership_reassigned"] is False
        and bridge["t2_membership_reassignment"] == "PROHIBITED"
        and partition["t2_ticker_list_sha256"] == EXPECTED_T2_TICKER_LIST_SHA256
        and history["t2_ticker_list_sha256"] == EXPECTED_T2_TICKER_LIST_SHA256
    )
    if not acquired or not opened or not access_zero or not membership_unchanged:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_PRESERVATION_CONDITION_DERIVATION_BLOCKED")
    no_research = acquired and opened and state["backtests"] == 0 and state["models_fitted"] == 0 and state["profit_calculated"] == 0
    no_features = no_research and historical_t2["T2_features_observed"] is False and terminal["t2_features_observed"] is False
    no_outcomes = no_research and historical_t2["T2_outcomes_observed"] is False and terminal["t2_outcomes_observed"] is False
    universe = (
        anchor["authorized_partition_manifest_sha256"] == EXPECTED_V8_PARTITION_MANIFEST_SHA256
        and anchor["authorized_partition_implementation_git_commit"] == EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
        and bridge["authorized_parent_v8_partition_manifest_sha256"] == EXPECTED_V8_PARTITION_MANIFEST_SHA256
        and bridge["expected_t2_ticker_list_sha256"] == EXPECTED_T2_TICKER_LIST_SHA256
        and historical_t2["universe_definition_compatible"] is True
    )
    algorithm = (
        history["partition_implementation_git_commit"] == EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
        and partition["partition_implementation_git_commit"] == EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
        and historical_t2["partition_algorithm_compatible"] is True
    )
    policy = design["policy_unchanged"] and historical_t2["data_quality_policy_unchanged"] is True
    return _validate_safe_evidence(
        {
            "T2_real_data_acquired": not acquired,
            "T2_opened": not opened,
            "T2_research_access_count": 0 if access_zero else 1,
            "T2_features_observed": not no_features,
            "T2_outcomes_observed": not no_outcomes,
            "T2_membership_reassigned": not membership_unchanged,
            "universe_definition_compatible": universe,
            "partition_algorithm_compatible": algorithm,
            "data_quality_policy_unchanged": policy,
            "v8_trusted_partition_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
            "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
            "t2_count": EXPECTED_T2_COUNT,
            "t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
        }
    )


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
    reviewed_support_implementation_sha: str,
    git_blob_resolver: Callable[[Any, str, str], str] = resolve_git_blob,
    git_object_reader: Callable[[Any, str, str], bytes] = read_git_object_bytes,
    safe_state_reader: Callable[[Any, str, Callable[[Any, str, str], bytes]], Mapping[str, Any]] = _default_read_safe_state,
    safe_bridge_reader: Callable[[Any, str, Callable[[Any, str, str], bytes]], Mapping[str, Any]] = _default_read_safe_bridge,
    trusted_anchor_reader: Callable[[Any, str], Mapping[str, Any]] = read_and_verify_v8_trusted_partition_anchor,
    runtime_state_reader: Callable[[Any, str], Mapping[str, Any]] | None = None,
    chronology_reader: Callable[[Any, str, str], list[Mapping[str, Any]]] | None = None,
    commit_ancestor_reader: Callable[[Any, str, str], bool] | None = None,
) -> dict[str, Any]:
    """DI-testable safe Git/audit/provenance resolver; no private reads."""
    reviewed_sha = _require_hex(
        reviewed_support_implementation_sha, 40, "V8E_T2_REVIEWED_SUPPORT_SHA_INVALID"
    )
    if _require_hex(verified_head, 40, "V8E_T2_HEAD_INVALID") != reviewed_sha:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REVIEWED_SUPPORT_HEAD_MISMATCH")
    _validate_reviewed_support_runtime(
        repository_root,
        reviewed_sha,
        verified_head=verified_head,
        runtime_state_reader=runtime_state_reader,
    )
    try:
        if git_blob_resolver(repository_root, V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md") != V8E_DESIGN_CANDIDATE_BLOB_SHA:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_DESIGN_CANDIDATE_BLOB_MISMATCH")
        if git_blob_resolver(repository_root, reviewed_sha, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md") != V8E_DESIGN_CANDIDATE_BLOB_SHA:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CURRENT_DESIGN_BLOB_MISMATCH")
        expected_blobs = (
            (V8_STATE_GIT_PATH, V8_STATE_BLOB_SHA),
            (V8B_T2_AUTHORITY_BRIDGE_GIT_PATH, V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA),
            (V8C_READINESS_ADJUDICATION_GIT_PATH, V8C_READINESS_ADJUDICATION_BLOB_SHA),
            (V8_TRUSTED_PARTITION_GIT_PATH, EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA),
        )
        for path, expected_blob in expected_blobs:
            if git_blob_resolver(repository_root, verified_head, path) != expected_blob:
                raise V8ET2PrefreezePreservationBlocked("V8E_T2_SAFE_BLOB_MISMATCH:" + path)
        if git_blob_resolver(repository_root, V8D_TERMINAL_COMMIT, V8D_TERMINAL_GIT_PATH) != V8D_TERMINAL_BLOB_SHA:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_V8D_TERMINAL_BLOB_MISMATCH")
        if git_blob_resolver(repository_root, V8D_TERMINAL_COMMIT, V8D_T2_PREFREEZE_GIT_PATH) != V8D_T2_PREFREEZE_BLOB_SHA:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_V8D_HISTORICAL_BLOB_MISMATCH")
        state = safe_state_reader(repository_root, verified_head, git_object_reader)
        bridge = safe_bridge_reader(repository_root, verified_head, git_object_reader)
        anchor = trusted_anchor_reader(repository_root, verified_head)
        historical_t2 = _validate_v8d_t2_historical_record(
            _strict_text_bytes(
                git_object_reader(repository_root, V8D_TERMINAL_COMMIT, V8D_T2_PREFREEZE_GIT_PATH),
                "V8E_T2_V8D_HISTORICAL_RECORD_INVALID",
            )
        )
        terminal = _validate_v8d_terminal(
            _strict_text_bytes(
                git_object_reader(repository_root, V8D_TERMINAL_COMMIT, V8D_TERMINAL_GIT_PATH),
                "V8E_T2_V8D_TERMINAL_INVALID",
            )
        )
        readiness = _validate_v8c_readiness(
            _strict_text_bytes(
                git_object_reader(repository_root, verified_head, V8C_READINESS_ADJUDICATION_GIT_PATH),
                "V8E_T2_READINESS_INVALID",
            )
        )
        design = _validate_v8e_design(
            _strict_text_bytes(
                git_object_reader(repository_root, reviewed_sha, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md"),
                "V8E_T2_DESIGN_INVALID",
            )
        )
        chronology = _validate_committed_chronology(
            (chronology_reader or _default_public_chronology)(
                repository_root, V8D_TERMINAL_COMMIT, reviewed_sha
            ),
            repository_root=repository_root,
            reviewed_support_sha=reviewed_sha,
            reviewed_design_candidate=V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
            commit_ancestor_reader=commit_ancestor_reader,
        )
    except V8ET2PrefreezePreservationBlocked:
        raise
    except V8CGitProvenanceBlocked as error:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_SAFE_GIT_EVIDENCE_UNAVAILABLE") from error
    safe = _derive_safe_evidence(
        state,
        bridge,
        anchor,
        historical_t2=historical_t2,
        terminal=terminal,
        readiness=readiness,
        design=design,
    )
    if chronology is None:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_INVALID")
    return _validate_safe_evidence(safe)


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


_REVIEWED_SUPPORT_RUNTIME_FIELDS = frozenset(
    {
        "resolved_support_sha",
        "branch",
        "head",
        "origin_head",
        "worktree_clean",
        "origin_url",
        "commits_after_reviewed_support_sha",
    }
)


def _strict_text_bytes(raw: bytes, reason: str) -> bytes:
    if not isinstance(raw, bytes):
        raise V8ET2PrefreezePreservationBlocked(reason)
    return raw


def _default_reviewed_support_runtime_state(repository_root, reviewed_sha: str) -> dict[str, Any]:
    resolved = _git_text(
        repository_root,
        ["rev-parse", "--verify", f"{reviewed_sha}^{{commit}}"],
        "V8E_T2_REVIEWED_SUPPORT_SHA_UNRESOLVABLE",
    )
    count_text = _git_text(
        repository_root,
        ["rev-list", "--count", f"{reviewed_sha}..HEAD"],
        "V8E_T2_REVIEWED_SUPPORT_CHRONOLOGY_INVALID",
    )
    if not count_text.isdecimal():
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REVIEWED_SUPPORT_CHRONOLOGY_INVALID")
    return {
        "resolved_support_sha": resolved,
        "branch": _git_text(repository_root, ["branch", "--show-current"], "V8E_T2_BRANCH_UNAVAILABLE"),
        "head": _git_text(repository_root, ["rev-parse", "HEAD"], "V8E_T2_HEAD_UNAVAILABLE"),
        "origin_head": _git_text(
            repository_root,
            ["rev-parse", "origin/" + V8E_PRODUCTION_BRANCH],
            "V8E_T2_ORIGIN_UNAVAILABLE",
        ),
        "worktree_clean": _git_text(repository_root, ["status", "--porcelain"], "V8E_T2_GIT_UNAVAILABLE") == "",
        "origin_url": _git_text(repository_root, ["config", "--get", "remote.origin.url"], "V8E_T2_ORIGIN_UNAVAILABLE"),
        "commits_after_reviewed_support_sha": int(count_text),
    }


def _validate_reviewed_support_runtime(
    repository_root,
    reviewed_sha: str,
    *,
    verified_head: str,
    runtime_state_reader: Callable[[Any, str], Mapping[str, Any]] | None = None,
) -> None:
    runtime = (runtime_state_reader or _default_reviewed_support_runtime_state)(repository_root, reviewed_sha)
    if not isinstance(runtime, Mapping) or set(runtime) != _REVIEWED_SUPPORT_RUNTIME_FIELDS:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REVIEWED_SUPPORT_RUNTIME_SCHEMA_INVALID")
    if (
        runtime["resolved_support_sha"] != reviewed_sha
        or runtime["branch"] != V8E_PRODUCTION_BRANCH
        or runtime["head"] != reviewed_sha
        or runtime["origin_head"] != reviewed_sha
        or runtime["worktree_clean"] is not True
        or runtime["origin_url"] not in V8E_ORIGIN_URLS
        or type(runtime["commits_after_reviewed_support_sha"]) is not int
        or runtime["commits_after_reviewed_support_sha"] != 0
        or verified_head != reviewed_sha
    ):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REVIEWED_SUPPORT_RUNTIME_BINDING_INVALID")


def _default_public_chronology(repository_root, lower: str, upper: str) -> list[dict[str, Any]]:
    if _git_text(repository_root, ["merge-base", "--is-ancestor", lower, upper], "V8E_T2_CHRONOLOGY_INVALID") != "":
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_INVALID")
    commits = _git_text(
        repository_root,
        ["rev-list", "--reverse", f"{lower}..{upper}"],
        "V8E_T2_CHRONOLOGY_INVALID",
    ).splitlines()
    records = []
    for commit in commits:
        paths = _git_text(
            repository_root,
            ["diff-tree", "--no-commit-id", "--name-only", "-r", commit],
            "V8E_T2_CHRONOLOGY_INVALID",
        ).splitlines()
        if not paths:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_INVALID")
        records.append({"commit": commit, "paths": paths})
    return records


def _default_commit_ancestor(repository_root, commit: str, candidate: str) -> bool:
    return _git_text(
        repository_root,
        ["merge-base", "--is-ancestor", commit, candidate],
        "V8E_T2_CHRONOLOGY_UNVERIFIABLE",
    ) == ""


def _validate_committed_chronology(
    records: Any,
    *,
    repository_root,
    reviewed_support_sha: str | None = None,
    reviewed_design_candidate: str = V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    commit_ancestor_reader: Callable[[Any, str, str], bool] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_INVALID")
    union: set[str] = set()
    validated: list[dict[str, Any]] = []
    ancestor_reader = commit_ancestor_reader or _default_commit_ancestor
    if reviewed_support_sha is not None and not ancestor_reader(
        repository_root, reviewed_design_candidate, reviewed_support_sha
    ):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_REVIEWED_DESIGN_CANDIDATE_NOT_ANCESTOR")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"commit", "paths"}:
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_UNVERIFIABLE")
        _require_hex(record["commit"], 40, "V8E_T2_CHRONOLOGY_UNVERIFIABLE")
        paths = record["paths"]
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths):
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_UNVERIFIABLE")
        if len(set(paths)) != len(paths) or any(path not in V8E_EXPECTED_PREFREEZE_CHRONOLOGY_PATHS for path in paths):
            raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_UNEXPECTED_PATH")
        union.update(paths)
        validated.append({"commit": record["commit"], "paths": list(paths)})
    if union != set(V8E_EXPECTED_PREFREEZE_CHRONOLOGY_PATHS):
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_CHRONOLOGY_PATH_UNION_INVALID")
    for record in validated:
        if "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md" in record["paths"]:
            if not ancestor_reader(
                repository_root,
                record["commit"],
                reviewed_design_candidate,
            ):
                raise V8ET2PrefreezePreservationBlocked("V8E_T2_POST_CANDIDATE_DESIGN_CHANGE")
    return validated


def resolve_and_verify_t2_prefreeze_preservation(
    *, reviewed_support_implementation_sha: str
) -> dict[str, Any]:
    """Resolve safe committed evidence and validate an in-memory record only."""
    root = CANONICAL_REPOSITORY_ROOT
    branch = _git_text(root, ["branch", "--show-current"], "V8E_T2_BRANCH_UNAVAILABLE")
    if branch != V8E_PRODUCTION_BRANCH:
        raise V8ET2PrefreezePreservationBlocked("V8E_T2_BRANCH_MISMATCH")
    head = _git_text(root, ["rev-parse", "HEAD"], "V8E_T2_HEAD_UNAVAILABLE")
    safe = _resolve_t2_prefreeze_safe_evidence_with_dependencies(
        root,
        verified_head=head,
        reviewed_support_implementation_sha=reviewed_support_implementation_sha,
    )
    record = build_t2_prefreeze_record(safe)
    verification = verify_t2_prefreeze_record(record, safe_evidence=safe)
    return {"safe_evidence": safe, "record": record, **verification}


__all__ = [
    "EXPECTED_T2_COUNT",
    "EXPECTED_T2_TICKER_LIST_SHA256",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA",
    "V8D_TERMINAL_BLOB_SHA",
    "V8D_TERMINAL_COMMIT",
    "V8D_T2_PREFREEZE_BLOB_SHA",
    "T2_SAFE_CONDITION_FIELDS",
    "V8E_EXPECTED_PREFREEZE_CHRONOLOGY_PATHS",
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
