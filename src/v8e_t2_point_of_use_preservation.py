"""V8E T2 point-of-use preservation contract.

This module defines the safe, committed/audit/provenance-only checkpoint
immediately before the later T2 acquisition preflight.  It never reads the
private partition manifest, never exposes T2 identities, never consumes a
human gate, and never writes either future point-of-use artifact.

The public production functions have no authority or evidence override
parameters.  The underscore-prefixed dependency-injected functions exist
only for synthetic tests of the fail-closed contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8e_authority_bridge import (
    T2_STAGE,
    V8EAuthorityBridgeBlocked,
    verify_stage_authority_bridge,
)
from src.v8e_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8EGitProvenanceBlocked,
    read_git_object_bytes,
    require_strict_git_ancestor,
    resolve_git_blob,
    resolve_verified_v8e_production_git_commit,
)
from src.v8e_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T2_RAW_ACQUISITION,
    V8EHumanGateConsumptionBlocked,
    has_gate_been_consumed,
)
from src.v8e_production_provenance import (
    EXPECTED_V8E_DESIGN_FREEZE_APPROVAL_BLOB,
    EXPECTED_V8E_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    STUDY_NAME,
    V8EProductionProvenanceBlocked,
    read_and_verify_v8_trusted_partition_anchor,
    verify_design_freeze_approval_blob,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8e_readiness_audit_verification import require_t2_readiness_audit_verification_pass


STUDY = STUDY_NAME
FROZEN_DESIGN_COMMIT = EXPECTED_V8E_FROZEN_DESIGN_COMMIT
POINT_OF_USE_ARTIFACT_PATH = "V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK.json"
POINT_OF_USE_REVIEW_PATH = "V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW.json"
POINT_OF_USE_SCHEMA_VERSION = "V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK_V1"
POINT_OF_USE_ARTIFACT_ROLE = "T2_POINT_OF_USE_PRESERVATION_RECHECK"
POINT_OF_USE_CHECKPOINT = "READ_ONLY_V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK"
POINT_OF_USE_RECHECK = "immediately_before_T2_acquisition"
READINESS_VERIFICATION_STAGE = "READ_ONLY_T2_READINESS_TRANSPORT_AUDIT_VERIFICATION"
READINESS_LOGICAL_STAGE = "T2_TRANSPORT_READINESS"
POINT_OF_USE_REVIEW_SCHEMA_VERSION = "V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW_V1"
POINT_OF_USE_REVIEW_ROLE = "T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW"
POINT_OF_USE_REVIEW_CHECKPOINT = "INDEPENDENT_V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW"

PREFREEZE_PRESERVATION_COMMIT = "22071e3fceaff56ac2043f79e2d79d617f3658a5"
PREFREEZE_PRESERVATION_BLOB = "24248bf96877ffb47bdba8fac7924684b1cae5cb"
PREFREEZE_PRESERVATION_PATH = "V8E_T2_PREFREEZE_PRESERVATION_RECHECK.md"
V8_STATE_PATH = "V8_STATE.json"

T2_COUNT = 300
T2_TICKER_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"

NINE_CONDITION_FIELDS = (
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

POINT_OF_USE_ARTIFACT_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "checkpoint",
    "recheck",
    "v8e_frozen_design_commit",
    "source_prefreeze_preservation_git_commit",
    "source_prefreeze_preservation_git_blob_sha",
    "readiness_verification_stage",
    "readiness_verification_result",
    "readiness_receipt_self_hash",
    "v8_trust_anchor_git_blob",
    "original_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "t2_count",
    "t2_ticker_list_sha256",
    *NINE_CONDITION_FIELDS,
    "t2_raw_acquisition_gate_consumed",
    "point_of_use_preservation_result",
)

POINT_OF_USE_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "checkpoint",
    "v8e_frozen_design_commit",
    "reviewed_recheck_git_commit",
    "reviewed_recheck_git_blob_sha",
    "review_result",
)


class V8ET2PointOfUsePreservationBlocked(RuntimeError):
    """Fail-closed point-of-use preservation contract error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8ET2PointOfUsePreservationBlocked(duplicate_reason)
            result[key] = value

        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8ET2PointOfUsePreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8ET2PointOfUsePreservationBlocked(invalid_reason)
    return parsed


def _require_exact_fields(value: Mapping[str, Any], fields: tuple[str, ...], reason: str) -> None:
    if set(value) != set(fields):
        raise V8ET2PointOfUsePreservationBlocked(reason)


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise V8ET2PointOfUsePreservationBlocked(reason)
    return value


def _require_bool(value: object, reason: str) -> bool:
    if type(value) is not bool:
        raise V8ET2PointOfUsePreservationBlocked(reason)
    return value


def _require_safe_conditions(conditions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(conditions, Mapping) or set(conditions) != set(NINE_CONDITION_FIELDS):
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_CONDITIONS_SCHEMA_INVALID")
    for field in (
        "T2_real_data_acquired",
        "T2_opened",
        "T2_features_observed",
        "T2_outcomes_observed",
        "T2_membership_reassigned",
    ):
        if _require_bool(conditions[field], "V8E_T2_POINT_OF_USE_CONDITION_TYPE_INVALID") is not False:
            raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_CONDITION_INVALID:" + field)
    access_count = conditions["T2_research_access_count"]
    if type(access_count) is not int or access_count != 0:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_research_access_count")
    for field in (
        "universe_definition_compatible",
        "partition_algorithm_compatible",
        "data_quality_policy_unchanged",
    ):
        if _require_bool(conditions[field], "V8E_T2_POINT_OF_USE_CONDITION_TYPE_INVALID") is not True:
            raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_CONDITION_INVALID:" + field)
    return dict(conditions)


def _require_state_mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V8ET2PointOfUsePreservationBlocked(reason)
    return value


def _require_state_bool(state: Mapping[str, Any], field: str, expected: bool, reason: str) -> None:
    if _require_bool(state.get(field), reason) is not expected:
        raise V8ET2PointOfUsePreservationBlocked(reason)


def _require_state_int(state: Mapping[str, Any], field: str, expected: int, reason: str) -> None:
    value = state.get(field)
    if type(value) is not int or value != expected:
        raise V8ET2PointOfUsePreservationBlocked(reason)


def _require_state_sha256(state: Mapping[str, Any], field: str, expected: str, reason: str) -> None:
    value = state.get(field)
    if value != expected or not isinstance(value, str) or len(value) != 64:
        raise V8ET2PointOfUsePreservationBlocked(reason)


def _require_state_commit(state: Mapping[str, Any], field: str, expected: str, reason: str) -> None:
    value = state.get(field)
    if value != expected or not isinstance(value, str) or len(value) != 40:
        raise V8ET2PointOfUsePreservationBlocked(reason)


def _derive_conditions_from_v8_state(
    repository_root: Path,
    verified_head: str,
    git_object_reader: Callable[[Path, str, str], bytes],
) -> dict[str, Any]:
    try:
        state = _strict_json_object(
            git_object_reader(repository_root, verified_head, V8_STATE_PATH),
            invalid_reason="V8E_T2_POINT_OF_USE_V8_STATE_INVALID_JSON",
            duplicate_reason="V8E_T2_POINT_OF_USE_V8_STATE_DUPLICATE_KEY",
        )
    except V8EGitProvenanceBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_V8_STATE_MISSING") from error

    t2 = _require_state_mapping(
        state.get("T2"), "V8E_T2_POINT_OF_USE_V8_STATE_T2_EVIDENCE_INVALID",
    )
    partition_history_records = state.get("real_partition_build_history")
    if not isinstance(partition_history_records, list) or len(partition_history_records) != 1:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_INVALID",
        )
    partition_history = _require_state_mapping(
        partition_history_records[0],
        "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_INVALID",
    )
    partition = _require_state_mapping(
        state.get("partition"), "V8E_T2_POINT_OF_USE_PARTITION_STATE_INVALID",
    )
    trust_anchor_pinning = _require_state_mapping(
        state.get("trust_anchor_pinning"),
        "V8E_T2_POINT_OF_USE_TRUST_PINNING_INVALID",
    )

    # T2 uses null as the committed unopened representation.  A later
    # explicitly recorded integer zero is also safe, but every other value
    # is contradictory evidence.
    _require_state_bool(
        t2, "raw_data_acquired", False,
        "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_real_data_acquired",
    )
    _require_state_bool(
        t2, "opened_for_research", False,
        "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_opened",
    )
    access_count = t2.get("sealed_holdout_access_count")
    if access_count is not None and (type(access_count) is not int or access_count != 0):
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_research_access_count",
        )
    _require_state_bool(
        t2, "real_acquisition_authorized", False,
        "V8E_T2_POINT_OF_USE_T2_ACQUISITION_AUTHORIZATION_INVALID",
    )
    if t2.get("research_access_authorized") not in (None, False):
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_T2_RESEARCH_AUTHORIZATION_INVALID",
        )
    _require_state_int(
        t2, "ticker_count_frozen", T2_COUNT,
        "V8E_T2_POINT_OF_USE_T2_COUNT_INVALID",
    )

    # The history is the committed record of the one authoritative
    # production partition build.  Require its complete current schema so a
    # missing/ambiguous alternate build cannot silently become the source of
    # the frozen compatibility conclusions.
    expected_history_fields = {
        "authorized_implementation_head",
        "mode",
        "process_result",
        "exit_code",
        "source_reproduction_status",
        "t0_reproduction_status",
        "partition_manifest_written",
        "real_block_assignments_created",
        "real_jpx_requests_this_attempt",
        "real_yahoo_requests_this_attempt",
        "manifest_sha256",
        "partition_implementation_git_commit",
        "manifest_schema_version",
        "block_sizes",
        "t1_ticker_list_sha256",
        "t2_ticker_list_sha256",
        "t3_ticker_list_sha256",
        "t_spare_ticker_list_sha256",
        "one_time_authorization_consumed",
        "retry_performed",
        "raw_jpx_bytes_persisted",
        "block_assignments_exposed",
    }
    if set(partition_history) != expected_history_fields:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_SCHEMA_INVALID",
        )
    if partition_history.get("mode") != "PRODUCTION":
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_MODE_INVALID",
        )
    if partition_history.get("process_result") != "PASS" or partition_history.get("exit_code") != 0:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_RESULT_INVALID",
        )
    if partition_history.get("source_reproduction_status") != "PASS" or partition_history.get("t0_reproduction_status") != "PASS":
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_REPRODUCTION_INVALID",
        )
    _require_state_bool(
        partition_history, "partition_manifest_written", True,
        "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_MANIFEST_INVALID",
    )
    _require_state_bool(
        partition_history, "real_block_assignments_created", True,
        "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_ASSIGNMENTS_INVALID",
    )
    _require_state_bool(
        partition_history, "one_time_authorization_consumed", True,
        "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_AUTHORIZATION_INVALID",
    )
    _require_state_bool(
        partition_history, "retry_performed", False,
        "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_RETRY_INVALID",
    )
    _require_state_bool(
        partition_history, "raw_jpx_bytes_persisted", False,
        "V8E_T2_POINT_OF_USE_PARTITION_HISTORY_RAW_BYTES_INVALID",
    )
    _require_state_bool(
        partition_history, "block_assignments_exposed", False,
        "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_features_observed",
    )
    _require_state_sha256(
        partition_history, "manifest_sha256", EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "V8E_T2_POINT_OF_USE_MANIFEST_SHA_MISMATCH",
    )
    _require_state_commit(
        partition_history, "partition_implementation_git_commit", EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "V8E_T2_POINT_OF_USE_PARTITION_IMPLEMENTATION_MISMATCH",
    )
    if partition_history.get("authorized_implementation_head") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_AUTHORIZED_IMPLEMENTATION_HEAD_MISMATCH",
        )
    if partition_history.get("manifest_schema_version") != "V8_PARTITION_MANIFEST_V3":
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_MANIFEST_SCHEMA_INVALID",
        )
    block_sizes = _require_state_mapping(
        partition_history.get("block_sizes"),
        "V8E_T2_POINT_OF_USE_PARTITION_BLOCK_SIZES_INVALID",
    )
    if set(block_sizes) != {"T0", "T1", "T2", "T3", "T_spare"}:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_PARTITION_BLOCK_SIZES_INVALID",
        )
    for block, size in block_sizes.items():
        if type(size) is not int or size <= 0:
            raise V8ET2PointOfUsePreservationBlocked(
                "V8E_T2_POINT_OF_USE_PARTITION_BLOCK_SIZES_INVALID:" + block,
            )
    if block_sizes["T2"] != T2_COUNT:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_T2_COUNT_INVALID",
        )
    _require_state_sha256(
        partition_history, "t2_ticker_list_sha256", T2_TICKER_LIST_SHA256,
        "V8E_T2_POINT_OF_USE_T2_TICKER_HASH_INVALID",
    )

    # The committed partition summary independently repeats the identity and
    # no-regeneration facts without exposing any block member.
    _require_state_bool(
        partition, "real_partition_manifest_exists", True,
        "V8E_T2_POINT_OF_USE_PARTITION_STATE_MANIFEST_INVALID",
    )
    _require_state_bool(
        partition, "real_partition_manifest_validated", True,
        "V8E_T2_POINT_OF_USE_PARTITION_STATE_VALIDATION_INVALID",
    )
    _require_state_bool(
        partition, "trusted_partition_authorized", True,
        "V8E_T2_POINT_OF_USE_PARTITION_STATE_AUTHORIZATION_INVALID",
    )
    _require_state_bool(
        partition, "real_partition_creation_authorization_consumed", True,
        "V8E_T2_POINT_OF_USE_PARTITION_STATE_AUTHORIZATION_INVALID",
    )
    _require_state_bool(
        partition, "block_assignments_recorded", False,
        "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_membership_reassigned",
    )
    _require_state_sha256(
        partition, "manifest_sha256", EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "V8E_T2_POINT_OF_USE_MANIFEST_SHA_MISMATCH",
    )
    _require_state_commit(
        partition, "partition_implementation_git_commit", EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "V8E_T2_POINT_OF_USE_PARTITION_IMPLEMENTATION_MISMATCH",
    )
    partition_block_sizes = _require_state_mapping(
        partition.get("block_sizes"),
        "V8E_T2_POINT_OF_USE_PARTITION_BLOCK_SIZES_INVALID",
    )
    if partition_block_sizes.get("T2") != T2_COUNT:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_T2_COUNT_INVALID",
        )
    if partition.get("t2_ticker_list_sha256") != T2_TICKER_LIST_SHA256:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_T2_TICKER_HASH_INVALID",
        )

    # This is the actual location of the trust-pin exposure field.  The
    # legacy trusted_partition_anchor_state object is intentionally not
    # queried for it.
    _require_state_sha256(
        trust_anchor_pinning, "authorized_partition_manifest_sha256",
        EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "V8E_T2_POINT_OF_USE_MANIFEST_SHA_MISMATCH",
    )
    _require_state_commit(
        trust_anchor_pinning, "authorized_partition_implementation_git_commit",
        EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "V8E_T2_POINT_OF_USE_PARTITION_IMPLEMENTATION_MISMATCH",
    )
    _require_state_bool(
        trust_anchor_pinning, "block_assignments_exposed", False,
        "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_features_observed",
    )
    _require_state_bool(
        trust_anchor_pinning, "t2_acquisition_authorized_by_this_pin", False,
        "V8E_T2_POINT_OF_USE_T2_ACQUISITION_AUTHORIZATION_INVALID",
    )
    _require_state_bool(
        trust_anchor_pinning, "one_time_authorization_consumed", True,
        "V8E_T2_POINT_OF_USE_TRUST_PINNING_AUTHORIZATION_INVALID",
    )

    if state.get("real_data_acquired") is not False or state.get("real_orders_allowed") is not False:
        raise V8ET2PointOfUsePreservationBlocked(
            "V8E_T2_POINT_OF_USE_CONDITION_INVALID:T2_real_data_acquired",
        )
    for counter in ("backtests", "models_fitted", "profit_calculated", "parameter_search"):
        _require_state_int(
            state, counter, 0,
            "V8E_T2_POINT_OF_USE_RESEARCH_COUNTER_INVALID:" + counter,
        )

    # These are existing committed design-policy facts.  They support the
    # frozen data-quality condition without inventing a new V8_STATE field or
    # reading any private/raw data.
    policy = _require_state_mapping(
        state.get("malformed_ohlcv_policy_clarification"),
        "V8E_T2_POINT_OF_USE_DATA_QUALITY_POLICY_INVALID",
    )
    for field in (
        "policy_applies_to_t1_t2",
        "policy_uniform_across_t0_t1_t2_t3",
        "existing_block_assignments_unchanged",
        "existing_partition_manifest_identity_unchanged",
    ):
        _require_state_bool(
            policy, field, True,
            "V8E_T2_POINT_OF_USE_DATA_QUALITY_POLICY_INVALID:" + field,
        )
    for field in (
        "ticker_removal_allowed",
        "ticker_replacement_allowed",
        "t_spare_replacement_allowed",
        "repartition_allowed",
        "imputation_allowed",
        "forward_fill_allowed",
        "back_fill_allowed",
        "alternate_source_substitution_allowed",
    ):
        _require_state_bool(
            policy, field, False,
            "V8E_T2_POINT_OF_USE_DATA_QUALITY_POLICY_INVALID:" + field,
        )

    research_activity_observed = any(
        state[counter] != 0
        for counter in ("backtests", "models_fitted", "profit_calculated", "parameter_search")
    )
    t2_real_data_acquired = t2["raw_data_acquired"]
    t2_opened = t2["opened_for_research"]
    t2_research_access_count = 0 if access_count is None else access_count
    t2_features_observed = (
        partition_history["block_assignments_exposed"]
        or trust_anchor_pinning["block_assignments_exposed"]
    )
    t2_outcomes_observed = research_activity_observed
    t2_membership_reassigned = not (
        partition_history["retry_performed"] is False
        and partition["block_assignments_recorded"] is False
        and policy["existing_partition_manifest_identity_unchanged"] is True
    )
    universe_definition_compatible = (
        partition["trusted_partition_authorized"] is True
        and partition["real_partition_manifest_validated"] is True
        and policy["existing_partition_manifest_identity_unchanged"] is True
        and partition_history["manifest_sha256"] == EXPECTED_V8_PARTITION_MANIFEST_SHA256
    )
    partition_algorithm_compatible = (
        partition_history["source_reproduction_status"] == "PASS"
        and partition_history["t0_reproduction_status"] == "PASS"
        and partition_history["partition_implementation_git_commit"] == EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
        and block_sizes["T2"] == T2_COUNT
    )
    data_quality_policy_unchanged = (
        policy["policy_applies_to_t1_t2"] is True
        and policy["policy_uniform_across_t0_t1_t2_t3"] is True
        and policy["existing_block_assignments_unchanged"] is True
        and policy["existing_partition_manifest_identity_unchanged"] is True
        and all(policy[field] is False for field in (
            "ticker_removal_allowed",
            "ticker_replacement_allowed",
            "t_spare_replacement_allowed",
            "repartition_allowed",
            "imputation_allowed",
            "forward_fill_allowed",
            "back_fill_allowed",
            "alternate_source_substitution_allowed",
        ))
    )

    # No private manifest or ticker identity is needed at this checkpoint.
    return _require_safe_conditions({
        "T2_real_data_acquired": t2_real_data_acquired,
        "T2_opened": t2_opened,
        "T2_research_access_count": t2_research_access_count,
        "T2_features_observed": t2_features_observed,
        "T2_outcomes_observed": t2_outcomes_observed,
        "T2_membership_reassigned": t2_membership_reassigned,
        "universe_definition_compatible": universe_definition_compatible,
        "partition_algorithm_compatible": partition_algorithm_compatible,
        "data_quality_policy_unchanged": data_quality_policy_unchanged,
    })


def _validate_anchor(anchor: Mapping[str, Any]) -> None:
    if anchor.get("authorization_status") != "AUTHORIZED":
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_TRUST_ANCHOR_NOT_AUTHORIZED")
    if anchor.get("authorized_partition_manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_MANIFEST_SHA_MISMATCH")
    if anchor.get("authorized_partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_PARTITION_IMPLEMENTATION_MISMATCH")


def _validate_readiness_metadata(readiness: Mapping[str, Any]) -> str:
    if (
        readiness.get("verification_stage") != READINESS_VERIFICATION_STAGE
        or readiness.get("logical_stage") != READINESS_LOGICAL_STAGE
        or readiness.get("verification_result") != "PASS"
        or readiness.get("frozen_design_commit") != FROZEN_DESIGN_COMMIT
    ):
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_READINESS_VERIFICATION_INVALID")
    return _require_hex(
        readiness.get("receipt_self_hash"), 64,
        "V8E_T2_POINT_OF_USE_READINESS_RECEIPT_HASH_INVALID",
    )


def _verify_prefreeze_binding(
    repository_root: Path,
    verified_head: str,
    git_blob_resolver: Callable[[Path, str, str], str],
    ancestor_checker: Callable[[Path, str, str, str], None],
) -> None:
    try:
        resolved = git_blob_resolver(repository_root, PREFREEZE_PRESERVATION_COMMIT, PREFREEZE_PRESERVATION_PATH)
    except V8EGitProvenanceBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_MISSING") from error
    if resolved != PREFREEZE_PRESERVATION_BLOB:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_BLOB_MISMATCH")
    try:
        ancestor_checker(
            repository_root, PREFREEZE_PRESERVATION_COMMIT, verified_head,
            "V8E_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_NOT_IN_CURRENT_HISTORY",
        )
    except V8EGitProvenanceBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked(error.reason) from error


def _derive_t2_point_of_use_preservation_with_dependencies(
    repository_root: Path,
    *,
    git_commit_resolver: Callable[[], str],
    frozen_design_verifier: Callable[[Path], None],
    freeze_approval_verifier: Callable[[Path, str], str],
    reviewed_implementation_binder: Callable[[Path, str], Mapping[str, Any]],
    anchor_reader: Callable[[Path, str], Mapping[str, Any]],
    authority_bridge_verifier: Callable[[Path, str, str], Mapping[str, Any]],
    readiness_reader: Callable[[], Mapping[str, Any]],
    gate_consumption_checker: Callable[[Path, str, str], bool],
    consumption_state_root: Path,
    state_conditions_reader: Callable[[Path, str], Mapping[str, Any]],
    prefreeze_blob_resolver: Callable[[Path, str, str], str],
    prefreeze_ancestor_checker: Callable[[Path, str, str, str], None],
) -> tuple[str, dict[str, Any]]:
    """Synthetic/DI-only implementation of the production derivation."""
    try:
        verified_head = git_commit_resolver()
    except V8EGitProvenanceBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked(error.reason) from error
    _require_hex(verified_head, 40, "V8E_T2_POINT_OF_USE_VERIFIED_HEAD_INVALID")

    try:
        frozen_design_verifier(repository_root)
        freeze_approval_blob = freeze_approval_verifier(repository_root, verified_head)
        reviewed = reviewed_implementation_binder(repository_root, verified_head)
    except (V8EProductionProvenanceBlocked, V8EGitProvenanceBlocked) as error:
        raise V8ET2PointOfUsePreservationBlocked(getattr(error, "reason", "V8E_T2_POINT_OF_USE_PROVENANCE_BLOCKED")) from error
    if freeze_approval_blob != EXPECTED_V8E_DESIGN_FREEZE_APPROVAL_BLOB:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_FREEZE_APPROVAL_INVALID")
    if not isinstance(reviewed, Mapping):
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_REVIEWED_IMPLEMENTATION_INVALID")
    _require_hex(reviewed.get("reviewed_implementation_git_commit"), 40, "V8E_T2_POINT_OF_USE_REVIEWED_IMPLEMENTATION_INVALID")

    try:
        anchor = anchor_reader(repository_root, verified_head)
    except (V8EProductionProvenanceBlocked, V8EGitProvenanceBlocked) as error:
        raise V8ET2PointOfUsePreservationBlocked(getattr(error, "reason", "V8E_T2_POINT_OF_USE_TRUST_ANCHOR_BLOCKED")) from error
    _validate_anchor(anchor)

    try:
        bridge = authority_bridge_verifier(repository_root, verified_head, T2_STAGE)
    except (V8EAuthorityBridgeBlocked, V8EGitProvenanceBlocked) as error:
        raise V8ET2PointOfUsePreservationBlocked(getattr(error, "reason", "V8E_T2_POINT_OF_USE_AUTHORITY_BRIDGE_BLOCKED")) from error
    if bridge.get("logical_block") != "T2" or bridge.get("review_result") != "PASS":
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_AUTHORITY_BRIDGE_NOT_PASS")

    try:
        readiness = readiness_reader()
    except BaseException as error:  # dependency boundary must fail closed
        raise V8ET2PointOfUsePreservationBlocked(getattr(error, "reason", "V8E_T2_POINT_OF_USE_READINESS_BLOCKED")) from error
    readiness_receipt_hash = _validate_readiness_metadata(readiness)

    try:
        consumed = gate_consumption_checker(
            consumption_state_root,
            GATE_T2_RAW_ACQUISITION, FROZEN_DESIGN_COMMIT,
        )
    except V8EHumanGateConsumptionBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked(error.reason) from error
    if consumed is not False:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_ACQUISITION_GATE_ALREADY_CONSUMED")

    _verify_prefreeze_binding(
        repository_root, verified_head, prefreeze_blob_resolver, prefreeze_ancestor_checker,
    )
    conditions = _require_safe_conditions(state_conditions_reader(repository_root, verified_head))

    artifact: dict[str, Any] = {
        "schema_version": POINT_OF_USE_SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": POINT_OF_USE_ARTIFACT_ROLE,
        "checkpoint": POINT_OF_USE_CHECKPOINT,
        "recheck": POINT_OF_USE_RECHECK,
        "v8e_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "source_prefreeze_preservation_git_commit": PREFREEZE_PRESERVATION_COMMIT,
        "source_prefreeze_preservation_git_blob_sha": PREFREEZE_PRESERVATION_BLOB,
        "readiness_verification_stage": READINESS_VERIFICATION_STAGE,
        "readiness_verification_result": "PASS",
        "readiness_receipt_self_hash": readiness_receipt_hash,
        "v8_trust_anchor_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": T2_COUNT,
        "t2_ticker_list_sha256": T2_TICKER_LIST_SHA256,
        **conditions,
        "t2_raw_acquisition_gate_consumed": False,
        "point_of_use_preservation_result": "PASS",
    }
    if set(artifact) != set(POINT_OF_USE_ARTIFACT_FIELDS):
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_ARTIFACT_SCHEMA_INVALID")
    return verified_head, artifact


def _production_derivation_dependencies(repository_root: Path) -> dict[str, Any]:
    return {
        "git_commit_resolver": lambda: resolve_verified_v8e_production_git_commit(repository_root),
        "frozen_design_verifier": verify_frozen_design_object,
        "freeze_approval_verifier": verify_design_freeze_approval_blob,
        "reviewed_implementation_binder": verify_reviewed_implementation_binding,
        "anchor_reader": read_and_verify_v8_trusted_partition_anchor,
        "authority_bridge_verifier": verify_stage_authority_bridge,
        "readiness_reader": require_t2_readiness_audit_verification_pass,
        "gate_consumption_checker": has_gate_been_consumed,
        "consumption_state_root": CANONICAL_CONSUMPTION_STATE_ROOT,
        "state_conditions_reader": lambda root, head: _derive_conditions_from_v8_state(root, head, read_git_object_bytes),
        "prefreeze_blob_resolver": resolve_git_blob,
        "prefreeze_ancestor_checker": require_strict_git_ancestor,
    }


def resolve_and_recheck_t2_point_of_use_preservation() -> dict[str, Any]:
    """Derive future point-of-use artifact content without writing it."""
    _head, artifact = _derive_t2_point_of_use_preservation_with_dependencies(
        CANONICAL_REPOSITORY_ROOT,
        **_production_derivation_dependencies(CANONICAL_REPOSITORY_ROOT),
    )
    return artifact


def derive_t2_point_of_use_preservation_artifact() -> dict[str, Any]:
    """Compatibility name for the zero-argument production derivation."""
    return resolve_and_recheck_t2_point_of_use_preservation()


def _validate_preservation_artifact(raw: bytes) -> dict[str, Any]:
    artifact = _strict_json_object(
        raw,
        invalid_reason="V8E_T2_POINT_OF_USE_ARTIFACT_INVALID_JSON",
        duplicate_reason="V8E_T2_POINT_OF_USE_ARTIFACT_DUPLICATE_KEY",
    )
    _require_exact_fields(artifact, POINT_OF_USE_ARTIFACT_FIELDS, "V8E_T2_POINT_OF_USE_ARTIFACT_SCHEMA_INVALID")
    expected = {
        "schema_version": POINT_OF_USE_SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": POINT_OF_USE_ARTIFACT_ROLE,
        "checkpoint": POINT_OF_USE_CHECKPOINT,
        "recheck": POINT_OF_USE_RECHECK,
        "v8e_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "source_prefreeze_preservation_git_commit": PREFREEZE_PRESERVATION_COMMIT,
        "source_prefreeze_preservation_git_blob_sha": PREFREEZE_PRESERVATION_BLOB,
        "readiness_verification_stage": READINESS_VERIFICATION_STAGE,
        "readiness_verification_result": "PASS",
        "v8_trust_anchor_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": T2_COUNT,
        "t2_ticker_list_sha256": T2_TICKER_LIST_SHA256,
        "t2_raw_acquisition_gate_consumed": False,
        "point_of_use_preservation_result": "PASS",
    }
    for key, value in expected.items():
        if artifact[key] != value:
            raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_ARTIFACT_VALUE_MISMATCH:" + key)
    _require_hex(artifact["source_prefreeze_preservation_git_commit"], 40, "V8E_T2_POINT_OF_USE_PREFREEZE_COMMIT_INVALID")
    _require_hex(artifact["source_prefreeze_preservation_git_blob_sha"], 40, "V8E_T2_POINT_OF_USE_PREFREEZE_BLOB_INVALID")
    _require_hex(artifact["v8e_frozen_design_commit"], 40, "V8E_T2_POINT_OF_USE_DESIGN_COMMIT_INVALID")
    _require_hex(artifact["readiness_receipt_self_hash"], 64, "V8E_T2_POINT_OF_USE_READINESS_HASH_INVALID")
    _require_hex(artifact["v8_trust_anchor_git_blob"], 40, "V8E_T2_POINT_OF_USE_ANCHOR_BLOB_INVALID")
    _require_hex(artifact["original_v8_partition_manifest_sha256"], 64, "V8E_T2_POINT_OF_USE_MANIFEST_HASH_INVALID")
    _require_hex(artifact["parent_v8_partition_implementation_commit"], 40, "V8E_T2_POINT_OF_USE_PARENT_COMMIT_INVALID")
    _require_hex(artifact["t2_ticker_list_sha256"], 64, "V8E_T2_POINT_OF_USE_T2_HASH_INVALID")
    if type(artifact["t2_count"]) is not int or artifact["t2_count"] != T2_COUNT:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_T2_COUNT_INVALID")
    _require_safe_conditions({key: artifact[key] for key in NINE_CONDITION_FIELDS})
    return artifact


def _read_review_artifact(repository_root: Path, verified_head: str) -> dict[str, Any]:
    try:
        raw = read_git_object_bytes(repository_root, verified_head, POINT_OF_USE_REVIEW_PATH)
    except V8EGitProvenanceBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8E_T2_POINT_OF_USE_REVIEW_INVALID_JSON",
        duplicate_reason="V8E_T2_POINT_OF_USE_REVIEW_DUPLICATE_KEY",
    )
    _require_exact_fields(review, POINT_OF_USE_REVIEW_FIELDS, "V8E_T2_POINT_OF_USE_REVIEW_SCHEMA_INVALID")
    expected = {
        "schema_version": POINT_OF_USE_REVIEW_SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": POINT_OF_USE_REVIEW_ROLE,
        "checkpoint": POINT_OF_USE_REVIEW_CHECKPOINT,
        "v8e_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "review_result": "PASS",
    }
    for key, value in expected.items():
        if review[key] != value:
            raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_REVIEW_VALUE_MISMATCH:" + key)
    _require_hex(review["reviewed_recheck_git_commit"], 40, "V8E_T2_POINT_OF_USE_REVIEW_COMMIT_INVALID")
    _require_hex(review["reviewed_recheck_git_blob_sha"], 40, "V8E_T2_POINT_OF_USE_REVIEW_BLOB_INVALID")
    return review


def _require_t2_point_of_use_preservation_review_pass_with_dependencies(
    repository_root: Path,
    *,
    dependencies: Mapping[str, Any],
    review_reader: Callable[[Path, str], Mapping[str, Any]],
    artifact_blob_resolver: Callable[[Path, str, str], str],
    artifact_reader: Callable[[Path, str, str], bytes],
    ancestor_checker: Callable[[Path, str, str, str], None],
) -> dict[str, Any]:
    """Synthetic/DI-only review-reader implementation."""
    first_head, _ = _derive_t2_point_of_use_preservation_with_dependencies(
        repository_root, **dependencies,
    )
    review = dict(review_reader(repository_root, first_head))
    try:
        ancestor_checker(
            repository_root, review["reviewed_recheck_git_commit"], first_head,
            "V8E_T2_POINT_OF_USE_REVIEW_NOT_IN_CURRENT_HISTORY",
        )
        resolved_blob = artifact_blob_resolver(
            repository_root, review["reviewed_recheck_git_commit"], POINT_OF_USE_ARTIFACT_PATH,
        )
        if resolved_blob != review["reviewed_recheck_git_blob_sha"]:
            raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_REVIEW_BLOB_MISMATCH")
        artifact = _validate_preservation_artifact(
            artifact_reader(repository_root, review["reviewed_recheck_git_commit"], POINT_OF_USE_ARTIFACT_PATH)
        )
    except V8EGitProvenanceBlocked as error:
        raise V8ET2PointOfUsePreservationBlocked(error.reason) from error

    second_head, live_artifact = _derive_t2_point_of_use_preservation_with_dependencies(
        repository_root, **dependencies,
    )
    if second_head != first_head:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_REVIEW_VERIFIED_HEAD_CHANGED")
    if artifact != live_artifact:
        raise V8ET2PointOfUsePreservationBlocked("V8E_T2_POINT_OF_USE_REVIEW_ARTIFACT_LIVE_MISMATCH")
    return {
        "schema_version": review["schema_version"],
        "study": review["study"],
        "artifact_role": review["artifact_role"],
        "checkpoint": review["checkpoint"],
        "v8e_frozen_design_commit": review["v8e_frozen_design_commit"],
        "reviewed_recheck_git_commit": review["reviewed_recheck_git_commit"],
        "reviewed_recheck_git_blob_sha": review["reviewed_recheck_git_blob_sha"],
        "review_result": review["review_result"],
        "preservation_artifact": artifact,
    }


def require_t2_point_of_use_preservation_review_pass() -> dict[str, Any]:
    """Require the independently reviewed, current T2 point-of-use PASS."""
    repository_root = CANONICAL_REPOSITORY_ROOT
    dependencies = _production_derivation_dependencies(repository_root)
    return _require_t2_point_of_use_preservation_review_pass_with_dependencies(
        repository_root,
        dependencies=dependencies,
        review_reader=_read_review_artifact,
        artifact_blob_resolver=resolve_git_blob,
        artifact_reader=read_git_object_bytes,
        ancestor_checker=require_strict_git_ancestor,
    )


__all__ = [
    "NINE_CONDITION_FIELDS",
    "POINT_OF_USE_ARTIFACT_FIELDS",
    "POINT_OF_USE_ARTIFACT_PATH",
    "POINT_OF_USE_CHECKPOINT",
    "POINT_OF_USE_RECHECK",
    "POINT_OF_USE_REVIEW_FIELDS",
    "POINT_OF_USE_REVIEW_PATH",
    "POINT_OF_USE_REVIEW_CHECKPOINT",
    "POINT_OF_USE_REVIEW_SCHEMA_VERSION",
    "POINT_OF_USE_SCHEMA_VERSION",
    "READINESS_VERIFICATION_STAGE",
    "V8ET2PointOfUsePreservationBlocked",
    "derive_t2_point_of_use_preservation_artifact",
    "require_t2_point_of_use_preservation_review_pass",
    "resolve_and_recheck_t2_point_of_use_preservation",
]
