"""Schema, builder, and validator for the future `V8C_TRUSTED_ALLOCATION.json`.

Mirrors `src.v8b_trust_pin`'s proven pattern for V8C's `T1C` allocation.
This module implements only the **schema, builder, and validator** for that
future artifact -- it never writes one to disk, and this repository does
not contain a `V8C_TRUSTED_ALLOCATION.json` file. Building or validating a
pin object in memory performs no I/O and requires no network access; it is
not, by itself, the ``CREATE_V8C_TRUSTED_ALLOCATION_PIN`` gate.

The schema deliberately contains only safe metadata -- hashes, counts,
commit IDs, schema/study/role identifiers, authorization status, and
timestamps -- and never a `T1C` or `T_spare` ticker identity.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

STUDY_NAME = "V8C_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8C_TRUSTED_ALLOCATION_V1"
ARTIFACT_ROLE = "T1C_ALLOCATION_TRUST_PIN"
LOGICAL_BLOCK = "T1C"

HUMAN_GATE_PREFIX = "V8C_HUMAN_AUTHORIZE_T1C_ALLOCATION_PIN_AT_"


def expected_human_gate(authorized_allocation_artifact_self_hash: str) -> str:
    """The single well-formed human_gate string for a pin authorizing the
    allocation artifact with this exact ``artifact_self_hash``."""
    return HUMAN_GATE_PREFIX + authorized_allocation_artifact_self_hash


AUTHORIZATION_STATUSES = ("NOT_AUTHORIZED", "AUTHORIZED")

TRUST_PIN_FIELDS = (
    "schema_version",
    "study_name",
    "artifact_role",
    "logical_block",
    "authorization_status",
    "authorized_allocation_artifact_self_hash",
    "parent_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "parent_t_spare_ticker_count",
    "parent_t_spare_ticker_list_sha256",
    "t1c_ticker_count",
    "t1c_ticker_list_sha256",
    "remaining_t_spare_ticker_count",
    "remaining_t_spare_ticker_list_sha256",
    "v8c_frozen_design_commit",
    "v8c_allocation_implementation_commit",
    "verification_result",
    "human_gate",
    "authorization_note",
)

_REQUIRED_VERIFICATION_SUMMARY_FIELDS = (
    "parent_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "parent_t_spare_ticker_count",
    "parent_t_spare_ticker_list_sha256",
    "t1c_ticker_count",
    "t1c_ticker_list_sha256",
    "remaining_t_spare_ticker_count",
    "remaining_t_spare_ticker_list_sha256",
    "v8c_frozen_design_commit",
    "v8c_allocation_implementation_commit",
    "artifact_self_hash",
)


class V8CTrustPinBlocked(RuntimeError):
    """Fail-closed trust-pin construction/validation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_git_commit(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8CTrustPinBlocked(reason)
    return value


def _require_sha256_hex(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8CTrustPinBlocked(reason)
    return value


def build_trust_pin(
    *,
    verification_result_summary: Mapping[str, Any],
    authorization_note: str,
) -> dict[str, Any]:
    """Build (never writes) an ``AUTHORIZED`` trust-pin from a PASS verification summary."""
    if verification_result_summary.get("result") != "PASS":
        raise V8CTrustPinBlocked("VERIFICATION_RESULT_NOT_PASS")
    for forbidden in ("t1c_tickers", "remaining_t_spare_tickers", "parent_t_spare_tickers"):
        if forbidden in verification_result_summary:
            raise V8CTrustPinBlocked("VERIFICATION_SUMMARY_CONTAINS_TICKER_IDENTITIES")
    missing = set(_REQUIRED_VERIFICATION_SUMMARY_FIELDS) - set(verification_result_summary)
    if missing:
        raise V8CTrustPinBlocked("VERIFICATION_SUMMARY_SCHEMA_INVALID")
    if not isinstance(authorization_note, str):
        raise V8CTrustPinBlocked("AUTHORIZATION_NOTE_INVALID")

    artifact_self_hash = _require_sha256_hex(
        verification_result_summary["artifact_self_hash"], "ARTIFACT_SELF_HASH_INVALID"
    )
    human_gate = expected_human_gate(artifact_self_hash)

    pin: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "artifact_role": ARTIFACT_ROLE,
        "logical_block": LOGICAL_BLOCK,
        "authorization_status": "AUTHORIZED",
        "authorized_allocation_artifact_self_hash": artifact_self_hash,
        "parent_v8_partition_manifest_sha256": _require_sha256_hex(
            verification_result_summary["parent_v8_partition_manifest_sha256"],
            "PARENT_V8_PARTITION_MANIFEST_SHA_INVALID",
        ),
        "parent_v8_partition_implementation_commit": _require_git_commit(
            verification_result_summary["parent_v8_partition_implementation_commit"],
            "PARENT_V8_PARTITION_IMPLEMENTATION_COMMIT_INVALID",
        ),
        "parent_t_spare_ticker_count": int(verification_result_summary["parent_t_spare_ticker_count"]),
        "parent_t_spare_ticker_list_sha256": _require_sha256_hex(
            verification_result_summary["parent_t_spare_ticker_list_sha256"],
            "PARENT_T_SPARE_TICKER_LIST_SHA_INVALID",
        ),
        "t1c_ticker_count": int(verification_result_summary["t1c_ticker_count"]),
        "t1c_ticker_list_sha256": _require_sha256_hex(
            verification_result_summary["t1c_ticker_list_sha256"], "T1C_TICKER_LIST_SHA_INVALID"
        ),
        "remaining_t_spare_ticker_count": int(verification_result_summary["remaining_t_spare_ticker_count"]),
        "remaining_t_spare_ticker_list_sha256": _require_sha256_hex(
            verification_result_summary["remaining_t_spare_ticker_list_sha256"],
            "REMAINING_T_SPARE_TICKER_LIST_SHA_INVALID",
        ),
        "v8c_frozen_design_commit": _require_git_commit(
            verification_result_summary["v8c_frozen_design_commit"], "V8C_FROZEN_DESIGN_COMMIT_INVALID"
        ),
        "v8c_allocation_implementation_commit": _require_git_commit(
            verification_result_summary["v8c_allocation_implementation_commit"],
            "V8C_ALLOCATION_IMPLEMENTATION_COMMIT_INVALID",
        ),
        "verification_result": "PASS",
        "human_gate": human_gate,
        "authorization_note": authorization_note,
    }
    if set(pin) != set(TRUST_PIN_FIELDS):
        raise V8CTrustPinBlocked("TRUST_PIN_SCHEMA_INVALID")
    return pin


def validate_trust_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``pin`` is a well-formed, internally-consistent trust pin."""
    if not isinstance(pin, Mapping) or set(pin) != set(TRUST_PIN_FIELDS):
        raise V8CTrustPinBlocked("TRUST_PIN_SCHEMA_INVALID")
    for forbidden in ("t1c_tickers", "remaining_t_spare_tickers", "parent_t_spare_tickers"):
        if forbidden in pin:
            raise V8CTrustPinBlocked("TRUST_PIN_CONTAINS_TICKER_IDENTITIES")
    if pin["schema_version"] != SCHEMA_VERSION:
        raise V8CTrustPinBlocked("TRUST_PIN_SCHEMA_VERSION_MISMATCH")
    if pin["study_name"] != STUDY_NAME:
        raise V8CTrustPinBlocked("TRUST_PIN_STUDY_NAME_MISMATCH")
    if pin["artifact_role"] != ARTIFACT_ROLE:
        raise V8CTrustPinBlocked("TRUST_PIN_ARTIFACT_ROLE_MISMATCH")
    if pin["logical_block"] != LOGICAL_BLOCK:
        raise V8CTrustPinBlocked("TRUST_PIN_LOGICAL_BLOCK_MISMATCH")

    status = pin["authorization_status"]
    if status not in AUTHORIZATION_STATUSES:
        raise V8CTrustPinBlocked("TRUST_PIN_AUTHORIZATION_STATUS_INVALID")

    nullable_when_unauthorized = (
        "authorized_allocation_artifact_self_hash",
        "parent_v8_partition_manifest_sha256",
        "parent_v8_partition_implementation_commit",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
        "t1c_ticker_count",
        "t1c_ticker_list_sha256",
        "remaining_t_spare_ticker_count",
        "remaining_t_spare_ticker_list_sha256",
        "v8c_frozen_design_commit",
        "v8c_allocation_implementation_commit",
        "verification_result",
    )
    if status == "NOT_AUTHORIZED":
        if any(pin[field] is not None for field in nullable_when_unauthorized):
            raise V8CTrustPinBlocked("TRUST_PIN_UNAUTHORIZED_FIELDS_INVALID")
        return dict(pin)

    _require_sha256_hex(pin["authorized_allocation_artifact_self_hash"], "ARTIFACT_SELF_HASH_INVALID")
    _require_sha256_hex(pin["parent_v8_partition_manifest_sha256"], "PARENT_V8_PARTITION_MANIFEST_SHA_INVALID")
    _require_git_commit(
        pin["parent_v8_partition_implementation_commit"], "PARENT_V8_PARTITION_IMPLEMENTATION_COMMIT_INVALID"
    )
    if type(pin["parent_t_spare_ticker_count"]) is not int or pin["parent_t_spare_ticker_count"] <= 600:
        raise V8CTrustPinBlocked("PARENT_T_SPARE_TICKER_COUNT_INVALID")
    _require_sha256_hex(pin["parent_t_spare_ticker_list_sha256"], "PARENT_T_SPARE_TICKER_LIST_SHA_INVALID")
    if type(pin["t1c_ticker_count"]) is not int or pin["t1c_ticker_count"] != 300:
        raise V8CTrustPinBlocked("T1C_TICKER_COUNT_INVALID")
    _require_sha256_hex(pin["t1c_ticker_list_sha256"], "T1C_TICKER_LIST_SHA_INVALID")
    if (
        type(pin["remaining_t_spare_ticker_count"]) is not int
        or pin["remaining_t_spare_ticker_count"] != pin["parent_t_spare_ticker_count"] - 300
    ):
        raise V8CTrustPinBlocked("REMAINING_T_SPARE_TICKER_COUNT_INVALID")
    _require_sha256_hex(pin["remaining_t_spare_ticker_list_sha256"], "REMAINING_T_SPARE_TICKER_LIST_SHA_INVALID")
    _require_git_commit(pin["v8c_frozen_design_commit"], "V8C_FROZEN_DESIGN_COMMIT_INVALID")
    _require_git_commit(pin["v8c_allocation_implementation_commit"], "V8C_ALLOCATION_IMPLEMENTATION_COMMIT_INVALID")
    if pin["verification_result"] != "PASS":
        raise V8CTrustPinBlocked("TRUST_PIN_VERIFICATION_RESULT_NOT_PASS")
    if pin["human_gate"] != expected_human_gate(pin["authorized_allocation_artifact_self_hash"]):
        raise V8CTrustPinBlocked("TRUST_PIN_HUMAN_GATE_INVALID")
    if not isinstance(pin["authorization_note"], str):
        raise V8CTrustPinBlocked("TRUST_PIN_AUTHORIZATION_NOTE_INVALID")
    return dict(pin)


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8CTrustPinBlocked("TRUST_PIN_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CTrustPinBlocked("TRUST_PIN_INVALID_JSON") from error
    if not isinstance(parsed, dict):
        raise V8CTrustPinBlocked("TRUST_PIN_INVALID_JSON")
    return parsed


def read_trust_pin_bytes(raw: bytes) -> dict[str, Any]:
    """Duplicate-key-safe parse plus full schema/consistency validation.

    There is deliberately no corresponding "write" function in this module:
    this implementation phase does not authorize creating the real
    ``V8C_TRUSTED_ALLOCATION.json`` pin artifact.
    """
    return validate_trust_pin(_strict_json_object(raw))


__all__ = [
    "ARTIFACT_ROLE",
    "AUTHORIZATION_STATUSES",
    "HUMAN_GATE_PREFIX",
    "LOGICAL_BLOCK",
    "SCHEMA_VERSION",
    "STUDY_NAME",
    "TRUST_PIN_FIELDS",
    "V8CTrustPinBlocked",
    "build_trust_pin",
    "expected_human_gate",
    "read_trust_pin_bytes",
    "validate_trust_pin",
]
