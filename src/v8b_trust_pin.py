"""Schema, builder, and validator for the future `V8B_TRUSTED_ALLOCATION.json` (§11.3.C).

`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §11.3.C describes a public,
repository-fixed artifact that pins a *verified* `T1B` allocation artifact's
``artifact_self_hash`` as ``AUTHORIZED``, mirroring how `V8_TRUSTED_
PARTITION.json` already pins the original V8 partition manifest. That
pinning is gated by a separate human authorization
(`HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION`, §12) that has not
occurred.

This module implements only the **schema, builder, and validator** for that
future artifact -- it never writes one to disk, and this repository does
not contain a `V8B_TRUSTED_ALLOCATION.json` file. Building or validating a
pin object in memory (e.g. for a test) performs no I/O and requires no
network access; it is not, by itself, the ``CREATE_V8B_TRUSTED_ALLOCATION_
PIN`` gate.

The schema deliberately contains only safe metadata -- hashes, counts,
commit IDs, schema/study/role identifiers, authorization status, and
timestamps -- and never a `T1B` or `T_spare` ticker identity, matching
§11.3.C's public/private artifact boundary exactly.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

STUDY_NAME = "V8B_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8B_TRUSTED_ALLOCATION_V1"
ARTIFACT_ROLE = "T1B_ALLOCATION_TRUST_PIN"
LOGICAL_BLOCK = "T1B"

# Frozen human-gate grammar for HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_
# ALLOCATION (§12), mirroring the exact-embedding pattern this repository
# already established for one-time pin authorizations --
# V8_STATE.json's V8 trust-anchor pin used
# "V8_HUMAN_AUTHORIZE_ONE_TRUST_ANCHOR_PIN_AT_<head>_FOR_MANIFEST_<hash>_
# IMPL_<commit>", and V8B_DESIGN_FREEZE_APPROVAL.json's human_gate is
# "V8B_HUMAN_DESIGN_FREEZE_APPROVED_FOR_COMMIT_<commit>". Both are a fixed
# prefix followed by the exact value the approval binds to, never freeform
# text. This grammar is a naming/audit-trail convention, not a
# methodology decision: no threshold, partition, or selection-rule
# content is embedded, only which exact artifact a future human approves.
# No real approval is asserted or invented here -- this only fixes what a
# well-formed future approval string must look like.
HUMAN_GATE_PREFIX = "V8B_HUMAN_AUTHORIZE_T1B_ALLOCATION_PIN_AT_"


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
    "t1b_ticker_count",
    "t1b_ticker_list_sha256",
    "remaining_t_spare_ticker_count",
    "remaining_t_spare_ticker_list_sha256",
    "v8b_frozen_design_commit",
    "v8b_allocation_implementation_commit",
    "verification_result",
    "human_gate",
    "authorization_note",
)

# The exact fields a §11.4-PASS verification summary
# (``src.v8b_allocation_verification.resolve_and_verify_t1b_allocation_artifact``'s
# return value) must supply for a pin to be built from it.
_REQUIRED_VERIFICATION_SUMMARY_FIELDS = (
    "parent_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "parent_t_spare_ticker_count",
    "parent_t_spare_ticker_list_sha256",
    "t1b_ticker_count",
    "t1b_ticker_list_sha256",
    "remaining_t_spare_ticker_count",
    "remaining_t_spare_ticker_list_sha256",
    "v8b_frozen_design_commit",
    "v8b_allocation_implementation_commit",
    "artifact_self_hash",
)


class V8BTrustPinBlocked(RuntimeError):
    """Fail-closed trust-pin construction/validation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_git_commit(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8BTrustPinBlocked(reason)
    return value


def _require_sha256_hex(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8BTrustPinBlocked(reason)
    return value


def build_trust_pin(
    *,
    verification_result_summary: Mapping[str, Any],
    authorization_note: str,
) -> dict[str, Any]:
    """Build (never writes) an ``AUTHORIZED`` trust-pin from a PASS verification summary.

    ``verification_result_summary`` must be exactly the safe public dict
    ``src.v8b_allocation_verification.resolve_and_verify_t1b_allocation_artifact``
    returns on a ``result == "PASS"`` outcome -- this function refuses any
    mapping that also carries a ticker-identity field, so a caller cannot
    accidentally launder raw ticker lists into a "public" pin.

    There is deliberately no ``human_gate`` parameter: the frozen grammar
    (``HUMAN_GATE_PREFIX`` + the exact artifact hash being authorized)
    leaves no freedom for a caller to supply arbitrary text, so the value
    is always derived here, never accepted as input.
    """
    if verification_result_summary.get("result") != "PASS":
        raise V8BTrustPinBlocked("VERIFICATION_RESULT_NOT_PASS")
    for forbidden in ("t1b_tickers", "remaining_t_spare_tickers", "parent_t_spare_tickers"):
        if forbidden in verification_result_summary:
            raise V8BTrustPinBlocked("VERIFICATION_SUMMARY_CONTAINS_TICKER_IDENTITIES")
    missing = set(_REQUIRED_VERIFICATION_SUMMARY_FIELDS) - set(verification_result_summary)
    if missing:
        raise V8BTrustPinBlocked("VERIFICATION_SUMMARY_SCHEMA_INVALID")
    if not isinstance(authorization_note, str):
        raise V8BTrustPinBlocked("AUTHORIZATION_NOTE_INVALID")

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
        "authorized_allocation_artifact_self_hash": _require_sha256_hex(
            verification_result_summary["artifact_self_hash"], "ARTIFACT_SELF_HASH_INVALID"
        ),
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
        "t1b_ticker_count": int(verification_result_summary["t1b_ticker_count"]),
        "t1b_ticker_list_sha256": _require_sha256_hex(
            verification_result_summary["t1b_ticker_list_sha256"], "T1B_TICKER_LIST_SHA_INVALID"
        ),
        "remaining_t_spare_ticker_count": int(verification_result_summary["remaining_t_spare_ticker_count"]),
        "remaining_t_spare_ticker_list_sha256": _require_sha256_hex(
            verification_result_summary["remaining_t_spare_ticker_list_sha256"],
            "REMAINING_T_SPARE_TICKER_LIST_SHA_INVALID",
        ),
        "v8b_frozen_design_commit": _require_git_commit(
            verification_result_summary["v8b_frozen_design_commit"], "V8B_FROZEN_DESIGN_COMMIT_INVALID"
        ),
        "v8b_allocation_implementation_commit": _require_git_commit(
            verification_result_summary["v8b_allocation_implementation_commit"],
            "V8B_ALLOCATION_IMPLEMENTATION_COMMIT_INVALID",
        ),
        "verification_result": "PASS",
        "human_gate": human_gate,
        "authorization_note": authorization_note,
    }
    if set(pin) != set(TRUST_PIN_FIELDS):
        raise V8BTrustPinBlocked("TRUST_PIN_SCHEMA_INVALID")
    return pin


def validate_trust_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``pin`` is a well-formed, internally-consistent trust pin.

    Accepts both ``NOT_AUTHORIZED`` placeholders (all pin-specific fields
    ``None``) and ``AUTHORIZED`` pins (all fields populated and well-formed).
    Never accepts a mapping containing a ticker-identity field.
    """
    if not isinstance(pin, Mapping) or set(pin) != set(TRUST_PIN_FIELDS):
        raise V8BTrustPinBlocked("TRUST_PIN_SCHEMA_INVALID")
    for forbidden in ("t1b_tickers", "remaining_t_spare_tickers", "parent_t_spare_tickers"):
        if forbidden in pin:
            raise V8BTrustPinBlocked("TRUST_PIN_CONTAINS_TICKER_IDENTITIES")
    if pin["schema_version"] != SCHEMA_VERSION:
        raise V8BTrustPinBlocked("TRUST_PIN_SCHEMA_VERSION_MISMATCH")
    if pin["study_name"] != STUDY_NAME:
        raise V8BTrustPinBlocked("TRUST_PIN_STUDY_NAME_MISMATCH")
    if pin["artifact_role"] != ARTIFACT_ROLE:
        raise V8BTrustPinBlocked("TRUST_PIN_ARTIFACT_ROLE_MISMATCH")
    if pin["logical_block"] != LOGICAL_BLOCK:
        raise V8BTrustPinBlocked("TRUST_PIN_LOGICAL_BLOCK_MISMATCH")

    status = pin["authorization_status"]
    if status not in AUTHORIZATION_STATUSES:
        raise V8BTrustPinBlocked("TRUST_PIN_AUTHORIZATION_STATUS_INVALID")

    nullable_when_unauthorized = (
        "authorized_allocation_artifact_self_hash",
        "parent_v8_partition_manifest_sha256",
        "parent_v8_partition_implementation_commit",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
        "t1b_ticker_count",
        "t1b_ticker_list_sha256",
        "remaining_t_spare_ticker_count",
        "remaining_t_spare_ticker_list_sha256",
        "v8b_frozen_design_commit",
        "v8b_allocation_implementation_commit",
        "verification_result",
    )
    if status == "NOT_AUTHORIZED":
        if any(pin[field] is not None for field in nullable_when_unauthorized):
            raise V8BTrustPinBlocked("TRUST_PIN_UNAUTHORIZED_FIELDS_INVALID")
        return dict(pin)

    # AUTHORIZED: every field must be well-formed and consistent.
    _require_sha256_hex(pin["authorized_allocation_artifact_self_hash"], "ARTIFACT_SELF_HASH_INVALID")
    _require_sha256_hex(pin["parent_v8_partition_manifest_sha256"], "PARENT_V8_PARTITION_MANIFEST_SHA_INVALID")
    _require_git_commit(
        pin["parent_v8_partition_implementation_commit"], "PARENT_V8_PARTITION_IMPLEMENTATION_COMMIT_INVALID"
    )
    if type(pin["parent_t_spare_ticker_count"]) is not int or pin["parent_t_spare_ticker_count"] <= 300:
        raise V8BTrustPinBlocked("PARENT_T_SPARE_TICKER_COUNT_INVALID")
    _require_sha256_hex(pin["parent_t_spare_ticker_list_sha256"], "PARENT_T_SPARE_TICKER_LIST_SHA_INVALID")
    if type(pin["t1b_ticker_count"]) is not int or pin["t1b_ticker_count"] != 300:
        raise V8BTrustPinBlocked("T1B_TICKER_COUNT_INVALID")
    _require_sha256_hex(pin["t1b_ticker_list_sha256"], "T1B_TICKER_LIST_SHA_INVALID")
    if (
        type(pin["remaining_t_spare_ticker_count"]) is not int
        or pin["remaining_t_spare_ticker_count"] != pin["parent_t_spare_ticker_count"] - 300
    ):
        raise V8BTrustPinBlocked("REMAINING_T_SPARE_TICKER_COUNT_INVALID")
    _require_sha256_hex(
        pin["remaining_t_spare_ticker_list_sha256"], "REMAINING_T_SPARE_TICKER_LIST_SHA_INVALID"
    )
    _require_git_commit(pin["v8b_frozen_design_commit"], "V8B_FROZEN_DESIGN_COMMIT_INVALID")
    _require_git_commit(
        pin["v8b_allocation_implementation_commit"], "V8B_ALLOCATION_IMPLEMENTATION_COMMIT_INVALID"
    )
    if pin["verification_result"] != "PASS":
        raise V8BTrustPinBlocked("TRUST_PIN_VERIFICATION_RESULT_NOT_PASS")
    # The human_gate must be exactly the frozen grammar bound to this
    # pin's own authorized artifact hash -- no arbitrary nonempty string,
    # however plausible-sounding, is accepted (HIGH-3 remediation).
    if pin["human_gate"] != expected_human_gate(pin["authorized_allocation_artifact_self_hash"]):
        raise V8BTrustPinBlocked("TRUST_PIN_HUMAN_GATE_INVALID")
    if not isinstance(pin["authorization_note"], str):
        raise V8BTrustPinBlocked("TRUST_PIN_AUTHORIZATION_NOTE_INVALID")
    return dict(pin)


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8BTrustPinBlocked("TRUST_PIN_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8BTrustPinBlocked("TRUST_PIN_INVALID_JSON") from error
    if not isinstance(parsed, dict):
        raise V8BTrustPinBlocked("TRUST_PIN_INVALID_JSON")
    return parsed


def read_trust_pin_bytes(raw: bytes) -> dict[str, Any]:
    """Duplicate-key-safe parse plus full schema/consistency validation.

    There is deliberately no corresponding "write" function in this module:
    this implementation phase does not authorize creating the real
    ``V8B_TRUSTED_ALLOCATION.json`` pin artifact (``CREATE_V8B_TRUSTED_
    ALLOCATION_PIN`` remains a separate, later, human-gated action, §12).
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
    "V8BTrustPinBlocked",
    "build_trust_pin",
    "expected_human_gate",
    "read_trust_pin_bytes",
    "validate_trust_pin",
]
