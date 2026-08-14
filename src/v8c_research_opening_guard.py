"""§10.2 research-opening point-of-use security hardening, for `T1C` and `T2`.

`V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §10.2 (inherited from V8B):

    official_opening_path_accepts_caller_crafted_arbitrary_mapping=PROHIBITED
    official_opening_path_accepts_caller_supplied_authority_path=PROHIBITED
    verified_acquisition_manifest_resolved_by_official_resolver=REQUIRED
    trusted_block_and_authority_binding_reverified_at_point_of_use=REQUIRED
    raw_payload_byte_count_binding_reverified_immediately_before_opening=REQUIRED
    raw_payload_sha256_binding_reverified_immediately_before_opening=REQUIRED
    earlier_read_only_artifact_verification_pass_is_sufficient_alone=FALSE
    post_verification_tampering_detected=BLOCK

This module implements only the **verification step** these requirements
describe -- ``verify_point_of_use_before_opening`` re-derives every trust
value fresh (never cached from an earlier call) by calling the sole public
production resolver, `src.v8c_acquisition_artifact_verification.resolve_
and_verify_acquisition_artifact`, which itself accepts no caller-supplied
authority mapping or trust-root override. It performs no feature/outcome
computation and returns no raw OHLCV data.

Consistent with `src.v8b_historical_acquisition`'s own MEDIUM-3 security
invariant, this module deliberately does **not** implement any
``open_for_*`` function that returns raw OHLCV/feature/outcome data --
research opening remains entirely behind its own separate, still-unreached
human gate (`SEPARATE_T1C_RESEARCH_OPENING_GATE` /
`SEPARATE_T2_RESEARCH_OPENING_GATE`, §6, §12). This verification function
does not authorize opening and does not consume either research-opening
gate; it is only the mandatory pre-check a future opening implementation
must call, and re-calling it after any tampering between calls
independently re-detects that tampering (it never trusts a cached prior
PASS).
"""

from __future__ import annotations

from typing import Any

from src.v8c_acquisition_artifact_verification import (
    V8CAcquisitionArtifactVerificationBlocked,
    resolve_and_verify_acquisition_artifact,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T1C_RESEARCH_OPENING,
    GATE_T2_RESEARCH_OPENING,
    V8CHumanGateConsumptionBlocked,
    has_gate_been_consumed,
)
from src.v8c_production_provenance import EXPECTED_V8C_FROZEN_DESIGN_COMMIT

RESEARCH_OPENING_GATE_BY_BLOCK = {
    "T1C": GATE_T1C_RESEARCH_OPENING,
    "T2": GATE_T2_RESEARCH_OPENING,
}


class V8CResearchOpeningGuardBlocked(RuntimeError):
    """Fail-closed §10.2 point-of-use research-opening security error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def no_research_opening_api_exists() -> bool:
    """Live evidence, not a self-declared claim: no ``open_for_*``/
    research-opening symbol that returns raw OHLCV/feature/outcome data
    exists anywhere in the bound `src.v8c_historical_acquisition`
    production module."""
    from src import v8c_historical_acquisition as _acquisition_module

    return not any(name.startswith("open_for") or name.startswith("open_t") for name in dir(_acquisition_module))


def verify_point_of_use_before_opening(output_root, block: str) -> dict[str, Any]:
    """Mandatory point-of-use re-verification a future opening
    implementation must call **immediately before** opening -- never
    relying on an earlier read-only artifact-verification PASS alone.

    Re-derives every trust value fresh from the sole public production
    resolver (never a caller-supplied mapping or path): resolves the
    verified acquisition manifest, re-checks the trusted block/authority
    binding, and re-verifies every raw payload's byte count and SHA-256
    binding. Any mismatch -- including tampering introduced after an
    earlier call to this same function -- is `BLOCK`. Does not itself
    authorize opening and does not consume `SEPARATE_T1C_RESEARCH_
    OPENING_GATE` / `SEPARATE_T2_RESEARCH_OPENING_GATE`.
    """
    if block not in RESEARCH_OPENING_GATE_BY_BLOCK:
        raise V8CResearchOpeningGuardBlocked("V8C_RESEARCH_OPENING_BLOCK_INVALID")

    try:
        result = resolve_and_verify_acquisition_artifact(output_root, block)
    except V8CAcquisitionArtifactVerificationBlocked as error:
        raise V8CResearchOpeningGuardBlocked("POINT_OF_USE_VERIFICATION_FAILED:" + error.reason) from error

    if result.get("result") != "PASS":
        raise V8CResearchOpeningGuardBlocked("POINT_OF_USE_VERIFICATION_NOT_PASS")

    gate = RESEARCH_OPENING_GATE_BY_BLOCK[block]
    try:
        already_opened = has_gate_been_consumed(
            CANONICAL_CONSUMPTION_STATE_ROOT, gate, EXPECTED_V8C_FROZEN_DESIGN_COMMIT
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CResearchOpeningGuardBlocked(error.reason) from error

    return {
        "result": "PASS",
        "block": block,
        "point_of_use_reverification": "COMPLETE",
        "authorizes_opening": False,
        "consumes_research_opening_gate": False,
        "research_opening_gate_already_consumed": already_opened,
        "no_research_opening_api_exists": no_research_opening_api_exists(),
    }


__all__ = [
    "RESEARCH_OPENING_GATE_BY_BLOCK",
    "V8CResearchOpeningGuardBlocked",
    "no_research_opening_api_exists",
    "verify_point_of_use_before_opening",
]
