"""`READ_ONLY_T2_REUSE_CONDITIONS_RECHECK` (§12.4, §9, §3.3).

Repository-safe recheck of `T2`'s preservation conditions using only safe
metadata the caller already holds (audit/state flags, committed
hashes/counts) -- this module never reads, accepts, or exposes a `T2`
ticker identity, and performs no I/O of its own. Any condition failing is
`V8B_T2_PRESERVATION_RECHECK_BLOCKED`: there is no silent `T_spare`/`T3`
substitution path -- this module defines no fallback block-selection
function of any kind.
"""

from __future__ import annotations

from typing import Any, Mapping


class V8BT2PreservationRecheckBlocked(RuntimeError):
    """Fail-closed §9/§12.4 `T2` reuse-condition recheck error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


REQUIRED_SAFE_METADATA_FIELDS = (
    "t2_acquired",
    "t2_opened",
    "t2_ticker_identities_exposed_to_human_public_research_loop",
    "t2_market_data_raw_ohlcv_feature_outcome_research_exposure",
    "t2_universe_definition_unchanged",
    "t2_partition_algorithm_unchanged",
    "t2_v8b_f1_c1_policy_fixed",
)


def recheck_t2_reuse_conditions(safe_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Verify §3.3/§9's T2 preservation conditions from safe metadata only.

    ``safe_metadata`` must supply every field in
    ``REQUIRED_SAFE_METADATA_FIELDS`` -- absence of evidence is never
    treated as an implicit PASS (mirroring §12.2's identical rule for the
    T_spare/T2/T3 preservation recheck).
    """
    if not isinstance(safe_metadata, Mapping):
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:SAFE_METADATA_INVALID")
    missing = set(REQUIRED_SAFE_METADATA_FIELDS) - set(safe_metadata)
    if missing:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA")

    checks_expect_false = (
        "t2_acquired",
        "t2_opened",
        "t2_ticker_identities_exposed_to_human_public_research_loop",
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure",
    )
    for field in checks_expect_false:
        if safe_metadata[field] is not False:
            raise V8BT2PreservationRecheckBlocked(
                "V8B_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper()
            )

    checks_expect_true = (
        "t2_universe_definition_unchanged",
        "t2_partition_algorithm_unchanged",
        "t2_v8b_f1_c1_policy_fixed",
    )
    for field in checks_expect_true:
        if safe_metadata[field] is not True:
            raise V8BT2PreservationRecheckBlocked(
                "V8B_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper()
            )

    return {"result": "PASS", "block": "T2"}


__all__ = [
    "REQUIRED_SAFE_METADATA_FIELDS",
    "V8BT2PreservationRecheckBlocked",
    "recheck_t2_reuse_conditions",
]
