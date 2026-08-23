# V9_005 HIGH-2B independent review

```text
REVIEWED_SHA=6f29b674a6962c84324f699c18e94fcd494ec684
CRITICAL=0
HIGH=1
MEDIUM=0
RESULT=BLOCK
```

FINDING=V9_005_HIGH_2B_PREFREEZE_PROBE_ENDPOINT_REFERENCES_NONEXISTENT_FROZEN_V9_SIGNAL_GRID

The Stage-A calendar-lock endpoint derivation in
`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` defined
`FINAL_SIGNAL_D0` as `last_frozen_V9_signal_grid_D0_lte_2025-12-31`. V9's
overall design is not frozen (`V9_design_frozen=false`), so this prefreeze
probe could not mechanically resolve its own calendar-lock endpoint without
either improperly presupposing a V9 design freeze that does not exist, or an
execution agent silently improvising the signal-grid rule -- both prohibited.

FINDING_STATUS=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW

## Remediation implemented

`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` no longer depends
on an overall V9 design freeze for this endpoint calculation. Instead it
binds the probe's `FINAL_SIGNAL_D0` calculation to the exact already-decided
signal-grid rule already recorded in Section 9 ("Training signal-date
cadence") of the following exact path/blob:

```text
BOUND_SIGNAL_GRID_PATH=V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md
BOUND_SIGNAL_GRID_BLOB_SHA=9135183b7fc5097602fa40fcda8f1b0448220244
PREFREEZE_BINDING=true
GLOBAL_V9_FREEZE_REQUIRED=false
```

The probe draft now states explicitly that this is a narrow, mechanical
prefreeze methodology binding to one already-decided rule inside that draft
for the sole purpose of computing this probe's calendar-lock endpoint, and
not a claim that the overall V9 design is frozen, finally reviewed, or
execution-authorized. It creates no design-freeze authority. The bound rule
itself (`j0 = calendar index of the first JPX trading day >= 2018-01-01`;
`D0 at calendar index j iff (j - j0) mod 3 == 0`; `FINAL_SIGNAL_D0 = last
such D0 <= 2025-12-31`) is not reinterpreted or altered -- it is copied
verbatim in mechanical form from the bound blob's already-decided cadence
rule. The resulting calendar date is not hard-coded; it remains a mechanical
derivation over the locked official JPX trading calendar.

A point-of-use contract check was added: before any Stage-A network
boundary, the probe design now requires mechanically verifying that
`V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md` at the exact commit in
use still resolves to Git blob SHA
`9135183b7fc5097602fa40fcda8f1b0448220244`. A mismatch fails closed as
`PROBE_SIGNAL_GRID_CONTRACT_MISMATCH`, stopping before any Stage-A network
boundary; the probe design explicitly prohibits silently binding to a newer
version of that file, and requires a fresh GPT methodology
review/rebinding decision before any further execution.

## Scope discipline

No other Stage-A/Stage-B rule was changed. Preserved unchanged: the JPX
official trading-calendar requirement; `REQUIRED_MARKET_DATES` /
`RETURNED_DATE_SET` exact coverage equality; internal-gap and
unexpected-returned-date failure rules; duplicate-date failure; the
corporate-action orientation (`post_action_shares / pre_action_shares`) and
exact rational normalization; exact corporate-action multiset equality;
zero date tolerance; the no-manual-reconciliation/override rule; the
no-Yahoo retry/replacement/reroll/suffix-substitution rule; and the
requirement for separate fresh Stage-A and Stage-B human authorizations.

## Authority created

```text
NO_NETWORK_REQUEST=true
NO_DATA_ACQUIRED=true
NO_T1_AUTHORITY_CREATED=true
NO_DESIGN_FREEZE_AUTHORITY_CREATED=true
V9_design_frozen=false
network_authorized=false
JQUANTS_PURCHASE_AUTHORIZED=false
```

This remediation changes only the mechanical binding used to compute the
probe's own calendar-lock endpoint. It does not authorize network access,
data acquisition, T1 membership generation or opening, model fitting,
backtesting, profit calculation, or V9 design freeze.

## Next action

`V9_005_HIGH_2B` remains `REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW` --
not `PASS` or `RESOLVED` -- until GPT independently reviews this remediation
at its exact commit SHA.
