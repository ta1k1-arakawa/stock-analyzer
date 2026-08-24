# V9_006 source-slot locator HIGH-1 review

```text
REVIEWED_SHA=667cdad05f6961835b254e0d77ce2cbd5ebeea0e
PARENT_SHA=9d87f1110d9baf3746851040383f42c61d743394
CRITICAL=0
HIGH=1
MEDIUM=0
RESULT=BLOCK
```

FINDING=V9_006_SOURCE_SLOT_LOCATOR_HIGH_1_OBJECT_TO_MONTHLY_COVERAGE_MAPPING_UNBOUND

`V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md` bound heterogeneous
`slot_kind` values (`MONTHLY`, `YEAR`, `TERMINAL`, `GLOBAL`) and exact
roots/traversal semantics for the seven Stage-A source families, but never
mechanically defined how a fetched/locked source object maps onto the
existing V9_005 756-record `MONTHLY_COVERAGE_MATRIX` (7 families × every
month 2017-01 through 2025-12). Without that mapping, a `YEAR`/`GLOBAL`
object's coverage of multiple months, F1's rolling-terminal-only semantics,
and F5's coverage-versus-comparability distinction were all left
underspecified -- leaving room for an execution agent to invent the
mapping rule itself, which is prohibited.

FINDING_STATUS=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW

## Remediation implemented

`V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md` now defines two
distinct, explicit layers:

1. **`SOURCE_OBJECT_INVENTORY`** -- the actual, uniquely fetched/locked
   JPX objects, each with exactly one `slot_kind` in
   `{MONTHLY, YEAR, TERMINAL, GLOBAL}` (no other value permitted; the
   previously used `MONTHLY_AUXILIARY` value is removed). A source object
   is fetched/locked once only.
2. **`MONTHLY_COVERAGE_MATRIX`** -- the exact, unchanged V9_005 756-record
   grid, each record status exactly one of `AVAILABLE`,
   `NOT_APPLICABLE_BY_SOURCE_CONTRACT`, or `MISSING` (no fourth status),
   with every `AVAILABLE` record mechanically referencing the
   `SOURCE_OBJECT_INVENTORY` slot ID(s) that support it.
   `NOT_APPLICABLE_BY_SOURCE_CONTRACT` may only be used where the official
   source cadence/range itself mechanically proves no monthly object is
   expected; unknown or ambiguous always remains `MISSING`.

Each of F1-F7 now has an explicit "Monthly coverage mapping" subsection:

- **F1** (`TERMINAL`): all 108 base cells are
  `NOT_APPLICABLE_BY_SOURCE_CONTRACT` (rolling terminal-only source, never
  a historical monthly archive); one separate mandatory `TERMINAL` object
  outside the matrix; absent/ambiguous sets `terminal_snapshot_pass=false`;
  terminal month `T` parsed mechanically.
- **F2** (`MONTHLY`): each cell `AVAILABLE` iff its exact monthly object is
  uniquely resolved/locked, else `MISSING`; post-2025 bridge months (to
  reverse terminal month `T` back through 2025-12) are additional mandatory
  slots outside the 756-record base matrix; a missing required bridge
  month sets `listing_transition_pass=false`.
- **F3** (`YEAR`): required years 2017-2025; one unique locked `YEAR`
  object that mechanically proves coverage of its complete calendar year
  maps that same object's slot ID to all 12 monthly cells as `AVAILABLE`;
  otherwise the affected cells are `MISSING`; never refetch a year object
  12 times.
- **F4** (`MONTHLY`): each cell `AVAILABLE` iff its exact monthly object is
  uniquely resolved/locked, else `MISSING` (ratio orientation unchanged).
- **F5** (`MONTHLY`, `auxiliary=true` -- `MONTHLY_AUXILIARY` removed):
  coverage and crosscheck comparability are now two separate booleans --
  an existing locked monthly object makes the cell `AVAILABLE`
  unconditionally on comparability; a separate `crosscheck_comparable`
  boolean (true only when exact scope equivalence to the reconstructed V9
  scope is mechanically proven) governs `CROSSCHECK_NOT_AVAILABLE`, which
  alone neither passes nor fails; official cadence proving no object
  expected gives `NOT_APPLICABLE_BY_SOURCE_CONTRACT`; expected-but-missing
  gives `MISSING`; company-count and security-count scopes are never
  compared without proven exact equivalence.
- **F6** (`GLOBAL`): one unique mandatory object; if it structurally covers
  every calendar year 2017-2025, all 108 monthly cells are `AVAILABLE`
  referencing that one object; a missing calendar year makes its 12 cells
  `MISSING`; never fetched repeatedly.
- **F7** (`MONTHLY`): each cell `AVAILABLE` iff the exact `YYYYMM` page is
  locked, else `MISSING`; the envelope months outside 2017-2025
  (2016-09..2016-12, 2026-01..2026-03) are additional mandatory slots
  outside the base matrix; a missing/ambiguous required envelope month
  sets `trading_calendar_pass=false`; the signal-grid endpoint derivation
  is unchanged.

## Scope discipline

No retry policy, URL, source root, F4 ratio orientation, calendar
envelope, V9 design element, pass threshold, or other previously bound
methodology was changed. This remediation only adds the explicit
object-to-monthly-coverage mapping layer that was missing.

## Authority created

```text
NETWORK_REQUESTS=0
DATA_ACQUIRED=false
CODE_CHANGED=false
PROBE_EXECUTED=false
HUMAN_GATE_CONSUMED=false
RETRY_POLICY_DECIDED=false
T1_OR_DESIGN_FREEZE_AUTHORITY_CREATED=false
```

This remediation is a docs-only methodology clarification. It does not
authorize network access, data acquisition, T1 membership generation or
opening, model fitting, backtesting, profit calculation, or V9 design
freeze, and does not consume the human's existing chat-given Stage-A
authorization.

## Next action

`V9_006_SOURCE_SLOT_LOCATOR_HIGH_1` remains
`REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW` -- not `PASS` or `RESOLVED`
-- until GPT independently reviews this remediation at its exact commit
SHA. `V9_006_SOURCE_SLOT_LOCATOR_METHODOLOGY` remains `BLOCK` overall
pending that review. Real Stage-A execution stays `BLOCK`ed regardless,
also pending a separate retry/backoff policy decision, implementation of
F1-F7, and a fresh Stage-A human network authorization.
