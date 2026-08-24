# V9_006 Stage-A source-slot locator methodology

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
task=V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY_BINDING
methodology_authority=GPT-5.6_Sol
document_role=PREFREEZE_METHODOLOGY_REFINEMENT_RECORD
network_authorized_by_this_task=false
v9_design_frozen=false
```

This records, exactly as decided by GPT methodology authority, a PREFREEZE
refinement of `V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md`'s
Stage-A source-slot semantics. It is not a V9 design freeze and does not
create network, data, T1, or design-freeze authority. The execution agent
records this methodology exactly; it does not extend, reinterpret, or
implement it in this task.

## Two-layer model (V9_006_SOURCE_SLOT_LOCATOR_HIGH_1)

Stage-A's required inventory is heterogeneous: a source object need not
have one distinct URL per calendar month. This methodology therefore
defines two distinct layers, and every coverage record must mechanically
reference zero or more `SOURCE_OBJECT_INVENTORY` slot IDs.

### 1. SOURCE_OBJECT_INVENTORY

The actual, uniquely fetched/locked JPX objects. Each has exactly one of
these slot kinds:

```text
MONTHLY
YEAR
TERMINAL
GLOBAL
```

No other slot kind is permitted. A source object is fetched/locked once
only. A `YEAR` or `GLOBAL` object may cover multiple reconstruction months
when its official semantics mechanically prove that coverage; reusing that
same object's reference across months never means refetching it.

### 2. MONTHLY_COVERAGE_MATRIX

This preserves the exact V9_005 requirement unchanged: 7 source families ×
every month 2017-01 through 2025-12 = 756 records. Each record's status
remains exactly one of:

```text
AVAILABLE
NOT_APPLICABLE_BY_SOURCE_CONTRACT
MISSING
```

No fourth status. `NOT_APPLICABLE_BY_SOURCE_CONTRACT` may only be used
where the official source cadence/range itself mechanically proves no
monthly object is expected for that cell. Unknown or ambiguous always
remains `MISSING`. Every `AVAILABLE` record mechanically references the
`SOURCE_OBJECT_INVENTORY` slot ID(s) that support it.

## F1 -- LISTED_ISSUES_MONTH_END

```text
slot_kind=TERMINAL
root=https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html
```

Resolve the unique same-domain `data_j.xls` link from that official page.
The page/file is rolling previous-month-end material, not a historical
monthly archive. Parse the terminal snapshot month `T` from official source
semantics. No guessed historical URL.

**Monthly coverage mapping.** All 108 base `MONTHLY_COVERAGE_MATRIX` cells
for this family are `NOT_APPLICABLE_BY_SOURCE_CONTRACT`, because the
official source is mechanically rolling terminal-only, not a historical
monthly archive -- no monthly object is ever expected for this family. One
separate `TERMINAL` object remains mandatory outside that matrix: an
absent or ambiguous terminal object sets `terminal_snapshot_pass=false`.
Terminal month `T` must be parsed mechanically from the object itself.

## F2 -- MONTHLY_STATISTICS_CHANGES_REPORT

```text
slot_kind=MONTHLY
root=https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html
```

Required 2017-01 through 2025-12, plus post-2025 months needed to reverse
from the terminal snapshot month `T` back to 2025-12. Discover the year
page only through the official archive selector from root, then the
semantic row label "Changes in Listed Companies and Issues, Etc.", then the
requested month column, then the unique same-domain linked object. Never
hardcode archive-N numbering.

**Monthly coverage mapping.** Each 2017-01 through 2025-12 cell is
`AVAILABLE` if and only if its exact official monthly object is uniquely
resolved and locked, referencing that object's `SOURCE_OBJECT_INVENTORY`
slot ID; otherwise `MISSING`. The post-2025 months required to reverse from
terminal month `T` back through 2025-12 are additional mandatory
`SOURCE_OBJECT_INVENTORY` slots outside the 756-record base matrix, not
extra base-matrix cells. Any missing required bridge month sets
`listing_transition_pass=false`.

## F3 -- DELISTED_COMPANY_ARCHIVE

```text
slot_kind=YEAR
root=https://www.jpx.co.jp/english/listing/stocks/delisted/index.html
```

Discover year pages only through the official archive selector. One locked
year object may cover its months. Use as official corroborating evidence;
conflicting authoritative objects fail under the existing V9_005 conflict
rule. No manual reconciliation.

**Monthly coverage mapping.** Required years 2017 through 2025. If one
unique locked `YEAR` object mechanically proves coverage of that object's
complete calendar year, map that same object's `SOURCE_OBJECT_INVENTORY`
slot ID to all 12 monthly cells of that year and mark them `AVAILABLE`.
Otherwise the affected cells are `MISSING`. Never refetch one year object
12 times to populate its 12 monthly cells.

## F4 -- EX_RIGHTS_SPLIT_RATIO_ARCHIVE

```text
slot_kind=MONTHLY
```

Use the same Monthly Statistics root/traversal as F2, but the semantic row
"Ex-New, Ex-Rights, Etc.", required 2017-01 through 2025-12. For a pure
split/consolidation displayed as an exact ratio `a:b`, the canonical V9
ratio is:

```text
R_e = post_action_shares / pre_action_shares = b / a
```

reduced to an exact positive rational. No float tolerance.

**Monthly coverage mapping.** Each 2017-01 through 2025-12 cell is
`AVAILABLE` if and only if its exact monthly object is uniquely resolved
and locked, referencing that object's `SOURCE_OBJECT_INVENTORY` slot ID;
otherwise `MISSING`.

## F5 -- MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS

```text
slot_kind=MONTHLY
auxiliary=true
root=https://www.jpx.co.jp/english/listing/co/index.html
```

`MONTHLY_AUXILIARY` is not a permitted `slot_kind` value; this family uses
`slot_kind=MONTHLY` with a separate `auxiliary=true` marker instead. Traverse
official archive links semantically; never use guessed archive-N URLs.

**Monthly coverage mapping.** Coverage and crosscheck comparability are two
separate booleans:

- If the official monthly object exists and is locked, the cell is
  `AVAILABLE`, regardless of whether its scope is comparable to V9 -- this
  is unconditional on comparability.
- A separate `crosscheck_comparable` boolean governs whether
  `MONTH_END_CROSSCHECK` may run for that month: `crosscheck_comparable`
  is true only when the source scope can be mechanically proven identical
  to the reconstructed V9 domestic-ordinary-common security scope for that
  month. When `crosscheck_comparable=false`:

```text
CROSSCHECK_NOT_AVAILABLE
```

  which alone neither passes nor fails.
- If official cadence mechanically proves no object is expected for that
  month, the cell is `NOT_APPLICABLE_BY_SOURCE_CONTRACT`.
- If an object is expected but missing or ambiguous, the cell is
  `MISSING`.

Never compare company-count and security-count scopes unless exact scope
equivalence is mechanically proven.

## F6 -- TOPIX_HISTORICAL_INDEX_VALUE

```text
slot_kind=GLOBAL
root=https://www.jpx.co.jp/english/markets/indices/topix/
```

Resolve the unique same-domain object under the semantic section
"Historical Index Value". Require structural coverage of calendar years
2017 through 2025. No 108 distinct monthly URLs, and no printing of index
values.

**Monthly coverage mapping.** One unique `GLOBAL` object is mandatory. If
it structurally covers every calendar year 2017 through 2025, all 108
monthly coverage cells are `AVAILABLE` and reference that same object's
`SOURCE_OBJECT_INVENTORY` slot ID. If a calendar year is missing from that
structural coverage, all 12 cells for that year are `MISSING`. Never fetch
the global object repeatedly.

## F7 -- JPX_CALENDAR

```text
slot_kind=MONTHLY
locator_template=https://www.jpx.co.jp/calendar/{YYYY}{MM:02d}.html
required_acquisition_envelope=2016-09_through_2026-03_inclusive
```

This is a GPT-bound locator template, not an execution-agent guess. The
required fixed acquisition envelope (2016-09 through 2026-03 inclusive) is
not itself the V9 endpoint: `FINAL_SIGNAL_D0` and the exit tail remain
mechanically derived from the locked official JPX calendar under the
already-bound V9 signal-grid rule (`V9_005_HIGH_2B_REVIEW.md`,
`BOUND_SIGNAL_GRID_BLOB_SHA=9135183b7fc5097602fa40fcda8f1b0448220244`),
unchanged by this document.
Missing or ambiguous classification for any required calendar date within
the envelope fails Stage A. No national-holiday library or inferred
weekday-only substitute may replace the locked official material.

**Monthly coverage mapping.** Each 2017-01 through 2025-12 cell is
`AVAILABLE` if and only if the exact `YYYYMM` page is locked, referencing
that object's `SOURCE_OBJECT_INVENTORY` slot ID; otherwise `MISSING`.
2016-09 through 2016-12 and 2026-01 through 2026-03 are additional
mandatory calendar object slots outside the 756-record base matrix. Any
missing or ambiguous required envelope month sets
`trading_calendar_pass=false`. The existing signal-grid endpoint
derivation is unchanged.

## Locator rules (apply to F1-F7)

- `jpx.co.jp` and subdomains only.
- No search engine or off-domain discovery during execution.
- Semantic official-link traversal only, starting from the bound roots
  above.
- Zero or multiple candidate links for a required slot resolve to
  `MISSING`/`FAIL`; never choose manually after seeing results.
- The existing V9_005 raw first-complete-payload/provenance requirements
  remain unchanged (requested URL, resolved URL, HTTP status, retrieval
  timestamp, byte length, SHA-256, source family, applicable period; no
  silent overwrite; parser/semantic repair reprocesses the same locked
  bytes, never a refetch).

## What this task does not decide

This task does not decide retry count or backoff policy. The existing
V9_006 implementation's invented retry policy (`MAX_ATTEMPTS`,
`MAX_RETRIES`, `BACKOFF_SECONDS` in `src/v9_005_stage_a_jpx_probe.py`)
remains unresolved by this methodology binding, and real Stage-A execution
stays `BLOCK`ed pending that separate decision. This task also does not
implement any of F1-F7 in code, does not execute the probe, does not make
any network request, and does not consume any human authorization,
including the Stage-A authorization already given in chat.

## Next action

`GPT_EXACT_SHA_V9_006_SOURCE_SLOT_LOCATOR_HIGH_1_REVIEW`: obtain GPT's
independent exact-SHA review of this HIGH-1 remediation (see
`V9_006_SOURCE_SLOT_LOCATOR_HIGH_1_REVIEW.md`). A future, separately
authorized implementation task would then wire F1-F7 -- with the
`SOURCE_OBJECT_INVENTORY` / `MONTHLY_COVERAGE_MATRIX` two-layer model --
into `src/v9_005_stage_a_jpx_probe.py`'s `resolve_month_locator` seam
under this exact binding -- still without executing any real network
request until a fresh, separate, explicit Stage-A human network
authorization is obtained after that implementation's own GPT exact-SHA
review PASS.
