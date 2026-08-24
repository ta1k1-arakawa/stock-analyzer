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

## Heterogeneous slot kinds

Stage-A's required inventory is heterogeneous: a source object need not
have one distinct URL per calendar month. Required slot kinds:

```text
MONTHLY
YEAR
TERMINAL
GLOBAL
```

A `YEAR` or `GLOBAL` object may cover multiple reconstruction months when
its official semantics mechanically prove that coverage. The same object
must not be fetched repeatedly merely to populate monthly coverage.

## F1 -- LISTED_ISSUES_MONTH_END

```text
slot_kind=TERMINAL
root=https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html
```

Resolve the unique same-domain `data_j.xls` link from that official page.
The page/file is rolling previous-month-end material, not a historical
monthly archive. Parse the terminal snapshot month `T` from official source
semantics. No guessed historical URL.

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

## F3 -- DELISTED_COMPANY_ARCHIVE

```text
slot_kind=YEAR
root=https://www.jpx.co.jp/english/listing/stocks/delisted/index.html
```

Discover year pages only through the official archive selector. One locked
year object may cover its months. Use as official corroborating evidence;
conflicting authoritative objects fail under the existing V9_005 conflict
rule. No manual reconciliation.

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

## F5 -- MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS

```text
slot_kind=MONTHLY_AUXILIARY
root=https://www.jpx.co.jp/english/listing/co/index.html
```

Traverse official archive links semantically; never use guessed archive-N
URLs. Only perform the `MONTH_END_CROSSCHECK` when the source scope can be
mechanically proven identical to the reconstructed V9 domestic-ordinary-
common security scope for that month. Otherwise:

```text
CROSSCHECK_NOT_AVAILABLE
```

That alone neither passes nor fails. Never compare unlike company-count and
security-count scopes.

## F6 -- TOPIX_HISTORICAL_INDEX_VALUE

```text
slot_kind=GLOBAL
root=https://www.jpx.co.jp/english/markets/indices/topix/
```

Resolve the unique same-domain object under the semantic section
"Historical Index Value". Require structural coverage of calendar years
2017 through 2025. No 108 distinct monthly URLs, and no printing of index
values.

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
`BOUND_SIGNAL_GRID_BLOB_SHA=9135183b7fc5097602fa40fcda8f1b0448220244`).
Missing or ambiguous classification for any required calendar date within
the envelope fails Stage A. No national-holiday library or inferred
weekday-only substitute may replace the locked official material.

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

`GPT_EXACT_SHA_V9_006_SOURCE_SLOT_LOCATOR_METHODOLOGY_REVIEW`: obtain GPT's
independent exact-SHA review of this methodology record. A future,
separately authorized implementation task would then wire F1-F7 into
`src/v9_005_stage_a_jpx_probe.py`'s `resolve_month_locator` seam under this
exact binding -- still without executing any real network request until a
fresh, separate, explicit Stage-A human network authorization is obtained
after that implementation's own GPT exact-SHA review PASS.
