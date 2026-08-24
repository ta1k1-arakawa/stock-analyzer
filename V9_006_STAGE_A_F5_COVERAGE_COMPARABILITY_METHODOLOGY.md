# V9_006 Stage-A F5 coverage and comparability methodology

```text
task=V9_006_STAGE_A_F5_COVERAGE_AND_COMPARABILITY_METHODOLOGY_BINDING
status=AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

F5 is `SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS`, with existing
`MONTHLY` slot kind, `auxiliary=true`, and `LISTING_CO_ROOT_URL`. Its base
months are exactly 2017-01 through 2025-12.

## 1. Support and monthly evidence identity

The exact F5 root response is a `RAW_PROVENANCE_OBJECT` support object, not
coverage evidence. It is owned by F5, has applicable period exactly
`LISTING_CO_DISCOVERY_ROOT`, and requested URL exactly `LISTING_CO_ROOT_URL`.
It is fetched/locked at most once per raw key; traversal/parser repair reuses
locked bytes. Any consumed intermediate official archive, index, or year page
is likewise support-only unless separately bound as monthly evidence. Support
IDs never populate F5 coverage.

For a base `YYYY-MM`, traversal proceeds only through mechanically resolved
official same-domain links from locked support bytes. Month identity must be
unique; relative links resolve against the final `resolved_url` of the locked
page containing the link. There is no archive-N, filename/suffix, provider,
mirror, or manual choice. The final exact monthly publication is the F5
`COVERAGE_EVIDENCE_OBJECT`, with F5 owner, exact `YYYY-MM` applicable period,
exact mechanically resolved requested URL, and the existing raw-lock key as
its `source_object_slot_id`.

## 2. Coverage status

`AVAILABLE` requires exactly one required official monthly evidence object
that is mechanically resolved and validly locked. `MISSING` applies when an
object is expected under mechanically proven source cadence but is missing,
ambiguous, traversal-invalid, or cannot be validly locked.

`NOT_APPLICABLE_BY_SOURCE_CONTRACT` is permitted only when locked official
JPX material mechanically and explicitly establishes that the cadence/contract
does not expect an F5 object for that exact month. Mere absence of a link or
object is never enough; without positive locked official proof the status is
`MISSING`. Cadence may not be inferred from neighboring months, current site
patterns, filename gaps, or modern publication behavior.

## 3. Comparability is independent

Coverage and `crosscheck_comparable` are independent: an `AVAILABLE` F5 cell
may be non-comparable. Comparability is true only when locked official source
structure/content mechanically proves the exact monthly F5 count scope equals
the frozen V9 reconstructed scope: TSE domestic ordinary common stocks under
the frozen V9 universe/security-type definition. Otherwise it is false and
the month is `CROSSCHECK_NOT_AVAILABLE`; this alone neither passes nor fails
Stage A.

No approximation is allowed. “Listed companies/issues” wording, close totals,
current scope, apparently small exclusions, or another month's comparability
do not prove company-count/security-count scope equivalence. Comparability is
never decided from agreement/disagreement with the reconstructed count;
outcomes are assessed only after independent comparability proof.

Any bytes consumed to prove equivalence are locked `RAW_PROVENANCE_OBJECTS`.
When an extra scope document is needed it remains support-only and cannot
replace the monthly F5 coverage ID. Only `AVAILABLE` and comparable months
participate in the existing exact `MONTH_END_CROSSCHECK`; no tolerance,
threshold, favorable subset, or manual exclusion is introduced.

## 4. Scope preserved

No F5 acquisition implementation is authorized here. This changes none of
F1/F2/F3/F4/F6/F7, the 648-cell matrix, F2 bridge, F7 envelope, retry,
redirect, raw provenance, semantic reconstruction, thresholds, periods,
human gates, or design freeze. No real network is authorized.

```text
REVIEWED_SHA=a9168df38b793525a56aef60699e0ece8e804c7e
PARENT_SHA=3b55ca8f34b6a2d9ccc565ad1cea25228a363e0f
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F3_ACQUISITION=PASS
```
