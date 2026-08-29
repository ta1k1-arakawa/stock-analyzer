# V9_006 F6 required-period coverage successor parser design

```text
status=AWAITING_GPT_REVIEW
scope=SUCCESSOR_PARSER_DESIGN_ONLY
historical_evidence_class=DEVELOPMENT_EVIDENCE
confirmation_debt=true
remediation_status=MEDIUM_1_MEDIUM_2_REMEDIATED_AWAITING_GPT_REVIEW
```

## Binding and unchanged safeguards

This successor reuses the exact locked official F6 CHILD, inherited Phase A/B
provenance and integrity checks, expected structural-profile SHA, reviewed
structural inspector/safe validator, DATE columns `[4,6]`, deterministic
`xlrd` opening/decoding, and cell-type-before-value discipline. It permits no
network, path-based alternate input, refetch/provider substitution, union,
preferred column, one-column fallback, neighboring-year inference, or
interpolation. Inherited `CHATGPT_DECISION_REQUIRED` and phase-total
fail-closed provenance remain unchanged.

## Required-period rule

Let `R={2017,...,2025}`. For each frozen DATE column, mechanically derive
the complete deterministic ascending year histogram from the same verified
object, retaining the reviewed lineage's per-column `DATE`-count
cross-validation against same-run, hash-verified structural evidence. Let
`year_set_col4` and `year_set_col6` be the corresponding histogram year sets.
Then:

```text
covered_required_years = sorted(R intersect year_set_col4 intersect year_set_col6)
missing_required_years = sorted(R minus covered_required_years)
```

For each required year, presence in both columns is `COVERED`; absence in
either is `MISSING` for that year only. No full-history histogram or count
equality is required to decide required-period membership or acceptance.

The bounded nonfatal diagnostic is derived exactly as follows:

```text
out_of_scope_histogram_col4 =
  deterministic column-4 histogram filtered to years not in R
out_of_scope_histogram_col6 =
  deterministic column-6 histogram filtered to years not in R
out_of_scope_disagreement =
  (out_of_scope_histogram_col4 != out_of_scope_histogram_col6)
```

The diagnostic comparison includes both year membership and each year count.
It MUST NOT affect required-period status selection or
`coverage_result_accepted`. Structural/source identity ambiguity that prevents
required-year membership remains fail-closed.

## Results and fan-out

After required-year membership is deterministically derived, success statuses
are selected only by `missing_required_years`:

```text
missing_required_years == []  => SUCCESSOR_REQUIRED_PERIOD_COVERAGE_CAPTURED
missing_required_years != []  => SUCCESSOR_REQUIRED_PERIOD_COVERAGE_PARTIAL
```

These two exhaustive predicates admit no successful required-membership input
that satisfies both or neither.

For both success statuses, `coverage_evaluated=true` and
`coverage_result_accepted=true`; `covered_required_years` and
`missing_required_years` are exact sorted complements within `R`; and
`all_required_years_covered == (missing_required_years == [])`.
Consequently, `CAPTURED` requires `all_required_years_covered=true` and an
empty missing list, while `PARTIAL` requires
`all_required_years_covered=false` and a nonempty missing list. Their domains
are disjoint. A missing required year remains claim-level `MISSING` for only
that year's twelve F6 cells and does not automatically terminate V9.

`IMPLEMENTATION_FAILURE` and inherited `CHATGPT_DECISION_REQUIRED` occur only
where required-year membership cannot be reliably determined or an existing
integrity boundary fails, never merely because a required year is missing.
`CHATGPT_DECISION_REQUIRED` remains reachable only through the already-
reviewed inherited Phase A/B checks; it is not a newly invented successor
derivation status. The historical `F6_YEAR_COVERAGE_AMBIGUOUS` label remains
immutable old-rule history and is not an output status of this successor.
There is no successor `AMBIGUOUS` result for a merely missing required year or
for an out-of-scope diagnostic; an inability to determine required membership
is instead handled only at the applicable inherited Phase A/B boundary or as
`IMPLEMENTATION_FAILURE`.

Each covered required year fans the existing GLOBAL slot ID to its twelve
months. Each missing required year leaves exactly those twelve months
`MISSING`; no monthly object or refetch is created.

## Closed successor safe-evidence contract

The future successor implementation MUST use a dedicated safe validator; it
MUST NOT reuse the old coverage validator unchanged because that validator's
old status set and full-history-equality acceptance rule are incompatible with
this successor. The reviewed structural inspector and `_safe_structural_evidence`
may be reused unchanged.

For a successor-stage evidence object, the exact allowed stage statuses are
`SUCCESSOR_REQUIRED_PERIOD_COVERAGE_CAPTURED`,
`SUCCESSOR_REQUIRED_PERIOD_COVERAGE_PARTIAL`, and
`IMPLEMENTATION_FAILURE`. Inherited pre-successor `CHATGPT_DECISION_REQUIRED`
is forwarded under its existing Phase A/B contract, not normalized into a
successor result.

Every successor-stage safe evidence object has only the following
always-present fields, with exact types: `status` (one allowed status),
`structural_profile_sha256` (either `null` or a 64-character lowercase-hex
string), `structural_profile_hash_verified` (bool),
`date_column_ordinals` (the exact list `[4,6]` of non-bool integers),
`raw_bytes_read_for_integrity` (bool or the inherited exact string
`"unknown"`), `child_content_inspected` (bool), `date_year_value_read`
(bool), `coverage_evaluated` (bool), `coverage_result_accepted` (bool), and
`network_request_count` (the exact non-bool integer `0`). No path, URL,
exception text, sheet/header text, serial, full date, row, or cell value is a
safe field.

The inherited structural-hash three-state rule remains exact: before hashing,
`structural_profile_sha256=null` and
`structural_profile_hash_verified=false`; a computed mismatch reports its
actual non-expected 64-hex hash with verified false; and only the expected
structural hash with verified true permits DATE-value processing. Expected hash
with verified false is invalid.

The phase-provenance booleans are not inferred from `status`:
`coverage_result_accepted` is true exactly for the two success statuses and
false for `IMPLEMENTATION_FAILURE`; `coverage_evaluated` becomes true only
after both complete histograms have passed their individual DATE-count
cross-validations and required-year membership has been determined, and it
remains true if a later failure is converted to `IMPLEMENTATION_FAILURE`.
Before that boundary it is false. The other inherited phase booleans retain
their actual reached boundaries, including a DATE-value read that remains true
after any later failure.

For either success status, `year_histograms` is required and is exactly a map
with keys `"4"` and `"6"`. Each value is an ascending, unique list of entries
with exactly `{year,count}`, where both values are non-bool integers and each
count is at least one. Each histogram's count sum MUST equal that exact
column's DATE count in the same-run hash-verified structural evidence. The
success object also requires `out_of_scope_histogram_col4`,
`out_of_scope_histogram_col6`, and `out_of_scope_disagreement`: the first two
are exact filtered copies of their respective complete histograms retaining the
same entry schema/order, and the last is their exact boolean inequality.

For either success status, the only required coverage-derived fields are
`covered_required_years`, `missing_required_years`, and
`all_required_years_covered`. The two lists contain ascending unique non-bool
integer years and MUST equal the exact expressions in the Required-period rule;
the boolean MUST equal `(missing_required_years == [])`. `CAPTURED` additionally
requires the missing list to be empty and the boolean true; `PARTIAL`
additionally requires the missing list to be nonempty and the boolean false.
Neither out-of-scope histogram inequality nor any out-of-scope count can alter
these requirements or acceptance.

For `IMPLEMENTATION_FAILURE`, existing phase-total, fail-closed provenance is
preserved: `covered_required_years`, `missing_required_years`,
`all_required_years_covered`, `out_of_scope_histogram_col4`,
`out_of_scope_histogram_col6`, and `out_of_scope_disagreement` are absent.
`year_histograms` is absent unless both complete histograms were already
independently validated against the same-run structural DATE counts before a
later failure and the inherited phase-total failure contract permits their
presence; when present it must meet the same exact histogram schema and count
cross-validation as a success status. No other conditional field is permitted
on failure. No histogram, diagnostic, or derived field may be fabricated.
Malformed, unhashable, unexpected, or internally inconsistent nested evidence
MUST be rejected non-crashingly by the future validator, never silently
normalized or copied to fallback failure evidence.

## Exposure governance

The old `F6_YEAR_COVERAGE_AMBIGUOUS` result is immutable. Its exposed
histograms are DEVELOPMENT_EVIDENCE, not confirmatory evidence or retroactive
preregistration of this successor. Any promotion materially relying on this
amended rule requires fresh, forward, or independent confirmation. This design
authorizes no implementation, execution, evaluation, private access, model
fit, backtest, or profitability claim.

## GPT exact-SHA review and remediation record

```text
REVIEWED_SHA=daef99943555f24c912eb232030383d6eb0f80da
PARENT_SHA=b835366fad8dda294b9eaf2f554b08b4dfb8ca75
CRITICAL=0
HIGH=0
MEDIUM=2
LOW=1
RESULT=BLOCK
MEDIUM_1=F6_SUCCESSOR_CAPTURED_AND_PARTIAL_STATUS_DOMAINS_OVERLAP
MEDIUM_2=F6_SUCCESSOR_SAFE_EVIDENCE_AND_OUT_OF_SCOPE_DIAGNOSTIC_NOT_CLOSED_DETERMINISTIC
LOW_1=PROJECT_STATE_CURRENT_STAGE_STILL_POINTS_TO_AMENDMENT_REVIEW
```

MEDIUM_1 is remediated by the exact, mutually exclusive missing-list status
partition in Results and fan-out. MEDIUM_2 is remediated by the closed
successor safe-evidence contract, deterministic filtered-histogram diagnostic,
and retained same-run DATE-count cross-validation. LOW_1 is resolved through
the contemporaneous project-state transition. This design remediation remains
awaiting GPT independent review and does not change the human-authorized
methodology direction, historical old-rule result, evidence class, or
confirmation debt.

## Bounded closure-sweep record

Three direct mechanical ambiguities were resolved without a methodology
change: the status partition is stated as exhaustive after deterministic
membership; the immutable old-rule AMBIGUOUS label is explicitly excluded from
successor outputs; and phase-total failure booleans plus conditional-field
absence/presence are made exact. No new gate, status, data source, coverage
rule, or authority is created.
