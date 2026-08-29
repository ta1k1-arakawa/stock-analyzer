# V9_006 Stage-A F6 date/year coverage parser design

```text
task=V9_006_F6_DATE_YEAR_COVERAGE_PARSER_DESIGN_CHECKPOINT
status=REMEDIATED_AWAITING_GPT_REVIEW
medium_1=V9_006_F6_DATE_YEAR_DESIGN_AMBIGUOUS_POST_EXPOSURE_GOVERNANCE_AND_COVERAGE_PROVENANCE_INCORRECT
medium_1_status=REMEDIATED_AWAITING_GPT_REVIEW
medium_2=V9_006_F6_DATE_YEAR_DESIGN_SAFE_OUTPUT_NOT_PHASE_TOTAL_BEFORE_STRUCTURAL_HASH_EXISTS
medium_2_status=REMEDIATED_AWAITING_GPT_REVIEW
scope=DATE_YEAR_COVERAGE_PARSER_DESIGN_ONLY_NO_IMPLEMENTATION_NO_EXECUTION
network_authorized_by_this_task=false
network_executed_by_this_task=false
production_child_read_by_this_task=false
child_content_inspected_by_this_task=false
coverage_evaluated_by_this_task=false
human_authorization_consumed_by_this_task=false
parser_implementation_authorized_by_this_task=false
```

This is a docs-only design for a future, separately reviewed and
separately implemented deterministic date/year coverage parser that
derives the exact F6 covered-year set from the same exact locked CHILD
already `STRUCTURAL_FORMAT_CAPTURED` by the reviewed OLE/BIFF structural
parser (`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md`, design
`PASS`; implementation `PASS` at
`b2fcb56c0e5ace654b638664786229761dc14df8`). It does not implement,
execute, or authorize execution of anything; it defines the exact
deterministic derivation a future, independently reviewed implementation
must follow, and the exact prohibitions that keep the result a mechanical
structural fact rather than a favorable, inferred, or hand-picked outcome.

## 1. Binding and authority boundary

This design inherits, unmodified, the exact CHILD identity/root/integrity
binding already frozen in
`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_DESIGN.md` section 1 and
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md` section 1, and the
exact Phase A (metadata-only locate) / Phase B (content-blind integrity
read) rules already implemented in
`src/v9_006_f6_offline_child_structural_probe.py`. No refetch, no
URL/provider/output-root substitution, no relock, no new human
authorization, no new network access, no second acquisition, and no
coverage determination is granted by this design. The existing F6
production raw-acquisition gate remains consumed and non-reusable; this
design consumes nothing.

This design additionally binds to the completed, GPT-`PASS`ed OLE/BIFF
structural-parser implementation and its recorded real production
execution:

```text
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPLEMENTATION=PASS (reviewed_sha=b2fcb56c0e5ace654b638664786229761dc14df8)
structural_execution_result=STRUCTURAL_FORMAT_CAPTURED
structural_profile_sha256=4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce
sheet_table_count=1
sheet_ordinal=1
sheet_row_count=86
sheet_column_count=10
sheet_visibility=VISIBLE
sheet_object_type=WORKSHEET
date_bearing_column_ordinals=[4, 6]
recorded_date_cell_count_column_4=19
recorded_date_cell_count_column_6=19
```

These are established, hash-verified structural facts about the exact
locked CHILD, recorded in
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPLEMENTATION_REVIEW.md`'s
real production execution record. This design does not re-derive,
re-select, or contradict them; it only specifies how a future
implementation may mechanically extract calendar years from the two
already-identified date-bearing columns, gated on reproving the exact same
structural shape first (section 2).

## 2. Structural-evidence identity gate (must run before any DATE value read)

**MEDIUM-2 remediation note**
(`V9_006_F6_DATE_YEAR_DESIGN_SAFE_OUTPUT_NOT_PHASE_TOTAL_BEFORE_STRUCTURAL_
HASH_EXISTS`): the original version of this section implied
`structural_profile_sha256` is always a 64-lowercase-hex string, but an
`IMPLEMENTATION_FAILURE` can occur before any hash has actually been
computed (for example, a crash while recomputing or canonicalizing the
structural evidence itself), which would force a future implementation to
either fabricate a hash value or leave the safe schema not phase-total.
This section, and section 10's schema, now freeze two always-present
provenance fields alongside the hash gate itself:

```text
structural_profile_sha256        always present; value is either null, or exactly 64 lowercase hex characters -- never fabricated, never a placeholder
structural_profile_hash_verified always present; bool
```

Before a future implementation may read a single `DATE`-typed cell value,
it MUST:

1. reopen the exact same already-integrity-verified CHILD bytes (the same
   bytes Phase B already proved match `expected_child_sha256`/
   `expected_child_byte_length`) and recompute the full reviewed structural
   evidence, using the exact same reviewed structural inspector already
   implemented in `src/v9_006_f6_offline_child_structural_probe.py`
   (`_default_structural_inspector` followed by `_safe_structural_evidence`,
   unmodified);
2. canonicalize that evidence exactly as:

```python
canonical_bytes = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
structural_profile_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
```

3. compare `structural_profile_sha256` to the frozen expected value:

```text
expected_structural_profile_sha256=4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce
```

A mismatch of any kind -- a different `status`, a different
`cell_type_profiles`/`structural_dimensions` shape, a different key set, or
any other divergence from the exact reviewed structural evidence this
design is bound to -- is `IMPLEMENTATION_FAILURE`. No `DATE` cell value may
be read, and no year may be derived, unless this exact hash comparison
passes first. This is not a courtesy re-check: it is the sole mechanism
that proves, immediately before any value-level read, that the file this
future implementation is about to parse is still exactly the structural
shape the two preregistered date columns (section 3) were identified
against -- never an assumption carried over from a prior run.

### 2.1 `structural_profile_sha256` / `structural_profile_hash_verified`: exact, phase-total states

Exactly three states are reachable for this pair, and a future
implementation MUST NOT fabricate a value outside them:

```text
(a) before a canonical structural hash has been successfully computed:
    structural_profile_sha256 = null
    structural_profile_hash_verified = false
    (reached only if evidence recomputation/canonicalization/hashing itself
    raises before producing a hash value -- status = IMPLEMENTATION_FAILURE,
    date_year_value_read = false, coverage_evaluated = false)

(b) a hash IS computed but does NOT equal the frozen expected value
    4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce:
    structural_profile_sha256 = <the actual recomputed hash>   (never the expected value, never null)
    structural_profile_hash_verified = false
    status = IMPLEMENTATION_FAILURE
    date_year_value_read = false
    coverage_evaluated = false

(c) the actual recomputed hash exactly equals the frozen expected value:
    structural_profile_sha256 = <the verified matching hash>
    structural_profile_hash_verified = true
    the DATE-value extraction stage (sections 3-8) may become reachable
```

No `DATE` cell value may ever be read while `structural_profile_hash_
verified == false` -- state (c) is the only state from which sections 3-8
may proceed. A failure reached before hash computation completes (state a)
or that computes a mismatching hash (state b) must never report the
expected/frozen hash value in place of the actual one, and must never
report `null` once an actual hash value exists.

## 3. Preregistered date-bearing columns: fixed, not selected

The only date-bearing columns this design recognizes are the structurally
preregistered 1-based column ordinals **4** and **6**, both on sheet
ordinal **1** (section 1). A future implementation MUST NOT choose one over
the other, MUST NOT search any other column for date-like content, and
MUST NOT add, drop, or substitute a column ordinal for any reason,
including a differently-shaped-but-plausible re-derivation of the CHILD.
These two ordinals are fixed facts about the one locked CHILD this design
is bound to (section 1), reproven structurally by the gate in section 2 on
every run, never re-selected.

## 4. Frozen reopen semantics

A future implementation reopens the exact same verified bytes using the
identical reviewed `xlrd==2.0.2` open call already frozen in
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md` section 2 -- the
same call already used by the structural-evidence identity gate in section
2 above, not a second, differently-parameterized open:

```python
import xlrd
book = xlrd.open_workbook(
    file_contents=verified_child_bytes,
    formatting_info=True,
    on_demand=False,
    ragged_rows=False,
)
```

No path/filename input, no pandas, no retry, no recovery attempt around
this call, exactly as already frozen. `book.datemode` (xlrd's own
documented 1900/1904 epoch flag for the workbook) is read from this same
`Book` object and used, unmodified, for every date-serial conversion in
section 5 -- never assumed, hardcoded, or independently inferred.

## 5. Per-column DATE-only extraction procedure

**MEDIUM-1 remediation note**: `date_year_value_read` (section 10) is a
single, monotonic provenance boolean -- not per-column -- that starts
`false` and becomes `true` immediately before the first permitted
`sheet.cell_value(...)` call on an `XL_CELL_DATE` cell (the marked line in
the procedure below), across either column. Once `true`, it can never
revert to `false` for the remainder of that run, regardless of what
happens afterward (a later `IMPLEMENTATION_FAILURE` still reports `date_
year_value_read=true` if the boundary was already crossed -- never
fabricated `false`). If a run fails before ever reaching that line (for
example, the section 2 hash gate fails, or the very first column processed
has zero `DATE` cells and its own count cross-validation in section 5.1
fails before any `cell_value` call), `date_year_value_read` remains
`false`, accurately reporting that no `DATE` value was ever read.

For each of the two preregistered date columns (section 3) independently,
on sheet ordinal 1:

```python
for rowx in range(sheet.nrows):
    cell_type = sheet.cell_type(rowx, column_index)   # inspect type FIRST
    if cell_type != xlrd.XL_CELL_DATE:
        continue                                       # no value read for any other type
    # date_year_value_read -> true HERE, immediately before this call, the
    # first time this line is reached across either column; never reverts.
    serial = sheet.cell_value(rowx, column_index)       # ONLY reachable when cell_type == XL_CELL_DATE
    year = xlrd.xldate_as_tuple(serial, book.datemode)[0]   # extract ONLY the integer calendar year
    # increment year_histogram[year] by 1; never store/emit `serial`, the
    # full tuple, or any component other than the integer year
```

This is total and exhaustive: every row `0 .. sheet.nrows - 1` (the exact
`row_count` already established and hash-verified in section 2, `86` for
this CHILD) is inspected via `cell_type` first, for both columns
independently. `cell_value` is read for a given `(row, column)` pair **only
when** `cell_type(row, column) == xlrd.XL_CELL_DATE` -- never for
`EMPTY`/`BLANK`/`TEXT`/`NUMBER`/`BOOLEAN`/`ERROR`, and never speculatively.
`xlrd.xldate_as_tuple(serial, book.datemode)` returns
`(year, month, day, hour, minute, second)`; only index `0` (the integer
calendar year) is ever extracted, stored, or emitted -- month, day, hour,
minute, second, and the raw `serial` value itself are read into a local
variable only insofar as the tuple-unpack/index operation requires, and
are never stored, logged, or emitted in any safe-evidence field. Any
exception raised by `xldate_as_tuple` for a cell already classified
`XL_CELL_DATE` (for example, `xlrd.xldate.XLDateAmbiguous`,
`XLDateNegative`, `XLDateTooLarge`, or `XLDateBadDatemode` for a
malformed/reserved serial) is not itself separately classified by this
design; it is `IMPLEMENTATION_FAILURE`, exactly like every other exception
this design's reviewed lineage does not explicitly classify elsewhere
(`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md` section 2.1's
"any other exception is `IMPLEMENTATION_FAILURE`" rule, reused unchanged,
not reinvented).

### 5.1 Per-column DATE-count cross-validation (fail-closed, not hardcoded)

After enumerating all rows for a given column, the total number of cells
encountered with `cell_type == xlrd.XL_CELL_DATE` MUST exactly equal that
column's own `cell_type_counts.DATE` value inside the same
hash-verified structural evidence recomputed in section 2 (currently `19`
for both column 4 and column 6, per section 1's recorded facts -- but this
rule is stated generically against the hash-verified evidence, never as a
hardcoded literal `19` inside a future implementation, so it remains
correct if this design is ever bound to a different exact CHILD with a
different recorded profile). Any mismatch -- more or fewer `DATE` cells
encountered than the structural evidence already recorded for that exact
column -- is `IMPLEMENTATION_FAILURE`. This proves the value-level
extraction pass agrees with the type-only structural pass that already
hash-gated this run, closing any gap between "the structural profile says
19 DATE cells exist in column 4" and "value-level extraction actually
found 19." `date_year_value_read` at this failure point is `true` if at
least one `cell_value` call for an `XL_CELL_DATE` cell already occurred in
this run (section 5's marked line), and remains accurately `false` only in
the specific edge case where the failing column's own encountered `DATE`
count is `0` while the hash-verified structural evidence recorded `>= 1`
for it, and it is the first column processed -- i.e. the mismatch is
detected without any `cell_value` call ever having happened yet.
`coverage_evaluated` is `false` for any section 5.1 failure: it requires
both columns' complete histograms and the full comparison (section 7), and
a section 5.1 failure always occurs before that point.

## 6. Deterministic per-column year histogram

For each of the two columns independently, a future implementation builds
a deterministic year histogram: a list of `{"year": <int>, "count": <int>}`
entries, one per distinct calendar year encountered in that column's
`DATE` cells (section 5), each `count` equal to the number of `DATE` cells
in that column whose extracted year equals that entry's `year`, sorted
strictly ascending by `year`. Every entry's `count` is `>= 1` by
construction (a year only appears in the histogram if at least one `DATE`
cell in that column produced it); there is no zero-count entry and no
duplicate year within one column's histogram. The sum of every entry's
`count` in a column's histogram MUST equal that column's cross-validated
`DATE` cell count (section 5.1) -- itself already proven equal to the
hash-verified structural evidence's own count. Coverage is never inferred
from the literal count value `19` itself, or from any count value in
isolation; only the **set of distinct years** feeds into the covered-year
derivation (section 8). No row position, first/last entry, or any
ordering fact other than the frozen `year` ascending sort (needed only for
deterministic, canonical output identity) contributes to the covered-year
result.

## 7. No favorable-column selection: cross-column identity requirement

**MEDIUM-1 remediation note**
(`V9_006_F6_DATE_YEAR_DESIGN_AMBIGUOUS_POST_EXPOSURE_GOVERNANCE_AND_
COVERAGE_PROVENANCE_INCORRECT`): the original version of this section said
`AMBIGUOUS` leaves "coverage remains unevaluated/false" and "returns to
GPT for a methodology decision," which incorrectly conflated two different
facts once both columns' real `DATE` values have already been read,
enumerated, and compared: (1) whether the mechanical comparison itself was
performed (it was), and (2) whether a covered-year set was accepted for
production coverage purposes (it was not, and never will be for this
preregistered rule). It also implied a live, open-ended methodology choice
remained available after seeing the histograms, which this remediation
closes.

The complete year histograms for column 4 and column 6 (section 6) MUST be
compared for exact equality -- the same set of `{year, count}` entries, in
the same sorted order, with no entry present in one histogram and absent
or differently-counted in the other.

```text
histogram(column_4) == histogram(column_6)  =>  status = F6_YEAR_COVERAGE_CAPTURED  (coverage_evaluated=true, coverage_result_accepted=true)
histogram(column_4) != histogram(column_6)  =>  status = F6_YEAR_COVERAGE_AMBIGUOUS (coverage_evaluated=true, coverage_result_accepted=false)
```

In both outcomes `coverage_evaluated=true`: both columns were completely
enumerated, both `DATE` counts cross-validated (section 5.1), both
complete histograms constructed (section 6), and the exact equality
comparison performed. `coverage_evaluated` reports only that this
mechanical work happened -- it never reports whether a covered-year set
was accepted. That is `coverage_result_accepted`, which is `true` only for
`F6_YEAR_COVERAGE_CAPTURED`.

If the two histograms differ in any way, **no covered-year set is
accepted**: `F6_YEAR_COVERAGE_AMBIGUOUS` is **terminal** -- for this
preregistered F6 date/year coverage rule, in the current V9 study
identity, this run's `AMBIGUOUS` result ends the matter, not a step toward
one. `V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED` (the separate,
project-state-level flag gating any strategy/backtest use of F6 coverage)
remains `false` after `AMBIGUOUS` -- not because no evaluation attempt was
made, but because that flag specifically means "an accepted covered-year
set exists for production coverage purposes," which `AMBIGUOUS` by
definition never produces. A future implementation MUST NOT resolve a
disagreement between the two columns by union, intersection, majority
vote, nearest match, manual selection, preferring either column by
name/position, or any other heuristic -- exact equality or `AMBIGUOUS`,
with no third option and no silent repair.

### 7.1 Post-`AMBIGUOUS` stopping rule (governance, not a parser mechanic)

After an `F6_YEAR_COVERAGE_AMBIGUOUS` result under this preregistered rule,
for the current V9 study identity:

- do **not** change to a union, intersection, majority-vote,
  preferred-column, or any other manual-selection rule within this study;
- do **not** redraw, reselect, add, or drop date-bearing columns;
- do **not** rerun this stage hoping for a different (favorable) result;
- do **not** refetch, retry with different parameters, or substitute a
  provider/URL.

GPT-5.6 Sol may adjudicate the already-preregistered `AMBIGUOUS` result
itself, any implementation fault that produced it, or how it should be
governance-classified -- but MAY NOT choose a new coverage methodology for
this same, already-exposed V9 study after having seen the two histograms.
Any materially different coverage rule considered after observing
`AMBIGUOUS` requires a separately identified successor study or new
preregistration under this repository's existing governance
(`AI_RESEARCH_EXECUTION_RULES.md`), unless a stricter pre-existing frozen
rule already explicitly authorizes it. This rule exists because a
methodology choice made after seeing the actual result it would apply to
is not a preregistered rule at all -- it is post-hoc selection of a
favorable outcome, exactly what section 9's prohibitions and this design's
whole premise exist to prevent.

## 8. Covered-year derivation (only when the two columns agree)

When, and only when, section 7's equality holds (`status =
F6_YEAR_COVERAGE_CAPTURED`, `coverage_evaluated=true`,
`coverage_result_accepted=true`):

```text
covered_years = sorted(unique(year for every entry in either identical histogram))
```

Required years remain exactly the nine years already frozen in
`V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY.md`:

```text
required_years = {2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025}
covered_required_years  = sorted(covered_years intersect required_years)
missing_required_years  = sorted(required_years minus covered_years)
all_required_years_covered = (missing_required_years == [])
```

A required year is proven covered only because it is a member of
`covered_years` -- which itself exists only when the exact same year
appeared in **both** preregistered date columns' independently derived
histograms (section 7). No required year may be marked covered on the
strength of only one column, a partial match, or any inference beyond this
mechanical set membership. `all_required_years_covered` is a structural
fact about `covered_years` versus the fixed `required_years` set; it is
**not** a coverage PASS/FAIL verdict for any strategy, profitability, or
data-quality purpose (section 9), and grants no such authority (section
9's non-authority rule, restated).

## 9. Explicit prohibitions

A future implementation MUST NOT, under any circumstance, use any of the
following to derive, adjust, or substitute for the mechanical
per-`DATE`-cell year extraction and two-column exact-match rule above:

- row-position, row-count, or row-index inference of any kind;
- first-date, last-date, minimum-date, or maximum-date inference;
- neighboring-year, continuity, or density inference;
- interpolation or extrapolation of any kind;
- inspection of any index/numeric/text cell value for coverage purposes
  (only `cell_type` may ever be inspected for a non-`DATE` cell; a
  `DATE`-typed cell's value may be read only for its year, per section 5);
- inspection of sheet names, header text, or any other cell text;
- refetch, retry-with-different-parameters, or provider/URL substitution
  of any kind;
- changing the required-years set, the preregistered column ordinals, or
  any other frozen parameter of this design after seeing any output it
  produces (no post-hoc adjustment to obtain a more favorable result).

## 10. Safe evidence: closed schema

**MEDIUM-2 remediation note**: this section is rewritten so every key's
presence/value rule is mechanically unambiguous and phase-total (never
requiring a fabricated placeholder) at every reachable status, including
`IMPLEMENTATION_FAILURE` reached before any structural hash exists.

### 10.1 Always-present keys (present at every status, once the coverage-stage wrapper has started)

```text
status                          in {F6_YEAR_COVERAGE_CAPTURED, F6_YEAR_COVERAGE_AMBIGUOUS, IMPLEMENTATION_FAILURE}
structural_profile_sha256        null, or exactly 64 lowercase-hex characters (section 2.1's three-state rule)
structural_profile_hash_verified bool (section 2.1)
date_column_ordinals             fixed value [4, 6] (section 3; non-bool ints >= 1, ascending, exactly two entries)
raw_bytes_read_for_integrity     bool or the literal string "unknown" (inherited Phase A/B/C phase-provenance contract, unchanged)
child_content_inspected          bool (inherited Phase A/B/C phase-provenance contract, unchanged)
date_year_value_read             bool (section 5's remediation note; monotonic, never reverts true -> false)
coverage_evaluated               bool (section 7; true only once both columns fully enumerated/cross-validated/histogrammed and compared)
coverage_result_accepted         bool; true iff status == F6_YEAR_COVERAGE_CAPTURED
network_request_count            fixed 0
```

### 10.2 `year_histograms`: conditionally present

`year_histograms` is present **only after both columns' complete
histograms have actually been built** (section 6) -- therefore:

```text
status == F6_YEAR_COVERAGE_CAPTURED   => year_histograms PRESENT (required)
status == F6_YEAR_COVERAGE_AMBIGUOUS  => year_histograms PRESENT (required)
status == IMPLEMENTATION_FAILURE      => year_histograms ABSENT, UNLESS both complete histograms had already been built (and the comparison already performed) before a later, separately-occurring failure -- in that specific case it is present, exactly as it would be for CAPTURED/AMBIGUOUS
```

When present, `year_histograms` has **exactly** two keys, the string forms
of the two preregistered column ordinals (`"4"` and `"6"`), each mapped to
that column's own histogram list (section 6): a list of objects each with
**exactly** the two keys `year` (non-bool int) and `count` (non-bool int
`>= 1`), sorted strictly ascending by `year`, with no duplicate `year`
within one column's list. When `status == F6_YEAR_COVERAGE_CAPTURED`, the
two histogram lists under `"4"` and `"6"` MUST be identical
element-for-element (section 7's equality, restated as a schema-level
invariant a validator can mechanically check); when `status ==
F6_YEAR_COVERAGE_AMBIGUOUS`, they MUST differ in at least one respect
(otherwise the status itself would be inconsistent with its own evidence).

### 10.3 Status-specific keys

```text
F6_YEAR_COVERAGE_CAPTURED only:
  covered_years                  sorted list of unique non-bool ints
  covered_required_years         sorted list of unique non-bool ints, subset of required_years
  missing_required_years         sorted list of unique non-bool ints, subset of required_years (empty list when all_required_years_covered)
  all_required_years_covered     bool

F6_YEAR_COVERAGE_AMBIGUOUS:
  covered_years / covered_required_years / missing_required_years / all_required_years_covered  ABSENT (never emitted -- AMBIGUOUS accepts no covered-year set, section 7)

IMPLEMENTATION_FAILURE:
  covered_years / covered_required_years / missing_required_years / all_required_years_covered  ABSENT
  coverage_result_accepted = false (10.1's general rule, restated)
  every phase/provenance boolean (structural_profile_hash_verified, date_year_value_read, coverage_evaluated, raw_bytes_read_for_integrity, child_content_inspected) MUST reflect the actual boundary reached at failure time -- never a blanket false, never a fabricated true (sections 2.1, 5, 5.1, 7)
```

No exact date, serial value, sheet name, header string, cell text, URL, or
machine-local path may appear anywhere in this payload, at any nesting
depth, regardless of field name -- the only permitted payload-derived
signal is the closed enum, the hash (or `null`), the two fixed column
ordinals, the bounded year/count integers, the phase/provenance booleans,
and the derived year sets specified above.

This design does not itself extend
`src/v9_006_f6_offline_child_structural_probe.py`'s `_safe_structural_
evidence` validator or invent a new module; it specifies the exact schema
a future, separately implemented and separately reviewed validator for
this new coverage-parser stage must enforce, fail-closed and non-crashing
for arbitrary malformed or unhashable input, mirroring the existing
`_is_allowed_enum_str`/closed-set/cardinality-cross-validation pattern
already reviewed and `RESOLVED` through this repository's MEDIUM-3/
MEDIUM-3A/MEDIUM-1(OLE-BIFF-impl) remediation chain -- never a divergent
validation style.

## 11. Determinism

```text
same_exact_child_bytes_plus_same_reviewed_xlrd_environment => identical structural_profile_sha256, identical year_histograms, identical covered-year derivation
column_enumeration_order                                    = fixed [4, 6], never re-selected or reordered
row_enumeration_order                                        = 0 .. sheet.nrows - 1, ascending, per column, matching xlrd's own indexing
histogram_ordering                                            = year ascending, per column (section 6)
cross_column_comparison                                       = exact equality, no heuristic resolution (section 7)
count_type                                                     = exact nonnegative integers only (non-bool)
unclassified_exception_anywhere_in_this_procedure              = IMPLEMENTATION_FAILURE (never silent mapping, never a guess)
structural_profile_hash_mismatch                               = IMPLEMENTATION_FAILURE, no DATE value read (section 2, checked first, every run)
```

Two runs against the exact same verified CHILD bytes and the exact same
reviewed environment must agree not only on `covered_years` but on every
intermediate value this design defines (`structural_profile_sha256`, both
column histograms in full) and their exact canonical ordering; two runs
that agree on content but disagree on order, or that report different
`year_histograms` for the same bytes, are not identical evidence.

## 12. Safe outcomes and inherited escalation

The safe outcomes this design's own derivation logic may produce are
exactly:

```text
F6_YEAR_COVERAGE_CAPTURED
F6_YEAR_COVERAGE_AMBIGUOUS
IMPLEMENTATION_FAILURE
```

`CHATGPT_DECISION_REQUIRED` is not a new outcome this design invents; it
remains reachable only through the unmodified, inherited Phase A binding
checks (`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_DESIGN.md`
section 1, already implemented in
`src/v9_006_f6_offline_child_structural_probe.py`) that this design's
section 1 preserves unchanged and that necessarily run before this stage's
own logic is ever reached. This design changes no F6 coverage methodology,
required-years set, structural-parser design, Phase A/B boundary, network/
refetch rule, or human-gate rule; it adds no new safe outcome beyond the
three above.

`F6_YEAR_COVERAGE_AMBIGUOUS` is not an intermediate or retriable state: per
section 7.1, it is terminal for this preregistered rule in the current V9
study identity. A future implementation must not offer, and no operator
may invoke, a "rerun with a different rule" or "select a column" mode
after an `AMBIGUOUS` result within this study.

## 13. `F6_YEAR_COVERAGE_CAPTURED` grants no strategy/profitability authority

`F6_YEAR_COVERAGE_CAPTURED` and `all_required_years_covered=true` mean only
that the exact locked CHILD's two preregistered date columns mechanically
agree on the same set of calendar years, and that set covers the nine
required years 2017-2025. This is not a coverage PASS for any strategy or
backtest purpose, not a data-quality verdict, not authorization to begin
model fitting, historical evaluation, or any private/sealed access, and
not authorization to inspect index values or begin any further parsing
stage. `V9_design_frozen`, `V9_historical_evaluation_authorized`, and
`future_profitability_established` remain `false`; this design creates no
authority toward any of them. After a future real execution of the
implementation this design authorizes designing, GPT must independently
review the exact safe coverage evidence produced before any such authority
is even considered.

## 14. Required future implementation-review scope

A future implementation of this design must, at minimum:

- reuse the existing, already-reviewed
  `_default_structural_inspector`/`_safe_structural_evidence` pair from
  `src/v9_006_f6_offline_child_structural_probe.py` unmodified for the
  section 2 identity gate -- never a re-derived or parallel structural
  parser;
- implement the section 2 canonicalization/hash gate exactly, and hard-stop
  `IMPLEMENTATION_FAILURE` before any `DATE` value read on any mismatch;
- use exactly the two preregistered column ordinals (section 3), the
  frozen reopen call (section 4), and the frozen `cell_type`-first/
  `cell_value`-only-for-`XL_CELL_DATE`/`xldate_as_tuple`-year-only
  extraction procedure (section 5), including the per-column DATE-count
  cross-validation (section 5.1);
- build both histograms exactly per section 6, compare them exactly per
  section 7 (no heuristic resolution), and derive `covered_years`/
  `covered_required_years`/`missing_required_years`/
  `all_required_years_covered` exactly per section 8, only when the two
  histograms are identical;
- implement a new closed-schema, fail-closed, non-crashing (arbitrary/
  unhashable input safe, matching the existing `_is_allowed_enum_str`
  pattern) safe-evidence validator enforcing exactly the schema in section
  10, including the `structural_profile_sha256`/`structural_profile_hash_
  verified` three-state rule (section 2.1), the `date_year_value_read`/
  `coverage_evaluated`/`coverage_result_accepted` provenance semantics
  (sections 5, 5.1, 7), and the histogram cardinality/ordering/
  element-identity cross-validation;
- preserve every existing Phase A/B boundary and the existing
  `raw_bytes_read_for_integrity`/`child_content_inspected` phase-provenance
  contract unchanged;
- enforce section 7.1's post-`AMBIGUOUS` stopping rule operationally: the
  implementation and any tooling around it must not expose a mode that
  reruns this stage with a different resolution rule, a different column
  selection, or a refetch, after an `AMBIGUOUS` result within this study;
  and
- be independently GPT exact-SHA reviewed before any real execution
  against the production CHILD, exactly like every prior stage.

This design authorizes none of that implementation work itself.

## Bounded design-closure sweep

```text
CLOSURE_SWEEP_EXTRA_FIXES=Named the exact xlrd date-conversion function (xlrd.xldate_as_tuple(serial, book.datemode), taking only the returned tuple's index-0 year) since the task fixed "convert that serial using xlrd's date conversion with book.datemode" without naming the specific function; xldate_as_tuple is xlrd's long-stable, publicly documented top-level date-conversion entry point (paired with the already-frozen Book.datemode attribute), and no other date-conversion behavior was introduced. Classified an unclassified xldate_as_tuple exception (for a cell already typed XL_CELL_DATE) as IMPLEMENTATION_FAILURE by direct, mechanical application of the already-frozen "any other exception is IMPLEMENTATION_FAILURE" rule from V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md section 2.1, rather than leaving it unclassified. Stated the section 5.1 per-column DATE-count cross-validation generically against the hash-verified structural evidence's own recorded count, rather than hardcoding the literal 19 inside the frozen procedure, since the task's own rule 2 already requires re-deriving and hash-verifying that evidence on every run -- the literal 19 is recorded only as a currently-known fact in section 1, not as a magic number in the extraction rule. Clarified that CHATGPT_DECISION_REQUIRED is not a new outcome invented by this design but remains reachable only through the already-implemented, unmodified Phase A binding checks that precede this stage. No coverage/date/year methodology, source/CHILD identity, evaluation/sample/threshold rule, dependency/version, Phase A/B boundary, network/refetch/gate policy, or required-years value was changed or newly decided by this sweep.
```

No other remaining mechanical ambiguity requiring a methodological choice
was found in this sweep; nothing else triggered `CHATGPT_DECISION_REQUIRED`.

## 15. Non-effects and preserved state

```text
GLOBAL_CHILD_FETCH_AUTHORIZED=false
GLOBAL_CHILD_FETCHED=true
GLOBAL_CHILD_CONTENT_INSPECTED=true
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN=PASS
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPLEMENTATION=PASS
V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
```

This design changes no F6 coverage methodology, required-years set,
threshold, source, retry policy, authority, structural-parser design, or
GLOBAL fanout rule, and creates no additional CHILD read authority, no
coverage-parser implementation authority, no network authority, and no
human-gate consumption. It is not self-called `PASS`. The next action is
GPT exact-SHA independent methodology review of this design.

## GPT design review (BLOCK, MEDIUM_1 + MEDIUM_2 findings, remediation)

```text
REVIEWED_SHA=1d4be751802d62dc53f039cbf730c06390e9d4be
PARENT_SHA=b2fcb56c0e5ace654b638664786229761dc14df8
CRITICAL=0
HIGH=0
MEDIUM=2
LOW=0
RESULT=BLOCK
MEDIUM_1=V9_006_F6_DATE_YEAR_DESIGN_AMBIGUOUS_POST_EXPOSURE_GOVERNANCE_AND_COVERAGE_PROVENANCE_INCORRECT
MEDIUM_2=V9_006_F6_DATE_YEAR_DESIGN_SAFE_OUTPUT_NOT_PHASE_TOTAL_BEFORE_STRUCTURAL_HASH_EXISTS
```

Finding `MEDIUM_1`: the design's original section 7 said `F6_YEAR_
COVERAGE_AMBIGUOUS` leaves "coverage remains unevaluated/false" and
"returns to GPT for a methodology decision," even though both real `DATE`
column year histograms would already have been read, constructed, and
compared by the time that status is reached -- an incorrect governance and
coverage-provenance characterization that also implied a live methodology
choice remained open after seeing the actual histograms.

Finding `MEDIUM_2`: the design required `structural_profile_sha256` to be
a 64-lowercase-hex string at every status, but `IMPLEMENTATION_FAILURE`
can occur before any hash has actually been computed (for example, a
crash while recomputing or canonicalizing the structural evidence), which
would force a future implementation to fabricate a hash value or leave the
safe schema not phase-total.

### Remediation (this revision)

**MEDIUM_1**: added `date_year_value_read`, `coverage_evaluated`, and
`coverage_result_accepted` as always-present provenance booleans (sections
5, 5.1, 7, 10.1); rewrote section 7 to state `coverage_evaluated=true` for
both `CAPTURED` and `AMBIGUOUS` (the mechanical comparison happened in
both cases) while `coverage_result_accepted` is `true` only for `CAPTURED`
(no covered-year set is ever accepted for `AMBIGUOUS`); added new section
7.1 freezing `F6_YEAR_COVERAGE_AMBIGUOUS` as **terminal** for this
preregistered rule in the current V9 study identity -- no rule change, no
column redraw/reselection, no rerun-for-a-favorable-result, no refetch;
GPT may adjudicate the already-preregistered result, implementation
faults, or governance classification, but may not choose a new coverage
methodology for the same exposed study after seeing the histograms; any
materially different rule requires a separately identified successor study
or new preregistration. Section 12 cross-references this terminal rule.

**MEDIUM_2**: added `structural_profile_hash_verified` as an always-present
boolean alongside `structural_profile_sha256`, and froze the exact three
reachable states for that pair in new section 2.1: (a) before a hash is
successfully computed, `sha256=null`/`hash_verified=false`; (b) a computed
hash that mismatches the frozen expected value, `sha256=<actual hash>`/
`hash_verified=false`/`status=IMPLEMENTATION_FAILURE`/`date_year_value_
read=false`; (c) the computed hash matches, `hash_verified=true`, and only
then may the `DATE`-value stage become reachable. No `DATE` value may ever
be read while `hash_verified=false`. A failure before hash computation
never fabricates the expected hash or a placeholder.

Section 10 (safe evidence schema) is rewritten in full (10.1 always-present
keys, 10.2 `year_histograms` conditional presence including the edge case
where both histograms were already built before a later failure, 10.3
status-specific keys) so every key's presence/value rule is mechanically
unambiguous at every reachable status, per both findings together.

### Bounded closure sweep (this remediation)

```text
CLOSURE_SWEEP_EXTRA_FIXES=Clarified in section 5.1 the specific edge case where date_year_value_read remains accurately false at an IMPLEMENTATION_FAILURE reached via the per-column DATE-count cross-validation: only when the failing column is the first one processed and its own encountered DATE count is zero, since in every other reachable case at least one cell_value call has already occurred by the time that check runs. Clarified in section 10.2 that year_histograms may exceptionally be present alongside IMPLEMENTATION_FAILURE only in the case where both complete histograms and the comparison had already finished before a separately-occurring later failure (defense-in-depth around section 8's derivation step), rather than leaving that combination unaddressed. Cross-referenced the new section 7.1 stopping rule from section 12 and added a corresponding operational bullet to section 14's implementation-review scope, so the prohibition on a rerun/reselect/refetch mode after AMBIGUOUS is stated as a concrete implementation requirement, not only prose in section 7. No coverage/date/year methodology, source/CHILD identity, structural profile expected hash, date columns [4,6], xlrd==2.0.2/xldate_as_tuple conversion, required years 2017-2025, the two-histogram exact-equality rule, coverage derivation, Phase A/B rules, network/refetch/gate rules, source/provider, or any threshold/study-sample rule was changed.
```

No other remaining mechanical ambiguity requiring a methodological choice
was found in this sweep; nothing else triggered `CHATGPT_DECISION_REQUIRED`.

```text
V9_006_F6_DATE_YEAR_DESIGN_AMBIGUOUS_POST_EXPOSURE_GOVERNANCE_AND_COVERAGE_PROVENANCE_INCORRECT=REMEDIATED_AWAITING_GPT_REVIEW
V9_006_F6_DATE_YEAR_DESIGN_SAFE_OUTPUT_NOT_PHASE_TOTAL_BEFORE_STRUCTURAL_HASH_EXISTS=REMEDIATED_AWAITING_GPT_REVIEW
V9_006_STAGE_A_F6_DATE_YEAR_COVERAGE_PARSER_DESIGN=BLOCK
```

No production CHILD/path/raw state was accessed, no Python was executed,
no DATE/year value was read, no coverage was executed or evaluated, no
human gate was consumed, and no network request was made beyond `git
fetch`/`push`. The prior OLE/BIFF structural-parser implementation remains
`PASS`; the prior structural execution remains `COMPLETE`/
`STRUCTURAL_FORMAT_CAPTURED`, unaffected by this remediation. This
remediation is not self-called `PASS`, and neither `MEDIUM_1` nor
`MEDIUM_2` is self-called `RESOLVED`. GPT-5.6 Sol remains the final
methodology/review authority.
