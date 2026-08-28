# V9_006 Stage-A F6 deterministic OLE/BIFF structural-parser design

```text
task=V9_006_F6_DETERMINISTIC_OLE_BIFF_STRUCTURAL_PARSER_DESIGN
status=REMEDIATED_AWAITING_GPT_REVIEW
medium_1=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_1_SAFE_PROFILE_TOPOLOGY_UNDERSPECIFIED
medium_1_status=RESOLVED
medium_2=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_2_XLRD_PARSE_AND_USED_RANGE_SEMANTICS_UNDERSPECIFIED
medium_2_status=REMEDIATED_AWAITING_GPT_REVIEW
scope=STRUCTURAL_FORMAT_PARSER_DESIGN_ONLY_NO_COVERAGE_NO_IMPLEMENTATION
network_authorized_by_this_task=false
network_executed_by_this_task=false
production_child_read_by_this_task=false
child_content_inspected_by_this_task=false
coverage_evaluated_by_this_task=false
human_authorization_consumed_by_this_task=false
parser_implementation_authorized_by_this_task=false
```

This is a docs-only design for a future, separately reviewed and
separately implemented deterministic OLE/BIFF structural parser for the
`STRUCTURAL_FORMAT_CAPTURED` outcome of
`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_DESIGN.md`'s Phase C
(structural inspection). It does not implement, execute, or authorize
execution of anything; it defines the exact deterministic structural
profile a future, independently reviewed implementation must produce, and
the exact prohibitions that keep that profile a structural artifact rather
than a coverage result.

## 1. Binding and authority boundary

This design inherits, unmodified, the exact CHILD identity/integrity/root
binding already frozen in
`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_DESIGN.md` section 1 and
already exercised by the completed structural-probe execution record in
`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_IMPLEMENTATION_REVIEW.md`:

```text
source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
applicable_period=TOPIX_GLOBAL_2017_2025
expected_child_sha256=060d74a7f5a3b413d351de05ed07f412d093a3ebf41f6ea3d4e0de3f313b4b0c
expected_child_byte_length=36352
expected_output_root_id_sha256=5705fa3dae30c17a57208a1a03edbb5f4fac8a0986603ba39d21229262abbeee
gate_consumed=true
authorization_reusable=false
second_execution_allowed=false
```

Already-established structural evidence this design binds to and does not
re-derive or contradict:

```text
container_format=OLE_COMPOUND_FILE
open_parse_status=PARSER_NOT_IMPLEMENTED
```

This design grants no refetch, no URL/provider/output-root substitution, no
relock, no new human authorization, no new network access, and no coverage
determination. The existing F6 production raw-acquisition gate remains
consumed and non-reusable; this design consumes nothing.

## 2. Frozen parser decision: `xlrd==2.0.2`, no pandas, no network

The future parser opens the already-integrity-verified CHILD bytes
directly from memory with the canonical protected environment's reviewed
`xlrd==2.0.2` (`REAL_EXECUTION_PYTHON_ENVIRONMENT.md` sections 4-5 and 7;
the same pin already in the frozen seven-package environment lock). It
does **not** use `pandas.read_excel` or any other pandas entry point for
this structural stage, and it performs no network access. The exact open
call, with every keyword argument frozen, is:

```python
import xlrd
book = xlrd.open_workbook(
    file_contents=verified_child_bytes,
    formatting_info=True,
    on_demand=False,
    ragged_rows=False,
)
```

`file_contents=` opens directly from the in-memory bytes already proven, in
Phase B of the structural-probe design, to match the exact expected SHA-256
and byte length -- never a fresh path/file read, never a re-fetch, and
never bytes obtained any other way. No path or filename input is ever
passed to `xlrd`. `formatting_info=True` is required for the cell-type
model this design relies on; `on_demand=False` loads every sheet eagerly
(no lazy/partial sheet loading that could make enumeration order- or
timing-dependent); `ragged_rows=False` is `xlrd`'s default and is stated
explicitly here so it is frozen, not merely assumed. No retry and no
recovery attempt wraps this call: it either returns a `Book` or raises,
handled exactly per section 2.1. Bypassing pandas at this stage is
deliberate: `xlrd` exposes the legacy BIFF sheet/cell/type model directly,
without pandas' DataFrame-shaped abstraction, which is what per-cell
structural typing (section 5 below) requires. This does not change
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md`'s direct-dependency specification
(`pandas` + `xlrd==2.0.2`) or its reviewed environment lock; it only
specifies which of those two already-reviewed packages this particular
structural stage uses.

### 2.1 Open handling and failure classification

```text
xlrd.open_workbook(...) returns a Book                                          => open_parse_status = OPEN_PARSE_OK
xlrd raises its documented format/open-rejection exception before a Book exists => status = STRUCTURAL_FORMAT_UNSUPPORTED, open_parse_status = OPEN_PARSE_UNSUPPORTED
any other exception, anywhere in the open call or the extraction pipeline
that follows it, not explicitly classified above                                => status = IMPLEMENTATION_FAILURE
```

`xlrd`'s own documented exception for a file it recognizes as not a valid,
supported, or readable BIFF/OLE2 workbook is `xlrd.biffh.XLRDError`; that
exception, raised by `xlrd.open_workbook` itself before any `Book` object
exists, is exactly the "format/open rejection" case above. Every other
exception -- from the open call, from sheet/row/column access, or from
per-cell type inspection (section 5.10) -- that is not itself one of this
design's other explicitly classified fail-closed conditions (an unknown
cell-type code, a topology-invariant violation, etc., each already mapped
to `IMPLEMENTATION_FAILURE` where they are defined) is likewise
`IMPLEMENTATION_FAILURE`. This makes exception classification total across
the entire parser: every reachable failure has exactly one classification,
and none falls through unclassified.

The future implementation must never emit exception text, traceback
content, or any other free-form error detail in its safe result -- only the
closed `status`/`open_parse_status` enum values above, per section 5.8's
existing "what must never be emitted" list.

## 3. Structural-only scope: explicit non-derivation list

This stage is structural only. The future implementation MUST NOT derive,
compute, infer, or emit any of the following, under any circumstance,
regardless of what the bytes happen to contain:

- dates or years;
- a covered-year set;
- a minimum or maximum date;
- continuity or completeness of any date/year series;
- TOPIX/index numerical values;
- a coverage PASS/FAIL verdict of any kind; or
- monthly F6 fanout (the existing GLOBAL-slot-to-108-cell fanout rule in
  `V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY.md` is untouched and is
  not triggered by this design).

`V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY.md`'s existing rule stands
unchanged: only a later, separately defined and separately reviewed
deterministic date/year coverage parser may derive an exact covered-year
set, and it must reject malformed/ambiguous date/year structure rather than
infer from row position, count, neighboring years, first/last dates,
continuity, interpolation, or index values. This structural-parser design
is a prerequisite input to that future work, never a substitute for it.

## 4. Sheet enumeration: structure only, never identity or content

The future parser may enumerate every workbook sheet, in workbook order,
but MUST NOT emit sheet names, header strings, cell text, or cell values in
any form -- not as plaintext, not as a hash, not as a truncated or encoded
fragment. Sheets are identified only by their 1-based ordinal position.

## 5. Safe structural evidence

Safe structural evidence extends
`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_IMPLEMENTATION_REVIEW.md`'s
already-reviewed and already-`RESOLVED` closed schema
(`container_format`, `open_parse_status`, `sheet_table_count`,
`structural_dimensions` with `ordinal`/`row_count`/`column_count`/
`visibility`/`object_type`) with one new nested category, frozen under the
new top-level safe-evidence key `cell_type_profiles`: a per-column
cell-storage/type profile. No existing top-level key, enum member, or
schema constraint from that prior remediation chain (MEDIUM-1 through
MEDIUM-4 and HIGH-1) is loosened, removed, or bypassed by this design.

**MEDIUM-1 remediation note**
(`V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_1_SAFE_PROFILE_
TOPOLOGY_UNDERSPECIFIED`): the original version of this section named the
new nested category but did not freeze the exact top-level key, nor enough
cardinality/topology invariants for a future validator to mechanically
prove complete, duplicate-free workbook coverage. Sections 5.1-5.7 below
are the exact, closed contract that remediates this; they supersede the
prior draft's looser wording, not the prior remediation chain's own
already-`RESOLVED` schema, which they only extend.

### 5.1 Frozen new top-level key

```text
cell_type_profiles
```

This is the exact, frozen name of the new top-level safe-evidence key. No
other name, alias, or synonym may be used.

### 5.2 Per-sheet fields (unchanged from the reviewed schema)

```text
status                          in {STRUCTURAL_FORMAT_CAPTURED, STRUCTURAL_FORMAT_UNSUPPORTED, STRUCTURAL_FORMAT_AMBIGUOUS, CHATGPT_DECISION_REQUIRED, IMPLEMENTATION_FAILURE}
container_format                in {OLE_COMPOUND_FILE, ZIP_CONTAINER, UNKNOWN_CONTAINER}   (already-reviewed enum; section 5.3 below further fixes its value for STRUCTURAL_FORMAT_CAPTURED specifically)
open_parse_status                in {PARSER_NOT_IMPLEMENTED, OPEN_PARSE_OK, OPEN_PARSE_UNSUPPORTED, OPEN_PARSE_AMBIGUOUS}
sheet_table_count                non-bool int >= 0

structural_dimensions[i]:
  ordinal                        non-bool int >= 1, unique, 1-based, workbook order
  row_count                      non-bool int >= 0   (sheet's structural used-range row count)
  column_count                   non-bool int >= 0   (sheet's structural used-range column count)
  visibility                     in {VISIBLE, HIDDEN, VERY_HIDDEN, UNKNOWN}   (exact mapping/fail-closed rule below)
  object_type                    = "WORKSHEET"
```

`visibility` is fixed to the exact three-way mapping from `xlrd`'s own
per-sheet BIFF visibility value: `0 => VISIBLE`, `1 => HIDDEN`,
`2 => VERY_HIDDEN`. Any other value `xlrd` returns for a sheet's
visibility, or any failure to access that value for a given sheet, is
**not** recorded as `UNKNOWN` -- it stops the whole run as
`IMPLEMENTATION_FAILURE` (per section 2.1's exception-classification
rule), never a guess, an omitted field, or a silently substituted value.
`UNKNOWN` remains part of the closed `visibility` enum inherited unchanged
from the prior, already-`RESOLVED` schema (it is not removed), but this
specific `xlrd`-based extraction procedure has no path that legitimately
produces it.

### 5.3 Exact required payload for `STRUCTURAL_FORMAT_CAPTURED`

When, and only when, `status == STRUCTURAL_FORMAT_CAPTURED`, the safe
structural-evidence payload must contain **exactly** these six top-level
keys -- no fewer, no more, and none of the prior schema's optional
`candidate_header_column_count`/`candidate_date_column_count`/
`candidate_value_column_count` keys, which are not part of this stage's
`CAPTURED` payload:

```text
status                = "STRUCTURAL_FORMAT_CAPTURED"
container_format      = "OLE_COMPOUND_FILE"     (fixed value, not merely enum membership -- already established at section 1)
open_parse_status     = "OPEN_PARSE_OK"          (fixed value, not merely enum membership -- CAPTURED implies a successful open/parse)
sheet_table_count      non-bool int >= 0
structural_dimensions  a list (see 5.4)
cell_type_profiles      a list (see 5.5)
```

### 5.4 `structural_dimensions` cardinality/topology invariants (for `CAPTURED`)

```text
type(structural_dimensions)                  = list
len(structural_dimensions)                   = sheet_table_count
{item.ordinal for item in structural_dimensions} = {1, 2, ..., sheet_table_count}   (exactly once each)
order                                         = ordinal ascending == workbook order
```

Every ordinal `1..sheet_table_count` appears in `structural_dimensions`
exactly once; there is no gap, no duplicate, and no ordinal outside that
exact range.

### 5.5 `cell_type_profiles`: per-item schema and per-sheet cardinality

Each `cell_type_profiles` item must have **exactly** these three keys, no
other:

```text
sheet_ordinal      non-bool int >= 1
column_ordinal      non-bool int >= 1
cell_type_counts     an object (see below)
```

`cell_type_counts` must have **exactly** these seven keys, no other, each a
non-bool integer `>= 0`:

```text
EMPTY
BLANK
TEXT
NUMBER
DATE
BOOLEAN
ERROR
```

These map one-to-one onto `xlrd`'s own closed cell-type constants
(`XL_CELL_EMPTY`, `XL_CELL_BLANK`, `XL_CELL_TEXT`, `XL_CELL_NUMBER`,
`XL_CELL_DATE`, `XL_CELL_BOOLEAN`, `XL_CELL_ERROR`) -- this design reuses
`xlrd`'s own type taxonomy rather than inventing a divergent one.

For every sheet ordinal `s` present in `structural_dimensions` (with that
sheet's own `column_count` and `row_count`):

```text
count(profiles with sheet_ordinal == s)            = structural_dimensions[s].column_count
{profile.column_ordinal for profile with sheet_ordinal == s} = {1, 2, ..., column_count}   (exactly once each)
if column_count == 0                                : zero profiles reference sheet s
for every such profile                              : sum(its seven cell_type_counts values) == structural_dimensions[s].row_count
```

Every column ordinal `1..column_count` for sheet `s` appears exactly once
among the profiles referencing sheet `s`; if that sheet's `column_count` is
`0`, no profile may reference it at all. Every profile's seven counts sum
exactly to its sheet's `row_count`, since every cell in the structural used
range has exactly one BIFF cell type -- this is a hard requirement, not
merely a should: a future implementation must treat any violation as
`IMPLEMENTATION_FAILURE`, since it would prove the profile does not
actually describe the claimed used range.

Therefore, across the whole payload:

```text
len(cell_type_profiles) = sum(structural_dimensions[s].column_count for every sheet s)
```

No `cell_type_profiles` entry may reference a `sheet_ordinal` absent from
`structural_dimensions`, or a `column_ordinal` outside `1..column_count`
for its sheet. No two entries may share the same `(sheet_ordinal,
column_ordinal)` pair. A future implementation must reject -- fail closed,
never silently drop or de-duplicate -- any payload violating any invariant
in this section or section 5.4.

### 5.6 Non-`CAPTURED` statuses: existing contract preserved, unbroadened

For every `status` value other than `STRUCTURAL_FORMAT_CAPTURED`
(`STRUCTURAL_FORMAT_UNSUPPORTED`, `STRUCTURAL_FORMAT_AMBIGUOUS`,
`CHATGPT_DECISION_REQUIRED`, `IMPLEMENTATION_FAILURE`), the `cell_type_
profiles` key **must be absent** -- never present as an empty list, never
`null`, never any other placeholder. The prior, already-`RESOLVED`
safe-evidence contract for those outcomes (the optional `container_format`/
`open_parse_status`/`sheet_table_count`/`structural_dimensions`/
`candidate_*_column_count` fields and their existing per-field validation)
is otherwise unchanged and unbroadened by this design; section 5.3's exact
six-key `CAPTURED`-only payload does not apply to them.

### 5.7 Canonical ordering (part of deterministic evidence identity)

```text
structural_dimensions   ordered by sheet ordinal ascending
cell_type_profiles       ordered by sheet ordinal ascending, then column ordinal ascending (within each sheet)
```

This ordering is part of the deterministic evidence identity required by
section 7 below -- the same exact CHILD bytes and reviewed environment must
produce evidence in this exact canonical order, not merely the same
multiset of entries.

### 5.8 What must never be emitted

Regardless of nesting depth or field name, the future implementation must
never emit: raw bytes; a raw URL; a machine-local path; a sheet or table
name or any other cell text; a header string or its hash; a cell value in
any form; a format string or its hash; a date or year; a row-level value;
or a coverage result. Counts and dimensions (sections 5.2-5.5) are the only
permitted payload-derived signal, and only in the closed, bounded,
enumerated forms specified above.

### 5.9 Frozen sheet/row/column extraction mechanics

```python
sheet_table_count = book.nsheets          # exact; no filter, no exclusion
for i in range(book.nsheets):
    sheet = book.sheet_by_index(i)
    ordinal = i + 1                        # emitted, 1-based
    row_count = sheet.nrows                # exact; no trim/recompute/normalization/inference
    column_count = sheet.ncols             # exact; no trim/recompute/normalization/inference
```

`sheet_table_count` is exactly `book.nsheets` -- every sheet `xlrd` reports,
with no filtering by visibility, name, content, or any other criterion.
Enumeration is `i = 0 .. book.nsheets - 1` in that exact order, via
`book.sheet_by_index(i)`; the emitted `ordinal` is `i + 1`. `row_count` and
`column_count` are exactly `sheet.nrows`/`sheet.ncols` as `xlrd` reports
them for that sheet -- never trimmed, recomputed from cell content,
normalized, or otherwise inferred. A `STRUCTURAL_FORMAT_CAPTURED`
`structural_dimensions` item has exactly the five keys already frozen in
section 5.2 (`ordinal`, `row_count`, `column_count`, `visibility`,
`object_type`); `object_type` is fixed to `"WORKSHEET"` for every item, as
already stated.

### 5.10 Frozen cell-profiling extraction mechanics

```python
for rowx in range(sheet.nrows):
    for colx in range(sheet.ncols):
        cell_type = sheet.cell_type(rowx, colx)   # type only -- never cell_value/row_values/col_values
        # increment exactly one of the seven frozen cell_type_counts buckets
```

For every sheet, for every `(rowx, colx)` in `range(sheet.nrows) x
range(sheet.ncols)` (0-based, matching `xlrd`'s own indexing), the future
implementation inspects only `sheet.cell_type(rowx, colx)` and increments
exactly one of the seven frozen `cell_type_counts` buckets (section 5.5).
It must never call `sheet.cell_value`, `sheet.row_values`, `sheet.col_values`,
or any other value-returning `xlrd` accessor, for any cell, at any point --
type inspection and value inspection are different API surfaces, and only
the former is ever invoked. No cell may be skipped, sampled, or trimmed
based on its value or its type; every cell in the sheet's full
`nrows x ncols` grid is inspected and counted exactly once. A cell-type
code that is not one of `xlrd`'s seven documented constants (section 5.5)
stops the run as `IMPLEMENTATION_FAILURE`, per section 2.1 and section 7 --
never silently mapped to an existing category, never dropped.

## 6. No coverage inference from structural evidence

Cell-type counts, row counts, column counts, and sheet dimensions are
structural evidence only. The future implementation, and any consumer of
its output, is explicitly prohibited from using these counts, row
positions, neighboring observations, continuity, or density -- directly or
indirectly -- to infer, approximate, or bound coverage, a covered-year set,
a minimum/maximum date, or any data-quality/completeness conclusion. A
`DATE`-type cell-type count is a structural fact about storage type only;
it is not, and must never be treated as, evidence of which dates, or how
many distinct years, are present.

## 7. Determinism

```text
same_exact_child_bytes_plus_same_reviewed_xlrd_environment => identical_structural_evidence
sheet_and_column_ordinals                                   = 1-based
enumeration_order                                            = workbook order
canonical_output_ordering                                    = section 5.7 (structural_dimensions and cell_type_profiles ordering)
heuristic_sheet_selection                                    = prohibited
favorable_subset_selection                                   = prohibited
sheet_and_column_coverage                                     = ALL sheets, ALL columns in each sheet's structural used range
cardinality_and_topology_invariants                           = sections 5.3-5.5 (exact payload keys, exact counts, no unknown/duplicate references)
count_type                                                    = exact nonnegative integers only (non-bool)
unknown_or_new_xlrd_cell_type                                 = IMPLEMENTATION_FAILURE (never silent mapping to an existing category)
```

No sheet or column may be skipped, sampled, truncated, or selected by any
criterion other than "every sheet in the workbook, every column in that
sheet's structural used range." A cell-type code `xlrd` returns that is not
one of its seven documented constants above must stop the run as
`IMPLEMENTATION_FAILURE` rather than be silently folded into an existing
category or dropped -- this design would rather fail closed than let a
future `xlrd` behavior change quietly corrupt the structural profile.
Deterministic evidence identity requires the exact same evidence values
(sections 5.1-5.6), computed via the exact same frozen extraction mechanics
(sections 2, 2.1, 5.9, 5.10), and the exact same canonical ordering
(section 5.7); two runs that agree on content but disagree on order are not
identical evidence.

## 8. Safe outcomes (unchanged)

The future parser may emit only one safe outcome, reusing exactly the
already-reviewed enum -- no outcome is added, removed, or renamed by this
design:

```text
STRUCTURAL_FORMAT_CAPTURED
STRUCTURAL_FORMAT_UNSUPPORTED
STRUCTURAL_FORMAT_AMBIGUOUS
CHATGPT_DECISION_REQUIRED
IMPLEMENTATION_FAILURE
```

## 9. `STRUCTURAL_FORMAT_CAPTURED` grants no coverage-parser authority

`STRUCTURAL_FORMAT_CAPTURED` means only that a safe structural profile
(sections 5.1-5.7, produced via the frozen extraction mechanics of
sections 2, 2.1, 5.9, and 5.10) was captured under this design's
constraints. It is not
a parser PASS for any coverage purpose, not a covered-year result, not an
F6 availability result, and not authorization to inspect further or to
begin coverage-parser implementation. After a future real execution of the
implementation this design authorizes designing, GPT must independently
review the exact safe profile produced before any deterministic date/year
coverage parser may be defined. This design creates no such authority
itself.

## 10. Required future implementation-review scope

A future implementation of this design must, at minimum:

- extend `src/v9_006_f6_offline_child_structural_probe.py`'s existing
  `_safe_structural_evidence` closed-schema validator (already independently
  reviewed and `RESOLVED` through MEDIUM-3/MEDIUM-3A) to mechanically
  enforce, fail-closed and without crashing on arbitrary malformed or
  unhashable input, **every** condition in sections 5.1-5.7: the frozen
  `cell_type_profiles` key name; the exact six-key `CAPTURED`-only payload
  and its fixed `container_format`/`open_parse_status` values;
  `structural_dimensions`'s exact length/ordinal-set/ordering invariants;
  each `cell_type_profiles` item's exact three-key shape and each
  `cell_type_counts`'s exact seven-key, nonnegative-integer-only shape; the
  per-sheet profile-count/column-ordinal-set/zero-column invariants; the
  per-profile sum-equals-row-count invariant; the cross-payload
  `len(cell_type_profiles) == sum(column_count)` invariant; the
  no-unknown-sheet/no-out-of-range-column/no-duplicate-pair invariants; the
  mandatory absence of `cell_type_profiles` for every non-`CAPTURED`
  status; and the canonical ordering of both `structural_dimensions` and
  `cell_type_profiles`. Any single violation of any of these must reject
  the payload the same way the existing validator already fails closed for
  its current schema -- never accept a partially-invalid payload, never
  silently repair, reorder, deduplicate, or truncate it;
- reuse `xlrd==2.0.2` from the canonical protected environment
  (`.venv-real-execution`) exclusively -- no pandas, no network, no new
  dependency;
- open the CHILD bytes with the exact frozen `xlrd.open_workbook(...)`
  call and keyword arguments (section 2), classify open/extraction
  exceptions exactly per section 2.1, extract sheet/row/column structure
  exactly per section 5.9, and profile cell types exactly per section
  5.10 -- never a path/filename input, never `cell_value`/`row_values`/
  `col_values`, never a retry or recovery attempt around the open call;
- preserve every existing Phase A/B/C boundary, the existing
  `raw_bytes_read_for_integrity`/`child_content_inspected` phase-provenance
  contract, and every existing safe-outcome/enum value unchanged; and
- be independently GPT exact-SHA reviewed before any real execution against
  the production CHILD, exactly like every prior stage of this probe.

This design authorizes none of that implementation work itself.

## 11. Non-effects and preserved state

```text
GLOBAL_CHILD_FETCH_AUTHORIZED=false
GLOBAL_CHILD_FETCHED=true
GLOBAL_CHILD_CONTENT_INSPECTED=true
V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_EXECUTION=COMPLETE
V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
```

This design changes no F6 coverage methodology, threshold, source, retry
policy, authority, or GLOBAL fanout rule, and creates no additional CHILD
read authority, no parser-implementation authority, no network authority,
and no human-gate consumption. It is not self-called `PASS`. The next
action is GPT exact-SHA independent methodology review of this design.

## GPT design review

```text
REVIEWED_SHA=bcebc76cd975c559c56feabb55dd9eb90b5199b8
PARENT_SHA=27ef2cf31d7cf2acbf9d0cdd3fa43aa889d91862
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=0
RESULT=BLOCK
MEDIUM_1=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_1_SAFE_PROFILE_TOPOLOGY_UNDERSPECIFIED
```

Finding: the design correctly specified all-sheet/all-column enumeration
and the closed per-column cell-type profile object, but did not freeze the
exact top-level JSON location/shape of the profile collection, nor enough
cardinality/topology invariants for a future validator to mechanically
prove complete, duplicate-free workbook coverage.

Remediation (this revision): freezes the new top-level key exactly as
`cell_type_profiles` (5.1); fixes the exact required six-key payload shape
for `STRUCTURAL_FORMAT_CAPTURED` with `container_format=OLE_COMPOUND_FILE`
and `open_parse_status=OPEN_PARSE_OK` fixed values (5.3); freezes
`structural_dimensions`'s exact length/ordinal-set/ordering invariants for
`CAPTURED` (5.4); freezes each `cell_type_profiles` item's exact three-key
shape and each `cell_type_counts`'s exact seven-key shape, the per-sheet
profile-count/column-ordinal-set/zero-column invariants, the per-profile
sum-equals-row-count invariant (now a hard requirement, not merely
"should"), the cross-payload total-count invariant, and the
no-unknown/no-out-of-range/no-duplicate-reference invariants (5.5);
freezes that `cell_type_profiles` must be absent -- never present, never
empty, never null -- for every non-`CAPTURED` status, while explicitly
preserving and not broadening the existing prior safe-evidence contract for
those outcomes (5.6); freezes canonical output ordering for both
`structural_dimensions` and `cell_type_profiles` as part of deterministic
evidence identity (5.7); and strengthens the required future
implementation-review scope (section 10) to require the extended validator
mechanically enforce every one of these conditions fail-closed and
non-crashing for arbitrary malformed input, never partially accept, repair,
reorder, deduplicate, or truncate a violating payload.

Unchanged by this remediation: the `xlrd==2.0.2` decision (section 2); the
CHILD identity/root/integrity binding (section 1); the Phase A/B/C
boundaries (inherited from
`V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_DESIGN.md`, untouched);
`V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY.md`'s coverage methodology
(section 3, section 6); the allowed safe-outcome enum (section 8); every
network/refetch rule (section 1); every human-gate rule (section 1); and
every date/year/value/coverage prohibition (sections 3, 5.8, 6). No parser
was implemented in this remediation; no production CHILD/path/raw state
was accessed; no network request was made beyond `git fetch`/`push`; no
human gate was consumed; no coverage was evaluated.

```text
V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_1_SAFE_PROFILE_TOPOLOGY_UNDERSPECIFIED=REMEDIATED_AWAITING_GPT_REVIEW
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN=BLOCK
```

This design is not self-called `PASS`, and `MEDIUM_1` is not self-called
`RESOLVED`. GPT-5.6 Sol remains the final methodology/review authority.

## GPT design review (MEDIUM-1 closure, MEDIUM-2 finding)

```text
REVIEWED_SHA=8ee844d8b94964ea4d1b2a3f7909fd4d224035ae
PARENT_SHA=bcebc76cd975c559c56feabb55dd9eb90b5199b8
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=0
RESULT=BLOCK
MEDIUM_1=RESOLVED
MEDIUM_2=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_2_XLRD_PARSE_AND_USED_RANGE_SEMANTICS_UNDERSPECIFIED
```

`MEDIUM_1` (safe-profile topology) is `RESOLVED`. Finding `MEDIUM_2`: the
design froze the closed evidence schema and its topology but not the exact
`xlrd` open call/parameters, the exact sheet/row/column/cell extraction API
calls, the exact visibility-value mapping's failure behavior, or the exact
open/extraction exception classification -- leaving room for two compliant
implementations to disagree on used-range semantics or failure
classification from the same exact bytes and environment.

### MEDIUM-2 remediation

- Section 2 now freezes the exact `xlrd.open_workbook(file_contents=...,
  formatting_info=True, on_demand=False, ragged_rows=False)` call, with no
  path/filename input, no pandas, no retry, and no recovery attempt around
  it.
- New section 2.1 freezes open/extraction exception classification: a
  successful `Book` => `OPEN_PARSE_OK`; `xlrd.biffh.XLRDError` raised
  before a `Book` exists => `STRUCTURAL_FORMAT_UNSUPPORTED` +
  `OPEN_PARSE_UNSUPPORTED`; every other exception anywhere in the open
  call or the extraction pipeline that follows, not otherwise classified,
  => `IMPLEMENTATION_FAILURE`. No exception text or free-form detail is
  ever emitted.
- Section 5.2's `visibility` fallback is corrected: `0/1/2` map exactly to
  `VISIBLE`/`HIDDEN`/`VERY_HIDDEN`; any other value or inaccessibility is
  no longer recorded as `UNKNOWN` -- it now stops the run as
  `IMPLEMENTATION_FAILURE`. `UNKNOWN` remains part of the inherited closed
  enum (not removed); this extraction procedure simply has no path that
  legitimately produces it.
- New section 5.9 freezes exact sheet/row/column extraction:
  `sheet_table_count = book.nsheets`; enumeration `i = 0..nsheets-1` via
  `book.sheet_by_index(i)`, emitted `ordinal = i + 1`; `row_count =
  sheet.nrows` and `column_count = sheet.ncols` exactly, with no trim,
  recompute, normalization, or inference.
- New section 5.10 freezes exact cell profiling: for every `(rowx, colx)`
  in the sheet's full `nrows x ncols` grid, inspect only
  `sheet.cell_type(rowx, colx)` and increment exactly one of the seven
  frozen `cell_type_counts` buckets; `cell_value`/`row_values`/
  `col_values` are never called; no cell is skipped or trimmed by value or
  type; an unrecognized cell-type code is `IMPLEMENTATION_FAILURE`.
- Section 10's implementation-review scope gained one bullet requiring the
  implementation to follow sections 2/2.1/5.9/5.10's exact extraction
  procedure, not merely the output schema.
- Cross-references to "sections 5.1-5.6/5.7" in sections 7 and 9 were
  extended to note they are produced via the frozen extraction mechanics
  of sections 2, 2.1, 5.9, and 5.10.

The complete MEDIUM-1 topology remediation (sections 5.1, 5.3-5.8) is
preserved unchanged by this revision.

### Bounded design-closure sweep

```text
CLOSURE_SWEEP_EXTRA_FIXES=Pinned the exact exception type (xlrd.biffh.XLRDError) that triggers the "format/open rejection" branch of section 2.1's open-handling classification, and generalized that classification into a total rule covering every other exception anywhere in the open call or post-open extraction pipeline (section 2.1). This is a direct, non-methodological consequence of the already-frozen three-way open-handling classification (OPEN_PARSE_OK / STRUCTURAL_FORMAT_UNSUPPORTED+OPEN_PARSE_UNSUPPORTED / IMPLEMENTATION_FAILURE) supplied for this remediation -- it only pins which exact exception class realizes the middle branch and closes the classification so no exception anywhere in the reachable extraction pipeline is left unclassified. No coverage/date/year methodology, source/CHILD identity, evaluation/sample/threshold rule, dependency/version, outcome enum, Phase A/B/C boundary, network/refetch/gate policy, or allowed-evidence-category choice was made or changed.
```

No other remaining mechanical ambiguity requiring a methodological choice
was found in this sweep; nothing else triggered `CHATGPT_DECISION_REQUIRED`.

```text
V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_1_SAFE_PROFILE_TOPOLOGY_UNDERSPECIFIED=RESOLVED
V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_MEDIUM_2_XLRD_PARSE_AND_USED_RANGE_SEMANTICS_UNDERSPECIFIED=REMEDIATED_AWAITING_GPT_REVIEW
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN=BLOCK
```

No parser was implemented, no Python was run, no production CHILD/path/raw
state was accessed, no coverage was evaluated, no human gate was consumed,
and no network request was made beyond `git fetch`/`push`. This design is
not self-called `PASS`, and `MEDIUM_2` is not self-called `RESOLVED`.
GPT-5.6 Sol remains the final methodology/review authority.
