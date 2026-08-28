# V9_006 Stage-A F6 deterministic OLE/BIFF structural-parser design

```text
task=V9_006_F6_DETERMINISTIC_OLE_BIFF_STRUCTURAL_PARSER_DESIGN
status=CANDIDATE_AWAITING_GPT_REVIEW
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
this structural stage, and it performs no network access:

```python
import xlrd
book = xlrd.open_workbook(file_contents=verified_child_bytes)
```

`file_contents=` opens directly from the in-memory bytes already proven, in
Phase B of the structural-probe design, to match the exact expected SHA-256
and byte length -- never a fresh path/file read, never a re-fetch, and
never bytes obtained any other way. Bypassing pandas at this stage is
deliberate: `xlrd` exposes the legacy BIFF sheet/cell/type model directly,
without pandas' DataFrame-shaped abstraction, which is what per-cell
structural typing (section 5 below) requires. This does not change
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md`'s direct-dependency specification
(`pandas` + `xlrd==2.0.2`) or its reviewed environment lock; it only
specifies which of those two already-reviewed packages this particular
structural stage uses.

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
`visibility`/`object_type`) with one new nested category: a per-column
cell-storage/type profile. No existing top-level key, enum member, or
schema constraint from that prior remediation chain (MEDIUM-1 through
MEDIUM-4 and HIGH-1) is loosened, removed, or bypassed by this design.

### 5.1 Top-level and per-sheet fields (unchanged from the reviewed schema)

```text
status                          in {STRUCTURAL_FORMAT_CAPTURED, STRUCTURAL_FORMAT_UNSUPPORTED, STRUCTURAL_FORMAT_AMBIGUOUS, CHATGPT_DECISION_REQUIRED, IMPLEMENTATION_FAILURE}
container_format                = "OLE_COMPOUND_FILE"   (already established; unchanged by this stage)
open_parse_status                in {PARSER_NOT_IMPLEMENTED, OPEN_PARSE_OK, OPEN_PARSE_UNSUPPORTED, OPEN_PARSE_AMBIGUOUS}
sheet_table_count                non-bool int >= 0

structural_dimensions[i]:
  ordinal                        non-bool int >= 1, unique, 1-based, workbook order
  row_count                      non-bool int >= 0   (sheet's structural used-range row count)
  column_count                   non-bool int >= 0   (sheet's structural used-range column count)
  visibility                     in {VISIBLE, HIDDEN, VERY_HIDDEN, UNKNOWN}   ("mechanically available" below)
  object_type                    = "WORKSHEET"
```

`visibility` uses the closed enum if the environment's reviewed `xlrd`
mechanically exposes per-sheet visibility (BIFF visibility state: visible /
hidden / very hidden); if that mechanism is unavailable or raises for a
given sheet, the future implementation records `UNKNOWN` for that sheet's
`visibility` rather than guessing, omitting the field, or emitting anything
else.

### 5.2 New: per-column cell-storage/type profile (nested closed schema)

For every sheet, for every column ordinal in that sheet's structural used
range (1-based, left to right), the future implementation records exactly
one profile object using only these closed BIFF/`xlrd` cell-type
categories:

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

```text
cell_type_profile:
  sheet_ordinal                  matches a structural_dimensions[i].ordinal
  column_ordinal                 non-bool int >= 1, 1-based
  cell_type_counts:
    EMPTY                        non-bool int >= 0
    BLANK                        non-bool int >= 0
    TEXT                         non-bool int >= 0
    NUMBER                       non-bool int >= 0
    DATE                         non-bool int >= 0
    BOOLEAN                      non-bool int >= 0
    ERROR                        non-bool int >= 0
```

Only `sheet_ordinal`, `column_ordinal`, and the seven closed-category
nonnegative counts may appear in a `cell_type_profile` entry -- no other
key, and no free-form nested value of any kind. Within one sheet, the sum
of every column's seven counts equals that sheet's `row_count` (every cell
in the structural used range has exactly one BIFF cell type); a future
implementation should treat a violation of that invariant as
`IMPLEMENTATION_FAILURE`, since it would indicate the profile does not
actually describe the claimed used range.

### 5.3 What must never be emitted

Regardless of nesting depth or field name, the future implementation must
never emit: raw bytes; a raw URL; a machine-local path; a sheet or table
name or any other cell text; a header string or its hash; a cell value in
any form; a format string or its hash; a date or year; a row-level value;
or a coverage result. Counts and dimensions (sections 5.1-5.2) are the only
permitted payload-derived signal, and only in the closed, bounded,
enumerated forms specified above.

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
heuristic_sheet_selection                                    = prohibited
favorable_subset_selection                                   = prohibited
sheet_and_column_coverage                                     = ALL sheets, ALL columns in each sheet's structural used range
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
(sections 5.1-5.2) was captured under this design's constraints. It is not
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
  reviewed and `RESOLVED` through MEDIUM-3/MEDIUM-3A) with an exact nested
  closed schema for the per-column `cell_type_profile` entries defined in
  section 5.2 -- closed key set, closed nonnegative-integer-only value
  types, no arbitrary or free-form nested values, and total (non-crashing)
  validation for arbitrary malformed/unhashable input, exactly as the
  existing enum/dimension validators already are;
- reuse `xlrd==2.0.2` from the canonical protected environment
  (`.venv-real-execution`) exclusively -- no pandas, no network, no new
  dependency;
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
