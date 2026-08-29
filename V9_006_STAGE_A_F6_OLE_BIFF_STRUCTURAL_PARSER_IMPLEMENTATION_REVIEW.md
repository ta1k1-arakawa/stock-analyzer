# V9_006 Stage-A F6 OLE/BIFF structural-parser implementation review

```text
task=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPLEMENTATION_CHECKPOINT
status=CANDIDATE_AWAITING_GPT_REVIEW
design_reviewed_sha=cc9efde8fa1531eae2f7544e7326d767cd5a4562
design_status=PASS
medium_1=RESOLVED
medium_2=RESOLVED
scope=IMPLEMENTATION_OF_FROZEN_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_ONLY
network_authorized_by_this_task=false
network_executed_by_this_task=false
production_child_read_by_this_task=false
child_content_inspected_by_this_task=false
coverage_evaluated_by_this_task=false
human_authorization_consumed_by_this_task=false
dependencies_changed=false
```

This implements, in
`src/v9_006_f6_offline_child_structural_probe.py`, the reviewed
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md` (design `PASS` at
`cc9efde8fa1531eae2f7544e7326d767cd5a4562`, `MEDIUM_1`/`MEDIUM_2` both
`RESOLVED`) exactly as frozen: the `xlrd==2.0.2` open call and its
parameters, open/extraction exception classification, the `cell_type_
profiles` safe-evidence key and its full cardinality/topology/ordering
contract, and the frozen sheet/row/column/cell extraction mechanics. No
production CHILD/path/raw state was read, no network request beyond `git
fetch`/`push` was made, no coverage was evaluated or derived, no human gate
was consumed, and no dependency was added or changed (`xlrd==2.0.2` was
already a committed dependency in `requirement.txt`,
`requirements-real-execution.txt`, and
`requirements-real-execution.lock.txt`; only a transient in-session
`pip3 install` of the already-committed pin, plus `xlwt==1.3.0` as a
generator-only dev/test fixture dependency, per the existing repository
convention already established by
`scripts/generate_synthetic_jpx_xls_fixture.py`).

## Implementation

`src/v9_006_f6_offline_child_structural_probe.py`:

- `import xlrd` added.
- `_SAFE_EVIDENCE_ALLOWED_KEYS` extended with the frozen new top-level key
  `cell_type_profiles` (design section 5.1). New closed-set constants added:
  `_CELL_TYPE_COUNT_KEYS` (the seven frozen bucket names, design section
  5.5), `_CELL_TYPE_PROFILE_ALLOWED_KEYS` (the exact three per-item keys),
  `_CAPTURED_REQUIRED_KEYS` (the exact six top-level keys required for
  `STRUCTURAL_FORMAT_CAPTURED`, design section 5.3),
  `_XLRD_CELL_TYPE_CATEGORY_BY_CODE` and `_XLRD_SHEET_VISIBILITY_BY_CODE`
  (verbatim reuse of `xlrd`'s own closed cell-type and visibility constants,
  design sections 5.2/5.10 -- no divergent taxonomy invented).
- `_default_structural_inspector` replaced with the frozen real parser: the
  exact `xlrd.open_workbook(file_contents=raw, formatting_info=True,
  on_demand=False, ragged_rows=False)` call (design section 2); a
  `try`/`except xlrd.XLRDError` classifying a pre-`Book` format rejection to
  `STRUCTURAL_FORMAT_UNSUPPORTED`/`OPEN_PARSE_UNSUPPORTED` (design section
  2.1) -- every other exception, from the open call or from extraction, is
  deliberately left uncaught here and propagates to the caller's existing
  Phase-C exception boundary, which converts it to `IMPLEMENTATION_FAILURE`
  with accurate `true`/`true` phase provenance, matching design section
  2.1's "any other exception => `IMPLEMENTATION_FAILURE`" rule without
  duplicating that classification inside the inspector itself; on a
  successful open, exact `book.nsheets`/`book.sheet_by_index(i)`/`ordinal =
  i + 1`/`sheet.nrows`/`sheet.ncols` extraction (design section 5.9), exact
  `sheet.visibility` 0/1/2 -> `VISIBLE`/`HIDDEN`/`VERY_HIDDEN` mapping with
  any other value raising (never a silent `UNKNOWN`, design section 5.2),
  and exact `sheet.cell_type(rowx, colx)`-only per-cell profiling into the
  seven frozen buckets, with any unrecognized code raising (design section
  5.10) -- `cell_value`/`row_values`/`col_values` are never called. The
  magic-byte container sniff already present before this task is preserved
  and used only for the `UNSUPPORTED` branch's `container_format`, since a
  successful `CAPTURED` open always fixes `container_format` to
  `OLE_COMPOUND_FILE` per design section 5.3.
- New closed-set validators added ahead of `_safe_structural_evidence`:
  `_is_positive_int`, `_is_valid_cell_type_counts`,
  `_is_valid_cell_type_profile_item`,
  `_is_valid_captured_structural_dimensions` (proves exact length + exact
  ordinal-set + ascending order in one `ordinals == list(range(1, N+1))`
  comparison), and `_is_valid_captured_cell_type_profiles` (proves
  no-duplicate-reference + canonical order in one strictly-increasing
  `(sheet_ordinal, column_ordinal)` key check, plus per-sheet
  cardinality/column-range/sum-equals-row-count cross-validation against
  `structural_dimensions`).
- `_safe_structural_evidence` extended with a `STRUCTURAL_FORMAT_CAPTURED`-
  specific block enforcing the exact six required keys, the fixed
  `container_format`/`open_parse_status` values, and the full
  `structural_dimensions`/`cell_type_profiles` topology cross-validation
  (design sections 5.3-5.5); and an `elif "cell_type_profiles" in value`
  branch rejecting its presence on any non-`CAPTURED` status (design
  section 5.6). Every new check reuses the existing `_is_allowed_enum_str`/
  `_is_nonneg_int` total-for-arbitrary-input pattern already established by
  the MEDIUM-3/MEDIUM-3A remediation chain, so no new unhashable-input
  crash surface was introduced.

No Phase A/B boundary, `ProbeBlocked` phase-provenance contract, CLI
contract, or existing pre-`CAPTURED` safe-evidence field was changed.
`scripts/run_v9_006_f6_offline_child_structural_probe.py` was not modified
-- outside this task's allowed-files scope, and no CLI-level change was
mechanically required.

## Bounded implementation closure sweep

```text
CLOSURE_SWEEP_EXTRA_FIXES=Two pre-existing tests in tests/test_v9_006_f6_offline_child_structural_probe.py assumed the pre-implementation default inspector's behavior and needed updating to the real xlrd-based parser's actual, empirically-verified behavior: (1) test_metadata_locator_and_default_unsupported_are_safe used the shared fixture default payload b"PK\x03\x04synthetic", which the real xlrd 2.0.2 parser routes through zipfile.BadZipFile (not xlrd.XLRDError) per its internal .xlsx/zip-format probing, so the test's own payload was changed to genuine non-Excel garbage bytes that cleanly raise xlrd.XLRDError, preserving the test's intent (proving the real UNSUPPORTED path is safe) rather than its incidental byte value; a new dedicated test (test_default_inspector_other_open_exception_is_implementation_failure) was added to explicitly cover the b"PK\x03\x04synthetic" zipfile.BadZipFile case as the required "other open/extraction exception => IMPLEMENTATION_FAILURE" coverage item, since design section 2.1 classifies it there, not as UNSUPPORTED. (2) test_synthetic_structural_outcomes_after_integrity's parametrized STRUCTURAL_FORMAT_CAPTURED case used container_format=ZIP_CONTAINER plus optional candidate_*_column_count keys not part of the now-frozen six-key CAPTURED contract, so it was split into test_synthetic_ambiguous_outcome_after_integrity (unchanged AMBIGUOUS shape, since AMBIGUOUS is not subject to the six-key contract) and test_exact_six_key_captured_payload_is_accepted (a genuinely valid six-key CAPTURED payload). Both were direct, mechanical consequences of implementing the already-reviewed design against the existing test suite, not methodology choices -- no CHATGPT_DECISION_REQUIRED condition was triggered. No further mechanical defect was found in this sweep.
```

## Test coverage

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. python3 -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=110
TESTS_PASSED=110
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
GIT_DIFF_CHECK=PASS
```

New targeted tests (77 prior + 33 new = 110) cover, at minimum:

- the exact `xlrd.open_workbook` call arguments, captured via monkeypatch
  (`file_contents`/`formatting_info`/`on_demand`/`ragged_rows`, no other
  keyword);
- deterministic, byte-identical repeated-call output against real
  `xlwt`-generated bytes;
- all seven `xlrd` cell-type buckets (`TEXT`/`NUMBER`/`DATE`/`BOOLEAN`/
  `ERROR`/`BLANK`/`EMPTY`), via genuine `xlwt.Row.set_cell_*` writes,
  cross-validated by the real `_safe_structural_evidence` validator;
- exact `nrows`/`ncols` including an unwritten-cell-implied `EMPTY` and an
  explicit `BLANK`;
- sheet visibility `0`/`1`/`2` round-tripping to `VISIBLE`/`HIDDEN`/
  `VERY_HIDDEN`, via direct `worksheet.visibility = N` assignment on the
  `xlwt` fixture (not `xlwt`'s unrelated active-tab "sheet_visible" state);
- an unrecognized `xlrd` sheet-visibility code failing closed to
  `IMPLEMENTATION_FAILURE` (fake `Book`/`Sheet`, since no valid BIFF file
  can carry this);
- a genuine `xlrd.XLRDError` before any `Book` exists mapping to
  `STRUCTURAL_FORMAT_UNSUPPORTED`/`OPEN_PARSE_UNSUPPORTED`;
- the empirically-verified `zipfile.BadZipFile` case mapping to
  `IMPLEMENTATION_FAILURE` as "other open/extraction exception";
- an unrecognized `xlrd` cell-type code failing closed to
  `IMPLEMENTATION_FAILURE` (fake `Book`/`Sheet`);
- the exact six-key `CAPTURED` payload being accepted end-to-end;
- seventeen distinct `CAPTURED`-topology-contract violations rejected
  (missing/extra required key, wrong fixed `container_format`/
  `open_parse_status` value, `sheet_table_count`/`structural_dimensions`
  cardinality mismatch, wrong per-sheet column cardinality, cell-type-count
  sum mismatch, duplicate `(sheet_ordinal, column_ordinal)` pair, wrong
  canonical order, out-of-range `column_ordinal`, unknown `sheet_ordinal`
  reference, missing/extra nested `cell_type_counts` key, missing/extra
  nested profile-item key, and `cell_type_profiles` present on a
  non-`CAPTURED` status for both `UNSUPPORTED` and `AMBIGUOUS`);
- eight new unhashable/malformed-nested-input cases specifically for
  `cell_type_profiles` (non-list, dict-instead-of-list, non-dict item,
  unhashable `sheet_ordinal`/`column_ordinal`, non-dict/unhashable-valued
  `cell_type_counts`) proving `ProbeBlocked` is raised, never `TypeError`,
  extending the existing MEDIUM-3A total-for-arbitrary-input regression
  suite.

All 77 pre-existing Phase A/B/C, MEDIUM-1/2/3/3A/4, and CLI-forwarding
tests remain passing unchanged (two were updated to match the real parser's
now-implemented behavior, per the closure sweep above, without weakening
their original intent). Test execution used the Claude Code Cloud Linux
environment's `python3` with `pytest`/`xlrd==2.0.2`/`xlwt==1.3.0` installed
in-session; no repository dependency file was changed.

## Non-effects and preserved state

```text
GLOBAL_CHILD_FETCH_AUTHORIZED=false
GLOBAL_CHILD_FETCHED=true
GLOBAL_CHILD_CONTENT_INSPECTED=true
V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
```

No production CHILD/path/raw state access, network request beyond `git
fetch`/`push`, human-gate consumption/reuse, coverage evaluation, or F6
coverage-methodology/design/source-identity change occurred in this
implementation task. This implementation is not self-called `PASS`. The
next action is GPT exact-SHA independent implementation review.

## GPT-5.6 Sol exact-SHA review: MEDIUM-1 finding (undeclared xlwt test dependency)

```text
REVIEWED_SHA=0143ea9b6bcf401dfd470f62dec4096b33051ca7
PARENT_SHA=cc9efde8fa1531eae2f7544e7326d767cd5a4562
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=0
RESULT=BLOCK
MEDIUM_1=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPL_MEDIUM_1_UNDECLARED_XLWT_TEST_DEPENDENCY_BREAKS_REPRODUCIBILITY
```

Finding: the implementation logic itself is design-conformant, but
`tests/test_v9_006_f6_offline_child_structural_probe.py` imported `xlwt` at
module import time to build synthetic legacy `.xls` bytes for real-xlrd
integration coverage. `xlwt` is not a declared normal/test dependency in
`requirement.txt`; repository governance already treats `xlwt==1.3.0` as a
manually installed fixture-generator-only tool (see
`scripts/generate_synthetic_jpx_xls_fixture.py`). The previously reported
110/110 targeted-test run therefore depended on a transient in-session
`pip install` and was not reproducible from the repository-declared test
environment.

### MEDIUM-1 remediation

`tests/test_v9_006_f6_offline_child_structural_probe.py` no longer imports
`xlwt`, `io`, or `datetime` (all were only needed to build `xlwt`
workbooks in memory). Real end-to-end OLE/BIFF `xlrd` integration coverage
now uses the already-committed synthetic fixture
`tests/fixtures/synthetic_jpx_source_snapshot.xls` instead of a
freshly-built-in-test workbook:
`test_default_inspector_genuine_ole_biff_fixture_is_captured_and_deterministic`
reads the committed bytes, verifies their identity against
`scripts.generate_synthetic_jpx_xls_fixture.EXPECTED_FIXTURE_SHA256` (that
module's own `import xlwt` is local to its `build_workbook_bytes()`
function body, not module-level, so importing the module for this constant
does not itself require `xlwt`), then proves
`_default_structural_inspector` reports `STRUCTURAL_FORMAT_CAPTURED` /
`OLE_COMPOUND_FILE` / `OPEN_PARSE_OK` and produces byte-identical safe
evidence across two calls on the same bytes -- cross-validated by the real
`_safe_structural_evidence` validator, not merely shaped like a valid
payload.

Every other real-`xlrd`-API test that previously built an `xlwt` workbook
now uses the existing `_FakeSheet`/`_FakeBook` classes instead (extended to
accept an optional full `cell_type_grid` matrix, in addition to the single
`cell_type_code` already used by the two fail-closed-only tests), with
`xlrd.open_workbook` monkeypatched to return the fake `Book`:
`test_default_inspector_covers_all_seven_cell_type_buckets` supplies all
seven of `xlrd`'s own documented type-code constants across one row;
`test_default_inspector_nrows_ncols_exact_with_blanks` supplies a
`row x column` matrix distinguishing `EMPTY`/`BLANK`/`NUMBER`/`TEXT` purely
by `cell_type()` code, proving `row_count`/`column_count` equal the fake
sheet's own `nrows`/`ncols` exactly; and
`test_default_inspector_visibility_zero_one_two_round_trip` supplies three
fake sheets with `visibility=0/1/2`, proving the exact
`VISIBLE`/`HIDDEN`/`VERY_HIDDEN` mapping. Every one of these fakes exposes
only `nrows`/`ncols`/`visibility`/`cell_type(row, col)` -- the exact `xlrd`
`Sheet` surface the reviewed parser touches -- never `cell_value`,
matching design section 5.10 unchanged. The existing
`test_default_inspector_calls_xlrd_open_workbook_with_frozen_arguments`,
`test_default_inspector_invalid_visibility_is_implementation_failure`, and
`test_default_inspector_unknown_cell_type_is_implementation_failure` tests
were already fake/monkeypatch-based and needed no change beyond the
`_FakeSheet` constructor extension (fully backward compatible with their
existing `cell_type_code=` usage).

No change was made to `src/v9_006_f6_offline_child_structural_probe.py`,
the frozen parser design/methodology, coverage/date/year logic, any
dependency file, production CHILD/path/raw state, or network/gate/authority
rules. `xlwt` was not added to `requirement.txt`,
`requirements-real-execution.txt`, or
`requirements-real-execution.lock.txt` -- it remains, as before, a manually
installed fixture-generator-only tool, now used by no committed test.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. python3 -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=110
TESTS_PASSED=110
TESTS_FAILED=0
XLWT_GREP_MATCHES=0
SOURCE_CODE_CHANGED=false
XLRD_REAL_FIXTURE_TEST=true
XLWT_TEST_DEPENDENCY_REMOVED=true
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
COVERAGE_EVALUATED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
GIT_DIFF_CHECK=PASS
```

All 110 targeted tests were run and passed twice: once with `xlwt==1.3.0`
still installed from the prior task's transient in-session `pip install`
(unchanged, since nothing in the remediated suite requires or forbids its
mere presence), and a second time after `pip3 uninstall -y xlwt` removed it
entirely from this Claude Code Cloud environment -- proving the remediated
suite genuinely no longer depends on it, not merely that its source no
longer names it. `git grep -n "import xlwt|xlwt." --
tests/test_v9_006_f6_offline_child_structural_probe.py` returns zero
matches (grep exit `1`, the expected success result for this absence
check). `git diff --check` passes.

The prior 110/110 evidence recorded in this document's "Test coverage"
section above was a genuine, valid result for the transient Claude Code
Cloud environment it ran in at the time; it is not retracted or
invalidated as a test outcome, only superseded as reproducibility evidence
by this remediation, since it depended on an undeclared dependency this
remediation now removes. The remediated targeted suite no longer requires
`xlwt` to run, in this or any other environment matching the repository's
declared test dependencies.

```text
V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPL_MEDIUM_1_UNDECLARED_XLWT_TEST_DEPENDENCY_BREAKS_REPRODUCIBILITY=REMEDIATED_AWAITING_GPT_REVIEW
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPLEMENTATION=BLOCK
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN=PASS
```

No production CHILD/path/raw state access, network request beyond `git
fetch`/`push`, human-gate consumption/reuse, coverage evaluation, or
dependency-file change occurred in this remediation task. This remediation
is not self-called `PASS`, and `MEDIUM_1` is not self-called `RESOLVED`.
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN` remains `PASS`,
unaffected by this implementation-level finding. GPT-5.6 Sol remains the
final independent review authority.

## Final GPT-5.6 Sol review: implementation PASS

```text
REVIEWED_SHA=b2fcb56c0e5ace654b638664786229761dc14df8
PARENT_SHA=0143ea9b6bcf401dfd470f62dec4096b33051ca7
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
MEDIUM_1=RESOLVED
V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPLEMENTATION=PASS
LOW_1=V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_HEADER_STATUS_STALE_AFTER_PASS
```

GPT-5.6 Sol's independent exact-SHA review of
`b2fcb56c0e5ace654b638664786229761dc14df8` closes `MEDIUM_1` (undeclared
`xlwt` test dependency) `RESOLVED`, with zero new `CRITICAL`/`HIGH`/
`MEDIUM` findings. `V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_
IMPLEMENTATION` is now `PASS`. One `LOW` finding,
`V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_HEADER_STATUS_STALE_AFTER_
PASS`: the header metadata block at the top of
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md` still read
`status=REMEDIATED_AWAITING_GPT_REVIEW` and
`medium_2_status=REMEDIATED_AWAITING_GPT_REVIEW`, stale since the design's
own GPT design-closure `PASS` (recorded lower in that same document at
`cc9efde8fa1531eae2f7544e7326d767cd5a4562`) already resolved both.

### LOW-1 remediation

`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md`'s header block is
updated mechanically: `status` -> `PASS`, `medium_2_status` -> `RESOLVED`.
`medium_1_status` was already `RESOLVED` and is unchanged. No section body,
frozen methodology, invariant, extraction mechanic, or safe-evidence
schema in that design was touched -- only the two stale top-of-file status
fields.

```text
V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN_HEADER_STATUS_STALE_AFTER_PASS=RESOLVED
```

No production CHILD/path/raw state access, network request beyond `git
fetch`/`push`, human-gate consumption/reuse, coverage evaluation, Python
execution, or methodology change occurred in this remediation. This
remediation is not self-called `PASS`; GPT-5.6 Sol remains the final
independent review authority.

## Real production offline structural probe execution (human-operated)

```text
PRE_EXECUTION_PROVENANCE=PASS
EXACT_HEAD=b2fcb56c0e5ace654b638664786229761dc14df8
PROTECTED_ENVIRONMENT_PREFLIGHT=PASS
REAL_EXECUTION_ENVIRONMENT_FROZEN=true
CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE=YES

PROBE_EXIT_CODE=0
EXECUTION_RESULT=COMPLETE
STATUS=STRUCTURAL_FORMAT_CAPTURED
NETWORK_REQUESTS=0
RAW_BYTES_READ_FOR_INTEGRITY=true
CHILD_CONTENT_INSPECTED=true
COVERAGE_EVALUATED=false

SAFE_STRUCTURAL_SUMMARY:
sheet_table_count=1
sheet_ordinal=1
row_count=86
column_count=10
visibility=VISIBLE
object_type=WORKSHEET
date_bearing_column_ordinals=[4, 6]
date_cell_count_column_4=19
date_cell_count_column_6=19

STRUCTURAL_EVIDENCE_CANONICAL_IDENTITY:
canonicalization=json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
structural_profile_sha256=4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce
```

This is the human operator's run of the already-reviewed, already-`PASS`ed
`_default_structural_inspector` (the frozen `xlrd==2.0.2` OLE/BIFF parser,
`V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md`, design `PASS`,
implementation `PASS` at this exact SHA), reached via
`run_offline_child_structural_probe`'s existing Phase A -> Phase B -> Phase
C boundary, against the exact already-locked production F6 GLOBAL CHILD,
using the protected environment already verified `FROZEN`/ready for this
implementation. **Claude did not execute this probe.**
`STRUCTURAL_FORMAT_CAPTURED` is one of the design's closed safe outcomes;
the safe structural summary above reports only closed-enum/bounded-integer
structural facts already permitted by the reviewed safe-evidence schema
(sheet/column/row counts, visibility, object type, the two date-bearing
column ordinals, and their respective `DATE` cell counts) -- it contains
no raw bytes, raw URL, machine-local path, sheet/table name or text,
header text, exact date, cell value, or coverage verdict.
`RAW_BYTES_READ_FOR_INTEGRITY=true` and `CHILD_CONTENT_INSPECTED=true`
report that the exact locked CHILD bytes passed Phase B integrity
verification before Phase C structural inspection was reached, per this
implementation's inherited phase-provenance contract.

The frozen structural-evidence canonical identity
(`structural_profile_sha256=4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce`,
computed as `json.dumps(evidence, sort_keys=True,
separators=(",", ":")).encode("utf-8")` then SHA-256-hashed) is recorded
here exactly as reported by the human operator's run. **Claude did not
independently recompute or verify this hash** -- doing so would require
executing Python against the exact reviewed structural evidence for the
real production CHILD, which this docs-and-recording task does not do (no
Python, no production CHILD read, per this task's own constraints). This
recorded hash is the exact value
`V9_006_STAGE_A_F6_DATE_YEAR_COVERAGE_PARSER_DESIGN.md` section 2 freezes
as the mandatory pre-date-read identity gate for any future coverage-parser
implementation.

Adjudication:

- the execution contract completed successfully;
- `STRUCTURAL_FORMAT_CAPTURED` is an allowed structural result;
- date-bearing column ordinals `4` and `6` are now established structural
  facts about the exact locked CHILD, feeding
  `V9_006_STAGE_A_F6_DATE_YEAR_COVERAGE_PARSER_DESIGN.md` section 1's
  binding -- not a coverage result themselves;
- this is **not** a coverage evaluation, data-quality result, or
  strategy/profitability result;
- no covered-year set, date value, or index value is derived, inferred, or
  exposed by this recording;
- no refetch, second acquisition, or new human-gate consumption is
  authorized by this result.

No new human authorization gate was consumed by this run; the existing F6
production raw-acquisition gate remains consumed and non-reusable.
`V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED` remains `false`: no
coverage was evaluated, computed, or inferred by this structural execution
or this recording task, and none may be until the separately defined and
independently reviewed `V9_006_STAGE_A_F6_DATE_YEAR_COVERAGE_PARSER_
DESIGN.md` is implemented and independently reviewed against the exact
preserved CHILD bytes. `V9_design_frozen=false` and `future_profitability_
established=false` remain unchanged. This recording task itself performed
no Python execution, no network operation beyond `git fetch`/`push`, no
production CHILD/path/raw state access, and consumed no human gate.

This recording commit is not self-called `PASS`. GPT-5.6 Sol remains the
final independent review authority over this exact-SHA structural-execution
record.
