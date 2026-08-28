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
