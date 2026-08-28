# V9_006 Stage-A F6 offline CHILD structural probe implementation review

```text
task=V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_IMPLEMENTATION
status=AWAITING_GPT_REVIEW
implementation_parent_sha=62b0fef78da75d474633368f23322884f8fa74bc
synthetic_only=true
production_child_reads=0
child_content_inspected=false
coverage_evaluated=false
source_data_network_requests=0
human_authorization_consumed=false
```

The implementation is limited to `src/v9_006_f6_offline_child_structural_
probe.py` and its privacy-safe JSON CLI. It preserves the reviewed phase
boundary: metadata-only root/receipt/meta validation, opaque content-blind
length/SHA-256/raw-meta integrity verification, then structural inspection.
No network function, gate input, refetch, or durable-state mutation path was
introduced.

Without adding dependencies, the default structural handler emits only a safe
unsupported container result; synthetic injected inspectors exercise the
captured and ambiguous safe outcomes. No parser, coverage result, fanout, date,
year, value, raw URL, raw path, payload, sheet/table name, or header text is
emitted by the CLI.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. .venv\Scripts\python.exe -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_PASSED=12
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
SOURCE_DATA_NETWORK_REQUESTS=0
```

The execution agent does not call this implementation PASS. GPT-5.6 Sol
remains the final independent reviewer.

## MEDIUM-1 remediation: failure-phase provenance false negative

```text
finding=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_1_FAILURE_PHASE_PROVENANCE_FALSE_NEGATIVE
reviewed_sha=34ceb0252dfd67f5a5a8e7a304f240ff56d313d5
status=REMEDIATED_AWAITING_GPT_REVIEW
open_not_in_scope=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_2_CANONICAL_RAW_LOCK_PROVENANCE_VALIDATION_INCOMPLETE,V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_3_SAFE_EVIDENCE_SCHEMA_NOT_FAIL_CLOSED
```

Before remediation, both CLI exception handlers in
`scripts/run_v9_006_f6_offline_child_structural_probe.py` hardcoded
`raw_bytes_read_for_integrity=false` and `child_content_inspected=false` for
every failure, regardless of which phase actually failed. A Phase B
integrity mismatch (bytes already read) or a Phase C structural-inspection
failure (bytes read and inspection reached) both falsely reported the
Phase-A "no CHILD byte read at all" state.

Remediation makes `ProbeBlocked` carry the exact phase boundary reached at
raise time (`raw_bytes_read_for_integrity: bool | "unknown"`,
`child_content_inspected: bool`), sets those fields explicitly at each
`_blocked(...)` call site in `src/v9_006_f6_offline_child_structural_probe.py`
per the reviewed three-phase methodology, and has the CLI forward `exc`'s
exact fields instead of hardcoding them. A byte-read attempt that itself
raises reports `raw_bytes_read_for_integrity="unknown"` (never fabricated
`false`), since a failed read does not prove no bytes were exposed. A
genuinely unanticipated, non-`ProbeBlocked` exception at the CLI boundary
also fails closed to `"unknown"`/`false` rather than fabricating `false`/
`false`. No phase boundary, allowed structural evidence, or methodology was
changed.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. .venv\Scripts\python.exe -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=20
TESTS_PASSED=20
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
```

New targeted synthetic tests cover: Phase A CLI failure (false/false), Phase
B SHA-mismatch failure (true/false, module and CLI-forwarding), Phase B
read-exception failure (unknown/false, module and CLI-forwarding), Phase C
inspector-exception failure (true/true, module and CLI-forwarding), Phase C
safe-evidence-validation failure (true/true, module), an unproven-phase
CLI exception fail-closed test (unknown/false), and privacy-safe JSON
non-leakage across all of the above.

The execution agent does not call `MEDIUM_1` `PASS`/`RESOLVED`. GPT-5.6 Sol
remains the final independent reviewer of this exact SHA.

## MEDIUM-2 remediation: canonical raw-lock provenance validation incomplete

```text
finding=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_2_CANONICAL_RAW_LOCK_PROVENANCE_VALIDATION_INCOMPLETE
reviewed_sha=0055bda2433f3063641bb5afb7e852c323e5a2bc
medium_1=RESOLVED
status=REMEDIATED_AWAITING_GPT_REVIEW
open_not_in_scope=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_3_SAFE_EVIDENCE_SCHEMA_NOT_FAIL_CLOSED
```

Before remediation, Phase A's `_metadata_is_schema_valid` only checked field
presence/types (non-empty `requested_url`/`resolved_url` strings, `sha256`
64-hex format, integer `http_status`/`byte_length`) but did not enforce the
canonical V9 raw-lock provenance rules: it never validated that
`requested_url`/`resolved_url` are actual JPX URLs, never bounded
`http_status` to a real HTTP range, never checked `retrieval_timestamp_utc`
against the canonical timestamp format, and never checked that the
metadata's own filename/raw-lock key matches the canonical key mechanically
derived from its `source_family`/`applicable_period`/`requested_url`. A
malformed, off-domain, or filename/content-mismatched candidate could
satisfy the old check and be selected as the unique Phase-A candidate.

Remediation strengthens `_metadata_is_schema_valid` (now also taking the
candidate's `meta_path`) to require, all still metadata-only and before any
`.bin` read: `requested_url` and `resolved_url` pass the existing canonical
`validate_jpx_url` (imported from `src/v9_005_stage_a_jpx_probe.py`, not
reinvented); `http_status` is a non-bool integer in `100..599`;
`retrieval_timestamp_utc` matches the existing canonical
`_is_canonical_raw_lock_timestamp` check (`%Y-%m-%dT%H:%M:%SZ` round-trip);
`sha256` is exactly lowercase 64-hex (unchanged, already correct); and the
metadata filename stem equals the existing canonical
`source_object_slot_id(source_family, applicable_period, requested_url)` raw-
lock key. Actual byte-length/SHA/raw-meta content integrity against the real
`.bin` bytes remains exclusively Phase B's responsibility, unchanged. No new
URL/key semantics were invented; all reused verbatim from the already-
reviewed `src/v9_005_stage_a_jpx_probe.py` module.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. python3 -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=39
TESTS_PASSED=39
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
```

New targeted synthetic tests cover, all asserting false/false Phase-A
provenance and no `.bin` read: off-domain `requested_url` rejected;
off-domain `resolved_url` rejected; five noncanonical-timestamp variants
rejected; five invalid-`http_status` variants (out-of-range, boolean,
string, zero) rejected; a metadata filename/raw-lock key that diverges from
its own otherwise-valid content's canonical key rejected; five malformed-SHA
variants (wrong length, non-hex, uppercase) rejected; and a fully canonical
synthetic JPX candidate (valid domain URLs, canonical timestamp/status/SHA
format, filename equal to its real canonical key) successfully reaching
Phase B. All 20 MEDIUM-1 tests remain passing unchanged. Test execution used
the Claude Code Cloud Linux environment's `python3` with `pytest` installed
in-session; the reviewed Windows `.venv\Scripts\python.exe` command itself
was not executed, and no repository dependency file was changed.

**Separately observed, not remediated by this task (flagging only):**
`ProbeBindings.source_family` (`SOURCE_FAMILY` constant, `"SOURCE_FAMILY_
TOPIX_HISTORICAL_INDEX_VALUE"`) does not textually equal the actual string
value the real production raw-lock metadata's `source_family` field holds.
`run_f6_production_root_global_raw_acquisition_network` in
`src/v9_005_stage_a_jpx_probe.py` locks the real F6 GLOBAL CHILD with
`source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE`, a v9_005 module
constant whose actual string value is `"TOPIX_HISTORICAL_INDEX_VALUE"` (no
`SOURCE_FAMILY_` prefix) -- so the real durable metadata's `source_family`
field is `"TOPIX_HISTORICAL_INDEX_VALUE"`, not the v9_006 module's frozen
`"SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE"`. If unresolved, a future real
run of this offline probe against the actual production output root would
find zero qualifying candidates and fail closed to
`CHATGPT_DECISION_REQUIRED` (a safe, non-leaking direction, never a false
acceptance), rather than reaching Phase B/C at all. This finding is outside
this task's explicit scope (`FINDING_TO_FIX` names only MEDIUM-2; MEDIUM-3
is explicitly `OPEN_NOT_IN_SCOPE`; the allowed-files list does not license
an unscoped binding-value change), and correcting which literal value is
canonically correct is a methodology decision this task does not delegate.
It is recorded here as a fact for GPT-5.6 Sol to triage, not self-remediated.

The execution agent does not call `MEDIUM_2` `PASS`/`RESOLVED`. GPT-5.6 Sol
remains the final independent reviewer of this exact SHA.

## MEDIUM-4 remediation: source_family binding literal mismatch

```text
finding=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_4_SOURCE_FAMILY_BINDING_LITERAL_MISMATCH
reviewed_sha=8bd42c58e211886012acc154815ce4b2ed2cd9bd
medium_1=RESOLVED
medium_2=RESOLVED
status=REMEDIATED_AWAITING_GPT_REVIEW
open_not_in_scope=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_3_SAFE_EVIDENCE_SCHEMA_NOT_FAIL_CLOSED
```

This is the exact issue flagged (but explicitly not self-remediated, as out
of scope) in the MEDIUM-2 section above, now confirmed by GPT-5.6 Sol as its
own finding and authorized for a targeted fix. The v9_006 module locally
redefined a literal, `SOURCE_FAMILY = "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_
VALUE"`, that froze the *identifier name* of the real production constant as
a string, instead of that constant's actual value. The real F6 production
raw acquisition (`run_f6_production_root_global_raw_acquisition_network` in
`src/v9_005_stage_a_jpx_probe.py`) locks both ROOT and CHILD with
`source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE`, a v9_005 module
constant whose actual string value is `"TOPIX_HISTORICAL_INDEX_VALUE"` (no
`SOURCE_FAMILY_` prefix). Unremediated, a future real run of this offline
probe against the actual production output root would find zero qualifying
candidates for the real production metadata and fail closed to
`CHATGPT_DECISION_REQUIRED`, never reaching Phase B/C.

Remediation imports `SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE` directly
from `src/v9_005_stage_a_jpx_probe.py` and sets
`SOURCE_FAMILY = SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE`, reusing the
canonical constant rather than redefining a divergent literal.
`ProbeBindings.source_family` (used by `FROZEN_BINDINGS`) now equals the
real production value through this single reused binding.
`APPLICABLE_PERIOD = "TOPIX_GLOBAL_2017_2025"` is unchanged. No other
constant, phase boundary, structural evidence schema, or MEDIUM-1/MEDIUM-2
behavior was touched; the test fixture already referenced `probe.
SOURCE_FAMILY` symbolically, so it now automatically builds synthetic
candidates against the corrected canonical value with no fixture-construction
change required.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. python3 -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=43
TESTS_PASSED=43
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
```

New targeted synthetic regression tests cover: `SOURCE_FAMILY_TOPIX_
HISTORICAL_INDEX_VALUE == "TOPIX_HISTORICAL_INDEX_VALUE"` directly against
the real v9_005 constant; `FROZEN_BINDINGS.source_family` (and `probe.
SOURCE_FAMILY`) equals that canonical value and no longer equals the old
erroneous identifier-name string; a synthetic candidate built with the real
production constant passes Phase A and reaches Phase B; and a candidate
whose `source_family` field holds the old erroneous identifier-name string
is rejected before any `.bin` read (with preserved false/false Phase-A
provenance). All 20 MEDIUM-1 and all 19 MEDIUM-2 tests remain passing
unchanged (39 prior + 4 new = 43). Test execution used the Claude Code Cloud
Linux environment's `python3` with `pytest` installed in-session; the
reviewed Windows `.venv\Scripts\python.exe` command itself was not executed,
and no repository dependency file or canonical Windows environment was
touched.

No production CHILD/path/raw state access, network request, human-gate
consumption/reuse, structural-evidence-schema change, dependency change, or
F6 coverage-methodology/design/source-identity change occurred.
`GLOBAL_CHILD_FETCH_AUTHORIZED=false`; `GLOBAL_CHILD_FETCHED=true`;
`GLOBAL_CHILD_CONTENT_INSPECTED=false`;
`V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false`;
`V9_006_STAGE_A_NETWORK_AUTHORIZED=false`; `V9_006_STAGE_A_EXECUTED=false`;
`ACQUISITION_IMPLEMENTATION_COMPLETE=false`.

The execution agent does not call `MEDIUM_4` `PASS`/`RESOLVED`. GPT-5.6 Sol
remains the final independent reviewer of this exact SHA.

## MEDIUM-3 remediation: safe evidence schema not fail-closed

```text
finding=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_3_SAFE_EVIDENCE_SCHEMA_NOT_FAIL_CLOSED
reviewed_sha=ea45801a5618c337d1d76193ecc307ed22f2b913
medium_1=RESOLVED
medium_2=RESOLVED
medium_4=RESOLVED
status=REMEDIATED_AWAITING_GPT_REVIEW
```

Before remediation, `_safe_structural_evidence` only checked the top-level
key set against the allowed-key list and that `status` was an `OUTCOMES`
member; every other allowed key (`container_format`, `open_parse_status`,
`sheet_table_count`, `structural_dimensions`, and the three
`candidate_*_column_count` fields) accepted any value of any type with no
further validation. An allowed top-level key was therefore an open channel:
an injected or buggy structural inspector could place an arbitrary string
(payload-derived text, a date, a year, a header/sheet name, a URL, a path)
directly in `container_format`/`open_parse_status`, or nest arbitrary
content inside `structural_dimensions` list items (extra keys, non-dict
items, negative or boolean counts, non-enum `visibility`/`object_type`
strings), and it would pass through to the CLI's privacy-safe JSON
untouched.

Remediation makes every allowed field closed-set/type/range validated, so no
allowed key can carry a free-form string:
- `status` -- unchanged, must be an exact `OUTCOMES` member;
- `container_format`, if present -- must be exactly one of
  `OLE_COMPOUND_FILE` / `ZIP_CONTAINER` / `UNKNOWN_CONTAINER`;
- `open_parse_status`, if present -- must be exactly one of
  `PARSER_NOT_IMPLEMENTED` / `OPEN_PARSE_OK` / `OPEN_PARSE_UNSUPPORTED` /
  `OPEN_PARSE_AMBIGUOUS`;
- `sheet_table_count` and the three `candidate_*_column_count` fields, if
  present -- must each be a non-bool, non-negative `int`;
- `structural_dimensions`, if present -- must be a `list` of `dict` items,
  each restricted to exactly the allowed keys `ordinal`/`row_count`/
  `column_count`/`visibility`/`object_type` (no extra or nested keys);
  `ordinal` must be a non-bool int `>= 1` and unique across the list;
  `row_count`/`column_count` must each be a non-bool, non-negative `int`;
  `visibility`, if present, must be exactly one of `VISIBLE`/`HIDDEN`/
  `VERY_HIDDEN`/`UNKNOWN`; `object_type`, if present, must be exactly one of
  `WORKSHEET`/`TABLE`/`UNKNOWN`.

Because every field now resolves to either a closed enum or a bounded
integer, there is no remaining channel -- at the top level or nested inside
`structural_dimensions` -- for payload-derived text, dates, years, header
or sheet names, URLs, or paths to leak through an allowed key, however
deeply placed. The reviewed evidence schema's allowed top-level key set is
unchanged (no key added or removed), and no coverage methodology, design,
or parsing was added. `_default_structural_inspector`'s output (`OLE_
COMPOUND_FILE`/`ZIP_CONTAINER`/`UNKNOWN_CONTAINER`, `PARSER_NOT_IMPLEMENTED`,
`sheet_table_count=0`, `structural_dimensions=[]`) already satisfies the
strict schema unchanged, so default production behavior is preserved
exactly. Every rejection still raises through the same `_blocked(...,
raw_bytes_read_for_integrity=True, child_content_inspected=True)` path used
before remediation, preserving the reviewed MEDIUM-1 phase-provenance
contract for any Phase-C rejection.

Existing synthetic captured/ambiguous fixtures previously used a test-only
`"SYNTHETIC"` placeholder for `container_format`/`open_parse_status`, which
the new strict validation correctly rejects; they now use only the strict
frozen enums (`ZIP_CONTAINER`/`OPEN_PARSE_OK`), per the task's explicit
instruction not to add a test-only production enum.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. python3 -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=66
TESTS_PASSED=66
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
```

New targeted synthetic tests (all asserting the preserved
`raw_bytes_read_for_integrity=true`/`child_content_inspected=true`
Phase-C-rejection contract) cover: an arbitrary `container_format`;
an arbitrary `open_parse_status`; an arbitrary string, a nested list, and a
nested dict injected into `structural_dimensions` items;
`structural_dimensions` itself not a list; an extra nested dimension key; a
date-like string smuggled under `container_format`; a URL-like string
smuggled under `open_parse_status`; header/path-like strings smuggled under
a dimension's `visibility`/`object_type` fields; boolean values where
`sheet_table_count`, `candidate_header_column_count`, a dimension's
`row_count`, and `ordinal` each expect a plain integer; negative
`sheet_table_count`, `candidate_date_column_count`, and a dimension's
`column_count`; an `ordinal` of `0` (below the required minimum); a
duplicate `ordinal` across two dimension items; a `status` outside
`OUTCOMES` (an unrecognized string and `None`); and an extra top-level key.
Also updated/added: valid strict captured and ambiguous evidence (using
only the frozen enums, all seven optional fields present with a fully
populated dimension item) are accepted; the default `STRUCTURAL_FORMAT_
UNSUPPORTED` production inspector path remains accepted unchanged. All 20
MEDIUM-1, all 19 MEDIUM-2, and all 4 MEDIUM-4 regression tests remain
passing unchanged (43 prior + 23 new = 66). Test execution used the Claude
Code Cloud Linux environment's `python3` with `pytest` installed in-session;
the reviewed Windows `.venv\Scripts\python.exe` command itself was not
executed, and no repository dependency file or canonical Windows
environment was touched.

No production CHILD/path/raw state access, network request, human-gate
consumption/reuse, evidence-schema key change, dependency change, or F6
coverage-methodology/design/source-identity change occurred.
`GLOBAL_CHILD_FETCH_AUTHORIZED=false`; `GLOBAL_CHILD_FETCHED=true`;
`GLOBAL_CHILD_CONTENT_INSPECTED=false`;
`V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false`;
`V9_006_STAGE_A_NETWORK_AUTHORIZED=false`; `V9_006_STAGE_A_EXECUTED=false`;
`ACQUISITION_IMPLEMENTATION_COMPLETE=false`.

The execution agent does not call `MEDIUM_3` `PASS`/`RESOLVED`. GPT-5.6 Sol
remains the final independent reviewer of this exact SHA.

## MEDIUM-3A remediation: unhashable value in safe-evidence enum checks

```text
finding=V9_006_F6_STRUCTURAL_PROBE_IMPL_MEDIUM_3A_UNHASHABLE_SAFE_EVIDENCE_FAIL_CLOSED
reviewed_sha=278290f6df3afee90e5e665587c2a5596c6ec13a
medium_1=RESOLVED
medium_2=RESOLVED
medium_4=RESOLVED
medium_3=OPEN_REMEDIATION_INCOMPLETE (this task's target)
status=REMEDIATED_AWAITING_GPT_REVIEW
```

The prior MEDIUM-3 remediation closed every allowed field to an exact enum
or bounded-integer set, but its five closed-set membership checks (`status`
against `OUTCOMES`; `container_format`; `open_parse_status`; a dimension's
`visibility`; a dimension's `object_type`) each used a bare `value not in
some_frozenset` test. Python's `in` operator on a `frozenset` hashes its
operand first, so a malformed inspector value that is itself unhashable
(a `list` or `dict` -- e.g. `status=[]`, `container_format={}`) raises a raw
`TypeError` from inside the membership test itself, before the intended
fail-closed `ProbeBlocked` path is ever reached. Reproduced directly:
`_safe_structural_evidence({"status": []})` raised `TypeError: unhashable
type: 'list'` rather than `ProbeBlocked`. Because this can happen after the
Phase-C structural-inspection boundary has already been crossed, a
`TypeError` escaping `_safe_structural_evidence` uncaught would propagate
out of `run_offline_child_structural_probe` and -- at the CLI boundary --
be caught only by the CLI's generic, non-`ProbeBlocked` fallback, which
reports `raw_bytes_read_for_integrity="unknown"`/`child_content_inspected=
false`: an inaccurate understatement of a failure that in fact occurred
after the CHILD bytes were read and structural inspection had begun.

Remediation adds `_is_allowed_enum_str(value, allowed) -> bool`
(`isinstance(value, str) and value in allowed`), making every closed-set
membership check total for arbitrary Python objects by guarding the hash-
requiring `in` test with an `isinstance(value, str)` check first. Applied to
all five enum checks: `status`, `container_format`, `open_parse_status`,
`visibility`, `object_type`. No enum member, allowed key, or evidence
category was added, removed, or broadened -- the exact schema from the
prior MEDIUM-3 remediation is preserved unchanged.

As defense in depth, `run_offline_child_structural_probe`'s Phase-C
boundary now also wraps the `_safe_structural_evidence(...)` call inside the
same `try`/`except ProbeBlocked`/`except Exception` block already used for
the `structural_inspector(...)` call (previously it was called *after* that
block, unprotected). Any genuinely unexpected exception raised while
validating safe structural evidence -- not only the specific unhashable-
value case fixed above -- is now translated fail-closed to `outcome=
IMPLEMENTATION_FAILURE`, `raw_bytes_read_for_integrity=true`,
`child_content_inspected=true`, and can never escape as an ordinary
exception to reach the CLI's unproven-phase `unknown`/`false` fallback. An
already-correct `ProbeBlocked` raised from `_safe_structural_evidence`
(e.g. for an ordinary malformed-evidence rejection) is still re-raised via
the same `_blocked(exc.outcome, ...)` call as before -- its outcome is
preserved, and its `raw_bytes_read_for_integrity`/`child_content_inspected`
fields are (re-)forced to `true`/`true`, which they already were.

```text
TARGETED_TEST_COMMAND=PYTHONPATH=. python3 -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
TESTS_RUN=77
TESTS_PASSED=77
TESTS_FAILED=0
PRODUCTION_CHILD_READS=0
CHILD_CONTENT_INSPECTED=false
SOURCE_DATA_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
DEPENDENCIES_CHANGED=false
```

New targeted synthetic tests cover: `status=[]`/`status={}`;
`container_format=[]`/`{}`; `open_parse_status=[]`/`{}`; a dimension's
`visibility=[]`/`{}`; a dimension's `object_type=[]`/`{}` -- each asserted,
via `pytest.raises(probe.ProbeBlocked)`, to raise specifically
`ProbeBlocked` (never `TypeError`, which would fail the assertion with a
different exception type) with the preserved
`raw_bytes_read_for_integrity=true`/`child_content_inspected=true`
Phase-C-rejection contract; and a dedicated regression that monkeypatches
`_safe_structural_evidence` itself to raise a generic, unmodeled
`RuntimeError`, proving `run_offline_child_structural_probe` converts it to
`ProbeBlocked(IMPLEMENTATION_FAILURE, true, true)` rather than letting it
escape -- which is exactly the condition that would otherwise reach the
CLI's `unknown`/`false` fallback. All 66 prior MEDIUM-1/2/3/4 tests remain
passing unchanged (66 prior + 11 new = 77). Test execution used the Claude
Code Cloud Linux environment's `python3` with `pytest` installed in-session;
the reviewed Windows `.venv\Scripts\python.exe` command itself was not
executed, and no repository dependency file or canonical Windows
environment was touched. `scripts/run_v9_006_f6_offline_child_structural_
probe.py` was not modified -- outside this task's allowed-files scope, and
the fix is entirely internal to the module.

No production CHILD/path/raw state access, network request, human-gate
consumption/reuse, evidence-schema key/enum change, dependency change, or
F6 coverage-methodology/design/source-identity change occurred.
`GLOBAL_CHILD_FETCH_AUTHORIZED=false`; `GLOBAL_CHILD_FETCHED=true`;
`GLOBAL_CHILD_CONTENT_INSPECTED=false`;
`V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false`;
`V9_006_STAGE_A_NETWORK_AUTHORIZED=false`; `V9_006_STAGE_A_EXECUTED=false`;
`ACQUISITION_IMPLEMENTATION_COMPLETE=false`.

The execution agent does not call `MEDIUM_3` `PASS`/`RESOLVED`. GPT-5.6 Sol
remains the final independent reviewer of this exact SHA.

## Final GPT-5.6 Sol review: implementation PASS

```text
task=V9_006_F6_STRUCTURAL_PROBE_FINAL_REVIEW_AND_CANONICAL_WINDOWS_TEST_RECORD
reviewed_sha=59dbf0081372c0bb559d03b457e629b9b11db639
parent_sha=278290f6df3afee90e5e665587c2a5596c6ec13a
CRITICAL=0
HIGH=0
MEDIUM=0
RESULT=PASS
MEDIUM_1=RESOLVED
MEDIUM_2=RESOLVED
MEDIUM_3=RESOLVED
MEDIUM_4=RESOLVED
V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_IMPLEMENTATION=PASS
```

GPT-5.6 Sol's independent exact-SHA review of `59dbf0081372c0bb559d03b457e
629b9b11db639` closes every open finding from this implementation's review
chain (MEDIUM_1 failure-phase provenance false negative; MEDIUM_2 canonical
raw-lock provenance validation; MEDIUM_3/MEDIUM_3A safe-evidence schema
fail-closed and its unhashable-value gap; MEDIUM_4 source_family binding
literal mismatch), all `RESOLVED`, with zero new `CRITICAL`/`HIGH`/`MEDIUM`
findings. `V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_IMPLEMENTATION`
is now `PASS`.

## Canonical Windows execution evidence

```text
execution_performed_by=HUMAN_OPERATOR
execution_environment=WINDOWS_POWERSHELL_LOCAL_MACHINE
canonical_python_interpreter=LOCAL_REPOSITORY_.venv_PYTHON
exact_head=59dbf0081372c0bb559d03b457e629b9b11db639
targeted_test_command=PYTHONPATH=. .venv\Scripts\python.exe -m pytest tests/test_v9_006_f6_offline_child_structural_probe.py -q
CANONICAL_WINDOWS_TESTS_RUN=77
CANONICAL_WINDOWS_TESTS_PASSED=77
CANONICAL_WINDOWS_TESTS_FAILED=0
elapsed_seconds=1.32
TARGETED_TEST=PASS
GIT_DIFF_CHECK=PASS
WORKING_TREE_CLEAN=true
PRODUCTION_CHILD_READS=0
COVERAGE_EVALUATED=false
REAL_NETWORK_REQUESTS=0
HUMAN_GATES_CONSUMED=0
```

This canonical Windows-grounded targeted-test rerun was performed by the
human operator on their local machine, directly against the same exact
`tests/test_v9_006_f6_offline_child_structural_probe.py` at head
`59dbf0081372c0bb559d03b457e629b9b11db639`, using the repository's canonical
local `.venv` interpreter. **Claude did not execute this Windows command --
all Windows-grounded execution in this repository's governance is reserved
to the human operator or direct-PowerShell entrypoints, never Claude Code
Cloud.** It supersedes, for readiness purposes, the earlier stale-command-
reporting LOW concern noted against this implementation's synthetic test
provenance: the exact reviewed SHA now has an independently human-executed,
canonical-environment, 77/77-passing confirmation, in addition to (not
replacing) the separate Claude Code Cloud Linux synthetic runs already
recorded above, which remain distinct evidence of the same test suite
passing in the non-canonical development/CI environment.

No production CHILD/path/raw state was read, no network request was made,
no human authorization gate was consumed or reused, and no coverage was
evaluated by either the canonical Windows rerun or this recording task.
`GLOBAL_CHILD_FETCH_AUTHORIZED=false`; `GLOBAL_CHILD_FETCHED=true`;
`GLOBAL_CHILD_CONTENT_INSPECTED=false`;
`V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false`;
`V9_006_STAGE_A_NETWORK_AUTHORIZED=false`; `V9_006_STAGE_A_EXECUTED=false`;
`ACQUISITION_IMPLEMENTATION_COMPLETE=false`; `future_profitability_
established=false` all remain unchanged. This PASS establishes readiness of
the offline structural-probe implementation only; it grants no network,
production-CHILD-content-inspection, coverage-evaluation, human-gate, or
design-freeze authority. GPT-5.6 Sol remains the final independent review
authority for any future change to this implementation.
