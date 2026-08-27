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
