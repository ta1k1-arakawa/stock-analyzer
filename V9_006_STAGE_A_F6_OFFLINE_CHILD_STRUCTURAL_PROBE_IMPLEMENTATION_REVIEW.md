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
