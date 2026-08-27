# V9_006 Stage-A F6 offline CHILD structural probe design

```text
task=V9_006_STAGE_A_F6_OFFLINE_CHILD_STRUCTURAL_PROBE_DESIGN
status=REMEDIATED_AWAITING_GPT_REVIEW
scope=READ_ONLY_OFFLINE_STRUCTURAL_FORMAT_PROBE_ONLY
network_authorized_by_this_task=false
network_executed_by_this_task=false
production_child_read_by_this_task=false
child_content_inspected_by_this_task=false
coverage_evaluated_by_this_task=false
human_authorization_consumed_by_this_task=false
```

## 1. Binding and authority boundary

This is a docs-only design for a future, separately reviewed direct-Windows-
PowerShell operation. It binds only the already acquired production F6 GLOBAL
CHILD evidenced in
`V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_EXECUTION_EVIDENCE.md`:

```text
source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
applicable_period=TOPIX_GLOBAL_2017_2025
expected_child_sha256=060d74a7f5a3b413d351de05ed07f412d093a3ebf41f6ea3d4e0de3f313b4b0c
expected_child_byte_length=36352
gate_consumed=true
authorization_reusable=false
second_execution_allowed=false
network_requests_allowed=0
```

The acquisition gate is already consumed and must never be supplied, checked
as reusable, reset, or reused. This design grants no human authority, network
authority, refetch, URL/provider substitution, raw-lock creation, durable
state repair, parser implementation, coverage determination, or F6 fanout.

## 2. Metadata-only locator phase

The future operation must run as one atomic PowerShell block directly on the
target Windows machine, with `$ErrorActionPreference = "Stop"` inside the
block. It must not be held by Codex or Claude Code. Its implementation and
exact-SHA review are separate prerequisites.

Before any CHILD bytes are read, opened, or hashed, the future runner must
perform this metadata-only, read-only locator phase. It must mechanically
establish one and only one
existing production output-root candidate from durable state, without guessing
a path, searching a wider location, copying a lock, or using a diagnostic
artifact. The candidate must contain exactly one complete raw/meta pair whose
metadata claims the bound source family and applicable period above. This
phase may read durable metadata and verify its schema and identity fields, but
it must not read the selected CHILD `.bin`, open the payload, perform a hash,
or perform structural/content inspection. All path, URL, receipt, and raw
metadata strings remain internal and must never be printed.

The future reviewed implementation must define the exact no-guess,
metadata-only durable-state locator before it can read the CHILD. If a unique
candidate cannot be established from existing durable state, or any pair is
missing, malformed, mismatched, unreadable, duplicate, or otherwise
unverifiable, it must stop before a CHILD read with
`CHATGPT_DECISION_REQUIRED` (for absent/ambiguous location identity) or
`IMPLEMENTATION_FAILURE` (for corrupt/unverifiable bound state). It must not
broaden discovery, infer a location from a path convention, repair state, or
use another object.

The receipt may be read only to verify its exact already-consumed state; it is
never an authorization input. `gate_consumed=true` does not authorize any
network or second acquisition.

## 3. Content-blind integrity read phase

Only after section 2 succeeds may the future probe read the exact selected
CHILD `.bin` as opaque bytes. This phase permits only actual byte-length
calculation, SHA-256 calculation, and exact raw/meta integrity verification.
It must prove the expected byte length and SHA-256 from section 1 before any
structural inspection. A length, SHA-256, or raw/meta integrity mismatch must
STOP as `IMPLEMENTATION_FAILURE`.

This opaque-byte read is not file-format parsing or structural/content
inspection. It must not open a workbook/container, decompress for inspection,
inspect sheets/tables/cells/headers/dates/years/values, emit structural
evidence, evaluate or infer coverage, or emit any payload-derived evidence
other than the bound integrity result. It accurately records that raw bytes
were read for integrity verification, while it does not set
`GLOBAL_CHILD_CONTENT_INSPECTED=true`.

## 4. Structural inspection phase

Only after section 3 proves the exact expected CHILD SHA-256 and byte length
may the future probe open the verified CHILD bytes, read-only, solely to
identify enough format structure for a later GPT-reviewed deterministic parser
design. Only this phase is CHILD structural/content inspection. It may not
modify, delete, reset, replace, copy, relock, or repair the production output
root, receipt, ROOT, or CHILD.

Permitted safe evidence is limited to structural format information:

- container/file-format enum and open/parse-success enum;
- sheet/table count;
- per-sheet safe ordinal, visibility/type, and structural dimensions;
- candidate header/date-column/value-column counts;
- cell storage/type profiles and formatting profiles required for later parser
  design; and
- SHA-256 hashes instead of sheet names, header text, format strings, or other
  text where disclosure could reveal dates, years, or values.

The probe must not emit raw bytes, raw URL, machine-local path, sheet/table
names or text, data-row dates or years, covered-year set, minimum/maximum
date, row-level values, TOPIX/index numerical values, or a coverage verdict.
It must not use row count, row position, neighboring dates, continuity,
first/last observations, or numerical values to infer coverage.

## 5. Safe outcomes and stopping rule

The future probe may emit only one safe outcome:

```text
STRUCTURAL_FORMAT_CAPTURED
STRUCTURAL_FORMAT_UNSUPPORTED
STRUCTURAL_FORMAT_AMBIGUOUS
CHATGPT_DECISION_REQUIRED
IMPLEMENTATION_FAILURE
```

`STRUCTURAL_FORMAT_CAPTURED` establishes only that safe structural evidence
was captured. It is not a parser PASS, covered-year result, F6 availability
result, or authorization to inspect further. If the necessary structural
distinction cannot be exposed without revealing a coverage outcome, the probe
must stop as `CHATGPT_DECISION_REQUIRED`; it must not broaden inspection or
derive an indirect coverage conclusion.

Any later parser methodology or implementation must be separately defined and
independently reviewed against the exact preserved CHILD bytes. The existing
F6 coverage rule remains unchanged: only a later deterministic parser may
derive an exact covered-year set, and no row-count, positional, continuity, or
value-based inference is permitted.

## 6. Medium-1 remediation scope and non-effects

The GPT review of this design at `REVIEWED_SHA=268453fb693cc90f3dc2c380c9873700bed356c6`
identified `V9_006_F6_STRUCTURAL_PROBE_DESIGN_MEDIUM_1_PRECONTENT_INTEGRITY_
READ_CIRCULARITY`. This remediation only separates the metadata-only locator,
content-blind integrity read, and structural inspection phases above. It does
not address the separate
`V9_006_F6_STRUCTURAL_PROBE_DESIGN_MEDIUM_2_PRODUCTION_ROOT_RUNTIME_BINDING_
UNDERSPECIFIED` finding.

```text
GLOBAL_CHILD_FETCH_AUTHORIZED=false
GLOBAL_CHILD_FETCHED=true
GLOBAL_CHILD_CONTENT_INSPECTED=false
V9_006_STAGE_A_F6_PRODUCTION_COVERAGE_EVALUATED=false
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
```

This design changes no methodology, coverage rule, threshold, source, retry
policy, authority, or GLOBAL fanout. The next action is GPT exact-SHA
independent review of this design.
