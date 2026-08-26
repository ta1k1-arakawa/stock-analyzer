# V9_006 Stage-A F6 one-level expanded neighborhood probe offline implementation review

~~~text
task=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
implementation_base_sha=c5b9fd5494042ab1a06d23fadbef4410e2a924a7
offline_only=true
real_raw_lock_execution=false
network_executed=false
global_child_selected=false
global_child_url_bound=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
~~~

This implementation applies the reviewed one-level expanded-neighborhood
design only to an existing F6_ROOT_STRUCTURE_DIAGNOSTIC raw lock. It does
not execute against the real stored raw lock during this task. The
offline seam reads the existing lock through the existing read-only lock
reader, derives the semantic heading H with the already-reviewed
proper-descendant rule, then derives exactly P = H.immediate_parent and
G = P.immediate_parent. It does not expand beyond G.

## Implementation

The implementation is in
src/v9_005_stage_a_jpx_probe.py and reuses the existing reviewed
full-DOM parser, DOM analysis, text normalization, semantic-heading
identification, element identity, DOM-path, and raw-attribute utilities.
No parallel parsing methodology was introduced.

The dedicated artifact constants are:

~~~text
schema_version=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_V1
diagnostic=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE
filename=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT.json
~~~

The serialized top-level key set is exactly:

~~~text
schema_version
diagnostic
requested_url
resolved_url
byte_length
sha256
retrieval_timestamp_utc
status
failure_reason
semantic_heading
parent_container
expanded_container
children
anchors
headings
~~~

H, P, and G identities contain only dom_path, tag, id, and classes.
Every direct ELEMENT child of G is retained in document order and is
classified only as BEFORE_P, P, or AFTER_P. P is required to occur
exactly once as a direct ELEMENT child of G.

Every descendant anchor of G is retained in document order. Its visible
text uses the existing normalization utility, its raw_href is the exact
source spelling or null, and its owner is the immediate ELEMENT child
of G with its BEFORE_P, P, or AFTER_P relation. A direct-child anchor
owns itself. Duplicate raw href attributes fail closed. No href is
resolved or followed.

Every descendant h1 through h6 of G, including H, is retained in
document order with normalized heading text and the same immediate-child
ownership and relation fields. Arbitrary page text, table/body/cell
text, numerical TOPIX or index values, raw bytes, resolved hrefs,
scores, ranks, tie-breaks, and chosen or bound GLOBAL children are not
serialized.

The exact diagnostic classifications are:

~~~text
EXPANDED_NEIGHBORHOOD_CAPTURED
SEMANTIC_HEADING_AMBIGUOUS
STRUCTURE_EXTRACTION_FAILED
~~~

The implementation never maps these classifications to F6
AVAILABLE/MISSING. It never selects, ranks, or binds a GLOBAL child.

## Offline and deterministic behavior

The offline runner accepts only an output root. It has no fetcher,
sleep, or clock parameter and does not call fetch_once_with_retry,
ensure_locked_payload, lock_first_complete_payload, network/socket
operations, run_stage_a, href resolution, or raw-lock creation,
modification, repair, or replacement.

Artifact writing uses the existing canonical JSON and atomic first-create
discipline. Reprocessing identical locked bytes reuses an existing
byte-identical artifact. A divergent existing artifact fails closed and
is never overwritten. Missing, corrupt, wrong-identity, invalid-UTF-8,
and malformed-DOM inputs are classified fail-closed according to the
design and existing lock-reader/parser conventions.

No production inventory or acquisition state is populated.
V9_006_STAGE_A_NETWORK_AUTHORIZED remains false,
V9_006_STAGE_A_EXECUTED remains false,
V9_006_STAGE_A_IMPLEMENTATION remains BLOCK, and
ACQUISITION_IMPLEMENTATION_COMPLETE remains false.

## Targeted verification

The requested target was run with synthetic offline fixtures only:

~~~text
PYTHONPATH=. pytest tests/test_v9_005_stage_a_jpx_probe.py -q
269 passed in 10.58s
~~~

The shell did not have a standalone pytest command on PATH, so the
same requested test target was executed through the repository virtual
environment as:

~~~text
$env:PYTHONPATH='.'
& .\.venv\Scripts\python.exe -m pytest tests/test_v9_005_stage_a_jpx_probe.py -q
~~~

The targeted tests cover mechanical H-to-P-to-G derivation with
nonliteral identifiers and classes; rejection of hardcoded
heading_14, section, JPX-section, and sibling indexes; semantic-heading
ambiguity; missing P or G; direct-child order and relation; the
exactly-one-direct-child P invariant; complete anchor order and
ownership; direct-child and nested anchors; exact raw href handling and
ambiguous raw attributes; heading order, normalization, and ownership;
exclusion of arbitrary, table, cell, and numerical text; exact artifact
keys without selection fields; byte-identical reprocessing and
no-overwrite on divergence; missing, corrupt, and wrong-identity locks;
invalid UTF-8 and malformed DOM; an offline seam with no forbidden
entrypoints or fetcher/sleep/clock parameters; no F6
AVAILABLE/MISSING population; and the unchanged acquisition-complete
false state.

~~~text
SOURCE_DATA_NETWORK_REQUESTS=0
git diff --check=clean
~~~

The implementation test result is recorded as awaiting GPT review. This
task does not independently call the implementation PASS.

## Prior exact-SHA GPT design review

~~~text
REVIEWED_SHA=c5b9fd5494042ab1a06d23fadbef4410e2a924a7
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
~~~

That prior PASS is the exact-SHA review of the one-level expanded
neighborhood design. The implementation remains
IMPLEMENTED_AWAITING_GPT_REVIEW pending the requested independent
review of this implementation.

## Deferred scope

The immediate-parent scope remains
EXHAUSTED_NO_GLOBAL_CHILD_EVIDENCE. No GLOBAL child URL is bound. The
one-level expanded neighborhood is diagnostic evidence only and is not
yet a production GLOBAL-child-locator scope. Any wider inspection
requires a new GPT methodology decision. V9_design_frozen remains false
and future_profitability_established remains false.
