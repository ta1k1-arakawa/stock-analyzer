# V9_006 Stage-A F6 GLOBAL child locator implementation review

~~~text
task=V9_006_F6_GLOBAL_LOCATOR_MEDIUM_1_FAILURE_CLASS_DRIFT
status=REMEDIATED_AWAITING_GPT_REVIEW
implementation_parent_sha=0d262b0b912ebeb01910992cf59ab9927bacb440
reviewed_implementation_sha=e93d0cbef674e8b679442b246ddb9a23fa3f05c1
offline_only=true
real_raw_lock_execution=false
source_data_network_requests=0
global_child_fetch_authorized=false
global_child_fetched=false
global_child_content_inspected=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
~~~

This document records the execution-agent implementation boundary. It
does not call the implementation PASS. GPT-5.6 Sol remains the final
independent reviewer.

## Method implemented

The implementation adds parse_f6_global_child_locator and the
read-only run_f6_global_child_locator_offline seam in
src/v9_005_stage_a_jpx_probe.py. Both reuse the reviewed F6 full-DOM
parser, strict UTF-8 decoder, normalization, semantic-heading rule,
proper-descendant helpers, direct-child ownership helpers,
source-exact raw-href extraction, and URL resolution/validation.

From the existing F6 root lock it derives, in order:

~~~text
H = reviewed semantic heading, including proper-descendant requirement
P = H.immediate_parent
G = P.immediate_parent
N = earliest qualifying later h2 in document order
SECTION_BODY = direct G children strictly after P and before N's owner
candidate_anchor_count = every descendant a in SECTION_BODY
~~~

P is required to occur exactly once as a direct ELEMENT child of G.
N must be a proper descendant of G, outside P, and owned by a direct G
child strictly after P. The first and only qualifying document-order
boundary is used; there is no later-boundary fallback or scope expansion.

The candidate anchor gate requires exactly one descendant anchor. The
implementation does not rank candidates or use filename, extension,
visible-text, URL-shape, nearest-anchor, manual, source-order tie-break,
alternate-provider, or guessed-URL logic. The unique anchor's raw href is
preserved exactly and is resolved only against the locked TOPIX root
resolved_url, then validated by the existing HTTPS and allowed-JPX
domain rules.

The successful diagnostic result uses the dedicated
V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_RESULT_V1 schema and exposes
only locked-root provenance, H/P/G identities, the boundary identity,
the SECTION_BODY direct-child identities, the exact-one anchor's DOM
path/raw href, and its mechanically resolved child URL identity. It
contains no child bytes, child content, year proof, ranking, score, or
F6 inventory status. Fail-closed locator ambiguity is reported through
the governed CHATGPT_DECISION_REQUIRED failure class; malformed or corrupt
locks and invalid UTF-8/DOM remain fail-closed implementation failures.

## Authority boundary

The offline seam reads only the existing
F6_ROOT_STRUCTURE_DIAGNOSTIC lock through
read_f6_root_structure_diagnostic. It has no fetcher, sleep, clock,
network, retry, raw-lock writer, child fetch, child-content parser, or
run_stage_a call. It does not populate AVAILABLE or MISSING, change
ACQUISITION_IMPLEMENTATION_COMPLETE, authorize Stage A, or freeze the
V9 design.

Targeted verification used synthetic locked bytes only. The required
command was run as:

~~~text
PYTHONPATH=. pytest tests/test_v9_005_stage_a_jpx_probe.py -q
296 passed in 10.67s
git diff --check
clean
~~~

The tests cover nonliteral H/P/G/N derivation, observed-literal decoys,
all required boundary and exact-one-anchor fail-closed gates, direct and
nested anchors, exact raw href spelling, final-URL-only resolution,
HTTPS/allowed-domain validation, malformed and invalid locks, no child
inspection, no acquisition entrypoint, no inventory status, and
ACQUISITION_IMPLEMENTATION_COMPLETE=false. No real stored raw lock was
executed and no GLOBAL child was fetched.

## Supplied prior methodology review

~~~text
REVIEWED_SHA=0d262b0b912ebeb01910992cf59ab9927bacb440
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
~~~

The tuple above is the supplied exact-SHA review of the methodology
parent. It is not an independent PASS of this implementation. The next
action is GPT exact-SHA review of the committed implementation.

## Medium-1 failure-class remediation

The supplied GPT review of the implementation SHA identified
V9_006_F6_GLOBAL_LOCATOR_MEDIUM_1_FAILURE_CLASS_DRIFT:

~~~text
REVIEWED_SHA=e93d0cbef674e8b679442b246ddb9a23fa3f05c1
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=1
RESULT=BLOCK
~~~

The remediation adds only locator-local exception translation. Ambiguous
or duplicate raw href extraction, URL-resolution failure, non-HTTPS
resolved URLs, and non-allowed-JPX resolved URLs now expose
failure_class=CHATGPT_DECISION_REQUIRED for the F6 GLOBAL locator. Missing
raw href already had that class and remains unchanged.

The shared _resolve_locked_page_link helper is unchanged. Focused
regression coverage proves the pre-existing F2, F3, F4, and F7 failure
classes remain IMPLEMENTATION_FAILURE. Malformed or corrupt locks remain
offline, fail-closed implementation failures. The full targeted command
now reports 298 passed in 11.44s with SOURCE_DATA_NETWORK_REQUESTS=0.

This one-finding remediation does not change H/P/G/N, SECTION_BODY,
candidate count, raw-href spelling, URL base, locator methodology, child
fetch/content authority, inventory population, or acquisition-complete
state. It remains awaiting GPT exact-SHA review and is not self-called
PASS by the execution agent.
