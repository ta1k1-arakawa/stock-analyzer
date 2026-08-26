# V9_006 Stage-A F6 one-level expanded neighborhood probe design

~~~text
task=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_DESIGN
status=AWAITING_GPT_REVIEW
scope=DIAGNOSTIC_ONLY
network_authorized_by_this_task=false
network_executed_by_this_task=false
source_data_network_requests=0
global_child_selected=false
global_child_url_bound=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
~~~

This is docs-only. It records the GPT adjudication of the already completed
offline section-neighborhood artifact and defines exactly one further
offline diagnostic scope. It does not authorize code, a raw-lock change,
network access, source-data acquisition, production Stage-A execution, or
selection of an F6 GLOBAL child.

The existing immediate-parent scope is exhausted because it contains no
GLOBAL-child evidence. The next diagnostic scope is exactly one ancestor
level above that scope: the immediate parent of the immediate parent of the
semantic heading. This document does not turn that diagnostic scope into a
production locator and does not permit automatic expansion beyond it.

## 1. GPT artifact adjudication to record

The offline execution used the reviewed repository SHA below. GPT
independently reviewed the resulting artifact. The raw payload is not
copied into this document.

~~~text
reviewed_sha=780da8ddc3185357caeda8a6122cabb3db72e289
existing_raw_payload_byte_length=62923
existing_raw_payload_sha256=22a0d8e6ef139ebe8ed94287e49a9e24a1feb08fd00f0aa36eb07eb071754433
neighborhood_artifact_sha256=470a3095bad6c117959f459f639cbb7dc21c8bacc0c7fbb1f2c44d45007ec368

status=NEIGHBORHOOD_CAPTURED
semantic_heading_present=true
parent_container_present=true
child_count=1
anchor_count=0
heading_count=1
JPX_NETWORK_REQUESTS=0
~~~

The exact GPT artifact-review tuple supplied for this task is:

~~~text
REVIEWED_SHA=780da8ddc3185357caeda8a6122cabb3db72e289
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
~~~

This tuple records the supplied GPT review of the completed artifact at
the reviewed SHA. It is not a PASS judgment by the execution agent on
this new design; this design remains AWAITING_GPT_REVIEW.

The reviewed artifact establishes these observed facts:

- the semantic heading is an h2 carrying the class token
  heading-title;
- its immediate parent P is a div;
- P has exactly one ELEMENT child, the semantic heading itself;
- P has zero descendant anchors; and
- the immediate parent G of P was observed as a section carrying the
  class token JPX-section.

The literal value heading_14, all sibling indexes, the tag literal
section, and the class literal JPX-section are observed evidence only.
They are not frozen identifiers, locator criteria, or production
methodology. In particular, no future implementation may hardcode
heading_14, require the observed sibling indexes, or select G by
matching section or JPX-section; G must be derived mechanically.

The GPT binding from this evidence is:

~~~text
V9_006_STAGE_A_F6_IMMEDIATE_PARENT_SCOPE=EXHAUSTED_NO_GLOBAL_CHILD_EVIDENCE
V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR=BLOCKED_PENDING_ONE_LEVEL_EXPANDED_OFFLINE_EVIDENCE
~~~

No GLOBAL child URL is bound. No href is guessed, resolved, or followed.
The execution AI must not determine which anchor, if any, is the GLOBAL
child.

## 2. Exact next diagnostic scope

The already-reviewed semantic-heading identity rule first identifies one
semantic heading H from the locked raw payload. The rule includes the
proper-descendant requirement: the target h2 itself cannot satisfy the
required descendant label occurrence. If that rule does not resolve to
exactly one semantic heading, the probe fails closed.

After H is identified, derive the containers mechanically and only in
this order:

~~~text
H = semantic heading
P = H.immediate_parent
G = P.immediate_parent
~~~

The probe scope is exactly G. It is one ancestor level above the
already-reviewed immediate-parent scope P. It includes the direct
ELEMENT children of G and the descendants of G required by section 4.
It does not include the parent of G, arbitrary siblings of G, a wider
page scope, or any automatically discovered ancestor.

If H, P, or G cannot be derived uniquely under the already-reviewed
rules, the probe fails closed. A unique H followed by an inability to
derive or extract P or G is a structure-extraction failure; it is not
permission to expand the scope. No automatic scope expansion beyond G
is allowed. Any need to inspect a wider scope requires a new GPT
methodology decision.

G remains diagnostic-only. This probe does not define, validate, rank,
select, or bind a production F6 GLOBAL child locator.

## 3. Future offline-only input and execution boundary

The future implementation may read only the existing
F6_ROOT_STRUCTURE_DIAGNOSTIC raw lock. It must use the exact locked bytes
and the raw-lock provenance already preserved for that payload.

The future implementation must not:

- fetch, refetch, or inspect a source outside the existing raw lock;
- create, modify, replace, or repair the raw lock;
- accept or invoke a fetcher, sleep, clock, retry, redirect, or network
  operation;
- resolve or follow any recorded href;
- call production Stage-A acquisition or populate F6 inventory; or
- choose, rank, or bind a GLOBAL child URL.

The existing one-shot network authorization for the root diagnostic is not
reused. This offline diagnostic consumes no network authorization and
creates no Stage-A authority. Invalid UTF-8, malformed DOM structure,
ambiguous raw attributes, missing required identity, or any other
non-deterministic extraction condition fails closed under the existing
diagnostic discipline.

## 4. Future artifact design

The future offline probe produces this separate artifact:

~~~text
V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT.json
~~~

It uses a dedicated schema/version identifier distinct from the existing
root-structure and immediate-parent section-neighborhood artifacts:

~~~text
schema_version=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_V1
diagnostic=V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE
~~~

The serialized top-level fields are exactly the following diagnostic
fields:

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

parent_container is P; expanded_container is G. The first six
provenance values are copied from the existing raw-lock metadata, not
refetched or recomputed from a different source. The artifact contains no
raw payload bytes.

### 4.1 Identity fields

semantic_heading, parent_container, and expanded_container each
record exactly:

~~~text
dom_path
tag
id
classes
~~~

The dom_path, tag, ID representation, and normalized class-token
representation use the already-reviewed deterministic DOM rules. No
text-node index, literal fragment value, observed sibling index, observed
tag, or observed class token is promoted into the identity rule.

### 4.2 Direct ELEMENT children of G

children records every direct ELEMENT child of G, in document order.
Each record contains:

~~~text
dom_path
tag
id
classes
relation_to_P
~~~

relation_to_P is exactly one of:

~~~text
BEFORE_P
P
AFTER_P
~~~

The P relation is assigned to the exact element P, not to an element
selected by tag, class, text, or position guessed from the observed
artifact. If P is not exactly one direct ELEMENT child of G, extraction
fails closed.

### 4.3 Descendant anchors of G

anchors records every descendant <a> element within G, in document
order. Each record contains:

~~~text
dom_path
normalized_visible_text
raw_href
owning_immediate_element_child_of_G
owning_child_relation_to_P
~~~

owning_immediate_element_child_of_G records the identity of the direct
ELEMENT child of G that owns the anchor, using:

~~~text
dom_path
tag
id
classes
~~~

owning_child_relation_to_P is exactly one of BEFORE_P, P, or
AFTER_P, determined from that owning direct child. Anchor ownership is
mechanically derived by walking the anchor's element ancestors to the
direct child of G; no anchor is selected because of its label, URL, or
position.

normalized_visible_text uses the already-reviewed visible-text rule:
Unicode whitespace runs are collapsed and trimmed, with no second HTML
entity decode. raw_href is the exact raw href attribute spelling from
the locked source, or null when the attribute is absent. It is never
resolved or followed. If the raw attribute cannot be preserved
unambiguously, extraction fails closed.

### 4.4 Descendant headings of G

headings records every descendant h1 through h6 element within G,
including H, in document order. Each record contains:

~~~text
dom_path
tag
normalized_heading_text
owning_immediate_element_child_of_G
owning_child_relation_to_P
~~~

The owning-child identity uses the same four fields as the anchor owner:
dom_path, tag, id, and classes. The owning-child relation is exactly
BEFORE_P, P, or AFTER_P. Heading text uses the same
whitespace-run-collapse-and-trim normalization as the reviewed label
rule, without recording arbitrary surrounding page text.

### 4.5 Exclusions and non-selection

The artifact must not record:

- arbitrary non-anchor/non-heading page text;
- table, body, or cell text;
- numerical TOPIX or index observations;
- raw payload bytes;
- resolved hrefs or fetched child content;
- candidate scores, rankings, or tie-breaks; or
- a selected, bound, or otherwise designated GLOBAL child URL.

Every anchor is evidence only. Execution AI MUST NOT determine which
anchor is the GLOBAL child.

## 5. Diagnostic statuses

The probe status is exactly one of the following values:

~~~text
EXPANDED_NEIGHBORHOOD_CAPTURED
SEMANTIC_HEADING_AMBIGUOUS
STRUCTURE_EXTRACTION_FAILED
~~~

EXPANDED_NEIGHBORHOOD_CAPTURED means that the already-reviewed rule
identified exactly one H, P and G were derived as specified, every
required child/anchor/heading fact was extracted deterministically, and
the artifact was created or an identical artifact was reused.

SEMANTIC_HEADING_AMBIGUOUS means that the existing semantic-heading
rule did not resolve exactly one H, including a zero, multiple, wrong-tag,
wrong-class, or inconsistent proper-descendant outcome. It is not an F6
AVAILABLE or MISSING result.

STRUCTURE_EXTRACTION_FAILED means that H resolved uniquely but P,
G, or any required structural field could not be extracted
deterministically. It is not an F6 AVAILABLE or MISSING result.

No one of these diagnostic statuses is automatically mapped to F6
AVAILABLE or MISSING. None selects, ranks, or binds a GLOBAL child
URL.

## 6. Determinism and no-overwrite

The artifact is derived only from the existing locked bytes and copied raw
provenance. The same locked bytes and the same reviewed rules must produce
a byte-identical artifact, including stable field ordering, document
ordering, and deterministic JSON encoding under the existing artifact
serialization discipline.

The artifact write is atomic and no-overwrite:

- if the target does not exist, create it atomically;
- if it exists with byte-identical content, reuse it; and
- if it exists with different content, fail closed and never overwrite it.

No parser or serialization repair may change the raw lock or silently
replace a divergent artifact. A need to inspect any scope wider than G
requires a new GPT methodology decision.

## 7. State and authority boundary

This design records the following state transition and preserves all prior
PASS, RESOLVED, and deferred-LOW states:

~~~text
V9_006_F6_NEIGHBORHOOD_MEDIUM_1_SEMANTIC_HEADING_SELF_ACCEPTED_AS_DESCENDANT=RESOLVED
V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION=PASS
V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_EXECUTION=COMPLETE
V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT=NEIGHBORHOOD_CAPTURED
V9_006_STAGE_A_F6_IMMEDIATE_PARENT_SCOPE=EXHAUSTED_NO_GLOBAL_CHILD_EVIDENCE
V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR=BLOCKED_PENDING_ONE_LEVEL_EXPANDED_OFFLINE_EVIDENCE
V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_DESIGN=AWAITING_GPT_REVIEW

V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
V9_006_STAGE_A_IMPLEMENTATION=BLOCK
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
~~~

These bindings create no network, source-data, private/sealed, historical-
evaluation, J-Quants, live-trading, or design-freeze authority. The next
action after this docs-only task is GPT exact-SHA review of this one-level
expanded neighborhood probe design.
