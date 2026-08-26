# V9_006 Stage-A F6 GLOBAL child locator methodology

~~~text
task=V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_METHODOLOGY
status=AWAITING_GPT_REVIEW
scope=METHODOLOGY_ONLY
network_authorized_by_this_task=false
network_executed_by_this_task=false
source_data_network_requests=0
global_child_fetch_authorized=false
global_child_fetched=false
global_child_url_bound=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
~~~

This document defines the exact production-locator methodology that may
be implemented only after its own GPT exact-SHA review. It does not
implement the locator, execute it against the real raw lock, fetch a
GLOBAL child, or bind a URL from the observed artifact. It does not
change the already-reviewed F6 coverage-year methodology.

## 1. GPT artifact adjudication

The expanded-neighborhood artifact was independently reviewed from the
following execution SHA and locked evidence:

~~~text
execution_sha=1e1ad4f79ddbec0e8c50f9f7f56cb4fac7c261eb
raw_payload_byte_length=62923
raw_payload_sha256=22a0d8e6ef139ebe8ed94287e49a9e24a1feb08fd00f0aa36eb07eb071754433
expanded_artifact_sha256=0fdfa47d667863c6876a1ae6909172b30498c838cc2f19025db224dffd0754f9
status=EXPANDED_NEIGHBORHOOD_CAPTURED
direct_child_count=31
anchor_count=73
heading_count=10
JPX_NETWORK_REQUESTS=0
~~~

The observed artifact facts are evidence only:

- H is the Historical Index Value semantic h2.
- P is H.immediate_parent.
- G is P.immediate_parent.
- P was observed at direct-child sibling index 15.
- The next observed h2 section heading after P was
  TOPIX New Index Code information at direct-child index 19.
- The direct children strictly after P and before that next h2 owner
  were observed at indexes 16, 17, and 18.
- That bounded interval contained exactly one descendant anchor.
- The source-exact raw href observed for that anchor was
  /english/markets/indices/topix/tvdivq00000030ne-att/topixyear_e.xls.

The literal values heading_14, heading_18, the sibling indexes, the
observed heading text, and topixyear_e.xls are not production
identifiers, predicates, tie-breaks, filenames, or frozen values. The
observed section tag and JPX-section class are likewise not locator
criteria. No observed value is hardcoded by this methodology.

The supplied exact-SHA review of the implementation that produced the
artifact is recorded here:

~~~text
REVIEWED_SHA=1e1ad4f79ddbec0e8c50f9f7f56cb4fac7c261eb
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
~~~

That PASS is the supplied review of the exact implementation SHA. It
does not make this new methodology document PASS; this methodology
remains AWAITING_GPT_REVIEW.

## 2. Existing semantic-heading and container derivation

A future implementation must use the already-reviewed semantic-heading
rule, including its proper-descendant requirement, to identify exactly
one semantic heading H from the locked TOPIX root bytes.

The containers are then derived mechanically and only in this order:

~~~text
H = semantic heading
P = H.immediate_parent
G = P.immediate_parent
~~~

H, P, and G are DOM objects derived from the same locked bytes. No tag,
class, id, text, sibling index, observed heading, or observed filename is
used as a substitute for an identity. If H is not uniquely identified,
or P or G cannot be uniquely derived, the locator fails closed. It does
not expand to another ancestor level.

The locator scope is exactly G. It does not inspect the parent of G,
arbitrary siblings of G, a wider page scope, another provider, or an
alternate source. The already-reviewed immediate-parent evidence
remains exhausted; this methodology is the narrowly bounded rule for
using the one-level-expanded evidence.

## 3. Exclusive section boundary N

After H, P, and G are derived, enumerate h2 elements in document order
strictly after H. Define the boundary candidate set as the h2 elements
that satisfy all of these conditions:

1. the h2 is a proper descendant of G;
2. the h2 is not inside P;
3. the h2 is owned by a direct ELEMENT child of G that occurs strictly
   after P.

N is the earliest h2 in that candidate set in the existing deterministic
DOM document order. Its owning direct ELEMENT child of G is the
exclusive end boundary. The observed next h2, its text, and its
observed child index are not used as fixed values.

The earliest boundary must itself be uniquely mechanically resolvable.
Any ambiguity in h2 identity, proper-descendant containment, direct-child
ownership, document order, or the earliest boundary fails closed and is
reported as CHATGPT_DECISION_REQUIRED. No heading is chosen by text,
class, tag-plus-position, nearest-heading fallback, or manual judgment.
There is no automatic search for a later boundary and no ancestor or
page-scope expansion.

If no qualifying later h2 N exists, the locator fails closed as
CHATGPT_DECISION_REQUIRED. The missing boundary is not repaired by
assuming the end of G, selecting the last child, or using any observed
index.

## 4. Exact SECTION_BODY and candidate anchors

Define SECTION_BODY as every direct ELEMENT child of G that is
strictly after P and strictly before N's owning direct ELEMENT child.
The body is the ordered element interval between those two mechanically
derived boundaries.

Collect EVERY descendant <a> within SECTION_BODY in document order.
The traversal includes an anchor when it is itself a SECTION_BODY direct
element, and includes nested anchors under each SECTION_BODY direct
element. It records no non-anchor text or other page content.

The candidate count is a hard gate:

~~~text
candidate_anchor_count == 1
~~~

Exactly one candidate is required. Zero candidates or multiple
candidates fail closed. The implementation must not rank or choose
among them and must not use:

- a filename or extension heuristic;
- anchor visible text or label heuristic;
- nearest-anchor selection;
- source order as a tie-break;
- URL shape or preferred-path heuristic;
- manual selection;
- guessed href;
- alternate provider or mirror; or
- a wider scope.

Every anchor is evidence until the exact-one condition passes. This
methodology does not infer that the observed single anchor remains the
future candidate merely because the observed artifact contained one.

## 5. Raw href and URL identity

For the unique candidate only, preserve the raw href exactly as spelled
in the locked source, or fail closed if the raw attribute is absent or
ambiguous under the existing raw-attribute rules. The observed href is
not hardcoded, and no literal filename or .xls requirement is imposed.

Only after the exact-one candidate gate passes, resolve that source-exact
href mechanically against the locked TOPIX root's final resolved_url,
using the existing repository URL-resolution convention. No other base
URL is permitted. A resolution failure fails closed.

The resulting GLOBAL child URL must pass the existing URL validation:

- scheme is HTTPS;
- host is an allowed JPX domain;
- no off-domain redirect is accepted;
- no fallback, mirror, alternate provider, or guessed URL is accepted.

The locator binds exactly one child URL identity only when all preceding
mechanical gates pass. It does not fetch, inspect, or semantically
validate that object in this methodology task. A future implementation
must not treat any URL as bound when any gate fails.

## 6. Separation from GLOBAL coverage proof

This locator identifies the one official child object for the F6 GLOBAL
slot. It does not prove that the child object contains 2017--2025
observations and does not inspect index values, row counts, continuity,
filenames, or presumed spreadsheet structure.

After the child object is later fetched under separate authority, its
raw bytes must be preserved and content-locked independently. The
existing F6 GLOBAL structural-year-coverage methodology remains
authoritative: the future locked child bytes must mechanically prove
the exact covered-year set, reject malformed or ambiguous date/year
structure, and establish each required year independently before
fanout. No year is inferred from a URL, filename, row position, row
count, neighboring years, first or last date, continuity,
interpolation, or numerical index values.

The one GLOBAL slot identity may later fan out only to structurally
proven whole years under the existing F6 methodology. No monthly child
objects or per-year refetches are created by this locator.

## 7. Fail-closed contract

At minimum, a future implementation fails closed for each of the
following:

- H is not unique under the reviewed semantic-heading rule;
- P is missing or cannot be uniquely derived;
- G is missing or cannot be uniquely derived;
- P is not exactly one direct ELEMENT child of G;
- no qualifying later h2 boundary N exists;
- the earliest qualifying N is ambiguous;
- candidate_anchor_count is zero or greater than one;
- the raw href is absent or ambiguous;
- URL resolution fails;
- the resolved URL is not HTTPS;
- the resolved URL is not on an allowed JPX domain; or
- a redirect or fallback would leave the allowed JPX domain.

No fail-closed condition permits automatic scope expansion, an alternate
source/provider, a guessed URL, a filename or extension heuristic, a
manual choice, or a network retry. If the methodology cannot resolve
the required identity mechanically under these rules, the result is
CHATGPT_DECISION_REQUIRED.

## 8. Authority boundary and state

This docs-only methodology task does not authorize:

~~~text
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
V9_006_STAGE_A_IMPLEMENTATION=BLOCK
ACQUISITION_IMPLEMENTATION_COMPLETE=false
GLOBAL_CHILD_FETCH_AUTHORIZED=false
global_child_url_bound=false
~~~

No real raw-lock execution, source-data request, child fetch, or
production Stage-A execution occurs here. The current observed raw href
is evidence only and is not bound as the F6 GLOBAL URL.

The existing F6 GLOBAL structural-year-coverage methodology remains
unchanged. This methodology does not freeze the design, establish
profitability, expose historical/private/J-Quants data, or authorize
live trading.

The next action after this docs-only task is GPT exact-SHA independent
review of this GLOBAL child locator methodology.
