# V9_006 Stage-A F6 production root and GLOBAL raw acquisition design

~~~text
task=V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_DESIGN
status=REMEDIATED_AWAITING_GPT_REVIEW
scope=F6_PRODUCTION_ROOT_AND_ONE_GLOBAL_CHILD_RAW_OBJECT_ONLY
network_authorized_by_this_task=false
network_executed_by_this_task=false
human_authorization_consumed=false
source_data_network_requests=0
GLOBAL_CHILD_FETCH_AUTHORIZED=false
GLOBAL_CHILD_FETCHED=false
GLOBAL_CHILD_CONTENT_INSPECTED=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
~~~

This docs-only design defines a future, fresh, human-authorized public F6
production acquisition. It is not full Stage A. It does not authorize code,
a real raw-lock execution, a human gate, a network request, a GLOBAL child
fetch, child-content inspection, coverage proof, inventory fanout, or a
design freeze.

## 1. GPT adjudication and offline evidence

The supplied exact-SHA review is recorded as:

~~~text
REVIEWED_SHA=282f13d904b692da673294d18020cfc70610f34b
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_F6_GLOBAL_LOCATOR_MEDIUM_1_FAILURE_CLASS_DRIFT=RESOLVED
V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_METHODOLOGY=PASS
V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_IMPLEMENTATION=PASS
~~~

Subsequent offline execution of the reviewed locator is recorded as safe
evidence only:

~~~text
execution_sha=282f13d904b692da673294d18020cfc70610f34b
status=GLOBAL_CHILD_LOCATOR_RESOLVED
candidate_anchor_count=1
diagnostic_root_raw_sha256=22a0d8e6ef139ebe8ed94287e49a9e24a1feb08fd00f0aa36eb07eb071754433
diagnostic_root_raw_byte_length=62923
JPX_NETWORK_REQUESTS=0
GLOBAL_CHILD_FETCHED=false
GLOBAL_CHILD_CONTENT_INSPECTED=false
F6_GLOBAL_CHILD_LOCATOR_OFFLINE_EXECUTION=PASS
F6_DIAGNOSTIC_GLOBAL_CHILD_URL_RESOLVED=true
F6_PRODUCTION_GLOBAL_CHILD_URL_BOUND=false
~~~

Any diagnostic raw href or mechanically resolved diagnostic URL is
OBSERVED_DIAGNOSTIC_EVIDENCE_ONLY. It is not a production constant,
fallback, requested child URL, or authorization to fetch.

The diagnostic raw identity F6_ROOT_STRUCTURE_DIAGNOSTIC is permanently
separate from the production root identity TOPIX_DISCOVERY_ROOT. A future
executor must never copy, alias, promote, or substitute a diagnostic root
lock or artifact for a production root lock.

## 2. Future operation scope and raw identities

The future operation acquires exactly two F6 source objects in one new
production Stage-A output root that is distinct from the diagnostic output
root. The machine-local production output-root path is runtime state and
must not be committed.

~~~text
ROOT
source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
applicable_period=TOPIX_DISCOVERY_ROOT
requested_url=TOPIX_ROOT_URL

CHILD
source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
applicable_period=TOPIX_GLOBAL_2017_2025
requested_url=the URL mechanically resolved from the newly locked production ROOT
~~~

No F1, F2, F3, F4, F5, or F7 request is permitted. No additional F6 root,
child, month, year, mirror, provider, filename, format, or URL is
permitted.

## 3. Mandatory future execution sequence

### A. Pre-authorization readiness

Before asking for, accepting, or consuming any human authorization, the
future executor must mechanically PASS all applicable readiness predicates:

- exact repository, authoritative branch, clean working tree, local HEAD,
  and authoritative remote HEAD;
- required frozen signal-grid blob binding at the execution commit where
  Stage-A binding is required;
- exact reviewed locator-implementation SHA binding;
- the reviewed implementation binding for the future production-acquisition
  executor itself;
- canonical .venv-real-execution existence and exact interpreter identity;
- Python, dependency, synthetic parser, environment-lock, and environment
  freeze checks required by AI_REAL_EXECUTION_RUNBOOK and
  REAL_EXECUTION_PYTHON_ENVIRONMENT;
- CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE=YES;
- a new production output root that does not collide with diagnostic or
  existing production state;
- absent production ROOT and CHILD raw locks for the exact identities in
  section 2;
- absent conflicting gate receipt, execution binding, audit state, result,
  or other durable receipt; and
- a provably unused stage-specific one-shot authorization state.

Each failure is PRE_GATE_FAILURE: the gate remains unconsumed and JPX
request count is zero. No preflight repair may change methodology. The
complete preflight must be rerun after any permitted repair.

### B. Fresh point-of-use human gate

The future operation requires this new, stage-specific confirmation
identity:

~~~text
V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_ONE_SHOT
~~~

It is fresh, explicit, point-of-use, non-reusable, and scoped only to this
two-object F6 production raw acquisition. It is not the prior F6 diagnostic
authorization, a general Stage-A authorization, a Stage-B authorization, or
production-trading authority. This design neither grants nor consumes it.

### C. Production ROOT acquisition

After the gate is consumed, fetch exactly TOPIX_ROOT_URL for the ROOT raw
identity. Apply the frozen Stage-A transport policy without alteration:

~~~text
maximum_attempts=3
maximum_retries=2
backoff_seconds=[5,30]
jitter=false
retry_scope=only_before_first_complete_payload_for_the_same_exact_request
~~~

Only the frozen retryable transport classes are retryable. On the first
complete ROOT payload, immediately raw-lock and hash it under
TOPIX_DISCOVERY_ROOT before locator parsing or any semantic inspection.
After that complete ROOT payload, no ROOT refetch is permitted.

### D. Locator on newly locked production ROOT

Run the exact GPT-reviewed F6 GLOBAL child locator against the newly locked
production ROOT bytes. The diagnostic observed URL must not be used as a
requested child URL, fallback, comparison selector, or repair input.

If the locator does not resolve exactly one valid HTTPS allowed-JPX child
identity, STOP with the production ROOT lock preserved. Do not request a
CHILD, widen scope, choose a different candidate, or substitute a
diagnostic URL.

### E. GLOBAL CHILD raw acquisition and stop

Only after locator success, fetch exactly its mechanically resolved child
URL. Apply the identical frozen transport policy only to that exact CHILD
request. On the first complete CHILD payload, immediately raw-lock and hash
it under TOPIX_GLOBAL_2017_2025.

After the CHILD lock is durable, STOP. This execution must not:

- parse, open, or otherwise inspect CHILD content;
- open a spreadsheet;
- inspect years, dates, rows, columns, or numerical TOPIX values;
- determine 2017--2025 coverage;
- populate F6 AVAILABLE or MISSING cells;
- fan out the GLOBAL slot; or
- refetch the CHILD because later parsing, schema, or data-quality work
  fails.

A later separately reviewed offline task may inspect only the exact locked
CHILD bytes and may design or implement deterministic structural
covered-year parsing if the format permits it.

## 4. Failure and recovery discipline

The gate is not reusable after any post-gate state. Existing frozen retries
remain permitted only before the first complete payload for the current
exact object and never change its requested URL, provider, period, or
parameters.

~~~text
pre-gate readiness failure
  -> no gate consumption; JPX requests=0; STOP

complete production ROOT plus locator identity/methodology-resolution
failure (semantic-heading ambiguity; missing or ambiguous P or G; invalid
P direct-child identity; missing or ambiguous N; candidate_anchor_count
!= 1; missing or ambiguous raw href; child URL resolution failure,
non-HTTPS result, or off-domain identity failure)
  -> preserve ROOT lock; CHILD requests=0; STOP/CHATGPT_DECISION_REQUIRED;
     no fallback URL, no diagnostic-URL substitution, no scope expansion

complete production ROOT plus locator IMPLEMENTATION_FAILURE (malformed
or corrupt ROOT raw-lock state, invalid UTF-8/DOM, or another failure the
reviewed locator classifies as IMPLEMENTATION_FAILURE)
  -> preserve ROOT lock; CHILD requests=0; STOP/IMPLEMENTATION_FAILURE

complete production CHILD
  -> preserve CHILD lock; STOP; no content-driven refetch
~~~

The two locator-failure cases above preserve the exact, already-reviewed
F6 GLOBAL locator failure taxonomy; neither collapses into the other, and
neither is a blanket STOP/CHATGPT_DECISION_REQUIRED outcome for every
locator failure. The already-locked production ROOT is never refetched
because of a software, parser, or implementation repair to either case.
No CHILD request occurs while either failure remains. A later, separately
reviewed repair or reprocessing task may operate only on the exact
preserved ROOT bytes; it does not reuse the consumed
V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_ONE_SHOT authorization,
and an offline same-byte repair does not require, and must not claim,
that the gate was unconsumed. Any later network continuation past either
failure -- a fresh locator attempt, a CHILD request, or any other network
step -- requires its own separately reviewed GPT/human authority
decision; none is pre-authorized by this document.

No failure under this contract is a strategy or profitability failure. No
failure permits provider, URL, filename, mirror, format, period, scope, or
retry-policy substitution.

## 5. Safe future execution report

The real-execution report must not include raw payload bytes or ordinary raw
URL text. It may report only safe evidence such as:

- exact local and remote HEADs and clean boolean;
- gate-consumed boolean;
- ROOT request count, CHILD request count, and total JPX request count;
- ROOT and CHILD raw SHA-256 and byte length;
- requested/resolved URL hashes or approved equality booleans;
- locator status and candidate count;
- safe failure class;
- authorization_reusable=false; and
- second_execution_allowed=false unless a later GPT and human authority
  explicitly changes that state.

## Medium-1 failure-class remediation

The supplied GPT review of this design at the exact SHA below identified
`V9_006_F6_PRODUCTION_RAW_DESIGN_MEDIUM_1_LOCATOR_FAILURE_CLASS_COLLAPSE`:

~~~text
REVIEWED_SHA=3c6873fea13d7bca8d16ae38bcba263ef6b4f461
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=1
RESULT=BLOCK
~~~

Section 4's blanket rule -- `complete production ROOT plus locator failure
-> preserve ROOT lock; CHILD requests=0; STOP/CHATGPT_DECISION_REQUIRED` --
collapsed the already GPT-reviewed and PASSed F6 GLOBAL child locator's two
distinct failure classes into one undifferentiated stop condition. The
remediation replaces it with the two mechanically separated cases in
section 4 above, reproducing the reviewed locator's own governed
`CHATGPT_DECISION_REQUIRED` identity/methodology-resolution list and its
separate `IMPLEMENTATION_FAILURE` class for malformed/corrupt raw-lock
state or invalid UTF-8/DOM, exactly as recorded in
`V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_METHODOLOGY.md` §7 and
`V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_IMPLEMENTATION_REVIEW.md`. Both
cases preserve the ROOT lock, make zero CHILD requests, and STOP; only the
public failure label and the explicit no-refetch/no-reuse/no-pre-
authorization consequences are now spelled out per case.

This is a docs-only, one-finding remediation. It does not change the
root->locator->child order, diagnostic-URL non-promotion, raw identities,
retry policy, the fresh one-shot human gate, pre-gate environment
readiness, child raw-lock-then-STOP, the child-content-parsing
prohibition, or any F1/F2/F3/F4/F5/F7 rule. No code, test, network
request, human-gate consumption, GLOBAL child fetch, or design freeze
occurred. This remediation is `REMEDIATED_AWAITING_GPT_REVIEW`, not
PASS/RESOLVED, and is not self-called PASS by the execution agent. The
next action is GPT exact-SHA independent review of this remediation.

## 6. Authority boundary

~~~text
GLOBAL_CHILD_FETCH_AUTHORIZED=false
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
V9_006_STAGE_A_IMPLEMENTATION=BLOCK
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
~~~

The next action after this document is GPT exact-SHA independent review of
this production root and GLOBAL raw acquisition design remediation.
