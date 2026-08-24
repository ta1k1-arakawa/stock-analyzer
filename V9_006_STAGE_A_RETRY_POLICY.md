# V9_006 Stage-A retry policy

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
task=V9_006_STAGE_A_RETRY_POLICY_BINDING
methodology_authority=GPT-5.6_Sol
document_role=PREFREEZE_METHODOLOGY_BINDING_RECORD
network_authorized_by_this_task=false
code_changed_by_this_task=false
probe_executed=false
v9_design_frozen=false
```

This records, exactly as decided by GPT methodology authority, the Stage-A
retry/backoff policy for `RETRIABLE_PUBLIC_PLUMBING` transport failures.
This is a pre-execution methodology binding, not post-result tuning: it
is decided before any Stage-A network request and is not a response to an
observed transport outcome. The execution agent records this decision
exactly; it does not reinterpret or extend it, and does not change code in
this task.

## Bound policy

```text
maximum_attempts_per_source_object=3
maximum_retries_per_source_object=2
backoff_seconds=[5,30]
jitter=false
```

Retries are permitted **only** before the first complete payload for a
given source object/request, and **only** for that exact same source
object/request. Every retry must preserve exactly:

- source family
- applicable period
- requested URL
- request parameters
- provider/domain

No alternate URL, provider, mirror, month, suffix, or parameter change is
permitted on any retry attempt.

## Retryable transport classes

Exactly these classes are retryable:

```text
NETWORK_TIMEOUT
CONNECTION_RESET
TEMPORARY_DNS_FAILURE
HTTP_408
HTTP_425
HTTP_429
HTTP_500
HTTP_502
HTTP_503
HTTP_504
```

No other condition is retryable. In particular, the following are
**never** retried, and each stops the attempt immediately with no
remaining attempts consumed:

- a nonretryable HTTP status;
- an off-domain or untrusted redirect;
- a response-host mismatch;
- a parser/schema failure;
- a semantic/data-quality failure;
- missing or ambiguous source evidence;
- an empty or invalid complete response classified as a source/data
  failure;
- a governance or implementation failure.

## First-complete-payload rule

Immediately preserve and hash the first complete payload before any
semantic inspection. Once a complete payload for a source object is
locked, that object must never be fetched again within the same Stage-A
execution. Parser or semantic repair may only reprocess the exact same
locked bytes; it is never grounds for a refetch.

## Exhaustion

If all attempts (up to the maximum) fail with only retryable transport
conditions:

```text
failure_class=PLUMBING_FAILURE_RETRIABLE
```

The Stage-A run STOPs. This must never be converted into
`SOURCE_OR_DATA_FEASIBILITY_FAILURE`. No new Stage-A run may start
automatically after exhaustion, and the same human authorization must not
be reused for another attempt. Return to GPT methodology authority and
obtain a fresh point-of-use human authorization before any new run.

## Nonretryable stop

A nonretryable condition stops the attempt immediately; no remaining
attempts are used, and no retry is issued for it under any circumstance.

## Scope

This task freezes retry behavior only. It does not invent or change any
broader failure-class semantics beyond what
`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` and
`AI_REAL_EXECUTION_RUNBOOK.md` already bind (transport classification
before a complete payload, content-lock after it, and the existing
`PLUMBING_FAILURE_RETRIABLE` / `DATA_QUALITY_FAILURE` / `SOURCE_OR_DATA_
FEASIBILITY_FAILURE` distinctions). F1's `TERMINAL_SEED`-only role, F2-F7
roots/traversal/mapping rules, F4's ratio orientation, the F7 acquisition
envelope, and the V9_005_HIGH_2B signal-grid binding are all unchanged by
this document.

## Rationale

This is public retriable plumbing (`RETRIABLE_PUBLIC_PLUMBING` per
`AI_RESEARCH_EXECUTION_RULES.md` SS6.1 / `AI_REAL_EXECUTION_RUNBOOK.md`
SS4.1), and bounded retry of a transient transport failure reduces false
transport negatives without allowing any result-dependent source
substitution or repeated semantic sampling: retries never change the
request, never occur after a complete payload, and never apply to a
semantic or data-quality outcome. This binding is made before any Stage-A
network boundary is crossed, not after observing a transport result.

## Provenance of the previously present constants

`src/v9_005_stage_a_jpx_probe.py` already contains `MAX_ATTEMPTS = 3`,
`MAX_RETRIES = 2`, and `BACKOFF_SECONDS: tuple[int, ...] = (5, 30)`, and
`src/v8c_transport.classify_transport_exception` already classifies
exactly `HTTP_408`, `HTTP_425`, `HTTP_429`, `HTTP_500`, `HTTP_502`,
`HTTP_503`, `HTTP_504` (via `RETRYABLE_HTTP_CODES`), plus
`NETWORK_TIMEOUT`, `CONNECTION_RESET`, and `TEMPORARY_DNS_FAILURE`, as
retryable. These values happen to numerically match the policy bound
above. That prior presence in the codebase was an unauthorized
execution-agent implementation choice, not a GPT methodology decision, and
must not be read as having constituted approval at the time it was
written. The retry policy is now, for the first time, independently
selected and authorized by GPT methodology authority exactly as recorded
in this document. No code is changed by this task; a subsequent
implementation-remediation task will separately verify and, if necessary,
repair `src/v9_005_stage_a_jpx_probe.py` against this exact contract.

## Authority created

```text
NETWORK_REQUESTS=0
DATA_ACQUIRED=false
CODE_CHANGED=false
PROBE_EXECUTED=false
HUMAN_GATE_CONSUMED=false
T1_OR_DESIGN_FREEZE_AUTHORITY_CREATED=false
```

This binding is a docs-only methodology decision. It does not authorize
network access, data acquisition, T1 membership generation or opening,
model fitting, backtesting, profit calculation, or V9 design freeze, and
does not consume the human's existing chat-given Stage-A authorization.

## Next action

`GPT_EXACT_SHA_V9_006_STAGE_A_RETRY_POLICY_REVIEW`: obtain GPT's
independent exact-SHA review of this retry-policy binding. A future,
separately authorized implementation task would then verify (and repair
if necessary) `src/v9_005_stage_a_jpx_probe.py` against this exact
contract, still without executing any real network request until a fresh,
separate, explicit Stage-A human network authorization is obtained after
that implementation's own GPT exact-SHA review PASS.
