# V9_006 F1 Semantic Successor Public Acquisition Design

```text
task=V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION
status=DESIGN_AWAITING_GPT_EXACT_SHA_REVIEW
operation_class=RETRIABLE_PUBLIC_PLUMBING
network_acquisition_authorized=false
fresh_human_authorization_required=true
single_execution_invocation=true
```

## 1. New successor identity and purpose

This is a new successor public-acquisition identity. It does not reopen,
retry, reset, reuse, or otherwise alter the terminated historical
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1` identity, its consumed gate, or its
receipt.

If and only if all later authorization conditions are satisfied, this identity
may acquire exactly these two public JPX objects, in order:

1. one fresh official F1 discovery-root payload; and
2. only after the frozen semantic successor locator uniquely succeeds on that
   locked discovery root, one selected F1 TERMINAL spreadsheet payload.

It does not acquire F2--F7, execute Phase 2, parse terminal month `T`, infer
`T`, or make any profitability or strategy finding.

## 2. Reviewed offline-proof binding

The predecessor offline proof is complete and binds the reviewed locator
methodology, not a network authority:

```text
result=SUCCESSOR_LOCATOR_MATCHED
input_payload_sha256=ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f
mechanical_candidate_count=2
qualifying_candidate_count=1
selected_raw_href_sha256=ee97b7976663aa4dd55f9f02d33e96ceb66ad76bb43fd2e4523a31fe4d4a6ec9
selected_resolved_url_sha256=a7088b6c7e5ea028ffad54bd95e835e32068dfafa324d737e2cef0424f90e613
structural_evidence_sha256=92d33292a19e4880010260a382c8a4c51971d85c0bbc5b1548f609584f204a2d
network_requests=0
OFFLINE_PROOF_BINDING=PASS
SECOND_EXECUTION_ALLOWED=false
```

The old Phase-1 identity remains terminal. This proof does not make its gate
reusable and does not permit another old-identity execution.

## 3. Frozen source and locator contract

The discovery root is exactly the existing official JPX F1 discovery root.
The runtime uses the existing reviewed `validate_jpx_url` contract without a
new validator or alternate-domain rule.

The frozen `V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR` methodology is used exactly:
the semantic locator is the sole selector after discovery-root locking. There
is no position, filename, extension, current-date, apparent-recency, provider,
or date-substitution rule, and no fallback provider. The selected terminal URL
remains private runtime state and is never emitted in safe evidence.

## 4. Authority boundary

This design grants no network authority.

```text
network_acquisition_authorized=false
fresh_human_authorization_required=true
old_phase1_retry=false
old_phase1_gate_reusable=false
phase2_execution=false
terminal_month_T_parsing_authorized=false
f2_bridge_acquisition_authorized=false
private_or_sealed_access_authorized=false
future_profitability_established=false
```

Only after all of the following may this new identity receive
`STANDING_RETRIABLE_PUBLIC_PLUMBING_AUTHORITY`:

1. GPT exact-SHA PASS of this design;
2. GPT exact-SHA PASS of its implementation; and
3. fresh explicit human approval at the point of use.

That standing authority is exclusive to this successor identity. It cannot be
reused for historical Phase-1, Phase 2, private or sealed data, production,
another study, or any other acquisition identity.

`STANDING_RETRIABLE_PUBLIC_PLUMBING_AUTHORITY` authorizes the bounded ROOT and
TERMINAL retry loops only inside one fresh, atomic real-execution invocation.
It does not authorize launching a second invocation of this acquisition
identity after that invocation crosses the durable execution-start boundary.

## 5. Durable execution identity and start boundary

After every preflight and the fresh human approval have passed, but before the
first network attempt, the implementation creates an exclusive durable
successor execution-binding receipt in a new successor-acquisition state root.
The machine-local state-root path is private and is never committed or emitted.
It is distinct from the terminated old Phase-1 root and every other
acquisition or production root.

On a fresh start, that state root must contain no conflicting acquisition
binding, result, raw lock, attempt state, or receipt. Creation is exclusive
and no-overwrite. The implementation must never delete, reset, overwrite, or
otherwise alter conflicting state to continue.

The receipt contains only safe binding values:

```text
task=V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION
design_git_sha=exact_GPT_PASS_design_SHA_supplied_to_implementation
implementation_git_sha=exact_GPT_reviewed_implementation_SHA_at_execution
operation_class=RETRIABLE_PUBLIC_PLUMBING
execution_started=true
```

It contains no raw human-authorization identity, path, or URL. Once this
execution-start receipt is durably created:

```text
SECOND_EXECUTION_ALLOWED=false
```

This remains false for this identity regardless of PASS, BLOCK, software
failure, transport failure, or crash. Any later invocation discovering the
receipt must stop with `GOVERNANCE_FAILURE` before network I/O. It may not use
the receipt as a restart token.

## 6. Bounded transport and complete-payload discipline

The following policy applies independently to the discovery root and, if it is
reached, the selected TERMINAL object:

```text
maximum_network_attempts_per_object=3
attempt_1_delay_seconds=0
attempt_2_delay_seconds=2
attempt_3_delay_seconds=5
retry_condition=no_complete_HTTP_200_payload_obtained_for_that_object
provider_and_endpoint=same_frozen_provider_and_exact_endpoint_only
provider_or_date_substitution=false
```

Attempt 1 is immediate. A non-complete request outcome may consume an attempt.
Retries occur only while no complete HTTP-200 payload has been obtained for
that object. After three unsuccessful attempts for an object, the result is
`PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED` and this successor identity stops:
there is no additional execution or retry under this identity after that
budget is exhausted.

Attempt counters are process-local only within this one authorized invocation,
but are exact because a second invocation is forbidden after execution start.
For each object, increment its attempt count exactly when its network-call
operation is invoked. No retry is permitted after complete HTTP-200 bytes have
been obtained.

For either ROOT or TERMINAL, the first complete HTTP-200 bytes returned are
the authoritative first-complete in-memory payload. Immediately persist those
same bytes into the exclusive no-overwrite raw lock. Local persistence is
separate from network transport and is frozen as:

```text
maximum_local_persistence_attempts_per_complete_payload=3
local_persistence_attempt_delays_seconds=[0,1,2]
local_persistence_bytes=the_same_first_complete_in_memory_bytes_only
network_refetch_after_persistence_failure=false
```

If all three local persistence attempts fail, record `GOVERNANCE_FAILURE` and
stop. A persistence failure never permits another request for that object.

If the process crashes or terminates after the execution-start receipt exists,
there is no second invocation and therefore no refetch or retry under this
identity. A later read-only adjudication treats the identity as terminal
`GOVERNANCE_FAILURE` unless already-durable `SUCCESS` evidence proves
otherwise. That adjudication is not a restart and must not claim an unknown
network-attempt total as zero.

## 7. Discovery-root acquisition and fresh-locator boundary

The first complete HTTP-200 discovery-root payload is authoritative. The
implementation immediately locks its exact bytes and provenance before any
semantic inspection. Once that complete root payload exists in this identity,
it must never fetch the root again.

The successor real acquisition must not use the existing payload-bound
`run_locator(output_root)` as its fresh-root selector. That runner remains
only the reviewed historical/offline-proof wrapper: it is intentionally bound
to the old exact payload/provenance, including `EXPECTED_PAYLOAD_SHA256`,
`EXPECTED_LENGTH`, and `EXPECTED_PRIOR_STRUCTURAL`.

The future implementation instead exposes one reusable fresh-root pure core
shared by both the historical wrapper and this successor acquisition. It must
not duplicate the selector implementation. The historical wrapper retains its
old binding before calling the shared core; the successor path binds only to
the new lock before calling the same core.

The successor fresh-root selector may run only after the new discovery root is
durably raw-locked and provenance-verified. It re-reads exactly those locked
fresh bytes and uses that lock's validated resolved-root URL. It dynamically
binds its input to that fresh lock's payload SHA-256 and byte length. It has no
old `EXPECTED_PAYLOAD_SHA256`, `EXPECTED_LENGTH`, or
`EXPECTED_PRIOR_STRUCTURAL` requirement.

The shared pure core reuses or refactors, without duplication or redesign, the
reviewed `_locate_private` and `_post_uniqueness_revalidate` mechanics;
existing candidate classes; token suppression and candidate-internal-token
exclusion; ASCII-year `P[0]`; exact `P[1]`; the uniqueness rule; and existing
JPX URL validation.

Its fresh-root result contract is exactly:

```text
one qualifying candidate plus post-uniqueness URL revalidation PASS
  -> SUCCESSOR_LOCATOR_MATCHED
zero or more than one qualifying candidate
  -> SOURCE_OR_DATA_FEASIBILITY_FAILURE
frozen parser structural failure
  -> HTML_STRUCTURE_UNSUPPORTED
selector/root-URL binding failure or post-uniqueness revalidation failure
  -> INPUT_BINDING_FAILURE
frozen safe-output validation failure
  -> SAFE_OUTPUT_VALIDATION_FAILURE
```

The raw href and resolved URL remain private runtime values only. Fresh-root
semantic evidence is a deterministic canonical safe object whose input payload
SHA-256 and byte length are those of the new lock, not the historical fixed
payload. It contains only safe hashes, counts, result, and structural hash;
it contains no raw URL, href, or path. The implementation task will define,
code, and test the exact closed schema from this design.

Acquisition maps the selector result without collapse:

```text
SUCCESSOR_LOCATOR_MATCHED
  -> continue to the TERMINAL request
SOURCE_OR_DATA_FEASIBILITY_FAILURE or HTML_STRUCTURE_UNSUPPORTED
  -> DATA_QUALITY_FAILURE / ROOT_LOCATOR
INPUT_BINDING_FAILURE
  -> INPUT_BINDING_FAILURE / ROOT_LOCATOR_INPUT_BINDING
SAFE_OUTPUT_VALIDATION_FAILURE
  -> IMPLEMENTATION_FAILURE / IMPLEMENTATION_ROOT_LOCATOR
```

Every non-success mapping stops with the locked ROOT preserved, terminal
attempts exactly zero, and no ROOT refetch.

## 8. TERMINAL object acquisition and persistence boundary

Only after unique locator success may the implementation use the selected URL
privately. It must first pass the existing `validate_jpx_url` contract, then
request exactly that selected object and no substitute.

The first complete HTTP-200 TERMINAL payload is authoritative. The
implementation immediately locks exact bytes and provenance before semantic
parsing. Once complete terminal bytes exist, it must never refetch the terminal
object in this identity.

The local persistence rule in section 6 governs all persistence attempts. A
crash or persistence failure never permits replacement network bytes.

## 9. Success condition

`SUCCESS` requires all of the following:

- exactly one authoritative locked discovery root;
- `SUCCESSOR_LOCATOR_MATCHED` on that root;
- exactly one authoritative locked TERMINAL object;
- no overwrite or replacement of either authoritative payload; and
- safe provenance verification PASS.

Success does not authorize terminal parsing, F2 bridge acquisition, Phase 2,
or any later stage.

## 10. Closed safe acquisition evidence and result matrix

The future implementation emits one closed deterministic safe result. Its
canonical JSON uses the established repository canonical JSON convention. It
may contain only the following categories: fixed schema/task/design and Git-SHA
bindings; result enum; SHA-256 values; byte lengths; HTTP status values;
attempt/request counts; closed booleans; the semantic-locator structural
evidence SHA-256; and safe lock-set count/SHA-256.

The closed schema fields are:

```text
schema_version
task
design_git_sha
implementation_git_sha
operation_class
result
failure_stage
discovery_root_http_status
terminal_http_status
discovery_root_payload_sha256
terminal_payload_sha256
discovery_root_byte_length
terminal_byte_length
discovery_root_attempt_count
terminal_attempt_count
network_request_count
discovery_root_locked
terminal_locked
semantic_locator_succeeded
semantic_locator_result
safe_provenance_verified
semantic_locator_structural_evidence_sha256
raw_lock_count
raw_lock_set_sha256
structural_evidence_sha256
```

The following fixed fields are exact literals or exact reviewed execution
bindings, never caller-selected values:

```text
schema_version=V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_V1
task=V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION
operation_class=RETRIABLE_PUBLIC_PLUMBING
design_git_sha=exact_GPT_PASS_design_SHA_supplied_to_implementation
implementation_git_sha=exact_GPT_reviewed_implementation_SHA_at_execution
```

All hashes are lowercase SHA-256 hex or `null` only where the result-specific
state permits `null`. Counts are exact nonnegative integers, never booleans.
Statuses are exact nonnegative integers or `null`. The final
`structural_evidence_sha256` is SHA-256 over the complete canonical safe object
excluding only itself.

The result enum is closed:

```text
SUCCESS
PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED
DATA_QUALITY_FAILURE
INPUT_BINDING_FAILURE
GOVERNANCE_FAILURE
IMPLEMENTATION_FAILURE
```

The exact `failure_stage` enum is closed:

```text
NONE
PRE_NETWORK_INPUT_BINDING
EXECUTION_BINDING_CONFLICT
ROOT_TRANSPORT
TERMINAL_TRANSPORT
ROOT_LOCATOR
ROOT_LOCATOR_INPUT_BINDING
ROOT_PERSISTENCE_EXHAUSTED
TERMINAL_PERSISTENCE_EXHAUSTED
IMPLEMENTATION_PRE_ROOT
IMPLEMENTATION_ROOT_TRANSPORT
IMPLEMENTATION_POST_ROOT_PRE_LOCATOR
IMPLEMENTATION_ROOT_LOCATOR
IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL
IMPLEMENTATION_TERMINAL_TRANSPORT
IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE
```

For every gracefully reported acquisition result, all of these invariants are
mandatory and reject bool-as-int values:

```text
0 <= discovery_root_attempt_count <= 3
0 <= terminal_attempt_count <= 3
network_request_count = discovery_root_attempt_count + terminal_attempt_count
terminal_attempt_count > 0 implies semantic_locator_succeeded=true
discovery_root_locked iff discovery_root_http_status=200 and discovery_root_payload_sha256/non-null byte length exist
terminal_locked iff terminal_http_status=200 and terminal_payload_sha256/non-null byte length exist
raw_lock_count = int(discovery_root_locked) + int(terminal_locked)
raw_lock_set_sha256 is null iff raw_lock_count=0; otherwise it is lowercase 64hex
semantic_locator_structural_evidence_sha256 is non-null iff locator was run
semantic_locator_result is null iff locator was not run; otherwise it is one of the frozen semantic-locator result enums
semantic_locator_succeeded=true iff semantic_locator_result=SUCCESSOR_LOCATOR_MATCHED
safe_provenance_verified=true only if every claimed lock verifies
```

The matrix below is exhaustive for this acquisition result. An HTTP status of
`null` means no completed HTTP response was safely recorded; a non-200 or
incomplete HTTP-200 response has null hash and byte-length fields and no lock.

| Result / `failure_stage` | ROOT state | Locator state | TERMINAL state | Provenance |
| --- | --- | --- | --- | --- |
| `SUCCESS` / `NONE` | locked; attempts 1..3 | run and succeeded | locked; attempts 1..3 | true; `raw_lock_count=2` |
| `PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED` / `ROOT_TRANSPORT` | unlocked; attempts exactly 3; payload fields null | not run; hash null | attempts 0; unlocked | false; count 0 |
| `PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED` / `TERMINAL_TRANSPORT` | locked; attempts 1..3 | run and succeeded | unlocked; attempts exactly 3; payload fields null | true for the one ROOT lock; count 1 |
| `DATA_QUALITY_FAILURE` / `ROOT_LOCATOR` | locked; attempts 1..3 | run and not succeeded | attempts 0; unlocked | true for the one ROOT lock; count 1 |
| `INPUT_BINDING_FAILURE` / `ROOT_LOCATOR_INPUT_BINDING` | locked; attempts 1..3 | run; result `INPUT_BINDING_FAILURE`; not succeeded | attempts 0; unlocked | true for the one ROOT lock; count 1 |
| `INPUT_BINDING_FAILURE` / `PRE_NETWORK_INPUT_BINDING` | attempts 0; unlocked; status/payload fields null | not run; hash null | attempts 0; unlocked | false; count 0 |
| `GOVERNANCE_FAILURE` / `EXECUTION_BINDING_CONFLICT` | attempts 0; unlocked; status/payload fields null | not run; hash null | attempts 0; unlocked | false; count 0 |
| `GOVERNANCE_FAILURE` / `ROOT_PERSISTENCE_EXHAUSTED` | attempts 1..3; status 200; unlocked; payload fields null | not run; hash null | attempts 0; unlocked | false; count 0 |
| `GOVERNANCE_FAILURE` / `TERMINAL_PERSISTENCE_EXHAUSTED` | locked; attempts 1..3 | run and succeeded | attempts 1..3; status 200; unlocked; payload fields null | true for the one ROOT lock; count 1 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_PRE_ROOT` | attempts 0; unlocked; status/payload fields null | not run; hash null | attempts 0; unlocked | false; count 0 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_ROOT_TRANSPORT` | attempts 1..2; unlocked; payload fields null | not run; hash null | attempts 0; unlocked | false; count 0 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_POST_ROOT_PRE_LOCATOR` | locked; attempts 1..3 | not run; hash null | attempts 0; unlocked | true for the one ROOT lock; count 1 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_ROOT_LOCATOR` | locked; attempts 1..3 | run; result `SAFE_OUTPUT_VALIDATION_FAILURE`; not succeeded | attempts 0; unlocked | true for the one ROOT lock; count 1 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL` | locked; attempts 1..3 | run and succeeded | attempts 0; unlocked | true for the one ROOT lock; count 1 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_TERMINAL_TRANSPORT` | locked; attempts 1..3 | run and succeeded | attempts 1..2; unlocked; payload fields null | true for the one ROOT lock; count 1 |
| `IMPLEMENTATION_FAILURE` / `IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE` | locked; attempts 1..3 | run and succeeded | locked; attempts 1..3 | false; count 2 |

`failure_stage=NONE` is valid only for `SUCCESS`; every other result must use
exactly the matching row above. This table forbids retaining arbitrary partial
evidence: every permitted lock, count, nullability state, and locator outcome
is mechanically determined by the row. A terminal read-only crash adjudication
is not a gracefully reported acquisition result and must not synthesize this
schema with unknown request counts; it records only the separately frozen
terminal governance disposition unless durable `SUCCESS` evidence already
exists.

Safe evidence must never emit raw href, raw URL, resolved URL, raw payload,
local or private path, operator identity, terminal month `T`, ticker identity,
or arbitrary exception text. None of these result classes is a source,
profitability, or strategy finding.

## 11. Required workflow

After GPT exact-SHA PASS of this design, the only next engineering work is
implementation plus synthetic tests. That implementation then requires GPT
exact-SHA review. Only then can fresh human approval authorize direct Windows
PowerShell real public acquisition under this identity. GPT reviews the safe
acquisition evidence after that execution; only then may it decide whether to
authorize terminal parsing or a subsequent stage.

No unlisted methodological choice is delegated to the execution AI. If one is
required and not already mechanically frozen in repository documentation,
execution must stop with `CHATGPT_DECISION_REQUIRED`.
