# V9_006 F1 Semantic Successor Public Acquisition Design

```text
task=V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION
status=DESIGN_AWAITING_GPT_EXACT_SHA_REVIEW
operation_class=RETRIABLE_PUBLIC_PLUMBING
network_acquisition_authorized=false
fresh_human_authorization_required=true
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

## 5. Bounded transport policy

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

## 6. Discovery-root acquisition and locator boundary

The first complete HTTP-200 discovery-root payload is authoritative. The
implementation immediately locks its exact bytes and provenance before any
semantic inspection. Once that complete root payload exists in this identity,
it must never fetch the root again.

The frozen semantic successor locator runs only against that locked root. If
its result is anything other than `SUCCESSOR_LOCATOR_MATCHED`, record
`DATA_QUALITY_FAILURE`, stop, and do not refetch the root.

## 7. TERMINAL object acquisition and persistence boundary

Only after unique locator success may the implementation use the selected URL
privately. It must first pass the existing `validate_jpx_url` contract, then
request exactly that selected object and no substitute.

The first complete HTTP-200 TERMINAL payload is authoritative. The
implementation immediately locks exact bytes and provenance before semantic
parsing. Once complete terminal bytes exist, it must never refetch the terminal
object in this identity.

If local persistence fails after complete bytes have been obtained, a local
persistence retry may use only those same in-memory bytes where mechanically
safe. It must not issue another network request to replace already-observed
complete bytes. If a crash or persistence failure prevents proof of the exact
content identity, record `GOVERNANCE_FAILURE` and stop.

## 8. Success condition

`SUCCESS` requires all of the following:

- exactly one authoritative locked discovery root;
- `SUCCESSOR_LOCATOR_MATCHED` on that root;
- exactly one authoritative locked TERMINAL object;
- no overwrite or replacement of either authoritative payload; and
- safe provenance verification PASS.

Success does not authorize terminal parsing, F2 bridge acquisition, Phase 2,
or any later stage.

## 9. Closed safe acquisition evidence

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
safe_provenance_verified
semantic_locator_structural_evidence_sha256
raw_lock_count
raw_lock_set_sha256
structural_evidence_sha256
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

The validator must make failure output mechanically closed. `SUCCESS` requires
two locked HTTP-200 objects, `raw_lock_count=2`, non-null byte/hash fields for
both objects, locator success, and provenance PASS. Root retry exhaustion has
no terminal attempt or lock. Terminal retry exhaustion retains the already
locked discovery-root evidence but has no terminal lock. `DATA_QUALITY_FAILURE`
retains the locked discovery-root evidence, has no terminal request or lock,
and records `semantic_locator_succeeded=false`. `INPUT_BINDING_FAILURE`,
`GOVERNANCE_FAILURE`, and `IMPLEMENTATION_FAILURE` must retain only evidence
mechanically established before their stop point; they never claim an
unproven lock, payload identity, locator success, or provenance PASS.

Safe evidence must never emit raw href, raw URL, resolved URL, raw payload,
local or private path, operator identity, terminal month `T`, ticker identity,
or arbitrary exception text. None of these result classes is a source,
profitability, or strategy finding.

## 10. Required workflow

After GPT exact-SHA PASS of this design, the only next engineering work is
implementation plus synthetic tests. That implementation then requires GPT
exact-SHA review. Only then can fresh human approval authorize direct Windows
PowerShell real public acquisition under this identity. GPT reviews the safe
acquisition evidence after that execution; only then may it decide whether to
authorize terminal parsing or a subsequent stage.

No unlisted methodological choice is delegated to the execution AI. If one is
required and not already mechanically frozen in repository documentation,
execution must stop with `CHATGPT_DECISION_REQUIRED`.
