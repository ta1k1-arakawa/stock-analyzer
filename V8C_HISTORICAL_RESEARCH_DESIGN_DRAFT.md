# V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT

```text
document_type=DESIGN_DRAFT_ONLY
study=V8C_HISTORICAL_RESEARCH
study_identity_is_new=true
predecessor_study=V8B_HISTORICAL_RESEARCH
predecessor_status=CLOSED_NO_RESULT_TRANSPORT_FAILURE
implementation_performed=false
production_code_changed=false
partition_creation_performed=false
data_acquisition_performed=false
research_opening_performed=false
real_network_requests_this_document=0
private_partition_accessed_this_document=false
private_ticker_identities_displayed_or_committed=false
tests_run_this_document=0
design_frozen=false
human_design_freeze_complete=false
implementation_authorized=false
Yahoo_network_authorized=false
T1C_allocation_authorized=false
T1C_acquisition_authorized=false
T2_acquisition_authorized=false
research_opening_authorized=false
```

This document is a design draft only. It authorizes no implementation,
allocation, acquisition, research opening, or real network request. It does
not modify research methodology, human-gate authority, or security rules.
It fixes the new V8C study identity and the transport-resilient execution
and gate protocol that a future, separately authorized implementation must
follow.

---

## 0. V8B closure and V8C study identity

`V8B_HISTORICAL_RESEARCH` is closed and must not be resumed:

```text
v8b_status=CLOSED_NO_RESULT_TRANSPORT_FAILURE
v8b_failure_class=TRANSPORT_HTTP_400_AFTER_NETWORK_BOUNDARY
v8b_t1b_raw_acquisition_result=BLOCKED
v8b_t1b_final_bundle_exists=false
v8b_t1b_research_opened=false
v8b_validation_result_exists=false
v8b_same_study_retry=PROHIBITED
v8b_t1b_reacquisition=PROHIBITED
v8b_successor_required_for_further_validation=true
```

The closure is an immutable provenance fact from
`V8B_T1B_ACQUISITION_FAILURE_ADJUDICATION.json`. The V8B HTTP 400 is a
transport failure, not a strategy result, validation result, profitability
result, or evidence that any methodology should be changed. V8C is a new
study with a new identity and its own design, review, freeze, and human
authorizations. It is not a retry, continuation, or amendment of V8B.

### 0.1 V8B T1B is burned for confirmatory use

V8B's `T1B` is burned for confirmatory use even though V8B produced no
research result:

```text
v8b_t1b_confirmatory_status=BURNED
v8b_t1b_reacquisition=PROHIBITED
v8b_t1b_reuse_as_v8c_validation=PROHIBITED
v8b_failed_ticker_investigation=PROHIBITED
v8b_request_position_reverse_analysis=PROHIBITED
v8b_http_400_threshold_tuning=PROHIBITED
v8b_failure_information_as_calibration_input=PROHIBITED
```

No V8B failed ticker may be investigated. No V8B request position may be
reverse-engineered. The V8B HTTP 400 may not be used to tune, relax, or
select a threshold. V8C must not use any V8B T1B identity, payload,
failure-specific fact, or request-order information.

---

## 1. Methodology inherited unchanged from V8B

V8C inherits the following scientific methodology from V8B without change.
None of these items may be re-derived, searched, tuned, or selected in the
V8C transport work:

```text
P_hist=UNCHANGED_FROM_V8B
walk_forward=UNCHANGED_FROM_V8B
causality=UNCHANGED_FROM_V8B
labels=UNCHANGED_FROM_V8B
promotion_metrics=UNCHANGED_FROM_V8B
promotion_thresholds=UNCHANGED_FROM_V8B
friction_grid=UNCHANGED_FROM_V8B
strategy_search_rules=UNCHANGED_FROM_V8B
```

The V8B values and definitions remain authoritative. In particular, V8C
does not change the historical span, walk-forward splits, label-confirmation
timing, required promotion metric set, `WALK_FORWARD_SURVIVOR` thresholds,
friction points, parameter-neighbourhood robustness, search registry, or
strategy stopping and promotion rules. Acquisition transport behavior is
specified separately below and must not be interpreted as a change to the
scientific methodology.

### 1.1 Data-quality policy

The V8B data-quality policy is carried forward unchanged:

```text
data_quality_policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE
invalid_fraction_threshold=1/252
max_consecutive_invalid_returned_rows=1
invalid_fraction_exact_comparison="invalid_returned_row_count * 252 <= total_returned_row_count"
floating_point_threshold_decision=PROHIBITED
threshold_source=V8B_frozen_policy_not_V8B_T1B_failure
```

The policy applies uniformly to the future V8C `T1C` raw acquisition and
to any future T2 raw acquisition authorized by this design. It does not
retroactively apply to V8B. A data-quality failure is terminal for the
current acquisition path and cannot trigger ticker replacement,
repartitioning, or threshold changes.

---

## 2. V8C validation block: `T1C`

V8C's validation block is fixed at design time from the original V8
`parent_T_spare`. The identities are private and are not displayed or
committed in this public design:

```text
v8c_validation_block=T1C
parent_block=original_parent_T_spare
v8b_t1b_definition=original_parent_T_spare[0:300]
t1c_definition=original_parent_T_spare[300:600]
t1c_slice_start_inclusive=300
t1c_slice_end_exclusive=600
t1c_size=300
t1c_selection=DETERMINISTIC
t1c_implementation_time_discretion=0
t1c_membership_fixed_before_acquisition=true
t1c_membership_conditional_on_data_quality=false
t1c_membership_conditional_on_strategy_result=false
t1c_data_quality_ticker_substitution=PROHIBITED
t1c_strategy_result_ticker_substitution=PROHIBITED
t1c_manual_ticker_substitution=PROHIBITED
t1c_repartition=PROHIBITED
```

The canonical rule is exactly:

```text
T1B = original parent_T_spare[0:300]
T1C = original parent_T_spare[300:600]
```

`T1C` is materialized only by a future, separately authorized allocation
implementation. That implementation may verify the fixed slice and its
provenance; it may not choose a different offset, inspect prices, use data
quality, use strategy results, replace a ticker, or redraw the block.

---

## 3. Yahoo transport readiness before T1C

Before any use of T1C, Yahoo transport readiness must be checked using
only the already-spent original V8 `T0` and non-evidential/public material:

```text
readiness_allowed_sources=[original V8 T0, synthetic data, Yahoo/provider public specifications]
readiness_forbidden_sources=[old T1, V8B T1B, V8C T1C, T2, T3, remaining fresh T_spare]
readiness_may_read_research_or_outcome_data=false
```

The readiness probe is not T1C acquisition and does not consume the T1C
allocation or acquisition authorization. It is a separate human gate and
must be executed only after the design and implementation prerequisites
for that gate have passed.

### 3.1 Fixed T0 readiness sentinel

The sentinel is fixed before execution and cannot be changed:

```text
source=original trusted V8 T0
indices=[0,149,299]
count=3
probe_start=2025-12-01
probe_end_exclusive=2025-12-08
sentinel_change=PROHIBITED
```

For all three fixed sentinel positions, the probe must require:

```text
http_status=200
trusted_yahoo_host=true
response_bytes_received=true
parser_success=true
expected_symbol_binding=true
nonempty_timestamp=true
valid_price_row_count>=1
```

The public result contains aggregate status and counts only. It must not
display or emit ticker names, prices, raw payloads, private paths, or any
other private partition identity. The probe must not print per-sentinel
details that permit identity reconstruction.

```text
all_three_sentinels_required=true
readiness_failure_consumes_t1c_gate=false
readiness_failure_action=BLOCK_T1C_GATE
same_fixed_sentinel_recheck_allowed=true
changed_sentinel_recheck=PROHIBITED
```

If readiness fails, T1C allocation/acquisition remains unauthorized and the
T1C gate is not consumed. Rechecking is allowed only with the exact same
sentinel and probe window.

---

## 4. Fixed T1C transport retry policy

The following transport retry policy is fixed before any T1C acquisition:

```text
maximum_attempts_per_ticker=3
maximum_retries=2
backoff_seconds=[5,30]
jitter=false
```

Retry is permitted only for these transport conditions:

```text
retryable=[
  NETWORK_TIMEOUT,
  CONNECTION_RESET,
  TEMPORARY_DNS_FAILURE,
  HTTP_408,
  HTTP_425,
  HTTP_429,
  HTTP_500,
  HTTP_502,
  HTTP_503,
  HTTP_504
]
```

These conditions are nonretryable:

```text
nonretryable=[
  HTTP_400,
  HTTP_401,
  HTTP_403,
  HTTP_404,
  HTTP_410,
  HTTP_422,
  UNTRUSTED_REDIRECT,
  RESPONSE_HOST_MISMATCH,
  PARSER_SCHEMA_FAILURE,
  SYMBOL_MISMATCH,
  DATA_QUALITY_GATE_FAILURE
]
```

An unknown error fails closed and is nonretryable:

```text
unknown_error=FAIL_CLOSED_NONRETRYABLE
```

Between retries, the following must remain identical:

```text
ticker_change_between_retries=PROHIBITED
request_period_change_between_retries=PROHIBITED
provider_change_between_retries=PROHIBITED
host_change_between_retries=PROHIBITED
request_parameter_change_between_retries=PROHIBITED
```

Retries are transport resilience only. They do not authorize another
ticker, alter the fixed T1C membership, bypass the quality gate, or create a
new research observation. A nonretryable failure, an exhausted retryable
failure, or a data-quality failure is terminal for the V8C T1C acquisition
attempt.

---

## 5. T1C one-shot acquisition gate

The T1C raw-acquisition gate is consumed at one precise boundary:

```text
t1c_one_shot_gate=ONE_TIME_HUMAN_AUTHORIZATION_TO_ACQUIRE_T1C
t1c_gate_consumed_immediately_before=first_real_T1C_Yahoo_opener_invocation
t1c_gate_reset_after_consumption=PROHIBITED
t1c_gate_reuse_after_consumption=PROHIBITED
```

The gate must not be consumed during any of the following:

```text
gate_consumption_forbidden_before_opener=[
  Git provenance check,
  design check,
  implementation review,
  allocation verification,
  trust pin verification,
  Asia/Tokyo check,
  output/staging check,
  T0 readiness probe,
  local request construction
]
```

The one-shot authorization is consumed immediately before the first real
T1C Yahoo opener invocation, regardless of the subsequent transport result.
The T0 readiness probe is a separate readiness action and never consumes
this T1C gate.

After T1C gate consumption, a terminal failure has the following result:

```text
same_study_retry=PROHIBITED
change_to_another_validation_block=PROHIBITED
research_opening=PROHIBITED
continue_with_fresh_validation=NEW_SUCCESSOR_STUDY_REQUIRED
```

No failure after the one-shot boundary may be reinterpreted as a strategy
failure, a profitability result, or evidence for changing the inherited
methodology.

---

## 6. Separate human gates

The following gates are separate and cannot authorize one another:

```text
T0 readiness=SEPARATE_HUMAN_GATE
T1C allocation=SEPARATE_HUMAN_GATE
T1C trust pin=SEPARATE_HUMAN_GATE
T1C raw acquisition=SEPARATE_HUMAN_GATE
T1C research opening=SEPARATE_HUMAN_GATE
T2 raw acquisition=SEPARATE_HUMAN_GATE
T2 research opening=SEPARATE_HUMAN_GATE
```

The authorization implications are explicitly one-way and limited:

```text
T0_readiness_PASS_allows_T1C_acquisition=false
T1C_authorization_allows_T2=false
T1C_raw_acquisition_PASS_allows_research_opening=false
T1C_research_opening_authorization_allows_T2=false
T2_raw_acquisition_PASS_allows_T2_research_opening=false
```

Readiness PASS only establishes that the fixed transport sentinel passed.
T1C allocation authorization only permits the fixed allocation operation.
The T1C trust-pin gate only permits pinning a verified allocation artifact.
T1C raw-acquisition authorization only permits the fixed T1C acquisition.
Raw acquisition PASS never opens research access. Every research-opening
gate requires its own human authorization and all required read-only
artifact checks.

---

## 7. T2 preservation and recheck policy

V8B's T2 was not acquired and was not opened, but V8C must not assume that
it is automatically reusable. T2 remains conditionally preservable only if
all of the following conditions pass a safe, read-only recheck:

```text
T2_real_data_acquired=false
T2_opened=false
T2_research_access_count=0
T2_features_observed=false
T2_outcomes_observed=false
T2_membership_reassigned=false
universe_definition_compatible=true
partition_algorithm_compatible=true
data_quality_policy_unchanged=true
```

The recheck may inspect only safe committed state, audit metadata, and
trusted provenance. It must not read, print, or expose private ticker
identities, private partition contents, raw payloads, features, or
outcomes.

### 7.1 Two mandatory recheck points

The same conditions must be rechecked at both points below:

```text
recheck_1=before_V8C_design_freeze
recheck_2=immediately_before_T2_acquisition
recheck_1_and_2_required=true
```

The first recheck is a prerequisite for `V8C_DESIGN_FINALIZED` and
`HUMAN_V8C_DESIGN_FREEZE`; it must not be replaced by an older favorable
status. The second recheck is the read-only gate immediately before any
T2 raw acquisition. Both checks are bound to the exact design and
provenance being used at that stage.

If any condition fails at either point:

```text
T2_preservation_result=BLOCK
automatic_T3_replacement=PROHIBITED
automatic_T_spare_replacement=PROHIBITED
automatic_new_holdout=PROHIBITED
T2_acquisition=PROHIBITED
research_opening=PROHIBITED
```

No automatic T3 replacement or automatic T_spare replacement is allowed.
Resolving a failed preservation check requires a separate successor study
and explicit design decision; this V8C design does not resolve it silently.

---

## 8. Privacy and scope boundary

This design keeps private partition information outside the public
repository and outside public outputs:

```text
private_ticker_identities_in_this_document=false
private_ticker_identities_in_commit=false
private_partition_read_during_this_task=false
private_partition_manifest_read_during_this_task=false
private_raw_payload_read_during_this_task=false
private_paths_in_public_output=false
```

Future allocation and acquisition implementations may verify private
artifacts only under their separately authorized trust and verification
gates. Public artifacts and logs may contain aggregate counts, hashes,
status values, and safe provenance metadata only. They must not contain
ticker names, prices, raw payloads, private paths, or request details that
could reconstruct a private identity.

This task makes no production-code change, performs no allocation, makes no
Yahoo request, performs no acquisition, and performs no research opening.

---

## 9. V8C stage sequence

The V8C sequence is fixed as follows. No stage may be skipped, merged with a
different human gate, or treated as authorization for a later stage:

```text
CREATE_V8C_DESIGN_DRAFT
INDEPENDENT_V8C_DESIGN_REVIEW
V8C_DESIGN_FINALIZED
HUMAN_V8C_DESIGN_FREEZE

V8C_TRANSPORT_AND_ACQUISITION_IMPLEMENTATION
INDEPENDENT_V8C_PRODUCTION_IMPLEMENTATION_REVIEW

T2_PRESERVATION_RECHECK

ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1C
EXECUTE_T1C_ALLOCATION
READ_ONLY_T1C_ALLOCATION_ARTIFACT_VERIFICATION

HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION
CREATE_V8C_TRUSTED_ALLOCATION_PIN
INDEPENDENT_TRUST_PIN_REVIEW

T0_TRANSPORT_READINESS_HUMAN_GATE
EXECUTE_FIXED_T0_TRANSPORT_READINESS_PROBE

T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_T1C_RAW_ACQUISITION
READ_ONLY_T1C_ACQUISITION_ARTIFACT_VERIFICATION

SEPARATE_T1C_RESEARCH_OPENING_GATE
LAYER_B

FROZEN_FINAL_CANDIDATE

READ_ONLY_T2_PRESERVATION_RECHECK
T2_RAW_ACQUISITION_HUMAN_GATE
T2_RAW_ACQUISITION
READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION

SEPARATE_T2_RESEARCH_OPENING_GATE
LAYER_C
```

`T2_PRESERVATION_RECHECK` is a required design-freeze preservation
condition and must be represented in the exact-stage audit trail before
allocation proceeds. `READ_ONLY_T2_PRESERVATION_RECHECK` is the second
mandatory check immediately before T2 acquisition. The two checks use the
same fixed conditions in section 7 and fail closed.

The T0 readiness gate and probe occur only after the T1C allocation and
trust-pin prerequisites in the sequence, but before any T1C opener. The
T1C raw-acquisition gate is consumed only at the boundary in section 5.
The T1C research-opening gate remains separate from raw acquisition. T2 is
not acquired or opened until after `FROZEN_FINAL_CANDIDATE`, its second
preservation recheck, and its own separate human gates.

---

## 10. Current design state

This file is an unfrozen design draft. Its current authorization state is:

```text
design_frozen=false
human_design_freeze_complete=false
implementation_authorized=false
Yahoo_network_authorized=false
T1C_allocation_authorized=false
T1C_acquisition_authorized=false
T2_acquisition_authorized=false
research_opening_authorized=false
```

The current state does not authorize any future agent to infer missing
methodology, consume a human gate, access private data, execute a Yahoo
request, run acquisition, open research, or substitute a block. Any
unspecified methodological decision remains `CHATGPT_DECISION_REQUIRED`
and must fail closed.

---

## 11. Design-only completion boundary

The only action represented by this document is creation of the initial
V8C design draft. The following remain explicitly unperformed:

```text
production_code_change=false
Yahoo_requests=0
private_partition_access=0
T1C_allocation=false
T1C_acquisition=false
T2_acquisition=false
research_opening=false
tests_run=0
```

The next required stage is `INDEPENDENT_V8C_DESIGN_REVIEW`.
