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

### 1.2 Exact classifier blob binding

`POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE` depends on a
specific parser/classifier implementation. V8C inherits that
implementation unchanged, bound by exact Git blob identity, not by file
path, version label, or semantic description:

```text
canonical_parser_classifier_file=src/v7_yahoo_collector.py
canonical_parser_classifier_git_commit=28e281c3ee30d6b4c2f981c5da3ddc983c09724d
canonical_parser_classifier_blob_sha=76b57b077f3214e666ff9dc06d9c224afc16df9f
classifier_version_binding=EXACT_GIT_BLOB
policy_valid_only_with_exact_classifier_blob=true
semantically_similar_different_blob_silently_accepted=PROHIBITED
```

A future production implementation must verify the exact
`canonical_parser_classifier_blob_sha` before any real Yahoo request, at
every one of the following stages, and BLOCK on mismatch before that
stage's acquisition gate is consumed and before the corresponding Yahoo
request is made:

```text
classifier_blob_verified_before=[
  T1C_transport_readiness,
  T1C_raw_acquisition,
  T2_transport_readiness,
  T2_raw_acquisition
]
classifier_blob_mismatch_error=V8C_PRODUCTION_CLASSIFIER_VERSION_MISMATCH
verification_occurs_before_acquisition_gate_consumption=true
verification_occurs_before_the_corresponding_yahoo_request=true
```

Any semantic change to the classifier, or to its thresholds, is a
methodological change, not a transport-resilience concern, and must stop
with `CHATGPT_DECISION_REQUIRED` before any production use. This document
does not change the classifier or its thresholds.

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

### 2.1 Pre-freeze T1C/T_spare freshness preservation recheck

Proving that the original immutable V8 parent provenance is unchanged is
not, by itself, proof that the `T1C` candidate slice itself is still
fresh — untouched, unopened, and not silently consumed by any other
allocation since V8B closed. A mandatory pre-freeze preservation gate for
the `T1C` candidate slice is therefore required, in addition to and
independent of the parent-provenance checks elsewhere in this document:

```text
gate=T1C_T_SPARE_FRESHNESS_PRESERVATION_RECHECK
occurs_before=[V8C_DESIGN_FINALIZED, HUMAN_V8C_DESIGN_FREEZE]
bound_to_exact_candidate_design_sha=true
bound_to_same_sha_as=[INDEPENDENT_V8C_DESIGN_REVIEW, T2_PRESERVATION_RECHECK]
evidence_sources=safe_committed_or_audit_or_provenance_metadata_only
reads_or_prints_actual_ticker_identities=PROHIBITED
```

At minimum, the recheck must find safe, positive evidence for every one of
the following -- absence of evidence is never treated as PASS, and any
missing required safe metadata is itself a BLOCK:

```text
original_parent_t_spare_provenance_unchanged=REQUIRED
original_parent_t_spare_count_and_hash_still_trusted=REQUIRED
v8b_t1b_confirmed_as_using_only_the_fixed_0_300_slice=REQUIRED
no_further_t_spare_allocation_has_consumed_any_member_of_300_600=REQUIRED
t1c_candidate_slice_300_600_reassigned=false
t1c_candidate_slice_ohlcv_acquired=false
t1c_candidate_slice_feature_or_outcome_research_access=false
t1c_candidate_slice_research_opened=false
t1c_candidate_slice_ticker_identities_exposed_to_human_or_public_research_loop=false
ordering_and_provenance_rule_required_to_interpret_300_600_unchanged=REQUIRED
```

```text
absence_of_evidence_counts_as_pass=false
missing_required_safe_metadata=BLOCK
```

If any condition fails, or if required safe metadata is missing:

```text
T1C_T_SPARE_FRESHNESS_PRESERVATION_RESULT=BLOCK
V8C_DESIGN_FINALIZED_without_pass=PROHIBITED
HUMAN_V8C_DESIGN_FREEZE_without_pass=PROHIBITED
automatic_alternate_t_spare_slice=PROHIBITED
automatic_redraw=PROHIBITED
automatic_t3_substitution=PROHIBITED
implementation_time_offset_choice=PROHIBITED
status=CHATGPT_DECISION_REQUIRED
```

A failed check must never be resolved by inspecting actual private
ticker identities to diagnose or work around it; diagnosis is limited to
the same safe committed/audit/provenance metadata the check itself may
use. Resolving a failed check requires a separate, explicit design
decision — this document does not resolve it silently, and no execution
agent may choose a substitute slice, offset, or block on its own
discretion.

The exact-SHA binding, and the requirement that this recheck, the
independent design review, and the T2 preservation recheck (§7) all PASS
for the same exact design commit SHA before `V8C_DESIGN_FINALIZED` may
proceed, are specified in §8.

---

## 3. Yahoo transport readiness before T1C raw acquisition

Before any real T1C Yahoo raw acquisition, Yahoo transport readiness must
be checked using only the already-spent original V8 `T0` and
non-evidential/public material. This whole section describes exactly one
canonical human gate; it is never split into, or duplicated as, a
separate generic "T0 readiness" gate:

```text
canonical_gate_name=T1C_TRANSPORT_READINESS_HUMAN_GATE
gate_purpose=readiness_for_future_T1C_raw_acquisition
sentinel_source=original_trusted_V8_T0
this_gate_is_not_a_generic_T0_research_gate=true
readiness_allowed_sources=[original V8 T0, synthetic data, Yahoo/provider public specifications]
readiness_forbidden_sources=[old T1, V8B T1B, V8C T1C, T2, T3, remaining fresh T_spare]
readiness_may_read_research_or_outcome_data=false
```

The readiness probe is not T1C acquisition and does not consume the T1C
allocation or acquisition authorization. Each readiness execution is a
separate real Yahoo network access and therefore has its own explicit human
authorization:

```text
one_readiness_authorization=exactly_one_readiness_probe_execution
readiness_authorization_reuse=PROHIBITED
each_readiness_recheck_requires=NEW_EXPLICIT_T1C_TRANSPORT_READINESS_HUMAN_GATE
```

The readiness gate must be executed only after the design and implementation
prerequisites for that gate have passed. A blocked readiness result may be
rechecked only with the same fixed sentinel, window, and parameters, and
only after a new explicit readiness authorization. A prior readiness
authorization never authorizes a later probe.

### 3.1 Fixed T0-sourced sentinel for T1C transport readiness

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
readiness_failure_consumes_t2_gate=false
readiness_failure_action=BLOCK_T1C_RAW_ACQUISITION
readiness_failure_leaves_fresh_validation_blocks_untouched=true
same_fixed_sentinel_recheck_allowed=true
changed_sentinel_recheck=PROHIBITED
changed_window_recheck=PROHIBITED
changed_parameter_recheck=PROHIBITED
new_readiness_authorization_required_for_recheck=true
```

If readiness fails, the T1C acquisition gate is not consumed and no fresh
validation block is touched. Because the stage sequence creates and verifies
the T1C allocation and trust pin before this readiness probe, an already-
created verified allocation remains valid but inert and an already-created
trust pin remains valid but inert. Allocation is not repeated or redrawn;
reallocation is prohibited. T1C raw acquisition remains BLOCKed until a
new explicit readiness authorization permits a recheck. The fixed sentinel,
window, and parameters cannot be changed.

### 3.2 Minimal private membership access for the T0 sentinel

The future T1C transport readiness implementation may resolve only the exact original
trusted V8 T0 members at sentinel indices `[0,149,299]`, using the minimum
read-only private membership access required to bind those positions to the
sentinel. This is the sole private membership access allowed for readiness:

```text
allowed_private_access=minimum_read_only_original_trusted_v8_t0_sentinel_membership_only
allowed_indices=[0,149,299]
other_t0_member_access=PROHIBITED
t1c_identity_access=PROHIBITED
t2_identity_access=PROHIBITED
t3_identity_access=PROHIBITED
t_spare_identity_access=PROHIBITED
private_identity_public_output=PROHIBITED
raw_payload_public_output=PROHIBITED
readiness_as_strategy_or_research_outcome=PROHIBITED
readiness_is_research_opening=false
```

This task does not perform that future access. The readiness result remains
aggregate-only and does not expose ticker identities.

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

### 4.1 T2 transport readiness and the same retry policy

The same fixed transport resilience policy applies to future T2 raw
acquisition. Before the first real T2 Yahoo opener invocation, the fixed
original V8 T0 sentinel readiness probe must be executed under a separate
T2 readiness gate:

```text
t2_readiness_source=original trusted V8 T0
t2_readiness_indices=[0,149,299]
t2_readiness_probe_start=2025-12-01
t2_readiness_probe_end_exclusive=2025-12-08
t2_readiness_sentinel_change=PROHIBITED
t2_readiness_window_change=PROHIBITED
t2_readiness_parameters_change=PROHIBITED
one_t2_readiness_authorization=exactly_one_t2_readiness_probe_execution
t2_readiness_authorization_reuse=PROHIBITED
t2_readiness_recheck_requires=NEW_EXPLICIT_T2_TRANSPORT_READINESS_HUMAN_GATE
```

T2 readiness uses the same aggregate-only output and the same required
sentinel checks in section 3.1. A T2 readiness PASS does not authorize T2
acquisition. A readiness BLOCK does not consume the T2 acquisition gate,
does not touch a fresh block, and may be rechecked only with a new explicit
T2 readiness authorization and the unchanged sentinel/window/parameters.

T2 uses exactly the same retry classes and bounds as T1C:

```text
t2_maximum_attempts_per_ticker=3
t2_maximum_retries=2
t2_backoff_seconds=[5,30]
t2_jitter=false
t2_retryable=[
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
t2_nonretryable=[
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
t2_unknown_error=FAIL_CLOSED_NONRETRYABLE
```

For T2, the one-shot acquisition gate is consumed immediately before the
first real T2 Yahoo opener invocation. It is not consumed by provenance,
design, implementation, allocation, trust-pin, readiness, or local request
construction work. Between T2 retries, ticker, request period, provider,
host, and all request parameters remain unchanged. A nonretryable error,
unknown error, exhausted retryable error, or data-quality failure is
terminal and cannot produce a successful bundle.

### 4.2 T2 one-shot acquisition gate and terminal-failure semantics

T2's one-shot acquisition gate and post-gate terminal-failure consequences
are made fully explicit here, symmetric with T1C's (§5):

```text
t2_one_shot_gate=ONE_TIME_HUMAN_AUTHORIZATION_TO_ACQUIRE_T2
t2_gate_consumed_immediately_before=first_real_T2_Yahoo_opener_invocation
t2_gate_reset_after_consumption=PROHIBITED
t2_gate_reuse_after_consumption=PROHIBITED
```

The allowed per-ticker transport retries *within* that single already-
consumed one-shot authorization remain exactly the frozen bounds already
stated above (`t2_maximum_attempts_per_ticker=3`,
`t2_maximum_retries=2`); nothing in this subsection prohibits those
within-attempt transport retries. What this subsection prohibits is a
**new** acquisition authorization or a **new** acquisition attempt after
the one-shot gate has already produced a terminal outcome.

After T2 gate consumption, if acquisition reaches any terminal failure
(a nonretryable error, an unknown error, an exhausted retryable error, or
a data-quality failure):

```text
same_study_t2_reacquisition=PROHIBITED
new_t2_acquisition_authorization_inside_v8c=PROHIBITED
same_study_retry_after_terminal_failure=PROHIBITED
alternate_t2_source=PROHIBITED
t2_membership_replacement=PROHIBITED
automatic_t3_replacement=PROHIBITED
automatic_t_spare_replacement=PROHIBITED
threshold_change=PROHIBITED
research_opening=PROHIBITED
layer_c_result_claimed=PROHIBITED
v8c_confirmatory_layer_c_path=CLOSES_WITHOUT_RESULT
```

A terminal T2 acquisition failure must not be interpreted as a strategy
failure, a model failure, profitability evidence, or a Layer C result. If
another sealed holdout acquisition is scientifically desired after such a
failure, it requires a `NEW_SUCCESSOR_STUDY_IDENTITY` — exactly as V8C
itself was required after V8B's T1B transport failure (§0) — not a retry,
continuation, or amendment of V8C.

### 4.3 Frozen concrete Python transport exception classification

The abstract retry classes above (`NETWORK_TIMEOUT`, `CONNECTION_RESET`,
`TEMPORARY_DNS_FAILURE`, and the frozen `HTTP_*` codes) must not be left
to implementation-agent discretion to map from concrete runtime
exceptions. This subsection freezes that mapping for the production
transport wrapper around the exact existing collector
(`src/v7_yahoo_collector.py`'s `fetch_chart_once`/`opener` boundary). It
applies identically, with no per-stage variation, to all four Yahoo-
request-bearing stages:

```text
classification_contract_applies_to=[
  T1C_transport_readiness,
  T1C_raw_acquisition,
  T2_transport_readiness,
  T2_raw_acquisition
]
```

**HTTP.** `urllib.error.HTTPError` is classified from its exact numeric
`.code` attribute only:

```text
http_classification_source=urllib.error.HTTPError.code (exact numeric value)
http_retryable_codes=[HTTP_408, HTTP_425, HTTP_429, HTTP_500, HTTP_502, HTTP_503, HTTP_504]
http_all_other_codes=NONRETRYABLE_FAIL_CLOSED_UNLESS_EXPLICITLY_LISTED
http_status_classification_by_message_or_substring=PROHIBITED
```

**`NETWORK_TIMEOUT`.** Retryable only for these concrete conditions:

```text
network_timeout_concrete_exceptions=[
  TimeoutError,
  socket.timeout,
  urllib.error.URLError (only when .reason resolves to a timeout condition)
]
other_urllib_URLError_reason_values=NOT_NETWORK_TIMEOUT
```

**`CONNECTION_RESET`.** Retryable only for these concrete conditions:

```text
connection_reset_concrete_exceptions=[
  ConnectionResetError,
  OSError or urllib.error.URLError.reason wrapping an OSError whose concrete errno equals the platform connection-reset errno (errno.ECONNRESET)
]
other_errno_values=NOT_CONNECTION_RESET
```

**`TEMPORARY_DNS_FAILURE`.** Retryable only for one concrete condition:

```text
temporary_dns_failure_concrete_exception=socket.gaierror whose concrete errno equals socket.EAI_AGAIN
other_socket_gaierror_values=NOT_TEMPORARY_DNS_FAILURE (includes permanent/non-temporary DNS failures such as EAI_NONAME/EAI_FAIL)
other_socket_gaierror_values_classification=NONRETRYABLE_FAIL_CLOSED
```

**Unknown or unmapped exception.** Never guessed:

```text
unmapped_or_unrecognized_exception=UNKNOWN
unknown_classification=FAIL_CLOSED_NONRETRYABLE
os_or_platform_exception_not_mechanically_mappable_by_these_rules=UNKNOWN_FAIL_CLOSED_NONRETRYABLE
```

**Implementation constraints, binding on the future production transport
wrapper:**

```text
original_concrete_exception_classification_retained_in_private_audit_metadata=REQUIRED
private_audit_metadata_leaks_ticker_identity=PROHIBITED
public_output_exposes_only_privacy_safe_aggregate_failure_classes=REQUIRED
substring_or_message_based_heuristic_classification=PROHIBITED
retryability_broadened_at_implementation_time=PROHIBITED
retryable_http_list_change=PROHIBITED
nonretryable_http_list_change=PROHIBITED
retry_count_change=PROHIBITED
backoff_change=PROHIBITED
jitter_change=PROHIBITED
```

This subsection fixes only how a concrete Python-level exception maps to
one of the already-frozen abstract retry classes; it does not add, remove,
or reclassify any `HTTP_*` code, and it does not change
`maximum_attempts_per_ticker`, `maximum_retries`, `backoff_seconds`, or
`jitter` anywhere in this document.

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
  T1C transport readiness probe,
  local request construction
]
```

The one-shot authorization is consumed immediately before the first real
T1C Yahoo opener invocation, regardless of the subsequent transport result.
The T1C transport readiness probe is a separate readiness action and never
consumes this T1C gate.

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
T1C allocation=SEPARATE_HUMAN_GATE
T1C trust pin=SEPARATE_HUMAN_GATE
T1C transport readiness=SEPARATE_HUMAN_GATE
T1C raw acquisition=SEPARATE_HUMAN_GATE
T1C research opening=SEPARATE_HUMAN_GATE
T2 authority bridge=SEPARATE_HUMAN_GATE
T2 transport readiness=SEPARATE_HUMAN_GATE
T2 raw acquisition=SEPARATE_HUMAN_GATE
T2 research opening=SEPARATE_HUMAN_GATE
```

`T1C transport readiness` is the single canonical gate for T1C readiness
(`T1C_TRANSPORT_READINESS_HUMAN_GATE`, §3); it is never split into, or
duplicated as, a separate "T0 readiness" gate. `T2 authority bridge`
(`HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE`, §7.2) is a V8C-specific gate,
distinct from and not satisfied by any V8B authorization.

The authorization implications are explicitly one-way and limited:

```text
T1C_transport_readiness_PASS_allows_T1C_acquisition=false
T1C_authorization_allows_T2=false
T1C_raw_acquisition_PASS_allows_research_opening=false
T1C_research_opening_authorization_allows_T2=false
V8C_T2_authority_bridge_review_PASS_allows_T2_acquisition=false
T2_readiness_PASS_allows_T2_acquisition=false
T2_raw_acquisition_PASS_allows_T2_research_opening=false
```

Each readiness PASS only establishes that its fixed transport sentinel
passed. It never authorizes the corresponding raw acquisition. Readiness
authorization is consumed exactly once per probe execution; an earlier
authorization is never reused for a recheck.
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
T2_PRESERVATION_RECHECK_PASS_required_for=V8C_DESIGN_FINALIZED
V8C_DESIGN_FINALIZED_without_pass=PROHIBITED
HUMAN_V8C_DESIGN_FREEZE_without_pass=PROHIBITED
automatic_T3_replacement=PROHIBITED
automatic_T_spare_replacement=PROHIBITED
automatic_new_holdout=PROHIBITED
T2_acquisition=PROHIBITED
research_opening=PROHIBITED
```

No automatic T3 replacement or automatic T_spare replacement is allowed.
Resolving a failed preservation check requires a separate successor study
and explicit design decision; this V8C design does not resolve it silently.

### 7.2 V8C-specific T2 authority bridge (required, not inherited from V8B)

`V8B_T2_AUTHORITY_BRIDGE.json` is scoped to `V8B_HISTORICAL_RESEARCH` and
the frozen V8B design commit. It must never be silently reused as V8C
authority:

```text
existing_V8B_T2_authority_bridge_authorizes_V8C=false
```

A separate future V8C `T2` authority bridge is required. It is the same
`OPTION_2`-style bridge V8B used — `T2` continues to use the original,
immutable V8 partition/trust authority, untouched, unmodified, unre-pinned
— re-bound to V8C's own study identity and frozen design commit.

**Conceptual artifact: `V8C_T2_AUTHORITY_BRIDGE.json`.** At minimum it
binds:

```text
schema_version
study=V8C_HISTORICAL_RESEARCH
role=SEALED_HOLDOUT
exact_frozen_v8c_design_commit
source_authority=ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY
v8_trust_anchor_git_identity=61faade0625139cec3fb61216ab2f97f572a7028
authorized_parent_v8_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
expected_t2_ticker_count=300
expected_t2_ticker_list_sha256=e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500
t2_membership_reassignment=PROHIBITED
v8_trusted_partition_json_mutated_or_repinned=false
t2_acquired_before_authorized_v8c_acquisition=false
t2_research_open_count_before_official_opening=0
reviewed_production_implementation_commit
exact_human_bridge_authorization_identity
authorization_note
```

The bridge must never modify, reinterpret, or re-pin
`V8_TRUSTED_PARTITION.json` or the original V8 partition manifest; it only
records V8C's own claim on the existing, immutable `T2` block and
re-verifies `T2`'s existing V8 provenance, exactly as V8B's own bridge did
for V8B.

**Separate authority sequence, before T2 raw acquisition:**

```text
HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE
CREATE_V8C_T2_AUTHORITY_BRIDGE
INDEPENDENT_V8C_T2_AUTHORITY_BRIDGE_REVIEW
```

The independent review must bind to all of:

```text
exact_bridge_git_commit
exact_bridge_git_blob_sha
exact_frozen_v8c_design_commit
exact_original_v8_authority_objects=[v8_trust_anchor_git_identity, authorized_parent_v8_partition_manifest_sha256]
exact_human_bridge_authorization_identity
```

T2 production acquisition must fail closed if the V8C-specific bridge is
missing, mismatched, stale (bound to a different frozen design commit or
a different original V8 authority object than the one currently verified),
or not independently reviewed. The V8B bridge itself is never treated as
V8C authorization, and this sequence does not replace, weaken, or merge
with `T2_PRESERVATION_RECHECK`/`READ_ONLY_T2_PRESERVATION_RECHECK` (§7,
§7.1) — both remain separately required.

---

## 8. Exact-SHA design-freeze protocol

V8C uses an exact Git binding for design review and freeze. A branch name,
working-tree state, `current branch`, or `latest HEAD` is never a design
freeze binding:

```text
design_freeze_binding=EXACT_ONE_40_HEX_GIT_COMMIT_SHA
moving_branch_binding=INVALID
latest_head_binding=INVALID
```

The protocol is:

```text
A. INDEPENDENT_V8C_DESIGN_REVIEW reviews exactly one 40-hex design commit SHA.
B. T1C_T_SPARE_FRESHNESS_PRESERVATION_RECHECK (§2.1) reviews the same exact design commit SHA as A.
C. T2_PRESERVATION_RECHECK (§7) reviews the same exact design commit SHA as A.
D. V8C_DESIGN_FINALIZED is allowed only if A, B, and C all PASS for that same SHA.
E. HUMAN_V8C_DESIGN_FREEZE explicitly names and approves that exact 40-hex SHA.
F. After approval, create a separate V8C_DESIGN_FREEZE_APPROVAL.json artifact in a later commit.
```

`V8C_DESIGN_FINALIZED` and `HUMAN_V8C_DESIGN_FREEZE` are prohibited unless
the independent design review, the pre-freeze T1C/T_spare freshness
preservation recheck, and the pre-freeze T2 preservation recheck all PASS
for the same exact SHA. No gate may silently rewrite the reviewed design
body.

The conceptual minimum fields for the separate freeze-approval artifact
are:

```text
schema_version
study
frozen_design_git_commit
design_document
final_independent_review_result
final_independent_review_design_commit
t1c_t_spare_freshness_preservation_recheck_result
t1c_t_spare_freshness_preservation_recheck_design_commit
t2_preservation_recheck_result
t2_preservation_recheck_design_commit
approval_status
human_gate
```

The freeze-approval artifact commit is not the frozen design commit. The
frozen design commit is the earlier exact SHA named by the human approval;
the later commit that records `V8C_DESIGN_FREEZE_APPROVAL.json` cannot
retroactively become that design commit.

Any semantic change to the design body after the independent design
review, the T1C/T_spare freshness preservation recheck, or the T2
preservation recheck has completed:

```text
prior_design_review=INVALID
prior_t1c_t_spare_freshness_preservation_recheck=INVALID
prior_t2_preservation_recheck=INVALID
new_candidate_sha=REQUIRED
repeat_independent_design_review=REQUIRED
repeat_t1c_t_spare_freshness_preservation_recheck=REQUIRED
repeat_t2_preservation_recheck=REQUIRED
new_human_design_freeze_approval=REQUIRED
```

Any semantic design change after `HUMAN_V8C_DESIGN_FREEZE` is:

```text
NEW_STUDY_REQUIRED
```

Future V8C production authority must bind to the exact
`frozen_design_git_commit` recorded in the freeze-approval artifact, never
to a branch or an unpinned working-tree file.

---

## 9. T1C successor trust and authority chain

The original V8 trust anchor names the original V8 blocks and does not name
the new logical block `T1C`. Therefore the original V8 trust anchor alone
is insufficient authority for T1C:

```text
original_v8_trust_anchor_names_t1c=false
original_v8_trust_anchor_alone_authorizes_t1c=false
separate_v8c_t1c_authority_chain=REQUIRED
```

### 9.1 Immutable parent provenance

The original V8 trusted partition, original V8 partition manifest, and
original `T_spare` provenance remain immutable and unchanged. They provide
parent provenance only; they do not silently authorize T1C. The T1C slice
must still bind to the fixed rule in section 2 and to a separate V8C
authority chain.

### 9.2 Private V8C T1C allocation artifact

A future allocation implementation must create a private V8C T1C
allocation artifact. It may contain private ticker assignments internally,
but those identities must never be printed, logged, or committed. At
minimum, the artifact binds:

```text
schema_version
study=V8C_HISTORICAL_RESEARCH
artifact_role=VALIDATION_BLOCK_ALLOCATION
logical_block=T1C
exact_frozen_v8c_design_commit
parent_v8_partition_manifest_identity
parent_v8_partition_manifest_sha256
parent_v8_partition_implementation_commit
parent_t_spare_ticker_count
parent_t_spare_ticker_list_sha256
selection_rule_id
selection_rule_canonical_text_or_hash
slice_start_inclusive=300
slice_end_exclusive=600
t1c_ticker_count=300
t1c_ticker_list_sha256
remaining_t_spare_ticker_count
remaining_t_spare_ticker_list_sha256
allocation_implementation_commit
artifact_self_hash
```

The artifact must prove the exact deterministic slice
`original_parent_T_spare[300:600]` and may not select membership from data
quality, strategy results, or implementation-time discretion. Ticker
identities are private and are not part of any public output or repository
commit.

### 9.3 Public V8C trusted allocation pin

After independent verification of the private allocation artifact, a
separate public V8C trusted allocation pin may be created only through its
own human gate. It contains safe metadata only and must bind at minimum:

```text
schema_version
study=V8C_HISTORICAL_RESEARCH
artifact_role=TRUSTED_T1C_ALLOCATION_PIN
exact_frozen_v8c_design_commit
authorized_allocation_artifact_self_hash
parent_v8_authority_identity
t1c_ticker_count=300
t1c_ticker_list_sha256
reviewed_allocation_implementation_commit
reviewed_production_implementation_commit
exact_human_pin_authorization_identity
authorization_status
```

The public pin must never contain T1C or remaining-`T_spare` ticker
identities.

### 9.4 Independent trust-pin review and chronology

The independent trust-pin review must bind to all of:

```text
exact_pin_git_commit
exact_pin_git_blob_sha
exact_allocation_artifact_self_hash
exact_human_pin_authorization_identity
exact_frozen_v8c_design_commit
```

The chronology is strict and cannot be reordered or collapsed:

```text
allocation_artifact
-> verified_allocation
-> human_pin_authorization
-> pin_commit
-> independent_pin_review
```

Production acquisition must resolve the public trust pin from the verified
Git object bound by this chain. It must not accept a caller-supplied
arbitrary trust mapping, arbitrary manifest mapping, caller-supplied path,
or working-tree-only authority as a substitute for the verified pin.

---

## 10. Acquisition artifact verification and research-opening security

For every successful T1C or T2 acquisition, the read-only artifact
verification must require all of the following:

```text
exact_study_identity=REQUIRED
correct_logical_block_and_role=REQUIRED
exact_frozen_v8c_design_commit=REQUIRED
exact_reviewed_production_implementation_commit=REQUIRED
correct_authority_chain=REQUIRED
t2_authority_chain_requires_v8c_specific_bridge=true
ticker_count=300
exact_ticker_list_hash=REQUIRED
request_start=2016-04-01
request_end_exclusive=2026-01-01
exact_data_quality_policy_metadata=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE
exact_classifier_blob_sha=76b57b077f3214e666ff9dc06d9c224afc16df9f
classifier_blob_mismatch=BLOCK
final_payload_record_count=300
expected_raw_payload_file_count=300
file_sha256_binding=REQUIRED
file_byte_count_binding=REQUIRED
missing_payload_file=BLOCK
extra_payload_file=BLOCK
symlink_payload_file=BLOCK
nonregular_payload_file=BLOCK
zero_research_opening_before_opening=REQUIRED
zero_feature_access_before_opening=REQUIRED
zero_outcome_access_before_opening=REQUIRED
public_output_privacy_safe_only=REQUIRED
```

The verification is data-integrity-only. It must not calculate features,
strategy results, profit, trades, or any other research outcome. Public
verification output is aggregate-only; private verification may inspect
the required internal records without emitting ticker identities.

### 10.1 Retry audit requirements

The acquisition artifact must also retain and independently verify the
transport retry audit:

```text
each_logical_member_attempts_inclusive_range=[1,3]
total_retry_count_inclusive_range=[0,600]
total_request_attempts=300 + total_retry_count
intermediate_retry_failure_class=RETRYABLE_CLASS_ONLY
successful_bundle_after_nonretryable_failure=PROHIBITED
retry_interval_request_fingerprint=IDENTICAL
ticker_change_between_retries=PROHIBITED
window_change_between_retries=PROHIBITED
provider_change_between_retries=PROHIBITED
host_change_between_retries=PROHIBITED
request_parameter_change_between_retries=PROHIBITED
retry_metadata_public_output=AGGREGATE_ONLY
private_retry_verification_may_check_internal_records=true
private_retry_verification_may_output_ticker_identity=false
```

The retry audit applies identically to T1C and T2. A successful bundle may
not hide a nonretryable intermediate failure. An exhausted retryable
failure, unknown failure, or data-quality failure is terminal and cannot be
converted into a successful bundle by changing the request.

### 10.2 Research-opening hardening inherited from V8B

The following V8B security hardening is mandatory for both T1C and T2
research opening:

```text
official_opening_path_accepts_caller_crafted_arbitrary_mapping=PROHIBITED
official_opening_path_accepts_caller_supplied_authority_path=PROHIBITED
verified_acquisition_manifest_resolved_by_official_resolver=REQUIRED
trusted_block_and_authority_binding_reverified_at_point_of_use=REQUIRED
raw_payload_byte_count_binding_reverified_immediately_before_opening=REQUIRED
raw_payload_sha256_binding_reverified_immediately_before_opening=REQUIRED
earlier_read_only_artifact_verification_pass_is_sufficient_alone=FALSE
post_verification_tampering_detected=BLOCK
applies_to=[T1C, T2]
```

The official opening path must obtain a verified acquisition manifest from
the official resolver, re-check its trusted block/authority binding at the
point of use, and re-check every raw payload's byte count and SHA-256
binding immediately before opening. Any mismatch or post-verification
tampering blocks research opening. These requirements do not authorize
opening and do not consume a research-opening gate.

---

## 11. Privacy and scope boundary

This design keeps private partition information outside the public
repository and outside public outputs:

```text
private_ticker_identities_in_this_document=false
private_ticker_identities_in_commit=false
private_partition_read_during_this_task=false
private_partition_manifest_read_during_this_task=false
private_raw_payload_read_during_this_task=false
private_paths_in_public_output=false
future_t0_readiness_private_access=minimum_read_only_original_v8_t0_sentinel_membership_only
future_t0_readiness_other_private_membership_access=PROHIBITED
readiness_gate_is_research_opening=false
```

Future allocation and acquisition implementations may verify private
artifacts only under their separately authorized trust and verification
gates. The sole readiness exception is the minimum read-only resolution of
the original V8 T0 sentinel members at indices `[0,149,299]`, as specified
in section 3.2. Public artifacts and logs may contain aggregate counts,
hashes, status values, and safe provenance metadata only. They must not
contain ticker names, prices, raw payloads, private paths, or request
details that could reconstruct a private identity.

This task makes no production-code change, performs no allocation, makes no
Yahoo request, performs no acquisition, and performs no research opening.

---

## 12. V8C stage sequence

The V8C sequence is fixed as follows. No stage may be skipped, merged with a
different human gate, or treated as authorization for a later stage:

```text
CREATE_V8C_DESIGN_DRAFT
INDEPENDENT_V8C_DESIGN_REVIEW

T1C_T_SPARE_FRESHNESS_PRESERVATION_RECHECK
T2_PRESERVATION_RECHECK

V8C_DESIGN_FINALIZED
HUMAN_V8C_DESIGN_FREEZE

V8C_TRANSPORT_AND_ACQUISITION_IMPLEMENTATION
INDEPENDENT_V8C_PRODUCTION_IMPLEMENTATION_REVIEW

ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1C
EXECUTE_T1C_ALLOCATION
READ_ONLY_T1C_ALLOCATION_ARTIFACT_VERIFICATION

HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION
CREATE_V8C_TRUSTED_ALLOCATION_PIN
INDEPENDENT_TRUST_PIN_REVIEW

T1C_TRANSPORT_READINESS_HUMAN_GATE
EXECUTE_FIXED_T0_TRANSPORT_READINESS_PROBE_FOR_T1C

T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_T1C_RAW_ACQUISITION
READ_ONLY_T1C_ACQUISITION_ARTIFACT_VERIFICATION

SEPARATE_T1C_RESEARCH_OPENING_GATE
LAYER_B

FROZEN_FINAL_CANDIDATE

READ_ONLY_T2_PRESERVATION_RECHECK
HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE
CREATE_V8C_T2_AUTHORITY_BRIDGE
INDEPENDENT_V8C_T2_AUTHORITY_BRIDGE_REVIEW
T2_TRANSPORT_READINESS_HUMAN_GATE
EXECUTE_FIXED_T0_TRANSPORT_READINESS_PROBE_FOR_T2
T2_RAW_ACQUISITION_HUMAN_GATE
T2_RAW_ACQUISITION
READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION

SEPARATE_T2_RESEARCH_OPENING_GATE
LAYER_C
```

`T1C_T_SPARE_FRESHNESS_PRESERVATION_RECHECK` and `T2_PRESERVATION_RECHECK`
both occur after independent design review and before
`V8C_DESIGN_FINALIZED`. Both are hard prerequisites, reviewed against the
same exact design commit SHA as `INDEPENDENT_V8C_DESIGN_REVIEW` (§8):
unless both PASS, `V8C_DESIGN_FINALIZED` and `HUMAN_V8C_DESIGN_FREEZE` are
prohibited. `READ_ONLY_T2_PRESERVATION_RECHECK` is the second mandatory T2
check, immediately before the V8C-specific T2 authority bridge, T2
transport readiness, and T2 acquisition. The T1C freshness check has no
second recheck point of its own in this design; T2's two checks use the
same fixed conditions in section 7 and fail closed.

`HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE`, `CREATE_V8C_T2_AUTHORITY_BRIDGE`, and
`INDEPENDENT_V8C_T2_AUTHORITY_BRIDGE_REVIEW` are a separate authority
sequence (§7.2) that must PASS before T2 transport readiness or T2 raw
acquisition. The existing `V8B_T2_AUTHORITY_BRIDGE.json` does not satisfy
this sequence; a V8C-specific bridge is required.

The T1C transport readiness gate and probe occur only after the T1C
allocation and trust-pin prerequisites in the sequence, but before any
T1C opener. The T1C raw-acquisition gate is consumed only at the boundary
in section 5.
The T1C research-opening gate remains separate from raw acquisition. T2 is
not acquired or opened until after `FROZEN_FINAL_CANDIDATE`, its second
preservation recheck, its separate T2 readiness gate/probe, and its own
separate T2 acquisition and research-opening gates. Each T2 readiness
recheck requires a new explicit T2 readiness authorization.

---

## 13. Current design state

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

## 14. Design-only completion boundary

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
