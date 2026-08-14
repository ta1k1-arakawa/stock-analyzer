# V8D Transport Audit Successor Design Draft

This document is a first successor-study design draft. It is design and
documentation only. It creates no implementation authority, network
authority, private-data authority, allocation authority, or research-opening
authority.

## 1. Study identity

```text
study=V8D_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8C_HISTORICAL_RESEARCH
predecessor_terminal_commit=d18368c1ec1c26d752ea5862115ab9f4315d1780
```

V8D is a new study identity. It is not a V8C retry, a V8C amendment, or a
continuation under the V8C frozen design.

The V8C readiness BLOCK is transport and auditability evidence only. It is
not strategy evidence, profitability evidence, or T1C data-quality evidence.

The immutable methodology and predecessor anchors are:

```text
inherited_methodology_authority_commit=c9c541ac7f7ba3bcca76db6250fe8273d9bb5756
predecessor_terminal_commit=d18368c1ec1c26d752ea5862115ab9f4315d1780
canonical_parser_classifier_file=src/v7_yahoo_collector.py
canonical_parser_classifier_commit=28e281c3ee30d6b4c2f981c5da3ddc983c09724d
canonical_parser_classifier_blob=76b57b077f3214e666ff9dc06d9c224afc16df9f
original_v8_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
original_v8_partition_implementation_commit=36cbed941050e728f7f96ce2af505e81175cc02c
original_v8_trusted_partition_blob=61faade0625139cec3fb61216ab2f97f572a7028
```

After `HUMAN_V8D_DESIGN_FREEZE`, semantic design changes inside V8D are
prohibited unless this frozen design explicitly pre-authorizes the exact
change. Otherwise a successor study is required.

## 2. Unchanged research methodology

Unless an explicitly authorized successor-study design later changes a
permitted item, V8D inherits the frozen V8C/V8B methodology unchanged:

- historical period;
- labels and target definition;
- walk-forward and causality rules;
- transaction costs and slippage;
- portfolio rules;
- data-quality policy and thresholds;
- search space;
- promotion criteria;
- robustness rules; and
- holdout and research-opening rules.

There is no recalibration, threshold tuning, result-conditioned
methodological change, or methodology change derived from the V8C BLOCK.

The frozen data-quality policy remains
`POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE`, with
`invalid_fraction_threshold=1/252` and
`max_consecutive_invalid_returned_rows=1`. The frozen historical period,
labels, walk-forward, causality, friction, strategy-search, promotion, and
opening rules remain binding.

## 3. Conditional preservation of the existing T1C

The only validation-block candidate V8D may consider is the exact already
allocated V8C T1C artifact. Its privacy-safe public bindings are:

```text
allocation_artifact_self_hash=16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c
t1c_ticker_count=300
t1c_ticker_list_sha256=85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54
parent_t_spare_ticker_list_sha256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
remaining_t_spare_ticker_list_sha256=699e7bc29b2714128de99203bd6fedb38ee24c6f7bfee7c725b605669c178632
```

No identity is to be inspected, materialized, or displayed by this design
draft.

Before V8D design freeze, a new privacy-safe preservation recheck must
positively establish all of the following:

- V8C T1C raw acquisition never occurred;
- T1C research opening never occurred;
- T1C OHLCV research access never occurred;
- T1C feature access never occurred;
- T1C outcome access never occurred;
- T1C identities were not publicly exposed or exposed to the research loop;
- T1C membership was not reassigned;
- the allocation self-hash is unchanged;
- original V8 provenance is unchanged; and
- the V8C terminal adjudication remains authoritative.

The pre-freeze preservation sequence is explicitly separate:

```text
HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE
V8D_T1C_PRESERVATION_RECHECK
INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW
```

The human gate authorizes exactly one minimum read-only verification of the
existing private V8C T1C allocation artifact and the authoritative private V8
partition manifest, solely to verify the existing allocation and provenance
commitments. It authorizes no ticker display, raw OHLCV access,
feature/outcome access, network, allocation, redraw, or research opening.

The preservation output must contain only public/privacy-safe hashes, counts,
booleans, and Git provenance. The private verification must positively verify
that the current allocation artifact self-hash is
`16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c` and that
the current T1C list remains `count=300` with
`sha256=85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54`.
Absence of evidence is not PASS. The preservation artifact must receive an
independent exact-SHA review before `V8D_DESIGN_FINALIZED`.

The preservation gate and authorization grammar are frozen as:

```text
gate=HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE
authorization_identity="V8D_HUMAN_AUTHORIZE_T1C_PRESERVATION_VERIFY_AT_"
                      + reviewed_design_candidate_commit
                      + "_FOR_"
                      + allocation_artifact_self_hash
```

`reviewed_design_candidate_commit` is the exact 40-hex V8D design candidate
that received independent design-review PASS. The gate is per-authorization
and one-shot. It is durably consumed immediately before the first byte is
read from either the private T1C allocation artifact or the authoritative
private V8 partition manifest. Public Git and provenance preflight may occur
before that boundary.

After consumption, reset, deletion, and reuse of the same identity are
prohibited. Failure does not restore authorization. Another private
verification requires a fresh explicit human authorization.

The privacy-safe durable consumption receipt is frozen as:

```text
schema_version=V8D_T1C_PRESERVATION_GATE_RECEIPT_V1
study=V8D_HISTORICAL_RESEARCH
artifact_role=T1C_PRESERVATION_PRIVATE_GATE_RECEIPT
gate=HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE
reviewed_design_candidate_commit
authorization_identity_sha256
authorized_allocation_artifact_self_hash
consumed=true
consumption_count=1
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ
consumption_timestamp_utc
receipt_self_hash
```

This receipt schema is an exact field set: no extra fields and no omitted
fields are permitted.

The receipt is machine-local/private-safe evidence and must never contain
the raw human authorization identity, ticker identities, private paths, raw
allocation or manifest content, OHLCV, features, or outcomes. Only
privacy-safe hashes, booleans, counts, and provenance may be surfaced
publicly.

A valid PASS preservation execution requires exactly one valid receipt with
the exact gate, exact `reviewed_design_candidate_commit`, exact
`authorized_allocation_artifact_self_hash=16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c`,
`consumed=true`, `consumption_count=1`, and
`consumption_boundary=IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ`. The
receipt `authorization_identity_sha256` must equal SHA-256 of the exact
human authorization identity supplied for that execution.

If the receipt is missing, malformed, duplicated, mismatched, has a
consumption count other than 1, is bound to another design SHA or allocation
artifact, or cannot prove the frozen boundary, then:

```text
V8D_T1C_PRESERVATION_RECHECK=BLOCK
INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW=BLOCK
V8D_DESIGN_FINALIZED=PROHIBITED
HUMAN_V8D_DESIGN_FREEZE=PROHIBITED
```

No authorization reset or reuse is permitted. The public preservation
artifact and independent review artifact must not contain the raw human
authorization identity.

Public preservation and review provenance may expose only the gate name,
`consumed=true`, `consumption_count=1`, authorization identity SHA-256,
receipt self-hash, PASS/BLOCK, and design/allocation hashes. It may not expose
the raw authorization identity or private paths or content.

The preservation artifact schema and exact field-set requirement are:

```text
schema_version=V8D_T1C_PRESERVATION_RECHECK_V1
artifact_role=T1C_PRESERVATION_RECHECK
study=V8D_HISTORICAL_RESEARCH
reviewed_design_candidate_commit
source_v8c_terminal_commit
allocation_artifact_self_hash
t1c_ticker_count
t1c_ticker_list_sha256
parent_t_spare_ticker_list_sha256
remaining_t_spare_ticker_list_sha256
t1c_raw_acquisition_performed=false
t1c_research_opened=false
t1c_ohlcv_research_access=false
t1c_feature_access=false
t1c_outcome_access=false
t1c_identities_publicly_exposed=false
t1c_membership_reassigned=false
allocation_self_hash_unchanged=true
parent_v8_provenance_unchanged=true
v8c_terminal_adjudication_authoritative=true
preservation_recheck_result=PASS
```

preservation_artifact_self_git_identity_inside_artifact=PROHIBITED

The two preservation artifact Git identity fields are intentionally absent:
`preservation_recheck_git_commit` and `preservation_recheck_git_blob_sha`.
The preservation artifact is a privacy-safe content artifact whose own Git
identity is established externally after its bytes and tree exist. Its
remaining fields are the privacy-safe frozen preservation commitments,
hashes, counts, and booleans required above; no extra or identity-bearing
field is permitted. Exact field-set validation is fail closed, and PASS is
valid only when every frozen preservation condition is positively verified.

`INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW` must review an exact
40-hex preservation artifact commit and mechanically resolve the artifact
blob from that exact commit. It must independently read and verify the
durable gate receipt and must not trust the preservation producer's statement
that authorization was consumed. At minimum it verifies the exact receipt
schema, receipt self-hash/integrity, exact gate, exact
`reviewed_design_candidate_commit`, exact allocation artifact self-hash,
`consumed=true`, `consumption_count=1`, the frozen consumption boundary, the
authorization identity hash associated with the authorized execution, and
that the receipt corresponds to the same preservation execution whose
artifact is reviewed.

The independent review record and provenance bind externally to:

```text
reviewed_preservation_recheck_git_commit
reviewed_preservation_recheck_git_blob_sha
reviewed_design_candidate_commit
preservation_recheck_result=PASS
reviewed_gate_receipt_self_hash
gate_receipt_validation_result=PASS
```

Absence of evidence is not PASS. If preservation cannot be positively
established, then:

```text
V8D_T1C_REUSE=BLOCK
automatic_alternate_T_spare_slice=PROHIBITED
automatic_redraw=PROHIBITED
automatic_T3_substitution=PROHIBITED
CHATGPT_DECISION_REQUIRED
```

This draft does not automatically authorize T1C reuse merely because the
candidate is named.

The validation identity is frozen:

```text
validation_block_identity=EXACT_EXISTING_V8C_T1C_ONLY
new_validation_block_creation=false
alternate_T_spare_slice=false
redraw=false
T3_substitution=false
```

## 4. V8D-specific allocation authority

`V8C_TRUSTED_ALLOCATION.json` may be used as historical allocation
provenance only. It does not authorize V8D network or research execution.

After V8D freeze, a new V8D-specific allocation-authority bridge must be
defined and independently reviewed. It must bind to:

- the exact V8D frozen design SHA;
- the exact existing allocation artifact self-hash;
- the V8C trust-pin Git commit and blob;
- original V8 partition provenance; and
- the successful V8D preservation recheck.

The bridge may contain only safe hashes, counts, and provenance. Its creation
requires its own explicit human authorization.

The allocation-authority bridge must not be semantically bound to a
particular V8D transport implementation commit. Transport implementation
authority is reviewed separately.

The V8D T1C allocation-authority bridge schema and validation semantics are
defined before freeze. It contains only safe fields:

```text
schema_version=V8D_T1C_ALLOCATION_AUTHORITY_BRIDGE_V1
study=V8D_HISTORICAL_RESEARCH
artifact_role=T1C_ALLOCATION_AUTHORITY_BRIDGE
logical_block=T1C
v8d_frozen_design_commit
source_v8c_terminal_commit=d18368c1ec1c26d752ea5862115ab9f4315d1780
source_v8c_trust_pin_git_commit=2a65674d8439f5964ff694494d5dad5ed19ad0f6
source_v8c_trust_pin_git_blob_sha=61082f9818efb68ca2a5ad29fa5918f887575c10
authorized_allocation_artifact_self_hash=16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c
t1c_ticker_count=300
t1c_ticker_list_sha256=85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54
parent_v8_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
parent_v8_partition_implementation_commit=36cbed941050e728f7f96ce2af505e81175cc02c
parent_t_spare_ticker_list_sha256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
preservation_recheck_git_commit=<exact independently reviewed PASS artifact commit>
preservation_recheck_git_blob_sha=<exact independently reviewed PASS artifact blob>
preservation_recheck_result=PASS
human_gate
authorization_status=AUTHORIZED
authorization_note
```

This is an exact field set: no extra fields and no omitted fields are
permitted. The preservation recheck commit and blob must be the exact
independently reviewed PASS artifact accepted by
`INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW`; the producer has zero
discretion to substitute another artifact. The bridge must not bind a V8D
production implementation commit. Its human gate grammar is frozen as:

```text
human_gate="V8D_HUMAN_AUTHORIZE_T1C_AUTHORITY_BRIDGE_AT_"
            + v8d_frozen_design_commit
            + "_FOR_"
            + authorized_allocation_artifact_self_hash
```

Bridge creation itself requires the explicit human gate and must not read
ticker identities or private OHLCV.

### Point-of-use authority revalidation

An earlier preservation PASS is not sufficient authority for raw acquisition.
Immediately before T1C raw-acquisition gate consumption and before the first
real Yahoo request, the authorized production acquisition preflight must
re-read and independently verify the current private V8C T1C allocation
artifact and the authoritative private V8 partition manifest. It must apply
the same semantic verification as the official V8C allocation verifier:

```text
authorized_allocation_artifact_self_hash=16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c
t1c_ticker_count=300
t1c_ticker_list_sha256=85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54
parent_t_spare_ticker_count=1904
parent_t_spare_ticker_list_sha256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
parent_v8_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
parent_v8_partition_implementation_commit=36cbed941050e728f7f96ce2af505e81175cc02c
```

The verified artifact must imply the exact existing T1C membership. No
redraw, substitution, or reassignment is permitted. This point-of-use
verification is part of the explicitly authorized T1C raw-acquisition
preflight, but occurs before raw-acquisition gate consumption and before any
Yahoo request. If it fails, then `Yahoo requests=0`, the raw-acquisition gate
remains unconsumed, raw acquisition is BLOCK, no automatic retry is allowed,
and execution stops for GPT adjudication. Only after every check PASS may
the one-shot raw-acquisition gate be consumed immediately before the first
Yahoo opener.

The equivalent point-of-use rule applies to T2: immediately before T2
raw-acquisition gate consumption, the authorized preflight must re-read the
authoritative V8 partition manifest and revalidate the exact T2 count, T2
list hash, and original authority. Failure leaves Yahoo requests at zero,
leaves the gate unconsumed, and is BLOCK without automatic retry.

## 5. Root defect V8D must fix

The predecessor finding is:

```text
V8C_POST_PRODUCTION_HIGH_1_READINESS_TRANSPORT_AUDIT_NOT_RETAINED
```

V8D production code must durably retain the frozen transport audit for every
Yahoo-request-bearing production path, including at least:

- T1C transport readiness;
- validation-block raw acquisition;
- T2 transport readiness; and
- T2 raw acquisition.

For every request attempt, private durable audit must preserve enough
privacy-safe metadata to independently re-derive the classification without
message heuristics:

- logical stage;
- privacy-safe sentinel or block coordinate, never ticker identity;
- attempt number;
- classification;
- retryable boolean;
- concrete exception type;
- HTTP code when applicable;
- reason type when applicable;
- errno when applicable;
- named condition when applicable;
- request fingerprint or equivalent frozen-request equality evidence; and
- terminal or nonterminal state.

The audit must never persist ticker identity in public output, a URL
containing a ticker, raw payload, price, raw exception message, or private
filesystem path. Private audit may contain only the minimum private-safe
metadata required by the frozen audit requirement.

## 6. Per-attempt durability order

The following order is frozen for every request attempt, not only terminal
failures:

```text
request attempt
→ construct privacy-safe audit record
→ durably/atomically persist that attempt record
→ only then permit:
   - retry backoff
   - next retry attempt
   - success return
   - failure aggregation
   - readiness/acquisition aggregate publication
```

A retryable failed attempt must be durable before sleeping or retrying. A
successful attempt must be durable before returning success. The code must
never convert an exception to `{"pass": false}` or an aggregate BLOCK in a
way that destroys the underlying audit.

A completed production execution must always leave independently readable
audit evidence for every attempted sentinel or logical request, including a
BLOCK. Persistence must be atomic and fail closed.

If attempt-audit persistence fails:

```text
execution=BLOCK
no_next_request=true
no_success_return=true
no_aggregate_PASS=true
```

The execution must stop without silently continuing, and missing audit
evidence cannot be treated as a successful or complete result.

## 7. Public privacy-safe summary

The public readiness or acquisition aggregate may expose only safe
aggregates, such as:

- result PASS or BLOCK;
- sentinel count and pass count;
- total request attempt count;
- retry count;
- terminal classification histogram;
- attempt-count histogram;
- retryable and nonretryable aggregate counts;
- HTTP-status histogram when applicable;
- audit artifact self-hash; and
- design, review, and classifier provenance.

No mapping from a classification to a ticker identity may be public.
Request fingerprints must not be exposed publicly. Only safe equality or
consistency results, or hashes explicitly established as safe by the frozen
design, may be exposed.

## 7A. Independently re-derivable named-condition evidence

The verifier must not merely trust a stored `named_condition`. For each
named condition below, the private audit stores only the frozen,
privacy-safe detector evidence, and the read-only verifier recomputes the
classification from that evidence. No raw URL, ticker, payload, price, or
exception message is permitted.

The finite evidence contract is:

```text
UNTRUSTED_REDIRECT:
  scheme_https: boolean
  hostname_matches_expected: boolean
  credentials_absent: boolean
  port_allowed: boolean
  context=REDIRECT_TARGET

RESPONSE_HOST_MISMATCH:
  scheme_https: boolean
  hostname_matches_expected: boolean
  credentials_absent: boolean
  port_allowed: boolean
  context=INITIAL_OR_FINAL_RESPONSE

PARSER_SCHEMA_FAILURE:
  parser_schema_valid=false
  canonical_collector_reason_code_or_family: EXACT_CANONICAL_ALLOWLIST

SYMBOL_MISMATCH:
  expected_symbol_binding=false

DATA_QUALITY_GATE_FAILURE:
  nonempty_timestamp: boolean
  valid_price_row_count: nonnegative safe integer
  trading_date_fields_valid: boolean
```

The exact canonical `V7YahooCollectorBlocked.reason` contract at the frozen
parser blob is:

```text
fixed literals:
EMPTY_TICKER
INVALID_REQUEST_DATE_ORDER
RESPONSE_HOST_MISMATCH
TIMESTAMP_INVALID
PAYLOAD_JSON_INVALID
PAYLOAD_ROOT_INVALID
CHART_ERROR
CHART_RESULT_INVALID
INDICATORS_MISSING
SPLIT_RATIO_INVALID
EVENTS_INVALID
SPLITS_INVALID
SPLIT_EVENT_INVALID
SPLIT_OUT_OF_REQUEST_WINDOW
DUPLICATE_SPLIT_EVENT
SPLIT_NUMERATOR_DENOMINATOR_MISSING
SPLIT_NUMERATOR_DENOMINATOR_INVALID
SPLIT_RATIO_MISMATCH
METADATA_MISSING
SYMBOL_MISMATCH
TIMESTAMP_MISSING
OUT_OF_REQUEST_WINDOW
DUPLICATE_TRADING_DATE
RESPONSE_BYTES_INVALID

dynamic families:
INVALID_DATE:<field> where field is exactly start or end_exclusive
HTTP_STATUS_<status> where status is exactly the literal None or a
  canonical base-10 signed integer string returned by response.status or
  response.getcode
INDICATOR_SECTION_INVALID:<section> where section is exactly quote or adjclose
ARRAY_LENGTH_MISMATCH:<field> where field is exactly open, high, low, close,
  volume, or adjclose
```

This allowlist and these suffix domains are mechanically bound to
`src/v7_yahoo_collector.py` at commit
`28e281c3ee30d6b4c2f981c5da3ddc983c09724d`, blob
`76b57b077f3214e666ff9dc06d9c224afc16df9f`. The audit stores the structured
canonical collector reason code or family, never `str(exception)`, a raw
exception message, ticker, URL, or payload.

`SYMBOL_MISMATCH` derives to `SYMBOL_MISMATCH`,
`RESPONSE_HOST_MISMATCH` derives to `RESPONSE_HOST_MISMATCH`, and every other
allowed canonical reason reaching the frozen readiness or acquisition
wrapper derives to `PARSER_SCHEMA_FAILURE`. Any reason outside this exact
allowlist or dynamic-family contract is verifier BLOCK. The evidence
contract stores neither symbol value. The verifier mechanically checks the
canonical reason against this contract, recomputes the named classification,
and rejects a producer-stated classification that disagrees. The separately
defined privacy-safe detector evidence for `UNTRUSTED_REDIRECT` and
`RESPONSE_HOST_MISMATCH` remains applicable when produced outside the
canonical collector.

## 8. Independent audit verifier

V8D must provide a read-only verifier that independently validates:

- private audit schema;
- self-hash and integrity;
- exact design binding;
- reviewed implementation binding;
- classifier binding;
- gate-consumption receipt binding;
- sentinel and window binding;
- attempt numbering;
- retry-policy compliance;
- exact retryable and nonretryable classifications;
- concrete metadata and classification consistency;
- identical request fingerprints across retries of the same request;
- agreement between aggregate public result and private audit; and
- absence of missing terminal-failure evidence.

The verifier must re-derive these conditions and must not trust a producer's
aggregate declaration.

## 9. Mandatory end-to-end test coverage

Before production review, synthetic tests must cover the complete path:

```text
actual concrete exception object
→ frozen classifier
→ retry wrapper
→ readiness/acquisition catch path
→ durable private audit
→ durable aggregate evidence
→ independent audit verifier
```

Tests must cover at minimum:

- every frozen retryable HTTP class;
- every frozen nonretryable HTTP class;
- `TimeoutError` and socket-timeout runtime behavior;
- `URLError` timeout;
- connection reset;
- `EAI_AGAIN` temporary DNS;
- permanent `gaierror`;
- unknown exception;
- `UNTRUSTED_REDIRECT`;
- `RESPONSE_HOST_MISMATCH`;
- `PARSER_SCHEMA_FAILURE`;
- `SYMBOL_MISMATCH`;
- `DATA_QUALITY_GATE_FAILURE`;
- retry exhaustion;
- success after retry;
- audit write failure;
- malformed or tampered audit;
- missing attempt;
- forged concrete metadata; and
- fingerprint mismatch across retries.

All tests must use synthetic or fake data and temporary state only. No test
may use real network access, private identities, or private production data.

## 10. Fixed readiness conditions

V8D must not change the readiness test in response to the V8C 0/3 result.
The readiness definition remains exactly:

```text
sentinel_source=original trusted V8 T0
sentinel_indices=[0,149,299]
sentinel_count=3
probe_start=2025-12-01
probe_end_exclusive=2025-12-08
```

The following remain frozen and unchanged:

- Yahoo host and source;
- parser and classifier semantic behavior;
- retryable HTTP list;
- nonretryable rules;
- maximum attempts=3;
- maximum retries=2;
- backoff=[5,30]; and
- jitter=false.

No provider, sentinel, or window substitution is permitted based on the V8C
BLOCK.

## 11. V8D readiness stopping rule

V8D permits exactly one top-level real T1C readiness execution, and only
after all of the following are complete:

- V8D design freeze;
- implementation review PASS;
- preservation recheck PASS;
- V8D allocation-authority bridge PASS; and
- fresh explicit readiness human authorization.

Frozen internal retries remain allowed within that one execution.

If the V8D readiness result is BLOCK:

- validation raw acquisition remains prohibited;
- no second V8D top-level readiness probe is permitted;
- no parameter, provider, sentinel, or window tuning is permitted;
- terminal classification must remain recoverable from the new audit; and
- further real readiness work requires a later successor-study decision.

This prevents repeated readiness tuning until PASS.

The T1C raw-acquisition hard gate is frozen as:

```text
T1C_RAW_ACQUISITION_ALLOWED iff:
  readiness_result=PASS
  AND READ_ONLY_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION=PASS
```

If either condition is not PASS, `T1C_RAW_ACQUISITION=PROHIBITED`.

## 12. Raw acquisition

Only after V8D readiness PASS and transport-audit verification PASS may a
separate human gate authorize exact T1C validation-block raw acquisition.
Readiness authorization never authorizes acquisition.

Acquisition must use the same frozen research period and data-quality rules
as V8C, together with the same transport-audit retention requirements. A
consumed acquisition gate remains one-shot. Acquisition PASS does not imply
research opening.

The T1C research-opening hard gate is frozen as:

```text
T1C_RESEARCH_OPENING_ALLOWED iff:
  raw_acquisition_result=PASS
  AND READ_ONLY_T1C_ACQUISITION_ARTIFACT_VERIFICATION=PASS
  AND READ_ONLY_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION=PASS
  AND fresh separate research-opening human authorization exists
```

No verification PASS may imply research opening by itself.

`READ_ONLY_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION` and
`READ_ONLY_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION` are distinct and
non-substitutable. Each verification is bound to the exact production
execution and transport-audit artifact from its own phase. Readiness
transport-audit verification cannot substitute for acquisition
transport-audit verification, and acquisition transport-audit verification
cannot retroactively satisfy the raw-acquisition gate.

## 13. T2

V8D must not inherit a V8B or V8C T2 human authorization.

Before any T2 action, V8D requires both mandatory preservation checkpoints,
as well as:

- a V8D-specific T2 authority bridge;
- independent bridge review;
- a separate T2 readiness human gate;
- a separate T2 acquisition human gate; and
- a separate T2 research-opening gate.

The nine T2 preservation conditions are inherited and frozen exactly:

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

V8D has two distinct mandatory T2 preservation checkpoints, and both evaluate
exactly the nine conditions above:

```text
V8D_T2_PREFREEZE_PRESERVATION_RECHECK
READ_ONLY_V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK
```

`V8D_T2_PREFREEZE_PRESERVATION_RECHECK` occurs before
`V8D_DESIGN_FINALIZED` and `HUMAN_V8D_DESIGN_FREEZE`. The pre-freeze result
requires the independent stage
`INDEPENDENT_V8D_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW`; both must PASS
before design finalization or human design freeze.

`READ_ONLY_V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK` occurs later,
immediately before the ordered T2 authority and acquisition path. Both
checkpoints use safe committed, audit, and provenance evidence only; neither
inspects T2 ticker identities or reads raw T2 data, makes a network request,
or creates acquisition or research authority.

Absence of evidence is not PASS. If positive verification cannot be
established at either checkpoint, the result is BLOCK and execution stops.

The conceptual V8D T2 authority bridge is defined before freeze and contains
only safe fields:

```text
schema_version=V8D_T2_AUTHORITY_BRIDGE_V1
study=V8D_HISTORICAL_RESEARCH
artifact_role=T2_AUTHORITY_BRIDGE
logical_block=T2
v8d_frozen_design_commit
source_authority=ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY
v8_trust_anchor_git_identity=61faade0625139cec3fb61216ab2f97f572a7028
authorized_parent_v8_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
parent_v8_partition_implementation_commit=36cbed941050e728f7f96ce2af505e81175cc02c
expected_t2_ticker_count=300
expected_t2_ticker_list_sha256=e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500
preservation_recheck_git_commit
preservation_recheck_git_blob_sha
preservation_recheck_result=PASS
human_gate
authorization_status=AUTHORIZED
authorization_note
```

The T2 bridge must not bind a V8D production implementation commit.
Transport implementation authority is verified separately at point of use.
Its exact human-gate grammar is:

```text
human_gate="V8D_HUMAN_AUTHORIZE_T2_AUTHORITY_BRIDGE_AT_"
            + v8d_frozen_design_commit
            + "_FOR_"
            + expected_t2_ticker_list_sha256
```

An independent exact-SHA T2 bridge review is required before T2 readiness.

No T2 access occurs during this design task.

## Exact-SHA design freeze protocol

### Exact design candidate identity

The design candidate binding is frozen as:

```text
design_candidate_binding=EXACT_ONE_40_HEX_GIT_COMMIT_SHA
moving_branch_binding=INVALID
latest_HEAD_binding=INVALID
working_tree_binding=INVALID
reviewed_design_candidate_commit=exactly one 40-hex Git commit containing V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md
```

`reviewed_design_candidate_commit` is the one exact 40-hex Git commit that
contains the candidate V8D design document. No branch name, moving branch,
latest HEAD, or working-tree state may substitute for that SHA.

The following three prerequisite chains must all refer to the same exact
`reviewed_design_candidate_commit`:

1. `INDEPENDENT_V8D_DESIGN_REVIEW`;
2. `HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE`,
   `V8D_T1C_PRESERVATION_RECHECK`, and
   `INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW`; and
3. `V8D_T2_PREFREEZE_PRESERVATION_RECHECK` and
   `INDEPENDENT_V8D_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW`.

### Exact finalization rule

`V8D_DESIGN_FINALIZED` is allowed only if all of the following are PASS for
the same exact `reviewed_design_candidate_commit`:

- `INDEPENDENT_V8D_DESIGN_REVIEW`;
- `V8D_T1C_PRESERVATION_RECHECK`;
- `INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW`;
- `V8D_T2_PREFREEZE_PRESERVATION_RECHECK`; and
- `INDEPENDENT_V8D_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW`.

If any result is missing, BLOCK, refers to a different design commit, or
cannot prove its exact SHA binding, then:

```text
V8D_DESIGN_FINALIZED=PROHIBITED
HUMAN_V8D_DESIGN_FREEZE=PROHIBITED
```

### Semantic change invalidation

Any semantic change to
`V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md` after any prerequisite
review or recheck has completed creates a new design candidate SHA. All
prior candidate-specific results become invalid for the new candidate:

```text
prior_independent_design_review=INVALID_FOR_NEW_CANDIDATE
prior_t1c_preservation_recheck=INVALID_FOR_NEW_CANDIDATE
prior_t1c_preservation_independent_review=INVALID_FOR_NEW_CANDIDATE
prior_t2_prefreeze_preservation_recheck=INVALID_FOR_NEW_CANDIDATE
prior_t2_prefreeze_preservation_independent_review=INVALID_FOR_NEW_CANDIDATE
```

The applicable stages must be repeated for the new exact candidate SHA. A
prior favorable preservation result may not be carried forward across a
semantic design change.

### Human design freeze

`HUMAN_V8D_DESIGN_FREEZE` must explicitly name and authorize the exact
40-hex `reviewed_design_candidate_commit` that satisfied every prerequisite
PASS condition. One human freeze authorization cannot silently authorize a
different design SHA.

After `HUMAN_V8D_DESIGN_FREEZE`:

```text
semantic_design_change=PROHIBITED
```

This prohibition may be relaxed only where the frozen design explicitly
pre-authorizes the exact change. Otherwise:

```text
NEW_SUCCESSOR_STUDY_REQUIRED
```

### Freeze approval record

After explicit human freeze authorization, a separate later
public/privacy-safe freeze-approval artifact records the authorization. The
freeze-approval artifact commit is not the frozen design commit and is not
self-referential.

Its conceptual minimum schema is:

```text
schema_version=V8D_DESIGN_FREEZE_APPROVAL_V1
study=V8D_HISTORICAL_RESEARCH
frozen_design_git_commit
design_document=V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md
final_independent_design_review_result=PASS
final_independent_design_review_commit
t1c_preservation_recheck_result=PASS
t1c_preservation_recheck_design_commit
t1c_preservation_independent_review_result=PASS
t1c_preservation_independent_review_design_commit
t2_prefreeze_preservation_recheck_result=PASS
t2_prefreeze_preservation_recheck_design_commit
t2_prefreeze_preservation_independent_review_result=PASS
t2_prefreeze_preservation_independent_review_design_commit
approval_status=APPROVED
human_gate
```

Every design-commit field above must resolve to the same exact
`frozen_design_git_commit` wherever it represents design-candidate binding.
The approval artifact must not contain a self-referential binding to its own
Git commit or blob.

## 14. Proposed stage sequence

The following stages are ordered and may not be skipped or silently merged:

```text
CREATE_V8D_DESIGN_DRAFT
INDEPENDENT_V8D_DESIGN_REVIEW

HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE
V8D_T1C_PRESERVATION_RECHECK
INDEPENDENT_V8D_T1C_PRESERVATION_RECHECK_REVIEW

V8D_T2_PREFREEZE_PRESERVATION_RECHECK
INDEPENDENT_V8D_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW

V8D_DESIGN_FINALIZED
HUMAN_V8D_DESIGN_FREEZE

V8D_TRANSPORT_AUDIT_IMPLEMENTATION
INDEPENDENT_V8D_PRODUCTION_IMPLEMENTATION_REVIEW

HUMAN_V8D_T1C_AUTHORITY_BRIDGE_GATE
CREATE_V8D_T1C_AUTHORITY_BRIDGE
INDEPENDENT_V8D_T1C_AUTHORITY_BRIDGE_REVIEW

T1C_TRANSPORT_READINESS_HUMAN_GATE
EXECUTE_FIXED_T0_TRANSPORT_READINESS_PROBE_FOR_V8D_T1C
READ_ONLY_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION

only if readiness PASS
AND READ_ONLY_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION PASS:
T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8D_T1C_RAW_ACQUISITION
READ_ONLY_T1C_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION

only if raw acquisition PASS
AND READ_ONLY_T1C_ACQUISITION_ARTIFACT_VERIFICATION PASS
AND READ_ONLY_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION PASS:
SEPARATE_T1C_RESEARCH_OPENING_GATE
LAYER_B

READ_ONLY_V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK
INDEPENDENT_V8D_T2_PRESERVATION_RECHECK_REVIEW
HUMAN_V8D_T2_AUTHORITY_BRIDGE_GATE
CREATE_V8D_T2_AUTHORITY_BRIDGE
INDEPENDENT_V8D_T2_AUTHORITY_BRIDGE_REVIEW
T2_TRANSPORT_READINESS_HUMAN_GATE
EXECUTE_FIXED_T0_TRANSPORT_READINESS_PROBE_FOR_T2
READ_ONLY_T2_TRANSPORT_AUDIT_VERIFICATION

only if readiness PASS AND audit verification PASS:
T2_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8D_T2_RAW_ACQUISITION
READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_T2_TRANSPORT_AUDIT_VERIFICATION

only after all required verification PASS:
SEPARATE_T2_RESEARCH_OPENING_GATE
LAYER_C
```

The independent design review, T1C pre-freeze preservation chain, and T2
pre-freeze preservation chain must all bind to the same exact candidate SHA
and PASS before V8D design finalization or human design freeze. Each T2
preservation, authority-bridge, readiness, acquisition, transport-audit, and
research-opening stage is separate and ordered. No stage may be skipped or
silently merged.

## 15. Design status

This first artifact is DRAFT only.

```text
design_finalized=false
human_design_freeze_complete=false
```

No network, private allocation, research, or human-gate authorization is
created by this draft.
