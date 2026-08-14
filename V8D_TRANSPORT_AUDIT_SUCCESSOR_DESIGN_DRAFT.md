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

## 5. Root defect V8D must fix

The predecessor finding is:

```text
V8C_POST_PRODUCTION_HIGH_1_READINESS_TRANSPORT_AUDIT_NOT_RETAINED
```

V8D production code must durably retain the frozen transport audit for every
Yahoo-request-bearing production path, including at least:

- T1C/T1D transport readiness;
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

## 6. Durability order

A terminal transport audit must be durably written before readiness or
acquisition code collapses or aggregates that failure. The code must never
convert an exception to `{"pass": false}` or an aggregate BLOCK in a way
that destroys the underlying audit.

A completed production execution must always leave independently readable
audit evidence for every attempted sentinel or logical request, including a
BLOCK. Persistence must be atomic and fail closed.

If audit persistence fails:

```text
execution_result=BLOCK
```

The execution must not silently continue, and missing audit evidence cannot
be treated as a successful or complete result.

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

V8D permits exactly one top-level real T1C/T1D readiness execution, and only
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

## 12. Raw acquisition

Only after V8D readiness PASS may a separate human gate authorize exact
validation-block raw acquisition. Readiness authorization never authorizes
acquisition.

Acquisition must use the same frozen research period and data-quality rules
as V8C, together with the same transport-audit retention requirements. A
consumed acquisition gate remains one-shot. Acquisition PASS does not imply
research opening.

## 13. T2

V8D must not inherit a V8B or V8C T2 human authorization.

Before any T2 action, V8D requires:

- a fresh V8D preservation recheck;
- a V8D-specific T2 authority bridge;
- independent bridge review;
- a separate T2 readiness human gate;
- a separate T2 acquisition human gate; and
- a separate T2 research-opening gate.

No T2 access occurs during this design task.

## 14. Proposed stage sequence

The following stages are ordered and may not be skipped or silently merged:

```text
CREATE_V8D_DESIGN_DRAFT
INDEPENDENT_V8D_DESIGN_REVIEW

V8D_T1C_PRESERVATION_RECHECK
V8D_DESIGN_FINALIZED
HUMAN_V8D_DESIGN_FREEZE

V8D_TRANSPORT_AUDIT_IMPLEMENTATION
INDEPENDENT_V8D_PRODUCTION_IMPLEMENTATION_REVIEW

HUMAN_V8D_T1C_AUTHORITY_BRIDGE_GATE
CREATE_V8D_T1C_AUTHORITY_BRIDGE
INDEPENDENT_V8D_T1C_AUTHORITY_BRIDGE_REVIEW

T1C_TRANSPORT_READINESS_HUMAN_GATE
EXECUTE_FIXED_T0_TRANSPORT_READINESS_PROBE_FOR_V8D_T1C
READ_ONLY_TRANSPORT_AUDIT_VERIFICATION

if readiness PASS:
T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8D_T1C_RAW_ACQUISITION
READ_ONLY_T1C_ACQUISITION_AND_TRANSPORT_AUDIT_VERIFICATION

SEPARATE_T1C_RESEARCH_OPENING_GATE
LAYER_B

then only through separately governed T2 stages.
```

The T1C preservation recheck must PASS before V8D design finalization and
human design freeze. The authority bridge, readiness, acquisition, and
research-opening gates remain separate human gates.

## 15. Design status

This first artifact is DRAFT only.

```text
design_finalized=false
human_design_freeze_complete=false
```

No network, private allocation, research, or human-gate authorization is
created by this draft.
