# V8F Transport Window Semantics Successor Design Draft

This is a successor-study design draft only. It creates no implementation
authority, network authority, private-data authority, allocation authority,
human authorization, gate receipt, or research-opening authority.

## 1. Study identity and predecessor disposition

```text
study=V8F_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8E_HISTORICAL_RESEARCH
predecessor_terminal_adjudication_commit=1dd7f838fd16996a8d2c9a9501e2d45440422cc7
predecessor_terminal_artifact=V8E_T1C_READINESS_TERMINAL_ADJUDICATION.json
predecessor_terminal_artifact_blob=4a9f4153bae40ca43533850eaf4953ac13ce5562
```

V8F is a new study identity. It is not a V8E retry, amendment, continuation,
or repair under the V8E frozen design.

The V8E terminal disposition is:

```text
disposition=BLOCK_CLOSED
failure_class=TRANSPORT_PARSER_FAILURE
terminal_classification=PARSER_SCHEMA_FAILURE
canonical_collector_reason_code_or_family=OUT_OF_REQUEST_WINDOW
sentinel_pass_count=0/3
```

That disposition is transport-readiness evidence only. It is not strategy
evidence, profitability evidence, or T1C/T2 data-quality evidence.

Per the frozen predecessor record, the V8E root-cause status is preserved
exactly and is carried forward unchanged:

```text
root_cause_status=IMMEDIATE_FAILURE_CONFIRMED_ROOT_CAUSE_NOT_FULLY_PROVEN
root_cause_hypothesis(historical, V8E)=UTC_REQUEST_PERIOD_BOUNDARY_VS_JST_TRADING_DATE_VALIDATION_MISMATCH
```

This draft does NOT claim that hypothesis is a proven root cause. V8E's
0/3 BLOCK is motivation for a new, independently frozen study contract; it
is not V8F evidence, and V8E's BLOCK is not upgraded to a confirmed defect
by this document. V8F stands on its own frozen methodology and its own
fresh one-shot readiness execution, evaluated on its own terms.

No V8E human authorization, gate, preservation result, implementation
review, freeze approval, allocation-authority bridge, or readiness receipt
authorizes V8F. All V8F authority must be fresh and V8F-specific.

## 2. Single methodological change

V8F changes exactly one methodological contract relative to V8E: the Yahoo
request-boundary wire encoding used to compute `period1`/`period2`. No other
methodological item is changed by this draft.

### 2.1 Frozen contract: `JST_EXCHANGE_LOCAL_MIDNIGHT_REQUEST_BOUNDARY_V1`

The logical research interval is unchanged and remains exactly:

```text
[start_date, end_exclusive_date)
```

Logical dates remain exchange-local (Asia/Tokyo, JST) calendar dates. Only
the epoch encoding used to construct the outbound Yahoo request boundary
changes. The frozen encoding is:

```text
period_epoch(d) =
  int(
    datetime(
      d.year, d.month, d.day,
      0, 0, 0,
      tzinfo=ZoneInfo("Asia/Tokyo")
    ).timestamp()
  )

period1 = period_epoch(start_date)
period2 = period_epoch(end_exclusive_date)
```

This replaces the historical V7 `_epoch()` encoding, which computed
`datetime(y, m, d, tzinfo=timezone.utc).timestamp()` — UTC midnight rather
than JST midnight — for the outbound `period1`/`period2` query parameters.
That historical UTC-midnight encoding is confirmed present at the frozen
historical parser provenance below; this draft states that fact only as
provenance and does not assert it is the proven cause of the V8E BLOCK.

The frozen contract is exact and closed:

```text
no_padding=true
no_plus_or_minus_one_day_adjustment=true
no_window_broadening=true
no_post_response_clipping=true
no_dropping_of_out_of_request_window_rows=true
```

Returned timestamps remain canonicalized exactly as before:

```text
unix_instant -> UTC -> Asia/Tokyo calendar date
```

A canonical JST trading date outside `[start_date, end_exclusive_date)`
remains `OUT_OF_REQUEST_WINDOW` and fail-closed, with no relaxation, no
silent drop, and no reclassification. Duplicate-JST-date rejection and all
existing schema/OHLCV/split validation remain fail-closed and unchanged in
every other respect.

### 2.2 Implementation boundary

The historical V7 source, `src/v7_yahoo_collector.py`, and its historical
tests, `tests/test_v7_yahoo_collector.py`, MUST NOT be edited in place by
any future V8F implementation task. A future V8F implementation must
introduce study-scoped code (a V8F-namespaced transport module) that
reuses the unchanged canonical parsing, schema validation, and
classification behavior but applies the §2.1 request-boundary contract for
its own outbound requests. The historical V7 UTC-midnight-boundary test,
`test_request_uses_explicit_utc_epoch_bounds`, remains valid, unmodified
historical-behavior evidence for V7; it does not describe V8F behavior.

### 2.3 Historical parser provenance

The frozen classification and schema-validation behavior — unchanged in
V8F except for the §2.1 boundary encoding — remains bound to:

```text
canonical_parser_classifier_file=src/v7_yahoo_collector.py
canonical_parser_classifier_commit=28e281c3ee30d6b4c2f981c5da3ddc983c09724d
canonical_parser_classifier_blob=76b57b077f3214e666ff9dc06d9c224afc16df9f
```

This provenance binding, the canonical `V7YahooCollectorBlocked.reason`
allowlist and dynamic-family contract, and the origin-guard/canonical
collector evidence contracts defined in
`V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md` §7A (inherited unchanged
through V8E) remain binding on V8F without modification, except that the
audit additionally records the §5 request-boundary evidence below.

## 3. Unchanged inherited methodology

V8F inherits the V8E/V8D/V8C/V8B methodology unchanged except for the one
request-boundary encoding change in §2. In particular, V8F does not change:

- the Yahoo provider/host;
- `interval=1d`;
- request headers, `events`, `includeAdjustedClose`;
- the logical research interval semantics `[start_date, end_exclusive_date)`;
- readiness sentinels `[0, 149, 299]`, sentinel count 3;
- the readiness window, exactly `2025-12-01` to `2025-12-08` (exclusive);
- maximum attempts=3;
- maximum retries=2;
- backoff=`[5, 30]`;
- jitter=false;
- the retry classifier;
- the historical research period;
- labels or target definition;
- transaction costs or slippage;
- portfolio rules;
- the search space, stopping rules, promotion criteria, robustness rules;
- universe or partition definition;
- T1C or T2 membership;
- sample inclusion or exclusion;
- research-opening rules;
- the V8E DQ evidence contract (§4 of
  `V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md`), including the frozen policy:

```text
policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE
invalid_fraction_threshold=1/252
max_consecutive_invalid_returned_rows=1
full_P_hist_check=true
test_years=2018..2025
calendar_missing_dates_are_not_malformed_returned_rows=true
threshold_failure_action=BLOCK_WHOLE_ACQUISITION
```

The fraction comparison remains an exact integer comparison. No threshold
recalibration, provider substitution, retry change, redraw, alternate
partition, or stopping-rule change is permitted.

## 4. Anti-overfitting constraints

```text
readiness_dates_changed_because_of_V8E=false
sentinel_changed=false
provider_changed=false
retry_policy_changed=false
V8E_used_as_V8F_pass_evidence=false
fresh_V8F_one_shot_readiness_required=true
boundary_choice_selected_by_testing_against_V8E_outcome=false
```

The §2.1 boundary contract is a predeclared exchange-local calendar-date
semantics choice, fixed before any V8F execution and stated in full in this
frozen draft. It is not tuned, grid-searched, or selected by testing
alternative encodings against the V8E 0/3 outcome or any other real
execution result. V8E motivates why this successor study exists; it
supplies no V8F PASS evidence and authorizes no V8F execution.

## 5. Audit requirement

A future V8F transport/readiness dossier must safely record, in addition to
the inherited V8D/V8E audit fields:

```text
request_boundary_contract=JST_EXCHANGE_LOCAL_MIDNIGHT_REQUEST_BOUNDARY_V1
request_period1_epoch
request_period2_epoch
```

`request_period1_epoch` and `request_period2_epoch` are safe nonnegative
integers with no ticker binding. The independent read-only verifier must
recompute `period1`/`period2` from the frozen logical `start_date` and
`end_exclusive_date` using the exact §2.1 formula and require exact integer
equality with the recorded audit values. A producer-declared period value
that disagrees with the independently recomputed value is verifier BLOCK.
This recomputation requirement is additive to, and does not replace, the
full inherited V8D/V8E audit-verification requirements (schema, self-hash,
design binding, reviewed-implementation binding, classifier binding,
gate-consumption receipt binding, sentinel/window binding, attempt
numbering, retry-policy compliance, and per-attempt classification
evidence).

## 6. Mandatory future synthetic tests

Before any V8F production implementation review, synthetic tests must cover
at minimum:

- the exact JST-midnight `period1`/`period2` epoch computation for
  representative dates, including a DST-transition-free check appropriate
  to `Asia/Tokyo` (which observes no DST);
- the `2025-12-01` / `2025-12-08` readiness-window boundary fixture,
  asserting the exact `period1`/`period2` values the frozen formula
  produces for that exact window;
- an epoch-to-JST-date round trip: `period_epoch(d)` interpreted back
  through `unix_instant -> UTC -> Asia/Tokyo` recovers exactly `d`;
- a first-row and an interior in-window row are both accepted;
- an `end_exclusive_date` row is rejected as `OUT_OF_REQUEST_WINDOW`;
- a pre-`start_date` row is rejected as `OUT_OF_REQUEST_WINDOW`;
- a duplicate JST trading date remains a fail-closed rejection, unchanged
  from V7/V8E behavior;
- split-event boundary handling (`SPLIT_OUT_OF_REQUEST_WINDOW`,
  `DUPLICATE_SPLIT_EVENT`) remains unchanged;
- a fake/injected opener asserts the exact V8F outbound `period1`/`period2`
  query values for a representative window, proving the request actually
  sent matches the frozen §2.1 formula;
- `network_requests=0` for all of the above (fake opener only, no real
  Yahoo access); and
- `private_reads=0` and no modification to any historical V7 file
  (`src/v7_yahoo_collector.py`, `tests/test_v7_yahoo_collector.py`) as part
  of test execution or fixture setup.

All tests must use synthetic or fake data and temporary state only. No test
may use real network access, private identities, or private production
data.

## 7. Privacy-safe boundary

Public V8F design, audit, and evidence artifacts may expose only:

- the fixed request-boundary contract identifier;
- safe nonnegative integer epoch values for `period1`/`period2`;
- safe booleans, counts, and hashes;
- fixed threshold constants inherited unchanged from V8E; and
- Git commit/blob provenance.

No ticker identity, URL, raw payload, private path, price, feature, or
outcome may appear in public evidence. No private identity is inspected or
surfaced by this design task.

## 8. Authority: inherited V8E stage/gate structure, fresh V8F namespace

V8F inherits the full V8E stage and gate structure mechanically, using the
same current-study namespace-substitution rule V8E used relative to V8D
(`V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md` §3A.1–§3A.5):

```text
V8E_<CURRENT_STUDY_TOKEN> -> V8F_<CURRENT_STUDY_TOKEN>
study=V8E_HISTORICAL_RESEARCH -> study=V8F_HISTORICAL_RESEARCH
reviewed_v8e_design_candidate_commit -> reviewed_v8f_design_candidate_commit
v8e_frozen_design_commit -> v8f_frozen_design_commit
```

This applies to every current-study `schema_version`, `artifact_role`,
gate, review, stage, receipt, and freeze-status literal, including the full
preserved stage order:

```text
CREATE_V8F_DESIGN_DRAFT
INDEPENDENT_V8F_DESIGN_REVIEW

V8F_PREFREEZE_PRESERVATION_SUPPORT_IMPLEMENTATION
INDEPENDENT_V8F_PREFREEZE_PRESERVATION_SUPPORT_REVIEW

V8F_T1C_PRESERVATION_AUTHORITY_GATE
V8F_T1C_PRESERVATION_RECHECK
INDEPENDENT_V8F_T1C_PRESERVATION_RECHECK_REVIEW

V8F_T2_PREFREEZE_PRESERVATION_RECHECK
INDEPENDENT_V8F_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW

V8F_DESIGN_FINALIZED
HUMAN_V8F_DESIGN_FREEZE

V8F_TRANSPORT_AUDIT_IMPLEMENTATION
INDEPENDENT_V8F_PRODUCTION_IMPLEMENTATION_REVIEW

V8F_T1C_AUTHORITY_BRIDGE_GATE
CREATE_V8F_T1C_AUTHORITY_BRIDGE
INDEPENDENT_V8F_T1C_AUTHORITY_BRIDGE_REVIEW
V8F_T1C_READINESS_HUMAN_GATE
EXECUTE_FIXED_V8F_T1C_TRANSPORT_READINESS
READ_ONLY_V8F_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION

only if readiness PASS and its audit verification PASS:
V8F_T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8F_T1C_RAW_ACQUISITION
READ_ONLY_V8F_T1C_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_V8F_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION

only if raw acquisition PASS and both acquisition verifications PASS:
SEPARATE_V8F_T1C_RESEARCH_OPENING_GATE
V8F_T1C_RESEARCH_OPENING

V8F_T2_AUTHORITY_BRIDGE_GATE
CREATE_V8F_T2_AUTHORITY_BRIDGE
INDEPENDENT_V8F_T2_AUTHORITY_BRIDGE_REVIEW
V8F_T2_READINESS_HUMAN_GATE
EXECUTE_FIXED_V8F_T2_TRANSPORT_READINESS
READ_ONLY_V8F_T2_READINESS_TRANSPORT_AUDIT_VERIFICATION

only if T2 readiness PASS and its audit verification PASS:
READ_ONLY_V8F_T2_POINT_OF_USE_PRESERVATION_RECHECK
INDEPENDENT_V8F_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW

only if both T2 point-of-use preservation stages PASS:
V8F_T2_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8F_T2_RAW_ACQUISITION
READ_ONLY_V8F_T2_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_V8F_T2_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION

only if T2 raw acquisition PASS and both acquisition verifications PASS:
SEPARATE_V8F_T2_RESEARCH_OPENING_GATE
V8F_T2_RESEARCH_OPENING
```

No stage may be skipped or silently merged. No V8F readiness verification
substitutes for acquisition verification; no acquisition verification
substitutes for a research-opening authorization.

### 8.1 No V8E authority carries forward

No V8E human authorization, gate, preservation result, implementation
review, freeze approval, allocation-authority bridge, readiness receipt, or
readiness result authorizes any V8F stage. Every V8F prerequisite requires
fresh, V8F-specific authorization bound to the exact V8F frozen design
commit. Historical V8E, V8D, V8C, V8B, and V8 identifiers (original
partition hashes, trust-pin commits/blobs, T1C/T2 membership hashes, and
the V8E terminal adjudication commit/blob cited in §1) remain historical
evidence only; they are never renamed to V8F and never substitute for V8F
authority.

### 8.2 Inherited security semantics

The following semantics are copied without weakening or reinterpretation,
per `V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md` §3A.3 and
`AI_REAL_EXECUTION_RUNBOOK.md`:

- every preservation, authority, readiness, acquisition, and
  research-opening gate is one-shot;
- each receipt is durably published with flush/fsync and exclusive
  no-overwrite rules;
- `consumed=true` and `consumption_count=1` are required for a valid PASS;
- authorization reset, deletion, replay, and reuse are prohibited;
- a failed or malformed receipt is fail-closed and does not restore
  authorization;
- exact receipt-key and receipt-byte bindings are independently
  recomputed;
- raw authorization identities, ticker identities, private paths, raw
  payloads, prices, features, and outcomes remain prohibited from public
  evidence; and
- missing, duplicate, extra, malformed, mismatched, or unverifiable
  evidence is BLOCK, never an implicit PASS.

The Yahoo-request-bearing T1C/T2 readiness and raw-acquisition gate
consumption boundary remains exactly:

```text
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_YAHOO_REQUEST
```

The T1C prefreeze preservation private-verification gate boundary remains
exactly:

```text
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ
```

## 9. V8F readiness stopping rule

V8F permits exactly one top-level real V8F T1C readiness execution, and
only after full V8F design freeze, implementation review PASS, preservation
PASS, and a fresh V8F-specific allocation-authority bridge PASS. Frozen
internal retries remain allowed within that one execution, per the
unchanged `max attempts=3`, `max retries=2`, `backoff=[5,30]`, `jitter=false`
policy.

If the V8F T1C readiness result is BLOCK:

```text
no_second_V8F_readiness=true
no_retuning=true
no_provider_change=true
no_sentinel_change=true
no_window_change=true
no_retry_change=true
no_raw_acquisition=true
no_research_opening=true
```

A terminal adjudication artifact, analogous to
`V8E_T1C_READINESS_TERMINAL_ADJUDICATION.json`, must be produced, and a
successor-study decision is then required for any further real T1C
readiness work. This prevents repeated readiness tuning until PASS and
applies the same frozen stopping-rule discipline as
`AI_REAL_EXECUTION_RUNBOOK.md` §10 ("Frozen failure discipline").

## 10. Design task scope boundary

This design task itself performs:

```text
network_requests=0
private_reads=0
ticker_identity_inspection=0
gate_consumption=0
raw_acquisition=0
research_opening=0
```

```text
design_finalized=false
human_design_freeze_complete=false
implementation_created=false
approval_artifact_created=false
network_access_authorized=false
private_data_access_authorized=false
human_gate_consumed=false
```

This draft does not implement the producer or verifier, create an approval
or freeze artifact, read private files, inspect ticker identities, access
Yahoo or JPX, consume a gate, acquire raw data, open research, or evaluate
strategy profitability. Future V8F prefreeze preservation support and
production implementation follow the same two-phase ordering V8E used
(`V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md` §9.A–§9.B), rebound to the
V8F namespace, and require their own independent exact-SHA reviews before
any real private preservation execution or production transport
implementation begins.
