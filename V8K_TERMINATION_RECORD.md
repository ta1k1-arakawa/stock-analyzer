# V8K Termination Record

```text
study=V8K_HISTORICAL_RESEARCH
termination_status=TERMINATED_PRE_PRIVATE_PARTITION
termination_reason=RESEARCH_PORTFOLIO_REALLOCATION_TO_HIGHER_EXPECTED_VALUE_ARCHITECTURE
termination_base_sha=2f9e9f9ab1d5982a650408982c809c2a7479b427
future_profitability_established=false
```

## Stage-1 historical provenance

```text
PUBLIC_SOURCE_PREPARATION=PASS
stage1_reviewed_support_sha=7fa38a6f74d631f7e1de37fae16fde944e18c580
stage1_source_raw_sha256=6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653
stage1_eligible_ticker_count=3110
stage1_eligible_ticker_list_sha256=37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405
stage1_t0_reproduction_status=PASS
stage1_first_complete_payload_locked=true
stage1_refetch_allowed=false
```

The existing Stage-1 locked raw payload is historical provenance. It must not
be deleted, reset, refetched, or rewritten. No Stage-2 or T1 authorization is
inferred from any prior Stage-1 authorization.

## Stage-2 archived implementation status

```text
stage2_initial_support_commit=3ebefdae90e534442d9efd5587e5c8cf70420e9b
stage2_high_1_remediation_reviewed_sha=2f9e9f9ab1d5982a650408982c809c2a7479b427
HIGH_1=RESOLVED
HIGH_2=UNRESOLVED_ARCHIVED
HIGH_2_finding=HIGH-2_ONE_SHOT_AUTHORIZATION_REUSED_FOR_DETERMINISTIC_CONTINUATION
HIGH_3=UNRESOLVED_ARCHIVED
HIGH_3_finding=HIGH-3_POST_GATE_BLOCK_CLOSED_STATE_LOST_IN_PUBLIC_FAILURE_OUTPUT
V8K_STAGE2_SUPPORT_OVERALL=BLOCK
```

HIGH-2 and HIGH-3 are archived unresolved findings. This termination record
does not complete either finding or authorize any remediation work.

## Private-boundary status

```text
PRIVATE_PARTITION_GENERATION_AUTHORIZATION_ISSUED=false
PRIVATE_PARTITION_GATE_CONSUMED=false
AUTHORITATIVE_PARTITION_SEED_CREATED=false
PRIVATE_PARTITION_ESTABLISHED=false
T1_MEMBERSHIP_OPENED=false
T1_ACCESS_AUTHORIZED=false
T1_CONSUMED=false
T2_AUTHORIZED=false
T3_AUTHORIZED=false
```

No seed was generated, no partition was generated, and no private membership
was inspected. No backtest, new profit calculation, JPX/Yahoo/network access,
or gate consumption is authorized or performed by this record.

## Scientific interpretation

```text
termination_is_strategy_failure=false
termination_is_profitability_failure=false
V8K_T1_hypothesis_independently_tested=false
V8K_T1_confirmation_result=NOT_RUN
future_profitability_established=false
```

The V8K T1 hypothesis was not independently tested. Candidate 004 is not
claimed to have passed or failed T1.

## Successor boundary

```text
successor_study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
successor_is_new_study_identity=true
```

Research effort is being reallocated because a structurally different
portfolio/execution architecture has higher prospective research value. This
does not establish V9 profitability or a known expected annual return.

V9 must not inherit or reuse V8K Stage-2 authorization, V8K T1 authorization,
a nonexistent V8K partition seed, a nonexistent V8K private partition, or any
unconsumed one-shot gate. V9 may reuse general infrastructure, governance, or
audits only where its own frozen design explicitly reviews and adopts them.
