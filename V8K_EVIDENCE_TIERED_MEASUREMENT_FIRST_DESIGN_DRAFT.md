# V8K Evidence-Tiered, Measurement-First Frozen Design

```text
study=V8K_HISTORICAL_RESEARCH
study_type=METHODOLOGICAL_SUCCESSOR_STUDY
approval_status=APPROVED_FROZEN
future_profitability_established=false
```

This human-approved frozen design records an explicit human methodology
decision. It authorizes no
implementation, public-network access, raw acquisition, private/sealed read,
partition seed, T1/T2/T3 exposure, research opening, production action, or
human-gate consumption. Exact future gate names and authorization grammars
for T1, T2, T3/reserve, membership disclosure, and research opening are
`CHATGPT_DECISION_REQUIRED` unless an inherited frozen rule already fixes
them.

## Human-approved minimal freeze metadata

```text
frozen_design_commit=7ccc7d1b7045e2e7371007327defbb90ae62fe8b
frozen_design_git_blob=f4b83e5e6c0a4f0ca0e38d5eb56cf3ce59363a23
independent_design_review_result=PASS
independent_design_review_critical=0
independent_design_review_high=0
independent_design_review_medium=0
human_design_freeze_gate=HUMAN_V8K_DESIGN_FREEZE
human_design_freeze_complete=true
human_design_freeze_authorization_sha256=19d60f6b56dc07245ccc1326183ee1c53b08764fea85fc692d645ee324130af5
raw_human_authorization_persisted=false
methodology_changed_during_freeze=false
```

The frozen scope is exactly the reviewed methodology below: proportional
strictness by statistical irreversibility; `RETRIABLE_PUBLIC_PLUMBING` and
`STANDING_RETRIABLE_PUBLIC_PLUMBING_AUTHORITY`; strict
`STATISTICALLY_IRREVERSIBLE_GATE`; `DETERMINISTIC_DURABLE_STATE`; the
first-complete-public-payload content lock; no fetch-until-PASS after
semantic/DQ failure; Layer A as
`HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN` with `EVIDENCE_CAPACITY=ZERO`;
its separation from evidence-bearing T1/T2/T3; artifact economy; and the
unchanged evaluator-v2 integrity rules. This freeze grants no JPX/Yahoo
acquisition, private/sealed access, T1/T2/T3 exposure, membership disclosure,
partition seed generation, research opening, broker action, or production
trading.

## 1. V8J historical disposition

```text
predecessor=V8J_HISTORICAL_RESEARCH
V8J_disposition=SUPERSEDED_PRE_EXECUTION_BY_HUMAN_METHODOLOGY_CHANGE
V8J_source_snapshot_gate_consumed=false
V8J_real_JPX_requests=0
V8J_private_reads=0
V8J_strategy_failure=false
V8J_profitability_failure=false
V8J_data_quality_failure=false
V8J_unconsumed_acquisition_authorization_status=REVOKED_UNUSED
```

V8J remains immutable historical evidence. Its frozen design, approval
record, and reviewed implementation are neither rewritten nor deleted. The
previous V8J acquisition authorization was never consumed; this draft does
not persist its raw text, digest, gate, receipt key, or authority, and none
may be reused by V8K.

## 2. Intentional methodology change

V8K is a successor because its governance/failure-domain methodology changes
intentionally; it must not claim that scientific methodology change is
`NONE`.

```text
V8K_scientific_statistical_methodology_changes=
  public_plumbing_is_not_a_one_shot_statistical_gate;
  irreversible_gates_are_limited_to_information_exposure_boundaries;
  public_raw_source_is_first_complete_payload_locked;
  deterministic_durable_state_retries_reuse_persisted_state;
  Layer_A_exploration_is_separate_from_evidence_bearing_validation
```

All evaluator-v2 integrity rules remain unchanged: no future leakage, no
duplicate capital, realistic execution/cost/slippage, deterministic
evaluation, no reference/test feedback into research selection, and no
favorable ticker or period substitution. Existing weak or negative results
remain visible historical evidence. Future profitability is unestablished;
Layer A may conclude `NO_PROMISING_HYPOTHESIS`.

## 3. Evidence tiers and failure classification

### 3.1 Retriable public plumbing

`RETRIABLE_PUBLIC_PLUMBING` includes explicitly public JPX transport,
DNS/TLS/HTTP failure, package/environment setup, parser execution,
filesystem/persistence plumbing, and deterministic processing of already
acquired public raw bytes. A later frozen V8K design may grant standing
authority for this public plumbing. It never authorizes private/sealed data,
broker actions, production trading, or statistical holdout exposure.

### 3.2 Statistically irreversible gates

One-shot/fresh authority remains required at actual information-exposure
boundaries: first T1 use, first sealed T2 use, T3/reserve use where
applicable, membership disclosure revealing a sealed block, private/sealed
outcome access, irreversible research opening, and production trading.
Retrying is prohibited unless frozen methodology specifies deterministic
continuation without additional information exposure.

### 3.3 Deterministic durable state

Partition generation must distinguish creating/persisting the authoritative
32-byte OS-CSPRNG seed from deterministic recomputation using that exact
seed. Persist and bind the seed before allocation. A crash after persistence
requires the exact same seed and identical allocation on rerun; never reroll
until PASS. Deterministic regeneration is not a new statistical exposure.

Failure classifications are distinct:

- `PLUMBING_FAILURE_RETRIABLE`
- `DATA_QUALITY_FAILURE`
- `GOVERNANCE_FAILURE`
- `IMPLEMENTATION_FAILURE`
- `STRATEGY_FAILURE`
- `PROFITABILITY_FAILURE`

An ordinary plumbing failure does not create a new research identity. Create
a successor only when scientific identity/methodology changes or an
irreversible statistical-information boundary requires it.

## 4. Public-source content lock

For V8K public JPX preparation, the canonical provider/endpoint is frozen
before transport. Transport may retry only until the first complete raw
payload exists. Immediately preserve that payload with its SHA-256 and safe
acquisition metadata before semantic inspection.

After that complete payload exists, parser, environment, or software repair
must reprocess the same preserved bytes. It must not re-fetch because parsing,
T0 reproduction, fresh eligible count (`>=900` when that inherited check is
in scope), or another semantic/data-quality check fails. Such a result is
`DATA_QUALITY_FAILURE` and stops the stage; it is not fetch-until-PASS.
Provider/date substitution requires a new explicit methodology decision.

## 5. Layer A: measurement first

```text
LAYER_A_ROLE=HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN
EVIDENCE_CAPACITY=ZERO
```

Layer A uses only already exposed/research/public data. It must not access
T1, T2, T3/reserve, sealed memberships, sealed/private outcomes, or
reference-period information prohibited by evaluator-v2. It may iterate in
one continuing exploration ledger/workspace; a new study ID is not required
for each idea or iteration because it carries zero confirmatory evidence.

Layer A may generate/reject hypotheses, compare candidate families, and
identify a candidate worth preregistering. It may not justify deployment,
satisfy promotion criteria, establish OOS/forward evidence, or establish
future profitability. Before any candidate reaches untouched T1, freeze the
exact candidate, rule, model, features, and search result; no more Layer A
tuning of that candidate is permitted after T1 is seen.

The first post-design priority is measurement, not further source-plumbing
artifact work. Layer A should use the existing evaluator-v2 exposed research
data without waiting for T1/T2/T3 partition construction. Its first output is
one compact viability scorecard labelled
`EXPLORATORY_ONLY / EVIDENCE_CAPACITY_ZERO`, containing where applicable:

- net profit after fees/slippage, maximum drawdown, trade count, profit
  factor, win/loss statistics, and turnover/exposure;
- fold/walk-forward stability and ticker concentration;
- parameter/neighbor robustness; and
- classifier/ranker discrimination (AUC, IC, or Spearman).

No profitability claim or deployment threshold is created here. GPT decides,
after seeing that scorecard, whether any candidate merits Layer B/T1
confirmation.

## 6. Artifact economy and authority

Create durable artifacts only when they protect scientific identity, leakage
boundaries, irreversible statistical exposure, material reproducibility or
provenance, or real production/private authority. Do not create an artifact
merely to prove another artifact exists. Routine public plumbing success may
be compactly logged and retried under its frozen scope. Important
implementation and evidence-bearing frozen designs still require independent
exact-SHA review.

This design does not alter evaluation periods, labels, costs/slippage,
thresholds, source semantics, search spaces, stopping rules, Layer A evidence
capacity, T1/T2/T3 protections, V8J historical disposition, or governance
semantics. It makes no deployment or profitability conclusion. No additional
source-plumbing or freeze artifact is required before the Layer A task because
Layer A uses only already exposed/research data and has
`EVIDENCE_CAPACITY=ZERO`; this does not permit sealed or reference-prohibited
data access.

```text
NEXT_ACTION=V8K_LAYER_A_FIRST_VIABILITY_MEASUREMENT
```
