# V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION

```text
study=V8B_HISTORICAL_RESEARCH
document_type=DATA_QUALITY_CALIBRATION_ADJUDICATION_RECORD
status=RECORDED
approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1
approved_plan_git_commit=8c15426166742c43745e604f6367788af6123c1a
implementation_commit=ceffa3b98a52cb46d44107539d5a0a59da72cd0a
this_task_calibration_executed=false
this_task_v5b_cache_accessed=false
this_task_raw_json_ohlcv_read=false
this_task_yahoo_jpx_or_other_network_accessed=false
this_task_methodology_or_core_implementation_modified=false
this_task_t1b_t2_t3_v7_forward_data_opened=false
```

This document is a repository-level audit/adjudication record of two prior
V8B data-quality calibration attempts, both already executed under the
already-approved `V8B_DATA_QUALITY_CALIBRATION_PLAN_V1`
(`V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json`). This task did **not**
re-run calibration, did **not** access the V5-B cache, did **not** read raw
JSON/OHLCV, and did **not** perform any real network access other than
`git push`. It records the two attempts' already-established outcomes only,
per `AI_RESEARCH_EXECUTION_RULES.md` §4 (fact-finding, not methodology
discretion).

---

## Attempt 1

```text
attempt_id=V8B_CALIBRATION_REAL_ATTEMPT_1
implementation_commit=ceffa3b98a52cb46d44107539d5a0a59da72cd0a
result=INVALID (technical environment failure)
run_invalid_reason=CALIBRATION_CLASSIFIER_VERSION_MISMATCH
artifact_self_hash=5d1519e404cd2df9690c3e62e0d078093e3405d75080bc163a2aa5ad00fb6483
persisted_invalid_artifact_semantic_validation=PASS
```

**Diagnostic established:** the pinned collector Git blob matched
`76b57b077f3214e666ff9dc06d9c224afc16df9f`
(`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §4's exact pin),
so the classifier itself was not out of date.

**Actual cause:** the Windows Python `ZoneInfo` installation lacked
`Asia/Tokyo` timezone data in the execution environment.

**Technical remediation only:** `tzdata==2026.3` was installed. No
methodology, grid, window, classifier, core implementation, or execution
adapter was changed.

**After remediation:**

```text
ZoneInfo("Asia/Tokyo")=PASS
pinned_collector_execution=PASS
classifier_blob=UNCHANGED
```

Per `AI_RESEARCH_EXECUTION_RULES.md` §7's preregistration §13.2/§22 retry
semantics, this was a permitted same-plan technical/conformance-only fix,
not a methodology change. Attempt 1 remains retained as audit history and
was **not** overwritten.

---

## Attempt 2

```text
attempt_id=V8B_CALIBRATION_REAL_ATTEMPT_2
implementation_commit=ceffa3b98a52cb46d44107539d5a0a59da72cd0a
result=VALID
artifact_self_hash=2aec28397e9bb01b0333df3c077a0f6b8fd497b68f20fb983ea95c1e47560426
persisted_artifact_integrity_validation=PASS
calibration_run_valid=true
run_invalid_reason=null
candidate_selection_executed=true
```

### Observed quality envelope (Component A)

```text
M_fraction=0/1
M_fraction_source_window_count=2352
M_consecutive=0
M_consecutive_source_window_count=2352
```

### Candidate evaluation

```text
candidate_count=30
all_30_candidates_DEFENSIBLE=true
selected_policy=F1_C1
selected_fraction_threshold=1/252
selected_max_consecutive=1
selected_fraction_headroom=1/252
selected_consecutive_headroom=1
```

```text
yearly_window_denominator=2052
all_candidate_yearly_pass_counts=2052/2052
full_ticker_denominator=300
all_candidate_full_ticker_pass_counts=300/300
```

### Synthetic component (Component B)

```text
synthetic_base_count=20
synthetic_base_ticker_count=20
synthetic_scenario_count=6000
synthetic_candidate_comparison_count=180000
synthetic_truth_table_mismatch_count=0
```

---

## Adjudication

```text
V8B_DATA_QUALITY_CALIBRATION_PASS
```

**Frozen selected policy for the next design step:**

```text
F1_C1 = fraction threshold 1/252 and max consecutive invalid returned rows 1
```

**Important interpretation.** The independent calibration envelope observed
zero malformed returned rows across the 2352 applicable yearly/full-span
windows, so the preregistered `STRICTEST_DEFENSIBLE` rule
(`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §18) mechanically
selected the strictest candidate, `F1_C1`. This must not be described as
tuning to improve strategy or profit — the preregistration explicitly
prohibits optimizing strategy returns, profit, profit factor, Sharpe,
drawdown, trade count, model accuracy, or candidate ranking (§1).

---

## Explicit scope limits of this adjudication

```text
calibration_retry_required=false
further_v5b_access_required_for_this_adjudication=false
t1b_t2_t3_authorization_implied=false
design_freeze_implied=false
research_or_validation_opening_implied=false
```

This record does not constitute `CALIBRATION_RESULT_REVIEW`
(`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §26) and does not
itself authorize `V8B_DESIGN_FINALIZED`. It is an audit/adjudication record
of the already-executed attempts, pending the independent review named
below.

---

## Privacy

No raw ticker identities, raw paths, raw payload content, OHLCV data, old
T1 identities/details beyond already-approved aggregate audit facts, or
T1B/T2/T3 identities are recorded in this document. Hashed synthetic-base
ticker metadata is not reproduced here.

---

## Task-execution confirmation (this adjudication-recording task)

```text
production_calibration_executions_by_this_task=0
real_v5b_cache_accesses_by_this_task=0
yahoo_jpx_or_other_network_accesses_by_this_task=0 (except git push)
production_code_modified_by_this_task=false
```

---

## Status

```text
status=RECORDED
next_action=INDEPENDENT_REVIEW_OF_CALIBRATION_ADJUDICATION
```
