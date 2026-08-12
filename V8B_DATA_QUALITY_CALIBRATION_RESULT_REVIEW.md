# V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW

```text
study=V8B_HISTORICAL_RESEARCH
document_type=CALIBRATION_RESULT_REVIEW_RECORD
gate=CALIBRATION_RESULT_REVIEW (V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md §26; V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md §12)
result=V8B_CALIBRATION_RESULT_REVIEW_PASS
approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1
approved_plan_git_commit=8c15426166742c43745e604f6367788af6123c1a
reviewed_attempt_id=V8B_CALIBRATION_REAL_ATTEMPT_2
reviewed_artifact_self_hash=2aec28397e9bb01b0333df3c077a0f6b8fd497b68f20fb983ea95c1e47560426
this_review_calibration_executed=false
this_review_v5b_cache_accessed=false
this_review_raw_json_ohlcv_read=false
this_review_network_accessed=false
this_review_methodology_changed=false
```

This is the formal `CALIBRATION_RESULT_REVIEW` gate
(`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §26;
`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §6/§12), performed against the
already-recorded audit facts in
`V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md`. This review did not
re-execute calibration, did not access the V5-B cache, and did not read
raw JSON/OHLCV — it verifies the already-established adjudication record
against §26's checklist and §13's run-validity gate.

---

## §26 checklist verification

Per `V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §26, this
review first confirms `CALIBRATION_RUN_VALID=true` for the attempt being
reviewed, then independently verifies each item below.

```text
CALIBRATION_RUN_VALID=true (attempt V8B_CALIBRATION_REAL_ATTEMPT_2)
plan_version_correct=VERIFIED (V8B_DATA_QUALITY_CALIBRATION_PLAN_V1)
plan_commit_or_hash_correct=VERIFIED (8c15426166742c43745e604f6367788af6123c1a)
attempt_history_complete=VERIFIED (attempt 1 and attempt 2 both recorded in V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md)
previous_invalid_attempts_retained=VERIFIED (attempt 1, CALIBRATION_CLASSIFIER_VERSION_MISMATCH, retained as audit history, not overwritten)
no_methodology_changed_between_invalid_attempt_and_valid_rerun=VERIFIED (attempt 1's fix was tzdata==2026.3, a technical/environment remediation only -- ZoneInfo("Asia/Tokyo") support; no grid, window, classifier, defensibility, selection, or tie-break rule changed; pinned collector blob 76b57b077f3214e666ff9dc06d9c224afc16df9f unchanged before and after)
implementation_fix_independently_reviewed_if_any=NOT_APPLICABLE (reason: no implementation change occurred between attempts; both attempt 1 and attempt 2 ran implementation_commit ceffa3b98a52cb46d44107539d5a0a59da72cd0a unchanged; the remediation was execution-environment-only -- tzdata==2026.3 / ZoneInfo("Asia/Tokyo") -- diagnostically verified before attempt 2, per V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md; §22's "implementation fix independently reviewed" clause presupposes an implementation fix, which did not occur here, so it does not apply)
exact_preregistration_commit_or_hash=VERIFIED (approved_plan_git_commit=8c15426166742c43745e604f6367788af6123c1a, approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1)
exact_input_provenance=VERIFIED (this review independently compared the persisted V8B_CALIBRATION_REAL_ATTEMPT_2 artifact's recorded provenance metadata against the frozen approved pins: manifest_sha256=797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5, payload_hash_list_sha256=a45ce89a7fa8be689e7d0affe34de56152552d7a3414935f0a364843cd3121f8, manifest_payload_count=300, bound_payload_count=300 -- all four values match the frozen calibration implementation's approved provenance contract exactly. This review compared already-persisted artifact metadata against already-frozen pins; it did not reread or re-derive the raw V5-B cache)
no_old_t1_t1b_t2_t3_inputs=VERIFIED (calibration inputs are V5-B evaluation cache + synthetic corruption only, per the approved plan §2/§6; no old T1/T1B/T2/T3 material used)
no_grid_changes=VERIFIED (candidate_count=30, unchanged from the frozen 6x5 grid)
all_30_candidates_executed=VERIFIED (candidate_count=30, all 30 DEFENSIBLE)
synthetic_scenario_set_unchanged=VERIFIED (synthetic_base_count=20, synthetic_scenario_count=6000, synthetic_candidate_comparison_count=180000 -- exactly the preregistered §10/§20 counts)
no_adaptive_reruns=VERIFIED (exactly two attempts: one CALIBRATION_RUN_INVALID technical failure, one CALIBRATION_RUN_VALID result; no grid/criteria change between them)
selection_applied_mechanically=VERIFIED (selected_policy=F1_C1 follows §18's STRICTEST_DEFENSIBLE rule mechanically from the DEFENSIBLE set; no human choice after results)
all_hashes_and_self_hashes_validate=VERIFIED (attempt 1: persisted INVALID artifact semantic validation=PASS; attempt 2: persisted artifact self-hash/integrity validation=PASS, including trusted commit/attempt-id binding -- both per V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md; see "Persisted-artifact validation precision" below for the exact scope of each check)
M_fraction_and_M_consecutive_correctly_computed_as_true_maxima=VERIFIED (M_fraction=0/1 over 2352 windows, M_consecutive=0 over 2352 windows; both accepted as official statistics only because the run is CALIBRATION_RUN_VALID=true, per §14's authority gate)
strict_headroom_rule_correctly_applied_no_boundary_equality_admitted=VERIFIED (selected candidate F1_C1's fraction threshold 1/252 and max_consecutive 1 both strictly exceed the observed maxima of 0/1 and 0 respectively -- selected fraction headroom=1/252, selected consecutive headroom=1; all 30 candidates independently satisfied the same strict inequality, so all 30 are DEFENSIBLE)
Q1_received_no_exemption_from_DEFENSIBLE_criteria=VERIFIED (Q1's own grid point, FQ1=1/100 with max_consecutive=5, was evaluated under the identical D1/D2 predicate as every other candidate; F1_C1 was independently stricter and DEFENSIBLE, so §18 selected it over Q1 mechanically -- Q1 was not selected, and was not exempted from the criteria that would have selected it had no stricter candidate qualified)
synthetic_expected_vs_observed_truth_table_fully_matched_or_mismatches_correctly_marked_the_run_invalid=VERIFIED (synthetic truth-table mismatch count=0 across all 180000 candidate/scenario comparisons)
```

---

## Persisted-artifact validation precision

This section states precisely what "validated" means for each attempt's
persisted artifact, so that self-hash/integrity validation is never read
as a claim of full source-bound semantic re-validation.

```text
persisted_invalid_artifact_semantic_validation_attempt_1=PASS (V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md; semantic validation of the INVALID-result artifact schema)
persisted_artifact_self_hash_integrity_validation_attempt_2=PASS (V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md; self-hash/integrity validation plus trusted commit/attempt-id binding, performed on the already-persisted artifact)
production_core_pre_persistence_semantic_validation_attempt_2=PERFORMED_INTERNALLY (the reviewed production core, run_data_quality_calibration(), validates the VALID result state -- observed windows, all 30 candidates, synthetic bases/scenarios, and manifest/provenance -- against §13's R0-R9 invariants before constructing and persisting the artifact)
FULL_SOURCE_BOUND_SEMANTIC_VALIDATION_POST_PERSISTENCE=NOT_REEXECUTED (this review, and the adjudication record it relies on, did not independently re-execute a second full source-bound semantic validation pass against the persisted attempt-2 artifact after the production core's own pre-persistence validation; the post-persistence check performed was self-hash/integrity and trusted commit/attempt-id binding only)
FULL_SOURCE_BOUND_SEMANTIC_VALIDATION_POST_PERSISTENCE_IS_A_FAILURE=false (this reflects the actual verification design -- the production core validates once, pre-persistence, against source; post-persistence checks confirm the artifact was not altered afterward -- not a gap or defect)
```

---

## Additional confirmations

```text
observed_yearly_window_denominator=2052
observed_yearly_window_pass_counts=2052/2052 for all 30 candidates
observed_full_ticker_denominator=300
observed_full_ticker_pass_counts=300/300 for all 30 candidates
candidate_selection_executed=true
selected_policy=F1_C1
selected_fraction_threshold=1/252
selected_max_consecutive=1
```

## Conclusion

```text
V8B_CALIBRATION_RESULT_REVIEW_PASS
```

The reviewed calibration result (`V8B_CALIBRATION_REAL_ATTEMPT_2`) is a
valid, independently reviewed calibration outcome. Candidate `F1_C1`
(`invalid_fraction_threshold=1/252`,
`max_consecutive_invalid_returned_rows=1`) **may now be proposed** for
`V8B_DESIGN_FINALIZED`
(`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §12, §7.4).

## Scope limits of this review

```text
this_review_authorizes_t1b_allocation=false
this_review_authorizes_acquisition=false
this_review_authorizes_research_opening=false
this_review_authorizes_real_network_access=false
this_review_authorizes_v8b_design_freeze=false
this_review_authorizes_human_design_freeze=false
```

This document records the completed `CALIBRATION_RESULT_REVIEW` gate only.
It does not itself finalize the V8B design, freeze it, or authorize any
`T1B`, `T2`, or `T3` access.

---

## Status

```text
status=RECORDED
next_action=REPEAT_INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_DRAFT
```
