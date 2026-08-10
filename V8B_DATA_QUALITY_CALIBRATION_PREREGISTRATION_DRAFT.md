# V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT

```text
study=V8B_HISTORICAL_RESEARCH
document_type=DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT
status=DRAFT_AWAITING_HUMAN_APPROVAL
source_v8b_design_draft_commit=a735c4a421628f615596cd2e8de267c3d165df7a
calibration_executed=false
real_network_authorized=false
numeric_policy_selected=false
old_t1_used_for_calibration=false
methodology_decision_authority=CHATGPT
execution_agent_methodology_discretion=false
```

This document is a **preregistration draft**, not an executed calibration
and not a frozen policy. It satisfies `V8B_HISTORICAL_RESEARCH_DESIGN_
DRAFT.md` §6.1's requirement that a future calibration plan freeze its
exact shape before any calibration run. Nothing in this document performs
calibration, accesses any real market-data provider, accesses any sealed
or private V8/V8B block, or authorizes a real network request. It follows
`AI_RESEARCH_EXECUTION_RULES.md`: every methodological field below was
fixed upstream and is encoded here faithfully, not chosen by this
execution agent. See §22 for the explicit list of actions this task does
not perform.

---

## 1. Calibration purpose (fixed)

The sole question this calibration answers:

> What returned-row malformed-OHLCV tolerance is independently defensible
> for V8B acquisition?

This calibration is **not** allowed to optimize:

```text
strategy_returns=PROHIBITED_OPTIMIZATION_TARGET
profit=PROHIBITED_OPTIMIZATION_TARGET
profit_factor=PROHIBITED_OPTIMIZATION_TARGET
sharpe=PROHIBITED_OPTIMIZATION_TARGET
drawdown=PROHIBITED_OPTIMIZATION_TARGET
trade_count=PROHIBITED_OPTIMIZATION_TARGET
model_accuracy=PROHIBITED_OPTIMIZATION_TARGET
candidate_ranking=PROHIBITED_OPTIMIZATION_TARGET
whether_old_t1_would_pass=PROHIBITED_OPTIMIZATION_TARGET
```

---

## 2. Forbidden information (fixed)

Never used as calibration input:

```text
old_t1_raw_payload=FORBIDDEN
old_t1_ticker_identity=FORBIDDEN
old_t1_exact_invalid_fraction=FORBIDDEN
old_t1_failing_year_or_date=FORBIDDEN
old_t1_request_position_as_threshold_evidence=FORBIDDEN
t1b_data=FORBIDDEN
t2_data=FORBIDDEN
t3_data=FORBIDDEN
v7_forward_outcomes=FORBIDDEN
```

`T1` attempt #1 and attempt #2 may be mentioned only as historical
provenance (exactly as `V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §0.1/§0.2/§6
already do). They must not affect:

```text
grid=UNAFFECTED_BY_OLD_T1_OUTCOME
criteria=UNAFFECTED_BY_OLD_T1_OUTCOME
selection=UNAFFECTED_BY_OLD_T1_OUTCOME
tie_break=UNAFFECTED_BY_OLD_T1_OUTCOME
fallback=UNAFFECTED_BY_OLD_T1_OUTCOME
```

---

## 3. Observed calibration dataset (fixed)

```text
observed_calibration_source=V5-B evaluation cache
declared_local_path=C:\taiki\hobbies\v5-b-evaluation-cache-retry1 (local Windows path; outside this repository; not committed; not inspected by this task)
```

Committed provenance, verified against `V8_DATA_EXPOSURE_AUDIT.md` (this
task's own read-only fact-check, not restated from the upstream prompt
without confirmation):

```text
declared_span=2019-01-04 through 2026-01-30
successful_tickers=300
failed_tickers=0
manifest_hash=797265BF671AF2245A342051FFAD02AA2929D67BA885945E7762149649148AA5
payload_list_hash=a45ce89a7fa8be689e7d0affe34de56152552d7a3414935f0a364843cd3121f8
verified_against=V8_DATA_EXPOSURE_AUDIT.md (row recording the V5-B evaluation cache)
verification_result=MATCH (declared span, ticker counts, and both hashes match the committed audit record exactly)
```

For V8B calibration use only:

```text
calibration_window_in_use=2020-01-01 through 2025-12-31
```

Reason: this is already-burned development/evaluation material and is
outside the fresh validation/holdout blocks (`T1B`, `T2`, `T3`). Strategy
outcomes associated with this cache are not used by this calibration.

**Additional fact-check note (not a contradiction, a disclosed
uncertainty already on record).** `V8_DATA_EXPOSURE_AUDIT.md`'s own
unresolved-question table records that the exact row-level contents of
the V5-B evaluation cache are a local, uncommitted path and are *trusted
by declared span*, not independently re-verified row-by-row in that
audit document. This does not contradict anything fixed in this
preregistration; it is exactly the kind of gap §3.1 below's availability
check exists to resolve before calibration execution, not something this
task resolves or works around.

### 3.1 Required future availability check (fixed)

Calibration execution may proceed only if the cache can reproduce the
canonical returned-row validity classification needed by
`src/v7_yahoo_collector.py::_row_invalid_reason` (§4 below).

This requires either:

```text
A. preserved_raw_yahoo_payloads=SUFFICIENT
B. equivalent_persisted_representation_retaining_original_returned_observations_including_would-be-invalid_rows=SUFFICIENT
```

A sanitized dataset containing only already-valid rows is **not**
sufficient for Component A (§6).

If this cannot be proven:

```text
OBSERVED_COMPONENT_STATUS=BLOCKED_INPUT_NOT_REPRODUCIBLE
```

and calibration execution **must stop**. Do not substitute another
dataset. Return to the research planner (ChatGPT) for a new decision.

**This preregistration-drafting task itself does not inspect the local
cache** — no verification of §3.1's availability condition has been
performed, and none is claimed here. That verification is future work,
to occur only at actual calibration-execution time, not at drafting time.

---

## 4. Canonical malformed-row classifier (fixed, independently verified against source)

Use exactly the existing canonical reason classes from
`src/v7_yahoo_collector.py::_row_invalid_reason`.

**Verification performed by this task:** the function was read directly
(`src/v7_yahoo_collector.py:155-167`). It iterates the fields `open`,
`high`, `low`, `close`, `adjclose` — for each, a non-finite value yields
`NONFINITE_<FIELD>` and a non-positive value yields `NONPOSITIVE_<FIELD>`
— then checks `volume`, where a non-finite value yields `NONFINITE_VOLUME`
and a negative value yields `NEGATIVE_VOLUME` (there is no
`NONPOSITIVE_VOLUME` class; zero volume is not itself invalid under this
classifier). The resulting twelve classes match the upstream
specification exactly:

```text
NONFINITE_OPEN
NONPOSITIVE_OPEN
NONFINITE_HIGH
NONPOSITIVE_HIGH
NONFINITE_LOW
NONPOSITIVE_LOW
NONFINITE_CLOSE
NONPOSITIVE_CLOSE
NONFINITE_ADJCLOSE
NONPOSITIVE_ADJCLOSE
NONFINITE_VOLUME
NEGATIVE_VOLUME
```

No corruption classes unrelated to this row-level policy are added.

**Schema-level hard failures — independently verified as distinct.**
`DUPLICATE_TRADING_DATE` (`src/v7_yahoo_collector.py:281`),
`ARRAY_LENGTH_MISMATCH` (`:179`), and `TIMESTAMP_INVALID` (`:134`) are
each raised as a `V7YahooCollectorBlocked` exception — a hard
parser/schema failure that aborts the whole fetch — and are structurally
separate from `_row_invalid_reason`'s per-row classification, which marks
a single row invalid without aborting the fetch. This calibration does
**not** convert any schema-level hard failure into a tolerated row-level
malformed observation.

---

## 5. Exact candidate grid (fixed by the research planner)

```text
F1  = 1/252
F2  = 2/252
FQ1 = 1/100   (Q1 control)
F3  = 3/252
F4  = 4/252
F5  = 5/252
```

**Canonical numerical ordering, verified by this task via exact rational
comparison:** `1/252 (≈0.0039683) < 2/252 (≈0.0079365) < 1/100 (0.01) <
3/252 (≈0.0119048) < 4/252 (≈0.0158730) < 5/252 (≈0.0198413)`. The
upstream-declared ordering (`1/252, 2/252, 1/100, 3/252, 4/252, 5/252`)
is arithmetically correct and is reproduced unchanged.

```text
Q1_control_fraction=1/100
Q1_control_max_consecutive=5
consecutive_candidates={1, 2, 3, 4, 5}
```

Evaluate the complete Cartesian product: **6 fraction values × 5
consecutive values = 30 candidates.** No other candidate may be added. No
candidate may be removed after calibration begins. Exact
integer/rational comparison semantics are used throughout; policy
decisions never rely on floating-point equality.

**Independent rationale (as supplied, and independently fact-checked by
this task).** `V8_HISTORICAL_RESEARCH_DESIGN.md` genuinely establishes a
pre-existing "latest 252 valid observations per ticker" convention
(`V8_HISTORICAL_RESEARCH_DESIGN.md:320`, §3.2, cited again at §17's own
threshold-derivation account at line 1274 as one of two independent bases
used to originally derive `POLICY_G_PRIME_V1`'s 1%/5 thresholds without
consulting old `T1`'s unknown failure). `1..5 / 252` therefore correspond
to one through five returned observations per standard 252-observation
year; five observations correspond approximately to one standard trading
week. `1/100` is included solely because it is the frozen Q1 control.
This grid was **not** derived from the unknown old-`T1` failure
magnitude — confirmed by this task's own reading of §17, which states
that its threshold review "derived candidate numeric thresholds from
constants already frozen elsewhere in this document," blind to attempt
#1's unpersisted exact reason and ticker.

```text
candidate_grid_size=30
grid_derived_from_old_t1_failure=false
```

---

## 6. Component A — observed burned-data characterization (fixed)

For every ticker in the verified V5-B cache, and for each applicable
year — `2020, 2021, 2022, 2023, 2024, 2025` — and for the full
calibration span `2020-01-01 through 2025-12-31`, compute only
returned-row data-quality quantities:

```text
total_returned = valid_returned + invalid_returned
invalid_fraction = invalid_returned / total_returned
max_consecutive_invalid_returned_rows = maximum run length in chronological returned-observation order
```

Expected calendar missing dates are **not** invalid returned rows. A
zero-returned-observation window is `NOT_APPLICABLE` for an individual
year. A ticker with zero returned observations over the complete
calibration span is an observed-data calibration failure and blocks the
calibration artifact. **No strategy metric may be calculated.**

---

## 7. Component B — synthetic parser/policy robustness (fixed, mandatory)

Use clean base observations only from the allowed burned V5-B material. A
clean base ticker-year is eligible only if the canonical classifier (§4)
reports zero malformed returned rows before injection.

**Deterministic base-sequence selection:**

```text
selection_order=canonical ticker order ascending
selection_count=first 20 eligible ticker-years
required_length_per_base_sequence=252 returned observations
```

If fewer than 20 such base sequences exist:

```text
SYNTHETIC_BASE_SELECTION_BLOCKED
```

and calibration stops. No substitute selection.

For every selected base sequence, inject each of the 12 row-invalid
reason classes (§4) separately. Deterministic corrupt values:

```text
NONFINITE_*=null/None in the corresponding Yahoo payload field
NONPOSITIVE_*_price=0.0
NEGATIVE_VOLUME=-1
```

Different reason classes are never combined in one synthetic scenario.

---

## 8. Synthetic counts / placements (fixed)

For each corruption class and each 252-row base sequence, evaluate total
invalid-row counts:

```text
K = {0, 1, 2, 3, 4, 5, 6}
```

For `K > 0`, evaluate deterministic placement families:

```text
A. ISOLATED_EVENLY_SPACED  -- K corrupted observations placed as evenly as possible across the 252-row sequence, with no adjacency when mathematically possible
B. CONSECUTIVE_RUN         -- K corrupted observations placed consecutively, centered around the sequence midpoint
C. START_RUN                -- K corrupted observations placed consecutively starting at index 0
D. END_RUN                  -- K corrupted observations placed consecutively ending at index 251
```

```text
synthetic_random_seed=NOT_USED
adaptive_new_scenarios_after_results_observed=false
```

---

## 9. Synthetic hard expectations (fixed)

Every candidate considered `DEFENSIBLE` (§11) must satisfy:

```text
A. uncorrupted K=0 scenarios pass
B. every K=1 scenario passes
C. every K=6 scenario fails by at least one frozen quality guard (fraction guard OR consecutive-run guard, depending on placement)
D. every 6-consecutive-row scenario fails the consecutive guard specifically, because every candidate's max-consecutive value is <=5
E. no malformed injected row may silently become valid
F. every non-corrupted row remains canonical and unchanged
```

---

## 10. Observed-data defensibility (fixed)

A candidate is **not** defensible if it rejects any applicable
ticker-year or full-series window in the allowed burned V5-B calibration
material. Therefore:

```text
observed_ticker_year_pass_rate_required=100%
observed_full_ticker_pass_rate_required=100%
```

This deliberately uses burned development/evaluation data as the
calibration set. It does **not** imply future `T1B`/`T2` must have zero
malformed rows. It means the chosen tolerance must be at least capable of
handling the already-observed source-quality envelope in the designated
calibration material, while staying within the predeclared upper grid
ceiling (§5). If no candidate in the fixed grid achieves this:

```text
NO_DEFENSIBLE_POLICY
```

The grid is **not** expanded in response.

---

## 11. `DEFENSIBLE(candidate)` — exact definition (fixed)

```text
DEFENSIBLE(candidate) = true
```

if and only if **all** of the following hold:

```text
1.  observed input provenance validates exactly
2.  canonical invalid-row classification is reproducible
3.  100% of applicable observed ticker-year windows pass
4.  100% of observed full-ticker windows pass
5.  all synthetic K=0 cases pass
6.  all synthetic K=1 cases pass
7.  all synthetic K=6 cases fail
8.  all 6-consecutive cases fail the consecutive guard
9.  parser/classifier behavior matches the canonical reason taxonomy (§4)
10. no schema-level hard failure is converted into a tolerated row-level quality event
```

Otherwise `DEFENSIBLE(candidate) = false`. No qualitative override
exists.

---

## 12. Selection rule (fixed)

Evaluate all 30 candidates. Collect the complete set:

```text
D = {candidate | DEFENSIBLE(candidate) = true}
```

If `D` is non-empty: choose the **strictest** candidate. `STRICTEST` is
defined lexicographically:

```text
1. smaller invalid_fraction_threshold is stricter
2. if fraction thresholds are identical, smaller max_consecutive_invalid_returned_rows is stricter
```

Exact rational comparison is used. No human choice is made after seeing
results.

---

## 13. Tie break (fixed)

The strictness order (§12) creates a total order over the 30 unique
candidates (all six fractions are distinct rational values, per §5's
verified ordering, so no substantive tie is expected). If canonical
representation unexpectedly produces duplicate-equivalent candidates:

```text
tie_break=choose the candidate with lexicographically smallest canonical candidate ID
duplicate_equivalence_reported_as=LOW review finding
```

A human is never asked to choose after seeing results.

---

## 14. Stopping rule (fixed)

The calibration run completes only after:

```text
all_verified_observed_windows_evaluated=REQUIRED
all_30_candidates_evaluated=REQUIRED
all_preregistered_synthetic_scenarios_evaluated=REQUIRED
all_per_candidate_results_persisted=REQUIRED
```

```text
early_stopping=false
second_grid=false
rerun_with_changed_criteria=false
candidate_addition_after_start=false
threshold_shopping=false
```

---

## 15. Fallback (fixed)

If `D` (§12) is empty:

```text
selected_policy=CALIBRATION_NO_DEFENSIBLE_POLICY
action=BLOCK_V8B_DESIGN_FINALIZATION
```

`Q1` is **not** automatically retained merely because it was the
historical baseline. Specifically:

```text
if Q1 fails DEFENSIBLE: Q1 is not retained by default
if Q1 is DEFENSIBLE but a stricter candidate is also DEFENSIBLE: the stricter candidate is selected per §12
if Q1 is the strictest DEFENSIBLE candidate: Q1 may be retained
```

---

## 16. Full result reporting (fixed)

Future calibration output must include all 30 candidates. For every
candidate, report at minimum:

```text
candidate_id
exact_fraction_rational
max_consecutive
observed_ticker_year_pass_count_over_denominator
observed_full_ticker_pass_count_over_denominator
synthetic_pass_fail_counts_by_corruption_class
synthetic_pass_fail_counts_by_K
synthetic_pass_fail_counts_by_placement_family
DEFENSIBLE_true_or_false
failed_criterion_ids
```

Also record:

```text
input_provenance_hashes
calibration_plan_hash
implementation_commit
run_timestamps
error_counts
artifact_self_hash
mechanically_selected_candidate_or_NO_DEFENSIBLE_POLICY
```

```text
best_only_output=PROHIBITED
```

---

## 17. Result review gate (fixed)

`CALIBRATION_RESULT_REVIEW` (`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md`
§12) must independently verify:

```text
exact_preregistration_commit_or_hash=VERIFY
exact_input_provenance=VERIFY
no_old_t1_t1b_t2_t3_inputs=VERIFY
no_grid_changes=VERIFY
all_30_candidates_executed=VERIFY
synthetic_scenario_set_unchanged=VERIFY
no_adaptive_reruns=VERIFY
selection_applied_mechanically=VERIFY
all_hashes_and_self_hashes_validate=VERIFY
```

Only after an independent `PASS` may the selected policy be proposed for
`V8B_DESIGN_FINALIZED` (`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §12).
**This document does not authorize that.**

---

## 18. Relationship to the V8B design draft's own calibration wall

This preregistration satisfies, but does not replace,
`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §6.1's requirement that a future
calibration plan freeze `calibration_plan_version`, `allowed_data_sources`,
`exact_included_calibration_datasets`, `exact_exclusions`, and the other
listed fields before approval. Cross-mapping:

```text
allowed_data_sources -> V5-B evaluation cache, 2020-01-01..2025-12-31 (basis A/D per §6's A-D categories: already-burned development/evaluation material outside T1B/T2/T3), plus mandatory synthetic corruption (basis B)
exact_included_calibration_datasets -> the verified V5-B cache, restricted to the 2020-01-01..2025-12-31 window (§3)
exact_exclusions -> old T1, T1B, T2, T3, V7 forward outcomes (§2)
synthetic_corruption_generation_procedure -> §7-§9
synthetic_random_seed_or_seeds -> NOT_USED (§8)
unit_of_analysis -> per-ticker-year window and full calibration-span window (§6)
evaluation_windows -> 2020..2025 individually, and 2020-01-01..2025-12-31 in full (§6)
malformed_row_classifier_and_version -> src/v7_yahoo_collector.py::_row_invalid_reason, verified against source (§4)
exact_finite_candidate_set_invalid_fraction_threshold -> {1/252, 2/252, 1/100, 3/252, 4/252, 5/252} (§5)
exact_finite_candidate_set_max_consecutive_invalid_returned_rows -> {1, 2, 3, 4, 5} (§5)
exact_metrics_computed_per_candidate -> §16
exact_aggregation_method_per_ticker -> §6 (total_returned, invalid_fraction, max_consecutive per ticker/window)
exact_aggregation_method_per_window_or_year -> §6 (per named year and full span, independently)
exact_aggregation_method_across_calibration_material -> §10 (100% pass-rate requirement across all applicable windows)
exact_defensibility_criterion -> §11
exact_deterministic_candidate_selection_rule -> §12
exact_tie_break_rule -> §13
exact_stopping_rule -> §14
exact_fallback_rule -> §15
exact_missing_or_error_handling -> §3.1 (OBSERVED_COMPONENT_STATUS=BLOCKED_INPUT_NOT_REPRODUCIBLE on unprovable input; §7's SYNTHETIC_BASE_SELECTION_BLOCKED on insufficient clean base sequences)
full_candidate_grid_results_retention -> §16 (MANDATORY, all 30 candidates)
best_only_reporting -> PROHIBITED (§16)
old_t1_input / t1b_input / t2_input / t3_input -> PROHIBITED (§2)
```

---

## 19. Current task does not execute calibration (fixed)

This task did **not**, and does not authorize any future task to
silently:

```text
inspect_local_cache_path=false
run_parser_against_cache=false
generate_synthetic_payloads=false
calculate_any_metrics=false
choose_a_winner=false
edit_src=false
edit_scripts=false
edit_tests=false
access_private_partition=false
access_yahoo=false
access_jpx=false
freeze_v8b=false
```

This task writes documentation only.

---

## 20. Status

```text
status=DRAFT_AWAITING_HUMAN_APPROVAL
```

Next required action after this draft: independent review of this
preregistration by the research planner (ChatGPT), followed by a separate
human approval gate (`DATA_QUALITY_CALIBRATION_PLAN_APPROVED`,
`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §12), before any calibration
execution may begin.
