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
execution agent. See §24 for the explicit list of actions this task does
not perform.

This revision corrects and strengthens the observed-window definition, the
synthetic-base definition, the placement formulas, the expected-vs-observed
policy semantics, and the `DEFENSIBLE(candidate)` criterion of the prior
draft. Nothing in §1 (purpose) or §2 (forbidden information) changes.

---

## 1. Calibration purpose (fixed, unchanged)

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

## 2. Forbidden information (fixed, unchanged)

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

## 3. Observed calibration dataset (window corrected)

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

**Corrected calibration window (binding):**

```text
calibration_window_in_use=2019-01-01 <= returned JST trading_date < 2026-01-01
calendar_years_evaluated_individually=2019, 2020, 2021, 2022, 2023, 2024, 2025
full_calibration_span=2019-01-01 through 2025-12-31
```

This replaces the prior draft's `2020-01-01..2025-12-31` window. Reason:
the V5-B acquisition requested from `2019-01-01` and preserves raw Yahoo
payloads (§3.2 below), so `2019` is legitimately includable observed
material — this is already-burned development/evaluation material, is
outside the fresh validation/holdout blocks (`T1B`, `T2`, `T3`), and
strategy outcomes associated with this cache are not used by this
calibration.

Absent calendar dates are **not** treated as malformed. No calendar year
is required to contain 252 observations; §3's window definition governs
observed windows only — the 252-observation quantity in §5's grid
rationale and §7's synthetic base length is a separate, unrelated design
constant, not a per-year requirement on observed data.

**Fact-check note carried forward (not a contradiction, a disclosed
uncertainty already on record).** `V8_DATA_EXPOSURE_AUDIT.md`'s own
unresolved-question table records that the exact row-level contents of
the V5-B evaluation cache are a local, uncommitted path and are *trusted
by declared span*, not independently re-verified row-by-row in that audit
document. This does not contradict anything fixed in this preregistration;
it is exactly the kind of gap §3.1/§3.2 below exist to resolve before
calibration execution, not something this task resolves or works around.

### 3.1 V5-B raw payload fact (independently verified against source)

**Verification performed by this task:** `scripts/acquire_v5_b_evaluation_
cache.py` was read directly. Its request window constant is
`START="2019-01-01"` (`:15`), confirming the V5-B acquisition requested
from `2019-01-01`. For each successful ticker, the exact unmodified Yahoo
response body is written to `raw/<ticker>.json` inside the cache
(`:86`, `(stage/rel).write_bytes(body)` where `rel = Path("raw")/(ticker
+ ".json")`) — the cache is therefore, by construction, a raw-payload
cache, not a pre-sanitized/valid-rows-only cache.

```text
v5b_raw_payload_write_verified=true (scripts/acquire_v5_b_evaluation_cache.py:86)
v5b_request_start_verified=2019-01-01 (scripts/acquire_v5_b_evaluation_cache.py:15)
```

### 3.2 Required future preflight (fixed, renamed and strengthened)

Component A (§6) may proceed in a future task **only if** a read-only
preflight, performed at that future calibration-execution time — not by
this task — proves that the existing local cache
(`C:\taiki\hobbies\v5-b-evaluation-cache-retry1`) still exists and matches
the committed provenance/hash expectations in §3 above, and that its raw
payloads can be parsed under the current canonical classification
semantics (§3.3 below).

If that preflight cannot establish this:

```text
V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED
```

Then **STOP**. Do not substitute another dataset.

**This preregistration-drafting task itself does not perform that
preflight and does not inspect the local cache** — no verification of
this section's availability condition has been performed, and none is
claimed here. That verification is future work, to occur only at actual
calibration-execution time, not at drafting time.

### 3.3 Canonical input parse requirement (fixed)

Before future calibration metrics are calculated, every designated V5-B
raw payload must be reproducible under the current canonical
row-classification semantics (§4). The execution implementation must
reconstruct returned-observation valid/invalid status using the current
canonical classifier, from the raw payload bytes, not from any
pre-derived summary.

If a designated payload produces a current canonical schema-level hard
failure (§4's `DUPLICATE_TRADING_DATE` / `ARRAY_LENGTH_MISMATCH` /
`TIMESTAMP_INVALID` class, or any other hard parser failure) such that
row-level classification cannot be reproduced for that ticker:

```text
CALIBRATION_INPUT_CANONICAL_PARSE_BLOCKED
```

and **STOP** the calibration for that run. Do not skip that ticker. Do
not repair the payload. Do not substitute another ticker or source.

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
malformed observation (see §3.3's `CALIBRATION_INPUT_CANONICAL_PARSE_
BLOCKED` gate).

---

## 5. Exact candidate grid (fixed, preserved exactly)

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
#1's unpersisted exact reason and ticker. This 252-per-year constant is
unrelated to §3's calendar-window definition (§3 explicitly does not
require 252 observations per calendar year).

```text
candidate_grid_size=30
grid_derived_from_old_t1_failure=false
candidate_grid_changed_by_this_revision=false
```

---

## 6. Observed window statistics (Component A; window and formulas corrected)

For every applicable **ticker × calendar-year** window — years
`2019, 2020, 2021, 2022, 2023, 2024, 2025` — and for every ticker's
**full calibration-span window** (`2019-01-01` through `2025-12-31`),
compute exactly:

```text
total_returned = valid_returned + invalid_returned
invalid_returned = count of returned observations classified invalid by §4
valid_returned = count of returned observations classified valid by §4
invalid_fraction = invalid_returned / total_returned  (exact rational, never float-rounded)
max_consecutive_invalid_returned_rows = maximum run length of consecutive invalid returned observations, in chronological returned-observation order
```

Definitions remain those of the frozen V8 malformed-row policy
(`V8_HISTORICAL_RESEARCH_DESIGN.md` §17): only observations Yahoo
actually returns with a timestamp are evaluated; expected calendar
missing dates are never invalid returned rows and are never counted in
either numerator or denominator.

```text
individual_calendar_year_with_zero_returned_observations=NOT_APPLICABLE
ticker_with_zero_returned_observations_over_full_calibration_span=CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED (STOP)
strategy_metric_calculation=PROHIBITED
```

---

## 7. Global observed quality envelope (new, binding)

After all applicable observed windows (§6) are characterized, define:

```text
M_fraction = maximum exact invalid_fraction over every applicable ticker-year window and every full-ticker calibration-span window
M_consecutive = maximum max_consecutive_invalid_returned_rows over every applicable ticker-year window and every full-ticker calibration-span window
```

`M_fraction` **must** be represented exactly from integer counts
(`invalid_returned` / `total_returned` as an exact rational, e.g. kept as
a numerator/denominator pair or an equivalent exact representation) —
never derived through floating-point rounding. The window counts used in
each maximum calculation (which window(s) attained `M_fraction`, and
which attained `M_consecutive`) must be reported (§21). No strategy
outcome is exposed by this computation.

---

## 8. Headroom requirement (new, binding — replaces the boundary-equality rule)

The prior preregistration allowed a candidate sitting exactly on the
worst observed boundary to qualify. **That rule is replaced.**

A candidate satisfies the **observed quality criterion** if and only if:

```text
fraction_threshold > M_fraction
AND
max_consecutive_invalid_returned_rows > M_consecutive
```

Both inequalities are **strict**. This deliberately requires positive
predeclared headroom above the worst quality defect observed in the
burned calibration material:

```text
threshold_exactly_equal_to_M_fraction=NOT_SUFFICIENT
max_consecutive_exactly_equal_to_M_consecutive=NOT_SUFFICIENT
```

This is intended to avoid fitting V8B's policy exactly to the empirical
maximum of the calibration set. The candidate grid (§5) is **not**
expanded if no candidate supplies this headroom — the fallback (§15)
governs that outcome instead.

---

## 9. 100% observed pass semantics (unchanged requirement, now mechanical)

The requirement that all applicable observed windows must pass is
preserved. After §8's strict-headroom rule, this follows **mechanically**
from the definition of `M_fraction`/`M_consecutive` as maxima: if
`candidate_threshold > M_fraction` and `candidate_max_consecutive >
M_consecutive`, then by construction every individual applicable window
(whose fraction/run cannot exceed the maxima that define `M_fraction`/
`M_consecutive`) also satisfies `candidate_threshold > window_fraction`
and `candidate_max_consecutive > window_run`, hence passes.

```text
window_specific_exception=NONE
ticker_exclusion=NONE
percentile_exception=NONE
partial_pass_relaxation_eg_99_percent=NONE
```

---

## 10. Synthetic base design (replaced — no longer a ticker-year rule)

**The prior requirement — "first 20 eligible ticker-years with exactly
252 observations" — is deleted and replaced completely by the
following.**

**Synthetic base unit:** 252 consecutive Yahoo-returned observations.
**Not** one calendar year.

For each ticker, in canonical ticker ascending order:

```text
1. use only returned observations inside the approved burned-data calibration span (§3: 2019-01-01 through 2025-12-31)
2. reconstruct valid/invalid flags using the canonical classifier (§4)
3. scan returned observations chronologically
4. find the EARLIEST contiguous slice of exactly 252 consecutive returned observations for which all 252 observations are canonically valid before synthetic injection
5. a ticker contributes at most one synthetic base slice
```

Then: take the **first 20 qualifying tickers** in canonical ticker
ascending order. Base selection order is fully deterministic.

If fewer than 20 distinct tickers possess such a 252-returned-observation
clean slice:

```text
SYNTHETIC_BASE_SELECTION_BLOCKED
```

and **STOP** calibration. Do not choose another base definition.

```text
synthetic_base_definition_changed_by_this_revision=true (was: first 20 eligible ticker-years with exactly 252 observations; now: earliest clean 252-consecutive-returned-observation slice per qualifying ticker, first 20 qualifying tickers)
```

---

## 11. Synthetic base privacy / scientific role (fixed, unchanged)

Synthetic bases come only from already-burned V5-B material. They carry
no V8B evidential weight. Do not use `T1B`, `T2`, `T3`, or old `T1` to
construct synthetic bases.

The synthetic component tests:

```text
canonical_row_classification=TESTED
exact_quality_policy_boundary_semantics=TESTED
```

It does **not** select a threshold based on strategy performance.

---

## 12. Synthetic row construction (fixed, unchanged values)

For synthetic testing, start from copies of the clean returned
observations in each 252-row base slice (§10). For the selected corrupted
indices, alter **exactly one** target field corresponding to one of the
12 canonical malformed-row classes (§4):

| Class | Field alteration |
|---|---|
| `NONFINITE_OPEN` | `open=None` |
| `NONPOSITIVE_OPEN` | `open=0.0` |
| `NONFINITE_HIGH` | `high=None` |
| `NONPOSITIVE_HIGH` | `high=0.0` |
| `NONFINITE_LOW` | `low=None` |
| `NONPOSITIVE_LOW` | `low=0.0` |
| `NONFINITE_CLOSE` | `close=None` |
| `NONPOSITIVE_CLOSE` | `close=0.0` |
| `NONFINITE_ADJCLOSE` | `adjclose=None` |
| `NONPOSITIVE_ADJCLOSE` | `adjclose=0.0` |
| `NONFINITE_VOLUME` | `volume=None` |
| `NEGATIVE_VOLUME` | `volume=-1.0` |

All non-target fields remain unchanged. Every corrupted row must be
classified as **exactly** the intended canonical reason. A different
reason:

```text
SYNTHETIC_CLASSIFIER_MISMATCH
```

and the calibration fails. Corruption classes are never combined in one
synthetic scenario.

---

## 13. Exact synthetic placement formulas (fixed, now fully deterministic — replaces ambiguous language)

Ambiguous phrases from the prior draft ("as evenly as possible",
"centered around the midpoint") are removed and replaced with these exact
formulas. Let `N = 252`, zero-based indices, `K ∈ {0,1,2,3,4,5,6}`.

**`K = 0`:**

```text
placement_family=NONE
corrupted_indices=[]
```

Exactly one uncorrupted scenario is required per
base-sequence × corruption-class combination.

**`K > 0`:**

```text
A. ISOLATED_EVENLY_SPACED
   for j = 0, ..., K-1:
     index_j = floor((j + 1) * N / (K + 1))
   corrupted_indices = [index_0, ..., index_(K-1)]

B. CONSECUTIVE_RUN
   start = floor((N - K) / 2)
   corrupted_indices = [start, start+1, ..., start+K-1]

C. START_RUN
   corrupted_indices = [0, 1, ..., K-1]

D. END_RUN
   corrupted_indices = [N-K, ..., N-1]
```

```text
implementation_discretion=NONE
random_seed=NOT_USED
alternative_rounding=NOT_PERMITTED
```

---

## 14. Synthetic expected policy result (new — exact truth table, replaces the old broad K=1/K=6-only expectations)

For a candidate whose fraction is represented exactly as
`numerator / denominator`, with `N = 252` and `K` injected invalid
returned rows:

```text
fraction_guard_expected_pass = (K * denominator <= N * numerator)
```

For a synthetic scenario, define:

```text
synthetic_max_run =
  0 for K = 0
  1 for ISOLATED_EVENLY_SPACED when K > 0
  K for CONSECUTIVE_RUN, START_RUN, or END_RUN
```

Then:

```text
consecutive_guard_expected_pass = (synthetic_max_run <= candidate_max_consecutive)

overall_policy_expected_pass = fraction_guard_expected_pass AND consecutive_guard_expected_pass
```

The future calibration implementation must reproduce this exact truth
table for **every** candidate and **every** synthetic scenario. Mismatch:

```text
SYNTHETIC_POLICY_SEMANTICS_MISMATCH
```

and the calibration fails.

**Consistency note (independently verified by this task).**
`fraction_guard_expected_pass`'s form (`K * denominator <= N * numerator`)
is the same exact-integer-comparison style already used by the real
production gate in `src/v8_historical_acquisition.py` (`invalid_count *
100 <= total` for the frozen 1% case), generalized to an arbitrary
rational threshold without introducing any float rounding — this is a
faithful generalization of an already-reviewed production semantics, not
a new comparison style invented for calibration.

---

## 15. Synthetic component does not fit a threshold (new clarification)

The synthetic component (§10-§14) is a **policy/classifier semantics
verification layer**. It is **not** an empirical argument that a
particular threshold magnitude is better because it makes more synthetic
cases pass.

```text
rank_candidates_by_synthetic_scenarios_passed=PROHIBITED
```

A candidate either behaves exactly according to its mathematically
expected policy semantics (§14), or it fails semantic verification.
Threshold *magnitude* is selected only through the preregistered observed
quality envelope and strict headroom rule (§7-§9), never through
synthetic pass counts.

---

## 16. `DEFENSIBLE(candidate)` — revised exact definition (replaces §11 of the prior draft)

```text
DEFENSIBLE(candidate) = true
```

if and only if **all** of the following hold:

```text
1.  V5-B calibration input provenance validates (§3)
2.  all designated raw payloads are reproducible under canonical classification semantics (§3.3)
3.  candidate_fraction_threshold > M_fraction (§8, strict)
4.  candidate_max_consecutive > M_consecutive (§8, strict)
5.  therefore 100% of applicable observed ticker-year windows pass (§9, mechanical consequence of 3-4)
6.  therefore 100% of observed full-ticker windows pass (§9, mechanical consequence of 3-4)
7.  every synthetic corrupted row is classified as its exact intended reason (§12, no SYNTHETIC_CLASSIFIER_MISMATCH)
8.  every non-corrupted synthetic row remains unchanged and valid
9.  for every synthetic scenario and every candidate, observed policy pass/fail equals the exact mathematical expected result in §14 (no SYNTHETIC_POLICY_SEMANTICS_MISMATCH)
10. no schema-level hard failure has been converted into a tolerated row-level quality event (§3.3, §4)
```

Otherwise `DEFENSIBLE(candidate) = false`. No qualitative override
exists.

---

## 17. Selection rule (fixed, unchanged)

Evaluate all 30 candidates. Collect the complete set:

```text
D = {candidate | DEFENSIBLE(candidate) = true}
```

If `D` is non-empty: choose the **strictest** candidate. `STRICTEST` is
defined lexicographically:

```text
1. smaller exact invalid_fraction threshold is stricter
2. if fraction thresholds are identical, smaller max_consecutive_invalid_returned_rows is stricter
```

Exact rational comparison is used. No human choice is made after seeing
results.

---

## 18. Tie break (fixed, unchanged)

The strictness order (§17) creates a total order over the 30 unique
candidates (all six fractions are distinct rational values, per §5's
verified ordering, so no substantive tie is expected). If canonical
representation unexpectedly produces duplicate-equivalent candidates:

```text
tie_break=choose the candidate with lexicographically smallest canonical candidate ID
duplicate_equivalence_reported_as=LOW review finding
```

A human is never asked to choose after seeing results.

---

## 19. Stopping rule (fixed, unchanged)

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

## 20. Fallback (fixed, Q1 receives no exemption)

If `D` (§17) is empty:

```text
selected_policy=CALIBRATION_NO_DEFENSIBLE_POLICY
action=BLOCK_V8B_DESIGN_FINALIZATION
```

The grid is not expanded. `Q1` is not automatically restored. No second
threshold grid is run in V8B. `Q1` remains one of the 30 candidates and
receives no exemption from §16's criteria:

```text
if Q1 fails DEFENSIBLE (e.g. does not satisfy the strict headroom rule): Q1 is not retained by default
if Q1 is DEFENSIBLE but a stricter candidate is also DEFENSIBLE: the stricter candidate is selected per §17
if Q1 is the strictest DEFENSIBLE candidate: Q1 may be selected
```

---

## 21. Full result reporting (expanded)

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

Also record, at the artifact level:

```text
input_provenance_hashes
calibration_plan_hash
implementation_commit
run_timestamps
error_counts
artifact_self_hash
mechanically_selected_candidate_or_NO_DEFENSIBLE_POLICY
```

**New additional required fields (this revision):**

```text
M_fraction_exact_numerator_denominator_or_equivalent_exact_representation
M_fraction_source_window_count
M_consecutive
M_consecutive_source_window_count
selected_candidate_fraction_headroom_over_M_fraction
selected_candidate_consecutive_headroom_over_M_consecutive
synthetic_base_ticker_count
synthetic_base_selection_rule
synthetic_base_window_start_and_end_metadata (per selected base slice)
exact_synthetic_placement_formulas_version
full_expected_vs_observed_synthetic_truth_table_mismatch_count
```

```text
best_only_output=PROHIBITED
```

**Privacy in result artifacts.** V5-B is burned development material, so
its exposure is not gated the way `T1B`/`T2`/`T3` are — but result
artifacts must still prefer hashes/counts over unnecessary ticker-level
dumps, consistent with existing repository privacy practice. Do not
publicly expose any data that existing privacy rules prohibit.

---

## 22. Result review gate (expanded)

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
M_fraction_and_M_consecutive_correctly_computed_as_true_maxima=VERIFY
strict_headroom_rule_correctly_applied_no_boundary_equality_admitted=VERIFY
Q1_received_no_exemption_from_DEFENSIBLE_criteria=VERIFY
synthetic_expected_vs_observed_truth_table_fully_matched_or_mismatches_correctly_failed_the_candidate=VERIFY
```

Only after an independent `PASS` may the selected policy be proposed for
`V8B_DESIGN_FINALIZED` (`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §12).
**This document does not authorize that.**

---

## 23. Relationship to the V8B design draft's own calibration wall

This preregistration satisfies, but does not replace,
`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §6.1's requirement that a future
calibration plan freeze `calibration_plan_version`, `allowed_data_sources`,
`exact_included_calibration_datasets`, `exact_exclusions`, and the other
listed fields before approval. Cross-mapping:

```text
allowed_data_sources -> V5-B evaluation cache, 2019-01-01..2025-12-31 (basis A/D per V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md §6's A-D categories: already-burned development/evaluation material outside T1B/T2/T3), plus mandatory synthetic corruption (basis B)
exact_included_calibration_datasets -> the verified V5-B cache, restricted to the 2019-01-01..2025-12-31 window (§3)
exact_exclusions -> old T1, T1B, T2, T3, V7 forward outcomes (§2)
synthetic_corruption_generation_procedure -> §10-§14
synthetic_random_seed_or_seeds -> NOT_USED (§13)
unit_of_analysis -> per-ticker-year window and full calibration-span window (§6)
evaluation_windows -> 2019..2025 individually, and 2019-01-01..2025-12-31 in full (§6)
malformed_row_classifier_and_version -> src/v7_yahoo_collector.py::_row_invalid_reason, verified against source (§4)
exact_finite_candidate_set_invalid_fraction_threshold -> {1/252, 2/252, 1/100, 3/252, 4/252, 5/252} (§5)
exact_finite_candidate_set_max_consecutive_invalid_returned_rows -> {1, 2, 3, 4, 5} (§5)
exact_metrics_computed_per_candidate -> §21
exact_aggregation_method_per_ticker -> §6 (total_returned, invalid_fraction, max_consecutive per ticker/window)
exact_aggregation_method_per_window_or_year -> §6 (per named year and full span, independently)
exact_aggregation_method_across_calibration_material -> §7-§9 (M_fraction/M_consecutive envelope; strict headroom)
exact_defensibility_criterion -> §16
exact_deterministic_candidate_selection_rule -> §17
exact_tie_break_rule -> §18
exact_stopping_rule -> §19
exact_fallback_rule -> §20
exact_missing_or_error_handling -> §3.2 (V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED), §3.3 (CALIBRATION_INPUT_CANONICAL_PARSE_BLOCKED), §6 (CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED), §10 (SYNTHETIC_BASE_SELECTION_BLOCKED)
full_candidate_grid_results_retention -> §21 (MANDATORY, all 30 candidates)
best_only_reporting -> PROHIBITED (§21)
old_t1_input / t1b_input / t2_input / t3_input -> PROHIBITED (§2)
```

---

## 24. Current task does not execute calibration (fixed, unchanged)

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

## 25. Status

```text
status=DRAFT_AWAITING_HUMAN_APPROVAL
```

Next required action after this draft: independent/final review of this
revised preregistration by the research planner (ChatGPT), followed by a
separate human approval gate (`DATA_QUALITY_CALIBRATION_PLAN_APPROVED`,
`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §12), before any calibration
execution may begin. `DATA_QUALITY_CALIBRATION_PLAN_APPROVED` is **not**
marked by this task.
