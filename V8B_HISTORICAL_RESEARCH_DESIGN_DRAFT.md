# V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT

```text
status=DRAFT_AWAITING_HUMAN_GATE
document_type=DESIGN_DRAFT_ONLY
implementation_performed=false
data_acquisition_performed=false
partition_creation_performed=false
real_network_requests_this_document=0
v8b_successor_trust_authority_model=HUMAN_DECISION_REQUIRED_BEFORE_V8B_DESIGN_FREEZE
```

This is a **design draft**, not a frozen design. Nothing in this document
authorizes acquisition, partition allocation, implementation, or any real
network request. It becomes actionable only by following the full gate
sequence in §12. The **immediate next gate** is
`DATA_QUALITY_CALIBRATION_PLAN_APPROVED` — not `HUMAN_DESIGN_FREEZE`.
Calibration execution itself still requires that approved preregistration
first (§6.1) and is not authorized by this document. The successor
trust/authority model for `T1B` (§11) must be resolved later in the
sequence, before `V8B_DESIGN_FINALIZED` and `HUMAN_DESIGN_FREEZE`, and
remains explicitly unresolved and marked
`HUMAN_DECISION_REQUIRED_BEFORE_V8B_DESIGN_FREEZE`; see §11.

This document does not edit, reinterpret, delete, or supersede
`V8_HISTORICAL_RESEARCH_DESIGN.md`. That document remains the frozen,
immutable design of `V8_HISTORICAL_RESEARCH`, which is treated here as a
closed, provenance-only ancestor study. Where this draft inherits a V8
rule unchanged, it cites the exact V8 section rather than restating or
re-deriving it, so there is exactly one authoritative copy of each
unchanged rule.

---

## 0. Relationship to V8_HISTORICAL_RESEARCH

```text
predecessor_study=V8_HISTORICAL_RESEARCH
predecessor_design_document=V8_HISTORICAL_RESEARCH_DESIGN.md (unmodified)
predecessor_status=CLOSED_IMMUTABLE_PROVENANCE
predecessor_design_document_editable_by_this_draft=false
successor_study=V8B_HISTORICAL_RESEARCH
successor_status=DRAFT_AWAITING_HUMAN_GATE
new_study_identity_required=true
```

### 0.1 Why a new study identity, not a V8 amendment

`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.4 requires a **new study** ("a new
sealed partition and a new preregistration") whenever a frozen condition,
parameter, feature, universe, or acceptance threshold is changed after a
sealed layer has been touched, "whether the change is motivated by a Layer
C failure or by an unrelated realisation." §10.3 states the general form:
`returning_to_an_earlier_stage_after_layer_C_access=NEW_STUDY_REQUIRED`.

`T1` under V8 never reached Layer C, and its
`validation_access_count` remains formally `0` (no completed bundle, no
research opening). But the acquisition-quality policy that gated `T1`'s
second attempt is upstream of, and structurally equivalent to, the kind
of frozen parameter those clauses protect.

**Corrected chronology, precise.** `T1` attempt #1 (authorized HEAD
`d5441020389452d85cb19a94f647448775fba8d8`) ran under the production
behavior that existed at the time: an older, undocumented fail-closed
rule under which **any** parser-invalid returned OHLCV row anywhere in a
ticker's response BLOCKed the entire acquisition. `POLICY_G_PRIME_V1_
UNIFORM_RETURNED_ROW_QUALITY_GATE` **did not exist yet** when attempt #1
ran; it was designed, frozen (`V8_HISTORICAL_RESEARCH_DESIGN.md` §17),
implemented, and independently reviewed only **after** attempt #1's
`BLOCKED/MALFORMED_OHLCV` result, and explicitly not fitted to attempt
#1's (unpersisted, unknown) exact invalid-row reason or ticker. `T1`
attempt #2 (authorized HEAD `a8710437db0c0752219d9aff34ac31d55b154d81`)
is the **only** attempt that ran under `POLICY_G_PRIME_V1`, and it BLOCKed
with `MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED`.

Only attempt #2 therefore establishes that at least one evaluated `T1`
ticker/window exceeded the frozen 1% invalid-row fraction threshold.
Attempt #1 establishes only that at least one parser-invalid returned row
occurred under the older, since-superseded any-invalid-row fail-closed
rule; **it must not be read as evidence that the same ticker/window would
also have failed the later 1% policy** — the exact invalid subtype and
ticker for attempt #1 were never persisted, so that question is
unanswerable from the committed record, not merely unanswered.

`POLICY_G_PRIME_V1` was nonetheless preregistered before the one real
attempt that tested it, and that attempt BLOCKed under it. Revising the
policy now, on the same study identity, would be indistinguishable from
exactly the kind of post-outcome parameter change §5.4 exists to prevent.
This draft therefore treats **any acquisition-quality policy change
following a `T1` acquisition outcome as new-study-triggering**, even
though the letter of §5.4 only names Layer C. `V8B_HISTORICAL_RESEARCH`
is that new study.

### 0.2 V8 final status (inherited as immutable fact, not reopened)

```text
v8_t1_attempt_1_result=BLOCKED (reason_class=MALFORMED_OHLCV, failing_request_position=298_of_300)
v8_t1_attempt_2_result=BLOCKED (reason=MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED)
v8_t1_successful_bundle=false
v8_t1_research_opened=false
v8_t1_validation_performed=false
v8_strategy_result_observed=false
v8_t2_acquired=false
v8_t2_opened=false
v8_t3_acquired=false
v8_t3_opened=false
v8_attempt_3_status=PROHIBITED_UNLESS_A_FUTURE_EXPLICIT_V8_DESIGN_DECISION_SAYS_OTHERWISE
v8b_performs_v8_attempt_3=false
v8_status=CLOSED_IMMUTABLE_PROVENANCE
```

`V8B_HISTORICAL_RESEARCH` does not perform, request, or depend on a `V8`
`T1` attempt #3. `V8` remains closed exactly as recorded in
`V8_STATE.json` / `V8_PROJECT_STATE.md` / `V8_HISTORICAL_RESEARCH_DESIGN.md`
at reviewed HEAD `fdf2fc8896db6ed013ddef4c7a66036280d3f23b`. No fact in
this section is inferred; each line is carried forward verbatim from that
HEAD's docs/state.

**Reading the two reason strings above correctly requires §0.1's
chronology.** `v8_t1_attempt_1_result`'s `reason_class=MALFORMED_OHLCV` is
the older any-invalid-row fail-closed rule's BLOCK, recorded before
`POLICY_G_PRIME_V1` existed. `v8_t1_attempt_2_result`'s
`MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED` is the only occurrence of
the 1% policy actually firing. The two results must not be read as "the
same 1% gate BLOCKed twice."

---

## 1. Purpose and position

`V8B_HISTORICAL_RESEARCH` is the direct successor to `V8_HISTORICAL_RESEARCH`,
run for the same reason V8 was run (historical strategy discovery on
genuinely fresh Japanese-equity cross-sections, kept isolated from
`V7_FORWARD_CAPACITY`), after V8 closed without ever successfully acquiring
`T1`. It is **not** a retry of V8 under a nudged parameter; it is a new
study with its own identity, because §0.1 requires that.

```text
v7_isolation_inherited_unchanged=true (V8_HISTORICAL_RESEARCH_DESIGN.md §3)
v8b_may_not_read_v7_forward_outcomes=true
v8b_may_not_modify_any_v7_file=true
v8b_may_not_modify_any_v8_file=true
```

---

## 2. Inherited unchanged from V8 (by reference, not restated)

Per the minimum-change principle, the following remain **exactly** as
frozen in `V8_HISTORICAL_RESEARCH_DESIGN.md` and are inherited by
reference. None of these is re-derived, re-justified, or re-typed here;
each citation is the sole authoritative source.

| Element | Source | Change in V8B |
|---|---|---|
| `P_hist` span (2016-04-01 → 2025-12-31) | §5.1 | none |
| `T0` role and non-evidential status | §5.2, Decision 1 | none (see §3.1 below for reuse scope) |
| 8-split expanding-window walk-forward scheme | §8.1 | none |
| Causality / label-confirmation rule | §8.2 | none |
| Required metric set for promotion | §8.3 | none |
| `WALK_FORWARD_SURVIVOR` thresholds (all nine) | §8.4 | none |
| Friction grid (0.03/0.05/0.10/0.15%, base 0.05, floor 0.10) | §8.5 | none |
| Parameter-neighbourhood robustness rule | §8.6 | none |
| Search-overfitting controls and registry requirements | §6, §7, §9 | none |
| Promotion gate sequence and invariants | §10 | none, except the new pre-`T1B` calibration sub-gate this draft adds (§12) |
| Real-money / deployment policy | §11 (V8) | none |
| Permanent prohibitions | §12.1 | none |
| Prohibited claim language | §12.3 | none, and extended — see §7.4 below |
| Deterministic ticker-block ordering rule (`sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending`) | §5.1 | none — reused to draw `T1B` (§4) |
| Yahoo Chart parser and transport (`src/v7_yahoo_collector.py`) | implementation | none — reused read-only, as V8 itself reused it from V7 |
| Production acquisition security boundary (Git-provenance pin, trust-anchor read, strict-origin transport, atomic staging/publish) | `src/v8_historical_acquisition.py`, `src/v8_partition.py` | none as a mechanism; see §10 for the two deferred hardening items that must land before it is reused for research-opening |

Any future change to an item in this table, once V8B itself freezes,
triggers the same new-study rule this draft invokes in §0.1 against V8.

---

## 3. Block reuse decision

This section resolves, per-block, the ambiguity the prior review left open.
`T0`, old `T1`, `T2`, `T3`, and `T_spare` are **not** lumped together. Each
was touched differently by V8's real attempts, and each is assessed on
that basis alone.

### 3.1 `T0` — `SAFE_TO_REUSE_FOR_DEVELOPMENT_ONLY`

`T0` was never sealed, is explicitly non-evidential (`V8_HISTORICAL_RESEARCH_DESIGN.md`
§5.2, Decision 1), and unlimited reuse of already-spent data "costs nothing,
provided nothing is claimed from it." Nothing about the `T1` acquisition
BLOCK changes this. `T0` remains available for Layer A development **and**,
under this draft, for the data-quality calibration phase (§6), because
calibration on `T0` carries the same non-evidential status and spends
nothing that isn't already spent.

### 3.2 Old `T1` — `RETIRED_BURNED_FOR_CONFIRMATORY_USE`

Even though `validation_access_count` is formally `0` for old `T1` — no
completed bundle, no formal research opening — the researcher has observed
a real fact about it, established conservatively and precisely (§0.1):
**only `T1` attempt #2** exposed that old `T1` fails `POLICY_G_PRIME_V1`'s
frozen 1% fraction gate at some (unknown) ticker/window. Attempt #1
predates that policy and establishes a different, unrelated fact — that
at least one parser-invalid returned row occurred under the older,
since-superseded any-invalid-row rule — which is not evidence about the
1% gate one way or the other. Per `V8_HISTORICAL_RESEARCH_DESIGN.md` §9.4,
"a crashed or aborted run that nevertheless exposed any outcome statistic
counts as an access; only a run that provably produced no outcome
information... does not." Attempt #2's BLOCK reveals "this specific
300-ticker set has at least one member failing the frozen 1% threshold"
— that is outcome information, even without the ticker's identity, and
even though it falls short of a formal Layer B validation result
(`validation_access_count` correctly remains `0`, since no bundle was
ever completed and no research opening ever occurred).

**Retirement justification, stated conservatively.** Because attempt #2
exposed this non-trivial acquisition-quality fact about old `T1`, reusing
old `T1` for any future threshold-sensitive confirmatory validation would
no longer be blind — the researcher would be selecting or evaluating a
threshold with knowledge that this specific 300-ticker set is known to
contain at least one member near or past a specific, real threshold. That
is sufficient, on its own, to retire old `T1` from validation use, without
needing to claim (and this draft does not claim) that attempt #1
contributes anything to that determination. Old `T1` is retired from
validation use in V8B and is not reused as `T1B` under any name.

### 3.3 Existing `T2` — `REUSE_WITH_CAVEAT`, conditionally preservable as V8B's sealed holdout

Preservable **if and only if** every one of the following holds at the
moment V8B's design freezes (§9 below gives the full argument):

```text
t2_acquired=false
t2_opened=false
t2_ticker_identities_exposed=false
t2_outcomes_or_features_observed=false
universe_definition_unchanged=true
partition_algorithm_unchanged=true
v8b_data_quality_policy_frozen_before_any_t2_acquisition=true
```

All of these hold today (verified from `V8_STATE.json` /
`V8_PROJECT_STATE.md` at the reviewed HEAD: `T2_real_data_acquired=false`,
`T2_opened=false`, no `T2` content has ever been read by any V8 process).
`T2` is therefore **conditionally reusable** as V8B's Layer C sealed
holdout — see §9 for why old-`T1`'s failure does not contaminate it.

### 3.4 Existing `T3` — `REUSE_WITH_CAVEAT`, preservable as reserve

`T3` remains untouched (`T3_data_acquired=false`, never opened, acquisition
unconditionally rejected at the code level regardless of confirmation
token). All seven of §3.3's conditions have a direct `T3` analogue, and
all seven hold today:

```text
t3_acquired=false
t3_opened=false
t3_ticker_identities_exposed=false
t3_outcomes_or_features_observed=false
universe_definition_unchanged=true
partition_algorithm_unchanged=true
v8b_data_quality_policy_frozen_before_any_t3_acquisition=true (not applicable in initial V8B since T3 acquisition is not planned; stated for completeness in case a future gate releases T3)
```

`T3` is preserved as `SEALED_RESERVE` under V8B on the same terms as under
V8 (`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.4, Decision 6): not used in
initial V8B, not opened for any purpose, release requires a separate
future human gate.

### 3.5 Existing `T_spare` — available for exactly one new validation block

`T_spare`'s membership was necessarily materialized and hashed inside the
original V8 partition build (the self-hash-verified manifest records a
`t_spare_ticker_list_sha256` alongside `T1`/`T2`/`T3`'s, and its size,
`~1,904`, is recorded in committed docs/state) — its content was not
"never computed." What is true, and is the property this draft actually
relies on, is narrower: `T_spare`'s ticker identities have not been
exposed to the human/public research loop by this draft or any prior
review, and no `T_spare` member has been used in any acquisition,
feature, or outcome computation. Available under §4/§5 below to source
exactly one new 300-ticker validation block, and for nothing else in this
design (no repeated drawing, no per-ticker replacement — see §5).

### 3.6 Why this is not "one new full partition"

A full new partition (redrawing `T0`–`T3`/`T_spare` from the universe from
scratch) is **not** required, because the condition that would force it —
a change to the universe definition (`universe_definition_unchanged`) or
the deterministic partition algorithm itself
(`partition_algorithm_unchanged`), both named explicitly in §3.3 — is not
proposed anywhere in this draft. Only `T1`'s *role* changes (old `T1` retired, one fresh
block drawn from `T_spare` under the *same* ordering rule). `T2`/`T3`
membership was likewise materialized and hashed once, at original
partition-build time, as part of the same manifest — that internal
materialization is not itself a form of exposure (see §9's corrected
non-contamination argument for `T2`, which applies identically to `T3`).
What has not happened, for either block, is any acquisition, exposure to
the research loop, or outcome computation; on that basis both remain
unaffected members of the same partition, and nothing about their
*membership* needs to be redrawn.

---

## 4. New validation block: `T1B`

```text
t1b_parent_block=T_spare (existing, untouched)
t1b_offset_within_parent_t_spare=0
t1b_slice_start_inclusive=0
t1b_slice_end_exclusive=300
t1b_size=300
t1b_selection_rule=DETERMINISTIC_PREDECLARED_ZERO_DISCRETION
t1b_selection_rule_text="T1B = parent_T_spare[0:300]; remaining_T_spare = parent_T_spare[300:], where parent_T_spare is the canonical ordered T_spare sequence already contained in / derivable from the trusted parent V8 partition manifest under V8_HISTORICAL_RESEARCH_DESIGN.md §5.1's frozen deterministic ordering (sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending)"
t1b_selection_conditional_on_data_quality=false
t1b_ticker_identities_exposed_in_this_document=false
t1b_ticker_identities_exposed_at_design_freeze=false
implementation_time_discretion_over_t1b_offset=false
old_t1_replaced_ticker_by_ticker=false
old_t1_retired_wholesale=true
```

`T1B` is a **new, distinct logical block**, not a repair of old `T1`. Old
`T1`'s 300-member set is retired in its entirety (§3.2); no member of old
`T1` is carried into `T1B`, and no single failing ticker inside old `T1`
is swapped out. This document does not read or expose which tickers land
in `T1B`; that remains a matter for the implementation phase (§12), after
design freeze, exactly as `T0`–`T3` assignment contents were never
exposed by `V8_HISTORICAL_RESEARCH_DESIGN.md` itself.

**The logical membership rule is frozen now, with zero implementation-time
discretion.** `T_spare`'s position *among the five original global blocks*
(`T0`, `T1`, `T2`, `T3`, `T_spare`, cut in that order from the eligible
universe under §5.1's ordering) is not itself a choice this draft is
making — it is simply where `T_spare` already sits in the existing,
already-frozen V8 partition. That global position is unrelated to `T1B`'s
membership rule. Within `T_spare`'s *own* internal ordering (the same
`T_spare` sequence, ordered by the same §5.1 rule, that the trusted parent
V8 partition manifest already fixes), `T1B`'s offset is exactly and only:

```text
T1B = parent_T_spare[0:300]
remaining_T_spare = parent_T_spare[300:]
```

There is no boundary offset left open for implementation time to choose.
A future implementation does not select where `T1B` begins; it only
*materializes and verifies* this already-frozen zero-offset slice against
the trusted parent `T_spare` sequence, and must be rejected by independent
review if it does otherwise. `T1B`'s allocation, and the authority
artifact that vouches for it, must additionally follow §11's successor
trust/authority model — the existing V8 trust anchor alone does not cover
`T1B` (§11.1).

---

## 5. One-shot validation-block draw rule (binding)

```text
fresh_validation_block_draw_count=1
repeated_draw_until_pass_allowed=false
automatic_replacement_on_failure_allowed=false
manual_replacement_on_failure_allowed=false
```

If `T1B`'s raw acquisition fails V8B's frozen data-quality policy (§6–§7),
the result is:

```text
V8B_VALIDATION_ACQUISITION_FAIL
```

On `V8B_VALIDATION_ACQUISITION_FAIL`:

- no second `T_spare` draw is performed, automatically or manually, under
  this study;
- `T1B`'s specific failure is **not** used to tune, relax, or otherwise
  select a next threshold (the same non-contamination rule as §6 applies
  recursively to any successor of `V8B`);
- `V8B_HISTORICAL_RESEARCH` itself stops at that point and is closed, on
  the same terms V8 closed under (§0.2), pending a genuinely new successor
  study (`V8C`, or similar) with its own identity, if one is later chosen.

This forecloses exactly the failure mode named in the task: draw `T1B` →
fails → draw `T1C` → fails → draw `T1D` → repeat until acquisition passes.
That sequence would select a validation block conditional on data-quality
outcomes, which is a search over blocks disguised as a single validation
attempt, and is exactly the kind of unregistered, outcome-conditioned
trial `V8_HISTORICAL_RESEARCH_DESIGN.md` §6.2 and §9 exist to prevent at
the strategy layer. The same logic applies one level up, at the block-draw
layer, and this rule closes it there.

---

## 6. Data-quality calibration phase (before design freeze)

```text
calibration_required_before_design_freeze=true
calibration_may_use_old_t1_outcome_as_calibration_input=false
calibration_may_retain_old_t1_outcome_as_provenance_narrative=true
```

**Permitted calibration bases** (any combination):

- **A. `T0` data.** Already burned, Layer A non-evidential
  (`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.2). Spending it on calibration
  costs nothing additional and contaminates nothing, since no expectancy
  claim may ever rest on `T0` regardless.
- **B. Synthetic corruption experiments.** Deliberately injected, known
  synthetic defects (missing rows, out-of-range values, non-finite
  values, duplicate timestamps, etc.) at controlled rates, to observe how
  a candidate threshold behaves against a *known ground truth* rather than
  an unknown real payload.
- **C. Provider documentation/specifications.** Any published Yahoo Chart
  API behavior/quality documentation, used descriptively, not fitted to
  the unknown old-`T1` failure.
- **D. Independent calibration data.** Any data source that is not old
  `T1`, not `T1B`, not `T2`, not `T3` — e.g., already-exposed V3–V7
  historical spans (per `V8_DATA_EXPOSURE_AUDIT.md`), which carry no
  V8B evidential weight but are not off-limits as calibration material
  precisely because they are already spent for evidential purposes.

**Forbidden calibration input**, absolutely:

```text
old_t1_raw_payload=FORBIDDEN
old_t1_ticker_identity=FORBIDDEN
old_t1_exact_invalid_fraction=FORBIDDEN
old_t1_failing_year_or_date=FORBIDDEN
reverse_engineering_a_threshold_that_would_have_passed_old_t1=FORBIDDEN
```

The following two facts may be retained and cited as provenance narrative
(exactly as this draft already does in §0.1/§0.2), but **must not** enter
any numeric derivation of a V8B threshold:

```text
old_t1_attempt_1_fact="old any-invalid-row fail-closed rule BLOCKed with reason_class=MALFORMED_OHLCV, before POLICY_G_PRIME_V1 existed"
old_t1_attempt_2_fact="POLICY_G_PRIME_V1 BLOCKed with MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED -- the only real test of the 1% gate"
```

These are two distinct facts about two different acquisition-quality
regimes, not "the 1% gate BLOCKed twice." A calibration record that
cannot demonstrate its output number was reachable without consulting
either fact does not satisfy this section.

**Calibration phase sub-gates** (detail of §12's
`DATA_QUALITY_CALIBRATION_PLAN_APPROVED` → `CALIBRATION_RESULT_REVIEW` span):

1. A calibration **preregistration/plan** — not merely a statement of
   which of A–D it will use — is written and approved **before** any
   calibration run. §6.1 below fixes exactly what that preregistration
   must freeze; retaining the full threshold distribution alone (item 3,
   unchanged) is necessary but not sufficient.
2. The calibration plan is implemented exactly as preregistered, using
   only the allowed material.
3. The calibration result — including the full distribution of outcomes
   at multiple candidate thresholds, not just the one selected — is
   reviewed independently before it is adopted into V8B's frozen policy,
   mirroring §9.3's "full trial distribution, not only the maximum"
   requirement.
4. If, after calibration, no defensible independent basis for a specific
   number emerges, V8B's design freezes with `POLICY_G_PRIME_V1` retained
   unchanged (Option Q1, §7) rather than an invented number — **unless**
   the preregistered plan itself, before any calibration result existed,
   predeclared a different scientifically justified no-selection outcome
   (§6.2).

### 6.1 Calibration preregistration requirements (binding on any future plan)

This draft does **not** perform calibration and does **not** invent the
numeric candidate grid — that remains for the future
`DATA_QUALITY_CALIBRATION_PLAN_APPROVED` gate. What this draft fixes now
is the **shape** every future calibration plan must satisfy before it may
be approved. A plan missing any of the following is not a valid
preregistration and does not satisfy sub-gate 1 above.

Before any calibration execution, a separate calibration
preregistration/plan document must freeze, in writing, all of:

```text
calibration_plan_version=<required>
allowed_data_sources=<required, drawn only from §6's A-D>
exact_included_calibration_datasets=<required, enumerated precisely>
exact_exclusions=<required, enumerated precisely -- old T1, T1B, T2, T3 always excluded>
synthetic_corruption_generation_procedure=<required if basis B is used>
synthetic_random_seed_or_seeds=<required if basis B involves any randomness>
unit_of_analysis=<required -- e.g. per-ticker-day observation, per-ticker-year window>
evaluation_windows=<required -- e.g. full P_hist and/or the frozen test years of V8_HISTORICAL_RESEARCH_DESIGN.md §8.1>
malformed_row_classifier_and_version=<required -- exact reason taxonomy/version used to label a row invalid>
exact_finite_candidate_set_invalid_fraction_threshold=<required -- an explicit finite list, not a range or search procedure>
exact_finite_candidate_set_max_consecutive_invalid_returned_rows=<required -- an explicit finite list>
exact_metrics_computed_per_candidate=<required>
exact_aggregation_method_per_ticker=<required>
exact_aggregation_method_per_window_or_year=<required>
exact_aggregation_method_across_calibration_material=<required>
exact_defensibility_criterion=<required -- the precise, checkable condition a candidate must satisfy to be called DEFENSIBLE>
exact_deterministic_candidate_selection_rule=<required -- how one candidate is chosen if more than one is DEFENSIBLE>
exact_tie_break_rule=<required>
exact_stopping_rule=<required -- when calibration execution is considered complete>
exact_fallback_rule=<required -- what happens if no candidate is DEFENSIBLE; default is Q1 per sub-gate 4 unless predeclared otherwise>
exact_missing_or_error_handling=<required -- how a calibration-data error/gap is handled, decided before any run>
full_candidate_grid_results_retention=MANDATORY (every candidate's result, not only the selected one)
best_only_reporting=PROHIBITED
old_t1_input=PROHIBITED
t1b_input=PROHIBITED
t2_input=PROHIBITED
t3_input=PROHIBITED
```

No value for any of these fields is invented or filled in by this draft.
They are requirements on the *shape* of a future document, not the
content of one.

### 6.2 Prohibition on adaptive threshold shopping

```text
calibration_candidate_grid_frozen_before_first_calibration_result=true
adaptive_grid_expansion_after_results=false
adaptive_metric_change_after_results=false
adaptive_acceptance_criterion_change_after_results=false
adaptive_tie_break_change_after_results=false
calibration_run_count=EXACTLY_THE_PREREGISTERED_RUN_SET
calibration_stops_after_full_preregistered_grid_evaluated=true
```

Once the preregistered candidate grid, metrics, aggregation method,
defensibility criterion, selection rule, tie-break rule, and stopping
rule are frozen (§6.1), none of them may be changed after any calibration
result — partial or complete — has been observed. The candidate grid for
`invalid_fraction_threshold` and `max_consecutive_invalid_returned_rows`
must each be an explicit finite list fixed before the first calibration
run; expanding, narrowing, or re-weighting that grid after seeing how any
candidate performed is prohibited, as is silently swapping the
acceptance criterion or the tie-break rule to favor whichever candidate
happened to look best. "Look at results, then decide what seems
reasonable" is not a valid calibration procedure under this design.

If the fully preregistered grid produces no candidate satisfying the
preregistered defensibility criterion, the fallback is `POLICY_G_PRIME_V1`
unchanged (Option Q1, §7), unless the plan itself predeclared, before any
result existed, a different scientifically justified no-selection
outcome. This mirrors, at the calibration layer, the same discipline
`V8_HISTORICAL_RESEARCH_DESIGN.md` §9 already requires of strategy
trials: the selection rule is declared before the search, not chosen
afterward to match whichever result looks best.

---

## 7. Policy options comparison

### Q1 — retain `POLICY_G_PRIME_V1` unchanged

```text
invalid_fraction_threshold=0.01 (unchanged)
max_consecutive_invalid_returned_rows=5 (unchanged)
membership_rule=T1B_drawn_once_from_T_spare_per_section_4
```

- **Selection bias:** none introduced; identical mechanism to V8.
- **Data-availability bias:** none beyond what V8 already had (per-block
  fail-whole-acquisition, no exclusion of tickers by availability).
- **Survivorship-like bias:** unchanged from V8's already-disclosed
  residual (`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.9) — not worsened, not
  fixed.
- **Reproducibility:** highest of the three options — the threshold is
  byte-identical to a threshold already independently reviewed twice
  (policy review + security/integer-invariant reviews) under V8.
- **Single-ticker brittleness:** unchanged — still fails the whole
  300-ticker block on one non-compliant ticker. This general shape (one
  ticker's row-level data quality can BLOCK an entire 300-ticker block) is
  shared by both the old any-invalid-row rule that produced attempt #1's
  `MALFORMED_OHLCV` BLOCK and `POLICY_G_PRIME_V1`'s fraction gate that
  produced attempt #2's `FRACTION_EXCEEDED` BLOCK — but they are two
  different rules with two different tolerances, not one gate that fired
  twice, and retaining Q1 does not resolve this brittleness pattern
  either way.
- **Operational feasibility:** highest — zero new design/implementation
  work; §5's one-shot rule already bounds the downside (one
  `V8B_VALIDATION_ACQUISITION_FAIL` and the study stops, rather than
  indefinite retries).
- **Risk this is chosen for the wrong reason:** none, provided it is
  chosen *because calibration found no independent basis for a different
  number* (§6.4), not merely because inertia is easiest.

### Q2 — new, independently-calibrated threshold; membership still frozen before acquisition

```text
invalid_fraction_threshold=<from calibration, if defensible>
max_consecutive_invalid_returned_rows=<from calibration, if defensible>
membership_rule=T1B_drawn_once_from_T_spare_per_section_4 (unchanged)
```

- **Selection bias:** none from the threshold itself, *provided* §6's
  calibration wall is actually respected — a threshold calibrated on
  forbidden material would reintroduce exactly the post-outcome-adaptation
  problem the prior design review identified for a bare "1%→2%" change.
- **Data-availability bias:** none — membership stays fixed regardless of
  the number chosen; the axis affected is per-ticker row inclusion, not
  ticker set membership.
- **Survivorship-like bias:** same as Q1 — unaffected either way.
- **Reproducibility:** lower than Q1 until the calibration record itself
  is independently reviewed and reproducible; equal to Q1 after that
  review passes.
- **Single-ticker brittleness:** potentially reduced or increased
  depending on the calibrated number's direction and rationale — not
  knowable in advance, which is exactly why this draft does not preselect
  a number (see below).
- **Operational feasibility:** lower than Q1 — requires the full
  calibration phase (§6) and its own independent review before design
  freeze.
- **Explicit non-selection:** this draft does **not** select Q2 merely
  because `POLICY_G_PRIME_V1` failed on old `T1`. Q2 becomes preferred
  over Q1 only if the calibration phase (§6) produces a specific,
  independently defensible number; absent that, §7's default is Q1 or the
  explicit placeholder below.

### Q3 — pre-partition data-quality eligibility screen

```text
screening_point=BEFORE_partition_block_cutting
membership_rule=CHANGES (eligible universe itself is filtered pre-partition)
```

- **Selection bias:** the most exposed of the three options. Filtering
  the universe by "does this ticker have sufficiently clean Yahoo OHLCV"
  *before* partitioning correlates study inclusion with data
  observability — a distinct but structurally similar mechanism to the
  survivorship bias `V8_HISTORICAL_RESEARCH_DESIGN.md` §5.9 already
  discloses as unresolved, and which §12.3 explicitly forbids claiming is
  "resolved by fresh tickers."
- **Data-availability bias:** by construction, this *is* a
  data-availability-conditioned selection rule; it must be treated and
  disclosed as its own named bias, not folded into or confused with
  survivorship bias.
- **Survivorship-like bias:** compounds, rather than replaces, the
  existing disclosed survivorship bias — the study would now select on
  two axes correlated with "being an easy company to observe," not one.
- **Reproducibility:** high, if the eligibility rule itself is
  mechanical/deterministic and frozen before the universe is drawn on
  (same reproducibility standard as §5.1's block-cutting rule) — but this
  requires building and freezing a *new* rule, which does not exist today.
- **Single-ticker brittleness:** structurally solved — no single ticker
  can later BLOCK a whole 300-member block, because non-compliant tickers
  are excluded from the eligible pool up front rather than discovered at
  acquisition time.
- **Operational feasibility:** lowest of the three — requires designing,
  freezing, and independently reviewing an entirely new eligibility rule,
  potentially re-fetching/re-screening the full ~3,115-ticker universe,
  and (per §3.6) would force a new full partition, unlike Q1/Q2.
- **Verdict for V8B specifically:** available as a future option but
  **not adopted by this draft**, because it is the highest-cost,
  highest-bias-surface option and nothing about `T1B`'s single frozen
  draw (§4–§5) requires it. It remains a legitimate direction for a
  further successor study if Q1/Q2 prove operationally unworkable across
  repeated *new* studies (not repeated draws within one study — §5 already
  forecloses that).

### 7.4 Threshold decision for this draft

```text
V8B_MALFORMED_OHLCV_THRESHOLD=TO_BE_CALIBRATED_BEFORE_DESIGN_FREEZE
V8B_MALFORMED_OHLCV_POLICY_PREFERENCE_PENDING_CALIBRATION=Q1_OR_Q2
V8B_MALFORMED_OHLCV_POLICY_Q3_ADOPTED=false
numeric_threshold_invented_in_this_draft=false
```

No number is set here. §6's calibration phase, not this draft, produces
either a defensible new number (→ Q2) or no defensible basis for a new
number (→ Q1 retained). Both outcomes are legitimate; inventing a number
in this document to avoid running the calibration phase would itself be
exactly the kind of post-outcome/ungrounded adaptation this whole
successor study exists to avoid.

### 7.5 Extension of prohibited claim language (§12.3 inheritance)

In addition to every phrase `V8_HISTORICAL_RESEARCH_DESIGN.md` §12.3
already prohibits (inherited unchanged, §2), V8B additionally prohibits:

```text
"the new threshold was validated against the old T1 failure"    PROHIBITED
"T1B fixes what T1 got wrong"                                    PROHIBITED
"the calibration confirms the old gate was too strict"           PROHIBITED
```

None of these characterizations is available under this design regardless
of what the calibration phase produces, because none of them is knowable
without inspecting forbidden material (§6).

---

## 8. Preferred methodological shape (summary)

```text
unchanged_from_V8=[P_hist, T0_development_role, walk_forward_scheme,
  friction_grid, promotion_thresholds, parser, yahoo_transport,
  deterministic_partition_ordering, existing_untouched_T2, existing_untouched_T3]
changed_from_V8=[study_identity, old_T1_status(retired),
  one_fresh_validation_block(T1B, one_shot_draw),
  acquisition_quality_policy(calibrated_or_retained_per_section_7, never_reused_verbatim_without_the_section_6_wall),
  acquisition_and_research_opening_security_boundary(hardened_per_section_10)]
```

No frozen V8 methodology outside this explicit changed-list is touched.
In particular, the walk-forward scheme, friction grid, and promotion
thresholds are **not** revisited merely because acquisition failed —
acquisition failure is a data-quality event, not a signal about whether
those methodological choices were right (this mirrors the "data
acquisition failure ≠ validation failure ≠ strategy failure" distinction
already established for V8).

---

## 9. `T2` reuse: full justification

```text
t2_reuse_recommended=CONDITIONAL_YES (see conditions below; final confirmation deferred to design-freeze step, not asserted final here)
```

**Why old `T1`'s failure does not contaminate `T2`.** Contamination in
this design's sense (§5.4, §9) means information flow from an outcome
into a decision that should have been made blind to it. The information
`T1` attempt #2 specifically produced (§0.1 — attempt #1 predates
`POLICY_G_PRIME_V1` and does not establish this) is: *"under
`POLICY_G_PRIME_V1`, at least one member of the old-`T1` 300-ticker set
fails the 1% gate."* This is a statement about **old `T1`'s specific
membership**, drawn under the same deterministic rule as every other
block but occupying a disjoint position in the ordering from `T2`. It
carries no information about which, if any, `T2` members would or would
not pass any gate, because:

1. **Corrected non-contamination claim.** The original V8 partition build
   necessarily created `T2`'s assignment, persisted it inside the private
   partition manifest, computed its `t2_ticker_list_sha256`, and was
   later re-validated for manifest integrity — so it is **not** accurate
   to claim `T2`'s membership was "never read," "never hashed," or "never
   internally touched." What *is* accurate, and is what this draft
   actually relies on: `T2` ticker identities have not been exposed to
   the human/public research loop; no `T2` Yahoo OHLCV acquisition has
   occurred; no `T2` raw payload has been opened; no `T2` feature
   distribution has been computed; no `T2` strategy outcome, profit,
   trade, or equity information has been observed; and `T2`'s research
   access/open count remains zero. `T2` is conditionally reusable on the
   basis of this absence of price/feature/outcome exposure — not on a
   false claim that its membership was never internally materialized.
2. The frozen partition invariant `T0 ∩ T1 ∩ T2 ∩ T3 = ∅`
   (`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.6) guarantees no ticker overlap
   between old `T1` and `T2` — old `T1`'s failing ticker(s), whoever they
   are, are definitionally not members of `T2`.
3. `T2_real_data_acquired=false` and `T2_opened=false` hold at the
   reviewed HEAD (§3.3) — no OHLCV row belonging to `T2` has ever been
   fetched, so there is no possible channel through which a `T2`-specific
   data-quality fact could have leaked into any decision made about
   `T1B` or about a calibrated threshold.

**Does changing the acquisition-quality policy before `T2` acquisition
affect the validity of keeping the same `T2` membership?** No, under one
condition that this draft makes binding: the policy must be finalized
(via §6's calibration wall, not via inspection of `T2` itself) **before**
`T2` is ever acquired, exactly as required for `T1B`. Changing *which
threshold governs acquisition* does not change *which tickers `T2`
contains* — those are orthogonal axes (a policy is a rule about row
validity, not about block membership). `V8_HISTORICAL_RESEARCH_DESIGN.md`
§17 clause 10 already establishes this orthogonality precedent within V8
itself: the malformed-row policy is declared uniform across `T0`–`T3`,
independent of which specific block is being acquired. Carrying a new,
independently-calibrated policy into `T2` acquisition is methodologically
identical to how V8's own §17 policy was meant to apply uniformly to `T2`
had `T1` succeeded — nothing about `T2`'s sealed status is weakened by
that policy having been revised in response to a *different* block's
acquisition outcome, provided (per §6) it was not revised in response to
*T2's own* outcome, which is impossible since `T2` has never been
acquired.

**If any one of all seven conditions in §3.3 stops holding** (e.g., `T2`
somehow gets acquired or opened before V8B's design freezes, or the
universe/partition algorithm changes), this recommendation is withdrawn
automatically and a new sealed block must be sourced from `T_spare`
instead, following the same one-shot rule as §4–§5.

---

## 10. Security requirements retained as design requirements (not implemented here)

Carried forward, unresolved, exactly as identified in the prior
independent security review and the prior design-support review:

```text
requirement_1=generic_open_for_functions_arbitrary_mapping_contract_must_be_bound (prose: the `open_for_*` guard functions)
requirement_2=persisted_acquisition_manifest_read_time_binding_to_block_trusted_partition_ticker_list_hash_and_partition_manifest_identity
implementation_performed_by_this_draft=false
```

1. **Generic `open_for_*` guard contract.** The five `open_for_*` guard
   functions currently accept an arbitrary `Mapping`. Before `T1B` is
   opened for Layer B research, or `T2` is opened for Layer C, the
   official research-opening path must be re-specified to accept
   **only** a manifest sourced through the hardened
   `read_acquisition_manifest()` (or an equivalent successor), never a
   caller-constructed mapping. A caller-crafted `{"sealed": False,
   "research_access_authorized": True}` must not be capable of
   authorizing research access through the *official* path, regardless
   of what a directly-importing malicious caller could always construct
   for itself.
2. **Read-time content binding.** `read_acquisition_manifest()` must, by
   the time it is relied on for `T1B` or `T2` opening, re-verify that a
   persisted manifest's `ticker_list_sha256` and `partition_manifest_sha256`
   fields actually correspond to the trusted partition anchor for the
   block being read — not merely that the manifest's own
   identity/label fields (`block`/`role`/`status`/`sealed`) are internally
   self-consistent, which is the current state of the art after V8's
   security remediation.

Neither requirement is implemented, designed in code form, or scheduled
against a commit in this document. Both remain **design requirements**
that block the two specific gates named above (§12:
`SEPARATE RESEARCH-OPENING GATE` and `T2 SEALED HOLDOUT GATE`), and do
not block `T1B` partition allocation or `T1B` raw acquisition themselves,
consistent with the prior review's finding that neither issue is
reachable from the raw-acquisition path.

---

## 11. Successor trust/authority model for `T1B`

```text
v8b_successor_trust_authority_model=HUMAN_DECISION_REQUIRED_BEFORE_V8B_DESIGN_FREEZE
existing_v8_trust_anchor_sufficient_for_t1b=false
implementation_performed_by_this_section=false
```

### 11.1 The gap this corrects

The current production acquisition code (`src/v8_historical_acquisition.py`)
permits only `block in ("T1", "T2")` and binds exclusively to
`V8_TRUSTED_PARTITION.json` plus the original V8 partition manifest that
anchor authorizes. That manifest's `block_assignments` contains exactly
`T0`, `T1`, `T2`, `T3`, and `T_spare` — it contains **no** logical block
named `T1B`, and cannot, without modification, be asked to authorize one.
An earlier version of this draft asserted that the existing trust anchor
"does not need re-pinning for `T1B`... since `T1B` is drawn from within
the already-anchored `T_spare` set." **That assertion is withdrawn as
incorrect.** Being drawn from an already-anchored parent set is not the
same as being an already-anchored block in its own right: the anchor's
authorization statement is scoped to the five block names it actually
enumerates, and silently treating `T1B` as if it inherited `T2`-or-`T1`-
grade authorization from its parent `T_spare` would be exactly the kind
of trust-boundary assumption the prior independent security review exists
to catch. `V8_TRUSTED_PARTITION.json` and the original V8 partition
manifest remain immutable V8 provenance and are not modified, reinterpreted,
or silently extended to cover `T1B` by this draft or any future V8B
artifact.

### 11.2 Requirement

`V8B_HISTORICAL_RESEARCH` must establish a **separate successor authority
chain** before any `T1B` acquisition, rather than reusing
`V8_TRUSTED_PARTITION.json` as-is as authority for a block it does not
name. This is a design requirement only; no code implementing it exists
or is proposed in this document.

### 11.3 Recommended conceptual shape (not implemented)

**A. Immutable parent provenance.** The original V8 partition manifest
(and `V8_TRUSTED_PARTITION.json`'s pin of it) remains untouched and
continues to provide the trusted parent `T_spare` membership/provenance —
i.e., the proof that whichever 300 tickers become `T1B` really are drawn
from the same authorized, self-hash-verified `T_spare` set, under the
same deterministic ordering rule, that V8's own trust anchor already
covers.

**B. A new private V8B allocation artifact.** A successor-study-private
record — conceptually, though not bindingly, named `T1B` allocation
manifest — that binds at minimum:

```text
schema_version=<required>
study_name=V8B_HISTORICAL_RESEARCH
artifact_role=VALIDATION_BLOCK_ALLOCATION
logical_block=T1B
parent_study=V8_HISTORICAL_RESEARCH
parent_v8_partition_manifest_sha256=<the existing authorized_partition_manifest_sha256>
parent_v8_partition_implementation_commit=<the existing authorized_partition_implementation_git_commit>
parent_t_spare_ticker_count=<exact count of the trusted parent T_spare set>
parent_t_spare_ticker_list_sha256=<the existing t_spare_ticker_list_sha256 from the trusted parent manifest>
selection_rule_id=<identifies the exact rule from §4>
selection_rule_canonical_text_or_hash=<the exact rule text or its hash, matching §4's frozen t1b_selection_rule_text>
t1b_offset_within_parent_t_spare=0
t1b_slice_start_inclusive=0
t1b_slice_end_exclusive=300
t1b_ticker_count=300
t1b_ticker_list_sha256=<computed at allocation time>
remaining_t_spare_ticker_count=<exact count, computed at allocation time>
remaining_t_spare_ticker_list_sha256=<exact hash, computed at allocation time>
v8b_frozen_design_commit=<this draft's eventual frozen commit>
v8b_allocation_implementation_commit=<the commit implementing the T1B draw>
created_at_utc=<allocation timestamp>
artifact_self_hash=<self-hash of this artifact, following the same pattern as the existing partition manifest>
```

`remaining_t_spare` is recorded as **both** an exact count and an exact
hash — never a single combined "hash or count" field — precisely so a
future verifier can independently prove both `len(T1B) + len(remaining_T_spare)
= len(original_parent_T_spare)` (§11.4) and the exact identity of what
remains, not merely one or the other.

**Private vs. public artifact boundary.** This (B) artifact is the
**private** layer: it may, and to let a production verifier actually
prove the allocation it must, contain the exact `T1B` and
remaining-`T_spare` ticker assignments themselves, not only their
hashes/counts. It stays outside the public repository, exactly like the
existing private V8 partition manifest, and its ticker-identity contents
are never printed, logged, or otherwise exposed by this draft or any
future document derived from it — only the hash/count fields above are
ever disclosed publicly (see (C) below for the public layer).

**C. A separate V8B trust/authorization artifact.** A public,
repository-fixed artifact — conceptually `V8B_TRUSTED_ALLOCATION.json`,
name not binding if a better schema is justified at implementation time —
that pins the **verified** (B) artifact's `artifact_self_hash` as
`AUTHORIZED`, following the same one-time human-authorization and
independent-review pattern `V8_TRUSTED_PARTITION.json` already
established for V8. This public artifact must contain only safe
metadata — hashes, counts, commit IDs, schema/study/role identifiers,
authorization status, timestamps, and parent-identity pointers — and must
**never** contain `T1B` or `T_spare` ticker identities. A separate human
gate authorizes the pin (§11.4 lists exactly what must be verified before
that gate may fire); the pin is then read from a verified Git object,
never a working-tree-only file, by the production acquisition path —
exactly as `V8_TRUSTED_PARTITION.json` is already read today.

**D. Production binding.** The `T1B` acquisition production path must
bind to this new V8B authority chain (B + C), not to
`V8_TRUSTED_PARTITION.json` alone, and must not treat `T1B` as if it were
the original V8 `T1`. Any code that eventually implements this must
reject a `T1B` acquisition attempt that cannot verify its chain back
through (B) and (C), exactly as the existing code rejects a `T1`/`T2`
acquisition that cannot verify its chain back through
`V8_TRUSTED_PARTITION.json`.

**E. `T2` authority integration — preferred pre-freeze design, pending
human approval.** The independent reviewer's recommendation is
`OPTION_2`, and this draft adopts it as the **preferred design**, not as
an already-approved decision:

```text
v8b_t2_authority_integration_preferred=OPTION_2
v8b_t2_authority_integration_human_approved=false
```

**`OPTION_2` semantics, fully specified.** `T1B` uses the new V8B-specific
allocation authority chain ((A)–(D) above). Existing `T2` continues to
use the **original, immutable V8 partition/trust authority**
(`V8_TRUSTED_PARTITION.json` plus the original V8 partition manifest) —
untouched, unmodified, unre-pinned. `V8_TRUSTED_PARTITION.json` is never
modified by `V8B_HISTORICAL_RESEARCH` under this option. However, `T2`'s
*use* under V8B must still be explicitly bridged to V8B's own study
identity, so that "V8B treats existing `T2` as its sealed holdout" is
itself a recorded, verifiable fact rather than an assumption. The
eventual frozen V8B design, and the official `T2`-opening configuration
built from it, must bind all of:

```text
study=V8B_HISTORICAL_RESEARCH
role=SEALED_HOLDOUT
source_authority=ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY
v8_trust_anchor_git_identity=<the exact Git object identity V8's V8_TRUSTED_PARTITION.json is read from>
authorized_parent_v8_partition_manifest_sha256=<the existing authorized_partition_manifest_sha256>
expected_t2_ticker_list_sha256=<the existing t2_ticker_list_sha256, or a verified derivation of it from the trusted private parent manifest>
t2_acquired_before_authorized_acquisition=false
t2_research_open_count_before_official_opening=0
v8b_frozen_design_commit=<this draft's eventual frozen commit>
t2_membership_reassignment=PROHIBITED
```

The `T2`-opening path (§10's two security requirements resolved first, in
either option) must verify **both**:

```text
A. v8b_frozen_design_explicitly_designates_original_t2_as_its_sealed_holdout=REQUIRED
B. original_t2_still_verifies_through_the_immutable_v8_authority_chain=REQUIRED
```

This bridge records V8B's *claim* on `T2` and re-verifies `T2`'s
*existing* V8 provenance; it never mutates, reinterprets, or re-pins the
V8 trust anchor itself.

**Why `OPTION_2` is preferred (not yet approved).** Least privilege — it
grants `T1B` a new authority scoped only to itself, rather than folding
`T2` into that new scope. It avoids mutating or replacing any part of
V8's existing authority. It keeps the two study-authority scopes
(V8's original `T2` provenance; V8B's new `T1B` provenance) simply
separated, which makes independent audit of either one easier in
isolation. And it leaves `T2`'s existing provenance chain completely
intact and re-verifiable exactly as it stands today.

**`OPTION_1`, kept for auditability, not deleted.**

```text
option_1=one_v8b_study_authority_record_referencing_both_original_t2_and_new_t1b (rejected/non-preferred; retained here for audit history only)
```

`OPTION_1` — one combined V8B study-authority record referencing both the
original `T2` and the new `T1B` — remains structurally sound and is not
deleted from this document's history; it is simply not the design this
draft recommends, because it would require the new V8B authority
artifact itself to become an additional dependency in `T2`'s provenance
chain, where `OPTION_2` keeps that chain exactly as V8 already
established it.

No production code implementing (A)–(E) is written, proposed in diff
form, or scheduled against a specific commit by this document. This
remains a conceptual requirement — with `OPTION_2` as the preferred but
not human-approved design for the `T2` piece — to be resolved, and then
implemented and independently reviewed, before `ONE_TIME_HUMAN_
AUTHORIZATION_TO_ALLOCATE_T1B` in §12's gate sequence.

### 11.4 Required allocation invariants (future verification requirement)

Before any `T1B` allocation may be pinned as trusted (§12's
`READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` gate), a future
independent verifier must prove all of the following against the concrete
allocation artifact produced by allocation — not merely against the
implementation code that produced it:

```text
T1B = original_parent_T_spare[0:300]
remaining_T_spare = original_parent_T_spare[300:]
len(T1B) = 300
len(T1B) + len(remaining_T_spare) = len(original_parent_T_spare)
T1B ∩ remaining_T_spare = ∅
T1B ∪ remaining_T_spare = original_parent_T_spare
T1B is disjoint from T0
T1B is disjoint from old T1
T1B is disjoint from T2
T1B is disjoint from T3
parent_t_spare_ticker_list_sha256 matches the original trusted V8 partition manifest
t1b_ticker_list_sha256 matches the allocation artifact
remaining_t_spare_ticker_list_sha256 matches the allocation artifact
artifact_self_hash validates
selection_rule_canonical_text_or_hash exactly matches the frozen V8B design (§4)
v8b_frozen_design_commit matches the authorized/frozen V8B design
no_membership_choice_based_on_ohlcv_or_data_quality_outcomes=true
```

If verification fails on any single invariant, the result is `BLOCK`: no
pin is created and no acquisition proceeds. This gate exists precisely so
that a defect in the allocation *implementation* — not just a defect in
this *design* — cannot silently produce a `T1B` that violates the
zero-discretion rule §4 already freezes.

---

## 12. V8B gate sequence

```text
V8B_DESIGN_DRAFT                                   <- this document
  ↓
DATA_QUALITY_CALIBRATION_PLAN_APPROVED             (human gate; §6 plan, no run yet)
  ↓
CALIBRATION_IMPLEMENTED_ON_ALLOWED_DATA_ONLY        (T0 / synthetic / provider-doc / independent data only; §6)
  ↓
CALIBRATION_RESULT_REVIEW                          (independent review of calibration output; full distribution, not just chosen point)
  ↓
SUCCESSOR_TRUST_AUTHORITY_MODEL_RESOLVED            (§11 A-E decided, including the §11.3.E T2-integration choice; human decision, not assumed by this draft; the authority model is itself part of what V8B_DESIGN_FINALIZED/HUMAN_DESIGN_FREEZE freeze, so it must resolve first)
  ↓
V8B_DESIGN_FINALIZED                                (Q1 retained, or Q2 with a specific reviewed number; never Q3 by default per §7; includes the now-resolved successor trust/authority model)
  ↓
HUMAN_DESIGN_FREEZE                                 (separate human gate; freezes V8B exactly as V8_HISTORICAL_RESEARCH_DESIGN.md §1 froze V8)
  ↓
T1B_ALLOCATION_IMPLEMENTATION                       (code implementing §4's frozen zero-discretion slice rule and §11.3.B's private allocation artifact schema; fake-only tests)
  ↓
INDEPENDENT_IMPLEMENTATION_REVIEW                   (of the T1B allocation implementation and of §11's new authority-chain implementation code, including §10's two security requirements if they are implemented at this stage)
  ↓
ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B        (separate one-time authorization; consumed on use, per V8's established pattern; authorizes running the reviewed allocation implementation, not yet trusting its output)
  ↓
EXECUTE_T1B_ALLOCATION                              (produces the concrete private §11.3.B allocation artifact against the trusted parent T_spare set; read-only with respect to T0/old T1/T2/T3)
  ↓
READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION      (independent, read-only check of every §11.4 invariant against the concrete artifact just produced, not merely against the implementation code; failure of any invariant = BLOCK, no pin, no acquisition)
  ↓
HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION  (separate human gate; only reachable after §11.4 verification PASSes)
  ↓
CREATE_V8B_TRUSTED_ALLOCATION_PIN                   (publishes the public §11.3.C artifact pinning the verified artifact's self-hash as AUTHORIZED; hashes/counts/commit IDs only, no ticker identities; read from a verified Git object by the production path, never a working-tree-only copy)
  ↓
INDEPENDENT_TRUST_PIN_REVIEW                        (independent review of the published pin artifact and its binding to production code, mirroring V8_TRUSTED_PARTITION.json's own review precedent)
  ↓
T1B_RAW_ACQUISITION_HUMAN_GATE                      (separate one-time authorization; consumed on first Yahoo request regardless of outcome, per V8's established pattern; authorizes acquisition under the now-pinned V8B authority chain, not under V8_TRUSTED_PARTITION.json alone)
  ↓
T1B_RAW_ACQUISITION                                 (real network; exactly one attempt per §5's one-shot rule — a BLOCK here ends V8B, does not trigger a redraw)
  ↓
SEPARATE RESEARCH-OPENING GATE                      (requires §10's two security requirements resolved first)
  ↓
Layer B                                             (V8_HISTORICAL_RESEARCH_DESIGN.md §5.3 rules, unchanged, applied to T1B)
  ↓
FROZEN FINAL CANDIDATE                               (§10 (V8) rules, unchanged)
  ↓
T2 SEALED HOLDOUT GATE                              (requires: §9's T2-reuse conditions still holding, §10's security requirements resolved, and §11.3.E's T2 authority-integration choice resolved)
```

```text
no_real_network_before_the_gates_marked_above=true
each_arrow_is_a_separate_human_or_independent_review_gate=true
no_gate_may_be_skipped=true (V8_HISTORICAL_RESEARCH_DESIGN.md §10.3: skipping_a_stage=false, inherited)
```

---

## 13. Required reuse matrix

| Item | Classification | Justification |
|---|---|---|
| Yahoo Chart parser (`src/v7_yahoo_collector.py::parse_chart_payload` etc.) | `SAFE_TO_REUSE` | Generic, ticker-agnostic, already independently reviewed and accepted under V7 and reused read-only under V8; carries no information about old `T1`'s specific failure. |
| Yahoo transport (`fetch_chart_once`, strict-origin opener) | `SAFE_TO_REUSE` | Same reasoning; transport mechanics are content-independent. |
| Production acquisition security code (Git-provenance pin, trust-anchor read, atomic staging, strict integer invariants) | `REUSE_WITH_CAVEAT` | The provenance/atomicity/integer-invariant machinery is sound and independently reviewed; the two items in §10 (generic guard contract, read-time content binding) must be resolved before this code is relied on for `T1B`/`T2` research-opening, though it is safe as-is for raw acquisition. |
| `P_hist` span | `SAFE_TO_REUSE` | Fixed calendar constant; not touched by, and carries no information from, the acquisition failure. |
| Walk-forward scheme (8 splits) | `SAFE_TO_REUSE` | Same reasoning; methodology-layer constant, never reached by either `T1` attempt. |
| Friction grid | `SAFE_TO_REUSE` | Same reasoning. |
| Promotion thresholds (§8.4, nine gates) | `SAFE_TO_REUSE` | Same reasoning; frozen before any search, unrelated to acquisition-quality outcomes. |
| `T0` | `SAFE_TO_REUSE_FOR_DEVELOPMENT_ONLY` | Already non-evidential by design; unlimited reuse costs nothing (§3.1). |
| Old `T1` (as a block) | `DO_NOT_REUSE` | Retired per §3.2 — carries outcome information that would contaminate any future validation use, even without ticker-level detail. |
| Existing `T2` | `REUSE_WITH_CAVEAT` | Conditionally preservable as V8B's sealed holdout — see full argument in §9; contingent on all seven conditions in §3.3 continuing to hold through design freeze. |
| Existing `T3` | `REUSE_WITH_CAVEAT` | Same conditional preservation as `T2`, held as `SEALED_RESERVE`, not opened (§3.4). |
| Existing `T_spare` | `REUSE_WITH_CAVEAT` | Available for exactly one new block draw (`T1B`) under the one-shot rule (§4–§5); not available for repeated drawing or for any other purpose in this study. |
| Deterministic partition ordering rule (SHA-256(code)-then-code ascending) | `SAFE_TO_REUSE` | Content-independent, deterministic, carries no outcome information; reused verbatim to draw `T1B`. |
| Old V8 partition trust anchor (`V8_TRUSTED_PARTITION.json`, its authorized manifest SHA / implementation commit) | `REUSE_WITH_CAVEAT` | Remains valid, immutable provenance for the *unchanged* `T0`–`T3`/`T_spare` partition it actually names (§3.6) — including as the proof that `T1B`'s parent `T_spare` set is authentic. It is **not** sufficient authority for `T1B` itself: the anchor's `block_assignments` enumerates only `T0`/`T1`/`T2`/`T3`/`T_spare` and contains no `T1B` entry, so it does not, and must not be treated as if it does, authorize a `T1B` acquisition on its own. A separate V8B successor authority chain is required before any `T1B` acquisition — see §11 (design requirement only, not implemented here). |
| Old `T1` failure information (attempt #1: old any-invalid-row fail-closed `MALFORMED_OHLCV` BLOCK, before `POLICY_G_PRIME_V1` existed; attempt #2: `POLICY_G_PRIME_V1` `FRACTION_EXCEEDED` BLOCK, the only real test of the 1% gate) | `DO_NOT_USE_FOR_CALIBRATION` | Retained only as provenance narrative (§0.1, §0.2, §6); explicitly forbidden as an input to any threshold derivation. |

---

## 14. Network / privacy compliance for this document

```text
yahoo_requests_this_document=0
jpx_requests_this_document=0
private_partition_manifest_accessed_this_document=false
t_spare_assignment_contents_read_this_document=false
t2_ticker_identities_read_this_document=false
t3_ticker_identities_read_this_document=false
raw_ohlcv_accessed_this_document=false
partition_build_performed_this_document=false
git_fetch_performed_this_document=true (provenance verification of source V8 HEAD only)
```

No ticker, block-assignment content, or raw data of any kind was read,
inferred, or exposed in the production of this draft. `git fetch origin`
was used only to verify that the source V8 HEAD matches the SHA specified
for this task before branching from it.

---

## 15. What this draft does not decide

For completeness, restating what remains open and requires a future,
separate human decision beyond this draft:

- The exact numeric threshold (if any) that emerges from §6's calibration
  phase.
- Whether `T2`/`T3` preservation (§3.3/§3.4) still holds at the moment
  design actually freezes — this draft only confirms it holds as of the
  reviewed HEAD.
- The concrete code changes implementing §10's two security requirements.
- **The successor trust/authority model for `T1B` (§11)** — this draft
  proposes a conceptual shape (§11.3.A–D) but does not implement it, and
  for the `T2`-integration piece (§11.3.E) states `OPTION_2` as the
  preferred pre-freeze design while explicitly leaving
  `v8b_t2_authority_integration_human_approved=false` —
  `HUMAN_DECISION_REQUIRED_BEFORE_V8B_DESIGN_FREEZE`.
- Any and all real network authorization — none is granted by this
  document.
