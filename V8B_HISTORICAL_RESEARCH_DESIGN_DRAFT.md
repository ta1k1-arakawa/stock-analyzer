# V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT

```text
status=V8B_FINAL_DESIGN_DRAFT_READY_FOR_INDEPENDENT_REVIEW
document_type=DESIGN_DRAFT_ONLY
implementation_performed=false
data_acquisition_performed=false
partition_creation_performed=false
real_network_requests_this_document=0
calibration_result_review=PASS (V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md)
v8b_malformed_ohlcv_policy=Q2 (selected candidate F1_C1; see §7.4)
v8b_successor_trust_authority_model=RESOLVED (OPTION_2; human-approved via V8B_T2_AUTHORITY_INTEGRATION_APPROVAL.json; see §11)
design_frozen=false
human_design_freeze_complete=false
t1b_allocation_authorized=false
```

This is a **design draft**, not a frozen design. Nothing in this document
authorizes acquisition, partition allocation, implementation, or any real
network request. It becomes actionable only by following the full gate
sequence in §12. `DATA_QUALITY_CALIBRATION_PLAN_APPROVED`,
`CALIBRATION_IMPLEMENTED_ON_ALLOWED_DATA_ONLY`, `CALIBRATION_RESULT_REVIEW`,
and `SUCCESSOR_TRUST_AUTHORITY_MODEL_RESOLVED` have all now passed (§6, §7.4,
§11, and the standalone `V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md` /
`V8B_T2_AUTHORITY_INTEGRATION_APPROVAL.json` audit records). The
**immediate next gate** is `FINAL_INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_
DRAFT` — the same gate earlier audit records
(`V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`,
`V8B_T2_AUTHORITY_INTEGRATION_APPROVAL.json`) refer to by its earlier,
shorter name `INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_DRAFT`, now bound to
§12.5's exact-SHA freeze protocol — ahead of `V8B_DESIGN_FINALIZED` and
the separate, still-unreached `HUMAN_DESIGN_FREEZE` gate. Nothing in this
document marks the design frozen, authorizes `T1B` allocation, or performs
any acquisition — those remain gated exactly as §12 already sequences
them.

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
successor_status=V8B_FINAL_DESIGN_DRAFT_READY_FOR_INDEPENDENT_REVIEW
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
| Permanent prohibitions | §12.1 (V8) | none |
| Prohibited claim language | §12.3 (V8) | none, and extended — see §7.5 below |
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
`T2_opened=false`; precisely: no `T2` market-data/raw-OHLCV/feature/
outcome/research content has been opened or observed — partition
membership materialization/hashing, which did occur during original V8
partition build (§9 item 1), is not itself research exposure and is not
what this condition asserts). `T2` is therefore **conditionally
reusable** as V8B's Layer C sealed holdout — see §9 for why old-`T1`'s
failure does not contaminate it.

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
calibration_phase_complete=true
calibration_result_review=PASS
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
   mirroring `V8_HISTORICAL_RESEARCH_DESIGN.md` §9.3's "full trial
   distribution, not only the maximum" requirement.
4. If, after calibration, no defensible independent basis for a specific
   number emerges, V8B's design freezes with `POLICY_G_PRIME_V1` retained
   unchanged (Option Q1, §7) rather than an invented number — **unless**
   the preregistered plan itself, before any calibration result existed,
   predeclared a different scientifically justified no-selection outcome
   (§6.2).

**Current status: all four sub-gates above are `COMPLETE`.** A
calibration preregistration/plan (`V8B_DATA_QUALITY_CALIBRATION_PLAN_V1`)
was written and approved (sub-gate 1); it was implemented and executed
exactly as preregistered (sub-gate 2); its result, including the full
30-candidate distribution, was reviewed independently and PASSed
(sub-gate 3, `V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`); and a
defensible independent basis for a specific number did emerge (candidate
`F1_C1`), so sub-gate 4's `POLICY_G_PRIME_V1`-retained fallback did not
trigger — see §7.4 for the adopted result. `DATA_QUALITY_CALIBRATION_
PLAN_APPROVED` and `CALIBRATION_RESULT_REVIEW` (§12) have both passed.

### 6.1 Calibration preregistration requirements (historical ex-ante contract; now instantiated)

At drafting time, this draft did **not** perform calibration itself and
did **not** invent the numeric candidate grid — fixing that grid was
reserved for a future `DATA_QUALITY_CALIBRATION_PLAN_APPROVED` gate. What
this section fixed, ex ante, is the **shape** every calibration plan had
to satisfy before it could be approved; a plan missing any of the
following would not have been a valid preregistration and would not have
satisfied sub-gate 1 above.

**Current status.** Those shape requirements were subsequently
instantiated by the separately approved `V8B_DATA_QUALITY_CALIBRATION_
PLAN_V1` (approved plan commit `8c15426166742c43745e604f6367788af6123c1a`;
`V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json`). Plan approval,
implementation, real-attempt execution, and the formal `CALIBRATION_
RESULT_REVIEW` are all now `COMPLETE` (`PASS`;
`V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md`,
`V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`). The requirement list
below is retained unchanged, exactly as originally frozen, as the audit
standard the completed plan had to satisfy — it is not weakened,
deleted, or rewritten by this status update.

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

No value for any of these fields was invented or filled in by this draft
at drafting time — they were, and remain, requirements on the *shape* a
calibration plan document must satisfy, not on its content. They were
later instantiated by the separately approved `V8B_DATA_QUALITY_
CALIBRATION_PLAN_V1` (see above). This final design draft does not
retroactively invent or change any of their values; it only records, in
§7.4, the specific values that approved, independently reviewed plan
produced.

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

**PRE-CALIBRATION OPTION DEFINITION / EX-ANTE COMPARISON.** The Q1/Q2/Q3
definitions and bias/reproducibility/feasibility analysis below were
written before calibration ran, to compare the options on their general
shape rather than on any specific number. They are retained unchanged for
audit history. The current adopted outcome — Q2, candidate `F1_C1`,
`invalid_fraction_threshold=1/252`, `max_consecutive_invalid_returned_
rows=1`, `CALIBRATION_RESULT_REVIEW=PASS` — is stated in §7.4, and each
"Outcome" line below cross-references it.

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
  number* (§6.2's fallback rule), not merely because inertia is easiest.
- **Outcome:** not selected. Calibration (§7.4) found a stricter candidate,
  `F1_C1`, independently `DEFENSIBLE`, so `V8B_DATA_QUALITY_CALIBRATION_
  PREREGISTRATION_DRAFT.md` §18's mechanical `STRICTEST_DEFENSIBLE` rule
  selected it over Q1's unchanged threshold.

### Q2 — new, independently-calibrated threshold; membership still frozen before acquisition

```text
invalid_fraction_threshold=<from calibration, if defensible>  [EX-ANTE PLACEHOLDER; resolved value is 1/252 -- §7.4]
max_consecutive_invalid_returned_rows=<from calibration, if defensible>  [EX-ANTE PLACEHOLDER; resolved value is 1 -- §7.4]
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
- **Single-ticker brittleness (ex-ante analysis):** potentially reduced or
  increased depending on the calibrated number's direction and
  rationale — this was not knowable in advance of calibration, which is
  exactly why this draft did not preselect a number ex ante (see the
  resolved outcome below and in §7.4).
- **Operational feasibility (ex-ante analysis):** lower than Q1 —
  required the full calibration phase (§6) and its own independent
  review before design freeze; both are now complete (§7.4,
  `V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`).
- **Explicit non-selection basis:** Q2 is not selected merely because
  `POLICY_G_PRIME_V1` failed on old `T1` — it is selected because the
  calibration phase (§6), run on allowed material only (never old `T1`),
  produced a specific, independently defensible number.
- **Outcome:** selected. §7.4 records the adopted policy:
  `invalid_fraction_threshold=1/252`,
  `max_consecutive_invalid_returned_rows=1` (candidate `F1_C1`), reviewed
  `PASS` in `V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`.

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
  discloses as unresolved, and which that same document's §12.3
  explicitly forbids claiming is "resolved by fresh tickers."
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
calibration_result_review=PASS (V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md)
V8B_MALFORMED_OHLCV_POLICY_ADOPTED=Q2
V8B_MALFORMED_OHLCV_SELECTED_CANDIDATE=F1_C1
V8B_MALFORMED_OHLCV_THRESHOLD=invalid_fraction_threshold=1/252, max_consecutive_invalid_returned_rows=1
V8B_MALFORMED_OHLCV_THRESHOLD_SOURCE=approved V8B data-quality calibration (V8B_DATA_QUALITY_CALIBRATION_PLAN_V1, attempt V8B_CALIBRATION_REAL_ATTEMPT_2); NOT derived from old T1
V8B_MALFORMED_OHLCV_POLICY_Q1_SELECTED=false (a stricter preregistered candidate, F1_C1, was DEFENSIBLE, so V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md §18's STRICTEST_DEFENSIBLE rule mechanically selected it over Q1's unchanged 1%/5 threshold)
V8B_MALFORMED_OHLCV_POLICY_Q3_ADOPTED=false
numeric_threshold_invented_in_this_draft=false
```

§6's calibration phase produced a defensible new number: of the 30
preregistered candidates, all were `DEFENSIBLE` (strict headroom over an
observed calibration envelope of `M_fraction=0/1`, `M_consecutive=0` across
2352 applicable windows), and `V8B_DATA_QUALITY_CALIBRATION_
PREREGISTRATION_DRAFT.md` §18's mechanical `STRICTEST_DEFENSIBLE` rule
selected candidate `F1_C1` (`invalid_fraction_threshold=1/252`,
`max_consecutive_invalid_returned_rows=1`). This is Option Q2, not Q1:
`POLICY_G_PRIME_V1`'s unchanged 1%/5 threshold is not selected, because a
strictly stricter candidate was independently defensible under the frozen
calibration plan — not because of any post-outcome preference for a
stricter number. Option Q3 (pre-partition eligibility screen) remains not
adopted (§7, Q3). No number was invented in this document; the number
above is carried unmodified from the independently reviewed calibration
result (`V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md`,
`V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`).

### 7.5 Extension of prohibited claim language (`V8_HISTORICAL_RESEARCH_DESIGN.md` §12.3 inheritance)

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

### 7.6 `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE` — production policy freeze

§7.4 established that Q2/`F1_C1` is the adopted numeric outcome. This
subsection freezes what that outcome means for **production acquisition**
code, as a distinct V8B policy identity — not a silent renumbering of
`POLICY_G_PRIME_V1`, and not an implicit assumption left for
implementation time to fill in.

```text
policy_name=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE
study=V8B_HISTORICAL_RESEARCH
inherits_semantics_from=V8_HISTORICAL_RESEARCH_DESIGN.md §17 (POLICY_G_PRIME_V1_UNIFORM_RETURNED_ROW_QUALITY_GATE), except the two calibrated numeric thresholds below
```

**Frozen numeric thresholds (source: §7.4, `V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md`, candidate `F1_C1`):**

```text
invalid_fraction_threshold=1/252
invalid_fraction_exact_comparison="invalid_returned_row_count * 252 <= total_returned_row_count"
floating_point_threshold_decision=PROHIBITED
max_consecutive_invalid_returned_rows=1
```

The exact-integer comparison form mirrors the same style already used, and
already independently reviewed, in `src/v8_historical_acquisition.py`'s
`_malformed_ohlcv_check_window()` for V8's own 1% gate (`invalid_count *
100 <= total`), generalized to the calibrated `1/252` rational without
introducing any float rounding. `max_consecutive_invalid_returned_rows=1`
means a run of 2 or more consecutive invalid returned rows BLOCKs (i.e.
`run > 1` triggers the consecutive gate), the same `run > threshold`
semantics already implemented for V8's `max_consecutive=5` case.

**Classifier binding (production must use exactly the calibration
classifier).** `F1_C1` was calibrated against one exact, pinned
parser/classifier Git blob
(`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §4). Production
acquisition must bind to that identical blob, not merely to a
"semantically similar" successor:

```text
canonical_parser_classifier_file=src/v7_yahoo_collector.py
canonical_parser_classifier_git_commit=28e281c3ee30d6b4c2f981c5da3ddc983c09724d
canonical_parser_classifier_blob_sha=76b57b077f3214e666ff9dc06d9c224afc16df9f
classifier_version_binding=EXACT_GIT_BLOB
```

**Production meaning:**

```text
policy_valid_only_with_exact_classifier_blob=true
semantically_similar_different_blob_silently_accepted=PROHIBITED
classifier_change=METHODOLOGICAL_CHANGE (requires CHATGPT_DECISION_REQUIRED before any production use)
```

**Production blocker:**

```text
V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH
```

The future V8B production implementation must verify
`canonical_parser_classifier_blob_sha` before any Yahoo request. This
check must occur before acquisition-gate consumption (i.e. before the
`T1B_RAW_ACQUISITION_HUMAN_GATE` / `T2_RAW_ACQUISITION_HUMAN_GATE`
authorization is treated as consumed), exactly like the §7.7 fail-before-
network ZoneInfo check. This document does not modify
`src/v7_yahoo_collector.py`; it only pins which existing blob production
must verify itself against.

**Returned-row denominator semantics (unchanged from V8 §17 clause 1):**

```text
only_yahoo_returned_timestamped_observations_count=true
expected_missing_calendar_dates_treated_as_malformed=false
pre_listing_absence_treated_as_malformed=false
```

**Window semantics for PRODUCTION acquisition (unchanged from V8 §17 clauses 7-8):**

```text
full_p_hist_check_required=true
per_test_year_checks_required=true
production_test_years=2018,2019,2020,2021,2022,2023,2024,2025
zero_returned_observations_over_full_applicable_series=BLOCK
zero_returned_observations_in_an_individual_year=NOT_APPLICABLE
```

**IMPORTANT — calibration windows are not production windows.** The V8B
data-quality calibration's own observed-window span was `2019-01-01`
through `2025-12-31`, evaluated per calendar year 2019-2025
(`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md` §3) — that span
is **calibration-evidence material only**, used to derive `M_fraction` and
`M_consecutive` and thereby to select the `1/252`/`1` thresholds above. It
is a distinct concept from, and **must not silently replace**, V8 §17's
own production acquisition yearly checks, which remain `2018..2025` (eight
years, including `2018`, which is outside the calibration's evidence
window) exactly as frozen in V8 §17 clause 7 and unchanged by §2's
inheritance table. A future implementation that evaluates production
`T1B`/`T2` acquisition only over `2019..2025` (omitting `2018`) would be
silently narrowing an inherited V8 invariant and is explicitly prohibited
by this section.

**Row handling (unchanged from V8 §17 clauses 2-4, 9-10):**

```text
row_classifier=src/v7_yahoo_collector.py canonical classification semantics (unchanged, read-only reuse)
invalid_returned_rows_may_be_excluded=true
fill_allowed=false
forward_fill_allowed=false
back_fill_allowed=false
interpolation_allowed=false
imputation_allowed=false
alternate_source_substitution_allowed=false
ticker_removal_allowed=false
ticker_replacement_allowed=false
t_spare_replacement_allowed=false
repartition_allowed=false
membership_change_conditional_on_quality_allowed=false
partial_fewer_than_300_publication_allowed=false
threshold_exceedance_action=BLOCK_WHOLE_ACQUISITION
retry_count=0
```

**Scope of application:**

```text
applies_to=[T1B raw acquisition, reused T2 raw acquisition]
applies_retroactively_to_old_V8_T1=false
```

This V8B policy governs `T1B` and `T2` acquisition under `V8B_HISTORICAL_
RESEARCH` only. It does not, and cannot, retroactively apply to old V8
`T1`'s two already-concluded attempts (§0.1, §0.2) — those remain governed
by whatever policy was actually in force when each ran (the older
any-invalid-row rule for attempt #1; `POLICY_G_PRIME_V1` for attempt #2),
exactly as already recorded. This section does not modify
`V8_HISTORICAL_RESEARCH_DESIGN.md` or `POLICY_G_PRIME_V1` in any way; both
remain frozen, immutable V8 provenance.

**Current production code is V8-only / Q1-only and is not sufficient
as-is.** `src/v8_historical_acquisition.py` hardcodes
`MALFORMED_OHLCV_INVALID_FRACTION_THRESHOLD = 0.01` and
`MALFORMED_OHLCV_MAX_CONSECUTIVE_INVALID_RETURNED_ROWS = 5` (V8's Q1
`POLICY_G_PRIME_V1` values, verified by direct reading of that file) and
restricts `ALLOWED_ACQUISITION_BLOCKS` to exactly `("T1", "T2")` — it has
no notion of `T1B`, no `1/252` threshold, and no V8B successor-authority
binding (§11). This design draft record states explicitly:

```text
src_v8_historical_acquisition_py_current_state=V8_ONLY_Q1_ONLY_IMPLEMENTATION
reusable_as_is_for_v8b_acquisition=false
requires_new_or_adapted_v8b_specific_implementation=true
```

A future V8B production acquisition implementation must apply
`POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE` (this section),
not `POLICY_G_PRIME_V1`, and must gate on the new V8B successor authority
chain (§11) for `T1B`, and on the `OPTION_2` bridge (§11.3.E) for `T2` —
see §12's `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_IMPLEMENTATION` gate.

### 7.7 Fail-before-network runtime prerequisite (technical hardening, not methodology)

The already-recorded `V8B_CALIBRATION_REAL_ATTEMPT_1` environment failure
(`V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md`: `run_invalid_reason=
CALIBRATION_CLASSIFIER_VERSION_MISMATCH`, actual cause `Windows Python
ZoneInfo installation lacked Asia/Tokyo timezone data`) is calibration-side
evidence of a real runtime-environment failure mode that also threatens
production acquisition, since the same pinned parser/runtime
(`src/v7_yahoo_collector.py`) depends on `Asia/Tokyo` `ZoneInfo` semantics
for correct trading-date classification.

```text
requirement=fail_closed_before_first_yahoo_request_if_asia_tokyo_zoneinfo_unavailable
applies_to=[T1B raw acquisition, T2 raw acquisition]
check_must_occur_before=[first Yahoo request, consumption of the one-shot acquisition attempt / human-gate authorization]
this_is_runtime_hardening_not_methodology=true
modifies_src_v7_yahoo_collector_py=false (this document does not modify that file; a future implementation adds an environment-prerequisite check ahead of it, not inside it)
```

This is recorded here as a required technical implementation prerequisite
for the future `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_IMPLEMENTATION`
gate (§12) to satisfy — it does not itself change any classification
semantics, threshold, or window rule.

---

## 8. Preferred methodological shape (summary)

```text
unchanged_from_V8=[P_hist, T0_development_role, walk_forward_scheme,
  friction_grid, promotion_thresholds, parser, yahoo_transport,
  deterministic_partition_ordering, existing_untouched_T2, existing_untouched_T3]
changed_from_V8=[study_identity, old_T1_status(retired),
  one_fresh_validation_block(T1B, one_shot_draw),
  acquisition_quality_policy(POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE, frozen per §7.6, calibrated per the section 6 wall),
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
this design's sense (`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.4, and this
draft's own §9) means information flow from an outcome
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
somehow gets acquired or opened, or the universe/partition algorithm
changes), this recommendation is withdrawn automatically. The result,
common to both the pre-freeze recheck (§12.2) and the post-freeze recheck
after `Layer B` (§12.4), is:

```text
V8B_T2_PRESERVATION_RECHECK_BLOCKED
```

**This is fail-closed, not implementation-time discretion, at either
stage.** An earlier revision of this section stated that "a new sealed
block must be sourced from `T_spare` instead, following the same one-shot
rule as §4–§5." That sentence is withdrawn as insufficiently
preregistered: no sealed-holdout `T_spare` offset, allocation rule, or
authority path for a *second* `T_spare`-drawn block is frozen anywhere in
this document (§4's frozen zero-offset slice rule and §11's authority
chain are specified for `T1B` only, not for a hypothetical `T2`
replacement), and inventing one at implementation time — after observing
that `T2` specifically failed preservation — would be exactly the
outcome-conditioned, unregistered block-selection discretion §5's
one-shot rule exists to foreclose, one layer up.

**Stage-dependent action.** The reason string is shared, but what happens
next depends on whether `V8B_DESIGN_FINALIZED`/`HUMAN_DESIGN_FREEZE`
(§12.5) has already occurred for this design.

**PRE-FREEZE** (detected by §12.2's recheck, before
`V8B_DESIGN_FINALIZED`/`HUMAN_DESIGN_FREEZE`):

```text
v8b_design_finalized=PROHIBITED
human_design_freeze=PROHIBITED
automatic_alternate_sealed_holdout_allocation=PROHIBITED
implementation_time_t_spare_offset_choice=PROHIBITED
automatic_t3_reuse_as_replacement=PROHIBITED
holdout_redesign_inside_this_task=PROHIBITED
```

The required result is:

```text
CHATGPT_DECISION_REQUIRED
```

A new explicit design for whatever replaces `T2` as V8B's sealed holdout,
and a new separate human gate approving it, are required before
`V8B_DESIGN_FINALIZED` may proceed. This draft does not resolve that
methodology question and does not authorize any execution agent to
resolve it either.

**POST-FREEZE** (detected by §12.4's recheck, after `Layer B`, i.e. after
`HUMAN_DESIGN_FREEZE` has already occurred for this exact design SHA):

```text
frozen_v8b_design_amended_to_substitute_another_holdout=PROHIBITED
t_spare_replacement=PROHIBITED
t3_automatic_replacement=PROHIBITED
t2_membership_reassignment=PROHIBITED
v8b_confirmatory_path=CLOSES_WITHOUT_A_LAYER_C_RESULT
replacement_holdout_requires=NEW_SUCCESSOR_STUDY_IDENTITY (V8C or equivalent, with its own preregistration/design and human gate)
post_freeze_chatgpt_decision_modifies_frozen_v8b_under_same_study_identity=PROHIBITED
```

Unlike the pre-freeze case, a post-freeze `CHATGPT_DECISION_REQUIRED`
does **not** authorize amending or continuing the already-frozen `V8B`
design under the same study identity — that would be exactly the
post-outcome, after-Layer-C-adjacent-access parameter change §5.4/§0.1
already treat as new-study-triggering, applied at the holdout-failure
layer instead of the acquisition-quality-policy layer. `V8B_HISTORICAL_
RESEARCH`'s own confirmatory (Layer C) path simply closes without a
result; any successor study that wants a different sealed holdout is a
new study, exactly like `V8C` was already contemplated as `V8B`'s own
successor on `V8B_VALIDATION_ACQUISITION_FAIL` (§5).

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
   security remediation. **Extended by this draft:** the official
   research-opening path must reverify, at point of use, **both** (a) the
   acquisition manifest's trusted block/authority binding described
   above, **and** (b) the raw payload bytes' `byte_count`/`SHA-256`
   binding against that same manifest (§12.6's checklist) — a fresh
   re-check performed at the moment of opening, not merely trust in
   `READ_ONLY_T1B_ACQUISITION_ARTIFACT_VERIFICATION`'s or `READ_ONLY_T2_
   ACQUISITION_ARTIFACT_VERIFICATION`'s earlier, point-in-time PASS. This
   prevents a successful post-acquisition verification from becoming
   stale if raw payload files are altered between that verification and
   the later research-opening attempt.

Neither requirement is implemented, designed in code form, or scheduled
against a commit in this document. Both remain **design requirements**
that block the two specific gates named above (§12:
`SEPARATE RESEARCH-OPENING GATE` and
`T2_RESEARCH_OPENING_HUMAN_GATE`/`T2_SEALED_HOLDOUT_GATE`), and do not
block `T1B` partition allocation, `T1B` raw acquisition, or `T2` raw
acquisition themselves, consistent with the prior review's finding that
neither issue is reachable from the raw-acquisition path.

---

## 11. Successor trust/authority model for `T1B`

```text
v8b_successor_trust_authority_model=RESOLVED (§11.3.A-E decided; §11.3.E's T2-integration choice is OPTION_2, human-approved via V8B_T2_AUTHORITY_INTEGRATION_APPROVAL.json)
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

**E. `T2` authority integration — approved.** The independent reviewer's
recommendation was `OPTION_2`. It is now the approved successor
`T2` authority-integration design, per the explicit human decision
`V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED`
(`V8B_T2_AUTHORITY_INTEGRATION_APPROVAL.json`):

```text
v8b_t2_authority_integration=OPTION_2
v8b_t2_authority_integration_human_approved=true
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

**Why `OPTION_2` was chosen.** Least privilege — it
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
form, or scheduled against a specific commit by this document. The design
decision for (A)–(E), including the `T2` piece's `OPTION_2` choice, is now
resolved and human-approved; implementation and independent review of that
implementation remain future work, still required before `ONE_TIME_HUMAN_
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
V8B_DESIGN_DRAFT                                   <- this document, initial draft
  ↓
DATA_QUALITY_CALIBRATION_PLAN_APPROVED             [COMPLETE] (human gate approving the §6 plan itself, before any run; V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json; the plan's subsequent implementation, real-attempt execution, and result review are separately tracked in the next two gates below, both also COMPLETE)
  ↓
CALIBRATION_IMPLEMENTED_ON_ALLOWED_DATA_ONLY        [COMPLETE] (T0 / synthetic / provider-doc / independent data only; §6; V8B_DATA_QUALITY_CALIBRATION_ADJUDICATION.md)
  ↓
CALIBRATION_RESULT_REVIEW                          [COMPLETE: PASS] (independent review of calibration output; full distribution, not just chosen point; V8B_DATA_QUALITY_CALIBRATION_RESULT_REVIEW.md; selected candidate F1_C1, §7.4)
  ↓
SUCCESSOR_TRUST_AUTHORITY_MODEL_RESOLVED            [COMPLETE] (§11 A-E decided, including the §11.3.E T2-integration choice; human decision, not assumed by this draft; OPTION_2 approved via V8B_T2_AUTHORITY_INTEGRATION_APPROVAL.json; the authority model is itself part of what V8B_DESIGN_FINALIZED/HUMAN_DESIGN_FREEZE freeze, so it had to resolve first)
  ↓
V8B_FINAL_DESIGN_DRAFT_READY_FOR_INDEPENDENT_REVIEW [CURRENT POSITION] (this document, as updated; calibration result and successor authority model both incorporated; not yet V8B_DESIGN_FINALIZED)
  ↓
FINAL_INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_DRAFT  (independent review of this updated draft as a whole, reviewing one exact design commit SHA per §12.5; not yet performed; earlier audit records refer to this gate by its shorter name INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_DRAFT)
  ↓
READ_ONLY_TSPARE_T2_T3_PRESERVATION_RECHECK         (repository-safe recheck of §3.3/§3.4/§3.5 conditions, reviewing the SAME exact design commit SHA as the gate above per §12.5 -- see §12.2; not yet performed; must PASS before V8B_DESIGN_FINALIZED, and must be repeated at the actual freeze candidate HEAD, not satisfied merely because an earlier HEAD looked fine)
  ↓
V8B_DESIGN_FINALIZED                                (Q2 with the specific reviewed number F1_C1 adopted; never Q3 by default per §7; includes the now-resolved successor trust/authority model and a PASSing §12.2 preservation recheck; proceeds only if both gates above PASSed for the same exact SHA, per §12.5; does not silently rewrite the design body)
  ↓
HUMAN_DESIGN_FREEZE                                 (separate human gate, not yet reached; must explicitly name the exact 40-hex frozen design commit SHA per §12.5 -- approval of "the current branch"/"latest HEAD" is insufficient; freezes V8B exactly as V8_HISTORICAL_RESEARCH_DESIGN.md §1 froze V8)
  ↓
V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_IMPLEMENTATION (implements the whole V8B production boundary: T1B deterministic allocation, T1B successor authority chain, T1B raw acquisition under POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE (§7.6), T2 raw acquisition under the same policy, the OPTION_2 T2 bridge, exact block/role/study/design-commit/content bindings, privacy-safe failure behavior, one-shot human-gate consumption behavior, and the §7.7 fail-before-network runtime prerequisite; fake-only tests; no real Yahoo/JPX access)
  ↓
INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW    (independent review of that concrete implementation -- see §12.3 for the required explicit checklist; §10's two research-opening security requirements may remain separate later implementation work if not done in this phase, but must be complete before either T1B or T2 RESEARCH opening)
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
T1B_RAW_ACQUISITION_HUMAN_GATE                      (separate one-time authorization; consumed on first Yahoo request regardless of outcome, per V8's established pattern; authorizes acquisition under the now-pinned V8B authority chain, not under V8_TRUSTED_PARTITION.json alone; T1B authorization never authorizes T2)
  ↓
T1B_RAW_ACQUISITION                                 (real network; exactly one attempt per §5's one-shot rule, under POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE §7.6 -- a BLOCK here ends V8B, does not trigger a redraw)
  ↓
READ_ONLY_T1B_ACQUISITION_ARTIFACT_VERIFICATION     (independent, read-only data-integrity check of the concrete successful T1B raw acquisition bundle -- see §12.6 for the full checklist shared with T2; failure = BLOCK, no research opening)
  ↓
SEPARATE RESEARCH-OPENING GATE                      (requires §10's two security requirements resolved first, including §10's raw-byte rebinding at point of use)
  ↓
Layer B                                             (V8_HISTORICAL_RESEARCH_DESIGN.md §5.3 rules, unchanged, applied to T1B)
  ↓
FROZEN FINAL CANDIDATE                               (§10 (V8) rules, unchanged)
  ↓
READ_ONLY_T2_REUSE_CONDITIONS_RECHECK               (repository-safe recheck of §3.3/§9's T2 preservation conditions at this point in time -- see §12.4; safe metadata only, no T2 identities exposed; any condition failing => V8B_T2_PRESERVATION_RECHECK_BLOCKED per §9, not a silent T_spare substitution)
  ↓
T2_RAW_ACQUISITION_HUMAN_GATE                       (separate, explicit, one-time human authorization -- see §12.4; T1B authorization never authorizes T2; pre-network technical failure does not consume it; consumed once the first Yahoo request is made)
  ↓
T2_RAW_ACQUISITION                                  (real network; original immutable V8 T2 membership/provenance, explicit OPTION_2 bridge (§11.3.E), POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE §7.6; remains sealed; no feature/outcome/research access; failure => V8B_T2_RAW_ACQUISITION_FAIL per §12.4, not a strategy/model/profitability/Layer-C conclusion)
  ↓
READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION      (independent, read-only check that the concrete artifact binds to V8B study identity, the frozen V8B design commit, the original immutable V8 T2 authority, the OPTION_2 bridge, correct T2 ticker-list hash/provenance, F1_C1 policy metadata, sealed/raw state, and zero research/open counters -- data-integrity checklist shared with T1B at §12.6; see also §12.4; failure = BLOCK)
  ↓
T2_RESEARCH_OPENING_HUMAN_GATE / T2_SEALED_HOLDOUT_GATE (separate human gate; only reachable after READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION PASSes; requires §9's T2-reuse conditions still holding, §10's security requirements resolved, and §11.3.E's OPTION_2 bridge verified)
  ↓
Layer C one-shot evaluation                         (V8_HISTORICAL_RESEARCH_DESIGN.md sealed-holdout rules, unchanged, applied to T2 as V8B's sealed holdout)
```

```text
no_real_network_before_the_gates_marked_above=true
each_named_human_gate_is_separate=true
each_named_independent_review_gate_is_separate=true
authorization_for_one_gate_never_authorizes_a_later_gate=true
no_gate_or_required_stage_may_be_skipped=true (V8_HISTORICAL_RESEARCH_DESIGN.md §10.3: skipping_a_stage=false, inherited)
```

### 12.1 `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_IMPLEMENTATION` — required coverage

Before any real `T1B` allocation/acquisition authorization, this future
implementation phase must cover at least:

```text
A. T1B deterministic allocation (§4's frozen zero-offset slice rule, §11.3.B's private allocation artifact schema)
B. T1B successor authority chain (§11.3.A-D)
C. T1B raw acquisition under POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE (§7.6)
D. T2 raw acquisition under the same V8B F1_C1 quality policy (§7.6)
E. the OPTION_2 T2 bridge to the original immutable V8 T2 authority (§11.3.E)
F. exact block/role/study/design-commit/content bindings (mirroring the existing V8 acquisition manifest invariants, extended for T1B and the OPTION_2 bridge)
G. privacy-safe failure behavior (no ticker identity, date, or raw payload content in any error/log/exit path -- mirroring the existing generic, ticker-free MALFORMED_OHLCV_QUALITY_GATE reason strings)
H. one-shot human-gate consumption behavior (each authorization named in §12's diagram is consumed on its own first real action, never reused for a later gate)
I. the §7.7 fail-before-network runtime prerequisite (Asia/Tokyo ZoneInfo availability check before any Yahoo request)
J. the §7.6 classifier-binding check (canonical_parser_classifier_blob_sha=76b57b077f3214e666ff9dc06d9c224afc16df9f verified before any Yahoo request; mismatch => V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH)
```

This is a single implementation phase covering the whole production
boundary (both `T1B` and `T2` acquisition code paths); it does not itself
consume any of the separate one-time human authorizations named later in
§12's diagram, and it performs no real Yahoo/JPX access.

### 12.2 `READ_ONLY_TSPARE_T2_T3_PRESERVATION_RECHECK` — required verification

Renamed and extended from the prior `READ_ONLY_T2_T3_PRESERVATION_
RECHECK` to make explicit that `T_spare` freshness -- not only `T2`/`T3`
preservation -- must be rechecked before freeze. Every cross-reference to
the old name in this document has been updated; the gate's substance
(repository-safe, no ticker identities, BLOCK on failure) is unchanged.

On the exact candidate design SHA being considered for freeze, using safe
committed state/trust metadata only (no private ticker identities;
identity-exposure checks below mean checking safe audit/state
flags/provenance fields, **never** printing or reading the actual
ticker assignments):

```text
T_spare: parent_t_spare_ticker_count=1904
T_spare: parent_t_spare_ticker_list_sha256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
T_spare: ticker_identities_exposed_to_human_public_research_loop=false
T_spare: ohlcv_acquisition_occurred=false
T_spare: feature_outcome_research_use_occurred=false
T_spare: t1b_allocation_occurred=false
T_spare: original_parent_membership_provenance_unchanged=true

T2: acquired=false
T2: opened=false
T2: ticker_identities_exposed_to_human_public_research_loop=false
T2: market_data_raw_ohlcv_feature_outcome_research_exposure=false
T2: universe_definition_unchanged=true
T2: partition_algorithm_unchanged=true
T2: v8b_f1_c1_policy_fixed=true (§7.6)

T3: acquired=false
T3: opened=false
T3: remains_SEALED_RESERVE=true
T3: ticker_identities_exposed_to_human_public_research_loop=false
T3: market_data_raw_ohlcv_feature_outcome_research_exposure=false
```

**Absence of evidence is not evidence of PASS.** If any required fact
above cannot be established from permitted safe metadata (audit/state
flags, committed hashes/counts, trust-anchor provenance), the result is
`BLOCK` -- a missing or unreadable safe-metadata field is never treated
as an implicit PASS.

If this check cannot PASS, the design must **not** be finalized or
frozen. This gate is not satisfied by a prior favorable read at an
earlier HEAD; it must be repeated, bound to the exact SHA actually
proposed for `V8B_DESIGN_FINALIZED`/`HUMAN_DESIGN_FREEZE` (§12.5). This
document does **not** mark this gate complete -- it only specifies what a
future, freeze-time recheck must verify. If the `T2`/`T3` portion fails,
§9's pre-freeze `V8B_T2_PRESERVATION_RECHECK_BLOCKED` /
`CHATGPT_DECISION_REQUIRED` outcome governs, not a silent substitution.
This task does not perform this recheck.

### 12.3 `INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` — required explicit verification

```text
1/252_implemented_by_exact_integer_rational_comparison=REQUIRED (invalid_returned_row_count * 252 <= total_returned_row_count; no floating-point threshold decision)
max_consecutive_equals_1=REQUIRED
production_classifier_exact_blob_match=REQUIRED (canonical_parser_classifier_blob_sha=76b57b077f3214e666ff9dc06d9c224afc16df9f, verified before any Yahoo request; mismatch => V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH, §7.6)
production_years_are_2018_through_2025_plus_full_p_hist=REQUIRED (not the calibration-evidence 2019-2025 span -- §7.6)
no_threshold_grid_window_caller_override_exists=REQUIRED
t1b_cannot_fall_back_to_old_t1_semantics=REQUIRED
t2_cannot_bypass_option_2_bridge=REQUIRED
no_v8_trust_anchor_mutated_or_repinned=REQUIRED
no_ticker_identity_leaks_via_public_errors_or_logs=REQUIRED
retry_count_equals_0=REQUIRED
tests_are_fake_synthetic_only=REQUIRED
no_real_yahoo_or_jpx_access_during_implementation_or_review=REQUIRED
```

Failure of any item above blocks this gate; the implementation must be
remediated and independently re-reviewed before any real allocation or
network authorization may proceed.

### 12.4 `T2` raw acquisition sequence — detailed rules

`T2` under V8B is **not yet acquired**. The following rules bind the
`READ_ONLY_T2_REUSE_CONDITIONS_RECHECK` → `T2_RAW_ACQUISITION_HUMAN_GATE`
→ `T2_RAW_ACQUISITION` → `READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION`
→ `T2_RESEARCH_OPENING_HUMAN_GATE`/`T2_SEALED_HOLDOUT_GATE` → Layer C
sequence in §12's diagram, positioned after `Layer B` and
`FROZEN FINAL CANDIDATE`.

**`READ_ONLY_T2_REUSE_CONDITIONS_RECHECK`:**

```text
recheck_stage=POST-FREEZE (occurs after Layer B / HUMAN_DESIGN_FREEZE, per §9's stage-dependent action)
recheck_scope=all §3.3 / §9 preservation conditions
data_used=safe repository/trust metadata only
t2_identities_exposed=false
on_any_condition_failing=BLOCK (V8B_T2_PRESERVATION_RECHECK_BLOCKED, §9 POST-FREEZE action -- confirmatory path closes, no holdout substitution under this study identity)
silent_t2_replacement=PROHIBITED
```

**`T2_RAW_ACQUISITION_HUMAN_GATE`:**

```text
authorization=separate, explicit, human
t1b_authorization_authorizes_t2=false
authorization_shape=one-shot
pre_network_technical_failure_counts_as_yahoo_attempt=false
authorization_consumed_at=first Yahoo request
```

**`T2_RAW_ACQUISITION`:**

```text
membership_provenance=original immutable V8 T2 membership/provenance
authority_bridge=explicit OPTION_2 bridge (§11.3.E)
quality_policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE (§7.6)
post_acquisition_state=remains sealed
feature_outcome_research_access=none
```

**On failure after network has begun:**

```text
V8B_T2_RAW_ACQUISITION_FAIL
```

```text
automatic_retry=PROHIBITED
manual_retry_inside_v8b=PROHIBITED
redraw=PROHIBITED
t_spare_replacement=PROHIBITED
threshold_change=PROHIBITED
alternate_source=PROHIBITED
layer_c_strategy_conclusion_claimed=PROHIBITED
v8b_confirmatory_path=CLOSES_AT_ACQUISITION_FAILURE
```

`V8B_T2_RAW_ACQUISITION_FAIL` must **not** be reinterpreted as a strategy
failure, model failure, profitability evidence, or a Layer C result --
the same non-reinterpretation discipline `V8_STATE.json`'s
`t1_raw_acquisition_attempt_history[].prohibited_reinterpretations`
already applies to V8's own acquisition-quality BLOCKs.

**`READ_ONLY_T2_ACQUISITION_ARTIFACT_VERIFICATION`** must verify the
concrete artifact is bound to, in addition to §12.6's full data-integrity
checklist (shared verbatim with `READ_ONLY_T1B_ACQUISITION_ARTIFACT_
VERIFICATION`):

```text
v8b_study_identity=REQUIRED
frozen_v8b_design_commit=REQUIRED
original_immutable_v8_t2_authority=REQUIRED
option_2_bridge=REQUIRED
correct_t2_ticker_list_hash_provenance=REQUIRED
f1_c1_policy_metadata=REQUIRED (§7.6)
sealed_raw_state=REQUIRED
zero_research_open_counters=REQUIRED
```

These items are `T2`-authority-specific (verifying the `OPTION_2` bridge
and original immutable V8 authority rather than the new `T1B` allocation
authority chain); §12.6 supplies the block-agnostic data-integrity checks
common to both blocks.

Only after this verification PASSes may the separate
`T2_RESEARCH_OPENING_HUMAN_GATE`/`T2_SEALED_HOLDOUT_GATE` occur.

### 12.5 Design-freeze Git binding protocol (exact-SHA binding)

The design freeze must authorize exactly one 40-hex Git commit SHA, never
a moving branch reference. This subsection freezes the binding protocol
now; it does not perform any step of it.

```text
A. FINAL_INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_DRAFT reviews one exact design commit SHA.
B. READ_ONLY_TSPARE_T2_T3_PRESERVATION_RECHECK (§12.2) reviews the SAME exact design commit SHA.
C. V8B_DESIGN_FINALIZED may proceed only if both A and B PASS for that same SHA. It does not silently rewrite the design body.
D. HUMAN_DESIGN_FREEZE must explicitly name that exact 40-hex design commit SHA. Approval of "the current branch" / "latest HEAD" is insufficient.
E. After explicit human approval, a separate repository audit artifact is created -- conceptually V8B_DESIGN_FREEZE_APPROVAL.json.
```

**(E) is specified now, not created now.** Its future minimum fields:

```text
schema_version=V8B_DESIGN_FREEZE_APPROVAL_V1
study=V8B_HISTORICAL_RESEARCH
frozen_design_git_commit=<exact approved 40-hex SHA>
design_document=V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md
final_independent_review_result=PASS
final_independent_review_design_commit=<same SHA>
preservation_recheck_result=PASS
preservation_recheck_design_commit=<same SHA>
approval_status=APPROVED
human_gate=<exact human approval string naming the same SHA>
```

**The freeze-approval commit is not the frozen design commit.** The
commit that later adds `V8B_DESIGN_FREEZE_APPROVAL.json` to the
repository is **not** itself the frozen design commit — the frozen design
commit is the exact, earlier design SHA the human actually approved,
which necessarily predates the artifact commit that records the
approval. This task does not create that artifact and does not perform
`HUMAN_DESIGN_FREEZE`.

**Any semantic change to the design document after (A) or (B):**

```text
invalidates_the_prior_review_or_recheck=true
requires_a_new_candidate_sha=true
requires_repeat_final_independent_review=true
requires_repeat_preservation_recheck=true
requires_a_new_exact_human_freeze_approval=true
```

**Any semantic design change after `HUMAN_DESIGN_FREEZE`:**

```text
NEW_STUDY_REQUIRED
```

unless the frozen design itself explicitly defines a permitted
append-only, non-methodological clarification path (mirroring
`V8_HISTORICAL_RESEARCH_DESIGN.md` §17's own append-only-erratum
precedent, §0.1).

**Future V8B production authority must bind to the exact
`frozen_design_git_commit` recorded in the (E) approval artifact** —
mirroring how `V8_TRUSTED_PARTITION.json` already binds production
acquisition to one exact, pinned partition-build commit rather than to a
branch.

### 12.6 Raw acquisition artifact verification — required integrity checks (`T1B` and `T2`)

`READ_ONLY_T1B_ACQUISITION_ARTIFACT_VERIFICATION` and `READ_ONLY_T2_
ACQUISITION_ARTIFACT_VERIFICATION` (§12's diagram) share this checklist.
Both are **data-integrity checks only**.

```text
calculates_features=PROHIBITED
calculates_strategy_results=PROHIBITED
calculates_profit_or_trades=PROHIBITED
calculates_any_other_research_outcome=PROHIBITED
```

For each successful `T1B`/`T2` acquisition artifact, independently verify:

```text
v8b_study_identity=REQUIRED
logical_block_and_role=REQUIRED (correct)
frozen_v8b_design_commit=REQUIRED (exact)
reviewed_production_implementation_commit=REQUIRED (exact)
authority_chain=REQUIRED (T1B: V8B successor allocation authority, §11; T2: original immutable V8 authority + OPTION_2 bridge, §11.3.E)
ticker_count=300
request_start=2016-04-01
request_end_exclusive=2026-01-01
yahoo_source_host_schema=REQUIRED (matches frozen design)
retry_count=0
request_count=300 (on a successful complete acquisition)
success_transport_count=300
f1_c1_policy_metadata=REQUIRED (exact -- §7.6)
production_years=2018,2019,2020,2021,2022,2023,2024,2025 plus full P_hist (§7.6; not the calibration-evidence 2019-2025 span)
sealed_raw_access_counter_invariants=REQUIRED (appropriate to the block: T1B raw-acquired-not-opened; T2 raw-acquired-sealed)
payload_manifest_record_count=300 (exactly)
stored_payload_manifest_hash=REQUIRED (validates)
raw_payload_files_present=REQUIRED (exactly the 300 manifest-designated files exist)
unexpected_extra_raw_payload_in_block_raw_directory=PROHIBITED
per_payload_byte_count_matches_manifest=REQUIRED (every file)
per_payload_sha256_matches_manifest=REQUIRED (every file)
missing_or_unreadable_payload=BLOCK
ticker_identity_path_raw_payload_ohlcv_value_emitted_publicly=PROHIBITED
```

**Read scope.** Verification may internally read the private raw bytes
solely to compute byte counts/SHA-256; it must not parse those bytes into
OHLCV for research. The public verification result contains aggregate
status/counts/hashes only.

**Any mismatch on any item above:**

```text
BLOCK
no_research_opening=true
```

This gate exists precisely so that a successful acquisition (data
transported without a `MALFORMED_OHLCV_QUALITY_GATE`/classifier/policy
BLOCK) cannot be assumed intact by the time research opening is
attempted — see §10's extended read-time content-binding requirement,
which additionally re-verifies raw-byte binding at the point research
opening is actually used, so this verification's result cannot go stale
if files are altered afterward.

Neither gate is implemented by this document; both remain design
requirements for the future `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_
IMPLEMENTATION` phase (§12.1) and `INDEPENDENT_V8B_PRODUCTION_
IMPLEMENTATION_REVIEW` (§12.3).

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
separate decision beyond this draft. Two items from the prior revision —
the calibrated numeric threshold and the `T2`-integration authority
choice — are now resolved (§7.4, §11.3.E) and are removed from this list;
what remains open is:

- Whether `T_spare` freshness and `T2`/`T3` preservation (§3.3/§3.4/§3.5)
  still hold at the moment design actually freezes — this draft only
  confirms they hold as of the reviewed HEAD. §12.2's `READ_ONLY_TSPARE_
  T2_T3_PRESERVATION_RECHECK` gate fixes what a future, freeze-time
  recheck must verify, bound to the exact freeze-candidate SHA; it has
  not been performed yet and is not satisfied by this document's own
  as-of-reviewed-HEAD confirmation.
- If that recheck's `T2` portion instead fails **before** freeze,
  resolving what replaces `T2` as V8B's sealed holdout — §9's pre-freeze
  `V8B_T2_PRESERVATION_RECHECK_BLOCKED` outcome requires a new explicit
  design and a new human gate (`CHATGPT_DECISION_REQUIRED`); this draft
  does not resolve that hypothetical case and is not authorized to. If
  the equivalent §12.4 recheck instead fails **after** freeze (post
  `HUMAN_DESIGN_FREEZE`, including after `Layer B`), §9's post-freeze
  rule governs instead: the frozen design is not amended, no holdout is
  substituted under the same study identity, and `V8B_HISTORICAL_
  RESEARCH`'s confirmatory path closes without a Layer C result — any
  replacement holdout requires a genuinely new successor study identity
  (`V8C` or equivalent), with its own preregistration/design and human
  gate.
- The concrete code changes implementing §10's two security requirements.
- **The concrete implementation of the whole V8B production boundary**
  (§12.1's `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_IMPLEMENTATION`:
  `T1B` allocation, `T1B`'s successor authority chain (§11.3.A–D, and the
  now-approved §11.3.E `OPTION_2` bridge), and both `T1B`/`T2` raw
  acquisition under `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_
  GATE`, §7.6) — the design decisions are resolved and, where required,
  human-approved, but no production code implementing any of this exists
  yet; implementation and the separate `INDEPENDENT_V8B_PRODUCTION_
  IMPLEMENTATION_REVIEW` (§12.3) remain future work, required before
  `ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B` (§12).
- Independent review of this updated final design draft itself
  (`FINAL_INDEPENDENT_REVIEW_OF_V8B_FINAL_DESIGN_DRAFT`, §12/§12.5), and
  the separate `V8B_DESIGN_FINALIZED` / `HUMAN_DESIGN_FREEZE` human gates
  that follow it — none of which this draft performs or grants. Nor does
  it perform the future `V8B_DESIGN_FREEZE_APPROVAL.json` artifact §12.5
  specifies; that artifact's minimum schema is frozen here, not created.
- Any and all real network authorization, `T1B`/`T2` allocation, or
  acquisition — none is granted by this document.
