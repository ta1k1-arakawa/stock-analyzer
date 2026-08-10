# V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT

```text
status=DRAFT_AWAITING_HUMAN_GATE
document_type=DESIGN_DRAFT_ONLY
implementation_performed=false
data_acquisition_performed=false
partition_creation_performed=false
real_network_requests_this_document=0
```

This is a **design draft**, not a frozen design. Nothing in this document
authorizes acquisition, partition allocation, implementation, or any real
network request. It becomes actionable only after the gate sequence in §11
is followed, starting with a separate human design-freeze decision.

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
research opening). But the acquisition-quality policy that gated `T1` is
upstream of, and structurally equivalent to, the kind of frozen parameter
those clauses protect: it was preregistered (`V8_HISTORICAL_RESEARCH_DESIGN.md`
§17) before either real `T1` attempt, and both attempts BLOCKed under it.
Revising it now, on the same study identity, would be indistinguishable
from exactly the kind of post-outcome parameter change §5.4 exists to
prevent. This draft therefore treats **any acquisition-quality policy
change following a `T1` acquisition outcome as new-study-triggering**,
even though the letter of §5.4 only names Layer C. `V8B_HISTORICAL_RESEARCH`
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
| Promotion gate sequence and invariants | §10 | none, except the new pre-`T1B` calibration sub-gate this draft adds (§11) |
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
a real fact about it: that it fails a specific, frozen malformed-row gate
at some (unknown) ticker/window, on two independent real attempts. Per
`V8_HISTORICAL_RESEARCH_DESIGN.md` §9.4, "a crashed or aborted run that
nevertheless exposed any outcome statistic counts as an access; only a run
that provably produced no outcome information... does not." A BLOCK that
reveals "this specific 300-ticker set has at least one member failing this
specific threshold" is outcome information, even without the ticker's
identity. Old `T1` can therefore never again serve as a blind confirmatory
block for any policy whose numeric thresholds are chosen with knowledge of
that outcome. It is retired from validation use in V8B and is not reused
as `T1B` under any name.

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
token). The same four preservation conditions in §3.3 apply and hold. `T3`
is preserved as `SEALED_RESERVE` under V8B on the same terms as under V8
(`V8_HISTORICAL_RESEARCH_DESIGN.md` §5.4, Decision 6): not used in initial
V8B, not opened for any purpose, release requires a separate future human
gate.

### 3.5 Existing `T_spare` — available for exactly one new validation block

Untouched, never read for assignment content by this draft or any prior
review. Available under §4/§5 below to source exactly one new 300-ticker
validation block, and for nothing else in this design (no repeated
drawing, no per-ticker replacement — see §5).

### 3.6 Why this is not "one new full partition"

A full new partition (redrawing `T0`–`T3`/`T_spare` from the universe from
scratch) is **not** required, because the condition that would force it —
a change to the universe definition or the deterministic partition
algorithm itself (§3.3's first two conditions) — is not proposed anywhere
in this draft. Only `T1`'s *role* changes (old `T1` retired, one fresh
block drawn from `T_spare` under the *same* ordering rule). `T2`/`T3`
membership, having never been read, computed over, or otherwise accessed
by anything, are unaffected set members under the same partition; nothing
about them needs to be redrawn.

---

## 4. New validation block: `T1B`

```text
t1b_source=T_spare (existing, untouched)
t1b_size=300
t1b_selection_rule=DETERMINISTIC_PREDECLARED
t1b_selection_rule_text="first 300 members of current T_spare under the existing frozen deterministic ordering (V8_HISTORICAL_RESEARCH_DESIGN.md §5.1: sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending), taken in that order starting immediately after the existing T3 boundary"
t1b_selection_conditional_on_data_quality=false
t1b_ticker_identities_exposed_in_this_document=false
t1b_ticker_identities_exposed_at_design_freeze=false
old_t1_replaced_ticker_by_ticker=false
old_t1_retired_wholesale=true
```

`T1B` is a **new, distinct logical block**, not a repair of old `T1`. Old
`T1`'s 300-member set is retired in its entirety (§3.2); no member of old
`T1` is carried into `T1B`, and no single failing ticker inside old `T1`
is swapped out. `T1B` is drawn once, mechanically, from the region of
`T_spare` immediately following the existing `T3` boundary, using the same
ordering rule already frozen for `T0`–`T3` — a rule that has no knowledge
of, and no dependency on, which ticker or ticker(s) caused old `T1`'s
BLOCK. This document does not read or expose which tickers land in `T1B`;
that remains a matter for the implementation phase (§11), after design
freeze, exactly as `T0`–`T3` assignment contents were never exposed by
`V8_HISTORICAL_RESEARCH_DESIGN.md` itself.

The exact boundary offset (how many `T_spare` members precede `T1B`'s
first member) is an implementation detail to be fixed at the partition-
allocation implementation step (§11), not invented in this draft, so that
the rule stays mechanical rather than requiring this document to encode
private partition internals.

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

The single fact **"the old 1% gate was reached and BLOCKed on real V8
`T1` data, twice, under two independently blind-derived reviews"** may be
retained and cited as provenance narrative (exactly as this draft already
does in §0.2) but **must not** enter any numeric derivation of a V8B
threshold. A calibration record that cannot demonstrate its output number
was reachable without consulting that fact does not satisfy this section.

**Calibration phase sub-gates** (detail of §11's
`DATA_QUALITY_CALIBRATION_PLAN_APPROVED` → `CALIBRATION_RESULT_REVIEW` span):

1. A calibration plan is written stating which of A–D it will use, and
   exactly what it will measure, **before** any calibration run.
2. The calibration plan is implemented using only the allowed material.
3. The calibration result — including the full distribution of outcomes
   at multiple candidate thresholds, not just the one selected — is
   reviewed independently before it is adopted into V8B's frozen policy,
   mirroring §9.3's "full trial distribution, not only the maximum"
   requirement.
4. If, after calibration, no defensible independent basis for a specific
   number emerges, V8B's design freezes with `POLICY_G_PRIME_V1` retained
   unchanged (Option Q1, §7) rather than an invented number.

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
  300-ticker block on one non-compliant ticker. This is the property that
  caused both V8 BLOCKs and is not resolved by keeping the threshold as
  is.
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
V8's `T1` attempts produced is: *"under `POLICY_G_PRIME_V1`, at least one
member of the old-`T1` 300-ticker set fails the 1% gate."* This is a
statement about **old `T1`'s specific membership**, drawn under the same
deterministic rule as every other block but occupying a disjoint position
in the ordering from `T2`. It carries no information about which,
if any, `T2` members would or would not pass any gate, because:

1. `T2`'s membership has never been read, hashed against, or otherwise
   touched by any process that also touched old `T1`'s data (only its
   *identity as a block name* appears in shared docs — its content does
   not).
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

**If any one of the four conditions in §3.3 stops holding** (e.g., `T2`
somehow gets acquired or opened before V8B's design freezes, or the
universe/partition algorithm changes), this recommendation is withdrawn
automatically and a new sealed block must be sourced from `T_spare`
instead, following the same one-shot rule as §4–§5.

---

## 10. Security requirements retained as design requirements (not implemented here)

Carried forward, unresolved, exactly as identified in the prior
independent security review and the prior design-support review:

```text
requirement_1=generic_open_for_star_arbitrary_mapping_contract_must_be_bound
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
that block the two specific gates named above (§11:
`SEPARATE RESEARCH-OPENING GATE` and `T2 SEALED HOLDOUT GATE`), and do
not block `T1B` partition allocation or `T1B` raw acquisition themselves,
consistent with the prior review's finding that neither issue is
reachable from the raw-acquisition path.

---

## 11. V8B gate sequence

```text
V8B_DESIGN_DRAFT                                   <- this document
  ↓
DATA_QUALITY_CALIBRATION_PLAN_APPROVED             (human gate; §6 plan, no run yet)
  ↓
CALIBRATION_IMPLEMENTED_ON_ALLOWED_DATA_ONLY        (T0 / synthetic / provider-doc / independent data only; §6)
  ↓
CALIBRATION_RESULT_REVIEW                          (independent review of calibration output; full distribution, not just chosen point)
  ↓
V8B_DESIGN_FINALIZED                                (Q1 retained, or Q2 with a specific reviewed number; never Q3 by default per §7)
  ↓
HUMAN_DESIGN_FREEZE                                 (separate human gate; freezes V8B exactly as V8_HISTORICAL_RESEARCH_DESIGN.md §1 froze V8)
  ↓
T1B PARTITION-ALLOCATION IMPLEMENTATION             (code implementing §4's deterministic draw; fake-only tests)
  ↓
INDEPENDENT REVIEW                                  (of the T1B allocation implementation, and of §10's two security requirements if they are implemented at this stage)
  ↓
ONE-TIME HUMAN AUTHORIZATION TO ALLOCATE T1B        (separate one-time authorization; consumed on use, per V8's established pattern)
  ↓
T1B RAW ACQUISITION HUMAN GATE                      (separate one-time authorization; consumed on first Yahoo request regardless of outcome, per V8's established pattern)
  ↓
T1B RAW ACQUISITION                                 (real network; exactly one attempt per §5's one-shot rule — a BLOCK here ends V8B, does not trigger a redraw)
  ↓
SEPARATE RESEARCH-OPENING GATE                      (requires §10's two security requirements resolved first)
  ↓
Layer B                                             (V8_HISTORICAL_RESEARCH_DESIGN.md §5.3 rules, unchanged, applied to T1B)
  ↓
FROZEN FINAL CANDIDATE                               (§10 (V8) rules, unchanged)
  ↓
T2 SEALED HOLDOUT GATE                              (requires both: §9's T2-reuse conditions still holding, and §10's security requirements resolved)
```

```text
no_real_network_before_the_gates_marked_above=true
each_arrow_is_a_separate_human_or_independent_review_gate=true
no_gate_may_be_skipped=true (V8_HISTORICAL_RESEARCH_DESIGN.md §10.3: skipping_a_stage=false, inherited)
```

---

## 12. Required reuse matrix

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
| Existing `T2` | `REUSE_WITH_CAVEAT` | Conditionally preservable as V8B's sealed holdout — see full argument in §9; contingent on the four conditions in §3.3 continuing to hold through design freeze. |
| Existing `T3` | `REUSE_WITH_CAVEAT` | Same conditional preservation as `T2`, held as `SEALED_RESERVE`, not opened (§3.4). |
| Existing `T_spare` | `REUSE_WITH_CAVEAT` | Available for exactly one new block draw (`T1B`) under the one-shot rule (§4–§5); not available for repeated drawing or for any other purpose in this study. |
| Deterministic partition ordering rule (SHA-256(code)-then-code ascending) | `SAFE_TO_REUSE` | Content-independent, deterministic, carries no outcome information; reused verbatim to draw `T1B`. |
| Old V8 partition trust anchor (`V8_TRUSTED_PARTITION.json`, its authorized manifest SHA / implementation commit) | `REUSE_WITH_CAVEAT` | The anchor mechanism and its existing pin remain valid provenance for the *unchanged* partition (`T0`–`T3`/`T_spare` membership is not being redrawn, §3.6); it does not need re-pinning for `T1B` since `T1B` is drawn from within the already-anchored `T_spare` set under the already-anchored ordering rule — but this must be independently re-confirmed at the `T1B` partition-allocation implementation/review step (§11), not assumed here. |
| Old `T1` failure information (fact that the 1% gate BLOCKed) | `DO_NOT_USE_FOR_CALIBRATION` | Retained only as provenance narrative (§0.2, §6); explicitly forbidden as an input to any threshold derivation. |

---

## 13. Network / privacy compliance for this document

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

## 14. What this draft does not decide

For completeness, restating what remains open and requires a future,
separate human decision beyond this draft:

- The exact numeric threshold (if any) that emerges from §6's calibration
  phase.
- The exact `T_spare` boundary offset that fixes `T1B`'s 300 members
  (an implementation detail, §4).
- Whether `T2`/`T3` preservation (§3.3/§3.4) still holds at the moment
  design actually freezes — this draft only confirms it holds as of the
  reviewed HEAD.
- The concrete code changes implementing §10's two security requirements.
- Any and all real network authorization — none is granted by this
  document.
