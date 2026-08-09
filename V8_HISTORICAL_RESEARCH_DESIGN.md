# V8_HISTORICAL_RESEARCH_DESIGN

## 0. Document status

```text
study_name=V8_HISTORICAL_RESEARCH
study_position=HISTORICAL_STRATEGY_DISCOVERY_AND_SELECTION
document_type=DESIGN_PROPOSAL
design_status=DRAFT_AWAITING_HUMAN_GATE
design_base_branch=v7-forward-capacity-gate3-dry-run
design_base_commit=fec1b85c2e6deb89b8c5d4fa31ff1ae58a62edbc
depends_on=V8_DATA_EXPOSURE_AUDIT.md

implementation_started=false
data_acquired=false
backtests_run=0
models_fitted=0
profit_calculated=0
network_requests=0
parameter_search_executed=false
formal_evaluation_started=false

v7_modified=false
real_orders_allowed=false
deployment_allowed=false
```

This document is a design only. It creates no code, acquires no data, runs no
backtest, fits no model, and computes no profit. Concrete calendar spans below
are stated **only because `V8_DATA_EXPOSURE_AUDIT.md` is complete**; they remain
proposals until the human gate in §12.

## 1. Purpose and position

### 1.1 What V8 is

V8 is a historical strategy discovery and selection line for Japanese equities.
It deliberately uses past data at speed, in order to search a much wider
hypothesis space than the one-shot preregistered experiments of V4–V6 allowed.

過去データを使うこと自体は禁止しない。禁止するのは、V4〜V7で既に結果を見た
データを、未使用holdoutとして偽装することである。

### 1.2 What V8 is optimising for

The objective is a strategy candidate with **reproducible positive expectancy
after transaction costs**, not the highest backtest profit.

A candidate that produces the best aggregate net profit but degrades sharply
under a small parameter perturbation, under higher slippage, or on a fresh
cross-section is explicitly **less** preferred than a candidate with lower
headline profit and stable behaviour across all three. This preference is
encoded in the promotion gates (§8), not left to judgement at selection time.

### 1.3 What V8 is not

- Not a re-run, re-evaluation, correction or backfill of V4, V5, V6 or V7.
- Not a continuation of the closed V3 research line.
- Not a deployment path. No V8 result alone can authorise real orders (§10).

## 2. Relationship to V7 (hard isolation)

V7_FORWARD_CAPACITY continues independently as a forward-only paper study. V8
runs in parallel as a separate research line.

```text
v7_code_modification=false
v7_design_modification=false
v7_artifact_modification=false
v7_activation_manifest_required_by_v8=false
v7_durable_study_root_access_by_v8=false

v7_forward_observations_used_in_v8_tuning=false
v7_interim_results_used_in_v8_parameter_selection=false
v7_results_used_in_v8_candidate_promotion=false
v8_results_used_to_alter_v7=false

independent_research_lines=true
real_orders=false
deployment=false
```

### 2.1 Why the isolation must be two-directional

If V7's forward outcomes were allowed to influence V8 parameter choices, V7
would stop being a clean prospective study and become an in-sample selection
set for V8. Equally, if V8 findings were allowed to alter V7's frozen
parameters mid-flight, V7's preregistration would be void. Both directions are
prohibited for the whole life of both studies.

### 2.2 Shared-lineage disclosure requirement

V7 and V8 share the `FIXED_V4_300` universe and, per the audit §3.9, would share
feature lineage over ≈2025-08 → 2026-08-07 if V8 uses that span. Any V8
document that reports a result over a span overlapping the V7 feature seed, or
over V7's forward window (2026-08-08 onward), must state the overlap and must
not present the result as independent corroboration of V7.

### 2.3 Reconciliation with the V3 closure statement

`PROJECT_RESEARCH_CONCLUSION.md` records `further_loop_on_same_data: PROHIBITED`
and requires that any future line be 「新しい仮説、新しい事前登録、新しい未観測
評価期間を持つ別系列」.

V8 complies as follows:

| Requirement | How V8 satisfies it |
|---|---|
| New hypothesis | V8 searches a new hypothesis space; it does not retry LOOP-003, the V4 meta-label, or the V5-B ranker as such |
| New preregistration | This document plus the per-trial registry (§6) and the frozen-candidate record (§8) |
| New unobserved evaluation period | The confirmatory layers B and C are built on **freshly acquired, never-used tickers** (§4), because the audit proved no unobserved *temporal* span of usable length exists for the old universe |
| No further loop on the same data | Already-exposed data is confined to Layer A, which is **declared non-evidential** (§4.2). No V8 verdict may rest on Layer A results |

The critical distinction: V8 does not claim that results on already-seen data
are evidence. It uses that data only to *generate* hypotheses, and places all
confirmatory weight on layers that were never part of a prior decision loop.

## 3. Inputs inherited from the exposure audit

The audit's operative conclusions that constrain this design:

1. Every span from 2016-04-01 through 2026-01-30 on `FIXED_V4_300` is
   outcome-exposed. 2020–2025 is the most heavily contaminated.
2. 2017–2019 is exposed at **year** granularity; 2020–2025 at **year**
   granularity for two separate studies.
3. The V6-A/V6-A-R2 acceptance gates encode V5-B's observed per-year numbers,
   so any strategy benchmarked against them is fitted to that sample.
4. ~2,808 eligible JPX Prime/Standard domestic codes have **never** been
   acquired. This is the strongest unused axis and it is cross-sectional.
5. No temporal span is both long enough for walk-forward and free of prior
   outcome exposure, absent new acquisition.
6. Survivorship bias is a property of the current-only universe list and is
   **not** removed by choosing fresh tickers or a fresh period.

Consequence, which drives §4: **the partition scheme cannot be built on time
alone.**

## 4. Data partition policy (proposed)

### 4.1 Two-axis partitioning

Partitions are defined on a ticker axis crossed with a time axis.

**Ticker blocks.** Derived from the same deterministic rule already recorded in
`V4_UNIVERSE_MANIFEST.json`:

```text
order = sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending
```

| Block | Definition | Size | Exposure status |
|---|---|---:|---|
| `T0` | rank 1–300 = existing `FIXED_V4_300` | 300 | fully burned |
| `T1` | next unused 300 in hash order | 300 | never acquired |
| `T2` | next unused 300 after `T1` | 300 | never acquired |
| `T3` | next unused 300 after `T2` | 300 | never acquired, reserve |
| `T_spare` | remainder | ~1,908–1,915 | never acquired, unallocated |

Every block excludes the 7 `LEGACY_8` codes that fall outside `T0`
(`1570`, `4689`, `5020`, `7211`, `7267`, `8306`, `9432`); those are exposed and
must not enter any fresh block. Block size 300 is chosen to match `T0` so that
breadth denominators, industry-concentration ratios and candidate counts are
directly comparable across blocks.

**Time blocks.**

| Block | Span | Notes |
|---|---|---|
| `P_hist` | 2016-04-01 → 2025-12-31 | regime known to the researcher in detail |
| `P_early` | before 2015-01-01 | never acquired; maximal survivorship bias |
| `P_gap` | 2026-01-31 → 2026-08-07 | inside the V7 feature seed; ~6 months |
| `P_future` | after the V8 design-freeze date | unobserved at freeze time |

### 4.2 Layer A — DEVELOPMENT

```text
layer=A_DEVELOPMENT
partition=T0 x P_hist
access=UNLIMITED
evidential_weight=NONE
```

Unlimited reuse. Feature engineering, strategy design, parameter search, model
selection and iteration all happen here.

**Layer A is declared non-evidential.** A result obtained here is a hypothesis,
never a finding. No promotion decision beyond `WALK_FORWARD_SURVIVOR` may cite a
Layer A number as evidence of expectancy. This declaration is what makes
unlimited reuse of already-burned data scientifically honest: the data is
already spent, so spending it further costs nothing, provided nothing is
claimed from it.

Optionally extendable with `T0 × P_early` for additional development-side
regime variety, subject to acquisition approval; the same non-evidential status
applies.

### 4.3 Layer B — VALIDATION

```text
layer=B_VALIDATION
partition=T1 x P_hist
access=COUNTED_AND_LIMITED
max_access_before_review=3
evidential_weight=SUPPORTING
```

Entered only after the candidate set has been narrowed to a small,
pre-registered shortlist by Layer A walk-forward survival (§7).

Rules:

- Every access increments `validation_access_count` in the registry, whether or
  not the result is acted on.
- After each access, the shortlist may be **narrowed** but not re-expanded, and
  no parameter may be re-tuned to improve a Layer B number.
- Discovering that all shortlisted candidates fail Layer B is a valid outcome:
  it returns the line to Layer A with a new hypothesis, and it consumes an
  access.
- Exceeding `max_access_before_review` freezes Layer B until a human review
  records why further access is legitimate.

**Honest limitation.** `T1 × P_hist` is a fresh cross-section over a *known*
regime. It tests whether a rule generalises across stocks; it does **not** test
whether it generalises across time. Layer B evidence is therefore
`SUPPORTING`, never conclusive.

### 4.4 Layer C — SEALED HOLDOUT

```text
layer=C_SEALED_HOLDOUT
partition_primary=T2 x P_hist
partition_optional_secondary=T3 x P_future
access_before_candidate_freeze=PROHIBITED
access_after_freeze=1
evidential_weight=PRIMARY
```

Before `FROZEN_FINAL_CANDIDATE` is recorded, **no** access is permitted to
outcomes, profit, trade results, equity curves or any derived statistic over
Layer C. Feature computation over Layer C is likewise prohibited before freeze,
because feature distributions leak information about the period.

After the final candidate (one, or a pre-registered small set) is frozen, Layer
C is evaluated **once**.

**Post-opening rule.** If any condition, parameter, feature, universe, cost
assumption or acceptance threshold is changed after Layer C has been opened,
the same Layer C partition may never serve as a final test again. The changed
line becomes a **new study** requiring a new sealed partition (`T3`, or a
`T_spare` block) and a new preregistration. This applies whether the change is
motivated by a Layer C failure or by an unrelated realisation.

**Optional strongest seal.** `T3 × P_future` is both a fresh cross-section and
an unobserved period. It is the closest historical analogue to a prospective
test and is recommended when the timeline permits waiting. It does not replace
the prospective forward study in §10.

### 4.5 Partition invariants

```text
T0 ∩ T1 ∩ T2 ∩ T3 = ∅
legacy_exposed_codes ∉ (T1 ∪ T2 ∪ T3)
layer_C_feature_computation_before_freeze=false
layer_C_outcome_access_before_freeze=false
layer_B_retuning_after_access=false
partition_manifest_written_before_first_feature_computation=true
partition_reassignment_after_first_access=false
```

`partition_reassignment_after_first_access=false` is essential: once a block has
been used at one layer it can never be promoted to a stricter layer. A block
touched during development can never later be called a holdout.

### 4.6 Acquisition dependency (not performed in this phase)

Layers B and C require price data for `T1`/`T2`/`T3`, which has never been
acquired. That acquisition is **out of scope for this design phase** and
requires its own human gate, its own preregistration, and its own manifest
(span, ticker list, per-payload hashes) recorded **before** any feature is
computed. Until then, Layers B and C exist only as design.

### 4.7 Residual contamination that no partition removes

Stated once here so that no downstream document has to rediscover it:

1. **Survivorship bias.** All blocks come from a current-only 2026-08-03
   listing. Delisted and merged companies are absent from every layer.
2. **Regime knowledge.** The researcher already knows how 2020–2025 behaved in
   detail. Any `P_hist` layer is contaminated by that knowledge even on fresh
   tickers.
3. **Researcher memory across studies.** V4–V7 conclusions shape which
   hypotheses get proposed at all. This cannot be undone; it is a reason to
   weight prospective evidence above all historical evidence.
4. **Cost-model uncertainty.** Slippage and fill assumptions are modelled, not
   measured. Sensitivity analysis (§7) bounds this but does not eliminate it.

## 5. Development methodology

### 5.1 Permitted search dimensions

Wide search is permitted in Layer A across at least:

breakout lookback; trend filters; volume confirmation; volatility contraction;
holding period; entry-gap threshold; stop vs no-stop; position sizing; max
open positions; market regime filters; ranking rules; simple deterministic
rules; ML ranking, regression and classification.

### 5.2 Constraint: no undisciplined search

Wide search is permitted; unrecorded search is not. Every trial that produces a
metric — including abandoned, crashed and uninteresting trials — must be
registered before the next trial starts. A trial that is run and not registered
is a protocol violation, because unregistered trials are exactly what makes a
reported best result unfalsifiable.

### 5.3 Hypothesis discipline

Each trial declares a hypothesis and the dimensions it changes relative to its
parent. Changing many dimensions at once is permitted, but the trial must say
so; `changed_dimensions` is used later to reconstruct how much of the search
space was actually explored.

## 6. Experiment registry (mandatory)

An append-only registry. One record per trial, written before the trial's result
is inspected.

### 6.1 Required fields

| Field | Type | Meaning |
|---|---|---|
| `trial_id` | string | Unique, monotonic, never reused |
| `parent_trial` | string \| null | Trial this one was derived from; null for a root |
| `hypothesis` | string | What is expected to improve and why, written before running |
| `changed_dimensions` | list[string] | Dimensions changed vs `parent_trial` |
| `training_range` | object | Partition layer + ticker block + date span used for fitting |
| `walk_forward_splits` | list[object] | Every split's train/test span, in the frozen split scheme |
| `metrics` | object | Full metric set from §7.2, per split and aggregate |
| `result` | string | Factual outcome summary |
| `decision` | enum | `KEEP`, `DISCARD`, `PROMOTE`, `BLOCKED` |
| `code_commit` | string | Git SHA of the code that produced the trial |
| `data_manifest` | string | Hash of the exact data partition consumed |

### 6.2 Additional required fields

| Field | Meaning |
|---|---|
| `created_utc` | Registration timestamp |
| `partition_layer` | `A_DEVELOPMENT` \| `B_VALIDATION` \| `C_SEALED_HOLDOUT` |
| `layer_access_counter_after` | Value of that layer's access counter after this trial |
| `random_seed` | Seed for any stochastic component |
| `cost_model_id` | Which commission/slippage assumption was used |
| `superseded_by` | Set if a later trial invalidates this one |

### 6.3 Registry invariants

```text
registry_append_only=true
registry_deletion_allowed=false
trial_id_reuse_allowed=false
registration_before_result_inspection=true
unregistered_trial_allowed=false
registry_committed_with_every_promotion=true
```

The registry is the artifact that makes §9's overfitting accounting possible.
A promotion decision that cannot point at the full registry behind it is not a
valid promotion.

## 7. Walk-forward requirements

### 7.1 Split policy

Single-period backtests are permitted inside Layer A for quick triage, but a
single-period result **can never justify a promotion**.

```text
minimum_walk_forward_splits=5
split_scheme_frozen_before_search=true
split_scheme_changes_require_new_study=true
test_windows_disjoint=true
test_window_ordering=CHRONOLOGICAL
train_window_type=EXPANDING_OR_ROLLING_DECLARED_UPFRONT
label_confirmation_before_test_start=REQUIRED
```

`label_confirmation_before_test_start` carries forward the rule already proven
necessary in V4 and V5-B: a training row is eligible only if its exit/label date
precedes the test window start.

Suggested instantiation over `P_hist` (2016-04-01 → 2025-12-31): expanding
train with yearly test folds 2018 … 2025, giving 8 splits. Frozen at
implementation start, not adjusted afterwards.

### 7.2 Metric set required for any promotion decision

All of the following, reported per split and in aggregate:

| Metric | Purpose |
|---|---|
| aggregate net profit | headline outcome, after costs |
| profit factor | gross win/loss ratio |
| max MTM drawdown | worst mark-to-market path, computed independently of book-cost DD |
| closed trades | sample size; a small-sample result is not a result |
| positive periods | count of splits with positive net profit |
| worst split | the single worst split's result |
| median split result | central tendency, resistant to one lucky split |
| top-5 trade profit concentration | whether profit depends on a handful of trades |
| industry concentration | whether profit depends on one sector |
| turnover / transaction-cost sensitivity | profit recomputed across a cost grid |
| slippage sensitivity | profit recomputed across a slippage grid |
| parameter neighbourhood robustness | profit across a local grid around the chosen parameters |

### 7.3 Robustness preference (explicit, not discretionary)

```text
single_parameter_spike_rejection=true
neighbourhood_degradation_threshold=DECLARED_BEFORE_SEARCH
worst_split_weighting=REQUIRED
median_split_preferred_over_mean=true
cost_sensitivity_must_be_monotone_and_survivable=true
```

A candidate whose performance collapses at an adjacent parameter value is
treated as a search artifact and is **not** promotable, regardless of its
headline metrics. The concrete degradation threshold is declared before the
search begins so it cannot be relaxed after seeing which candidate it would
disqualify.

Similarly, a candidate that is profitable only at the optimistic end of the
cost/slippage grid is not promotable. The cost grid must include a
pessimistic-but-plausible point, declared upfront.

## 8. Search overfitting control

### 8.1 The problem, stated plainly

The more trials that are run, the more likely it is that the best observed
backtest result is luck rather than expectancy. With a large enough search, an
excellent-looking equity curve is essentially guaranteed to appear on any
dataset. Nothing in a backtest engine's correctness protects against this;
only accounting does.

### 8.2 Required controls

```text
trial_count_recorded=true
full_trial_registry_retained=true
best_only_reporting_prohibited=true
validation_access_count_recorded=true
sealed_holdout_access_count_recorded=true
post_holdout_change_creates_new_study=true
selection_rule_declared_before_search=true
```

### 8.3 Reporting rules

- Every promotion document reports `trial_count` alongside the promoted
  candidate's metrics. A metric reported without its trial count is incomplete.
- The **distribution** of trial results is reported, not only the maximum. A
  best result that is unremarkable within its own trial distribution is treated
  as a null result.
- The selection rule (how the shortlist is chosen from the registry) is declared
  before the search, so that "pick whichever looked best afterwards" cannot be
  the rule.
- Layer B and Layer C access counters are reported at every gate.

### 8.4 Counter semantics

```text
counter_increments_on_access_regardless_of_outcome=true
counter_decrement_allowed=false
counter_reset_allowed=false
failed_run_still_counts=true
```

A crashed or aborted run that nevertheless exposed any outcome statistic counts
as an access. Only a run that provably produced no outcome information — for
example, one that failed before reading the partition — does not.

## 9. Promotion gates

### 9.1 Sequence

```text
RESEARCH
  ↓
WALK_FORWARD_SURVIVOR
  ↓
VALIDATION_CANDIDATE
  ↓
FROZEN_FINAL_CANDIDATE
  ↓
SEALED_HOLDOUT_PASS | SEALED_HOLDOUT_FAIL
  ↓
FORWARD_STUDY_CANDIDATE
```

### 9.2 Gate definitions

| Stage | Entry requirement | Layer used | Exit artifact |
|---|---|---|---|
| `RESEARCH` | Registered hypothesis | A | Registry records |
| `WALK_FORWARD_SURVIVOR` | ≥5 splits; full §7.2 metric set; passes the pre-declared robustness thresholds of §7.3 | A | Survivor list + trial count + trial distribution |
| `VALIDATION_CANDIDATE` | Shortlist frozen and preregistered **before** the first Layer B access; survives Layer B without re-tuning | B | Validation report + `validation_access_count` |
| `FROZEN_FINAL_CANDIDATE` | Exactly one candidate, or a preregistered small set, with every parameter, feature, cost model and universe fixed and hashed | none | Frozen candidate record (immutable) |
| `SEALED_HOLDOUT_PASS` / `FAIL` | One evaluation of Layer C against thresholds declared **in the frozen record** | C | Sealed holdout report + `sealed_holdout_access_count` |
| `FORWARD_STUDY_CANDIDATE` | `SEALED_HOLDOUT_PASS` plus human review | none | Forward study preregistration |

### 9.3 Gate invariants

```text
skipping_a_stage=false
returning_to_an_earlier_stage_after_layer_C_access=NEW_STUDY_REQUIRED
frozen_candidate_modification_after_freeze=false
sealed_holdout_reuse_after_condition_change=false
promotion_without_registry_reference=false
promotion_on_layer_A_evidence_alone=false
```

### 9.4 On `SEALED_HOLDOUT_FAIL`

A failure is a legitimate scientific outcome and is recorded as such. The
candidate is closed. Work may continue in Layer A with a new hypothesis, but:

- the failed candidate's parameters may not be nudged and retested against the
  same Layer C partition;
- the next candidate reaching freeze needs a **different** sealed partition;
- the failure and its trial count are retained in the registry permanently.

## 10. Real-money policy

```text
v8_historical_result_authorizes_real_money=false
v8_sealed_holdout_pass_authorizes_real_money=false
deployment_from_historical_evidence_alone=PROHIBITED
```

Real money may be considered only after **all** of the following, in order:

1. historical development (Layer A)
2. validation (Layer B, access-counted)
3. sealed historical holdout (Layer C, one-shot)
4. **prospective forward study** — a live-data, paper-only, preregistered
   forward observation period, structurally similar to V7 but for the V8
   candidate
5. a **separate human deployment gate**, distinct from every gate above

Historical evidence, however clean the partitioning, remains retrospective.
Only step 4 produces evidence from data that did not exist when the strategy
was designed. The deployment gate at step 5 is a human decision and is never
implied by passing step 4.

## 11. Prohibitions

### 11.1 Permanent, for the life of V8

```text
v7_code_or_artifact_modification=false
v7_forward_observation_use_in_v8=false
real_orders=false
deployment=false
same_data_retuning_after_result_observation=false
sealed_holdout_reuse_after_condition_change=false
unregistered_trials=false
best_only_reporting=false
presenting_exposed_data_as_holdout=false
```

`presenting_exposed_data_as_holdout=false` is the governing principle of this
study. Every result document must state which layer produced it.

### 11.2 For this design phase specifically

Not performed in this phase, and not authorised by this document:

backtests; network access; data acquisition; ML training; profit calculation;
portfolio simulation; parameter search; formal evaluation; V7 activation
manifest creation or use; V7 durable study root access; V7 code modification;
V7 forward acquisition; historical replay; historical backfill.

## 12. Human gate required before implementation

`PROJECT_STATE.md` §9 requires explicit human approval for a **new strategy
family**. V8 is a new strategy family, so implementation cannot begin on this
document alone.

Approval is requested for, and only for, the following — each separable:

| # | Item | Notes |
|---|---|---|
| 1 | Accept `V8_HISTORICAL_RESEARCH` as a new research line | Independent of V7, which continues unchanged |
| 2 | Accept the two-axis partition policy of §4 | Including the non-evidential status of Layer A |
| 3 | Authorise acquisition of `T1` / `T2` price data | Separate preregistration and manifest required; acquisition ≠ use |
| 4 | Fix the walk-forward split scheme and robustness thresholds | Must be frozen before any search |
| 5 | Approve the registry and gate machinery of §6, §8, §9 | Implementation of the harness only, no strategy search |

Items 3–5 may be granted independently. Nothing in §4's Layer B/C design is
binding until item 3 is granted and the acquisition manifest exists.

## 13. Open questions for the human reviewer

Recorded rather than silently decided:

1. **Block size.** `T1`/`T2`/`T3` at 300 each matches `T0`, but consumes 900 of
   ~2,808 available codes. A smaller block preserves more future sealed
   partitions; a larger one gives more statistical power per test. Which
   trade-off is preferred?
2. **`P_early` acquisition.** Should pre-2015 data be acquired to widen Layer A's
   regime coverage, accepting worse survivorship bias, or is `P_hist` breadth
   sufficient?
3. **`P_future` seal.** Is waiting for a `T3 × P_future` seal acceptable given
   the timeline, or should `T2 × P_hist` be the only seal?
4. **Layer B access budget.** Is `max_access_before_review = 3` the right
   number? Lower is stricter but risks stalling on a single ambiguous result.
5. **Cost grid.** What pessimistic slippage/commission point should be treated
   as the survivability floor? This must be fixed before search, and it
   materially determines which candidates are promotable.
6. **V7 forward-window collision.** If V8 eventually wants `P_future` data
   overlapping V7's forward window, is the shared-regime entanglement of §2.2
   acceptable with disclosure, or should V8's `P_future` be deferred until V7
   concludes?

## 14. Design assertions

```text
design_document_complete=true
depends_on_completed_audit=true
concrete_spans_fixed_after_audit=true
partition_layers=3
cross_sectional_holdout_designed=true
new_universe_created=false
new_universe_acquired=false
experiment_registry_specified=true
walk_forward_minimum_splits=5
promotion_gates=6
overfitting_controls_specified=true
real_money_gate_stages=5

implementation_started=false
backtests_run=0
network_requests=0
models_fitted=0
profit_calculated=0
data_acquired=0
v7_modified=false
deployment_allowed=false
next_authorized_action=HUMAN_REVIEW_OF_V8_DESIGN
```
