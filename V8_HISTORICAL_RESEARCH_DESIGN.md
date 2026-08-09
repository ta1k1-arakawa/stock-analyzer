# V8_HISTORICAL_RESEARCH_DESIGN

## 0. Document status

```text
study_name=V8_HISTORICAL_RESEARCH
study_position=HISTORICAL_STRATEGY_DISCOVERY_AND_SELECTION
document_type=FROZEN_DESIGN
design_status=HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION
design_base_branch=v7-forward-capacity-gate3-dry-run
design_base_commit=fec1b85c2e6deb89b8c5d4fa31ff1ae58a62edbc
design_draft_commit=3bc502d8d6b554822aa98b946947e4a6730603f2
human_design_review=PASS
human_review_date=2026-08-09
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

This document is a frozen design. It creates no code, acquires no data, runs no
backtest, fits no model, and computes no profit.

The calendar spans, ticker-block sizes, walk-forward scheme, friction grid and
promotion thresholds below are **frozen** as of the human design review recorded
in §1. They may not be changed by implementation work. Changing any of them
requires a new human gate and, where the change would affect a partition that
has already been opened, a new study (§10.3).

## 1. Human design review record (2026-08-09)

Human design review of draft commit `3bc502d8d6b554822aa98b946947e4a6730603f2`
returned `PASS` with ten binding decisions. They are recorded verbatim in intent
below and are reflected throughout the body of this document.

### 1.1 Decision 1 — Layer A reconciliation approved, with strict meaning

The reconciliation between `PROJECT_RESEARCH_CONCLUSION.md`'s
`further_loop_on_same_data: PROHIBITED` and V8's Layer A is **approved**, with
the meaning fixed strictly as:

- Layer A reuse of already-exposed data is for **hypothesis generation only**,
  within the new research line V8.
- Layer A profit, profit factor, drawdown and every other performance figure
  carry `evidential_weight = NONE`.
- No expectancy or generalisation claim may rest on a Layer A result.
- The V3, V4, V5 and V6 verdicts are **not** reopened, revisited or revised.
- `PROJECT_RESEARCH_CONCLUSION.md` itself is **not** modified.
- This gate must **not** be described as deleting, overriding, superseding or
  weakening the earlier prohibition. The earlier prohibition stands as written.
- Confirmatory evidence is carried **only** by the fresh cross-sectional layers.

See §3.3 for the wording that must be used.

### 1.2 Decision 2 — Ticker block sizes fixed

```text
T0 = existing exposed 300
T1 = fresh validation 300
T2 = fresh sealed holdout 300
T3 = fresh reserve 300
T_spare = remainder
```

Known exposed legacy tickers outside `T0` are excluded from fresh allocation to
`T1`, `T2`, `T3` and `T_spare`.

**Reproducibility BLOCK condition.** When the V8 partition manifest is actually
built, it must fix: the JPX universe source snapshot used, its acquisition UTC,
its raw SHA-256, the eligible ticker list SHA-256, the exclude list, and the
deterministic ordering rule. It must **not** be assumed that the complete
eligible-3,115 ticker list exists inside this repository — it does not. If exact
source-list reproducibility cannot be demonstrated at partition-implementation
time, the work **BLOCKS**. See §5.7.

### 1.3 Decision 3 — `P_early` deferred

Initial V8 uses `P_hist = 2016-04-01 → 2025-12-31` only. Pre-2015 data is **not**
acquired for initial research. It may be acquired later, solely for Layer A
regime expansion, under a separate human gate.

### 1.4 Decision 4 — Layer B access budget reduced to one

`max_access_before_review = 3` is replaced by `max_validation_access = 1`.

The shortlist and every parameter must be frozen **before** `T1` is opened. One
validation batch evaluates all shortlisted candidates under identical
conditions. After `T1` is opened:

- re-tuning against `T1` is prohibited;
- a candidate that failed on `T1` may not be modified and resubmitted to `T1`;
- `T1` becomes burned validation data;
- if a new hypothesis later needs validation, a **new** fresh validation block
  must be allocated from `T_spare` under a human gate.

The access count becomes 1 the moment any result is seen, however partially.

### 1.5 Decision 5 — Layer C takes exactly one candidate

`FROZEN_FINAL_CANDIDATE` is **exactly one** candidate. The draft's "one, or a
preregistered small set" is **not** permitted. Comparing or selecting among
multiple candidates on the `T2` sealed holdout is prohibited.

### 1.6 Decision 6 — Initial sealed holdout fixed

Layer C primary is `T2 × P_hist`. `T3 × P_future` is **not** used in initial V8;
`T3` is held as `SEALED_RESERVE`.

A `T2` sealed-holdout PASS authorises promotion to `FORWARD_STUDY_CANDIDATE`
only. It is **not** a real-money authorisation. The prospective forward study is
a separate study with its own preregistration.

### 1.7 Decision 7 — Walk-forward scheme frozen

Expanding-window chronological walk-forward over
`P_hist = 2016-04-01 → 2025-12-31`, with test years exactly:

```text
2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
total_splits = 8
```

Only data preceding a test window's start may be used for training or design for
that split. Where ML labels are used, `label/exit confirmation date < test
start` is mandatory. Test windows are disjoint. The split scheme cannot change
after search begins.

### 1.8 Decision 8 — Cost/slippage grid frozen

All-in execution friction **per side**, not a broker-specific fee schedule:

```text
friction_grid = 0.03%, 0.05%, 0.10%, 0.15%
base_evaluation_friction = 0.05% per side
survivability_floor = 0.10% per side
stress_report = 0.15% per side (mandatory report, not a hard gate)
```

At 0.10% per side a promotable candidate must show aggregate net profit > 0 and
aggregate profit factor ≥ 1.05. The 0.15% point must always be reported but does
not on its own decide a promotion failure.

### 1.9 Decision 9 — Layer A promotion thresholds frozen before search

See §8.4 for the full frozen threshold set, including the hard parameter
neighbourhood requirement, and §9.3 for mandatory trial-count and
trial-distribution reporting.

### 1.10 Decision 10 — Survivorship-bias wording constrained

`T1` and `T2` being fresh cross-sections does **not** remove survivorship bias,
because the past is still being evaluated through a 2026 current-only universe.

The meaning of a `T2` `SEALED_HOLDOUT_PASS` is therefore limited to:

> the same historical rule reproduced on a cross-section that was not used in any
> prior strategy-development loop.

Prohibited characterisations are listed in §12.3. A prospective forward study
remains necessary regardless of the sealed-holdout outcome.

### 1.11 Frozen parameters (machine-readable)

```text
v8_design_status=HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION
human_review_date=2026-08-09
human_decisions_recorded=10

ticker_block_size=300
T0=EXISTING_EXPOSED_300
T1=FRESH_VALIDATION_300
T2=FRESH_SEALED_HOLDOUT_300
T3=FRESH_RESERVE_300
T3_status=SEALED_RESERVE_NOT_USED_IN_INITIAL_V8
T_spare=REMAINDER_UNALLOCATED
legacy_exposed_tickers_excluded_from_fresh_allocation=true
partition_manifest_source_reproducibility_required=true
partition_blocks_if_source_list_not_reproducible=true
eligible_ticker_list_assumed_present_in_repo=false

P_hist=2016-04-01..2025-12-31
P_early_acquisition=DEFERRED_SEPARATE_HUMAN_GATE
P_future_initial_use=false

layer_A_partition=T0xP_hist
layer_A_access=UNLIMITED
layer_A_evidential_weight=NONE
layer_B_partition=T1xP_hist
max_validation_access=1
layer_B_shortlist_frozen_before_open=true
layer_B_retuning_after_access=false
layer_B_failed_candidate_resubmission=false
layer_B_reentry_requires_new_block_from_T_spare=true
layer_C_partition=T2xP_hist
layer_C_candidate_count=1
layer_C_access_after_freeze=1
layer_C_multi_candidate_comparison=false

walk_forward_type=EXPANDING_WINDOW_CHRONOLOGICAL
walk_forward_test_years=2018,2019,2020,2021,2022,2023,2024,2025
total_splits=8
test_windows_disjoint=true
label_confirmation_before_test_start=REQUIRED
split_scheme_mutable_after_search_start=false

friction_model=ALL_IN_EXECUTION_FRICTION_PER_SIDE
friction_grid_percent_per_side=0.03,0.05,0.10,0.15
base_evaluation_friction_percent_per_side=0.05
survivability_floor_friction_percent_per_side=0.10
survivability_floor_requires_net_profit_gt=0
survivability_floor_requires_profit_factor_gte=1.05
stress_report_friction_percent_per_side=0.15
stress_friction_is_hard_gate=false

promotion_min_positive_test_splits=5
promotion_total_test_splits=8
promotion_median_split_net_profit_gt=0
promotion_min_aggregate_profit_factor_at_base_friction=1.15
promotion_max_aggregate_mtm_drawdown_percent=25
promotion_min_total_closed_trades=120
promotion_max_top5_positive_profit_share_percent=50
promotion_max_industry_positive_profit_share_percent=40
neighbourhood_robustness_is_hard_requirement=true
neighbourhood_min_share_positive_net_profit_percent=70
neighbourhood_median_profit_factor_gt=1.0
neighbourhood_center_isolated_spike_reject_multiple=3
trial_count_reporting=MANDATORY
trial_distribution_reporting=MANDATORY

sealed_holdout_pass_meaning=CROSS_SECTIONAL_REPRODUCTION_ONLY
unbiased_historical_profitability_claim=PROHIBITED
real_world_expectancy_proven_claim=PROHIBITED
sealed_holdout_pass_authorizes=FORWARD_STUDY_CANDIDATE_ONLY
real_money_authorization=false
prospective_forward_study_still_required=true

v7_isolation_weakened=false
```

## 2. Purpose and position

### 2.1 What V8 is

V8 is a historical strategy discovery and selection line for Japanese equities.
It deliberately uses past data at speed, in order to search a much wider
hypothesis space than the one-shot preregistered experiments of V4–V6 allowed.

過去データを使うこと自体は禁止しない。禁止するのは、V4〜V7で既に結果を見た
データを、未使用holdoutとして偽装することである。

### 2.2 What V8 is optimising for

The objective is a strategy candidate with **reproducible positive expectancy
after transaction costs**, not the highest backtest profit.

A candidate that produces the best aggregate net profit but degrades sharply
under a small parameter perturbation, under higher friction, or on a fresh
cross-section is explicitly **less** preferred than a candidate with lower
headline profit and stable behaviour across all three. This preference is
encoded in the frozen thresholds of §8.4 and the promotion gates of §10, not
left to judgement at selection time.

### 2.3 What V8 is not

- Not a re-run, re-evaluation, correction or backfill of V4, V5, V6 or V7.
- Not a continuation of the closed V3 research line.
- Not a deployment path. No V8 result alone can authorise real orders (§11).

## 3. Relationship to V7 (hard isolation)

V7_FORWARD_CAPACITY continues independently as a forward-only paper study. V8
runs in parallel as a separate research line. **The human review did not weaken
any V7 isolation invariant; all of the following remain in force unchanged.**

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

### 3.1 Why the isolation must be two-directional

If V7's forward outcomes were allowed to influence V8 parameter choices, V7
would stop being a clean prospective study and become an in-sample selection
set for V8. Equally, if V8 findings were allowed to alter V7's frozen
parameters mid-flight, V7's preregistration would be void. Both directions are
prohibited for the whole life of both studies.

### 3.2 Shared-lineage disclosure requirement

V7 and V8 share the `FIXED_V4_300` universe and, per the audit §3.9, would share
feature lineage over ≈2025-08 → 2026-08-07 if V8 used that span. Initial V8 does
not use it: `P_hist` ends 2025-12-31 and `P_future` is unused (§1.3, §1.6).

Should any future V8 phase report a result over a span overlapping the V7
feature seed, or over V7's forward window (2026-08-08 onward), that document
must state the overlap and must not present the result as independent
corroboration of V7.

### 3.3 Reconciliation with the V3 closure statement (Decision 1)

`PROJECT_RESEARCH_CONCLUSION.md` records `further_loop_on_same_data: PROHIBITED`
and requires that any future line be 「新しい仮説、新しい事前登録、新しい未観測
評価期間を持つ別系列」.

**That prohibition stands exactly as written.** It is not deleted, overridden,
superseded or weakened by this design or by the human gate of §1.
`PROJECT_RESEARCH_CONCLUSION.md` is not modified, and the V3, V4, V5 and V6
verdicts are not reopened.

V8 is a **separate research line** that satisfies the closure document's own
conditions for a new line:

| Requirement | How V8 satisfies it |
|---|---|
| New hypothesis | V8 searches a new hypothesis space; it does not retry LOOP-003, the V4 meta-label, or the V5-B ranker as such |
| New preregistration | This frozen document plus the per-trial registry (§7) and the frozen-candidate record (§10.2) |
| New unobserved evaluation period | The confirmatory layers B and C are built on **freshly acquired, never-used tickers** (§5), because the audit proved no unobserved *temporal* span of usable length exists for the old universe |
| No further loop on the same data | Already-exposed data is confined to Layer A, whose `evidential_weight = NONE` (§5.2). No V8 verdict may rest on a Layer A result |

The operative distinction: V8 does not claim that results on already-seen data
are evidence. Already-seen data is used **only to generate hypotheses**, and all
confirmatory weight is carried by layers that were never part of a prior
decision loop.

## 4. Inputs inherited from the exposure audit

The audit's operative conclusions that constrain this design:

1. Every span from 2016-04-01 through 2026-01-30 on `FIXED_V4_300` is
   outcome-exposed. 2020–2025 is the most heavily contaminated.
2. 2017–2019 is exposed at **year** granularity; 2020–2025 at **year**
   granularity for two separate studies.
3. The V6-A/V6-A-R2 acceptance gates encode V5-B's observed per-year numbers,
   so any strategy benchmarked against them is fitted to that sample.
4. Roughly 2,808–2,815 eligible JPX Prime/Standard domestic codes have **never**
   been acquired. This is the strongest unused axis and it is cross-sectional.
5. No temporal span is both long enough for walk-forward and free of prior
   outcome exposure, absent new acquisition.
6. Survivorship bias is a property of the current-only universe list and is
   **not** removed by choosing fresh tickers or a fresh period.
7. The complete eligible ticker list is **not** committed to this repository;
   only its derived hashes are. This is why §5.7 exists.

Consequence, which drives §5: **the partition scheme cannot be built on time
alone.**

## 5. Data partition policy (frozen)

### 5.1 Two-axis partitioning

Partitions are defined on a ticker axis crossed with a time axis.

**Ticker blocks (frozen, Decision 2).** Derived from the same deterministic rule
already recorded in `V4_UNIVERSE_MANIFEST.json`:

```text
order = sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending
```

| Block | Definition | Size | Status |
|---|---|---:|---|
| `T0` | rank 1–300 = existing `FIXED_V4_300` | 300 | fully burned; Layer A |
| `T1` | next unused 300 in hash order | 300 | fresh; Layer B validation |
| `T2` | next unused 300 after `T1` | 300 | fresh; Layer C sealed holdout |
| `T3` | next unused 300 after `T2` | 300 | fresh; `SEALED_RESERVE`, unused in initial V8 |
| `T_spare` | remainder | ~1,908–1,915 | fresh; unallocated |

Every fresh block excludes the 7 `LEGACY_8` codes that fall outside `T0`
(`1570`, `4689`, `5020`, `7211`, `7267`, `8306`, `9432`); those are exposed and
must not enter `T1`, `T2`, `T3` or `T_spare`. Block size 300 matches `T0` so
that breadth denominators, industry-concentration ratios and candidate counts
are directly comparable across blocks.

**Time blocks (frozen, Decision 3 and Decision 6).**

| Block | Span | Status in initial V8 |
|---|---|---|
| `P_hist` | 2016-04-01 → 2025-12-31 | **the only time block used** |
| `P_early` | before 2015-01-01 | **deferred**; separate human gate, Layer A regime expansion only |
| `P_gap` | 2026-01-31 → 2026-08-07 | not used |
| `P_future` | after design freeze | not used; reserved with `T3` |

### 5.2 Layer A — DEVELOPMENT

```text
layer=A_DEVELOPMENT
partition=T0 x P_hist
access=UNLIMITED
evidential_weight=NONE
```

Unlimited reuse. Feature engineering, strategy design, parameter search, model
selection and iteration all happen here.

**Layer A is declared non-evidential (Decision 1).** A result obtained here is a
hypothesis, never a finding. No promotion decision beyond
`WALK_FORWARD_SURVIVOR` may cite a Layer A number as evidence of expectancy, and
no expectancy or generalisation claim may rest on one. This declaration is what
makes unlimited reuse of already-burned data honest: the data is already spent,
so spending it further costs nothing, provided nothing is claimed from it.

`P_early` extension of Layer A is **deferred** and requires a separate human
gate (Decision 3). If later granted, the same non-evidential status applies.

### 5.3 Layer B — VALIDATION

```text
layer=B_VALIDATION
partition=T1 x P_hist
access=COUNTED
max_validation_access=1
evidential_weight=SUPPORTING
```

Entered only after Layer A walk-forward survival (§8) has produced a shortlist.

**Single-batch rule (Decision 4).** Before `T1` is opened, the shortlist and
every parameter of every shortlisted candidate must be frozen and preregistered.
One validation batch then evaluates all shortlisted candidates under identical
conditions.

```text
shortlist_frozen_before_open=true
parameters_frozen_before_open=true
single_batch_evaluation=true
identical_conditions_across_shortlist=true
```

**After `T1` has been opened:**

- `validation_access_count` becomes 1 the moment **any** result is seen, however
  partial, and whether or not it is acted on;
- re-tuning any parameter against `T1` is prohibited;
- a candidate that failed on `T1` may **not** be modified and resubmitted to
  `T1`;
- `T1` is burned validation data from that point onward;
- if a new hypothesis later reaches the validation stage, a **new** fresh
  validation block must be allocated from `T_spare` under a human gate. `T2` and
  `T3` may never be repurposed for validation.

**Honest limitation.** `T1 × P_hist` is a fresh cross-section over a *known*
regime. It tests whether a rule generalises across stocks; it does **not** test
whether it generalises across time. Layer B evidence is therefore
`SUPPORTING`, never conclusive.

### 5.4 Layer C — SEALED HOLDOUT

```text
layer=C_SEALED_HOLDOUT
partition=T2 x P_hist
candidate_count=1
access_before_candidate_freeze=PROHIBITED
access_after_freeze=1
evidential_weight=PRIMARY
T3_reserved=true
```

Before `FROZEN_FINAL_CANDIDATE` is recorded, **no** access is permitted to
outcomes, profit, trade results, equity curves or any derived statistic over
Layer C. Feature computation over Layer C is likewise prohibited before freeze,
because feature distributions leak information about the partition.

**Exactly one candidate (Decision 5).** `FROZEN_FINAL_CANDIDATE` is a single
candidate. Submitting a set of candidates, or comparing and selecting among
candidates using `T2` results, is prohibited.

```text
frozen_final_candidate_count=1
multi_candidate_submission=false
candidate_selection_using_T2_results=false
```

After the single final candidate is frozen, Layer C is evaluated **once**.

**Post-opening rule.** If any condition, parameter, feature, universe, cost
assumption or acceptance threshold is changed after Layer C has been opened, the
`T2` partition may never serve as a final test again. The changed line becomes a
**new study** requiring a new sealed partition and a new preregistration. This
applies whether the change is motivated by a Layer C failure or by an unrelated
realisation.

**`T3` status (Decision 6).** `T3` is `SEALED_RESERVE`. It is not used in
initial V8, is not opened for any purpose, and its release requires a human
gate. `T3 × P_future` is explicitly **not** part of initial V8.

### 5.5 What a `T2` PASS does and does not mean (Decision 6, Decision 10)

```text
sealed_holdout_pass_authorizes=FORWARD_STUDY_CANDIDATE_ONLY
sealed_holdout_pass_authorizes_real_money=false
sealed_holdout_pass_meaning=CROSS_SECTIONAL_REPRODUCTION_ONLY
```

A `T2` `SEALED_HOLDOUT_PASS` means exactly this and nothing more:

> the same historical rule reproduced on a cross-section that was not used in any
> prior strategy-development loop.

It authorises promotion to `FORWARD_STUDY_CANDIDATE`. The prospective forward
study is a separate study with its own preregistration. Prohibited
characterisations are enumerated in §12.3.

### 5.6 Partition invariants

```text
T0 ∩ T1 ∩ T2 ∩ T3 = ∅
legacy_exposed_codes ∉ (T1 ∪ T2 ∪ T3 ∪ T_spare)
layer_C_feature_computation_before_freeze=false
layer_C_outcome_access_before_freeze=false
layer_B_retuning_after_access=false
layer_B_failed_candidate_resubmission=false
partition_manifest_written_before_first_feature_computation=true
partition_reassignment_after_first_access=false
T2_or_T3_repurposed_for_validation=false
```

`partition_reassignment_after_first_access=false` is essential: once a block has
been used at one layer it can never be promoted to a stricter layer. A block
touched during development can never later be called a holdout.

### 5.7 Partition manifest reproducibility (BLOCK condition, Decision 2)

The complete eligible-3,115 ticker list is **not** present in this repository.
`V4_UNIVERSE_MANIFEST.json` records only derived values
(`eligible_current_only: 3115`, `raw_file_sha256`, `ticker_list_sha256`,
`universe_csv_sha256`) and the selection rule. **It must not be assumed that the
eligible list itself can be reconstructed from what is committed here.**

When the V8 partition manifest is built, it must fix all of:

| Field | Requirement |
|---|---|
| `jpx_universe_source_snapshot` | The exact source artifact used |
| `source_acquired_at_utc` | Acquisition timestamp |
| `source_raw_sha256` | Raw source bytes hash |
| `eligible_ticker_list_sha256` | Hash of the full derived eligible list |
| `exclude_list` | `T0` members plus the 7 exposed legacy codes outside `T0` |
| `deterministic_ordering_rule` | Verbatim ordering rule used to cut blocks |
| `block_assignments` | Explicit `T1` / `T2` / `T3` / `T_spare` membership |

```text
partition_implementation_requires_exact_source_reproducibility=true
partition_blocks_if_reproducibility_unproven=true
assume_eligible_list_present_in_repo=false
```

If exact source-list reproducibility cannot be demonstrated at
partition-implementation time, the work **BLOCKS** and no block assignment is
written. A partition that cannot be re-derived is not a partition; it is an
unverifiable claim about which tickers were sealed.

### 5.8 Acquisition dependency (not performed in this phase)

Layers B and C require price data for `T1` and `T2`, which has never been
acquired. That acquisition is **out of scope for this design phase** and
requires its own human gate, its own preregistration, and its own manifest
(span, ticker list, per-payload hashes) recorded **before** any feature is
computed. Until then, Layers B and C exist only as design.

### 5.9 Residual contamination that no partition removes

Stated once here so that no downstream document has to rediscover it:

1. **Survivorship bias.** All blocks come from a current-only 2026-08-03
   listing. Delisted and merged companies are absent from every layer, including
   `T1` and `T2`. Fresh cross-sections do **not** fix this (Decision 10).
2. **Regime knowledge.** The researcher already knows how 2020–2025 behaved in
   detail. Any `P_hist` layer is contaminated by that knowledge even on fresh
   tickers.
3. **Researcher memory across studies.** V4–V7 conclusions shape which
   hypotheses get proposed at all. This cannot be undone; it is a reason to
   weight prospective evidence above all historical evidence.
4. **Cost-model uncertainty.** Friction is modelled, not measured. The frozen
   grid of §8.5 bounds this but does not eliminate it.

## 6. Development methodology

### 6.1 Permitted search dimensions

Wide search is permitted in Layer A across at least:

breakout lookback; trend filters; volume confirmation; volatility contraction;
holding period; entry-gap threshold; stop vs no-stop; position sizing; max
open positions; market regime filters; ranking rules; simple deterministic
rules; ML ranking, regression and classification.

### 6.2 Constraint: no undisciplined search

Wide search is permitted; unrecorded search is not. Every trial that produces a
metric — including abandoned, crashed and uninteresting trials — must be
registered before the next trial starts. A trial that is run and not registered
is a protocol violation, because unregistered trials are exactly what makes a
reported best result unfalsifiable.

### 6.3 Hypothesis discipline

Each trial declares a hypothesis and the dimensions it changes relative to its
parent. Changing many dimensions at once is permitted, but the trial must say
so; `changed_dimensions` is used later to reconstruct how much of the search
space was actually explored.

## 7. Experiment registry (mandatory)

An append-only registry. One record per trial, written before the trial's result
is inspected.

### 7.1 Required fields

| Field | Type | Meaning |
|---|---|---|
| `trial_id` | string | Unique, monotonic, never reused |
| `parent_trial` | string \| null | Trial this one was derived from; null for a root |
| `hypothesis` | string | What is expected to improve and why, written before running |
| `changed_dimensions` | list[string] | Dimensions changed vs `parent_trial` |
| `training_range` | object | Partition layer + ticker block + date span used for fitting |
| `walk_forward_splits` | list[object] | Every split's train/test span, in the frozen 8-split scheme |
| `metrics` | object | Full metric set from §8.3, per split and aggregate |
| `result` | string | Factual outcome summary |
| `decision` | enum | `KEEP`, `DISCARD`, `PROMOTE`, `BLOCKED` |
| `code_commit` | string | Git SHA of the code that produced the trial |
| `data_manifest` | string | Hash of the exact data partition consumed |

### 7.2 Additional required fields

| Field | Meaning |
|---|---|
| `created_utc` | Registration timestamp |
| `partition_layer` | `A_DEVELOPMENT` \| `B_VALIDATION` \| `C_SEALED_HOLDOUT` |
| `layer_access_counter_after` | Value of that layer's access counter after this trial |
| `random_seed` | Seed for any stochastic component |
| `friction_point` | Which friction grid point produced these metrics |
| `superseded_by` | Set if a later trial invalidates this one |

### 7.3 Registry invariants

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

## 8. Walk-forward requirements (frozen)

### 8.1 Frozen split scheme (Decision 7)

```text
walk_forward_type=EXPANDING_WINDOW_CHRONOLOGICAL
period=P_hist=2016-04-01..2025-12-31
test_years=2018,2019,2020,2021,2022,2023,2024,2025
total_splits=8
test_windows_disjoint=true
test_window_ordering=CHRONOLOGICAL
split_scheme_frozen_before_search=true
split_scheme_mutable_after_search_start=false
split_scheme_changes_require_new_study=true
```

| Split | Train / design data | Test year |
|---:|---|---|
| 1 | 2016-04-01 → 2017-12-31 | 2018 |
| 2 | 2016-04-01 → 2018-12-31 | 2019 |
| 3 | 2016-04-01 → 2019-12-31 | 2020 |
| 4 | 2016-04-01 → 2020-12-31 | 2021 |
| 5 | 2016-04-01 → 2021-12-31 | 2022 |
| 6 | 2016-04-01 → 2022-12-31 | 2023 |
| 7 | 2016-04-01 → 2023-12-31 | 2024 |
| 8 | 2016-04-01 → 2024-12-31 | 2025 |

### 8.2 Causality requirements

```text
only_data_before_test_start_usable_for_training_or_design=true
label_confirmation_before_test_start=REQUIRED
```

For each test year, only data preceding that test window's start may be used for
training **or design**. Where ML labels are used, a training row is eligible only
if its label/exit confirmation date precedes the test window start. This carries
forward the rule already proven necessary in V4 and V5-B.

Single-period backtests are permitted inside Layer A for quick triage, but a
single-period result **can never justify a promotion**. All 8 splits are required
for any promotion decision.

### 8.3 Metric set required for any promotion decision

All of the following, reported per split and in aggregate:

| Metric | Purpose |
|---|---|
| aggregate net profit | headline outcome, after friction |
| profit factor | gross win/loss ratio |
| max MTM drawdown | worst mark-to-market path, computed independently of book-cost DD |
| closed trades | sample size; a small-sample result is not a result |
| positive periods | count of splits with positive net profit |
| worst split | the single worst split's result |
| median split result | central tendency, resistant to one lucky split |
| top-5 trade profit concentration | whether profit depends on a handful of trades |
| industry concentration | whether profit depends on one sector |
| turnover / transaction-cost sensitivity | metrics recomputed across the frozen friction grid |
| slippage sensitivity | covered by the same all-in friction grid (§8.5) |
| parameter neighbourhood robustness | metrics across the pre-defined local grid (§8.6) |

### 8.4 Frozen Layer A → `WALK_FORWARD_SURVIVOR` thresholds (Decision 9)

All thresholds are frozen **before** search begins and may not be relaxed after
seeing which candidate they would disqualify.

```text
min_positive_test_splits=5           # of 8
median_split_net_profit_gt=0
min_aggregate_profit_factor=1.15     # at base friction 0.05% per side
max_aggregate_mtm_drawdown_percent=25
min_total_closed_trades=120
max_top5_positive_profit_share_percent=50
max_industry_positive_profit_share_percent=40
survivability_at_0.10_percent_per_side_net_profit_gt=0
survivability_at_0.10_percent_per_side_profit_factor_gte=1.05
neighbourhood_robustness=HARD_REQUIREMENT
```

| # | Threshold | Value |
|---:|---|---|
| 1 | positive test splits | ≥ 5 of 8 |
| 2 | median split net profit | > 0 |
| 3 | aggregate profit factor at base friction | ≥ 1.15 |
| 4 | aggregate MTM maximum drawdown | ≤ 25% |
| 5 | total CLOSED trades | ≥ 120 |
| 6 | top-5 positive profit share | ≤ 50% |
| 7 | maximum industry positive profit share | ≤ 40% |
| 8 | at 0.10% per-side friction | net profit > 0 **and** profit factor ≥ 1.05 |
| 9 | parameter neighbourhood robustness | must pass §8.6 |

Failing any one of the nine blocks promotion.

### 8.5 Frozen friction grid (Decision 8)

Friction is modelled as **all-in execution friction per side**, not as a
broker-specific fee schedule.

```text
friction_model=ALL_IN_EXECUTION_FRICTION_PER_SIDE
friction_grid_percent_per_side=0.03,0.05,0.10,0.15
base_evaluation_friction_percent_per_side=0.05
survivability_floor_friction_percent_per_side=0.10
stress_report_friction_percent_per_side=0.15
stress_friction_is_hard_gate=false
```

| Friction (per side) | Role |
|---|---|
| 0.03% | optimistic reference point; reported |
| **0.05%** | **base evaluation friction** — headline metrics are quoted here |
| **0.10%** | **survivability floor** — net profit > 0 and PF ≥ 1.05 required |
| 0.15% | mandatory stress report; **not** a hard gate on its own |

A candidate that is profitable only at 0.03% is not promotable. The 0.15% point
must always be reported so that the degradation slope is visible, but a failure
there alone does not decide promotion.

### 8.6 Frozen parameter-neighbourhood rule (Decision 9)

Neighbourhood robustness is a **hard requirement**, not a preference.

For at least every principal numeric parameter of the candidate, a pre-defined
set of adjacent values is evaluated. Neighbouring values are declared before
the search, together with the parameter list.

```text
neighbourhood_min_share_positive_net_profit_percent=70
neighbourhood_median_profit_factor_gt=1.0
neighbourhood_center_isolated_spike_reject_multiple=3
```

1. At least **70%** of evaluated neighbouring configurations must show net
   profit > 0.
2. The **median profit factor** across neighbouring configurations must be > 1.0.
3. If the centre configuration's aggregate net profit exceeds **3×** the
   neighbourhood median net profit, the candidate is rejected as an isolated
   spike.

Rule 3 is deliberately a rejection on *unusually good* centre performance: a
peak that its own neighbourhood cannot approach is the signature of a search
artifact, not of expectancy.

## 9. Search overfitting control

### 9.1 The problem, stated plainly

The more trials that are run, the more likely it is that the best observed
backtest result is luck rather than expectancy. With a large enough search, an
excellent-looking equity curve is essentially guaranteed to appear on any
dataset. Nothing in a backtest engine's correctness protects against this;
only accounting does.

### 9.2 Required controls

```text
trial_count_recorded=true
full_trial_registry_retained=true
best_only_reporting_prohibited=true
validation_access_count_recorded=true
sealed_holdout_access_count_recorded=true
post_holdout_change_creates_new_study=true
selection_rule_declared_before_search=true
```

### 9.3 Reporting rules (Decision 9)

- Every promotion document reports `trial_count` alongside the promoted
  candidate's metrics. A metric reported without its trial count is incomplete.
- The **full trial distribution** is reported, not only the maximum. A best
  result that is unremarkable within its own trial distribution is treated as a
  null result.
- The selection rule (how the shortlist is chosen from the registry) is declared
  before the search, so that "pick whichever looked best afterwards" cannot be
  the rule.
- Layer B and Layer C access counters are reported at every gate.

```text
trial_count_reporting=MANDATORY
trial_distribution_reporting=MANDATORY
```

### 9.4 Counter semantics

```text
counter_increments_on_access_regardless_of_outcome=true
counter_increments_on_partial_result_view=true
counter_decrement_allowed=false
counter_reset_allowed=false
failed_run_still_counts=true
```

A crashed or aborted run that nevertheless exposed any outcome statistic counts
as an access. Only a run that provably produced no outcome information — for
example, one that failed before reading the partition — does not.

## 10. Promotion gates

### 10.1 Sequence

```text
RESEARCH
  ↓
WALK_FORWARD_SURVIVOR
  ↓
VALIDATION_CANDIDATE
  ↓
FROZEN_FINAL_CANDIDATE            (exactly one candidate)
  ↓
SEALED_HOLDOUT_PASS | SEALED_HOLDOUT_FAIL
  ↓
FORWARD_STUDY_CANDIDATE
```

### 10.2 Gate definitions

| Stage | Entry requirement | Layer | Exit artifact |
|---|---|---|---|
| `RESEARCH` | Registered hypothesis | A | Registry records |
| `WALK_FORWARD_SURVIVOR` | All 8 frozen splits; full §8.3 metric set; passes **all nine** frozen thresholds of §8.4 including neighbourhood robustness | A | Survivor list + trial count + full trial distribution |
| `VALIDATION_CANDIDATE` | Shortlist and all parameters frozen and preregistered **before** `T1` is opened; survives the single Layer B batch without re-tuning | B | Validation report + `validation_access_count = 1` |
| `FROZEN_FINAL_CANDIDATE` | **Exactly one** candidate, with every parameter, feature, friction assumption and universe fixed and hashed | none | Frozen candidate record (immutable) |
| `SEALED_HOLDOUT_PASS` / `FAIL` | One evaluation of `T2 × P_hist` against thresholds declared **in the frozen record** | C | Sealed holdout report + `sealed_holdout_access_count = 1` |
| `FORWARD_STUDY_CANDIDATE` | `SEALED_HOLDOUT_PASS` plus human review | none | Forward study preregistration (separate study) |

### 10.3 Gate invariants

```text
skipping_a_stage=false
returning_to_an_earlier_stage_after_layer_C_access=NEW_STUDY_REQUIRED
frozen_candidate_modification_after_freeze=false
frozen_final_candidate_count=1
sealed_holdout_reuse_after_condition_change=false
promotion_without_registry_reference=false
promotion_on_layer_A_evidence_alone=false
frozen_thresholds_relaxable_after_seeing_results=false
```

### 10.4 On `SEALED_HOLDOUT_FAIL`

A failure is a legitimate scientific outcome and is recorded as such. The
candidate is closed. Work may continue in Layer A with a new hypothesis, but:

- the failed candidate's parameters may not be nudged and retested against `T2`;
- the next candidate reaching freeze needs a **different** sealed partition,
  whose release (`T3`, or a block carved from `T_spare`) requires a human gate;
- if that next candidate also needs validation, a new fresh validation block
  from `T_spare` is required, because `T1` is already burned (§5.3);
- the failure and its trial count are retained in the registry permanently.

## 11. Real-money policy

```text
v8_historical_result_authorizes_real_money=false
v8_sealed_holdout_pass_authorizes_real_money=false
deployment_from_historical_evidence_alone=PROHIBITED
prospective_forward_study_still_required=true
```

Real money may be considered only after **all** of the following, in order:

1. historical development (Layer A, non-evidential)
2. validation (Layer B, single access)
3. sealed historical holdout (Layer C, `T2`, one candidate, one shot)
4. **prospective forward study** — a live-data, paper-only, preregistered
   forward observation period, structurally similar to V7 but for the V8
   candidate, run as a **separate study with its own preregistration**
5. a **separate human deployment gate**, distinct from every gate above

Historical evidence, however clean the partitioning, remains retrospective. Only
step 4 produces evidence from data that did not exist when the strategy was
designed. The deployment gate at step 5 is a human decision and is never implied
by passing step 4.

## 12. Prohibitions

### 12.1 Permanent, for the life of V8

```text
v7_code_or_artifact_modification=false
v7_forward_observation_use_in_v8=false
real_orders=false
deployment=false
same_data_retuning_after_result_observation=false
sealed_holdout_reuse_after_condition_change=false
layer_B_retuning_after_access=false
layer_C_multi_candidate_comparison=false
unregistered_trials=false
best_only_reporting=false
presenting_exposed_data_as_holdout=false
frozen_parameter_modification_without_human_gate=false
```

`presenting_exposed_data_as_holdout=false` is the governing principle of this
study. Every result document must state which layer produced it.

### 12.2 For this design phase specifically

Not performed in this phase, and not authorised by this document:

backtests; network access; data acquisition; ML training; profit calculation;
portfolio simulation; parameter search; formal evaluation; V7 activation
manifest creation or use; V7 durable study root access; V7 code modification;
V7 forward acquisition; historical replay; historical backfill.

### 12.3 Prohibited claim language (Decision 10)

The following characterisations of any V8 historical result — including a `T2`
`SEALED_HOLDOUT_PASS` — are **prohibited**:

```text
"unbiased historical profitability proven"        PROHIBITED
"real-world expectancy proven"                    PROHIBITED
"survivorship bias eliminated"                    PROHIBITED
"survivorship bias resolved by fresh tickers"     PROHIBITED
"deployable"                                      PROHIBITED
"validated strategy"                              PROHIBITED
"out-of-sample profitability confirmed"           PROHIBITED
```

The only permitted characterisation of a `T2` PASS is the §5.5 wording: the same
historical rule reproduced on a cross-section not used in any prior
strategy-development loop. Every such statement must be accompanied by the
survivorship-bias and regime-knowledge caveats of §5.9 and by the statement that
a prospective forward study is still required.

Additionally, per Decision 1, no V8 document may describe the human gate of §1
as deleting, overriding, superseding or weakening
`PROJECT_RESEARCH_CONCLUSION.md`'s `further_loop_on_same_data: PROHIBITED`.

## 13. Human gate status

`PROJECT_STATE.md` §9 requires explicit human approval for a **new strategy
family**. That approval was given on 2026-08-09 (§1).

| # | Item | Status |
|---:|---|---|
| 1 | Accept `V8_HISTORICAL_RESEARCH` as a new research line | **GRANTED** |
| 2 | Accept the two-axis partition policy of §5, including Layer A non-evidential status | **GRANTED** |
| 3 | Fix ticker block sizes at 300 each (`T0`–`T3`) | **GRANTED** (Decision 2) |
| 4 | Fix the walk-forward split scheme | **GRANTED** (Decision 7) |
| 5 | Fix the friction grid and promotion thresholds | **GRANTED** (Decisions 8, 9) |
| 6 | Approve the registry and gate machinery of §7, §9, §10 | **GRANTED** |

Still requiring separate human gates — **not** granted by this document:

| Item | Why still gated |
|---|---|
| Acquisition of `T1` / `T2` price data | Separate preregistration and manifest required (§5.8); acquisition ≠ use |
| Writing the partition manifest | Must satisfy the reproducibility requirement of §5.7 or BLOCK |
| `P_early` (pre-2015) acquisition | Deferred (Decision 3); Layer A regime expansion only |
| Release of `T3` from `SEALED_RESERVE` | Reserved (Decision 6) |
| Allocation of a new validation block from `T_spare` | Only after `T1` is burned (§5.3) |
| Prospective forward study | Separate study, separate preregistration (§11) |
| Real-money deployment | Separate human deployment gate (§11 step 5) |

## 14. Resolved review questions

The draft's open questions are now closed by the §1 decisions.

| # | Draft question | Resolution |
|---:|---|---|
| 1 | Ticker block size | **Resolved** — 300 each for `T0`–`T3`, remainder to `T_spare` (Decision 2) |
| 2 | `P_early` acquisition | **Resolved** — deferred; separate human gate, Layer A only (Decision 3) |
| 3 | `P_future` seal | **Resolved** — not used in initial V8; `T3` held as `SEALED_RESERVE` (Decision 6) |
| 4 | Layer B access budget | **Resolved** — `max_validation_access = 1`, single frozen batch (Decision 4) |
| 5 | Cost grid and survivability floor | **Resolved** — all-in per-side grid 0.03/0.05/0.10/0.15, base 0.05, floor 0.10 (Decision 8) |
| 6 | V7 forward-window collision | **Resolved** — `P_future` unused in initial V8, so no collision arises; any future use falls under §3.2 disclosure |

Two further points settled by the review, beyond the draft's questions:

- Layer C multiplicity: **exactly one** candidate (Decision 5), replacing the
  draft's "one, or a preregistered small set".
- Sealed-holdout claim language: constrained to the §5.5 wording, with §12.3
  enumerating prohibited phrasings (Decision 10).

## 15. Design assertions

```text
design_document_complete=true
design_status=HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION
human_design_review=PASS
human_decisions_applied=10
depends_on_completed_audit=true
concrete_spans_fixed_after_audit=true

partition_layers=3
ticker_block_size=300
T3_status=SEALED_RESERVE
cross_sectional_holdout_designed=true
new_universe_created=false
new_universe_acquired=false
partition_manifest_written=false
partition_reproducibility_block_condition_specified=true

experiment_registry_specified=true
walk_forward_total_splits=8
walk_forward_test_years=2018,2019,2020,2021,2022,2023,2024,2025
friction_grid_percent_per_side=0.03,0.05,0.10,0.15
base_evaluation_friction_percent_per_side=0.05
survivability_floor_friction_percent_per_side=0.10
promotion_thresholds_frozen=9
max_validation_access=1
frozen_final_candidate_count=1
promotion_gates=6
overfitting_controls_specified=true
real_money_gate_stages=5
prohibited_claim_phrases_enumerated=true

implementation_started=false
backtests_run=0
network_requests=0
models_fitted=0
profit_calculated=0
data_acquired=0
parameter_search_executed=false
v7_modified=false
v7_isolation_weakened=false
project_research_conclusion_modified=false
past_verdicts_reopened=false
deployment_allowed=false

next_authorized_action=HUMAN_GATE_FOR_T1_T2_ACQUISITION_AND_PARTITION_MANIFEST
```
