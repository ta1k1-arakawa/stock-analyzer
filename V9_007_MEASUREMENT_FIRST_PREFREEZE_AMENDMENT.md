# V9_007 Measurement-First Prefreeze Amendment

```text
task=V9_007_MEASUREMENT_FIRST_PREFREEZE_AMENDMENT
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=PREFREEZE_METHODOLOGY_AMENDMENT_AWAITING_GPT_EXACT_SHA_REVIEW
supersedes=conflicting_prefreeze_roadmap_model_selection_and_pre_T1_gate_text_only
V9_design_frozen=false
future_profitability_established=false
```

This is a design/methodology-only amendment. It records the GPT methodology
decisions for the next V9 step and does not freeze the whole V9 study. It
authorizes no cache read, diagnostic execution, model fit, backtest, network
request, purchase, private/sealed/T1 access, real terminal payload read,
terminal-month parsing, F2 bridge, Phase 2, or human-gate consumption.

## 1. Supersession boundary and roadmap

This amendment supersedes only conflicting prefreeze roadmap, model-selection,
and pre-T1-gate text in `V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md`.
All other V9 design text remains in force unless this amendment explicitly
changes it. In particular, the frozen ten features, D1-to-D3 target semantics,
fixed model parameterizations, evaluator-v2 integrity rules, and the later T1
portfolio criteria remain unchanged except where this amendment expressly
states otherwise.

The V9 order is now:

```text
methodology amendment
  -> existing-T0 binary kill screen
  -> only if CONTINUE: early source/PIT/delisted feasibility decision
  -> only if the data path remains viable: resume F1/T/F2
  -> formal pre-T1 measurement/gate
  -> T1 once
```

The current F1 terminal diagnostic M2/M3 findings remain parked. There is no
real successor TERMINAL payload read, diagnostic execution, T parse, F2
bridge, Phase 2, network, purchase, private/sealed access, or T1 access in
this task. No terminal payload is read and no V9 outcome is calculated here.

## 2. Existing-T0 binary kill screen

The first measurement is a hypothesis-viability kill screen using only the
existing `FIXED_V4_300` caches. Its evidential capacity is zero.

```text
role=HYPOTHESIS_VIABILITY_KILL_SCREEN
evidential_capacity=ZERO
input=existing_FIXED_V4_300_caches_only
evaluation_signal_years=2020..2025
network_requests=0
T1_access=0
sealed_private_access=0
```

The future implementation must use the exact ten frozen V9 features and the
exact V9 D1-to-D3 cross-sectional target semantics. It must not use an
approximate or substitute feature, target, period, model, universe, refetch,
or methodological fallback. If the existing caches cannot mechanically
support those exact semantics, no research verdict is permitted. The result
must be a non-research fail-closed state such as
`NO_VERDICT_DATA_INCOMPATIBLE`, returned to GPT for a decision; incompatibility
must not be converted into `STOP`.

The exact feature names are:

```text
return_1d
return_5d
return_20d
return_60d
volatility_20
atr14_percent
close_to_ma20
close_to_ma60
distance_from_high20
volume_dryup
```

The exact frozen target is the V9 D1-to-D3 split-neutral economic PRICE
return, excluding dividends, transformed to the already-frozen same-D0
cross-sectional percentile rank. No alternate target or return conversion is
allowed. The screen must preserve the exact V9 valid-D0 membership, chronology,
and model definitions.

For each fixed model independently, define:

```text
rank_top10_edge(D0) =
  mean(
    frozen target percentile for the first 10 rows by model score descending,
    canonical-code ascending tie-break
  ) - 0.5
```

This is a ranker-only diagnostic. Sector cap, cash, open-position, fill, and
portfolio state must not enter this statistic. For each model:

```text
aggregate_edge = equal-D0-weight mean(rank_top10_edge)
yearly_edge[y] = equal-D0-weight mean(rank_top10_edge within year y)
positive_edge_years = count(yearly_edge > 0)
```

The screen emits only `STOP` or `CONTINUE`, together with non-outcome
provenance required for reproducibility. Detailed outcome/effect metrics must
not be persisted for human tuning. `STOP` is emitted if and only if both fixed
models independently satisfy:

```text
aggregate_edge <= 0
AND positive_edge_years <= 3 of 6
```

Otherwise the screen emits `CONTINUE`. `CONTINUE` means only that V9 is not
killed by this zero-evidence screen. It is not promising evidence, promotion
evidence, out-of-sample evidence, or profitability evidence. `STOP` is
`DEVELOPMENT_VIABILITY_REJECT`, not formal profitability failure.

After either result, feature, model, target, period, and threshold changes
inside V9 based on that result are prohibited. A scientific change requires a
new explicit methodology decision and a new study where applicable.

## 3. Model-selection estimand amendment

The old global-IC-led Ridge-versus-LightGBM selection rule is superseded.
Formal T0 model selection uses the amended ranker estimand:

- LightGBM wins only when its `aggregate rank_top10_edge` is strictly greater
  than Ridge's, and its `positive_edge_years` count is at least Ridge's.
- Otherwise Ridge wins by simplicity.

No hyperparameter search is introduced. The existing fixed Ridge and fixed
LightGBM parameterizations remain unchanged. Global mean IC remains a
mandatory diagnostic, but it is not model-selection authority.

## 4. Formal pre-T1 signal gate

The old selected-model requirement `mean_daily_IC > 0` and
`positive_IC_years >= 5/8` is superseded as the primary gate. The formal
pre-T1 gate requires both hard criteria below:

1. The observed selected-model aggregate `rank_top10_edge` is strictly greater
   than the one-sided 95th percentile of the frozen permutation null.
2. `yearly rank_top10_edge > 0` in at least 6 of the 8 formal years
   `2018..2025`.

The permutation contract is fixed as follows:

- exactly 1000 deterministic permutations;
- permute frozen target percentiles only within each valid D0;
- preserve D0 membership, score arrays, canonical identities, and chronology;
- for every permutation, re-apply the amended frozen Ridge/LightGBM selection
  rule before taking the selected-model aggregate top-10 edge;
- no favorable rerun, seed redraw, or early stopping;
- the exact deterministic seed construction must be frozen in the later
  implementation design before execution.

Mean daily IC and score-bin monotonicity remain mandatory reporting
diagnostics, but neither is a hard rejection criterion. V9 trades the upper
tail rather than the full cross-section.

This gate is signal/ranking viability only. This amendment does not invent a
percentile-to-return economic conversion. Economic materiality belongs to the
later frozen T1 portfolio criteria using actual net portfolio returns, costs,
fill stress, drawdown, and benchmarks.

## 5. V3 negative control

`V3_FREE_PROTOTYPE` is contextual negative-control evidence only. After this
amendment and gate are frozen, later code may compute only mechanically
comparable V3 diagnostics. No V9 threshold may be tuned to make V3 reject.
V3 lacks the V9 point-in-time universe, eight-year window, exact target, and
exact execution semantics. Therefore inability to compute a V9 criterion from
V3 is not failure, and passing or failing V3 never proves gate validity.

## 6. Data-source order and fixed period

If the T0 screen returns `CONTINUE`, the next priority is the existing V9_004
primary blocker: active plus delisted OHLCV coverage and point-in-time-universe
feasibility, before more F2 work.

The following V9 periods remain fixed:

```text
SOURCE_FEATURE_HISTORY_START=2016-09-01
FORMAL_EVALUATION_SIGNAL_WINDOW=2018-01-01..2025-12-31
```

V9 must not add source-availability-driven automatic period shortening,
rolling-start degradation, or denominator rescaling. A candidate source that
cannot satisfy the frozen period/history requirement is an infeasible source
or path. A later period change requires a separate explicit methodology
decision and a new study where applicable.

J-Quants purchase remains `NOT AUTHORIZED` and requires a human money
decision.

## 7. Governance and existing findings

The independent-review PASS bar is unchanged: `C=0/H=0/M=0` remains
required. No Tier-A/Tier-B waiver is introduced. Current F1 M2/M3 remain
open and parked. If T0 returns `STOP`, they may become unnecessary. If T0
returns `CONTINUE` and the data path remains viable, GPT may later authorize a
single strongly-coupled remediation task for M2 and M3.

This amendment does not add a `|raw return|` automatic exclusion rule and
does not invent a corporate-action missing-data threshold. Those belong to a
later data-quality design.

## 8. Provenance and non-actions

This amendment was decided before running the new V9 T0 kill screen and
without observing any new V9 outcome produced by that screen. No future
profitability claim is made.

```text
T0_kill_screen=NOT_RUN
T0_evidential_capacity=ZERO
real_terminal_read_authorized=false
terminal_month_T_parsed=false
Phase2_authorized=false
F2_authorized=false
network_requests=0
JQUANTS_PURCHASE_AUTHORIZED=false
V9_design_frozen=false
future_profitability_established=false
```

The later implementation design must close any remaining mechanical details,
including the exact deterministic permutation seed construction, before any
T0 or formal pre-T1 execution. No implementation, model fit, backtest, cache
read, outcome calculation, or human-gate consumption occurred in V9_007.

## 9. Required next action

```text
NEXT_ACTION=GPT_EXACT_SHA_REVIEW
```
