# V9_008 Single-Position Objective Prefreeze Amendment

```text
document_role=V9_METHODOLOGY_AMENDMENT
task=V9_008_SINGLE_POSITION_OBJECTIVE_PREFREEZE_AMENDMENT
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=DESIGN_AWAITING_GPT_EXACT_SHA_REVIEW
objective_change_before_new_T0_outcome=true
T0_STATUS=NOT_RUN
T0_EVIDENTIAL_CAPACITY=ZERO
```

This amendment changes V9's operational objective and rank-depth estimand
before any new V9 T0 kill-screen outcome is observed. It supersedes conflicting
prefreeze roadmap, model-selection, ten-position portfolio, and pre-T1-gate
text in `V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md`. It does not freeze
the whole V9 study and does not authorize implementation or execution.

## 1. User objective and portfolio architecture

V9's operational objective is changed before any V9 T0 screen result is seen.
All conflicting ten-position and equal-notional architecture is superseded.

```text
PRIMARY_CAPITAL=400000_JPY
ROBUSTNESS_CAPITAL=300000_JPY
MAX_CONCURRENT_POSITIONS=1
SELECTED_NAMES_PER_ENTRY_CYCLE=1
POSITION_CONCENTRATION=SINGLE_BEST_NAME
QUANTITY_GRANULARITY=1_SHARE
MARGIN_OR_BORROWING=false
BEST_ONE_OR_CASH=true
```

On every eligible decision cycle, V9 will score the complete applicable daily
eligible universe and choose at most the single highest-ranked eligible name.
If the later frozen abstention rule permits trading and the one position slot
is free, the later frozen causal quantity/cash rule may allocate as much of the
scenario's available capital as it safely permits to that one name. Otherwise
the portfolio remains in cash.

The following are explicitly not preserved:

- `max_concurrent_positions=10`;
- `target_invested_fraction=0.90`;
- equal-notional ten-name allocation;
- ten-name sector diversification; and
- `target_notional=0.90*equity/10`.

Exact causal D0 quantity reservation and D1 cash-buffer mechanics have not yet
been decided. They are
`CHATGPT_DECISION_REQUIRED_BEFORE_PORTFOLIO_EXECUTION`. This amendment does
not invent a cash buffer, quantity-resize rule, or other execution substitute.

If the one position remains open because an exit does not execute, no new
position may be opened. This prohibits both a same-ticker and a different-
ticker second position.

## 2. Best-one-or-cash abstention

`BEST_ONE_OR_CASH=true` is a required final-portfolio capability. The exact
daily abstention/confidence threshold is not invented here. It must be frozen
before formal portfolio/T1 execution using only permitted development
information and may not be tuned on T1.

The rank-only T0 kill screen in this amendment neither requires nor optimizes
the abstention threshold.

## 3. T0 binary kill screen: TOP1

The V9_007 `rank_top10_edge` is superseded as the primary T0 screen estimand.
The screen remains a zero-evidence hypothesis-viability kill screen using only
existing `FIXED_V4_300` caches and signal years `2020..2025`.

```text
role=HYPOTHESIS_VIABILITY_KILL_SCREEN
evidential_capacity=ZERO
input=existing FIXED_V4_300 caches only
evaluation_signal_years=2020..2025
network_requests=0
T1_access=0
sealed_private_access=0
```

For each fixed model independently:

```text
rank_top1_edge(D0) =
  target_percentile of the score-ranked #1 row
  - 0.5
```

The ordering is model score descending, with canonical code ascending as the
tie-break. This is a ranker-only statistic: cash, quantity, sector, fill,
position state, and the abstention threshold are not applied.

For `2020..2025`:

```text
aggregate_top1_edge = equal-D0-weight mean(rank_top1_edge)
yearly_top1_edge[y] = equal-D0-weight mean within year
positive_top1_edge_years = count(yearly_top1_edge > 0)
```

The screen emits only `STOP` or `CONTINUE`, plus non-outcome provenance needed
for reproducibility. Detailed outcome/effect metrics must not be persisted for
human tuning.

`STOP` is emitted if and only if both fixed models independently satisfy:

```text
aggregate_top1_edge <= 0
positive_top1_edge_years <= 3 of 6
```

Otherwise the result is `CONTINUE`. `CONTINUE` means only that V9 is not killed
by this zero-evidence screen. It is not promising evidence, promotion
evidence, out-of-sample evidence, or profitability evidence. `STOP` is a
`DEVELOPMENT_VIABILITY_REJECT`, not a formal profitability failure.

The screen must use the exact V9 ten frozen features and exact V9 D1-to-D3
cross-sectional target semantics. No approximate or substitute feature,
target, period, model, universe, refetch, or methodological fallback is
allowed. If the existing caches cannot mechanically support those exact
semantics, no research verdict is permitted. The fail-closed state is
`NO_VERDICT_DATA_INCOMPATIBLE` and requires a GPT decision; incompatibility
must not be converted into `STOP`.

After `STOP` or `CONTINUE`, feature, model, target, period, and threshold
changes inside V9 based on that result are prohibited. Such scientific change
requires a new explicit methodology decision or new study as applicable.

Supporting internal Top3/Top5/Top10 edge diagnostics may be computed later to
assess rank-depth shape, but they are not T0 decision authorities and must not
be used for tuning after results are observed.

## 4. Formal model selection: TOP1

The old global-IC-led Ridge-versus-LightGBM selection rule and its TOP10
authority are superseded. The formal development model-selection primary
estimand is TOP1.

LightGBM wins only if both conditions hold:

1. LightGBM aggregate `rank_top1_edge` is strictly greater than Ridge
   aggregate `rank_top1_edge`; and
2. LightGBM's positive-top1-edge-year count is at least Ridge's count.

Otherwise Ridge wins by simplicity. Fixed Ridge and fixed LightGBM
parameterizations remain unchanged. No per-ticker models, new model family,
or hyperparameter search is introduced. Global IC and Top3/Top5/Top10 edge
remain mandatory diagnostics only.

Independent per-ticker, hierarchical, or ticker-aware models are not
introduced in V9_008. Considering them, especially after seeing V9 outcomes,
requires a separate future methodology decision or new study as applicable.

## 5. Formal pre-T1 signal gate: TOP1

The old selected-model requirement
`mean_daily_IC > 0 AND positive_IC_years >=5/8` is superseded as the primary
gate. The selected-model primary estimand is `rank_top1_edge`.

Both hard criteria are required:

A. observed selected-model aggregate `rank_top1_edge` is strictly greater than
   the one-sided 95th percentile of the exact deterministic permutation null;
   and
B. yearly `rank_top1_edge > 0` in at least `6/8` formal years,
   `2018..2025`.

The permutation contract is unchanged from V9_007 except for the amended TOP1
estimand and selection rule:

- exactly 1000 deterministic permutations;
- permute frozen target percentiles only within each valid D0;
- preserve D0 membership, score arrays, canonical identities, and chronology;
- for every permutation, re-apply the amended frozen Ridge/LightGBM TOP1
  selection rule before taking the selected-model aggregate TOP1 edge;
- no favorable rerun, seed redraw, or early stopping; and
- the exact deterministic seed construction must be frozen in the later
  implementation design before execution.

Top3/Top5/Top10 edge, mean daily IC, and score-bin monotonicity remain
mandatory reporting diagnostics, but they are not hard rejection criteria.
V9 trades the upper tail rather than the full cross-section. This gate is
signal/ranking viability only and does not establish economic profitability.
No percentile-to-return economic conversion is invented here. Economic
materiality belongs to later frozen T1 portfolio criteria using actual net
portfolio returns, costs, fill stress, drawdown, and benchmarks.

## 6. Portfolio and benchmark consequences

All T1 rules that depend on a ten-name portfolio are suspended until a later
single-position T1 amendment is independently reviewed. In particular, the
following are suspended or must be replaced before T1:

- executable Random-K with `K=10`;
- ten-name equal-notional allocation;
- max-position-10 assumptions;
- `sector-cap=2` portfolio semantics;
- ten-name cash-reservation mechanics; and
- existing T1 thresholds whose null or benchmark distribution depends on
  `K=10`.

The future executable random benchmark must be single-name compatible, such as
Random-1 using the same eligible universe, capital, execution, and carry
mechanics. Final T1 thresholds are not invented in this amendment.

TOPIX and equal-weight reporting remain conceptually relevant.

No T1 execution is authorized until single-position portfolio mechanics,
`BEST_ONE_OR_CASH` abstention, Random-1, costs/fill stress, drawdown, and
promotion criteria are separately frozen and reviewed.

## 7. Preserved methodology and governance

The following non-conflicting decisions remain in force:

- the measurement-first roadmap: methodology amendment -> existing-T0 binary
  kill screen -> only if `CONTINUE`, early active/delisted OHLCV and PIT
  feasibility decision -> only if viable, resume F1/T/F2 -> formal pre-T1
  measurement/gate -> T1 once;
- the exact ten V9 features and exact D1-to-D3 target;
- fixed formal periods and no automatic source-driven period shortening;
- early active/delisted/PIT feasibility after T0 `CONTINUE`;
- J-Quants purchase not authorized and requiring a human money decision;
- the independent-review bar `C=0/H=0/M=0`;
- existing F1 diagnostic M2/M3 remain `OPEN_PARKED`; and
- no future-profitability claim.

No Tier-A/Tier-B waiver is introduced. No `|raw return|` automatic exclusion
rule or corporate-action missing-data threshold is introduced; those belong to
a later DQ design.

If T0 `CONTINUE`s, the next priority is the existing V9_004 primary blocker:
active plus delisted OHLCV coverage and PIT-universe feasibility before more F2
work. Within V9, preserve:

```text
SOURCE_FEATURE_HISTORY_START=2016-09-01
FORMAL_EVALUATION_SIGNAL_WINDOW=2018-01-01..2025-12-31
```

Do not add source-availability-driven automatic period shortening,
rolling-start degradation, or denominator rescaling. A candidate source/path
that cannot satisfy the frozen V9 period/history requirement is infeasible.
Any later period change requires a separate explicit methodology decision or
new study as applicable.

## 8. V3 negative control

`V3_FREE_PROTOTYPE` is contextual negative-control evidence only. After this
amendment and gate are frozen, later code may compute only mechanically
comparable V3 diagnostics. A V9 threshold must never be tuned to make V3
reject.

V3 does not have the V9 PIT universe, eight-year window, exact target, or exact
execution semantics. Therefore inability to compute a V9 criterion from V3 is
not failure, and passing or failing V3 never proves gate validity.

## 9. Explicit non-authorizations and current status

This is design/methodology only. It authorizes no cache read, model fit,
backtest, T0 execution, diagnostic execution, T parse, F2 bridge, Phase 2,
network request, purchase, private/sealed/T1 access, real TERMINAL payload
read, or human-gate consumption. The current F1 terminal diagnostic M2/M3
remain parked and are not remediated by this task. A later strongly coupled
M2+M3 remediation may be authorized by GPT only if T0 continues and the data
path remains viable.

```text
T0_STATUS=NOT_RUN
T0_KILL_SCREEN=NOT_RUN
T0_EVIDENTIAL_CAPACITY=ZERO
real_terminal_reads=0
terminal_month_T_parsed=false
F2_authorized=false
Phase2_authorized=false
network_requests=0
JQUANTS_PURCHASE_AUTHORIZED=false
V9_design_frozen=false
future_profitability_established=false
```

## 10. Provenance

This single-position objective amendment follows user clarification of the
intended capital-use objective and was decided before any new V9 T0 kill-screen
outcome was observed. `T0_STATUS=NOT_RUN`; this is not outcome-driven tuning.
It makes no claim of future profitability. The amendment awaits GPT-5.6 Sol
exact-SHA review before any later implementation or execution decision.
