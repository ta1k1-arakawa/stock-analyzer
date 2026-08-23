# V9 Cross-Sectional Close-Auction Design Draft

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=PREFREEZE_METHODOLOGY_DECIDED_DRAFT
design_frozen=false
execution_authorized=false
future_profitability_established=false
```

This records supplied methodology decisions only. It authorizes no purchase,
data acquisition, model fitting, backtest, profit calculation, private/sealed
read, human-gate consumption, or design freeze.

## Objective

Maximize prospective real-world net expected value and robustness under
300,000–400,000 JPY capital, not historical backtest profit.

## 1. Historical source and universe

Preferred source: J-Quants individual Standard, subject to a future explicit
human paid-data gate. Historical research window: 2018-01-01 through
2025-12-31. Warm-up may use data from 2016-09-01 onward solely as required for
causal 252-trading-day history.

The daily universe is point-in-time TSE-listed domestic ordinary common stocks
only. Exclude ETFs/ETPs, REITs, foreign issues, preferred/special securities,
and ambiguous reused security codes; never use today's constituent list
retrospectively. Growth, Standard, and Prime are not separately excluded where
the security otherwise meets domestic-common-stock and liquidity requirements.

Daily eligibility: listed on that date; >=252 prior valid trading observations;
finite positive required OHLCV; 60-trading-day median trading value
>=100,000,000 JPY; and no interpolation/imputation of missing market bars.

## 2. Evidence architecture

`T0_DEVELOPMENT` is existing FIXED_V4_300 codes only. Its historical outcomes
are contaminated development evidence with `evidential_weight=NONE`.

The fresh metadata-only candidate pool is historical point-in-time domestic
common-stock codes excluding every ticker classified exposed by
`V8_DATA_EXPOSURE_AUDIT.md`. It requires at least 500 listed JPX trading days
during 2017-01-01..2025-12-31 using security-master metadata only; ambiguous or
reused codes are excluded.

```text
allocation_key=SHA256(UTF8("V9_T1_PARTITION_V1\0" + canonical_code))
sort=(allocation_key,canonical_code)
first_600=T1_FRESH_CROSS_SECTION_CONFIRMATION
remainder=T_RESERVE
secret_reroll=false
redraw=false
balancing=false
manual_substitution=false
```

Allocation occurs without observing prices/outcomes. T1 identities/membership
must not appear in public artifacts. No T1 price/outcome access before
design-freeze PASS, reviewed acquisition/evaluation support, and explicit
future human authorization. Initial V9 has no historical T2; after T1,
temporal forward evidence is preferred to another historical block.

## 3. Portfolio and execution

```text
primary_capital=400000_JPY
capital_robustness_case=300000_JPY
max_concurrent_positions=10
target_invested_fraction=0.90
cash_buffer_fraction=0.10
max_same_33_sector_positions=2
quantity_granularity=1_SHARE
allocation=APPROXIMATELY_EQUAL_NOTIONAL
```

Per-name target notional is `0.90 * current portfolio equity / 10`. D0
features/score use information through D0 close only. D1 orders are submitted
during SBI's applicable closing-auction window; entry price is D1 official raw
close if filled. The fixed horizon is exactly two close-to-close intervals and
the planned exit is D3 official close. D3 close may generate the next signal,
but next entries occur D4 close; same-close sale proceeds never fund D3 buys.

No per-position stop loss. If entry close is unavailable/unfilled, cancel with
no same-day replacement. If planned exit cannot execute, remain open and retry
subsequent closing auctions; never fabricate an exit. Historical S-share-
specific non-fill is not assumed observable from OHLC.

## 4. Quantity and cash

Rank candidates sequentially. `quantity=floor(target_notional / D0_raw_close)`.
If quantity <1, skip and consider the next candidate. Actual D1 execution cost
uses actual D1 close; cash may never be negative. The 10% buffer absorbs price
movement/reservation mismatch. Never exceed cash silently to force ten names.

## 5. Portfolio selection, cash, backfill, and carry semantics

### Global cycle calendar

Use frozen JPX calendar. `D0_0` is the first JPX trading day >=2018-01-01. For
cycle n, `D0_n` is calendar index of `D0_0 + 3*n`; `D1_n`, `D2_n`, and `D3_n`
are respectively first, second, and third next JPX trading days after D0.
Thus `D0_(n+1)=D3_n`. Create cycles while `D0_n <= 2025-12-31`; D1/D2/D3 and
HIGH-2 exit tail may extend later. Global dates never delay/redraw for full
portfolio, failed order, or delayed exit, and are shared by strategy, 300k,
stress, and every Random-K scenario.

### D0 close processing order

When D0 is an existing-position exit-attempt day, process at D0 close:

1. Apply effective HIGH-2 corporate actions.
2. Evaluate/execute required existing-position exits.
3. Update cash, pending proceeds, and open-position state.
4. Compute authoritative end-of-D0 MTM equity.
5. Only then construct D0 features/scores and next-D1 entry submissions.

Using D0 exit outcome to size D1 orders is causal. No D0-close buy exists;
successfully executed D0-exit proceeds may be D1 buying power under frozen cash
ledger.

### Strategy ordering and score ties

For each D0 use frozen daily scoreable universe. Each score must be finite. A
nonfinite score for otherwise valid finite model input fails closed:

```text
failure_class=IMPLEMENTATION_FAILURE
reason=NONFINITE_MODEL_SCORE
```

Traversal order is model score descending then canonical code ascending; equal
scores therefore use canonical-code ascending. Duplicate canonical-code rows
on D0 fail closed as `DATA_QUALITY_FAILURE: DUPLICATE_D0_CANONICAL_CODE`; do
not remove arbitrarily. Random-K retains frozen SHA-256 order without scores.

### Open-position slots and sectors

After D0 close, `carry_count` is still-open positions. If `carry_count >10`,
fail closed as `IMPLEMENTATION_FAILURE: MAX_POSITION_INVARIANT_BROKEN`.
Otherwise `entry_order_slot_budget=10-carry_count`. A position scheduled/retrying
exit at D1 counts as open: D1 same-close exit never provides D1 entry slot. If
budget is zero, submit no entries but do not skip/shift cycle.

Each filled position stores `entry_sector33` from entry-signal D0 and retains
it until exit. Start D0 sector counts from carry positions. Immediately reserve
one sector slot per submitted D1 order, even if it later fails/no-fills; D1
outcomes cannot trigger same-cycle sector backfill. Missing/invalid candidate
sector is `PORTFOLIO_SKIP_SECTOR33_UNAVAILABLE`; continue traversal. The cap
remains two positions/submitted entries per stored sector33.

### Duplicate/open code and D0 cash budget

If candidate code is open after D0 close, record
`PORTFOLIO_SKIP_DUPLICATE_OPEN_CODE` and continue. A code submitted in the D0
pass cannot be submitted again; no same-close exit/re-entry assumption.

After exits compute once per scenario:

```text
current_equity_D0=authoritative_HIGH_3_end_of_D0_MTM_equity
target_notional=0.90*current_equity_D0/10
cash_floor_D0=0.10*current_equity_D0
known_D1_buying_power=cash_mechanically_known_available_after_D0
submission_cash_budget=max(0,known_D1_buying_power-cash_floor_D0)
```

Do not include proceeds of a position still open after D0. This D0 cash floor
is submission budgeting only; actual post-D1 cash may be below it. For candidate
c: `qty_c=floor(target_notional/D0_raw_close_c)`. If less than one, record
`PORTFOLIO_SKIP_QUANTITY_ZERO` and continue. Estimate
`estimated_submission_cost_c=qty_c*D0_raw_close_c`, using raw D0 close only—no
future D1 price/no-fill/stress/outcome. If it exceeds remaining budget, record
`PORTFOLIO_SKIP_D0_ESTIMATED_CASH` and continue; otherwise reserve it. Budget
never becomes negative.

### Exact D0 traversal/backfill

Traverse strategy score order or frozen Random-K order sequentially. Continue
lower only when current candidate fails a D0-known constraint: duplicate/open
code, sector unavailable/cap, quantity zero, or D0 estimated cash. This is
causal pre-submission traversal, not outcome backfill. A candidate passing all
D0 constraints is submitted; reserve estimated cash and sector; increment
submitted count. Stop at `submitted_entry_order_count==entry_order_slot_budget`
or exhausted eligible order. At most ten are submitted. After D0 selection, set
is immutable: no D1-or-later event adds another candidate.

### D1 entry execution and same-close exits

Process submitted orders in exact D0 submission sequence. D1 entries never use
proceeds from a D1 closing-auction exit. For each submitted order:

1. Require finite positive official D1 raw close; otherwise `ENTRY_MARKET_NO_CLOSE`.
2. Apply HIGH-1 deterministic ENTRY no-fill where applicable; if no-fill, `ENTRY_NO_FILL`.
3. Use frozen execution price: base raw close, stress only its frozen adverse rule.
4. Calculate actual cost from that price and frozen qty.
5. If cost exceeds actual D1 entry cash, `ENTRY_INSUFFICIENT_CASH_AT_CLOSE`.
6. Otherwise fill and deduct cost.

Every non-fill is no replacement. Actual D1 execution may consume the D0 10%
buffer and reduce cash below `cash_floor_D0`, but never below zero. There is no
D1 quantity resize, partial synthetic quantity, or substitution. A D1 carry
exit provides no slot, sector capacity, or buying power for pre-close D1
entries; successful D1 exits update future-cycle state only.

### Carry positions and scenario state separation

An unexited position remains carry under HIGH-1/HIGH-2. At each later D0 it
consumes a max-position slot and stored sector slot, remains in MTM equity,
keeps capital tied, and naturally reduces known buying power. No forced
liquidation, replacement, or cycle delay; fewer free slots means fewer orders.
Random-K has identical mechanics in its own scenario state.

Base 400k, 300k, Stress A+B, Stress C, strategy, and each Random-K simulation
have separate causal cash/open-position state. They share global cycle calendar,
applicable score or Random-K ordering, and deterministic shocks, but never
mutable state or another scenario's future state. Differences caused by capital,
fills, stress, or carries are allowed.

### No post-result discretionary fill-up

There is no rule to keep trying lower-ranked names until ten actually fill.
Fill up to D0 submission slots using only D0-known constraints, then accept D1
outcomes. Filled count may be below ten. Underinvestment from no-fill, D1 cash
movement, missing D1 close, or carries is a frozen architecture result, never
repaired afterward.

## 6. Target and corporate actions

### Corporate-action ratio semantic

For a pure stock split or consolidation, define
`R_e = post_action_shares / pre_action_shares`. `R_e` must be a finite positive
exact ratio derived from the future frozen J-Quants source contract. The
source-contract implementation must mechanically verify the J-Quants adjustment
field's orientation against this semantic; it must not silently assume it.

### Causal price normalization

Fully retroactively adjusted J-Quants OHLC must not directly feed D0 feature
history. For observation date A and historical trading date `t <= A`:

```text
F(t,A)=product(R_e for corporate actions where t < effective_date_e <= A)
causal_normalized_price(t,A)=raw_price(t) / F(t,A)
causal_normalized_volume(t,A)=raw_volume(t) * F(t,A)
empty_product=1
```

For D0 features, `A=D0`; only corporate actions effective by D0 may alter D0
feature history. Use causal-normalized O/H/L/C for every price-derived V9
feature. For `volume_dryup`, use causal-normalized volume:

```text
volume_dryup=1 - mean(causal_normalized_volume,5) / mean(causal_normalized_volume,20)
```

This is corporate-action normalization clarification, not a new feature.

### Target

The D1-to-D3 target is split-neutral economic PRICE return. Let
`R_hold=product(R_e where D1 < effective_date_e <= D3)`:

```text
target_raw_return=(raw_close_D3 * R_hold / raw_close_D1) - 1
target=already_frozen_same_D0_cross_sectional_percentile_rank(target_raw_return)
historical_return_type=PRICE_RETURN_EX_DIVIDENDS
```

Cash dividends remain excluded. No future corporate action may affect D0
features or candidate eligibility; do not infer total-return alpha.

### Historical position accounting for pure split/consolidation

Execution prices remain official RAW closing prices. On each effective pure
corporate-action date, before that day's close valuation/execution, an open
position applies `economic_quantity := economic_quantity * R_e`; total
historical entry cost remains unchanged.

If economic quantity is an integer, quantity becomes that exact integer, no
synthetic cash is created, per-share economic cost basis adjusts inversely, and
MTM uses adjusted quantity times that day's raw close. If it is not an integer,
fail closed:

```text
failure_class=OPERATIONAL_EXECUTION_FAILURE
reason=NONINTEGER_CORPORATE_ACTION_ENTITLEMENT_UNMODELED
```

Do not round quantity, assume fractional-share sale or cash-in-lieu price, or
substitute another security. This is not a strategy failure and yields no
strategy-confirmation verdict for the affected formal evaluation attempt.

### Other corporate actions and delisting

If an open position is affected before successful exit by an event not purely
representable by that exact same-code `R_e` share-ratio transformation—including
unresolved merger, share exchange/transfer, code replacement, cash-out,
delisting, or other security conversion—fail closed:

```text
failure_class=DATA_QUALITY_FAILURE
reason=UNMODELED_CORPORATE_ACTION
```

Do not fabricate a last/zero price, favorably replace, or substitute a ticker;
this is not strategy rejection.

### Missing close, suspension, and exit retry

The first planned exit attempt is D3. For an unresolved open position, retry at
every subsequent JPX trading-day closing auction. On each exit-attempt date:

1. Apply effective pure split/consolidation first.
2. Verify the security remains valid under the point-in-time master.
3. Check for a finite positive official raw close.
4. If no official close exists while listed, set `status=MARKET_NO_CLOSE` and keep it open.
5. If valid raw close exists, apply the frozen scenario no-fill rule where applicable.
6. If not no-fill, execute at official raw close with applicable frozen stress.

No price is fabricated.

### Bounded unresolved exit and evaluation tail

The maximum exit-attempt window is 20 consecutive JPX trading days, with D3 as
attempt-day 1. A `MARKET_NO_CLOSE` day counts. If still open after attempt-day
20, fail closed:

```text
failure_class=OPERATIONAL_EXECUTION_FAILURE
reason=EXIT_NOT_RESOLVED_WITHIN_20_JPX_TRADING_DAYS
```

There is no strategy-confirmation verdict and no forced mark-to-zero or
last-close liquidation; this is a risk/operational—not profitability—failure.

Signal generation/model evaluation ends at the frozen signal-window end. After
the final allowed D0, extra source dates are permitted only for planned/retry
exits, MTM of already-open positions, and already-effective corporate-action
processing. They permit no new signals, features, model targets, or training
rows. The required acquisition horizon includes enough JPX trading days for the
20-day exit-resolution tail, derived mechanically from the frozen JPX calendar
rather than a favorable endpoint selected after outcomes.

## 7. Fixed feature set

Exactly ten causal raw factors, then same-date cross-sectional percentile rank
within the applicable study block:

- `return_1d`, `return_5d`, `return_20d`, `return_60d`
- `volatility_20`, `atr14_percent`, `close_to_ma20`, `close_to_ma60`
- `distance_from_high20`
- `volume_dryup = 1 - mean(raw_volume,5) / mean(raw_volume,20)`

Ticker identity, company name, market segment, and industry are not model
features. Industry is permitted only for the sector-cap rule. `volume_dryup`
is a preregistered prior from earlier exploratory work, receives no special
weight, and is not compared with/without inside V9.

## 8. Models

Exactly two development models:

```text
CONTROL=StandardScaler_fitted_on_training_rows_only + Ridge(alpha=10.0,fit_intercept=true)
CANDIDATE=LightGBMRegressor(
  n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
  min_child_samples=100, subsample=0.7, subsample_freq=1,
  colsample_bytree=0.7, reg_lambda=10.0, random_state=20260823,
  n_jobs=1, deterministic=true, force_col_wise=true)
hyperparameter_search=false
```

## 9. Training and model choice

T0 only; expanding training; retrain once per calendar month. A month's
predictions use only rows whose realized target exit date is strictly before
that month's first JPX trading day. Future labels are prohibited. Development
evaluation years are 2018..2025.

LightGBM wins only if it has all: higher mean daily cross-sectional Spearman
IC, higher median yearly IC, and no fewer positive-IC years. Otherwise Ridge
wins by simplicity. The selected model also requires mean daily IC >0 and
positive yearly IC in >=5 of 8 years. Otherwise `V9_REJECT_PRE_T1` and T1 is
never opened. Freeze the selected model before T1; T1 rows never fit or choose
a model.

## 10. Benchmarks

Mandatory: TOPIX price index, eligible-universe equal-weight price return, and
executable random-K. The following is the complete Random-K protocol.

### Random-K universe

For each D0, draw only from the exact same daily scoreable/eligible universe
available to the strategy before model-score ordering and portfolio selection.
Eligibility includes frozen point-in-time membership, 252-history, OHLCV
validity, liquidity, and feature-computability rules. Random-K uses neither
model scores nor realized outcomes.

### Random-K simulations and ordering

Simulation identifiers are the integers `0..999`. For each `simulation_id`,
D0, and `canonical_code`:

```text
random_key=SHA256(UTF8(
  "V9_RANDOM_K_V1\0"
  + format(simulation_id,"04d")
  + "\0" + D0_ISO_DATE
  + "\0" + canonical_code))
sort=(random_key_hex,canonical_code)_ascending
random_k_prng_library_used=false
random_k_numpy_random_sklearn_rng_used=false
reseed=false
redraw=false
```

For each D0, traverse this deterministic order and apply the same
portfolio-eligibility and order-submission constraints as strategy. Stop after
at most K=10 entry orders have been submitted or the eligible pool is
exhausted. A later D1 non-fill, invalid execution, or insufficient-cash failure
does not substitute another same-day candidate. Random-K selection therefore
cannot use D1-or-later information.

## 11. Historical transaction/fill model

```text
base_commission=0
base_explicit_slippage=0
filled_order_price=official_closing_price
historical_fill_rate_100_percent_established=false
tax_in_promotion_metric=false
```

This is an execution assumption, not proof of 100% fills. Mandatory stress A
is 2.5 bp adverse price each side. Stress B has p=0.02; Stress C has p=0.05
and is report-only. Tax may be reported separately.

### Deterministic no-fill protocol

For every ENTRY or EXIT execution attempt, use SHA-256—not a language/library
`hash()`—to generate a domain-separated deterministic event:

```text
digest=SHA256(UTF8(
  "V9_NOFILL_V1\0"
  + SCENARIO
  + "\0" + SIDE
  + "\0" + ORIGINAL_D0_ISO_DATE
  + "\0" + ATTEMPT_DATE_ISO
  + "\0" + canonical_code))

SCENARIO=B_or_C
SIDE=ENTRY_or_EXIT
u64=unsigned_big_endian_integer(digest_bytes[0:8])
threshold=floor(p * 2^64)
NO_FILL=(u64 < threshold)
```

The digest excludes model identity, strategy/random-K arm, random-K simulation
identifier, price, predicted score, and realized outcome. Thus the same
security/date/side attempt receives the same operational shock across strategy
and benchmark portfolios.

An ENTRY no-fill cancels the submitted entry, with no same-day replacement and
no delayed entry. On EXIT no-fill, the position remains open and retries at the
next JPX trading day's closing auction; that new attempt date creates a new
deterministic event and no exit price is fabricated. Stress A + B uses A's
adverse 2.5 bp each side and exactly scenario-B events above—no separate random
stream. The same protocol applies to strategy and every Random-K simulation.
Stress C remains report-only.

Carry-position/capacity semantics after an exit no-fill are not resolved here;
they remain HIGH-4.

## 12. Verdict metric and IC aggregation semantics

### Daily cross-sectional IC

For every valid D0 and model, use all finite scoreable rows in the model's
applicable study block before K=10 selection. Let `s_i` be prediction/score and
`y_i` the frozen target percentile rank. `daily_IC(D0)` is Spearman correlation
of s and y: ascending numeric ranks, average rank for ties, then ordinary
Pearson correlation of the two rank vectors. Although raw target return has the
same Spearman ordering, frozen target percentile rank is authoritative.

Fewer than two finite rows fails closed as
`DATA_QUALITY_FAILURE: INSUFFICIENT_ROWS_FOR_DAILY_IC`. If either rank vector
has zero variance, `daily_IC(D0)=0.0`. A valid D0 is never dropped for an
unfavorable IC.

### IC aggregation and model selection

For exactly 2018, 2019, 2020, 2021, 2022, 2023, 2024, and 2025:

```text
yearly_IC(Y)=arithmetic_mean(daily_IC(D0) where calendar_year(D0)=Y)
positive_IC_year(Y)=(yearly_IC(Y)>0.0)
positive_IC_year_count=sum_over_exact_8_years(positive_IC_year(Y))
mean_daily_IC=arithmetic_mean(all_daily_IC_values_2018_to_2025)
median_yearly_IC=(sort(yearly_IC_values)[3]+sort(yearly_IC_values)[4])/2
```

Every valid D0 has equal weight; do not pool ticker rows across a year. A year
with zero valid D0 dates is `DATA_QUALITY_FAILURE: YEAR_WITH_NO_VALID_IC_DATES`.
LightGBM beats Ridge only if all hold: higher mean daily IC, higher median
yearly IC, and positive-IC-year count at least Ridge's. Equality on either
strict comparison means LightGBM does not win and Ridge wins by simplicity.
The selected model still requires mean daily IC >0 and positive IC in >=5 of 8
years, otherwise existing `V9_REJECT_PRE_T1` behavior applies.

### Continuous equity and aggregate return

Each strategy/random-K scenario has one continuous cash/portfolio/equity path,
with no calendar-year capital reset. Start at frozen starting capital before the
first permitted 2018 cycle and continue through the 2018..2025 signal window
and exit-resolution tail:

```text
portfolio_equity(t)=available_cash + frozen_pending_cash
 + sum(current_economic_quantity_j * official_raw_close_j(t) for open j)
```

Use HIGH-2 economic quantity. An unavailable required MTM close for an open
listed position fails closed as `DATA_QUALITY_FAILURE: MTM_PRICE_UNAVAILABLE`;
do not use future, fabricated, or stale favorable price, and issue no
strategy-confirmation verdict from that attempt.

After final signals stop and all positions validly close in the tail,
`terminal_equity` is final available cash with no open positions:

```text
aggregate_net_price_return=terminal_equity / starting_capital - 1
```

It includes frozen execution prices, commission, slippage/stress, no-fill, and
cash constraints; excludes tax and dividends. It is not annual compounding, a
sum of trade returns, or a yearly-reset result. Existing strategy gate
`aggregate_net_price_return > 0` uses exactly this metric.

### Yearly returns and Random-K yearly excess

At official close of the last JPX trading day in Y, calculate
`YEAR_END_EQUITY(Y)` with the continuous MTM definition. Set
`BASE_EQUITY(2018)=starting_capital`; for later years,
`BASE_EQUITY(Y)=YEAR_END_EQUITY(Y-1)`.

```text
yearly_portfolio_return(Y)=YEAR_END_EQUITY(Y)/BASE_EQUITY(Y)-1
randomK_yearly_median(Y)=(sort(randomK_returns_Y)[499]+sort(randomK_returns_Y)[500])/2
strategy_yearly_excess(Y)=strategy_yearly_portfolio_return(Y)-randomK_yearly_median(Y)
positive_yearly_excess=(strategy_yearly_excess(Y)>0)
```

Apply identically to all r=0..999 Random-K simulations. This is a continuous
path, not independent annual simulations. The post-2025 exit tail creates no
ninth yearly gate, but remains in aggregate return and maximum DD. Existing
requirement is positive yearly excess in >=5 of 8 years; equality is not
positive.

### Random-K aggregate percentile and drawdown

Let S be strategy aggregate return and `R_0..R_999` corresponding Random-K
aggregate returns. With `L=count(R_i<S)` and `E=count(R_i==S)`:

```text
strategy_randomK_return_percentile=100*(L+0.5*E)/1000
```

No interpolation or scipy/numpy percentile convention applies; existing gate
is >=70. Include starting capital as initial equity. For every JPX trading day
in evaluation/tail:

```text
running_peak(t)=max(starting_capital,portfolio_equity(u) for u<=t)
drawdown(t)=(running_peak(t)-portfolio_equity(t))/running_peak(t)
maximum_MTM_DD=max(drawdown(t))
randomK_DD_P75=sort(1000_randomK_maximum_MTM_DD)[749]
```

Maximum DD is the authoritative fraction (percent optional), no yearly reset.
`randomK_DD_P75` is nearest-rank 75th percentile (750th one-based), not an
interpolated quantile. Existing `strategy.maximum_MTM_DD <= randomK_DD_P75`
passes on equality.

### 300k robustness and combined Stress A+B

For existing robustness, run frozen mechanics with `starting_capital=300000_JPY`
only. Reuse model/scores and deterministic Random-K order; do not refit.
Rankings differ only where frozen feasibility mechanics naturally prevent order:

```text
robustness_aggregate_return=terminal_equity/300000-1
randomK_300k_median=(sort(randomK_returns)[499]+sort(randomK_returns)[500])/2
robustness_excess=strategy_300k_aggregate_return-randomK_300k_median
```

Existing `robustness_excess >=0` passes on equality. For existing Stress A+B at
400000 JPY, apply simultaneous A (2.5 bp adverse each side) and HIGH-1 B (2%
deterministic no-fill), with no new stream. Compute aggregate returns
identically:

```text
stress_AB_randomK_median=(sort(randomK_stress_AB_returns)[499]+sort(randomK_stress_AB_returns)[500])/2
stress_AB_excess=strategy_stress_AB_aggregate_return-stress_AB_randomK_median
```

Existing `stress_AB_excess >=0` passes on equality.

### Reporting-only TOPIX and equal-weight benchmarks

TOPIX is reporting-only, using J-Quants TOPIX PRICE (not total-return) index:

```text
TOPIX_period_return=TOPIX_close(last_evaluation_signal_window_JPX_day)
 / TOPIX_close(first_evaluation_signal_window_JPX_day)-1
```

Do not apply portfolio exit tail to TOPIX; disclose strategy aggregate may have
tail effects. Equal weight is reporting-only: for each D0, use the exact same
pre-score daily scoreable/eligible universe and every constituent's frozen
D1-to-D3 split-neutral PRICE return. Its cycle return is arithmetic mean of all
valid constituent returns. It uses no future membership, capital weighting,
K=10, or executability claim, and is named
`EQUAL_WEIGHT_SIGNAL_COHORT_PRICE_BENCHMARK`.

### Invalid formal attempts

An attempt invalidated by `DATA_QUALITY_FAILURE`, `GOVERNANCE_FAILURE`,
`IMPLEMENTATION_FAILURE`, or `OPERATIONAL_EXECUTION_FAILURE` yields
`strategy_confirmation_verdict=NOT_AVAILABLE`, not `V9_T1_REJECT`.
`V9_T1_REJECT` applies only to valid completed T1 evaluation failing one or
more frozen confirmation criteria.

## 13. T1 promotion criteria

One-shot T1 PASS requires all:

- aggregate strategy net price return >0;
- net-return percentile among 1,000 random-K portfolios >=70;
- positive yearly excess versus random-K median in >=5 of 8 years;
- strategy maximum MTM DD <= random-K 75th-percentile DD;
- 300,000 JPY robustness-case excess versus random-K median >=0; and
- combined stress A + stress B excess versus corresponding random-K median >=0.

TOPIX/equal-weight comparisons are mandatory reporting, not independent PASS
gates. Any valid criterion failure is `V9_T1_REJECT`. Do not change thresholds,
features, model, holding period, K, or rerun within V9 after T1.

## 14. Failure taxonomy

Separate `DATA_QUALITY_FAILURE`, `GOVERNANCE_FAILURE`,
`IMPLEMENTATION_FAILURE`, `OPERATIONAL_EXECUTION_FAILURE`, and
`STRATEGY_CONFIRMATION_FAILURE`. Transport/data/implementation failures are
not strategy failures.

## 15. Forward progression

Only after T1 PASS: PAPER is later of three calendar months or 20 completed
portfolio cycles; no tuning. It measures real S-share fills/prices/operations.
Targets: fill rate >=97%, manual/order-submission miss rate <=2%, and explicit
broker fill-price versus official-close mismatch audit.

Then only after separate GPT review and fresh human authorization:
`SMALL_CAPITAL_LIVE=200000_JPY`, for later of three calendar months or 20
completed cycles, using the same frozen mechanics. Pilot kill rule: portfolio
MTM drawdown >=10% from pilot equity peak stops new entries. This is a
capital-risk stop, not proof of strategy invalidity. Full 300,000–400,000 JPY
use requires another fresh human authorization and independent forward-evidence
review.

## 16. Paid-data boundary

```text
HUMAN_V9_JQUANTS_STANDARD_PURCHASE_GATE_REQUIRED=true
JQUANTS_PURCHASE_AUTHORIZED=false
JQUANTS_API_ACCESS_AUTHORIZED=false
JQUANTS_RAW_ACQUISITION_AUTHORIZED=false
```

Before purchase/access, GPT must independently review this draft and request a
fresh explicit human gate.
