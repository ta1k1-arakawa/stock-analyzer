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

## 5. Target and corporate actions

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

## 6. Fixed feature set

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

## 7. Models

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

## 8. Training and model choice

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

## 9. Benchmarks

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

## 10. Historical transaction/fill model

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

## 11. T1 promotion criteria

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

## 12. Failure taxonomy

Separate `DATA_QUALITY_FAILURE`, `GOVERNANCE_FAILURE`,
`IMPLEMENTATION_FAILURE`, `OPERATIONAL_EXECUTION_FAILURE`, and
`STRATEGY_CONFIRMATION_FAILURE`. Transport/data/implementation failures are
not strategy failures.

## 13. Forward progression

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

## Paid-data boundary

```text
HUMAN_V9_JQUANTS_STANDARD_PURCHASE_GATE_REQUIRED=true
JQUANTS_PURCHASE_AUTHORIZED=false
JQUANTS_API_ACCESS_AUTHORIZED=false
JQUANTS_RAW_ACQUISITION_AUTHORIZED=false
```

Before purchase/access, GPT must independently review this draft and request a
fresh explicit human gate.
