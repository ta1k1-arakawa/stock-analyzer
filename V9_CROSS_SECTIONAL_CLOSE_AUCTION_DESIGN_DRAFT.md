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

For D0, the target is split-adjusted PRICE return from D1 close to D3 close,
converted to same-signal-date cross-sectional percentile rank [0,1]. Cash
dividends are excluded unless a later frozen source contract explicitly
supplies them.

```text
historical_return_type=PRICE_RETURN_EX_DIVIDENDS
```

Disclose this limitation; do not infer total-return alpha. Corporate-action
adjustment must be causal/mechanical. Do not skip observations because a future
split/delisting is known ex post.

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
executable random-K. Random-K uses 1,000 simulations; K=10; the same cycle
dates, capital, 10% buffer, 1-share quantity rules, sector cap, missing-bar
behavior, and execution assumptions as strategy; and a deterministic seed
derived from `UTF8("V9_RANDOM_K_V1")`. No reseeding after results.

## 10. Historical transaction/fill model

```text
base_commission=0
base_explicit_slippage=0
filled_order_price=official_closing_price
historical_fill_rate_100_percent_established=false
tax_in_promotion_metric=false
```

This is an execution assumption, not proof of 100% fills. Mandatory stress A:
2.5 bp adverse price each side. B: 2% deterministic outcome-independent
no-fill via `hash(order_id,fixed_seed)`, with no same-day substitution. C: 5%
deterministic no-fill, report-only severe sensitivity. Tax may be reported
separately.

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
