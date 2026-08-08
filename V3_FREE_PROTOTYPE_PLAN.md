# V3 free yfinance prototype — pre-registered plan

## Status and scope

- `evaluation_type`: `SURVIVORSHIP_BIASED_RESEARCH_ONLY`
- `formal_backtest`: `false`
- `point_in_time_universe`: `false`
- `deployment_decision_allowed`: `false`
- `shadow_replacement_allowed`: `false`
- `reference_period_used`: `false`
- Experiment purpose: determine whether one pooled, cross-stock return-regression model shows enough out-of-fold predictive signal to justify a one-month J-Quants Light point-in-time feasibility audit.
- This prototype cannot establish that the strategy could historically have been traded or is deployable. It applies a current listed-company universe to history, omits delisted securities, and therefore has survivorship bias.
- This plan is fixed before universe download, OHLCV download, model fitting, or evaluation. Methods and acceptance criteria will not be changed after results are observed.

## Data and period

- Current universe source: the current JPX listed-company information file from an official JPX domain. If it cannot be obtained or its fields cannot deterministically distinguish Prime/Standard domestic common stocks, stop with `FREE_DATA_INSUFFICIENT`; do not invent a list.
- Prices: `yfinance` only. No J-Quants API or J-Quants data.
- Raw download interval: `2019-01-01` through `2025-03-31`, inclusive. 2019 is feature warm-up only.
- Evaluated signal interval: `2020-01-01` through `2025-03-31`, inclusive, subject to label confirmation by `2025-03-31`. Any signal requiring a price after that date is excluded.
- Data on or after `2025-04-01`, reference replay outputs, shadow outputs, and LOOP-003/004 results are prohibited.
- Allowed network destinations: official JPX domains and `query1.finance.yahoo.com` / `query2.finance.yahoo.com`. Redirects and contacted hosts are recorded; a redirect or request outside the allowlist fails closed.
- Raw JPX files, full universe lists, and raw OHLCV are stored only under `C:\taiki\hobbies\stock-analyzer-v3-data\free-prototype`, never committed. Existing cached files are hash-checked and not silently overwritten. Retry count is at most three with bounded exponential backoff.

## Current-only universe and deterministic sample

1. Keep only current TSE Prime and Standard domestic common stocks.
2. Exclude Growth, ETF, ETN, REIT, preferred shares, foreign shares/JDR, infrastructure funds, and all other non-common-stock security types.
3. Normalize the numeric security code without `.T`, sort codes, calculate `SHA-256(UTF-8 code)`, then select the first 300 by `(code_hash, code)` ascending.
4. No industry stratification is used because the availability and point-in-time meaning of the free industry field are not yet established.
5. Save only retrieval date, input/eligible/selected counts, exclusion counts, algorithm version, and a SHA-256 of the newline-joined selected codes. Do not commit the full list.

The universe is explicitly `CURRENT_ONLY`; selection uses no returns, future liquidity, or performance.

## Daily eligibility

At each signal date, using information no later than that date, require:

- at least 252 prior observations including the signal date;
- trailing 60-session median turnover value of at least JPY 100,000,000;
- trailing 60-session median volume of at least 50,000 shares;
- signal close × 100 plus estimated commission no greater than JPY 300,000;
- unique, ordered, finite, internally consistent OHLCV history;
- a next-session open available within the capped snapshot.

Turnover is `Close × Volume` because yfinance does not provide official turnover value. This approximation is an explicit limitation. No later liquidity or entry price is used to create eligibility; the actual next open is used only at execution time.

## Execution and label

- Initial cash: JPY 300,000.
- Lot: 100 shares; maximum open positions: 1.
- Holding horizon: 2 trading sessions.
- Stop loss: 5% below entry; no target-profit exit.
- Entry slippage: 0.03%; exit slippage: 0.03%; stop slippage: 0.10%.
- Commission: 0%, matching the existing v3 base configuration. Cost sensitivity is diagnostic only and is not used to alter this experiment.
- Gap stop, normal stop, then horizon close are handled by the same shared execution function used to generate the label and portfolio result.
- Exit proceeds become available only for the next trading day's opening orders.

Target:

```text
realized_net_return_percent =
    (exit_value - entry_value - entry_commission - exit_commission)
    / entry_value * 100
```

The entry is the next trading day's slipped open. The exit is the actual normal STOP, GAP STOP, or second-holding-session slipped close. `label_confirmed_date` is the actual exit date.

## Fixed features

The only feature set is:

1. `SMA_5_Rate`
2. `SMA_25_Rate`
3. `RSI_14`
4. `MACD_Rate`
5. `BB_Position`
6. `ATR_Rate`
7. `ADX_14`
8. `Change_Rate_1`
9. `Change_Rate_3`
10. `Change_Rate_5`
11. `Volume_Change_1`
12. `Realized_Volatility_20`
13. `Realized_Volatility_60`
14. `Log_Turnover_20`
15. `Median_Turnover_60`
16. `Cross_Sectional_Return_Rank`
17. `Cross_Sectional_Volatility_Rank`

All rolling values end on the signal date. Cross-sectional ranks use only that date's eligible current-only sample. Security code, company name, and embeddings are not model features. Features will not be added, removed, selected, or tuned.

## Fixed model

One pooled `lightgbm.LGBMRegressor` per outer fold; no stock-specific models and no alternative model comparison.

```text
objective = huber
alpha = 0.90
n_estimators = 300
learning_rate = 0.03
num_leaves = 31
max_depth = -1
min_child_samples = 20
subsample = 1.0
colsample_bytree = 1.0
reg_alpha = 0.0
reg_lambda = 0.0
random_state = 42
n_jobs = 1
deterministic = true
force_col_wise = true
verbosity = -1
```

No hyperparameter search, feature search, loss comparison, classification, or learning-to-rank is allowed.

## Purged walk-forward folds

All securities for a signal date belong to the same fold. Calendar boundaries are fixed; non-trading endpoints resolve to the observations within the stated interval.

| Fold | Training feature dates | Validation signal dates |
|---|---|---|
| 1 | 2020-01-01–2020-12-31 | 2021-01-01–2022-03-31 |
| 2 | 2020-01-01–2022-03-31 | 2022-04-01–2023-09-30 |
| 3 | 2020-01-01–2023-09-30 | 2023-10-01–2025-03-31 |

For every fold, training rows must have `label_confirmed_date < validation_start`. Rows whose holding interval overlaps the boundary are purged. A minimum two-trading-session embargo is enforced immediately before validation. Only OOF validation predictions are scored. Rows needing post-2025-03-31 prices are excluded.

## Portfolio rule

- Rank eligible candidates by predicted `realized_net_return_percent` descending, tie by stock code ascending.
- If the highest prediction is at most 0%, record `NO_TRADE_NON_POSITIVE_PREDICTION`.
- At the next open, test candidates in rank order; if one lot is unaffordable, continue to the next.
- If none is affordable, record `NO_TRADE_INSUFFICIENT_CASH`.
- Entries occur before same-day exits; same-day proceeds cannot fund an opening order.

## Baselines

- A: no trade.
- B: random eligible-candidate ranking, 500 deterministic runs with seeds `10000..10499`, under identical dates, candidates, execution, cash, and position limits. Save profit and drawdown distribution plus median, 5th, and 95th percentiles.
- C: trailing 5-session return descending, tie by code.
- D: trailing 20-session return descending, tie by code.
- E: existing LOOP-000, reported only as an eight-stock, different-universe reference and not used as the primary comparator.

## Metrics

Prediction: MAE, RMSE, Huber loss with alpha 0.90, overall and per-fold Spearman correlation, daily cross-sectional Spearman IC mean/median/standard deviation/positive-day rate, top-decile realized return, all-candidate realized return, their difference, and realized return where prediction is positive.

Portfolio: realized net profit, ending equity, maximum drawdown, monthly win rate, yearly and fold profits, closed trades, win rate, maximum stock and industry positive-profit share, STOP/GAP STOP/TIME counts, no-trade counts, negative-cash, duplicate-capital, duplicate-order, and missing-day counts.

## Pre-registered decision

`FREE_PROTOTYPE_PROMISING` requires all of:

1. At least 90% of selected securities have the required price interval.
2. Spearman correlation is positive in all three folds.
3. Overall OOF Spearman correlation exceeds 0.02.
4. Mean daily IC is positive.
5. More than 52% of valid daily ICs are positive.
6. Top-decile mean realized return exceeds the all-candidate mean.
7. At least two folds have positive portfolio profit.
8. At least two folds exceed the matching random-profit median.
9. At least two folds exceed the better of the matching 5-day and 20-day return baselines.
10. Overall maximum drawdown is no more than 25%.
11. At least 150 trades are closed.
12. Negative cash, duplicate capital use, and duplicate orders are all zero.
13. Future access is zero.
14. Deterministic outputs match on two executions over the same frozen cache.

If any evaluable condition fails, the result is `FREE_PROTOTYPE_NOT_PROMISING`. If universe classification, download coverage, or data integrity prevents evaluation, it is `FREE_DATA_INSUFFICIENT`.

Even `FREE_PROTOTYPE_PROMISING` authorizes only consideration of a one-month J-Quants Light point-in-time audit. It does not authorize formal adoption, shadow deployment, Standard, or production trading.

## Stop conditions

Stop without expanding scope on: inability to obtain the official current JPX list; inability to classify ordinary Prime/Standard issues; prohibited-date data; unauthorized host/redirect; changed cached bytes; download success below a level allowing evaluation; feature or label leakage; non-finite prices; safety invariant failure; nondeterministic output; or unavailable LightGBM dependency. Do not repair a failed result by changing the universe, features, fold dates, model, costs, or decision thresholds.
