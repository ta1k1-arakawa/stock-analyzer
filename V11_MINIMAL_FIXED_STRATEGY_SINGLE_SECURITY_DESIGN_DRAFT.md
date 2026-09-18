# V11 Minimal Fixed-Strategy Single-Security Study Design

```text
document_type=V11_MINIMAL_FIXED_STRATEGY_SINGLE_SECURITY_DESIGN
status=DRAFT_AWAITING_GPT_REVIEW
study=V11_MINIMAL_FIXED_STRATEGY_SINGLE_SECURITY
design_purpose=MINIMAL_FIXED_STRATEGY_EVALUATION
authoritative_branch=v11-minimal-single-security-study
parent_commit=4fc91c19c5534d59e041876f5310b97e373eb1fc
design_frozen=false
future_profitability_established=false
```

This is a new study identity following the explicit project decision to pause
the V10C/V10D historical route. It is intentionally narrow: determine, with
minimal development cost, whether one fixed Japanese security can support a
reproducible long-only `BUY`, `NO-TRADE`, and `EXIT` notification system worth
advancing to future forward paper observation.

This design is prospective and design-only. It does not select a ticker,
acquire data, inspect historical outcomes, fit a model, run a backtest, read
protected/private data, send Slack messages, paper trade, or trade real money.
No V10C/V10D artifact or methodology is reopened, repaired, or reinterpreted.

## 1. Security selection

Selection must not use historical strategy return, model score, validation
result, Sharpe, PnL, drawdown, or prior selected-rules output. In particular,
the selector must not inspect old profitability outputs or
`data/backtest_selection.yaml`, `data/backtest_results`, or
`data/backtest_comparison` to choose the security.

The canonical candidate universe is the TOPIX Core30 constituent snapshot from
the latest JPX periodic selection result published before this design freeze:

```text
JPX_TOPIX_CORE30_PERIODIC_SELECTION_ASOF=2025-10-31
```

At the later authorized selection step, exclude every numeric security code
whose code already appears as a filename under `data/benchmark/ohlcv/*.csv`
at parent commit
`4fc91c19c5534d59e041876f5310b97e373eb1fc`. This avoids contamination from
prior stock-analyzer historical research. The parent snapshot and exclusion
set must be bound before selection; no old strategy output may be read for
this purpose.

Sort the remaining numeric security codes ascending. Let:

```text
SEED_TEXT=V11_MINIMAL_SINGLE_SECURITY|4fc91c19c5534d59e041876f5310b97e373eb1fc
selection_hash=SHA256(UTF8(SEED_TEXT))
selection_integer=integer value of the full 256-bit selection_hash
selected_index=selection_integer mod eligible_count
```

If `eligible_count <= 0`, stop the selection procedure. Otherwise select
exactly the code at `selected_index`. Selection occurs only after this design
receives GPT exact-SHA PASS and separate design-freeze approval. It is not
performed in this design commit.

After selection, the ticker identity is frozen. There is no fallback ticker,
redraw, substitution, or change because price, lot affordability, data
quality, model result, or performance is inconvenient:

```text
V11_TICKER_SELECTED=false
V11_TICKER_SUBSTITUTION_ALLOWED=false
V11_SELECTION_REDRAW_ALLOWED=false
```

## 2. Fixed public data source and byte boundary

The historical/public source is Yahoo Finance Chart API with exactly:

```text
host=query1.finance.yahoo.com
endpoint=/v8/finance/chart/{ticker}.T
interval=1d
events=div,splits
includeAdjustedClose=true
```

The later implementation should reuse reviewed transport and parser behavior
from `src/v7_yahoo_collector.py` where applicable and must not silently
broaden its parser semantics. The fixed historical acquisition window is:

```text
2015-01-01 <= date < 2026-09-01
```

The first complete raw public payload must be durably preserved and SHA-256
bound before semantic processing. A transport failure before a complete first
payload is:

```text
PLUMBING_FAILURE_RETRIABLE
```

Once the first complete payload is locked, there is no refetch-until-pass,
provider substitution, ticker substitution, or outcome-driven acquisition.
Parser or software repair, if later separately reviewed, may reprocess only
the identical locked bytes. An intrinsic failure of those fixed bytes is:

```text
DATA_QUALITY_FAILURE
```

This design does not authorize acquisition or parser repair.

## 3. Features

Use exactly these five price-ratio or return features:

1. `return_1d`
2. `return_5d`
3. `return_20d`
4. `volatility_20`
5. `close_to_ma20`

There is no volume feature, fundamental data, news, alternative data, feature
search, or feature tuning. The exact causal split-normalized construction is
fixed before any result is observed.

For each decision trading row `t`, use only split events whose effective date
is less than or equal to `t`. For any observed row `d <= t`:

```text
split_factor_t(d) = product of split ratios for events s where
    d < effective_date(s) <= t
split_ratio = post_split_shares / pre_split_shares = numerator / denominator
split_factor_t(d) = 1 when no such event exists
P_t(d) = raw_close(d) / split_factor_t(d)
```

This is the reviewed causal split normalization direction. Future split events
with effective date greater than `t` must not affect features at `t`.

Using observed trading rows rather than calendar-day offsets:

```text
return_1d(t)  = P_t(t) / P_t(t-1row)  - 1
return_5d(t)  = P_t(t) / P_t(t-5rows) - 1
return_20d(t) = P_t(t) / P_t(t-20rows) - 1
r_j = P_t(j) / P_t(j-1row) - 1
volatility_20(t) = sample standard deviation of the final 20 r_j ending at t
    ddof=1
close_to_ma20(t) = P_t(t) / mean(P_t over the final 20 observed price rows
    including t) - 1
```

No adjusted-close series may silently replace this definition. Adjusted close
may be parsed or validated as required by the reviewed Yahoo parser, but V11
feature construction uses only the causal raw-close/split-normalized series.
No volume enters a feature.

A feature row requires every preceding observed row needed for its lookbacks;
rows without sufficient history are mechanically ineligible. Do not impute or
shorten a lookback. Missing or nonfinite required values after sufficient
history eligibility fail closed as `DATA_QUALITY_FAILURE`, rather than being
silently dropped to improve results.

Training features for a historical decision row `d` are the values available
at `d`, using split events effective no later than `d`. They must not be
recomputed using splits learned after `d`.

## 4. Model and causal training

Use exactly one model pipeline:

```text
StandardScaler + Ridge regression
alpha=10.0
fit_intercept=true
```

There is no LightGBM, model comparison, hyperparameter search, threshold
optimization, or alternative scaler behavior. Refit using expanding causal
history. For every decision date `t`, only training examples whose target exit
price is fully observable by `t` may enter training. The minimum training
sample count is:

```text
minimum_training_examples=252
```

## 5. Target and timing

The decision occurs after the close on observed trading row `t`. The predicted
target is the gross return from the next observed raw open `O[t+1]` to the raw
open `O[t+6]`, a five-trading-interval holding period. Define:

```text
entry_date = row t+1
exit_date = row t+6
R = product of split ratios with entry_date < effective_date <= exit_date
R = 1 when no such split event exists
target_5 = (raw_open(exit_date) * R / raw_open(entry_date)) - 1
holding_intervals=5
```

Split events between entry and exit must be economically accounted for using
the frozen split ratio. Dividends are not added to strategy PnL. No intraday
target is permitted.

## 6. Fixed long-only trading rule

When flat, issue `BUY` if the predicted five-session return is at least
`0.006`; otherwise issue `NO-TRADE`. Entry is 100 shares at the next observed
raw open. Exit is the open five trading intervals after entry, `O[t+6]`
relative to decision `t`.

```text
position=long_only
base_threshold=0.006
shares_per_trade=100
entry=next_observed_raw_open
exit=O[t+6]
ENTRY_COST_RATE=0.0015
EXIT_COST_RATE=0.0015
BASE_ROUND_TRIP_FRICTION=0.003
```

There is no stop loss, take profit, pyramiding, overlapping position, or short
selling. The effective friction rates include conservative commission and
slippage assumptions. No parameter may be altered after outcome inspection.
For any scenario with total round-trip friction `f`, use symmetric rates
`f/2` per side. For one completed trade:

```text
entry_cost_rate = f / 2
exit_cost_rate = f / 2
entry_cash = 100 * raw_open(entry_date) * (1 + entry_cost_rate)
exit_share_count = 100 * R
exit_cash = (100 * R) * raw_open(exit_date) * (1 - exit_cost_rate)
net_pnl_yen = exit_cash - entry_cash
net_trade_return = exit_cash / entry_cash - 1
```

The base rates are `0.0015` per side for `f=0.003`. Do not round
intermediate values; reporting may round only after all calculations. Report
the maximum required 100-share entry notional. Historical research may
simulate the fixed 100-share trade even before real-user affordability is
approved; real-money deployment requires a separate future human capital
gate.

When flat after close `t`, evaluate `BUY` or `NO-TRADE`. If `BUY`, enter at
the `t+1` open and remain non-flat through the scheduled `t+6` exit open. No
new entry signal may be acted upon while the position is open. Because the
position was not flat at the close immediately preceding its scheduled exit,
do not exit and re-enter at the same `t+6` open. The next eligible decision is
after the close of the exit day. Positions never overlap.

For completed trades in chronological exit order, define normalized sequential
equity independently from fixed-lot yen PnL:

```text
E_0 = 1.0
E_i = E_(i-1) * (1 + net_trade_return_i)
Peak_i = max(E_0 ... E_i)
Drawdown_i = (Peak_i - E_i) / Peak_i
MAX_DRAWDOWN = max(Drawdown_i)
```

Use this exact normalized equity definition for historical criterion D and the
forward maximum-drawdown criterion. Do not invent starting yen capital to
alter the drawdown gate. Fixed-100-share net-yen PnL remains separately
reported and is used for positive-PnL criteria.

## 7. Historical screen

The fixed historical screen covers `2023-01-01` through `2026-08-31`. Only
trades whose scheduled exit is fully contained by `2026-08-31` count. Because
the security identity is selected using a current constituent universe, this
screen is supporting evidence only, not fully prospective evidence. Its sole
purpose is to decide whether time should be spent on future forward paper
observation.

The base-screen continuation criteria are frozen:

- A: `completed_trades >= 20`;
- B: base net PnL after costs is positive;
- C: among complete calendar years 2023, 2024, and 2025, at least two have
  positive realized net PnL;
- D: `MAX_DRAWDOWN` from the exact normalized sequential equity definition is
  at most `0.15`;
- E: largest winning-trade contribution to total positive net-yen PnL is at
  most `0.50`, with no positive trades failing the criterion;
- F: the 3x3 frozen robustness acceptance check covers:
  thresholds `0.004`, `0.006`, `0.008` crossed with round-trip frictions
  `0.002`, `0.003`, `0.005`.

The base scenario is threshold `0.006` and friction `0.003`. The base must be
positive and at least 6 of 9 scenarios must be positive. The grid is a frozen
robustness acceptance check; it must not choose a new threshold or cost after
results are seen.

Report buy-and-hold over the same period, cash baseline, exposure fraction,
trade count, win rate, average trade, maximum drawdown, maximum required
100-share lot notional, and yearly PnL. Buy-and-hold is diagnostic only and
not a pass criterion.

Assign each completed trade's realized net PnL to the calendar year of its
`exit_date`; criterion C uses this exact attribution. For positive net-yen
trades:

```text
positive_pnl_total = sum(max(net_pnl_yen_i, 0))
largest_winner_contribution =
    max(max(net_pnl_yen_i, 0)) / positive_pnl_total
```

If there are no positive trades, criterion E fails.

For the exposure diagnostic, count five held trading intervals for every
completed non-overlapping trade:

```text
exposure_intervals = sum of the five held trading intervals per completed trade
screen_intervals = observed trading-row transitions from the first observed
    row on/after 2023-01-01 through the final observed row on/before 2026-08-31
exposure_fraction = exposure_intervals / screen_intervals
```

Exposure is diagnostic only and is not a pass criterion.

For reproducible buy-and-hold, enter 100 shares at the first observed raw open
on or after `2023-01-01` and exit at the final observed raw open on or before
`2026-08-31`. Adjust the share count for intervening splits using the same
split-ratio convention, apply base entry and exit rates `0.0015` and
`0.0015`, and exclude dividends. Buy-and-hold remains diagnostic only. The
cash baseline is `0` yen PnL.

```text
if completed_trades < 20:
    HISTORICAL_SCREEN_RESULT=INSUFFICIENT_ACTIVITY
elif any(B, C, D, E, F) fails:
    HISTORICAL_SCREEN_RESULT=REJECT
elif all(A, B, C, D, E, F) pass:
    HISTORICAL_SCREEN_RESULT=PASS_TO_FORWARD_PAPER
```

There is no same-study tuning, retry until profitable, favorable-period
replacement, holdout redraw, or methodology change after any outcome.

## 8. Forward paper evidence

Forward paper trading is not authorized by this design task. If the historical
screen later passes and GPT independently approves, freeze the exact ticker,
model, features, threshold, cost, and execution rules before the forward
window begins.

The target forward observation is at least 60 JPX trading sessions and at
least 8 completed trades. If 8 trades have not occurred at 60 sessions,
continue unchanged up to a maximum of 120 sessions. Fewer than 8 completed
trades after 120 sessions gives:

```text
FORWARD_RESULT=INSUFFICIENT_EVIDENCE
```

No parameter changes are permitted during observation. Supportive forward
criteria are net PnL positive, normalized maximum drawdown at most `0.15`,
largest winning-trade contribution at most `0.50`, and no governance or data-
integrity violation. Passing gives only:

```text
FORWARD_EVIDENCE_SUPPORTIVE
```

It does not establish guaranteed future profitability. Real-money execution
remains separately human-gated.

## 9. Anti-sprawl and development budget

Before the first historical screen, the active implementation budget is:

```text
ACTIVE_IMPLEMENTATION_BUDGET_MINUTES=240
MAX_SUBSTANTIVE_REMEDIATION_ROUNDS=2
```

If implementation cannot reach independently reviewed readiness within either
limit:

```text
PROJECT_DISPOSITION=PAUSE_V11
```

Do not automatically create V11B, V11C, or other successor studies. Parser or
software plumbing on already locked public bytes is not a new scientific
study, but it remains inside the same budget and remediation-round limits.

## 10. Research integrity and authority

Prohibit use of prior backtest profitability to choose the V11 ticker, use of
selection or comparison outputs for selection, ticker substitution, threshold
or feature tuning, model tuning, favorable-period replacement, provider
replacement after outcome, holdout redraw, retry until profitable, and real
broker execution.

This design does not authorize ticker selection, JPX or Yahoo acquisition,
historical price inspection, model fitting, backtesting, old profitability
output inspection, package/environment mutation, Slack sending, paper trading,
broker action, or a new human gate. It authorizes no protected/private data
access and no T0.

```text
V11_DESIGN_FROZEN=false
V11_TICKER_SELECTED=false
V11_PUBLIC_ACQUISITION_AUTHORIZED=false
V11_HISTORICAL_SCREEN_AUTHORIZED=false
V11_FORWARD_PAPER_AUTHORIZED=false
V11_REAL_TRADING_AUTHORIZED=false
future_profitability_established=false
```

The V10C/V10D route remains `PAUSE_CURRENT_ROUTE`. A future study may be
considered only through a new explicit human decision and the normal design,
freeze, implementation, and independent-review sequence.
