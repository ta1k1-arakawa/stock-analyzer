# V12 Minimal Fixed-Strategy Single-Security Study Design

```text
document_type=V12_MINIMAL_FIXED_STRATEGY_SINGLE_SECURITY_DESIGN
status=DRAFT_AWAITING_GPT_REVIEW
study_id=V12_MINIMAL_FIXED_STRATEGY_SINGLE_SECURITY
design_purpose=MINIMAL_FIXED_STRATEGY_EVALUATION
authoritative_branch=v12-minimal-single-security-study
parent_commit=bd9ab0cb030d00b0e0fb4b563cbae15de947f02f
design_frozen=false
scope=DESIGN_DRAFT_ONLY
future_profitability_established=false
V11_PROJECT_DISPOSITION=PAUSE_V11
STRATEGY_METHODOLOGY_CHANGED=false
CORE30_UNIVERSE_SEMANTICS_CHANGED=true
```

This is a new study identity following the explicit human decision to proceed
after V11 reached its terminal `PAUSE_V11` disposition. V11 is not reopened,
repaired, or reinterpreted. V12 carries forward the V11 frozen strategy and
evaluation methodology in scientific meaning; its only scientific delta is
that the locked official source is treated as a 31-candidate universe rather
than assuming that the label `TOPIX Core30` implies exactly 30 constituents.

This design is prospective and design-only. It does not select a ticker,
acquire data, inspect historical outcomes, fit a model, run a backtest, read
the machine-local locked PDF, access protected/private data, send messages,
paper trade, or trade real money. It grants no execution authority.

## 1. Locked public source and candidate universe

V12 reuses only the already durably locked public JPX bytes. The design does
not read the machine-local locked artifact, refetch it, or authorize any JPX
network request.

```text
source_url=https://www.jpx.co.jp/news/6030/um3qrc0000023smr-att/mei2_12_size.pdf
snapshot_effective_date=2025-10-31
V12_CORE30_SOURCE_MODE=REUSE_EXISTING_LOCKED_PUBLIC_BYTES_ONLY
source_byte_count=1299485
source_sha256=b584ad25a182f4f17341ce2ebd77a23010957a777dde64100d71644bbff6e1ce
JPX_NETWORK_REQUIRED=false
JPX_REFETCH_AUTHORIZED=false
```

The candidate universe is every unique numeric four-digit security code from
the official constituent-list rows whose normalized `NEW classification`
equals exactly `TOPIX Core30`. The source table semantics are:

```text
No. | コード | 銘柄名 | OLD classification | NEW classification
```

The security code is taken from the shared `コード` column. The following
safe public-source facts are integrity guards before any ticker selection:

```text
V12_CORE30_CANDIDATE_COUNT=31
V12_CORE30_CANDIDATE_CODES_SHA256=acb8777834e2f836e0f815f739515f9ab6dd25ecbe8e1f9db77b15015b793344
ORIENTATION_GUARD_6857_MUST_BE_PRESENT=true
ORIENTATION_GUARD_6981_MUST_BE_ABSENT=true
```

The canonical candidate-set hash is computed from unique candidate codes in
numeric ascending order by concatenating each code followed by `"\n"`,
encoding the resulting text as UTF-8, and taking lowercase hexadecimal
SHA-256. The identical locked bytes must reproduce count 31, the exact
candidate-set hash, the presence guard, and the absence guard. If any guard
fails, stop. A parser or software mismatch is
`IMPLEMENTATION_FAILURE`; it is not a source `DATA_QUALITY_FAILURE` and is
not permission to refetch or substitute bytes.

The design must not expose or hardcode the remaining candidate identities.
No constituent may be dropped to force a count of 30. No favorable 30-of-31
subset, redraw, source substitution, or outcome-driven universe change is
permitted.

## 2. Profitability-independent exclusion and deterministic selection

Selection must not use historical strategy return, model score, validation
result, Sharpe, PnL, drawdown, prior selected-rules output, or any other
profitability result. In particular, selection must not inspect old
profitability outputs, `data/backtest_selection.yaml`,
`data/backtest_results`, or `data/backtest_comparison`.

Use the same profitability-independent exclusion source as V11. At the exact
frozen parent SHA below, exclude every numeric ticker whose filename exists
under `data/benchmark/ohlcv/*.csv`:

```text
EXCLUSION_PARENT_SHA=4fc91c19c5534d59e041876f5310b97e373eb1fc
```

The V12 deterministic seed and selection hash are frozen before any result
can be observed:

```text
SEED_TEXT=V12_MINIMAL_SINGLE_SECURITY|bd9ab0cb030d00b0e0fb4b563cbae15de947f02f
SELECTION_HASH=3c11aa5ef55452e3f1178339788bebf6c63f1cd20a903df46f699bccce16f8d3
```

Mechanically require `SELECTION_HASH` to equal the lowercase SHA-256 of the
UTF-8 bytes of `SEED_TEXT`. Form the eligible set as the candidate universe
minus the frozen-parent exclusions, sort eligible numeric codes ascending,
and require `eligible_count > 0`. The selected index is the full 256-bit
integer value of `SELECTION_HASH` modulo `eligible_count`; the selected ticker
is the code at that index.

Selection occurs only after this design receives GPT exact-SHA PASS and a
separate design-freeze approval. There is no alternative seed, fallback,
redraw, or ticker substitution because of price, lot affordability, data
quality inconvenience, model result, or profitability. After successful
selection, the ticker identity is frozen permanently.

```text
V12_TICKER_SELECTED=false
V12_TICKER_SUBSTITUTION_ALLOWED=false
V12_SELECTION_REDRAW_ALLOWED=false
```

## 3. Fixed historical data source and byte boundary

The historical/public source remains the Yahoo Finance Chart API with exactly:

```text
host=query1.finance.yahoo.com
endpoint=/v8/finance/chart/{ticker}.T
interval=1d
events=div,splits
includeAdjustedClose=true
```

The fixed historical acquisition window remains:

```text
2015-01-01 <= date < 2026-09-01
```

Later implementation may reuse the reviewed transport and parser behavior
from `src/v7_yahoo_collector.py` where applicable, without broadening parser
semantics. The first complete raw public payload must be durably preserved
and SHA-256 bound before semantic processing. A transport failure before a
complete first payload is `PLUMBING_FAILURE_RETRIABLE`. Once locked, there is
no refetch-until-pass, provider substitution, ticker substitution, or
outcome-driven acquisition. Any separately reviewed parser/software repair
may reprocess only identical locked bytes. An intrinsic fixed-byte failure is
`DATA_QUALITY_FAILURE`.

This V12 design does not authorize Yahoo acquisition, parser repair, or any
historical price read. Yahoo is a separate later real-network stage requiring
its own point-of-use authority.

## 4. Features

Use exactly these five price-ratio or return features:

1. `return_1d`
2. `return_5d`
3. `return_20d`
4. `volatility_20`
5. `close_to_ma20`

There is no volume feature, fundamental data, news, alternative data, feature
search, or feature tuning. The exact causal split-normalized construction is
fixed before any result is observed.

For each decision row `t`, use only split events whose effective date is less
than or equal to `t`. For any observed row `d <= t`:

```text
split_factor_t(d) = product of split ratios for events s where
    d < effective_date(s) <= t
split_ratio = post_split_shares / pre_split_shares = numerator / denominator
split_factor_t(d) = 1 when no such event exists
P_t(d) = raw_close(d) / split_factor_t(d)
```

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
may be parsed or validated as required by the reviewed Yahoo parser, but V12
feature construction uses only the causal raw-close/split-normalized series.
No volume enters a feature. Future split events with effective date greater
than `t` must not affect features at `t`.

A feature row requires every preceding observed row needed for its lookbacks;
rows without sufficient history are mechanically ineligible. Do not impute or
shorten a lookback. Missing or nonfinite required values after sufficient
history eligibility fail closed as `DATA_QUALITY_FAILURE`, rather than being
silently dropped to improve results. Training features for decision row `d`
use only values available at `d` and split events effective no later than
`d`.

## 5. Model and causal training

Use exactly one model pipeline:

```text
StandardScaler + Ridge regression
alpha=10.0
fit_intercept=true
minimum_training_examples=252
```

There is no LightGBM, model comparison, hyperparameter search, threshold
optimization, or alternative scaler behavior. Refit using expanding causal
history. For each decision date `t`, only training examples whose target exit
price is fully observable by `t` may enter training.

## 6. Target and timing

The decision occurs after the close on observed trading row `t`. The predicted
target is the gross return from the next observed raw open `O[t+1]` to the
raw open `O[t+6]`, a five-trading-interval holding period:

```text
entry_date = row t+1
exit_date = row t+6
R = product of split ratios with entry_date < effective_date <= exit_date
R = 1 when no such split event exists
target_5 = (raw_open(exit_date) * R / raw_open(entry_date)) - 1
holding_intervals=5
```

Split events between entry and exit are economically accounted for using the
frozen split ratio. Dividends are not added to strategy PnL. No intraday
target is permitted.

## 7. Fixed long-only trading rule

When flat, issue `BUY` if the predicted five-session return is at least
`0.006`; otherwise issue `NO-TRADE`. Entry is 100 shares at the next observed
raw open. Exit is the open five trading intervals after entry.

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
selling. For any scenario with total round-trip friction `f`, use symmetric
rates `f/2` per side. For one completed trade:

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
intermediate values; report maximum required 100-share entry notional. Fixed
100-share historical simulation does not authorize real-money deployment.

When flat after close `t`, evaluate `BUY` or `NO-TRADE`. If `BUY`, enter at
the `t+1` open and remain non-flat through the scheduled `t+6` exit open. No
new entry signal may be acted upon while the position is open. Because the
position was not flat at the close immediately preceding its scheduled exit,
do not exit and re-enter at the same `t+6` open. The next eligible decision is
after the close of the exit day. Positions never overlap.

For completed trades in chronological exit order, define normalized
sequential equity independently from fixed-lot yen PnL:

```text
E_0 = 1.0
E_i = E_(i-1) * (1 + net_trade_return_i)
Peak_i = max(E_0 ... E_i)
Drawdown_i = (Peak_i - E_i) / Peak_i
MAX_DRAWDOWN = max(Drawdown_i)
```

Use this exact normalized equity definition for historical criterion D and
forward maximum drawdown. Fixed-100-share net-yen PnL remains separately
reported and is used for positive-PnL criteria.

## 8. Historical screen

The fixed historical screen covers `2023-01-01` through `2026-08-31`. Only
trades whose scheduled exit is fully contained by `2026-08-31` count. Because
the security identity is selected from a current constituent universe, this
screen is supporting evidence only, not fully prospective evidence. Its sole
purpose is to decide whether time should be spent on future forward paper
observation.

The base-screen continuation criteria are frozen:

- A: `completed_trades >= 20`;
- B: base net PnL after costs is positive;
- C: among complete calendar years 2023, 2024, and 2025, at least two have
  positive realized net PnL;
- D: `MAX_DRAWDOWN` is at most `0.15`;
- E: largest winning-trade contribution to total positive net-yen PnL is at
  most `0.50`, with no positive trades failing the criterion;
- F: the frozen 3x3 robustness acceptance check crosses thresholds `0.004`,
  `0.006`, `0.008` with round-trip frictions `0.002`, `0.003`, `0.005`.

The base scenario is threshold `0.006` and friction `0.003`. The base must be
positive and at least 6 of 9 scenarios must be positive. The grid is a frozen
robustness acceptance check; it must not choose a new threshold or cost after
results are seen.

Report buy-and-hold over the same period, cash baseline, exposure fraction,
trade count, win rate, average trade, maximum drawdown, maximum required
100-share lot notional, and yearly PnL. Buy-and-hold is diagnostic only and
not a pass criterion. Assign each completed trade's realized net PnL to the
calendar year of its `exit_date`. For positive net-yen trades:

```text
positive_pnl_total = sum(max(net_pnl_yen_i, 0))
largest_winner_contribution =
    max(max(net_pnl_yen_i, 0)) / positive_pnl_total
```

If there are no positive trades, criterion E fails. For exposure:

```text
exposure_intervals = sum of the five held trading intervals per completed trade
screen_intervals = observed trading-row transitions from the first observed
    row on/after 2023-01-01 through the final observed row on/before 2026-08-31
exposure_fraction = exposure_intervals / screen_intervals
```

Exposure is diagnostic only. For reproducible buy-and-hold, enter 100 shares
at the first observed raw open on or after `2023-01-01` and exit at the final
observed raw open on or before `2026-08-31`. Adjust share count for intervening
splits using the same split-ratio convention, apply base entry and exit rates
`0.0015` and `0.0015`, exclude dividends, and report the cash baseline as
`0` yen PnL.

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

## 9. Forward paper evidence

Forward paper trading is not authorized by this design task. If the historical
screen later passes and GPT independently approves, freeze the exact ticker,
model, features, threshold, cost, and execution rules before the forward
window begins.

The target forward observation is at least 60 JPX trading sessions and at
least 8 completed trades. If 8 trades have not occurred at 60 sessions,
continue unchanged up to a maximum of 120 sessions. Fewer than 8 completed
trades after 120 sessions gives `FORWARD_RESULT=INSUFFICIENT_EVIDENCE`.

No parameter changes are permitted during observation. Supportive forward
criteria are net PnL positive, normalized maximum drawdown at most `0.15`,
largest winning-trade contribution at most `0.50`, and no governance or
data-integrity violation. Passing gives only `FORWARD_EVIDENCE_SUPPORTIVE`;
it does not establish guaranteed future profitability. Real-money execution
remains separately human-gated.

## 10. Anti-sprawl and development budget

Because V11 already established most implementation machinery, V12 freezes a
smaller implementation budget:

```text
ACTIVE_IMPLEMENTATION_BUDGET_MINUTES=120
MAX_SUBSTANTIVE_REMEDIATION_ROUNDS=1
```

Before the first historical screen, if implementation cannot achieve GPT
independent PASS within either limit:

```text
PROJECT_DISPOSITION=PAUSE_V12
```

Do not automatically create V12B, V12C, or V13. Locked-byte parser/software
plumbing remains inside this V12 budget and remediation limit; it does not
authorize a new study or a source refetch.

## 11. Research integrity and authority

The V12 design prohibits use of prior profitability for selection, selection
or comparison outputs for selection, ticker substitution, threshold or
feature tuning, model tuning, favorable-period replacement, provider
replacement after outcome, holdout redraw, retry until profitable, and real
broker execution. The only scientific change from V11 is the explicit
31-candidate Core30 universe semantics and V12 deterministic seed.

This design authorizes none of the following: ticker selection, real JPX
network, JPX refetch, Yahoo acquisition, historical price inspection, model
fitting, backtesting, historical profitability inspection, forward paper,
real trading, Slack sending, broker action, package/environment mutation, or
private/sealed access.

```text
V12_DESIGN_FROZEN=false
V12_TICKER_SELECTED=false
V12_PUBLIC_ACQUISITION_AUTHORIZED=false
V12_HISTORICAL_SCREEN_AUTHORIZED=false
V12_FORWARD_PAPER_AUTHORIZED=false
V12_REAL_TRADING_AUTHORIZED=false
V12_FUTURE_PROFITABILITY_ESTABLISHED=false
```

V11 remains terminally `PAUSE_V11`. Any later methodology change requires a
new explicit human decision and new design/freeze/review cycle.
