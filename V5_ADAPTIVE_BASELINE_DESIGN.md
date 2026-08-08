# V5-A Adaptive Portfolio Baseline — Design

## Scope and research status

V5-A is an exploratory, non-AI baseline for the hypothesis that selecting several
short pullbacks within medium-term uptrends, holding five trading days, using a
fixed ATR stop, and allocating by risk improves a ¥400,000 portfolio's return and
stability.  It is distinct from V4's formal hypothesis.  The 2017–2019 folds have
already been viewed and are not an unused final evaluation period.  This code must
not fetch prices, contact Yahoo, fit an AI model, tune a threshold, or run a
production evaluation.

## Fixed data boundary

* Universe: existing `FIXED_V4_300`, retaining its existing universe CSV and
  ticker-list SHA-256 values.
* Frozen read-only cache: `C:\taiki\hobbies\v4-meta-label-formal-cache`.
* Prices: 2015-01-01 through 2019-12-31 only; signals: 2016-04-01 through
  2019-12-31 only.  2020 onward is rejected.
* Yahoo access, cache acquisition, and price re-download are prohibited.
* The current-constituent universe has survivorship bias.  The anticipated usable
  population is 283 price-success tickers.
* Japanese equity quantities are whole 100-share lots only; fractional shares are
  never simulated.  Existing V4 cache files and formal artifacts are read-only.

## Candidate generation and ranking

Adjusted OHLC is derived using V4's adjustment factor.  Every signal-day candidate
has at least 252 rows of history, raw 60-day median turnover >= ¥100m, raw 60-day
median volume >= 50,000 shares, all required features finite, adjusted close above
MA60, `return_60d > 0`, `-5% <= return_5d <= 0`, and `close_to_ma20 >= -3%`.
It must also have D1 entry data, D5 exit data, no split from D1 through D5, and at
least one affordable/risk-permitted lot.  Candidates rank by `return_60d` descending,
then `return_20d` descending, then ticker ascending; at most 20 are retained per
day.  Unlike V4, a skipped candidate does not prevent trying lower ranks.

## Execution, stop, and allocation

Signal D0 is known after close.  D1 entry uses raw open only when it is <= D0 raw
close × 1.01; otherwise it records `ENTRY_GAP_TOO_HIGH`.  Fill price is open ×
1.0003.  D5 time exit is raw open × (1 - 0.0003), unless the reusable V4 execution
semantics trigger an earlier stop: gap through stop fills at raw open, otherwise at
the stop; both then use 0.10% stop slippage.  There is no profit target.

ATR14 is calculated from adjusted OHLC using causal true range.  The entry-time
stop is immutable: `clamp(1.8 * ATR14 / adjusted_close, 4%, 8%)`, with stop price
`entry_price * (1 - stop_pct)`.

Each independent fold starts at ¥400,000.  At most two positions may be open,
with ¥40,000 minimum available cash, ¥220,000 maximum entry capital per ticker,
no duplicate ticker, and no concurrent same-industry position.  Same-day exits are
pending cash and cannot fund entries.  Commission is zero.  Risk budget is ¥8,000:
lots are `min(floor(8000/(entry*100*stop_pct)), floor(min(220000, cash-40000)/(entry*100)))`.
Quantity is lots × 100.  Distinct skip reasons include all fixed reasons in the
task specification.

## Folds, metrics, gates, and artifacts

Folds are 2017, 2018, and 2019 (with retained V4-compatible fold metadata but no
training).  Per-fold and aggregate summaries include candidate/attempt/fill counts,
skips, P&L, equity, drawdown, distribution metrics, exit counts, holdings/allocation,
concentration, and all listed safety counters.  The gate records every stated
condition.  A safety violation is `BLOCKED`; otherwise all gate conditions produce
`V5_ADAPTIVE_BASELINE_PROMISING`, and any failure produces
`V5_ADAPTIVE_BASELINE_NOT_PROMISING`.

When separately authorized in the future, the evaluator atomically writes exactly
four files outside the repository: `summary.json`, `trades.csv`, `candidates.csv`,
and `daily_equity.csv`.  Synthetic smoke tests use only a temporary directory and
remove it afterward.
