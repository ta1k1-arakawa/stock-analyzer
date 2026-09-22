# V13 Conditional Cross-Sectional Short-Horizon Design Draft

```text
STUDY_ID=V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON
DESIGN_STATUS=DRAFT_UNFROZEN
PARENT_SHA=f9c38ad771710ffd157ac4fad0da15185db82707
V12_DISPOSITION=PAUSE_V12_UNCHANGED
PRIMARY_OBJECTIVE=REALISTIC_FUTURE_SHORT_HORIZON_NET_PROFITABILITY
FUTURE_PROFITABILITY_ESTABLISHED=false
```

This document transcribes the binding methodology in GitHub Issue #20. The human adopted the V13 direction on 2026-09-22 after V12's terminal pause. This task creates a design draft only. V13 is a new scientific identity and does not reinterpret V3–V12 results. No design freeze, acquisition, model fit, backtest, forward paper, broker, or trading authority follows from this draft.

## Hypothesis and evidential scope

Japanese individual-stock short-horizon returns may contain conditional cross-sectional structure distinguishing continuation from reversal. Industry-relative price moves, intraday versus overnight moves, trading-activity/liquidity proxies, distance to the 52-week high, and market-state variables may allow a common cross-sectional nonlinear model to rank stocks by net return from the next session's open to the third session's close better than linear and simple momentum/reversal baselines after realistic costs.

Scientific background is rationale only, with no paper retrieval or methodology change in this task: Miwa, “Short-Term Return Reversals and Intraday Transactions”, QJF, DOI 10.1142/S2010139219500022; Medhat & Schmeling, “Short-term Momentum”, RFS 35(3), DOI 10.1093/rfs/hhab055; Chen, Stivers & Sun, “Short-term momentum and reversals, turnover, and a stock’s price-to-52-week-high ratio”, JEF 2024, DOI 10.1016/j.jempfin.2024.101556; “Short-term reversal persists globally—If properly measured”, Economics Letters 267 (2026), DOI 10.1016/j.econlet.2026.113113.

The first V13 study is a free/public historical signal-viability diagnostic, separated from later point-in-time paid-data or prospective-forward confirmation.

```text
HISTORICAL_VIABILITY_EVIDENTIAL_LEVEL=DIAGNOSTIC_ONLY
SURVIVORSHIP_BIAS_ACKNOWLEDGED=true
REGIME_KNOWLEDGE_CONTAMINATION_ACKNOWLEDGED=true
FORMAL_POINT_IN_TIME_PROOF=false
FORWARD_PROOF=false
```

A historical PASS may justify designing later point-in-time/forward confirmation. It does not establish future profitability.

## Universe and provenance contract

The starting pool uses the V8 audited `ELIGIBLE_CURRENT_ONLY` semantics: current JPX Prime/Standard domestic common stocks with a 4-character code. Before any future real acquisition, rerun and reconcile exposure provenance at the then-current exact HEAD. Exclude all `FIXED_V4_300`, all `LEGACY_8` outside that set, and any additional ticker that the updated exposure audit proves had historical price outcomes acquired before V13 universe creation. No ticker identity is selected or exposed by this design task.

```text
V13_UNIVERSE_SIZE=500
V13_UNIVERSE_SEED=V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON|f9c38ad771710ffd157ac4fad0da15185db82707
ORDER_KEY=SHA256(UTF8(V13_UNIVERSE_SEED + "|" + code)), then numeric code ascending
SELECTION=first 500 after exclusions
```

The future universe manifest must bind the exact official JPX source snapshot, acquisition UTC, raw SHA-256 and byte count, full eligible-list hash, exact exclusion-list hash, ordering rule, selected-500 hash, and implementation SHA. Failure to establish reproducibility is a BLOCK before price acquisition. The current-only universe is survivorship-biased and is only for viability.

## Public historical data and price basis

```text
PRICE_SOURCE=Yahoo Finance chart daily OHLCV
RAW_PRICE_WINDOW=2015-01-01..2025-12-31
INITIAL_FEATURE_TRAINING_SIGNAL_WINDOW=2016-01-01..2019-12-31
OOS_VIABILITY_SIGNAL_WINDOW=2020-01-01..2025-12-31
2026_PRICE_HISTORY_USED=false
```

The 2015 span is warm-up only. JPX current metadata supplies current 33-sector classification for the diagnostic universe; applying it historically is a known diagnostic-only bias. Model features use a split-adjusted, non-dividend-total-return price basis. Actual execution and affordability use raw historical Open/Close; traded-value features use raw `Close * Volume`. If the future implementation cannot deterministically establish the required split-only adjustment semantics from the specified source contract, STOP before the historical viability run. Do not silently substitute total-return adjusted prices or another provider. This design task authorizes no data or source-network action.

## Signal, label, and execution timing

For a signal date `t`, every feature uses information available no later than its close, and ranking occurs after that close. Intended entry is raw Open at `t+1`; intended exit is raw Close at `t+3`, where `t+1`, `t+2`, and `t+3` are the first three trading sessions after the signal. There is no take-profit, stop-loss, trailing stop, early exit, or discretionary exit in the first study. Target and simulated exit semantics must match exactly.

```text
TARGET=realized_net_return_percent from t+1 Open to t+3 Close
BASE_ENTRY_FRICTION=0.10%
BASE_EXIT_FRICTION=0.10%
STRESS_ENTRY_FRICTION=0.20%
STRESS_EXIT_FRICTION=0.20%
COMMISSION_SEPARATE=0
```

Use identical friction mechanics in labels and portfolio execution. Evaluate the stress-cost portfolio without retraining. Rows lacking required future prices are invalid training labels. If an actually selected/executed simulated trade loses its required exit, classify `DATA_QUALITY_FAILURE`; never silently drop it.

## Signal-date eligibility

A ticker is rank-eligible at `t` only when all conditions hold:

- At least 252 prior valid trading observations exist and all required features are finite.
- Raw Close and Volume are positive.
- `raw_close_t * 100 <= 270000 JPY`.
- Trailing 20-session median raw yen traded value is at least `100,000,000 JPY`.
- Its current 33-sector group has at least five valid study-universe members on that date for sector-relative calculations.

No imputation is allowed; a failure omits that ticker for the signal date. The 270,000-JPY cap is the fixed 10% signal-date affordability buffer for the 300,000-JPY cash budget. Actual affordability is checked again at the next Open.

## Fixed features and causal transforms

All stock-return features use split-adjusted prices, expressed as decimal returns or consistently scaled equivalents. The stock and relative-price features are exactly:

```text
ret_1
ret_3
ret_5
ret_20
intraday_1 = close_t / open_t - 1
overnight_1 = open_t / close_(t-1) - 1
sector_rel_ret_1
sector_rel_ret_3
sector_rel_ret_5
sector_rel_ret_20
market_rel_ret_1
market_rel_ret_3
market_rel_ret_5
market_rel_ret_20
```

`sector_rel_ret_k = stock_ret_k - median(stock_ret_k for same current 33-sector)`. `market_rel_ret_k = stock_ret_k - median(stock_ret_k for all valid V13 study-universe members)`.

The trading-activity, liquidity, and market-state features are exactly:

```text
log_traded_value_ratio_20 = log((raw_close_t * raw_volume_t) / median_20(raw_close * raw_volume))
log_median_traded_value_20 = log(median_20(raw_close * raw_volume))
log_amihud_20 = log(mean_20(abs(ret_1) / max(raw_close * raw_volume, 1)))
volatility_20 = std_20(ret_1)
dist_52w_high = close_t / max_252(close) - 1
breadth_1 = fraction of valid universe with ret_1 > 0
breadth_5 = fraction of valid universe with ret_5 > 0
market_median_ret_1
market_median_ret_5
cross_section_dispersion_1 = std across valid universe of ret_1
market_median_volatility_20
```

Within each signal date, winsorize stock-varying features at cross-sectional 1st/99th percentiles, then z-score with that date's valid study universe only. Market-state features identical across tickers on a date are not cross-sectionally z-scored. No future information may enter these transforms. Do not add RSI, MACD, Bollinger Bands, ADX, arbitrary TA libraries, embeddings, news, fundamentals, or any other feature in this first study.

## Fixed models and walk-forward training

Exactly two common-across-tickers models are defined, with no grid search or hyperparameter tuning. The primary V13 strategy is LightGBM, chosen before observing results:

```text
model=LightGBM regressor
objective=huber
alpha=0.9
learning_rate=0.03
n_estimators=400
num_leaves=15
min_child_samples=100
subsample=0.8
subsample_freq=1
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=1.0
random_state=20260922
n_jobs=1
deterministic=true
force_col_wise=true
```

The linear comparator is `StandardScaler` followed by `Ridge(alpha=10.0, fit_intercept=true)`. Ridge uses exactly the same input rows and features. If Ridge succeeds while LightGBM fails frozen V13 primary criteria, V13 does not switch to Ridge; a successor needs an explicit new design decision.

No random split is allowed. For each calendar month in the 2020–2025 OOS window, build one model using only rows whose label and exit are fully known by the final trading session of the prior calendar month. Use expanding training beginning with 2016 signal rows. Predict the full month without refitting on any outcomes in that month; retrain only at the next calendar-month boundary. First OOS predictions are January 2020. Do not tune features, horizon, friction, model parameters, eligibility thresholds, or model using 2020–2025 outcomes.

## Single-capital portfolio simulator

```text
STARTING_CASH=300000 JPY
LONG_ONLY=true
SHORTING=false
LEVERAGE=false
MAX_CONCURRENT_POSITIONS=1
LOT_SIZE=100
```

When flat after signal date `t`, sort eligible candidates by LightGBM predicted net-return score descending, with numeric code ascending for exact ties. Only scores strictly greater than zero qualify. At `t+1` Open, traverse the frozen ranking and take the first candidate with a valid Open and at least one 100-share lot affordable after entry friction. Buy the largest affordable multiple of 100 shares. If none is executable, record `NO_FILL` and stay in cash. Hold exactly through `t+3` Close and exit with exit friction. Open no new position while one is held; sale proceeds are available for the next trading session. Negative cash, duplicate allocation, overlapping positions, and double use of cash are forbidden. Ranking while a position is open may be computed for diagnostics but creates no order.

## Baselines and comparators

All comparators use the identical universe, signal-day eligibility, budget, lot size, timing, no-overlap rule, and cost assumptions:

1. `RIDGE`: Ridge predicted-score ranking with positive-score gate.
2. `SECTOR_REL_REVERSAL_1D`: most negative `sector_rel_ret_1`.
3. `SECTOR_REL_MOMENTUM_20D`: most positive `sector_rel_ret_20`.
4. `RANDOM_500`: 500 deterministic random-selection portfolio paths from the eligible set, with seeds `202609220000..202609220499`.
5. `CASH`: zero-trade baseline.

No comparator parameter may be tuned after outcomes are seen.

## Required metrics

Report at minimum for LightGBM and relevant baselines: base-cost total net profit and ending equity; stress-cost total net profit; daily mark-to-market maximum drawdown; closed-trade count; win rate; profit factor; mean and median trade net return; positive calendar-year count and per-year PnL; best-year positive-profit share; maximum single-ticker positive-profit share; maximum current-sector positive-profit share; exposure fraction; `NO_FILL`/affordability skips; daily cross-sectional Spearman IC mean/median and per-year mean; fraction of signal dates with positive IC; and mean net target return of the predicted top decile versus the full eligible cross-section. Daily equity marks open positions at that day's raw Close without pretending liquidation.

## Frozen historical-viability criteria

All A–Q must pass to return `CONTINUE_TO_FORMAL_CONFIRMATION_DESIGN`:

```text
A_BASE_COST_NET_PROFIT_GT_0=true
B_POSITIVE_CALENDAR_YEARS_AT_LEAST=4_of_6
C_MAX_DRAWDOWN_PCT_LE=20
D_CLOSED_TRADES_GE=150
E_LIGHTGBM_NET_PROFIT_GT_RIDGE=true
F_LIGHTGBM_NET_PROFIT_GT_SECTOR_REL_REVERSAL=true
G_LIGHTGBM_NET_PROFIT_GT_SECTOR_REL_MOMENTUM=true
H_LIGHTGBM_NET_PROFIT_GT_RANDOM_95TH_PERCENTILE=true
I_MEAN_DAILY_CROSS_SECTIONAL_SPEARMAN_IC_GT=0.01
J_POSITIVE_MEAN_IC_YEARS_AT_LEAST=4_of_6
K_TOP_DECILE_MINUS_ALL_MEAN_NET_RETURN_GT_0=true
L_TOP_DECILE_SPREAD_POSITIVE_YEARS_AT_LEAST=4_of_6
M_STRESS_0.20PCT_PER_SIDE_NET_PROFIT_GE_0=true
N_BEST_YEAR_POSITIVE_PROFIT_SHARE_LE=50%
O_MAX_SINGLE_TICKER_POSITIVE_PROFIT_SHARE_LE=25%
P_MAX_SINGLE_SECTOR_POSITIVE_PROFIT_SHARE_LE=40%
Q_SAFETY_AND_LEAKAGE_INVARIANTS_PASS=true
```

If any criterion fails, `V13_VIABILITY_RESULT=STOP_HYPOTHESIS_NOT_PROMOTED` and `same_study_parameter_rescue=false`. Do not rescue V13 through other holding periods, TA indicators, LightGBM grids, thresholds, stop losses, universe sizes, liquidity floors, or cost assumptions. A successor requires an explicit new methodology decision; event-driven strategies are only a possible later direction. Even if A–Q pass, `future_profitability_established=false` and `historical_viability_pass_is_not_deployment_proof=true`. A PASS permits only later design of stronger confirmation, preferably point-in-time universe/data and ultimately prospective forward evidence. Do not automatically purchase paid data.

## Future stage and human-gate sequence

1. `DESIGN_DRAFT` — this Issue only.
2. GPT exact-SHA review.
3. Human design-freeze approval.
4. Freeze-approval record and GPT review.
5. Offline/synthetic feasibility implementation and tests.
6. GPT implementation review.
7. Separate human gate for retriable public JPX/Yahoo acquisition.
8. Reproducible universe manifest and historical data lock.
9. Pre-run safe validation.
10. Separate one-shot human authorization for the first outcome-bearing V13 historical viability run.
11. Result adjudication against A–Q.
12. Only on PASS, design a stronger point-in-time/forward confirmation study.

Public acquisition plumbing may be `RETRIABLE_PUBLIC_PLUMBING` only after an explicit future human gate and frozen scope. The first outcome-bearing historical viability run is `STATISTICALLY_IRREVERSIBLE_GATE` and one-shot. No private/sealed partition is required for this diagnostic first study.

## Anti-sprawl and synthetic feasibility budget

```text
ACTIVE_IMPLEMENTATION_BUDGET_MINUTES_BEFORE_FIRST_MEASUREMENT=240
MAX_SUBSTANTIVE_REMEDIATION_ROUNDS_BEFORE_FIRST_MEASUREMENT=2
INITIAL_IMPLEMENTATION_ROUND=0
SUBSTANTIVE_REMEDIATION_ROUNDS_USED=0
SUBSTANTIVE_REMEDIATION_ROUNDS_REMAINING=2
```

These are future implementation-stage limits; the implementation has not started. Track budget and rounds from round zero in the active Issue/state. GPT may identify design defects before freeze. After freeze, each remediation remains subject to the repository pre-remediation budget gate. Future feasibility work must prove the complete synthetic path: universe manifest → feature calculation → monthly as-of training → ranking → affordability fallback → 100-share execution → three-session exit → daily equity → A–Q metric computation. It uses no real network acquisition or outcome observation. If the complete path cannot reach GPT PASS within the frozen implementation budget, pause V13 before historical acquisition.

## Current authority boundary

```text
V13_DIRECTION_HUMAN_ADOPTED=true
V13_DESIGN_FROZEN=false
V13_HUMAN_DESIGN_FREEZE_APPROVED=false
V13_PUBLIC_ACQUISITION_AUTHORIZED=false
V13_HISTORICAL_VIABILITY_AUTHORIZED=false
V13_MODEL_FIT_AUTHORIZED=false
V13_BACKTEST_AUTHORIZED=false
V13_FORWARD_PAPER_AUTHORIZED=false
V13_REAL_TRADING_AUTHORIZED=false
V13_FUTURE_PROFITABILITY_ESTABLISHED=false
future_profitability_established=false
```

This draft makes no ticker selection, raw/private/locked payload read, historical-price inspection, model fit, backtest, package/environment mutation, forward paper, or broker/trading action. It does not modify V3–V12 artifacts or claims. Any binding-specification inconsistency that requires a methodological decision must be sent back as `CHATGPT_DECISION_REQUIRED`, without executor repair.
