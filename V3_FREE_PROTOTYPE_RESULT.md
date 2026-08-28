# V3 free yfinance prototype result

## Classification

- `evaluation_type`: `SURVIVORSHIP_BIASED_RESEARCH_ONLY`
- `formal_backtest`: `false`
- `point_in_time_universe`: `false`
- `deployment_decision_allowed`: `false`
- `shadow_replacement_allowed`: `false`
- `reference_period_used`: `false`

- `decision`: `FREE_PROTOTYPE_NOT_PROMISING`

> This is a survivorship-biased research-only prototype using a current universe. It is not a formal historical backtest, does not establish deployability, and cannot replace shadow evaluation.

## Data

- Current universe acquisition: `2026-08-03T00:28:58+00:00`
- Selected / successful: `300` / `274`
- Evaluated dates: `2021-01-04` to `2025-03-27`
- Snapshot hash: `1caaec36328a822a7598a277fb14ab63826c6e4948e4a262743a93d4ed9d47fc`

## OOF prediction

- MAE / RMSE / Huber: `1.9981854688211813` / `2.867180339719765` / `3.4736771058129143`
- Spearman: `0.015889979039380876`
- Daily IC mean / median / positive rate: `0.01771163909725169` / `0.01645606817513958` / `0.5507246376811594`
- Top-decile minus all-candidate return: `-0.007074576495587417` percentage points

## Portfolio

- Profit / ending equity: `-51797.56761559911` / `248202.4323844009` JPY
- Fold profits: `[-73264.42084530194, 49.31357451216354, 21417.539655190663]`
- Maximum drawdown: `31.052900099670232`%
- Closed trades / win rate: `439` / `0.4646924829157175`

## Pre-registered conditions

- FAIL: `price_coverage_at_least_90_percent`
- FAIL: `all_fold_spearman_positive`
- FAIL: `overall_spearman_above_0_02`
- PASS: `mean_daily_ic_positive`
- PASS: `daily_ic_positive_rate_above_52_percent`
- FAIL: `top_decile_beats_all`
- PASS: `two_positive_profit_folds`
- PASS: `two_folds_beat_random_median`
- PASS: `two_folds_beat_best_return_baseline`
- FAIL: `max_drawdown_at_most_25_percent`
- PASS: `at_least_150_closed_trades`
- PASS: `cash_and_order_invariants`
- PASS: `future_access_zero`
- PASS: `deterministic`

No J-Quants data, post-2025-03-31 prices, reference replay, shadow results, real-order code, or raw market data was used or committed.
