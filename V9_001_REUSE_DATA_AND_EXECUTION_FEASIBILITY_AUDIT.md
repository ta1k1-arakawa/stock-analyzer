# V9_001 Reuse, Data, and Execution Feasibility Audit

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=PREFREEZE_REPOSITORY_AUDIT
audited_head=65e6aa02d9d790c3a02ebd9c3f67492744d8e236
methodology_changed=false
design_frozen=false
```

This is a read-only repository feasibility audit. Classifications describe
implementation fit only; reuse requires an explicit future V9 design binding.

## A. Portfolio and execution

| Component | Exact implementation | Current semantics | Classification |
|---|---|---|---|
| Shared cash and pending proceeds | `src/trade_simulator.py:simulate_portfolio`; `src/v7_capacity_engine.py:CausalEventEngine` / `EngineState` | One shared cash account; released proceeds are delayed to the next calendar/engine day. | REUSE_WITH_V9_ADAPTER |
| Reserved/bound cash | `src/v7_capacity_engine.py:V7EngineParameters`, `CausalEventEngine.phase2_execute_entries`; `src/v6_a_r2_causal_breakout.py:CausalEventEngine` | Cash reserve and per-position capital checks precede debit; no distinct V9 auction-order reservation state. | REUSE_WITH_V9_ADAPTER |
| Same-day competition/order ranking | `src/trade_simulator.py:simulate_portfolio`; `src/v5_b_candidate_ranker.py:simulate_portfolio`; `src/v7_capacity_engine.py:CausalEventEngine.phase2_execute_entries` | Deterministic sorted entry orders; V5-B uses rank/ticker and V7 consumes preregistered candidate rank. | REUSE_WITH_V9_ADAPTER |
| Position limit | `src/v7_capacity_engine.py:V7EngineParameters.max_open_positions`; `src/v6_a_r2_causal_breakout.py:CausalEventEngine.phase2_execute_entries` | Hard frozen limits (V7 only 2 or 3; V6-R2 2). | REPLACE_FOR_V9 |
| Quantity sizing | `src/trade_simulator.py:simulate_portfolio` | Calculates quantity from allocation and configurable `lot_size`; this is the only inspected portfolio path that natively permits lot size 1. | REUSE_WITH_V9_ADAPTER |
| Round-lot assumptions | `src/v5_b_candidate_ranker.py:QUANTITY`; `src/v5_a2_fixed100_stop_study.py:_run_arm`; `src/v6_a_r2_causal_breakout.py:CausalEventEngine`; `src/v7_capacity_engine.py:V7EngineParameters` | These paths hard-code or freeze quantity 100; V5 adaptive also uses `LOT_SIZE=100`. | REPLACE_FOR_V9 |
| Entry timing/price | `src/v6_a_r2_causal_breakout.py:CausalEventEngine.phase2_execute_entries`; `src/v7_capacity_engine.py:CausalEventEngine.phase2_execute_entries` | D0 signal, D1 raw Open, with 3 bp entry slippage and a D1 gap rule. | REPLACE_FOR_V9 |
| Exit timing/price | `src/v6_a_r2_causal_breakout.py:CausalEventEngine.phase3_execute_exits`; `src/v7_capacity_engine.py:CausalEventEngine.phase3_execute_exits` | Planned exit at raw Open with 3 bp exit slippage; proceeds next day. | REPLACE_FOR_V9 |
| Stop execution/gap handling | `src/trade_simulator.py:simulate_execution`; `src/v5_adaptive_portfolio.py:_execution` | Stop checks intraday Low; a gap through stop uses Open rather than an optimistic stop price. | REUSE_WITH_V9_ADAPTER |
| Slippage and fees | `src/trade_simulator.py:simulate_execution`, `simulate_portfolio`; `config.yaml:ai_params`; V6/V7 engine parameters | Generic simulator has configurable commission and slippage. V6/V7 use fixed 3 bp open slippage and no fee ledger. | REUSE_WITH_V9_ADAPTER |
| Daily equity / MTM DD | `src/v7_capacity_engine.py:CausalEventEngine.phase4_record_equity`; `src/v6_a_r2_causal_breakout.py:fold_max_drawdown`; `src/v5_b_candidate_ranker.py:calculate_metrics` | Book and raw-Close MTM equity, pending proceeds, and drawdown are recorded. | REUSE_WITH_V9_ADAPTER |
| Skipped/rejected orders | `src/v7_capacity_engine.py:CausalEventEngine` and `SKIP_REASONS`; `src/v6_a_r2_formal.py:compute_fold_metrics` | Ledgered skipped rows and reason counts; reasons are based on the old opening execution rules. | REUSE_WITH_V9_ADAPTER |

The V7 engine has stronger causal/event invariants, but its documented study
scope is forward-only and its parameter validation fixes the V7 rules. V9 must
not reuse V7 forward state, evidence, or authority.

## B. Data and leakage

| Area | Exact implementation / finding | V9 feasibility finding |
|---|---|---|
| Raw Close and AdjClose | `src/v7_yahoo_collector.py:parse_chart_payload` preserves `raw_close` and `adj_close`; `src/v5_b_candidate_ranker.py:_one_features` derives adjusted OHLC from `AdjClose/Close`. | Reusable provenance/representation pattern; execution-price treatment needs V9 binding. |
| OHLCV source | `src/v7_yahoo_collector.py:fetch_chart_once` and `src/fetchers/yfinance.py:YFinanceFetcher` are Yahoo paths. | No V9 network use is authorized by this audit. |
| Corporate actions | `parse_chart_payload` records split events and separate adjusted close; `src/v7_capacity_engine.py:phase2b_check_open_position_splits` blocks split-spanning open positions. | Reuse only with a V9 corporate-action policy. |
| Revisions/cache locking | `src/v8_historical_acquisition.py:acquire_historical_block_bundle` records raw payload hashes, canonical rows, split hashes, and rejects duplicate ticker/date rows; `src/benchmark.py:FixedOHLCVLoader` verifies immutable local snapshot hashes. | Reusable infrastructure concepts, not V9 authority or a V9 acquisition design. |
| Feature warm-up / causality | `src/v5_b_candidate_ranker.py:_one_features` requires 252 rows and slices through signal date; `src/v4_meta_label_mvp.py:build_feature_frame` applies a 252-row preliminary-eligibility condition. | Existing semantics are causal, but windows/features are not V9 decisions. |
| Signal-date causality | `src/v6_a_r2_causal_breakout.py:validate_candidate_schema` enforces signal before next-day entry; `src/v7_capacity_engine.py:read_engine_price` guards future reads. | Guard pattern reusable; close-auction same-day semantics require replacement/design. |
| Cross-sectional ranking | `src/v4_meta_label_mvp.py:build_feature_frame` has same-date percentile/median/breadth features; `src/v5_b_candidate_ranker.py:build_features` ranks selected candidates per signal date. | Partial reusable implementation; full-universe eligibility and V9 score protocol are unresolved. |
| Ticker identities | `src/v4_meta_label_mvp.py:build_feature_frame` emits ticker/industry/market metadata; `src/v5_b_candidate_ranker.py:build_features` carries ticker, industry, rank, and uses ticker as deterministic tie-break. | Model feature exclusion is not mechanically enforced across existing paths; V9 must implement and test it. |
| Universe / survivorship / delistings | `src/v4_meta_label_mvp.py:load_fixed_universe`, `config.yaml`, and historical modules use fixed/current lists. No inspected code maintains dated membership or a complete delisting-aware security master. | `POINT_IN_TIME_UNIVERSE_FEASIBILITY=PARTIAL_NOT_PROVEN`; no point-in-time historical universe is supported without new methodology/data engineering. |
| Missing bars / duplicates | `src/v7_yahoo_collector.py:parse_chart_payload` separates invalid rows and rejects duplicate trading dates; V8 acquisition applies a quality gate with no fill/interpolation/imputation. | Reusable validation primitives; V9 missing-bar policy remains a GPT decision. |
| Trading calendar | `src/v7_jpx_calendar.py:CalendarSnapshot`, `is_jpx_trading_day`, `next_jpx_trading_day`, and `generate_engine_days` provide a JPX holiday snapshot utility. | Reusable with a V9 calendar/source binding. |

## C. Model infrastructure

| Capability | Existing evidence | Classification / gap |
|---|---|---|
| Pooled rows and models | `src/v5_b_candidate_ranker.py:prepare_dataset`, `fit_year`, `walk_forward_predict` pool ticker rows. | REUSE_WITH_V9_ADAPTER. |
| LightGBM | `requirement.txt` includes `lightgbm`; V5-B imports `LGBMRegressor` with fixed deterministic parameters. | REUSE_WITH_V9_ADAPTER; shallow parameterization is not frozen. |
| Ridge / sklearn linear model | `requirement.txt` includes `scikit-learn`; no inspected source imports or constructs `Ridge`. | REPLACE_FOR_V9: Ridge baseline implementation, specification, and tests are absent. |
| Feature standardization | No inspected source uses `StandardScaler` or an equivalent fit-on-training-only transformer. | REPLACE_FOR_V9. |
| Cross-sectional targets/ranks | V4/V5 provide same-date feature ranks; V5 target is single-name D5 realized return. | REPLACE_FOR_V9 target/rank protocol. |
| Walk-forward / purging | V5-B has annual expanding cutoff and requires training exits before cutoff. No purged/embargo implementation was found. | REUSE_WITH_V9_ADAPTER for expanding chronology; REPLACE_FOR_V9 for purging/embargo if chosen. |
| Determinism | V5-B `MODEL_PARAMS` fixes `random_state`, `n_jobs=1`, `deterministic=True`; canonical hashes exist in V7/V8 helpers. | REUSE_WITH_V9_ADAPTER. |

## D. Benchmark infrastructure

`src/benchmark.py:FixedOHLCVLoader` is an immutable local OHLCV loader, not a
TOPIX or portfolio-benchmark engine. No inspected source/test/config supports a
TOPIX or tradable TOPIX proxy, equal-weight-universe benchmark, random-K
same-capital Monte Carlo, or benchmark-specific cash/quantity accounting.
Those components are missing and must be newly specified and implemented after
methodology resolution.

## E. Preserved exposure constraints

Per `V8_DATA_EXPOSURE_AUDIT.md`, FIXED_V4_300 historical outcomes are
contaminated development data and cannot become untouched OOS. The strongest
unused historical axis is cross-sectional. A current-constituent historical
evaluation has survivorship bias. V7 forward evidence cannot be silently
reused or tuned on. This audit does not reclassify any exposure.

## F. GPT-supplied current broker facts

```text
source_class=GPT_WEB_VERIFIED_OFFICIAL_SBI
verification_date=2026-08-23
SBI_S_share_internet_buy_fee_yen=0
SBI_S_share_internet_sell_fee_yen=0
TSE_Prime_Standard_Growth_order_window=10:30-14:00_target_same_day_closing_auction_or_closing_price
accepted_S_share_order_may_fail_to_execute=true
closing_stop_allocation_may_result_in_no_S_share_allocation=true
historical_fill_rate_100_percent_established=false
forward_operational_fill_measurement_required_before_full_capital_use=true
```

These are prefreeze evidence supplied by the GPT methodology authority, not
frozen methodology. They require V9 to avoid assuming a historical 100% fill
rate as an established fact.

## G. GPT-supplied point-in-time source research status

```text
source_class=GPT_WEB_VERIFIED_OFFICIAL_JPX
verification_date=2026-08-23
JPX_current_listed_issues_page=RECENT_MONTH_END_SNAPSHOTS
JPX_delisted_company_history=APPROXIMATELY_PAST_11_YEARS
partial_reconstruction_suggested=true
complete_V9_point_in_time_historical_security_master_demonstrated=false
POINT_IN_TIME_UNIVERSE_FEASIBILITY=PARTIAL_NOT_PROVEN
```

Survivorship bias is not solved.

## Reusable components table

| Reusable component | Classification |
|---|---|
| Shared-cash/pending-proceeds accounting, deterministic ledger, safety invariants | REUSE_WITH_V9_ADAPTER |
| Generic 1-share-capable quantity arithmetic | REUSE_WITH_V9_ADAPTER |
| Conservative gap-stop logic | REUSE_WITH_V9_ADAPTER |
| Raw/adjusted OHLCV separation, hashing, duplicate and invalid-row validation | REUSE_WITH_V9_ADAPTER |
| JPX calendar snapshot utility | REUSE_WITH_V9_ADAPTER |
| Pooled LightGBM/expanding-time/determinism scaffolding | REUSE_WITH_V9_ADAPTER |

## Replacement/gap table

| Replacement or gap | Classification |
|---|---|
| Close-auction/close-to-close fills, order timing, non-fill model, and operational mismatch measurement | REPLACE_FOR_V9 |
| Equal-notional 1-share allocation and concurrent-position choice | REPLACE_FOR_V9 |
| Frozen 100-share, D1-open/D10-open, and fixed 2/3-position semantics | REPLACE_FOR_V9 |
| Ridge baseline, train-only standardization, and cross-sectional target protocol | REPLACE_FOR_V9 |
| Point-in-time security master / survivorship treatment | REPLACE_FOR_V9 |
| TOPIX/proxy, equal-weight, random-K, and benchmark cash accounting | REPLACE_FOR_V9 |
| Purged/embargoed validation, if required | REPLACE_FOR_V9 |

## Methodology questions still requiring GPT decision

The V9 charter's thirteen prefreeze decisions remain unresolved. This audit
adds no choices. In particular: point-in-time universe construction; evidence
architecture; number of positions; auction/rebalance/holding definitions;
target/features including `volume_dryup`; Ridge/LightGBM parameters; cost and
fill assumptions; benchmark/random-K protocol; promotion and stopping rules;
and paper/small-capital forward gates all remain `CHATGPT_DECISION_REQUIRED`.

## Exact files inspected

- `AGENTS.md`, `PROJECT_STATE.md`, `PROJECT_DECISION_LOG.md`, `AI_RESEARCH_EXECUTION_RULES.md`, `AI_REAL_EXECUTION_RUNBOOK.md`, `V9_CROSS_SECTIONAL_CLOSE_AUCTION_CHARTER.md`, `V8_DATA_EXPOSURE_AUDIT.md`, `V8K_TERMINATION_RECORD.md`
- `src/trade_simulator.py`, `src/v4_meta_label_mvp.py`, `src/v4_meta_label_formal.py`, `src/v5_adaptive_portfolio.py`, `src/v5_a2_fixed100_stop_study.py`, `src/v5_b_candidate_ranker.py`
- `src/v6_a_confirmed_breakout.py`, `src/v6_a_r2_causal_breakout.py`, `src/v6_a_r2_formal.py`, `src/v7_capacity_engine.py`, `src/v7_yahoo_collector.py`, `src/v7_jpx_calendar.py`, `src/v7_seed_acquisition.py`
- `src/v8_historical_acquisition.py`, `src/benchmark.py`, `src/fetchers/base.py`, `src/fetchers/yfinance.py`, `config.yaml`, `requirement.txt`, `requirements-real-execution.txt`
- `tests/test_trade_simulator.py`, `tests/test_v4_meta_label_mvp.py`, `tests/test_v4_meta_label_formal.py`, `tests/test_v5_adaptive_portfolio.py`, `tests/test_v5_a2_fixed100_stop_study.py`, `tests/test_v5_b_candidate_ranker.py`
- `tests/test_v6_a_confirmed_breakout.py`, `tests/test_v6_a_r2_causal_breakout.py`, `tests/test_v6_a_r2_formal.py`, `tests/test_v7_capacity_engine.py`, `tests/test_v7_yahoo_collector.py`, `tests/test_v7_jpx_calendar.py`, `tests/test_v8_historical_acquisition.py`

```text
network_requests=0
models_fitted=0
backtests_run=0
profit_calculated=false
private_reads=0
gate_consumption=0
NEXT_ACTION=GPT_EXACT_SHA_V9_001_REVIEW
```
