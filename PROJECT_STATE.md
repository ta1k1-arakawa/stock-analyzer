# PROJECT_STATE

## 1. Document purpose

This file is the project-level external memory and current-state source of truth.
Experiment-specific design documents remain authoritative for their frozen specifications.
Code and tests remain authoritative for executable behavior.
Formal artifacts remain authoritative for measured results.

このファイルはプロジェクト全体の外部記憶であり、現在状態の正本です。実験固有の設計文書は凍結された仕様の正本、コードとテストは実行可能な挙動の正本、正式成果物は測定結果の正本です。

## 2. Current status

Date: 2026-08-06

Current research stage: V5-B candidate ranker formal exploratory evaluation completed.
V5-B AI ranking was NOT PROMISING.
The next preregistered experiment is V6-A confirmed breakout baseline.

```text
research_status=ACTIVE_EXPLORATORY
deployment_allowed=false
live_trading_allowed=false
real_order_code_allowed=false
paid_data_purchase_approved=false
automatic_strategy_tuning_allowed=false
human_gate_required=true
```

## 3. Repository lineage

| Milestone | Commit / tag | Role |
|---|---|---|
| evaluator-v2 final | `ada7977f30e5edf9c10887dceb2fd6ac3a0b00be` (`tag=evaluator-v2.0`) | Final evaluator-v2 implementation and frozen evaluation foundation. |
| V4 formal hash fix | `5fc4aa70abe41ed44be8ff8dfe765787014b4fe2` | Corrected V4 formal artifact/hash handling. |
| V5-A adaptive baseline final | `68e778c6ee2bb6385d13035931804402f1979f32` | Final V5-A adaptive portfolio baseline implementation. |
| V5-A2 fixed100 study final runner | `342352bc26096d1fe6d7b37fa948d72b6dc15bab` | Final V5-A2 fixed100 stop-study runner. |
| V5-B acquisition hardening | `b9fe89fa54bade207fcc2fdb034cd6f47641c629` | Hardened V5-B data acquisition and cache handling. |
| V5-B evaluation parser fix | `37b650bd1834112d05dac4db68ca1835d4ec8084` | Fixed V5-B evaluation parsing. |
| V5-B source-aware formal evaluator | `4d066510481e9b852514665e2865bdf59e33290c` | Frozen source-aware V5-B formal evaluator used for the final exploratory run. |

## 4. Frozen data and universe

### Universe

`V4_UNIVERSE.csv` is a fixed current-universe list of 300 Japanese stocks. `survivorship_bias=true`.

- ticker list SHA-256: `12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7`
- universe CSV SHA-256: `d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997`

### Training cache

- local path: `C:\taiki\hobbies\v4-meta-label-formal-cache`
- manifest SHA-256: `72AE3DB1186F2C9C113B1BAFE1D37FB74A5627AC7CEED1DFC2473A24E060DE85`
- successful payloads: 283

### Evaluation cache

- local path: `C:\taiki\hobbies\v5-b-evaluation-cache-retry1`
- manifest SHA-256: `797265BF671AF2245A342051FFAD02AA2929D67BA885945E7762149649148AA5`
- payload hash list SHA-256: `a45ce89a7fa8be689e7d0affe34de56152552d7a3414935f0a364843cd3121f8`
- successful payloads: 300
- failed payloads: 0
- raw payload bytes: 31,799,718
- date range: 2019-01-04 through 2026-01-30

### Source-aware boundary

For tickers present in the training cache, use training-cache prices through 2019-12-31 and evaluation-cache prices from 2020-01-01 onward. For evaluation-only tickers, use evaluation-cache prices from 2019-01-01 onward for feature warm-up. Pre-2020 model-training rows must come only from the training cache. 2026 price rows may only support exits from 2025 signals. 2026 signals are prohibited.

### Overlap historical revision

- overlap tickers: 283
- overlap rows: 67,843
- overlap dates: 2019-01-04 through 2019-12-30
- raw OHLCV mismatches: 0
- AdjClose mismatches: 482
- affected tickers: `4768`, `7609`
- classification: Yahoo historical AdjClose revision

## 5. Experiment ledger

### V4 pooled meta-label classifier

Verdict: `FREE_META_LABEL_PROTOTYPE_NOT_PROMISING`

| Arm | Trades | Net profit | Max drawdown | Profit factor |
|---|---:|---:|---:|---:|
| baseline | 403 | -28,515 yen | 39.22% | 0.973 |
| V4 | 56 | -4,605 yen | 9.16% | 0.967 |

- candidate-level AUC: 0.5083
- accepted positive rate: 39.29%
- abstained positive rate: 43.80%

The classifier reduced exposure and drawdown but did not identify winners. The V4 hypothesis was not supported.

Formal artifact: `C:\taiki\hobbies\v4-meta-label-formal-results.zip`. Artifact SHA is not recorded here because it is unknown.

### V5-A adaptive portfolio baseline

Verdict: `V5_ADAPTIVE_BASELINE_NOT_PROMISING`

- net_profit=-61,563.58 yen
- ending_equity_equivalent=338,436.42 yen
- trades=311
- win_rate=46.62%
- profit_factor=0.90555
- reported_max_drawdown=19.7789%
- conservative_reconstructed_MTM_DD≈20.96%
- positive_folds=0/3

Formal artifacts: `C:\taiki\hobbies\v5-adaptive-baseline-output`

| Artifact | SHA-256 |
|---|---|
| summary.json | `0FE3BD5DB4FFD7C3DC8387067903B7693983CEE2E518638D63B670568A15E191` |
| trades.csv | `00488F516C15E797FCFF7F751D5C3640C1D73EE38FD91D1EB49D64FD5C249018` |
| candidates.csv | `9264BE2CAEFAE2A3E993CA9000D429CAD6187B3DD3C45CC7E3B8C9672FDE007D` |
| daily_equity.csv | `99C1982F2D2A0AD9108C2ADA8EE0F3FE60AA2BA82CC64FA62F21D051C195DD11` |

### V5-A2 fixed100 stop study

Verdict: `V5_A2_NEITHER_EXPLORATORY_SUPPORTED`

Current-stop arm: net_profit=-119,718.35 yen; trades=322; profit_factor=0.851306; MTM_DD=29.0331%; positive_folds=0/3.

D5-only arm: net_profit=-535.91 yen; trades=297; profit_factor=0.999099; MTM_DD=25.8589%; positive_folds=2/3; 2017=+52,824.62 yen; 2018=+31,324.37 yen; 2019=-84,684.90 yen.

D5-only was much better as a full policy on this sample, but it was still unsupported. The comparison is policy-level because downstream fills, cash, and slot paths differed. It does not prove that the stop alone caused all losses.

Formal artifacts: `C:\taiki\hobbies\v5-a2-fixed100-stop-output`

| Artifact | SHA-256 |
|---|---|
| comparison.csv | `33A4B6801EAFE6950A95E6BA12A39757AE9FEA24D2B2166175C81E522D4069DC` |
| daily_equity.csv | `28B33A6146D74404BF3D17A8F83D9EA81AC3EAA57D9EBEDD5E168DDF68C2F3D6` |
| summary.json | `EE9F42533512BF0EFBB6592C09C4CB658C221480AE1ADDBF83BBE723A3E61A1E` |
| trades.csv | `8B3BCE3F13A1532769AA60AE0CDC1CBCE4F9565F2CE8D077D9E4C08D71B5B8E1` |

### V5-B candidate ranker

Verdict: `V5_B_CANDIDATE_RANKER_EXPLORATORY_NOT_PROMISING`

Formal execution commit: `4d066510481e9b852514665e2865bdf59e33290c`

#### Baseline rank

- net_profit=122,536.15709488306 yen
- ending_equity_equivalent=522,536.15709488303 yen
- filled_trades=569
- win_rate=49.3849%
- profit_factor=1.1138514271409448
- MTM_DD=26.782565969991488%
- positive_years=3/6

Yearly profit: 2020=-27,792.634676513204; 2021=-106,195.98642242365; 2022=-45,253.59194076466; 2023=+114,181.43414215161; 2024=+102,867.2727392584; 2025=+84,729.66325317451.

#### AI rank

- net_profit=110,665.55789764876 yen
- ending_equity_equivalent=510,665.5578976488 yen
- filled_trades=571
- win_rate=49.3870%
- profit_factor=1.1106809159699296
- MTM_DD=19.74845880237245%
- positive_years=2/6
- candidate_spearman=0.011915730755454533

Yearly profit: 2020=-40,027.005033263675; 2021=-38,537.184363554916; 2022=-27,566.832756041877; 2023=+100,255.14500412073; 2024=-1,973.253967894656; 2025=+118,514.68901428314.

AI ranking reduced MTM drawdown by about 7.03 percentage points, but reduced net profit by about 11,870.60 yen, slightly reduced profit factor, and produced only 2 positive years. The aggregate candidate Spearman correlation was only 0.0119. The current 20-feature LightGBM ranker did not demonstrate reliable winner-selection ability and must not be tuned further on the same data.

Formal artifacts: `C:\taiki\hobbies\v5-b-candidate-ranker-output`

| Artifact | SHA-256 |
|---|---|
| daily_equity.csv | `6C1FD62676CEEC4C6BB9B3450D5A1A80741AF28C981638C20DDE35461A2E78E8` |
| predictions.csv | `6C730F3AE129486EC8198733ECDAA2D4930F68E3CF96E973C618F650405F18E2` |
| summary.json | `2394928FCD711E4A8915DC0C7E87A57E6221696A9FA004BE845F7FFF421996FF` |
| trades.csv | `1DC70D9AD63697AA26D944C973398094785B7B72041EADD8584F13951743A920` |

### V6-A confirmed breakout baseline

- hypothesis family: broad-market-gated volatility-contraction breakout with volume confirmation
- exit: fixed D10 TIME exit
- AI: not used
- formal status: not run
- deployment: not allowed
- design status: `FROZEN`
- design branch: `v6-a-confirmed-breakout-baseline`
- design commit: `2e227787067805138c40e19f33a52cb03ef730fe`
- design_status=FROZEN
- implementation_status=BLOCKED_AFTER_RETRY
- implementation_commit=`ecd8a0f7f6341cf78e7d7bd8590c83ea934308e7`
- formal_run_started=false
- formal_result=NOT_RUN
- The original implementation commit `3cac45b036f34e8402ada9385cf07c606beac743` was not formal-ready because concentration metrics and safety counters were fixed zeros and aggregate drawdown crossed independent fold boundaries.
- The single allowed genuine implementation-bug retry was used before any formal evaluation.
- single_implementation_bug_retry_used=true
- additional_retry_allowed=false
- next_authorized_action=`HUMAN_DECISION_REQUIRED`

The accepted V6-A implementation was found before formal evaluation
to process D1 entry prices and cash/position state during the D0
signal-date loop. This introduced look-ahead and shifted portfolio
cash, slot, industry, and equity paths one trading day early.

Because the single preregistered implementation-bug retry had already
been used, V6-A was closed as engineering blocked without a formal run.
The V6-A scientific hypothesis was not tested or rejected.

## 6. Known limitations and artifact caveats

The V5-B scientific `NOT_PROMISING` conclusion is fixed and does not require rerunning.

Known reporting limitations:

1. `book_cost_dd` and `mark_to_market_dd` were calculated from the same MTM-equity formula.
2. The preregistered yearly Spearman gate was not included in the final gate object.
3. Some safety counters were emitted as fixed zeros rather than reconstructed from artifacts.

These limitations cannot reverse the result to PROMISING because the AI arm already failed profit versus baseline, PF versus baseline, four positive years, and four years beating baseline.

2020–2025 is exploratory and is not a pristine unused holdout. The universe is based on a current 300-stock list and has survivorship bias. Historical periods examined in earlier experiments must not be described as unseen. No result authorizes deployment or live trading.

## 7. Current accepted conclusions

### REJECTED / NOT SUPPORTED

- V4 meta-label abstention classifier
- V5-A adaptive quantity baseline
- V5-A2 current-stop arm
- V5-A2 D5-only arm as a deployable policy
- V5-B 20-feature LightGBM candidate ranker

### PRESERVED AS COMPARISON BASELINE ONLY

V5-B non-AI `BASELINE_RANK`: profit=122,536.16 yen; PF=1.11385; MTM_DD=26.78%.

It is not an adopted strategy and must not be described as deployable or operationally usable.

### NOT APPROVED

deployment; live trading; automatic orders; parameter tuning on the same evaluation data; paid data subscription.

## 8. Do-not-rerun registry

Do not rerun:

- V4 corrected formal evaluation
- V5-A one-shot formal evaluation
- V5-A2 one-shot exploratory evaluation
- V5-B production cache acquisition
- V5-B one-shot exploratory evaluation

Formal results and artifact hashes already exist. Rerunning after observing outcomes would weaken the preregistered process. Read-only diagnostics are allowed when they do not modify strategy behavior. Past executions that failed due to environment issues and did not reach scientific evaluation do not count as formal results.

## 9. Human gates

No explicit human approval means no:

- new paid data contract
- new strategy family
- formal production evaluation
- deployment
- broker integration
- order placement
- automatic iterative tuning
- changing frozen thresholds after seeing results
- deleting formal artifacts or caches

## 10. Next step

Current next research task: `V6-A confirmed breakout baseline`.

Purpose: Replace the shallow-pullback candidate family with a non-AI, broad-market-gated, volatility-contraction breakout strategy with volume confirmation and a fixed D10 exit.

```text
V6-A design status=FROZEN
design branch=v6-a-confirmed-breakout-baseline
design commit=2e227787067805138c40e19f33a52cb03ef730fe
design_status=FROZEN
implementation_status=BLOCKED_AFTER_RETRY
implementation_commit=ecd8a0f7f6341cf78e7d7bd8590c83ea934308e7
state_commit_before_block=7a6dc28b04a38d5bf561ac1d4879eb7637c8d576
formal_run_started=false
formal_result=NOT_RUN
scientific_hypothesis_tested=false
deployment_allowed=false
single_implementation_bug_retry_used=true
additional_retry_allowed=false
next_authorized_action=HUMAN_DECISION_REQUIRED
```

## 11. State update protocol

Update this file after each formal milestone:

- a design is frozen
- an implementation commit is accepted
- a cache is acquired
- a formal evaluation completes
- a strategy family is closed
- deployment status changes

Every update must include date, branch, commit SHA, verdict, artifact paths and hashes, what changed, what remains frozen, and the next authorized action.

Authority boundaries:

- Experiment design MD: authoritative for frozen scientific specification
- Source code and tests: authoritative for executable behavior
- Formal artifacts: authoritative for measured results
- PROJECT_STATE.md: authoritative for current project-level status and navigation

Any mismatch is resolved by the applicable authority above; PROJECT_STATE describes the current navigation and status without replacing experiment-specific specifications.

## Stage 1 verification record

- Values, hashes, and commits are recorded only from the supplied state; unknown artifact hashes are explicitly marked unknown.
- V5-B is recorded as `NOT_PROMISING` and comparison-only, never as PROMISING or deployable.
- 2020–2025 is explicitly not an unused holdout.
- Survivorship bias is explicit.
- The do-not-rerun registry is present.
- The next step is V6-A.
- `git diff --check`: to be run before commit.

## V6-A-R2 design start

```text
experiment=V6-A-R2
status=DESIGN_FROZEN_IMPLEMENTATION_NOT_STARTED
derived_from=V6-A
reason=causal D0/D1/D10 event-engine preregistration
design_branch=v6-a-r2-causal-breakout-baseline
design_commit=eae60d7a472c1365afb8f8da69db7878dbf3c6a0
human_design_review=PASS_AFTER_CLARIFICATION
design_clarification_parameter_change=false
formal_run_started=false
formal_result=NOT_RUN
scientific_hypothesis_tested=false
deployment_allowed=false
next_authorized_action=IMPLEMENTATION
```

Human design review found that “no alternative candidate” could
conflict with the frozen V6-A rule of processing later ranked
candidates after a skipped entry.

The design was clarified before implementation:
all D0 top-20 queued orders are processed on D1 in frozen rank order,
processing continues after skips, and no candidate outside the
frozen top 20 may be added.

The D0 equity invariant was also clarified to compare state immediately
before and after Phase 5 order queuing, not D0 equity against the
previous trading day.

## V6-A-R2 implementation retry record

```text
V6-A-R2 implementation_status=GATE2_REVIEW_PENDING_AFTER_RETRY
initial_engine_commit=88cc4344f31225fbcc23b54fac991156ec542dea
fixed_engine_commit=548288f9e16739fe0bff2d21996a7c53274f3e54
single_implementation_bug_retry_used=true
additional_implementation_retry_allowed=false
formal_run_started=false
formal_result=NOT_RUN
scientific_hypothesis_tested=false
next_authorized_action=INDEPENDENT_GATE2_CODE_REVIEW
```

Independent Gate 2 review found before any real-cache execution that
book equity omitted pending proceeds, the same-day proceeds counter did
not measure cross-order reuse, CLOSED rows skipped entry-date
invariants, and several synthetic acceptance tests were insufficient.

The single preregistered implementation-bug retry was used to correct
the causal engine and strengthen negative tests before real-cache
preflight.

## V6-A-R2 Gate 2 review pass

```text
V6-A-R2 implementation_status=GATE2_ACCEPTED
independent_gate2_code_review=PASS
accepted_engine_commit=548288f9e16739fe0bff2d21996a7c53274f3e54
single_implementation_bug_retry_used=true
additional_implementation_retry_allowed=false
formal_run_started=false
formal_result=NOT_RUN
scientific_hypothesis_tested=false
next_authorized_action=READ_ONLY_REAL_CACHE_PREFLIGHT
```

Independent Gate 2 review confirmed the corrected five-phase causal
engine, pending-proceeds accounting, future-read guard, measured safety
audits, CLOSED-row invariants, and strengthened synthetic negative tests.

Real-cache candidate preflight is authorized, but real-cache portfolio
simulation and formal evaluation remain prohibited.

## V6-A-R2 read-only real-cache preflight

```text
V6-A-R2 real_cache_preflight=PASS
candidate_parity=PASS
accepted_candidate_count=608
signal_day_count=346
yearly_candidate_counts=109,107,63,118,87,124
market_gate_pass_days=691
market_gate_blocked_days=774
accepted_candidate_key_sha256=4c550c8635a192fc4d60a753d8ac77ca9f992dc62bad3f36f19ef7512c29e818
preflight_diagnostic_correction=true
engine_code_changed=false
portfolio_simulation_started=false
formal_run_started=false
formal_result=NOT_RUN
next_authorized_action=INDEPENDENT_GATE3_STATIC_REVIEW
```
