# V8_DATA_EXPOSURE_AUDIT

## 0. Document status

```text
document=V8_DATA_EXPOSURE_AUDIT
audit_type=READ_ONLY_REPOSITORY_EVIDENCE_AUDIT
audit_date=2026-08-09
audit_base_branch=v7-forward-capacity-gate3-dry-run
audit_base_commit=fec1b85c2e6deb89b8c5d4fa31ff1ae58a62edbc
network_requests=0
backtests_run=0
data_acquired=0
models_fitted=0
profit_calculated=0
v7_artifacts_modified=0
```

This document records which historical data has already been consumed by the
V3–V7 development loops, and with what degree of outcome exposure. It changes
no prior verdict, reopens no closed experiment, and authorizes no data
acquisition.

本書の目的は「過去データの使用を禁止すること」ではない。目的は、V4〜V7で既に
結果を見たデータを、V8において未使用holdoutとして偽装しないことである。

## 1. Method and evidence boundary

### 1.1 What was inspected

Read-only inspection of the repository at the audit base commit:

- `PROJECT_STATE.md`, `PROJECT_RESEARCH_CONCLUSION.md`, `IMPLEMENTATION_STATUS.md`
- `V3_DESIGN.md`, `V3_FREE_PROTOTYPE_PLAN.md`, `V3_FREE_PROTOTYPE_RESULT.md`,
  `V3_PHASE0_DATA_SOURCE_AUDIT.md`
- `V4_META_LABEL_DESIGN.md`, `V4_META_LABEL_MVP_PROTOCOL.md`
- `V5_ADAPTIVE_BASELINE_DESIGN.md`, `V5_A2_FIXED100_STOP_STUDY.md`,
  `V5_B_CANDIDATE_RANKER_DESIGN.md`
- `V6_A_CONFIRMED_BREAKOUT_DESIGN.md`, `V6_A_R2_CAUSAL_BREAKOUT_DESIGN.md`
- `V7_FORWARD_CAPACITY_DESIGN.md`
- `config.yaml`, `V4_UNIVERSE.csv`, `V4_UNIVERSE_MANIFEST.json`,
  `free_prototype_manifest.json`, `data/benchmark/manifest.json`,
  `data/backtest_results/*`, `data/loop_validation_results/summary.json`,
  `data/v7_jpx_calendar_2026_2027.json`
- Source constants in `src/` (period literals, cache hashes, seed provenance)
- `git log` commit lineage

### 1.2 What could NOT be inspected

The following are **outside this repository** and were not read. Any statement
about them is derived from committed manifests/hashes and design text only:

| Asset | Declared location | Why unreadable |
|---|---|---|
| V4 training cache | `C:\taiki\hobbies\v4-meta-label-formal-cache` | Local Windows path, not in repo |
| V5-B evaluation cache | `C:\taiki\hobbies\v5-b-evaluation-cache-retry1` | Local Windows path, not in repo |
| V4/V5/V6 formal artifact bundles | `C:\taiki\hobbies\*-output`, `*.zip` | Local Windows paths, not in repo |
| V7 production seed CSV + seed acquisition manifest | Not committed | Private artifact; only its hashes appear in `src/v7_activation_manifest.py` |
| V7 durable study root | Not created in repo | Verified absent: no `activation_manifest*.json`, no `acquisitions/`, no `days/` |
| V3 free-prototype raw cache | `data/free_prototype_raw/`, `free-prototype-cache/` | Excluded by `.gitignore` |

**Consequence:** this audit can prove *that* a period was consumed and at what
exposure level, from committed period literals, manifests and recorded results.
It cannot independently re-derive the private cache contents. Anything not
provable from committed evidence is classified `UNKNOWN_EXPOSURE`, never as
fresh.

### 1.3 Classification definitions

| Class | Meaning |
|---|---|
| `EXPOSED_FOR_STRATEGY_DEVELOPMENT` | Outcomes/characteristics were observed **and** informed a subsequent design decision. Strongest contamination. |
| `EXPOSED_FOR_EVALUATION_ONLY` | Outcomes were computed and observed as an evaluation. Cannot be called a pure holdout again, even if it did not directly drive the next design. |
| `FEATURE_ONLY_NOT_OUTCOME_EXPOSED` | Rows were read for feature/indicator initialization only. No signal, no order, no trade, no profit was produced over that span. **Asserted only where a committed period literal or design invariant proves it.** |
| `UNKNOWN_EXPOSURE` | Cannot be proven either way from committed evidence. Treated as contaminated for planning purposes. |
| `POTENTIALLY_FRESH_HISTORICAL` | No evidence in this repository that the span/tickers entered any past decision loop. **A candidate for future holdout, not a certification of freshness.** |

Rule applied throughout: **absence of evidence of use is not evidence of
freshness.** `POTENTIALLY_FRESH_HISTORICAL` is a research lead requiring
confirmation, never a license to call something an unused holdout.

## 2. Data asset inventory

### 2.1 Universes

| Universe id | Definition | Count | Provenance |
|---|---|---:|---|
| `LEGACY_8` | Hand-fixed 8 codes: `1570`, `4188`, `4689`, `5020`, `7211`, `7267`, `8306`, `9432` | 8 | `data/benchmark/manifest.json`, snapshot `yahoo-jp-adjusted-2020-01-01-2026-05-20-788f37b6724d` |
| `FIXED_V4_300` | 300 codes in `V4_UNIVERSE.csv` | 300 | `V4_UNIVERSE_MANIFEST.json`, acquired `2026-08-03T06:53:30+00:00` from `www.jpx.co.jp` |
| `ELIGIBLE_CURRENT_ONLY` | JPX Prime/Standard domestic common stocks, 4-char code, current-only | 3115 | `V4_UNIVERSE_MANIFEST.json` field `eligible_current_only` |

`FIXED_V4_300` selection rule (verbatim from the manifest):

```text
sort by (SHA-256(UTF-8 code), code) ascending and select first 300
```

Key hashes:

```text
universe_csv_sha256 = d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997
ticker_list_sha256  = 12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
raw_file_sha256     = d99706334b3a9ca56b13805dac08d53ae1c2cd7df2ae77e7a9fad767cac51460
```

Two facts that matter for V8 holdout construction:

1. `V4_UNIVERSE_MANIFEST.json` records `matches_v3_ticker_list: true` and
   `v3_ticker_list_sha256` identical to `ticker_list_sha256`. **V3 and V4–V7
   share the same 300 tickers.** There is no separate V3 universe to reuse.
2. Only **one** of the 8 legacy codes (`4188`) is inside `FIXED_V4_300`. The
   other seven (`1570`, `4689`, `5020`, `7211`, `7267`, `8306`, `9432`) are
   exposed but outside the 300, so they must be excluded from any
   "not-in-V4-universe" holdout pool.
3. All universes are **current-constituent** lists captured in 2026-08.
   `survivorship_bias=true`, `point_in_time_universe=false`,
   `formal_point_in_time_universe=false`. Survivorship bias is a property of
   the universe, not of the period, and it is **not** removed by choosing a
   fresh period or fresh tickers.

### 2.2 Price caches and snapshots

| Cache / snapshot | Declared span | Tickers | Manifest hash | Committed? |
|---|---|---:|---|---|
| `LEGACY_8` benchmark snapshot | 2020-01-01 → 2026-05-20 | 8 | `788f37b6724d34f236977815ec6283af638e71fbafefa3dd9bc0f20d83764c2c` | yes (`data/benchmark/ohlcv/*.csv`) |
| V3 free-prototype cache | 2019-01-01 → 2025-03-31 | 274 of 300 acquired | `1caaec36328a822a7598a277fb14ab63826c6e4948e4a262743a93d4ed9d47fc` (data manifest), payload-hash-hash `cf0cfd575687ddf2cab5faa3c173dfa8442cc7c31454100f7c9c1e526f130126` | no (gitignored) |
| V4 training cache | 2015-01-01 → 2019-12-31 | 283 successful | `72AE3DB1186F2C9C113B1BAFE1D37FB74A5627AC7CEED1DFC2473A24E060DE85` | no |
| V5-B evaluation cache | 2019-01-04 → 2026-01-30 | 300 successful, 0 failed | `797265BF671AF2245A342051FFAD02AA2929D67BA885945E7762149649148AA5`, payload list `a45ce89a7fa8be689e7d0affe34de56152552d7a3414935f0a364843cd3121f8` | no |
| V7 production feature seed | last 252 valid trading days per ticker, cutoff `2026-08-07` | 300 × 252 = 75,600 rows | see 2.3 | no |

Source-aware boundary in force for V5-B / V6-A / V6-A-R2 (from
`PROJECT_STATE.md` §4 and `src/v5_b_candidate_ranker.py:230`): training-cache
prices through 2019-12-31, evaluation-cache prices from 2020-01-01 onward, for
tickers present in both. Overlap reconciliation recorded 283 overlap tickers /
67,843 rows over 2019-01-04 → 2019-12-30, 0 raw OHLCV mismatches, 482 AdjClose
mismatches on `4768` and `7609` (classified as a Yahoo historical AdjClose
revision).

### 2.3 V7 production seed provenance (real, already acquired)

`src/v7_activation_manifest.py:534-541` contains **non-placeholder** production
seed hashes, and `src/v7_gate4_preflight.py:46,50` pins the generating commit
and cutoff:

```text
seed_source_payload_manifest_sha256 = f71446043ad88e1688069ce1f438b11fa0e5172ca5ab21e96fe679ff1b74043f
seed_ticker_manifest_sha256         = edd06a02103f36b22552124d73f81f9826f609ea10a327d817ccd2c4281d0eff
seed_canonical_csv_sha256           = 8ac3adde3be58ea62072bb6fd7af242ba8c7c5701df1cc67ca2f3b411cde84d3
seed_ticker_count                   = 300
seed_row_count                      = 75600
seed_cutoff_trading_date            = 2026-08-07
SEED_GENERATION_COMMIT              = 0facf819c14e681036d2a081db0a5208c14b7cf9
```

`seed_row_count == 300 × 252` exactly, so every ticker contributed a full 252
valid observations. Per `src/v7_seed_acquisition.py:372-380`, selection is
`ticker_rows[-252:]` filtered to `trading_date <= 2026-08-07`.

**Inferred seed span:** approximately **2025-08 → 2026-08-07** (252 JPX trading
days ≈ 12 calendar months). The exact `first_seed_trading_date` is recorded
per ticker in the private seed manifest and is **not** committed, so the start
date is stated as approximate and is listed in the `UNKNOWN_EXPOSURE` register
(§7, U-5).

## 3. Per-experiment exposure records

### 3.1 LOOP-000 … LOOP-004 (evaluator-v2 lineage)

| Field | Value |
|---|---|
| version / experiment | `LOOP-000` … `LOOP-004`, evaluator-v2 |
| universe | `LEGACY_8` |
| ticker count | 8 |
| training period | per-ticker classifiers; labels confirmed through 2025-03-31 (raw-probability audit) |
| validation period | research window 2020-01-01 → 2025-03-31, 3 folds |
| evaluation / replay period | research 2020-01-01 → 2025-03-31; **reference replay 2025-04-01 → 2026-05-20** |
| candidate generation period | 2020-01-01 → 2025-03-31 (+ reference span) |
| portfolio simulation period | same; single-portfolio evaluator, `max_open_positions=1` |
| human saw results for | **2020-01-01 → 2026-05-20** (both research and reference) |
| results used for next design | **Yes** — drove LOOP-001…004 iteration, the `NOT_COMPARABLE` probability audit, and the decision to move to a pooled regression in V3 |
| data source | Yahoo Finance chart API, fixed snapshot |
| known hashes / manifests | snapshot `788f37b6724d…64c2c`; `config_hash 4ae7585c…1ef90`; summary `data/backtest_results/summary.json` |
| contamination | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` |

Additional evidence of heavy search on this window:
`data/backtest_results/summary.json` records `coordinate_evaluations: 1800`
across `coordinate_passes: 3`. That is 1,800 parameter evaluations on the
2020-2025 research window for 8 tickers.

**Reference-period caveat (important).** `free_prototype_manifest.json` records
`reference_period_used: False` and `PROJECT_RESEARCH_CONCLUSION.md` records
`reference replay・shadow結果の利用: 0件` — i.e. V3 *model selection* did not
consume the reference period. However, the reference replay **was executed and
its outcomes are committed in this repository**:

- `data/backtest_results/reference_trades.csv` — 140 filled rows with a
  `profit` column, spanning 2025-04-02 → 2026-05-19
- `data/backtest_results/reference_predictions.csv`,
  `reference_skipped_orders.csv`
- `data/backtest_results/summary.json`: `profit: -110765.79826181`,
  `trades: 140`, `max_drawdown_percent: -41.26732857`, per-stock profit split

Therefore 2025-04-01 → 2026-05-20 on `LEGACY_8` is **at minimum**
`EXPOSED_FOR_EVALUATION_ONLY`. It is readable by anyone with the repository and
must never be presented as an unopened holdout for those 8 codes.

### 3.2 V3 free prototype

| Field | Value |
|---|---|
| version / experiment | V3 free prototype (pooled LightGBM regression) |
| universe | `FIXED_V4_300` (identical ticker-list hash) |
| ticker count | 300 selected; 274 price-acquired (91.33%); 255 full coverage (85.0%) |
| training period | fold 1 `2020-01-01 → 2020-12-31`; fold 2 `2020-01-01 → 2022-03-31`; fold 3 `2020-01-01 → 2023-09-30` |
| validation period | fold 1 `2021-01-01 → 2022-03-31`; fold 2 `2022-04-01 → 2023-09-30`; fold 3 `2023-10-01 → 2025-03-31` |
| evaluation / replay period | actual evaluated span `2021-01-04 → 2025-03-27`, 90,859 rows |
| candidate generation period | 2021-01-04 → 2025-03-27 |
| portfolio simulation period | same; 439 closed trades |
| human saw results for | 2019-01-01 → 2025-03-31 (data span); outcome metrics for 2021-01-04 → 2025-03-27 |
| results used for next design | **Yes** — the sub-0.02 Spearman and −51,797.57 yen result produced `DO_NOT_PURCHASE` and directly motivated V4 meta-labelling |
| data source | Yahoo Finance chart API |
| known hashes / manifests | data manifest `1caaec36…d47fc`; universe source `d9970633…51460`; codes hash `12777a83…ce4f7` |
| contamination | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` |

Data span 2019-01-01 → 2019-12-31 was warm-up/training feed only
(`evaluation_from: 2020-01-01`); 2025-04-01 onward recorded as `0` uses.

### 3.3 V4 pooled meta-label classifier

| Field | Value |
|---|---|
| version / experiment | V4 meta-label abstention classifier |
| universe | `FIXED_V4_300` |
| ticker count | 300 declared; 283 price-success in the training cache |
| training period | fold 1 `2016-04-01 → 2016-12-31`; fold 2 `2016-04-01 → 2017-12-31`; fold 3 `2016-04-01 → 2018-12-31` (`src/v4_meta_label_mvp.py:45-47`) |
| validation period | none separate — test folds served as the evaluation |
| evaluation / replay period | `2017-01-01 → 2019-12-31` (OOF evaluation over 2017–2019) |
| candidate generation period | `2016-04-01 → 2019-12-31` (`SIGNAL_FROM`/`SIGNAL_TO`, `src/v4_meta_label_mvp.py:23-24`) |
| portfolio simulation period | 2017–2019, three independent folds from ¥300,000 |
| human saw results for | 2017-01-01 → 2019-12-31 (baseline 403 trades / −28,515 yen; V4 56 trades / −4,605 yen; AUC 0.5083) |
| results used for next design | **Yes** — the null result produced V5-A |
| data source | V4 training cache (Yahoo), prices `2015-01-01 → 2019-12-31` (`src/v4_meta_label_formal.py:171`) |
| known hashes / manifests | training cache manifest `72AE3DB1…60DE85`; universe hashes as §2.1 |
| contamination | 2017–2019 `EXPOSED_FOR_STRATEGY_DEVELOPMENT`; 2015-01-01 → 2016-03-31 `FEATURE_ONLY_NOT_OUTCOME_EXPOSED` |

**Feature-only proof for 2015-01-01 → 2016-03-31.** `src/v4_meta_label_mvp.py`
declares `PRICE_FROM = 2015-01-01`, `PRICE_TO = 2019-12-31`,
`SIGNAL_FROM = 2016-04-01`, `SIGNAL_TO = 2019-12-31`. No signal can be emitted
before `SIGNAL_FROM`, so no candidate, order, trade or profit exists on
2015-01-01 → 2016-03-31. The design text confirms the intent: 「2015年のデータは、
252営業日の履歴条件と60日特徴量を2016年から計算するためのwarm-upだけに使用する」.
This is the strongest feature-only claim in the audit — it rests on a committed
period literal, not on narrative.

### 3.4 V5-A adaptive portfolio baseline

| Field | Value |
|---|---|
| version / experiment | V5-A adaptive quantity baseline |
| universe | `FIXED_V4_300` (unchanged hashes) |
| ticker count | 300 declared; 283 anticipated usable |
| training period | none (no model fitted) |
| validation period | none |
| evaluation / replay period | folds 2017, 2018, 2019 |
| candidate generation period | `2016-04-01 → 2019-12-31` |
| portfolio simulation period | 2017 / 2018 / 2019, independent, ¥400,000 each |
| human saw results for | 2017–2019: net −61,563.58 yen, 311 trades, win 46.62%, PF 0.90555, DD 19.78% (reconstructed ≈20.96%), positive folds 0/3 |
| results used for next design | **Yes** — produced V5-A2 |
| data source | V4 training cache; prices 2015-01-01 → 2019-12-31 (`src/v5_adaptive_portfolio.py:165`) |
| known hashes / manifests | summary `0FE3BD5D…15E191`, trades `00488F51…5C249018`, candidates `9264BE2C…FDE007D`, daily_equity `99C1982F…C195DD11` |
| contamination | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` |

The V5-A design states this explicitly: 「The 2017–2019 folds have already been
viewed and are not an unused final evaluation period.」

### 3.5 V5-A2 fixed100 stop study

| Field | Value |
|---|---|
| version / experiment | V5-A2 FIXED100 stop mechanism comparison |
| universe | `FIXED_V4_300` |
| ticker count | 300 declared / 283 usable |
| training period | none |
| validation period | none |
| evaluation / replay period | folds 2017, 2018, 2019 (re-opened, already-viewed) |
| candidate generation period | `2016-04-01 → 2019-12-31` |
| portfolio simulation period | 2017 / 2018 / 2019, two arms |
| human saw results for | 2017–2019 **at yearly granularity**: current-stop arm net −119,718.35 / 322 trades / PF 0.851306 / DD 29.03% / 0-of-3 folds; D5-only arm net −535.91 / 297 trades / PF 0.999099 / DD 25.86% / 2-of-3 folds, with 2017 = +52,824.62, 2018 = +31,324.37, 2019 = −84,684.90 |
| results used for next design | **Yes** — the D5-only exit was carried into V5-B as `FIXED100_D5_ONLY` |
| data source | V4 training cache |
| known hashes / manifests | comparison `33A4B680…4D069DC`, daily_equity `28B33A61…DDF68C2F3D6`, summary `EE9F4253…E3A61A1E`, trades `8B3BCE3F…8D71B5B8E1` |
| contamination | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` |

Self-declared in the design: `exploratory_only=true`, `unused_holdout=false`,
「already-viewed-period mechanistic comparison … not a final holdout evaluation」.
Yearly 2017/2018/2019 profits were observed, so 2017–2019 is exposed at
**year granularity**, not merely in aggregate.

### 3.6 V5-B candidate ranker

| Field | Value |
|---|---|
| version / experiment | V5-B 20-feature pooled LightGBM candidate ranker |
| universe | `FIXED_V4_300` |
| ticker count | 300 (evaluation cache: 300 successful, 0 failed) |
| training period | expanding: for evaluation year Y, labels with `exit_date < Y-01-01`, ≥1,000 rows required; training dataset built from `2016-04-01 → 2019-12-31` on the training cache plus prior evaluation years (`src/v5_b_candidate_ranker.py:230-233`) |
| validation period | none separate — the 2020–2025 span served as the exploratory evaluation |
| evaluation / replay period | `2020-01-01 → 2025-12-31` |
| candidate generation period | `2016-04-01 → 2025-12-31` (`generate_candidates` defaults, `src/v5_b_candidate_ranker.py:149`) |
| portfolio simulation period | 2020–2025, yearly independent folds |
| human saw results for | **2020–2025 at yearly granularity, for both arms.** Baseline: net 122,536.157…, 569 fills, PF 1.11385, DD 26.78%, 3/6 positive years, yearly 2020 −27,792.63 / 2021 −106,195.99 / 2022 −45,253.59 / 2023 +114,181.43 / 2024 +102,867.27 / 2025 +84,729.66. AI: net 110,665.558…, 571 fills, PF 1.11068, DD 19.75%, 2/6 positive years, Spearman 0.0119 |
| results used for next design | **Yes** — became the fixed comparison baseline embedded in the V6-A / V6-A-R2 acceptance gates |
| data source | V4 training cache (≤2019-12-31) + V5-B evaluation cache (≥2020-01-01) |
| known hashes / manifests | evaluation cache `797265BF…9148AA5`; daily_equity `6C1FD626…5461A2E78E8`, predictions `6C730F3A…50405F18E2`, summary `23949287…F421996FF`, trades `1DC70D9A…13951743A920`; formal commit `4d066510481e9b852514665e2865bdf59e33290c` |
| contamination | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` |

Self-declared: 「Years 2020–2025 are exploratory, not holdout」.

### 3.7 V6-A confirmed breakout baseline (closed without a formal run)

| Field | Value |
|---|---|
| version / experiment | V6-A confirmed breakout baseline |
| universe | `FIXED_V4_300` |
| ticker count | 300 |
| training period | none (`ai_used=false`) |
| validation period | none |
| evaluation / replay period | **planned** 2020-01-01 → 2025-12-31; `formal_run_started=false`, `formal_result=NOT_RUN` |
| candidate generation period | none executed |
| portfolio simulation period | none executed |
| human saw results for | **no V6-A outcome was produced.** The V5-B comparison values were embedded in the design, so the human saw *V5-B's* 2020–2025 numbers while writing V6-A |
| results used for next design | Indirectly — the discovered D1/D0 look-ahead defect produced V6-A-R2 |
| data source | declared caches; no formal read reached evaluation |
| known hashes / manifests | design commit `2e227787067805138c40e19f33a52cb03ef730fe`; implementation `ecd8a0f7f6341cf78e7d7bd8590c83ea934308e7` |
| contamination | Adds **no new** exposure. The planned window was already exposed by V5-B. |

Recorded state: `single_implementation_bug_retry_used=true`,
`additional_retry_allowed=false`, 「The V6-A scientific hypothesis was not tested
or rejected.」

### 3.8 V6-A-R2 causal breakout baseline (one-shot formal run completed)

| Field | Value |
|---|---|
| version / experiment | V6-A-R2 causal five-phase breakout engine |
| universe | `FIXED_V4_300` |
| ticker count | 300 |
| training period | none (`ai_used=false`) |
| validation period | none |
| evaluation / replay period | `2020-01-01 → 2025-12-31`; January 2026 prices used **only** for D10 exits of 2025 year-end signals; 2026 signals fail-closed (gate 19: `2026 signal count = 0`) |
| candidate generation period | 2020-01-01 → 2025-12-31; 608 accepted candidates over 346 signal days; yearly candidate counts 109/107/63/118/87/124; market gate pass 691 / blocked 774 days |
| portfolio simulation period | 2020–2025 yearly independent folds, ¥400,000 each; 1,457 daily-equity rows; 132 closed trades of 608 rows; 204,769 candidate-audit rows |
| human saw results for | **2020–2025 at yearly granularity, plus full metric surface.** Net 93,503.80, PF 1.34227, MTM DD 14.9477%, book DD 12.6095%, win 53.79%, monthly win 56.86%, top-5 share 39.23%, max industry share 16.03%; yearly 2020 +29,435.29 / 2021 −47,277.44 / 2022 −10,602.28 / 2023 +12,595.34 / 2024 +49,483.71 / 2025 +59,869.20; skip histogram; 18/20 gates passed |
| results used for next design | **Yes** — the `NOT_PROMISING` verdict and the CONTROL semantics were carried directly into V7 (`derived_from=V6-A-R2`) |
| data source | training cache ≤2019-12-31 + evaluation cache ≥2020-01-01 |
| known hashes / manifests | summary `58EE0C43…EDE24C4A`, trades `A2D84ABB…9CACD25D68`, candidates `B66FA8A5…6006C2F76AB6`, daily_equity `5F6F4A2A…1BEA4EB3545A30`; candidate key `4c550c8635a192fc4d60a753d8ac77ca9f992dc62bad3f36f19ef7512c29e818`; repo commit `4be04b96e1bf2dea702b93bc493172836602a6bf` |
| contamination | 2020–2025 `EXPOSED_FOR_STRATEGY_DEVELOPMENT`; 2026-01-01 → 2026-01-30 `EXPOSED_FOR_EVALUATION_ONLY` (exit prices feeding observed 2025 profits) |

Note the two-stage exposure: a **read-only preflight** first revealed candidate
counts, signal-day counts and market-gate day counts for 2020–2025 *before* any
portfolio simulation; the one-shot formal run then revealed profit. Candidate
structure on 2020–2025 was therefore exposed even before profit was.

### 3.9 V7 forward capacity study

| Field | Value |
|---|---|
| version / experiment | `V7_FORWARD_CAPACITY` (CONTROL max_open_positions=2 vs CAPACITY_3 max_open_positions=3) |
| universe | `FIXED_V4_300`, `universe_changes_during_study=0` |
| ticker count | 300 |
| training period | none (`ai_used=false`) |
| validation period | none |
| evaluation / replay period | **none historical.** `historical_backtest_allowed=false`, `historical_replay_allowed=false`, `historical_candidate_generation_allowed=false`, `historical_portfolio_replay_allowed=false`, `historical_profit_calculation_allowed=false` |
| candidate generation period | forward only, from the activation boundary onward; not activated in-repo |
| portfolio simulation period | forward only; no durable study root exists in this repository |
| human saw results for | **no V7 forward outcome exists yet.** All executed CLIs are synthetic-only (`network_requests=0`, `actual_activation_created=false`) |
| results used for next design | Not applicable — and **prohibited** from informing V8 (see §8) |
| data source | V7 production feature seed (252 valid trading days/ticker, cutoff 2026-08-07) + forward daily acquisition after activation |
| known hashes / manifests | seed hashes in §2.3; design `e3e1367efd913b601a70328a815d88c20af6d147`; collector `4ca41c53895e75910ae65809fea6018868929afa`; calendar `03ce048b0eedca632f79ad925a627cb9e967d78d` |
| contamination | seed window: see below — **split classification** |

**V7 seed classification is not uniform.** The seed span (≈2025-08 → 2026-08-07)
straddles previously-exposed and previously-unexposed data:

| Sub-span | Classification | Reason |
|---|---|---|
| ≈2025-08 → 2025-12-31 | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` | Inside V5-B and V6-A-R2 signal windows; 2025 yearly profit observed in both |
| 2026-01-01 → 2026-01-30 | `EXPOSED_FOR_EVALUATION_ONLY` | Inside the V5-B evaluation cache; used as D10 exit prices producing observed 2025 profit |
| 2026-01-31 → 2026-08-07 | `FEATURE_ONLY_NOT_OUTCOME_EXPOSED` for the 300 universe | Beyond the evaluation cache end (2026-01-30); read only as V7 feature seed; no candidate/order/trade/profit permitted over the seed period |

Exception inside the last row: ticker `4188` is in **both** `LEGACY_8` and
`FIXED_V4_300`, and the legacy reference replay produced per-trade profit
through 2026-05-19. For `4188`, 2026-01-31 → 2026-05-20 is
`EXPOSED_FOR_EVALUATION_ONLY`, not feature-only.

**Feature-only proof strength for the seed.** The design fixes
`historical_feature_seed_role=FEATURE_INITIALIZATION_ONLY` and states 「profit/loss,
and evaluation must not be run over the seed period」; the forward protocol
fail-closes on any signal before the activation boundary. No study root,
candidate file, trade file or equity file exists in the repository. This is a
**strong but not absolute** proof: it rests on design invariants plus the
absence of artifacts, and the private seed CSV itself was not read. Recorded as
feature-only with that caveat.

## 4. Consolidated exposure matrix

Rows are calendar spans; columns are the universes. `—` means no data was ever
acquired for that combination in any recorded experiment.

| Span | `LEGACY_8` (8) | `FIXED_V4_300` (300) | `ELIGIBLE_CURRENT_ONLY` minus the above (~2,808) |
|---|---|---|---|
| before 2015-01-01 | — | — | — |
| 2015-01-01 → 2016-03-31 | — | `FEATURE_ONLY_NOT_OUTCOME_EXPOSED` (V4/V5 warm-up) | — |
| 2016-04-01 → 2016-12-31 | — | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (V4/V5-B train) | — |
| 2017-01-01 → 2019-12-31 | — | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (V4, V5-A, V5-A2 folds, yearly granularity) | — |
| 2019-01-01 → 2019-12-31 | — | also V3 warm-up / eval-cache overlap reconciliation | — |
| 2020-01-01 → 2021-12-31 | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (LOOP research, 1,800 evals) | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (V3 folds, V5-B, V6-A-R2) | — |
| 2022-01-01 → 2025-03-31 | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (V3 through 2025-03-27, V5-B, V6-A-R2) | — |
| 2025-04-01 → 2025-12-31 | `EXPOSED_FOR_EVALUATION_ONLY` (reference replay, profits committed) | `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (V5-B, V6-A-R2 2025 fold) | — |
| 2026-01-01 → 2026-01-30 | `EXPOSED_FOR_EVALUATION_ONLY` | `EXPOSED_FOR_EVALUATION_ONLY` (D10 exits of 2025 signals) | — |
| 2026-01-31 → 2026-05-20 | `EXPOSED_FOR_EVALUATION_ONLY` | `FEATURE_ONLY_NOT_OUTCOME_EXPOSED` (V7 seed) — except `4188` | — |
| 2026-05-21 → 2026-08-07 | — | `FEATURE_ONLY_NOT_OUTCOME_EXPOSED` (V7 seed) | — |
| 2026-08-08 onward | — | **V7 forward territory** — reserved, see §8 | — |

## 5. Search-intensity record (why "exposed" understates the risk)

Contamination is not only about whether a period was seen; it is about how many
times a decision was conditioned on it.

| Loop | Recorded search intensity on already-seen data |
|---|---|
| evaluator-v2 / LOOP | `coordinate_evaluations: 1800`, `coordinate_passes: 3` on 2020-01-01 → 2025-03-31 (8 tickers) |
| LOOP-000 → LOOP-004 | 5 preregistered strategy iterations, each conditioned on the previous result |
| V3 | 1 preregistered run; result drove the paid-data decision |
| V4 → V5-A → V5-A2 → V5-B | 4 sequential designs on 2017–2019 (V4/V5-A/V5-A2) then 2020–2025 (V5-B), each informed by the previous verdict |
| V6-A → V6-A-R2 | 2 designs against a **fixed V5-B comparison target** — acceptance gates literally encode V5-B's 2020–2025 numbers |
| V7 | forward-only; no historical search |

The V6-A/V6-A-R2 acceptance gates are the clearest contamination marker in the
repository: gates 7–10 compare against V5-B's observed 2020–2025 net profit,
profit factor, drawdown and per-year profit. A strategy tuned to beat those
specific numbers on those specific years is fitted to that sample by
construction, regardless of how causal the engine is.

## 6. POTENTIALLY_FRESH_HISTORICAL candidates

These are **candidates**, not certified holdouts. None may be described as
unused until the confirmation steps in §6.4 are completed.

### 6.1 Candidate F-1 — cross-sectional: eligible tickers never acquired

```text
candidate_id=F-1
dimension=CROSS_SECTIONAL
pool=ELIGIBLE_CURRENT_ONLY minus FIXED_V4_300 minus (LEGACY_8 outside the 300)
approximate_size=3115 - 300 - (0..7) = 2808..2815
evidence_of_prior_use=NONE_FOUND
strength=STRONGEST_AVAILABLE
```

The size is a range, not a number: whether the seven exposed legacy codes are
themselves members of the 3,115 eligible pool is **unverified**, because the raw
JPX listing behind `raw_file_sha256 = d9970633…51460` is not committed. The
conservative figure (2,808) assumes all seven are inside the pool and must be
removed. Resolving this requires re-deriving the eligible list from the raw
source, which is an acquisition step and is out of scope for this audit.

No price data for these codes was acquired by V3, V4, V5-A, V5-A2, V5-B, V6-A,
V6-A-R2 or V7. Every recorded cache is scoped to `FIXED_V4_300` or `LEGACY_8`.
Selection is reproducible: the same
`sort by (SHA-256(code), code)` rule can deterministically carve disjoint
blocks beyond the first 300.

**Limits that must be stated whenever F-1 is used:**

1. The pool is still a **current-only 2026-08-03 list** — `survivorship_bias=true`
   applies exactly as it does to the 300. Fresh tickers do not fix survivorship.
2. The **regimes** of 2020–2025 are known to the researcher in detail (per-year
   profitability, market-gate pass/block day counts, breadth behaviour). A new
   cross-section over an old, well-characterised period is **not** equivalent to
   an unseen period. Cross-sectional freshness reduces, but does not eliminate,
   contamination.
3. Liquidity and price-level distribution differ across the eligible pool; the
   300 were not chosen for tradeability, but neither were the other 2,808.

### 6.2 Candidate F-2 — temporal: pre-2015 history

```text
candidate_id=F-2
dimension=TEMPORAL
span=before 2015-01-01
evidence_of_prior_use=NONE_FOUND
strength=MODERATE
```

The earliest acquisition in any recorded manifest is `price_from = 2015-01-01`
(V4 training cache). Nothing earlier has been fetched for any universe.

**Limits:** survivorship bias grows monotonically the further back a
current-only universe is projected; corporate actions, code reassignments and
delistings before 2015 are unverified; the 2015 warm-up requirement (252
trading days + 60-day features) means usable signals would start ~1 year after
whatever acquisition start is chosen.

### 6.3 Candidate F-3 — temporal: 2026-01-31 → 2026-08-07 for the 300

```text
candidate_id=F-3
dimension=TEMPORAL
span=2026-01-31 → 2026-08-07
universe=FIXED_V4_300 minus ticker 4188
evidence_of_prior_use=V7_FEATURE_SEED_ONLY
strength=WEAK
```

Beyond the V5-B evaluation cache end (2026-01-30) and before the V7 seed cutoff.
Outcome-unexposed, but:

**Limits:** ~6 calendar months is far too short for a walk-forward evaluation;
it is *inside* the V7 feature seed, so V7 and V8 would share feature-state
lineage; and it sits immediately adjacent to V7's forward window, so results
there correlate strongly with V7's own forward period. **Not recommended as a
sealed holdout.** Recorded for completeness only.

### 6.4 Confirmation required before any candidate is treated as a holdout

For each candidate, before it is used:

1. Re-run this audit at the then-current HEAD and confirm no new experiment has
   consumed it.
2. Record the acquisition manifest hash and the exact span/ticker list at
   acquisition time.
3. Register it in the V8 partition manifest **before** any feature is computed
   over it.
4. State the survivorship-bias and regime-knowledge caveats in the same document
   that reports any result derived from it.

## 7. UNKNOWN_EXPOSURE register

Items that cannot be resolved from committed evidence. All are treated as
contaminated for planning purposes.

| id | Item | Why unknown | Planning treatment |
|---|---|---|---|
| U-1 | Exact row-level contents of the V4 training cache and V5-B evaluation cache | Local Windows paths, not committed | Trust declared spans; treat both spans as fully exposed |
| U-2 | Whether any ad-hoc/exploratory chart, notebook or manual inspection touched spans beyond those recorded | No log of informal inspection exists | Assume any span inside an acquired cache may have been inspected |
| U-3 | Whether V6-A's blocked implementation produced any transient partial output before being closed | `formal_run_started=false` is recorded, but no negative artifact proof | Adds no new span beyond V5-B's; treat as no additional exposure |
| U-4 | The `4768` / `7609` AdjClose revision (482 mismatched rows, 2019) | Yahoo revised history after acquisition; current upstream values differ from cached values | Any re-acquisition of 2019 will not byte-match the caches; do not treat re-fetched 2019 as identical data |
| U-5 | Exact `first_seed_trading_date` of the V7 seed | Recorded only in the private per-ticker seed manifest | Seed start stated as ≈2025-08; treat the whole ≈2025-08 → 2026-08-07 span conservatively per §3.9 |
| U-6 | Whether the 26 tickers that failed V3 price acquisition (300 − 274) differ systematically | Failure list not committed | Do not assume those tickers are less exposed |
| U-7 | Whether the 17 tickers absent from the V4 training cache (300 − 283) were later covered | Evaluation cache reports 300 successful, but per-ticker overlap detail is private | Treat all 300 as exposed for 2020–2025 |

## 8. V7 isolation constraints inherited by V8

These follow from the V7 design and the user's V8 instruction. They are
recorded here because they are *data-exposure* constraints, not merely process
constraints.

```text
v7_code_modification_allowed=false
v7_artifact_modification_allowed=false
v7_forward_observation_use_in_v8_tuning=false
v7_interim_result_use_in_v8_parameter_selection=false
v7_activation_manifest_required_by_v8=false
v7_durable_study_root_access_by_v8=false
v8_may_read_v7_forward_outcomes=false
```

Additional consequence of the seed overlap (§3.9): V8 and V7 share the
`FIXED_V4_300` universe and would share feature lineage over ≈2025-08 →
2026-08-07 if V8 uses that span. V8 must either avoid that span or declare the
shared lineage explicitly; it must never treat it as independent evidence
corroborating V7.

**Forward-period reservation.** 2026-08-08 onward is V7's forward observation
territory. If V8 later acquires that span as "historical" data, any V8 result
over it is statistically entangled with V7's live study and must not be
presented as independent confirmation.

## 9. Findings summary

1. **Every calendar span for which this project has ever acquired price data
   for the 300-ticker universe, from 2016-04-01 through 2026-01-30, is
   outcome-exposed.** The only in-universe exceptions are the 2015-01-01 →
   2016-03-31 warm-up (feature-only, code-proven) and 2026-01-31 → 2026-08-07
   (V7 feature seed).
2. **2020–2025 is the most heavily contaminated span**, consumed by V3, V5-B and
   V6-A-R2, with V6-A/V6-A-R2 acceptance gates literally encoding V5-B's
   observed per-year numbers.
3. **2017–2019 is contaminated at year granularity** by V4, V5-A and V5-A2.
4. **The committed `data/backtest_results/reference_*.csv` files mean the
   2025-04-01 → 2026-05-20 reference span is outcome-exposed for the 8 legacy
   tickers**, notwithstanding that V3 model selection did not consume it.
5. **The strongest genuinely-unused axis is cross-sectional, not temporal**:
   ~2,808 eligible JPX Prime/Standard domestic codes have never been acquired.
6. **No temporal span both (a) has enough length for walk-forward evaluation and
   (b) is free of prior outcome exposure**, unless new data is acquired
   (pre-2015, or future post-2026-08 data which collides with V7).
7. Therefore V8's partition scheme **cannot** be built on time alone. It must
   combine a fresh cross-section with a time split, and must state its residual
   contamination honestly rather than claim a clean holdout.

## 10. Audit assertions

```text
audit_complete=true
versions_audited=LOOP/evaluator-v2,V3,V4,V5-A,V5-A2,V5-B,V6-A,V6-A-R2,V7
network_requests=0
backtests_run=0
profit_calculated=0
models_fitted=0
data_acquired=0
v7_code_modified=false
v7_artifacts_modified=false
v7_activation_requested=false
private_caches_read=false
unknown_items_registered=7
potentially_fresh_candidates=3
fresh_certified=0
```

`fresh_certified=0` is intentional. This audit certifies nothing as fresh; it
only identifies candidates and the confirmation each would require.

---

## Appendix A — Human design review note (append-only, 2026-08-09)

```text
note_type=APPEND_ONLY_REVIEW_NOTE
note_date=2026-08-09
audit_findings_modified=false
audit_classifications_modified=false
audit_tables_modified=false
```

**No audit fact, classification, table, hash or `UNKNOWN_EXPOSURE` entry above
has been changed by this note.** Sections 0–10 stand exactly as originally
recorded. This appendix records only how the human design review of
2026-08-09 disposed of the candidates this audit identified, so that a later
reader can see which findings were acted on and which were deferred.

### A.1 Disposition of the `POTENTIALLY_FRESH_HISTORICAL` candidates

| Candidate | Audit §  | Disposition in frozen V8 design |
|---|---|---|
| F-1 — cross-sectional, ~2,808–2,815 never-acquired eligible codes | §6.1 | **Adopted** as the basis of the fresh layers. Cut into `T1` (validation, 300), `T2` (sealed holdout, 300), `T3` (reserve, 300, unused in initial V8), `T_spare` (remainder) |
| F-2 — pre-2015 history | §6.2 | **Deferred.** Not acquired for initial V8. May later be acquired for Layer A regime expansion only, under a separate human gate |
| F-3 — 2026-01-31 → 2026-08-07 | §6.3 | **Not used**, consistent with the audit's own "not recommended as a sealed holdout" |

`fresh_certified` remains **0**. Adopting F-1 as the basis for `T1`/`T2` is a
design allocation, not a freshness certification; §6.4's confirmation steps
still apply in full before any block is treated as a holdout.

### A.2 Audit findings promoted into binding design constraints

Three findings became hard constraints rather than advisory notes:

1. **§1.2 / §2.1 — the complete eligible-3,115 ticker list is not committed.**
   The frozen design makes this a BLOCK condition: if exact source-list
   reproducibility cannot be demonstrated when the partition manifest is built,
   the work stops and no block assignment is written.
2. **§2.1 note 2 — only `4188` of the eight legacy codes is inside `T0`.** The
   other seven are excluded from `T1`, `T2`, `T3` and `T_spare` by an explicit
   exclude list recorded in the partition manifest.
3. **§6.1 limit 1 and §9 finding 5 — fresh tickers do not remove survivorship
   bias.** The frozen design constrains the permitted meaning of a sealed-holdout
   PASS and enumerates prohibited claim phrasings.

### A.3 Spans confirmed unused by initial V8

Initial V8 uses `P_hist = 2016-04-01 → 2025-12-31` only. Consequently these
audited spans are **not** consumed by initial V8:

```text
pre_2015_used_by_initial_v8=false
2026-01-31..2026-08-07_used_by_initial_v8=false
v7_forward_window_2026-08-08_onward_used_by_initial_v8=false
v7_feature_seed_span_used_by_initial_v8=false
```

The V7 feature-seed overlap identified in audit §3.9 and §8 therefore does not
arise in initial V8. Should a future V8 phase use any of these spans, the
shared-lineage disclosure requirement of audit §8 applies unchanged.

### A.4 Unchanged

```text
unknown_exposure_items=7               # unchanged, all still treated as contaminated
contamination_classifications=unchanged
project_research_conclusion_modified=false
past_verdicts_reopened=false
v7_artifacts_modified=false
network_requests=0
```

---

## Appendix B — Erratum: V7 feature-seed overlap (append-only, 2026-08-09)

```text
note_type=APPEND_ONLY_ERRATUM
note_date=2026-08-09
audit_findings_modified=false
audit_classifications_modified=false
audit_tables_modified=false
supersedes=Appendix_A_A.3_summary_line_only
```

**No audit fact in Sections 0–10 above is changed by this note.** This appendix
corrects an overbroad summary line that appeared in Appendix A §A.3, discovered
during implementation review of the frozen V8 design. Per the append-only
principle, A.3 is left as originally written; this entry supersedes only its
`v7_feature_seed_span_used_by_initial_v8=false` line.

**The error.** A.3 asserted that initial V8 does not use any part of the V7
feature-seed span. That assertion was too broad. The audit's own §2.3 records
the V7 production seed as the latest 252 valid observations per ticker with
cutoff `2026-08-07`, estimated to begin **approximately 2025-08**
(`U-5`, §7). Initial V8's `P_hist` runs `2016-04-01 → 2025-12-31` — which
**does** include the approximately 2025-08 → 2025-12-31 portion of the seed
span, even though it correctly excludes the later portion,
`2026-01-31 → 2026-08-07` (`P_gap`, unaffected).

**Corrected statement.**

```text
v7_feature_seed_span_used_by_initial_v8=PARTIAL
v7_feature_seed_overlap_subspan=approximately_2025-08..2025-12-31
v7_feature_seed_overlap_layer=A_DEVELOPMENT_T0_ONLY
v7_feature_seed_overlap_tickers=T0_ONLY_i.e._FIXED_V4_300
T1_T2_overlap_with_v7_seed=false
2026-01-31..2026-08-07_used_by_initial_v8=false      # A.3 line unaffected, still correct
pre_2015_used_by_initial_v8=false                     # A.3 line unaffected, still correct
v7_forward_window_2026-08-08_onward_used_by_initial_v8=false   # A.3 line unaffected, still correct
```

**Why this does not change any audit finding.** The overlap is on the `T0`
ticker block, which the audit already classified in full: §3.9 already records
≈2025-08 → 2025-12-31 as `EXPOSED_FOR_STRATEGY_DEVELOPMENT` (inside the V5-B and
V6-A-R2 signal windows) and the consolidated matrix in §4 already carries that
classification for `FIXED_V4_300`. This erratum does not reclassify any span;
it only corrects which *layer* of the frozen V8 design touches that
already-classified span. The corresponding correction in
`V8_HISTORICAL_RESEARCH_DESIGN.md` §3.2 explains why the overlap does not alter
any frozen design parameter: it is confined to `T0` (`evidential_weight=NONE`
in Layer A), and `T1`/`T2` remain untouched fresh cross-sections.

```text
network_requests=0
data_acquired=0
v7_artifacts_modified=false
```
