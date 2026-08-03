# V4R1 Engineering Plan — Pre-registered

## Status and lineage

- `series`: `V4R1_ENGINEERING_PLAN`
- `base_commit`: `9279417769bf755db38a1589e026477f386d23e0`
- Old V4 final implementation status: `IMPLEMENTATION_RETRY_BLOCKED`.
- Old V4 implementation HEAD: `9279417769bf755db38a1589e026477f386d23e0`.
- Old V4 price acquisition: `0`; model fits: `0`; real-data backtests: `0`; research result: unjudged.

V4R1 does not resume, overwrite, amend, or reinterpret old V4.  It is not a
post-result adjustment: no market data, model fit, backtest, or research result
was produced by old V4.  V4R1 newly pre-registers engineering method and
engineering budget only.

## Immutable scientific conditions

The sole scientific specification remains `V4_META_LABEL_DESIGN.md`, SHA-256
`07039948aa7a1180d506b3089a0bd5612dda24559968c510e0cb92935b48055a`.
Its hypothesis, fixed universe of 300, periods, 15 features, model parameters,
threshold `0.55`, three folds, execution rules, 10 BLOCKED conditions, and 17
acceptance conditions are unchanged.

- Canonical LF universe SHA-256: `d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997`.
- Ticker-list SHA-256: `12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7`.
- Price period: `2015-01-01` through `2019-12-31`; signal period:
  `2016-04-01` through `2019-12-31`; all 2020-or-later data are forbidden.
- Scope is `SURVIVORSHIP_BIASED_RESEARCH_ONLY`; Yahoo is the only permitted
  price source.  Real orders, shadow, deploy, and schedule are forbidden.

## Shared engineering controls

Each stage uses a new branch and worktree.  Its implementation may modify only
`src/v4_meta_label.py`, `scripts/run_v4_meta_label_prototype.py`, and
`tests/test_v4_meta_label.py`, and only within the stage scope below.  Each
stage has at most one implementation commit and at most one bug-fix retry.
After implementation, a separate session performs a read-only independent
review.  A PASS is required before the next stage; a BLOCKED review never
triggers automatic correction.  Design changes require a HUMAN_GATE.  Amend,
rebase, force-push, merge, and tag are prohibited.

## Stage 1 — DATA_AND_CACHE

Scope: Yahoo URL construction and exact HTTPS host validation; redirect
rejection; transport injection; Yahoo chart parser; split extraction;
immutable cache; payload hashes; network audit; and offline cache validation.

Permitted: fake transport, synthetic JSON, and repository-external temporary
directories.  Forbidden: real Yahoo communication or price acquisition,
LightGBM, portfolio logic, and research-artifact generation.

Completion gate: independent review passes direct Stage 1 tests.  No automatic
transition occurs.  Budget: one implementation commit and one bug-fix retry.

## Stage 2 — FEATURE_CANDIDATE_LABEL

Scope: raw/adjusted OHLC; all fixed 15 features; eligibility; cross-sectional
features; candidate selection; split exclusion; STOP/GAP STOP/TIME; commission
fields; labels; and fold training boundaries.

Permitted: synthetic OHLCV and temporary directories only.  Forbidden: real
prices, LightGBM fit, portfolio evaluation, and formal artifacts.

Completion gate: every feature equation has a direct test and an independent
review passes candidate, execution, and label behavior.  Budget: one
implementation commit and one bug-fix retry.

## Stage 3 — PORTFOLIO_AND_METRICS

Scope: Baseline portfolio; fixed Baseline trade list; V4 accept/ABSTAIN;
capital path; no exit-day reuse; classification and portfolio metrics; named
10 BLOCKED conditions; named 17 acceptance conditions; and three final
decision states.

Permitted: fixed synthetic candidates and fixed fake probabilities.  Forbidden:
real prices, LightGBM fit, and formal research judgement.

Completion gate: independent review passes capital, metric, and decision tests.
Budget: one implementation commit and one bug-fix retry.

## Stage 4 — MODEL_ORCHESTRATOR_ARTIFACTS

Scope: LightGBM factory; fake-classifier injection; fit/predict_proba execution
path; OOF; CLI modes; two complete runs; byte-identical comparison; eight
artifacts; and atomic output.  Tests use fake classifiers only; real LightGBM
fit is forbidden.

Completion gate: all 24 required test classifications have independent direct
tests, there are at least 32 test methods, and the final independent review is
`IMPLEMENTATION_REVIEW_PASS`.  Budget: one implementation commit and one
bug-fix retry.

## Formal-execution HUMAN_GATE

Even after all four reviews pass, no Yahoo acquisition starts automatically.
A separate HUMAN_GATE must verify every stage commit SHA, final review PASS,
unchanged design/universe, zero real Yahoo calls/model fits/real-data
backtests, remaining experiment budget, and explicit external `cache-dir` and
`output-dir`.  This plan creates no cache, output, result, or order.
