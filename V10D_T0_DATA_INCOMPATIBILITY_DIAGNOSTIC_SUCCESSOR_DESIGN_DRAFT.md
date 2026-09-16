# V10D T0 Data-Incompatibility Diagnostic Successor Design

document_type=V10D_T0_DATA_INCOMPATIBILITY_DIAGNOSTIC_SUCCESSOR_DESIGN
status=DRAFT_AWAITING_GPT_REVIEW
study=V10D_T0_DATA_INCOMPATIBILITY_DIAGNOSTIC_SUCCESSOR
design_purpose=DIAGNOSTIC_ONLY

This is a new study identity created after the immutable V10C T0 Attempt 1
terminal outcome. It is not a rerun, repair, recovery, or reinterpretation of
V10C. It grants no execution authority. Its only purpose is to mechanically
localize the earliest failing frozen data contract, if an independently
reviewed future implementation is later authorized to inspect the exact
fixed artifacts.

## 1. Immutable V10C terminal boundary

V10C T0 Attempt 1 remains immutable terminal evidence:

- `implementation_sha=98090eeacc062a956456fb27aa37c87a8822e150`;
- Phase A=`PASS`, Phase B=`COMPLETED`, Phase C=`PASS`;
- one-shot authority consumed=`true`, human gate consumed=`true`, retry
  authorized=`false`;
- result class and T0 result=`NO_VERDICT_DATA_INCOMPATIBLE`;
- scientific T0 result established=`false`;
- strategy failure and profitability failure established=`false`.

The public safe adjudication record is
`V10C_T0_ATTEMPT_1_SAFE_ADJUDICATION.json`, schema
`V10C_T0_ATTEMPT_SAFE_ADJUDICATION_V1`, and its safe file hash is bound by
that record and repository history. V10D must not edit, reread, repair,
replace, or reinterpret that record. V10C must not be rerun under any name,
including by changing only a parser, cache, ticker, period, or wrapper.

`NO_VERDICT_DATA_INCOMPATIBLE` is neither `STOP` nor `CONTINUE`. It is not a
strategy, profitability, governance, implementation, or preflight verdict.
No exact failing payload, ticker, reason, price, return, score, or prediction
is inferred by this design.

## 2. Inherited scientific identities

V10D inherits the exact input identities and semantics used by V10C. It may
not select a replacement identity.

### 2.1 Training and evaluation artifacts

- promoted training manifest SHA-256:
  `887c031a004f91a080fa53ab511711fff92c92527cb119878ab2c295ee13cd44`;
- training schema=`V10B_TRAINING_CACHE_MANIFEST_V1`, complete=`true`, observed
  partition `283` successful / `17` failed / `300` universe entries;
- V10C adoption-record Git blob SHA-1:
  `7febddd4af7c82fe00ef7ba618403f4dcf8f6758`;
- V10C adoption-record file SHA-256:
  `9f850c793434f574655b1a37592dc435d7b085a1798457a3b589621983badb68`;
- training attempt-receipt file SHA-256:
  `44387e68bfc5de8bf54de8ca9694ba9f12bad2be779224946888eaea36d96eb7`;
- evaluation manifest SHA-256:
  `797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5`;
- evaluation payload count=`300`.

The V4 universe remains independently authoritative:

- universe CSV SHA-256:
  `d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997`;
- canonical ticker-list SHA-256:
  `12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7`;
- universe mode=`FIXED_V4_300`, count=`300`.

The evaluation manifest is bound by its exact raw-byte SHA and its actual
schema-2 metadata. The fixed V4 universe is a separate identity binding; no
manifest-provided ticker order is assumed. No evaluation or training payload
bytes may be read in V10D Phase A.

### 2.2 Calendar identity

The already-reviewed V10A calendar bridge remains the sole calendar
authority. V10D inherits it unchanged:

- bridge design Git blob SHA-1:
  `6df95aa8354c3d335a51747ee98ed9f2741c2410`;
- canonical calendar artifact `V10A_CANONICAL_CALENDAR.json` Git blob SHA-1:
  `b3d9dee8fb20abfd966400873a7f1ff18df2880b`;
- canonical calendar artifact SHA-256:
  `2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0`;
- calendar receipt Git blob SHA-1:
  `da76889db285062a8f9ac902263ed7c2e63dc43a`;
- calendar receipt SHA-256:
  `e7266539c8a11c59d3be775623fde6023e835940e981f17c03f6e0bdf65b005c`;
- calendar=`JPX`, coverage=`2017-01-01..2026-01-31`, trading-date count=`2217`;
- calendar package=`pandas_market_calendars==5.4.0`, Python=`3.12.10`.

V10D may verify these public identities, but may not regenerate, alter,
substitute, or post-outcome filter the calendar.

### 2.3 Parser and data semantics

The diagnostic must reuse the reviewed V10C/V9 production functions wherever
possible, including the inherited chart-payload parser, split-action
normalization, `normalize_inputs`, `normalize_calendar`, `build_dataset`,
`build_scoreable_population`, `attach_historical_targets`, and dataset
validation. It must not duplicate or silently broaden parser behavior.

The V10B acquisition-time parser shim is not evidence of compatibility with
the inherited V9 parser. Compatibility is an observation for V10D, not a
design assumption. Parser, OHLCV, date, split, missing-data, and malformed
input semantics remain exactly those of the reviewed V9/V10C path.

### 2.4 Frozen V9/V10C methodology

V10D does not change the scientific methodology. The inherited identity
remains V9 TOP1 kill-screen design SHA
`8079bb0956c105a3972e59f0e4ba21ea5b81b14a`, including:

- the ten frozen features: `return_1d`, `return_5d`, `return_20d`,
  `return_60d`, `volatility_20`, `atr14_percent`, `close_to_ma20`,
  `close_to_ma60`, `distance_from_high20`, and `volume_dryup`;
- D1-to-D3 target semantics and causal target availability;
- source feature history beginning `2016-09-01`, pre-evaluation training
  beginning `2017-01-01`, formal signal period `2018-01-01..2025-12-31`,
  and T0 years `2020..2025`;
- TOP1 estimand, signal grid, calendar, portfolio rules, thresholds, costs,
  slippage, model-selection rule, search space, and stopping rule;
- Ridge parameters `alpha=10.0`, `fit_intercept=true`;
- LightGBM parameters `n_estimators=300`, `learning_rate=0.02`,
  `num_leaves=7`, `max_depth=3`, `min_child_samples=100`, `subsample=0.7`,
  `subsample_freq=1`, `colsample_bytree=0.7`, `reg_lambda=10.0`,
  `random_state=20260823`, `n_jobs=1`, `deterministic=true`, and
  `force_col_wise=true`.

The observed `283/17` training partition is provenance identity, not a
threshold or tuning target. V10D has zero profitability evidential capacity.

## 3. Diagnostic contract

V10D is diagnostic only. A future execution may use the exact frozen
artifacts to identify the earliest failing stage, but it must stop at the
first failure and emit no scientific `STOP` or `CONTINUE` result. It must
never repeat attempts until a favorable category occurs.

### 3.1 Deterministic stage order

The future implementation must evaluate logical checkpoints in this exact
production-trace order. The diagnostic stops at the first failed checkpoint;
it does not execute the scientific scoring or screening operation represented
by a later checkpoint.

1. `INPUT_BYTE_OR_FILESET_CONTRACT`: verify exact manifest and file-set
   closure, raw-byte/hash/size metadata, fixed V4 universe identity, and
   V10A calendar metadata that are available before protected payload parsing.
2. `PARSER_NORMALIZATION_CONTRACT`: apply the inherited payload parser and
   its OHLCV, date, split-action, code, and missing-data normalization to the
   exact locked payload bytes, including malformed-input behavior.
3. `COMBINED_SERIES_CONTRACT`: verify the inherited training/evaluation
   combination, chronology, duplicate-date handling, split-action
   consistency, and required-series presence.
4. `FEATURE_TARGET_DATASET_CONTRACT`: follow the production construction
   order `build_scoreable_population`, `attach_historical_targets`, target
   percentile construction, and dataset structural, finite-value, and
   chronology preconditions.
5. `FORMAL_SCORING_PRECONDITION_CONTRACT`: preserve the pre-model order in
   `score_formal_dataset`: dataset validation; calendar normalization and
   `month_start` semantics; formal signal-year completeness; per-month
   `causal_training_rows`; causal training-row availability; and finite
   model-input X/y preconditions. This checkpoint ends before any real
   `Ridge.fit`, `StandardScaler.fit`/`transform` used for scoring,
   `LightGBM.fit`, prediction, or score generation.
6. `POST_SCORING_STRUCTURAL_TARGET_CONTRACT`: for conditions checked by the
   production path only after scoring, derive only their safe structural
   equivalents from the already-constructed dataset, calendar, and row
   coverage. This includes `FORMAL_TARGET_UNAVAILABLE`,
   `KILL_SCREEN_YEAR_INCOMPLETE`, and any other post-scoring
   `T0DataIncompatible` structural condition whose production order is
   established. This checkpoint never fits a model, generates predictions,
   calculates scores, or computes STOP/CONTINUE metrics. It must not be moved
   ahead of an earlier checkpoint; if exact production order or a safe
   structural equivalent cannot be established, report
   `UNKNOWN_DATA_INCOMPATIBILITY`.

Only an inherited `T0DataIncompatible` condition may map to one of these
stage classes. The reported class is the earliest such condition that the
frozen production trace would encounter; later stages are not evaluated
after an earlier failure.

### 3.2 No-model-fit and exception boundary

V10D is a structural data-contract diagnostic, not a scientific scoring run.
Its execution must not call `Ridge.fit`, `LightGBM.fit`, or any prediction
method; calculate model scores, TOP1 edge, yearly or aggregate metrics; or
calculate or emit scientific `STOP` or `CONTINUE`. `StandardScaler` fit or
transform operations that would be part of actual scoring are likewise
forbidden. The implementation may use only the inherited parsing,
normalization, combination, feature/target structure, calendar, and row
coverage logic needed to localize the existing incompatibility. A diagnostic
PASS therefore has zero profitability evidential capacity.

The exception taxonomy is explicit:

- only an inherited `T0DataIncompatible` exception or equivalent established
  frozen data-contract condition may map to a V10D stage class;
- inherited `T0ImplementationFailure`, unexpected Python or library
  exceptions, diagnostic-wrapper defects, serializer or safe-output
  validator defects, and impossible or ambiguous internal state are
  `IMPLEMENTATION_FAILURE`;
- an implementation failure must never be relabeled as
  `INPUT_BYTE_OR_FILESET_CONTRACT`, `PARSER_NORMALIZATION_CONTRACT`,
  `COMBINED_SERIES_CONTRACT`, `FEATURE_TARGET_DATASET_CONTRACT`,
  `FORMAL_SCORING_PRECONDITION_CONTRACT`,
  `POST_SCORING_STRUCTURAL_TARGET_CONTRACT`, or
  `UNKNOWN_DATA_INCOMPATIBILITY`;
- `UNKNOWN_DATA_INCOMPATIBILITY` is reserved for an observed and established
  `T0DataIncompatible` condition whose exact stage cannot safely be mapped.

Safe evidence must distinguish `RESULT_CLASS=DATA_INCOMPATIBILITY_DIAGNOSTIC`
from `RESULT_CLASS=IMPLEMENTATION_FAILURE`. Neither result authorizes
refetch, retry, repair, methodology change, or scientific T0. An
implementation failure is closed evidence requiring separate reviewed
handling, not permission to search for a favorable category.

### 3.3 Safe diagnostic output

The only permitted durable or emitted diagnostic fields are:

- schema and study identity;
- the exact future implementation SHA and inherited artifact hashes;
- boolean validation states and safe counts;
- the first failed stage and one of the stage-level classes above, or
  `UNKNOWN_DATA_INCOMPATIBILITY`;
- authority, retry, and execution counters.

Output must never contain ticker identities, private or machine-local paths,
raw JSON, OHLCV, split details, feature values, targets, labels, predictions,
model scores, returns, yearly metrics, aggregate metrics, or protected
payload content. A stage-level class is not a claim about any particular
ticker or data row.

## 4. Prohibitions and authority

V10D must never:

- refetch, recover, replace, rewrite, repair, or substitute a payload,
  manifest, cache, ticker, period, calendar, or parser;
- rerun V10C T0 or reuse V10C's consumed human gate;
- change features, targets, model parameters, thresholds, costs, slippage,
  partitions, signal grid, portfolio rules, stopping rule, or acceptance
  criteria;
- tune after observing a diagnostic category or repeat until a favorable
  category occurs;
- access private/sealed data, market research payloads beyond the exact
  authorized fixed-artifact diagnostic read, or any future T0 outcome.

This design grants no execution authority. A future Phase-A implementation
must be no-network and must not read protected payload bytes. Reading the
fixed payload bytes for stage localization requires a fresh V10D point-of-use
human authority. V10C authority is consumed and non-reusable.

Once a future V10D protected-data boundary is crossed, its authority is
one-shot and sticky consumed. No automatic retry, reset, deletion,
recreation, cache repair, or second attempt is permitted. The future safe
inspector must run after every post-boundary outcome and classify wrapper or
inspection failure separately from data incompatibility.

## 5. Future checkpoint sequence

The prospective sequence is:

1. V10D design review: GPT exact-SHA review with `CRITICAL=0`, `HIGH=0`, and
   `MEDIUM=0` required;
2. explicit design freeze and fresh human approval scoped only to this
   diagnostic design;
3. implementation and synthetic tests using no real payloads;
4. GPT exact-SHA implementation review with `CRITICAL=0`, `HIGH=0`, and
   `MEDIUM=0` required;
5. Phase A no-network preflight of repository, frozen provenance, public
   metadata, and diagnostic namespace, with zero payload-byte reads;
6. GPT Phase-A adjudication PASS;
7. fresh point-of-use one-shot V10D authority, distinct from V10C authority,
   before any fixed payload byte read;
8. exactly one bounded diagnostic execution over the exact artifacts;
9. mandatory no-network safe inspection of every post-boundary outcome; and
10. GPT exact-SHA adjudication of the safe diagnostic evidence.

No step grants T0 authority. A diagnostic PASS only means that the
diagnostic completed its fixed contract; it is not a data, strategy,
profitability, or T0 PASS.

## 6. Possible post-diagnostic decisions

Only after independent GPT adjudication may the project decide, without
using any protected detail in public evidence, whether:

- the fixed promoted artifact is intrinsically incompatible, in which case
  the data-quality outcome remains recorded as `NO_VERDICT_DATA_INCOMPATIBLE`;
- an implementation-only diagnostic defect exists, in which case it receives
  a separately scoped non-methodological remediation and review; or
- a genuinely new acquisition/data study identity is required, with its own
  design, freeze, authority, and methodology decision.

V10D itself does not authorize any of these follow-on actions and does not
authorize a subsequent scientific T0. `future_profitability_established`
remains `false`.

## 7. Research-integrity invariants

The V10C/V9 study remains unchanged. In particular, V10D preserves the
2020--2025 T0 periods, TOP1 estimand, D1/D2/D3 semantics, exact feature set,
Ridge/LightGBM parameters, scaler behavior, signal grid, thresholds,
stopping rule, costs/slippage, portfolio rules, promoted training
provenance, evaluation identity, and V10A calendar authority.

No outcome, diagnostic class, or implementation convenience may be used to
select favorable tickers, periods, payloads, parser behavior, features,
targets, models, thresholds, or execution assumptions. Until a separate
authorized study establishes sufficient forward-only evidence,
`future_profitability_established=false`.
