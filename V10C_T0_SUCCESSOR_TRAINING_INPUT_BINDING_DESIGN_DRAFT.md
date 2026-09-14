# V10C T0 Successor Training-Input Binding Design

Status: `DRAFT_AWAITING_GPT_REVIEW`

This document defines input binding only. It does not implement or authorize
T0 execution, historical evaluation, model fitting, network access, or a
human gate.

## 1. Successor boundary

V10C is the completed training-provenance promotion successor. Its training
input is adopted exclusively from the fixed V10B locked artifact set. This
does not recover, reinterpret, or repair the historical V9_009 training
cache. V10B remains `TERMINAL_BLOCK_NONREUSABLE`.

The successor changes only the identity and binding of the unavailable
training-cache artifact. It does not change the inherited V9_009 economic or
T0 methodology: features, D1-to-D3 targets, 2020--2025 T0 years, TOP1
estimand, model parameters, signal grid, calendar, evaluation cache,
thresholds, costs, slippage, portfolio rules, model-selection rules, search
space, or stopping rule remain frozen.

The promoted manifest's observed `283` successful and `17` failed ticker
partition is immutable provenance identity. It is not a success threshold,
tuning target, or profitability evidence.

## 2. Exact promoted identities

The future successor implementation must bind all of the following before
any research payload, outcome, or model-relevant read:

### V10C adoption record

- schema: `V10C_TRAINING_PROVENANCE_ADOPTION_RECORD_V1`
- Git blob SHA-1:
  `7febddd4af7c82fe00ef7ba618403f4dcf8f6758`
- file SHA-256:
  `9f850c793434f574655b1a37592dc435d7b085a1798457a3b589621983badb68`

### Promoted V10B training artifact

- manifest SHA-256:
  `887c031a004f91a080fa53ab511711fff92c92527cb119878ab2c295ee13cd44`
- schema: `V10B_TRAINING_CACHE_MANIFEST_V1`
- `complete=true`
- `universe_mode=FIXED_V4_300`
- `successful_ticker_count=283`
- `failed_ticker_count=17`
- `ticker_count=300`
- payload convention: `locked_raw/{ticker}.json`
- attempt-receipt SHA-256:
  `44387e68bfc5de8bf54de8ca9694ba9f12bad2be779224946888eaea36d96eb7`

The exact V4 universe remains bound to:

- universe CSV SHA-256:
  `d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997`
- canonical ticker-list SHA-256:
  `12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7`
- ticker count: `300`

The evaluation cache remains separately bound to manifest SHA-256
`797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5` and
payload count `300`. It is not reacquired, rewritten, or substituted by this
successor.

The already-reviewed V10A calendar bridge remains the sole calendar
authority. Its fixed canonical calendar artifact, receipt, schema, coverage,
calendar hash, and provenance checks are inherited unchanged. This design
does not reopen or modify the calendar study.

## 3. Direct-read training loader contract

The future implementation must provide a dedicated V10C-promoted training
loader. It must not silently broaden the generic V9 cache loader and must
keep the reviewed evaluation loader as a separate unchanged path.

The training loader must:

1. accept only `V10B_TRAINING_CACHE_MANIFEST_V1` with the exact promoted
   manifest SHA-256 above;
2. require `complete=true`, fixed V4 universe identity, the exact observed
   `283`/`17`/`300` partition, and an exhaustive disjoint partition over the
   canonical V4 ticker set;
3. require every accepted payload path to be exactly
   `locked_raw/{ticker}.json`, with no path escape, symlink, junction, or
   reparse traversal;
4. verify every locked payload's exact byte count and SHA-256 against the
   manifest before parsing it;
5. reject missing or extra files under `locked_raw` and reject any manifest,
   receipt, or payload identity mismatch;
6. perform no refetch, recovery, fallback, ticker substitution, copy,
   cache reconstruction, or manifest rewrite; and
7. preserve the inherited missing-training behavior: a ticker absent from
   the training set may use the existing evaluation-cache path only where
   the inherited T0 logic already permits it.

The old unrecoverable training manifest
`72ae3db1186f2c9c113b1bafe1d37fb74a5627ac7ceed1dfc2473a24e060de85` is not
an accepted production identity in this successor and must never be
relabeled as recovered.

The external training root is an operational locator only. The implementation
must receive one explicitly supplied root, validate it, and never search,
guess, or select another root. The absolute path is not scientific identity
and must not be committed or emitted in safe provenance.

## 4. Parser and semantic boundary

The future T0 path must use the inherited V9 parser semantics required for
DataFrame and split-ratio construction. It must not invent a new Yahoo
parser, alter date or OHLCV behavior, or weaken malformed-input handling.

The V10B acquisition-time `v10b_v4_yahoo_parser_shim` acceptance was
acquisition evidence only. It is not proof that all promoted payloads are
compatible with the later inherited T0 parser. Compatibility is therefore a
future execution observation, not a design claim.

After exact binding and required authority pass, the future loader may read
the exact locked bytes, verify their hash/size closure, and parse them using
the inherited T0 parser. If a locked payload is incompatible with that
parser or the frozen T0 data contract:

- emit `NO_VERDICT_DATA_INCOMPATIBLE`;
- do not emit `STOP` or `CONTINUE`;
- do not refetch, substitute, repair, or change parser methodology; and
- do not infer a strategy or profitability failure.

## 5. No-leakage validation order

The future real path must fail closed in this order:

1. validate repository identity, authoritative branch, clean tree, reviewed
   implementation SHA, and local/remote provenance;
2. validate the reviewed V10A calendar input binding;
3. validate the exact V10C adoption-record Git blob, file hash, schema, and
   promoted manifest binding;
4. validate the external training-root safety and exact manifest and attempt
   receipt metadata without reading locked payload bytes;
5. validate fixed V4 universe identity and the unchanged evaluation-cache
   identity;
6. require the future point-of-use authority boundary;
7. only then read the exact training/evaluation payload bytes, verify the
   hash/byte closure, and parse with inherited semantics;
8. only after those checks build inherited V9 features, targets, models, and
   the frozen T0 safe result.

No cache, outcome, target, label, return, price, or model work may occur
before the required input binding passes. No post-observation calendar
modification is permitted.

## 6. Authority and future implementation checkpoints

This design grants no authority. Before implementation can begin, this
design requires GPT exact-SHA review with `CRITICAL=0`, `HIGH=0`, and
`MEDIUM=0`. The future implementation must then receive its own GPT
exact-SHA PASS; its exact implementation SHA is unknown at this design
commit and must be supplied and bound only after that review.

The future sequence is:

1. design review PASS;
2. successor training-loader implementation and targeted synthetic tests;
3. implementation exact-SHA review PASS;
4. Phase A NO-NETWORK preflight of Git, calendar, adoption record,
   manifest/receipt metadata, root safety, universe, and evaluation identity;
5. GPT Phase-A adjudication PASS;
6. fresh point-of-use T0 authorization, distinct from V10C adoption
   authorization, before any locked training raw byte or evaluation/outcome
   payload read;
7. one bounded T0 execution under the frozen methodology; and
8. safe adjudication of the resulting T0 result.

Until those checkpoints complete:

- `V10C_T0_AUTHORIZED=false`;
- `V10C_HISTORICAL_EVALUATION_AUTHORIZED=false`;
- `V10C_PRIVATE_SEALED_ACCESS_AUTHORIZED=false`; and
- no fresh human gate is consumed.

The V10C adoption authorization is not reusable for T0 and does not
authorize evaluation, model fitting, backtesting, or private/sealed access.

## 7. Failure taxonomy

The successor must keep these classes disjoint:

### GOVERNANCE/PREFLIGHT_FAILURE

Repository, branch, reviewed-SHA, Git provenance, path-safety, authority,
or pre-payload binding failure. It emits no T0 scientific result and stops
before research-data access.

### NO_VERDICT_DATA_INCOMPATIBLE

The exact promoted artifact cannot satisfy the frozen manifest, payload,
universe, evaluation, or inherited parser/data contract. It emits the
existing safe `NO_VERDICT_DATA_INCOMPATIBLE` result with no `STOP` or
`CONTINUE`, and never authorizes refetch or substitution.

### IMPLEMENTATION_FAILURE

Unexpected programming, wrapper, runtime, serializer, or environment defect.
It does not become a data verdict and does not emit `STOP` or `CONTINUE`.

### SCIENTIFIC_T0_RESULT

Only after all exact bindings, required authority, payload validation, and
inherited computation pass may the existing scientific result be emitted.
`STOP` or `CONTINUE` retains its frozen meaning; `CONTINUE` means only “not
killed” and is not profitability evidence.

## 8. Research integrity and evidential capacity

The future implementation must preserve:

- no post-outcome selection;
- no favorable ticker substitution;
- no refetch to alter the observed `283`/`17` partition;
- no threshold or hyperparameter tuning;
- no holdout or evaluation inspection before the authorized binding point;
- no methodology, cost, slippage, calendar, portfolio, or stopping-rule
  change; and
- no claim that V9_009 historical training data was recovered.

The promoted training provenance and this input-binding design have zero
profitability evidential capacity. `future_profitability_established=false`
remains required.
