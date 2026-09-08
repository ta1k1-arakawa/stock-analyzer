# V9_017 SOURCE_B 2017 Public Schema Discovery Successor Design Draft

```text
study_id=V9_017_SOURCE_B_2017_PUBLIC_SCHEMA_DISCOVERY_SUCCESSOR
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
design_status=REMEDIATED_AWAITING_GPT_REVIEW
execution_authorized=false
discovery_execution_limit=1
automatic_successor=false
```

## 1. Study identity and purpose

V9_017 is a new successor study created after the terminal V9_016 result. It
is not a retry or rerun of V9_016, V9_015, or V9_014. Its sole purpose is one
bounded, no-network public-schema discovery observation against the already
preserved exact 2017 public year-page lock. The observation may expose only
the structural labels needed for later Source-B input binding.

V9_017 has `evidence_role=INPUT_BINDING_ONLY` and
`profitability_evidential_capacity=ZERO`. It has no authority over trading
outcomes, T0, model selection, backtests, or profitability.

The study does not claim that any provider label changed and does not
presuppose a replacement report label. The future discovery runner only
reports bounded structural observations; GPT methodology authority decides
whether an exact report-row anchor and deterministic month-header grammar can
later be frozen.

## 2. Frozen input and execution boundary

The future execution may use only the already authoritative V9_016-bound 2017
year-page input:

```text
target_year=2017
expected_year_byte_count=98936
expected_year_sha256=1dc982e97b1d4ce7d52bc25631ddc46a219d22f82b393881f40e5d2478177821
```

The input must be bound by the exact byte count and SHA-256 above before the
structural observation. No root or year-page refetch, substitution,
reconstruction, or second input is permitted. The future runner may read the
single caller-supplied preserved input exactly once and must hash those same
bytes; it must not reopen or reread the file for hashing.

This design performs zero semantic reads, zero network requests, and consumes
zero human gates. The V9_016 authorization is not reusable. If governance
requires point-of-use authorization for the later observation, it must be a
fresh authorization for V9_017.

## 3. Implementation-before-observation sequence

The V9_017 stage ordering is frozen as follows:

1. V9_017 design receives GPT exact-SHA `PASS_FROZEN`.
2. A synthetic-only discovery runner is implemented.
3. Targeted synthetic tests pass.
4. The implementation is committed and pushed.
5. GPT exact-SHA review of the implementation returns `PASS`.
6. A no-network preflight verifies the exact repository, frozen-design
   provenance, and exact bound input provenance.
7. Exactly one real preserved-2017 semantic discovery execution occurs.
8. The safe result is inspected without network access.
9. GPT makes the methodology judgment.
10. Only if the evidence is sufficient, GPT separately freezes the exact
    report-row anchor and deterministic month-header grammar.
11. Any later production locator implementation/application is a separate
    implementation, targeted-test, and GPT exact-SHA review stage.

The real preserved 2017 input must not be semantically read during design,
runner implementation, or synthetic tests. This design task therefore has
zero real semantic reads.

The one-shot discovery budget is consumed when the real bound 2017 bytes
first enter semantic parsing or structural observation. After that point
there is no rerun, retry, alternate parser, additional observation, or
expanded output. A pre-semantic-read operational or preflight failure does
not authorize an automatic rerun; it is `STOP=CHATGPT_DECISION_REQUIRED`.

## 4. Prohibited scope

V9_017 must not perform or authorize:

- any network acquisition, refetch, URL resolution, or provider request;
- any PDF read or PDF semantic inspection;
- any protected Source-A read or private/sealed-data access;
- any T0, model, backtest, trading-date, outcome, or profitability
  calculation;
- any use or reuse of the consumed V9_016 authorization;
- any fallback, fuzzy match, category selection, report-label selection, or
  month-grammar selection by the runner; or
- any semantic inspection of 2019, 2020, 2022, or 2026 inputs.

Normalization for every exposed structural text value remains exactly:

```text
" ".join(raw_text.split())
```

No case-folding, lowercasing, NFC/NFKC normalization, fuzzy matching,
substring matching, regex repair, punctuation repair, first/last choice, or
fallback is permitted.

## 5. Deterministic bounded structural evidence universe

The future discovery runner may emit only a bounded safe result containing
closed status/failure fields, the verified input hash and byte count, and
structural observations sufficient for GPT judgment. Permitted observation
types are:

- table count;
- table, row, and column coordinates;
- table dimensions and row cell counts;
- normalized `TH` texts;
- normalized first-cell texts; and
- bounded counts and multiplicities.

The extraction universe is deterministic and exhaustive within those
structural roles:

- process tables in inherited-parser/document order;
- process rows in each table in row order;
- represent every column with its exact zero-based coordinate;
- emit every table dimension and every row cell count;
- emit every `TH` structural cell with its exact table, row, and column
  coordinate;
- for every nonempty row, emit the cell at exact column zero as the row's
  first-cell observation, including when that cell has empty normalized text;
- emit the first-cell observation and the `TH` observation separately when
  the same cell satisfies both roles; and
- order observations deterministically by table index, row index, column
  index, and role order `FIRST_CELL` before `TH` at an equal coordinate.

No content-based selection determines which `TH` or first-cell values are
exposed. There is no arbitrary additional cell dump, first-N/last-N
truncation, or silent omission. An empty row has no first-cell observation,
because it has no cell at column zero, while its zero row-cell count remains
emitted.

Normalization for all emitted structural text is exactly
`" ".join(raw_text.split())`. Structural role is the sole inclusion rule;
the runner performs no content filter, numeric parsing, or outcome
calculation. Thus any emitted value is a bounded structural text observation,
not market/trading numeric outcome data.

The safe result must never contain hrefs, URLs, raw HTML, filesystem paths,
exception text, PDF bytes or text, market/trading numeric outcome data,
protected/private data, or human identity. The runner must not select or
freeze a new report label. It reports observations only.

## 6. Required identity set

The deterministic month-header grammar judged after discovery must be
sufficient for exactly this inherited NORMAL identity set:

1. `2017-01 NORMAL_MONTHLY_REPORT2_OBJECT`
2. `2019-12 NORMAL_MONTHLY_REPORT2_OBJECT`
3. `2020-01 NORMAL_MONTHLY_REPORT2_OBJECT`
4. `2022-03 NORMAL_MONTHLY_REPORT2_OBJECT`
5. `2022-04 NORMAL_MONTHLY_REPORT2_OBJECT`
6. `2022-05 NORMAL_MONTHLY_REPORT2_OBJECT`
7. `2026-01 NORMAL_MONTHLY_REPORT2_OBJECT`

`2022-04 PRE_APRIL_1_REFERENCE_OBJECT` is not part of the month-header
grammar. Its inherited special resolver semantics are unchanged and receive
no new authority from V9_017.

## 7. Hard stopping rule and later authority

There is exactly one bounded V9_017 discovery execution. If that one
observation does not provide mechanically sufficient structural evidence for
GPT to freeze both:

1. one exact Source-B report-row anchor; and
2. one deterministic month-header grammar sufficient for the required fixed
   identities,

V9_017 terminates. The Source-B archive locator recovery route is then
abandoned, with no automatic V9_018 blind label/month-guessing successor.

The discovery result does not itself establish an anchor, label, category, or
grammar. If the evidence is sufficient, GPT must separately decide whether a
single exact row anchor and deterministic month-header grammar may be frozen.
No post-observation category, label, fallback, or grammar invention is
allowed.

If discovery succeeds, any subsequent work requires this sequence:

```text
V9_017 design GPT exact-SHA PASS
  -> one bounded discovery execution
  -> GPT methodology judgment and explicit freeze
  -> separate implementation and targeted tests
  -> GPT exact-SHA implementation review
  -> separately authorized execution as required
```

No later implementation, application to other years, PDF acquisition, or
network operation is authorized by this design draft.

## 8. Current checkpoint and non-claims

At this design checkpoint:

```text
V9_016=TERMINAL
V9_016_C2=FAIL_TERMINAL
V9_016_C2_FAILURE_CLASS=DATA_QUALITY_FAILURE
V9_016_C2_FAILURE_REASON=EXACT_SOURCE_B_REPORT_ROW_NOT_ESTABLISHED
V9_016_C2_ADMISSIBLE_CATEGORY_COUNT=0
V9_016_C2_CATEGORY_SELECTION=NONE
V9_016_C2_RERUN_ALLOWED=false
V9_016_STAGE_D=NOT_REACHED
V9_016_STAGE_E=NOT_REACHED
V9_016_REAL_YEAR_PAGE_READS=1
V9_016_NETWORK_REQUESTS=0
V9_016_PROTECTED_SOURCE_A_READS=0
V9_016_HUMAN_GATES_CONSUMED=0
T0=NOT_RUN
future_profitability=UNESTABLISHED
```

The V9_016 `EN_MONTH_ABBR_DOT` observation is not selected, and no actual
alternative report label was observed or established. The prior terminal
failure is not a transport, implementation, strategy, or profitability
failure. V9_017 itself has not executed, has made no network request, and has
performed no semantic read.

The next methodology decision is:

```text
NEXT_STUDY_METHODOLOGY=CHATGPT_DECISION_REQUIRED
```

The next checkpoint for this draft is `GPT_EXACT_SHA_INDEPENDENT_REVIEW`.
