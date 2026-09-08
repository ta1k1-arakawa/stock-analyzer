# V9_017 SOURCE_B 2017 Public Schema Discovery Successor Design Draft

```text
study_id=V9_017_SOURCE_B_2017_PUBLIC_SCHEMA_DISCOVERY_SUCCESSOR
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
design_status=DRAFT_AWAITING_GPT_REVIEW
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

## 3. Prohibited scope

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

## 4. Bounded safe structural evidence

The future discovery runner may emit only a bounded safe result containing
closed status/failure fields, the verified input hash and byte count, and
structural observations sufficient for GPT judgment. Permitted observation
types are:

- table count;
- table, row, and column coordinates;
- table dimensions and row cell counts;
- normalized `TH` texts;
- normalized first-cell or row-label texts; and
- bounded counts and multiplicities.

The structural observation must preserve coordinates so that GPT can
distinguish table-local and row-local relationships. Text output is limited
to the permitted normalized `TH`, first-cell, and row-label observations;
the runner must not emit arbitrary cell dumps.

The safe result must never contain hrefs, URLs, raw HTML, filesystem paths,
exception text, PDF bytes or text, market/trading numeric outcome data,
protected/private data, or human identity. The runner must not select or
freeze a new report label. It reports observations only.

## 5. Hard stopping rule and later authority

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

## 6. Current checkpoint and non-claims

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
