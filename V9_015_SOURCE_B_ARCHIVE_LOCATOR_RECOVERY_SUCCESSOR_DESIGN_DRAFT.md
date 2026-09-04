# V9_015 SOURCE_B Archive-Locator Recovery Successor Design Draft

```text
study_id=V9_015_SOURCE_B_ARCHIVE_LOCATOR_RECOVERY_SUCCESSOR
design_status=DRAFT_AWAITING_GPT_REVIEW
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
```

## 1. Purpose and boundary

V9_015 is a new, explicitly created successor study identity. It exists only
to recover the official SOURCE_B archive input-binding path required for the
minimum route toward a future T0. It is not an automatic retry, repair, or
reopening of V9_014, and it is not a profitability study.

V9_014 remains terminal with its accepted Attempt-2 result:
`DATA_QUALITY_FAILURE` / `ROOT_LOCATOR_FAILURE`. No V9_014 Attempt 3 is
permitted. The preserved Attempt-2 official root payload is immutable and is
the sole root calibration input for this successor. V9_015 never refetches,
replaces, reconstructs, or guesses the root.

This document freezes design intent only. It performs no network request, no
root-payload read or hash, no parser/locator execution, no protected read,
and no human-gate consumption.

## 2. Exact inherited authority

The following references are inherited unchanged by exact Git identity. A
later implementation may not restate, weaken, or substitute their semantics:

| Authority | Exact Git object at the design checkpoint |
|---|---|
| V9_014 authority successor design draft | `V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR_DESIGN_DRAFT.md` / blob `2bbacbf37ab961d1cbf416b7fd476db18778c5b7` |
| SOURCE_B PDF structural calibration contract | `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md` / blob `bc9d5ca4383ddfed0be429f8ea8bb1fd16eccfa9` |
| SOURCE_B locator mechanics | `src/v9_014_jpx_monthly_auction_activity_source_b_locator.py` / blob `3e69d7ca31ef3237035855c897fd19e2039b582c` |
| SOURCE_B archive HTML extraction mechanics | `src/v9_014_jpx_monthly_auction_activity_source_b_archive_parser.py` / blob `080a28eb7022de528df2d0e40eda2d1fa04924e5` |
```

The inherited authority includes all of the following, without alteration:

- official JPX/TSE SOURCE_B provider and the exact official archive root
  `https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html`;
- exact Report 2 identity, `Stock Trading Volume & Value`;
- inclusive coverage `2017-01-01..2026-01-31`, or logical months
  `2017-01..2026-01`;
- `110` required physical SOURCE_B objects, including the two distinct
  2022-04 objects and the fixed eight-object calibration bundle;
- `pdfplumber==0.11.10` and the outcome-safe structural PDF contract;
- zero-redirect transport with `ALLOWED_REDIRECT_COUNT=0`, exact
  requested/resolved URL equality, and `REDIRECT_IS_RETRYABLE=false`;
- `timeout=30`, `MAX_PRE_COMPLETE_ATTEMPTS_PER_URL=3`, and retries only for
  transport exceptions and HTTP statuses `{408, 429, 500, 502, 503, 504}`;
- first-complete-payload locking and immutable raw-lock provenance;
- SOURCE_B PDF unit, interval, and activity semantics;
- the inherited SOURCE_A and SOURCE_C roles and their exact boundary rules;
- no T0, profitability, model, or downstream research authority.

V9_015 adds no source, provider, archive alias, URL template, report alias,
month substitution, or interpretation of the PDF unit contract.

## 3. Sole preserved root input

The root input for V9_015 is the already-preserved official root payload from
V9_014 Attempt 2. It is not copied into Git and is not rewritten. Its raw
bytes remain the source of truth. The V9_015 state and evidence records may
contain only safe provenance such as an expected/observed SHA-256, byte count,
and closed status fields; they must not contain raw HTML, arbitrary hrefs,
URLs beyond the frozen public authority, response bodies, or local durable
paths.

The following operations are prohibited at every V9_015 stage:

- root refetch, root replacement, root-page reconstruction, or root fallback;
- deriving a year URL from a filename, path pattern, archive number, or year;
- accepting a different provider, language site, archive root, or response;
- treating a failed or unavailable root binding as permission to retry until
  PASS.

If the preserved root cannot be safely bound by the gates below, V9_015
fails terminal. The result is not a V9_014 retry and does not authorize an
automatic successor.

## 4. Fixed calibration identities

The inherited C1 bundle remains exactly eight identities, once each and in
this canonical order:

```text
2017-01 NORMAL_MONTHLY_REPORT2_OBJECT
2019-12 NORMAL_MONTHLY_REPORT2_OBJECT
2020-01 NORMAL_MONTHLY_REPORT2_OBJECT
2022-03 NORMAL_MONTHLY_REPORT2_OBJECT
2022-04 PRE_APRIL_1_REFERENCE_OBJECT
2022-04 NORMAL_MONTHLY_REPORT2_OBJECT
2022-05 NORMAL_MONTHLY_REPORT2_OBJECT
2026-01 NORMAL_MONTHLY_REPORT2_OBJECT
```

Any missing, extra, duplicate, substituted, or differently ordered identity
fails closed. V9_015 does not expand the calibration bundle or make a unit,
date, relation, inactivity, trading-date, or profitability decision.

## 5. Preregistered successor stages

The stages below must complete in order. A later stage cannot repair or
reinterpret an earlier failure.

### A — V9_015 design review

GPT must independently review this exact design and return PASS. Until then,
the design remains `DRAFT_AWAITING_GPT_REVIEW` and no successor execution is
authorized.

### B — exact hash-only preserved-root input binding

After Stage A PASS, perform a no-network binding check against the preserved
Attempt-2 root payload only. The caller must privately supply the pre-existing
Attempt-2 durable artifact and capture locations at execution time. Those
locations are inputs to the check only: they must never be committed or
printed, and no environment variable or path is recorded in safe output.

The binding is ordered and fail-closed. Before reading or hash-validating any
raw root payload, mechanically verify the supplied safe Attempt-2 capture
against this frozen evidence:

1. The stdout capture has exactly `918` bytes and SHA-256
   `eea4a352087d6a0438e8384449a01eea1934bf9d925b9a771fafdb9a4f27fb30`.
2. The stderr capture SHA-256 is
   `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
3. The captured process exit code is `1`.
4. The attempt artifact SHA-256 is
   `e044431beb62c42e28d571ee3516d7b1123312839e071fe2fb5fe5a69791e0ac`.
5. The failure artifact SHA-256 is
   `c7f6dbcd74a8b4b7560d1e6d0ded8999d25f1bbe83d88f0900322f7740123bf`.

Only after the stdout capture size and hash match may it be parsed as the
reviewed safe `AcquisitionResult` JSON. Its closed fields must then match:

```text
status=CALIBRATION_ACQUISITION_FAILURE
failure_class=DATA_QUALITY_FAILURE
reason=ROOT_LOCATOR_FAILURE
unique_complete_payload_count=1
year_page_count=0
calibration_pdf_count=0
probe_invocations=0
```

The parsed `locked_payloads` value must contain exactly one entry, with no
duplicates or alternate entries, and that entry must be exactly the root
record shape below:

```text
role=root
relative_path=raw/archive_root.html
identity=null
sha256=<exact 64 lowercase hexadecimal characters>
byte_count=<nonnegative integer>
```

No year-page or PDF lock is accepted in this record, and every required field
must be present. The `sha256` and `byte_count` extracted from this already
bound safe stdout record become the only expected root baseline. They must not
be recomputed from later bytes, inferred from the attempt/failure artifacts,
or replaced by a current-file value.

Only after all preceding capture, JSON, field, and lock-record checks pass may
Stage B read/hash metadata of the caller-supplied preserved raw root. Its
SHA-256 and byte count must equal the bound baseline exactly. Missing,
malformed, inaccessible, or mismatched safe captures, artifacts, lock record,
raw root, hash, or byte count is a terminal input-binding/governance failure;
there is no refetch, replacement, repair, recomputation, or blessing of
current bytes. Root refetch remains zero.

Stage-B output may expose only the approved hashes, counts, booleans, and
closed status. It must never expose raw HTML, hrefs, arbitrary URLs, local
paths, unmasked text, or exception text.

### C1 — offline structural probe implementation and synthetic GPT PASS

Before any preserved-root structural read, implement and independently review
the dedicated V9_015 structural probe using synthetic fixtures only. C1 takes
bytes as its only document input and freezes these mechanics:

1. Decode the input strictly as UTF-8. A decode failure is a closed
   `IMPLEMENTATION_FAILURE` or `DATA_QUALITY_FAILURE`, as determined by the
   implementation's closed input-validation boundary; no alternate encoding
   detection or fallback is allowed.
2. Use exactly Python stdlib
   `html.parser.HTMLParser(convert_charrefs=True)`. BeautifulSoup, lxml,
   browsers, regex parsers, fallback parsers, and parser shopping are
   prohibited.
3. Candidate-bearing structural categories are closed to `ANCHOR_HREF`
   (`<a>` inner visible text plus its `href` attribute) and `OPTION_VALUE`
   (`<option>` inner visible text plus its `value` attribute). No other tag or
   category can become a locator candidate without a new study/methodology
   decision.
4. `VISIBLE_TEXT` is diagnostic-only. It can never supply a child URL,
   satisfy deterministic binding, or be promoted to a candidate after the
   real root is observed.
5. For anchor and option candidates, accumulate only contained text from the
   deterministic parser event stream and normalize it exactly as
   `" ".join(raw_text.split())`. No lower/casefold, NFKC/NFC, fuzzy or
   substring matching, year-from-URL repair, punctuation repair, or other
   normalization is permitted.
6. Required year labels are exactly `2017`, `2019`, `2020`, `2022`, and
   `2026`. A candidate is eligible only when its normalized inner text equals
   one exact required token and its candidate attribute exists as a nonempty
   string. Attribute content is never emitted in C1/C2 safe evidence.
7. Preserve multiplicity and never select first or last. For every required
   year and each candidate category, classify multiplicity exactly as
   `ZERO`, `ONE`, or `MANY`.
8. Malformed relevant nesting or state, duplicate relevant attributes,
   impossible parser state, non-bytes input, unsafe output, and schema
   violations fail closed. No malformed HTML is silently repaired.

The deterministic category-selection rule is frozen before C2. One category
is admissible only if the same category has multiplicity `ONE` for every one
of the five required years and the other candidate-bearing category has
multiplicity `ZERO` for every required year. If neither category satisfies
this rule, or if both or mixed categories supply candidates, deterministic
archive-year binding is not established. There is no post-observation
category choice, precedence, merge, or fallback. Raw href/value strings are
never emitted; URL resolution and validation remain later under the reviewed
locator mechanics.

The separate C1 implementation task must use synthetic fixtures only and must
complete targeted tests for: unique all-five-year anchor candidates; unique
all-five-year option candidates; a missing year; duplicate/MANY candidates;
mixed anchor and option candidates; both categories complete; required year
only in `VISIBLE_TEXT`; empty or missing href/value; near or nonexact labels;
malformed relevant structure; and output-free raw href/value/HTML/arbitrary
text. It must perform no preserved-root read, hash, or content access, no
network, no protected read, no child URL resolution, and no real locator
execution. GPT exact-SHA independent PASS is required before C2.

### C2 — no-network execution on the exact Stage-B-bound root

Only after C1 implementation and its GPT exact-SHA PASS, and only after Stage
B PASS, run that exact reviewed probe once against the exact Stage-B-bound
preserved root bytes. C2 must bind the reviewed C1 implementation SHA/blob,
prove that the root SHA-256 and byte count still equal the Stage-B baseline,
and emit only the closed category-specific schema in Section 6. C2 performs
zero network requests, zero root refetches, zero child URL resolution or
fetches, and never emits raw HTML, href/value strings, arbitrary text, local
paths, or exception text.

The C2 result is safe structural evidence only. It cannot select a category
after observation; Stage D may freeze archive-year mechanics only if the
result is consistent with the preregistered category rule above. A failed
Stage-B binding, missing reviewed C1 binding, root baseline mismatch, probe
failure, or unsafe receipt is terminal and cannot trigger a refetch or a
different probe.

### D — GPT freeze of archive-year mechanics

After C2 PASS, GPT must freeze the deterministic navigation mechanics that
are supported by the safe structural evidence. A Stage D freeze may select
only the predeclared category-selection rule and exact-token rule; it may not
be inferred from an arbitrary href pattern or from a hidden/raw root
inspection. The freeze must state the accepted category, exact multiplicity
rule, URL-binding rule, and closed failure behavior.

If the preserved root does not establish one deterministic official
year-navigation path, V9_015 fails terminal here. No provider substitution,
URL guessing, root retry, or automatic successor is allowed.

### E — locator implementation and synthetic verification

After Stage D PASS, implement only the GPT-frozen mechanics by reusing the
reviewed SOURCE_B parser, locator resolvers, and `validate_jpx_url` directly.
Add synthetic tests for exact year selection, exact parent binding, zero and
multiple candidates, off-domain/non-PDF rejection, and no URL construction.
GPT exact-SHA PASS is required before real execution.

### F — fresh human-authorized real execution from the preserved root

Only after Stages A-E PASS may a fresh, separately authorized execution
begin. It starts from the preserved Attempt-2 root lock and performs zero
root requests. It may fetch only the required five year pages and the fixed
eight calibration PDFs through the frozen transport and locator mechanics.
It must create fresh durable attempt metadata before the first child request,
use exclusive first-complete-payload locks, and preserve all prior locks on
failure. Stage F is not a V9_014 retry and does not reuse Attempt-2
authorization.

The inherited transport remains exact: HTTPS JPX URLs through
`validate_jpx_url`, `ALLOWED_REDIRECT_COUNT=0`,
`REQUESTED_URL_MUST_EQUAL_RESOLVED_URL=true`,
`REDIRECT_IS_RETRYABLE=false`, `timeout=30`, at most three attempts before a
complete payload per URL, and retries only for the frozen transport
exceptions/statuses `{408, 429, 500, 502, 503, 504}`. A redirect, URL
mismatch, nonretryable status, locator failure, or lock/provenance failure
fails closed. No root request may be introduced.

### G — inherited fixed-eight PDF calibration

Only after all eight child PDF locks pass may the unchanged C1
`probe_calibration_bundle` run once against the exact locked bytes and hashes.
The C1 probe remains limited to masked structural evidence with
`pdfplumber==0.11.10`; it performs no unit acceptance, classification,
relation evaluation, trading-date generation, or profitability inspection.

### H — downstream continuation gate

Only after every required V9_015 gate has GPT PASS, and only under a fresh
authority explicitly covering the next operation, may the inherited full
SOURCE_B materialization path be considered. Stages E-H do not themselves
authorize T0, model work, profitability analysis, protected SOURCE_A reads,
or any SOURCE_C action outside its inherited role.

## 6. Safe structural calibration contract

The C2 output is a bounded input-binding diagnostic, not research data.
Its complete allowed observation schema is:

```text
schema_version=V9_015_ROOT_STRUCTURE_CALIBRATION_V1
root_sha256=<64 lowercase hex>
root_byte_count=<nonnegative integer>
html_parser_success=<true|false>
structure_failure_class=<closed enum or null>
tag_counts={html,head,body,table,thead,tbody,tr,th,td,a,option}
anchor_count=<nonnegative integer>
option_count=<nonnegative integer>
visible_text_node_count=<nonnegative integer>
required_year_anchor_token_counts={2017,2019,2020,2022,2026}
required_year_anchor_nonempty_href_counts={2017,2019,2020,2022,2026}
required_year_anchor_multiplicity={2017,2019,2020,2022,2026}
required_year_option_token_counts={2017,2019,2020,2022,2026}
required_year_option_nonempty_value_counts={2017,2019,2020,2022,2026}
required_year_option_multiplicity={2017,2019,2020,2022,2026}
required_year_visible_token_counts={2017,2019,2020,2022,2026}
anchor_category_complete_unique=<true|false>
option_category_complete_unique=<true|false>
other_candidate_category_all_zero=<true|false>
deterministic_candidate_category=<ANCHOR_HREF|OPTION_VALUE|null>
all_required_years_deterministically_bindable=<true|false>
safe_calibration_status=<PASS|FAIL_TERMINAL>
```

The maps use only the five preregistered year labels and integer counts or
closed multiplicity buckets (`ZERO`, `ONE`, `MANY`). They may not contain
raw text, arbitrary labels, href strings, URL strings, HTML fragments, or
exception messages. Numeric values in this schema are counts/metadata only;
they are never trading values, units, or classification inputs.

The C1/C2 structural pass may observe exact parser tag/category membership,
whether a required year token occurs in each allowed category, whether the
corresponding structural candidate has a nonempty href/value, and whether
the candidate multiplicity is zero/one/many. The five-year category fields
are never merged. `required_year_visible_token_counts` is diagnostic only
and cannot provide a candidate. The probe may not normalize arbitrary text,
lowercase/case-fold, fuzzy-match, repair punctuation, infer a year from a
URL, or choose among multiple candidates.

The only acceptable deterministic binding is one exact required-year token
with one eligible structural candidate in the same admissible category for
all five years, while the other candidate category is zero for all five. Its
href/value is resolved and validated only by the reviewed mechanics at the
later Stage D/E boundary. Structural calibration alone never exposes that
attribute or accepts a URL.

## 7. Failure and evidence policy

V9_015 uses closed safe outcomes only:

- `PLUMBING_FAILURE_RETRIABLE` for bounded pre-complete transport plumbing
  failure where the frozen transport policy explicitly permits retry;
- `DATA_QUALITY_FAILURE` for missing, ambiguous, substituted, malformed, or
  non-authoritative structural/input-binding evidence;
- `GOVERNANCE_FAILURE` for missing stage authority, root replacement,
  output collision, unauthorized execution, or gate-order violation;
- `IMPLEMENTATION_FAILURE` for an implementation defect or unsafe receipt/
  lock operation;
- `CHATGPT_DECISION_REQUIRED` when the frozen design does not determine a
  safe next action.

The safe evidence role is `INPUT_BINDING_ONLY` and its profitability
evidential capacity is `ZERO`. No V9_015 artifact may claim a signal, a
profitable edge, a trading calendar, a model result, or T0 readiness.

## 8. Explicit non-claims and current status

At this design checkpoint:

```text
V9_014_STATUS=FAIL_TERMINAL
V9_014_REOPENED=false
V9_014_ATTEMPT_3=false
V9_015_AUTOMATIC_RETRY=false
V9_015_ROOT_REFETCH_ALLOWED=false
V9_015_LOCKED_ROOT_CONTENT_INSPECTED=false
V9_015_REAL_NETWORK_REQUESTS=0
V9_015_PROTECTED_READS=0
V9_015_HUMAN_GATES_CONSUMED=0
V9_015_T0_STATUS=NOT_RUN
V9_015_FUTURE_PROFITABILITY=UNESTABLISHED
V9_015_DESIGN_STATUS=DRAFT_AWAITING_GPT_REVIEW
V9_015_NEXT_STATE=GPT_EXACT_SHA_INDEPENDENT_REVIEW
V9_015_NO_AUTOMATIC_SUCCESSOR=true
```

No stage-specific execution is authorized by this draft. A later decision
must explicitly authorize each boundary, and any unresolved methodological
choice stops with `CHATGPT_DECISION_REQUIRED`.
