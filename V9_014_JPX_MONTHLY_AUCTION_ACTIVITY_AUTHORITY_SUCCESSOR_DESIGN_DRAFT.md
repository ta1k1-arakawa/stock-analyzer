# V9_014 JPX Monthly Auction-Activity Authority Successor Design Draft

## 1. Status and study identity

```text
study_id=V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
design_status=BLOCKED_AWAITING_REMEDIATION_REVIEW
document_status=DESIGN_REMEDIATION_DRAFT
```

V9_014 is a NEW successor study identity. It is not a retry, repair,
continuation, or re-execution of V9_011, V9_012, or V9_013. Those studies
remain terminal and immutable:

```text
V9_011_RESULT=FAIL_TERMINAL
V9_012_RESULT=FAIL_TERMINAL
V9_013_RESULT=DIAGNOSTIC_COMPLETE
V9_013_DIAGNOSTIC_CLASS=RELATION_OR_SENTINEL_FAILURE
```

This document authorizes no implementation, no code or test change, no
Python execution, no JPX / J-Quants / Yahoo / external-data acquisition, no
protected or private state read, no API-key read, no raw-payload acquisition
or reuse, no materialization, no T0 / cache / outcome / backtest / model
execution, and no profitability evaluation.

### 1.1 Review state

```text
V9_014_DESIGN_REVIEWED_SHA=48fb8c59975b5afa967c6ff34e3e9343a8551c51
V9_014_DESIGN_REVIEW_PARENT_SHA=05d0ed3ca5cea8e84ff386fb73c65dba575030c2
V9_014_DESIGN_REVIEW_RESULT=BLOCK
CRITICAL=0
HIGH=0
MEDIUM=1
V9_014_DESIGN_HIGH_1=RESOLVED
V9_014_DESIGN_HIGH_2=RESOLVED
MEDIUM_1=DECLARED_SHARE_UNIT_TOKEN_TO_MULTIPLIER_MAPPING_NOT_EXACTLY_FROZEN
```

The prior HIGH_1 and HIGH_2 remediations are accepted and remain in force.
This revision remediates MEDIUM_1 only, by freezing the exact declared
share-unit token-to-multiplier mapping (Section 5.2) and the exact per-column
required unit expectations (Section 7.5). OPEN_1 through OPEN_5 remain
CLOSED. No V9_014 design PASS is claimed.

## 2. Why a successor study is required

V9_012 froze TOPIX daily index bars as its actual-activity evidence source
and failed terminally with `ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`. V9_013
diagnosed that failure without repairing it and established that both V9_012
sources were schema- and data-quality-valid (`SOURCE_A=A_VALID`,
`SOURCE_B=B_VALID`) and that the failure was located in the frozen
relation/sentinel layer, with
`V9_013_FAILURE_CAUSE=TOPIX_ACTIVITY_EVIDENCE_DOES_NOT_EXCLUDE_FROZEN_2020_10_01_FULL_DAY_TSE_OUTAGE`.

### 2.1 Mandatory interpretation of the V9_013 sentinel value

```text
V9_013_SENTINEL_2020_10_01_INACTIVE=false
```

`false` means the frozen "inactive" sentinel check **FAILED**. It must not be
described, summarized, or reused as if `2020-10-01` had been classified
inactive by TOPIX activity evidence. The already-recorded companion values
`V9_013_LEFT_DIFF_COUNT=0` and `V9_013_MISSING_EXPECTED_EXCEPTION_COUNT=1`
are consistent with that reading.

### 2.2 Consequence for source selection

TOPIX / index OHLC activity cannot serve as actual TSE cash-equity
auction-activity evidence and is **prohibited** as a V9_014 source.

## 3. Frozen coverage and authority boundary

```text
coverage_start=2017-01-01
coverage_end=2026-01-31
logical_coverage_month_count=109
```

The required inclusive coverage is exactly `2017-01-01..2026-01-31`, spanning
exactly `109` logical calendar months `2017-01 .. 2026-01`, consistent with
the already-recorded `V9_010_CALENDAR_REQUIRED_SOURCE_MONTH_COUNT=109`.

`logical_coverage_month_count` is a month count, not a physical object count.
The required physical SOURCE_B object count is `110`: one Report 2 monthly
object for each of the 109 logical months, plus the one additional official
`(Reference) Status on April 1, 2022` object described in Section 7.3.

A missing, unresolved, duplicated, or ambiguous required object is a
feasibility/data-quality failure. It is never permission to shorten coverage,
interpolate a month, substitute a provider, or accept a partial calendar.

V9_014 has `evidence_role=INPUT_BINDING_ONLY`. It produces no signal, model,
outcome, backtest, or profitability evidence. Its
`profitability_evidential_capacity` is `ZERO` at every stage, including after
a full PASS.

## 4. Independent official source roles

Three independent official sources with separate provenance are required. No
source is authoritative alone.

### 4.1 SOURCE_A — scheduled TSE business-day superset

```text
role=SCHEDULED_TSE_BUSINESS_DAY_SUPERSET
provider=OFFICIAL_JQUANTS_MARKETS_CALENDAR
endpoint=https://api.jquants.com/v2/markets/calendar
base_query={"from":"2017-01-01","to":"2026-01-31"}
hol_div=omitted
scheduled_open_predicate=HolDiv in {"1","2"}
byte_provenance=PRESERVED_IMMUTABLE_V9_012_SOURCE_A_CHAIN
source_a_chain_sha256=aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45
source_a_page_count=1
fresh_source_a_acquisition_authorized=false
```

SOURCE_A keeps exactly the semantic role it had in V9_012. Its frozen
scheduled-open interpretation is inherited unchanged and must not be
modified, relaxed, widened, or reinterpreted:

- the payload root must be a JSON object with a list-valued `data` field;
- every row must be an object carrying required `Date` and `HolDiv` fields;
- `Date` must be a strict `YYYY-MM-DD` string, a valid ISO date, and within
  the frozen inclusive coverage;
- `HolDiv` must be exactly one of `{"0","1","2","3"}`;
- dates must be unique, and the SOURCE_A date set must equal every calendar
  date in the frozen coverage;
- `scheduled_open_dates` is exactly the set of `Date` values whose `HolDiv`
  is `"1"` or `"2"`.

V9_014 **must reuse** the preserved immutable V9_012 SOURCE_A locked chain
identified by the exact chain SHA-256 and page count above. There is no fresh
SOURCE_A acquisition, no new SOURCE_A network request, and no alternate
calendar endpoint. Before any SOURCE_A semantic read, the implementation must
mechanically prove the exact chain SHA-256 and page count; a mismatch is
`PRESERVED_V9_012_SOURCE_A_INPUT_BINDING_FAILURE` and stops the study.

That preserved state is protected durable evidence. Reading it later requires
fresh point-of-use human authorization; no prior V9_012 or V9_013
authorization is reusable.

### 4.2 SOURCE_B — regular-auction activity proof evidence

```text
role=ACTUAL_TSE_REGULAR_AUCTION_ACTIVITY_DATE_EVIDENCE
provider=OFFICIAL_JPX_TSE_MONTHLY_STATISTICS_REPORT_ARCHIVE
archive_root=https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html
report=Report 2 "Stock Trading Volume & Value"
table=Trading Volume & Value (Daily)
object_format=PDF
```

SOURCE_B supplies, per date and per required domestic-stock market segment,
the reported total Trading Volume and the reported "of which ToSTNeT"
Trading Volume. Because those figures are unit-quantized (Section 5),
SOURCE_B can **prove activity** but can **never prove inactivity**.

### 4.3 SOURCE_C — exceptional full-day auction-closure authority

```text
role=EXCEPTIONAL_FULL_DAY_AUCTION_CLOSURE_AUTHORITY
provider=OFFICIAL_JPX_TSE_MARKET_NEWS
document_date=2020-10-01
document_title=Treatment of Trades for Today at arrowhead
document_language=ENGLISH
```

SOURCE_C is a fixed public official document identity, preregistered before
acquisition. Its required semantic content is:

- the Auction Market had **no execution** that day; and
- ToSTNeT orders received by **08:56** had executions.

Both assertions must be present. SOURCE_C is the only permitted authority for
a full-day auction closure, precisely because SOURCE_B's quantized figures
cannot establish exact zero.

`source_c_confirmed_exception_set` is derived from the SOURCE_C document's own
official date together with its verified semantic content. It must equal
exactly:

```text
source_c_confirmed_exception_set == {"2020-10-01"}
```

No manually injected date is permitted. The exception date is never a literal
typed into the calendar pipeline, never appended, and never widened. Any
other resulting set is terminal `ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`.

### 4.4 Prohibited sources and substitutions

- TOPIX or any other index OHLC series as activity evidence;
- total stock trading volume alone as an activity criterion, because
  `2020-10-01` had no auction executions while ToSTNeT executions existed;
- generic weekday ranges, `pd.bdate_range`, national-holiday subtraction,
  OS/locale calendars, Yahoo observed dates, broker calendars, or any
  vendor-derived calendar;
- any alternate provider, endpoint, index, ticker, report, table, or document
  substituted after results are observed.

## 5. Reported-value interval semantics (HIGH_1 remediation)

The previous revision claimed that reported `thous.shs.` values could be
reconstructed as exact share counts. **That claim is deleted.** It was
invalid: JPX reports omit quantities below the indicated unit, so a reported
figure is a quantized lower-truncated observation, not an exact share count.
No exact share count and no exact zero can be recovered from a quantized
cell.

### 5.1 Interval rule

For a numeric integer cell value `q` with declared share-unit multiplier `m`:

```text
m == 1  =>  [q, q]
m >  1  =>  [q * m, q * m + (m - 1)]
```

All interval endpoints are exact integers computed with exact integer
arithmetic. Binary floating point must not appear anywhere on the parsing,
normalization, comparison, or aggregation path: no `float()`, no float dtype,
no float-backed numeric container, no float intermediate.

The declared unit is parsed explicitly from the source object, per report,
per table, and per required column. It is never assumed, never defaulted, and
never inherited across objects. An absent, unparseable, ambiguous, or
multiply-declared unit is a data-quality failure.

### 5.2 Declared share-unit token mapping

The accepted Trading Volume share-unit semantics are frozen exactly. There
are exactly two canonical semantic units:

```text
SHARES           multiplier=1
THOUSAND_SHARES  multiplier=1000
```

The source PDF may present the declared unit bilingually. Parser recognition
must be based on the exact normalized semantic tokens taken from the declared
unit cell, never on fuzzy substring matching.

Accepted English tokens:

```text
"shs."        -> SHARES
"thous.shs."  -> THOUSAND_SHARES
```

Accepted Japanese equivalents, recognized **only** when they occur in the
SAME declared bilingual unit cell:

```text
"株"    -> SHARES
"千株"  -> THOUSAND_SHARES
```

A bilingual declaration must be semantically consistent:

```text
株   + shs.        => valid SHARES
千株 + thous.shs.  => valid THOUSAND_SHARES
```

Any contradictory bilingual declaration is a data-quality failure.

The following are never accepted:

- guessed abbreviations;
- fuzzy matches;
- case-insensitive invented aliases;
- multipliers inferred from numeric magnitude;
- value-based unit guessing;
- unit inheritance from a previous month;
- any other multiplier.

Any other unit token or value is:

```text
UNSUPPORTED_SHARE_UNIT_DQ_FAILURE
```

The exact interval rule of Section 5.1 restated over the two frozen units,
using integer arithmetic only:

```text
SHARES           [q, q]
THOUSAND_SHARES  [q * 1000, q * 1000 + 999]
```

### 5.3 Per-segment adjudication

For each required segment on a given date, with total interval
`[total_lower, total_upper]` and ToSTNeT interval
`[tostnet_lower, tostnet_upper]`:

```text
DQ failure                  iff total_upper < tostnet_lower
DEFINITELY_AUCTION_ACTIVE   iff total_lower > tostnet_upper
otherwise                   NOT_PROVEN
```

The first condition is the structural impossibility check: ToSTNeT volume is
a subset of total volume, so an interval pair that cannot satisfy
`tostnet <= total` is a data-quality failure. The second condition is
equivalent to a strictly positive lower bound on
`auction_volume = total - ToSTNeT`.

`NOT_PROVEN` is **never** silently treated as zero, inactive, closed, or
absent. It means exactly that the reported quantization does not prove
regular-auction activity for that segment on that date.

### 5.4 Date-level rule

```text
d is PROVEN_AUCTION_ACTIVE  iff  at least one required segment for d's era
                                 is DEFINITELY_AUCTION_ACTIVE
```

`proven_auction_active_dates` is the set of dates satisfying that condition.
There is no `PROVEN_INACTIVE` state derivable from SOURCE_B. A date that is
not `PROVEN_AUCTION_ACTIVE` is unproven, not inactive.

### 5.5 JPX token semantics

- A numeric `0` in a column with `m > 1` is **not** an exact zero. Its
  interval is `[0, m - 1]`, exactly as the Section 5.1 rule yields for
  `q = 0`.
- A dash / "Nil or no value" token is **not** malformed and is not a parse
  error.
- A dash never, by itself, proves positive activity and never, by itself,
  proves exact zero. Mechanically: a dash establishes no lower bound above
  zero and no upper bound below the structural bound, so a required segment
  whose total or ToSTNeT cell is a dash can never be
  `DEFINITELY_AUCTION_ACTIVE`. It is `NOT_PROVEN` unless the structural
  impossibility check fires.
- Mixed or structurally impossible token combinations fail closed as
  data-quality failures — for example a dash total against a strictly
  positive numeric ToSTNeT — **unless** the frozen era schema explicitly
  marks that segment as out-of-era or not applicable, in which case the
  segment is absent by preregistration and contributes no proof.
- A blank required in-era cell remains a data-quality failure.

No cell may be repaired, imputed, defaulted, cross-filled from an adjacent
date or segment, or reconstructed from a total-minus-others identity.

## 6. Frozen validation relation and sentinels

Evaluated only after SOURCE_A, SOURCE_B, and SOURCE_C have each passed every
binding, schema, unit, interval, and data-quality check above:

```text
EXPECTED_UNPROVEN_SET={"2020-10-01"}

scheduled_open_dates - proven_auction_active_dates
    == EXPECTED_UNPROVEN_SET

proven_auction_active_dates - scheduled_open_dates
    == empty set
```

The mandatory neighbor sentinels are:

```text
2020-09-30 PROVEN_AUCTION_ACTIVE=true
2020-10-02 PROVEN_AUCTION_ACTIVE=true
```

`2020-10-01` has no SOURCE_B activity sentinel, by design. SOURCE_B is
required only to leave it **unproven**; the assertion that no auction
execution occurred that day comes solely from SOURCE_C. Restating
`2020-10-01` as "proven inactive by SOURCE_B" is exactly the invalid
inference HIGH_1 removes.

The frozen cross-source consistency requirement is:

```text
(scheduled_open_dates - proven_auction_active_dates)
    == source_c_confirmed_exception_set
    == {"2020-10-01"}
```

Any mismatch of the set relation, either neighbor sentinel, or the
cross-source consistency requirement is terminal:

```text
ACTUAL_TRADING_DAY_AUTHORITY_FAILURE
```

### 6.1 Deferred final calendar rule

Final actual trading dates may later be formed as:

```text
actual_trading_dates = scheduled_open_dates - source_c_confirmed_exception_set
```

**only if** SOURCE_B proves every other scheduled-open date
`PROVEN_AUCTION_ACTIVE` and every frozen relation and sentinel above passes.
The subtraction is authority-driven, never a manual edit: no date is deleted
by hand, hard-coded, or injected into the calendar pipeline.

### 6.2 Prohibitions on the adjudication

- No manual deletion of `2020-10-01` from any set or artifact.
- No hard-coded correction in the produced calendar.
- No post-hoc override, exception widening, or expected-set edit after
  observation.
- No treatment of `NOT_PROVEN` as inactive to force the relation to pass.
- No alternate source substitution after seeing results.
- No retry, refetch, redraw, or reparse-until-PASS.
- A data-quality, parser, era-schema, or relation failure never authorizes a
  new network attempt; it is terminal for V9_014 under the failure-class
  discipline frozen in `AI_RESEARCH_EXECUTION_RULES.md` §6.1.

## 7. Era and object binding (HIGH_2 remediation)

Era handling is explicit and preregistered. It is never inferred post hoc,
never selected by whichever schema happens to parse, and never chosen after
observing the relation result.

### 7.1 Frozen required segments

```text
PRE  (through 2022-04-01):
  1st Section
  2nd Section
  Mothers
  JASDAQ Standard
  JASDAQ Growth

POST (from 2022-04-04):
  Prime
  Standard
  Growth
```

TOKYO PRO Market is **not** a required SOURCE_B proof segment. It contributes
no proof, its cells are not required, and its absence or dash content is not
a data-quality failure.

Each daily row binds to its era by the row's own date. The implementation
must assert that the observed table structure matches the preregistered era
schema for that row and fail closed on any era/schema mismatch, rather than
adapting the parser to the observed table.

### 7.2 Era boundaries

```text
ERA_PRE   = daily rows dated 2017-01-01 .. 2022-04-01
ERA_POST  = daily rows dated 2022-04-04 .. 2026-01-31
```

The 2022 market restructure boundary falls between `2022-04-01` and
`2022-04-04`; `2022-04-02` and `2022-04-03` are non-business days.

### 7.3 The 2022-04 two-part source bundle

The 2022-04 logical month is a **two-part source bundle**, not one object:

```text
PRE  part: the official "(Reference) Status on April 1, 2022" object,
           covering 2022-04-01 on the pre-restructure segment schema
POST part: the normal Report 2 monthly object,
           covering 2022-04-04 onward on the post-restructure segment schema
```

It must **not** be claimed that one physical Report 2 object covers April 1.
Both parts are mandatory; a missing part is a feasibility/data-quality
failure for the whole study.

### 7.4 Archive locator discipline

Object resolution is semantic and fail-closed, following the already-frozen
V9_006 official monthly-archive traversal discipline:

```text
official Monthly Statistics archive root
  -> required year
  -> Report 2 "Stock Trading Volume & Value"
  -> required month
```

with the special official `(Reference) Status on April 1, 2022` branch used
for the PRE April object. Objects are **PDF only**. Each required object must
resolve to a unique same-domain object. Archive numbering must never be
hardcoded and no URL may be guessed, pattern-derived, or reconstructed from
memory. Ambiguous or non-unique resolution is a fail-closed locator failure.

### 7.5 Required unit expectations

Every required segment column has an exactly preregistered expected unit,
expressed in the canonical semantic units of Section 5.2.

`ERA_PRE` required segments (through `2022-04-01`):

```text
1st Section       total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES

2nd Section       total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES

Mothers           total   = THOUSAND_SHARES
                  ToSTNeT: 2017-01 .. 2019-12      = SHARES
                           2020-01 .. 2022-04-01   = THOUSAND_SHARES

JASDAQ Standard   total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES

JASDAQ Growth     total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES
```

`ERA_POST` required segments (from `2022-04-04`):

```text
Prime             total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES

Standard          total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES

Growth            total   = THOUSAND_SHARES
                  ToSTNeT = THOUSAND_SHARES
```

The declared unit must be parsed from each object for every required column
and must match the exact expected unit above. Any mismatch is a data-quality
failure.

There is no first-observed-unit learning. A required column's unit is never
established by observed intra-era constancy, never inferred from the data,
never re-derived from numeric magnitude, and never inherited from a previous
month or object. The preregistered expectation above is the only authority,
and the parsed declaration is checked against it rather than defining it. An
unexpected unit or layout change is a data-quality failure, never an accepted
variation.

## 8. Staged gates and V9_009 binding

V9_014 does not resolve `V9_009_HIGH_2`. That finding remains:

```text
V9_009_HIGH_2=OPEN_REQUIRES_HISTORICAL_JPX_CALENDAR_BINDING
```

until every one of the following has completed, in this exact order:

```text
1. V9_014 design PASS                      (GPT exact-SHA review)
2. V9_014 implementation PASS              (GPT exact-SHA review)
3. authorized real acquisition / execution (fresh human point-of-use gate;
                                            includes the protected SOURCE_A
                                            preserved-state read)
4. exact frozen validation PASS            (Section 6, offline)
5. canonical artifact review PASS          (GPT exact-SHA review)
```

Final `trading_dates` may be materialized **only** in a later reviewed
implementation/execution stage after all frozen validations pass. No stage may
be skipped, reordered, merged, or satisfied by a partial result. A PASS at any
earlier stage confers no calendar authority and no V9_009 consumption right.
Until stage 5 completes, V9_009 must not read, consume, or bind any V9_014
output, and must not substitute any other calendar.

## 9. Open-item resolution

All five previously open items are now closed by GPT methodology decision.

- **OPEN_1 — CLOSED.** Required domestic-stock segments are frozen in
  Section 7.1 for both eras; TOKYO PRO Market is excluded as a proof segment.
- **OPEN_2 — CLOSED.** SOURCE_A must reuse the preserved immutable V9_012
  SOURCE_A chain
  `aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45` with
  `page_count=1`. No fresh SOURCE_A acquisition. A later protected read
  requires fresh human authorization (Section 4.1).
- **OPEN_3 — CLOSED.** The SOURCE_B locator, the PDF-only object format, the
  2022-04 two-part bundle, and the `(Reference) Status on April 1, 2022`
  branch are frozen in Sections 7.3 and 7.4.
- **OPEN_4 — CLOSED.** The canonical share-unit token-to-multiplier mapping
  is frozen in Section 5.2, and the exact per-column required unit
  expectations, including the Mothers ToSTNeT boundary at 2020-01, are frozen
  in Section 7.5.
- **OPEN_5 — CLOSED.** A scheduled-open date missing entirely from the
  required in-era date/table coverage is a **data-quality failure**. It is
  never treated as inactive, never as an accepted exception, and never as
  zero.

No item remains `CHATGPT_DECISION_REQUIRED` in this design.

## 10. Non-actions and execution boundary

This design task performed and authorized none of the following:

```text
implementation_or_code_or_test_change=false
jpx_network_requests=0
jquants_network_requests=0
yahoo_or_external_data_network_requests=0
protected_or_private_state_reads=0
api_key_reads=0
raw_payload_reads=0
t0_cache_outcome_backtest_model_execution=false
profitability_evaluation=false
trading_dates_materialized=false
human_gates_consumed=0
```

Git fetch and push to `origin` were the only network operations performed by
this task.

The source roles, byte-provenance binding, coverage, prohibited sources,
reported-value interval semantics, per-segment and date-level activity
adjudication, token semantics, era and object binding, frozen relation,
sentinels, cross-source consistency requirement, deferred final calendar
rule, failure class, prohibitions, and staged gates above are frozen for
V9_014 subject to GPT exact-SHA review. No implementation-time source
substitution, semantic weakening, threshold relaxation, or methodology
invention is permitted. Where an item is not covered by this design, the
correct behavior is to stop and report `CHATGPT_DECISION_REQUIRED`, never to
guess.

GPT-5.6 Sol remains the final methodology authority and exact-SHA reviewer.
