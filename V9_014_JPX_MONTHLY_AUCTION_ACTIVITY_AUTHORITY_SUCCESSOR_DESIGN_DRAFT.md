# V9_014 JPX Monthly Auction-Activity Authority Successor Design Draft

## 1. Status and study identity

```text
study_id=V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
design_status=AWAITING_GPT_REVIEW
document_status=DESIGN_DRAFT_ONLY
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
Python execution, no JPX / J-Quants / Yahoo / external-data network request,
no protected or private state read, no API-key read, no raw-payload
acquisition or reuse, no materialization, no T0 / cache / outcome / backtest
/ model execution, and no profitability evaluation. It is a design draft
awaiting GPT-5.6 Sol exact-SHA review.

## 2. Why a successor study is required

V9_012 froze TOPIX daily index bars as its actual-activity evidence source
and failed terminally with `ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`. V9_013
diagnosed that failure without repairing it and established, as recorded
public-safe evidence, that both V9_012 sources were schema- and
data-quality-valid (`SOURCE_A=A_VALID`, `SOURCE_B=B_VALID`) and that the
failure was located in the frozen relation/sentinel layer
(`V9_013_DIAGNOSTIC_CLASS=RELATION_OR_SENTINEL_FAILURE`), with
`V9_013_FAILURE_CAUSE=TOPIX_ACTIVITY_EVIDENCE_DOES_NOT_EXCLUDE_FROZEN_2020_10_01_FULL_DAY_TSE_OUTAGE`.

### 2.1 Mandatory interpretation of the V9_013 sentinel value

The recorded V9_013 diagnostic value is:

```text
V9_013_SENTINEL_2020_10_01_INACTIVE=false
```

`false` means the frozen "inactive" sentinel check **FAILED**. It must not be
described, summarized, or reused as if `2020-10-01` had been classified
inactive by TOPIX activity evidence. The already-recorded companion values
`V9_013_LEFT_DIFF_COUNT=0` and
`V9_013_MISSING_EXPECTED_EXCEPTION_COUNT=1` are consistent with that reading:
`2020-10-01` was not present in the observed left set difference, so the
frozen `EXPECTED_EXCEPTION_SET` was not reproduced by that source.

This clarification is additive. It corrects no historical record, changes no
frozen V9_012 or V9_013 result, and creates no new evidential capacity.

### 2.2 Consequence for source selection

TOPIX / index OHLC activity cannot serve as actual TSE cash-equity
auction-activity evidence and is therefore **prohibited** as a V9_014
source. V9_014 replaces the failed activity source with official JPX/TSE
monthly auction/ToSTNeT trading-volume evidence, keeping the scheduled
business-day superset role unchanged.

## 3. Frozen coverage and authority boundary

```text
coverage_start=2017-01-01
coverage_end=2026-01-31
```

The required inclusive coverage is exactly `2017-01-01..2026-01-31`. The
required SOURCE_B monthly-object count over that coverage is exactly `109`
consecutive calendar months `2017-01 .. 2026-01`, consistent with the
already-recorded repository value
`V9_010_CALENDAR_REQUIRED_SOURCE_MONTH_COUNT=109`. A missing, unresolved,
duplicated, or ambiguous required month is a feasibility/data-quality
failure. It is never permission to shorten coverage, interpolate a month,
substitute a provider, or accept a partial calendar.

V9_014 has `evidence_role=INPUT_BINDING_ONLY`. It can produce, at most, a
calendar input binding. It produces no signal, no model, no outcome, no
backtest, and no profitability evidence. Its
`profitability_evidential_capacity` is `ZERO` at every stage, including
after a full PASS.

## 4. Independent official source roles

Two independent official sources with separate provenance are required.
Neither source is authoritative alone for actual trading dates.

### 4.1 SOURCE_A — scheduled TSE business-day superset

```text
role=SCHEDULED_TSE_BUSINESS_DAY_SUPERSET
provider=OFFICIAL_JQUANTS_MARKETS_CALENDAR
endpoint=https://api.jquants.com/v2/markets/calendar
base_query={"from":"2017-01-01","to":"2026-01-31"}
hol_div=omitted
scheduled_open_predicate=HolDiv in {"1","2"}
```

SOURCE_A keeps exactly the semantic role it had in V9_012. Its frozen
scheduled-open interpretation is inherited unchanged and must not be
modified, relaxed, widened, or reinterpreted by V9_014:

- the payload root must be a JSON object with a list-valued `data` field;
- every row must be an object carrying required `Date` and `HolDiv` fields;
- `Date` must be a strict `YYYY-MM-DD` string, a valid ISO date, and within
  the frozen inclusive coverage;
- `HolDiv` must be exactly one of `{"0","1","2","3"}`;
- dates must be unique, and the SOURCE_A date set must equal every calendar
  date in the frozen coverage;
- `scheduled_open_dates` is exactly the set of `Date` values whose `HolDiv`
  is `"1"` or `"2"`.

SOURCE_A remains a scheduled superset. It never defines actual trading dates
by itself, and `hol_div` remains intentionally absent from the base query,
not null.

### 4.2 SOURCE_B — actual TSE regular-auction activity evidence

```text
role=ACTUAL_TSE_REGULAR_AUCTION_ACTIVITY_DATE_EVIDENCE
provider=OFFICIAL_JPX_TSE_MONTHLY_STATISTICS_REPORT_ARCHIVE
archive_root=https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html
report=Report 2 "Stock Trading Volume & Value"
table=Trading Volume & Value (Daily)
```

The archive root above is the same official monthly-statistics root already
established in this repository's frozen V9_006 source-slot locator
methodology. Monthly-object resolution must follow that already-frozen
traversal discipline: discover the year page only through the official
archive selector from the root, then the semantic report/row label, then the
requested month, then the unique same-domain linked object. Archive
numbering must never be hardcoded, and no month may be resolved by guessing
a URL pattern.

SOURCE_B supplies, per date and per required domestic-stock market segment,
the total Trading Volume and the "of which ToSTNeT" Trading Volume, from
which regular-auction activity is derived in Section 6.

### 4.3 Prohibited sources and substitutions

The following are prohibited for V9_014 at every stage:

- TOPIX or any other index OHLC series as activity evidence;
- total stock trading volume alone as an activity criterion, because
  `2020-10-01` had no auction executions while ToSTNeT executions existed,
  so a total-volume-only rule cannot separate the two;
- generic weekday ranges, `pd.bdate_range`, national-holiday subtraction,
  OS/locale calendars, Yahoo observed dates, broker calendars, or any
  vendor-derived calendar;
- any alternate provider, endpoint, index, ticker, report, or table
  substituted after results are observed.

## 5. Parsing, unit declaration, and exact integer normalization

For every date and every required domestic-stock market segment, the
implementation must:

1. parse the total Trading Volume cell;
2. parse the corresponding "of which ToSTNeT" Trading Volume cell;
3. parse the **declared unit** explicitly from the source object, per report
   and per table, never assumed and never inherited across objects;
4. normalize both parsed values exactly to **integer shares** using that
   declared unit before any arithmetic.

Normalization is exact-integer only. Binary floating point must not appear
anywhere on the parsing, normalization, comparison, or aggregation path:
no `float()`, no float dtype, no float-backed numeric container, and no
float intermediate. Exact integer arithmetic (arbitrary-precision integers,
or exact decimal arithmetic converted to an exact integer) is the only
permitted mechanism. If a declared unit and a parsed literal do not yield an
exact integer share count, that is a data-quality failure, never a rounding,
truncation, or nearest-value decision.

An absent, unparseable, ambiguous, or multiply-declared unit is a
data-quality failure. There is no default unit.

Per date and per required segment, the frozen value checks are:

```text
require total >= 0
require tostnet >= 0
require tostnet <= total
auction_volume = total - tostnet
```

Any violation is a data-quality failure.

### 5.1 Non-silent cell semantics

Blank, dash, placeholder, malformed, or ambiguous cells are **NOT** silently
treated as zero. Each is a data-quality failure, **unless** the frozen era
schema for that exact object explicitly marks that segment as nonexistent or
not applicable for that era, in which case the segment contributes nothing
and its absence is recorded as an explicit era-schema fact rather than as an
observed zero.

No cell may be repaired, imputed, defaulted, cross-filled from an adjacent
date or segment, or reconstructed from a total-minus-others identity.

## 6. Frozen auction-activity rule

For each date `d` in the frozen coverage, define the exact sum over the
frozen required domestic-stock market segments for the era applicable to `d`:

```text
auction_volume_total(d) = SUM over required segments of
                          (total(d, segment) - tostnet(d, segment))
```

computed entirely in exact integer arithmetic. Then:

```text
d is ACTUAL_AUCTION_ACTIVE  iff  auction_volume_total(d) > 0
d is inactive               iff  auction_volume_total(d) == 0
```

`actual_auction_active_dates` is the set of dates satisfying the first
condition. There is no third state: a date whose required-segment data is
not completely and validly parsed is a data-quality failure and stops the
study, rather than being classified active or inactive.

## 7. Era handling

Era handling is explicit and preregistered. It is never inferred post hoc,
never selected by whichever schema happens to parse, and never chosen after
observing the relation result.

```text
ERA_LEGACY            = daily rows dated 2017-01-01 .. 2022-03-31
ERA_RESTRUCTURE_2022_04 = the 2022-04 monthly object, which spans both schemas
ERA_POST_RESTRUCTURE  = daily rows dated 2022-04-04 .. 2026-01-31
```

The 2022-04 market restructure is an **intra-month** boundary and must be
handled as a special transition case, not as a whole-month era assignment:

```text
2022-04-01 uses the pre-restructure market-segment schema
2022-04-04 onward uses the post-restructure Prime / Standard / Growth schema
```

The post-restructure schema applies to every later month through the end of
coverage. The implementation must bind each daily row to its era by the
row's own date, must assert that the observed table structure matches the
preregistered era schema for that row, and must fail closed on any
era/schema mismatch rather than adapting the parser to the observed table.

### 7.1 Declared-unit / layout change

The official 2020 unit and layout change affecting the ToSTNeT fields must
be explicitly accounted for. The frozen rule is that the **declared unit
drives normalization**: the implementation reads and records the declared
unit for every monthly object, report, table, and required segment column,
and normalizes with that declared unit only. It must additionally record
where the declared unit or layout changes across the monthly sequence and
assert that observed change points match the preregistered era boundaries.
A unit or layout change that is observed but not preregistered is a
data-quality failure, not an accepted variation.

### 7.2 Required segment enumeration — OPEN

The design must enumerate the exact required domestic-stock market-segment
columns for each era **before** implementation begins. The repository facts
currently available, and the public facts available to this design task
without any external-data network request, are not sufficient to enumerate
those exact column names. Column names must not be invented, guessed,
translated, or reconstructed from memory.

```text
CHATGPT_DECISION_REQUIRED
V9_014_ERA_SEGMENT_ENUMERATION=OPEN
```

See Section 10 for the full open-item list. Implementation may not begin
while this item is OPEN, because the auction-activity rule in Section 6 is
defined over exactly that frozen segment set.

## 8. Frozen validation relation and sentinels

The following relation is frozen **before** any V9_014 acquisition and is
evaluated only after SOURCE_A and SOURCE_B have both passed every schema,
unit, normalization, and data-quality check above:

```text
EXPECTED_EXCEPTION_SET={"2020-10-01"}

scheduled_open_dates - actual_auction_active_dates
    == EXPECTED_EXCEPTION_SET

actual_auction_active_dates - scheduled_open_dates
    == empty set
```

The mandatory sentinels are:

```text
2020-09-30 active=true
2020-10-01 active=false
2020-10-02 active=true
```

Any mismatch of the set relation or of any sentinel is terminal:

```text
ACTUAL_TRADING_DAY_AUTHORITY_FAILURE
```

This relation captures the exceptional exchange-wide closure directly: the
official scheduled superset lists `2020-10-01`, while independent official
auction-activity evidence must show zero regular-auction volume on that date
even though ToSTNeT executions existed.

### 8.1 Prohibitions on the adjudication

- No manual deletion of `2020-10-01` from any set or artifact.
- No hard-coded correction in the produced calendar.
- No post-hoc override, exception widening, or expected-set edit after
  observation.
- No alternate source substitution after seeing results.
- No retry, refetch, redraw, reparse-until-PASS, or repeated acquisition
  attempt aimed at obtaining a passing relation.
- A data-quality, parser, era-schema, or relation failure never authorizes a
  new network attempt; it is terminal for V9_014 under the failure-class
  discipline already frozen in `AI_RESEARCH_EXECUTION_RULES.md` §6.1.

## 9. Staged gates and V9_009 binding

V9_014 does not resolve `V9_009_HIGH_2`. That finding remains:

```text
V9_009_HIGH_2=OPEN_REQUIRES_HISTORICAL_JPX_CALENDAR_BINDING
```

until every one of the following has completed, in this exact order:

```text
1. V9_014 design PASS                      (GPT exact-SHA review)
2. V9_014 implementation PASS              (GPT exact-SHA review)
3. authorized real acquisition / execution (fresh human point-of-use gate)
4. exact frozen validation PASS            (Section 8, offline)
5. canonical artifact review PASS          (GPT exact-SHA review)
```

Final `trading_dates` may be materialized **only** in a later reviewed
implementation/execution stage after all frozen validations pass. No stage
of V9_014 may be skipped, reordered, merged, or satisfied by a partial
result. A PASS at any earlier stage confers no calendar authority and no
V9_009 consumption right.

Until stage 5 completes, V9_009 must not read, consume, or bind any V9_014
output, and must not substitute any other calendar.

## 10. Open items requiring ChatGPT decision

```text
CHATGPT_DECISION_REQUIRED
```

The following are OPEN. They are methodological or scope decisions reserved
to the GPT methodology authority under `AI_RESEARCH_EXECUTION_RULES.md` §1.2
and §2, and this execution agent must not resolve them. Implementation may
not begin while OPEN_1 stands, and each remaining item must be closed or
explicitly deferred by GPT before the stage it governs.

- **OPEN_1 — `V9_014_ERA_SEGMENT_ENUMERATION`.** The exact required
  domestic-stock market-segment column names for `ERA_LEGACY`,
  `ERA_RESTRUCTURE_2022_04` (both the `2022-04-01` pre-restructure and the
  `2022-04-04` onward post-restructure sides), and `ERA_POST_RESTRUCTURE`
  are not enumerable from repository or currently available public facts
  without an external-data network request. They must be supplied exactly.
  No column name is invented here.

- **OPEN_2 — SOURCE_A byte provenance.** Whether V9_014 SOURCE_A binds the
  preserved immutable V9_012 locked `/markets/calendar` chain
  (`V9_012_LAST_SOURCE_A_CHAIN_SHA256=aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45`,
  page count `1`) as protected durable state, or performs a fresh
  independently authorized acquisition under a new non-colliding V9_014
  root, is unspecified. This determines whether a protected-read gate, a
  network gate, or both apply, so it is not an execution-agent choice.

- **OPEN_3 — SOURCE_B object locator and format binding.** The exact frozen
  archive traversal labels, per-era object file formats, and per-era table
  locators for Report 2 "Stock Trading Volume & Value" / "Trading Volume &
  Value (Daily)" are not established in this repository. The V9_006
  traversal discipline is inherited, but the exact frozen label strings and
  format handling per era must be supplied before implementation.

- **OPEN_4 — declared-unit token set and 2020 change boundary.** The exact
  declared-unit tokens and the exact effective boundary of the official 2020
  unit/layout change affecting the ToSTNeT fields are not established here.
  The "declared unit drives normalization" rule in Section 7.1 is frozen,
  but the preregistered change points against which observed changes are
  asserted must be supplied.

- **OPEN_5 — missing daily row semantics.** The treatment of a scheduled-open
  date within coverage that has no daily row at all in a required segment's
  table is unspecified by the frozen rule set. Section 6 defines active and
  inactive only over completely parsed required-segment data. Until GPT
  closes this item, the implementation must fail closed on that case under
  `AI_RESEARCH_EXECUTION_RULES.md` §7; it must not silently treat a missing
  row as zero, as inactive, or as an accepted exception.

## 11. Non-actions and execution boundary

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

The provider roles, coverage, prohibited sources, parsing and exact-integer
normalization contract, auction-activity rule, era-handling discipline,
frozen relation, sentinels, failure class, prohibitions, and staged gates
above are frozen for V9_014 subject to GPT exact-SHA review. No
implementation-time source substitution, semantic weakening, threshold
relaxation, or methodology invention is permitted. Where this design leaves
an item OPEN, the correct behavior is to stop and report
`CHATGPT_DECISION_REQUIRED`, never to guess.

GPT-5.6 Sol remains the final methodology authority and exact-SHA reviewer.
