# V9_005 free-source public-network probe design draft

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=PREFREEZE_PROBE_DESIGN
probe_executed=false
network_authorized=false
price_values_reportable=false
returns_reportable=false
T1_membership_access=false
```

This document preregisters a public-network-only feasibility probe. It does
not authorize network access, data acquisition, T1 membership generation or
opening, model fitting, backtesting, profit calculation, or design freeze.

The probe has two separately authorized stages. Authorization for Stage A must
not authorize Stage B and cannot be reused.

## Stage A — free JPX metadata feasibility

Purpose: determine whether free official JPX material can support a
deterministic point-in-time security-master and corporate-action metadata
contract. The only allowed domain is `jpx.co.jp`; allowed information is
public metadata only. Individual-stock historical price/OHLCV requests are
prohibited.

The inventory must include these source families:

1. List of TSE-listed Issues / month-end listed-issue material.
2. Monthly Statistics Report: Changes in Listed Companies and Issues, Etc.
3. Delisted-company archive.
4. Ex-New / Ex-Rights / stock-split-ratio archive.
5. Monthly aggregate listed-issue counts, where available.
6. TOPIX Historical Index Value.
7. JPX Calendar monthly market-business-day / market-holiday material.

### Deterministic source inventory

Before any feasibility verdict, Stage A must create a deterministic
`SOURCE_INVENTORY`. For every calendar month from 2017-01 through 2025-12,
there must be one inventory record for every source family required by the
reconstruction contract. Each record is exactly one of:

```text
AVAILABLE
NOT_APPLICABLE_BY_SOURCE_CONTRACT
MISSING
```

`NOT_APPLICABLE_BY_SOURCE_CONTRACT` is permitted only when the official JPX
source family has a mechanically documented cadence/range proving that no file
is expected for that month. Unknown or ambiguous is `MISSING`. Any required
`MISSING` record is Stage-A FAIL; no alternate non-preregistered source may be
substituted after a missing result.

Target chronology is metadata sufficient to reconstruct V9 roles from
2017-01-01 through 2025-12-31, plus terminal/current metadata only where
required for exact reverse replay. Every Stage-A record must lock requested
URL, final resolved URL, HTTP status, byte length, SHA256 raw bytes, retrieval
timestamp, source category, and applicable month/year. No security
price/return field may be printed.

The calendar source family exists solely to define Stage-B expected market
dates. Stage-A execution must acquire and lock enough official JPX calendar
material to cover 2016-09-01 through the final possible HIGH-2/V9 exit-tail
date. Do not hard-code that endpoint. Derive it mechanically:

```text
FINAL_SIGNAL_D0=last_frozen_V9_signal_grid_D0_lte_2025-12-31
FINAL_PLANNED_D3=third_JPX_business_day_after(FINAL_SIGNAL_D0)
FINAL_POSSIBLE_EXIT_DAY=20th_JPX_business_day_exit_attempt_date(
  counting_FINAL_PLANNED_D3_as_attempt_day_1)
STAGE_B_GLOBAL_END_EXCLUSIVE=calendar_date_immediately_after(
  FINAL_POSSIBLE_EXIT_DAY)
```

The locked calendar material must derive this endpoint unambiguously. Any
unresolved required historical calendar-archive gap fails Stage A. No national
holiday library, pandas holiday calendar, exchange calendar, Yahoo timestamp,
or inferred weekday-only calendar may replace locked official JPX calendar
evidence.

For calendar date `d`, `JPX_BUSINESS_DAY(d)=true` iff the locked official JPX
calendar contract classifies TSE cash-equity auction trading as open on `d`.
Saturday and Sunday are false. If official material cannot classify a required
date unambiguously, Stage A fails. The deterministic inventory and provenance
rules apply to the calendar source family.

### Required Stage-A evidence

Stage A passes only if all of the following are satisfied:

1. `TERMINAL_SNAPSHOT`: a terminal/month-end security snapshot sufficient to
   seed deterministic backward/forward reconstruction exists and its raw bytes
   are locked.
2. `LISTING_TRANSITIONS`: every inventory month has a locked official record
   sufficient to identify all new listings and exact effective dates, or is
   mechanically valid `NOT_APPLICABLE_BY_SOURCE_CONTRACT`.
3. `DELISTING_TRANSITIONS`: the same rule holds for all delistings and exact
   effective dates.
4. `MARKET_TRANSITIONS`: all required market/segment transitions are
   representable with exact effective dates; any encountered transition class
   that is not representable fails Stage A.
5. `SECURITY_TYPE`: domestic ordinary-common-stock eligibility is determinable
   for every reconstructed identity/date needed by V9 without future security
   state; `UNKNOWN` fails Stage A.
6. `CANONICAL_IDENTITY`: define a canonical identity tuple from official
   metadata fields. Its exact serialization must be frozen in the Stage-A
   execution artifact before Stage B use. Any code-reuse case that cannot be
   disambiguated without future/outcome data fails Stage A.
7. `EFFECTIVE_DATE`: every state transition has one mechanically defined
   effective JPX date; an ambiguous date fails Stage A.
8. `RECONSTRUCTION`: beginning with the locked terminal snapshot and applying
   only locked official transition records deterministically produces the same
   security state on repeated runs.
9. `MONTH_END_CROSSCHECK`: for every month with a comparable official JPX
   aggregate listed-issue count, reconstructed count equals the official count
   after the exact same documented scope definition. A mismatch fails Stage A;
   do not manually reconcile. If unavailable, record
   `CROSSCHECK_NOT_AVAILABLE`; that alone neither passes nor fails, but the
   transition inventory must remain complete.
10. `RAW_PROVENANCE`: every consumed source object records requested URL,
    resolved URL, HTTP status, retrieval timestamp, byte count, SHA256 raw
    bytes, source family, and applicable period. Duplicate/conflicting objects
    for the same authoritative slot fail unless an official revision relation
    is mechanically established.

Stage A PASS iff:

```text
FREE_JPX_METADATA_PROBE_PASS=(
  required_inventory_missing_count == 0
  AND terminal_snapshot_pass == true
  AND listing_transition_pass == true
  AND delisting_transition_pass == true
  AND market_transition_pass == true
  AND security_type_pass == true
  AND canonical_identity_pass == true
  AND effective_date_pass == true
  AND trading_calendar_pass == true
  AND deterministic_reconstruction_pass == true
  AND comparable_month_end_mismatch_count == 0
  AND raw_provenance_pass == true)
FREE_JPX_METADATA_PROBE_FAIL=otherwise
failure_class=SOURCE_OR_DATA_FEASIBILITY_FAILURE
```

There is no weighted score, majority vote, manual override, favorable
substitution, or source-family redraw. Failure is not strategy failure, and a
Stage-A pass does not authorize Stage B.

## Stage B — Yahoo coverage probe

Stage B may occur only after Stage A completes, its artifact receives
independent GPT review, and a new explicit human network authorization is
given. Stage-A authorization cannot be reused.

Purpose: test Yahoo historical transport/date/schema feasibility without
observing or reporting price outcomes. The sample is public and independent of
T1 membership; the T1 partition must not be generated before or during this
probe. Build the sample from Stage-A public metadata before any Yahoo request.

| Stratum | Definition | Required count |
| --- | --- | --- |
| A_ACTIVE | Canonical identities listed at 2025-12-31 | 20 |
| B_DELISTED | Identities delisted in each calendar year 2017 through 2025 | 2 per year; up to 18 |
| C_LATE_LISTING | Identities first listed during 2021 through 2025 | 10 |
| D_CORPORATE_ACTION | Identities with an official JPX split/consolidation event during 2017 through 2025 | 10 |

For each stratum candidate:

```text
selection_key=SHA256(UTF8(
  "V9_FREE_SOURCE_PROBE_V1\0"
  + STRATUM
  + "\0"
  + canonical_security_identity))
sort=(selection_key_hex,canonical_security_identity)
take=first_required_N
random_library_used=false
reroll=false
```

Deduplicate identities across strata only after each stratum's selection. If a
stratum has fewer than its required identities,
`STAGE_B_SAMPLE_CONSTRUCTION_FAIL` and stop before Yahoo requests. There is no
replacement after Yahoo results. Public output stores only final sample-manifest
hash, count, and categories; ordinary report text must not print sample
identities where repository governance prefers safe aggregate provenance.

### Frozen request interval

For each preregistered canonical security identity `i`, Stage-A PIT
reconstruction must provide `listed_state(i,d)` for every required calendar
date. Define `REQUIRED_MARKET_DATES_i` as the ordered dates `d` satisfying all:

- `d >= 2016-09-01`;
- `d < STAGE_B_GLOBAL_END_EXCLUSIVE`;
- `listed_state(i,d) == true`; and
- `JPX_BUSINESS_DAY(d) == true`.

If `REQUIRED_MARKET_DATES_i` is empty, set
`STAGE_B_SAMPLE_CONSTRUCTION_FAIL` and stop before any Yahoo request. Define:

```text
REQUEST_START_i=first_date(REQUIRED_MARKET_DATES_i)
REQUEST_END_EXCLUSIVE_i=calendar_date_immediately_after(
  last_date(REQUIRED_MARKET_DATES_i))
```

Freeze these values in the Stage-B sample/request manifest before the first
Yahoo request. One identity receives exactly one request. No request-window
change after Yahoo results, retry, alternate suffix, or replacement is allowed.

Permitted observations/reporting are only: HTTP status; schema-valid boolean;
response-host-valid boolean; requested start/end; earliest and latest returned
trading date; valid and invalid row count; duplicate-date boolean; split-event
count/presence; raw payload byte count; raw payload SHA256; canonical row hash;
and canonical split-event hash. Do not print Open, High, Low, Close, AdjClose,
Volume, returns, gains/losses, model features, signals, or profits.

### Exact Yahoo returned-date coverage

For each response, define:

```text
RETURNED_DATE_SET_i=set(trading_date from
  valid_price_rows UNION invalid_price_rows)
MISSING_EXPECTED_DATES_i=REQUIRED_MARKET_DATES_i - RETURNED_DATE_SET_i
UNEXPECTED_RETURNED_DATES_i=RETURNED_DATE_SET_i - REQUIRED_MARKET_DATES_i
```

An invalid returned price row proves only that Yahoo returned a timestamp; it
does not become a valid market observation. Duplicate dates are an immediate
FAIL. Date-coverage PASS for identity `i` requires exactly:

```text
missing_expected_date_count_i == 0
AND unexpected_returned_date_count_i == 0
```

Do not infer coverage from earliest/latest dates, interpolate missing dates,
ignore an internal gap, or treat an absent expected JPX-business-day timestamp
as an exchange holiday unless the locked official JPX calendar says so. This
failure is `SOURCE_OR_DATA_FEASIBILITY_FAILURE`, not strategy failure. Ordinary
public output must not print identity-specific missing-date lists; aggregate
counts/hashes are permitted.

### Structural Yahoo schema PASS

For every preregistered identity:

```text
HTTP_PASS=(HTTP_status == 200)
HOST_PASS=(final_response_host == frozen_trusted_Yahoo_host)
SCHEMA_PASS=(
  exactly_one_chart_result
  AND expected_ticker_symbol_contract_pass
  AND timestamps_array_exists_and_is_nonempty
  AND raw_open_high_low_close_volume_arrays_structurally_exist
  AND adjusted_close_array_structurally_exists
  AND every_required_array_length_equals_timestamp_array_length
  AND duplicate_trading_dates == false)
HASH_PASS=all_required_raw_and_canonical_SHA256_values_produced
VALID_ROW_PASS=(valid_price_row_count >= 1)
```

Do not add an invalid-row percentage threshold here. Invalid-row treatment
remains governed by later frozen V9 data-quality rules; this probe tests source
transport/coverage feasibility.

### Exact corporate-action comparison

For D_CORPORATE_ACTION, Stage-A official JPX metadata is authoritative. Define
the pure split/consolidation ratio `R_e = post_action_shares / pre_action_shares`.
For each D_CORPORATE_ACTION identity, `JPX_CA_SET_i` contains all official
events with `REQUEST_START_i <= effective_date < REQUEST_END_EXCLUSIVE_i`. Each
event is the canonical tuple:

```text
(effective_date_iso,
 reduced_positive_ratio_numerator,
 reduced_positive_ratio_denominator)
```

The ratio is exactly post-action shares over pre-action shares. Normalize it as
an exact rational in lowest terms; decimal lexical values must be converted to
an exact rational. Binary-float tolerance is not authoritative. Normalize Yahoo
events to that same orientation. If Yahoo provides both numerator/denominator
and `splitRatio`, they must mechanically agree before canonicalization or Stage
B fails. `YAHOO_CA_SET_i` is the canonical multiset of Yahoo pure
split/consolidation events in the same request interval.

Corporate-action PASS for identity `i` requires:

```text
YAHOO_CA_SET_i == JPX_CA_SET_i
```

JPX missing-in-Yahoo events, Yahoo extras, date or ratio mismatch, orientation
ambiguity, unparseable ratio, conflicting Yahoo fields, and duplicate-event
ambiguity all fail. There is no plus/minus-one-day or ratio tolerance, manual
reconciliation, or event replacement. Dividends are outside this comparison.

### Exact Stage-B identity and overall verdicts

For each unique preregistered identity `i`:

```text
IDENTITY_PASS_i=(
  HTTP_PASS_i
  AND HOST_PASS_i
  AND SCHEMA_PASS_i
  AND HASH_PASS_i
  AND VALID_ROW_PASS_i
  AND missing_expected_date_count_i == 0
  AND unexpected_returned_date_count_i == 0
  AND (no_D_CORPORATE_ACTION_membership_i
       OR corporate_action_exact_match_i == true))
```

Preserve all original stratum memberships after cross-stratum deduplication. An
identity in multiple strata makes one request but retains every applicable
validation obligation. Let `N` be the number of unique preregistered identities
after deterministic selection/deduplication. Stage B PASS iff:

```text
sample_construction_pass == true
AND N > 0
AND passed_identity_count == N
AND failed_identity_count == 0
```

Otherwise `STAGE_B_FAIL`. There is no majority threshold, 95-percent threshold,
reroll, replacement, retry, suffix substitution, manual override, or favorable
source substitution. A Stage-B failure is source/data feasibility failure, not
profitability failure.

## Result boundaries and human gates

Even a two-stage pass means only:

```text
FREE_SOURCE_TRANSPORT_AND_METADATA_FEASIBILITY=SUPPORTED
```

It does not establish profitability, T1 PASS, Yahoo authority over JPX,
guaranteed future identity coverage, or design freeze. If either stage fails:

```text
FREE_SOURCE_STATUS=FREE_PATH_INSUFFICIENT_UNDER_CURRENT_CONTRACT
```

Do not automatically buy J-Quants; return to GPT methodology authority for a
source decision. `JQUANTS_PURCHASE_AUTHORIZED=false` remains unchanged.

```text
probe_design_gate_consumption=0
FRESH_EXPLICIT_PUBLIC_NETWORK_AUTH_REQUIRED=true
SEPARATE_FRESH_EXPLICIT_PUBLIC_NETWORK_AUTH_REQUIRED=true
stage_A_authorization_authorizes_stage_B=false
```

Neither future authorization permits J-Quants purchase, T1 opening, model
fitting, backtesting, or production trading.
