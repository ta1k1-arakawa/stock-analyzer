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

For each preregistered identity, request the fixed interval mechanically
required by its public listing interval intersected with 2016-09-01 through the
mechanically required V9 end/tail horizon. Determine that exact interval before
its Yahoo request.

Permitted observations/reporting are only: HTTP status; schema-valid boolean;
response-host-valid boolean; requested start/end; earliest and latest returned
trading date; valid and invalid row count; duplicate-date boolean; split-event
count/presence; raw payload byte count; raw payload SHA256; canonical row hash;
and canonical split-event hash. Do not print Open, High, Low, Close, AdjClose,
Volume, returns, gains/losses, model features, signals, or profits.

No retry, replacement, alternate ticker suffix after failure, or manual
successful-source substitution is allowed.

### Stage-B PASS rule

Stage B passes only if every preregistered sample identity has HTTP 200,
correct Yahoo response host, expected valid schema, no duplicate trading dates,
nonempty valid rows, returned date span covering the mechanically required
listed/request interval subject only to documented exchange non-trading days,
structurally present raw/adjusted fields, and all required hashes. For every
D_CORPORATE_ACTION identity whose Stage-A official JPX metadata places a split
or consolidation in the request interval, Yahoo split-event metadata must
permit mechanical comparison with that official event. Any mismatch or missing
event is `STAGE_B_FAIL`.

There is no majority threshold, 95-percent threshold, or replacement of failed
identities.

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
