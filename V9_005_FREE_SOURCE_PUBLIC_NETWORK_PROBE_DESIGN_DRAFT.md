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

Target chronology is metadata sufficient to reconstruct V9 roles from
2017-01-01 through 2025-12-31, plus terminal/current metadata only where
required for exact reverse replay. Every Stage-A record must lock requested
URL, final resolved URL, HTTP status, byte length, SHA256 raw bytes, retrieval
timestamp, source category, and applicable month/year. No security
price/return field may be printed.

Stage A must answer whether a deterministic terminal snapshot exists; every
relevant listing addition and delisting can be represented; segment/market
transitions can be represented; domestic ordinary common-stock status can be
determined; code reuse can be disambiguated into canonical security identities;
effective dates are mechanically available; revisions can be detected and
locked; no months/files are missing; and reconstructed comparable month-end
counts can be cross-checked against official aggregate counts. Archive gaps
must not be silently filled.

```text
FREE_JPX_METADATA_PROBE_PASS=all_required_deterministic_source_contract_elements_demonstrated
FREE_JPX_METADATA_PROBE_FAIL=otherwise
failure_class=SOURCE_OR_DATA_FEASIBILITY_FAILURE
```

Failure is not strategy failure. No favorable substitution or manual gap
filling is allowed after results.

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
