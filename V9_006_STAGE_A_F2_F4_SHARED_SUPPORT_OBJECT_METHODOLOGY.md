# V9_006 Stage-A F2/F4 shared support-object methodology

```text
task=V9_006_STAGE_A_F2_F4_SHARED_SUPPORT_OBJECT_METHODOLOGY_BINDING
status=AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

This is a docs-only methodology binding for a future, separately reviewed
acquisition implementation. It does not authorize network access or implement
any fetch, raw lock, parser, or coverage update.

## 1. Shared root support object

F2 and F4 share the exact official root `MONTHLY_STATISTICS_ROOT_URL`. Its
response is a `RAW_PROVENANCE_OBJECT`, not a `COVERAGE_EVIDENCE_OBJECT`.

Its canonical raw-provenance owner is
`SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT`, with applicable period
exactly `MONTHLY_STATISTICS_DISCOVERY_ROOT`. F4 MUST reuse those same locked
bytes and MUST NOT fetch or separately raw-lock a duplicate root under its own
source-family key. This ownership is provenance/reuse bookkeeping only; it
does not make the root F2 coverage evidence.

## 2. Shared year-page support object

For every required calendar year `YYYY`, the unique year-page URL resolved
from the locked shared root is fetched and locked at most once. Its canonical
raw-provenance owner is
`SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT`, with applicable period
exactly `MONTHLY_STATISTICS_DISCOVERY_YEAR_YYYY`, where `YYYY` is the exact
four-digit selected year. Its requested URL is the exact unique year-page URL
obtained from locked-root traversal.

Both F2 and F4 traversal reuse those same locked year-page bytes. F4 must not
perform another fetch or raw lock for that page. A shared year page is a
`RAW_PROVENANCE_OBJECT` support object, not coverage evidence.

## 3. F2/F4 monthly evidence objects

After traversal of a locked shared year page, each F2 child uses source family
`SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT`; each F4 child uses source
family `SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE`. For every child:

- `applicable_period` is the exact requested evidence month `YYYY-MM`.
- `requested_url` is the exact unique child URL resolved from the locked
  year-page bytes.
- The child is independently fetched and first-complete-payload locked.

These child locks are `COVERAGE_EVIDENCE_OBJECTS`. Their coverage
`source_object_slot_id` is exactly the existing raw-lock key. Only child
evidence-object IDs may populate F2/F4 `MONTHLY_COVERAGE_MATRIX` cells;
shared-root and shared-year support IDs must never populate coverage cells.

## 4. F2 bridge

Post-2025 F2 bridge months use the same F2 child rule with
`applicable_period=YYYY-MM`. They are mandatory evidence slots outside the
648-cell base matrix. This binding neither authorizes nor enumerates a bridge
from observed terminal month `T`; the existing `f2_bridge_months(T)` rule is
unchanged. F4 has no analogous post-2025 bridge.

## 5. Reuse and failure

The shared root is fetched at most once per execution; each shared year page
is fetched at most once per year per execution. Existing locked bytes are
always reused, including parser/traversal repairs. Zero or multiple traversal
candidates remain `FAIL`/`MISSING`; there is no alternate URL, provider,
archive-N, or manual choice. A child object is not fetched until its unique URL
is resolved from already-locked support bytes. Existing retry, redirect, and
raw-provenance rules are unchanged.

## 6. No other methodology change

This does not change F2/F4 roots or semantic labels, base evaluation months,
the 648-cell model, F2 bridge definition, F1/F3/F5/F6/F7, retry policy,
semantic validation, human gates, thresholds/periods, or design freeze.
`ACQUISITION_IMPLEMENTATION_COMPLETE` remains `false`; no real network is
authorized.

## Exact GPT review preceding this binding

```text
REVIEWED_SHA=04455334511f49ec8f8029d2a07022d78d8b758f
PARENT_SHA=e03b959b149852c50a17576a204592d2a3ddb51f
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F2_F4_TRAVERSAL=PASS
```

## Next action

`GPT_EXACT_SHA_V9_006_STAGE_A_F2_F4_SHARED_SUPPORT_OBJECT_METHODOLOGY_REVIEW`.
