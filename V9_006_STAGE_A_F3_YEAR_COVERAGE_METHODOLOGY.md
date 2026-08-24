# V9_006 Stage-A F3 YEAR coverage methodology

```text
task=V9_006_STAGE_A_F3_YEAR_COVERAGE_METHODOLOGY_BINDING
status=AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

## 1. F3 root support object

`SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE` (F3) uses the exact existing
`DELISTED_COMPANY_ROOT_URL` and `YEAR` slot kind. The F3 root response is a
`RAW_PROVENANCE_OBJECT` support object, not a `COVERAGE_EVIDENCE_OBJECT`.

Its canonical raw-provenance owner is F3, its applicable period is exactly
`DELISTED_COMPANY_DISCOVERY_ROOT`, and its requested URL is exactly
`DELISTED_COMPANY_ROOT_URL`. It is fetched and locked at most once per
execution; traversal and parser repair always use those locked bytes.

## 2. YEAR evidence-object identity

Required years are 2017 through 2025 inclusive. For a requested four-digit
year `YYYY`, traversal uses only the locked F3 root bytes and its official
archive-year selector. Exactly one same-domain link with an exact semantic
year label `YYYY` is required. Relative links resolve against the locked
root's final `resolved_url`. Zero or multiple candidates fail closed; there
is no archive-N guess, provider substitution, search, mirror, or manual
choice.

The selected object is F3's `YEAR` `COVERAGE_EVIDENCE_OBJECT`, owned by F3,
with applicable period exactly `YYYY` and requested URL equal to the exact
unique mechanically resolved URL. Its `source_object_slot_id` is the existing
raw-lock key.

## 3. Complete-year coverage proof

The locked root's unique official archive-selector binding to `YYYY` is the
structural complete-calendar-year source contract. One selected YEAR object
proves coverage for all twelve F3 base cells only when all of the following
hold:

- the root raw lock is valid;
- exactly one selector candidate was resolved for `YYYY`;
- the YEAR object was successfully first-complete-payload locked and its raw
  provenance validates; and
- its metadata has `source_family=SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE`,
  `applicable_period=YYYY`, and the exact selected requested URL.

This proves source coverage, not the presence of a delisting in every month;
zero rows/events in a covered month are not `MISSING`. If any condition fails,
the entire year fails closed: all twelve F3 cells remain `MISSING`, with no
partial fan-out and no twelve-fold refetch.

## 4. Fan-out

For a proven `YYYY`, the exact same YEAR slot ID is referenced by each
`YYYY-01` through `YYYY-12` F3 matrix cell. No additional slot ID or monthly
raw-lock identity is created. All nine proven 2017--2025 YEAR objects fan out
to exactly 108 F3 base cells.

## 5. Scope preserved

F3 remains official corroborating delisting evidence; the existing
conflicting-authoritative-object rule remains unchanged, with no manual
reconciliation. Detailed delisting event parsing is not defined or
implemented here and remains distinct from coverage proof.

This binding changes none of F1/F2/F4/F5/F6/F7, the 648-cell matrix, F2
bridge, F7 envelope, retry/redirect/raw-provenance/semantic policy, periods,
thresholds, human gates, or design freeze. No real network is authorized.

```text
REVIEWED_SHA=edb71bb969f694b29e63c395ad16bae65d7311f1
PARENT_SHA=ae62c1dd1e5aa7753a03a765fc40dcfb6e7adc6f
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F2_F4_ENUMERATION=PASS
```
