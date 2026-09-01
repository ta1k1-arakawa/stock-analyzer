# V9_013 V9_012 Authority-Failure Diagnostic Design

```text
study_id=V9_013_V9_012_AUTHORITY_FAILURE_DIAGNOSTIC
evidence_role=DIAGNOSTIC_ONLY
profitability_evidential_capacity=ZERO
calendar_authority_evidential_capacity=ZERO
V9_009_HIGH_2_resolution_capacity=ZERO
design_status=AWAITING_GPT_REVIEW
```

## 1. Purpose and non-authority

V9_013 exists only to identify which frozen V9_012 semantic invariant
produced `ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`. It is not a V9_012 retry,
replacement calendar, V9_009 authority, permission to repair V9_012, or
permission to change `EXPECTED_EXCEPTION_SET`. It is not profitability
evidence. V9_012 remains permanently `V9_012_RESULT=FAIL_TERMINAL`.

V9_013 may inform the design of a later successor study only. It cannot
authorize refetch, source substitution, manual correction, V9_009 calendar
consumption, T0, or any trading conclusion.

## 2. Sole input binding

The only permitted future diagnostic input is the preserved immutable V9_012
attempt-1 locked state, independently evidenced by these safe provenance
values:

```text
SOURCE_A_CHAIN_SHA256=aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45
SOURCE_B_CHAIN_SHA256=7b4c8624b78d51a30625672c411a76fcd85ab692765e99ee9cf6cc2239a3e33e
SOURCE_A_PAGE_COUNT=1
SOURCE_B_PAGE_COUNT=1
```

Before any semantic read, a future implementation must mechanically prove
that the locked source-chain SHA values exactly equal both frozen values and
that the page counts are exactly one. A mismatch is
`PRESERVED_V9_012_INPUT_BINDING_FAILURE` and stops the diagnostic. No locked
byte may be modified, deleted, overwritten, normalized, or reused through a
different acquisition root, source version, request, or state directory.

## 3. No network authority

V9_013 contains no acquisition, refetch, pagination, API-key, or network
path. Its frozen network facts are:

```text
JQUANTS_API_REQUESTS=0
API_KEY_READS=0
```

The diagnostic reads the preserved locked bytes only after the separately
reviewed one-shot protected-read workflow below. V9_011 authorization and
durable roots are not reusable.

## 4. Diagnostic semantic mirror

The future implementation must mirror V9_012 exactly, without relaxing,
correcting, extending, or substituting any check. It must not call V9_012
materialization as a retry. It must operate against synthetic fixtures during
development and use the preserved locked input only after the required
reviews and authorization.

### 4.1 SOURCE_A categories and semantics

The closed categories, in exact order, are:

```text
A_PAYLOAD_JSON_DECODE_FAILURE
A_PAYLOAD_ROOT_SCHEMA_FAILURE
A_DATA_FIELD_SCHEMA_FAILURE
A_ROW_SCHEMA_FAILURE
A_REQUIRED_FIELD_MISSING
A_DATE_TYPE_OR_FORMAT_INVALID
A_DATE_VALUE_INVALID
A_DATE_OUT_OF_COVERAGE
A_HOLDIV_TYPE_OR_DOMAIN_INVALID
A_DUPLICATE_DATE
A_COVERAGE_DATE_SET_MISMATCH
A_VALID
```

The mirror requires a JSON object root, a list-valued `data` field, object
rows, and required `Date` and `HolDiv` fields. `Date` must be strict
`YYYY-MM-DD`, a valid ISO date, and within the frozen inclusive coverage
`2017-01-01..2026-01-31`. `HolDiv` must be exactly one of `{"0","1","2","3"}`.
Dates must be unique, and the SOURCE_A date set must equal every calendar
date in the frozen coverage. A scheduled open date is exactly a row whose
`HolDiv` is `"1"` or `"2"`.

### 4.2 SOURCE_B categories and semantics

The closed categories, in exact order, are:

```text
B_PAYLOAD_JSON_DECODE_FAILURE
B_PAYLOAD_ROOT_SCHEMA_FAILURE
B_DATA_FIELD_SCHEMA_FAILURE
B_ROW_SCHEMA_FAILURE
B_REQUIRED_FIELD_MISSING
B_DATE_TYPE_OR_FORMAT_INVALID
B_DATE_VALUE_INVALID
B_DATE_OUT_OF_COVERAGE
B_DUPLICATE_DATE
B_OHLC_MIXED_NULL_FAILURE
B_OHLC_NONFINITE_OR_TYPE_FAILURE
B_VALID
```

The mirror requires `Date`, `O`, `H`, `L`, and `C`. `Date` uses the same
strict format, ISO validity, and frozen coverage checks as SOURCE_A. All four
OHLC fields null means inactive. A mixed null/non-null OHLC row fails. In all
other cases each OHLC value must be a finite real numeric value; booleans are
invalid. Dates must be unique. SOURCE_B has no full-calendar coverage
requirement. An active date is exactly a row with four valid, non-null,
finite real numeric OHLC values.

## 5. Deterministic first-failure location

For either source, pages are processed in ascending page index and rows in
source order. Envelope checks precede row checks; for each row the category
checks are applied in the exact order listed above. The first failure for
each source is reported. `page_index` is a positive page ordinal and
`row_index` is a one-based source-row ordinal; either is null when the
failure is not associated with that level.

Failure metadata is limited to:

```text
source_role
page_index
row_index
field_name
observed_json_type
```

`field_name` is null or comes from the closed source-specific field enum:
`root`, `data`, `Date`, `HolDiv` for SOURCE_A, and `root`, `data`, `Date`,
`O`, `H`, `L`, `C` for SOURCE_B. `observed_json_type` is null or exactly one
of `null`, `bool`, `int`, `float`, `string`, `list`, `object`. No raw value,
date, payload fragment, pagination key, private path, or credential is
exposed.

## 6. Relation and sentinel diagnostic

Relation evaluation occurs only when SOURCE_A is `A_VALID` and SOURCE_B is
`B_VALID`. Define:

```text
LEFT_DIFF=scheduled_open_dates - topix_active_dates
RIGHT_DIFF=topix_active_dates - scheduled_open_dates
EXPECTED_EXCEPTION_SET={"2020-10-01"}
```

The safe relation result contains only `left_diff_count`, `right_diff_count`,
`unexpected_left_diff_count`, `missing_expected_exception_count`,
`left_diff_sha256`, and `right_diff_sha256`, where each hash is SHA-256 over
`CANONICAL_JSON_NO_LF(sorted date list)` using the frozen deterministic JSON
encoding. It never exposes either diff list.

It also contains these booleans:

```text
left_exact_expected
right_empty
neighbor_2020_09_30_active
sentinel_2020_10_01_inactive
neighbor_2020_10_02_active
```

`left_exact_expected` means `LEFT_DIFF == EXPECTED_EXCEPTION_SET` and
`right_empty` means `RIGHT_DIFF` is empty. The three neighbor/sentinel
booleans are the only raw date labels permitted in the public diagnostic
contract, because they are preregistered in the frozen design.

## 7. Top-level diagnostic class

The closed top-level classes and precedence are:

```text
SOURCE_A_SEMANTIC_FAILURE
SOURCE_B_SEMANTIC_FAILURE
RELATION_OR_SENTINEL_FAILURE
NO_V9_012_FAILURE_REPRODUCED
```

If SOURCE_A is not `A_VALID`, the class is `SOURCE_A_SEMANTIC_FAILURE`. Else,
if SOURCE_B is not `B_VALID`, it is `SOURCE_B_SEMANTIC_FAILURE`. Else, if any
relation or sentinel condition fails, it is `RELATION_OR_SENTINEL_FAILURE`.
Otherwise it is `NO_V9_012_FAILURE_REPRODUCED`. The last class is not a
V9_012 pass; it is a diagnostic inconsistency requiring GPT adjudication.

## 8. Exact public diagnostic result

The frozen public result schema is
`V9_013_AUTHORITY_FAILURE_DIAGNOSTIC_RESULT_V1` and contains exactly these
fields, with no extras:

```text
schema_version
study_id
status
diagnostic_class
source_a_category
source_a_failure_location
source_b_category
source_b_failure_location
source_a_row_count
source_b_row_count
scheduled_open_count
topix_active_count
relation_evaluated
left_diff_count
right_diff_count
unexpected_left_diff_count
missing_expected_exception_count
left_diff_sha256
right_diff_sha256
left_exact_expected
right_empty
neighbor_2020_09_30_active
sentinel_2020_10_01_inactive
neighbor_2020_10_02_active
source_a_chain_sha256
source_b_chain_sha256
diagnostic_design_git_sha
diagnostic_implementation_git_sha
```

`source_a_failure_location` and `source_b_failure_location` are either null
or objects with exactly the five safe metadata fields in Section 5. Counts
are null when the corresponding source was not parsed sufficiently. Relation
fields are null when `relation_evaluated=false`. The chain hashes are the
two frozen values above. `diagnostic_design_git_sha` and
`diagnostic_implementation_git_sha` are exact lowercase 40-hex reviewed Git
SHAs. `status=COMPLETE` means only that diagnosis completed; it does not mean
calendar authority passed.

The result has no `trading_dates`, raw diff dates, raw payload, OHLC
magnitudes, private path, API key, pagination key, ticker identity, price, or
unrelated source data.

## 9. Protected one-shot execution workflow

### Phase A — metadata-only, no-network preflight

Phase A is strictly metadata-only with respect to the preserved V9_012 raw
payloads. Before fresh V9_013 protected-read authorization, Phase A must not
open or read any `raw_pages/*.bin` bytes, hash raw payload contents, parse raw
JSON, inspect a pagination envelope, inspect `Date`, `HolDiv`, or `O/H/L/C`,
call `read_locked_chain` or another helper that reads protected raw bytes, or
reconstruct a source-chain SHA from protected bytes.

Phase A may verify only safe, non-content preflight evidence:

- Git/branch/HEAD/remote/cleanliness;
- reviewed design and implementation blob bindings;
- expected preserved-state container existence;
- SOURCE_A and SOURCE_B directory existence;
- expected page/lock filename and count structure;
- filesystem file-size metadata without opening raw content;
- reviewed/safe page-lock JSON provenance fields already classified as safe
  metadata, without opening raw payload content;
- the previously recorded public evidence of one page for each source and
  the two frozen source-chain SHA values;
- durable-root non-ambiguity/existence without printing private paths.

Phase A does not prove that raw bytes generate the frozen source-chain SHA
values. That proof necessarily occurs after the protected-content boundary.
Phase A has no network and no protected semantic read. It requires GPT
adjudication before proceeding.

### Phase B — one protected byte-read and semantic invocation

After Phase A PASS, obtain fresh human authorization specifically covering
all V9_013 protected locked-payload byte reads required for immutable
lock/raw integrity verification, pagination/chain reconstruction, exact
source-chain SHA/page-count binding, and—only after those pass—Date/HolDiv/
OHLC diagnostic semantic inspection.

Invoke the reviewed diagnostic runner once through a minimal direct Windows
PowerShell command with no network and no API-key read, capturing stdout,
stderr, and exit status separately. Authorization is consumed at the first
protected raw-payload byte read, even if the diagnostic stops before semantic
inspection. The exact order is:

1. consume fresh V9_013 protected-read authorization at the first raw-byte
   read;
2. validate immutable raw/lock pairing and payload hashes;
3. validate pagination, terminal, and source-order provenance needed for the
   preserved chains;
4. reconstruct both source-chain SHA values;
5. require exactly the frozen one-page SOURCE_A and SOURCE_B chain SHA
   values:

   ```text
   SOURCE_A_CHAIN_SHA256=aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45
   SOURCE_B_CHAIN_SHA256=7b4c8624b78d51a30625672c411a76fcd85ab692765e99ee9cf6cc2239a3e33e
   SOURCE_A_PAGE_COUNT=1
   SOURCE_B_PAGE_COUNT=1
   ```
6. on any mismatch, return `PRESERVED_V9_012_INPUT_BINDING_FAILURE` and stop
   before Date/HolDiv/OHLC semantic inspection;
7. only after every binding check passes, begin the diagnostic semantic
   processing.

No new network authorization exists. A binding failure after the protected
byte boundary consumes the authorization and does not permit an automatic
rerun. V9_013 has `JQUANTS_API_REQUESTS=0` and `API_KEY_READS=0`.

### Phase C — no-network result inspection

Inspect only the safe result without network, retry, rerun, refetch, or
protected exploratory debugging. Preserve any diagnostic failure and return
it to GPT. Once the protected read occurs, no automatic rerun is permitted.

## 10. Development, prohibitions, and interpretation

Before protected data is read, implementation and targeted tests use
synthetic fixtures only. Tests must cover every closed category, safe
failure-location rule, relation count/hash/boolean, and top-level precedence.
No real V9_012 state or raw data may tune the diagnostic logic.

V9_013 prohibits V9_012 retry/refetch, alternate J-Quants endpoints,
alternate tickers or indexes, source substitution, manual date inspection,
manual correction, `EXPECTED_EXCEPTION_SET` changes, T0, V9_009 calendar
consumption, and profitability inference. It can only classify V9_012's
terminal failure. Regardless of its result,
`V9_009_HIGH_2=OPEN_REQUIRES_HISTORICAL_JPX_CALENDAR_BINDING` until a
separate successor authority study passes.

This design records no new network request, protected read, materialization,
or outcome. It is a draft awaiting GPT exact-SHA design review.
