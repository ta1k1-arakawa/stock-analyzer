# V9_006 Stage-A acquisition object model methodology

```text
task=V9_006_STAGE_A_ACQUISITION_OBJECT_MODEL_METHODOLOGY_BINDING
status=AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

This is a docs-only methodology binding. It defines the object model a future,
separately reviewed F2-F7 acquisition/parser implementation must follow. It
does not authorize, implement, or execute acquisition.

## 1. Two object domains

`RAW_PROVENANCE_OBJECT` means every network response whose bytes are consumed
for discovery, traversal, parsing, semantic derivation, coverage, or evidence.
Every such object MUST be first-complete-payload locked and provenance verified
under the existing raw-lock contract.

`COVERAGE_EVIDENCE_OBJECT` means a raw-provenance object which directly
satisfies one required `SOURCE_OBJECT_INVENTORY` evidence slot and may be
referenced by `MONTHLY_COVERAGE_MATRIX`.

Discovery, root, index, or year-selector HTML used only to locate another
evidence object is a `RAW_PROVENANCE_OBJECT`, but is not solely for that reason
a coverage-evidence slot. This creates no new `SOURCE_OBJECT_INVENTORY`
`slot_kind`; the allowed evidence slot kinds remain exactly `MONTHLY`, `YEAR`,
`TERMINAL`, and `GLOBAL`.

For example, F1 discovery HTML is a raw-provenance support object, while F1
`data_j.xls` is a `TERMINAL` coverage-evidence object.

## 2. Source object slot ID

Every `COVERAGE_EVIDENCE_OBJECT` has a `source_object_slot_id`, equal to the
existing V9 raw-lock record key for that evidence object:

```text
SHA256(UTF8(
  "V9_005_STAGE_A_RAW_LOCK_KEY_V1\0"
  + source_family
  + "\0"
  + applicable_period
  + "\0"
  + requested_url
))
```

The ID is lowercase 64-hex. No second or unrelated identity scheme is allowed.
It identifies the exact requested evidence object and links the coverage model
to its immutable raw lock.

Discovery-support objects retain their own raw-lock keys, but are not referenced
by `MONTHLY_COVERAGE_MATRIX` unless they are themselves the methodology-bound
coverage evidence object for that family.

## 3. Monthly coverage references

Every F2-F7 base `MONTHLY_COVERAGE_MATRIX` record must include
`source_object_slot_ids`.

- `AVAILABLE` requires a non-empty, sorted, unique slot-ID list sufficient
  under the relevant family contract.
- `MISSING` requires an empty list.
- `NOT_APPLICABLE_BY_SOURCE_CONTRACT` requires an empty list.
- No arbitrary object or Python sentinel can constitute `AVAILABLE`.
- Duplicate slot IDs collapse deterministically.
- Invalid or non-64-hex IDs fail implementation validation; they are never
  silently dropped.

The family mapping is unchanged: F2 uses the exact monthly object; F3 may fan
out one `YEAR` object to its twelve months only after complete-year coverage is
mechanically proven; F4 uses the exact monthly object; F5 uses its monthly
object while `crosscheck_comparable` remains a separate boolean; F6 may fan out
one `GLOBAL` object only to structurally proven covered years; and F7 uses its
exact `YYYYMM` monthly page. F1 remains outside the base 648-cell matrix.

F2 post-2025 bridge and F7 envelope slots remain mandatory evidence slots
outside the base matrix and use the same slot-ID rule.

## 4. Discovery support reuse

A bound discovery/root object is fetched and locked at most once per exact raw
lock key in one execution. Every traversal or parser repair reuses those same
locked bytes. A discovery object may locate multiple required evidence objects;
that never authorizes refetching it once per month.

Zero or multiple candidate child links for a required slot remain
`FAIL`/`MISSING` under the existing methodology. There is no manual choice,
guessed URL, reroll, alternate provider, or archive-N.

## 5. No methodology changes

This binding does not change source families, roots or traversal labels, the F1
`TERMINAL_SEED` amendment, the 648 base cells, F2 bridge, F7 envelope, retry
policy, raw-provenance rules, semantic validation, thresholds or periods,
design freeze, or human gates. No real network or source-data read is
authorized. `ACQUISITION_IMPLEMENTATION_COMPLETE` remains `false`.

## Exact GPT review preceding this binding

```text
REVIEWED_SHA=243273a90b983f250301f973038b38862c0642da
PARENT_SHA=aca54748a1d838cbd3c4ad603fc91bb6624d7ae2
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_HIGH_4=RESOLVED
```

## Next action

`GPT_EXACT_SHA_V9_006_STAGE_A_ACQUISITION_OBJECT_MODEL_METHODOLOGY_REVIEW`.
