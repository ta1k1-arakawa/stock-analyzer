# V9_006 Stage-A F6 GLOBAL coverage methodology

```text
task=V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY_BINDING
status=PASS
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

F6 is `SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE`, with `GLOBAL` slot kind,
`TOPIX_ROOT_URL`, and semantic section `F6_SEMANTIC_SECTION_LABEL` (Historical
Index Value). Required coverage years are 2017--2025.

The locked TOPIX root is a support `RAW_PROVENANCE_OBJECT`, owned by F6 with
`applicable_period=TOPIX_DISCOVERY_ROOT` and `requested_url=TOPIX_ROOT_URL`.
It is fetched once per raw key and reused for traversal/parser repair.

From those locked bytes, exactly one official same-domain object under the
bound semantic section must resolve; relative links use the root final
`resolved_url`. There is no search, guessed URL/filename, alternate provider,
mirror, manual choice, or reroll. The selected object is the single F6 GLOBAL
`COVERAGE_EVIDENCE_OBJECT`, owned by F6 with
`applicable_period=TOPIX_GLOBAL_2017_2025`, exact resolved requested URL, and
the existing raw-lock key as its slot ID.

Coverage is structural and independent of index values. A required year is
covered only if the locked GLOBAL object mechanically establishes historical
observations for that exact year. A future parser must derive the exact covered
year set from locked bytes, reject malformed/ambiguous date/year structure,
and never infer from row position/count, neighboring years, first/last dates,
continuity, interpolation, or index values. If the real format is not frozen
enough for deterministic parsing, execution stops `CHATGPT_DECISION_REQUIRED`.

The one GLOBAL slot ID may fan out to all twelve months of each structurally
proven whole year. Unproven years retain twelve `MISSING` cells; when all nine
years are proven, one ID fans out to all 108 F6 cells. No monthly GLOBAL
objects or refetches are created.

No F6 implementation or network access is authorized, and no F1/F2/F3/F4/F5/
F7, matrix, bridge, envelope, provenance, retry, redirect, semantic, threshold,
authority, or freeze rule changes.

## F6 methodology review/adjudication

```text
REVIEWED_SHA=0993a26c43e65c07a718b7559b971c4218759136
PARENT_SHA=65bc62c79ed3757654f68e9c5556af45907c764c
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY=PASS
```

```text
REVIEWED_SHA=65bc62c79ed3757654f68e9c5556af45907c764c
PARENT_SHA=7682fe67d20f7ad8028df3fca82d82f85b686bc3
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F7_ACQUISITION=PASS
```
