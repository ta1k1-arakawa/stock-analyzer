# V9_006 Stage-A F6 root structure probe implementation contract

```text
task=V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT
status=AWAITING_GPT_REVIEW
network_authorized_by_this_task=false
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

## Purpose and authority

This document binds the exact implementation semantics for the future
root-structure diagnostic probe whose purpose and authority were fixed by
`V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md`. It is docs-only:
it authorizes no code change, no network request, and no production
Stage-A action. A future implementation task must bind exactly the
semantics below; it may not substitute methodology discretion for any rule
in this contract.

## 1. Diagnostic raw identity

The probe reuses the existing F6 source family,
`SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE`, and the existing bound
`TOPIX_ROOT_URL` as its only requested URL. Its raw lock uses a dedicated
diagnostic applicable period, `F6_ROOT_STRUCTURE_DIAGNOSTIC`, distinct from
the existing production F6 support/evidence periods
(`TOPIX_DISCOVERY_ROOT`, `TOPIX_GLOBAL_2017_2025`).

This diagnostic raw identity is explicitly separate from production F6
support/evidence identity. It is not a coverage slot, cannot populate the
648-cell matrix, and cannot be substituted for the existing
`TOPIX_DISCOVERY_ROOT` support object or any F6 `COVERAGE_EVIDENCE_OBJECT`.

Future execution must persist this diagnostic raw lock and its derived
artifact only under a dedicated diagnostic output root, separate from the
production Stage-A output tree. No future step may copy, alias, or
otherwise promote a diagnostic lock or artifact into production Stage-A
output.

## 2. Label normalization

The target exact label is `Historical Index Value`
(`F6_SEMANTIC_SECTION_LABEL`).

Normalization, applied identically to the candidate element's complete
descendant text before comparison:

- resolve HTML character references;
- collapse every run of Unicode whitespace to a single ASCII space;
- strip leading and trailing whitespace.

Comparison after normalization remains case-sensitive. No casefolding,
punctuation rewriting, fuzzy matching, or substring matching is permitted.

An exact-label occurrence is the leaf-most element whose normalized
complete descendant text equals the exact target label: the element
itself matches the normalized label, and no descendant element also
matches the normalized label. This leaf-most rule exists to prevent an
ancestor wrapper from being double-counted merely because it contains the
same text as its matching descendant, while still allowing inline markup
(for example a `<span>` or `<em>` split within the label) inside the
matching element.

Every such occurrence must be recorded. The probe never selects or ranks a
"correct" candidate among multiple occurrences.

## 3. DOM path

For every recorded occurrence, the artifact records the root-to-element
path. Each path component contains:

- the lowercase tag name;
- the zero-based index of the element among ELEMENT-type siblings sharing
  the same parent (text nodes are not indexed and never appear in the
  path);
- the `id` attribute value if present; and
- the element's normalized class tokens, sorted lexicographically, with
  duplicates removed.

Given the same locked raw bytes, this path derivation must be fully
deterministic: repeated execution against the same locked payload must
produce a byte-identical JSON artifact.

## 4. Anchor recording

The probe never follows or resolves any `href`. For every recorded
occurrence, it records, without following any of them:

- the self anchor, if the occurrence element itself is an `<a>`;
- every immediate child `<a>` element of the occurrence element;
- every immediate child `<a>` element of the occurrence element's
  immediate parent; and
- every immediate child `<a>` element of the occurrence element's
  immediate following element sibling.

For each recorded anchor, the artifact records:

- its normalized visible text, using the same whitespace normalization as
  §2;
- its exact raw `href` attribute string, or `null` if the attribute is
  absent; and
- its deterministic DOM path, derived the same way as §3.

The probe never records any numerical TOPIX value.

## 5. Result classification

The probe's structural outcome is exactly one of:

- `STRUCTURE_CAPTURED`: exactly one exact-label occurrence exists and
  extraction completed deterministically.
- `STRUCTURE_AMBIGUOUS`: zero or more than one exact-label occurrence
  exists and extraction completed deterministically.
- `STRUCTURE_EXTRACTION_FAILED`: parser/DOM extraction cannot complete
  deterministically.

These are diagnostic-only outcomes. They must never be mapped to, or
treated as equivalent to, F6 `AVAILABLE` or `MISSING`.

## 6. Fail-closed / raw lock

Future implementation must first-complete-payload raw-lock the root
before any parsing or semantic inspection, consistent with the design's
existing raw-lock contract.

If parsing fails:

- the diagnostic raw lock must be preserved;
- a safe failure artifact is written only if it can be produced
  deterministically;
- the executor must never refetch after a complete payload has been
  locked; and
- no child request may be made as a repair or retry step.

This contract creates no network authorization by itself; any future
execution still requires its own fresh, explicit, one-shot human
authorization at the point of use, exactly as fixed by
`V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md`.

## 7. No other change

This contract changes nothing else. In particular it does not alter:

- the F6 GLOBAL coverage methodology;
- the existing retry/redirect policy;
- production F6 support/evidence identity;
- F1, F2, F3, F4, F5, or F7;
- Stage-A authority or any authority flag;
- the existing deferred LOW finding; or
- the existing design freeze status.

```text
REVIEWED_SHA=ad3b31ec55afb530a7d970ca5eddafa080af965a
PARENT_SHA=61a9d7f79ceac0b967d2a89469c734edce982ee7
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN=PASS
```
