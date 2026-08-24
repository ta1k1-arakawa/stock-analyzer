# V9_006 Stage-A F6 public root structure probe design

```text
task=V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN
status=AWAITING_GPT_REVIEW
network_authorized_by_this_task=false
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

## Purpose and authority

This is a design for a future, one-shot, human-authorized public diagnostic
probe. Its only possible network target is the exact bound `TOPIX_ROOT_URL`.
It exists solely to preserve enough locked TOPIX-root HTML structure for GPT to
later bind deterministic traversal under `F6_SEMANTIC_SECTION_LABEL`:
`Historical Index Value`.

This design does not authorize or consume a network authorization, authorize a
GLOBAL-child request, authorize F5 or another source, or authorize production
Stage-A execution. The future diagnostic needs a new explicit human
authorization at its point of use. That authorization is one-shot and cannot
be reused for production Stage A or for a later child-object probe.

## Future root-only diagnostic contract

The future executor may request exactly `TOPIX_ROOT_URL` and no other URL. It
uses the already-reviewed retry and redirect policy without an alternate URL,
mirror, search, provider, language variant, or fallback. It must not fetch or
follow a GLOBAL child object and must not inspect TOPIX numerical index values.

On the first complete payload, the executor first-complete-payload locks the
root before semantic inspection, preserving the actual requested URL, final
resolved URL, HTTP status, byte length, SHA256, and UTC retrieval timestamp.
Existing same-domain redirect rules remain applicable. The raw bytes are
durably retained in the diagnostic output root. Once a complete payload
exists, parser or extraction repair must reuse those locked bytes and must not
refetch.

## Deterministic safe structure artifact

Future execution produces
`V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT.json` with a schema/version,
diagnostic identity, requested URL, resolved URL, HTTP status, byte length,
SHA256, and retrieval timestamp. Its structural facts are derived only from
the locked raw bytes.

The artifact records the normalized exact-label occurrence count for
`Historical Index Value`. For every exact normalized-label occurrence, it
records deterministically:

- element tag, ID if present, and normalized class list if present;
- the ancestor chain of tag/ID/class to the document root;
- a deterministic DOM path/index representation;
- direct anchors on that element, if any, as normalized anchor text and raw
  href;
- immediate parent direct anchors, if any; and
- immediate following-sibling direct anchors, if any.

The probe records all exact-label occurrences and does not choose a candidate
as correct. It does not follow any href or print or record index values. If
malformed HTML prevents deterministic artifact production, extraction fails
closed while the already-locked payload remains available for later offline
repair.

## Classification and non-reuse

The diagnostic has only structural outcomes, such as `STRUCTURE_CAPTURED`,
`STRUCTURE_AMBIGUOUS`, or `STRUCTURE_EXTRACTION_FAILED`; it never returns F6
`AVAILABLE` or `MISSING`. Zero or multiple label occurrences are descriptive
evidence for later GPT methodology/traversal adjudication, not automatically a
Stage-A source failure. Execution AI cannot freeze a child URL.

The diagnostic raw object and artifact are explicitly diagnostic, not an F6
`COVERAGE_EVIDENCE_OBJECT`. They cannot populate the 648-cell matrix, cannot
be silently promoted into production Stage-A acquisition, and cannot set
`ACQUISITION_IMPLEMENTATION_COMPLETE=true`. Later offline repair may reuse the
same diagnostic locked bytes without another network request.

No source-family, slot, coverage, retry, redirect, provenance, semantic,
period, threshold, human-gate, or design-freeze methodology changes are bound
beyond this future diagnostic contract.

```text
REVIEWED_SHA=61a9d7f79ceac0b967d2a89469c734edce982ee7
PARENT_SHA=0993a26c43e65c07a718b7559b971c4218759136
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_F1_TERMINAL_REDIRECT_BASE=PASS
V9_006_STAGE_A_F6_GLOBAL_COVERAGE_METHODOLOGY=PASS
```
