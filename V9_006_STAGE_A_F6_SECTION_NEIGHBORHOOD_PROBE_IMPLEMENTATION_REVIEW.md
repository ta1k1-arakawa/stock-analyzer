# V9_006 Stage-A F6 section neighborhood probe offline implementation review

```text
task=V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
global_child_selected=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

This implements only the OFFLINE section-neighborhood diagnostic bound by
section 4 (and the section-2.2 semantic-heading identity rule it depends
on) of `V9_006_STAGE_A_F6_ROOT_STRUCTURE_ADJUDICATION_AND_NEIGHBORHOOD_
PROBE_DESIGN.md`. It performs no network request, real or synthetic,
creates or modifies no raw lock, and selects, ranks, or binds no F6 GLOBAL
child URL.

## What was added (`src/v9_005_stage_a_jpx_probe.py`)

- A small pure refactor, `_f6_parse_full_dom(text)`, factored out of the
  existing `_f6_extract_label_occurrences` with no behavior change (same
  exception, same `_F6_MALFORMED_DOM_STRUCTURE` reason token, same
  existing tests still passing unchanged): both the root-structure
  label-occurrence extractor and the new section neighborhood probe now
  share exactly one HTML normalization/DOM-building methodology instead
  of a second, parallel one.
- `F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION`,
  `F6_SECTION_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME`,
  `F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME` (result file
  `V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT.json`), each
  distinct from the existing root-structure probe's own schema/diagnostic/
  filename constants.
- `NEIGHBORHOOD_CAPTURED`, `SEMANTIC_HEADING_AMBIGUOUS` (the third outcome,
  `STRUCTURE_EXTRACTION_FAILED`, is the existing constant reused verbatim);
  `NEIGHBORHOOD_RELATION_BEFORE_HEADING`, `_HEADING`, `_AFTER_HEADING`,
  `_INSIDE_HEADING`.
- `_f6_identify_semantic_heading(doc_order, occurrence_elements)`:
  implements exactly the seven-step deterministic rule bound in design
  section 2.2 -- exactly one leaf-most exact-label occurrence whose
  element is `<a>` with a raw `href` matching exactly `#X` (`X` non-empty;
  an ambiguous raw `href` on such a candidate fails closed via the same
  `_F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE` reason the existing anchor-reading
  primitive already uses elsewhere in this module, consistent with its
  existing fail-closed convention); exactly one DOM element anywhere in
  the document with `id == X`; that target's tag must be exactly `h2`;
  its normalized class tokens must contain `heading-title`; and it must
  contain (as itself or a descendant) exactly one leaf-most exact-label
  occurrence. Any zero/multiple/wrong-tag/wrong-class/inconsistent
  outcome returns `None` -- never a fallback, ranked, or guessed
  candidate. No literal fragment or `id` value, including `heading_14`,
  is ever hardcoded; every candidate is derived only from the locked
  payload's own DOM each run (proven by
  `test_f6_neighborhood_literal_heading_14_not_hardcoded`, whose fixture
  contains `id="heading_14"` on a decoy element under the wrong tag that
  must never be picked, while the real target uses an unrelated id).
- `_f6_neighborhood_children(parent, heading)`: every ELEMENT child of
  `parent`, in document order, each tagged `BEFORE_HEADING`, `HEADING`
  (exactly the heading's own sibling position), or `AFTER_HEADING`.
- `_f6_neighborhood_anchors_and_headings(parent, heading, doc_order,
  raw_text)`: a single document-order pass over every proper descendant of
  `parent`, splitting `<a>` elements (reusing the existing `_f6_anchor_of`
  primitive verbatim, so raw `href` stays byte-exact source spelling and
  is never resolved/followed, and an ambiguous raw `href` still fails
  closed) into the `anchors` list and `h1`-`h6` elements (normalized text
  via the existing `_f6_normalize_text`) into the `headings` list, each
  tagged `BEFORE_HEADING`/`INSIDE_HEADING`/`AFTER_HEADING` (anchors) or
  left untagged (headings, per the design's exact required field list)
  using document-order index comparison plus an explicit
  self-or-descendant-of-heading check for `INSIDE_HEADING`.
- `parse_f6_section_neighborhood_probe(locked)`: pure and deterministic;
  composes the above into the exact artifact schema section 4.2 of the
  design requires -- `schema_version`, `diagnostic`, `requested_url`,
  `resolved_url`, `byte_length`, `sha256`, `retrieval_timestamp_utc`
  (explicitly no `http_status`, matching the design's own provenance field
  list, which differs from the root-structure artifact's base fields),
  `status`, `failure_reason`, `semantic_heading`, `parent_container`,
  `children`, `anchors`, `headings`. No arbitrary page text, numerical
  TOPIX/index observation, raw payload bytes, resolved href, or chosen/
  ranked child URL is ever included (`test_f6_neighborhood_artifact_
  never_selects_or_binds_a_global_child` pins the exact key set).
- `write_f6_section_neighborhood_probe_artifact` /
  `run_f6_section_neighborhood_probe_offline`: mirror the existing
  root-structure probe's atomic-create/byte-identical-reuse/never-
  overwrite artifact-write discipline and single-`output_root`-parameter
  offline seam exactly, reusing `read_f6_root_structure_diagnostic_lock`
  (the existing, already-reviewed read-only reader of the existing
  `F6_ROOT_STRUCTURE_DIAGNOSTIC` raw lock) unchanged -- no new or modified
  raw lock, no fetcher/sleep/clock parameter, no
  network/fetch/retry/`ensure_locked_payload`/`lock_first_complete_
  payload` call anywhere in this seam.

Nothing here calls `run_stage_a` or `build_source_inventory`; the
diagnostic raw lock and this new artifact cannot populate F6's
`INVENTORY_AVAILABLE`/`MONTHLY_COVERAGE_MATRIX` for the same reason the
root-structure diagnostic already could not
(`test_f6_neighborhood_diagnostic_cannot_populate_f6_inventory`).
`ACQUISITION_IMPLEMENTATION_COMPLETE` is untouched (`False`).

## Tests (`tests/test_v9_005_stage_a_jpx_probe.py`)

A new offline-only section (every fixture is synthetic already-locked
bytes; no test performs or requires network access) proves, at minimum:
the observed nav-anchor/content-heading shape resolves the semantic
heading without depending on any literal id; the literal string
`heading_14` is never hardcoded (a same-tag, same-class decoy carrying
that exact id, but not the fragment target, is correctly ignored); zero
and multiple fragment-anchor candidates, a duplicate `id` target, a
wrong target tag, a missing `heading-title` class token, and zero/
multiple label descendants inside the target are all
`SEMANTIC_HEADING_AMBIGUOUS`; parent ELEMENT children and descendant
anchors both preserve document order and their exact
`BEFORE_HEADING`/`HEADING`/`AFTER_HEADING`/`INSIDE_HEADING` relation;
raw `href` stays source-exact and is never resolved against the
requested/resolved URL; an ambiguous duplicate raw `href` fails
`STRUCTURE_EXTRACTION_FAILED`; only `h1`-`h6` descendants are recorded,
with normalized text; unrelated page text and numeric/date TOPIX-like
values never reach the serialized artifact; reprocessing the same locked
bytes is byte-identical and never overwrites, while a differing artifact
for the same path fails closed; a missing, corrupted, or wrong-identity
(different `applicable_period`) diagnostic lock fails closed; invalid
UTF-8 and malformed DOM (mismatched/unclosed tags) both fail
`STRUCTURE_EXTRACTION_FAILED` deterministically; the offline seam never
calls `fetch_once_with_retry`, `ensure_locked_payload`,
`lock_first_complete_payload`, or `run_stage_a`, and none of its entry
points accept a `fetcher`/`sleep`/`clock`; the diagnostic slot cannot
populate F6 inventory; the artifact's exact key set never contains a
chosen/ranked child URL field; and `ACQUISITION_IMPLEMENTATION_COMPLETE`
remains `False`.

`PYTHONPATH=. pytest tests/test_v9_005_stage_a_jpx_probe.py -q`: 245
passed (217 existing + 28 new, including 3 parametrized malformed-DOM
cases from one new test function). `git diff --check`: clean.
`SOURCE_DATA_NETWORK_REQUESTS=0`.

## Exact GPT review preceding this implementation

```text
REVIEWED_SHA=5fc680513a64ee130184cf872d5d67ad0777a514
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F6_ROOT_STRUCTURE_ADJUDICATION_AND_NEIGHBORHOOD_PROBE_DESIGN=PASS
```

## What this implementation does not do

No real or synthetic network request was made. No raw lock was created or
modified; only the existing `F6_ROOT_STRUCTURE_DIAGNOSTIC` lock is read.
No F6 GLOBAL child URL was selected, ranked, or bound -- the artifact
schema has no such field. No production Stage-A output, F1-F5/F7,
retry/redirect policy, F6 GLOBAL coverage methodology, or design-freeze
status was touched. `ACQUISITION_IMPLEMENTATION_COMPLETE` remains
`False`; `V9_006_STAGE_A_IMPLEMENTATION` remains `BLOCK`. This does not
authorize any future real acquisition or any future production
child-locator use -- both still require their own fresh, explicit,
one-shot human authorization at their own point of use, per the governing
design documents.
