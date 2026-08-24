# V9_006 Stage-A F6 root structure probe offline implementation review

```text
task=V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_OFFLINE_IMPLEMENTATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

This implements only the OFFLINE parsing seam for the F6 root-structure
diagnostic bound by
`V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md`. It
performs no network request, real or synthetic, and creates no network
authorization.

## What was added (`src/v9_005_stage_a_jpx_probe.py`)

- `F6_ROOT_STRUCTURE_DIAGNOSTIC = "F6_ROOT_STRUCTURE_DIAGNOSTIC"`, a
  dedicated raw-lock `applicable_period`, distinct from the existing
  production F6 `TOPIX_DISCOVERY_ROOT`/`TOPIX_GLOBAL_2017_2025` periods.
- `read_f6_root_structure_diagnostic_lock(output_root)`: reads ONLY the
  already-existing raw lock via the existing `read_locked_payload(
  output_root, SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
  F6_ROOT_STRUCTURE_DIAGNOSTIC, TOPIX_ROOT_URL)`. It accepts no
  fetcher/sleep/clock and calls no network/fetch/retry/
  `ensure_locked_payload` function. Absent, corrupt, or wrong-identity
  locks fail closed (raise `V9005StageABlocked`); `read_locked_payload`'s
  own existing schema/hash/URL/timestamp validation is reused unchanged.
- `parse_f6_root_structure_probe(locked)`: pure, deterministic. Decodes the
  locked raw bytes strict-UTF-8 (an optional UTF-8 BOM is stripped first;
  no fallback encoding), then parses with a new generic, full-DOM,
  stack-validated `HTMLParser` subclass (`_F6RootStructureHtmlParser`) that
  fails closed on any mismatched/unclosed tag or nested `<a>` (mirroring
  the existing `_MonthlyStatisticsHtmlParser`'s reviewed nested-anchor
  rejection). Label matching normalizes via `html.unescape` + Unicode
  whitespace collapse + trim, stays case-sensitive, and selects only
  leaf-most exact occurrences (an element matches only if no descendant
  element also matches). Each occurrence records a root-to-element DOM
  path (lowercase tag, zero-based index among ELEMENT siblings only,
  `id`, sorted-unique normalized class tokens) and the four anchor
  relation categories (self/children/parent's children/following
  sibling's children), each anchor's `href` read from the parser's exact
  raw start-tag source text (never the auto-entity-decoded attribute
  value) so entity spelling is preserved byte-for-byte and never resolved;
  an unparseable/duplicate raw `href` fails extraction rather than
  guessing. Outcomes are exactly `STRUCTURE_CAPTURED` (exactly one
  occurrence), `STRUCTURE_AMBIGUOUS` (zero or multiple), or
  `STRUCTURE_EXTRACTION_FAILED` (decode or DOM failure, with a stable
  non-secret `failure_reason` token and `occurrences=[]`).
- `write_f6_root_structure_probe_artifact(output_root, artifact)`: writes
  `V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT.json` under the same
  `output_root` using the existing `canonical_bytes`/`_atomic_create`
  utilities. First write is an atomic create; a pre-existing byte-identical
  artifact is reused; a differing one fails closed. Never overwrites.
- `run_f6_root_structure_probe_offline(output_root)`: composes the three
  functions above. Its only parameter is `output_root`; it never accepts a
  fetcher/sleep/clock.

Nothing here calls `build_source_inventory`, so the diagnostic raw lock and
artifact cannot populate F6's `INVENTORY_AVAILABLE`/`MONTHLY_COVERAGE_
MATRIX`; `F6_ROOT_STRUCTURE_DIAGNOSTIC` is also not a valid inventory month,
so `build_source_inventory` itself rejects it if ever mis-wired as a
coverage reference (`test_f6_root_structure_diagnostic_slot_cannot_populate_
f6_inventory`). `ACQUISITION_IMPLEMENTATION_COMPLETE` is untouched
(`False`); `run_stage_a`, the locator/retry/redirect/inventory/semantic
machinery, and F1-F5/F7 are untouched.

## Tests (`tests/test_v9_005_stage_a_jpx_probe.py`)

A new offline-only section (all fixtures are synthetic already-locked
bytes; no test performs or requires network access) proves: a single exact
occurrence captures; inline markup within the label still matches; a
matching ancestor is excluded by the leaf-most rule when a descendant also
matches; zero and multiple occurrences are both `STRUCTURE_AMBIGUOUS`;
whitespace-run and entity-reference normalization is exact and comparison
stays case-sensitive; DOM sibling indices ignore text nodes and classes are
sorted/deduplicated; all four anchor relation categories resolve exactly;
an anchor `self` case; raw `href` is preserved source-exact (entity
spelling included) and never resolved against the requested/resolved URL;
unrelated numerical/page text never appears in the artifact; reprocessing
the same locked bytes is byte-identical and never overwrites, while a
differing artifact for the same path fails closed; a missing or corrupted
diagnostic lock fails closed; malformed DOM (mismatched/unclosed tags,
nested `<a>`) fails closed deterministically and reproducibly;
invalid UTF-8 fails closed with no fallback while a UTF-8 BOM is accepted
and stripped; an ambiguous raw `href` fails extraction; no
network/fetch/retry/`ensure_locked_payload` function is ever invoked and no
offline entry point accepts a fetcher/sleep/clock; the diagnostic slot
cannot populate F6 inventory; and `ACQUISITION_IMPLEMENTATION_COMPLETE`
remains `False`.

`pytest tests/test_v9_005_stage_a_jpx_probe.py -q`: 193 passed (full
existing targeted suite plus the new section), `SOURCE_DATA_NETWORK_
REQUESTS=0`. `git diff --check`: clean.

## Exact GPT review preceding this implementation

```text
REVIEWED_SHA=83469514b4111fceb25983345f92590121b759f6
PARENT_SHA=ad3b31ec55afb530a7d970ca5eddafa080af965a
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT=PASS
```

## What this implementation does not do

No real network request was made or authorized. No production Stage-A
output, F1-F5/F7, retry/redirect policy, F6 GLOBAL coverage methodology, or
design-freeze status was touched. `ACQUISITION_IMPLEMENTATION_COMPLETE`
remains `False`; `V9_006_STAGE_A_IMPLEMENTATION` remains `BLOCK`. This does
not itself authorize any future real acquisition of the diagnostic raw
payload -- that still requires its own fresh, explicit, one-shot human
authorization at the point of use, exactly as fixed by
`V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md`.

## MEDIUM_1 review and remediation

```text
REVIEWED_SHA=e76ddcca1a153634676bb81ab143c054c09e1079
CRITICAL=0
HIGH=0
MEDIUM=1
RESULT=BLOCK
FINDING=V9_006_F6_ROOT_OFFLINE_MEDIUM_1_DOUBLE_HTML_ENTITY_DECODE
```

`V9_006_F6_ROOT_OFFLINE_MEDIUM_1_DOUBLE_HTML_ENTITY_DECODE`:
`_F6RootStructureHtmlParser` uses `HTMLParser(convert_charrefs=True)`, so
`handle_data` text (and therefore every element's/anchor's raw descendant
text) already has HTML character references resolved exactly once before
`_f6_normalize_text` ever sees it. The reviewed `_f6_normalize_text` called
`html.unescape()` again on that already-decoded text, a second, recursive
decode pass. For source text such as `Historical&amp;#32;Index Value`, the
parser's one real decode (`&amp;` -> `&`) correctly yields the literal
`Historical&#32;Index Value`; the reviewed code's extra `html.unescape()`
pass then wrongly decoded the remaining literal `&#32;` into an actual
space, producing `Historical Index Value` and falsely matching text the
source never actually rendered as the label. The same double-decode
affected anchor visible text.

**Remediation implemented this task:** `_f6_normalize_text` no longer calls
`html.unescape`. It only collapses Unicode whitespace runs to one ASCII
space and trims -- exactly the whitespace normalization the contract
requires beyond the parser's own single upstream entity-resolution pass.
Comparison remains case-sensitive; no other normalization, DOM-path, anchor
category, raw-`href`, UTF-8, classification, or diagnostic-identity logic
was touched. The now-unused `import html` was removed. See
`_f6_normalize_text`'s updated docstring in
`src/v9_005_stage_a_jpx_probe.py` for the exact rationale.

New tests in `tests/test_v9_005_stage_a_jpx_probe.py` prove: a real
`&nbsp;` entity still single-decodes to a matching space
(`STRUCTURE_CAPTURED`, unchanged); `Historical&amp;#32;Index Value` and
`Historical&amp;nbsp;Index Value` (only `&amp;` is a real entity; the
remainder is literal text after one decode) both correctly produce
`STRUCTURE_AMBIGUOUS` with `label_occurrence_count=0`
(`test_f6_root_structure_whitespace_entity_normalization_is_exact_and_case_sensitive`);
and anchor visible text `A&amp;nbsp;B` is recorded literally as `"A&nbsp;B"`,
never `"A B"`
(`test_f6_root_structure_anchor_visible_text_is_decoded_exactly_once`). The
existing raw-`href` source-exactness test and the full existing F6 offline
suite remain `PASS`: `pytest tests/test_v9_005_stage_a_jpx_probe.py -q` ->
196 passed (193 existing + 3 new); `git diff --check` clean;
`SOURCE_DATA_NETWORK_REQUESTS=0`.

`V9_006_F6_ROOT_OFFLINE_MEDIUM_1_DOUBLE_HTML_ENTITY_DECODE=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`
`V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_OFFLINE_IMPLEMENTATION=BLOCK`
