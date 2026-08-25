# V9_006 Stage-A F6 root structure adjudication and neighborhood probe design

```text
task=V9_006_STAGE_A_F6_ROOT_STRUCTURE_ADJUDICATION_AND_NEIGHBORHOOD_PROBE_DESIGN
status=AWAITING_GPT_REVIEW
network_authorized_by_this_task=false
network_executed_by_this_task=false
global_child_requested=false
global_child_url_bound=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

This is docs-only. It performs no network request, requests or binds no F6
GLOBAL child URL, and authorizes no code, acquisition, or production
Stage-A action. It records the safe evidence of the already-authorized F6
root-structure diagnostic one-shot network execution fixed by
`V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md` and
`V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md`, binds
GPT's adjudication of that execution's two recorded label occurrences, and
designs a future offline-only neighborhood probe. No V9 design freeze,
historical-evaluation authority, private/sealed access authority,
J-Quants purchase authority, or live-trading authority is created or
changed.

## 1. Recorded one-shot execution evidence

The reviewed network-executor implementation (`V9_006_STAGE_A_F6_ROOT_
STRUCTURE_PROBE_NETWORK_EXECUTOR`, `RESULT=PASS`) was exercised exactly
once against the frozen `TOPIX_ROOT_URL` under its own fresh, one-shot
human authorization, exactly as fixed by `V9_006_STAGE_A_F6_PUBLIC_ROOT_
STRUCTURE_PROBE_DESIGN.md`. Only safe provenance -- hashes, counts,
booleans, the frozen public JPX URL, and the resulting diagnostic
classification -- is recorded here. No raw payload bytes, page text, or
index value is recorded.

```text
execution_sha=b9f6f52738c484f026f812da52b43f0175d4f857

requested_url=https://www.jpx.co.jp/english/markets/indices/topix/
resolved_url=same_as_requested_url
http_status=200
byte_length=62923

raw_payload_sha256=22a0d8e6ef139ebe8ed94287e49a9e24a1feb08fd00f0aa36eb07eb071754433
diagnostic_artifact_sha256=89c473abb09fc359f39222dde158ede7ab2a81b5f770c4fd9e44a630d26a5974

retrieval_timestamp_utc=2026-08-25T14:15:43Z
network_request_count=1

probe_status=STRUCTURE_AMBIGUOUS
label_occurrence_count=2

human_authorization_consumed=true
human_authorization_reusable=false
```

No GLOBAL child was fetched. No href was followed. No F5 or production
Stage-A execution occurred. This one-shot diagnostic authorization is
fully consumed and is not reusable for this or any future diagnostic or
production request; any future network step, including the future
neighborhood-probe design's own eventual production child-locator use,
requires its own fresh, explicit, one-shot human authorization at its own
point of use.

## 2. GPT adjudication of the two occurrences

`STRUCTURE_AMBIGUOUS` with `label_occurrence_count=2` is not itself an F6
`AVAILABLE`/`MISSING` outcome, consistent with the design's classification
rule. GPT has reviewed the two recorded occurrences and binds the
following observed relationship and mechanical rule.

### 2.1 Observed relationship (evidence, not a literal binding)

Occurrence A and Occurrence B are bound as a navigation-anchor /
content-heading counterpart pair, not competing source sections:

- Occurrence A: exact normalized label `Historical Index Value`; element
  `a`; raw `href="#heading_14"`; located under the tab-submenu navigation
  structure.
- Occurrence B: exact normalized label `Historical Index Value`;
  leaf-most occurrence element is `span`; its direct semantic heading
  ancestor is `h2`, `id=heading_14`, with a class token set containing
  `heading-title`.

The literal value `heading_14` is observed evidence from this one
execution only. It is never bound as a permanent literal production ID.
Production and future diagnostic traversal must derive the semantic
heading mechanically, from the fragment relationship below, every time --
never by matching a hardcoded `heading_14` string.

### 2.2 Deterministic semantic-heading identity rule (bound methodology)

This rule is now bound as the mechanical procedure a future execution
(diagnostic or production) must use to identify the F6 semantic heading
from a locked TOPIX root payload, given the existing leaf-most
exact-label occurrence set already produced by the reviewed offline F6
root-structure parser:

1. Take the existing leaf-most exact-label occurrence set for
   `Historical Index Value` (§2, `V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_
   IMPLEMENTATION_CONTRACT.md`).
2. From that set, find every occurrence whose element itself is `<a>` and
   whose raw `href` is fragment-only, matching exactly `#X` for some
   non-empty fragment `X` (no scheme, host, path, query, or additional
   `#`).
3. Require exactly one such navigation candidate. Zero or more than one
   candidate fails this rule.
4. Require exactly one DOM element in the locked payload with
   `id` attribute value exactly `X`. Zero or more than one match fails
   this rule.
5. Require that target element's tag is `h2` and that its normalized
   class token set contains the token `heading-title`. Any other tag or
   a missing `heading-title` token fails this rule.
6. Require that this `h2` element contains exactly one leaf-most
   exact-label occurrence for `Historical Index Value` among its
   descendants. Zero or more than one such descendant occurrence fails
   this rule.
7. Only when steps 2-6 each resolve to exactly one consistent match does
   this rule designate that `h2` element as the F6 semantic heading.

Any zero/multiple navigation-candidate outcome (step 3), zero/multiple
target-`id`-match outcome (step 4), wrong target tag or missing
`heading-title` class (step 5), or inconsistent label-descendant
relationship (step 6) fails this rule closed as
`CHATGPT_DECISION_REQUIRED` for methodology. It is never automatically
interpreted as F6 `MISSING`, and it never falls back to guessing,
ranking, or otherwise silently choosing among the ambiguous candidates.

Applied to the recorded evidence in §1/§2.1, this rule resolves
uniquely (one `<a href="#heading_14">` navigation candidate, one
`id=heading_14` target, that target is `h2` with `heading-title`, and it
contains exactly one leaf-most `Historical Index Value` occurrence), so:

```text
F6_SEMANTIC_HEADING_RELATIONSHIP=BOUND
```

## 3. Explicit limit: GLOBAL child locator remains blocked

The reviewed diagnostic artifact's self/children/parent-children/
immediate-following-sibling-children view of the semantic heading, as
captured by the existing anchor-recording rule (§4, `V9_006_STAGE_A_F6_
ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md`), contains no anchor
that this task may bind as the F6 GLOBAL child locator. No guessed
`href`, search engine, alternate JPX URL, or network refetch is used to
fill this gap.

```text
F6_GLOBAL_CHILD_LOCATOR=BLOCKED_PENDING_OFFLINE_NEIGHBORHOOD_EVIDENCE
```

No F6 GLOBAL child URL is bound or chosen by this task. Execution AI must
not decide which anchor, if any later found, is the GLOBAL child; that
remains a future GPT methodology/traversal adjudication over evidence a
future offline neighborhood probe produces.

## 4. Future offline-only neighborhood probe design

This section is a design for a future, offline-only, no-network
diagnostic. It authorizes no code, network request, or new raw lock by
this task. Its only permitted input is the raw bytes already preserved
under the existing `F6_ROOT_STRUCTURE_DIAGNOSTIC` raw lock (§1 of
`V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md`,
exercised in §1 above). It must not create or modify a raw lock, must not
accept a fetcher, sleep, or clock parameter, and must not perform any
network operation.

### 4.1 Diagnostic container candidate

A future implementation first locates the F6 semantic heading `h2` inside
the already-locked payload using the exact deterministic rule bound in
§2.2. It then defines the diagnostic container candidate as that
semantic heading's immediate parent element. This parent-element scope is
diagnostic only; it is not yet a production child-locator scope and does
not itself define, bind, or validate any GLOBAL child.

### 4.2 New artifact

Future execution produces a new, separate artifact,
`V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT.json`, distinct from
the existing `V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_RESULT.json`, using
its own dedicated schema/version identifier. It records, deterministically
from the same locked raw bytes:

- schema/version;
- the original raw payload's provenance, copied from the existing
  `F6_ROOT_STRUCTURE_DIAGNOSTIC` raw-lock metadata: `requested_url`,
  `resolved_url`, `byte_length`, `sha256`, and `retrieval_timestamp_utc`;
- the semantic heading: DOM path, tag, `id`, and normalized classes,
  derived using the exact existing DOM-path rule (§3, `V9_006_STAGE_A_F6_
  ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md`);
- the immediate parent container: DOM path, tag, `id`, and normalized
  classes, using the same DOM-path rule;
- every ELEMENT child of that parent, in document order, each with: DOM
  path; tag; `id`; classes; and its relation to the semantic heading,
  exactly one of `BEFORE_HEADING`, `HEADING`, or `AFTER_HEADING`;
- every descendant `<a>` element contained in that parent, in document
  order, each with: DOM path; normalized visible anchor text (same
  whitespace-run-collapse-and-trim normalization as the existing label
  rule, no `html.unescape` reapplication per the existing double-decode
  remediation); the exact raw `href` attribute string or `null` if
  absent; and whether it is `BEFORE_HEADING`, `INSIDE_HEADING`, or
  `AFTER_HEADING`, where `INSIDE_HEADING` means the anchor is a
  descendant of the semantic heading element itself; and
- every descendant `h1`-`h6` element contained in that parent, in
  document order, each with: DOM path; heading tag; and normalized
  heading text (same normalization as above).

The artifact must not record any non-heading, non-anchor page text, and
must not record any numerical TOPIX observation. It must not resolve or
follow any recorded `href`. It must not choose, rank, or otherwise
designate any anchor as the GLOBAL child.

### 4.3 Determinism and no-overwrite

Given the same locked raw bytes, repeated execution must produce a
byte-identical artifact, following the same fail-closed, atomic,
no-overwrite discipline already used for the existing root-structure
artifact: if a different artifact already exists at the target path,
future execution fails closed rather than overwriting it.

If the exact raw `href` (or any other required field) cannot be preserved
unambiguously from the locked bytes, extraction fails closed.

### 4.4 Result classification

The future probe's structural outcome is exactly one of:

- `NEIGHBORHOOD_CAPTURED`: the semantic heading was uniquely identified
  per §2.2, its immediate parent and every required child/anchor/heading
  fact were extracted deterministically, and the artifact was written (or
  byte-identically reproduced).
- `SEMANTIC_HEADING_AMBIGUOUS`: §2.2 did not resolve to exactly one
  semantic heading (any zero/multiple/wrong-tag/wrong-class/inconsistent
  outcome at steps 2-6).
- `STRUCTURE_EXTRACTION_FAILED`: the semantic heading resolved uniquely,
  but parent/child/anchor/heading extraction could not complete
  deterministically (for example, unresolvable raw-`href` ambiguity or
  malformed DOM nesting under the existing malformed-HTML fail-closed
  discipline).

These three outcomes are diagnostic-only. None of them is automatically
mapped to F6 `AVAILABLE` or `MISSING`, and none of them by itself selects,
ranks, or binds a GLOBAL child URL.

## 5. Purpose of the next artifact

`V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_RESULT.json` exists only so
that GPT can later determine, from safe structural evidence, whether a
mechanically unique official child object exists under the observed
semantic section. Execution AI must not decide which anchor, if any, is
the GLOBAL child; that decision remains a future GPT methodology/
traversal adjudication over this artifact's evidence, exactly as F6's
existing evidence-tier and no-execution-agent-methodology-discretion
rules already require (`AI_RESEARCH_EXECUTION_RULES.md` §2, §6).

## 6. No other change

This task changes nothing else. In particular it does not alter:

- the F6 GLOBAL coverage methodology;
- the existing F6 root-structure diagnostic raw identity, label
  normalization, DOM-path rule, or anchor-recording rule;
- the existing retry/redirect policy;
- production F6 support/evidence identity;
- F1, F2, F3, F4, F5, or F7;
- Stage-A authority or any authority flag;
- the existing deferred LOW finding; or
- the existing design-freeze status.

```text
V9_006_STAGE_A_NETWORK_AUTHORIZED=false
V9_006_STAGE_A_EXECUTED=false
V9_006_STAGE_A_IMPLEMENTATION=BLOCK
ACQUISITION_IMPLEMENTATION_COMPLETE=false
V9_design_frozen=false
future_profitability_established=false
```
