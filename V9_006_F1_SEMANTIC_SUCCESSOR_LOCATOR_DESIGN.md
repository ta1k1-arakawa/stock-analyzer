# V9_006 F1 Semantic Successor Locator Design

```text
document_type=SUCCESSOR_LOCATOR_DESIGN
status=DESIGN_AWAITING_GPT_EXACT_SHA_REVIEW
task=V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR_DESIGN
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
methodology_authority=GPT-5.6_Sol
replacement_locator_authorized=false
network_acquisition_authorized=false
phase1_retry=false
phase2_execution=false
human_gate_required=false
human_gate_consumed=false
```

## Identity and purpose

This is a design-only successor-locator identity. It does not implement,
execute, or authorize anything. It does not reopen or retry the terminated
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1` identity, and it does not itself
create acquisition authority for any future JPX request. Its sole purpose is
to freeze a deterministic, semantics-based successor locator for the F1
`LISTED_ISSUES_MONTH_END` `TERMINAL` object -- the "List of TSE-listed
Issues" monthly spreadsheet -- so that a future, separately reviewed
implementation and a future, separately authorized acquisition can locate
that object without any position/extension/currentness heuristic.

## Recorded prior offline evidence and binding

This design is built on top of the two already-reviewed offline artifacts,
bound exactly by hash; it does not re-derive, re-read, or reinterpret their
underlying bytes.

```text
prior_diagnostic_result=EVIDENCE_CAPTURED
prior_diagnostic_network_requests=0
input_payload_sha256=ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f
prior_diagnostic_structural_evidence_sha256=986029641d10d36d33219d729f2c7bdb7c5495447e91be59e11650dd807efad5
probe_diagnostic_result=EVIDENCE_CAPTURED
probe_prior_diagnostic_binding_verified=true
probe_network_requests=0
probe_structural_evidence_sha256=23693a6002b6424695bbf4bf723283b84f774df952f308bda8201da8ba0e5edc
candidate_count=2
candidate_anchor_ordinals=[52,55]
```

## GPT methodology decision

```text
decision=A_UNIQUE_SEMANTIC_EVIDENCE_SUPPORTS_SUCCESSOR_LOCATOR_DESIGN
```

Evidence basis, drawn only from the reviewed candidate-neighborhood probe's
safe evidence:

- Candidate 52: same-JPX-domain resolved URL, `target_extension_class=XLS`,
  empty normalized visible text, nearest preceding eligible data token
  reading `List of TSE-listed Issues (Jul. 2026)`, second-nearest preceding
  eligible data token reading `List of TSE-listed Issues as of previous
  month-end is available.`, and a descendant image whose `alt`/`title`
  normalized text is `icon-xls`.
- Candidate 55: same-JPX-domain resolved URL, `target_extension_class=XLSX`,
  whose local token neighborhood belongs to a distinct TOPIX-labeled
  object (`TOPIX Data`, `TOPIX Constituents (as of month end)`, `List of
  updated issues`) and does not satisfy the F1 semantic grammar below.

Per GPT's explicit instruction, this decision -- and the rule frozen below --
must not, and does not, rely on: XLS vs. XLSX, ordinal 52 vs. 55, presumed
newer file format, the specific value `Jul. 2026`, current date, retrieval
date, or an inferred terminal month `T`. Those facts are recorded above only
as the evidence GPT reviewed; the frozen rule itself (Section 3) never tests
any of them.

## 1. Input/root binding

The successor locator uses the same official JPX listed-issues discovery
root and the same existing `validate_jpx_url` contract already bound in
`src/v9_005_stage_a_jpx_probe.py`. No alternate provider, endpoint, or root
is introduced. A future implementation's only permitted input for a given
run is one root HTML payload obtained (or, for the offline proof plan in
Section 8, already locked) under that exact existing binding; before any
semantic parsing, it must mechanically verify byte length, SHA-256, and
`verify_raw_provenance` PASS against that exact binding. Any mismatch,
uncertainty, or unreadability is `INPUT_BINDING_FAILURE` with no fallback,
alternate input, or additional read.

## 2. Candidate population

Anchors are enumerated in document order using the same reviewed
`HTMLParser(convert_charrefs=True)` mechanics, UTF-8 `errors="replace"`
decode, and structural-failure conditions already frozen for the F1 locked-
root locator successor diagnostic (`src/v9_006_f1_locked_root_locator_successor_diagnostic.py`):
lower-cased tracked-tag comparison, single tracked title, no nested/self-
closing/unmatched/unclosed tracked anchor, and the same `_text` visible-text
normalization (Unicode `\s+` collapsed to one ASCII space, stripped, then
`[:160]` code points).

An anchor is a mechanical candidate if and only if all three hold:

1. `href_present=true` (exactly one `href` attribute, with a string value);
2. resolving that raw href with `urllib.parse.urljoin` against the bound
   root's `resolved_url`, and validating the result with the existing
   `validate_jpx_url` contract, succeeds (`same_jpx_domain_after_resolution
   =true`);
3. the resolved URL's path-only extension, classified by the same one-pass
   suffix mapping already frozen for the successor diagnostic, is in the
   spreadsheet class `{XLS, XLSX}` -- the same class the completed successor
   diagnostic and candidate-neighborhood probe already used to admit
   candidates 52 and 55.

This mechanical population step never inspects, ranks, or prefers one
extension in `{XLS, XLSX}` over the other; both are equally eligible
candidates, and extension is never a selection signal at any later step.

## 3. Semantic qualification

Qualification is decided purely by the F1 target's meaning as expressed in
the two nearest eligible non-empty data tokens immediately preceding the
candidate anchor's `<a>` start tag, using the exact same nearest-first
preceding-token mechanics already frozen for the candidate-neighborhood
probe (`preceding_data_tokens`, descending global token ordinal,
candidate-internal data excluded, each token already canonical under the
frozen `_text` rule). Let `P` be that ordered token list for a given
candidate, with `P[0]` nearest and `P[1]` second-nearest.

A candidate is semantically qualified if and only if all of the following
hold:

- `len(P) >= 2`;
- `P[0]` matches, in its entirety (not as a substring), exactly this
  locale-specific English grammar:

  ```text
  List of TSE-listed Issues (<MON>. <YYYY>)
  ```

  where `<MON>` is exactly one of `Jan`, `Feb`, `Mar`, `Apr`, `May`, `Jun`,
  `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec`; the literal period immediately
  after `<MON>` is required; a single ASCII space separates the period from
  `<YYYY>`; `<YYYY>` is exactly four ASCII digits; and the literal
  parentheses and literal text `List of TSE-listed Issues (` / `)` are
  required exactly as written. Equivalently, as a Python `re.fullmatch`
  pattern against the canonical `_text`-normalized token:

  ```text
  ^List of TSE-listed Issues \((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\. \d{4}\)$
  ```

- `P[1]` equals, exactly and in its entirety, this literal string:

  ```text
  List of TSE-listed Issues as of previous month-end is available.
  ```

Both comparisons are on tokens that are already canonical under the frozen
`_text` rule (surrounding whitespace already collapsed/stripped, already
bounded to 160 code points); the rule performs no additional trimming,
casing, or locale substitution of its own. The parsed `<MON>`/`<YYYY>`
values are locator evidence only (see Section 6) and are never compared
against, or used to select, a specific month or year.

Candidate visible text and a descendant image's `alt`/`title` (e.g. an
`icon-xls`-labeled image, as observed for candidate 52) may be captured as
safe supporting evidence in a future implementation's diagnostic output, but
neither is a qualification condition, and neither may become one without a
separate GPT methodology decision. No other candidate attribute, position,
extension, filename, or apparent recency may be added as a qualification
condition.

## 4. Uniqueness

Across all mechanical candidates from Section 2, exactly one must satisfy
the semantic rule in Section 3.

- Zero qualifying candidates: `SOURCE_OR_DATA_FEASIBILITY_FAILURE`.
- More than one qualifying candidate: `SOURCE_OR_DATA_FEASIBILITY_FAILURE`.

There is no tie-break by ordinal, extension, date, filename, first-match, or
last-match. A future implementation must not add one.

## 5. URL handling

After -- and only after -- semantic uniqueness (Section 4) selects exactly
one candidate, the exact raw href string of that candidate is retained
internally only. It is resolved exactly once more, deterministically, with
`urllib.parse.urljoin` against the same bound official root's `resolved_url`
used in Section 2, and that result must again pass the existing
`validate_jpx_url` contract. Because Section 2's candidate population
already required this same resolution/validation to succeed for this exact
candidate, a mismatch here can only reflect an input-binding inconsistency
(never a semantic condition); any such mismatch is `INPUT_BINDING_FAILURE`.

No safe artifact, log, or report produced by a future implementation may
ever print the raw href or the resolved URL. Provenance is expressed only
through `raw_href_sha256` and `resolved_url_sha256` (SHA-256 of the exact
UTF-8 bytes), computed the same way as the existing reviewed diagnostics.

## 6. Terminal-month discipline

This successor locator's only job is to locate the F1 `TERMINAL` object. It
must never define the terminal snapshot month `T` from: the page label
month/year captured as locator evidence in Section 3, current date,
retrieval timestamp, filename, URL, F7, or any other proxy. `T` remains
exclusively the output of the separately reviewed F1 terminal
parser/boundary, mechanically parsed later from the exact bytes of the
object this locator selects. This design creates, changes, and authorizes
no such parsing.

## 7. Successor identity / authority

```text
replacement_locator_authorized=false
network_acquisition_authorized=false
phase1_retry=false
phase2_execution=false
```

This design does not reopen the consumed Phase-1 one-shot execution and does
not authorize a second execution of that old identity. A future network
acquisition using this locator must be a separately reviewed successor
acquisition identity, with its own explicit point-of-use human/gate
authority as `AI_RESEARCH_EXECUTION_RULES.md` and
`AI_REAL_EXECUTION_RUNBOOK.md` require for any real JPX request. Nothing in
this design is, or substitutes for, that authority.

## 8. Offline proof plan

Before any new JPX request, a future implementation of this locator must
first be exercised with synthetic HTML fixtures covering: zero candidates,
one qualifying candidate, more than one qualifying candidate, a candidate
whose `P[0]` almost matches the grammar but fails on one field (month
outside the 12-token enum, missing period, non-4-digit year, extra/missing
parenthesis or space), and a candidate with `P[0]` matching but `P[1]` not
matching the exact literal.

Only after that synthetic coverage passes may the implementation be run
offline, with zero network requests, against the exact already-locked F1
`TERMINAL_DISCOVERY_ROOT` payload bound in this document
(`payload_sha256=ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f`).
The required expected result on that exact bound root is:

```text
result=SUCCESSOR_LOCATOR_MATCHED
qualifying_candidate_count=1
selected_raw_href_sha256=ee97b7976663aa4dd55f9f02d33e96ceb66ad76bb43fd2e4523a31fe4d4a6ec9
selected_resolved_url_sha256=a7088b6c7e5ea028ffad54bd95e835e32068dfafa324d737e2cef0424f90e613
candidate_55_qualifies=false
network_requests=0
raw_or_receipt_state_modified=false
```

That is: the selected match must bind, by hash only, to the already-reviewed
candidate-52 `raw_href_sha256`/`resolved_url_sha256` recorded in the
candidate-neighborhood probe's frozen `_CANDIDATE_BINDINGS[52]`; candidate 55
must not satisfy Section 3's semantic rule. This offline run is a locator
*validation* only. It creates no acquisition authority, consumes no gate,
and modifies no raw or receipt state -- exactly like the two prior offline
diagnostics it is bound to.

## 9. Closed failures

A future implementation's result is exactly one of:

```text
SUCCESSOR_LOCATOR_MATCHED
INPUT_BINDING_FAILURE
HTML_STRUCTURE_UNSUPPORTED
SOURCE_OR_DATA_FEASIBILITY_FAILURE
SAFE_OUTPUT_VALIDATION_FAILURE
```

`SUCCESSOR_LOCATOR_MATCHED` is the only success class, and requires exactly
one semantically qualifying candidate (Section 4) whose URL re-validation
(Section 5) passed. `INPUT_BINDING_FAILURE` covers Section 1's root/payload
binding failures and Section 5's recomputation-mismatch case.
`HTML_STRUCTURE_UNSUPPORTED` covers the inherited parser structural-failure
conditions from Section 2. `SOURCE_OR_DATA_FEASIBILITY_FAILURE` covers
Section 4's zero-match and multiple-match cases. `SAFE_OUTPUT_VALIDATION_FAILURE`
covers any attempt to emit a safe artifact whose text fields fail the same
existing unsafe-text rejection (`http://`, `https://`, `file:`, or a Windows
drive path) already frozen for the reviewed diagnostics.

No result, including `SUCCESSOR_LOCATOR_MATCHED`, may contain a raw href,
raw URL, local path, timestamp affecting deterministic content, operator
identity, or arbitrary exception text. No partial/unsafe output is ever
emitted for a non-success result; a future implementation's closed schema
follows the same mechanically-empty-on-failure discipline already frozen for
the two prior diagnostics.

## 10. Research integrity and scope discipline

This is source/locator remediation only. It does not alter, and no future
implementation of it may alter: F1 source semantics beyond locating the
`TERMINAL` object; target/label definitions; evaluation, validation, or
holdout design; the sample universe; costs or slippage assumptions;
thresholds; search space; Phase-2 rules; the terminal-month definition
(Section 6); or V9 strategy methodology generally. F2 through F7's roots,
traversal semantics, monthly-coverage mappings, and all other previously
bound methodology are unchanged and untouched by this design.

## Authority created

```text
NETWORK_REQUESTS=0
ACTUAL_LOCKED_PAYLOAD_READS=0
HUMAN_GATES_CONSUMED_BY_THIS_TASK=0
CODE_CHANGED=false
IMPLEMENTATION_AUTHORIZED=false
EXECUTION_AUTHORIZED=false
replacement_locator_authorized=false
network_acquisition_authorized=false
phase1_retry=false
phase2_execution=false
future_profitability_established=false
```

This document is a docs-only design decision. It does not authorize
implementation, offline execution, network access, data acquisition, T
parsing, Phase-2 execution, or V9 design freeze, and does not consume any
existing or future human gate.

## Next action

`GPT_EXACT_SHA_DESIGN_REVIEW`: obtain GPT-5.6 Sol's independent exact-SHA
review of this design. Only after PASS may a future, separately authorized
task implement this locator and run the Section 8 offline proof plan; only
after that implementation's own GPT exact-SHA review PASS, and only under a
fresh separately reviewed successor acquisition identity with its own
point-of-use human/gate authority, may any new JPX network request be made.
