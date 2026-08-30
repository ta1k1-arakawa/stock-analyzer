# V9_006 F1 Candidate Token-Neighborhood Probe Design

```text
document_type=SUCCESSOR_OFFLINE_DIAGNOSTIC_DESIGN
task=V9_006_F1_CANDIDATE_TOKEN_NEIGHBORHOOD_PROBE
status=AWAITING_GPT_EXACT_SHA_DESIGN_REVIEW
study_identity=SUCCESSOR_DIAGNOSTIC_ONLY
```

## Purpose and decision boundary

This new successor diagnostic identity is offline-only. It reads only the
same already-locked F1 `TERMINAL_DISCOVERY_ROOT` HTML bytes after the required
binding succeeds. It does not reopen or retry the terminal Phase-1 identity,
choose a candidate, authorize a replacement locator, or authorize public
acquisition.

GPT methodology authority has decided exactly:

```text
DECISION=B_AMBIGUOUS_EVIDENCE_REQUIRES_NARROWER_OFFLINE_PROBE
```

Candidate 52 and candidate 55 are symmetric evidence subjects. File
extension, ordinal position, presumed recency, and apparent filename
modernity must never select either candidate. GPT alone interprets a future
safe result and chooses A (successor locator design), B (another narrower
offline probe), or C (terminate this successor path).

## Recorded completed prior diagnostic and binding

The completed reviewed offline diagnostic is bound exactly:

```text
diagnostic_result=EVIDENCE_CAPTURED
network_requests=0
metadata_identity_verified=true
payload_binding_verified=true
raw_provenance_verified=true
total_anchor_count=83
total_heading_count=1
candidate_anchor_ordinals=[52,55]
candidate_count=2
structural_evidence_sha256=986029641d10d36d33219d729f2c7bdb7c5495447e91be59e11650dd807efad5
payload_sha256=ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f
```

| Anchor | Raw href SHA-256 | Resolved URL SHA-256 | Extension | Same JPX domain | Visible text | Nearest heading |
|---:|---|---|---|---|---|---:|
| 52 | `ee97b7976663aa4dd55f9f02d33e96ceb66ad76bb43fd2e4523a31fe4d4a6ec9` | `a7088b6c7e5ea028ffad54bd95e835e32068dfafa324d737e2cef0424f90e613` | XLS | true | empty | 1 |
| 55 | `759c2f9e683c85ebcd865ea962e7e24a28ce2db4d0c7ff0592e5cdd03dba632b` | `b8953e84885003e03ee9feafd5408c313c91404582d58f2c489030dfbb4b98b0` | XLSX | true | empty | 1 |

Before probe parsing, execution must invoke and recompute the reviewed
offline diagnostic against the same candidate input; validate its safe
result; and require every value above, including the candidate list/order and
both candidate rows. Any missing, malformed, noncanonical, or unequal value
is `PRIOR_DIAGNOSTIC_BINDING_FAILURE`: stop with no fallback, alternative
payload, locator inference, parsing, or partial semantic output.

## Absolute prohibitions

```text
network_requests=0
current_jpx_page=false
browser_or_web_lookup=false
alternate_provider=false
phase1_retry=false
receipt_or_raw_state_change=false
phase2_execution=false
terminal_month_inference=false
replacement_locator_decision=false
```

Never emit raw hrefs, resolved URLs, filenames, download values, image
sources, local paths, raw HTML, raw bytes, arbitrary exceptions, timestamps,
operator identity, or confirmation material.

## Frozen parser and token mechanics

Use the reviewed UTF-8 `errors="replace"` decode and Python stdlib
`HTMLParser(convert_charrefs=True)` mechanics. Existing tracked
anchor/title/heading structural failures remain failures. Track nesting depth
for `script`, `style`, `noscript`, and `template`; a `handle_data` event is
eligible only when every depth is zero.

For each eligible event, apply the existing `_text` rule: Unicode-regex
whitespace to one ASCII space, strip, then at most 160 Unicode code points.
Empty normalized values are not tokens. Non-empty values receive
`data_token_ordinal` starting at exactly 1 in `handle_data` event order.
Every emitted text value must pass the existing unsafe-text rejection for
`http://`, `https://`, `file:`, and Windows drive paths. A violation is
`SAFE_OUTPUT_VALIDATION_FAILURE`; do not redact or substitute it.

For each candidate independently:

- `preceding_data_tokens` contains up to eight eligible non-empty tokens
  strictly before the candidate `<a>` start tag, in **nearest-first** order
  (descending global token ordinal).
- `following_data_tokens` contains up to eight eligible non-empty tokens
  strictly after the matching candidate `</a>` end tag, in **nearest-first**
  order (ascending global token ordinal).
- Data events within the candidate anchor are neither preceding nor following.
- Each token has exactly `data_token_ordinal` (positive exact integer) and
  `normalized_text` (canonical safe text). Duplicate text stays distinct.

Candidate anchors may not nest. A duplicate `title`, `aria-label`, or
`download` attribute on a candidate anchor is `HTML_STRUCTURE_UNSUPPORTED`.
Nested `<img>` is allowed while a candidate is active, but duplicate `alt`,
`title`, or `src` on an image is unsupported. Unclosed tracked candidate
state at EOF or unmatched tracked closes are unsupported. Ordinary nested
elements contribute eligible data normally.

## Candidate context contract

`candidate_contexts` is exactly `[52,55]` order. Each object has exactly:

```text
candidate_anchor_ordinal
candidate_binding
preceding_data_tokens
following_data_tokens
previous_anchor_summaries
next_anchor_summaries
title_present
normalized_title
aria_label_present
normalized_aria_label
download_present
download_value_sha256
total_image_count
images
```

`candidate_binding` is exactly the seven-field prior-diagnostic safe summary:
`anchor_ordinal`, `normalized_visible_text`,
`nearest_preceding_heading_ordinal`, `same_jpx_domain_after_resolution`,
`target_extension_class`, `raw_href_sha256`, and `resolved_url_sha256`. It
must equal its table row above.

`previous_anchor_summaries` contains up to three prior-diagnostic summaries
for immediately lower anchor ordinals, nearest-first (descending ordinal).
`next_anchor_summaries` contains up to three immediately higher summaries,
nearest-first (ascending ordinal). Each summary has exactly the same seven
safe fields; neither list contains its candidate. They are copied only from
the bound recomputed prior diagnostic, never a new URL-resolution algorithm.

Candidate `title` and `aria-label` each have exact absent/None/string states:
absent means its `_present=false` and normalized value `null`; present with
HTMLParser value `None` means `_present=true` and normalized value `null`;
present with a string, including empty, means `_present=true` and the
normalized bounded safe string. For `download`, absent means
`download_present=false`/hash `null`, present `None` means true/null, and a
string (including empty) means true/SHA-256 of its exact HTMLParser-decoded
UTF-8 string. Raw values are never emitted.

`total_image_count` is the exact non-bool nonnegative count of descendant
`img` start tags while the candidate is active. `images` contains the first
up to eight in image start-tag document order. Each image object has exactly:

```text
image_ordinal_within_candidate
alt_present
normalized_alt
title_present
normalized_title
src_present
src_sha256
```

For image `alt` and `title`, absent is false/null, present `None` is
true/null, and a string (including empty) is true/normalized bounded safe
text. For `src`, absent is false/null, present `None` is true/null, and a
string (including empty) is true/SHA-256 of its exact HTMLParser-decoded
UTF-8 string. Raw image sources are never emitted.

## Closed safe result and failures

Canonical JSON is exactly:

```python
json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
```

The closed top-level key set is exactly:

```text
schema_version
task
input_payload_sha256
prior_diagnostic_structural_evidence_sha256
prior_diagnostic_binding_verified
diagnostic_result
candidate_contexts
locator_decision
replacement_locator_authorized
network_requests
structural_evidence_sha256
```

Fixed values are `locator_decision="NOT_MADE"`,
`replacement_locator_authorized=false`, and `network_requests=0`. Captured
results contain exactly both contexts in `[52,55]` order, and validate all
types, lists, bounds, counts, canonical text, ordinals, hashes, and bindings.
`structural_evidence_sha256` is SHA-256 of the canonical complete object
excluding only itself.

The only result enum is:

```text
EVIDENCE_CAPTURED
PRIOR_DIAGNOSTIC_BINDING_FAILURE
HTML_STRUCTURE_UNSUPPORTED
SAFE_OUTPUT_VALIDATION_FAILURE
```

Every failure is mechanically empty: `candidate_contexts=[]`; no token,
neighbor, attribute, image, parser, or partial semantic observation appears.
`PRIOR_DIAGNOSTIC_BINDING_FAILURE` requires
`prior_diagnostic_binding_verified=false`; the other three results require it
true. Fixed input and prior-hash fields remain provenance bindings for all
results.

## Future execution topology

After independent design and implementation PASSes, this may run only as a
direct Windows PowerShell offline operation over the existing machine-local
locked payload. It consumes no human gate and creates no public-acquisition
authority. GPT alone interprets the safe evidence and makes the next decision.
