# V9_006 F1 locked-root locator successor diagnostic design

```text
document_type=SUCCESSOR_OFFLINE_DIAGNOSTIC_DESIGN
status=DESIGN_AWAITING_GPT_EXACT_SHA_REVIEW
task=V9_006_F1_LOCKED_ROOT_LOCATOR_SUCCESSOR_DIAGNOSTIC
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
execution_authorized=false
network_authorized=false
human_gate_required=false
human_gate_consumed=false
```

## Identity and purpose

This is a new successor diagnostic identity only. It does not reopen, retry,
or alter the terminated `V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1` identity;
it does not select, approve, or define a replacement F1 locator.

Its sole purpose is to use only the already-locked F1
`TERMINAL_DISCOVERY_ROOT` public payload to capture deterministic, safe
structural evidence sufficient for GPT methodology authority to later decide
whether a successor F1 locator methodology can be defined.

## Authoritative input binding and preconditions

The only permitted future input is the existing canonical raw-lock pair with
these safe bindings:

```text
source_family=LISTED_ISSUES_MONTH_END
applicable_period=TERMINAL_DISCOVERY_ROOT
http_status=200
byte_length=30059
payload_sha256=ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f
terminal_adjudication_raw_lock_id_set_sha256=8e0f0798c6da09292c964e56efb6954dd2e57ac2191b1e80c7fc03eb1d9ba621
```

Before parsing, future execution must mechanically verify exact metadata
identity, payload byte length and SHA-256, and `verify_raw_provenance` PASS.
Every mismatch, uncertainty, unreadability, or provenance failure is
`INPUT_BINDING_FAILURE` and BLOCKs with no fallback, alternate input, or
additional read.

## Absolute prohibitions

The diagnostic has exactly zero network requests. It must not use a
current/live JPX page, browser/web lookup, alternate provider, historical F1
search, F2--F7 evidence, a second Phase-1 execution, receipt reset/delete/
change, or a new OutputRoot acquisition. It must not choose a replacement
locator or infer one from a filename, anchor position, currentness, terminal
month T, or any other heuristic. It must not alter the historical
`data_j.xls` contract.

Committed or safe output must not contain raw payload bytes, raw hrefs,
resolved URLs, local paths, confirmation tokens, arbitrary HTML fragments,
body/paragraph dumps, timestamps affecting deterministic content, operator
identity, or arbitrary exception text.

## Deterministic parsing contract

After the binding preconditions PASS, decode only the exact locked payload
with UTF-8 `errors="replace"` and parse it with Python stdlib
`HTMLParser(convert_charrefs=True)`. Anchors are enumerated by `<a>` start-tag
order; headings are `h1`--`h6` start-tag order. A nested `<a>`, nested tracked
heading, duplicate `href` attributes, unmatched tracked closing tag, tracked
self-closing tag, or unclosed tracked state at EOF is
`HTML_STRUCTURE_UNSUPPORTED`. The tracked title follows the same discipline;
more than one title element is `HTML_STRUCTURE_UNSUPPORTED`.

Anchor visible text is the concatenation of `handle_data` events while that
anchor is active, including text in ordinary nested elements. Heading and
title text use the same rule. Normalize every such text with Unicode regex
`\s+` to one ASCII space, strip, then take `[:160]` Unicode code points.
Before safe emission, the safe-output validator rejects any normalized title,
heading, or anchor visible text containing case-insensitive `http://`,
`https://`, or `file:`, or a Windows drive path matching
`[A-Za-z]:[\\/]`; this is `SAFE_OUTPUT_VALIDATION_FAILURE`, with no redaction
or substitute-and-continue behavior.

For every anchor, emit its ordinal, normalized visible text, and the nearest
preceding heading ordinal (or `null`). The latter is the most recent heading
start-tag ordinal observed before the anchor start-tag, including a heading
currently containing that anchor. Determine href semantics exactly as follows:

- No href attribute: `href_present=false`,
  `same_jpx_domain_after_resolution="unknown"`, extension `NONE`, and both
  URL hashes `null`.
- Exactly one href attribute: `href_present=true`. If its value is `None`,
  set same-domain to `"unknown"`, extension to `OTHER`, and both hashes to
  `null`.
- A string href, including the empty string: `raw_href_sha256` is SHA-256 of
  the exact HTMLParser-decoded href UTF-8 bytes.

The only resolution base is canonical raw-lock metadata `resolved_url`;
`requested_url` must never be used. During input binding, that metadata
`resolved_url` must itself pass the existing `validate_jpx_url` contract.
Resolve a string href only with `urllib.parse.urljoin(locked_resolved_url,
raw_href)` and SHA-256 the exact result's UTF-8 bytes as
`resolved_url_sha256`. `same_jpx_domain_after_resolution=true` only when that
result passes the existing `validate_jpx_url` contract; a deterministic
resolution whose JPX validation fails is `false`. Any URL-resolution or URL-
parsing exception gives `"unknown"` and a `null` resolved hash.

For `href_present=true`, if no resolved URL exists the extension is `OTHER`.
Otherwise inspect only `urllib.parse.urlsplit(resolved_url).path`, percent-
decode exactly once with `urllib.parse.unquote`, lowercase, and map its final
suffix exactly as follows; query and fragment text never affect the class:

```text
XLS | XLSX | CSV | ZIP | PDF | HTML | OTHER | NONE
```

```text
.xls -> XLS     .xlsx -> XLSX     .csv -> CSV     .zip -> ZIP
.pdf -> PDF     .html/.htm -> HTML     otherwise -> OTHER
```

These hashes are evidence identities only and must never be reverse-used to
guess URLs.

The deterministic candidate subset contains exactly anchors where
`href_present=true`, `same_jpx_domain_after_resolution=true`, and
`target_extension_class` is one of `XLS`, `XLSX`, `CSV`, or `ZIP`. This is
an evidence subset, not a locator selection or authorization.

## Closed safe result schema

Future execution may write one deterministic canonical-JSON safe artifact
with exactly this schema; no optional or extra fields are permitted.

```json
{
  "schema_version": "V9_006_F1_LOCKED_ROOT_LOCATOR_SUCCESSOR_DIAGNOSTIC_V1",
  "task": "V9_006_F1_LOCKED_ROOT_LOCATOR_SUCCESSOR_DIAGNOSTIC",
  "input": {
    "source_family": "LISTED_ISSUES_MONTH_END",
    "applicable_period": "TERMINAL_DISCOVERY_ROOT",
    "http_status": 200,
    "byte_length": 30059,
    "payload_sha256": "ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f",
    "terminal_adjudication_raw_lock_id_set_sha256": "8e0f0798c6da09292c964e56efb6954dd2e57ac2191b1e80c7fc03eb1d9ba621",
    "metadata_identity_verified": true,
    "payload_binding_verified": true,
    "raw_provenance_verified": true
  },
  "diagnostic_result": "EVIDENCE_CAPTURED | INPUT_BINDING_FAILURE | HTML_STRUCTURE_UNSUPPORTED | SAFE_OUTPUT_VALIDATION_FAILURE",
  "document_parse_status": "PARSED | UNSUPPORTED | NOT_PARSED",
  "title": "bounded normalized text or null",
  "total_anchor_count": 0,
  "total_heading_count": 0,
  "headings": [{"ordinal": 1, "level": 1, "normalized_text": "bounded normalized text"}],
  "anchors": [{
    "anchor_ordinal": 1,
    "normalized_visible_text": "bounded normalized text",
    "nearest_preceding_heading_ordinal": null,
    "href_present": true,
    "same_jpx_domain_after_resolution": true,
    "target_extension_class": "XLS",
    "raw_href_sha256": "lowercase SHA-256 or null",
    "resolved_url_sha256": "lowercase SHA-256 or null"
  }],
  "candidate_anchor_ordinals": [1],
  "candidate_count": 1,
  "locator_decision": "NOT_MADE",
  "replacement_locator_authorized": false,
  "network_requests": 0,
  "structural_evidence_sha256": "lowercase SHA-256"
}
```

Counts and arrays must agree exactly. `headings` has strictly increasing
document-order ordinals and levels in `1..6`; `anchors` has strictly
increasing document-order ordinals. `candidate_anchor_ordinals` is strictly
increasing, unique, and references only matching entries in `anchors`.
`structural_evidence_sha256` is SHA-256 of the canonical JSON UTF-8 encoding
of all safe result fields except `structural_evidence_sha256` itself.

Failure output is mechanically closed. For `INPUT_BINDING_FAILURE`, use
`NOT_PARSED`, `title=null`, zero counts, empty arrays, and candidate count
zero; `metadata_identity_verified` is true only if that exact check passed,
`payload_binding_verified` is true only if metadata passed and byte/hash
passed, and `raw_provenance_verified` is true only if prior bindings passed
and provenance passed (otherwise each respective boolean is false). Emit no
partial parse observations.

For `HTML_STRUCTURE_UNSUPPORTED`, all three input-verification booleans are
true, `document_parse_status=UNSUPPORTED`, `title=null`, all counts are zero,
all arrays empty, and candidate count zero; discard every partial parser
observation. For `SAFE_OUTPUT_VALIDATION_FAILURE`, all three input-
verification booleans are true, `document_parse_status=PARSED`, `title=null`,
all counts zero, all arrays empty, and candidate count zero; discard every
otherwise parsed observation. Only `EVIDENCE_CAPTURED` has all three
verification booleans true, `document_parse_status=PARSED`, and complete
validated evidence.

For every result class, `locator_decision`, `replacement_locator_authorized`,
and `network_requests` remain exactly `"NOT_MADE"`, `false`, and `0`.
`structural_evidence_sha256` is computed over the complete closed safe object,
excluding only `structural_evidence_sha256` itself.

## Decision separation and topology

The diagnostic must not select a winning anchor, infer correctness from an
apparently favorable filename, infer terminal month T, or alter the old F1
locator contract. Following separately reviewed future execution, only GPT
exact-SHA methodology authority may choose: (A) sufficient unique semantic
evidence and a successor locator design; (B) ambiguous evidence and a
narrower offline probe over these same locked bytes; or (C) no usable
evidence and termination of this successor path. Execution AI must not choose
A, B, or C.

After this design and a later implementation PASS, execution is direct
Windows PowerShell against the already-locked machine-local payload. It is
offline-only, consumes no human one-shot gate, and creates no public
acquisition authority.
