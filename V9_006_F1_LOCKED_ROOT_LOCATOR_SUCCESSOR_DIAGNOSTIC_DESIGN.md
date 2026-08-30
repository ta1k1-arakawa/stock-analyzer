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

After the binding preconditions PASS, parse only the exact locked HTML bytes.
Deterministically enumerate HTML anchors and headings in document order.
Whitespace normalization replaces every maximal Unicode whitespace run with
one ASCII space and trims leading/trailing whitespace. Title, heading, and
visible anchor text are normalized then truncated to at most 160 Unicode code
points; no other text is emitted.

For every anchor, determine its ordinal, normalized visible text, nearest
preceding heading ordinal (or `null`), href presence, and, when a href is
present and deterministic URL resolution can be performed from the locked
document context alone, whether resolution remains on the JPX domain. A
failure to make that determination is `"unknown"`, not a guess. Classify a
present target only into this closed enum:

```text
XLS | XLSX | CSV | ZIP | PDF | HTML | OTHER | NONE
```

`NONE` applies only when no href is present. For present hrefs, emit the
SHA-256 of the raw href and the SHA-256 of the resolved URL when resolution
succeeds; otherwise the relevant value is `null`. These hashes are evidence
identities only and must never be reverse-used to guess URLs.

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

For failure results, no field may invent a parse observation: binding failure
uses `NOT_PARSED`, zero counts, and empty arrays; HTML-structure or safe-output
validation failure retains only observations mechanically validated for safe
emission. `locator_decision`, `replacement_locator_authorized`, and
`network_requests` remain exactly `"NOT_MADE"`, `false`, and `0` in every
result class.

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
