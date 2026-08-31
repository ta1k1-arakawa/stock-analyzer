# V9-006 F1 Terminal-Month Structure Evidence Diagnostic Design

Status: candidate design awaiting GPT-5.6 Sol exact-SHA review. This document
defines a future offline diagnostic only; it does not authorize execution,
payload access, T parsing, Phase 2, or the F2 bridge.

## Authority and immutable input binding

The diagnostic is subordinate to the frozen public-acquisition design
(`DESIGN_GIT_SHA=0ee4b338110c626fb92267343586fa6936699805`) and to the reviewed
Stage-2C implementation identity
(`IMPLEMENTATION_GIT_SHA=4efd6ab8ca951a9bbc67bc0146ecb86a20533a0e`). It may run
only against the one already-successful successor acquisition, whose
machine-private durable state is identified by the same stable sibling-state
derivation used by Stage 2A/2C, including the fixed state-root basename
`v9-006-f1-successor-public-acquisition-state`. The parent path is internal and
is never printed or placed in safe evidence.

Before opening any payload, the future implementation must read and verify the
existing safe acquisition result and both private raw-lock objects. These are
exact bindings, not caller inputs or inferred values:

| Binding | Required value |
| --- | --- |
| acquisition result | `SUCCESS` / `NONE` |
| implementation SHA | `4efd6ab8ca951a9bbc67bc0146ecb86a20533a0e` |
| design SHA | `0ee4b338110c626fb92267343586fa6936699805` |
| terminal payload SHA-256 | `3119fb5c0854544b0f17b2abda1db836201fac60027695ff95d10bea103df187` |
| terminal byte length | `851456` |
| raw-lock-set SHA-256 | `f7d641052f3cb1e1ab33936303e2e504bc480ff9d89cde85ccade5d214f193cf` |
| acquisition network requests | `2`; this diagnostic must use `0` |
| second execution | `false` (no refetch or restart path) |

The implementation must require the exact reviewed safe-result key set,
lowercase-hex/type/nullability rules, exactly two raw-lock directories (ROOT
and TERMINAL), and metadata that independently recomputes each payload
hash/length and the raw-lock-set digest. The terminal metadata URL is validated
privately but is never emitted. Any missing, malformed, conflicting,
substituted, or newly-created state is
`INPUT_BINDING_FAILURE`; no alternate file, filename, URL, date, current month,
download, or locator-page label may be consulted.

## Read-only execution seam

The future entrypoint receives no URL, period, filename, or payload argument.
It derives the stable state root internally, opens the already durable TERMINAL
lock read-only, and passes the exact reread built-in `bytes` object to the
reviewed schema-discovery profiler. It must not create, modify, delete, rename,
repair, or replace any receipt, lock, result, or audit file. A software-only
failure may be retried against those same bytes, but no network callback or
source acquisition is available. `NETWORK_REQUESTS` is always `0`.

The parser/profile seam should reuse the reviewed OLE/BIFF container and cell
taxonomy logic from `src/v9_006_stage_a_schema_discovery.py` (in particular its
verified-lock profile path), rather than implementing a second workbook
parser. The diagnostic may project that profile into the narrower safe
structure below; it must not expose the broader internal object.

## Narrow safe structural projection

The only public evidence is one canonical JSON object with exactly these keys:

```text
task
implementation_git_sha
design_git_sha
terminal_payload_sha256
terminal_byte_length
raw_lock_set_sha256
diagnostic_result
failure_stage
container_format
sheet_count
sheets
text_neighborhood
neighborhood_truncated
structural_evidence_sha256
network_request_count
safe_provenance_verified
```

`diagnostic_result` is one of `EVIDENCE_CAPTURED`,
`INPUT_BINDING_FAILURE`, `FORMAT_OR_STRUCTURE_UNSUPPORTED`,
`SAFE_OUTPUT_VALIDATION_FAILURE`, or `IMPLEMENTATION_FAILURE`. The only
`failure_stage` values are `NONE`, `PRE_READ_BINDING`, `TERMINAL_LOCK_READ`,
`STRUCTURE_PROFILE`, `SAFE_PROJECTION`, and `IMPLEMENTATION`. The pair is
closed: `EVIDENCE_CAPTURED` uses `NONE`; each failure uses its corresponding
stage. No other enum or extra key is valid.

For `EVIDENCE_CAPTURED`, `safe_provenance_verified=true`, the immutable
identity fields equal the bindings above, and `network_request_count=0`:

* `container_format` is the profiler's bounded format identifier; `OLE_BIFF`
  is the supported path for this diagnostic.
* `sheet_count` is a nonnegative integer and `sheets` has exactly that many
  entries ordered by one-based sheet ordinal.
* Each sheet contains only `sheet_ordinal`, `sheet_name`, `visibility`,
  `row_count`, `column_count`, and `column_cell_type_counts`. Names and text
  are bounded to the reviewed code-point limit; dimensions/counts are exact
  nonnegative integers; cell-type counts are taxonomy only.
* `text_neighborhood` contains deterministically ordered visible-sheet rows
  and cells bounded to the reviewed row/cell limits. Each cell has only
  sheet/row/column ordinals, `cell_type`, and (for `TEXT` cells only) bounded
  normalized text that passes the allowlist below. Numeric, date, boolean,
  error, formula, ticker, price, outcome, and membership values are represented
  only by taxonomy, never by their values.
* `neighborhood_truncated` is an explicit boolean carried from deterministic
  profiler truncation. Truncation means a narrower follow-up design may be
  needed; it is not permission to guess a month.

For every non-success state, structure fields are null or empty according to a
single frozen schema rule, `safe_provenance_verified=false`, and no structural
digest is fabricated. A structural digest, when present, is SHA-256 of the
canonical JSON projection with its digest field removed. Canonical JSON uses
UTF-8, sorted keys, compact separators, and `ensure_ascii=false`.

The closed value matrix is:

| `diagnostic_result` | `failure_stage` | Structure fields | Digest |
| --- | --- | --- | --- |
| `EVIDENCE_CAPTURED` | `NONE` | populated bounded projection | actual projection digest |
| `INPUT_BINDING_FAILURE` | `PRE_READ_BINDING` or `TERMINAL_LOCK_READ` | `container_format=null`, `sheet_count=null`, `sheets=[]`, `text_neighborhood=[]`, `neighborhood_truncated=false` | `null` |
| `FORMAT_OR_STRUCTURE_UNSUPPORTED` | `STRUCTURE_PROFILE` | same empty structure | `null` |
| `SAFE_OUTPUT_VALIDATION_FAILURE` | `SAFE_PROJECTION` | same empty structure | `null` |
| `IMPLEMENTATION_FAILURE` | `IMPLEMENTATION` | same empty structure | `null` |

All rows retain the five immutable identity fields and
`network_request_count=0`; only the captured row may set
`safe_provenance_verified=true`.

The projection excludes private paths, resolved URLs, receipt contents, raw
workbook bytes, arbitrary exception text, and numeric or membership data
unrelated to locating a future date/month field.

## Frozen text allowlist

The allowlist is applied after the existing bounded normalization (trim,
character-reference normalization, and the reviewed maximum text length).
Matching is whole-string, ASCII case-insensitive for English month names, and
uses ASCII decimal digits only. No substring search, token search, fuzzy match,
or fallback is permitted. Define:

```text
MONTH = January|February|March|April|May|June|July|August|September|October|November|December
DAY = (0?[1-9]|[12][0-9]|3[01])
YEAR = [0-9]{4}
NUMERIC = YEAR[-/.](0?[1-9]|1[0-2])(?:[-/.]DAY)?
JAPANESE = YEAR年(0?[1-9]|1[0-2])月(?:DAY日)?
ENGLISH = MONTH[ ]+YEAR | MONTH[ ]+DAY[,]?[ ]+YEAR
DATE = (?:ENGLISH|NUMERIC|JAPANESE)[.]?
AS_OF = As[ ]+of[ ]+DATE
HEADING = List[ ]+of[ ]+TSE-listed[ ]+Issues[ ]\([ ]*DATE[ ]*\)
```

The implementation compiles these productions into anchored regular
expressions equivalent to `^...$`; `DAY` is expanded in `ENGLISH` and
`JAPANESE` rather than interpolated as a free substring. Thus accepted examples
include `January 2026`, `January 31, 2026`, `2026-01`, `2026/01/31`,
`2026年1月`, `As of January 2026`, and the exact generic heading
`List of TSE-listed Issues (January 2026)`. A bare four-digit year, bare
integer, security code, company name, or text containing one of these forms is
rejected. Punctuation outside the explicitly shown comma, period, separators,
and heading parentheses is rejected.

For every TEXT cell that fails this whole-string allowlist, the public
projection emits only its coordinates and `cell_type`; its text is discarded
before serialization, logging, exception construction, structural hashing, or
any reversible lookup artifact. The implementation must not retain a hidden
side channel containing discarded text.

The allowlist is evidence filtering, not a T parser. Zero qualifying date-text
cells, multiple qualifying cells, or any profiler truncation are reported as
bounded evidence conditions and never trigger a guessed month, rule broadening,
or downstream action. GPT will decide later whether the evidence is sufficient
to freeze a deterministic T rule.

## No T decision and no downstream authorization

This diagnostic does not emit `T`, a candidate month, a selected date, or a
parser decision. It does not inspect the root HTML label, URL, filename,
retrieval date, or current date to infer one. It does not call
`f2_bridge_months(T)`, acquire F2, parse a terminal-month table, execute Phase
2, or authorize any human gate. GPT will inspect the bounded evidence and, only
in a later separately reviewed design task, freeze a mechanical T parser rule.

## Failure-closed state machine

1. Bind the stable state identity and exact safe acquisition result; otherwise
   return `INPUT_BINDING_FAILURE/PRE_READ_BINDING` without opening payload bytes.
2. Reopen and independently verify exactly the terminal lock and its payload;
   any read, type, hash, length, metadata, URL-validation, or lock-set
   contradiction returns `INPUT_BINDING_FAILURE/TERMINAL_LOCK_READ`.
3. Invoke only the reviewed offline structural profiler. A recognized but
   unsupported container/structure returns
   `FORMAT_OR_STRUCTURE_UNSUPPORTED/STRUCTURE_PROFILE`.
4. Build and validate the narrow projection and its digest. A projection
   validation failure returns `SAFE_OUTPUT_VALIDATION_FAILURE/SAFE_PROJECTION`
   with no invented evidence. An unexpected implementation exception propagates
   to the future execution envelope; this design does not manufacture one.
5. Only a fully validated projection produces `EVIDENCE_CAPTURED/NONE` with
   `safe_provenance_verified=true`. The diagnostic then terminates; there is
   no retry, refetch, T selection, or downstream bridge.

## Closure and later review requirements

The implementation task that follows this design must provide synthetic
offline tests for every state, exact binding mismatch, lock corruption,
unsupported format, truncation, projection digest/key/type violation, and
unexpected exception. Tests must assert zero network calls, zero durable-state
writes, no private path/URL leakage, absence of T/Phase-2/F2 fields, and
deterministic canonical output. This design task itself performs no payload
read, diagnostic execution, network request, or human-gate consumption.
