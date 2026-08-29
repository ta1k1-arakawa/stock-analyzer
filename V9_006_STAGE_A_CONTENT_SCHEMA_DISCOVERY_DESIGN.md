# Stage-A content schema discovery design

Status: `BLOCKED_PENDING_MEDIUM_3`. Public locked-payload structure is
development plumbing only. The closed scope is F1, F2, F3, F4, and F7; F5 and
F6 are excluded. Core profiling accepts only hash-verified lock objects and
has no fetcher, URL discovery, filesystem traversal, or network capability.

Detection is magic-based (`OLE_BIFF`, `OOXML_ZIP`, `HTML`, `PDF`, `UNKNOWN`).
PDF/unknown/OOXML require follow-up. OLE and HTML profiles emit bounded safe
schema neighborhoods and a structural fingerprint that excludes sampled text.
Future acquisition requires the distinct one-shot identity
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT`; current batch
orchestration is deliberately fail-closed because reviewed helpers do not yet
define the aggregate terminal-to-bridge invocation boundary. The identity is
documented only: it is not an argv, environment, prompt, receipt, or other
consumable authority, and this task performs no acquisition.

## Canonical raw-lock input

`VerifiedLockedObject` is accepted only after mechanical validation of the
complete canonical raw-lock contract: exact `V9_005_STAGE_A_RAW_LOCK_V1`, an
allowed F1/F2/F3/F4/F7 family, exact nonempty string period, canonical
JPX-requested and resolved URLs using the existing V9_005 validator, the
existing canonical raw-lock timestamp validator, an exact non-bool integer
HTTP status in 100..599, exact non-bool nonnegative byte length equal to raw
bytes, a lowercase 64-hex SHA-256 equal to those bytes, and a lowercase
64-hex slot ID exactly derived by the existing
`source_object_slot_id(family, period, requested_url)` algorithm. Raw bytes
are exact `bytes`. Any malformed or arbitrary/unhashable field fails closed
as `IMPLEMENTATION_FAILURE`; the dataclass does not itself establish trust.

## Closed object domains and representatives

The explicit domains are `TERMINAL`, `BASE`, `BRIDGE`, `ENVELOPE_EXTRA`, and
`YEAR`. Valid pairs are F1/TERMINAL, F2/BASE or BRIDGE, F3/YEAR, F4/BASE, and
F7/BASE or ENVELOPE_EXTRA; every other pair fails closed. MONTHLY domains use
canonical `YYYY-MM`; F3/YEAR is exactly one of 2017..2025. Content never
determines a domain or a representative.

F1 terminal and every F3 YEAR object are selected. F2 BRIDGE objects are all
selected. For F2/F4 BASE, per family/calendar year select earliest, latest,
and every profile differing from that year's earliest profile. For F7, apply
the identical annual rule across BASE plus ENVELOPE_EXTRA together. Earliest
and latest are always retained even when profiles match. Inputs must have
unique slot IDs and output is unique and ordered by family, period, slot ID.
Only `structural_profile_sha256` is used beyond identity/period/domain; no
title, headings, headers, samples, values, parser outcome, price, outcome, or
model information affects selection.

## Deferred boundary

`MEDIUM_3=DEFERRED_TO_NEXT_SUBSTANTIVE_REMEDIATION`: OLE/HTML full profiler,
closed safe-output validator, and comprehensive profiler tests remain
unresolved. Overall Stage A remains BLOCK until that distinct implementation
and GPT exact-SHA review complete.
