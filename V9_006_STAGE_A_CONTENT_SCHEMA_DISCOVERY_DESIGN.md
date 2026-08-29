# Stage-A content schema discovery design

Status: `IMPLEMENTED_AWAITING_GPT_REVIEW`. Public locked-payload structure is
development plumbing only. The closed scope is F1, F2, F3, F4, and F7; F5 and
F6 are excluded. Core profiling accepts only hash-verified lock objects and
has no fetcher, URL discovery, filesystem traversal, or network capability.

Detection is magic-based (`OLE_BIFF`, `OOXML_ZIP`, `HTML`, `PDF`, `UNKNOWN`).
PDF/unknown/OOXML require follow-up. OLE and HTML profiles emit bounded safe
schema neighborhoods and a structural fingerprint that excludes sampled text.
Future acquisition requires the distinct one-shot identity
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT`; current batch
orchestration is deliberately fail-closed because reviewed helpers do not yet
define the aggregate terminal-to-bridge invocation boundary.
