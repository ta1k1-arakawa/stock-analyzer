# Stage-A schema discovery implementation

`src/v9_006_stage_a_schema_discovery.py` provides deterministic offline
profile generation from synthetic or future verified locks. It reuses the
V9_005 canonical URL, timestamp, and slot-ID checks; it never accepts an
arbitrary claimed slot ID. The explicit closed object-domain contract and
period/profile-only representative selector are implemented for F1/F2/F3/F4/
F7. The same `_validate_domain_period` seam is used by verified-lock and
profile-input validation, reusing V9_005 `TERMINAL_PERIOD`, `inventory_months`,
`calendar_envelope_extra_months`, canonical monthly parsing, and inventory
upper-bound constants. It records safe provenance, format, byte length, hash,
and a value-free structural hash. It does not bind parser mappings or execute
acquisition.

The runner accepts no argv/environment/prompt authorization and only calls
the fail-closed preparatory status entrypoint, which reports
`CHATGPT_DECISION_REQUIRED`. No authority is consumed. MEDIUM_3's full
OLE/HTML profiler and safe-output work remains deferred and this foundation
is overall BLOCK pending its separate remediation and review.

M3 now provides bounded OLE and HTML evidence plus `_validate_safe_profile`,
called by `profile_verified_lock` before output. The implementation is
offline-only and uses synthetic fake `xlrd` Book/Sheet tests; no dependency,
acquisition, raw production read, or semantic mapping was added.

The final M3 boundary revision removes shadowed legacy profiler definitions,
tracks all truncation at observation time, treats headings as atomic DOM
elements, rejects unfinished HTML state, and validates safe provenance and
sample ordering independently of the producer.
The final closure additionally fails closed on overlapping/unmatched HTML row
and cell state and rejects invalid profile status/container pairs.
The safe validator now also closes exact row/header coordinates, structural
attribute ordering, TEXT-key semantics, table-count lower bounds, and OLE
sheet-name truncation provenance.
