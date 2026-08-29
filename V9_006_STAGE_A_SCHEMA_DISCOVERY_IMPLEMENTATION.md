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

Phase-1 plumbing now exposes only two reviewed V9_005 seams:
`acquire_f1_terminal_evidence` preserves F1's discovery-root to
`extract_data_j_xls_url` to TERMINAL raw-lock sequence and returns only the
TERMINAL slot ID plus attempts; `read_locked_payload_by_slot_id` reads one
existing exact canonical lock pair and fails closed on absence or malformed
provenance. Neither seam parses terminal month T, constructs schema profiles,
or creates Phase-1 execution/gate authority. This narrow implementation is
awaiting GPT exact-SHA review.

The Phase-1 aggregate executor core now calls only the reviewed F1 terminal,
per-month F2/F4, F3 fan-out, F7 envelope, and exact-slot lock-reader seams
through injected fetcher/sleep/clock dependencies. It mechanically binds and
profiles exactly 341 evidence locks (F1 1, F2 BASE 108, F3 YEAR 9, F4 BASE
108, F7 115); support locks never become profile inputs. It returns safe
profiles, representative profiles, evidence IDs, and aggregate attempts only.
It creates no receipt, runner, PowerShell entrypoint, gate, or real network
execution and awaits GPT exact-SHA review.

The reviewed Python one-shot boundary now wraps that core with a fixed,
task-global `%LOCALAPPDATA%` receipt reader, exclusive OutputRoot creation,
atomic no-overwrite receipt publication before the core call, and a canonical
safe no-overwrite result. It is synthetic-testable only through injected
dependencies; it adds no CLI/PowerShell wrapper or production fetcher wiring
and awaits GPT exact-SHA review.

Its global receipt reader now uses fail-closed `lstat` inspection before and
after a regular-file read: every symlink (including dangling), stat/read/type/
schema uncertainty, or entry change is unknown and blocks before OutputRoot
creation.

The success closure derives raw-lock IDs with exact suffix-aware `.bin` and
`.json` matching, requires equal 353-ID sets and exactly 706 entries, and
rejects malformed or extra filenames.

Synthetic Phase-1 boundary coverage now executes that real closure validator
directly and through the real one-shot boundary using one exact canonical
341-evidence/12-support/353-pair fixture. It also proves fail-closed behavior
for a missing member, wrong support identity, unrepresented evidence ID, and
corrupt provenance; all acquisition inputs remain synthetic and no production
source, network, receipt, or human gate is used. This MEDIUM_2 remediation
awaits GPT exact-SHA review; the deferred gate-docstring LOW_1 is unchanged.
