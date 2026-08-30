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
is PASS following exact-SHA review. The Phase-1 core and Python one-shot
boundary are reviewed; production CLI/PowerShell real-execution wiring remains
intentionally unavailable. `prepare_future_acquisition()` remains a runtime
fail-closed placeholder for unauthorized or unreviewed future execution wiring;
its documentation-only LOW_1 remediation awaits GPT exact-SHA review.

The Phase-1 Python CLI now binds the exact existing V9_005 production JPX
fetcher and UTC clock to the reviewed one-shot boundary. Its confirmation is
environment-only, never argv; stdout is limited to safe counts/statuses and a
slot-ID digest. Dependency seams keep its tests synthetic, including a real
one-shot/closure success path. Production real execution and its PowerShell
entrypoint remain unauthorized and unimplemented; this CLI layer awaits GPT
exact-SHA review.

The reviewed atomic Windows PowerShell Phase-1 real-execution entrypoint
(`scripts/run_v9_006_stage_a_schema_discovery_phase1_real_execution.ps1`) now
wraps that reviewed CLI. It derives the repository root mechanically from its
own script location, verifies the authoritative branch, an exact
40-lowercase-hex reviewed execution SHA, clean working tree, local/remote HEAD
equality, and required reviewed file presence; verifies a fresh OutputRoot,
the canonical `.venv-real-execution` interpreter, and complete no-network
readiness through the existing `scripts/check_real_execution_env.py` checker;
and verifies the canonical task-global receipt is mechanically proven absent
solely through the exact reviewed Python reader, never a PowerShell
reimplementation of that authority. Only after every pre-authorization check
passes does it request a fresh point-of-use human confirmation that is never
a script parameter; after confirmation it reruns every applicable
non-destructive binding before setting the confirmation only in the process
environment, immediately before the single invocation of the reviewed CLI,
always clearing it in `finally`. This entrypoint performs zero JPX/Yahoo
requests itself, creates no OutputRoot or receipt, and contains no retry
path; it awaits GPT exact-SHA review.

GPT-5.6 Sol exact-SHA review of `556897adbd90cb820f84aacad3da51e09e04d19b`
(parent `6f26a834d64925e73743d7ead8d0ff33f7c56c35`) is `CRITICAL=0`, `HIGH=0`,
`MEDIUM=1`, `LOW=0`, `RESULT=BLOCK`:
`MEDIUM_1=POST_CONFIRMATION_AUTHORITATIVE_BRANCH_BINDING_NOT_REVERIFIED` --
the post-confirmation/pre-consumption block rechecked working tree, local and
remote HEAD, OutputRoot freshness, the canonical interpreter, environment
readiness, and receipt absence, but never rechecked the authoritative branch
itself. MEDIUM_1 is remediated: the post-confirmation block now also
rereads `git branch --show-current`, requires it to equal exactly
`v9-cross-sectional-close-auction-design`, and mechanically reuses the
existing required-reviewed-file presence loop -- both strictly before the
confirmation environment variable is set and before the Python CLI is
invoked. A mismatch or uncertainty stops as `PRE_GATE`, without setting the
confirmation environment variable and without invoking Python acquisition.
No confirmation contract, receipt semantics, OutputRoot semantics, expected-
SHA semantics, environment-readiness semantics, retry behavior, Python
invocation, Python source/CLI, or methodology changed. This remediation
awaits GPT exact-SHA review.

GPT-5.6 Sol exact-SHA review of `26d0ecf284e85325f3a9f107356cdce5604294e0`
(parent `556897adbd90cb820f84aacad3da51e09e04d19b`) is `CRITICAL=0`, `HIGH=1`,
`MEDIUM=0`, `LOW=0`, `RESULT=BLOCK`; MEDIUM_1 is resolved. `HIGH_1=PRE_GATE_
REAL_EXECUTION_ENVIRONMENT_FREEZE_NOT_ENFORCED`: `scripts/check_real_
execution_env.py` computes `REAL_EXECUTION_ENVIRONMENT_READY` and
`REAL_EXECUTION_ENVIRONMENT_FROZEN` separately, and its process exit code
depends only on `READY`; the entrypoint checked only that exit code, so a
`READY=true`/`FROZEN=false` environment could pass the PowerShell
pre-authorization check and reach human authorization.

HIGH_1 is remediated: a single new helper, `Assert-RealExecutionEnvironmentFrozen`,
invokes the exact existing `scripts\check_real_execution_env.py` via the
canonical `.venv-real-execution\Scripts\python.exe`, captures its JSON output
only into a local variable (never printing the full JSON, which can carry
local paths in nested detail fields), and fails closed unless the checker's
process exit code is `0` AND its JSON parses AND
`REAL_EXECUTION_ENVIRONMENT_READY` is exactly boolean `true` AND
`REAL_EXECUTION_ENVIRONMENT_FROZEN` is exactly boolean `true` AND
`ENVIRONMENT_FREEZE_CHECK` is exactly the string `"PASS"` AND
`ENVIRONMENT_LOCK_FINGERPRINT_STATUS` is exactly the string `"FROZEN"`; on
success it prints only those four safe status fields. Both the
pre-authorization environment check and the post-confirmation/pre-consumption
recheck now call this same helper -- reused rather than duplicated, so the
predicates cannot drift between the two call sites -- and the post-
confirmation call is proven, by line order, to occur after human confirmation
and strictly before the confirmation environment variable is set and before
the Python CLI is invoked; any failure clears the in-memory confirmation
token before rethrowing, without setting the confirmation environment
variable or invoking Python acquisition. The MEDIUM_1 branch recheck remains
intact and unchanged. `scripts/check_real_execution_env.py`, the Python
acquisition CLI/source, the confirmation contract, receipt semantics,
OutputRoot semantics, expected-SHA semantics, retry behavior, and source-
family/count methodology are unchanged; there is still exactly one
acquisition-CLI invocation and no retry path. This remediation awaits GPT
exact-SHA review.

The Phase-1 CLI failure report now treats the canonical gate reader as the
authority for post-gate reporting: when a `V9005StageABlocked` failure occurs
and that reader mechanically returns `true`, it reports
`gate_consumed=true` and `network_attempt_count="unknown"`, never the
exception-local count. Mechanically proven pre-gate failures retain their
exact zero count. A synthetic CLI regression runs the real one-shot/core
path with an isolated temporary receipt root and one injected F1 HTTP-200
payload that lacks the frozen `data_j.xls` locator; it proves the terminal
source/data feasibility failure, one synthetic fetch, and unknown post-gate
attempt count without real network. Success reporting and every acquisition,
locator, retry, receipt, confirmation, OutputRoot, and gate semantic remain
unchanged. This bounded reporting remediation awaits GPT exact-SHA review.
