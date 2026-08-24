# V9_006 Stage-A implementation

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
task=V9_006_STAGE_A_PUBLIC_JPX_PROBE_IMPLEMENTATION
network_executed=false
network_authorized_by_this_task=false
t1_or_design_freeze_authority_created=false
```

This implements, without executing any real network request, the reviewed
V9_005 Stage-A free official JPX metadata/calendar probe defined in
`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` (independently
reviewed PASS: `V9_005_HIGH_2B_REVIEW.md`, reviewed SHA
`137e6ba50b916720adeef66f09049010185534d8`). It creates a durable,
fail-closed implementation suitable for later direct Windows PowerShell
execution under a fresh, separately obtained Stage-A human network
authorization -- it does not consume the authorization a human already gave
in chat, and it makes zero real network requests itself.

## Files

- `src/v9_005_stage_a_jpx_probe.py` -- Stage-A logic: off-domain rejection,
  first-complete-payload raw locking with full provenance, the deterministic
  monthly `SOURCE_INVENTORY`, the reconstruction/validation evidence items,
  the exact `FREE_JPX_METADATA_PROBE_PASS` conjunction, the V9_005_HIGH_2B
  signal-grid blob-binding point-of-use check, and the mechanical
  `FINAL_SIGNAL_D0` / `STAGE_B_GLOBAL_END_EXCLUSIVE` derivation.
- `scripts/run_v9_005_stage_a_jpx_probe.py` -- production entrypoint; reads
  the Stage-A confirmation token only from the
  `V9_005_STAGE_A_CONFIRMATION` environment variable (never argv, never
  hardcoded), and prints only the safe JSON summary.
- `scripts/run_v9_005_stage_a_jpx_probe.ps1` -- single atomic `& { ... }`
  Windows entrypoint per `AI_REAL_EXECUTION_RUNBOOK.md` SS1. Performs the
  full preflight (repository, authoritative branch, exact expected local
  HEAD, exact remote HEAD, clean tree, the V9_005_HIGH_2B design-blob
  binding, output-root collision check, and an interactively typed Stage-A
  confirmation token) before any network request; no chat-supplied
  authorization is baked into the file.
- `tests/test_v9_005_stage_a_jpx_probe.py` -- 58 tests, entirely offline
  (synthetic fixtures and injected fake fetchers/git callables only).

## Source-locator discipline (contract item 4)

No JPX endpoint URL in this implementation was guessed. Exactly two
concrete endpoints are reused verbatim from already-reviewed repository
evidence/code:

- the listed-issues page (`https://www.jpx.co.jp/markets/statistics-equities/misc/01.html`)
  and its `data_j.xls` link-extraction pattern, reused from
  `src/v8k_public_source_preparation.py` / `scripts/build_v8_partition_manifest.py`;
- the JPX Calendar page (`https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html`),
  reused from `src/v7_jpx_calendar.py`.

`V9_004_EXTERNAL_SOURCE_EVIDENCE.md` and
`V9_005_HIGH_2_EXTERNAL_SOURCE_EVIDENCE.md` describe the *existence* of five
other source families (Monthly Statistics Report, delisted-company archive,
split/right-treatment archive, monthly aggregate counts, TOPIX historical
index) in prose, but no reviewed repository evidence supplies their exact
per-month archive URLs. Per the source-locator discipline, inventing a URL
pattern for those families would itself be a new methodology decision, so
this implementation does not attempt it. `resolve_month_locator` is the
single documented seam a future, separately reviewed task would extend with
additional reviewed per-month locators; it returns no locator for any
family/month today.

**V9_006_HIGH_1 remediation.** The original implementation converted that
universal "no locator" condition into a real fetch-and-FAIL run: it still
crossed the JPX network boundary (fetching the two non-monthly artifacts
that do have a locator) before computing a guaranteed `FREE_JPX_METADATA_
PROBE_FAIL`. GPT's review correctly identified this as unacceptable: a
knowingly doomed real-network run is not a substitute for stopping before
the boundary, and `V9_004_FREE_DATA_SOURCE_FEASIBILITY_AUDIT.md`'s own
`FREE_PIT_UNIVERSE_FEASIBILITY=PARTIAL_NOT_PROVEN` conclusion means the
locator contract for the seven required source families/monthly slots was
never actually complete in the first place. `run_stage_a` now calls
`verify_locator_contract_complete()` as its very first step -- before
touching the filesystem, git, or the network -- which raises
`V9005StageABlocked(STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE)`
(`failure_class=CHATGPT_DECISION_REQUIRED`, never `SOURCE_OR_DATA_
FEASIBILITY_FAILURE`) whenever any required family/month cell lacks a
mechanically resolvable locator. Under current reviewed evidence every cell
lacks one, so real execution today stops immediately: zero fetch calls,
zero git calls, and no output-root directory is even created. See
`V9_006_HIGH_1_REVIEW.md` for the full finding and remediation record.

`SOURCE_OR_DATA_FEASIBILITY_FAILURE` remains reserved for a genuine result
produced only after the locator contract is complete and the actual
approved source probe has run -- it is exercised in tests only with the
locator-contract gate forced complete via monkeypatch, simulating a future,
separately reviewed extension (`test_run_stage_a_offline_reports_fail_with_
safe_evidence_once_contract_forced_complete`). In that forced scenario, the
single locked calendar page also only covers 2026-2027 (the same years
`src/v7_jpx_calendar.py` already handles), which is insufficient to
mechanically derive `FINAL_SIGNAL_D0` (requires coverage back to
2018-01-01); `derive_stage_b_global_end_exclusive` fails closed with
`SOURCE_OR_DATA_FEASIBILITY_FAILURE` in that case rather than compute a
wrong index from a narrower window.

## Signal-grid binding (contract item 5)

`verify_signal_grid_binding` is called immediately after output-root
initialization and before any Stage-A network request. It re-derives
`V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md`'s exact Git blob SHA at
the current HEAD and compares it to the bound reviewed value
`9135183b7fc5097602fa40fcda8f1b0448220244`
(`BOUND_SIGNAL_GRID_BLOB_SHA`). A mismatch raises
`V9005StageABlocked(PROBE_SIGNAL_GRID_CONTRACT_MISMATCH)` before any
network call; the `.ps1` entrypoint performs the identical check with `git
rev-parse` before its own network step. `FINAL_SIGNAL_D0` is derived
mechanically from whatever JPX trading-day calendar was actually locked --
never hard-coded -- using exactly the bound rule: `j0` = calendar index of
the first JPX trading day `>= 2018-01-01`; a date at index `j` is a V9
signal-grid `D0` iff `(j - j0) mod 3 == 0`; `FINAL_SIGNAL_D0` = the last such
`D0 <= 2025-12-31`.

## Raw content locking (contract item 3)

`lock_first_complete_payload` persists raw bytes and a full-provenance JSON
record (requested URL, resolved URL, HTTP status, retrieval timestamp, byte
length, SHA-256, source family, applicable period) atomically via
hard-link-based no-overwrite creation (the same technique already reviewed
in `src/v8k_public_source_preparation.py`); a second attempt at the same key
fails closed rather than overwriting. `ensure_locked_payload` always checks
for an existing lock before fetching, so a parser/semantic repair path
reprocesses the exact same locked bytes and never triggers a second network
request. Transport failures are classified only via the already-reviewed
`src/v8c_transport.classify_transport_exception`, retried up to the frozen
attempt/backoff policy, and unresolved retryable failures surface as
`PLUMBING_FAILURE_RETRIABLE` per `AI_REAL_EXECUTION_RUNBOOK.md`.

## Off-domain rejection (contract item 1)

`validate_jpx_url` accepts only `https://jpx.co.jp` or an exact subdomain of
it, with no credentials, no nonstandard port, and no fragment.
`fetch_once_with_retry` applies this check to both the requested URL (before
any request) and the response's final/resolved URL (before any content is
consumed), so an off-domain redirect is rejected before its body is ever
read.

## Durable output and atomic PowerShell preflight (contract items 6-7)

`initialize_output_root` requires a fresh, not-yet-existing output
directory (a durable-execution collision check per
`AI_REAL_EXECUTION_RUNBOOK.md` SS8); it is never silently reused or
overwritten. `scripts/run_v9_005_stage_a_jpx_probe.ps1` is a single
`& { $ErrorActionPreference = "Stop" ... }` scope that verifies, in order,
the repository, authoritative branch, clean tree, exact expected local
HEAD, exact remote HEAD, the design-blob binding, the output-root collision
check, and an interactively typed Stage-A confirmation token, before running
the canonical `.venv-real-execution` interpreter against the Python
entrypoint; the confirmation token is cleared from the process environment
in a `finally` block even on failure, and is never written to disk, logged,
or embedded in this file.

## What this task does not do

- No real network request was made (`NETWORK_REQUESTS=0`).
- The human's existing chat-given Stage-A authorization was not consumed or
  referenced by any code path.
- No JPX/Yahoo/J-Quants/broker data was acquired; no price or outcome was
  inspected.
- No T1 partition was generated or opened; no model was fit; no backtest
  ran; no V9 design freeze occurred; no Stage B artifact was produced.
- `V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` was not
  modified; none of its thresholds, source families, retry rules, dates, or
  corporate-action semantics changed.

## Next action

`GPT_EXACT_SHA_V9_006_HIGH_1_REVIEW`: obtain GPT's independent exact-SHA
review of this HIGH-1 remediation commit before any real Stage-A execution.
Real execution additionally requires: this review's PASS (and PASS of any
other still-open V9_006 findings); the environment readiness ordering in
`AI_REAL_EXECUTION_RUNBOOK.md` SS16-19; and a fresh, separate, explicit
point-of-use human network authorization obtained after that review PASS
(not the authorization already given in chat, which this task did not
consume).
