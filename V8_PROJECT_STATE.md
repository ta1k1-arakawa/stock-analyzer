# V8_PROJECT_STATE

Human-readable current-state document for `V8_HISTORICAL_RESEARCH`. This
file is navigation and status; it is not authoritative for frozen design
values (see `V8_HISTORICAL_RESEARCH_DESIGN.md`) or for exposure
classifications (see `V8_DATA_EXPOSURE_AUDIT.md`). For a machine-readable
equivalent of this file, see `V8_STATE.json`.

If this file and `V8_STATE.json` ever disagree, treat that as a documentation
bug to fix, not as license to guess which one is right — re-derive both from
the actual repository state at the current remote HEAD.

## Current phase

```text
SOURCE_SNAPSHOT_CLARIFICATION_APPROVED_IMPLEMENTATION_PENDING
```

The independent source-preflight review passed
(`SOURCE_PREFLIGHT_REVIEW_PASS`), and the human-authorized real JPX
source-only preflight was run exactly once. It returned
`status=BLOCKED, reason=V8_PARTITION_SOURCE_NOT_REPRODUCIBLE, exit_code=2`
because the currently-served JPX raw bytes did not hash-match the `V4`
2026-08-03 `raw_file_sha256`; no retry was performed. A human design gate then
resolved the underlying ambiguity: `V8_HISTORICAL_RESEARCH_DESIGN.md` §16
(append-only, 2026-08-10) clarifies that the V8 partition may use an official
JPX snapshot fetched at partition-implementation time, and does not require
that snapshot's raw bytes to equal `V4`'s 2026-08-03 raw bytes — while `T0`
exact reproduction against the already-frozen `V4_UNIVERSE.csv` remains
mandatory. This clarification authorizes **no** real network action by
itself; implementing the corresponding code change is the next step, still
followed by its own independent review and its own real-network human gate.

## Completed milestones

| # | Milestone | Commit | Verdict |
|---|---|---|---|
| 1 | V4–V7 data exposure audit (read-only) | part of `c414d3191c` lineage | `V8_DATA_EXPOSURE_AUDIT.md` complete, `fresh_certified=0` |
| 2 | V8 research design drafted | `3bc502d8d6b554822aa98b946947e4a6730603f2` | `DRAFT_AWAITING_HUMAN_GATE` |
| 3 | V8 design human review — 10 binding decisions applied, design frozen | `c414d3191cba356734d7ed08bdf1abc7d51fc384` | `HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION` |
| 4 | Design erratum: T0×P_hist does overlap V7 feature-seed sub-span (append-only correction, no frozen parameter changed) | `c5848ced1a5c800f384cb7b86fb642e5c748c2c2` | corrected, does not reopen the design gate |
| 5 | Partition manifest builder + raw historical acquisition module, both synthetic-only, 106 new tests | `c5848ced1a5c800f384cb7b86fb642e5c748c2c2` | `V8_PARTITION_ACQUISITION_STATIC_PASS` |
| 6 | Fail-closed production partition-manifest CLI | `23667bb855db405cf488755f0f166d91d8f75f32` | implemented; fake-only tests, no real request or manifest |
| 7 | Acquisition binding to validated partition manifest + implementation provenance | `aea2cb40efaf15bb749ee8545b021d65c2c52821` | Finding 2 resolved; 136 V8 tests passing |
| 8 | Fail-closed production acquisition CLI/runner | `53c951d4e0dfc9cce92e38a223d74636406c6cce` | Finding 1 resolved; 149 V8 tests passing, fake-only |
| 9 | First critical-review remediation: trusted partition anchor, GitHub-ref provenance, JPX preflight/redirect hardening, atomic no-overwrite publication | `53556e7fdbaf6f08d72fc216122a25a475dd6c7c` | implementation pending independent retest; fake-only |
| 10 | Second critical-review remediation: fixed public acquisition boundary, Git-HEAD anchor bytes, fixed dates, production partition metadata checks, and exact JPX/Yahoo origins | `172f35d1fa747ffb4acb006a0f59c36700cd53a3` | implementation pending independent retest; 199 fake-only tests passing |
| 11 | Partition production public-boundary remediation: public runner accepts only `output_path`; fake opener/parser/V4/clock dependencies are private test seams | `297cb8aa599a74bd9a09953ce7acae10c9cfec95` | `CLOSED_PENDING_REVIEW`; 206 fake-only V8 tests passing |
| 12 | Source-only JPX/T0 reproduction preflight with closed public boundary; no allocation or publication | `38697c9ede51cac7bd500206d857ee585464996b` | `SOURCE_REPRODUCTION_PREFLIGHT_IMPLEMENTED_PENDING_REVIEW`; 226 fake-only V8 tests passing |
| 13 | Independent source-preflight review passed; one human-authorized real JPX source-only preflight run (BLOCKED: `V8_PARTITION_SOURCE_NOT_REPRODUCIBLE`, no retry); source-snapshot ambiguity resolved by human design gate | `eb13eb6cad4d0f5a920929cf0eaf97d1f673743d` | `SOURCE_PREFLIGHT_REVIEW_PASS`; append-only design clarification recorded, `V8_HISTORICAL_RESEARCH_DESIGN.md` §16 |

## Human approvals

| Gate | Status |
|---|---|
| New strategy family (`V8_HISTORICAL_RESEARCH`) accepted | GRANTED — recorded in `V8_HISTORICAL_RESEARCH_DESIGN.md` §13 |
| Design frozen (10 decisions: Layer A reconciliation, block sizes, `P_early` deferred, Layer B access=1, Layer C one-candidate, `T2` sealed holdout scope, walk-forward split scheme, friction grid, Layer A promotion thresholds, survivorship-bias wording) | GRANTED — `V8_HISTORICAL_RESEARCH_DESIGN.md` §1 |
| `V8_T1_T2_ACQUISITION_AND_PARTITION_APPROVED` (build the partition/acquisition **code**, still no real network) | GRANTED |
| Real JPX source fetch | **NOT GRANTED** |
| Real JPX source-only preflight | One-time authorization **GRANTED and consumed** (2026-08-10) — result `BLOCKED/V8_PARTITION_SOURCE_NOT_REPRODUCIBLE`, no retry performed; a fresh authorization would be required for any further attempt |
| Source-snapshot semantics clarification (append-only design gate) | **GRANTED** (2026-08-10) — `V8_HISTORICAL_RESEARCH_DESIGN.md` §16; does not itself authorize any real network action |
| Real T1 acquisition | **NOT GRANTED** |
| Real T2 acquisition | **NOT GRANTED** |
| T3 acquisition of any kind | **NOT GRANTED** (and the code path unconditionally rejects it regardless of any future gate wording short of a design amendment) |
| Layer A search, Layer B validation, Layer C evaluation | **NOT GRANTED** — not implemented at all yet |
| Prospective forward study | **NOT GRANTED** — separate future study |
| Real-money deployment | **NOT GRANTED** — separate future human gate, downstream of everything above |

## Git provenance

```text
canonical_v7_branch = v7-forward-capacity-gate3-dry-run
canonical_v7_branch_sha = fec1b85c2e6deb89b8c5d4fa31ff1ae58a62edbc   (unchanged by every V8 phase so far)

v8_design_branch = v8-historical-research-design
v8_design_frozen_commit = c414d3191cba356734d7ed08bdf1abc7d51fc384

v8_implementation_branch = v8-partition-acquisition
v8_pre_remediation_implementation_commit = 53c951d4e0dfc9cce92e38a223d74636406c6cce
v8_current_remediation_state = 38697c9ede51cac7bd500206d857ee585464996b (source-only JPX/T0 preflight; verify current remote HEAD before acting)
```

Verify current remote state with:

```text
git ls-remote origin v8-partition-acquisition
git ls-remote origin v7-forward-capacity-gate3-dry-run
```

## Current implementation

| File | Role |
|---|---|
| `src/v8_partition.py` | Reconstructs the eligible JPX universe, proves official-source and `T0` reproduction, records `partition_implementation_git_commit`, and atomically publishes a self-hash-verified manifest without replacement. Never imports any V7 module. |
| `src/v8_historical_acquisition.py` | Raw-only OHLCV acquisition for `T1`/`T2` only. Its public production boundary accepts only output root, block, and persisted manifest path. Before transport it requires clean `HEAD == origin`, reads `V8_TRUSTED_PARTITION.json` bytes from that verified Git object, requires authorization, exact manifest/provenance/production-JPX metadata, identity/300-ticker/hash binding, fixed historical dates, and a strict Yahoo-origin opener. |
| `scripts/build_v8_partition_manifest.py` | Synthetic CLI plus `--production-build-manifest` and `--production-source-preflight`. The source-only public runner accepts no inputs and fixes JPX transport, parser, V4 paths, UTC clock, repository root, and Git resolver internally; it stops after source/T0 reproduction and cannot allocate or publish. Fake dependencies are available only through private test seams. Neither production mode has been invoked with real JPX. |
| `scripts/acquire_v8_historical.py` | Synthetic CLI plus implemented `--production-acquire` path. Production mode accepts only block, persisted partition manifest, private output root, and block-specific confirmation; neither CLI nor runner exposes transport, date, Git, repository-root, or trust-anchor overrides. |
| `tests/test_v8_partition.py`, `tests/test_v8_partition_cli.py`, `tests/test_v8_historical_acquisition.py`, `tests/test_v8_historical_acquisition_cli.py` | 226 fake-only tests passing after the source-only preflight implementation. Zero real JPX/Yahoo calls anywhere in the suite. |

## Data state

```text
real_partition_manifest_exists = false
real_jpx_source_fetched = true (one preflight attempt, 2026-08-10; result BLOCKED, no partition/manifest/allocation followed)
real_jpx_source_preflight_executed = true (attempt_count=1; result=BLOCKED/V8_PARTITION_SOURCE_NOT_REPRODUCIBLE; retry_performed=false)
private_v8_storage_location = NOT_YET_DEFINED
requirements = absolute path; outside this repository; never committed
trusted_partition_authorization = false
partition_public_dependency_injection = CLOSED_PENDING_REVIEW
source_snapshot_clarification = IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT (V8_HISTORICAL_RESEARCH_DESIGN.md §16, 2026-08-10; V4 raw SHA equality not required; T0 exact reproduction still required)
```

## T1 state

```text
role = VALIDATION
raw_data_acquired = false
validation_access_count = N/A (no manifest exists yet)
layer_B_opened = false
```

## T2 state

```text
role = SEALED_HOLDOUT
raw_data_acquired = false
sealed = N/A (no manifest exists yet; when acquired, sealed=true by construction)
research_access_authorized = N/A (would be false by construction)
opened_for_research = false
guard_verified_in_tests = true (5/5 official entry points BLOCK on a sealed T2 manifest, synthetic fixtures only)
```

## T3 state

```text
role = SEALED_RESERVE
raw_data_acquired = false
acquisition_code_path = unconditionally rejects any T3 request (V8_BLOCK_ACQUISITION_PROHIBITED)
release_requires = separate future human gate (not requested, not granted)
```

## V7 isolation

```text
v7_code_modified_by_any_v8_phase = false
v7_artifacts_modified_by_any_v8_phase = false
v7_forward_observations_read_by_v8 = false
v7_interim_results_used_in_v8_selection = false
v8_results_used_to_alter_v7 = false
v7_module_reused_by_v8 = src.v7_yahoo_collector only (read-only import: fetch_chart_once, canonical_ticker, FRAME_FIELDS, HOST, V7YahooCollectorBlocked)
canonical_v7_branch_sha_unchanged_across_all_v8_commits = true
```

The only V7 code V8 touches, anywhere, is a plain read-only Python `import`
of the already-accepted, generic Yahoo Chart transport in
`src/v7_yahoo_collector.py`. No V7 file has been edited by any V8 commit.

## Current pre-production blockers and next action

### Previous critical review and Finding 1

The production partition-manifest runner is implemented at
`23667bb855db405cf488755f0f166d91d8f75f32`; it preserves the existing
fail-closed source and T0 reproduction guards. The production acquisition
runner is implemented at `53c951d4e0dfc9cce92e38a223d74636406c6cce`, then
further hardened at `172f35d1fa747ffb4acb006a0f59c36700cd53a3`. Two
independent critical reviews have BLOCKed the combined production path. The
second remediation is implemented but Finding 1 must not be treated as
review-passed until `INDEPENDENT_CRITICAL_REVIEW_RETEST` completes. Neither
runner has been used against a real service, and no real manifest exists.

### Finding 2 — remediated pending independent retest

At `aea2cb40efaf15bb749ee8545b021d65c2c52821`, the public acquisition path:

1. reads the canonical Git-tracked `V8_TRUSTED_PARTITION.json`, verifies its
   identity, and requires `authorization_status=AUTHORIZED`;
2. reads the persisted partition manifest with `read_partition_manifest()`
   and its self-hash verification;
3. requires its manifest SHA and `partition_implementation_git_commit` to
   exactly match the trusted anchor;
4. verifies `schema_version`, `study_name`, and `design_commit`;
5. permits only `T1` and `T2`, with `T3` unconditionally blocked;
6. sources tickers solely from the verified `block_assignments[block]`;
7. requires exactly 300 tickers and verifies the authoritative block
   ticker-list SHA256; and
8. derives `partition_manifest_sha256` and records validated acquisition
   `implementation_git_commit` provenance.

Every failed Git, anchor, binding, provenance, or exact-origin check blocks
before transport. Before any future real production command, an operator must
run `git fetch origin` successfully and record the fetched remote SHA and
local `HEAD`; code then verifies the local clean `HEAD == origin` state. The
current fake-only V8 regression is 226 passed / 0 failed.

## Current ordered next steps

1. ~~Obtain an independent review of the source-only production preflight.~~ **Done** — `SOURCE_PREFLIGHT_REVIEW_PASS`.
2. ~~Obtain human authorization for exactly one real JPX source-only preflight.~~ **Done** — authorized 2026-08-10.
3. ~~Run the source-only preflight.~~ **Done** — one attempt, `BLOCKED/V8_PARTITION_SOURCE_NOT_REPRODUCIBLE`, no retry.
4. ~~Inspect and report the source-only preflight result; resolve the raw-SHA-vs-design ambiguity.~~ **Done** — `V8_HISTORICAL_RESEARCH_DESIGN.md` §16, human clarification `IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT`.
5. Implement the source-snapshot semantics in `src/v8_partition.py` /
   `scripts/build_v8_partition_manifest.py`: the source-only and full-build
   paths must stop requiring `source_raw_sha256 == V4 raw_file_sha256`,
   while continuing to require exact `T0` reproduction and to fix the new
   snapshot's own provenance (`eligible_ticker_list_sha256`, byte count,
   acquisition UTC, etc.) into the result/manifest. **Not done yet — code
   unchanged by this handoff.**
6. Obtain an independent review of that code change.
7. Obtain a fresh human authorization for the next real JPX source-only
   preflight attempt under the new semantics.
8. Obtain separate human authorization for real partition creation.
9. Create the real partition manifest.
10. Verify manifest SHA and partition implementation commit.
11. Obtain a separate human gate to Git-pin the trust anchor.
12. Obtain separate T1 authorization.
13. Obtain later separate T2 authorization.

None of these steps is authorized by this documentation update beyond step 4,
which is already recorded as done. This handoff makes no code change (step 5
remains open) and authorizes no real network action, no partition creation,
no trust-anchor authorization, and no `T1`/`T2` acquisition. Actual private
storage remains `NOT_YET_DEFINED`; when selected it must be an absolute path
outside this repository and must never be committed.

## Historical pre-production blockers at c5848ced1a5c800f384cb7b86fb642e5c748c2c2

Two findings, both confirmed against the actual code at
`c5848ced1a5c800f384cb7b86fb642e5c748c2c2`, must be resolved before any real
network call:

### Finding 1 — no production runner exists

`scripts/build_v8_partition_manifest.py` and `scripts/acquire_v8_historical.py`
each expose exactly one flag, `--synthetic-test`. Neither has a production
code path: no real JPX fetch, no real partition manifest creation, no real
Yahoo `T1`/`T2` acquisition. This is intentional for the static phase and
does not invalidate `V8_PARTITION_ACQUISITION_STATIC_PASS` — but production
acquisition cannot start until a production runner is implemented.

### Finding 2 — acquisition is not bound to a validated partition manifest (CRITICAL)

`acquire_historical_block_bundle()` in `src/v8_historical_acquisition.py`
accepts `block`, `tickers`, and `partition_manifest_sha256` as independent
caller-supplied arguments. It verifies transport/schema integrity of what it
fetches, but it does **not** itself:

- read an actual, persisted partition manifest via `read_partition_manifest()`,
- verify that manifest's own self-hash,
- verify `schema_version` / `study_name` / `design_commit` on that manifest,
- restrict `tickers` to that manifest's `block_assignments[block]`,
- verify the ticker list is exactly 300 and hashes to that manifest's
  `t1_ticker_list_sha256` / `t2_ticker_list_sha256`, or
- derive `partition_manifest_sha256` itself from that manifest, rather than
  accepting it as a free string.

Before any production network access, the production acquisition path must
fail-closed-bind to a real, validated partition manifest. Minimum required
checks, in order, before the first HTTP request:

1. `read_partition_manifest(path)` — loads and self-hash-verifies the actual manifest.
2. Confirm `schema_version`, `study_name`, `design_commit` match this
   implementation's expectations.
3. Confirm the requested block is exactly `"T1"` or `"T2"` (never `"T3"`).
4. Tickers are **not** accepted from CLI/user input at all — they are read
   **only** from `partition_manifest["block_assignments"][block]`.
5. `len(tickers) == 300` exactly.
6. The computed `ticker_list_sha256` over those tickers equals the
   manifest's corresponding `t1_ticker_list_sha256` / `t2_ticker_list_sha256`.
7. `acquisition_manifest["partition_manifest_sha256"]` is set from the
   actual, just-verified partition manifest's own `manifest_sha256` — never
   passed in freely.
8. `T3` continues to BLOCK unconditionally, regardless of any future gate
   wording short of a design amendment.

Any of these failing must result in `NETWORK_ACCESS_BEFORE_BLOCK = 0` —
i.e. fail closed before the first HTTP request is made, exactly as the
existing `V8_PARTITION_SOURCE_NOT_REPRODUCIBLE` and
`V8_T0_REPRODUCTION_MISMATCH` guards already do in `v8_partition.py`.

Additionally, the acquisition manifest schema should be extended (future
work, not done in this handoff) to record the exact V8 implementation git
commit that performed a real acquisition, as production provenance.

This handoff phase makes **no code change** to close this gap. It is
recorded here as the binding requirement for the next implementation phase.

## Historical next action at c5848ced1a5c800f384cb7b86fb642e5c748c2c2

```text
IMPLEMENT_PRODUCTION_PARTITION_AND_ACQUISITION_RUNNER
+ BIND_ACQUISITION_TO_VALIDATED_PARTITION_MANIFEST
+ ADD_IMPLEMENTATION_COMMIT_PROVENANCE
```

Concretely, in order:

1. Implement a production partition-manifest CLI path (real JPX fetch,
   still gated by the existing `V8_PARTITION_SOURCE_NOT_REPRODUCIBLE` /
   `V8_T0_REPRODUCTION_MISMATCH` guards, which are already correct and do
   not need to change).
2. Implement the eight-point acquisition-binding hardening in Finding 2
   above, inside `src/v8_historical_acquisition.py` or a thin wrapper
   around it — the existing transport/integrity logic does not need to
   change, only how `block`/`tickers`/`partition_manifest_sha256` are
   sourced.
3. Add implementation-commit provenance to the acquisition manifest schema.
4. Re-run synthetic/static verification against the hardened code.
5. A separate, independent critical review of the hardened production path
   (not self-review) before any real network call is authorized.
6. Only after that review passes: real JPX source reproduction (still no
   ticker acquisition — this only builds the real partition manifest).
7. Real partition creation (writes the real manifest to private storage
   outside this repository).
8. `T1` raw acquisition (real network, real Yahoo host, real 300-ticker
   fetch bound to the real manifest).
9. `T2` raw acquisition + procedural seal.

Each of steps 6–9 requires its own explicit human authorization; none is
implied by completing the step before it.

## Things that must NOT happen yet

```text
must_not_fetch_real_jpx_source = true
must_not_create_real_partition_manifest = true
must_not_acquire_real_T1_data = true
must_not_acquire_real_T2_data = true
must_not_acquire_T3_data_under_any_condition_this_phase = true
must_not_open_T2_for_research = true
must_not_run_layer_A_search = true
must_not_run_layer_B_validation = true
must_not_run_layer_C_evaluation = true
must_not_compute_features_candidates_or_profit = true
must_not_modify_any_V7_file = true
must_not_read_V7_forward_outcomes = true
must_not_commit_private_raw_data_to_this_public_repository = true
```

## Historical definition of next-action success

Success is: a production partition-manifest CLI and a production
acquisition CLI exist, are independently reviewed, fail closed on every
integrity check above (verified by tests using fake network fixtures, not
real calls), and **still make zero real network requests** during that
review. Passing this stage authorizes moving to step 6 above (real JPX
source reproduction) under separate explicit human sign-off — it does not
itself authorize any real network call.

## Historical downstream sequence

Once real partition creation and real `T1`/`T2` raw acquisition are
complete (Phases 4–5 in `V8_RUNBOOK.md`), the sequence continues: Layer A
historical research (non-evidential, Phase 6) → Layer B one-shot validation
(Phase 7) → exactly-one final candidate freeze (Phase 8) → `T2` one-shot
sealed holdout (Phase 9) → a separate prospective forward study (Phase 10)
→ a separate human deployment gate (Phase 11). See `V8_RUNBOOK.md` for the
per-phase contract.

## Source-of-truth priority

When any two of the following disagree, resolve in this order — do not
guess, do not average, do not defer to whichever is more recent in your own
context window:

1. **Immutable/raw acquisition manifests and their hashes**, once they exist
   (they do not exist yet as of this commit).
2. **`V8_STATE.json`** — current workflow state.
3. **`V8_PROJECT_STATE.md`** (this file) — narrative current state.
4. **`V8_HISTORICAL_RESEARCH_DESIGN.md`** — frozen study rules.
5. **`V8_DATA_EXPOSURE_AUDIT.md`** — historical exposure classification.
6. **Actual code and tests at a verified Git SHA.**
7. **Chat history / model memory** — lowest priority, never authoritative.

If GitHub state and a private manifest disagree, **BLOCK** — do not guess
which one is correct.
