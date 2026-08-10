# V8_START_HERE

Read this file first. It is short by design — it exists so a new chat does
not have to read the full 1000+ line design document before it can act.

## Identity

```text
study = V8_HISTORICAL_RESEARCH
repository = ta1k1-arakawa/stock-analyzer
active_branch = v8-partition-acquisition
pre_remediation_baseline_commit = 53c951d4e0dfc9cce92e38a223d74636406c6cce
```

The baseline commit predates the critical-review remediation described below.
Always re-check the remote branch SHA yourself
(`git ls-remote origin v8-partition-acquisition`) before acting; the current
remote implementation, not a handoff SHA, is authoritative.

## Purpose

V8 is a historical strategy-discovery research line for Japanese equities,
run in parallel with, and fully independent from, `V7_FORWARD_CAPACITY`
(the forward-only paper study). V8 uses past data deliberately, but a data
exposure audit first established which historical spans and tickers were
already seen by V3–V7, so V8's confirmatory evidence is built only on
genuinely fresh cross-sections. Full rationale: `V8_HISTORICAL_RESEARCH_DESIGN.md`.

**V7 is a separate research line. Nothing in V8 may read V7 forward
outcomes, and nothing in V8 may modify any V7 file, manifest, or study
root. This has held for every V8 phase so far and must keep holding.**

## Read order

1. **This file** (`V8_START_HERE.md`)
2. **`V8_STATE.json`** — machine-readable current state; read this before
   trusting anything else, including this file's prose below
3. **`V8_PROJECT_STATE.md`** — human-readable narrative of the same state,
   with the "what must NOT happen yet" list
4. **`V8_RUNBOOK.md`** — only if you need to know what a specific phase
   (0–11) requires, permits, or prohibits
5. **`V8_HISTORICAL_RESEARCH_DESIGN.md`** — only for a specific frozen
   design question (exact thresholds, partition rules, promotion gates);
   this is the authoritative source for all frozen numbers, but it is long
   and you should not need to read it end to end
6. **`V8_DATA_EXPOSURE_AUDIT.md`** — only if you need to check whether a
   specific ticker/period was already exposed by an earlier study

## Current phase

```text
TRUST_ANCHOR_PINNED_PENDING_T1_HUMAN_GATE
```

## Last completed gate

```text
human_gate = TRUST_ANCHOR_PIN_AT_46023d92d359c222438b9c0b2dbe410e6623c1f6_FOR_MANIFEST_0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62_AUTHORIZED (consumed)
trust_anchor_result = AUTHORIZED (authorized_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62; authorized_partition_implementation_git_commit=36cbed941050e728f7f96ce2af505e81175cc02c)
real_partition_build_result = PASS (authorized_implementation_head 36cbed941050e728f7f96ce2af505e81175cc02c; exit_code=0); read-only validation PASS
real_jpx_source_only_attempt_3_result = PASS (source_reproduction_status=PASS, t0_reproduction_status=PASS, exit_code=0; authorized_head 371f547fc9f32c0aac84e34634b0f97a40e083c6)
review_result = SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS (reviewed_head 9b260e898aa019f8ee5102f3a00e7e1ec7a22584; review model Claude Opus 5 / claude-opus-5; 0 CRITICAL/HIGH/MEDIUM; 3 LOW deliberately unfixed)
implementation_commit = 1306d7be39ef9b73d049d5c4899ce286080ec1c2
test_fix_commit = 68b836d314b98955aa7d76e390ce6235a765b183 (test-only; no production code)
human_pc_pytest = 235 passed / 0 failed, exit code 0 (4 V8 test files; this sandbox has no pytest/pandas installed, so verification was done on the operator's own PC against transfer branch v8-partition-acquisition-transfer-pytest-check, then fast-forwarded onto v8-partition-acquisition with no merge/rebase/rewrite)
real_jpx_attempt_2 = ERROR/LOCAL_ENVIRONMENT_DEPENDENCY_MISSING_XLRD before T0 was reached (not a T0 result); no retry; local environment since remediated (xlrd==2.0.2 installed on operator's PC, no repository file changed)
```

## Latest real partition audit record

```text
authorized_implementation_head = 36cbed941050e728f7f96ce2af505e81175cc02c
mode = PRODUCTION
process_result = PASS
exit_code = 0
partition_manifest_written = true
real_block_assignments_created = true
manifest_sha256 = 0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
partition_implementation_git_commit = 36cbed941050e728f7f96ce2af505e81175cc02c
manifest_schema = V8_PARTITION_MANIFEST_V3
block_sizes = T0:300, T1:300, T2:300, T3:300, T_spare:1904
t1_ticker_list_sha256 = 262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
t2_ticker_list_sha256 = e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500
t3_ticker_list_sha256 = 43a585f4c3341307e7c67561c54780322b0f253fefa628a7c6129773901a7b7a
t_spare_ticker_list_sha256 = 360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
one_time_authorization_consumed = true
retry_performed = false
block_assignments_exposed = false
```

The exact private storage path is intentionally not recorded here. The
private manifest exists outside the repository and is not copied or committed.

Read-only validation subsequently returned `PASS` with exit code `0` using
`src.v8_partition.read_partition_manifest()`. Manifest self-hash,
implementation commit, schema, source PASS, T0 PASS, and T3 prohibition all
verified; the manifest remained present. No block assignment contents were
printed.

## Trust anchor pinning record

A further separate one-time human authorization then pinned
`V8_TRUSTED_PARTITION.json`:

```text
human_authorization = V8_HUMAN_AUTHORIZE_ONE_TRUST_ANCHOR_PIN_AT_46023d92d359c222438b9c0b2dbe410e6623c1f6_FOR_MANIFEST_0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62_IMPL_36cbed941050e728f7f96ce2af505e81175cc02c
anchor_base_head = 46023d92d359c222438b9c0b2dbe410e6623c1f6
authorization_status = AUTHORIZED
authorized_partition_manifest_sha256 = 0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
authorized_partition_implementation_git_commit = 36cbed941050e728f7f96ce2af505e81175cc02c
one_time_authorization_consumed = true
retry_performed = false
real_jpx_requests_this_task = 0
real_yahoo_requests_this_task = 0
```

`authorized_partition_implementation_git_commit` is the implementation
commit recorded *inside* the partition manifest at build time, not the SHA
of this pinning commit. This anchor does **not** itself authorize `T1`
acquisition, `T2` acquisition, `T3` acquisition (still unconditionally
prohibited), or `T_spare` acquisition. No block assignments were exposed;
the private manifest was neither opened nor copied.

## Current production status (do not contradict this)

```text
real_jpx_requests = 8 (source-only attempts #1-#3: 2 each; real partition build: 2; no retry)
real_yahoo_requests = 0
real_partition_created = true
accepted_source_snapshot = true (attempt #3 only; attempts #1 and #2 did not reach an accepted/reproduced snapshot)
partition_manifest_written = true
real_block_assignments_created = true
T1_real_data_acquired = false
T2_real_data_acquired = false
T2_opened = false
T3_data_acquired = false
backtests = 0
models_fitted = 0
profit_calculated = 0
parameter_search = 0
real_orders = 0
partition_public_dependency_injection = CLOSED_PENDING_REVIEW
trusted_partition_authorization = true (pinned by one-time trust-anchor authorization; see "Trust anchor pinning record" above)
real_jpx_authorization = false
real_jpx_source_fetch_authorized = false (attempt #3's authorization was consumed by its single PASS outcome; a fresh authorization is required for any further real JPX action)
real_T1_authorization = false
real_T2_authorization = false
real_T3_authorization = false
real_partition_creation_authorization_consumed = true
v4_raw_sha_equality_required_for_v8_partition = false (V8_HISTORICAL_RESEARCH_DESIGN.md §16; V8_PARTITION_SOURCE_NOT_REPRODUCIBLE raw-hash gate removed in code, implementation_commit 1306d7be39ef9b73d049d5c4899ce286080ec1c2; independently reviewed, SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS)
t0_300_exact_reproduction_required = true (unchanged; V8_T0_REPRODUCTION_MISMATCH still BLOCKs before allocate_fresh_blocks; reached and PASSED by attempt #3)
manifest_schema_version = V8_PARTITION_MANIFEST_V3 (was V8_PARTITION_MANIFEST_V2)
source_only_pass_authorizes_real_partition_creation = false
source_only_pass_authorizes_trust_anchor_pinning = false
source_only_pass_authorizes_t1_t2_t3_acquisition = false
private_v8_storage_location = DEFINED_OUTSIDE_REPOSITORY
real_partition_manifest_validated = true
trust_anchor_authorized = true
authorized_partition_manifest_sha256 = 0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
authorized_partition_implementation_git_commit = 36cbed941050e728f7f96ce2af505e81175cc02c
trust_anchor_pin_authorizes_t1_acquisition = false
trust_anchor_pin_authorizes_t2_acquisition = false
trust_anchor_pin_authorizes_t3_acquisition = false
```

The production partition-manifest CLI is implemented in
`scripts/build_v8_partition_manifest.py` as `--production-build-manifest`.
The source-only production preflight is implemented as
`--production-source-preflight`; it verifies raw JPX source reproduction and
T0 reproduction only, with no block allocation or partition publication.
The one-time real production partition build has completed and was validated
read-only; no retry was performed. The production acquisition CLI is now implemented in
`scripts/acquire_v8_historical.py` as `--production-acquire`, with only a
block, persisted partition manifest, private output root, and block-specific
confirmation as inputs.

`run_production_source_preflight()` accepts no inputs; it fixes JPX transport,
parsing, V4 provenance, UTC clock, repository root, and Git provenance
internally. `run_production_partition_build()` now accepts only `output_path`; it fixes
JPX transport, parsing, V4 provenance, UTC clock, repository root, and Git
provenance internally. Its dependency-injected implementation is a private
fake-test seam only and remains `CLOSED_PENDING_REVIEW`.

## Immediate next action

```text
T1_ACQUISITION_HUMAN_GATE
```

`V8_HISTORICAL_RESEARCH_DESIGN.md` §16 is implemented in code
(`1306d7be39ef9b73d049d5c4899ce286080ec1c2`, plus test-only fix
`68b836d314b98955aa7d76e390ce6235a765b183`), has **passed independent
review** (`SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS`, reviewed HEAD
`9b260e898aa019f8ee5102f3a00e7e1ec7a22584`; 0 CRITICAL/HIGH/MEDIUM, 3 LOW
deliberately unfixed), and a real JPX source-only preflight has **PASSED**
(attempt #3, against `371f547fc9f32c0aac84e34634b0f97a40e083c6`:
`source_reproduction_status=PASS`, `t0_reproduction_status=PASS`, exit code
0; `t0_ticker_list_sha256` matches the independently-verified committed
`V4_UNIVERSE.csv`). The subsequent one-time real partition build at
`36cbed941050e728f7f96ce2af505e81175cc02c` returned PASS and was validated
read-only. **A further one-time authorization has now pinned the trust
anchor**: `V8_TRUSTED_PARTITION.json` is `AUTHORIZED` with
`authorized_partition_manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62`
and `authorized_partition_implementation_git_commit=36cbed941050e728f7f96ce2af505e81175cc02c`.

**Trust-anchor pinning is complete, but this update does not authorize
`T1`/`T2`/`T3`/`T_spare` acquisition.** Each remains its own separate human
gate. The private manifest remains outside the repository and is not
committed; this documentation/state update performs no network operation.

See `V8_PROJECT_STATE.md` → "Current ordered next steps" for the full requirement
list, and `V8_STATE.json` → `trust_anchor_pinning` /
`trust_anchor_pin_authorizes_t1_acquisition` for the machine-readable form.

## Current blockers before any real network call

1. **No T1 acquisition authorization exists.** Trust-anchor pinning does not
   imply it; `acquire_historical_block_bundle()` still requires its own
   separate, explicit human authorization before any real Yahoo request for
   `T1`.
2. **Git provenance is required.** Immediately before every future real
   production command, an operator must run `git fetch origin` successfully
   and record the remote SHA and local `HEAD`. Both production runners then
   require a clean checkout whose `HEAD` equals that locally fetched
   `origin/v8-partition-acquisition` ref before network access. Acquisition
   reads its trust anchor from the verified `HEAD` Git object, never from the
   working-tree file.

No real JPX or Yahoo request is authorized by this documentation update.
Cumulative real JPX requests across all attempts and the real partition
build: 8. Real Yahoo requests: 0.

## Historical blockers at c5848ced1a5c800f384cb7b86fb642e5c748c2c2

1. **No production runner exists.** Both current CLIs are synthetic-only.
2. **Acquisition is not bound to a validated partition manifest.**
   `acquire_historical_block_bundle()` currently accepts `tickers` and
   `partition_manifest_sha256` as free caller-supplied arguments — it does
   not itself read an actual partition manifest, verify its self-hash, or
   confirm the tickers match `block_assignments[block]`. This must be fixed
   before any real acquisition call, or fresh-block integrity cannot be
   proven. Full detail: `V8_PROJECT_STATE.md` → "Critical pre-production
   blockers".

## Hard rules for every future session

- **Never commit private raw data to this repository.** No Yahoo raw
  payloads, no T1/T2 price rows, no private bundle contents, no secrets,
  no credentials, no tokens. This repository is public. Only status,
  hashes, and logical role belong in Git.
- **Chat memory is not a source of truth.** If a past conversation implies
  something this repository's files don't confirm, trust the repository
  and re-derive from GitHub state, not from memory. Priority order is in
  `V8_PROJECT_STATE.md` → "Source-of-truth priority".
- **Do not weaken V7 isolation.** Do not read V7 forward outcomes into V8,
  do not modify any V7 file, and do not use V8 findings to alter V7 mid-flight.

## New-chat starter prompt

Paste this to a fresh ChatGPT or Claude Code session to resume:

> GitHubの `ta1k1-arakawa/stock-analyzer` の `v8-partition-acquisition` を確認し、
> まず `V8_START_HERE.md` と `V8_STATE.json` を読んでください。
> 次に必要なら `V8_PROJECT_STATE.md` を読んでください。
> GitHubをsource of truthとしてremote HEADを確認し、
> 過去チャットの記憶には依存せず、
> 現在地・BLOCK事項・次に実行すべき1ステップを教えてください。
