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
SOURCE_SNAPSHOT_CLARIFICATION_APPROVED_IMPLEMENTATION_PENDING
```

## Last completed gate

```text
human_gate = SOURCE_SNAPSHOT_CLARIFICATION (V8_HISTORICAL_RESEARCH_DESIGN.md §16, append-only, 2026-08-10)
human_clarification = IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT
design_status = HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION  (design_commit c414d3191cba356734d7ed08bdf1abc7d51fc384; §16 is an append-only erratum, not a reopening)
static_implementation_verdict = SOURCE_PREFLIGHT_REVIEW_PASS (source-only preflight implementation commit 38697c9ede51cac7bd500206d857ee585464996b)
```

## Current production status (do not contradict this)

```text
real_jpx_requests = 2 (one source-only preflight attempt, 2026-08-10; result BLOCKED/V8_PARTITION_SOURCE_NOT_REPRODUCIBLE; no retry)
real_yahoo_requests = 0
real_partition_created = false
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
trusted_partition_authorization = false
real_jpx_authorization = false
real_jpx_source_fetch_authorized = false (the single 2026-08-10 authorization was consumed by the one attempt above; a fresh authorization is required for the next attempt)
real_T1_authorization = false
real_T2_authorization = false
v4_raw_sha_equality_required_for_v8_partition = false (V8_HISTORICAL_RESEARCH_DESIGN.md §16)
t0_300_exact_reproduction_required = true (unchanged)
```

The production partition-manifest CLI is implemented in
`scripts/build_v8_partition_manifest.py` as `--production-build-manifest`.
The source-only production preflight is implemented as
`--production-source-preflight`; it verifies raw JPX source reproduction and
T0 reproduction only, with no block allocation or partition publication.
Neither production path has been invoked: no real JPX request or real
partition manifest has been created. The production acquisition CLI is now implemented in
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
IMPLEMENT_SOURCE_SNAPSHOT_SEMANTICS
```

Implement `V8_HISTORICAL_RESEARCH_DESIGN.md` §16 in code: the source-only and
full-build paths in `src/v8_partition.py` /
`scripts/build_v8_partition_manifest.py` must stop requiring
`source_raw_sha256 == V4 raw_file_sha256`, while continuing to require exact
`T0` reproduction (BLOCK on mismatch) and to fix the newly-fetched snapshot's
own provenance (`eligible_ticker_list_sha256`, raw byte count, acquisition
UTC, etc.) into the result/manifest. This has **not** been implemented yet —
this handoff is a design/state clarification only, no code changed. The
implementation then requires its own independent review before any further
real JPX request is authorized.

See `V8_PROJECT_STATE.md` → "Current ordered next steps" for the full requirement
list, and `V8_STATE.json` → `production_blockers` / `source_snapshot_clarification`
for the machine-readable form.

## Current blockers before any real network call

1. **Source-snapshot semantics not yet implemented in code.** The design
   clarification (§16) is recorded, but `src/v8_partition.py` still enforces
   the stricter raw-hash-equality check that caused the 2026-08-10 BLOCK.
   Until that code changes and is independently reviewed, a real re-attempt
   would very likely BLOCK again for the same reason.
2. **Trusted partition authorization is false.** Production acquisition reads
   the canonical Git-tracked `V8_TRUSTED_PARTITION.json` before the partition
   manifest and blocks with `TRUSTED_PARTITION_NOT_AUTHORIZED` before Yahoo
   transport until a separate human gate pins a real manifest SHA and its
   partition implementation commit.
3. **Git provenance is required.** Immediately before every future real
   production command, an operator must run `git fetch origin` successfully
   and record the remote SHA and local `HEAD`. Both production runners then
   require a clean checkout whose `HEAD` equals that locally fetched
   `origin/v8-partition-acquisition` ref before network access. Acquisition
   reads its trust anchor from the verified `HEAD` Git object, never from the
   working-tree file.

No real JPX or Yahoo request is authorized by this documentation update. The
one 2026-08-10 source-only preflight authorization was consumed by its single
attempt (BLOCKED, no retry); any further real attempt needs a fresh human
authorization.

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
