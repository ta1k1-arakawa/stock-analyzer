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
SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS_PENDING_FRESH_REAL_JPX_HUMAN_GATE
```

## Last completed gate

```text
human_gate = FRESH_REAL_JPX_SOURCE_ONLY_PREFLIGHT_ATTEMPT_2_AUTHORIZED (consumed)
static_implementation_verdict = SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS
review_result = SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS (reviewed_head 9b260e898aa019f8ee5102f3a00e7e1ec7a22584; review model Claude Opus 5 / claude-opus-5; 0 CRITICAL/HIGH/MEDIUM; 3 LOW deliberately unfixed)
implementation_commit = 1306d7be39ef9b73d049d5c4899ce286080ec1c2
test_fix_commit = 68b836d314b98955aa7d76e390ce6235a765b183 (test-only; no production code)
human_pc_pytest = 235 passed / 0 failed, exit code 0 (4 V8 test files; this sandbox has no pytest/pandas installed, so verification was done on the operator's own PC against transfer branch v8-partition-acquisition-transfer-pytest-check, then fast-forwarded onto v8-partition-acquisition with no merge/rebase/rewrite)
real_jpx_attempt_2 = ERROR/LOCAL_ENVIRONMENT_DEPENDENCY_MISSING_XLRD before T0 was reached (not a T0 result); no retry; local environment since remediated (xlrd==2.0.2 installed on operator's PC, no repository file changed)
```

## Current production status (do not contradict this)

```text
real_jpx_requests = 4 (attempt #1, 2026-08-10: 2 requests, result BLOCKED/V8_PARTITION_SOURCE_NOT_REPRODUCIBLE; attempt #2: 2 requests, result ERROR/LOCAL_ENVIRONMENT_DEPENDENCY_MISSING_XLRD before T0 was reached; no retry either time)
real_yahoo_requests = 0
real_partition_created = false
accepted_source_snapshot = false (neither attempt reached an accepted/reproduced snapshot)
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
real_jpx_source_fetch_authorized = false (attempt #2's authorization was consumed by its single ERROR outcome; a fresh authorization is required for attempt #3)
real_T1_authorization = false
real_T2_authorization = false
v4_raw_sha_equality_required_for_v8_partition = false (V8_HISTORICAL_RESEARCH_DESIGN.md §16; V8_PARTITION_SOURCE_NOT_REPRODUCIBLE raw-hash gate removed in code, implementation_commit 1306d7be39ef9b73d049d5c4899ce286080ec1c2; independently reviewed, SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS)
t0_300_exact_reproduction_required = true (unchanged; V8_T0_REPRODUCTION_MISMATCH still BLOCKs before allocate_fresh_blocks; not yet reached by either real attempt)
manifest_schema_version = V8_PARTITION_MANIFEST_V3 (was V8_PARTITION_MANIFEST_V2)
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
FRESH_REAL_JPX_SOURCE_PREFLIGHT_HUMAN_GATE
```

`V8_HISTORICAL_RESEARCH_DESIGN.md` §16 is implemented in code
(`1306d7be39ef9b73d049d5c4899ce286080ec1c2`, plus test-only fix
`68b836d314b98955aa7d76e390ce6235a765b183`) and has **passed independent
review** (`SOURCE_SNAPSHOT_SEMANTICS_REVIEW_PASS`, reviewed HEAD
`9b260e898aa019f8ee5102f3a00e7e1ec7a22584`; 0 CRITICAL/HIGH/MEDIUM, 3 LOW
deliberately unfixed). Attempt #2 of the real JPX source-only preflight was
then authorized and run against that reviewed HEAD, but errored before
reaching `T0` reproduction: `ERROR/LOCAL_ENVIRONMENT_DEPENDENCY_MISSING_
XLRD` (`pandas.read_excel` needs `xlrd>=2.0.1` for `.xls`, missing on the
operator's PC at the time). This is **not** a `T0` PASS/FAIL result — the
source snapshot was never accepted, no raw bytes were persisted, no hash was
recorded. The operator has since installed `xlrd==2.0.2` locally (already
declared in this repo's `requirements.txt`; no repository file changed).
Attempt #2's authorization is consumed; **attempt #3 is not yet authorized**.
This documentation/state update itself grants no real network authorization.

See `V8_PROJECT_STATE.md` → "Current ordered next steps" for the full requirement
list, and `V8_STATE.json` → `source_snapshot_semantics_review` /
`source_preflight_attempt_history` for the machine-readable form.

## Current blockers before any real network call

1. **No fresh real-JPX-attempt authorization exists.** Attempt #2's
   authorization was consumed by its single ERROR outcome (pre-`T0`,
   environment-caused, not a design/implementation failure). A new, separate
   human authorization is required before attempt #3.
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

No real JPX or Yahoo request is authorized by this documentation update.
Cumulative real JPX requests across both attempts: 4. Real Yahoo requests: 0.

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
