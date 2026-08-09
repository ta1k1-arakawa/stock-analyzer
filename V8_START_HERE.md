# V8_START_HERE

Read this file first. It is short by design — it exists so a new chat does
not have to read the full 1000+ line design document before it can act.

## Identity

```text
study = V8_HISTORICAL_RESEARCH
repository = ta1k1-arakawa/stock-analyzer
active_branch = v8-partition-acquisition
state_basis_commit = aea2cb40efaf15bb749ee8545b021d65c2c52821
```

`state_basis_commit` is the commit this handoff package describes. It is
**not** a claim about the current remote HEAD — always re-check the remote
branch SHA yourself (`git ls-remote origin v8-partition-acquisition`) before
acting. If the remote has moved past this commit, treat this handoff as a
description of history up to this point, not of the present.

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
PRODUCTION_ACQUISITION_RUNNER_IMPLEMENTATION_PENDING
```

## Last completed gate

```text
human_gate = V8_T1_T2_ACQUISITION_AND_PARTITION_APPROVED
design_status = HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION  (design_commit c414d3191cba356734d7ed08bdf1abc7d51fc384)
static_implementation_verdict = V8_PARTITION_ACQUISITION_STATIC_PASS  (136 passed / 0 failed at aea2cb40efaf15bb749ee8545b021d65c2c52821)
```

## Current production status (do not contradict this)

```text
real_jpx_requests = 0
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
```

The production partition-manifest CLI is implemented in
`scripts/build_v8_partition_manifest.py` as `--production-build-manifest`.
It remains unused: no real JPX request or real partition manifest has been
created. `scripts/acquire_v8_historical.py` remains synthetic-only; no
production acquisition runner has been written.

## Immediate next action

```text
IMPLEMENT_PRODUCTION_ACQUISITION_RUNNER
```

See `V8_PROJECT_STATE.md` → "Current ordered next steps" for the full requirement
list, and `V8_STATE.json` → `production_blockers` for the machine-readable
form.

## Current blockers before any real network call

1. **Finding 1 is partially resolved.** The production partition-manifest
   runner is implemented, but the production acquisition runner remains
   unimplemented and synthetic-only.
2. **Finding 2 is resolved.** Acquisition is bound to a self-hash-verified,
   identity-verified partition manifest; only verified T1/T2 300-ticker
   assignments and their authoritative hashes are accepted, the manifest SHA
   is derived rather than supplied, `implementation_git_commit` is recorded,
   and T3 remains unconditionally blocked.

The next implementation task is `IMPLEMENT_PRODUCTION_ACQUISITION_RUNNER`.
It must use fake/mock transport during implementation and review, followed by
static/synthetic regression and an independent critical review. No real JPX
or Yahoo request is authorized by completing those steps.

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
