# V8_RUNBOOK

Operational reference for `V8_HISTORICAL_RESEARCH`, phase by phase. This
file describes the **workflow contract** — what each phase may and may not
do, and what it hands to the next phase. It is not the source of frozen
numeric values; those live in `V8_HISTORICAL_RESEARCH_DESIGN.md` and are
only summarized here for day-to-day convenience (see the box at the end).

Update this file only when a human gate changes the workflow contract
itself (a phase's prerequisites, allowed/prohibited actions, or ordering).
Do not update it just because a phase completed — that belongs in
`V8_PROJECT_STATE.md` and `V8_STATE.json`.

## Phase index

| Phase | Name | Status as of `91a049c137df014b8ac2d7f50ce8f79289f2b8f7` |
|---|---|---|
| 0 | Exposure audit | COMPLETE |
| 1 | Frozen design | COMPLETE |
| 2 | Partition/acquisition static implementation | COMPLETE |
| 3 | Production acquisition hardening | COMPLETE — implemented, remediated, independently reviewed, and locally regression-tested |
| 4 | Production partition creation | COMPLETE — one-time build PASS, read-only validation PASS, trust anchor pinned |
| 5 | T1/T2 raw acquisition | **CURRENT** — T1 attempts #1 and #2 BLOCKED; human design gate pending |
| 6 | Layer A historical research | NOT STARTED |
| 7 | Layer B one-shot validation | NOT STARTED |
| 8 | Exactly-one final candidate freeze | NOT STARTED |
| 9 | T2 one-shot sealed holdout | NOT STARTED |
| 10 | Separate prospective forward study | NOT STARTED |
| 11 | Separate human deployment gate | NOT STARTED |

---

## Phase 0 — Exposure audit

- **Prerequisites:** none (first phase).
- **Allowed actions:** read-only inspection of V3–V7 code, design docs,
  manifests, and committed artifacts already in this repository.
- **Prohibited actions:** any network access; any data acquisition; any
  backtest, model fit, or profit calculation; reopening any V3–V6 verdict.
- **Input artifacts:** the repository itself at the audit commit.
- **Output artifacts:** `V8_DATA_EXPOSURE_AUDIT.md`.
- **PASS condition:** every exposure claim traceable to committed evidence;
  `UNKNOWN_EXPOSURE` used for anything unprovable; `fresh_certified=0`
  (the audit certifies nothing as fresh, only identifies candidates).
- **Human gate:** none required to produce the audit itself.
- **Next phase:** 1.

## Phase 1 — Frozen design

- **Prerequisites:** Phase 0 complete.
- **Allowed actions:** writing/revising `V8_HISTORICAL_RESEARCH_DESIGN.md`;
  incorporating human decisions into a frozen machine-readable block.
- **Prohibited actions:** any implementation code; any network access; any
  data acquisition; modifying `V8_DATA_EXPOSURE_AUDIT.md`'s findings
  (append-only corrections are permitted, in a clearly marked appendix).
- **Input artifacts:** `V8_DATA_EXPOSURE_AUDIT.md`.
- **Output artifacts:** `V8_HISTORICAL_RESEARCH_DESIGN.md`.
- **PASS condition:** `design_status = HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION`.
- **Human gate:** required — design review approving (or amending) every
  partition, methodology, threshold, and gate decision.
- **Next phase:** 2.

## Phase 2 — Partition/acquisition static implementation

- **Prerequisites:** Phase 1 complete (`design_status = HUMAN_APPROVED_FROZEN_FOR_IMPLEMENTATION`)
  and a human gate authorizing implementation work
  (`V8_T1_T2_ACQUISITION_AND_PARTITION_APPROVED`).
- **Allowed actions:** implementing `src/v8_partition.py`,
  `src/v8_historical_acquisition.py`, synthetic-only CLIs, and tests using
  fake network fixtures.
- **Prohibited actions:** any real network request; any real partition
  creation; any real T1/T2/T3 acquisition; any V7 file modification; any
  feature/candidate/profit computation.
- **Input artifacts:** the frozen design.
- **Output artifacts:** `src/v8_partition.py`, `src/v8_historical_acquisition.py`,
  `scripts/build_v8_partition_manifest.py --synthetic-test`,
  `scripts/acquire_v8_historical.py --synthetic-test`, their test suites.
- **PASS condition:** `V8_PARTITION_ACQUISITION_STATIC_PASS` — all synthetic
  tests and CLIs pass with `network_requests=0` throughout.
- **Human gate:** the authorization to build code at all
  (`V8_T1_T2_ACQUISITION_AND_PARTITION_APPROVED`); this did **not**
  authorize real network access.
- **Next phase:** 3.

## Phase 3 — Production acquisition hardening (COMPLETE)

- **Prerequisites:** Phase 2 complete.
- **Allowed actions:** implementing a production partition-manifest CLI
  path; implementing the acquisition-to-manifest binding hardening
  described in `V8_PROJECT_STATE.md` → "Critical pre-production blockers,
  Finding 2"; adding implementation-commit provenance to the acquisition
  manifest schema; synthetic/static re-verification of the hardened code.
- **Prohibited actions:** any real network request of any kind. This phase
  is still entirely synthetic/static, even though it produces
  production-capable code paths.
- **Input artifacts:** the Phase 2 implementation; the two findings
  recorded in `V8_PROJECT_STATE.md`.
- **Output artifacts:** a hardened acquisition module/wrapper that reads an
  actual partition manifest, self-verifies it, and derives `tickers` and
  `partition_manifest_sha256` from it rather than accepting them freely; a
  production partition-manifest CLI path (still gated by the existing
  source-hash and T0-reproduction checks).
- **PASS condition:** all eight binding checks in `V8_PROJECT_STATE.md`
  Finding 2 implemented and covered by tests using fake fixtures; an
  independent (not self-) critical review of the hardened path completed;
  `network_requests=0` maintained throughout this phase.
- **Human gate:** required before this phase's output may be used for any
  real network call — this phase's completion authorizes moving to Phase 4
  only after that separate sign-off.
- **Next phase:** 4.

## Phase 4 — Production partition creation

- **Prerequisites:** Phase 3 complete and independently reviewed; explicit
  human authorization for real JPX network access.
- **Allowed actions:** one real fetch of the official JPX listing; running
  it through the already-implemented `V8_PARTITION_SOURCE_NOT_REPRODUCIBLE`
  / `V8_T0_REPRODUCTION_MISMATCH` guards; writing the real partition
  manifest to **private storage outside this repository**.
- **Prohibited actions:** any ticker price acquisition (that is Phase 5);
  committing the real manifest, or any raw JPX bytes, to this repository;
  bypassing either guard for any reason.
- **Input artifacts:** the hardened Phase 3 code; a real JPX network path.
- **Output artifacts:** exactly one real, private `partition_manifest.json`
  with `source_reproduction_status = PASS`.
- **PASS condition:** the real manifest reproduces `T0` exactly and passes
  its own self-hash verification; `block_assignments` for `T1`/`T2`/`T3`/
  `T_spare` are recorded; the manifest's hash is recorded (hash only) in
  `V8_STATE.json` once this phase completes.
- **Human gate:** required (this phase performs real network I/O).
- **Next phase:** 5.

## Phase 5 — T1/T2 raw acquisition

Current state: T1 raw acquisition attempt #1 was separately authorized and
BLOCKed at request 298 of 300 with reason class `MALFORMED_OHLCV`; no final
bundle was published and that authorization was consumed. After the policy
was selected and implemented, the separately authorized T1 attempt #2 ran at
authorized HEAD `a8710437db0c0752219d9aff34ac31d55b154d81` and BLOCKed with
`MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED` (exit code 2). No final
bundle or staging remained, research was not opened, validation was not
performed, T2/T3/T_spare were not touched, and the attempt #2 authorization
was consumed. Its exact Yahoo request count was not persisted; the safe lower
bound is >=1, so cumulative Yahoo requests are exact UNKNOWN with lower bound
>=299 (attempt #1 exact 298). No retry or attempt #3 is authorized. The next
action is `HUMAN_DESIGN_REVIEW_AFTER_T1_ATTEMPT_2_QUALITY_GATE_BLOCK`; this
runbook update does not choose a future policy or authorize network access.

- **Prerequisites:** Phase 4 complete; a real, validated partition manifest
  exists; explicit human authorization for real Yahoo network access,
  granted separately for `T1` and for `T2` (they may be acquired in
  separate runs).
- **Allowed actions:** one real Yahoo Chart acquisition run per block,
  strictly bound to the real partition manifest's `block_assignments["T1"]`
  / `["T2"]` (never free ticker input); sequential requests, ≥2.0s spacing,
  zero automatic retries; writing raw payloads and the acquisition manifest
  to **private storage outside this repository**.
- **Prohibited actions:** acquiring `T3` under any condition; computing any
  return, feature, signal, candidate, or profit value; opening `T2` for
  research (it must publish `sealed=true`,
  `research_access_authorized=false` by construction); committing any raw
  payload or private manifest to this repository.
- **Input artifacts:** the real partition manifest from Phase 4.
- **Output artifacts:** real, private `acquisitions/T1/...` and
  `acquisitions/T2/...` bundles, each with its own acquisition manifest
  recording counts and hashes only (never raw price values), plus the
  implementation-commit provenance added in Phase 3.
- **PASS condition:** `T1` status `RAW_ACQUIRED_NOT_OPENED`,
  `validation_access_count=0`; `T2` status `RAW_ACQUIRED_SEALED`,
  `sealed=true`, all four access counters `0`; both manifests' ticker lists
  hash-match the partition manifest's corresponding block hash.
- **Human gate:** required, separately for `T1` and `T2`.
- **Next phase:** 6.

## Phase 6 — Layer A historical research

- **Prerequisites:** Phase 5's `T1` acquisition complete (Layer A itself
  runs on `T0`, already available since Phase 0/repository state, per the
  design's Layer A definition — `T0 × P_hist`).
- **Allowed actions:** unlimited feature engineering, strategy design,
  parameter search, and model selection on `T0 × P_hist`, with full
  experiment-registry logging (`trial_id`, `hypothesis`,
  `changed_dimensions`, `walk_forward_splits`, `metrics`, `decision`,
  `code_commit`, `data_manifest`, and the additional fields in the design
  §7.2).
- **Prohibited actions:** treating any Layer A result as evidence of
  expectancy or generalization (`evidential_weight=NONE` is absolute);
  reopening any V3–V6 verdict; any access to `T1`/`T2`/`T3`.
- **Input artifacts:** `T0` (already exposed data — no new acquisition).
- **Output artifacts:** the full experiment registry; a walk-forward
  survivor shortlist that passes all nine frozen thresholds (design §8.4),
  frozen and preregistered before Phase 7 begins.
- **PASS condition:** `WALK_FORWARD_SURVIVOR` status for the shortlist,
  with full trial-count and trial-distribution reporting.
- **Human gate:** not required to run Layer A itself, but the shortlist
  freeze before Phase 7 is a preregistration event that should be recorded.
- **Next phase:** 7.

## Phase 7 — Layer B one-shot validation

- **Prerequisites:** Phase 6's shortlist frozen and preregistered; `T1`
  acquired (Phase 5); `validation_access_count` currently `0`.
- **Allowed actions:** exactly one validation batch evaluating the entire
  frozen shortlist under identical conditions against `T1 × P_hist`.
- **Prohibited actions:** re-tuning any parameter after `T1` is opened;
  resubmitting a failed candidate to `T1`; opening any block other than
  `T1`; a second validation access without a new human-gated fresh block
  from `T_spare`.
- **Input artifacts:** the frozen shortlist; the real `T1` acquisition.
- **Output artifacts:** the validation report; `validation_access_count = 1`.
- **PASS condition:** at least one candidate survives Layer B without
  re-tuning.
- **Human gate:** required to open `T1` for validation (this is the first
  access to previously-sealed-by-non-use fresh data).
- **Next phase:** 8.

## Phase 8 — Exactly-one final candidate freeze

- **Prerequisites:** Phase 7 produced at least one validation survivor.
- **Allowed actions:** selecting **exactly one** candidate (never a set) and
  freezing every parameter, feature, friction assumption, and universe
  choice for it, with all values hashed.
- **Prohibited actions:** freezing more than one candidate; any access to
  `T2` before this freeze is recorded.
- **Input artifacts:** the Layer B validation report.
- **Output artifacts:** the immutable `FROZEN_FINAL_CANDIDATE` record.
- **PASS condition:** exactly one candidate record exists, immutable from
  this point forward.
- **Human gate:** required — this is the point of no return before the
  sealed holdout opens.
- **Next phase:** 9.

## Phase 9 — T2 one-shot sealed holdout

- **Prerequisites:** Phase 8's single frozen candidate; `T2` acquired
  (Phase 5) and still `sealed=true` (never opened before this point).
- **Allowed actions:** exactly one evaluation of the frozen candidate
  against `T2 × P_hist`, using the official access-guard API to formally
  transition `research_access_authorized` for this one evaluation.
- **Prohibited actions:** comparing or selecting among multiple candidates
  on `T2`; any re-access after this one evaluation; changing any condition,
  parameter, feature, universe, or threshold and reusing `T2` — that
  requires a new sealed partition and a new study.
- **Input artifacts:** the frozen candidate; the real `T2` acquisition.
- **Output artifacts:** the sealed-holdout report;
  `sealed_holdout_access_count = 1`.
- **PASS condition:** `SEALED_HOLDOUT_PASS` or `SEALED_HOLDOUT_FAIL` — both
  are legitimate scientific outcomes and must be recorded either way. A
  PASS means only "the same historical rule reproduced on a cross-section
  unused by prior development loops" — never "unbiased historical
  profitability proven" or "real-world expectancy proven" (both phrases are
  explicitly prohibited by the frozen design, §12.3).
- **Human gate:** required to open `T2`.
- **Next phase:** 10 (only on PASS; on FAIL, the candidate is closed and a
  new hypothesis returns to Phase 6 with a new eventual validation/holdout
  block pair).

## Phase 10 — Separate prospective forward study

- **Prerequisites:** Phase 9 `SEALED_HOLDOUT_PASS`.
- **Allowed actions:** designing and preregistering a new, separate,
  live-data, paper-only forward observation study for the V8 candidate,
  structurally similar to V7 but fully independent of it.
- **Prohibited actions:** using V7's forward observations as if they were
  V8's; skipping preregistration; any real order.
- **Input artifacts:** the `SEALED_HOLDOUT_PASS` candidate.
- **Output artifacts:** a new prospective-study design document (not yet
  created; out of scope for this handoff).
- **PASS condition:** defined by that future study's own design.
- **Human gate:** required — this is a new study, not an extension of V8's
  historical work.
- **Next phase:** 11.

## Phase 11 — Separate human deployment gate

- **Prerequisites:** Phase 10's prospective forward study reaches its own
  positive conclusion.
- **Allowed actions:** human decision-making only.
- **Prohibited actions:** treating historical development, validation, or
  sealed-holdout evidence alone — or even combined with a positive forward
  study — as automatically authorizing deployment. This gate is never
  implied by any prior step.
- **Input artifacts:** the complete evidentiary chain: Phase 6 through
  Phase 10.
- **Output artifacts:** a human deployment decision, recorded separately.
- **PASS condition:** explicit human authorization, distinct from every
  gate above.
- **Human gate:** required — this is the real-money gate.
- **Next phase:** none (terminal).

---

## Summary of frozen design (convenience only — not authoritative)

The values below are restated here only so day-to-day operation does not
require opening the full design document for common numbers. **If this box
and `V8_HISTORICAL_RESEARCH_DESIGN.md` ever disagree, the design document is
authoritative and this box is wrong and must be fixed.**

```text
P_hist: 2016-04-01 .. 2025-12-31

walk-forward: expanding-window chronological, exactly 8 splits
  test years: 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025

T1: fresh validation, 300 tickers, max_validation_access = 1
T2: fresh sealed holdout, 300 tickers, exactly one frozen final candidate,
    sealed_holdout_access_count = 1 (one-shot)
T3: sealed reserve, 300 tickers, not used in initial V8

friction grid (all-in, per side): 0.03% / 0.05% / 0.10% / 0.15%
  base evaluation friction: 0.05%
  survivability floor: 0.10% (net profit > 0 and PF >= 1.05 required there)
```

Ticker block source: `V8_HISTORICAL_RESEARCH_DESIGN.md` §5, Decision 2.
Walk-forward source: §8.1, Decision 7. Friction source: §8.5, Decision 8.
Promotion thresholds (nine of them, including the parameter-neighbourhood
rule): §8.4, Decision 9 — not restated here because they are numerous and
easy to misquote from memory; read them directly when needed.
