# V13 Pre-Acquisition Exposure Provenance Audit

## A. Exact audit boundary

This is a repository-evidence audit at the exact public committed HEAD below.
Only tracked repository files and public GitHub/Git history/evidence were
inspected. Ignored caches, private/sealed manifests and payloads, and
market-data providers were not read or contacted. No V13 universe was
selected.

```text
AUDIT_HEAD=fadf65ae33ee2c267e24cc848902ee53ef67938a
NETWORK_REQUESTS=0
PRIVATE_OR_SEALED_READS=0
HISTORICAL_PRICE_READS=0
MODEL_FITS=0
BACKTESTS=0
V13_TICKER_SELECTIONS=0
```

`NETWORK_REQUESTS=0` is the audit counter for prohibited market-data/source
acquisition. Public GitHub issue/commit evidence was used only as the task and
provenance boundary.

The Issue #35 authorization record is GPT-reviewed PASS at its reviewed HEAD
`61d2ca8f409d61c49d822402cfa740eec7ab81b5`. It authorizes only retriable
public JPX/Yahoo plumbing; it does not
authorize historical viability, model fitting, backtesting, private/sealed
access, forward paper, or trading.

## B. Known public exclusion foundations

The fixed V4 universe is bound without duplicating its 300-code list:

```text
FIXED_V4_300_SOURCE=V4_UNIVERSE.csv
FIXED_V4_300_GIT_BLOB_SHA1=5a19ea918be6773e0d43d98eb5a9f3afc9920346
FIXED_V4_300_COUNT=300
FIXED_V4_300_TICKER_LIST_SHA256=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
FIXED_V4_300_CSV_SHA256=d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997
```

The V4 blob identity is the Git tree binding. The CSV and ticker-list hashes
are the corresponding values in `V4_UNIVERSE_MANIFEST.json`.

The public legacy foundation is bound by
`V8_DATA_EXPOSURE_AUDIT.md` §2.1 and `data/benchmark/manifest.json`:

```text
LEGACY_8_SOURCE=V8_DATA_EXPOSURE_AUDIT.md §2.1; data/benchmark/manifest.json
LEGACY_8_COUNT=8
LEGACY_OUTSIDE_V4_COUNT=7
LEGACY_OUTSIDE_V4_CANONICAL_ENCODING=UTF-8 numeric-ascending codes, LF between entries, final LF
LEGACY_OUTSIDE_V4_CODE_LIST=
1570
4689
5020
7211
7267
8306
9432
LEGACY_OUTSIDE_V4_CODE_LIST_SHA256=1c840273c60f8b901f954694e869f2218368f72a2940fca6986fab3c8fd4942e
```

Only `4188` from `LEGACY_8` is inside `FIXED_V4_300`; the seven-code list
above is therefore the unique legacy addition.

These public components establish the existing V4/LEGACY exclusion
foundation. They do not form a complete current V13 exclusion set: committed
V8 T1 evidence proves additional acquired price payloads whose identities are
not publicly resolvable. Do not treat a combined V4/LEGACY count or hash as
the completed V13 exclusion list while those identities remain unresolved.

## C. V8–V12 exposure scan

The V8 audit already establishes the pre-existing V3–V7 outcome exposure:
the V3–V7 fixed universe is `FIXED_V4_300`, and the separate committed
benchmark exposure is `LEGACY_8`. The rows below account for post-audit
evidence through `AUDIT_HEAD`.

| Version/study | Evidence artifact / commit | Complete real historical-price payload acquired? | Public ticker count / identity resolution | All acquired identities covered by V4/LEGACY? | Additional exclusion identities required? | Metadata-only / no-price classification |
| --- | --- | --- | --- | --- | --- | --- |
| V8 T1 raw acquisition attempts | `V8_STATE.json` at `fadf65ae...`; attempt-1 implementation `d5441020389452d85cb19a94f647448775fba8d8` | Attempt #1 made 298 real Yahoo requests. Its first 297 successful ticker payloads passed the per-ticker quality check and were written to staging before request 298 failed `MALFORMED_OHLCV`. No complete bundle remained; staging was cleaned. | Frozen T1 block size is 300; the identities are private/uncommitted and not publicly resolvable. | No. Frozen T1 membership is fresh after T0=`FIXED_V4_300`, and every fresh block excludes all seven `LEGACY_8` codes outside T0. | Yes; the acquired identity set is unresolved private/uncommitted. Do not infer or publish identities or a complete list/hash. | Partial historical-price byte acquisition is proven. The staged bytes were not opened for research; cleanup does not undo acquisition. This is provenance, not outcome/model exposure. |
| V8B T1B | `V8B_T1B_ACQUISITION_FAILURE_ADJUDICATION.json` at `61d2ca8f...` | No. `t1b_staging_count=0`, `t1b_final_bundle_exists=false`, and the raw acquisition result is `BLOCKED`. | No public ticker count or identities. | N/A. | No. | Acquisition attempted and transport-blocked; no price payload set or outcome. |
| V8C–V8J successor/preservation work | `V8C_PREFREEZE_PRESERVATION_RECHECK.md`, `V8D_T2_PREFREEZE_PRESERVATION_RECHECK.md`, `V8E_T2_PREFREEZE_PRESERVATION_RECHECK.md`, `V8I_SOURCE_SNAPSHOT_TERMINAL_ADJUDICATION.json`, and `V8J_SOURCE_SNAPSHOT_ENVIRONMENT_SUCCESSOR_DESIGN_DRAFT.md` at `61d2ca8f...` | No historical-price payload. The committed T2 and successor records retain `raw_data_acquired=false`/no research opening. | No public price-ticker identities. | N/A. | No. | Gate, source-snapshot, and environment evidence only; no price/outcome exposure. |
| V8K public source preparation | `V8K_TERMINATION_RECORD.md` and `PROJECT_STATE.md` at `61d2ca8f...` | No historical-price payload. A complete public universe-source payload was locked, but it is listing/eligibility metadata, not prices. | Safe evidence reports 3,110 eligible rows and a list hash, not identities. | N/A: metadata-only. | No. | Metadata/current-universe preparation only; Stage-2/T1 and profitability result were not run. |
| V9.006 F1/F6 public source acquisition | `V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_EXECUTION_EVIDENCE.md`, `PROJECT_STATE.md`, and `PROJECT_DECISION_LOG.md` at `61d2ca8f...` | No historical-price payload. The locked objects are JPX HTML/spreadsheet source-structure objects; F6 child content was not inspected. | No price-ticker set is emitted by the safe evidence. | N/A: source-structure acquisition only. | No. | Public source/metadata acquisition; no OHLCV, model, or outcome. |
| V9.010, V9.014, V9.015, V9.017 | `V9_010_STAGE_A_JPX_CALENDAR_SOURCE_MANIFEST.json`, `V9_015_STAGE_F_TERMINAL_ADJUDICATION.json`, `V9_017_FIXED_EIGHT_REAL_APPLICATION_TERMINAL_ADJUDICATION.json`, and `PROJECT_STATE.md` at `61d2ca8f...` | No historical-price payload. These records concern calendars, archive HTML/PDF locator evidence, and fixed-eight source application; V9.017 records zero network data requests and no result JSON. | Safe evidence may expose source/fixed-identity counts, but no price-ticker identity set. | N/A: no historical prices. | No. | Metadata/source-calendar activity only; no outcome-bearing execution. |
| V10A, V10C, V10D environment/calendar/T0 paths | `V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json`, `V10C_T0_ATTEMPT_1_SAFE_ADJUDICATION.json`, and `PROJECT_STATE.md` at `61d2ca8f...` | No. Calendar bytes and environment evidence are not prices; V10C records zero payload reads for the safe phase-C inspection and no scientific T0 result. | No price-ticker identity set. | N/A. | No. | Environment/calendar/provenance work only; no price outcome, model fit, or backtest. |
| V10B training-cache reacquisition / V10C provenance adoption | `V10B_PHASE_B_PHASE_C_TERMINAL_ADJUDICATION.json`, `V10C_TRAINING_PROVENANCE_ADOPTION_RECORD.json`, and the V10B frozen design at `61d2ca8f...` | Yes: 283 accepted first-complete Yahoo payloads out of the terminal 300-ticker processing, with `manifest_complete=true` and `locked_payload_hash_closure=PASS`. No semantic payload parsing, T0, model fit, or outcome calculation occurred. | 300 fixed attempted identities are positively bound to `FIXED_V4_300` by the manifest contract (`universe_mode`, V4 CSV SHA, ticker-list SHA); individual payload identities are intentionally not emitted. | Yes. The 283 accepted identities are a subset of the fixed V4 set; the 17 failed identities are also members of that same fixed set and were not acquired as complete price payloads. | No. | Historical-price bytes acquired, but fully covered by the existing V4 exclusion; no outcome-bearing execution. |
| V11 Core30 selector | `PROJECT_STATE.md`, `V11_DESIGN_FREEZE_APPROVAL.json`, and the bound source evidence at `61d2ca8f...` | No historical-price payload or outcome-bearing run. | Public state records source/candidate counts and hashes only; no price-ticker identities are evidenced. | N/A: metadata/selection only. | No. | Metadata-only ticker-selection activity; `V11_TICKER_SELECTED=false`. |
| V12 Core30 selector | `PROJECT_STATE.md`, `V12_DESIGN_FREEZE_APPROVAL.json`, and the bound source evidence at `61d2ca8f...` | No historical-price payload or outcome-bearing run. | Public state records 31 candidate rows and a selection hash, not a completed price-ticker set. | N/A: metadata/selection only. | No. | Metadata-only ticker-selection activity; `V12_TICKER_SELECTED=false`. |

The absence of a ticker identity in a safe metadata artifact is not treated as
evidence of freshness. The V10B row closes only because its positive fixed-V4
universe binding covers the acquired subset. V8 T1 is a separate confirmed
partial historical-price acquisition: its acquired identities fall outside
the known V4/LEGACY foundation, but cannot be resolved from public committed
evidence. The other rows remain classified by their committed metadata/source
or no-price evidence.

## D. Unresolved exclusion identity verdict

```text
EXPOSURE_PROVENANCE_STATUS=BLOCKED_UNRESOLVED_PRIVATE_OR_UNCOMMITTED_IDENTITIES
V8_T1_ATTEMPT1_REAL_YAHOO_REQUESTS=298
V8_T1_ATTEMPT1_FAILING_REQUEST_POSITION=298_OF_300
V8_T1_ATTEMPT1_DEFINITELY_STAGED_SUCCESSFUL_PAYLOAD_COUNT=297
V8_T1_IDENTITIES_PUBLICLY_RESOLVABLE=false
V8_T1_IDENTITIES_OUTSIDE_V4_LEGACY=true
ADDITIONAL_EXCLUSION_IDENTITY_SET_STATUS=UNRESOLVED_PRIVATE_OR_UNCOMMITTED
```

The 298th request triggered the whole-block failure. This bookkeeping does
not assign an exclusion treatment to that failing request or infer any
private ticker identity. Fail-closed cleanup deleted staging; it does not
retroactively mean the first 297 price payloads were never acquired. No V8
research or validation outcome was opened from those bytes, so this finding
records acquisition provenance rather than strategy, profitability, or
outcome exposure.

The canonical exclusion construction rule is:

```text
EXCLUSION_SET = codes(V4_UNIVERSE.csv) UNION LEGACY_8 UNION any additional public code whose complete historical-price payload or outcome-bearing execution is positively evidenced before V13 universe creation
```

The additional identity set is not publicly resolvable, so a complete
exclusion list or hash cannot be published and the provenance audit remains
blocked. A separate explicit access or methodology decision is required to
resolve the private exclusion identities before V13 universe construction
can proceed. The public acquisition authorization remains granted but
unexecuted; it does not unblock universe construction or authorize historical
viability, model fitting, or backtesting.
