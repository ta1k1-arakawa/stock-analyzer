# V8B_PRODUCTION_IMPLEMENTATION_SPEC

```text
document_type=PRODUCTION_IMPLEMENTATION_RECORD
study=V8B_HISTORICAL_RESEARCH
gate=V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_IMPLEMENTATION (V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md §12.1)
frozen_design_git_commit=eedf198b93185b963b825170ed0be97e93f923b7
implementation_performed=true
tests_are_fake_synthetic_only=true
real_t1b_allocation_executed=false
private_v8_partition_accessed=false
t_spare_t1b_t2_t3_ticker_identities_accessed=false
real_t1b_acquisition_executed=false
real_t2_acquisition_executed=false
real_yahoo_requests=0
real_jpx_requests=0
research_opening_performed=false
v8b_trusted_allocation_json_created=false
frozen_v8b_design_modified=false
v8_files_modified=false
```

This document records what `V8B_ALLOCATION_AUTHORITY_AND_ACQUISITION_
IMPLEMENTATION` (§12's gate sequence) actually built, and explicitly what
it did **not** do. It is a repository audit record, not a new methodology
document -- every threshold, rule, and gate boundary below is carried
forward verbatim from `V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md`; nothing
here makes a new methodological decision.

---

## 1. Modules added

| Module | §12.1 coverage | Role |
|---|---|---|
| `src/v8b_allocation.py` | A | Deterministic `T1B` slice (`parent_T_spare[0:300]`/`[300:]`, §4), private §11.3.B allocation-artifact builder/self-hash reader. Never touches Yahoo/JPX/the real trusted V8 partition. |
| `src/v8b_allocation_verification.py` | A, §11.4 | `READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION`: checks every §11.4 invariant against a concrete allocation artifact; returns a safe aggregate public result only; BLOCKs (no partial pin, no partial pass) on any single invariant failure. |
| `src/v8b_trust_pin.py` | B, C | Schema/builder/validator for the future public `V8B_TRUSTED_ALLOCATION.json` (§11.3.C). No writer function exists in this module by design -- `CREATE_V8B_TRUSTED_ALLOCATION_PIN` remains a separate, later, human-gated action this implementation does not perform. |
| `V8B_T2_AUTHORITY_BRIDGE.json` | E | The safe, committed, public OPTION_2 bridge artifact (§11.3.E): binds V8B's use of `T2` to the original, immutable V8 `T2` authority. Contains only hashes/counts/commit IDs/identifiers -- no ticker identities. |
| `src/v8b_historical_acquisition.py` | B, C, D, E, F, G, H, I, J | The whole production acquisition boundary for `T1B` and `T2` under `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE` (§7.6), including the §7.7 fail-before-network prerequisite and the §7.6 classifier-blob check. |

No script under `scripts/` was added in this phase. The library-level
public entrypoint (`acquire_v8b_historical_block_bundle`) is the complete
production boundary §12.1 requires; a CLI wrapper mirroring
`scripts/acquire_v8_historical.py` is deferred to future wiring work once
`ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B` is actually reachable --
this is an interface-only choice, not a methodology decision (it is not on
`AI_RESEARCH_EXECUTION_RULES.md` §2's enumerated discretion list), and it
narrows rather than widens what this phase makes runnable.

---

## 2. §7.6 / §7.7 coverage

```text
policy_name=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE
invalid_fraction_exact_comparison="invalid_returned_row_count * 252 <= total_returned_row_count"
floating_point_threshold_decision=NOT_USED (stored/compared as integer numerator=1, denominator=252)
max_consecutive_invalid_returned_rows=1 (run > 1, i.e. 2+ consecutive, BLOCKs)
canonical_parser_classifier_file=src/v7_yahoo_collector.py
canonical_parser_classifier_git_commit=28e281c3ee30d6b4c2f981c5da3ddc983c09724d
canonical_parser_classifier_blob_sha=76b57b077f3214e666ff9dc06d9c224afc16df9f
classifier_check_occurs_before_first_yahoo_request=true
full_p_hist_check_required=true
production_test_years=2018,2019,2020,2021,2022,2023,2024,2025
zoneinfo_check=Asia/Tokyo, verified before first Yahoo request (§7.7)
src_v7_yahoo_collector_py_modified=false
src_v8_historical_acquisition_py_modified=false
```

`src/v8b_historical_acquisition.py` does not reuse
`src/v8_historical_acquisition.py`'s `POLICY_G_PRIME_V1` thresholds or its
`("T1", "T2")` block allowlist -- it defines its own `("T1B", "T2")`
allowlist and its own F1_C1 threshold constants, exactly as §7.6 requires
(`reusable_as_is_for_v8b_acquisition=false`).

---

## 3. Pre-network ordering (§12.1.G-J, this task's explicit checklist)

`_acquire_production_v8b_historical_block_bundle_with_dependencies` runs,
strictly in this order, strictly before any Yahoo request:

1. repo/provenance (`resolve_verified_production_git_commit`: clean
   worktree, local HEAD == origin ref)
2. freeze approval (`V8B_DESIGN_FREEZE_APPROVAL.json`, read from the
   verified Git HEAD: `frozen_design_git_commit`, `approval_status`,
   `final_independent_review_result`, `preservation_recheck_result`)
3. reviewed implementation binding (`V8B_PRODUCTION_IMPLEMENTATION_
   REVIEW.json`, read from the verified Git HEAD -- this artifact does not
   exist in this repository yet, so real production BLOCKs here today by
   construction; tests inject a fake reader to exercise ordering)
4. classifier blob (§7.6)
5. Asia/Tokyo `ZoneInfo` (§7.7)
6. authority chain -- `T1B`: allocation artifact + trust pin (§11.3.A-D);
   `T2`: `V8_TRUSTED_PARTITION.json` anchor + `V8B_T2_AUTHORITY_BRIDGE.json`
   (§11.3.E)
7. block count/hash (folded into step 6's binding checks: exactly 300
   tickers, ticker-list SHA-256 match)
8. output/staging safety (`require_absolute_output_path_outside_repository`,
   already-exists / partial-staging rejection)

Only after all eight steps pass does the per-ticker acquisition loop --
the only place any Yahoo request can occur -- begin.

---

## 4. Privacy-safe failure behavior (§12.1.G)

Every `V8BHistoricalAcquisitionBlocked`/`V8BAllocationBlocked`/
`V8BAllocationVerificationBlocked`/`V8BTrustPinBlocked` reason string is a
fixed, generic identifier (e.g. `MALFORMED_OHLCV_QUALITY_GATE:
FRACTION_EXCEEDED`, `V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH`) -- never
a ticker, trading date, file path contents, or raw OHLCV value. The only
functions in `src/v8b_allocation.py` safe to log or print are
`public_allocation_summary()`'s return value and a `PASS`-result
`verify_t1b_allocation_artifact()` summary; both explicitly strip every
ticker-identity field.

---

## 5. One-shot / retry semantics

```text
retry_count=0 (repository-fixed constant; RETRY_COUNT is not caller-overridable)
no_threshold_grid_window_caller_override_exists=true (public entrypoint accepts only output_root/block/paths)
t1b_cannot_fall_back_to_old_t1_semantics=true (ALLOWED_ACQUISITION_BLOCKS = ("T1B","T2") only; old "T1" is in PROHIBITED_ACQUISITION_BLOCKS)
t2_cannot_bypass_option_2_bridge=true (T2 binding always cross-checks V8B_T2_AUTHORITY_BRIDGE.json against the verified V8 partition manifest and anchor)
no_v8_trust_anchor_mutated_or_repinned=true (V8_TRUSTED_PARTITION.json and src/v8_historical_acquisition.py are read-only inputs to this phase; neither file is modified)
```

---

## 6. What this implementation explicitly does not do

- Does not perform a real `T1B` allocation (`EXECUTE_T1B_ALLOCATION`
  remains a separate, later, human-gated action).
- Does not read, open, or reason about the real, private trusted V8
  partition manifest or `V8_TRUSTED_PARTITION.json`'s real pinned data at
  any point during this implementation task.
- Does not create `V8B_TRUSTED_ALLOCATION.json`.
- Does not make any real Yahoo or JPX network request.
- Does not implement §10's two research-opening security requirements
  (generic `open_for_*` guard contract, read-time content rebinding) --
  those remain separate, later, gated work, consistent with §10's own
  scope note that neither issue is reachable from the raw-acquisition path.
- Does not modify `V8_HISTORICAL_RESEARCH_DESIGN.md`,
  `V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md`, `V8_TRUSTED_PARTITION.json`,
  `src/v8_partition.py`, `src/v8_historical_acquisition.py`, or
  `src/v7_yahoo_collector.py`.

`NEXT ACTION = INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` (§12.3).
