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
| `src/v8b_trust_pin.py` | B, C | Schema/builder/validator for the future public `V8B_TRUSTED_ALLOCATION.json` (§11.3.C). No writer function exists in this module by design -- `CREATE_V8B_TRUSTED_ALLOCATION_PIN` remains a separate, later, human-gated action, implemented (but not executed) by `src/v8b_trust_pin_creation.py`. |
| `src/v8b_trust_pin_creation.py` | C | The production-gated `CREATE_V8B_TRUSTED_ALLOCATION_PIN` boundary (§12): obtains its verification summary only from the real `resolve_and_verify_t1b_allocation_artifact` resolver, requires `HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION` bound to the exact verified artifact hash, and requires a fresh `INDEPENDENT_TRUST_PIN_REVIEW` bound to that same hash. Not executed by this phase. Added in remediation round 5 (finding HIGH-2). |
| `src/v8b_human_gate_consumption.py` | H | Durable, fsync'd, atomically-created, fail-closed one-shot consumption receipts for `ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B`, `T1B_RAW_ACQUISITION_HUMAN_GATE`, and `T2_RAW_ACQUISITION_HUMAN_GATE`, replacing the prior in-memory-only `authorization_consumed` boolean. Added in remediation round 5 (finding HIGH-1). |
| `V8B_T2_AUTHORITY_BRIDGE.json` | E | The safe, committed, public OPTION_2 bridge artifact (§11.3.E): binds V8B's use of `T2` to the original, immutable V8 `T2` authority. Contains only hashes/counts/commit IDs/identifiers -- no ticker identities. |
| `src/v8b_historical_acquisition.py` | B, C, D, E, F, G, H, I, J | The whole production acquisition boundary for `T1B` and `T2` under `POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE` (§7.6), including the §7.7 fail-before-network prerequisite and the §7.6 classifier-blob check. |
| `src/v8b_git_provenance.py` | -- | V8B-specific Git provenance primitives: branch resolution bound to `v8b-allocation-authority-acquisition-implementation` (never V8's branch), malicious-`GIT_*`-environment isolation, blob/object readers. Added in remediation round 2 (finding HIGH-1). |
| `src/v8b_production_provenance.py` | -- | Exact-blob-bound verification of the frozen design object, design-freeze approval, reviewed-implementation binding, and the original immutable V8 `T2` authority (anchor + OPTION_2 bridge). Added in remediation round 2 (findings HIGH-2, HIGH-4, HIGH-5). |
| `src/v8b_t1b_allocator.py` | -- | The production-gated `T1B` allocation boundary (§11.3.B, `EXECUTE_T1B_ALLOCATION`). Not executed by this phase. Added in remediation round 2 (finding HIGH-7). |

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

1. repo/provenance (`src.v8b_git_provenance.resolve_verified_v8b_production_
   git_commit`: clean worktree, local HEAD == `origin/v8b-allocation-
   authority-acquisition-implementation` -- V8B's own production branch,
   never V8's `v8-partition-acquisition`; every git subprocess strips
   redirection-capable `GIT_*` environment variables first)
2. frozen design object + freeze approval (`V8B_HISTORICAL_RESEARCH_
   DESIGN_DRAFT.md` verified at the exact frozen commit's own blob;
   `V8B_DESIGN_FREEZE_APPROVAL.json` read from the verified Git HEAD and
   checked against its exact expected blob, plus every field: `frozen_
   design_git_commit`, `approval_status`, `final_independent_review_
   result`, `preservation_recheck_result`, `human_gate`, `design_
   finalized`, `human_design_freeze_complete`, and that allocation/
   network/acquisition/research-opening authorization fields all remain
   `false`)
3. reviewed implementation binding (`V8B_PRODUCTION_IMPLEMENTATION_
   REVIEW.json`, read from the verified Git HEAD -- this artifact does not
   exist in this repository yet, so real production BLOCKs here today by
   construction; tests inject a fake reader to exercise ordering). The
   review's own `reviewed_implementation_git_commit` -- not current HEAD --
   becomes the commit recorded in every resulting artifact; every file in
   `src.v8b_production_provenance.BOUND_PRODUCTION_FILES` must have an
   identical Git blob at HEAD and at that reviewed commit, or the whole
   phase BLOCKs. A later docs/audit-only commit may move HEAD forward
   without invalidating the review, as long as no bound blob actually
   changed.
4. classifier blob (§7.6)
5. Asia/Tokyo `ZoneInfo` (§7.7)
6. authority chain -- `T1B`: private allocation artifact (caller-supplied
   path) + the future public `V8B_TRUSTED_ALLOCATION.json` trust pin, read
   from a **verified Git object**, never a caller-supplied path (§11.3.A-D);
   `T2`: `V8_TRUSTED_PARTITION.json` anchor, verified against its exact
   frozen Git blob (`61faade0625139cec3fb61216ab2f97f572a7028`) before any
   of its fields are trusted, plus `V8B_T2_AUTHORITY_BRIDGE.json`,
   verified against exact frozen field values (§11.3.E)
7. block count/hash -- pinned to the exact frozen literal `T2` ticker
   count/hash (`e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9
   cc3d6500`), never merely "whatever the current anchor/manifest says"
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

---

## 7. Remediation round 2 (`INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` first round: CRITICAL=0, HIGH=7, MEDIUM=3)

```text
high_findings_fixed=7
medium_findings_fixed=3
```

| Finding | Fix |
|---|---|
| HIGH-1: reused V8's branch-hardcoded Git resolver | New `src/v8b_git_provenance.py`, bound to `v8b-allocation-authority-acquisition-implementation`; `src.v8_partition.resolve_verified_production_git_commit` is no longer called anywhere in V8B production code. |
| HIGH-2: `implementation_git_commit` was audit HEAD, not the reviewed commit | `verify_reviewed_implementation_binding` resolves the exact `reviewed_implementation_git_commit` from `V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json` and requires every file in `BOUND_PRODUCTION_FILES` to have an identical blob at HEAD and at that commit; the reviewed commit, not HEAD, is what acquisition/allocation artifacts record. |
| HIGH-3: T1B trust pin accepted from a caller-supplied path; `human_gate` accepted any nonempty string | `acquire_v8b_historical_block_bundle` no longer has a `t1b_trust_pin_path` parameter; the pin is read only from a verified Git object. `src/v8b_trust_pin.py`'s `human_gate` must now exactly equal `HUMAN_GATE_PREFIX + artifact_self_hash` (`build_trust_pin` no longer accepts a `human_gate` argument at all -- it is always derived). |
| HIGH-4: V8 anchor/T2 hash trusted at face value | `read_and_verify_v8_trusted_partition_anchor` checks the anchor's exact Git blob (`61faade0...`) before parsing any field; `T2` ticker count/hash are pinned to the exact frozen literals, not merely cross-checked for internal consistency. |
| HIGH-5: frozen design/approval trusted without exact-blob binding | `verify_frozen_design_object` and `read_and_verify_design_freeze_approval` check exact Git blobs (`33e6789e...`, `545ffaa3...`) plus every approval field. |
| HIGH-6: lower-layer exception text could reach public reasons | `_safe_transport_reason` (finite whitelist for `V7YahooCollectorBlocked.reason`) and `_classify_transport_exception` (type/code only, never `str(error)`/`.args`/`.reason` on unrecognised exceptions) replace the prior direct-passthrough handling. |
| HIGH-7: no production-gated allocator existed | New `src/v8b_t1b_allocator.py`; requires the exact confirmation literal `V8B_PRODUCTION_ALLOCATE_T1B`, the full provenance/freeze/review chain, the exact frozen parent `T_spare` count/hash (`1904` / `360d5c87...`), and atomically publishes the private artifact. Not executed. |
| MEDIUM-1: `verify_acquisition_artifact` under-checked | Now recomputes `payload_manifest_sha256`, and checks exact F1_C1 policy metadata, classifier blob, data-source fields, role/status/sealed/access counters, and `authority_binding` schema per block. |
| MEDIUM-2: T2 recheck accepted an arbitrary caller mapping as trust root | Split into a private pure evaluator (fake tests only) and `resolve_and_recheck_t2_reuse_conditions`, which derives `safe_metadata` from `V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md` read from a verified Git object, pinned to its exact blob. |
| MEDIUM-3: `open_for_*` research-opening API was exported | Removed entirely from `src/v8b_historical_acquisition.py` -- no research-opening API of any kind exists in this module. |

`NEXT ACTION = REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` (§12.3).

---

## 8. Remediation round 3 (repeat `INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW`: CRITICAL=0, HIGH=4, MEDIUM=1)

```text
high_findings_fixed=4
medium_findings_fixed=1
raw_acquisition_human_gate_tokens_implemented=true (V8B_PRODUCTION_ACQUIRE_T1B, V8B_PRODUCTION_ACQUIRE_T2)
one_shot_authorization_consumption_tests_present=true
fresh_post_freeze_t2_recheck_required=true (V8B_T2_REUSE_CONDITIONS_RECHECK.json does not exist -- fails closed)
v8_partition_py_bound_to_review=true
section_12_6_exact_ticker_and_authority_verification=true
production_t1b_allocation_verifier_implemented=true
real_allocation_performed=false
private_real_data_accessed=false
yahoo_jpx_requests=0
```

| Finding | Fix |
|---|---|
| HIGH-1: raw acquisition/allocation human gates had no mechanical confirmation token or one-shot consumption semantics | `src/v8b_historical_acquisition.py` now requires the exact block-specific literal (`T1B_ACQUISITION_CONFIRMATION`/`T2_ACQUISITION_CONFIRMATION`, checked before any other step) and tracks `authorization_consumed` -- `False` through every pre-network step, `True` from the first per-ticker opener attempt onward, exposed only on `V8BHistoricalAcquisitionBlocked`, never on the published manifest. `src/v8b_t1b_allocator.py` mirrors this for `V8B_PRODUCTION_ALLOCATE_T1B`, with consumption beginning at the first private partition-manifest read. Neither confirmation literal is real human authorization; both remain mechanical anti-fat-finger syntax, matching this repository's existing `--confirmation` convention. |
| HIGH-2: T2 reuse recheck read the wrong (§12.2 pre-freeze) evidence document for the §12.4 post-freeze gate | `src/v8b_t2_reuse_recheck.py` no longer reads `V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md`. It now resolves the future `V8B_T2_REUSE_CONDITIONS_RECHECK.json` (schema `V8B_T2_REUSE_CONDITIONS_RECHECK_V1`, `stage=POST_FREEZE`, bound to the frozen design commit) from a verified Git object; that artifact does not exist in this repository, so `T2` production acquisition fails closed by construction today. |
| HIGH-3: reviewed-implementation binding omitted `src/v8_partition.py`, which V8B production executes directly | `BOUND_PRODUCTION_FILES` in `src/v8b_production_provenance.py` now includes `src/v8_partition.py`; a later drift in that file's blob blocks acquisition/allocation even when every V8B-authored module is unchanged. |
| HIGH-4: §12.6 verifier accepted caller-supplied expected hashes/authority strings as its trust root, and never checked the manifest's own `ticker_list_sha256` | `verify_acquisition_artifact` (private/pure) now also requires an exact `expected_ticker_list_sha256` and full-value (not merely same-keys) `expected_authority_binding` match. The new production boundary `resolve_and_verify_acquisition_artifact` derives all of these from verified Git objects -- the exact frozen `T2` hash/anchor/bridge, or the Git-sourced `T1B` trust pin -- and never from a caller-supplied value. |
| MEDIUM-1: no production wrapper existed for `READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` | New `resolve_and_verify_t1b_allocation_artifact` in `src/v8b_allocation_verification.py`: resolves verified Git HEAD, verifies the frozen design/freeze-approval/reviewed-implementation chain and the exact immutable V8 anchor, reads the private V8 partition manifest and `T1B` allocation artifact from caller-supplied paths, derives every block's ticker assignment internally, checks the artifact's parent-manifest SHA/implementation-commit and the exact frozen parent `T_spare` count/hash, checks `v8b_allocation_implementation_commit` equals the reviewed commit, and only then invokes the pure §11.4 evaluator. Returns only the pure evaluator's safe hash/count summary. |

`NEXT ACTION = REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` (§12.3).

---

## 9. Remediation round 4 (FULL-BOUNDARY repeat `INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW`: CRITICAL=0, HIGH=3, MEDIUM=2)

```text
high_findings_fixed=3
medium_findings_fixed=2
intended_repository_identity_enforced=true (ta1k1-arakawa/stock-analyzer, github.com only)
production_repository_root_override_possible=false
payload_membership_hash_recomputed_from_concrete_records=true
filesystem_errors_privacy_redacted=true
authorization_consumed_exactly_at_first_opener_invocation=true
unsafe_pure_helpers_removed_from_production_public_surface=true
real_allocation_performed=false
private_real_data_accessed=false
yahoo_jpx_requests=0
```

| Finding | Fix |
|---|---|
| HIGH-1: `resolve_verified_v8b_production_git_commit` proved `HEAD == origin/<branch>` but never that `origin` was `ta1k1-arakawa/stock-analyzer`; several production resolvers also accepted a caller-selectable `repository_root`. | `src/v8b_git_provenance.py` now also requires `origin` to resolve to one of the ordinary HTTPS/SSH forms of exactly `ta1k1-arakawa/stock-analyzer` on `github.com` (`_canonical_github_owner_repo`), checked before the origin-ref/HEAD comparison -- a same-named branch in any other repository, a look-alike host, or a bare local path all BLOCK. `resolve_and_verify_acquisition_artifact`, `resolve_and_verify_t1b_allocation_artifact`, `resolve_and_recheck_t2_reuse_conditions`, and `resolve_t2_reuse_safe_metadata_from_verified_head` no longer accept a `repository_root` parameter -- each is now a public wrapper (always `CANONICAL_REPOSITORY_ROOT`) around a private, DI-testable implementation used only by fake/synthetic tests. |
| HIGH-2: §12.6 verification compared the manifest's claimed `ticker_list_sha256` to the trusted hash but never recomputed it from the concrete `payload_manifest` records. | `src/v8b_acquisition_artifact_verification.py` now validates every `payload_manifest` record's exact schema and canonical-ticker form, requires exactly 300 unique tickers (preserving `payload_manifest`'s own order), and recomputes `ticker_list_sha256` from those concrete values via `src.v8_partition.ticker_list_sha256` -- a forged bundle whose manifest carries the correct trusted hash while its `payload_manifest`/raw files actually name a different 300-ticker set now BLOCKs (`PAYLOAD_TICKER_MEMBERSHIP_HASH_MISMATCH`). |
| HIGH-3: several filesystem operations in the acquisition/allocation path could let a raw `OSError` (potentially carrying a private path or ticker-derived filename) escape uncaught. | Every filesystem call in the production write/read path -- `_write_bytes` (raw payload/manifest/seal writes), the private `T1B` allocation-artifact read, staging-directory creation (`tempfile.mkdtemp`/`mkdir`), the output-directory `mkdir`/`iterdir`, and the atomic `os.replace` publish in `src/v8b_historical_acquisition.py`, plus the staging write in `src/v8b_t1b_allocator.py`'s `_write_allocation_artifact_once` -- is now wrapped and mapped to a fixed, generic reason; none ever forward `str(error)`, `.args`, or a path. |
| MEDIUM-1: `authorization_consumed` was set to `True` before the local pacing/wait call, not at the actual opener/network boundary. | The transition now happens inside `recording_opener`, immediately before the real, underlying `opener(request_obj)` call -- strictly after the local URL-origin check succeeds. A pacing failure (now itself wrapped as `REQUEST_PACING_FAILED`) or a local request-preparation failure leaves `authorization_consumed=False` with zero opener calls for that ticker; once the opener is actually invoked, it is `True` regardless of outcome and is never reset. |
| MEDIUM-2: `verify_acquisition_artifact`, `verify_t1b_allocation_artifact`, and `recheck_t2_reuse_conditions` were caller-controlled pure evaluators exported as part of the production public surface. | All three are now private (`_verify_acquisition_artifact`, `_verify_t1b_allocation_artifact`, `_recheck_t2_reuse_conditions`), removed from their modules' `__all__`; fake/synthetic tests import and call them directly as internal helpers. Each module's public surface now exposes only its Git-/authority-grounded resolver plus safe constants. |

Full-boundary self-review (implementation review → T1B allocation → allocation verification → future trust pin → T1B acquisition → T1B artifact verification → post-freeze T2 reuse check → T2 acquisition → T2 artifact verification) found no further trust-root injection, stale-time evidence, un-recomputed claimed hash, caller-controlled authority value, filesystem/exception privacy leak, caller override of a frozen rule, cross-block token reuse, research-opening path, retry/fallback/substitution, or non-blob-bound reviewed dependency. No methodological ambiguity was found requiring `CHATGPT_DECISION_REQUIRED`.

`NEXT ACTION = FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` (§12.3).

---

## 10. Remediation round 5 (`FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW`: CRITICAL=0, HIGH=3, MEDIUM=2)

```text
high_findings_fixed=3
medium_findings_fixed=2
durable_one_shot_gate_consumption_implemented=true (ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B, T1B_RAW_ACQUISITION_HUMAN_GATE, T2_RAW_ACQUISITION_HUMAN_GATE)
trust_pin_creation_production_gated_but_not_executed=true
t2_reuse_recheck_derives_from_authoritative_state=true
t2_reuse_recheck_public_resolver_accepts_zero_arguments=true
allocation_verification_exact_binds_full_semantic_field_set=true
public_acquisition_entrypoint_returns_redacted_summary_only=true
real_allocation_performed=false
private_real_data_accessed=false
yahoo_jpx_requests=0
```

| Finding | Fix |
|---|---|
| HIGH-1: `authorization_consumed`/`consumed` was only an in-memory boolean scoped to a single call -- nothing durable prevented a second call, a new process, or a restart from repeating the exact same one-time action. | New `src/v8b_human_gate_consumption.py`: a durable, fsync'd, atomically-created (never overwritten) receipt file per `(gate, v8b_frozen_design_commit)` pair, under the fixed, non-overridable `CANONICAL_CONSUMPTION_STATE_ROOT` (a sibling directory of the repository checkout, outside the Git worktree so it never dirties `git status --porcelain`). `src/v8b_t1b_allocator.py::allocate_t1b_production` and `src/v8b_historical_acquisition.py::acquire_v8b_historical_block_bundle` (both `T1B` and `T2`) now check `require_gate_not_yet_consumed` immediately after the confirmation-token check (before any provenance/network step) and durably `consume_gate_once` at the exact same point consumption already occurred (the first private partition read for the allocator; the first trusted Yahoo opener invocation, once per call, for acquisition) -- strictly before that private/network action. A second invocation using the same authorization now BLOCKs (`V8B_HUMAN_GATE_ALREADY_CONSUMED:<gate>`) before any private access or Yahoo request, with `authorization_consumed=False`. No retry/resume semantics were added -- a durable-write failure (as opposed to an already-consumed receipt) leaves the gate unconsumed and the caller's next attempt starts the whole sequence over, exactly as before. |
| HIGH-2: `src/v8b_trust_pin.py::build_trust_pin` accepted any caller-supplied mapping shaped like a PASS verification summary -- an arbitrary `{"result": "PASS", ...}` dict was sufficient to build an "AUTHORIZED" pin, and nothing required an accompanying `INDEPENDENT_TRUST_PIN_REVIEW`. | New `src/v8b_trust_pin_creation.py::create_v8b_trusted_allocation_pin_production` closes the full chain: it obtains the verification summary **only** by calling the real, Git-grounded `resolve_and_verify_t1b_allocation_artifact` (never accepts a summary as a parameter -- confirmed by `test_no_way_to_supply_a_favorable_mapping_directly`), requires `human_pin_authorization` to exactly equal `expected_human_gate(artifact_self_hash)` for that verified artifact's own hash, and requires a fresh `V8B_TRUST_PIN_INDEPENDENT_REVIEW.json` (new reader `src.v8b_production_provenance.read_and_verify_trust_pin_independent_review`, read from a verified Git object, bound to that exact hash) before it will build and write-once an `AUTHORIZED` pin. Symmetrically, `src/v8b_historical_acquisition.py`'s `T1B` acquisition branch now also requires that same fresh, exact-hash-bound `INDEPENDENT_TRUST_PIN_REVIEW` (`trust_pin_review_reader`) in addition to the trust pin's own `human_gate` grammar check -- T1B acquisition cannot proceed on the human pin authorization alone. Neither module is executed against real data; both new artifacts (`V8B_TRUST_PIN_INDEPENDENT_REVIEW.json`, a real allocation-verification PASS) do not exist, so both fail closed today by construction. `src/v8b_trust_pin_creation.py` and `src/v8b_human_gate_consumption.py` are added to `BOUND_PRODUCTION_FILES`. |
| HIGH-3: `resolve_and_recheck_t2_reuse_conditions(verified_head)` accepted a caller-supplied `verified_head`, and five of the seven §3.3/§9 conditions were trusted purely because `V8B_T2_REUSE_CONDITIONS_RECHECK.json` self-declared them true. | `src/v8b_t2_reuse_recheck.py`'s public resolver now takes **zero arguments** -- it resolves the current verified production HEAD itself. `t2_universe_definition_unchanged`/`t2_partition_algorithm_unchanged` are now derived from the exact-frozen-blob immutable V8 trust anchor (`read_and_verify_v8_trusted_partition_anchor`) succeeding; `t2_v8b_f1_c1_policy_fixed` from the exact-blob reviewed-implementation binding (`verify_reviewed_implementation_binding`, which covers `src/v8_partition.py` and `src/v8b_historical_acquisition.py`) succeeding; `t2_opened`/the two research-exposure flags from a live check that no `open_for_*` API exists in the bound `src.v8b_historical_acquisition` module; and `t2_acquired` from whether `T2_RAW_ACQUISITION_HUMAN_GATE`'s durable HIGH-1 consumption receipt already exists (`has_gate_been_consumed`) -- never a self-reported boolean. The evidence artifact's own claimed value for each of these fields must still *agree* with the derived value (`V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:<field>` otherwise), but the derived value is what actually governs PASS/BLOCK. `layer_b_completed`/`frozen_final_candidate_established` remain read from the artifact (no independently git-derivable proxy exists for whether Layer B/`FROZEN_FINAL_CANDIDATE` actually occurred). `src.v8b_historical_acquisition`'s T2 branch now calls the zero-argument resolver directly. |
| MEDIUM-1: `READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` under-bound the allocation artifact's own trust-bearing semantic fields -- `verify_allocation_artifact_self_hash` alone accepts any internally self-consistent (self-hash-recomputed) forgery, since it only proves the hash matches the artifact's *own* other fields, not that those fields have the *correct* values. | `src/v8b_allocation_verification.py::_verify_t1b_allocation_artifact` now exact-binds `schema_version`, `study_name`, `artifact_role`, `logical_block`, `parent_study`, `selection_rule_id`, `t1b_offset_within_parent_t_spare`, `t1b_slice_start_inclusive`, `t1b_slice_end_exclusive`, and `parent_t_spare_ticker_count` against this module's own frozen constants/caller-supplied trusted parent length, in addition to the existing hash/commit/invariant checks. New parametrized tamper tests (`test_self_hash_recomputed_wrong_semantic_field_blocks`) construct a self-hash-recomputed artifact for each field and prove it BLOCKs; `test_self_hash_alone_would_not_have_caught_a_wrong_artifact_role` demonstrates the exact gap this closes. |
| MEDIUM-2: `acquire_v8b_historical_block_bundle` returned the full, ticker-identity-bearing acquisition manifest (including `payload_manifest`, which names all 300 tickers) from the public production entrypoint. | `src/v8b_historical_acquisition.py` now persists the full manifest only inside the private acquisition bundle on disk (unchanged) and returns `public_acquisition_summary(manifest)` -- every field except `payload_manifest` (counts, hashes, schema/status identifiers, commit IDs; `PUBLIC_ACQUISITION_SUMMARY_FIELDS`). The private DI-testable seam (`_acquire_production_v8b_historical_block_bundle_with_dependencies`) still returns the full manifest, as fake/synthetic tests require to assert against `payload_manifest`/raw bytes; only the true public boundary is redacted. |

Full-boundary self-review (implementation review → T1B allocation → allocation verification → trust pin → T1B acquisition → T1B artifact verification → post-freeze T2 reuse check → T2 acquisition → T2 artifact verification) repeated after this round found no further trust-root injection, stale-time evidence, un-recomputed claimed hash, caller-controlled authority value, filesystem/exception privacy leak, caller override of a frozen rule, cross-block/cross-gate token reuse, research-opening path, retry/fallback/substitution, non-blob-bound reviewed dependency, or process-local-only one-shot control. No methodological ambiguity was found requiring `CHATGPT_DECISION_REQUIRED`.

`NEXT ACTION = FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` (§12.3).
