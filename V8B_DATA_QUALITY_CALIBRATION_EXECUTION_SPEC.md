# V8B_DATA_QUALITY_CALIBRATION_EXECUTION_SPEC

```text
study=V8B_HISTORICAL_RESEARCH
document_type=CALIBRATION_EXECUTION_ADAPTER_SPEC
status=IMPLEMENTED_NOT_EXECUTED
implements=production execution adapter for the already-approved
  V8B_DATA_QUALITY_CALIBRATION_PLAN_V1 (V8B_DATA_QUALITY_CALIBRATION_
  PLAN_APPROVAL.json: approval_status=APPROVED, next_phase_authorized=
  CALIBRATION_IMPLEMENTATION_ONLY_NO_REAL_DATA)
approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1
approved_plan_git_commit=8c15426166742c43745e604f6367788af6123c1a
methodology_change=false
real_v5b_cache_accessed_by_this_task=false
calibration_executed_by_this_task=false
```

This document specifies the production adapter that wires the single
fixed, real V5-B cache root to the existing, frozen, pure
`run_data_quality_calibration()` (`src/v8b_data_quality_calibration.py`).
It performs no methodology of its own: no grid, window, parser,
synthetic-design, defensibility, selection, or result-semantics decision
is made here. Its entire job is to securely acquire exactly the manifest
and 300 designated payload byte sequences from the real cache and hand
them, unmodified, to that already-reviewed pure function.

---

## 1. Relationship to the reviewed R1 preflight (fixed)

This adapter is **not** the R1 preflight
(`src/v8b_v5b_calibration_input_preflight.py`,
`V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_SPEC.md`) and does not invoke it. A
prior real preflight `PASS` artifact is **never** treated as permission to
skip this adapter's own verification:

```text
prior_preflight_pass_grants_cache_access=false
this_adapter_re_derives_and_re_verifies_provenance_from_scratch=true
```

Every real execution independently re-reads the manifest and every
payload, and independently re-verifies manifest provenance and every
payload's byte-count/SHA-256 against the same frozen pins the preflight
uses -- via the same reused, already-reviewed
`validate_v5b_manifest_provenance()` -- before any parsing happens. This
adapter's own Git-HEAD/dirty-file binding (§4) also reuses the preflight's
already-reviewed `_verify_implementation_matches_repository_head()`
directly, rather than re-implementing that logic a second time.

---

## 2. Required human gate (fixed)

```text
required_confirmation_token=V8B_DATA_QUALITY_CALIBRATION_EXECUTION_GATE
```

This token is distinct from the R1 preflight's own
`V5B_CALIBRATION_INPUT_PREFLIGHT_GATE` token -- passing the preflight gate
never satisfies this one. As with the preflight, the token's presence in
source code is not itself an authorization; it only prevents accidental
invocation absent an explicit, deliberate human decision. This
implementation task does not invoke
`run_production_v8b_data_quality_calibration()` against the real fixed
cache root; only tests exercise it, and only with `V5B_CACHE_ROOT` and
`_REPO_ROOT` monkeypatched to synthetic fixtures.

---

## 3. Fixed production input (fixed)

```text
fixed_cache_root=C:\taiki\hobbies\v5-b-evaluation-cache-retry1
```

Identical to the R1 preflight's fixed root.
`src/v8b_data_quality_calibration_execution.py` exposes this as
`V5B_CACHE_ROOT`, a module-level constant. The production entry point,
`run_production_v8b_data_quality_calibration()`, reads only this path; it
takes no cache-path, manifest-path, input-directory, or dataset parameter.
`scripts/run_v8b_data_quality_calibration.py`, the CLI that wraps it,
likewise exposes no such option -- its only arguments are `--static-check`
(no filesystem access) and `--confirm` / `--implementation-git-commit` /
`--calibration-attempt-id` (gated, fixed-root-only execution).

**No module-level filesystem-capable callable exists at all except the
gated entry point**, mirroring the hardened preflight pattern exactly.
The byte-acquisition logic that walks `V5B_CACHE_ROOT` is a closure
nested inside `run_production_v8b_data_quality_calibration`'s own
function body: it has no module-level name, cannot be imported,
monkeypatched onto, or invoked independently of a call to the gated entry
point, and does not appear in `dir()` of this module. This logic is
duplicated from (not imported from) the preflight module's own nested
closure, because that closure is deliberately private with no module-level
name in the preflight module -- there is no way to import it, by design.
`run_static_check()` (§9) scans this module's **entire** callable surface
(via `dir()`, not merely `__all__`) on every invocation to enforce that no
callable defined in this module, exported or not, accepts a
`cache_root`/`path`/`manifest_path`/`input_dir`/`dataset` parameter.

---

## 4. Implementation Git-HEAD binding (fixed)

Immediately after confirmation and format checks pass, and strictly
before any V5-B cache filesystem access,
`run_production_v8b_data_quality_calibration()` requires:

```text
1. implementation_git_commit is syntactically a 40-hex Git commit
2. calibration_attempt_id is a non-empty string, at most 128 characters,
   containing no control characters (identical rule to
   src/v8b_data_quality_calibration.py's own provenance validation)
3. this repository's actual Git HEAD is resolvable (via the reused
   preflight verifier, itself using a GIT_*-sanitized subprocess
   environment) -- otherwise BLOCK with detail_reason=GIT_HEAD_UNRESOLVABLE
4. this repository's Git top-level resolves to exactly the intended
   _REPO_ROOT -- otherwise BLOCK with
   detail_reason=GIT_REPOSITORY_IDENTITY_MISMATCH
5. implementation_git_commit exactly equals that actual Git HEAD --
   otherwise BLOCK with detail_reason=IMPLEMENTATION_COMMIT_HEAD_MISMATCH
6. every file in the fixed set below has on-disk bytes in the working
   tree that are byte-identical to what is committed at that verified
   HEAD -- otherwise BLOCK with detail_reason=IMPLEMENTATION_FILE_DIRTY
   (or IMPLEMENTATION_FILE_UNVERIFIABLE if the committed blob itself
   cannot be read)
```

The fixed set of files bound by step 6
(`_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS`):

```text
src/v8b_data_quality_calibration_execution.py   (this module)
scripts/run_v8b_data_quality_calibration.py     (this module's CLI)
V8B_DATA_QUALITY_CALIBRATION_EXECUTION_SPEC.md  (this document)
src/v8b_data_quality_calibration.py             (reused: run_data_quality_calibration,
                                                  validate_v5b_manifest_provenance,
                                                  InMemoryPayload, EXPECTED_V5B_* pins)
src/v8b_v5b_calibration_input_preflight.py      (reused: the Git-verification
                                                  helper itself)
```

`src/v7_yahoo_collector.py` (the pinned canonical row classifier),
`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md`, and
`V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json` are **not** in this
list, because they are already independently protected by a different,
equally rigorous mechanism: `run_data_quality_calibration()` calls
`verify_repository_contract()` internally, on every invocation, which
git-blob-hash-pins all three against frozen constants
(`PINNED_COLLECTOR_BLOB_SHA`, `APPROVED_PLAN_BLOB_SHA`,
`APPROVAL_ARTIFACT_BLOB_SHA`) -- a dirty copy of any of them is caught
inside the frozen calibration core itself, every time, regardless of this
adapter's own Git-HEAD binding.

Steps 3-6 above reuse
`src/v8b_v5b_calibration_input_preflight.py::_verify_implementation_
matches_repository_head()` directly (imported, not reimplemented),
called with this module's own `_RELEVANT_IMPLEMENTATION_RELATIVE_PATHS`.
This is pure reuse of already-reviewed logic; neither the preflight module
nor the calibration core is modified by this adapter.

---

## 5. Allowed real-execution read scope (fixed, exhaustive)

A future, separately authorized execution reads, in order:

```text
1. stat/check existence of the exact fixed cache root (a directory);
   reject symlink/junction/reparse redirection of the root or any parent
2. read the exact bytes of <cache_root>/cache_manifest.json through one
   checked, handle-bound, reparse/TOCTOU-safe read (identical technique to
   the reviewed preflight: CreateFileW + OPEN_REPARSE_POINT +
   BACKUP_SEMANTICS + GetFinalPathNameByHandleW containment check on
   Windows; O_NOFOLLOW + dir_fd-chained per-component opens + /proc/self/fd
   realpath containment check on POSIX)
3. strict-parse and validate those bytes using the existing, frozen
   validate_v5b_manifest_provenance() -- reused verbatim, not
   reimplemented, exactly as the preflight does
4. obtain the manifest-designated 300 payload relative paths from that
   validated manifest -- never any other file in the cache root
5. for those exact 300 payloads only, through the same checked,
   handle-bound read as the manifest:
   a. reject symlink/reparse/path-escape redirection
   b. read raw bytes -- exactly once per file
   c. verify the read byte length equals the manifest's declared
      byte_count exactly
   d. compute SHA-256 over the exact same bytes just read and require it
      to equal the manifest's declared sha256 for that payload
6. only once all 300 payloads pass steps 5b-5d with zero mismatches and
   zero missing/unreadable files, hand the manifest bytes and every
   payload's exact in-memory bytes (the same bytes object read in step
   5b, never a second, separate read) to the existing, frozen
   run_data_quality_calibration(), and return its result unmodified
```

No dataset outside these exact 300 manifest-designated files is ever
enumerated or read. No unrelated file in the cache root is read. No
fallback, substitution, or reacquisition path exists anywhere in this
module.

---

## 6. Strictly forbidden in this adapter's own code (fixed, exhaustive)

This adapter's own code never:

```text
json_parses_raw_payload_body=false
inspects_ohlcv_values=false
executes_row_classifier_on_payload_rows=false
computes_m_fraction_or_m_consecutive=false
constructs_calibration_windows=false
selects_synthetic_bases=false
generates_synthetic_corruption=false
evaluates_candidates=false
selects_or_adopts_a_policy=false
accesses_old_t1_t1b_t2_t3=false
accesses_yahoo_network=false
accesses_jpx_network=false
```

All of R0-R9 (classifier pinning, manifest provenance, payload binding,
canonical parsing, window statistics, synthetic verification, candidate
evaluation, selection) happens exclusively inside the existing, frozen,
already-reviewed `run_data_quality_calibration()` -- this adapter never
reimplements, shortcuts, or duplicates any of that methodology. This
adapter's own code performs only secure byte acquisition (§5) and the
gate checks (§2, §4) that must happen before that acquisition.

---

## 7. Production result privacy (fixed)

Only two shapes ever reach stdout/stderr:

```text
1. this adapter's own gate-level "execution status" (§8) -- produced only
   when confirmation, commit/attempt-id format, Git verification, or the
   cache byte-acquisition walk itself fails, i.e. run_data_quality_
   calibration() was never reached
2. the unmodified calibration result artifact returned by
   run_data_quality_calibration() -- the already-approved, already-
   privacy-reviewed schema frozen in src/v8b_data_quality_calibration.py
```

Neither shape ever contains:

```text
raw ticker identities
payload relative or absolute paths
raw payload contents (Yahoo JSON bodies)
OHLCV rows
```

The existing, approved calibration result artifact's
`synthetic_base_window_start_and_end_metadata` field records only a
`ticker_sha256` (an opaque hash, not the ticker itself) per selected
synthetic base -- this is pre-existing, already-reviewed calibration-core
behavior this adapter does not change and does not add to. This adapter
introduces no new identity-bearing field anywhere.

---

## 8. Execution status schema (fixed)

The gate-level "did not reach the frozen calibration at all" status:

```text
schema_version                          "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_STATUS_V1"
study                                   "V8B_HISTORICAL_RESEARCH"
role                                    "EXECUTION_GATE"
status                                  "BLOCKED"
detail_reason                           safe structural code
implementation_git_commit               caller-supplied, 40-hex, or null pre-format-check
calibration_attempt_id                  caller-supplied, or null pre-format-check
expected_manifest_sha256
observed_manifest_sha256                null if the manifest was never read
expected_payload_hash_list_sha256
observed_payload_hash_list_sha256       null if the manifest never validated
expected_payload_count                  300
checked_payload_count
byte_count_mismatch_count
sha256_mismatch_count
missing_or_unreadable_count
run_started_utc
run_completed_utc
artifact_self_hash
```

Evidence acceptance for this shape must call the independent public
`validate_execution_status_semantics()`, which requires a mandatory,
externally trusted `expected_implementation_git_commit` argument -- the
persisted status's own `implementation_git_commit` field is never its own
authority, exactly as `validate_preflight_result_semantics()` already
requires for the R1 preflight artifact. This validator checks ONLY this
gate-level status shape; it is not, and must not be treated as, a
substitute for `src/v8b_data_quality_calibration.py::
validate_result_artifact_semantics()`, the existing acceptance API for
the calibration RESULT artifact this adapter returns unmodified once
`run_data_quality_calibration()` is actually reached.

A gate-level status can never claim a fully clean 300/300 bind (zero
mismatches, zero missing, `checked_payload_count == 300`) -- that state
means the frozen calibration WAS invoked, and its result is the
calibration result artifact (a different schema), never this one; the
semantic verifier enforces this exclusivity.

---

## 9. Static check (fixed)

`run_static_check()` is repository-only: it performs zero V5-B cache
access and zero network access, reading only this module's own source and
introspecting its own **entire** module callable surface. `--static-check`
on the CLI calls it and prints
`V8B_DATA_QUALITY_CALIBRATION_EXECUTION_STATIC_PASS` only after it returns
without raising. It verifies, at minimum:

```text
1. FIXED_V5B_CACHE_ROOT_WINDOWS_PATH equals the exact declared local path
2. EXECUTION_GATE_CONFIRMATION equals the exact gate token
3. run_production_v8b_data_quality_calibration()'s parameter set is
   exactly {confirmation, implementation_git_commit, calibration_attempt_id}
4. no module-level callable defined in this module (via dir(), not
   merely __all__) accepts a cache_root/path/manifest_path/input_dir/
   dataset parameter
5. EXPECTED_V5B_TICKER_COUNT remains 300
6. validate_v5b_manifest_provenance and run_data_quality_calibration are
   still the exact function objects reused from
   src/v8b_data_quality_calibration.py (not shadowed/redefined)
7. the reused Git-verification helper from the preflight module still
   exposes the expected parameter shape
8. this module's own functional source (excluding run_static_check's own
   body) contains none of the forbidden calibration/OHLCV execution
   calls (§6) or network strings
```

---

## 10. No methodology change (fixed)

This adapter does not alter, and is not authorized to alter, any of:

```text
candidate_grid=30 candidates (unchanged)
synthetic_base_count=20 (unchanged)
synthetic_scenario_count=6000 (unchanged)
synthetic_candidate_comparison_count=180000 (unchanged)
headroom_rule=strict (unchanged)
no_defensible_policy_semantics=STRICTEST_DEFENSIBLE (unchanged)
approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1 (unchanged)
approved_plan_commit=8c15426166742c43745e604f6367788af6123c1a (unchanged)
run_data_quality_calibration_signature_and_semantics=unchanged (this
  adapter calls it exactly as declared; it is not modified by this task)
```

---

## 11. Status

```text
status=IMPLEMENTED_NOT_EXECUTED
next_required_gate=INDEPENDENT_CALIBRATION_EXECUTION_ADAPTER_REVIEW,
  followed by a separate human gate before first real filesystem access
  to the V5-B cache via this adapter, which is itself distinct from and
  additional to the human gate already granted for the R1 preflight
```
