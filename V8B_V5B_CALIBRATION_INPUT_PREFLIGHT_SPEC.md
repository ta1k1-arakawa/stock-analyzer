# V8B_V5B_CALIBRATION_INPUT_PREFLIGHT_SPEC

```text
study=V8B_HISTORICAL_RESEARCH
document_type=CALIBRATION_INPUT_PREFLIGHT_SPEC
status=IMPLEMENTED_NOT_EXECUTED
implements=R1 of V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md §13.1
satisfies=§3.2's "required future preflight"
approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1
approved_plan_git_commit=8c15426166742c43745e604f6367788af6123c1a
methodology_change=false
real_v5b_cache_accessed_by_this_task=false
```

This document specifies the V5-B calibration input preflight: an isolated,
narrowly scoped adapter that establishes only the **R1** input-provenance /
byte-binding condition (`V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_
DRAFT.md` §13.1) before any future calibration execution. It is not a
calibration run and does not itself perform one.

---

## 1. Scope (fixed)

This preflight satisfies **only**:

```text
R1_v5b_preflight = V5-B cache provenance/preflight validates exactly (§3.2)
```

It does **not** satisfy, claim, or substitute for:

```text
R0  canonical parser/classifier Git blob identity
R2  designated raw payloads reconstructible under the pinned canonical parser/classifier semantics (§3.3)
R3-R9  every other CALIBRATION_RUN_VALID condition (§13.1)
```

Concretely, this preflight:

```text
satisfies_only_R1=true
claims_R2_canonical_parse_validity=false
sets_CALIBRATION_RUN_VALID_true=false
authorizes_calibration_execution=false
authorizes_v5b_cache_access_merely_by_being_implemented=false
```

A `PASS` result from this preflight means the designated raw payload bytes
on disk are exactly the bytes the pinned manifest says they are -- nothing
about whether those bytes parse under the pinned canonical row classifier
(R2), and nothing about any other R0/R3-R9 condition.

---

## 2. Required human gates (fixed)

Two separate, explicit human gates remain required after this
implementation, neither of which this document or this implementation
task grants:

```text
gate_1=required before the FIRST real filesystem access to the V5-B cache
       (i.e. before src/v8b_v5b_calibration_input_preflight.py's
       run_production_v5b_calibration_input_preflight() is ever invoked
       against the real, fixed V5B_CACHE_ROOT)
gate_2=required, separately, after a preflight PASS and before any OHLCV
       parsing (R2, §3.3) or calibration execution
```

Implementing this preflight adapter, its CLI, or its confirmation-token
check does **not** itself constitute either gate. The confirmation token
defined below is a mechanical safeguard against accidental invocation, not
a substitute for a human decision.

```text
gate_1_granted_by_this_task=false
gate_2_granted_by_this_task=false
```

This implementation task itself never invokes
`run_production_v5b_calibration_input_preflight()` against the real fixed
cache root. All tests exercise the preflight logic only against synthetic,
temporary fixtures with `V5B_CACHE_ROOT` monkeypatched.

---

## 3. Fixed production input (§1 of the implementation task)

```text
fixed_cache_root=C:\taiki\hobbies\v5-b-evaluation-cache-retry1
```

`src/v8b_v5b_calibration_input_preflight.py` exposes this as
`V5B_CACHE_ROOT`, a module-level constant. The production entry point,
`run_production_v5b_calibration_input_preflight()`, reads only this path;
it takes no cache-path, manifest-path, input-directory, or dataset
parameter. `scripts/preflight_v8b_v5b_calibration_input.py`, the CLI that
wraps it, likewise exposes no such option -- its only arguments are
`--static-check` (no filesystem access) and `--confirm` /
`--implementation-git-commit` (gated, fixed-root-only execution).

Tests monkeypatch the `V5B_CACHE_ROOT` module attribute itself to a
temporary synthetic fixture directory; this is the only sanctioned way to
exercise the preflight logic against non-production data, and it is never
done against the real path.

---

## 4. Human gate token (§2 of the implementation task)

```text
required_confirmation_token=V5B_CALIBRATION_INPUT_PREFLIGHT_GATE
```

`run_production_v5b_calibration_input_preflight(confirmation=...)` raises
`V5BCalibrationInputPreflightBlocked` unless `confirmation` exactly equals
this token. As stated in §2 above, the token's presence in source code is
not itself an authorization; it only prevents accidental invocation absent
an explicit, deliberate caller decision.

---

## 5. Allowed real-preflight read scope (fixed, exhaustive)

A future, separately authorized execution reads, in order:

```text
1. stat/check existence of the exact fixed cache root (a directory)
2. read the exact bytes of <cache_root>/cache_manifest.json
3. strict-parse and validate those bytes using the existing, frozen
   validate_v5b_manifest_provenance() (src/v8b_data_quality_calibration.py)
   -- reused verbatim, not reimplemented -- which itself: pins the
   whole-manifest SHA-256, requires UTF-8, strict-JSON-parses (rejecting
   malformed JSON and duplicate keys), and validates every structural
   field, including the manifest-declared payload count (exactly 300),
   payload hash-list, and each payload's ticker/relative_path/sha256/
   byte_count shape
4. obtain the manifest-designated 300 payload relative paths from that
   validated manifest -- never any other file in the cache root
5. for those exact 300 payloads only:
   a. resolve the relative path against the cache root and verify the
      resolved path remains inside the cache root (containment check,
      independent of the relative-path-shape check already inside
      validate_v5b_manifest_structure())
   b. check existence (a regular, readable file)
   c. read raw bytes
   d. verify the read byte length equals the manifest's declared
      byte_count exactly
   e. compute SHA-256 over the raw bytes
   f. require that SHA-256 to exactly equal the manifest's declared
      sha256 for that payload
```

No dataset outside these exact 300 manifest-designated files is ever
enumerated or read. No unrelated file in the cache root is read.

---

## 6. Strictly forbidden at preflight (fixed, exhaustive)

This preflight never:

```text
json_parses_raw_payload_body=false
inspects_ohlcv_values=false
calls_parse_ticker_observations=false
calls_run_data_quality_calibration=false
runs_pinned_classifier_against_payload_rows=false
constructs_observed_windows=false
selects_synthetic_bases=false
generates_synthetic_corruption=false
computes_M_fraction_or_M_consecutive=false
evaluates_candidates=false
selects_a_policy=false
accesses_old_T1_T1B_T2_T3=false
accesses_yahoo_network=false
accesses_jpx_network=false
```

Hashing raw payload bytes (§5 step 5e) is allowed only inside a future,
separately authorized real preflight execution -- never performed by this
implementation task itself, which exercises that code path only against
synthetic temporary fixtures.

---

## 7. Input validation -- PASS requires all of (fixed, exhaustive)

```text
1. exact fixed cache root accessible (exists, is a directory)
2. exact manifest SHA-256 pin (EXPECTED_V5B_MANIFEST_SHA256)
3. strict manifest JSON validity (UTF-8, no duplicate keys, well-formed)
4. validate_v5b_manifest_provenance() PASS (reused from
   src/v8b_data_quality_calibration.py, unmodified)
5. exactly 300 designated manifest payload records
6. each designated relative path resolves to remain inside the fixed root
7. each designated payload exists and is a regular, readable file
8. exact manifest byte_count match for all 300
9. exact SHA-256 match for all 300
10. checked payload count exactly 300
11. no substitution / reacquisition / alternate source (enforced
    structurally: there is no parameter through which an alternate
    source could be supplied)
```

Any single failure among the above:

```text
status=BLOCK
blocker=V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED
```

The implementation never continues into R2 canonical parsing or
calibration execution after a BLOCK, and never partially proceeds.

---

## 8. Safe output only (fixed)

A preflight result (PASS or BLOCK) contains only:

```text
schema_version                          "V5B_CALIBRATION_INPUT_PREFLIGHT_RESULT_V1"
study                                   "V8B_HISTORICAL_RESEARCH"
role                                    "R1_V5B_CALIBRATION_INPUT_PREFLIGHT"
status                                  "PASS" | "BLOCK"
detail_reason                           safe structural code, or null on PASS
implementation_git_commit               caller-supplied, 40-hex
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

Never included, under any status:

```text
ticker identities
payload relative paths
payload contents (raw or parsed)
individual ticker/year results
OHLCV values
any T1B/T2/T3 identity
```

`detail_reason` is drawn from a small, fixed, structural vocabulary (e.g.
`CACHE_ROOT_INACCESSIBLE`, `MANIFEST_UNREADABLE`, `MANIFEST_PROVENANCE_
INVALID:<inner reason>`, `DESIGNATED_PAYLOAD_COUNT_MISMATCH`, `PAYLOAD_
PATH_ESCAPE_DETECTED`, `PAYLOAD_BINDING_FAILED`) -- none of these values
ever contain a ticker symbol or a file path.

---

## 9. No methodology change (fixed)

This preflight does not alter, and is not authorized to alter, any of:

```text
candidate_grid=30 candidates (unchanged)
synthetic_base_count=20 (unchanged)
synthetic_scenario_count=6000 (unchanged)
synthetic_candidate_comparison_count=180000 (unchanged)
headroom_rule=strict (unchanged)
no_defensible_policy_semantics=STRICTEST_DEFENSIBLE (unchanged)
approved_plan_version=V8B_DATA_QUALITY_CALIBRATION_PLAN_V1 (unchanged)
approved_plan_commit=8c15426166742c43745e604f6367788af6123c1a (unchanged)
core_semantic_verifier_behavior=unchanged (validate_result_artifact_semantics
  and its supporting functions in src/v8b_data_quality_calibration.py are
  not modified by this task)
```

A `PASS` from this preflight is consumed, in a future task, only as one
input to R1 of `CALIBRATION_RUN_VALID` (§13.1). It never itself sets
`CALIBRATION_RUN_VALID=true`, which additionally requires R0 and R2-R9.

---

## 10. Relationship to existing modules

```text
src/v8b_data_quality_calibration.py::validate_v5b_manifest_provenance
    reused verbatim (imported, not reimplemented) for manifest-level
    validation (§5 steps 2-3 above).
src/v8b_data_quality_calibration.py::V8BCalibrationBlocked
    caught internally when validate_v5b_manifest_provenance() raises;
    its .reason is folded into this module's own detail_reason as
    "MANIFEST_PROVENANCE_INVALID:<reason>", never surfaced as this
    module's outward .reason (which is always PREFLIGHT_BLOCKER).
```

---

## 11. Status

```text
status=IMPLEMENTED_NOT_EXECUTED
next_required_gate=CHATGPT_PREFLIGHT_ADAPTER_REVIEW, followed by a
  separate human gate before first real filesystem access to the V5-B
  cache (§2 gate_1), and another separate human gate after a real PASS
  before R2 parsing or calibration execution (§2 gate_2)
```
