# V10A T0 Calendar Input-Binding Bridge Design Draft

```text
study_identity=V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR
design_status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
scope=V9_009_T0_CALENDAR_INPUT_BINDING_ONLY
T0_AUTHORIZED=false
HISTORICAL_EVALUATION_AUTHORIZED=false
PRIVATE_SEALED_ACCESS_AUTHORIZED=false
profitability_evidential_capacity=ZERO
future_profitability_established=false
```

## 1. Successor boundary

V10A does not reopen, repair, reuse, or reinterpret a failed V9 calendar
study. The GPT-reviewed V10A canonical calendar replaces only the failed
historical TSE session/date authority mechanism used by the inherited V9_009
economic/T0 methodology.

The inherited evaluation period, target, labels, costs/slippage, thresholds,
models, partitions, portfolio rule, stopping rule, and profitability criterion
are unchanged. This bridge selects no calendar, date, or outcome based on
results and creates no scientific-methodology change outside calendar input
binding.

## 2. Sole canonical calendar authority

The later real T0 path may consume calendar dates only from the repository-root
file `V10A_CANONICAL_CALENDAR.json`. Before any research-data, cache, universe,
price, outcome, target, label, return, or model-relevant read, it must require
all of the following exact bindings:

```text
artifact_repo_path=V10A_CANONICAL_CALENDAR.json
artifact_git_blob_sha1=b3d9dee8fb20abfd966400873a7f1ff18df2880b
artifact_file_sha256=b24cab4b322a216e6cf24e55524c5b360ac4de6404f4b161014a1a2d0d64814d
canonical_calendar_sha256=2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0
schema_version=V10A_CANONICAL_CALENDAR_V1
coverage_start=2017-01-01
coverage_end=2026-01-31
trading_date_count=2217
generator_implementation_git_sha=0830d86675f447231e77b1687c7a23cf0b135d7f
runtime_environment_lock_sha256=d7f54bc69029ba9b25a9920e867fe6487745af6ef985898bad91bd951003fc3a
```

The future implementation must resolve this fixed file only relative to a
validated repository root and reject symlinks, junction escapes, non-files,
or a resolved path outside that root. It must calculate the raw file SHA-256
and inspect the Git blob at the exact reviewed checkout before parsing. It
must reject any mismatch without an alternate path or calendar source.

The artifact must have its frozen exact field set, validate its canonical
self-excluding `canonical_calendar_sha256`, and have the fixed provenance and
coverage values above. Its dates must be a sorted, unique, exact
`YYYY-MM-DD` sequence, count exactly `2217`, and remain within the inclusive
coverage. The implementation must not create dates through `pd.bdate_range`,
weekday inference, operating-system locale data, Yahoo, J-Quants, or any
fallback/alternate calendar.

## 3. Required safe-receipt binding

Before any such research-data access, the real T0 path must additionally
validate repository-root `V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json`:

```text
receipt_repo_path=V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json
receipt_git_blob_sha1=da76889db285062a8f9ac902263ed7c2e63dc43a
receipt_file_sha256=e7266539c8a11c59d3be775623fde6023e835940e981f17c03f6e0bdf65b005c
status=PASS
failure_code=NONE
calendar_artifact_created=true
design_git_sha=b14cc5510685210e928000af0815e188bc1aadc0
generator_implementation_git_sha=0830d86675f447231e77b1687c7a23cf0b135d7f
anchor_2020_10_01=INELIGIBLE
anchor_2020_10_02=ELIGIBLE
```

Its exact receipt schema and field set must validate. The receipt's
`canonical_calendar_sha256` and `trading_date_count` must equal the validated
artifact values; both must bind the same reviewed runtime-environment-lock
SHA-256. Any mismatch is an input-binding failure, not a scientific result.

## 4. Fixed real CLI boundary

Future implementation selects option A: remove `--calendar-file` entirely
from the real T0 CLI. The only calendar input is the fixed repository-root
canonical artifact above. This is the smaller fail-closed surface: no caller
calendar path exists to validate, retain, substitute, or accidentally route
to a noncanonical JSON file.

No real CLI or internal real-T0 entrypoint may accept a calendar path, JSON,
provider, coverage, anchor, fallback, or calendar-selection override. Synthetic
test helpers may construct in-memory date sequences only outside the real-T0
production boundary and must not create a production bypass.

## 5. Required validation order and stop behavior

The future real-T0 implementation must perform these stages in order:

1. Verify repository identity, authoritative branch, exact reviewed
   implementation SHA, clean tree, and fixed bridge-design provenance.
2. Resolve, raw-hash, Git-blob-bind, parse, and fully validate the fixed
   canonical artifact and its fixed provenance.
3. Resolve, raw-hash, Git-blob-bind, parse, and fully validate the fixed safe
   receipt and its fixed PASS facts.
4. Validate artifact/receipt cross-bindings: calendar digest, trading-date
   count, runtime-lock SHA-256, and generator implementation SHA.
5. Validate trading-date structural rules: canonical labels, sorted uniqueness,
   exact count, and inclusive coverage.
6. Validate frozen anchor facts: `2020-10-01=INELIGIBLE` and
   `2020-10-02=ELIGIBLE`.
7. Derive the exact calendar grid exclusively from the validated artifact
   trading-date sequence.
8. Only after stages 1–7 have passed may a separately authorized T0 path read
   training/evaluation caches, universe content, prices, outcomes, targets,
   labels, returns, or invoke model-relevant work.

Any failure through stage 7 must stop before research-data/outcome access and
produce `NO_VERDICT_DATA_INCOMPATIBLE` or the distinct fixed input-binding
failure path. It must not be collapsed into `STOP`, profitability, or any
scientific result; it must not trigger repair, fallback, or retry.

## 6. Exact failure classification

The bridge has three disjoint failure classes. Classification is determined by
the stage and origin of the failure; no class may be silently relabeled as
another class.

### A. Pre-T0 governance / provenance failure

Failure of repository identity, authoritative branch, expected reviewed HEAD,
clean-tree state, or bridge-design provenance is a `GOVERNANCE/PREFLIGHT
FAILURE`. It must stop before T0 computation, read no cache or outcome, and
write no T0 safe-result JSON. It must never be mapped to
`NO_VERDICT_DATA_INCOMPATIBLE`, `T0_RESULT=STOP`, or `T0_RESULT=CONTINUE`.

### B. Expected calendar input-binding contract failure

After governance preflight passes, an anticipated failure of the fixed V10A
artifact/receipt contract is a calendar input-binding failure with the
existing `T0DataIncompatible` external semantics. This includes a missing or
non-file artifact/receipt, unsafe symlink or junction/path resolution, file
SHA or Git-blob mismatch, JSON/schema/fieldset or fixed-provenance mismatch,
artifact self-digest or artifact/receipt cross-binding mismatch,
trading-date structural/count/coverage mismatch, frozen-anchor mismatch, or
deterministic calendar-grid derivation incompatibility.

This class must stop before any training/evaluation cache, universe content,
price, outcome, target, label, return, or model-relevant read. It emits the
existing validated safe result `T0_RESULT=NO_VERDICT_DATA_INCOMPATIBLE` with
`cache_identity=false` and `exact_semantics=false`, using the existing
zero/synthetic provenance representation because research caches were not
opened. It never emits `T0_RESULT=STOP` or `T0_RESULT=CONTINUE`, and never
retries, repairs, or falls back.

### C. Implementation / wrapper failure

An unexpected programming/runtime exception from the bridge validator, an
unexpected serializer or result-construction failure, or a wrapper defect is
an `IMPLEMENTATION_FAILURE`. It must not be converted to `T0DataIncompatible`,
must not emit `NO_VERDICT_DATA_INCOMPATIBLE`, and must not emit `STOP` or
`CONTINUE`. It uses the existing nonzero implementation-failure path and has
no retry.

`T0_RESULT=STOP` remains reserved for a later separately authorized
scientific T0 kill-screen result, after exact calendar inputs have passed and
the authorized research data has been accessed. This classification section
does not change any inherited V9 methodology.

## 7. Inherited causal derivation

Only the exact validated `trading_dates` sequence may supply the inherited:

- global three-JPX-trading-day signal grid;
- D1/D2/D3 lookup; and
- causal `MONTH_START` monthly cutoff.

The implementation must make no post-observation calendar modification,
neighboring-date repair, omission repair, or outcome-dependent decision.

## 8. Future implementation and authority chain

This design grants no T0 or historical-evaluation authority. A later task
must first implement the fixed CLI/input-binding change with synthetic targeted
tests, then receive GPT exact-SHA review. Only after that review may a later
T0 Phase A be designed or authorized. If the eventual T0 crosses a
protected/private/outcome boundary, fresh point-of-use human authority remains
separately required under the applicable frozen governance.

Calendar feasibility and this calendar-input bridge have
`profitability_evidential_capacity=ZERO`; neither establishes future
profitability.
