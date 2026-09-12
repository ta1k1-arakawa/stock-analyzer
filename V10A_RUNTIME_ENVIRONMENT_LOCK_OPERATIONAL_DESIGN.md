# V10A Runtime Environment Lock Operational Design

```text
study_identity=V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR
design_status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
design_frozen=false
human_freeze_approved=false
artifact=V10A_RUNTIME_ENVIRONMENT_LOCK.json
```

## 1. Scope and inheritance

This document defines only the operational sequence for creating and
reviewing the V10A runtime-environment lock before any semantic JPX calendar
generation. It does not create the lock, inspect the canonical Python
environment, generate a calendar, inspect sessions or dates, or authorize
execution.

V10A inherits the V10 runtime-lock scientific and reproducibility semantics
unchanged. The only adaptations are:

1. a new V10A artifact/provenance namespace, so V10 failed-study state is not
   reused as V10A authority; and
2. the corrected V10A release-artifact source identities.

These are operational and provenance adaptations only. They do not change
coverage, `calendar_name`, session/date semantics, anchors, feasibility
budget, labels, target, evaluation period, partitions, costs, slippage,
thresholds, models, search space, or stopping rules.

The old V10 JPX blob
`0c2041b1300d1dbbd505202b00ac0ada38c712e1` is historical failed-study
provenance and MUST NEVER be accepted by V10A.

## 2. Starting state and authority boundary

The starting state for this design is:

```text
V10A_ENVIRONMENT_STATE=CANONICAL_FROZEN
V10A_CANONICAL_ENVIRONMENT_PROMOTED=true
V10A_ENVIRONMENT_FROZEN=true
V10A_EXECUTION_AUTHORIZED=false
V10A_CALENDAR_GENERATION_AUTHORIZED=false
V10A_T0_AUTHORIZED=false
V10A_HISTORICAL_EVALUATION_AUTHORIZED=false
V10A_RUNTIME_ENVIRONMENT_LOCK=NOT_CREATED_FOR_PROMOTION
V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_AUTHORIZED=false
future_profitability_established=false
```

The canonical-environment promotion/freeze state is not calendar authority.
Runtime-lock creation is a separate, read-only, fresh-authority operation.
The V10A runtime lock is not created in this design task.

The already-reviewed provenance inputs are bound as follows:

```text
FROZEN_V10A_DESIGN_SHA=b14cc5510685210e928000af0815e188bc1aadc0
V10A_FREEZE_RECORD_SHA=86ceda3dee531b08afa5db4df7af1298ca770fad
P5_REVIEWED_P4_SHA=1280c222f6114aa0363684a22d9b24e47cb1d5e0
P5_BOOKKEEPING_SHA=4090c9693ee7afa0fb6439a187d2cef4399f945c
FINAL_FREEZE_EVIDENCE_BLOB=d880b84fa00233e58653739fd510385fdf94de4e
FINAL_FREEZE_EVIDENCE_SHA256=658e264a70ab15ba402e7bf56d5e4b8abe5d81f2f7bb22f28bc797b7b8062b01
P3_ADJUDICATION_BLOB=6f5571961bd1fa533ea7aea30730dc3383ddc2e8
```

Before any future live observation, tooling must bind the exact
authoritative branch, reviewed implementation SHA, clean tree, frozen V10A
design and freeze record, P5-reviewed P4 SHA, P5 bookkeeping SHA, final
evidence blob/SHA-256, P3 adjudication blob, and `CANONICAL_FROZEN` state.
No later live observation may occur after an earlier provenance failure.

## 3. V10A runtime-lock artifact

The future repository artifact is exactly:

```text
V10A_RUNTIME_ENVIRONMENT_LOCK.json
```

Its exact top-level key set is:

```text
schema_version
python_version
calendar_distribution_name
calendar_distribution_version
calendar_name
calendar_source_blob
holiday_source_blob
runtime_distributions
runtime_distribution_count
```

No extra or missing top-level fields are permitted. Fixed values are:

```text
schema_version=V10A_RUNTIME_ENVIRONMENT_LOCK_V1
python_version=3.12.10
calendar_distribution_name=pandas_market_calendars
calendar_distribution_version=5.4.0
calendar_name=JPX
calendar_source_blob=a7a59b6cf910e325c85fc042459ff57ca8f70613
holiday_source_blob=4c34214d06862e02ac22e946757463f748074fde
runtime_distribution_count=20
```

`runtime_distributions` contains exactly 20 objects. Every object has exactly
the two keys `name` and `version`; no other object keys are permitted.

The top-level `calendar_distribution_name` deliberately preserves the
inherited exact V10 contract and therefore uses the underscore form
`pandas_market_calendars`. This is distinct from the normalized installed
distribution representation in `runtime_distributions`, whose corresponding
entry is `pandas-market-calendars`. These two fields have intentionally
different exact representations and must not be conflated.

Installed distribution names are normalized by lowercasing and replacing each
maximal run of `-`, `_`, or `.` with `-`. Empty normalized names, duplicate
normalized names, and duplicate distributions after normalization fail
closed. The normalized objects are sorted lexicographically by `name`.

The exact name/version mapping is the mapping already established in
`V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json`:

```text
cffi==2.1.1
charset-normalizer==3.5.1
cryptography==50.0.1
exchange-calendars==4.13.2
korean-lunar-calendar==0.4.0
numpy==2.5.2
pandas==3.0.5
pandas-market-calendars==5.4.0
pdfminer-six==20260107
pdfplumber==0.11.10
pillow==12.3.0
pip==25.0.1
pycparser==3.0
pyluach==2.3.0
pypdfium2==5.13.0
python-dateutil==2.9.0.post0
six==1.17.0
toolz==1.1.0
tzdata==2026.3
xlrd==2.0.2
```

No package may be added, removed, upgraded, downgraded, resolved, or selected
during lock creation. The runtime lock is a snapshot of the existing
canonical environment, not an installation or repair input.

## 4. Canonical bytes and source identity

The lock bytes are canonicalized exactly as follows:

```text
encoding=UTF-8
ensure_ascii=false
sort_keys=true
separators=(',', ':')
allow_nan=false
final_terminator=exactly_one_LF
```

The lock contains no self-hash. Later artifacts bind
`runtime_environment_lock_sha256` to the SHA-256 of the exact repository
lock bytes.

No alternative mapping embedding, alternate serializer, newline conversion,
or self-referential hash is permitted.

The only accepted calendar source identity is the V10A release artifact:

```text
pandas-market-calendars==5.4.0
calendar_source_file=pandas_market_calendars/calendars/jpx.py
calendar_source_blob=a7a59b6cf910e325c85fc042459ff57ca8f70613
holiday_source_file=pandas_market_calendars/holidays/jp.py
holiday_source_blob=4c34214d06862e02ac22e946757463f748074fde
```

Future tooling must locate the two installed files through distribution
metadata and read exact raw bytes. It must compute each Git blob as
`sha1(b"blob " + ascii_decimal_byte_length + b"\0" + raw_bytes)` and fail
closed on either mismatch. It must not decode, normalize newlines, compare
ASTs, import the package, or reconstruct source.

## 5. Future live snapshot boundary

Runtime-lock creation is a protected, read-only, no-network operation. The
only allowed interpreter is exactly:

```text
.venv-real-execution\Scripts\python.exe
```

Activated `python`, `.venv`, PATH `python`/`python3`/`py`, and alternate
virtual environments are forbidden. Package installation, repair, download,
resolution, and mutation are forbidden.

The future runner may inspect only:

- `sys.version_info`;
- `sys.executable` identity;
- installed distribution metadata through stdlib `importlib.metadata`;
- exact installed `jpx.py` bytes; and
- exact installed `jp.py` bytes.

It must not import or instantiate `pandas_market_calendars`, JPX calendar
classes, or calendar objects. It must not call `get_calendar`, `schedule`,
generate sessions, inspect dates or `market_close`, inspect the 2020 anchors,
read research data/prices/outcomes, run T0, or perform historical evaluation.

## 6. Future implementation and execution chain

### R1 — offline implementation

A later implementation task creates exactly:

```text
scripts/v10a_runtime_environment_lock_runner.py
tests/test_v10a_runtime_environment_lock_runner.py
```

R1 performs no canonical-environment read. Tests use synthetic/temp fixtures
only. The production CLI must not expose `--observations`, `--synthetic`,
`--skip-*`, `--force`, `--repair`, or `--install` options, and must not offer
injection seams for fake observations or alternate interpreters/authorities.

### R2 — GPT exact-SHA implementation review

GPT must independently review the complete R1 implementation commit,
including its production entrypoint, exact constants, staged observation
ordering, canonical serialization, safe publication, failure contract, and
targeted tests. R2 PASS requires `CRITICAL=0`, `HIGH=0`, and `MEDIUM=0`.
GPT records one exact R2-reviewed implementation SHA, the exact reviewed
runner Git blob, and the exact reviewed test Git blob. R2 PASS does not
authorize live runtime-lock creation. These exact values are bound only after
R1 is committed and reviewed; no future SHA is fabricated here.

### R3 — live runtime-lock snapshot

The first real attempt reached runtime `PASS` but is terminally
non-promotable because the then-frozen execution-evidence schema did not
carry all wrapper/GPT authority facts required by later repository review:

```text
R3_ATTEMPT_1_TERMINAL_NONPROMOTABLE=true
R3_ATTEMPT_1_RUNTIME_RESULT=PASS
R3_ATTEMPT_1_PROMOTION_CHAIN_RESULT=BLOCK
```

The attempt-1 lock and execution evidence are preserved in place. They are
not deleted, reset, rewritten, retried, or reused as R4/R5 promotion
authority. The attempt-1 authorization is consumed and cannot be reused.
The remediation is governance-only and does not reinterpret the runtime
snapshot or change the scientific calendar methodology.

Every future attempt is a new identity. After a future R2 PASS, the phased
rule is:

```text
R3 remote precheck 1 through connected GitHub
  -> R3 Phase A no-network preflight
  -> GPT inspection
  -> R3 remote precheck 2 through connected GitHub
  -> fresh point-of-use human authorization
  -> R3 Phase B exactly one reviewed runner process
  -> R3 Phase C no-network result inspection
```

R3 remote precheck 1 must verify through connected GitHub that the
authoritative branch HEAD equals the exact future R2-reviewed SHA immediately
before Phase A. A mismatch is `EXPECTED_HEAD_MISMATCH` and stops the chain.

Phase A must be local and no-network. Before any canonical-environment
observation it verifies the exact branch, local HEAD, local tracking ref,
clean tree, reviewed R2 runner/test/design blobs, frozen V10A design/freeze/P5
provenance, final-freeze evidence/adjudication provenance, and exclusive
durable-root availability. Phase A writes no durable attempt state, launches
no canonical Python, and does not read the canonical environment, wheelhouse,
packages, or calendar.

After Phase A PASS and before fresh authorization, R3 remote precheck 2 must
again verify through connected GitHub that the authoritative branch HEAD is
the exact future R2-reviewed SHA. A mismatch is `EXPECTED_HEAD_MISMATCH` and
stops the chain. No `git fetch` is required inside the no-network Phase A/B/C
path.

The future R3 attempt has a fresh one-shot identity, for example
`R3_RUNTIME_LOCK_ATTEMPT_2`. Its authorization is distinct from all V10 and
V10A authorizations and is consumed only at the reviewed attempt boundary.
Phase B may launch only the exact R2-reviewed runner once, against
`.venv-real-execution`, and may only read the permitted runtime metadata and
source bytes and write one runtime lock and one runner-produced execution
evidence receipt into exclusive durable roots. It performs no repo commit,
git write, mutation, installation, network operation, or calendar operation.

```text
R3_CREATES_REPOSITORY_COMMIT=false
```

Phase C is read-only. It performs no runner rerun and no environment
observation. It may inspect only the durable runtime lock, durable
runner-produced execution evidence, durable wrapper prelaunch/result files,
stdout/stderr captures, and repository provenance. The R3 adjudication
artifact does not yet exist and MUST NOT be inspected or created in Phase C.
Phase C reports safe facts to GPT; it creates no repository commit and does
not consume authority. The fresh R3 authority is never inferred from R2
PASS, the terminal attempt-1 PASS, or any earlier V10A authorization.

On launch failure, nonzero exit, validation failure, or durable write
failure: do not retry, rerun, delete, reset, repair, reinstall, recreate the
environment, or use an alternate provider. Preserve durable state, complete
Phase C where safe, and return to GPT review.

### GPT R3 adjudication and future artifact contract

After Phase C, GPT adjudicates the reported safe facts as `PASS` or `BLOCK`.
Only if GPT adjudicates R3 `PASS` does GPT freeze the exact safe values that
R4 may record. R4 then deterministically creates
`V10A_RUNTIME_ENVIRONMENT_LOCK_R3_ADJUDICATION.json` from those already
frozen safe values. The artifact therefore does not exist during Phase C,
and R4 does not infer or recalculate facts from a live environment.

The future adjudication artifact has exactly this top-level key set and no
extra or missing fields:

```text
schema_version
study_identity
attempt_identity
r2_reviewed_sha
runtime_lock_runner_git_blob_sha1
runtime_lock_test_git_blob_sha1
runtime_lock_design_git_blob_sha1
remote_precheck_1
phase_a
remote_precheck_2
point_of_use_remote_check
phase_b
phase_c
authorization_consumed
retry_authorized
process_start_attempted
process_started
process_exit_code
runtime_lock_size
runtime_lock_sha256
execution_evidence_size
execution_evidence_sha256
execution_evidence_status
execution_failure_code
python_version
runtime_distribution_count
exact_package_mapping
calendar_source_blob
holiday_source_blob
network_requests
package_installations
environment_mutations
calendar_imports
calendar_object_creations
calendar_dates_inspected
protected_private_research_reads
t0_run
runtime_result
promotion_chain_result
runtime_environment_lock_gpt_reviewed_pass
execution_authorized
calendar_generation_authorized
t0_authorized
historical_evaluation_authorized
future_profitability_established
```

Its schema version is exactly
`V10A_RUNTIME_ENVIRONMENT_LOCK_R3_ADJUDICATION_V1`. Its canonical bytes are
UTF-8 with `ensure_ascii=false`, `sort_keys=true`,
`separators=(',', ':')`, `allow_nan=false`, and exactly one final LF. It
contains no self-hash.

Field sources are fixed as follows. GPT-fixed/repository provenance supplies
`schema_version`, `study_identity`, `attempt_identity`, `r2_reviewed_sha`,
the runner/test/design blobs, and `remote_precheck_1`,
`remote_precheck_2`, and `point_of_use_remote_check`. Phase A supplies
`phase_a`. The Phase-B durable wrapper supplies `phase_b`,
`authorization_consumed`, `retry_authorized`, `process_start_attempted`,
`process_started`, and `process_exit_code`. Phase C independently verifies
the durable artifacts and supplies `phase_c`, lock/evidence sizes and
hashes, runner evidence status/failure code, Python/package/source facts,
and all operation counters. GPT R3 adjudication supplies `runtime_result`
and `promotion_chain_result`. The six downstream/profitability fields are
fixed non-authority values and must remain false.

For an R3 promotable PASS, require exactly: all remote-precheck and
Phase-A/B/C fields equal `PASS`; `authorization_consumed=true`;
`retry_authorized=false`; `process_start_attempted=true`;
`process_started=true`; integer `process_exit_code=0`;
`execution_evidence_status=PASS`; `execution_failure_code=NONE`;
`python_version=3.12.10`; `runtime_distribution_count=20`;
`exact_package_mapping=true`; corrected JPX/JP source blobs exact; all
prohibited-operation counters equal zero; `t0_run=false`;
`runtime_result=PASS`; `promotion_chain_result=PASS`; and every downstream
authority/profitability field false.

R4 must construct the exact canonical bytes only after GPT R3 PASS. It must
validate the exact key set, every SHA/hash format, all fixed PASS semantics,
the lock hash/size against the copied durable lock, the evidence hash/size
against the copied durable runner evidence, and every exact R2/blob binding.
No alternate value, missing-field default, post-hoc repair, or fallback is
allowed. The adjudication contains no local paths, raw authorization,
human identity, protected data, source bytes, prices, tickers, or outcomes.

### R4 — repository lock/evidence/adjudication commit

Only after a future R3 PASS may a repository-only commit add exactly these
five authorized changes:

```text
V10A_RUNTIME_ENVIRONMENT_LOCK.json
V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json
V10A_RUNTIME_ENVIRONMENT_LOCK_R3_ADJUDICATION.json
PROJECT_STATE.md
PROJECT_DECISION_LOG.md
```

R4 must be exactly one direct commit after the exact future R2-reviewed SHA:

```text
R4_PARENT_SHA=exact_R2_reviewed_SHA
ahead_by=1
behind_by=0
```

There is no intermediate commit between R2 and R4. R4 must not modify this
design, the R1 runner or test, the frozen scientific design, the V10A freeze
record, P5 promotion/freeze evidence or adjudication, the final-freeze
candidate/runner/test, attempt-1 evidence/adjudication, or package and
environment authority files.

Before commit and push, R4 mechanically verifies that the reviewed runner,
reviewed test, and this operational-design blobs equal their exact R2-bound
blobs. Both durable runner artifacts are copied byte-for-byte without
reserialization; their repository/staged SHA-256 values and sizes must equal
the durable R3 values. The R3 adjudication is created from safe wrapper facts
only. R4 performs no runner rerun, environment read, package operation,
calendar operation, or gate consumption.

The lock and runner execution evidence remain byte-for-byte copies of the
durable R3 artifacts. The separate R3 adjudication is the only artifact
carrying the wrapper/GPT authority facts. Together, the three artifacts bind
the exact R2-reviewed SHA, reviewed runner/test/design blobs, lock and
execution-evidence Git/SHA-256/size values, all remote/phase results and
process facts, fresh authorization consumption, `retry_authorized=false`,
the exact runtime snapshot, all prohibited-operation counters zero, and all
downstream authority/profitability values false.

### R5 — GPT exact-SHA runtime-lock review

GPT reviews the R4 commit and all three exact artifacts. R5 must mechanically
verify the R4 parent, `ahead_by=1`, `behind_by=0`, exactly the five
R4-authorized files, unchanged R2-bound runner/test/design blobs, unchanged
frozen scientific/protected artifacts, exact durable-to-repository lock and
execution-evidence bytes and SHA-256 values, exact R3 adjudication bindings,
one-shot authorization, `retry=false`, runtime snapshot PASS, zero
prohibited-operation counters, and no calendar/T0/historical authority
claim.

Only R5 PASS with `CRITICAL=0`, `HIGH=0`, and `MEDIUM=0` establishes
`V10A_RUNTIME_ENVIRONMENT_LOCK=GPT_REVIEWED_PASS`. R5 does not authorize
semantic calendar generation. Even after R5 PASS,
`V10A_EXECUTION_AUTHORIZED`, `V10A_CALENDAR_GENERATION_AUTHORIZED`,
`V10A_T0_AUTHORIZED`, and `V10A_HISTORICAL_EVALUATION_AUTHORIZED` remain
false, and future profitability remains unestablished.

## 7. Result and failure contract

The exact future safe receipt is
`V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json`. It contains safe
provenance only and no machine-local paths, raw exception text, human
identity/raw authorization, source bytes, prices, ticker identities,
outcomes, or protected data.

The closed failure-code enum is:

```text
NONE
UNAUTHORIZED_OPERATION_OBSERVED
PROVENANCE_BINDING_FAILURE
WRONG_CANONICAL_INTERPRETER
PYTHON_VERSION_MISMATCH
PACKAGE_SET_MISMATCH
CALENDAR_DISTRIBUTION_VERSION_MISMATCH
CALENDAR_SOURCE_BLOB_MISMATCH
HOLIDAY_SOURCE_BLOB_MISMATCH
RUNTIME_LOCK_CANONICALIZATION_FAILURE
DURABLE_OUTPUT_COLLISION
DURABLE_WRITE_FAILURE
```

Failure evaluation is fail-closed and ordered: unauthorized-operation guard;
provenance; interpreter identity; Python version; normalized package set;
calendar distribution/version; source blobs; canonical serialization; and
durable publication. A later failure or PASS is not collected after an
earlier terminal failure.

A PASS requires all of the following:

```text
failure_code=NONE
runtime_distribution_count=20
exact_package_mapping=true
python_version=3.12.10
pandas_market_calendars=5.4.0
exchange_calendars=4.13.2
calendar_source_blob=a7a59b6cf910e325c85fc042459ff57ca8f70613
holiday_source_blob=4c34214d06862e02ac22e946757463f748074fde
durable_lock_created=true
network_requests=0
package_installations=0
environment_mutations=0
calendar_object_creations=0
calendar_dates_inspected=0
protected_or_private_research_reads=0
t0_run=false
```

Before the lock is reviewed at R5, safe result state remains non-authorizing:

```text
V10A_CANONICAL_ENVIRONMENT_PROMOTED=true
V10A_ENVIRONMENT_FROZEN=true
V10A_EXECUTION_AUTHORIZED=false
V10A_CALENDAR_GENERATION_AUTHORIZED=false
V10A_T0_AUTHORIZED=false
V10A_HISTORICAL_EVALUATION_AUTHORIZED=false
future_profitability_established=false
```

Runtime-lock creation or a runtime-lock receipt PASS never promotes or
freezes the environment and never authorizes calendar generation, T0,
historical evaluation, network/private/sealed access, installation, or
mutation.

## 8. Non-execution statement

This design task performs no canonical Python launch, canonical-environment
read, wheelhouse read, runtime-lock creation, calendar import/object/date
operation, package installation, environment mutation, network data/package
request, private/sealed read, T0, historical evaluation, backtest, model fit,
or human-gate consumption.
