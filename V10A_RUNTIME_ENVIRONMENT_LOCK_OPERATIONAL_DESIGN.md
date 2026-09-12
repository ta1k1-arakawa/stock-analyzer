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
V10A_RUNTIME_ENVIRONMENT_LOCK=NOT_CREATED
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
calendar_distribution_name=pandas-market-calendars
calendar_distribution_version=5.4.0
calendar_name=JPX
calendar_source_blob=a7a59b6cf910e325c85fc042459ff57ca8f70613
holiday_source_blob=4c34214d06862e02ac22e946757463f748074fde
runtime_distribution_count=20
```

`runtime_distributions` contains exactly 20 objects. Every object has exactly
the two keys `name` and `version`; no other object keys are permitted.

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
R2 PASS does not authorize live runtime-lock creation. The exact reviewed
implementation SHA is bound only after R1 is committed and reviewed; no
future SHA is fabricated here.

### R3 — live runtime-lock snapshot

After R2 PASS, the repository phased rule is:

```text
connected-GitHub remote precheck 1
  -> Phase A no-network preflight
  -> GPT inspection
  -> connected-GitHub remote precheck 2
  -> fresh point-of-use human authorization
  -> Phase B exactly one reviewed runner process
  -> Phase C no-network result inspection
```

The R3 attempt has a new one-shot identity. Its fresh authorization is
distinct from all V10/V10A promotion and validation authorizations and is
consumed only at the reviewed attempt boundary. The runner may write one
runtime lock and one safe result receipt into exclusive durable roots.

On launch failure, nonzero exit, validation failure, or durable write
failure: do not retry, rerun, delete, reset, repair, reinstall, recreate the
environment, or use an alternate provider. Preserve durable state, complete
Phase C where safe, and return to GPT review.

### R4 — repository lock/evidence commit

Only after an R3 PASS may a repository-only commit add the exact durable
`V10A_RUNTIME_ENVIRONMENT_LOCK.json` bytes and the frozen safe adjudication
artifact:

```text
V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json
```

R4 performs no runner rerun, environment read, package operation, calendar
operation, or gate consumption. It verifies exact bytes through staging and
requires the repository lock SHA-256 to equal the durable lock SHA-256.

### R5 — GPT exact-SHA runtime-lock review

GPT reviews the R4 commit and its exact lock/evidence blobs. Only R5 PASS
establishes `V10A_RUNTIME_ENVIRONMENT_LOCK=GPT_REVIEWED_PASS`. R5 does not
authorize calendar generation; a separate calendar-feasibility
implementation and authorization stage is required.

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
