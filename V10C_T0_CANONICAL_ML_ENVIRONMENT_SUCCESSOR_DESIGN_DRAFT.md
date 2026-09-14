# V10C T0 Canonical ML Environment Successor Design Draft

```text
document_role=V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN
study_identity=V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR
status=DRAFT_AWAITING_GPT_REVIEW
predecessor_study=V10C_T0_SUCCESSOR_TRAINING_INPUT_BINDING
predecessor_environment_state=PRE_GATE_ENVIRONMENT_BLOCK
```

## 1. Boundary and purpose

The reviewed V10C successor input-binding implementation at
`be9d3f1e7429df4f0e0a8626807b01c4040a9494` received exact-SHA GPT PASS.
Its first no-network Phase-A attempt stopped before the protected boundary
because the canonical protected interpreter could not import `lightgbm`.
No training or evaluation payload bytes were read, no model was fit, no T0
was run, and no human gate was consumed.

This document defines a new prospective environment successor only.  It does
not change the V9 T0 study, repair the prior Phase-A attempt, or infer any
T0 result.  It authorizes no package resolution or environment mutation.

The observed general-environment versions `lightgbm==4.6.0` and
`scikit-learn==1.9.0` are pre-outcome version-selection authority for the two
new direct T0 dependencies.  They were observed before any T0 outcome and
are not a result of outcome-dependent tuning.

## 2. Immutable predecessor environment

The successor starts from the exact reviewed 20-package canonical lock:

```text
requirements-real-execution.lock.txt
git_blob_sha1=99395e7a5be752fb3ea92fd31be0334f38792261
sha256=eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444
package_count=20
```

The predecessor package set is exactly:

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

Every one of these pins is immutable.  If resolving the new direct
dependencies would change, remove, or upgrade any predecessor pin, the
result is `CHATGPT_DECISION_REQUIRED` and the workflow stops.

## 3. Successor direct dependencies

The new direct T0 specification is prospective and must use a new file:

```text
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt
```

It contains the unchanged protected direct requirements plus exactly these
new direct requirements:

```text
lightgbm==4.6.0
scikit-learn==1.9.0
```

These versions are not upgradeable or replaceable by another ML library in
this study.  They do not change Ridge, LightGBM, scaler, random-state, or
other frozen T0 semantics.

The new direct-spec file is not created by this design task.  Its future
Git blob and file SHA-256 are unknown until the design is reviewed and the
artifact is created in a later authorized task.

## 4. Transitive closure policy

No transitive package version is selected from the general environment in
this design.  The future Windows-grounded resolver must mechanically derive
the closure for CPython `3.12.10` on `Windows/AMD64/win-amd64`, using the
unchanged predecessor lock as both requirements and constraints, together
with the two new direct pins.

Expected dependency categories include `scipy`, `joblib`,
`threadpoolctl`, and `narwhals`, but their versions remain unresolved.  The
resolver may add only packages mechanically required by the exact direct
requirements and predecessor pins.

```text
PREDECESSOR_PACKAGE_SET = the exact 20 pinned packages above
SUCCESSOR_PACKAGE_SET = PREDECESSOR_PACKAGE_SET
  + lightgbm==4.6.0
  + scikit-learn==1.9.0
  + exact resolver-required transitive packages
SUCCESSOR_DELTA_PACKAGE_SET = packages in SUCCESSOR_PACKAGE_SET whose
  normalized names are absent from PREDECESSOR_PACKAGE_SET
```

Only `SUCCESSOR_DELTA_PACKAGE_SET` may later be installed.  The existing 20
packages must not be reinstalled or upgraded.  If no wheel-only closure
exists while retaining every predecessor pin, stop with
`CHATGPT_DECISION_REQUIRED`.

## 5. One bounded Windows resolution

After this design receives GPT exact-SHA PASS, explicit human
`DESIGN_FREEZE_ONLY` approval, durable approval-record creation, and GPT
PASS of that record, a separate resolution implementation may be created.
The design itself consumes no gate and performs no resolution.

The future resolution must use the canonical interpreter and resolver:

```text
canonical_interpreter=.venv-real-execution\\Scripts\\python.exe
python_version=3.12.10
platform=Windows/AMD64/win-amd64
resolver=pip==25.0.1
index=PYPI_OFFICIAL_SIMPLE
```

It may perform exactly one bounded public package-index resolution, with an
empty durable wheelhouse and no installation.  Its effective options must be
equivalent to:

```text
python -m pip download
  --dest <NEW_EMPTY_DURABLE_WHEELHOUSE>
  --only-binary=:all:
  --no-cache-dir
  --disable-pip-version-check
  --no-input
  --progress-bar off
  --retries 0
  --timeout 15
  --index-url https://pypi.org/simple
  --requirement requirements-real-execution.lock.txt
  --requirement V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt
  --constraint requirements-real-execution.lock.txt
```

There is no alternate index, source distribution, package substitution,
unbounded retry, or second resolution to obtain preferable versions.  The
first complete resolution is the candidate and is locked as provenance.

## 6. Candidate and wheel provenance

Future artifacts must bind, at minimum, the unchanged predecessor lock
blob/SHA-256, the new direct-spec blob/SHA-256, canonical interpreter and
platform, resolver version, official-index identity, and the complete
resolved package and wheel lists.  A candidate is valid only when package
names/versions, wheel filenames, hashes, and counts are internally closed.

The future resolver must classify failures only as bounded environment
states:

```text
PRE_GATE_ENVIRONMENT_BLOCK
RESOLUTION_FAILURE
PREDECESSOR_PIN_DRIFT
SOURCE_DISTRIBUTION_REQUIRED
WHEEL_PROVENANCE_FAILURE
```

No environment failure is a STOP/CONTINUE result and no failure authorizes a
second resolution or a T0 run.

## 7. Future canonical mutation boundary

After a separately reviewed candidate and wheelhouse exist, a later
canonical-mutation task must use its own Phase A/B/C sequence:

```text
Phase A = no-network exact candidate, wheel, lock, interpreter, and
          repository provenance preflight
Phase B = one bounded install of SUCCESSOR_DELTA_PACKAGE_SET only,
          from reviewed wheel paths with --no-deps --no-index
Phase C = no-network live inspection
```

The mutation task may not install from PyPI, reinstall predecessor packages,
alter the frozen lock, or retry after a complete mutation result.  It must
not reuse a resolution or mutation authorization across attempts.

## 8. Live readiness contract before T0 Phase A

Before any future T0 Phase A can pass, the reviewed successor environment
must prove, using only invented in-memory data and no market data:

- canonical interpreter is exactly Python `3.12.10`;
- exact successor package set is installed and matches the reviewed lock;
- `lightgbm` imports successfully;
- `Ridge` and `StandardScaler` import successfully;
- `LGBMRegressor` construction succeeds;
- bounded synthetic LightGBM fit/predict succeeds;
- bounded synthetic Ridge/StandardScaler fit/predict succeeds;
- no training/evaluation cache or raw payload is read;
- no T0 result is produced.

Only after exact-SHA review of the successor environment freeze may
`CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE`
become `YES`.  A readiness failure remains an environment block, not a
scientific result.

## 9. Scientific invariants

This environment successor must not change any scientific input or rule:

- V9 feature definitions and D1-to-D3 target semantics;
- T0 periods, 2020–2025 signal years, and TOP1 estimand;
- Ridge and LightGBM parameters, random states, and model-selection rules;
- signal grid, STOP/CONTINUE rule, stopping rule, and thresholds;
- promoted training membership `283/17` and evaluation identity;
- V10A calendar authority;
- costs, slippage, portfolio rules, or search space.

No T0 outcome has been observed.  No numerical-result equivalence may be
claimed before synthetic/runtime validation and a later authorized T0.
The environment version choice is prospective and is not profitability
evidence.

## 10. Authority sequence

The successor workflow is frozen as:

```text
DESIGN
-> GPT exact-SHA design review
-> PASS with C=0/H=0/M=0
-> explicit human DESIGN_FREEZE_ONLY approval
-> durable successor design-freeze approval record
-> GPT exact-SHA approval-record review
-> resolution implementation/reuse audit
-> targeted synthetic resolver tests
-> GPT exact-SHA implementation review
-> Phase A no-network resolution preflight
-> fresh point-of-use human authorization for one bounded public resolution
-> Phase B resolution exactly once
-> Phase C no-network inspection
-> GPT resolution adjudication
-> successor lock/wheel provenance promotion only after PASS
-> separate mutation design/implementation/review and authority sequence
-> live readiness validation
-> only then a future T0 Phase A may be considered
```

The design-freeze approval, public package-index resolution authorization,
canonical mutation authorization, and T0 authority are distinct.  None is
inherited from V10C data-adoption work, and no one-shot authority is reused
for another stage.

## 11. Failure and integrity rules

The future implementation must keep these classes disjoint:

```text
PRE_GATE_ENVIRONMENT_BLOCK
RESOLUTION_FAILURE
PREDECESSOR_PIN_DRIFT
SOURCE_DISTRIBUTION_REQUIRED
WHEEL_PROVENANCE_FAILURE
CANONICAL_MUTATION_FAILURE
LIVE_ENVIRONMENT_VALIDATION_FAILURE
T0_DATA_INCOMPATIBLE
T0_IMPLEMENTATION_FAILURE
SCIENTIFIC_T0_RESULT
```

Environment, resolution, wheel, and mutation failures never become
scientific STOP/CONTINUE results.  They never authorize package fallback,
re-resolution for a preferred result, cache refetch, ticker substitution,
model fitting, or T0 execution.

## 12. Current status and prohibitions

```text
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR=DESIGN_DRAFT_AWAITING_GPT_REVIEW
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_FROZEN=false
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_IMPLEMENTATION_AUTHORIZED=false
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_AUTHORIZED=false
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_AUTHORIZED=false
V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_T0_AUTHORIZED=false
```

During this design task there are zero package imports for readiness, zero
package resolution or installation, zero network requests, zero cache or
payload reads, zero model fits, zero T0 runs, and zero human-gate
consumption.  No package version beyond the two direct pins is preselected.

The historical canonical environment and its lock remain unchanged.  The
future direct-spec, resolver evidence, successor lock candidate, wheelhouse,
live validation evidence, and freeze record are separate artifacts and are
not created here.

## 13. Review criteria

The next action is `GPT_EXACT_SHA_V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_REVIEW`.
The review must verify the exact predecessor lock preservation, the two
prospective direct pins, unresolved transitive closure, wheel-only bounded
resolution, separate authority gates, and unchanged V9 scientific
methodology.
