# V10C T0 Canonical Environment Mutation Design Draft

```text
document_role=V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN
study_identity=V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR
status=DRAFT_AWAITING_GPT_REVIEW
scope=PROSPECTIVE_CANONICAL_ENVIRONMENT_MUTATION_ONLY
```

## 1. Purpose, boundary, and non-authority

This design specifies a future, one-shot mutation of the existing canonical
environment `.venv-real-execution`. It refines the already-frozen parent
sequence without changing its package set, versions, dependency closure, or
scientific methodology:

```text
Phase A = no-network provenance and readiness preflight
Phase B = exactly one bounded local-wheel installation of the frozen delta
Phase C = no-network live inspection and synthetic readiness validation
```

This draft neither executes nor authorizes any phase. In particular:

```text
network_authorized=false
package_installation_authorized=false
canonical_environment_mutation_authorized=false
t0_authorized=false
payload_read_authorized=false
private_sealed_access_authorized=false
resolution_authority_consumed=true
resolution_retry_authorized=false
future_profitability_established=false
```

No authority for resolution, installation, mutation, T0, historical
evaluation, payload access, or private/sealed access is inherited from the
design freeze, the source resolution, or this design review.

## 2. Immutable bindings

Every future phase must bind the following values exactly; a mismatch is a
closed failure before the corresponding boundary.

| Binding | Required value |
| --- | --- |
| Authoritative branch | `v9-cross-sectional-close-auction-design` |
| Reviewed promotion provenance commit | `89e0998f55b8bd4acf646fe8ca6f41120a191171` |
| Successor lock Git blob | `13636e58fbe40071be04cbfa57c3990c1d8ff2e0` |
| Successor lock SHA-256 | `f38dd4c7319465bb7e6ff429e8dff4a476d9966c744b19e50264dcc0b18e8300` |
| Promotion-record Git blob | `b866e6d77508d6366569ee6a229587c59c3c8be2` |
| Source resolution head | `3aee6c2772f30c6dc35d2a7efb862ae15091febc` |
| Source wheel manifest SHA-256 | `5d5953f14b0609767972679554e1999e754621056d863f8c33def96988797b74` |
| Offline candidate SHA-256 | `893881cbb9612e3402b0f4e1e434edfc4283da81a0f6a6d264d019ae5573c48e` |
| Offline readjudication evidence SHA-256 | `b4398e32be354de03e64202148dc6933ed45ef888657e909de1f52a4a051206b` |
| Source wheel/package count | `27` |
| Immutable predecessor package count | `20` |
| Successor package count | `27` |

The immutable predecessor lock remains `requirements-real-execution.lock.txt`
at blob `99395e7a5be752fb3ea92fd31be0334f38792261` and is never modified,
reinstalled, upgraded, downgraded, or uninstalled by this study.

The exact successor delta is the closed ordered set:

```text
cloudpickle==3.1.2
joblib==1.6.0
lightgbm==4.6.0
narwhals==2.26.0
scikit-learn==1.9.0
scipy==1.18.1
threadpoolctl==3.6.0
```

No package outside that seven-package set is installable under this design.

### 2.1 Current V10A canonical predecessor authority

The Phase-A baseline authority is the reviewed and promoted V10A canonical
chain, not the historical generic 15-package readiness closure. The exact
bindings are:

```text
V10A_CANONICAL_ENVIRONMENT_PROMOTED=true
V10A_ENVIRONMENT_FROZEN=true
V10A_ENVIRONMENT_STATE=CANONICAL_FROZEN
V10A_APPROVED_DESIGN_SHA=b14cc5510685210e928000af0815e188bc1aadc0
V10A_FREEZE_RECORD_SHA=86ceda3dee531b08afa5db4df7af1298ca770fad
V10A_FINAL_FREEZE_VERIFICATION_EVIDENCE_FILE=V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json
V10A_FINAL_FREEZE_VERIFICATION_EVIDENCE_GIT_BLOB_SHA1=d880b84fa00233e58653739fd510385fdf94de4e
V10A_FINAL_FREEZE_VERIFICATION_EVIDENCE_SHA256=658e264a70ab15ba402e7bf56d5e4b8abe5d81f2f7bb22f28bc797b7b8062b01
V10A_RUNTIME_ENVIRONMENT_LOCK_FILE=V10A_RUNTIME_ENVIRONMENT_LOCK.json
V10A_RUNTIME_ENVIRONMENT_LOCK_GIT_BLOB_SHA1=9dfe03cf807b3580d432146839e8eb013bfa3c63
V10A_RUNTIME_ENVIRONMENT_LOCK_SHA256=d7f54bc69029ba9b25a9920e867fe6487745af6ef985898bad91bd951003fc3a
V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE_FILE=V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json
V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE_GIT_BLOB_SHA1=e07040f75a92ef0669215f4a7e2e98b71ea29d36
V10A_PREDECESSOR_LOCK_FILE=requirements-real-execution.lock.txt
V10A_PREDECESSOR_LOCK_GIT_BLOB_SHA1=99395e7a5be752fb3ea92fd31be0334f38792261
V10A_PREDECESSOR_LOCK_SHA256=eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444
V10A_PREDECESSOR_PACKAGE_COUNT=20
```

The exact 20-entry predecessor mapping bound by that lock is:

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

The historical generic 15-package checker/lock and its historical freeze
artifacts are not current V10C predecessor authority and must not be treated
as a required Phase-A PASS source.

### 2.2 Future execution-time implementation binding

The reviewed promotion commit is immutable provenance only. It is not an
execution-time repository-head target and it must never be used to reset the
future execution checkout. This design deliberately does not fabricate a
future commit SHA.

```text
MUTATION_IMPLEMENTATION_REVIEWED_SHA=UNESTABLISHED
MUTATION_IMPLEMENTATION_REVIEW_RESULT=UNESTABLISHED
```

These bindings are established only when a future mutation implementation has
received a GPT exact-SHA `PASS`. Execution is prohibited until both are
established. Thereafter, Phase A and the Phase B immediate pre-launch recheck
require local `HEAD` and local
`refs/remotes/origin/v9-cross-sectional-close-auction-design` to equal that
exact reviewed mutation-implementation SHA. The promotion commit remains a
separate provenance binding in every such recheck.

## 3. Future durable attempt namespace

The future implementation shall receive one logical durable attempt root
named `V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_ATTEMPT_1` under
the preconfigured machine-local protected audit root. The root is not created
by this design task and its machine-local path is not recorded in Git.

Before Phase A, the implementation must prove that this exact root and every
planned child output name are absent. The reserved child names are:

```text
mutation_state.json
mutation_stdout.txt
mutation_stderr.txt
mutation_evidence.json
```

The root must be distinct from the source-resolution attempt, the offline
readjudication namespace, candidate/evidence locations, wheel storage, cache,
and every prior mutation attempt. Any existing root, child, symlink, junction,
reparse point, non-regular file, failed stat/read, or ambiguous resolution is
`PRE_GATE_ENVIRONMENT_BLOCK`. No cleanup, delete, overwrite, or alternate
location selection is permitted.

## 4. Phase A — no-network preflight

Phase A is read-only with respect to the canonical environment and must run
before point-of-use mutation authority is requested or consumed.

It must prove all of the following:

1. The repository is on the authoritative branch; the future
   `MUTATION_IMPLEMENTATION_REVIEWED_SHA` and its GPT exact-SHA `PASS` result
   are established; `HEAD` and the local
   `refs/remotes/origin/v9-cross-sectional-close-auction-design` both equal
   that exact reviewed implementation SHA; and the working tree is clean. The
   reviewed promotion commit remains provenance only. It performs no fetch,
   `ls-remote`, or other network operation.
2. The committed successor lock has the exact blob and SHA-256 in section 2;
   the promotion record has its exact blob; and the candidate, evidence, and
   source-wheel-manifest SHA-256 bindings exactly match section 2.
3. The canonical interpreter resolves uniquely to
   `.venv-real-execution\Scripts\python.exe`, is not the general `.venv`, and
   reports exactly Python `3.12.10`.
4. A canonical-interpreter `importlib.metadata` observer, not pip-freeze
   presentation, reports exactly the immutable predecessor set of 20
   normalized `name==version` entries, with no missing, extra, duplicate, or
   version-drifted distribution.
5. The protected source wheel set is exactly 27 regular, non-reparse wheel
   files matching the frozen manifest. Exactly one regular reviewed wheel for
   each normalized successor-delta distribution/version exists; each is part
   of the reviewed 27-wheel set. This check must not select substitutes by
   filename similarity, another version, another wheelhouse, or a network
   index.
6. The durable attempt namespace in section 3 is fresh, non-overlapping, and
   unambiguous.
7. Before requesting or accepting human mutation authorization, the
   operation-specific pre-gate closure described in section 4.1 is `YES`.

Phase A has no network, installation, environment mutation, model fit, T0,
training/evaluation payload, raw-market payload, candidate/evidence payload,
or private/sealed-data activity. It produces only privacy-safe booleans,
counts, fixed hashes, and closed failure codes. Human mutation authority is
not consumed in Phase A.

Any failure in these predicates is `PRE_GATE_ENVIRONMENT_BLOCK`; it permits
only a separately authorized non-methodological preflight repair and a new
complete preflight. It never permits a second resolution or alternate package
selection.

### 4.1 Mandatory mutation-operation readiness closure

Before requesting or accepting fresh mutation authority, Phase A must
mechanically establish readiness for every software and durable mechanism
reachable after the mutation gate. The historical
`scripts/check_real_execution_env.py` generic readiness checker is bound to
the stale 15-package closure and is not the current V10C predecessor
authority; Phase A must not require that checker itself to PASS as the
20-package baseline. Phase A instead binds the reviewed/promoted V10A chain
and the exact 20-package predecessor lock in section 2.1, while preserving
the relevant interpreter, environment-lock/fingerprint, and filesystem
readiness semantics in the operation-specific checks below.

The required closure is all of the following:

1. The V10A promotion/freeze chain in section 2.1 is bound exactly,
   including the approved V10A design, freeze record, final-freeze
   verification evidence, runtime lock, runtime-lock execution evidence, and
   their recorded Git/SHA-256 provenance. The historical generic 15-package
   freeze/checker artifacts are not substituted for any of these bindings.
2. The canonical interpreter resolves uniquely to
   `.venv-real-execution\Scripts\python.exe`, is not the general `.venv`,
   reports exactly Python `3.12.10`, and uses `importlib.metadata` to observe
   exactly the 20 normalized predecessor `name==version` entries from
   section 2.1. The observer must reject any missing, extra,
   duplicate-normalized, or version-drifted distribution.
3. The future reviewed mutation implementation's canonical subprocess builder
   is statically bound to the resolved canonical interpreter and exactly
   `-m pip install --no-deps --no-index`. A no-network canonical
   `python -m pip --version` probe confirms that the `pip` invocation
   machinery itself is reachable before the gate; it must not install,
   download, resolve, or contact an index.
4. The reviewed 27-wheel source set is bound exactly to its frozen manifest,
   and exactly one reviewed wheel is selected for each of the seven delta
   entries. No alternate wheel, version, wheel root, or resolver selection is
   permitted.
5. The planned attempt root and all reserved output names are checked
   read-only for collision, reparse/symlink/junction status, non-regular
   entries, failed stat/read, parent-volume ambiguity, and overlap with prior
   namespaces. The exact future reviewed implementation SHA, its protected
   source blobs, the promotion provenance, the seven-wheel selection rule,
   canonical interpreter, and these read-only root checks are bound in the
   Phase-A safe evidence.
6. The implementation-reviewed durable stdout/stderr/state/evidence
   publication semantics are bound to the future implementation's exact
   GPT-reviewed SHA and targeted synthetic tests. Phase A itself creates or
   preserves no durable probe namespace and performs no live filesystem
   write/delete probe; it only observes the existing machine state
   read-only. Any uncertain permission, volume, reparse, exclusivity, or
   publication-contract result fails closed.

Only when every item is mechanically proven may Phase A emit the required
operation-specific predicate:

```text
CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE_FOR_MUTATION=YES
```

`NO` or `UNKNOWN` is `PRE_GATE_ENVIRONMENT_BLOCK`, consumes no mutation
authority, and prohibits requesting or accepting the human authorization.
This predicate does not alter the global
`CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE`,
which remains `NO` until the separate successor T0 readiness sequence is
successfully completed and reviewed.

## 5. Phase B — one bounded canonical mutation

Phase B is reachable only after every Phase A predicate passes and a fresh,
point-of-use human authorization specifically naming V10C canonical-
environment mutation is present. That authorization is distinct from and not
inherited from the design freeze, resolution, promotion, T0, or any prior
authority.

Immediately before process launch, the implementation reruns the
non-destructive repository, future reviewed-implementation SHA, promotion
provenance, mutation-readiness-`YES`, interpreter, predecessor-set, wheel-set,
and durable-root bindings. It then atomically records a privacy-safe mutation
receipt/state in the fresh attempt root and launches one process using only
the canonical interpreter:

```text
.venv-real-execution\Scripts\python.exe -m pip install --no-deps --no-index <the seven individually validated local wheel files>
```

The seven arguments are exactly one validated wheel for each section-2 delta
entry and no others. `--no-deps` and `--no-index` are mandatory. No index,
network endpoint, cache fallback, requirements file, package resolver,
alternate environment, or general `.venv` may be used.

The durable state records the process-launch boundary, and stdout/stderr are
captured only in the reserved attempt-root files. Once that state is published
or the install process is crossed, the one-shot mutation authority is
consumed. There is exactly one bounded install process. The process may not
reinstall, upgrade, downgrade, or uninstall any predecessor package.

No automatic retry is allowed. A launch error, nonzero exit, partial install,
or interrupted process is preserved. It never restores authority or permits
rollback, reset, deletion, environment recreation, predecessor reinstall,
alternate version, second resolution, or T0. Network is prohibited under all
conditions.

## 6. Phase C — no-network validation

Phase C runs only after the Phase B process has completed and uses the
canonical interpreter. It never calls pip, an index, a wheel root, a market
payload, training/evaluation payload, or T0 code.

The implementation must use `importlib.metadata`, normalized distribution
names, and exact versions to require all of the following simultaneously:

1. The live set is exactly the 27-entry successor lock: no missing, extra,
   duplicate-normalized, or version-drifted distribution.
2. The 20 immutable predecessor entries are unchanged.
3. The exact seven-delta set is present at the frozen versions.
4. The interpreter is exactly Python `3.12.10` and resolves to the canonical
   `.venv-real-execution\Scripts\python.exe` path.

It must then run only these deterministic, invented, in-memory operational
probes:

```text
import lightgbm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
y = [0.0, 1.0, 2.0, 3.0]
```

The LightGBM probe constructs `LGBMRegressor(n_estimators=1, random_state=0,
n_jobs=1, verbosity=-1)`, fits `X, y`, predicts one invented row, and requires
one finite prediction. The Ridge probe fits `StandardScaler` on `X`, fits
`Ridge()` on the transformed `X, y`, predicts one transformed invented row,
and requires one finite prediction. These probes do not read any file or
network source and are not research-model fitting or T0.

Phase C publishes only deterministic privacy-safe evidence: fixed bindings,
pass/fail status, closed failure class, package counts, normalized package
identities/versions, interpreter version, bounded-probe booleans, and hashes
of the reserved durable outputs. It must not emit machine-local paths, raw
payloads, credentials, ticker identities, prices, or private information.

## 7. Closed failure discipline

The implementation must use these distinct non-scientific classes:

| Condition | Class | Consequence |
| --- | --- | --- |
| Any failed Phase A predicate before the mutation boundary | `PRE_GATE_ENVIRONMENT_BLOCK` | No authority consumed; no installation or retry. |
| Launch, installation, or interruption after the Phase B boundary | `CANONICAL_MUTATION_FAILURE` | Authority consumed; preserve state; no retry/rollback/reset. |
| Phase C metadata or synthetic-readiness failure | `LIVE_ENVIRONMENT_VALIDATION_FAILURE` | Authority remains consumed; preserve state; no retry/rollback/reset. |

None of these is a scientific T0 STOP/CONTINUE result. None authorizes
alternate versions, re-resolution, predecessor reinstallation, environment
recreation, T0, or a new mutation attempt.

## 8. Post-success boundary

A Phase C `PASS` is readiness evidence only. It must first receive GPT exact-
SHA result adjudication/review. Even after GPT review, mutation readiness does
not itself authorize T0: T0 authority remains separately false until a future
explicit authority names that boundary. This design does not change
`CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE`
from `NO`, and makes no profitability claim.
