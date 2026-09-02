# V9_014 PDF Real-Execution Environment Successor Design

STUDY_ID: V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR
STATUS: DRAFT_AWAITING_GPT_REVIEW
FROZEN_DESIGN_GIT_SHA: efee3d0efca368645c00aeed63cb8e0637cd3672
FROZEN_DESIGN_BLOB_SHA: 2bbacbf37ab961d1cbf416b7fd476db18778c5b7
PREDECESSOR_GOVERNANCE_HEAD: af0ac863f6e25abd4335ca2cb4c129ed7c3c3814
PRIOR_GPT_REVIEW: 48b0b63a0f78cd75f48429a9df0c77607a63fd45_BLOCK_CRITICAL_0_HIGH_1_MEDIUM_1

This revision remediates that `BLOCK`'s `HIGH_1`
(`PREPROMOTION_CANONICAL_ENVIRONMENT_LIFECYCLE_INCOHERENT`) and `MEDIUM_1`
(`PDFPLUMBER_OPERATIONAL_READINESS_NOT_PROVEN`).

This is a **DOC-ONLY execution-environment successor design**. It performs
NO package resolution, NO install, NO environment mutation, NO network
access, NO PDF access, NO code implementation, and NO scientific-
methodology change. It does not amend, override, reinterpret, or claim
invalidity of the frozen V9_014 design draft, nor of any predecessor
real-execution-environment artifact.

---

## 1. Predecessor immutability

The currently reviewed/frozen real-execution environment, its exact
lock/candidate/evidence/freeze record, and canonical interpreter contract
are treated as **immutable predecessor evidence**. This task does not edit,
regenerate, supersede, reinterpret, or claim invalidity of any of them.

Exact Git blob SHAs, as read at `EXPECTED_HEAD=af0ac863f6e25abd4335ca2cb4c129ed7c3c3814`:

| Predecessor artifact | Git blob SHA |
|---|---|
| `requirements-real-execution.txt` | `8ab77d5a7d08be9482712ab0f80248c39fefefd8` |
| `requirements-real-execution.lock.txt` | `e039311236f8138d8878112399af67fd4cbf249e` |
| `REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json` | `1678136e43e3a42bcbac41d135e2d8a139b2f29b` |
| `REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json` | `8c672ade846e4c3ec67cd462b3417b8068e665c4` |
| `REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json` | `4f70425965d9a314218f3d136abd6fe7362931c4` |
| `scripts/check_real_execution_env.py` | `513d99db2cd02f17ebeb4791abfc94793fcdf8fb` |
| `scripts/bootstrap_real_execution_env.ps1` | `a2171e90218fec0665daea0515dca4fa893301fd` |
| `REAL_EXECUTION_PYTHON_ENVIRONMENT.md` | `4432db4d2ed64400d1cff14e2dd36760f86ffe4e` |
| `V8J_SOURCE_SNAPSHOT_ENVIRONMENT_SUCCESSOR_DESIGN_DRAFT.md` | `b3e03ca0bef40380fef66083d91d74ebd79e617e` |

Predecessor freeze provenance, reproduced exactly (unedited) from
`REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json` at `EXPECTED_HEAD`:

```
artifact_status = REAL_EXECUTION_ENVIRONMENT_FROZEN
real_execution_environment_frozen = true
real_execution_environment_ready = true
canonical_environment_directory = .venv-real-execution
canonical_interpreter = .venv-real-execution\Scripts\python.exe
package_set = numpy==2.5.2, pandas==3.0.5, pip==25.0.1,
              python-dateutil==2.9.0.post0, six==1.17.0, tzdata==2026.3,
              xlrd==2.0.2
python = CPython 3.12.10, win-amd64, Windows/AMD64
resolved_lock.path = requirements-real-execution.lock.txt
resolved_lock.sha256 = b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96
resolved_lock.package_count = 7
reviewed_lock_candidate_git_sha = 107430894723c2bdc2f8493cb12c467fccd8665e
reviewed_windows_validation_evidence_git_sha = f52f31ab6305e321cd9e8e9855d6efd83238f552
source_git_sha = b74e0f787599475cd9fe719d254202dc9bfc14d5
source_requirements.path = requirements-real-execution.txt
source_requirements.canonical_git_sha256 = 2cdcfd7a87023c4e9c3ec463cf16f77d88f72ccc8d1f0e5de242e6c68b0cf601
tested_implementation_git_sha = 84d4512d800b18b858b6f129be9a4ba0ea73d4ca
future_protected_execution_authorized = false
```

Predecessor direct-dependency spec, reproduced exactly (unedited) from
`requirements-real-execution.txt` at `EXPECTED_HEAD`: `pandas` (unpinned,
by deliberate predecessor design -- no authoritative pandas version is
established anywhere in this repository, and this task does not invent
one) and `xlrd==2.0.2`.

This predecessor closure exists to serve the V8-lineage source-snapshot
acquisition path (`src/v8i_source_snapshot.py`,
`src/v8_partition.py`'s `parse_eligible_universe`,
`scripts/build_v8_partition_manifest.py`). It is unrelated to, and remains
completely unaffected by, V9_014 PDF work.

## 2. Canonical environment and successor staging environment

### 2a. Canonical environment lifecycle state

`.venv-real-execution` is the exact predecessor canonical protected
environment (Section 1):

```
.venv-real-execution
.venv-real-execution\Scripts\python.exe
```

Its lifecycle through this successor path is governed by exactly three
states, in this order, each defined precisely so that no stage's own
description can contradict it. Every other section of this document that
describes `.venv-real-execution`'s mutation status must be read subject to
this state definition, and is written to match it exactly.

**State 1 -- `CANONICAL_ENVIRONMENT_STATE=PREDECESSOR_CANONICAL_FROZEN`**
(holds through Stage E8's own `PASS`, i.e. for all of E1-E8):

- `.venv-real-execution` **MUST NOT be mutated** by any activity in
  Stages E1-E8 -- design, offline implementation, staging-venv creation,
  staging resolution, staging inspection, candidate/evidence review, or
  promotion-completeness review. None of these stages creates, writes to,
  deletes, or otherwise touches `.venv-real-execution` in any way.
- the predecessor frozen package/environment authority (Section 1) remains
  valid and continues to serve already-authorized predecessor (V8-lineage)
  work exactly as it did before this design existed; successor staging
  activity (Section 2b) does not change that authorization in any way;
- V9_014 PDF execution requiring `pdfplumber` remains **BLOCKED**;
- no environment may satisfy `CANONICAL_ENVIRONMENT_STATUS` other than
  `.venv-real-execution` itself, and `.venv-real-execution` holds that
  status only under its existing predecessor (V8-lineage) authorization,
  not for any V9_014 PDF purpose.

**State 2 -- `CANONICAL_ENVIRONMENT_STATE=SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED`**
(begins at the instant Stage E9 starts its canonical mutation, holds
through Stage E10):

- from this instant, `.venv-real-execution` is **no longer** described as
  unchanged, frozen, or "live canonical" in the predecessor sense -- it is
  actively being migrated, and neither this document nor any stage output
  may claim otherwise while this state holds;
- the successor environment is **not yet** canonical or frozen either --
  no stage may claim `CANONICAL_ENVIRONMENT_STATUS` for the successor
  package set until State 3 is reached;
- **NO** protected/private/research execution of any kind is permitted
  under either the predecessor's or the successor's environment authority
  while this state holds;
- only the reviewed migration operation itself (Stage E9) and its
  no-network/live-validation operation (Stage E10) are permitted, each
  under its own separate point-of-use authority (Section 5, Stages E9/E10);
- **no automatic rollback, reset, delete, recreate, or retry** occurs on
  failure at any point while this state holds (Section 6); a failure is
  preserved as durable evidence and returns control to GPT/human authority
  rather than being silently resolved.

**State 3 -- `CANONICAL_ENVIRONMENT_STATE=SUCCESSOR_CANONICAL_FROZEN`**
(begins only once Stage E11's GPT review of the Stage E10 evidence is
`PASS`):

- only now is the mutated `.venv-real-execution` accepted as successor
  canonical/frozen for the successor package set;
- only now may V9_014 PDF calibration-runner Stage C (per
  `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`
  Section 7) begin, and only subject to that stage's own required
  authority -- this design does not itself grant it.

The correct mutation gate, stated once and unambiguously: **no mutation of
any kind through Stage E8's `PASS`; Stage E9 alone is the sole stage
authorized to perform the canonical mutation, under its own fresh
point-of-use authority; Stage E11 alone is the sole stage that may declare
the mutated environment accepted as successor canonical/frozen.** E9 and
E11 are deliberately not the same stage: E9 performs the mutation, E11
reviews its evidence and only then confers canonical status. Between them
(State 2), the environment is in a migration-in-progress state that is
neither the old canonical nor the new canonical, and no protected
execution of any kind is permitted against it.

### 2b. Successor staging environment

Exactly **one** explicitly **NON-CANONICAL** successor staging venv is
permitted, for Windows dependency resolution and operational validation
only (Section 5, Stages E5-E7). It is governed by the following
constraints, all of which bind every future stage that touches it:

- It is **never** accepted for JPX/private/protected/research execution of
  any kind -- it is a dependency-resolution and operational-readiness
  workspace only, nothing more.
- It **cannot** satisfy `CANONICAL_ENVIRONMENT_STATUS` under any
  circumstance, at any stage, regardless of its own validation outcome.
- It **must be distinct** from `.venv-real-execution` -- a different
  directory, never a rename, symlink, or alias of the canonical path.
- It is **durable and preserved on failure** under the future Phase A/B/C
  contract (Section 5): a failed one-shot resolution (Stage E5) or a
  failed inspection (Stage E6) leaves the staging venv and its failure
  evidence exactly as they are.
- It is **never silently reset or recreated** after a failed one-shot
  resolution. Any deliberate recreation requires its own separate review
  and is never performed merely to retry toward a passing result (Section
  6).

The exact machine-local staging-path mechanics (directory name, parent
location, naming convention) are **implementation-stage governance**, not
frozen here. Before any mutation that would create or write to the staging
venv, the implementation stage's own Phase A preflight (Section 5, Stage
E3) must mechanically collision-check the chosen staging path against
`.venv-real-execution` and against every other path already tracked by
repository governance, and must fail closed rather than proceed if a
collision is detected.

## 3. Successor direct dependency spec

The successor direct closure is exactly the existing reviewed predecessor
direct dependencies (`pandas` unpinned, `xlrd==2.0.2`) **plus**:

```
pdfplumber==0.11.10
```

No other new direct dependency may be invented by this design or by any
downstream implementation stage without its own separate review.

This design task does **not** guess or freeze transitive versions.
Transitive dependency closure (the successor's own resolved lock, analogous
to the predecessor's `requirements-real-execution.lock.txt`) comes only
from the later reviewed Windows-grounded one-shot resolution procedure
defined in Section 5, Stage E5, performed in the staging environment
(Section 2b) -- never in `.venv-real-execution`.

No parser substitution is permitted: `pdfplumber==0.11.10` is the sole PDF
engine, per `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`
Section 1, unchanged and reaffirmed here, not re-decided.

### 3a. Delta-preserving resolution

The predecessor's reviewed 7-package versions (Section 1:
`numpy==2.5.2`, `pandas==3.0.5`, `pip==25.0.1`,
`python-dateutil==2.9.0.post0`, `six==1.17.0`, `tzdata==2026.3`,
`xlrd==2.0.2`) are **constraints** and **MUST remain exactly unchanged**
in the successor candidate produced by Stage E5/E6. They are pins to
preserve, not merely a starting point.

The only intended dependency addition is `pdfplumber==0.11.10` **plus**
the transitive closure mechanically required by it (Stage E5's actual
Windows-grounded resolution output -- never guessed or pre-specified by
this design).

The historical fact that the direct `pandas` *specification* is unpinned
in `requirements-real-execution.txt` is not license to let the *resolved*
`pandas==3.0.5` pin drift: an unpinned direct spec does not permit the
resolver to select a different pandas version than the one already
resolved and frozen for the predecessor. No existing predecessor pin may
be upgraded or downgraded by the successor resolution merely because it
was convenient, newer, or requested by `pdfplumber`'s own transitive
requirements, unless doing so is mechanically unavoidable.

If resolving `pdfplumber==0.11.10` against the existing predecessor pins
surfaces a genuine incompatibility that would require changing any
predecessor pin, that is `CHATGPT_DECISION_REQUIRED` and the successor
implementation stage must **STOP** rather than silently resolve the
conflict by upgrading, downgrading, or removing a predecessor pin.

No transitive pin is guessed by this design or by Stage E2's offline
implementation; every transitive version comes only from Stage E5's actual
one-shot resolution against the staging environment.

## 4. Successor identity / provenance

A distinct V9_014 environment-successor identity is defined so the
predecessor's frozen environment history (Section 1) remains fully
auditable and is never silently overwritten. The successor path is
composed of four distinct artifacts, each with its own name and its own
lifecycle status, mirroring -- but never replacing -- the predecessor's
four-artifact shape (`requirements-real-execution.txt` /
`REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json` /
`REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json` /
`REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json`):

| Role | Successor artifact name (future; not created by this task) | Predecessor analogue | Produced by |
|---|---|---|---|
| Successor direct-spec candidate | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt` | `requirements-real-execution.txt` | Stage E2 (offline, no environment involved) |
| Successor resolved lock candidate | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json` | `REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json` | Stage E5/E6 (staging-environment one-shot resolution result), committed at Stage E7 |
| Successor Windows validation evidence | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_WINDOWS_VALIDATION_EVIDENCE.json` | `REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json` | Stage E6 (staging-environment inspection, including the operational PDF probe result), committed at Stage E7 |
| Successor freeze/promotion record | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json` | `REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json` | Stage E11 (only after live canonical-promotion review is `PASS`) |

None of these four artifacts is created, populated, or reserved by this
design task; it only fixes their names, roles, and exact producing stage.

**Correction to the prior revision:** Stage E2 produces **only** the
successor direct-spec candidate (the first artifact above) plus the
offline tooling and tests needed for its later resolution and validation
(Section 5, Stage E2). Stage E2 **does not**, and **must not**, produce or
claim a Windows-grounded resolved-lock candidate (the second artifact) --
that artifact exists only as the actual output of the one-shot staging-
environment resolution in Stages E5/E6, and is never fabricated,
pre-guessed, or synthesized offline.

If the repository's existing generic canonical environment files
(`requirements-real-execution.txt`, `requirements-real-execution.lock.txt`,
`REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json`,
`REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json`,
`REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json`) must eventually change to
incorporate the successor, that change occurs **only** as part of the live
canonical-promotion sequence (Section 5, Stages E9-E11), and only after the
successor candidate/evidence artifacts have themselves passed exact-SHA
review (Section 5, Stage E7 `PASS`) and the promotion-completeness review
(Section 5, Stage E8 `PASS`). This design task does not perform, schedule,
or pre-approve that promotion; it only defines the conditions under which
one may later occur. See Section 6a for the full promotion-completeness
scope.

## 5. Resolution / execution phases

The following order is frozen. No stage may be skipped or reordered; each
stage marked "GPT ... PASS" requires an independent exact-SHA GPT-5.6 Sol
review before the next stage may begin. The sequence follows the
three-state canonical lifecycle of Section 2a exactly: Stages E1-E8 all
occur under `CANONICAL_ENVIRONMENT_STATE=PREDECESSOR_CANONICAL_FROZEN`,
validating the successor **only** in the non-canonical staging environment
(Section 2b) -- `.venv-real-execution` itself is untouched throughout
E1-E8, including E8's promotion-completeness review. Stage E9 is the sole
stage that transitions the state to
`SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED` and performs the
canonical mutation, gated on both E7's `PASS` and E8's `PASS`. Stage E10
runs under that same in-progress state. Stage E11 alone transitions the
state to `SUCCESSOR_CANONICAL_FROZEN`.

```
E1  This successor-design exact-SHA GPT PASS.

E2  Offline implementation, producing ONLY:
    - the successor direct-spec candidate (Section 4, artifact 1)
    - the offline bootstrap/checker/candidate tooling and its targeted
      synthetic/static tests needed for later staging-environment
      resolution and validation
    - the wholly synthetic PDF operational-readiness probe fixture and
      its own implementation review (Section 5a) -- fixed, committed
      bytes/hash, non-JPX, non-sensitive
    Plus GPT exact-SHA PASS on all of the above.
    No network, no install, no environment mutation of any kind (neither
    staging nor canonical). Stage E2 MUST NOT produce or claim a
    Windows-grounded resolved-lock candidate (Section 4, artifact 2) --
    that is exclusively Stage E5/E6's output.

E3  Phase A NO-NETWORK Windows preflight:
    - repo/branch/exact HEAD/clean tree verification
    - predecessor hash verification (Section 1 table, re-derived and
      compared, never trusted from memory)
    - canonical Python/platform prerequisite verification
    - mechanical collision-check of the proposed staging-venv path
      against `.venv-real-execution` and every other repository-tracked
      path (Section 2b)
    - NO environment mutation of any kind (neither staging nor canonical)

E4  GPT reviews Phase A.
    A fresh point-of-use human/network/durable-state authority is obtained
    if repository governance (AI_RESEARCH_EXECUTION_RULES.md,
    AI_REAL_EXECUTION_RUNBOOK.md) requires it at that time; no authority
    from this document or from any prior V9 stage is reused. This
    authority covers staging-environment creation only -- it never covers
    `.venv-real-execution` mutation, which is gated on Stage E9's own
    separate fresh point-of-use authority (never this one) and, before
    that, on both E7's `PASS` and E8's `PASS`.

E5  Phase B MINIMAL REAL ENVIRONMENT RESOLUTION (staging only):
    - create or reuse the collision-checked, non-canonical staging venv
      (Section 2b); `.venv-real-execution` is not touched
    - resolve exactly once from the reviewed successor direct spec
      (Section 3), subject to the delta-preserving constraint (Section
      3a: the predecessor's 7 pins remain exactly unchanged; only
      `pdfplumber==0.11.10` and its mechanically required transitive
      closure may be added)
    - NO PDF/source acquisition of any kind
    - stdout/stderr preserved separately, unmodified
    - a nonzero exit means: no retry; proceed directly to Phase C (E6)
      with the failure evidence intact, and the staging venv itself left
      exactly as it is (Section 2b) -- never reset or recreated to try
      again
    - a genuine predecessor-pin incompatibility (Section 3a) is
      `CHATGPT_DECISION_REQUIRED`, not a resolver decision

E6  Phase C NO-NETWORK inspection (staging only):
    - exact package freeze, checked against Section 3a's delta-preserving
      constraint (all 7 predecessor pins byte-for-byte unchanged; only
      `pdfplumber` and its actual resolved transitive closure added)
    - Python/platform fingerprint
    - direct import proof, including `pdfplumber`
    - the synthetic PDF operational-readiness probe (built at Stage E2)
      executed against the staging interpreter, per Section 5a, with its
      exact expected result checked
    - candidate/evidence hashes
    - process exit status
    - predecessor/successor provenance cross-reference
    - NO retry and NO package repair of any kind

E7  Commit the successor candidate/evidence artifacts (Section 4,
    artifacts 1-3: direct-spec candidate, resolved lock candidate,
    Windows validation evidence including the PDF probe result) only, and
    obtain GPT exact-SHA review of that commit. `.venv-real-execution`
    remains untouched through this stage.

E8  Promotion-completeness review: for the E7-reviewed successor
    candidate/evidence, determine and draft (still without mutating
    `.venv-real-execution`) exactly which of the mechanically coupled
    canonical-authority artifacts in Section 6a require an update, and
    prepare the corresponding promotion diff/tooling for review. GPT
    exact-SHA PASS on the promotion artifacts/tooling.

E9  Separately authorized live canonical-promotion operation: only after
    both E7 `PASS` and E8 `PASS`, a Phase A/B/C operation with its own
    fresh point-of-use authority (never reused from E4 or from this
    document) mutates `.venv-real-execution` from the reviewed successor
    lock (Section 4, artifact 2). This is the only stage in this entire
    design permitted to touch the canonical environment. At the instant
    this mutation begins, `CANONICAL_ENVIRONMENT_STATE` transitions from
    `PREDECESSOR_CANONICAL_FROZEN` to
    `SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED` (Section 2a, State
    2): the predecessor environment is no longer claimed unchanged/
    frozen/canonical from this point, the successor is not yet claimed
    canonical/frozen, and NO protected/private/research execution is
    permitted under either environment's authority until State 3.

E10 Post-mutation live validation, performed entirely under
    `CANONICAL_ENVIRONMENT_STATE=SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED`
    (Section 2a, State 2) -- the environment is still not canonical at
    this point:
    - run the promoted canonical checker (the Section 6a-updated
      `scripts/check_real_execution_env.py`) against
      `.venv-real-execution`
    - require the exact successor package set (predecessor's 7 pins
      unchanged, plus `pdfplumber==0.11.10` and its actual resolved
      transitive closure)
    - require the existing synthetic XLS probe to still pass (predecessor
      regression)
    - require the new synthetic PDF operational-readiness probe (Section
      5a) to pass against the now-live canonical interpreter
    - preserve execution evidence in full; on failure there is NO
      automatic rollback, reset, delete, recreate, or retry (Section 6)
      -- the failure is preserved as durable evidence and control returns
      to GPT/human authority, not silently resolved by this stage

E11 GPT exact-SHA review of the Stage E10 evidence. Only a `PASS` here
    transitions `CANONICAL_ENVIRONMENT_STATE` to
    `SUCCESSOR_CANONICAL_FROZEN` (Section 2a, State 3) and promotes
    `.venv-real-execution` to `CANONICAL_ENVIRONMENT_STATUS` for the
    successor package set. Only after this `PASS` may V9_014 PDF
    calibration-runner Stage C (per
    `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`
    Section 7) begin, subject to that stage's own required authority.
```

### 5a. Operational pdfplumber probe

Successor readiness MUST require more than a bare `import pdfplumber`
succeeding. This design freezes the requirement for a wholly synthetic,
non-JPX, non-sensitive PDF operational fixture/probe, implemented and
reviewed at Stage E2 and executed at Stages E6 (staging) and E10 (live
canonical):

- the fixture is **fixed, committed bytes**, with its own committed hash,
  established through Stage E2's own implementation review -- it is never
  generated on the fly and never sourced from any real JPX material;
- the probe proves `pdfplumber==0.11.10` actually **opens** the fixture;
- the probe proves the **expected synthetic page/text/table structure** is
  extracted from it;
- the probe checks an **exact, safe, predetermined expected result**
  (structure and content known in advance from the fixture's own
  construction) -- not merely "no exception raised";
- the probe involves **zero network access and zero protected data** of
  any kind.

This probe is **environment readiness only**. It does **not** observe any
of the 8 calibration PDFs named in
`V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md` Section 2,
and it does **not** resolve, narrow, or amend frozen design LOW_1 in any
way -- it says nothing about real JPX PDF structure, only that the
pinned `pdfplumber` build behaves as expected against a fully synthetic,
already-known input.

## 6. Fail-closed

The following are never permitted at any stage of this successor path,
in either the staging environment or the live canonical environment:

- "pip install missing package" as an ad hoc remediation
- dependency/version substitution of any kind
- a second resolution attempt after a Phase B (E5) failure
- reuse of prior human authorization (from this document, from any prior
  V9 stage, from Stage E4's staging authority for Stage E9's canonical
  mutation, or from the predecessor environment's own authorization)
- environment deletion/reset performed to erase failure evidence, whether
  of the staging venv (Section 2b) or of `.venv-real-execution` after a
  Stage E9/E10 failure
- upgrading, downgrading, or removing an existing predecessor pin to
  resolve an incompatibility (Section 3a) rather than stopping under
  `CHATGPT_DECISION_REQUIRED`

No package shopping is permitted until a stage's own GPT `PASS` is
recorded.

Any governance decision needed to fully specify E2-E11 that is not already
answered by this document, by
`V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`, or by
existing repository governance (`AI_RESEARCH_EXECUTION_RULES.md`,
`AI_RESEARCH_CHECKPOINT_WORKFLOW.md`, `AI_REAL_EXECUTION_RUNBOOK.md`,
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md`) is `CHATGPT_DECISION_REQUIRED` and
may not be decided unilaterally by an executor at implementation time.

### 6a. Promotion completeness

Stage E8's promotion-completeness review is not limited to the four
canonical data artifacts named in Section 4. At minimum, that review must
explicitly assess whether an update is needed to **each** of the following
mechanically coupled canonical-authority items, and record its
determination (update needed / not needed, with reasoning) for every one
of them -- no executor may decide to omit review of any item merely to
reduce diff size:

- `requirements-real-execution.txt`
- `requirements-real-execution.lock.txt`
- `REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json`
- `REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json`
- `REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json`
- `scripts/check_real_execution_env.py`
- `scripts/bootstrap_real_execution_env.ps1`
- `REAL_EXECUTION_PYTHON_ENVIRONMENT.md`
- `AI_REAL_EXECUTION_RUNBOOK.md`
- relevant targeted tests/fixture provenance (any test or fixture whose
  content, hash, or expected result depends on the predecessor package
  set, the canonical checker's behavior, or the synthetic XLS/PDF probes)

Any item this review determines needs no update must still be listed with
its "no update needed" determination and reasoning in the E8 record --
silent omission of any item from the review is itself a promotion-
completeness defect.

## 7. Authority

This document itself grants:

```
NETWORK_AUTHORIZED=false
PACKAGE_INSTALL_AUTHORIZED=false
ENVIRONMENT_MUTATION_AUTHORIZED=false
PDF_ACQUISITION_AUTHORIZED=false
PROTECTED_SOURCE_A_READ_AUTHORIZED=false
HUMAN_GATE_CONSUMED=false
```

`ENVIRONMENT_MUTATION_AUTHORIZED=false` here covers **both** the
non-canonical staging environment (Section 2b) and the canonical
`.venv-real-execution` (Section 2a): this document only defines that a
staging environment and a later live canonical promotion are permitted
patterns under future, separately reviewed authorization -- it does not
itself authorize creating, mutating, or resolving into either one.

It does not resolve frozen design LOW_1. It does not authorize SOURCE_B
calibration acquisition. It does not resolve `V9_009_HIGH_2`. It does not
materialize `trading_dates` or establish any future profitability claim.

---

## Non-claims

This document, and the act of drafting it, does not:

- edit, regenerate, supersede, or invalidate any predecessor real-
  execution-environment artifact
- mutate, create, or resolve into `.venv-real-execution` or any staging
  environment
- resolve V9_014 design LOW_1
- authorize SOURCE_B PDF calibration acquisition
- resolve `V9_009_HIGH_2`
- materialize `trading_dates`
- run or authorize T0, a backtest, a model, or any profitability claim
- perform any network request, package resolution, package install,
  environment mutation, PDF acquisition, PDF read, protected/private read,
  or API-key read
- consume any human gate
- claim any overall V9_014 implementation PASS

`V9_009_HIGH_2` remains `OPEN_REQUIRES_HISTORICAL_JPX_CALENDAR_BINDING`.
`T0_STATUS` remains `NOT_RUN`. `future_profitability_established` remains
`false`.

GPT-5.6 Sol remains the sole `FINAL_INDEPENDENT_REVIEWER` for this design,
as for every other V9_014 artifact in this study.
