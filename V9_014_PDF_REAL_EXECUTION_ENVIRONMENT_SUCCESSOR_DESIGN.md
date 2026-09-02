# V9_014 PDF Real-Execution Environment Successor Design

STUDY_ID: V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR
STATUS: DRAFT_AWAITING_GPT_REVIEW
FROZEN_DESIGN_GIT_SHA: efee3d0efca368645c00aeed63cb8e0637cd3672
FROZEN_DESIGN_BLOB_SHA: 2bbacbf37ab961d1cbf416b7fd476db18778c5b7
PREDECESSOR_GOVERNANCE_HEAD: af0ac863f6e25abd4335ca2cb4c129ed7c3c3814

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

## 2. Canonical environment

The successor remains governed by the repository's single canonical path:

```
.venv-real-execution
.venv-real-execution\Scripts\python.exe
```

No parallel V9-specific venv, conda environment, system-Python fallback, or
ad-hoc interpreter is created or authorized by this design or by any
downstream stage it defines.

Until successor promotion (Section 5, Stage E9) is independently reviewed
`PASS`:

- the current frozen environment (Section 1) remains canonical and
  continues to serve already-authorized predecessor (V8-lineage) work
  unchanged;
- V9_014 PDF execution requiring `pdfplumber` remains **BLOCKED**.

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
defined in Section 5, Stage E5.

No parser substitution is permitted: `pdfplumber==0.11.10` is the sole PDF
engine, per `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`
Section 1, unchanged and reaffirmed here, not re-decided.

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

| Role | Successor artifact name (future; not created by this task) | Predecessor analogue |
|---|---|---|
| Successor direct-spec candidate | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt` | `requirements-real-execution.txt` |
| Successor resolved lock candidate | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json` | `REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json` |
| Successor Windows validation evidence | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_WINDOWS_VALIDATION_EVIDENCE.json` | `REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json` |
| Successor freeze/promotion record | `V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json` | `REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json` |

None of these four artifacts is created, populated, or reserved by this
design task; it only fixes their names and roles for the implementation
stage (Section 5, Stage E2) that will actually produce the first two, and
the execution stages (E3-E7) that will produce the latter two.

If the repository's existing generic canonical environment files
(`requirements-real-execution.txt`, `requirements-real-execution.lock.txt`,
`REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json`,
`REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json`,
`REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json`) must eventually change to
incorporate the successor, that change occurs **only** in a separately
reviewed **PROMOTION commit** (Section 5, Stage E8), and only after the
successor candidate/evidence artifacts have themselves passed exact-SHA
review (Section 5, Stage E7 `PASS`). This design task does not perform,
schedule, or pre-approve that promotion commit; it only defines the
condition under which one may later occur.

## 5. Resolution / execution phases

The following order is frozen. No stage may be skipped or reordered; each
stage marked "GPT ... PASS" requires an independent exact-SHA GPT-5.6 Sol
review before the next stage may begin.

```
E1  This successor-design exact-SHA GPT PASS.

E2  Offline implementation of successor bootstrap/checker/candidate
    machinery + targeted synthetic/static tests + GPT exact-SHA PASS.
    (No network, no install, no environment mutation; produces the
    successor direct-spec candidate and the tooling that will later
    produce the successor lock candidate -- code and tests only.)

E3  Phase A NO-NETWORK Windows preflight:
    - repo/branch/exact HEAD/clean tree verification
    - predecessor hash verification (Section 1 table, re-derived and
      compared, never trusted from memory)
    - canonical Python/platform prerequisite verification
    - NO environment mutation of any kind

E4  GPT reviews Phase A.
    A fresh point-of-use human/network/durable-state authority is obtained
    if repository governance (AI_RESEARCH_EXECUTION_RULES.md,
    AI_REAL_EXECUTION_RUNBOOK.md) requires it at that time; no authority
    from this document or from any prior V9 stage is reused.

E5  Phase B MINIMAL REAL ENVIRONMENT RESOLUTION:
    - resolve exactly once from the reviewed successor direct spec
      (Section 3) on the canonical Windows environment path
      (Section 2)
    - NO PDF/source acquisition of any kind
    - stdout/stderr preserved separately, unmodified
    - a nonzero exit means: no retry; proceed directly to Phase C (E6)
      with the failure evidence intact

E6  Phase C NO-NETWORK inspection:
    - exact package freeze
    - Python/platform fingerprint
    - direct import proof, including `pdfplumber`
    - candidate/evidence hashes
    - process exit status
    - predecessor/successor provenance cross-reference
    - NO retry and NO package repair of any kind

E7  Commit the successor candidate/evidence artifacts (Section 4) only,
    and obtain GPT exact-SHA review of that commit.

E8  A separate, independently reviewed **promotion commit** updates the
    canonical generic lock/freeze authority (the five files named in
    Section 4) atomically, if and only if E7 is `PASS`.

E9  GPT exact-SHA promotion review `PASS`.
    Only then may V9_014 PDF calibration-runner Stage C (per
    `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`
    Section 7) begin.
```

## 6. Fail-closed

The following are never permitted at any stage of this successor path:

- "pip install missing package" as an ad hoc remediation
- dependency/version substitution of any kind
- a second resolution attempt after a Phase B (E5) failure
- reuse of prior human authorization (from this document, from any prior
  V9 stage, or from the predecessor environment's own authorization)
- environment deletion/reset performed to erase failure evidence

No package shopping is permitted until a stage's own GPT `PASS` is
recorded.

Any governance decision needed to fully specify E2-E9 that is not already
answered by this document, by
`V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`, or by
existing repository governance (`AI_RESEARCH_EXECUTION_RULES.md`,
`AI_RESEARCH_CHECKPOINT_WORKFLOW.md`, `AI_REAL_EXECUTION_RUNBOOK.md`,
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md`) is `CHATGPT_DECISION_REQUIRED` and
may not be decided unilaterally by an executor at implementation time.

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

It does not resolve frozen design LOW_1. It does not authorize SOURCE_B
calibration acquisition. It does not resolve `V9_009_HIGH_2`. It does not
materialize `trading_dates` or establish any future profitability claim.

---

## Non-claims

This document, and the act of drafting it, does not:

- edit, regenerate, supersede, or invalidate any predecessor real-
  execution-environment artifact
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
