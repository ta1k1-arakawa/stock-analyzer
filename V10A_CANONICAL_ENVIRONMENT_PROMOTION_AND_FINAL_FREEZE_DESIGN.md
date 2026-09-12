# V10A Canonical-Environment Promotion and Final-Freeze Design

```text
study_identity=V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR
document_role=FUTURE_PROMOTION_AND_FINAL_FREEZE_SEQUENCE_DESIGN
design_status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
design_frozen=false
human_freeze_approved=false
current_environment_state=VALIDATED_NOT_PROMOTED
```

This document defines a future promotion and final-freeze sequence for the
already frozen V10A design. It does not execute validation, promote or freeze
the canonical environment, mutate software, consume a gate, or authorize
downstream research. V10A remains a successor study to the terminal V10
study; no V10 artifact or failed V10 Step-5 attempt is reused as V10A
execution authority.

## 1. Starting state and immutable bindings

The sequence starts only from this state:

```text
V10A_ENVIRONMENT_STATE=VALIDATED_NOT_PROMOTED
V10A_NO_NETWORK_VALIDATION_ATTEMPT_1=PASS
V10A_CANONICAL_ENVIRONMENT_PROMOTED=false
V10A_ENVIRONMENT_FROZEN=false
V10A_EXECUTION_AUTHORIZED=false
V10A_CALENDAR_GENERATION_AUTHORIZED=false
V10A_T0_AUTHORIZED=false
V10A_HISTORICAL_EVALUATION_AUTHORIZED=false
future_profitability_established=false
```

The already-reviewed evidence chain is bound conjunctively:

```text
validation_evidence_sha256=bcef3587f8a86c889306a4808e3cf033b12e16fef45009788ded444983c77d1e
validation_evidence_git_blob_sha1=0a8e3afd914a905887f58d1f23a224ce5138c3ac
attempt_1_adjudication_git_blob_sha1=fbc0357d211bfa6f3450f909d6044c2e6f25ac19
reviewed_evidence_record_commit=faea2e710fe62ebc17ab2bad032ca44ea496aee0
approved_v10a_design_sha=b14cc5510685210e928000af0815e188bc1aadc0
freeze_record_sha=86ceda3dee531b08afa5db4df7af1298ca770fad
```

The source and environment requirements remain exactly those of the frozen
V10A design: the reviewed 20-package environment, CPython 3.12.10 on
Windows/AMD64 with `win-amd64`, `pandas-market-calendars==5.4.0`,
`exchange-calendars==4.13.2`, official wheel SHA-256
`bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3`, exact
JPX/JP wheel-entry uniqueness, installed-to-wheel raw-byte equality, release
blobs `a7a59b6cf910e325c85fc042459ff57ca8f70613` and
`4c34214d06862e02ac22e946757463f748074fde`, XLS/PDF synthetic probes PASS,
and zero network, installation, mutation, calendar/date, private, and T0
activity.

## 2. Promotion chain

Promotion is not inferred from the existing validation PASS. It requires the
following ordered chain. Every stage is fail-closed, and a failure preserves
its evidence without retry, rollback, reinstall, repair, environment
recreation, alternate provider, or promotion.

### P1 — promotion/freeze artifact preparation

P1 is a documentation/artifact-preparation checkpoint only. It may read the
repository's already-committed V10A evidence and source-authority records, but
it must not read the live canonical environment or wheelhouse, launch Python,
mutate software, consume authority, or perform a validation rerun.

P1 prepares a future freeze-record candidate whose unapproved state remains:

```text
canonical_environment_promoted=false
environment_frozen=false
future_protected_execution_authorized=false
```

The candidate binds the validation evidence SHA-256/blob, attempt-1
adjudication blob, reviewed evidence-record commit, approved V10A design and
freeze-record SHAs, and the frozen source/package identities. It must not
invent live P3 values or any future commit SHA. P1 does not create promotion
authority merely by creating this candidate.

### P2 — GPT exact-SHA review

The complete P1 artifact-preparation commit is independently reviewed at its
exact commit SHA. P2 is PASS only when GPT records `CRITICAL=0`, `HIGH=0`,
and `MEDIUM=0`. A P2 PASS verifies artifact/design correctness only; it does
not promote or freeze the environment and does not authorize P3.

The exact P2-reviewed commit SHA becomes a required input to P3. A moved
branch, dirty checkout, missing commit, or mismatch between the reviewed
checkout and the local execution checkout is a provenance failure.

### P3 — fresh final no-network live freeze verification

P3 is permitted only after P2 PASS and a fresh point-of-use human
authorization specifically for P3. The P3 authorization is distinct from the
V10A attempt-1 authorization and cannot be inferred from, or reused from,
that attempt.

P3 uses direct Windows PowerShell phased execution:

```text
Phase A: no-network preflight and exact provenance binding
Phase B: one reviewed final-verification runner/check
Phase C: no-network evidence inspection and durable capture
```

Before Phase B, P3 must require the authoritative remote branch and local
checkout to equal the exact P2-reviewed SHA, with a clean tree. The final
verification runner binds:

- the frozen V10A design and freeze-record identities;
- validation evidence SHA-256 `bcef3587f8a86c889306a4808e3cf033b12e16fef45009788ded444983c77d1e` and blob `0a8e3afd914a905887f58d1f23a224ce5138c3ac`;
- attempt-1 adjudication blob `fbc0357d211bfa6f3450f909d6044c2e6f25ac19`;
- the exact P1 freeze-record candidate blob/hash;
- the P2-reviewed checkout SHA and the reviewed final-verification tooling
  identity.

It verifies the same already-existing canonical environment as read-only
state. It does not recreate, mutate, reinstall, repair, or otherwise alter
that environment. It rechecks the exact package set, Python/platform,
package versions, official wheel archive SHA, exact unique JPX/JP ZIP
entries, raw installed-to-wheel bytes, release-tag Git blobs, XLS/PDF
synthetic readiness probes, and zero-operation counters. It must not create
calendar objects, inspect sessions/dates, run T0, read research/private
inputs, or use network/package resolution.

P3 failure stops the chain. There is no retry, rollback, reinstall, repair,
environment recreation, alternate provider, or promotion after a P3
failure. The fresh P3 authority is consumed only according to its own
point-of-use one-shot boundary, never from attempt 1.

### P4 — final-verification evidence commit

After a successful P3 capture, a separate repository-writing checkpoint
commits the exact durable P3 evidence and minimal authorized state/log
updates. P4 performs no rerun and no live-environment or wheelhouse read.
The artifact is:

```text
V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json
```

It is safe provenance-only evidence: no machine-local raw paths, human
authorization identity, raw protected/private material, or source bytes.
It records the exact P2-reviewed SHA, P1 candidate blob/hash, V10A design and
freeze identities, prior validation evidence bindings, final observed
package/platform/source/probe results, operation counters, and the exact
P3 authority/process result. Until P5 PASS, its promotion and freeze fields
remain false. P4 must not claim that its own future commit was reviewed.

### P5 — GPT exact-SHA final promotion review

P5 independently reviews the exact P4 evidence commit and its final
verification artifact. Only P5 PASS may transition the rolling state to:

```text
V10A_CANONICAL_ENVIRONMENT_PROMOTED=true
V10A_ENVIRONMENT_FROZEN=true
V10A_ENVIRONMENT_STATE=CANONICAL_FROZEN
```

P5 PASS must not set any downstream research authority true. These remain
separate later authorities:

```text
V10A_EXECUTION_AUTHORIZED=false
V10A_CALENDAR_GENERATION_AUTHORIZED=false
V10A_T0_AUTHORIZED=false
V10A_HISTORICAL_EVALUATION_AUTHORIZED=false
```

This document is not P5 evidence and does not make that transition.

## 3. Self-reference and exact-SHA rules

- The P1 freeze-record candidate cannot contain its own future commit SHA.
- No P1 or P4 artifact may fabricate a future SHA, blob, live result, or
  authority.
- P3 binds the already-existing exact P2-reviewed checkout SHA and exact
  artifact blobs before any live observation.
- P4 records the actual P2-reviewed SHA and the exact P3 evidence; it does
  not claim that its own commit is reviewed.
- P5 reviews P4's committed evidence and is the only stage in this sequence
  that may declare the environment promoted/frozen.
- A provenance mismatch at any stage is a stop condition, not permission to
  substitute a current or convenient artifact.

## 4. Scope and non-claims

V10A promotion/freeze is an environment-governance transition only. It does
not change V10A scientific methodology, source identity, package versions,
coverage, anchors, periods, labels, targets, economics, costs, slippage,
models, thresholds, feasibility criteria, or stopping rules.

Until the future P5 PASS, V10A remains not promoted and not environment
frozen. No stage defined here authorizes calendar generation, T0, historical
evaluation, protected/private access, broker activity, or a profitability
claim. `future_profitability_established=false` remains true as a state
claim throughout this design task.
