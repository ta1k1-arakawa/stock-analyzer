# V8I Source-Snapshot Execution Incidents

This record separates operational wrapper and environment incidents from the
scientific/terminal V8I source-snapshot result, following the same
chronological-incident-plus-adjudication pattern already used by
`V8G_LOCATOR_EXECUTION_INCIDENTS.md`. See
`V8I_SOURCE_SNAPSHOT_TERMINAL_ADJUDICATION.json` for the machine-readable
disposition this record supports.

## Chronological record

1. **Operational — stale local checkout, PRE_GATE.** The first attempted
   execution ran before local synchronization. Local HEAD and the
   authoritative remote were both stale at `5f0d7eff2df1728abcf58b7eddd16329a9010f8e`
   (a commit prior to the V8I design freeze and its reviewed source-snapshot
   implementation). The Python process failed to launch, reporting Windows
   Access Denied, before any authorized operation began. No durable
   `HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE` receipt was published at
   this point, no JPX request was made, and no private read occurred. This
   was a `PRE_GATE` transport/environment failure only, per
   `AI_REAL_EXECUTION_RUNBOOK.md` §4; it did not consume the gate and
   created no authority.

2. **Operational — safe synchronization.** The local repository was then
   synchronized so that local HEAD, the authoritative remote branch
   (`v8g-private-partition-locator-successor-design`), and the reviewed
   V8I source-snapshot implementation SHA all matched exactly at
   `0dddc380538e405895cbba18d415b32f9ef615c8`, with a clean working tree
   and a working Python launcher (`py -3`). This was a sync-only recovery
   per `AI_REAL_EXECUTION_RUNBOOK.md` §3: it created no methodology
   authority and consumed no gate.

3. **Scientific/production — authorized one-shot execution, POST_GATE
   failure.** The authorized V8I one-shot source-snapshot execution was
   then run under the fresh, exact-SHA-bound human authorization
   (`V8I_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_<reviewed design
   commit>_WITH_<reviewed implementation SHA>`; only its SHA-256 is
   recorded, never the raw sentence). Per the frozen V8I design (§3.1's
   `IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST` boundary), the durable
   `HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE` receipt was published, and
   only then was the one authorized official JPX source-snapshot request
   made and its raw bytes obtained. Execution then failed with
   `ModuleNotFoundError: No module named 'pandas'` inside
   `default_parse_source_table(raw_bytes)`, at the `import pandas as pd`
   statement, before any parsing of the raw bytes into a source table
   occurred. The trace proves this failure occurred strictly after the
   durable gate receipt boundary and strictly after the one authorized JPX
   request had already returned raw bytes.

4. **Read-only state verification.** A subsequent read-only check of local
   state found exactly:

   ```text
   receipt_exists=True
   evidence_exists=False
   private_raw_file_count=0
   ```

   The durable gate receipt exists and is consumed. No
   `V8I_SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE_V1` artifact was ever
   published, because the post-gate failure occurred before parsing
   completed and before the evidence-producing stage (which also performs
   private raw-source preservation, per design §3.2 and the `preserve_raw_
   source_bytes_once` primitive) was reached. Consequently no private raw
   source snapshot was preserved either.

## Adjudication

Per `AI_REAL_EXECUTION_RUNBOOK.md` §4 and §10, and the frozen V8I design's
own §6 failure semantics (`POST_GATE_TERMINAL_CONDITIONS` includes parsing
failure; "ANY failure here is terminal for this V8I source-snapshot
attempt"):

- The `HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE` one-shot gate was
  consumed. A crash after consumption does not restore authorization.
- One real JPX source acquisition occurred: the one authorized request was
  made and raw bytes were obtained.
- No valid `V8I_SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE_V1` artifact was ever
  produced or published.
- No privately preserved raw source artifact exists (`private_raw_file_
  count=0`); the raw bytes obtained during the one authorized request were
  never durably preserved because the crash occurred before the
  preservation step in the execution boundary was reached.
- This V8I source-snapshot attempt is therefore terminal:
  `BLOCK_CLOSED`. No same-study retry, no second JPX request, no receipt
  reset/deletion/reuse, and no reuse of this authorization are permitted
  under any circumstance. Installing the missing dependency and retrying
  under this same V8I study and this same consumed gate is explicitly
  prohibited by the same rule.
- No partition seed was generated and no partition allocation occurred;
  partition generation, membership disclosure, research opening, and
  production all remain unauthorized, exactly as they were before this
  attempt.
- This is classified as an execution-environment/readiness failure — a
  missing required production dependency discovered only after the gate
  boundary — and explicitly **not** a strategy failure and **not** a
  profitability failure. No hypothesis, label, cost model, promotion rule,
  search space, or evaluated strategy was ever reached; `T0` reproduction
  is not claimed to have passed or failed, because parsing never completed
  to a valid result; no fresh-eligible-count is claimed.
- Future profitability remains unestablished.
- A successor-study decision is required before any further V8I-lineage
  source-snapshot attempt.

## Lessons and future runbook notes

- A stale local checkout that fails before the gate boundary must be
  resynchronized and re-verified from the beginning, never treated as a
  scientific result.
- The production runner's dependency set (at minimum `pandas`, required by
  `default_parse_source_table`) must be verified present in the execution
  environment as part of preflight, *before* the gate's durable receipt is
  published — moving this check earlier would have kept this failure
  `PRE_GATE` instead of terminal `POST_GATE`.
- Once a `HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE` receipt exists,
  absolutely no rerun, reset, dependency-repair-and-retry, or substitution
  is permitted for this study; any further attempt requires a fresh
  successor-study identity, exactly mirroring the permanence discipline
  V8G froze for its own locator gate and V8H and V8I both inherited.
- Operational/environment-readiness failures must not be misclassified as
  strategy or profitability failures, and must not be allowed to imply a
  passed or failed `T0` reproduction or any fresh-eligible-count when
  parsing never reached a valid result.
