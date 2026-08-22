# V8J Source-Snapshot Environment Successor Design Draft

This is a successor-study **design draft only**. It creates no implementation,
freeze, network, private/sealed-data, source-snapshot, partition-generation,
membership-disclosure, research-opening, production, human-authorization, or
gate-consumption authority.

## 1. Study identity, frozen predecessor, and reason for succession

```text
study=V8J_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8I_HISTORICAL_RESEARCH
predecessor_terminal_adjudication_commit=5e0a1b9f66f9264d1bc5d3f6d978a23bbbe9a1c8
predecessor_frozen_design_commit=ec4c3709afcce24c0a07373b982de5bfd9bb4d23
predecessor_frozen_design_blob=5ffccc0421d0f58832dca9cc0a3541318281aa76
```

V8I remains permanently `BLOCK_CLOSED`. Its one-shot source-snapshot gate
was consumed and its terminal disposition was `POST_GATE
EXECUTION_ENVIRONMENT_FAILURE`; this V8J draft neither reopens nor amends
that disposition.

V8I's receipt, authorization, gate, raw bytes, and failed attempt MUST NOT
be reset, deleted, reused, reinterpreted, reconstructed, or retried. No V8I
authorization, receipt, receipt key, or gate can satisfy a V8J requirement.

```text
V8I_strategy_failure=false
V8I_profitability_failure=false
V8I_future_profitability_established=false
V8J_reason_for_successor=POST_GATE_EXECUTION_ENVIRONMENT_FAILURE
```

No scientific result is inferred from that V8I failure.

## 2. Scientific methodology is inherited unchanged

```text
V8J_scientific_methodological_change_relative_to_V8I=NONE
future_profitability_established=false
```

V8J inherits V8I's complete frozen scientific methodology unchanged. In
particular, it retains without recalibration, substitution, or reinterpretation:

- the source-snapshot receipt/evidence split and official JPX source
  semantics;
- exact T0 reproduction and the V4 raw-byte-equality policy;
- fresh eligible minimum `900`; `T0`/`T1`/`T2`/`T3`/`T_spare` semantics;
- block size `300`, 32-byte OS-CSPRNG seed semantics, HMAC allocation,
  canonical hashes/manifests, and private/public manifest separation;
- all leakage, outcome-selection, manual-membership, and future-price-access
  prohibitions;
- partition-generation semantics and preservation/point-of-use verification;
- evaluation period, labels/targets, strategy, cost/slippage, DQ and
  promotion thresholds, search space, stopping rules, and
  forward-performance interpretation.

The inherited V8I methodology includes the exact V8I receipt/evidence
structure. This draft successorizes that structure only into the fresh V8J
namespace described below; it does not change its scientific meaning.

If an inherited V8I rule conflicts with an explicit V8J decision in this
draft, execution MUST stop with `CHATGPT_DECISION_REQUIRED`. No execution
agent may invent a resolution.

## 3. New operational pre-gate environment prerequisite

V8J adds one operational prerequisite only; it is not a scientific
methodology change:

```text
environment_freeze_promotion_commit=f26c4138bd7b1fb9ea1394ed04a1a600a3fee425
CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE=YES
```

The V8J source-snapshot gate MUST NOT be consumed unless the reviewed frozen
real-execution environment is mechanically `PASS` at the exact point of
execution. Before publication of the V8J gate receipt, the canonical
interpreter MUST be:

```text
.venv-real-execution\Scripts\python.exe
```

Using that canonical interpreter, `scripts/check_real_execution_env.py` MUST
exit with code `0` and its safe result contract MUST establish every one of:

```text
REAL_EXECUTION_ENVIRONMENT_READY=true
ENVIRONMENT_LOCK_CHECK=PASS
ENVIRONMENT_FREEZE_CHECK=PASS
ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH=true
ENVIRONMENT_LOCK_FINGERPRINT_STATUS=FROZEN
REAL_EXECUTION_ENVIRONMENT_FROZEN=true
REAL_NETWORK_REQUESTS=0
PRIVATE_READS=0
GATES_CONSUMED=0
```

At that same point of use, the following environment-critical files MUST be
unchanged relative to commit
`f26c4138bd7b1fb9ea1394ed04a1a600a3fee425`; a comparison failure, absence,
or inability to prove equality is a failure:

```text
REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json
REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json
REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json
requirements-real-execution.lock.txt
requirements-real-execution.txt
scripts/check_real_execution_env.py
scripts/bootstrap_real_execution_env.ps1
tests/fixtures/synthetic_jpx_source_snapshot.xls
```

This prerequisite is entirely `PRE_GATE`: any `FAIL`, `UNKNOWN`, malformed
output, wrong interpreter, nonzero checker exit, file-identity mismatch, or
failure to prove the stated dependency-readiness proposition is a
`PRE_GATE_BLOCK` and consumes no V8J gate. A later intentional environment
change requires separate GPT review and freeze; V8J must not silently accept
it.

The FROZEN environment is a necessary readiness condition only. It is
explicitly **not** acquisition authorization, does not authorize JPX/Yahoo
or private/sealed access, and does not consume any research gate.

## 4. Fresh V8J source-snapshot gate and receipt/evidence contract

```text
gate=HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST
```

The gate permits only the one V8J source-snapshot acquisition and inherited
T0-reproduction validation. It does not authorize partition generation,
membership disclosure, raw historical-price acquisition, research opening,
or production.

### 4.1 Pre-request receipt

Publish a durable, exclusive, no-overwrite receipt strictly before the first
authorized JPX request. It contains only values knowable at that instant:

```text
schema_version (= "V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_V1")
artifact_role (= "SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT")
study (= "V8J_HISTORICAL_RESEARCH")
gate (= "HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE")
reviewed_v8j_design_candidate_commit
reviewed_v8j_source_snapshot_support_implementation_sha
authorization_identity_sha256
consumed (= true)
consumption_count (= 1)
consumption_boundary (= "IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST")
consumption_timestamp_utc
```

It MUST NOT contain post-request values such as a source raw hash, source
acquisition timestamp, eligible-ticker count/list hash, or T0-reproduction
status. A missing, extra, or post-request field makes the receipt
schema-invalid and fail-closed.

### 4.2 Post-request evidence

After the one authorized request has completed and its source snapshot has
been parsed under the inherited rules, publish a distinct canonical,
write-once, public-safe evidence artifact:

```text
schema_version (= "V8J_SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE_V1")
artifact_role (= "SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE")
study (= "V8J_HISTORICAL_RESEARCH")
reviewed_v8j_design_candidate_commit
reviewed_v8j_source_snapshot_support_implementation_sha
source_snapshot_gate_receipt_key_sha256
source_snapshot_gate_receipt_bytes_sha256
source_snapshot_semantics
source_snapshot_clarification_commit
v4_raw_sha_equality_required (= false)
source_raw_sha256
source_raw_byte_count
source_acquisition_utc
t0_reproduction_status
eligible_ticker_count
eligible_ticker_list_sha256
t0_ticker_list_sha256
fresh_eligible_count
ticker_identities_exposed (= false)
private_path_exposed (= false)
raw_payload_exposed (= false)
historical_price_raw_acquisition_performed (= false)
partition_generation_authorized (= false)
membership_disclosure_authorized (= false)
research_opened (= false)
source_snapshot_result (= "PASS")
source_snapshot_artifact_self_sha256
```

Before evidence publication, the exact durable receipt bytes MUST be read and
structurally revalidated; their bound design, implementation, and
authorization-hash fields MUST exactly equal the current authorized values.
The evidence must bind cryptographically to both the fixed V8J receipt key
and the exact validated receipt bytes. It may report `PASS` only after all
inherited source-universe reconstruction, T0 reproduction, and fresh
eligible-count checks pass.

No ticker identity, private path, raw payload, partition membership, or raw
human authorization identity may appear in public output, logs, exceptions,
or stdout.

## 5. Fresh V8J receipt key and authorization grammar

```text
receipt_key_material =
    "V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1\0"
  + "ta1k1-arakawa/stock-analyzer"
  + "\0"
  + "V8J_HISTORICAL_RESEARCH"
  + "\0"
  + "HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE"

source_snapshot_gate_receipt_key_sha256=SHA256(UTF8(receipt_key_material))
```

This V8J-only key is fixed and excludes attempt-varying inputs. It neither
equals, derives from, accepts, nor may be substituted with any V8I key or
receipt.

```text
authorization_identity =
    "V8J_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_"
  + reviewed_v8j_design_candidate_commit
  + "_WITH_"
  + reviewed_v8j_source_snapshot_support_implementation_sha
```

Both SHA components are exactly 40 lowercase hexadecimal characters. A
wrong length, uppercase character, non-hex character, or malformed grammar
is a `PRE_GATE_BLOCK`, never coerced. Only
`SHA256(UTF8(authorization_identity))` may be persisted or logged; the raw
authorization identity MUST NOT be persisted, printed, or logged.

No V8I authorization identity, its hash, gate, receipt, receipt key, or raw
human authorization may satisfy this V8J gate.

## 6. Failure semantics

Before durable V8J receipt publication, every provenance, design/implementation
binding, authorization, cleanliness, environment-readiness, or prerequisite
failure is `PRE_GATE`; the V8J gate remains unconsumed.

After durable publication of the
`HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE` receipt, **any** failure is
terminal for the V8J source-snapshot attempt, including request, private
preservation, parse, T0 reproduction, fresh-eligible-count, hashing, or
evidence-publication failure.

```text
no_second_jpx_request=true
no_provider_or_date_substitution=true
no_receipt_reset_or_deletion=true
no_same_study_retry=true
no_authorization_reuse=true
successor_study_required_on_post_gate_failure=true
```

## 7. Stage authority and task boundary

The future design-freeze gate is:

```text
HUMAN_V8J_DESIGN_FREEZE
```

This draft does not consume or approve that gate. It does not authorize a
V8J design freeze, support implementation, JPX/Yahoo access, private/sealed
reads, partition generation, membership disclosure, research opening, or
production. Any later stage requires its own applicable human and GPT
authority under the inherited governance.

```text
network=0
JPX=0
Yahoo=0
private_reads=0
gate_consumption=0
raw_acquisition=0
partition_generation=0
research_opening=0
implementation=0
design_finalized=false
human_design_freeze_complete=false
V8I_gate_or_receipt_reused=false
V8I_authorization_reused=false
```

```text
next_action=GPT_EXACT_SHA_INDEPENDENT_DESIGN_REVIEW
```
