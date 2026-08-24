# V9_006 Stage-A semantic validation implementation review

```text
REVIEWED_SHA=85d22b3c409a467bcd8084ba4b3d3a4f4516ee64
CRITICAL=0
HIGH=2
MEDIUM=1
RESULT=BLOCK

FINDINGS:
1. V9_006_HIGH_2_SEM_IMPL_HIGH_1_TERMINAL_IDENTITY_VALIDATION_AND_COLLISION
2. V9_006_HIGH_2_SEM_IMPL_HIGH_2_ORPHAN_EVENT_IDENTITY_NOT_REJECTED
3. V9_006_HIGH_2_SEM_IMPL_MEDIUM_1_TWO_RUN_DETERMINISM_NOT_IN_PASS_GATE
```

This records GPT's independent exact-SHA `BLOCK` review of the Stage-A
semantic-validation implementation (`src/v9_005_stage_a_semantics.py` and
its `v9_005_stage_a_jpx_probe.py` wiring) at reviewed commit
`85d22b3c409a467bcd8084ba4b3d3a4f4516ee64`, and this task's remediation of
exactly the first of its three findings.

## Finding 1 (remediated this task)

`V9_006_HIGH_2_SEM_IMPL_HIGH_1_TERMINAL_IDENTITY_VALIDATION_AND_COLLISION`:
`compute_semantic_validation_result` consumed every `terminal_identities`
entry without validating its shape or fields, and normalized canonical
codes were mapped into `identities` via a plain dict assignment
(`identities[code] = state`) that silently overwrote an earlier entry
whenever two distinct raw keys normalized to the same `canonical_code`
(e.g. `"1301"` and `" 1301 "`, or `"130a"` and `"130A"`). A caller could
therefore smuggle a non-`TerminalIdentityState` value, a non-bool
`listed_state` (including a truthy int like `1`), an empty/non-string
`market_state`, or an out-of-enum `security_type_state` straight into
semantic reconstruction, and a genuine code collision would resolve to
whichever entry happened to be inserted last -- both silently, with no
failure signal.

**Remediation implemented this task:** a new `_validate_terminal_identity_
state` helper checks each of the three fields independently (`listed_
state` must be an actual `bool` -- `isinstance(1, bool)` is `False` in
Python, so `1` is correctly rejected despite `1 == True`; `market_state`
must be a non-empty `str`, no market enumeration invented; `security_type_
state` must be exactly one of `VALID_SECURITY_TYPE_STATES`; a non-
`TerminalIdentityState` value fails all three). Each field's invalidity
fails closed exactly the evidence items the task specifies: invalid
`listed_state` fails `listing_transition_pass`/`delisting_transition_
pass`/`canonical_identity_pass`/`deterministic_reconstruction_pass`;
invalid `market_state` fails `market_transition_pass`/`canonical_identity_
pass`/`deterministic_reconstruction_pass`; invalid `security_type_state`
fails `security_type_pass`/`canonical_identity_pass`/`deterministic_
reconstruction_pass`. Overall semantic validation therefore cannot PASS
with any invalid terminal state.

Separately, canonical-code collision detection now runs strictly AFTER
normalization: raw `terminal_identities` keys are grouped by their
normalized `canonical_code`, and any code with more than one distinct raw
key is a collision -- neither entry is used (no silent overwrite), the
internal reason `DUPLICATE_CANONICAL_IDENTITY` is recorded, and
`canonical_identity_pass`/`deterministic_reconstruction_pass` both fail.
This holds even when both colliding states are byte-identical -- duplicate
identity always fails regardless of content equality, since the true
defect is that two distinct raw representations of the same identity were
supplied at all, which is itself unresolvable without further reviewed
methodology (not execution-agent discretion).

The exact canonical-code grammar and all previously reviewed methodology
(events, conflict/corroboration dedup, reused-code detection, reverse/
forward reconstruction, the UNKNOWN-while-listed check) are unchanged.
Orphan-event-identity rejection (finding 2) and the two-run determinism
gate (finding 3) were explicitly not touched, per this task's scope.

`V9_006_HIGH_2_SEM_IMPL_HIGH_1=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`

## Finding 2 (OPEN -- explicitly out of scope this task)

`V9_006_HIGH_2_SEM_IMPL_HIGH_2_ORPHAN_EVENT_IDENTITY_NOT_REJECTED`: not
remediated. Events whose `canonical_code` has no corresponding
`terminal_identities` entry are still silently ignored rather than
rejected, per this task's explicit prohibition ("Do not fix orphan event
codes in this task"). `V9_006_HIGH_2_SEM_IMPL_HIGH_2=OPEN`.

## Finding 3 (OPEN -- explicitly out of scope this task)

`V9_006_HIGH_2_SEM_IMPL_MEDIUM_1_TWO_RUN_DETERMINISM_NOT_IN_PASS_GATE`:
not remediated. The "two independent deterministic reconstructions from
identical input produce byte-identical output" check
(`reconstruction_is_deterministic` in `v9_005_stage_a_jpx_probe.py`)
remains a standalone, separately tested property not folded into
`deterministic_reconstruction_pass`, per this task's explicit prohibition
("Do NOT alter deterministic two-run gate in this task").
`V9_006_HIGH_2_SEM_IMPL_MEDIUM_1=OPEN`.

## What this remediation does not do

This is not a GPT review -- it records the BLOCK review this task responds
to and this task's own remediation claim for finding 1 only. It creates no
network, data, T1, or design-freeze authority, and does not by itself
authorize any Stage-A execution, which remains `BLOCK`ed pending: GPT's
independent exact-SHA review of this remediation
(`GPT_EXACT_SHA_V9_006_HIGH_2_SEM_IMPL_HIGH_1_REVIEW`); remediation and
PASS of findings 2 and 3; PASS of the still-open original HIGH_3 (raw
provenance/content-lock) and HIGH_4 (redirect-before-body-consumption)
findings; a future, separately reviewed F2-F7 acquisition/parser-
integration implementation task; and a fresh, separate, explicit
point-of-use human network authorization obtained after all of the above.
`src/v9_005_stage_a_jpx_probe.py` was not modified by this task;
`ACQUISITION_IMPLEMENTATION_COMPLETE` remains `False` and
`verify_acquisition_implementation_ready()` is unchanged, so a valid real
run still stops before any filesystem/git/network access.
