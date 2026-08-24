# V9_006 HIGH-1 independent review

```text
REVIEWED_SHA=8cdc1d89c1fc0eb83a0d50b69c2769e2e79d5761
CRITICAL=0
HIGH=4
MEDIUM=1
RESULT=BLOCK
```

THIS TASK REMEDIATES ONLY: `V9_006_HIGH_1_STAGE_A_LOCATOR_CONTRACT_FAIL_CLOSED`

FINDING=V9_006_HIGH_1_STAGE_A_LOCATOR_CONTRACT_FAIL_CLOSED

The V9_006 implementation's `run_stage_a` knowingly constructed all monthly
`SOURCE_INVENTORY` cells as `MISSING` for every one of the seven required
source families, and still proceeded to cross the JPX network boundary --
fetching and locking the two non-monthly artifacts that do have a reviewed
locator (the listed-issues page and the JPX Calendar page) -- before
computing a guaranteed `FREE_JPX_METADATA_PROBE_FAIL`. The original
execution task required `CHATGPT_DECISION_REQUIRED` if the locator contract
was methodologically insufficient. A knowingly doomed real-network run is
not an acceptable substitute for stopping before the boundary: the seven
source families' deterministic locator/cadence contract was never actually
complete, so real Stage-A execution should never have been allowed to
reach the network at all under current reviewed evidence.

FINDING_STATUS=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW

## Remediation implemented

A new pre-network locator-readiness check,
`verify_locator_contract_complete()`, is added to
`src/v9_005_stage_a_jpx_probe.py`. It performs no I/O and invents no URL,
cadence, N/A rule, archive period, retry rule, or source substitution: it
only asks the existing `resolve_month_locator(family, month)` seam, for
every one of the seven source families across all 108 required inventory
months (2017-01 through 2025-12), whether a locator is already mechanically
resolvable from already-reviewed repository evidence. If any required cell
has no resolvable locator, it raises:

```text
V9005StageABlocked(STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE)
failure_class=CHATGPT_DECISION_REQUIRED
```

`run_stage_a` now calls this check as its very first step, immediately
after validating the Stage-A confirmation token and before
`initialize_output_root`, `verify_signal_grid_binding`, or any fetch.
Because every monthly cell currently lacks a resolvable locator (only two
non-monthly artifacts -- the listed-issues page and the calendar page --
have any reviewed locator at all, and neither is a per-month archive), this
check unconditionally stops real Stage-A execution today: zero fetch
calls, zero git calls, and no output-root directory is even created.

`CHATGPT_DECISION_REQUIRED` is a new, distinct public status/failure class,
separate from `SOURCE_OR_DATA_FEASIBILITY_FAILURE`. The internal-reason-to-
public-failure-class mapping now sends
`STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE` to `CHATGPT_DECISION_REQUIRED`
only; `SOURCE_OR_DATA_FEASIBILITY_FAILURE` is never produced for this
condition, and remains reserved for a genuine probe result produced only
after the locator contract is complete and the actual approved source
probe has run.

`scripts/run_v9_005_stage_a_jpx_probe.py`'s safe JSON report is extended
(minimally) so that, when `failure_class == CHATGPT_DECISION_REQUIRED`, it
also includes explicit `status` and `reason` fields
(`"status":"CHATGPT_DECISION_REQUIRED"`,
`"reason":"STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE"`) alongside the
existing safe fields -- both values come from the module's fixed,
non-secret reason vocabulary, never a raw exception message or private
data. `scripts/run_v9_005_stage_a_jpx_probe.ps1` required no change: it
already only forwards to the Python entrypoint and reports its exit code
and JSON output, so it automatically inherits the new fail-closed
behavior.

## Scope discipline

No other GPT finding from this review is remediated in this commit, per
the task's explicit instruction. In particular, semantic validation,
transport provenance, redirect handling, and retry methodology are
unchanged from the prior implementation. No JPX URL, URL pattern, source
cadence, N/A rule, archive period, retry rule, or source substitution was
invented anywhere in this change.

## Tests

`tests/test_v9_005_stage_a_jpx_probe.py` grows from 54 to 58 tests, all
offline (synthetic fixtures and injected fake fetchers/git callables only):

- `test_locator_contract_is_currently_incomplete` -- ground truth: the
  contract is incomplete under current reviewed evidence.
- `test_locator_contract_complete_passes_when_no_missing_cells` -- the
  check itself is correct when every cell resolves.
- `test_run_stage_a_incomplete_locator_contract_stops_before_any_network`
  -- `run_stage_a` stops with `CHATGPT_DECISION_REQUIRED`, zero fetcher
  calls, zero git calls, and no output-root directory created; explicitly
  asserts `failure_class != SOURCE_OR_DATA_FEASIBILITY_FAILURE`.
- `test_cli_script_incomplete_locator_contract_prints_safe_chatgpt_decision_required`
  -- the CLI's safe JSON report carries the exact `status`/`reason`
  contract with zero network requests.
- The two prior integration tests that exercised the fetch/lock/evidence
  pipeline below the gate (`test_run_stage_a_offline_reports_fail_with_
  safe_evidence`, `test_run_stage_a_wrong_signal_grid_blob_stops_before_
  any_fetch`) are preserved as regression coverage, renamed to make explicit
  that they run with the locator-contract gate forced complete via
  `monkeypatch` (simulating a future, separately reviewed extension) --
  they are not a claim that the real contract is complete today.
- All other existing offline tests are unchanged and still pass.

## Authority created

```text
NETWORK_REQUESTS=0
DATA_ACQUIRED=false
HUMAN_GATE_CONSUMED=false
T1_OR_DESIGN_FREEZE_AUTHORITY_CREATED=false
```

This remediation only adds a pre-network fail-closed check and its minimal
supporting report field. It does not authorize network access, data
acquisition, T1 membership generation or opening, model fitting,
backtesting, profit calculation, or V9 design freeze, and does not consume
the human's existing chat-given Stage-A authorization.

## Next action

`V9_006_HIGH_1` remains `REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW` --
not `PASS` or `RESOLVED` -- until GPT independently reviews this
remediation at its exact commit SHA. `V9_006_STAGE_A_IMPLEMENTATION`
remains `BLOCK` pending that review (and any other still-open V9_006
findings from the original review).

## GPT exact-SHA independent review — PASS

```text
REVIEWED_SHA=c525fe0a71a841ad6a01e0b911c4386f2672f9e3
PARENT_SHA=8cdc1d89c1fc0eb83a0d50b69c2769e2e79d5761
CRITICAL=0
HIGH=0
MEDIUM=0
RESULT=PASS
```

FINDING_STATUS=RESOLVED

`V9_006_HIGH_1` is `RESOLVED`. `V9_006_STAGE_A_IMPLEMENTATION` remains
`BLOCK` overall because other findings from the original review remain
open. This PASS covers only the pre-network locator-readiness gate; it
creates no network, data, T1, or design-freeze authority, and does not
authorize Stage-A execution. Before any real Stage-A network request, GPT
methodology authority must still bind the exact source-slot cadence and a
deterministic official-JPX locator traversal for the seven required source
families -- `resolve_month_locator` currently resolves no locator for any
family/month, so `verify_locator_contract_complete()` continues to stop
every real run until that methodology binding exists and is implemented.
