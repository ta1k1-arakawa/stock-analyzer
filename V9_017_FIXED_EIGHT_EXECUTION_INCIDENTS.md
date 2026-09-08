# V9_017 Fixed-Eight Execution Incidents

## Phase-B wrapper observability incident

The external Phase-B PowerShell wrapper started the V9_017 runner but failed
to durably capture the runner process exit code. This is classified as
`EXECUTION_ORCHESTRATION_OBSERVABILITY_INCIDENT` and is separate from the
runner's research result.

The runner process is no longer active. Its stdout was valid safe JSON and
stderr was empty. The durable runner `attempt.json` and `failure.json`
receipts are present; `result.json`, `complete.json`, and a durable process
exit-code capture are absent. The runner terminal receipt therefore remains
adjudicable as `FAIL` / `DATA_QUALITY_FAILURE` /
`FIXED_EIGHT_LOCATOR_FAILURE`.

The wrapper incident does not convert the runner's data-quality failure into
an implementation failure and creates no retry authority. No second
fixed-eight execution, refetch, or same-study post-observation locator repair
is allowed.
