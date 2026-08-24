# V9_006 Stage-A retry policy review

```text
REVIEWED_SHA=34d4117d48b378946682acb2f7d313c0593271af
CRITICAL=0
HIGH=0
MEDIUM=0
RESULT=PASS
```

`V9_006_STAGE_A_RETRY_POLICY=PASS`

This records GPT's independent exact-SHA review PASS of the Stage-A
retry/backoff policy binding recorded in `V9_006_STAGE_A_RETRY_POLICY.md`
and reflected in `V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md`'s
"Retry policy binding" subsection, at reviewed commit
`34d4117d48b378946682acb2f7d313c0593271af`.

The reviewed policy:

```text
maximum_attempts_per_source_object=3
maximum_retries_per_source_object=2
backoff_seconds=[5,30]
jitter=false
```

Retries permitted only before the first complete payload, only for the
exact same source object/request (source family, applicable period,
requested URL, request parameters, and provider/domain all preserved
exactly). Retryable transport classes are exactly `NETWORK_TIMEOUT`,
`CONNECTION_RESET`, `TEMPORARY_DNS_FAILURE`, `HTTP_408`, `HTTP_425`,
`HTTP_429`, `HTTP_500`, `HTTP_502`, `HTTP_503`, `HTTP_504`; no other
condition is retryable. Exhaustion of all retryable-only failures yields
`failure_class=PLUMBING_FAILURE_RETRIABLE` and STOPs the run -- never
`SOURCE_OR_DATA_FEASIBILITY_FAILURE` -- with no automatic new run and no
reuse of the same human authorization.

This PASS creates no network, data, T1, or design-freeze authority, and
does not by itself authorize any Stage-A execution. It closes the retry-
policy methodology question that
`V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_HIGH_1_MONTHLY_COVERAGE_MAPPING`'s
prohibited-items list had explicitly left open ("this task does NOT decide
retry count/backoff; existing V9_006 invented retry policy remains
unresolved and real execution stays BLOCKED"); real Stage-A execution
remains `BLOCK`ed pending the separately tracked locator/inventory
implementation review and a fresh, separate, explicit Stage-A human
network authorization.
