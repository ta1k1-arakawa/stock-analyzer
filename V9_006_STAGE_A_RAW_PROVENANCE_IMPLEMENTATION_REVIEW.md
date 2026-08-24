# V9_006 Stage-A raw provenance implementation review

```text
task=V9_006_HIGH_3_RAW_PROVENANCE_CONTENT_LOCK_BOUNDARY
status=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
```

This task remediates original `V9_006_HIGH_3` only. Transport responses now
carry their observed payload, resolved URL, and HTTP status together. Lock
metadata records that exact response rather than fabricating status `200`.

For F1, the `LISTED_ISSUES_PAGE_URL` discovery HTML is raw-locked as
`TERMINAL_DISCOVERY_ROOT` before `data_j.xls` is extracted from its bytes.
The derived XLS is then fetched and locked separately as `TERMINAL`; its
`requested_url` is the derived XLS URL actually sent to the fetcher. Both
objects retain independent resolved URLs, statuses, byte lengths, and hashes.

`verify_raw_provenance` now fails closed for an orphan `.bin`, orphan `.json`,
malformed metadata, record-key mismatch, or hash/length mismatch. Existing
complete locked objects are still reprocessed without refetch; duplicate lock
attempts remain no-overwrite implementation failures.

This does not change HIGH_4: the production urllib fetcher still reads the
body before looking at final redirect information. It does not implement
F2-F7 acquisition/parser integration or alter
`ACQUISITION_IMPLEMENTATION_COMPLETE=false`.

## Exact GPT review preceding this remediation

```text
REVIEWED_SHA=c603d1b0e86b85d98ae79d5969f83c9bf99113c4
PARENT_SHA=2584bcf925e710f98c52e08c9de23e8886d2f189
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
```
