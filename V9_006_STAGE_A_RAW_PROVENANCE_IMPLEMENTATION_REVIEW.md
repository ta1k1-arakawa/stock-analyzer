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

This review now also records the separately scoped HIGH_4 redirect-boundary
remediation below. It does not implement F2-F7 acquisition/parser integration
or alter `ACQUISITION_IMPLEMENTATION_COMPLETE=false`.

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

## HIGH_3 residual MEDIUM review and remediation

```text
REVIEWED_SHA=c77eff15ecd5b6a250ebcb960214cc99dd8950a2
PARENT_SHA=c603d1b0e86b85d98ae79d5969f83c9bf99113c4
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=1
RESULT=BLOCK
FINDING=V9_006_HIGH_3_MEDIUM_1_RAW_PROVENANCE_VALIDATION_NOT_TOTAL
```

This revision remediates exactly that MEDIUM. `lock_first_complete_payload`
now accepts the immutable `FetchResult` as its only payload/resolved-URL/
HTTP-status source; metadata derives those fields directly from that one
object. No separate production-capable lock API accepts arbitrary status or
payload provenance.

Retrieval timestamps must now be exact canonical UTC text:
`YYYY-MM-DDTHH:MM:SSZ`. The same strict parse-and-round-trip validation is
applied before writing, while reading, and when verifying stored provenance.
Malformed persisted timestamps fail closed. F1 discovery/XLS locking,
orphan/hash/length/duplicate/reuse checks, HIGH_4 ordering, the deferred LOW,
and `ACQUISITION_IMPLEMENTATION_COMPLETE=false` are unchanged.

`V9_006_HIGH_3=REMEDIATION_REVISED_AWAITING_GPT_REVIEW`.

## HIGH_4 exact-SHA review and remediation

```text
REVIEWED_SHA=aca54748a1d838cbd3c4ad603fc91bb6624d7ae2
PARENT_SHA=c77eff15ecd5b6a250ebcb960214cc99dd8950a2
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_HIGH_3=RESOLVED
```

The production urllib boundary now installs a redirect handler which validates
every redirect target as HTTPS on `jpx.co.jp` or a subdomain before urllib can
follow it. The final response URL is independently validated before its body
is read. Unsafe redirect targets therefore remain nonretryable
`OFF_DOMAIN_REDIRECT_REJECTED` failures with no redirected-body consumption.
Same-domain JPX redirects remain permitted. FetchResult coupling, strict
timestamp validation, F1 discovery-before-extraction locking, and truthful
XLS request provenance remain unchanged.

`V9_006_HIGH_4=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`.
