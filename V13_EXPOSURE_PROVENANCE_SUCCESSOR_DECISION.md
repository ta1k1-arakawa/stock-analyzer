# V13 Exposure-Provenance Successor Decision

```text
DECISION_STATUS=FROZEN_AWAITING_GPT_EXACT_SHA_REVIEW
SOURCE_RECOVERY_STRATEGY=JQUANTS_STANDARD_SEMANTIC_AND_BLOCK_IDENTITY
SOURCE_RECOVERY_REVIEWED_SHA=7565ca723c76801d74d8d319d65d280a689b3cfa
SOURCE_RECOVERY_RESULT_RECORD_SHA=0c961927b0f61fd1fdf23166a795d85acaa05b3e
T1_COUNT=300
T1_SHA256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
KNOWN_DEFINITELY_ACQUIRED_PREFIX_COUNT=297
LATER_ATTEMPT_EXACT_ACQUIRED_COUNT=UNKNOWN_NOT_PERSISTED
EXCLUSION_DISPOSITION=EXCLUDE_FULL_RECOVERED_T1_BLOCK
PRIVATE_T1_IDENTITIES_COMMITTED=false
PRIVATE_T1_IDENTITIES_READ_IN_THIS_TASK=false
V13_UNIVERSE_SELECTED=false
```

## Decision and evidence boundary

Issue #71 received an exact-SHA GPT review PASS with zero critical, high,
medium, and low findings at `0c961927b0f61fd1fdf23166a795d85acaa05b3e`.
The reviewed J-Quants recovery restored the original V8 eligible-universe
identity and every partition block identity under
`SEMANTIC_UNIVERSE_PLUS_ALL_BLOCK_HASHES_EXACT`. The original T1 block has
300 identities and the pinned hash above. Its membership exists only in the
private recovered artifact; this public decision neither reads nor lists it.

The historical V8 T1 attempt #1 definitely staged successful historical-price
payloads for the first 297 ordered T1 identities before request 298 failed.
The later attempt's exact acquired count was not durably persisted. No
assumption is made about which, if any, additional T1 identities were fetched
in that attempt.

For V13 exposure provenance, exclude **all 300 identities in the recovered
original T1 block**. This conservative pre-outcome leakage-control rule
ensures that an identity possibly exposed to historical prices cannot be
retained merely because the later attempt's acquired count is unknown. It
supersedes only the deferred `297 vs 300` exclusion disposition; it does not
alter the historical acquisition facts or the frozen V13 base design.

`V13_PRE_ACQUISITION_EXPOSURE_AUDIT.md` remains truthful: at its original
public committed evidence boundary, T1 identities were private/uncommitted and
the audit's exclusion verdict was blocked. This successor freezes a private
exclusion source and treatment for a later protected step. It does not close
that historical audit from public committed evidence, apply the private
exclusion, or construct or select the V13 500.

## Next boundary

After GPT exact-SHA PASS on this decision, a separately reviewed protected
implementation may consume only the private recovered T1 membership, verify
its count and pinned hash, and apply the full-block exclusion during V13
universe-provenance construction. That step must preserve private identities
and its own authorization and review boundaries. This decision grants no
private-content read, market-data request, historical-price read, model fit,
backtest, forward paper, or trading authority. Future profitability remains
unestablished.
