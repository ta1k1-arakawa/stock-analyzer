# V4R1 Stage 1A Remediation Amendment — Pre-registered

## Lineage and scope lock

The source Stage 1A implementation commit is
`5cc7ed53db4749788adf85fec3b564ec959925b4`.  Its implementation verdict was
PASS; its independent-review verdict was `V4R1_STAGE1A_REVIEW_BLOCKED`.
Source implementation-commit and retry budgets are both zero.  The source
branch/worktree is frozen and will not be changed, deleted, reset, amended,
rebased, or reused.

There were zero external communications, price acquisitions, cache creations,
model fits, and real-data backtests; research results remain unjudged.  This is
not a post-result change.  Hypothesis, data periods, universe, features, model,
threshold, folds, and execution conditions are unchanged.  No Stage 1B-or-later
scope is added.

## Sole remediation scope

Only the review findings below may be changed in a later implementation, and
only `src/v4_meta_label.py` and `tests/test_v4_meta_label.py` may change.

1. **Constant single source.** Replace duplicated scheme, host, timeout,
   maximum-attempt, and retry-delay literals so URL construction, transport
   call, audit, and retry loop all reference the same constants.
2. **Malformed port safety.** Convert all invalid port parsing, including
   `:abc`, into coded `V4SafetyError`; no raw `ValueError` may escape.
3. **Response validation.** Before attribute use, reject `None`, dicts,
   incomplete objects, bool/non-int status, non-bytes body, non-string
   final URL, and bool/negative/non-int redirect count as coded
   `V4DataBlockedError`.  Redirect/final-URL changes remain `V4SafetyError`.
4. **Audit consistency.** Set attempt outcome only after full response
   validation.  HTTP 200 with invalid body is never `SUCCESS`.  `retryable`
   means an actual subsequent retry will occur, so it is false on a final
   attempt.  Success, HTTP BLOCKED, exhausted retry, malformed response, and
   redirect safety records use consistent outcome values.
5. **Separated direct tests.** Provide individually diagnosable methods for
   HTTP, ordinary/spoof host, userinfo, port, malformed port, fragment, path,
   missing/extra/duplicate/reordered/changed query, all listed malformed
   responses, non-bytes audit, final-retry audit, full multi-attempt audit, and
   captured stdout without raw body.  Stage1A test-method total is at least 28.

Parser, splits, payload hash, cache, manifest, offline validation, CLI,
features, models, portfolio, metrics, and artifacts are prohibited.

## Budget and review gate

This HUMAN_GATE registers exactly one remediation implementation commit and
zero remediation retries.  A BLOCKED implementation or review registers no
additional budget and ends Stage 1A.  A separate read-only independent review
is mandatory.  PASS is `V4R1_STAGE1A_REMEDIATION_REVIEW_PASS`; BLOCKED is
`V4R1_STAGE1A_REMEDIATION_REVIEW_BLOCKED`.

PASS requires both CRITICAL findings, both HIGH findings, and literal
inconsistency resolved; all 28 Stage 1A items directly tested; pytest and
unittest passing; zero external communications; and zero Stage 1B-or-later
implementation.  Even PASS completes Stage 1A only: Stage 1B requires another
HUMAN_GATE and never begins automatically.
