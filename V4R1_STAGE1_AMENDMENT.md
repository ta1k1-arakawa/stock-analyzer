# V4R1 Stage 1 Subdivision Amendment — Pre-registered

## Authority and immutable context

This HUMAN_GATE amendment subdivides engineering work only.  The parent plan
commit is `014cc2862bda6299c25661d83485e23b14c313be`; its Stage 1 outcome is
`V4R1_STAGE1_BLOCKED`, with zero Stage 1 implementation commits, real Yahoo
calls, price acquisitions, model fits, and real-data backtests.  Research
results remain unjudged.  The frozen `v4r1-stage1-data-cache` branch/worktree
is neither changed, deleted, nor reused.  The parent plan is neither amended
nor overwritten.  This is not a post-result adjustment.

Scientific conditions do not change: the hypothesis; fixed 300-ticker universe;
universe and ticker-list hashes; price period `2015-01-01`–`2019-12-31`; signal
period `2016-04-01`–`2019-12-31`; 2020 prohibition; 15 features; LightGBM
parameters; threshold `0.55`; three folds; execution rules; 10 BLOCKED and 17
acceptance conditions; and `SURVIVORSHIP_BIASED_RESEARCH_ONLY`.  Real orders,
shadow, deploy, and schedule remain prohibited.  The immutable scientific
design is `V4_META_LABEL_DESIGN.md`, SHA-256
`07039948aa7a1180d506b3089a0bd5612dda24559968c510e0cb92935b48055a`.
Stages 2–4 and the formal-execution HUMAN_GATE are unchanged.

## Common substage controls

Each substage uses a new branch/worktree based on the immediately preceding
PASS review commit.  Only `src/v4_meta_label.py` and
`tests/test_v4_meta_label.py` may change.  Each has at most one implementation
commit and one bug-fix retry, followed by a separate read-only independent
review.  No next substage begins before PASS; BLOCKED never auto-corrects.
Amend, rebase, force-push, merge, tag, real communication, and all work beyond
the named substage are prohibited.  Only temporary directories, fake transport,
and synthetic JSON are permitted.

## Stage 1A — URL_AND_TRANSPORT

Scope: fixed Yahoo constants; URL construction; ticker validation; exact HTTPS,
host, path, and query validation; port/userinfo/fragment rejection; redirect
rejection; transport interface and fake transport; maximum-three retry policy;
retry/non-retry classification; per-attempt audit; and distinct safety/BLOCKED
exceptions.

Forbidden: payload price parsing, splits, cache writes, manifests, offline
cache, real communication, and feature-or-later processing.

At least 12 direct tests cover normal URL, `.T`, invalid ticker, HTTP, foreign
and spoofed host, port, userinfo, missing/extra/duplicate query, redirect,
retry success and limit, 429/5xx versus ordinary 4xx, and fixed timeout with
redirects disabled.  PASS review is the only gate to Stage 1B.

## Stage 1B — PAYLOAD_PARSER

Begins only after Stage 1A PASS.  Scope: pure Yahoo chart JSON parsing,
chart.error/result cardinality/timestamp/OHLCV/Adjusted Close validation,
Asia/Tokyo dates, splits, array lengths, order/duplicates, period boundaries,
null/NaN/Infinity, OHLC relation, timezone consistency, and raw payload
SHA-256.

Forbidden: transport changes, real communication, cache writes, manifests, and
feature-or-later processing.

At least 14 direct tests cover normal payload, splits, chart error, missing or
multiple result, missing timestamp/quote/adjclose, length mismatch, null,
nonfinite values, nonpositive price, invalid volume/OHLC, duplicate dates,
pre-2015, 2020-or-later, timezone inconsistency, and unchanged payload bytes.
PASS review is the only gate to Stage 1C.

## Stage 1C — IMMUTABLE_CACHE_AND_AUDIT

Begins only after Stage 1B PASS.  Scope: repository-external cache validation,
payload storage, SHA sidecars, atomic rename, identical-byte reuse, different
byte rejection, cache manifest/stable JSON, offline validation, parser
revalidation, ticker ordering, aggregated network audit, fixed clock, and
failure records.

Forbidden: real communication/prices, feature-or-later processing, eight
research artifacts, and CLI changes.

At least 14 direct tests cover new save, identical reuse, different bytes,
sidecar/payload/manifest tamper, duplicate ticker, order, offline missing and
success with zero calls, repository cache rejection, atomic cleanup, fixed-clock
determinism, manifest final newline, no raw payload stdout, and parser failure
without save.  PASS review completes the original Stage 1 only.

## Budget and continuation gate

The original Stage 1 initial attempt is consumed; its retry is not reused.
This amendment registers one implementation commit and one bug-fix retry for
each of 1A, 1B, and 1C.  This is engineering-budget subdivision before price
acquisition, not additional scientific experiment budget.  After all three
independent PASS reviews, Stage 2 still requires a separate HUMAN_GATE and does
not start automatically.
