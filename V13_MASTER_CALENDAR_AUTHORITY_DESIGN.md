# V13 Master-Calendar Authority Design

```text
study_identity=V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON
task=V13_MASTER_CALENDAR_AUTHORITY_DESIGN
github_issue=86
predecessor_issue=85
predecessor_gpt_result=PASS
predecessor_reviewed_sha=45bd13adc42cefcbbad3be13aa1379cb97a779c7
design_status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
scope=DESIGN_STATE_ONLY
```

## 1. Purpose and scope

This document closes the remaining V13 master-calendar authority gap by
freezing the *source/provenance* identity for the one shared, ordered
`MASTER_CALENDAR` that the frozen V13 methodology already requires. It does
not change V13 economic or scientific methodology and does not authorize
generating the real V13 calendar, inspecting live calendar output, reading
private data, selecting the real universe, or acquiring JPX/Yahoo data.

`V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md` and
`V13_PRE_IMPLEMENTATION_DETERMINISM_AMENDMENT_DRAFT.md` already freeze one
shared ordered exchange-session calendar, position-based `t+1`/`t+2`/`t+3`
offsets from a signal date, and calendar-year PnL boundaries. Neither
document freezes which provider/source produces that calendar. This task
resolves only that deferred provenance decision, using the already
GPT-reviewed, human-frozen, production-proven V10A calendar source
authority, with only the coverage window changed to the V13-required
window.

## 2. Reused V10A source authority (mechanically bound)

The V13 master-calendar source authority is frozen to the exact V10A
release-artifact identity recorded in
`V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR_DESIGN_DRAFT.md`
Section 4. These constants are copied, not re-derived:

```text
V13_MASTER_CALENDAR_SOURCE_IDENTITY=PANDAS_MARKET_CALENDARS_JPX_5_4_0_RELEASE_ARTIFACT
CALENDAR_NAME=JPX
PANDAS_MARKET_CALENDARS_VERSION=5.4.0
EXCHANGE_CALENDARS_VERSION=4.13.2
RELEASE_TAG=v5.4.0
RELEASE_TAG_COMMIT=275890784073a3a3a347e4f05f4dc986456e6a75
JPX_SOURCE_FILE=pandas_market_calendars/calendars/jpx.py
JPX_SOURCE_BLOB=a7a59b6cf910e325c85fc042459ff57ca8f70613
HOLIDAY_SOURCE_FILE=pandas_market_calendars/holidays/jp.py
HOLIDAY_SOURCE_BLOB=4c34214d06862e02ac22e946757463f748074fde
OFFICIAL_PYPI_WHEEL=pandas_market_calendars-5.4.0-py3-none-any.whl
OFFICIAL_PYPI_WHEEL_SHA256=bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3
```

These match, byte-for-byte, the frozen V10A source-identity block. No
constant above has been altered, re-derived, or independently recomputed by
this task. The three-part source-identity binding method (wheel SHA-256,
exact-uniqueness ZIP central-directory entry check, raw wheel-entry-bytes
Git blob SHA-1 equality) defined in that document's Section 4 is inherited
unchanged as the required V13 validation method whenever V13 later performs
the real environment/source validation; it is not re-executed by this
design-only task.

## 3. V13-specific calendar contract

```text
MASTER_CALENDAR_START=2015-01-01
MASTER_CALENDAR_END=2025-12-31
MASTER_CALENDAR_SOURCE=PANDAS_MARKET_CALENDARS_JPX_5_4_0_RELEASE_ARTIFACT
MASTER_CALENDAR_SESSION_SEMANTICS=ordered JPX schedule session labels
MASTER_CALENDAR_SHARED_ACROSS_INSTRUMENTS=true
MASTER_CALENDAR_OFFSET_RULE=positions in this exact ordered calendar
ANCHOR_2020_10_01=INELIGIBLE
ANCHOR_2020_10_02=ELIGIBLE
NO_FALLBACK_PROVIDER=true
NO_PRICE_DERIVED_SESSION_UNION=true
NO_JQUANTS_CALENDAR_SUBSTITUTION=true
NO_JPX_MONTHLY_SCRAPE_SUBSTITUTION=true
```

`MASTER_CALENDAR_START`/`END` match the coverage already frozen in
`V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md` and
`src/v13_public_data_lock.py` (`START = date(2015, 1, 1)`,
`END = date(2025, 12, 31)`). This window is **not** the V10A promoted
coverage window (`2017-01-01..2026-01-31`); the two windows are disjoint in
purpose and must not be conflated (see Section 4).

`t+k` offsets remain defined purely as positions in this exact ordered
calendar, exactly as `src/v13_public_data_lock.py::session_offset` already
implements (`sessions.index(signal) + offset`, with no calendar-adjacent
substitution). This design introduces no new offset semantics.

## 4. Why the promoted V10A artifact is evidence only, not the V13 artifact

`V10A_CANONICAL_CALENDAR.json` (`schema_version=V10A_CANONICAL_CALENDAR_V1`,
`coverage_start=2017-01-01`, `coverage_end=2026-01-31`,
`trading_date_count=2217`,
`canonical_calendar_sha256=2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0`)
and `V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json`
(`schema_version=V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT_V1`, `status=PASS`)
are V10A predecessor-study evidence for the source authority only. Their
coverage window does not include all of the V13 window (V13 starts
2015-01-01, two years before V10A's earliest date), so they cannot be
sliced, extended, or reused as the V13 calendar artifact.

```text
V10A_ARTIFACT_REUSED_AS_V13_ARTIFACT=false
V13_DATE_WINDOW_CHANGED=false
```

V13 must later generate a fresh calendar artifact over exactly
`2015-01-01..2025-12-31` from the same frozen source authority above, under
a new V13-specific artifact/output identity, once a separately reviewed and
human-approved bounded calendar-generation authority is granted. This
design freezes only the source identity and the target contract; it does
not perform or authorize that generation.

## 5. Bridge from the V10A schedule result to the existing V13 calendar parser

`src/v13_public_data_lock.py::parse_calendar` is the existing frozen V13
calendar-lock interface and is retained unchanged as the downstream
consumer:

```python
def parse_calendar(lock: RawLock) -> tuple[tuple[date, ...], dict[str, Any]]:
    """Offline ordered exchange-session list, one ISO date per line."""
    lines = lock.raw.decode("utf-8-sig").splitlines()
    if any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", line) is None for line in lines):
        raise ValueError("CALENDAR_FORMAT_MISMATCH")
    sessions = tuple(date.fromisoformat(line) for line in lines)
    if not sessions or any(not START <= d <= END for d in sessions) or tuple(sorted(set(sessions))) != sessions:
        raise ValueError("CALENDAR_ORDER_OR_RANGE_MISMATCH")
    ...
```

`parse_calendar` already fixes the payload shape it accepts: UTF-8 (BOM
tolerated on read but not required), one ISO-8601 (`YYYY-MM-DD`) date per
line via `str.splitlines()`, every date within `START..END`
(`2015-01-01..2025-12-31`), strictly ascending, and no duplicates
(`tuple(sorted(set(sessions))) != sessions` fails closed on any disorder or
duplicate). This design does not invent a competing format; it defines the
exact bridge that produces bytes satisfying that parser from a
`pandas_market_calendars` JPX `5.4.0` schedule result:

```text
V13_CALENDAR_LOCK_PAYLOAD=ordered exchange-session dates only
DATE_FORMAT=YYYY-MM-DD
ENCODING=UTF-8
BOM=false
LINE_ENDING=LF
ORDER=strict ascending
DUPLICATES=forbidden
FINAL_NEWLINE=true
CALENDAR_SHA256=SHA256(exact payload bytes)
```

Bridge steps (future bounded-generation authority only; not executed here):

1. Obtain the JPX `5.4.0` schedule result for `2015-01-01..2025-12-31`
   through the exact reviewed source identity in Section 2, using emitted
   session labels (the same `market_close`/emitted-session semantics
   already frozen for V10A) with no fallback provider.
2. Extract the ordered session dates and format each as `date.isoformat()`
   (`YYYY-MM-DD`), one per line, joined with `\n`, with a single trailing
   `\n` and no leading/trailing blank lines, no BOM, and no CRLF.
3. Encode as UTF-8. This byte sequence is the `RawLock.raw` payload passed
   to `parse_calendar`.
4. `CALENDAR_SHA256` is `RawLock.sha256`
   (`hashlib.sha256(raw).hexdigest()` per `src/v13_public_data_lock.py::digest`),
   computed over those exact payload bytes before parsing.
5. By construction, `parse_calendar`'s own derived
   `session_sha256` (`code_hash(d.isoformat() for d in sessions)` from
   `RawLock`; see `src/v13_public_data_lock.py::code_hash`, which joins with
   `"\n"` plus a trailing `"\n"`) must equal `CALENDAR_SHA256` from step 4.
   If they differ, the payload violated the LF-only/no-BOM/no-duplicate
   contract above and must fail closed before use — never be silently
   accepted.

No new parser, schema, or byte-level representation is introduced.
`parse_calendar`'s existing validation (format regex, range check, strict
ascending/no-duplicate check) remains the sole acceptance gate for the
downstream V13 interface.

## 6. Non-goals and non-claims

This task does not reopen and does not change:

- universe seed, size, or exclusion set;
- features, models, targets, costs, ranking, or portfolio construction;
- execution semantics or date windows already frozen for V13;
- the V13 A–Q feasibility criteria.

```text
V13_METHODOLOGY_CHANGED=false
SEED_CHANGED=false
UNIVERSE_SIZE_CHANGED=false
EXCLUSION_SET_CHANGED=false
PROVIDER_SUBSTITUTION=false
```

V13 already froze (unchanged by this task): one shared actual JPX
exchange-session master calendar; coverage through
`2015-01-01..2025-12-31`; position-based `t+k` offsets; no 2026 price
history; Yahoo price source and JPX current metadata source. This task only
resolves the previously deferred calendar source/provenance authority using
an already-reviewed source identity.

This design does not authorize real calendar generation, calendar object
creation, session inspection, package installation, environment mutation,
any network or private/sealed read, universe selection, public data
acquisition, market-data locking, model fitting, backtesting, forward
paper trading, or real trading.

```text
CALENDAR_OBJECT_CREATION=false
REAL_CALENDAR_GENERATION=false
REAL_CALENDAR_SESSION_INSPECTION=false
REAL_V13_UNIVERSE_SELECTED=false
V13_SELECTED500_PUBLIC_ACQUISITION_EXECUTED=false
REAL_MARKET_DATA_LOCKED=false
MODEL_FITS=0
BACKTESTS=0
FORWARD_PAPER_RUNS=0
REAL_TRADING=0
NETWORK_REQUESTS=0
PRIVATE_READS=0
```

## 7. Historical evidence bound (V10A calendar-feasibility chain)

The V10A calendar-feasibility chain is complete `PASS` evidence for the
reused source identity, verified against current repository chronology:

```text
V10A_CALENDAR_FEASIBILITY_IMPLEMENTATION_REVIEWED_SHA=0830d86675f447231e77b1687c7a23cf0b135d7f
V10A_CALENDAR_FEASIBILITY_IMPLEMENTATION_REVIEW_RESULT=PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0
V10A_CALENDAR_FEASIBILITY_PROMOTION_REVIEWED_SHA=8b9878030c08d1f5638ce80d3426903f3f6b54a4
V10A_CALENDAR_FEASIBILITY_PROMOTION_REVIEW_RESULT=PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0
V10A_CALENDAR_FEASIBILITY_STAGE=COMPLETE_PASS
V10A_CALENDAR_FEASIBILITY_RESULT=PASS
V10A_TRADING_DATE_COUNT_2017_TO_2026_01=2217
V10A_ANCHOR_2020_10_01=INELIGIBLE
V10A_ANCHOR_2020_10_02=ELIGIBLE
V10A_CANONICAL_CALENDAR_DIGEST=2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0
```

Both reviewed commits exist in repository history exactly as claimed
(`0830d86675f447231e77b1687c7a23cf0b135d7f` — "Harden V10A feasibility
anchor states"; `8b9878030c08d1f5638ce80d3426903f3f6b54a4` — "Promote V10A
calendar feasibility artifacts"), and `PROJECT_STATE.md` records
`V10A_CALENDAR_FEASIBILITY_STAGE=COMPLETE_PASS`.

The V10A date count (`2217`, over `2017-01-01..2026-01-31`) and digest
(`2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0`) are
bound here strictly as historical predecessor evidence for the source
identity. They are explicitly **not** the expected V13 date count or
digest, because the coverage windows differ:

```text
V13_EXPECTED_TRADING_DATE_COUNT=UNKNOWN_UNTIL_GENERATED
V13_EXPECTED_CANONICAL_CALENDAR_DIGEST=UNKNOWN_UNTIL_GENERATED
```

## 8. Governance and next action

```text
V13_MASTER_CALENDAR_AUTHORITY_DESIGN_STATUS=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
V13_MASTER_CALENDAR_AUTHORITY_DESIGN_FROZEN=false
V13_MASTER_CALENDAR_AUTHORITY_HUMAN_APPROVED=false
V13_CALENDAR_REAL_ACQUISITION_SCOPE_EXTENSION_REQUIRED=true
V13_FIRST_AQ_EXECUTION_AUTHORIZED=false
future_profitability_established=false
```

`V13_CALENDAR_REAL_ACQUISITION_SCOPE_EXTENSION_REQUIRED` remains `true`
until this design is separately GPT-reviewed, human-approved, implemented,
reviewed, and a later bounded calendar-generation authority is explicitly
granted. No calendar object was created, no session was generated or
inspected, and no network or private/sealed operation occurred while
producing this document. Next action is
`GPT_EXACT_SHA_INDEPENDENT_REVIEW` of this design at the commit that
introduces it.
