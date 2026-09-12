# V10A Calendar Authority Release-Artifact Successor Design Draft

```text
study_identity=V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR
design_status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
design_frozen=false
human_freeze_approved=false
predecessor_study=V10_CALENDAR_AUTHORITY_SUCCESSOR
predecessor_terminal_reason=FROZEN_DESIGN_SOURCE_IDENTITY_INCONSISTENCY
```

## 1. Scope and study identity

V10A is a new successor study, not an amendment to V10. V10 terminates at
the canonical-environment-extension Step 5 source-identity failure because
its frozen `jpx.py` source blob is inconsistent with the reviewed
`pandas-market-calendars==5.4.0` release artifact. V10A corrects that source
authority under a new study identity while preserving the V10 scientific
methodology stated below.

This draft is design-only and awaits GPT exact-SHA review. It does not freeze
V10A, authorize environment mutation, authorize calendar generation, or
authorize historical evaluation.

## 2. Terminal V10 adjudication

The durable V10 terminal record is
`V10_CALENDAR_SOURCE_IDENTITY_TERMINAL_ADJUDICATION.json`.

The adjudicated facts are:

- V10 Step-5 attempt 1 terminated with `PROVENANCE_BINDING_FAILURE`.
- V10 Step-5 attempt 2 terminated with `JPX_SOURCE_BLOB_MISMATCH`.
- The installed bytes matched the reviewed official wheel. The official
  wheel's `jpx.py` matched the `v5.4.0` release tag and did not match the
  V10-frozen source blob; this establishes source-identity inconsistency,
  not installation corruption or a wrong wheel.
- `jp.py` matched the frozen V10 source identity, the release tag, and the
  official wheel.
- Calendar generation never started; no T0, historical evaluation, or
  profitability conclusion exists for V10.

The old V10 source blob is historical failed-study provenance only. It is not
V10A execution authority and must not be silently substituted into V10A.

## 3. Inheritance boundary

V10A inherits the reviewed V10 scientific methodology unchanged except for
the calendar source identity correction in Section 4. In particular, V10A
preserves:

- all evaluation periods and partitions;
- labels, targets, economics, costs, and slippage;
- models, thresholds, stopping rules, and feasibility criteria;
- coverage `2017-01-01..2026-01-31`;
- `calendar_name=JPX`;
- `pandas-market-calendars==5.4.0` and `exchange-calendars==4.13.2`;
- the anchors `2020-10-01=INELIGIBLE` and `2020-10-02=ELIGIBLE`;
- emitted-session and `market_close` semantics;
- no fallback and no alternate provider;
- the single feasibility execution budget; and
- no historical evaluation or T0 authority at this design stage.

No V10A decision may be selected from an evaluation outcome. No V10
methodology parameter, period, threshold, cost, slippage, model, target, or
partition is reopened by this source correction.

## 4. Corrected calendar source authority

The V10A calendar source identity is:

```text
calendar_source_identity=PANDAS_MARKET_CALENDARS_JPX_5_4_0_RELEASE_ARTIFACT
release_tag=v5.4.0
release_tag_commit=275890784073a3a3a347e4f05f4dc986456e6a75
calendar_source_file=pandas_market_calendars/calendars/jpx.py
calendar_source_blob=a7a59b6cf910e325c85fc042459ff57ca8f70613
holiday_source_file=pandas_market_calendars/holidays/jp.py
holiday_source_blob=4c34214d06862e02ac22e946757463f748074fde
official_pypi_wheel_filename=pandas_market_calendars-5.4.0-py3-none-any.whl
official_pypi_wheel_sha256=bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3
```

The source blob algorithm is the Git blob SHA-1 over the exact source bytes:
`sha1(b"blob " + decimal_length + b"\0" + raw_bytes)`. V10A must bind the
release-tag source identities and the reviewed official wheel bytes exactly.

V10A live validation must establish both conjunctive conditions:

1. installed package bytes equal the reviewed official wheel bytes; and
2. the wheel's `jpx.py` and `jp.py` source blobs equal the `v5.4.0`
   release-tag source blobs above.

The old V10 `jpx.py` blob `0c2041b1300d1dbbd505202b00ac0ada38c712e1` is never
accepted as a V10A source match.

## 5. Required post-freeze operational sequence

After V10A design freeze and all required reviews/authorities, the first
operational environment step is a fresh no-network validation under a new
V10A artifact/output identity. It must validate the already-existing
canonical machine environment as pre-existing software state against the
V10A source authority above.

That validation must fail closed on any package-set, platform, package
version, source-blob, parser-probe, provenance, or output-safety mismatch.
There is no automatic repair, reinstall, replacement, fallback provider,
calendar generation, or reuse of a failed V10 attempt. A fresh separately
reviewed operational design and point-of-use authority are required before
any mutation or protected execution.

The existing environment state may be reused only as state. V10 human
authorization, one-shot gates, failed Step-5 attempts, and any V10 calendar
feasibility budget are not reusable authority. A V10A validation PASS would
be evidence for V10A only and would not itself freeze the environment or
authorize T0.

## 6. Calendar and parser boundaries

V10A must use only `JPX` through the reviewed package/source identity. It
must preserve the emitted-session and `market_close` semantics and must not
introduce an alternate provider or fallback. Source validation is performed
from exact package bytes and source metadata.

Calendar object creation, session generation, and date inspection remain
prohibited during this design task. They become operationally relevant only
after V10A design freeze, the required implementation and exact-SHA reviews,
and the separately authorized V10A validation sequence.

The XLS/PDF synthetic readiness probes, if reused by later reviewed tooling,
must remain wholly synthetic and must not read private or research inputs.

## 7. Governance and non-claims

```text
V10A_DESIGN_FROZEN=false
V10A_EXECUTION_AUTHORIZED=false
V10A_HUMAN_FREEZE_APPROVED=false
V10A_HISTORICAL_EVALUATION_AUTHORIZED=false
V10A_T0_AUTHORIZED=false
V10A_CALENDAR_GENERATION_AUTHORIZED=false
```

This draft does not claim that the canonical environment is V10A-ready or
frozen, does not consume V10 authority, and does not authorize a new human
gate. V10A future profitability is unestablished. No strategy or
profitability conclusion follows from the V10 terminal source-identity
finding.

## 8. Review and promotion requirements

Before V10A can proceed, GPT exact-SHA review must approve this design and
the implementation that binds the corrected release artifact. Any human
freeze approval must be explicit and scoped to V10A design freeze. The later
operational sequence must separately bind the reviewed V10A artifacts,
validate the pre-existing environment with no network, and preserve a
one-shot/no-retry boundary for any future mutation.

No V10A package installation, calendar generation, T0, historical
evaluation, model fitting, backtest, broker operation, private/sealed read,
or profitability claim is authorized by this document.
