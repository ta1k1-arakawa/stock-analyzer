# V10A Calendar Authority Release-Artifact Successor Design Draft

```text
study_identity=V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR
design_status=FROZEN_AFTER_GPT_PASS_AND_HUMAN_APPROVAL
approved_design_sha=b14cc5510685210e928000af0815e188bc1aadc0
gpt_final_design_review_sha=b14cc5510685210e928000af0815e188bc1aadc0
gpt_final_design_review_result=PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0
design_frozen=true
human_freeze_approved=true
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

The design content approved by GPT and the human is exactly the content at
`b14cc5510685210e928000af0815e188bc1aadc0`. V10A design freeze is recorded
after the exact-SHA GPT PASS and explicit human approval. This freeze does not
authorize environment mutation, calendar generation, T0, or historical
evaluation.

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
release-tag source identities and the reviewed official wheel archive exactly.

V10A does not require equality between the complete wheel ZIP archive and the
complete installed distribution tree:

```text
FULL_INSTALLED_DISTRIBUTION_TREE_BYTE_EQUALITY_REQUIRED=false
```

Instead, V10A source identity is closed by three independent bindings:

1. The preserved wheel archive has the exact filename and SHA-256 above.
2. Before reading either scientific source entry, enumerate the ZIP
   central-directory entry names deterministically. For each exact,
   case-sensitive POSIX name below, `EXACT_ENTRY_OCCURRENCE_COUNT=1` is
   mandatory:

   ```text
   pandas_market_calendars/calendars/jpx.py
   pandas_market_calendars/holidays/jp.py
   ```

   An occurrence count of zero or greater than one fails closed. There is no
   first-entry or last-entry selection. Entry selection performs no case
   folding, slash/backslash normalization, URL decoding, Unicode
   normalization, basename matching, suffix matching, or archive extraction.
   If ZIP parsing cannot enumerate the central directory deterministically,
   validation fails closed.
3. Only after exact uniqueness is established, for each of exactly two
   scientific source files, raw wheel-entry bytes
   equal raw installed-file bytes without decoding, whitespace normalization,
   newline normalization, AST comparison, or import-based reconstruction:

   ```text
   wheel_entry=pandas_market_calendars/calendars/jpx.py
   installed_relative_path=pandas_market_calendars/calendars/jpx.py
   wheel_entry=pandas_market_calendars/holidays/jp.py
   installed_relative_path=pandas_market_calendars/holidays/jp.py
   ```

4. The Git blob SHA-1 of each exact wheel-entry byte sequence equals the
   corresponding `v5.4.0` release-tag blob above. The installed-file Git
   blob may also be computed and must equal that same expected value as a
   redundant fail-closed cross-check.

Therefore V10A live validation must establish all of the following:

```text
official_wheel_sha256_match=true
installed_jpx_equals_wheel_entry=true
wheel_jpx_git_blob_sha1=a7a59b6cf910e325c85fc042459ff57ca8f70613
installed_jp_equals_wheel_entry=true
wheel_jp_git_blob_sha1=4c34214d06862e02ac22e946757463f748074fde
```

These source-entry checks prove source identity only. They do not substitute
for the complete reviewed installed package-set/count, package versions,
Python/platform requirements, provenance, or synthetic XLS/PDF readiness
probes, all of which remain separate V10A requirements.

The exact-entry uniqueness and raw-byte checks apply only to the two
scientific source files. No alternate archive entry, transformed extraction,
or reconstructed source is an accepted authority.

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
V10A_DESIGN_FROZEN=true
V10A_EXECUTION_AUTHORIZED=false
V10A_HUMAN_FREEZE_APPROVED=true
V10A_APPROVED_DESIGN_SHA=b14cc5510685210e928000af0815e188bc1aadc0
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

The GPT exact-SHA review and explicit human freeze approval recorded above are
bound to the approved design SHA. Any later implementation and operational
sequence must be separately reviewed and authorized, must bind the reviewed
V10A artifacts, validate the pre-existing environment with no network, and
preserve a one-shot/no-retry boundary for any future mutation.

No V10A package installation, calendar generation, T0, historical
evaluation, model fitting, backtest, broker operation, private/sealed read,
or profitability claim is authorized by this document.
