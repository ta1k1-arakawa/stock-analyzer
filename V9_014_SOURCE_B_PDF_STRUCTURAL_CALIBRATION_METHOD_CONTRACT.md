# V9_014 SOURCE_B PDF Structural Calibration Method Contract

STUDY_ID: V9_014_JPX_MONTHLY_AUCTION_ACTIVITY_AUTHORITY_SUCCESSOR
STATUS: DRAFT_AWAITING_GPT_REVIEW
FROZEN_DESIGN_GIT_SHA: efee3d0efca368645c00aeed63cb8e0637cd3672
FROZEN_DESIGN_BLOB_SHA: 2bbacbf37ab961d1cbf416b7fd476db18778c5b7

This document is a **subordinate preregistered implementation-method
contract**. It governs how a future, separately reviewed implementation
resolves raw JPX PDF structure and the still-deferred raw unit-cell
normalization mechanics (frozen design LOW_1). It does not amend, override,
supersede, or reinterpret any clause of the frozen design draft above; it
freezes execution-level decisions that the design left to a later
checkpoint.

This contract itself performs and authorizes **NO PDF acquisition, NO PDF
read, NO package installation, NO code implementation, and NO
scientific-methodology change**. Nothing below grants execution authority;
see Section 6.

---

## 1. PDF engine authority

All future PDF structural calibration and parser implementation work for
V9_014 SOURCE_B must use exactly:

```
pdfplumber==0.11.10
```

- No `pypdf`, `PyMuPDF`, OCR, `pdftotext`, or any alternate parser is
  permitted as a fallback or substitute.
- No "parser shopping" after seeing extraction results is permitted: the
  engine choice is frozen now, before any real PDF has been read, and may
  not be revisited based on how well it appears to parse actual content.
- This contract does **not** modify `requirement.txt`,
  `requirements-real-execution.txt`, or `requirements-real-execution.lock.txt`.
  A separate, independently reviewed dependency/environment-lock task is
  required before any real PDF parse occurs; that task alone freezes the
  exact transitive dependency closure.

## 2. Fixed calibration physical object set

Exactly **8** physical SOURCE_B objects may inform PDF layout/parser
mechanics during calibration. No other physical object may be used for this
purpose.

| # | Logical month | Object part | Preregistered structural regime |
|---|---|---|---|
| 1 | 2017-01 | `NORMAL_MONTHLY_REPORT2_OBJECT` | Coverage start |
| 2 | 2019-12 | `NORMAL_MONTHLY_REPORT2_OBJECT` | Immediately before Mothers ToSTNeT 2019->2020 unit boundary |
| 3 | 2020-01 | `NORMAL_MONTHLY_REPORT2_OBJECT` | Immediately after Mothers ToSTNeT 2019->2020 unit boundary |
| 4 | 2022-03 | `NORMAL_MONTHLY_REPORT2_OBJECT` | Immediately before pre/post market-restructure boundary |
| 5 | 2022-04 | `PRE_APRIL_1_REFERENCE_OBJECT` | Mandatory April-2022 special reference object |
| 6 | 2022-04 | `NORMAL_MONTHLY_REPORT2_OBJECT` | Immediately after pre/post market-restructure boundary |
| 7 | 2022-05 | `NORMAL_MONTHLY_REPORT2_OBJECT` | First full normal month after the restructure boundary |
| 8 | 2026-01 | `NORMAL_MONTHLY_REPORT2_OBJECT` | Coverage end |

`CALIBRATION_OBJECT_COUNT = 8`

These reasons are the **entire and exclusive** justification for object
selection: coverage start/end, the Mothers ToSTNeT 2019->2020 unit
boundary, the pre/post market-restructure boundary, and the mandatory
April-2022 special object (both of its required object parts). No
favorable/unfavorable content selection or replacement is permitted at any
point, before or after inspecting any calibration object's content.

If any required calibration object cannot be resolved, acquired, or parsed,
the calibration effort must **STOP** under the applicable frozen failure
class (locator failure, acquisition failure, or implementation failure, as
applicable per the reviewed mechanics in force at that time). No other
month or object part may be substituted for a required calibration object
under any circumstance.

The remaining **102** physical SOURCE_B objects (`110 - 8 = 102`, per the
frozen `REQUIRED_PHYSICAL_SOURCE_B_OBJECT_COUNT = 110`) are
**conformance-only**. After the final parser mechanics are frozen (Section
7, Stage F), the conformance-only set must never be used to adapt, retune,
redraw, or select parsing rules. They exist solely to be processed against
already-frozen mechanics at Stage H.

`CONFORMANCE_ONLY_OBJECT_COUNT = 102`

## 3. Calibration role

Calibration may inspect **only** the raw PDF structural representation
needed to freeze:

- page count / order
- text / character geometry
- table / title / header / segment structure
- raw declared-unit cell representation
- line / rectangle / table geometry needed for deterministic extraction

Calibration **must not**:

- derive auction-active dates
- classify `NumericCell`/`DashCell` values into activity
- run `classify_date`
- run cross-source relation/sentinel evaluation
- materialize `trading_dates`
- run T0/backtest/model/profitability logic
- use numeric trading-volume outcomes to choose a more favorable extraction
  rule

Calibration is a structural-mechanics exercise only; it produces no
scientific classification of any kind.

## 4. Raw unit normalization governance

The frozen accepted semantic unit token set remains **exactly**:

```
"shs."
"thous.shs."
"株"
"千株"
```

This contract does not authorize any new semantic token or alias, and does
not itself change this set in any way.

Before calibration evidence exists, none of the following are permitted:

- NFKC/NFC-based acceptance broadening
- casefold/lowercase normalization
- punctuation substitution
- internal-whitespace deletion/collapse that turns a non-token into a
  token
- abbreviation expansion
- magnitude-based inference
- previous-month inheritance
- OCR-style repair
- regex/fuzzy alias matching

Calibration records the **raw representation only** — the literal bytes/text
as they structurally appear in the PDF, unmodified. After calibration
evidence exists, **GPT-5.6 Sol alone** decides the exact deterministic
representation-to-frozen-token mechanics; this decision is out of scope for
this contract and out of scope for Claude Code's independent judgment.

Any proposed transformation that would make an otherwise-unsupported raw
token become accepted is `CHATGPT_DECISION_REQUIRED` and cannot be adopted
unilaterally by Claude Code under any circumstance.

Frozen design **LOW_1 is NOT claimed resolved by this task**:
`V9_014_PDF_DESIGN_LOW_1_RESOLVED=false`.

## 5. No reparse-until-pass

After parser mechanics are later frozen (Section 7, Stage F):

- each locked PDF is processed deterministically exactly once from its
  preserved raw bytes
- software repair may only reprocess the **same** locked bytes when that
  reprocessing is itself separately reviewed and authorized
- no refetch, no redraw, no substitute-month, and no parser-switch is
  permitted merely to try to reach a passing result
- a conformance failure remains a `DQ` or implementation failure, as
  applicable under the mechanics in force, rather than a trigger to retry
  with different inputs or rules

## 6. Authority / execution boundary

This contract itself grants no execution authority:

```
NETWORK_AUTHORIZED=false
PDF_ACQUISITION_AUTHORIZED=false
PROTECTED_SOURCE_A_READ_AUTHORIZED=false
HUMAN_GATE_CONSUMED=false
```

A later real public SOURCE_B calibration acquisition (Section 7, Stage D)
requires the exact reviewed runner/environment and the point-of-use
authority required by repository governance (`AI_RESEARCH_EXECUTION_RULES.md`,
`AI_REAL_EXECUTION_RUNBOOK.md`) at the time that stage is actually executed.
No prior V9 authorization — from V9_005/V9_006 or any other prior study or
stage — may be reused or treated as still-standing authorization for this
work.

## 7. Successor checkpoints

The following order is frozen. No stage may be skipped or performed out of
order; each stage requires its own GPT-5.6 Sol PASS (where marked) before
the next stage may begin.

```
A. this method-contract GPT PASS
B. pdfplumber dependency/environment lock implementation + GPT PASS
C. offline calibration/probe runner implementation with synthetic fixtures + GPT PASS
D. authorized fixed-8-object public calibration acquisition/execution
E. no-network safe calibration inspection
F. GPT freezes exact PDF table settings + unit representation mechanics
G. production parser implementation + synthetic tests + GPT PASS
H. only then full 110-object acquisition/materialization path
```

---

## Non-claims

This document, and the act of drafting it, does not:

- resolve V9_014 design LOW_1
- claim any overall V9_014 implementation PASS
- claim resolution of `V9_009_HIGH_2`
- materialize `trading_dates`
- run or authorize T0, a backtest, a model, or any profitability claim
- perform any network request, PDF acquisition, PDF read, package
  installation, protected/private read, or API-key read
- consume any human gate

`V9_009_HIGH_2` remains `OPEN_REQUIRES_HISTORICAL_JPX_CALENDAR_BINDING`.
`T0_STATUS` remains `NOT_RUN`. `future_profitability_established` remains
`false`.

GPT-5.6 Sol remains the sole `FINAL_INDEPENDENT_REVIEWER` for this contract,
as for every other V9_014 artifact in this study.
