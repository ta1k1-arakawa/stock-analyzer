# V9_016 SOURCE_B 2017 Month-Header Locator Recovery Successor Design Draft

```text
study_id=V9_016_SOURCE_B_2017_MONTH_HEADER_LOCATOR_RECOVERY_SUCCESSOR
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
design_status=DRAFT_REMEDIATED_AWAITING_GPT_REVIEW
automatic_retry=false
```

## 1. Study identity and scope

V9_016 is an explicit successor study created by the GPT methodology
decision recorded with the V9_015 terminal adjudication. It is not a retry of
V9_015 or V9_014. It does not authorize a root refetch, a year-page refetch,
arbitrary raw-HTML inspection, T0, model work, profitability work, or any
other research opening.

The purpose is to test one mechanistic hypothesis for the first failed fixed
identity, `2017-01 NORMAL_MONTHLY_REPORT2_OBJECT`: the inherited parser may be
too restrictive because it requires the year-page `TH` text to equal the
logical month `2017-01`.

The exact SOURCE_B report label remains:

```text
Stock Trading Volume & Value
```

V9_016 does not broaden, alias, repair, or otherwise change that report
label. It changes no provider, source, coverage, calibration identity,
transport policy, retry policy, threshold, partition, stopping rule, or
downstream authority.

## 2. Frozen month-header candidate categories

The candidate categories below are frozen before any real 2017 year-page
semantic read:

1. `LOGICAL_YYYY_MM`: exact normalized text `2017-01`.
2. `EN_MONTH_ABBR_DOT`: exact normalized text `Jan.`.
3. `EN_MONTH_ABBR`: exact normalized text `Jan`.
4. `EN_MONTH_FULL`: exact normalized text `January`.
5. `NUMERIC_MONTH`: exact normalized text `1`.
6. `NUMERIC_MONTH_ZERO_PADDED`: exact normalized text `01`.

Normalization is exactly:

```text
" ".join(raw_text.split())
```

No lowercasing, case-folding, NFKC, NFC, fuzzy matching, substring
matching, regex repair, punctuation repair, arbitrary-text output, or
year/month inference from a URL is allowed. No first/last selection is
allowed.

### Fixed-eight identity and month-token derivation

The fixed-eight calibration identities remain unchanged. The seven NORMAL
identities use the single month-header category frozen by D, with the exact
token mapping below. The special identity does not use this mapping.

```text
2017-01 NORMAL_MONTHLY_REPORT2_OBJECT
2019-12 NORMAL_MONTHLY_REPORT2_OBJECT
2020-01 NORMAL_MONTHLY_REPORT2_OBJECT
2022-03 NORMAL_MONTHLY_REPORT2_OBJECT
2022-04 NORMAL_MONTHLY_REPORT2_OBJECT
2022-05 NORMAL_MONTHLY_REPORT2_OBJECT
2026-01 NORMAL_MONTHLY_REPORT2_OBJECT

2022-04 PRE_APRIL_1_REFERENCE_OBJECT
```

For logical month `YYYY-MM`, let `Y` be the four-digit year and `M` be the
two-digit month number. Only the following month numbers are required by the
fixed-seven NORMAL identities in this study:

| Candidate category | 01 | 03 | 04 | 05 | 12 |
| --- | --- | --- | --- | --- | --- |
| `LOGICAL_YYYY_MM` | `Y-01` | `Y-03` | `Y-04` | `Y-05` | `Y-12` |
| `EN_MONTH_ABBR_DOT` | `Jan.` | `Mar.` | `Apr.` | `May.` | `Dec.` |
| `EN_MONTH_ABBR` | `Jan` | `Mar` | `Apr` | `May` | `Dec` |
| `EN_MONTH_FULL` | `January` | `March` | `April` | `May` | `December` |
| `NUMERIC_MONTH` | `1` | `3` | `4` | `5` | `12` |
| `NUMERIC_MONTH_ZERO_PADDED` | `01` | `03` | `04` | `05` | `12` |

For `LOGICAL_YYYY_MM`, the exact token is the four-digit `Y`, a hyphen, and
the zero-padded two-digit `M`; the table's `Y-..` notation is only the
mechanical year-parameterized form of that exact token. No speculative
mappings for other months are part of this study.

The `2022-04 PRE_APRIL_1_REFERENCE_OBJECT` identity must not use the
month-header recovery mapping. It continues to use the inherited unchanged
`extract_april_pre_candidates` and `resolve_source_b_april_pre_object`
mechanics and the exact inherited `APRIL_1_2022_REFERENCE_LABEL`.

The mapping is deterministic and global to the later fixed-eight validation:
there is no alternate abbreviation, `Sept`/`Sep` choice, locale lookup,
Python locale/calendar-dependent runtime choice, per-year or per-month
category switching, category fallback, fuzzy matching, case folding,
punctuation repair, substring matching, or first/last selection.

## 3. Preregistered stages

### A — design GPT exact-SHA review

GPT must independently review this exact design and return PASS. Until that
review, the design remains `DRAFT_REMEDIATED_AWAITING_GPT_REVIEW` and no V9_016
execution or implementation is authorized.

### B — hash-only binding of preserved V9_015 year locks

After Stage A PASS, bind all five preserved V9_015 Stage-F year locks only by
the supplied safe byte-count and SHA-256 metadata. This stage performs no
network request and no semantic HTML inspection. It does not read or expose
raw HTML, URLs, local paths, exception text, or arbitrary labels.

The five bindings must match the terminal adjudication metadata exactly. Any
missing, malformed, inaccessible, ambiguous, or mismatched metadata is
`FAIL_TERMINAL`. No root or year-page refetch, replacement, reconstruction,
or authorization reuse is allowed.

### C1 — synthetic-only structural probe implementation

C1 may begin only after the design receives GPT exact-SHA PASS. It is a
synthetic-only implementation checkpoint. Its only document input is
synthetic bytes; it performs no network request, filesystem access, private
read, or production-payload read.

The probe uses the reviewed standard-library HTML parser mechanics. It
reports safe counts only for exact `2017-01`:

- parser success;
- table count;
- exact SOURCE_B report cell/row multiplicity;
- `TH` multiplicity for each of the six frozen month-header categories;
- for each category, same-table-with-exact-report multiplicity;
- for each category, intersection in-bounds boolean/count; and
- for each category, intersection href multiplicity `ZERO`, `ONE`, or `MANY`.

It also reports the inherited legacy candidate count. It never reports raw
cell text, href, URL, HTML, exception text, or local path. C1 requires GPT
exact-SHA PASS before C2.

### C2 — one no-network execution on the exact bound 2017 lock

Only after C1 GPT exact-SHA PASS, C2 runs once with no network and only
against the exact hash-bound 2017 preserved year lock. C2 does not inspect
the semantic content of the 2019, 2020, 2022, or 2026 locks.

The exact SOURCE_B report row must be uniquely established. Exactly one
month-header category may establish exactly one same-table month column with
exactly one href at the report-row/month-column intersection. Every other
month-header category must have zero eligible intersections. Many
intersections, multiple admissible categories, or no admissible category is
`FAIL_TERMINAL`.

There is no category merge, fallback, first/last choice, URL construction,
or post-observation category invention.

If `LOGICAL_YYYY_MM` is uniquely admissible with exactly one href despite
the inherited V9_015 locator having returned `PDF_LOCATOR_FAILURE` before any
PDF lock, C2 must not reinterpret that as success. The result is
`CHATGPT_DECISION_REQUIRED` with reason `IMPLEMENTATION_INCONSISTENCY`, and
the process stops.

### D — GPT freeze of the single supported alternate category

After C2, GPT freezes the single supported alternate category, if the safe
evidence supports one. No new category may be introduced after observation.
The D freeze may state only the supported category and the already-frozen
exact-token, multiplicity, same-table, intersection, and fail-closed rules.

### E — later implementation checkpoint

Only after D PASS may a later implementation task implement the frozen rule.
Only after that implementation receives GPT exact-SHA PASS may a separately
authorized no-network stage apply it to the exact five preserved year pages
for the fixed eight identities.

That later fixed-eight validation must apply the one D-frozen category and the
exact mapping above to all seven NORMAL identities, while applying only the
inherited special-reference mechanics to `2022-04 PRE_APRIL_1_REFERENCE_OBJECT`.
It must require exactly one candidate for every required identity. `ZERO` or
`MANY` for any identity is `FAIL_TERMINAL`; another category must not be tried
as fallback for a failed month, mixed category success is prohibited, and no
category may be invented or changed after C2 observation.

Any later PDF network acquisition requires a fresh point-of-use human
authorization. The consumed V9_015 authorization is not reusable.

## 4. Safe evidence and non-claims

V9_016 evidence remains `INPUT_BINDING_ONLY` with zero profitability
evidential capacity. Safe outputs may contain only closed statuses,
failure classes/reasons, hashes, byte counts, booleans, category names,
multiplicity buckets, and bounded counts required by the preregistered
schema.

No stage may emit raw HTML, raw cell text, hrefs, URLs, local paths,
exception text, private identities, PDF bytes/text, trading dates, unit
acceptance, relation results, T0 results, model results, or profitability
claims.

At this design checkpoint:

```text
V9_015=FAIL_TERMINAL
V9_015_RETRY=false
V9_015_AUTOMATIC_SUCCESSOR=false
V9_015_STAGE_G=NOT_REACHED
V9_015_STAGE_H=NOT_REACHED
V9_016_REAL_RAW_READS=0
V9_016_NETWORK_REQUESTS=0
V9_016_PROTECTED_SOURCE_A_READS=0
V9_016_HUMAN_GATES_CONSUMED=0
T0=NOT_RUN
future_profitability=UNESTABLISHED
```

The next action is `GPT_EXACT_SHA_INDEPENDENT_REVIEW` of this design and
terminal record.
