# V9_015 Stage D OPTION_VALUE Year-Navigation Freeze

```text
status=FROZEN_AWAITING_GPT_EXACT_SHA_REVIEW
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
```

## C2 safe evidence

```text
root_sha256=2e839c60bfb9d6edb59a903a590a505130124b8380b096ae84b06e4b0972098c
root_byte_count=75185
html_parser_success=true
structure_failure_class=null
safe_calibration_status=PASS
deterministic_candidate_category=OPTION_VALUE
all_required_years_deterministically_bindable=true
```

- Required-year `ANCHOR_HREF` multiplicity is `ZERO` for `2017`, `2019`, `2020`, `2022`, and `2026`.
- Required-year `OPTION_VALUE` multiplicity is `ONE` for `2017`, `2019`, `2020`, `2022`, and `2026`.
- No raw HTML, value, href, or URL was observed or recorded.
- Visible `2026` token count `2` is diagnostic-only and has no candidate authority.

## Frozen mechanics

```text
ACCEPTED_ROOT_CANDIDATE_CATEGORY=OPTION_VALUE

REQUIRED_YEAR_LABELS=
2017
2019
2020
2022
2026

LABEL_NORMALIZATION=" ".join(raw_text.split())
```

### OPTION_VALUE candidate eligibility

- Normalized inner visible text exactly equals one required year label.
- The `value` attribute exists exactly once.
- `value` is a nonempty string.
- Script/style text is excluded exactly as in the reviewed C1 implementation.
- No fuzzy matching, lower/casefold, NFKC/NFC, substring matching, punctuation repair, or year-from-URL inference is allowed.

### Multiplicity

- Exactly one eligible `OPTION_VALUE` candidate is required for every required year.
- Required-year `ANCHOR_HREF` multiplicity must remain zero for every required year.
- `ZERO` or `MANY` fails closed.
- No first/last selection, mixed-category selection, fallback to `ANCHOR_HREF`, or fallback to `VISIBLE_TEXT` is allowed.

### URL binding rule for Stage E

- Use only the raw value belonging to the exact selected `OPTION_VALUE` candidate.
- Do not inspect or choose the candidate based on value content.
- Resolve the raw value only against the exact frozen `SOURCE_B_ARCHIVE_ROOT`.
- Use the already-reviewed locked-page-link resolution discipline: `urllib.parse.urljoin(exact frozen root, raw candidate value)` followed by the existing reviewed JPX URL validation.
- After resolution and validation, represent the result as `RootYearCandidate` with the exact required year label.
- Pass that candidate through existing `resolve_source_b_year_page` unchanged.
- No URL construction from year, archive-number construction, URL/path-pattern guessing, alternate root, alternate provider, alternate language site, redirect-based substitution, or fallback is allowed.

### Closed failures

The following fail closed: missing `OPTION_VALUE` candidate; empty option value; duplicate value attribute; zero or many eligible option candidates; a required-year `ANCHOR_HREF` candidate where the frozen rule requires zero; malformed relevant structure; invalid or off-domain resolved URL; root provenance or hash mismatch; and implementation or schema failure.

There is no root refetch, alternate category, or alternative provider/root.

## Explicit implementation boundary

- Existing V9_014 `extract_root_year_candidates` is anchor-based and must not be used as the V9_015 root extraction mechanism.
- Stage E must implement a new `OPTION_VALUE` root extractor under this exact freeze.
- Stage E implementation is synthetic-only first.
- Existing downstream locator resolver and URL validation may be reused unchanged.
- Stage E implementation requires GPT exact-SHA independent PASS before any real child URL resolution or request.
- Stage F human/network authority is not granted by this freeze.
