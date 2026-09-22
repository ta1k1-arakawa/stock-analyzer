# V13 pre-implementation contract matrix

This matrix maps the frozen V13 base design and determinism amendment to the
offline implementation.  It is a synthetic proof plan, not market evidence.

| Frozen area | Authority | Implementation | Synthetic success / fail-closed proof | Boundary / Q |
|---|---|---|---|---|
| 500 manifest and order | Design: Universe | `select_universe` | repeatable 500/hash / duplicate or short pool raises | supplied codes only; Q12 |
| Session calendar, t+N, study end | Design: Timing; Amendment §1 | `SessionCalendar` | exact next sessions / no entry at end | no 2026 read; Q5, Q8 |
| Stage A/B/C and transforms | Design: staged population | `build_rank_population` | two eligible sectors / invalid dispersion is `NO_RANK_DATA_QUALITY` | in-memory only; Q1, Q4, Q11 |
| Features and target | Design: features/timing | `feature_row`, `base_target` | known synthetic value / missing future value undefined | no imputation; Q1, Q11 |
| Monthly as-of training | Design: walk-forward | `monthly_predictions` | two monthly fits / current-month label rejected | supplied labels only; Q2, Q3 |
| LightGBM and Ridge | Design: models | factory functions | actual fit/predict / fixed hyperparameter assertion | synthetic fit only; Q11 |
| Comparator/random rankings | Design: comparators; Amendment §§2–3 | `rank_candidates`, `random_ranking` | ties and SHA key / no re-randomization | Q6, Q10, Q12 |
| Entry, affordability, lot size | Design: simulator; Amendment §1 | `simulate` | fallback and 100 shares / invalid Open and no fill | Q6, Q7 |
| Close exit/event order | Design: simulator; Amendment §1 | `simulate` | exit then next signal / missing exit fails | Q8, Q9 |
| Base/stress replay | Design: timing | `simulate_pair` | shared rankings / stress only changes execution | Q10 |
| Equity and metrics | Design: metrics; Amendment §§4–5 | `metrics` | daily mark / zero-denominator statuses | finite JSON; Q7 |
| IC, decile, concentration, percentile | Design: metrics | `diagnostics`, `linear_percentile` | defined values / undefined statuses | no outcome claims; Q11 |
| A–Q adjudication | Design: criteria; Amendment §6 | `adjudicate` | exact A–Q/Q1–Q12 keys / any false fails | Q1–Q12 |
| Serialization | Issue implementation architecture | `canonical_json` | byte-identical outputs / NaN rejected | offline-only |
| Frozen bindings | Issue frozen bindings | `verify_frozen_bindings` | exact blobs/SHAs / mismatch raises | provenance; Q11 |

`FROZEN_REMEDIATION_BUDGET_STATUS=WITHIN_LIMIT`; substantive rounds: `0 used / 2 remaining`.
