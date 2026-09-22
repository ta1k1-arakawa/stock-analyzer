# V13 pre-implementation contract matrix

This matrix maps the frozen V13 base design and determinism amendment to the
offline implementation.  It is a synthetic proof plan, not market evidence.

| Frozen area | Authority | Implementation | Synthetic success / fail-closed proof | Boundary / Q |
|---|---|---|---|---|
| 500 manifest and order | Design: Universe | `select_universe` | repeatable 500/hash / duplicate or short pool raises | supplied codes only; Q12 |
| Session calendar, t+N, study end | Design: Timing; Amendment §1 | `SessionCalendar` | exact next sessions / no entry at end | no 2026 read; Q5, Q8 |
| Stage A/B/C and transforms | Design: staged population | `build_rank_population` | two eligible sectors / invalid dispersion is `NO_RANK_DATA_QUALITY` | in-memory only; Q1, Q4, Q11 |
| Raw Stage-A features and target | Design: features/timing | `stage_a_from_raw`, `base_target` | raw OHLCV derivation / missing future value undefined | no imputation; Q1, Q11 |
| Monthly as-of training | Design: walk-forward | `monthly_predictions` | two monthly fits / current-month label rejected | supplied labels only; Q2, Q3 |
| LightGBM and Ridge | Design: models | factory functions | actual fit/predict / fixed hyperparameter assertion | synthetic fit only; Q11 |
| Comparator/random rankings | Design: comparators; Amendment §§2–3 | `rank_candidates`, `ranking_hash` | ties and SHA key / no re-randomization | Q6, Q10, Q12 |
| Entry, affordability, lot size | Design: simulator; Amendment §1 | `simulate` | fallback and 100 shares / invalid Open and no fill | Q6, Q7 |
| Close exit/event order | Design: simulator; Amendment §1 | `simulate` | exit then next signal / missing exit fails | Q8, Q9 |
| Base/stress replay | Design: timing | `simulate` | same ranking object replayed with friction-only difference | Q10 |
| Equity and metrics | Design: metrics; Amendment §§4–5 | `trade_metrics`, `diagnostics` | daily mark / zero-denominator statuses | finite JSON; Q7 |
| IC, decile, concentration, percentile | Design: metrics | `diagnostics`, `linear_percentile` | defined values / undefined statuses | no outcome claims; Q11 |
| A–Q adjudication | Design: criteria; Amendment §6 | `adjudicate` | exact A–Q/Q1–Q12 keys / any false fails | Q1–Q12 |
| Serialization | Issue implementation architecture | `canonical_json` | byte-identical outputs / NaN rejected | offline-only |
| Frozen bindings | Issue frozen bindings | `verify_frozen_bindings` | exact four artifact bindings / false on mismatch | provenance; Q11 |
| Complete production path | Issue #31 HIGH_1 | `run_synthetic_feasibility` | manifest → raw Stage A/B/C → labels → monthly fits → all rankings → base/stress → metrics → A-P/Q | synthetic-only; Q1–Q12 |

`FROZEN_REMEDIATION_BUDGET_STATUS=FINAL_ROUND_CONSUMED`; substantive rounds: `2 used / 0 remaining`.
