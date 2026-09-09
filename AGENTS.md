# Evaluator v2 rules

- Run fixed evaluation only with the CSV snapshot capped at 2026-05-20.
- Research selection ends 2025-03-31. The later period is diagnostic only.
- Never modify the detached baseline worktree. Use `python compare_evaluators.py --baseline-worktree ../stock-analyzer-baseline`. The runner validates and injects `data/benchmark` without an HTTP fallback.
- Treat `data/backtest_comparison` as read-only diagnostics. Never feed reference-period comparison results into rule selection, and verify that `selected_rules.csv` has the same SHA-256 before and after comparison.
- Use `python backtest.py --mode loop-validation` for iterative research. It may expose only `data/loop_validation_results/summary.json`; do not open research-test, reference-period, or baseline-comparison outputs during the loop.
- Use the default `python backtest.py` only for a separately authorized full diagnostic run after the loop decision is frozen.
- For acceptance runs, keep the candidate worktree clean and write artifacts outside it with `--output-dir`. Pass that full output explicitly to `compare_evaluators.py --candidate-results`; the recorded candidate commit must equal the current HEAD.
- `config_hash` always uses `utf8-normalized-lf-v1`; raw checkout bytes must never be hashed directly.
- Run `python -m pytest -q` before accepting an iteration. Reject any leakage, future-price access, negative cash, or duplicate capital allocation.
- Record every comparison in `experiments/iterations.csv`.

## Fixed OHLCV benchmark

- Generate only when intentionally refreshing the frozen input: `python scripts/generate_benchmark.py`.
- Normal benchmark code must use `FixedOHLCVLoader`; it never falls back to HTTP.
- Validate before use with `python scripts/validate_benchmark.py`. Do not hand-edit CSVs or `manifest.json`.
