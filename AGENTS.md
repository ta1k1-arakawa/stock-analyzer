# Evaluator v2 rules

- Run fixed evaluation only with the CSV snapshot capped at 2026-05-20.
- Research selection ends 2025-03-31. The later period is diagnostic only.
- Never modify the detached baseline worktree. Use `python compare_evaluators.py --baseline-worktree ../stock-analyzer-baseline --prices data/price_snapshot`.
- Run `python -m pytest -q` before accepting an iteration. Reject any leakage, future-price access, negative cash, or duplicate capital allocation.
- Record every comparison in `experiments/iterations.csv`.

## Fixed OHLCV benchmark

- Generate only when intentionally refreshing the frozen input: `python scripts/generate_benchmark.py`.
- Normal benchmark code must use `FixedOHLCVLoader`; it never falls back to HTTP.
- Validate before use with `python scripts/validate_benchmark.py`. Do not hand-edit CSVs or `manifest.json`.
