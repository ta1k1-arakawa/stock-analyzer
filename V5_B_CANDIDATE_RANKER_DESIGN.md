# V5-B Candidate Ranker (pre-registered exploratory design)

V5-B changes only the order of same-day V5-A2 FIXED100_D5_ONLY candidates. BASELINE_RANK is return_60d, return_20d, ticker; AI_RANK is predicted D5 return descending, baseline rank, ticker. There is no regime gate, threshold, stop, sizing, or execution change.

The target is raw D1 open with 0.03% entry slippage to raw D5 open with 0.03% exit slippage. Gap-up, split, missing and non-finite rows are excluded from training. The exactly twenty causal features are listed in `src/v5_b_candidate_ranker.py`; same-date percentiles use average ties and normalized percentile ranks.

One pooled LightGBM L1 regressor and the fixed parameters in `MODEL_PARAMS` are pre-registered. For evaluation year Y, only labels with `exit_date < Y-01-01` are eligible and at least 1,000 rows are required. Years 2020–2025 are exploratory, not holdout; deployment remains false. Future evaluation cache overlap must be OHLCV-identical before any run.

The formal runner is cache-only, validates repository state and overlap rows, performs two identical core passes, and writes only after byte equality. It was not invoked in this implementation turn. Synthetic smoke is the only executed evaluation.
