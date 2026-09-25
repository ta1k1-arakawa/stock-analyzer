"""EXP-001: overnight holding vs weekly reversal on the fixed 7-stock data.

Pre-registration: docs/experiments/EXP-001_PREREGISTRATION.md
Usage: python experiments/exp001_short_horizon.py <ohlcv_dir> [--holdout]
The holdout period is evaluated only with --holdout, per the pre-registration
(only for a candidate that passes every dev criterion).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODES = ["4188", "4689", "5020", "7211", "7267", "8306", "9432"]
BUDGET = 300_000
DEV = (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-03-31"))
HOLD = (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-05-20"))
COSTS = {"base": 0.0003, "stress": 0.0010}
N_PICK, LOOKBACK, N_RANDOM = 2, 5, 1000


def load(ohlcv_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for code in CODES:
        df = pd.read_csv(ohlcv_dir / f"{code}.csv", parse_dates=["Date"]).set_index("Date")
        frames[code] = df
    dates = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    return {f: pd.DataFrame({c: frames[c].loc[dates, f] for c in CODES}) for f in ("Open", "High", "Low", "Close")}


def in_period(dates: pd.Index, period) -> np.ndarray:
    return np.asarray((dates >= period[0]) & (dates <= period[1]))


def stats(returns: pd.Series, marks: pd.Series | None = None) -> dict:
    """returns: per-trade portfolio returns indexed by entry date; marks: optional equity marks for drawdown."""
    r = returns.dropna()
    if r.empty:
        return {"trades": 0}
    equity = BUDGET * (1 + r).cumprod()
    path = marks if marks is not None else equity
    dd = (path / path.cummax() - 1).min()
    years = (r.index[-1] - r.index[0]).days / 365.25
    total = equity.iloc[-1] / BUDGET - 1
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {
        "trades": int(len(r)),
        "profit": round(float(equity.iloc[-1] - BUDGET)),
        "annual_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2) if years > 0 else None,
        "max_dd_pct": round(float(dd) * 100, 2),
        "mean_pct": round(float(r.mean()) * 100, 4),
        "t_stat": round(float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))), 2),
        "win_pct": round(float((r > 0).mean()) * 100, 1),
        "years_positive": f"{int((by_year > 0).sum())}/{len(by_year)}",
        "by_year_pct": {int(y): round(float(v) * 100, 1) for y, v in by_year.items()},
    }


def overnight(p, c):
    ret = (p["Open"].shift(-1) * (1 - c)) / (p["Close"] * (1 + c)) - 1
    return ret.mean(axis=1)


def intraday(p, c):
    ret = (p["Close"] * (1 - c)) / (p["Open"] * (1 + c)) - 1
    return ret.mean(axis=1)


def weeks(dates: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """(first_pos, last_pos) of each ISO week of trading dates, in order."""
    iso = dates.isocalendar()
    key = (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()
    out, start = [], 0
    for i in range(1, len(dates) + 1):
        if i == len(dates) or key[i] != key[start]:
            out.append((start, i - 1))
            start = i
    return out


def weekly(p, c, pick, rng=None):
    """Signal at the close of each week's last day, hold the next week (open of first day -> close of last day)."""
    close, open_ = p["Close"].to_numpy(), p["Open"].to_numpy()
    dates = p["Close"].index
    wk = weeks(dates)
    rets, marks_idx, marks_val = {}, [], []
    for (s0, e0), (s1, e1) in zip(wk[:-1], wk[1:]):
        if e0 - LOOKBACK < 0:
            continue
        past = close[e0] / close[e0 - LOOKBACK] - 1
        order = np.lexsort((np.arange(len(CODES)), past))  # ascending return, code order on ties
        if pick == "bottom":
            chosen = order[:N_PICK]
        elif pick == "top":
            chosen = order[::-1][:N_PICK]
        else:
            chosen = rng.choice(len(CODES), N_PICK, replace=False)
        entry = open_[s1, chosen] * (1 + c)
        rets[dates[s1]] = float(np.mean(close[e1, chosen] * (1 - c) / entry - 1))
        for d in range(s1, e1 + 1):
            marks_idx.append(dates[d])
            marks_val.append(float(np.mean(close[d, chosen] / entry)))
    return pd.Series(rets), pd.Series(marks_val, index=marks_idx)


def weekly_equity_marks(rets: pd.Series, marks: pd.Series) -> pd.Series:
    """Daily mark-to-market equity for the weekly strategy (for drawdown)."""
    eq_start = (BUDGET * (1 + rets).cumprod()).shift(1).fillna(BUDGET)
    out = []
    starts = list(rets.index)
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else marks.index[-1] + pd.Timedelta(days=1)
        seg = marks[(marks.index >= s) & (marks.index < end)]
        out.append(seg * eq_start.iloc[i])
    return pd.concat(out)


def run(ohlcv_dir: Path, include_holdout: bool = False) -> dict:
    p = load(ohlcv_dir)
    dates = p["Close"].index
    result = {"dates": [str(dates[0].date()), str(dates[-1].date())]}
    periods = [("dev", DEV)] + ([("holdout", HOLD)] if include_holdout else [])
    for period_name, period in periods:
        mask = in_period(dates, period)
        sub = {k: v[mask] for k, v in p.items()}
        res = {}
        bh = sub["Close"].pct_change().mean(axis=1).iloc[1:]
        res["B1_buy_hold"] = stats(bh)
        for cname, c in COSTS.items():
            on = overnight(sub, c).iloc[:-1]
            res[f"S1_overnight_{cname}"] = stats(on)
            res[f"diag_intraday_{cname}"] = stats(intraday(sub, c))
            for pick, label in (("bottom", "S2_weekly_reversal"), ("top", "diag_weekly_momentum")):
                r, m = weekly(sub, c, pick)
                res[f"{label}_{cname}"] = stats(r, weekly_equity_marks(r, m))
            rand = [
                float(np.prod(1 + weekly(sub, c, "random", np.random.default_rng(seed))[0]) - 1) * BUDGET
                for seed in range(N_RANDOM)
            ]
            res[f"B2_random_{cname}"] = {
                "median_profit": round(float(np.median(rand))),
                "p95_profit": round(float(np.percentile(rand, 95))),
            }
        result[period_name] = res
    return result


if __name__ == "__main__":
    import json

    out = run(Path(sys.argv[1]), include_holdout="--holdout" in sys.argv[2:])
    print(json.dumps(out, ensure_ascii=False, indent=1))
