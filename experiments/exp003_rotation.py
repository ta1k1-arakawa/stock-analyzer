"""EXP-003: value + profitability rotation portfolio, rebalanced in 10 staggered tranches.

Pre-registration: docs/experiments/EXP-003_PREREGISTRATION.md
Data: the EXP-002 J-Quants cache (docs/experiments/EXP-002_DATA_MANIFEST.json).
Usage:
  python experiments/exp003_rotation.py <cache_dir> --count-only   # eligible-universe counts, no returns
  python experiments/exp003_rotation.py <cache_dir>                # dev evaluation
  python experiments/exp003_rotation.py <cache_dir> --holdout      # holdout, once, only if dev passes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUDGET = 400_000
DEV = (pd.Timestamp("2017-01-04"), pd.Timestamp("2024-01-31"))  # trading days simulated
HOLD = (pd.Timestamp("2024-02-01"), pd.Timestamp("2026-09-25"))
COSTS = {"base": 0.0003, "stress": 0.0010}
N_TRANCHES, PER_TRANCHE, HOLD_DAYS = 10, 2, 20  # 20 names; one tranche every 2 trading days
LIQ_DAYS, LIQ_MIN = 20, 100_000_000
N_RANDOM = 1000


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ----------------------------------------------------------------------------
# data and point-in-time signals
# ----------------------------------------------------------------------------

def load(cache: Path):
    topix = pd.DataFrame(json.loads((cache / "topix.json").read_text()))
    topix["Date"] = pd.to_datetime(topix["Date"])
    topix = topix.set_index("Date").sort_index()[["O", "C"]].astype(float)
    cols = {"O": "AdjO", "C": "AdjC", "Va": "Va", "MktCap": "MktCap"}
    frames = {k: {} for k in cols}
    fins = {}
    for f in sorted((cache / "bars").glob("*.json")):
        bars = pd.DataFrame(json.loads(f.read_text()))
        if bars.empty:
            continue
        bars["Date"] = pd.to_datetime(bars["Date"])
        bars = bars.set_index("Date").sort_index()
        for k, src in cols.items():
            frames[k][f.stem] = pd.to_numeric(bars[src], errors="coerce")
        fins[f.stem] = json.loads((cache / "fins" / f.name).read_text())
    panel = {k: pd.DataFrame(v).reindex(topix.index) for k, v in frames.items()}
    # drop market-wide halts (e.g. the 2020-10-01 TSE outage: no stock traded) from the calendar
    traded = panel["O"].notna().any(axis=1)
    topix = topix[traded]
    panel = {k: v[traded] for k, v in panel.items()}
    return topix, panel, fins


def known_from(dates: pd.DatetimeIndex, records: list[tuple[pd.Timestamp, str, float]]) -> pd.Series:
    """Values usable in a signal computed at the close of each trading day: a disclosure dated D is
    usable from the first trading day >= D (trades happen at the next day's open)."""
    s = pd.Series(np.nan, index=dates)
    for date, _time, val in sorted(records, key=lambda r: (r[0], r[1])):
        pos = dates.searchsorted(date, side="left")
        if pos < len(dates) and np.isfinite(val):
            s.iloc[pos] = val
    return s.ffill()


def fundamentals(fins: dict, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time equity (latest financial statement) and net-profit forecast (latest disclosure:
    NxFNp on FY statements, FNP otherwise)."""
    eq, npf = {}, {}
    for code, recs in fins.items():
        e, n = [], []
        for r in recs:
            d = pd.Timestamp(r["DiscDate"])
            if "FinancialStatements" in r["DocType"]:
                e.append((d, r["DiscTime"], num(r["Eq"])))
            v = r["NxFNp"] if r["CurPerType"] == "FY" and "FinancialStatements" in r["DocType"] else r["FNP"]
            n.append((d, r["DiscTime"], num(v)))
        eq[code] = known_from(dates, e)
        npf[code] = known_from(dates, n)
    return pd.DataFrame(eq), pd.DataFrame(npf)


def signals(panel, fins, dates) -> dict[str, pd.DataFrame]:
    """Scores at each close (higher = preferred); NaN where the stock is not eligible."""
    eq, npf = fundamentals(fins, dates)
    eq, npf = eq.reindex(columns=panel["C"].columns), npf.reindex(columns=panel["C"].columns)
    mcap = panel["MktCap"] * 1e6
    liq = panel["Va"].fillna(0.0).rolling(LIQ_DAYS, min_periods=LIQ_DAYS).mean()  # no-trade day = 0 yen
    ok = (liq >= LIQ_MIN) & (mcap > 0) & (eq > 0) & npf.notna() & panel["C"].notna()
    bp = (eq / mcap).where(ok)
    roe = (npf / eq).where(ok)
    r_bp = bp.rank(axis=1, pct=True)
    r_roe = roe.rank(axis=1, pct=True)
    comp = r_bp + r_roe
    return {"composite": comp, "value_only": r_bp, "profit_only": r_roe, "bottom_composite": -comp}


# ----------------------------------------------------------------------------
# staggered-tranche simulation
# ----------------------------------------------------------------------------

def simulate(score: np.ndarray, o: np.ndarray, c_ff: np.ndarray, start: int, end: int, cost: float,
             rng: np.random.Generator | None = None, marks: bool = True):
    """Simulate trading days start..end (positions in the full calendar).

    All tranches buy on day `start` (i = 0); afterwards tranche k trades at the open of day start + i
    when i % 2 == 0 and (i % HOLD_DAYS) // 2 == k, so each tranche is held HOLD_DAYS trading days. The signal is the score at the previous close. Each trade sells the
    tranche's names and buys the top PER_TRANCHE eligible names not held by other tranches (random
    eligible names when rng is given), equal-weighted. Costs are charged on every sale and purchase,
    including names that are kept (conservative). At `end` everything is sold at the close.
    Returns (final value, daily values or None, names changed per trade).
    """
    step = HOLD_DAYS // N_TRANCHES
    holdings = [dict() for _ in range(N_TRANCHES)]  # code index -> shares
    cash = [BUDGET / N_TRANCHES] * N_TRANCHES
    values, changes = [], []
    for t in range(start, end + 1):
        i = t - start
        if i == 0:
            due = range(N_TRANCHES)
        elif i % step == 0:
            due = [(i % HOLD_DAYS) // step]
        else:
            due = []
        for k in due:
            px_sell = {j: (o[t, j] if o[t, j] > 0 else c_ff[t - 1, j]) for j in holdings[k]}
            cash[k] += sum(sh * px_sell[j] * (1 - cost) for j, sh in holdings[k].items())
            others = set().union(*(holdings[m] for m in range(N_TRANCHES) if m != k))
            s = score[t - 1]
            cand = np.flatnonzero(np.isfinite(s) & (o[t] > 0))
            cand = cand[~np.isin(cand, list(others))]
            if rng is None:
                order = cand[np.lexsort((cand, -s[cand]))]
                picks = order[:PER_TRANCHE]
            else:
                picks = rng.choice(cand, min(PER_TRANCHE, len(cand)), replace=False)
            changes.append(len(set(picks) - set(holdings[k])))
            holdings[k] = {}
            if len(picks):
                alloc = cash[k] / len(picks)
                holdings[k] = {j: alloc / (o[t, j] * (1 + cost)) for j in picks}
                cash[k] = 0.0
        if marks:
            values.append(sum(cash) + sum(sh * c_ff[t, j] for h in holdings for j, sh in h.items()))
    final = sum(cash) + sum(sh * c_ff[end, j] * (1 - cost) for h in holdings for j, sh in h.items())
    if marks:
        values[-1] = final
    return final, (np.array(values) if marks else None), changes


def summarize(values: np.ndarray, dates: pd.DatetimeIndex, topix: pd.DataFrame, changes: list[int]) -> dict:
    v = pd.Series(values, index=dates)
    tp = topix["C"].loc[dates] / topix["O"].loc[dates[0]]  # TOPIX bought at the first open
    years = (dates[-1] - dates[0]).days / 365.25
    total = v.iloc[-1] / BUDGET - 1
    m_v, m_t = v.resample("ME").last(), tp.resample("ME").last()
    ex = (m_v / m_v.shift(1).fillna(BUDGET)) - (m_t / m_t.shift(1).fillna(1.0))
    y_v, y_t = v.groupby(v.index.year).last(), tp.groupby(tp.index.year).last()
    by_year = y_v / y_v.shift(1).fillna(BUDGET) - 1
    tp_year = y_t / y_t.shift(1).fillna(1.0) - 1
    return {
        "profit": round(float(v.iloc[-1] - BUDGET)),
        "annual_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2),
        "max_dd_pct": round(float((v / v.cummax() - 1).min()) * 100, 2),
        "monthly_excess_mean_pct": round(float(ex.mean()) * 100, 3),
        "t_monthly_excess": round(float(ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))), 2),
        "months": int(len(ex)),
        "months_beating_topix_pct": round(float((ex > 0).mean()) * 100, 1),
        "trades": len(changes),
        "names_changed_per_trade": round(float(np.mean(changes)), 2),
        "by_year_pct": {int(y): round(float(x) * 100, 1) for y, x in by_year.items()},
        "topix_by_year_pct": {int(y): round(float(x) * 100, 1) for y, x in tp_year.items()},
    }


def evaluate(period, topix, panel, sig, variants, with_random: bool) -> dict:
    dates = topix.index
    start, end = int(dates.searchsorted(period[0])), int(dates.searchsorted(period[1], side="right")) - 1
    span = dates[start : end + 1]
    o = panel["O"].to_numpy()
    c_ff = panel["C"].ffill().to_numpy()
    tp = topix["C"].iloc[end] / topix["O"].iloc[start] - 1
    ew = panel["C"].iloc[start - 1 : end + 1].pct_change().mean(axis=1).iloc[1:]
    res = {
        "period": [str(span[0].date()), str(span[-1].date())],
        "B0_topix_buy_hold": {"profit": round(float(tp) * BUDGET)},
        "B1_equal_weight_universe_buy_hold": {"profit": round(float(np.prod(1 + ew) - 1) * BUDGET)},
    }
    for name, costs in variants.items():
        score = sig[name].to_numpy()
        for cname in costs:
            _, values, changes = simulate(score, o, c_ff, start, end, COSTS[cname])
            res[f"{name}_{cname}"] = summarize(values, span, topix, changes)
    if with_random:
        score = sig["composite"].to_numpy()
        prof = np.array([
            simulate(score, o, c_ff, start, end, COSTS["base"], rng=np.random.default_rng(seed), marks=False)[0]
            for seed in range(N_RANDOM)
        ]) - BUDGET
        res["B2_random_base"] = {"median_profit": round(float(np.median(prof))),
                                 "p95_profit": round(float(np.percentile(prof, 95))),
                                 "p5_profit": round(float(np.percentile(prof, 5)))}
    return res


def verdict(res: dict) -> dict:
    b0 = res["B0_topix_buy_hold"]["profit"]
    base, stress = res["composite_base"], res["composite_stress"]
    checks = {
        "1_base_profit_gt_topix": base["profit"] > b0,
        "2_t_monthly_excess_gt_2": base["t_monthly_excess"] > 2,
        "3_stress_profit_gt_topix": stress["profit"] > b0,
        "4_beats_random_p95": base["profit"] > res["B2_random_base"]["p95_profit"],
    }
    return {**checks, "promising": all(checks.values())}


def counts(topix, sig) -> dict:
    out = {}
    elig = sig["composite"].notna().sum(axis=1)
    for pname, period in (("dev", DEV), ("holdout", HOLD)):
        e = elig[(elig.index >= period[0] - pd.Timedelta(days=7)) & (elig.index <= period[1])]
        out[pname] = {"eligible_min": int(e.min()), "eligible_median": int(e.median()), "eligible_max": int(e.max()),
                      "trading_days": int(((topix.index >= period[0]) & (topix.index <= period[1])).sum()),
                      "eligible_codes_ever": int(sig["composite"].loc[period[0]:period[1]].notna().any().sum())}
    return out


def main(argv: list[str]) -> None:
    topix, panel, fins = load(Path(argv[0]))
    sig = signals(panel, fins, topix.index)
    if "--count-only" in argv:
        out = counts(topix, sig)
    elif "--holdout" in argv:
        out = evaluate(HOLD, topix, panel, sig, {"composite": ("base", "stress")}, with_random=True)
    else:
        variants = {"composite": ("base", "stress"), "value_only": ("base",), "profit_only": ("base",),
                    "bottom_composite": ("base",)}
        out = evaluate(DEV, topix, panel, sig, variants, with_random=True)
        out["verdict"] = verdict(out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
