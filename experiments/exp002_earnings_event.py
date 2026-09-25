"""EXP-002: earnings-event long trades on the V13-excluded universe (J-Quants v2 cache).

Pre-registration: docs/experiments/EXP-002_PREREGISTRATION.md
Usage:
  python experiments/exp002_earnings_event.py <cache_dir> --count-only   # event counts, no returns
  python experiments/exp002_earnings_event.py <cache_dir>                # dev evaluation
  python experiments/exp002_earnings_event.py <cache_dir> --holdout E1   # holdout, one candidate, once
The holdout is evaluated only with --holdout, per the pre-registration
(only for the single candidate selected on dev).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUDGET = 300_000
DEV = (pd.Timestamp("2017-01-01"), pd.Timestamp("2023-12-31"))  # disclosure dates
DEV_PRICE_END = pd.Timestamp("2024-01-31")  # the dev evaluation never reads prices after this date
HOLD = (pd.Timestamp("2024-02-01"), pd.Timestamp("2026-08-14"))
COSTS = {"base": 0.0003, "stress": 0.0010}
HOLD_DAYS = 20
DIAG_HOLD_DAYS = (5, 60)
LIQ_DAYS, LIQ_MIN = 20, 100_000_000  # mean trading value (yen) over the 20 trading days before disclosure
N_RANDOM = 1000
CANDIDATES = {"E1": "E1_forecast_surprise_pos", "E2": "E2_reaction_pos"}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(cache: Path):
    topix = pd.DataFrame(json.loads((cache / "topix.json").read_text()))
    topix["Date"] = pd.to_datetime(topix["Date"])
    topix = topix.set_index("Date").sort_index()[["O", "C"]].astype(float)
    opens, closes, value, fins = {}, {}, {}, []
    for f in sorted((cache / "bars").glob("*.json")):
        bars = pd.DataFrame(json.loads(f.read_text()))
        if bars.empty:
            continue
        bars["Date"] = pd.to_datetime(bars["Date"])
        bars = bars.set_index("Date").sort_index()
        opens[f.stem] = bars["AdjO"].astype(float)
        closes[f.stem] = bars["AdjC"].astype(float)
        value[f.stem] = bars["Va"].astype(float)
        fins += [{**r, "code": f.stem} for r in json.loads((cache / "fins" / f.name).read_text())]
    panel = {k: pd.DataFrame(v).reindex(topix.index) for k, v in (("O", opens), ("C", closes), ("Va", value))}
    fins = pd.DataFrame(fins)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"])
    fins = fins.sort_values(["code", "DiscDate", "DiscTime", "DiscNo"], kind="stable").reset_index(drop=True)
    return topix, panel, fins


# ----------------------------------------------------------------------------
# events (disclosures, prices up to the disclosure, and the trading calendar only)
# ----------------------------------------------------------------------------

def statement_events(fins: pd.DataFrame) -> pd.DataFrame:
    """First disclosure of each financial statement (code, period type, fiscal year end)."""
    st = fins[fins["DocType"].str.contains("FinancialStatements") & (fins["CurFYEn"] != "")]
    return st.drop_duplicates(["code", "CurPerType", "CurFYEn"], keep="first")


def forecast_surprise(fins: pd.DataFrame) -> pd.DataFrame:
    """E1 events: sign of the surprise against the company's own operating-profit forecast.

    - First FY statement of a fiscal year: sign(actual OP - latest OP forecast for that year
      disclosed on an earlier date).
    - Otherwise, a date with a full-year OP forecast (FOP, or NxFOP for the next year) for a year not
      yet reported: sign(last forecast of the day - latest forecast for that year on an earlier date).
    """
    obs = []
    for r in fins.itertuples(index=False):
        if r.FOP != "" and r.CurFYEn != "":
            obs.append((r.code, r.DiscDate, r.CurFYEn, num(r.FOP)))
        if r.NxFOP != "" and r.NxtFYEn != "":
            obs.append((r.code, r.DiscDate, r.NxtFYEn, num(r.NxFOP)))
    obs = pd.DataFrame(obs, columns=["code", "date", "fy", "val"]).dropna()
    obs_by_code = dict(tuple(obs.groupby("code", sort=False)))

    first_fy = statement_events(fins)
    first_fy = first_fy[(first_fy["CurPerType"] == "FY") & (first_fy["OP"] != "")]
    actual = {(r.code, r.CurFYEn): (r.DiscDate, num(r.OP)) for r in first_fy.itertuples(index=False)}

    events = {}
    for (code, fy), (date, op) in actual.items():
        o = obs_by_code.get(code)
        if o is None or np.isnan(op):
            continue
        prior = o[(o["fy"] == fy) & (o["date"] < date)]
        if len(prior):
            events[(code, date)] = ("fy_actual", np.sign(op - prior["val"].iloc[-1]))
    for code, o in obs_by_code.items():
        for (date, fy), day in o.groupby(["date", "fy"], sort=True):  # current year first on a date
            if (code, date) in events:
                continue
            reported = actual.get((code, fy))
            if reported is not None and reported[0] <= date:
                continue  # that year is already reported; not a forecast revision
            prior = o[(o["fy"] == fy) & (o["date"] < date)]
            if len(prior):
                events[(code, date)] = ("revision", np.sign(day["val"].iloc[-1] - prior["val"].iloc[-1]))
    out = pd.DataFrame([(c, d, k, s) for (c, d), (k, s) in events.items()],
                       columns=["code", "disc_date", "kind", "sign"])
    return out.sort_values(["disc_date", "code"]).reset_index(drop=True)


def yoy_sign(fins: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic D5: cumulative OP vs the same period of the previous fiscal year (first disclosures)."""
    st = statement_events(fins)
    st = st[st["OP"] != ""].assign(fy_end=lambda d: pd.to_datetime(d["CurFYEn"]), op=lambda d: d["OP"].map(num))
    prev = st[["code", "CurPerType", "fy_end", "op"]].assign(fy_end=lambda d: d["fy_end"] + pd.DateOffset(years=1))
    m = st.merge(prev, on=["code", "CurPerType", "fy_end"], suffixes=("", "_prev"))
    m = m.assign(sign=np.sign(m["op"] - m["op_prev"])).rename(columns={"DiscDate": "disc_date"})
    return m[["code", "disc_date", "sign"]].dropna().drop_duplicates(["code", "disc_date"]).reset_index(drop=True)


def attach_timing(ev: pd.DataFrame, panel, topix: pd.DataFrame) -> pd.DataFrame:
    """t0 = last trading day before the disclosure date, t1 = first trading day after it.

    liq = mean trading value over the 20 trading days ending at t0.
    ear = stock close(t0) -> close(t1) minus TOPIX over the same days (known at the close of t1).
    """
    dates = topix.index
    d = ev["disc_date"].to_numpy()
    t0 = dates.searchsorted(d, side="left") - 1
    t1 = dates.searchsorted(d, side="right")
    liq_panel = panel["Va"].rolling(LIQ_DAYS, min_periods=1).mean().to_numpy()
    c = panel["C"].to_numpy()
    tc = topix["C"].to_numpy()
    col = {code: i for i, code in enumerate(panel["C"].columns)}
    liq, ear = np.full(len(ev), np.nan), np.full(len(ev), np.nan)
    for k, (code, a, b) in enumerate(zip(ev["code"], t0, t1)):
        j = col.get(code)
        if j is None or a < LIQ_DAYS - 1 or b >= len(dates):
            continue
        liq[k] = liq_panel[a, j]
        ear[k] = c[b, j] / c[a, j] - 1 - (tc[b] / tc[a] - 1)
    return ev.assign(t0=t0, t1=t1, t2=t1 + 1, liq=liq, ear=ear)


def build_events(fins, panel, topix) -> dict[str, pd.DataFrame]:
    fs = statement_events(fins).rename(columns={"DiscDate": "disc_date"})[["code", "disc_date"]]
    return {
        "E1": attach_timing(forecast_surprise(fins), panel, topix),
        "FS": attach_timing(fs.drop_duplicates(["code", "disc_date"]).reset_index(drop=True), panel, topix),
        "YOY": attach_timing(yoy_sign(fins), panel, topix),
    }


def select(ev: pd.DataFrame, period) -> pd.DataFrame:
    """Events disclosed within the period that pass the liquidity filter."""
    ev = ev[(ev["disc_date"] >= period[0]) & (ev["disc_date"] <= period[1])]
    return ev[ev["liq"] >= LIQ_MIN].reset_index(drop=True)


# ----------------------------------------------------------------------------
# trades and the calendar-time portfolio
# ----------------------------------------------------------------------------

def trade_paths(ev: pd.DataFrame, entry_col: str, n_days: int, cost: float, panel, topix, first_pos, last_pos):
    """Per-trade results and daily return paths (trades x days from first_pos to last_pos).

    Entry at the open of the entry day, exit at the close of the n-th trading day (entry day = day 1).
    Events without a traded open on the entry day, or whose exit falls after last_pos, are dropped.
    Days without a trade keep the previous close.
    """
    o = panel["O"].to_numpy()
    c = panel["C"].ffill().to_numpy()
    col = {code: i for i, code in enumerate(panel["C"].columns)}
    to, tc = topix["O"].to_numpy(), topix["C"].to_numpy()
    rows, cols, vals, trades = [], [], [], []
    for k, (code, e) in enumerate(zip(ev["code"], ev[entry_col])):
        j = col.get(code)
        x = e + n_days - 1
        if j is None or e < first_pos or x > last_pos or not o[e, j] > 0:
            continue
        entry = o[e, j] * (1 + cost)
        prev = entry
        for t in range(e, x + 1):
            px = c[t, j] * (1 - cost) if t == x else c[t, j]
            rows.append(len(trades))
            cols.append(t - first_pos)
            vals.append(px / prev - 1)
            prev = c[t, j]
        ret = c[x, j] * (1 - cost) / entry - 1
        trades.append({"idx": k, "code": code, "entry": topix.index[e], "exit": topix.index[x],
                       "ret": ret, "excess": ret - (tc[x] / to[e] - 1)})
    shape = (len(trades), last_pos - first_pos + 1)
    mat, act = np.zeros(shape), np.zeros(shape)
    mat[rows, cols] = vals
    act[rows, cols] = 1.0
    return pd.DataFrame(trades, columns=["idx", "code", "entry", "exit", "ret", "excess"]), mat, act


def portfolio(mat: np.ndarray, act: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Daily return: equal weight across open positions; cash (0%) on days with none open."""
    w = np.zeros(mat.shape[0])
    w[rows if rows is not None else slice(None)] = 1.0
    n = w @ act
    return np.divide(w @ mat, n, out=np.zeros(mat.shape[1]), where=n > 0)


def summarize(trades: pd.DataFrame, daily: np.ndarray, n_open: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    if trades.empty:
        return {"trades": 0}
    live = np.flatnonzero(n_open > 0)
    span = slice(live[0], live[-1] + 1)
    r = pd.Series(daily[span], index=dates[span])
    equity = BUDGET * (1 + r).cumprod()
    total = equity.iloc[-1] / BUDGET - 1
    years = (r.index[-1] - r.index[0]).days / 365.25
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    monthly = trades.groupby(trades["entry"].dt.to_period("M"))["excess"].mean()
    k = n_open[span]
    return {
        "trades": int(len(trades)),
        "profit": round(float(equity.iloc[-1] - BUDGET)),
        "annual_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2),
        "max_dd_pct": round(float((equity / equity.cummax() - 1).min()) * 100, 2),
        "mean_ret_pct": round(float(trades["ret"].mean()) * 100, 3),
        "mean_excess_pct": round(float(trades["excess"].mean()) * 100, 3),
        "t_excess_month_clustered": round(float(monthly.mean() / monthly.std(ddof=1) * np.sqrt(len(monthly))), 2),
        "months": int(len(monthly)),
        "win_pct": round(float((trades["ret"] > 0).mean()) * 100, 1),
        "exposure_pct": round(float((k > 0).mean()) * 100, 1),
        "avg_open_positions": round(float(k[k > 0].mean()), 1),
        "years_positive": f"{int((by_year > 0).sum())}/{len(by_year)}",
        "by_year_pct": {int(y): round(float(v) * 100, 1) for y, v in by_year.items()},
    }


def random_profits(pool_trades: pd.DataFrame, picked: pd.DataFrame, mat, act) -> np.ndarray:
    """Profits of N_RANDOM portfolios that take, in each entry month, as many trades from the pool as the
    candidate took in that month (seed 0..N_RANDOM-1)."""
    pool_month = pool_trades["entry"].dt.to_period("M").to_numpy()
    by_month = {m: np.flatnonzero(pool_month == m) for m in np.unique(pool_month)}
    need = picked.groupby(picked["entry"].dt.to_period("M")).size()
    out = np.empty(N_RANDOM)
    for seed in range(N_RANDOM):
        rng = np.random.default_rng(seed)
        rows = np.concatenate([rng.choice(by_month[m], k, replace=False) for m, k in need.items()])
        out[seed] = (np.prod(1 + portfolio(mat, act, rows)) - 1) * BUDGET
    return out


def run_strategy(ev, pool, entry_col, n_days, cost, panel, topix, first_pos, last_pos, with_random=False):
    """Simulate `ev` (a subset of `pool`); with_random adds the matched random baseline drawn from pool."""
    pool_tr, mat, act = trade_paths(pool, entry_col, n_days, cost, panel, topix, first_pos, last_pos)
    picked_idx = set(pool.index.get_indexer(ev.index))
    rows = np.flatnonzero(pool_tr["idx"].isin(picked_idx).to_numpy())
    tr = pool_tr.iloc[rows]
    dates = topix.index[first_pos : last_pos + 1]
    s = summarize(tr, portfolio(mat, act, rows), act[rows].sum(axis=0), dates)
    if with_random and len(tr):
        rnd = random_profits(pool_tr, tr, mat, act)
        s["random_median_profit"] = round(float(np.median(rnd)))
        s["random_p95_profit"] = round(float(np.percentile(rnd, 95)))
    return s


def evaluate(period_name: str, period, price_end: pd.Timestamp, events, panel, topix, candidates) -> dict:
    topix = topix.loc[:price_end]
    panel = {k: v.loc[:price_end] for k, v in panel.items()}
    dates = topix.index
    first_pos, last_pos = int(dates.searchsorted(period[0])), len(dates) - 1
    e1 = select(events["E1"], period)
    fs = select(events["FS"], period)
    fs = fs[np.isfinite(fs["ear"])].reset_index(drop=True)
    yoy = select(events["YOY"], period)
    specs = {  # name: (events, pool for the random baseline, entry column)
        CANDIDATES["E1"]: (e1[e1["sign"] > 0], e1, "t1"),
        CANDIDATES["E2"]: (fs[fs["ear"] > 0], fs, "t2"),
    }
    diags = {
        "D1_all_statements": (fs, "t1"),
        "D2_E1_negative": (e1[e1["sign"] < 0], "t1"),
        "D2_E1_unchanged": (e1[e1["sign"] == 0], "t1"),
        "D3_reaction_neg": (fs[fs["ear"] < 0], "t2"),
        "D5_yoy_pos": (yoy[yoy["sign"] > 0], "t1"),
        "D5_yoy_neg": (yoy[yoy["sign"] < 0], "t1"),
    }
    args = (panel, topix, first_pos, last_pos)
    tpx = topix["C"].iloc[first_pos:]
    ew = panel["C"].iloc[first_pos:].pct_change().mean(axis=1).iloc[1:]
    res = {
        "period": [str(dates[first_pos].date()), str(dates[last_pos].date())],
        "B0_topix_buy_hold": {"profit": round(float(tpx.iloc[-1] / tpx.iloc[0] - 1) * BUDGET)},
        "B1_equal_weight_universe_buy_hold": {"profit": round(float(np.prod(1 + ew) - 1) * BUDGET)},
    }
    for name in candidates:
        ev, pool, ecol = specs[name]
        for cname, cost in COSTS.items():
            res[f"{name}_{cname}"] = run_strategy(ev, pool, ecol, HOLD_DAYS, cost, *args, with_random=cname == "base")
        if period_name == "dev":
            for n_days in DIAG_HOLD_DAYS:
                res[f"diag_{name}_hold{n_days}_base"] = run_strategy(ev, ev, ecol, n_days, COSTS["base"], *args)
    if period_name == "dev":
        for name, (ev, ecol) in diags.items():
            res[f"diag_{name}_base"] = run_strategy(ev, ev, ecol, HOLD_DAYS, COSTS["base"], *args)
    return res


def verdict(res: dict, name: str) -> dict:
    b, s = res[f"{name}_base"], res[f"{name}_stress"]
    checks = {
        "1_base_profit_gt_0": b["profit"] > 0,
        "2_t_excess_gt_2": b["t_excess_month_clustered"] > 2,
        "3_stress_profit_ge_0": s["profit"] >= 0,
        "4_beats_random_p95": b["profit"] > b["random_p95_profit"],
    }
    return {**checks, "promising": all(checks.values())}


def counts(events) -> dict:
    """Event counts only (no returns), for the pre-registration."""
    out = {}
    for pname, period in (("dev", DEV), ("holdout", HOLD)):
        raw = {k: v[(v["disc_date"] >= period[0]) & (v["disc_date"] <= period[1])] for k, v in events.items()}
        e1, fs, yoy = (select(events[k], period) for k in ("E1", "FS", "YOY"))
        out[pname] = {
            "E1_all": len(raw["E1"]),
            "E1_liquid": len(e1),
            "E1_liquid_by_sign": {int(k): int(v) for k, v in e1["sign"].value_counts().sort_index().items()},
            "E1_liquid_pos_by_kind": {k: int(v) for k, v in e1[e1["sign"] > 0]["kind"].value_counts().items()},
            "FS_all": len(raw["FS"]),
            "FS_liquid": len(fs),
            "FS_liquid_codes": int(fs["code"].nunique()),
            "YOY_liquid": len(yoy),
        }
        if pname == "dev":  # the reaction sign uses prices after the disclosure: never count it on holdout
            out[pname]["FS_liquid_ear_pos"] = int((fs["ear"] > 0).sum())
    liq = events["FS"]["liq"].dropna()
    out["FS_liquidity_quantiles_yen"] = {str(q): round(float(liq.quantile(q))) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return out


def main(argv: list[str]) -> None:
    topix, panel, fins = load(Path(argv[0]))
    events = build_events(fins, panel, topix)
    if "--count-only" in argv:
        out = counts(events)
    elif "--holdout" in argv:
        name = CANDIDATES[argv[argv.index("--holdout") + 1]]
        out = evaluate("holdout", HOLD, topix.index[-1], events, panel, topix, [name])
    else:
        out = evaluate("dev", DEV, DEV_PRICE_END, events, panel, topix, list(CANDIDATES.values()))
        out["verdict"] = {name: verdict(out, name) for name in CANDIDATES.values()}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
