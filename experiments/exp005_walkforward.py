"""EXP-005: walk-forward test of the EXP-004 LightGBM ranker (M) against the EXP-003 rule (R).

Pre-registration: docs/experiments/EXP-005_PREREGISTRATION.md
Usage: python experiments/exp005_walkforward.py <cache_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp003_rotation as E3  # noqa: E402
import exp004_forward as F  # noqa: E402

TEST = (pd.Timestamp("2019-01-04"), pd.Timestamp("2026-09-25"))
TRAIN_START = pd.Timestamp("2017-01-04")
GAP = 22  # trading days between the last training signal and the first test day
COSTS = {"base": 0.0003, "stress": 0.0010}


def walk_forward_scores(feats, target, ok, dates) -> pd.DataFrame:
    import lightgbm as lgb

    scores = pd.DataFrame(np.nan, index=dates, columns=ok.columns)
    first = int(dates.searchsorted(TRAIN_START))
    for year in range(TEST[0].year, TEST[1].year + 1):
        test_idx = np.flatnonzero((dates.year == year) & (dates >= TEST[0]) & (dates <= TEST[1]))
        last_train = test_idx[0] - GAP
        rows = []
        for t in range(first, last_train + 1, 5):
            d = dates[t]
            x = pd.DataFrame({k: v.loc[d] for k, v in feats.items()})
            x["y"] = target.loc[d]
            rows.append(x.dropna(subset=["y"]))
        data = pd.concat(rows)
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                                  deterministic=True, force_row_wise=True, verbose=-1)
        model.fit(data[F.FEATURES], data["y"])
        for t in test_idx:
            d = dates[t]
            x = pd.DataFrame({k: feats[k].loc[d] for k in F.FEATURES})[ok.loc[d]]
            if len(x):
                scores.loc[d, x.index] = model.predict(x[F.FEATURES])
        print(f"{year}: trained on {len(data)} rows up to {dates[last_train].date()}", file=sys.stderr)
    return scores


def monthly_ic(score: pd.DataFrame, fwd: pd.DataFrame, dates) -> pd.Series:
    """Mean Spearman correlation per month between score at close t and the open(t+1)->open(t+21) return."""
    ics = {}
    for d in dates:
        s, r = score.loc[d], fwd.loc[d]
        m = s.notna() & r.notna()
        if m.sum() >= 20:
            ics[d] = s[m].rank().corr(r[m].rank())
    ic = pd.Series(ics)
    return ic.groupby(ic.index.to_period("M")).mean()


def main(cache: Path) -> None:
    topix, panel, fins = E3.load(cache)
    dates = topix.index
    feats, ok = F.features(panel, fins, dates)
    o = panel["O"]
    fwd = o.shift(-(F.HOLD_DAYS + 1)) / o.shift(-1) - 1
    target = fwd.where(ok).rank(axis=1, pct=True)
    m_score = walk_forward_scores(feats, target, ok, dates)
    r_score = E3.signals(panel, fins, dates)["composite"]

    start, end = int(dates.searchsorted(TEST[0])), int(dates.searchsorted(TEST[1], side="right")) - 1
    span = dates[start : end + 1]
    o_np, c_np = o.to_numpy(), panel["C"].ffill().to_numpy()
    res = {"period": [str(span[0].date()), str(span[-1].date())],
           "B0_topix_buy_hold": {"profit": round(float(topix["C"].iloc[end] / topix["O"].iloc[start] - 1) * E3.BUDGET)}}
    values = {}
    for name, sc in (("R", r_score), ("M", m_score)):
        for cname, cost in COSTS.items():
            _, v, ch = E3.simulate(sc.to_numpy(), o_np, c_np, start, end, cost)
            res[f"{name}_{cname}"] = E3.summarize(v, span, topix, ch)
            values[(name, cname)] = pd.Series(v, index=span)
    mv = {k: s.resample("ME").last() for k, s in values.items()}
    diff = (mv[("M", "base")] / mv[("M", "base")].shift(1).fillna(E3.BUDGET)
            - mv[("R", "base")] / mv[("R", "base")].shift(1).fillna(E3.BUDGET))
    res["monthly_diff_M_minus_R"] = {"mean_pct": round(float(diff.mean()) * 100, 3),
                                     "t": round(float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))), 2),
                                     "months": int(len(diff)), "months_M_better_pct": round(float((diff > 0).mean()) * 100, 1)}
    ic = {n: monthly_ic(sc, fwd, span) for n, sc in (("R", r_score), ("M", m_score))}
    res["reference_monthly_rank_ic"] = {n: {"mean": round(float(s.mean()), 4),
                                            "t": round(float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))), 2)}
                                        for n, s in ic.items()}
    res["verdict"] = {
        "1_M_profit_gt_R_base": res["M_base"]["profit"] > res["R_base"]["profit"],
        "2_t_monthly_diff_gt_2": res["monthly_diff_M_minus_R"]["t"] > 2,
        "3_M_profit_gt_R_stress": res["M_stress"]["profit"] > res["R_stress"]["profit"],
    }
    res["verdict"]["M_better"] = all(res["verdict"].values())
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
