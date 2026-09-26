"""EXP-006: LightGBM as a value-trap filter on top of the EXP-003 rule (walk-forward).

Pre-registration: docs/experiments/EXP-006_PREREGISTRATION.md
Usage: python experiments/exp006_trap_filter.py <cache_dir>
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
from exp005_walkforward import COSTS, GAP, TEST, TRAIN_START  # noqa: E402

TOP_N, DROP_N, TRAP_PCT = 30, 10, 0.3


def main(cache: Path) -> None:
    import lightgbm as lgb

    topix, panel, fins = E3.load(cache)
    dates = topix.index
    feats, ok = F.features(panel, fins, dates)
    comp = E3.signals(panel, fins, dates)["composite"]
    top = comp.rank(axis=1, ascending=False, method="first") <= TOP_N  # the rule's top 30 each day
    o = panel["O"]
    fwd = o.shift(-(F.HOLD_DAYS + 1)) / o.shift(-1) - 1
    fwd_rank = fwd.where(ok).rank(axis=1, pct=True)
    trap = (fwd_rank < TRAP_PCT).astype(float).where(fwd_rank.notna())

    prob = pd.DataFrame(np.nan, index=dates, columns=comp.columns)
    first = int(dates.searchsorted(TRAIN_START))
    for year in range(TEST[0].year, TEST[1].year + 1):
        test_idx = np.flatnonzero((dates.year == year) & (dates >= TEST[0]) & (dates <= TEST[1]))
        last_train = test_idx[0] - GAP
        rows = []
        for t in range(first, last_train + 1, 5):
            d = dates[t]
            x = pd.DataFrame({k: v.loc[d] for k, v in feats.items()})[top.loc[d]]
            x["y"] = trap.loc[d]
            rows.append(x.dropna(subset=["y"]))
        data = pd.concat(rows)
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                                   deterministic=True, force_row_wise=True, verbose=-1)
        model.fit(data[F.FEATURES], data["y"].astype(int))
        for t in test_idx:
            d = dates[t]
            x = pd.DataFrame({k: feats[k].loc[d] for k in F.FEATURES})[top.loc[d]]
            if len(x):
                prob.loc[d, x.index] = model.predict_proba(x[F.FEATURES])[:, 1]
        print(f"{year}: trained on {len(data)} rows (trap share {data['y'].mean():.2f}) up to {dates[last_train].date()}",
              file=sys.stderr)

    drop = prob.rank(axis=1, ascending=False, method="first") <= DROP_N  # the 10 most trap-like of the top 30
    filtered = comp.mask(drop)

    start, end = int(dates.searchsorted(TEST[0])), int(dates.searchsorted(TEST[1], side="right")) - 1
    span = dates[start : end + 1]
    o_np, c_np = o.to_numpy(), panel["C"].ffill().to_numpy()
    res = {"period": [str(span[0].date()), str(span[-1].date())],
           "B0_topix_buy_hold": {"profit": round(float(topix["C"].iloc[end] / topix["O"].iloc[start] - 1) * E3.BUDGET)}}
    values = {}
    for name, sc in (("R", comp), ("F", filtered)):
        for cname, cost in COSTS.items():
            _, v, ch = E3.simulate(sc.to_numpy(), o_np, c_np, start, end, cost)
            res[f"{name}_{cname}"] = E3.summarize(v, span, topix, ch)
            values[(name, cname)] = pd.Series(v, index=span).resample("ME").last()
    rf, rr = values[("F", "base")], values[("R", "base")]
    diff = rf / rf.shift(1).fillna(E3.BUDGET) - rr / rr.shift(1).fillna(E3.BUDGET)
    res["monthly_diff_F_minus_R"] = {"mean_pct": round(float(diff.mean()) * 100, 3),
                                     "t": round(float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))), 2),
                                     "months": int(len(diff)), "months_F_better_pct": round(float((diff > 0).mean()) * 100, 1)}
    sl = slice(span[0], span[-1])
    dropped = fwd.loc[sl][drop.loc[sl]].stack()
    kept = fwd.loc[sl][top.loc[sl] & ~drop.loc[sl] & prob.loc[sl].notna()].stack()
    res["reference_20d_return_pct"] = {"dropped_mean": round(float(dropped.mean()) * 100, 3),
                                       "kept_mean": round(float(kept.mean()) * 100, 3)}
    res["verdict"] = {
        "1_F_profit_gt_R_base": res["F_base"]["profit"] > res["R_base"]["profit"],
        "2_t_monthly_diff_gt_2": res["monthly_diff_F_minus_R"]["t"] > 2,
        "3_F_profit_gt_R_stress": res["F_stress"]["profit"] > res["R_stress"]["profit"],
    }
    res["verdict"]["F_better"] = all(res["verdict"].values())
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
