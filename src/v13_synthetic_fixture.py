"""Deterministic, visibly artificial raw inputs for the offline V13 probe."""
from __future__ import annotations
from datetime import date, timedelta
import math
from .v13_feasibility import SessionCalendar, select_universe, sha256_text

def synthetic_calendar() -> SessionCalendar:
    days=[]; cursor=date(2015,1,1)
    while cursor<=date(2020,3,31):
        if cursor.weekday()<5: days.append(cursor)
        cursor+=timedelta(days=1)
    return SessionCalendar(tuple(days))

def synthetic_manifest() -> dict:
    eligible=[f"{n:04d}" for n in range(1000,1705)]; excluded=["1001","1011","1021"]
    seed="V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON|f9c38ad771710ffd157ac4fad0da15185db82707"; selected=select_universe(eligible,excluded,seed)
    return {"seed":seed,"eligible":eligible,"excluded":excluded,"selected":selected,"selected_sha256":sha256_text("|".join(selected))}

def synthetic_metadata():
    return {c:("SYNTHETIC_ALPHA" if i<6 else "SYNTHETIC_BETA") for i,c in enumerate(synthetic_manifest()["selected"][:12])}

def raw_ohlcv(variant="success"):
    out={}; cal=synthetic_calendar()
    for i,d in enumerate(cal.sessions):
        for j,(c,_) in enumerate(synthetic_metadata().items()):
            close=(850+j*31)*(1+.00028*i+.025*math.sin(i*.071+j*.43)); opening=close*(1+.003*math.sin(i*.13+j))
            out[c,d]={"open":opening,"high":max(opening,close)*1.01,"low":min(opening,close)*.99,"close":close,"volume":220000+j*8000.,"adj_open":opening,"adj_close":close}
    c,d=next(iter(out))
    if variant=="affordability":out[c,d]["close"]=3001
    elif variant=="missing_open":out[c,d]["adj_open"]=float("nan")
    elif variant=="missing_exit":out[c,d]["close"]=float("nan")
    elif variant=="no_rank":
        for r in out.values(): r["close"]=r["adj_close"]=1000.
    return out

def stage_a_rows():
    from .v13_feasibility import stage_a_from_raw
    cal=synthetic_calendar(); return stage_a_from_raw(cal,raw_ohlcv(),synthetic_metadata(),date(2020,1,2))

def active_rows():
    """Raw-derived labeled rows: 2016+ training and Jan/Feb 2020 prediction."""
    from .v13_feasibility import stage_a_from_raw, build_rank_population, base_target
    cal=synthetic_calendar(); prices=raw_ohlcv(); meta=synthetic_metadata(); rows=[]
    # A sparse, deterministic expanding sample is sufficient for the synthetic fit contract.
    signals=[d for i,d in enumerate(cal.sessions) if i>=252 and ((d.year in range(2016,2020) and d.month in (1,4,7,10)) or (d.year==2020 and d.month in (1,2)))]
    for d in signals:
        exit_day=cal.plus(d,3)
        if exit_day is None: continue
        status,pop=build_rank_population(stage_a_from_raw(cal,prices,meta,d))
        if status!="OK": continue
        for r in pop:
            entry=cal.plus(d,1); target=base_target(prices[r["code"],entry]["open"],prices[r["code"],exit_day]["close"])
            if target is not None: rows.append(dict(r,exit=exit_day,target=target))
    return rows
