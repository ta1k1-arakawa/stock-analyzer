"""Execute the complete deterministic synthetic-only V13 path; not research evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.v13_feasibility import canonical_json, sha256_text, monthly_predictions, rank_candidates, simulate, trade_metrics, diagnostics, linear_percentile
from src.v13_synthetic_fixture import synthetic_manifest, synthetic_calendar, raw_ohlcv, active_rows

def main():
    manifest=synthetic_manifest(); cal=synthetic_calendar(); prices=raw_ohlcv(); rows=active_rows(); predictions,fits=monthly_predictions(rows)
    dates=sorted({r["signal"] for r in predictions}); strategies={}
    for strategy in ("LIGHTGBM","RIDGE","SECTOR_REL_REVERSAL_1D","SECTOR_REL_MOMENTUM_20D"):
        rankings={d:rank_candidates([r for r in predictions if r["signal"]==d],strategy) for d in dates}
        strategies[strategy]={"base":trade_metrics(simulate(cal,prices,rankings,.001)),"stress":trade_metrics(simulate(cal,prices,rankings,.002))}
    random_profits=[]
    for seed in range(202609220000,202609220500):
        rankings={d:rank_candidates([r for r in predictions if r["signal"]==d],"RANDOM_500",d,seed) for d in dates}
        random_profits.append(trade_metrics(simulate(cal,prices,rankings,.001))["total_net_profit"])
    result={"SYNTHETIC_ONLY_NOT_RESEARCH_EVIDENCE":True,"REAL_MARKET_DATA_USED":False,"V13_HISTORICAL_VIABILITY_RESULT":"NOT_RUN","manifest_count":len(manifest["selected"]),"manifest_sha256":manifest["selected_sha256"],"raw_ohlcv_rows":len(prices),"stage_a_from_raw":True,"stage_b_relative_only":True,"stage_c_transform_once":True,"prediction_months":sorted(fits),"synthetic_model_fits":sum(fits.values()),"strategies":strategies,"RANDOM_500":{"seed_first":202609220000,"seed_last":202609220499,"count":len(random_profits),"base_profit_p95_linear":linear_percentile(random_profits,95)},"diagnostics":diagnostics(predictions),"Q_keys":[f"Q{i}" for i in range(1,13)]}
    result["canonical_sha256"]=sha256_text(canonical_json(result)); print(canonical_json(result))
if __name__=="__main__": main()
