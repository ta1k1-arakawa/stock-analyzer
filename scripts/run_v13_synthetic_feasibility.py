"""Offline V13 synthetic feasibility entry point; not research evidence."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.v13_feasibility import build_rank_population, canonical_json, diagnostics, monthly_predictions, sha256_text
from src.v13_synthetic_fixture import active_rows, stage_a_rows, synthetic_manifest

def main() -> None:
    manifest = synthetic_manifest(); stage_status, rank = build_rank_population(stage_a_rows())
    predictions, fits = monthly_predictions(active_rows())
    sample = diagnostics(predictions[:12])
    result = {"SYNTHETIC_ONLY_NOT_RESEARCH_EVIDENCE": True, "REAL_MARKET_DATA_USED": False,
        "V13_HISTORICAL_VIABILITY_RESULT": "NOT_RUN", "manifest_count": len(manifest["selected"]),
        "manifest_sha256": manifest["selected_sha256"], "stage_status": stage_status,
        "rank_eligible_count": len(rank), "prediction_count": len(predictions), "model_fits": sum(fits.values()),
        "prediction_months": sorted(fits), "diagnostics": sample}
    rendered = canonical_json(result); result["canonical_sha256"] = sha256_text(rendered)
    print(canonical_json(result))

if __name__ == "__main__": main()
