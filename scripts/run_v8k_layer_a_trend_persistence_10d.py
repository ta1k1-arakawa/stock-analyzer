from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.v8k_layer_a_trend_persistence_10d import run_cache_measurement
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--measure-cache",action="store_true");p.add_argument("--evaluation-cache");p.add_argument("--output-dir");a=p.parse_args(argv)
 if not a.measure_cache: raise SystemExit("MEASURE_CACHE_FLAG_REQUIRED")
 if not a.evaluation_cache or not a.output_dir: raise SystemExit("EVALUATION_CACHE_AND_OUTPUT_DIR_REQUIRED")
 run_cache_measurement(Path(a.evaluation_cache),Path(a.output_dir),Path(__file__).resolve().parents[1]);return 0
if __name__=="__main__":raise SystemExit(main())
