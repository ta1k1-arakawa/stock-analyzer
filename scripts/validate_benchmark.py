from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark import validate_snapshot
p=argparse.ArgumentParser(); p.add_argument("--benchmark",default="data/benchmark"); a=p.parse_args(); m=validate_snapshot(a.benchmark)
print(json.dumps({"status":"PASS","snapshot_id":m["snapshot_id"],"snapshot_hash":m["snapshot_hash"]},indent=2))
