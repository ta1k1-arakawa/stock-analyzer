from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.v9_006_f6_date_year_coverage_parser import CoverageBlocked, run_date_year_coverage_parser
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--production-state-parent", required=True); parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try: result = run_date_year_coverage_parser(production_state_parent=args.production_state_parent, output_root=args.output_root); code = 0
    except CoverageBlocked as exc: result = {"execution_result":"BLOCKED", **exc.evidence}; code = 2
    except Exception: result = {"execution_result":"BLOCKED", "status":"IMPLEMENTATION_FAILURE", "structural_profile_sha256":None, "structural_profile_hash_verified":False, "date_column_ordinals":[4,6], "raw_bytes_read_for_integrity":"unknown", "child_content_inspected":False, "date_year_value_read":False, "coverage_evaluated":False, "coverage_result_accepted":False, "network_request_count":0}; code = 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"))); return code
if __name__ == "__main__": raise SystemExit(main())
