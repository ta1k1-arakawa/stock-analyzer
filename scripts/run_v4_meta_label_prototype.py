"""Safe entry point for the V4 pre-registered prototype.

The standard mode refuses network/evaluation.  Only --preflight-only is enabled
until a separately authorised execution phase supplies external data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.v4_meta_label import preflight

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.preflight_only:
        raise SystemExit("NETWORK_AND_EVALUATION_DISABLED: use --preflight-only")
    result = preflight(ROOT, args.cache_dir, args.output_dir)
    print(json.dumps({"decision": "PREFLIGHT_PASS", **result}, ensure_ascii=False, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
