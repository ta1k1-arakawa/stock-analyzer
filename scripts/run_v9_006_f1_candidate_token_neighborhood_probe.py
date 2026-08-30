from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.v9_006_f1_candidate_token_neighborhood_probe import canonical_json, run_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.output_root)
    except Exception:
        print("V9_006_F1_CANDIDATE_NEIGHBORHOOD_IMPLEMENTATION_FAILURE", file=sys.stderr)
        return 3
    print(canonical_json(result))
    return 0 if result["diagnostic_result"] == "EVIDENCE_CAPTURED" else 2


if __name__ == "__main__": raise SystemExit(main())
