from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v9_006_f6_offline_child_structural_probe import ProbeBlocked, run_offline_child_structural_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-state-parent", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_offline_child_structural_probe(production_state_parent=args.production_state_parent, output_root=args.output_root)
    except ProbeBlocked as exc:
        result = {"execution_result": "BLOCKED", "status": exc.outcome, "network_request_count": 0, "raw_bytes_read_for_integrity": False, "child_content_inspected": False, "coverage_evaluated": False}
        code = 2
    except Exception:
        result = {"execution_result": "BLOCKED", "status": "IMPLEMENTATION_FAILURE", "network_request_count": 0, "raw_bytes_read_for_integrity": False, "child_content_inspected": False, "coverage_evaluated": False}
        code = 2
    else:
        code = 0
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
