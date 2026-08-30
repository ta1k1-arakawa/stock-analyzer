"""Read-only CLI for the offline F1 semantic successor locator."""
from __future__ import annotations
import argparse
import sys
from src.v9_006_f1_semantic_successor_locator import canonical_json, run_locator

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--output-root", required=True)
    try:
        args = parser.parse_args(argv); result = run_locator(args.output_root)
        print(canonical_json(result)); return 0 if result["result"] == "SUCCESSOR_LOCATOR_MATCHED" else 2
    except Exception:
        print("V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR_IMPLEMENTATION_FAILURE", file=sys.stderr); return 3

if __name__ == "__main__": raise SystemExit(main())
