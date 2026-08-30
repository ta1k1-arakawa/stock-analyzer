"""Read-only CLI for the offline F1 semantic successor locator."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_FAILURE_MARKER = "V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR_IMPLEMENTATION_FAILURE"


def _load_locator():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.v9_006_f1_semantic_successor_locator import canonical_json, run_locator
    return canonical_json, run_locator

def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--output-root", required=True)
        args = parser.parse_args(argv)
        canonical_json, run_locator = _load_locator()
        result = run_locator(args.output_root)
        print(canonical_json(result)); return 0 if result["result"] == "SUCCESSOR_LOCATOR_MATCHED" else 2
    except Exception:
        print(_FAILURE_MARKER, file=sys.stderr); return 3

if __name__ == "__main__": raise SystemExit(main())
