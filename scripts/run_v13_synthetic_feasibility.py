"""Thin CLI wrapper around the single production synthetic orchestrator."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v13_feasibility import canonical_json, run_synthetic_feasibility


if __name__ == "__main__":
    print(canonical_json(run_synthetic_feasibility()))
