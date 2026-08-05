"""V5-B runner.  Production evaluation is intentionally opt-in and disabled here."""
from __future__ import annotations
import argparse, tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v5_b_candidate_ranker import synthetic_artifacts, atomic_write

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--synthetic-smoke-test",action="store_true")
    ap.add_argument("--evaluate-cache",action="store_true")
    ap.add_argument("--training-cache"); ap.add_argument("--evaluation-cache"); ap.add_argument("--output-dir"); ap.add_argument("--confirmation")
    a=ap.parse_args()
    if a.evaluate_cache:
        raise SystemExit("FORMAL_EVALUATION_DISABLED: evaluation cache is not acquired or executed in this implementation turn")
    if not a.synthetic_smoke_test: raise SystemExit("use --synthetic-smoke-test")
    # Smoke artifacts live only in a temporary directory and are removed on exit.
    with tempfile.TemporaryDirectory(prefix="v5b-smoke-") as td:
        out=Path(td)/"output"
        atomic_write(out, synthetic_artifacts(), Path.cwd())
        first={p.name:p.read_bytes() for p in out.iterdir()}
        out2=Path(td)/"output2"; atomic_write(out2, synthetic_artifacts(), Path.cwd())
        second={p.name:p.read_bytes() for p in out2.iterdir()}
        if first!=second: raise SystemExit("TWO_PASS_ARTIFACT_MISMATCH")
        print("V5-B synthetic smoke PASS; two-pass byte-identical; artifacts=4")
    return 0
if __name__ == "__main__": raise SystemExit(main())
