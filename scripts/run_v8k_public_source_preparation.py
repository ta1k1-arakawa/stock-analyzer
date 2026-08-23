"""V8K public-source production boundary; authorization is never accepted on argv."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v8k_public_source_preparation import V8KPublicSourceBlocked, prepare

AUTH_ENV = "V8K_PUBLIC_SOURCE_PREPARATION_AUTHORIZATION"

def main() -> int:
    raw = os.environ.get(AUTH_ENV)
    if not raw:
        raise SystemExit("GOVERNANCE_FAILURE")
    try:
        evidence = prepare(raw_authorization=raw)
    except V8KPublicSourceBlocked:
        raise SystemExit("GOVERNANCE_FAILURE")
    finally:
        raw = ""
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0
if __name__ == "__main__":
    main()
