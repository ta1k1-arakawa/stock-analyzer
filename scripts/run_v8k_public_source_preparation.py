"""V8K public-source production boundary; authorization is never accepted on argv."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v8k_public_source_preparation import STUDY, V8KPublicSourceBlocked, prepare

AUTH_ENV = "V8K_PUBLIC_SOURCE_PREPARATION_AUTHORIZATION"
FAILURE_SCHEMA_VERSION = "V8K_PUBLIC_SOURCE_PREPARATION_FAILURE_V1"
STAGE = "PUBLIC_SOURCE_PREPARATION"

def _failure_report(exc: V8KPublicSourceBlocked) -> dict[str, object]:
    """Safe failure report: only the mapped public failure_class, never the raw/internal reason."""
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "study": STUDY,
        "stage": STAGE,
        "execution_result": "BLOCKED",
        "failure_class": exc.failure_class,
        "network_request_count": exc.network_request_count,
        "jpx_request_count": exc.network_request_count,
        "first_complete_payload_locked": exc.first_complete_payload_locked,
    }

def main() -> int:
    raw = os.environ.get(AUTH_ENV)
    if not raw:
        raise SystemExit("GOVERNANCE_FAILURE")
    try:
        evidence = prepare(raw_authorization=raw)
    except V8KPublicSourceBlocked as exc:
        print(json.dumps(_failure_report(exc), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit(exc.failure_class)
    finally:
        raw = ""
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0
if __name__ == "__main__":
    main()
