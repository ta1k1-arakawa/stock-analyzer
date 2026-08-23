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

def _safe_report(failure_class: str, network_request_count: int, first_complete_payload_locked: bool) -> dict[str, object]:
    """Safe failure report: only the mapped public failure_class, never a raw/internal reason."""
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "study": STUDY,
        "stage": STAGE,
        "execution_result": "BLOCKED",
        "failure_class": failure_class,
        "network_request_count": network_request_count,
        "jpx_request_count": network_request_count,
        "first_complete_payload_locked": first_complete_payload_locked,
    }

def main() -> int:
    raw = os.environ.get(AUTH_ENV)
    if not raw:
        print(json.dumps(_safe_report("GOVERNANCE_FAILURE", 0, False), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit("GOVERNANCE_FAILURE")
    try:
        evidence = prepare(raw_authorization=raw)
    except V8KPublicSourceBlocked as exc:
        print(json.dumps(_safe_report(exc.failure_class, exc.network_request_count, exc.first_complete_payload_locked), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit(exc.failure_class)
    except Exception:
        # Defense-in-depth: an ordinary (non-V8KPublicSourceBlocked) exception
        # must never reach the user as a raw traceback/message. Fail closed.
        print(json.dumps(_safe_report("IMPLEMENTATION_FAILURE", 0, False), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit("IMPLEMENTATION_FAILURE")
    finally:
        raw = ""
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0
if __name__ == "__main__":
    main()
