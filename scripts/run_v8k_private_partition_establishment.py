"""V8K private-partition-establishment production boundary; authorization is
never accepted on argv."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v8k_private_partition_establishment import (
    STUDY,
    V8KPrivatePartitionBlocked,
    establish_private_partition,
)

AUTH_ENV = "V8K_PRIVATE_PARTITION_GENERATION_AUTHORIZATION"
FAILURE_SCHEMA_VERSION = "V8K_PRIVATE_PARTITION_ESTABLISHMENT_FAILURE_V1"
STAGE = "PRIVATE_PARTITION_ESTABLISHMENT"


def _safe_report(failure_class: str) -> dict[str, object]:
    """Safe failure report: only the mapped public failure_class, never a
    raw/internal reason, traceback, private path, or raw authorization."""
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "study": STUDY,
        "stage": STAGE,
        "execution_result": "BLOCKED",
        "failure_class": failure_class,
    }


def main() -> int:
    raw = os.environ.get(AUTH_ENV)
    if not raw:
        print(json.dumps(_safe_report("GOVERNANCE_FAILURE"), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit("GOVERNANCE_FAILURE")
    try:
        evidence = establish_private_partition(raw_authorization=raw)
    except V8KPrivatePartitionBlocked as exc:
        print(json.dumps(_safe_report(exc.failure_class), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit(exc.failure_class)
    except Exception:
        # Defense-in-depth: an ordinary (non-V8KPrivatePartitionBlocked)
        # exception must never reach the user as a raw traceback/message.
        print(json.dumps(_safe_report("IMPLEMENTATION_FAILURE"), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        raise SystemExit("IMPLEMENTATION_FAILURE")
    finally:
        raw = ""
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    main()
