"""V8K public-source production boundary; authorization is never accepted on argv."""
from __future__ import annotations
import os
from src.v8k_public_source_preparation import V8KPublicSourceBlocked, production_provenance

AUTH_ENV = "V8K_PUBLIC_SOURCE_PREPARATION_AUTHORIZATION"

def main() -> int:
    raw = os.environ.get(AUTH_ENV)
    if not raw:
        raise SystemExit("GOVERNANCE_FAILURE")
    try:
        production_provenance()
    except V8KPublicSourceBlocked:
        raise SystemExit("GOVERNANCE_FAILURE")
    raise SystemExit("GOVERNANCE_FAILURE: reviewed production fetch/parser dependencies require a later authorized invocation")
if __name__ == "__main__":
    main()
