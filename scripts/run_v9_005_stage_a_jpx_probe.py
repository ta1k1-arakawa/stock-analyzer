"""V9_005 Stage-A production boundary; the confirmation token is never
accepted on argv -- only via the V9_005_STAGE_A_CONFIRMATION environment
variable, set at point of use by the atomic PowerShell entrypoint after its
own preflight passes. Running this script directly against a real network
is NOT authorized by the V9_006 implementation task; it exists so a later,
separately authorized run can invoke it."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v9_005_stage_a_jpx_probe import (  # noqa: E402
    CHATGPT_DECISION_REQUIRED,
    FetchResult,
    STAGE,
    STUDY,
    V9005StageABlocked,
    run_stage_a,
)

CONFIRMATION_ENV = "V9_005_STAGE_A_CONFIRMATION"
FAILURE_SCHEMA_VERSION = "V9_005_STAGE_A_FAILURE_V1"
PRODUCTION_USER_AGENT = "V9-005-Stage-A-JPX-Probe/1.0"


def _production_fetcher(url: str) -> FetchResult:
    request = urllib.request.Request(url, headers={"User-Agent": PRODUCTION_USER_AGENT})
    response = urllib.request.urlopen(request, timeout=30)
    try:
        payload = response.read()
        final_url = getattr(response, "url", url)
        http_status = getattr(response, "status", None)
        if http_status is None:
            getcode = getattr(response, "getcode", None)
            http_status = getcode() if callable(getcode) else None
        if isinstance(http_status, bool) or not isinstance(http_status, int):
            raise RuntimeError("missing observed HTTP status")
        return FetchResult(payload=payload, resolved_url=final_url, http_status=http_status)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def _safe_report(failure_class: str, network_request_count: int, *, reason: str | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "study": STUDY,
        "stage": STAGE,
        "execution_result": "BLOCKED",
        "failure_class": failure_class,
        "network_request_count": network_request_count,
    }
    if failure_class == CHATGPT_DECISION_REQUIRED:
        # A governance/methodology stop, not a source/data feasibility
        # result: surface the explicit STATUS/reason contract so an
        # operator or reviewer sees this is not a real probe outcome.
        report["status"] = CHATGPT_DECISION_REQUIRED
        report["reason"] = reason or CHATGPT_DECISION_REQUIRED
    return report


def _print(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V9_005 Stage-A free JPX metadata probe")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    confirmation = os.environ.get(CONFIRMATION_ENV)
    if not confirmation:
        _print(_safe_report("GOVERNANCE_FAILURE", 0))
        return 2
    try:
        summary = run_stage_a(
            output_root=args.output_root,
            repo_root=args.repo_root,
            confirmation=confirmation,
            fetcher=_production_fetcher,
            sleep=time.sleep,
            clock=_utc_clock,
        )
    except V9005StageABlocked as exc:
        _print(_safe_report(exc.failure_class, exc.network_request_count, reason=exc.reason))
        return 2
    except Exception:
        # Defense-in-depth: an ordinary (non-V9005StageABlocked) exception
        # must never reach the user as a raw traceback/message.
        _print(_safe_report("IMPLEMENTATION_FAILURE", 0))
        return 2
    finally:
        confirmation = ""
    _print(summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
