"""Validate the repository-safe Issue #107 result without opening local attempt state."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import check_current_protected_environment as current
from scripts import v13_xlsx_successor_freeze as freeze
from scripts import v13_xlsx_successor_isolated_realization as realization

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "docs/v13/V13_XLSX_SUCCESSOR_ISOLATED_REALIZATION_SAFE_RESULT.json"


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def _exact(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def validate_result(*, result_path: Path = RESULT) -> dict:
    # This validator reads only Git-reviewed artifacts and the supplied result file.
    frozen = freeze.validate_freeze()
    if frozen["status"] != "PASS" or frozen["candidate_package_count"] != 29 or frozen["current_authority_package_count"] != 27:
        raise ValueError("FROZEN_LOCKS_INVALID")

    authorization = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:docs/v13/V13_XLSX_SUCCESSOR_ISOLATED_REALIZATION_AUTHORIZATION.json"],
        capture_output=True, check=True,
    ).stdout
    if realization.blob(authorization) != realization.AUTH_BLOB:
        raise ValueError("AUTHORIZATION_BLOB_MISMATCH")
    auth = json.loads(authorization, object_pairs_hook=_unique)
    if auth["candidate_lock"]["sha256"] != freeze.LOCK_SHA256 or auth["freeze_record"]["source_safe_evidence_sha256"] != freeze.SOURCE_EVIDENCE_SHA256 or auth["current_authority_lock"]["sha256"] != current.CURRENT_AUTHORITY_LOCK_SHA256:
        raise ValueError("AUTHORIZATION_BINDING_MISMATCH")

    expected = {
        "schema_version": "V13_XLSX_SUCCESSOR_ISOLATED_REALIZATION_SAFE_RESULT_V1",
        "artifact_role": "ISOLATED_CANDIDATE_REALIZATION_RESULT_NOT_PROMOTION",
        "source_issue": 105,
        "source_safe_result_comment_id": 5848040057,
        "source_remediation_issue": 106,
        "source_remediation_gpt_result": "PASS",
        "execution_head": "2f25f8961ed6ad115e9faea10d4edccb011e7056",
        "authorization_blob": realization.AUTH_BLOB,
        "candidate_lock_sha256": freeze.LOCK_SHA256,
        "source_safe_evidence_sha256": freeze.SOURCE_EVIDENCE_SHA256,
        "wheel_manifest_sha256": freeze.MANIFEST_SHA256,
        "status": "PASS_CONSUMED",
        "authorization_consumed": True,
        "authorization_reusable": False,
        "isolated_candidate_realized": True,
        "package_count": 29,
        "wheel_count": 29,
        "installed_package_count": 29,
        "production_xlsx_readiness": "PASS",
        "network_requests": 0,
        "wheel_downloads": 0,
        "process_exit_code": 0,
        "active_environment_mutated": False,
        "current_authority_package_count": 27,
        "current_authority_preserved": True,
        "successor_promoted": False,
        "successor_promotion_authorized": False,
        "current_authority_change_authorized": False,
        "jpx_yahoo_requests_authorized": False,
        "private_reads_authorized": False,
        "selected_500_authorized": False,
        "model_fits_authorized": False,
        "backtests_authorized": False,
        "a_to_q_authorized": False,
        "paper_trading_authorized": False,
        "real_trading_authorized": False,
        "future_profitability_established": False,
    }
    result = json.loads(result_path.read_bytes(), object_pairs_hook=_unique)
    if not _exact(result, expected):
        raise ValueError("SAFE_RESULT_SCHEMA_OR_BINDING_MISMATCH")
    return {"status": "PASS", "candidate_package_count": 29,
            "current_authority_package_count": 27, "successor_promoted": False}


if __name__ == "__main__":
    print(json.dumps(validate_result(), sort_keys=True))
