from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import run_v9_009_t0_top1_kill_screen as bridge


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v9_009_t0_top1_kill_screen.py"


def _valid_artifact() -> dict:
    return json.loads((ROOT / bridge.CALENDAR_ARTIFACT_NAME).read_text(encoding="utf-8"))


def _valid_receipt() -> dict:
    return json.loads((ROOT / bridge.SAFE_RECEIPT_NAME).read_text(encoding="utf-8"))


def _main_args(tmp_path: Path) -> list[str]:
    return [
        "--training-cache",
        str(tmp_path / "training"),
        "--evaluation-cache",
        str(tmp_path / "evaluation"),
        "--universe-csv",
        str(tmp_path / "universe.csv"),
        "--implementation-sha",
        "a" * 40,
    ]


def _preflight_git_values(implementation_sha: str) -> dict[tuple[str, ...], str]:
    return {
        ("remote", "get-url", "origin"): bridge.EXPECTED_REPOSITORY_URL,
        ("rev-parse", "--abbrev-ref", "HEAD"): bridge.AUTHORITATIVE_BRANCH,
        ("rev-parse", "HEAD"): implementation_sha,
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("rev-parse", f"HEAD:{bridge.BRIDGE_DESIGN_PATH}"): bridge.BRIDGE_DESIGN_GIT_BLOB_SHA1,
    }


def test_checked_in_artifact_and_receipt_bind_without_calendar_generation():
    dates = bridge.load_fixed_calendar_binding(ROOT, "a" * 40)
    assert len(dates) == bridge.EXPECTED_TRADING_DATE_COUNT
    assert dates[0] == "2017-01-04"
    assert dates[-1] == "2026-01-30"


def test_artifact_and_receipt_have_exact_frozen_keys_and_values():
    artifact = _valid_artifact()
    receipt = _valid_receipt()
    assert set(artifact) == bridge._ARTIFACT_KEYS
    assert set(receipt) == bridge._RECEIPT_KEYS
    assert bridge.validate_canonical_artifact(artifact)[0] == "2017-01-04"
    assert bridge.validate_safe_receipt(receipt)["status"] == "PASS"


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "calendar_method",
        "calendar_package",
        "calendar_package_version",
        "upstream_commit",
        "calendar_source_blob",
        "holiday_source_blob",
        "calendar_name",
        "runtime_environment_lock_sha256",
        "coverage_start",
        "coverage_end",
        "python_version",
        "pandas_version",
        "generator_implementation_git_sha",
    ],
)
def test_artifact_fixed_provenance_mismatch_is_input_binding_failure(field):
    artifact = _valid_artifact()
    artifact[field] = "0" * 64 if "sha" in field or "blob" in field else "wrong"
    with pytest.raises(bridge.T0DataIncompatible):
        bridge.validate_canonical_artifact(artifact)


def test_artifact_self_digest_and_dates_are_validated():
    artifact = _valid_artifact()
    artifact["canonical_calendar_sha256"] = "0" * 64
    with pytest.raises(bridge.T0DataIncompatible):
        bridge.validate_canonical_artifact(artifact)
    artifact = _valid_artifact()
    artifact["trading_dates"][0] = "2017-1-04"
    with pytest.raises(bridge.T0DataIncompatible):
        bridge.validate_canonical_artifact(artifact)


@pytest.mark.parametrize(
    "field,value",
    [
        ("canonical_calendar_sha256", "0" * 64),
        ("trading_date_count", 0),
        ("calendar_artifact_created", False),
        ("anchor_2020_10_01", "ELIGIBLE"),
        ("anchor_2020_10_02", "INELIGIBLE"),
    ],
)
def test_receipt_fixed_pass_semantics_are_validated(field, value):
    receipt = _valid_receipt()
    receipt[field] = value
    with pytest.raises(bridge.T0DataIncompatible):
        bridge.validate_safe_receipt(receipt)


@pytest.mark.parametrize(
    "field",
    [
        "canonical_calendar_sha256",
        "trading_date_count",
        "runtime_environment_lock_sha256",
        "generator_implementation_git_sha",
    ],
)
def test_artifact_receipt_cross_binding_mismatch_is_rejected(field):
    artifact = _valid_artifact()
    receipt = _valid_receipt()
    receipt[field] = "0" * 64 if "sha" in field else 0
    with pytest.raises(bridge.T0DataIncompatible):
        bridge.validate_calendar_cross_binding(artifact, receipt)


def test_valid_repo_provenance_accepts_reviewed_head(monkeypatch):
    implementation_sha = "a" * 40
    values = _preflight_git_values(implementation_sha)
    monkeypatch.setattr(bridge, "_git_value", lambda _root, *parts: values[parts])
    bridge.validate_repository_preflight(ROOT, implementation_sha)


def test_wrong_head_is_governance_failure(monkeypatch):
    implementation_sha = "a" * 40
    values = _preflight_git_values(implementation_sha)
    values[("rev-parse", "HEAD")] = "b" * 40
    monkeypatch.setattr(bridge, "_git_value", lambda _root, *parts: values[parts])
    with pytest.raises(bridge.GovernanceFailure):
        bridge.validate_repository_preflight(ROOT, implementation_sha)


def test_contract_failure_stops_before_run_from_cache_and_emits_no_verdict(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(bridge, "validate_repository_preflight", lambda *_: None)
    monkeypatch.setattr(
        bridge,
        "load_fixed_calendar_binding",
        lambda *_: (_ for _ in ()).throw(bridge.T0DataIncompatible("CONTRACT")),
    )
    monkeypatch.setattr(bridge, "run_from_cache", lambda *args: calls.append(args))
    assert bridge.main(_main_args(tmp_path)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["T0_RESULT"] == "NO_VERDICT_DATA_INCOMPATIBLE"
    assert result["validation"]["cache_identity"] is False
    assert result["validation"]["exact_calendar_grid"] is False
    assert calls == []


def test_governance_failure_stops_without_t0_json_or_cache(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(
        bridge,
        "validate_repository_preflight",
        lambda *_: (_ for _ in ()).throw(bridge.GovernanceFailure("DIRTY")),
    )
    monkeypatch.setattr(bridge, "run_from_cache", lambda *args: calls.append(args))
    assert bridge.main(_main_args(tmp_path)) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == bridge.PREFLIGHT_FAILURE + "\n"
    assert calls == []


def test_unexpected_bridge_exception_is_implementation_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bridge, "validate_repository_preflight", lambda *_: None)
    monkeypatch.setattr(
        bridge,
        "load_fixed_calendar_binding",
        lambda *_: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    assert bridge.main(_main_args(tmp_path)) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == bridge.IMPLEMENTATION_FAILURE + "\n"


def test_successful_binding_passes_only_fixed_dates_to_cache_runner(monkeypatch, tmp_path, capsys):
    seen = {}
    fixed_dates = ["2017-01-04", "2020-10-02"]
    monkeypatch.setattr(bridge, "validate_repository_preflight", lambda *_: None)
    monkeypatch.setattr(bridge, "load_fixed_calendar_binding", lambda *_: fixed_dates)

    def fake_run(*args):
        seen["args"] = args
        return bridge.make_safe_result(
            "CONTINUE",
            args[-1],
            bridge.synthetic_provenance(),
            cache_identity=False,
            exact_semantics=False,
        )

    monkeypatch.setattr(bridge, "run_from_cache", fake_run)
    assert bridge.main(_main_args(tmp_path)) == 0
    assert seen["args"][3] is fixed_dates
    assert json.loads(capsys.readouterr().out)["T0_RESULT"] == "CONTINUE"


def test_cli_has_no_caller_calendar_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--calendar-file" not in source
    assert "--implementation-sha" in source
    with pytest.raises(SystemExit) as error:
        bridge._arguments(
            [
                "--training-cache",
                "t",
                "--evaluation-cache",
                "e",
                "--universe-csv",
                "u",
                "--implementation-sha",
                "a" * 40,
                "--calendar-file",
                "arbitrary.json",
            ]
        )
    assert error.value.code == 2


def test_production_surface_has_no_real_calendar_import_or_provider_branch():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import pandas_market_calendars" not in source
    assert "from pandas_market_calendars" not in source
    assert "import exchange_calendars" not in source
    assert "from exchange_calendars" not in source
    assert "import pandas" not in source
    assert "from pandas" not in source
    assert "pd.bdate_range" not in source
    assert "get_calendar" not in source
    assert "calendar-file" not in source


def test_source_module_is_not_changed_by_bridge_implementation():
    assert (ROOT / "src" / "v9_009_t0_top1_kill_screen.py").exists()
