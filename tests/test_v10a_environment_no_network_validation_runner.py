from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts import v10a_environment_no_network_validation_runner as runner
from scripts.v10_environment_extension_contract import PREDECESSOR_PACKAGE_SET


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _blob(raw: bytes) -> str:
    return runner.git_blob_sha1(raw)


def _packages() -> list[dict[str, str]]:
    packages = [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET]
    packages.extend({"name": name, "version": version} for name, version in runner.EXPECTED_DELTA)
    return packages


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[runner.V10AValidationConfig, dict[str, object], bytes, bytes]:
    repo = tmp_path / "repo"
    repo.mkdir()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    attempt = tmp_path / "step4-attempt"
    attempt.mkdir()
    output = tmp_path / "output"
    jpx = b"# synthetic jpx\r\nvalue = 1\n"
    jp = b"# synthetic jp\nvalue = 2\r\n"
    wheel_path = wheelhouse / runner.OFFICIAL_WHEEL_FILENAME
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(runner.JPX_ENTRY, jpx)
        archive.writestr(runner.JP_ENTRY, jp)
        archive.writestr("pandas_market_calendars-5.4.0.dist-info/METADATA", b"Name: pandas-market-calendars\nVersion: 5.4.0\n")
    installed_jpx = tmp_path / "installed" / runner.JPX_ENTRY
    installed_jp = tmp_path / "installed" / runner.JP_ENTRY
    installed_jpx.parent.mkdir(parents=True)
    installed_jp.parent.mkdir(parents=True)
    installed_jpx.write_bytes(jpx)
    installed_jp.write_bytes(jp)
    monkeypatch.setattr(runner, "OFFICIAL_WHEEL_SHA256", _sha256(wheel_path.read_bytes()))
    monkeypatch.setattr(runner, "JPX_RELEASE_GIT_BLOB_SHA1", _blob(jpx))
    monkeypatch.setattr(runner, "JP_RELEASE_GIT_BLOB_SHA1", _blob(jp))
    config = runner.V10AValidationConfig(
        repo_root=repo,
        expected_current_head="c" * 40,
        expected_live_validation_runner_commit_sha="a" * 40,
        expected_live_validation_runner_blob_sha1="b" * 40,
        wheelhouse=wheelhouse,
        step4_attempt_root=attempt,
        output_root=output,
    )
    observations: dict[str, object] = {
        "repository_identity": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "branch": runner.AUTHORITATIVE_BRANCH,
        "head": config.expected_current_head,
        "clean": True,
        "approved_design_commit_exists": True,
        "approved_design_blob_sha1": runner.APPROVED_DESIGN_BLOB_SHA1,
        "freeze_record_commit_exists": True,
        "freeze_record_blob_sha1": runner.FREEZE_RECORD_BLOB_SHA1,
        "current_frozen_design_blob_sha1": runner.FREEZE_RECORD_BLOB_SHA1,
        "current_design_matches_freeze_record": True,
        "v10a_runner_commit_exists": True,
        "reviewed_v10a_runner_blob_sha1": config.expected_live_validation_runner_blob_sha1,
        "current_v10a_runner_blob_sha1": config.expected_live_validation_runner_blob_sha1,
        "historical_step4_provenance_valid": True,
        "reviewed_wheelhouse_provenance_valid": True,
        "official_wheel_path": str(wheel_path),
        "installed_jpx_path": str(installed_jpx),
        "installed_jp_path": str(installed_jp),
        "observed_packages": _packages(),
        "interpreter_executable": str(config.canonical_interpreter.resolve()),
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "4.13.2",
        "xls_probe_status": "PASS",
        "pdf_probe_status": "PASS",
        "package_index_network_requests": 0,
        "package_installations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
    }
    return config, observations, jpx, jp


def test_exact_pass_and_non_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    result = runner.run_validation(config, observations, publish=True)
    assert result["status"] == "PASS"
    evidence = result["evidence"]
    runner.validate_evidence(evidence)
    assert evidence["jpx_entry_occurrence_count"] == 1
    assert evidence["jp_entry_occurrence_count"] == 1
    assert evidence["official_wheel_sha256_match"] is True
    assert result["canonical_environment_ready"] is False
    assert result["environment_frozen"] is False
    assert result["execution_authorized"] is False
    assert result["artifact_path"].read_bytes() == runner.canonical_json_bytes(evidence)


def test_production_constants_are_frozen_v10a_values() -> None:
    assert runner.APPROVED_DESIGN_SHA == "b14cc5510685210e928000af0815e188bc1aadc0"
    assert runner.FREEZE_RECORD_SHA == "86ceda3dee531b08afa5db4df7af1298ca770fad"
    assert runner.PMC_VERSION == "5.4.0"
    assert runner.EXCHANGE_CALENDARS_VERSION == "4.13.2"
    assert runner.OFFICIAL_WHEEL_FILENAME == "pandas_market_calendars-5.4.0-py3-none-any.whl"
    assert runner.JPX_RELEASE_GIT_BLOB_SHA1 == "a7a59b6cf910e325c85fc042459ff57ca8f70613"
    assert runner.JP_RELEASE_GIT_BLOB_SHA1 == "4c34214d06862e02ac22e946757463f748074fde"


@pytest.mark.parametrize("mutation", ["missing_jpx", "missing_jp", "duplicate_jpx", "duplicate_jp", "case", "backslash", "suffix"])
def test_exact_entry_selection_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    config, observations, jpx, jp = _fixture(tmp_path, monkeypatch)
    wheel = Path(str(observations["official_wheel_path"]))
    names = [(runner.JPX_ENTRY, jpx), (runner.JP_ENTRY, jp)]
    if mutation == "missing_jpx":
        names = names[1:]
    elif mutation == "missing_jp":
        names = names[:1]
    elif mutation == "duplicate_jpx":
        names.append((runner.JPX_ENTRY, jpx))
    elif mutation == "duplicate_jp":
        names.append((runner.JP_ENTRY, jp))
    elif mutation == "case":
        names[0] = (runner.JPX_ENTRY.upper(), jpx)
    elif mutation == "backslash":
        names[0] = (runner.JPX_ENTRY.replace("/", "\\"), jpx)
    else:
        names[0] = ("other/" + runner.JPX_ENTRY.rsplit("/", 1)[-1], jpx)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, raw in names:
            archive.writestr(name, raw)
    if mutation == "backslash":
        raw_archive = wheel.read_bytes()
        raw_archive = raw_archive.replace(
            runner.JPX_ENTRY.encode("ascii"),
            runner.JPX_ENTRY.replace("/", "\\").encode("ascii"),
        )
        wheel.write_bytes(raw_archive)
    monkeypatch.setattr(runner, "OFFICIAL_WHEEL_SHA256", _sha256(wheel.read_bytes()))
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"
    assert result["evidence"]["jpx_installed_equals_wheel_entry"] is None
    assert result["evidence"]["jp_installed_equals_wheel_entry"] is None


def test_both_unique_counts_precede_source_reads_and_no_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    original_read = zipfile.ZipFile.read
    original_extract = zipfile.ZipFile.extract
    reads = 0

    def read(self: zipfile.ZipFile, *args: object, **kwargs: object) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(self, *args, **kwargs)

    def extract(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("archive extraction is forbidden")

    monkeypatch.setattr(zipfile.ZipFile, "read", read)
    monkeypatch.setattr(zipfile.ZipFile, "extract", extract)
    result = runner.run_validation(config, observations, publish=False)
    assert result["status"] == "PASS"
    assert reads == 2


@pytest.mark.parametrize("which", ["jpx", "jp"])
def test_installed_source_raw_bytes_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    path = Path(str(observations["installed_jpx_path"] if which == "jpx" else observations["installed_jp_path"]))
    path.write_bytes(path.read_bytes() + b"\n")
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == ("JPX_INSTALLED_WHEEL_BYTES_MISMATCH" if which == "jpx" else "HOLIDAY_INSTALLED_WHEEL_BYTES_MISMATCH")


@pytest.mark.parametrize("which", ["jpx", "jp"])
def test_release_blob_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    constant = "JPX_RELEASE_GIT_BLOB_SHA1" if which == "jpx" else "JP_RELEASE_GIT_BLOB_SHA1"
    monkeypatch.setattr(runner, constant, "0" * 40)
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == ("JPX_RELEASE_BLOB_MISMATCH" if which == "jpx" else "HOLIDAY_RELEASE_BLOB_MISMATCH")


def test_raw_bytes_and_crlf_are_not_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    result = runner.run_validation(config, observations, publish=False)
    assert result["status"] == "PASS"


def test_official_wheel_hash_failure_precedes_entry_enumeration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "OFFICIAL_WHEEL_SHA256", "0" * 64)
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == "OFFICIAL_WHEEL_IDENTITY_MISMATCH"
    assert result["evidence"]["jpx_entry_occurrence_count"] is None


def test_provenance_failure_precedes_live_and_source_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    bad = copy.deepcopy(observations)
    bad["approved_design_blob_sha1"] = "0" * 40
    live_called = False

    def live(_config: runner.V10AValidationConfig) -> dict[str, object]:
        nonlocal live_called
        live_called = True
        raise AssertionError("live observation must not run")

    monkeypatch.setattr(runner, "_default_live_observations", live)
    result = runner.run_validation(config, bad, publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert live_called is False


def test_unauthorized_operation_has_highest_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    observations["package_installations"] = 1
    observations["approved_design_blob_sha1"] = "0" * 40
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == "UNAUTHORIZED_OPERATION_OBSERVED"


@pytest.mark.parametrize("change", ["missing", "extra", "bad_bool", "bad_int"])
def test_evidence_validator_is_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    evidence = runner.run_validation(config, observations, publish=False)["evidence"]
    if change == "missing":
        evidence.pop("jpx_source_blob_match")
    elif change == "extra":
        evidence["extra"] = True
    elif change == "bad_bool":
        evidence["t0_run"] = 0
    else:
        evidence["jpx_entry_occurrence_count"] = True
    with pytest.raises(runner.V10AValidationError):
        runner.validate_evidence(evidence)


def test_cli_has_no_synthetic_or_injection_options() -> None:
    options = {option for action in runner._build_parser()._actions for option in action.option_strings}
    assert not any("observ" in option or "process" in option or "fake" in option or "synthetic" in option for option in options)


def test_cli_calls_production_path_without_observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}

    def fake(config: runner.V10AValidationConfig, observations: object, *, publish: bool) -> dict[str, object]:
        captured.update(config=config, observations=observations, publish=publish)
        return {"status": "FAIL", "failure_code": "PROVENANCE_BINDING_FAILURE", "canonical_environment_ready": False, "environment_frozen": False, "execution_authorized": False}

    monkeypatch.setattr(runner, "run_validation", fake)
    args = [
        "--repo-root", str(tmp_path / "repo"),
        "--expected-current-head", "c" * 40,
        "--expected-live-validation-runner-commit-sha", "a" * 40,
        "--expected-live-validation-runner-blob-sha1", "b" * 40,
        "--wheelhouse", str(tmp_path / "wheelhouse"),
        "--step4-attempt-root", str(tmp_path / "attempt"),
        "--output-root", str(tmp_path / "output"),
    ]
    assert runner.main(args) == 1
    assert captured["observations"] is None
    assert captured["publish"] is True
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False


def _stub_production_stages(
    monkeypatch: pytest.MonkeyPatch,
    config: runner.V10AValidationConfig,
    observations: dict[str, object],
    order: list[str],
    fail_stage: str | None = None,
) -> None:
    base = {key: observations[key] for key in observations if key in {
        "repository_identity", "branch", "head", "clean", "approved_design_commit_exists",
        "approved_design_blob_sha1", "freeze_record_commit_exists", "freeze_record_blob_sha1",
        "current_frozen_design_blob_sha1", "current_design_matches_freeze_record",
        "v10a_runner_commit_exists", "reviewed_v10a_runner_blob_sha1", "current_v10a_runner_blob_sha1",
    }}

    def provenance(_config: runner.V10AValidationConfig) -> dict[str, object]:
        order.append("stage2")
        return base

    def repository(_config: runner.V10AValidationConfig, _obs: object) -> bool:
        order.append("stage2_validate")
        return fail_stage != "stage2"

    def historical(_config: runner.V10AValidationConfig, obs: object) -> dict[str, object]:
        order.append("stage3")
        result = dict(obs)  # type: ignore[arg-type]
        result.update(
            historical_step4_provenance_valid=fail_stage != "stage3",
            reviewed_wheelhouse_provenance_valid=fail_stage != "stage3",
        )
        return result

    def historical_valid(obs: object) -> bool:
        order.append("stage3_validate")
        return fail_stage != "stage3"

    def packages(_config: runner.V10AValidationConfig) -> dict[str, object]:
        order.append("package")
        return {"observed_packages": _packages()} if fail_stage != "package" else {"observed_packages": []}

    def platform_observation(_config: runner.V10AValidationConfig) -> dict[str, object]:
        order.append("platform")
        return {
            "interpreter_executable": str(config.canonical_interpreter.resolve()),
            "python_version": "3.12.10",
            "platform_system": "Windows" if fail_stage != "platform" else "Linux",
            "platform_machine": "AMD64",
            "sysconfig_platform": "win-amd64",
        }

    def versions(_config: runner.V10AValidationConfig) -> dict[str, object]:
        order.append("versions")
        return {
            "pandas_market_calendars_version": "5.4.0" if fail_stage != "versions" else "0.0.0",
            "exchange_calendars_version": "4.13.2",
        }

    def wheel(_config: runner.V10AValidationConfig, _obs: object, phase: dict[str, object]) -> str:
        order.append("wheel_sha")
        phase["_wheel_path"] = Path("synthetic-wheel.whl")
        phase["official_wheel_filename"] = runner.OFFICIAL_WHEEL_FILENAME
        phase["observed_official_wheel_sha256"] = runner.OFFICIAL_WHEEL_SHA256
        phase["official_wheel_sha256_match"] = True
        return "NONE" if fail_stage != "wheel" else "OFFICIAL_WHEEL_IDENTITY_MISMATCH"

    def entries(phase: dict[str, object]) -> str:
        order.append("entries")
        phase["jpx_entry_occurrence_count"] = 1
        phase["jp_entry_occurrence_count"] = 1
        return "NONE" if fail_stage != "entries" else "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"

    def installed_paths(_config: runner.V10AValidationConfig) -> dict[str, object]:
        order.append("source_paths")
        return {"installed_jpx_path": "synthetic-jpx.py", "installed_jp_path": "synthetic-jp.py"}

    def source(_config: runner.V10AValidationConfig, _obs: object, _phase: dict[str, object]) -> str:
        order.append("source")
        _phase.update(
            jpx_installed_equals_wheel_entry=True,
            jp_installed_equals_wheel_entry=True,
            jpx_wheel_git_blob_sha1=runner.JPX_RELEASE_GIT_BLOB_SHA1,
            jp_wheel_git_blob_sha1=runner.JP_RELEASE_GIT_BLOB_SHA1,
            jpx_installed_git_blob_sha1=runner.JPX_RELEASE_GIT_BLOB_SHA1,
            jp_installed_git_blob_sha1=runner.JP_RELEASE_GIT_BLOB_SHA1,
            jpx_source_blob_match=True,
            holiday_source_blob_match=True,
        )
        return "NONE"

    def xls() -> str:
        order.append("xls")
        return "PASS" if fail_stage != "xls" else "FAIL"

    def pdf() -> str:
        order.append("pdf")
        return "PASS"

    monkeypatch.setattr(runner, "_default_provenance_observations", provenance)
    monkeypatch.setattr(runner, "_validate_repository_provenance", repository)
    monkeypatch.setattr(runner, "_default_historical_provenance_observations", historical)
    monkeypatch.setattr(runner, "_validate_historical_provenance", historical_valid)
    monkeypatch.setattr(runner, "_default_package_observations", packages)
    monkeypatch.setattr(runner, "_default_platform_observations", platform_observation)
    monkeypatch.setattr(runner, "_default_package_version_observations", versions)
    monkeypatch.setattr(runner, "_validate_official_wheel_identity", wheel)
    monkeypatch.setattr(runner, "_enumerate_unique_source_entries", entries)
    monkeypatch.setattr(runner, "_default_installed_source_paths", installed_paths)
    monkeypatch.setattr(runner, "_validate_source_entries", source)
    monkeypatch.setattr(runner, "_default_xls_probe", xls)
    monkeypatch.setattr(runner, "_default_pdf_probe", pdf)


@pytest.mark.parametrize(
    ("fail_stage", "forbidden"),
    [
        ("stage2", {"stage3", "package", "platform", "versions", "wheel_sha", "entries", "source_paths", "source", "xls", "pdf"}),
        ("stage3", {"package", "platform", "versions", "wheel_sha", "entries", "source_paths", "source", "xls", "pdf"}),
        ("package", {"platform", "versions", "wheel_sha", "entries", "source_paths", "source", "xls", "pdf"}),
        ("platform", {"versions", "wheel_sha", "entries", "source_paths", "source", "xls", "pdf"}),
        ("versions", {"wheel_sha", "entries", "source_paths", "source", "xls", "pdf"}),
        ("wheel", {"entries", "source_paths", "source", "xls", "pdf"}),
        ("entries", {"source_paths", "source", "xls", "pdf"}),
        ("xls", {"pdf"}),
    ],
)
def test_production_stage_failure_suppresses_later_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_stage: str,
    forbidden: set[str],
) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    order: list[str] = []
    _stub_production_stages(monkeypatch, config, observations, order, fail_stage)
    result = runner.run_validation(config, observations=None, publish=False)
    assert result["status"] == "FAIL"
    assert not forbidden.intersection(order)


def test_production_pass_traverses_stages_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    order: list[str] = []
    _stub_production_stages(monkeypatch, config, observations, order)
    result = runner.run_validation(config, observations=None, publish=False)
    assert result["status"] == "PASS"
    assert order == [
        "stage2", "stage2_validate", "stage3", "stage3_validate", "package",
        "platform", "versions", "wheel_sha", "entries", "source_paths", "source", "xls", "pdf",
    ]


def test_production_wheel_hash_failure_never_opens_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "OFFICIAL_WHEEL_SHA256", "0" * 64)

    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ZIP must not open after archive hash failure")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden_open)
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == "OFFICIAL_WHEEL_IDENTITY_MISMATCH"


def test_production_uniqueness_failure_never_resolves_installed_source_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _, _ = _fixture(tmp_path, monkeypatch)
    wheel = Path(str(observations["official_wheel_path"]))
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(runner.JPX_ENTRY, b"duplicate")
    monkeypatch.setattr(runner, "OFFICIAL_WHEEL_SHA256", _sha256(wheel.read_bytes()))

    def forbidden_paths(_config: runner.V10AValidationConfig) -> dict[str, object]:
        raise AssertionError("installed source paths must not resolve before unique entries")

    monkeypatch.setattr(runner, "_default_installed_source_paths", forbidden_paths)
    result = runner.run_validation(config, observations, publish=False)
    assert result["failure_code"] == "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"
