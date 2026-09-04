"""Targeted synthetic tests for V9_014 PDF real-execution environment
successor Stage E2 offline tooling.

Every test here is synthetic and offline: no network, no `pip install`, no
real or staging environment creation/mutation, and no import of a real
installed `pdfplumber` (the lazy-import boundary is exercised exclusively
via an injected fake module). Real `pdfplumber==0.11.10` execution occurs
only at the later, separately reviewed Stage E6/E10/E14 checkpoints.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scripts.v9_014_pdf_env_successor as env_successor
import scripts.generate_v9_014_synthetic_pdf_probe as pdf_generator

REPO_ROOT = env_successor.REPO_ROOT
STAGING_RUNNER_PATH = REPO_ROOT / "scripts" / "v9_014_pdf_env_successor_staging_runner.ps1"
ATTRIBUTES_PATH = REPO_ROOT / ".gitattributes"


# =============================================================================
# Direct-spec byte validation
# =============================================================================


def test_direct_spec_exact_bytes_pass():
    result = env_successor.validate_direct_spec_bytes(env_successor.EXPECTED_DIRECT_SPEC_BYTES)
    assert result.status == env_successor.DIRECT_SPEC_OK


def test_direct_spec_committed_file_matches_exact_bytes():
    on_disk = env_successor.DIRECT_SPEC_PATH.read_bytes()
    result = env_successor.validate_direct_spec_bytes(on_disk)
    assert result.status == env_successor.DIRECT_SPEC_OK
    assert on_disk == b"pandas\nxlrd==2.0.2\npdfplumber==0.11.10\n"


@pytest.mark.parametrize(
    "mutated",
    [
        b"pandas\nxlrd==2.0.2\npdfplumber==0.11.10",  # missing trailing LF
        b"pandas\r\nxlrd==2.0.2\r\npdfplumber==0.11.10\r\n",  # CRLF
        b"# comment\npandas\nxlrd==2.0.2\npdfplumber==0.11.10\n",  # comment
        b"pandas\nxlrd==2.0.2\npdfplumber==0.11.10\nrequests==2.32.0\n",  # extra dependency
        b"pandas\npdfplumber==0.11.10\nxlrd==2.0.2\n",  # reordered
        b"pandas==2.2.0\nxlrd==2.0.2\npdfplumber==0.11.10\n",  # pinned pandas (spec must stay unpinned)
        b"pandas\nxlrd==2.0.2\npdfplumber==0.11.11\n",  # wrong pdfplumber version
        b"",  # empty
    ],
)
def test_direct_spec_any_deviation_fails(mutated: bytes):
    result = env_successor.validate_direct_spec_bytes(mutated)
    assert result.status == env_successor.DIRECT_SPEC_MISMATCH_FAILURE
    assert result.actual_bytes == mutated


# =============================================================================
# Strict pip-freeze parsing
# =============================================================================


def test_parse_pip_freeze_all_clean_exact_pins():
    text = "numpy==2.5.2\npandas==3.0.5\npip==25.0.1\n"
    result = env_successor.parse_pip_freeze_all(text)
    assert result.is_clean
    assert result.packages == {"numpy": "2.5.2", "pandas": "3.0.5", "pip": "25.0.1"}


def test_parse_pip_freeze_all_rejects_url_form():
    text = "numpy==2.5.2\nfoo @ file:///tmp/foo-1.0-py3-none-any.whl\n"
    result = env_successor.parse_pip_freeze_all(text)
    assert not result.is_clean
    assert any("foo @" in line for line in result.invalid_lines)


def test_parse_pip_freeze_all_rejects_editable_vcs_form():
    text = "numpy==2.5.2\n-e git+https://example.invalid/foo.git#egg=foo\n"
    result = env_successor.parse_pip_freeze_all(text)
    assert not result.is_clean
    assert result.invalid_lines


def test_parse_pip_freeze_all_detects_duplicates_via_normalization():
    text = "Foo_Bar==1.0\nfoo-bar==2.0\n"
    result = env_successor.parse_pip_freeze_all(text)
    assert not result.is_clean
    assert result.duplicate_lines == ("foo-bar==2.0",)
    assert result.packages == {"foo-bar": "1.0"}


def test_parse_pip_freeze_all_skips_comments_and_blanks():
    text = "# header\n\nnumpy==2.5.2\n  \n"
    result = env_successor.parse_pip_freeze_all(text)
    assert result.is_clean
    assert result.packages == {"numpy": "2.5.2"}


def test_parse_pip_freeze_all_rejects_bare_name_no_version():
    text = "numpy\n"
    result = env_successor.parse_pip_freeze_all(text)
    assert not result.is_clean
    assert result.invalid_lines == ("numpy",)


# =============================================================================
# Predecessor-baseline validation (exact 7 pins)
# =============================================================================

_EXACT_PREDECESSOR_FREEZE_TEXT = (
    "numpy==2.5.2\n"
    "pandas==3.0.5\n"
    "pip==25.0.1\n"
    "python-dateutil==2.9.0.post0\n"
    "six==1.17.0\n"
    "tzdata==2026.3\n"
    "xlrd==2.0.2\n"
)


def test_predecessor_baseline_exact_seven_pins_pass():
    freeze = env_successor.parse_pip_freeze_all(_EXACT_PREDECESSOR_FREEZE_TEXT)
    result = env_successor.validate_predecessor_baseline(freeze)
    assert result.status == env_successor.BASELINE_OK


def test_predecessor_baseline_version_drift_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT.replace("numpy==2.5.2", "numpy==2.5.3")
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_predecessor_baseline(freeze)
    assert result.status == env_successor.BASELINE_FAILURE
    assert result.version_mismatches == {"numpy": "2.5.3"}


def test_predecessor_baseline_missing_pin_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT.replace("xlrd==2.0.2\n", "")
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_predecessor_baseline(freeze)
    assert result.status == env_successor.BASELINE_FAILURE
    assert result.missing == ("xlrd",)


def test_predecessor_baseline_unexpected_extra_package_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT + "requests==2.32.0\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_predecessor_baseline(freeze)
    assert result.status == env_successor.BASELINE_FAILURE
    assert result.unexpected_extra == ("requests",)


def test_predecessor_baseline_duplicate_line_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT + "numpy==2.5.2\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_predecessor_baseline(freeze)
    assert result.status == env_successor.BASELINE_FAILURE


def test_predecessor_baseline_malformed_line_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT + "-e git+https://example.invalid/foo.git\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_predecessor_baseline(freeze)
    assert result.status == env_successor.BASELINE_FAILURE


# =============================================================================
# Successor-after-resolution validation
# =============================================================================


def test_successor_after_resolution_preserves_all_seven_pins_and_adds_pdfplumber():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT + "pdfplumber==0.11.10\ncharset-normalizer==3.4.0\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_successor_after_resolution(freeze)
    assert result.status == env_successor.SUCCESSOR_OK


def test_successor_after_resolution_missing_pdfplumber_fails():
    freeze = env_successor.parse_pip_freeze_all(_EXACT_PREDECESSOR_FREEZE_TEXT)
    result = env_successor.validate_successor_after_resolution(freeze)
    assert result.status == env_successor.SUCCESSOR_FAILURE
    assert result.new_package_missing is True


def test_successor_after_resolution_wrong_pdfplumber_version_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT + "pdfplumber==0.11.9\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_successor_after_resolution(freeze)
    assert result.status == env_successor.SUCCESSOR_FAILURE
    assert result.new_package_version_wrong == "0.11.9"


def test_successor_after_resolution_predecessor_drift_fails_even_with_pdfplumber_present():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT.replace("pandas==3.0.5", "pandas==3.1.0") + "pdfplumber==0.11.10\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_successor_after_resolution(freeze)
    assert result.status == env_successor.SUCCESSOR_FAILURE
    assert result.predecessor_version_drift == {"pandas": "3.1.0"}


def test_successor_after_resolution_missing_predecessor_pin_fails():
    text = _EXACT_PREDECESSOR_FREEZE_TEXT.replace("six==1.17.0\n", "") + "pdfplumber==0.11.10\n"
    freeze = env_successor.parse_pip_freeze_all(text)
    result = env_successor.validate_successor_after_resolution(freeze)
    assert result.status == env_successor.SUCCESSOR_FAILURE
    assert result.missing_predecessor_pins == ("six",)


# =============================================================================
# Before/after delta derivation
# =============================================================================


def test_delta_added_packages_deterministic():
    before = dict(env_successor.PREDECESSOR_PINS)
    after = dict(before)
    after["pdfplumber"] = "0.11.10"
    after["charset-normalizer"] = "3.4.0"
    result = env_successor.compute_before_after_delta(before, after)
    assert result.status == env_successor.DELTA_OK
    assert result.added == {"pdfplumber": "0.11.10", "charset-normalizer": "3.4.0"}
    assert result.removed == ()
    assert result.predecessor_pin_drift == {}


def test_delta_removed_predecessor_package_fails():
    before = dict(env_successor.PREDECESSOR_PINS)
    after = {name: version for name, version in before.items() if name != "six"}
    result = env_successor.compute_before_after_delta(before, after)
    assert result.status == env_successor.DELTA_REMOVED_PACKAGE_FAILURE
    assert result.removed == ("six",)


def test_delta_predecessor_pin_drift_fails():
    before = dict(env_successor.PREDECESSOR_PINS)
    after = dict(before)
    after["xlrd"] = "2.0.3"
    result = env_successor.compute_before_after_delta(before, after)
    assert result.status == env_successor.DELTA_PREDECESSOR_DRIFT_FAILURE
    assert result.predecessor_pin_drift == {"xlrd": ("2.0.2", "2.0.3")}


def test_delta_added_packages_never_appear_in_before_only_scenario():
    # A package present only in BEFORE and absent from AFTER is "removed",
    # never spuriously reported as "added" -- added is strictly AFTER-only.
    before = {"numpy": "2.5.2", "stale-only": "1.0.0"}
    after = {"numpy": "2.5.2", "pdfplumber": "0.11.10"}
    result = env_successor.compute_before_after_delta(before, after)
    assert result.status == env_successor.DELTA_REMOVED_PACKAGE_FAILURE
    assert result.removed == ("stale-only",)
    assert result.added == {"pdfplumber": "0.11.10"}


# =============================================================================
# Platform evidence schema validation
# =============================================================================


def _valid_platform_payload() -> dict:
    return {
        "implementation": env_successor.CANONICAL_PYTHON_IMPLEMENTATION,
        "os_name": env_successor.CANONICAL_OS_NAME,
        "platform_machine": env_successor.CANONICAL_PLATFORM_MACHINE,
        "platform_system": env_successor.CANONICAL_PLATFORM_SYSTEM,
        "sysconfig_platform": env_successor.CANONICAL_SYSCONFIG_PLATFORM,
        "version": env_successor.CANONICAL_PYTHON_EXACT_VERSION,
    }


def test_platform_evidence_exact_schema_pass():
    result = env_successor.validate_platform_evidence(_valid_platform_payload())
    assert result.status == env_successor.PLATFORM_OK


def test_platform_evidence_missing_key_fails():
    payload = _valid_platform_payload()
    del payload["os_name"]
    result = env_successor.validate_platform_evidence(payload)
    assert result.status == env_successor.PLATFORM_SCHEMA_FAILURE
    assert result.missing_keys == ("os_name",)


def test_platform_evidence_extra_key_fails():
    payload = _valid_platform_payload()
    payload["extra_field"] = "unexpected"
    result = env_successor.validate_platform_evidence(payload)
    assert result.status == env_successor.PLATFORM_SCHEMA_FAILURE
    assert result.unexpected_extra_keys == ("extra_field",)


def test_platform_evidence_wrong_version_fails():
    payload = _valid_platform_payload()
    payload["version"] = (3, 11, 9)
    result = env_successor.validate_platform_evidence(payload)
    assert result.status == env_successor.PLATFORM_VALUE_MISMATCH_FAILURE


def test_platform_evidence_wrong_platform_system_fails():
    payload = _valid_platform_payload()
    payload["platform_system"] = "Linux"
    result = env_successor.validate_platform_evidence(payload)
    assert result.status == env_successor.PLATFORM_VALUE_MISMATCH_FAILURE


# =============================================================================
# Future Stage E7 lock-candidate / Windows-evidence schema construction
# (pure functions; E2 never writes these artifacts to disk)
# =============================================================================


def test_build_lock_candidate_payload_schema_valid():
    payload = env_successor.build_lock_candidate_payload(
        direct_spec_sha256="deadbeef",
        resolved_packages={"pdfplumber": "0.11.10"},
    )
    assert env_successor.validate_lock_candidate_schema(payload)
    assert payload["predecessor_pins_preserved"] == dict(sorted(env_successor.PREDECESSOR_PINS.items()))
    assert payload["new_direct_package"] == "pdfplumber==0.11.10"


def test_build_windows_evidence_payload_schema_valid():
    payload = env_successor.build_windows_evidence_payload(
        platform_evidence=_valid_platform_payload(),
        before_freeze={"numpy": "2.5.2"},
        after_freeze={"numpy": "2.5.2", "pdfplumber": "0.11.10"},
        delta_status=env_successor.DELTA_OK,
    )
    assert env_successor.validate_windows_evidence_schema(payload)


def test_lock_candidate_schema_rejects_missing_key():
    payload = env_successor.build_lock_candidate_payload(direct_spec_sha256="x", resolved_packages={})
    del payload["schema_version"]
    assert not env_successor.validate_lock_candidate_schema(payload)


def test_windows_evidence_schema_rejects_extra_key():
    payload = env_successor.build_windows_evidence_payload(
        platform_evidence=_valid_platform_payload(), before_freeze={}, after_freeze={}, delta_status="X"
    )
    payload["unexpected"] = True
    assert not env_successor.validate_windows_evidence_schema(payload)


def test_stage_e2_does_not_write_future_artifacts_to_disk():
    # Stage E2 must not create the future E7 artifacts at all.
    assert not (REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json").exists()
    assert not (REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_WINDOWS_VALIDATION_EVIDENCE.json").exists()


# =============================================================================
# Staging-path validation
# =============================================================================


def test_staging_path_canonical_general_absolute_pass(tmp_path: Path):
    candidate = tmp_path / "staging-root"
    result = env_successor.validate_staging_path(candidate, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_OK


def test_staging_path_relative_fails():
    result = env_successor.validate_staging_path(Path("relative/staging"), repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_NOT_ABSOLUTE_FAILURE


def test_staging_path_inside_repo_fails():
    result = env_successor.validate_staging_path(REPO_ROOT / "some-staging-dir", repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_INSIDE_REPO_FAILURE


def test_staging_path_matches_canonical_environment_name_fails(tmp_path: Path):
    candidate = tmp_path / ".venv-real-execution"
    result = env_successor.validate_staging_path(candidate, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_MATCHES_CANONICAL_NAME_FAILURE


def test_staging_path_matches_general_environment_name_fails(tmp_path: Path):
    candidate = tmp_path / ".venv"
    result = env_successor.validate_staging_path(candidate, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_MATCHES_GENERAL_NAME_FAILURE


def test_staging_path_already_exists_fails_when_required(tmp_path: Path):
    candidate = tmp_path / "already-here"
    candidate.mkdir()
    result = env_successor.validate_staging_path(candidate, repo_root=REPO_ROOT, require_not_exists=True)
    assert result.status == env_successor.STAGING_PATH_ALREADY_EXISTS_FAILURE


def test_staging_path_already_exists_not_checked_by_default(tmp_path: Path):
    candidate = tmp_path / "already-here-2"
    candidate.mkdir()
    result = env_successor.validate_staging_path(candidate, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_OK


def test_staging_path_symlink_collision_fails(tmp_path: Path):
    real_dir = tmp_path / "real-target"
    real_dir.mkdir()
    symlinked_parent = tmp_path / "symlinked-parent"
    symlinked_parent.symlink_to(real_dir, target_is_directory=True)
    candidate = symlinked_parent / "staging-under-symlink"
    result = env_successor.validate_staging_path(candidate, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_IS_SYMLINK_FAILURE


# =============================================================================
# Synthetic PDF fixture: committed hash and generator --check
# =============================================================================


def test_v9_014_exact_checkout_provenance_attributes_are_narrow():
    rules = ATTRIBUTES_PATH.read_text(encoding="utf-8").splitlines()
    assert "tests/fixtures/v9_014_synthetic_pdf_env_probe.pdf binary" in rules
    assert "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt text eol=lf" in rules


def test_synthetic_pdf_fixture_committed_hash_matches_expected():
    assert pdf_generator.FIXTURE_PATH.exists()
    committed = pdf_generator.FIXTURE_PATH.read_bytes()
    import hashlib

    assert hashlib.sha256(committed).hexdigest() == pdf_generator.EXPECTED_FIXTURE_SHA256
    assert hashlib.sha256(committed).hexdigest() == env_successor.PROBE_EXPECTED_FIXTURE_SHA256


def test_synthetic_pdf_generator_rebuild_is_byte_identical_to_committed():
    committed = pdf_generator.FIXTURE_PATH.read_bytes()
    rebuilt = pdf_generator.build_pdf_bytes()
    assert rebuilt == committed


def test_synthetic_pdf_generator_check_flag_passes(capsys):
    # Invoke --check via the module's own argument parsing path.
    import sys as _sys

    old_argv = _sys.argv
    try:
        _sys.argv = ["generate_v9_014_synthetic_pdf_probe.py", "--check"]
        result = pdf_generator.main()
    finally:
        _sys.argv = old_argv
    captured = capsys.readouterr()
    assert result == 0
    assert "committed_matches_expected=true" in captured.out
    assert "rebuild_matched_committed_bytes=true" in captured.out
    assert "real_pdfplumber_imported=false" in captured.out


def test_synthetic_pdf_fixture_contains_no_jpx_or_price_like_content():
    raw = pdf_generator.FIXTURE_PATH.read_bytes()
    forbidden_markers = (b"JPX", b"TSE", b"\xc2\xa5", b"yen", b"YEN")
    for marker in forbidden_markers:
        assert marker not in raw


# =============================================================================
# Synthetic PDF operational probe (fake pdfplumber boundary only)
# =============================================================================


class _FakePage:
    def __init__(self, text: str, table: list, text_kwargs_seen: list, table_settings_seen: list):
        self._text = text
        self._table = table
        self._text_kwargs_seen = text_kwargs_seen
        self._table_settings_seen = table_settings_seen

    def extract_text(self, **kwargs):
        self._text_kwargs_seen.append(kwargs)
        return self._text

    def extract_table(self, settings):
        self._table_settings_seen.append(settings)
        return self._table


class _FakePdf:
    def __init__(self, pages: list):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _make_fake_pdfplumber(*, version="0.11.10", page_count=1, text=None, table=None):
    text = env_successor.PROBE_EXPECTED_TEXT if text is None else text
    table = env_successor.PROBE_EXPECTED_TABLE if table is None else table
    text_kwargs_seen: list = []
    table_settings_seen: list = []
    pages = [_FakePage(text, table, text_kwargs_seen, table_settings_seen) for _ in range(page_count)]

    class _FakeModule:
        __version__ = version

        @staticmethod
        def open(path):
            return _FakePdf(pages)

    return _FakeModule, text_kwargs_seen, table_settings_seen


def test_probe_fake_pdfplumber_exact_pass():
    fake_module, text_kwargs_seen, table_settings_seen = _make_fake_pdfplumber()
    result = env_successor.run_synthetic_pdf_operational_probe(pdfplumber_module=fake_module)
    assert result.status == env_successor.PROBE_PASS
    assert result.observed_pdfplumber_version == "0.11.10"
    assert result.observed_page_count == 1
    assert result.observed_text == env_successor.PROBE_EXPECTED_TEXT
    assert result.observed_table == env_successor.PROBE_EXPECTED_TABLE
    assert text_kwargs_seen == [{"x_tolerance": 3, "y_tolerance": 3}]
    assert table_settings_seen == [env_successor.PROBE_TABLE_SETTINGS]


def test_probe_wrong_fixture_hash_fails_before_any_pdfplumber_import(tmp_path: Path, monkeypatch):
    wrong_fixture = tmp_path / "wrong.pdf"
    wrong_fixture.write_bytes(b"%PDF-1.4\nnot the real fixture\n")

    def _explode(*args, **kwargs):  # pragma: no cover -- must never be called
        raise AssertionError("pdfplumber_module should never be constructed on a hash-mismatch path")

    result = env_successor.run_synthetic_pdf_operational_probe(fixture_path=wrong_fixture, pdfplumber_module=_explode)
    assert result.status == env_successor.PROBE_FIXTURE_HASH_MISMATCH_FAILURE
    assert result.observed_fixture_sha256 is not None


def test_probe_missing_fixture_fails(tmp_path: Path):
    missing = tmp_path / "does-not-exist.pdf"
    result = env_successor.run_synthetic_pdf_operational_probe(fixture_path=missing, pdfplumber_module=object())
    assert result.status == env_successor.PROBE_FIXTURE_MISSING_FAILURE


def test_probe_wrong_pdfplumber_version_fails():
    fake_module, _, _ = _make_fake_pdfplumber(version="0.10.0")
    result = env_successor.run_synthetic_pdf_operational_probe(pdfplumber_module=fake_module)
    assert result.status == env_successor.PROBE_VERSION_MISMATCH_FAILURE
    assert result.observed_pdfplumber_version == "0.10.0"


def test_probe_wrong_page_count_fails():
    fake_module, _, _ = _make_fake_pdfplumber(page_count=2)
    result = env_successor.run_synthetic_pdf_operational_probe(pdfplumber_module=fake_module)
    assert result.status == env_successor.PROBE_PAGE_COUNT_MISMATCH_FAILURE
    assert result.observed_page_count == 2


def test_probe_wrong_text_fails():
    fake_module, _, _ = _make_fake_pdfplumber(text="SOMETHING_ELSE")
    result = env_successor.run_synthetic_pdf_operational_probe(pdfplumber_module=fake_module)
    assert result.status == env_successor.PROBE_TEXT_MISMATCH_FAILURE
    assert result.observed_text == "SOMETHING_ELSE"


def test_probe_wrong_table_fails():
    fake_module, _, _ = _make_fake_pdfplumber(table=[["WRONG", "TABLE"]])
    result = env_successor.run_synthetic_pdf_operational_probe(pdfplumber_module=fake_module)
    assert result.status == env_successor.PROBE_TABLE_MISMATCH_FAILURE
    assert result.observed_table == [["WRONG", "TABLE"]]


def test_probe_never_imports_real_pdfplumber_when_module_injected(monkeypatch):
    # If the real `pdfplumber` were imported as a module-level side effect
    # of calling this function, `sys.modules` would already contain a
    # cached entry keyed "pdfplumber" bound to something other than our
    # fake by the time this assertion runs. We assert only that supplying
    # a fake module completely bypasses any import machinery: the fake's
    # own `open` is what gets called, never a real library's.
    fake_module, _, _ = _make_fake_pdfplumber()
    calls = []
    original_open = fake_module.open

    @staticmethod
    def _tracking_open(path):
        calls.append(path)
        return original_open(path)

    fake_module.open = _tracking_open
    result = env_successor.run_synthetic_pdf_operational_probe(pdfplumber_module=fake_module)
    assert result.status == env_successor.PROBE_PASS
    assert len(calls) == 1


def test_probe_function_source_only_imports_pdfplumber_inside_the_function():
    import inspect

    source = inspect.getsource(env_successor.run_synthetic_pdf_operational_probe)
    assert "import pdfplumber" in source
    module_source = inspect.getsource(env_successor)
    top_level_lines = module_source.splitlines()[: module_source.splitlines().index('def run_synthetic_pdf_operational_probe(')]  # noqa: E501
    # The literal substring "import pdfplumber" must not appear anywhere
    # before the function definition begins (i.e. not at module level).
    assert "import pdfplumber" not in "\n".join(top_level_lines)


# =============================================================================
# Static PowerShell source assertions (never executed)
# =============================================================================


@pytest.fixture(scope="module")
def staging_runner_source() -> str:
    return STAGING_RUNNER_PATH.read_text(encoding="utf-8")


def test_staging_runner_file_exists():
    assert STAGING_RUNNER_PATH.exists()


def test_staging_runner_declares_required_mandatory_parameters(staging_runner_source: str):
    assert re.search(r"\[Parameter\(Mandatory = \$true\)\]\[string\]\$ExpectedHead", staging_runner_source)
    assert re.search(r"\[Parameter\(Mandatory = \$true\)\]\[string\]\$StagingRoot", staging_runner_source)
    assert re.search(r"\[Parameter\(Mandatory = \$true\)\]\[string\]\$OutputRoot", staging_runner_source)


def _strip_powershell_comment_lines(source: str) -> str:
    """Drop every line whose first non-whitespace character is '#', so a
    negative claim made in a comment (e.g. "contains no Remove-Item
    call") cannot itself trip a substring search for the forbidden term.
    """
    return "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))


def test_staging_runner_never_calls_remove_item(staging_runner_source: str):
    code_only = _strip_powershell_comment_lines(staging_runner_source)
    assert "Remove-Item" not in code_only


def test_staging_runner_never_uses_retry_or_reset_language(staging_runner_source: str):
    # No git history-mutating command appears anywhere in the file.
    lowered = staging_runner_source.lower()
    assert ".reset(" not in lowered
    assert "git reset" not in lowered
    assert "git clean" not in lowered
    assert "git checkout" not in lowered


def test_staging_runner_no_loop_construct_wraps_successor_resolution(staging_runner_source: str):
    # A `while` loop legitimately exists elsewhere in this file (path-
    # ancestor walking inside Resolve-ExistingAncestorRealPath -- not a
    # retry of anything). What must never happen is a loop construct
    # wrapping the ONE successor-resolution pip invocation itself. Slice
    # out just the E5 step 4 region (from its own comment marker to the
    # line that captures its exit code) and assert no loop keyword
    # appears inside that slice.
    start_marker = "# E5 step 4:"
    end_marker = "$successorResolutionExitCode = Invoke-NativeProcessRawCapture"
    start = staging_runner_source.index(start_marker)
    end = staging_runner_source.index(end_marker, start) + len(end_marker)
    successor_resolution_region = staging_runner_source[start:end]
    assert "while (" not in successor_resolution_region
    assert "for (" not in successor_resolution_region
    assert "do {" not in successor_resolution_region
    assert "foreach (" not in successor_resolution_region


def test_staging_runner_captures_successor_streams_at_process_level(staging_runner_source: str):
    """The E5 resolver must bypass PowerShell 5.1 native stderr handling."""
    helper_start = staging_runner_source.index("function Invoke-NativeProcessRawCapture")
    e5_start = staging_runner_source.index("# E5 step 4:")
    helper_block = staging_runner_source[helper_start:e5_start]
    e5_end = staging_runner_source.index("$successorResolutionExitCode =", e5_start)
    e5_block = staging_runner_source[e5_start:e5_end]

    assert "System.Diagnostics.ProcessStartInfo" in helper_block
    assert "System.Diagnostics.Process" in helper_block
    assert "$processStartInfo.RedirectStandardOutput = $true" in helper_block
    assert "$processStartInfo.RedirectStandardError = $true" in helper_block
    assert helper_block.count("BaseStream.CopyToAsync") == 2
    assert "[System.Threading.Tasks.Task]::WaitAll" in helper_block
    assert "[System.IO.FileMode]::CreateNew" in helper_block
    assert "$capturedNativeExitCode = $nativeProcess.ExitCode" in helper_block
    assert "$successorResolutionExitCode = Invoke-NativeProcessRawCapture" in staging_runner_source
    assert "1>" not in e5_block
    assert "2>" not in e5_block


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell 5.1 regression only")
def test_windows_powershell_native_stderr_is_captured_without_abort(tmp_path: Path):
    """A wholly synthetic native child proves the process-level contract."""
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell executable unavailable")

    stdout_path = tmp_path / "synthetic-stdout.bin"
    stderr_path = tmp_path / "synthetic-stderr.bin"
    exit_code_path = tmp_path / "synthetic-exit-code.txt"
    powershell_code = textwrap.dedent(
        r'''
        $ErrorActionPreference = "Stop"
        $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processStartInfo.FileName = $env:ComSpec
        $processStartInfo.Arguments = '/c "echo synthetic-stdout&echo synthetic-stderr 1>&2&exit /b 23"'
        $processStartInfo.RedirectStandardOutput = $true
        $processStartInfo.RedirectStandardError = $true
        $processStartInfo.UseShellExecute = $false
        $processStartInfo.CreateNoWindow = $true
        $nativeProcess = New-Object System.Diagnostics.Process
        $nativeProcess.StartInfo = $processStartInfo
        $stdoutFileStream = New-Object System.IO.FileStream(
            $env:SYNTHETIC_STDOUT_PATH,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $stderrFileStream = New-Object System.IO.FileStream(
            $env:SYNTHETIC_STDERR_PATH,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            if (-not $nativeProcess.Start()) { throw "synthetic child did not start" }
            $stdoutCopyTask = $nativeProcess.StandardOutput.BaseStream.CopyToAsync($stdoutFileStream)
            $stderrCopyTask = $nativeProcess.StandardError.BaseStream.CopyToAsync($stderrFileStream)
            $nativeProcess.WaitForExit()
            $capturedNativeExitCode = $nativeProcess.ExitCode
            $copyTasks = [System.Threading.Tasks.Task[]]@($stdoutCopyTask, $stderrCopyTask)
            [System.Threading.Tasks.Task]::WaitAll($copyTasks)
            $stdoutFileStream.Flush()
            $stderrFileStream.Flush()
        }
        finally {
            $stdoutFileStream.Dispose()
            $stderrFileStream.Dispose()
            $nativeProcess.Dispose()
        }
        [System.IO.File]::WriteAllText($env:SYNTHETIC_EXIT_CODE_PATH, [string]$capturedNativeExitCode)
        Write-Output "captured=$capturedNativeExitCode"
        '''
    )
    encoded_code = base64.b64encode(powershell_code.encode("utf-16le")).decode("ascii")
    test_env = os.environ.copy()
    test_env.update(
        {
            "SYNTHETIC_STDOUT_PATH": str(stdout_path),
            "SYNTHETIC_STDERR_PATH": str(stderr_path),
            "SYNTHETIC_EXIT_CODE_PATH": str(exit_code_path),
        }
    )
    result = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_code],
        env=test_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "NativeCommandError" not in result.stdout + result.stderr
    assert result.stdout.count("captured=23") == 1
    assert stdout_path.read_bytes() == b"synthetic-stdout\r\n"
    # cmd.exe retains the separator before the stderr redirection as part
    # of this deterministic echo payload; preservation includes that byte.
    assert stderr_path.read_bytes() == b"synthetic-stderr \r\n"
    assert exit_code_path.read_text(encoding="utf-8") == "23"


def test_staging_runner_canonical_and_general_environments_never_write_targets(staging_runner_source: str):
    # The canonical/general environment directory names may appear only
    # in comparisons/comments/throws (collision or reserved-name checks),
    # never as the target of a write cmdlet (New-Item, Set-Content,
    # Out-File, venv creation, pip install destination).
    write_cmdlets = ("New-Item", "Set-Content", "Out-File", "python -m venv", "pip install")
    for line in staging_runner_source.splitlines():
        if ".venv-real-execution" in line or ('"$env' not in line and re.search(r"\.venv(?!-real-execution)\b", line)):
            for cmdlet in write_cmdlets:
                if cmdlet in line:
                    raise AssertionError(
                        f"Write cmdlet {cmdlet!r} found on a line mentioning a reserved environment name: {line!r}"
                    )


def test_staging_runner_writes_only_staging_and_output_roots(staging_runner_source: str):
    write_targets = re.findall(r"(?:New-Item -ItemType Directory -Path|Set-Content -LiteralPath)\s+(\$\w+)", staging_runner_source)
    assert write_targets, "expected at least one detectable write-target reference"
    for target in write_targets:
        assert target in ("$OutputRoot", "$beforeFreezePath", "$afterFreezePath", "$exitCodePath"), target


def test_staging_runner_venv_creation_targets_staging_root_only(staging_runner_source: str):
    assert re.search(r"python -m venv \$StagingRoot", staging_runner_source)
    assert "python -m venv \"$env" not in staging_runner_source


def test_staging_runner_exactly_one_successor_resolution_invocation(staging_runner_source: str):
    # The one successor-resolution process invocation carries the exact
    # pip arguments in its single explicit Arguments assignment above it.
    successor_calls = re.findall(r"\$successorResolutionExitCode = Invoke-NativeProcessRawCapture", staging_runner_source)
    assert len(successor_calls) == 1


def test_staging_runner_predecessor_install_uses_no_deps(staging_runner_source: str):
    assert "pip install --no-deps -r $predecessorLockRelativePath" in staging_runner_source


def test_staging_runner_hardcodes_reviewed_predecessor_lock_hash(staging_runner_source: str):
    assert "b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96" in staging_runner_source


def test_staging_runner_hardcodes_all_seven_predecessor_pins(staging_runner_source: str):
    for name, version in env_successor.PREDECESSOR_PINS.items():
        assert f'"{name}"' in staging_runner_source or f"'{name}'" in staging_runner_source
        assert version in staging_runner_source


def test_staging_runner_fails_closed_on_expected_head_mismatch(staging_runner_source: str):
    assert "PRE_GATE_EXPECTED_HEAD_MISMATCH" in staging_runner_source


def test_staging_runner_fails_closed_on_dirty_tree(staging_runner_source: str):
    assert "PRE_GATE_DIRTY_WORKING_TREE" in staging_runner_source


def test_staging_runner_fails_closed_on_staging_path_collision(staging_runner_source: str):
    assert "PRE_GATE_STAGING_PATH_ALREADY_EXISTS" in staging_runner_source


def test_staging_runner_fails_closed_on_non_windows(staging_runner_source: str):
    assert "PRE_GATE_NON_WINDOWS_HOST" in staging_runner_source


def test_staging_runner_fails_closed_on_wrong_python_version(staging_runner_source: str):
    assert "PRE_GATE_NON_CANONICAL_PYTHON" in staging_runner_source
    assert "3.12.10" in staging_runner_source


def test_staging_runner_never_performs_jpx_or_yahoo_network_request(staging_runner_source: str):
    lowered = staging_runner_source.lower()
    for forbidden in ("jpx.co.jp", "finance.yahoo", "yahoo.co.jp"):
        assert forbidden not in lowered


# =============================================================================
# HIGH_1 remediation: mandatory tests 1-6 -- predecessor-baseline-file
# CLI/helper semantics, not just string presence.
# =============================================================================

_SEVEN_PIN_FREEZE_TEXT = (
    "numpy==2.5.2\n"
    "pandas==3.0.5\n"
    "pip==25.0.1\n"
    "python-dateutil==2.9.0.post0\n"
    "six==1.17.0\n"
    "tzdata==2026.3\n"
    "xlrd==2.0.2\n"
)


def test_baseline_file_helper_exact_seven_pins_pass(tmp_path: Path):
    freeze_file = tmp_path / "before_freeze.txt"
    freeze_file.write_text(_SEVEN_PIN_FREEZE_TEXT, encoding="utf-8")
    result = env_successor.validate_predecessor_baseline_file(freeze_file)
    assert result.status == env_successor.BASELINE_OK


def test_baseline_file_helper_missing_pin_fails(tmp_path: Path):
    freeze_file = tmp_path / "before_freeze.txt"
    freeze_file.write_text(_SEVEN_PIN_FREEZE_TEXT.replace("xlrd==2.0.2\n", ""), encoding="utf-8")
    result = env_successor.validate_predecessor_baseline_file(freeze_file)
    assert result.status == env_successor.BASELINE_FAILURE
    assert result.missing == ("xlrd",)


def test_baseline_file_helper_extra_package_fails(tmp_path: Path):
    freeze_file = tmp_path / "before_freeze.txt"
    freeze_file.write_text(_SEVEN_PIN_FREEZE_TEXT + "pdfplumber==0.11.10\n", encoding="utf-8")
    result = env_successor.validate_predecessor_baseline_file(freeze_file)
    assert result.status == env_successor.BASELINE_FAILURE
    assert result.unexpected_extra == ("pdfplumber",)


def test_baseline_file_helper_version_drift_fails(tmp_path: Path):
    freeze_file = tmp_path / "before_freeze.txt"
    freeze_file.write_text(_SEVEN_PIN_FREEZE_TEXT.replace("xlrd==2.0.2", "xlrd==2.0.3"), encoding="utf-8")
    result = env_successor.validate_predecessor_baseline_file(freeze_file)
    assert result.status == env_successor.BASELINE_FAILURE
    assert result.version_mismatches == {"xlrd": "2.0.3"}


def test_baseline_file_helper_malformed_and_duplicate_freeze_fails(tmp_path: Path):
    malformed_file = tmp_path / "malformed.txt"
    malformed_file.write_text(_SEVEN_PIN_FREEZE_TEXT + "-e git+https://example.invalid/foo.git\n", encoding="utf-8")
    malformed_result = env_successor.validate_predecessor_baseline_file(malformed_file)
    assert malformed_result.status == env_successor.BASELINE_FAILURE

    duplicate_file = tmp_path / "duplicate.txt"
    duplicate_file.write_text(_SEVEN_PIN_FREEZE_TEXT + "numpy==2.5.2\n", encoding="utf-8")
    duplicate_result = env_successor.validate_predecessor_baseline_file(duplicate_file)
    assert duplicate_result.status == env_successor.BASELINE_FAILURE


def test_baseline_file_helper_bom_input_handled_deterministically(tmp_path: Path):
    bom_file = tmp_path / "bom_before_freeze.txt"
    bom_file.write_bytes(b"\xef\xbb\xbf" + _SEVEN_PIN_FREEZE_TEXT.encode("utf-8"))
    plain_file = tmp_path / "plain_before_freeze.txt"
    plain_file.write_text(_SEVEN_PIN_FREEZE_TEXT, encoding="utf-8")

    bom_result = env_successor.validate_predecessor_baseline_file(bom_file)
    plain_result = env_successor.validate_predecessor_baseline_file(plain_file)
    assert bom_result.status == env_successor.BASELINE_OK
    assert plain_result.status == env_successor.BASELINE_OK
    assert bom_result == plain_result


def test_baseline_file_helper_missing_file_fails_closed(tmp_path: Path):
    result = env_successor.validate_predecessor_baseline_file(tmp_path / "does-not-exist.txt")
    assert result.status == env_successor.BASELINE_FILE_MISSING_FAILURE


def test_baseline_file_helper_undecodable_bytes_fails_closed(tmp_path: Path):
    bad_file = tmp_path / "bad_bytes.txt"
    bad_file.write_bytes(b"\xff\xfe\x00\x01numpy==2.5.2")
    result = env_successor.validate_predecessor_baseline_file(bad_file)
    assert result.status == env_successor.BASELINE_FILE_DECODE_FAILURE


def test_baseline_file_helper_reuses_existing_parse_and_validate_functions():
    import inspect

    source = inspect.getsource(env_successor.validate_predecessor_baseline_file)
    assert "parse_pip_freeze_all(" in source
    assert "validate_predecessor_baseline(" in source


def test_baseline_cli_exit_code_reflects_validation_status(tmp_path: Path, capsys):
    import sys as _sys

    good_file = tmp_path / "good.txt"
    good_file.write_text(_SEVEN_PIN_FREEZE_TEXT, encoding="utf-8")
    old_argv = _sys.argv
    try:
        exit_code = env_successor.main(["--validate-predecessor-baseline-file", str(good_file)])
    finally:
        _sys.argv = old_argv
    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"status={env_successor.BASELINE_OK}" in captured.out
    assert "real_pdfplumber_imported=false" in captured.out

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text(_SEVEN_PIN_FREEZE_TEXT.replace("six==1.17.0\n", ""), encoding="utf-8")
    exit_code_bad = env_successor.main(["--validate-predecessor-baseline-file", str(bad_file)])
    captured_bad = capsys.readouterr()
    assert exit_code_bad == 1
    assert f"status={env_successor.BASELINE_FAILURE}" in captured_bad.out
    assert "missing=six" in captured_bad.out


def test_baseline_cli_no_args_is_still_a_noop():
    exit_code = env_successor.main([])
    assert exit_code == 0


# =============================================================================
# HIGH_1 remediation: mandatory tests 7-10 -- runner static/order proof
# that predecessor-baseline validation is mechanically enforced, not
# merely present as seven hardcoded strings.
# =============================================================================


def test_staging_runner_baseline_validation_occurs_after_before_capture(staging_runner_source: str):
    before_capture_index = staging_runner_source.index('$beforeFreezePath = Join-Path $OutputRoot "before_freeze.txt"')
    validation_invocation_index = staging_runner_source.index("--validate-predecessor-baseline-file $beforeFreezePath")
    assert before_capture_index < validation_invocation_index


def test_staging_runner_baseline_validation_occurs_before_successor_resolution(staging_runner_source: str):
    validation_invocation_index = staging_runner_source.index("--validate-predecessor-baseline-file $beforeFreezePath")
    successor_resolution_index = staging_runner_source.index("# E5 step 4:")
    assert validation_invocation_index < successor_resolution_index

    # The nonzero-exit throw for the baseline validator must also appear
    # strictly before the successor-resolution invocation begins.
    throw_index = staging_runner_source.index("E5_STEP3_PREDECESSOR_BASELINE_VALIDATION_FAILED")
    assert validation_invocation_index < throw_index < successor_resolution_index


def test_staging_runner_baseline_validation_nonzero_exit_has_explicit_throw(staging_runner_source: str):
    validation_block_start = staging_runner_source.index("--validate-predecessor-baseline-file $beforeFreezePath")
    validation_block_end = staging_runner_source.index("# E5 step 4:")
    validation_block = staging_runner_source[validation_block_start:validation_block_end]
    assert "$baselineValidationExitCode -ne 0" in validation_block
    assert "throw" in validation_block
    assert "E5_STEP3_PREDECESSOR_BASELINE_VALIDATION_FAILED" in validation_block


def test_staging_runner_baseline_validation_invokes_reviewed_python_tooling_not_bare_grep(staging_runner_source: str):
    # The seven hardcoded pin strings alone (already asserted present by
    # test_staging_runner_hardcodes_all_seven_predecessor_pins) are not
    # treated as sufficient proof of enforcement: the runner must
    # actually invoke the reviewed Python CLI/helper against the staging
    # interpreter, and must not implement a parallel `Select-String`/grep-
    # based comparison as a substitute.
    assert "$stagingInterpreter $offlineToolingPath --validate-predecessor-baseline-file" in staging_runner_source
    assert "Select-String" not in staging_runner_source
    assert "-match" not in staging_runner_source or "$baseInterpreterVersion -notmatch" in staging_runner_source


# =============================================================================
# Output-root validation (Python-side authority the runner mirrors)
# =============================================================================


def test_output_path_inside_repo_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    result = env_successor.validate_output_path(REPO_ROOT / "some-output-dir", staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_INSIDE_REPO_FAILURE


def test_output_path_matches_canonical_environment_name_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    result = env_successor.validate_output_path(tmp_path / ".venv-real-execution", staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_MATCHES_CANONICAL_NAME_FAILURE


def test_output_path_matches_general_environment_name_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    result = env_successor.validate_output_path(tmp_path / ".venv", staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_MATCHES_GENERAL_NAME_FAILURE


def test_output_path_nested_inside_reserved_environment_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    result = env_successor.validate_output_path(
        tmp_path / ".venv-real-execution" / "nested-output", staging_root=staging_root
    )
    assert result.status == env_successor.OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT_FAILURE


def test_output_path_already_exists_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    existing_output = tmp_path / "existing-output"
    existing_output.mkdir()
    result = env_successor.validate_output_path(existing_output, staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_ALREADY_EXISTS_FAILURE


def test_output_path_equals_staging_root_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    result = env_successor.validate_output_path(staging_root, staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_EQUALS_STAGING_ROOT_FAILURE


def test_output_path_ancestor_of_staging_root_fails(tmp_path: Path):
    staging_root = tmp_path / "outer" / "staging"
    result = env_successor.validate_output_path(tmp_path / "outer", staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT_FAILURE


def test_output_path_descendant_of_staging_root_fails(tmp_path: Path):
    staging_root = tmp_path / "staging"
    result = env_successor.validate_output_path(staging_root / "nested-output", staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT_FAILURE


def test_output_path_safe_distinct_nonexisting_external_pass(tmp_path: Path):
    staging_root = tmp_path / "staging-root"
    output_root = tmp_path / "output-root"
    result = env_successor.validate_output_path(output_root, staging_root=staging_root)
    assert result.status == env_successor.OUTPUT_PATH_OK


def test_output_path_not_absolute_fails():
    result = env_successor.validate_output_path(Path("relative/output"), staging_root=Path("/tmp/staging"))
    assert result.status == env_successor.OUTPUT_PATH_NOT_ABSOLUTE_FAILURE


# =============================================================================
# HIGH_1 remediation: mandatory tests 11-20 -- OutputRoot static
# assertions on the runner itself.
# =============================================================================


def test_staging_runner_output_root_inside_repo_fails_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_INSIDE_REPO" in staging_runner_source


def test_staging_runner_output_root_canonical_name_fails_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_MATCHES_CANONICAL_NAME" in staging_runner_source


def test_staging_runner_output_root_general_name_fails_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_MATCHES_GENERAL_NAME" in staging_runner_source


def test_staging_runner_output_root_already_exists_fails_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_ALREADY_EXISTS" in staging_runner_source


def test_staging_runner_output_root_equals_staging_root_fails_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_EQUALS_STAGING_ROOT" in staging_runner_source


def test_staging_runner_output_root_staging_overlap_both_directions_fail_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT" in staging_runner_source
    assert "PRE_GATE_OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT" in staging_runner_source


def test_staging_runner_output_root_reserved_environment_nesting_fails_closed(staging_runner_source: str):
    assert "PRE_GATE_OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT" in staging_runner_source


def test_staging_runner_no_force_reuse_of_output_root(staging_runner_source: str):
    # `-Force` is legitimately used elsewhere (e.g. `Get-Item -Force` to
    # reliably read a hidden/system reparse point during path-safety
    # resolution) -- what must never happen is `New-Item` creating
    # OutputRoot with `-Force`, which would silently permit reuse of an
    # existing directory.
    code_only = _strip_powershell_comment_lines(staging_runner_source)
    assert not re.search(r"New-Item[^\n]*-Force", code_only)
    assert re.search(r"New-Item -ItemType Directory -Path \$OutputRoot \|", code_only)


def test_staging_runner_output_root_created_only_after_all_preflight(staging_runner_source: str):
    creation_index = staging_runner_source.index("New-Item -ItemType Directory -Path $OutputRoot")
    preflight5_start = staging_runner_source.index("# Preflight 5: OutputRoot fail-closed validation")
    all_preflight_passed_index = staging_runner_source.index("All preflight checks passed, including StagingRoot and OutputRoot real-path fail-closed validation")
    assert preflight5_start < all_preflight_passed_index < creation_index


def test_staging_runner_still_never_calls_remove_item_after_output_root_remediation(staging_runner_source: str):
    code_only = _strip_powershell_comment_lines(staging_runner_source)
    assert "Remove-Item" not in code_only


def test_staging_runner_still_exactly_one_successor_resolution_invocation_after_remediation(staging_runner_source: str):
    successor_calls = re.findall(
        r"\$successorResolutionExitCode = Invoke-NativeProcessRawCapture",
        staging_runner_source,
    )
    assert len(successor_calls) == 1


def test_staging_runner_path_safety_undetermined_routes_to_chatgpt_decision_required(staging_runner_source: str):
    assert "CHATGPT_DECISION_REQUIRED" in staging_runner_source


# =============================================================================
# HIGH_2 remediation: mandatory tests 1-12 -- StagingRoot REAL-PATH
# protected-state guard (the real resolved StagingRoot was computed but
# never re-validated against the repo and reserved environments).
# =============================================================================


def test_staging_runner_realpath_staging_vs_repo_rejection_present(staging_runner_source: str):
    # Test 1: explicit realResolvedStagingRoot-vs-realRepo rejection.
    assert "PRE_GATE_STAGING_PATH_REALPATH_INSIDE_REPO" in staging_runner_source
    assert "$realResolvedStagingRoot.StartsWith($realResolvedRepoRoot" in staging_runner_source


def test_staging_runner_realpath_staging_vs_canonical_environment_rejection_present(staging_runner_source: str):
    # Test 2: realResolvedStagingRoot equal/descendant of canonical env
    # is rejected.
    assert "PRE_GATE_STAGING_PATH_REALPATH_INSIDE_CANONICAL_ENVIRONMENT" in staging_runner_source
    assert "$realResolvedStagingRoot -eq $realResolvedCanonicalEnvironmentPath" in staging_runner_source
    assert "$realResolvedStagingRoot.StartsWith($realResolvedCanonicalEnvironmentPath" in staging_runner_source


def test_staging_runner_realpath_staging_vs_general_environment_rejection_present(staging_runner_source: str):
    # Test 3: realResolvedStagingRoot equal/descendant of general ".venv"
    # is rejected.
    assert "PRE_GATE_STAGING_PATH_REALPATH_INSIDE_GENERAL_ENVIRONMENT" in staging_runner_source
    assert "$realResolvedStagingRoot -eq $realResolvedGeneralEnvironmentPath" in staging_runner_source
    assert "$realResolvedStagingRoot.StartsWith($realResolvedGeneralEnvironmentPath" in staging_runner_source


def test_staging_runner_realpath_checks_use_real_resolved_path_not_only_lexical(staging_runner_source: str):
    # Test 4: the new StagingRoot protected-state checks are keyed off
    # $realResolvedStagingRoot (the symlink/junction/reparse-resolved
    # value), not merely $resolvedStagingRoot (the lexical
    # GetFullPath value) -- both variables exist, and the new
    # REALPATH_INSIDE_* checks specifically reference the "real" one.
    preflight_4b_start = staging_runner_source.index("# Preflight 4b: StagingRoot REAL-PATH protected-state guard")
    preflight_5_start = staging_runner_source.index("# Preflight 5: OutputRoot fail-closed validation")
    preflight_4b_block = staging_runner_source[preflight_4b_start:preflight_5_start]
    assert "$realResolvedStagingRoot" in preflight_4b_block
    assert "$resolvedStagingRoot" not in preflight_4b_block


def test_staging_runner_realpath_staging_guards_occur_before_output_root_creation(staging_runner_source: str):
    # Test 5: all real staging safety checks occur before OutputRoot
    # creation.
    preflight_4b_start = staging_runner_source.index("# Preflight 4b: StagingRoot REAL-PATH protected-state guard")
    last_staging_realpath_check = staging_runner_source.rindex("PRE_GATE_STAGING_PATH_REALPATH_INSIDE_GENERAL_ENVIRONMENT")
    output_root_creation = staging_runner_source.index("New-Item -ItemType Directory -Path $OutputRoot")
    assert preflight_4b_start < last_staging_realpath_check < output_root_creation


def test_staging_runner_realpath_staging_guards_occur_before_venv_creation(staging_runner_source: str):
    # Test 6: all real staging safety checks occur before
    # "python -m venv $StagingRoot".
    last_staging_realpath_check = staging_runner_source.rindex("PRE_GATE_STAGING_PATH_REALPATH_INSIDE_GENERAL_ENVIRONMENT")
    venv_creation = staging_runner_source.index("python -m venv $StagingRoot")
    assert last_staging_realpath_check < venv_creation


def test_staging_runner_realpath_staging_guards_occur_before_any_pip_install(staging_runner_source: str):
    # Test 7: all real staging safety checks occur before any pip
    # install invocation.
    last_staging_realpath_check = staging_runner_source.rindex("PRE_GATE_STAGING_PATH_REALPATH_INSIDE_GENERAL_ENVIRONMENT")
    first_pip_install = staging_runner_source.index("pip install")
    assert last_staging_realpath_check < first_pip_install


def test_staging_runner_realpath_output_vs_staging_overlap_still_rejected(staging_runner_source: str):
    # Test 8: real StagingRoot / real OutputRoot equal or
    # ancestor/descendant remains rejected.
    output_vs_staging_realpath_block_start = staging_runner_source.index(
        "$realResolvedOutputRoot = Resolve-ExistingAncestorRealPath -Path $OutputRoot"
    )
    output_vs_staging_realpath_block = staging_runner_source[output_vs_staging_realpath_block_start:]
    assert "$realResolvedOutputRoot -eq $realResolvedStagingRoot" in output_vs_staging_realpath_block
    assert "$realResolvedStagingRoot.StartsWith($realResolvedOutputRoot" in output_vs_staging_realpath_block
    assert "$realResolvedOutputRoot.StartsWith($realResolvedStagingRoot" in output_vs_staging_realpath_block


def test_staging_runner_baseline_exact7_ordering_preserved_after_high2_remediation(staging_runner_source: str):
    # Test 9: predecessor exact-7 validation ordering remains intact.
    before_capture_index = staging_runner_source.index('$beforeFreezePath = Join-Path $OutputRoot "before_freeze.txt"')
    validation_invocation_index = staging_runner_source.index("--validate-predecessor-baseline-file $beforeFreezePath")
    successor_resolution_index = staging_runner_source.index("# E5 step 4:")
    assert before_capture_index < validation_invocation_index < successor_resolution_index


def test_staging_runner_single_successor_resolution_preserved_after_high2_remediation(staging_runner_source: str):
    # Test 10: exactly one successor-resolution invocation remains.
    successor_calls = re.findall(
        r"\$successorResolutionExitCode = Invoke-NativeProcessRawCapture",
        staging_runner_source,
    )
    assert len(successor_calls) == 1


def test_staging_runner_no_remove_item_reset_retry_preserved_after_high2_remediation(staging_runner_source: str):
    # Test 11: no Remove-Item/reset/retry.
    code_only = _strip_powershell_comment_lines(staging_runner_source)
    assert "Remove-Item" not in code_only
    lowered = code_only.lower()
    assert "git reset" not in lowered
    assert "git clean" not in lowered
    assert "git checkout" not in lowered


def test_staging_path_external_parent_symlink_toward_reserved_environment_rejected(tmp_path: Path):
    # Optional path-semantic fixture: an external parent symlink whose
    # real target is a stand-in reserved-environment directory, with a
    # NONEXISTENT child StagingRoot underneath it. The existing Python
    # semantic authority (validate_staging_path, unmodified by this
    # HIGH_2 remediation) already fails closed on any symlink encountered
    # while walking the path's ancestors -- rejecting outright rather
    # than following the symlink to inspect its target -- which is at
    # least as strict as the .ps1 runner's follow-and-re-check contract.
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlink support unavailable in this environment")
    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir()
    fake_canonical_environment = fake_repo / ".venv-real-execution"
    fake_canonical_environment.mkdir()
    external_parent = tmp_path / "external-parent-symlink"
    try:
        external_parent.symlink_to(fake_canonical_environment, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    nonexistent_staging_child = external_parent / "nonexistent-staging-root"
    assert not nonexistent_staging_child.exists()
    result = env_successor.validate_staging_path(nonexistent_staging_child, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_IS_SYMLINK_FAILURE


# =============================================================================
# HIGH_2B remediation: Resolve-ExistingAncestorRealPath (the .ps1 shared
# helper) must walk the COMPLETE existing ancestor chain, not merely the
# nearest existing ancestor. Root cause: the prior implementation found the
# nearest existing ancestor and checked LinkType only on that single item;
# a HIGHER ancestor further up the chain being a symlink/junction/reparse
# point -- while the nearest existing ancestor below it is an ordinary
# directory -- was not mechanically inspected.
# =============================================================================


@pytest.fixture(scope="module")
def resolve_existing_ancestor_realpath_helper_block(staging_runner_source: str) -> str:
    helper_start = staging_runner_source.index("function Resolve-ExistingAncestorRealPath")
    preflight_4b_start = staging_runner_source.index("# Preflight 4b: StagingRoot REAL-PATH protected-state guard")
    assert helper_start < preflight_4b_start
    return staging_runner_source[helper_start:preflight_4b_start]


def test_staging_runner_realpath_helper_walks_full_existing_ancestor_chain(
    resolve_existing_ancestor_realpath_helper_block: str,
):
    # Test 12: a dedicated second loop walks EVERY existing ancestor, from
    # the nearest existing one up to the filesystem root -- not merely
    # `$existingItem`, the single nearest-existing-ancestor variable used
    # by the pre-HIGH_2B implementation.
    helper_block = resolve_existing_ancestor_realpath_helper_block
    assert helper_block.count("while (") >= 2, "expected both the nearest-ancestor search loop and the full chain-walk loop"
    assert "while ($true)" in helper_block
    assert "$chainNode" in helper_block
    assert "$chainItem" in helper_block
    assert "$nearestExistingItem" in helper_block
    assert "$existingItem" not in helper_block
    # The chain walk terminates only at filesystem root, exactly like the
    # nearest-existing-ancestor search preceding it -- not after a single
    # inspection.
    assert helper_block.count("[string]::IsNullOrEmpty(") >= 2
    assert "Split-Path -Path $chainNode -Parent" in helper_block


def test_staging_runner_realpath_helper_rejects_any_reparse_point_anywhere_in_chain(
    resolve_existing_ancestor_realpath_helper_block: str,
):
    # Test 13: any existing ancestor anywhere in the chain -- not only the
    # nearest one -- that is a symlink/junction/reparse point is rejected,
    # via both LinkType and the ReparsePoint attributes bitmask (belt and
    # suspenders: junctions do not always populate LinkType the same way
    # symlinks do).
    helper_block = resolve_existing_ancestor_realpath_helper_block
    assert "PRE_GATE_PATH_REPARSE_ANCESTOR_CHATGPT_DECISION_REQUIRED" in helper_block
    assert "$chainItem.LinkType" in helper_block
    assert "$chainItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint" in helper_block


def test_staging_runner_realpath_helper_never_follows_reparse_target_and_continues(
    resolve_existing_ancestor_realpath_helper_block: str,
):
    # Test 14: the pre-HIGH_2B behavior of following a detected reparse
    # point's `.Target` and continuing resolution through it must be
    # entirely gone -- a detected reparse point is now an unconditional
    # fail-closed `throw`, never a followed alias.
    helper_block = resolve_existing_ancestor_realpath_helper_block
    assert ".Target" not in helper_block
    assert "Select-Object -First 1" not in helper_block


def test_staging_runner_realpath_helper_fails_closed_on_ancestor_inspection_failure(
    resolve_existing_ancestor_realpath_helper_block: str,
):
    # Test 15: an inability to inspect any ancestor's attributes -- not
    # only the nearest one -- is CHATGPT_DECISION_REQUIRED, never assumed
    # safe.
    helper_block = resolve_existing_ancestor_realpath_helper_block
    assert "try {" in helper_block
    assert "} catch {" in helper_block
    assert "-ErrorAction Stop" in helper_block
    assert helper_block.count("PRE_GATE_PATH_SAFETY_UNDETERMINED_CHATGPT_DECISION_REQUIRED") >= 2


def test_staging_runner_realpath_helper_shared_by_staging_and_output_root(staging_runner_source: str):
    # Test 16: the same strengthened helper governs BOTH -StagingRoot and
    # -OutputRoot (requirement #6 of the HIGH_2B remediation contract) --
    # neither path gets a separate, weaker chain-walk implementation.
    assert "$realResolvedStagingRoot = Resolve-ExistingAncestorRealPath -Path $StagingRoot" in staging_runner_source
    assert "$realResolvedOutputRoot = Resolve-ExistingAncestorRealPath -Path $OutputRoot" in staging_runner_source


def _make_alias_with_existing_normal_parent(tmp_path: Path, reserved_leaf_name: str) -> Path:
    """Build the MANDATORY regression shape: `external_alias` is a
    symlink toward a stand-in reserved-environment directory;
    `external_alias/existing-normal-parent/` EXISTS as an ORDINARY
    (non-symlink) directory; the caller then appends a NONEXISTENT leaf
    underneath it. Returns that nonexistent leaf path. Raises
    `pytest.skip` if symlinks are unavailable in this environment.
    """
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlink support unavailable in this environment")
    fake_repo = tmp_path / "fake-repo-alias-chain"
    fake_repo.mkdir()
    fake_reserved_environment = fake_repo / reserved_leaf_name
    fake_reserved_environment.mkdir()
    external_alias = tmp_path / "external-alias-toward-reserved-environment"
    try:
        external_alias.symlink_to(fake_reserved_environment, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    existing_normal_parent = external_alias / "existing-normal-parent"
    existing_normal_parent.mkdir()
    assert existing_normal_parent.exists()
    assert not existing_normal_parent.is_symlink()

    nonexistent_leaf = existing_normal_parent / "nonexistent-leaf"
    assert not nonexistent_leaf.exists()
    return nonexistent_leaf


def test_staging_path_nested_normal_child_under_higher_alias_rejected(tmp_path: Path):
    # MANDATORY regression case (runtime fixture): the nearest existing
    # ancestor of StagingRoot ("existing-normal-parent") is an ORDINARY
    # directory -- not itself a symlink -- but a HIGHER ancestor
    # ("external_alias") IS a symlink/junction toward a stand-in reserved
    # environment. This shape must still be rejected before any mutation.
    # The Python semantic authority (validate_staging_path, unmodified by
    # this HIGH_2B remediation) already walks every ancestor -- existing
    # or not -- of the supplied path, so it already covers this exact
    # shape; this proves that contract explicitly for the specific shape
    # named in the HIGH_2B task rather than relying on the more general
    # existing symlink regression above.
    nonexistent_staging_child = _make_alias_with_existing_normal_parent(tmp_path, ".venv-real-execution")
    result = env_successor.validate_staging_path(nonexistent_staging_child, repo_root=REPO_ROOT)
    assert result.status == env_successor.STAGING_PATH_IS_SYMLINK_FAILURE


def test_output_path_nested_normal_child_under_higher_alias_rejected(tmp_path: Path):
    # Analogous OutputRoot regression: the shared .ps1 helper governs both
    # -StagingRoot and -OutputRoot path-safety decisions, so both must
    # reject the same "ordinary nearest ancestor beneath a higher alias"
    # shape. Uses a distinct, non-overlapping staging_root argument so the
    # StagingRoot-overlap checks in validate_output_path never trigger
    # first and mask the symlink-chain check under test.
    nonexistent_output_child = _make_alias_with_existing_normal_parent(tmp_path, ".venv")
    unrelated_staging_root = tmp_path / "unrelated-staging-root-not-created"
    result = env_successor.validate_output_path(
        nonexistent_output_child,
        staging_root=unrelated_staging_root,
        repo_root=REPO_ROOT,
    )
    assert result.status == env_successor.OUTPUT_PATH_IS_SYMLINK_FAILURE


# =============================================================================
# No real source/PDF/network/trading_dates/profitability paths anywhere in
# the new E2 files
# =============================================================================


@pytest.fixture(scope="module")
def all_e2_source_texts() -> dict:
    return {
        "v9_014_pdf_env_successor.py": (REPO_ROOT / "scripts" / "v9_014_pdf_env_successor.py").read_text(encoding="utf-8"),
        "generate_v9_014_synthetic_pdf_probe.py": (
            REPO_ROOT / "scripts" / "generate_v9_014_synthetic_pdf_probe.py"
        ).read_text(encoding="utf-8"),
        "v9_014_pdf_env_successor_staging_runner.ps1": STAGING_RUNNER_PATH.read_text(encoding="utf-8"),
    }


def _strip_python_docstrings_and_comments(source: str) -> str:
    """Remove triple-quoted docstring blocks and '#'-led comment lines,
    so an explanatory negative claim in prose (e.g. "makes no
    profitability claim") cannot itself trip a substring search for the
    forbidden term the code must never actually use.
    """
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return "\n".join(line for line in without_docstrings.splitlines() if not line.strip().startswith("#"))


def test_no_trading_dates_or_profitability_surface(all_e2_source_texts: dict):
    forbidden = ("trading_dates", "profitability", "backtest", "classify_date", "DateClassification")
    for filename, text in all_e2_source_texts.items():
        code_only = (
            _strip_python_docstrings_and_comments(text)
            if filename.endswith(".py")
            else _strip_powershell_comment_lines(text)
        )
        for marker in forbidden:
            assert marker not in code_only, f"{marker!r} unexpectedly present in {filename} outside comments/docstrings"


def test_no_real_jpx_network_host_referenced(all_e2_source_texts: dict):
    forbidden = ("jpx.co.jp", "www.jpx", "finance.yahoo", "yahoo.co.jp", "j-quants")
    for filename, text in all_e2_source_texts.items():
        lowered = text.lower()
        for marker in forbidden:
            assert marker not in lowered, f"{marker!r} unexpectedly present in {filename}"


def test_no_float_conversion_or_arithmetic_in_python_tooling(all_e2_source_texts: dict):
    text = all_e2_source_texts["v9_014_pdf_env_successor.py"]
    assert "float(" not in text


def test_no_pip_install_invocation_inside_python_module():
    # The Python module must never itself shell out to pip / create an
    # environment -- only the (unexecuted) PowerShell runner contains
    # pip/venv invocations, and only as source text reviewed, not run.
    source = (REPO_ROOT / "scripts" / "v9_014_pdf_env_successor.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "os.system" not in source
    assert "pip install" not in source
    assert "venv.create" not in source
