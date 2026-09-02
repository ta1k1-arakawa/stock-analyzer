"""Targeted synthetic tests for V9_014 PDF real-execution environment
successor Stage E2 offline tooling.

Every test here is synthetic and offline: no network, no `pip install`, no
real or staging environment creation/mutation, and no import of a real
installed `pdfplumber` (the lazy-import boundary is exercised exclusively
via an injected fake module). Real `pdfplumber==0.11.10` execution occurs
only at the later, separately reviewed Stage E6/E10/E14 checkpoints.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import scripts.v9_014_pdf_env_successor as env_successor
import scripts.generate_v9_014_synthetic_pdf_probe as pdf_generator

REPO_ROOT = env_successor.REPO_ROOT
STAGING_RUNNER_PATH = REPO_ROOT / "scripts" / "v9_014_pdf_env_successor_staging_runner.ps1"


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
    # No looping/retry construct wraps the successor-resolution
    # invocation, and no git history-mutating command appears anywhere.
    lowered = staging_runner_source.lower()
    assert "while (" not in staging_runner_source
    assert "for (" not in staging_runner_source
    assert "do {" not in staging_runner_source
    assert ".reset(" not in lowered
    assert "git reset" not in lowered
    assert "git clean" not in lowered
    assert "git checkout" not in lowered


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
    # Count invocations that reference BOTH the predecessor lock as a
    # constraint (-c ...) and the direct spec as the requirement (-r ...)
    # in the same pip install call -- the successor-resolution shape.
    successor_calls = re.findall(
        r"pip install `\s*\n\s*-c \$predecessorLockRelativePath `\s*\n\s*-r \$directSpecRelativePath",
        staging_runner_source,
    )
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
