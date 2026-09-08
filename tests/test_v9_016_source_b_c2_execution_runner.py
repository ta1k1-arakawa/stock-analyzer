import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.v9_016_source_b_c2_execution_runner as runner
from src.v9_016_source_b_2017_month_header_structural_probe import (
    CATEGORY_NAMES,
    probe_2017_month_header_structure,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "src" / "v9_016_source_b_c2_execution_runner.py"
GIT_SHA = "a" * 40


def _synthetic_page() -> bytes:
    return (
        b'<table><tr><th>Other</th><th>2017-01</th></tr><tr><td>'
        b"Stock Trading Volume &amp; Value</td>"
        b'<td><a href="synthetic-child.pdf">synthetic</a></td>'
        b"</tr></table>"
    )


def _run_synthetic(tmp_path, payload=None, *, probe=None, **overrides):
    payload = _synthetic_page() if payload is None else payload
    input_path = tmp_path / "synthetic-year-page.html"
    output_root = tmp_path / "synthetic-output"
    input_path.write_bytes(payload)
    expected_count = overrides.pop("expected_year_byte_count", len(payload))
    expected_sha = overrides.pop(
        "expected_year_sha256", hashlib.sha256(payload).hexdigest()
    )
    return runner._run_c2_with_binding(
        input_path,
        output_root,
        overrides.pop("expected_git_sha", GIT_SHA),
        overrides.pop("confirmation", runner.CONFIRMATION_CONTRACT),
        execute_c2=overrides.pop("execute_c2", True),
        expected_year_byte_count=expected_count,
        expected_year_sha256=expected_sha,
        probe=probe or runner.probe_2017_month_header_structure,
    ), input_path, output_root


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_missing_execute_flag_fails_before_input_read(tmp_path, monkeypatch):
    calls = 0

    def fail_if_read(_):
        nonlocal calls
        calls += 1
        raise AssertionError("year page must not be read")

    monkeypatch.setattr(Path, "read_bytes", fail_if_read)
    receipt, _, output_root = _run_synthetic(tmp_path, execute_c2=False)

    assert receipt["reason"] == "EXECUTE_C2_FLAG_REQUIRED"
    assert calls == 0
    assert not output_root.exists()


def test_cli_without_execute_flag_is_safe_from_external_working_directory(tmp_path):
    output_root = tmp_path / "must-not-be-created"
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--year-page",
        str(tmp_path / "private-real-input.html"),
        "--output-root",
        str(output_root),
        "--expected-git-sha",
        GIT_SHA,
        "--confirmation",
        runner.CONFIRMATION_CONTRACT,
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert receipt["reason"] == "EXECUTE_C2_FLAG_REQUIRED"
    assert not output_root.exists()
    assert "private-real-input.html" not in completed.stdout
    assert completed.stderr == ""


def test_exact_confirmation_is_required(tmp_path):
    receipt, _, output_root = _run_synthetic(tmp_path, confirmation="wrong")

    assert receipt["reason"] == "CONFIRMATION_CONTRACT_MISMATCH"
    assert not output_root.exists()


def test_invalid_expected_sha_fails_before_network_or_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _: (_ for _ in ()).throw(AssertionError("must not read")),
    )
    receipt, _, output_root = _run_synthetic(tmp_path, expected_git_sha="not-a-sha")

    assert receipt["reason"] == "EXPECTED_GIT_SHA_INVALID"
    assert not output_root.exists()


def test_output_root_collision_precedes_input_read(tmp_path, monkeypatch):
    output_root = tmp_path / "synthetic-output"
    output_root.mkdir()
    calls = 0

    def fail_if_read(_):
        nonlocal calls
        calls += 1
        raise AssertionError("collision must precede read")

    monkeypatch.setattr(Path, "read_bytes", fail_if_read)
    input_path = tmp_path / "missing-year-page.html"
    receipt = runner._run_c2_with_binding(
        input_path,
        output_root,
        GIT_SHA,
        runner.CONFIRMATION_CONTRACT,
        execute_c2=True,
        expected_year_byte_count=1,
        expected_year_sha256="b" * 64,
        probe=runner.probe_2017_month_header_structure,
    )

    assert receipt["reason"] == "OUTPUT_ROOT_COLLISION"
    assert calls == 0


def test_exact_synthetic_binding_success_uses_real_c1_probe(tmp_path):
    receipt, _, output_root = _run_synthetic(tmp_path)

    assert receipt["status"] == "C2_PROBE_EXECUTION_COMPLETE"
    assert receipt["probe_invocations"] == 1
    assert receipt["probe_result"] == probe_2017_month_header_structure(_synthetic_page()).to_dict()
    assert (output_root / "attempt.json").exists()
    assert (output_root / "result.json").exists()
    assert (output_root / "complete.json").exists()


def test_byte_count_mismatch_does_not_invoke_probe(tmp_path):
    calls = 0

    def counted_probe(page_bytes):
        nonlocal calls
        calls += 1
        return probe_2017_month_header_structure(page_bytes)

    receipt, _, output_root = _run_synthetic(
        tmp_path,
        probe=counted_probe,
        expected_year_byte_count=999,
    )

    assert receipt["reason"] == "YEAR_PAGE_BYTE_COUNT_MISMATCH"
    assert receipt["probe_invocations"] == 0
    assert calls == 0
    assert _read_json(output_root / "failure.json")["probe_invocations"] == 0


def test_sha_mismatch_does_not_invoke_probe(tmp_path):
    calls = 0

    def counted_probe(page_bytes):
        nonlocal calls
        calls += 1
        return probe_2017_month_header_structure(page_bytes)

    receipt, _, output_root = _run_synthetic(
        tmp_path,
        probe=counted_probe,
        expected_year_sha256="b" * 64,
    )

    assert receipt["reason"] == "YEAR_PAGE_SHA256_MISMATCH"
    assert receipt["probe_invocations"] == 0
    assert calls == 0
    assert _read_json(output_root / "failure.json")["probe_invocations"] == 0


def test_input_read_failure_is_safe_and_durable(tmp_path):
    output_root = tmp_path / "synthetic-output"
    missing_input = tmp_path / "private-missing-year-page.html"
    receipt = runner._run_c2_with_binding(
        missing_input,
        output_root,
        GIT_SHA,
        runner.CONFIRMATION_CONTRACT,
        execute_c2=True,
        expected_year_byte_count=1,
        expected_year_sha256="b" * 64,
        probe=runner.probe_2017_month_header_structure,
    )

    assert receipt["reason"] == "YEAR_PAGE_READ_FAILURE"
    assert (output_root / "attempt.json").exists()
    assert (output_root / "failure.json").exists()
    assert "private-missing-year-page.html" not in json.dumps(receipt)


def test_bound_bytes_call_c1_exactly_once(tmp_path):
    calls = []

    def counted_probe(page_bytes):
        calls.append(page_bytes)
        return probe_2017_month_header_structure(page_bytes)

    payload = _synthetic_page()
    receipt, _, _ = _run_synthetic(tmp_path, payload, probe=counted_probe)

    assert receipt["status"] == "C2_PROBE_EXECUTION_COMPLETE"
    assert calls == [payload]


def test_c1_probe_failure_is_safe(tmp_path):
    calls = 0

    def failing_probe(_):
        nonlocal calls
        calls += 1
        return probe_2017_month_header_structure("not-bytes")

    receipt, _, output_root = _run_synthetic(tmp_path, probe=failing_probe)

    assert receipt["reason"] == "C1_PROBE_FAILURE"
    assert receipt["failure_class"] == "IMPLEMENTATION_FAILURE"
    assert receipt["probe_invocations"] == 1
    assert calls == 1
    assert (output_root / "attempt.json").exists()
    assert (output_root / "failure.json").exists()


def test_success_persists_safe_attempt_result_complete_and_no_failure(tmp_path):
    receipt, _, output_root = _run_synthetic(tmp_path)

    attempt = _read_json(output_root / "attempt.json")
    result = _read_json(output_root / "result.json")
    complete = _read_json(output_root / "complete.json")
    assert attempt["status"] == "IN_PROGRESS"
    assert attempt["target_year"] == 2017
    assert attempt["target_probe_invocations"] == 1
    assert result == receipt
    assert complete == {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "C2_PROBE_EXECUTION_COMPLETE",
        "probe_invocations": 1,
    }
    assert not (output_root / "failure.json").exists()


def test_result_contains_exact_safe_c1_schema_without_adjudication(tmp_path):
    receipt, _, _ = _run_synthetic(tmp_path)
    probe_result = receipt["probe_result"]

    assert set(probe_result) == {
        "schema_version",
        "status",
        "failure_class",
        "parser_success",
        "table_count",
        "report_cell_count",
        "report_row_count",
        "categories",
        "legacy_candidate_count",
    }
    assert set(probe_result["categories"]) == set(CATEGORY_NAMES)
    assert "admissible" not in json.dumps(receipt)
    assert "selected_category" not in json.dumps(receipt)


def test_probe_result_is_not_modified_or_reinterpreted(tmp_path):
    payload = _synthetic_page()
    expected = probe_2017_month_header_structure(payload).to_dict()
    receipt, _, _ = _run_synthetic(tmp_path, payload)

    assert receipt["probe_result"] == expected
    assert receipt["probe_result"]["legacy_candidate_count"] == 1


def test_failure_preserves_attempt_and_failure_only(tmp_path):
    receipt, _, output_root = _run_synthetic(tmp_path, expected_year_sha256="c" * 64)

    assert receipt["reason"] == "YEAR_PAGE_SHA256_MISMATCH"
    assert _read_json(output_root / "attempt.json")["status"] == "IN_PROGRESS"
    assert _read_json(output_root / "failure.json") == receipt
    assert not (output_root / "result.json").exists()
    assert not (output_root / "complete.json").exists()


def test_success_safe_serialization_contains_no_raw_href_html_path_or_url(tmp_path):
    receipt, _, output_root = _run_synthetic(tmp_path)
    serialized = json.dumps(receipt, sort_keys=True)
    serialized_files = json.dumps(
        {
            name: _read_json(output_root / name)
            for name in ("attempt.json", "result.json", "complete.json")
        },
        sort_keys=True,
    )

    for text in (serialized, serialized_files):
        assert "synthetic-child.pdf" not in text
        assert "<table>" not in text
        assert "Stock Trading Volume & Value" not in text
        assert "synthetic-year-page.html" not in text
        assert "http" not in text


def test_only_one_year_page_argument_and_one_read_path_exist():
    parameters = inspect.signature(runner.run_c2).parameters

    assert set(parameters) == {
        "year_page",
        "output_root",
        "expected_git_sha",
        "confirmation",
        "execute_c2",
    }
    assert "other_year" not in inspect.getsource(runner.run_c2)


def test_output_files_use_exclusive_non_overwriting_boundary(tmp_path):
    first, _, output_root = _run_synthetic(tmp_path)
    original_result = (output_root / "result.json").read_bytes()
    second = runner._run_c2_with_binding(
        tmp_path / "synthetic-year-page.html",
        output_root,
        GIT_SHA,
        runner.CONFIRMATION_CONTRACT,
        execute_c2=True,
        expected_year_byte_count=len(_synthetic_page()),
        expected_year_sha256=hashlib.sha256(_synthetic_page()).hexdigest(),
        probe=runner.probe_2017_month_header_structure,
    )

    assert first["status"] == "C2_PROBE_EXECUTION_COMPLETE"
    assert second["reason"] == "OUTPUT_ROOT_COLLISION"
    assert (output_root / "result.json").read_bytes() == original_result


def test_production_cli_binds_immutable_year_constants(tmp_path):
    payload = _synthetic_page()
    input_path = tmp_path / "synthetic-year-page.html"
    output_root = tmp_path / "production-constant-check"
    input_path.write_bytes(payload)
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--year-page",
        str(input_path),
        "--output-root",
        str(output_root),
        "--expected-git-sha",
        GIT_SHA,
        "--confirmation",
        runner.CONFIRMATION_CONTRACT,
        "--execute-c2",
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert receipt["reason"] == "YEAR_PAGE_BYTE_COUNT_MISMATCH"
    attempt = _read_json(output_root / "attempt.json")
    assert attempt["expected_year_byte_count"] == 98936
    assert attempt["expected_year_sha256"] == (
        "1dc982e97b1d4ce7d52bc25631ddc46a219d22f82b393881f40e5d2478177821"
    )
    assert "private" not in completed.stdout.lower()
    assert completed.stderr == ""


def test_runner_has_no_network_capable_import_or_call_path():
    source = RUNNER_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "requests",
        "urllib",
        "urlopen",
        "socket",
        "http.client",
        "probe_calibration_bundle",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("bad_sha", ["", "f" * 39, "f" * 41, "g" * 40])
def test_all_non_exact_sha_forms_fail_closed(tmp_path, bad_sha):
    receipt, _, _ = _run_synthetic(tmp_path, expected_git_sha=bad_sha)

    assert receipt["reason"] == "EXPECTED_GIT_SHA_INVALID"
    assert receipt["probe_invocations"] == 0
