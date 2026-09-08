import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src import v9_017_source_b_2017_public_schema_discovery_runner as runner


def _document(*tables: str) -> bytes:
    return ("<html><body>" + "".join(tables) + "</body></html>").encode("utf-8")


def _table(*rows: str) -> str:
    return "<table>" + "".join(f"<tr>{row}</tr>" for row in rows) + "</table>"


def _row(*cells: str) -> str:
    return "".join(cells)


def _cell(text: str, tag: str = "td") -> str:
    return f"<{tag}>{text}</{tag}>"


def _bound_run(
    tmp_path: Path,
    page_bytes: bytes,
    *,
    discovery_fn=runner.discover_public_schema,
    expected_count: int | None = None,
    expected_sha: str | None = None,
    **kwargs,
):
    page_path = tmp_path / "synthetic-year-page.html"
    output_root = tmp_path / "output"
    page_path.write_bytes(page_bytes)
    return runner._run_execution(
        year_page=page_path,
        output_root=output_root,
        expected_git_sha="0" * 40,
        confirmation=runner.CONFIRMATION_CONTRACT,
        execute_discovery=True,
        _expected_year_byte_count=(
            len(page_bytes) if expected_count is None else expected_count
        ),
        _expected_year_sha256=(
            hashlib.sha256(page_bytes).hexdigest()
            if expected_sha is None
            else expected_sha
        ),
        _discovery_fn=discovery_fn,
        **kwargs,
    ), page_path, output_root


def test_pure_schema_exact_and_dimensions() -> None:
    page = _document(
        _table(
            _row(_cell("  01  ", "th"), _cell("hidden quantity")),
            "",
            _row(_cell("  first\ncell  "), _cell("arbitrary value")),
        ),
        _table(_row(_cell("second table"))),
    )
    result = runner.discover_public_schema(page).to_dict()

    assert list(result) == [
        "schema_version",
        "status",
        "failure_class",
        "parser_success",
        "table_count",
        "tables",
        "observation_count",
        "observations",
    ]
    assert result["status"] == "PASS"
    assert result["parser_success"] is True
    assert result["table_count"] == 2
    assert result["tables"] == [
        {
            "table_index": 0,
            "row_count": 3,
            "max_column_count": 2,
            "rows": [
                {"row_index": 0, "cell_count": 2},
                {"row_index": 1, "cell_count": 0},
                {"row_index": 2, "cell_count": 2},
            ],
        },
        {
            "table_index": 1,
            "row_count": 1,
            "max_column_count": 1,
            "rows": [{"row_index": 0, "cell_count": 1}],
        },
    ]
    assert result["observations"] == [
        {
            "table_index": 0,
            "row_index": 0,
            "column_index": 0,
            "role": "FIRST_CELL",
            "normalized_text": "01",
        },
        {
            "table_index": 0,
            "row_index": 0,
            "column_index": 0,
            "role": "TH",
            "normalized_text": "01",
        },
        {
            "table_index": 0,
            "row_index": 2,
            "column_index": 0,
            "role": "FIRST_CELL",
            "normalized_text": "first cell",
        },
        {
            "table_index": 1,
            "row_index": 0,
            "column_index": 0,
            "role": "FIRST_CELL",
            "normalized_text": "second table",
        },
    ]
    assert result["observation_count"] == 4


def test_structural_text_is_opaque_and_nonstructural_cells_are_omitted() -> None:
    page = _document(
        _table(
            _row(
                _cell("01!?"),
                _cell(
                    "ARBITRARY_NONSTRUCTURAL_TEXT",
                ),
            ),
            _row(
                _cell(
                    '<a href="https://example.invalid/private.pdf"> Jan. </a>',
                    "th",
                ),
                _cell("raw html must not appear"),
            ),
        )
    )
    result = runner.discover_public_schema(page)
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert "01!?" in serialized
    assert "Jan." in serialized
    assert "ARBITRARY_NONSTRUCTURAL_TEXT" not in serialized
    assert "raw html must not appear" not in serialized
    assert "https://example.invalid/private.pdf" not in serialized
    assert "<a" not in serialized
    assert "href" not in serialized


@pytest.mark.parametrize("value", [None, "not-bytes", bytearray(b"<table></table>")])
def test_non_bytes_fails_closed(value) -> None:
    result = runner.discover_public_schema(value).to_dict()
    assert result == {
        "schema_version": runner.PURE_SCHEMA_VERSION,
        "status": "FAIL",
        "failure_class": "IMPLEMENTATION_FAILURE",
        "parser_success": False,
        "table_count": 0,
        "tables": [],
        "observation_count": 0,
        "observations": [],
    }


def test_malformed_html_fails_closed() -> None:
    result = runner.discover_public_schema(b"<table><tr><td>unclosed").to_dict()
    assert result["status"] == "FAIL"
    assert result["failure_class"] == "IMPLEMENTATION_FAILURE"
    assert result["parser_success"] is False
    assert result["observations"] == []


def test_safe_repr_has_no_raw_payload_or_href() -> None:
    page = _document(
        _table(
            _row(
                _cell('<a href="https://example.invalid/x">Visible</a>', "th"),
                _cell("private-value"),
            )
        )
    )
    rendered = repr(runner.discover_public_schema(page))
    assert "Visible" in rendered
    assert "https://example.invalid/x" not in rendered
    assert "private-value" not in rendered
    assert "<a" not in rendered


def test_cli_without_execute_does_not_read_or_create(tmp_path: Path, capsys) -> None:
    missing_page = tmp_path / "does-not-exist.html"
    output_root = tmp_path / "output"
    exit_code = runner.main(
        [
            "--year-page",
            str(missing_page),
            "--output-root",
            str(output_root),
            "--expected-git-sha",
            "0" * 40,
            "--confirmation",
            runner.CONFIRMATION_CONTRACT,
        ]
    )
    captured = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert captured["reason"] == "EXECUTE_DISCOVERY_REQUIRED"
    assert captured["semantic_discovery_invocations"] == 0
    assert not output_root.exists()


def test_wrong_confirmation_and_invalid_sha_fail_before_input_read(tmp_path: Path) -> None:
    missing_page = tmp_path / "does-not-exist.html"
    for expected_sha, confirmation, reason in [
        ("0" * 40, "wrong", "CONFIRMATION_CONTRACT_MISMATCH"),
        ("bad", runner.CONFIRMATION_CONTRACT, "EXPECTED_GIT_SHA_INVALID"),
    ]:
        result = runner._run_execution(
            year_page=missing_page,
            output_root=tmp_path / reason,
            expected_git_sha=expected_sha,
            confirmation=confirmation,
            execute_discovery=True,
        )
        assert result["reason"] == reason
        assert result["semantic_discovery_invocations"] == 0


def test_output_collision_fails_before_input_read(tmp_path: Path) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()
    result = runner._run_execution(
        year_page=tmp_path / "does-not-exist.html",
        output_root=output_root,
        expected_git_sha="0" * 40,
        confirmation=runner.CONFIRMATION_CONTRACT,
        execute_discovery=True,
    )
    assert result["reason"] == "OUTPUT_ROOT_COLLISION"
    assert result["semantic_discovery_invocations"] == 0


def test_byte_count_mismatch_does_not_probe(tmp_path: Path) -> None:
    page = _document(_table(_row(_cell("label"))))
    calls: list[bytes] = []

    def discovery(data: bytes):
        calls.append(data)
        return runner.discover_public_schema(data)

    result, _, output_root = _bound_run(
        tmp_path,
        page,
        discovery_fn=discovery,
        expected_count=len(page) + 1,
    )
    assert result["reason"] == "YEAR_PAGE_BYTE_COUNT_MISMATCH"
    assert result["semantic_discovery_invocations"] == 0
    assert calls == []
    assert (output_root / "attempt.json").exists()
    assert (output_root / "failure.json").exists()


def test_sha_mismatch_does_not_probe(tmp_path: Path) -> None:
    page = _document(_table(_row(_cell("label"))))
    calls: list[bytes] = []

    def discovery(data: bytes):
        calls.append(data)
        return runner.discover_public_schema(data)

    result, _, output_root = _bound_run(
        tmp_path,
        page,
        discovery_fn=discovery,
        expected_sha="0" * 64,
    )
    assert result["reason"] == "YEAR_PAGE_SHA256_MISMATCH"
    assert result["semantic_discovery_invocations"] == 0
    assert calls == []
    assert (output_root / "attempt.json").exists()
    assert (output_root / "failure.json").exists()


def test_input_read_failure_is_safe(tmp_path: Path) -> None:
    result = runner._run_execution(
        year_page=tmp_path / "does-not-exist.html",
        output_root=tmp_path / "output",
        expected_git_sha="0" * 40,
        confirmation=runner.CONFIRMATION_CONTRACT,
        execute_discovery=True,
    )
    assert result == {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "FAIL",
        "failure_class": "GOVERNANCE_FAILURE",
        "reason": "YEAR_PAGE_READ_FAILURE",
        "semantic_discovery_invocations": 0,
    }


def test_exact_binding_calls_real_discovery_once_and_persists_safe_set(
    tmp_path: Path,
) -> None:
    page = _document(
        _table(
            _row(_cell("  2017-01  ", "th"), _cell("not emitted")),
            _row(_cell("first")),
        )
    )
    calls: list[bytes] = []

    def discovery(data: bytes):
        calls.append(data)
        return runner.discover_public_schema(data)

    result, page_path, output_root = _bound_run(
        tmp_path, page, discovery_fn=discovery
    )
    assert result["status"] == "DISCOVERY_COMPLETE"
    assert result["semantic_discovery_invocations"] == 1
    assert calls == [page]
    assert result["discovery_result"] == runner.discover_public_schema(page).to_dict()
    assert set(path.name for path in output_root.iterdir()) == {
        "attempt.json",
        "result.json",
        "complete.json",
    }
    assert json.loads((output_root / "attempt.json").read_text()) == {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "IN_PROGRESS",
        "expected_git_sha": "0" * 40,
        "confirmation_contract": runner.CONFIRMATION_CONTRACT,
        "target_year": 2017,
        "expected_year_byte_count": len(page),
        "expected_year_sha256": hashlib.sha256(page).hexdigest(),
        "target_semantic_discovery_invocations": 1,
    }
    assert json.loads((output_root / "complete.json").read_text()) == {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "DISCOVERY_COMPLETE",
        "semantic_discovery_invocations": 1,
    }
    assert not (output_root / "failure.json").exists()
    assert str(page_path) not in json.dumps(result)


def test_input_is_opened_once(tmp_path: Path, monkeypatch) -> None:
    page = _document(_table(_row(_cell("label"))))
    page_path = tmp_path / "page.html"
    page_path.write_bytes(page)
    output_root = tmp_path / "output"
    open_count = 0
    original_open = Path.open

    def tracking_open(path_object, *args, **kwargs):
        nonlocal open_count
        if path_object == page_path:
            open_count += 1
        return original_open(path_object, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    result = runner._run_execution(
        year_page=page_path,
        output_root=output_root,
        expected_git_sha="0" * 40,
        confirmation=runner.CONFIRMATION_CONTRACT,
        execute_discovery=True,
        _expected_year_byte_count=len(page),
        _expected_year_sha256=hashlib.sha256(page).hexdigest(),
    )
    assert result["status"] == "DISCOVERY_COMPLETE"
    assert open_count == 1


def test_discovery_failure_is_safe_and_preserves_attempt(tmp_path: Path) -> None:
    page = _document(_table(_row(_cell("label"))))

    def failing_discovery(data: bytes):
        raise RuntimeError("must not leak")

    result, _, output_root = _bound_run(
        tmp_path, page, discovery_fn=failing_discovery
    )
    assert result["reason"] == "DISCOVERY_FAILURE"
    assert result["semantic_discovery_invocations"] == 1
    assert (output_root / "attempt.json").exists()
    assert (output_root / "failure.json").exists()
    assert "must not leak" not in json.dumps(result)


def test_exclusive_write_cannot_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "marker.json"
    runner._write_exclusive_json(target, {"status": "first"})
    with pytest.raises(FileExistsError):
        runner._write_exclusive_json(target, {"status": "second"})
    assert json.loads(target.read_text()) == {"status": "first"}


def test_production_constants_and_wrapper_have_no_selection_or_network_path() -> None:
    assert runner.TARGET_YEAR == 2017
    assert runner.EXPECTED_YEAR_BYTE_COUNT == 98936
    assert (
        runner.EXPECTED_YEAR_SHA256
        == "1dc982e97b1d4ce7d52bc25631ddc46a219d22f82b393881f40e5d2478177821"
    )
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "requests" not in source
    assert "urllib" not in source
    assert "category" not in source.lower()
    assert "report-label" not in source.lower()
    assert "month-grammar" not in source.lower()


def test_direct_script_external_cwd_without_execute_is_safe(tmp_path: Path) -> None:
    external_cwd = tmp_path / "external-cwd"
    external_cwd.mkdir()
    output_root = tmp_path / "output"
    missing_page = tmp_path / "missing-placeholder.html"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--year-page",
            str(missing_page),
            "--output-root",
            str(output_root),
            "--expected-git-sha",
            "0" * 40,
            "--confirmation",
            runner.CONFIRMATION_CONTRACT,
        ],
        cwd=external_cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stderr == ""
    safe_result = json.loads(completed.stdout)
    assert safe_result["reason"] == "EXECUTE_DISCOVERY_REQUIRED"
    assert safe_result["semantic_discovery_invocations"] == 0
    assert not output_root.exists()


def test_direct_script_external_cwd_invalid_sha_is_safe(tmp_path: Path) -> None:
    external_cwd = tmp_path / "external-cwd"
    external_cwd.mkdir()
    output_root = tmp_path / "output"
    missing_page = tmp_path / "missing-placeholder.html"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--year-page",
            str(missing_page),
            "--output-root",
            str(output_root),
            "--expected-git-sha",
            "invalid-sha",
            "--confirmation",
            runner.CONFIRMATION_CONTRACT,
            "--execute-discovery",
        ],
        cwd=external_cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stderr == ""
    safe_result = json.loads(completed.stdout)
    assert safe_result["reason"] == "EXPECTED_GIT_SHA_INVALID"
    assert safe_result["semantic_discovery_invocations"] == 0
    assert not output_root.exists()
