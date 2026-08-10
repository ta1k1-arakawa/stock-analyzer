from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path

import pytest

from scripts import build_v8_partition_manifest as cli
from src import v8_partition as partition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v8_partition_manifest.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


@pytest.fixture(autouse=True)
def test_production_git_provenance(monkeypatch):
    """Keep fake-only production-path tests deterministic without weakening CLI."""
    monkeypatch.setattr(cli, "resolve_verified_production_git_commit", lambda _: "a" * 40)
    yield


@pytest.fixture(scope="module")
def synthetic_result():
    return cli.run_synthetic_partition_test()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _cli_option_names() -> set[str]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }


def test_cli_has_exactly_the_authorized_options():
    """The three mutually-exclusive modes plus their shared arguments."""
    assert _cli_option_names() == {
        "--synthetic-test", "--production-build-manifest", "--production-source-preflight",
        "--output-path", "--confirmation",
    }


def test_cli_has_no_bypass_option():
    """Checks actual add_argument() calls, not raw source text -- this
    file's own module docstring names bypass-flag examples of what must NOT
    exist (e.g. "no --skip-source-hash"), so a substring search over the
    whole source would false-positive on its own documentation."""
    options = _cli_option_names()
    for flag in ("--skip-source-hash", "--force", "--ignore-parity", "--network", "--all"):
        assert flag not in options


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run([PYTHON, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--force"], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0


def test_cli_subprocess_exit_zero_and_reports_result():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-test"], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["network_requests"] == 0
    assert payload["real_partition_created"] is False


def test_cli_subprocess_rejects_both_modes_together():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-test", "--production-build-manifest"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0


def test_cli_subprocess_rejects_source_preflight_with_partition_build():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--production-source-preflight", "--production-build-manifest"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0


def test_cli_subprocess_production_missing_args_errors_before_any_network(tmp_path):
    """--production-build-manifest without --output-path/--confirmation must
    fail via argparse before main() ever calls the fetch function -- a real
    subprocess run proves this without needing to fake urlopen."""
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--production-build-manifest"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_production_wrong_confirmation_blocks_before_any_network(tmp_path):
    """Confirmation is checked before run_production_partition_build (and
    therefore before fetch_real_jpx_source) is ever called, so this real
    subprocess run is safe to execute without any network fake: a wrong
    confirmation can never reach the network path."""
    output_path = tmp_path / "would-be-manifest.json"
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--production-build-manifest",
         "--output-path", str(output_path), "--confirmation", "WRONG_CONFIRMATION"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"status": "BLOCKED", "reason": "CONFIRMATION_MISMATCH"}
    assert not output_path.exists()


def test_cli_source_preflight_wrong_confirmation_blocks_before_any_network(monkeypatch):
    def forbidden():
        raise AssertionError("source preflight runner reached")

    monkeypatch.setattr(cli, "run_production_source_preflight", forbidden)
    result = cli.main([
        "--production-source-preflight", "--confirmation", "WRONG_CONFIRMATION",
    ])
    assert result == 2


def test_cli_source_preflight_requires_confirmation_before_any_network():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--production-source-preflight"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_leaves_no_manifest_in_repository():
    assert not list(ROOT.glob("**/partition_manifest.json"))
    assert not (ROOT / "V8_UNIVERSE_MANIFEST.json").exists()


# ---------------------------------------------------------------------------
# Synthetic result shape (in-process)
# ---------------------------------------------------------------------------


REQUIRED_RESULT_FIELDS = {
    "status", "mode", "source_reproduction_status", "v4_raw_sha_equality_required",
    "block_sizes", "t1_role", "t2_role", "t3_role", "t3_price_acquisition_authorized",
    "manifest_sha256_verified", "write_once_enforced",
    "raw_hash_mismatch_does_not_block", "t0_mismatch_blocks_before_allocation",
    "network_requests", "real_partition_created", "real_source_fetch_performed",
}


def test_synthetic_result_has_required_fields(synthetic_result):
    assert REQUIRED_RESULT_FIELDS <= set(synthetic_result)


def test_synthetic_result_values(synthetic_result):
    assert synthetic_result["status"] == "PASS"
    assert synthetic_result["mode"] == "STATIC_SYNTHETIC_ONLY"
    assert synthetic_result["source_reproduction_status"] == "PASS"
    assert synthetic_result["v4_raw_sha_equality_required"] is False
    assert synthetic_result["t1_role"] == "VALIDATION"
    assert synthetic_result["t2_role"] == "SEALED_HOLDOUT"
    assert synthetic_result["t3_role"] == "SEALED_RESERVE"
    assert synthetic_result["t3_price_acquisition_authorized"] is False
    assert synthetic_result["manifest_sha256_verified"] is True
    assert synthetic_result["write_once_enforced"] is True
    assert synthetic_result["raw_hash_mismatch_does_not_block"] is True
    assert synthetic_result["t0_mismatch_blocks_before_allocation"] is True
    assert synthetic_result["network_requests"] == 0
    assert synthetic_result["real_partition_created"] is False
    assert synthetic_result["real_source_fetch_performed"] is False


def test_synthetic_result_block_sizes_equal_for_t0_t1_t2_t3(synthetic_result):
    sizes = synthetic_result["block_sizes"]
    assert sizes["T0"] == sizes["T1"] == sizes["T2"] == sizes["T3"] == cli.SYNTHETIC_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Production path (in-process, always with an injected fake opener --
# no test in this file ever calls the real urllib.request.urlopen default)
# ---------------------------------------------------------------------------


class FakeProductionResponse:
    def __init__(self, payload: bytes, url: str) -> None:
        self.payload = payload
        self.url = url
        self.status = 200

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        pass


class FakeJpxOpener:
    """Deterministic fake JPX page+xls opener; performs no network I/O."""

    def __init__(
        self,
        xls_bytes: bytes,
        *,
        data_link: str = "/files/data_j.xls",
        page_final_url: str | None = None,
        xls_final_url: str | None = None,
    ) -> None:
        self.xls_bytes = xls_bytes
        self.data_link = data_link
        self.page_final_url = page_final_url or cli.JPX_PAGE
        self.xls_final_url = xls_final_url
        self.calls: list[str] = []

    def __call__(self, request_obj):
        self.calls.append(request_obj.full_url)
        if request_obj.full_url == cli.JPX_PAGE:
            page_html = f'<a href="{self.data_link}">data_j.xls</a>'.encode("utf-8")
            return FakeProductionResponse(page_html, self.page_final_url)
        return FakeProductionResponse(self.xls_bytes, self.xls_final_url or request_obj.full_url)


def _ordered_production_codes(total: int, *, start: int = 1000, pool: int = 4000) -> list[str]:
    import hashlib

    candidates = [str(start + i) for i in range(pool)]
    return sorted(candidates, key=lambda code: hashlib.sha256(code.encode("utf-8")).hexdigest())[:total]


@pytest.fixture(scope="module")
def production_fixture_data():
    """Full production scale (300 T0 + 900+ fresh), matching the frozen,
    non-overridable BLOCK_SIZE run_production_partition_build actually uses
    -- module-scoped since the code-ordering computation is the same for
    every test in this file and is pure/side-effect-free."""
    import hashlib

    import pandas as pd

    all_codes = _ordered_production_codes(partition.BLOCK_SIZE * 4 + 10)
    t0_codes = all_codes[:partition.BLOCK_SIZE]
    fresh_codes = all_codes[partition.BLOCK_SIZE:]

    t0_rows = [{"code": c, "market": "プライム（内国株式）", "industry": "IND"} for c in t0_codes]
    csv_bytes = partition.build_universe_csv_bytes(t0_rows)
    xls_bytes = b"FAKE_PRODUCTION_XLS_BYTES"

    manifest_payload = {
        "source_host": "www.jpx.co.jp",
        "source_page": cli.JPX_PAGE,
        "raw_file_sha256": hashlib.sha256(xls_bytes).hexdigest(),
        "universe_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "ticker_list_sha256": hashlib.sha256(("\n".join(t0_codes) + "\n").encode()).hexdigest(),
        "selection_rule": "fixture",
        "selected_count": partition.BLOCK_SIZE,
        "eligible_current_only": len(all_codes),
    }

    frame = pd.DataFrame(
        [{"コード": c, "銘柄名": "SYN", "市場・区分": "プライム（内国株式）", "33業種区分": "IND"} for c in t0_codes]
        + [{"コード": c, "銘柄名": "SYN", "市場・区分": "スタンダード（内国株式）", "33業種区分": "IND"} for c in fresh_codes]
    )

    return {
        "csv_bytes": csv_bytes,
        "xls_bytes": xls_bytes,
        "manifest_payload": manifest_payload,
        "frame": frame,
    }


@pytest.fixture()
def production_fixture(tmp_path, production_fixture_data):
    """Per-test file paths for the module-scoped fixture data above."""
    import json as json_module

    universe_csv_path = tmp_path / "V4_UNIVERSE.csv"
    universe_csv_path.write_bytes(production_fixture_data["csv_bytes"])

    manifest_path = tmp_path / "V4_UNIVERSE_MANIFEST.json"
    manifest_path.write_bytes(
        json_module.dumps(production_fixture_data["manifest_payload"], ensure_ascii=False).encode("utf-8")
    )

    return {
        "manifest_path": manifest_path,
        "universe_csv_path": universe_csv_path,
        "xls_bytes": production_fixture_data["xls_bytes"],
        "frame": production_fixture_data["frame"],
    }


def _private_test_seam_kwargs(fixture, output_path, opener):
    """PRIVATE TEST SEAM ONLY -- NOT PRODUCTION PUBLIC BOUNDARY."""
    return dict(
        output_path=output_path,
        opener=opener,
        parse_source_table=lambda _raw: fixture["frame"],
        v4_manifest_path=fixture["manifest_path"],
        v4_universe_csv_path=fixture["universe_csv_path"],
        clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
        repository_root=ROOT,
        git_commit_resolver=cli.resolve_verified_production_git_commit,
    )


def _private_source_preflight_seam_kwargs(fixture, opener):
    """PRIVATE TEST SEAM ONLY -- source/T0 path, never a public override."""
    return dict(
        opener=opener,
        parse_source_table=lambda _raw: fixture["frame"],
        v4_manifest_path=fixture["manifest_path"],
        v4_universe_csv_path=fixture["universe_csv_path"],
        clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
        repository_root=ROOT,
        git_commit_resolver=cli.resolve_verified_production_git_commit,
    )


def test_source_preflight_public_runner_signature_has_no_inputs():
    signature = inspect.signature(cli.run_production_source_preflight)
    assert tuple(signature.parameters) == ()


@pytest.mark.parametrize("override_name, override_value", (
    ("opener", lambda _request: None),
    ("parse_source_table", lambda _raw: None),
    ("v4_manifest_path", Path("fake-manifest.json")),
    ("v4_universe_csv_path", Path("fake-universe.csv")),
    ("clock", lambda: datetime(2026, 8, 9, tzinfo=timezone.utc)),
    ("repository_root", ROOT),
    ("git_commit_resolver", lambda _root: "a" * 40),
    ("source_url", "https://evil.example/source.xls"),
    ("raw_source_bytes", b"fake"),
))
def test_source_preflight_public_runner_rejects_dependency_overrides(override_name, override_value):
    with pytest.raises(TypeError):
        cli.run_production_source_preflight(**{override_name: override_value})


def test_source_preflight_public_runner_wires_canonical_dependencies(monkeypatch):
    captured = {}

    def fake_private_helper(**kwargs):
        captured.update(kwargs)
        return {"source_reproduction_status": "PASS", "t0_reproduction_status": "PASS"}

    monkeypatch.setattr(cli, "_run_production_source_preflight_with_dependencies", fake_private_helper)
    result = cli.run_production_source_preflight()

    assert result["source_reproduction_status"] == "PASS"
    assert captured == {
        "opener": cli._default_trusted_jpx_opener,
        "parse_source_table": cli.default_parse_source_table,
        "v4_manifest_path": cli.V4_MANIFEST_PATH,
        "v4_universe_csv_path": cli.V4_UNIVERSE_CSV_PATH,
        "clock": cli._utc_clock,
        "repository_root": cli.ROOT,
        "git_commit_resolver": cli.resolve_verified_production_git_commit,
    }


def test_source_preflight_valid_source_t0_passes_without_partition_artifact(
    tmp_path, production_fixture, monkeypatch
):
    allocation_calls = []
    write_calls = []

    def forbidden_allocate(*args, **kwargs):
        allocation_calls.append((args, kwargs))
        raise AssertionError("allocate_fresh_blocks reached")

    def forbidden_write(*args, **kwargs):
        write_calls.append((args, kwargs))
        raise AssertionError("write_partition_manifest_once reached")

    monkeypatch.setattr(partition, "allocate_fresh_blocks", forbidden_allocate)
    monkeypatch.setattr(cli, "write_partition_manifest_once", forbidden_write)
    opener = FakeJpxOpener(production_fixture["xls_bytes"])

    result = cli._run_production_source_preflight_with_dependencies(
        **_private_source_preflight_seam_kwargs(production_fixture, opener)
    )

    assert result["source_reproduction_status"] == "PASS"
    assert result["t0_reproduction_status"] == "PASS"
    assert result["source_raw_byte_count"] == len(production_fixture["xls_bytes"])
    assert result["eligible_ticker_count"] == len(production_fixture["frame"])
    assert result["t0_ticker_list_sha256"]
    assert set(result).isdisjoint({
        "t1_ticker_list_sha256", "t2_ticker_list_sha256", "t3_ticker_list_sha256",
        "t_spare_ticker_list_sha256", "block_assignments",
    })
    assert opener.calls == [cli.JPX_PAGE, "https://www.jpx.co.jp/files/data_j.xls"]
    assert allocation_calls == []
    assert write_calls == []
    assert not (tmp_path / "partition_manifest.json").exists()


def test_source_preflight_raw_sha_mismatch_does_not_block_when_t0_reproduces(
    tmp_path, production_fixture
):
    """V8_HISTORICAL_RESEARCH_DESIGN.md §16: raw bytes differing from V4's
    2026-08-03 reference must not block on their own -- parsing proceeds and
    the preflight PASSes as long as T0 still reproduces."""
    parse_calls = []

    def parser(_raw):
        parse_calls.append(True)
        return production_fixture["frame"]

    kwargs = _private_source_preflight_seam_kwargs(
        production_fixture, FakeJpxOpener(b"CURRENT_SNAPSHOT_DIFFERENT_FROM_V4_REFERENCE")
    )
    kwargs["parse_source_table"] = parser
    result = cli._run_production_source_preflight_with_dependencies(**kwargs)

    assert result["source_reproduction_status"] == "PASS"
    assert result["t0_reproduction_status"] == "PASS"
    assert result["v4_raw_sha_equality_required"] is False
    assert result["source_raw_sha256"] != result["v4_source_raw_sha256_reference"]
    assert parse_calls == [True]
    assert not (tmp_path / "partition_manifest.json").exists()


def test_source_preflight_wrong_t0_blocks_without_partition_artifact(production_fixture, tmp_path):
    tampered_frame = production_fixture["frame"].copy()
    market_column = tampered_frame.columns[2]
    tampered_frame.loc[0, market_column] = tampered_frame.loc[len(tampered_frame) - 1, market_column]
    fixture = dict(production_fixture, frame=tampered_frame)
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_source_preflight_with_dependencies(
            **_private_source_preflight_seam_kwargs(
                fixture, FakeJpxOpener(production_fixture["xls_bytes"])
            )
        )
    assert excinfo.value.reason == "V8_T0_REPRODUCTION_MISMATCH"
    assert not (tmp_path / "partition_manifest.json").exists()


@pytest.mark.parametrize("git_reason", (
    "PRODUCTION_GIT_WORKTREE_DIRTY",
    "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
))
def test_source_preflight_git_failure_blocks_before_jpx(git_reason, production_fixture):
    def blocked(_):
        raise partition.V8PartitionBlocked(git_reason)

    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    kwargs = _private_source_preflight_seam_kwargs(production_fixture, opener)
    kwargs["git_commit_resolver"] = blocked
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_source_preflight_with_dependencies(**kwargs)
    assert excinfo.value.reason == git_reason
    assert opener.calls == []


def test_production_public_runner_signature_exposes_output_path_only():
    signature = inspect.signature(cli.run_production_partition_build)
    assert tuple(signature.parameters) == ("output_path",)
    for forbidden in (
        "opener",
        "parse_source_table",
        "v4_manifest_path",
        "v4_universe_csv_path",
        "clock",
        "repository_root",
        "git_commit_resolver",
        "raw_source_bytes",
        "source_url",
        "source_host",
        "ticker_frame",
    ):
        assert forbidden not in signature.parameters


@pytest.mark.parametrize("override_name, override_value", (
    ("opener", lambda _request: None),
    ("parse_source_table", lambda _raw: None),
    ("v4_manifest_path", Path("fake-manifest.json")),
    ("v4_universe_csv_path", Path("fake-universe.csv")),
    ("clock", lambda: datetime(2026, 8, 9, tzinfo=timezone.utc)),
))
def test_production_public_runner_rejects_dependency_overrides(tmp_path, override_name, override_value):
    with pytest.raises(TypeError):
        cli.run_production_partition_build(
            output_path=tmp_path / "partition_manifest.json",
            **{override_name: override_value},
        )


def test_production_public_runner_wires_canonical_dependencies(monkeypatch, tmp_path):
    """Public wiring is inspected without making a real network request."""
    captured = {}

    def fake_private_helper(**kwargs):
        captured.update(kwargs)
        return {"manifest": {}, "written_path": kwargs["output_path"]}

    monkeypatch.setattr(cli, "_run_production_partition_build_with_dependencies", fake_private_helper)
    output_path = tmp_path / "partition_manifest.json"
    result = cli.run_production_partition_build(output_path=output_path)

    assert result["written_path"] == output_path
    assert captured == {
        "output_path": output_path,
        "opener": cli._default_trusted_jpx_opener,
        "parse_source_table": cli.default_parse_source_table,
        "v4_manifest_path": cli.V4_MANIFEST_PATH,
        "v4_universe_csv_path": cli.V4_UNIVERSE_CSV_PATH,
        "clock": cli._utc_clock,
        "repository_root": cli.ROOT,
        "git_commit_resolver": cli.resolve_verified_production_git_commit,
    }


def test_private_test_seam_build_succeeds_with_valid_fixture(tmp_path, production_fixture):
    """PRIVATE TEST SEAM ONLY -- NOT PRODUCTION PUBLIC BOUNDARY."""
    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    output_path = tmp_path / "private-output" / "partition_manifest.json"
    result = cli._run_production_partition_build_with_dependencies(
        **_private_test_seam_kwargs(production_fixture, output_path, opener)
    )
    assert result["manifest"]["source_reproduction_status"] == "PASS"
    assert result["written_path"] == output_path
    assert opener.calls == [cli.JPX_PAGE, "https://www.jpx.co.jp/files/data_j.xls"]


def test_production_manifest_self_hash_verified_on_readback(tmp_path, production_fixture):
    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    output_path = tmp_path / "output" / "partition_manifest.json"
    result = cli._run_production_partition_build_with_dependencies(
        **_private_test_seam_kwargs(production_fixture, output_path, opener)
    )
    reread = partition.read_partition_manifest(output_path)
    assert reread == result["manifest"]


def test_production_raw_sha_mismatch_does_not_block_full_build_when_t0_reproduces(
    tmp_path, production_fixture
):
    """V8_HISTORICAL_RESEARCH_DESIGN.md §16: the full production build path
    must not require raw-hash equality against V4's 2026-08-03 reference --
    only exact T0 reproduction gates block allocation and publication."""
    opener = FakeJpxOpener(b"CURRENT_SNAPSHOT_DIFFERENT_FROM_V4_REFERENCE")
    output_path = tmp_path / "output" / "partition_manifest.json"
    result = cli._run_production_partition_build_with_dependencies(
        **_private_test_seam_kwargs(production_fixture, output_path, opener)
    )
    manifest = result["manifest"]
    assert manifest["source_reproduction_status"] == "PASS"
    assert manifest["t0_reproduction_status"] == "PASS"
    assert manifest["v4_raw_sha_equality_required"] is False
    assert manifest["source_raw_sha256"] != manifest["v4_source_raw_sha256_reference"]
    assert output_path.exists()


def test_production_t0_reproduction_mismatch_blocks_before_block_assignment(tmp_path, production_fixture):
    import pandas as pd

    # Raw bytes are irrelevant to this BLOCK under §16 -- what matters is
    # a frame whose T0 market string diverges from the committed
    # V4_UNIVERSE.csv, so the reconstructed T0 no longer byte-reproduces it.
    tampered_frame = production_fixture["frame"].copy()
    tampered_frame.loc[0, "市場・区分"] = "スタンダード（内国株式）"
    fixture = dict(production_fixture, frame=tampered_frame)

    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    output_path = tmp_path / "output" / "partition_manifest.json"
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(fixture, output_path, opener)
        )
    assert excinfo.value.reason == "V8_T0_REPRODUCTION_MISMATCH"
    assert not output_path.exists()


def test_production_relative_output_path_blocks(production_fixture):
    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(production_fixture, Path("relative/pm.json"), opener)
        )
    assert excinfo.value.reason == "OUTPUT_PATH_NOT_ABSOLUTE"
    assert opener.calls == []


def test_production_in_repository_output_path_blocks(production_fixture):
    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    inside_repo = ROOT / "tmp-v8-production-cli-test.json"
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(production_fixture, inside_repo, opener)
        )
    assert excinfo.value.reason == "OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY"
    assert not inside_repo.exists()
    assert opener.calls == []


def test_production_overwrite_of_existing_manifest_blocks(tmp_path, production_fixture):
    output_path = tmp_path / "output" / "partition_manifest.json"
    cli._run_production_partition_build_with_dependencies(
        **_private_test_seam_kwargs(
            production_fixture, output_path, FakeJpxOpener(production_fixture["xls_bytes"])
        )
    )
    second_opener = FakeJpxOpener(production_fixture["xls_bytes"])
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(production_fixture, output_path, second_opener)
        )
    assert excinfo.value.reason == "PARTITION_MANIFEST_ALREADY_EXISTS"
    assert second_opener.calls == []


@pytest.mark.parametrize("git_reason", (
    "PRODUCTION_GIT_WORKTREE_DIRTY",
    "PRODUCTION_GIT_HEAD_UNAVAILABLE",
    "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE",
    "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
))
def test_production_git_provenance_failure_blocks_before_jpx_network(tmp_path, production_fixture, monkeypatch, git_reason):
    def blocked(_):
        raise partition.V8PartitionBlocked(git_reason)

    monkeypatch.setattr(cli, "resolve_verified_production_git_commit", blocked)
    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(
                production_fixture, tmp_path / "output" / "partition_manifest.json", opener
            )
        )
    assert excinfo.value.reason == git_reason
    assert opener.calls == []


def test_production_data_link_not_found_blocks(tmp_path, production_fixture):
    class NoLinkOpener:
        calls: list[str] = []

        def __call__(self, request_obj):
            self.calls.append(request_obj.full_url)
            return FakeProductionResponse(b"<html>no xls link here</html>", cli.JPX_PAGE)

    opener = NoLinkOpener()
    output_path = tmp_path / "output" / "partition_manifest.json"
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(production_fixture, output_path, opener)
        )
    assert excinfo.value.reason == "V8_PARTITION_SOURCE_LINK_NOT_FOUND"
    assert opener.calls == [cli.JPX_PAGE]  # never attempted the (nonexistent) xls fetch


def test_production_resolved_source_host_must_be_jpx(tmp_path, production_fixture):
    opener = FakeJpxOpener(production_fixture["xls_bytes"], data_link="https://evil.example.com/data_j.xls")
    output_path = tmp_path / "output" / "partition_manifest.json"
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(production_fixture, output_path, opener)
        )
    assert excinfo.value.reason == "V8_PARTITION_SOURCE_HOST_INVALID"


def test_trusted_jpx_same_host_redirect_is_permitted():
    handler = cli.TrustedJpxRedirectHandler()
    request = urllib.request.Request(cli.JPX_PAGE)
    redirected = handler.redirect_request(
        request, None, 302, "Found", Message(), "https://www.jpx.co.jp/files/data_j.xls"
    )
    assert redirected is not None
    assert redirected.full_url == "https://www.jpx.co.jp/files/data_j.xls"


def test_trusted_jpx_explicit_standard_https_port_is_permitted():
    assert cli._require_trusted_jpx_url("https://www.jpx.co.jp:443/files/data_j.xls").startswith("https://")


@pytest.mark.parametrize("redirect_url", (
    "https://attacker.example/data_j.xls",
    "http://www.jpx.co.jp/files/data_j.xls",
    "https://www.jpx.co.jp:444/files/data_j.xls",
    "https://user@www.jpx.co.jp/files/data_j.xls",
    "https://user:pass@www.jpx.co.jp/files/data_j.xls",
))
def test_trusted_jpx_redirect_rejected_before_off_host_request(redirect_url):
    handler = cli.TrustedJpxRedirectHandler()
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        handler.redirect_request(
            urllib.request.Request(cli.JPX_PAGE), None, 302, "Found", Message(), redirect_url
        )
    assert excinfo.value.reason == "V8_PARTITION_SOURCE_HOST_INVALID"


def test_production_final_response_host_must_be_jpx(tmp_path, production_fixture):
    opener = FakeJpxOpener(
        production_fixture["xls_bytes"], xls_final_url="https://attacker.example/data_j.xls"
    )
    output_path = tmp_path / "output" / "partition_manifest.json"
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        cli._run_production_partition_build_with_dependencies(
            **_private_test_seam_kwargs(production_fixture, output_path, opener)
        )
    assert excinfo.value.reason == "V8_PARTITION_SOURCE_HOST_INVALID"
    assert opener.calls == [cli.JPX_PAGE, "https://www.jpx.co.jp/files/data_j.xls"]


def test_production_network_call_count_is_exactly_two_per_attempt(tmp_path, production_fixture):
    """One request for the listing page, one for the resolved data_j.xls --
    never more, never fewer, and never any other host."""
    opener = FakeJpxOpener(production_fixture["xls_bytes"])
    output_path = tmp_path / "output" / "partition_manifest.json"
    cli._run_production_partition_build_with_dependencies(
        **_private_test_seam_kwargs(production_fixture, output_path, opener)
    )
    assert len(opener.calls) == 2
    assert opener.calls[0] == cli.JPX_PAGE
    assert opener.calls[1].startswith("https://www.jpx.co.jp/")


def test_production_confirmation_constant_is_used_by_main(tmp_path, production_fixture, monkeypatch):
    """Exercises main() end-to-end with a correct confirmation, entirely
    through injected fakes -- monkeypatches run_production_partition_build
    itself so this test never touches the real network path at all."""
    captured = {}

    def fake_build(*, output_path):
        captured["output_path"] = output_path
        return {
            "manifest": {
                "source_reproduction_status": "PASS",
                "block_sizes": {"T0": 5, "T1": 5, "T2": 5, "T3": 5, "T_spare": 0},
                "manifest_sha256": "f" * 64,
            },
            "written_path": output_path,
        }

    monkeypatch.setattr(cli, "run_production_partition_build", fake_build)
    output_path = tmp_path / "manifest.json"
    exit_code = cli.main([
        "--production-build-manifest", "--output-path", str(output_path),
        "--confirmation", cli.PRODUCTION_CONFIRMATION,
    ])
    assert exit_code == 0
    assert captured["output_path"] == output_path


# ---------------------------------------------------------------------------
# Static safety: src/v8_partition.py and this CLI
# ---------------------------------------------------------------------------


def test_module_source_has_no_network_imports():
    text = Path(partition.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines), token


def test_cli_source_has_no_urlopen_call():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert calls == []


def test_module_never_touches_v7_files():
    text = Path(partition.__file__).read_text(encoding="utf-8")
    assert "v7_" not in text.lower()
    cli_text = SCRIPT.read_text(encoding="utf-8")
    assert "v7_" not in cli_text.lower()


def test_frozen_design_commit_matches_current_design():
    assert partition.DESIGN_COMMIT == "c414d3191cba356734d7ed08bdf1abc7d51fc384"


def test_frozen_block_size_and_p_hist_unchanged():
    assert partition.BLOCK_SIZE == 300
    assert partition.P_HIST_START == "2016-04-01"
    assert partition.P_HIST_END == "2025-12-31"


def test_source_snapshot_semantics_binding_matches_design_clarification():
    assert partition.SCHEMA_VERSION == "V8_PARTITION_MANIFEST_V3"
    assert partition.SOURCE_SNAPSHOT_SEMANTICS == "IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT"
    assert partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT == "266999a8e48c77905dd7c7312fd41c7f38241d78"
    assert partition.V4_RAW_SHA_EQUALITY_REQUIRED is False
