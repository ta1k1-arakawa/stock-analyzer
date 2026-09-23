from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path

import pytest

from scripts.v13_resolve_v8_t1_identity_state import (
    EXPECTED_MANIFEST_STATED_SHA256,
    EXPECTED_T1_COUNT,
    EXPECTED_T1_TICKER_LIST_SHA256,
    IdentityResolutionBlocked,
    _ExpectedBindings,
    _resolve_identity_state,
    _write_once,
    resolve_identity_state,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def test_dir():
    with tempfile.TemporaryDirectory(prefix="v13-t1-synthetic-", dir=str(_REPOSITORY_ROOT.parent)) as directory:
        yield Path(directory)


def _codes(start: int, count: int) -> list[str]:
    return [f"{value:04X}" for value in range(start, start + count)]


def _ticker_sha(members: list[str]) -> str:
    return hashlib.sha256(("\n".join(members) + "\n").encode("utf-8")).hexdigest()


def _fixture(
    *,
    t1: list[str] | None = None,
    t1_stated_sha: str | None = None,
    manifest_stated_sha: str | None = None,
    t2: list[str] | None = None,
    t3: list[str] | None = None,
    t_spare: list[str] | None = None,
) -> tuple[dict[str, object], _ExpectedBindings]:
    members = list(t1) if t1 is not None else _codes(300, 300)
    stated_t1_sha = t1_stated_sha or _ticker_sha(members)
    manifest_sha = manifest_stated_sha or hashlib.sha256(b"synthetic stated manifest binding").hexdigest()
    assignments: dict[str, list[str]] = {
        "T0": _codes(0, 300),
        "T1": members,
        "T2": list(t2) if t2 is not None else _codes(600, 300),
        "T3": list(t3) if t3 is not None else _codes(900, 300),
        "T_spare": list(t_spare) if t_spare is not None else ["1200"],
    }
    manifest: dict[str, object] = {
        "schema_version": "V8_PARTITION_MANIFEST_V3",
        "study_name": "synthetic-study",
        "design_commit": "1" * 40,
        "source_snapshot_semantics": "SYNTHETIC",
        "source_snapshot_clarification_commit": "2" * 40,
        "partition_implementation_git_commit": "3" * 40,
        "created_utc": "2026-01-01T00:00:00Z",
        "source_url": "https://synthetic.invalid/",
        "source_host": "synthetic.invalid",
        "source_acquisition_utc": "2026-01-01T00:00:00Z",
        "source_raw_sha256": "4" * 64,
        "source_raw_byte_count": 1,
        "v4_source_raw_sha256_reference": "5" * 64,
        "v4_raw_sha_equality_required": False,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": 2000,
        "eligible_ticker_list_sha256": "6" * 64,
        "selection_rule": "synthetic selection rule",
        "deterministic_ordering_rule": "synthetic ordering rule",
        "t0_ticker_list_sha256": "7" * 64,
        "t1_ticker_list_sha256": stated_t1_sha,
        "t2_ticker_list_sha256": "8" * 64,
        "t3_ticker_list_sha256": "9" * 64,
        "t_spare_ticker_list_sha256": "a" * 64,
        "legacy_exclude_list": ["1570"],
        "legacy_exclude_list_sha256": "b" * 64,
        "block_sizes": {"T0": 300, "T1": len(members), "T2": len(assignments["T2"]), "T3": len(assignments["T3"]), "T_spare": len(assignments["T_spare"])},
        "block_assignments": assignments,
        "p_hist_start": "2016-04-01",
        "p_hist_end": "2025-12-31",
        "t1_role": "VALIDATION",
        "t2_role": "SEALED_HOLDOUT",
        "t3_role": "SEALED_RESERVE",
        "t3_price_acquisition_authorized": False,
        "manifest_sha256": manifest_sha,
    }
    bindings = _ExpectedBindings(
        manifest_stated_sha256=manifest_sha,
        t1_ticker_list_sha256=stated_t1_sha,
        t1_count=EXPECTED_T1_COUNT,
    )
    return manifest, bindings


def _write_source(tmp_path: Path, manifest: dict[str, object], name: str = "synthetic.json") -> Path:
    source = tmp_path / name
    source.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return source


def _resolve_synthetic(
    source: Path,
    output: Path,
    repo: Path,
    bindings: _ExpectedBindings,
    **kwargs,
) -> dict[str, object]:
    return _resolve_identity_state(source, output, repo, bindings=bindings, **kwargs)


def test_valid_fixture_extracts_only_t1_and_writes_private_state_equivalent_output(
    test_dir: Path, tmp_path: Path
) -> None:
    manifest, bindings = _fixture()
    source = _write_source(tmp_path, manifest)
    output = test_dir / "identity-state.json"

    result = _resolve_synthetic(source, output, _REPOSITORY_ROOT, bindings)

    expected_members = manifest["block_assignments"]["T1"]  # type: ignore[index]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert result["t1_membership"] == expected_members
    assert saved["t1_membership"] == expected_members
    assert set(saved) == {
        "schema",
        "source_partition_manifest_stated_sha256",
        "t1_ticker_list_sha256",
        "t1_count",
        "known_definitely_acquired_prefix_count",
        "t1_membership",
    }
    assert saved["schema"] == "V13_V8_T1_IDENTITY_STATE_V1"
    assert saved["t1_count"] == 300
    assert saved["known_definitely_acquired_prefix_count"] == 297
    assert saved["source_partition_manifest_stated_sha256"] == bindings.manifest_stated_sha256
    assert saved["t1_ticker_list_sha256"] == bindings.t1_ticker_list_sha256


def test_first_byte_callback_precedes_scanner_and_is_invoked_once(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    events: list[str] = []
    original_open = Path.open

    class ObservedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            events.append("open")
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            events.append("close")
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            events.append("first-byte" if size == 1 else "remainder")
            return self.stream.read(size)

    def observed_open(path, *args, **kwargs):
        return ObservedStream(original_open(path, *args, **kwargs))

    def on_first_byte():
        assert events == ["open", "first-byte"]
        events.append("callback")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", observed_open)
        result = _resolve_synthetic(
            source, test_dir / "state.json", test_dir.parent / "repository-root", bindings,
            on_first_byte=on_first_byte,
        )
    assert events == ["open", "first-byte", "callback", "remainder", "close"]
    assert result["t1_count"] == 300


def test_production_entrypoint_uses_fixed_public_bindings() -> None:
    module_path = Path(__file__).parents[1] / "scripts" / "v13_resolve_v8_t1_identity_state.py"
    source = module_path.read_text(encoding="utf-8")
    assert EXPECTED_MANIFEST_STATED_SHA256 == "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
    assert EXPECTED_T1_TICKER_LIST_SHA256 == "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d"
    assert EXPECTED_T1_COUNT == 300
    assert "bindings=_PRODUCTION_BINDINGS" in source
    assert resolve_identity_state.__name__ == "resolve_identity_state"


def test_wrong_stated_manifest_hash_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    manifest["manifest_sha256"] = "f" * 64
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="MANIFEST_STATED_SHA_MISMATCH"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_wrong_stated_t1_hash_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    manifest["t1_ticker_list_sha256"] = "e" * 64
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="T1_STATED_SHA_MISMATCH"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_actual_t1_hash_mismatch_blocks(test_dir: Path) -> None:
    members = _codes(300, 300)
    manifest, _ = _fixture(t1=members, t1_stated_sha="d" * 64)
    bindings = _ExpectedBindings(
        manifest_stated_sha256=manifest["manifest_sha256"],  # type: ignore[arg-type]
        t1_ticker_list_sha256="d" * 64,
        t1_count=300,
    )
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="T1_HASH_MISMATCH"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_wrong_t1_count_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture(t1=_codes(300, 299))
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="T1_COUNT_MISMATCH"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_duplicate_t1_code_blocks(test_dir: Path) -> None:
    members = _codes(300, 300)
    members[-1] = members[0]
    manifest, bindings = _fixture(t1=members)
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="T1_DUPLICATE_CODE"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_invalid_t1_code_format_blocks(test_dir: Path) -> None:
    members = _codes(300, 300)
    members[10] = "BAD!"
    manifest, bindings = _fixture(t1=members)
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="T1_CODE_FORMAT_INVALID"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_duplicate_relevant_root_key_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    raw = source.read_bytes()
    duplicate = b',"manifest_sha256":"' + bindings.manifest_stated_sha256.encode("ascii") + b'"}'
    source.write_bytes(raw[:-1] + duplicate)
    with pytest.raises(IdentityResolutionBlocked, match="MANIFEST_INVALID_OR_AMBIGUOUS"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_second_t1_assignment_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    raw = source.read_bytes()
    needle = b'"T1":['
    assert raw.count(needle) == 1
    source.write_bytes(raw.replace(needle, b'"T1":[],"T1":[', 1))
    with pytest.raises(IdentityResolutionBlocked, match="MANIFEST_INVALID_OR_AMBIGUOUS"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_duplicate_block_assignments_object_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    raw = source.read_bytes()
    duplicate = b',"block_assignments":{}}'
    source.write_bytes(raw[:-1] + duplicate)
    with pytest.raises(IdentityResolutionBlocked, match="MANIFEST_INVALID_OR_AMBIGUOUS"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_malformed_manifest_structure_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    manifest["block_assignments"]["T1"] = {"nested": "not an array"}  # type: ignore[index]
    source = _write_source(test_dir, manifest)
    with pytest.raises(IdentityResolutionBlocked, match="MANIFEST_INVALID_OR_AMBIGUOUS"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_truncated_json_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(IdentityResolutionBlocked, match="MANIFEST_INVALID_OR_AMBIGUOUS"):
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)


def test_existing_output_blocks_without_overwrite(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    output = test_dir / "outside.json"
    output.write_bytes(b"preserve-existing")
    with pytest.raises(IdentityResolutionBlocked, match="OUTPUT_ALREADY_EXISTS"):
        _resolve_synthetic(source, output, test_dir.parent / "repository-root", bindings)
    assert output.read_bytes() == b"preserve-existing"


def test_write_once_publication_is_no_overwrite_and_preserves_existing_bytes(test_dir: Path) -> None:
    output = test_dir / "already-published.json"
    output.write_bytes(b"preserve-existing")
    with pytest.raises(IdentityResolutionBlocked, match="OUTPUT_ALREADY_EXISTS"):
        _write_once(output, b"replacement")
    assert output.read_bytes() == b"preserve-existing"


def test_repository_inside_output_blocks(test_dir: Path) -> None:
    manifest, bindings = _fixture()
    source = _write_source(test_dir, manifest)
    repo = test_dir / "repo"
    repo.mkdir()
    with pytest.raises(IdentityResolutionBlocked, match="OUTPUT_PATH_INSIDE_REPOSITORY"):
        _resolve_synthetic(source, repo / "state.json", repo, bindings)
    assert not (repo / "state.json").exists()


def test_non_absolute_output_blocks_before_source_read(test_dir: Path) -> None:
    with pytest.raises(IdentityResolutionBlocked, match="OUTPUT_PATH_INVALID"):
        _resolve_synthetic(
            test_dir / "missing-source.json",
            Path("relative-state.json"),
            test_dir.parent / "repository-root",
            _fixture()[1],
        )


def test_t2_t3_and_spare_sentinels_never_leak(test_dir: Path) -> None:
    sentinel = "FORBIDDEN_PRIVATE_SENTINEL"
    manifest, bindings = _fixture(
        t2=[sentinel] + _codes(601, 299),
        t3=[sentinel] + _codes(901, 299),
        t_spare=[sentinel],
    )
    source = _write_source(test_dir, manifest)
    output = test_dir / "outside.json"
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = _resolve_synthetic(source, output, test_dir.parent / "repository-root", bindings)
    persisted = output.read_text(encoding="utf-8")
    assert sentinel not in repr(result)
    assert sentinel not in persisted
    assert sentinel not in stdout.getvalue()
    assert sentinel not in stderr.getvalue()
    assert set(result) == {
        "schema",
        "source_partition_manifest_stated_sha256",
        "t1_ticker_list_sha256",
        "t1_count",
        "known_definitely_acquired_prefix_count",
        "t1_membership",
    }


def test_identity_is_absent_from_error_and_stdout(test_dir: Path) -> None:
    identity = "LEAK"
    members = _codes(300, 300)
    members[1] = identity
    members[2] = identity
    manifest, bindings = _fixture(t1=members)
    source = _write_source(test_dir, manifest)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), pytest.raises(IdentityResolutionBlocked) as caught:
        _resolve_synthetic(source, test_dir / "outside.json", test_dir.parent / "repository-root", bindings)
    assert identity not in str(caught.value)
    assert identity not in stdout.getvalue()


def test_production_code_has_no_network_or_full_object_json_loader() -> None:
    module_path = Path(__file__).parents[1] / "scripts" / "v13_resolve_v8_t1_identity_state.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_modules = {"http", "httpx", "requests", "socket", "subprocess", "urllib", "urllib3"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_modules
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (node.func.attr == "loads" and isinstance(node.func.value, ast.Name) and node.func.value.id == "json")
