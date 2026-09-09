from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from backtest import sort_skipped_orders
from src.benchmark import FixedOHLCVLoader, sha256_file
from src.reproducibility import CONFIG_HASH_METHOD, config_hash


def _run(command: list[str], cwd: Path) -> None:
    subprocess.check_call(command, cwd=cwd)


def test_autocrlf_fresh_clone_preserves_all_fixed_csv_hashes(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    (source / "data" / "benchmark" / "ohlcv").mkdir(parents=True)
    shutil.copy2(project / ".gitattributes", source / ".gitattributes")
    shutil.copy2(
        project / "data" / "benchmark" / "manifest.json",
        source / "data" / "benchmark" / "manifest.json",
    )
    manifest = json.loads(
        (project / "data" / "benchmark" / "manifest.json").read_text(encoding="utf-8")
    )
    for code in manifest["stock_codes"]:
        shutil.copy2(
            project / "data" / "benchmark" / "ohlcv" / f"{code}.csv",
            source / "data" / "benchmark" / "ohlcv" / f"{code}.csv",
        )
    _run(["git", "init", "-q"], source)
    _run(["git", "add", "."], source)
    _run([
        "git", "-c", "user.name=Tests", "-c", "user.email=tests@example.invalid",
        "commit", "-q", "-m", "snapshot",
    ], source)
    _run(["git", "-c", "core.autocrlf=true", "clone", "-q", str(source), str(clone)], tmp_path)
    _run(["git", "config", "core.autocrlf", "true"], clone)
    for code, metadata in manifest["files"].items():
        assert sha256_file(clone / "data" / "benchmark" / "ohlcv" / f"{code}.csv") == metadata["sha256"]
    loader = FixedOHLCVLoader(clone / "data" / "benchmark")
    assert sorted(loader.manifest["stock_codes"]) == sorted(manifest["stock_codes"])
    assert len(loader.manifest["stock_codes"]) == 8


def test_config_hash_normalizes_bom_crlf_and_cr(tmp_path: Path) -> None:
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    cr = tmp_path / "cr.yaml"
    bom = tmp_path / "bom.yaml"
    lf.write_bytes("alpha: 1\nbeta: 2\n".encode("utf-8"))
    crlf.write_bytes("alpha: 1\r\nbeta: 2\r\n".encode("utf-8"))
    cr.write_bytes("alpha: 1\rbeta: 2\r".encode("utf-8"))
    bom.write_bytes(b"\xef\xbb\xbf" + lf.read_bytes())
    assert len({config_hash(path) for path in (lf, crlf, cr, bom)}) == 1
    assert CONFIG_HASH_METHOD == "utf8-normalized-lf-v1"


def test_config_hash_changes_with_content(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("value: 1\n", encoding="utf-8")
    second.write_text("value: 2\n", encoding="utf-8")
    assert config_hash(first) != config_hash(second)


def test_skipped_orders_use_defined_stable_composite_order() -> None:
    frame = pd.DataFrame([
        {"signal_date": "2025-01-02", "planned_entry_date": "2025-01-03", "prob": .8, "code": "B", "status": "Z"},
        {"signal_date": "2025-01-01", "planned_entry_date": "2025-01-03", "prob": .7, "code": "B", "status": "Z"},
        {"signal_date": "2025-01-01", "planned_entry_date": "2025-01-03", "prob": .9, "code": "B", "status": "Z"},
        {"signal_date": "2025-01-01", "planned_entry_date": "2025-01-03", "prob": .9, "code": "A", "status": "Z"},
        {"signal_date": "2025-01-01", "planned_entry_date": "2025-01-03", "prob": .9, "code": "A", "status": "A"},
    ])
    expected = [
        ("2025-01-01", "2025-01-03", .9, "A", "A"),
        ("2025-01-01", "2025-01-03", .9, "A", "Z"),
        ("2025-01-01", "2025-01-03", .9, "B", "Z"),
        ("2025-01-01", "2025-01-03", .7, "B", "Z"),
        ("2025-01-02", "2025-01-03", .8, "B", "Z"),
    ]
    first = sort_skipped_orders(frame)
    second = sort_skipped_orders(frame)
    assert list(first.itertuples(index=False, name=None)) == expected
    pd.testing.assert_frame_equal(first, second)
