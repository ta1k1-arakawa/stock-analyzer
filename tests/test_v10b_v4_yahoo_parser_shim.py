from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.v10b_training_cache_reacquisition as v10b
import src.v10b_v4_yahoo_parser_shim as shim


ROOT = Path(__file__).resolve().parents[1]


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _assignment_value(path: Path, name: str) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
    raise AssertionError(f"missing assignment: {name}")


def _payload(*, timestamps=None, open_values=None, high_values=None, low_values=None, close_values=None, adjusted=None, volume=None, splits=None):
    timestamps = [1546300800, 1546387200] if timestamps is None else timestamps
    open_values = [100.0] * len(timestamps) if open_values is None else open_values
    high_values = [101.0] * len(timestamps) if high_values is None else high_values
    low_values = [99.0] * len(timestamps) if low_values is None else low_values
    close_values = [100.5] * len(timestamps) if close_values is None else close_values
    adjusted = [100.5] * len(timestamps) if adjusted is None else adjusted
    volume = [1000] * len(timestamps) if volume is None else volume
    result = {
        "timestamp": timestamps,
        "indicators": {
            "quote": [{"open": open_values, "high": high_values, "low": low_values, "close": close_values, "volume": volume}],
            "adjclose": [{"adjclose": adjusted}],
        },
        "events": {"splits": splits or {}},
    }
    return {"chart": {"error": None, "result": [result]}}


def test_shim_function_asts_are_identical_to_reviewed_legacy_functions():
    pairs = (
        ("validate_ohlcv", ROOT / "src" / "free_prototype.py"),
        ("validate_v4_ohlcv", ROOT / "src" / "v4_meta_label_mvp.py"),
        ("parse_v4_yahoo_chart", ROOT / "src" / "v4_meta_label_mvp.py"),
    )
    shim_path = ROOT / "src" / "v10b_v4_yahoo_parser_shim.py"
    for name, legacy_path in pairs:
        assert ast.dump(_function_node(shim_path, name), include_attributes=False) == ast.dump(
            _function_node(legacy_path, name), include_attributes=False
        )


def test_shim_constants_match_reviewed_legacy_source_and_values():
    shim_path = ROOT / "src" / "v10b_v4_yahoo_parser_shim.py"
    free_path = ROOT / "src" / "free_prototype.py"
    mvp_path = ROOT / "src" / "v4_meta_label_mvp.py"
    assert ast.literal_eval(_assignment_value(free_path, "DATE_TO")) == "2025-03-31"
    assert ast.dump(_assignment_value(shim_path, "DATE_TO"), include_attributes=False) == ast.dump(
        _assignment_value(free_path, "DATE_TO"), include_attributes=False
    )
    for name, expected in (("PRICE_FROM", "2015-01-01"), ("PRICE_TO", "2019-12-31")):
        assert ast.dump(_assignment_value(shim_path, name), include_attributes=False) == ast.dump(
            _assignment_value(mvp_path, name), include_attributes=False
        )
        assert getattr(shim, name) == pd.Timestamp(expected)


def test_shim_import_closure_isolated_subprocess():
    code = """
import sys
import src.v10b_training_cache_reacquisition as v10b
parser = v10b._resolve_inherited_parser()
assert parser.__module__ == 'src.v10b_v4_yahoo_parser_shim'
assert callable(parser)
for forbidden in ('requests', 'lightgbm', 'sklearn', 'scipy', 'src.v4_meta_label_formal', 'src.v4_meta_label_mvp', 'src.free_prototype'):
    assert forbidden not in sys.modules, forbidden
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_shim_imports_only_allowed_runtime_dependencies():
    tree = ast.parse((ROOT / "src" / "v10b_v4_yahoo_parser_shim.py").read_text(encoding="utf-8"))
    allowed = {"__future__", "typing", "numpy", "pandas"}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    assert set(imports) <= allowed
    source = (ROOT / "src" / "v10b_v4_yahoo_parser_shim.py").read_text(encoding="utf-8")
    for forbidden in ("requests", "lightgbm", "sklearn", "scipy", "src."):
        assert forbidden not in source


def test_production_parser_binding_uses_shim():
    parser = v10b._resolve_inherited_parser()
    assert parser is shim.parse_v4_yahoo_chart


def test_valid_synthetic_yahoo_chart_and_split_are_parsed():
    frame, splits = shim.parse_v4_yahoo_chart(
        _payload(splits={"0": {"date": 1546387200, "numerator": 2, "denominator": 1}})
    )
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert len(frame) == 2
    assert splits == {pd.Timestamp("2019-01-02")}


def _payload_without_adjclose():
    payload = _payload()
    payload["chart"]["result"][0]["indicators"]["adjclose"] = None
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        {"chart": {"error": {"code": "Not Found"}, "result": None}},
        {"chart": {"error": None, "result": []}},
        _payload(timestamps=[]),
        _payload_without_adjclose(),
    ],
)
def test_chart_error_missing_result_empty_timestamps_and_missing_adjclose_fail(payload):
    with pytest.raises(ValueError):
        shim.parse_v4_yahoo_chart(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_values": [102.0, 99.0]},
        {"open_values": [0.0, 100.0]},
        {"open_values": [-1.0, 100.0]},
        {"volume": [-1, 1000]},
    ],
)
def test_invalid_ohlcv_relationship_zero_negative_and_volume_fail(kwargs):
    with pytest.raises(ValueError):
        shim.parse_v4_yahoo_chart(_payload(**kwargs))


@pytest.mark.parametrize(
    "timestamps",
    [
        [1546300800, 1546300800],
        [1546387200, 1546300800],
    ],
)
def test_duplicate_or_unordered_dates_fail(timestamps):
    with pytest.raises(ValueError):
        shim.parse_v4_yahoo_chart(_payload(timestamps=timestamps))


@pytest.mark.parametrize("timestamp", [1419984000, 1577836801])
def test_v4_prohibited_date_boundaries_fail(timestamp):
    with pytest.raises(ValueError, match="PROHIBITED_V4_PRICE_DATE"):
        shim.parse_v4_yahoo_chart(_payload(timestamps=[timestamp]))
