"""Immutable OHLCV benchmark snapshot loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
MAX_BENCHMARK_DATE = pd.Timestamp("2026-05-20")


class BenchmarkValidationError(RuntimeError):
    """Raised when an immutable benchmark snapshot is incomplete or corrupt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_hash(files: dict[str, dict[str, Any]]) -> str:
    canonical = "\n".join(
        f"{code}:{files[code]['sha256']}" for code in sorted(files)
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class FixedOHLCVLoader:
    """Read only a validated local snapshot; this class has no network fallback."""

    def __init__(self, benchmark_dir: str | Path = "data/benchmark") -> None:
        self.root = Path(benchmark_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise BenchmarkValidationError(f"manifest missing: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkValidationError(f"invalid manifest: {manifest_path}") from exc
        self._frames: dict[str, pd.DataFrame] = {}
        self._validate_all()

    def _validate_all(self) -> None:
        files = self.manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise BenchmarkValidationError("manifest.files must be a non-empty object")
        if self.manifest.get("columns") != REQUIRED_COLUMNS:
            raise BenchmarkValidationError("manifest columns do not match required schema")
        if pd.Timestamp(self.manifest.get("date_to")) > MAX_BENCHMARK_DATE:
            raise BenchmarkValidationError("manifest includes dates after 2026-05-20")
        for code, metadata in files.items():
            path = self.root / "ohlcv" / f"{code}.csv"
            if not path.is_file():
                raise BenchmarkValidationError(f"CSV missing for {code}: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != metadata.get("sha256"):
                raise BenchmarkValidationError(f"SHA-256 mismatch for {code}")
            try:
                raw = pd.read_csv(path, dtype={"Date": str})
            except Exception as exc:
                raise BenchmarkValidationError(f"CSV unreadable for {code}") from exc
            missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
            if missing:
                raise BenchmarkValidationError(f"columns missing for {code}: {missing}")
            if list(raw.columns) != REQUIRED_COLUMNS:
                raise BenchmarkValidationError(f"unexpected columns/order for {code}")
            try:
                dates = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
            except Exception as exc:
                raise BenchmarkValidationError(f"invalid Date values for {code}") from exc
            if dates.duplicated().any():
                raise BenchmarkValidationError(f"duplicate dates for {code}")
            if not dates.is_monotonic_increasing:
                raise BenchmarkValidationError(f"dates not ascending for {code}")
            if dates.dt.tz is not None:
                raise BenchmarkValidationError(f"timezone-aware dates for {code}")
            if not dates.empty and dates.max() > MAX_BENCHMARK_DATE:
                raise BenchmarkValidationError(f"rows after 2026-05-20 for {code}")
            if len(raw) != int(metadata.get("rows", -1)):
                raise BenchmarkValidationError(f"row count mismatch for {code}")
            first = dates.iloc[0].strftime("%Y-%m-%d") if len(dates) else None
            last = dates.iloc[-1].strftime("%Y-%m-%d") if len(dates) else None
            if first != metadata.get("first_date") or last != metadata.get("last_date"):
                raise BenchmarkValidationError(f"date range mismatch for {code}")
            raw["Date"] = dates
            self._frames[str(code)] = raw.set_index("Date")
        if snapshot_hash(files) != self.manifest.get("snapshot_hash"):
            raise BenchmarkValidationError("snapshot hash mismatch")

    def get_daily_stock_prices(
        self, stock_code: str, date_from_str: str | None = None,
        date_to_str: str | None = None,
    ) -> pd.DataFrame:
        code = str(stock_code).removesuffix(".T")
        if code not in self._frames:
            raise BenchmarkValidationError(f"stock is not in snapshot: {code}")
        requested_from = pd.Timestamp(date_from_str or self.manifest["date_from"])
        requested_to = pd.Timestamp(date_to_str or self.manifest["date_to"])
        available_from = pd.Timestamp(self.manifest["date_from"])
        available_to = pd.Timestamp(self.manifest["date_to"])
        if requested_from < available_from or requested_to > available_to:
            raise BenchmarkValidationError(
                f"requested period outside snapshot: {requested_from.date()}..{requested_to.date()}"
            )
        frame = self._frames[code]
        return frame[(frame.index >= requested_from) & (frame.index <= requested_to)].copy()


def validate_snapshot(benchmark_dir: str | Path = "data/benchmark") -> dict[str, Any]:
    """Validate every file and return the manifest when successful."""
    return FixedOHLCVLoader(benchmark_dir).manifest
