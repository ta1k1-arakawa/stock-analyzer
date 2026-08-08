"""Cross-platform provenance helpers for deterministic evaluator artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


CONFIG_HASH_METHOD = "utf8-normalized-lf-v1"


def normalized_utf8_bytes(path: str | Path) -> bytes:
    """Remove an optional BOM and normalize only line endings to LF."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def config_hash(path: str | Path = "config.yaml") -> str:
    return hashlib.sha256(normalized_utf8_bytes(path)).hexdigest()
