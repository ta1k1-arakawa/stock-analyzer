"""Guarded external-cwd entrypoint for V9_015 Stage-F child acquisition."""

from __future__ import annotations

from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.v9_015_source_b_stage_f_child_acquisition import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
