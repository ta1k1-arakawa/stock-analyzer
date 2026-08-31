"""Command-line boundary for the future V9_010 JPX Stage-A run.

The command requires point-of-use implementation provenance and fresh human
confirmation.  It is intentionally not imported or invoked by tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v9_010_jpx_calendar_stage_a_acquisition import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
