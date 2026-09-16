"""Production entrypoint for the read-only V10D diagnostic.

The entrypoint exposes only the frozen module CLI.  It contains no source
transport, cache discovery, alternative-root fallback, or diagnostic bypass.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v10d_t0_data_incompatibility_diagnostic import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
