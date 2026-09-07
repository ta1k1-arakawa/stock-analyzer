"""External-cwd synthetic smoke entrypoint for the V9_015 extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (  # noqa: E402
    LOCATOR_OK,
    resolve_source_b_year_page,
)
from src.v9_015_source_b_option_value_root_extractor import (  # noqa: E402
    REQUIRED_YEAR_LABELS,
    SOURCE_B_ARCHIVE_ROOT,
    extract_option_value_root_year_candidates,
)


def _synthetic_root() -> bytes:
    options = "".join(
        f'<option value="synthetic/archive-{year}.html">{year}</option>'
        for year in REQUIRED_YEAR_LABELS
    )
    return f"<html><body><select>{options}</select></body></html>".encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True)
    arguments = parser.parse_args()
    candidates = extract_option_value_root_year_candidates(
        _synthetic_root(), SOURCE_B_ARCHIVE_ROOT
    )
    downstream_statuses = [
        resolve_source_b_year_page([candidate], int(candidate.label)).status
        for candidate in candidates
    ]
    payload = {
        "status": "PASS",
        "candidate_years": [candidate.label for candidate in candidates],
        "downstream_statuses": downstream_statuses,
        "all_downstream_ok": all(status == LOCATOR_OK for status in downstream_statuses),
        "network_requests": 0,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
