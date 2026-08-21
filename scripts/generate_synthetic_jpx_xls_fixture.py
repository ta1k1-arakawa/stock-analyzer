"""Generate the committed synthetic legacy ".xls" readiness fixture.

Resolves the former
`CHATGPT_DECISION_REQUIRED: REAL_EXECUTION_XLS_SYNTHETIC_FIXTURE_STRATEGY`
per the binding GPT decision recorded in
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md` §6.

This script writes `tests/fixtures/synthetic_jpx_source_snapshot.xls`: a
completely artificial legacy OLE2/BIFF workbook used only to prove, before
any protected boundary, that the canonical `.venv` interpreter plus pandas
plus the `xlrd` engine can actually parse legacy `.xls` bytes through the
real production parsing path. It exists solely for pre-gate environment
readiness.

The fixture contains NO real JPX payload, NO real or private ticker
membership, NO prices, and NO outcomes. Every code below is an obviously
synthetic `9XXX`-style placeholder paired with a `SYNTHETIC_*` name; none
is asserted to be, or derived from, any real listed instrument. The rows
exist only to exercise column detection and the eligible/excluded
branches of the production filter.

```text
generator_dependency = "xlwt==1.3.0"
generator_dependency_is_production_dependency = false
```

`xlwt` is a fixture-generation/dev tool ONLY. It is deliberately NOT added
to `requirements-real-execution.txt`, because production parsing reads
`.xls` (via pandas + `xlrd`) and never writes it. Nothing on the real
execution path imports `xlwt`.

This script performs no network access and reads no private data. Run it
only to regenerate or inspect the fixture:

    python3 scripts/generate_synthetic_jpx_xls_fixture.py

Byte-for-byte determinism across runs is NOT claimed (see
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md` §6): `xlwt` embeds workbook
metadata whose stability across versions/platforms this repository has not
established. The committed, reviewed fixture bytes and their recorded
SHA-256 are therefore the canonical identity; this generator is
explanatory/reconstruction tooling only. `--check` re-derives a workbook
in memory and reports whether it happens to match the committed bytes,
without ever claiming determinism it has not proven.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "synthetic_jpx_source_snapshot.xls"

# Exact headings the production parser's column detection requires.
# src/v8_partition.py::parse_eligible_universe resolves, after whitespace
# removal: ("コード",), ("銘柄名",), ("市場", "区分") -- satisfied by
# "市場・商品区分" -- and optionally a heading containing "33業種区分".
COLUMN_HEADINGS = ("コード", "銘柄名", "市場・商品区分", "33業種区分")

# Wholly artificial rows. `expected_eligible` documents, per row, what the
# production filter (prime/standard AND domestic AND 4-char [0-9A-Z] code)
# must decide -- so the fixture proves both the accept and the reject
# branches, not merely that parsing returned something.
SYNTHETIC_ROWS: tuple[tuple[str, str, str, str, bool], ...] = (
    # code,   name,                  market,                       industry,           expected_eligible
    ("9001", "SYNTHETIC_ALPHA", "プライム（内国株式）", "SYNTHETIC_SECTOR_A", True),
    ("9002", "SYNTHETIC_BRAVO", "プライム（内国株式）", "SYNTHETIC_SECTOR_A", True),
    ("9003", "SYNTHETIC_CHARLIE", "スタンダード（内国株式）", "SYNTHETIC_SECTOR_B", True),
    ("9004", "SYNTHETIC_DELTA", "スタンダード（内国株式）", "SYNTHETIC_SECTOR_B", True),
    ("9005", "SYNTHETIC_ECHO", "プライム（内国株式）", "SYNTHETIC_SECTOR_C", True),
    # Excluded: not prime/standard.
    ("9006", "SYNTHETIC_FOXTROT_GROWTH", "グロース（内国株式）", "SYNTHETIC_SECTOR_C", False),
    # Excluded: prime/standard but not a domestic stock.
    ("9007", "SYNTHETIC_GOLF_FOREIGN", "プライム（外国株式）", "SYNTHETIC_SECTOR_D", False),
    # Excluded: prime/standard domestic but not a 4-character ordinary code.
    ("90080", "SYNTHETIC_HOTEL_LONGCODE", "プライム（内国株式）", "SYNTHETIC_SECTOR_D", False),
)

EXPECTED_ELIGIBLE_CODES: tuple[str, ...] = tuple(
    row[0] for row in SYNTHETIC_ROWS if row[4]
)
EXPECTED_TOTAL_ROW_COUNT = len(SYNTHETIC_ROWS)
EXPECTED_ELIGIBLE_ROW_COUNT = len(EXPECTED_ELIGIBLE_CODES)


def build_workbook_bytes() -> bytes:
    """Build the synthetic legacy .xls workbook in memory via xlwt."""
    try:
        import xlwt
    except ImportError as error:  # pragma: no cover -- generator-only tool
        raise SystemExit(
            "xlwt is required to generate this fixture. Install the generator-only "
            "dependency with: python3 -m pip install xlwt==1.3.0 "
            "(do NOT add it to requirements-real-execution.txt)."
        ) from error

    workbook = xlwt.Workbook(encoding="utf-8")
    sheet = workbook.add_sheet("SYNTHETIC")
    for column_index, heading in enumerate(COLUMN_HEADINGS):
        sheet.write(0, column_index, heading)
    for row_index, (code, name, market, industry, _expected) in enumerate(SYNTHETIC_ROWS, start=1):
        # Codes are written as text, mirroring how the production parser
        # coerces them (`astype(str).str.strip()`), so the fixture never
        # depends on float-formatting behavior.
        sheet.write(row_index, 0, code)
        sheet.write(row_index, 1, name)
        sheet.write(row_index, 2, market)
        sheet.write(row_index, 3, industry)

    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def write_fixture() -> tuple[Path, str]:
    payload = build_workbook_bytes()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(payload)
    return FIXTURE_PATH, hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write. Report the committed fixture's SHA-256, and whether a freshly "
            "built workbook happens to match those exact bytes. A mismatch is NOT an error: "
            "byte-for-byte determinism is not claimed for xlwt output."
        ),
    )
    args = parser.parse_args()

    if args.check:
        if not FIXTURE_PATH.exists():
            print(f"committed_fixture_present=false path={FIXTURE_PATH}")
            return 1
        committed = FIXTURE_PATH.read_bytes()
        committed_digest = hashlib.sha256(committed).hexdigest()
        rebuilt = build_workbook_bytes()
        rebuilt_digest = hashlib.sha256(rebuilt).hexdigest()
        print(f"committed_fixture_present=true path={FIXTURE_PATH}")
        print(f"committed_fixture_sha256={committed_digest}")
        print(f"rebuilt_workbook_sha256={rebuilt_digest}")
        print(f"rebuild_matched_committed_bytes={str(committed_digest == rebuilt_digest).lower()}")
        print("byte_determinism_claimed=false")
        print("canonical_identity=COMMITTED_FIXTURE_SHA256")
        return 0

    path, digest = write_fixture()
    print(f"fixture_written={path}")
    print(f"fixture_sha256={digest}")
    print(f"total_row_count={EXPECTED_TOTAL_ROW_COUNT}")
    print(f"expected_eligible_row_count={EXPECTED_ELIGIBLE_ROW_COUNT}")
    print("contains_real_jpx_payload=false")
    print("contains_real_ticker_membership=false")
    print("contains_prices_or_outcomes=false")
    print("network_requests=0")
    print("private_reads=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
