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
membership, NO prices, and NO outcomes. Every code below is drawn from the
`ZZ`-prefixed synthetic namespace (`ZZA1`, `ZZB2`, ...) paired with a
`SYNTHETIC_*` name -- visibly artificial from the code value itself, not
merely from the company name -- and none is a real JPX security code. The
rows exist only to exercise column detection and the eligible/excluded
branches of the production filter.

```text
finding_resolved = "REAL_EXECUTION_XLS_FIXTURE_MEDIUM_1_REAL_TICKER_COLLISION"
```

An earlier revision of this fixture used plain `9001`-`9007`-style numeric
placeholders, which collide with real JPX security codes (e.g. `9001` and
`9003` are real listed-instrument identities) and therefore did not satisfy
"no real ticker identities." Every code is now drawn from the `ZZ`-prefixed
namespace below instead; see `SYNTHETIC_TICKER_NAMESPACE_PREFIX` and
`_assert_synthetic_namespace`, which mechanically fail at import time if a
later edit reintroduces an ordinary numeric JPX-looking code.

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

# Every fixture code is drawn from this unmistakably artificial namespace --
# never an ordinary numeric JPX-looking code -- so no code here can collide
# with a real JPX security identity. Enforced mechanically below, at import
# time, by `_assert_synthetic_namespace`.
SYNTHETIC_TICKER_NAMESPACE_PREFIX = "ZZ"

# Wholly artificial rows. `expected_eligible` documents, per row, what the
# production filter (prime/standard AND domestic AND 4-char [0-9A-Z] code)
# must decide -- so the fixture proves both the accept and the reject
# branches, not merely that parsing returned something.
SYNTHETIC_ROWS: tuple[tuple[str, str, str, str, bool], ...] = (
    # code,     name,                     market,                       industry,           expected_eligible
    ("ZZA1", "SYNTHETIC_ALPHA", "プライム（内国株式）", "SYNTHETIC_SECTOR_A", True),
    ("ZZB2", "SYNTHETIC_BRAVO", "プライム（内国株式）", "SYNTHETIC_SECTOR_A", True),
    ("ZZC3", "SYNTHETIC_CHARLIE", "スタンダード（内国株式）", "SYNTHETIC_SECTOR_B", True),
    ("ZZD4", "SYNTHETIC_DELTA", "スタンダード（内国株式）", "SYNTHETIC_SECTOR_B", True),
    ("ZZE5", "SYNTHETIC_ECHO", "プライム（内国株式）", "SYNTHETIC_SECTOR_C", True),
    # Excluded: not prime/standard.
    ("ZZG6", "SYNTHETIC_FOXTROT_GROWTH", "グロース（内国株式）", "SYNTHETIC_SECTOR_C", False),
    # Excluded: prime/standard but not a domestic stock.
    ("ZZF7", "SYNTHETIC_GOLF_FOREIGN", "プライム（外国株式）", "SYNTHETIC_SECTOR_D", False),
    # Excluded: prime/standard domestic but not a 4-character ordinary code.
    ("ZZZZ8", "SYNTHETIC_HOTEL_LONGCODE", "プライム（内国株式）", "SYNTHETIC_SECTOR_D", False),
)


def _assert_synthetic_namespace(rows: tuple[tuple[str, str, str, str, bool], ...]) -> None:
    """Fail loudly (at import time) if any fixture code is not visibly
    artificial -- so a later edit cannot silently reintroduce an ordinary
    numeric JPX-looking code. Purely a string-prefix check against fixed
    repo content: offline, public-safe, deterministic, no JPX/network
    lookup of any kind.
    """
    for code, _name, _market, _industry, _expected in rows:
        if not code.startswith(SYNTHETIC_TICKER_NAMESPACE_PREFIX):
            raise AssertionError(
                "SYNTHETIC_FIXTURE_CODE_OUTSIDE_NAMESPACE: "
                f"{code!r} does not start with {SYNTHETIC_TICKER_NAMESPACE_PREFIX!r}"
            )


_assert_synthetic_namespace(SYNTHETIC_ROWS)

EXPECTED_ELIGIBLE_CODES: tuple[str, ...] = tuple(
    row[0] for row in SYNTHETIC_ROWS if row[4]
)
EXPECTED_TOTAL_ROW_COUNT = len(SYNTHETIC_ROWS)
EXPECTED_ELIGIBLE_ROW_COUNT = len(EXPECTED_ELIGIBLE_CODES)

# Recorded canonical identity of the committed fixture bytes (updated by
# this script's own maintainer whenever the fixture is regenerated -- see
# `--check`, which reports rather than silently trusts this value).
EXPECTED_FIXTURE_SHA256 = "ca47744896a286e1c56d4d0c09260775772c7df0c01b80d81b7e9a515e6d6aa7"


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
        print(f"expected_fixture_sha256={EXPECTED_FIXTURE_SHA256}")
        print(f"committed_matches_expected={str(committed_digest == EXPECTED_FIXTURE_SHA256).lower()}")
        print(f"rebuilt_workbook_sha256={rebuilt_digest}")
        print(f"rebuild_matched_committed_bytes={str(committed_digest == rebuilt_digest).lower()}")
        print("byte_determinism_claimed=false")
        print("canonical_identity=COMMITTED_FIXTURE_SHA256")
        print(f"synthetic_ticker_namespace_prefix={SYNTHETIC_TICKER_NAMESPACE_PREFIX}")
        print("synthetic_namespace_verified=true")  # _assert_synthetic_namespace already ran at import time
        return 0

    path, digest = write_fixture()
    print(f"fixture_written={path}")
    print(f"fixture_sha256={digest}")
    print(f"expected_fixture_sha256={EXPECTED_FIXTURE_SHA256}")
    print(f"matches_recorded_canonical_sha256={str(digest == EXPECTED_FIXTURE_SHA256).lower()}")
    print(f"total_row_count={EXPECTED_TOTAL_ROW_COUNT}")
    print(f"expected_eligible_row_count={EXPECTED_ELIGIBLE_ROW_COUNT}")
    print(f"synthetic_ticker_namespace_prefix={SYNTHETIC_TICKER_NAMESPACE_PREFIX}")
    print("contains_real_jpx_payload=false")
    print("contains_real_ticker_membership=false")
    print("contains_prices_or_outcomes=false")
    print("network_requests=0")
    print("private_reads=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
