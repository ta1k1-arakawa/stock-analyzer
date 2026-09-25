"""Future guarded public acquisition entrypoint; never run for Issue #84.

The reviewed PowerShell wrapper supplies all paths after a later human gate.
No source is inferred from the machine. Complete responses are locked before
HTML, workbook, or chart parsing. No retries occur after a complete response.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import v13_public_data_lock as pipeline


class _OfficialLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if re.search(r"(?:^|/)data_j\.xls(?:\?|$)", href, re.I):
                self.links.append(href)


def _fetch(url: str, destination: Path) -> pipeline.RawLock:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.jpx.co.jp", "query1.finance.yahoo.com"}:
        raise ValueError("PROVIDER_MISMATCH")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if urllib.parse.urlparse(response.geturl()).hostname != parsed.hostname:
            raise ValueError("REDIRECT_HOST_MISMATCH")
        raw = response.read()
    return pipeline.lock_payload(raw, destination)


def _safe_json(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-state", type=Path, required=True)
    parser.add_argument("--v4-csv", type=Path, required=True)
    parser.add_argument("--calendar-lock", type=Path, required=True)
    parser.add_argument("--calendar-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-sha", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.implementation_sha):
        raise ValueError("IMPLEMENTATION_SHA_REQUIRED")
    if args.output.exists():
        raise ValueError("OUTPUT_COLLISION")
    # Explicit private input only; no path discovery. All preflight belongs to
    # the reviewed wrapper and must precede this invocation.
    t1 = pipeline.validate_t1_state(json.loads(args.t1_state.read_bytes()))
    v4 = pipeline.read_v4_codes(args.v4_csv.read_bytes())
    excluded, exclusion_manifest = pipeline.build_exclusions(v4, t1)
    calendar_raw = args.calendar_lock.read_bytes()
    if pipeline.digest(calendar_raw) != args.calendar_sha256:
        raise ValueError("CALENDAR_LOCK_MISMATCH")
    calendar, calendar_manifest = pipeline.parse_calendar(pipeline.RawLock.from_bytes(calendar_raw))
    if calendar[0].year != 2015 or calendar[-1].year != 2025:
        raise ValueError("CALENDAR_SPAN_INCOMPLETE")
    args.output.mkdir(mode=0o700)
    page = _fetch(pipeline.JPX_PAGE, args.output / "jpx-page.raw")
    links = _OfficialLinks()
    links.feed(page.raw.decode("utf-8", errors="replace"))
    candidates = sorted(set(urllib.parse.urljoin(pipeline.JPX_PAGE, href) for href in links.links))
    if len(candidates) != 1 or urllib.parse.urlparse(candidates[0]).hostname != "www.jpx.co.jp":
        raise ValueError("JPX_SOURCE_AMBIGUOUS")
    jpx = _fetch(candidates[0], args.output / "jpx-listed-issues.raw")
    eligible = pipeline.parse_jpx(jpx)
    selected, universe_manifest = pipeline.select_universe(
        eligible, excluded, jpx, args.implementation_sha, datetime.now().astimezone().isoformat())
    universe_manifest["jpx_payload_url_sha256"] = pipeline.digest(candidates[0].encode("utf-8"))
    universe_manifest["jpx_page_raw_sha256"] = page.sha256
    # Identity-bearing output is local and must never be committed or printed.
    _safe_json(args.output / "selected.private.json", {"selected": selected})
    price_manifests = []
    tokyo = ZoneInfo("Asia/Tokyo")
    period1 = int(datetime(2015, 1, 1, tzinfo=tokyo).timestamp())
    period2 = int(datetime(2026, 1, 1, tzinfo=tokyo).timestamp())
    for code in selected:
        params = urllib.parse.urlencode({"period1": period1, "period2": period2,
                                         "interval": "1d", "events": "history",
                                         "includeAdjustedClose": "true"})
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T?{params}"
        lock = _fetch(url, args.output / f"yahoo-{code}.raw")
        rows, manifest = pipeline.parse_yahoo(lock, code)
        _safe_json(args.output / f"price-{code}.private.json",
                   {day.isoformat(): row for day, row in rows.items()})
        price_manifests.append(manifest)
    _safe_json(args.output / "safe-manifest.json", {
        "jpx_page": page.manifest(), "jpx": jpx.manifest(), "exclusions": exclusion_manifest,
        "universe": universe_manifest, "calendar": calendar_manifest,
        "yahoo_payload_count": len(price_manifests),
        "yahoo_raw_set_sha256": pipeline.digest(json.dumps(price_manifests, sort_keys=True).encode()),
        "model_fits": 0, "backtests": 0, "outcome_calculations": 0})
    print("PUBLIC_DATALOCK_PASS=true")
    print("YAHOO_PAYLOAD_COUNT=500")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("PUBLIC_DATALOCK_PASS=false", file=sys.stderr)
        sys.exit(1)
