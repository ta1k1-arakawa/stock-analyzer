"""EXP-002 data acquisition: J-Quants v2 daily bars, financial summaries and TOPIX.

Universe (docs/experiments/EXP-002_HANDOFF.md): only stocks already excluded from V13,
i.e. the public V4_UNIVERSE.csv (300 stocks, v13 branch) plus the legacy 8 minus the ETF 1570.
Every request is per code; date-wide requests (which would return every listed stock) are never made.

The API key is read from the environment variable JQUANTS_API_KEY and is never printed or written.
Raw responses are cached under <out_dir> (keep it out of git: J-Quants data may not be redistributed);
the manifest with record counts and SHA-256 hashes is written to <manifest>.

Usage:
  python experiments/exp002_fetch.py <V4_UNIVERSE.csv> <out_dir> <manifest.json>
  (V4_UNIVERSE.csv: git show origin/v13-conditional-cross-sectional-short-horizon:V4_UNIVERSE.csv)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.jquants.com/v2"
UNIVERSE_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"  # V4_UNIVERSE_MANIFEST.json
LEGACY = ["4188", "4689", "5020", "7211", "7267", "8306", "9432"]  # legacy 8 without the ETF 1570
PAUSE_SEC = 0.6
RETRIES = 4


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_codes(universe_csv: Path) -> list[str]:
    raw = universe_csv.read_bytes()
    if sha256_bytes(raw) != UNIVERSE_SHA256:
        raise SystemExit("V4_UNIVERSE.csv hash mismatch")
    with universe_csv.open(encoding="utf-8") as f:
        codes = [row["ticker"].strip().upper() for row in csv.DictReader(f)]
    return sorted(set(codes) | set(LEGACY))


def get_all(session: requests.Session, path: str, params: dict) -> tuple[list[dict], int]:
    """GET with pagination_key handling. Returns (records, request_count)."""
    records, n_req, params = [], 0, dict(params)
    while True:
        for attempt in range(RETRIES + 1):
            n_req += 1
            try:
                resp = session.get(f"{BASE}{path}", params=params, timeout=60)
            except requests.RequestException as e:
                err = type(e).__name__
            else:
                if resp.status_code == 200:
                    break
                err = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise SystemExit(f"{path} {params.get('code', '')}: {err}")
            if attempt == RETRIES:
                raise SystemExit(f"{path} {params.get('code', '')}: {err} after {RETRIES + 1} attempts")
            time.sleep(2 ** (attempt + 1))
        body = resp.json()
        records.extend(body.get("data", []))
        time.sleep(PAUSE_SEC)
        if not body.get("pagination_key"):
            return records, n_req
        params["pagination_key"] = body["pagination_key"]


def dump(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return sha256_bytes(raw)


def main(universe_csv: Path, out_dir: Path, manifest_path: Path) -> None:
    key = os.environ.get("JQUANTS_API_KEY")
    if not key:
        raise SystemExit("JQUANTS_API_KEY is not set")
    codes = load_codes(universe_csv)
    session = requests.Session()
    session.headers["x-api-key"] = key

    files, n_requests = {}, 0
    recs, n = get_all(session, "/indices/bars/daily/topix", {})
    n_requests += n
    files["topix.json"] = {"records": len(recs), "sha256": dump(out_dir / "topix.json", recs),
                           "first": recs[0]["Date"], "last": recs[-1]["Date"]}
    for i, code in enumerate(codes, 1):
        for kind, path, date_key in (("bars", "/equities/bars/daily", "Date"), ("fins", "/fins/summary", "DiscDate")):
            recs, n = get_all(session, path, {"code": code})
            n_requests += n
            rel = f"{kind}/{code}.json"
            entry = {"records": len(recs), "sha256": dump(out_dir / rel, recs)}
            if recs:
                dates = sorted(r[date_key] for r in recs)
                entry.update(first=dates[0], last=dates[-1])
            files[rel] = entry
        if i % 25 == 0:
            print(f"{i}/{len(codes)} codes", file=sys.stderr)

    combined = sha256_bytes("\n".join(f"{k} {v['sha256']}" for k, v in sorted(files.items())).encode())
    manifest = {
        "experiment": "EXP-002",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": BASE,
        "endpoints": ["/indices/bars/daily/topix", "/equities/bars/daily?code=", "/fins/summary?code="],
        "universe_csv_sha256": UNIVERSE_SHA256,
        "legacy_codes": LEGACY,
        "codes": len(codes),
        "http_requests": n_requests,
        "records_total": {k: sum(v["records"] for f, v in files.items() if f.startswith(k))
                          for k in ("bars/", "fins/", "topix")},
        "codes_without_bars": [f[5:-5] for f, v in files.items() if f.startswith("bars/") and v["records"] == 0],
        "codes_without_fins": [f[5:-5] for f, v in files.items() if f.startswith("fins/") and v["records"] == 0],
        "cache_file_format": "json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(',', ':')) UTF-8",
        "combined_sha256": combined,
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
