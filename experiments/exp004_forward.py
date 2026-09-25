"""EXP-004: forward comparison of three 20-stock rotation portfolios.

  R  rule      : EXP-003 composite (B/P rank + forecast-ROE rank). Real money follows this one.
  M  LightGBM  : same rotation, stocks ranked by a LightGBM model frozen before the start.
  L  Claude    : R's ranking, but skips stocks whose latest earnings report (決算短信) Claude rates
                 as a likely value trap (trap_risk >= 4).

Pre-registration: docs/experiments/EXP-004_PREREGISTRATION.md
Commands (run in this order every trading day after the data is published, ~20:00 JST):
  python experiments/exp004_forward.py update-data   # J-Quants, per code only
  python experiments/exp004_forward.py score         # Claude reads new 決算短信 (needs ANTHROPIC_API_KEY)
  python experiments/exp004_forward.py step          # executes today's paper orders, decides tomorrow's
  python experiments/exp004_forward.py notify        # sends R's order sheet to Slack (SLACK_WEBHOOK_URL)
One-time, before the start:
  python experiments/exp004_forward.py train         # fits and freezes the LightGBM model
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp003_rotation as E3  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "jquants_exp002"
FWD = ROOT / "forward"
MODEL = ROOT / "experiments" / "models" / "exp004_lgbm.txt"
MODEL_META = ROOT / "experiments" / "models" / "exp004_lgbm_meta.json"
JQ = "https://api.jquants.com/v2"
START = pd.Timestamp("2026-09-28")  # first trading day (all tranches buy at the open)
BUDGET = 400_000
COST = 0.0003
N_TRANCHES, PER_TRANCHE, HOLD_DAYS = 10, 2, 20
SPARE = 2  # extra names per tranche in case an open does not trade
TRAIN_END = pd.Timestamp("2026-08-26")  # last signal date whose 20-day target is fully known
TRAP_MAX = 3  # L keeps stocks with trap_risk <= 3; unscored stocks count as 3
CLAUDE_MODEL = "claude-opus-5"
PDF_PAGES = 6  # the summary and the management discussion come first in a 決算短信
MAX_CALLS_PER_RUN = 80  # cost cap per daily run
PORTFOLIOS = ("R", "M", "L")


# ----------------------------------------------------------------------------
# data update (per code only; never date-wide requests)
# ----------------------------------------------------------------------------

def jq_get(session: requests.Session, path: str, params: dict) -> list[dict]:
    out, params = [], dict(params)
    while True:
        for attempt in range(5):
            try:
                r = session.get(f"{JQ}{path}", params=params, timeout=60)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                break
            if r is not None and r.status_code not in (429, 500, 502, 503, 504):
                raise SystemExit(f"{path} {params.get('code', '')}: HTTP {r.status_code}")
            time.sleep(2 ** (attempt + 1))
        else:
            raise SystemExit(f"{path} {params.get('code', '')}: failed after retries")
        body = r.json()
        out += body.get("data", [])
        time.sleep(0.3)
        if not body.get("pagination_key"):
            return out
        params["pagination_key"] = body["pagination_key"]


def dump(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def update_data() -> None:
    key = os.environ.get("JQUANTS_API_KEY")
    if not key:
        raise SystemExit("JQUANTS_API_KEY is not set")
    s = requests.Session()
    s.headers["x-api-key"] = key
    codes = (FWD / "universe.txt").read_text().split()
    dump(CACHE / "topix.json", jq_get(s, "/indices/bars/daily/topix", {}))
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    cal = jq_get(s, "/markets/calendar", {"from": str((today - pd.Timedelta(days=10)).date()),
                                         "to": str((today + pd.Timedelta(days=40)).date())})
    dump(CACHE / "calendar.json", cal)
    names_file = FWD / "names.json"
    if not names_file.exists():
        names = {}
        for code in codes:
            rec = jq_get(s, "/equities/master", {"code": code})
            names[code] = rec[-1]["CoName"] if rec else ""
        names_file.write_text(json.dumps(names, ensure_ascii=False, indent=0, sort_keys=True) + "\n")
    for i, code in enumerate(codes, 1):
        f = CACHE / "bars" / f"{code}.json"
        old = json.loads(f.read_text()) if f.exists() else []
        params = {"code": code}
        if old:
            params["from"] = max(r["Date"] for r in old)
        new = jq_get(s, "/equities/bars/daily", params)
        merged = {r["Date"]: r for r in old}
        merged.update({r["Date"]: r for r in new})
        dump(f, [merged[d] for d in sorted(merged)])
        dump(CACHE / "fins" / f"{code}.json", jq_get(s, "/fins/summary", {"code": code}))
        if i % 50 == 0:
            print(f"{i}/{len(codes)} codes", file=sys.stderr)
    print(json.dumps({"updated_codes": len(codes), "topix_last": json.loads((CACHE / "topix.json").read_text())[-1]["Date"]}))


def next_trading_day(after: pd.Timestamp) -> pd.Timestamp:
    cal = json.loads((CACHE / "calendar.json").read_text())
    days = sorted(pd.Timestamp(r["Date"]) for r in cal if r["HolDiv"] in ("1", "2"))
    later = [d for d in days if d > after]
    if not later:
        raise SystemExit("calendar does not cover the next trading day; run update-data")
    return later[0]


# ----------------------------------------------------------------------------
# LightGBM features (cross-sectional percentile ranks among eligible stocks)
# ----------------------------------------------------------------------------

FEATURES = ["bp", "roe", "mom20", "mom60", "mom250", "vol60", "size", "liq", "roe_chg60"]


def features(panel, fins, dates) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sig = E3.signals(panel, fins, dates)
    ok = sig["composite"].notna()
    eq, npf = E3.fundamentals(fins, dates)
    cols = panel["C"].columns
    eq, npf = eq.reindex(columns=cols), npf.reindex(columns=cols)
    c = panel["C"].ffill()
    ret = c.pct_change(fill_method=None)
    raw = {
        "bp": sig["value_only"],
        "roe": sig["profit_only"],
        "mom20": c / c.shift(20) - 1,
        "mom60": c / c.shift(60) - 1,
        "mom250": c.shift(20) / c.shift(250) - 1,
        "vol60": ret.rolling(60, min_periods=40).std(),
        "size": np.log(panel["MktCap"]),
        "liq": np.log(panel["Va"].fillna(0.0).rolling(20, min_periods=20).mean() + 1.0),
        "roe_chg60": (npf - npf.shift(60)) / eq,
    }
    feats = {k: v.where(ok).rank(axis=1, pct=True) for k, v in raw.items()}
    return feats, ok


def train() -> None:
    import lightgbm as lgb

    topix, panel, fins = E3.load(CACHE)
    dates = topix.index
    feats, ok = features(panel, fins, dates)
    o = panel["O"]
    fwd = o.shift(-(HOLD_DAYS + 1)) / o.shift(-1) - 1  # buy next open, sell 20 trading days later at the open
    target = fwd.where(ok).rank(axis=1, pct=True)
    first = int(dates.searchsorted(pd.Timestamp("2017-01-04")))
    last = int(dates.searchsorted(TRAIN_END, side="right")) - 1
    rows = []
    for t in range(first, last + 1, 5):
        d = dates[t]
        x = pd.DataFrame({k: v.loc[d] for k, v in feats.items()})
        x["y"] = target.loc[d]
        rows.append(x.dropna(subset=["y"]))
    data = pd.concat(rows)
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                              deterministic=True, force_row_wise=True, verbose=-1)
    model.fit(data[FEATURES], data["y"])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL))
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signal_dates": [str(dates[first].date()), str(dates[last].date())], "every_n_days": 5,
        "rows": int(len(data)), "features": FEATURES,
        "target": "percentile rank of open(t+21)/open(t+1)-1 among eligible stocks",
        "params": model.get_params(),
        "feature_importance_gain": dict(zip(FEATURES, [round(float(g), 1) for g in
                                                      model.booster_.feature_importance("gain")])),
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    }
    MODEL_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "params"}, ensure_ascii=False, indent=1))


def model_scores(feats: dict[str, pd.DataFrame], ok: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    import lightgbm as lgb

    meta = json.loads(MODEL_META.read_text())
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != meta["model_sha256"]:
        raise SystemExit("frozen LightGBM model hash mismatch")
    booster = lgb.Booster(model_file=str(MODEL))
    x = pd.DataFrame({k: feats[k].loc[d] for k in FEATURES})[ok.loc[d]]
    return pd.Series(booster.predict(x[FEATURES]), index=x.index)


# ----------------------------------------------------------------------------
# Claude reads each new 決算短信 (earnings report) once, when it is published
# ----------------------------------------------------------------------------

SCORE_FILE = FWD / "llm_scores.jsonl"
SYSTEM = (
    "あなたは日本株の決算短信を読む株式アナリストです。"
    "この会社は株価が割安（PBRが低い）で、会社予想ベースのROEも高い銘柄の候補です。"
    "決算短信を読み、今後3〜6か月でこの会社の稼ぐ力が落ちる、または会社予想が下方修正される可能性"
    "（いわゆる割安のワナ）がどれくらいあるかを1〜5で評価してください。"
    "5=非常に高い（一時的な利益で予想がかさ上げされている、受注や需要が急に弱っている、"
    "継続企業の前提に注記がある、など）、3=どちらとも言えない、1=非常に低い（需要が堅調で予想も保守的）。"
    "書かれていることだけを根拠にし、株価の動きは考えないでください。"
)
SCHEMA = {
    "type": "object",
    "properties": {"trap_risk": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                   "reason": {"type": "string"}},
    "required": ["trap_risk", "reason"],
    "additionalProperties": False,
}


def tdnet_pdf(disc_no: str, disc_date: str) -> bytes | None:
    """Find the report PDF on TDnet (kept there for about 30 days after the disclosure)."""
    base = "https://www.release.tdnet.info/inbs/"
    for page in range(1, 30):
        url = f"{base}I_list_{page:03d}_{disc_date.replace('-', '')}.html"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        m = re.search(r'href="(\d*' + re.escape(disc_no) + r'\d*\.pdf)"', r.text)
        if m:
            pdf = requests.get(base + m.group(1), timeout=60)
            return pdf.content if pdf.status_code == 200 else None
        time.sleep(0.5)
    return None


def first_pages(pdf: bytes, n: int) -> bytes:
    from pypdf import PdfReader, PdfWriter

    reader, writer = PdfReader(io.BytesIO(pdf)), PdfWriter()
    for page in reader.pages[:n]:
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def ask_claude(client, pdf: bytes) -> dict:
    import anthropic

    try:
        resp = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                 "data": base64.standard_b64encode(pdf).decode()}},
                {"type": "text", "text": "この決算短信を評価してください。"},
            ]}],
        )
    except anthropic.RateLimitError:
        return {"status": "rate_limited"}
    except anthropic.APIStatusError as e:
        return {"status": f"api_error_{e.status_code}"}
    except anthropic.APIConnectionError:
        return {"status": "connection_error"}
    if resp.stop_reason == "refusal":
        return {"status": "refusal"}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "bad_json"}
    return {"status": "ok", "trap_risk": int(out["trap_risk"]), "reason": out["reason"], "model": resp.model}


def score() -> None:
    """Score financial statements disclosed in the last 20 days that are not scored yet."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(json.dumps({"skipped": "ANTHROPIC_API_KEY is not set"}))
        return
    import anthropic

    client = anthropic.Anthropic()
    done = set()
    if SCORE_FILE.exists():
        done = {json.loads(line)["disc_no"] for line in SCORE_FILE.read_text().splitlines() if line.strip()}
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    counts, calls = {}, 0
    for f in sorted((CACHE / "fins").glob("*.json")):
        bars = json.loads((CACHE / "bars" / f.name).read_text())[-E3.LIQ_DAYS:]
        if sum(float(b["Va"] or 0) for b in bars) / E3.LIQ_DAYS < E3.LIQ_MIN:
            continue  # only stocks that can enter the portfolio (same liquidity rule)
        for r in json.loads(f.read_text()):
            if calls >= MAX_CALLS_PER_RUN:
                break
            if "FinancialStatements" not in r["DocType"] or r["DiscNo"] in done:
                continue
            if pd.Timestamp(r["DiscDate"]) < max(START - pd.Timedelta(days=20), today - pd.Timedelta(days=20)):
                continue
            pdf = tdnet_pdf(r["DiscNo"], r["DiscDate"])
            calls += pdf is not None
            res = {"status": "pdf_not_found"} if pdf is None else ask_claude(client, first_pages(pdf, PDF_PAGES))
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if res["status"] in ("rate_limited", "connection_error") or res["status"].startswith("api_error_5"):
                continue  # retry on the next run
            rec = {"disc_no": r["DiscNo"], "code": f.stem, "disc_date": r["DiscDate"], "doc_type": r["DocType"],
                   "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **res}
            with SCORE_FILE.open("a") as out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"scored": counts}, ensure_ascii=False))


def trap_risk(asof: pd.Timestamp) -> dict[str, int]:
    """Latest trap_risk per code among reports disclosed on or before `asof` (scored ones only)."""
    latest = {}
    if SCORE_FILE.exists():
        for line in SCORE_FILE.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and pd.Timestamp(r["disc_date"]) <= asof:
                if r["code"] not in latest or r["disc_date"] >= latest[r["code"]][0]:
                    latest[r["code"]] = (r["disc_date"], r["trap_risk"])
    return {c: v[1] for c, v in latest.items()}


# ----------------------------------------------------------------------------
# paper portfolios
# ----------------------------------------------------------------------------

def load_state() -> dict:
    p = FWD / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {name: {"tranches": [{"cash": BUDGET / N_TRANCHES, "holdings": {}} for _ in range(N_TRANCHES)],
                   "pending": None} for name in PORTFOLIOS} | {"last_step": None}


def due_tranches(day: pd.Timestamp, trading_days: list[pd.Timestamp]) -> list[int]:
    if day < START:
        return []
    i = sum(1 for d in trading_days if START <= d < day)
    if i == 0:
        return list(range(N_TRANCHES))
    return [(i % HOLD_DAYS) // 2] if i % 2 == 0 else []


def step() -> None:
    """Process every trading day after the last processed one, in order (catches up missed runs)."""
    topix_all, panel_all, fins = E3.load(CACHE)
    state = load_state()
    last = pd.Timestamp(state["last_step"]) if state["last_step"] else topix_all.index[-2]
    todo = [d for d in topix_all.index if d > last]
    if not todo:
        print(json.dumps({"skipped": f"already stepped for {last.date()}"}))
        return
    for day in todo:
        state = load_state()
        step_day(state, topix_all.loc[:day], {k: v.loc[:day] for k, v in panel_all.items()}, fins)


def step_day(state: dict, topix: pd.DataFrame, panel: dict, fins: dict) -> None:
    dates = topix.index
    today = dates[-1]
    o, c = panel["O"], panel["C"].ffill()
    ledger_rows = []
    for name in PORTFOLIOS:
        pf = state[name]
        pend = pf["pending"]
        if pend and pd.Timestamp(pend["exec_date"]) == today:
            bought = set().union(*(pf["tranches"][m]["holdings"] for m in range(N_TRANCHES)
                                   if str(m) not in pend["orders"]))
            for k_str, cands in pend["orders"].items():
                tr = pf["tranches"][int(k_str)]
                for code, sh in tr["holdings"].items():
                    px = o.at[today, code] if o.at[today, code] > 0 else c[code].loc[:today].iloc[-2]
                    tr["cash"] += sh * px * (1 - COST)
                tr["holdings"] = {}
                buys = [x for x in cands if o.at[today, x] > 0 and x not in bought][:PER_TRANCHE]
                bought |= set(buys)
                if buys:
                    alloc = tr["cash"] / len(buys)
                    tr["holdings"] = {x: alloc / (o.at[today, x] * (1 + COST)) for x in buys}
                    tr["cash"] = 0.0
            pf["pending"] = None
        elif pend and pd.Timestamp(pend["exec_date"]) < today:
            raise SystemExit(f"{name}: pending orders for {pend['exec_date']} were never executed (missed a day?)")
        value = sum(t["cash"] + sum(sh * c.at[today, x] for x, sh in t["holdings"].items()) for t in pf["tranches"])
        ledger_rows.append((name, value, sum(len(t["holdings"]) for t in pf["tranches"])))

    nxt = next_trading_day(today)
    cal = json.loads((CACHE / "calendar.json").read_text())
    trading_days = sorted(set(dates[dates >= START]) | {pd.Timestamp(r["Date"]) for r in cal
                                                         if r["HolDiv"] in ("1", "2") and pd.Timestamp(r["Date"]) >= START})
    due = due_tranches(nxt, trading_days)
    if due:
        sig = E3.signals(panel, fins, dates)
        comp = sig["composite"].loc[today].dropna()
        feats, ok = features(panel, fins, dates)
        rank = {
            "R": comp.sort_values(ascending=False, kind="stable"),
            "M": model_scores(feats, ok, today).sort_values(ascending=False, kind="stable"),
        }
        risk = trap_risk(today)
        rank["L"] = rank["R"][[risk.get(x, 3) <= TRAP_MAX for x in rank["R"].index]]
        for name in PORTFOLIOS:
            pf = state[name]
            # names kept by tranches that do not trade tomorrow, plus names given to earlier due tranches
            others = set().union(*(pf["tranches"][m]["holdings"] for m in range(N_TRANCHES) if m not in due))
            orders, taken = {}, set()
            for k in due:
                pool = [x for x in rank[name].index if x not in others and x not in taken]
                orders[str(k)] = pool[:PER_TRANCHE + SPARE]
                taken |= set(pool[:PER_TRANCHE])
            pf["pending"] = {"decided_on": str(today.date()), "exec_date": str(nxt.date()), "orders": orders}
        write_order_sheet(state, today, nxt, due)

    state["last_step"] = str(today.date())
    (FWD / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n")
    ledger = FWD / "ledger.csv"
    new = not ledger.exists()
    with ledger.open("a") as f:
        if new:
            f.write("date,portfolio,value,positions,topix_close\n")
        for name, value, n in ledger_rows:
            f.write(f"{today.date()},{name},{value:.0f},{n},{topix['C'].iloc[-1]:.2f}\n")
    print(json.dumps({"date": str(today.date()), "values": {n: round(v) for n, v, _ in ledger_rows},
                      "next_trading_day": str(nxt.date()), "tranches_due": due}, ensure_ascii=False))


def write_order_sheet(state: dict, today, nxt, due) -> None:
    """Human-readable order sheet for the real-money portfolio R (and the paper ones for reference)."""
    names_file = FWD / "names.json"
    names = json.loads(names_file.read_text()) if names_file.exists() else {}
    nm = lambda x: f"{x} {names.get(x, '')}".strip()  # noqa: E731
    lines = [f"# {nxt.date()} の寄付の注文（{today.date()} の終値で決定）", ""]
    for name in PORTFOLIOS:
        pf = state[name]
        title = {"R": "R（実運用：ルール）", "M": "M（ペーパー：LightGBM）", "L": "L（ペーパー：Claude が決算短信を確認）"}[name]
        lines += [f"## {title}", "", "| 組 | 売る（寄成） | 買う（寄成、上から2つ。寄付がつかなければ次の候補） |", "|---|---|---|"]
        for k, cands in pf["pending"]["orders"].items():
            sell = "、".join(nm(x) for x in sorted(pf["tranches"][int(k)]["holdings"])) or "なし"
            lines.append(f"| {int(k) + 1} | {sell} | {'、'.join(nm(x) for x in cands)} |")
        lines.append("")
    lines.append("同じ銘柄を売って買い直す場合は、実際には何もしなくてよい。")
    out = FWD / "orders" / f"{nxt.date()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def notify() -> None:
    """Send the real-money (R) part of tomorrow's order sheet to Slack, once per sheet."""
    url = os.environ.get("SLACK_WEBHOOK_URL")
    state = load_state()
    pend = state["R"]["pending"]
    if not url or not pend or state.get("notified") == pend["exec_date"]:
        print(json.dumps({"notified": False}))
        return
    sheet = (FWD / "orders" / f"{pend['exec_date']}.md").read_text()
    r_part = sheet.split("## M")[0].strip()
    r = requests.post(url, json={"text": r_part + "\n\n（EXP-004 前向き検証。同じ銘柄は売買不要）"}, timeout=10)
    if r.status_code == 200:
        state["notified"] = pend["exec_date"]
        (FWD / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"notified": r.status_code == 200}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"update-data": update_data, "train": train, "score": score, "step": step, "notify": notify}.get(
        cmd, lambda: sys.exit(__doc__))()
