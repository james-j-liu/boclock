"""Forecast the Governing Council's next decision.

The Council publishes no votes, so the unit is the decision itself: an ordered logit
over the change from the previous decision (Δ, 25bp steps, -2…+2). Inputs, measured
over the window from the previous announcement up to the day before the next one:

  council    council_level   decision the Council's own last words point to (the last
                             announcement, press-conference statement and Summary of
                             deliberations, coded by Jev), as a level
             council_gap     the same minus the previous decision
  members    member_level    decision the Council members' speeches and evidence since
                             then point to (falls back to council_level when none)
  data       surprise        CPI releases in the window against the Bank's MPR projection
                             for that quarter, plus the change in core inflation
                             (average of CPI-trim and CPI-median)
  oil        oil             % change in WTI over the window (per 10%)
  fed        fed             change in the Fed funds target (25bp steps)
  fx         fx             % change in USD/CAD (a weaker loonie is inflationary; per 2%)
  tone       tone_council    hawkishness (tournament score) of the Council's own documents
                             (announcement, statement, deliberations, MPR) in the window,
                             minus their average over the previous 12 months
             tone_members    the same for members' speeches and evidence
  pace       pace, moved_last, mpr_x_pace   as for the MPC: has this cycle moved at every
                             announcement or only at MPR ones, did the Bank move last time
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/processed/forecast_inputs.json"
DELTAS = np.array([-2, -1, 0, 1, 2])
FEATURE_GROUPS = {
    "council": ["council_level", "council_gap"],
    "members": ["member_level"],
    "pace": ["pace", "moved_last", "mpr_x_pace"],
    "inertia": ["first_move", "run_len", "paused_after_move"],
    "data": ["surprise"],
    "oil": ["oil"],
    "fed": ["fed"],
    "fx": ["fx"],
    "tone": ["tone_council", "tone_members"],
}

TONE_COUNCIL = ("mp_announce", "mp_statement", "mp_account", "mp_report")
TONE_MEMBERS = ("speech", "testimony", "interview")


# --- inputs ------------------------------------------------------------------------
_HOST_UA = {"fred.stlouisfed.org": "curl/8.5.0", "query1.finance.yahoo.com": "Mozilla/5.0"}


def _get(url, params=None, tries=4):
    import requests
    host = re.sub(r"^https?://([^/]+).*", r"\1", url)
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": _HOST_UA.get(host, "Mozilla/5.0")},
                             timeout=60)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)[:100]
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _fred(sid):
    import pandas as pd
    df = pd.read_csv(io.StringIO(_get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid}).text))
    df.columns = ["d", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    return dict(zip(df.d[df.d >= "2014-01-01"], df.v[df.d >= "2014-01-01"].round(4)))


def _yahoo(t):
    j = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", {"range": "2y", "interval": "1d"}).json()
    j = j["chart"]["result"][0]
    return {dt.datetime.utcfromtimestamp(ts).date().isoformat(): round(float(c), 4)
            for ts, c in zip(j["timestamp"], j["indicators"]["quote"][0]["close"]) if c is not None}


def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def refresh_inputs(corpus) -> dict:
    """Markets (FRED, topped up from Yahoo for the latest days) and the CPI projections
    read from every MPR in the corpus."""
    from .mpr_table import cpi_projection
    c = load_cache()
    m = c.setdefault("markets", {})
    for k, fn in {"wti": lambda: {**_fred("DCOILWTICO"), **_safe(lambda: _yahoo("CL=F"))},
                  "fed": lambda: _fred("DFEDTARU"),
                  "usdcad": lambda: {**_fred("DEXCAUS"), **_safe(lambda: _yahoo("CAD=X"))}}.items():
        try:
            m[k] = fn()
        except Exception as e:  # noqa: BLE001
            print(f"[inputs] {k} failed ({e}); keeping cached copy")
    c["mpr"] = {s.date: p for s in corpus if s.source_type == "mp_report" and s.date >= "2015-01-01"
                for p in [cpi_projection(s.text)] if p}
    c["updated"] = dt.date.today().isoformat()
    CACHE.write_text(json.dumps(c, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return c


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return {}


def cpi_release(period: str) -> str:
    """Statistics Canada publishes CPI for month m around the third Tuesday of m+1."""
    y, m = int(period[:4]), int(period[5:7])
    m, y = (m + 1, y) if m < 12 else (1, y + 1)
    d = dt.date(y, m, 15)
    while d.weekday() != 1:
        d += dt.timedelta(days=1)
    return d.isoformat()


# --- context -------------------------------------------------------------------------
class Context:
    def __init__(self, corpus, macro=None):
        from .guidance import COUNCIL_TYPES, MEMBER_TYPES, load_codes
        from ..macro.boc_decisions import decisions
        macro = macro or json.loads((ROOT / "site/macro.json").read_text())
        self.decisions = [(d["date"], int(round(d["units"]))) for d in decisions(corpus)
                          if d.get("units") is not None]
        self.rate = [(d, float(v)) for d, v in macro["series"]["policy_rate"]["data"]]
        ser = macro["series"]
        self.cpi = {d[:7]: v for d, v in ser["cpi_headline"]["data"]}
        self.core = {d[:7]: (v + dict(ser["cpi_median"]["data"]).get(d, v)) / 2
                     for d, v in ser["cpi_trim"]["data"]}
        self.inputs = load_cache()
        self.mpr = sorted(self.inputs.get("mpr", {}).items())
        self.mpr_dates = [d for d, _ in self.mpr] + [s.date for s in corpus if s.source_type == "mp_report"]
        codes = load_codes()
        self.docs = sorted([(s.date, s.source_type, s.speaker, codes[s.id], s.id) for s in corpus
                            if s.id in codes and s.source_type in COUNCIL_TYPES | MEMBER_TYPES],
                           key=lambda x: (x[0], x[1], x[4]))
        self.tones = sorted((s.date, s.source_type, (s.mu - 50) / 5) for s in corpus if s.mu is not None)


def _signal(docs):
    if not docs:
        return None
    w = np.array([(0.25 + (c.get("focus") or 0.0)) * (2.0 if i == len(docs) - 1 else 1.0)
                  for i, (_, _, _, c, _) in enumerate(docs)])
    return float((w * np.array([c["ev"] for _, _, _, c, _ in docs])).sum() / w.sum())


def _mkt(ser, start, end, pct=True):
    if not ser:
        return 0.0
    keys = sorted(ser)

    def avg(d):
        v = [ser[k] for k in keys if k < d][-5:]
        return sum(v) / len(v) if v else None
    a, b = avg(start), avg(end)
    if a is None or b is None:
        return 0.0
    return math.log(b / a) * 100 if pct else b - a


def _quarter(p):
    y, m = int(p[:4]), int(p[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def surprises(ctx, start, end):
    det = []
    for p, v in sorted(ctx.cpi.items()):
        rel = cpi_release(p)
        if not (start < rel < end):
            continue
        proj = next((pr[_quarter(p)] for d, pr in reversed(ctx.mpr) if d < rel and _quarter(p) in pr), None)
        if proj is None:
            continue
        det.append({"period": p, "released": rel, "actual": v, "expected": proj,
                    "z": round((v - proj) / 0.3, 2)})
    core_now = [v for p, v in sorted(ctx.core.items()) if cpi_release(p) < end]
    core_then = [v for p, v in sorted(ctx.core.items()) if cpi_release(p) < start]
    core_chg = (core_now[-1] - core_then[-1]) if core_now and core_then else 0.0
    z = np.mean([d["z"] for d in det]) if det else 0.0
    return float(np.clip(z + core_chg / 0.2, -4, 4)), det, core_chg


def pace(ctx, date):
    past = [(d, u) for d, u in ctx.decisions if d < date]
    moves = [(d, np.sign(u)) for d, u in past if u]
    is_mpr = lambda d: any(abs((dt.date.fromisoformat(d) - dt.date.fromisoformat(x)).days) <= 2
                           for x in ctx.mpr_dates)
    if not moves:
        return {"pace": 0.0, "moved_last": 0.0, "mpr_x_pace": 0.0, "mpr": float(is_mpr(date)),
                "first_move": 0.0, "run_len": 0.0, "paused_after_move": 0.0}
    cdir = moves[-1][1]
    start = moves[-1][0]
    for d, s in reversed(moves):
        if s != cdir:
            break
        start = d
    active = (dt.date.fromisoformat(date) - dt.date.fromisoformat(moves[-1][0])).days < 400
    this = is_mpr(date)
    same = [(d, u) for d, u in past if d >= start and is_mpr(d) == this]
    moved = sum(1 for _, u in same if np.sign(u) == cdir)
    pc = (moved + 0.5) / (len(same) + 1.0) if active else 0.0
    # inertia: the Bank of Canada tends to follow a cycle's first move straight away, and a
    # pause after a run tends to last
    run = 0
    for _, u in reversed(past):
        if u and np.sign(u) == cdir:
            run += 1
        else:
            break
    n_cycle_moves = sum(1 for d, u in past if d >= start and np.sign(u) == cdir)
    first = float(cdir) if (run == 1 and n_cycle_moves == 1) else 0.0
    paused = float(cdir) if (len(past) >= 2 and past[-1][1] == 0 and np.sign(past[-2][1]) == cdir) else 0.0
    return {"pace": float(cdir * pc), "moved_last": float(np.sign(past[-1][1])),
            "mpr_x_pace": float(cdir * pc * (1 if this else -1)) if active else 0.0, "mpr": float(this),
            "first_move": first if active else 0.0, "run_len": float(cdir * min(run, 5)) if active else 0.0,
            "paused_after_move": paused if active else 0.0}


def _tone_shift(ctx, types, start, end):
    back = (dt.date.fromisoformat(start) - dt.timedelta(days=365)).isoformat()
    now = [t for d, ty, t in ctx.tones if ty in types and start <= d < end]
    before = [t for d, ty, t in ctx.tones if ty in types and back <= d < start]
    return float(np.mean(now) - np.mean(before)) if now and before else 0.0


def features(ctx, prev_date, prev_dec, date):
    council = [x for x in ctx.docs if prev_date <= x[0] < date and x[1] in ("mp_announce", "mp_statement", "mp_account")]
    members = [x for x in ctx.docs if prev_date < x[0] < date and x[1] in ("speech", "testimony", "interview")]
    cs, ms = _signal(council), _signal(members)
    cs = prev_dec if cs is None else cs
    ms_v = cs if ms is None else ms
    surp, det, core_chg = surprises(ctx, prev_date, date)
    mk = ctx.inputs.get("markets", {})
    oil = _mkt(mk.get("wti", {}), prev_date, date)
    pc = pace(ctx, date)
    return {"council_level": cs, "council_gap": cs - prev_dec, "member_level": ms_v,
            "surprise": surp, "oil": float(np.clip(oil / 10, -4, 4)),
            "fed": _mkt(mk.get("fed", {}), prev_date, date, pct=False) * 4,
            "fx": float(np.clip(_mkt(mk.get("usdcad", {}), prev_date, date) / 2, -4, 4)),
            **{k: pc[k] for k in ("pace", "moved_last", "mpr_x_pace", "first_move", "run_len",
                                  "paused_after_move")},
            "tone_council": _tone_shift(ctx, TONE_COUNCIL, prev_date, date),
            "tone_members": _tone_shift(ctx, TONE_MEMBERS, prev_date, date),
            "prev_decision": prev_dec,
            "_mpr": pc["mpr"], "_detail": det, "_core_chg": round(core_chg, 2),
            "_oil_pct": round(oil, 1), "_fx_pct": round(_mkt(mk.get("usdcad", {}), prev_date, date), 1),
            "_council_docs": [{"date": d, "type": t, "signal": c["ev"]} for d, t, _, c, _ in council],
            "_member_docs": [{"date": d, "type": t, "speaker": sp, "signal": c["ev"]} for d, t, sp, c, _ in members],
            "_member_signal": ms}


def panel(ctx, since="2017-01-01"):
    rows = []
    for (d0, u0), (d1, u1) in zip(ctx.decisions, ctx.decisions[1:]):
        if d1 < since:
            continue
        r = features(ctx, d0, u0, d1)
        r.update(date=d1, decision=u1, y=int(np.clip(u1 - u0, -2, 2)))
        rows.append(r)
    return rows


class OrderedLogit:
    def __init__(self, feats, l2=1.0):
        self.feats, self.l2 = feats, l2

    def _X(self, rows):
        return np.array([[r[f] for f in self.feats] for r in rows], float) if self.feats else np.zeros((len(rows), 0))

    @staticmethod
    def _probs(eta, cuts):
        cdf = 1 / (1 + np.exp(-(cuts[None, :] - eta[:, None])))
        cdf = np.hstack([np.zeros((len(eta), 1)), cdf, np.ones((len(eta), 1))])
        return np.clip(np.diff(cdf, axis=1), 1e-9, 1)

    def fit(self, rows):
        X, y = self._X(rows), np.array([r["y"] for r in rows]) - DELTAS[0]
        k = X.shape[1]

        def unpack(th):
            return th[:k], np.cumsum(np.concatenate([[th[k]], np.exp(th[k + 1:])]))

        def nll(th):
            b, c = unpack(th)
            p = self._probs(X @ b, c)
            return -np.log(p[np.arange(len(y)), y]).sum() + self.l2 * (b @ b)
        res = minimize(nll, np.concatenate([np.zeros(k), [-4.0], np.log([2.0, 4.0, 2.0])]), method="L-BFGS-B")
        self.b, self.cuts = unpack(res.x)
        return self

    def predict(self, rows):
        return self._probs(self._X(rows) @ self.b, self.cuts)
