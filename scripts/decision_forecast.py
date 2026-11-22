"""Backtest the Governing Council decision model and forecast the next announcement.

    python scripts/decision_forecast.py              # backtest + site/decision_forecast.json
    python scripts/decision_forecast.py --refresh    # first refresh markets/MPR and code new docs

Every announcement since 2017 is predicted by a model fitted on the other years only
(leave-one-year-out); input groups are added one at a time and the live model is the
step with the lowest out-of-sample log loss.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from boclock.forecast import decision as D
from boclock.schema import load_corpus

G = D.FEATURE_GROUPS
LADDER = [("last decision only", [])]
_acc = []
for name, grp in [("+ Council's own words", "council"), ("+ pace", "pace"), ("+ members' speeches", "members"),
                  ("+ CPI vs MPR", "data"), ("+ oil", "oil"), ("+ Fed", "fed"), ("+ Canadian dollar", "fx")]:
    _acc = _acc + G[grp]
    LADDER.append((name, list(_acc)))
LABELS = {-2: "Cut 50bp", -1: "Cut 25bp", 0: "Hold", 1: "Hike 25bp", 2: "Hike 50bp"}


def cv(rows, feats, l2=2.0):
    out = np.zeros((len(rows), len(D.DELTAS)))
    for y in sorted({r["date"][:4] for r in rows}):
        tr = [r for r in rows if r["date"][:4] != y]
        idx = [i for i, r in enumerate(rows) if r["date"][:4] == y]
        P = D.OrderedLogit(feats, l2).fit(tr).predict([rows[i] for i in idx])
        out[idx] = P
    return out


def evaluate(rows, P):
    y = np.array([r["y"] for r in rows]) - D.DELTAS[0]
    hit = np.mean(np.argmax(P, axis=1) == y)
    moves = np.array([r["decision"] != 0 for r in rows])
    return {"log_loss": float(-np.mean(np.log(P[np.arange(len(y)), y]))),
            "brier": float(np.mean(np.sum((P - np.eye(len(D.DELTAS))[y]) ** 2, axis=1))),
            "hit": float(hit), "p_actual": float(np.mean(P[np.arange(len(y)), y])),
            "move_hit": float(np.mean((np.argmax(P, axis=1) == y)[moves])) if moves.any() else None,
            "n": len(rows), "n_moves": int(moves.sum())}


def next_date(corpus, last):
    ann = sorted([s for s in corpus if s.source_type == "mp_announce"], key=lambda s: s.date)[-1].text or ""
    m = re.search(r"next scheduled date for announcing the overnight rate target is\s+([A-Z][a-z]+ \d{1,2}, \d{4})", ann)
    if m:
        return dt.datetime.strptime(m.group(1), "%B %d, %Y").date().isoformat()
    return (dt.date.fromisoformat(last) + dt.timedelta(days=42)).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--since", default="2017-01-01")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    if args.refresh:
        try:
            D.refresh_inputs(corpus)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] input refresh failed: {e}")
        try:
            from boclock.forecast.guidance import code_documents, documents
            rate = [(d, float(r)) for d, r in
                    json.loads((ROOT / "site/macro.json").read_text())["series"]["policy_rate"]["data"]]
            code_documents(documents(corpus), rate)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] guidance coding skipped: {e}")
    ctx = D.Context(corpus)
    rows = D.panel(ctx, args.since)
    recent = (dt.date.today() - dt.timedelta(days=5 * 365 + 1)).isoformat()
    ridx = [i for i, r in enumerate(rows) if r["date"] >= recent]
    print(f"backtest: {len(rows)} announcements since {args.since} "
          f"({sum(r['decision'] != 0 for r in rows)} moves), leave-one-year-out")
    print(f"  {'model':24} {'logloss':>8} {'Brier':>7} {'called':>7} {'P(actual)':>9} {'moves called':>13}   last 5y: called  P(actual)")
    ladder, best = [], None
    for name, feats in LADDER:
        P = cv(rows, feats)
        ev, er = evaluate(rows, P), evaluate([rows[i] for i in ridx], P[ridx])
        print(f"  {name:24} {ev['log_loss']:8.4f} {ev['brier']:7.4f} {ev['hit']:7.1%} {ev['p_actual']:9.1%} "
              f"{ev['move_hit']:13.1%}   {er['hit']:7.1%} {er['p_actual']:9.1%}")
        ladder.append({"model": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()},
                       "recent": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in er.items()}})
        if best is None or ev["log_loss"] < best[1] - 1e-4:
            best = (name, ev["log_loss"], feats, P)
    chosen, Pc = best[2], best[3]
    print(f"chosen: {best[0]}")
    calls = [{"date": r["date"], "actual": r["decision"], "predicted": int(r["prev_decision"] + D.DELTAS[np.argmax(p)]),
              "p_actual": round(float(p[r["y"] - D.DELTAS[0]]), 3)} for r, p in zip(rows, Pc)]
    model = D.OrderedLogit(chosen, 2.0).fit(rows)
    coefs = dict(zip(chosen, map(float, model.b)))

    last_d, last_u = ctx.decisions[-1]
    nxt = next_date(corpus, last_d)
    asof = min(dt.date.today().isoformat(), nxt)
    f = D.features(ctx, last_d, last_u, (dt.date.fromisoformat(asof) + dt.timedelta(days=1)).isoformat())
    p = model.predict([f])[0]
    probs = {}
    for d, q in zip(D.DELTAS, p):
        k = int(np.clip(last_u + d, -2, 2))
        probs[k] = probs.get(k, 0.0) + float(q)
    rate = ctx.rate[-1][1]
    out = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "last": {"date": last_d, "decision_bp": last_u * 25, "rate": rate},
        "next": nxt, "as_of": asof,
        "probs": {str(k * 25): round(v, 4) for k, v in sorted(probs.items())},
        "expected_bp": round(25 * sum(k * v for k, v in probs.items()), 1),
        "contributions": {k: round(coefs[k] * f[k], 3) for k in chosen},
        "features": {k: round(float(f[k]), 3) for g in G.values() for k in g},
        "inputs": {"cpi_detail": f["_detail"], "core_change_pp": f["_core_chg"], "oil_pct": f["_oil_pct"],
                   "usdcad_pct": f["_fx_pct"], "fed_bp": round(f["fed"] * 25), "mpr_meeting": bool(f["_mpr"])},
        "council_docs": f["_council_docs"], "member_docs": f["_member_docs"],
        "model": {"features": chosen, "coefficients": {k: round(v, 3) for k, v in coefs.items()}},
        "backtest": {"since": args.since, "recent_since": recent, "ladder": ladder, "chosen": best[0],
                     "calls": calls},
    }
    (ROOT / "site/decision_forecast.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"next announcement {nxt} (as of {asof}): " +
          ", ".join(f"{LABELS[k]} {v:.0%}" for k, v in sorted(probs.items()) if v >= 0.005))


if __name__ == "__main__":
    main()
