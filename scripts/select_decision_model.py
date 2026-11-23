"""Choose the decision model's inputs, with a significance check.

Every announcement since 2001 is predicted by a model fitted on the other years
(leave-one-year-out). The core structure (the Council's own words and the pace of the
cycle) is kept only if, as a bundle, it beats "repeat the last decision" significantly;
every other input group — including the inertia inputs (first move of a cycle, run
length, pause after a move) — is kept only if it clears the same test on its own: a
paired bootstrap over announcements (2000 draws) must put the 95% interval of the change
in log loss wholly below zero. Inputs not available early in the sample (coded words from
2016, MPR projections from 2017, market series from 2014) are simply absent before then.

    python scripts/select_decision_model.py   # -> data/processed/decision_model.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from boclock.forecast import decision as D
from boclock.schema import load_corpus

OUT = ROOT / "data/processed/decision_model.json"
G = D.FEATURE_GROUPS
CORE = [("Council's own words", G["council"]), ("pace", G["pace"])]
ADDONS = [("inertia (first move, run length, pause)", G["inertia"]), ("members' speeches", G["members"]),
          ("CPI vs MPR", G["data"]), ("oil", G["oil"]), ("Fed", G["fed"]), ("Canadian dollar", G["fx"]),
          ("hawkishness scores (Council & members)", G["tone"])]
SINCE = "2001-01-01"


def cv(rows, feats, l2=2.0):
    P = np.zeros((len(rows), len(D.DELTAS)))
    for y in sorted({r["date"][:4] for r in rows}):
        tr = [r for r in rows if r["date"][:4] != y]
        idx = [i for i, r in enumerate(rows) if r["date"][:4] == y]
        P[idx] = D.OrderedLogit(feats, l2).fit(tr).predict([rows[i] for i in idx])
    return P


def loss(rows, P):
    y = np.array([r["y"] for r in rows]) - D.DELTAS[0]
    return -np.log(P[np.arange(len(y)), y])


def boot(a, b, n=2000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    warnings.filterwarnings("ignore")
    ctx = D.Context(load_corpus(ROOT / "data/processed/corpus.jsonl"))
    rows = D.panel(ctx, SINCE)
    print(f"{len(rows)} announcements since {SINCE} ({sum(r['decision'] != 0 for r in rows)} moves)")
    lb = loss(rows, cv(rows, []))
    core = [f for _, g in CORE for f in g]
    lc = loss(rows, cv(rows, core))
    m, lo, hi = boot(lc, lb)
    rep = {"baseline_log_loss": round(float(lb.mean()), 4),
           "core": {"log_loss": round(float(lc.mean()), 4), "change": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "significant": hi < 0, "groups": []}}
    print(f"core vs last decision: {lc.mean():.4f} vs {lb.mean():.4f}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if not hi < 0:
        core, lc = [], lb
    for name, g in CORE:
        rest = [f for f in core if f not in g]
        mm, l2, h2 = boot(lc, loss(rows, cv(rows, rest)))
        rep["core"]["groups"].append({"group": name, "contribution": round(mm, 4), "ci95": [round(l2, 4), round(h2, 4)],
                                      "significant_alone": h2 < 0})
        print(f"  core {name:24} {mm:+.4f} [{l2:+.4f}, {h2:+.4f}]")
    cur, cl = list(core), lc
    rep["addons"] = []
    for name, g in ADDONS:
        nl = loss(rows, cv(rows, cur + g))
        mm, l2, h2 = boot(nl, cl)
        keep = h2 < 0
        rep["addons"].append({"group": name, "log_loss": round(float(nl.mean()), 4), "change": round(mm, 4),
                              "ci95": [round(l2, 4), round(h2, 4)], "kept": keep})
        print(f"  add-on {'KEEP' if keep else 'drop'} {name:40} {mm:+.4f} [{l2:+.4f}, {h2:+.4f}]")
        if keep:
            cur, cl = cur + g, nl
    out = {"features": cur, "since": SINCE, "log_loss": round(float(cl.mean()), 4), "report": rep,
           "selected": dt.date.today().isoformat(),
           "method": "core structure kept as a bundle if significantly better than 'repeat the last decision'; "
                     "each other input group kept only if it significantly lowers log loss on its own. "
                     "Leave-one-year-out; paired bootstrap over announcements, 2000 draws, 95% interval."}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"selected {cur}")


if __name__ == "__main__":
    main()
