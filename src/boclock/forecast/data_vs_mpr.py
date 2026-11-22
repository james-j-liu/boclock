"""CPI against the Bank of Canada's own projections, for the site's "Data vs MPR" tab.

Each Monetary Policy Report's summary table projects CPI inflation (y/y) quarter by
quarter (forecast/mpr_table.py). Here: the monthly CPI outturns (Statistics Canada,
via the Bank's Valet API), each quarter's average once complete, every Report's
quarterly projections, every monthly release against the projection for its quarter
in the latest Report published before it, and CPI-trim / CPI-median alongside (the
Bank's preferred core measures; the MPR table does not project them).
"""
from __future__ import annotations

import numpy as np

from .decision import _quarter, cpi_release


def build(ctx, since: str = "2016-01") -> dict:
    import json
    from pathlib import Path
    macro = json.loads((Path(__file__).resolve().parents[3] / "site/macro.json").read_text())["series"]
    trim = {d[:7]: v for d, v in macro["cpi_trim"]["data"] if d[:7] >= since}
    median = {d[:7]: v for d, v in macro["cpi_median"]["data"] if d[:7] >= since}
    cpi = {p: v for p, v in sorted(ctx.cpi.items()) if p >= since}
    # quarterly averages of complete quarters
    qs: dict[str, list] = {}
    for p, v in cpi.items():
        qs.setdefault(_quarter(p), []).append(v)
    q_avg = {q: round(float(np.mean(v)), 2) for q, v in qs.items() if len(v) == 3}
    reports = [{"date": d, "points": sorted(pr.items())} for d, pr in ctx.mpr if d >= since[:4]]
    releases = []
    for p, v in cpi.items():
        rel = cpi_release(p)
        src = next(((d, pr) for d, pr in reversed(ctx.mpr) if d < rel and _quarter(p) in pr), None)
        if src:
            e = src[1][_quarter(p)]
            releases.append({"period": p, "released": rel, "actual": v, "expected": e, "mpr": src[0],
                             "diff": round(v - e, 2), "z": round((v - e) / 0.3, 2)})
    # how each quarter's projection evolved across Reports, against the outturn
    quarters = {}
    for d, pr in ctx.mpr:
        for q, v in pr.items():
            quarters.setdefault(q, []).append({"mpr": d, "proj": v})
    qtable = [{"quarter": q, "outturn": q_avg.get(q), "projections": quarters[q][-4:]}
              for q in sorted(quarters) if q >= since[:4]]
    latest = ctx.mpr[-1] if ctx.mpr else None
    upcoming = [{"quarter": q, "proj": v} for q, v in sorted(latest[1].items()) if q not in q_avg] if latest else []
    return {"cpi": sorted(cpi.items()), "trim": sorted(trim.items()), "median": sorted(median.items()),
            "quarterly": sorted(q_avg.items()), "reports": reports, "releases": releases,
            "quarters": qtable, "latest_mpr": latest[0] if latest else None, "upcoming": upcoming}
