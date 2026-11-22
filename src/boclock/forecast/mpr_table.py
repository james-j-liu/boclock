"""The Bank of Canada's CPI projection for each quarter of the current year, read from
the Monetary Policy Report's summary table ("Summary of the quarterly projection for
Canada" / "Summary of the base-case projection"):

    Summary of the quarterly projection for Canada*
    2026 2025 2026 2027 2028
    Q1 Q2 Q3 Q4 Q4 Q4 Q4 Q4
    CPI inflation (year-over-year percentage change)
    2.2 (2.2) 3.0 (3.0) 3.0 (2.5) 3.4 …

The first year header names the quarters; the first four values not in parentheses are
that year's Q1–Q4 (bracketed numbers are the previous Report's).
"""
from __future__ import annotations

import re

_HEAD = re.compile(r"Summary of the (?:quarterly |base[- ]case )?projection(?: for Canada)?", re.I)
_CPI = re.compile(r"CPI\s+in(?:fl|ﬂ)\s?ation", re.I)
_NUM = re.compile(r"(?<![\d(])(-?\d{1,2}\.\d)(?![\d)])")


def cpi_projection(text: str) -> dict[str, float]:
    """{'2026Q1': 2.2, …} — every quarterly column of the table, or {}.

    The header lists the quarterly columns' years, then the annual (Q4/Q4) columns'
    years as a consecutive run: '2021 2022 2020 2021 2022 2023 / Q2 Q3 Q4 Q1 Q4 Q4 Q4 Q4'
    means 2021Q2, 2021Q3, 2021Q4, 2022Q1, then annual 2020-2023."""
    for h in _HEAD.finditer(text or ""):
        seg = text[h.end():h.end() + 2500]
        c = _CPI.search(seg)
        if not c:
            continue
        head = seg[:c.start()]
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", head)]
        qs = [int(q) for q in re.findall(r"\bQ([1-4])\b", head)]
        if len(years) < 2 or len(qs) < 2:
            continue
        # trailing consecutive run of years = the annual columns
        k = len(years) - 1
        while k > 0 and years[k - 1] == years[k] - 1:
            k -= 1
        n_annual = len(years) - k
        q_years = years[:k] or years[:1]
        n_q = len(qs) - n_annual
        if n_q <= 0:
            continue
        body = re.sub(r"\(\s*-?\d{1,2}\.\d\s*\)", " ", seg[c.end():c.end() + 500])
        vals = [float(x) for x in _NUM.findall(body)][:n_q]
        if len(vals) < n_q or not all(-3 < v < 15 for v in vals):
            continue
        out, y = {}, q_years[0]
        for i, (q, v) in enumerate(zip(qs[:n_q], vals)):
            if i and q <= qs[i - 1]:
                y += 1
            out[f"{y}Q{q}"] = v
        return out
    return {}
