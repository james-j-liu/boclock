"""Code each Bank of Canada document for the decision it signals at the next announcement.

The Governing Council decides by consensus and publishes no votes, so the forecast is
of the Council's decision. Every rate announcement, press-conference opening
statement, Summary of deliberations, and every speech or parliamentary evidence by a
sitting Council member is put to a decision model (Drex Max Finance, blind) as one
typed question — "on this text alone, what will the Council most likely decide at its
next announcement?" — over cut 50 / cut 25 / hold / hike 25 / hike 50. A second
question rates how much the text is about the near-term policy outlook.

Codes are cached by document id in data/processed/guidance_codes.json.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "data/processed/guidance_codes.json"
OPTIONS = {"cut_50": -2, "cut_25": -1, "hold": 0, "hike_25": 1, "hike_50": 2}
CRITERIA = {
    "cut_50": "Cut the target for the overnight rate by 50 basis points or more.",
    "cut_25": "Cut the target for the overnight rate by 25 basis points.",
    "hold": "Keep the target for the overnight rate unchanged.",
    "hike_25": "Raise the target for the overnight rate by 25 basis points.",
    "hike_50": "Raise the target for the overnight rate by 50 basis points or more.",
}
INSTRUCTIONS = (
    "The text is a Bank of Canada rate announcement, press-conference opening statement, "
    "Summary of Governing Council deliberations, or a speech or parliamentary evidence by a "
    "member of the Governing Council, given on the date shown, when the policy rate stood at "
    "the level shown. Based ONLY on this text, what is the Governing Council most likely to "
    "decide at its NEXT scheduled rate announcement after this text? Treat explicit forward "
    "guidance as the strongest evidence ('further increases will be needed', 'we are prepared "
    "to raise rates further', 'if the economy evolves as expected, further cuts are likely', "
    "'the current rate is about the right level', 'we will take decisions one meeting at a "
    "time'); then the stated balance of risks to inflation and how restrictive policy is "
    "judged to be. An announcement that only reports a decision just taken, with no view on "
    "the next one, points to holding. If the text does not discuss the outlook for policy at "
    "all, choose hold."
)
FOCUS_LEVELS = ["Not about monetary policy", "Mentions policy in passing",
                "Some discussion of the policy outlook", "Substantial discussion of the policy outlook",
                "Mainly about the next policy decisions"]
FOCUS_INSTRUCTIONS = ("How much of this text is about the outlook for Canadian monetary policy — "
                      "the path of the policy rate over the coming announcements?")
COUNCIL_TYPES = {"mp_announce", "mp_statement", "mp_account"}
MEMBER_TYPES = {"speech", "testimony", "interview"}


# Live reader: Drex Max Finance, blind (no date; names -> [OFFICIAL]; years -> [YEAR]).
# Chosen on the MPC project, where it was the only decision model significantly better
# than Jev 1.13 through the vote model; blind so a model trained on later data cannot
# simply recall what the Council went on to do.
LIVE_MODEL = "nace-ai/drex-max-finance"
LIVE_BLIND = True
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def load_codes() -> dict:
    try:
        return json.loads(CODES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_codes(codes: dict) -> None:
    CODES.write_text(json.dumps(codes, sort_keys=True, indent=0), encoding="utf-8")


def _rate_on(rate, d):
    r = rate[0][1]
    for dd, v in rate:
        if dd <= d:
            r = v
        else:
            break
    return r


def documents(corpus, since: str = "2016-01-01"):
    from ..roster_gc import is_gc
    from ..tenure import for_corpus
    ten = for_corpus(corpus)
    out = []
    for s in corpus:
        if s.date < since or not (s.text or "").strip():
            continue
        if s.source_type in COUNCIL_TYPES:
            out.append(s)
        elif s.source_type in MEMBER_TYPES and is_gc(s.speaker) and ten.active(s.speaker, s.date) \
                and s.is_policy is not False:
            out.append(s)
    return out


def code_documents(docs, rate, max_workers: int = 8, cap: int = 90000, limit: int | None = None,
                   verbose: bool = True, model: str | None = None, blind: bool | None = None) -> dict:
    import requests
    from ..judge.jev import _Client
    from ..judge.openrouter import excerpt
    MODEL = model or LIVE_MODEL
    blind = LIVE_BLIND if blind is None else blind
    anon = None
    if blind:
        from ..process.anonymize import Anonymizer
        from ..process.roster import build_roster
        anon = Anonymizer(build_roster([s.speaker for s in docs]))

    codes = load_codes()

    def stale(c):   # coded by another reader or setting (older codes have no 'model': Jev 1.13)
        return c.get("model", "typesafe/jev-1.13") != MODEL or bool(c.get("blind", False)) != bool(blind)
    todo = [s for s in docs if s.id not in codes or stale(codes[s.id])]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"guidance: {len(codes)} coded, {len(todo)} to code with {MODEL}{' (blind)' if blind else ''}")
    if not todo:
        return codes
    client = _Client()
    instr = INSTRUCTIONS.replace("given on the date shown, when", "given when") if blind else INSTRUCTIONS

    def one(s):
        state = {"date": s.date, "policy_rate_percent": _rate_on(rate, s.date),
                 "document_type": s.source_type}
        text = s.text
        if blind:
            state.pop("date")
            text = _YEAR.sub("[YEAR]", anon(s.text))

        def build(c):
            return {"model": MODEL, "state": {**state, "text": excerpt(text, c, s.source_type)},
                    "questions": {
                        "next_decision": {"type": "choice", "instructions": instr, "criteria": CRITERIA},
                        "focus": {"type": "score", "instructions": FOCUS_INSTRUCTIONS, "criteria": FOCUS_LEVELS}}}
        try:
            res = client.ask_shrinking(build, cap)
        except (requests.RequestException, RuntimeError) as e:
            return s, None, str(e)[:120]
        a = res["answers"]["next_decision"]
        probs = {k: float(a.get("probabilities", {}).get(k, 0.0)) for k in OPTIONS}
        tot = sum(probs.values()) or 1.0
        probs = {k: v / tot for k, v in probs.items()}
        f = res["answers"]["focus"].get("score")
        return s, {"date": s.date, "speaker": s.speaker, "type": s.source_type,
                   "model": MODEL, "served_model": res.get("model"), "blind": blind,
                   "p": {k: round(v, 4) for k, v in probs.items()},
                   "ev": round(sum(OPTIONS[k] * v for k, v in probs.items()), 4),
                   "focus": None if f is None else round(float(f) / (len(FOCUS_LEVELS) - 1), 3)}, None

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fu in as_completed([ex.submit(one, s) for s in todo]):
            s, rec, err = fu.result()
            if rec:
                codes[s.id] = rec
            elif verbose:
                print(f"  [skip] {s.date} {s.speaker}: {err}")
            done += 1
            if done % 100 == 0:
                save_codes(codes)
                if verbose:
                    print(f"  {done}/{len(todo)} coded, ${client.cost:.3f}")
    save_codes(codes)
    if verbose:
        print(f"guidance: coded {done}, spend ${client.cost:.4f}")
    return codes
