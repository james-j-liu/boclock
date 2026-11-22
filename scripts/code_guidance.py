"""Code Council and member documents for the decision they signal at the next
announcement (Jev). Needs OPENROUTER_API_KEY. -> data/processed/guidance_codes.json

    python scripts/code_guidance.py [--since 2016-01-01] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boclock.forecast.guidance import code_documents, documents
from boclock.schema import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2016-01-01")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    rate = [(d, float(r)) for d, r in
            json.loads((ROOT / "site/macro.json").read_text())["series"]["policy_rate"]["data"]]
    docs = documents(corpus, args.since)
    print(f"{len(docs)} documents since {args.since}")
    code_documents(docs, rate, limit=args.limit)


if __name__ == "__main__":
    main()
