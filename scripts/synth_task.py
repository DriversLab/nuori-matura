#!/usr/bin/env python
"""Deterministic sharding of data/wiki/articles.jsonl into generation tasks + prompt printing.

  python scripts/synth_task.py --list                          # JSON: [{shard, subject, n_articles, n_items}]
  python scripts/synth_task.py --shard historia-03             # print generation prompts for every article in the shard
  python scripts/synth_task.py --shard historia-03 --article 2 # print the prompt for one article (1-based)
  python scripts/synth_task.py --shard historia-03 --articles  # print article texts only (open-book verification)

Used by the Claude generator/verifier agents and by train/generate_synthetic.py (API backends).
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

from eval.schema import SUBJECTS, read_jsonl
from train.synth_prompt import generation_prompt

SHARD_SIZE = 6


def items_for(article: dict) -> int:
    n = len(article["text"])
    return 7 if n >= 12000 else 6 if n >= 5000 else 4


def build_shards(articles_path: str | pathlib.Path = ROOT / "data/wiki/articles.jsonl", shard_size: int = SHARD_SIZE) -> dict[str, list[dict]]:
    arts = read_jsonl(articles_path)
    shards: dict[str, list[dict]] = {}
    for subject in SUBJECTS:
        subj = sorted((a for a in arts if a["subject"] == subject), key=lambda a: a["title"])
        for i in range(0, len(subj), shard_size):
            shard = f"{subject}-{i // shard_size + 1:02d}"
            shards[shard] = [
                {"article": a, "n_items": items_for(a), "id_prefix": f"syn-{shard}-{k}"}
                for k, a in enumerate(subj[i : i + shard_size], 1)
            ]
    return shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles-path", default=str(ROOT / "data/wiki/articles.jsonl"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard")
    ap.add_argument("--article", type=int)
    ap.add_argument("--articles", action="store_true", help="print article texts only")
    args = ap.parse_args()
    shards = build_shards(args.articles_path)
    if args.list:
        print(json.dumps([
            {"shard": s, "subject": s.rsplit("-", 1)[0], "n_articles": len(t), "n_items": sum(x["n_items"] for x in t),
             "titles": [x["article"]["title"] for x in t]}
            for s, t in shards.items()
        ], ensure_ascii=False, indent=1))
        return
    if not args.shard or args.shard not in shards:
        sys.exit(f"unknown shard {args.shard!r}; use --list")
    tasks = shards[args.shard]
    if args.article:
        tasks = [tasks[args.article - 1]]
    for t in tasks:
        a = t["article"]
        if args.articles:
            print(f"==================== ARTYKUŁ: {a['title']} ({a['url']})\n{a['text']}\n")
        else:
            print(f"==================== TASK id_prefix={t['id_prefix']} n_items={t['n_items']}")
            print(generation_prompt(a, t["n_items"], t["id_prefix"]))
            print()


if __name__ == "__main__":
    main()
