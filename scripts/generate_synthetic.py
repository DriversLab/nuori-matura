#!/usr/bin/env python
"""Regenerate synthetic matura items from data/wiki/articles.jsonl through an LLM API.

  python scripts/generate_synthetic.py --dry-run --shards historia-01                 # print the first prompt, no HTTP
  ANTHROPIC_API_KEY=... python scripts/generate_synthetic.py --shards historia-01 --max-articles 1
  python scripts/generate_synthetic.py --backend openai --base-url http://localhost:8000/v1 \\
      --model speakleash/Bielik-11B-v2.6-Instruct --shards biologia                    # all biologia shards via vLLM

Writes data/synthetic/raw/<shard>.<backend>.jsonl (existing files are skipped unless --force). These shards have no
blind-solver verification: build with  python scripts/build_data.py --skip-verify-glob "*.<backend>.jsonl"
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import logging
import os

from train import generate_synthetic as gen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(gen.BACKENDS), default="anthropic")
    ap.add_argument("--model", help=f"default: {gen.DEFAULT_MODELS}")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="openai backend: server base URL")
    ap.add_argument("--api-key-env", help="env var holding the API key (default: ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    ap.add_argument("--shards", help="comma-separated shard names or subjects (default: all shards)")
    ap.add_argument("--max-articles", type=int, help="articles per shard (cheap trial runs)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.7, help="openai backend only")
    ap.add_argument("--no-fallbacks", action="store_true", help="anthropic: do not request server-side refusal fallbacks")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out-dir", default=str(ROOT / "data/synthetic/raw"))
    ap.add_argument("--articles-path", default=str(ROOT / "data/wiki/articles.jsonl"))
    ap.add_argument("--force", action="store_true", help="overwrite existing <shard>.<backend>.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    shards = [s.strip() for s in args.shards.split(",") if s.strip()] if args.shards else None
    try:
        if args.dry_run:
            print(gen.first_prompt(args.backend, shards=shards, articles_path=args.articles_path))
            return 0
        key_env = args.api_key_env or ("ANTHROPIC_API_KEY" if args.backend == "anthropic" else "OPENAI_API_KEY")
        backend = gen.BackendConfig(
            name=args.backend, model=args.model or gen.DEFAULT_MODELS[args.backend], api_key=os.environ.get(key_env),
            base_url=args.base_url, max_tokens=args.max_tokens, temperature=args.temperature,
            fallbacks=not args.no_fallbacks, retries=args.retries, timeout=args.timeout,
        )
        if args.backend == "anthropic" and not backend.api_key:
            raise ValueError(f"{key_env} is not set")
        reports = gen.run(backend, shards=shards, max_articles=args.max_articles, out_dir=args.out_dir,
                          articles_path=args.articles_path, force=args.force)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"generate_synthetic failed: {exc}", file=sys.stderr)
        return 1
    n_items = sum(r.get("items", 0) for r in reports.values())
    for shard, r in reports.items():
        if "skipped" in r:
            print(f"{shard}: skipped ({r['skipped']})")
        else:
            print(f"{shard}: {r['items']}/{r['expected_items']} items, {len(r['failed_articles'])} failed articles, "
                  f"{len(r['errors'])} parse/validation errors, fixes {json.dumps(r['fixes'])}")
    print(f"total items written: {n_items}")
    generated = [r for r in reports.values() if "skipped" not in r]
    return 1 if generated and n_items == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
