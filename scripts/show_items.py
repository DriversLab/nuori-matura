#!/usr/bin/env python
"""Print items exactly as the model sees them (canonical prompt), WITHOUT answer keys.
Used for blind verification of generated items and for eyeballing data.

  python scripts/show_items.py data/synthetic/raw/historia-003.jsonl [--variant canonical] [--with-keys]
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from eval.answer_format import canonical_payload
from eval.prompts import build_user_prompt
from eval.schema import load_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--variant", default="canonical")
    ap.add_argument("--with-keys", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    for item in load_items(args.path, validate=not args.no_validate):
        print(f"### {item['id']}  [{item['subject']} / {item['type']}]")
        print(build_user_prompt(item, args.variant))
        if args.with_keys:
            print(f">>> KEY: {canonical_payload(item)}")
            if item.get("rationale"):
                print(f">>> RATIONALE: {item['rationale']}")
        print()


if __name__ == "__main__":
    main()
