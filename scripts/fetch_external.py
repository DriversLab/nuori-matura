#!/usr/bin/env python
"""Download external data: secondary eval, dedup blocklist, general instruction pool + general held-out set.

  python scripts/fetch_external.py                      # everything
  python scripts/fetch_external.py --only blocklist     # ext | blocklist | general (comma separated)

Outputs: data/eval/ext_llmzszl_matura.jsonl and data/blocklist/*.jsonl (CKE exam text: gitignored, hackathon rule),
data/blocklist/README.md, data/eval/general_heldout.jsonl, data/general/train_pool.jsonl (committed).
Provenance/counts: data/blocklist/README.md. On a fresh clone, run `--only ext,blocklist` before the old pipeline.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import datetime as dt
import json
import logging

from eval.schema import read_jsonl
from train import external

STAGES = ("ext", "blocklist", "general")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--general-pool-n", type=int, default=6000)
    ap.add_argument("--general-heldout-n", type=int, default=300)
    ap.add_argument("--ext-out", default="data/eval/ext_llmzszl_matura.jsonl")
    ap.add_argument("--blocklist-dir", default="data/blocklist")
    ap.add_argument("--heldout-out", default="data/eval/general_heldout.jsonl")
    ap.add_argument("--pool-out", default="data/general/train_pool.jsonl")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    stages = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown --only stages {sorted(unknown)}; choose from {STAGES}")

    blocklist_dir = ROOT / args.blocklist_dir
    today = dt.date.today().isoformat()
    summary: dict = {}
    failures: list[str] = []

    if "ext" in stages:
        try:
            stats: dict = {}
            summary["ext_llmzszl_matura"] = external.build_ext_llmzszl_matura(ROOT / args.ext_out, stats=stats)
            external.update_readme(blocklist_dir, {"ext_llmzszl_matura": {**stats, "fetched": today}})
        except Exception as exc:  # noqa: BLE001 - report and exit non-zero at the end
            failures.append(f"ext: {exc}")

    if "blocklist" in stages:
        counts = external.build_blocklist(blocklist_dir)
        summary["blocklist"] = counts
        missing = [s.name for s in external.BLOCK_SOURCES if s.name not in counts]
        if missing:
            summary["blocklist_failed"] = missing
        if not counts:
            failures.append("blocklist: every source failed")

    if "general" in stages:
        try:
            stats = {}
            summary["general_heldout"] = external.build_general_heldout(
                ROOT / args.heldout_out, args.general_heldout_n, args.seed, stats=stats)
            external.update_readme(blocklist_dir, {"general_heldout": {**stats, "fetched": today}})
            heldout_prompts = [r["messages"][0]["content"] for r in read_jsonl(ROOT / args.heldout_out)]
            stats = {}
            summary["general_pool"] = external.build_general_pool(
                ROOT / args.pool_out, args.general_pool_n, args.seed, exclude_prompts=heldout_prompts, stats=stats)
            external.update_readme(blocklist_dir, {"general_pool": {**stats, "fetched": today}})
        except Exception as exc:  # noqa: BLE001
            failures.append(f"general: {exc}")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
