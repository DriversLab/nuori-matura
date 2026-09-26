#!/usr/bin/env python
"""Build the Polish-history matura SFT/dev data from the teammates' nuori-ai dump (docs/DATA.md).

  python scripts/build_history_data.py                                    # NUORI_DIR or ../nuori-ai -> data/processed/history/
  python scripts/build_history_data.py --nuori-dir /path/to/nuori-ai --mock-exam exams/mock/exam.json
  python scripts/build_history_data.py --include-old                      # + 2003-2014 papers (targets from --filled only)
  python scripts/build_history_data.py --filled a.jsonl --filled 'dir/*.jsonl'   # LLM-filled targets (default: see below)
  python scripts/build_history_data.py --synthetic 'data/history/synthetic/*.jsonl'  # synthetic items (default glob)
  python scripts/build_history_data.py --style rag --rag-index data/rag/<index>      # prompts with retrieved passages
  python scripts/build_history_data.py --stats-only                       # counts only, writes nothing

Reads <nuori-dir>/tasks.jsonl (never copied into the repo). Writes (gitignored, CKE-derived):
  data/processed/history/{train,dev,needs_answer,dropped}.jsonl, blocklist_ids.json, stats.json
  (+ dev_rag.jsonl with --style rag: the dev rows with retrieved passages; dev.jsonl stays organizer-style)
Prompts come from harness.prompts.build_messages(item, None, style=--style) - the exam-time renderer; keep --style equal
to run_exam.py --style (both default to "organizer").
Every exams/*/exam.json found (or each --mock-exam) is used as a leakage blocklist: near-duplicate rows are dropped.
Filled targets default to data/history/llm_filled_answers*.jsonl + data/history/filled_*/*.jsonl ("SKIP" = no answer);
synthetic items default to data/history/synthetic/*.jsonl. Missing files are fine (nothing is added).
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import logging

from train import history_data as H

DEFAULT_OUT = "data/processed/history"
DEFAULT_FILLED = ["data/history/llm_filled_answers*.jsonl", "data/history/filled_*/*.jsonl"]
DEFAULT_SYNTHETIC = ["data/history/synthetic/*.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuori-dir", default=None, help="teammates' nuori-ai checkout (default: $NUORI_DIR or ../nuori-ai)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--mock-exam", action="append", default=None,
                    help="exam.json to block (repeatable; default: every exams/*/exam.json)")
    ap.add_argument("--filled", action="append", default=None, metavar="PATH_OR_GLOB",
                    help="JSONL of LLM-filled answers ({id, answer}); repeatable, later files win. "
                         f"Default: {' + '.join(DEFAULT_FILLED)}")
    ap.add_argument("--no-filled", action="store_true", help="use no filled answers at all")
    ap.add_argument("--synthetic", action="append", default=None, metavar="PATH_OR_GLOB",
                    help=f"synthetic item JSONL (repeatable). Default: {' + '.join(DEFAULT_SYNTHETIC)}")
    ap.add_argument("--no-synthetic", action="store_true", help="use no synthetic items")
    ap.add_argument("--min-year", type=int, default=H.DEFAULT_MIN_YEAR,
                    help=f"oldest paper year to convert (default {H.DEFAULT_MIN_YEAR}); rows older than "
                         f"{H.DEFAULT_MIN_YEAR} are trained only with a filled target")
    ap.add_argument("--include-old", action="store_true",
                    help=f"convert the {H.OLD_MIN_YEAR}-{H.DEFAULT_MIN_YEAR - 1} papers too (= --min-year {H.OLD_MIN_YEAR})")
    ap.add_argument("--keyless-filled", choices=("none", "old", "all"), default="old",
                    help="train filled answers of rows whose paper has no answer key: for pre-2015 rows only (default), "
                         "all rows, or none (then they are dropped as no_answer_key)")
    ap.add_argument("--stats-only", action="store_true", help="print the counts, write nothing (no harness needed)")
    ap.add_argument("--style", choices=("organizer", "formatted", "rag"), default="organizer",
                    help="harness.prompts style of the training prompts; must equal run_exam.py --style at serve time "
                         "(default organizer = the organizers' benchmark protocol; rag = + retrieved passages)")
    ap.add_argument("--rag-index", default=None, help="harness.rag index path (required with --style rag)")
    ap.add_argument("--rag-k", type=int, default=4, help="passages per item (default 4)")
    ap.add_argument("--rag-fraction", type=float, default=0.85,
                    help="share of TRAIN rows rendered with passages; the rest get the rag prompt without passages "
                         "(default 0.85; dev_rag.jsonl always has passages)")
    ap.add_argument("--rag-max-chars", type=int, default=2400, help="format_passages max_chars (default 2400)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the --rag-fraction selection (default 0)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    nuori = pathlib.Path(args.nuori_dir) if args.nuori_dir else H.default_nuori_dir(ROOT)
    rows = H.load_nuori_tasks(nuori)
    exam_paths = args.mock_exam if args.mock_exam is not None else sorted(str(p) for p in (ROOT / "exams").glob("*/exam.json"))
    mock_items: list[dict] = []
    for p in exam_paths:
        mock_items += H.load_mock_items(p)
    min_year = min(args.min_year, H.OLD_MIN_YEAR) if args.include_old else args.min_year
    filled, filled_info = ({}, {"files": []}) if args.no_filled else H.load_filled_files(args.filled or DEFAULT_FILLED, ROOT)
    synthetic, syn_info = ([], {"files": []}) if args.no_synthetic else H.load_synthetic_files(
        args.synthetic or DEFAULT_SYNTHETIC, ROOT)
    rag = None
    if args.style == "rag" and not args.stats_only:
        if not args.rag_index:
            ap.error("--style rag needs --rag-index PATH (the harness.rag index)")
        idx = pathlib.Path(args.rag_index)
        rag = H.make_rag_context(idx if idx.is_absolute() else ROOT / idx, k=args.rag_k, fraction=args.rag_fraction,
                                 seed=args.seed, max_chars=args.rag_max_chars)
    out_dir = None if args.stats_only else (ROOT / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out))
    stats = H.build_history_data(rows, out_dir, min_year=min_year, mock_items=mock_items, filled_answers=filled,
                                 stats_only=args.stats_only, style=args.style, synthetic=synthetic, rag=rag,
                                 keyless_filled=args.keyless_filled)
    stats.pop("_kept", None)
    stats.pop("_dropped", None)
    stats["nuori_dir"] = str(nuori)
    stats["mock_exams"] = exam_paths
    stats["filled_files"] = filled_info
    stats["synthetic_files"] = syn_info
    if out_dir is not None:       # re-write stats.json with the input provenance added above
        (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    show = {k: stats[k] for k in ("input_rows", "split", "per_origin", "per_origin_kind", "needs_answer",
                                  "needs_answer_split", "needs_answer_origin", "filled", "synthetic", "drop", "flags",
                                  "per_kind", "per_formula", "blocklist", "item_kind_mismatch", "prompt_builder",
                                  "prompt_style", "rag", "min_year") if k in stats}
    show["filled_files"] = {k: v for k, v in filled_info.items() if k != "files"} | {"n_files": len(filled_info["files"])}
    show["synthetic_files"] = {k: v for k, v in syn_info.items() if k != "files"} | {"n_files": len(syn_info["files"])}
    print(json.dumps(show, ensure_ascii=False, indent=1))
    if out_dir is not None:
        print(f"[build_history_data] wrote {out_dir}: {stats.get('written')}")


if __name__ == "__main__":
    main()
