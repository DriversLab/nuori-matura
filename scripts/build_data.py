#!/usr/bin/env python
"""Clean the generated synthetic items: validate -> blind-solver verification -> dedup vs protected sets.

  python scripts/build_data.py                                   # data/synthetic/raw + verify -> clean.jsonl + dedup_report.json
  python scripts/build_data.py --skip-verify-glob "*.anthropic.jsonl"   # API-generated shards have no verify file
  python scripts/build_data.py --dry-run --thr-qa 0.85           # try thresholds without writing anything
  python scripts/build_data.py --extra-protected data/eval/new_set.jsonl   # a new eval set must be deduplicated too

Protected sets (never trained on; synthetic near-duplicates are dropped): data/eval/heldout.jsonl, data/eval/dev.jsonl,
data/eval/ext_llmzszl_matura.jsonl, every data/blocklist/*.jsonl and every --extra-protected file. Their sha256 is
recorded in the report (inputs.protected_files): train/dataset.build_processed refuses a clean.jsonl deduplicated
against different files. Fails when a raw shard's verify coverage is below --min-verify-coverage (verify workflow
unfinished?). Loads BAAI/bge-m3 (~2.3 GB) for the embedding check.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import datetime as dt
import fnmatch
import json
import logging
from collections import Counter, defaultdict

from eval.schema import SchemaError, read_jsonl, validate_item, write_jsonl
from train.config import ROOT, resolve_path
from train.dataset import file_sha256_16
from train.dedup import INTRA, DedupConfig, EmbedFn, dedup_items, reason_for
from train.synth_verify import verify_filter

N_EXAMPLES = 20
VERIFY_STAGES = {"unverified": "unverified", "solver_disagrees": "disagree", "low_confidence": "low_conf", "solver_concern": "concern"}


def load_raw(raw_dir: pathlib.Path) -> tuple[list[tuple[str, dict]], dict]:
    """Validated items as (file name, item) in sorted-file order; drops invalid lines and repeated ids."""
    items: list[tuple[str, dict]] = []
    seen: set[str] = set()
    counts: Counter = Counter()
    per_file: dict[str, dict] = {}
    invalid_examples: list[dict] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        file_counts: Counter = Counter()
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                file_counts["raw"] += 1
                try:
                    item = validate_item(json.loads(line))
                except (json.JSONDecodeError, SchemaError) as exc:
                    file_counts["invalid"] += 1
                    if len(invalid_examples) < N_EXAMPLES:
                        invalid_examples.append({"file": path.name, "line": lineno, "reason": str(exc)[:300]})
                    continue
                if item["id"] in seen:
                    file_counts["duplicate_id"] += 1
                    continue
                seen.add(item["id"])
                items.append((path.name, item))
        per_file[path.name] = dict(file_counts)
        counts.update(file_counts)
    return items, {"raw": counts["raw"], "invalid": counts["invalid"], "duplicate_id": counts["duplicate_id"],
                   "per_file": per_file, "invalid_examples": invalid_examples}


def load_answers(verify_dir: pathlib.Path) -> dict[str, dict]:
    answers: dict[str, dict] = {}
    if verify_dir.exists():
        for path in sorted(verify_dir.glob("*.jsonl")):
            for row in read_jsonl(path):
                if isinstance(row, dict) and row.get("id"):
                    answers.setdefault(row["id"], row)
    return answers


def load_protected(args: argparse.Namespace) -> tuple[dict[str, list[dict]], list[pathlib.Path]]:
    """Protected rows per set name, plus every protected source file (hashed into the report)."""
    protected: dict[str, list[dict]] = {}
    files: list[pathlib.Path] = []
    for name, rel in (("heldout", args.heldout), ("dev", args.dev), ("ext_llmzszl_matura", args.ext)):
        path = resolve_path(rel)
        if not path.exists():
            raise FileNotFoundError(f"protected set {name} not found: {path}")
        protected[name] = read_jsonl(path)
        files.append(path)
    block_files = sorted(resolve_path(args.blocklist_dir).glob("*.jsonl"))
    if not block_files:
        raise FileNotFoundError(f"no blocklist files in {args.blocklist_dir} (run: python scripts/fetch_external.py --only blocklist)")
    protected["blocklist"] = [row for path in block_files for row in read_jsonl(path)]
    files += block_files
    extra = [resolve_path(rel) for rel in args.extra_protected]
    missing = [str(p) for p in extra if not p.exists()]
    if missing:
        raise FileNotFoundError(f"--extra-protected files not found: {missing}")
    if extra:
        protected["extra"] = [row for path in extra for row in read_jsonl(path)]
        files += extra
    return protected, files


def _portable(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _skip_verify(file_name: str, args: argparse.Namespace) -> bool:
    return args.skip_verify or any(fnmatch.fnmatch(file_name, g) for g in args.skip_verify_glob)


def verify_coverage(raw_items: list[tuple[str, dict]], answers: dict[str, dict], args: argparse.Namespace) -> dict[str, dict]:
    """Per non-skipped raw shard: items, items with a solver answer, fraction answered (matched by item id)."""
    cov: dict[str, Counter] = defaultdict(Counter)
    for f, it in raw_items:
        if not _skip_verify(f, args):
            cov[f]["items"] += 1
            cov[f]["answered"] += it["id"] in answers
    return {f: {**c, "coverage": round(c["answered"] / c["items"], 4)} for f, c in sorted(cov.items())}


def build_clean(args: argparse.Namespace, embed_fn: EmbedFn | None = None) -> tuple[list[dict], dict]:
    raw_items, raw_stats = load_raw(resolve_path(args.raw_dir))
    if not raw_items:
        raise RuntimeError(f"no valid items in {args.raw_dir}/*.jsonl (raw lines: {raw_stats['raw']})")
    protected, protected_files = load_protected(args)

    answers = load_answers(resolve_path(args.verify_dir))
    coverage = verify_coverage(raw_items, answers, args)
    low = {f: c for f, c in coverage.items() if c["coverage"] < args.min_verify_coverage}
    if low and not args.allow_missing_verify:
        raise RuntimeError(
            "verify coverage below %.0f%% for %d shard(s): %s (verify workflow unfinished? use --skip-verify-glob or "
            "--allow-missing-verify)" % (100 * args.min_verify_coverage, len(low),
                                         ", ".join(f"{f} {c['answered']}/{c['items']}" for f, c in low.items()))
        )
    to_verify = [it for f, it in raw_items if not _skip_verify(f, args)]
    unverified_ok = [it for f, it in raw_items if _skip_verify(f, args)]
    verified, verify_report = verify_filter(to_verify, answers, drop_concerns=not args.keep_concerns)
    verified_ids = {it["id"] for it in verified} | {it["id"] for it in unverified_ok}
    candidates = [it for _, it in raw_items if it["id"] in verified_ids]  # keep raw (file) order

    cfg = DedupConfig(model=args.embed_model, thr_qa=args.thr_qa, thr_q=args.thr_q, thr_fuzzy=args.thr_fuzzy, thr_intra_qa=args.thr_intra_qa)
    kept, dedup_report = dedup_items(candidates, protected, cfg, embed_fn=embed_fn)

    stages = {"raw": raw_stats["raw"], "invalid": raw_stats["invalid"], "duplicate_id": raw_stats["duplicate_id"],
              "verify_skipped": len(unverified_ok)}
    stages.update({short: verify_report["counts"][reason] for reason, short in VERIFY_STAGES.items()})
    stages.update({reason_for(name): dedup_report["counts"][reason_for(name)] for name in protected})
    stages.update({INTRA: dedup_report["counts"][INTRA], "kept": len(kept)})

    per_subject_type: dict[str, Counter] = defaultdict(Counter)
    for it in kept:
        per_subject_type[it["subject"]][it["type"]] += 1
    report = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stages": stages,
        "kept_per_subject": {s: sum(c.values()) for s, c in sorted(per_subject_type.items())},
        "kept_per_type": dict(sorted(Counter(it["type"] for it in kept).items())),
        "kept_per_subject_type": {s: dict(sorted(c.items())) for s, c in sorted(per_subject_type.items())},
        "mc_answer_letters": dict(sorted(Counter(it["answer"] for it in kept if it["type"] == "mc").items())),
        "inputs": {
            "raw_dir": str(args.raw_dir), "verify_dir": str(args.verify_dir),
            "raw_files": raw_stats["per_file"],
            "verify_skipped_files": sorted({f for f, _ in raw_items if _skip_verify(f, args)}),
            "verify_coverage": coverage,
            "protected_sizes": {k: len(v) for k, v in protected.items()},
            "protected_files": {_portable(path): file_sha256_16(path) for path in protected_files},
        },
        "invalid_examples": raw_stats["invalid_examples"],
        "verify": verify_report,
        "dedup": dedup_report,
    }
    return kept, report


def format_summary(report: dict) -> str:
    st = report["stages"]
    drops = ", ".join(f"{k} {v}" for k, v in st.items() if k not in ("raw", "kept"))
    letters = report["mc_answer_letters"]
    total_mc = sum(letters.values()) or 1
    lines = [
        f"raw {st['raw']} -> kept {st['kept']}  ({drops})",
        "kept per subject: " + ", ".join(f"{s} {n}" for s, n in report["kept_per_subject"].items()),
        "kept per type:    " + ", ".join(f"{t} {n}" for t, n in report["kept_per_type"].items()),
        "mc answer letters: " + ", ".join(f"{k} {100 * v / total_mc:.0f}%" for k, v in letters.items()),
    ]
    partial = {f: c for f, c in report["inputs"]["verify_coverage"].items() if c["coverage"] < 1}
    if partial:
        lines.append("verify coverage < 100%: " + ", ".join(f"{f} {c['answered']}/{c['items']}" for f, c in partial.items()))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = DedupConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/synthetic/raw")
    ap.add_argument("--verify-dir", default="data/synthetic/verify")
    ap.add_argument("--out", default="data/synthetic/clean.jsonl")
    ap.add_argument("--report", default="data/synthetic/dedup_report.json")
    ap.add_argument("--heldout", default="data/eval/heldout.jsonl")
    ap.add_argument("--dev", default="data/eval/dev.jsonl")
    ap.add_argument("--ext", default="data/eval/ext_llmzszl_matura.jsonl")
    ap.add_argument("--blocklist-dir", default="data/blocklist")
    ap.add_argument("--extra-protected", action="append", default=[], metavar="PATH",
                    help="another eval/blocklist jsonl to deduplicate against (repeatable), e.g. a new eval set")
    ap.add_argument("--skip-verify", action="store_true", help="keep items without solver verification (all shards)")
    ap.add_argument("--skip-verify-glob", action="append", default=[], metavar="GLOB",
                    help="raw shard file names that bypass verification, e.g. '*.anthropic.jsonl' (repeatable)")
    ap.add_argument("--min-verify-coverage", type=float, default=0.9,
                    help="fail if a non-skipped raw shard has a smaller fraction of items with a solver answer")
    ap.add_argument("--allow-missing-verify", action="store_true",
                    help="proceed despite low verify coverage (the unverified items are dropped)")
    ap.add_argument("--keep-concerns", action="store_true",
                    help="keep items whose (agreeing) solver still flagged a concern; dropped by default")
    ap.add_argument("--embed-model", default=d.model)
    ap.add_argument("--thr-qa", type=float, default=d.thr_qa, help="cosine(stem+answer) drop threshold")
    ap.add_argument("--thr-q", type=float, default=d.thr_q, help="cosine(stem) drop threshold")
    ap.add_argument("--thr-fuzzy", type=float, default=d.thr_fuzzy, help="rapidfuzz token_set_ratio drop threshold")
    ap.add_argument("--thr-intra-qa", type=float, default=d.thr_intra_qa, help="cosine(stem+answer) between synthetic items")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, embed_fn: EmbedFn | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        kept, report = build_clean(args, embed_fn=embed_fn)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"build_data failed: {exc}", file=sys.stderr)
        return 1
    print(format_summary(report))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if not kept:
        print("build_data failed: no items survived (missing verify files? see --skip-verify-glob); nothing written", file=sys.stderr)
        return 1
    out, report_path = resolve_path(args.out), resolve_path(args.report)
    write_jsonl(kept, out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(kept)} items) and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
