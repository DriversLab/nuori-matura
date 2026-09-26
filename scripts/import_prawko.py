#!/usr/bin/env python
"""Import the organizers' prawko-v2 driving-licence questions as organizer-only eval sets.

  python scripts/import_prawko.py                               # download, write dev + test
  python scripts/import_prawko.py --from-file /path/data.json   # offline (no network)
  python scripts/import_prawko.py --splits test                 # one split only

Source rows: {"id", "question", "options": [3 strings], "answer": int index, "points": int, ...}. We import their
held-out dev/test splits only (never `train`: it is their training data and must not become ours) and convert them to
our item schema, keeping the option texts verbatim - the organizers' grader renders them exactly as they are:

    id       "prawko-<their id>"      subject  "wos" (no matura subject covers driving law; see `source`)
    type     "mc"                     options  {"A": ..., "B": ..., "C": ...}   answer  the letter of their int index
    points   1 (their harness is unweighted: accuracy over rows, so their per-row points are dropped)
    source   "prawko-v2:<split>"

Outputs: data/eval/ext_prawko_dev.jsonl (25), data/eval/ext_prawko_test.jsonl (40). They are gitignored (third-party
rows, licence not verified), so they are never in the repo: re-create them with this script after a fresh clone. They
are protected eval files (train/dataset.py) and configs/base.yaml scores them under the organizers' protocol only
(variants: []; optional: true, so a missing file is skipped, not an error).
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import collections
import json

from eval.schema import validate_item, write_jsonl

DATA_URL = "https://raw.githubusercontent.com/stared/train-llm-from-scratch/main/datasets/prawko-v2/data.json"
LETTERS = "ABC"
SUBJECT = "wos"  # closest matura subject; the real provenance lives in `source`
SPLITS = ("dev", "test")
EXPECTED_COUNTS = {"dev": 25, "test": 40}


def fetch_data(url: str = DATA_URL, from_file: str | pathlib.Path | None = None, *, timeout: int = 60) -> dict:
    """The organizers' data.json as {split: [row, ...]} (downloaded, or read from a local copy)."""
    if from_file:
        return json.loads(pathlib.Path(from_file).read_text(encoding="utf-8"))
    import requests  # lazy: --from-file works without network deps

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def convert_row(row: dict, split: str) -> dict:
    """One prawko-v2 row -> one validated item (option texts verbatim, answer index -> letter)."""
    options = row.get("options")
    if not isinstance(options, list) or len(options) != len(LETTERS):
        raise ValueError(f"prawko row {row.get('id')!r}: expected {len(LETTERS)} options, got {options!r}")
    index = row.get("answer")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(LETTERS):
        raise ValueError(f"prawko row {row.get('id')!r}: answer must be an index 0..{len(LETTERS) - 1}, got {index!r}")
    return validate_item({
        "id": f"prawko-{row['id']}",
        "subject": SUBJECT,
        "type": "mc",
        "question": row["question"],
        "options": {letter: text for letter, text in zip(LETTERS, options)},
        "answer": LETTERS[index],
        "points": 1,  # their harness is unweighted; row["points"] is reported but not imported
        "source": f"prawko-v2:{split}",
    })


def convert_split(rows: list[dict], split: str) -> list[dict]:
    """All rows of one split, in source order. Raises ValueError on a duplicate id."""
    items = [convert_row(row, split) for row in rows]
    duplicates = [i for i, n in collections.Counter(it["id"] for it in items).items() if n > 1]
    if duplicates:
        raise ValueError(f"prawko split {split!r}: duplicate ids {duplicates}")
    return items


def answer_counts(items: list[dict]) -> dict[str, int]:
    """{letter: n} over the answer keys, in letter order (their manifest reports the same distribution)."""
    counts = collections.Counter(it["answer"] for it in items)
    return {letter: counts.get(letter, 0) for letter in LETTERS}


def source_points(rows: list[dict]) -> dict[str, int]:
    """{their points value: n rows} - reported only: their grader scores accuracy over rows, so we import points 1."""
    counts = collections.Counter(str(row.get("points")) for row in rows)
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DATA_URL, help="source data.json (default: the organizers' repo)")
    ap.add_argument("--from-file", help="read this local data.json instead of downloading (offline)")
    ap.add_argument("--out-dir", default="data/eval", help="directory for ext_prawko_<split>.jsonl (default: data/eval)")
    ap.add_argument("--splits", default=",".join(SPLITS), help=f"comma-separated subset of {SPLITS}")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLITS]
    if unknown:
        ap.error(f"unknown --splits {unknown}; choose from {SPLITS} (their `train` split is theirs, never ours)")

    data = fetch_data(args.url, args.from_file)
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    for split in splits:
        rows = data.get(split)
        if not rows:
            print(f"FAILED: split {split!r} missing or empty in {args.from_file or args.url}", file=sys.stderr)
            return 1
        items = convert_split(rows, split)
        out_path = out_dir / f"ext_prawko_{split}.jsonl"
        n = write_jsonl(items, out_path)
        expected = EXPECTED_COUNTS.get(split)
        warn = "" if expected in (None, n) else f"   WARNING: expected {expected} rows"
        letters = " ".join(f"{k}={v}" for k, v in answer_counts(items).items())
        print(f"{out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}: {n} items   "
              f"answers {letters}{warn}")
        print(f"  source points (their per-row weights, unused by their harness -> imported as points 1): "
              f"{source_points(rows)}")
    print("These files are protected eval sets: rerun scripts/build_data.py with "
          "--extra-protected data/eval/ext_prawko_dev.jsonl --extra-protected data/eval/ext_prawko_test.jsonl "
          "before the next training run (train/dataset.py refuses a clean.jsonl deduplicated against other eval files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
