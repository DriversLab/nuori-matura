#!/usr/bin/env python
"""Merge two answers.json files of the same exam: every answer from --base, except the item ids in --ids, which are
taken from --override (e.g. the essay written by a second model).

  python scripts/merge_answers.py --base runs/tuned-11b/answers.json --override runs/essay-gemma/answers.json \
      --ids 26 --out runs/final/answers.json
  python scripts/validate_answers.py runs/final/answers.json --exam-dir exams/final

Both files must have the same exam_id and the overridden ids must have a non-empty answer in --override. The output
keeps the item order of --base and passes scripts/validate_answers.py (run it on --out, with --exam-dir). A
provenance file <out>.sources.json records which file each answer came from.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json


def load(path: str) -> dict:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "exam_id" not in data or not isinstance(data.get("answers"), list):
        raise SystemExit(f"{path}: not an answers.json ({{exam_id, answers: [...]}})")
    return data


def merge(base: dict, override: dict, ids: list[str]) -> tuple[dict, dict]:
    if base["exam_id"] != override["exam_id"]:
        raise SystemExit(f"exam_id differs: {base['exam_id']!r} vs {override['exam_id']!r}")
    over = {str(a["id"]): a["answer"] for a in override["answers"]}
    base_ids = [str(a["id"]) for a in base["answers"]]
    for i in ids:
        if i not in base_ids:
            raise SystemExit(f"id {i!r} is not in --base")
        if not str(over.get(i) or "").strip():
            raise SystemExit(f"id {i!r} has no answer in --override")
    out = {k: v for k, v in base.items() if k != "answers"}
    out["answers"] = [{"id": a["id"], "answer": over[str(a["id"])] if str(a["id"]) in ids else a["answer"]}
                      for a in base["answers"]]
    sources = {str(a["id"]): ("override" if str(a["id"]) in ids else "base") for a in base["answers"]}
    return out, sources


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--override", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated item ids taken from --override, e.g. 26")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    merged, sources = merge(load(args.base), load(args.override), ids)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".sources.json").write_text(json.dumps(
        {"base": args.base, "override": args.override, "ids_from_override": ids, "per_item": sources},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(merged['answers'])} answers, {len(ids)} from {args.override} ({', '.join(ids)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
