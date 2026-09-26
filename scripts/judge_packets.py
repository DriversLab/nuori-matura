#!/usr/bin/env python
"""Pair every exam answer with its question and its CKE marking-scheme block, for an LLM judge.

  python scripts/judge_packets.py --exam-dir exams/mock --rubric-text /path/zasady.txt \
      --answers runs/base/answers.json --out runs/judge/base.packets.jsonl
  python scripts/judge_packets.py --exam-dir exams/mock --rubric-text /path/zasady.txt \
      --benchmark-json /path/bielik-4-5b.json --out runs/judge/org-bielik.packets.jsonl   # organizers' format

The marking scheme (CKE "Zasady oceniania", pdftotext -layout) is split on its "Zadanie X.Y. (0–N)" headers. It is
copyrighted CKE material: keep the text file and the packets outside the repo (runs/ is gitignored). Judging only;
never train on the mock's marking scheme.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import re

from harness.exam_io import load_exam

HEADER = re.compile(r"^Zadanie (\d+(?:\.\d+)?)\. \(0[–-](\d+)\)", re.M)


def split_rubric(text: str) -> dict[str, dict]:
    marks = list(HEADER.finditer(text))
    blocks = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = re.sub(r"\n{3,}", "\n\n", text[m.start():end]).strip()
        blocks[m.group(1)] = {"max_points": int(m.group(2)), "rubric": body}
    return blocks


def load_answers(args) -> dict[str, str]:
    if args.answers:
        data = json.loads(pathlib.Path(args.answers).read_text(encoding="utf-8"))
        return {a["id"]: a["answer"] for a in data["answers"]}
    data = json.loads(pathlib.Path(args.benchmark_json).read_text(encoding="utf-8"))
    return {a["id"]: a["answer"] for a in data["answers"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam-dir", required=True)
    ap.add_argument("--rubric-text", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--answers", help="our answers.json")
    src.add_argument("--benchmark-json", help="organizers' benchmark run file (answers[].id/answer)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, items = load_exam(args.exam_dir)
    rubric = split_rubric(pathlib.Path(args.rubric_text).read_text(encoding="utf-8"))
    answers = load_answers(args)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    missing_rubric, n = [], 0
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            iid = item["id"]
            block = rubric.get(iid)
            if block is None:
                missing_rubric.append(iid)
            fh.write(json.dumps({
                "id": iid,
                "max_points": item.get("max_points"),
                "question": item["question"],
                "source_text": (item.get("source_text") or "")[:3000],
                "rubric": block["rubric"] if block else "",
                "answer": answers.get(iid),
                "answered": iid in answers,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} packets to {out}; answers for {sum(1 for i in items if i['id'] in answers)}/{n} items; "
          f"missing rubric blocks: {missing_rubric or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
