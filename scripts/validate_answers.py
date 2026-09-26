#!/usr/bin/env python
"""Check an answers.json against the upload site's rules for an exam package (exit 1 on any problem).

  python scripts/validate_answers.py runs/tuned/answers.json --exam-dir exams/mock
  python scripts/validate_answers.py runs/tuned/answers.json --exam-dir exams/mock --strict   # warnings fail too

Problems (the site would reject the file): exam_id, top-level/entry keys, id and answer types, missing, unexpected
or duplicate ids, 100,000-character answers, 1 MiB, UTF-8/JSON. Warnings (accepted but worth fixing): blank
answers, closed answers off the answer_format syntax, essay under 300 words or without a topic number, leftover
think/DRY-RUN markers.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from harness.exam_io import lint_answers, validate_answers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", help="answers.json to check")
    ap.add_argument("--exam-dir", required=True, help="exam package folder (exam.json, answers-template.json)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    args = ap.parse_args(argv)
    problems = validate_answers(args.answers, args.exam_dir)
    warnings = lint_answers(args.answers, args.exam_dir) if not problems else []
    for w in warnings:
        print(f"warning: {w}")
    if problems:
        print(f"INVALID: {len(problems)} problem(s) in {args.answers}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {args.answers} follows the upload rules" + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 1 if (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
