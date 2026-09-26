#!/usr/bin/env python
"""Fetch Polish Wikipedia articles listed in data/wiki/titles.yaml -> data/wiki/articles.jsonl.

titles.yaml format:  {subject: [title, title, ...], ...}   (subjects from eval/schema.py SUBJECTS)
Idempotent: articles already present in the output are skipped unless --refresh.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import datetime as dt
import json

import yaml

from eval.schema import SUBJECTS, read_jsonl, write_jsonl
from train.wiki import clean_extract, fetch_extract, resolve_titles, truncate_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--titles", default="data/wiki/titles.yaml")
    ap.add_argument("--out", default="data/wiki/articles.jsonl")
    ap.add_argument("--max-chars", type=int, default=40000, help="truncate very long articles (generation reads a prefix anyway)")
    ap.add_argument("--min-chars", type=int, default=800, help="skip stubs shorter than this (after math-markup cleanup)")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    titles_by_subject = yaml.safe_load(open(ROOT / args.titles, encoding="utf-8"))
    unknown = set(titles_by_subject) - set(SUBJECTS)
    if unknown:
        sys.exit(f"unknown subjects in {args.titles}: {unknown}")

    out_path = ROOT / args.out
    existing = {} if args.refresh or not out_path.exists() else {a["title"]: a for a in read_jsonl(out_path)}
    articles: dict[str, dict] = {}
    report = {"requested": 0, "resolved": 0, "missing": [], "stubs": [], "duplicates": [], "subject_changed": [],
              "pruned": [], "fetched": 0, "cached": 0}
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    claimed: dict[str, str] = {}  # canonical title -> subject that claimed it in THIS run (first listing wins)

    for subject, titles in titles_by_subject.items():
        titles = [str(t) for t in (titles or []) if t is not None]
        report["requested"] += len(titles)
        resolved = resolve_titles(titles)
        report["resolved"] += len(resolved)
        report["missing"] += [f"{subject}:{t}" for t in titles if t.strip() not in resolved]
        for orig, final in resolved.items():
            if final in claimed:
                report["duplicates"].append(f"{subject}:{orig}->{final} (already in {claimed[final]})")
                continue
            claimed[final] = subject
            cached = existing.get(final)
            if cached is not None:
                if len(cached["text"]) < args.min_chars:
                    report["stubs"].append(f"{subject}:{final} ({len(cached['text'])} chars, cached)")
                    continue
                if cached["subject"] != subject:
                    report["subject_changed"].append(f"{final}: {cached['subject']} -> {subject}")
                articles[final] = {**cached, "subject": subject}
                report["cached"] += 1
                continue
            page = fetch_extract(final)
            if not page:
                report["missing"].append(f"{subject}:{orig}")
                continue
            text = clean_extract(page["extract"])
            if len(text) < args.min_chars:
                report["stubs"].append(f"{subject}:{final} ({len(text)} chars)")
                continue
            articles[final] = {
                "title": final, "subject": subject, "pageid": page["pageid"], "revid": page["revid"],
                "url": page["url"], "fetched_at": now, "n_chars_full": len(text),
                "text": truncate_text(text, args.max_chars),
            }
            report["fetched"] += 1
            print(f"[{subject}] {final}: {len(text)} chars", flush=True)

    report["pruned"] = sorted(set(existing) - set(articles) - {s.split(":", 1)[1].split(" (")[0] for s in report["stubs"]})
    for key in ("subject_changed", "pruned"):
        if report[key]:
            print(f"warning: {key}: {report[key]}", file=sys.stderr)

    rows = sorted(articles.values(), key=lambda a: (SUBJECTS.index(a["subject"]), a["title"]))
    write_jsonl(rows, out_path)
    report["total_articles"] = len(rows)
    report["per_subject"] = {s: sum(1 for a in rows if a["subject"] == s) for s in SUBJECTS}
    (out_path.parent / "fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
