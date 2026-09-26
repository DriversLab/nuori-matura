#!/usr/bin/env python
"""Totals and per-item comparison of rubric-judge results.

  python scripts/judge_report.py runs/judge/base.judge.json runs/judge/tuned.judge.json
  python scripts/judge_report.py org-bielik.judge.json --vs bielik-4-5b.reviews.json      # judge calibration

Accepted inputs: the judge workflow result ({"label", "total", "scores": [{"id", "points", "max_points", "reason"}]},
optionally wrapped as {"result": {...}} in a task output file) or the organizers' reviews list
([{"id", "points", "reason", ...}]). An item a file does not grade counts as 0 points (the organizers' reviews omit
unanswered items). The first file is the reference column; deltas are against it.
"""
import argparse
import json
import pathlib
import sys


def _id_key(item_id: str):
    return tuple(int(p) if p.isdigit() else p for p in str(item_id).split("."))


def load_scores(path: str) -> tuple[str, dict[str, dict]]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
        data = data["result"]
    if isinstance(data, dict):
        label, rows = data.get("label") or pathlib.Path(path).stem, data.get("scores") or []
    elif isinstance(data, list):
        label, rows = pathlib.Path(path).stem, data
    else:
        raise SystemExit(f"{path}: unrecognised judge file")
    return label, {str(r["id"]): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="judge results; the first is the reference column")
    ap.add_argument("--vs", help="organizers' reviews (or another judge file) to check agreement with the first file")
    ap.add_argument("--reasons", action="store_true", help="print both reasons for every disagreeing item")
    ap.add_argument("--packets", help="judge_packets.py output: authoritative item list and max_points")
    args = ap.parse_args()

    runs = [load_scores(f) for f in args.files]
    ids = sorted({i for _, s in runs for i in s}, key=_id_key)
    max_pts = {i: next((s[i].get("max_points") for _, s in runs if i in s and s[i].get("max_points") is not None),
                       None) for i in ids}
    if args.packets:
        packets = [json.loads(line) for line in pathlib.Path(args.packets).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        max_pts = {str(p["id"]): p.get("max_points") for p in packets}
        ids = sorted(max_pts, key=_id_key)
    pts = lambda s, i: float(s[i]["points"]) if i in s else 0.0  # noqa: E731

    labels = [lab for lab, _ in runs]
    print("item   max  " + "  ".join(f"{lab[:14]:>14}" for lab in labels))
    for i in ids:
        cells = []
        for k, (_, s) in enumerate(runs):
            p = pts(s, i)
            cell = f"{p:g}"
            if k and p != pts(runs[0][1], i):
                cell += f" ({p - pts(runs[0][1], i):+g})"
            cells.append(f"{cell:>14}")
        print(f"{i:<6} {max_pts[i] if max_pts[i] is not None else '?':>3}  " + "  ".join(cells))
    total_max = sum(m for m in max_pts.values() if m)
    totals = [sum(pts(s, i) for i in ids) for _, s in runs]
    print(f"{'total':<6} {total_max:>3g}  " + "  ".join(f"{t:>14g}" for t in totals))
    for lab, t in zip(labels[1:], totals[1:]):
        print(f"delta {lab} vs {labels[0]}: {t - totals[0]:+g}")

    if args.vs:
        ref_label, ref = load_scores(args.vs)
        _, ours = runs[0]
        both = sorted(set(ours) | set(ref), key=_id_key)
        agree = [i for i in both if pts(ours, i) == pts(ref, i)]
        print(f"\nagreement {labels[0]} vs {ref_label}: {len(agree)}/{len(both)} items exact; "
              f"totals {sum(pts(ours, i) for i in both):g} vs {sum(pts(ref, i) for i in both):g}")
        for i in both:
            if pts(ours, i) != pts(ref, i):
                print(f"  {i}: {pts(ours, i):g} vs {pts(ref, i):g}")
                if args.reasons:
                    print(f"    ours: {ours.get(i, {}).get('reason', '-')}\n    ref:  {ref.get(i, {}).get('reason', '-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
