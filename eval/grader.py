"""Auto-grader: raw model output -> points (strict = official-like default; lenient = diagnostic)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from eval.answer_format import (
    DEFAULT_VARIANT,
    Parse,
    numeric_close,
    parse_lenient,
    parse_strict,
    short_match,
)
from eval.schema import n_parts


def count_correct_parts(item: dict, value: Any) -> int:
    t, gold = item["type"], item["answer"]
    if t == "tf":
        return sum(1 for p, g in zip(value, gold) if p == g)
    if t == "match":
        return sum(1 for k, g in gold.items() if value.get(k) == g)
    if t == "multi":
        pred, gold_s = set(value), set(gold)
        wrong = len(pred - gold_s)
        return max(0, len(pred & gold_s) - wrong)
    raise ValueError(t)


def points_for(item: dict, parse: Parse, *, lenient: bool) -> float:
    if not parse.ok:
        return 0.0
    t, gold, pts = item["type"], item["answer"], item.get("points", 1)
    v = parse.value
    if t == "mc":
        return float(pts) if v == gold else 0.0
    if t == "short":
        return float(pts) if short_match(v, gold, lenient=lenient) else 0.0
    if t == "numeric":
        return float(pts) if numeric_close(v, float(gold), item.get("tolerance")) else 0.0
    # tf / multi / match
    if t == "multi" and sorted(v) == sorted(gold):
        return float(pts)
    n_ok = count_correct_parts(item, v)
    if t in ("tf", "match") and n_ok == n_parts(item):
        return float(pts)
    table = item.get("partial_credit") or []
    best = 0
    for min_correct, p in table:
        if n_ok >= min_correct:
            best = max(best, p)
    return float(best)


def grade(item: dict, output: str, variant: str = DEFAULT_VARIANT) -> dict:
    """Grade one output. Returns a flat record written to predictions.jsonl."""
    ps = parse_strict(item, output, variant)
    pl = parse_lenient(item, output)
    max_pts = float(item.get("points", 1))
    pts_strict = points_for(item, ps, lenient=False)
    pts_lenient = max(points_for(item, pl, lenient=True), pts_strict)
    return {
        "id": item["id"],
        "subject": item["subject"],
        "type": item["type"],
        "variant": variant,
        "max_points": max_pts,
        "points": pts_strict,
        "correct": pts_strict == max_pts,
        "parse_ok": ps.ok,
        "parse_reason": ps.reason,
        "parsed": ps.value,
        "points_lenient": pts_lenient,
        "parse_ok_lenient": pl.ok,
        "parsed_lenient": pl.value,
        "output": output,
    }


def _bucket() -> dict:
    return {"n": 0, "points": 0.0, "max_points": 0.0, "parse_fail": 0, "points_lenient": 0.0}


def summarize(records: list[dict]) -> dict:
    """Aggregate graded records (single variant) into headline + per-subject + per-type numbers."""
    total = _bucket()
    by_subject: dict[str, dict] = defaultdict(_bucket)
    by_type: dict[str, dict] = defaultdict(_bucket)
    reasons: dict[str, int] = defaultdict(int)
    for r in records:
        for b in (total, by_subject[r["subject"]], by_type[r["type"]]):
            b["n"] += 1
            b["points"] += r["points"]
            b["max_points"] += r["max_points"]
            b["points_lenient"] += r["points_lenient"]
            b["parse_fail"] += 0 if r["parse_ok"] else 1
        if not r["parse_ok"]:
            reasons[r["parse_reason"] or "unknown"] += 1

    def finish(b: dict) -> dict:
        n, mx = b["n"], b["max_points"]
        return {
            **b,
            "score_pct": round(100.0 * b["points"] / mx, 2) if mx else 0.0,
            "lenient_score_pct": round(100.0 * b["points_lenient"] / mx, 2) if mx else 0.0,
            "parse_fail_rate": round(b["parse_fail"] / n, 4) if n else 0.0,
            "format_loss_points": round(b["points_lenient"] - b["points"], 2),
        }

    return {
        **finish(total),
        "per_subject": {k: finish(v) for k, v in sorted(by_subject.items())},
        "per_type": {k: finish(v) for k, v in sorted(by_type.items())},
        "parse_fail_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
