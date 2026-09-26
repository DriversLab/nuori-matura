"""Item schema shared by the held-out eval set AND the synthetic training data.

One JSON object per line. Fields:

    id          str   unique, e.g. "hist-003" or "syn-biologia-00042"
    subject     str   one of SUBJECTS
    type        str   one of TYPES (see below)
    question    str   task text (Polish)
    context     str?  optional source text / passage shown above the task
    options     dict? {"A": "...", "B": "...", ...}      -> mc, multi
    statements  list? ["zdanie 1", "zdanie 2", ...]       -> tf
    left        list? ["element 1", "element 2", ...]     -> match (numbered 1..n)
    right       dict? {"A": "...", "B": "...", ...}      -> match (letters)
    answer      any   answer key, shape depends on type:
                        mc      "B"
                        multi   ["A", "C"]
                        tf      ["P", "F", "P"]           (one per statement)
                        match   {"1": "C", "2": "A"}      (one per left element)
                        short   ["Mieszko I", "Mieszko"]  (accepted answers; first = canonical)
                        numeric 3.5                        (float or int)
    tolerance   float? numeric only: absolute tolerance (default 1e-6 relative-ish, see grader)
    unit        str?  numeric only: unit the model may append (e.g. "m/s")
    points      int   max points for the item (default 1)
    partial_credit list? [[min_correct, points], ...] for tf/multi/match; default all-or-nothing
    rationale   str?  short justification (used only for "reason_then_answer" training renders)
    source      str   provenance: "authored", "wikipedia:<title>", ...
    difficulty  str?  easy | medium | hard
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

SUBJECTS = (
    "jezyk_polski",
    "historia",
    "wos",
    "geografia",
    "biologia",
    "chemia",
    "fizyka",
    "matematyka",
)

TYPES = ("mc", "multi", "tf", "match", "short", "numeric")

LETTERS = "ABCDEFGH"
TF_VALUES = ("P", "F")


class SchemaError(ValueError):
    pass


def _err(item: dict, msg: str) -> SchemaError:
    return SchemaError(f"[{item.get('id', '?')}] {msg}")


def validate_item(item: dict, *, require_rationale: bool = False) -> dict:
    """Validate one item in place (light normalisation) and return it. Raises SchemaError."""
    if not isinstance(item, dict):
        raise SchemaError(f"item is not an object: {item!r}")
    for key in ("id", "subject", "type", "question", "answer"):
        if key not in item or item[key] in (None, ""):
            raise _err(item, f"missing required field '{key}'")
    if not isinstance(item["id"], str):
        raise _err(item, "id must be a string")
    if item["subject"] not in SUBJECTS:
        raise _err(item, f"unknown subject '{item['subject']}' (allowed: {SUBJECTS})")
    t = item["type"]
    if t not in TYPES:
        raise _err(item, f"unknown type '{t}' (allowed: {TYPES})")
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise _err(item, "question must be a non-empty string")
    if "context" in item and item["context"] is not None and not isinstance(item["context"], str):
        raise _err(item, "context must be a string")

    item.setdefault("points", 1)
    if not isinstance(item["points"], int) or item["points"] < 1:
        raise _err(item, "points must be a positive int")

    ans = item["answer"]
    if t in ("mc", "multi"):
        opts = item.get("options")
        if not isinstance(opts, dict) or len(opts) < 2:
            raise _err(item, "options must be a dict with >= 2 entries")
        keys = list(opts.keys())
        if keys != list(LETTERS[: len(keys)]):
            raise _err(item, f"option keys must be consecutive letters from A, got {keys}")
        if any(not isinstance(v, str) or not v.strip() for v in opts.values()):
            raise _err(item, "option texts must be non-empty strings")
        if len({v.strip().casefold() for v in opts.values()}) != len(opts):
            raise _err(item, "duplicate option texts")
        if t == "mc":
            if not (isinstance(ans, str) and ans in opts):
                raise _err(item, f"mc answer must be one of {keys}, got {ans!r}")
        else:
            if not (isinstance(ans, list) and len(ans) >= 1 and all(a in opts for a in ans)):
                raise _err(item, f"multi answer must be a non-empty list of option letters, got {ans!r}")
            if len(set(ans)) != len(ans):
                raise _err(item, "multi answer has duplicates")
            item["answer"] = sorted(ans)
    elif t == "tf":
        st = item.get("statements")
        if not (isinstance(st, list) and len(st) >= 1 and all(isinstance(s, str) and s.strip() for s in st)):
            raise _err(item, "tf needs a non-empty list 'statements'")
        if not (isinstance(ans, list) and len(ans) == len(st) and all(a in TF_VALUES for a in ans)):
            raise _err(item, f"tf answer must be a list of P/F with len == len(statements), got {ans!r}")
    elif t == "match":
        left, right = item.get("left"), item.get("right")
        if not (isinstance(left, list) and len(left) >= 2):
            raise _err(item, "match needs 'left' list with >= 2 elements")
        if not (isinstance(right, dict) and len(right) >= 2):
            raise _err(item, "match needs 'right' dict with >= 2 entries")
        rkeys = list(right.keys())
        if rkeys != list(LETTERS[: len(rkeys)]):
            raise _err(item, f"right keys must be consecutive letters from A, got {rkeys}")
        expected = [str(i) for i in range(1, len(left) + 1)]
        if not (isinstance(ans, dict) and sorted(ans.keys(), key=int) == expected and all(v in right for v in ans.values())):
            raise _err(item, f"match answer must map {expected} -> letters of right, got {ans!r}")
    elif t == "short":
        if isinstance(ans, str):
            ans = [ans]
        if not (isinstance(ans, list) and ans and all(isinstance(a, str) and a.strip() for a in ans)):
            raise _err(item, "short answer must be a non-empty list of accepted strings")
        item["answer"] = ans
    elif t == "numeric":
        if isinstance(ans, bool) or not isinstance(ans, (int, float)):
            raise _err(item, f"numeric answer must be a number, got {ans!r}")
        tol = item.get("tolerance")
        if tol is not None and (not isinstance(tol, (int, float)) or tol < 0):
            raise _err(item, "tolerance must be a non-negative number")

    pc = item.get("partial_credit")
    if pc is not None:
        if t not in ("tf", "multi", "match"):
            raise _err(item, "partial_credit only allowed for tf/multi/match")
        if not (isinstance(pc, list) and all(isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) for x in r) for r in pc)):
            raise _err(item, "partial_credit must be [[min_correct, points], ...]")
        if any(p > item["points"] or p < 0 for _, p in pc):
            raise _err(item, "partial_credit points out of range")

    if require_rationale and not (isinstance(item.get("rationale"), str) and item["rationale"].strip()):
        raise _err(item, "rationale required")
    return item


def n_parts(item: dict) -> int:
    """Number of independently gradable parts (for partial credit)."""
    t = item["type"]
    if t == "tf":
        return len(item["statements"])
    if t == "match":
        return len(item["left"])
    if t == "multi":
        return len(item["answer"])
    return 1


def load_items(path: str | Path, *, validate: bool = True) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if validate:
                validate_item(item)
            if item["id"] in seen:
                raise SchemaError(f"{path}:{lineno}: duplicate id {item['id']}")
            seen.add(item["id"])
            items.append(item)
    return items


def write_jsonl(rows: Iterable[Any], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Any]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_WS = re.compile(r"\s+")


def item_text(item: dict) -> str:
    """Flattened question text used for dedup / similarity (question + context + options/statements)."""
    parts = [item.get("context") or "", item["question"]]
    if item.get("options"):
        parts.extend(item["options"].values())
    if item.get("statements"):
        parts.extend(item["statements"])
    if item.get("left"):
        parts.extend(item["left"])
    if item.get("right"):
        parts.extend(item["right"].values())
    return _WS.sub(" ", " ".join(p for p in parts if p)).strip()
