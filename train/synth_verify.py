"""Keep only synthetic items whose answer key an independent open-book solver reproduced.

Verify rows (data/synthetic/verify/<shard>.jsonl): {"id", "payload", "confidence": high|medium|low, "concern"}.
The solver payload is graded with the evaluation grader itself: it agrees when it gets full points under strict
payload_only parsing, or under lenient parsing when the payload names exactly ONE candidate answer. The lenient parser
was built to read model output (first letter / first number / substring wins), so a hedged payload such as "B lub C",
"3,5 albo 7" or "nie Mieszko I, lecz Bolesław Chrobry" would otherwise count as confirming a doubtful key.
"""
from __future__ import annotations

import re
from typing import Any

from eval.answer_format import canonical_payload, format_number, normalize_short, short_match
from eval.grader import grade

REASONS = ("unverified", "solver_disagrees", "low_confidence", "solver_concern")
N_EXAMPLES = 20

_WORD_EDGE = r"(?<![A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż])({})(?![A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż])"
_LETTER = re.compile(_WORD_EDGE.format("[A-H]"))
_TF_TOKEN = re.compile(_WORD_EDGE.format("P|F|prawda|fałsz|prawdziwe|fałszywe"), re.IGNORECASE)
_MATCH_PAIR = re.compile(r"(\d+)\s*[-–—:→)\.]*\s*([A-H])(?![a-ząćęłńóśźż])")
_NUMBER = re.compile(r"[-−]?(?:\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,]\d+)?")
_HEDGE = re.compile(r"(?<!\w)(lub|albo|bądź|czy|ewentualnie|względnie|lecz|ale|nie|zamiast|or)(?!\w)", re.IGNORECASE)
_PREFIX = re.compile(r"^\s*(?:\*\*|__|`)?\s*odpowied[źz]\s*:\s*", re.IGNORECASE)
_SINGLE_ANSWER_TYPES = ("mc", "short", "numeric")


def payload_text(payload: Any) -> str:
    """Solver payload -> answer string; tolerates structured payloads (letters list, match dict, numbers)."""
    if payload is None:
        return ""
    if isinstance(payload, bool):
        return str(payload)
    if isinstance(payload, (int, float)):
        return format_number(payload)
    if isinstance(payload, list):
        return ", ".join(payload_text(p) for p in payload)
    if isinstance(payload, dict):
        return ", ".join(f"{k}-{v}" for k, v in sorted(payload.items(), key=lambda kv: (len(str(kv[0])), str(kv[0]))))
    return str(payload).strip()


def _clean(text: str) -> str:
    text = re.sub(r"\*\*|__|`+", "", _PREFIX.sub("", text)).strip()
    return text.rstrip(" .;")


def _single_answer(item: dict, text: str) -> bool:
    """True when the payload names exactly one candidate answer (lenient parsing would pick the first/last one otherwise)."""
    t = item["type"]
    if t == "mc":
        return {x for x in _LETTER.findall(text) if x in item["options"]} <= {item["answer"]}
    if t == "numeric":
        unit = item.get("unit")
        return len(_NUMBER.findall(text.replace(unit, " ") if unit else text)) == 1
    if t == "tf":
        return len(_TF_TOKEN.findall(text)) == len(item["statements"])
    if t == "match":
        keys = [k for k, _ in _MATCH_PAIR.findall(text)]
        return len(keys) == len(set(keys)) == len(item["left"])
    if t == "short":
        key_words = {w for a in item["answer"] for w in normalize_short(a).split()}
        if any(h.casefold() not in key_words for h in _HEDGE.findall(text)):
            return False
        parts = [p for p in re.split(r"[,;/|]", text) if p.strip()]
        return all(short_match(p, item["answer"], lenient=True) for p in parts)
    return True  # multi: the lenient letter set must already equal the key exactly


def _agrees_text(item: dict, text: str) -> tuple[bool, dict]:
    text = _clean(text)
    if item["type"] in ("tf", "multi") and re.fullmatch(r"[A-HPF]{2,}", text):
        text = ", ".join(text)  # compact "PFP" / "AC" -> "P, F, P" / "A, C"
    rec = grade(item, text, "payload_only")
    if rec["points"] >= rec["max_points"]:
        return True, rec
    return rec["points_lenient"] >= rec["max_points"] and _single_answer(item, text), rec


def solver_agrees(item: dict, payload: Any) -> tuple[bool, dict]:
    """Full points under strict payload_only parsing, or under lenient parsing of a payload naming exactly one answer.
    A list payload for a single-answer type (mc/short/numeric) agrees only if every element does."""
    if item["type"] in _SINGLE_ANSWER_TYPES and isinstance(payload, list) and len(payload) > 1:
        results = [_agrees_text(item, payload_text(p)) for p in payload]
        return all(ok for ok, _ in results), next((rec for ok, rec in results if not ok), results[0][1])
    return _agrees_text(item, payload_text(payload))


def verify_filter(items: list[dict], answers: dict[str, dict], *, drop_concerns: bool = True) -> tuple[list[dict], dict]:
    """Drop items without a solver answer, with a disagreeing solver answer, with confidence == "low", or (drop_concerns)
    with a non-empty solver concern: an agreeing solver that still flags ambiguity / a debatable statement / missing
    accepted variants marks an item whose key could teach the model a wrong or contested answer."""
    kept: list[dict] = []
    dropped: dict[str, list[str]] = {r: [] for r in REASONS}
    examples: dict[str, list[dict]] = {r: [] for r in REASONS}

    def drop(reason: str, item: dict, **extra: Any) -> None:
        dropped[reason].append(item["id"])
        if len(examples[reason]) < N_EXAMPLES:
            examples[reason].append({"id": item["id"], "subject": item["subject"], "type": item["type"],
                                     "question": item["question"], **extra})

    for item in items:
        ans = answers.get(item["id"])
        if ans is None:
            drop("unverified", item)
            continue
        agrees, rec = solver_agrees(item, ans.get("payload"))
        confidence = str(ans.get("confidence") or "").strip().casefold()
        if not agrees:
            drop("solver_disagrees", item, solver_payload=ans.get("payload"), key=canonical_payload(item),
                 solver_points=rec["points_lenient"], max_points=rec["max_points"], concern=ans.get("concern"))
        elif confidence == "low":
            drop("low_confidence", item, solver_payload=ans.get("payload"), key=canonical_payload(item), concern=ans.get("concern"))
        elif drop_concerns and str(ans.get("concern") or "").strip():
            drop("solver_concern", item, solver_payload=ans.get("payload"), key=canonical_payload(item), concern=ans.get("concern"))
        else:
            kept.append(item)

    report = {
        "n_in": len(items),
        "n_kept": len(kept),
        "counts": {r: len(v) for r, v in dropped.items()},
        "examples": examples,
        "dropped_ids": dropped,
    }
    return kept, report
