"""THE answer-format contract. Single source of truth for:

  * the format instruction appended to prompts (per prompt variant),
  * the exact target completion used in training data,
  * the STRICT parser (default grading) and the LENIENT parser (diagnostic).

Training renders and eval parsing both import from here, so they can never drift apart.
Change ANSWER_PREFIX / instructions here (and bump FORMAT_VERSION) if organizers publish a different format.

Canonical output (what the fine-tuned model is trained to emit):

    Odpowiedź: B                  mc
    Odpowiedź: A, C               multi (sorted letters)
    Odpowiedź: P, F, P            tf (one per statement, in order)
    Odpowiedź: 1-C, 2-A, 3-B      match (sorted by number)
    Odpowiedź: Mieszko I          short
    Odpowiedź: 3,5                numeric (Polish decimal comma; dot also accepted)
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

FORMAT_VERSION = "v1"
ANSWER_PREFIX = "Odpowiedź:"

# ----------------------------------------------------------------------------- prompt variants


@dataclass(frozen=True)
class Variant:
    name: str
    expects: str  # "line" -> an "Odpowiedź: ..." line; "payload" -> the whole response is the payload
    require_last: bool = False  # answer line must be the last non-empty line
    description: str = ""


VARIANTS: dict[str, Variant] = {
    "canonical": Variant("canonical", "line", False, "explicit instruction to answer as 'Odpowiedź: X'"),
    "bare": Variant("bare", "line", False, "no format instruction at all; model must default to 'Odpowiedź: X'"),
    "payload_only": Variant("payload_only", "payload", False, "instruction to output only the bare answer (e.g. just 'B')"),
    "reason_then_answer": Variant("reason_then_answer", "line", True, "short rationale, then final line 'Odpowiedź: X'"),
    "organizer_letter": Variant("organizer_letter", "payload", False,
                                "organizers' constrained-letter wording; the whole response is the bare answer"),
}
DEFAULT_VARIANT = "canonical"

# Wording of the organizers' reference grader (workshop `scripts/prawko.py`), which never reads generated text: it scores
# the logits of the single-token letters at the FIRST answer position. Their literal first sentence names the driving
# exam ("Rozwiąż pytanie egzaminacyjne na prawo jazdy w Polsce."), which is wrong for matura items, so the subject is
# dropped here and the rest is verbatim. eval/organizers.py keeps their literal sentence for measuring under their
# protocol; this one is the training-side twin, used only by the `organizer_letter` variant (mc items). eval/prompts.py
# still appends it after the task body (their prompt puts it first), so it trains the wording, not their exact layout.
ORGANIZER_LETTER_INSTRUCTION = (
    "Rozwiąż pytanie egzaminacyjne. Wybierz jedną poprawną odpowiedź. Odpowiedz wyłącznie literą {letters}."
)

_TYPE_HINT = {
    "mc": "Wybierz jedną poprawną odpowiedź.",
    "multi": "Wybierz wszystkie poprawne odpowiedzi.",
    "tf": "Oceń, czy każde ze zdań jest prawdziwe (P), czy fałszywe (F).",
    "match": "Przyporządkuj każdemu elementowi oznaczonemu cyfrą właściwy element oznaczony literą.",
    "short": "Udziel krótkiej odpowiedzi (słowo, nazwa lub krótkie wyrażenie).",
    "numeric": "Podaj sam wynik liczbowy.",
}


def placeholder(item: dict) -> str:
    t = item["type"]
    if t == "mc":
        return "<litera>"
    if t == "multi":
        return "<litery oddzielone przecinkami>"
    if t == "tf":
        return ", ".join(["<P/F>"] * len(item["statements"]))
    if t == "match":
        return ", ".join(f"{i}-<litera>" for i in range(1, len(item["left"]) + 1))
    if t == "short":
        return "<odpowiedź>"
    if t == "numeric":
        return "<liczba>"
    raise ValueError(t)


def letters_phrase(letters: Any) -> str:
    """['A', 'B', 'C'] -> 'A, B albo C' (the organizers' enumeration; same wording as eval/organizers.letter_list)."""
    ls = list(letters)
    if not ls:
        raise ValueError("no letters")
    return ls[0] if len(ls) == 1 else f"{', '.join(ls[:-1])} albo {ls[-1]}"


def format_instruction(item: dict, variant: str = DEFAULT_VARIANT) -> str:
    """Instruction text appended after the task. Empty string for the 'bare' variant."""
    hint = _TYPE_HINT[item["type"]]
    ph = placeholder(item)
    if variant == "canonical":
        return f"{hint} Odpowiedz wyłącznie w formacie:\n{ANSWER_PREFIX} {ph}"
    if variant == "bare":
        return ""
    if variant == "payload_only":
        return f"{hint} Podaj wyłącznie odpowiedź w postaci: {ph} — bez żadnego dodatkowego tekstu."
    if variant == "reason_then_answer":
        return (
            f"{hint} Najpierw krótko uzasadnij odpowiedź (1–3 zdania), "
            f"a w ostatniej linii napisz:\n{ANSWER_PREFIX} {ph}"
        )
    if variant == "organizer_letter":
        # Only mc items have the letters the organizers' wording enumerates; every other type falls back to
        # payload_only, which asks for the same bare-payload answer, so all item types still render.
        if item["type"] == "mc":
            return ORGANIZER_LETTER_INSTRUCTION.format(letters=letters_phrase(item["options"]))
        return format_instruction(item, "payload_only")
    raise KeyError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")


# ----------------------------------------------------------------------------- canonical payload / target


def format_number(x: float) -> str:
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e15):
        return str(int(x))
    s = repr(float(x))
    if "e" in s or "E" in s:
        s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def canonical_payload(item: dict) -> str:
    t, a = item["type"], item["answer"]
    if t == "mc":
        return a
    if t == "multi":
        return ", ".join(sorted(a))
    if t == "tf":
        return ", ".join(a)
    if t == "match":
        return ", ".join(f"{k}-{a[k]}" for k in sorted(a, key=int))
    if t == "short":
        return a[0] if isinstance(a, list) else a
    if t == "numeric":
        return item.get("answer_display") or format_number(a)
    raise ValueError(t)


def render_target(item: dict, variant: str = DEFAULT_VARIANT) -> str:
    """Exact assistant completion the model is trained to produce for this item/variant."""
    payload = canonical_payload(item)
    if variant in ("canonical", "bare"):
        return f"{ANSWER_PREFIX} {payload}"
    if variant in ("payload_only", "organizer_letter"):
        return payload
    if variant == "reason_then_answer":
        rationale = (item.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(f"[{item.get('id')}] reason_then_answer needs a rationale")
        return f"{rationale}\n{ANSWER_PREFIX} {payload}"
    raise KeyError(variant)


# ----------------------------------------------------------------------------- parsing


@dataclass
class Parse:
    ok: bool
    value: Any = None
    reason: str = ""  # failure reason code when not ok
    payload: str | None = None  # raw payload string that was interpreted
    notes: list[str] = field(default_factory=list)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return s.replace("\r\n", "\n").replace("\r", "\n")


def normalize_short(s: str) -> str:
    """Normalisation for short-answer comparison: casefold, unify quotes/dashes, strip edge punctuation."""
    s = normalize_text(s).casefold()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"[\"'„”“«»`]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .;:!?,")
    return s


_NUM_STRICT = re.compile(r"^[-−]?(?:\d{1,3}(?: \d{3})+|\d+)(?:[.,]\d+)?$")
_NUM_ANY = re.compile(r"[-−]?(?:\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,]\d+)?")


def _to_float(tok: str) -> float:
    tok = tok.replace("−", "-").replace(" ", "").replace(" ", "").replace(",", ".")
    return float(tok)


def _payload_strict(item: dict, payload: str) -> Parse:
    t = item["type"]
    p = payload.strip()
    if not p:
        return Parse(False, reason="empty_payload", payload=payload)
    if "\n" in p:
        return Parse(False, reason="multiline_payload", payload=payload)
    if t == "mc":
        if re.fullmatch(r"[A-H]", p) and p in item["options"]:
            return Parse(True, p, payload=payload)
        return Parse(False, reason="bad_mc_payload", payload=payload)
    if t == "multi":
        if not re.fullmatch(r"[A-H](?:\s*,\s*[A-H])*", p):
            return Parse(False, reason="bad_multi_payload", payload=payload)
        letters = [x.strip() for x in p.split(",")]
        if len(set(letters)) != len(letters) or any(x not in item["options"] for x in letters):
            return Parse(False, reason="bad_multi_letters", payload=payload)
        return Parse(True, sorted(letters), payload=payload)
    if t == "tf":
        if not re.fullmatch(r"[PF](?:\s*,\s*[PF])*", p):
            return Parse(False, reason="bad_tf_payload", payload=payload)
        vals = [x.strip() for x in p.split(",")]
        if len(vals) != len(item["statements"]):
            return Parse(False, reason="tf_wrong_count", payload=payload)
        return Parse(True, vals, payload=payload)
    if t == "match":
        if not re.fullmatch(r"\d+\s*-\s*[A-H](?:\s*,\s*\d+\s*-\s*[A-H])*", p):
            return Parse(False, reason="bad_match_payload", payload=payload)
        pairs = [tuple(x.strip() for x in part.split("-")) for part in p.split(",")]
        keys = [k for k, _ in pairs]
        expected = [str(i) for i in range(1, len(item["left"]) + 1)]
        if sorted(keys, key=int) != expected or any(v not in item["right"] for _, v in pairs):
            return Parse(False, reason="match_wrong_keys", payload=payload)
        return Parse(True, {k: v for k, v in pairs}, payload=payload)
    if t == "short":
        if len(p) > 200:
            return Parse(False, reason="short_too_long", payload=payload)
        return Parse(True, p, payload=payload)
    if t == "numeric":
        num = p
        unit = item.get("unit")
        if unit and num.endswith(unit):
            num = num[: -len(unit)].strip()
        if not _NUM_STRICT.fullmatch(num):
            return Parse(False, reason="bad_numeric_payload", payload=payload)
        return Parse(True, _to_float(num), payload=payload)
    raise ValueError(t)


_STRICT_LINE = re.compile(r"^[ \t]*" + re.escape(ANSWER_PREFIX) + r"[ \t]*(.*?)[ \t]*$", re.MULTILINE)


def parse_strict(item: dict, text: str, variant: str = DEFAULT_VARIANT) -> Parse:
    """Strict parsing. 'line' variants: exactly one distinct answer among lines that start with
    'Odpowiedź:' (exact spelling/case, no markdown), payload must match the canonical shape for the type.
    'payload' variants (payload_only, organizer_letter): the entire stripped response must be a valid payload."""
    v = VARIANTS[variant]
    text = normalize_text(text)
    if not text.strip():
        return Parse(False, reason="empty_output")
    if v.expects == "payload":
        return _payload_strict(item, text.strip())
    matches = list(_STRICT_LINE.finditer(text))
    if not matches:
        return Parse(False, reason="no_answer_line")
    parses = [_payload_strict(item, m.group(1)) for m in matches]
    if any(not pr.ok for pr in parses):
        bad = next(pr for pr in parses if not pr.ok)
        return bad
    distinct = {repr(pr.value) if not isinstance(pr.value, str) else pr.value.casefold() for pr in parses}
    if len(distinct) > 1:
        return Parse(False, reason="conflicting_answer_lines", payload=" | ".join(m.group(1) for m in matches))
    if v.require_last:
        last_nonempty = [ln for ln in text.split("\n") if ln.strip()][-1]
        if not _STRICT_LINE.fullmatch(last_nonempty):
            return Parse(False, reason="answer_not_last_line", payload=parses[-1].payload)
    return parses[-1]


# ---- lenient ---------------------------------------------------------------------------------

_MD = re.compile(r"(\*\*|__|`+|^#+\s*)", re.MULTILINE)
_MARKER_HEAD = r"(?:(?:prawidłow|poprawn)\w*\s+)?odpowied[źz]\w*\s*(?:to|jest)?\s*[:\-–—=]?"
# A payload ends at the end of its line or right before the next marker head followed by "to"/"jest"/":", so
# "... odpowiedź to A. Odpowiedź: C" yields both markers (the last one wins) while a short answer that merely contains
# the word ("zasada odpowiedzialności zbiorowej") stays whole.
_LENIENT_MARKER = re.compile(
    r"(?<!\w)" + _MARKER_HEAD
    + r"\s*(.+?)(?=[ \t]*(?<!\w)(?:(?:prawidłow|poprawn)\w*\s+)?odpowied[źz]\w*\s*(?:to\b|jest\b|:)|$)",
    re.IGNORECASE | re.MULTILINE,
)
_POLISH_LETTER = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż"


def _strip_md(text: str) -> str:
    return _MD.sub("", text)


def _lenient_mc(item: dict, s: str) -> Any:
    opts = item["options"]
    m = re.match(r"^\s*[\(\[]?([A-H])(?:[\)\]\.:,]|\s|$)", s) or re.match(r"^\s*[\(\[]?([a-h])[\)\]]", s)
    if m and m.group(1).upper() in opts:
        return m.group(1).upper()
    norm = normalize_short(s)
    hits = [k for k, v in opts.items() if normalize_short(v) and normalize_short(v) in norm]
    if len(hits) == 1:
        return hits[0]
    return None


def _lenient_letters(s: str, allowed: dict) -> list[str]:
    compact = re.fullmatch(r"\s*([A-H]{2,})[\s.]*", s)  # CKE-style "BE"
    if compact and len(set(compact.group(1))) == len(compact.group(1)) and set(compact.group(1)) <= set(allowed):
        return list(compact.group(1))
    found = re.findall(rf"(?<![{_POLISH_LETTER}])([A-H])(?![{_POLISH_LETTER}])", s)
    out: list[str] = []
    for x in found:
        if x in allowed and x not in out:
            out.append(x)
    return out


def _lenient_tf(item: dict, s: str) -> Any:
    n = len(item["statements"])
    toks = re.findall(rf"(?<![{_POLISH_LETTER}])(P|F|prawda|fałsz|prawdziwe|fałszywe)(?![{_POLISH_LETTER}])", s, re.IGNORECASE)
    mapped = []
    for tok in toks:
        low = tok.casefold()
        if low in ("p", "prawda", "prawdziwe"):
            mapped.append("P")
        elif low in ("f", "fałsz", "fałszywe"):
            mapped.append("F")
    if len(mapped) >= n:
        return mapped[-n:] if len(mapped) > n else mapped
    compact = re.findall(rf"(?<![{_POLISH_LETTER}])([PF]{{{n}}})(?![{_POLISH_LETTER}])", s)  # CKE-style "FPP"
    return list(compact[-1]) if compact else None


def _lenient_match(item: dict, s: str) -> Any:
    pairs = re.findall(r"(\d+)\s*[-–—:→)\.]*\s*([A-H])(?![a-ząćęłńóśźż])", s)
    out: dict[str, str] = {}
    for k, v in pairs:
        if v in item["right"] and k not in out:
            out[k] = v
    expected = {str(i) for i in range(1, len(item["left"]) + 1)}
    return out if set(out) == expected else None


def _lenient_numeric(s: str) -> Any:
    toks = _NUM_ANY.findall(s)
    if not toks:
        return None
    try:
        return _to_float(toks[0])
    except ValueError:
        return None


def _lenient_from_payload(item: dict, s: str) -> Any:
    t = item["type"]
    if t == "mc":
        return _lenient_mc(item, s)
    if t == "multi":
        letters = _lenient_letters(s, item["options"])
        return sorted(letters) if letters else None
    if t == "tf":
        return _lenient_tf(item, s)
    if t == "match":
        return _lenient_match(item, s)
    if t == "short":
        first = s.strip().split("\n")[0].strip()
        return first if first else None
    if t == "numeric":
        return _lenient_numeric(s)
    raise ValueError(t)


def parse_lenient(item: dict, text: str) -> Parse:
    """Best-effort extraction of the intended answer from free-form output. Diagnostic only:
    strict - lenient gap == points lost purely to answer formatting."""
    text = _strip_md(normalize_text(text)).strip()
    if not text:
        return Parse(False, reason="empty_output")
    # 1) answer marker (last one wins)
    markers = list(_LENIENT_MARKER.finditer(text))
    for m in reversed(markers):
        val = _lenient_from_payload(item, m.group(1))
        if val is not None:
            return Parse(True, val, payload=m.group(1), notes=["marker"])
    t = item["type"]
    # 2) whole response is (nearly) the payload
    first_line = text.split("\n")[0]
    if len(text) <= 80:
        val = _lenient_from_payload(item, text)
        if val is not None:
            return Parse(True, val, payload=text, notes=["short_response"])
    # 3) type specific fallbacks over the whole text
    if t == "mc":
        line_letters = {m.group(1).upper() for m in re.finditer(r"^\s*[\(\[]?([A-Ha-h])[\)\]\.:]\s", text, re.MULTILINE)}
        line_letters &= set(item["options"])
        if len(line_letters) == 1:
            return Parse(True, line_letters.pop(), notes=["single_letter_line"])
        val = _lenient_mc(item, first_line)
        if val is not None:
            return Parse(True, val, payload=first_line, notes=["first_line"])
        return Parse(False, reason="lenient_no_mc")
    if t == "numeric":
        toks = _NUM_ANY.findall(text)
        if toks:
            try:
                return Parse(True, _to_float(toks[-1]), payload=toks[-1], notes=["last_number"])
            except ValueError:
                pass
        return Parse(False, reason="lenient_no_number")
    if t == "short":
        return Parse(True, first_line.strip(), payload=first_line, notes=["first_line"])
    val = _lenient_from_payload(item, text)
    if val is not None:
        return Parse(True, val, payload=text, notes=["whole_text"])
    return Parse(False, reason=f"lenient_no_{t}")


# ----------------------------------------------------------------------------- comparison helpers


def numeric_close(pred: float, gold: float, tolerance: float | None) -> bool:
    if tolerance is None:
        tolerance = 1e-6 * max(1.0, abs(gold))
    return math.isfinite(pred) and abs(pred - gold) <= tolerance + 1e-12


def short_match(pred: str, accepted: list[str], *, lenient: bool = False) -> bool:
    p = normalize_short(pred)
    acc = [normalize_short(a) for a in accepted]
    if p in acc:
        return True
    if lenient:
        for a in acc:
            if a and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", p) and len(p) <= 3 * len(a) + 20:
                return True
    return False
