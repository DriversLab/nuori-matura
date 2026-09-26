"""Raw model output -> final answer string for answers.json.

clean_answer(item, raw) -> str is the contract; clean(item, raw) -> CleanResult also reports what happened
(closed syntax recovered or not, essay words/topic). Rules:
  * think blocks, chat special tokens, markdown emphasis/headings, "Odpowiedź:" prefixes, preamble lines
    ("Odpowiedź na zadanie 3:", "Rozwiązanie zadania:") and trailing chatter are removed; text the model appended for
    another task ("Zadanie 4.2 ...") is cut
  * closed items: letters / P-F / numbers are recovered from messy text and re-emitted in the exact answer_format
    syntax (row count from the question). If any row is unrecoverable the cleaned text is kept instead (never "")
  * essay: the chosen topic is detected (explicit "temat 2" near the start, else topic-keyword overlap with a clear
    margin) and the answer starts with "Temat N." when it did not already; word count excludes that header
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from harness.prompts import CLOSED_KINDS, ESSAY_MIN_WORDS, FormatSpec, format_spec
from harness.prompts import answer_labels

_PL = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż"
_SPECIAL = re.compile(r"<\|im_(?:start|end)\|>(?:assistant|user|system)?\n?|</?s>|<\|endoftext\|>|<\|eot_id\|>")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_EMPH = re.compile(r"\*\*|__|(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])")
_HEADING = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*")
_QUOTE = re.compile(r"(?m)^[ \t]*>[ \t]?")
_LATEX_WRAP = re.compile(r"\\(?:boxed|text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}")
_PREAMBLE = re.compile(
    r"(?i)^(?:oto\s+)?(?:moja\s+|twoja\s+|ostateczna\s+|końcowa\s+)?"
    r"(?:odpowied[źz]|rozwiązanie|rozwiązania)"
    r"(?:\s+(?:na|do|dla)\s+(?:zadani\w*|pytani\w*|polecen\w*)(?:\s+(?:nr\.?\s*)?[\d.]+)?)?"
    r"(?:\s+(?:zadania|do\s+zadania)(?:\s+(?:nr\.?\s*)?[\d.]+)?)?"
    r"(?:\s*\([^)\n]{0,160}\))?\s*[:.]?\s*$"
)
_CONT_PREAMBLE = re.compile(
    r"(?i)^(?:oto\s+)?(?:dalsza\s+część|dalszy\s+ciąg|ciąg\s+dalszy|kontynuacja)(?:\s+[^\n:]{0,60})?\s*[:.]?\s*$"
)
_ANSWER_PREFIX = re.compile(r"(?i)^(?:ostateczna\s+|końcowa\s+|moja\s+)?odpowied[źz](?:\s+końcowa)?\s*[:\-–—]\s*")
_CHATTER = re.compile(
    r"(?i)^(?:mam nadzieję|jeśli (?:masz|potrzebujesz|chcesz)|daj (?:mi )?znać|w razie (?:pytań|wątpliwości)|"
    r"czy (?:mogę|chcesz)|powodzenia)"
)
_TASK_HEADER = re.compile(r"(?im)^[ \t]*(?:zadanie|zad\.)[ \t]+(\d+(?:\.\d+)*)\b")
_WORD = re.compile(r"[^\W_]+(?:['’\-–][^\W_]+)*")
_TOPIC_HEADER = re.compile(r"(?i)^\s*temat(?:u)?\s*(?:nr\.?|numer)?\s*[:\-–—]?\s*\(?\s*\d{1,2}\b")

_NEGATIVE_AFTER = re.compile(r"(?i)^[^\n]{0,25}?\b(?:jest\s+)?(?:błędn|niepoprawn|nieprawidłow|fałszyw|odpada|nie\s+jest)")


@dataclass
class CleanResult:
    answer: str
    kind: str
    closed_ok: bool | None = None          # None for open / essay
    essay_words: int | None = None
    essay_topic: int | None = None
    topic_source: str | None = None        # "explicit" | "restated" | "content" | None
    notes: list[str] = field(default_factory=list)


# ---- generic text cleaning -----------------------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks; text after the last </think> wins; a dangling <think> tag is dropped."""
    text = _THINK_BLOCK.sub("", text or "")
    if re.search(r"</think>", text, re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>", "", text, flags=re.IGNORECASE)
    return _SPECIAL.sub("", text)


def _unlatex(t: str) -> str:
    """\\boxed{\\text{P, F}} -> P, F ; drops \\[ \\] display delimiters (Bielik likes them)."""
    for _ in range(3):
        t2 = _LATEX_WRAP.sub(r"\1", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"(?m)^[ \t]*\\[\[\]][ \t]*$\n?", "", t)
    return t.replace("\\[", "").replace("\\]", "")


def _plain(text: str) -> str:
    """Markdown-free, whitespace-normalised text for pattern extraction."""
    t = strip_think(text).replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    t = _unlatex(t)
    t = _EMPH.sub("", t)
    t = _HEADING.sub("", t)
    t = _QUOTE.sub("", t)
    return t


def _cut_other_tasks(text: str, item_id: str) -> str:
    """Cut at a line that starts another task ("Zadanie 4.2 ..."), keeping at least some content before it."""
    for m in _TASK_HEADER.finditer(text):
        if m.group(1) != str(item_id) and text[:m.start()].strip():
            return text[:m.start()]
    return text


def clean_text(raw: str, item_id: str | None = None) -> str:
    """Open-answer cleaning: final text only, requested elements (labels, justification) untouched."""
    t = _plain(raw)
    lines = [ln.rstrip() for ln in t.split("\n")]
    # leading preambles / own task header
    while lines and (not lines[0].strip() or _PREAMBLE.match(lines[0].strip())
                     or _CONT_PREAMBLE.match(lines[0].strip())
                     or (item_id is not None and re.match(rf"(?i)^\s*zadanie\s+{re.escape(str(item_id))}\b"
                                                          rf"(?:\s*\([^)]*\))?\s*[:.]?\s*$", lines[0]))):
        lines.pop(0)
    t = "\n".join(lines)
    if item_id is not None:
        t = _cut_other_tasks(t, str(item_id))
    t = t.strip()
    t = _ANSWER_PREFIX.sub("", t, count=1)
    # trailing chatter
    lines = t.split("\n")
    while lines and (not lines[-1].strip() or _CHATTER.match(lines[-1].strip())):
        lines.pop()
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        t = re.sub(r"\n{3,}", "\n\n", strip_think(raw or "")).strip()
    return t


# ---- closed extraction ---------------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = s.casefold().replace("„", '"').replace("”", '"').replace("“", '"')
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _letter_re(allowed: list[str]) -> str:
    return "[" + "".join(sorted(set(allowed))) + "]"


def _option_line_letters(text: str, allowed: list[str]) -> set[str]:
    L = _letter_re(allowed)
    return set(re.findall(rf"(?m)^[ \t]*(?:[-*•][ \t]*)?\(?({L})[.)][ \t]+\S", text))


def extract_choice(text: str, allowed: list[str], options: Mapping[str, str] | None = None) -> str | None:
    """One option letter from free text, or None when ambiguous."""
    if not allowed:
        return None
    t = _plain(text).strip()
    if not t:
        return None
    L = _letter_re(allowed)
    after = rf"(?![{_PL}])"
    m = re.fullmatch(rf"[\s(\[]*({L})[\s.)\]:]*", t)
    if m:
        return m.group(1)
    # explicit markers: "Odpowiedź: C", "Poprawna odpowiedź to C", "Wybieram B"; skip "odpowiedź A jest błędna"
    marker = re.compile(
        r"(?i:(?:prawidłow\w*|poprawn\w*|właściw\w*|ostateczn\w*)\s+odpowied[źz]\w*|odpowied[źz]\w*|odp\.|"
        r"wybieram|zaznaczam|wybór)"
        r"(?:\s+(?i:to|jest|brzmi|na\s+(?:zadanie|pytanie)(?:\s+[\d.]+)?))?\s*[:\-–—=]?\s*(?:\n\s*)*"
        rf"[(\[]?(?i:(?:odpowied[źz]|opcja|wariant)\s+)?({L}){after}"
    )
    for mm in marker.finditer(t):
        if _NEGATIVE_AFTER.match(t[mm.end():]):
            continue
        return mm.group(1)
    restated = len(_option_line_letters(t, allowed)) >= 2
    first = next((ln.strip() for ln in t.split("\n") if ln.strip()), "")
    if not restated:
        m = re.match(rf"^(?:[-*•]\s*)?[(\[]?({L})(?:[.)\]:]|\s+[–—-]\s|\s*$)", first)
        if m:
            return m.group(1)
    if options:
        nt = _norm(t)
        hits = [k for k, v in options.items() if k in allowed and _norm(v) and _norm(v) in nt]
        if len(hits) == 1 and not restated:
            return hits[0]
    if not restated:
        found = {x for x in re.findall(rf"(?<![{_PL}\d])({L}){after}", t)}
        if len(found) == 1:
            return found.pop()
    return None


_VERDICT = re.compile(
    rf"(?<![{_PL}])(?:(nieprawdziw\w*|fałszyw\w*|fałsz|nieprawda)|(prawdziw\w*|prawda)|"
    rf"\(\s*([PF])\s*\)|([PF]))(?![{_PL}])",
    re.IGNORECASE,
)


def _verdict_tokens(text: str) -> list[tuple[int, str]]:
    out = []
    for m in _VERDICT.finditer(text):
        if m.group(1):
            out.append((m.start(), "F"))
        elif m.group(2):
            out.append((m.start(), "P"))
        else:
            tok = m.group(3) or m.group(4)
            if tok not in ("P", "F"):  # lower-case p/f are not verdicts
                continue
            out.append((m.start(), tok))
    return out


def _segments(text: str, keys: list[str], header_extra: str = "") -> dict[str, str]:
    """Split text into per-key segments at line-start headers "k." / "k:" / "k)" (in key order)."""
    starts: list[tuple[str, int, int]] = []
    pos = 0
    for k in keys:
        rx = re.compile(rf"(?m)^[ \t]*(?:[-*•|][ \t]*)?(?:{header_extra})?\(?{re.escape(k)}[ \t]*[.:)][ \t]*")
        m = rx.search(text, pos)
        if not m:
            continue
        starts.append((k, m.start(), m.end()))
        pos = m.end()
    out: dict[str, str] = {}
    for i, (k, st, body_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        out[k] = text[body_start:end]
    return out


def extract_tf(text: str, keys: list[str]) -> dict[str, str]:
    t = _plain(text)
    res: dict[str, str] = {}
    # 1) direct pairs "1: P", "1. Prawda", "1 – F", "1) fałsz" (last occurrence wins: final summaries come last)
    for k in keys:
        rx = re.compile(rf"(?<![\w.]){re.escape(k)}[ \t]*[:.)\-–—=][ \t]*(?:\(\s*)?"
                        rf"(nieprawdziw\w*|fałszyw\w*|fałsz|nieprawda|prawdziw\w*|prawda|[PF])(?![{_PL}])",
                        re.IGNORECASE)
        hits = [m.group(1) for m in rx.finditer(t) if m.group(1) not in ("p", "f")]
        if hits:
            tok = hits[-1]
            res[k] = tok if tok in ("P", "F") else ("F" if re.match(r"(?i)nie|fałsz", tok) else "P")
    # 2) per-statement segments: first verdict token inside the segment
    if len(res) < len(keys):
        segs = _segments(t, keys, header_extra=r"(?i:stwierdzenie|zdanie)\s+")
        for k, seg in segs.items():
            if k in res:
                continue
            toks = _verdict_tokens(seg)
            if toks:
                res[k] = toks[0][1]
    # 3) bare sequence "P, F, P" / "PFP" on one line
    if len(res) < len(keys):
        for ln in t.split("\n"):
            s = ln.strip().strip(".")
            if re.fullmatch(r"[PF](?:[\s,;/–\-]*[PF])*", s):
                seq = re.findall(r"[PF]", s)
                if len(seq) == len(keys):
                    for k, v in zip(keys, seq):
                        res.setdefault(k, v)
                    break
    return res


def extract_match(text: str, keys: list[str], rows: Mapping[str, str] | None = None) -> dict[str, str]:
    t = _plain(text)
    res: dict[str, str] = {}
    num = r"(?:(?i:fragment\w*|źródł\w*|tekst\w*|nr\.?|numer)\s*)?(\d{1,2})(?!\d)"
    # 1) direct pairs "A: 3", "A – fragment 3", "A → 3"
    for k in keys:
        rx = re.compile(rf"(?<![\w.]){re.escape(k)}[ \t]*(?:[:.)=\-–—>→]+|[ \t]+-+>?)[ \t]*{num}")
        hits = [m.group(1) for m in rx.finditer(t)]
        if hits:
            res[k] = hits[-1]
    # 2) lines that carry the row (its letter at line start, or its description, e.g. a markdown table row);
    #    the number is searched in the text after the description / letter
    if len(res) < len(keys):
        lines = t.split("\n")
        for k in keys:
            if k in res:
                continue
            desc = _norm((rows or {}).get(k, ""))
            probe = " ".join(desc.split()[:6])
            row_head = re.compile(rf"^[ \t|]*(?:[-*•][ \t]*)?{re.escape(k)}[ \t]*[.:)|]")
            for ln in lines:
                nln = _norm(ln)
                if desc and desc in nln:
                    rest = nln.split(desc, 1)[1]
                elif probe and probe in nln:
                    rest = nln.split(probe, 1)[1]
                elif row_head.match(ln):
                    rest = _norm(row_head.sub("", ln, count=1))
                    if desc and _norm(rest).startswith(probe):
                        rest = rest[len(desc):] if rest.startswith(desc) else rest[len(probe):]
                else:
                    continue
                m = re.search(r"(?:fragment\w*|źródł\w*|tekst\w*|nr|numer)\s*(\d{1,2})\b", rest) or \
                    re.search(r"(?<!\d)(\d{1,2})(?!\d)", rest)
                if m:
                    res[k] = m.group(1)
                    break
    return res


def extract_multi_part(text: str, spec: FormatSpec) -> dict[str, str]:
    t = _plain(text)
    res: dict[str, str] = {}
    allowed_all = sorted({x for k in spec.keys for x in spec.allowed.get(k, [])}) or ["A", "B", "C", "D"]
    # 1) compact answers: every non-empty line is "k: X" / "k. X" / "k – X" (or all on one line)
    L = _letter_re(allowed_all)
    pairs = re.findall(rf"(?<![\w.])(\d{{1,2}})[ \t]*[:.)\-–—=]?[ \t]*\(?({L})(?![{_PL}])", t)
    compact = re.fullmatch(rf"(?:[\s,;]*\d{{1,2}}[ \t]*[:.)\-–—=]?[ \t]*\(?{L}\)?\.?)+[\s,;.]*", t)
    if compact and pairs:
        for k, v in pairs:
            if k in spec.keys and v in spec.allowed.get(k, allowed_all):
                res[k] = v
        if len(res) == len(spec.keys):
            return res
        res = {}
    # 2) per-part segments, each solved like a single-choice answer
    segs = _segments(t, spec.keys, header_extra=r"(?i:część|zdanie|pytanie|punkt)\s+")
    for k, seg in segs.items():
        v = extract_choice(seg, spec.allowed.get(k, allowed_all), spec.options.get(k))
        if v:
            res[k] = v
    # 3) direct pairs anywhere for the rest ("1 – C"), unless the part's options were restated
    for k in spec.keys:
        if k in res:
            continue
        Lk = _letter_re(spec.allowed.get(k, allowed_all))
        hits = re.findall(rf"(?<![\w.]){re.escape(k)}[ \t]*[:)\-–—=][ \t]*\(?({Lk})(?![{_PL}])", t)
        if hits:
            res[k] = hits[-1]
    return res


def normalise_closed(item: Mapping[str, Any], raw: str, spec: FormatSpec | None = None) -> str | None:
    """Exact answer_format syntax, or None if some row cannot be recovered."""
    spec = spec or format_spec(item)
    if spec.kind == "closed_single":
        v = extract_choice(raw, spec.allowed[""], spec.options.get(""))
        return spec.render({"": v}) if v else None
    if spec.kind == "closed_tf":
        vals = extract_tf(raw, spec.keys)
    elif spec.kind == "closed_match":
        vals = extract_match(raw, spec.keys, spec.rows)
    elif spec.kind == "closed_multi_part":
        vals = extract_multi_part(raw, spec)
    else:
        return None
    if not spec.keys or any(k not in vals for k in spec.keys):
        return None
    return spec.render(vals)


def closed_syntax_ok(item: Mapping[str, Any], answer: str) -> bool:
    """True if answer already is the exact answer_format syntax (right keys, one row per key, allowed values)."""
    spec = format_spec(item)
    a = (answer or "").strip()
    if spec.kind == "closed_single":
        return a in spec.allowed[""]
    if spec.kind not in CLOSED_KINDS:
        return True
    lines = a.split("\n")
    if len(lines) != len(spec.keys):
        return False
    for k, ln in zip(spec.keys, lines):
        m = re.fullmatch(rf"{re.escape(k)}: (\S+)", ln)
        if not m:
            return False
        v = m.group(1)
        allowed = spec.allowed.get(k) or []
        if spec.kind == "closed_match":
            if not v.isdigit():
                return False
        elif v not in allowed:
            return False
    return True


# ---- essay ---------------------------------------------------------------------------------------------------------

def has_topic_header(text: str) -> bool:
    return bool(_TOPIC_HEADER.match(text or ""))


def essay_word_count(text: str, *, skip_topic_header: bool = True) -> int:
    """Words (letter/digit runs, hyphenated compounds count once). A leading "Temat N." line is not counted."""
    t = text or ""
    if skip_topic_header:
        t = re.sub(r"(?i)^\s*temat(?:u)?\s*(?:nr\.?|numer)?\s*[:\-–—]?\s*\(?\s*\d{1,2}\)?\s*[.:]?", "", t, count=1)
    return len(_WORD.findall(t))


_STOP = {
    "zajmij", "stanowisko", "wobec", "powyższej", "tezy", "uzasadnij", "uwzględniając", "swojej", "argumentacji",
    "wybranych", "wybrane", "charakteryzując", "trzech", "okresu", "wieku", "tego", "oraz", "który", "które",
    "aspekt", "przykładzie", "odwołując", "wydarzenia",
}


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(rf"[{_PL}]+", (text or "").casefold()) if len(w) >= 5 and w not in _STOP}


def detect_essay_topic(text: str, topics: Mapping[int, str] | None = None) -> tuple[int | None, str | None]:
    """(topic number, how it was found). First explicit choice near the start wins (benchmark rule)."""
    allowed = set(topics or {}) or {1, 2, 3}
    t = _plain(text)
    head = t[:600]
    rx = re.compile(r"(?i)(?:\btemat(?:u|em)?\b|\bwybieram\b|\bwybrany\b|\bwybrałem\b|\bwybrałam\b)"
                    r"(?:\s+temat\w*)?\s*(?:nr\.?|numer)?\s*[:\-–—]?\s*\(?\s*(\d{1,2})\b")
    for m in rx.finditer(head):
        if head[max(0, m.start() - 3):m.start()].casefold() == "na ":
            continue  # "na temat 3 wydarzeń"
        n = int(m.group(1))
        if n in allowed:
            return n, "explicit"
    first = next((ln.strip() for ln in t.split("\n") if ln.strip()), "")
    m = re.match(r"^(\d{1,2})[.)]\s+(.+)", first)
    if m and int(m.group(1)) in allowed and topics:
        topic_stems = _stems(topics.get(int(m.group(1)), ""))
        if topic_stems and len(topic_stems & _stems(m.group(2))) >= 2:
            return int(m.group(1)), "restated"
    if topics and len(topics) >= 2:
        essay = _stems(t)
        scores = sorted(((len(_stems(v) & essay), n) for n, v in topics.items()), reverse=True)
        best, second = scores[0], scores[1]
        if best[0] >= 3 and best[0] >= 2 * max(second[0], 1):
            return best[1], "content"
    return None, None


def clean_essay(item: Mapping[str, Any], raw: str, spec: FormatSpec | None = None) -> CleanResult:
    spec = spec or format_spec(item)
    topic, source = detect_essay_topic(raw, spec.topics)
    body = clean_text(raw, item.get("id"))
    notes: list[str] = []
    if topic is not None and not has_topic_header(body):
        # drop a first line that only restates the chosen topic ("2. Rewolucja amerykańska ...") after prefixing
        first, _, rest = body.partition("\n")
        if source == "restated" and re.match(rf"^{topic}[.)]\s", first):
            body = rest.strip()
        body = f"Temat {topic}.\n\n{body}"
    elif topic is None:
        notes.append("essay topic not detected; no topic number added")
    words = essay_word_count(body)
    if words < ESSAY_MIN_WORDS:
        notes.append(f"essay has {words} words (< {ESSAY_MIN_WORDS})")
    return CleanResult(answer=body, kind="essay", essay_words=words, essay_topic=topic, topic_source=source,
                       notes=notes)


# ---- entry points --------------------------------------------------------------------------------------------------

def clean(item: Mapping[str, Any], raw: str) -> CleanResult:
    spec = format_spec(item)
    raw = raw or ""
    if spec.kind == "essay":
        return clean_essay(item, raw, spec)
    if spec.kind in CLOSED_KINDS:
        exact = normalise_closed(item, raw, spec)
        if exact is not None:
            return CleanResult(answer=exact, kind=spec.kind, closed_ok=True)
        text = clean_text(raw, item.get("id"))
        return CleanResult(answer=text, kind=spec.kind, closed_ok=False,
                           notes=["closed answer not recoverable; kept cleaned text"])
    return CleanResult(answer=clean_text(raw, item.get("id")), kind=spec.kind)


def clean_answer(item: Mapping[str, Any], raw: str) -> str:
    """Final answer string for one item (see module docstring)."""
    return clean(item, raw).answer


# ---- run_exam extras: plan-mode essay topic, answer-label completeness, closed-answer voting ------------------------

def clean_essay_with_topic(item: Mapping[str, Any], raw: str, topic: int | None,
                           spec: FormatSpec | None = None) -> CleanResult:
    """clean_essay for --essay-mode plan: the topic chosen in the plan step wins over a content-based guess (or no
    guess). A topic the essay states itself ("Temat 2.", "Wybieram temat 2", a restated topic line) is kept."""
    spec = spec or format_spec(item)
    cr = clean_essay(item, raw, spec)
    if topic is None or cr.topic_source in ("explicit", "restated") or cr.essay_topic == topic:
        return cr
    if spec.topics and topic not in spec.topics:
        return cr
    body = clean_text(raw, item.get("id"))
    if has_topic_header(body):
        return cr
    answer = f"Temat {topic}.\n\n{body}"
    words = essay_word_count(answer)
    notes = [n for n in cr.notes if "not detected" not in n and not n.startswith("essay has ")]
    notes.append(f"topic {topic} taken from the essay plan")
    if words < ESSAY_MIN_WORDS:
        notes.append(f"essay has {words} words (< {ESSAY_MIN_WORDS})")
    return CleanResult(answer=answer, kind="essay", essay_words=words, essay_topic=topic, topic_source="plan",
                       notes=notes)


def label_present(label: str, text: str) -> bool:
    """True if `text` (already markdown-free) carries the answer label: "Uzasadnienie: ...", "Uzasadnienie – ...",
    "uzasadnienie:" or a line that is just "Uzasadnienie". "Opis 1.:" also matches "Opis 1: ..."."""
    core = label.strip().rstrip(":").strip().rstrip(".").strip()
    if not core:
        return True
    core_rx = r"[ \t]+".join(re.escape(w) for w in core.split())
    rx = re.compile(rf"(?im)(?<![{_PL}\d]){core_rx}\.?[ \t]*(?::|[-–—=]|$)")
    return bool(rx.search(text or ""))


def missing_labels(item: Mapping[str, Any], answer: str, labels: list[str] | None = None) -> list[str]:
    """Answer labels the question lists ("Rozstrzygnięcie:", "Uzasadnienie:", ...) that the answer lacks."""
    labels = answer_labels(item) if labels is None else labels
    t = _plain(answer or "")
    return [lab for lab in labels if not label_present(lab, t)]


ROW_VOTE_KINDS = frozenset({"closed_tf", "closed_multi_part"})


def closed_vote_units(item: Mapping[str, Any], raw: str, spec: FormatSpec | None = None) -> dict[str, str]:
    """One closed answer as votes. True/false and multi-part items vote per row (the rows are independent
    questions); single-choice and matching items vote with the whole normalised answer under key "" (matching rows
    are coupled). A partially recovered true/false or multi-part answer still votes for the rows it has."""
    spec = spec or format_spec(item)
    if spec.kind == "closed_tf":
        return {k: v for k, v in extract_tf(raw or "", spec.keys).items() if k in spec.keys}
    if spec.kind == "closed_multi_part":
        return {k: v for k, v in extract_multi_part(raw or "", spec).items() if k in spec.keys}
    if spec.kind in CLOSED_KINDS:
        norm = normalise_closed(item, raw or "", spec)
        return {"": norm} if norm is not None else {}
    raise ValueError(f"voting is only for closed items, not {spec.kind!r}")


def _vote_units(spec: FormatSpec) -> list[str]:
    return list(spec.keys) if spec.kind in ROW_VOTE_KINDS else [""]


@dataclass
class VoteResult:
    answer: str
    closed_ok: bool
    greedy_answer: str                     # what clean(item, greedy_raw) gives (the answer without voting)
    changed: bool                          # answer != greedy_answer
    rule: str                              # "unanimous" | "majority" | "tie->greedy" | "tie->first" | "fallback-greedy"
    tally: dict[str, dict[str, int]]       # unit ("" = whole answer, else row key) -> value -> votes
    rules: dict[str, str]                  # unit -> rule
    n_votes: int                           # greedy + samples


def vote_closed(item: Mapping[str, Any], greedy_raw: str, sample_raws: list[str]) -> VoteResult:
    """Majority over the normalised greedy answer and the sampled answers (per row for true/false and multi-part
    items). Ties go to the greedy answer's value; when the greedy answer has none for that unit, to the value seen
    first. If some row gets no valid vote at all, the greedy-only result is kept."""
    spec = format_spec(item)
    base = clean(item, greedy_raw)
    votes = [closed_vote_units(item, greedy_raw, spec)] + [closed_vote_units(item, r, spec) for r in sample_raws]
    greedy = votes[0]
    chosen: dict[str, str] = {}
    tally: dict[str, dict[str, int]] = {}
    rules: dict[str, str] = {}
    for u in _vote_units(spec):
        seq = [v[u] for v in votes if u in v]
        counts = Counter(seq)
        order = list(dict.fromkeys(seq))
        tally[u] = {v: counts[v] for v in order}
        if not counts:
            rules[u] = "no votes"
            continue
        top = max(counts.values())
        winners = [v for v in order if counts[v] == top]
        if len(winners) == 1:
            chosen[u] = winners[0]
            rules[u] = "unanimous" if len(counts) == 1 else "majority"
        elif greedy.get(u) is not None:
            chosen[u] = greedy[u]
            rules[u] = "tie->greedy"
        else:
            chosen[u] = winners[0]
            rules[u] = "tie->first"
    units = _vote_units(spec)
    if all(u in chosen for u in units):
        answer = chosen[""] if units == [""] else spec.render(chosen)
        ok = True
        rule = next((r for r in ("tie->first", "tie->greedy", "majority") if r in rules.values()), "unanimous")
    else:
        answer, ok, rule = base.answer, bool(base.closed_ok), "fallback-greedy"
    return VoteResult(answer=answer, closed_ok=ok, greedy_answer=base.answer, changed=answer != base.answer,
                      rule=rule, tally=tally, rules=rules, n_votes=len(votes))


def vote_decided(item: Mapping[str, Any], greedy_raw: str, sample_raws: list[str], remaining: int) -> bool:
    """True when `remaining` further votes cannot change vote_closed's answer (lets run_exam stop sampling early;
    the answer is then identical to the one all samples would give)."""
    if remaining <= 0:
        return True
    spec = format_spec(item)
    votes = [closed_vote_units(item, greedy_raw, spec)] + [closed_vote_units(item, r, spec) for r in sample_raws]
    greedy = votes[0]
    for u in _vote_units(spec):
        ranked = Counter(v[u] for v in votes if u in v).most_common()
        if not ranked:
            return False
        leader, c1 = ranked[0]
        c2 = ranked[1][1] if len(ranked) > 1 else 0
        if c1 > c2 + remaining:
            continue
        if c1 == c2 + remaining and c1 > c2 and greedy.get(u) == leader:
            continue  # the worst case is a tie, and ties go to the greedy value = the leader
        return False
    return True
