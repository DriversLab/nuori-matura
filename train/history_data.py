"""Polish-history matura SFT/dev data built from the teammates' nuori-ai dump (CKE arkusze + zasady oceniania).

Input: ``<nuori_dir>/tasks.jsonl`` (one row per "Zadanie", produced by nuori-ai/parse.py from CKE PDFs) and,
optionally, the organizers' mock ``exam.json`` (used only as a leakage blocklist). Nothing CKE-derived is written into
the repo: outputs go to the gitignored ``data/processed/history/``:

    train.jsonl         TRL prompt-completion rows, papers <= 2024 (minus the mock paper and its sibling), LLM-filled
                        targets, 2003-2014 papers with filled targets (--include-old), synthetic items
    dev.jsonl           same shape, 2025-2026 papers (our judge's dev set; rows keep the CKE rubric text)
    dev_rag.jsonl       style="rag" only: the dev rows with retrieved passages (dev.jsonl stays organizer-style)
    needs_answer.jsonl  rows whose key has no usable model answer (rubric-only keys, essays, tables, old papers) -> LLM fill queue
    dropped.jsonl       {id, reason} for every row / synthetic item that was not emitted (audit trail)
    blocklist_ids.json  the mock paper (MHIP-R0-100-2305) and sibling (EHIP-R0-100-2305): papers, row ids, fingerprints
    stats.json          counts per split / year / kind / origin / drop reason / flag; prompt style and RAG settings

Every row is first converted to the organizers' exam item shape ({id, group, max_points, question, source_text, images,
answer_format}) and its prompt is rendered with ``harness.prompts.build_messages(item, None, style="organizer")`` - the
same function that renders the exam at inference (train/serve consistency; ``style`` follows run_exam.py --style and is
recorded in stats.json as prompt_style; style="rag" adds harness.rag passages to a seeded share of the train rows).
Targets are exemplary final answers:
closed items in the exact answer_format syntax (``B`` / ``1: P\\n2: F`` / ``A: 3\\nB: 1`` / ``1: A\\n2: C``), open
items in Polish with every element the question asks for, derived from the key's model answer (never its rubric).
Every row has an ``origin``: cke-2015+ | llm-filled | cke-old | synthetic-open | synthetic-essay.
"""
from __future__ import annotations

import difflib
import glob
import hashlib
import inspect
import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- constants

# The mock exam is the full May 2023 CKE paper MHIP-R0-100-2305 (formula 2023). EHIP-R0-100-2305 is its formula-2015
# sibling from the same session (shared sources). Both - and any other copy of them in the dump - never enter any split.
MOCK_PAPER_CODES = ("MHIP-R0-100-2305", "EHIP-R0-100-2305")
TRAIN_MAX_YEAR = 2024
DEV_YEARS = (2025, 2026)
# Pre-2015 papers have "filled-in sheet" keys (answers inline in the question text, no model-answer section) and the
# old podstawowa/rozszerzona split; they are not converted by default. --include-old (= --min-year 2003) converts them
# into exam items; they become training rows only through an LLM-filled target (flag llm_filled_old).
DEFAULT_MIN_YEAR = 2015
OLD_MIN_YEAR = 2003

OPEN_FORMAT = "Tekst po polsku. Podaj wszystkie wymagane elementy odpowiedzi."
ESSAY_FORMAT = "Jeden tekst: numer wybranego tematu i całe wypracowanie. Minimum 300 wyrazów zgodnie z poleceniem."
KINDS = ("closed_single", "closed_tf", "closed_match", "closed_multi_part", "open", "essay")
CLOSED_KINDS = ("closed_single", "closed_tf", "closed_match", "closed_multi_part")
# Where a training row comes from (row["origin"], stats per_origin).
ORIGINS = ("cke-2015+", "cke-old", "llm-filled", "synthetic-essay", "synthetic-open")
SYNTHETIC_ORIGINS = ("synthetic-essay", "synthetic-open")
ESSAY_MIN_WORDS = 300
# Answers a filler writes for rows it could not answer; never a target.
FILLED_SKIP = {"SKIP", "POMIŃ", "BRAK", "N/A", "NONE"}
# Instructions for whoever fills needs_answer.jsonl (an LLM while building): what an exemplary target looks like.
NEEDS_HINTS = {
    "essay": "Wypracowanie po polsku: pierwsza linia 'Temat N.' (numer wybranego tematu), potem pełny tekst, "
             "co najmniej 300 wyrazów, teza, argumentacja z faktografią, wnioski. Oprzyj się na kryteriach z rubric.",
    "open": "Wzorcowa odpowiedź maturalna po polsku, tylko odpowiedź (bez rozumowania), z wszystkimi elementami "
            "wymaganymi przez polecenie i etykietami z polecenia (labels) w tej samej kolejności, np. "
            "'Rozstrzygnięcie: ...\nUzasadnienie: ...'. Wykorzystaj key_solution i rubric; nie przepisuj punktacji.",
    "closed_tf": "Dokładnie w składni answer_format: 'N: P' albo 'N: F' dla każdego stwierdzenia, jedno na linię.",
    "closed_single": "Jedna litera wybranej odpowiedzi.",
    "closed_match": "Dokładnie w składni answer_format: 'LITERA: numer', jedna para na linię.",
    "closed_multi_part": "Dokładnie w składni answer_format: 'numer: LITERA', jedna para na linię.",
}

DASH = "–—-"
_D = r"[–—\-]"

# ----------------------------------------------------------------------------- layout cleaner

# Whole lines that are exam-sheet furniture, never content (matched after strip(); case-sensitive unless (?i)).
_JUNK_LINES = [re.compile(p) for p in (
    r"(?i)strona\s+\d+\s+z\s+\d+",
    r"\d{1,3}\s+z\s+\d{1,3}",                                   # "16 z 44" page counters
    r"(?i)więcej arkuszy znajdziesz na stronie:?\s*arkusze\.pl\.?",
    r"(?i)brudnopis.*",
    r"(?i)\(\s*nie podlega ocenie\s*\)",
    r"(?i)wypełnia\s*(egzaminator|zdający)?",
    r"egzaminator|sprawdzający|zdający",
    r"(?i)nr zadania.*", r"(?i)maks(\.|ymalna)?\s+liczba\s+(pkt|punktów).*", r"(?i)uzyskana\s+liczba\s+(pkt|punktów).*",
    r"(?i)miejsce na naklejkę.*", r"(?i)(z kodem|z numerem)?\s*pesel.*", r"KOD( ZDAJĄCEGO)?", r"(?i)kod egzaminatora",
    r"(?i)podpis (egzaminatora|zdającego).*", r"(?i)układ graficzny.*", r"©\s*CKE.*",
    r"(?=.*\d)[A-Z]{2,5}[-_][A-Z0-9][A-Z0-9_\-]*",              # paper codes: MHI_1R, MHIP-R0-100-2305, EHIP-R0-100-A-2405
    r"(?i)egzamin maturalny z historii.*", r"(?i)(próbny\s+)?egzamin maturalny z nową erą.*",
    r"(?i)historia\s*[–-]\s*poziom (rozszerzony|podstawowy)", r"(?i)poziom (rozszerzony|podstawowy)",
    r"HISTORIA", r"(?i)formuła\s+20(15|23)", r"(?i)zasady oceniania rozwiązań zadań",
    r"(?i)termin (główny|dodatkowy|poprawkowy).*20\d\d.*",
    r"WYPRACOWANIE", r"(?i)na temat nr\b.*", r"(?i)wybieram temat\b.*", r"CZĘŚĆ\s+[IVX]+\b.*",
    r"(?i)\(\s*(0\s*[–-]\s*)?\d+\s*(pkt|p\.)\s*\)",            # "(1 pkt)", "(0–2 pkt)"
    r"[•▪◦·●]",                                           # empty answer bullets
    r"(?i)miejsce na (odpowiedź|rozwiązanie|obliczenia|notatki).*",
    r"(?i)(odpowiedzi|rozwiązania) (nie )?podlegają ocenie.*",
)]
_BOX_LABEL = re.compile(r"(?i)wypełnia\s*(egzaminator|zdający)?|nr zadania.*|maks(\.|ymalna)?\s+liczba.*|uzyskana\s+liczba.*")
_INLINE_JUNK = [re.compile(p) for p in (
    r"(?i)więcej arkuszy znajdziesz na stronie:?\s*arkusze\.pl\.?",
    r"(?i)strona\s+\d+\s+z\s+\d+",
)]
_DOTS = re.compile(r"(?:[.…·_]\s?){4,}|…{2,}")                      # dotted / underscored answer lines
_POINT_RANGE = r"0(?:\s*[–-]\s*\d{1,2})+\s*[–-]?"
_POINT_RANGE_LINE = re.compile(rf"^\s*{_POINT_RANGE}\s*$")
# a task-number marker ("5.1." / "24.") followed by its point range ("0–1") or a score box ("1"), incl. blank lines
_MARKER_WITH_POINTS = re.compile(
    rf"(?m)^[ \t]*\d{{1,2}}(?:\.\d{{1,2}})?\.[ \t]*\n(?:[ \t]*\n)*[ \t]*(?:{_POINT_RANGE}|\d{{1,2}})[ \t]*$\n?")
_SPLIT_RANGE = re.compile(r"(?m)^([ \t]*0(?:\s*[–-]\s*\d)+\s*[–-])[ \t]*\n[ \t]*(\d(?:\s*[–-]\s*\d)*)[ \t]*$")
_SUBTASK_MARKER = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.?\s*$")
_MARKERS_ONLY = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2})?\.\s*){2,}$")   # score-box row "23.1. 23.2."
_ENUM_ONLY = re.compile(r"^\s*([A-H]|\d{1,2})\.\s*$")


def clean_layout(text: str | None) -> str:
    """Strip exam-sheet layout junk from pdftotext output; keep the content and its line structure.

    Removes page headers/footers ("Strona x z y", "Egzamin maturalny z historii", "Poziom rozszerzony", paper codes),
    the arkusze.pl footer, examiner boxes ("Wypełnia egzaminator", "Nr zadania", "Maks. liczba pkt", score boxes,
    PESEL/sticker boxes), "Brudnopis", task-number markers with their point ranges ("5.1.\\n0–1"), "(1 pkt)", dotted
    answer lines (a label such as "Rozstrzygnięcie: ....." keeps its label), empty answer bullets, essay sheet headers.
    Joins a lone enumerator line ("1." / "A.") with the text line after it. Collapses blank-line runs.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).replace("\r", "").replace("\f", "\n").replace(" ", " ")
    for rx in _INLINE_JUNK:
        t = rx.sub("", t)
    t = _SPLIT_RANGE.sub(r"\1\2", t)                 # "0–1–\n2–3" -> "0–1–2–3"
    t = _MARKER_WITH_POINTS.sub("", t)
    out: list[str] = []
    in_box = False
    for raw in t.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            out.append("")
            continue
        pieces = [line]
        if _DOTS.search(line):
            pieces = []
            for piece in _DOTS.split(line):       # "Nazwa: ...... Rok: ....." -> "Nazwa:" / "Rok:"
                piece = re.sub(r"\s+([:;,])", r"\1", piece.strip())
                if re.search(rf"\s{_D}$", piece):   # "Fragment B – ......" -> "Fragment B:"
                    piece = re.sub(rf"\s*{_D}$", ":", piece)
                piece = re.sub(r"\s{2,}", " ", piece).strip()
                if piece and not re.fullmatch(r"[\W_]*", piece):
                    pieces.append(piece)
        for piece in pieces:
            if _POINT_RANGE_LINE.match(piece) or _SUBTASK_MARKER.match(piece) or _MARKERS_ONLY.match(piece):
                continue
            if any(rx.fullmatch(piece) for rx in _JUNK_LINES):
                in_box = bool(_BOX_LABEL.match(piece)) or in_box
                continue
            if in_box and re.fullmatch(r"[\d.\s]+", piece):
                continue                              # values of the "Nr zadania / Maks. liczba pkt" box
            in_box = False
            out.append(piece)
    # a column of lone enumerators followed by as many text lines: "A.\nB.\nC.\nD. x\ny\nz\nw" -> "A. x" ...
    zipped: list[str] = []
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and _ENUM_ONLY.match(out[j]):
            j += 1
        if j - i >= 2 and j < len(out):
            labels = [_ENUM_ONLY.match(x).group(1) for x in out[i:j]]
            m = re.match(r"^\s*([A-H]|\d{1,2})\.\s+(\S.*)$", out[j])
            vals, k = ([m.group(2)], j + 1) if m else ([], j)
            if m:
                labels.append(m.group(1))
            while k < len(out) and len(vals) < len(labels):
                if out[k]:
                    if _ENUM_ONLY.match(out[k]) or re.match(r"^\s*([A-H]|\d{1,2})[.)]\s", out[k]):
                        break
                    vals.append(out[k])
                k += 1
            if len(vals) == len(labels) and all(re.search(r"[.;?!]$", v) and len(v) <= 90 for v in vals):
                zipped.extend(f"{a}. {v}" for a, v in zip(labels, vals))
                i = k
                continue
        zipped.append(out[i])
        i += 1
    out = zipped
    # join lone enumerators ("1." / "A.") with the next non-empty line unless that line is itself an enumerator
    joined: list[str] = []
    i = 0
    while i < len(out):
        line = out[i]
        if _ENUM_ONLY.match(line):
            j = i + 1
            while j < len(out) and not out[j]:
                j += 1
            if j < len(out) and not _ENUM_ONLY.match(out[j]) and not re.match(r"^\s*([A-H]|\d{1,2})\.\s", out[j]):
                joined.append(f"{line.strip()} {out[j]}")
                i = j + 1
                continue
        joined.append(line)
        i += 1
    t = "\n".join(joined)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ----------------------------------------------------------------------------- source / instruction split

_POLECENIE = re.compile(
    r"^(?:Rozstrzygnij|Podaj|Wyjaśnij|Oceń|Dokończ|Uzupełnij|Wymień|Przyporządkuj|Wskaż|Zaznacz|Napisz|Określ|Porównaj|"
    r"Scharakteryzuj|Przedstaw|Uporządkuj|Nazwij|Wypisz|Sformułuj|Ustal|Udowodnij|Rozpoznaj|Wybierz|Zapisz|Uzasadnij|"
    r"Zinterpretuj|Zaproponuj|Zidentyfikuj|Oblicz|Opisz|Przeanalizuj|Wykaż|Zestaw|Wpisz|Skreśl|Połącz|Przypisz|Dopasuj|"
    r"Dopisz|Uszereguj|Podkreśl|Zakreśl|Ułóż|Do każdego|Oceń|"
    r"Odwołując się|Korzystając|W oparciu o|Spośród|Zadanie zawiera|"
    r"Na podstawie (?:(?:tekst|źródł|map|ilustracj|fotografi|tabel|danych|wykres|rysunk|karykatur|plakat|analiz|informacj|"
    r"własnej|obu|przytoczon|zamieszczon|podanych|powyższ|treści|fragment|materiał|zaprezentowan|przedstawion)\w*))\b")
# after the instruction: start of source material that belongs to the NEXT tasks (formula-2015 layout)
_TRAILING_SOURCE = re.compile(
    r"^(?:Źródło\s+(?:\d+|[A-Z])\b|Materiały(?:\s+źródłowe)?\s+do\s+(?:zadania|zadań|tematu)|"
    r"Na podstawie (?:źródła|źródeł|tekstu|mapy|ilustracji)\b.*wykonaj polecen)")


_INTRO = re.compile(r"(?i)^(?:na podstawie|korzystając z)\b.{0,80}\bwykonaj\s+polecen\w*\.?$")
_TRAILING_JUNK = re.compile(r"^(?:\d{1,3}|(?:\d{1,2}(?:\.\d{1,2})?\.\s*)+|[A-Z]_\d[A-Z])$")


def _strip_trailing_junk(text: str) -> str:
    """Drop score-box leftovers ("1", "23.1. 23.2.") at the end of an instruction."""
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or _TRAILING_JUNK.match(lines[-1].strip())):
        lines.pop()
    return "\n".join(lines).strip()


def split_source_question(body: str) -> tuple[str, str, str]:
    """Split a cleaned task body into (source_part, question_part, trailing_source_for_next_tasks)."""
    # formula-2015 intro lines ("Na podstawie źródła A i własnej wiedzy wykonaj polecenia.") are not instructions
    lines = [ln for ln in body.split("\n") if not _INTRO.match(ln.strip())]
    body = "\n".join(lines)
    start = next((i for i, ln in enumerate(lines) if _POLECENIE.match(ln.strip())), None)
    if start is None:
        return "", _strip_trailing_junk(body), ""
    source = "\n".join(lines[:start]).strip()
    rest = lines[start:]
    cut = next((i for i, ln in enumerate(rest) if i > 0 and _TRAILING_SOURCE.match(ln.strip())), None)
    if cut is None:
        return source, _strip_trailing_junk("\n".join(rest)), ""
    return source, _strip_trailing_junk("\n".join(rest[:cut])), "\n".join(rest[cut:]).strip()


# ----------------------------------------------------------------------------- picture dependence

# Inflection-robust stems of visual source types (headers/captions and references in the instruction).
_VIS_STEMS = (
    r"obraz(?!uj|ow)\w*|ilustracj\w*|rysun\w*|rysunk\w*|map(?:a|y|ie|ę|ą|om|ami|ach)?|mapk\w*|fotografi\w*|"
    r"fotomontaż\w*|zdjęci\w*|zdjęć|karykatur\w*|plakat\w*|afisz\w*|monet\w*|banknot\w*|medal\w*|"
    r"znacz(?:ek|ka|ku|kiem|ki|ków|kach|kami)|pocztów\w*|widokówk\w*|schemat\w*|wykres\w*|diagram\w*|infografik\w*|"
    r"drzeworyt\w*|rycin\w*|miedzioryt\w*|litografi\w*|grafik\w*|graficzn\w*|malowid\w*|fresk\w*|mozaik\w*|relief\w*|"
    r"rzeźb\w*|miniatur\w*|witraż\w*|herb\w*|godł\w*|pieczę\w*|portret\w*|kadr\w*|komiks\w*|okładk\w*|ulotk\w*|"
    r"obrazk\w*|drzew\w* genealogiczn\w*|tablic\w* genealogiczn\w*|genealogi\w*|fotografiach|kolaż\w*|"
    r"plakac\w*|monec\w*|schemac\w*|drzeworyc\w*|miedzioryc\w*|portrec\w*|bankno\w*|grafic\w*|mozaic\w*|"
    r"okładc\w*|ulotc\w*|widokówc\w*|karykaturz\w*|rycin\w*|fotomontaż\w*"
)
VISUAL_WORD = re.compile(rf"(?i)\b(?:{_VIS_STEMS})\b")
_PLAN_WORD = re.compile(r"(?i)\bplan(?:y|u|ie|ów|ach|ami)?\b")        # visual only in a header/caption
TABLE_WORD = re.compile(r"(?i)\btabel\w*")
_TABLE_REF = re.compile(r"(?i)\btabel(?:a|i|ach)\b")                   # "w tabeli" (source) vs "uzupełnij tabelę" (answer)
_VISUAL_REF = re.compile(
    rf"(?i)\b(?:{_VIS_STEMS})\b|\bna\s+plan(?:ie|ach)\b|\bplan(?:ów|ach)\b|podstawie\s+planu|elementy?\s+graficzn|widoczn\w*\s+na|\blegend\w*|"
    r"\b(?:przedstawion|ukazan|zaznaczon|oznaczon|uwiecznion|zobrazowan)\w*\s+na\b|\bzobrazowan\w*|"
    r"\b(?:przedstawion|ukazan|widoczn|zaprezentowan|prezentowan)\w*\s+(?:\w+\s+)?"
    r"(?:budowl|budyn|obiekt|świątyn|kości|katedr|zamk|pałac|pomnik|rzeźb|scen|postaci\b|stroj|broni|wizerun)\w*|"
    r"\b(?:budowl|budyn|obiekt|świątyn|kości|katedr|zamk|pałac|pomnik|rzeźb|wizerun)\w*\s+(?:przedstawion|ukazan|widoczn|zaprezentowan|prezentowan)")
_HEADER = re.compile(r"^(?:Źródło|Zrodlo)\s+(\d+|[A-Z])\b\.?\s*(.*)$")
_SOURCE_LABEL_REF = re.compile(r"(?i)źród(?:ło|ła|le|łem|eł|łach|łami|łom)\s+((?:\d+|[A-Z])\b(?:\.?\s*(?:i|oraz|,|–|-)\s*(?:\d+|[A-Z])\b)*)")
_ALL_SOURCES_REF = re.compile(r"(?i)\b(?:obu|obydwu|wszystkich|tych)\s+źród|\bźródłach\b|\bźródeł\b|\bźródła\s+(?:tekstow|ikonograf)")


@dataclass
class SourceBlock:
    label: str | None
    header: str
    visual: bool
    table: bool
    lines: list[str] = field(default_factory=list)


def _caption_is_visual(line: str) -> bool:
    head = " ".join(line.split()[:10])
    return bool(VISUAL_WORD.search(head) or _PLAN_WORD.search(head))


def _is_caption(line: str, first: bool) -> bool:
    s = line.strip()
    if not s or len(s) > 160:
        return False
    if _HEADER.match(s):
        return True
    # a bare caption ("Rysunek z epoki", "Fragment tablicy genealogicznej ...") at the start of a block
    return first and not s.endswith((".", ",", ";")) and (_caption_is_visual(s) or bool(TABLE_WORD.match(s)))


def source_blocks(source_text: str) -> list[SourceBlock]:
    """Split sources into blocks at "Źródło N." headers (or a caption at the very start)."""
    blocks: list[SourceBlock] = []
    cur: SourceBlock | None = None
    prev_blank = True
    for line in source_text.split("\n"):
        s = line.strip()
        m = _HEADER.match(s)
        if m or (_is_caption(s, first=prev_blank and (cur is None or not cur.lines)) and cur is None):
            label = m.group(1) if m else None
            cur = SourceBlock(label=label, header=s, visual=_caption_is_visual(s),
                              table=bool(TABLE_WORD.search(" ".join(s.split()[:6]))), lines=[])
            blocks.append(cur)
        elif cur is None:
            cur = SourceBlock(label=None, header="", visual=False, table=False, lines=[line])
            blocks.append(cur)
        else:
            cur.lines.append(line)
        prev_blank = not s
    for b in blocks:
        body = [ln.strip() for ln in b.lines if ln.strip()]
        short = [ln for ln in body if len(ln) <= 3 and not re.fullmatch(r"[\[\]…. ]+", ln)]
        # scattered one-letter/number lines = labels of a map or diagram that pdftotext pulled out of the figure
        if not b.visual and len(short) >= 6 and len(short) >= 0.4 * max(1, len(body)):
            b.visual = True
    return blocks


def _is_citation(line: str) -> bool:
    s = line.strip()
    return bool(re.search(r"(?i)https?://|www\.|\bs\.\s*\d|na podstawie:|\.(?:pl|org|com|gov|net|uk|de|fr)\b", s)
                or (re.search(r"\b(?:1[5-9]|20)\d\d\b", s) and len(s.split()) >= 3 and re.search(r"[A-ZŁŚŻ]\.\s?[A-ZŁŚŻ]", s)))


def _has_prose(lines: list[str]) -> bool:
    """A short quote or poem (two or more sentences) is text, not a picture caption."""
    body = [ln.strip() for ln in lines if ln.strip() and not _is_citation(ln)]
    return len(re.findall(r"[.?!](?=\s|$)", " ".join(body[1:] if len(body) > 1 else body))) >= 2


def thin_block(block: SourceBlock) -> bool:
    """Less than ~150 characters of text besides captions and citations: CKE text sources are always longer."""
    text = " ".join(ln.strip() for ln in [block.header, *block.lines] if ln.strip() and not _is_citation(ln))
    return 0 < len(block.header) + sum(len(x) for x in block.lines) and len(text) < 150


def scrambled_table(block: SourceBlock) -> bool:
    body = [ln.strip() for ln in block.lines if ln.strip()]
    short = [ln for ln in body if len(ln) <= 15]
    return len(short) >= 8 and len(short) >= 0.5 * len(body)


def referenced_blocks(question: str, blocks: list[SourceBlock]) -> list[SourceBlock]:
    """Source blocks the instruction points at ("źródło 2.", "obu źródeł", a lone unlabeled source, ...)."""
    if not blocks:
        return []
    labels = {b.label for b in blocks if b.label}
    refs: set[str] = set()
    for m in _SOURCE_LABEL_REF.finditer(question):
        refs.update(re.findall(r"\d+|[A-Z]", m.group(1)))
    if _ALL_SOURCES_REF.search(question):
        refs.update(labels)
    out = [b for b in blocks if b.label and b.label in refs]
    if not out and re.search(r"(?i)\bźród\w*|\btekst\w*|\bfragment\w*", question):
        out = [b for b in blocks if b.header or b.lines]
    return out


@dataclass
class PictureCheck:
    dependent: bool
    reason: str | None
    has_visual: bool
    flags: list[str]


def picture_check(question: str, source_text: str) -> PictureCheck:
    """Does answering need a picture the model will not see?

    Dependent when the instruction refers to a visual source (by label, "obu źródeł", or a visual word such as
    "na mapie", "ilustracji", "elementy graficzne", any inflection), or to a table whose layout pdftotext scrambled.
    A visual source the instruction never uses (pure knowledge question) is kept with the flag visual_source_unused.
    """
    blocks = source_blocks(source_text)
    for b in blocks:
        if not b.visual and (b.header or len(blocks) == 1) and thin_block(b) and not _has_prose(b.lines):
            b.visual = True                      # a caption + citation with no text: the source was a picture
    has_visual = any(b.visual for b in blocks)
    refs = referenced_blocks(question, blocks)
    flags: list[str] = []
    if _VISUAL_REF.search(question):
        return PictureCheck(True, "question_refers_to_picture", has_visual, flags)
    if any(b.visual for b in refs):
        return PictureCheck(True, "refers_to_visual_source", has_visual, flags)
    text_chars = sum(len(ln.strip()) for b in blocks if not b.visual for ln in [b.header, *b.lines]
                     if ln.strip() and not _is_citation(ln))
    prose = any(_has_prose(b.lines) for b in blocks)
    if source_text.strip() and text_chars < 150 and not prose and _CONTENT_REF.search(question):
        return PictureCheck(True, "thin_source", has_visual, flags)
    if has_visual and _letters_only_in_picture(question, blocks):
        return PictureCheck(True, "labels_in_picture", has_visual, flags)
    if any(b.table and scrambled_table(b) for b in refs) or (
            _TABLE_REF.search(question) and any(b.table and scrambled_table(b) for b in blocks)):
        return PictureCheck(True, "scrambled_table", has_visual, flags)
    if has_visual:
        flags.append("visual_source_unused")
    return PictureCheck(False, None, has_visual, flags)


_CONTENT_REF = re.compile(r"(?i)\b(?:źród\w*|tekst\w*|fragment\w*|artykuł\w*|dokument\w*|wypowied\w*|autor\w*|treś\w*|"
                          r"przedstawion\w*|opisan\w*|cytowan\w*|przytoczon\w*|zamieszczon\w*|zaprezentowan\w*|"
                          r"widoczn\w*|ukazan\w*|gazet\w*|list\w*|plan\w*)")
_LETTER_CHOICE = re.compile(r"\b([A-F])\b(?:\s*,\s*([A-F])\b)*\s*(?:czy|lub|albo|i|oraz|–|-)\s*([A-F])\b")


def _letters_only_in_picture(question: str, blocks: list[SourceBlock]) -> bool:
    """"(A, B czy C)" in the instruction, but no text source labels anything with those letters: they live in a picture."""
    q = "\n".join(ln for ln in question.split("\n") if not _OPTION_LINE.match(ln))
    letters: set[str] = set()
    for m in _LETTER_CHOICE.finditer(q):
        letters.update(x for x in m.groups() if x)
    if not letters:
        return False
    text = "\n".join(b.header + "\n" + "\n".join(b.lines) for b in blocks if not b.visual)
    labelled = set(re.findall(r"(?m)(?:^|\b(?:Fragment|Tekst|Dokument|Wersja|Traktat|Opis|Wypowied\w*|Postać|Polityk)\s+)([A-F])\b\s*[.:)–-]", text))
    return not letters <= labelled


def missing_source(question: str, source_text: str) -> bool:
    """The instruction cites a labeled source ("źródło B") that is not in the source text."""
    refs: set[str] = set()
    for m in _SOURCE_LABEL_REF.finditer(question):
        refs.update(re.findall(r"\d+|[A-Z]", m.group(1)))
    if not refs:
        if not source_text.strip() and re.search(r"(?i)\bźród\w*\s|\bw\s+tekście\b|\bfragment\w*\s+(tekstu|źródła)", question):
            return True
        return False
    have = {m.group(1) for m in (_HEADER.match(ln.strip()) for ln in source_text.split("\n")) if m}
    if not have:
        return not source_text.strip()
    return not refs <= have


def mark_images(source_text: str, group: str) -> str:
    """Replace each visual source's figure (caption kept, stray figure labels dropped) with an image marker.

    Mirrors the organizers' separate-text-and-images format ("[Obraz: images/Z05-S2.png]"), so a kept row whose
    picture the instruction does not use looks exactly like an exam item whose image was not described.
    """
    blocks = source_blocks(source_text)
    if not any(b.visual for b in blocks):
        return source_text
    g = re.sub(r"\D", "", group) or "0"
    out: list[str] = []
    for n, b in enumerate(blocks, 1):
        if b.header:
            out.append(b.header)
        if not b.visual:
            out.extend(b.lines)
            continue
        name = f"Z{int(g):02d}-S{b.label or n}.png" if len(blocks) > 1 else f"Z{int(g):02d}.png"
        out.append(f"[Obraz: images/{name}]")
        out.append("")
        body = [ln for ln in b.lines if ln.strip()]
        k = 0
        while k < len(body) and len(body[k].strip()) <= 30 and not _is_citation(body[k]):
            k += 1                                   # figure labels pulled out of the picture
        out.extend(body[k:])
        out.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ----------------------------------------------------------------------------- answer key

_RUBRIC_HDR = re.compile(r"(?mi)^[ \t]*(?:zasady oceniania(?!\s+rozwiązań)|schemat punktowania|schemat oceniania)\b[ \t]*:?[ \t]*$")
_SOLUTION_HDR = re.compile(
    r"(?mi)^[ \t]*(?:rozwiązani[ea]|przykładow\w*\s+(?:rozwiązani\w*|odpowied\w*|realizacj\w*)|poprawn\w*\s+odpowied\w*|"
    r"prawidłow\w*\s+odpowied\w*|model\s+odpowiedzi)\b[ \t]*:?[ \t]*")
_POINTS_LINE = re.compile(r"(?i)^\s*(\d{1,2})\s*(?:pkt|p\.|punkt\w*)\s*[–—-]")
_SOLUTION_END = re.compile(
    r"(?mi)^[ \t]*(?:Temat\s+\d|Część\s+[IVX]+\b|Wymagani[ea]\s+(?:ogóln|szczegół|egzaminacyjn)|Kryteri|"
    r"DODATKOWE INFORMACJE|Zasady oceniania|Schemat punktowania|Zadanie\s+\d)")
_NOTE = re.compile(r"(?mi)^[ \t]*uwag[ai]\b.*(?:\n(?![ \t]*\n).*)*")


@dataclass
class ParsedKey:
    rubric: str
    solution: str
    max_points: int | None


def parse_key(raw: str | None) -> ParsedKey:
    """Split a CKE key section into (rubric = points scheme, solution = model answer). Requirements preamble dropped."""
    if not raw:
        return ParsedKey("", "", None)
    t = clean_layout(raw)
    rm = _RUBRIC_HDR.search(t)
    sm = _SOLUTION_HDR.search(t)
    rubric, solution = "", ""
    if rm:
        after = t[rm.end():]
        lines = after.split("\n")
        # the rubric is the "N pkt – ..." lines (with wrapped continuations) right after its header
        end, seen_zero, i = 0, False, 0
        while i < len(lines):
            ln = lines[i].strip()
            pm = _POINTS_LINE.match(ln)
            if pm:
                seen_zero = pm.group(1) == "0"
                end = i + 1
            elif ln and end and not seen_zero and not _SOLUTION_HDR.match(ln):
                end = i + 1                       # continuation of a points line
            elif ln and seen_zero and end == i and not lines[i - 1].rstrip().endswith(".") and not _SOLUTION_HDR.match(ln):
                end = i + 1                       # continuation of the "0 pkt" line
            elif ln and end:
                break
            i += 1
        rubric = "\n".join(lines[:end]).strip()
        rest = "\n".join(lines[end:])
        if sm and sm.start() < rm.start():
            solution = t[sm.end():rm.start()]
        else:
            m2 = _SOLUTION_HDR.search(rest)
            if m2 and not rest[:m2.start()].strip():
                rest = rest[m2.end():]
            elif m2 and len(rest[:m2.start()].strip()) < 400 and _NOTE.match(rest[:m2.start()].strip() or "x"):
                rest = rest[m2.end():]            # grader note between rubric and solution
            solution = rest
    elif sm:
        solution = t[sm.end():]
    end = _SOLUTION_END.search(solution)
    if end:
        solution = solution[:end.start()]
    solution = _NOTE.sub("", solution)
    pts = [int(m.group(1)) for m in (_POINTS_LINE.match(ln.strip()) for ln in rubric.split("\n")) if m]
    return ParsedKey(rubric=rubric.strip(), solution=solution.strip(), max_points=max(pts) if pts else None)


# ----------------------------------------------------------------------------- answer formats / closed items

_TF_Q = re.compile(r"(?i)oceń prawdziwość|(?:zaznacz|wybierz|wpisz)\s+P,?\s+jeśli|prawdziw\w+.{0,40}fałszyw")
_CHOICE_Q = re.compile(r"(?i)\b(?:zaznacz|wybierz|podkreśl)\w*\b.{0,80}\b(?:odpowied\w*|dokończeni\w*)|dokończ\s+zdani")
_MULTI_Q = re.compile(r"(?i)dokończ\s+zdania\s+\d|zdania\s+1\.?\s*(?:i|–|-)\s*\d|zaznacz\s+właściwe\s+odpowiedzi\s+spośród")
_MATCH_Q = re.compile(r"(?i)przyporządkuj|uzupełnij\s+tabel|wpisz\s+(?:obok|w\s+tabel|numer|liter|odpowiedn)|"
                      r"połącz|dopasuj|obok\s+(?:opisu|każdego|nazwy|fragmentu)\s+wpisz|przypisz")
_JUSTIFY_Q = re.compile(r"(?i)uzasadnij")
_OPTION_LINE = re.compile(r"(?m)^\s*([A-F])[.)]\s+\S")
_STATEMENT_LINE = re.compile(r"(?m)^\s*(\d{1,2})[.)]\s+\S")
_PAIR = re.compile(rf"^\s*([A-H]|\d{{1,2}})\s*[.)]?\s*(?:{_D}|:|\.|→|=)?\s*([A-H]|\d{{1,2}})\s*[.,;]?\s*$")
_TF_PAIR = re.compile(rf"(?:^|[\s,;])(\d{{1,2}})\s*[.)]?\s*(?:{_D}|:|\.)?\s*(P|F)\b")
_LETTER_ONLY = re.compile(rf"^\s*(?:odpowied[źz]\s*:?\s*)?([A-F])\s*[.)]?\s*(?:{_D}.*)?$", re.I)


def tf_format(n: int) -> str:
    return "\n".join(f"{i}: {'P' if i % 2 else 'F'}" for i in range(1, n + 1))


def pairs_format(keys: list[str], placeholder: str) -> str:
    return "\n".join(f"{k}: {placeholder}" for k in keys)


def render_closed(kind: str, answer: list[tuple[str, str]] | str) -> str:
    """Exact answer_format syntax: "B" | "1: P\\n2: F" | "A: 3\\nB: 1" | "1: A\\n2: C"."""
    if kind == "closed_single":
        return str(answer).strip()
    return "\n".join(f"{k}: {v}" for k, v in answer)


def _statement_numbers(question: str) -> list[int]:
    nums = [int(m.group(1)) for m in _STATEMENT_LINE.finditer(question)]
    run: list[int] = []
    for n in nums:                                # longest 1..n run
        if n == len(run) + 1:
            run.append(n)
    return run


def _parse_pairs(solution: str) -> list[tuple[str, str]] | None:
    """Strict: every non-empty line (or comma-separated chunk) is "X – Y"."""
    chunks: list[str] = []
    for ln in solution.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        parts = [p for p in re.split(r"\s*[,;]\s*", ln) if p]
        chunks.extend(parts if len(parts) > 1 else [ln])
    pairs = []
    for c in chunks:
        m = _PAIR.match(c)
        if not m:
            return None
        pairs.append((m.group(1), m.group(2)))
    return pairs if len(pairs) >= 2 else None


@dataclass
class Closed:
    kind: str
    answer_format: str
    target: str | None
    problem: str | None = None
    question: str | None = None          # re-laid-out instruction (P/F items)


_PF_MARK = re.compile(r"^(?:\d{1,2}\.\s*)?(?:P|F|PF|P\s+F|P\s*/\s*F)$")


def rebuild_tf_question(question: str) -> tuple[str, int]:
    """Lay a P/F item out like the organizers: instruction, then "1. statement" lines, answer columns removed.

    pdftotext leaves the P / F answer columns between statements and sometimes glues a statement number to the
    statement's second line; statements are recovered as the text between P/F column marks.
    Returns (question, number_of_statements); 0 statements when the layout cannot be recovered.
    """
    lines = question.split("\n")
    head_end = next((i for i, ln in enumerate(lines) if re.search(r"(?i)fałszyw\w*\.?\s*$", ln.strip())), None)
    if head_end is None:
        return question, 0
    stmts: list[list[str]] = []
    cur: list[str] = []
    for ln in lines[head_end + 1:]:
        s = ln.strip()
        if _PF_MARK.match(s):
            if cur:
                stmts.append(cur)
                cur = []
            continue
        if s:
            cur.append(s)
    if cur:
        stmts.append(cur)
    if len(stmts) < 2:
        return question, 0
    body = [re.sub(r"^\d{1,2}[.)]\s*", "", "\n".join(st).strip()) for st in stmts]
    body = [re.sub(r"(?m)^\d{1,2}\.\s+(?=\S)", "", b) for b in body]
    head = "\n".join(lines[:head_end + 1]).strip()
    return head + "\n" + "\n".join(f"{i}. {b}" for i, b in enumerate(body, 1)), len(body)


def _tf_key(solution: str) -> list[str] | None:
    found: dict[int, str] = {}
    for m in _TF_PAIR.finditer(solution):
        found.setdefault(int(m.group(1)), m.group(2))
    if found and sorted(found) == list(range(1, len(found) + 1)):
        return [found[i] for i in range(1, len(found) + 1)]
    s = solution.strip()
    if re.fullmatch(r"[PF](?:[\s,;]*[PF])+", s):          # "FP" / "P, F, P"
        return re.findall(r"[PF]", s)
    return None


def detect_closed(question: str, solution: str) -> Closed | None:
    """Recognize the four closed syntaxes the organizers' answer_format uses; None -> open item."""
    if _TF_Q.search(question):
        key = _tf_key(solution)
        rebuilt, m = rebuild_tf_question(question)
        nums = _statement_numbers(question)
        if not key or len(key) < 2:
            n = m or len(nums) or 2
            return Closed("closed_tf", tf_format(n), None, "tf_unparsed")
        n = len(key)
        fmt = tf_format(n)
        target = render_closed("closed_tf", [(str(i), v) for i, v in enumerate(key, 1)])
        if m == n:
            return Closed("closed_tf", fmt, target, question=rebuilt)
        if len(nums) == n:
            return Closed("closed_tf", fmt, target, question=re.sub(r"(?m)^\s*(?:P|F|P\s+F)\s*$\n?", "", question).strip())
        return Closed("closed_tf", fmt, None, "tf_layout_mismatch")
    pairs = _parse_pairs(solution)
    options = _OPTION_LINE.findall(question)
    if pairs:
        left = [a for a, _ in pairs]
        right = [b for _, b in pairs]
        letters_left = all(re.fullmatch(r"[A-H]", a) for a in left)
        digits_left = all(a.isdigit() for a in left)
        if letters_left and all(b.isdigit() for b in right) and len(set(left)) == len(left):
            return Closed("closed_match", pairs_format(left, "1"), render_closed("closed_match", pairs))
        if digits_left and all(re.fullmatch(r"[A-H]", b) for b in right) and len(set(left)) == len(left):
            kind = "closed_multi_part" if (_MULTI_Q.search(question) or options) else "closed_match"
            return Closed(kind, pairs_format(left, "A"), render_closed(kind, pairs))
    if len(options) >= 2 and _CHOICE_Q.search(question) and not _MULTI_Q.search(question):
        first = next((ln for ln in solution.split("\n") if ln.strip()), "")
        m = _LETTER_ONLY.match(first)
        if m and m.group(1) in options:
            if _JUSTIFY_Q.search(question):
                return None                       # choice + justification: an open answer ("C\nUzasadnienie: ...")
            return Closed("closed_single", "A", m.group(1))
    return None


def is_essay(question: str, points: int | None) -> bool:
    return bool(re.search(r"(?i)wybierz\s+jeden\s+z\s+(?:nich|tematów|podanych)|zadanie\s+zawiera\s+\w+\s+temat", question)
                or (points or 0) >= 10)


# ----------------------------------------------------------------------------- open-answer targets

_LABEL_LINE = re.compile(r"^([A-ZŁŚŻŹĆÓ][^\n:]{0,48}?)\s*:\s*(.*)$")
_BULLET = re.compile(r"^\s*(?:[•▪◦●]|[–-](?=\s)|\*|[a-h]\)|\d{1,2}\))\s*")
_EXAMPLE_HDR = re.compile(r"(?i)^przykładow\w*\s+(\w+)\s*:?\s*(.*)$")
_EXAMPLE_LABELS = {
    "uzasadnienie": "Uzasadnienie", "uzasadnienia": "Uzasadnienie",
    "wyjaśnienie": "Wyjaśnienie", "wyjaśnienia": "Wyjaśnienie", "wniosek": "Wniosek", "wnioski": "Wniosek",
    "ocena": "Ocena", "argument": None, "argumenty": None, "cecha": None, "cechy": None,
}
_COUNT_WORDS = {"dwa": 2, "dwie": 2, "dwóch": 2, "dwu": 2, "dwoma": 2, "trzy": 3, "trzech": 3, "cztery": 4, "czterech": 4}
_COUNT_Q = re.compile(r"(?i)\b(dwa|dwie|dwóch|dwu|dwoma|trzy|trzech|cztery|czterech)\b\s+(?:\w+\s+){0,2}?"
                      r"(argument\w*|przykład\w*|cech\w*|przyczyn\w*|skutk\w*|czynnik\w*|zmian\w*|element\w*|informacj\w*|"
                      r"postanowie\w*|decyzj\w*|reform\w*|wydarze\w*|nazw\w*|dział\w*|sposob\w*|konsekwencj\w*|"
                      r"różnic\w*|podobieństw\w*|powod\w*|fakt\w*|osiągnię\w*|dowod\w*|przejaw\w*|cel\w*|zasad\w*)")


def question_labels(question: str) -> list[str]:
    """Answer-scaffold labels left in the instruction ("Rozstrzygnięcie:", "Uzasadnienie:", "Nazwa stylu 1.:")."""
    out = []
    for ln in question.split("\n"):
        s = ln.strip()
        m = re.fullmatch(r"([A-ZŁŚŻŹĆÓ][^:]{0,48}?)\s*:", s)
        if m and len(m.group(1).split()) <= 5 and not _POLECENIE.match(s):
            out.append(m.group(1).strip())
    return out


def required_count(question: str) -> int:
    m = _COUNT_Q.search(question)
    return _COUNT_WORDS[m.group(1).lower()] if m else 1


_ALT_SEP = re.compile(r"\s+/\s+")
_ENUM_VALUE = re.compile(rf"^([A-H]|\d{{1,2}})\s*(?:[.)]\s*(?:{_D}\s*)?|{_D}\s*|:\s*)(\S.*)$")


def _resolve_slashes(rest: str) -> str:
    parts = _ALT_SEP.split(rest)
    bare = re.sub(r"\([^()]*\)", "", rest)
    if all(len(p.split()) <= 6 for p in parts) and not re.search(r"[,;:!?]|\.\s", bare):
        end = re.search(r"[.;!?]\s*$", parts[-1])        # a list of variants: keep the first
        return parts[0].rstrip(" ,;.") + (end.group(0).strip() if end else "")
    while True:                                        # variants inside a sentence, left to right
        m = _ALT_SEP.search(rest)
        if not m:
            return rest
        tail = rest[m.end():]
        seg = re.match(r"[^,.;:!?/()]*", tail).group(0)
        if len(seg.split()) <= 1:                      # word-level: "Turków / Arabów, a ..." -> "Turków, a ..."
            rest = rest[:m.start()] + tail[len(seg.rstrip()):]
            continue
        stop = re.search(r"[.;!?](?=\s|$)", tail)      # phrase-level: keep the first phrase, cut to sentence end
        rest = rest[:m.start()] + (tail[stop.start():] if stop else "")


def resolve_alternatives(text: str) -> str:
    """Pick one wording where the key lists accepted variants: "X / Y / Z" and "unia lubelska [unia realna]".

    A short list of variants keeps the first; inside a sentence a one-word variant ("Turków / Arabów") is dropped
    and a phrase-level variant list keeps the first phrase up to the end of that sentence. Bracketed optional parts
    are unwrapped at the start of an answer ("[konstytucja] nihil novi" -> "konstytucja nihil novi") and dropped
    elsewhere ("unia lubelska [unia realna]" -> "unia lubelska"); source ellipses "[…]" stay.
    """
    lines = []
    for ln in text.split("\n"):
        s = ln
        lab = ""
        m = _LABEL_LINE.match(s.strip())
        if m and len(m.group(1).split()) <= 5:
            lab, s = m.group(1).strip() + ": ", m.group(2)
        e = re.match(r"^(\d{1,2}\.\s+)(.*)$", s)
        if e:
            lab, s = lab + e.group(1), e.group(2)
        s = re.sub(r"^(\s*)\[([^\[\]…]{1,60})\]\s*", r"\1\2 ", s)
        s = re.sub(r"\s*\[(?!\s*(?:…|\.\.\.)\s*\])[^\[\]]{1,200}\]", "", s)
        s = re.sub(r"\(\s*(?:lub|albo|ew\.|ewentualnie)\s[^()]*\)", "", s)
        if _ALT_SEP.search(s):
            s = _resolve_slashes(s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        s = re.sub(r"\s+([,.;:])", r"\1", s)
        lines.append((lab + s).rstrip())
    return "\n".join(lines).strip()


def _join_wrapped(lines: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(ln.strip() for ln in lines if ln.strip())).strip()


def _finish(text: str) -> str:
    """Bullet fragments end with ";" or "," in keys; an answer sentence ends with a period, a name with nothing."""
    t = text.strip().rstrip(";,").strip()
    if len(t.split()) >= 6 and not re.search(r"[.!?…”\"»)]$", t):
        t += "."
    return t


def _norm_label(label: str) -> str:
    return re.sub(r"[\s.:]+$", "", label.strip()).lower()


def _match_qlabel(left: str, qlabels: list[str]) -> str | None:
    l = _norm_label(left)
    for q in qlabels:
        nq = _norm_label(q)
        if nq == l or (len(l) <= 3 and nq.endswith(" " + l)):
            return q
    return None


def _sections(solution: str, qlabels: list[str]) -> list[tuple[str | None, list[str]]]:
    """[(label, content lines)] - labels are "Label: ..." lines, "Label – value" lines naming a question label,
    or a line equal to a question label; "Przykładowe uzasadnienie:" headers become the "Uzasadnienie" label."""
    qset = {_norm_label(q) for q in qlabels}
    solution = re.sub(r"(?<=\S)[ \t]+(?=Przykładow\w*\s+\w+\s*:)", "\n", solution)
    secs: list[tuple[str | None, list[str]]] = [(None, [])]
    for ln in solution.split("\n"):
        s = ln.strip()
        if not s:
            secs[-1][1].append("")
            continue
        ex = _EXAMPLE_HDR.match(s)
        if ex:
            noun = ex.group(1).lower()
            label = _EXAMPLE_LABELS.get(noun)
            if label is None and noun in _EXAMPLE_LABELS:          # "Przykładowe argumenty": items of this section
                secs[-1][1].extend([ex.group(2)] if ex.group(2) else [])
                continue
            secs.append((label, [ex.group(2)] if ex.group(2) else []))
            continue
        d = re.match(rf"^(.{{1,40}}?)\s+{_D}\s+(.+)$", s)
        if d and qlabels and not _BULLET.match(s):
            q = _match_qlabel(d.group(1), qlabels)
            if q:
                secs.append((q, [d.group(2)]))
                continue
        m = _LABEL_LINE.match(s)
        if m and len(m.group(1).split()) <= 5 and not _BULLET.match(s) and not re.search(r"\d{3,}", m.group(1)):
            secs.append((m.group(1).strip(), [m.group(2)] if m.group(2) else []))
            continue
        if _norm_label(s) in qset:
            secs.append((_match_qlabel(s, qlabels), []))
            continue
        secs[-1][1].append(s)
    return [(lab, body) for lab, body in secs if lab is not None or any(x.strip() for x in body)]


def _content(lines: list[str], k: int) -> str:
    """Section content: bulleted alternatives -> first k (numbered when k > 1); plain text -> joined paragraph(s)."""
    items: list[list[str]] = []
    plain: list[str] = []
    for ln in lines:
        if not ln.strip():
            if plain and not items:
                plain.append("")
            continue
        if _BULLET.match(ln):
            items.append([_BULLET.sub("", ln)])
        elif items:
            items[-1].append(ln)
        else:
            plain.append(ln)
    if items:
        chosen = [_finish(_join_wrapped(it)) for it in items[:k]]
        head = _join_wrapped(plain)
        body = chosen[0] if len(chosen) == 1 else "\n".join(f"{i}. {c}" for i, c in enumerate(chosen, 1))
        if not head:
            return body
        return f"{head}\n{body}" if len(chosen) > 1 else f"{head} {body}"
    paras, cur = [], []
    for ln in plain:
        if ln == "":
            if cur:
                paras.append(_join_wrapped(cur))
                cur = []
        else:
            cur.append(ln)
    if cur:
        paras.append(_join_wrapped(cur))
    return "\n".join(paras)


@dataclass
class OpenTarget:
    target: str | None
    problem: str | None


def _table_like(solution: str) -> bool:
    """Answer-table cells that pdftotext pulled apart (one word per line, headers mixed with values)."""
    body = [ln.strip() for ln in solution.split("\n") if ln.strip() and not _BULLET.match(ln)]
    body = [ln for ln in body if not (_LABEL_LINE.match(ln) and len(ln.split(":")[1].strip()) > 0)]
    if len(body) < 4:
        return False
    single = [ln for ln in body if len(ln.split()) == 1]
    short = [ln for ln in body if len(ln) <= 30 and not re.search(r"[!?;]$", ln) and not _ENUM_VALUE.match(ln)]
    return len(short) >= 0.7 * len(body) or (len(single) >= 4 and len(single) >= 0.4 * len(body))


def _enumerated_values(solution: str, qlabels: list[str]) -> str | None:
    """Keys like "A. [Aleksander] Wielopolski\\nB. [Romuald] Traugutt" -> "A: Aleksander Wielopolski\\nB: ..."."""
    lines = [ln.strip() for ln in solution.split("\n") if ln.strip()]
    ms = [_ENUM_VALUE.match(ln) for ln in lines]
    if len(lines) < 2 or not all(ms):
        return None
    labels = [m.group(1) for m in ms]
    if len(set(labels)) != len(labels):
        return None
    use_q = len(qlabels) == len(labels) and all(_match_qlabel(l, qlabels) for l in labels)
    out = []
    for lab, m in zip(labels, ms):
        name = _match_qlabel(lab, qlabels) if use_q else lab
        out.append(f"{name}: {_finish(resolve_alternatives(m.group(2)))}")
    return "\n".join(out)


def build_open_target(question: str, solution: str, choice_letter: str | None = None) -> OpenTarget:
    """Exemplary Polish answer from the key's model answer, laid out on the instruction's own labels."""
    if not solution.strip():
        return OpenTarget(None, "rubric_only")
    if re.search(r"(?i)\b\d\s*(pkt|p\.)\s*[–—-]\s*za\b", solution):
        return OpenTarget(None, "rubric_leak")
    qlabels = question_labels(question)
    if not choice_letter:
        ev = _enumerated_values(solution, qlabels)
        if ev:
            return OpenTarget(ev, None)
    if _table_like(solution):
        return OpenTarget(None, "table_answer")
    k = max(required_count(question), 1)
    secs = _sections(solution, qlabels)
    rendered: list[tuple[str | None, str]] = []
    for lab, body in secs:
        txt = resolve_alternatives(_content(body, k))
        if lab is None and not txt:
            continue
        rendered.append((lab, txt))
    if not rendered:
        return OpenTarget(None, "rubric_only")
    if choice_letter:
        just = next((t for lab, t in rendered if lab and lab.lower().startswith("uzasadn")), None)
        if not just:
            return OpenTarget(None, "choice_without_justification")
        return OpenTarget(f"{choice_letter}\nUzasadnienie: {just}", None)
    slabels = [lab for lab, _ in rendered if lab]
    if qlabels:
        if len(slabels) == len(rendered) == len(qlabels):
            rendered = [(q, t) for q, (_, t) in zip(qlabels, rendered)]
        elif rendered[0][0] is None and len(slabels) == len(rendered) - 1 == len(qlabels) - 1:
            rendered = [(q, t) for q, (_, t) in zip(qlabels, rendered)]   # "B.\nPrzykładowe uzasadnienie: ..."
        elif not slabels and len(qlabels) == 1:
            rendered = [(qlabels[0], "\n".join(t for _, t in rendered))]
        elif not slabels and len(rendered) == 1 and len(rendered[0][1].split("\n")) == len(qlabels):
            rendered = list(zip(qlabels, rendered[0][1].split("\n")))
        else:
            return OpenTarget(None, "label_mismatch")
    elif slabels and rendered[0][0] is None and len(slabels) == len(rendered) - 1:
        if re.search(r"(?i)\brozstrzygnij\b", question):     # "Tak, ...\nPrzykładowe uzasadnienie ..." w/o labels
            rendered = [("Rozstrzygnięcie", rendered[0][1])] + rendered[1:]
    elif slabels and len(slabels) != len(rendered):
        return OpenTarget(None, "label_mismatch")
    parts = []
    for lab, txt in rendered:
        if not txt:
            return OpenTarget(None, "empty_section")
        txt = txt if "\n" in txt else _finish(txt)
        parts.append(f"{lab}:{chr(10) if txt.startswith('1. ') else ' '}{txt}" if lab else txt)
    target = re.sub(r"\n{3,}", "\n\n", "\n".join(parts).strip())
    if len(target) > 4000:
        return OpenTarget(None, "too_long")
    return OpenTarget(target, None)


# ----------------------------------------------------------------------------- key/question consistency

_STOP5 = {"który", "która", "które", "którego", "której", "których", "został", "zosta", "przez", "podcza", "jedne",
          "jeden", "swoje", "swoją", "swoic", "także", "równi", "wówcz", "ponie", "któr", "źródł", "odpow", "uzasa",
          "rozst", "wyjaś", "infor", "tekst", "fragm", "opisa", "przed", "polsk", "wojny", "wojna", "wieku", "roku",
          "latac", "okres", "tylko", "miało", "miała", "było", "była", "były", "jednak", "jako", "oraz", "między"}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[^\W\d_]{5,}", text.lower())
    return {w[:5] for w in words} - _STOP5


def key_overlap(target: str, context: str) -> float:
    t = _tokens(target)
    return len(t & _tokens(context)) / len(t) if t else 1.0


# ----------------------------------------------------------------------------- papers, dedup, blocklist

_CODE = re.compile(r"^([A-Z]{3,4})-(R\d)(?:_\dP)?-(\d{3})(?:-[A-Z])?-(\d{4})", re.I)
_CODE_OLD = re.compile(r"^(MHI)-(R\d)_\dP-(\d{3})$", re.I)


def paper_id(arkusz: str) -> str:
    """Canonical paper name: CKE code ("MHIP-R0-100-2405") or the flat file stem ("historia-2024-maj-matura-...")."""
    stem = re.sub(r"(?i)\.pdf$", "", Path(arkusz).name)
    stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
    m = _CODE.match(stem)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}-{m.group(3)}-{m.group(4)}"
    m = _CODE_OLD.match(stem)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}-{m.group(3)}"
    return stem


def paper_formula(pid: str, year: int) -> str:
    if pid.startswith("MHIP"):
        return "2023"
    if pid.startswith(("EHIP", "MHI-")):
        return "2015"
    if "nowa-era" in pid:
        return "nowa_era"
    if "stara" in pid:
        return "2015"
    if year >= 2023:
        return "2023"
    return "2015" if year >= 2015 else "pre2015"


def norm_text(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def fingerprint(text: str, n: int = 400) -> str:
    return hashlib.sha1(norm_text(text)[:n].encode()).hexdigest()[:16]


def _grams(text: str, n: int = 4) -> set[str]:
    t = norm_text(text).replace(" ", "")
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cluster_papers(fps_by_paper: dict[str, set[str]], year_by_paper: dict[str, int], thr: float = 0.95) -> dict[str, str]:
    """Same-year papers sharing most question fingerprints are one paper (CKE code file vs arkusze.pl copy, "(1)")."""
    parent = {p: p for p in fps_by_paper}

    def find(p: str) -> str:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    papers = sorted(fps_by_paper)
    for i, a in enumerate(papers):
        for b in papers[i + 1:]:
            if year_by_paper[a] != year_by_paper[b]:
                continue
            fa, fb = fps_by_paper[a], fps_by_paper[b]
            if fa and fb and len(fa & fb) / min(len(fa), len(fb)) >= thr:
                parent[find(a)] = find(b)
    return {p: find(p) for p in papers}


def pick_canonical(members: list[str], keyed: dict[str, int]) -> str:
    """Prefer the official CKE code file, then the copy with more keyed tasks, then the shorter name."""
    return sorted(members, key=lambda p: (not bool(_CODE.match(p) or _CODE_OLD.match(p)), -keyed.get(p, 0), len(p), p))[0]


def load_mock_items(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        log.warning("mock exam %s not found; mock near-duplicate check skipped", p)
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("items", [])


# ----------------------------------------------------------------------------- pre-2015 papers

# Symbol-font glyphs pdftotext leaves in 2010-2014 keys (bullets, dashes, check marks).
_PUA_MAP = {"": "•", "": "•", "": "–", "": " ", "": "✓"}
_OLD_JUNK_LINES = [re.compile(p) for p in (
    r"(?i)wypełnia\b.*", r"(?i)egzaminator\s*!?", r"(?i)zdający\s*!?",
    r"(?:\d{1,2}\.(?:[A-F]\.)?\s*)*\d{1,2}\.[A-F]\.(?:\s*\d{1,2}\.(?:[A-F]\.)?)*",   # score boxes "30.A." "16.A. 16.B."
    r"\d{1,3}",                                                   # page numbers, score-box values
    r"(?i)(?:rozwiązania zadań i\s+)?schemat\w*\s+(?:punktowania|oceniania)\b.*",
    r"(?i)klucz\s+(?:punktowania|odpowiedzi)\b.*", r"(?i)kryteri\w*\s+oceniania\b.*",
    r"(?i)przykładowy zestaw zadań\b.*", r"(?i)arkusz\s+[IVX]+\b.*", r"(?i)czas pracy\b.*",
    r"(?i)zadanie rozszerzonej odpowiedzi\b.*", r"(?i)\(\s*wpisuje zdający.*",
    r"(?i)(?:\d{1,2}\.\s*)?wype\S{0,2}nia\b.*", r"(?i)arkusz odpowiedzi\b.*",            # "24. WYPE£NIA ZDAJ¥CY"
    r"(?i)nr\.?", r"(?i)zad\.", r"(?i)punkty", r"(?i)odpowied(?:zi|ź)\s*:?",               # score / answer boxes
    r"(?i)z nr pesel.*", r"(?i)kod zdaj\S*", r"SUMA", r"TEMAT:?", r"(?i)zadanie\s+\d{1,2}\.?",   # answer sheets
    r"(?:\d\s+){4,}\d", r"(?:\S\s){4,}\S",                     # "0 1 2 3 4 5", "W Y P E £ N I A"
    r"[¾•▪◦●*]", r"(?i)\(\s*\d\s*[-–]\s*\d{1,2}\s*(?:pkt|p)\.?\s*\)",          # symbol-font bullets, "(0-3 pkt.)"
)]
# Key-only furniture: requirement areas and skill descriptors ("Usytuowanie obszaru w przestrzeni (II 1)").
_OLD_KEY_JUNK = [re.compile(p) for p in (
    r"(?i)(?:korzystanie z informacji|tworzenie informacji|wiadomości i rozumienie)\.?",
    r".{0,160}\((?:I{1,3}|IV)(?:\s*\d{1,2})?(?:\s*[PR])?\)",
)]
_OLD_BOX_PAIR = re.compile(r"(?m)^[ \t]*\d{1,2}\.(?:[A-F]\.)?[ \t]*\n(?:[ \t]*\n)*[ \t]*\d{1,2}[ \t]*$\n?")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
_OLD_TOPIC = re.compile(r"^\s*Temat\s+(I{1,3}|IV|V)\b\.?\s*(.*)$")
_POLISH_DIACRITICS = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_MOJIBAKE = frozenset("ĊĞĪĔħĨĩ")          # 2011 poprawkowa: "SpoĞród", "wydarzeĔ", "ZwyciĊstwo"


def clean_old_layout(text: str | None, key: bool = False) -> str:
    """Pre-pass for 2003-2014 sheets before clean_layout: symbol-font glyphs, "Wypełnia egzaminator!" boxes, score-box
    markers ("30.A.", "16.A. 16.B."), lone page numbers, key page headers; with key=True also requirement areas and
    skill descriptors. Only used for old rows, so the 2015+ output is unchanged."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).replace("\r", "").replace("\f", "\n")
    for a, b in _PUA_MAP.items():
        t = t.replace(a, b)
    t = _OLD_BOX_PAIR.sub("", t)                                  # "12.\n1" / "13.A.\n2": task marker + score box
    t = re.sub(r"([^\W\d_])\.([ \t]*(?:[.…_][ \t]?){4,})", r"\1.\n\2", t)   # keep the period before an answer line
    pats = _OLD_JUNK_LINES + (_OLD_KEY_JUNK if key else [])
    out = []
    for line in t.split("\n"):
        s = re.sub(r"[ \t]+", " ", line).strip()
        if s and any(rx.fullmatch(s) for rx in pats):
            continue
        out.append(line)
    return "\n".join(out)


def renumber_old_topics(text: str) -> str:
    """Old essay sheets list "Temat I" / "Temat II" headers; the exam (and the "Temat N." answer) uses "1." / "2."."""
    lines = text.split("\n")
    if sum(bool(_OLD_TOPIC.match(ln)) for ln in lines) < 2:
        return text
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _OLD_TOPIC.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        rest, j = m.group(2).strip(), i + 1
        if not rest:
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                rest, j = lines[j].strip(), j + 1
        out.append(f"{_ROMAN[m.group(1)]}. {rest}".rstrip())
        i = j
    return "\n".join(out)


def diacritic_ratio(text: str) -> tuple[int, float]:
    letters = [c for c in text if c.isalpha()]
    return len(letters), (sum(c in _POLISH_DIACRITICS for c in letters) / len(letters) if letters else 0.0)


def mojibake_count(text: str) -> int:
    return sum(c in _MOJIBAKE for c in text)


def broken_encoding_papers(by_paper: dict[str, list[dict]], min_letters: int = 1500, min_ratio: float = 0.035) -> set[str]:
    """Paper files whose pdftotext lost the Polish letters ("staro ytnej", 2003) or mangled them ("SpoĞród", 2011).
    Polish prose has ~6% diacritics; every 2004-2026 paper in the dump is above 4.7%."""
    bad = set()
    for pkey, rs in by_paper.items():
        text = "\n".join((r.get("question") or "") + "\n" + (r.get("context") or "") for r in rs)
        n, ratio = diacritic_ratio(text)
        if (n >= min_letters and ratio < min_ratio) or mojibake_count(text) >= 10:
            bad.add(pkey)
    return bad


_OLD_SOL_HDR = re.compile(
    r"(?mi)^[ \t]*(?:poprawn\w*\s+odpowied\w*|prawidłow\w*\s+odpowied\w*|"
    r"przykład\w*\s+(?:poprawn\w*\s+|prawidłow\w*\s+)?(?:odpowied\w*|rozwiązani\w*|realizacj\w*))[ \t]*:?[ \t]*")
_OLD_POINTS = re.compile(r"(?mi)^[ \t]*(\d{1,2})\s*(?:p\.|pkt\.?|punkt\w*)\s*[–—-]")
_OLD_SOL_STOP = re.compile(
    r"(?mi)^[ \t]*(?:\d{1,2}\s*(?:p\.|pkt\.?|punkt\w*)\s*[–—-]|przykład\w*\s+(?:błędn|niepoprawn|nieprawidłow)|"
    r"uwag[ai]\b|za\s+(?:poprawn|prawidłow|każd))")
_OLD_PART = re.compile(r"(?m)^[ \t]*([A-F])\.?[ \t]*\(\s*0\s*[–-]\s*\d+\s*(?:pkt|p\.)?\s*\)[ \t]*$")


_FUNCTION_WORDS = frozenset("z w i na do od o a we ze się oraz lub by po za".split())


def join_word_runs(text: str) -> str:
    """pdftotext sometimes prints an old instruction one word per line ("Uporządkuj\\nchronologicznie\\nwydarzenia\\nz\\n
    historii\\nstarożytnej."): a run of >= 4 one-word lines that contains a function word is one line again. Lists of
    single-word items ("republika\\nkonsulat\\n...") have no function words and stay."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and lines[j].strip() and " " not in lines[j].strip():
            j += 1
            if re.search(r"[.:?!]$", lines[j - 1].strip()):
                break                                  # the instruction's sentence ends here
        run = [ln.strip() for ln in lines[i:j]]
        if len(run) >= 4 and any(w.lower() in _FUNCTION_WORDS for w in run):
            out.append(" ".join(run))
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def trim_old_sources(question: str, source_text: str) -> str:
    """Old papers print all sources of a section (Źródło A ... I) before its tasks; keep the ones the instruction
    names ("na podstawie źródła B"), as the exam item shows only its own sources."""
    blocks = source_blocks(source_text)
    labels = {b.label for b in blocks if b.label}
    refs: set[str] = set()
    for m in _SOURCE_LABEL_REF.finditer(question):
        refs.update(re.findall(r"\d+|[A-Z]", m.group(1)))
    if len(labels) < 2 or not refs or not refs < labels:
        return source_text
    keep = [b for b in blocks if b.label in refs]
    return "\n\n".join("\n".join([b.header, *b.lines]).strip() for b in keep).strip()


def old_source_scrambled(source_text: str) -> bool:
    """A source block that is a text dump of a diagram or genealogical table: mostly short lines, many with years or
    "&" (marriage) marks. Poems have short lines too but no digits."""
    for b in source_blocks(source_text):
        lines = [ln.strip() for ln in b.lines if ln.strip()]
        if len(lines) < 8:
            continue
        short = sum(len(ln) <= 25 for ln in lines)
        marked = sum(bool(re.search(r"\d|&", ln)) for ln in lines)
        if short >= 0.6 * len(lines) and marked >= 0.25 * len(lines):
            return True
    return False


def old_table_scrambled(text: str) -> bool:
    """An old sheet mentions a table and its lines are mostly cell fragments ("Rok", "Kraj", "1787", ...).
    Tables whose rows survived as sentences ("A. W starożytnej Grecji forma rządów ...") are kept."""
    if not TABLE_WORD.search(text):
        return False
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    short = [ln for ln in lines if len(ln) <= 25]
    return len(short) >= 4 and len(short) >= 0.45 * len(lines)


def inline_answers(question: str, key: str) -> str:
    """Answers written into a "filled-in sheet" key (2003-2008): the words the key adds to the question, one segment
    per line ("Podaj nazwisko adresata tego listu. Sikorski" -> "Sikorski"). "" when the key is not a copy of the
    sheet."""
    qa, ka = question.split(), key.split()
    if not qa or not ka:
        return ""
    sm = difflib.SequenceMatcher(None, qa, ka, autojunk=False)
    if sm.ratio() < 0.5:
        return ""
    segs = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            seg = " ".join(ka[j1:j2]).strip()
            if seg and re.search(r"[^\W\d_]", seg):
                segs.append(seg)
    return "\n".join(segs)


def parse_old_key(raw: str | None, question: str = "") -> ParsedKey:
    """Model answer + rubric of a 2003-2014 key.

    2009-2014 keys: per task (or per part "A. (0–1)") a "Poprawna odpowiedź:" / "Przykład(y) poprawnej odpowiedzi:"
    section followed by "N p. – ..." lines. 2003-2008 keys: the sheet itself with the answers written in (see
    inline_answers). A few 2014 próbna keys already use the 2015 layout (parse_key)."""
    if not raw:
        return ParsedKey("", "", None)
    t = clean_layout(clean_old_layout(raw, key=True))
    marks = list(_OLD_PART.finditer(t))
    blocks = ([(None, t)] if not marks else
              [(m.group(1), t[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(t))]) for i, m in enumerate(marks)])
    sols: list[tuple[str | None, str]] = []
    rubs: list[str] = []
    points: list[int] = []
    for label, text in blocks:
        m = _OLD_SOL_HDR.search(text)
        pm_all = [int(x.group(1)) for x in _OLD_POINTS.finditer(text)]
        if pm_all:
            points.append(max(pm_all))
        if not m:
            continue
        rest = text[m.end():]
        stop = _OLD_SOL_STOP.search(rest)
        sol = rest[:stop.start()] if stop else rest
        end = _SOLUTION_END.search(sol)
        sol = (sol[:end.start()] if end else sol).strip()
        pm = _OLD_POINTS.search(rest)
        if pm:
            rub = rest[pm.start():]
            cut = re.search(r"(?mi)^[ \t]*(?:przykład\w*|uwag[ai]\b)", rub)
            rubs.append(((f"{label}. " if label else "") + (rub[:cut.start()] if cut else rub).strip())[:1500])
        if sol:
            sols.append((label, sol))
    max_points = sum(points) if marks and points else (max(points) if points else None)
    if sols:
        solution = sols[0][1] if len(sols) == 1 and sols[0][0] is None else "\n".join(
            f"{lab}. {s}" if lab else s for lab, s in sols)
        return ParsedKey(rubric="\n".join(rubs).strip(), solution=solution.strip(), max_points=max_points)
    new = parse_key(clean_old_layout(raw, key=True))
    if new.solution:
        return new
    q = clean_layout(clean_old_layout(question))
    return ParsedKey(rubric="\n".join(rubs).strip(), solution=inline_answers(q, t), max_points=max_points)


# ----------------------------------------------------------------------------- nuori row -> exam item

def _group(task: str) -> str:
    return task.split(".")[0]


def _points_from_text(text: str) -> int | None:
    m = re.search(r"\(\s*(?:0\s*[–-]\s*)?(\d{1,2})\s*pkt\s*\)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?m)^\s*0((?:\s*[–-]\s*\d{1,2})+)\s*$", text)
    if m:
        return int(re.findall(r"\d{1,2}", m.group(1))[-1])
    return None


@dataclass
class Converted:
    item: dict
    trailing: str
    raw_points: int | None


def convert_row(row: dict, carry: str = "", old: bool = False) -> Converted:
    """nuori tasks.jsonl row -> organizers' exam item {id, group, max_points, question, source_text, images, answer_format}.
    old=True (2003-2014 sheets) adds the clean_old_layout pre-pass and renumbers "Temat I/II" essay topics."""
    task = str(row["task"])
    if old:
        ctx = join_word_runs(clean_layout(clean_old_layout(row.get("context") or "")))
        body = renumber_old_topics(join_word_runs(clean_layout(clean_old_layout(row.get("question") or ""))))
    else:
        ctx = clean_layout(row.get("context") or "")
        body = clean_layout(row.get("question") or "")
    body = re.split(r"(?m)^\s*Zadanie\s*\d+\.", body)[0]          # a leaked next-task header ends the task
    body = re.sub(r"\s*Zadanie\s*\d+\.\s*\(\s*\d.*$", "", body, flags=re.S)
    pre, question, trailing = split_source_question(body)
    parts = [p for p in (ctx, pre) if p]
    source = "\n\n".join(parts)
    if not source and carry and re.search(r"(?i)\bźród\w*|\btekst\w*|\bfragment\w*|\bmap\w*|\bilustracj\w*", question):
        source = carry
    points = row.get("max_points") or _points_from_text(row.get("question") or "")
    item = {
        "id": task,
        "group": int(_group(task)) if _group(task).isdigit() else _group(task),
        "max_points": points,
        "question": question,
        "source_text": source,
        "images": [],
        "answer_format": OPEN_FORMAT,
    }
    return Converted(item=item, trailing=trailing, raw_points=points)


# ----------------------------------------------------------------------------- answer syntax (filled + synthetic)

_CLOSED_LINE = {"closed_tf": re.compile(r"(\d{1,2}): ([PF])"), "closed_match": re.compile(r"([A-H]): (\d{1,2})"),
                "closed_multi_part": re.compile(r"(\d{1,2}): ([A-H])")}
_ESSAY_HEAD = re.compile(r"^Temat (\d{1,2})\.")


def answer_format_keys(kind: str, answer_format: str) -> list[str] | None:
    """Row keys of a closed answer_format ("1: P\\n2: F" -> ["1", "2"]; "A" -> []); None when it is not that syntax."""
    lines = [ln.strip() for ln in (answer_format or "").split("\n") if ln.strip()]
    if kind == "closed_single":
        return [] if len(lines) == 1 and re.fullmatch(r"[A-H]", lines[0]) else None
    ms = [_CLOSED_LINE[kind].fullmatch(ln) for ln in lines]
    if not lines or not all(ms):
        return None
    keys = [m.group(1) for m in ms]
    return keys if len(set(keys)) == len(keys) else None


def essay_topics(question: str) -> dict[int, str]:
    """Numbered topics of an essay question ("1. Temat ...", "2. ...") -> {number: text}."""
    marks = [(int(m.group(1)), m.start()) for m in re.finditer(r"(?m)^[ \t]*(\d{1,2})\.[ \t]+\S", question)]
    run, starts = 1, []
    for n, st in marks:
        if n == run:
            starts.append((n, st))
            run += 1
    out = {}
    for i, (n, st) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(question)
        out[n] = " ".join(re.sub(r"^\s*\d{1,2}\.\s*", "", question[st:end]).split())
    return out


def missing_labels(question: str, answer: str) -> list[str]:
    """Question labels ("Rozstrzygnięcie:", "Uzasadnienie:") that the answer does not use as "Label:" lines."""
    return [lab for lab in question_labels(question) if not re.search(rf"(?m)^\s*{re.escape(lab)}\s*:", answer)]


def answer_problem(kind: str, item: dict, answer: str, strict_labels: bool = True) -> str | None:
    """Why `answer` is not a valid target for `item` in the exam's answer syntax (None = valid).

    closed kinds: exactly the answer_format syntax with the same row keys (single: one option letter of the question);
    essay: starts with "Temat N." (N one of the listed topics), at least 300 words; open: non-empty, in Polish, and
    (strict_labels) every label left in the question used as a "Label:" line."""
    a = (answer or "").strip()
    if not a:
        return "empty"
    question = item.get("question") or ""
    if kind in CLOSED_KINDS:
        keys = answer_format_keys(kind, item.get("answer_format") or "")
        if keys is None:
            return "bad_answer_format"
        if kind == "closed_single":
            if not re.fullmatch(r"[A-H]", a):
                return "closed_syntax"
            opts = _OPTION_LINE.findall(question)
            return "closed_option" if len(opts) >= 2 and a not in opts else None
        ms = [_CLOSED_LINE[kind].fullmatch(ln.strip()) for ln in a.split("\n") if ln.strip()]
        if not all(ms):
            return "closed_syntax"
        return None if [m.group(1) for m in ms] == keys else "closed_keys"
    n, ratio = diacritic_ratio(a)
    if n >= 200 and ratio < 0.02:
        return "not_polish"
    if kind == "essay":
        m = _ESSAY_HEAD.match(a)
        if not m:
            return "essay_header"
        topics = essay_topics(question)
        if topics and int(m.group(1)) not in topics:
            return "essay_topic"
        if len(re.findall(r"[^\W_]+", a[m.end():])) < ESSAY_MIN_WORDS:        # words and numbers ("1791")
            return "essay_short"
        return None
    if strict_labels and missing_labels(question, a):
        return "labels_missing"
    return None


def expand_paths(patterns: Iterable[str | Path] | None, root: str | Path | None = None) -> list[Path]:
    """Paths and globs (relative to root) -> existing files, in the given order, globs sorted, no repeats."""
    out: list[Path] = []
    for pat in patterns or []:
        p = Path(pat)
        if root is not None and not p.is_absolute():
            p = Path(root) / p
        hits = sorted(Path(x) for x in glob.glob(str(p))) if glob.has_magic(str(p)) else ([p] if p.exists() else [])
        if not hits and not glob.has_magic(str(p)):
            log.warning("%s not found; skipped", p)
        for h in hits:
            if h.is_file() and h not in out:
                out.append(h)
    return out


def load_filled_files(patterns: Iterable[str | Path] | None, root: str | Path | None = None) -> tuple[dict[str, str], dict]:
    """{id: answer} from JSONL files of {"id", "answer"} (or {"id", "completion": [{"content"}]}) rows.

    Several files and globs; a later file overrides an earlier one for the same id. "SKIP" (and empty) answers are
    a filler's "cannot answer" and never become targets."""
    out: dict[str, str] = {}
    info = {"files": [], "rows": 0, "skipped": 0, "overridden": 0, "bad_json": 0}
    for path in expand_paths(patterns, root):
        info["files"].append(str(path))
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    info["bad_json"] += 1
                    continue
                ans = r.get("answer")
                if ans is None and r.get("completion"):
                    ans = r["completion"][-1].get("content")
                if not isinstance(ans, str) or not ans.strip() or ans.strip().upper() in FILLED_SKIP or "id" not in r:
                    info["skipped"] += 1
                    continue
                rid = str(r["id"])
                if rid in out and out[rid] != ans:
                    info["overridden"] += 1
                out[rid] = ans
                info["rows"] += 1
    info["ids"] = len(out)
    return out, info


# ----------------------------------------------------------------------------- synthetic items

SYNTHETIC_FIELDS = ("id", "kind", "question", "source_text", "answer_format", "max_points", "target")
# Essay themes of the mock paper's session (decentralisation of Poland, 11th-12th c.): never imitated by synthetic essays.
MOCK_ESSAY_GUARD = re.compile(
    r"(?i)rozbici\w*\s+dzielnicow|decentralizacj\w*|rozdrobnieni\w*\s+(?:feudaln|dzielnicow|polityczn)\w*|"
    r"podzia\w*\s+dzielnicow|testament\w*\s+(?:Bolesława\s+)?Krzywoust|kryzys\w*\s+monarchii\s+(?:wczesno)?piastowsk")


def load_synthetic_files(patterns: Iterable[str | Path] | None, root: str | Path | None = None) -> tuple[list[dict], dict]:
    """Synthetic item records from JSONL files (docs/DATA.md "Synthetic items"); each gets "_file" / "_line"."""
    records: list[dict] = []
    info = {"files": [], "bad_json": 0}
    for path in expand_paths(patterns, root):
        info["files"].append(str(path))
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    info["bad_json"] += 1
                    continue
                if not isinstance(rec, dict):
                    info["bad_json"] += 1
                    continue
                records.append({**rec, "_file": str(path), "_line": n})
    return records, info


def synthetic_task_id(syn_id: str, kind: str) -> str:
    """Exam-like task number for the prompt header ("Zadanie 12.2 (2 pkt)"), stable per synthetic id: prompts must
    look like exam prompts, never "Zadanie syn-open-03-12"."""
    h = int(hashlib.sha1(syn_id.encode("utf-8")).hexdigest()[:8], 16)
    if kind == "essay":
        return str(21 + h % 8)                        # the essay closes the paper
    group, sub = 1 + h % 24, (h >> 8) % 4
    return f"{group}.{sub}" if sub else str(group)


@dataclass
class SynItem:
    id: str
    kind: str
    origin: str
    item: dict
    target: str
    source_urls: list[str]


def validate_synthetic(rec: dict) -> SynItem | str:
    """A synthetic record -> SynItem, or the reason it is rejected (field types, answer_format, target syntax)."""
    for f in SYNTHETIC_FIELDS:
        if f not in rec:
            return f"missing_{f}"
    sid, kind = rec["id"], rec["kind"]
    if not isinstance(sid, str) or not sid.startswith("syn-"):
        return "bad_id"
    if kind not in KINDS:
        return "bad_kind"
    q, src, af, tgt = rec["question"], rec["source_text"], rec["answer_format"], rec["target"]
    if not all(isinstance(x, str) for x in (q, src, af, tgt)):
        return "bad_type"
    q, src, tgt = q.strip(), src.strip(), tgt.replace("\r", "").strip()
    pts = rec["max_points"]
    if isinstance(pts, bool) or not isinstance(pts, int) or not 1 <= pts <= 20:
        return "bad_max_points"
    if not q or not tgt:
        return "empty"
    if len(q) > 6000 or len(src) > 12000 or len(tgt) > (12000 if kind == "essay" else 4000):
        return "too_long"
    origin = rec.get("origin") or ("synthetic-essay" if kind == "essay" else "synthetic-open")
    if origin not in SYNTHETIC_ORIGINS:
        return "bad_origin"
    if (kind == "essay") != (origin == "synthetic-essay"):
        return "origin_kind_mismatch"
    urls = rec.get("source_urls") or []
    if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
        return "bad_source_urls"
    if kind == "open" and af != OPEN_FORMAT:
        return "bad_answer_format"
    if kind == "essay" and af != ESSAY_FORMAT:
        return "bad_answer_format"
    if kind in CLOSED_KINDS:
        keys = answer_format_keys(kind, af)
        if keys is None:
            return "bad_answer_format"
        af = "\n".join(ln.strip() for ln in af.split("\n") if ln.strip())
        if kind == "closed_single" and len(_OPTION_LINE.findall(q)) < 2:
            return "no_options"
        if kind == "closed_tf" and len(_statement_numbers(q)) != len(keys):
            return "tf_statement_count"
        if kind == "closed_multi_part" and len(_statement_numbers(q)) < len(keys):
            return "multi_part_count"
        if kind == "closed_match":
            text = q + "\n" + src
            for k in keys:
                if not re.search(rf"(?m)^\s*{k}[.)]\s|\b(?:Fragment|Tekst|Źródło|Opis|Postać|Wydarzenie|Dokument)\s+{k}\b", text):
                    return "match_rows"
    task = synthetic_task_id(sid, kind)
    item = {"id": task, "group": int(task.split(".")[0]), "max_points": pts, "question": q, "source_text": src,
            "images": [], "answer_format": af}
    problem = answer_problem(kind, item, tgt, strict_labels=True)
    if problem:
        return problem
    return SynItem(id=sid, kind=kind, origin=origin, item=item, target=tgt, source_urls=list(urls))


@dataclass
class _Ref:
    """4-gram sets of an item's question / source (and essay topics) for near-duplicate checks."""
    qg: set
    sg: set
    tag: str
    topics: list[set] = field(default_factory=list)


def _ref(question: str, source: str, tag: str, essay: bool = False) -> _Ref:
    return _Ref(_grams(question), _grams(source[:1500]), tag, [_grams(t) for t in essay_topics(question).values()] if essay else [])


def _similar(a: _Ref, b: _Ref, q_thr: float, s_thr: float, q_only: float) -> bool:
    """Questions match at q_thr and sources at s_thr; a question without a source needs q_only."""
    if not a.qg or not b.qg:
        return False
    la, lb = len(a.qg), len(b.qg)
    if min(la, lb) < min(q_thr, q_only) * max(la, lb):         # Jaccard upper bound
        return False
    j = jaccard(a.qg, b.qg)
    if a.sg and b.sg:
        return j >= q_thr and jaccard(a.sg, b.sg) >= s_thr
    return j >= q_only


def _same_source(a: _Ref, b: _Ref, thr: float = 0.5) -> bool:
    if len(a.sg) < 150 or len(b.sg) < 150 or min(len(a.sg), len(b.sg)) < thr * max(len(a.sg), len(b.sg)):
        return False
    return jaccard(a.sg, b.sg) >= thr


def _same_topic(a: _Ref, b: _Ref, thr: float) -> bool:
    return any(jaccard(x, y) >= thr for x in a.topics for y in b.topics)


def _gram_matrices(groups: list[list[set]]):
    """Binary CSR matrices (one per group of gram sets) over a shared vocabulary, plus each row's set size."""
    import numpy as np
    from scipy import sparse

    vocab: dict[str, int] = {}
    built = []
    for sets in groups:
        indptr, indices = [0], []
        for st in sets:
            indices.extend(vocab.setdefault(g, len(vocab)) for g in st)
            indptr.append(len(indices))
        built.append((indptr, indices, np.array([len(st) for st in sets], dtype=np.float32)))
    out = []
    for indptr, indices, sizes in built:
        m = sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), np.array(indices, dtype=np.int64),
                               np.array(indptr, dtype=np.int64)), shape=(len(indptr) - 1, max(1, len(vocab))))
        out.append((m, sizes))
    return out


def near_duplicate_matches(queries: list[_Ref], refs: list[_Ref], q_thr: float, s_thr: float, q_only: float,
                           src_only: float | None = None, chunk: int = 256) -> list[list[int]]:
    """For each query, the indices of refs it near-duplicates (4-gram Jaccard, sparse matrix products).

    A pair matches when both questions are non-empty and: with both sources present, question >= q_thr and source
    >= s_thr; otherwise question >= q_only. src_only (mock): sources of >= 150 grams matching at src_only alone."""
    import numpy as np

    out: list[list[int]] = [[] for _ in queries]
    if not queries or not refs:
        return out
    (qa, qa_n), (sa, sa_n), (qb, qb_n), (sb, sb_n) = _gram_matrices(
        [[r.qg for r in queries], [r.sg for r in queries], [r.qg for r in refs], [r.sg for r in refs]])
    qbt, sbt = qb.T.tocsc(), sb.T.tocsc()
    for st in range(0, len(queries), chunk):
        sl = slice(st, st + chunk)
        iq = (qa[sl] @ qbt).toarray()
        jq = iq / np.maximum(qa_n[sl, None] + qb_n[None, :] - iq, 1.0)
        i_s = (sa[sl] @ sbt).toarray()
        js = i_s / np.maximum(sa_n[sl, None] + sb_n[None, :] - i_s, 1.0)
        both_q = (qa_n[sl, None] > 0) & (qb_n[None, :] > 0)
        both_s = (sa_n[sl, None] > 0) & (sb_n[None, :] > 0)
        hit = both_q & np.where(both_s, (jq >= q_thr) & (js >= s_thr), jq >= q_only)
        if src_only is not None:
            hit |= (sa_n[sl, None] >= 150) & (sb_n[None, :] >= 150) & (js >= src_only)
        for i, j in zip(*np.nonzero(hit)):
            out[st + int(i)].append(int(j))
    return out


def synthetic_duplicates(syn: list[_Ref], cke: list[_Ref], mock: list[_Ref], essay_questions: list[str]) -> list[str | None]:
    """Why each synthetic item is a near-duplicate (None = keep), in order; earlier kept items win.

    mock (paper MHIP/EHIP-R0-100-2305 rows and the organizers' exam.json): question >= 0.5 with source >= 0.4, a
    question alone >= 0.6, the same source text (>= 0.5), an essay topic >= 0.4, or a synthetic essay on the mock
    session's essay theme (MOCK_ESSAY_GUARD). CKE rows (every converted item): question >= 0.6 with source >= 0.5
    (question alone >= 0.75), essay topic >= 0.6; "dup_dev" when a dev item matches. Other synthetic items: question
    >= 0.8 with source >= 0.8 (question alone >= 0.9)."""
    reasons: list[str | None] = [None] * len(syn)
    mock_hits = near_duplicate_matches(syn, mock, 0.5, 0.4, 0.6, src_only=0.5)
    cke_hits = near_duplicate_matches(syn, cke, 0.6, 0.5, 0.75)
    self_hits = near_duplicate_matches(syn, syn, 0.8, 0.8, 0.9)
    kept = [False] * len(syn)
    for i, s in enumerate(syn):
        if essay_questions[i] and MOCK_ESSAY_GUARD.search(essay_questions[i]):
            reasons[i] = "dup_mock_topic"
        elif mock_hits[i] or (s.topics and any(_same_topic(s, r, 0.4) for r in mock if r.topics)):
            reasons[i] = "dup_mock"
        else:
            tags = [cke[j].tag for j in cke_hits[i]]
            if s.topics:
                tags += [r.tag for r in cke if r.topics and _same_topic(s, r, 0.6)]
            if tags:
                reasons[i] = "dup_dev" if "dev" in tags else "dup_cke"
            elif any(kept[j] for j in self_hits[i] if j < i):
                reasons[i] = "dup_synthetic"
        kept[i] = reasons[i] is None
    return reasons


# ----------------------------------------------------------------------------- RAG-style rendering

def path_fingerprint(path: str | Path) -> str:
    """16-hex sha256 of a file, or of every file under a directory (relative names + contents)."""
    p = Path(path)
    h = hashlib.sha256()
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    for f in files:
        h.update(str(f.relative_to(p) if p.is_dir() else f.name).encode("utf-8"))
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class RagContext:
    """Retriever + passage formatter used for style="rag" (harness.rag, or fakes in tests)."""
    index: object                       # .search(query, k) -> [{"title", "section", "text", "url", "score"}]
    build_query: Callable               # item -> query text
    format_passages: Callable           # (passages, max_chars=...) -> text
    k: int = 4
    fraction: float = 0.85              # share of TRAIN rows rendered with passages (the rest: rag prompt, no passages)
    seed: int = 0
    max_chars: int = 2400
    index_path: str | None = None
    index_hash: str | None = None

    def retrieve(self, item: dict) -> tuple[str, list[dict]]:
        hits = list(self.index.search(self.build_query(item), k=self.k) or [])
        text = self.format_passages(hits, max_chars=self.max_chars) if hits else ""
        return (text or "").strip(), hits


def make_rag_context(index_path: str | Path, k: int = 4, fraction: float = 0.85, seed: int = 0,
                     max_chars: int = 2400) -> RagContext:
    """Load the RAG index with harness.rag (lazy import: the builder works without it for the other styles)."""
    try:
        from harness import rag as R  # noqa: N812 - lazy: the RAG builder owns harness/rag.py
    except ImportError as e:  # pragma: no cover - depends on harness/rag.py being present
        raise RuntimeError("--style rag needs harness/rag.py (load_index, build_query, format_passages)") from e
    if not Path(index_path).exists():
        raise FileNotFoundError(f"RAG index {index_path} not found (build it with the RAG builder first)")
    return RagContext(index=R.load_index(index_path), build_query=R.build_query, format_passages=R.format_passages,
                      k=k, fraction=fraction, seed=seed, max_chars=max_chars, index_path=str(index_path),
                      index_hash=path_fingerprint(index_path))


def select_fraction(ids: Iterable[str], fraction: float, seed: int) -> set[str]:
    """Exactly round(fraction * n) ids, chosen by a seeded hash: deterministic and independent of row order."""
    ids = list(ids)
    order = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode("utf-8")).hexdigest())
    return set(order[:round(max(0.0, min(1.0, fraction)) * len(order))])


# ----------------------------------------------------------------------------- build

@dataclass
class Result:
    status: str                    # "ok" | "needs_answer" | "drop"
    kind: str
    target: str | None = None
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def make_target(item: dict, key: ParsedKey, essay_by_points: bool = True) -> Result:
    """Decide kind + answer_format (mutates item) and derive the target, or say why it needs an answer.
    essay_by_points=False: only the wording marks an essay (old papers have multi-part tasks worth 10+ points)."""
    q, sol = item["question"], key.solution
    if is_essay(q, item.get("max_points") if essay_by_points else None):
        item["answer_format"] = ESSAY_FORMAT
        return Result("needs_answer", "essay", reason="essay")
    closed = detect_closed(q, sol)
    if closed:
        item["answer_format"] = closed.answer_format
        if closed.question:
            item["question"] = closed.question
        if closed.target is None:
            return Result("needs_answer", closed.kind, reason=closed.problem)
        return Result("ok", closed.kind, target=closed.target)
    item["answer_format"] = OPEN_FORMAT
    letter = None
    options = _OPTION_LINE.findall(q)
    if len(options) >= 2 and _CHOICE_Q.search(q) and _JUSTIFY_Q.search(q):
        first = next((ln for ln in sol.split("\n") if ln.strip()), "")
        m = _LETTER_ONLY.match(first)
        if m and m.group(1) in options:
            letter = m.group(1)
            sol = sol.split("\n", 1)[1] if "\n" in sol else ""
    ot = build_open_target(q, sol, choice_letter=letter)
    if ot.target is None:
        return Result("needs_answer", "open", reason=ot.problem)
    return Result("ok", "open", target=ot.target)


def _default_prompt_fns() -> tuple[Callable, Callable | None]:
    try:
        from harness.prompts import build_messages  # lazy: builder A owns harness/
    except ImportError as e:  # pragma: no cover - depends on the harness being present
        raise RuntimeError("harness.prompts.build_messages is not available yet; run with --stats-only or add harness/") from e
    try:
        from harness.prompts import item_kind
    except ImportError:  # pragma: no cover
        item_kind = None
    return build_messages, item_kind


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_nuori_tasks(nuori_dir: str | Path) -> list[dict]:
    path = Path(nuori_dir) / "tasks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (clone https://github.com/DriversLab/nuori-ai and point --nuori-dir at it)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def default_nuori_dir(repo_root: Path) -> Path:
    return Path(os.environ.get("NUORI_DIR") or (repo_root.parent / "nuori-ai"))


def build_history_data(
    rows: list[dict],
    out_dir: str | Path | None = None,
    *,
    min_year: int = DEFAULT_MIN_YEAR,
    mock_items: list[dict] | None = None,
    filled_answers: dict[str, str] | None = None,
    build_messages: Callable | None = None,
    item_kind: Callable | None = None,
    stats_only: bool = False,
    overlap_threshold: float = -1.0,
    style: str = "organizer",
    synthetic: list[dict] | None = None,
    rag: RagContext | None = None,
    keyless_filled: str = "old",
) -> dict:
    """Convert nuori rows (plus synthetic items) into train/dev/needs_answer rows. Returns stats (and writes files
    unless stats_only).

    `style` is the harness.prompts style; it must equal run_exam.py --style at serve time (default "organizer").
    style="rag" needs `rag` (make_rag_context): rag.fraction of the TRAIN rows get retrieved passages, the rest the rag
    prompt without passages; dev is written twice (dev.jsonl organizer, dev_rag.jsonl with passages).
    Rows from papers older than 2015 (min_year < 2015) are converted but become training rows only with a filled
    target (flag llm_filled_old); otherwise they go to needs_answer with reason "old_paper".
    keyless_filled: filled targets of rows whose paper has no answer key are trained for "old" rows (default: the
    filler is their only target source anyway), "all" rows or "none"; otherwise such rows are dropped (no_answer_key).
    `synthetic`: records in the synthetic item format (load_synthetic_files); validated, deduplicated against every
    converted CKE item, the mock paper and each other, and added to train."""
    mock_items = mock_items or []
    filled_answers = filled_answers or {}
    synthetic = synthetic or []
    if keyless_filled not in ("none", "old", "all"):
        raise ValueError(f"keyless_filled must be none, old or all (got {keyless_filled!r})")
    if style == "rag" and rag is None and not stats_only:
        raise ValueError("style='rag' needs a RagContext (make_rag_context(index_path, ...))")
    if not stats_only and build_messages is None:
        build_messages, item_kind = _default_prompt_fns()
    if style == "rag" and not stats_only and "passages" not in inspect.signature(build_messages).parameters:
        raise RuntimeError("harness.prompts.build_messages has no passages= parameter (needed for --style rag)")

    stats: dict = {"input_rows": len(rows), "drop": Counter(), "needs_answer": Counter(), "flags": Counter(),
                   "split": Counter(), "per_year": defaultdict(Counter), "per_kind": defaultdict(Counter),
                   "per_formula": defaultdict(Counter), "item_kind_mismatch": 0,
                   "per_origin": defaultdict(Counter), "per_origin_kind": defaultdict(Counter), "filled": Counter()}
    dropped: list[dict] = []

    def drop(rid: str, reason: str, nuori_id: str | None = None) -> None:
        stats["drop"][reason] += 1
        dropped.append({"id": rid, "reason": reason, "nuori_id": nuori_id})

    # ---- 1) papers: canonical ids, same-paper clusters, mock-paper exclusion
    by_paper: dict[str, list[dict]] = defaultdict(list)
    year_by_paper: dict[str, int] = {}
    for r in rows:
        pid = paper_id(r["arkusz"])
        key = f"{pid}@@{r['arkusz']}"             # the file (a "(1)" copy has the same pid but is a separate file)
        by_paper[key].append(r)
        year_by_paper[key] = int(r["year"])
    fps = {k: {fingerprint(clean_layout(r.get("question") or ""), 100_000) for r in v if (r.get("question") or "").strip()}
           for k, v in by_paper.items()}
    clusters = cluster_papers(fps, year_by_paper)
    members: dict[str, list[str]] = defaultdict(list)
    for k, root in clusters.items():
        members[root].append(k)
    keyed = {k: sum(bool(r.get("answer_key")) for r in v) for k, v in by_paper.items()}
    canonical_of: dict[str, str] = {}
    excluded_papers: set[str] = set()
    for root, ms in members.items():
        canon = pick_canonical(ms, keyed)
        is_mock = any(k.split("@@")[0] in MOCK_PAPER_CODES for k in ms)
        for k in ms:
            canonical_of[k] = canon
            if is_mock:
                excluded_papers.add(k)
    broken_papers = broken_encoding_papers(by_paper)
    stats["papers"] = {"files": len(by_paper), "distinct": len(members),
                       "duplicate_files": sorted(k.split("@@")[1] for k in by_paper if canonical_of[k] != k),
                       "excluded_mock_files": sorted(k.split("@@")[1] for k in excluded_papers),
                       "broken_encoding_files": sorted(k.split("@@")[1] for k in broken_papers)}

    mock_q = [(it, _grams(it.get("question", "")), _grams(it.get("source_text", "")[:1500])) for it in mock_items]
    blocklist = {"mock_source_exam_ids": list(MOCK_PAPER_CODES), "excluded_paper_files": stats["papers"]["excluded_mock_files"],
                 "nuori_task_ids": [], "row_ids": [], "question_fingerprints": sorted(
                     {fingerprint(it.get("question", ""), 300) for it in mock_items}), "mock_near_duplicates": []}
    # near-duplicate references for synthetic items: the mock (exam.json + its paper rows) and every converted CKE item
    mock_refs = [_ref(it.get("question", ""), it.get("source_text", ""), "mock",
                      essay=is_essay(it.get("question", ""), it.get("max_points"))) for it in mock_items]
    cke_refs: list[_Ref] = []

    # ---- 2) convert rows (per paper, in task order: formula-2015 sources can trail the previous task)
    out = {"train": [], "dev": [], "needs": []}
    seen_task: set[tuple[str, str]] = set()
    seen_q: dict[str, str] = {}                  # fingerprint(question+source) -> split of the kept row
    candidates: list[tuple[str, dict]] = []
    for pkey in sorted(by_paper, key=lambda k: (year_by_paper[k], k)):
        pid = pkey.split("@@")[0]
        year = year_by_paper[pkey]
        is_old = year < DEFAULT_MIN_YEAR
        carry = ""
        for r in by_paper[pkey]:
            rid = f"{pid}:{r['task']}"
            if pkey in excluded_papers:
                drop(rid, "mock_paper", r["id"])
                blocklist["nuori_task_ids"].append(r["id"])
                blocklist["row_ids"].append(rid)
                blocklist["question_fingerprints"].append(fingerprint(clean_layout(r.get("question") or ""), 300))
                mi = convert_row(r).item                 # in memory only: a blocklist for synthetic items
                mock_refs.append(_ref(mi["question"], mi["source_text"], "mock", essay=is_essay(mi["question"], mi["max_points"])))
                continue
            if canonical_of[pkey] != pkey:
                drop(rid, "duplicate_paper_copy", r["id"])
                continue
            if year < min_year:
                drop(rid, "pre_min_year", r["id"])
                continue
            if year > TRAIN_MAX_YEAR and year not in DEV_YEARS:
                drop(rid, "year_out_of_range", r["id"])
                continue
            if pkey in broken_papers or mojibake_count((r.get("question") or "") + (r.get("context") or "")) >= 2:
                drop(rid, "broken_encoding", r["id"])
                continue
            if (pid, str(r["task"])) in seen_task:
                drop(rid, "duplicate_task_id", r["id"])
                continue
            seen_task.add((pid, str(r["task"])))
            conv = convert_row(r, carry, old=is_old)
            if conv.trailing:
                carry = conv.trailing
            item = conv.item
            split = "dev" if year in DEV_YEARS else "train"
            candidates.append((split, {"rid": rid, "row": r, "item": item, "pid": pid, "year": year, "old": is_old}))

    # ---- 3) filter + targets; dev rows first (a train duplicate of a dev item is the one dropped), 2015+ before old
    candidates.sort(key=lambda c: (c[0] != "dev", c[1]["old"], c[1]["year"], c[1]["rid"]))
    for split, c in candidates:
        rid, r, item, pid, year, is_old = c["rid"], c["row"], c["item"], c["pid"], c["year"], c["old"]

        def drop(rid: str, reason: str, _nid: str = r["id"]) -> None:   # noqa: F811 - same helper, nuori id bound
            stats["drop"][reason] += 1
            dropped.append({"id": rid, "reason": reason, "nuori_id": _nid})

        if not item["question"].strip():
            drop(rid, "empty_question")
            continue
        cke_refs.append(_ref(item["question"], item["source_text"], split,
                             essay=is_essay(item["question"], item["max_points"])))
        qfp = fingerprint(item["question"], 400) + fingerprint(item["source_text"], 600)
        if qfp in seen_q:
            drop(rid, "duplicate_question" if seen_q[qfp] == split else "duplicate_of_dev")
            continue
        if mock_q:
            qg, sg = _grams(item["question"]), _grams(item["source_text"][:1500])
            hit = next((it for it, mqg, msg in mock_q if jaccard(qg, mqg) >= 0.6 and (not msg or not sg or jaccard(sg, msg) >= 0.5)), None)
            if hit is not None or fingerprint(item["question"], 300) in blocklist["question_fingerprints"]:
                drop(rid, "mock_near_duplicate")
                blocklist["mock_near_duplicates"].append(rid)
                continue
        seen_q[qfp] = split
        key = parse_old_key(r.get("answer_key"), r.get("question") or "") if is_old else parse_key(r.get("answer_key"))
        if item["max_points"] is None and key.max_points:
            item["max_points"] = key.max_points
        filled = filled_answers.get(rid)
        keyless_ok = keyless_filled == "all" or (keyless_filled == "old" and is_old)
        if not r.get("answer_key") and not (keyless_ok and filled):
            if filled:
                stats["filled"]["ignored_keyless"] += 1
            drop(rid, "no_answer_key")
            continue
        essay = is_essay(item["question"], None if is_old else item["max_points"])
        if is_old and not essay:
            item["source_text"] = trim_old_sources(item["question"], item["source_text"])
        pc = picture_check(item["question"], item["source_text"]) if not essay else PictureCheck(False, None, False, [])
        if pc.dependent:
            drop(rid, f"picture:{pc.reason}")
            continue
        if not essay and missing_source(item["question"], item["source_text"]):
            drop(rid, "missing_source")
            continue
        if is_old and not essay and (old_table_scrambled(item["question"] + "\n" + item["source_text"])
                                     or old_source_scrambled(item["source_text"])):
            drop(rid, "old_table")                 # a pre-2015 table / diagram that pdftotext cut into cells
            continue
        flags = list(pc.flags)
        if pc.has_visual:
            item["source_text"] = mark_images(item["source_text"], str(item["group"]))
        res = make_target(item, key, essay_by_points=not is_old)
        res.flags = flags
        auto: Result | None = None
        if is_old:
            if res.kind == "closed_tf" and res.status != "ok":
                n = len(_statement_numbers(item["question"]))
                if n >= 2:                               # numbered "prawda/fałsz" statements: the exam's P/F syntax
                    item["answer_format"] = tf_format(n)
                else:                                    # lettered statements / lost layout: an open answer
                    item["answer_format"] = OPEN_FORMAT
                    res = Result(res.status, "open", res.target, res.reason, flags)
            auto = res
            res = Result("needs_answer", res.kind, reason="old_paper", flags=flags)
        elif res.status == "ok" and res.kind == "open":
            ov = key_overlap(res.target or "", item["question"] + "\n" + item["source_text"])
            if len(_tokens(res.target or "")) >= 8 and ov <= overlap_threshold:
                res = Result("needs_answer", "open", reason="key_mismatch_suspect", flags=flags)
        filled_problem = None
        if filled and res.status == "needs_answer":
            filled_problem = answer_problem(res.kind, item, filled, strict_labels=False)
            if filled_problem is None:
                extra = ["llm_filled_old" if is_old else "llm_filled"]
                if missing_labels(item["question"], filled):
                    extra.append("labels_missing")
                if not r.get("answer_key"):
                    extra.append("keyless")
                res = Result("ok", res.kind, target=filled.strip(), flags=flags + extra)
                stats["filled"]["used_old" if is_old else "used"] += 1
            else:
                stats["filled"][f"rejected:{filled_problem}"] += 1
        if is_old and item["max_points"] is None and res.status == "ok":
            # every exam prompt shows "(N pkt)": estimate 1 point per lettered part ("A. Podaj …"), at least 1
            parts = [m for m in re.findall(r"(?m)^\s*(?:[A-F]\.|[a-f]\))\s+(\S.*)$", item["question"] + "\n" + item["source_text"])
                     if _POLECENIE.match(m)]
            item["max_points"] = max(1, len(parts))
            res.flags.append("points_estimated")
        formula = paper_formula(pid, year)
        origin = "cke-old" if is_old else ("llm-filled" if "llm_filled" in res.flags else "cke-2015+")
        if item["max_points"] is None:
            stats["flags"]["no_max_points"] += 1
        base = {"id": rid, "paper": pid, "year": year, "formula": formula, "task": str(r["task"]), "kind": res.kind,
                "points": item["max_points"], "rubric": key.rubric, "flags": res.flags, "source": "nuori-ai",
                "origin": origin, "level": r.get("level")}
        for fl in res.flags:
            stats["flags"][fl] += 1
        if res.status == "needs_answer":
            stats["needs_answer"][res.reason] += 1
            stats["per_kind"][res.kind]["needs_answer"] += 1
            need = {**base, "split": split, "reason": res.reason, "item": item, "prompt": None,
                    "key_solution": key.solution, "labels": question_labels(item["question"]),
                    "hint": NEEDS_HINTS.get(res.kind, NEEDS_HINTS["open"])}
            if is_old:
                need.update({"auto_target": auto.target if auto else None, "auto_problem": auto.reason if auto else None,
                             "key_text": clean_layout(clean_old_layout(r.get("answer_key") or "", key=True))[:4000]})
            if filled_problem:
                need["filled_rejected"] = filled_problem
            out["needs"].append(need)
            continue
        stats["split"][split] += 1
        stats["per_year"][str(year)][res.kind] += 1
        stats["per_kind"][res.kind][split] += 1
        stats["per_formula"][formula][split] += 1
        stats["per_origin"][origin][split] += 1
        stats["per_origin_kind"][origin][res.kind] += 1
        out[split].append({"prompt": None, "completion": [{"role": "assistant", "content": res.target}],
                           **base, "item": item})

    # ---- 4) synthetic items: validate, deduplicate (mock, CKE, each other), add to train
    syn = {"records": len(synthetic), "kept": 0, "invalid": Counter(), "duplicates": Counter(), "per_kind": Counter()}
    valid: list[tuple[SynItem, dict]] = []
    seen_syn: set[str] = set()
    for rec in synthetic:
        v = validate_synthetic(rec)
        where = {"file": rec.get("_file"), "line": rec.get("_line")}
        if isinstance(v, str) or v.id in seen_syn:
            reason = v if isinstance(v, str) else "duplicate_id"
            syn["invalid"][reason] += 1
            dropped.append({"id": rec.get("id"), "reason": f"synthetic_invalid:{reason}", "nuori_id": None, **where})
            continue
        seen_syn.add(v.id)
        valid.append((v, where))
    syn_refs = [_ref(v.item["question"], v.item["source_text"], "synthetic", essay=v.kind == "essay") for v, _ in valid]
    dups = synthetic_duplicates(syn_refs, cke_refs, mock_refs,
                                [v.item["question"] if v.kind == "essay" else "" for v, _ in valid])
    for (v, where), dup in zip(valid, dups):
        if dup:
            syn["duplicates"][dup] += 1
            dropped.append({"id": v.id, "reason": f"synthetic_{dup}", "nuori_id": None, **where})
            if dup.startswith("dup_mock"):
                blocklist["mock_near_duplicates"].append(v.id)
            continue
        syn["kept"] += 1
        syn["per_kind"][v.kind] += 1
        stats["split"]["train"] += 1
        stats["per_kind"][v.kind]["train"] += 1
        stats["per_formula"]["synthetic"]["train"] += 1
        stats["per_origin"][v.origin]["train"] += 1
        stats["per_origin_kind"][v.origin][v.kind] += 1
        stats["flags"][v.origin] += 1
        out["train"].append({"prompt": None, "completion": [{"role": "assistant", "content": v.target}],
                             "id": v.id, "paper": "synthetic", "year": None, "formula": "synthetic", "task": v.item["id"],
                             "kind": v.kind, "points": v.item["max_points"], "rubric": "", "flags": [v.origin],
                             "source": "synthetic", "origin": v.origin, "level": None, "source_urls": v.source_urls,
                             "item": v.item})
    stats["synthetic"] = syn

    # ---- 5) prompts: the exam-time renderer (train/serve consistency)
    base_style = "organizer" if style == "rag" else style
    dev_rag: list[dict] = []
    rag_stats: dict | None = None

    def check_kind(row: dict) -> None:
        if item_kind is not None and item_kind(row["item"]) != row["kind"]:
            stats["item_kind_mismatch"] += 1

    def rag_meta(hits: list[dict], used: bool) -> dict:
        return {"with_passages": used, "titles": [str(h.get("title", "")) for h in hits] if used else []}

    if not stats_only:
        if style == "rag":
            chosen = select_fraction([r["id"] for r in out["train"]], rag.fraction, rag.seed)
            rag_stats = {"index_path": rag.index_path, "index_hash": rag.index_hash, "k": rag.k,
                         "fraction": rag.fraction, "seed": rag.seed, "max_chars": rag.max_chars,
                         "train_with_passages": 0, "train_without_passages": 0, "empty_retrievals": 0, "dev_rag_rows": 0}
            for r in out["train"]:
                text, hits = rag.retrieve(r["item"]) if r["id"] in chosen else ("", [])
                if r["id"] in chosen and not text:
                    rag_stats["empty_retrievals"] += 1
                r["prompt"] = build_messages(r["item"], None, style="rag", passages=text or None)
                r["rag"] = rag_meta(hits, bool(text))
                rag_stats["train_with_passages" if text else "train_without_passages"] += 1
                check_kind(r)
            for r in out["dev"]:
                r["prompt"] = build_messages(r["item"], None, style=base_style)
                check_kind(r)
                text, hits = rag.retrieve(r["item"])
                rag_stats["empty_retrievals"] += 0 if text else 1
                dev_rag.append({**r, "prompt": build_messages(r["item"], None, style="rag", passages=text or None),
                                "rag": rag_meta(hits, bool(text))})
            rag_stats["dev_rag_rows"] = len(dev_rag)
        else:
            for r in out["train"] + out["dev"]:
                r["prompt"] = build_messages(r["item"], None, style=style)
                check_kind(r)
        for r in out["needs"]:
            r["prompt"] = build_messages(r["item"], None, style=base_style)
            check_kind(r)

    stats["needs_answer_split"] = dict(Counter(n["split"] for n in out["needs"]))
    stats["needs_answer_origin"] = dict(Counter(n["origin"] for n in out["needs"]))
    stats["prompt_builder"] = None if stats_only else f"{build_messages.__module__}.{build_messages.__name__}"
    stats["prompt_style"] = style
    stats["rag"] = rag_stats
    stats["min_year"] = min_year
    stats["include_old"] = min_year < DEFAULT_MIN_YEAR
    stats["keyless_filled"] = keyless_filled
    stats["dev_years"] = list(DEV_YEARS)
    stats["train_max_year"] = TRAIN_MAX_YEAR
    stats["mock_items_checked"] = len(mock_items)
    stats["mock_refs_for_synthetic"] = len(mock_refs)
    blocklist["question_fingerprints"] = sorted(set(blocklist["question_fingerprints"]))
    stats = json.loads(json.dumps(stats, default=dict))   # Counters/defaultdicts -> plain dicts
    stats["blocklist"] = {"rows": len(blocklist["row_ids"]), "files": len(blocklist["excluded_paper_files"]),
                          "mock_near_duplicates": len(blocklist["mock_near_duplicates"])}

    if not stats_only and out_dir is not None:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        stats["written"] = {
            "train.jsonl": _write_jsonl(od / "train.jsonl", out["train"]),
            "dev.jsonl": _write_jsonl(od / "dev.jsonl", out["dev"]),
            "needs_answer.jsonl": _write_jsonl(od / "needs_answer.jsonl", out["needs"]),
            "dropped.jsonl": _write_jsonl(od / "dropped.jsonl", dropped),
        }
        if style == "rag":
            stats["written"]["dev_rag.jsonl"] = _write_jsonl(od / "dev_rag.jsonl", dev_rag)
        elif (od / "dev_rag.jsonl").exists():
            (od / "dev_rag.jsonl").unlink()           # stale: from an earlier --style rag build
            log.info("removed stale %s (this build is --style %s)", od / "dev_rag.jsonl", style)
        (od / "blocklist_ids.json").write_text(json.dumps(blocklist, ensure_ascii=False, indent=1), encoding="utf-8")
        (od / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    stats["_kept"] = {"train": out["train"], "dev": out["dev"], "needs": out["needs"], "dev_rag": dev_rag}  # in memory
    stats["_dropped"] = dropped
    return stats
