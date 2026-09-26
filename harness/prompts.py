"""Prompts for the history matura: one builder for training rows AND exam-time requests (train/serve consistency).

style="organizer" reproduces the organizers' published benchmark protocol byte for byte where the package allows it
(verified against their Bielik-4.5B run data, benchmark v0.1, text input with fixed image descriptions):

    system: ORGANIZER_SYSTEM_PROMPT
    user:   "Zadanie {id} ({max_points} pkt)\n\n{source_text}\n\n{question}"

with every "[Obraz: images/X.png]" marker replaced by "[Opis źródła X]\n{description}". There is no answer-format
instruction in that protocol; answers were graded by an LLM against the CKE rubric. The only known difference is the
source text itself: the new package keeps image credit lines that the benchmark text dropped.

style="formatted" is the same layout plus one short Polish line derived from answer_format (exact closed syntax,
"Temat N." + 300 words for the essay, the question's own answer labels for open items). Use it only for runs whose
prompts were trained the same way.

style="rag" is the organizer layout plus retrieved reference passages (tuned runs only; the base run stays bare):

    system: ORGANIZER_SYSTEM_PROMPT + " " + RAG_SYSTEM_SENTENCE
    user:   "Materiały pomocnicze:\n{passages}\n\n" + the organizer user message   (prefix only when passages != "")

`passages` is the text of harness.rag.format_passages(...). The same build_messages call renders training rows, so a
LoRA trained on rag rows sees exactly the exam-time layout.

The essay-plan and label-repair helpers at the end of this module (essay_plan_messages, essay_from_plan_messages,
label_repair_instruction) are the follow-up prompts of scripts/run_exam.py --essay-mode plan / --repair-labels.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

ORGANIZER_SYSTEM_PROMPT = (
    "Rozwiąż zadanie z historii po polsku. Otrzymujesz tekst źródeł, a obrazy zastąpiono opisami. "
    "Wykorzystaj źródła i własną wiedzę zgodnie z poleceniem. Udziel tylko odpowiedzi na podane zadanie. "
    "Nie dopisuj innych zadań. Nie masz dostępu do narzędzi ani internetu."
)

STYLES = ("organizer", "formatted", "rag")
RAG_SYSTEM_SENTENCE = "Możesz korzystać z załączonych materiałów pomocniczych; mogą być niekompletne lub nieistotne."
RAG_MATERIALS_HEADER = "Materiały pomocnicze:\n"
KINDS = ("closed_single", "closed_tf", "closed_match", "closed_multi_part", "open", "essay")
CLOSED_KINDS = frozenset(k for k in KINDS if k.startswith("closed_"))

MAX_NEW_TOKENS_SHORT = 2048
MAX_NEW_TOKENS_ESSAY = 4096
ESSAY_MIN_WORDS = 300

IMAGE_MARKER_RE = re.compile(r"\[Obraz:\s*([^\]]+?)\s*\]")

_AF_SINGLE = re.compile(r"^\s*([A-Z])\s*$")
_AF_TF = re.compile(r"^\s*(\d+)\s*:\s*([PF])\s*$")
_AF_MATCH = re.compile(r"^\s*([A-Z])\s*:\s*(\d+)\s*$")
_AF_MULTI = re.compile(r"^\s*(\d+)\s*:\s*([A-Z])\s*$")
_Q_NUMBERED = re.compile(r"(?m)^[ \t]*(\d{1,2})\.[ \t]+\S")
_Q_LETTERED = re.compile(r"(?m)^[ \t]*([A-Z])\.[ \t]+\S")
_Q_LABEL = re.compile(r"(?m)^[ \t]*([A-ZĄĆĘŁŃÓŚŹŻ][^:\n]{0,40}:)[ \t]*$")


# ---- item kinds and answer-format specs --------------------------------------------------------------------------

@dataclass
class FormatSpec:
    """Exact answer syntax of one item.

    closed_single:      keys == [];            allowed[""] = option letters        -> "B"
    closed_tf:          keys == ["1","2",...]; allowed[k] = ["P","F"]              -> "1: P\\n2: F"
    closed_match:       keys == ["A","B",...]; allowed[k] = [] (any number)        -> "A: 3\\nB: 1"
    closed_multi_part:  keys == ["1","2",...]; allowed[k] = option letters per part -> "1: C\\n2: A"
    open / essay:       keys == []
    """
    kind: str
    keys: list[str] = field(default_factory=list)
    allowed: dict[str, list[str]] = field(default_factory=dict)
    options: dict[str, dict[str, str]] = field(default_factory=dict)  # part key ("" for single) -> letter -> text
    rows: dict[str, str] = field(default_factory=dict)                # closed_match row key -> row description
    topics: dict[int, str] = field(default_factory=dict)              # essay topic number -> topic text

    def render(self, values: Mapping[str, str]) -> str:
        """Exact answer string for closed kinds (values keyed like self.keys; "" for single choice)."""
        if self.kind == "closed_single":
            return str(values[""])
        return "\n".join(f"{k}: {values[k]}" for k in self.keys)


def _af_lines(item: Mapping[str, Any]) -> list[str]:
    return [ln.strip() for ln in str(item.get("answer_format") or "").splitlines() if ln.strip()]


def _is_essay(item: Mapping[str, Any]) -> bool:
    af = str(item.get("answer_format") or "").casefold()
    q = str(item.get("question") or "").casefold()
    if "wypracowani" in af or "wybranego tematu" in af:
        return True
    if "wyrazów" in q and "temat" in q and ("wybierz jeden" in q or "300" in q):
        return True
    return False


def item_kind(item: Mapping[str, Any]) -> str:
    """"closed_single" | "closed_tf" | "closed_match" | "closed_multi_part" | "open" | "essay" (from answer_format)."""
    if _is_essay(item):
        return "essay"
    lines = _af_lines(item)
    if len(lines) == 1 and _AF_SINGLE.match(lines[0]):
        return "closed_single"
    if lines:
        if all(_AF_TF.match(ln) for ln in lines):
            return "closed_tf"
        if all(_AF_MATCH.match(ln) for ln in lines):
            return "closed_match"
        if all(_AF_MULTI.match(ln) for ln in lines):
            return "closed_multi_part"
    return "open"


def _consecutive(found: list[str], first: str) -> list[str]:
    """Longest run first, first+1, ... in order of appearance (numbers or capital letters)."""
    out: list[str] = []
    expect = first
    for tok in found:
        if tok == expect:
            out.append(tok)
            expect = str(int(tok) + 1) if tok.isdigit() else chr(ord(tok) + 1)
    return out


def _numbered_blocks(question: str) -> list[tuple[str, str]]:
    """[(number, text until the next numbered line)] for consecutive 1., 2., ... lines of the question."""
    marks = [(m.group(1), m.start()) for m in _Q_NUMBERED.finditer(question)]
    keys = _consecutive([k for k, _ in marks], "1")
    blocks: list[tuple[str, str]] = []
    pos = 0
    starts: list[tuple[str, int]] = []
    for k in keys:
        for mk, st in marks:
            if mk == k and st >= pos:
                starts.append((k, st))
                pos = st + 1
                break
    for i, (k, st) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(question)
        blocks.append((k, question[st:end]))
    return blocks


def _lettered_options(text: str) -> dict[str, str]:
    """Consecutive "A. text", "B. text", ... lines -> {letter: text (continuation lines joined)}."""
    marks = list(_Q_LETTERED.finditer(text))
    letters = _consecutive([m.group(1) for m in marks], "A")
    out: dict[str, str] = {}
    chosen = []
    pos = 0
    for letter in letters:
        for m in marks:
            if m.group(1) == letter and m.start() >= pos:
                chosen.append(m)
                pos = m.start() + 1
                break
    for i, m in enumerate(chosen):
        end = chosen[i + 1].start() if i + 1 < len(chosen) else len(text)
        body = text[m.start():end]
        body = re.sub(r"^[ \t]*[A-Z]\.[ \t]+", "", body, count=1)
        # stop the option at a following numbered line (next part of a multi-part question)
        body = re.split(r"(?m)^[ \t]*\d{1,2}\.[ \t]+", body, maxsplit=1)[0]
        out[m.group(1)] = " ".join(body.split())
    return out


def _essay_topics(question: str) -> dict[int, str]:
    return {int(k): " ".join(re.sub(r"^\s*\d{1,2}\.\s*", "", body).split()) for k, body in _numbered_blocks(question)}


def format_spec(item: Mapping[str, Any]) -> FormatSpec:
    """Answer syntax of an item. Row keys come from the question when it lists more rows than the answer_format
    example (the example shows syntax, not the row count), else from the example."""
    kind = item_kind(item)
    question = str(item.get("question") or "")
    lines = _af_lines(item)
    spec = FormatSpec(kind=kind)
    if kind == "essay":
        spec.topics = _essay_topics(question)
        return spec
    if kind == "closed_single":
        opts = _lettered_options(question)
        spec.options[""] = opts
        spec.allowed[""] = list(opts) if len(opts) >= 2 else ["A", "B", "C", "D"]
        return spec
    if kind == "closed_tf":
        af_keys = [_AF_TF.match(ln).group(1) for ln in lines]
        q_keys = [k for k, _ in _numbered_blocks(question)]
        spec.keys = q_keys if len(q_keys) >= max(2, len(af_keys)) else af_keys
        spec.allowed = {k: ["P", "F"] for k in spec.keys}
        return spec
    if kind == "closed_match":
        af_keys = [_AF_MATCH.match(ln).group(1) for ln in lines]
        rows = _lettered_options(question)
        spec.keys = list(rows) if len(rows) >= max(2, len(af_keys)) else af_keys
        spec.rows = {k: rows.get(k, "") for k in spec.keys}
        spec.allowed = {k: [] for k in spec.keys}
        return spec
    if kind == "closed_multi_part":
        af_keys = [_AF_MULTI.match(ln).group(1) for ln in lines]
        blocks = _numbered_blocks(question)
        q_keys = [k for k, _ in blocks]
        spec.keys = q_keys if len(q_keys) >= max(2, len(af_keys)) else af_keys
        by_key = dict(blocks)
        for k in spec.keys:
            opts = _lettered_options(by_key.get(k, ""))
            spec.options[k] = opts
            spec.allowed[k] = list(opts) if len(opts) >= 2 else ["A", "B", "C", "D"]
        return spec
    return spec


def placeholder_answer(item: Mapping[str, Any]) -> str:
    """Syntactically valid dummy answer (dry runs / plumbing tests). Never a real solution."""
    spec = format_spec(item)
    if spec.kind == "closed_single":
        return spec.allowed[""][0]
    if spec.kind == "closed_tf":
        return spec.render({k: "P" for k in spec.keys})
    if spec.kind == "closed_match":
        return spec.render({k: "1" for k in spec.keys})
    if spec.kind == "closed_multi_part":
        return spec.render({k: spec.allowed[k][0] for k in spec.keys})
    if spec.kind == "essay":
        first = min(spec.topics) if spec.topics else 1
        return f"Temat {first}.\n\n" + " ".join(["[DRY-RUN] To jest testowe wypracowanie."] * 60)
    return f"[DRY-RUN] Testowa odpowiedź na zadanie {item.get('id')}."


def max_new_tokens(item: Mapping[str, Any]) -> int:
    """Organizer caps: 4096 generated tokens for the essay, 2048 otherwise."""
    return MAX_NEW_TOKENS_ESSAY if item_kind(item) == "essay" else MAX_NEW_TOKENS_SHORT


# ---- rendering ----------------------------------------------------------------------------------------------------

def _norm_path(p: str) -> str:
    p = str(p).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def lookup_description(path: str, image_descriptions: Mapping[str, Any] | None) -> str | None:
    """Description for an image path; accepts {path: str} or {path: {"description": str, ...}} maps."""
    if not image_descriptions:
        return None
    want = _norm_path(path)
    hit: Any = None
    for key, val in image_descriptions.items():
        if _norm_path(key) == want:
            hit = val
            break
    else:
        base = PurePosixPath(want).name
        cands = [v for k, v in image_descriptions.items() if PurePosixPath(_norm_path(k)).name == base]
        if len(cands) == 1:
            hit = cands[0]
    if isinstance(hit, Mapping):
        hit = hit.get("description")
    if hit is None:
        return None
    text = str(hit).strip()
    return text or None


def image_label(path: str) -> str:
    """"images/Z03-S2.png" -> "Z03-S2" (the organizers' description label)."""
    return PurePosixPath(_norm_path(path)).stem


def description_block(path: str, description: str) -> str:
    return f"[Opis źródła {image_label(path)}]\n{description.strip()}"


def render_source(item: Mapping[str, Any], image_descriptions: Mapping[str, Any] | None = None) -> str:
    """source_text with described markers replaced in place; described images that have no marker (the package
    has a few, e.g. photos whose caption is the whole source text) are appended at the end. Markers without a
    description stay as they are."""
    source = str(item.get("source_text") or "")
    seen: set[str] = set()

    def repl(m: re.Match) -> str:
        path = _norm_path(m.group(1))
        seen.add(path)
        desc = lookup_description(path, image_descriptions)
        return description_block(path, desc) if desc else m.group(0)

    out = IMAGE_MARKER_RE.sub(repl, source)
    extra = []
    for img in item.get("images") or []:
        path = _norm_path(img.get("path", "") if isinstance(img, Mapping) else img)
        if not path or path in seen:
            continue
        desc = lookup_description(path, image_descriptions)
        if desc:
            extra.append(description_block(path, desc))
            seen.add(path)
    if extra:
        out = (out.rstrip("\n") + "\n\n" if out.strip() else "") + "\n\n".join(extra)
    return out


def _points(item: Mapping[str, Any]) -> str:
    pts = item.get("max_points")
    if isinstance(pts, float) and pts.is_integer():
        pts = int(pts)
    return f"{pts} pkt"


def _join_or(values: list[str]) -> str:
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " lub " + values[-1]


def format_line(item: Mapping[str, Any]) -> str:
    """Short Polish answer-format instruction (style="formatted")."""
    spec = format_spec(item)
    if spec.kind == "closed_single":
        return f"Odpowiedz wyłącznie w formacie: jedna litera ({_join_or(spec.allowed[''])})."
    if spec.kind == "closed_tf":
        rows = "\n".join(f"{k}: P albo F" for k in spec.keys)
        return f"Odpowiedz wyłącznie w formacie (P – prawda, F – fałsz):\n{rows}"
    if spec.kind == "closed_match":
        rows = "\n".join(f"{k}: numer" for k in spec.keys)
        return f"Odpowiedz wyłącznie w formacie (przy każdej literze wpisz numer):\n{rows}"
    if spec.kind == "closed_multi_part":
        rows = "\n".join(f"{k}: litera" for k in spec.keys)
        return f"Odpowiedz wyłącznie w formacie (przy każdym numerze wpisz literę wybranej odpowiedzi):\n{rows}"
    if spec.kind == "essay":
        nums = [str(n) for n in sorted(spec.topics)] or ["1", "2", "3"]
        return (f"Zacznij od słowa „Temat” i numeru wybranego tematu ({_join_or(nums)}), a następnie napisz całe "
                f"wypracowanie po polsku, liczące co najmniej {ESSAY_MIN_WORDS} wyrazów.")
    labels: list[str] = []
    for lab in _Q_LABEL.findall(str(item.get("question") or "")):
        if lab not in labels:
            labels.append(lab)
    line = "Odpowiedz zwięźle po polsku i podaj wszystkie wymagane elementy odpowiedzi."
    if labels:
        line += " Użyj etykiet z polecenia: " + ", ".join(labels)
    return line


def _check_passages(style: str, passages: str | None) -> str:
    """Passages text ("" when none). Passages are only valid with style="rag" (a base run must never get them)."""
    if passages is None:
        return ""
    if not isinstance(passages, str):
        raise TypeError(f"passages must be the text of harness.rag.format_passages(), got {type(passages).__name__}")
    text = passages.strip()
    if text and style != "rag":
        raise ValueError(f"passages need style='rag' (got style={style!r})")
    return text


def render_user_content(item: Mapping[str, Any], image_descriptions: Mapping[str, Any] | None = None,
                        style: str = "organizer", passages: str | None = None) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {STYLES}")
    materials = _check_passages(style, passages)
    header = f"Zadanie {item.get('id')}"
    if item.get("max_points") is not None:
        header += f" ({_points(item)})"
    content = f"{header}\n\n{render_source(item, image_descriptions)}\n\n{item.get('question') or ''}"
    if style == "formatted":
        content = content.rstrip() + "\n\n" + format_line(item)
    if materials:
        content = RAG_MATERIALS_HEADER + materials + "\n\n" + content
    return content


def rag_system_prompt(system_prompt: str = ORGANIZER_SYSTEM_PROMPT) -> str:
    """System prompt of style="rag": the given (organizer) prompt plus the reference-materials sentence."""
    return f"{system_prompt.rstrip()} {RAG_SYSTEM_SENTENCE}"


def build_messages(item: Mapping[str, Any], image_descriptions: Mapping[str, Any] | None = None,
                   system_prompt: str | None = ORGANIZER_SYSTEM_PROMPT, style: str = "organizer",
                   passages: str | None = None) -> list[dict]:
    """Chat messages for one item. system_prompt=None/"" sends the user turn only.

    style="rag" appends RAG_SYSTEM_SENTENCE to the system prompt and, when `passages` (harness.rag.format_passages
    text) is non-empty, prefixes the user message with "Materiały pomocnicze:\\n{passages}\\n\\n"."""
    user = render_user_content(item, image_descriptions, style, passages)
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": rag_system_prompt(system_prompt) if style == "rag" else system_prompt})
    msgs.append({"role": "user", "content": user})
    return msgs


def prompt_sha256(messages: list[dict]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def missing_descriptions(items: list[Mapping[str, Any]], image_descriptions: Mapping[str, Any] | None) -> list[str]:
    """Image paths referenced by the items that have no description."""
    out: list[str] = []
    for it in items:
        for img in it.get("images") or []:
            path = _norm_path(img.get("path", "") if isinstance(img, Mapping) else img)
            if path and lookup_description(path, image_descriptions) is None and path not in out:
                out.append(path)
    return out


# ---- follow-up prompts: essay plan (run_exam --essay-mode plan) and label repair (--repair-labels) ------------------

ESSAY_PLAN_MAX_TOKENS = 1024      # the plan is short; the essay itself keeps MAX_NEW_TOKENS_ESSAY
ESSAY_PLAN_MAX_CHARS = 3000       # plan text passed on to the essay step
ESSAY_TARGET_WORDS = (450, 650)


def answer_labels(item: Mapping[str, Any]) -> list[str]:
    """Answer labels the question lists on lines of their own ("Rozstrzygnięcie:", "Uzasadnienie:", "Nazwa:"),
    in order, without duplicates."""
    labels: list[str] = []
    for lab in _Q_LABEL.findall(str(item.get("question") or "")):
        if lab not in labels:
            labels.append(lab)
    return labels


def _topic_numbers(item: Mapping[str, Any]) -> list[str]:
    return [str(n) for n in sorted(format_spec(item).topics)] or ["1", "2", "3"]


def with_user_suffix(messages: list[dict], text: str) -> list[dict]:
    """Copy of `messages` with `text` appended (after a blank line) to the last user turn."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = str(m.get("content") or "").rstrip() + "\n\n" + text
            return out
    out.append({"role": "user", "content": text})
    return out


def essay_plan_instruction(item: Mapping[str, Any]) -> str:
    nums = _join_or(_topic_numbers(item))
    return (
        "Zanim napiszesz wypracowanie, przygotuj jego plan. Wybierz ten temat, do którego znasz najwięcej konkretnych "
        "faktów i który potrafisz najlepiej uzasadnić.\n"
        f"W pierwszym wierszu wpisz słowo „Temat” i numer wybranego tematu ({nums}) zakończony kropką. "
        "Potem podaj plan; każdy punkt to jedno lub dwa zdania:\n"
        "Teza: …\n"
        "Argument 1: … (konkretne fakty: daty, postacie, wydarzenia)\n"
        "Argument 2: …\n"
        "Argument 3: …\n"
        "Argument 4 (opcjonalnie): …\n"
        "Kontrargument: … (i jego odparcie)\n"
        "Wniosek: …\n"
        "Argumenty mają dotyczyć wszystkich elementów tematu (każdego okresu, państwa lub zjawiska, które temat "
        "wymienia). Nie pisz jeszcze wypracowania."
    )


def essay_from_plan_instruction(item: Mapping[str, Any], plan: str, topic: int | None) -> str:
    lo, hi = ESSAY_TARGET_WORDS
    topic_text = format_spec(item).topics.get(topic) if topic is not None else None
    if topic is not None:
        what = f"na temat {topic}" + (f" („{topic_text}”)" if topic_text else "")
        start = f"Zacznij od „Temat {topic}.” w pierwszym wierszu"
    else:
        what = "na wybrany temat"
        start = (f"W pierwszym wierszu wpisz słowo „Temat” i numer wybranego tematu ({_join_or(_topic_numbers(item))}) "
                 "zakończony kropką")
    return (
        f"Plan wypracowania:\n{plan.strip()}\n\n"
        f"Napisz teraz całe wypracowanie {what}, zgodnie z tym planem. {start}, a potem pisz ciągłym tekstem po polsku, "
        "w akapitach, bez punktów i nagłówków: wstęp z tezą, rozwinięcie z argumentami popartymi konkretnymi faktami "
        "(daty, postacie, wydarzenia, pojęcia), kontrargument z odparciem i zakończenie z wnioskiem. "
        f"Wypracowanie ma liczyć od {lo} do {hi} wyrazów. Podawaj tylko fakty, których jesteś pewien."
    )


def essay_plan_messages(messages: list[dict], item: Mapping[str, Any]) -> list[dict]:
    """Step 1 of --essay-mode plan: the item's own messages + the planning instruction."""
    return with_user_suffix(messages, essay_plan_instruction(item))


def essay_from_plan_messages(messages: list[dict], item: Mapping[str, Any], plan: str,
                             topic: int | None) -> list[dict]:
    """Step 2 of --essay-mode plan: the item's messages (for rag: with passages for the chosen topic) + the plan and
    the instruction to write the whole essay."""
    return with_user_suffix(messages, essay_from_plan_instruction(item, plan, topic))


def label_repair_instruction(labels: list[str], missing: list[str]) -> str:
    """Follow-up turn when an open answer lacks some of the question's answer labels."""
    quoted = ", ".join(f"„{lab}”" for lab in missing)
    what = f"elementu {quoted}" if len(missing) == 1 else f"elementów {quoted}"
    return (f"W odpowiedzi brakuje {what}. Napisz całą odpowiedź jeszcze raz. Każdy element zacznij od etykiety "
            "z polecenia:\n" + "\n".join(labels) + "\nNie dodawaj nic więcej.")
