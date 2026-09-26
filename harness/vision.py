"""Image-description pre-pass for the history matura (exam input_format "separate-text-and-images-v1").

Bielik is text-only, and the organizers' benchmark gave the model image *descriptions* instead of images
("a obrazy zastąpiono opisami"). So before the text model starts, a small VLM served ALONE on an
OpenAI-compatible endpoint (llama-server --mmproj, or `qvac serve --openai`) describes every image the exam items
reference. It is unloaded before Bielik loads, so peak RAM is the larger model, not the sum. The harness then
replaces each "[Obraz: images/X.png]" marker with the description (harness/prompts; images without a marker are
appended); base and tuned runs get the very same descriptions, so the progress delta is not confounded by vision.

One VLM call per unique image (deduplicated by sha256). The request carries the image as a data URI (image part
BEFORE the text part) plus a Polish instruction with the question/source context of every item that uses the
image, asking for an exam-neutral description (kind, verbatim text, what is depicted) and NOT the answer. The
rules follow the organizers' benchmark descriptions: scanned text transcribed, tables keep their data, trees keep
their relationships, no solutions, inferred identities or interpretations; prose that starts with the kind.
Decoding is deterministic: temperature 0, top_k 1, seed 42, thinking disabled, all params sent explicitly
(QVAC otherwise defaults to temp 0.8 / repeat_penalty 1.1 and ignores some fields).

Cache file (JSON, keyed by exam-relative image path):
    {"images/X.png": {"sha256": "...", "description": "...", "model": "...", "ocr": null | "..."}}
An entry is reused whenever the file's current sha256 matches (whatever path or model produced it); force=True
redoes everything. Failed images are never cached, so a rerun retries only them. The cache holds text derived
from the exam, so it lives next to exam.json by default and must never be committed.

Optional OCR cross-check: when `tesseract` with Polish data (`pol`) is installed, its raw output is stored in
"ocr" and `ocr_coverage` measures how much of it the VLM transcription covers. `reference_scores` compares a
description with a reference one (e.g. the organizers' mock descriptions) to pick a VLM.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_BASE_URL = "http://127.0.0.1:8081"
DEFAULT_MODEL = "vlm"
DEFAULT_MAX_TOKENS = 700
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_PARALLEL = 2
SEED = 42
CACHE_FILENAME = "descriptions.json"
API_KEY_ENV = "VLM_API_KEY"

IMAGE_KINDS = ("mapa", "ilustracja", "fotografia", "karykatura", "moneta", "tabela", "skan tekstu", "schemat")

MARKER_RE = re.compile(r"\[Obraz:\s*([^\]\n]+?)\s*\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_LOOP_RE = re.compile(r"(.{4,80}?)(?:\s*\1){5,}", re.S)
_WORD_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}")

_MAX_QUESTION_CHARS = 600
_MAX_SOURCE_CHARS = 800
_MAX_CONTEXT_CHARS = 2500
_SOURCE_WINDOW = 400

THIS_IMAGE = "[TEN OBRAZ]"
OTHER_IMAGE = "[inny obraz]"

VLM_INSTRUCTION = """\
Opisujesz materiał źródłowy z arkusza matury z historii. Twój opis zastąpi ten obraz osobie, która go nie widzi, \
więc musi oddać wszystko, co na nim widać.

Kontekst z arkusza (tylko po to, żebyś wiedział, na co zwrócić uwagę):
{context}

Jak pisać:
- Zacznij od rodzaju materiału, np. „Mapa…”, „Plan…”, „Ilustracja przedstawia…”, „Fotografia…”, „Karykatura…”, \
„Moneta…”, „Tabela…”, „Tablica genealogiczna…”, „Schemat…”, „Skan tekstu…”.
- Każdy widoczny napis przytocz dosłownie w cudzysłowie „…”, z zachowaniem pisowni: tytuły, podpisy, nazwy na mapie, \
legendę, liczby, daty, skróty. Nieczytelny fragment oznacz jako [nieczytelne].
- Skan tekstu (dokument, depeszę, gazetę, plakat, ulotkę) przepisz w całości, dosłownie.
- Tabelę przepisz ze wszystkimi wierszami, kolumnami i wartościami. W drzewie genealogicznym i w schemacie zachowaj \
wszystkie powiązania. Legendę mapy podaj jako listę z myślnikami: oznaczenie i jego znaczenie.
- Opisz, co widać: osoby (wygląd, strój, gesty, atrybuty), symbole, herby, flagi, przedmioty, budowle i ich elementy \
(łuki, okna, wieże, dachy, zdobienia), krajobraz, kompozycję i technikę. Dla mapy: obszar, granice, strzałki, \
oznaczenia i kolory. Dla monety: wizerunki, napisy, nominał.
- Jeśli czegoś ważnego na obrazie nie ma (np. daty, tytułu, podpisu, nazwy), możesz to krótko zaznaczyć.

Zasady:
- Tylko to, co rzeczywiście widać. Niczego nie zmyślaj; gdy nie masz pewności, napisz „prawdopodobnie”.
- Nie rozwiązuj zadania i nie interpretuj: nie wybieraj odpowiedzi A–D, nie oceniaj zdań jako prawdziwe lub \
fałszywe, nie ustalaj tożsamości osób ani epoki, stylu, wydarzenia czy daty, jeśli nie są napisane na obrazie.
- Pisz po polsku zwykłym tekstem: bez nagłówków, bez Markdown, bez wstępu i podsumowania. Zwykle 80–250 słów; \
dłużej tylko wtedy, gdy trzeba przepisać dłuższy tekst lub tabelę."""


class VisionError(RuntimeError):
    """A VLM request failed or returned nothing usable."""


@dataclass
class ImageRef:
    """One unique image (by sha256) and everything the exam says around it."""

    sha256: str
    paths: list[str]  # exam-relative ("images/Z01.png"); several when files are byte-identical
    abs_path: Path
    item_ids: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    source_contexts: list[str] = field(default_factory=list)
    declared_sha256: str | None = None  # from exam.json; differs from sha256 only for a corrupted package

    @property
    def path(self) -> str:
        return self.paths[0]


@dataclass
class ImageResult:
    ref: ImageRef
    status: str  # "described" | "cached" | "failed" | "planned" (dry run)
    description: str = ""
    model: str = ""
    ocr: str | None = None
    seconds: float = 0.0
    error: str | None = None
    retried: bool = False  # degenerate (looping) output, re-asked once with repetition penalties
    truncated: bool = False  # finish_reason == "length"

    @property
    def path(self) -> str:
        return self.ref.path


# ----------------------------------------------------------------------------------------------- exam parsing


def normalize_image_path(path: str) -> str:
    p = str(path).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def find_markers(text: str | None) -> list[str]:
    """Exam-relative image paths of every "[Obraz: ...]" marker in `text`, in order."""
    return [normalize_image_path(m.group(1)) for m in MARKER_RE.finditer(text or "")]


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_mime(data: bytes, name: str = "") -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    ext = Path(name).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
            ".gif": "image/gif"}.get(ext, "application/octet-stream")


def image_data_uri(path: str | os.PathLike) -> str:
    """base64 data URI of an image file. QVAC accepts only PNG/JPEG data URIs; llama-server takes more."""
    data = Path(path).read_bytes()
    mime = image_mime(data, str(path))
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def source_context(source_text: str | None, rel_path: str, window: int = _SOURCE_WINDOW) -> str:
    """The text around this image's marker (caption above, attribution below). This image's marker becomes
    "[TEN OBRAZ]", other markers "[inny obraz]". Without a marker (the image is only in item["images"]),
    the start of the source text is returned instead."""
    text = source_text or ""
    found = False

    def _sub(m: re.Match) -> str:
        nonlocal found
        if not found and normalize_image_path(m.group(1)) == rel_path:
            found = True
            return THIS_IMAGE
        return OTHER_IMAGE

    text = MARKER_RE.sub(_sub, text)
    if not found:
        return _clip(text, 2 * window)
    at = text.index(THIS_IMAGE)
    start, end = max(0, at - window), at + len(THIS_IMAGE) + window
    before = text[start:at]
    after = text[at + len(THIS_IMAGE): end]
    if start > 0 and re.search(r"\s", before):  # do not start or end mid-word
        before = re.split(r"\s", before, maxsplit=1)[1]
    if end < len(text) and re.search(r"\s", after):
        after = after[: max(m.start() for m in re.finditer(r"\s", after))] + " …"
    return (before + THIS_IMAGE + after).strip()


def item_image_paths(item: dict) -> list[str]:
    """Every image an item uses: its `images` list first, then markers not listed there (deduplicated)."""
    paths: list[str] = []
    for img in item.get("images") or []:
        raw = img.get("path") if isinstance(img, dict) else img
        if not isinstance(raw, str) or not raw.strip():
            continue
        p = normalize_image_path(raw)
        if p not in paths:
            paths.append(p)
    for p in find_markers(item.get("source_text")) + find_markers(item.get("question")):
        if p not in paths:
            paths.append(p)
    return paths


def collect_image_refs(exam_dir: str | os.PathLike, items: Iterable[dict],
                       missing: list[str] | None = None) -> list[ImageRef]:
    """Unique images (by the sha256 of the file on disk) referenced by `items`, in first-use order.
    Paths whose file does not exist are skipped and appended to `missing` when given."""
    exam_dir = Path(exam_dir)
    by_sha: dict[str, ImageRef] = {}
    sha_of_path: dict[str, str] = {}
    for item in items:
        declared = {normalize_image_path(i["path"]): i.get("sha256")
                    for i in item.get("images") or [] if isinstance(i, dict) and "path" in i}
        for rel in item_image_paths(item):
            abs_path = exam_dir / rel
            if rel not in sha_of_path:
                if not abs_path.is_file():
                    if missing is not None and rel not in missing:
                        missing.append(rel)
                    continue
                sha_of_path[rel] = sha256_file(abs_path)
            sha = sha_of_path[rel]
            ref = by_sha.get(sha)
            if ref is None:
                ref = by_sha[sha] = ImageRef(sha256=sha, paths=[rel], abs_path=abs_path,
                                             declared_sha256=declared.get(rel))
            elif rel not in ref.paths:
                ref.paths.append(rel)
            item_id = str(item.get("id", ""))
            if item_id and item_id not in ref.item_ids:
                ref.item_ids.append(item_id)
            q = _clip(item.get("question") or "", _MAX_QUESTION_CHARS)
            if q and q not in ref.questions:
                ref.questions.append(q)
            sc = _clip(source_context(item.get("source_text"), rel), _MAX_SOURCE_CHARS)
            if sc and sc not in ref.source_contexts:
                ref.source_contexts.append(sc)
    return list(by_sha.values())


# ----------------------------------------------------------------------------------------------- prompt


def build_context(ref: ImageRef, limit: int = _MAX_CONTEXT_CHARS) -> str:
    parts: list[str] = []
    if ref.source_contexts:
        parts.append("Podpis i otoczenie obrazu w arkuszu:")
        parts.extend(ref.source_contexts[:2])
    if ref.questions:
        parts.append("Polecenia, do których potrzebny jest ten obraz:")
        parts.extend(f"- {q}" for q in ref.questions)
    return _clip("\n".join(parts), limit) if parts else "(brak)"


def build_vlm_prompt(ref: ImageRef) -> str:
    return VLM_INSTRUCTION.format(context=build_context(ref))


def build_vlm_messages(ref: ImageRef, data_uri: str) -> list[dict]:
    """One user turn: the image part FIRST, then the Polish instruction."""
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": build_vlm_prompt(ref)},
    ]}]


def request_payload(model: str, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                    retry: bool = False) -> dict:
    """Deterministic chat-completions body. Every sampling field is explicit: QVAC defaults to temp 0.8 and
    repeat_penalty 1.1. `chat_template_kwargs` turns Qwen3.5/Gemma 4 thinking off on llama-server,
    `reasoning_budget: 0` does the same on QVAC; each server ignores the other's field. The retry body (after a
    looping answer) adds repetition penalties, still greedy and seeded."""
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": SEED,
        "max_tokens": int(max_tokens),
        "repeat_penalty": 1.1 if retry else 1.0,
        "presence_penalty": 1.0 if retry else 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
    }


# ----------------------------------------------------------------------------------------------- transport


def api_url(base_url: str, route: str) -> str:
    """`route` under the OpenAI /v1 prefix; accepts base URLs with or without a trailing /v1."""
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return f"{base}/{route.lstrip('/')}"


def _http_json(url: str, payload: dict | None = None, timeout: float = DEFAULT_TIMEOUT_S,
               api_key: str | None = None) -> dict:
    """POST `payload` as JSON (GET when None) and decode the JSON reply. Tests monkeypatch this."""
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise VisionError(f"HTTP {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise VisionError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from e
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise VisionError(f"non-JSON reply from {url}: {body[:200]!r}") from e


def check_server(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0, api_key: str | None = None) -> str | None:
    """None when GET /v1/models answers, else a one-line reason."""
    try:
        _http_json(api_url(base_url, "models"), None, timeout=timeout, api_key=api_key)
    except VisionError as e:
        return str(e)
    return None


def _extract_content(resp: dict) -> tuple[str, str | None]:
    try:
        choice = resp["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise VisionError(f"unexpected response shape: {json.dumps(resp, ensure_ascii=False)[:300]}") from e
    if isinstance(content, list):  # content-part form
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or "", choice.get("finish_reason")


# ----------------------------------------------------------------------------------------------- output cleanup


def clean_description(text: str) -> str:
    """Drop think blocks (and an unterminated one), markdown emphasis and code fences; tidy whitespace."""
    text = _THINK_RE.sub("", text or "")
    if re.search(r"<think>", text, re.I):  # thinking never closed: nothing usable after it
        text = re.split(r"<think>", text, flags=re.I)[0]
    text = text.replace("</think>", "")
    text = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", text, flags=re.M)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_degenerate(text: str) -> bool:
    """A greedy-decoding loop: a short phrase repeated 6+ times in a row, or a line occurring 4+ times."""
    if _LOOP_RE.search(text):
        return True
    counts: dict[str, int] = {}
    for line in (ln.strip() for ln in text.splitlines()):
        if len(line) >= 3:
            counts[line] = counts.get(line, 0) + 1
            if counts[line] >= 4:
                return True
    return False


def collapse_repetition(text: str) -> str:
    """Keep one copy of each loop and drop consecutive duplicate lines (last resort after the retry)."""
    text = _LOOP_RE.sub(lambda m: m.group(1), text)
    out: list[str] = []
    for line in text.splitlines():
        if out and line.strip() and line.strip() == out[-1].strip():
            continue
        out.append(line)
    return "\n".join(out).strip()


# ----------------------------------------------------------------------------------------------- OCR (optional)


def tesseract_binary() -> str | None:
    return shutil.which("tesseract")


def tesseract_has_lang(binary: str, lang: str = "pol") -> bool:
    try:
        out = subprocess.run([binary, "--list-langs"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    langs = {ln.strip() for ln in (out.stdout + "\n" + out.stderr).splitlines()}
    return all(part in langs for part in lang.split("+"))


def run_ocr(path: str | os.PathLike, binary: str, lang: str = "pol", timeout: float = 120.0) -> str | None:
    """`tesseract <image> stdout -l pol`; None on any failure or empty output."""
    try:
        out = subprocess.run([binary, str(path), "stdout", "-l", lang], capture_output=True, text=True,
                             timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = re.sub(r"\n{3,}", "\n\n", out.stdout.replace("\f", "")).strip()
    return text or None


def ocr_coverage(description: str, ocr: str | None) -> float | None:
    """Share of OCR words (4+ letters) that also occur in the description: a cheap check that the VLM
    transcribed the visible text. None when OCR found fewer than 3 such words."""
    if not ocr:
        return None
    words = {w.lower() for w in _WORD_RE.findall(ocr)}
    if len(words) < 3:
        return None
    have = {w.lower() for w in _WORD_RE.findall(description or "")}
    return round(len(words & have) / len(words), 3)


_QUOTED_RE = re.compile(r"[„\"“]([^”\"“„\n]{1,80})[”\"“]")


def _stems(text: str) -> set[str]:
    return {w.lower()[:5] for w in _WORD_RE.findall(text or "")}


def reference_scores(description: str, reference: str) -> dict[str, float | None]:
    """Crude agreement with a reference description (e.g. the organizers' description of a mock image):
    label_recall = share of the reference's quoted labels („…”) found verbatim (case-insensitive) in ours,
    word_recall = share of the reference's content-word stems (first 5 letters of 4+ letter words) in ours."""
    ours = (description or "").lower()
    labels = {m.group(1).strip().lower() for m in _QUOTED_RE.finditer(reference or "") if m.group(1).strip()}
    ref_stems = _stems(reference)
    return {
        "label_recall": round(sum(lab in ours for lab in labels) / len(labels), 3) if labels else None,
        "word_recall": round(len(ref_stems & _stems(description)) / len(ref_stems), 3) if ref_stems else None,
    }


# ----------------------------------------------------------------------------------------------- cache


def default_cache_path(exam_dir: str | os.PathLike) -> Path:
    return Path(exam_dir) / CACHE_FILENAME


def load_cache(cache_path: str | os.PathLike) -> dict[str, dict]:
    p = Path(cache_path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VisionError(f"{p}: expected a JSON object keyed by image path")
    return {normalize_image_path(k): v for k, v in data.items() if isinstance(v, dict)}


def save_cache(cache_path: str | os.PathLike, cache: dict[str, dict]) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def load_descriptions(cache_path: str | os.PathLike, exam_dir: str | os.PathLike | None = None,
                      items: Iterable[dict] | None = None) -> dict[str, str]:
    """{image path: description} from a cache file, as the harness consumes it. With `exam_dir` and `items`,
    only paths the items reference whose file sha256 still matches the entry are returned, so a cache
    left over from another exam (same file names, different images) is never injected."""
    cache = load_cache(cache_path)
    out = {k: v["description"] for k, v in cache.items() if isinstance(v.get("description"), str) and v["description"].strip()}
    if exam_dir is None or items is None:
        return out
    checked: dict[str, str] = {}
    for ref in collect_image_refs(exam_dir, items):
        for p in ref.paths:
            if p in out and cache[p].get("sha256") == ref.sha256:
                checked[p] = out[p]
    return checked


def find_cached_entry(cache: dict[str, dict], sha: str) -> dict | None:
    """Any cache entry with this sha256 and a non-empty description (path and model do not matter)."""
    for entry in cache.values():
        if entry.get("sha256") == sha and isinstance(entry.get("description"), str) and entry["description"].strip():
            return entry
    return None


# ----------------------------------------------------------------------------------------------- main entry


def describe_one(ref: ImageRef, base_url: str, model: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 timeout: float = DEFAULT_TIMEOUT_S, api_key: str | None = None) -> ImageResult:
    """Describe one image; retry once with repetition penalties when greedy decoding loops."""
    t0 = time.monotonic()
    messages = build_vlm_messages(ref, image_data_uri(ref.abs_path))
    url = api_url(base_url, "chat/completions")
    retried = False
    resp = _http_json(url, request_payload(model, messages, max_tokens), timeout=timeout, api_key=api_key)
    raw, finish = _extract_content(resp)
    text = clean_description(raw)
    if is_degenerate(text):
        retried = True
        resp = _http_json(url, request_payload(model, messages, max_tokens, retry=True), timeout=timeout,
                          api_key=api_key)
        raw, finish = _extract_content(resp)
        text = clean_description(raw)
        if is_degenerate(text):
            text = collapse_repetition(text)
    if not text:
        raise VisionError("empty description (thinking not disabled, or max_tokens too small?)")
    served = resp.get("model") if isinstance(resp, dict) else None
    return ImageResult(ref=ref, status="described", description=text,
                       model=served if isinstance(served, str) and served else model,
                       seconds=time.monotonic() - t0, retried=retried, truncated=finish == "length")


def describe_images_detailed(
    exam_dir: str | os.PathLike,
    items: Iterable[dict],
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    cache_path: str | os.PathLike | None = None,
    *,
    force: bool = False,
    parallel: int = DEFAULT_PARALLEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_S,
    ocr: str = "auto",
    api_key: str | None = None,
    only: Iterable[str] | None = None,
    dry_run: bool = False,
    on_result: Callable[[ImageResult], None] | None = None,
) -> tuple[dict[str, str], list[ImageResult]]:
    """Describe every referenced image not already cached; returns ({path: description}, per-image results).

    ocr: "auto" (use tesseract when installed with `pol`), "on" (same, warns via error field if unavailable),
    "off". dry_run: no network, uncached images come back with status "planned". The cache file is rewritten
    after every finished image, so an interrupted run loses nothing.
    """
    exam_dir = Path(exam_dir)
    cache_path = Path(cache_path) if cache_path is not None else default_cache_path(exam_dir)
    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV) or None
    refs = collect_image_refs(exam_dir, list(items))
    if only:
        wanted = {normalize_image_path(p) for p in only}
        refs = [r for r in refs if wanted & set(r.paths)]
    cache = load_cache(cache_path)
    lock = threading.Lock()
    results: list[ImageResult] = []

    ocr_bin = None
    ocr_note = None
    if ocr != "off" and not dry_run:
        b = tesseract_binary()
        if b and tesseract_has_lang(b, "pol"):
            ocr_bin = b
        elif ocr == "on":
            ocr_note = "tesseract with 'pol' language data not found; OCR skipped"

    def _record(res: ImageResult, write: bool) -> None:
        with lock:
            if write:
                for p in res.ref.paths:
                    cache[p] = {"sha256": res.ref.sha256, "description": res.description, "model": res.model,
                                "ocr": res.ocr}
                save_cache(cache_path, cache)
            results.append(res)
            if on_result is not None:
                on_result(res)

    todo: list[ImageRef] = []
    for ref in refs:
        hit = None if force else find_cached_entry(cache, ref.sha256)
        if hit is None:
            todo.append(ref)
            continue
        res = ImageResult(ref=ref, status="cached", description=hit["description"], model=hit.get("model") or "",
                          ocr=hit.get("ocr"))
        # copy the entry to any new path with the same bytes; rewrite only if something changed
        _record(res, write=not dry_run and any(cache.get(p, {}).get("sha256") != ref.sha256 for p in ref.paths))

    def _work(ref: ImageRef) -> None:
        if dry_run:
            _record(ImageResult(ref=ref, status="planned"), write=False)
            return
        try:
            res = describe_one(ref, base_url, model, max_tokens=max_tokens, timeout=timeout, api_key=api_key)
        except Exception as e:  # one bad image must not sink the whole pre-pass
            msg = str(e) if isinstance(e, VisionError) else f"{type(e).__name__}: {e}"
            _record(ImageResult(ref=ref, status="failed", error=msg), write=False)
            return
        if ocr_bin:
            res.ocr = run_ocr(ref.abs_path, ocr_bin)
        elif ocr_note:
            res.error = ocr_note
        _record(res, write=True)

    workers = max(1, min(int(parallel or 1), len(todo) or 1))
    if workers == 1:
        for ref in todo:
            _work(ref)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_work, todo))

    order = {r.sha256: i for i, r in enumerate(refs)}
    results.sort(key=lambda r: order[r.ref.sha256])
    descriptions = {p: r.description for r in results if r.description for p in r.ref.paths}
    return descriptions, results


def describe_images(
    exam_dir: str | os.PathLike,
    items: Iterable[dict],
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    cache_path: str | os.PathLike | None = None,
    **kwargs,
) -> dict[str, str]:
    """{"images/X.png": description} for every image the items reference (failed images are absent, so the
    harness leaves their markers as they are). See describe_images_detailed for keyword options."""
    descriptions, _ = describe_images_detailed(exam_dir, items, base_url=base_url, model=model,
                                               cache_path=cache_path, **kwargs)
    return descriptions
