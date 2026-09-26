"""External data: secondary eval (LLMzSzŁ matura), the dedup blocklist and general Polish instruction pairs.

Verified source formats (docs/research/matura_formats_datasets.md §B, docs/research/data_sources.md Task A):
  * LLMzSzŁ `correct_answer_index` is 0-based into `answers`; INCLUDE `answer` is 0-based into option_a..d;
    dokato/exam-polish-matura `answer` is a 1-based STRING; dokato/multimodal-PL-exams `answer` is a 0-based int;
    Abituria `correctOption` is 1-based (checked against the letter named in each item's `solution`).
  * NASK-PIB/PLLuM-Align files have different schemas -> each file is downloaded on its own.
  * openeurollm/EU-Instruct-Synthetic `pl` answers contain generator artifacts ("Final thought", English bold headers).

Converters are pure functions (unit-tested on inline rows); build_* functions download, convert and write.
Blocklist rows: {id, source, subject|null, question, options?: {A: ..}, answer_text?: str} (dedup targets only).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from eval.answer_format import format_number
from eval.schema import LETTERS, SchemaError, read_jsonl, validate_item, write_jsonl

log = logging.getLogger(__name__)

USER_AGENT = f"MaturaBielikFT-data/0.1 (hackathon research) python-requests/{requests.__version__}"

LLMZSZL_REPO, LLMZSZL_FILE = "amu-cai/llmzszl-dataset", "llmzszl-test.jsonl"
LLMZSZL_MATURA, LLMZSZL_VOCATIONAL = "Egzaminy Maturalne", "Egzaminy Zawodowe"
INCLUDE_REPO = "CohereLabs/include-base-44"
INCLUDE_FILES = {"test": "Polish/test-00000-of-00001.parquet", "validation": "Polish/validation-00000-of-00001.parquet"}
INCLUDE_PROFESSIONAL = "Professional certification"
DOKATO_MATURA_REPO, DOKATO_MATURA_FILE = "dokato/exam-polish-matura", "exam_polish_matura_converted.json"
DOKATO_MM_REPO, DOKATO_MM_FILE = "dokato/multimodal-PL-exams", "exams_pl.json"
PAWEL_OPEN_REPO, PAWEL_OPEN_FILE = "pawel04/otwarte-pytania-matura-cke", "otwarte-pytania-matura-cke.jsonl"
PAWEL_LLMZSZL_REPO, PAWEL_LLMZSZL_FILE = "pawel04/llmzszl-open-ended", "llmzszl-open-ended.jsonl"
ABITURIA_API = "https://api.github.com/repos/haribo841/Abituria/contents/Content"
ABITURIA_RAW = "https://raw.githubusercontent.com/haribo841/Abituria/main/Content/{name}"
LMAMACL_URL = "https://raw.githubusercontent.com/lMamacl/zadania-maturalne/master/data/baza_zada%C5%84.json"
PLLUM_REPO, PLLUM_FILES = "NASK-PIB/PLLuM-Align", ("ranking.jsonl", "rating.jsonl")
EUIS_REPO, EUIS_FILE = "openeurollm/EU-Instruct-Synthetic", "pl/train.parquet"

EUIS_ARTIFACT = re.compile(r"Final thought|\*\*(Caveats|Details|Summary|Overview|Answer)\*\*")
EUIS_MAX_USER_CHARS, EUIS_ANSWER_CHARS = 2000, (20, 2500)
EXT_OPTIONS_RANGE = (4, 6)
BLOCK_LETTERS = "ABCDEFGHIJKLMNOP"

_SUBJECTS = {
    "matematyka": "matematyka", "math": "matematyka",
    "fizyka": "fizyka", "physics": "fizyka",
    "biologia": "biologia", "biology": "biologia",
    "chemia": "chemia", "chemistry": "chemia",
    "geografia": "geografia", "geography": "geografia",
    "historia": "historia", "history": "historia",
    "wiedza o społeczeństwie": "wos", "sociology": "wos", "society": "wos",
    "język polski": "jezyk_polski",
}


def map_subject(name: str | None) -> str | None:
    """Source subject label -> eval/schema.py subject (None for subjects we do not cover, e.g. Przyroda)."""
    return _SUBJECTS.get((name or "").strip().casefold())


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


_MD_MARKS = re.compile(r"\*\*|__|(?<!\w)_|_(?!\w)|^[ \t]*#+[ \t]*|^[ \t]*-[ \t]+", re.MULTILINE)
_TASK_HEADER = re.compile(r"^\s*Zadanie\s+\d+(?:\.\d+)*\.?\s*(?:\(\s*0\s*[–-]\s*\d+\s*\))?\s*", re.IGNORECASE)


def clean_exam_text(text: str) -> str:
    """Strip markdown emphasis/headings, 'Zadanie N. (0–1)' headers and redundant whitespace."""
    s = _MD_MARKS.sub("", text or "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s).strip()
    return _TASK_HEADER.sub("", s).strip()


# ----------------------------------------------------------------------------- converters (pure)


def llmzszl_item(row: dict, row_idx: int) -> dict:
    """One LLMzSzŁ row -> schema mc item (not validated)."""
    answers = [str(a).strip() for a in row["answers"]]
    return {
        "id": f"llmzszl-{row_idx}",
        "subject": map_subject(row.get("name")),
        "type": "mc",
        "question": row["question"].strip(),
        "options": {LETTERS[i]: a for i, a in enumerate(answers[: len(LETTERS)])},
        "answer": LETTERS[int(row["correct_answer_index"])],
        "points": 1,
        "source": f"llmzszl:{row['year']}",
    }


def llmzszl_matura_items(rows: list[dict]) -> tuple[list[dict], dict]:
    """Matura rows -> validated mc items; row index = position in the test split. Drops exact-duplicate questions."""
    items: list[dict] = []
    seen: set[str] = set()
    stats: Counter = Counter()
    for idx, row in enumerate(rows):
        if row.get("type") != LLMZSZL_MATURA:
            continue
        stats["matura_rows"] += 1
        question = row["question"].strip()
        if question in seen:
            stats["duplicate_question"] += 1
            continue
        lo, hi = EXT_OPTIONS_RANGE
        try:
            if not lo <= len(row["answers"]) <= hi:
                raise SchemaError(f"[llmzszl-{idx}] expected {lo}-{hi} options, got {len(row['answers'])}")
            item = validate_item(llmzszl_item(row, idx))
            if item["subject"] not in ("matematyka", "fizyka", "biologia"):
                raise SchemaError(f"[llmzszl-{idx}] unexpected subject {row.get('name')!r}")
        except (SchemaError, IndexError, ValueError) as exc:
            stats["invalid"] += 1
            log.warning("ext_llmzszl_matura: dropping row %d: %s", idx, exc)
            continue
        seen.add(question)
        items.append(item)
    stats["kept"] = len(items)
    stats["per_subject"] = dict(Counter(i["subject"] for i in items))
    return items, dict(stats)


def block_row(
    row_id: str,
    source: str,
    subject: str | None,
    question: str,
    options: list | None = None,
    answer_index: int | None = None,
    answer_text: Any = None,
) -> dict:
    """Build one blocklist row. answer_index is 0-based into options (callers convert other bases)."""
    row: dict[str, Any] = {"id": row_id, "source": source, "subject": subject, "question": clean_exam_text(question)}
    if options:
        opts = [str(o).strip() for o in options]
        row["options"] = {BLOCK_LETTERS[i]: o for i, o in enumerate(opts[: len(BLOCK_LETTERS)])}
        if answer_index is not None and 0 <= answer_index < len(opts) and opts[answer_index]:
            answer_text = opts[answer_index]
    if answer_text is not None and str(answer_text).strip():
        row["answer_text"] = str(answer_text).strip()
    return row


def llmzszl_blocklist(rows: list[dict]) -> list[dict]:
    """All non-vocational LLMzSzŁ rows (matura + gimnazjum + 8th grade)."""
    return [
        block_row(f"llmzszl-{idx}", f"llmzszl:{r['type']}:{r['year']}", map_subject(r.get("name")), r["question"],
                  r["answers"], int(r["correct_answer_index"]))
        for idx, r in enumerate(rows)
        if r.get("type") != LLMZSZL_VOCATIONAL
    ]


def include_blocklist(rows: list[dict], split: str) -> list[dict]:
    """INCLUDE Polish rows except professional certification; `answer` is 0-based."""
    return [
        block_row(f"include-pl-{split}-{idx}", f"include-base-44:Polish/{split}", map_subject(r.get("subject")),
                  r["question"], [r[f"option_{c}"] for c in "abcd"], int(r["answer"]))
        for idx, r in enumerate(rows)
        if r.get("subject") != INCLUDE_PROFESSIONAL
    ]


def dokato_matura_blocklist(rows: list[dict]) -> list[dict]:
    """dokato/exam-polish-matura: `answer` is a 1-based string ("1".."4")."""
    return [
        block_row(f"dokato-matura-{idx}", f"dokato/exam-polish-matura:{r.get('file_name', '')}",
                  map_subject(r.get("category_original_lang")), r["question"], r["options"], int(r["answer"]) - 1)
        for idx, r in enumerate(rows)
    ]


def dokato_multimodal_blocklist(rows: list[dict]) -> list[dict]:
    """dokato/multimodal-PL-exams: `answer` is a 0-based int (P/F and bracket items are split into 2-option rows)."""
    return [
        block_row(f"dokato-mm-{idx}", f"dokato/multimodal-PL-exams:{r.get('file_name', '')}",
                  map_subject(r.get("category_original_lang")), r["question"], r["options"], int(r["answer"]))
        for idx, r in enumerate(rows)
    ]


def pawel_open_blocklist(rows: list[dict]) -> list[dict]:
    """Extended-math open tasks. `klucz` is a marking scheme, not an answer, so no answer_text (Q + fuzzy only)."""
    return [
        block_row(f"pawel-otwarte-{r['id']}", f"pawel04/otwarte-pytania-matura-cke:{r['id']}", "matematyka", r["tresc_zadania"])
        for r in rows
    ]


def pawel_llmzszl_open_blocklist(rows: list[dict]) -> list[dict]:
    return [
        block_row(f"pawel-llmzszl-open-{idx}", f"pawel04/llmzszl-open-ended:{r.get('type', '')}", map_subject(r.get("name")),
                  r["question"], answer_text=r.get("answer"))
        for idx, r in enumerate(rows)
    ]


def abituria_blocklist(exam_json: dict) -> list[dict]:
    """One Abituria exam file. multipleChoice: `correctOption` is 1-BASED (297/297 items whose `solution` names the
    letter agree; the research note saying 0-based is wrong); numeric: `expectedValue`; other modes: no answer."""
    exam = exam_json["exam"]
    rows = []
    for ex in exam["exercises"]:
        mode = ex.get("mode")
        is_mc = mode == "multipleChoice" and ex.get("options")
        value = ex.get("expectedValue") if mode == "numeric" else None
        rows.append(block_row(
            f"abituria-{ex['id']}", f"github:haribo841/Abituria:{exam['id']}", "matematyka", ex.get("prompt", ""),
            ex["options"] if is_mc else None,
            int(ex["correctOption"]) - 1 if is_mc and ex.get("correctOption") is not None else None,
            format_number(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        ))
    return rows


def lmamacl_blocklist(data: dict) -> list[dict]:
    """Polish-language matura tasks (no answers): one row per task intro + one per sub-task."""
    rows = []
    for task in data["zadania"]:
        base_id = f"lmamacl-{task['id']}"
        source = f"github:lMamacl/zadania-maturalne:{task.get('arkusz', {}).get('plik_źródłowy', '')}"
        rows.append(block_row(base_id, source, "jezyk_polski", task.get("treść_wprowadzająca", "")))
        for sub in task.get("podzadania") or []:
            rows.append(block_row(f"{base_id}-{sub['numer_podzadania']}", source, "jezyk_polski", sub.get("treść", "")))
    return [r for r in rows if r["question"]]


def eu_instruct_ok(messages: list[dict] | None) -> bool:
    """Single user->assistant turn, no generator artifacts, sane lengths."""
    if not messages or len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        return False
    user, answer = messages[0].get("content") or "", messages[1].get("content") or ""
    lo, hi = EUIS_ANSWER_CHARS
    return bool(user.strip()) and len(user) <= EUIS_MAX_USER_CHARS and lo <= len(answer) <= hi and not EUIS_ARTIFACT.search(answer)


def pllum_candidates(files: dict[str, list[dict]]) -> list[dict]:
    """PLLuM-Align `chosen` conversations: single-turn user->assistant, deduplicated by normalized prompt (file order)."""
    out: list[dict] = []
    seen_prompts: set[str] = set()
    seen_ids: set[str] = set()
    for fname, rows in files.items():
        for r in rows:
            msgs = [{"role": m.get("role"), "content": m.get("content") or ""} for m in r.get("chosen") or []]
            if [m["role"] for m in msgs] != ["user", "assistant"] or not all(m["content"].strip() for m in msgs):
                continue
            key = normalize_prompt(msgs[0]["content"])
            row_id = f"pllum-{r['id']}"
            if key in seen_prompts or row_id in seen_ids:
                continue
            seen_prompts.add(key)
            seen_ids.add(row_id)
            out.append({"id": row_id, "source": f"{PLLUM_REPO}:{fname}", "messages": msgs})
    return out


def unique_ids(rows: list[dict]) -> list[dict]:
    """Suffix repeated ids (-dup1, -dup2, ...) so every blocklist file has unique ids."""
    seen: Counter = Counter()
    for r in rows:
        seen[r["id"]] += 1
        if seen[r["id"]] > 1:
            r["id"] = f"{r['id']}-dup{seen[r['id']] - 1}"
    return rows


# ----------------------------------------------------------------------------- downloads


def _hf_file(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))


def _read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _http_get(url: str, *, retries: int = 3, timeout: float = 60.0) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


_ABITURIA_EXAM = re.compile(r"^exam-\d{4}-[\w-]+\.json$")
_ABITURIA_STEMS = ("main-basic", "main-extended", "correction", "correction-basic", "correction-extended",
                   "f2015-main-basic", "f2015-main-extended", "f2015-correction-basic", "f2015-correction-extended")


def _abituria_filenames() -> list[str]:
    """Exam JSON names via the GitHub contents API; falls back to probing known name patterns (API is rate-limited)."""
    try:
        listing = _http_get(ABITURIA_API, retries=1).json()
        return sorted(x["name"] for x in listing if _ABITURIA_EXAM.match(x["name"]))
    except (RuntimeError, requests.RequestException, ValueError, TypeError, KeyError) as exc:
        log.warning("Abituria: GitHub API listing failed (%s); probing file-name patterns", exc)
    names = []
    for year in range(2015, dt.date.today().year + 2):
        for stem in _ABITURIA_STEMS:
            name = f"exam-{year}-{stem}.json"
            r = requests.head(ABITURIA_RAW.format(name=name), headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 200:
                names.append(name)
    return names


def fetch_llmzszl_rows() -> list[dict]:
    return read_jsonl(_hf_file(LLMZSZL_REPO, LLMZSZL_FILE))


def _fetch_include() -> list[dict]:
    rows: list[dict] = []
    for split, fname in INCLUDE_FILES.items():
        rows += include_blocklist(_read_parquet(_hf_file(INCLUDE_REPO, fname)), split)
    return rows


def _fetch_abituria() -> list[dict]:
    names = _abituria_filenames()
    if not names:
        raise RuntimeError("no Abituria exam files found")
    rows: list[dict] = []
    for name in names:
        rows += abituria_blocklist(_http_get(ABITURIA_RAW.format(name=name)).json())
    return rows


@dataclass(frozen=True)
class BlockSource:
    name: str  # output file stem: data/blocklist/<name>.jsonl
    origin: str
    license: str
    fetch: Callable[[], list[dict]]


BLOCK_SOURCES: tuple[BlockSource, ...] = (
    BlockSource("llmzszl", f"HF {LLMZSZL_REPO} (test; non-vocational: matura, gimnazjum, 8th grade)",
                "none in dataset card (paper CC BY 4.0; data license UNVERIFIED)",
                lambda: llmzszl_blocklist(fetch_llmzszl_rows())),
    BlockSource("include_pl", f"HF {INCLUDE_REPO} (Polish test+validation, non-professional)", "apache-2.0", _fetch_include),
    BlockSource("dokato_exam_polish_matura", f"HF {DOKATO_MATURA_REPO}", "cc-by-nc-sa-2.0",
                lambda: dokato_matura_blocklist(_read_json_or_jsonl(_hf_file(DOKATO_MATURA_REPO, DOKATO_MATURA_FILE)))),
    BlockSource("dokato_multimodal_pl_exams", f"HF {DOKATO_MM_REPO}", "cc-by-nc-sa-2.0",
                lambda: dokato_multimodal_blocklist(_read_json_or_jsonl(_hf_file(DOKATO_MM_REPO, DOKATO_MM_FILE)))),
    BlockSource("pawel04_otwarte_pytania_matura_cke", f"HF {PAWEL_OPEN_REPO}", "none stated",
                lambda: pawel_open_blocklist(_read_json_or_jsonl(_hf_file(PAWEL_OPEN_REPO, PAWEL_OPEN_FILE)))),
    BlockSource("pawel04_llmzszl_open_ended", f"HF {PAWEL_LLMZSZL_REPO}", "none stated",
                lambda: pawel_llmzszl_open_blocklist(_read_json_or_jsonl(_hf_file(PAWEL_LLMZSZL_REPO, PAWEL_LLMZSZL_FILE)))),
    BlockSource("abituria_math", "GitHub haribo841/Abituria Content/exam-*.json (main + correction sessions)",
                "code MIT; CONTENT_PROVENANCE.md: MIT does not automatically cover exam texts (CKE)", _fetch_abituria),
    BlockSource("lmamacl_zadania_maturalne", "GitHub lMamacl/zadania-maturalne data/baza_zadań.json (no answers)",
                "none stated (texts from arkusze.pl / CKE)", lambda: lmamacl_blocklist(_http_get(LMAMACL_URL).json())),
)


# ----------------------------------------------------------------------------- builders


def build_ext_llmzszl_matura(out_path: str | Path, *, stats: dict | None = None) -> int:
    items, st = llmzszl_matura_items(fetch_llmzszl_rows())
    if stats is not None:
        stats.update(st)
    return write_jsonl(items, out_path)


def build_blocklist(out_dir: str | Path) -> dict[str, int]:
    """Write data/blocklist/<source>.jsonl per source; a failing source warns and keeps its previous file."""
    out_dir = Path(out_dir)
    counts: dict[str, int] = {}
    status: dict[str, dict] = {}
    today = dt.date.today().isoformat()
    for src in BLOCK_SOURCES:
        try:
            rows = unique_ids(src.fetch())
            if not rows:
                raise RuntimeError("source returned no rows")
            counts[src.name] = write_jsonl(rows, out_dir / f"{src.name}.jsonl")
            status[src.name] = {"status": "ok", "rows": counts[src.name], "fetched": today}
            log.info("blocklist %s: %d rows", src.name, counts[src.name])
        except Exception as exc:  # noqa: BLE001 - one broken source must not stop the others
            log.warning("blocklist %s FAILED: %s", src.name, exc)
            status[src.name] = {"status": f"failed: {exc}", "fetched": today}
    update_readme(out_dir, {"blocklist": status})
    return counts


def build_general_heldout(out_path: str | Path, n: int, seed: int, *, stats: dict | None = None) -> int:
    files = {fname: read_jsonl(_hf_file(PLLUM_REPO, fname)) for fname in PLLUM_FILES}
    candidates = pllum_candidates(files)
    rows = random.Random(seed).sample(candidates, min(n, len(candidates)))
    if stats is not None:
        stats.update({"rows_per_file": {k: len(v) for k, v in files.items()}, "single_turn_unique_prompts": len(candidates),
                      "sampled": len(rows), "seed": seed})
    return write_jsonl(rows, out_path)


def build_general_pool(
    out_path: str | Path, n: int, seed: int, *, exclude_prompts: Iterable[str] = (), stats: dict | None = None
) -> int:
    """Filter EU-Instruct-Synthetic `pl`, drop duplicate / excluded (held-out) prompts, sample n rows."""
    import pyarrow.parquet as pq

    table = pq.read_table(_hf_file(EUIS_REPO, EUIS_FILE), columns=["messages"])
    excluded = {normalize_prompt(p) for p in exclude_prompts}
    seen: set[str] = set()
    keep: list[int] = []
    counts: Counter = Counter(filtered_artifact_or_length=0, heldout_prompt_overlap=0, duplicate_prompt=0)
    offset = 0
    for batch in table.to_batches(max_chunksize=10_000):
        for i, msgs in enumerate(batch.column(0).to_pylist()):
            if not eu_instruct_ok(msgs):
                counts["filtered_artifact_or_length"] += 1
                continue
            key = normalize_prompt(msgs[0]["content"])
            if key in excluded:
                counts["heldout_prompt_overlap"] += 1
            elif key in seen:
                counts["duplicate_prompt"] += 1
            else:
                seen.add(key)
                keep.append(offset + i)
        offset += batch.num_rows
    chosen = random.Random(seed).sample(keep, min(n, len(keep)))
    messages = table.take(chosen).column("messages").to_pylist()
    rows = [
        {"id": f"euis-{idx}", "source": f"{EUIS_REPO}:pl", "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}
        for idx, msgs in zip(chosen, messages)
    ]
    if stats is not None:
        stats.update({"rows": table.num_rows, **counts, "eligible": len(keep), "sampled": len(rows), "seed": seed})
    return write_jsonl(rows, out_path)


# ----------------------------------------------------------------------------- provenance README

_META = re.compile(r"<!-- fetch_meta\n(.*?)\n-->", re.DOTALL)
OTHER_OUTPUTS = {
    "ext_llmzszl_matura": ("data/eval/ext_llmzszl_matura.jsonl", f"HF {LLMZSZL_REPO} test, type == '{LLMZSZL_MATURA}'",
                           "none in dataset card (paper CC BY 4.0; UNVERIFIED)", "NEVER train (secondary eval)"),
    "general_heldout": ("data/eval/general_heldout.jsonl", f"HF {PLLUM_REPO} ranking+rating `chosen`", "cc-by-sa-4.0",
                        "general-capability NLL check"),
    "general_pool": ("data/general/train_pool.jsonl", f"HF {EUIS_REPO} config pl (filtered)", "apache-2.0",
                     "general instruction pairs mixed into training"),
}


def read_meta(readme: Path) -> dict:
    if not readme.exists():
        return {}
    m = _META.search(readme.read_text(encoding="utf-8"))
    return json.loads(m.group(1)) if m else {}


def update_readme(blocklist_dir: str | Path, updates: dict) -> Path:
    """Merge `updates` into the machine-readable meta block of data/blocklist/README.md and re-render it."""
    readme = Path(blocklist_dir) / "README.md"
    meta = read_meta(readme)
    for key, value in updates.items():
        meta[key] = {**meta.get(key, {}), **value} if isinstance(value, dict) else value
    lines = [
        "# External-exam blocklist (dedup targets only — NEVER train on these)",
        "",
        "Generated by `python scripts/fetch_external.py`; do not edit by hand.",
        "Rows: `{id, source, subject|null, question, options?, answer_text?}`. `scripts/build_data.py` drops synthetic items",
        "that near-duplicate any row here (embedding Q / Q+answer + fuzzy stem match, see `train/dedup.py`).",
        "Exam texts are © CKE; the datasets below only redistribute them, so they are used for leakage filtering only.",
        "",
        "| file | source | license | rows | fetched | status |",
        "|---|---|---|---|---|---|",
    ]
    for src in BLOCK_SOURCES:
        st = meta.get("blocklist", {}).get(src.name, {})
        path = Path(blocklist_dir) / f"{src.name}.jsonl"
        rows = sum(1 for line in open(path, encoding="utf-8") if line.strip()) if path.exists() else 0
        lines.append(f"| `{src.name}.jsonl` | {src.origin} | {src.license} | {rows} | {st.get('fetched', '-')} | {st.get('status', 'not fetched')} |")
    lines += ["", "## Other outputs of fetch_external.py", "", "| file | source | license | rows | fetched | use | counts |", "|---|---|---|---|---|---|---|"]
    for key, (path, origin, lic, use) in OTHER_OUTPUTS.items():
        st = meta.get(key, {})
        counts = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in sorted(st.items()) if k != "fetched")
        lines.append(f"| `{path}` | {origin} | {lic} | {st.get('sampled', st.get('kept', '-'))} | {st.get('fetched', '-')} | {use} | {counts} |")
    lines += ["", "<!-- fetch_meta", json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True), "-->", ""]
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme
