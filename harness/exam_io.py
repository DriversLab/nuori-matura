"""Exam package I/O and the upload-site rules for answers.json.

Package layout (organizers' "separate-text-and-images-v1"): <exam_dir>/exam.json, <exam_dir>/images/*.png and
usually <exam_dir>/answers-template.json. Exam packages are copyrighted: keep them outside the repo or under a
gitignored folder (exams/, runs/).

Upload rules replicated by validate_answers (see the organizers' guide):
  * UTF-8 JSON file, .json, non-empty, at most 1 MiB
  * top level: only "exam_id" and "answers"; exam_id equals the exam's
  * answers: a list; each entry only "id" and "answer"; id is a string; answer is a string ("" allowed)
  * every template id exactly once (no missing, unexpected or duplicate ids)
  * each answer at most 100,000 characters
Content rules the site cannot check (final answer only, essay >= 300 words, closed syntax) are warnings from
lint_answers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

MAX_FILE_BYTES = 1024 * 1024
MAX_ANSWER_CHARS = 100_000
TOP_LEVEL_KEYS = ("exam_id", "answers")
ENTRY_KEYS = ("id", "answer")
EXAM_FILE = "exam.json"
TEMPLATE_FILE = "answers-template.json"


def _warn(msg: str) -> None:
    print(f"[exam_io] WARNING: {msg}", file=sys.stderr)


def exam_json_path(exam_dir: str | os.PathLike) -> Path:
    p = Path(exam_dir)
    return p if p.is_file() else p / EXAM_FILE


def load_exam(exam_dir: str | os.PathLike) -> tuple[dict, list[dict]]:
    """(meta, items). meta is exam.json without "items" plus "exam_dir"; items keep the package's fields, with
    defaults for optional ones (source_text "", images [], answer_format "") and ids as strings."""
    path = exam_json_path(exam_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError(f"{path}: expected an object with an 'items' list")
    meta = {k: v for k, v in data.items() if k != "items"}
    meta["exam_dir"] = str(path.parent)
    items: list[dict] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["items"]):
        if not isinstance(raw, dict) or "id" not in raw or "question" not in raw:
            raise ValueError(f"{path}: item #{i} lacks id/question")
        it = dict(raw)
        it["id"] = str(it["id"])
        if it["id"] in seen:
            raise ValueError(f"{path}: duplicate item id {it['id']!r}")
        seen.add(it["id"])
        it["source_text"] = it.get("source_text") or ""
        it["images"] = list(it.get("images") or [])
        it["answer_format"] = it.get("answer_format") or ""
        items.append(it)
    if "exam_id" not in meta:
        raise ValueError(f"{path}: missing exam_id")
    return meta, items


def load_template(exam_dir: str | os.PathLike) -> tuple[str, list[str]]:
    """(exam_id, ids) from answers-template.json, or from exam.json when the package has no template."""
    exam_path = exam_json_path(exam_dir)
    tpl = exam_path.parent / TEMPLATE_FILE
    if tpl.is_file():
        data = json.loads(tpl.read_text(encoding="utf-8"))
        return str(data["exam_id"]), [str(a["id"]) for a in data["answers"]]
    meta, items = load_exam(exam_path)
    return str(meta["exam_id"]), [it["id"] for it in items]


def image_sha256(exam_dir: str | os.PathLike, rel_path: str) -> str | None:
    p = exam_json_path(exam_dir).parent / rel_path
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_descriptions(path: str | os.PathLike | None, items: Iterable[Mapping[str, Any]] | None = None, *,
                      include_ocr: bool = False, problems: list[str] | None = None) -> dict[str, str]:
    """Read the image-description cache {"images/X.png": {"sha256", "description", "model", "ocr"}} into
    {path: description}. With items, an entry whose sha256 differs from the item's image checksum is DROPPED (a
    stale cache from another exam would otherwise inject wrong descriptions: packages reuse names like Z01.png).
    Plain {path: "text"} maps are accepted too. Problems are appended to `problems` (and printed)."""
    out: dict[str, str] = {}
    if path is None:
        return out
    probs = problems if problems is not None else []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    want_sha: dict[str, str] = {}
    for it in items or []:
        for img in it.get("images") or []:
            if isinstance(img, Mapping) and img.get("path") and img.get("sha256"):
                want_sha[str(img["path"])] = str(img["sha256"])
    for key, val in data.items():
        if isinstance(val, str):
            desc, sha, ocr = val, None, None
        elif isinstance(val, Mapping):
            desc, sha, ocr = val.get("description"), val.get("sha256"), val.get("ocr")
        else:
            continue
        if not desc or not str(desc).strip():
            continue
        if key in want_sha and sha and str(sha) != want_sha[key]:
            msg = f"description for {key} was made for a different image (sha256 mismatch); ignored"
            probs.append(msg)
            _warn(msg)
            continue
        if key in want_sha and not sha:
            msg = f"description for {key} has no sha256; used without checksum check"
            probs.append(msg)
            _warn(msg)
        text = str(desc).strip()
        if include_ocr and ocr and str(ocr).strip() and str(ocr).strip() not in text:
            text += "\nTekst widoczny na obrazie: " + str(ocr).strip()
        out[str(key)] = text
    return out


def build_payload(exam_id: str, answers_by_id: Mapping[str, Any], ids: Iterable[str]) -> dict:
    """{"exam_id", "answers"} with every id exactly once, in template order; missing -> "", non-strings coerced,
    over-long answers cut to the site limit (with a warning)."""
    ids = [str(i) for i in ids]
    norm = {str(k): v for k, v in answers_by_id.items()}
    extra = [k for k in norm if k not in set(ids)]
    if extra:
        _warn(f"dropping answers for ids not in the template: {extra}")
    answers = []
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        val = norm.get(i, "")
        if val is None:
            val = ""
        elif not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)
        if utf16_len(val) > MAX_ANSWER_CHARS:
            _warn(f"answer {i} longer than {MAX_ANSWER_CHARS} characters; cut to the limit")
            val = _cut_utf16(val, MAX_ANSWER_CHARS)
        answers.append({"id": i, "answer": val})
    return {"exam_id": str(exam_id), "answers": answers}


def write_answers(path: str | os.PathLike, exam_id: str, answers_by_id: Mapping[str, Any],
                  ids: Iterable[str] | None = None, *, exam_dir: str | os.PathLike | None = None) -> dict:
    """Write answers.json (UTF-8, no BOM, ensure_ascii=False, atomic). The id list comes from `ids`, else the
    exam_dir template, else the keys of answers_by_id. Returns the written payload."""
    if ids is None and exam_dir is not None:
        tpl_exam_id, ids = load_template(exam_dir)
        if str(tpl_exam_id) != str(exam_id):
            _warn(f"exam_id {exam_id!r} differs from the template's {tpl_exam_id!r}")
    if ids is None:
        _warn("no template ids given; writing only the ids present in answers_by_id")
        ids = list(answers_by_id)
    payload = build_payload(exam_id, answers_by_id, ids)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return payload


def utf16_len(s: str) -> int:
    """String length as JavaScript counts it (the site's limit is checked in the browser)."""
    return len(s.encode("utf-16-le")) // 2


def _cut_utf16(s: str, limit: int) -> str:
    out, n = [], 0
    for ch in s:
        w = 2 if ord(ch) > 0xFFFF else 1
        if n + w > limit:
            break
        out.append(ch)
        n += w
    return "".join(out)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def _pairs_no_dupes(pairs: list[tuple[str, Any]]) -> dict:
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate key(s) in one JSON object: {dupes}")
    return dict(pairs)


def validate_payload(payload: Any, exam_id: str, ids: list[str]) -> list[str]:
    """Structural rules on an already-parsed payload."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"top level must be a JSON object with exam_id and answers, got {type(payload).__name__}"]
    extra = [k for k in payload if k not in TOP_LEVEL_KEYS]
    if extra:
        problems.append(f"only exam_id and answers are allowed at the top level; found extra key(s) {extra}")
    if "exam_id" not in payload:
        problems.append("missing exam_id")
    elif not isinstance(payload["exam_id"], str):
        problems.append(f"exam_id must be a string, got {type(payload['exam_id']).__name__}")
    elif payload["exam_id"] != exam_id:
        problems.append(f"exam_id is {payload['exam_id']!r}, expected {exam_id!r}")
    if "answers" not in payload:
        problems.append("missing answers")
        return problems
    answers = payload["answers"]
    if not isinstance(answers, list):
        problems.append(f"answers must be a list, got {type(answers).__name__}")
        return problems
    counts: dict[str, int] = {}
    for n, entry in enumerate(answers):
        where = f"answers[{n}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} must be an object with id and answer, got {type(entry).__name__}")
            continue
        eid = entry.get("id")
        if isinstance(eid, str):
            where = f"answers[{n}] (id {eid!r})"
        extra = [k for k in entry if k not in ENTRY_KEYS]
        if extra:
            problems.append(f"{where}: only id and answer are allowed; found extra key(s) {extra}")
        if "id" not in entry:
            problems.append(f"{where}: missing id")
        elif not isinstance(eid, str):
            problems.append(f"{where}: id must be a string (e.g. \"2.1\"), got {type(eid).__name__} {eid!r}")
        else:
            counts[eid] = counts.get(eid, 0) + 1
        if "answer" not in entry:
            problems.append(f"{where}: missing answer")
        elif not isinstance(entry["answer"], str):
            got = "null" if entry["answer"] is None else type(entry["answer"]).__name__
            problems.append(f"{where}: answer must be a string (use \"\" for blank), got {got}")
        elif utf16_len(entry["answer"]) > MAX_ANSWER_CHARS:
            problems.append(f"{where}: answer has {utf16_len(entry['answer'])} characters, limit {MAX_ANSWER_CHARS}")
    dupes = sorted((k for k, c in counts.items() if c > 1), key=_id_key)
    if dupes:
        problems.append(f"duplicate id(s): {dupes}")
    want = set(ids)
    missing = [i for i in ids if i not in counts]
    unexpected = sorted((k for k in counts if k not in want), key=_id_key)
    if missing:
        problems.append(f"missing id(s): {missing}")
    if unexpected:
        problems.append(f"unexpected id(s) not in the exam: {unexpected}")
    return problems


def validate_answers(path: str | os.PathLike, exam_dir: str | os.PathLike) -> list[str]:
    """Every upload-site rule; returns human-readable problems (empty list = the site will accept the file)."""
    p = Path(path)
    problems: list[str] = []
    try:
        exam_id, ids = load_template(exam_dir)
    except Exception as e:  # noqa: BLE001
        return [f"cannot read exam package {exam_dir}: {e}"]
    try:
        meta, items = load_exam(exam_dir)
        exam_ids = [it["id"] for it in items]
        if set(exam_ids) != set(ids) or str(meta["exam_id"]) != exam_id:
            problems.append("answers-template.json and exam.json disagree on exam_id or ids; checked against the "
                            "template")
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot read exam.json: {e}")
    if not p.is_file():
        return problems + [f"{p} does not exist"]
    if p.suffix.lower() != ".json":
        problems.append(f"file name must end with .json: {p.name}")
    blob = p.read_bytes()
    if not blob:
        return problems + ["file is empty"]
    if len(blob) > MAX_FILE_BYTES:
        problems.append(f"file is {len(blob)} bytes; limit is {MAX_FILE_BYTES} (1 MiB)")
    if blob.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM; write plain UTF-8")
        blob = blob[3:]
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as e:
        return problems + [f"file is not valid UTF-8: {e}"]
    try:
        payload = json.loads(text, object_pairs_hook=_pairs_no_dupes, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        return problems + [f"invalid JSON: {e.msg} at line {e.lineno} column {e.colno}"]
    except ValueError as e:
        return problems + [f"invalid JSON: {e}"]
    return problems + validate_payload(payload, exam_id, ids)


def lint_answers(path: str | os.PathLike, exam_dir: str | os.PathLike) -> list[str]:
    """Warnings the site does not enforce: blanks, closed answers off the exact syntax, essay topic/length,
    leftover reasoning markers."""
    from harness.postprocess import closed_syntax_ok, essay_word_count, has_topic_header
    from harness.prompts import ESSAY_MIN_WORDS, item_kind

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        answers = {a["id"]: a["answer"] for a in payload["answers"] if isinstance(a, dict)}
        _, items = load_exam(exam_dir)
    except Exception as e:  # noqa: BLE001
        return [f"cannot lint: {e}"]
    warns: list[str] = []
    blanks = [it["id"] for it in items if not str(answers.get(it["id"]) or "").strip()]
    if blanks:
        warns.append(f"{len(blanks)} blank answer(s): {blanks}")
    think = [i for i, a in answers.items() if isinstance(a, str) and re.search(r"(?i)</?think>", a)]
    if think:
        warns.append(f"think markers left in: {think}")
    dry = [i for i, a in answers.items() if isinstance(a, str) and "[dry-run]" in a.casefold()]
    if dry:
        warns.append(f"DRY-RUN placeholders in {len(dry)} answer(s) - do not submit this file")
    for it in items:
        ans = answers.get(it["id"])
        if not isinstance(ans, str) or not ans.strip():
            continue
        kind = item_kind(it)
        if kind.startswith("closed_") and not closed_syntax_ok(it, ans):
            warns.append(f"{it['id']}: closed answer does not follow answer_format syntax: {ans[:80]!r}")
        if kind == "essay":
            words = essay_word_count(ans)
            if words < ESSAY_MIN_WORDS:
                warns.append(f"{it['id']}: essay has {words} words (< {ESSAY_MIN_WORDS})")
            if not has_topic_header(ans):
                warns.append(f"{it['id']}: essay does not start with the chosen topic number")
    return warns


def _id_key(s: str) -> tuple:
    parts = []
    for p in str(s).split("."):
        parts.append((0, int(p), "") if p.isdigit() else (1, math.inf, p))
    return tuple(parts)
