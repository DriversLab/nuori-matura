#!/usr/bin/env python
"""Sit a history-matura exam package against an OpenAI-compatible server and write answers.json + a run log.

  # base (organizer protocol) and tuned from ONE llama-server started with --lora adapter.gguf
  python scripts/run_exam.py --exam-dir exams/mock --label base  --lora-scale 0 --descriptions runs/desc/mock.json
  python scripts/run_exam.py --exam-dir exams/mock --label tuned --lora-scale 1 --descriptions runs/desc/mock.json

  # plumbing check without any server
  python scripts/run_exam.py --exam-dir exams/mock --dry-run
  # see the exact prompts
  python scripts/run_exam.py --exam-dir exams/mock --show-prompts --only 1,3,26

Defaults follow the organizers' benchmark protocol: their system prompt, style "organizer", greedy (temperature 0,
top_k 1), seed 42, 2048 new tokens (4096 for the essay), repeat_penalty 1.0. Start llama-server with an 8192-token
context per slot (-c 8192, or -np N -c N*8192 for --parallel N). Only --base-url is contacted.
Outputs default to runs/<label>/answers.json and runs/<label>/run_log.jsonl (runs/ is gitignored: it holds exam
text). The log has one "run" record, one "item" record per item (prompt hash, messages, raw output, cleaned answer,
tokens, latency) and a "summary" record.
Exit status: 0 ok, 1 validation problems or failed items, 2 bad options / RAG index not loadable, 3 server unreachable.

Tuned-run extras. All are OFF by default, so a base run sends exactly the requests it always did; every extra call is
logged in the item's record:
  --style rag --rag-index data/rag/index [--rag-k 4]
      reference passages retrieved per item (harness.rag: BM25, no model) go before the task as
      "Materiały pomocnicze: ..." (harness.prompts style "rag"); the record's "rag" field lists query and passages
  --essay-mode plan
      the essay in two steps: the model picks the topic it can argue best and writes a short plan (teza, 3-4
      arguments with facts, counter-argument, wniosek), then writes the essay (450-650 words) from the plan, starting
      with "Temat N."; with --style rag, passages are retrieved again for the chosen topic. Logged as "essay_plan"
      and "essay_messages". --essay-retry still extends an essay under 300 words
  --vote K
      closed items: K sampled answers (temperature 0.7, seeds 1..K) plus the greedy one; majority of the normalised
      answers (per row for true/false and multi-part items), ties -> greedy. Sampling stops early once the remaining
      samples cannot change the result. Logged as "votes"
  --repair-labels
      open items whose question lists answer labels ("Rozstrzygnięcie:", "Uzasadnienie:", "Nazwa:"): when the answer
      lacks one, the model is asked once more with a short repair instruction; the new answer is used if it lacks
      fewer labels. Logged as "repair"
  tuned: --style rag --rag-index data/rag/index --essay-mode plan --vote 5 --repair-labels --lora-scale 1
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import datetime as dt
import hashlib
import importlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

import requests

from harness import exam_io, prompts
from harness.client import DEFAULT_BASE_URL, ChatClient, ChatResult, list_models, run_parallel
from harness.postprocess import (clean, clean_essay_with_topic, clean_text, detect_essay_topic, missing_labels,
                                 vote_closed, vote_decided)
from harness.prompts import (CLOSED_KINDS, ESSAY_MIN_WORDS, ESSAY_PLAN_MAX_CHARS, ESSAY_PLAN_MAX_TOKENS,
                             ORGANIZER_SYSTEM_PROMPT, STYLES, answer_labels, build_messages, format_spec, item_kind,
                             max_new_tokens, missing_descriptions, placeholder_answer, prompt_sha256)

DEFAULT_DESCRIPTIONS = "descriptions.json"  # harness/vision.py CACHE_FILENAME (library default, next to exam.json)
ESSAY_CONTINUE = ("Twoje wypracowanie ma {words} wyrazów, a wymagane jest co najmniej {minimum}. Napisz jego dalszą "
                  "część: kolejne akapity z argumentami i przykładami oraz zakończenie. Nie powtarzaj wcześniejszego "
                  "tekstu ani numeru tematu.")
ESSAY_MODES = ("single", "plan")
ESSAY_RAG_MODES = ("topic", "entities")
ESSAY_ENTITY_MAX = 6          # named entities from the plan, each queried on its own
ESSAY_ENTITY_HITS = 2         # passages kept per entity (only from articles whose title names the entity)
ESSAY_ENTITY_MAX_CHARS = 4800 # materials budget for the essay in entities mode (the plan names 3-4 rulers/periods)
VOTE_TEMPERATURE = 0.7
VOTE_TOP_K = 40            # the greedy protocol sends top_k 1, which would make every "sample" greedy
RAG_K = 4
RAG_MAX_CHARS = 2400
RAG_MIN_SCORE = 0.0   # BM25 score below which a passage is dropped (0 = keep all). Off-topic hits on the mock scored < ~50
RAG_QUERY_MAX_CHARS = 1500
# plan scaffolding words, dropped from the essay's retrieval query (they would only match unrelated passages)
_PLAN_LABEL = re.compile(r"(?im)^[ \t]*(?:[-*•][ \t]*)?(?:temat[ \t]*\d{1,2}[ \t]*[.:]?|teza|argument[ \t]*\d*|"
                         r"kontrargument|wnioski?)(?:[ \t]*\([^)\n]*\))?[ \t]*[:.\-–—]?[ \t]*")


def default_description_paths(exam_dir: Path) -> list[Path]:
    """Where describe_images writes by default: scripts/describe_images.py -> runs/desc/<exam folder name>.json,
    harness.vision.describe_images() without cache_path -> <exam-dir>/descriptions.json. First existing one wins."""
    folder = exam_io.exam_json_path(exam_dir).parent
    return [ROOT / "runs" / "desc" / f"{folder.resolve().name}.json", folder / DEFAULT_DESCRIPTIONS]


def descriptions_sha256(descriptions: dict[str, str]) -> str | None:
    """Fingerprint of the descriptions a run used (base and tuned must share it)."""
    if not descriptions:
        return None
    blob = json.dumps(descriptions, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _lora(value: str) -> float | None:
    if value is None or str(value).strip().lower() in ("", "none", "null", "off"):
        return None
    return float(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam-dir", required=True, help="folder with exam.json (+ images/, answers-template.json)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"OpenAI-compatible server (default {DEFAULT_BASE_URL})")
    ap.add_argument("--model", help="model name/alias to send (default: first id from GET /v1/models)")
    ap.add_argument("--lora-scale", type=_lora, default=None,
                    help='per-request LoRA scale for llama-server ("none" = do not send; 0 = base, 1 = tuned)')
    ap.add_argument("--label", default="base", help="run label, e.g. base | tuned (default base)")
    ap.add_argument("--descriptions",
                    help='image-description cache JSON (scripts/describe_images.py output; default: the first that '
                         'exists of runs/desc/<exam-dir name>.json and <exam-dir>/descriptions.json; "none" = no '
                         'descriptions)')
    ap.add_argument("--include-ocr", action="store_true", help="append cached OCR text to image descriptions")
    ap.add_argument("--style", choices=STYLES, default="organizer",
                    help='prompt style: "organizer" = benchmark protocol (default), "formatted" = + answer-format line, '
                         '"rag" = organizer + retrieved reference passages (needs --rag-index; tuned runs only)')
    ap.add_argument("--rag-index", help="RAG index for --style rag (harness.rag.load_index path, e.g. data/rag/index)")
    ap.add_argument("--rag-k", type=int, default=RAG_K, help=f"passages retrieved per item (default {RAG_K})")
    ap.add_argument("--rag-min-score", type=float, default=RAG_MIN_SCORE,
                    help="drop retrieved passages whose BM25 score is below this (default 0 = keep all). A weak match "
                         "then gives no materials instead of off-topic ones (mock: off-topic top hits scored 23-46, "
                         "useful ones 50-160)")
    ap.add_argument("--rag-max-chars", type=int, default=RAG_MAX_CHARS,
                    help=f"characters of passages per prompt (harness.rag.format_passages; default {RAG_MAX_CHARS})")
    ap.add_argument("--essay-rag", choices=ESSAY_RAG_MODES, default="topic",
                    help="essay retrieval in --essay-mode plan: 'topic' = one query from the topic + plan (default); "
                         "'entities' = one short query per person/period the plan names (e.g. 'Kazimierz Odnowiciel'), "
                         "keeping only passages from articles whose title names it, no score cutoff")
    ap.add_argument("--essay-mode", choices=ESSAY_MODES, default="single",
                    help='"single" = one request (default, organizer protocol); "plan" = topic choice + plan, then '
                         'the essay from the plan')
    ap.add_argument("--vote", type=int, default=1, metavar="K",
                    help="closed items: K sampled answers + the greedy one, majority of the normalised answers "
                         "(default 1 = greedy only)")
    ap.add_argument("--cache-prompt", action="store_true",
                    help="let llama-server reuse cached prompt prefixes (faster, but the same prompt can then get a "
                         "different greedy answer depending on earlier requests; off by default for reproducible runs)")
    ap.add_argument("--vote-temperature", type=float, default=VOTE_TEMPERATURE,
                    help=f"sampling temperature of the --vote samples (default {VOTE_TEMPERATURE})")
    ap.add_argument("--repair-labels", action="store_true",
                    help="open items: if the answer lacks one of the question's answer labels, ask once more")
    ap.add_argument("--system-prompt-file", help="replace the organizer system prompt with this file's text")
    ap.add_argument("--no-system-prompt", action="store_true", help="send only the user turn")
    ap.add_argument("--parallel", type=int, default=4, help="concurrent requests (server needs -np N)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per request")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llama-extras", action="store_true",
                    help="omit top_k/repeat_penalty/cache_prompt (for servers that reject unknown fields)")
    ap.add_argument("--out", help="answers.json path (default runs/<label>/answers.json)")
    ap.add_argument("--log", help="run log JSONL (default next to --out: run_log.jsonl)")
    ap.add_argument("--only", action="append", default=[], metavar="IDS",
                    help="debug: only these item ids (comma/space separated, repeatable); other answers are kept "
                         "from an existing --out file, else left blank")
    ap.add_argument("--resume", action="store_true",
                    help="reuse raw outputs from the log for items whose prompt and settings are unchanged")
    ap.add_argument("--max-tokens-short", type=int, default=None,
                    help="output-token limit for non-essay items (default: the protocol's 2048). Raise it for a model "
                         "whose thinking mode spends tokens before answering; serve with --ctx large enough for "
                         "prompt + limit")
    ap.add_argument("--max-tokens-essay", type=int, default=None,
                    help="output-token limit for the essay (default: the protocol's 4096)")
    ap.add_argument("--essay-retry", action="store_true",
                    help=f"if the essay has < {ESSAY_MIN_WORDS} words, ask once to continue it and append")
    ap.add_argument("--dry-run", action="store_true", help="no server: echo syntax placeholders to test plumbing")
    ap.add_argument("--show-prompts", action="store_true", help="print the rendered messages and exit")
    args = ap.parse_args(argv)
    if args.style == "rag" and not args.rag_index:
        ap.error("--style rag needs --rag-index (e.g. data/rag/index)")
    if args.rag_index and args.style != "rag":
        ap.error("--rag-index is only used with --style rag (the base run must stay without passages)")
    if args.rag_k < 1 or args.rag_max_chars < 1:
        ap.error("--rag-k and --rag-max-chars must be >= 1")
    if args.vote < 1:
        ap.error("--vote must be >= 1 (1 = greedy only)")
    return args


def _only_ids(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        for tok in v.replace(",", " ").split():
            if tok not in out:
                out.append(tok)
    return out


def _dry_raw(item: dict) -> str:
    kind = item_kind(item)
    ph = placeholder_answer(item)
    if kind.startswith("closed_"):
        return "Odpowiedź:\n" + ph  # exercises the closed-answer cleaner
    if kind == "essay":
        return ph + " [DRY-RUN] To jest testowe wypracowanie." * 5
    return ph


def _dry_plan(item: dict) -> str:
    topics = sorted(format_spec(item).topics) or [1]
    return (f"Temat {topics[0]}.\nTeza: [DRY-RUN] testowa teza.\nArgument 1: [DRY-RUN] testowy argument.\n"
            "Kontrargument: [DRY-RUN] testowy kontrargument.\nWniosek: [DRY-RUN] testowy wniosek.")


def load_rag(path: str) -> tuple[Any, Any]:
    """(harness.rag module, index). Imported lazily: only --style rag runs touch the RAG code."""
    rag = importlib.import_module("harness.rag")
    return rag, rag.load_index(path)


def _passage_record(hit: Mapping[str, Any]) -> dict:
    out: dict[str, Any] = {k: (None if hit.get(k) is None else str(hit.get(k))) for k in ("title", "section", "url")}
    try:
        out["score"] = round(float(hit.get("score")), 4)
    except (TypeError, ValueError):
        out["score"] = None
    out["text"] = str(hit.get("text") or "")
    return out


def item_query(rag: Any, item: Mapping[str, Any]) -> str:
    """harness.rag.build_query(item); if it fails, the raw question + source text (the exam must go on)."""
    try:
        return str(rag.build_query(item))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: build_query failed for item {item.get('id')} ({type(e).__name__}: {e}); using the raw "
              "question text", file=sys.stderr)
        return f"{item.get('question') or ''}\n{item.get('source_text') or ''}".strip()


def retrieve(rag: Any, index: Any, query: str, k: int, max_chars: int,
             min_score: float = 0.0) -> tuple[str, dict]:
    """(passages text for build_messages, log record {query, k, max_chars, chars, passages[, dropped_low_score,
    error]}). Hits scoring below min_score are dropped (logged). A failed search is logged and gives no passages
    instead of stopping the exam."""
    info: dict[str, Any] = {"query": query, "k": k, "max_chars": max_chars, "chars": 0, "passages": []}
    try:
        hits = list(index.search(query, k=k) or [])
        if min_score > 0:
            low = [h for h in hits if float(h.get("score") or 0.0) < min_score]
            if low:
                info["dropped_low_score"] = [_passage_record(h) for h in low]
                hits = [h for h in hits if float(h.get("score") or 0.0) >= min_score]
        text = rag.format_passages(hits, max_chars=max_chars) if hits else ""
        info["passages"] = [_passage_record(h) for h in hits]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        print(f"WARNING: retrieval failed ({info['error']}); this prompt gets no passages", file=sys.stderr)
        return "", info
    info["chars"] = len(text)
    return text, info


def essay_rag_query(topic_text: str | None, plan: str) -> str:
    """Retrieval query for the chosen essay topic: the topic text plus the plan's content words (names, events,
    dates the model means to use), without the plan's scaffolding labels."""
    plan_terms = _PLAN_LABEL.sub("", plan or "")
    return f"{topic_text or ''}\n{plan_terms}".strip()[:RAG_QUERY_MAX_CHARS]


_NAME_WORD = r"[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+"
_ENTITY = re.compile(rf"\b({_NAME_WORD}(?:[ \t]+(?:[IVX]{{1,4}}\b|{_NAME_WORD})){{1,3}})")
_ENTITY_STOP = {"Temat", "Teza", "Argument", "Kontrargument", "Wniosek", "Rzeczpospolita", "Polska", "Królestwo",
                "Księstwo", "Państwo", "Kościół", "Rzesza", "Cesarstwo"}


def essay_entities(plan: str, topic_text: str | None = None, limit: int = ESSAY_ENTITY_MAX) -> list[str]:
    """Multi-word proper names (rulers, dynasties, treaties, 'Mieszko II', 'Kazimierz Odnowiciel') in the plan, then
    the topic text, in order of appearance, without duplicates and without names that start with a scaffolding or
    generic word."""
    out: list[str] = []
    for text in (_PLAN_LABEL.sub("", plan or ""), topic_text or ""):
        for m in _ENTITY.finditer(text):
            name = " ".join(m.group(1).split())
            if name.split()[0] in _ENTITY_STOP or name in out:
                continue
            out.append(name)
            if len(out) >= limit:
                return out
    return out


def retrieve_entities(rag: Any, index: Any, names: list[str], per_entity: int, max_chars: int) -> tuple[str, dict]:
    """Essay materials in entities mode: for each name, the top passages from articles whose title shares a word
    (compared by 5-letter stems, so inflected forms match) with it. Failures give no passages, like retrieve()."""
    info: dict[str, Any] = {"mode": "entities", "queries": names, "max_chars": max_chars, "chars": 0, "passages": []}
    kept: list[dict] = []
    seen: set[str] = set()
    try:
        for name in names:
            stems = {w.lower()[:5] for w in name.split() if len(w) >= 4}   # Polish inflection: 'Kazimierza' ~ 'Kazimierz'
            n = 0
            for h in index.search(name, k=6) or []:
                title = str(h.get("title") or "")
                title_stems = {w.lower()[:5] for w in title.replace("(", " ").replace(")", " ").split() if len(w) >= 4}
                if not stems & title_stems:
                    continue
                key = (title, str(h.get("text") or "")[:80])
                if str(key) in seen:
                    continue
                seen.add(str(key))
                kept.append(h)
                n += 1
                if n >= per_entity:
                    break
        text = rag.format_passages(kept, max_chars=max_chars) if kept else ""
        info["passages"] = [_passage_record(h) for h in kept]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        print(f"WARNING: entity retrieval failed ({info['error']}); the essay gets no passages", file=sys.stderr)
        return "", info
    info["chars"] = len(text)
    return text, info


def _call_record(r: ChatResult) -> dict:
    return {"raw": r.text, "finish_reason": r.finish_reason, "usage": r.usage, "latency_s": round(r.latency_s, 3),
            "attempts": r.attempts}


def _add_usage(total: dict, usage: Mapping | None) -> None:
    for k in ("prompt_tokens", "completion_tokens"):
        total[k] = total.get(k, 0) + int((usage or {}).get(k) or 0)


def _read_log(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") == "item" and not rec.get("error"):
            latest[str(rec.get("id"))] = rec
    return latest


def _feature_key(rec: dict) -> tuple:
    """The extras that change an item's answer, per kind; records written before the extras existed map to the
    defaults, so --resume still reuses them for default runs."""
    kind = str(rec.get("kind") or "")
    vote = int(rec.get("vote") or 1)
    return ((rec.get("essay_mode") or "single") if kind == "essay" else None,
            (vote, rec.get("vote_temperature")) if kind.startswith("closed_") and vote > 1 else None,
            bool(rec.get("repair_labels")) if kind == "open" else None)


def _settings_key(rec: dict) -> tuple:
    return (rec.get("label"), rec.get("prompt_sha256"), rec.get("lora_scale"), rec.get("model"),
            rec.get("max_tokens"), rec.get("seed"), bool(rec.get("dry_run")), rec.get("base_url"),
            _feature_key(rec))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.monotonic()
    exam_dir = Path(args.exam_dir)
    meta, items = exam_io.load_exam(exam_dir)
    exam_id, template_ids = exam_io.load_template(exam_dir)
    if str(meta["exam_id"]) != exam_id:
        print(f"WARNING: exam.json exam_id {meta['exam_id']!r} != template {exam_id!r}; using the template's",
              file=sys.stderr)

    desc_problems: list[str] = []
    explicit_none = bool(args.descriptions and args.descriptions.strip().lower() == "none")
    if args.descriptions is None:
        found = next((p for p in default_description_paths(exam_dir) if p.is_file()), None)
        if found is not None:
            args.descriptions = str(found)
            print(f"NOTE: using image descriptions from {args.descriptions}", file=sys.stderr)
    if explicit_none:
        args.descriptions = None
    descriptions = exam_io.load_descriptions(args.descriptions, items, include_ocr=args.include_ocr,
                                             problems=desc_problems) if args.descriptions else {}
    missing = missing_descriptions(items, descriptions)
    if missing and not explicit_none:
        print(f"WARNING: {len(missing)} image(s) have no description, so their items keep the raw "
              f"[Obraz: ...] marker: {missing}. Run scripts/describe_images.py --exam-dir {args.exam_dir} first "
              "(or pass --descriptions <cache>); base and tuned must use the same descriptions.", file=sys.stderr)

    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    else:
        system_prompt = ORGANIZER_SYSTEM_PROMPT

    only = _only_ids(args.only)
    unknown = [i for i in only if i not in {it["id"] for it in items}]
    if unknown:
        print(f"ERROR: --only ids not in the exam: {unknown}", file=sys.stderr)
        return 1
    selected = [it for it in items if not only or it["id"] in only]

    rag = index = None
    if args.style == "rag":
        try:
            rag, index = load_rag(args.rag_index)
        except Exception as e:  # noqa: BLE001 - missing module, missing/corrupt index: say so and stop
            print(f"ERROR: cannot load the RAG index {args.rag_index!r} via harness.rag: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 2

    prepared = []
    for it in selected:
        passages, rag_info = None, None
        if index is not None:
            passages, rag_info = retrieve(rag, index, item_query(rag, it), args.rag_k, args.rag_max_chars,
                                          args.rag_min_score)
        msgs = build_messages(it, descriptions, system_prompt=system_prompt, style=args.style, passages=passages)
        prepared.append((it, msgs, rag_info))

    if args.show_prompts:
        for it, msgs, rag_info in prepared:
            print(f"===== {it['id']} [{item_kind(it)}] max_new_tokens={max_new_tokens(it)} "
                  f"sha256={prompt_sha256(msgs)[:12]}")
            if rag_info is not None:
                print("rag passages: " + "; ".join(f"{p['title']} ({p['score']})" for p in rag_info["passages"]))
            for m in msgs:
                print(f"--- {m['role']}\n{m['content']}")
        return 0

    out = Path(args.out) if args.out else ROOT / "runs" / args.label / "answers.json"
    log_path = Path(args.log) if args.log else out.with_name("run_log.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model = args.model
    if not args.dry_run:
        try:
            if not model:
                ids = list_models(args.base_url)
                model = ids[0] if ids else None
                if len(ids) > 1:
                    print(f"NOTE: server lists {ids}; using {model!r} (pass --model to choose)", file=sys.stderr)
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"ERROR: cannot reach {args.base_url}: {e}", file=sys.stderr)
            return 3
        except Exception as e:  # noqa: BLE001 - /v1/models is optional
            print(f"NOTE: GET /v1/models failed ({e}); sending no model name", file=sys.stderr)

    if args.lora_scale is None and not args.dry_run:
        print("NOTE: no --lora-scale: requests carry no \"lora\" field, so the server's default adapter scale "
              "decides base vs tuned (serve_model.sh: 0 unless --lora-apply)", file=sys.stderr)
    client = ChatClient(args.base_url, model, timeout=args.timeout, retries=args.retries, lora_scale=args.lora_scale,
                        seed=args.seed, llama_extras=not args.no_llama_extras, cache_prompt=args.cache_prompt)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    common = {"label": args.label, "exam_id": exam_id, "run_id": run_id, "style": args.style,
              "lora_scale": args.lora_scale, "model": model, "seed": args.seed, "base_url": args.base_url,
              "dry_run": bool(args.dry_run), "essay_mode": args.essay_mode, "vote": args.vote,
              "vote_temperature": args.vote_temperature if args.vote > 1 else None,
              "repair_labels": bool(args.repair_labels), "rag_index": args.rag_index,
              "rag_k": args.rag_k if index is not None else None,
              "rag_min_score": args.rag_min_score if index is not None else None,
              "essay_rag": args.essay_rag if index is not None else None}
    vote_overrides: dict[str, Any] = {"temperature": args.vote_temperature}
    if not args.no_llama_extras:
        vote_overrides["top_k"] = VOTE_TOP_K

    previous = _read_log(log_path) if (args.resume or only) else {}
    append = bool(args.resume or only) and log_path.is_file()
    log_f = log_path.open("a" if append else "w", encoding="utf-8")

    def log(rec: dict) -> None:
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

    log({"type": "run", **common, "started": dt.datetime.now().isoformat(timespec="seconds"),
         "exam_dir": str(exam_dir), "only": only, "parallel": args.parallel, "essay_retry": args.essay_retry,
         "system_prompt": system_prompt, "descriptions": args.descriptions,
         "descriptions_used": len(descriptions), "descriptions_sha256": descriptions_sha256(descriptions),
         "missing_descriptions": missing, "description_problems": desc_problems,
         "llama_extras": not args.no_llama_extras,
         "decoding": {k: v for k, v in client.payload([], 0).items() if k not in ("messages", "max_tokens")},
         "rag_max_chars": args.rag_max_chars if index is not None else None,
         "vote_decoding": ({k: v for k, v in client.payload([], 0, **vote_overrides).items()
                            if k not in ("messages", "max_tokens", "seed")} if args.vote > 1 else None)})

    def call(messages: list[dict], max_tokens: int, dry_text: str, **overrides: Any) -> ChatResult:
        if args.dry_run:
            return ChatResult(text=dry_text, finish_reason="stop")
        return client.chat(messages, max_tokens, **overrides)

    def plan_step(item: dict, msgs: list[dict]) -> tuple[dict, list[dict]]:
        """--essay-mode plan, step 1: topic choice + plan. Returns (log record, step-2 messages)."""
        spec = format_spec(item)
        plan_msgs = prompts.essay_plan_messages(msgs, item)
        try:
            r = call(plan_msgs, ESSAY_PLAN_MAX_TOKENS, _dry_plan(item))
        except Exception as e:  # noqa: BLE001 - the essay is still asked, in one step
            return {"messages": plan_msgs, "raw": None, "error": f"{type(e).__name__}: {e}", "plan": "",
                    "topic": None, "topic_source": None, "fallback": "plan request failed: essay asked in one step"}, msgs
        plan = clean_text(r.text, item["id"])[:ESSAY_PLAN_MAX_CHARS].strip()
        topic, source = detect_essay_topic(r.text, spec.topics)
        info = {"messages": plan_msgs, **_call_record(r), "plan": plan, "topic": topic, "topic_source": source}
        if not plan:
            info["fallback"] = "empty plan: essay asked in one step"
            return info, msgs
        base = msgs
        if index is not None and topic is not None and args.essay_rag == "entities":
            passages, info["rag"] = retrieve_entities(rag, index, essay_entities(plan, spec.topics.get(topic)),
                                                      ESSAY_ENTITY_HITS, ESSAY_ENTITY_MAX_CHARS)
            base = build_messages(item, descriptions, system_prompt=system_prompt, style=args.style,
                                  passages=passages)
        elif index is not None and topic is not None:
            passages, info["rag"] = retrieve(rag, index, essay_rag_query(spec.topics.get(topic), plan), args.rag_k,
                                             args.rag_max_chars, args.rag_min_score)
            base = build_messages(item, descriptions, system_prompt=system_prompt, style=args.style,
                                  passages=passages)
        return info, prompts.essay_from_plan_messages(base, item, plan, topic)

    def repair_step(item: dict, gen_msgs: list[dict], raw: str, mnt: int, labels: list[str], missing: list[str],
                    old: dict | None) -> dict:
        """--repair-labels: one follow-up turn asking for the full answer with the question's labels."""
        if old and old.get("raw") is not None:
            return {k: v for k, v in old.items() if k not in ("missing_after", "used")}
        instruction = prompts.label_repair_instruction(labels, missing)
        follow = gen_msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": instruction}]
        try:
            return {"instruction": instruction, **_call_record(call(follow, mnt, _dry_raw(item)))}
        except Exception as e:  # noqa: BLE001 - the first answer stands
            return {"instruction": instruction, "raw": None, "error": f"{type(e).__name__}: {e}"}

    def vote_samples(item: dict, msgs: list[dict], greedy_raw: str, mnt: int) -> list[dict]:
        """--vote K: samples with seeds 1..K until the remaining ones cannot change the majority."""
        samples: list[dict] = []
        for seed in range(1, args.vote + 1):
            got = [s["raw"] for s in samples if s.get("raw") is not None]
            if vote_decided(item, greedy_raw, got, args.vote - len(samples)):
                break
            try:
                r = call(msgs, mnt, _dry_raw(item), **{**vote_overrides, "seed": seed})
                samples.append({"seed": seed, **_call_record(r)})
            except Exception as e:  # noqa: BLE001 - a failed sample is just a missing vote
                samples.append({"seed": seed, "raw": None, "error": f"{type(e).__name__}: {e}"})
        return samples

    def solve(job: tuple[dict, list[dict], dict | None]) -> dict:
        item, msgs, rag_info = job
        kind = item_kind(item)
        mnt = max_new_tokens(item)
        if kind == "essay" and args.max_tokens_essay:
            mnt = args.max_tokens_essay
        elif kind != "essay" and args.max_tokens_short:
            mnt = args.max_tokens_short
        rec = {"type": "item", **common, "id": item["id"], "kind": kind, "max_points": item.get("max_points"),
               "prompt_sha256": prompt_sha256(msgs), "max_tokens": mnt, "messages": msgs}
        if rag_info is not None:
            rec["rag"] = rag_info
        prev = previous.get(item["id"]) if args.resume else None
        plan_mode = kind == "essay" and args.essay_mode == "plan"
        gen_msgs = msgs            # the messages whose reply is `raw`
        plan_topic = None
        extra: list[dict] = []     # records of the extra calls made now (token / latency totals)
        if prev and _settings_key(prev) == _settings_key(rec):
            res = ChatResult(text=prev["raw"], finish_reason=prev.get("finish_reason"), usage=prev.get("usage") or {},
                             latency_s=prev.get("latency_s") or 0.0, attempts=0)
            rec["reused"] = True
            raw = prev["raw"]
            rec["continuation"] = prev.get("continuation")
            if plan_mode and prev.get("essay_plan"):
                rec["essay_plan"] = prev["essay_plan"]
                rec["essay_messages"] = prev.get("essay_messages")
                gen_msgs = prev.get("essay_messages") or msgs
                plan_topic = prev["essay_plan"].get("topic")
        else:
            rec["reused"] = False
            rec["continuation"] = None
            if plan_mode:
                plan_info, gen_msgs = plan_step(item, msgs)
                rec["essay_plan"] = plan_info
                rec["essay_messages"] = gen_msgs
                plan_topic = plan_info.get("topic")
                extra.append(plan_info)
            res = call(gen_msgs, mnt, _dry_raw(item))
            raw = res.text

        def _clean(text: str):
            return clean_essay_with_topic(item, text, plan_topic) if plan_topic is not None else clean(item, text)

        cr = _clean(raw)
        if (cr.kind == "essay" and args.essay_retry and not rec["reused"] and not args.dry_run
                and (cr.essay_words or 0) < ESSAY_MIN_WORDS):
            follow = gen_msgs + [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": ESSAY_CONTINUE.format(words=cr.essay_words,
                                                                                  minimum=ESSAY_MIN_WORDS)}]
            r2 = client.chat(follow, mnt)
            raw = raw.rstrip() + "\n\n" + clean_text(r2.text)
            rec["continuation"] = {"raw": r2.text, "finish_reason": r2.finish_reason, "usage": r2.usage,
                                   "latency_s": round(r2.latency_s, 3), "words_before": cr.essay_words}
            cr = _clean(raw)
        if rec.get("continuation") and not rec["reused"]:
            extra.append(rec["continuation"])

        if kind == "open" and args.repair_labels:
            labels = answer_labels(item)
            missing_before = missing_labels(item, cr.answer, labels) if labels else []
            if missing_before:
                info = repair_step(item, gen_msgs, raw, mnt, labels, missing_before,
                                   (prev or {}).get("repair") if rec["reused"] else None)
                if not rec["reused"]:
                    extra.append(info)
                info.update({"labels": labels, "missing_before": missing_before, "used": False})
                if info.get("raw") is not None:
                    cr2 = clean(item, info["raw"])
                    info["missing_after"] = missing_labels(item, cr2.answer, labels)
                    info["used"] = bool(cr2.answer.strip()) and len(info["missing_after"]) < len(missing_before)
                    if info["used"]:
                        cr = cr2
                rec["repair"] = info

        answer, closed_ok = cr.answer, cr.closed_ok
        if kind in CLOSED_KINDS and args.vote > 1:
            old = (prev or {}).get("votes") if rec["reused"] else None
            if old and old.get("samples") is not None:
                samples = old["samples"]
            else:
                samples = vote_samples(item, msgs, raw, mnt)
                extra.extend(samples)
            vr = vote_closed(item, raw, [s["raw"] for s in samples if s.get("raw") is not None])
            rec["votes"] = {"k": args.vote, "temperature": args.vote_temperature, "samples": samples,
                            "drawn": len(samples), "skipped": args.vote - len(samples),
                            "greedy_answer": vr.greedy_answer, "answer": vr.answer, "changed": vr.changed,
                            "rule": vr.rule, "rules": vr.rules, "tally": vr.tally, "n_votes": vr.n_votes}
            answer, closed_ok = vr.answer, vr.closed_ok

        rec.update({
            "raw": raw if rec.get("continuation") else res.text,
            "answer": answer, "closed_ok": closed_ok, "essay_words": cr.essay_words,
            "essay_topic": cr.essay_topic, "topic_source": cr.topic_source, "notes": cr.notes,
            "finish_reason": res.finish_reason, "truncated": res.finish_reason == "length", "usage": res.usage,
            "latency_s": round(res.latency_s, 3), "attempts": res.attempts, "error": None,
        })
        if rec.get("continuation"):
            rec["raw_first"] = res.text
        if rec["reused"]:
            rec["usage_total"] = prev.get("usage_total") or dict(res.usage)
            rec["latency_total_s"] = prev.get("latency_total_s", rec["latency_s"])
        else:
            total: dict = {}
            _add_usage(total, res.usage)
            for x in extra:
                _add_usage(total, x.get("usage"))
            rec["usage_total"] = total
            rec["latency_total_s"] = round(res.latency_s + sum(x.get("latency_s") or 0 for x in extra), 3)
        return rec

    # longest jobs first so the essay does not become the tail
    jobs = sorted(prepared, key=lambda j: -max_new_tokens(j[0]))
    records: dict[str, dict] = {}
    done = 0
    for (item, msgs, _rag_info), rec, err in run_parallel(solve, jobs, max(1, args.parallel)):
        done += 1
        if err is not None:
            rec = {"type": "item", **common, "id": item["id"], "kind": item_kind(item),
                   "prompt_sha256": prompt_sha256(msgs), "messages": msgs, "raw": None, "answer": "",
                   "error": f"{type(err).__name__}: {err}"}
            print(f"[{done}/{len(jobs)}] {item['id']:>5} ERROR {rec['error'][:200]}", file=sys.stderr)
        else:
            tok = (rec.get("usage") or {}).get("completion_tokens")
            flag = " REUSED" if rec.get("reused") else ""
            flag += " TRUNCATED" if rec.get("truncated") else ""
            flag += " SYNTAX?" if rec.get("closed_ok") is False else ""
            flag += " VOTE-CHANGED" if (rec.get("votes") or {}).get("changed") else ""
            flag += " REPAIRED" if (rec.get("repair") or {}).get("used") else ""
            flag += f" PLAN(topic {rec['essay_plan'].get('topic')})" if rec.get("essay_plan") else ""
            print(f"[{done}/{len(jobs)}] {item['id']:>5} {rec['kind']:<17} {rec['latency_s']:7.1f}s "
                  f"{tok if tok is not None else '-':>5} tok{flag}", file=sys.stderr)
        records[item["id"]] = rec
        if not rec.get("reused"):
            log(rec)

    answers: dict[str, str] = {}
    if only and out.is_file():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
            if old.get("exam_id") == exam_id:
                answers.update({str(a["id"]): a.get("answer") or "" for a in old.get("answers", [])})
                print(f"NOTE: --only: kept the other answers from {out}", file=sys.stderr)
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    for i, rec in records.items():
        answers[i] = rec.get("answer") or ""
    exam_io.write_answers(out, exam_id, answers, template_ids)

    problems = exam_io.validate_answers(out, exam_dir)
    warnings = exam_io.lint_answers(out, exam_dir)
    wall = time.monotonic() - t_start

    recs = list(records.values())
    errors = {r["id"]: r["error"] for r in recs if r.get("error")}
    empty = [i for i in template_ids if not str(answers.get(i) or "").strip()]
    syntax_fail = [r["id"] for r in recs if r.get("closed_ok") is False]
    closed_ok = [r["id"] for r in recs if r.get("closed_ok") is True]
    truncated = [r["id"] for r in recs if r.get("truncated")]
    essays = {r["id"]: {"words": r.get("essay_words"), "topic": r.get("essay_topic"),
                        "topic_source": r.get("topic_source"), "retried": bool(r.get("continuation")),
                        "plan_topic": (r.get("essay_plan") or {}).get("topic")}
              for r in recs if r.get("kind") == "essay" and not r.get("error")}
    ptok = sum((r.get("usage_total") or r.get("usage") or {}).get("prompt_tokens") or 0 for r in recs)
    ctok = sum((r.get("usage_total") or r.get("usage") or {}).get("completion_tokens") or 0 for r in recs)
    gen_s = sum(r.get("latency_total_s", r.get("latency_s")) or 0 for r in recs if not r.get("reused"))
    votes = {r["id"]: {"rule": r["votes"]["rule"], "changed": r["votes"]["changed"], "drawn": r["votes"]["drawn"],
                       "greedy_answer": r["votes"]["greedy_answer"], "answer": r["votes"]["answer"]}
             for r in recs if r.get("votes")}
    repairs = {r["id"]: {"missing_before": r["repair"]["missing_before"],
                         "missing_after": r["repair"].get("missing_after"), "used": r["repair"]["used"]}
               for r in recs if r.get("repair")}
    rag_empty = [r["id"] for r in recs if r.get("rag") is not None and not r["rag"]["passages"]]
    summary = {"type": "summary", **common, "items": len(recs), "empty": empty, "errors": errors,
               "closed_ok": closed_ok, "closed_syntax_failures": syntax_fail, "truncated": truncated,
               "essays": essays, "prompt_tokens": ptok, "completion_tokens": ctok,
               "sum_latency_s": round(gen_s, 1), "wall_s": round(wall, 1), "answers": str(out),
               "validation_problems": problems, "warnings": warnings, "missing_descriptions": missing,
               "descriptions_sha256": descriptions_sha256(descriptions), "votes": votes, "repairs": repairs,
               "rag_items_without_passages": rag_empty}
    log(summary)
    log_f.close()

    print(f"\n== run_exam [{args.label}] {exam_id}: {len(recs)} item(s), style={args.style}, "
          f"lora_scale={args.lora_scale}, model={model}{' DRY-RUN' if args.dry_run else ''}")
    dsha = descriptions_sha256(descriptions)
    print(f"image descriptions: {len(descriptions)} from {args.descriptions or 'none'}"
          f"{f' (sha256 {dsha[:12]}; must match between base and tuned)' if dsha else ''}")
    if missing:
        print(f"images without description ({len(missing)}): {missing}")
    for p in desc_problems:
        print(f"description cache: {p}")
    print(f"empty answers ({len(empty)}): {empty}")
    print(f"closed: {len(closed_ok)} exact syntax, {len(syntax_fail)} unrecoverable {syntax_fail}")
    for i, e in essays.items():
        planned = f", plan chose topic {e['plan_topic']}" if args.essay_mode == "plan" else ""
        print(f"essay {i}: {e['words']} words, topic {e['topic']} ({e['topic_source']})"
              f"{', continued once' if e['retried'] else ''}{planned}")
    if index is not None:
        print(f"rag: {args.rag_index}, k={args.rag_k}, max {args.rag_max_chars} chars; items without passages "
              f"({len(rag_empty)}): {rag_empty}")
    if args.vote > 1:
        changed = [i for i, v in votes.items() if v["changed"]]
        drawn = sum(v["drawn"] for v in votes.values())
        print(f"votes (K={args.vote}, T={args.vote_temperature}): {len(votes)} closed item(s), {drawn} sample(s) "
              f"drawn of {args.vote * len(votes)}; answer changed by the vote ({len(changed)}): {changed}")
    if args.repair_labels:
        used = [i for i, v in repairs.items() if v["used"]]
        print(f"label repairs: {len(repairs)} asked, {len(used)} used {used}"
              + "".join(f"; {i} still lacks {v['missing_after']}" for i, v in repairs.items()
                        if v.get("missing_after") and not v["used"]))
    if truncated:
        print(f"hit max_tokens: {truncated}")
    if errors:
        print(f"FAILED items ({len(errors)}): " + "; ".join(f"{k}: {v[:120]}" for k, v in errors.items()))
    print(f"tokens: prompt {ptok}, completion {ctok}; generation {gen_s:.1f}s summed, wall {wall:.1f}s")
    print(f"answers: {out}\nlog:     {log_path}")
    for w in warnings:
        print(f"warning: {w}")
    if problems:
        print(f"VALIDATION FAILED ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("validation: OK (the upload site will accept this file)")
    if args.dry_run:
        print("DRY RUN: placeholder answers - do not submit this file.")
    return 1 if (problems or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
