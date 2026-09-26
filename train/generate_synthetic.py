"""API regeneration path for synthetic items (the committed raw shards were produced by Claude generator agents).

Same shards (scripts/synth_task.build_shards) and the same prompt (train/synth_prompt.generation_prompt) as the
agent path; output data/synthetic/raw/<shard>.<backend>.jsonl feeds scripts/build_data.py. API shards have no blind-
solver verification file, so build them with `--skip-verify-glob "*.<backend>.jsonl"` (or verify them separately).

Backends (plain HTTP via requests):
  anthropic  POST https://api.anthropic.com/v1/messages  (ANTHROPIC_API_KEY; default model claude-opus-5 with
             server-side refusal fallbacks; sampling parameters are not sent: Opus 5 rejects them)
  openai     POST {base_url}/chat/completions, any OpenAI-compatible server, e.g. `vllm serve speakleash/Bielik-11B-v2.6-Instruct`
"""
from __future__ import annotations

import importlib.util
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

from eval.schema import SUBJECTS, SchemaError, validate_item, write_jsonl
from train.config import ROOT
from train.synth_prompt import generation_prompt

log = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MODELS = {"anthropic": "claude-opus-5", "openai": "speakleash/Bielik-11B-v2.6-Instruct"}
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

Complete = Callable[[str], str]


class GenerationError(RuntimeError):
    """Non-retryable failure for one prompt (bad request, refusal, empty output)."""


# ----------------------------------------------------------------------------- HTTP


def post_json(url: str, headers: dict, body: dict, *, retries: int = 5, timeout: float = 600.0, backoff: float = 2.0,
              sleep: Callable[[float], None] = time.sleep) -> dict:
    """POST with exponential backoff on connection errors and retryable statuses (honours Retry-After)."""
    last = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = f"{type(exc).__name__}: {exc}"
            delay = backoff * 2**attempt
        else:
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise GenerationError(f"non-JSON response: {resp.text[:300]}") from exc
            last = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code not in RETRYABLE_STATUS:
                raise GenerationError(last)
            retry_after = resp.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else backoff * 2**attempt
        if attempt < retries:
            log.warning("POST %s failed (%s); retry %d/%d in %.1fs", url, last, attempt + 1, retries, delay)
            sleep(delay + random.uniform(0, 0.5))
    raise RuntimeError(f"POST {url} failed after {retries + 1} attempts: {last}")


@dataclass
class BackendConfig:
    name: str  # anthropic | openai
    model: str
    api_key: str | None = None
    base_url: str = "http://localhost:8000/v1"
    max_tokens: int = 16000
    temperature: float = 0.7  # openai-compatible only
    fallbacks: bool = True  # anthropic: server-side refusal fallbacks
    retries: int = 5
    timeout: float = 600.0
    extra_headers: dict = field(default_factory=dict)


def anthropic_complete(cfg: BackendConfig, prompt: str, **kw) -> str:
    if not cfg.api_key:
        raise GenerationError("ANTHROPIC_API_KEY is not set")
    headers = {"x-api-key": cfg.api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json", **cfg.extra_headers}
    body: dict = {"model": cfg.model, "max_tokens": cfg.max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if cfg.fallbacks:
        headers["anthropic-beta"] = ANTHROPIC_FALLBACK_BETA
        body["fallbacks"] = "default"
    data = post_json(ANTHROPIC_URL, headers, body, retries=cfg.retries, timeout=cfg.timeout, **kw)
    if data.get("stop_reason") == "refusal":
        raise GenerationError(f"refusal: {data.get('stop_details')}")
    if data.get("stop_reason") == "max_tokens":
        log.warning("anthropic: output hit max_tokens=%d; keeping complete lines only", cfg.max_tokens)
    return "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")


def openai_complete(cfg: BackendConfig, prompt: str, **kw) -> str:
    headers = {"content-type": "application/json", **cfg.extra_headers}
    if cfg.api_key:
        headers["authorization"] = f"Bearer {cfg.api_key}"
    body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
            "messages": [{"role": "user", "content": prompt}]}
    data = post_json(f"{cfg.base_url.rstrip('/')}/chat/completions", headers, body, retries=cfg.retries, timeout=cfg.timeout, **kw)
    choices = data.get("choices") or []
    if not choices:
        raise GenerationError(f"no choices in response: {str(data)[:300]}")
    if choices[0].get("finish_reason") == "length":
        log.warning("openai: output hit max_tokens=%d; keeping complete lines only", cfg.max_tokens)
    return choices[0].get("message", {}).get("content") or ""


BACKENDS: dict[str, Callable[..., str]] = {"anthropic": anthropic_complete, "openai": openai_complete}


def make_complete(cfg: BackendConfig) -> Complete:
    if cfg.name not in BACKENDS:
        raise ValueError(f"unknown backend {cfg.name!r}; choose from {sorted(BACKENDS)}")
    return lambda prompt: BACKENDS[cfg.name](cfg, prompt)


# ----------------------------------------------------------------------------- parsing


_FENCE = re.compile(r"^\s*```")


def parse_jsonl(text: str) -> tuple[list[dict], list[str]]:
    """JSON objects from a JSONL response; tolerates code fences, blank/prose lines, trailing commas and a JSON array."""
    stripped = "\n".join(line for line in (text or "").splitlines() if not _FENCE.match(line)).strip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            return [d for d in data if isinstance(d, dict)], [f"non-object array element: {d!r}"[:200] for d in data if not isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    objs: list[dict] = []
    errors: list[str] = []
    for line in stripped.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON ({exc.msg}): {line[:160]}")
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    return objs, errors


def normalize_items(objs: list[dict], *, article: dict, id_prefix: str) -> tuple[list[dict], list[str], dict]:
    """Force subject/source from the article, give every item a unique '<id_prefix>-NN' id, validate.

    Returns (valid items, validation errors, counts of fixes applied to the valid items)."""
    items: list[dict] = []
    errors: list[str] = []
    fixes = {"id": 0, "subject": 0, "source": 0}
    claimed = {o.get("id") for o in objs}
    used: set[str] = set()
    id_re = re.compile(rf"^{re.escape(id_prefix)}-\d{{2,}}$")
    source = f"wikipedia:{article['title']}"
    next_num = 1
    for obj in objs:
        item = dict(obj)
        fixed = {"subject": "subject" in item and item["subject"] != article["subject"], "source": item.get("source") != source}
        item.update(subject=article["subject"], source=source)
        fixed["id"] = not (isinstance(item.get("id"), str) and id_re.match(item["id"]) and item["id"] not in used)
        if fixed["id"]:
            while f"{id_prefix}-{next_num:02d}" in used | claimed:
                next_num += 1
            item["id"] = f"{id_prefix}-{next_num:02d}"
        try:
            items.append(validate_item(item))
        except SchemaError as exc:
            errors.append(str(exc)[:300])
            continue
        used.add(item["id"])
        for key, was_fixed in fixed.items():
            fixes[key] += int(was_fixed)
    return items, errors, fixes


# ----------------------------------------------------------------------------- shards


def load_build_shards() -> Callable[..., dict[str, list[dict]]]:
    """scripts/ is not a package; load scripts/synth_task.py by path."""
    spec = importlib.util.spec_from_file_location("synth_task", ROOT / "scripts" / "synth_task.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/synth_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_shards


def select_shards(all_shards: dict[str, list[dict]], names: list[str] | None, max_articles: int | None) -> dict[str, list[dict]]:
    if names:
        unknown = [n for n in names if n not in all_shards and n not in SUBJECTS]
        if unknown:
            raise ValueError(f"unknown shards {unknown}; see python scripts/synth_task.py --list (a subject name selects all its shards)")
        chosen = {s: t for s, t in all_shards.items() if s in names or s.rsplit("-", 1)[0] in names}
    else:
        chosen = dict(all_shards)
    return {s: t[:max_articles] if max_articles else t for s, t in chosen.items()}


def generate_shard(shard: str, tasks: list[dict], complete: Complete, backend: str) -> tuple[list[dict], dict]:
    """One request per article; a failing article is logged and skipped."""
    items: list[dict] = []
    report: dict = {"articles": 0, "failed_articles": [], "items": 0, "expected_items": 0, "errors": [], "fixes": {"id": 0, "subject": 0, "source": 0}}
    for task in tasks:
        article, id_prefix = task["article"], f"{task['id_prefix']}-{backend}"
        report["articles"] += 1
        report["expected_items"] += task["n_items"]
        try:
            text = complete(generation_prompt(article, task["n_items"], id_prefix))
        except RuntimeError as exc:  # GenerationError or retries exhausted
            log.warning("%s / %s: generation failed: %s", shard, article["title"], exc)
            report["failed_articles"].append({"title": article["title"], "error": str(exc)[:300]})
            continue
        objs, parse_errors = parse_jsonl(text)
        new, errors, fixes = normalize_items(objs, article=article, id_prefix=id_prefix)
        items.extend(new)
        report["errors"].extend(f"{article['title']}: {e}" for e in parse_errors + errors)
        for k, v in fixes.items():
            report["fixes"][k] += v
    report["items"] = len(items)
    return items, report


def run(
    backend: BackendConfig,
    *,
    shards: list[str] | None = None,
    max_articles: int | None = None,
    out_dir: str | Path = ROOT / "data/synthetic/raw",
    articles_path: str | Path = ROOT / "data/wiki/articles.jsonl",
    force: bool = False,
    complete: Complete | None = None,
) -> dict[str, dict]:
    """Generate every selected shard not already on disk (unless force). Returns {shard: report}."""
    selected = select_shards(load_build_shards()(articles_path), shards, max_articles)
    complete = complete or make_complete(backend)
    out_dir = Path(out_dir)
    reports: dict[str, dict] = {}
    for shard, tasks in selected.items():
        out = out_dir / f"{shard}.{backend.name}.jsonl"
        if out.exists() and not force:
            reports[shard] = {"skipped": f"{out.name} exists (use --force)"}
            continue
        items, report = generate_shard(shard, tasks, complete, backend.name)
        if items:
            write_jsonl(items, out)
            report["out"] = str(out)
        reports[shard] = report
    return reports


def first_prompt(backend_name: str, *, shards: list[str] | None, articles_path: str | Path = ROOT / "data/wiki/articles.jsonl") -> str:
    selected = select_shards(load_build_shards()(articles_path), shards, 1)
    task = next(iter(selected.values()))[0]
    return generation_prompt(task["article"], task["n_items"], f"{task['id_prefix']}-{backend_name}")
