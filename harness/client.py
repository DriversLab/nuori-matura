"""OpenAI-compatible chat client for llama.cpp llama-server and QVAC ("qvac serve --openai"), requests only.

Every request carries the full decoding config explicitly, because QVAC ignores logprobs/stop/n>1 and otherwise
defaults to temperature 0.8 / repeat_penalty 1.1:
  temperature 0, top_p 1, seed 42, max_tokens, and (llama_extras=True) top_k 1, repeat_penalty 1.0, cache_prompt
  (False by default: reusing a cached prompt prefix changes the arithmetic, so the same prompt could get a different
  greedy answer depending on what the server served before; measured on the mock: 22 of 37 harness answers differed
  between a fresh and a used server. False makes a run reproducible at the cost of re-reading each prompt).
lora_scale=None sends no "lora" field (server default). lora_scale=s sends "lora": [{"id": 0, "scale": s}], so one
llama-server started with `--lora adapter.gguf` serves base (0.0) and tuned (1.0) from the same weights.

The server must be able to hold prompt + max_tokens: organizer protocol is an 8192-token context per request, so
start llama-server with `-c 8192` per parallel slot (`-np N -c N*8192`). Only base_url is ever contacted.
"""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, TypeVar

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_SEED = 42
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

T = TypeVar("T")
R = TypeVar("R")


class ChatError(RuntimeError):
    """Request failed for good (non-retryable status, or retries exhausted)."""


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 1
    model: str | None = None
    timings: dict | None = None
    reasoning: str | None = None  # "reasoning_content" if the server split it off (not part of the answer)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + ("/models" if base.endswith("/v1") else "/v1/models")


def list_models(base_url: str, timeout: float = 10.0) -> list[str]:
    r = requests.get(models_url(base_url), timeout=timeout)
    r.raise_for_status()
    return [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]


class ChatClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str | None = None, *, timeout: float = 1800.0,
                 retries: int = 3, backoff_s: float = 2.0, lora_scale: float | None = None, seed: int = DEFAULT_SEED,
                 llama_extras: bool = True, extra_params: dict | None = None, cache_prompt: bool = False):
        self.base_url = base_url
        self.url = chat_url(base_url)
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff_s = backoff_s
        self.lora_scale = lora_scale
        self.seed = seed
        self.llama_extras = llama_extras
        self.cache_prompt = cache_prompt
        self.extra_params = dict(extra_params or {})
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def payload(self, messages: list[dict], max_tokens: int, **overrides: Any) -> dict:
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.seed,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if self.model:
            body["model"] = self.model
        if self.llama_extras:
            body.update({"top_k": 1, "repeat_penalty": 1.0, "cache_prompt": bool(self.cache_prompt)})
        if self.lora_scale is not None:
            body["lora"] = [{"id": 0, "scale": float(self.lora_scale)}]
        body.update(self.extra_params)
        body.update(overrides)
        return body

    def chat(self, messages: list[dict], max_tokens: int, **overrides: Any) -> ChatResult:
        body = self.payload(messages, max_tokens, **overrides)
        last_err = "no attempt made"
        t0 = time.monotonic()
        for attempt in range(1, self.retries + 2):
            try:
                resp = self._session().post(self.url, json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        choice = data["choices"][0]
                        msg = choice.get("message") or {}
                        text = msg.get("content")
                        if text is None:
                            text = choice.get("text") or ""
                        return ChatResult(
                            text=text,
                            finish_reason=choice.get("finish_reason"),
                            usage=data.get("usage") or {},
                            latency_s=time.monotonic() - t0,
                            attempts=attempt,
                            model=data.get("model"),
                            timings=data.get("timings"),
                            reasoning=msg.get("reasoning_content"),
                        )
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        last_err = f"malformed response ({e}): {resp.text[:300]}"
                elif resp.status_code in RETRY_STATUS:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                else:
                    raise ChatError(f"HTTP {resp.status_code} from {self.url}: {resp.text[:500]}")
            if attempt <= self.retries:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)) * (1 + 0.1 * random.random()))
        raise ChatError(f"giving up after {self.retries + 1} attempts: {last_err}")


def run_parallel(fn: Callable[[T], R], items: Iterable[T], parallel: int = 4) -> Iterator[tuple[T, R | None, BaseException | None]]:
    """Yield (item, result, error) as calls finish; one failing call never stops the others."""
    items = list(items)
    if parallel <= 1:
        for it in items:
            try:
                yield it, fn(it), None
            except Exception as e:  # noqa: BLE001
                yield it, None, e
        return
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                yield it, fut.result(), None
            except Exception as e:  # noqa: BLE001
                yield it, None, e
