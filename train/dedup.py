"""Near-duplicate filtering of synthetic items against protected sets (eval, dev, external eval, exam blocklist).

Calibrated rule (docs/research/data_sources.md, Task C; BAAI/bge-m3, dense, no prefix, normalized):
  * QA text = stem (options and exam instructions removed) + "\\nOdpowiedź: " + gold answer TEXT;  Q text = stem alone.
    Instructions ("Dokończ zdanie. Wybierz właściwą odpowiedź spośród podanych.") are stripped on both sides: the shared
    boilerplate would otherwise make token_set_ratio match unrelated short stems (and dominate short embeddings).
  * candidate vs protected: drop if cos_QA >= thr_qa (only when the protected row has an answer)
    or cos_Q >= thr_q or rapidfuzz token_set_ratio(stem, stem) >= thr_fuzzy (both stems >= min_fuzzy_words content
    words and comparable length, see max_fuzzy).
  * then intra-synthetic near-duplicates: cos_QA >= thr_intra_qa -> keep the first occurrence.
Question-only similarity cannot separate paraphrases from same-topic questions; the gold answer text can.

Rows are either schema items (eval/schema.py) or blocklist rows {id, source, subject, question, options?, answer_text?}.
Similarities are computed in chunks (memory ~ chunk_size x n_protected floats).
"""
from __future__ import annotations

import gc
import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from eval.answer_format import format_number
from eval.schema import TYPES
from train.config import resolve_path


EmbedFn = Callable[[list[str]], np.ndarray]
METHODS = ("qa", "q", "fuzzy")
INTRA = "intra_dup"


@dataclass
class DedupConfig:
    model: str = "BAAI/bge-m3"
    thr_qa: float = 0.80
    thr_q: float = 0.90
    thr_fuzzy: float = 92
    thr_intra_qa: float = 0.95
    batch_size: int = 64
    min_fuzzy_words: int = 6      # content words (>= 2 chars incl. a letter) required on both stems
    fuzzy_min_len_ratio: float = 0.5  # shorter/longer token count; see max_fuzzy
    max_seq_length: int = 512  # bge-m3 accepts 8192; long exam passages would blow up attention memory
    chunk_size: int = 1024
    n_examples: int = 20
    cache_dir: str | None = "data/synthetic/.emb_cache"  # protected-set embeddings (default embedder only)


def reason_for(set_name: str) -> str:
    return f"dup_vs_{set_name}"


# ----------------------------------------------------------------------------- text building

_WS = re.compile(r"\s+")
_OPT_A = re.compile(r"(?:(?<=\s)|^)\(?A[.)](?=\s)")
_OPT_B = re.compile(r"(?<=\s)\(?B[.)](?=\s)")
# CKE / generator boilerplate anywhere in a stem (GENERATION_RULES tells the generator to write the first two).
_INSTRUCTION = re.compile(
    r"Dokończ\s+zdani[ea]\s*[.:]?"
    r"|(?:Wybierz|Zaznacz|Wskaż|Podkreśl)\s+(?:(?:właściwą|poprawną|prawidłową)\s+odpowiedź"
    r"|(?:dwie|trzy)\s+(?:(?:właściwe|poprawne|prawidłowe)\s+)?odpowiedzi)(?:\s+spośród\s+podanych)?(?:\s*\([^)]*\))?\s*[.:]?"
    r"|Oceń\s+prawdziwość\s+(?:podanych\s+)?(?:stwierdzeń|informacji|zdań)[^.]*\.(?:\s*Zaznacz\s+P[^.]*\.)?",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def strip_inline_options(question: str) -> str:
    """Cut an inline option list ("... A. x B. y" / "A) x" / "(A) x")."""
    stem = question or ""
    for m in _OPT_A.finditer(stem):
        if _OPT_B.search(stem, m.end()):
            stem = stem[: m.start()]
            break
    stem = stem.strip().rstrip(":").strip()
    return stem or (question or "").strip()


def strip_instructions(text: str) -> str:
    """Remove exam instructions anywhere in the text (the original text when nothing else would remain)."""
    stripped = _clean(_INSTRUCTION.sub(" ", text))
    return stripped or _clean(text)


def is_schema_item(row: dict) -> bool:
    return row.get("type") in TYPES and "answer" in row


def stem_text(row: dict) -> str:
    """Schema items: question (+ statements / left elements; no context, no options). Blocklist rows: question
    with any inline option list stripped. Exam instructions are removed from both."""
    if not is_schema_item(row):
        return strip_instructions(_clean(strip_inline_options(row.get("question", ""))))
    parts = [row["question"]]
    if row["type"] == "tf":
        parts += row.get("statements") or []
    elif row["type"] == "match":
        parts += row.get("left") or []
    return strip_instructions(_clean("\n".join(parts)))


def answer_text(row: dict) -> str | None:
    """Gold answer as TEXT (never a bare letter), or None when the row has no answer."""
    if not is_schema_item(row):
        text = row.get("answer_text")
        return _clean(str(text)) if text not in (None, "") else None
    t, a = row["type"], row["answer"]
    if t == "mc":
        return _clean(row["options"][a])
    if t == "multi":
        return _clean("; ".join(row["options"][x] for x in a))
    if t == "tf":
        return _clean("; ".join(f"{s} ({v})" for s, v in zip(row["statements"], a)))
    if t == "match":
        left = row["left"]
        return _clean("; ".join(f"{left[int(k) - 1]} -> {row['right'][v]}" for k, v in sorted(a.items(), key=lambda kv: int(kv[0]))))
    if t == "short":
        return _clean(a[0] if isinstance(a, list) else a)
    return format_number(a)  # numeric


def qa_text(stem: str, answer: str) -> str:
    return f"{stem}\nOdpowiedź: {answer}"


_NON_WORD = re.compile(r"[^\w]+")


def fuzzy_norm(text: str) -> str:
    return _WS.sub(" ", _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold())).strip()


_CONTENT_WORD = re.compile(r"\w*[^\W\d_]\w*")


def content_words(normed: str) -> int:
    """Words with >= 2 characters and at least one letter (math symbols / single letters do not count)."""
    return sum(1 for w in normed.split() if len(w) >= 2 and _CONTENT_WORD.fullmatch(w))


# ----------------------------------------------------------------------------- embeddings


class SentenceTransformerEmbedder:
    """Loads the model on first use; call close() to release it (the laptop is shared)."""

    def __init__(self, model: str, *, batch_size: int = 64, max_seq_length: int = 512, device: str | None = None):
        self.model_name, self.batch_size, self.max_seq_length, self.device = model, batch_size, max_seq_length, device
        self._model = None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from eval.modeling import detect_device

            self._model = SentenceTransformer(self.model_name, device=self.device or detect_device())
            self._model.max_seq_length = self.max_seq_length
        emb = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def _embed(texts: list[str], embed_fn: EmbedFn, *, cache_dir: Path | None, model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    path = None
    if cache_dir is not None:
        digest = hashlib.sha256("\x1e".join([model, *texts]).encode("utf-8")).hexdigest()[:24]
        path = cache_dir / f"{digest}.npy"
        if path.exists():
            return np.load(path)
    emb = _normalize_rows(embed_fn(texts))
    if emb.shape[0] != len(texts):
        raise ValueError(f"embed_fn returned {emb.shape[0]} vectors for {len(texts)} texts")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    return emb


def max_similarity(a: np.ndarray, b: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise max cosine (inputs L2-normalized) and argmax, computed in chunks. Empty b -> (-1, -1)."""
    best = np.full(len(a), -1.0, dtype=np.float32)
    idx = np.full(len(a), -1, dtype=np.int64)
    if len(a) == 0 or len(b) == 0:
        return best, idx
    for start in range(0, len(a), chunk_size):
        sim = a[start : start + chunk_size] @ b.T
        arg = sim.argmax(axis=1)
        idx[start : start + len(sim)] = arg
        best[start : start + len(sim)] = sim[np.arange(len(sim)), arg]
    return best, idx


def max_fuzzy(
    cand: list[str], prot: list[str], *, min_words: int, min_len_ratio: float, cutoff: float, chunk_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise best token_set_ratio over fuzzy-normalized stems (scores below cutoff count as 0).

    Eligible pairs: both stems have >= min_words content words, and token counts within min_len_ratio of each other.
    The ratio guard matters: token_set_ratio is 100 whenever one token set contains the other, so without it a short
    synthetic stem "matches" any long exam passage that happens to contain its words (measured on the 80 authored
    eval items vs the blocklist: 5 false hits -> 1, while cross-source copies of real exam items drop 169 -> 165).
    """
    from rapidfuzz import fuzz, process

    best = np.zeros(len(cand), dtype=np.float32)
    idx = np.full(len(cand), -1, dtype=np.int64)
    ci = [i for i, s in enumerate(cand) if content_words(s) >= min_words]
    pi = [j for j, s in enumerate(prot) if content_words(s) >= min_words]
    if not ci or not pi:
        return best, idx
    choices = [prot[j] for j in pi]
    p_len = np.array([len(prot[j].split()) for j in pi], dtype=np.float32)
    for start in range(0, len(ci), chunk_size):
        rows = ci[start : start + chunk_size]
        scores = process.cdist([cand[i] for i in rows], choices, scorer=fuzz.token_set_ratio,
                               score_cutoff=cutoff, dtype=np.float32, workers=-1)
        c_len = np.array([len(cand[i].split()) for i in rows], dtype=np.float32)[:, None]
        scores *= (np.minimum(c_len, p_len) / np.maximum(c_len, p_len)) >= min_len_ratio
        arg = scores.argmax(axis=1)
        for r, i in enumerate(rows):
            if scores[r, arg[r]] > 0:
                best[i], idx[i] = scores[r, arg[r]], pi[arg[r]]
    return best, idx


# ----------------------------------------------------------------------------- dedup


def _example(cand: dict, texts: dict, match: dict | None, scores: dict, methods: list[str]) -> dict:
    ex = {"id": cand.get("id"), "stem": texts["stem"], "answer": texts["answer"], "methods": methods,
          "scores": {k: round(float(v), 4) for k, v in scores.items()}}
    if match is not None:
        # id + source only: protected rows can be third-party exam text, and dedup_report.json is committed
        ex.update(matched_id=match.get("id"), matched_source=match.get("source"))
    return ex


def dedup_items(
    candidates: list[dict],
    protected: dict[str, list[dict]],
    cfg: DedupConfig | None = None,
    embed_fn: EmbedFn | None = None,
) -> tuple[list[dict], dict]:
    """Drop candidates that near-duplicate a protected row, then intra-candidate near-duplicates (keep first).

    protected: {"heldout": [...], "dev": [...], "ext_llmzszl_matura": [...], "blocklist": [...]}; a candidate is
    attributed to the FIRST set (dict order) whose rule fires. Returns (kept, report).
    """
    cfg = cfg or DedupConfig()
    own_embedder = embed_fn is None
    embedder: EmbedFn = SentenceTransformerEmbedder(cfg.model, batch_size=cfg.batch_size, max_seq_length=cfg.max_seq_length) if own_embedder else embed_fn
    cache_dir = resolve_path(cfg.cache_dir) if (own_embedder and cfg.cache_dir) else None
    try:
        return _dedup(candidates, protected, cfg, embedder, cache_dir)
    finally:
        if own_embedder:
            embedder.close()  # type: ignore[attr-defined]


def _dedup(candidates: list[dict], protected: dict[str, list[dict]], cfg: DedupConfig, embed_fn: EmbedFn, cache_dir: Path | None) -> tuple[list[dict], dict]:
    c_stem = [stem_text(c) for c in candidates]
    c_ans = [answer_text(c) or "" for c in candidates]
    c_q = _embed(c_stem, embed_fn, cache_dir=None, model=cfg.model)
    c_qa = _embed([qa_text(s, a) for s, a in zip(c_stem, c_ans)], embed_fn, cache_dir=None, model=cfg.model)
    c_fuzzy = [fuzzy_norm(s) for s in c_stem]

    n = len(candidates)  # 0 candidates: protected sets are not embedded (no model load)
    reason: list[str | None] = [None] * n
    report_examples: dict[str, list[dict]] = {}
    methods_count: dict[str, dict[str, int]] = {}
    dropped_ids: dict[str, list[str]] = {}

    cache_key = f"{cfg.model}|max_seq_length={cfg.max_seq_length}"
    for set_name, rows in protected.items():
        key = reason_for(set_name)
        methods_count[key] = dict.fromkeys(METHODS, 0)
        dropped_ids[key], report_examples[key] = [], []
        if not rows or not n:
            continue
        p_stem = [stem_text(r) for r in rows]
        p_ans = [answer_text(r) for r in rows]
        with_ans = [j for j, a in enumerate(p_ans) if a]
        p_q = _embed(p_stem, embed_fn, cache_dir=cache_dir, model=cache_key)
        p_qa = _embed([qa_text(p_stem[j], p_ans[j]) for j in with_ans], embed_fn, cache_dir=cache_dir, model=cache_key)
        best_q, idx_q = max_similarity(c_q, p_q, cfg.chunk_size)
        best_qa, idx_qa_local = max_similarity(c_qa, p_qa, cfg.chunk_size)
        idx_qa = np.array([with_ans[k] if k >= 0 else -1 for k in idx_qa_local], dtype=np.int64)
        best_fz, idx_fz = max_fuzzy(c_fuzzy, [fuzzy_norm(s) for s in p_stem], min_words=cfg.min_fuzzy_words,
                                    min_len_ratio=cfg.fuzzy_min_len_ratio, cutoff=cfg.thr_fuzzy, chunk_size=cfg.chunk_size)
        for i in range(n):
            if reason[i] is not None:
                continue
            fired = {"qa": bool(best_qa[i] >= cfg.thr_qa), "q": bool(best_q[i] >= cfg.thr_q), "fuzzy": bool(best_fz[i] >= cfg.thr_fuzzy)}
            methods = [m for m in METHODS if fired[m]]
            if not methods:
                continue
            reason[i] = key
            dropped_ids[key].append(candidates[i].get("id"))
            for m in methods:
                methods_count[key][m] += 1
            if len(report_examples[key]) < cfg.n_examples:
                match_idx = {"qa": idx_qa[i], "q": idx_q[i], "fuzzy": idx_fz[i]}[methods[0]]
                scores = {"qa": best_qa[i], "q": best_q[i], "fuzzy": best_fz[i]}
                report_examples[key].append(_example(candidates[i], {"stem": c_stem[i], "answer": c_ans[i]},
                                                     rows[int(match_idx)], scores, methods))

    survivors = [i for i in range(n) if reason[i] is None]
    dropped_ids[INTRA], report_examples[INTRA] = [], []
    for i, j, score in _intra_duplicates(c_qa, survivors, cfg):
        reason[i] = INTRA
        dropped_ids[INTRA].append(candidates[i].get("id"))
        if len(report_examples[INTRA]) < cfg.n_examples:
            report_examples[INTRA].append(_example(candidates[i], {"stem": c_stem[i], "answer": c_ans[i]},
                                                   candidates[j], {"qa": score}, ["qa"]))

    kept = [c for c, r in zip(candidates, reason) if r is None]
    report = {
        "model": cfg.model,
        "thresholds": {k: v for k, v in asdict(cfg).items() if k.startswith("thr_") or k in ("min_fuzzy_words", "fuzzy_min_len_ratio")},
        "n_candidates": n,
        "n_kept": len(kept),
        "protected_sizes": {k: len(v) for k, v in protected.items()},
        "counts": {k: len(v) for k, v in dropped_ids.items()},
        "methods": methods_count,
        "examples": report_examples,
        "dropped_ids": dropped_ids,
    }
    return kept, report


def _intra_duplicates(qa: np.ndarray, order: list[int], cfg: DedupConfig) -> list[tuple[int, int, float]]:
    """Greedy keep-first among `order`: returns (dropped_idx, kept_idx_it_duplicates, score)."""
    if len(order) < 2:
        return []
    emb = qa[order]
    earlier: list[list[tuple[int, float]]] = [[] for _ in order]
    for start in range(0, len(order), cfg.chunk_size):
        sim = emb[start : start + cfg.chunk_size] @ emb.T
        for r in range(len(sim)):
            pos = start + r
            hits = np.nonzero(sim[r, :pos] >= cfg.thr_intra_qa)[0]
            earlier[pos] = [(int(h), float(sim[r, h])) for h in hits]
    kept = [True] * len(order)
    out = []
    for pos, hits in enumerate(earlier):
        match = next(((h, s) for h, s in hits if kept[h]), None)
        if match is not None:
            kept[pos] = False
            out.append((order[pos], order[match[0]], match[1]))
    return out
