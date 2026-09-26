"""Local RAG knowledge base for the history matura: BM25 over Polish Wikipedia passages (pure Python + numpy, no model).

Contract (used by train/history_data.py make_rag_context and scripts/run_exam.py --style rag):

    index = load_index("data/rag/index")
    hits  = index.search(build_query(item), k=4)      # [{"title", "section", "text", "url", "score"}, ...]
    text  = format_passages(hits, max_chars=2400)     # numbered passages, each with its article title

Build the index with scripts/build_rag.py (titles in data/rag/titles_history.yaml -> data/rag/articles.jsonl ->
data/rag/index/). Everything under data/rag/ except the title list is gitignored.

Text normalisation (the same for passages and queries): NFC, lowercase, every non-alphanumeric character is a
separator, a small Polish stopword list (function words and exam-instruction verbs), then a light stemmer: strip the
longest common inflectional ending (keeping >= 3 letters) and cut to a 6-letter prefix, so "powstanie / powstania /
powstaniu" and "Kazimierz / Kazimierza" meet. Numbers (years) are kept verbatim.

Scoring: BM25 (k1 1.2, b 0.75) over passage terms (article title x2 + section + text), multiplied by
1 + TITLE_BOOST * (share of the article's title named by the query). Adjacent-term bigrams were tried and dropped:
no gain on a 54-query check set and ~12x the RAM.

Index layout (data/rag/index/):
    meta.json        params (k1, b), counts, avgdl, article list [{title, url, epoch}]
    vocab.json       term list; a term's position is its id
    postings.npz     CSR postings: indptr (int64, n_terms + 1), docs (int32), tfs (uint16); doc_len (float32)
    passages.jsonl   one passage per line {"a": article index, "s": section, "t": text}
    offsets.npy      byte offset of every passages.jsonl line (texts are read from disk only for the hits)
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FORMAT_VERSION = 3
K1 = 1.2
B = 0.75
TITLE_WEIGHT = 2          # the article title is indexed this many times in each of its passages (+ the section once)
PREFIX = 6
TITLE_IDF_FULL = 8.0     # title idf mass at which the title boost applies in full (lower = scaled down)
TITLE_BOOST = 0.0        # passage score x (1 + TITLE_BOOST * share of the article title named by the query)

# ---- normaliser ---------------------------------------------------------------------------------------------------

STOPWORDS = frozenset("""
a aby ach albo ale ani aż bardzo bez bo by być był była było były będzie będą co czy czyli dla do gdy gdzie go i ich
ile im in inne ja jak jaki jaka jakie jako je jego jej jest jeszcze jeśli już ją każdy kiedy kto która które którego
której który których którym którzy lub ma mają mi mnie może można na nad nam nas nie niej niż nim o od oraz oto po pod
przez przy się sobie są ta tak także tam te tego tej ten też to tu tych tylko tym u w we więc z za ze że żeby jednak
oraz według wobec wśród między około poza ponad np tzw itp itd r w. wiek wieku wiekach roku rok lata latach lat
zadanie zadania pkt punkt punkty podaj podał wyjaśnij wyjaśnienie oceń rozstrzygnij rozstrzygnięcie uzasadnij
uzasadnienie uzasadniając wymień przedstaw określ wskaż wpisz zaznacz dokończ uzupełnij odpowiedź odpowiedzi
źródło źródła źródłem źródle tekst tekstu tekście fragment fragmentu fragmencie informacja informacje informacji
poniższy poniższe poniższego podanych podany podane przedstawiony przedstawione przytoczony opis opisu
prawdziwe fałszywe prawdziwa fałszywa zdanie zdania
""".split())

# longest first; a suffix is stripped only when >= 3 characters of the word remain
_SUFFIXES = tuple(sorted("""
owiami ościami ościach owych owego owemu owymi owej iego iemu iach iami ymi ich ych iej ego emu ami ach owi owa owe
ów om em ie ia ii iu ią ię ę ą a e i o u y
""".split(), key=len, reverse=True))
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
# genitive month names stay unstemmed: "stycznia" (a date) must not meet "styczniowe" (powstanie styczniowe), and
# "3 maja" keeps its own term
MONTHS = frozenset("stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada "
                   "grudnia".split())


def stem(word: str) -> str:
    if word.isdigit() or word in MONTHS:
        return word
    if len(word) > 3:
        for suf in _SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                word = word[: -len(suf)]
                break
    return word[:PREFIX]


def normalize(text: str) -> list[str]:
    """Text -> stemmed index terms (stopwords and 1-letter tokens dropped; numbers kept)."""
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        out.append(stem(tok))
    return out


# ---- chunking -----------------------------------------------------------------------------------------------------

_SECTION = re.compile(r"(?m)^##\s+(.+?)\s*$")
_SENT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„\"(])")


def _words(s: str) -> int:
    return len(s.split())


def _split_long(par: str, max_words: int) -> list[str]:
    """A paragraph longer than max_words -> sentence groups of <= max_words (a single huge sentence is cut by words)."""
    if _words(par) <= max_words:
        return [par]
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for sent in _SENT.split(par):
        w = _words(sent)
        if w > max_words:
            if cur:
                out.append(" ".join(cur))
                cur, n = [], 0
            toks = sent.split()
            out += [" ".join(toks[i : i + max_words]) for i in range(0, len(toks), max_words)]
            continue
        if n + w > max_words and cur:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_article(text: str, min_words: int = 180, max_words: int = 260) -> list[dict]:
    """Cleaned article text (train.wiki.clean_extract, "## Heading" lines) -> [{"section", "text"}] passages of about
    min_words..max_words words. Sentences are packed greedily within a section (paragraph breaks kept) and a passage
    closes once it reaches the midpoint of the range; a short section tail joins the previous passage when that stays
    <= max_words * 1.25. Then adjacent passages shorter than min_words are merged across sections while the result
    stays <= max_words (section labels joined with " / "). The lead section has section ""."""
    sections: list[tuple[str, str]] = []
    pos, name = 0, ""
    for m in _SECTION.finditer(text or ""):
        sections.append((name, text[pos : m.start()]))
        name, pos = m.group(1).strip(), m.end()
    sections.append((name, (text or "")[pos:]))
    target = (min_words + max_words) // 2
    limit = int(max_words * 1.25)

    out: list[dict] = []
    for sec, body in sections:
        units: list[tuple[str, bool]] = []          # (sentence group, ends a paragraph)
        for par in re.split(r"\n", body):
            par = " ".join(par.split())
            if not par:
                continue
            sents = _split_long(par, max_words // 4) if _words(par) > max_words // 4 else [par]
            units += [(s, i == len(sents) - 1) for i, s in enumerate(sents)]
        chunks: list[str] = []
        cur = ""
        n = 0
        cur_par_end = True
        for s, par_end in units:
            w = _words(s)
            if cur and n + w > max_words:
                chunks.append(cur)
                cur, n = "", 0
            cur = s if not cur else cur + ("\n" if cur_par_end else " ") + s
            cur_par_end = par_end
            n += w
            if n >= target:
                chunks.append(cur)
                cur, n = "", 0
        if cur:
            if chunks and n < min_words // 2 and _words(chunks[-1]) + n <= limit:
                chunks[-1] = chunks[-1] + "\n" + cur
            else:
                chunks.append(cur)
        out += [{"section": sec, "text": c} for c in chunks]

    merged: list[dict] = []
    for c in out:
        if merged:
            prev = merged[-1]
            pw, cw = _words(prev["text"]), _words(c["text"])
            if ((pw < min_words or cw < min_words // 2) and pw + cw <= max_words) or (min(pw, cw) < 40 and pw + cw <= limit):
                secs = [s for s in dict.fromkeys(prev["section"].split(" / ") + [c["section"]]) if s]
                merged[-1] = {"section": " / ".join(secs[:3]), "text": prev["text"] + "\n" + c["text"]}
                continue
        merged.append(dict(c))
    return merged


# ---- index --------------------------------------------------------------------------------------------------------

class BM25Index:
    """Sparse BM25 index. Build with BM25Index.build(passages) or load_index(path); query with search()."""

    def __init__(self, vocab: list[str], indptr: np.ndarray, docs: np.ndarray, tfs: np.ndarray, doc_len: np.ndarray,
                 articles: list[dict], passage_article: np.ndarray, passage_section: list[str],
                 texts: list[str] | None = None, path: Path | None = None, offsets: np.ndarray | None = None,
                 k1: float = K1, b: float = B):
        self.vocab = vocab
        self.term_id = {t: i for i, t in enumerate(vocab)}
        self.indptr, self.docs, self.tfs = indptr, docs, tfs
        self.doc_len = doc_len.astype(np.float32)
        self.articles = articles
        self.passage_article = passage_article
        self.passage_section = passage_section
        self._texts = texts
        self._path = path
        self._offsets = offsets
        self.k1, self.b = k1, b
        self.n = len(doc_len)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df = np.diff(indptr).astype(np.float64)
        self.idf = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._norm = (k1 * (1.0 - b + b * self.doc_len / max(self.avgdl, 1e-9))).astype(np.float32)
        self._tt: tuple[list[list[int]], dict[int, list[int]]] | None = None   # title terms, built lazily

    # -- build / save ---------------------------------------------------------------------------------------------

    @classmethod
    def build(cls, passages: Iterable[Mapping[str, Any]], k1: float = K1, b: float = B) -> "BM25Index":
        """passages: [{"title", "section", "text", "url"[, "epoch"]}] (one article = consecutive passages or not)."""
        art_index: dict[str, int] = {}
        articles: list[dict] = []
        p_art: list[int] = []
        p_sec: list[str] = []
        texts: list[str] = []
        vocab: dict[str, int] = {}
        rows: list[list[int]] = []   # per term: flat [doc, tf, doc, tf, ...]
        doc_len: list[int] = []
        for d, p in enumerate(passages):
            title = str(p.get("title") or "")
            if title not in art_index:
                art_index[title] = len(articles)
                articles.append({"title": title, "url": p.get("url"), **({"epoch": p["epoch"]} if p.get("epoch") else {})})
            p_art.append(art_index[title])
            sec = str(p.get("section") or "")
            p_sec.append(sec)
            text = str(p.get("text") or "")
            texts.append(text)
            toks = normalize(title) * TITLE_WEIGHT + normalize(sec) + normalize(text)
            doc_len.append(len(toks))
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                tid = vocab.get(t)
                if tid is None:
                    tid = vocab[t] = len(rows)
                    rows.append([])
                rows[tid] += (d, min(c, 65535))
        term_list = sorted(vocab, key=vocab.get)
        indptr = np.zeros(len(term_list) + 1, dtype=np.int64)
        for i, r in enumerate(rows):
            indptr[i + 1] = indptr[i] + len(r) // 2
        flat = np.fromiter((x for r in rows for x in r), dtype=np.int64, count=int(indptr[-1]) * 2)
        docs = flat[0::2].astype(np.int32)
        tfs = flat[1::2].astype(np.uint16)
        return cls(term_list, indptr, docs, tfs, np.asarray(doc_len, dtype=np.float32), articles,
                   np.asarray(p_art, dtype=np.int32), p_sec, texts=texts, k1=k1, b=b)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        offsets = np.zeros(self.n, dtype=np.int64)
        with open(path / "passages.jsonl.tmp", "wb") as f:
            for i in range(self.n):
                offsets[i] = f.tell()
                row = {"a": int(self.passage_article[i]), "s": self.passage_section[i], "t": self.text(i)}
                f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        np.save(path / "offsets.npy", offsets)
        np.savez(path / "postings.npz", indptr=self.indptr, docs=self.docs, tfs=self.tfs, doc_len=self.doc_len)
        (path / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        meta = {"format": FORMAT_VERSION, "k1": self.k1, "b": self.b, "prefix": PREFIX, "title_weight": TITLE_WEIGHT,
                "n_passages": self.n, "n_articles": len(self.articles), "n_terms": len(self.vocab),
                "n_postings": int(self.indptr[-1]), "avgdl": self.avgdl, "articles": self.articles}
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        (path / "passages.jsonl.tmp").replace(path / "passages.jsonl")
        self._path, self._offsets = path, offsets
        return path

    # -- query ----------------------------------------------------------------------------------------------------

    def text(self, i: int) -> str:
        if self._texts is not None:
            return self._texts[i]
        with open(self._path / "passages.jsonl", "rb") as f:
            f.seek(int(self._offsets[i]))
            return json.loads(f.readline())["t"]

    def _title_terms(self) -> tuple[list[list[int]], dict[int, list[int]]]:
        """Per article: unigram term ids of its title (a trailing "(...)" qualifier dropped); inverted term -> articles."""
        if self._tt is None:
            per: list[list[int]] = []
            inv: dict[int, list[int]] = {}
            for a, art in enumerate(self.articles):
                ids = sorted({self.term_id[t] for t in normalize(_QUALIFIER.sub("", art["title"])) if t in self.term_id})
                per.append(ids)
                for tid in ids:
                    inv.setdefault(tid, []).append(a)
            self._tt = (per, inv)
        return self._tt

    def title_coverage(self, query_ids: Iterable[int]) -> np.ndarray:
        """Per article: idf-weighted share of its title's terms that occur in the query (1.0 = the query names it)."""
        per, inv = self._title_terms()
        cov = np.zeros(len(self.articles), dtype=np.float32)
        q = set(query_ids)
        hit_articles = {a for tid in q for a in inv.get(tid, ())}
        for a in hit_articles:
            tot = float(self.idf[per[a]].sum())
            if tot > 0:
                # a generic one-word title ("Rzym", "Szlachta") names its article less surely than "Kodeks Hammurabiego"
                cov[a] = float(self.idf[[t for t in per[a] if t in q]].sum()) / tot * min(1.0, tot / TITLE_IDF_FULL)
        return cov

    def scores(self, query: str, min_idf: float = 0.2, title_boost: float | None = None) -> np.ndarray:
        """BM25 score of every passage. Query terms are weighted 1 + ln(count); near-ubiquitous terms are skipped. The
        sum is multiplied by 1 + title_boost * coverage, where coverage is the idf-weighted share of the passage's
        article title named by the query (scaled down for generic one-word titles, see title_coverage)."""
        title_boost = TITLE_BOOST if title_boost is None else title_boost
        counts: dict[int, int] = {}
        for t in normalize(query):
            tid = self.term_id.get(t)
            if tid is not None:
                counts[tid] = counts.get(tid, 0) + 1
        out = np.zeros(self.n, dtype=np.float32)
        k1p1 = self.k1 + 1.0
        for tid, c in counts.items():
            idf = self.idf[tid]
            if idf < min_idf:
                continue
            s, e = self.indptr[tid], self.indptr[tid + 1]
            d = self.docs[s:e]
            tf = self.tfs[s:e].astype(np.float32)
            out[d] += (idf * (1.0 + math.log(c))) * tf * k1p1 / (tf + self._norm[d])
        if title_boost > 0 and counts:
            cov = self.title_coverage(counts)
            out *= 1.0 + title_boost * cov[self.passage_article]
        return out

    def search(self, query: str, k: int = 5, max_per_title: int = 2, **score_kw) -> list[dict]:
        """Top-k passages [{"title", "section", "text", "url", "score"}], at most max_per_title per article, no
        duplicate texts, only passages with a positive score."""
        if not self.n or k <= 0:
            return []
        sc = self.scores(query, **score_kw)
        pool = min(self.n, max(k * 10, 50))
        cand = np.argpartition(-sc, pool - 1)[:pool] if pool < self.n else np.arange(self.n)
        cand = cand[np.argsort(-sc[cand], kind="stable")]
        hits: list[dict] = []
        per_title: dict[int, int] = {}
        seen: set[str] = set()
        for i in cand:
            if sc[i] <= 0 or len(hits) >= k:
                break
            a = int(self.passage_article[i])
            if per_title.get(a, 0) >= max_per_title:
                continue
            text = self.text(int(i))
            if text in seen:
                continue
            seen.add(text)
            per_title[a] = per_title.get(a, 0) + 1
            art = self.articles[a]
            hits.append({"title": art["title"], "section": self.passage_section[i], "text": text,
                         "url": art.get("url"), "score": float(sc[i])})
        return hits

    def memory_bytes(self) -> int:
        """Rough resident size of a loaded index (arrays + vocab dict + section strings; texts stay on disk)."""
        arrays = sum(a.nbytes for a in (self.indptr, self.docs, self.tfs, self.doc_len, self.idf, self._norm,
                                         self.passage_article))
        vocab = sum(len(t.encode("utf-8")) + 49 + 8 + 100 for t in self.vocab)   # str + list slot + dict entry
        sections = sum(len(s) + 49 + 8 for s in self.passage_section)
        texts = sum(len(t.encode("utf-8")) + 49 for t in self._texts) if self._texts is not None else 0
        return int(arrays + vocab + sections + texts + (self._offsets.nbytes if self._offsets is not None else 0))


def build_index(passages: Iterable[Mapping[str, Any]], k1: float = K1, b: float = B) -> BM25Index:
    return BM25Index.build(passages, k1=k1, b=b)


def load_index(path: str | Path) -> BM25Index:
    """Load an index saved by BM25Index.save (the directory, or any file inside it)."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(f"RAG index {path} has format {meta.get('format')}, expected {FORMAT_VERSION}; rebuild it")
    if meta.get("prefix", PREFIX) != PREFIX:
        raise ValueError(f"RAG index {path} was built with prefix {meta.get('prefix')}; this code uses {PREFIX}")
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    z = np.load(path / "postings.npz")
    offsets = np.load(path / "offsets.npy")
    p_art = np.zeros(meta["n_passages"], dtype=np.int32)
    p_sec: list[str] = []
    with open(path / "passages.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            p_art[i] = row["a"]
            p_sec.append(row["s"])
    if len(p_sec) != meta["n_passages"]:
        raise ValueError(f"RAG index {path}: passages.jsonl has {len(p_sec)} rows, meta says {meta['n_passages']}")
    return BM25Index(vocab, z["indptr"], z["docs"], z["tfs"], z["doc_len"], meta["articles"], p_art, p_sec,
                     texts=None, path=path, offsets=offsets, k1=meta["k1"], b=meta["b"])


# ---- query building and passage formatting ------------------------------------------------------------------------

_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")
_IMAGE_MARKER = re.compile(r"\[Obraz:\s*[^\]]*\]")
_DESC_MARKER = re.compile(r"\[Opis źródła[^\]]*\]")
_TASK_HEADER = re.compile(r"(?m)^\s*Zadanie\s+\d+(?:\.\d+)?\s*(?:\(\s*\d+\s*pkt\s*\))?\s*$")
_BLANKS = re.compile(r"[_.…]{3,}")


def _clean(text: str) -> str:
    text = _IMAGE_MARKER.sub(" ", str(text or ""))
    text = _DESC_MARKER.sub(" ", text)
    text = _TASK_HEADER.sub(" ", text)
    text = _BLANKS.sub(" ", text)
    return " ".join(text.split())


def build_query(item: Mapping[str, Any], max_chars: int = 1600) -> str:
    """Retrieval query for an exam item: the question, then the source text (image markers, task headers and answer
    blanks removed; names and dates kept), trimmed to max_chars at a word boundary. The question comes first so it
    survives the trim."""
    q = _clean(item.get("question", ""))
    s = _clean(item.get("source_text", ""))
    text = f"{q}\n{s}".strip() if s else q
    if len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        text = text[: cut if cut > max_chars * 0.6 else max_chars]
    return text.strip()


def format_passages(passages: Iterable[Mapping[str, Any]], max_chars: int = 2400) -> str:
    """Numbered passages "[1] Title – Section\\ntext", separated by blank lines, total <= max_chars. A passage that does
    not fit is cut at a word boundary with "…" when at least ~200 characters of it fit; otherwise it is left out."""
    blocks: list[str] = []
    used = 0
    n = 0
    for p in passages or []:
        title = " ".join(str(p.get("title") or "").split())
        sec = " ".join(str(p.get("section") or "").split())
        text = " ".join(str(p.get("text") or "").split())
        if not text:
            continue
        head = f"[{n + 1}] {title}" + (f" – {sec}" if sec else "")
        sep = 2 if blocks else 0
        room = max_chars - used - sep - len(head) - 1
        if room < 1:
            break
        if len(text) > room:
            if room < 200:
                break
            cut = text.rfind(" ", 0, room - 1)
            text = text[: cut if cut > room * 0.5 else room - 1].rstrip(" ,;:") + "…"
        block = f"{head}\n{text}"
        blocks.append(block)
        used += sep + len(block)
        n += 1
    return "\n\n".join(blocks)
