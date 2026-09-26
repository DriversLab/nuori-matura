#!/usr/bin/env python
"""Build the local RAG knowledge base: Polish Wikipedia articles -> ~180-260-word passages -> BM25 index.

    python scripts/build_rag.py                      # fetch missing articles (resumable), chunk, index
    python scripts/build_rag.py --no-fetch           # rebuild the index from the cached articles only
    python scripts/build_rag.py --limit 20           # smoke test on the first 20 titles

Inputs/outputs (everything but the title list is gitignored):
    data/rag/titles_history.yaml   {epoch: [title, ...]}  (committed)
    data/rag/resolved.json         requested title -> canonical article title (null = missing/disambiguation)
    data/rag/articles.jsonl        one fetched article per line, appended as it arrives (resume = rerun)
    data/rag/index/                harness.rag index (load_index("data/rag/index"))
    data/rag/build_report.json     counts, missing titles, stubs

Wikipedia access goes through train.wiki (descriptive User-Agent, maxlag, Retry-After), one request at a time with a
pause between article fetches.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import datetime as dt
import json
import time

import yaml

from harness.rag import BM25Index, chunk_article, load_index
from train.wiki import clean_extract, fetch_extract, resolve_titles


def load_titles(path: pathlib.Path) -> list[tuple[str, str]]:
    """[(epoch, title)] in file order, first listing wins."""
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for epoch, titles in data.items():
        for t in titles or []:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                out.append((str(epoch), t))
    return out


def read_cache(path: pathlib.Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:   # a line cut by an interrupted run
                    continue
                out[row["title"]] = row
    return out


def resolve_all(titles: list[str], cache_path: pathlib.Path, refresh: bool = False) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    if cache_path.exists() and not refresh:
        resolved = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [t for t in titles if t not in resolved]
    for i in range(0, len(todo), 50):
        chunk = todo[i : i + 50]
        got = resolve_titles(chunk)
        for t in chunk:
            resolved[t] = got.get(t)
        cache_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"resolved {min(i + 50, len(todo))}/{len(todo)} titles", flush=True)
    return resolved


def fetch_missing(canon: list[str], cache: dict[str, dict], out_path: pathlib.Path, sleep: float) -> int:
    todo = [t for t in canon if t not in cache]
    n = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, title in enumerate(todo):
            page = fetch_extract(title)
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            if page:
                row = {"title": page["title"], "requested": title, "pageid": page["pageid"], "revid": page["revid"],
                       "url": page["url"], "fetched_at": now, "text": clean_extract(page["extract"])}
            else:
                row = {"title": title, "requested": title, "missing": True, "fetched_at": now, "text": ""}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            cache[title] = row
            if page and page["title"] != title:
                cache[page["title"]] = row
            n += 1
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"fetched {i + 1}/{len(todo)} ({rate:.1f}/s, ~{(len(todo) - i - 1) / max(rate, 1e-9) / 60:.1f} min "
                      f"left) last: {title} ({len(row['text'])} chars)", flush=True)
            if sleep > 0:
                time.sleep(sleep)
    return n


def dir_size(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--titles", default="data/rag/titles_history.yaml")
    ap.add_argument("--articles", default="data/rag/articles.jsonl")
    ap.add_argument("--resolved", default="data/rag/resolved.json")
    ap.add_argument("--index", default="data/rag/index")
    ap.add_argument("--report", default="data/rag/build_report.json")
    ap.add_argument("--no-fetch", action="store_true", help="index only what is already cached")
    ap.add_argument("--refresh-resolve", action="store_true", help="re-resolve every title (redirects may change)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N titles (smoke test)")
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between article requests in seconds (default 0.2)")
    ap.add_argument("--min-chars", type=int, default=300, help="skip articles shorter than this (default 300)")
    ap.add_argument("--min-words", type=int, default=180)
    ap.add_argument("--max-words", type=int, default=260)
    args = ap.parse_args()

    rp = lambda p: pathlib.Path(p) if pathlib.Path(p).is_absolute() else ROOT / p  # noqa: E731
    titles = load_titles(rp(args.titles))
    if args.limit:
        titles = titles[: args.limit]
    epoch_of = {t: e for e, t in titles}
    art_path = rp(args.articles)
    art_path.parent.mkdir(parents=True, exist_ok=True)
    cache = read_cache(art_path)
    print(f"{len(titles)} titles, {len(cache)} cached articles", flush=True)

    if args.no_fetch:
        resolved_path = rp(args.resolved)
        resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else {}
        resolved = {t: resolved.get(t, t if t in cache else None) for _, t in titles}
    else:
        resolved = resolve_all([t for _, t in titles], rp(args.resolved), refresh=args.refresh_resolve)
    canon_order: list[str] = []
    canon_epoch: dict[str, str] = {}
    for _, t in titles:
        c = resolved.get(t)
        if c and c not in canon_epoch:
            canon_epoch[c] = epoch_of[t]
            canon_order.append(c)
    if not args.no_fetch:
        n = fetch_missing(canon_order, cache, art_path, args.sleep)
        print(f"fetched {n} new articles", flush=True)

    report = {"titles": len(titles), "resolved": sum(1 for _, t in titles if resolved.get(t)),
              "unique_articles": len(canon_order),
              "unresolved": [t for _, t in titles if not resolved.get(t)], "missing": [], "stubs": [],
              "duplicates": sorted(t for _, t in titles if resolved.get(t) and resolved[t] != t
                                   and sum(1 for _, u in titles if resolved.get(u) == resolved[t]) > 1)}
    passages: list[dict] = []
    used: set[str] = set()
    for c in canon_order:
        row = cache.get(c)
        if row is None or row.get("missing"):
            report["missing"].append(c)
            continue
        if row["title"] in used:
            continue
        used.add(row["title"])
        if len(row["text"]) < args.min_chars:
            report["stubs"].append(f"{c} ({len(row['text'])} chars)")
            continue
        for ch in chunk_article(row["text"], min_words=args.min_words, max_words=args.max_words):
            passages.append({"title": row["title"], "section": ch["section"], "text": ch["text"],
                             "url": row.get("url"), "epoch": canon_epoch[c]})

    t0 = time.time()
    index = BM25Index.build(passages)
    idx_path = index.save(rp(args.index))
    build_s = time.time() - t0
    t0 = time.time()
    loaded = load_index(idx_path)
    load_s = time.time() - t0
    t0 = time.time()
    for q in ("unia lubelska 1569", "przyczyny wybuchu powstania styczniowego", "reformy Kazimierza Wielkiego"):
        loaded.search(q, k=5)
    search_ms = (time.time() - t0) / 3 * 1000

    words = [len(p["text"].split()) for p in passages]
    words.sort()
    report.update({
        "indexed_articles": len(used) - len(report["stubs"]), "passages": len(passages),
        "passage_words": {"min": words[0] if words else 0, "median": words[len(words) // 2] if words else 0,
                          "max": words[-1] if words else 0,
                          "share_180_260": round(sum(180 <= w <= 260 for w in words) / max(len(words), 1), 3)},
        "terms": len(loaded.vocab), "postings": int(loaded.indptr[-1]),
        "index_dir": str(idx_path), "index_bytes_on_disk": dir_size(idx_path),
        "ram_estimate_bytes": loaded.memory_bytes(), "build_seconds": round(build_s, 1),
        "load_seconds": round(load_s, 2), "search_ms": round(search_ms, 1),
        "per_epoch": {e: sum(1 for c in used if canon_epoch.get(c) == e) for e in dict.fromkeys(canon_epoch.values())},
    })
    rp(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}
    summary["index_MB_on_disk"] = round(report["index_bytes_on_disk"] / 2**20, 1)
    summary["ram_estimate_MB"] = round(report["ram_estimate_bytes"] / 2**20, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
