"""Polish Wikipedia plaintext fetching (MediaWiki API, TextExtracts full-article extracts).

Verified behaviour (docs/research/data_sources.md, Task B):
  * a descriptive User-Agent is REQUIRED (default python-requests UA gets HTTP 403);
  * prop=extracts&explaintext=1 returns the FULL article but only for ONE title per request;
  * send maxlag=5, one request at a time, honour Retry-After on maxlag/429/503;
  * resolve redirects / drop missing + disambiguation pages with prop=pageprops (<=50 titles per call).
"""
from __future__ import annotations

import re
import time
from typing import Iterable

import requests

API = "https://pl.wikipedia.org/w/api.php"
USER_AGENT = (
    "MaturaBielikFT-data/0.1 (hackathon research; https://github.com/speakleash; contact via repo issues) "
    f"python-requests/{requests.__version__}"
)
DROP_SECTIONS = {
    "Przypisy", "Bibliografia", "Linki zewnętrzne", "Zobacz też", "Uwagi", "Galeria", "Literatura",
    "Źródła", "Bibliografia uzupełniająca", "Filmografia", "Dyskografia", "Odznaczenia i wyróżnienia",
}
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)

_session: requests.Session | None = None


def _s() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    return _session


def api(**params) -> dict:
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    for attempt in range(8):
        try:
            r = _s().get(API, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 503) or (r.ok and r.json().get("error", {}).get("code") == "maxlag"):
            time.sleep(int(r.headers.get("Retry-After", 5)) * (attempt + 1))
            continue
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"])
        return j
    raise RuntimeError(f"MediaWiki API gave up after retries: {params}")


def resolve_titles(titles: Iterable[str]) -> dict[str, str]:
    """Map requested title -> canonical existing article title. Missing/invalid/disambiguation pages are omitted."""
    titles = list(dict.fromkeys(str(t).strip() for t in titles if t is not None and str(t).strip()))
    out: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        q = api(action="query", titles="|".join(chunk), redirects=1, prop="pageprops", ppprop="disambiguation")["query"]
        mapping = {t: t for t in chunk}
        for n in q.get("normalized", []):
            mapping = {k: (n["to"] if v == n["from"] else v) for k, v in mapping.items()}
        for rd in q.get("redirects", []):
            mapping = {k: (rd["to"] if v == rd["from"] else v) for k, v in mapping.items()}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for orig, final in mapping.items():
            p = pages.get(final, {})
            if not p or p.get("missing") or p.get("invalid") or "disambiguation" in p.get("pageprops", {}):
                continue
            out[orig] = final
    return out


def fetch_extract(title: str) -> dict | None:
    q = api(action="query", prop="extracts|info", explaintext=1, exsectionformat="wiki", inprop="url", titles=title, redirects=1)
    pages = q["query"]["pages"]
    if not pages:
        return None
    p = pages[0]
    if p.get("missing") or p.get("invalid"):
        return None
    return {"title": p["title"], "pageid": p.get("pageid"), "revid": p.get("lastrevid"), "url": p.get("fullurl"), "extract": p.get("extract", "")}


_ANNOTATION = re.compile(r"^\{\\(?:displaystyle|textstyle)\s*(.*)\}$")
_HEADING_LINE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def _is_math_token(line: str) -> bool:
    """MathML token lines are indented >= 4 spaces (a block opens with a whitespace-only line and may contain empty
    lines); text that continues after a formula starts with at most one space."""
    return line.startswith("    ") or (line != "" and not line.strip())


def strip_math_markup(text: str) -> str:
    """TextExtracts renders each <math> as indented MathML tokens (one per line) followed by an indented
    '{\\displaystyle LATEX}' (or textstyle) line. Collapse every such block into '$LATEX$': inline when it continues a
    text line, on its own paragraph when an empty line precedes it. Headings and non-math indented text are untouched,
    and the text following a formula is re-attached verbatim (so ' i', ', gdzie', '-wymiarowej' keep their spacing)."""
    out: list[str] = []
    joining = False
    for line in text.split("\n"):
        m = _ANNOTATION.match(line.strip()) if line.startswith("    ") else None
        if m:
            latex = m.group(1).strip()
            j = len(out)
            while j > 0 and (_is_math_token(out[j - 1]) or out[j - 1] == ""):
                j -= 1
            region = out[j:]
            del out[j:]
            display = bool(region) and region[0] == ""
            if out and not display and not _HEADING_LINE.match(out[-1]):
                out[-1] = f"{out[-1]}{' ' if out[-1].endswith('$') else ''}${latex}$"
            else:
                if out and display:
                    out.append("")
                out.append(f"${latex}$")
            joining = True
            continue
        if joining:
            if line != "" and not line.strip():
                continue  # the block's closing whitespace-only line
            joining = False
            if line != "" and not _is_math_token(line) and not _HEADING_LINE.match(line):
                out[-1] += line
                continue
        out.append(line)
    return "\n".join(out)


def clean_extract(extract: str, keep_headings: bool = True) -> str:
    extract = strip_math_markup(extract)
    parts: list[tuple] = []
    pos = 0
    for m in _HEADING.finditer(extract):
        parts.append((None, extract[pos : m.start()]))
        parts.append(((len(m.group(1)), m.group(2)), None))
        pos = m.end()
    parts.append((None, extract[pos:]))
    out: list[str] = []
    skip: int | None = None
    for head, body in parts:
        if head:
            lvl, name = head
            if skip is not None and lvl <= skip:
                skip = None
            if skip is None and name in DROP_SECTIONS:
                skip = lvl
            if skip is None and keep_headings:
                out.append(f"\n## {name}\n")
        elif skip is None:
            out.append(body)
    text = "".join(out).replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[: cut if cut > max_chars * 0.7 else max_chars].rstrip() + "\n[…]"
