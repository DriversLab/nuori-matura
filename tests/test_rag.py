"""harness.rag: normaliser, chunking, BM25 index (build/save/load/search), query building, passage formatting.
Tiny in-memory corpus; no network, no model."""
import importlib.util
import json
import pathlib
import sys

import yaml

from harness import rag
from harness.prompts import build_messages

ROOT = pathlib.Path(__file__).resolve().parents[1]

CORPUS = [
    {"title": "Unia lubelska", "section": "", "url": "https://pl.wikipedia.org/wiki/Unia_lubelska",
     "text": "Unia lubelska – umowa zawarta 1 lipca 1569 roku na sejmie w Lublinie między Koroną Królestwa Polskiego a "
             "Wielkim Księstwem Litewskim. Utworzyła Rzeczpospolitą Obojga Narodów ze wspólnym monarchą, sejmem i "
             "polityką zagraniczną; oba państwa zachowały odrębne skarby, wojska i urzędy."},
    {"title": "Unia lubelska", "section": "Skutki", "url": "https://pl.wikipedia.org/wiki/Unia_lubelska",
     "text": "Przed zawarciem unii do Korony włączono Podlasie, Wołyń i Kijowszczyznę. Wspólna elekcja króla odbywała "
             "się odtąd w Polsce."},
    {"title": "Powstanie styczniowe", "section": "Przyczyny", "url": "https://pl.wikipedia.org/wiki/Powstanie_styczniowe",
     "text": "Bezpośrednią przyczyną wybuchu powstania była branka, czyli pobór do wojska rosyjskiego zarządzony przez "
             "Aleksandra Wielopolskiego. Powstanie wybuchło 22 stycznia 1863 roku; manifest Tymczasowego Rządu "
             "Narodowego ogłosił uwłaszczenie chłopów."},
    {"title": "Kazimierz III Wielki", "section": "Reformy", "url": "https://pl.wikipedia.org/wiki/Kazimierz_III_Wielki",
     "text": "Kazimierz Wielki wydał statuty wiślicko-piotrkowskie, ujednolicające prawo, założył w 1364 roku Akademię "
             "Krakowską i przeprowadził reformę monetarną, wprowadzając grosz krakowski."},
    {"title": "Okrągły Stół (Polska)", "section": "Postanowienia", "url": "https://pl.wikipedia.org/wiki/Okrągły_Stół",
     "text": "Obrady Okrągłego Stołu w 1989 roku zakończyły się legalizacją NSZZ „Solidarność”, utworzeniem Senatu i "
             "urzędu prezydenta oraz zapowiedzią częściowo wolnych wyborów do Sejmu."},
]


def _index():
    return rag.build_index(CORPUS)


def test_normalize_stems_polish_inflection_and_keeps_years():
    assert rag.normalize("Powstanie") == rag.normalize("powstania") == rag.normalize("powstaniu")
    assert rag.normalize("Kazimierz") == rag.normalize("Kazimierza")
    assert rag.normalize("unia") == rag.normalize("unii") == rag.normalize("unią")
    assert rag.normalize("w 1569 roku, oraz – i") == ["1569"]
    assert rag.normalize("Wyjaśnij, podaj źródło") == []
    assert rag.normalize("22 stycznia") != rag.normalize("22 styczniowe")   # a date month does not meet the uprising
    assert rag.normalize("Konstytucja 3 maja")[-1] == "maja"


def test_search_ranks_the_right_article_first():
    ix = _index()
    for query, title in [("postanowienia unii lubelskiej 1569", "Unia lubelska"),
                         ("przyczyny wybuchu powstania styczniowego", "Powstanie styczniowe"),
                         ("reformy Kazimierza Wielkiego", "Kazimierz III Wielki"),
                         ("Okrągły Stół 1989 postanowienia", "Okrągły Stół (Polska)")]:
        hits = ix.search(query, k=3)
        assert hits and hits[0]["title"] == title, (query, hits)
        assert set(hits[0]) == {"title", "section", "text", "url", "score"}
        assert hits[0]["score"] > 0 and hits[0]["url"].startswith("https://pl.wikipedia.org/")
        assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_search_limits_per_title_and_ignores_unknown_terms():
    ix = _index()
    assert len([h for h in ix.search("unia lubelska", k=5, max_per_title=1) if h["title"] == "Unia lubelska"]) == 1
    assert len(ix.search("unia lubelska Korona Litwa", k=5, max_per_title=2)) >= 2
    assert ix.search("xyzzy qwerty", k=5) == []
    assert ix.search("", k=5) == []


def test_title_boost_prefers_the_named_article():
    ix = _index()
    cov = ix.title_coverage(ix.term_id[t] for t in rag.normalize("unia lubelska") if t in ix.term_id)
    arts = [a["title"] for a in ix.articles]
    assert cov[arts.index("Unia lubelska")] > 0 and cov[arts.index("Powstanie styczniowe")] == 0
    hits = ix.search("Korona i Litwa, unia", k=2, title_boost=4.0)
    assert hits[0]["title"] == "Unia lubelska"
    plain, boosted = ix.scores("unia lubelska", title_boost=0.0), ix.scores("unia lubelska", title_boost=1.0)
    assert (boosted >= plain).all() and (boosted > plain).any()


def test_save_load_roundtrip(tmp_path):
    ix = _index()
    ix.save(tmp_path / "index")
    loaded = rag.load_index(tmp_path / "index")
    assert loaded.n == ix.n and loaded.vocab == ix.vocab
    q = "branka Wielopolski 1863"
    assert loaded.search(q, k=3) == ix.search(q, k=3)
    assert rag.load_index(tmp_path / "index" / "meta.json").n == ix.n
    assert loaded.memory_bytes() > 0


def test_build_query_drops_markers_and_keeps_names_dates():
    item = {"id": "5", "question": "Zadanie 5.1 (2 pkt)\nWyjaśnij, jaki był skutek unii lubelskiej. ………",
            "source_text": "[Obraz: images/5.png]\n[Opis źródła 5]\nZygmunt August w 1569 roku zwołał sejm do Lublina."}
    q = rag.build_query(item)
    assert "Obraz" not in q and "Opis źródła" not in q and "Zadanie 5.1" not in q and "…" not in q
    assert "Zygmunt August" in q and "1569" in q and q.startswith("Wyjaśnij")
    long_item = {"question": "Kto? " + "słowo " * 1000, "source_text": "x " * 1000}
    assert len(rag.build_query(long_item, max_chars=500)) <= 500
    assert rag.build_query({"question": "Pytanie"}) == "Pytanie"


def test_format_passages_numbered_with_titles_and_budget():
    ix = _index()
    hits = ix.search("unia lubelska powstanie styczniowe", k=4)
    text = rag.format_passages(hits, max_chars=2400)
    assert text.startswith("[1] ") and "\n\n[2] " in text
    assert "Unia lubelska" in text and "Powstanie styczniowe – Przyczyny" in text
    short = rag.format_passages(hits, max_chars=300)
    assert len(short) <= 300 and short.startswith("[1] ")
    assert rag.format_passages([], max_chars=2400) == ""
    long = [{"title": "T", "section": "", "text": "słowo " * 2000}]
    cut = rag.format_passages(long, max_chars=1000)
    assert len(cut) <= 1000 and cut.endswith("…")


def test_chunk_article_sizes_and_sections():
    para = " ".join(f"Zdanie numer {i} opowiada o wydarzeniach historycznych w Polsce." for i in range(12))  # ~96 words
    text = "Wstęp artykułu. " * 5 + "\n\n## Tło\n\n" + "\n\n".join([para] * 8) + "\n\n## Skutki\n\nKrótko."
    chunks = rag.chunk_article(text, min_words=180, max_words=260)
    words = [len(c["text"].split()) for c in chunks]
    assert all(w <= 325 for w in words), words
    assert sum(180 <= w <= 260 for w in words) >= len(words) - 2, words
    assert all(c["text"].strip() for c in chunks)
    assert any(c["section"].startswith("Tło") for c in chunks)
    assert "Krótko." in chunks[-1]["text"]                  # a tiny tail section is folded in, not dropped
    joined = " ".join(" ".join(c["text"].split()) for c in chunks)
    assert joined.count("Zdanie numer 11") == 8              # nothing lost or duplicated
    assert rag.chunk_article("") == []


def test_rag_prompt_uses_formatted_passages():
    ix = _index()
    item = {"id": "3", "question": "Podaj bezpośrednią przyczynę wybuchu powstania styczniowego.", "source_text": "",
            "max_points": 1, "answer_format": "Tekst po polsku. Podaj wszystkie wymagane elementy odpowiedzi."}
    passages = rag.format_passages(ix.search(rag.build_query(item), k=2))
    msgs = build_messages(item, None, style="rag", passages=passages)
    assert msgs[1]["content"].startswith("Materiały pomocnicze:\n[1] Powstanie styczniowe")


def test_titles_file_is_deduplicated():
    data = yaml.safe_load(open(ROOT / "data" / "rag" / "titles_history.yaml", encoding="utf-8"))
    titles = [t for ts in data.values() for t in ts]
    assert len(titles) == len(set(titles)) > 2000
    assert "Unia lubelska" in titles and "Okrągły Stół (Polska)" in titles


def test_build_script_offline(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("build_rag", ROOT / "scripts" / "build_rag.py")
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    body = "\n\n".join(" ".join(f"{name} zdanie {i} o historii Polski." for i in range(40)) for name in ("Aaa", "Bbb"))
    calls = []
    monkeypatch.setattr(br, "resolve_titles", lambda ts: {t: t for t in ts if t != "Brak"})
    monkeypatch.setattr(br, "fetch_extract", lambda t: calls.append(t) or {
        "title": t, "pageid": 1, "revid": 2, "url": f"https://pl.wikipedia.org/wiki/{t}", "extract": f"{t}. {body}"})
    titles = tmp_path / "titles.yaml"
    titles.write_text(yaml.safe_dump({"e1": ["Alfa", "Brak"], "e2": ["Alfa", "Beta"]}, allow_unicode=True))
    args = ["--titles", str(titles), "--articles", str(tmp_path / "a.jsonl"), "--resolved", str(tmp_path / "r.json"),
            "--index", str(tmp_path / "idx"), "--report", str(tmp_path / "rep.json"), "--sleep", "0"]
    monkeypatch.setattr(sys, "argv", ["build_rag.py", *args])
    br.main()
    report = json.loads((tmp_path / "rep.json").read_text())
    assert calls == ["Alfa", "Beta"] and report["unresolved"] == ["Brak"] and report["indexed_articles"] == 2
    assert rag.load_index(tmp_path / "idx").search("Beta historia", k=1)[0]["title"] == "Beta"
    monkeypatch.setattr(sys, "argv", ["build_rag.py", *args])
    br.main()                                                     # resume: everything cached, no new requests
    assert calls == ["Alfa", "Beta"]
