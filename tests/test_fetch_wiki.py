import importlib.util
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_wiki", ROOT / "scripts" / "fetch_wiki.py")
fetch_wiki = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_wiki)

LONG = "Treść artykułu. " * 100


def _run(tmp_path, monkeypatch, titles: dict, *args, pages=None):
    calls = []
    pages = pages or {}

    def fake_resolve(ts):
        return {str(t): str(t) for t in ts if str(t) != "Brak"}

    def fake_fetch(title):
        calls.append(title)
        return {"title": title, "pageid": 1, "revid": 2, "url": f"https://pl.wikipedia.org/wiki/{title}", "extract": pages.get(title, LONG)}

    monkeypatch.setattr(fetch_wiki, "resolve_titles", fake_resolve)
    monkeypatch.setattr(fetch_wiki, "fetch_extract", fake_fetch)
    tfile, out = tmp_path / "titles.yaml", tmp_path / "articles.jsonl"
    tfile.write_text(yaml.safe_dump(titles, allow_unicode=True, sort_keys=False))
    monkeypatch.setattr(sys, "argv", ["fetch_wiki.py", "--titles", str(tfile), "--out", str(out), *args])
    fetch_wiki.main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    report = json.loads((tmp_path / "fetch_report.json").read_text())
    return rows, report, calls


def test_cross_subject_duplicates_are_reported_on_first_and_cached_runs(tmp_path, monkeypatch):
    titles = {"historia": ["Rewolucja przemysłowa", "Brak", 1984], "geografia": ["Rewolucja przemysłowa", "Klimat"]}
    rows, report, calls = _run(tmp_path, monkeypatch, titles)
    assert {r["title"]: r["subject"] for r in rows} == {"Rewolucja przemysłowa": "historia", "1984": "historia", "Klimat": "geografia"}
    assert len(report["duplicates"]) == 1 and report["missing"] == ["historia:Brak"] and report["fetched"] == 3
    rows2, report2, calls2 = _run(tmp_path, monkeypatch, titles)
    assert calls2 == [] and report2["cached"] == 3 and len(report2["duplicates"]) == 1 and rows2 == rows


def test_rerun_applies_subject_moves_pruning_and_min_chars(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, {"historia": ["A", "B", "C"]}, "--min-chars", "100", pages={"C": "krótki " * 30})
    rows, report, calls = _run(tmp_path, monkeypatch, {"wos": ["A"], "historia": ["C"]})  # default --min-chars 800
    assert {r["title"]: r["subject"] for r in rows} == {"A": "wos"}
    assert report["subject_changed"] == ["A: historia -> wos"] and report["pruned"] == ["B"]
    assert calls == [] and report["stubs"] == ["historia:C (209 chars, cached)"]
