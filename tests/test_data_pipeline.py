"""Data pipeline: external converters, dedup, blind-solver verification, build_data, processed dataset, API generation."""
from __future__ import annotations

import copy
import importlib.util
import json
import random
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from eval.answer_format import VARIANTS, render_target
from eval.grader import grade
from eval.schema import load_items, read_jsonl, validate_item, write_jsonl
from train import external
from train.config import ROOT, load_config
from train.dataset import (
    LeakageError,
    build_processed,
    check_no_eval_leak,
    general_count,
    render_train,
    shuffle_options,
    split_by_subject,
)
from train.dedup import DedupConfig, dedup_items, stem_text
from train.synth_verify import solver_agrees, verify_filter

FIXTURES = Path(__file__).parent / "fixtures"
SYNTH = FIXTURES / "synthetic_sample.jsonl"
GENERAL = FIXTURES / "general_sample.jsonl"


def fixture_items() -> list[dict]:
    return load_items(SYNTH)


def by_id(items: list[dict]) -> dict[str, dict]:
    return {it["id"]: it for it in items}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Fake embedding: hashed bag of 5-char word prefixes with a tiny synonym/stopword table, so a question paraphrase
# ("Kiedy ... Podaj rok." vs "W którym roku ...") lands close while unrelated questions stay far apart.
_SYNONYMS = {"kiedy": "rok", "roku": "rok", "rok": "rok"}
_STOPWORDS = {"w", "którym", "i", "z", "na", "się", "jest", "a", "to"}


def fake_embed(texts: list[str], dim: int = 4096) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"\w+", text.casefold()):
            if tok not in _STOPWORDS:
                out[i, zlib.crc32(_SYNONYMS.get(tok, tok[:5]).encode()) % dim] += 1
    return out


def random_embed(texts: list[str]) -> np.ndarray:
    """Unrelated vectors for different texts: isolates the fuzzy rule."""
    return np.stack([np.random.default_rng(zlib.crc32(t.encode())).normal(size=256) for t in texts]).astype(np.float32)


# ----------------------------------------------------------------------------- fixtures


def test_synthetic_fixture_covers_subjects_types_and_letters():
    items = fixture_items()
    assert len(items) >= 48
    assert all(it["id"].startswith("fixture-") and (it.get("rationale") or "").strip() for it in items)
    assert {it["subject"] for it in items} == {"jezyk_polski", "historia", "wos", "geografia", "biologia", "chemia", "fizyka", "matematyka"}
    assert {it["type"] for it in items} == {"mc", "multi", "tf", "match", "short", "numeric"}
    letters = Counter(it["answer"] for it in items if it["type"] == "mc")
    assert set(letters) == set("ABCD") and max(letters.values()) - min(letters.values()) <= 1


def test_general_fixture_shape():
    rows = read_jsonl(GENERAL)
    assert len(rows) >= 40
    assert all([m["role"] for m in r["messages"]] == ["user", "assistant"] for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)


# ----------------------------------------------------------------------------- external converters


def test_llmzszl_matura_items_zero_based_dedup_and_invalid():
    rows = [
        {"question": "Zawodowe?", "answers": ["a", "b", "c", "d"], "correct_answer_index": 0, "year": 2020, "type": "Egzaminy Zawodowe", "name": "x"},
        {"question": "Liczba log9(27)+log9(3) jest równa", "answers": ["81", "9", "4", "2"], "correct_answer_index": 3, "year": 2023, "type": "Egzaminy Maturalne", "name": "Matematyka"},
        {"question": "Liczba log9(27)+log9(3) jest równa", "answers": ["81", "9", "4", "2"], "correct_answer_index": 3, "year": 2023, "type": "Egzaminy Maturalne", "name": "Matematyka"},
        {"question": "Pięć opcji", "answers": ["1", "2", "3", "4", "5"], "correct_answer_index": 4, "year": 2019, "type": "Egzaminy Maturalne", "name": "Fizyka"},
        {"question": "Pusta opcja", "answers": ["x y", "", "z", "w"], "correct_answer_index": 0, "year": 2019, "type": "Egzaminy Maturalne", "name": "Biologia"},
    ]
    items, stats = external.llmzszl_matura_items(rows)
    assert [it["id"] for it in items] == ["llmzszl-1", "llmzszl-3"]
    assert items[0]["answer"] == "D" and items[0]["options"]["D"] == "2" and items[0]["source"] == "llmzszl:2023"
    assert items[1]["answer"] == "E" and items[1]["subject"] == "fizyka"
    assert stats["duplicate_question"] == 1 and stats["invalid"] == 1 and stats["kept"] == 2


def test_blocklist_converters_index_bases():
    include = [{"question": "Q?", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d", "answer": 2, "subject": "Math"},
               {"question": "Prof?", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d", "answer": 0, "subject": "Professional certification"}]
    rows = external.include_blocklist(include, "test")
    assert len(rows) == 1 and rows[0]["answer_text"] == "c" and rows[0]["subject"] == "matematyka"

    dokato = [{"question": "Zaznacz państwo.", "options": ["Włochy", "Rosja", "Francja", "USA"], "answer": "3",
               "category_original_lang": "Wiedza o społeczeństwie", "file_name": "MWO.pdf"}]
    assert external.dokato_matura_blocklist(dokato)[0]["answer_text"] == "Francja"

    mm = [{"question": "Cząsteczki tRNA są zbudowane:", "options": ["z jednej nici", "z dwóch nici"], "answer": 0,
           "category_original_lang": "Biologia", "file_name": "EBIP.pdf"}]
    assert external.dokato_multimodal_blocklist(mm)[0]["answer_text"] == "z jednej nici"

    llm = [{"question": "Q", "answers": ["a", "b"], "correct_answer_index": 1, "year": 2010, "type": "Egzaminy Gimnazjalne", "name": "Przyroda"},
           {"question": "V", "answers": ["a", "b"], "correct_answer_index": 1, "year": 2010, "type": "Egzaminy Zawodowe", "name": "x"}]
    rows = external.llmzszl_blocklist(llm)
    assert len(rows) == 1 and rows[0]["answer_text"] == "b" and rows[0]["subject"] is None


def test_abituria_correct_option_is_one_based():
    exam = {"exam": {"id": "matura-maj-2025-podstawowa", "exercises": [
        {"id": "z01", "mode": "multipleChoice", "prompt": "Liczba (√32 − √2)² jest równa", "options": ["16", "18", "30", "34"], "correctOption": 2},
        {"id": "z09", "mode": "numeric", "prompt": "Oblicz kwotę.", "expectedValue": 735000},
        {"id": "z05", "mode": "revealOnly", "prompt": "Wykaż, że liczba jest podzielna przez 4."},
    ]}}
    rows = external.abituria_blocklist(exam)
    assert rows[0]["answer_text"] == "18"
    assert rows[1]["answer_text"] == "735000" and "options" not in rows[1]
    assert "answer_text" not in rows[2]


def test_lmamacl_and_markdown_cleaning():
    data = {"zadania": [{"id": "2014_z01", "arkusz": {"plik_źródłowy": "2014_pp"}, "treść_wprowadzająca": "## **Zadanie 1.** _****_\n\nNa podstawie tekstu określ **temat**.",
                         "podzadania": [{"numer_podzadania": "1.1", "treść": "Wyjaśnij sens tytułu."}]},
                        {"id": "2014_z02", "arkusz": {}, "treść_wprowadzająca": "", "podzadania": []}]}
    rows = external.lmamacl_blocklist(data)
    assert [r["id"] for r in rows] == ["lmamacl-2014_z01", "lmamacl-2014_z01-1.1"]
    assert rows[0]["question"] == "Na podstawie tekstu określ temat." and "answer_text" not in rows[0]


def test_general_filters_and_pllum_dedup():
    ok = [{"role": "user", "content": "Podaj stolicę Polski."}, {"role": "assistant", "content": "Stolicą Polski jest Warszawa."}]
    assert external.eu_instruct_ok(ok)
    assert not external.eu_instruct_ok([ok[0], {"role": "assistant", "content": "Warszawa. Final thought: ok"}])
    assert not external.eu_instruct_ok([ok[0], {"role": "assistant", "content": "krótko"}])
    assert not external.eu_instruct_ok([{"role": "user", "content": "x" * 2001}, ok[1]])
    assert not external.eu_instruct_ok(ok + ok)

    files = {"ranking.jsonl": [{"id": "a1", "chosen": ok}, {"id": "a2", "chosen": [{"role": "user", "content": "  podaj STOLICĘ polski. "}, ok[1]]}],
             "rating.jsonl": [{"id": "b1", "chosen": ok + ok}, {"id": "b2", "chosen": [{"role": "user", "content": "Inne pytanie?"}, ok[1]]}]}
    rows = external.pllum_candidates(files)
    assert [r["id"] for r in rows] == ["pllum-a1", "pllum-b2"]


def test_build_general_pool_excludes_heldout_prompts(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def pair(u, a):
        return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]

    msgs = [pair(f"Pytanie numer {i}?", f"Odpowiedź numer {i} jest poprawna.") for i in range(10)]
    msgs += [pair("Pytanie numer 3?", "Duplikat promptu, inna odpowiedź."), pair("Ciekawe?", "**Summary** artefakt generatora")]
    parquet = tmp_path / "train.parquet"
    pq.write_table(pa.table({"messages": msgs}), parquet)
    monkeypatch.setattr(external, "_hf_file", lambda repo, fname: parquet)
    stats: dict = {}
    n = external.build_general_pool(tmp_path / "pool.jsonl", 100, 0, exclude_prompts=["pytanie   NUMER 5?"], stats=stats)
    rows = read_jsonl(tmp_path / "pool.jsonl")
    prompts = {r["messages"][0]["content"] for r in rows}
    assert n == len(rows) == 9 and "Pytanie numer 5?" not in prompts
    assert stats["heldout_prompt_overlap"] == 1 and stats["duplicate_prompt"] == 1 and stats["filtered_artifact_or_length"] == 1
    assert all(r["id"].startswith("euis-") for r in rows)


# ----------------------------------------------------------------------------- dedup


def test_dedup_text_building():
    items = by_id(fixture_items())
    assert stem_text({"id": "b", "question": "W którym roku uchwalono Konstytucję 3 maja? A. 1772 B. 1791 C. 1795 D. 1807"}) == \
        "W którym roku uchwalono Konstytucję 3 maja?"
    from train.dedup import answer_text

    assert answer_text(items["fixture-historia-01"]) == "1410"
    assert answer_text(items["fixture-historia-03"]).endswith("utrzymała zasadę liberum veto. (F)")
    assert "unia lubelska -> 1569" in answer_text(items["fixture-historia-05"])
    assert answer_text(items["fixture-matematyka-06"]) == "22,5"
    assert "hołd pruski" in stem_text(items["fixture-historia-05"])  # left elements are part of the stem
    assert "1569" not in stem_text(items["fixture-historia-05"])


def test_dedup_catches_paraphrase_blocklist_and_intra_keeps_distinct():
    distinct = fixture_items()
    paraphrase = {"id": "cand-para", "subject": "historia", "type": "short", "question": "Kiedy uchwalono Konstytucję 3 maja? Podaj rok.", "answer": ["1791"]}
    no_answer_dup = {"id": "cand-block", "subject": "chemia", "type": "mc", "question": "Który gaz stanowi największą część powietrza atmosferycznego?",
                     "options": {"A": "tlen", "B": "azot", "C": "argon", "D": "neon"}, "answer": "B"}
    twin_a = {"id": "cand-twin-a", "subject": "biologia", "type": "short", "question": "Jak nazywa się barwnik nadający liściom zieloną barwę?", "answer": ["chlorofil"]}
    twin_b = {**twin_a, "id": "cand-twin-b"}
    candidates = [validate_item(copy.deepcopy(c)) for c in (paraphrase, no_answer_dup, twin_a, twin_b)] + distinct
    protected = {
        "heldout": [validate_item({"id": "held-1", "subject": "historia", "type": "short", "question": "W którym roku uchwalono Konstytucję 3 maja?", "answer": ["1791"]})],
        "dev": [],
        "blocklist": [{"id": "blk-1", "source": "test", "subject": None,
                       "question": "Który gaz stanowi największą część powietrza atmosferycznego? A. tlen B. azot C. argon D. dwutlenek węgla"}],
    }
    kept, report = dedup_items(candidates, protected, DedupConfig(cache_dir=None), embed_fn=fake_embed)

    assert {c["id"] for c in kept} == {"cand-twin-a"} | {it["id"] for it in distinct}
    assert report["counts"] == {"dup_vs_heldout": 1, "dup_vs_dev": 0, "dup_vs_blocklist": 1, "intra_dup": 1}
    assert report["dropped_ids"]["intra_dup"] == ["cand-twin-b"]
    para = report["examples"]["dup_vs_heldout"][0]
    assert para["id"] == "cand-para" and para["methods"] == ["qa"] and para["matched_id"] == "held-1"
    assert 0.8 <= para["scores"]["qa"] and para["scores"]["q"] < 0.9
    block = report["examples"]["dup_vs_blocklist"][0]
    assert block["matched_id"] == "blk-1" and "q" in block["methods"] and block["scores"]["qa"] == -1.0  # no answer -> no QA check
    assert report["n_candidates"] == len(candidates) and report["n_kept"] == len(kept)
    assert report["thresholds"]["thr_qa"] == 0.8 and report["model"] == "BAAI/bge-m3"


def test_dedup_strips_exam_instructions_before_matching():
    """'Dokończ zdanie. Wybierz właściwą odpowiedź spośród podanych.' alone supplied 7 shared words: unrelated stems matched."""
    lead = "Dokończ zdanie. Wybierz właściwą odpowiedź spośród podanych."
    candidate = validate_item({"id": "c1", "subject": "geografia", "type": "mc", "question": f"{lead} Stolicą Australii jest",
                               "options": {"A": "Sydney", "B": "Canberra"}, "answer": "B"})
    long_q = "Liczba rozwiązań równania kwadratowego o ujemnym wyróżniku w zbiorze liczb rzeczywistych wynosi"
    copy_item = validate_item({"id": "c2", "subject": "matematyka", "type": "mc", "question": f"{lead} {long_q}",
                               "options": {"A": "0", "B": "1"}, "answer": "A"})
    protected = {"blocklist": [
        {"id": "p1", "source": "t", "subject": None, "question": f"{lead} Liczba √60 jest A. 7 B. 8"},
        {"id": "p2", "source": "t", "subject": None, "question": f"{long_q} {lead} A. 0 B. 1"},
    ]}
    kept, report = dedup_items([candidate, copy_item], protected, DedupConfig(cache_dir=None), embed_fn=random_embed)
    assert [c["id"] for c in kept] == ["c1"] and report["dropped_ids"]["dup_vs_blocklist"] == ["c2"]
    assert stem_text(candidate) == "Stolicą Australii jest" and stem_text(protected["blocklist"][0]) == "Liczba √60 jest"


def test_dedup_fuzzy_rule_word_count_and_length_guard():
    def short_item(i, q):
        return validate_item({"id": f"c{i}", "subject": "matematyka", "type": "short", "question": q, "answer": ["180"]})

    near_copy = short_item(1, "Oblicz sumę miar kątów wewnętrznych dowolnego trójkąta na płaszczyźnie euklidesowej.")
    too_short = short_item(2, "Podaj sumę kątów trójkąta.")
    inside_long_passage = short_item(3, "Jakie zwierzę opisuje autor tekstu w drugim akapicie?")
    protected = {"dev": [
        {"id": "p1", "source": "t", "subject": None, "question": "Oblicz sumę miar kątów wewnętrznych dowolnego trójkąta na płaszczyźnie."},
        {"id": "p2", "source": "t", "subject": None, "question": "Podaj sumę kątów w trójkącie."},
        {"id": "p3", "source": "t", "subject": None, "question": "Przeczytaj tekst. " + "Autor opisuje w drugim akapicie zwierzę jakie spotkał w lesie tekstu. " * 6},
    ]}
    kept, report = dedup_items([near_copy, too_short, inside_long_passage], protected, DedupConfig(cache_dir=None), embed_fn=random_embed)
    assert [c["id"] for c in kept] == ["c2", "c3"]
    ex = report["examples"]["dup_vs_dev"][0]
    assert ex["methods"] == ["fuzzy"] and ex["scores"]["fuzzy"] >= 92 and report["methods"]["dup_vs_dev"]["fuzzy"] == 1


def test_max_similarity_chunking_matches_full():
    from train.dedup import max_similarity

    rng = np.random.default_rng(0)
    a = rng.normal(size=(37, 8)).astype(np.float32)
    b = rng.normal(size=(53, 8)).astype(np.float32)
    best, idx = max_similarity(a, b, chunk_size=5)
    full = a @ b.T
    assert np.array_equal(idx, full.argmax(1)) and np.allclose(best, full.max(1))


# ----------------------------------------------------------------------------- synth_verify


def test_solver_agreement_rejects_hedged_payloads():
    """The lenient parser picks the first letter / number or a substring: a hedged payload must not confirm the key."""
    mc = validate_item({"id": "mc", "subject": "historia", "type": "mc", "question": "q",
                        "options": {"A": "Bolesław Chrobry", "B": "Mieszko I", "C": "Kazimierz", "D": "Władysław"}, "answer": "B"})
    short = validate_item({"id": "s", "subject": "historia", "type": "short", "question": "q", "answer": ["Mieszko I", "Mieszko"]})
    num = validate_item({"id": "n", "subject": "fizyka", "type": "numeric", "question": "q", "answer": 3.5, "unit": "cm"})
    tf = validate_item({"id": "t", "subject": "biologia", "type": "tf", "question": "q", "statements": ["a", "b", "c"], "answer": ["P", "F", "P"]})
    match = validate_item({"id": "m", "subject": "historia", "type": "match", "question": "q", "left": ["x", "y", "z"],
                           "right": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": {"1": "B", "2": "D", "3": "A"}})
    hedged = [(mc, "B lub C"), (mc, "B albo C"), (mc, "B (ewentualnie C)"), (mc, "B, C"), (mc, ["B", "C"]),
              (short, "nie Mieszko I, lecz Bolesław Chrobry"), (short, "Mieszko I lub Bolesław Chrobry"),
              (short, ["Mieszko I", "Bolesław Chrobry"]), (num, "3,5 lub 7"), (num, "ok. 3,5 albo 35"), (num, [3.5, 7]),
              (tf, "F, P, F, P"), (match, "1-B, 2-D, 3-A, 3-C")]
    assert [p for item, p in hedged if solver_agrees(item, p)[0]] == []
    legit = [(mc, "Odpowiedź: B"), (mc, "**B**"), (mc, "B."), (mc, "B) Mieszko I"), (mc, "Mieszko I"), (mc, ["B"]),
             (short, "książę Mieszko I"), (short, ["Mieszko I", "Mieszko"]), (num, "3,5 cm"), (tf, "PFP"), (tf, "prawda, fałsz, prawda"),
             (match, "1B 2D 3A"), (match, {"1": "B", "2": "D", "3": "A"})]
    assert [p for item, p in legit if not solver_agrees(item, p)[0]] == []


def test_verify_filter_reasons():
    items = by_id(fixture_items())
    answers = {
        "fixture-historia-01": {"id": "fixture-historia-01", "payload": "A", "confidence": "high"},
        "fixture-historia-03": {"id": "fixture-historia-03", "payload": "PPF", "confidence": "medium"},
        "fixture-geografia-05": {"id": "fixture-geografia-05", "payload": "2 km", "confidence": "high"},
        "fixture-historia-05": {"id": "fixture-historia-05", "payload": {"1": "B", "2": "D", "3": "A"}, "confidence": "high"},
        "fixture-historia-04": {"id": "fixture-historia-04", "payload": "Jerzy Waszyngton", "confidence": "high"},
        "fixture-wos-01": {"id": "fixture-wos-01", "payload": "C", "confidence": "high", "concern": "klucz?"},
        "fixture-wos-02": {"id": "fixture-wos-02", "payload": "D", "confidence": "low"},
        "fixture-wos-03": {"id": "fixture-wos-03", "payload": "P, P, P", "confidence": "low"},
    }
    chosen = [items[i] for i in answers] + [items["fixture-chemia-01"]]
    kept, report = verify_filter(chosen, answers)
    assert {it["id"] for it in kept} == {"fixture-historia-01", "fixture-historia-03", "fixture-geografia-05", "fixture-historia-05", "fixture-historia-04"}
    assert report["counts"] == {"unverified": 1, "solver_disagrees": 2, "low_confidence": 1, "solver_concern": 0}
    assert report["dropped_ids"]["unverified"] == ["fixture-chemia-01"]
    assert report["dropped_ids"]["low_confidence"] == ["fixture-wos-02"]
    disagree = {e["id"]: e for e in report["examples"]["solver_disagrees"]}
    assert disagree["fixture-wos-01"]["solver_payload"] == "C" and disagree["fixture-wos-01"]["key"] == "B"
    assert disagree["fixture-wos-03"]["key"] == "P, P, F"  # partial credit is not agreement; disagreement wins over low confidence


def test_verify_filter_drops_agreeing_items_with_a_concern_unless_kept():
    items = by_id(fixture_items())
    answers = {
        "fixture-historia-01": {"id": "fixture-historia-01", "payload": "A", "confidence": "high", "concern": "akceptuj też inne warianty"},
        "fixture-historia-04": {"id": "fixture-historia-04", "payload": "Jerzy Waszyngton", "confidence": "high", "concern": "  "},
    }
    chosen = [items[i] for i in answers]
    kept, report = verify_filter(chosen, answers)
    assert [it["id"] for it in kept] == ["fixture-historia-04"]  # whitespace-only concern is no concern
    assert report["dropped_ids"]["solver_concern"] == ["fixture-historia-01"]
    kept_all, report_all = verify_filter(chosen, answers, drop_concerns=False)
    assert len(kept_all) == 2 and report_all["counts"]["solver_concern"] == 0


# ----------------------------------------------------------------------------- processed dataset


def smoke_cfg(tmp_path: Path, **data_overrides) -> dict:
    heldout = tmp_path / "general_heldout.jsonl"  # the first 5 pool rows double as the general held-out set
    write_jsonl(read_jsonl(GENERAL)[:5], heldout)
    overrides = {
        "run_name": "pipeline-test",
        "seed": 7,
        "paths.processed_root": str(tmp_path / "processed"),
        "data.synthetic": str(SYNTH),
        "data.general_pool": str(GENERAL),
        "data.general_heldout": str(heldout),
        "data.general_heldout_n": 5,
        "data.general_val_n": 5,
        "data.val_fraction": 0.1,
        "data.renders_per_item": 2,
        "data.general_ratio": 0.25,
    }
    overrides.update({f"data.{k}": v for k, v in data_overrides.items()})
    return load_config(ROOT / "configs" / "base.yaml", overrides=overrides)


def test_build_processed_is_deterministic(tmp_path):
    a = build_processed(smoke_cfg(tmp_path), tmp_path / "a")
    b = build_processed(smoke_cfg(tmp_path), tmp_path / "b")
    for key in ("train", "val_matura", "val_general", "general_heldout"):
        assert a[key].read_bytes() == b[key].read_bytes(), key
    strip = lambda s: {k: v for k, v in s.items() if k != "processed_dir"}  # noqa: E731
    assert strip(a["stats"]) == strip(b["stats"])
    c = build_processed(smoke_cfg(tmp_path, general_ratio=0.1), tmp_path / "c")
    matura = lambda p: [r for r in read_jsonl(p) if r["kind"] == "matura"]  # noqa: E731
    assert sorted(map(json.dumps, matura(a["train"]))) == sorted(map(json.dumps, matura(c["train"])))  # stages use separate RNG streams


def test_build_processed_ratio_split_and_disjointness(tmp_path):
    out = build_processed(smoke_cfg(tmp_path))
    stats = out["stats"]
    assert out["train"].parent == tmp_path / "processed" / "pipeline-test"
    assert stats["n_val_items"] == 5 and stats["n_train_items"] == 43 and stats["n_train_matura_rows"] == 86
    assert stats["n_train_general_rows"] == general_count(86, 0.25) == 29
    assert set(stats["per_subject"]) == {it["subject"] for it in fixture_items()}
    assert all(v["train"] >= 1 for v in stats["per_subject"].values())

    train, val = read_jsonl(out["train"]), read_jsonl(out["val_matura"])
    val_general, heldout = read_jsonl(out["val_general"]), read_jsonl(out["general_heldout"])
    assert Counter(r["kind"] for r in train) == {"matura": 86, "general": 29}
    train_matura_ids = {r["id"] for r in train if r["kind"] == "matura"}
    train_general_ids = {r["id"] for r in train if r["kind"] == "general"}
    assert not train_matura_ids & {r["id"] for r in val}
    assert not train_general_ids & {r["id"] for r in val_general} and len(val_general) == 5
    heldout_prompts = {r["prompt"][-1]["content"] for r in heldout}
    assert len(heldout) == 5 and not heldout_prompts & {r["prompt"][-1]["content"] for r in train + val_general if r["kind"] == "general"}
    assert all(r["variant"] == "canonical" and r["prompt"][0]["role"] == "user" for r in val)
    assert all(set(r) == {"prompt", "completion", "kind", "id", "variant"} for r in train + val + val_general + heldout)
    assert all(r["completion"][0]["role"] == "assistant" for r in train)


def test_general_count_math():
    assert general_count(300, 0.25) == 100
    assert general_count(100, 0.0) == 0
    assert general_count(7, 0.5) == 7
    with pytest.raises(ValueError):
        general_count(10, 1.0)


def test_general_ratio_zero_needs_no_pool(tmp_path):
    out = build_processed(smoke_cfg(tmp_path, general_ratio=0.0, general_val_n=0, general_pool=str(tmp_path / "missing.jsonl")), tmp_path / "z")
    assert {r["kind"] for r in read_jsonl(out["train"])} == {"matura"} and read_jsonl(out["val_general"]) == []


def test_split_by_subject_stratified():
    items = [{"id": f"{s}-{i}", "subject": s} for s in ("a", "b", "c") for i in range(10 if s != "c" else 2)]
    train, val = split_by_subject(items, 0.1, random.Random(0))
    assert len(val) == 2 and len(train) + len(val) == len(items)
    assert Counter(it["subject"] for it in val) == {"a": 1, "b": 1}
    assert split_by_subject(items, 0.0, random.Random(0)) == (items, [])


def test_variant_mix_and_system_prompt_frequencies():
    items = fixture_items()
    mix = {"canonical": 0.6, "bare": 0.2, "payload_only": 0.1, "reason_then_answer": 0.1}
    data_cfg = {"variant_mix": mix, "system_prompt_prob": 0.3, "system_prompts": ["S1", "S2"], "shuffle_options": True, "renders_per_item": 50}
    rows, counts = render_train(items, data_cfg, random.Random(3))
    n = len(rows)
    assert n == 50 * len(items)
    freq = Counter(r["variant"] for r in rows)
    for variant, weight in mix.items():
        assert abs(freq[variant] / n - weight) < 0.03, (variant, freq[variant] / n)
    with_system = sum(r["prompt"][0]["role"] == "system" for r in rows)
    assert abs(with_system / n - 0.3) < 0.03
    letters = Counter(r["completion"][0]["content"].split()[-1] for r in rows if r["variant"] == "canonical" and "fixture-" in r["id"]
                      and by_id(items)[r["id"]]["type"] == "mc")
    assert all(0.18 < letters[x] / sum(letters.values()) < 0.32 for x in "ABCD")

    no_rationale = [{**items[0], "rationale": ""}]
    rows, counts = render_train(no_rationale, {"variant_mix": {"reason_then_answer": 1.0}, "renders_per_item": 3}, random.Random(0))
    assert {r["variant"] for r in rows} == {"canonical"} and counts["reason_fallback_canonical"] == 3
    with pytest.raises(ValueError):
        render_train(items, {"variant_mix": {"nonexistent": 1.0}}, random.Random(0))


def test_option_shuffle_keeps_full_points_and_meaning():
    rng = random.Random(11)
    for item in fixture_items():
        for _ in range(5):
            shuffled, _ = shuffle_options(item, rng)
            validate_item(copy.deepcopy(shuffled))
            for variant in VARIANTS:
                rec = grade(shuffled, render_target(shuffled, variant), variant)
                assert rec["points"] == rec["max_points"], (item["id"], variant)
            if item["type"] == "mc":
                assert shuffled["options"][shuffled["answer"]] == item["options"][item["answer"]]
            elif item["type"] == "multi":
                assert {shuffled["options"][a] for a in shuffled["answer"]} == {item["options"][a] for a in item["answer"]}
            elif item["type"] == "match":
                assert {k: shuffled["right"][v] for k, v in shuffled["answer"].items()} == {k: item["right"][v] for k, v in item["answer"].items()}
            else:
                assert shuffled is item


def test_option_shuffle_skips_order_dependent_options():
    item = validate_item({"id": "x", "subject": "wos", "type": "mc", "question": "Które zdanie jest prawdziwe?",
                          "options": {"A": "tylko jedno", "B": "tylko drugie", "C": "odpowiedzi A i B", "D": "żadne z powyższych"}, "answer": "C"})
    assert shuffle_options(item, random.Random(0)) == (item, False)

    def mc(question: str, options: dict) -> dict:
        return validate_item({"id": "y", "subject": "wos", "type": "mc", "question": question, "options": options, "answer": "C"})

    letter_references = [
        mc("Które zdanie jest prawdziwe?", {"A": "x", "B": "y", "C": "tylko A", "D": "w"}),
        mc("Które zdanie jest prawdziwe?", {"A": "x", "B": "y", "C": "zarówno A, jak i B", "D": "w"}),
        mc("Które zdanie jest prawdziwe?", {"A": "x", "B": "y", "C": "ani A, ani B", "D": "w"}),
        mc("Które zdanie jest prawdziwe?", {"A": "x", "B": "y", "C": "A–C", "D": "w"}),
        mc("Wybierz odpowiedź A, jeśli oba zdania są prawdziwe, lub B, jeśli tylko pierwsze.", {"A": "p", "B": "q", "C": "r", "D": "s"}),
    ]
    for it in letter_references:
        assert all(shuffle_options(it, random.Random(seed)) == (it, False) for seed in range(10)), it
    plain = mc("Która z poniższych witamin jest rozpuszczalna w wodzie w temperaturze 25°C?",
               {"A": "witamina A", "B": "witamina D", "C": "witamina C", "D": "witamina K"})
    shuffled = [shuffle_options(plain, random.Random(seed)) for seed in range(10)]
    assert any(changed for _, changed in shuffled), "vitamin / unit letters and 'poniższych' in a question do not block shuffling"
    assert all(new["options"][new["answer"]] == "witamina C" for new, _ in shuffled)


def test_split_keeps_source_articles_on_one_side():
    """Sibling items generated from one Wikipedia article must not straddle train/val (val loss would be optimistic)."""
    items = [{"id": f"a{a}-{i}", "subject": "historia", "source": f"wikipedia:Article {a}"} for a in range(2) for i in range(6)]
    items += [{"id": f"s{i}", "subject": "historia", "source": f"wikipedia:Single {i}"} for i in range(20)]
    for seed in range(20):
        train, val = split_by_subject(items, 0.2, random.Random(seed))
        assert val and not {it["source"] for it in train} & {it["source"] for it in val}, seed
        assert len(train) + len(val) == len(items) and len(val) >= round(len(items) * 0.2)
    lonely = [{"id": f"x{i}", "subject": "wos", "source": "wikipedia:Only"} for i in range(5)]
    assert split_by_subject(lonely, 0.5, random.Random(0)) == (lonely, []), "a subject keeps at least one train article"


def test_general_heldout_exclusion_uses_the_whole_heldout_file(tmp_path):
    """The gate scores up to eval.general_loss_max_items rows of the file, not only the first general_heldout_n."""
    cfg = smoke_cfg(tmp_path, general_heldout=str(GENERAL), general_val_n=0, general_ratio=0.25)
    out = build_processed(cfg, tmp_path / "all")
    heldout_prompts = {r["messages"][-2]["content"] for r in read_jsonl(GENERAL)}
    general = [r for r in read_jsonl(out["train"]) + read_jsonl(out["val_general"]) if r["kind"] == "general"]
    assert not {r["prompt"][-1]["content"] for r in general} & heldout_prompts and general == []
    assert out["stats"]["inputs"]["general_pool"]["excluded_heldout_prompts"] == len(heldout_prompts)
    assert len(read_jsonl(out["general_heldout"])) == 5


def test_general_completion_cap(tmp_path):
    """General answers are ~25x longer than matura targets and the loss is token-weighted: long ones are left out."""
    lengths = {r["id"]: len(r["messages"][-1]["content"]) for r in read_jsonl(GENERAL)}
    out = build_processed(smoke_cfg(tmp_path, general_max_completion_chars=60), tmp_path / "capped")
    general = [r for r in read_jsonl(out["train"]) + read_jsonl(out["val_general"]) if r["kind"] == "general"]
    assert general and all(lengths[r["id"]] <= 60 for r in general)
    pool = out["stats"]["inputs"]["general_pool"]
    assert pool["general_max_completion_chars"] == 60 and pool["excluded_long_completions"] == sum(
        n > 60 for i, n in lengths.items() if i not in {r["id"] for r in read_jsonl(GENERAL)[:5]})
    uncapped = build_processed(smoke_cfg(tmp_path, general_max_completion_chars=None), tmp_path / "uncapped")
    assert uncapped["stats"]["inputs"]["general_pool"]["excluded_long_completions"] == 0


def test_build_processed_checks_every_eval_set_and_dedup_freshness(tmp_path):
    items = fixture_items()
    new_eval = tmp_path / "new_eval.jsonl"
    write_jsonl(items[:3], new_eval)  # a config-added eval set whose items are in the synthetic data
    cfg = smoke_cfg(tmp_path)
    cfg["eval"]["sets"] = [*cfg["eval"]["sets"], {"path": str(new_eval)}]
    with pytest.raises(LeakageError, match="fixture-historia-01"):
        build_processed(cfg, tmp_path / "leak")

    synth_dir = tmp_path / "synth"
    synth_dir.mkdir()
    synthetic = synth_dir / "clean.jsonl"
    write_jsonl(items, synthetic)
    cfg = smoke_cfg(tmp_path, synthetic=str(synthetic))
    assert build_processed(cfg, tmp_path / "no-report")["stats"]["dedup_fresh"] is None  # no report: nothing to compare

    from train.dataset import PROTECTED_EVAL_FILES, file_sha256_16

    # ext_llmzszl_matura.jsonl and data/blocklist/*.jsonl hold CKE exam text: gitignored, present only after
    # scripts/fetch_external.py, so a fresh clone checks the files it has
    protected = [ROOT / rel for rel in PROTECTED_EVAL_FILES if (ROOT / rel).exists()] + sorted(
        (ROOT / "data" / "blocklist").glob("*.jsonl"))
    hashes = {str(p.relative_to(ROOT)): file_sha256_16(p) for p in protected}
    report = synth_dir / "dedup_report.json"
    report.write_text(json.dumps({"inputs": {"protected_files": hashes}}), encoding="utf-8")
    assert build_processed(cfg, tmp_path / "fresh")["stats"]["dedup_fresh"] is True

    report.write_text(json.dumps({"inputs": {"protected_files": {**hashes, "data/eval/heldout.jsonl": "0" * 16}}}), encoding="utf-8")
    with pytest.raises(LeakageError, match="changed=\\['data/eval/heldout.jsonl'\\]"):
        build_processed(cfg, tmp_path / "stale")
    stale = build_processed(cfg, tmp_path / "stale-allowed", allow_stale_dedup=True)["stats"]
    assert stale["dedup_fresh"] is False and stale["dedup_stale"]["changed"] == ["data/eval/heldout.jsonl"]

    if any("blocklist" in k for k in hashes):  # blocklist files exist (fetched); a fresh clone has none
        report.write_text(json.dumps({"inputs": {"protected_files": {k: v for k, v in hashes.items() if "blocklist" not in k}}}), encoding="utf-8")
        with pytest.raises(LeakageError, match="not_deduplicated=\\['data/blocklist/"):
            build_processed(cfg, tmp_path / "missing-blocklist")


def test_no_eval_items_in_training(tmp_path):
    heldout_item = read_jsonl(ROOT / "data/eval/heldout.jsonl")[0]
    check_no_eval_leak(fixture_items())  # fixture is clean
    with pytest.raises(LeakageError):
        check_no_eval_leak(fixture_items() + [heldout_item])
    renamed = {**heldout_item, "id": "syn-renamed"}
    with pytest.raises(LeakageError):
        check_no_eval_leak([renamed])  # identical task text under another id
    leaky = tmp_path / "leaky.jsonl"
    write_jsonl(fixture_items() + [heldout_item], leaky)
    with pytest.raises(LeakageError):
        build_processed(smoke_cfg(tmp_path, synthetic=str(leaky)), tmp_path / "leak")

    out = build_processed(smoke_cfg(tmp_path), tmp_path / "clean")
    eval_ids = {r["id"] for p in ("heldout", "dev", "ext_llmzszl_matura") if (ROOT / f"data/eval/{p}.jsonl").exists()
                for r in read_jsonl(ROOT / f"data/eval/{p}.jsonl")}  # ext_llmzszl (CKE text) is gitignored
    assert not eval_ids & {r["id"] for r in read_jsonl(out["train"]) + read_jsonl(out["val_matura"])}


# ----------------------------------------------------------------------------- build_data end-to-end


def test_build_data_end_to_end(tmp_path, capsys):
    build_data = load_script("build_data")
    items = fixture_items()
    raw, verify, block = tmp_path / "raw", tmp_path / "verify", tmp_path / "blocklist"
    raw.mkdir(), verify.mkdir(), block.mkdir()
    first, second = items[:30], items[30:]
    paraphrase = {"id": "syn-para", "subject": "historia", "type": "short", "question": "Kiedy uchwalono Konstytucję 3 maja? Podaj rok.",
                  "answer": ["1791"], "rationale": "r", "source": "wikipedia:Konstytucja 3 maja"}
    with open(raw / "historia-01.jsonl", "w", encoding="utf-8") as fh:
        for it in first + [paraphrase]:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        fh.write("{not json\n")
        fh.write(json.dumps({"id": "syn-bad", "subject": "historia", "type": "mc", "question": "?", "options": {"A": "x"}, "answer": "A"}) + "\n")
    write_jsonl(second + [items[0]], raw / "wos-01.jsonl")  # items[0] repeats an id
    api_item = {"id": "syn-api-01", "subject": "biologia", "type": "short", "question": "Jak nazywa się barwnik nadający liściom zieloną barwę?",
                "answer": ["chlorofil"], "source": "wikipedia:Chlorofil"}
    write_jsonl([api_item], raw / "biologia-01.anthropic.jsonl")

    answers = [{"id": it["id"], "payload": render_target(it, "payload_only"), "confidence": "high"} for it in items]
    answers[1] = {**answers[1], "payload": "B" if items[1]["answer"] != "B" else "A"}  # disagree
    answers[2] = {**answers[2], "confidence": "low"}
    del answers[3]  # unverified
    answers.append({"id": "syn-para", "payload": "1791", "confidence": "high"})
    write_jsonl(answers[:20], verify / "historia-01.jsonl")
    write_jsonl(answers[20:], verify / "wos-01.jsonl")

    heldout = tmp_path / "heldout.jsonl"
    write_jsonl([{"id": "held-1", "subject": "historia", "type": "short", "question": "W którym roku uchwalono Konstytucję 3 maja?", "answer": ["1791"]}], heldout)
    write_jsonl([], tmp_path / "dev.jsonl")
    write_jsonl([], tmp_path / "ext.jsonl")
    write_jsonl([{"id": "blk-1", "source": "t", "subject": None, "question": items[5]["question"] + " A. x B. y"}], block / "src.jsonl")

    out, report_path = tmp_path / "clean.jsonl", tmp_path / "report.json"
    argv = ["--raw-dir", str(raw), "--verify-dir", str(verify), "--out", str(out), "--report", str(report_path),
            "--heldout", str(heldout), "--dev", str(tmp_path / "dev.jsonl"), "--ext", str(tmp_path / "ext.jsonl"),
            "--blocklist-dir", str(block), "--skip-verify-glob", "*.anthropic.jsonl"]
    assert build_data.main(argv, embed_fn=fake_embed) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    st = report["stages"]
    assert st["raw"] == len(items) + 1 + 2 + 1 + 1 and st["invalid"] == 2 and st["duplicate_id"] == 1
    assert (st["unverified"], st["disagree"], st["low_conf"], st["verify_skipped"]) == (1, 1, 1, 1)
    assert st["dup_vs_heldout"] == 1 and st["dup_vs_blocklist"] == 1 and st["dup_vs_dev"] == 0 and st["dup_vs_ext_llmzszl_matura"] == 0
    kept = load_items(out)
    expected = {it["id"] for it in items} - {items[1]["id"], items[2]["id"], items[3]["id"], items[5]["id"]} | {"syn-api-01"}
    assert {it["id"] for it in kept} == expected and st["kept"] == len(expected) and st["intra_dup"] == 0
    assert sum(report["kept_per_subject"].values()) == len(kept) and sum(report["mc_answer_letters"].values()) == sum(it["type"] == "mc" for it in kept)
    assert report["inputs"]["verify_skipped_files"] == ["biologia-01.anthropic.jsonl"]
    assert report["inputs"]["verify_coverage"]["historia-01.jsonl"] == {"items": 31, "answered": 30, "coverage": 0.9677}
    from train.dataset import file_sha256_16

    assert report["inputs"]["protected_files"] == {str(p): file_sha256_16(p) for p in (heldout, tmp_path / "dev.jsonl", tmp_path / "ext.jsonl",
                                                                                       block / "src.jsonl")}
    assert "raw" in capsys.readouterr().out

    out.unlink()
    assert build_data.main(argv + ["--dry-run"], embed_fn=fake_embed) == 0 and not out.exists()
    assert build_data.main(argv[:-2] + ["--blocklist-dir", str(tmp_path / "nope")], embed_fn=fake_embed) == 1

    # a raw shard whose verify file is missing (verify workflow unfinished) fails instead of silently losing the shard
    write_jsonl([{**it, "id": f"late-{it['id']}"} for it in items[:10]], raw / "chemia-01.jsonl")
    assert build_data.main(argv, embed_fn=fake_embed) == 1 and not out.exists()
    assert "chemia-01.jsonl 0/10" in capsys.readouterr().err
    assert build_data.main(argv + ["--allow-missing-verify"], embed_fn=fake_embed) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["inputs"]["verify_coverage"]["chemia-01.jsonl"]["coverage"] == 0.0 and "verify coverage < 100%" in capsys.readouterr().out


# ----------------------------------------------------------------------------- generate_synthetic (mocked HTTP)


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


GEN_TEXT = """Oto zadania:
```jsonl
{"id": "zly-id", "subject": "historia", "type": "mc", "question": "W którym roku odbył się chrzest Polski?", "options": {"A": "966", "B": "1000", "C": "1025", "D": "1410"}, "answer": "A", "rationale": "966."},
{"id": "syn-historia-01-1-anthropic-02", "subject": "wos", "type": "short", "question": "Kto przyjął chrzest w 966 roku?", "answer": ["Mieszko I"]}
{"id": "x", "subject": "historia", "type": "mc", "question": "Zepsute", "options": {"A": "1"}, "answer": "A"}
{"broken json
```"""


def test_parse_and_normalize_generated_items():
    from train.generate_synthetic import normalize_items, parse_jsonl

    objs, errors = parse_jsonl(GEN_TEXT)
    assert len(objs) == 3 and len(errors) == 1
    article = {"title": "Chrzest Polski", "subject": "historia"}
    items, errors, fixes = normalize_items(objs, article=article, id_prefix="syn-historia-01-1-anthropic")
    assert [it["id"] for it in items] == ["syn-historia-01-1-anthropic-01", "syn-historia-01-1-anthropic-02"]
    assert all(it["subject"] == "historia" and it["source"] == "wikipedia:Chrzest Polski" for it in items)
    assert fixes == {"id": 1, "subject": 1, "source": 2} and len(errors) == 1
    arr, _ = parse_jsonl('[{"a": 1}, {"b": 2}]')
    assert arr == [{"a": 1}, {"b": 2}]


def test_anthropic_backend_retries_and_request_shape(monkeypatch):
    from train import generate_synthetic as gen

    calls: list[dict] = []
    responses = [FakeResponse(529, {"error": "overloaded"}),
                 FakeResponse(200, {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})]
    monkeypatch.setattr(gen.requests, "post", lambda url, headers, json, timeout: calls.append({"url": url, "headers": headers, "body": json}) or responses.pop(0))
    cfg = gen.BackendConfig(name="anthropic", model="claude-opus-5", api_key="k", retries=2)
    assert gen.anthropic_complete(cfg, "prompt", sleep=lambda s: None) == "ab"
    assert len(calls) == 2 and calls[0]["url"] == gen.ANTHROPIC_URL
    h, body = calls[1]["headers"], calls[1]["body"]
    assert h["x-api-key"] == "k" and h["anthropic-version"] == "2023-06-01" and h["anthropic-beta"] == gen.ANTHROPIC_FALLBACK_BETA
    assert body["model"] == "claude-opus-5" and body["fallbacks"] == "default" and "temperature" not in body
    assert body["messages"] == [{"role": "user", "content": "prompt"}]

    monkeypatch.setattr(gen.requests, "post", lambda *a, **k: FakeResponse(200, {"stop_reason": "refusal", "content": []}))
    with pytest.raises(gen.GenerationError):
        gen.anthropic_complete(cfg, "prompt", sleep=lambda s: None)
    posts: list[int] = []
    monkeypatch.setattr(gen.requests, "post", lambda *a, **k: posts.append(1) or FakeResponse(400, {"error": "bad"}))
    with pytest.raises(gen.GenerationError):
        gen.anthropic_complete(cfg, "prompt", sleep=lambda s: None)
    assert len(posts) == 1  # 4xx is not retried


def test_openai_backend_request_shape(monkeypatch):
    from train import generate_synthetic as gen

    seen: dict = {}

    def post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return FakeResponse(200, {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(gen.requests, "post", post)
    cfg = gen.BackendConfig(name="openai", model="speakleash/Bielik-11B-v2.6-Instruct", api_key="t", base_url="http://h:8000/v1/")
    assert gen.make_complete(cfg)("p") == "hello"
    assert seen["url"] == "http://h:8000/v1/chat/completions" and seen["headers"]["authorization"] == "Bearer t"
    assert seen["body"]["model"] == "speakleash/Bielik-11B-v2.6-Instruct" and seen["body"]["temperature"] == 0.7


def test_generate_run_writes_shard_and_skips_existing(tmp_path):
    from train import generate_synthetic as gen

    articles = tmp_path / "articles.jsonl"
    write_jsonl([{"title": "Chrzest Polski", "subject": "historia", "url": "u", "text": "Mieszko I przyjął chrzest w 966 roku. " * 20},
                 {"title": "Unia lubelska", "subject": "historia", "url": "u", "text": "Unia lubelska 1569. " * 20}], articles)
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        if "Unia lubelska" in prompt:
            raise gen.GenerationError("refusal")
        return GEN_TEXT

    backend = gen.BackendConfig(name="anthropic", model="claude-opus-5", api_key="k")
    reports = gen.run(backend, shards=["historia"], out_dir=tmp_path / "raw", articles_path=articles, complete=complete)
    out = tmp_path / "raw" / "historia-01.anthropic.jsonl"
    items = load_items(out)
    assert len(items) == 2 and all(it["id"].startswith("syn-historia-01-1-anthropic-") for it in items)
    assert "Identyfikatory: syn-historia-01-1-anthropic-01" in prompts[0]
    rep = reports["historia-01"]
    assert rep["items"] == 2 and len(rep["failed_articles"]) == 1 and rep["articles"] == 2
    again = gen.run(backend, shards=["historia-01"], out_dir=tmp_path / "raw", articles_path=articles, complete=complete)
    assert "skipped" in again["historia-01"] and len(prompts) == 2
    with pytest.raises(ValueError):
        gen.select_shards({"historia-01": []}, ["nope-01"], None)


def test_generate_script_dry_run(capsys):
    script = load_script("generate_synthetic")
    assert script.main(["--dry-run", "--shards", "historia-01"]) == 0
    assert "Identyfikatory: syn-historia-01-1-anthropic-01" in capsys.readouterr().out


# ----------------------------------------------------------------------------- slow: real bge-m3


C1 = "W którym roku uchwalono Konstytucję 3 maja?"
C1_OPTIONS = {"A": "1772", "B": "1791", "C": "1795", "D": "1807"}
MICK = "Który z wymienionych utworów napisał Adam Mickiewicz?"
MICK_OPTIONS = {"A": "Lalka", "B": "Pan Tadeusz", "C": "Kordian", "D": "Wesele"}
PHOTO = "Proces, w którym rośliny wytwarzają glukozę i tlen z dwutlenku węgla i wody przy udziale światła."
# (candidate question, candidate options or None, candidate answer, protected question, protected answer text)
RESEARCH_PARAPHRASES = [
    (C1, None, "1791", "Kiedy uchwalono Konstytucję 3 Maja? Podaj rok.", "1791"),
    ("Kto jest autorem powieści „Lalka”?", None, "Bolesław Prus", "Podaj nazwisko pisarza, który napisał „Lalkę”.", "Bolesław Prus"),
    ("Oblicz pole koła o promieniu 5 cm.", None, "25π cm²", "Ile wynosi pole powierzchni koła, którego promień ma długość 5 cm?", "25π cm²"),
    ("Jaki gaz stanowi największą część powietrza atmosferycznego?", None, "azot", "Który gaz jest głównym składnikiem powietrza?", "azot"),
    ("Wyjaśnij, czym jest fotosynteza.", None, PHOTO, "Na czym polega proces fotosyntezy? Wyjaśnij.", PHOTO),
    ("Podaj datę bitwy pod Grunwaldem.", None, "15 lipca 1410", "Kiedy odbyła się bitwa pod Grunwaldem?", "15 lipca 1410"),
    ("Rozwiąż równanie 2x + 6 = 14.", None, "x = 4", "Znajdź x, jeśli 2x + 6 = 14.", "x = 4"),
    (C1, C1_OPTIONS, "B", "Kiedy uchwalono Konstytucję 3 Maja? Wybierz poprawną odpowiedź: A) 1791 B) 1772 C) 1807 D) 1795", "1791"),
    (C1, C1_OPTIONS, "B", C1, "1791"),
    (MICK, MICK_OPTIONS, "B", "Wskaż utwór, którego autorem jest Adam Mickiewicz: A) Pan Tadeusz B) Wesele C) Lalka D) Kordian", "Pan Tadeusz"),
]
RESEARCH_SAME_TOPIC = [
    (C1, None, "1791", "Który organ uchwalił Konstytucję 3 maja?", "Sejm Czteroletni"),
    ("Kto jest autorem powieści „Lalka”?", None, "Bolesław Prus", "W którym mieście rozgrywa się akcja „Lalki”?", "Warszawa"),
    ("Jaki gaz stanowi największą część powietrza atmosferycznego?", None, "azot", "Jaki gaz wydychają ludzie podczas oddychania?", "dwutlenek węgla"),
    ("Wyjaśnij, czym jest fotosynteza.", None, PHOTO, "W których organellach komórkowych zachodzi fotosynteza?", "w chloroplastach"),
    ("Podaj datę bitwy pod Grunwaldem.", None, "15 lipca 1410", "Kto dowodził wojskami polsko-litewskimi w bitwie pod Grunwaldem?", "Władysław II Jagiełło"),
    (C1, C1_OPTIONS, "B", "Który organ uchwalił Konstytucję 3 maja? A. Sejm Czteroletni B. Sejm Niemy C. Rada Nieustająca D. Senat", "Sejm Czteroletni"),
]


@pytest.mark.slow
def test_bge_m3_flags_paraphrases_not_same_topic():
    """Research pairs (docs/research/data_sources.md, Task C) through the full rule: paraphrases dropped, same-topic kept."""
    from train.dedup import SentenceTransformerEmbedder

    def is_dropped(i: int, pair: tuple, embed) -> tuple[bool, list]:
        question, options, answer, prot_question, prot_answer = pair
        if options:
            cand = {"id": f"c{i}", "subject": "historia", "type": "mc", "question": question, "options": options, "answer": answer}
        else:
            cand = {"id": f"c{i}", "subject": "historia", "type": "short", "question": question, "answer": [answer]}
        prot = {"blocklist": [{"id": f"p{i}", "source": "research", "subject": None, "question": prot_question, "answer_text": prot_answer}]}
        kept, report = dedup_items([validate_item(cand)], prot, DedupConfig(cache_dir=None), embed_fn=embed)
        return not kept, report["examples"]["dup_vs_blocklist"]

    embed = SentenceTransformerEmbedder("BAAI/bge-m3")
    try:
        para = [is_dropped(i, pair, embed) for i, pair in enumerate(RESEARCH_PARAPHRASES)]
        same = [is_dropped(i, pair, embed) for i, pair in enumerate(RESEARCH_SAME_TOPIC)]
    finally:
        embed.close()
    assert all(dropped for dropped, _ in para), [ex for dropped, ex in para if not dropped]
    assert not any(dropped for dropped, _ in same), [ex for dropped, ex in same if dropped]
