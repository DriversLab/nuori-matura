"""Exam harness: prompts, postprocess, answers.json I/O + validator, client, run_exam end to end.

Fast: no model, no network (the HTTP path runs against an in-process fake or a 127.0.0.1 server).
Fixture tests/fixtures/history_exam is a synthetic package written for these tests (not CKE material).
"""
import hashlib
import http.server
import importlib.util
import json
import re
import shutil
import sys
import threading
import types
from pathlib import Path

import pytest
import requests

from harness import exam_io, postprocess, prompts
from harness.client import ChatClient, ChatError, chat_url, run_parallel
from harness.postprocess import clean, clean_answer, closed_syntax_ok, essay_word_count
from harness.prompts import ORGANIZER_SYSTEM_PROMPT, build_messages, format_spec, item_kind, max_new_tokens

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "history_exam"
EXAM_ID = "test-history-fixture-v1"
IDS = ["1", "2.1", "2.2", "3", "4", "5"]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_harness_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def items():
    _, its = exam_io.load_exam(FIX)
    return {it["id"]: it for it in its}


@pytest.fixture
def exam_dir(tmp_path):
    dst = tmp_path / "exam"
    shutil.copytree(FIX, dst)
    return dst


def _write(tmp_path, payload, name="answers.json", raw: bytes | None = None):
    p = tmp_path / name
    p.write_bytes(raw if raw is not None else json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return p


def _good_payload():
    return {"exam_id": EXAM_ID, "answers": [{"id": i, "answer": ""} for i in IDS]}


# ---- prompts ---------------------------------------------------------------------------------------------------------

def test_organizer_system_prompt_is_verbatim():
    assert ORGANIZER_SYSTEM_PROMPT == (
        "Rozwiąż zadanie z historii po polsku. Otrzymujesz tekst źródeł, a obrazy zastąpiono opisami. Wykorzystaj "
        "źródła i własną wiedzę zgodnie z poleceniem. Udziel tylko odpowiedzi na podane zadanie. Nie dopisuj innych "
        "zadań. Nie masz dostępu do narzędzi ani internetu.")


def test_item_kinds(items):
    assert {i: item_kind(it) for i, it in items.items()} == {
        "1": "open", "2.1": "closed_tf", "2.2": "closed_match", "3": "closed_single", "4": "closed_multi_part",
        "5": "essay"}


def test_format_spec_row_counts_come_from_the_question(items):
    tf = format_spec(items["2.1"])  # answer_format example has 3 rows, the question 4 statements
    assert tf.keys == ["1", "2", "3", "4"]
    match = format_spec(items["2.2"])  # example has A, B; the question rows A, B, C
    assert match.keys == ["A", "B", "C"]
    assert match.rows["B"].startswith("Grupa mieszkańców")
    single = format_spec(items["3"])
    assert single.allowed[""] == ["A", "B", "C", "D"] and single.options[""]["B"] == "republika miejska."
    multi = format_spec(items["4"])
    assert multi.keys == ["1", "2"] and multi.options["2"]["B"] == "ceł."
    assert sorted(format_spec(items["5"]).topics) == [1, 2, 3]


def test_format_spec_falls_back_to_answer_format_rows():
    item = {"id": "9", "question": "Oceń prawdziwość stwierdzeń zawartych w źródle.", "source_text": "",
            "answer_format": "1: P\n2: F\n3: P"}
    assert format_spec(item).keys == ["1", "2", "3"]


def test_max_new_tokens(items):
    assert max_new_tokens(items["5"]) == 4096
    assert {max_new_tokens(items[i]) for i in IDS if i != "5"} == {2048}


def test_organizer_layout_matches_benchmark_protocol(items):
    msgs = build_messages(items["3"])
    assert msgs[0] == {"role": "system", "content": ORGANIZER_SYSTEM_PROMPT}
    it = items["3"]
    assert msgs[1] == {"role": "user",
                       "content": f"Zadanie 3 (1 pkt)\n\n{it['source_text']}\n\n{it['question']}"}
    # empty source text keeps the benchmark's blank block (their essay prompt has four newlines)
    assert build_messages(items["5"])[1]["content"].startswith("Zadanie 5 (15 pkt)\n\n\n\nZadanie zawiera")
    assert "Odpowiedz wyłącznie" not in build_messages(items["2.1"])[1]["content"]  # no format line in organizer style


def test_image_markers_replaced_with_descriptions(items):
    descs = {"images/Z01.png": "Rysunek domów z drewna.", "images/Z02.png": "Brązowy grot włóczni."}
    user = build_messages(items["1"], descs)[1]["content"]
    assert "[Obraz:" not in user
    assert "Rysunek poglądowy osady (opracowanie testowe)\n[Opis źródła Z01]\nRysunek domów z drewna.\n\n" in user
    # an image without a marker in source_text is appended after the source text
    src_end = user.index("\n\nRozstrzygnij")
    assert user[:src_end].endswith("[Opis źródła Z02]\nBrązowy grot włóczni.")
    # cache-shaped values and basename keys work too
    user2 = build_messages(items["1"], {"Z01.png": {"description": "Opis A", "sha256": "x"}})[1]["content"]
    assert "[Opis źródła Z01]\nOpis A" in user2 and "[Opis źródła Z02]" not in user2


def test_markers_without_description_are_left_as_is(items):
    user = build_messages(items["1"], {})[1]["content"]
    assert "[Obraz: images/Z01.png]" in user and "Opis źródła" not in user
    assert prompts.missing_descriptions([items["1"]], {"images/Z01.png": "x"}) == ["images/Z02.png"]


def test_formatted_style_adds_format_lines(items):
    tf = build_messages(items["2.1"], style="formatted")[1]["content"]
    assert tf.endswith("Odpowiedz wyłącznie w formacie (P – prawda, F – fałsz):\n"
                       "1: P albo F\n2: P albo F\n3: P albo F\n4: P albo F")
    assert build_messages(items["3"], style="formatted")[1]["content"].endswith(
        "Odpowiedz wyłącznie w formacie: jedna litera (A, B, C lub D).")
    match = build_messages(items["2.2"], style="formatted")[1]["content"]
    assert match.endswith("A: numer\nB: numer\nC: numer")
    assert "1: litera\n2: litera" in build_messages(items["4"], style="formatted")[1]["content"]
    essay = build_messages(items["5"], style="formatted")[1]["content"]
    assert "„Temat”" in essay and "300 wyrazów" in essay and "(1, 2 lub 3)" in essay
    op = build_messages(items["1"], style="formatted")[1]["content"]
    assert op.endswith("Użyj etykiet z polecenia: Rozstrzygnięcie:, Uzasadnienie:")
    with pytest.raises(ValueError):
        build_messages(items["1"], style="nope")


def test_no_system_prompt_and_prompt_hash(items):
    msgs = build_messages(items["3"], system_prompt=None)
    assert [m["role"] for m in msgs] == ["user"]
    assert prompts.prompt_sha256(msgs) == prompts.prompt_sha256(build_messages(items["3"], system_prompt=""))
    assert prompts.prompt_sha256(msgs) != prompts.prompt_sha256(build_messages(items["3"]))


# ---- postprocess -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("B", "B"),
    ("  (B).  ", "B"),
    ("<think>A? nie, raczej C</think>\nOdpowiedź: B", "B"),
    ("Odpowiedź na zadanie:\n\n**B. republika miejska.**\n\n**Uzasadnienie:** rajcy byli wybierani.", "B"),
    ("A. monarchia absolutna.\nB. republika miejska.\nC. teokracja.\nD. dyktatura wojskowa.\n\n"
     "Poprawna odpowiedź to B.", "B"),
    ("Odpowiedź A jest błędna, bo nie było króla. Prawidłowa odpowiedź: B", "B"),
    ("Ustrój opisany w tekście to republika miejska.", "B"),
    ("Najbardziej prawdopodobna odpowiedź:\n\\[\n\\boxed{B}\n\\]", "B"),
])
def test_single_choice_extraction(items, raw, expected):
    r = clean(items["3"], raw)
    assert r.answer == expected and r.closed_ok is True


def test_single_choice_unrecoverable_keeps_text(items):
    raw = "A. monarchia absolutna.\nB. republika miejska.\nTrudno powiedzieć."
    r = clean(items["3"], raw)
    assert r.closed_ok is False and r.answer and "Trudno powiedzieć." in r.answer


@pytest.mark.parametrize("raw", [
    "1: P\n2: F\n3: F\n4: P",
    "P, F, F, P",
    "Odpowiedź:\n1. P\n2. F\n3. F\n4. P",
    "1 – prawda\n2 – fałsz\n3 – fałsz\n4 – prawda",
    "Ocena prawdziwości stwierdzeń:\n\n"
    "1. **Autor tekstu testowego opisuje miasto portowe.**\n   **PRAWDA (P)**\n   - Fragment 1. mówi o porcie.\n\n"
    "2. **W tekście testowym wymieniono dwóch władców.**\n   **FAŁSZ (F)**\n   - Nie ma władców.\n\n"
    "3. **Tekst testowy powstał w XX wieku.**\n   - Fałsz (F)\n\n"
    "4. **W tekście testowym nie podano żadnej daty.**\n   - To stwierdzenie jest prawdziwe.\n\n"
    "Odpowiedź w formacie: \\[\\boxed{\\text{P, F, F, P}}\\]",
])
def test_true_false_extraction(items, raw):
    r = clean(items["2.1"], raw)
    assert r.answer == "1: P\n2: F\n3: F\n4: P" and r.closed_ok


def test_true_false_missing_row_keeps_text(items):
    r = clean(items["2.1"], "1: P\n2: F\n3: F")
    assert r.closed_ok is False and r.answer == "1: P\n2: F\n3: F"


@pytest.mark.parametrize("raw", [
    "A: 2\nB: 3\nC: 1",
    "A – 2, B – 3, C – 1",
    "A. fragment 2\nB. fragment 3\nC. fragment 1",
    "Rozwiązanie zadania:\n\n| Opis | Numer fragmentu |\n|---|---|\n"
    "| Zgromadzenie decydujące o sprawach miasta. | **2** |\n"
    "| Grupa mieszkańców pozbawiona praw politycznych. | **3** |\n"
    "| Miejsce przechowywania zboża w porcie. | **1** |",
])
def test_matching_extraction(items, raw):
    r = clean(items["2.2"], raw)
    assert r.answer == "A: 2\nB: 3\nC: 1" and r.closed_ok


@pytest.mark.parametrize("raw", [
    "1: C\n2: B",
    "1C 2B",
    "1. **Dokument testowy wystawił:**\n   - **A. burmistrz.**\n   - **B. biskup.**\n   - **C. król.**\n"
    "   - **D. wójt.**\n\n   **Odpowiedź:** C. król.\n\n"
    "2. **Dokument testowy dotyczył:**\n   - **A. podatków.**\n   - **B. ceł.**\n   - **C. sądów.**\n"
    "   - **D. wojska.**\n\n   **Odpowiedź:** B. ceł.",
    "1. C. król.\n2. B. ceł.",
])
def test_multi_part_extraction(items, raw):
    r = clean(items["4"], raw)
    assert r.answer == "1: C\n2: B" and r.closed_ok


def test_closed_syntax_ok(items):
    assert closed_syntax_ok(items["2.1"], "1: P\n2: F\n3: F\n4: P")
    assert not closed_syntax_ok(items["2.1"], "1: P\n2: F\n3: F")
    assert not closed_syntax_ok(items["2.1"], "1: X\n2: F\n3: F\n4: P")
    assert closed_syntax_ok(items["3"], "D") and not closed_syntax_ok(items["3"], "E")
    assert closed_syntax_ok(items["2.2"], "A: 1\nB: 2\nC: 3") and not closed_syntax_ok(items["2.2"], "A: x\nB: 2\nC: 3")
    assert closed_syntax_ok(items["1"], "cokolwiek")


def test_open_answer_cleaning(items):
    raw = ("<think>rozważam</think>**Odpowiedź na zadanie 1:**\n\n## Rozstrzygnięcie:\nepoka żelaza\n\n"
           "**Uzasadnienie:** widoczne są piece.\n\n\n\nMam nadzieję, że to pomoże!")
    assert clean_answer(items["1"], raw) == "Rozstrzygnięcie:\nepoka żelaza\n\nUzasadnienie: widoczne są piece."
    assert clean_answer(items["1"], "Odpowiedź: epoka żelaza") == "epoka żelaza"
    cut = clean_answer(items["1"], "Rozstrzygnięcie: epoka żelaza\n\nZadanie 2.1\n1: P\n2: F")
    assert cut == "Rozstrzygnięcie: epoka żelaza"
    assert clean_answer(items["1"], "Zadanie 1 (1 pkt)\nepoka żelaza") == "epoka żelaza"
    assert clean_answer(items["1"], "   ") == ""
    assert clean_answer(items["1"], "<think>tylko myśli</think>") == ""
    assert clean_answer(items["1"], "Odpowiedź:") == "Odpowiedź:"  # never empty when the model said something


def _essay_body(n_words: int, word: str = "miasto") -> str:
    return " ".join([word] * n_words)


def test_essay_topic_prefix_and_word_count(items):
    raw = ("**Odpowiedź na zadanie 5 (temat 3: kolej żelazna)**\n\n**Stanowisko:**\n" + _essay_body(320))
    r = clean(items["5"], raw)
    assert r.answer.startswith("Temat 3.\n\nStanowisko:\n")
    assert r.essay_topic == 3 and r.topic_source == "explicit"
    assert r.essay_words == 321 and essay_word_count(r.answer) == 321


def test_essay_topic_already_present_is_kept(items):
    raw = "Temat 2: Reformy oświatowe\n\n" + _essay_body(310)
    r = clean(items["5"], raw)
    assert r.answer.startswith("Temat 2: Reformy") and r.answer.count("Temat 2") == 1


def test_essay_topic_from_content_and_na_temat_ignored(items):
    body = ("Kolej żelazna przyspieszyła industrializację ziem polskich. Na temat 3 odcinków kolei pisano wiele. "
            "Budowa kolei i industrializacja ziem polskich w XIX wieku zmieniła gospodarkę. ") * 20
    r = clean(items["5"], body)
    assert r.essay_topic == 3 and r.topic_source == "content" and r.answer.startswith("Temat 3.\n\n")


def test_essay_topic_undetectable_gets_no_prefix(items):
    r = clean(items["5"], _essay_body(50, "tekst"))
    assert r.essay_topic is None and not r.answer.startswith("Temat")
    assert any("not detected" in n for n in r.notes) and any("< 300" in n for n in r.notes)


def test_essay_word_count():
    assert essay_word_count("Temat 1.\n\nAla ma kota, a XI–XII wiek był burzliwy (1025-1031).") == 9
    assert essay_word_count("Temat 1.\nAla", skip_topic_header=False) == 3
    assert essay_word_count("") == 0


# ---- answers.json I/O and the upload rules ---------------------------------------------------------------------------

def test_load_exam_and_template(items):
    meta, its = exam_io.load_exam(FIX)
    assert meta["exam_id"] == EXAM_ID and [it["id"] for it in its] == IDS
    assert exam_io.load_template(FIX) == (EXAM_ID, IDS)
    assert exam_io.load_exam(FIX / "exam.json")[0]["exam_id"] == EXAM_ID


def test_write_answers_emits_every_template_id(tmp_path, exam_dir):
    out = tmp_path / "sub" / "answers.json"
    payload = exam_io.write_answers(out, EXAM_ID, {"3": "B", "2.1": None, "4": 7, "99": "x"}, exam_dir=exam_dir)
    assert [a["id"] for a in payload["answers"]] == IDS
    got = {a["id"]: a["answer"] for a in payload["answers"]}
    assert got["3"] == "B" and got["2.1"] == "" and got["4"] == "7" and got["1"] == ""
    assert set(payload) == {"exam_id", "answers"}
    assert all(set(a) == {"id", "answer"} for a in payload["answers"])
    assert not out.read_bytes().startswith(b"\xef\xbb\xbf")
    assert exam_io.validate_answers(out, exam_dir) == []
    exam_io.write_answers(out, EXAM_ID, {"1": "Łódź – ąę"}, IDS)
    assert "Łódź – ąę" in out.read_text("utf-8")  # ensure_ascii=False, real UTF-8


def test_write_answers_cuts_overlong_answers(tmp_path, exam_dir):
    out = tmp_path / "answers.json"
    exam_io.write_answers(out, EXAM_ID, {"5": "słowo " * 30000}, IDS)
    assert exam_io.validate_answers(out, exam_dir) == []
    assert len(json.loads(out.read_text("utf-8"))["answers"][5]["answer"]) == exam_io.MAX_ANSWER_CHARS


def test_validator_accepts_good_file(tmp_path, exam_dir):
    assert exam_io.validate_answers(_write(tmp_path, _good_payload()), exam_dir) == []


def _mutated(mut):
    p = _good_payload()
    mut(p)
    return p


@pytest.mark.parametrize("mut,needle", [
    (lambda p: p.__setitem__("exam_id", "other-exam"), "expected 'test-history-fixture-v1'"),
    (lambda p: p.__setitem__("exam_id", 5), "exam_id must be a string"),
    (lambda p: p.pop("exam_id"), "missing exam_id"),
    (lambda p: p.__setitem__("team_key", "x"), "extra key(s) ['team_key']"),
    (lambda p: p.pop("answers"), "missing answers"),
    (lambda p: p.__setitem__("answers", {"1": ""}), "answers must be a list"),
    (lambda p: p["answers"][0].__setitem__("reasoning", "x"), "only id and answer are allowed"),
    (lambda p: p["answers"][1].__setitem__("id", 2.1), "id must be a string"),
    (lambda p: p["answers"][0].pop("answer"), "missing answer"),
    (lambda p: p["answers"][0].pop("id"), "missing id"),
    (lambda p: p["answers"][0].__setitem__("answer", None), "got null"),
    (lambda p: p["answers"][0].__setitem__("answer", 3), "answer must be a string"),
    (lambda p: p["answers"][0].__setitem__("answer", ["A"]), "answer must be a string"),
    (lambda p: p["answers"].append({"id": "3", "answer": "B"}), "duplicate id(s): ['3']"),
    (lambda p: p["answers"].pop(), "missing id(s): ['5']"),
    (lambda p: p["answers"].append({"id": "7", "answer": ""}), "unexpected id(s) not in the exam: ['7']"),
    (lambda p: p["answers"][5].__setitem__("answer", "x" * 100_001), "limit 100000"),
    (lambda p: p["answers"].__setitem__(0, "1"), "must be an object"),
])
def test_validator_rules(tmp_path, exam_dir, mut, needle):
    problems = exam_io.validate_answers(_write(tmp_path, _mutated(mut)), exam_dir)
    assert any(needle in pr for pr in problems), problems


def test_validator_counts_utf16_like_the_browser(tmp_path, exam_dir):
    p = _good_payload()
    p["answers"][5]["answer"] = "😀" * 50_001  # 50,001 code points, 100,002 UTF-16 units
    assert any("limit 100000" in pr for pr in exam_io.validate_answers(_write(tmp_path, p), exam_dir))


@pytest.mark.parametrize("raw,needle", [
    (b"", "file is empty"),
    (b"\xef\xbb\xbf" + json.dumps(_good_payload()).encode(), "BOM"),
    (b'{"exam_id": "test-history-fixture-v1", "answers": [\xff]}', "not valid UTF-8"),
    (b'{"exam_id": "test-history-fixture-v1", "answers": [', "invalid JSON"),
    (b'{"exam_id": "a", "exam_id": "test-history-fixture-v1", "answers": []}', "duplicate key"),
    (b'{"exam_id": "test-history-fixture-v1", "answers": [{"id": "1", "answer": NaN}]}', "non-standard"),
    (b"[]", "top level must be a JSON object"),
])
def test_validator_file_level_rules(tmp_path, exam_dir, raw, needle):
    problems = exam_io.validate_answers(_write(tmp_path, None, raw=raw), exam_dir)
    assert any(needle in pr for pr in problems), problems


def test_validator_size_and_extension(tmp_path, exam_dir):
    p = _good_payload()
    for a in p["answers"]:
        a["answer"] = "x" * 99_000
    p["answers"].extend({"id": f"x{i}", "answer": "y" * 99_000} for i in range(6))
    problems = exam_io.validate_answers(_write(tmp_path, p), exam_dir)
    assert any("limit is 1048576" in pr for pr in problems)
    txt = _write(tmp_path, _good_payload(), name="answers.txt")
    assert any("must end with .json" in pr for pr in exam_io.validate_answers(txt, exam_dir))
    assert any("does not exist" in pr for pr in exam_io.validate_answers(tmp_path / "nope.json", exam_dir))


def test_lint_warnings(tmp_path, exam_dir):
    p = _good_payload()
    ans = {"1": "<think>x</think> ok", "2.1": "P, F", "3": "B", "5": "Za krótko."}
    for a in p["answers"]:
        a["answer"] = ans.get(a["id"], "")
    warns = exam_io.lint_answers(_write(tmp_path, p), exam_dir)
    text = "\n".join(warns)
    assert "blank answer(s): ['2.2', '4']" in text
    assert "think markers" in text
    assert "2.1: closed answer does not follow" in text
    assert "5: essay has 2 words" in text and "5: essay does not start with the chosen topic" in text


def test_load_descriptions_checks_sha(tmp_path, items):
    cache = {
        "images/Z01.png": {"sha256": items["1"]["images"][0]["sha256"], "description": "Dobry opis",
                           "model": "vlm", "ocr": "NAPIS"},
        "images/Z02.png": {"sha256": "0" * 64, "description": "Opis innego obrazu", "model": "vlm", "ocr": None},
    }
    path = tmp_path / "desc.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    probs: list[str] = []
    got = exam_io.load_descriptions(path, [items["1"]], problems=probs)
    assert got == {"images/Z01.png": "Dobry opis"} and any("sha256 mismatch" in p for p in probs)
    with_ocr = exam_io.load_descriptions(path, [items["1"]], include_ocr=True)
    assert with_ocr["images/Z01.png"].endswith("Tekst widoczny na obrazie: NAPIS")
    assert exam_io.load_descriptions(path) == {"images/Z01.png": "Dobry opis", "images/Z02.png": "Opis innego obrazu"}


def test_validate_answers_cli(tmp_path, exam_dir, capsys):
    mod = _load_script("validate_answers")
    good = _write(tmp_path, _good_payload())
    assert mod.main([str(good), "--exam-dir", str(exam_dir)]) == 0
    assert mod.main([str(good), "--exam-dir", str(exam_dir), "--strict"]) == 1  # blank answers are warnings
    bad = _write(tmp_path, {"exam_id": EXAM_ID, "answers": []}, name="bad.json")
    assert mod.main([str(bad), "--exam-dir", str(exam_dir)]) == 1
    assert "INVALID" in capsys.readouterr().out


# ---- client ----------------------------------------------------------------------------------------------------------

class _Resp:
    def __init__(self, status: int, body: dict | None = None, text: str = ""):
        self.status_code = status
        self._body = body
        self.text = text or json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _ok(text: str, finish: str = "stop", ptok: int = 100, ctok: int = 20) -> _Resp:
    return _Resp(200, {"model": "fake", "choices": [{"message": {"role": "assistant", "content": text},
                                                     "finish_reason": finish}],
                       "usage": {"prompt_tokens": ptok, "completion_tokens": ctok}})


def test_chat_url():
    assert chat_url("http://h:8080") == "http://h:8080/v1/chat/completions"
    assert chat_url("http://h:11434/v1/") == "http://h:11434/v1/chat/completions"


def test_client_payload_and_lora(monkeypatch):
    sent = []

    def fake_post(self, url, json=None, timeout=None):
        sent.append((url, json, timeout))
        return _ok("B")

    monkeypatch.setattr(requests.Session, "post", fake_post)
    msgs = [{"role": "user", "content": "x"}]
    r = ChatClient("http://127.0.0.1:9", "bielik", timeout=5, lora_scale=0.0).chat(msgs, 2048)
    assert r.text == "B" and r.finish_reason == "stop" and r.usage["completion_tokens"] == 20
    url, body, timeout = sent[0]
    assert url == "http://127.0.0.1:9/v1/chat/completions" and timeout == 5
    assert body["messages"] == msgs and body["model"] == "bielik"
    assert (body["temperature"], body["top_p"], body["seed"], body["max_tokens"]) == (0.0, 1.0, 42, 2048)
    assert (body["top_k"], body["repeat_penalty"], body["cache_prompt"]) == (1, 1.0, False)
    assert body["lora"] == [{"id": 0, "scale": 0.0}] and body["stream"] is False
    ChatClient("http://127.0.0.1:9", None, llama_extras=False).chat(msgs, 10)
    body2 = sent[1][1]
    assert "lora" not in body2 and "model" not in body2 and "repeat_penalty" not in body2


def test_client_retries_then_fails(monkeypatch):
    calls = {"n": 0}

    def flaky(self, url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("down")
        if calls["n"] == 2:
            return _Resp(503, text="loading model")
        return _ok("ok")

    monkeypatch.setattr(requests.Session, "post", flaky)
    r = ChatClient("http://x", retries=3, backoff_s=0).chat([], 5)
    assert r.text == "ok" and r.attempts == 3

    monkeypatch.setattr(requests.Session, "post", lambda self, url, json=None, timeout=None: _Resp(400, text="bad"))
    with pytest.raises(ChatError, match="HTTP 400"):
        ChatClient("http://x", retries=3, backoff_s=0).chat([], 5)
    monkeypatch.setattr(requests.Session, "post", lambda self, url, json=None, timeout=None: _Resp(500, text="boom"))
    with pytest.raises(ChatError, match="giving up after 2 attempts"):
        ChatClient("http://x", retries=1, backoff_s=0).chat([], 5)


def test_client_against_local_http_server():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = json.dumps({"choices": [{"message": {"content": "Odpowiedź: C"}, "finish_reason": "length"}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as e:  # sandbox without loopback sockets
        pytest.skip(f"cannot bind 127.0.0.1: {e}")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        r = ChatClient(f"http://127.0.0.1:{srv.server_address[1]}", lora_scale=1.0, timeout=10).chat(
            [{"role": "user", "content": "q"}], 7)
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.text == "Odpowiedź: C" and r.truncated
    assert seen[0][0] == "/v1/chat/completions" and seen[0][1]["lora"] == [{"id": 0, "scale": 1.0}]


def test_run_parallel_isolates_failures():
    def fn(x):
        if x == 3:
            raise RuntimeError("bad")
        return x * 2

    got = {it: (res, err) for it, res, err in run_parallel(fn, [1, 2, 3, 4], parallel=3)}
    assert got[1] == (2, None) and got[4] == (8, None) and isinstance(got[3][1], RuntimeError)
    assert [r for _, r, _ in run_parallel(fn, [1, 2], parallel=1)] == [2, 4]


# ---- run_exam end to end ---------------------------------------------------------------------------------------------

MESSY = {
    "1": "**Odpowiedź na zadanie 1:**\n\nRozstrzygnięcie: epoka żelaza\nUzasadnienie: na rysunku widać piec.",
    "2.1": "Ocena:\n1. **PRAWDA (P)**\n2. FAŁSZ\n3. F\n4. P\n\nMam nadzieję, że pomogłem.",
    "2.2": "A – fragment 2\nB – fragment 3\nC – fragment 1",
    "3": "Poprawna odpowiedź to **B. republika miejska.**",
    "4": "1. C\n2. B",
}


def _fake_server(monkeypatch, essay: str, continuation: str = ""):
    calls = []

    def fake_post(self, url, json=None, timeout=None):
        calls.append(json)
        user = [m for m in json["messages"] if m["role"] == "user"][0]["content"]
        item_id = user.split(" ", 2)[1]
        if len(json["messages"]) > 2 and json["messages"][-1]["role"] == "user" and json["messages"][-2]["role"] == "assistant":
            return _ok(continuation, ctok=400)
        text = essay if item_id == "5" else MESSY[item_id]
        return _ok(text, ctok=len(text.split()))

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _Resp(200, {"data": [{"id": "fake-bielik"}]}))
    return calls


def test_run_exam_end_to_end(monkeypatch, tmp_path, exam_dir, capsys):
    essay = "Wybieram temat 2.\n\n" + _essay_body(330, "reforma")
    calls = _fake_server(monkeypatch, essay)
    desc = tmp_path / "desc.json"
    desc.write_text(json.dumps({"images/Z01.png": {"sha256": None, "description": "Domy z drewna.", "model": "t",
                                                   "ocr": None}}), encoding="utf-8")
    run_exam = _load_script("run_exam")
    out = tmp_path / "run" / "answers.json"
    rc = run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--label", "tuned",
                        "--lora-scale", "1", "--descriptions", str(desc), "--parallel", "3", "--out", str(out)])
    assert rc == 0, capsys.readouterr()
    got = {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}
    assert got["1"] == "Rozstrzygnięcie: epoka żelaza\nUzasadnienie: na rysunku widać piec."
    assert got["2.1"] == "1: P\n2: F\n3: F\n4: P"
    assert got["2.2"] == "A: 2\nB: 3\nC: 1"
    assert got["3"] == "B" and got["4"] == "1: C\n2: B"
    assert got["5"].startswith("Temat 2.\n\n")
    assert exam_io.validate_answers(out, exam_dir) == []
    assert len(calls) == 6
    assert all(c["lora"] == [{"id": 0, "scale": 1.0}] and c["model"] == "fake-bielik" and c["seed"] == 42
               for c in calls)
    assert {c["max_tokens"] for c in calls} == {2048, 4096}
    item1 = next(c for c in calls if "Zadanie 1 (1 pkt)" in c["messages"][1]["content"])
    assert "[Opis źródła Z01]\nDomy z drewna." in item1["messages"][1]["content"]
    assert item1["messages"][0]["content"] == ORGANIZER_SYSTEM_PROMPT
    log = [json.loads(ln) for ln in (out.parent / "run_log.jsonl").read_text("utf-8").splitlines()]
    assert [r["type"] for r in log].count("item") == 6 and log[0]["type"] == "run" and log[-1]["type"] == "summary"
    rec = next(r for r in log if r.get("id") == "2.1")
    assert rec["raw"] == MESSY["2.1"] and rec["answer"] == "1: P\n2: F\n3: F\n4: P" and rec["closed_ok"] is True
    assert len(rec["prompt_sha256"]) == 64 and rec["label"] == "tuned" and rec["usage"]["completion_tokens"] > 0
    assert log[-1]["essays"]["5"]["topic"] == 2 and log[-1]["validation_problems"] == []
    assert log[-1]["missing_descriptions"] == ["images/Z02.png"]
    assert "validation: OK" in capsys.readouterr().out


def test_run_exam_essay_retry_and_resume(monkeypatch, tmp_path, exam_dir):
    calls = _fake_server(monkeypatch, "Temat 1.\n\n" + _essay_body(120, "port"),
                         continuation="Oto dalsza część:\n" + _essay_body(250, "handel"))
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    args = ["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out), "--parallel", "1"]
    assert run_exam.main(args + ["--essay-retry"]) == 0
    essay = {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}["5"]
    assert essay.startswith("Temat 1.\n\n") and "Oto dalsza" not in essay and essay_word_count(essay) == 370
    assert len(calls) == 7 and "lora" not in calls[0]
    n = len(calls)
    assert run_exam.main(args + ["--resume"]) == 0  # everything reused from the log
    assert len(calls) == n
    assert {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}["5"] == essay


def test_run_exam_only_keeps_other_answers(monkeypatch, tmp_path, exam_dir):
    _fake_server(monkeypatch, "Temat 3.\n\n" + _essay_body(300))
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    exam_io.write_answers(out, EXAM_ID, {"1": "stara odpowiedź"}, IDS)
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out),
                          "--only", "3,2.2"]) == 0
    got = {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}
    assert got["1"] == "stara odpowiedź" and got["3"] == "B" and got["2.2"] == "A: 2\nB: 3\nC: 1" and got["4"] == ""
    assert run_exam.main(["--exam-dir", str(exam_dir), "--out", str(out), "--only", "42"]) == 1


def test_run_exam_dry_run_never_touches_the_network(monkeypatch, tmp_path, exam_dir, capsys):
    def boom(*a, **k):
        raise AssertionError("network used in dry run")

    monkeypatch.setattr(requests.Session, "post", boom)
    monkeypatch.setattr(requests, "get", boom)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--dry-run", "--out", str(out)]) == 0
    assert exam_io.validate_answers(out, exam_dir) == []
    got = {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}
    assert got["2.1"] == "1: P\n2: P\n3: P\n4: P" and got["3"] == "A" and got["5"].startswith("Temat 1.")
    assert "DRY RUN" in capsys.readouterr().out


def test_run_exam_reports_unreachable_server(monkeypatch, tmp_path, exam_dir):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", down)
    run_exam = _load_script("run_exam")
    assert run_exam.main(["--exam-dir", str(exam_dir), "--out", str(tmp_path / "a.json")]) == 3


def test_run_exam_failed_item_is_blank_and_exit_1(monkeypatch, tmp_path, exam_dir):
    _fake_server(monkeypatch, "Temat 1.\n\n" + _essay_body(300))
    orig = requests.Session.post

    def partly_down(self, url, json=None, timeout=None):
        if "Zadanie 3 (" in json["messages"][-1]["content"]:
            return _Resp(400, text="context too long")
        return orig(self, url, json=json, timeout=timeout)

    monkeypatch.setattr(requests.Session, "post", partly_down)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out)]) == 1
    got = {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}
    assert got["3"] == "" and got["4"] == "1: C\n2: B" and exam_io.validate_answers(out, exam_dir) == []


def test_run_exam_picks_up_default_descriptions(capsys, exam_dir, items):
    (exam_dir / "descriptions.json").write_text(json.dumps({"images/Z01.png": {
        "sha256": items["1"]["images"][0]["sha256"], "description": "Opis z cache.", "model": "vlm", "ocr": None}}),
        encoding="utf-8")
    run_exam = _load_script("run_exam")
    assert run_exam.main(["--exam-dir", str(exam_dir), "--show-prompts", "--only", "1"]) == 0
    assert "[Opis źródła Z01]\nOpis z cache." in capsys.readouterr().out
    assert run_exam.main(["--exam-dir", str(exam_dir), "--show-prompts", "--only", "1", "--descriptions", "none"]) == 0
    assert "[Obraz: images/Z01.png]" in capsys.readouterr().out


def test_run_exam_finds_the_describe_images_default_cache(monkeypatch, capsys, tmp_path, exam_dir, items):
    """describe_images.py writes runs/desc/<exam folder>.json by default; run_exam must pick it up without
    --descriptions (it wins over <exam-dir>/descriptions.json) and warn about images still undescribed."""
    run_exam = _load_script("run_exam")
    monkeypatch.setattr(run_exam, "ROOT", tmp_path / "repo")
    sha = items["1"]["images"][0]["sha256"]
    cache = tmp_path / "repo" / "runs" / "desc" / f"{exam_dir.name}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"images/Z01.png": {"sha256": sha, "description": "Opis z runs/desc.", "model": "vlm",
                                                    "ocr": None}}), encoding="utf-8")
    (exam_dir / "descriptions.json").write_text(json.dumps({"images/Z01.png": {
        "sha256": sha, "description": "Opis obok exam.json.", "model": "vlm", "ocr": None}}), encoding="utf-8")
    assert run_exam.main(["--exam-dir", str(exam_dir), "--show-prompts", "--only", "1"]) == 0
    out, err = capsys.readouterr()
    assert "[Opis źródła Z01]\nOpis z runs/desc." in out and "Opis obok exam.json." not in out
    assert "WARNING: 1 image(s) have no description" in err and "images/Z02.png" in err
    assert run_exam.main(["--exam-dir", str(exam_dir), "--show-prompts", "--only", "1", "--descriptions", "none"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_run_log_records_the_descriptions_fingerprint(monkeypatch, tmp_path, exam_dir, items):
    _fake_server(monkeypatch, "Temat 1.\n\n" + _essay_body(300))
    desc = tmp_path / "desc.json"
    desc.write_text(json.dumps({"images/Z01.png": {"sha256": items["1"]["images"][0]["sha256"],
                                                   "description": "Domy.", "model": "t", "ocr": None}}), "utf-8")
    run_exam = _load_script("run_exam")
    shas = []
    for label in ("base", "tuned"):
        out = tmp_path / label / "answers.json"
        assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--label", label,
                              "--descriptions", str(desc), "--out", str(out)]) == 0
        log = [json.loads(ln) for ln in (out.parent / "run_log.jsonl").read_text("utf-8").splitlines()]
        assert log[0]["descriptions_used"] == 1 and log[0]["descriptions_sha256"] == log[-1]["descriptions_sha256"]
        shas.append(log[0]["descriptions_sha256"])
    assert shas[0] == shas[1] and len(shas[0]) == 64


def test_show_prompts(capsys, exam_dir):
    run_exam = _load_script("run_exam")
    assert run_exam.main(["--exam-dir", str(exam_dir), "--show-prompts", "--only", "3", "--style", "formatted"]) == 0
    out = capsys.readouterr().out
    assert "===== 3 [closed_single]" in out and "jedna litera (A, B, C lub D)" in out


# ---- tuned-run extras: rag style, essay plan, closed voting, label repair (all OFF by default) ----------------------

# sha256 of the request bodies (json.dumps(calls, ensure_ascii=False, sort_keys=True)) that run_exam sent for the
# fixture with default settings, captured before the extras existed. A base run must keep sending exactly these.
BASE_REQUESTS_SHA256 = "6052eccb5cf2d1632d0b54cf7812e2a67ed0ae73e03d2366dcdda618591c1461"  # cache_prompt false since 26.09 (reproducible runs)
BASE_REQUESTS_RETRY_SHA256 = "632cbdabae5b6bf4347896bc83739a567fbc254bbbc26eafe7c898b29ea2936d"
DESC_Z01 = {"images/Z01.png": "Domy z drewna."}


def _requests_sha(calls: list[dict]) -> str:
    return hashlib.sha256(json.dumps(calls, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _log(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out.parent / "run_log.jsonl").read_text("utf-8").splitlines()]


def _answers(out: Path) -> dict[str, str]:
    return {a["id"]: a["answer"] for a in json.loads(out.read_text("utf-8"))["answers"]}


def test_organizer_requests_are_byte_identical_with_defaults(monkeypatch, tmp_path, exam_dir):
    run_exam = _load_script("run_exam")
    desc = tmp_path / "desc.json"
    desc.write_text(json.dumps(DESC_Z01), encoding="utf-8")
    base = ["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--lora-scale", "0", "--parallel", "1",
            "--descriptions", str(desc)]
    calls = _fake_server(monkeypatch, "Temat 2.\n\n" + _essay_body(330, "reforma"))
    out = tmp_path / "a" / "answers.json"
    assert run_exam.main(base + ["--out", str(out)]) == 0
    assert _requests_sha(calls) == BASE_REQUESTS_SHA256
    _, its = exam_io.load_exam(exam_dir)
    by_id = {it["id"]: it for it in its}
    keys = {"messages", "temperature", "top_p", "seed", "max_tokens", "stream", "model", "top_k", "repeat_penalty",
            "cache_prompt", "lora"}
    for body in calls:
        item_id = re.match(r"Zadanie (\S+) \(", body["messages"][1]["content"]).group(1)
        assert set(body) == keys
        assert body["messages"] == build_messages(by_id[item_id], DESC_Z01)
        assert (body["temperature"], body["top_k"], body["seed"], body["lora"]) == (0.0, 1, 42, [{"id": 0, "scale": 0.0}])
        assert "Materiały pomocnicze" not in json.dumps(body, ensure_ascii=False)
    for rec in (r for r in _log(out) if r["type"] == "item"):
        assert not {"rag", "votes", "repair", "essay_plan", "essay_messages"} & set(rec)
    # the essay continuation request (--essay-retry) is unchanged too
    calls = _fake_server(monkeypatch, "Temat 1.\n\n" + _essay_body(120, "port"), continuation=_essay_body(250, "handel"))
    assert run_exam.main(base + ["--essay-retry", "--out", str(tmp_path / "b" / "answers.json")]) == 0
    assert len(calls) == 7 and _requests_sha(calls) == BASE_REQUESTS_RETRY_SHA256


def test_rag_style_messages(items):
    it = items["3"]
    organizer = build_messages(it)
    rag = build_messages(it, style="rag", passages="[1] Rada miejska\nRada rządziła miastem.\n")
    assert rag[0]["content"] == ORGANIZER_SYSTEM_PROMPT + " " + prompts.RAG_SYSTEM_SENTENCE
    assert prompts.RAG_SYSTEM_SENTENCE == ("Możesz korzystać z załączonych materiałów pomocniczych; mogą być "
                                           "niekompletne lub nieistotne.")
    assert rag[1]["content"] == ("Materiały pomocnicze:\n[1] Rada miejska\nRada rządziła miastem.\n\n"
                                 + organizer[1]["content"])
    for empty in (None, "", "  \n"):
        msgs = build_messages(it, style="rag", passages=empty)
        assert msgs[1] == organizer[1] and msgs[0]["content"].endswith(prompts.RAG_SYSTEM_SENTENCE)
    assert build_messages(it, passages=None) == organizer and build_messages(it, passages="") == organizer
    assert build_messages(it, system_prompt=None, style="rag", passages="x") == [
        {"role": "user", "content": "Materiały pomocnicze:\nx\n\n" + organizer[1]["content"]}]
    with pytest.raises(ValueError, match="style='rag'"):
        build_messages(it, passages="x")  # a base (organizer) prompt never carries passages
    with pytest.raises(ValueError):
        build_messages(it, style="formatted", passages="x")
    with pytest.raises(TypeError):
        build_messages(it, style="rag", passages=[{"title": "x"}])


def test_follow_up_prompts(items):
    essay = items["5"]
    plan_msgs = prompts.essay_plan_messages(build_messages(essay), essay)
    assert plan_msgs[0] == build_messages(essay)[0]
    assert plan_msgs[1]["content"].startswith(build_messages(essay)[1]["content"].rstrip() + "\n\n")
    assert "(1, 2 lub 3)" in plan_msgs[1]["content"] and "Kontrargument:" in plan_msgs[1]["content"]
    step2 = prompts.essay_from_plan_messages(build_messages(essay), essay, "Temat 3.\nTeza: tak.", 3)[1]["content"]
    assert "Plan wypracowania:\nTemat 3.\nTeza: tak." in step2 and "„Temat 3.”" in step2
    assert "od 450 do 650 wyrazów" in step2 and "Kolej żelazna" in step2
    assert "numer wybranego tematu (1, 2 lub 3)" in prompts.essay_from_plan_instruction(essay, "plan", None)
    assert prompts.answer_labels(items["1"]) == ["Rozstrzygnięcie:", "Uzasadnienie:"]
    assert prompts.answer_labels(items["3"]) == []
    rep = prompts.label_repair_instruction(["Rozstrzygnięcie:", "Uzasadnienie:"], ["Uzasadnienie:"])
    assert rep.startswith("W odpowiedzi brakuje elementu „Uzasadnienie:”.")
    assert "Rozstrzygnięcie:\nUzasadnienie:\n" in rep


@pytest.mark.parametrize("answer,missing", [
    ("Rozstrzygnięcie: epoka żelaza\nUzasadnienie: piec.", []),
    ("**Rozstrzygnięcie**: epoka żelaza\n**Uzasadnienie:** piec.", []),
    ("rozstrzygnięcie – epoka żelaza. uzasadnienie – piec.", []),
    ("Rozstrzygnięcie\nepoka żelaza\n\nUzasadnienie\npiec", []),
    ("Rozstrzygnięcie: epoka żelaza, bo widać piec.", ["Uzasadnienie:"]),
    ("Epoka żelaza, ponieważ na rysunku widać piec.", ["Rozstrzygnięcie:", "Uzasadnienie:"]),
])
def test_missing_labels(items, answer, missing):
    assert postprocess.missing_labels(items["1"], answer) == missing


def test_label_present_variants():
    assert postprocess.label_present("Opis 1.:", "Opis 1: król")
    assert postprocess.label_present("Opis 1.:", "Opis 1.: król")
    assert not postprocess.label_present("Opis 1.:", "Opis 2: król")
    assert not postprocess.label_present("Fragment A:", "Fragment AB: x")
    assert postprocess.missing_labels({"question": "Podaj nazwę."}, "cokolwiek") == []


def test_clean_essay_with_topic(items):
    body = _essay_body(320, "tekst")
    r = postprocess.clean_essay_with_topic(items["5"], body, 2)
    assert r.answer == "Temat 2.\n\n" + body and r.essay_topic == 2 and r.topic_source == "plan"
    assert r.essay_words == 320 and not any("not detected" in n for n in r.notes)
    explicit = postprocess.clean_essay_with_topic(items["5"], "Temat 1.\n\n" + body, 2)
    assert explicit.essay_topic == 1 and explicit.answer.startswith("Temat 1.")  # the essay's own choice wins
    assert postprocess.clean_essay_with_topic(items["5"], body, None).answer == clean(items["5"], body).answer
    assert postprocess.clean_essay_with_topic(items["5"], body, 7).answer == clean(items["5"], body).answer


def test_vote_closed_rules(items):
    single = items["3"]
    v = postprocess.vote_closed(single, "A", ["B", "Odpowiedź: B", "B. republika miejska.", "A", "C"])
    assert (v.answer, v.rule, v.changed, v.closed_ok) == ("B", "majority", True, True)
    assert v.tally[""] == {"A": 2, "B": 3, "C": 1} and v.greedy_answer == "A" and v.n_votes == 6
    tie = postprocess.vote_closed(single, "A", ["B", "B", "A", "C", "C"])
    assert (tie.answer, tie.rule) == ("A", "tie->greedy")  # A 2, B 2, C 2
    assert postprocess.vote_closed(single, "A", ["A", "A"]).rule == "unanimous"
    # unparseable greedy: the majority of the valid samples decides
    assert postprocess.vote_closed(single, "Nie wiem.", ["C", "C", "B"]).answer == "C"
    # true/false votes per row: each row takes its own majority
    tf = items["2.1"]
    rows = postprocess.vote_closed(tf, "1: P\n2: F\n3: F\n4: P",
                                   ["1: F\n2: F\n3: P\n4: P", "1: F\n2: F\n3: P\n4: P", "1: P\n2: P\n3: F\n4: P",
                                    "1: F\n2: F\n3: P", "F, F, P, F"])
    assert rows.answer == "1: F\n2: F\n3: P\n4: P" and rows.rules["4"] == "majority" and rows.closed_ok
    assert rows.tally["1"] == {"P": 2, "F": 4} and rows.tally["4"] == {"P": 4, "F": 1}
    even = postprocess.vote_closed(tf, "1: P\n2: F\n3: F\n4: P", ["1: F\n2: F\n3: F\n4: P"])
    assert even.answer == "1: P\n2: F\n3: F\n4: P" and even.rules["1"] == "tie->greedy"
    # multi-part rows too; a tie in part 2 keeps the greedy letter
    multi = postprocess.vote_closed(items["4"], "1: C\n2: B", ["1: C\n2: A", "1: A\n2: A", "1: C\n2: B"])
    assert multi.answer == "1: C\n2: B" and multi.rules == {"1": "majority", "2": "tie->greedy"}
    # matching votes with the whole answer (rows are coupled)
    match = postprocess.vote_closed(items["2.2"], "A: 1\nB: 3\nC: 2", ["A: 2\nB: 3\nC: 1"] * 2)
    assert match.answer == "A: 2\nB: 3\nC: 1" and list(match.tally) == [""]
    # nothing recoverable anywhere: the greedy-only result stands
    none = postprocess.vote_closed(tf, "Trudno powiedzieć.", ["Nie wiem."])
    assert none.rule == "fallback-greedy" and none.closed_ok is False and none.answer == "Trudno powiedzieć."
    with pytest.raises(ValueError):
        postprocess.closed_vote_units(items["1"], "x")


def test_vote_decided_never_changes_the_result(items):
    single = items["3"]
    assert not postprocess.vote_decided(single, "A", [], 5)
    assert not postprocess.vote_decided(single, "A", ["A", "A"], 4)       # 3 v 0 with 4 left: B could win 4 v 3
    assert postprocess.vote_decided(single, "A", ["A", "A"], 3)           # 3 v 3 at worst -> tie -> greedy A
    assert not postprocess.vote_decided(single, "B", ["A", "A"], 1)       # A 2 v B 1: a tie would go to greedy B
    assert postprocess.vote_decided(single, "B", ["A", "A", "A"], 1)      # A 3 v B 1
    assert postprocess.vote_decided(single, "A", ["A", "A", "B"], 1)      # 3 v 1, 1 left: tie at worst -> greedy A
    assert not postprocess.vote_decided(single, "A", ["B", "B", "A"], 1)  # B 2 v A 2
    assert not postprocess.vote_decided(single, "Nie wiem.", [], 3)
    assert postprocess.vote_decided(single, "A", ["B"], 0)
    tf = items["2.1"]
    assert not postprocess.vote_decided(tf, "1: P\n2: F\n3: F", ["1: P\n2: F\n3: F\n4: P"] * 3, 3)  # row 4: 3 v 0,
    # three left and no greedy value for row 4 to break a 3 v 3 tie
    assert postprocess.vote_decided(tf, "1: P\n2: F\n3: F", ["1: P\n2: F\n3: F\n4: P"] * 3, 2)
    assert postprocess.vote_decided(tf, "1: P\n2: F\n3: F\n4: P", ["1: P\n2: F\n3: F\n4: P"] * 3, 2)


# fake server for the extras: answers by item id (found in the "Zadanie X (" header, wherever passages put it) and by
# request type (essay plan, essay from plan, label repair, sampled vote, continuation)
def _smart_server(monkeypatch, respond):
    calls = []

    def fake_post(self, url, json=None, timeout=None):
        calls.append(json)
        user0 = next(m for m in json["messages"] if m["role"] == "user")["content"]
        item_id = re.search(r"Zadanie (\S+) \(", user0).group(1)
        last = json["messages"][-1]["content"]
        if len(json["messages"]) > 2 and json["messages"][-2]["role"] == "assistant":
            kind = "repair" if last.startswith("W odpowiedzi brakuje") else "continue"
        elif "Nie pisz jeszcze wypracowania." in last:
            kind = "plan"
        elif "Plan wypracowania:" in last:
            kind = "essay_from_plan"
        elif json.get("temperature", 0.0) > 0:
            kind = "sample"
        else:
            kind = "greedy"
        text = respond(item_id, kind, json)
        if text is None:
            return _Resp(400, text="bad request")
        return _ok(text, ctok=len(text.split()))

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _Resp(200, {"data": [{"id": "fake-bielik"}]}))
    return calls


class _FakeIndex:
    def __init__(self):
        self.queries: list[tuple[str, int]] = []

    def search(self, query, k=5):
        import numpy as np
        self.queries.append((query, k))
        if "RAISE" in query:
            raise RuntimeError("index broken")
        topic = "Kolej warszawsko-wiedeńska" if query.startswith("Kolej") else "Rada miejska"
        return [{"title": f"{topic} {i}", "section": "Historia", "text": f"Fakt {i} o haśle {topic}.",
                 "url": f"https://pl.wikipedia.org/wiki/H{i}", "score": np.float32(3.5 - i)} for i in range(1, k + 1)]


def _fake_rag(monkeypatch):
    """A stand-in harness.rag (contract: load_index, index.search, build_query, format_passages)."""
    mod = types.ModuleType("harness.rag")
    index = _FakeIndex()
    loaded: list[str] = []

    def load_index(path):
        if "missing" in str(path):
            raise FileNotFoundError(path)
        loaded.append(str(path))
        return index

    mod.load_index = load_index
    mod.build_query = lambda item: f"{item.get('question', '')}\n{item.get('source_text', '')}".strip()
    mod.format_passages = lambda passages, max_chars=2400: "\n\n".join(
        f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))[:max_chars]
    monkeypatch.setitem(sys.modules, "harness.rag", mod)
    return index, loaded


GOOD = {
    "1": "Rozstrzygnięcie: epoka żelaza\nUzasadnienie: na rysunku widać piec do wytopu żelaza.",
    "2.1": "1: P\n2: F\n3: F\n4: P",
    "2.2": "A: 2\nB: 3\nC: 1",
    "3": "B",
    "4": "1: C\n2: B",
}
PLAN_3 = ("Temat 3.\nTeza: Kolej przyspieszyła industrializację ziem polskich.\n"
          "Argument 1: Kolej Warszawsko-Wiedeńska (1845–1848) połączyła Warszawę z Zagłębiem Dąbrowskim.\n"
          "Argument 2: Łódź rozwinęła się dzięki kolei fabryczno-łódzkiej (1865).\n"
          "Kontrargument: Galicja miała słabszą sieć kolejową.\nWniosek: Kolej była motorem industrializacji.")


def test_run_exam_rag_style_logs_passages(monkeypatch, tmp_path, exam_dir):
    index, loaded = _fake_rag(monkeypatch)
    calls = _smart_server(monkeypatch, lambda i, kind, body: GOOD.get(i, "Temat 3.\n\n" + _essay_body(320)))
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out),
                          "--descriptions", "none", "--style", "rag", "--rag-index", "data/rag/index",
                          "--rag-k", "2", "--parallel", "1"]) == 0
    assert loaded == ["data/rag/index"] and len(index.queries) == 6 and {k for _, k in index.queries} == {2}
    assert len(calls) == 6
    for body in calls:
        assert body["messages"][0]["content"].endswith(prompts.RAG_SYSTEM_SENTENCE)
        assert body["messages"][1]["content"].startswith("Materiały pomocnicze:\n[1] ")
        assert (body["temperature"], body["top_k"]) == (0.0, 1)
    got = _answers(out)
    assert got["3"] == "B" and got["5"].startswith("Temat 3.")
    recs = {r["id"]: r for r in _log(out) if r["type"] == "item"}
    rag = recs["3"]["rag"]
    assert rag["k"] == 2 and rag["query"].startswith("Dokończ zdanie.") and len(rag["passages"]) == 2
    assert rag["passages"][0] == {"title": "Rada miejska 1", "section": "Historia", "url": "https://pl.wikipedia.org/wiki/H1",
                                  "score": 2.5, "text": "Fakt 1 o haśle Rada miejska."}
    assert _log(out)[0]["rag_index"] == "data/rag/index" and _log(out)[0]["style"] == "rag"
    # a base run never imports or queries the retriever
    index.queries.clear()
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--descriptions", "none",
                          "--out", str(tmp_path / "base" / "answers.json")]) == 0
    assert index.queries == [] and loaded == ["data/rag/index"]
    # a broken retrieval never stops the exam: the item is asked without passages and the error is logged
    monkeypatch.setattr(sys.modules["harness.rag"], "build_query",
                        lambda item: "RAISE" if item["id"] == "3" else item["question"])
    out3 = tmp_path / "broken" / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out3),
                          "--descriptions", "none", "--style", "rag", "--rag-index", "idx", "--only", "3,4"]) == 0
    assert _answers(out3)["3"] == "B"
    rec3 = next(r for r in _log(out3) if r.get("id") == "3")
    assert "index broken" in rec3["rag"]["error"] and rec3["rag"]["passages"] == []
    assert rec3["messages"][1]["content"].startswith("Zadanie 3 (1 pkt)")


def test_run_exam_rag_option_errors(monkeypatch, tmp_path, exam_dir, capsys):
    _fake_rag(monkeypatch)
    run_exam = _load_script("run_exam")
    base = ["--exam-dir", str(exam_dir), "--dry-run", "--out", str(tmp_path / "a.json")]
    with pytest.raises(SystemExit):
        run_exam.main(base + ["--style", "rag"])
    with pytest.raises(SystemExit):
        run_exam.main(base + ["--rag-index", "data/rag/index"])  # organizer + index: refused
    with pytest.raises(SystemExit):
        run_exam.main(base + ["--vote", "0"])
    assert run_exam.main(base + ["--style", "rag", "--rag-index", "missing/index"]) == 2
    assert "cannot load the RAG index" in capsys.readouterr().err
    assert run_exam.main(["--exam-dir", str(exam_dir), "--show-prompts", "--only", "3", "--style", "rag",
                          "--rag-index", "x"]) == 0
    shown = capsys.readouterr().out
    assert "rag passages: Rada miejska 1 (2.5)" in shown and "Materiały pomocnicze:\n[1] Rada miejska 1" in shown


def test_run_exam_essay_plan_mode_with_rag(monkeypatch, tmp_path, exam_dir):
    index, _ = _fake_rag(monkeypatch)
    essay_text = _essay_body(480, "kolej")  # no topic header: the plan's topic is added

    def respond(i, kind, body):
        if i != "5":
            return GOOD[i]
        return {"plan": "**Plan**\n" + PLAN_3, "essay_from_plan": essay_text}[kind]

    calls = _smart_server(monkeypatch, respond)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out),
                          "--descriptions", "none", "--style", "rag", "--rag-index", "idx", "--essay-mode", "plan",
                          "--parallel", "1"]) == 0
    essay_calls = [c for c in calls if "Zadanie 5 (" in c["messages"][1]["content"]]
    assert len(essay_calls) == 2 and len(calls) == 7
    plan_req, essay_req = essay_calls
    assert plan_req["max_tokens"] == prompts.ESSAY_PLAN_MAX_TOKENS and essay_req["max_tokens"] == 4096
    assert plan_req["messages"][1]["content"].startswith("Materiały pomocnicze:\n[1] ")
    user2 = essay_req["messages"][1]["content"]
    assert "Plan wypracowania:\nPlan\nTemat 3." in user2 and "Zacznij od „Temat 3.”" in user2
    assert user2.startswith("Materiały pomocnicze:\n[1] Kolej warszawsko-wiedeńska 1")  # passages for topic 3
    topic_query = index.queries[-1][0]
    assert topic_query.startswith("Kolej żelazna przyspieszyła industrializację")
    assert "Łódź rozwinęła się" in topic_query and "Kontrargument" not in topic_query and "Teza" not in topic_query
    got = _answers(out)
    assert got["5"] == "Temat 3.\n\n" + essay_text
    rec = next(r for r in _log(out) if r.get("id") == "5")
    assert rec["essay_plan"]["topic"] == 3 and rec["essay_plan"]["raw"].endswith("motorem industrializacji.")
    assert rec["essay_plan"]["rag"]["passages"][0]["title"] == "Kolej warszawsko-wiedeńska 1"
    assert rec["essay_messages"] == essay_req["messages"] and rec["topic_source"] == "plan"
    # step 1 is the item's own rag prompt (passages for the whole question) + the planning instruction
    assert rec["messages"][1]["content"].startswith("Materiały pomocnicze:\n[1] Rada miejska 1")
    assert plan_req["messages"][1]["content"].startswith(rec["messages"][1]["content"] + "\n\nZanim napiszesz")
    assert rec["usage_total"]["completion_tokens"] == len(("**Plan**\n" + PLAN_3).split()) + 480
    assert _log(out)[-1]["essays"]["5"]["plan_topic"] == 3


def test_run_exam_essay_plan_mode_organizer_and_retry(monkeypatch, tmp_path, exam_dir):
    def respond(i, kind, body):
        if i != "5":
            return GOOD[i]
        return {"plan": "Wybieram temat 1.\nTeza: porty wzbogaciły państwo.", "essay_from_plan":
                "Temat 1.\n\n" + _essay_body(150, "port"), "continue": _essay_body(200, "handel")}[kind]

    calls = _smart_server(monkeypatch, respond)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out),
                          "--essay-mode", "plan", "--essay-retry", "--parallel", "1"]) == 0
    essay_calls = [c for c in calls if "Zadanie 5 (" in c["messages"][1]["content"]]
    assert len(essay_calls) == 3
    plan_req, essay_req, cont_req = essay_calls
    assert plan_req["messages"][0]["content"] == ORGANIZER_SYSTEM_PROMPT  # "same style": organizer stays organizer
    assert "Materiały" not in json.dumps(calls, ensure_ascii=False)
    assert cont_req["messages"][:2] == essay_req["messages"] and cont_req["messages"][2]["role"] == "assistant"
    assert essay_word_count(_answers(out)["5"]) == 350
    rec = next(r for r in _log(out) if r.get("id") == "5")
    assert rec["continuation"]["words_before"] == 150 and rec["essay_plan"]["topic"] == 1


def test_run_exam_essay_plan_failure_falls_back_to_one_step(monkeypatch, tmp_path, exam_dir):
    def respond(i, kind, body):
        if i != "5":
            return GOOD[i]
        return {"plan": None, "greedy": "Temat 2.\n\n" + _essay_body(320, "reforma")}[kind]

    calls = _smart_server(monkeypatch, respond)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out), "--retries",
                          "0", "--essay-mode", "plan", "--parallel", "1"]) == 0
    assert _answers(out)["5"].startswith("Temat 2.\n\n")
    rec = next(r for r in _log(out) if r.get("id") == "5")
    assert "HTTP 400" in rec["essay_plan"]["error"] and rec["essay_messages"] == rec["messages"]
    assert len([c for c in calls if "Zadanie 5 (" in c["messages"][1]["content"]]) == 2


def test_run_exam_vote_on_closed_items(monkeypatch, tmp_path, exam_dir):
    samples = {
        "3": {1: "B", 2: "Odpowiedź: B", 3: "A", 4: "B", 5: "C"},          # greedy A, majority B
        "2.1": {s: "1: P\n2: F\n3: F\n4: F" for s in range(1, 6)},          # row 4: F outvotes the greedy P
    }

    def respond(i, kind, body):
        if kind == "sample":
            return samples.get(i, {}).get(body["seed"], GOOD.get(i, "x"))
        return {"3": "A"}.get(i, GOOD.get(i, "Temat 1.\n\n" + _essay_body(310)))

    calls = _smart_server(monkeypatch, respond)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out),
                          "--vote", "5", "--parallel", "2"]) == 0
    got = _answers(out)
    assert got["3"] == "B" and got["2.1"] == "1: P\n2: F\n3: F\n4: F"
    assert got["2.2"] == "A: 2\nB: 3\nC: 1" and got["4"] == "1: C\n2: B"
    sampled = [c for c in calls if c["temperature"] > 0]
    assert all(c["temperature"] == 0.7 and c["top_k"] == 40 and c["seed"] in range(1, 6) for c in sampled)
    per_item = {}
    for c in sampled:
        per_item.setdefault(re.search(r"Zadanie (\S+) \(", c["messages"][1]["content"]).group(1), []).append(c["seed"])
    assert set(per_item) == {"2.1", "2.2", "3", "4"}          # closed items only
    assert per_item["3"] == [1, 2, 3, 4, 5]        # B 3 v A 2 after four samples: the fifth could still tie
    assert per_item["2.1"] == [1, 2, 3, 4]           # row 4: F 4 v P 1 with one sample left
    assert per_item["2.2"] == [1, 2] and per_item["4"] == [1, 2]  # unanimous 3 v 0, three left: a tie at worst
    greedy = [c for c in calls if c["temperature"] == 0.0]
    assert len(greedy) == 6 and all(c["seed"] == 42 and c["top_k"] == 1 for c in greedy)
    recs = {r["id"]: r for r in _log(out) if r["type"] == "item"}
    v = recs["3"]["votes"]
    assert v["greedy_answer"] == "A" and v["answer"] == "B" and v["changed"] and v["rule"] == "majority"
    assert v["tally"] == {"": {"A": 2, "B": 3, "C": 1}} and [s["seed"] for s in v["samples"]] == [1, 2, 3, 4, 5]
    assert recs["2.2"]["votes"]["skipped"] == 3 and recs["2.2"]["votes"]["rule"] == "unanimous"
    assert "votes" not in recs["1"] and "votes" not in recs["5"]
    summary = _log(out)[-1]
    assert summary["votes"]["3"]["changed"] and not summary["votes"]["4"]["changed"]
    assert _log(out)[0]["vote_decoding"]["temperature"] == 0.7


def test_run_exam_repair_labels(monkeypatch, tmp_path, exam_dir):
    replies = {"greedy": "Epoka żelaza, bo na rysunku widać piec.",
               "repair": "Rozstrzygnięcie: epoka żelaza\nUzasadnienie: na rysunku widać piec."}
    calls = _smart_server(monkeypatch, lambda i, kind, body: replies[kind] if i == "1" else
                          GOOD.get(i, "Temat 1.\n\n" + _essay_body(310)))
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out),
                          "--repair-labels", "--parallel", "1"]) == 0
    assert _answers(out)["1"] == replies["repair"]
    repair_req = next(c for c in calls if len(c["messages"]) == 4)
    assert repair_req["messages"][2] == {"role": "assistant", "content": replies["greedy"]}
    assert repair_req["messages"][3]["content"].startswith("W odpowiedzi brakuje elementów „Rozstrzygnięcie:”")
    assert (repair_req["temperature"], repair_req["seed"]) == (0.0, 42)
    rec = next(r for r in _log(out) if r.get("id") == "1")
    assert rec["repair"]["used"] and rec["repair"]["missing_before"] == ["Rozstrzygnięcie:", "Uzasadnienie:"]
    assert rec["repair"]["missing_after"] == [] and rec["raw"] == replies["greedy"]
    assert len(calls) == 7  # 6 items + one repair (labelled answers and closed items are never re-asked)
    # a repair that is no better is not used
    replies["repair"] = "Epoka żelaza."
    out2 = tmp_path / "b" / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out2),
                          "--repair-labels", "--only", "1"]) == 0
    assert _answers(out2)["1"] == replies["greedy"]
    assert next(r for r in _log(out2) if r.get("id") == "1")["repair"]["used"] is False


def test_run_exam_extras_resume_reuses_everything(monkeypatch, tmp_path, exam_dir):
    def respond(i, kind, body):
        if i == "1":
            return {"greedy": "Epoka żelaza.", "repair": GOOD["1"]}[kind]
        if i == "5":
            return {"plan": PLAN_3, "essay_from_plan": _essay_body(460, "kolej")}[kind]
        if kind == "sample":
            return {"3": "C"}.get(i, GOOD[i]) if body["seed"] <= 4 else GOOD[i]
        return GOOD[i]

    calls = _smart_server(monkeypatch, respond)
    _fake_rag(monkeypatch)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    args = ["--exam-dir", str(exam_dir), "--base-url", "http://fake:1", "--out", str(out), "--descriptions", "none",
            "--style", "rag", "--rag-index", "idx", "--essay-mode", "plan", "--vote", "5", "--repair-labels",
            "--parallel", "1"]
    assert run_exam.main(args) == 0
    first = _answers(out)
    assert first["3"] == "C" and first["1"] == GOOD["1"] and first["5"].startswith("Temat 3.\n\n")
    n = len(calls)
    assert run_exam.main(args + ["--resume"]) == 0
    assert len(calls) == n and _answers(out) == first
    # another vote setting is not reused for closed items, but open items and the essay are
    assert run_exam.main(args + ["--vote", "3", "--resume"]) == 0  # the last --vote wins
    new = calls[n:]
    assert new and all(re.search(r"Zadanie (2\.1|2\.2|3|4) \(", c["messages"][1]["content"]) for c in new)


def test_run_exam_dry_run_with_all_extras(monkeypatch, tmp_path, exam_dir, capsys):
    def boom(*a, **k):
        raise AssertionError("network used in dry run")

    monkeypatch.setattr(requests.Session, "post", boom)
    monkeypatch.setattr(requests, "get", boom)
    _fake_rag(monkeypatch)
    run_exam = _load_script("run_exam")
    out = tmp_path / "answers.json"
    assert run_exam.main(["--exam-dir", str(exam_dir), "--dry-run", "--out", str(out), "--style", "rag",
                          "--rag-index", "idx", "--essay-mode", "plan", "--vote", "5", "--repair-labels"]) == 0
    assert exam_io.validate_answers(out, exam_dir) == []
    got = _answers(out)
    assert got["3"] == "A" and got["5"].startswith("Temat 1.")
    text = capsys.readouterr().out
    assert "votes (K=5, T=0.7)" in text and "label repairs: 1 asked, 0 used" in text and "rag: idx, k=4" in text


def test_retrieve_min_score_drops_weak_hits():
    run_exam = _load_script("run_exam")

    class FakeIndex:
        def search(self, query, k=4):
            return [{"title": "Liberum veto", "section": "", "text": "Zasada jednomyślności w sejmie.", "url": "u1",
                     "score": 93.8},
                    {"title": "Styl zakopiański", "section": "", "text": "Styl w architekturze.", "url": "u2",
                     "score": 23.1}]

    class FakeRag:
        @staticmethod
        def format_passages(hits, max_chars=2400):
            return "\n\n".join(h["title"] for h in hits)

    text, info = run_exam.retrieve(FakeRag, FakeIndex(), "q", 4, 2400)
    assert text == "Liberum veto\n\nStyl zakopiański" and "dropped_low_score" not in info
    text, info = run_exam.retrieve(FakeRag, FakeIndex(), "q", 4, 2400, min_score=50)
    assert text == "Liberum veto"
    assert [p["title"] for p in info["passages"]] == ["Liberum veto"]
    assert [p["title"] for p in info["dropped_low_score"]] == ["Styl zakopiański"]
    text, info = run_exam.retrieve(FakeRag, FakeIndex(), "q", 4, 2400, min_score=200)
    assert text == "" and info["passages"] == [] and len(info["dropped_low_score"]) == 2


def test_essay_entities_and_entity_retrieval():
    run_exam = _load_script("run_exam")
    plan = ("Temat 1.\nTeza: W XI–XII wieku dominowały tendencje decentralizacyjne.\n"
            "Argument 1: Mieszko II Lambert utracił koronę w 1031.\nArgument 2: Kazimierz Odnowiciel pokonał Masława.\n"
            "Wniosek: Polska weszła w rozbicie dzielnicowe.")
    names = run_exam.essay_entities(plan, "Oceń, czy w okresie XI–XII wieku dominowały tendencje decentralizacyjne.")
    assert names == ["Mieszko II Lambert", "Kazimierz Odnowiciel"]

    class FakeIndex:
        def search(self, query, k=6):
            return [{"title": "Styl zakopiański", "section": "", "text": "x", "url": "u0", "score": 30.0},
                    {"title": query.split()[0] + " I", "section": "", "text": "fakty " + query, "url": "u1",
                     "score": 15.0}]

    class FakeRag:
        @staticmethod
        def format_passages(hits, max_chars=2400):
            return "|".join(h["title"] for h in hits)

    text, info = run_exam.retrieve_entities(FakeRag, FakeIndex(), names, 2, 4800)
    assert text == "Mieszko I|Kazimierz I"          # the off-topic higher-scoring hit is not kept
    assert info["mode"] == "entities" and info["queries"] == names


def test_merge_answers(tmp_path):
    merge_answers = _load_script("merge_answers")
    base = {"exam_id": "x", "answers": [{"id": "1", "answer": "A"}, {"id": "26", "answer": "Temat 1. krótko"}]}
    over = {"exam_id": "x", "answers": [{"id": "1", "answer": ""}, {"id": "26", "answer": "Temat 2. długo"}]}
    merged, sources = merge_answers.merge(base, over, ["26"])
    assert merged["answers"] == [{"id": "1", "answer": "A"}, {"id": "26", "answer": "Temat 2. długo"}]
    assert sources == {"1": "base", "26": "override"}
    import pytest
    with pytest.raises(SystemExit):
        merge_answers.merge(base, over, ["1"])          # empty override answer is refused
    with pytest.raises(SystemExit):
        merge_answers.merge(base, {"exam_id": "y", "answers": []}, ["26"])
