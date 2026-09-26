import pytest

from eval.answer_format import (
    VARIANTS,
    canonical_payload,
    format_instruction,
    letters_phrase,
    parse_lenient,
    parse_strict,
    render_target,
)
from eval.grader import grade, summarize
from eval.prompts import build_messages, build_training_example
from eval.schema import validate_item

MC = validate_item({
    "id": "t-mc", "subject": "historia", "type": "mc",
    "question": "W którym roku uchwalono Konstytucję 3 maja?",
    "options": {"A": "1772", "B": "1791", "C": "1795", "D": "1807"}, "answer": "B",
    "rationale": "Konstytucję uchwalił Sejm Czteroletni 3 maja 1791 roku.",
})
MULTI = validate_item({
    "id": "t-multi", "subject": "biologia", "type": "multi", "question": "Wybierz organelle z własnym DNA.",
    "options": {"A": "mitochondrium", "B": "rybosom", "C": "chloroplast", "D": "lizosom"}, "answer": ["C", "A"],
})
TF = validate_item({
    "id": "t-tf", "subject": "geografia", "type": "tf", "question": "Oceń zdania.",
    "statements": ["Wisła uchodzi do Bałtyku.", "Rysy leżą w Sudetach.", "Warszawa leży nad Wisłą."],
    "answer": ["P", "F", "P"], "points": 2, "partial_credit": [[2, 1]],
})
MATCH = validate_item({
    "id": "t-match", "subject": "jezyk_polski", "type": "match", "question": "Dopasuj autora do utworu.",
    "left": ["Pan Tadeusz", "Lalka"], "right": {"A": "Bolesław Prus", "B": "Adam Mickiewicz", "C": "Henryk Sienkiewicz"},
    "answer": {"1": "B", "2": "A"},
})
SHORT = validate_item({
    "id": "t-short", "subject": "historia", "type": "short", "question": "Kto był pierwszym historycznym władcą Polski?",
    "answer": ["Mieszko I", "Mieszko Pierwszy"],
})
NUM = validate_item({
    "id": "t-num", "subject": "fizyka", "type": "numeric", "question": "Oblicz prędkość w m/s.",
    "answer": 3.5, "unit": "m/s", "tolerance": 0.01,
})


def test_canonical_payloads():
    assert canonical_payload(MC) == "B"
    assert canonical_payload(MULTI) == "A, C"
    assert canonical_payload(TF) == "P, F, P"
    assert canonical_payload(MATCH) == "1-B, 2-A"
    assert canonical_payload(SHORT) == "Mieszko I"
    assert canonical_payload(NUM) == "3,5"


@pytest.mark.parametrize("item", [MC, MULTI, TF, MATCH, SHORT, NUM])
@pytest.mark.parametrize("variant", ["canonical", "bare", "payload_only"])
def test_render_target_round_trips_to_full_points(item, variant):
    rec = grade(item, render_target(item, variant), variant)
    assert rec["parse_ok"], rec
    assert rec["points"] == rec["max_points"], rec


def test_reason_then_answer_round_trip():
    out = render_target(MC, "reason_then_answer")
    assert out.endswith("Odpowiedź: B")
    assert grade(MC, out, "reason_then_answer")["points"] == 1.0


@pytest.mark.parametrize("out,ok,value", [
    ("Odpowiedź: B", True, "B"),
    ("  Odpowiedź:B  ", True, "B"),
    ("Uzasadnienie...\nOdpowiedź: B", True, "B"),
    ("Odpowiedź: B\nOdpowiedź: B", True, "B"),
    ("Odpowiedź: B\nOdpowiedź: C", False, None),
    ("**Odpowiedź:** B", False, None),
    ("Odpowiedź: B.", False, None),
    ("Odpowiedź: B) 1791", False, None),
    ("odpowiedź: B", False, None),
    ("B", False, None),
    ("Odpowiedź: E", False, None),
    ("", False, None),
])
def test_strict_mc(out, ok, value):
    p = parse_strict(MC, out, "canonical")
    assert p.ok is ok, (out, p)
    if ok:
        assert p.value == value


@pytest.mark.parametrize("out,value", [
    ("**Odpowiedź:** B", "B"),
    ("Poprawna odpowiedź to B.", "B"),
    ("B) 1791", "B"),
    ("Konstytucję uchwalono w 1791 roku.", "B"),
    ("odpowiedz: b)", "B"),
])
def test_lenient_mc_recovers(out, value):
    p = parse_lenient(MC, out)
    assert p.ok and p.value == value, (out, p)


def test_lenient_reads_cke_compact_answers_and_the_last_marker():
    """CKE notation ("FPP", "BE") and a corrected answer on the same line are format-independent credit."""
    tf = validate_item({"id": "t", "subject": "biologia", "type": "tf", "question": "q", "statements": ["a", "b", "c"], "answer": ["F", "P", "P"]})
    multi = validate_item({"id": "m", "subject": "biologia", "type": "multi", "question": "q",
                           "options": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}, "answer": ["B", "E"]})
    short = validate_item({"id": "s", "subject": "wos", "type": "short", "question": "q", "answer": ["zasada odpowiedzialności zbiorowej"]})
    for item, out, value in [
        (tf, "Odpowiedź: FPP", ["F", "P", "P"]), (tf, "FPP", ["F", "P", "P"]), (multi, "Odpowiedź: BE", ["B", "E"]), (multi, "BE", ["B", "E"]),
        (MC, "A więc odpowiedź to A. Odpowiedź: B", "B"), (MC, "Odpowiedź: B\nUzasadnienie krótkie.", "B"),
        (short, "Odpowiedź: zasada odpowiedzialności zbiorowej", "zasada odpowiedzialności zbiorowej"),
    ]:
        p = parse_lenient(item, out)
        assert p.ok and p.value == value, (out, p)
    assert grade(tf, "Odpowiedź: FPP")["points_lenient"] == 1.0 and grade(tf, "Odpowiedź: FPP")["points"] == 0.0
    assert not parse_lenient(tf, "Odpowiedź: PF").ok and not parse_lenient(multi, "Odpowiedź: BEE").ok


def test_payload_only_variant_is_strict_about_extra_text():
    assert parse_strict(MC, "B", "payload_only").ok
    assert not parse_strict(MC, "Odpowiedź: B", "payload_only").ok


def test_require_last_line_for_reasoning_variant():
    assert not parse_strict(MC, "Odpowiedź: B\nBo tak.", "reason_then_answer").ok
    assert parse_strict(MC, "Bo tak.\nOdpowiedź: B", "reason_then_answer").ok


def test_multi_order_insensitive_and_strict_letters():
    assert grade(MULTI, "Odpowiedź: C, A")["points"] == 1.0
    assert grade(MULTI, "Odpowiedź: A,C")["points"] == 1.0
    assert grade(MULTI, "Odpowiedź: A, A")["parse_ok"] is False
    assert grade(MULTI, "Odpowiedź: A")["points"] == 0.0


def test_tf_partial_credit_and_count():
    assert grade(TF, "Odpowiedź: P, F, P")["points"] == 2.0
    assert grade(TF, "Odpowiedź: P, P, P")["points"] == 1.0
    assert grade(TF, "Odpowiedź: F, P, F")["points"] == 0.0
    assert grade(TF, "Odpowiedź: P, F")["parse_reason"] == "tf_wrong_count"


def test_match():
    assert grade(MATCH, "Odpowiedź: 2-A, 1-B")["points"] == 1.0
    assert grade(MATCH, "Odpowiedź: 1-B")["parse_ok"] is False


def test_short_normalisation():
    assert grade(SHORT, "Odpowiedź: mieszko i")["points"] == 1.0
    assert grade(SHORT, "Odpowiedź: „Mieszko I”.")["points"] == 1.0
    assert grade(SHORT, "Odpowiedź: Bolesław Chrobry")["points"] == 0.0
    rec = grade(SHORT, "Pierwszym władcą był Mieszko I")
    assert rec["points"] == 0.0 and rec["points_lenient"] == 1.0


def test_numeric():
    assert grade(NUM, "Odpowiedź: 3.5")["points"] == 1.0
    assert grade(NUM, "Odpowiedź: 3,50")["points"] == 1.0
    assert grade(NUM, "Odpowiedź: 3,5 m/s")["points"] == 1.0
    assert grade(NUM, "Odpowiedź: 3,6")["points"] == 0.0
    assert grade(NUM, "Odpowiedź: ok. 3,5")["parse_ok"] is False
    assert grade(NUM, "Wynik to 3,5 m/s")["points_lenient"] == 1.0


def test_summary_counts():
    recs = [grade(MC, "Odpowiedź: B"), grade(MC, "B"), grade(TF, "Odpowiedź: P, P, P")]
    s = summarize(recs)
    assert s["n"] == 3 and s["points"] == 2.0 and s["max_points"] == 4.0
    assert s["parse_fail"] == 1 and s["points_lenient"] == 3.0
    assert s["per_subject"]["historia"]["n"] == 2


def test_messages_and_training_example():
    msgs = build_messages(MC, "canonical")
    assert msgs[-1]["role"] == "user" and "Odpowiedź: <litera>" in msgs[-1]["content"]
    assert "B. 1791" in msgs[-1]["content"]
    assert "Odpowiedź" not in build_messages(MC, "bare")[-1]["content"]
    ex = build_training_example(MC, "canonical", system_prompt="Jesteś asystentem.")
    assert ex["prompt"][0]["role"] == "system"
    assert ex["completion"] == [{"role": "assistant", "content": "Odpowiedź: B"}]
    assert set(VARIANTS) == {"canonical", "bare", "payload_only", "reason_then_answer", "organizer_letter"}


# --------------------------------------------------------------------- organizer_letter (constrained-letter variant)

MC3 = validate_item({
    "id": "t-mc3", "subject": "wos", "type": "mc", "question": "Ile jest izb w polskim parlamencie?",
    "options": {"A": "jedna", "B": "dwie", "C": "trzy"}, "answer": "B",
})

# The four pre-organizer variants, frozen: adding a variant must not change any existing render.
_EXISTING_RENDERS = [
    (MC, "canonical", "Wybierz jedną poprawną odpowiedź. Odpowiedz wyłącznie w formacie:\nOdpowiedź: <litera>", "Odpowiedź: B"),
    (MC, "bare", "", "Odpowiedź: B"),
    (MC, "payload_only",
     "Wybierz jedną poprawną odpowiedź. Podaj wyłącznie odpowiedź w postaci: <litera> — bez żadnego dodatkowego tekstu.", "B"),
    (MC, "reason_then_answer",
     "Wybierz jedną poprawną odpowiedź. Najpierw krótko uzasadnij odpowiedź (1–3 zdania), a w ostatniej linii napisz:\n"
     "Odpowiedź: <litera>", "Konstytucję uchwalił Sejm Czteroletni 3 maja 1791 roku.\nOdpowiedź: B"),
    (MULTI, "canonical",
     "Wybierz wszystkie poprawne odpowiedzi. Odpowiedz wyłącznie w formacie:\nOdpowiedź: <litery oddzielone przecinkami>",
     "Odpowiedź: A, C"),
    (MULTI, "payload_only", "Wybierz wszystkie poprawne odpowiedzi. Podaj wyłącznie odpowiedź w postaci: "
     "<litery oddzielone przecinkami> — bez żadnego dodatkowego tekstu.", "A, C"),
    (TF, "canonical", "Oceń, czy każde ze zdań jest prawdziwe (P), czy fałszywe (F). Odpowiedz wyłącznie w formacie:\n"
     "Odpowiedź: <P/F>, <P/F>, <P/F>", "Odpowiedź: P, F, P"),
    (TF, "payload_only", "Oceń, czy każde ze zdań jest prawdziwe (P), czy fałszywe (F). Podaj wyłącznie odpowiedź w "
     "postaci: <P/F>, <P/F>, <P/F> — bez żadnego dodatkowego tekstu.", "P, F, P"),
    (MATCH, "canonical", "Przyporządkuj każdemu elementowi oznaczonemu cyfrą właściwy element oznaczony literą. "
     "Odpowiedz wyłącznie w formacie:\nOdpowiedź: 1-<litera>, 2-<litera>", "Odpowiedź: 1-B, 2-A"),
    (MATCH, "bare", "", "Odpowiedź: 1-B, 2-A"),
    (SHORT, "canonical", "Udziel krótkiej odpowiedzi (słowo, nazwa lub krótkie wyrażenie). Odpowiedz wyłącznie w "
     "formacie:\nOdpowiedź: <odpowiedź>", "Odpowiedź: Mieszko I"),
    (SHORT, "payload_only", "Udziel krótkiej odpowiedzi (słowo, nazwa lub krótkie wyrażenie). Podaj wyłącznie "
     "odpowiedź w postaci: <odpowiedź> — bez żadnego dodatkowego tekstu.", "Mieszko I"),
    (NUM, "canonical", "Podaj sam wynik liczbowy. Odpowiedz wyłącznie w formacie:\nOdpowiedź: <liczba>", "Odpowiedź: 3,5"),
    (NUM, "payload_only",
     "Podaj sam wynik liczbowy. Podaj wyłącznie odpowiedź w postaci: <liczba> — bez żadnego dodatkowego tekstu.", "3,5"),
]


@pytest.mark.parametrize("item,variant,instruction,target", _EXISTING_RENDERS,
                         ids=[f"{it['id']}-{v}" for it, v, _, _ in _EXISTING_RENDERS])
def test_existing_variants_render_unchanged(item, variant, instruction, target):
    """Snapshot: organizer_letter is purely additive — no existing instruction or target moved."""
    assert format_instruction(item, variant) == instruction
    assert render_target(item, variant) == target


@pytest.mark.parametrize("item", [MC, MC3, MULTI, TF, MATCH, SHORT, NUM])
def test_organizer_letter_round_trips_to_full_points(item):
    out = render_target(item, "organizer_letter")
    assert out == canonical_payload(item)  # bare payload: the answer is the FIRST thing the model emits
    rec = grade(item, out, "organizer_letter")
    assert rec["parse_ok"], rec
    assert rec["points"] == rec["max_points"], rec


def test_organizer_letter_mc_instruction_names_the_items_letters():
    instr = format_instruction(MC, "organizer_letter")
    assert "Odpowiedz wyłącznie literą A, B, C albo D." in instr
    assert format_instruction(MC3, "organizer_letter").endswith("Odpowiedz wyłącznie literą A, B albo C.")
    assert "prawo jazdy" not in instr  # subject-neutral: our items are matura, not the driving exam
    assert letters_phrase("ABC") == "A, B albo C" and letters_phrase("AB") == "A albo B" and letters_phrase("A") == "A"


@pytest.mark.parametrize("item", [MULTI, TF, MATCH, SHORT, NUM])
def test_organizer_letter_falls_back_to_payload_only_for_non_mc(item):
    """The organizers' wording only fits single-letter items; every other type still renders."""
    assert format_instruction(item, "organizer_letter") == format_instruction(item, "payload_only")


def test_organizer_letter_is_strict_about_extra_text():
    assert parse_strict(MC, "B", "organizer_letter").ok
    assert not parse_strict(MC, "Odpowiedź: B", "organizer_letter").ok
    assert not parse_strict(MC, "B) 1791", "organizer_letter").ok


def test_organizer_letter_training_example():
    ex = build_training_example(MC3, "organizer_letter", system_prompt="Jesteś asystentem.")
    user = ex["prompt"][-1]["content"]
    assert ex["prompt"][0] == {"role": "system", "content": "Jesteś asystentem."}
    assert "B. dwie" in user and "Odpowiedz wyłącznie literą A, B albo C." in user
    assert ex["completion"] == [{"role": "assistant", "content": "B"}]
    assert "Odpowiedź" not in user and "Odpowiedź" not in ex["completion"][0]["content"]


def test_letters_phrase_matches_the_organizer_scorer():
    """Same enumeration wording as eval/organizers.letter_list (skipped until that module lands)."""
    organizers = pytest.importorskip("eval.organizers")
    for letters in ("ABC", "ABCD"):
        assert letters_phrase(letters) == organizers.letter_list(letters)
