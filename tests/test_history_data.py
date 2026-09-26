"""History matura data builder (train/history_data.py): cleaner, picture detection, exclusions, splits, targets.

Inline fixtures only (shaped like nuori-ai tasks.jsonl rows); no CKE content is stored in the repo.
"""
from __future__ import annotations

import json

import pytest

from train import history_data as H

# ----------------------------------------------------------------------------- fixtures

KEY_TEMPLATE = (
    "Wymagania egzaminacyjne 2023 i 2024\nWymaganie ogólne\nII. Analiza i interpretacja historyczna.\n\n"
    "Zasady oceniania\n{rubric}\n0 pkt – za odpowiedź niepełną lub błędną albo za brak odpowiedzi.\n"
    "Rozwiązanie\n{solution}\n"
)


def key(solution: str, rubric: str = "1 pkt – za prawidłową odpowiedź.") -> str:
    return KEY_TEMPLATE.format(rubric=rubric, solution=solution)


def row(arkusz: str, year: int, task: str, question: str, answer_key: str | None, context: str = "",
        max_points: int | None = 1) -> dict:
    return {"id": f"{year}_rozszerzona_{arkusz}_{task}", "year": str(year), "level": "rozszerzona",
            "arkusz": arkusz, "task": task, "max_points": max_points, "context": context, "question": question,
            "needs_image": False, "page": 1, "answer_key": answer_key}


TEXT_SOURCE = ("Źródło 1. Fragment opracowania historycznego\n"
               "Po śmierci Bolesława Krzywoustego jego synowie objęli dzielnice, a senior sprawował władzę zwierzchnią\n"
               "nad pozostałymi książętami. Z czasem zasada senioratu przestała być przestrzegana, a kraj uległ\n"
               "dalszemu podziałowi na coraz mniejsze księstwa.\n"
               "J. Kowalski, Dzieje Polski, Warszawa 2001, s. 12.\n\n")
Q_CHOICE = ("Dokończ zdanie. Zaznacz właściwą odpowiedź spośród podanych.\nOpisany okres nazywamy\n"
            "A. monarchią patrymonialną.\nB. rozbiciem dzielnicowym.\nC. monarchią stanową.\nD. demokracją szlachecką.")
Q_TF = ("Oceń prawdziwość poniższych stwierdzeń. Zaznacz P, jeśli stwierdzenie jest\nprawdziwe, albo F – jeśli jest fałszywe.\n"
        "1.\n\nW źródle opisano skutki testamentu Bolesława Krzywoustego.\n\nP\n\nF\n\n"
        "2.\n\nSenior rezydował w Gnieźnie.\n\nP\n\nF\n\n"
        "3.\n\nPodział trwał do koronacji Władysława Łokietka.\n\nP\n\nF")
Q_OPEN = ("Rozstrzygnij, czy zasada senioratu była przestrzegana do końca XII wieku. Odpowiedź uzasadnij.\n"
          "Rozstrzygnięcie: ..............................\n"
          "Uzasadnienie: ............................................................................................")


def paper(arkusz: str, year: int, suffix: str = "") -> list[dict]:
    """A small 3-task paper; `suffix` makes its questions distinct from other papers."""
    return [
        row(arkusz, year, "1", TEXT_SOURCE + "1.\n0–1\n\n" + Q_CHOICE + suffix, key("B")),
        row(arkusz, year, "2", TEXT_SOURCE + "2.\n0–1–2\n\n" + Q_TF + suffix, key("1–P\n2–F\n3–P", "2 pkt – za trzy.")),
        row(arkusz, year, "3", TEXT_SOURCE + Q_OPEN + suffix,
            key("Rozstrzygnięcie: Nie\nPrzykładowe uzasadnienie:\nKsiążęta dzielnicowi walczyli o tron krakowski,\n"
                "a zasada senioratu została złamana już w 1146 roku.")),
    ]


def fake_build_messages(item, image_descriptions=None, system_prompt="SYS", style="organizer"):
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Zadanie {item['id']} ({item['max_points']} pkt)\n\n{item['source_text']}\n\n{item['question']}"}]


def build(rows, **kw):
    kw.setdefault("build_messages", fake_build_messages)
    stats = H.build_history_data(rows, None, **kw, stats_only=False)
    return stats


# ----------------------------------------------------------------------------- cleaner

def test_clean_layout_strips_exam_sheet_junk():
    raw = ("Strona 3 z 32\nWięcej arkuszy znajdziesz na stronie: arkusze.pl\nMHIP-R0-100-2305\nMHI_1R\n"
           "Egzamin maturalny z historii\nPoziom rozszerzony\nWypełnia egzaminator\nNr zadania\n1.1.\n"
           "Maks. liczba pkt\n1\nUzyskana liczba pkt\nMiejsce na naklejkę z numerem PESEL\nBRUDNOPIS\n(nie podlega ocenie)\n"
           "16 z 44\n(1 pkt)\n"
           "Źródło 1. Tekst […] prawdziwy.\n5.1.\n0–1\n"
           "Rozstrzygnięcie: ..................\nUzasadnienie: ........................................\n"
           "•\n\n•\n25.\n1\n\n26.\n2")
    out = H.clean_layout(raw)
    assert out == "Źródło 1. Tekst […] prawdziwy.\nRozstrzygnięcie:\nUzasadnienie:"


def test_clean_layout_splits_labels_and_joins_enumerators():
    assert H.clean_layout("Nazwa wydarzenia: ........................ Rok: ..........") == "Nazwa wydarzenia:\nRok:"
    assert H.clean_layout("Fragment B – ............................") == "Fragment B:"
    assert H.clean_layout("1.\n\nWyspa została przyłączona\nw wyniku wojny.") == "1. Wyspa została przyłączona\nw wyniku wojny."
    zipped = H.clean_layout("A.\nB.\nC.\nD. o Narvik.\no Arnhem.\npod Falaise.\npod Monte Cassino.")
    assert zipped.split("\n") == ["A. o Narvik.", "B. o Arnhem.", "C. pod Falaise.", "D. pod Monte Cassino."]
    assert H.clean_layout("0–1–\n2–3\nTekst") == "Tekst"


def test_split_source_question():
    body = H.clean_layout(TEXT_SOURCE + "1.\n0–1\n\n" + Q_CHOICE)
    src, q, trailing = H.split_source_question(body)
    assert src.startswith("Źródło 1.") and "Dzieje Polski" in src
    assert q.startswith("Dokończ zdanie.") and "D. demokracją szlachecką." in q
    assert trailing == ""
    # formula-2015 intro line is not the instruction; a trailing "Źródło B" belongs to the next tasks
    body2 = ("Na podstawie źródła A i własnej wiedzy wykonaj polecenia.\nTekst źródła A o Rzymie i jego urzędach.\n"
             "Podaj nazwę urzędu.\nŹródło B\nTekst źródła B.")
    src2, q2, tr2 = H.split_source_question(body2)
    assert src2 == "Tekst źródła A o Rzymie i jego urzędach." and q2 == "Podaj nazwę urzędu." and tr2.startswith("Źródło B")


# ----------------------------------------------------------------------------- picture dependence

MAP_SOURCE = TEXT_SOURCE + "Źródło 2. Mapa. Polska dzielnicowa\n\nA\nB\n1\n2\n\nNa podstawie: Atlas historyczny, Warszawa 2005, s. 12."


@pytest.mark.parametrize("question", [
    "Podaj nazwę dzielnicy oznaczonej na mapie literą A.",
    "Wyjaśnij, co przedstawia ilustracja.",
    "Podaj nazwisko postaci ukazanej na fotografii.",
    "Wyjaśnij wymowę karykatury.",
    "Rozstrzygnij, czy na plakacie przedstawiono robotnika.",
    "Wyjaśnij przesłanie rysunku, interpretując jego elementy graficzne.",
    "Podaj nazwisko władcy przedstawionego na monecie.",
    "Na podstawie schematu podaj nazwę systemu.",
    "Odczytaj z wykresu rok największego eksportu.",
    "Wyjaśnij treść drzeworytu.",
    "Rozstrzygnij, czy źródło 2. przedstawia ziemie Bolesława Kędzierzawego. Odpowiedź uzasadnij.",
    "Rozstrzygnij, czy oba źródła dotyczą tego samego okresu.",
])
def test_picture_dependent(question):
    pc = H.picture_check(question, H.clean_layout(MAP_SOURCE))
    assert pc.dependent, question


def test_picture_unused_is_kept_and_flagged():
    pc = H.picture_check("Podaj imię księcia, który jako senior objął dzielnicę krakowską po 1138 roku.",
                         H.clean_layout(MAP_SOURCE))
    assert not pc.dependent and pc.flags == ["visual_source_unused"]
    # "planu Marshalla" / "w tabeli" as answer scaffold are not pictures
    assert not H.picture_check("Wyjaśnij cele planu Marshalla.", TEXT_SOURCE).dependent
    assert not H.picture_check("Uzupełnij tabelę – wpisz nazwy dynastii.", TEXT_SOURCE).dependent


def test_thin_source_and_picture_labels():
    thin = "Jan Nowak, Wczoraj i dziś, 1950 r.\n\nNa podstawie: Satyra z lat 1945–1989, Warszawa 2001, s. 10."
    assert H.picture_check("Wyjaśnij, odwołując się do źródła, wizję emancypacji kobiet.", thin).dependent
    scheme = "Źródło 1. Schemat\n\nNa podstawie: http://example.pl\n\n" + TEXT_SOURCE.replace("Źródło 1.", "Źródło 2.")
    assert H.picture_check("Rozstrzygnij, na którym etapie (A, B czy C) była opisana społeczność.", scheme).dependent


def test_mark_images_inserts_organizer_style_marker():
    out = H.mark_images(H.clean_layout(MAP_SOURCE), "4")
    assert "[Obraz: images/Z04-S2.png]" in out
    assert "\nA\n" not in out and "Na podstawie: Atlas historyczny" in out


# ----------------------------------------------------------------------------- key parsing and targets

def test_parse_key_separates_rubric_and_solution():
    k = H.parse_key(key("Rozstrzygnięcie: Tak\nPrzykładowe uzasadnienie:\nTekst.", "1 pkt – za rozstrzygnięcie\nwraz z uzasadnieniem."))
    assert k.rubric.startswith("1 pkt – za rozstrzygnięcie") and "0 pkt" in k.rubric
    assert k.solution.startswith("Rozstrzygnięcie: Tak") and "Wymagania" not in k.solution
    assert k.max_points == 1
    old = H.parse_key("Schemat punktowania\n1 p. – za poprawną odpowiedź.\n0 p. – za błędną.\nPoprawna odpowiedź\nB")
    assert old.solution == "B"


def test_single_letter_answer_is_kept():
    item = {"question": Q_CHOICE, "max_points": 1, "answer_format": H.OPEN_FORMAT}
    res = H.make_target(item, H.parse_key(key("B")))
    assert res.status == "ok" and res.kind == "closed_single" and res.target == "B"
    assert item["answer_format"] == "A"


@pytest.mark.parametrize("solution,expected", [("1–F\n2–P\n3–P", "1: F\n2: P\n3: P"),
                                               ("1. – F; 2. – P; 3. – P.", "1: F\n2: P\n3: P"),
                                               ("FPP", "1: F\n2: P\n3: P")])
def test_true_false_rendering(solution, expected):
    item = {"question": H.clean_layout(Q_TF), "max_points": 2, "answer_format": H.OPEN_FORMAT}
    res = H.make_target(item, H.parse_key(key(solution)))
    assert (res.kind, res.target) == ("closed_tf", expected)
    assert item["answer_format"] == "1: P\n2: F\n3: P"
    assert item["question"].endswith("3. Podział trwał do koronacji Władysława Łokietka.")
    assert "\nP\n" not in item["question"]


def test_match_and_multi_part_rendering():
    q_match = "Przyporządkuj opisom numery fragmentów.\nA. Opis pierwszy.\nB. Opis drugi."
    c = H.detect_closed(q_match, "A–3\nB–2")
    assert (c.kind, c.answer_format, c.target) == ("closed_match", "A: 1\nB: 1", "A: 3\nB: 2")
    q_multi = ("Dokończ zdania 1. i 2. Zaznacz właściwą odpowiedź spośród podanych.\n1. Pierwsze\nA. a.\nB. b.\n"
               "2. Drugie\nA. c.\nB. d.")
    c = H.detect_closed(q_multi, "1–B\n2–A")
    assert (c.kind, c.answer_format, c.target) == ("closed_multi_part", "1: A\n2: A", "1: B\n2: A")
    assert H.render_closed("closed_single", "C") == "C"


def test_open_target_uses_question_labels_and_resolves_variants():
    item = {"question": H.clean_layout(Q_OPEN), "max_points": 1, "answer_format": H.OPEN_FORMAT}
    sol = ("Rozstrzygnięcie: Nie\nPrzykładowe uzasadnienia:\n• Książęta walczyli o tron krakowski [pryncypat],\n"
           "co złamało zasadę.\n• Drugi wariant.")
    res = H.make_target(item, H.parse_key(key(sol)))
    assert res.status == "ok" and res.kind == "open"
    assert res.target == "Rozstrzygnięcie: Nie\nUzasadnienie: Książęta walczyli o tron krakowski, co złamało zasadę."
    assert H.resolve_alternatives("unia lubelska [unia realna, unia z 1569 r.]") == "unia lubelska"
    assert H.resolve_alternatives("Bizancjum / Cesarstwo Bizantyjskie / Cesarstwo wschodniorzymskie") == "Bizancjum"
    assert H.resolve_alternatives("[konstytucja] nihil novi / przywilej radomski") == "konstytucja nihil novi"
    assert H.resolve_alternatives("Nazwa: bitwa warszawska / cud nad Wisłą") == "Nazwa: bitwa warszawska"
    assert (H.resolve_alternatives("Najechali ją Turcy / Arabowie, a potem Mongołowie.")
            == "Najechali ją Turcy, a potem Mongołowie.")
    two = H.build_open_target("Podaj dwa argumenty potwierdzające tezę.", "• Pierwszy argument;\n• Drugi argument;\n• Trzeci.")
    assert two.target == "1. Pierwszy argument\n2. Drugi argument"
    names = H.build_open_target("Wpisz obok biogramów nazwiska postaci.", "A. [Aleksander] Wielopolski\nB. [Romuald] Traugutt")
    assert names.target == "A: Aleksander Wielopolski\nB: Romuald Traugutt"


def test_rubric_only_and_essay_go_to_queue():
    item = {"question": "Wyjaśnij przyczyny upadku senioratu.", "max_points": 2, "answer_format": H.OPEN_FORMAT}
    res = H.make_target(item, H.parse_key(KEY_TEMPLATE.format(rubric="2 pkt – za pełne wyjaśnienie.", solution="")))
    assert (res.status, res.reason) == ("needs_answer", "rubric_only")
    essay = {"question": "Zadanie zawiera trzy tematy. Wybierz jeden z nich do opracowania.\n1. Temat A.\n2. Temat B.",
             "max_points": 15, "answer_format": H.OPEN_FORMAT}
    res = H.make_target(essay, H.parse_key(key("")))
    assert (res.status, res.kind, res.reason) == ("needs_answer", "essay", "essay")
    assert essay["answer_format"] == H.ESSAY_FORMAT


# ----------------------------------------------------------------------------- papers, exclusions, splits

def test_paper_ids():
    assert H.paper_id("2024/MHIP-R0-100-A-2405-arkusz.pdf") == "MHIP-R0-100-2405"
    assert H.paper_id("2023/EHIP-R0-100-2305.pdf") == "EHIP-R0-100-2305"
    assert H.paper_id("2016/MHI-R1_1P-162.pdf") == "MHI-R1-162"
    assert H.paper_id("X/historia-2016-maj-matura-stara-rozszerzona (1).pdf") == "historia-2016-maj-matura-stara-rozszerzona"


def test_mock_paper_and_its_copies_are_excluded_everywhere():
    rows = (paper("2023/MHIP-R0-100-2305.pdf", 2023, " [mock]")
            + paper("H/historia-2023-maj-matura-rozszerzona.pdf", 2023, " [mock]")      # arkusze.pl copy of the mock
            + paper("2023/EHIP-R0-100-2305.pdf", 2023, " [sibling]")
            + paper("H/historia-2022-maj-matura-rozszerzona.pdf", 2022, " [2022]"))
    stats = build(rows)
    assert stats["drop"]["mock_paper"] == 9
    kept = _written(stats) + stats["_kept"]["needs"]
    assert kept and all("[2022]" in r["item"]["question"] for r in kept)
    assert len(stats["papers"]["excluded_mock_files"]) == 3


def test_splits_by_year_dedup_and_single_letters(tmp_path):
    rows = (paper("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024, " [24]")
            + paper("H/historia-2024-maj-matura-rozszerzona.pdf", 2024, " [24]")     # identical copy -> dropped
            + paper("2025/MHIP-R0-100-A-2505-arkusz.pdf", 2025, " [25]")
            + paper("H/historia-2014-maj-matura-rozszerzona.pdf", 2014, " [14]")
            + [row("H/historia-2024-czerwiec-matura-rozszerzona.pdf", 2024, "9", TEXT_SOURCE + Q_CHOICE + " [25]",
                   key("B"))])                                                       # same item as the 2025 dev one
    stats = H.build_history_data(rows, tmp_path, build_messages=fake_build_messages)
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    dev = [json.loads(l) for l in (tmp_path / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["year"] for r in train} == {2024} and {r["year"] for r in dev} == {2025}
    assert len(train) == 3 and len(dev) == 3
    assert stats["drop"]["duplicate_paper_copy"] == 3
    assert stats["drop"]["pre_min_year"] == 3
    assert stats["drop"]["duplicate_of_dev"] == 1
    single = [r for r in train if r["kind"] == "closed_single"]
    assert single and single[0]["completion"][0]["content"] == "B"
    for r in train + dev:
        assert set(r) >= {"prompt", "completion", "kind", "id", "paper", "year", "points", "rubric"}
        assert r["prompt"][-1]["role"] == "user" and r["completion"][0]["role"] == "assistant"
        assert r["rubric"]
    assert (tmp_path / "blocklist_ids.json").exists() and (tmp_path / "stats.json").exists()


def test_filled_answers_are_merged():
    essay = row("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024, "26",
                "Zadanie zawiera trzy tematy. Wybierz jeden z nich do opracowania.\n1. Temat A.\n2. Temat B.",
                key(""), max_points=15)
    stats = build([essay])
    assert stats["needs_answer"] == {"essay": 1}
    short = build([essay], filled_answers={"MHIP-R0-100-2405:26": "Temat 1.\n\nTekst wypracowania."})
    assert not _written(short) and short["filled"] == {"rejected:essay_short": 1}      # < 300 words: never a target
    stats = build([essay], filled_answers={"MHIP-R0-100-2405:26": ESSAY_TEXT})
    rows = _written(stats)
    assert rows[0]["kind"] == "essay" and "llm_filled" in rows[0]["flags"] and rows[0]["origin"] == "llm-filled"
    assert stats["per_origin"] == {"llm-filled": {"train": 1}}


def test_prompts_come_from_the_harness_and_kinds_agree():
    prompts = pytest.importorskip("harness.prompts")
    stats = H.build_history_data(paper("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024), None,
                                 build_messages=prompts.build_messages, item_kind=prompts.item_kind)
    rows = _written(stats)
    assert stats["item_kind_mismatch"] == 0 and len(rows) == 3
    assert rows[0]["prompt"][0]["content"] == prompts.ORGANIZER_SYSTEM_PROMPT
    assert {r["kind"] for r in rows} == {"closed_single", "closed_tf", "open"}


def test_prompt_style_follows_the_serve_time_style():
    prompts = pytest.importorskip("harness.prompts")
    rows = paper("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024)
    organizer = H.build_history_data(rows, None, build_messages=prompts.build_messages, item_kind=prompts.item_kind)
    formatted = H.build_history_data(rows, None, build_messages=prompts.build_messages, item_kind=prompts.item_kind,
                                     style="formatted")
    assert organizer["prompt_style"] == "organizer" and formatted["prompt_style"] == "formatted"
    for o, f in zip(_written(organizer), _written(formatted)):
        assert o["prompt"] == prompts.build_messages(o["item"], None, style="organizer")
        assert f["prompt"] == prompts.build_messages(f["item"], None, style="formatted")
        assert f["prompt"][-1]["content"].startswith(o["prompt"][-1]["content"].rstrip())
        assert o["completion"] == f["completion"]


def _written(stats: dict) -> list[dict]:
    """Rows kept in memory by build_history_data when out_dir is None."""
    return stats["_kept"]["train"] + stats["_kept"]["dev"]


# ----------------------------------------------------------------------------- essays (fixture)

ESSAY_TEXT = """Temat 1.

Konstytucja 3 maja, uchwalona w 1791 roku przez Sejm Czteroletni, była próbą ratowania Rzeczypospolitej przed upadkiem. \
Uważam, że stanowiła najważniejszą reformę ustrojową XVIII wieku w Polsce, choć nie zdołała zapobiec rozbiorom.

Po pierwszym rozbiorze w 1772 roku stało się jasne, że państwo bez silnego rządu i sprawnej armii nie obroni swoich granic. \
Król Stanisław August Poniatowski oraz stronnictwo patriotyczne, do którego należeli między innymi Ignacy Potocki i Hugo \
Kołłątaj, wykorzystali korzystną sytuację międzynarodową, gdy Rosja prowadziła wojnę z Turcją. Sejm zwołany w 1788 roku \
zawiązał się w konfederację, dzięki czemu obradował bez groźby zerwania przez liberum veto. Uchwalił aukcję wojska do stu \
tysięcy żołnierzy, ustawę o sejmikach, która odsunęła od udziału w nich gołotę, oraz prawo o miastach królewskich, dające \
mieszczanom bezpieczeństwo osobiste i możliwość nabywania dóbr ziemskich.

Sama konstytucja wprowadziła trójpodział władzy. Władzę ustawodawczą sprawował dwuizbowy sejm, w którym decydowała \
większość głosów, a liberum veto i konfederacje zostały zniesione. Władzę wykonawczą powierzono królowi i Straży Praw, \
czyli radzie złożonej z prymasa i ministrów. Wolną elekcję zastąpiono tronem dziedzicznym, który po śmierci Stanisława \
Augusta miał przejść na dynastię saską. Chłopów wzięto pod opiekę prawa i rządu krajowego, co nie znosiło pańszczyzny, \
ale zapowiadało dalsze zmiany. Konstytucja zachowała jednak dominującą pozycję szlachty, a katolicyzm pozostał religią \
panującą przy zachowaniu tolerancji dla innych wyznań.

Reformy spotkały się z oporem magnatów przeciwnych zmianom. W 1792 roku zawiązali oni konfederację targowicką \
i poprosili o pomoc carycę Katarzynę II. Wojna w obronie konstytucji zakończyła się przystąpieniem króla do Targowicy, \
a w 1793 roku Rosja i Prusy dokonały drugiego rozbioru. Sejm grodzieński unieważnił dzieło Sejmu Wielkiego.

Podsumowując, Konstytucja 3 maja była nowoczesnym aktem, który usuwał najgroźniejsze wady ustroju szlacheckiego \
i wzmacniał państwo. Jej upadek wynikał nie z błędów samej ustawy, lecz z przewagi sąsiednich mocarstw i zdrady części \
elit. Pamięć o niej stała się jednak ważnym elementem tradycji narodowej w okresie niewoli, a rocznicę jej uchwalenia \
obchodzono jako święto narodowe w II Rzeczypospolitej i ponownie od 1990 roku."""


# ----------------------------------------------------------------------------- pre-2015 papers

OLD_Q = ("(1 pkt)\nPrzeczytaj fragment tekstu i wykonaj polecenie.\n"
         "Na sejmie w Radomiu postanowiono, że odtąd nic nowego nie może być uchwalone bez wspólnej zgody senatu\n"
         "i posłów ziemskich, co zapewniło izbie poselskiej udział w stanowieniu prawa.\n"
         "Źródło: J. Kowalski, Dzieje sejmu, Warszawa 2001, s. 10.\n\n"
         "Podaj nazwę aktu prawnego, o którym mowa w tekście. .......................................\n\n"
         "Wypełnia Maks. liczba pkt\negzaminator!\n\n12.\n1\n\n13.A.\n1\n\nEgzamin maturalny z historii\n"
         "Poziom rozszerzony\n\n7")
OLD_KEY = ("Korzystanie z informacji\n\nRozpoznanie aktu prawnego (II 1)\n\nPoprawna odpowiedź:\nkonstytucja nihil novi\n"
           "1 p. – za podanie nazwy aktu prawnego\n0 p. – za błędną odpowiedź lub brak odpowiedzi\n\n5\n\n"
           "Egzamin maturalny z historii\nKryteria oceniania odpowiedzi – poziom rozszerzony")
OLD_SHEET_Q = "(1 pkt)\nPodaj imię i przydomek władcy, który w 1374 r. wydał przywilej koszycki. ..............................\n"
OLD_SHEET_KEY = "(1 pkt)\nPodaj imię i przydomek władcy, który w 1374 r. wydał przywilej koszycki. Ludwik Węgierski\n"


def old_paper() -> list[dict]:
    return [
        row("H/historia-2012-maj-matura-rozszerzona.pdf", 2012, "12", OLD_Q, OLD_KEY, max_points=None),
        row("H/historia-2012-maj-matura-rozszerzona.pdf", 2012, "13", OLD_SHEET_Q, OLD_SHEET_KEY, max_points=None),
        row("H/historia-2012-maj-matura-rozszerzona.pdf", 2012, "14",
            "Podaj nazwę bitwy stoczonej w 1410 roku przez wojska polsko-litewskie z Krzyżakami.", None,
            max_points=None),
    ]


def test_clean_old_layout_and_old_keys():
    q = H.clean_layout(H.clean_old_layout(OLD_Q))
    assert "Wypełnia" not in q and "egzaminator" not in q and "13.A." not in q and not q.rstrip().endswith("7")
    assert q.endswith("Podaj nazwę aktu prawnego, o którym mowa w tekście.")
    k = H.parse_old_key(OLD_KEY)
    assert k.solution == "konstytucja nihil novi" and k.max_points == 1 and k.rubric.startswith("1 p. – za podanie")
    parts = H.parse_old_key("A. (0–1)\nKorzystanie z informacji\nPoprawna odpowiedź:\nKazimierz Wielki\n"
                            "1 p. – za podanie imienia władcy\nB. (0–1)\nPoprawna odpowiedź:\n1364\n1 p. – za podanie roku")
    assert parts.solution == "A. Kazimierz Wielki\nB. 1364" and parts.max_points == 2
    assert H.parse_old_key(OLD_SHEET_KEY, OLD_SHEET_Q).solution == "Ludwik Węgierski"      # filled-in sheet
    assert H.renumber_old_topics("Zadanie zawiera dwa tematy.\nTemat I\nPrzedstaw A.\n\nTemat II\nPrzedstaw B.") == \
        "Zadanie zawiera dwa tematy.\n1. Przedstaw A.\n\n2. Przedstaw B."
    assert H.join_word_runs("Uporządkuj\nchronologicznie\nwydarzenia\nz\nhistorii.\nrepublika\nkonsulat\ncesarstwo\ndyrektoriat") == \
        "Uporządkuj chronologicznie wydarzenia z historii.\nrepublika\nkonsulat\ncesarstwo\ndyrektoriat"


def test_old_rows_are_trained_only_with_a_filled_target():
    new = paper("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024)
    default = build(new + old_paper())
    assert default["drop"]["pre_min_year"] == 3 and default["min_year"] == 2015 and not default["include_old"]

    queued = build(new + old_paper(), min_year=H.OLD_MIN_YEAR)
    assert queued["needs_answer"].get("old_paper") == 2 and queued["drop"]["no_answer_key"] == 1
    needs = {n["id"]: n for n in queued["_kept"]["needs"]}
    first = needs["historia-2012-maj-matura-rozszerzona:12"]
    assert first["origin"] == "cke-old" and first["auto_target"] == "konstytucja nihil novi"
    assert first["item"]["max_points"] == 1 and first["item"]["source_text"].startswith("Przeczytaj fragment")
    assert first["item"]["question"] == "Podaj nazwę aktu prawnego, o którym mowa w tekście."
    assert needs["historia-2012-maj-matura-rozszerzona:13"]["key_solution"] == "Ludwik Węgierski"

    filled = {"historia-2012-maj-matura-rozszerzona:12": "konstytucja nihil novi",
              "historia-2012-maj-matura-rozszerzona:13": "SKIP",                   # never read as a target
              "historia-2012-maj-matura-rozszerzona:14": "bitwa pod Grunwaldem"}  # keyless: trained only if allowed
    filled = {k: v for k, v in filled.items() if v != "SKIP"}
    stats = build(new + old_paper(), min_year=H.OLD_MIN_YEAR, filled_answers=filled, keyless_filled="none")
    old = [r for r in _written(stats) if r["origin"] == "cke-old"]
    assert [r["id"] for r in old] == ["historia-2012-maj-matura-rozszerzona:12"]
    assert old[0]["flags"] == ["llm_filled_old"] and old[0]["completion"][0]["content"] == "konstytucja nihil novi"
    assert old[0]["prompt"][-1]["content"].startswith("Zadanie 12 (1 pkt)")
    assert stats["filled"] == {"used_old": 1, "ignored_keyless": 1}
    assert stats["per_origin"]["cke-old"] == {"train": 1} and stats["per_origin"]["cke-2015+"] == {"train": 3}
    keyless = build(new + old_paper(), min_year=H.OLD_MIN_YEAR, filled_answers=filled)       # default: "old"
    kept = {r["id"]: r for r in _written(keyless) if r["origin"] == "cke-old"}
    assert set(kept) == {"historia-2012-maj-matura-rozszerzona:12", "historia-2012-maj-matura-rozszerzona:14"}
    grunwald = kept["historia-2012-maj-matura-rozszerzona:14"]
    assert grunwald["flags"] == ["llm_filled_old", "keyless", "points_estimated"] and grunwald["points"] == 1
    assert grunwald["prompt"][-1]["content"].startswith("Zadanie 14 (1 pkt)")
    # the 2015+ rows are the same with or without the old papers
    assert [(r["id"], r["prompt"], r["completion"]) for r in _written(default)] == \
        [(r["id"], r["prompt"], r["completion"]) for r in _written(stats) if r["origin"] == "cke-2015+"]


def test_load_filled_files_merges_files_and_skips(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id": "x", "answer": "A"}\n{"id": "y", "answer": "SKIP"}\n', encoding="utf-8")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b.jsonl").write_text('{"id": "x", "answer": "B"}\n{"id": "z", "completion": [{"content": "C"}]}\n',
                                            encoding="utf-8")
    answers, info = H.load_filled_files(["a.jsonl", "d/*.jsonl", "missing.jsonl"], root=tmp_path)
    assert answers == {"x": "B", "z": "C"} and info["skipped"] == 1 and info["overridden"] == 1


def test_broken_encoding_papers_are_dropped():
    broken = [row("H/historia-2011-sierpien-poprawkowa-podstawowa.pdf", 2011, str(i),
                  "SpoĞród wymienionych wydarzeĔ wybierz zwyciĊstwo Rzymian.", "Poprawna odpowiedź:\nA") for i in (1, 2)]
    stats = build(broken, min_year=H.OLD_MIN_YEAR)
    assert stats["drop"] == {"broken_encoding": 2}


def test_answer_problem():
    tf = {"question": "Oceń prawdziwość.\n1. Jedno.\n2. Drugie.", "answer_format": "1: P\n2: F"}
    assert H.answer_problem("closed_tf", tf, "1: F\n2: P") is None
    assert H.answer_problem("closed_tf", tf, "1: F") == "closed_keys"
    assert H.answer_problem("closed_tf", tf, "1 – F\n2 – P") == "closed_syntax"
    single = {"question": Q_CHOICE, "answer_format": "A"}
    assert H.answer_problem("closed_single", single, "B") is None
    assert H.answer_problem("closed_single", single, "E") == "closed_option"
    assert H.answer_problem("closed_single", single, "B. rozbiciem") == "closed_syntax"
    opn = {"question": H.clean_layout(Q_OPEN), "answer_format": H.OPEN_FORMAT}
    assert H.answer_problem("open", opn, "Rozstrzygnięcie: Nie\nUzasadnienie: Już w 1146 r. wygnano seniora.") is None
    assert H.answer_problem("open", opn, "Nie, bo wygnano seniora.") == "labels_missing"
    assert H.answer_problem("open", opn, "Nie, bo wygnano seniora.", strict_labels=False) is None
    essay = {"question": "Zadanie zawiera dwa tematy.\n1. Temat A.\n2. Temat B.", "answer_format": H.ESSAY_FORMAT}
    assert H.answer_problem("essay", essay, ESSAY_TEXT) is None
    assert H.answer_problem("essay", essay, ESSAY_TEXT.replace("Temat 1.", "Temat 3.")) == "essay_topic"
    assert H.answer_problem("essay", essay, ESSAY_TEXT.replace("Temat 1.", "Wypracowanie")) == "essay_header"


# ----------------------------------------------------------------------------- synthetic items

YEAR_FACTS = [("bitwa pod Grunwaldem", "1410"), ("chrzest Mieszka I", "966"), ("zjazd gnieźnieński", "1000"),
              ("unia lubelska", "1569"), ("hołd pruski", "1525"), ("odsiecz wiedeńska", "1683"),
              ("uchwalenie Konstytucji 3 maja", "1791"), ("wybuch powstania listopadowego", "1830"),
              ("wybuch powstania styczniowego", "1863"), ("bitwa warszawska", "1920"),
              ("koronacja Bolesława Chrobrego", "1025"), ("drugi pokój toruński", "1466"),
              ("konfederacja warszawska", "1573"), ("sejm niemy", "1717"), ("pierwszy rozbiór Polski", "1772"),
              ("utworzenie Księstwa Warszawskiego", "1807"), ("założenie Akademii Krakowskiej", "1364"),
              ("wydanie przywileju koszyckiego", "1374"), ("zawarcie unii w Krewie", "1385"), ("hołd lenny Brandenburgii", "1611")]


def syn_open(n: int, fact: tuple[str, str], shard: str = "t1") -> dict:
    return {"id": f"syn-open-{shard}-{n}", "kind": "open", "question": f"Podaj rok wydarzenia: {fact[0]}.",
            "source_text": "", "answer_format": H.OPEN_FORMAT, "max_points": 1, "target": fact[1],
            "origin": "synthetic-open", "source_urls": ["https://pl.wikipedia.org/wiki/Historia_Polski"]}


MOCK_LIKE_SOURCE = ("Źródło 1. Fragment dokumentu\nMy, król, wraz ze stanami skonfederowanymi postanawiamy, że wszelka władza\n"
                    "społeczności ludzkiej początek swój bierze z woli narodu, a rząd ma być złożony z trzech władz.\n"
                    "Na podstawie: Konstytucja 3 maja, Warszawa 1991, s. 5.\n\n")


def synthetic_records() -> list[dict]:
    q_tf = ("Oceń prawdziwość podanych stwierdzeń. Zaznacz P, jeśli stwierdzenie jest prawdziwe, albo F – jeśli jest fałszywe.\n"
            "1. Unia lubelska została zawarta w 1569 roku.\n2. Hołd pruski złożono w Krakowie w 1410 roku.")
    return [
        {**syn_open(1, YEAR_FACTS[0]), "question": "Rozstrzygnij, czy bitwa pod Grunwaldem zakończyła wielką wojnę z "
         "zakonem krzyżackim. Odpowiedź uzasadnij.\nRozstrzygnięcie:\nUzasadnienie:",
         "target": "Rozstrzygnięcie: Nie\nUzasadnienie: Wojnę zakończył dopiero pierwszy pokój toruński w 1411 roku.",
         "max_points": 2},
        {"id": "syn-closed_tf-t1-1", "kind": "closed_tf", "question": q_tf, "source_text": "", "answer_format": "1: P\n2: F",
         "max_points": 1, "target": "1: P\n2: F", "origin": "synthetic-open", "source_urls": []},
        {"id": "syn-essay-t1-1", "kind": "essay", "question": "Zadanie zawiera dwa tematy. Wybierz jeden z nich do "
         "opracowania.\n1. Oceń reformy Sejmu Czteroletniego.\n2. Scharakteryzuj politykę zagraniczną Władysława IV.",
         "source_text": "", "answer_format": H.ESSAY_FORMAT, "max_points": 15, "target": ESSAY_TEXT,
         "origin": "synthetic-essay", "source_urls": []},
        # invalid: open answer_format must be exact; closed target must follow answer_format; unknown kind
        {**syn_open(2, YEAR_FACTS[1]), "answer_format": "Tekst."},
        {"id": "syn-closed_single-t1-1", "kind": "closed_single", "question": Q_CHOICE, "source_text": "",
         "answer_format": "A", "max_points": 1, "target": "B – rozbicie", "origin": "synthetic-open", "source_urls": []},
        {**syn_open(3, YEAR_FACTS[2]), "kind": "table"},
        # duplicates: of a CKE dev item, of the mock paper, of the mock's essay theme, of an earlier synthetic item
        {**syn_open(4, YEAR_FACTS[3]), "question": H.clean_layout(Q_OPEN), "source_text": H.clean_layout(TEXT_SOURCE),
         "target": "Rozstrzygnięcie: Nie\nUzasadnienie: Zasada senioratu została złamana już w 1146 roku."},
        {**syn_open(5, YEAR_FACTS[4]), "question": "Wyjaśnij, jaką zasadę ustrojową wyraża przytoczony fragment.",
         "source_text": H.clean_layout(MOCK_LIKE_SOURCE), "target": "Zasadę suwerenności narodu i trójpodziału władzy."},
        {"id": "syn-essay-t1-2", "kind": "essay", "question": "Zadanie zawiera dwa tematy. Wybierz jeden z nich do "
         "opracowania.\n1. Wyjaśnij przyczyny rozbicia dzielnicowego Polski w XII wieku.\n2. Oceń panowanie Kazimierza Wielkiego.",
         "source_text": "", "answer_format": H.ESSAY_FORMAT, "max_points": 15, "target": ESSAY_TEXT,
         "origin": "synthetic-essay", "source_urls": []},
        {**syn_open(6, YEAR_FACTS[0]), "id": "syn-open-t2-9"},
        {**syn_open(6, YEAR_FACTS[0]), "id": "syn-open-t2-10"},
        {**syn_open(7, YEAR_FACTS[5]), "id": "syn-open-t2-9"},                  # duplicate id of the one above
    ]


def test_synthetic_items_are_validated_deduplicated_and_rendered(tmp_path):
    recs = synthetic_records()
    d = tmp_path / "synthetic"
    d.mkdir()
    (d / "shard-a.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs[:6]) + "\nnot json\n",
                                     encoding="utf-8")
    (d / "shard-b.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs[6:]) + "\n", encoding="utf-8")
    loaded, info = H.load_synthetic_files([str(d / "*.jsonl")])
    assert len(loaded) == len(recs) and info["bad_json"] == 1 and len(info["files"]) == 2
    mock_paper = [row("2023/MHIP-R0-100-2305.pdf", 2023, "7", MOCK_LIKE_SOURCE + "Podaj nazwę dokumentu.",
                      key("Konstytucja 3 maja"))]
    cke = paper("2025/MHIP-R0-100-A-2505-arkusz.pdf", 2025) + paper("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024, " [24]")
    stats = build(cke + mock_paper, synthetic=loaded)
    syn = stats["synthetic"]
    assert syn["invalid"] == {"bad_answer_format": 1, "closed_syntax": 1, "bad_kind": 1, "duplicate_id": 1}
    assert syn["duplicates"] == {"dup_dev": 1, "dup_mock": 1, "dup_mock_topic": 1, "dup_synthetic": 1}
    rows = [r for r in _written(stats) if r["source"] == "synthetic"]
    assert [r["id"] for r in rows] == ["syn-open-t1-1", "syn-closed_tf-t1-1", "syn-essay-t1-1", "syn-open-t2-9"]
    assert stats["per_origin"]["synthetic-open"] == {"train": 3} and stats["per_origin"]["synthetic-essay"] == {"train": 1}
    assert stats["per_origin_kind"]["synthetic-essay"] == {"essay": 1}
    for r in rows:
        assert r["origin"] in r["flags"] and r["year"] is None and r["kind"] == r["id"].split("-")[1]
        head = r["prompt"][-1]["content"].split("\n", 1)[0]
        assert head.startswith(f"Zadanie {r['item']['id']} (") and "syn-" not in head   # exam-like task number
    assert rows[2]["item"]["id"] in {str(n) for n in range(21, 29)}
    reasons = {d["id"]: d["reason"] for d in stats["_dropped"] if d["id"] and str(d["id"]).startswith("syn-")}
    assert reasons["syn-open-t1-5"] == "synthetic_dup_mock" and reasons["syn-open-t2-10"] == "synthetic_dup_synthetic"
    assert stats["blocklist"]["mock_near_duplicates"] == 2                      # dup_mock + dup_mock_topic


# ----------------------------------------------------------------------------- RAG-style rendering

class FakeIndex:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, k=5):
        self.queries.append(query)
        return [{"title": f"Artykuł {i}", "section": "Historia", "text": f"Treść {i}: {query[:30]}", "url": "", "score": 1.0 / i}
                for i in range(1, k + 1)]


def fake_query(item):
    return (item["question"] + " " + item["source_text"])[:300]


def fake_format(passages, max_chars=2400):
    return "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))[:max_chars]


def fake_build_messages_rag(item, image_descriptions=None, system_prompt="SYS", style="organizer", passages=None):
    msgs = fake_build_messages(item, image_descriptions, system_prompt + (" RAG" if style == "rag" else ""), style)
    if passages:
        msgs[-1]["content"] = "Materiały pomocnicze:\n" + passages + "\n\n" + msgs[-1]["content"]
    return msgs


def rag_ctx(fraction=0.85, seed=0, k=4):
    return H.RagContext(index=FakeIndex(), build_query=fake_query, format_passages=fake_format, k=k, fraction=fraction,
                        seed=seed, index_path="fake", index_hash="0" * 16)


def rag_rows() -> tuple[list[dict], list[dict]]:
    cke = paper("2025/MHIP-R0-100-A-2505-arkusz.pdf", 2025) + paper("2024/MHIP-R0-100-A-2405-arkusz.pdf", 2024, " [24]")
    return cke, [syn_open(n, f) for n, f in enumerate(YEAR_FACTS)]


def test_rag_rendering_fraction_and_determinism(tmp_path):
    cke, syn = rag_rows()
    ctx = rag_ctx()
    stats = H.build_history_data(cke, tmp_path, build_messages=fake_build_messages_rag, style="rag", rag=ctx,
                                 synthetic=syn)
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    dev = [json.loads(l) for l in (tmp_path / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    dev_rag = [json.loads(l) for l in (tmp_path / "dev_rag.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 23 and len(dev) == len(dev_rag) == 3
    with_p = [r for r in train if r["rag"]["with_passages"]]
    assert len(with_p) == round(0.85 * 23)
    for r in train:
        assert r["prompt"][0]["content"] == "SYS RAG"                        # every train row: the rag system prompt
        assert r["prompt"][-1]["content"].startswith("Materiały pomocnicze:\n") == r["rag"]["with_passages"]
        assert r["prompt"] == fake_build_messages_rag(r["item"], None, style="rag",
                                                      passages=fake_format(ctx.index.search(fake_query(r["item"]), k=4)).strip()
                                                      if r["rag"]["with_passages"] else None)
    for o, g in zip(dev, dev_rag):
        assert o["prompt"] == fake_build_messages(o["item"], None, style="organizer")   # dev.jsonl stays organizer
        assert g["rag"]["with_passages"] and g["prompt"][-1]["content"].startswith("Materiały pomocnicze:\n[1] Artykuł 1")
        assert o["completion"] == g["completion"] and o["id"] == g["id"]
    rag = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))["rag"]
    assert rag["k"] == 4 and rag["fraction"] == 0.85 and rag["index_hash"] == "0" * 16
    assert rag["train_with_passages"] == len(with_p) and rag["dev_rag_rows"] == 3
    # deterministic: same seed -> same rows and prompts; another seed -> another selection
    again = H.build_history_data(cke, None, build_messages=fake_build_messages_rag, style="rag", rag=rag_ctx(), synthetic=syn)
    assert [r["prompt"] for r in again["_kept"]["train"]] == [r["prompt"] for r in train]
    other = H.build_history_data(cke, None, build_messages=fake_build_messages_rag, style="rag", rag=rag_ctx(seed=1),
                                 synthetic=syn)
    assert {r["id"] for r in other["_kept"]["train"] if r["rag"]["with_passages"]} != {r["id"] for r in with_p}
    none = H.build_history_data(cke, None, build_messages=fake_build_messages_rag, style="rag", rag=rag_ctx(fraction=0),
                                synthetic=syn)
    assert not any(r["rag"]["with_passages"] for r in none["_kept"]["train"])
    # a later organizer build removes the stale dev_rag.jsonl
    H.build_history_data(cke, tmp_path, build_messages=fake_build_messages_rag, synthetic=syn)
    assert not (tmp_path / "dev_rag.jsonl").exists()


def test_rag_style_with_the_real_prompt_builder():
    prompts = pytest.importorskip("harness.prompts")
    if "rag" not in getattr(prompts, "STYLES", ()):
        pytest.skip("harness.prompts has no rag style yet")
    cke, syn = rag_rows()
    stats = H.build_history_data(cke, None, build_messages=prompts.build_messages, item_kind=prompts.item_kind,
                                 style="rag", rag=rag_ctx(), synthetic=syn)
    assert stats["item_kind_mismatch"] == 0
    for r in stats["_kept"]["train"]:
        assert r["prompt"][0]["content"] == prompts.rag_system_prompt()
        assert r["prompt"][-1]["content"].startswith(prompts.RAG_MATERIALS_HEADER) == r["rag"]["with_passages"]
    for r in stats["_kept"]["dev"]:
        assert r["prompt"] == prompts.build_messages(r["item"], None, style="organizer")


def test_organizer_style_is_unchanged_by_old_synthetic_and_rag_options():
    prompts = pytest.importorskip("harness.prompts")
    cke, syn = rag_rows()
    base = H.build_history_data(cke, None, build_messages=prompts.build_messages, item_kind=prompts.item_kind)
    full = H.build_history_data(cke + old_paper(), None, build_messages=prompts.build_messages, item_kind=prompts.item_kind,
                                min_year=H.OLD_MIN_YEAR, synthetic=syn, rag=rag_ctx())
    assert full["rag"] is None and not full["_kept"]["dev_rag"]
    cke_rows = [r for r in _written(full) if r["origin"] == "cke-2015+"]
    assert [(r["id"], r["prompt"], r["completion"]) for r in _written(base)] == \
        [(r["id"], r["prompt"], r["completion"]) for r in cke_rows]
    for r in _written(full):
        assert r["prompt"] == prompts.build_messages(r["item"], None, style="organizer")
        assert r["prompt"][0]["content"] == prompts.ORGANIZER_SYSTEM_PROMPT
        assert "Materiały pomocnicze" not in r["prompt"][-1]["content"]
