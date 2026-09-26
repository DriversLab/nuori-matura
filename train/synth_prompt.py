"""Specification for generating matura-style synthetic items from Polish Wikipedia articles.

Single source of truth used by:
  * the Claude generator agents that produced data/synthetic/raw/ (they were given generation_prompt()),
  * train/generate_synthetic.py API backends (anthropic / openai-compatible, e.g. vLLM serving Bielik-11B).

Items follow eval/schema.py, so they are rendered into training rows by the SAME prompt/format code
the evaluator uses (eval/prompts.py + eval/answer_format.py).
"""
from __future__ import annotations

import json

from eval.schema import __doc__ as SCHEMA_DOC

# Type mix per subject (fractions of items). Mirrors CKE formuła-2023 closed/short tasks
# (docs/research/matura_formats_datasets.md, section A.2).
SUBJECT_MIX: dict[str, dict[str, float]] = {
    "historia":     {"mc": 0.45, "tf": 0.20, "short": 0.20, "match": 0.10, "multi": 0.05},
    "wos":          {"mc": 0.45, "tf": 0.20, "short": 0.20, "match": 0.05, "multi": 0.10},
    "geografia":    {"mc": 0.45, "tf": 0.20, "short": 0.15, "match": 0.05, "multi": 0.05, "numeric": 0.10},
    "biologia":     {"mc": 0.45, "tf": 0.25, "short": 0.15, "match": 0.05, "multi": 0.10},
    "chemia":       {"mc": 0.40, "tf": 0.20, "short": 0.15, "numeric": 0.20, "multi": 0.05},
    "fizyka":       {"mc": 0.40, "tf": 0.15, "short": 0.10, "numeric": 0.35},
    "matematyka":   {"mc": 0.40, "tf": 0.10, "short": 0.05, "numeric": 0.45},
    "jezyk_polski": {"mc": 0.45, "tf": 0.15, "short": 0.20, "match": 0.15, "multi": 0.05},
}

SUBJECT_FOCUS: dict[str, str] = {
    "historia": "chronologia, przyczyny i skutki, postanowienia aktów/traktatów, postacie i ich dokonania, analiza fragmentu źródła",
    "wos": "ustrój RP (Konstytucja 1997), kompetencje organów, procedury, prawa i wolności, ideologie, organizacje międzynarodowe, prawo",
    "geografia": "procesy przyrodnicze i społeczno-gospodarcze, położenie, klimat, gospodarka Polski i świata, obliczenia (skala, czas słoneczny, gęstość zaludnienia, amplituda)",
    "biologia": "budowa i funkcje komórki, genetyka (krzyżówki), fizjologia człowieka, ekologia, ewolucja, botanika, zoologia",
    "chemia": "budowa atomu, wiązania, reakcje (typy, bilansowanie), stechiometria, stężenia, pH, chemia organiczna (nazewnictwo, grupy funkcyjne)",
    "fizyka": "kinematyka, dynamika, energia i praca, grawitacja, elektryczność, magnetyzm, drgania i fale, optyka, fizyka jądrowa; zadania obliczeniowe z podanymi stałymi",
    "matematyka": "funkcje (liniowa, kwadratowa, wykładnicza, log), równania i nierówności, ciągi, geometria, trygonometria, prawdopodobieństwo i statystyka; zadania obliczeniowe",
    "jezyk_polski": "epoki literackie, lektury obowiązkowe (autorzy, bohaterowie, motywy), środki stylistyczne, gatunki, gramatyka i składnia, rozumienie tekstu",
}

GENERATION_RULES = """\
ZASADY (bezwzględne):
1. Pisz poprawną, naturalną polszczyzną w stylu zadań maturalnych CKE (formuła 2023): "Dokończ zdanie. Wybierz właściwą odpowiedź spośród podanych.", "Oceń prawdziwość stwierdzeń…", "Przyporządkuj…", "Podaj…", "Oblicz…".
2. Fakty MUSZĄ wynikać z podanego artykułu (albo być podstawową wiedzą szkolną niezbędną do rozwiązania). Nie wymyślaj faktów. Zadania obliczeniowe (matematyka/fizyka/chemia/geografia) mogą mieć zmyślone dane liczbowe, ale wynik musi być policzony i sprawdzony (Python), a wszystkie potrzebne stałe podane w treści.
3. Dokładnie jedna obroniona poprawna odpowiedź. Dystraktory wiarygodne (ta sama kategoria, podobna długość i forma), ale jednoznacznie błędne. Bez "wszystkie powyższe"/"żadne z powyższych". Pozycję poprawnej litery zmieniaj.
4. Pytaj o rzeczy, które powinien wiedzieć maturzysta (daty przełomowe, przyczyny/skutki, definicje, procesy, klasyfikacje, autorzy/utwory, zależności) – NIE o ciekawostki, liczby odwiedzin, szczegóły bibliograficzne, meta-informacje o artykule.
5. Każde zadanie samodzielne: nie odwołuj się do "artykułu" ani "tekstu powyżej" (chyba że cytujesz fragment w polu "context" jako tekst źródłowy – wtedy ≤ 700 znaków).
6. Formaty:
   - mc: 4 opcje A–D; answer = litera.
   - multi: 5 opcji A–E, polecenie "Wybierz dwie odpowiedzi…", answer = 2 litery.
   - tf: 3 stwierdzenia, mieszanka P i F; fałszywe stwierdzenia powstają przez minimalną zmianę faktu (data, osoba, miejsce, kierunek zależności); points 2, partial_credit [[2,1]].
   - match: 3 elementy po lewej (1–3), 4 po prawej (A–D, jeden nadmiarowy); answer {"1": .., "2": .., "3": ..}.
   - short: odpowiedź jednoznaczna, jeden termin/nazwa/liczba słowna; w "answer" podaj WSZYSTKIE realistyczne poprawne warianty (formy fleksyjne, skróty, z imieniem i bez). Grader porównuje tylko po casefold i obcięciu cudzysłowów/interpunkcji na brzegach.
   - numeric: "answer" liczba; "tolerance" zgodna z żądanym zaokrągleniem (np. 0,01 przy wyniku do dwóch miejsc); "unit" jeśli wynik ma jednostkę; w treści napisz, w jakiej jednostce i z jaką dokładnością podać wynik.
7. "rationale": 1–2 zdania po polsku uzasadniające klucz (fakt z artykułu lub obliczenie).
8. "source": "wikipedia:<tytuł artykułu>"; "difficulty": easy|medium|hard (celuj w ~30% easy, 50% medium, 20% hard).
9. Nie powtarzaj tego samego faktu w dwóch zadaniach. Nie kopiuj prawdziwych zadań z arkuszy CKE.
"""


def type_quota(subject: str, n_items: int) -> dict[str, int]:
    mix = SUBJECT_MIX[subject]
    raw = {t: mix[t] * n_items for t in mix}
    quota = {t: int(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - quota[t], reverse=True):
        if sum(quota.values()) >= n_items:
            break
        quota[t] += 1
    return {t: q for t, q in quota.items() if q > 0}


def generation_prompt(article: dict, n_items: int, id_prefix: str, max_article_chars: int = 18000) -> str:
    """Prompt for one article. Output contract: JSONL, one item per line, schema = eval/schema.py."""
    subject = article["subject"]
    quota = type_quota(subject, n_items)
    text = article["text"][:max_article_chars]
    return f"""Jesteś doświadczonym egzaminatorem CKE. Na podstawie artykułu z polskiej Wikipedii ułóż {n_items} zadań w stylu matury z przedmiotu: {subject}.
Zakres tematyczny przedmiotu: {SUBJECT_FOCUS[subject]}.
Rozkład typów zadań: {json.dumps(quota, ensure_ascii=False)}.
Identyfikatory: {id_prefix}-01, {id_prefix}-02, …

{GENERATION_RULES}
SCHEMAT (JSON, jeden obiekt na linię):
{SCHEMA_DOC}

ARTYKUŁ: {article['title']}
---
{text}
---

Zwróć WYŁĄCZNIE {n_items} linii JSONL (bez komentarzy, bez bloków kodu)."""
