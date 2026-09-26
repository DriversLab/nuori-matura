"""train.wiki math-markup cleanup (TextExtracts renders <math> as indented MathML tokens + a {\\displaystyle ...} line)."""
import json
import pathlib
import re

import pytest

from train.wiki import clean_extract, strip_math_markup

HERE = pathlib.Path(__file__).parent / "fixtures" / "wiki"


def block(latex: str, tokens=("x",), style="displaystyle") -> str:
    """TextExtracts rendering of one <math> element (layout copied from live pl.wikipedia extracts)."""
    toks = "".join(f"        {t}\n" for t in tokens)
    return f"\n  \n    \n      \n{toks}      \n    \n    {{\\{style} {latex}}}\n  \n"


# ---------------------------------------------------------------- requested tricky inputs


def test_formula_at_start_of_text():
    text = block("x").lstrip("\n") + " oznacza liczbę."
    assert strip_math_markup(text) == "$x$ oznacza liczbę."


def test_displaystyle_with_braces():
    text = "Wzór " + block(r"{\frac {a}{b}}=\{p:\rho (p,c)<r\}", tokens=("a", "b")) + " koniec."
    assert strip_math_markup(text) == r"Wzór ${\frac {a}{b}}=\{p:\rho (p,c)<r\}$ koniec."


def test_consecutive_formulas_do_not_leak_tokens():
    # live: "Niewiadoma $x$ i wielkości $a,$ b $b,$ c $c$ mogą być" (Równanie kwadratowe)
    # live layout: the next formula's block starts right after the previous one's closing line (no empty line between)
    text = "wielkości " + block("a,", ("a", ",")) + block("b,", ("b", ",")).lstrip("\n") + block("c", ("c",)).lstrip("\n") + " mogą być"
    out = strip_math_markup(text)
    assert out == "wielkości $a,$ $b,$ $c$ mogą być", out


def test_display_formula_then_new_paragraph_keeps_break():
    text = "zdefiniowany jako:\n" + block("V=1") + "\n„Pole” kuli jest inne."
    out = strip_math_markup(text)
    assert "\n" in out.split("$V=1$", 1)[1], out  # paragraph after the display formula must not be glued on


def test_poem_with_leading_spaces_untouched():
    poem = "Litwo! Ojczyzno moja!\n   ty jesteś jak zdrowie.\n\t Ile cię trzeba cenić\n\n  Ten tylko się dowie"
    assert strip_math_markup(poem) == poem
    assert clean_extract(poem) == poem.strip().replace("\n\n\n", "\n\n")


def test_formula_followed_by_heading_keeps_heading():
    # live: Kula "... $K_{c,r}=...$ == Informacja ogólna ==", Prawo Coulomba "... $... .$ == Zobacz też ==\nbariera kulombowska"
    text = "Tekst " + block("x.") + "\n\n== Zobacz też ==\nbariera kulombowska\ntrzecia siła"
    out = clean_extract(text)
    assert "==" not in out and "bariera" not in out, out


def test_heading_followed_by_formula_keeps_heading():
    text = "Wstęp.\n\n\n== Wzór ==" + block("E=mc^{2}") + " gdzie c to prędkość światła."
    out = clean_extract(text)
    assert "## Wzór" in out and "==" not in out, out


def test_suffix_glued_to_formula_keeps_no_space():
    # live: raw "Objętość \n...{\displaystyle n}\n  \n-wymiarowej" -> "$n$ -wymiarowej"
    text = "Objętość \n  \n    \n      \n        n\n      \n    \n    {\\displaystyle n}\n  \n-wymiarowej kuli"
    assert strip_math_markup(text) == "Objętość $n$-wymiarowej kuli"


def test_textstyle_blocks_are_collapsed_too():
    # committed data/wiki/articles.jsonl "Rozkład dwumianowy" still holds raw MathML tokens + {\textstyle ...}
    text = "Wyrażenie " + block("p^{k}", ("p", "k"), style="textstyle") + " określa"
    assert strip_math_markup(text) == "Wyrażenie $p^{k}$ określa"


def test_indented_text_directly_before_formula_is_not_deleted():
    text = "Wiersz:\n   pierwszy wers\n   drugi wers\n" + block("x").lstrip("\n")
    out = strip_math_markup(text)
    assert "pierwszy wers" in out and "drugi wers" in out, out


def test_formula_at_end_of_text():
    assert strip_math_markup("Wynik " + block("x=1.")).strip() == "Wynik $x=1.$"


# ---------------------------------------------------------------- live extracts (pl.wikipedia, fetched 2026-09-16)

LIVE = sorted(HERE.glob("*.json"))


@pytest.mark.parametrize("path", LIVE, ids=[p.stem for p in LIVE])
def test_live_extract_has_no_markup_leaks(path):
    extract = json.loads(path.read_text())["extract"]
    out = clean_extract(extract)
    assert "\\displaystyle " not in out.replace("\\displaystyle {", "")  # inner \displaystyle inside cases is LaTeX, fine
    assert not re.search(r"^[ \t]+\S", out, re.M), "indented MathML token residue"
    assert not re.search(r"={2,6} [^=\n]+ ={2,6}", out), re.findall(r".{60}={2,6} [^=\n]+ ={2,6}", out)
    leaks = [t for t in re.findall(r"\$ (\S{1,2}) \$", out) if t not in ("i", "a", "w", "z", "o", "u", "to")]
    assert not leaks, leaks
