# Polish matura exams (CKE formuła 2023): task formats, scoring rules and available datasets

Everything below was checked against CKE PDFs, the Hugging Face APIs, the datasets I downloaded and code I ran. Anything I couldn't confirm is marked **UNVERIFIED**.

## 0. Recommendations
1. **How to set up the auto-grader.**
   - **Single-choice (A–D):** support both styles organizers are likely to use.
     - **Loglikelihood:** score the letters A–D with the LLMzSzŁ prompt (§B.2) and take the highest.
     - **Generation:** generate with temperature 0, then take the first match of `\b[ABCD]\b` (the speakleash leaderboard method).
     - Several letters in one answer should score 0. CKE does this: "Gdy do jednego polecenia zdający podaje kilka odpowiedzi… nie otrzymuje punktów".
   - **Multi-statement items** (true/false, fill-the-brackets, matching): score by how many parts are correct, using CKE's rules (§A.2). Don't make them all-or-nothing, except for items worth 0–1.
2. **External eval.** Use `amu-cai/llmzszl-dataset`, config `default`, split `test`, filtered to `type == "Egzaminy Maturalne"`.
   - That gives **377 items**: Matematyka 220 (2015–2023), Fizyka 136 (2002–2020), Biologia 21 (2015–2023).
   - All are A–D single choice. There are no history, WOS, geography, chemistry or Polish-language items.
   - For open math answers, add `pawel04/otwarte-pytania-matura-cke` (187 extended-math open tasks, 2015–2025, with the CKE marking key).
3. **Leakage blocklist (never train on these).** Take the union of:
   - the full LLMzSzŁ set (18,821)
   - `CohereLabs/include-base-44` Polish (548 test + 15 validation)
   - `dokato/exam-polish-matura` (52)
   - `dokato/multimodal-PL-exams` (153)
   - both `pawel04/*-matura-cke*` sets (187, of which the 100-item set is a subset)
   - `pawel04/llmzszl-open-ended` (177)
   - GitHub `haribo841/Abituria` math JSON (793 items, 2015–2026)
   - GitHub `lMamacl/zadania-maturalne` Polish-language tasks (1,041)
   - the text of the CKE papers and informatory PDFs from 2023–2026, extracted with pdftotext.

   **No machine-readable dataset covers the formuła-2023 true/false, bracket-choice or matching items for history, WOS, geography, chemistry or physics.** Those only exist as CKE PDFs.

---

## TASK A — CKE matura conventions (formuła 2023)

### A.1 Sources read
- **Informatory** (task types and scoring rules), under `https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2023/Informatory/2024/`:
  - `Informator_EM2025_historia_2025_2026.pdf`
  - `Informator_EMod2025_Wos.pdf`
  - `Informator_EM2024_{geografia,biologia,chemia,fizyka,matematyka_pp,matematyka_pr,jezyk_polski_pp,informatyka}.pdf`
  - The informatyka file is still the 2022/23 edition.
- **Real May-2025 marking schemes**, under `https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2023/Arkusze_egzaminacyjne/2025/zasady_oceniania/`:
  - `MHIP-R0-100-2505-zasady.pdf`, `MWOP-R0-100-2505-zasady.pdf`, `MGEP-R0-100-2505-zasady.pdf`, `MBIP-R0-100-2505-zasady.pdf`
  - `MCHP-R0-100-2505-zasady.pdf`, `MFAP-R0-100-2505-zasady.pdf`, `MMAP-P0-100-2505-zasady.pdf`, `MPOP-P1-100-2505-zasady.pdf`, `MINP-R0-100-2505-zasady.pdf`
- Local text copies are in `<research scratch>/dl/cke/*.txt` (local research workspace, not committed).

### A.2 Task types: wording, answer format, scoring
Every subject's informator says closed tasks are "zadania, w których zdający wybiera odpowiedź spośród podanych" and include **wyboru wielokrotnego, prawda–fałsz, na dobieranie**. Open tasks include **z luką, krótkiej odpowiedzi, rozszerzonej odpowiedzi**.

| Type | Typical instruction (verbatim) | Gold format | Scoring (verified examples) |
|---|---|---|---|
| **Single choice** (called "wielokrotnego wyboru"; in math PP "wyboru jednokrotnego") | "Dokończ zdanie. Zaznacz właściwą odpowiedź spośród podanych." / "Wybierz właściwą odpowiedź spośród podanych." Usually A–D, sometimes A–C. | one letter | 0–1. Math PP 2025 has **Wersja A/B**: the same item with shuffled options, so the gold letter differs (e.g. Z1: A=B, B=C). |
| **True/false (P/F)** | "Oceń prawdziwość poniższych stwierdzeń. Zaznacz P, jeśli stwierdzenie jest prawdziwe, albo F – jeśli jest fałszywe." | e.g. `1. – P, 2. – F, 3. – P` or `PFP` | 2 statements → 0–1, both must be right (WOS "poprawne ocenienie prawdziwości dwóch stwierdzeń"; biologia 2025 Z1.1; geografia 2025 Z3; math PP 2025 Z21 `FP`). 3 statements → 0–2: 3 right = 2, 2 right = 1, fewer = 0 (historia 2025 Z14.2; fizyka 2025 Z1.1 "poprawne zaznaczenia w trzech stwierdzeniach"; biologia Z4.4; geografia Z18; polski informator "0 pkt – jedna poprawna odpowiedź"). 4 statements (informatyka 2025 Z1.2, 0–2): 4 = 2, 3 = 1. |
| **Fill the brackets** (bracket choice) | "W każdym nawiasie podkreśl właściwe określenie." Text like "(wzrostu / spadku)". | chosen word per bracket | biologia: "1 pkt – za podkreślenie poprawnych określeń w dwóch nawiasach"; 0–2 with 3 brackets: 3 = 2, 2 = 1. Chemia 2025 Z42.1 (0–1): all brackets. |
| **Matching** (dobieranie / przyporządkowanie) | "Przyporządkuj…" / "Dopasuj…" | `A – 2, B – 3, C – 1` or `A4, B3` | 0–1 all-or-nothing (biologia 4.1, WOS 14.1, polski Z25). 0–2 with 3 pairs: 3 = 2, 2 = 1 (historia 2025 Z4, Z11.1; geografia Z22). Geografia informator 16.1 with 4 pairs: 4 = 2, **2 or 3 = 1**. Pair counts per item can differ. |
| **Choose two** (in math PP called "wielokrotnego wyboru") | "Zaznacz dwie odpowiedzi, tak aby dla każdej z nich dokończenie zdania było prawdziwe." Options A–F. | two letters, e.g. `BD` | 0–2. Math PP informator Z15: "2 pkt – B i D. **1 pkt – wybranie jednej lub dwóch odpowiedzi, z których jedna jest poprawna**". Geografia Z35: 2 right = 2, 1 right = 1. More than 2 picks: UNVERIFIED (general rule suggests 0). |
| **Double closed choice** | "Dokończ zdania. Zaznacz odpowiedź spośród A–D oraz odpowiedź spośród E–H." | `1. C 2. E` | Math PP informator Z2 (0–2): 1 point per sentence. |
| **Choice + justification** | "Dokończ zdanie. Zaznacz odpowiedź A, B albo C oraz jej uzasadnienie 1., 2. albo 3." | `C1`, `B2` | 0–1, both parts must be right (biologia, fizyka informatory). |
| **Choice + written justification** (history, WOS) | "…rozstrzygnięcie wraz z uzasadnieniem…" | letter/yes-no + free text | 0–1 or 0–2; the letter alone gets 0 (historia 2025 Z1.1, Z2…; WOS Z2.2). |
| **Gap fill** (z luką) | "Uzupełnij zdania." / "Wpisz…" / table cells | word, number or symbol | historia and informatyka use the closed-task rules. **Math PP: "za każdą poprawnie uzupełnioną lukę można otrzymać po 1 pkt"** (2025 Z11 0–4 = four sentences; Z30 0–2 with "Nie akceptuje się zaokrągleń"). Biologia 2025 Z2.1: 4 gaps = 2, 3 gaps = 1. |
| **Short answer, including numeric** | "Oblicz…", "Podaj…", "Wyjaśnij…", "Uzasadnij…" | value + unit, or short text | Math: 2/3/4-point step rubrics ("pokonanie zasadniczych trudności" earns at least half the points; calculation errors → at most n−1). Chemia calculation: 2 = method + calculation + result with unit; 1 = method but a calculation, unit or precision error; 0 = wrong method. Fizyka: result without or with wrong unit → not full marks; wrong number of significant figures → not full marks; a consistent calculation error → −1. |

**General rules from the 2025 marking schemes (apply to the grader):**
- **All subjects:** "Akceptowane są wszystkie odpowiedzi merytorycznie poprawne i spełniające warunki zadania."
- **"odpowiedź niepełna" = 0 on a 0–1 item.** All informatory: "1 pkt – odpowiedź poprawna. 0 pkt – odpowiedź niepoprawna lub niepełna albo brak odpowiedzi. ALBO 2 pkt – całkowicie poprawna. 1 pkt – częściowo poprawna lub niepełna. 0 pkt – …"
- **Several answers to one question → 0** (biologia, chemia, polski 2025): "Gdy do jednego polecenia zdający podaje kilka odpowiedzi, z których jedna jest poprawna, a inne – błędne, nie otrzymuje punktów za żadną z nich."
- **Vague answers are wrong** (biologia and polski): "Odpowiedzi nieprecyzyjne, niejednoznaczne, niejasno sformułowane uznaje się za błędne."
- **Any way of marking counts as a choice** (biologia, polski): underline, circle, cross out, etc. Polski adds "pod warunkiem że zdający konsekwentnie go stosuje w jednym zadaniu".
- **Chemia:** a correct formula given instead of a name is fine; a wrong formula or name, even one copied from the task, loses the point. A solution built on a factually wrong assumption scores 0 in full.
- **Biologia:** "Za poprawne rozwiązania zadań będą przyznawane jedynie pełne punkty."
- **Polski:** an answer that is only a quotation → 0. A factual error about a required set book (lektura obowiązkowa) → 0.

### A.3 Per-subject structure
In the right-hand column, a digit before the colon is the item's maximum points and the number after it is how many items had that maximum (e.g. 1:30 = 30 items worth 0–1).

| Subject | Paper (informator) | Closed vs open | Point caps | Point caps across items in 2025 (parsed from the marking scheme) |
|---|---|---|---|---|
| historia (PR) | 180 min, 60 pts; closed ~20% (6–9 items), short open ~80% (22–26), 1 essay worth 15 | wiązki (groups of items on one source); source-based | closed/gap 0–1 or 0–2; short open 0–3 | 1:30, 2:6, 3:1, plus essay 0–15 |
| WOS (PR) | 180 min; informator for 2025/26 onward says **50 pts** | closed MCQ/PF/matching; gap; short answer (names/numbers or descriptive); extended 5 or 7 | closed 0–1/0–2; short and gap 0–1/0–2 | 1:38, 2:5, plus 0–5 and 0–7 |
| geografia (PR) | 180 min, 60 pts, 38–46 items; closed 8–12 items (10–18 pts, ≤30%) | matching includes cause–effect models; gaps include figures and tables | closed 0–1/0–2; open 1–3 | 1:23, 2:17, 3:1 |
| biologia (PR) | 180 min, 60 pts, 44–56 items; closed 10–16 items (12–20 pts, ~30%) | P/F, brackets, matching, choice + justification | closed 0–1/0–2; open 1–3, whole points only | 1:42, 2:9 |
| chemia (PR) | 60 pts | gaps with symbols, formulas, numbers; reaction equations; calculations | closed and short 0–1/0–2; calculation 2/1/0 | 1:27, 2:13, 3:1, 4:1 |
| fizyka (PR) | 180 min, 60 pts, 25–35 items; closed 6–15 items (8–15 pts, ~20%) | P/F (3 statements), choice + justification | closed 0–1/0–2; open 1–4 | 1:4, 2:9, 3:10, 4:2 |
| matematyka PP | 180 min, 50 pts, 27–39 items; **closed 20–25 items = 25 pts (50%)**; answers go on an answer sheet | single, multiple (choose two), P/F, matching, gaps | open 1–4 | 1:26, 2:5, 3:2, 4:2 |
| matematyka PR | 10–14 items, **all open**, 50 pts | short and extended answers | 2–6 | – |
| język polski PP | Test "Język polski w użyciu" + historical-literary test + essay (35 pts) = 60 | closed MCQ/PF/matching; notatka syntetyzująca (summary note, 4 pts); short open 0–4 | closed 0–1/0–2 | Paper 1: 1:11, 2:5, 4:1 |
| informatyka (PR) | closed, open, practical (computer) tasks | P/F / Tak–Nie tables, gaps | closed 0–1/0–2; short open 0–3; extended 0–5 | 1:2, 2:12, 3:4, 4:3 |

Counts of the standard instruction phrasing in the informatory: "Zaznacz P, jeśli stwierdzenie jest prawdziwe, albo F – jeśli jest fałszywe." appears in every subject. "W każdym nawiasie podkreśl właściwe określenie." appears 10× in biologia. "Zaznacz właściwą odpowiedź spośród podanych." appears 24× in math PP.

---

## TASK B — Machine-readable Polish exam datasets

### B.1 Candidates (checked with `/api/datasets`, `/splits`, `/size`, `/first-rows`, plus full downloads)

| Repo id | Gated | License | Size | Contents | Answer key | Matura items |
|---|---|---|---|---|---|---|
| **`amu-cai/llmzszl-dataset`** | no | **none in card** (arXiv paper is CC BY 4.0; the data license is UNVERIFIED) | 1 config `default`, split `test`, 18,821 rows, `llmzszl-test.jsonl` | CKE exams 2002–2024: vocational 18,219; **matura 377**; gimnazjum 175; 8th grade 50 | `correct_answer_index` (int, **0-based**) into `answers` (list; 4 options, 5 in 5 rows, 6 in 1) | **377** (Mat 220, Fiz 136, Bio 21). 3 items have 5 options; 2 duplicate question texts; math notation garbled |
| `CohereLabs/include-base-44` (INCLUDE) | no | apache-2.0 (card) | config `Polish`: test 548, validation 15 | 496 professional certification; Math 47; Sociology (WOS) 4; Biology 1 | `answer` (int, **0-based**) into `option_a..option_d` | 52 in test. 43/52 overlap with LLMzSzŁ matura (≥50% 5-gram containment) |
| `CohereLabs/include-lite-44` | no | apache-2.0 | `Polish` test 250 | 246 professional, 4 sociology | `choices` list + `answer` (0-based) | 4 |
| `dokato/exam-polish-matura` (INCLUDE source) | no | cc-by-nc-sa-2.0 (row field `license: "unknown"`) | `default/train` 52 | Math 47, WOS 4, Bio 1; 2018–2023; includes CKE PDF URL and question number | `answer` is a **string, 1-based** ("1".."4"; checked on math items) | 52 (same items as INCLUDE PL non-professional; some OCR loss, e.g. dropped minus signs) |
| `dokato/multimodal-PL-exams` | no | cc-by-nc-sa-2.0 | `exams_pl.json` 153 rows (the viewer shows 111 image rows) + `images.zip` | 2020–2024: Bio 62, Mat 33, Fiz 32, Geo 16, Hist 7, Chem 2, Philosophy 1. **Bracket and P/F items split into 2-option rows** (79 two-option rows) | `answer` is an **int, 0-based** | 153; the image is `essential` for 132, so text-only eval is invalid |
| `pawel04/otwarte-pytania-matura-cke` | no | none | `default/train` 187 (`-100` = 100-row subset) | **Extended math, open tasks, 2015–2025** (May/June/July, some marked `stara_formula` = old 2015 exam format) | `klucz` = full CKE "Zasady oceniania" text + model solution; `punkty_max` "2".."7" | 187 |
| `pawel04/llmzszl-open-ended` | no | none | train 177 | LLMzSzŁ items rewritten as open questions | `answer` string (e.g. "16√3 cm^2", "45%") | 107 (Mat 67, Fiz 38, Bio 2) |
| `MikolajLangner/llmzszl` | no | none | train 900 | sample of LLMzSzŁ | `correct_answer_index` + `A`–`D` columns + `id` | 11 |
| `mhardalov/exams` (EXAMS) | no | cc-by-sa-4.0 | `crosslingual_pl` train 1,577 / val 394; `multilingual` Polish rows: train 739 / val 246 / test 986 | **All Polish rows are `subject: "Professional"`, grade 12 (vocational)** | `answerKey` letter; `question.choices.label` | **0** |
| `sdadas/gpt-exams` / `AndromedaPL/prometheus-exams-0.1` | no | cc-by-nc-sa-4.0 / none | 8,131 each (same data) | GPT-3.5-generated university exam Q&A | free text | 0 (synthetic) |
| `speakleash/PES-2018-2022`, `amu-cai/medical-exams-*`, `NASK-PIB/Reassess-Polish-Medical-Exams` | no | – | – | medical exams | – | 0 |
| `fegyobeno/NLP_Matura_CR` | no | cc-by-nc-4.0 | 1,519 | **Hungarian** DPO data | – | 0 |
| GitHub `haribo841/Abituria` `Content/exam-20XX-main-{basic,extended}.json` | public | MIT for code; the repo's `CONTENT_PROVENANCE.md` says MIT does not automatically cover the texts | 32 main-session papers, **793 math items, 2015–2026** (formuła 2023: basic 141 + extended 54). Checked against CKE `documentCode` + PDF SHA256 | `mode`: multipleChoice 299 (`options`, `correctOption` **1-based**: all 297 MC items whose worked solution names the letter agree with `correctOption - 1`, checked when building the blocklist), numeric 67 (`expectedValue`), compound 30 (`answerParts`, e.g. choose two), revealOnly 397; `scoringCriteria` | math only |
| GitHub `lMamacl/zadania-maturalne` `data/baza_zadań.json` | public | none | 1,041 tasks from 108 papers, 2011–2026 (extracted from arkusze.pl via MinerU) | Polish-language matura; `typ_zadania` otwarte 399 / zamknięte 206 / wypracowanie 160 / … | **no answers** | blocklist only |
| arXiv 2608.12343 "Polish History Matura Benchmark" | – | – | historia papers 2023/2024/2025 (36+39+37 questions + essays) | scored with the CKE rubric + a CKE-trained examiner | – | Code at anonymous.4open.science; **no dataset found on HF (availability UNVERIFIED)** |
| Open PL LLM Leaderboard (`speakleash/open_pl_llm_leaderboard`, `src/about.py`) | – | – | tasks: polemo2, 8tags, belebele, dyk, ppc, psc, cbd, klej_ner, polqa, poquad, eq_bench, poleval2018, **polish_pes (medical)** | **no matura task** | – | 0 |

The overlap numbers are lower bounds. The check was word 5-gram containment ≥0.5 after NFKC, lowercasing and stripping punctuation, and LaTeX vs Unicode math notation causes misses. Results: LLMzSzŁ matura ↔ Abituria 77/377; `pawel04` open ↔ Abituria 43/187; multimodal ↔ LLMzSzŁ 3/153. **Add embedding similarity** (sentence-transformers is installed) for math.

### Example rows (datasets-server `first-rows` or the downloaded files)
The rows below show each dataset's schema and answer indexing only. Question, option, stem and answer-key text is replaced with "…" because it is CKE or third-party exam content and is not republished here. Field names, value types, ids, file names, source links and answer-index values are kept as downloaded.

`amu-cai/llmzszl-dataset` (default/test). The first-rows endpoint returns gimnazjum items, so these two matura rows come from the downloaded file:
```json
{"question": "…", "answers": ["…", "…", "…", "…"], "correct_answer_index": 3, "year": 2023, "type": "Egzaminy Maturalne", "name": "Matematyka"}
{"question": "…", "answers": ["…", "…", "…", "…"], "correct_answer_index": 1, "year": 2022, "type": "Egzaminy Maturalne", "name": "Biologia"}
```
`CohereLabs/include-base-44` (Polish/test, first-rows):
```json
{"language":"Polish","country":"Poland","domain":"Social Science","subject":"Sociology","regional_feature":"region implicit","level":"Academic","question":"…","option_a":"…","option_b":"…","option_c":"…","option_d":"…","answer":2}
{"language":"Polish","country":"Poland","domain":"Social Science","subject":"Sociology","regional_feature":"region implicit","level":"Academic","question":"…","option_a":"…","option_b":"…","option_c":"…","option_d":"…","answer":2}
```
`dokato/exam-polish-matura` (train, 1-based `answer`):
```json
{"language":"pol","country":"Poland","file_name":"MWO-R1_1P-182.pdf","source":"https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2015/Arkusze_egzaminacyjne/2018/formula_od_2015/wiedza_o_spoleczenstwie/MWO-R1_1P-182.pdf","license":"unknown","level":"A level","category_en":"Society","category_original_lang":"Wiedza o społeczeństwie","original_question_num":11,"question":"…","options":["…","…","…","…"],"answer":"3"}
{"file_name":"MMA-P1_1P-182.pdf","original_question_num":1,"question":"…","options":["…","…","…","…"],"answer":"2"}
```
`dokato/multimodal-PL-exams` (`exams_pl.json`, 0-based `answer`):
```json
{"language":"pl","file_name":"EBIP-R0-100-A-2405-arkusz.pdf","level":"High school exam","category_original_lang":"Biologia","original_question_num":"10.1","question":"…","options":["…","…","…","…"],"answer":1,"image_png":"exams_polish_matura_1.png","image_type":"graph","image_information":"essential"}
{"file_name":"EBIP-R0-100-A-2405-arkusz.pdf","original_question_num":"17.1","question":"…","options":["…","…"],"answer":0,"image_type":"graph","image_information":"essential"}
```
`pawel04/otwarte-pytania-matura-cke` (train):
```json
{"id":"2025_maj_1","rok":"2025","miesiac":"Maj","numer_zadania":"1","punkty_max":"2","tresc_zadania":"…","klucz":"…"}
{"id":"2024_maj_1","rok":"2024","miesiac":"Maj","numer_zadania":"1","punkty_max":"2","tresc_zadania":"…","klucz":"…"}
```
`MikolajLangner/llmzszl` (first-rows):
```json
{"question":"…","answers":["…","…","…","…"],"correct_answer_index":2,"year":2010,"type":"Egzaminy Gimnazjalne","name":"Matematyka","id":0,"A":"…","B":"…","C":"…","D":"…"}
```
`mhardalov/exams` (crosslingual_pl/train, vocational only):
```json
{"id":"5ffb4369-7726-11ea-9116-54bef70b159e","question":{"stem":"…","choices":{"text":["…","…","…","…"],"label":["A","B","C","D"],"para":["","","",""]}},"answerKey":"B","info":{"grade":12,"subject":"Professional","language":"Polish"}}
```

### B.2 How each benchmark grades answers
- **LLMzSzŁ (arXiv:2501.02266 §3.2, quoted from the paper's HTML):**
  - Uses the LM Evaluation Harness; "task configuration was based on the MMLU configuration".
  - Method: "for each answer, a language model … return[s] the probability (likelihood), and the answer with the highest probability was compared with the gold answer"; the metric is accuracy.
  - Template:
    ```
    Przykładowe pytanie egzaminacyjne, test jednokrotnego wyboru

    {{question.strip()}}
    A. {{answers[0]}}
    B. {{answers[1]}}
    C. {{answers[2]}}
    D. {{answers[3]}}
    Prawidłowa odpowiedź:
    ```
  - The task YAML, few-shot count, and whether the scored continuations were the letters A–D or the answer texts are not published (UNVERIFIED). The upstream MMLU default it cites uses `output_type: multiple_choice` and `doc_to_choice: ["A","B","C","D"]`, which I verified.
  - The paper filtered out items needing pictures or charts; PDFs were extracted with PyPDF (plus manual work).
- **INCLUDE (upstream lm-eval `lm_eval/tasks/include/default/Polish/_polish_template_yaml`, verified):** `dataset_path: CohereForAI/include-base-44`, `dataset_name: Polish`, `output_type: multiple_choice`, `doc_to_text: "{{question.strip()}}\nA. {{option_a}}\nB. {{option_b}}\nC. {{option_c}}\n D. {{option_d}}\nAnswer:"`, `doc_to_choice: [A,B,C,D]`, `doc_to_target: answer`, metric `acc`.
- **speakleash Open PL LLM Leaderboard (fork `speakleash/lm-evaluation-harness`, branch `polish3`, verified).** Each task has two variants:
  - an `_mc` loglikelihood variant with `doc_to_choice: ["A","B","C","D"]`, metrics acc and acc_norm;
  - a generative `_regex` variant:
    ```yaml
    generation_kwargs: {until: [".", ","], do_sample: false, temperature: 0.0, max_gen_toks: 50}
    filter_list:
      - name: "score-first"
        filter:
          - function: "regex"
            regex_pattern: "(\\b[ABCD]\\b)"
          - function: "take_first"
    metric_list: [{metric: exact_match}]
    ```
    The PES prompt (`polish_pes/pes.yaml`) is "Spośród wszystkich odpowiedzi wybierz tylko jedną. Odpowiedz tylko i wyłącznie jedną literą.\n{{question}}\nPrawidłowa odpowiedź:" with the pattern `(\b[ABCDE]\b)`.
- **EuroEval** (`src/scripts/dataset_creation/create_llmzszl.py`) builds `EuroEval/llmzszl-mini` as a random 1024/256/2048 train/val/test split of the whole set, so mostly vocational items, with lowercase `a.`–`d.` labels. It is not a matura eval.

### B.3 Verified code (run in the project venv: datasets 5.0.1)
```python
from datasets import load_dataset
import re
llm = load_dataset("amu-cai/llmzszl-dataset", split="test")                 # 18821
matura = llm.filter(lambda r: r["type"] == "Egzaminy Maturalne")            # 377: Mat 220, Fiz 136, Bio 21
inc = load_dataset("CohereLabs/include-base-44", "Polish", split="test")    # 548 (496 Professional certification)

L = "ABCD"
def llmzszl_prompt(r):
    opts = "\n".join(f"{L[i]}. {a}" for i, a in enumerate(r["answers"][:4]))
    return ("Przykładowe pytanie egzaminacyjne, test jednokrotnego wyboru\n\n"
            f"{r['question'].strip()}\n{opts}\nPrawidłowa odpowiedź:")
gold = L[matura[0]["correct_answer_index"]]

def extract_letter(gen, letters="ABCD"):          # speakleash "score-first"
    m = re.search(rf"\b[{letters}]\b", gen); return m.group(0) if m else None
# 'Odpowiedź: D'->D, ' C. bo...'->C, 'Prawidłowa odpowiedź to A'->A, 'a) nie wiem'->None
```
- **Regex pitfall:** a sentence-initial Polish conjunction "A" matches `\b[ABCD]\b`. Force an output format such as "Odpowiedź: X", or anchor the regex.
- A CKE partial-credit helper with passing asserts is at `<research scratch>/code/cke_score.py` (local research workspace, not committed):
  - 0–1 → all parts correct
  - 0–2 → all parts = 2, one part wrong = 1, otherwise 0
  - choose-two → points = number of correct picks, capped at 2
  - several marks on a single-choice item → 0
- **Per-item exceptions exist** (e.g. geografia 16.1: 4 pairs with 2–3 correct = 1). Keep the rubric per item when you can.

### B.4 Pitfalls for eval and blocklist use
- **Index base differs between sources.** LLMzSzŁ, INCLUDE and dokato-multimodal are 0-based ints. `dokato/exam-polish-matura` is 1-based strings. Abituria `correctOption` is a 1-based int. EXAMS uses letters.
- **Match on text, not letters.** Math PP 2025 has versions A and B with shuffled options, so the same stem has different gold letters.
- **LLMzSzŁ matura quality:**
  - Physics is 2002–2020 only (old exam formats).
  - Math formulas are flattened, e.g. "√(3)(−27/16)⋅√(3)2" should be cube roots.
  - Biology has 21 items total.
  - There are no P/F, matching or bracket items: CKE multi-part items were dropped or reduced to single-choice.
- **Coverage gap:** real formuła-2023 closed items for historia, WOS, geografia, chemia and fizyka (true/false, fill-the-brackets, matching) are only in CKE PDFs:
  - papers: `https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2023/arkusze/{2023-2,2024-2,2025-2,2026-2}/`
  - informatory: the files listed in §A.1

  Organizers most likely take items or style from these. Add their text to the blocklist and don't train on it.

Files are in `<research scratch>/` (local research workspace, not committed):
- `dl/llmzszl-test.jsonl`
- `dl/include_base_pl_*.parquet`, `dl/exams/`
- `dl/dokato_*.json`, `dl/pawel_*.jsonl`
- `dl/abit/*.json`, `dl/lmamacl_baza.json`
- `dl/cke/*.txt` (text of the CKE PDFs)
- `code/load_eval.py`, `code/cke_score.py`
- `blocks.py`, `zas.py` (scoring-scheme parsers)