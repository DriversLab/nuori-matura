# History matura data (`train/history_data.py`)

Training and dev rows for the fine-tuning track exam: a CKE-style **Polish history matura** (poziom rozszerzony).
The rows are built from the teammates' `nuori-ai` dump and rendered with the exam-time prompt builder, so training
prompts look exactly like the prompts the model will get during the exam.

The training set of the tuned model (the max-score build, counts in "Counts: the max-score build" below):

```bash
python scripts/build_history_data.py --nuori-dir ../nuori-ai --mock-exam exams/mock/exam.json \
  --filled data/history/llm_filled_answers.jsonl --filled 'data/history/filled_old/*.jsonl' --include-old \
  --synthetic 'data/history/synthetic/*.jsonl' --style rag --rag-index data/rag/index --rag-k 4 --rag-fraction 0.85
python scripts/train_history.py --dev data/processed/history/dev_rag.jsonl --protect-exam exams/mock/exam.json
```

The individual switches:

```bash
python scripts/build_history_data.py --nuori-dir ../nuori-ai --mock-exam exams/<mock>/exam.json
python scripts/build_history_data.py --stats-only                     # counts only, writes nothing
python scripts/build_history_data.py --include-old                    # + 2003-2014 papers (trained via filled targets)
python scripts/build_history_data.py --filled a.jsonl --filled 'dir/*.jsonl'   # LLM-written targets (repeatable)
python scripts/build_history_data.py --synthetic 'data/history/synthetic/*.jsonl'   # synthetic items (repeatable)
python scripts/build_history_data.py --include-old --style rag --rag-index data/rag/<index>   # RAG-style prompts
python scripts/train_history.py --protect-exam exams/<mock>/exam.json      # consumes train.jsonl / dev.jsonl
```

`--nuori-dir` defaults to `$NUORI_DIR`, then to `../nuori-ai`. By default every `exams/*/exam.json` is used as a
leakage blocklist; pass `--mock-exam` to name the files yourself. Without `--filled` the builder reads
`data/history/llm_filled_answers*.jsonl` and `data/history/filled_*/*.jsonl`; without `--synthetic` it reads
`data/history/synthetic/*.jsonl` (`--no-filled` / `--no-synthetic` turn them off). Missing files add nothing.

## Sources

| Source | What we use | Where it lives |
|---|---|---|
| [DriversLab/nuori-ai](https://github.com/DriversLab/nuori-ai) (private team repository) `tasks.jsonl` | One row per "Zadanie", made by their `parse.py` (pdftotext) from CKE arkusze and zasady oceniania: task text, parent context (sources), answer-key section | Their checkout. **Read in place, never copied** |
| CKE papers and keys (cke.gov.pl), plus arkusze.pl copies of the same papers | Content of the rows above | Not in this repo |
| Nowa Era "próbna matura" papers (2018, 2019) | Content of the rows above (40 train rows) | Not in this repo |
| Organizers' mock `exam.json` | Leakage blocklist only (near-duplicate check) | `exams/` (gitignored) |
| LLM-filled targets `{id, answer}`: written by Claude agents during the hackathon, from the CKE key where the dump has one (1,445 lines: 54 for 2015+ rows, 1,391 for 2003–2014 rows; 792 answers, 653 `SKIP`) | Targets for `needs_answer` rows and for pre-2015 rows | `data/history/llm_filled_answers*.jsonl`, `data/history/filled_*/*.jsonl` (**gitignored**: CKE-derived) |
| Synthetic items: 560 exam-style items (80 essays, 480 open and closed) written by Claude agents during the hackathon from Polish Wikipedia (CC BY-SA 4.0) and Wikisource (public-domain documents); facts checked against pl.wikipedia | Extra train rows, see "Synthetic items" | `data/history/synthetic/*.jsonl` (committed) |
| `harness.rag` index: 25,324 passages of 2,451 Polish Wikipedia articles (CC BY-SA 4.0), BM25, no model | Reference passages for `--style rag` (training rows and the tuned run) | `data/rag/` (gitignored except `titles_history.yaml`; `scripts/build_rag.py` rebuilds it) |

We do not use their `sft_*.jsonl`, `rag_corpus.jsonl` or `pages/`. Those files have the defects listed below, and the
page scans are copyrighted.

## Licence notes

- CKE papers and keys, and the third-party sources quoted in them, are copyrighted. Nowa Era papers belong to the
  publisher. **Nothing CKE-derived is committed.** Every output goes to `data/processed/history/`, which is
  gitignored because `data/processed/` is. Anyone can rebuild the outputs from the teammates' repo, or from
  cke.gov.pl by running their `inventory.py` and `parse.py`.
- The LLM-filled answers are keyed by CKE task ids and many repeat the key's model answer, so they are CKE-derived
  too: `data/history/llm_filled_answers*.jsonl` and `data/history/filled_*/` are gitignored and travel to the GPU box
  with the built rows (`docs/RUNBOOK.md` 6.1). A rebuild without them loses the essays and every pre-2015 row.
- The synthetic items (`data/history/synthetic/`) are our own text, written by Claude agents during the hackathon
  (closed APIs are allowed for building, never in the exam harness). Source texts quote Polish Wikipedia (CC BY-SA
  4.0, attributed in the item and in `source_urls`) or public-domain documents from Wikisource (the 1791
  constitution, the 1792 Targowica act, the 1794 insurrection act and Połaniec proclamation, the 1863 manifesto,
  Gloger's encyclopaedia), so the files are shared under CC BY-SA 4.0. They are checked against every converted CKE
  item and the mock paper (0 near-duplicates). Some quote the same famous primary sources that CKE papers also quote
  (the 1791 constitution, the 1863 manifesto, the 21 postulates of 1980).
- The RAG knowledge base is Polish Wikipedia text (CC BY-SA 4.0). Only the title list is committed;
  `scripts/build_rag.py` fetches the articles (`train/wiki.py`: one request at a time, polite User-Agent) and
  `--no-fetch` rebuilds the same index byte for byte from the cached articles.
- `blocklist_ids.json` stores row ids, file names and 16-hex sha1 fingerprints of normalized question text. It does
  not store exam text, but it is still written under the gitignored folder.
- The mock package (`exam.json` and images) never enters the repo. Only a path to it is passed in.

## What the builder does

1. **Papers.** Each file gets a canonical id: the CKE code (`MHIP-R0-100-2405`, `MHI-R1-162`, `EHIP-R0-100-2105`) or
   the flat stem (`historia-2024-czerwiec-matura-rozszerzona`). Files from the same year whose cleaned questions
   match at 95% or more are merged into one paper. The official CKE file is kept, then the copy with more answer
   keys. This removes arkusze.pl copies of CKE files and the `(1)` duplicate.
2. **Mock exclusion.** The mock exam is `MHIP-R0-100-2305`, the full May 2023 paper. Its formula-2015 sibling
   `EHIP-R0-100-2305` comes from the same session and shares sources and most tasks with it. Both papers, and every
   copy of them (`historia-2023-maj-matura-rozszerzona`, `historia-2023-maj-matura-stara-rozszerzona`), are dropped
   from every split. Their ids and fingerprints go to `blocklist_ids.json`. The organizers' `exam.json` is also
   checked: any row whose question matches a mock item by 4-gram Jaccard ≥ 0.6 and whose source matches at ≥ 0.5
   is dropped.
3. **Cleaner** (`clean_layout`). It removes exam-sheet layout text:
   - page headers and footers: "Strona x z y", "16 z 44", "Egzamin maturalny z historii", "Poziom rozszerzony",
     "Formuła 2023", paper codes, the arkusze.pl footer;
   - examiner and sticker boxes: "Wypełnia egzaminator", "Nr zadania", "Maks./Uzyskana liczba pkt" and their
     values, PESEL/sticker boxes, "Brudnopis", "(nie podlega ocenie)";
   - task markers with their point ranges ("5.1.\n0–1"), "(1 pkt)", score-box rows, empty answer bullets, and the
     essay sheet ("WYPRACOWANIE na temat nr …").

   It also turns dotted answer lines into bare labels, so "Nazwa: …… Rok: ……" becomes "Nazwa:\nRok:", as in the
   organizers' `question` field. It rejoins enumerators that pdftotext split from their text, including A–D
   columns.
4. **Item shape.** Each row becomes an organizer item `{id, group, max_points, question, source_text, images, answer_format}`.
   The sources are the parent context plus the text before the instruction. Formula-2015 intro lines ("Na podstawie
   źródła A … wykonaj polecenia") are removed. A source printed after one task's instruction is carried to the next
   tasks. P/F items are rebuilt as "instruction + `1. statement`" lines with the P/F columns removed, like the mock.
5. **Picture dependence** (`picture_check`). A row is dropped when the model would need a picture it never sees:
   - The instruction names a visual. The stem match catches every inflection, including t→c and k→c forms:
     mapa/mapie, ilustracj-, fotografi-, zdjęci-, karykatur-, plakat/plakacie, moneta/monecie, schemat/schemacie,
     wykres-, drzeworyt/drzeworycie, rycin-, portret/portrecie, znaczek, herb, godło, plan (only as "na planie" or
     "na podstawie planu"), tablica/drzewo genealogiczne, "elementy graficzne", "przedstawiony/ukazany/widoczny na",
     "prezentowana świątynia", "legenda", and others.
   - The instruction cites a visual source by label ("źródło 2.", "obu źródeł"), or asks "(A, B czy C)" about
     letters that only exist inside a picture.
   - A table that pdftotext scrambled is referenced ("w tabeli", "tabela ilustruje"). "Uzupełnij tabelę" is
     answer scaffolding, not a source.
   - The source is only a caption and a credit line (less than 150 characters of text and no sentences). A short
     quote or poem still counts as text.

   Visual sources that the instruction never uses are kept, flagged `visual_source_unused`, and replaced by
   `[Obraz: images/Z05-S2.png]` markers in the mock's format. At exam time those markers get the VLM description.
6. **Targets** (`make_target`):
   - Closed items use the exact `answer_format` syntax of the four closed kinds:
     - `closed_single`: `B`. Single-letter answers are **kept**; nuori's `len(gold) < 2` filter dropped all 84.
     - `closed_tf`: `1: F\n2: P\n3: P`. Keys written as `1–F`, `1. – F;` or `FPP` are all read.
     - `closed_match`: `A: 3\nB: 1`.
     - `closed_multi_part`: `1: C\n2: A`.

     `answer_format` follows the organizers' examples: `A`, alternating `1: P\n2: F…`, `A: 1\nB: 1`, `1: A\n2: A`.
     The harness's `item_kind()` agrees on every emitted row.
   - Open items are built from the key's **model answer**, never from the rubric:
     - The answer is laid out on the question's own labels ("Rozstrzygnięcie: … / Uzasadnienie: …",
       "Nazwa: … / Rok: …").
     - "Przykładowe uzasadnienie" becomes "Uzasadnienie".
     - When the key lists alternative bullets, the first one is used. When the question asks for "dwa/trzy …", the
       first N are used, numbered.
     - Accepted variants are resolved: "unia lubelska [unia realna]" becomes "unia lubelska", "Bizancjum / Cesarstwo
       Bizantyjskie" becomes "Bizancjum", and "[konstytucja] nihil novi" becomes "konstytucja nihil novi".
     - Answers in the form "A. [Aleksander] Wielopolski" become "A: Aleksander Wielopolski".
     - A choice item that asks for a justification becomes "C\nUzasadnienie: …".
   - When the key has no usable model answer, the row goes to **`needs_answer.jsonl`** with its prompt, rubric, key
     text, the question's labels and a Polish `hint`. This covers rubric-only keys, essays, scrambled answer tables,
     label mismatches and P/F layouts that could not be recovered.
7. **Splits.** `train` holds papers from 2015 to 2024. `dev` holds papers from 2025 and 2026: the judge's dev set,
   with rubric text kept in `rubric`. Dev rows are processed first, so a train row that duplicates a dev item is the
   one dropped. Rows are also deduplicated on question and source fingerprints (2015+ rows before pre-2015 rows, so
   an old copy of a newer task is the one dropped). Papers from before 2015 are not converted by default; see
   "Pre-2015 papers".
8. **Filled targets.** A `needs_answer` row whose id has an answer in a `--filled` file becomes a train/dev row
   (flag `llm_filled`, origin `llm-filled`) if the answer passes `answer_problem`: closed answers in the exact
   `answer_format` syntax with the same row keys (single choice: one of the question's option letters); essays start
   with `Temat N.` (N one of the listed topics) and have at least 300 words; open answers are non-empty Polish (a
   missing question label only adds the flag `labels_missing`). Rejected answers keep the row in the queue with
   `filled_rejected` and are counted in `stats.filled` (`rejected:<reason>`). `SKIP` (the filler's "cannot answer")
   and empty answers are ignored. Several files and globs can be given; a later file wins for the same id.
9. **Encoding.** Paper files whose text lost the Polish letters (under 3.5% diacritics; Polish prose has about 6%)
   or carries mojibake (`SpoĞród`, `zwyciĊstwo`) are dropped as `broken_encoding` (three 2003/2011 files; no 2015+
   file is affected).

## Pre-2015 papers (`--include-old`)

`--include-old` (the same as `--min-year 2003`) converts the 2003–2014 papers (both levels; `level` is kept in each
row). Their rows go through the same conversion, picture, source and duplicate filters as the 2015+ rows, plus:

- **Old sheet layout** (`clean_old_layout`, old rows only, so the 2015+ output does not change): "Wypełnia
  egzaminator!" boxes, task markers with their score values ("12.\n1", "13.A.\n1", "16.A. 16.B."), lone page
  numbers, answer-sheet headers ("WYPEŁNIA ZDAJĄCY", PESEL and code boxes, spaced-out letters), key page headers
  ("Klucz punktowania …", "Kryteria oceniania …") and, in keys, requirement areas and skill codes ("… (II 1)").
  Symbol-font bullets become "•". An instruction that pdftotext printed one word per line is joined again. "Temat I /
  Temat II" essay headers become "1. / 2." so the answer can start with `Temat N.`.
- **Sources.** Old papers print all sources of a section (Źródło A … I) before its tasks; a row keeps only the
  sources its instruction names. Tables and diagrams cut into cells (mostly short lines, often with years or "&"
  marks) are dropped as `old_table`.
- **Keys** (`parse_old_key`): 2009–2014 keys have a "Poprawna odpowiedź:" / "Przykład(y) poprawnej odpowiedzi:"
  section per task or part ("A. (0–1)") followed by "N p. – …" rubric lines; 2003–2008 keys are the sheet with the
  answers written in, which `inline_answers` recovers as a word diff against the question ("… tego listu. Sikorski"
  → "Sikorski"). A few 2014 próbna keys already use the 2015 layout.
- **Essays** are recognised by their wording only: old multi-part tasks are often worth 10+ points.
- **Targets.** Old rows never get an automatic target. Every old row goes to `needs_answer` (reason `old_paper`,
  origin `cke-old`) with `key_solution`, `rubric`, `key_text` (the cleaned key) and `auto_target` (what the 2015
  parser would have produced, as a hint). With a filled answer it becomes a train row with flag `llm_filled_old`
  and origin `cke-old`. P/F items whose statements are numbered keep the `closed_tf` syntax; lettered or lost
  layouts become open items.
- **Keyless rows.** About half of the old papers have no key in the dump. `--keyless-filled old` (the default) trains
  a filled answer for such an old row anyway (flag `keyless`; the filler writes `SKIP` when unsure), `none` drops
  them as `no_answer_key`, and `all` also accepts filled answers for keyless 2015+ rows (June papers).

## Synthetic items (`--synthetic`)

JSONL, one object per line:

```json
{"id": "syn-<kind>-<shard>-<n>", "kind": "open", "question": "…", "source_text": "", "answer_format": "…",
 "max_points": 2, "target": "…", "origin": "synthetic-open", "source_urls": ["https://pl.wikipedia.org/…"]}
```

- `kind`: `essay | open | closed_single | closed_tf | closed_match | closed_multi_part`; `origin`: `synthetic-essay`
  for essays, `synthetic-open` for everything else (derived when missing).
- `answer_format`: open `Tekst po polsku. Podaj wszystkie wymagane elementy odpowiedzi.`; essay `Jeden tekst: numer
  wybranego tematu i całe wypracowanie. Minimum 300 wyrazów zgodnie z poleceniem.`; closed kinds the exact syntax
  with the right number of lines (`A`, `1: P\n2: F\n3: P`, `A: 1\nB: 2`, `1: A\n2: B`).
- **Validation** (`validate_synthetic`, reasons in `stats.synthetic.invalid` and `dropped.jsonl`): required fields and
  types, `max_points` 1–20, lengths, the answer_format above, the question's structure (at least two options for
  single choice; as many numbered statements as P/F lines; numbered parts for multi-part; every match key visible
  as a row), and the target (`answer_problem` with strict labels: open answers must use every "Label:" left in the
  question). Duplicate ids are dropped.
- **Deduplication** (4-gram Jaccard over question and source, sparse matrix products, `stats.synthetic.duplicates`):
  - `dup_mock`: the mock paper and its sibling (the nuori rows of MHIP/EHIP-R0-100-2305, kept in memory only) and
    the organizers' `exam.json`: question ≥ 0.5 with source ≥ 0.4, question alone ≥ 0.6, the same source text
    (≥ 0.5), or an essay topic ≥ 0.4;
  - `dup_mock_topic`: a synthetic essay on the mock session's essay theme (rozbicie dzielnicowe / decentralizacja /
    testament Krzywoustego / kryzys monarchii piastowskiej, `MOCK_ESSAY_GUARD`);
  - `dup_dev` / `dup_cke`: every converted CKE item (dev first; train, needs_answer and picture-dropped items too):
    question ≥ 0.6 with source ≥ 0.5, question alone ≥ 0.75, or an essay topic ≥ 0.6;
  - `dup_synthetic`: an earlier kept synthetic item, question ≥ 0.8 with source ≥ 0.8 (alone ≥ 0.9).
- **Rendering.** The prompt header shows an exam-like task number derived from the id (`Zadanie 12.2 (2 pkt)`;
  essays 21–28), never `syn-…`. Rows go to train with `paper: "synthetic"`, `source: "synthetic"`, the origin as
  flag, and `source_urls`.

## RAG-style rendering (`--style rag`)

`--style rag --rag-index PATH` loads `harness.rag` (`load_index`, `build_query`, `format_passages`; lazy import, so
the other styles work without it) and renders every row with `harness.prompts.build_messages(item, None,
style="rag", passages=…)`: the organizer system prompt plus the reference-materials sentence, and the user message
prefixed with `Materiały pomocnicze:\n{passages}\n\n`.

- **Train:** `--rag-fraction` (default 0.85) of the train rows get `--rag-k` (default 4) passages (`format_passages`,
  `--rag-max-chars` 2400); the rest get the rag prompt without passages, so the model also works when retrieval
  returns nothing. The selection is exactly `round(fraction × n)` ids ordered by `sha256("{seed}:{id}")`
  (`--seed`, default 0): deterministic and independent of row order. Each row records
  `rag: {with_passages, titles}`.
- **Dev:** `dev.jsonl` stays organizer-style (the base protocol) and `dev_rag.jsonl` has the same rows with passages.
  A later non-rag build deletes a stale `dev_rag.jsonl`.
- **Needs-answer** prompts stay organizer-style.
- `stats.rag` records the index path and 16-hex hash, k, fraction, seed, max_chars and the with/without counts.
  Serve with the same index, k and max_chars (`run_exam.py --style rag`).

## Outputs (`data/processed/history/`)

| File | Rows |
|---|---|
| `train.jsonl`, `dev.jsonl` | `{"prompt": [system, user], "completion": [{"role": "assistant", "content": target}], "id": "<paper>:<task>", "paper", "year", "formula", "task", "kind", "points", "rubric", "flags", "source": "nuori-ai", "origin", "level", "item": {...}}`; synthetic rows: `id` `syn-…`, `paper`/`formula`/`source` `"synthetic"`, `year` null, `source_urls`; `--style rag` rows also `rag: {with_passages, titles}` |
| `dev_rag.jsonl` | `--style rag` only: `dev.jsonl` rows with retrieved passages in the prompt |
| `needs_answer.jsonl` | `{id, paper, year, split, kind, reason, points, rubric, key_solution, labels, hint, item, prompt, origin, level}`; old rows add `auto_target`, `auto_problem`, `key_text`; a rejected filled answer adds `filled_rejected`. Fill it, then rebuild with `--filled FILE`, where FILE has one `{"id", "answer"}` per line |
| `dropped.jsonl` | `{id, reason, nuori_id}`. `nuori_id` joins to nuori's `render_map.json`, which maps tasks to page PNGs, for a later picture-description pass. Synthetic items: `synthetic_invalid:<reason>` / `synthetic_dup_<what>` with `file`, `line` |
| `blocklist_ids.json` | mock paper codes, excluded files, nuori task ids, our row ids, question fingerprints, mock near-duplicates (CKE rows and synthetic ids) |
| `stats.json` | counts per split, year, kind, formula and origin (`per_origin`, `per_origin_kind`: cke-2015+, llm-filled, cke-old, synthetic-open, synthetic-essay); drop and needs_answer reasons; `filled` (used / rejected / ignored keyless) and the filled files; `synthetic` (records, kept, invalid, duplicates) and its files; flags; prompt builder and style; `rag` settings |

`prompt` comes from `harness.prompts.build_messages(item, None, style="organizer")`: the organizers' system prompt,
then `Zadanie {id} ({pts} pkt)`, the sources, and the question. `--style formatted` renders the same layout plus the
answer-format line; it must match `run_exam.py --style` at serve time (`stats.json` records it as `prompt_style`).
Integration check (2026-09-26): for converted closed (`closed_tf`, `closed_single`, `closed_multi_part`) and open rows,
the messages `run_exam.py` sends for the same item are byte-identical to the rows' `prompt`, and every train/dev
target passes through `harness.postprocess.clean` unchanged. `scripts/train_history.py` checks that train and dev
are disjoint and that no train row contains a protected exam question. Both checks pass on this build.

## Counts: the max-score build (2026-09-26, 17:54; the tuned model's training set)

The command at the top of this file: nuori-ai @ c173503 (3,507 rows), the organizers' mock `exam.json` as blocklist,
29 filled files (792 answers, 653 `SKIP`), 28 synthetic files (560 items), `--style rag` with `data/rag/index`
(hash 72f83c66c85f940c), k 4, 2,400 characters, fraction 0.85, seed 0. **1,360 train rows, 98 dev rows** (plus
`dev_rag.jsonl`: the same 98 rows with passages), 51 rows in `needs_answer`.

| Origin | Split | open | closed_single | closed_tf | closed_match | closed_multi_part | essay | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cke-2015+ (targets from the key) | train | 326 | 35 | 27 | 0 | 3 | 0 | 391 |
| cke-2015+ | dev | 68 | 15 | 14 | 0 | 1 | 0 | 98 |
| llm-filled (2015+ `needs_answer` rows) | train | 18 | 0 | 2 | 0 | 0 | 33 | 53 |
| cke-old (2003–2014, filled targets) | train | 354 | 0 | 2 | 0 | 0 | 0 | 356 |
| synthetic-open | train | 265 | 71 | 96 | 48 | 0 | 0 | 480 |
| synthetic-essay | train | 0 | 0 | 0 | 0 | 0 | 80 | 80 |
| **all train** | | 963 | 106 | 127 | 48 | 3 | 113 | **1,360** |

- **RAG:** 1,156 train rows (85%) carry 4 passages and 204 get the rag prompt without passages; 0 empty retrievals.
  Every train, dev and dev_rag prompt equals `build_messages(item, None, style=…, passages=…)` recomputed from the
  index, every target passes `harness.postprocess.clean` unchanged, and `item_kind` agrees on every row.
- **Filled answers:** 409 used (53 for 2015+ rows; 356 for old rows, 156 of them `keyless`), 0 rejected. The other
  383 belong to rows the builder drops: missing_source 163, picture 127, old_table 73, broken_encoding 19,
  duplicate 1. One filled essay was set to `SKIP` by hand: `historia-2019-maj-matura-stara-rozszerzona:26` had chosen
  its topic 1 (rozbicie dzielnicowe), the mock session's essay theme, which `MOCK_ESSAY_GUARD` also keeps out of the
  synthetic essays.
- **Synthetic:** 560 records, 560 kept (0 invalid, 0 near-duplicates of dev, CKE, the mock or each other), all with
  `source_urls`: 559 cite pl.wikipedia.org, 15 cite pl.wikisource.org (one of them only Wikisource). The 80 essays
  have 504–641 words (median 568) and choose topics 1/2/3 27/30/23 times.
- **needs_answer:** 40 old rows the fillers skipped (21 essays, 19 open), 8 essays of 2015+ papers (the SKIP above
  included), 1 label_mismatch, 1 tf_layout_mismatch, 1 tf_unparsed.
- **Answer balance:** closed_single letters A/B/C/D 20/40/29/17; P/F statements 203/178; essays in train 113.
- **Token lengths** (chat template, prompt + answer, counted with the locally cached Bielik-11B-v2.3 tokenizer files;
  the 4.5B's Polish tokenizer needs fewer tokens, so these are upper bounds): train median 1,525, p95 3,176, max
  5,408 (an essay with passages); 2 rows above 4,096, none above 6,144, so `configs/history.yaml` has
  `model.max_length: 6144`. Dev median 631 (organizer) and 1,675 (rag), max 2,263.
- **Training mix** (`train_history.py --prepare-only --dev …/dev_rag.jsonl --protect-exam …`): 1,360 history rows
  + 151 general replay rows = 1,511; train and dev are disjoint; 37 mock questions checked, none in train.
- **Mock leakage check** (ids and numbers only, the mock text is never printed; reference: the 37 `exam.json` items
  plus the 140 dump rows of MHIP/EHIP-R0-100-2305 and their copies, questions, sources and keys):
  - train/dev/dev_rag: 0 rows from those papers, 0 mock questions (≥ 40 characters) contained verbatim.
  - Highest 4-gram Jaccard of a train question with a mock question: 0.62, a June 2023 task with the same
    instruction wording and a different source (source Jaccard 0.10). Dev: 0.45.
  - Highest share of a mock source's word 8-grams inside one train row: 0.31 (two CKE tasks of other sessions that
    quote an overlapping source); mock key 8-grams: at most 0.15 in train, 0 in dev.
  - RAG corpus: 2,451 articles, all pl.wikipedia.org; mock questions and sources at most 0.06 8-gram containment in
    any passage, mock key text at most 0.08 (a "Wojna stuletnia" passage with the same factual phrasing). Retrieval
    for 5 mock items returns 4 Wikipedia passages each, sharing no 8-gram with any CKE key.

## Counts: `--include-old` build with filled answers (2026-09-26, 16:5x; superseded by the max-score build)

`python scripts/build_history_data.py --include-old` (organizer style, default filled files: 54 answers in
`llm_filled_answers.jsonl` + 1,391 old-paper answers in `filled_old/`, 652 of them `SKIP`; the synthetic essays
written so far; no `exams/*/exam.json` on the laptop). **831 train rows, 98 dev rows**, 50 rows in `needs_answer`.

| Origin | train | dev | kinds |
|---|---:|---:|---|
| cke-2015+ (targets from the key) | 391 | 98 | open 394, closed_single 50, closed_tf 41, closed_multi_part 4 |
| llm-filled (2015+ `needs_answer` rows) | 54 | – | essay 34, open 18, closed_tf 2 |
| cke-old (2003–2014, filled targets) | 356 | – | open 354, closed_tf 2; 156 of them `keyless`, 28 `points_estimated` |
| synthetic-essay | 30 | – | essay 30 (0 invalid, 0 duplicates) |
| synthetic-open | 0 | – | – |

`needs_answer`: 40 old rows (21 essays, 19 open; the filler skipped them), 7 essays, 1 label_mismatch,
1 tf_layout_mismatch, 1 tf_unparsed. Old-row drops: pictures 646 + 165 (all years), no_answer_key 564 (keyless rows the
filler skipped, plus June 2015+ papers), missing_source 186, broken_encoding 86 (three 2003/2011 files), old_table 73.
Checks on this build: every train/dev prompt equals `harness.prompts.build_messages(item, None, style="organizer")`,
every target passes `harness.postprocess.clean` unchanged, `item_kind` mismatches 0, and `train_history.py`'s
train/dev disjointness check passes. The 2015+ rows are identical (ids, order, prompts, completions) to a build
without `--include-old`.

## Counts: 2015+ conversion only (build of 2026-09-26, before filled answers; superseded)

The input is 3,507 rows. The builder emitted **391 train rows** (30 papers, 434 points) and **98 dev rows** (9 papers,
115 points). 64 rows went to `needs_answer` (55 train, 9 dev).

| Kind | train | dev | needs_answer |
|---|---:|---:|---:|
| open | 326 | 68 | 19 |
| closed_single | 35 | 15 | – |
| closed_tf | 27 | 14 | 4 |
| closed_multi_part | 3 | 1 | – |
| essay | – | – | 41 |

`needs_answer` reasons: essay 41, label_mismatch 10, table_answer 7, tf_layout_mismatch 3, rubric_only 2,
tf_unparsed 1.

| Drop reason | Rows |
|---|---:|
| pre_min_year (2003–2014 papers) | 1391 |
| duplicate_paper_copy (arkusze.pl / "(1)" copies of CKE files) | 453 |
| picture: question refers to a picture | 422 |
| duplicate_question (formula-2015/2023 sibling papers share tasks) | 197 |
| no_answer_key (June papers without keys in the dump) | 185 |
| mock_paper (MHIP/EHIP-R0-100-2305 and their copies) | 140 |
| picture: cites a visual source | 116 |
| duplicate_of_dev | 23 |
| picture: thin source (caption only) | 10 |
| missing_source (the cited source is not in the text) | 8 |
| picture: scrambled table | 7 |
| picture: labels only in the picture | 2 |

Rows flagged `visual_source_unused` (kept, with the picture marked): 95. Harness kind mismatches: 0.

Train rows by year: 2015: 23 · 2016: 22 · 2017: 34 · 2018: 43 · 2019: 45 · 2020: 46 · 2021: 50 · 2022: 45 · 2023: 38 ·
2024: 45. Dev rows: 2025: 43 · 2026: 55.

## Why not nuori's `sft_train.jsonl`

These defects were verified on their dump:

- The mock paper appears twice, as `MHIP-R0-100-2305` and its arkusze.pl copy (58 rows), and it is also in the
  RAG corpus.
- The same papers are duplicated as CKE-code files and arkusze.pl copies.
- `len(gold) < 2` drops every single-letter answer (84 rows).
- The picture regex misses inflected forms, so about 330 rows ask about pictures the model never sees.
- The questions keep the sheet layout: dots, markers, score boxes and P/F columns.
- The targets keep grading debris: "Przykładowe uzasadnienie", every bullet variant, and "[…]" or "/" alternatives.

## Known limits and next steps

- Old rows are trained from LLM-filled answers only; 156 of them have no CKE key to check against (`keyless`,
  `--keyless-filled none` drops them). Old multi-part tasks ("A. … B. …") keep all parts in one item, and a missing
  "(N pkt)" is estimated as one point per lettered part.
- `--style rag` renders essays with passages retrieved for the whole question; `run_exam.py --essay-mode plan`
  retrieves again for the chosen topic and plan, so essay prompts differ slightly between training and serving.
- 383 filled answers are not used because their rows are dropped (missing_source 163, pictures 127, old_table 73,
  broken encoding 19). The 163 missing_source rows are old tasks whose cited source `trim_old_sources` could not find
  in the section text; keeping the whole section's sources would bring some back.
- The RAG index usually finds the right article but not always the right section (`harness/rag.py`, BM25 only).
- nuori-ai has a newer commit (d86d555) with the same `tasks.jsonl` plus page crops and 401 picture descriptions.
  The descriptions are not used: spot checks found grammar errors and invented details in several of them, and
  training on them would teach wrong facts. Using them would also need the mock sibling's crops filtered out.
- The biggest loss is about 560 picture-dependent rows. They could come back with page descriptions: nuori's
  `render_map.json` plus `dropped.jsonl` `nuori_id`s, described with a closed API while building. The descriptions
  would then need to be inserted as `[Opis źródła …]` blocks via `image_descriptions`.
- The "/" variant resolver is heuristic. Phrase-level variants inside a sentence keep the first phrase and cut to the
  end of that sentence.
- Tables that pdftotext scrambled remain in some kept rows where the question does not use them.
