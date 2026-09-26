Made during the Warsaw Model Trainers hackathon, Kolektyw3, 25–27.09.2026

# Sources

Team Nuori, fine-tuning track. The list below covers what the models, training data, harness and tools are built from.
Copyrighted material is not in this repository. That includes CKE exam papers, the organizers' exam packages, page
scans and anything derived from them: they are credited and linked here, never copied. No script downloads CKE
papers. The organizers' packages come from the organizers (mock package link below), and the history training rows
are built by `scripts/build_history_data.py` from the teammates' private nuori-ai dataset (see "Team
contributions"). All of this goes into gitignored folders (`exams/`, `runs/`, `data/processed/`, `checkpoints/`).
The LLM-filled targets for CKE tasks (`data/history/llm_filled_answers*.jsonl`, `data/history/filled_*/`) and the
Wikipedia knowledge base (`data/rag/`) are gitignored too. So are the third-party exam datasets
(`data/blocklist/*.jsonl`, `data/eval/ext_*.jsonl`), which scripts re-create.

## Models

| Role | Model | Licence | Link |
|---|---|---|---|
| Base (benchmark and fine-tuned) | speakleash/Bielik-4.5B-v3.0-Instruct | Apache-2.0 | https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct |
| Served base weights | `Bielik-4.5B-v3.0-Instruct.Q8_0.gguf` (sha256 `562f2291de257890adf2b4a914da8b194affe6a7a838a6b7ef3d342f306c1b7f`) | Apache-2.0 | https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF |
| Base, candidate exam model (QLoRA base of the 11B adapter, `configs/history-11b.yaml`) | speakleash/Bielik-11B-v3.0-Instruct (11.17B params, gated; HF commit `735bfee1125fe8b497ac2769de94822a11f77167`) | Apache-2.0 | https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct |
| Served 11B weights | `speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf`, 6,364,684,992 bytes (sha256 `8b072b7f0e291b8a1586010e1416376f69ea6ea4ea864f5be4c143dd088d71d6`); fallback `speakleash_Bielik-11B-v3.0-Instruct-Q3_K_M.gguf`, 5,404,996,288 bytes (sha256 `b47ad0f4f9cc11b6604a1e103deaa07e88608f5e1e9dad563d8e80738c8369b9`); repo commit `764b30b79a2a77b0f3469ecab81e0707894ffbf0` | Apache-2.0 (the quant card also points to https://bielik.ai/terms) | https://huggingface.co/bartowski/speakleash_Bielik-11B-v3.0-Instruct-GGUF |
| Image descriptions (pre-pass, not fine-tuned) | Qwen3.5-4B GGUF (base model Qwen/Qwen3.5-4B) | Apache-2.0 | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF |
| Served image-description weights | `Qwen3.5-4B-Q8_0.gguf`, 4,482,403,488 bytes, plus the projector `mmproj-F16.gguf`, 672,423,616 bytes (sha256: recorded on the box at freeze) | Apache-2.0 | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF |
| Alternative VLM (compared on the mock) | Gemma 4 E4B QAT GGUF | Apache-2.0 | https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf |
| Candidate, under evaluation (untouched run on the mock, not fine-tuned) | google/gemma-4-12b-it (11.96B params, the organizers' count with the vision tower), first run as `gemma-4-12b-it-Q4_K_S.gguf`, 6,764,526,400 bytes (sha256 `8bfbcccb50049e670dcc55ba1aabf7e79c65c06eea96e42a0689baf7503aa81f`; repo commit `fc034cfff751157913579611efad8462ac1be606`), which measured over the 8.8 GB team cap; the memory-safe file is `gemma-4-12b-it-IQ4_XS.gguf`, 6,375,734,080 bytes (sha256: recorded on the box at freeze) | Gemma licence terms as stated on the google/gemma-4-12b-it model card (the GGUF metadata says apache-2.0) | https://huggingface.co/unsloth/gemma-4-12b-it-GGUF (base model: https://huggingface.co/google/gemma-4-12b-it) |

- Sizes are bytes on disk. The sha256 values are the Hugging Face LFS hashes of those files. The 4.5B Q8_0 and the 11B
  Q4_K_S were also re-hashed on the box and match.
- Only one base model sits the exam. The other candidates are listed because we ran or trained them during the event.
- **What Qwen3.5-4B does:** it only transcribes and describes the exam's images before the exam run
  (`scripts/describe_images.py`, `harness/vision.py`). It is served alone and stopped before Bielik loads. It never
  answers a question: its instruction (`VLM_INSTRUCTION` in `harness/vision.py`) says "Nie rozwiązuj zadania i nie
  interpretuj: nie wybieraj odpowiedzi A–D, nie oceniaj zdań jako prawdziwe lub fałszywe". Its descriptions are saved
  once to a fixed file, and that same file feeds both the base run and the tuned run. `scripts/run_exam.py` prints the
  file's sha256, which must match between the two runs, so image handling does not change the base-vs-tuned comparison.
- Bielik v3 Small technical report: https://arxiv.org/abs/2505.02550
- Our LoRA adapters (history-v0 on the 4.5B, and the 11B adapter from `configs/history-11b.yaml`) are Apache-2.0,
  following the Bielik base model's licence. A Gemma adapter, if one is trained, follows the Gemma terms of
  google/gemma-4-12b-it instead. The adapters are trained on data derived from exam papers and answer keys (mostly
  CKE, © Centralna Komisja Egzaminacyjna, plus two Nowa Era papers), so they may be used for educational and research
  purposes only (hackathon rules). More in data/LICENSE.md, "Model weights".

## Exam material (links only; © Centralna Komisja Egzaminacyjna unless noted)

- CKE matura papers and answer keys, 2023 formula: https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2023/arkusze/
- CKE matura papers and answer keys, 2015 formula: https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2015/arkusze/
  (archives: https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2023/archiwum/,
  https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2015/archiwum/)
- arkusze.pl copies of CKE papers and answer keys: the files with flat names (`historia-<year>-<month>-….pdf`) in the
  `path_in_nuori_ai` column of `data/history/sources.tsv` (96 papers and 95 keys, the two Nowa Era papers below
  included; the dataset records no per-file download URL). Where the same paper is also there as a CKE-code file (for
  example `2019/MHI-R1_1P-192.pdf`), the copy is dropped as a duplicate and its rows count once. 24 of the flat-named
  papers are trial papers (`próbna`) whose issuer the dataset does not record; their notes in sources.tsv say so.
- Nowa Era "próbna matura" 2018 and 2019, extended level (`historia-2018-nowa-era-probna-rozszerzona` and
  `historia-2019-nowa-era-probna-rozszerzona` in sources.tsv): © Nowa Era, a publisher, not CKE; not redistributed.
  They give 40 train rows (2018: 24, 2019: 16, of which 6 have LLM-filled targets; `per_formula.nowa_era` in the
  build's `stats.json`) and no dev rows, so they are in the trained adapters.
- The mock exam is the May 2023 history paper (extended level), MHIP-R0-100-2305:
  https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2023/Arkusze_egzaminacyjne/2023/Historia/MHIP-R0-100-2305.pdf.
  The organizers' JSON package is at https://matura-json-guide.ania-olchowik.chatgpt.site/#downloads and is used only
  for evaluation, never for training (`scripts/train_history.py --protect-exam`).
- Organizers' benchmark and grading protocol: "Lost in Historical Time? A Polish History Matura Benchmark for Large
  Language Models", https://arxiv.org/abs/2608.12343, and https://warsawmodeltrainers.dev/matura

## Team contributions

- **nuori-ai (teammates), https://github.com/DriversLab/nuori-ai:** the teammates' **private** repository. It
  contains CKE page scans and text, which may not be republished, so it stays private. It is used here as an input
  dataset only and is not the repository of this submission. It is the teammates' pipeline that parses history exam
  papers and answer keys (2003–2026, mostly CKE, including two Nowa Era trial papers) into one row per task
  (`tasks.jsonl`, paper list in `inventory.json`), with page renders and image-description passes. Nothing from it is
  copied into this repository: its checkout is read in place. Its first commit is dated 26.09.2026 13:59 (+02:00).
- The history rows are rebuilt from that dataset with `scripts/build_history_data.py`, from a clone at the recorded
  commit (full build command in `docs/DATA.md`). The `git clone` below works only with access to the teammates'
  private repository:

  ```bash
  git clone https://github.com/DriversLab/nuori-ai ../nuori-ai
  git -C ../nuori-ai checkout d86d5558406a22f3dd90122bb87ca43c52fe5211
  python scripts/build_history_data.py --nuori-dir ../nuori-ai --mock-exam exams/mock/exam.json   # + flags in docs/DATA.md
  ```

  `tasks.jsonl` and `inventory.json` are byte-identical at d86d555 and at c173503, the commit the recorded build
  (26.09 17:54) used. The mock paper MHIP-R0-100-2305, its formula-2015 sibling EHIP-R0-100-2305 and their copies in
  the dump are excluded from every training and dev split (`train/history_data.py` `MOCK_PAPER_CODES`).

  Without access to nuori-ai, the rows built from exam papers (800 of the 1,360 train rows and all 98 dev rows)
  cannot be rebuilt from this repository alone, and neither can the gitignored LLM-filled targets that some of them
  use; `data/history/sources.tsv` lists the papers they come from. The other parts can: the 560 synthetic history
  items are committed (with `--nuori-dir` pointing to a folder that holds an empty `tasks.jsonl`, the script builds
  just their 560 train rows), and the Wikipedia knowledge base behind the prompts is rebuilt with
  `scripts/build_rag.py`.
- `data/history/sources.tsv` lists the exam papers and answer keys behind the rows, as identifiers and paths only:
  `paper_id`, `year`, `level`, `kind` (`arkusz` or `zasady`), `path_in_nuori_ai` (the PDF path as nuori-ai's
  `inventory.json` records it; the PDFs themselves are not in that repository), `official_url` and a `note` with the
  rows each paper gives (from the 26.09 17:54 build) or why it gives none. `official_url` is filled only where the data
  gives one (the mock paper, from the organizers). For the others, use the CKE archive pages above; we do not guess
  file URLs.

## Pre-existing tools (built before Fri 25.09 18:00)

The rules allow our own pre-existing tools. The following was written on 16.09.2026, or on Fri 25.09 before the 18:00
opening, for our earlier 8-subject closed-format pipeline. No model was trained with it before the hackathon.

- Generic LoRA/SFT and eval infrastructure: `eval/`, `train/` (all of it except `train/history_data.py`),
  `configs/base.yaml`, `configs/run1.yaml`, `configs/bielik11b.yaml`, `configs/smoke.yaml`, `configs/sweep.yaml`,
  `requirements.txt`, `pytest.ini` and the tests of these modules. The history run reuses only the generic parts: the
  LoRA trainer `train/sft.py`, `train/config.py`, the `train/dataset.py` and `train/registry.py` helpers,
  `eval/schema.py`, `eval/modeling.py` and the `train/wiki.py` fetch helpers. Some of these files were edited during
  the hackathon (for example `train/sft.py` and `configs/base.yaml` on 26.09); those edits are hackathon work.
- Scripts of that pipeline: `scripts/train.py`, `build_data.py`, `generate_synthetic.py`, `synth_task.py`,
  `show_items.py`, `sweep.py`, `promote.py`, `status.py`, `writeup.py`, `run_eval.py`, `compare.py`,
  `merge_and_export.py`, `fetch_wiki.py` and `fetch_external.py`.
- Legacy 8-subject material, not used to train the history adapters: `data/synthetic/`, `data/wiki/`,
  `data/eval/dev.jsonl`, `data/eval/heldout.jsonl`, `data/eval/verification_log.jsonl`, the prawko eval add-on
  (`eval/organizers.py`, `configs/organizer.yaml`, `scripts/import_prawko.py`), `docs/KICKOFF.md`,
  `docs/ARCHITECTURE.md` and `docs/research/`.
- Public data sampled then (16.09.2026), not written by us: `data/general/train_pool.jsonl` (EU-Instruct-Synthetic
  rows; 151 of them are the general replay share of the history training mix), `data/eval/general_heldout.jsonl` (the
  forgetting check) and the third-party exam sets used for dedup and evaluation (`data/blocklist/`,
  `data/eval/ext_llmzszl_matura.jsonl`). `scripts/fetch_external.py` rebuilds them.

**Created during the hackathon** (from Fri 25.09 18:00; the files are dated Sat 26.09): `harness/`,
`train/history_data.py`, the history scripts (`scripts/train_history.py`, `build_history_data.py`, `build_rag.py`,
`run_exam.py`, `describe_images.py`, `judge_packets.py`, `judge_report.py`, `validate_answers.py`,
`check_tokenizer.py`, `judge/*.workflow.js`, `serve_model.sh`, `export_lora_gguf.sh`, `setup_labqoat.sh`),
`configs/history.yaml`, `configs/history-11b.yaml`, `data/history/`, `data/rag/titles_history.yaml`,
`docs/RUNBOOK.md`, `docs/DATA.md`, `docs/VISION.md`, the tests of these modules, the history training rows
(`data/processed/history/`, built 26.09 17:54) and the LoRA adapters (trained Sat 26.09 from freshly initialised LoRA
weights on the untouched public bases).

## Training and evaluation data

Data licences: see data/LICENSE.md (licence, source, attribution and rebuild command for every data folder).

| Data | Source | Licence | Use |
|---|---|---|---|
| History SFT rows (`data/processed/history/`, gitignored): 1,360 train + 98 dev rows | rebuilt by `scripts/build_history_data.py` from the teammates' private nuori-ai dataset (see "Team contributions"; exam papers 2003–2026 listed in `data/history/sources.tsv`; the mock paper MHIP-R0-100-2305 and its sibling EHIP-R0-100-2305 excluded), the LLM-filled targets and the synthetic items below; prompts from `harness/prompts.py` style "rag" with passages from the knowledge base below | derived from © CKE (40 train rows from © Nowa Era papers), not redistributed | LoRA training (train) and dev loss (dev) |
| LLM-filled targets (`data/history/llm_filled_answers.jsonl`, `data/history/filled_old/`, gitignored): 792 answers | written by Claude (Anthropic) agents during the hackathon for CKE tasks whose key has no usable model answer (essays, 2003–2014 papers), following the CKE key where the dump has one | derived from © CKE, not redistributed | LoRA training targets (409 used) |
| Synthetic history items (`data/history/synthetic/`, committed): 560 items (80 essays, 480 open and closed) | written by Claude (Anthropic) agents during the hackathon from Polish Wikipedia and public-domain documents on Wikisource (https://pl.wikisource.org/), facts checked against pl.wikipedia; each item lists its `source_urls` | CC BY-SA 4.0 (derived from Wikipedia text) | LoRA training |
| History knowledge base (`data/rag/`, gitignored except `titles_history.yaml`): 25,324 passages from 2,451 articles | https://pl.wikipedia.org/ via `scripts/build_rag.py` (titles in `data/rag/titles_history.yaml`), BM25 index `harness/rag.py`, no model | CC BY-SA 4.0 | reference passages in the training prompts and in the tuned exam run (`run_exam.py --style rag`); not part of the model |
| Polish Wikipedia articles (`data/wiki/articles.jsonl`, fetched 16.09.2026, before the hackathon) | https://pl.wikipedia.org/ (`scripts/fetch_wiki.py`) | CC BY-SA 4.0 | source texts for the 8-subject synthetic items (`data/synthetic/`) |
| Wikipedia test fixtures (`tests/fixtures/wiki/Kula.json`, `Prawo_Coulomba.json`, committed, fetched 16.09.2026) | https://pl.wikipedia.org/ (plain-text extracts with `url`, `pageid` and `revid`) | CC BY-SA 4.0 | unit tests of the Wikipedia fetch helpers (`tests/test_wiki.py`) |
| Synthetic matura-style items (`data/synthetic/`, committed) | written by Claude (Anthropic) agents from the Wikipedia articles on 16.09.2026, before the hackathon (closed APIs are allowed for building, never in the exam harness) | CC BY-SA 4.0 | earlier 8-subject runs; not used by the history adapters |
| Legacy 8-subject eval items (`data/eval/heldout.jsonl`, `dev.jsonl`, `verification_log.jsonl`, committed): 60 + 20 items and their solver/critic log | written by Claude (Anthropic) agents from Polish Wikipedia text (https://pl.wikipedia.org/) on 16.09.2026, before the hackathon | CC BY-SA 4.0 | legacy 8-subject eval (held-out and dev sets of the earlier runs); never trained, not used by the history adapters |
| General Polish replay (`data/general/train_pool.jsonl`, 6,000 public rows sampled 16.09.2026 by `scripts/fetch_external.py --only general`, not written by us) | https://huggingface.co/datasets/openeurollm/EU-Instruct-Synthetic (config `pl`) | Apache-2.0 | ~10% of training rows (151 in the history runs; limits forgetting) |
| General held-out (`data/eval/general_heldout.jsonl`, sampled 16.09.2026) | https://huggingface.co/datasets/NASK-PIB/PLLuM-Align | CC BY-SA 4.0 | forgetting check (loss only) |
| LLMzSzŁ matura items (`data/eval/ext_llmzszl_matura.jsonl`, gitignored, `scripts/fetch_external.py --only ext`) | https://huggingface.co/datasets/amu-cai/llmzszl-dataset | none stated | evaluation and dedup only, never trained |
| Organizers' prawko-v2 questions (`data/eval/ext_prawko_*.jsonl`: 25 dev + 40 test rows; gitignored, re-created with `python scripts/import_prawko.py`) | the driving-test question catalogue of the Ministry of Infrastructure on gov.pl (https://www.gov.pl/web/infrastruktura/jak-uzyskac-prawo-jazdy), taken via the organizers' workshop repo https://github.com/stared/train-llm-from-scratch (`datasets/prawko-v2/data.json`) | licence of the question text not verified by us, so the rows are not committed | evaluation only, never trained |
| Exam blocklist (`data/blocklist/`: only `README.md` is committed; the `*.jsonl` files are gitignored, `scripts/fetch_external.py --only blocklist`) | public Polish exam datasets listed in `data/blocklist/README.md` | per source (see that file) | leakage filtering only, never trained |

## Software and compute

- llama.cpp (MIT), `llama-server` and `convert_lora_to_gguf.py`: https://github.com/ggml-org/llama.cpp
- QVAC Fabric, Tether's llama.cpp fork with TurboQuant KV cache (MIT): https://github.com/tetherto/qvac-fabric-llm.cpp
- QVAC CLI / SDK, `qvac serve --openai` (Apache-2.0): https://github.com/tetherto/qvac (npm `@qvac/cli`)
- Hugging Face transformers, TRL, PEFT (Apache-2.0): https://github.com/huggingface/transformers,
  https://github.com/huggingface/trl, https://github.com/huggingface/peft; bitsandbytes (QLoRA)
- Compute: Forgehand by Labqoat (NVIDIA L4 sessions; CLI: npm `@qforge/forgehand`), https://app.forgehand.app,
  plus Nebius credits.
