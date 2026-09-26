# Bielik × matura: fine-tune that never scores below base

## History exam (hackathon final)

The Warsaw Model Trainers exam is a CKE **Polish history** matura (extended level, 60 points, separate text and images).
Our entry is one exam model with a history LoRA, served by llama-server or QVAC: one of the candidates in
[docs/RUNBOOK.md](docs/RUNBOOK.md) 6.5b (Bielik-4.5B-v3.0-Instruct with its official Q8_0 GGUF,
Bielik-11B-v3.0-Instruct, Gemma-4-12B), picked by the judged mock results. Until that pick is made, this README names no
single model; the commands below show the 4.5B. A small VLM describes the images in a pre-pass. **Step-by-step:
[docs/RUNBOOK.md](docs/RUNBOOK.md)** (box setup, base and tuned runs, upload, memory check, Sunday 11:00 procedure,
freeze). Sources and the required hackathon line: [SOURCE.md](SOURCE.md).

```bash
bash scripts/setup_labqoat.sh --with-vlm                    # Forgehand L4 box: venv, models, llama.cpp CUDA
python scripts/train_history.py --dev data/processed/history/dev_rag.jsonl --protect-exam exams/mock/exam.json  # rows: RUNBOOK 6.1
bash scripts/export_lora_gguf.sh checkpoints/history-v0/adapter
bash scripts/serve_model.sh --lora checkpoints/history-v0/lora-f16.gguf
python scripts/run_exam.py --exam-dir exams/mock --label tuned --lora-scale 1 --parallel 1 --descriptions runs/desc/mock.json \
  --style rag --rag-index data/rag/index --essay-mode plan --vote 5 --repair-labels --essay-retry
```

Exam packages, run outputs, descriptions and weights are gitignored (copyrighted exam text). The sections below
describe the earlier 8-subject closed-format pipeline (`eval/`, `train/`), which the history run reuses for training.

## Earlier 8-subject pipeline

Fine-tune **speakleash/Bielik-4.5B-v3.0-Instruct** (or Bielik-11B) with LoRA/QLoRA so it scores more **points** than the
untouched base model on auto-graded, matura-style exam questions. The score is *base points + gain over base*, so the repo
is built around three things: **answer-format compliance as a training target**, **delta over base as the primary
metric**, and **a promotion gate that never lets a worse model become the submission**.

## Teammate quick start (Linux + CUDA GPU)

```bash
pip install -r requirements.txt            # Python >= 3.12 required; bitsandbytes installs on Linux only
hf auth login                              # Bielik 4.5B is gated: also click "accept" once at
                                           # https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct

# 1) Baseline (hour 0-4): scores the untouched base (delta 0 by definition) and makes it the standing CANDIDATE
python scripts/run_eval.py --base --model speakleash/Bielik-4.5B-v3.0-Instruct

# 2) v0-safe (before hour 16): processed data -> QLoRA SFT -> merge -> eval -> compare vs base -> gate
python scripts/train.py --config configs/run1.yaml

# 3) Re-print scores + delta for any checkpoint (reuses the cached eval; a gate-rejected run whose merged weights
#    were cleaned up is re-evaluated once from its adapter)
python scripts/run_eval.py --model checkpoints/run1

# 4) What do we submit right now?
python scripts/status.py
```

All training and eval data is committed under `data/`, except the CKE-derived external files (`data/eval/ext_llmzszl_matura.jsonl`, `data/blocklist/*.jsonl`), which are gitignored under the hackathon's no-CKE-content rule (`python scripts/fetch_external.py --only ext,blocklist` re-creates them), and the organizers' prawko-v2 driving-test eval splits (`data/eval/ext_prawko_*.jsonl`, third-party rows, also gitignored: `python scripts/import_prawko.py` re-creates them).
`scripts/train.py` runs the base eval itself if step 1 was skipped.

**The submission is always** `results/bielik-4.5b-v3.0-instruct/CANDIDATE.json`, plus the symlink
`checkpoints/CANDIDATE-bielik-4.5b-v3.0-instruct` once a fine-tune has been promoted. Until then the candidate is the
untouched base model (the standing fallback).

## 46-hour runbook

| When | Do | Done when |
|---|---|---|
| H0–4 | Install, log in, run the **baseline** eval. Compare the `lenient` column with `points`: the gap is what the base loses only to answer formatting, the easiest gain. | `results/<slug>/runs/base/summary.json` exists |
| H4–16 | `train.py --config configs/run1.yaml` (**v0-safe**). If the gate passes, run1 becomes the CANDIDATE and gets the `v0-safe` tag. If it fails, base stays the candidate: read the printed `FAIL [rule]` lines and retry with `--set run_name=run1b --set ...`. | A fine-tune is trained, merged, scored, and either promoted or rejected with reasons |
| H16+ | `python scripts/sweep.py --sweep configs/sweep.yaml --dry-run`, then run it with `--budget-hours N`. Each run is gated, so the candidate only improves. | Leaderboard ranked by Δ over base |
| Any time | `python scripts/writeup.py`: regenerates `results/<slug>/WRITEUP.md` (eval provenance, data funnel, baseline, every run's Δ split into format vs knowledge, gate verdicts). | Judges get the "what we tried and why" story from real numbers |
| Final hour | `python scripts/status.py` and submit the CANDIDATE path. `promote.py --run base --reason ...` rolls back instantly. | — |

Durations were not measured here (no CUDA on the build machine). Each run records `train.runtime_s` and per-stage
timings (`train.stage_seconds` in `runs.jsonl`), and `sweep.py` learns the average run duration from the runs it finished.

## How scoring works here

- **Answer format** (`eval/answer_format.py`, the single source of truth for training targets *and* grading). The model is
  trained to end with exactly one line `Odpowiedź: X`:
  `B` (single choice) · `A, C` (multi-select) · `P, F, P` (true/false) · `1-C, 2-A, 3-B` (matching) · `Mieszko I` (short) · `3,5` (numeric).
- **Strict grading is the headline** (assumed official behaviour). **Lenient** grading recovers answers from free text, so every
  delta splits into a *format* gain and a *knowledge* gain.
- **Prompt robustness.** Training and eval both use several prompt variants: `canonical` (explicit format instruction), `bare`
  (no instruction; the model must default to the format) and `payload_only` ("answer only with the letter"). The fine-tune
  then keeps following whatever instruction the official harness actually uses.
- **Also reported:** MC log-likelihood accuracy (lm-eval / LLMzSzŁ style), per-subject points, and parse-failure rate.
- **If organizers publish their prompt or parser:** edit `ANSWER_PREFIX` and the instructions in `eval/answer_format.py`, then bump
  `FORMAT_VERSION`. The eval fingerprint changes, so the base is re-evaluated automatically, and training rows are rebuilt
  from the new format on the next run. **Confirming the official format is the top open risk.**

### If the official grader scores letters, not text

The organizers' reference exam grader (workshop `scripts/prawko.py`) never reads the generated answer: it applies the
chat template, takes the logits at the **first** answer position, keeps only the single-token ids of the letters
`A`/`B`/`C`, renormalises over them and takes the argmax. Under that protocol answer-format compliance earns nothing
and the `Odpowiedź: ` prefix sits in front of the letter being read. We measure ourselves under it either way:
`eval/organizers.py` scores any model their way and reports it in `summary.json` / `compare.json` (gate rule 9) next to
the strict number, which stays the headline. If that turns out to be the official grading style, the switch is one
command — `python scripts/train.py --config configs/organizer.yaml` (letter-first variant mix). It needs a full
retrain, so only flip it once the grading style is known; `configs/run1.yaml` stays the default until then.

## Eval sets

| File | What | Role |
|---|---|---|
| `data/eval/heldout.jsonl` | 60 original matura-style items (70 pts), 8 subjects, 6 formats. Each was solved by 2 independent blind solvers + a critic (`verification_log.jsonl`) | **Primary** |
| `data/eval/dev.jsonl` | 20 more verified items | Prompt iteration without touching held-out |
| `data/eval/ext_llmzszl_matura.jsonl` | 374 real CKE matura MC items (math/physics/biology) | Secondary: catches regressions on real exam questions (6× more items) |
| `data/eval/general_heldout.jsonl` | 300 human-annotated Polish instruction pairs (PLLuM-Align), disjoint from the training mix | General-capability NLL check (required by the gate) |

## Never worse than base: the gate

`train/registry.py` promotes a run only if **all** of these hold (full list: the "Gate" section of `docs/ARCHITECTURE.md`):
- **Strict points:** the held-out strict gain is at least `gate.min_delta_points` (2.0), and the parse-fail rate is not above base.
- **General ability:** general held-out NLL is at most 5% worse than base.
- **No regressions elsewhere:** no other prompt variant drops more than 1 point, the real-exam set drops at most 1 pp, loglik accuracy drops at most 3 pp, and the lenient (knowledge) score does not regress.
- **Comparable eval:** the eval fingerprint matches base's (items, grading code, generation settings, precision, library versions), and the run was not a limited or narrowed debug eval.
- **Beats the candidate:** the run beats the current candidate by at least 1 point.

Other guards:
- **Recording:** every run is appended to `runs.jsonl`, and `LEADERBOARD.md` is regenerated after each one.
- **Checkpoint cleanup:** rejected runs keep their adapter, but their merged weights are deleted (`pipeline.keep_merged`).
- **Debug evals** made with `run_eval.py` / `compare.py` (`--limit`, `--set eval.*`) go under separate `-limit<N>` / `-dbg<fp>` names and are never recorded. (`train.py --set eval.*` changes the eval for that run *and* re-evaluates base under the new settings, so every run in a sweep must use the same eval settings.)
- **Base results:** the name `base` is reserved, so a mistyped run name can't overwrite the baseline.
- **Candidate protection:** retraining over the candidate's checkpoint is refused.
- **Config typos:** an unknown config key fails loudly with a did-you-mean hint.

## Training data (committed)

1. **Source articles.** `data/wiki/articles.jsonl`: 453 Polish Wikipedia articles, 54–58 per subject, curated against the
   matura syllabus. Re-fetch with `scripts/fetch_wiki.py`.
2. **Synthetic items.** `data/synthetic/raw/` holds 2,744 matura-style items, written from those articles in the exact
   target format.
3. **Verification.** An independent open-book solver answered each item blind (`data/synthetic/verify/`).
4. **Cleaning.** `scripts/build_data.py` produces `data/synthetic/clean.jsonl`, **2,513 items**:
   - drops 2 key disagreements and 125 items the agreeing solver still flagged (ambiguity, debatable statement, typo or low
     matura relevance);
   - drops near-duplicates of protected sets (bge-m3 embeddings on question+answer, plus fuzzy match): 20 vs held-out,
     2 vs dev, 38 vs the real-exam set and 40 vs a blocklist of public Polish exam datasets (`data/blocklist/`), plus 4
     internal duplicates;
   - writes counts and examples of every drop to `dedup_report.json`. Training refuses a clean set that was
     deduplicated against different eval files.
   - **Key audit** (`data/synthetic/key_audit.json`): a separate blind solver re-answered a random 64-item sample of the
     final set: 0 wrong keys (95% upper bound 5.7%), 2 items correct but low matura relevance.
5. **General replay.** `data/general/train_pool.jsonl` has 6,000 Polish instruction pairs (EU-Instruct-Synthetic `pl`).
   Only the 2,637 pairs with answers of at most 500 characters are used (`data.general_max_completion_chars`), so matura
   tokens keep roughly 20–24% of the loss at `general_ratio` 0.3.
6. **Rendering per run.** `train/dataset.py` renders prompt-completion rows for each run:
   - the train/val split is made per source article, so val never shares an article with train;
   - options are re-lettered to balance answer letters;
   - prompt variants are sampled per `data.variant_mix`, system prompts per `data.system_prompt_prob`.

   The loss covers only the completion (`Odpowiedź: …<|im_end|>`).

**Provenance and licensing.**
- **Held-out, dev and synthetic items** were written by Claude agents (Anthropic) from Wikipedia text (CC BY-SA 4.0).
  **Check the hackathon rules on proprietary-LLM-generated data.** If it is not allowed, regenerate with an open-weight
  generator:
  `python scripts/generate_synthetic.py --backend openai --base-url http://localhost:8000/v1 --model speakleash/Bielik-11B-v2.6-Instruct`
  (vLLM). Move the Claude-written shards out of `data/synthetic/raw/` first (build_data reads every `*.jsonl` there), then
  `python scripts/build_data.py --skip-verify-glob "*.openai.jsonl"`.
- **LLMzSzŁ** has no license in its dataset card; we use it only for evaluation and dedup, never for training.
- **Other sources:** PLLuM-Align is cc-by-sa-4.0 and EU-Instruct-Synthetic is apache-2.0. Blocklist sources are listed in
  `data/blocklist/README.md`.

## Model choice (details: `docs/research/bielik.md`)

- **4.5B** (`configs/run1.yaml`): fits bf16 LoRA or QLoRA on a 24 GB GPU and gives fast iterations. It is relatively strong
  on matura-type questions (LLMzSzŁ matura subset 48.9).
- **11B** (`configs/bielik11b.yaml`, v2.6; v2.3 is the ungated alternative): v2.6 beats 4.5B by +2 points on the LLMzSzŁ matura
  subset up to +24 on PLCC (Polish culture); 11B-v3.0 is stronger still. It needs QLoRA on a 24 GB card, and bf16 eval/merge wants at least 40 GB (on smaller cards set
  `eval.quantization=4bit`). Pick it if you have an A100/H100 and the rules allow it: final points = base + gain.
- Results are namespaced per base model: `results/<slug>/`.

## Local plumbing check (Mac/CPU, tiny model, ~1 min)

```bash
python scripts/run_eval.py --config configs/smoke.yaml --base --model HuggingFaceTB/SmolLM2-135M-Instruct
python scripts/train.py --config configs/smoke.yaml       # writes only under .smoke/
python scripts/status.py --config configs/smoke.yaml
```
On MPS/CPU training falls back to plain LoRA (bitsandbytes is CUDA-only). Tests: `pytest` (fast, ~10 s) and
`pytest -m slow` (loads SmolLM2 a few times, the Bielik-11B-v2.3 tokenizer and bge-m3; a few minutes).

## Layout

```
configs/   base.yaml (defaults) | run1.yaml (v0-safe) | sweep.yaml | bielik11b.yaml | smoke.yaml
data/      eval/ (held-out, dev, ext, general held-out) | synthetic/ (raw, verify, clean) | general/ | wiki/ | blocklist/
eval/      answer_format, grader, prompts, schema, modeling, generation, loglik, general_loss, runner, stats, compare
train/     config, wiki, synth_prompt, external, dedup, synth_verify, dataset, generate_synthetic, sft, merge, registry, pipeline
scripts/   run_eval, compare, train, merge_and_export, sweep, status, promote, writeup, build_data, fetch_wiki,
           fetch_external, generate_synthetic, synth_task, show_items
results/   <base_slug>/{runs.jsonl, CANDIDATE.json, LEADERBOARD.md, WRITEUP.md, runs/<run>/...}
checkpoints/ <run>/ (merged model) + <run>/adapter/ ; CANDIDATE-<base_slug> -> current candidate   (gitignored)
docs/      ARCHITECTURE.md (module contracts, schemas, gate) | research/ (verified library facts, Bielik, matura formats, data sources)
```

## Known limitations

- **Unknown official grader.** The format is an assumption; see "How scoring works" for how to retarget it quickly.
- **Small held-out set.** 60 items is statistically small: use the bootstrap CI and the 374-item real-exam set before believing a small Δ.
- **CUDA paths untested here.** QLoRA/bitsandbytes code was checked against the installed library sources and the smoke run
  (plain LoRA on MPS), but it has not run on a CUDA machine in this repo yet. Watch the first run1 log.

## Licence

- **Code** (`harness/`, `train/`, `eval/`, `scripts/`, `tests/`, `configs/`): MIT, see [LICENSE](LICENSE). The MIT
  licence does not cover data or model weights: they are licensed as listed in [data/LICENSE.md](data/LICENSE.md) and
  [SOURCE.md](SOURCE.md). That includes two exceptions inside `tests/`: the Wikipedia extracts in `tests/fixtures/wiki/`
  (CC BY-SA 4.0) and the third-party driving-test questions quoted in the tests (source: the gov.pl question
  catalogue; licence of the question text not verified by us), see "Data outside `data/`" in data/LICENSE.md.
- **Data:** per folder in [data/LICENSE.md](data/LICENSE.md). Wikipedia-derived files, our synthetic items included, are
  CC BY-SA 4.0; the general replay and held-out sets keep their upstream licences (Apache-2.0, CC BY-SA 4.0).
  CKE-derived material is not in the repo.
- **LoRA adapters:** follow the base model's licence (Apache-2.0 for Bielik; a Gemma-4-12B adapter, if one is trained,
  follows the Gemma terms of google/gemma-4-12b-it). They are trained on data derived from CKE exam content, so they are
  for educational and research use only.
