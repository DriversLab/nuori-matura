# Kick-off cheat sheet — Warsaw Model Trainers hackathon

**When:** Fri 25 Sep 18:00 → Sun 27 Sep 16:00 (Warsaw), Kolektyw3, Koszykowa 54. Rules and scoring are explained at kick-off.

**Our track:** fine-tuning — *"train a small model to be better at the exam. Take an open model of a few gigabytes and tune it on data you build yourself. Two size classes, GPU credits included. Scored on matura points plus the gain over the untouched base model."*
(The other track, "harness", means no training at all: retrieval, tools, voting around a frozen model.)

The official brief also says the questions are **matura-style, generated from Polish Wikipedia, graded automatically, scored in points** — which is exactly what this repo builds and evaluates.

## Five questions to ask, and what each answer changes

| # | Ask | If the answer is… | Do this |
|---|---|---|---|
| 1 | **How is the answer extracted — log-likelihood over letter tokens, or parsing generated text?** | **Letters / log-likelihood** (their workshop grader `prawko.py` works this way) | Format compliance earns nothing. Retrain with `python scripts/train.py --config configs/organizer.yaml` (letter-first mix) and watch `organizer` accuracy + ABC mass. |
| | | **Parsed text** | Our default `configs/run1.yaml` is already aimed at this. Ask for the exact regex/prompt and, if it differs, edit `ANSWER_PREFIX` / instructions in `eval/answer_format.py`, bump `FORMAT_VERSION`, re-run base eval. |
| | | **They publish a harness/script** | Point our eval at it: import their items with a converter (see `scripts/import_prawko.py` as the template) and add the set to `eval.sets`. |
| 2 | **Allowed model list + what are the two size classes?** | Bielik 4.5B allowed | Stay on `configs/run1.yaml` (default). |
| | | Bigger class allowed and we have an A100/H100 | `configs/bielik11b.yaml` (11B v2.6, QLoRA). Note results are namespaced per base model, so both can run side by side. |
| | | Bielik not allowed | Set `model.base` to an allowed model; everything else is model-agnostic. Re-run the baseline (fingerprint changes automatically). |
| 3 | **Exact points formula for "matura points plus gain"?** | Gain weighted heavily | Optimise delta: keep the gate strict, prefer the run with the largest `delta_points`. |
| | | Absolute points weighted heavily | Prefer the bigger base model; a smaller gain on a stronger base can win. |
| 4 | **May fine-tuning submissions use inference-time scaffolding (self-consistency voting, retrieval, a second pass)?** | Yes | Worth real points and cheap: majority vote over several samples on top of the fine-tuned model. Not built here yet — ask me and I'll add it. |
| | | No | Keep greedy decoding, which is what `eval/generation.py` already does. |
| 5 | **Do they give an eval set / dev split / example items?** | Yes | Import it as an eval set immediately and re-run the baseline. **Select checkpoints on their dev split only, never on their test.** |

Also worth confirming: submission format (model weights, adapter, or a script?), team size, and whether the base model must be evaluated by them or by us.

## First 30 minutes after kick-off

```bash
python scripts/status.py                 # what we'd submit right now (starts as the untouched base)
python scripts/run_eval.py --base --model speakleash/Bielik-4.5B-v3.0-Instruct   # baseline, if not done yet
python scripts/train.py --config configs/run1.yaml                                # v0-safe fallback
```

Then, once question 1 is answered, either keep `run1` or start the `organizer` mix. Both are gated: a run only becomes the submission candidate if it beats the base and regresses nothing.

## What we already have

- **60 held-out matura-style questions** (70 points, 8 subjects, 6 formats), each verified by two blind solvers and a critic; plus 20 dev items.
- **2,513 training questions** generated from 453 Polish Wikipedia articles, blind-verified, deduplicated against every eval set (0 wrong keys in a 64-item audit).
- **374 real CKE matura questions** (LLMzSzŁ) as a secondary check, and **65 driving-exam questions** (`prawko-v2` dev/test) that the organizers themselves use (`data/eval/ext_prawko_*.jsonl`, gitignored: re-create with `python scripts/import_prawko.py`).
- **Two graders:** our strict `Odpowiedź: X` parser, and the organizers' constrained first-token letter argmax (`eval/organizers.py`), reported side by side with an ABC-mass diagnostic.
- **A promotion gate** that never lets a worse model become the submission, and `scripts/writeup.py` for the judges' story.
