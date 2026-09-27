# Sunday checklist (27.09, freeze and exam at 11:00)

**Decide first: may the upload list more than one model?** (the upload form has "List every model in your solution"
and an "Add another model" button; the kick-off FAQ allows several models, each under 8 GB)

| | Command on the box | Models to list | Mock (our judge): base → ours |
|---|---|---|---|
| **A. several models allowed (default)** | `bash scripts/final_exam.sh exams/final final` | Bielik-11B + Qwen3.5-4B | 39 → **46** (+7) |
| B. strictly one model | `MODEL=bielik bash scripts/final_exam_single.sh exams/final final` | Bielik-11B only | 33 → 37 (+4) |

Variant B never sees the images (no descriptions), which costs about 9 points on the mock. Gemma-4-12B as a single
model that reads the images itself measured 9.0–9.6 GB in image mode, over the 8.8 GB team cap, so it is not an option.
Both commands write `runs/final-base/answers.json` and `runs/final-tuned/answers.json`.

The system that sits the exam (decided on the judged mock, 26.09 night):

- **All 37 items, essay included:** Bielik-11B-v3.0-Instruct, bartowski Q4_K_S GGUF, no adapter, with our harness
  (image descriptions, local Wikipedia retrieval with relevance cutoff 50, 9-sample votes on closed items, label repair,
  essay planned first). One answering model.
- **Option, only if the organizers confirm several models are fine:** `ESSAY_MODEL=gemma` hands the essay to
  Gemma-4-12B-it (unsloth IQ4_XS GGUF, thinking). Same mock score (46); Gemma's essays average slightly higher over
  several runs, but it adds a second answering model.
- **Image descriptions:** Qwen3.5-4B, unsloth Q8_0 GGUF + mmproj-F16 (describes images only; the same descriptions
  feed the base run and our system).
- **Base (the untouched benchmark):** the same Bielik-11B Q4_K_S file, no adapter, organizers' protocol.
- Mock result (our calibrated judge, reproducible across two runs): base 39 → our system 46 of 60 (+7).
- Memory, measured at the peak of a run: Qwen 7.37 GB, Bielik-11B 8.27 GB (Gemma, if used, 8.59 GB); team cap 8.8 GB.
- One command, about 11 minutes on the L40S.

## Before 10:00 (a teammate with the TEAM_KEY)

1. **Update team** at https://warsawmodeltrainers.dev/matura (TEAM_KEY), all fields in one save:
   - Base model: `speakleash/Bielik-11B-v3.0-Instruct` (Q4_K_S GGUF)
   - Link: https://huggingface.co/bartowski/speakleash_Bielik-11B-v3.0-Instruct-GGUF/blob/main/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf
   - Size: 6.36 GB
   - Repository: https://github.com/DriversLab/nuori-matura
2. **Repository access:** make it public (`gh repo edit DriversLab/nuori-matura --visibility public
   --accept-visibility-change-consequences`) or add the organizers' and jury's GitHub accounts as readers.
3. **Optional official check of the final system on the mock** (one upload per hour, graded in batches):
   file `~/Downloads/nuori-submissions/nuori-final-system-mock.answers.json`, exam "Mock", name
   `nuori-final-system-mock`, models: the Bielik-11B and Qwen3.5-4B links below, "biggest improvement" ticked.
   Also worth one line on Telegram: "Our harness answers with Bielik-11B; Qwen3.5-4B only describes the images (same
   descriptions for base and tuned); may a second model (Gemma-4-12B) write the essay, or should one model answer?"
4. **Box alive:** `ssh root@<box>` → `nvidia-smi` shows the L40S; `ls /workspace/models/bielik-11b
   /workspace/models/gemma-4-12b /workspace/models/qwen3.5-4b` shows the GGUFs; `/workspace/venv-exam/bin/python -V`.
   If the Labqoat session was restarted, everything needed is on /workspace (llama.cpp build, models, the exam venv);
   the harness runs with /workspace/venv-exam, no reinstall needed.

## From 11:00

```bash
# laptop: download the final ZIP from https://warsawmodeltrainers.dev/submissions.html, then
scp ~/Downloads/<final>.zip root@<box>:/workspace/fine-tune/exams/final.zip
# box
cd /workspace/fine-tune && mkdir -p exams/final && unzip -o exams/final.zip -d exams/final
tmux new -s final 'bash scripts/final_exam.sh exams/final final 2>&1 | tee runs/final.log'
```

At the end it prints the two files. Copy them to the laptop:

```bash
scp root@<box>:/workspace/fine-tune/runs/final-base/answers.json  ~/Downloads/nuori-final-base.answers.json
scp root@<box>:/workspace/fine-tune/runs/final-tuned/answers.json ~/Downloads/nuori-final-tuned.answers.json
```

Upload both at https://warsawmodeltrainers.dev/submissions.html, exam **Final**:

| File | Name | Models to list | Improvement box |
|---|---|---|---|
| `nuori-final-base.answers.json` | `nuori-final-base` | Bielik-11B GGUF link; Qwen3.5-4B GGUF link (image descriptions) | no |
| `nuori-final-tuned.answers.json` | `nuori-final-tuned` | Bielik-11B GGUF link; Qwen3.5-4B GGUF link (+ Gemma-4-12B GGUF link only if run with `ESSAY_MODEL=gemma`) | **yes** |

Model links:

- https://huggingface.co/bartowski/speakleash_Bielik-11B-v3.0-Instruct-GGUF (file `speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf`)
- https://huggingface.co/unsloth/gemma-4-12b-it-GGUF (file `gemma-4-12b-it-IQ4_XS.gguf`, only with `ESSAY_MODEL=gemma`)
- https://huggingface.co/unsloth/Qwen3.5-4B-GGUF (files `Qwen3.5-4B-Q8_0.gguf`, `mmproj-F16.gguf`)

If `final_exam.sh` stops with `FAILED: ...`, the message names the step; the manual steps are in docs/RUNBOOK.md
section 8. Report problems to the organizers by 11:15.

## Presentation

Deck: the "Nuori — Can a small model pass the matura?" artifact (9 slides, speaker notes on each). Fill the three
numbers on the last slide when the official grades arrive; until then say "graded later" and show the mock result
(base 39 → 46). Stage time is a few minutes: cover, pipeline, judge, results, fine-tuning, lessons.
