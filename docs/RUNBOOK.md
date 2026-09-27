# Runbook: Saturday evening to the Sunday 11:00 freeze

Team Nuori, Warsaw Model Trainers hackathon, fine-tuning track. The task is the CKE Polish history matura (extended
level). Everything below runs on the Forgehand GPU box unless a step says otherwise. The mock runs of sections 5
and 6 have run on the GPU box (numbers in the log at the bottom); the Sunday commands of section 8 are tested only
by the dry rehearsal on the mock. Measure times and memory on the mock run and write the numbers in the log.

**Score = 40% final score + 40% progress (tuned minus untouched base, absolute points) + 20% presentation.** The
team submits **one** base+tuned pair from **one** exam model (6.5b): the app holds one base model and one final
base/tuned pair per team, and on Sunday the benchmark of the model that sits the exam counts. Both answer files go
to the organizers. The base run must follow their benchmark protocol exactly (bare organizer prompt, greedy, no
extras) on a server started without the adapter. The tuned run is the same GGUF, slot count and decoding, restarted
with our LoRA, plus the tuned-run extras of section 6.5: passages from our local knowledge base (allowed by the
rules; it does not count toward the 8 GB on disk), a planned essay (continued once if it has under 300 words),
majority voting on closed items and a label repair for open items. The LoRA is trained on prompts in the same "rag"
layout (6.1). All extras are off by default and every extra request is logged, so the presentation can split the
gain into LoRA and extras (6.5, ablation).

**Shell conventions (every command below):** run from `/workspace/fine-tune` with the venv active
(`source .venv/bin/activate` in every new shell; Ubuntu has no bare `python` without it). Commands started inside
`tmux new -d ... '<cmd>'` call `.venv/bin/python` explicitly, because a tmux server started earlier does not inherit
the venv. `runs/` and `exams/` must exist before a `tee runs/...` (`mkdir -p runs exams`).

## 0. Cheat sheet

```bash
# box (section 1-2)
fh session start nuori --class gpu-l4 --wait      # then: fh session ls; fh session ssh <session-id>
cd /workspace/fine-tune && bash scripts/setup_labqoat.sh --with-vlm
source .venv/bin/activate && mkdir -p runs exams
# exam pack (section 4): unzip into exams/<name>/ (gitignored)
# LoRA (section 6): rows built on the laptop (6.1), copied with data/rag/ as nuori-data.tgz, then on the box:
tar xzf nuori-data.tgz
tmux new -d -s train '.venv/bin/python scripts/train_history.py --dev data/processed/history/dev_rag.jsonl --protect-exam exams/mock/exam.json 2>&1 | tee runs/train-history-v0.log'
bash scripts/export_lora_gguf.sh checkpoints/history-v0/adapter    # after training -> checkpoints/history-v0/lora-f16.gguf
# one GPU: start the VLM / Bielik servers below only when training is not running
# images -> descriptions (VLM alone, then stop it)
tmux new -d -s vlm 'bash scripts/serve_model.sh --vlm 2>&1 | tee runs/vlm.log'
until curl -sf http://127.0.0.1:8081/health; do sleep 2; done
python scripts/describe_images.py --exam-dir exams/mock            # -> runs/desc/mock.json (rerun retries failures)
tmux kill-session -t vlm                                           # the VLM must be gone before Bielik loads
# the exam model: ONE candidate (6.5b), the same GGUF for base and tuned. Other candidates, commented out:
EXAM_GGUF=/workspace/models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf; EXAM_LORA=checkpoints/history-v0/lora-f16.gguf; EXAM_SRV_ARGS=
# EXAM_GGUF=/workspace/models/bielik-11b/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf; EXAM_LORA=checkpoints/history-11b-v0/lora-f16.gguf; EXAM_SRV_ARGS=
# EXAM_GGUF=/workspace/models/gemma-4-12b/gemma-4-12b-it-IQ4_XS.gguf; EXAM_LORA=checkpoints/<gemma run>/lora-f16.gguf
# EXAM_SRV_ARGS="-- --reasoning-budget 0 --chat-template-kwargs '{\"enable_thinking\":false}'"   # Gemma: thinking OFF
# base: a server WITHOUT --lora (a LoRA loaded at scale 0 changed 23/37 mock answers), one slot
tmux new -d -s srv "bash scripts/serve_model.sh --model $EXAM_GGUF --parallel 1 $EXAM_SRV_ARGS 2>&1 | tee runs/srv-base.log"
until curl -sf http://127.0.0.1:8080/health; do sleep 2; done
python scripts/run_exam.py --exam-dir exams/mock --label base  --parallel 1 --descriptions runs/desc/mock.json
# tuned: stop that server, restart the SAME GGUF with the adapter applied (scale 1), same --parallel 1
tmux kill-session -t srv && sleep 2
tmux new -d -s srv "bash scripts/serve_model.sh --model $EXAM_GGUF --parallel 1 --lora $EXAM_LORA --lora-apply $EXAM_SRV_ARGS 2>&1 | tee runs/srv-tuned.log"
until curl -sf http://127.0.0.1:8080/health; do sleep 2; done
bash scripts/serve_model.sh --mem-check                            # must print ok (team cap 8.8 GB)
python scripts/run_exam.py --exam-dir exams/mock --label tuned --lora-scale 1 --parallel 1 --descriptions runs/desc/mock.json \
  --style rag --rag-index data/rag/index --essay-mode plan --vote 5 --repair-labels --essay-retry   # extras: 6.5; needs data/rag/index
bash scripts/serve_model.sh --mem-check                            # again after the run (Gemma's host RSS, section 7)
python scripts/validate_answers.py runs/base/answers.json  --exam-dir exams/mock
python scripts/validate_answers.py runs/tuned/answers.json --exam-dir exams/mock
# upload both at https://warsawmodeltrainers.dev/submissions.html (TEAM_KEY, solution name, every model used:
# the exam model + unsloth/Qwen3.5-4B-GGUF for the image descriptions; section 5)
```

Both runs print `image descriptions: N from ... (sha256 ...)`: the hash must be the same for base and tuned. The
`tmux` commands use double quotes so `$EXAM_GGUF`, `$EXAM_LORA` and `$EXAM_SRV_ARGS` expand in the current shell.
`EXAM_SRV_ARGS` is empty for Bielik; for Gemma it turns thinking off on both servers (6.5b). It must stay last on the
`serve_model.sh` line, because everything after `--` goes to `llama-server`. Use a new `--label` per model (e.g.
`base-11b-q4ks`) so earlier runs are not overwritten.

| Time (Sat 26 → Sun 27 Sep) | Goal | Done when |
|---|---|---|
| Sat evening, first hour | Box up (sections 1-3), mock pack unpacked, tokenizer check read | `setup_labqoat.sh` prints versions; `check_tokenizer.py` report saved |
| Sat evening | **Base on the mock** (section 5) uploaded | receipt for `nuori-base-q8` |
| Sat evening → night | History data built, LoRA trained (section 6), runs in tmux overnight | `checkpoints/history-v0/adapter/` + `train_metrics.json` |
| Sat night / Sun early | Tuned on the mock uploaded, compared with base | receipt for `nuori-tuned-history-v0` |
| Sun by 09:00 | Exam model and its adapter picked from the judged mock results (6.5b); app registration matches it (section 5) | section 8's `EXAM_GGUF` / `EXAM_LORA` / `EXAM_SRV_ARGS` point to it |
| Sun by 10:00 | Weights uploaded from the box, repo frozen and pushed from the laptop (section 9) | tag `final` pushed; memory check logged |
| Sun 11:00-11:15 | Final exam: base (server without `--lora`), then tuned (server with it), validate, upload both (section 8) | two receipts; problems reported by 11:15 |

## 1. Get a GPU box (Forgehand by Labqoat)

Forgehand (https://app.forgehand.app) is Labqoat's compute platform. Our team's compute allowance comes from the
Labqoat credits ($2,000 split between teams). The persistent directory is `/workspace` (home is `/workspace/.home`),
`/team` is shared across the team's workspaces, and `/scratch` is fast local storage that is **lost when the session
stops**. The machine class for training is `gpu-l4` (NVIDIA L4, 24 GB).

On your laptop (Node.js >= 24):

```bash
npm install -g @qforge/forgehand      # installs `fh` (also: npx @qforge/forgehand --help)
fh login                              # e-mail + 6-digit code; or: fh login --token <personal access token>
fh whoami                             # team slug and team UUID
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519      # never overwrite an existing key
fh ssh-key add ~/.ssh/id_ed25519.pub  # public key only; installed for root on the team's sessions
fh workspaces                         # reuse the team workspace if it exists, otherwise:
fh workspace create <team-slug> nuori --image nvidia/cuda:12.8.1-devel-ubuntu24.04
fh classes                            # prices; gpu-l4 must be enabled for the team
fh session start nuori --class gpu-l4 --wait
fh session ls                         # session id
fh session ssh <session-id>           # or copy the `ssh root@...` line from the web session row
fh session ssh <session-id> -- -L 8080:localhost:8080   # forward the model server to the laptop
fh session stop <session-id>          # stop when idle: compute is billed from launch to stop
```

- **Image:** a CUDA *devel* image provides `nvcc` for the llama.cpp CUDA build. `nvidia/cuda:12.8.1-devel-ubuntu24.04`
  exists on Docker Hub. Check `nvidia-smi` on the box: the driver's CUDA version must be >= the image's CUDA version.
  The image of an existing workspace changes in the web UI (workspace, Environment) and applies to the next session.
- **Secrets:** add `HF_TOKEN` on the web Secrets page (Team scope). It is injected at the **next** session start. The
  Hugging Face account behind it must have accepted the Bielik licence (step 2).
- **Long jobs:** always inside `tmux`, with outputs under `/workspace`. An SSH connection or a running process
  prevents idle stop, but check that a detached job is still running.
- If SSH fails, try `ssh -v`, check the key in Settings, or use the JupyterLab terminal (`fh session jupyter <id>`).

## 2. Set up the box (idempotent, re-run after every session start)

```bash
cd /workspace && git clone <repo url> fine-tune && cd fine-tune     # first time only
bash scripts/setup_labqoat.sh --with-vlm
source .venv/bin/activate && mkdir -p runs exams                   # every new shell: activate the venv
```

The repo has to be on GitHub first. Git runs only on the laptop, in the cleaned tree (`~/soft/pets/ml_hack/fine-tune`
on the laptop that ran the cleanup); never run `git init`, `git add` or `git commit` on the box (section 9). On the
laptop, before the first push: `git init`, `git remote add origin <github url>`, `git add -A`, `git status --short`, and check that nothing from `exams/`,
`runs/`, `data/processed/`, `checkpoints/`, no `*.zip`, `*.tgz`, `*.gguf`, `*.safetensors`, `*.pdf`,
`descriptions*.json`, `data/blocklist/*.jsonl`, `data/eval/ext_llmzszl_matura.jsonl`, `data/eval/ext_prawko_*.jsonl`,
`data/history/llm_filled_answers*.jsonl`, `data/history/filled_*/`, no exam pack unzipped in the repo root
(`/exam.json`, `/answers-template.json`, `/images/`: the mock zip has no top folder) and nothing in `data/rag/` except
`titles_history.yaml` is listed (all gitignored: CKE text, CKE-derived targets, third-party rows, CKE PDFs, weights or
the Wikipedia cache). Then run both checks of section 9 step 3: the path grep, and the content check of
`data/synthetic/dedup_report.json` (a committed file, so only a content check catches exam text coming back into it).
`data/history/synthetic/` is committed (our items, written from Wikipedia; docs/DATA.md).

The script is idempotent: it installs only what is missing and re-downloads nothing that is already there. It:

- installs missing OS packages;
- pulls the repo;
- creates `.venv` (Python >= 3.12) and installs `requirements.txt`;
- checks `hf auth` and access to the gated `speakleash/Bielik-4.5B-v3.0-Instruct`. If access fails, open
  https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct with the token's account and accept the licence;
- downloads models into `/workspace/models`:
  - the HF base, for training;
  - `Bielik-4.5B-v3.0-Instruct.Q8_0.gguf` (5,061,215,424 bytes; the script checks sha256
    `562f2291de257890adf2b4a914da8b194affe6a7a838a6b7ef3d342f306c1b7f`);
  - with `--with-vlm`, the VLM `unsloth/Qwen3.5-4B-GGUF` Q8_0 plus `mmproj-F16.gguf`, the pick in `docs/VISION.md`;
- builds upstream llama.cpp with CUDA (`/workspace/llama.cpp/build/bin/llama-server`, `llama-tokenize`), plus a
  separate converter venv for `convert_lora_to_gguf.py`;
- prints versions.

Optional flags: `--with-qvac-fabric` builds Tether's llama.cpp fork with Vulkan and TurboQuant KV. `--with-qvac-cli`
installs `npm i -g @qvac/cli`.

## 3. Tokenizer check (once, 2 minutes, no weights loaded)

```bash
python scripts/check_tokenizer.py --gguf /workspace/models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf \
  --llama-bin /workspace/llama.cpp/build/bin --json-out runs/tokenizer_check.json
# after the server is up (section 5), also check what llama-server really feeds the model:
python scripts/check_tokenizer.py --gguf /workspace/models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf \
  --llama-bin /workspace/llama.cpp/build/bin --server http://127.0.0.1:8080 --exam exams/mock
```

Known risk: the official GGUF has `add_space_prefix=False`, while the HF tokenizer used in training prepends `▁` after
special tokens (`▁user`, `▁assistant`). The script reports:

- every differing token block;
- template parity (the GGUF chat template against the HF template);
- the server's `prompt_tokens`. HF length + 1 means a double BOS.

Two outcomes:

- **Identical:** nothing to do.
- **Differences only at role tokens:** base and tuned are affected the same way, so the comparison stays fair.
  The exact fix for training/serving parity is to send HF-tokenized prompts as token ids to llama-server
  `/completion`. That is a harness change: decide on the mock whether it is worth it.

## 4. Exam packages (never commit them)

The guide is at https://matura-json-guide.ania-olchowik.chatgpt.site/#downloads. The mock is
`history-2023-mock-v1.zip` (26.9 MB): 37 items, 60 points, the May 2023 paper. The download button runs in the
browser (a plain `curl` of the zip path returns the site's HTML), so download on the laptop and copy the zip to the
box with `scp` (host and port from the `ssh root@...` line of the session row). Unzip every pack into
`exams/<name>/`, keeping `exam.json` next to `images/`. The `exams/`, `runs/`, `descriptions*.json` and `*.zip`
paths are gitignored.

```bash
# laptop: scp ~/Downloads/history-2023-mock-v1.zip root@<box>:/workspace/fine-tune/exams/
mkdir -p exams/mock && unzip -o exams/history-2023-mock-v1.zip -d exams/mock
ls exams/mock          # exam.json  images/  answers-template.json  README.md (the mock zip has no top folder)
python -c "import json; e=json.load(open('exams/mock/exam.json')); print(e['exam_id'], len(e['items']), e['max_points'])"
```

## 5. Baseline: the untouched base on the mock (organizers' protocol)

The protocol has these settings:

- **System prompt:** `harness.prompts.ORGANIZER_SYSTEM_PROMPT`, verbatim.
- **Chat template:** the model's native template.
- **Decoding:** greedy, seed 42.
- **Output limit:** 2048 new tokens; 4096 for the essay (item 26).
- **Context:** 8192 tokens per request.
- **Images:** replaced by descriptions.

`run_exam.py` defaults to all of the above. Without `--descriptions` it uses `runs/desc/<exam folder>.json` (the
`describe_images.py` default) and warns when an image has no description; pass the cache explicitly anyway. Use the
same `--parallel` for base and tuned: `1` is the reproducible choice (continuous batching of several requests can
change greedy outputs slightly), and 37 items should take only minutes. Measure it on the mock.

```bash
# 5.1 images -> descriptions: the VLM runs ALONE, then is stopped (peak RAM = larger model, not the sum)
tmux new -d -s vlm 'bash scripts/serve_model.sh --vlm 2>&1 | tee runs/vlm.log'
until curl -sf http://127.0.0.1:8081/health; do sleep 2; done   # loading takes a minute; Ctrl-C and read runs/vlm.log if it never ends
python scripts/describe_images.py --exam-dir exams/mock        # -> runs/desc/mock.json (re-run retries failures)
tmux kill-session -t vlm && sleep 2 && nvidia-smi              # the VLM must be gone before Bielik loads

# 5.2 base server: official Q8_0 GGUF, no adapter (the benchmark the organizers compare against)
tmux new -d -s srv 'bash scripts/serve_model.sh 2>&1 | tee runs/srv-base.log'
until curl -sf http://127.0.0.1:8080/health; do sleep 2; done
bash scripts/serve_model.sh --mem-check                        # write the numbers in the log below

# 5.3 sit the exam, validate, upload (no --lora-scale: this server has no adapter)
python scripts/run_exam.py --exam-dir exams/mock --label base --parallel 1 --descriptions runs/desc/mock.json
python scripts/validate_answers.py runs/base/answers.json --exam-dir exams/mock
```

`serve_model.sh` defaults: `-c 8192 × --parallel 4` (every slot gets the protocol's 8192 tokens), q8_0 KV cache on
CUDA/Metal, `--temp 0 --seed 42 --repeat-penalty 1.0`, no host prompt cache. Run `--dry-run` to see the exact
command and the memory estimate (about 6.6 GB: weights 5.06 GB + KV 1.07 GB + buffers).

**Upload:**

1. Open https://warsawmodeltrainers.dev/submissions.html.
2. Choose **Mock exam**, enter the TEAM_KEY and the solution name `nuori-base-q8`.
3. The form asks for **every** model in the solution. List the exam model (here `speakleash/Bielik-4.5B-v3.0-Instruct`,
   official Q8_0 GGUF; on Sunday the model registered below) and `unsloth/Qwen3.5-4B-GGUF` (Q8_0 + `mmproj-F16`; it
   only describes the images, it never answers).
4. Upload `runs/base/answers.json` and keep the receipt.

The form also has an optional checkbox, "Enter the biggest improvement category". It is for the final: tick it on the
tuned final upload if we compete for progress (section 8).

A successful upload means the file was received, not graded. The organizers grade with an LLM against the CKE rubric,
in batches about every 30 minutes. Pace uploads: the rules allow at most one submission per hour per team.

Also register the base model in the app (https://warsawmodeltrainers.dev/matura, "Update team" with the TEAM_KEY).
The app holds **one** base model per team, and it checks the checksum of the untouched weights: register the exam
model (6.5b), i.e. exactly the GGUF that both final runs serve. If the pick changes, update it before Sun 11:00.
Fill every field in one save: a field left blank is saved as empty.

| Candidate | Name | Weights link (the exact file) | Size on disk |
|---|---|---|---|
| Bielik-4.5B Q8_0 (default) | `speakleash/Bielik-4.5B-v3.0-Instruct` (official Q8_0 GGUF) | https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF/blob/main/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf | 5.06 GB |
| Bielik-11B Q4_K_S | `speakleash/Bielik-11B-v3.0-Instruct` (bartowski Q4_K_S GGUF) | https://huggingface.co/bartowski/speakleash_Bielik-11B-v3.0-Instruct-GGUF/blob/main/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf | 6.36 GB |
| Gemma-4-12B IQ4_XS | `google/gemma-4-12b-it` (unsloth IQ4_XS GGUF) | https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/blob/main/gemma-4-12b-it-IQ4_XS.gguf | 6.38 GB (+0.18 GB mmproj if used) |

- **Repository:** the repo URL (the same save).
- Link the exact file, never the repo root: the bartowski and unsloth repos hold many quantizations, and the app checks
  the checksum of that one file.
- The limit is 8 GB on disk per model; all three are under it. The size field takes one decimal: 5.1 (4.5B), 6.4
  (11B), 6.4 (Gemma IQ4_XS; 6.6 with its mmproj).
- If Gemma is picked, register the IQ4_XS file, the memory-safe one: 8.40 GB measured at one slot, against
  8.80-9.04 GB for the Q4_K_S (6.76 GB on disk; section 7). Its base and tuned runs must both serve that file.
- The 11B fallback Q3_K_M (6.5b) is a different file, with its own base run:
  https://huggingface.co/bartowski/speakleash_Bielik-11B-v3.0-Instruct-GGUF/blob/main/speakleash_Bielik-11B-v3.0-Instruct-Q3_K_M.gguf
  (5.40 GB).

## 6. Train the LoRA, export it, run tuned

```bash
# 6.1 history rows -> data/processed/history/ (gitignored, CKE-derived), the max-score build: 2015+ CKE rows, the
#     2003-2014 papers through LLM-filled targets, 560 synthetic items and style "rag" prompts (85% of train rows get 4
#     passages from data/rag/index, exactly as run_exam.py --style rag renders them). Build it on the LAPTOP: it needs
#     the nuori-ai dump, data/rag/ and the LLM-filled answers, all gitignored (docs/DATA.md). Mock pack in exams/mock/.
python scripts/build_history_data.py --nuori-dir ../nuori-ai --mock-exam exams/mock/exam.json \
  --filled data/history/llm_filled_answers.jsonl --filled 'data/history/filled_old/*.jsonl' --include-old \
  --synthetic 'data/history/synthetic/*.jsonl' --style rag --rag-index data/rag/index --rag-k 4 --rag-fraction 0.85
#     expected (nuori-ai @ c173503, build of 2026-09-26 17:54): train 1360, dev 98, dev_rag 98, needs_answer 51,
#     prompt_style rag, rag.index_hash 72f83c66c85f940c, train_with_passages 1156 / without 204, item_kind_mismatch 0,
#     synthetic kept 560 (0 invalid, 0 duplicates), blocklist rows 140, mock_near_duplicates 0
#     laptop -> box: the built rows, the RAG index and the inputs of a rebuild (tar: ~41 MB compressed)
tar czf nuori-data.tgz data/processed/history data/rag data/history/llm_filled_answers.jsonl data/history/filled_old
scp nuori-data.tgz root@<box>:/workspace/fine-tune/      # box: cd /workspace/fine-tune && tar xzf nuori-data.tgz
#     A rebuild on the box needs the same inputs plus the dump (git clone --depth 1 https://github.com/DriversLab/nuori-ai
#     /workspace/nuori-ai); without the filled answers it silently loses the essays and the old papers.

# 6.2 check the mix without a model, then train (QLoRA on the L4; ~1-2 h, measure it). The rows are rag style, so the
#     dev loss is measured on dev_rag.jsonl (same dev rows, rag layout); the mock must not be in train:
python scripts/train_history.py --prepare-only --dev data/processed/history/dev_rag.jsonl --protect-exam exams/mock/exam.json
#     expected: n_history_train 1360, n_general_train 151 (10% replay), n_train_rows 1511, protected_questions_checked 37
tmux new -d -s train '.venv/bin/python scripts/train_history.py --dev data/processed/history/dev_rag.jsonl --protect-exam exams/mock/exam.json 2>&1 | tee runs/train-history-v0.log'
#     configs/history.yaml: max_length 6144 (longest row ~5.4k tokens: an essay with passages), 3 epochs, lr 1e-4,
#     effective batch 16 -> about 285 optimizer steps. The log must show dropped_too_long_by_kind {general: 0, matura: 0}.
#     watch: tail -f runs/train-history-v0.log; eval_matura_loss (dev) must fall, eval_general_heldout_loss must not climb much
#     another run: --set run_name=history-v1 --set train.learning_rate=2e-4 (never reuse a run name you still need)

# 6.3 PEFT adapter -> GGUF LoRA (f16, ~100 MB)
bash scripts/export_lora_gguf.sh checkpoints/history-v0/adapter        # -> checkpoints/history-v0/lora-f16.gguf

# 6.4 one server, adapter at scale 0 by default; tuned requests send "lora": [{"id": 0, "scale": 1}]
tmux kill-session -t srv
tmux new -d -s srv 'bash scripts/serve_model.sh --lora checkpoints/history-v0/lora-f16.gguf 2>&1 | tee runs/srv-lora.log'
until curl -sf http://127.0.0.1:8080/health; do sleep 2; done
curl -s http://127.0.0.1:8080/lora-adapters                               # [{"id":0,"path":...,"scale":0}]
ls data/rag/index                                                         # the RAG index (6.5) must exist
python scripts/run_exam.py --exam-dir exams/mock --label tuned --lora-scale 1 --parallel 1 --descriptions runs/desc/mock.json \
  --style rag --rag-index data/rag/index --essay-mode plan --vote 5 --repair-labels --essay-retry
python scripts/validate_answers.py runs/tuned/answers.json --exam-dir exams/mock
# sanity check once: the base through the adapter server at scale 0 must equal the pure base run (same --parallel 1)
python scripts/run_exam.py --exam-dir exams/mock --label base-s0 --lora-scale 0 --parallel 1 --descriptions runs/desc/mock.json
diff <(python -m json.tool runs/base/answers.json) <(python -m json.tool runs/base-s0/answers.json) && echo "scale 0 == base"
```

Upload `runs/tuned/answers.json` as `nuori-tuned-history-v0` (Mock exam; list the same models as in section 5).
Compare base and tuned item by item (`runs/*/run_log.jsonl`) before the LLM grades come back. Look for three things:

- closed items in the exact `answer_format` syntax;
- justifications present where the question asks for them;
- the essay has a topic number and at least 300 words.

`validate_answers.py` warns about all three.

The check failed on the mock: `runs/after_train.log` has `SANITY: scale 0 differs from base in 23 items` (of 37).
The server settings also differed (the pure base ran on a 4-slot server), so the cause is not settled. Either way, a
base that counts never goes through an adapter server: the final base runs on a server without `--lora`, then the
same GGUF is restarted with `--lora ... --lora-apply` for the tuned run, both at `--parallel 1` (section 8).

### 6.5 Tuned-run extras (`run_exam.py`; all off by default, base runs never use them)

| Flag | What it does | Extra requests | Logged in the item record |
|---|---|---|---|
| `--style rag --rag-index data/rag/index [--rag-k 4]` | `harness.rag` (BM25, no model) retrieves 4 passages per item; the prompt gets "Materiały pomocnicze: ..." before the task and one sentence in the system prompt (`harness.prompts` style `rag`) | none | `rag`: query, passages (title, section, url, score, text) |
| `--essay-mode plan` | essay in two steps: the model picks the topic it can argue best and writes a plan (teza, 3-4 arguments with facts, counter-argument, wniosek); with `--style rag` passages are retrieved again for the chosen topic; then it writes the essay (450-650 words) from the plan, starting with "Temat N." | +1 (the plan, max 1024 tokens) | `essay_plan` (messages, raw, topic, rag), `essay_messages` |
| `--vote 5` | closed items: 5 samples (temperature 0.7, top_k 40, seeds 1-5) plus the greedy answer; majority of the normalised answers, per row for true/false and multi-part items, ties go to the greedy answer. Sampling stops as soon as the remaining samples cannot change the result | up to +5 per closed item | `votes` (samples, tally, rule, greedy_answer, changed) |
| `--repair-labels` | open items whose question lists answer labels ("Rozstrzygnięcie:", "Uzasadnienie:", "Nazwa:"): if the answer lacks one, the model is asked once more for the full answer with the labels; the new answer is used only if it lacks fewer labels | +1 per incomplete open answer | `repair` (missing_before, missing_after, used) |

- `--essay-retry` (part of the tuned command, a safety net) asks once for a continuation when the essay has under 300
  words, also after the plan. The training essays have 500-650 words, so it should rarely fire.
- The RAG index lives in `data/rag/index` (gitignored, built from Wikipedia by `scripts/build_rag.py`, see
  `harness/rag.py`). It reaches the box inside `nuori-data.tgz` (6.1), or alone with
  `scp -r data/rag root@<box>:/workspace/fine-tune/data/`. It is not the model: it does not count toward the 8 GB
  on disk. Serve with the index the rows were built with: `stats.json` → `rag.index_hash` (72f83c66c85f940c for the
  2026-09-26 build), k 4 and 2400 characters, the `run_exam.py` defaults. `build_rag.py --no-fetch` rebuilds the same
  index byte for byte from `data/rag/articles.jsonl`.
- The summary prints `rag: ...`, `votes (K=5, T=0.7): ... answer changed by the vote (...)` and
  `label repairs: N asked, M used`. The end of `run_log.jsonl` has the same numbers (`votes`, `repairs`, `essays`).
- Time: the extras roughly add one essay-plan request, up to 5 short samples per closed item and a few repairs. Time
  the tuned mock run; if it is too slow for Sunday, use `--vote 3`.
- `--resume` reuses plans, votes and repairs from the log, so a crashed tuned run continues where it stopped.
- **Ablation for the presentation** (mock, if upload slots allow): LoRA alone vs LoRA + extras shows what each part adds.
  ```bash
  python scripts/run_exam.py --exam-dir exams/mock --label tuned-bare --lora-scale 1 --parallel 1 --descriptions runs/desc/mock.json
  python scripts/run_exam.py --exam-dir exams/mock --label tuned-rag  --lora-scale 1 --parallel 1 --descriptions runs/desc/mock.json \
    --style rag --rag-index data/rag/index
  ```

### 6.5b Exam-model candidates: 4.5B, 11B, Gemma-4-12B (one pair is submitted)

The team submits **one** base+tuned pair from **one** exam model. The Bielik-4.5B Q8_0, the Bielik-11B Q4_K_S and
Gemma-4-12B IQ4_XS (still being evaluated on the mock) are **candidates**, not separate entries for separate prizes:
the app holds one base model and one final base/tuned pair per team, and "Sunday uses the benchmark of the model
that sits the exam". **Never pair one model's base with another model's tuned run** (e.g. a 4.5B base with an 11B
tuned run): the model that sits the exam would have no benchmark.

**Pick the exam model by Sun 09:00** from the judged mock results (6.6, and the official grades where they
exist). The comparison counts only runs that match the frozen setup of section 8: the base on a server without
`--lora` (bare protocol), and the tuned run with exactly the step 3b command (`--lora $EXAM_LORA --lora-apply`, one
slot, all the 6.5 extras). A run with the extras but no LoRA cannot happen on Sunday, so it is a reference only.
Compare each candidate's combined score, about 0.4 × tuned + 0.4 × (tuned − base) = 0.8 × tuned − 0.4 × base, in raw
points (the rules do not say how progress is normalized). A higher base also raises what progress subtracts, so it
wins only if its tuned score is high enough.

Judged mock scores so far (local judge, out of 60):

| Model | Base (bare protocol) | Tuned: LoRA + extras (counts) | Extras only, no LoRA (reference) |
|---|---|---|---|
| Bielik-4.5B Q8_0 | 20 | 18 (LoRA `history-v0`) | 21 |
| Bielik-11B Q4_K_S | 38 | pending (`history-11b-v0`) | 39 |
| Bielik-11B Q3_K_M | 37 | - | - |
| Gemma-4-12B Q4_K_S | 40 with thinking on (4 items blank: the thinking ran past the token limit); thinking-off run pending | - (no adapter yet) | - |

With the only 4.5B LoRA so far, the 4.5B's combined score is 0.8 × 18 − 0.4 × 20 = 6.4 points (progress −2). The 11B
(base 38) beats that with a tuned score of 28 or more (27 ties). The 21 of the 4.5B extras-only run is not its
"best": it has no LoRA. Recompute the bar whenever a LoRA + extras run comes back. Then point section 8's
`EXAM_GGUF` / `EXAM_LORA` / `EXAM_SRV_ARGS`, the app registration (section 5) and the HF repo name (section 9) at
that one model.

**11B.** On the mock the 11B base scores far above the 4.5B base (local judge: 38 vs 20 of 60). Memory decides the
file (strict `--mem-check`, 1 slot, decimal GB; team cap 8.8 GB): Q4_K_S 8.34 GB (+0.13 GB adapter = ~8.47 GB) is
the pick, Q3_K_M 7.31 GB (~7.44 GB with the adapter) the fallback. Both score the same on the mock base (judge: 38 vs
37). The 11B always runs with one slot: 4 slots measured 10.5 GB, over the cap.

**Gemma-4-12B** (unsloth GGUF, 11.96B parameters). The file is IQ4_XS
(`/workspace/models/gemma-4-12b/gemma-4-12b-it-IQ4_XS.gguf`, 6.38 GB on disk), the memory-safe pick: 8.40 GB measured
at one slot, while the Q4_K_S (6.76 GB on disk) read 8.80-9.04 GB, at or over the 8.8 GB cap. The judged 40 comes from
a Q4_K_S base run with thinking **on**, which left 4 items blank (the thinking used up the output limit), so it does
not count. Gemma is always served with thinking **off**, the base run included:
`serve_model.sh ... -- --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'`, on the base and the
tuned server alike (`EXAM_SRV_ARGS` in sections 0 and 8). Its host RSS is about 1.1-1.3 GB, because its large
embedding table stays in host memory, so run `--mem-check` again after a run, not only at start. It needs its own
adapter (train + export) like the 11B.

The Gemma base mock command, thinking off (port 8083, one slot):

```bash
MG=/workspace/models/gemma-4-12b/gemma-4-12b-it-IQ4_XS.gguf
tmux new -d -s srvg "bash scripts/serve_model.sh --model $MG --port 8083 --parallel 1 -- --reasoning-budget 0 --chat-template-kwargs '{\"enable_thinking\":false}' 2>&1 | tee runs/srv-gemma.log"
until curl -sf http://127.0.0.1:8083/health; do sleep 2; done
python scripts/run_exam.py --exam-dir exams/mock --label base-gemma-iq4xs --base-url http://127.0.0.1:8083/v1 --parallel 1 \
  --descriptions runs/desc/mock.json
bash scripts/serve_model.sh --mem-check                     # AFTER the run: must still print "ok" (<= 8.8 GB)
tmux kill-session -t srvg
```

The 11B mock commands (they used port 8082 because the 4.5B server was on 8080):

```bash
M11=/workspace/models/bielik-11b/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf   # fallback: ...-Q3_K_M.gguf
# base (the untouched 11B): its own server on :8082 WITHOUT --lora, one slot
tmux new -d -s srv11 "bash scripts/serve_model.sh --model $M11 --port 8082 --parallel 1 2>&1 | tee runs/srv-11b.log"
until curl -sf http://127.0.0.1:8082/health; do sleep 2; done
python scripts/run_exam.py --exam-dir exams/mock --label base-11b-q4ks --base-url http://127.0.0.1:8082/v1 --parallel 1 \
  --descriptions runs/desc/mock.json
# LoRA: QLoRA on the same rows (~190 steps), then GGUF with the 11B config as the base
python scripts/train_history.py --config configs/history-11b.yaml --dev data/processed/history/dev_rag.jsonl \
  --protect-exam exams/mock/exam.json 2>&1 | tee runs/train-history-11b-v0.log
HF_BASE=speakleash/Bielik-11B-v3.0-Instruct bash scripts/export_lora_gguf.sh checkpoints/history-11b-v0/adapter
tmux kill-session -t srv11
tmux new -d -s srv11 "bash scripts/serve_model.sh --model $M11 --port 8082 --parallel 1 \
  --lora checkpoints/history-11b-v0/lora-f16.gguf 2>&1 | tee runs/srv-11b-lora.log"
until curl -sf http://127.0.0.1:8082/health; do sleep 2; done
bash scripts/serve_model.sh --mem-check                     # must print "ok" (<= 8.8 GB) with the adapter loaded
python scripts/run_exam.py --exam-dir exams/mock --label tuned-11b --base-url http://127.0.0.1:8082/v1 --lora-scale 1 \
  --parallel 1 --descriptions runs/desc/mock.json --style rag --rag-index data/rag/index --essay-mode plan --vote 5 \
  --repair-labels --essay-retry
```

### 6.6 Local rubric judge (between the hourly official grades)

Official grading takes one mock upload per hour, so we score every run ourselves first. Claude agents grade each item
against the CKE marking scheme ("Zasady oceniania" of the mock, a public PDF, kept outside the repo), just as the
organizers' "AI rubric assessment" does. This is build-time tooling; nothing here runs during the exam.

```bash
# the marking scheme as text (pdftotext -layout), once; keep it and the packets outside the repo
python scripts/judge_packets.py --exam-dir exams/mock --rubric-text ~/judge/zasady_2305.txt \
  --answers runs/<label>/answers.json --out runs/judge/<label>.packets.jsonl
# Claude Code: run the Workflow scripts/judge/cke_rubric_judge.workflow.js
#   with args {"packets": "runs/judge/<label>.packets.jsonl", "label": "<label>"}; save its result as runs/judge/<label>.judge.json
python scripts/judge_report.py runs/judge/base.judge.json runs/judge/tuned.judge.json --packets runs/judge/base.packets.jsonl
```

Calibration (2026-09-26): the judge graded the organizers' own Bielik-4.5B benchmark answers and matched their
per-item points on **36/37 items** (22 vs 23 total). The only difference is the essay (1 vs 2 of 15). Check with
`judge_report.py <judge.json> --vs bielik-4-5b.reviews.json --reasons`. What cost that run points: three image-only
items left out of their scope (7, 8, 15: 5 points), hedged answers ("X (lub Y)" earns 0), wrong factual links in
justifications, and an essay answering all three topics (only the first is graded).

## 7. Memory check (team cap: the running model <= 8.8 GB)

```bash
bash scripts/serve_model.sh --mem-check     # host RSS per server process + nvidia-smi per process / GPU total
```

- The team cap is 8.8 GB (8 GB + 10%), decimal GB, host RSS + GPU per server process, enforced by
  `serve_model.sh --mem-check` (`RAM_CAP_GB`). It is the team's own rule: the organizers' limit is 8 GB on disk per
  model (the LoRA and the knowledge base do not count), with no RAM limit.
- Bielik-4.5B server: Q8_0 weights 5.06 GB, KV about 1.07 GB (q8_0 at 32,768 tokens = 4 slots × 8192), and compute
  buffers. The LoRA adds about 0.1 GB. The estimate is about 6.6-6.8 GB. With `--parallel 1` the KV drops to about
  0.27 GB.
- Bielik-11B Q4_K_S: 8.34 GB at 1 slot, about 8.47 GB with the adapter; 4 slots measured 10.5 GB (over). Gemma-4-12B
  at 1 slot: IQ4_XS 8.40 GB (the registered file), Q4_K_S 8.80-9.04 GB (at or over the cap; 6.5b). The final runs use
  `--parallel 1` on every candidate.
- Gemma keeps about 1.1-1.3 GB in host RSS (its large embedding table stays on the host), so a reading at server
  start can be lower than the one during a run: run `--mem-check` again **after** a run (mock, rehearsal), not only
  at start, and with the adapter loaded for the tuned server.
- The VLM runs in its own pre-pass and is stopped first, so it does not add to the Bielik peak: Qwen3.5-4B Q8_0
  4.48 GB + mmproj 0.67 GB + KV.
- On CUDA, count the process's GPU memory plus its host RSS. If RSS is inflated by the memory-mapped GGUF, measure
  once with `bash scripts/serve_model.sh -- --no-mmap`.
- Over the cap? Lower `--parallel`, use `--kv q4_0` (CUDA/Metal), or use TurboQuant KV (`tbq4_0,pq4_0`), which works
  only on Vulkan/CPU: `--llama-bin /workspace/qvac-fabric-llm.cpp/build/bin --backend vulkan`, or `--engine qvac`.
- QVAC note: `qvac serve --openai` (port 11434 by default) ignores `logprobs`, `stop` and `n>1`, and defaults to temp
  0.8 / repeat_penalty 1.1. `serve_model.sh --engine qvac` pins greedy settings in the model config, and the harness
  sends them per request. QVAC has no per-request LoRA: restart the server with or without `--lora` between the base
  and tuned runs, and call `run_exam.py` without `--lora-scale` (plus `--base-url http://127.0.0.1:<qvac port>`;
  `--no-llama-extras` if it rejects `top_k`/`repeat_penalty`/`cache_prompt`).
- Laptop (Metal) fallback: `brew install llama.cpp`, then
  `bash scripts/serve_model.sh --llama-bin /opt/homebrew/bin --backend metal --parallel 1`.

## 8. Sunday: the final exam (11:00 unlock, problems reported by 11:15)

**The short version: one command on the box** (rehearsed on the mock 26.09: 533 s on a free L40S with cached image
descriptions, about 11 min with fresh ones; every server under the 8.8 GB cap: VLM 7.37, Bielik-11B 8.27, Gemma 8.59):

```bash
# laptop: scp <final zip> root@<box>:/workspace/fine-tune/exams/final.zip
cd /workspace/fine-tune && mkdir -p exams/final && unzip -o exams/final.zip -d exams/final
tmux new -s final 'bash scripts/final_exam.sh exams/final final 2>&1 | tee runs/final.log'
#   [+ EXAM_LORA=checkpoints/<run>/lora-f16.gguf in front if an adapter beat the harness-only system on the mock]
# at the end it prints the two upload files and the model list:
#   base : runs/final-base/answers.json   (untouched Bielik-11B, no adapter loaded, organizers' protocol)
#   tuned: runs/final-tuned/answers.json  (Bielik-11B + harness, essay included; ESSAY_MODEL=gemma: Gemma writes the essay)
```

The script runs one model at a time: Qwen3.5-4B image descriptions → untouched Bielik-11B base → Bielik-11B with the
harness (RAG with relevance cutoff 50, 9 votes on closed items, label repair, planned essay) → [only with
ESSAY_MODEL=gemma: Gemma-4-12B IQ4_XS with thinking for the essay, merged with `scripts/merge_answers.py`] → validate. It stops with
`FAILED: ...` on any error or if a server goes over the memory cap. Requests go out with `cache_prompt: false`, so a
run is reproducible. Upload both files (Final exam; base first); in the model list give every model the run used
(Bielik-11B GGUF link and Qwen3.5-4B GGUF link; plus Gemma-4-12B only with ESSAY_MODEL=gemma), and tick "biggest
improvement" on the tuned upload. Details: docs/SUNDAY_CHECKLIST.md.
The detailed manual steps below remain as the fallback.

**People:** presenter `<TBD>` and backup `<TBD>`, both on the registered lineup and on site (Kolektyw3) by 10:30;
`<TBD>` runs the exam commands below from 11:00 (not the presenter, so an early stage slot does not stop the runs).

**Before 10:00:**

- The exam model and its adapter were picked by 09:00 from the judged mock results (6.5b). `EXAM_GGUF`,
  `EXAM_LORA` and `EXAM_SRV_ARGS` in step 3 below point to that one model, and the app registration names the same
  GGUF (section 5).
- The box runs the frozen code. Its `/workspace/fine-tune` is not a git repo: copy any file under `harness/`,
  `scripts/` or `configs/` that changed on the laptop since (scp), and never commit from the box (section 9).
- `/workspace/models` holds the exam model's GGUF and the VLM; `$EXAM_LORA` is on the box.
- One full dry rehearsal on the mock with the frozen code and the step 3 commands, timed. Both runs use
  `--parallel 1` on the server and in `run_exam.py`: it is bit-reproducible, and more slots break the 8.8 GB team cap
  on the 11B (4 slots measured 10.5 GB). If the tuned run is too slow, use `--vote 3` on the tuned run, never more
  slots. With Gemma, run `--mem-check` again after each rehearsal run (section 7).

**At 10:50:**

- No model server is running: not the VLM, and not the exploratory servers (11B on :8082, Gemma on :8083).
  `nvidia-smi` lists no `llama-server`.
- The session is up, tmux is open, and the upload page is open with the TEAM_KEY at hand.

**From 11:00:**

```bash
# 1. the final pack unlocks at 11:00: the submissions page (and the guide) then shows "Download final exam ZIP".
#    Download it on the laptop, then: scp <zip> root@<box>:/workspace/fine-tune/exams/final.zip
mkdir -p exams/final && unzip -o exams/final.zip -d exams/final    # exam.json must sit next to images/
python -c "import json; e=json.load(open('exams/final/exam.json')); print(e['exam_id'], len(e['items']), e['max_points'])"
# 2. descriptions (VLM alone); retry failures once, then ALWAYS stop the VLM
tmux new -d -s vlm 'bash scripts/serve_model.sh --vlm 2>&1 | tee runs/vlm-final.log'
until curl -sf http://127.0.0.1:8081/health; do sleep 2; done
python scripts/describe_images.py --exam-dir exams/final || python scripts/describe_images.py --exam-dir exams/final
tmux kill-session -t vlm && sleep 2
# 3. the exam model (6.5b): keep exactly ONE pair uncommented; base and tuned serve the same GGUF
EXAM_GGUF=/workspace/models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf                         # Bielik-4.5B Q8_0 (default)
EXAM_LORA=checkpoints/history-v0/lora-f16.gguf                                          # or the 4.5B run picked in 6.5b
EXAM_SRV_ARGS=                                                                          # Bielik: no extra server flags
# EXAM_GGUF=/workspace/models/bielik-11b/speakleash_Bielik-11B-v3.0-Instruct-Q4_K_S.gguf   # Bielik-11B Q4_K_S
# EXAM_LORA=checkpoints/history-11b-v0/lora-f16.gguf
# EXAM_SRV_ARGS=
# EXAM_GGUF=/workspace/models/gemma-4-12b/gemma-4-12b-it-IQ4_XS.gguf                     # Gemma-4-12B IQ4_XS
# EXAM_LORA=checkpoints/<gemma run>/lora-f16.gguf
# EXAM_SRV_ARGS="-- --reasoning-budget 0 --chat-template-kwargs '{\"enable_thinking\":false}'"   # Gemma: thinking OFF
ls -l "$EXAM_GGUF" "$EXAM_LORA"; echo "server flags: ${EXAM_SRV_ARGS:-none}"
# 3a. BASE: a server WITHOUT --lora (a LoRA loaded at scale 0 changed 23/37 mock answers, 6.4), one slot
tmux new -d -s srv "bash scripts/serve_model.sh --model $EXAM_GGUF --parallel 1 $EXAM_SRV_ARGS 2>&1 | tee runs/srv-final-base.log"
until curl -sf http://127.0.0.1:8080/health; do sleep 2; done
bash scripts/serve_model.sh --mem-check                                  # must print ok (<= 8.8 GB)
python scripts/run_exam.py --exam-dir exams/final --label final-base  --parallel 1 --descriptions runs/desc/final.json
python scripts/validate_answers.py runs/final-base/answers.json  --exam-dir exams/final     # then upload it at once (step 4)
# 3b. TUNED: stop the base server, restart the SAME GGUF with the adapter applied at scale 1, same --parallel 1
tmux kill-session -t srv && sleep 2
tmux new -d -s srv "bash scripts/serve_model.sh --model $EXAM_GGUF --parallel 1 --lora $EXAM_LORA --lora-apply $EXAM_SRV_ARGS 2>&1 | tee runs/srv-final-tuned.log"
until curl -sf http://127.0.0.1:8080/health; do sleep 2; done
curl -s http://127.0.0.1:8080/lora-adapters                              # [{"id":0,"path":...,"scale":1}]
bash scripts/serve_model.sh --mem-check                                  # must print ok with the adapter loaded
python scripts/run_exam.py --exam-dir exams/final --label final-tuned --lora-scale 1 --parallel 1 --descriptions runs/desc/final.json \
  --style rag --rag-index data/rag/index --essay-mode plan --vote 5 --repair-labels --essay-retry
bash scripts/serve_model.sh --mem-check                                  # after the run, for the log (Gemma: section 7)
# 4. validate (errors = the site would reject the file) and upload BOTH, each as soon as it validates: exam "Final",
#    names nuori-final-base-<model> / nuori-final-tuned-<model> (e.g. bielik45b-q8, bielik11b-q4ks, gemma12b-iq4xs)
python scripts/validate_answers.py runs/final-tuned/answers.json --exam-dir exams/final
```

- The two servers differ only in `--lora $EXAM_LORA --lora-apply`: same GGUF, `--parallel 1`, `$EXAM_SRV_ARGS`,
  context and descriptions file. Both runs must print the same descriptions sha256. The `tmux` commands use double
  quotes so the variables expand in the current shell; `$EXAM_SRV_ARGS` stays last on the line (everything after its
  `--` goes to `llama-server`).
- On both uploads, list every model in the solution: the exam model (the GGUF registered in section 5) and
  `unsloth/Qwen3.5-4B-GGUF` (image descriptions only). On the **tuned** upload, tick the optional "Enter the biggest
  improvement category" if we compete for progress.
- Upload `final-base` as soon as it validates (while the tuned run is going): without the benchmark of the model
  that sits the exam, the team is disqualified.
- Use the final pack's own `exam_id` and template, never the mock's. `run_exam.py` takes both from the pack.
- The base command stays bare (no `--style rag`, no extras): it is the organizers' protocol. Only the tuned command
  carries the extras of section 6.5, and `data/rag/index` must be on the box before 11:00.
- If an item fails or the server dies: restart that run's server with the step 3a or 3b command (base: without
  `--lora`; tuned: with it; both with `$EXAM_SRV_ARGS`), then run
  the same `run_exam.py` command again with `--resume` added. It reuses every answer already in
  `runs/<label>/run_log.jsonl` and asks only the missing items.
- A few images still without a description after the retry: run anyway (their items keep the `[Obraz: ...]`
  marker; base and tuned see the same text). The run prints which images are missing.
- If a step blocks (download, server, upload), post in the Telegram group (https://t.me/warsawmodeltrainers) with the
  team name and the error **before 11:15**. Later reports are not considered.
- Keep both receipts and the `runs/final-*/run_log.jsonl` files for the presentation. Do not commit the logs: they
  contain exam text.

## 9. Freeze (before 11:00)

Two machines, two jobs. The **box** only uploads the adapter to the Hugging Face Hub (step 2) and runs the exam. The
**laptop** does all of git (step 3), in the cleaned tree (`~/soft/pets/ml_hack/fine-tune` on the laptop that ran the
cleanup). Never run `git init`, `git add` or `git commit` on the box: its `/workspace/fine-tune` is the pre-cleanup
copy, not a git repo, and it still holds files the cleanup removed or changed (e.g. the exam snippets in
`data/synthetic/dedup_report.json`, the old `.gitignore`). If a file made on the box must go into the repo, copy just
that file to the laptop (scp) and read it before adding it.

1. `SOURCE.md` contains the required sentence (checked into the repo root). The README points here.
2. **On the box:** upload the frozen weights. The organizers need the untouched base (the GGUF link registered in
   section 5) and the trained model. Upload the exam model's adapter (PEFT dir + GGUF, `$EXAM_LORA` of section 8) to
   the Hugging Face Hub, in a repo named after that model. The weights follow the base model's licence (Apache-2.0
   for Bielik, the Gemma terms for Gemma) and are for education and research only, because they are built from
   third-party exam content.
   - **Model card first.** The `README.md` that PEFT writes into the adapter folder has no licence. The commands below
     replace it with our card before the upload, and it becomes the repo's model card. Front matter: `license:`
     (`apache-2.0` for a Bielik adapter; for a Gemma adapter the Gemma terms, i.e. the `license:` value on the
     `google/gemma-4-12b-it` card) and `base_model:` (the HF repo of the exam model's base:
     `speakleash/Bielik-4.5B-v3.0-Instruct`, `speakleash/Bielik-11B-v3.0-Instruct` or `google/gemma-4-12b-it`). The
     body says: "Trained on data derived from CKE exam papers (© CKE); education and research use only".
   - **Access for the organizers:** a private HF repo cannot be shared with single accounts. The repo is created
     private, and once the card and the files are up it is opened: either **public**, or **public and gated with
     manual approval** (the organizers request access on the repo page, and we approve them in the repo's settings,
     under access requests).
   - The upload needs a **write** token (the box's `HF_TOKEN` secret is read-only). Paste it only into this shell
     (`read -rs`), never into a file, the repo or a commit message: **never commit the token**. It is passed with
     `--token` and unset at the end (no `hf auth login`, so it is not stored on the box either).
   - `training_args.bin` (a pickle with local paths) stays out of the upload.
   ```bash
   # EXAM_LORA set as in section 8, step 3 (checkpoints/<run>/lora-f16.gguf; its PEFT dir is checkpoints/<run>/adapter)
   ADAPTER_DIR="$(dirname "$EXAM_LORA")/adapter"
   BASE_REPO=speakleash/Bielik-4.5B-v3.0-Instruct           # or speakleash/Bielik-11B-v3.0-Instruct, google/gemma-4-12b-it
   LICENSE_ID=apache-2.0                                     # Bielik; Gemma: the license: value on the google/gemma-4-12b-it card
   HF_REPO=<hf-user>/nuori-<exam model>-history-lora        # e.g. nuori-bielik-4.5b-, nuori-bielik-11b-, nuori-gemma-4-12b-history-lora
   printf '%s\n' '---' "license: $LICENSE_ID" "base_model: $BASE_REPO" 'library_name: peft' 'tags:' '- lora' '- polish' \
     '---' '' "# $HF_REPO" '' \
     "LoRA adapter for $BASE_REPO, trained by team Nuori at the Warsaw Model Trainers hackathon (September 2026) for the" \
     'Polish history matura (extended level). The PEFT files are at the repo root; lora-f16.gguf is the same adapter' \
     'for llama.cpp (llama-server --lora).' '' \
     'Trained on data derived from CKE exam papers (© CKE); education and research use only.' '' \
     "Licence: $LICENSE_ID, the licence of the base model $BASE_REPO (its model card has the terms)." \
     > "$ADAPTER_DIR/README.md"
   head -3 "$ADAPTER_DIR/README.md"                         # the license: and base_model: lines are filled in
   read -rs HF_WRITE_TOKEN                                  # paste the write token; not echoed, not in shell history
   hf repo create "$HF_REPO" --private --token "$HF_WRITE_TOKEN"
   hf upload "$HF_REPO" "$ADAPTER_DIR" . --exclude training_args.bin --token "$HF_WRITE_TOKEN"   # PEFT files + card, repo root
   hf upload "$HF_REPO" "$EXAM_LORA" lora-f16.gguf --token "$HF_WRITE_TOKEN"
   hf repo settings "$HF_REPO" --public --token "$HF_WRITE_TOKEN"          # or gated: --public --gated manual
   unset HF_WRITE_TOKEN
   ```
3. **On the laptop**, in the cleaned tree (never on the box): commit and tag. `git status` must show no exam packs,
   runs or descriptions, and no `*.gguf` or weights, and the committed `dedup_report.json` must hold no exam text.
   Commit only when both checks print clean.
   ```bash
   cd ~/soft/pets/ml_hack/fine-tune          # the laptop's cleaned tree
   git add -A && git status --short          # read it: nothing from exams/, runs/, checkpoints/, data/processed/
   # path check; tests/fixtures/history_exam/ is our made-up fixture (exam.json, answers-template.json, images/)
   git status --short | grep -v ' tests/fixtures/history_exam/' | grep -E 'exams/|runs/|checkpoints/|data/processed|\.zip|\.tgz|\.gguf|\.safetensors|\.pdf|descriptions|exam\.json|answers-template|images/|blocklist/.*jsonl|ext_llmzszl|ext_prawko_|llm_filled_answers|data/history/filled_|data/rag/(index|articles|resolved|build_report)' \
     && echo "STOP: exam text or weights staged" || echo "clean"
   # content check: the file is committed, so only its content shows exam text coming back
   grep -q '"matched_stem"\|"matched_answer"' data/synthetic/dedup_report.json \
     && echo "STOP: exam text in dedup_report" || echo "dedup_report clean"
   git commit -m "Freeze for the final exam" && git tag final && git push && git push --tags
   ```
4. If the repo is private, give the organizers and the jury read access. Enter the repo URL in the app
   (https://warsawmodeltrainers.dev/matura, "Update team") and check that its base model is the exam model
   (section 5). The app has no field for the trained weights: put the HF adapter link in the README (on the laptop,
   before the step 3 commit). The team lineup was due Saturday 16:00.

## 10. Commands and who owns them

| Step | Command | Owner / file |
|---|---|---|
| box setup | `scripts/setup_labqoat.sh` | this runbook's author |
| tokenizer parity | `scripts/check_tokenizer.py` | " |
| serve (text, LoRA, VLM, QVAC) | `scripts/serve_model.sh` | " |
| train | `scripts/train_history.py` + `configs/history.yaml` | " |
| export LoRA | `scripts/export_lora_gguf.sh` | " |
| prompts / postprocess | `harness/prompts.py`, `harness/postprocess.py` | builder A |
| image descriptions | `scripts/describe_images.py`, `harness/vision.py` (`docs/VISION.md`) | builder B |
| sit the exam / validate | `scripts/run_exam.py`, `scripts/validate_answers.py` | harness builder |
| history rows | `scripts/build_history_data.py` (`train/history_data.py`, `docs/DATA.md`) → `data/processed/history/{train,dev,dev_rag}.jsonl` | data builder |
| knowledge base | `scripts/build_rag.py` (`harness/rag.py`) → `data/rag/index` | RAG builder |
| synthetic items, filled answers | `data/history/synthetic/*.jsonl` (committed), `data/history/llm_filled_answers.jsonl`, `data/history/filled_old/` (gitignored) | writing streams |

## 11. Troubleshooting

- **`run_exam.py` exits 2 with "cannot load the RAG index":** the index is missing or unreadable. Check
  `ls data/rag/index` and rebuild or copy it (6.5). `--style rag` without `--rag-index`, or `--rag-index` without
  `--style rag`, is refused on purpose (the base run must stay without passages). A retrieval error during the run
  only costs that item its passages (logged as `rag.error`), never the answer.

- **`llama-server` rejects a flag:** `serve_model.sh` adds optional flags only when `llama-server --help` lists them.
  Print the command with `--dry-run` and compare it with `llama-server --help`.
- **Essay truncated or "context exceeded":** each slot has `--ctx` tokens (default 8192). The essay needs prompt +
  4096 tokens. Do not lower `--ctx`; lower `--parallel` instead.
- **Tuned answers identical to base:** the request did not carry the LoRA. Check `curl :8080/lora-adapters`, the
  `--lora-scale 1` flag and the run log's `lora_scale`.
- **`train_history.py` drops rows (`dropped_too_long_by_kind`):** rows over `model.max_length` (6144 in
  `configs/history.yaml`) are not trained. The 2026-09-26 build has none: its longest row is about 5.4k tokens (an
  essay with passages; counted with the Bielik-11B tokenizer, an upper bound for the 4.5B).
- **Training OOM on the L4:** `--set train.per_device_batch_size=1 --set train.grad_accum=16` (same effective batch).
  The long essay rows are the peak; do not lower `max_length` to save memory, since that drops them.
- **`qvac serve` rejects the generated config:** `serve_model.sh --engine qvac` writes `$WORK/.serve/qvac.config.*.json`
  with `"type": "llm"` (the documented ModelEntry type) and the llama addon keys (`ctx_size`, `cache-type-k`, ...). If
  a key or the type is refused, try `"type": "llamacpp-completion"` (as in `docs/VISION.md`) and check the accepted
  keys with `qvac configure`.
- **GGUF conversion fails on imports:** re-run `setup_labqoat.sh` (converter venv), or point `CONVERT_PYTHON` to a
  Python with llama.cpp's `requirements/requirements-convert_lora_to_gguf.txt`.

## Measurement log (fill in)

| What | Value | When / who |
|---|---|---|
| mock base run time (parallel 4 / 1) | 4.5B Q8_0: - / 537 s; 11B Q4_K_S: - / 627 s (L40S, training alongside) | 2026-09-26 |
| mock tuned run time (bare / with the 6.5 extras) | | |
| Exam-model server memory (RSS + GPU, decimal GB, --mem-check, cap 8.8) | 4.5B Q8_0 4 slots 7.20; 11B Q4_K_S 1 slot 8.34 (ok), 4 slots 10.5 (OVER); 11B Q3_K_M 1 slot 7.31; Gemma-4-12B Q4_K_S 1 slot 8.80-9.04 (at/over), IQ4_XS 1 slot 8.40 (ok; host RSS ~1.1-1.3 of it, check after a run) | 2026-09-26 |
| VLM server memory | | |
| training time history-v0 | | |
| tokenizer check result | | |
| local judge vs organizers (their Bielik run) | 36/37 items exact, 22 vs 23 | 2026-09-26, Claude |
| local judge: base 4.5B / base 11B / base Gemma / tuned (mock, of 60; 6.5b) | 20 / 38 (Q4_K_S), 37 (Q3_K_M) / Q4_K_S with thinking 40 (4 blank), thinking off pending / 4.5B LoRA v0 + extras 18; no-LoRA references: 4.5B extras only 21, 11B extras only 39 | 2026-09-26 |
