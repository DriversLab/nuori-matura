# Image descriptions: the VLM pre-pass

`harness/vision.py` + `scripts/describe_images.py`. Bielik reads text only, so before the text model runs, a small
vision-language model (VLM) describes every exam image once. The harness then puts each description where the image
was: it replaces each `[Obraz: images/X.png]` marker, and appends the description when an image has no marker (items
7, 8 and 15 on the mock). The same cached descriptions feed the **base** and the **tuned** run.

Nothing in this document was run against a real VLM on the build machine (no model may be loaded there). File names
and sizes were read from the Hugging Face API, the QVAC registry and the llama.cpp sources on 2026-09-26. RAM and time
figures are estimates. Measure them on the first real run.

## Why a separate pre-pass rather than one VLM for everything

1. **The organizers' base protocol is text plus fixed descriptions.** Their benchmark system prompt says *"obrazy
   zastąpiono opisami"* (the images were replaced with descriptions), and their methodology reads: "Fixed Polish
   descriptions replace images. Scanned text is transcribed; tables preserve their data; trees preserve relationships.
   Do not insert solutions, inferred identities or interpretations. All models see exactly the same frozen
   descriptions." Our base run has to follow that protocol, so the text model gets text.
2. **The Polish history knowledge lives in Bielik.** Bielik-4.5B-v3.0-Instruct has no vision tower. The CKE
   answering skill and our LoRA sit on Bielik. A 4B VLM's job is only to see: read labels, legends, scans and
   captions, and say what is depicted.
3. **Progress is measured as tuned minus base.** Both runs read identical cached descriptions, so the delta shows the
   fine-tune and not vision noise.
4. **RAM.** The VLM runs alone and is stopped before Bielik loads. Peak RAM is therefore the larger of the two
   models, not their sum, and both fit the 8.8 GB team cap (8 GB + 10%, host RSS + GPU).
5. **Train/serve consistency.** The instruction copies the organizers' description style: prose that starts with the
   kind of material, quotes labels in „…”, and uses lists for legends, tables and family trees. Training prompts built
   from organizer-style descriptions therefore look like exam-time prompts.

## Order of operations on exam day (offline)

```bash
# 1. start the VLM alone (section "Launch"), then:
python scripts/describe_images.py --exam-dir exams/final --base-url http://127.0.0.1:8081 --model vlm
#    -> runs/desc/final.json   (re-run = only missing/failed images; --force redoes all)
# 2. stop the VLM (Ctrl-C / kill; for QVAC: curl -X DELETE http://127.0.0.1:11434/v1/models/vlm)
# 3. start Bielik (base GGUF + LoRA), then
python scripts/run_exam.py --exam-dir exams/final --label final-base  --lora-scale 0 --parallel 1 --descriptions runs/desc/final.json
python scripts/run_exam.py --exam-dir exams/final --label final-tuned --lora-scale 1 --parallel 1 --descriptions runs/desc/final.json
```

(Labels as in docs/RUNBOOK.md section 8: `--label base` would overwrite the mock's `runs/base/`.)

`--dry-run` lists the images and cache hits and prints the first prompt, without a server. The exit code is 0 when
every image is done, 1 when some failed (re-run to retry only those), and 2 when the server is unreachable.

## Recommended VLMs (each must fit in 8 GB when run alone)

| | Qwen3.5-4B (**primary**) | Gemma 4 E4B QAT (second opinion) |
|---|---|---|
| Repo | `unsloth/Qwen3.5-4B-GGUF` (apache-2.0, not gated) | `google/gemma-4-E4B-it-qat-q4_0-gguf` (apache-2.0, not gated) |
| Weights | `Qwen3.5-4B-Q4_K_M.gguf` 2.741 GB · `Qwen3.5-4B-Q8_0.gguf` 4.482 GB | `gemma-4-E4B_q4_0-it.gguf` 5.155 GB |
| Projector | `mmproj-F16.gguf` 0.672 GB | `gemma-4-E4B-it-mmproj.gguf` 0.992 GB |
| QVAC registry | `QWEN3_5_4B_MULTIMODAL_Q4_K_M` (same unsloth file) or `_Q6_K` 3.526 GB + `MMPROJ_QWEN3_5_4B_MULTIMODAL_F16`. No Q8_0 weights constant: use a file path. | Only bartowski non-QAT files: `GEMMA4_4B_MULTIMODAL_Q4_K_M` 5.405 GB + `MMPROJ_GEMMA4_4B_MULTIMODAL_F16` 0.990 GB. For the Google QAT file, use a file path. |
| KV cache at 8192 ctx | 8 of 32 layers are full attention, 4 KV heads × 256 dim: 32 KiB/token → 0.27 GB f16, 0.14 GB q8_0 | 2 KV heads, mostly 512-token sliding window, 18 of 42 layers share KV: well under 0.5 GB |
| Image tokens (llama.cpp) | 32 px per token (patch 16, merge 2), limits 8–4096. Mock images (≤ 1539×1284 px) come to ≤ ~1,650 tokens. | 48 px per token (patch 16, pool 3), limits 70–1120 |
| RAM estimate (weights + mmproj + KV + ~1 GB buffers) | Q4_K_M ≈ 4.5 GB · Q8_0 ≈ 6.3 GB | ≈ 7.3 GB (tight: measure it) |
| Thinking | On by default: must be disabled (done per request and by the server flags below) | Disabled the same way |

**Pick:** Qwen3.5-4B at Q8_0 if the RAM measurement allows, otherwise Q4_K_M, the QVAC registry file. Compare it
against Gemma 4 E4B on the mock with `--reference` (see "Choosing a VLM"), and keep whichever transcribes the Polish
labels more faithfully (diacritics on Z20-S2, Z21, Z24-S1).

Other sizes in the same repos, all verified: unsloth `Qwen3.5-4B-Q6_K.gguf` 3.526 GB, `Qwen3.5-4B-UD-Q4_K_XL.gguf`
2.912 GB, `mmproj-BF16.gguf` 0.676 GB. `ggml-org/gemma-4-E4B-it-GGUF` has `gemma-4-E4B-it-Q4_0.gguf` 4.591 GB and
`mmproj-gemma-4-E4B-it-Q8_0.gguf` 0.560 GB. That smaller projector should also pair with the Google QAT weights,
since the vision tower is the same, but nobody has tested it.

### Download (weights only on the machine that runs them, never into the repo)

```bash
export VLM_DIR=$HOME/models/vlm
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf mmproj-F16.gguf --local-dir $VLM_DIR/qwen3.5-4b
#   (or Qwen3.5-4B-Q4_K_M.gguf instead of Q8_0)
hf download google/gemma-4-E4B-it-qat-q4_0-gguf gemma-4-E4B_q4_0-it.gguf gemma-4-E4B-it-mmproj.gguf \
  --local-dir $VLM_DIR/gemma-4-e4b-qat
```

## Launch

### llama-server (upstream llama.cpp; Metal on Mac, CUDA on the L4, or CPU)

```bash
# Qwen3.5-4B, 2 parallel slots of 8192 tokens each (use -c 8192 -np 1 with --parallel 1)
llama-server -m $VLM_DIR/qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf --mmproj $VLM_DIR/qwen3.5-4b/mmproj-F16.gguf \
  -c 16384 -np 2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  --image-min-tokens 1024 --image-max-tokens 2048 \
  --jinja --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}' \
  --host 127.0.0.1 --port 8081

# Gemma 4 E4B QAT
llama-server -m $VLM_DIR/gemma-4-e4b-qat/gemma-4-E4B_q4_0-it.gguf \
  --mmproj $VLM_DIR/gemma-4-e4b-qat/gemma-4-E4B-it-mmproj.gguf \
  -c 16384 -np 2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --image-max-tokens 1120 \
  --jinja --reasoning-budget 0 --host 127.0.0.1 --port 8081
```

- `-ctk/-ctv q8_0` is the right KV type on CUDA and Metal; TurboQuant KV (`tbq4_0`/`pq4_0`) is QVAC-only and works on
  CPU and Vulkan only. The VLM's KV is tiny anyway. On CPU use `-ngl 0`; f16 KV is fine.
- `--image-min-tokens 1024` for Qwen follows llama.cpp's own warning that Qwen-VL models need at least 1024 image
  tokens. It upscales small crops such as Z24-S1 (1020×354 px, ~350 native tokens) so small print stays legible.
  `--image-max-tokens 2048` never downsamples the mock images. For Gemma, 1120 is the projector's maximum.
- `-np 2` matches the script's default `--parallel 2`. If `--parallel` is higher than the slot count, requests just
  queue.
- Memory check: Linux `/usr/bin/time -v llama-server …` ("Maximum resident set size"). macOS `/usr/bin/time -l …`;
  Metal buffers are unified memory, so also watch Activity Monitor.

### QVAC (`qvac serve --openai`)

QVAC takes images as base64 `image_url` data URIs, PNG or JPEG only, which is how the script sends them. It routes
Qwen3.5 thinking to `reasoning_content` and honours `temperature`, `top_p`, `max_tokens` and `reasoning_budget` per
request. It ignores `stop`, `logprobs` and `n>1`, and warns on `seed`. Its defaults (temp 0.8, repeat_penalty 1.1,
top_k 40, seed -1) are therefore also pinned in the model config:

```json
{
  "serve": {
    "models": {
      "vlm": {
        "model": "QWEN3_5_4B_MULTIMODAL_Q4_K_M",
        "preload": true,
        "config": {
          "projectionModelSrc": "MMPROJ_QWEN3_5_4B_MULTIMODAL_F16",
          "device": "gpu", "gpu_layers": 99, "ctx_size": 8192,
          "reasoning_budget": 0, "temp": 0, "top_k": 1, "seed": 42, "repeat_penalty": 1.0,
          "flash-attn": "on", "cache-type-k": "q8_0", "cache-type-v": "q8_0"
        }
      }
    }
  }
}
```

```bash
qvac serve --openai -c qvac.vlm.json -p 8081
python scripts/describe_images.py --exam-dir exams/final --base-url http://127.0.0.1:8081 --model vlm --parallel 1
```

- To use local files (Qwen Q8_0, Gemma QAT), replace the `model` key with
  `"src": "/abs/path/Qwen3.5-4B-Q8_0.gguf", "type": "llamacpp-completion"` and set
  `"projectionModelSrc": "/abs/path/mmproj-F16.gguf"`. Config fields ending in `ModelSrc` accept constants or paths.
- On CPU (`"device": "cpu"`), QVAC keeps f16 KV by default. Drop the `cache-type-*` keys there.
- **One server for both models:** put `vlm` and the Bielik alias in the same config with `preload: false` and
  `serve.load.concurrency: 1`. After the pre-pass, `curl -X DELETE http://127.0.0.1:11434/v1/models/vlm` unloads the
  VLM; the first Bielik request then lazy-loads Bielik. Peak RAM stays at the larger model.

## What the script sends

- One request per **unique image by sha256**, so byte-identical files are described once. Shared images (Z04-S2 for
  4.1/4.2, Z05-* for 5.1–5.3, Z09-S2, Z13-*, Z14) get one description, with the questions of every item that uses
  them in the context.
- A user turn with the **image part first** (data URI), then the Polish instruction. The instruction carries:
  - the source text around the marker (caption above, attribution below; this image as `[TEN OBRAZ]`, other images
    as `[inny obraz]`), and the item questions;
  - the kind first (mapa / ilustracja / fotografia / karykatura / moneta / tabela / skan tekstu / schemat / plan /
    tablica genealogiczna);
  - every visible label quoted verbatim in „…”;
  - scans transcribed in full, tables with all their data, family-tree and diagram relations kept, map legend as a
    list;
  - people, symbols, buildings and style features described, but **no answer**: no A–D choice, no P/F verdict, no
    inferred identity, epoch, style, event or date that is not written on the image;
  - plain prose of 80–250 words.
- `temperature 0, top_k 1, top_p 1, seed 42, max_tokens 700, repeat_penalty 1.0`, with
  `chat_template_kwargs.enable_thinking=false` (llama-server) and `reasoning_budget: 0` (QVAC). Each server ignores the
  other's field.
- Output cleanup: `<think>` blocks and Markdown are stripped. If the output is a greedy loop (a phrase repeated 6+
  times or a line 4+ times), the image is re-asked once with `presence_penalty 1.0, repeat_penalty 1.1`, still greedy;
  if the second answer loops too, the repetitions are collapsed. An empty answer counts as a failure and is not
  cached.

## Cache

`runs/desc/<exam-dir name>.json` from the script (override with `--cache`; the library default for
`describe_images(..., cache_path=None)` is `<exam-dir>/descriptions.json`):
`{"images/X.png": {"sha256": "...", "description": "...", "model": "...", "ocr": null}}`.

- An entry is reused when the image file's **sha256** matches, whatever path or model produced it. A new path with the
  same bytes gets a copy of the entry. `--force` redoes everything; `--only images/X.png` limits a run to named
  images.
- The file is rewritten after every finished image, so an interrupted run loses nothing. Failed images are never
  written.
- `model` is the name the server reports: the GGUF file name on llama-server, the alias on QVAC.
- Readers (`harness.exam_io.load_descriptions`, `harness.vision.load_descriptions(cache, exam_dir, items)`) drop
  entries whose sha256 does not match the current package. Exam packages reuse names like `Z01.png`, so a leftover
  mock cache can never leak into the final exam.
- **Never commit it:** it contains transcriptions of CKE material. `.gitignore` covers `runs/`, `exams/` and
  `descriptions*.json`.

## Choosing a VLM on the mock (`--reference`)

The organizers' own mock descriptions (16 images; tasks 7, 8 and 15 have none because the benchmark dropped them) can
serve as a yardstick. Put them in a JSON `{"images/Z01.png": "...", ...}` outside the repo. Then:

```bash
python scripts/describe_images.py --exam-dir exams/mock --cache runs/desc/mock.qwen.json  --reference /path/organizer_mock.json
python scripts/describe_images.py --exam-dir exams/mock --cache runs/desc/mock.gemma.json --reference /path/organizer_mock.json
```

Per image, the script prints `label_recall` (the share of the reference's quoted labels found verbatim, which rewards
faithful OCR of names like „RAMORINO”, „Łomża”) and `word_recall` (the share of the reference's content-word stems
found, a rough coverage measure), plus the means. Cached results are re-scored without a server. These are crude
proxies. Also read Z13-A/B (map labels), Z20-S2, Z21 and Z24-S1 (scanned print) by eye.

## Optional OCR cross-check

If `tesseract` with Polish data is installed (`tesseract --list-langs` shows `pol`), the script runs
`tesseract <image> stdout -l pol` after each description. It stores the raw text in `ocr` and prints `ocr-coverage`,
the share of OCR words that the VLM also transcribed. A low value on a text-heavy image means the VLM skipped text.
`--ocr on` warns when tesseract is missing, and `--ocr off` disables the check. Raw OCR is noisy on photos and maps.
The harness therefore ignores it by default (`exam_io.load_descriptions(include_ocr=False)`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `failed: empty description` | Thinking was not disabled, so the whole budget went to `<think>`. Add `--reasoning-budget 0` / `--chat-template-kwargs '{"enable_thinking":false}'` (llama-server) or `"reasoning_budget": 0` in the QVAC config. |
| QVAC `400 unsupported image type` | Only PNG/JPEG data URIs are accepted. The exam packages are PNG. |
| Output loops or stops at `[hit max_tokens]` | The retry handles most loops. Otherwise raise `--max-tokens` (up to ~1000 still fits 8192 ctx with ≤ 2048 image tokens), or lower `--image-max-tokens`. |
| Exit code 2 | Start the server first; `curl http://127.0.0.1:8081/v1/models` must answer. |
| Very slow on CPU | Expected: a 24-layer vision encoder over ~1,600 tokens per image. Use `--parallel 1` on CPU, or run the pre-pass on the GPU box and copy the cache file over (entries are checked by sha256). |
