# Bielik model facts for LoRA/QLoRA fine-tuning on matura-style questions

I checked everything I could on Sep 16, 2026, using the HF API, raw files, the tech report PDFs and code run in the installed venv (transformers 5.17.0, trl 1.13.0, peft 0.21.0). Three repos are gated and gave 401 without a token: 4.5B, 1.5B and 11B-v2.6/v3.0. For those I took `config.json` and tokenizer files from SpeakLeash's own ungated copies (`*-FP8-Dynamic`, `*-MLX-8bit`, `*-bnb-4bit`). The architecture fields, parameter counts and file sizes in those copies match the gated repos' API data exactly. Anything I could not confirm is marked UNVERIFIED.

## 1. Model summary

| | 1.5B-v3.0-Instruct | 4.5B-v3.0-Instruct | 11B-v2.3-Instruct | 11B-v2.6-Instruct | 11B-v3.0-Instruct |
|---|---|---|---|---|---|
| Gating (API `gated`) | `auto` | `auto` | **False** (open) | `auto` | `auto` |
| License | apache-2.0 | apache-2.0 | apache-2.0 | apache-2.0 (quant cards add bielik.ai/terms) | apache-2.0 (same note) |
| `architectures` / `model_type` | LlamaForCausalLM / llama | LlamaForCausalLM / llama | **MistralForCausalLM / mistral** | LlamaForCausalLM / llama | LlamaForCausalLM / llama |
| Built from | Qwen2.5-1.5B, deepened, tokenizer swapped | Qwen2.5-3B, 36 → 60 layers, tokenizer swapped | merge of v2.0/2.1/2.2 (Mistral-7B-v0.2, deepened to 50 layers) | SFT + DPO-P + GRPO on Bielik-11B-v2 | SFT + DPO-P + GRPO on Bielik-11B-v3-Base-20250730 |
| hidden / layers / intermediate | 1536 / 32 / 8960 | 2048 / 60 / 11008 | 4096 / 50 / 14336 | 4096 / 50 / 14336 | 4096 / 50 / 14336 |
| heads / kv_heads / head_dim | 12 / 2 / 128 | 16 / 2 / 128 | 32 / 8 / 128 | 32 / 8 / 128 | 32 / 8 / 128 |
| `attention_bias` / `mlp_bias` | **true / true** | **true / true** | false / false | false / false | false / false |
| `vocab_size` | 32000 (APT4 tokenizer) | 32000 (APT4) | 32128 | 32128 | 32128 |
| `max_position_embeddings` | **8192** | 32768 (instruct SFT used 8192) | 32768 | 32768 | 32768 (YaRN to 131072 per report; `rope_scaling` is null in config) |
| rope_theta / rms_norm_eps / act | 1e6 / 1e-6 / `silu` | 1e6 / 1e-6 / `silu` | 1e6 / 1e-5 / `silu` | 1e6 / 1e-5 / `silu` | 1e6 / 1e-5 / `silu` |
| `tie_word_embeddings` | false | false | false | false | false |
| Weight dtype (safetensors) | BF16 | BF16 | **F16** (`torch_dtype: float16`) | BF16 | BF16 |
| Total params (API) | 1,596,507,648 | 4,757,260,288 | 11,168,796,672 | 11,168,796,672 | 11,168,796,672 |
| Safetensors on disk | 3.19 GB (1 file) | 9.51 GB (2 shards) | 22.34 GB (5 shards) | 22.34 GB (5) | 22.34 GB (5) |
| Card default `inference.parameters.temperature` | 0.4 | 0.4 | 0.2 | 0.2 | 0.2 |
| `generation_config.json` | UNVERIFIED (gated); FP8 copy: `eos_token_id [4,2]` | same as 1.5B | `eos [32001,2]`, `pad 2`, no sampling params | FP8 copy: `eos [32001,2]`, `pad 2` | same as v2.6 |

- **Parameter counts:** I built each config on the meta device with transformers 5.17 and got exactly the API totals. `silu` is a valid `ACT2FN` key.
- **Generation settings:** the `generation_config.json` files set no sampling parameters. SpeakLeash's vLLM examples in the FP8 cards use `temperature=0.2, top_p=0.95, max_tokens=4096`, and `temperature=0` for their SGLang example.
- **Card inconsistencies:** the 4.5B and 1.5B cards use the id `speakleash/Bielik-4.5B-v3-Instruct` in their code. The org listing only has `speakleash/Bielik-4.5B-v3.0-Instruct`, so use the v3.0 id. The cards also say "4.6B" and "1.6B" parameters.
- **Newer gated variants (all `auto`):**
  - `Bielik-PL-11B-v3.0-Instruct` (2026-02): the 11B v3 model with the APT4 tokenizer, 11,167,748,096 params.
  - `Bielik-Minitron-7B-v3.0-Instruct` (2026-01): a pruned and distilled 11B v3, 7,477,727,232 params, with the 32128-token vocab.
  - `Bielik-PL-Minitron-7B-v3.0-Instruct`.
- **What `auto` gating means:** on the CUDA machine you need an HF token and must click "accept" once on each gated repo page. The form has the checkbox "I agree to be contacted for feedback about Bielik models", and approval is automatic.

## 2. Tokenizers, special tokens, chat template

### Special tokens (checked with `AutoTokenizer` in transformers 5.17)

| | 1.5B / 4.5B v3 (APT4) | 11B v2.3 | 11B v2.6 / v3.0 |
|---|---|---|---|
| Loaded class | `LlamaTokenizer` | `TokenizersBackend` (see quirk 3) | `LlamaTokenizer` |
| `len(tok)` | 32000 | 32128 | 32128 |
| bos | `<s>`=1 | `<s>`=1 | `<s>`=1 |
| eos | **`<|im_end|>`=4** | **`<|im_end|>`=32001** | **`<|im_end|>`=32001** |
| pad | `</s>`=2 (set, not missing) | `</s>`=2 | `</s>`=2 |
| unk | `<unk>`=0 | 0 | 0 |
| `<|im_start|>` | 3 | 32000 | 32000 |
| Extra added tokens | `<tool_call>`=5, `</tool_call>`=6 | `<|function_list|>`… 32002–4, `<|control_6..128|>` | 32002–4 function tokens, `<tool_call>`=32005, `</tool_call>`=32006, `<think>`=32007, `</think>`=32008, then `<|control_10..|>` |
| `padding_side` default | left | right | left |
| `add_bos_token` (tokenizer_config) | True | False | False |
| Post-processor adds `<s>` on `tok(text)` | **yes** | **yes** | no |

### Chat template

All five models return the same ChatML template in the API's `tokenizer_config.chat_template`. Verbatim from `speakleash/Bielik-11B-v2.3-Instruct/tokenizer_config.json`:

```jinja
{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

- **System role:** supported. The template passes any role straight through, and every card's example starts with `{"role":"system",...}`. Rendered output: `<s><|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`.
- **v3 (4.5B, 1.5B, 11B v3.0):** same template string. The 11B v3.0 FP8 copy ships it as a 209-byte `chat_template.jinja`.
- **v2.6 variant:** the original gated repo uses the simple template (per API). The ungated `Bielik-11B-v2.6-Instruct-FP8-Dynamic/chat_template.jinja` is a 3976-byte extended version:
  - it supports `tools` via a `<tool_call>{"name":..,"arguments":..}</tool_call>` block in the system turn, and a `tool` role;
  - it supports `enable_thinking`: that appends a Polish "reason step by step inside `<think>…</think>`" instruction to the system message, and `enable_thinking=False` pre-fills `<think>\n\n</think>\n\n`;
  - it raises on roles other than system, user, assistant, tool, tool_results and function.

  Without tools or thinking it renders the same string as the simple template (verified).
- **Spaces in the cards' examples:** the card text `<|im_start|> user` and `<|im_end|> \n` is just how decoding displays it, not a different format. The normalizer prepends `▁` to each segment between special tokens, so tokens are `['<|im_start|>','▁user','<0x0A>',…,'<|im_end|>','▁','<0x0A>']`. Training and inference both go through the same path, so this is consistent.
- **Token efficiency:** APT4 (v3 small) is about 2x more efficient on Polish text. The report's Polish constitution preamble takes 375 tokens with APT4 vs 747 with the Mistral v0.1 tokenizer. My test sentence took 18 vs 26 tokens.

## 3. Quirks (all reproduced in the venv)

1. **Double BOS with pre-rendered text (4.5B, 1.5B, 11B-v2.3).** `tok(tok.apply_chat_template(..., tokenize=False))` gives `[1, 1, 3, …]`. With TRL's `"text"` dataset field it gives `['<s>','<s>','<|im_start|>'] … ['<|im_end|>','▁','<0x0A>','<|im_end|>']`: double BOS plus an extra `<|im_end|>`, because TRL appends eos when the text does not end with it. 11B v2.6/v3.0 do not add BOS but still get the extra eos.
   - Safe paths: `apply_chat_template(tokenize=True)` (single BOS), or `tok(text, add_special_tokens=False)`.
2. **The conversational prompt-completion format in TRL 1.13 is clean on all three tokenizer families.**
   - Dataset shape: `{"prompt":[system,user msgs], "completion":[{"role":"assistant","content":"B"}]}`.
   - Result: `completion_only_loss` switches on automatically, one BOS, and the trained tokens are exactly `['B','<|im_end|>','▁','<0x0A>']`. There is no prefix-mismatch warning. TRL 1.13 writes `labels` directly; there is no `completion_mask` column.
3. **`assistant_only_loss=True` with the stock template fails.** The error is `ValueError: The chat template is not training-compatible (missing prefix-preservation or {% generation %} markers) and patching is not supported for this template`; TRL only has patches for Qwen, Llama-3, Gemma and a few others. This drop-in template renders byte-identical output to the stock one (verified for both tokenizer families) and trains only on `['B','<|im_end|>','C','<|im_end|>']`:
   ```python
   BIELIK_TRAIN_TEMPLATE = (
   "{{ bos_token }}{% for message in messages %}"
   "{% if message['role'] == 'assistant' %}"
   "{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}{{ '\n' }}"
   "{% else %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endif %}"
   "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
   tokenizer.chat_template = BIELIK_TRAIN_TEMPLATE   # then SFTConfig(assistant_only_loss=True)
   ```
4. **v2.3 tokenizer loading.** Its `model_type` is `mistral`, so in transformers 5.17 `AutoTokenizer` maps it to `MistralCommonBackend` when `mistral-common` is installed (vLLM pulls that in), else to `TokenizersBackend`. It only switches to MistralCommonBackend if a `tekken.json` exists, and v2.3 has none, so it falls back to `TokenizersBackend` with `padding_side='right'`. Every other model loads as `LlamaTokenizer` with `padding_side='left'`.
   - For batched `generate`, set `tokenizer.padding_side="left"` explicitly.
   - Passing `mistral_format=False` also forces the standard HF tokenizer.
5. **Pad and eos.** The pad token is present (`</s>`), so there is no need to set pad = eos, and pad ≠ eos, so the collator will not mask eos. `generation_config.eos_token_id` is a list `[<|im_end|>, </s>]`. When given a model whose config differs, SFTTrainer logs "…aligned accordingly… Updated tokens: {'eos_token_id': …}" and resets it to the single tokenizer eos (seen on the tiny test models).
6. **4.5B and 1.5B have biases** in q/k/v/o and gate/up/down. PEFT LoRA wraps them fine; the base biases stay frozen. Some kernels or quantizers that assume bias-free Llama can break (UNVERIFIED which ones). vLLM runs them: SpeakLeash's FP8 card documents vLLM and SGLang for 4.5B.
7. **v2.3 is F16 on disk.** Loading with `dtype=torch.bfloat16` is fine; the card itself does this.
8. **Small KV caches** (8.7 GiB):

   | Model | kv_heads | KV cache per token (bf16) |
   |---|---|---|
   | 4.5B | 2 | 60 KiB |
   | 1.5B | 2 | 32 KiB |
   | 11B | 8 | 200 KiB |

   This makes 4.5B very cheap for high-concurrency vLLM evaluation.

## 4. LoRA target modules

The linear layers are named the same in all five models (`LlamaAttention`/`LlamaMLP`, or the Mistral equivalents for v2.3): `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, plus `lm_head`.

- Checked with PEFT 0.21: `target_modules="all-linear"` resolves to exactly those 7 per layer and excludes `lm_head`.
- PEFT's default for llama/mistral is `['q_proj','v_proj']` only.

| Trainable LoRA params, r=16 on all 7 | Count |
|---|---|
| 1.5B | 21,102,592 |
| 4.5B | 49,889,280 |
| 11B | 65,536,000 |

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
           target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
```

For reference, SpeakLeash's own SFT for v3 small was full fine-tuning: AdamW β=(0.9, 0.95), weight decay 0.05, LR 7e-6 cosine down to 6e-7, 50 warmup steps, global batch 128, packing, 8192 context, 1.2 epochs, prompt tokens masked.

## 5. VRAM

These are my estimates, not measured on CUDA. The one hard check: my NF4 estimate for 11B (5.73 GiB) exactly matches the real `Bielik-11B-v2.6-Instruct-bnb-4bit` repo (6,153,567,316 bytes = 5.73 GiB; nf4, double quant, uint8 storage). bitsandbytes is not installed on this Mac.

| | 4.5B | 11B |
|---|---|---|
| bf16 weights | 8.86 GiB | 20.80 GiB |
| NF4 weights (embeddings and lm_head kept bf16) | ≈2.5 GiB | 5.73 GiB (checked) |
| LoRA r16 all-linear: fp32 weights, grads, Adam | ≈0.74 GiB | ≈0.98 GiB |
| Activations per sequence, gradient checkpointing + SDPA, incl. fp32 logits | ≈1.2 GiB @2048, 2.3 @4096 | ≈1.6 GiB @2048, 3.2 @4096 |
| **bf16 inference, HF, batch 1** | ≈10–11 GiB | ≈22–24 GiB (over 24 GiB with long context) |
| **bf16 LoRA, bs 4 @2048** | ≈15–17 GiB: fits L4/A10 24GB | ≈29–32 GiB: A100-40 OK (tight), A100-80 comfortable, **not 24GB** |
| **QLoRA, bs 4 @2048** | ≈8–9 GiB | ≈13–15 GiB: fits L4/A10 24GB |

Add about 1–2 GiB for the CUDA context and allocator.

**Rough throughput, UNVERIFIED** (LoRA with checkpointing ≈ 5N FLOPs per token; 40% MFU on A100, 35% on L4):

| | A100 | L4 bf16 LoRA | L4 QLoRA (≈30–40% dequant overhead) |
|---|---|---|---|
| 4.5B | ≈5k tok/s | ≈1.7k tok/s | slower still |
| 11B | ≈2.2k tok/s | does not fit | ≈0.5k tok/s |

For example, an epoch of 10M tokens is about 0.5 h (4.5B) vs 1.3 h (11B) on an A100, and about 1.6 h (4.5B) vs about 5–6 h (11B QLoRA) on an L4.

Other inference notes:
- 11B bf16 in vLLM on 24GB leaves under 2 GiB for KV cache.
- The L4 (Ada) can use SpeakLeash's `*-FP8-Dynamic` checkpoints; the A10 (Ampere) has no FP8 compute.
- There is also a ready AWQ build: `Bielik-11B-v3.0-Instruct-awq`, ungated, about 6.2 GB.

## 6. Published exam and benchmark results, and how they prompt

**No SpeakLeash benchmark targets the matura directly.** Their eval harness fork (`github.com/speakleash/lm-evaluation-harness`, branches `polish3`/`polish4`) has no matura or LLMzSzŁ task. The closest is `polish_pes` (medical board exams).

**Open PL LLM Leaderboard** (SpeakLeash, lm-eval-harness; tasks polemo2, klej-ner, 8tags, belebele, dyk, ppc, psc, cbd, polqa, poquad, eq-bench):
- Each task has two variants:
  - `*_mc`: `output_type: multiple_choice` (log-likelihood over `["A","B","C","D"]`)
  - `*_regex`: `generate_until`, greedy, `max_gen_toks: 50`, stops at `"."` or `","`, then extracts the answer with regex `(\b[ABCD]\b)`
- Tasks run 0-shot and 5-shot; the average is normalized against baselines.
- Example prompt (`polish_belebele`): `Fragment: "…"\nPytanie: "…"\nMożliwe odpowiedzi:\nA - …\nB - …\nC - …\nD - …\nPrawidłowa odpowiedź:`
- PES generative prompt: `Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród wszystkich odpowiedzi wybierz tylko jedną. Odpowiedz tylko i wyłącznie jedną literą.\n{{question_final}}\nPrawidłowa odpowiedź:`
- Instruct 5-shot averages: **11B-v3.0 65.93**, 11B-v2.3 65.71, 11B-v2.6 64.26, **4.5B-v3.0 56.13**, 1.5B-v3.0 41.36.

**LLMzSzŁ** (AMU, arXiv 2501.02266; dataset `amu-cai/llmzszl-dataset`, ungated):
- `llmzszl-test.jsonl`: 18,821 rows with fields `question, answers[4], correct_answer_index, year, type, name`.
- Matura rows (`type == "Egzaminy Maturalne"`): only 377 closed questions (Matematyka 220, Fizyka 136, Biologia 21). There is **no Polish-language matura**. Most of the dataset is vocational exams.
- Method: lm-eval-harness, MMLU-style **log-likelihood**, 0-shot. Prompt: `Przykładowe pytanie egzaminacyjne, test jednokrotnego wyboru\n{{question.strip()}}\nA. …\nB. …\nC. …\nD. …\nPrawidłowa odpowiedź:`

| Model | LLMzSzŁ overall | Matura subset (unweighted mean of 2002–2023, from `all_types_years.json`) |
|---|---|---|
| 11B-v2.6 | **59.23** | 50.86 |
| 11B-v2.5 | 58.93 | 50.67 |
| 11B-v2.3 | 57.40 | 46.55 |
| 4.5B-v3.0 | **54.76** | 48.91 |
| 1.5B-v3.0 | 46.81 | 39.31 |
| Mistral-Large-2407 (for reference) | 67.17 | 63.41 |

11B-v3.0 is not on this leaderboard (UNVERIFIED).

**Other exam-style results** (11B v3 report arXiv 2601.11579 and the v3 small report arXiv 2505.02550):

| Benchmark | 11B-v3.0 | 11B-v2.6 | 11B-v2.3 | 4.5B-v3.0 |
|---|---|---|---|---|
| INCLUDE-base-44, Polish (MC exams) | **69.0** | 59.3 | — | 48.7 (base model; instruct not reported) |
| Polish Medical / PES (5-shot) | 50.21 | 44.88 | 43.26 | 43.55 |
| PLCC (cultural; MC + open) | 71.83 | 65.50 | 62.17 | 41.47 (42.33 in the small-model report) |

A separate paper on the Polish history matura, with open questions and essays (arXiv 2608.12343), evaluates 8 models. The abstract does not name Bielik; I did not check the full text.

## 7. Recommendation: 4.5B vs 11B for a 46h hackathon

**If 11B is allowed and you get an A100 40/80GB: fine-tune 11B.** Choose `Bielik-11B-v3.0-Instruct` if any version is allowed, otherwise `Bielik-11B-v2.6-Instruct` over v2.3. Use bf16 LoRA on the A100-80; on the A100-40, use bf16 LoRA with bs 2–4 at 2048, or QLoRA.
- **Why:** score = base points + gain, so what counts is the final absolute score, and the base starts well ahead:
  - INCLUDE-PL (exam MC): 69.0 for 11B-v3 vs 48.7 for 4.5B base.
  - LLMzSzŁ: 59.2 for v2.6 vs 54.8 for 4.5B.
  - Open PL: 65.9 vs 56.1.
  - PLCC: 71.8 vs 42.
- **Knowledge gap:** a few thousand LoRA examples rarely close a 5–20 point knowledge gap. Most gains on exam MC come from answer format and extraction and short CoT, and both sizes get those.
- **Reasoning training:** v2.6 and v3 also went through GRPO on verifiable math/STEM, which helps generative or CoT matura math.
- **v2.6 vs v2.3:** v2.6 beats v2.3 on the matura subset (50.9 vs 46.6), on LLMzSzŁ, PLCC and medical; it is lower on Open PL (64.3 vs 65.7).

**If you only have L4/A10 24GB: 4.5B is the pragmatic choice, or run 11B QLoRA for a single final run.**
- 11B needs QLoRA there, at about 0.5k tok/s on an L4, and bf16 eval in vLLM barely fits.
- In 46h that means about 3x fewer training/eval iterations. Iteration count (data-mix and prompt-format sweeps) usually matters more than model size for "gain".
- 4.5B trains in bf16 LoRA on 24GB, uses about half as many tokens per Polish text, and its 60 KiB/token KV cache makes vLLM eval fast.
- 4.5B is also relatively strong on the matura subset (48.9, above 11B-v2.3's 46.6), so its disadvantage is smallest on exactly this kind of question.

**If gain is weighted more heavily than base points:** 4.5B has more headroom, and small models usually show bigger relative gains from format and SFT. That gain has to exceed roughly 5–10 points to beat an 11B's absolute score.

**Suggested plan:**
1. First measure base scores on your actual eval set for both models (vLLM with the FP8 or AWQ builds is fine for triage). Use both log-likelihood A–D and generative + regex, matching the harness above, since instruct models can rank differently under each.
2. Then commit to one model for training by hour ~6.
3. Train on the conversational prompt/completion format (quirk 2), or on `messages` with the generation-marker template (quirk 3).
4. Don't feed pre-rendered `"text"` (quirk 1).

## Files

Scratch scripts are in `<research scratch>/` (local research workspace, not committed):
- `inspect_arch.py` (meta-device architecture and parameter counts)
- `tok_test.py` (tokenizer and template)
- `trl_test.py`, `trl_test2.py`, `trl_test3.py` (TRL 1.13 masking, BOS and eos)
- `peft_test.py` (target modules)
- `vram.py` (memory estimates)
- downloaded configs under per-model subfolders, `llmzszl-test.jsonl`, `all_types.json`, `bielik11bv3.txt`

Sources: [HF API speakleash models](https://huggingface.co/api/models?author=speakleash), [Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct), [Bielik-4.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct), [Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct), [Bielik v3 Small report](https://arxiv.org/abs/2505.02550), [Bielik 11B v3 report](https://arxiv.org/pdf/2601.11579), [Bielik 11B v2 report](https://arxiv.org/pdf/2505.02410), [LLMzSzŁ paper](https://arxiv.org/abs/2501.02266), [LLMzSzŁ dataset](https://huggingface.co/datasets/amu-cai/llmzszl-dataset), [LLMzSzŁ leaderboard](https://huggingface.co/spaces/amu-cai/LLMZSZL_Leaderboard), [speakleash lm-evaluation-harness](https://github.com/speakleash/lm-evaluation-harness), [History matura benchmark](https://arxiv.org/abs/2608.12343), [Tokenizer optimization in Bielik v3 7B/11B](https://arxiv.org/pdf/2604.10799)