I checked everything below in two ways: by reading the installed source, and by running scripts on MPS with `HuggingFaceTB/SmolLM2-135M-Instruct`. The only exceptions are marked **(source only)** or **UNVERIFIED**.

The most important result: in TRL 1.13, a conversational prompt-completion dataset computes loss **only on the completion**. §2.4 has the script and its output. Two things will break code written from memory. The mac model load crashes when the requested dtype differs from the checkpoint's (G1). And `apply_chat_template` now returns a dict of tensors, not a tensor (G3).

All scripts and logs are in `<research scratch>/` (local research workspace, not committed) (files `t1_…py` to `t11_…py`, `t2.log`). Nothing was written inside the project folder.

Installed stack: torch 2.14.0, transformers 5.17.0, peft 0.21.0, trl 1.13.0, accelerate 1.15.0, datasets 5.0.1, sentence-transformers 6.0.1. bitsandbytes is not installed.

---

## 1. transformers 5.17

### 1.1 `from_pretrained`: dtype, device_map, attention
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,          # new name. torch_dtype= still works but logs "`torch_dtype` is deprecated! Use `dtype` instead!"
    device_map={"": 0},            # CUDA single GPU; "mps" or "auto" on mac
    attn_implementation="sdpa",    # default when available
)
```
- **The `dtype` default is now `"auto"`.** It uses `config.dtype`, falling back to the legacy `torch_dtype` key. Verified: SmolLM2 loads in bf16 when no dtype is given. **Bielik-11B-v2.3's config.json says `"torch_dtype": "float16"`, so without `dtype=` it loads in fp16.** Always pass `dtype=torch.bfloat16`.
- `device_map`:
  - `None` (default) loads to CPU. Verified: `m.device == cpu`.
  - `"mps"` and `"auto"` on mac both put the whole model on `mps:0`. Verified.
  - Source: the string `"cuda"` becomes `cuda:{LOCAL_RANK}`. `"auto"`, `"balanced"`, `"balanced_low_0"` and `"sequential"` go through accelerate.
  - Source: the bnb 4-bit quantizer turns `device_map=None` into `{"": torch.cuda.current_device()}`.
- `attn_implementation`:
  - Registered keys: `sdpa`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`, `flex_attention`, `paged|eager`, `paged|sdpa`, `paged|flash_attention_{2,3,4}`. `"eager"` is always allowed.
  - Hub kernels in the form `"org/repo[@rev][:kernel]"` are also accepted.
  - Verified: `eager` and `sdpa` work. `flash_attention_2` raises `ImportError` when the package isn't installed.

### 1.2 `BitsAndBytesConfig` for NF4 double-quant QLoRA (source only; bnb is missing on mac)
```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # default "fp4"
    bnb_4bit_use_double_quant=True,       # default False
    bnb_4bit_compute_dtype=torch.bfloat16,# default None -> float32 (slow). Accepts str or torch.dtype
    # bnb_4bit_quant_storage=torch.uint8  # default uint8; set bf16 only for FSDP
    # llm_int8_skip_modules=[...]         # modules kept un-quantized
)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
```
- Full field list: `load_in_8bit`, `load_in_4bit`, `llm_int8_threshold=6.0`, `llm_int8_skip_modules`, `llm_int8_enable_fp32_cpu_offload`, `llm_int8_has_fp16_weight`, `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_quant_storage`.
- Minimum bitsandbytes version is 0.46.1.
- Verified on mac: building the config works. `from_pretrained` then raises `ImportError: ... requires bitsandbytes>=0.46.1`.

### 1.3 `tokenizer.apply_chat_template`
Signature: `(conversation, tools=None, documents=None, chat_template=None, add_generation_prompt=False, continue_final_message=False, tokenize=True, padding=False, truncation=False, max_length=None, return_tensors=None, return_dict=True, return_assistant_tokens_mask=False, tokenizer_kwargs=None, **kwargs)`

**`return_dict` now defaults to True**, so the tokenized output is a `BatchEncoding`, not a list or tensor. Verified:
```python
s   = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)          # str
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")     # BatchEncoding{input_ids, attention_mask}
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)       # list[int]
pre = tok.apply_chat_template(msgs + [{"role":"assistant","content":"Odpowiedź:"}], tokenize=False, continue_final_message=True)  # ends '...assistant\nOdpowiedź:'
# continue_final_message=True together with add_generation_prompt=True -> ValueError
```

### 1.4 Batched greedy generation with left padding (verified: batched output == one-by-one output)
```python
tok.padding_side = "left"                     # SmolLM2 and Bielik-11B-v2.3 both load as "right" (transformers 5.17). Always set it.
if tok.pad_token is None: tok.pad_token = tok.eos_token
# Option A: batched chat template (verified)
enc = tok.apply_chat_template(convs, add_generation_prompt=True, padding=True, return_tensors="pt").to(model.device)
# Option B: render then tokenize. add_special_tokens=False is REQUIRED for Bielik (see G5)
texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=64, do_sample=False)
answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
```

### 1.5 Greedy decoding without "invalid generation flags" warnings (verified)
The warning comes through the transformers **logger**, not Python `warnings`: `The following generation flags are not valid and may be ignored: [...]`. For the test, the model's `generation_config` was given `do_sample=True, temperature=0.6, top_p=0.9, top_k=20`.

| Call | Warns? |
|---|---|
| `generate(**enc, max_new_tokens=N, do_sample=False)` | no |
| same, plus `temperature=0.6` kwarg | yes `['temperature']` |
| `generate(**enc, generation_config=GenerationConfig(do_sample=False, ...))` | yes `['temperature','top_p','top_k']` (None fields are refilled from `model.generation_config`) |
| deepcopy of `model.generation_config` with those fields set to `None` | yes (refilled) |
| only `model.generation_config.do_sample=False` | yes |
| **`model.generation_config.update(do_sample=False, temperature=None, top_p=None, top_k=None)`**, then any call style | **no** |

Also: `save_pretrained` validates the generation config strictly. Saving with `do_sample=False, temperature=0.6` raises `ValueError: GenerationConfig is invalid` (verified). The sanitize line in the last row fixes this too.

### 1.6 TrainingArguments in v5 (verified by constructing each)
- **Removed (TypeError):**
  - `evaluation_strategy` → use `eval_strategy`
  - `warmup_ratio` → use `warmup_steps=0.03` (a float below 1 is read as a ratio; verified `get_warmup_steps(1000)=100` for 0.1)
  - `overwrite_output_dir`, `logging_dir`
  - `group_by_length` → use `train_sampling_strategy="group_by_length"`
  - `save_safetensors`, `use_mps_device`
  - `no_cuda` → use `use_cpu=True`
  - `push_to_hub_token`, `include_tokens_per_second`, `dispatch_batches`
- **Defaults:**
  - `report_to="none"` (becomes `[]`), `output_dir=None` (becomes `"trainer_output"`)
  - `optim="adamw_torch_fused"` (verified fused=True on MPS)
  - `eval_strategy="no"`, `save_strategy="steps"`, `save_steps=500`, `logging_steps=500`
  - `bf16=False`, `gradient_checkpointing=False`
- **bf16 check:** `bf16=True` requires `is_torch_bf16_gpu_available()`. On MPS that means macOS 14 or newer. Verified: works on this mac and `accelerator.mixed_precision == "bf16"`. `fp16=True` also runs on MPS (verified).
- **`gradient_checkpointing_kwargs`:** Trainer removes `every_n_layers` and `offload` and passes the rest to `torch.utils.checkpoint`. If it is None, `model.gradient_checkpointing_enable` uses `{"use_reentrant": False}` (source). Verified `use_reentrant` True and False both train on MPS.

### 1.7 Callbacks and eval metric names (verified)
```python
class CB(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs): ...
    def on_epoch_end(self, args, state, control, **kwargs): ...
    def on_log(self, args, state, control, logs=None, **kwargs): ...
# kwargs = model, processing_class (NOT tokenizer), optimizer, lr_scheduler, train_dataloader, eval_dataloader
```
- **With `eval_dataset={"matura": ds1, "other": ds2}`:** `evaluate()` recurses with prefix `eval_<name>`. Keys are `eval_matura_loss`, `eval_matura_runtime`, `eval_matura_samples_per_second`, `eval_matura_steps_per_second`, and the same for `other`.
- **`on_evaluate` fires once per dataset**, not once per eval round.
- **TRL's extra eval metrics are not per-dataset.** `eval_entropy`, `eval_mean_token_accuracy` and `eval_num_tokens` have no dataset name, and in the merged dict the last dataset's value wins.
- `metric_for_best_model="matura_loss"` is auto-prefixed to `eval_matura_loss`. Verified: `load_best_model_at_end` picks `checkpoint-4`.
- With `eval_strategy="epoch"`, `on_epoch_end` fires **before** that epoch's evaluation. With `"steps"`, eval runs inside the step loop.

---

## 2. TRL 1.13: SFTConfig and SFTTrainer

### 2.1 SFTConfig fields (from `trl/trainer/sft_config.py`; values checked at runtime)
| Field | Default / note |
|---|---|
| `max_length` | `1024`. **`max_seq_length` no longer exists (TypeError).** Truncation is `truncation_mode="keep_start"`, so the *end* is cut off. |
| `packing` / `packing_strategy` / `padding_free` / `eval_packing` | `False` / `"bfd"` / `False` / `None`. The bfd strategy forces padding_free and warns unless you use a flash-attn variant. |
| `completion_only_loss` | `None`: True for prompt-completion data, False for LM data |
| `assistant_only_loss` | `False`. Conversational data only. Needs `{% generation %}` markers (§2.5). |
| `dataset_text_field` | `"text"` |
| `dataset_kwargs` | `None`. The only key is `skip_prepare_dataset`. |
| `dataset_num_proc`, `shuffle_dataset`, `pad_to_multiple_of` | `None`, `False`, `None` |
| `eos_token` | `None` (uses tokenizer's). Only appended for **non-conversational** text. |
| `pad_token` | **Deprecated** (FutureWarning). Set `tok.pad_token` instead. |
| `model_init_kwargs` | Only used when `model` is a string. **dtype defaults to float32 there** (verified). |
| `chat_template_path`, `trust_remote_code`, `activation_offloading` | |
| `loss_type` | `None`, which becomes **`"chunked_nll"`** (or `"nll"` with liger). Also `"dft"`. |
| Overridden TrainingArguments defaults | `learning_rate=2e-5`, `logging_steps=10`, **`gradient_checkpointing=True`**, **`bf16=True` unless fp16** |

### 2.2 SFTTrainer `__init__`
`SFTTrainer(model: str|PreTrainedModel|PeftModel, args=None, data_collator=None, train_dataset=None, eval_dataset: Dataset|IterableDataset|DatasetDict|dict[str, Dataset]|None=None, processing_class=None, compute_loss_func=None, compute_metrics=None, callbacks=None, optimizers=(None,None), optimizer_cls_and_kwargs=None, preprocess_logits_for_metrics=None, quantization_config=None, peft_config=None, formatting_func=None)`

- There is **no `tokenizer=` argument**. The same is true for `Trainer`.
- A dict `eval_dataset` is supported: each entry is prepared under its own key (verified).
- Passing a `PeftModel` together with `peft_config` raises ValueError.
- SFTTrainer **does not call** `prepare_model_for_kbit_training`. For a 4-bit model it casts trainable (adapter) params to bf16 (source).
- `loss_type="chunked_nll"` raises ValueError if `lm_head` is a LoRA target. `"all-linear"` excludes `lm_head` (verified target list: q,k,v,o,gate,up,down).
- **(source only)** `chunked_nll` returns `logits=None` when labels are present. If you need `compute_metrics` or `preprocess_logits_for_metrics` on logits, use `loss_type="nll"`.

### 2.3 Supported dataset formats (source: `data_utils.is_conversational`, `_prepare_dataset`)
- Standard LM: `{"text": str}`
- Standard prompt-completion: `{"prompt": str, "completion": str}`
- Conversational LM: `{"messages": [{"role","content"},...]}`. Loss is on the **whole sequence** unless `assistant_only_loss` is set (verified: `n_loss == n_tokens == 44`).
- **Conversational prompt-completion:** `{"prompt": [{"role":"system",...},{"role":"user",...}], "completion": [{"role":"assistant","content":...}]}`. The prompt is tokenized with `add_generation_prompt=True`, prompt+completion without it, and `completion_mask` covers the difference.
- Also accepted:
  - ShareGPT `conversations` / `from` / `value`, auto-converted to ChatML
  - Pre-tokenized `input_ids` (+ `labels` / `completion_mask` / `assistant_masks`)
  - Optional per-row `chat_template_kwargs` and `tools`

### 2.4 Proof: loss only on completion tokens (`t2_sft_completion_only.py`, run verbatim)
```python
"""Proof: TRL 1.13 SFTTrainer + conversational prompt-completion dataset => loss only on completion tokens."""
import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

M = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_sft")
SYS = "Jesteś ekspertem od matury. Odpowiadaj zwięźle."
qa = [
    ("Ile wynosi 2+2?", "Odpowiedź: 4"),
    ("Podaj stolicę Polski.", "Odpowiedź: Warszawa"),
    ("Kto napisał 'Pan Tadeusz'?", "Odpowiedź: Adam Mickiewicz"),
    ("Rozwiąż równanie x+3=5.", "x = 2, bo 5-3=2. Odpowiedź: 2"),
    ("Ile boków ma sześciokąt?", "Odpowiedź: 6"),
    ("Jaki jest wzór na pole koła?", "Pole koła to πr². Odpowiedź: πr²"),
]
def row(q, a):
    return {"prompt": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            "completion": [{"role": "assistant", "content": a}]}
train_ds = Dataset.from_list([row(q, a) for q, a in qa])
eval_ds = {"matura": Dataset.from_list([row(q, a) for q, a in qa[:2]]),
           "other": Dataset.from_list([row(q, a) for q, a in qa[2:4]])}

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32)  # trainer autocasts with bf16=True

class CB(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"[CB.on_evaluate] step={state.global_step} keys={sorted(metrics)}")
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[CB.on_epoch_end] epoch={state.epoch} kwargs={sorted(kwargs)}")
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"[CB.on_log] {logs}")

args = SFTConfig(
    output_dir=OUT, max_steps=3, per_device_train_batch_size=3, per_device_eval_batch_size=2,
    learning_rate=2e-4, logging_steps=1, eval_strategy="steps", eval_steps=3, save_strategy="no",
    report_to="none", max_length=512, seed=0,
    # left at defaults on purpose: completion_only_loss=None, bf16=None->True, gradient_checkpointing=True, loss_type=None->chunked_nll
)
print("SFTConfig resolved: completion_only_loss=", args.completion_only_loss, "bf16=", args.bf16,
      "gradient_checkpointing=", args.gradient_checkpointing, "loss_type=", args.loss_type,
      "report_to=", args.report_to, "device=", args.device)

peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                     processing_class=tok, peft_config=peft_cfg, callbacks=[CB()])
print("trainer.completion_only_loss =", trainer.completion_only_loss)
print("prepared train columns:", trainer.train_dataset.column_names)
print("prepared eval keys:", list(trainer.eval_dataset.keys()), trainer.eval_dataset["matura"].column_names)

# ---- CHECK 1: per-example labels vs independently computed prompt boundary ----
all_ok = True
for i, ex in enumerate(trainer.train_dataset):
    raw = train_ds[i]
    prompt_ids = tok.apply_chat_template(raw["prompt"], add_generation_prompt=True, return_dict=False)
    full_ids = tok.apply_chat_template(raw["prompt"] + raw["completion"], return_dict=False)
    ids, labels = ex["input_ids"], ex["labels"]
    n_prompt = len(prompt_ids)
    ok = (ids == full_ids
          and all(l == -100 for l in labels[:n_prompt])
          and labels[n_prompt:] == ids[n_prompt:])
    all_ok &= ok
    if i < 2:
        print(f"\nexample {i}: len={len(ids)} prompt_len={n_prompt} n_loss_tokens={sum(l != -100 for l in labels)}")
        print("  MASKED (prompt) tail:", repr(tok.decode(ids[:n_prompt])[-45:]))
        print("  LOSS tokens decoded :", repr(tok.decode([t for t, l in zip(ids, labels) if l != -100])))
print("\nCHECK1 per-example labels==-100 on prompt, ==input_ids on completion:", all_ok)

# ---- CHECK 2: collated batch from the real train dataloader ----
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print("batch keys:", sorted(batch.keys()), "shapes:", {k: tuple(v.shape) for k, v in batch.items()})
lab, ids, am = batch["labels"], batch["input_ids"], batch["attention_mask"]
print("padding positions all -100 in labels:", bool((lab[am == 0] == -100).all()))
for r in range(lab.shape[0]):
    print(f"  row {r}: loss-token text = {tok.decode(ids[r][lab[r] != -100])!r}")

# ---- CHECK 3: the trainer's loss == manual CE over completion tokens only (and != full-seq CE) ----
dev = trainer.args.device
b = {k: v.to(dev) for k, v in batch.items()}
trainer.model.eval()
with torch.no_grad():
    trainer_loss = trainer.compute_loss(trainer.model, dict(b)).item()
    logits = trainer.model.get_base_model().__class__.forward(trainer.model.get_base_model(), input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
    sl, lg = b["labels"][:, 1:], logits[:, :-1]
    manual_completion = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), sl.reshape(-1), ignore_index=-100).item()
    full = b["input_ids"][:, 1:].masked_fill(b["attention_mask"][:, 1:] == 0, -100)
    manual_full = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), full.reshape(-1), ignore_index=-100).item()
print(f"CHECK3 trainer.compute_loss={trainer_loss:.5f} manual_completion_only_CE={manual_completion:.5f} manual_full_seq_CE={manual_full:.5f}")
trainer.model.train()

# ---- train 3 steps ----
res = trainer.train()
print("train result:", res.metrics)
trainer.save_model(os.path.join(OUT, "adapter"))
print("adapter files:", sorted(os.listdir(os.path.join(OUT, "adapter"))))
print("adapter_config:", {k: v for k, v in json.load(open(os.path.join(OUT, "adapter", "adapter_config.json"))).items() if k in ("r", "lora_alpha", "target_modules", "task_type", "base_model_name_or_path", "use_rslora", "use_dora")})
print("final evaluate():", trainer.evaluate())
```
**Output (trimmed; full log in `t2.log`):**
```
SFTConfig resolved: completion_only_loss= None bf16= True gradient_checkpointing= True loss_type= chunked_nll report_to= [] device= mps
trainer.completion_only_loss = True
prepared train columns: ['prompt', 'completion', 'input_ids', 'labels']          # completion_mask folded into labels, then dropped
prepared eval keys: ['matura', 'other'] ['prompt', 'completion', 'input_ids', 'labels']
example 0: len=57 prompt_len=46 n_loss_tokens=11
  MASKED (prompt) tail: ' wynosi 2+2?<|im_end|>\n<|im_start|>assistant\n'
  LOSS tokens decoded : 'Odpowiedź: 4<|im_end|>\n'
CHECK1 per-example labels==-100 on prompt, ==input_ids on completion: True
batch keys: ['attention_mask', 'input_ids', 'labels'] shapes: {... (3, 77) ...}
padding positions all -100 in labels: True
  row 0: loss-token text = 'Odpowiedź: Adam Mickiewicz<|im_end|>\n'
CHECK3 trainer.compute_loss=3.66972 manual_completion_only_CE=3.66972 manual_full_seq_CE=4.15118
[CB.on_log] {'loss': 3.685, 'grad_norm': 2.14, 'learning_rate': 0.0002, 'entropy': 2.35, 'num_tokens': 210.0, 'mean_token_accuracy': 0.466, 'epoch': 0.5}
[CB.on_evaluate] step=3 keys=['epoch','eval_entropy','eval_matura_loss','eval_matura_runtime',...]
[CB.on_evaluate] step=3 keys=[...,'eval_other_loss',...]
adapter files: ['README.md','adapter_config.json','adapter_model.safetensors','chat_template.jinja','tokenizer.json','tokenizer_config.json','training_args.bin']
adapter_config: {'target_modules': ['k_proj','gate_proj','o_proj','up_proj','q_proj','down_proj','v_proj'], 'r': 8, 'lora_alpha': 16, ...}
```
What this shows:
- Labels are -100 on every prompt token, including `<|im_start|>assistant\n`.
- The completion **plus `<|im_end|>\n`** is trained, so the model learns to stop.
- The trainer's loss exactly equals a manual completion-only cross-entropy, and differs from full-sequence cross-entropy.
- Eval datasets are masked the same way.

### 2.5 `assistant_only_loss` (verified)
- It needs `{% generation %}…{% endgeneration %}` markers in the chat template.
- If they are missing, TRL tries a built-in training template: Qwen, Llama3, Gemma, Phi3, DeepSeek, GLM, GPT-OSS, Cohere, LFM2, Nemotron and a few others. For any other template it **raises `ValueError: The chat template is not training-compatible ...`**.
- Verified for SmolLM2's template and for **Bielik-11B-v2.3's template** (`get_training_chat_template` raises).
- A custom ChatML template with generation markers works: loss text = `'4<|im_end|>6<|im_end|>'`.
- **For Bielik, use conversational prompt-completion data** (completion-only loss by default) rather than `messages` + `assistant_only_loss`.

### 2.6 Bielik tokenizer checks (Bielik-11B-v2.3 is ungated; tokenizer files only)
- Class `LlamaTokenizer`; bos `<s>`(1), **eos `<|im_end|>`(32001)**, pad `</s>`(2). Re-checked with transformers 5.17: it loads as
  `TokenizersBackend` with `padding_side="right"` (not "left"), so generation must set left padding explicitly (G6).
- Template: `{{bos_token}}` + ChatML. The model's `generation_config` has `eos_token_id=[32001, 2]` and no sampling params.
- The TRL prompt / prompt+completion token prefix is preserved (no "Mismatch" warning), and there is exactly 1 BOS.
- Completion tokens: `['O','dp','ow','ied','ź',':','▁A','<|im_end|>','▁','<0x0A>']`. This is a legacy-tokenizer quirk (`▁` before the newline). Training and inference are consistent as long as both use `apply_chat_template`.
- Bielik-4.5B-v3.0 and 11B-v2.6 return 401: **UNVERIFIED**, including their architecture, template and generation_config.

---

## 3. peft 0.21

### 3.1 LoraConfig and wrapping
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",  # or ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
                 task_type="CAUSAL_LM", use_rslora=False, use_dora=False, bias="none")
# QLoRA outside TRL: model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
peft_model = get_peft_model(model, cfg)   # get_peft_model(model, peft_config, adapter_name="default", mixed=False, autocast_adapter_dtype=True, revision=None, low_cpu_mem_usage=False)
```
- Defaults: `r=8`, `lora_alpha=8`, `lora_dropout=0.0`. Other fields: `exclude_modules`, `modules_to_save`, `init_lora_weights`, `layers_to_transform`, `rank_pattern`, `alpha_pattern`, `trainable_token_indices`, `target_parameters`, `ensure_weight_tying`.
- `"all-linear"` expands to every `nn.Linear`/`Conv1D` **except the output embedding**. Verified on tied-embedding SmolLM2.
- **`prepare_model_for_kbit_training` upcasts every fp16/bf16 parameter (except `Params4bit`) to fp32, even on a non-quantized model.** Verified: a bf16 model becomes all fp32. Only use it on 4-bit models.
- Without `gradient_checkpointing_kwargs` it passes `{}`, so torch warns and uses reentrant=True (source; torch 2.14 `checkpoint` defaults `use_reentrant=None`, which warns and becomes True).

### 3.2 Save, load, merge (verified)
- `peft_model.save_pretrained(dir)` writes `adapter_config.json`, `adapter_model.safetensors`, `README.md`. `trainer.save_model()` also adds tokenizer files, `chat_template.jinja` and `training_args.bin`.
- A checkpoint additionally has `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`.
```python
base   = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16)      # un-quantized base
model  = PeftModel.from_pretrained(base, ADAPTER_DIR)     # (model, model_id, adapter_name="default", is_trainable=False, config=None, autocast_adapter_dtype=True, ...)
merged = model.merge_and_unload()                         # (progressbar=False, safe_merge=False, adapter_names=None) -> plain LlamaForCausalLM
merged.generation_config.update(do_sample=False, temperature=None, top_p=None, top_k=None)  # avoid strict-save ValueError if needed
merged.save_pretrained(OUT_DIR)       # v5: always safetensors; safe_serialization= is accepted via **kwargs and ignored
tok.save_pretrained(OUT_DIR)
reloaded = AutoModelForCausalLM.from_pretrained(OUT_DIR, dtype=torch.bfloat16)
```
- The merged dir contains `config.json` (with `dtype`), `generation_config.json`, `model.safetensors` (272 tensors, 0 LoRA keys), tokenizer files and `chat_template.jinja`.
- **Equivalence, strong random adapter** (base vs base+adapter logits differ by up to 22):

  | Setup | Result |
  |---|---|
  | CPU fp32 | reloaded merged vs base+adapter: max\|Δlogit\| 1.5e-4, argmax equal, greedy generations identical |
  | MPS fp32 | max\|Δlogit\| 9.9e-5, generations identical |
  | MPS bf16 | max\|Δ\| 0.94, mean 0.18, **2 of 3 generations differ** (bf16 rounding: the unmerged path runs LoRA in fp32 via `autocast_adapter_dtype`, the merged path adds a bf16 delta) |
  | MPS bf16, real 3-step adapter | generations identical |

  Evaluate the **merged model you ship**, not base+adapter.
- `AutoModelForCausalLM.from_pretrained(ADAPTER_DIR)` auto-loads base + adapter (`_hf_peft_config_loaded=True`). `peft.AutoPeftModelForCausalLM.from_pretrained(ADAPTER_DIR)` returns a `PeftModelForCausalLM`.
- Adapter trained on a 4-bit base, merged onto a bf16 base (source only; magnitude **UNVERIFIED**):
  - Load the base un-quantized in bf16 and merge there.
  - Merging into a `Linear4bit` layer dequantizes, adds the delta and re-quantizes, which adds rounding error.
  - The adapter learned against NF4-dequantized weights, so merged bf16 won't exactly reproduce 4-bit+adapter.
  - TRL saves QLoRA adapter params as bf16. `from_pretrained` upcasts them to fp32, and the merge casts the delta to the base weight dtype (`orig_weight += delta.to(orig_dtype)`).

---

## 4. MPS (torch 2.14) and CPU: LoRA SFT matrix
SFTTrainer, SmolLM2, LoRA r=16 all-linear, batch size 4, 4 steps. Every run had finite, decreasing loss and `lora_B` weights changed. No CPU-fallback or MPS warnings appeared, and `PYTORCH_ENABLE_MPS_FALLBACK` was not needed.

| Device | Weights / flags | Forward dtypes (base q_proj out / lora_A out) | Time |
|---|---|---|---|
| mps | fp32 weights, `bf16=True`, GC | bf16 / bf16 (autocast active) | 3.1 s |
| mps | **bf16 weights, `bf16=True`, GC** | bf16 / bf16; LoRA params fp32 | **1.9 s** |
| mps | bf16 weights, `bf16=False`, no GC | bf16 / fp32 | 2.9 s |
| mps | fp32 weights, `fp16=True`, GC | fp16 / fp16 | 3.6 s |
| mps | fp32, GC `use_reentrant=True` | fp32 | 2.4 s |
| cpu | `use_cpu=True`, fp32 | fp32 | 5.5 s |
| cpu | `use_cpu=True`, **`bf16=True`** | bf16 autocast | **320 s (≈60× slower)** |

Also: fused AdamW works on MPS. Gradient checkpointing works with PEFT (TRL calls `enable_input_require_grads`). `TrainingArguments` picks `mps` automatically.

---

## 5. sentence-transformers 6.0.1 (verified with `intfloat/multilingual-e5-small` on MPS)
```python
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("intfloat/multilingual-e5-small", device="mps",
                        prompts={"query": "query: ", "document": "passage: "})   # e5 ships NO saved prompts
q = m.encode_query("Kto napisał Pana Tadeusza?", normalize_embeddings=True)      # prepends prompts["query"]
D = m.encode_document(docs, batch_size=32, normalize_embeddings=True)            # prepends prompts["document"] (then "passage", "corpus")
E = m.encode(texts, prompt="passage: ", batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
sims = util.cos_sim(q, D)        # torch.Tensor [n_q, n_d]; equals m.similarity(q, D) for cosine
```
- Signature: `encode(inputs, prompt_name=None, prompt=None, batch_size=32, show_progress_bar=None, output_value='sentence_embedding', precision='float32', convert_to_numpy=True, convert_to_tensor=False, device=None, normalize_embeddings=False, truncate_dim=None, pool=None, chunk_size=None, **kwargs)`.
- `sentences=` still works but logs "renamed and is now deprecated".
- Return types:
  - list input: `ndarray (n, 384) float32`
  - str input: `(384,)`
  - `convert_to_tensor=True`: a tensor on `mps:0`
- Loaded without `prompts=`, `m.prompts == {'query': '', 'document': ''}`, so **`encode_query` adds no prefix** (verified). `prompt="query: "` gives the same result as a manual prefix (verified). `max_seq_length=512`.

---

## GOTCHAS
1. **Mac load crash (segfault, exit 139):** `from_pretrained(..., dtype=<not the checkpoint dtype>, device_map="mps")` crashes in the async weight loader (fp32 and fp16 both crash; `dtype="auto"` is fine). **`SFTTrainer(model="<id>")` on mac crashes the same way**, because it defaults to float32 + `device_map="auto"`. Fixes: `HF_DEACTIVATE_ASYNC_LOAD=1`, or load on CPU and call `.to("mps")`, or pass a model object. Whether CUDA is affected is **UNVERIFIED**.
2. `dtype` defaults to `"auto"` and reads the legacy `torch_dtype` key, so Bielik-11B-v2.3 loads in **fp16**. Pass `dtype=torch.bfloat16`. The TRL string-model path defaults to **float32**; use `model_init_kwargs={"dtype": "bfloat16"}`.
3. `apply_chat_template(tokenize=True)` returns a **BatchEncoding** by default. Use `**enc` or `return_dict=False`.
4. `continue_final_message` and `add_generation_prompt` together raise ValueError.
5. **Double BOS with Bielik:** `tok(rendered_template)` gives `[1, 1, 32000, ...]`. Use `add_special_tokens=False`, or tokenize through `apply_chat_template`.
6. Set `tok.padding_side="left"` explicitly for generation (defaults differ per tokenizer). TRL's collator pads right on its own.
7. Greedy warnings: pass `do_sample=False` as a kwarg, or clear the sampling fields on `model.generation_config`. A fresh `GenerationConfig(do_sample=False)` still inherits temperature/top_p/top_k. A bad generation config makes `save_pretrained` raise.
8. **Removed TrainingArguments** (§1.6): `evaluation_strategy`, `warmup_ratio`, `overwrite_output_dir`, `logging_dir`, `group_by_length`, `save_safetensors`, `no_cuda`, `use_mps_device`. **SFTConfig `max_seq_length` is gone; use `max_length`.** `pad_token` is deprecated. There is no `tokenizer=` argument anywhere.
9. `report_to` defaults to `"none"`, `output_dir` to `"trainer_output"`.
10. SFTConfig silently enables `bf16=True` and `gradient_checkpointing=True`, and uses `learning_rate=2e-5` (low for LoRA; set about 1e-4 to 2e-4).
11. `max_length` truncation keeps the start. A long prompt can push the completion out, and **fully masked examples are silently dropped**. Check dataset length after preparation.
12. Packing with the bfd strategy forces padding-free, which is only safe with flash-attn. Otherwise TRL warns about cross-contamination. Keep `packing=False` unless flash-attn is installed.
13. Default `loss_type="chunked_nll"`: gives no logits for `compute_metrics` (source) and errors if `lm_head` is a LoRA target. Use `loss_type="nll"` if you need logits.
14. `assistant_only_loss=True` raises ValueError for Bielik/SmolLM2 templates. Use prompt-completion data instead; completion-only loss is on by default and includes `<|im_end|>`.
15. With a dict `eval_dataset`, `on_evaluate` fires once per dataset. `eval_entropy`, `eval_mean_token_accuracy` and `eval_num_tokens` are **not** per-dataset (last one wins). Use `eval_<name>_loss`, and `metric_for_best_model="<name>_loss"` works.
16. With `eval_strategy="epoch"`, `on_epoch_end` runs before that epoch's evaluation.
17. Don't call `prepare_model_for_kbit_training` on a non-quantized model: it upcasts everything to fp32. Pass `gradient_checkpointing_kwargs={"use_reentrant": False}` when you do call it.
18. Merge onto an un-quantized bf16 base. A merged dir must **not** contain `adapter_config.json`, or `from_pretrained` treats it as an adapter. `safe_serialization=` is ignored (always safetensors). In bf16, merged and unmerged outputs can differ slightly, so eval the merged model.
19. MPS: set `dataloader_pin_memory=False` to silence the pin_memory UserWarning.
20. CPU: **never use `bf16=True` with `use_cpu=True`** (≈60× slower than fp32).
21. sentence-transformers: e5 models ship no prompts. Pass `prompts={"query":"query: ","document":"passage: "}` or `prompt=`, otherwise `encode_query` adds nothing.

## UNVERIFIED (not runnable here)
- Real bitsandbytes NF4 loading/training, QLoRA→bf16 merge error size, and CUDA `device_map` / flash-attn behaviour (source-read only).
- Whether the segfault in G1 also happens on CUDA.
- Bielik-4.5B-v3.0-Instruct and 11B-v2.6 configs, chat templates and generation_config (gated, 401).