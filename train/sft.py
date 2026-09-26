"""LoRA / QLoRA supervised fine-tuning with TRL on processed prompt-completion rows.

Layout written under checkpoint_dir(cfg):
    adapter/              LoRA adapter + tokenizer (trainer.save_model)
    trainer/              SFTTrainer output_dir (train.save_strategy "no" by default; "epoch" keeps checkpoint-<step>/ per epoch)
    train_metrics.json    step-0 (== base) losses, per-eval-round losses, data counts, loss-token split, runtime
    resolved_config.yaml  the fully resolved run config
    training_meta.json    run name, base model, config hash, device/quantization/dtype, data stats, library versions

Verified library facts: docs/research/hf_stack_api.md (SFTConfig names, completion-only loss, dict eval_dataset metric
names, bf16 on CPU, pin_memory on MPS, prepare_model_for_kbit_training only for 4-bit).
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

# eval.modeling sets HF_DEACTIVATE_ASYNC_LOAD before transformers is imported (macOS load-crash gotcha)
from eval.modeling import (
    detect_device,
    dtype_name,
    free_model,
    load_model,
    load_tokenizer,
    resolve_dtype,
    resolve_quantization,
)
from eval.schema import read_jsonl
from train.config import checkpoint_dir, checkpoints_root, config_hash, dump_config, get_dotted, is_base_run_name, results_dir
from transformers import TrainerCallback, set_seed

# eval_dataset key -> key of the build_processed() output dict
EVAL_SPLITS = {"matura": "val_matura", "general": "val_general", "general_heldout": "general_heldout"}
LOSS_KEYS = tuple(f"{name}_loss" for name in EVAL_SPLITS)


# ----------------------------------------------------------------------------- config mapping (unit-tested)


def build_sft_kwargs(cfg: dict, *, output_dir: str | Path, device: str, dtype: torch.dtype, has_eval: bool = True) -> dict:
    """Map run config -> trl.SFTConfig kwargs (names verified against transformers 5.17 / trl 1.13)."""
    t = cfg["train"]
    warmup_ratio = float(t["warmup_ratio"])
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"train.warmup_ratio must be in [0, 1), got {warmup_ratio} (values >= 1 would mean steps)")
    eval_steps = t.get("eval_steps")
    if not has_eval:
        eval_strategy = "no"
    else:
        eval_strategy = "steps" if eval_steps else "epoch"
    on_accelerator = device != "cpu"  # never mixed precision on CPU (bf16 autocast is ~60x slower there)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(t["epochs"]),
        "learning_rate": float(t["learning_rate"]),
        "lr_scheduler_type": t["lr_scheduler"],
        "warmup_steps": warmup_ratio,  # a float < 1 is read as a ratio (warmup_ratio was removed in v5)
        "per_device_train_batch_size": int(t["per_device_batch_size"]),
        "per_device_eval_batch_size": int(t["per_device_batch_size"]),
        "gradient_accumulation_steps": int(t["grad_accum"]),
        "weight_decay": float(t["weight_decay"]),
        "max_grad_norm": float(t["max_grad_norm"]),
        "gradient_checkpointing": bool(t["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": t["logging_steps"],
        "eval_strategy": eval_strategy,
        "save_strategy": t.get("save_strategy") or "no",  # "epoch": one adapter checkpoint per epoch in trainer/
        "max_steps": int(t["max_steps"]),
        "bf16": on_accelerator and dtype == torch.bfloat16,
        "fp16": on_accelerator and dtype == torch.float16,
        "use_cpu": device == "cpu",
        "report_to": "none",
        "seed": int(cfg["seed"]),
        "max_length": int(cfg["model"]["max_length"]),
        "packing": bool(t["packing"]),
        "completion_only_loss": True,
        "dataloader_pin_memory": device == "cuda",
    }
    if eval_strategy == "steps":
        kwargs["eval_steps"] = eval_steps
    return kwargs


def build_lora_config(cfg: dict):
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
        use_rslora=bool(lora.get("use_rslora", False)),
        bias="none",
    )


# ----------------------------------------------------------------------------- data


def to_prompt_completion(row: dict) -> dict:
    """Keep only TRL columns. Accepts processed rows or raw general pairs ({"messages": [..., assistant]})."""
    if "prompt" in row and "completion" in row:
        return {"prompt": row["prompt"], "completion": row["completion"]}
    msgs = row.get("messages")
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError(f"row {row.get('id', '?')}: expected prompt/completion or messages ending with an assistant turn")
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def token_lengths(rows: list[dict], tokenizer, *, prompt_only: bool = False, chunk: int = 256) -> list[int]:
    """Token counts exactly as TRL tokenizes prompt-completion rows: apply_chat_template(prompt + completion), or
    apply_chat_template(prompt, add_generation_prompt=True) for the prompt part."""
    out: list[int] = []
    for start in range(0, len(rows), chunk):
        batch = rows[start : start + chunk]
        conversations = [r["prompt"] if prompt_only else r["prompt"] + r["completion"] for r in batch]
        ids = tokenizer.apply_chat_template(conversations, add_generation_prompt=prompt_only, return_dict=False)
        out.extend(len(seq) for seq in ids)
    return out


def train_composition(raw_rows: list[dict], lengths: list[int], tokenizer, max_length: int) -> dict:
    """What the trainer actually sees of the train split after the length filter. The loss is normalized by the number
    of completion tokens, so each row weighs by its completion length: loss_tokens.matura_share is the real share of
    the training signal that comes from matura rows (general_ratio counts rows)."""
    kinds = [r.get("kind") or "unknown" for r in raw_rows]
    keep = [n <= max_length for n in lengths]
    kept = [to_prompt_completion(r) for r, ok in zip(raw_rows, keep) if ok]
    kept_kinds = [k for k, ok in zip(kinds, keep) if ok]
    prompt_lengths = token_lengths(kept, tokenizer, prompt_only=True)
    tokens: Counter = Counter()
    for kind, full, prompt in zip(kept_kinds, (n for n, ok in zip(lengths, keep) if ok), prompt_lengths):
        tokens[kind] += max(full - prompt, 0)
    total = sum(tokens.values())
    return {
        "dropped_too_long_by_kind": {k: sum(1 for kk, ok in zip(kinds, keep) if kk == k and not ok) for k in sorted(set(kinds))},
        "general_ratio_trained": round(kept_kinds.count("general") / len(kept_kinds), 4) if kept_kinds else 0.0,
        "loss_tokens": {"matura": tokens["matura"], "general": tokens["general"],
                        "matura_share": round(tokens["matura"] / total, 4) if total else 0.0},
    }


def load_splits(data: dict, tokenizer, max_length: int) -> tuple[dict[str, list[dict]], dict[str, int], dict]:
    """{"train", "matura", "general", "general_heldout"} -> filtered rows, per-split dropped counts, and the train
    split's composition after filtering (train_composition).

    Rows whose tokenized prompt+completion exceeds max_length are dropped: TRL truncates the END of long rows and
    silently drops fully masked ones, so over-long rows would train on a cut-off (or missing) answer.
    Eval splits whose file is missing/unset or empty are returned as empty lists (and later skipped).
    """
    sources = {"train": data["train"], **{name: data.get(key) for name, key in EVAL_SPLITS.items()}}
    rows: dict[str, list[dict]] = {}
    dropped: dict[str, int] = {}
    composition: dict = {}
    for name, path in sources.items():
        if path is None or not Path(path).exists():
            if name == "train":
                raise FileNotFoundError(f"training rows not found: {path}")
            rows[name], dropped[name] = [], 0
            continue
        raw_rows = read_jsonl(path)
        pairs = [to_prompt_completion(r) for r in raw_rows]
        lengths = token_lengths(pairs, tokenizer)
        rows[name] = [r for r, n in zip(pairs, lengths) if n <= max_length]
        dropped[name] = len(pairs) - len(rows[name])
        if name == "train":
            composition = train_composition(raw_rows, lengths, tokenizer, max_length)
    if not rows["train"]:
        raise RuntimeError(f"no training rows left (file {data['train']}, dropped_too_long={dropped['train']})")
    return rows, dropped, composition


# ----------------------------------------------------------------------------- loss recording


class LossRecorder(TrainerCallback):
    """Collects train losses (on_log) and eval_<name>_loss (on_evaluate), grouped per eval round (global step).

    on_evaluate fires once per eval dataset, so a round is keyed by global_step; the round at step 0 is the
    pre-training evaluate() (== base model losses).
    """

    def __init__(self) -> None:
        self.rounds: dict[int, dict] = {}
        self._pending_train: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._pending_train.append(float(logs["loss"]))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = int(state.global_step)
        rec = self.rounds.get(step)
        if rec is None:
            train_loss = fmean(self._pending_train) if self._pending_train else None
            self._pending_train = []
            rec = {"epoch": round(float(state.epoch or 0.0), 4), "step": step, "train_loss": train_loss}
            rec.update({k: None for k in LOSS_KEYS})
            self.rounds[step] = rec
        for key in LOSS_KEYS:
            value = (metrics or {}).get(f"eval_{key}")
            if value is not None:
                rec[key] = float(value)

    def last_step(self) -> int | None:
        return max(self.rounds) if self.rounds else None

    def step0(self) -> dict:
        rec = self.rounds.get(0, {})
        return {k: rec.get(k) for k in LOSS_KEYS}

    def epochs(self) -> list[dict]:
        return [self.rounds[s] for s in sorted(self.rounds) if s > 0]


# ----------------------------------------------------------------------------- training


def assert_trainable_run_name(cfg: dict) -> None:
    """Refuse run names that would clobber the base eval, a candidate's (only shippable) checkpoint of ANY base model,
    or a checkpoint trained on another base model (checkpoints/<run_name> is shared by all base models)."""
    from train.registry import protected_checkpoints  # lazy: sibling module

    name = cfg["run_name"]
    if is_base_run_name(name):
        raise ValueError(f"run_name {name!r} is reserved for the untouched base model's eval")
    ckpt = checkpoint_dir(cfg)
    cand_file = results_dir(cfg) / "CANDIDATE.json"
    named_here = cand_file.exists() and json.loads(cand_file.read_text(encoding="utf-8")).get("run_name") == name
    if named_here or ckpt.name in protected_checkpoints(results_dir(cfg), checkpoints_root(cfg)):
        raise RuntimeError(f"run_name {name!r}: {ckpt} is a current CANDIDATE checkpoint (of some base model); refusing to "
                           "overwrite it. Use --set run_name=<new>.")
    meta = ckpt / "training_meta.json"
    other = json.loads(meta.read_text(encoding="utf-8")).get("base_model") if meta.exists() else None
    if other and other != cfg["model"]["base"]:
        raise RuntimeError(f"{ckpt} holds run {name!r} trained on {other}, not {cfg['model']['base']}; refusing to overwrite "
                           "another base model's run. Use --set run_name=<new>.")


def _library_versions() -> dict[str, str]:
    import peft
    import transformers
    import trl

    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__}


def _write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _fit(cfg: dict, rows: dict[str, list[dict]], tokenizer, *, device: str, quant: str, dtype: torch.dtype, ckpt: Path) -> dict:
    """Load model, train, save adapter. Model and trainer live only in this frame so the caller can free them."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model = load_model(
        cfg["model"]["base"],
        dtype=dtype_name(dtype),
        quantization=quant,
        device=device,
        attn_implementation=get_dotted(cfg, "model.attn_implementation", "auto"),
        for_training=True,
    )
    if quant == "4bit":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg["train"]["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    eval_sets = {name: Dataset.from_list(rows[name]) for name in EVAL_SPLITS if rows[name]}
    args = SFTConfig(**build_sft_kwargs(cfg, output_dir=ckpt / "trainer", device=device, dtype=dtype, has_eval=bool(eval_sets)))
    recorder = LossRecorder()
    # SFTTrainer wraps the model with get_peft_model (lora_A kaiming init from the global torch RNG) BEFORE
    # Trainer.__init__ seeds; without this the LoRA init depends on the process's random torch seed.
    set_seed(int(cfg["seed"]))
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows["train"]),
        eval_dataset=eval_sets or None,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
        callbacks=[recorder],
    )
    del model  # the trainer owns the (PEFT-wrapped) model from here on

    if eval_sets:
        trainer.evaluate()  # LoRA B=0 -> these are the base model's losses
    result = trainer.train()
    if eval_sets and recorder.last_step() != trainer.state.global_step:
        trainer.evaluate()  # max_steps / eval_steps runs can end between eval rounds
    trainer.save_model(str(ckpt / "adapter"))
    return {
        "step0": recorder.step0(),
        "epochs": recorder.epochs(),
        "n_train_prepared": len(trainer.train_dataset),
        "global_steps": int(trainer.state.global_step),
        "train_loss_mean": float(result.training_loss),
    }


def train_model(cfg: dict, data: dict) -> dict:
    """Train a LoRA adapter on build_processed() output. Returns {"adapter_dir": Path, "train_metrics": dict}."""
    start = time.monotonic()
    assert_trainable_run_name(cfg)
    device = detect_device()
    quant = resolve_quantization(get_dotted(cfg, "model.quantization", "auto"), device)
    dtype = resolve_dtype(get_dotted(cfg, "model.dtype", "auto"), device)
    base = cfg["model"]["base"]
    ckpt = checkpoint_dir(cfg)
    if ckpt.exists():
        print(f"[sft] removing previous artifacts of run {cfg['run_name']!r}: {ckpt}")
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True)

    tokenizer = load_tokenizer(base, padding_side="right")
    max_length = int(cfg["model"]["max_length"])
    rows, dropped, composition = load_splits(data, tokenizer, max_length)
    n_eval = {name: len(rows[name]) for name in EVAL_SPLITS}
    print(
        f"[sft] {cfg['run_name']}: base={base} device={device} quantization={quant} dtype={dtype_name(dtype)} "
        f"n_train={len(rows['train'])} n_eval={n_eval} dropped_too_long={dropped} "
        f"general_ratio_trained={composition['general_ratio_trained']} matura_share_of_loss_tokens={composition['loss_tokens']['matura_share']}"
    )

    fit = _fit(cfg, rows, tokenizer, device=device, quant=quant, dtype=dtype, ckpt=ckpt)
    free_model(None)  # model/trainer went out of scope with _fit: collect them and empty the accelerator cache

    train_metrics = {
        "step0": fit["step0"],
        "epochs": fit["epochs"],
        "dropped_too_long": dropped["train"],  # training rows lost; eval-split drops are in dropped_too_long_by_split
        "dropped_too_long_by_split": dropped,
        **composition,
        "n_train": len(rows["train"]),
        "n_train_prepared": fit["n_train_prepared"],
        "n_eval": n_eval,
        "global_steps": fit["global_steps"],
        "train_loss_mean": fit["train_loss_mean"],
        "runtime_s": round(time.monotonic() - start, 1),
        "device": device,
        "quantization": quant,
        "dtype": dtype_name(dtype),
    }
    _write_json(train_metrics, ckpt / "train_metrics.json")
    dump_config(cfg, ckpt / "resolved_config.yaml")
    _write_json(
        {
            "run_name": cfg["run_name"],
            "base_model": base,
            "config_hash": config_hash(cfg),
            "config_path": cfg.get("_config_path"),
            "device": device,
            "quantization": quant,
            "dtype": dtype_name(dtype),
            "data": {"paths": {k: str(v) for k, v in data.items() if k != "stats"}, "stats": data.get("stats", {})},
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "versions": _library_versions(),
        },
        ckpt / "training_meta.json",
    )
    return {"adapter_dir": ckpt / "adapter", "train_metrics": train_metrics}
