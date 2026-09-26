"""Device / dtype / quantization resolution and model+tokenizer loading shared by eval, training and merge.

Verified gotchas handled here (docs/research/hf_stack_api.md):
  * macOS: from_pretrained(dtype!=checkpoint dtype, device_map="mps") segfaults in the async loader
    -> HF_DEACTIVATE_ASYNC_LOAD=1 (must be set before transformers is imported).
  * Bielik-11B-v2.3 config says float16 -> always pass an explicit dtype.
  * Tokenizers differ in default padding side -> force "left" for generation.
  * Greedy decoding warnings / strict GenerationConfig validation on save -> clear sampling fields.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import warnings
from pathlib import Path
from typing import Any

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def resolve_dtype(name: str | None, device: str | None = None) -> torch.dtype:
    device = device or detect_device()
    if name and name != "auto":
        return _DTYPES[name]
    if device == "cuda":  # pre-Ampere GPUs (T4/V100/P100) only emulate bf16: slow kernels, so fp16 there
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def resolve_quantization(name: str | None, device: str | None = None) -> str:
    """'auto' -> '4bit' on CUDA with bitsandbytes installed, else 'none'."""
    device = device or detect_device()
    if name in (None, "auto"):
        return "4bit" if device == "cuda" and bitsandbytes_available() else "none"
    if name not in ("4bit", "none"):
        raise ValueError(f"quantization must be auto|4bit|none, got {name!r}")
    if name == "4bit" and device != "cuda":
        raise RuntimeError("4-bit (QLoRA) requires CUDA + bitsandbytes; use quantization: none on MPS/CPU")
    return name


def dtype_name(dt: torch.dtype) -> str:
    return {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dt]


def is_adapter_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "adapter_config.json").exists()


def adapter_base_model(path: str | Path) -> str:
    with open(Path(path) / "adapter_config.json", encoding="utf-8") as fh:
        return json.load(fh)["base_model_name_or_path"]


def load_tokenizer(path_or_id: str | Path, padding_side: str = "left"):
    from transformers import AutoTokenizer

    src = str(path_or_id)
    if is_adapter_dir(src) and not (Path(src) / "tokenizer_config.json").exists():
        src = adapter_base_model(src)
    tok = AutoTokenizer.from_pretrained(src)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def sanitize_generation_config(model) -> None:
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.update(do_sample=False, temperature=None, top_p=None, top_k=None)


def resolve_attn_implementation(name: str | None) -> str:
    """'auto' / None -> 'sdpa'; anything else as given (e.g. 'flash_attention_2', 'eager')."""
    return "sdpa" if name in (None, "auto") else name


def load_model(
    path_or_id: str | Path,
    *,
    dtype: str | None = "auto",
    quantization: str | None = "none",
    device: str | None = None,
    attn_implementation: str | None = "auto",
    for_training: bool = False,
    merge_adapter: bool = True,
) -> Any:
    """Load a causal LM from a hub id, a merged model dir, or a LoRA adapter dir.

    Adapter dirs load base + adapter; for inference (for_training=False, unquantized) the adapter is merged
    in memory so what we evaluate is numerically what merge_and_export ships.
    """
    from transformers import AutoModelForCausalLM

    device = device or detect_device()
    torch_dtype = resolve_dtype(dtype, device)
    quant = resolve_quantization(quantization, device)
    src = str(path_or_id)
    adapter_dir = src if is_adapter_dir(src) else None
    base_src = adapter_base_model(src) if adapter_dir else src

    kwargs: dict[str, Any] = {"dtype": torch_dtype, "attn_implementation": resolve_attn_implementation(attn_implementation)}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif device == "cuda":
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))} if for_training else "auto"
    elif device == "mps":
        kwargs["device_map"] = {"": "mps"}
    # cpu: default (no device_map)

    model = AutoModelForCausalLM.from_pretrained(base_src, **kwargs)
    offloaded = sorted(k for k, v in (getattr(model, "hf_device_map", None) or {}).items() if v in ("cpu", "disk"))
    if offloaded:
        warnings.warn(
            f"{len(offloaded)} module(s) offloaded to CPU/disk ({', '.join(offloaded[:4])}...): the model does not fit in GPU "
            "memory, so every forward pass shuttles weights over PCIe (several times slower). Use a larger GPU or "
            "quantization='4bit' (eval.quantization=4bit evaluates base and run alike).",
            RuntimeWarning, stacklevel=2,
        )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=for_training)
        if merge_adapter and not for_training and quant == "none":
            model = model.merge_and_unload()
    sanitize_generation_config(model)
    if not for_training:
        model.eval()
    attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    model._matura_load_info = {  # type: ignore[attr-defined]
        "device": device,
        "dtype": dtype_name(torch_dtype),
        "quantization": quant,
        "attn_implementation": attn if isinstance(attn, str) else resolve_attn_implementation(attn_implementation),
        "source": src,
        "base_source": base_src,
        "adapter": adapter_dir,
    }
    return model


def load_model_and_tokenizer(path_or_id: str | Path, **kwargs):
    model = load_model(path_or_id, **kwargs)
    tok = load_tokenizer(path_or_id, padding_side="left")
    return model, tok


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def free_model(model) -> None:
    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
