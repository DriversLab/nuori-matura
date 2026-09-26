"""Merge a LoRA adapter into its UNQUANTIZED base and export an eval/submission-ready model dir.

QLoRA adapters are merged onto a bf16 base too (merging into Linear4bit re-quantizes and adds rounding error).
Merging runs on CPU unless CUDA is requested or available: on macOS a CPU merge avoids MPS unified-memory pressure
and the async-loader crash (eval.modeling also sets HF_DEACTIVATE_ASYNC_LOAD). A CPU bf16 merge is pure tensor
addition, so it is fast enough even for 11B (needs ~2x the weights in RAM).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

from eval.modeling import adapter_base_model, free_model, is_adapter_dir, load_model, load_tokenizer, sanitize_generation_config

RUN_ARTIFACTS = ("train_metrics.json", "resolved_config.yaml", "training_meta.json")
DEVICES = ("auto", "cpu", "cuda", "mps")


def resolve_merge_device(device: str) -> str:
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return device


def copy_run_artifacts(adapter_dir: Path, out_dir: Path) -> list[str]:
    """Copy training metadata next to the merged weights (looked up in the adapter dir, then its parent)."""
    copied = []
    for name in RUN_ARTIFACTS:
        if (out_dir / name).exists():
            continue
        src = next((d / name for d in (adapter_dir, adapter_dir.parent) if (d / name).exists()), None)
        if src is not None:
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    return copied


def _merge_weights(base: str, adapter_dir: Path, out_dir: Path, *, dtype: str, device: str) -> None:
    from peft import PeftModel

    model = load_model(base, dtype=dtype, quantization="none", device=device)
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    sanitize_generation_config(merged)
    merged.save_pretrained(str(out_dir))


def merge_and_export(
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    base_model: str | None = None,
    dtype: str = "bf16",
    device: str = "auto",
) -> Path:
    """adapter + unquantized base -> merged weights + tokenizer + run metadata in out_dir. Returns out_dir.

    out_dir must be the adapter's own checkpoint dir or a dir that holds no other run: a checkpoint dir whose adapter/
    is a different adapter (e.g. checkpoints/CANDIDATE-<slug> or another run's checkpoint) is refused, because its
    metadata would stay while its merged weights silently became another model.
    """
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    if not is_adapter_dir(adapter_dir):
        raise FileNotFoundError(f"not a LoRA adapter dir (no adapter_config.json): {adapter_dir}")
    if out_dir.resolve() == adapter_dir.resolve():
        raise ValueError("out_dir must differ from adapter_dir (a merged dir must not contain adapter_config.json)")
    own_adapter = out_dir / "adapter"
    if own_adapter.exists() and own_adapter.resolve() != adapter_dir.resolve():
        raise ValueError(f"{out_dir} belongs to another run (its adapter is {own_adapter.resolve()}, not {adapter_dir.resolve()}); "
                         "refusing to overwrite its merged model")
    base = base_model or adapter_base_model(adapter_dir)
    device = resolve_merge_device(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    _merge_weights(base, adapter_dir, out_dir, dtype=dtype, device=device)
    free_model(None)  # the models went out of scope with _merge_weights: collect them and empty the accelerator cache
    # the tokenizer the adapter was trained with; left padding because the export is used for batched generation
    load_tokenizer(adapter_dir, padding_side="left").save_pretrained(str(out_dir))
    copy_run_artifacts(adapter_dir, out_dir)

    if (out_dir / "adapter_config.json").exists():
        raise RuntimeError(f"{out_dir} contains adapter_config.json: from_pretrained would load it as an adapter")
    return out_dir
