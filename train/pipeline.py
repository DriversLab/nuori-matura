"""End-to-end run: candidate check -> processed data -> LoRA SFT -> merge -> base/run eval -> compare -> gate/registry.

Sibling stages are imported lazily so this module (and scripts/train.py --help) import without them.
"""
from __future__ import annotations

import json
import os
import shlex
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from train.config import (
    ROOT,
    base_slug,
    candidate_link_name,
    checkpoint_dir,
    checkpoints_root,
    get_dotted,
    resolve_path,
    run_dir,
)

BASE_RUN = "base"


@contextmanager
def _stage(timings: dict[str, float], name: str) -> Iterator[None]:
    start = time.monotonic()
    try:
        yield
    finally:
        timings[name] = round(time.monotonic() - start, 1)


def display_path(path: str | Path) -> str:
    """Repo-relative path when inside the repo (copy-pasteable commands), else as given. Symlinks are shown, not
    followed (checkpoints/CANDIDATE must print as itself)."""
    try:
        return str(Path(os.path.abspath(path)).relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_flags(cfg: dict, *, do_merge: bool, do_eval: bool) -> tuple[bool, bool]:
    """CLI flags AND pipeline.{merge_after_train, auto_eval}. Evaluating without merging is refused: the gate must
    judge the merged export that would be shipped (an adapter on a 4-bit eval base is a different model)."""
    merge = do_merge and bool(get_dotted(cfg, "pipeline.merge_after_train", True))
    evaluate = do_eval and bool(get_dotted(cfg, "pipeline.auto_eval", True))
    if evaluate and not merge:
        raise ValueError("evaluation requires the merged export: enable merging or pass --no-eval")
    return merge, evaluate


def candidate_commands(cfg: dict, candidate: dict) -> list[str]:
    """Human lines: which model to submit and the exact command that re-evaluates it under this config."""
    config = shlex.quote(display_path(cfg["_config_path"])) if cfg.get("_config_path") else None
    config_arg = f" --config {config}" if config else ""
    if candidate.get("run_name", BASE_RUN) == BASE_RUN:
        model = candidate.get("model_path") or cfg["model"]["base"]
        return [
            f"  submit model: {model}  (untouched base: no run has passed the gate yet)",
            f"  evaluate:     python scripts/run_eval.py{config_arg} --base --model {shlex.quote(model)}",
        ]
    model_path = resolve_path(candidate.get("model_path") or checkpoint_dir(cfg, candidate["run_name"]))
    model = display_path(model_path)
    link = checkpoints_root(cfg) / candidate_link_name(base_slug(cfg["model"]["base"]))
    points_here = link.is_symlink() and link.resolve() == model_path.resolve()  # another base model's link must not show
    stable = f"  (stable link: {display_path(link)})" if points_here else ""
    return [
        f"  submit model: {model}{stable}",
        f"  evaluate:     python scripts/run_eval.py{config_arg} --model {shlex.quote(model)}",
    ]


def format_verdict(cfg: dict, *, run_name: str, passed: bool, reasons: list[str], candidate: dict) -> str:
    lines = [f"Gate: {'PASS' if passed else 'FAIL'}"]
    lines += [f"  - {r}" for r in reasons]
    cand_name = candidate.get("run_name", BASE_RUN)
    promoted = "  (promoted by this run)" if cand_name == run_name else ""
    delta = candidate.get("delta_points")
    delta_txt = f", delta_points={delta:+.2f}" if isinstance(delta, (int, float)) else ""
    lines.append(f"Current candidate: {cand_name}{delta_txt}{promoted}")
    lines += [f"  WARNING: {w}" for w in candidate.get("warnings") or []]
    lines += candidate_commands(cfg, candidate)
    return "\n".join(lines)


def _bf16_weight_bytes(base: str) -> int | None:
    """Approximate bf16 weight bytes of a local model dir or hub id (None when unknown, e.g. offline or gated)."""
    local = resolve_path(base)
    if local.is_dir():
        index = local / "model.safetensors.index.json"
        if index.exists():
            return int(json.loads(index.read_text(encoding="utf-8")).get("metadata", {}).get("total_size") or 0) or None
        return sum(f.stat().st_size for f in local.glob("*.safetensors")) or None
    try:
        from huggingface_hub import HfApi

        return 2 * sum(HfApi().get_safetensors_metadata(base).parameter_count.values())
    except Exception:  # offline, gated without token, no safetensors: skip the check
        return None


def check_eval_vram(cfg: dict) -> None:
    """Refuse before training when the unquantized bf16 merge + evals (eval.quantization none) cannot fit the GPU:
    device_map="auto" would silently offload layers to CPU (several times slower) or OOM hours into the run."""
    from eval.modeling import detect_device, resolve_quantization

    if detect_device() != "cuda" or resolve_quantization(get_dotted(cfg, "eval.quantization") or "none", "cuda") != "none":
        return
    import torch

    base = cfg["model"]["base"]
    weights, total = _bf16_weight_bytes(base), torch.cuda.mem_get_info()[1]
    if weights and weights > 0.8 * total:  # accelerate budgets ~90% of free memory and reserves the largest layer
        raise RuntimeError(
            f"bf16 merge/eval of {base} needs ~{weights / 2**30:.1f} GiB of weights but the GPU has {total / 2**30:.1f} GiB: "
            "pass --set eval.quantization=4bit (base and run are then both evaluated in 4-bit) or use a >= 40 GB GPU"
        )


def refresh_stale_candidate(cfg: dict, fingerprint: str | None) -> dict | None:
    """Re-evaluate and re-record a non-base candidate that was measured under a different eval fingerprint, so the new
    run is gated against the candidate's delta under the SAME eval (gate rule 7 refuses to compare across
    fingerprints). A candidate that fails the gate under the current eval is demoted to base by record_run.
    Returns the candidate's new row, or None when nothing was re-measured."""
    if not fingerprint:
        return None
    from train.registry import ensure_candidate

    candidate = ensure_candidate(cfg)
    if candidate["run_name"] == BASE_RUN or candidate.get("fingerprint") in (None, fingerprint):
        return None
    name, model = candidate["run_name"], resolve_path(candidate["model_path"])
    if not model.exists():
        print(f"[pipeline] candidate {name} was measured under eval fingerprint {candidate['fingerprint']} (now {fingerprint}) "
              f"but {model} is not on this machine: gate rule 7 fails until it is re-evaluated here")
        return None
    from eval.compare import write_compare
    from eval.runner import run_eval
    from train.registry import record_run

    print(f"[pipeline] re-evaluating candidate {name} under the current eval (fingerprint {candidate['fingerprint']} -> {fingerprint})")
    run_eval(str(model), name, cfg)
    cmp = write_compare(run_dir(cfg, name), run_dir(cfg, BASE_RUN))
    adapter = candidate.get("adapter_path")
    return record_run(cfg, run_dir=run_dir(cfg, name), cmp=cmp, train_metrics=None, model_path=model,
                      adapter_path=resolve_path(adapter) if adapter else None)


def _save_train_metrics(ckpt: Path, train_metrics: dict) -> None:
    (ckpt / "train_metrics.json").write_text(json.dumps(train_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_pipeline(cfg: dict, *, do_merge: bool = True, do_eval: bool = True) -> dict:
    """Train (+merge) and, unless disabled, evaluate against base, write compare.json and record/gate the run.

    Returns {"run_name", "data", "adapter_dir", "model_dir" (None without merge), "train_metrics"} plus, when evaluated,
    {"summary", "compare", "gate": {"passed", "reasons"}, "row", "candidate"}. Stage durations are stored in
    train_metrics["stage_seconds"] (and in <checkpoint>/train_metrics.json).
    """
    from train.dataset import build_processed
    from train.merge import merge_and_export
    from train.registry import ensure_candidate
    from train.sft import assert_trainable_run_name, train_model

    do_merge, do_eval = resolve_flags(cfg, do_merge=do_merge, do_eval=do_eval)
    run_name = cfg["run_name"]
    ckpt = checkpoint_dir(cfg)
    timings: dict[str, float] = {}
    ensure_candidate(cfg)
    assert_trainable_run_name(cfg)
    if do_eval:
        from eval.runner import eval_sets_from_cfg

        eval_sets_from_cfg(cfg)  # a missing required eval set fails now, not after training and merging
        check_eval_vram(cfg)

    with _stage(timings, "data"):
        data = build_processed(cfg)
    with _stage(timings, "train"):
        trained = train_model(cfg, data)
    adapter_dir, train_metrics = trained["adapter_dir"], trained["train_metrics"]
    train_metrics["stage_seconds"] = timings  # same dict object: later stages show up in every save below
    model_dir = None
    if do_merge:
        with _stage(timings, "merge"):
            model_dir = merge_and_export(adapter_dir, ckpt, base_model=cfg["model"]["base"])
    _save_train_metrics(ckpt, train_metrics)
    result = {"run_name": run_name, "data": data, "adapter_dir": adapter_dir, "model_dir": model_dir, "train_metrics": train_metrics}
    if not do_eval:
        return result

    from eval.compare import format_compare_table, write_compare
    from eval.runner import run_eval
    from train.registry import record_run

    base_dir, this_dir = run_dir(cfg, BASE_RUN), run_dir(cfg)
    with _stage(timings, "eval_base"):
        run_eval(cfg["model"]["base"], BASE_RUN, cfg, is_base=True)  # cached when the eval fingerprint matches
    with _stage(timings, "eval_run"):
        summary = run_eval(str(model_dir), run_name, cfg, force=True)
    with _stage(timings, "compare"):
        cmp = write_compare(this_dir, base_dir)
    with _stage(timings, "refresh_candidate"):
        refresh_stale_candidate(cfg, cmp.get("fingerprint"))
    _save_train_metrics(ckpt, train_metrics)
    row = record_run(cfg, run_dir=this_dir, cmp=cmp, train_metrics=train_metrics, model_path=model_dir, adapter_path=adapter_dir)
    passed, reasons = bool(row["gate_passed"]), list(row["gate_reasons"])  # gated against the candidate before this run
    candidate = ensure_candidate(cfg)

    print(format_compare_table(cmp))
    print(format_verdict(cfg, run_name=run_name, passed=passed, reasons=reasons, candidate=candidate))
    result.update(summary=summary, compare=cmp, gate={"passed": passed, "reasons": reasons}, row=row, candidate=candidate)
    return result
