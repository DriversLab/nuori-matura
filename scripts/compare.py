#!/usr/bin/env python
"""Compare a run's eval with the base model's eval, write compare.json, print the report and record the run
in the registry (promotion gate -> CANDIDATE.json, runs.jsonl, LEADERBOARD.md).

  Mode A, both already evaluated:
    python scripts/compare.py --run run1 [--base-run base] [--config configs/run1.yaml] [--allow-fingerprint-mismatch] [--no-record]
  Mode B, evaluate base and/or model when their summaries are missing or stale, then compare:
    python scripts/compare.py --model checkpoints/run1 [--run-name run1] [--base-model ID] [--config ...] [--set k=v] [--limit N]

Mode B without --config evaluates a checkpoint with the eval settings it was trained/evaluated with
(<checkpoint>/resolved_config.yaml), so the pipeline's cached evals are reused.
Debug evals never overwrite full evals (same naming as scripts/run_eval.py): --limit N stores <name>-limit<N>, and a
--set that changes the eval fingerprint stores <name>-dbg<fingerprint[:8]>. Limited (--limit or eval.limit) and -dbg
evals are never recorded: the gate verdict is printed as a dry run.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

import yaml

from eval.compare import format_compare_table, write_compare
from train.config import (
    checkpoint_dir,
    get_dotted,
    is_base_run_name,
    is_debug_run_name,
    load_config,
    resolve_path,
    results_dir,
    run_dir,
    set_dotted,
)
from train.registry import BASE_RUN, check_run_name_owner, evaluate_gate, load_candidate, record_run, reference_eval_scope


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", help="mode A: name of an already-evaluated run under results/<base_slug>/runs/")
    mode.add_argument("--model", help="mode B: model path / hub id / adapter dir to evaluate (if needed) and compare")
    ap.add_argument("--base-run", default=BASE_RUN, help="mode A: run name of the base-model eval (default: base)")
    ap.add_argument("--run-name", help="mode B: run name for --model (default: model dir name)")
    ap.add_argument("--base-model", help="mode B: base model id (default: inferred from the checkpoint, else the config)")
    ap.add_argument("--limit", type=int, help="mode B: debug-only cap on items per set (such runs are never recorded)")
    ap.add_argument("--config", help="config file (default: configs/base.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override, repeatable")
    ap.add_argument("--allow-fingerprint-mismatch", action="store_true",
                    help="compare despite differing eval fingerprints (recorded; never auto-promotes)")
    ap.add_argument("--no-record", action="store_true", help="do not record the run in the registry (dry-run gate only)")
    args = ap.parse_args(argv)
    if args.run and (args.run_name or args.base_model or args.limit is not None):
        ap.error("--run-name/--base-model/--limit are mode B (--model) options")
    if args.model and args.base_run != BASE_RUN:
        ap.error("--base-run is a mode A (--run) option; mode B evaluates the base model itself")
    return args


def _resolve_model(model: str) -> str:
    """Existing local paths (relative to cwd or the repo root) become absolute; anything else is a hub id."""
    for cand in (pathlib.Path(model).expanduser(), ROOT / model):
        if cand.exists():
            return str(cand.resolve())
    from train.config import looks_like_local_path

    if looks_like_local_path(model):
        raise FileNotFoundError(f"model path not found: {model} (no such checkpoint yet? train it first: "
                                "python scripts/train.py --config configs/<run>.yaml)")
    return model


def _default_run_name(model: str) -> str:
    p = pathlib.PurePath(model.rstrip("/"))
    return p.parent.name if p.name == "adapter" and p.parent.name else p.name


def _infer_base_model(model: str) -> str | None:
    """model.base from the checkpoint's resolved_config.yaml, else the LoRA adapter's base_model_name_or_path."""
    path = pathlib.Path(model)
    if not path.is_dir():
        return None
    checkpoint = path.parent if path.name == "adapter" else path
    resolved = checkpoint / "resolved_config.yaml"
    if resolved.exists():
        base = get_dotted(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}, "model.base")
        if base:
            return base
    for adapter in (path, path / "adapter"):
        if (adapter / "adapter_config.json").exists():
            return json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]
    return None


def _evaluate(cfg: dict, model: str, run_name: str, base_run: str, limit: int | None) -> None:
    """run_eval reuses a summary only when eval fingerprint, model identity and weights signature all match."""
    from eval.runner import run_eval  # heavy (torch/transformers): only needed in mode B

    run_eval(cfg["model"]["base"], base_run, cfg, is_base=True, limit=limit)
    run_eval(model, run_name, cfg, limit=limit)


def _load_summary(run_path: pathlib.Path) -> dict:
    with open(run_path / "summary.json", encoding="utf-8") as fh:
        return json.load(fh)


def _adapter_path(cfg: dict, run_name: str, model_path: str) -> str | None:
    model = resolve_path(model_path)
    for cand in (model, model / "adapter", checkpoint_dir(cfg, run_name) / "adapter"):
        if (cand / "adapter_config.json").exists():
            return str(cand)
    return None


def _print_gate(passed: bool, reasons: list[str], header: str, cfg: dict) -> None:
    print(f"\nGATE {'PASSED' if passed else 'FAILED'}: {header}")
    for reason in reasons:
        print(f"  {reason}")
    cand = load_candidate(results_dir(cfg))
    if cand:
        score = "n/a" if cand.get("score_pct") is None else f"{cand['score_pct']:.2f}%"
        print(f"CANDIDATE: {cand['run_name']}  delta {cand['delta_points']:+.2f} pts  score {score}  model {cand['model_path']}")
        for warning in cand.get("warnings") or []:
            print(f"  WARNING: {warning}")
    else:
        print("CANDIDATE: none recorded yet (the untouched base model is the fallback)")


def _other_slug_hint(cfg: dict, run_name: str) -> None:
    """Without --config, results resolve under configs/base.yaml's base model: point at the slug(s) that have the run."""
    if (run_dir(cfg, run_name) / "summary.json").exists():
        return
    root = resolve_path(get_dotted(cfg, "paths.results_root", "results"))
    slugs = sorted(p.parents[2].name for p in root.glob(f"*/runs/{run_name}/summary.json"))
    if slugs:
        raise ValueError(f"run {run_name!r} is not under {results_dir(cfg)} (default config configs/base.yaml) but exists for "
                         f"base model(s) {', '.join(slugs)}; pass --config with that base model's config")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.model:
        from eval.runner import checkpoint_eval_overrides, debug_suffix  # heavy (torch): only needed in mode B

        model = _resolve_model(args.model)
        settings = [] if args.config else checkpoint_eval_overrides(model)
        cfg, plain = load_config(args.config, settings + args.set), load_config(args.config, settings)
        base_model = args.base_model or _infer_base_model(model)
        for c in (cfg, plain):
            if base_model:
                set_dotted(c, "model.base", base_model)
        limit = args.limit if args.limit is not None else get_dotted(cfg, "eval.limit")
        suffix = debug_suffix(cfg, plain, limit=args.limit)
        run_name, base_run = (args.run_name or _default_run_name(model)) + suffix, BASE_RUN + suffix
    else:
        cfg = load_config(args.config, args.set)
        run_name, base_run = args.run, args.base_run
        if not args.config:
            _other_slug_hint(cfg, run_name)
    if run_name == base_run or is_base_run_name(run_name):
        raise ValueError(f"run name {run_name!r} is reserved for base-model evals; pick a different --run/--run-name")
    if args.model:
        check_run_name_owner(cfg, run_name, model)  # before evaluating: the eval would overwrite runs/<candidate>/
        _evaluate(cfg, model, run_name, base_run, limit)

    rd = run_dir(cfg, run_name)
    cmp = write_compare(rd, run_dir(cfg, base_run), allow_fingerprint_mismatch=args.allow_fingerprint_mismatch)
    print(format_compare_table(cmp))
    print(f"\nwrote {rd / 'compare.json'}")

    summary = _load_summary(rd)
    eval_limit = (summary.get("fingerprint_parts") or {}).get("limit")
    if args.no_record or eval_limit is not None or is_debug_run_name(run_name):
        why = ("--no-record" if args.no_record else f"eval limit={eval_limit}, debug evals are never recorded" if eval_limit is not None
               else "CLI overrides changed the eval (-dbg run), debug evals are never recorded")
        passed, reasons = evaluate_gate(cmp, load_candidate(results_dir(cfg)), cfg["gate"], required_scope=reference_eval_scope())
        _print_gate(passed, reasons, f"dry run, not recorded ({why})", cfg)
        return 0

    row = record_run(
        cfg, run_dir=rd, cmp=cmp, train_metrics=None, model_path=summary["model"],
        adapter_path=_adapter_path(cfg, run_name, summary["model"]),
    )
    header = f"{run_name} promoted to CANDIDATE" if row["promoted"] else f"{run_name} not promoted (candidate unchanged)"
    _print_gate(row["gate_passed"], row["gate_reasons"], header, cfg)
    print(f"recorded in {results_dir(cfg) / 'runs.jsonl'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
