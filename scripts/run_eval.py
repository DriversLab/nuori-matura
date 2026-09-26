#!/usr/bin/env python
"""Evaluate a model on the configured eval sets; for non-base runs, compare against the base model's eval.

  python scripts/run_eval.py --base --model speakleash/Bielik-4.5B-v3.0-Instruct
  python scripts/run_eval.py --model checkpoints/run1 --with-base
  python scripts/run_eval.py --model checkpoints/run1 --limit 5 --variants canonical --set eval.batch_size=4

Without --config a checkpoint is evaluated with the eval settings it was trained/evaluated with
(<checkpoint>/resolved_config.yaml: eval.*, model.dtype/attn_implementation/max_length, data.general_heldout), so the
pipeline's cached evals are reused; anything else uses configs/base.yaml.
Debug evals never overwrite a full eval: --limit N stores <name>-limit<N> (compared with base-limit<N>), and any
--set / --variants / --sets that changes the eval fingerprint stores <name>-dbg<fingerprint[:8]> (base-dbg<...>).
"base" names are reserved for --base evals.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="hub id, merged checkpoint dir, or LoRA adapter dir")
    ap.add_argument("--run-name", help='results name (default: "base" with --base, else the model dir name)')
    ap.add_argument("--config", help="config file (default: the checkpoint's resolved eval settings, else configs/base.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, repeatable (e.g. eval.batch_size=4)")
    ap.add_argument("--base", action="store_true", help="this is the untouched base model (model.base := --model)")
    ap.add_argument("--limit", type=int, help="debug: cap items per set (recorded in the fingerprint; results go to <name>-limit<N>)")
    ap.add_argument("--variants", help="comma-separated prompt variants for every set (e.g. canonical,bare)")
    ap.add_argument("--sets", help="comma-separated eval set paths replacing eval.sets")
    ap.add_argument("--force", action="store_true", help="re-evaluate even if a matching summary exists")
    ap.add_argument("--base-model", help="base model id (default: inferred from the checkpoint, else the config)")
    ap.add_argument("--with-base", action="store_true", help="evaluate the base model first when its eval is missing or stale")
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison with the base eval")
    return ap.parse_args(argv)


def resolve_model_arg(model: str) -> str:
    """Existing local paths (relative to cwd or the repo) become absolute; anything else is a hub id."""
    for candidate in (Path(model).expanduser(), ROOT / model):
        if candidate.exists():
            return str(candidate.resolve())
    from train.config import looks_like_local_path

    if looks_like_local_path(model):
        raise FileNotFoundError(f"model path not found: {model} (no such checkpoint yet? train it first: "
                                "python scripts/train.py --config configs/<run>.yaml)")
    return model


def default_run_name(model: str, is_base: bool) -> str:
    if is_base:
        return "base"
    path = Path(model).resolve() if Path(model).exists() else Path(model)
    return path.parent.name if path.name == "adapter" else path.name


def infer_base_model(model: str) -> str | None:
    """model.base from <checkpoint>/resolved_config.yaml, else the adapter's base_model_name_or_path."""
    import yaml

    from eval.modeling import adapter_base_model, is_adapter_dir
    from train.config import get_dotted

    path = Path(model)
    if not path.is_dir():
        return None
    checkpoint = path.parent if path.name == "adapter" else path
    resolved = checkpoint / "resolved_config.yaml"
    if resolved.exists():
        base = get_dotted(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}, "model.base")
        if base:
            return base
    for candidate in (path, path / "adapter"):
        if is_adapter_dir(candidate):
            return adapter_base_model(candidate)
    return None


def repo_path(path: str) -> str:
    """Set paths as the config writes them: repo-relative when inside the repo."""
    from train.config import resolve_path

    p = Path(path).expanduser()
    p = p.resolve() if p.exists() else resolve_path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def apply_set_overrides(cfg: dict, sets_arg: str | None, variants_arg: str | None) -> None:
    from train.config import resolve_path

    sets = [{"path": s} if isinstance(s, str) else dict(s) for s in cfg["eval"].get("sets") or []]
    if sets_arg:
        configured = {str(resolve_path(s["path"])): s for s in sets}
        sets = []
        for raw in filter(None, (p.strip() for p in sets_arg.split(","))):
            path = repo_path(raw)
            sets.append(dict(configured.get(str(resolve_path(path)), {"path": path, "loglik": True})))
    if variants_arg:
        variants = [v.strip() for v in variants_arg.split(",") if v.strip()]
        for s in sets:
            s["variants"] = variants
    cfg["eval"]["sets"] = sets


def compare_with_base(cfg: dict, summary: dict, base_run: str = "base") -> None:
    from eval.compare import fingerprint_diff, format_compare_table, write_compare
    from eval.runner import load_summary
    from train.config import run_dir

    base_dir = run_dir(cfg, base_run)
    base = load_summary(base_dir)
    if base is None:
        print(f"\nno base eval at {base_dir}; rerun with --with-base for delta-over-base")
        return
    if base.get("fingerprint") != summary["fingerprint"]:
        parts = ", ".join(fingerprint_diff(base, summary)) or "?"
        print(f"\nbase eval fingerprint {base.get('fingerprint')} != {summary['fingerprint']} (differs in: {parts}): evaluate the "
              f"base under the same --config/--set flags, e.g. rerun with --with-base (it re-evaluates {base_dir.name} with "
              "these flags and replaces the base eval made under other settings)")
        return
    cmp = write_compare(run_dir(cfg, summary["run_name"]), base_dir)
    print()
    print(format_compare_table(cmp))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from eval.runner import checkpoint_eval_overrides, debug_suffix, format_summary, run_eval
    from train.config import is_base_run_name, load_config, run_dir
    from train.registry import check_run_name_owner

    model = resolve_model_arg(args.model)
    if args.base and args.run_name not in (None, "base"):
        raise ValueError("--base evals are always stored as 'base' (or base-limit<N> / base-dbg<fp>); drop --run-name")
    settings = [] if args.config or args.base else checkpoint_eval_overrides(model)
    cfg, plain = load_config(args.config, settings + args.overrides), load_config(args.config, settings)
    for c in (cfg, plain):
        if args.base:
            c["model"]["base"] = args.base_model or args.model
        else:
            c["model"]["base"] = args.base_model or infer_base_model(model) or c["model"]["base"]
    apply_set_overrides(cfg, args.sets, args.variants)
    suffix = debug_suffix(cfg, plain, limit=args.limit)
    run_name, base_run = (args.run_name or default_run_name(model, args.base)) + suffix, "base" + suffix
    if not args.base and is_base_run_name(run_name):
        raise ValueError(f"run name {run_name!r} is reserved for the base model's eval: add --base, or pick another --run-name")
    if not args.base:
        check_run_name_owner(cfg, run_name, model)

    if args.with_base and not args.base:
        base_summary = run_eval(cfg["model"]["base"], base_run, cfg, is_base=True, limit=args.limit)
        print(format_summary(base_summary))
        print()
    summary = run_eval(model, run_name, cfg, is_base=args.base, limit=args.limit, force=args.force)
    print(format_summary(summary))
    print(f"results: {run_dir(cfg, run_name)}")
    if args.base and run_name == "base":  # a full (non-debug) baseline: the untouched base is now the standing candidate
        from train.registry import LEADERBOARD_FILE, _write_text, ensure_candidate, render_leaderboard
        from train.config import results_dir

        candidate = ensure_candidate(cfg)
        _write_text(results_dir(cfg) / LEADERBOARD_FILE, render_leaderboard(results_dir(cfg)))
        print(f"CANDIDATE: {candidate['run_name']} ({candidate['model_path']})  -> {results_dir(cfg) / 'CANDIDATE.json'}")
    if not (args.base or args.no_compare or run_name == base_run):
        compare_with_base(cfg, summary, base_run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
