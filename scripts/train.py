#!/usr/bin/env python
"""Train a LoRA/QLoRA run end to end: processed data -> SFT -> merge -> eval vs base -> compare -> promotion gate.

  python scripts/train.py --config configs/run1.yaml
  python scripts/train.py --config configs/run1.yaml --set run_name=run1b --set train.learning_rate=2e-4
  python scripts/train.py --config configs/smoke.yaml               # tiny local plumbing check (.smoke/ roots)
  python scripts/train.py --config configs/run1.yaml --data-only    # build processed rows, print stats, exit
  python scripts/train.py --config configs/run1.yaml --no-eval      # train + merge only
  python scripts/train.py --config configs/run1.yaml --no-merge     # adapter only (implies --no-eval)
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import traceback

from train.config import load_config, results_dir, run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run config (implicitly inherits configs/base.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override (repeatable)")
    ap.add_argument("--no-merge", action="store_true", help="keep only the LoRA adapter (implies --no-eval)")
    ap.add_argument("--no-eval", action="store_true", help="skip base/run eval, compare and the promotion gate")
    ap.add_argument("--data-only", action="store_true", help="build processed training rows, print stats and exit")
    args = ap.parse_args(argv)
    bad = [s for s in args.set if "=" not in s]
    if bad:
        ap.error(f"--set expects KEY=VALUE, got {bad}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.data_only:
        from train.dataset import build_processed

        data = build_processed(cfg)
        for key, path in data.items():
            if key != "stats":
                print(f"{key:16s} {path}")
        print(json.dumps(data.get("stats", {}), ensure_ascii=False, indent=2, default=str))
        return

    from train.pipeline import display_path, run_pipeline

    result = run_pipeline(cfg, do_merge=not args.no_merge, do_eval=not (args.no_eval or args.no_merge))
    adapter, model_dir = result["adapter_dir"], result["model_dir"]
    ckpt = adapter.parent
    print(f"\nRun {result['run_name']}: artifacts")
    print(f"  adapter:        {display_path(adapter)}")
    if model_dir is not None and (model_dir / "config.json").exists():
        print(f"  merged model:   {display_path(model_dir)}")
    elif model_dir is not None:  # registry cleanup (pipeline.keep_merged=candidate_only) removed non-candidate weights
        print(f"  merged model:   removed (not the candidate); rebuild with "
              f"python scripts/merge_and_export.py --adapter {display_path(adapter)} --out {display_path(model_dir)}")
    print(f"  train metrics:  {display_path(ckpt / 'train_metrics.json')}")
    if "compare" in result:
        print(f"  eval + compare: {display_path(run_dir(cfg))}")
        print(f"  leaderboard:    {display_path(results_dir(cfg) / 'LEADERBOARD.md')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("train.py: interrupted")
    except FileNotFoundError as exc:  # missing inputs: the message says what to run, a traceback adds nothing
        sys.exit(f"train.py: FAILED: {exc}")
    except Exception as exc:  # full traceback for debugging, one-line reason last (sweep logs end with it)
        traceback.print_exc()
        sys.exit(f"train.py: FAILED: {type(exc).__name__}: {exc}")
