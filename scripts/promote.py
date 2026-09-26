#!/usr/bin/env python
"""Manually set the submission CANDIDATE (logged in runs.jsonl as a manual_promote event).

  python scripts/promote.py --run run2 --reason "better on bare variant" [--tag best --tag v1] [--config configs/run2.yaml]
  python scripts/promote.py --run run3 --reason "gate NLL threshold too strict" --force   # run failed the gate
  python scripts/promote.py --run base --reason "roll back to the untouched base model"
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from train.config import get_dotted, load_config, resolve_path, results_dir
from train.registry import BASE_RUN, RUNS_FILE, load_runs, promote


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="recorded run name to promote, or 'base'")
    ap.add_argument("--reason", required=True, help="why (stored in CANDIDATE.json and runs.jsonl)")
    ap.add_argument("--tag", action="append", default=[], help="tag to point at this run in tags.json, repeatable")
    ap.add_argument("--force", action="store_true", help="promote even though the run's latest record failed the gate")
    ap.add_argument("--config", help="config selecting the base model / paths (default: configs/base.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override, repeatable")
    return ap.parse_args(argv)


def _recorded(rdir: pathlib.Path, run_name: str) -> bool:
    return any(r.get("event") == "run" and r.get("run_name") == run_name for r in load_runs(rdir))


def _other_slug_hint(cfg: dict, run_name: str) -> None:
    """Without --config the registry is configs/base.yaml's base model: point at the slug(s) that recorded the run
    (before promote() touches the wrong slug's registry)."""
    if run_name == BASE_RUN or _recorded(results_dir(cfg), run_name):
        return
    root = resolve_path(get_dotted(cfg, "paths.results_root", "results"))
    slugs = sorted(d.name for d in root.iterdir() if (d / RUNS_FILE).exists() and _recorded(d, run_name)) if root.is_dir() else []
    if slugs:
        raise ValueError(f"run {run_name!r} has no record under {results_dir(cfg)} (default config configs/base.yaml) but is "
                         f"recorded for base model(s) {', '.join(slugs)}; pass --config with that base model's config")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config, args.set)
    if not args.config:
        _other_slug_hint(cfg, args.run)
    cand = promote(cfg, args.run, reason=args.reason, tags=args.tag, force=args.force)
    forced = " (FORCED past a failed gate)" if cand["forced"] else ""
    score = "n/a" if cand.get("score_pct") is None else f"{cand['score_pct']:.2f}%"
    print(f"CANDIDATE -> {cand['run_name']}{forced}  delta {cand['delta_points']:+.2f} pts  score {score}  model {cand['model_path']}")
    if cand["tags"]:
        print(f"tags: {', '.join(cand['tags'])}")
    for warning in cand["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
