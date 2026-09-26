#!/usr/bin/env python
"""Show the current CANDIDATE, leaderboard and tags for every base model under results_root (read-only).

  python scripts/status.py                      # all base slugs under paths.results_root
  python scripts/status.py --config configs/run1.yaml   # only that config's base model
  python scripts/status.py --json               # machine-readable
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

from train.config import get_dotted, load_config, resolve_path, results_dir
from train.registry import CANDIDATE_FILE, RUNS_FILE, format_leaderboard_table, leaderboard_rows, load_candidate, load_runs, load_tags


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="only show this config's base model (default: every base slug under results_root)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override, e.g. paths.results_root=.smoke/results")
    ap.add_argument("--json", action="store_true", help="print JSON instead of tables")
    return ap.parse_args(argv)


def _slug_dirs(cfg: dict, only_config: bool) -> list[pathlib.Path]:
    if only_config:
        return [results_dir(cfg)]
    root = resolve_path(get_dotted(cfg, "paths.results_root", "results"))
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and ((d / CANDIDATE_FILE).exists() or (d / RUNS_FILE).exists()))


def _status(rdir: pathlib.Path) -> dict:
    return {
        "results_dir": str(rdir),
        "candidate": load_candidate(rdir),
        "leaderboard": leaderboard_rows(rdir),
        "tags": load_tags(rdir),
        "n_records": len(load_runs(rdir)),
    }


def _print_text(slug: str, st: dict) -> None:
    print(f"== {slug}  ({st['results_dir']})")
    cand = st["candidate"]
    if cand is None:
        print("CANDIDATE: none recorded yet (the untouched base model is the fallback)")
    else:
        score = "n/a" if cand.get("score_pct") is None else f"{cand['score_pct']:.2f}%"
        flags = " [forced]" if cand.get("forced") else ""
        print(f"CANDIDATE: {cand['run_name']}{flags}  delta {cand['delta_points']:+.2f} pts  score {score}  model {cand['model_path']}")
        print(f"  promoted {cand['promoted_at']}: {cand['reason']}")
        for warning in cand.get("warnings") or []:
            print(f"  WARNING: {warning}")
    print(format_leaderboard_table(st["leaderboard"]))
    tags = ", ".join(f"{t} -> {r}" for t, r in sorted(st["tags"].items())) or "none"
    print(f"tags: {tags}   records: {st['n_records']}\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config, args.set)
    dirs = _slug_dirs(cfg, only_config=bool(args.config))
    statuses = {d.name: _status(d) for d in dirs}
    if args.json:
        print(json.dumps(statuses, ensure_ascii=False, indent=2))
    elif not statuses:
        print(f"no results under {resolve_path(get_dotted(cfg, 'paths.results_root', 'results'))}")
    else:
        for slug, st in statuses.items():
            _print_text(slug, st)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
