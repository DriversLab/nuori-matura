#!/usr/bin/env python
"""Budgeted sequential sweep: one `scripts/train.py` subprocess per run (train -> merge -> eval -> compare -> gate).

  python scripts/sweep.py --sweep configs/sweep.yaml --dry-run
  python scripts/sweep.py --sweep configs/sweep.yaml --budget-hours 10 --max-runs 4

Sweep file: {name, base_config, budget_hours, runs: [{name, set: {dotted.key: value}}], grid: {dotted.key: [values]}}.
`grid` is optional; its cartesian product is appended to `runs` with names "g-<key>=<value>_...".
Every run also gets tags=[<sweep name>] and a notes line unless its `set` overrides them, so a promoted sweep run is
tagged with the sweep instead of inheriting the base config's tags (e.g. run1's v0-safe).

Runs already recorded in results/<slug>/runs.jsonl WITH THE SAME resolved config (config_hash) are skipped; a run name
recorded with a different config (edited overrides, grid or base config) makes the sweep refuse to start: give such runs
new names, so a ranking never shows an old result under a name that now means other settings.
Before each launch the remaining budget is compared
with the estimated run duration (mean of this sweep's successful runs; until one succeeds, --est-hours if given, else
no estimate: a run is launched while any budget is left) and the sweep stops launching when the next run would not fit.
A failed run is logged and the sweep continues with the next one.
Logs: results/<slug>/sweeps/<name>/<run>.log   Summary: results/<slug>/sweeps/<name>/sweep_summary.json
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import itertools
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import yaml

from train.config import config_hash, get_dotted, load_config, resolve_path, results_dir

TRAIN_SCRIPT = "scripts/train.py"
SUMMARY_FILE = "sweep_summary.json"
_MISSING = object()

Runner = Callable[[list[str], Path], int]


@dataclass
class PlannedRun:
    name: str
    overrides: dict[str, Any]  # dotted key -> value, excluding run_name
    command: list[str]
    results_dir: Path
    log_path: Path
    config_hash: str  # of the config scripts/train.py resolves from `command` (what record_run stores)


# ----------------------------------------------------------------------------- sweep file -> planned runs


def load_sweep(path: str | Path) -> dict:
    path = resolve_path(path) if not Path(path).exists() else Path(path)
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    for key in ("name", "base_config"):
        if not spec.get(key):
            raise ValueError(f"{path}: sweep needs '{key}'")
    for i, run in enumerate(spec.get("runs") or []):
        if not isinstance(run, dict) or not run.get("name") or not isinstance(run.get("set") or {}, dict):
            raise ValueError(f"{path}: runs[{i}] must be {{name: str, set: {{dotted.key: value}}}}")
    grid = spec.get("grid") or {}
    if not isinstance(grid, dict) or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError(f"{path}: grid must map dotted keys to non-empty lists of values")
    return spec


def _name_value(value: Any) -> str:
    """Compact, quote-free rendering for run names and notes: {canonical:0.45,bare:0.15}, [16,32], 0.0002."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_name_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_name_value(v) for v in value) + "]"
    return str(value)


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product -> [{"name": "g-<key>=<value>_...", "set": {...}}]; keys are shortened to their last
    component unless two grid keys share it."""
    if not grid:
        return []
    keys = list(grid)
    lasts = [k.split(".")[-1] for k in keys]
    labels = lasts if len(set(lasts)) == len(lasts) else keys
    runs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        name = "g-" + "_".join(f"{label}={_name_value(v)}" for label, v in zip(labels, combo))
        runs.append({"name": name, "set": dict(zip(keys, combo))})
    return runs


def format_value(value: Any) -> str:
    """CLI text that train.config.parse_value turns back into `value` (JSON is valid YAML flow syntax)."""
    if isinstance(value, str):
        try:
            if yaml.safe_load(value) == value:
                return value
        except yaml.YAMLError:
            pass
    return json.dumps(value, ensure_ascii=False)


def _validate_keys(base_cfg: dict, overrides: dict[str, Any], run_name: str) -> None:
    unknown = [k for k in overrides if get_dotted(base_cfg, k, _MISSING) is _MISSING]
    if unknown:
        raise ValueError(f"run {run_name!r}: unknown config keys {unknown} (not in the base config; typo?)")


def config_arg(path: str | Path) -> str:
    """Config path valid from the repo root (where runs execute): repo-relative when inside the repo, else absolute."""
    p = Path(path).expanduser()
    p = p.resolve() if p.exists() else resolve_path(p)
    if not p.exists():
        raise FileNotFoundError(f"base_config not found: {path}")
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def plan_runs(spec: dict) -> list[PlannedRun]:
    """Validated runs in launch order (explicit runs, then grid). Raises on duplicate names or unknown keys."""
    sweep_name, base_config = spec["name"], config_arg(spec["base_config"])
    base_cfg = load_config(base_config)
    entries = list(spec.get("runs") or []) + expand_grid(spec.get("grid") or {})
    names = [e["name"] for e in entries]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate run names in sweep {sweep_name!r}: {dupes}")

    planned = []
    for entry in entries:
        name, sets = str(entry["name"]), dict(entry.get("set") or {})
        _validate_keys(base_cfg, sets, name)
        summary = ", ".join(f"{k}={_name_value(v)}" for k, v in sets.items()) or "base config"
        overrides = {"tags": [sweep_name], "notes": f"sweep {sweep_name}: {summary}", **sets}
        command = [sys.executable, TRAIN_SCRIPT, "--config", base_config, "--set", f"run_name={format_value(name)}"]
        for key, value in overrides.items():
            command += ["--set", f"{key}={format_value(value)}"]
        run_cfg = load_config(base_config, [a for prev, a in zip(command, command[1:]) if prev == "--set"])
        rdir = results_dir(run_cfg)
        planned.append(PlannedRun(name, overrides, command, rdir, rdir / "sweeps" / sweep_name / f"{name}.log", config_hash(run_cfg)))
    return planned


# ----------------------------------------------------------------------------- execution


def recorded_rows(rdir: Path) -> dict[str, dict]:
    """run_name -> latest "run" row of results/<slug>/runs.jsonl."""
    from train.registry import load_runs

    return {r["run_name"]: r for r in load_runs(rdir) if r.get("event") == "run"}


def recorded_status(run: PlannedRun) -> str | None:
    """None = not recorded; "same" = recorded with this config (or a legacy row without a hash); "changed" = the name
    was recorded with a different config."""
    row = recorded_rows(run.results_dir).get(run.name)
    if row is None:
        return None
    return "same" if row.get("config_hash") in (None, run.config_hash) else "changed"


def check_changed_runs(runs: list[PlannedRun]) -> None:
    changed = [r.name for r in runs if recorded_status(r) == "changed"]
    if changed:
        raise ValueError(f"run(s) {changed} were already recorded with a different config (overrides, grid or base config "
                         "edited since); give them new names so results are never shown under a name that now means other settings")


def run_logged(command: list[str], log_path: Path) -> int:
    """Run command from the repo root, streaming stdout+stderr to the console and to log_path. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {shlex.join(command)}\n")
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()  # keep the console live when the sweep itself is piped (e.g. | tee)
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise


def estimate_hours(durations_s: list[float], initial_hours: float | None) -> float:
    """Mean duration of successful runs; before the first success the initial guess (None -> 0: nothing measured yet)."""
    return fmean(durations_s) / 3600 if durations_s else (initial_hours or 0.0)


def run_sweep(
    runs: list[PlannedRun],
    *,
    budget_hours: float,
    est_hours: float | None,
    max_runs: int | None = None,
    runner: Runner = run_logged,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict]:
    """Launch runs in order within the budget. Returns one status record per planned run:
    status in ok | failed | skipped (already recorded) | not-run (budget / max-runs / interrupted) | interrupted."""
    start = clock()
    durations: list[float] = []
    launched = 0
    stop_reason: str | None = None
    records = []
    for run in runs:
        rec: dict[str, Any] = {"name": run.name, "overrides": run.overrides, "command": shlex.join(run.command),
                               "log": str(run.log_path), "returncode": None, "duration_s": None}
        records.append(rec)
        if recorded_status(run) == "same":
            rec["status"] = "skipped"
            continue
        remaining = budget_hours - (clock() - start) / 3600
        estimate = estimate_hours(durations, est_hours)
        if stop_reason is None and max_runs is not None and launched >= max_runs:
            stop_reason = f"max-runs {max_runs} reached"
        if stop_reason is None and (remaining <= 0 or remaining < estimate):
            stop_reason = f"budget: {remaining:.2f} h left < estimated {estimate:.2f} h per run"
        if stop_reason is not None:
            rec["status"] = f"not-run ({stop_reason})"
            continue

        print(f"\n=== [{run.name}] launching ({remaining:.2f} h budget left, est {estimate:.2f} h)  log: {run.log_path}", flush=True)
        launched += 1
        t0 = clock()
        try:
            code = runner(run.command, run.log_path)
        except KeyboardInterrupt:
            rec.update(status="interrupted", duration_s=round(clock() - t0, 1))
            stop_reason = "interrupted"
            continue
        rec.update(returncode=code, duration_s=round(clock() - t0, 1), status="ok" if code == 0 else "failed")
        if code == 0:
            durations.append(clock() - t0)
        print(f"=== [{run.name}] {rec['status']} (exit {code}) after {rec['duration_s'] / 3600:.2f} h", flush=True)
    return records


# ----------------------------------------------------------------------------- reporting


def rank(runs: list[PlannedRun], records: list[dict]) -> list[dict]:
    """Latest recorded row per sweep run, ranked by headline delta_points (unrecorded runs last)."""
    status = {r["name"]: r["status"] for r in records}
    rows_by_dir: dict[Path, dict[str, dict]] = {}
    ranking = []
    for run in runs:
        rows = rows_by_dir.setdefault(run.results_dir, recorded_rows(run.results_dir))
        row = rows.get(run.name) or {}
        ranking.append({
            "name": run.name,
            "status": status.get(run.name, "planned"),
            "delta_points": row.get("delta_points"),
            "score_pct": row.get("score_pct"),
            "ci95": row.get("ci95"),
            "general_nll_rel_change": row.get("general_nll_rel_change"),
            "gate_passed": row.get("gate_passed"),
            "gate_failed_rules": row.get("gate_failed_rules"),
            "promoted": row.get("promoted"),
        })
    ranking.sort(key=lambda r: (r["delta_points"] is None, -(r["delta_points"] or 0.0)))
    return ranking


def _fmt(x: Any, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def format_ranking(ranking: list[dict]) -> str:
    lines = [f"{'#':>2}  {'run':<24}{'status':<12}{'delta':>8}{'score%':>8}  {'CI95':<16}{'gNLL':>8}  gate"]
    for i, r in enumerate(ranking, 1):
        ci = "-" if not r["ci95"] else f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]"
        gnll = "-" if r["general_nll_rel_change"] is None else f"{100 * r['general_nll_rel_change']:+.1f}%"
        if r["gate_passed"] is None:
            gate = "-"
        else:
            gate = "PASS" if r["gate_passed"] else f"FAIL [{','.join(r['gate_failed_rules'] or [])}]"
        status = r["status"].split(" ")[0]
        lines.append(f"{i:>2}  {r['name']:<24}{status:<12}{_fmt(r['delta_points'], '+.2f'):>8}"
                     f"{_fmt(r['score_pct'], '.2f'):>8}  {ci:<16}{gnll:>8}  {gate}")
    return "\n".join(lines)


def current_candidates(runs: list[PlannedRun]) -> dict[str, dict | None]:
    from train.registry import load_candidate

    return {str(d): load_candidate(d) for d in dict.fromkeys(r.results_dir for r in runs)}


def print_dry_run(spec: dict, runs: list[PlannedRun], *, budget_hours: float, est_hours: float | None, max_runs: int | None) -> None:
    to_launch = 0
    for run in runs:
        if recorded_status(run) == "same":
            print(f"# {run.name}: skip (already recorded in {run.results_dir / 'runs.jsonl'})")
            continue
        to_launch += 1
        print(f"# {run.name}  (log: {run.log_path})")
        print(shlex.join(run.command))
    capped = min(to_launch, max_runs) if max_runs is not None else to_launch
    print(f"\n{spec['name']}: {len(runs)} planned, {to_launch} to launch"
          f"{f' (capped at {capped} by --max-runs)' if capped != to_launch else ''}; "
          + (f"~{capped * est_hours:.1f} h at {est_hours:.2f} h/run vs budget {budget_hours:.1f} h" if est_hours
             else f"budget {budget_hours:.1f} h (pass --est-hours for a time estimate)"))


# ----------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, help="sweep yaml (e.g. configs/sweep.yaml)")
    ap.add_argument("--budget-hours", type=float, help="wall-clock budget (default: budget_hours from the sweep file)")
    ap.add_argument("--est-hours", type=float,
                    help="per-run duration estimate until a run succeeds (default: none, launch while budget is left)")
    ap.add_argument("--max-runs", type=int, help="launch at most N runs (skipped runs do not count)")
    ap.add_argument("--dry-run", action="store_true", help="print the planned commands and exit")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, *, runner: Runner = run_logged) -> int:
    args = parse_args(argv)
    spec = load_sweep(args.sweep)
    budget = args.budget_hours if args.budget_hours is not None else spec.get("budget_hours")
    if budget is None:
        raise ValueError("no budget: set budget_hours in the sweep file or pass --budget-hours")
    runs = plan_runs(spec)
    check_changed_runs(runs)
    if args.dry_run:
        print_dry_run(spec, runs, budget_hours=float(budget), est_hours=args.est_hours, max_runs=args.max_runs)
        return 0

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.monotonic()
    records = run_sweep(runs, budget_hours=float(budget), est_hours=args.est_hours, max_runs=args.max_runs, runner=runner)
    ranking = rank(runs, records)
    candidates = current_candidates(runs)

    print(f"\nSweep {spec['name']}: ranking by headline delta_points over base")
    print(format_ranking(ranking))
    for rdir, cand in candidates.items():
        cand_txt = "none" if cand is None else f"{cand['run_name']} ({cand['delta_points']:+.2f} pts)  model {cand['model_path']}"
        print(f"Current candidate [{rdir}]: {cand_txt}")

    summary_path = results_dir(load_config(config_arg(spec["base_config"]))) / "sweeps" / spec["name"] / SUMMARY_FILE
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "sweep": spec["name"],
        "base_config": spec["base_config"],
        "budget_hours": float(budget),
        "est_hours_initial": args.est_hours,
        "max_runs": args.max_runs,
        "started_at": started,
        "elapsed_h": round((time.monotonic() - t0) / 3600, 3),
        "runs": records,
        "ranking": ranking,
        "candidates": candidates,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"summary: {summary_path}")

    failed = [r["name"] for r in records if r["status"] in ("failed", "interrupted")]
    if failed:
        print(f"sweep.py: {len(failed)} run(s) failed or were interrupted: {', '.join(failed)} (see their logs)", file=sys.stderr)
        return 130 if any(r["status"] == "interrupted" for r in records) else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"sweep.py: FAILED: {type(exc).__name__}: {exc}")
