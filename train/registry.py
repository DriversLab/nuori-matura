"""Run registry: promotion gate, CANDIDATE.json, runs.jsonl, tags.json, LEADERBOARD.md, checkpoint cleanup.

Invariants (docs/ARCHITECTURE.md, principle 4 and #gate):
  * <results_dir>/CANDIDATE.json exists after any registry call; it starts as the untouched base model.
  * The candidate changes only via evaluate_gate (record_run: promotion, or demotion to base when the candidate
    itself fails re-evaluation) or a logged manual promote().
  * runs.jsonl is append-only: one JSON object per line ("event": "run" | "manual_promote").
  * <checkpoints_root>/CANDIDATE-<base_slug> is a relative symlink to that base model's candidate checkpoint dir,
    absent for the base model (checkpoint dirs are shared by every base model, links are not).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from train.config import ROOT, candidate_link_name, checkpoint_dir, checkpoints_root, config_hash, get_dotted, load_config, resolve_path
from train.config import results_dir as cfg_results_dir

BASE_RUN = "base"
RUNS_FILE = "runs.jsonl"
CANDIDATE_FILE = "CANDIDATE.json"
TAGS_FILE = "tags.json"
LEADERBOARD_FILE = "LEADERBOARD.md"
CANDIDATE_LINK = "CANDIDATE"  # prefix of the per-base-model links (CANDIDATE-<base_slug>)
GATE_KEYS = (
    "min_delta_points",
    "max_parse_fail_rate_increase",
    "max_general_nll_increase",
    "max_variant_regression_points",
    "max_secondary_score_drop_pct",
    "min_net_score_pct",
    "max_loglik_acc_drop_pct",
    "must_beat_current_candidate",
    "min_improvement_points",
    "max_lenient_regression_points",
)
# Rule 9 (organizer first-token letter accuracy) is additive: older gate configs (a checkpoint's resolved_config.yaml
# written before it existed) keep working with this default instead of failing _gate_values.
DEFAULT_MAX_ORGANIZER_ACC_DROP_PCT = 3.0
# Failed rules that make a comparison non-comparable rather than a verdict on the model: a re-recorded candidate whose
# comparison fails any of these keeps its place; otherwise any failed rule demotes it to the base model.
NON_VERDICT_RULES = frozenset({"fp", "limit", "scope"})
# Files of a merged, eval-ready model at a checkpoint root; rebuildable from adapter/ via merge_and_export.
MERGED_PATTERNS = (
    "model*.safetensors",
    "model.safetensors.index.json",
    "*.bin",
    "pytorch_model.bin.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
)
_EPS = 1e-9  # float slack for threshold comparisons (rates/points are rounded upstream)


# ----------------------------------------------------------------------------- io helpers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _append_row(results_dir: Path, row: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / RUNS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def _locked(results_dir: Path) -> Iterator[None]:
    """Serialize registry mutations (read candidate -> gate -> promote) across concurrent processes."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / ".registry.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_runs(results_dir: Path) -> list[dict]:
    path = Path(results_dir) / RUNS_FILE
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: corrupt runs.jsonl line: {exc}") from exc
    return rows


def load_candidate(results_dir: Path) -> dict | None:
    return _read_json(Path(results_dir) / CANDIDATE_FILE)


def load_tags(results_dir: Path) -> dict[str, str]:
    return _read_json(Path(results_dir) / TAGS_FILE) or {}


def _base_summary(results_dir: Path) -> dict | None:
    """runs/base/summary.json when it holds a base-model eval (never take the base's score from another model's eval)."""
    summary = _read_json(Path(results_dir) / "runs" / BASE_RUN / "summary.json")
    return summary if summary and summary.get("is_base") is True else None


def _latest_run_row(results_dir: Path, run_name: str) -> dict | None:
    rows = [r for r in load_runs(results_dir) if r.get("event") == "run" and r.get("run_name") == run_name]
    return rows[-1] if rows else None


def _local_path(p: str | Path) -> Path:
    """Absolute path for a model/checkpoint path recorded by the registry (relative paths are repo-relative)."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else resolve_path(path)


def _portable(p: str | Path | None) -> str | None:
    """Repo-relative path when inside the repo, else absolute: results/ is shared, so rows and CANDIDATE.json must not
    carry one machine's absolute checkout path."""
    if p is None:
        return None
    path = _local_path(p).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _checkpoint_subdir(ck_root: Path, model_path: str | Path | None) -> str | None:
    """Name of the top-level run dir under checkpoints_root that contains model_path, else None."""
    if not model_path:
        return None
    try:
        rel = _local_path(model_path).resolve().relative_to(Path(ck_root).resolve())
    except ValueError:
        return None
    if not rel.parts or rel.parts[0].startswith(CANDIDATE_LINK):
        return None
    return rel.parts[0]


def _same_model(ck_root: Path, a: str | Path | None, b: str | Path | None) -> bool:
    """Same checkpoint run dir (its merged root or its adapter/), else the same portable path / hub id."""
    if not a or not b:
        return False
    sub_a, sub_b = _checkpoint_subdir(ck_root, a), _checkpoint_subdir(ck_root, b)
    if sub_a or sub_b:
        return sub_a == sub_b
    return _portable(a) == _portable(b)


def _owner_conflict(candidate: dict | None, ck_root: Path, run_name: str, model_path: str | Path) -> str | None:
    if not candidate or candidate["run_name"] != run_name or candidate["run_name"] == BASE_RUN:
        return None
    if _same_model(ck_root, candidate["model_path"], model_path):
        return None
    return (f"run name {run_name!r} belongs to the current candidate ({candidate['model_path']}), but this eval is of "
            f"{_portable(model_path)}; evaluate/record it under a different run name (--run-name)")


def check_run_name_owner(cfg: dict, run_name: str, model_path: str | Path) -> None:
    """Raise ValueError when run_name is this base model's current candidate but model_path is a different model:
    evaluating it under that name would overwrite the candidate's eval and could replace the candidate."""
    conflict = _owner_conflict(load_candidate(cfg_results_dir(cfg)), checkpoints_root(cfg), run_name, model_path)
    if conflict:
        raise ValueError(conflict)


# ----------------------------------------------------------------------------- candidate


def _base_candidate(model_path: str, summary: dict | None, *, reason: str, tags: list[str]) -> dict:
    return {
        "run_name": BASE_RUN,
        "model_path": model_path,
        "delta_points": 0.0,
        "score_pct": summary["headline"]["score_pct"] if summary else None,
        "promoted_at": _now(),
        "tags": tags,
        "reason": reason,
        "fingerprint": summary.get("fingerprint") if summary else None,
    }


def ensure_candidate(cfg: dict) -> dict:
    """Create CANDIDATE.json for the untouched base model if missing; refresh the base's score when it is the candidate."""
    rdir = cfg_results_dir(cfg)
    path = rdir / CANDIDATE_FILE
    candidate = _read_json(path)
    summary = _base_summary(rdir)
    if candidate is None:
        candidate = _base_candidate(
            cfg["model"]["base"], summary, reason="untouched base model (standing fallback)", tags=["base-fallback"]
        )
        _write_json(path, candidate)
    elif candidate["run_name"] == BASE_RUN and summary is not None:
        refreshed = {**candidate, "score_pct": summary["headline"]["score_pct"], "fingerprint": summary.get("fingerprint")}
        if refreshed != candidate:
            _write_json(path, refreshed)
            candidate = refreshed
    return candidate


def candidate_link(ck_root: Path, results_dir: Path) -> Path:
    """<checkpoints_root>/CANDIDATE-<base_slug> for the base model whose registry is results_dir."""
    return Path(ck_root) / candidate_link_name(Path(results_dir).name)


def _sync_candidate_link(ck_root: Path, results_dir: Path, candidate: dict) -> None:
    link = candidate_link(ck_root, results_dir)
    if link.exists() and not link.is_symlink():
        raise FileExistsError(f"{link} exists and is not a symlink; refusing to replace it")
    target = None if candidate["run_name"] == BASE_RUN else _checkpoint_subdir(ck_root, candidate["model_path"])
    if target is None:
        if link.is_symlink():
            link.unlink()
        return
    tmp = link.with_name(f".{link.name}.tmp")
    if tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)  # relative: sibling dir name inside checkpoints_root
    os.replace(tmp, link)


def _install_candidate(results_dir: Path, ck_root: Path | None, candidate: dict) -> None:
    if ck_root is not None:  # first: a blocked link (FileExistsError) must not leave a half-installed candidate
        _sync_candidate_link(ck_root, results_dir, candidate)
    _write_json(results_dir / CANDIDATE_FILE, candidate)
    if candidate["tags"]:
        tags = load_tags(results_dir)
        tags.update({t: candidate["run_name"] for t in candidate["tags"]})
        _write_json(results_dir / TAGS_FILE, tags)


# ----------------------------------------------------------------------------- gate


def _gate_values(gate_cfg: dict) -> dict:
    missing = [k for k in GATE_KEYS if k not in gate_cfg]
    if missing:
        raise ValueError(f"gate config missing keys: {missing} (see configs/base.yaml gate:)")
    return {k: gate_cfg[k] for k in GATE_KEYS}


def reference_eval_scope() -> dict[str, dict]:
    """{set name: {"variants", "loglik"}} of configs/base.yaml eval.sets (optional sets only when their file exists):
    the eval a comparison must cover to be auto-promoted (gate [scope]).

    A set whose spec asks for no variants and no loglik (organizer-only, e.g. the prawko sets) requires nothing: it is
    judged by gate rule 9, which is skipped when it was not measured, so it never blocks a promotion."""
    from eval.runner import eval_sets_from_cfg  # lazy: heavy (torch)

    cfg = load_config(None)
    specs = {}
    for entry in get_dotted(cfg, "eval.sets") or []:
        spec = {"path": entry} if isinstance(entry, str) else dict(entry)
        specs[spec.get("name") or Path(str(spec["path"])).stem] = spec
    scope = {}
    for s in eval_sets_from_cfg(cfg):
        spec = specs.get(s["name"], {})
        organizer_only = spec.get("variants") == [] and not spec.get("loglik")  # explicit "variants: []" in the spec
        scope[s["name"]] = {"variants": [] if organizer_only else list(s["variants"]), "loglik": s["loglik"]}
    return scope


def _scope_gaps(cmp: dict, required: dict[str, dict]) -> list[str]:
    """Reference sets / variants / loglik that the comparison did not cover."""
    covered = {s["name"]: s for s in cmp.get("eval_sets") or []} or {
        name: {"variants": list(d["variants"]), "loglik": d.get("loglik") is not None} for name, d in cmp["sets"].items()
    }
    gaps: list[str] = []
    for name, spec in required.items():
        if not spec["variants"] and not spec["loglik"]:  # organizer-only set: nothing [scope] can require
            continue
        have = covered.get(name)
        if have is None:
            gaps.append(name)
            continue
        gaps += [f"{name}/{v}" for v in spec["variants"] if v not in have["variants"]]
        if spec["loglik"] and not have["loglik"]:
            gaps.append(f"{name}/loglik")
    return gaps


def evaluate_gate(
    cmp: dict, candidate: dict | None, gate_cfg: dict, *, required_scope: dict[str, dict] | None = None
) -> tuple[bool, list[str]]:
    """Promotion gate (docs/ARCHITECTURE.md#gate rules 1-9 and 5b) plus [fp]: a comparison explicitly made despite an
    eval fingerprint mismatch never auto-promotes, [limit]: neither does one on a debug eval capped by eval.limit, and
    [scope] (when required_scope is given, see reference_eval_scope): neither does one that skipped reference eval
    sets, variants or loglik. candidate None means the untouched base model (delta 0).

    Rule 7 passes when the run IS the current candidate (re-recorded; record_run refuses a different model under the
    candidate's run name), and fails when a non-base candidate was measured under another eval fingerprint.
    Rule 9 (organizer first-token letter accuracy, gate.max_organizer_acc_drop_pct, default
    DEFAULT_MAX_ORGANIZER_ACC_DROP_PCT) yields no line at all for sets without an organizer block on both sides.
    Returns (passed, reasons); every check yields one "PASS [rule] ..." or "FAIL [rule] ..." line.
    """
    g = _gate_values(gate_cfg)
    candidate = candidate or {"run_name": BASE_RUN, "delta_points": 0.0}
    h, pset, pvar = cmp["headline"], cmp["primary_set"], cmp["primary_variant"]
    sets = cmp["sets"]
    checks: list[tuple[bool, str]] = []

    fp_ok = cmp.get("fingerprint_match", True) is not False
    checks.append((fp_ok, "[fp] eval fingerprint matches base" if fp_ok else
                   f"[fp] eval fingerprint differs from base ({', '.join(cmp.get('fingerprint_diff') or {}) or '?'})"))

    limit = cmp.get("eval_limit")
    checks.append((limit is None, "[limit] no eval.limit item cap" if limit is None else
                   f"[limit] debug eval capped at {limit} items per set: never auto-promoted"))

    if required_scope is not None:
        gaps = _scope_gaps(cmp, required_scope)
        checks.append((not gaps, "[scope] covers every configs/base.yaml eval set, variant and loglik" if not gaps else
                       f"[scope] eval narrower than configs/base.yaml eval.sets, never auto-promoted (not compared: {', '.join(gaps)})"))

    d, min_d = h["delta_points"], g["min_delta_points"]
    ok = d >= min_d - _EPS
    checks.append((ok, f"[1] headline delta {d:+.2f} pts {'>=' if ok else '<'} min {min_d:+.2f}"))

    run_pf, base_pf, inc = h["run_parse_fail_rate"], h["base_parse_fail_rate"], g["max_parse_fail_rate_increase"]
    ok = run_pf <= base_pf + inc + _EPS
    checks.append((ok, f"[2] parse-fail rate {run_pf:.2%} {'<=' if ok else '>'} base {base_pf:.2%} + allowed {inc:.2%}"))

    general, max_nll = cmp.get("general_heldout"), g["max_general_nll_increase"]
    if general is None:
        checks.append((False, "[3] general heldout NLL missing (eval.general_loss disabled or not evaluated)"))
    else:
        rel = general["rel_change"]
        ok = rel <= max_nll + _EPS
        checks.append((ok, f"[3] general NLL relative change {rel:+.2%} {'<=' if ok else '>'} max {max_nll:+.2%}"))

    max_reg = g["max_variant_regression_points"]
    for variant, v in sets[pset]["variants"].items():
        ok = v["delta_points"] >= -max_reg - _EPS
        checks.append((ok, f"[4] {pset}/{variant} delta {v['delta_points']:+.2f} pts {'>=' if ok else '<'} {-max_reg:+.2f}"))

    max_drop = g["max_secondary_score_drop_pct"]
    secondary = [s for s in sets if s != pset]
    if not secondary:
        checks.append((True, "[5] no secondary eval sets"))
    for set_name in secondary:
        v = sets[set_name]["variants"].get(pvar)
        if v is None:
            checks.append((True, f"[5] {set_name}: no {pvar} variant evaluated, nothing to check"))
            continue
        ok = v["delta_score_pct"] >= -max_drop - _EPS
        checks.append((ok, f"[5] {set_name}/{pvar} delta {v['delta_score_pct']:+.2f} pp {'>=' if ok else '<'} {-max_drop:+.2f}"))

    drops = sum(min(0.0, sets[s]["variants"][pvar]["delta_score_pct"]) for s in secondary if pvar in sets[s]["variants"])
    net, min_net = h["delta_score_pct"] + drops, g["min_net_score_pct"]
    ok = net >= min_net - _EPS
    checks.append((ok, f"[5b] net score {net:+.2f} pp (primary {h['delta_score_pct']:+.2f} + secondary drops {drops:+.2f}) "
                       f"{'>=' if ok else '<'} {min_net:+.2f}"))

    max_ll = g["max_loglik_acc_drop_pct"]
    loglik = {s: sd["loglik"] for s, sd in sets.items() if sd.get("loglik")}
    if not loglik:
        checks.append((True, "[6] no loglik results"))
    for set_name, ll in loglik.items():
        ok = ll["delta_acc_pct"] >= -max_ll - _EPS
        checks.append((ok, f"[6] {set_name} loglik acc delta {ll['delta_acc_pct']:+.2f} pp {'>=' if ok else '<'} {-max_ll:+.2f}"))

    checks.append(_rule7(cmp, candidate, g))

    max_len = g["max_lenient_regression_points"]
    dl = h["delta_lenient_points"]
    ok = dl >= -max_len - _EPS
    checks.append((ok, f"[8] {pset}/{pvar} lenient (format-independent) delta {dl:+.2f} pts {'>=' if ok else '<'} {-max_len:+.2f}"))
    for set_name in secondary:
        v = sets[set_name]["variants"].get(pvar)
        if v is None or not v.get("max_points"):
            continue
        dl_pct = 100.0 * v["delta_lenient_points"] / v["max_points"]
        ok = dl_pct >= -max_drop - _EPS
        checks.append((ok, f"[8] {set_name}/{pvar} lenient delta {dl_pct:+.2f} pp {'>=' if ok else '<'} {-max_drop:+.2f}"))

    max_org = gate_cfg.get("max_organizer_acc_drop_pct", DEFAULT_MAX_ORGANIZER_ACC_DROP_PCT)
    for set_name, org in ((s, sd["organizer"]) for s, sd in sets.items() if sd.get("organizer")):
        ok = org["delta_acc_pct"] >= -max_org - _EPS
        checks.append((ok, f"[9] {set_name} organizer (first-token letter) acc delta {org['delta_acc_pct']:+.2f} pp "
                           f"{'>=' if ok else '<'} {-max_org:+.2f}"))

    return all(ok for ok, _ in checks), [f"{'PASS' if ok else 'FAIL'} {msg}" for ok, msg in checks]


def _rule7(cmp: dict, candidate: dict, g: dict) -> tuple[bool, str]:
    if not g["must_beat_current_candidate"]:
        return True, "[7] must_beat_current_candidate disabled"
    name, d = candidate["run_name"], cmp["headline"]["delta_points"]
    if name == cmp["run_name"]:
        return True, f"[7] {name} is the current candidate (re-recorded)"
    cd, cfp, rfp = candidate["delta_points"], candidate.get("fingerprint"), cmp.get("fingerprint")
    if name != BASE_RUN and cfp and rfp and cfp != rfp:  # the base's delta is 0 under any fingerprint
        model = candidate.get("model_path") or name
        return False, (f"[7] candidate {name} (delta {cd:+.2f}) was measured under eval fingerprint {cfp}, this run under "
                       f"{rfp}: deltas are not comparable; re-evaluate the candidate first: "
                       f"python scripts/compare.py --model {model} --run-name {name}")
    margin = g["min_improvement_points"]
    ok = d > cd + _EPS and d >= cd + margin - _EPS
    if margin > 0:
        return ok, f"[7] delta {d:+.2f} {'>=' if ok else '<'} candidate {name} delta {cd:+.2f} + min improvement {margin:.2f}"
    return ok, f"[7] delta {d:+.2f} {'>' if ok else '<='} candidate {name} delta {cd:+.2f}"


def failed_rules(reasons: list[str]) -> list[str]:
    return [m.group(1) for r in reasons if (m := re.match(r"FAIL \[(\w+)\]", r))]


# ----------------------------------------------------------------------------- run rows


def _artifact_dirs(cfg: dict, run_name: str, model_path: str | Path | None, adapter_path: str | Path | None) -> list[Path]:
    """Dirs that may hold the run's resolved_config.yaml / train_metrics.json (checkpoint root of the run)."""
    dirs: list[Path] = []
    if model_path:
        p = _local_path(model_path)
        dirs += [p, p.parent] if (p / "adapter_config.json").exists() else [p]
    if adapter_path:
        dirs.append(_local_path(adapter_path).parent)
    dirs.append(checkpoint_dir(cfg, run_name))
    return [d for d in dict.fromkeys(dirs) if d.is_dir()]


def _run_config(cfg: dict, run_name: str, dirs: list[Path]) -> dict | None:
    """Config the run was trained with: cfg when it names this run, else the checkpoint's resolved_config.yaml."""
    if cfg.get("run_name") == run_name:
        return cfg
    for d in dirs:
        if (d / "resolved_config.yaml").exists():
            with open(d / "resolved_config.yaml", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return None


def _counts_only(obj: Any) -> Any:
    """Keep nested numeric counts/thresholds of a report; drop examples and text."""
    if isinstance(obj, dict):
        kept = {k: _counts_only(v) for k, v in obj.items()}
        return {k: v for k, v in kept.items() if v not in (None, {})}
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return obj
    return None


def _dedup_counts(cfg: dict) -> dict | None:
    report_path = resolve_path(get_dotted(cfg, "data.synthetic", "data/synthetic/clean.jsonl")).parent / "dedup_report.json"
    report = _read_json(report_path)
    return _counts_only(report) if report else None


def _hparams(run_cfg: dict | None, train_metrics: dict | None) -> dict | None:
    if run_cfg is None and train_metrics is None:
        return None
    c = run_cfg or {}
    return {
        "lr": get_dotted(c, "train.learning_rate"),
        "r": get_dotted(c, "lora.r"),
        "alpha": get_dotted(c, "lora.alpha"),
        "epochs": get_dotted(c, "train.epochs"),
        "general_ratio": get_dotted(c, "data.general_ratio"),
        "general_ratio_trained": (train_metrics or {}).get("general_ratio_trained"),
        "variant_mix": get_dotted(c, "data.variant_mix"),
        "n_train": (train_metrics or {}).get("n_train"),
    }


def _train_summary(train_metrics: dict | None) -> dict | None:
    if not train_metrics:
        return None
    keys = ("n_train", "dropped_too_long", "dropped_too_long_by_split", "dropped_too_long_by_kind", "loss_tokens",
            "runtime_s", "stage_seconds", "device", "quantization", "dtype", "step0")
    epochs = train_metrics.get("epochs") or []
    return {**{k: train_metrics.get(k) for k in keys}, "epochs": epochs, "last_epoch": epochs[-1] if epochs else None}


def _build_row(
    cfg: dict, run_cfg: dict | None, *, run_dir: Path, cmp: dict, train_metrics: dict | None, model_path: str | Path,
    adapter_path: str | Path | None, ck_root: Path, candidate: dict, passed: bool, reasons: list[str],
) -> dict:
    h, pset, pvar = cmp["headline"], cmp["primary_set"], cmp["primary_variant"]
    paired = cmp.get("paired") or {}
    general = cmp.get("general_heldout")
    return {
        "event": "run",
        "run_name": cmp["run_name"],
        "timestamp": _now(),
        "base_model": cfg["model"]["base"],
        "model_path": _portable(model_path),
        "adapter_path": _portable(adapter_path) if adapter_path else None,
        "run_dir": _portable(run_dir),
        "checkpoints_root": _portable(ck_root),
        "fingerprint": cmp.get("fingerprint"),
        "fingerprint_match": cmp["fingerprint_match"],
        "primary_set": pset,
        "primary_variant": pvar,
        "base_points": h["base_points"],
        "run_points": h["run_points"],
        "max_points": h["max_points"],
        "delta_points": h["delta_points"],
        "score_pct": cmp["sets"][pset]["variants"][pvar]["run_score_pct"],
        "delta_score_pct": h["delta_score_pct"],
        "parse_fail_rate": h["run_parse_fail_rate"],
        "base_parse_fail_rate": h["base_parse_fail_rate"],
        "delta_lenient_points": h["delta_lenient_points"],
        "secondary_delta_score_pct": {
            s: d["variants"][pvar]["delta_score_pct"] for s, d in cmp["sets"].items() if s != pset and pvar in d["variants"]
        },
        "loglik_delta_acc_pct": {s: d["loglik"]["delta_acc_pct"] for s, d in cmp["sets"].items() if d.get("loglik")},
        "organizer_delta_acc_pct": {s: d["organizer"]["delta_acc_pct"] for s, d in cmp["sets"].items() if d.get("organizer")},
        "organizer_abc_mass": {
            s: d["organizer"]["run_mean_abc_mass"] for s, d in cmp["sets"].items()
            if d.get("organizer") and d["organizer"].get("run_mean_abc_mass") is not None
        },
        "general_nll_rel_change": general["rel_change"] if general else None,
        "ci95": paired.get("bootstrap_ci95_delta_points"),
        "wins": paired.get("wins"),
        "losses": paired.get("losses"),
        "ties": paired.get("ties"),
        "sign_test_p": paired.get("sign_test_p"),
        "gate_passed": passed,
        "gate_failed_rules": failed_rules(reasons),
        "gate_reasons": reasons,
        "promoted": passed,
        "demoted": False,
        "previous_candidate": candidate["run_name"],
        "tags": list(get_dotted(run_cfg or {}, "tags") or []),
        "notes": get_dotted(run_cfg or {}, "notes") or "",
        "config_hash": config_hash(run_cfg) if run_cfg else None,
        "config_path": (run_cfg or {}).get("_config_path"),
        "hparams": _hparams(run_cfg, train_metrics),
        "train": _train_summary(train_metrics),
        "dedup": _dedup_counts(run_cfg or cfg),
    }


def record_run(
    cfg: dict, *, run_dir: Path, cmp: dict, train_metrics: dict | None, model_path: str | Path, adapter_path: str | Path | None,
) -> dict:
    """Gate the compared run against the current candidate, promote on pass, append to runs.jsonl,
    regenerate LEADERBOARD.md and apply pipeline.keep_merged. Returns the appended row.

    Re-recording the candidate itself (same run name) must evaluate the candidate's own checkpoint (ValueError
    otherwise, before anything is written). A re-recorded candidate that fails the gate on a comparable measurement
    (no [fp]/[limit]/[scope] failure) is demoted to the untouched base model (row "demoted": true): under the current
    eval it no longer passes the gate.

    When train_metrics / the run's config are not passed (compare.py-only records), they are read from the run's
    checkpoint dir (train_metrics.json, resolved_config.yaml) if present.
    """
    run_name = cmp["run_name"]
    if run_name == cmp["base_run"]:
        raise ValueError(f"refusing to record {run_name!r}: it is the base run it was compared against")
    rdir, ck_root = cfg_results_dir(cfg), checkpoints_root(cfg)
    dirs = _artifact_dirs(cfg, run_name, model_path, adapter_path)
    run_cfg = _run_config(cfg, run_name, dirs)
    if train_metrics is None:
        train_metrics = next((_read_json(d / "train_metrics.json") for d in dirs if (d / "train_metrics.json").exists()), None)
    required_scope = reference_eval_scope()

    with _locked(rdir):
        candidate = ensure_candidate(cfg)
        conflict = _owner_conflict(candidate, ck_root, run_name, model_path)
        if conflict:
            raise ValueError(conflict)
        passed, reasons = evaluate_gate(cmp, candidate, cfg["gate"], required_scope=required_scope)
        row = _build_row(
            cfg, run_cfg, run_dir=Path(run_dir), cmp=cmp, train_metrics=train_metrics, model_path=model_path,
            adapter_path=adapter_path, ck_root=ck_root, candidate=candidate, passed=passed, reasons=reasons,
        )
        if passed:
            promoted = {
                "run_name": run_name,
                "model_path": row["model_path"],
                "adapter_path": row["adapter_path"],
                "delta_points": row["delta_points"],
                "score_pct": row["score_pct"],
                "promoted_at": row["timestamp"],
                "tags": row["tags"],
                "reason": f"passed gate: {row['delta_points']:+.2f} pts over base "
                          f"(previous candidate {candidate['run_name']} {candidate['delta_points']:+.2f})",
                "fingerprint": row["fingerprint"],
            }
            promoted["warnings"] = _promotion_warnings(ck_root, promoted)
            _install_candidate(rdir, ck_root, promoted)
        elif candidate["run_name"] == run_name and not set(row["gate_failed_rules"]) & NON_VERDICT_RULES:
            reason = (f"candidate {run_name} failed re-evaluation (rules {', '.join(row['gate_failed_rules'])}): "
                      "fell back to the untouched base model")
            _install_candidate(rdir, ck_root, _base_candidate(cfg["model"]["base"], _base_summary(rdir), reason=reason, tags=["base-fallback"]))
            row["demoted"] = True
        _append_row(rdir, row)
        _write_text(rdir / LEADERBOARD_FILE, render_leaderboard(rdir))
        cleanup_checkpoints(rdir, ck_root, get_dotted(cfg, "pipeline.keep_merged", "all"))
    return row


# ----------------------------------------------------------------------------- manual promotion


def promote(cfg_or_results_dir: dict | str | Path, run_name: str, *, reason: str, tags: tuple[str, ...] | list[str] = (), force: bool = False) -> dict:
    """Manually set the candidate (logged as a "manual_promote" row). A run whose latest record failed the gate
    needs force=True. Promoting "base" (the safe fallback) never needs force. Returns the new candidate."""
    if not reason or not reason.strip():
        raise ValueError("a non-empty reason is required for manual promotion")
    if isinstance(cfg_or_results_dir, dict):
        cfg: dict | None = cfg_or_results_dir
        rdir, ck_root = cfg_results_dir(cfg), checkpoints_root(cfg)
    else:
        cfg, rdir = None, Path(cfg_or_results_dir)
        runs = [r for r in load_runs(rdir) if r.get("event") == "run"]
        ck_root = _local_path(runs[-1]["checkpoints_root"]) if runs else None

    with _locked(rdir):
        previous = ensure_candidate(cfg) if cfg else load_candidate(rdir)
        latest = None
        if run_name == BASE_RUN:
            summary = _base_summary(rdir)
            base_model = cfg["model"]["base"] if cfg else (summary or {}).get("model")
            if not base_model:
                raise ValueError(f"cannot determine the base model path for {rdir}; pass a config")
            candidate = _base_candidate(base_model, summary, reason=reason, tags=list(dict.fromkeys(["base-fallback", *tags])))
        else:
            latest = _latest_run_row(rdir, run_name)
            if latest is None:
                raise ValueError(f"run {run_name!r} has no recorded row in {rdir / RUNS_FILE}; run scripts/compare.py --run {run_name} first")
            if not latest["gate_passed"] and not force:
                fails = "\n".join(f"  {r}" for r in latest["gate_reasons"] if r.startswith("FAIL"))
                raise ValueError(f"run {run_name!r} failed the promotion gate:\n{fails}\nuse force to promote anyway (logged)")
            candidate = {
                "run_name": run_name,
                "model_path": latest["model_path"],
                "adapter_path": latest.get("adapter_path"),
                "delta_points": latest["delta_points"],
                "score_pct": latest["score_pct"],
                "promoted_at": _now(),
                "tags": list(tags),
                "reason": reason,
                "fingerprint": latest.get("fingerprint"),
            }
        candidate["manual"] = True
        candidate["forced"] = bool(latest is not None and not latest["gate_passed"])
        candidate["warnings"] = _promotion_warnings(ck_root, candidate)
        _install_candidate(rdir, ck_root, candidate)
        _append_row(rdir, {
            "event": "manual_promote",
            "run_name": run_name,
            "timestamp": candidate["promoted_at"],
            "reason": reason,
            "tags": candidate["tags"],
            "force": force,
            "forced": candidate["forced"],
            "gate_passed": latest["gate_passed"] if latest else None,
            "model_path": candidate["model_path"],
            "previous_candidate": previous["run_name"] if previous else None,
            "warnings": candidate["warnings"],
        })
        _write_text(rdir / LEADERBOARD_FILE, render_leaderboard(rdir))
    return candidate


def _promotion_warnings(ck_root: Path | None, candidate: dict) -> list[str]:
    """A non-base candidate without a loadable merged model (never exported, or cleaned up) cannot be submitted as is."""
    if ck_root is None or candidate["run_name"] == BASE_RUN or _checkpoint_subdir(ck_root, candidate["model_path"]) is None:
        return []
    p = _local_path(candidate["model_path"])
    if (p / "config.json").exists() or (p / "adapter_config.json").exists():
        return []
    adapter = candidate.get("adapter_path") or p / "adapter"
    rebuild = f"python scripts/merge_and_export.py --adapter {_portable(adapter)} --out {_portable(p)}"
    return [f"no merged model at {_portable(p)} (cleaned up or never exported); rebuild it before submitting: {rebuild}"]


# ----------------------------------------------------------------------------- leaderboard


def leaderboard_rows(results_dir: Path) -> list[dict]:
    """Latest record per run plus the base model (delta 0), sorted by headline delta desc."""
    rdir = Path(results_dir)
    candidate = load_candidate(rdir) or {"run_name": BASE_RUN}
    tag_map = load_tags(rdir)
    latest: dict[str, dict] = {}
    for row in load_runs(rdir):
        if row.get("event") == "run" and row["run_name"] != BASE_RUN:
            latest[row["run_name"]] = row
    base = _base_summary(rdir)

    def tags_of(name: str, own: list[str] | None = None) -> list[str]:
        cand_tags = (candidate.get("tags") or []) if candidate["run_name"] == name else []
        return sorted(set(own or []) | set(cand_tags) | {t for t, r in tag_map.items() if r == name})

    entries = [{
        "run_name": BASE_RUN,
        "is_candidate": candidate["run_name"] == BASE_RUN,
        "delta_points": 0.0,
        "score_pct": base["headline"]["score_pct"] if base else None,
        "parse_fail_rate": base["headline"]["parse_fail_rate"] if base else None,
        "gate": "base",
        "tags": tags_of(BASE_RUN),
        "timestamp": base.get("timestamp") if base else None,
    }]
    for name, row in latest.items():
        hp = row.get("hparams") or {}
        entries.append({
            "run_name": name,
            "is_candidate": candidate["run_name"] == name,
            "delta_points": row["delta_points"],
            "score_pct": row["score_pct"],
            "parse_fail_rate": row["parse_fail_rate"],
            "delta_lenient_points": row["delta_lenient_points"],
            "secondary_delta_score_pct": row["secondary_delta_score_pct"],
            "loglik_delta_acc_pct": row["loglik_delta_acc_pct"],
            "organizer_delta_acc_pct": row.get("organizer_delta_acc_pct"),  # absent in rows recorded before rule 9
            "general_nll_rel_change": row["general_nll_rel_change"],
            "ci95": row["ci95"],
            "gate": ("pass" if row["gate_passed"] else "FAIL " + ",".join(row["gate_failed_rules"]))
                    + (" (demoted)" if row.get("demoted") else ""),
            "fingerprint_match": row["fingerprint_match"],
            "tags": tags_of(name, row.get("tags")),
            **{k: hp.get(k) for k in ("lr", "r", "epochs", "general_ratio", "n_train")},
            "timestamp": row["timestamp"],
        })
    return sorted(entries, key=lambda e: (-e["delta_points"], e["run_name"] != BASE_RUN, e["timestamp"] or ""))


def _signed(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x:+.{nd}f}"


def _plain(x: Any, spec: str = "") -> str:
    return "—" if x is None else format(x, spec)


def _per_set(d: dict | None) -> str:
    if not d:
        return "—"
    if len(d) == 1:
        return _signed(next(iter(d.values())))
    return "; ".join(f"{k} {_signed(v)}" for k, v in d.items())


def format_leaderboard_table(entries: list[dict]) -> str:
    cols = ["run", "Δpts", "score%", "parse-fail%", "Δlenient", "secondary Δ%", "loglik Δ%", "org Δ%", "general NLL Δ%",
            "CI95", "gate", "tags", "lr", "r", "epochs", "general_ratio", "n_train", "time"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for e in entries:
        name = ("★ " if e["is_candidate"] else "") + e["run_name"] + ("" if e.get("fingerprint_match", True) else " (fp≠)")
        pf = e["parse_fail_rate"]
        nll = e.get("general_nll_rel_change")
        ci = e.get("ci95")
        ts = e.get("timestamp")
        cells = [
            name,
            _signed(e["delta_points"]),
            _plain(e["score_pct"], ".2f"),
            "—" if pf is None else f"{100 * pf:.1f}",
            _signed(e.get("delta_lenient_points")),
            _per_set(e.get("secondary_delta_score_pct")),
            _per_set(e.get("loglik_delta_acc_pct")),
            _per_set(e.get("organizer_delta_acc_pct")),
            "—" if nll is None else f"{100 * nll:+.2f}",
            "—" if not ci else f"[{ci[0]:+.1f}, {ci[1]:+.1f}]",
            e["gate"],
            ", ".join(e["tags"]) or "—",
            _plain(e.get("lr"), "g"),
            _plain(e.get("r")),
            _plain(e.get("epochs")),
            _plain(e.get("general_ratio")),
            _plain(e.get("n_train")),
            ts[5:16].replace("T", " ") if ts else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_leaderboard(results_dir: Path) -> str:
    rdir = Path(results_dir)
    entries = leaderboard_rows(rdir)
    candidate = load_candidate(rdir)
    primary = next((r for r in reversed(load_runs(rdir)) if r.get("event") == "run"), None)
    scope = f"{primary['primary_set']}/{primary['primary_variant']}" if primary else "the primary eval set/variant"
    cand_line = "none yet (the untouched base model is the fallback)" if candidate is None else (
        f"**{candidate['run_name']}**, Δ {_signed(candidate['delta_points'])} pts, "
        f"score {_plain(candidate.get('score_pct'), '.2f')}% (`{candidate['model_path']}`)"
    )
    header = [
        f"# Leaderboard: {rdir.name}",
        "",
        f"Current candidate (★): {cand_line}.",
        "Regenerated by `train/registry.py` whenever a run is recorded or promoted; do not edit by hand.",
        "",
        f"Latest record per run, sorted by Δpts. Strict grading on {scope}, compared with the untouched base model "
        "evaluated under an identical eval fingerprint (`(fp≠)` marks a run compared despite a mismatch).",
        "",
        "- **Δpts**: headline strict points minus base (gate rule 1). **score%**: the run's strict score.",
        "- **parse-fail%**: share of primary items with no strictly parseable answer. **Δlenient**: lenient-points delta (format-independent).",
        "- **secondary Δ%**: score delta in percentage points on each non-primary eval set (primary variant).",
        "- **loglik Δ%**: MC log-likelihood accuracy delta in percentage points per set.",
        "- **org Δ%**: accuracy delta in percentage points under the organizers' protocol (first-token letter argmax, "
        "no generated text; gate rule 9) per set.",
        "- **general NLL Δ%**: relative change of mean assistant-token NLL on the general Polish held-out set (lower is better).",
        "- **CI95**: paired bootstrap 95% CI of Δpts over items.",
        "- **gate**: `pass`, or `FAIL` with the failed rule numbers (docs/ARCHITECTURE.md#gate; `fp` = eval fingerprint mismatch, "
        "`limit` = debug eval capped by eval.limit, `scope` = eval narrower than configs/base.yaml, `5b` = net score over "
        "all sets, `8` = lenient (format-independent) regression, `9` = organizer first-token letter accuracy); "
        "`(demoted)` = the candidate failed re-evaluation "
        "and was replaced by the base model.",
        "- **lr, r, epochs, general_ratio, n_train**: training hyperparameters. **time**: when the run was recorded (UTC).",
        "",
        "",  # blank line: otherwise Markdown renders the table inside the last list item
    ]
    return "\n".join(header) + format_leaderboard_table(entries) + "\n"


# ----------------------------------------------------------------------------- cleanup


def protected_checkpoints(results_dir: Path, ck_root: Path) -> set[str]:
    """Checkpoint dir names that must never be cleaned or overwritten: the candidate of every base slug under the
    results root + the target of every CANDIDATE* symlink (checkpoint dirs are shared by all base models)."""
    ck_root = Path(ck_root)
    protected: set[str] = set()
    for cand_path in Path(results_dir).parent.glob(f"*/{CANDIDATE_FILE}"):
        name = _checkpoint_subdir(ck_root, (_read_json(cand_path) or {}).get("model_path"))
        if name:
            protected.add(name)
    for link in ck_root.glob(f"{CANDIDATE_LINK}*") if ck_root.is_dir() else []:
        if link.is_symlink():
            protected.add(Path(os.readlink(link)).name)
    return protected


def cleanup_checkpoints(results_dir: Path, checkpoints_root: Path, policy: str) -> list[str]:
    """keep_merged policy. "all": no-op. "candidate_only": delete merged-model files at the root of every recorded,
    non-candidate run checkpoint, keeping adapter/, train_metrics.json, resolved_config.yaml, training_meta.json and
    anything else. Checkpoints without adapter/ are never touched (the merged model would be the only copy).
    Returns deleted file paths; idempotent."""
    if policy == "all":
        return []
    if policy != "candidate_only":
        raise ValueError(f"pipeline.keep_merged must be all|candidate_only, got {policy!r}")
    ck_root = Path(checkpoints_root)
    protected = protected_checkpoints(Path(results_dir), ck_root)
    names = dict.fromkeys(
        name
        for row in load_runs(results_dir) if row.get("event") == "run"
        for name in (_checkpoint_subdir(ck_root, row.get("model_path")), _checkpoint_subdir(ck_root, row.get("adapter_path")))
        if name and name not in protected
    )
    deleted: list[str] = []
    for name in names:
        run_ck = ck_root / name
        if not (run_ck / "adapter").is_dir():
            continue
        for f in sorted({f for pattern in MERGED_PATTERNS for f in run_ck.glob(pattern) if f.is_file()}):
            f.unlink()
            deleted.append(str(f))
    return deleted
