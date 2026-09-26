"""Run-vs-base comparison (compare.json schema: docs/ARCHITECTURE.md#compare).

Aggregate deltas come from the two summary.json files; paired per-item statistics (wins/losses, sign test,
bootstrap CI, flipped items) come from predictions.<primary_set>.<primary_variant>.jsonl matched by item id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from eval.stats import paired_bootstrap_ci, sign_test


def _load_summary(run_dir: Path) -> dict:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"no summary.json in {run_dir} (evaluate it first: scripts/run_eval.py)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dict -> {dotted.key: leaf}. Lists of named dicts (fingerprint "sets") are keyed by name, so a diff
    reads "sets.heldout.sha256" instead of dumping both whole lists."""
    if isinstance(obj, list) and obj and all(isinstance(x, dict) and "name" in x for x in obj):
        obj = {str(x["name"]): {k: v for k, v in x.items() if k != "name"} for x in obj}
    if not isinstance(obj, dict):
        return {prefix: obj}
    out: dict[str, Any] = {}
    for k, v in obj.items():
        out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def fingerprint_diff(base: dict, run: dict) -> dict[str, dict]:
    """Differing fingerprint_parts as {dotted.key: {"base": ..., "run": ...}} (missing on one side -> None)."""
    b = _flatten(base.get("fingerprint_parts") or {})
    r = _flatten(run.get("fingerprint_parts") or {})
    return {k: {"base": b.get(k), "run": r.get(k)} for k in sorted(set(b) | set(r)) if b.get(k) != r.get(k)}


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _pct(points: float, max_points: float) -> float:
    return 100.0 * points / max_points if max_points else 0.0


def _variant_delta(base: dict, run: dict) -> dict:
    b_pct, r_pct = _pct(base["points"], base["max_points"]), _pct(run["points"], run["max_points"])
    return {
        "base_points": _r(base["points"]),
        "run_points": _r(run["points"]),
        "max_points": _r(run["max_points"]),
        "delta_points": _r(run["points"] - base["points"]),
        "base_score_pct": _r(b_pct, 2),
        "run_score_pct": _r(r_pct, 2),
        "delta_score_pct": _r(r_pct - b_pct),
        "base_parse_fail_rate": base["parse_fail_rate"],
        "run_parse_fail_rate": run["parse_fail_rate"],
        "delta_lenient_points": _r(run["points_lenient"] - base["points_lenient"]),
    }


def _loglik_delta(base: dict | None, run: dict | None) -> dict | None:
    if not base or not run or base.get("acc") is None or run.get("acc") is None:
        return None
    return {"base_acc": base["acc"], "run_acc": run["acc"], "delta_acc_pct": _r(100.0 * (run["acc"] - base["acc"]))}


def _organizer_delta(base: dict | None, run: dict | None) -> dict | None:
    """Organizer-protocol deltas (first-token A/B/C argmax, eval/organizers.py) for one set, or None when either side
    has no organizer block (the eval set did not enable it, or the summary predates it)."""
    if not base or not run or base.get("accuracy") is None or run.get("accuracy") is None:
        return None
    b_mass, r_mass = base.get("mean_abc_mass"), run.get("mean_abc_mass")
    b_rot, r_rot = base.get("rotated") or {}, run.get("rotated") or {}
    rotated = None
    if b_rot.get("accuracy") is not None and r_rot.get("accuracy") is not None:
        rotated = _r(100.0 * (r_rot["accuracy"] - b_rot["accuracy"]))
    return {
        "base_accuracy": base["accuracy"],
        "run_accuracy": run["accuracy"],
        "delta_acc_pct": _r(100.0 * (run["accuracy"] - base["accuracy"])),
        "base_mean_abc_mass": b_mass,
        "run_mean_abc_mass": r_mass,
        "delta_mean_abc_mass": None if b_mass is None or r_mass is None else _r(r_mass - b_mass),
        "rotated_delta_acc_pct": rotated,
    }


def _set_deltas(base_sets: dict, run_sets: dict) -> dict:
    out: dict[str, dict] = {}
    for name in [s for s in run_sets if s in base_sets]:
        bv, rv = base_sets[name].get("variants", {}), run_sets[name].get("variants", {})
        entry = {
            "variants": {v: _variant_delta(bv[v], rv[v]) for v in rv if v in bv},
            "loglik": _loglik_delta(base_sets[name].get("loglik"), run_sets[name].get("loglik")),
        }
        organizer = _organizer_delta(base_sets[name].get("organizer"), run_sets[name].get("organizer"))
        if organizer is not None:  # absent (not null) when the set was not scored under the organizer protocol
            entry["organizer"] = organizer
        out[name] = entry
    return out


def _per_subject(base: dict, run: dict) -> dict:
    bs, rs = base.get("per_subject", {}), run.get("per_subject", {})
    out = {}
    for subj in sorted(set(bs) | set(rs)):
        b_pts = bs.get(subj, {}).get("points", 0.0)
        r_pts = rs.get(subj, {}).get("points", 0.0)
        max_pts = (rs.get(subj) or bs.get(subj))["max_points"]
        out[subj] = {"base_points": _r(b_pts), "run_points": _r(r_pts), "max_points": _r(max_pts), "delta_points": _r(r_pts - b_pts)}
    return out


def _load_points(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return {rec["id"]: float(rec["points"]) for rec in map(json.loads, filter(str.strip, fh))}


def _paired(base_dir: Path, run_dir: Path, set_name: str, variant: str) -> tuple[dict | None, dict | None]:
    fname = f"predictions.{set_name}.{variant}.jsonl"
    base_pts, run_pts = _load_points(base_dir / fname), _load_points(run_dir / fname)
    if base_pts is None or run_pts is None:
        return None, None
    ids = sorted(set(base_pts) & set(run_pts))
    b = [base_pts[i] for i in ids]
    r = [run_pts[i] for i in ids]
    gained = [i for i in ids if run_pts[i] > base_pts[i]]
    lost = [i for i in ids if run_pts[i] < base_pts[i]]
    lo, hi = paired_bootstrap_ci(b, r)
    paired = {
        "wins": len(gained),
        "losses": len(lost),
        "ties": len(ids) - len(gained) - len(lost),
        "sign_test_p": sign_test(len(gained), len(lost)),
        "bootstrap_ci95_delta_points": [_r(lo, 2), _r(hi, 2)],
        "n_paired": len(ids),
        "n_unpaired": len(set(base_pts) ^ set(run_pts)),
    }
    return paired, {"gained": gained, "lost": lost}


def _general_delta(base: dict | None, run: dict | None) -> dict | None:
    if not base or not run or base.get("nll") is None or run.get("nll") is None or base["nll"] <= 0:
        return None
    return {"base_nll": base["nll"], "run_nll": run["nll"], "rel_change": _r((run["nll"] - base["nll"]) / base["nll"], 6)}


def _configured_sets(summary: dict) -> list[dict] | None:
    """[{name, variants, loglik}] configured for the eval (from its fingerprint parts); None for old summaries."""
    sets = (summary.get("fingerprint_parts") or {}).get("sets")
    if sets is None:
        return None
    return [{"name": s["name"], "variants": list(s["variants"]), "loglik": bool(s["loglik"])} for s in sets]


def compare_runs(base_dir: Path, run_dir: Path, *, allow_fingerprint_mismatch: bool = False) -> dict:
    """Compare a run's eval to the base model's eval. Refuses a base_dir that does not hold a base-model eval and
    differing eval fingerprints (unless allowed).

    flipped_items lists paired item ids whose strict points went up (gained) / down (lost) vs base.
    """
    base_dir, run_dir = Path(base_dir), Path(run_dir)
    base, run = _load_summary(base_dir), _load_summary(run_dir)
    if base.get("is_base") is not True:
        raise ValueError(
            f"{base_dir} does not hold a base-model eval (model={base.get('model')!r}, is_base={base.get('is_base')!r}); "
            "re-evaluate the base with scripts/run_eval.py --base --model <base id>"
        )
    fp_match = base.get("fingerprint") is not None and base.get("fingerprint") == run.get("fingerprint")
    diff = {} if fp_match else fingerprint_diff(base, run)
    if not fp_match and not allow_fingerprint_mismatch:
        lines = "\n".join(f"  {k}: base={v['base']!r} run={v['run']!r}" for k, v in diff.items()) or "  (no part-level diff recorded)"
        raise ValueError(
            f"eval fingerprint mismatch: base {base_dir.name}={base.get('fingerprint')} vs run {run_dir.name}={run.get('fingerprint')}\n"
            f"{lines}\nRe-evaluate with identical eval settings, or pass allow_fingerprint_mismatch (recorded in compare.json)."
        )

    pset, pvar = run["primary_set"], run["primary_variant"]
    sets = _set_deltas(base.get("sets", {}), run.get("sets", {}))
    if pvar not in sets.get(pset, {}).get("variants", {}):
        raise ValueError(f"primary {pset}/{pvar} is not present in both summaries ({base_dir}, {run_dir})")
    primary = sets[pset]["variants"][pvar]
    paired, flipped = _paired(base_dir, run_dir, pset, pvar)
    limits = [(s.get("fingerprint_parts") or {}).get("limit") for s in (run, base)]
    return {
        "run_name": run.get("run_name", run_dir.name),
        "base_run": base.get("run_name", base_dir.name),
        "fingerprint": run.get("fingerprint"),
        "fingerprint_match": fp_match,
        "fingerprint_diff": diff,
        "eval_limit": next((n for n in limits if n is not None), None),  # debug cap on items per set; None = full eval
        "eval_sets": _configured_sets(run),  # what the eval covered (gate [scope])
        "primary_set": pset,
        "primary_variant": pvar,
        "headline": {
            "base_points": primary["base_points"],
            "run_points": primary["run_points"],
            "max_points": primary["max_points"],
            "delta_points": primary["delta_points"],
            "delta_score_pct": primary["delta_score_pct"],
            "base_parse_fail_rate": primary["base_parse_fail_rate"],
            "run_parse_fail_rate": primary["run_parse_fail_rate"],
            "delta_lenient_points": primary["delta_lenient_points"],
        },
        "sets": sets,
        "per_subject": _per_subject(base["sets"][pset]["variants"][pvar], run["sets"][pset]["variants"][pvar]),
        "paired": paired,
        "general_heldout": _general_delta(base.get("general_heldout"), run.get("general_heldout")),
        "flipped_items": flipped,
    }


def write_compare(run_dir: Path, base_dir: Path, **kw: Any) -> dict:
    """compare_runs(base_dir, run_dir, **kw) written atomically to run_dir/compare.json."""
    cmp = compare_runs(base_dir, run_dir, **kw)
    path = Path(run_dir) / "compare.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cmp, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return cmp


# ----------------------------------------------------------------------------- text table


def _s(x: float | None, nd: int = 2, suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}{suffix}"


def _rate(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _ids(ids: list[str], limit: int = 12) -> str:
    if not ids:
        return "-"
    more = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return ", ".join(ids[:limit]) + more


def format_compare_table(cmp: dict) -> str:
    """Compact human-readable report: headline, variants, secondary sets, loglik, organizer protocol, general NLL,
    subjects, paired stats. Sections whose data is missing are skipped."""
    h, pset, pvar = cmp["headline"], cmp["primary_set"], cmp["primary_variant"]
    p_row = cmp["sets"][pset]["variants"][pvar]
    fp = "match" if cmp["fingerprint_match"] else f"MISMATCH ({', '.join(cmp.get('fingerprint_diff') or {}) or '?'})"
    lines = [
        f"{cmp['run_name']} vs {cmp['base_run']}  [{pset}/{pvar}]  fingerprint: {fp}",
        f"HEADLINE  {h['run_points']:g}/{h['max_points']:g} ({p_row['run_score_pct']:.2f}%)  "
        f"base {h['base_points']:g} ({p_row['base_score_pct']:.2f}%)  "
        f"delta {_s(h['delta_points'])} pts ({_s(h['delta_score_pct'], suffix=' pp')})",
        f"          parse-fail {_rate(h['base_parse_fail_rate'])} -> {_rate(h['run_parse_fail_rate'])}   "
        f"delta lenient {_s(h['delta_lenient_points'])} pts",
        "",
        f"{'set/variant':<34}{'base':>8}{'run':>8}{'max':>7}{'dpts':>8}{'dpp':>8}  parse-fail     dlenient",
    ]
    ordered = [pset] + [s for s in cmp["sets"] if s != pset]
    for set_name in ordered:
        for variant, v in cmp["sets"][set_name]["variants"].items():
            label = f"{set_name}/{variant}" + (" *" if (set_name, variant) == (pset, pvar) else "")
            lines.append(
                f"{label:<34}{v['base_points']:>8g}{v['run_points']:>8g}{v['max_points']:>7g}"
                f"{_s(v['delta_points']):>8}{_s(v['delta_score_pct']):>8}  "
                f"{_rate(v['base_parse_fail_rate']):>5} -> {_rate(v['run_parse_fail_rate']):<6}{_s(v['delta_lenient_points']):>8}"
            )
        if set_name == pset and len(ordered) > 1:
            lines.append("  -- secondary sets --")

    loglik = {s: d["loglik"] for s, d in cmp["sets"].items() if d.get("loglik")}
    if loglik:
        lines += ["", "loglik accuracy (MC, raw prompt)"]
        for s, d in loglik.items():
            lines.append(f"  {s:<32}{100 * d['base_acc']:>7.2f}% -> {100 * d['run_acc']:>6.2f}%  {_s(d['delta_acc_pct'], suffix=' pp')}")

    organizer = {s: d["organizer"] for s, d in cmp["sets"].items() if d.get("organizer")}
    if organizer:
        lines += ["", "organizer (first-token letter argmax)"]
        for s, d in organizer.items():
            mass = ""
            if d.get("base_mean_abc_mass") is not None and d.get("run_mean_abc_mass") is not None:
                mass = f"  ABC mass {d['base_mean_abc_mass']:.3f} -> {d['run_mean_abc_mass']:.3f}"
            rot = "" if d.get("rotated_delta_acc_pct") is None else f"  rotated {_s(d['rotated_delta_acc_pct'], suffix=' pp')}"
            lines.append(
                f"  {s:<32}{100 * d['base_accuracy']:>7.2f}% -> {100 * d['run_accuracy']:>6.2f}%  "
                f"{_s(d['delta_acc_pct'], suffix=' pp')}{mass}{rot}"
            )

    g = cmp.get("general_heldout")
    lines.append("")
    lines.append(
        "general heldout NLL: n/a (missing -> gate fails)" if g is None
        else f"general heldout NLL: {g['base_nll']:.4f} -> {g['run_nll']:.4f}  ({_s(100 * g['rel_change'], suffix='%')})"
    )

    if cmp.get("per_subject"):
        lines += ["", f"per subject ({pset}/{pvar})", f"  {'subject':<16}{'base':>7}{'run':>7}{'max':>7}{'dpts':>8}"]
        for subj, d in cmp["per_subject"].items():
            lines.append(f"  {subj:<16}{d['base_points']:>7g}{d['run_points']:>7g}{d['max_points']:>7g}{_s(d['delta_points']):>8}")

    p, flipped = cmp.get("paired"), cmp.get("flipped_items")
    lines.append("")
    if p is None:
        lines.append("paired: n/a (predictions files missing)")
    else:
        lo, hi = p["bootstrap_ci95_delta_points"]
        lines.append(
            f"paired ({p['n_paired']} items): {p['wins']} wins / {p['losses']} losses / {p['ties']} ties   "
            f"sign-test p={p['sign_test_p']:.3g}   bootstrap CI95 delta pts [{lo:+.2f}, {hi:+.2f}]"
        )
        lines.append(f"  gained: {_ids(flipped['gained'])}")
        lines.append(f"  lost:   {_ids(flipped['lost'])}")
    return "\n".join(lines)
